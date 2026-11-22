// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Bounded finalization for a directly consuming unordered coalescer.

use arrow::array::{Array, AsArray, BooleanArray, PrimitiveArray};
use arrow::buffer::BooleanBuffer;
use arrow::compute::filter_record_batch;
use arrow::datatypes::{ArrowPrimitiveType, DataType, Int64Type, UInt64Type};
use arrow::record_batch::RecordBatch;
use datafusion_common::Result;
use datafusion_functions_aggregate::count::Count;
use datafusion_physical_expr::expressions::Column;

use super::{AggregateExec, AggregateMode, GroupClusteringMode};

// A small limit bounds both the output and the fraction we expect to retain.
const MAX_FETCH: usize = 32;
const BITMAP_WORDS: usize = 64;
const BUCKETS: usize = BITMAP_WORDS * u64::BITS as usize;

pub(super) struct FinalGroupLimit {
    column: usize,
    data_type: KeyType,
    pub(super) remaining: usize,
}

#[derive(Clone, Copy)]
enum KeyType {
    Int64,
    UInt64,
}

impl FinalGroupLimit {
    pub(super) fn try_new(aggregate: &AggregateExec, fetch: usize) -> Option<Self> {
        if !(1..=MAX_FETCH).contains(&fetch)
            || !matches!(
                aggregate.mode(),
                AggregateMode::Final | AggregateMode::FinalPartitioned
            )
            || aggregate.group_clustering_mode != GroupClusteringMode::None
            || !aggregate.group_by().is_single()
            || aggregate.aggr_expr().is_empty()
            || aggregate.aggr_expr().iter().any(|expr| {
                expr.is_distinct() || expr.fun().inner().downcast_ref::<Count>().is_none()
            })
        {
            return None;
        }

        // Plain column references can be filtered before group key evaluation.
        let schema = aggregate.input().schema();
        let mut eligible = None;
        for (expr, _) in aggregate.group_by().expr() {
            let column = expr.downcast_ref::<Column>()?.index();
            let data_type = match schema.fields().get(column)?.data_type() {
                DataType::Int64 => KeyType::Int64,
                DataType::UInt64 => KeyType::UInt64,
                _ => continue,
            };
            eligible.get_or_insert((column, data_type));
        }
        let (column, data_type) = eligible?;
        Some(Self {
            column,
            data_type,
            remaining: fetch,
        })
    }

    pub(super) fn take_output(&mut self, mut batch: RecordBatch) -> (RecordBatch, bool) {
        if batch.num_rows() > self.remaining {
            batch = batch.slice(0, self.remaining);
        }
        self.remaining -= batch.num_rows();
        (batch, self.remaining == 0)
    }
}

/// A filter on a single plain integer grouping key. Equal grouping tuples
/// always stay in the same bucket. Once the first batch supplies `fetch`
/// distinct buckets, they prove there are at least `fetch` distinct groups.
/// Keep every row in those buckets, including collisions and rows in later
/// batches, so every retained group has complete state even after spill/replay.
pub(super) struct FinalGroupFilter {
    column: usize,
    data_type: KeyType,
    buckets: [u64; BITMAP_WORDS],
}

impl FinalGroupFilter {
    pub(super) fn try_new(limit: &FinalGroupLimit, batch: &RecordBatch) -> Option<Self> {
        let mut filter = Self {
            column: limit.column,
            data_type: limit.data_type,
            buckets: [0; BITMAP_WORDS],
        };
        let values = batch.column(filter.column);
        let ready = match filter.data_type {
            KeyType::Int64 => {
                filter.select(values.as_primitive::<Int64Type>(), limit.remaining)
            }
            KeyType::UInt64 => {
                filter.select(values.as_primitive::<UInt64Type>(), limit.remaining)
            }
        };
        ready.then_some(filter)
    }

    fn select<T>(&mut self, values: &PrimitiveArray<T>, fetch: usize) -> bool
    where
        T: ArrowPrimitiveType,
        T::Native: Into<i128>,
    {
        let mut distinct = 0;
        for value in values.iter().flatten() {
            let (word, bit) = Self::bucket(value);
            if self.buckets[word] & bit == 0 {
                self.buckets[word] |= bit;
                distinct += 1;
                if distinct == fetch {
                    return true;
                }
            }
        }
        false
    }

    fn bucket(value: impl Into<i128>) -> (usize, u64) {
        let bucket = value.into() as usize & (BUCKETS - 1);
        (
            bucket / u64::BITS as usize,
            1 << (bucket % u64::BITS as usize),
        )
    }

    pub(super) fn filter(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let values = batch.column(self.column);
        let mask = match self.data_type {
            KeyType::Int64 => self.mask(values.as_primitive::<Int64Type>()),
            KeyType::UInt64 => self.mask(values.as_primitive::<UInt64Type>()),
        };
        Ok(filter_record_batch(batch, &BooleanArray::new(mask, None))?)
    }

    fn mask<T>(&self, array: &PrimitiveArray<T>) -> BooleanBuffer
    where
        T: ArrowPrimitiveType,
        T::Native: Into<i128>,
    {
        let keep = |value| {
            let (word, bit) = Self::bucket(value);
            self.buckets[word] & bit != 0
        };
        let values = array.values().as_ref();
        match array.nulls() {
            Some(nulls) => BooleanBuffer::collect_bool(values.len(), |index| {
                keep(values[index]) && nulls.is_valid(index)
            }),
            None => {
                BooleanBuffer::collect_bool(values.len(), |index| keep(values[index]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringViewArray};
    use arrow::datatypes::{Field, Schema};
    use datafusion_execution::TaskContext;
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_functions_aggregate::count::count_udaf;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::col;

    use super::*;
    use crate::ExecutionPlan;
    use crate::aggregates::PhysicalGroupBy;
    use crate::coalesce_partitions::CoalescePartitionsExec;
    use crate::collect;
    use crate::empty::EmptyExec;
    use crate::metrics::MetricsSet;
    use crate::test::TestMemoryExec;

    type Row<'a> = (Option<i64>, &'a str, i64);
    type Counts = BTreeMap<(Option<i64>, String), i64>;

    async fn collect_with_fetch(
        partitions: &[&[&[Row<'_>]]],
        fetch: usize,
        context: TaskContext,
    ) -> Result<(Counts, MetricsSet)> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, true),
            Field::new("phrase", DataType::Utf8View, false),
        ]));
        let grouping = PhysicalGroupBy::new_single(vec![
            (col("key", &schema)?, "key".to_owned()),
            (col("phrase", &schema)?, "phrase".to_owned()),
        ]);
        let count = Arc::new(
            AggregateExprBuilder::new(count_udaf(), vec![col("phrase", &schema)?])
                .schema(Arc::clone(&schema))
                .alias("count")
                .build()?,
        );
        let partial = AggregateExec::try_new(
            AggregateMode::Partial,
            grouping.clone(),
            vec![Arc::clone(&count)],
            vec![None],
            Arc::new(EmptyExec::new(Arc::clone(&schema))),
            Arc::clone(&schema),
        )?;
        let partial_schema = partial.schema();
        let batches = partitions
            .iter()
            .map(|partition| {
                partition
                    .iter()
                    .map(|rows| {
                        let keys = Int64Array::from_iter(rows.iter().map(|row| row.0));
                        let phrases = StringViewArray::from_iter_values(
                            rows.iter().map(|row| row.1),
                        );
                        let counts =
                            Int64Array::from_iter_values(rows.iter().map(|row| row.2));
                        RecordBatch::try_new(
                            Arc::clone(&partial_schema),
                            vec![Arc::new(keys), Arc::new(phrases), Arc::new(counts)],
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input = TestMemoryExec::try_new_exec(&batches, partial_schema, None)?;
        let mode = if partitions.len() == 1 {
            AggregateMode::Final
        } else {
            AggregateMode::FinalPartitioned
        };
        let final_ = Arc::new(AggregateExec::try_new(
            mode,
            grouping.as_final(),
            vec![count],
            vec![None],
            input,
            schema,
        )?);
        let plan = Arc::new(
            CoalescePartitionsExec::new(Arc::clone(&final_) as Arc<dyn ExecutionPlan>)
                .with_fetch(Some(fetch)),
        );
        let batches = collect(plan, Arc::new(context)).await?;
        let mut rows = BTreeMap::new();
        for batch in batches {
            let keys = batch.column(0).as_primitive::<Int64Type>();
            let phrases = batch.column(1).as_string_view();
            let counts = batch.column(2).as_primitive::<Int64Type>();
            for (row, key) in keys.iter().enumerate() {
                assert!(
                    rows.insert((key, phrases.value(row).to_owned()), counts.value(row))
                        .is_none()
                );
            }
        }
        Ok((rows, final_.metrics().unwrap()))
    }

    #[tokio::test]
    async fn reads_late_states_for_every_returned_group() -> Result<()> {
        let input: &[&[Row<'_>]] = &[
            &[],
            &[(Some(1), "a", 10), (Some(2), "b", 20), (Some(3), "c", 30)],
            &[
                (Some(1), "a", 1),
                (Some(1), "a2", 2),
                (Some(4097), "a3", 3),
                (Some(4097), "a3", 4),
                (Some(2), "b", 5),
                (Some(3), "c", 6),
                (None, "c", 7),
            ],
        ];
        let (rows, metrics) =
            collect_with_fetch(&[input], 2, TaskContext::default()).await?;
        assert_eq!(rows.len(), 2);
        assert_eq!(metrics.output_rows(), Some(2));
        let complete = BTreeMap::from([
            ((Some(1), "a".to_owned()), 11),
            ((Some(1), "a2".to_owned()), 2),
            ((Some(2), "b".to_owned()), 25),
            ((Some(4097), "a3".to_owned()), 7),
        ]);
        for (group, count) in rows {
            assert_eq!(complete.get(&group), Some(&count));
        }
        Ok(())
    }

    #[tokio::test]
    async fn insufficient_buckets_keep_all_groups_until_output() -> Result<()> {
        let input: &[&[Row<'_>]] = &[
            &[(Some(1), "a", 1), (Some(4097), "b", 2)],
            &[(None, "b", 5), (Some(1), "a", 3), (Some(4097), "b", 4)],
        ];
        let (rows, metrics) =
            collect_with_fetch(&[input], 4, TaskContext::default()).await?;
        assert_eq!(metrics.output_rows(), Some(3));
        assert_eq!(
            rows,
            BTreeMap::from([
                ((Some(1), "a".to_owned()), 4),
                ((Some(4097), "b".to_owned()), 6),
                ((None, "b".to_owned()), 5),
            ]),
        );
        Ok(())
    }

    #[tokio::test]
    async fn partitioned_fetch_returns_complete_counts_after_spill() -> Result<()> {
        const FETCH: usize = 8;
        const BATCH_SIZE: usize = 64;
        let payload = "x".repeat(512);
        let phrases: Vec<_> = (0..4096).map(|i| format!("{i:04}-{payload}")).collect();
        // The integer keys are disjoint across partitions. The first batch
        // selects all eight buckets used by that partition, so its distinct
        // phrases force spills even with the filter active.
        let rows = [0_i64, 1024].map(|base| {
            [3, 7]
                .into_iter()
                .flat_map(|count| {
                    phrases.iter().enumerate().map(move |(i, phrase)| {
                        (Some(base + (i % FETCH) as i64), phrase.as_str(), count)
                    })
                })
                .collect::<Vec<_>>()
        });
        let batches = rows
            .each_ref()
            .map(|rows| rows.chunks(BATCH_SIZE).collect::<Vec<_>>());
        let partitions = batches.each_ref().map(Vec::as_slice);
        let mut expected = BTreeMap::new();
        for &(key, phrase, count) in rows.iter().flatten() {
            *expected.entry((key, phrase)).or_insert(0) += count;
        }
        let context = TaskContext::default()
            .with_session_config(SessionConfig::new().with_batch_size(BATCH_SIZE))
            .with_runtime(
                RuntimeEnvBuilder::new()
                    .with_memory_limit(1024 * 1024, 1.0)
                    .build_arc()?,
            );
        let (output, metrics) = collect_with_fetch(&partitions, FETCH, context).await?;
        assert!(
            metrics.spill_count().unwrap_or_default() > 0,
            "final aggregate must spill"
        );
        assert_eq!(output.len(), FETCH.min(expected.len()));
        for ((key, phrase), count) in output {
            assert_eq!(expected.get(&(key, phrase.as_str())), Some(&count));
        }
        Ok(())
    }

    #[test]
    fn bucket_filter_keeps_late_collisions_and_high_unsigned_keys() {
        let mut filter = FinalGroupFilter {
            column: 0,
            data_type: KeyType::UInt64,
            buckets: [0; BITMAP_WORDS],
        };
        assert!(filter.select(
            &PrimitiveArray::<UInt64Type>::from(vec![Some(u64::MAX), Some(1), Some(2)]),
            2,
        ));
        let later = PrimitiveArray::<UInt64Type>::from(vec![
            Some(4095),
            Some(4097),
            Some(2),
            None,
            Some(u64::MAX),
        ]);
        assert_eq!(
            filter.mask(&later).iter().collect::<Vec<_>>(),
            [true, true, false, false, true],
        );
    }
}
