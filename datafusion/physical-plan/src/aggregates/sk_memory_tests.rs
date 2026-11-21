// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use arrow::array::{ArrayRef, AsArray, Int32Array, Int64Array};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Field, Int32Type, Int64Type, Schema,
};
use arrow::record_batch::RecordBatch;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion_execution::memory_pool::{
    FairSpillPool, GreedyMemoryPool, MemoryConsumer, MemoryPool, MemoryReservation,
    PeakRecordingPool,
};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_expr::{ColumnarValue, Volatility, create_udf};
use datafusion_functions_aggregate::count::count_udaf;
use datafusion_functions_aggregate::sum::sum_udaf;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_expr::expressions::Literal;

use crate::ExecutionPlan;
use crate::common::collect;
use crate::expressions::col;
use crate::test::TestMemoryExec;

use super::group_values::new_group_values;
use super::group_values::{
    GroupValues, GroupValuesPrimitive, packed::migration_needs_memory,
};
use super::order::{GroupClustering, GroupClusteringMode};
use super::{AggregateExec, AggregateMode, PhysicalGroupBy, StreamType};

fn count_groups(keys: Vec<ArrayRef>, data_type: DataType) -> Result<AggregateExec> {
    let schema = Arc::new(Schema::new(vec![Field::new("k", data_type, true)]));
    let batches = keys
        .into_iter()
        .map(|key| RecordBatch::try_new(Arc::clone(&schema), vec![key]))
        .collect::<Result<Vec<_>, _>>()?;
    let input = TestMemoryExec::try_new_exec(&[batches], Arc::clone(&schema), None)?;
    let count = AggregateExprBuilder::new(
        count_udaf(),
        vec![Arc::new(Literal::new(ScalarValue::Int64(Some(1))))],
    )
    .schema(Arc::clone(&schema))
    .alias("count")
    .build()?;
    AggregateExec::try_new(
        AggregateMode::Single,
        PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".to_string())]),
        vec![Arc::new(count)],
        vec![None],
        input,
        schema,
    )
}

fn finite_pool(bytes: usize, spill: bool, migrated: bool) -> Result<Arc<TaskContext>> {
    let pool = Arc::new(GreedyMemoryPool::new(bytes));
    context_with_pool(pool, spill, migrated)
}

fn context_with_pool(
    pool: Arc<dyn MemoryPool>,
    spill: bool,
    migrated: bool,
) -> Result<Arc<TaskContext>> {
    let mut runtime = RuntimeEnvBuilder::new().with_memory_pool(pool);
    if !spill {
        runtime = runtime.with_disk_manager_builder(
            DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
        );
    }
    Ok(Arc::new(
        TaskContext::default()
            .with_runtime(runtime.build_arc()?)
            .with_session_config(
                SessionConfig::new().with_batch_size(8192).set_bool(
                    "datafusion.execution.enable_migration_aggregate",
                    migrated,
                ),
            ),
    ))
}

fn result_counts<T: ArrowPrimitiveType>(
    batches: &[RecordBatch],
) -> BTreeMap<Option<T::Native>, i64>
where
    T::Native: Ord,
{
    let mut counts = BTreeMap::new();
    for batch in batches {
        let keys = batch.column(0).as_primitive::<T>();
        let values = batch.column(1).as_primitive::<Int64Type>();
        for (key, count) in keys.iter().zip(values.values().iter().copied()) {
            assert!(counts.insert(key, count).is_none(), "duplicate output key");
        }
    }
    counts
}

#[tokio::test]
async fn sparse_integer_groups_fit_128_kib_without_spill() -> Result<()> {
    let keys = Int32Array::from_iter_values([0, 1 << 20].into_iter().cycle().take(8192));
    let aggregate = count_groups(vec![Arc::new(keys)], DataType::Int32)?;
    let context = finite_pool(128 * 1024, false, true)?;
    let stream = aggregate.execute_typed(0, &context)?;
    assert!(matches!(stream, StreamType::SingleHash(_)));
    let result = collect(stream.into()).await?;
    assert_eq!(
        result_counts::<Int32Type>(&result),
        BTreeMap::from([(Some(0), 4096), (Some(1 << 20), 4096)]),
    );
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}

fn widening_input() -> Vec<Int64Array> {
    let mut batches = (0..4)
        .map(|batch| {
            Int64Array::from_iter(
                (batch * 8191..(batch + 1) * 8191)
                    .map(Some)
                    .chain(std::iter::once(None)),
            )
        })
        .collect::<Vec<_>>();
    batches.push(Int64Array::from(vec![
        None,
        Some(7),
        Some(i64::MAX),
        Some(8201),
        Some(7),
        None,
    ]));
    batches
}

#[tokio::test]
async fn late_integer_fallback_requires_a_migration_grant() -> Result<()> {
    let keys = widening_input()
        .into_iter()
        .map(|array| Arc::new(array) as ArrayRef)
        .collect();
    let aggregate = count_groups(keys, DataType::Int64)?;
    let context = finite_pool(2 * 1024 * 1024, false, true)?;
    let stream = aggregate.execute_typed(0, &context)?;
    assert!(matches!(stream, StreamType::SingleHash(_)));
    let error = collect(stream.into())
        .await
        .err()
        .expect("migration peak should exceed the available pool");
    assert!(matches!(
        error.find_root(),
        DataFusionError::ResourcesExhausted(_)
    ));
    assert!(
        error.to_string().contains("packed single-key migration"),
        "{error}"
    );
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}

#[rstest::rstest]
#[case::single_hash_greedy(true, 2 * 1024 * 1024, false)]
#[case::legacy_hash_greedy(false, 3 * 1024 * 1024, false)]
#[case::single_hash_fair(true, 2 * 1024 * 1024, true)]
#[case::legacy_hash_fair(false, 3 * 1024 * 1024, true)]
#[tokio::test]
async fn late_integer_fallback_spills_and_merges_repeated_keys_and_null(
    #[case] migrated: bool,
    #[case] bytes: usize,
    #[case] fair: bool,
) -> Result<()> {
    let input = widening_input();
    let mut expected = BTreeMap::new();
    for key in input.iter().flat_map(|array| array.iter()) {
        *expected.entry(key).or_insert(0) += 1;
    }
    let keys = input
        .into_iter()
        .map(|array| Arc::new(array) as ArrayRef)
        .collect();
    let aggregate = count_groups(keys, DataType::Int64)?;
    let pool: Arc<dyn MemoryPool> = if fair {
        Arc::new(FairSpillPool::new(bytes))
    } else {
        Arc::new(GreedyMemoryPool::new(bytes))
    };
    let pool = Arc::new(PeakRecordingPool::new(pool));
    let context =
        context_with_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>, true, migrated)?;
    let stream = aggregate.execute_typed(0, &context)?;
    assert!(match stream {
        StreamType::SingleHash(_) => migrated,
        StreamType::GroupedHash(_) => !migrated,
        _ => false,
    });
    let result = collect(stream.into()).await?;
    assert_eq!(result_counts::<Int64Type>(&result), expected);
    assert!(
        pool.peak_reserved() <= bytes,
        "peak {} exceeds {bytes}",
        pool.peak_reserved()
    );
    assert!(aggregate.metrics().unwrap().spill_count().unwrap() >= 2);
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}

#[test]
fn clustered_primitive_keeps_published_initial_capacity() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    for mode in [
        GroupClusteringMode::Full,
        GroupClusteringMode::Partial(vec![0]),
    ] {
        let grouping = GroupClustering::try_new(&mode)?;
        let values = new_group_values(Arc::clone(&schema), &grouping)?;
        assert_eq!(
            values.size(),
            GroupValuesPrimitive::<Int32Type>::new(DataType::Int32).size()
        );
    }
    Ok(())
}

/// Greedy pool with a competing consumer that takes all bytes available as
/// soon as the owner gives some back.
#[derive(Debug)]
struct PoolWithCompetitor {
    pool: Arc<dyn MemoryPool>,
    competitor: MemoryReservation,
    capacity: usize,
}

impl fmt::Display for PoolWithCompetitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl MemoryPool for PoolWithCompetitor {
    fn name(&self) -> &str {
        "PoolWithCompetitor"
    }

    fn grow(&self, owner: &MemoryReservation, bytes: usize) {
        self.pool.grow(owner, bytes);
    }

    fn shrink(&self, owner: &MemoryReservation, bytes: usize) {
        self.pool.shrink(owner, bytes);
        let available = self.capacity - self.pool.reserved();
        self.competitor.try_grow(available).unwrap();
    }

    fn try_grow(&self, owner: &MemoryReservation, bytes: usize) -> Result<()> {
        self.pool.try_grow(owner, bytes)
    }

    fn reserved(&self) -> usize {
        self.pool.reserved()
    }
}

fn int64_groups() -> Result<Box<dyn GroupValues>> {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
    new_group_values(schema, &GroupClustering::None)
}

#[test]
fn late_fallback_keeps_replacement_charged_when_competitor_takes_released_bytes()
-> Result<()> {
    let capacity = 32 * 1024;
    let backing: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(capacity));
    let pool = Arc::new(PoolWithCompetitor {
        competitor: MemoryConsumer::new("competitor").register(&backing),
        pool: backing,
        capacity,
    });
    let owner = MemoryConsumer::new("group owner")
        .register(&(Arc::clone(&pool) as Arc<dyn MemoryPool>));
    let mut values = int64_groups()?;
    let mut ids = Vec::with_capacity(2);
    values.intern_reserved(&[Arc::new(Int64Array::from(vec![0]))], &mut ids, &owner)?;
    owner.try_resize(values.size())?;
    values.intern_reserved(
        &[Arc::new(Int64Array::from(vec![i64::MAX]))],
        &mut ids,
        &owner,
    )?;
    assert_eq!(ids, vec![1]);
    assert!(
        pool.competitor.size() > 0,
        "the migration must release scratch"
    );
    owner.try_resize(values.size())?;
    assert_eq!(owner.size(), values.size());
    Ok(())
}

#[test]
fn initial_wide_integer_batch_requires_primitive_startup_grant() -> Result<()> {
    let mut values = int64_groups()?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(values.size()));
    let owner = MemoryConsumer::new("group owner").register(&pool);
    owner.try_grow(values.size())?;
    let mut ids = vec![];
    let error = values
        .intern_reserved(
            &[Arc::new(Int64Array::from(vec![i64::MIN, i64::MAX]))],
            &mut ids,
            &owner,
        )
        .expect_err("primitive startup does not fit alongside the packed owner");
    assert!(migration_needs_memory(&error), "{error}");
    assert!(values.is_empty());
    assert!(ids.is_empty());
    assert_eq!(owner.size(), values.size());
    Ok(())
}

#[rstest::rstest]
#[case::single_hash(true, 2 * 1024 * 1024)]
#[case::legacy_hash(false, 3 * 1024 * 1024)]
#[tokio::test]
async fn migration_retry_evaluates_volatile_sum_argument_once(
    #[case] migrated: bool,
    #[case] memory: usize,
) -> Result<()> {
    async fn run(
        memory: usize,
        migrated: bool,
    ) -> Result<(BTreeMap<Option<i64>, i64>, i64, usize)> {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let batches = widening_input()
            .into_iter()
            .map(|keys| RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(keys)]))
            .collect::<Result<Vec<_>, _>>()?;
        let calls = Arc::new(AtomicI64::new(0));
        let udf = Arc::new(create_udf(
            "call_sequence",
            vec![DataType::Int64],
            DataType::Int64,
            Volatility::Volatile,
            Arc::new({
                let calls = Arc::clone(&calls);
                move |_| {
                    let sequence = calls.fetch_add(1, Ordering::Relaxed) + 1;
                    Ok(ColumnarValue::Scalar(ScalarValue::Int64(Some(sequence))))
                }
            }),
        ));
        let value = Arc::new(ScalarFunctionExpr::new(
            "call_sequence",
            udf,
            vec![col("k", &schema)?],
            Arc::new(Field::new("sequence", DataType::Int64, false)),
            Arc::new(ConfigOptions::default()),
        ));
        let sum = AggregateExprBuilder::new(sum_udaf(), vec![value])
            .schema(Arc::clone(&schema))
            .alias("sum(sequence)")
            .build()?;
        let input = TestMemoryExec::try_new_exec(&[batches], Arc::clone(&schema), None)?;
        let aggregate = AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
            vec![Arc::new(sum)],
            vec![None],
            input,
            schema,
        )?;
        let context = finite_pool(memory, true, migrated)?;
        let output = collect(aggregate.execute(0, Arc::clone(&context))?).await?;
        assert_eq!(context.memory_pool().reserved(), 0);
        Ok((
            result_counts::<Int64Type>(&output),
            calls.load(Ordering::Relaxed),
            aggregate.metrics().unwrap().spill_count().unwrap(),
        ))
    }

    let (unpressured, baseline_calls, baseline_spills) =
        run(8 * 1024 * 1024, migrated).await?;
    let (pressured, calls, spills) = run(memory, migrated).await?;
    assert_eq!(baseline_spills, 0);
    assert!(spills >= 2, "expected migration relief and replay");
    assert_eq!(baseline_calls, widening_input().len() as i64);
    assert_eq!(calls, baseline_calls);
    assert_eq!(pressured, unpressured);
    Ok(())
}
