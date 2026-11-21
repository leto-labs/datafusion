// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, Int32Array, LargeStringArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::{JoinSide, JoinType, NullEquality, Result};
use datafusion_execution::config::SessionConfig;
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::expressions::Column;
use futures::TryStreamExt;

use super::{key, memory_input, metric, no_reversal, spill_config, task_context};
use crate::ExecutionPlan;
use crate::joins::utils::{ColumnIndex, JoinFilter};
use crate::joins::{HashJoinExec, PartitionMode};

const MATCHES: i32 = 8192;
const BYTE_LIMIT: usize = 8 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq, Hash)]
struct CheckedFilter;

impl fmt::Display for CheckedFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "checked_filter")
    }
}

impl PhysicalExpr for CheckedFilter {
    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }

    fn data_type(&self, _: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let width = batch.column(1).get_buffer_memory_size();
        assert!(
            width <= BYTE_LIMIT,
            "residual filter gathered {width} bytes"
        );
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        Ok(ColumnarValue::Array(Arc::new(
            ids.values()
                .iter()
                .map(|id| Some(id % 2 == 0))
                .collect::<arrow::array::BooleanArray>(),
        )))
    }

    fn children(&self) -> Vec<&PhysicalExprRef> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<PhysicalExprRef>,
    ) -> Result<PhysicalExprRef> {
        Ok(self)
    }
}

async fn check_wide_fanout(
    join_type: JoinType,
    filtered: bool,
    spill: bool,
) -> Result<()> {
    let build = RecordBatch::try_from_iter([
        (
            "k",
            Arc::new(Int32Array::from(vec![7; MATCHES as usize])) as _,
        ),
        (
            "id",
            Arc::new(Int32Array::from_iter_values(0..MATCHES)) as _,
        ),
    ])?;
    // 128 MiB of copies before nulls, offsets and key columns. Keep the test
    // small enough to reproduce the original failure without exhausting RAM.
    let value = "x".repeat(16 * 1024);
    let probe = RecordBatch::try_from_iter([
        ("k", Arc::new(Int32Array::from(vec![7, 8])) as _),
        (
            "wide",
            Arc::new(LargeStringArray::from(vec![value.as_str(), "unmatched"])) as _,
        ),
    ])?;
    let build = (0..MATCHES as usize)
        .step_by(1024)
        .map(|start| build.slice(start, 1024))
        .collect::<Vec<_>>();
    let filter = filtered.then(|| {
        JoinFilter::new(
            Arc::new(CheckedFilter),
            vec![
                ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 1,
                    side: JoinSide::Right,
                },
            ],
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("wide", DataType::LargeUtf8, false),
            ])),
        )
    });
    let plan = HashJoinExec::try_new(
        memory_input(&build, 1),
        memory_input(std::slice::from_ref(&probe), 1),
        vec![(key(&build[0].schema()), Arc::new(Column::new("k", 0)))],
        filter,
        &join_type,
        filtered.then_some(vec![1]),
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
        false,
    )?;
    let (config, pool) = if spill {
        (no_reversal(spill_config(1, 32 * 1024, 1)), Some(56 * 1024))
    } else {
        (SessionConfig::new(), None)
    };
    let context = task_context(config.with_batch_size(MATCHES as usize), pool, true);
    let mut stream = plan.execute(0, Arc::clone(&context))?;
    let mut rows = Vec::new();
    let mut max_bytes = 0;
    let mut batches = 0;
    while let Some(batch) = stream.try_next().await? {
        let bytes = batch.get_array_memory_size();
        max_bytes = max_bytes.max(bytes);
        batches += 1;
        assert!(bytes <= BYTE_LIMIT, "output gathered {bytes} bytes");
        assert!(batch.num_rows() <= MATCHES as usize);
        let id_column = usize::from(!filtered);
        let ids = batch
            .column(id_column)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        if !filtered {
            let wide = batch
                .column(3)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .unwrap();
            for (id, text) in ids.iter().zip(wide.iter()) {
                assert_eq!(
                    text,
                    Some(if id.is_some() {
                        value.as_str()
                    } else {
                        "unmatched"
                    })
                );
            }
        }
        rows.extend(ids.iter());
    }
    rows.sort_unstable();
    let mut expected = (0..MATCHES)
        .filter(|id| !filtered || id % 2 == 0 || join_type != JoinType::Inner)
        .map(Some)
        .collect::<Vec<_>>();
    if join_type == JoinType::Full {
        expected.insert(0, None);
    }
    assert_eq!(rows, expected);
    eprintln!(
        "{join_type} spill={spill} residual={filtered}: {} rows in {batches} batches, max output {max_bytes} bytes",
        rows.len()
    );
    if spill {
        assert!(metric(&plan.metrics().unwrap(), "spill_chunked_runs").unwrap_or(0) > 0);
    }
    drop(stream);
    drop(plan);
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn wide_join_output_bytes() -> Result<()> {
    for join_type in [JoinType::Inner, JoinType::Left, JoinType::Full] {
        for spill in [false, true] {
            check_wide_fanout(join_type, false, spill).await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn wide_residual_filter_output_bytes() -> Result<()> {
    for join_type in [JoinType::Inner, JoinType::Left, JoinType::Full] {
        for spill in [false, true] {
            check_wide_fanout(join_type, true, spill).await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn probe_replay_charges_one_file_permit_and_releases_on_drop() -> Result<()> {
    use super::super::files::{FileSource, Origin, SpillFilesExec};
    use super::super::{
        JoinSpillMetrics, PartWriter, REPLAY_BATCH_BYTES, Side, spill_manager,
    };
    use crate::metrics::{ExecutionPlanMetricsSet, Gauge};
    use datafusion_execution::memory_pool::MemoryConsumer;

    let rows = 128;
    let value = "x".repeat(64 * 1024);
    let batch = RecordBatch::try_from_iter([(
        "v",
        Arc::new(LargeStringArray::from_iter_values(std::iter::repeat_n(
            value.as_str(),
            rows,
        ))) as _,
    )])?;
    let context = task_context(SessionConfig::new(), Some(4096), true);
    let metrics = JoinSpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0, Gauge::new());
    let manager = spill_manager(&context, &metrics, batch.schema());
    let mut writer = PartWriter::new(&manager, "wide probe replay", None)?;
    writer.write(&[batch.clone()], &metrics, Side::Probe)?;
    let file = writer.finish(&metrics, Side::Probe)?.unwrap();
    assert!(file.replay_bytes <= REPLAY_BATCH_BYTES);
    let permit = file.replay_bytes;
    let leaf = SpillFilesExec::new(
        batch.schema(),
        FileSource::Fixed(vec![vec![file.clone(), file]]),
        manager,
        1,
        Origin {
            rows,
            parent_rows: rows,
        },
        metrics,
    );
    let pool = context.memory_pool();
    let held = MemoryConsumer::new("full pool").register(pool);
    held.try_grow(4096)?;

    for complete in [false, true] {
        let (mut stream, scratch) = leaf.execute_replay(0, &context)?;
        assert_eq!(pool.reserved(), 4096);
        let mut read = 0;
        while let Some(next) = stream.try_next().await? {
            assert!(next.num_rows() < rows, "writer must split before replay");
            assert_eq!(pool.reserved(), 4096 + permit);
            read += next.num_rows();
            if !complete {
                break;
            }
        }
        if complete {
            assert_eq!(read, 2 * rows);
            assert_eq!(pool.reserved(), 4096);
        }
        drop(stream);
        drop(scratch);
        assert_eq!(pool.reserved(), 4096);
    }
    Ok(())
}
