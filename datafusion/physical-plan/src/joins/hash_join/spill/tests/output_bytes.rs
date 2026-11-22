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
    data_type: &DataType,
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
            arrow::compute::cast(
                &LargeStringArray::from(vec![value.as_str(), "unmatched"]),
                data_type,
            )?,
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
                Field::new("wide", data_type.clone(), false),
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
    let config = no_reversal(spill_config(1, 32 * 1024, 1));
    let context = task_context(
        config.with_batch_size(MATCHES as usize),
        Some(56 * 1024),
        true,
    );
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
            let wide = arrow::compute::cast(batch.column(3), &DataType::LargeUtf8)?;
            let wide = wide.as_any().downcast_ref::<LargeStringArray>().unwrap();
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
        "{join_type} {data_type} residual={filtered}: {} rows in {batches} batches, max output {max_bytes} bytes",
        rows.len()
    );
    assert!(metric(&plan.metrics().unwrap(), "spill_chunked_runs").unwrap_or(0) > 0);
    drop(stream);
    drop(plan);
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn wide_join_output_bytes() -> Result<()> {
    for data_type in [
        DataType::Utf8,
        DataType::LargeUtf8,
        DataType::Utf8View,
        DataType::Binary,
        DataType::LargeBinary,
        DataType::BinaryView,
    ] {
        check_wide_fanout(JoinType::Inner, false, &data_type).await?;
    }
    for join_type in [JoinType::Left, JoinType::Full] {
        check_wide_fanout(join_type, false, &DataType::LargeUtf8).await?;
    }
    Ok(())
}

#[tokio::test]
async fn wide_residual_filter_output_bytes() -> Result<()> {
    for join_type in [JoinType::Inner, JoinType::Left, JoinType::Full] {
        check_wide_fanout(join_type, true, &DataType::LargeUtf8).await?;
    }
    Ok(())
}

#[tokio::test]
async fn replay_registers_only_when_its_first_file_is_read() -> Result<()> {
    use super::super::files::{FileSource, Origin, SpillFilesExec};
    use super::super::{JoinSpillMetrics, PartWriter, Side, spill_manager};
    use crate::metrics::{ExecutionPlanMetricsSet, Gauge};
    use datafusion_execution::memory_pool::{FairSpillPool, MemoryConsumer, MemoryPool};

    let value = "x".repeat(64 * 1024);
    let rows = 128;
    let batch = RecordBatch::try_from_iter([(
        "payload",
        Arc::new(LargeStringArray::from_iter_values(std::iter::repeat_n(
            value.as_str(),
            rows,
        ))) as _,
    )])?;
    let pool: Arc<dyn MemoryPool> = Arc::new(FairSpillPool::new(4096));
    let context = super::review::pool_context(SessionConfig::new(), Arc::clone(&pool));
    let metrics = JoinSpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0, Gauge::new());
    let manager = spill_manager(&context, &metrics, batch.schema());
    let mut writer =
        PartWriter::new(&manager, "probe permit", None)?.with_byte_limits(true);
    writer.write(std::slice::from_ref(&batch), &metrics, Side::Probe)?;
    let file = writer.finish(&metrics, Side::Probe)?.unwrap();
    let permit = file.replay_bytes;
    assert!(permit > 0 && permit <= BYTE_LIMIT);
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

    for complete in [false, true] {
        let build = MemoryConsumer::new("build")
            .with_can_spill(true)
            .register(&pool);
        let mut stream = leaf.execute_replay(0, &context)?;
        // Registration with zero bytes would cut this allowance in half.
        build.try_grow(4096)?;
        let mut read = 0;
        while let Some(batch) = stream.try_next().await? {
            assert!(
                batch.num_rows() < rows,
                "split before reading the IPC message"
            );
            assert_eq!(pool.reserved(), 4096 + permit);
            read += batch.num_rows();
            if !complete {
                break;
            }
        }
        if complete {
            assert_eq!(read, 2 * rows);
            assert_eq!(pool.reserved(), 4096);
        }
        drop(stream);
        assert_eq!(pool.reserved(), 4096);
    }
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn ordinary_joins_keep_the_stock_batch_boundaries_and_fetch_prefix() -> Result<()> {
    use crate::joins::HashJoinExecBuilder;

    let build = RecordBatch::try_from_iter([
        ("k", Arc::new(Int32Array::from(vec![7, 7, 7, 8])) as _),
        (
            "v",
            Arc::new(LargeStringArray::from(vec![
                "a",
                "b",
                "c",
                &"z".repeat(BYTE_LIMIT + 8),
            ])) as _,
        ),
    ])?;
    let probe = RecordBatch::try_from_iter([(
        "k",
        Arc::new(Int32Array::from(vec![7, 7, 7])) as _,
    )])?;
    for fetch in [None, Some(7)] {
        let mut results = Vec::new();
        for config in [SessionConfig::new(), spill_config(0, 16 * 1024 * 1024, 4)] {
            let plan = HashJoinExecBuilder::new(
                memory_input(std::slice::from_ref(&build), 1),
                memory_input(std::slice::from_ref(&probe), 1),
                vec![(key(&build.schema()), key(&probe.schema()))],
                JoinType::Inner,
            )
            .with_partition_mode(PartitionMode::CollectLeft)
            .with_fetch(fetch)
            .build()?;
            let context =
                task_context(config.with_batch_size(4), Some(64 * 1024 * 1024), true);
            let batches = plan.execute(0, context)?.try_collect::<Vec<_>>().await?;
            assert_eq!(
                super::spill_metrics(&(Arc::new(plan) as Arc<dyn ExecutionPlan>)).0,
                0
            );
            assert_eq!(
                batches
                    .iter()
                    .map(RecordBatch::num_rows)
                    .collect::<Vec<_>>(),
                if fetch.is_some() {
                    vec![4, 3]
                } else {
                    vec![4, 4, 1]
                },
            );
            results.push(batches);
        }
        let serialize = |batches: &[RecordBatch]| -> Result<Vec<u8>> {
            let mut writer = arrow::ipc::writer::StreamWriter::try_new(
                Vec::new(),
                &batches[0].schema(),
            )?;
            for batch in batches {
                writer.write(batch)?;
            }
            Ok(writer.into_inner()?)
        };
        assert_eq!(serialize(&results[0])?, serialize(&results[1])?);
    }
    Ok(())
}

#[tokio::test]
async fn empty_spilled_build_limits_right_and_full_batches_before_null_extension()
-> Result<()> {
    // The forced write moves the build key out of the in-memory table.
    // Its probe has an indivisible last row that must still make progress.
    let build = RecordBatch::try_from_iter([
        ("k", Arc::new(Int32Array::from(vec![7; 16])) as _),
        ("v", Arc::new(Int32Array::from(vec![1; 16])) as _),
    ])?;
    let value = "p".repeat(128 * 1024);
    let oversized = "q".repeat(BYTE_LIMIT + 1);
    let rows = 96;
    let probe = RecordBatch::try_from_iter([
        (
            "k",
            Arc::new(Int32Array::from_iter_values(100..100 + rows as i32)) as _,
        ),
        (
            "v",
            Arc::new(LargeStringArray::from_iter_values(
                std::iter::repeat_n(value.as_str(), rows - 1).chain([oversized.as_str()]),
            )) as _,
        ),
    ])?;
    for join_type in [JoinType::Right, JoinType::Full] {
        let plan = HashJoinExec::try_new(
            memory_input(std::slice::from_ref(&build), 1),
            memory_input(std::slice::from_ref(&probe), 1),
            vec![(key(&build.schema()), key(&probe.schema()))],
            None,
            &join_type,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?;
        let context =
            task_context(no_reversal(spill_config(1, 32 * 1024, 3)), None, true);
        let batches = plan.execute(0, context)?.try_collect::<Vec<_>>().await?;
        assert!(metric(&plan.metrics().unwrap(), "spilled_partitions").unwrap_or(0) > 0);
        let mut probe_rows = 0;
        let mut oversized_batches = 0;
        for batch in batches {
            probe_rows += batch.column(2).len() - batch.column(2).null_count();
            if batch.get_array_memory_size() > BYTE_LIMIT {
                assert_eq!(batch.num_rows(), 1);
                assert_eq!(batch.column(0).null_count(), 1);
                assert_eq!(
                    batch
                        .column(3)
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .unwrap()
                        .value(0),
                    oversized
                );
                oversized_batches += 1;
            }
        }
        assert_eq!(probe_rows, rows);
        assert_eq!(oversized_batches, 1);
    }
    Ok(())
}

#[test]
fn nested_keys_dictionary_payloads_and_fixed_binary_use_the_row_path() -> Result<()> {
    use super::super::{SubJoinTemplate, bytes};
    use arrow::datatypes::Int8Type;

    let child = Arc::new(Field::new("item", DataType::Int8, true));
    for (key_type, payload_type) in [
        (DataType::List(child), DataType::Int32),
        (
            DataType::Int32,
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        ),
        (DataType::Int32, DataType::FixedSizeBinary(4096)),
    ] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", key_type, true),
            Field::new("payload", payload_type, true),
        ]));
        let template = SubJoinTemplate {
            on: vec![(key(&schema), key(&schema))],
            filter: None,
            join_type: JoinType::Full,
            projection: None,
            null_equality: NullEquality::NullEqualsNothing,
            build_schema: Arc::clone(&schema),
            probe_schema: Arc::clone(&schema),
        };
        assert!(!template.byte_limits(), "{schema:?}");
        assert!(!bytes::supported_schema(&schema), "{schema:?}");
    }
    // A nested key can need this child hash vector even for one probe row.
    let nested = arrow::array::ListArray::from_iter_primitive::<Int8Type, _, _>([Some(
        std::iter::repeat_n(Some(1), BYTE_LIMIT / 8 + 1),
    )]);
    assert!(nested.values().len() * size_of::<u64>() > BYTE_LIMIT);
    Ok(())
}

#[tokio::test]
async fn distinct_dictionary_payloads_keep_legacy_coalescing_during_spill() -> Result<()>
{
    use arrow::array::StringDictionaryBuilder;
    use arrow::datatypes::Int8Type;

    let build =
        RecordBatch::try_from_iter([("k", Arc::new(Int32Array::from(vec![7])) as _)])?;
    let payload_bytes = BYTE_LIMIT / 4 + 1024;
    let probe = (0..4_u8)
        .map(|index| {
            let mut values = StringDictionaryBuilder::<Int8Type>::new();
            values.append(char::from(b'a' + index).to_string().repeat(payload_bytes))?;
            RecordBatch::try_from_iter([
                ("k", Arc::new(Int32Array::from(vec![7])) as _),
                ("payload", Arc::new(values.finish()) as _),
            ])
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let plan = HashJoinExec::try_new(
        memory_input(std::slice::from_ref(&build), 1),
        memory_input(&probe, 1),
        vec![(key(&build.schema()), key(&probe[0].schema()))],
        None,
        &JoinType::Inner,
        None,
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
        false,
    )?;
    let context = task_context(no_reversal(spill_config(1, 32 * 1024, 4)), None, true);
    let out = plan.execute(0, context)?.try_collect::<Vec<_>>().await?;
    assert!(metric(&plan.metrics().unwrap(), "probe_spilled_rows").unwrap_or(0) > 0);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].num_rows(), 4);
    assert!(out[0].column(2).get_buffer_memory_size() > BYTE_LIMIT);
    let values = arrow::compute::cast(out[0].column(2), &DataType::Utf8)?;
    let mut first_bytes = values
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap()
        .iter()
        .map(|value| value.unwrap().as_bytes()[0])
        .collect::<Vec<_>>();
    first_bytes.sort_unstable();
    assert_eq!(first_bytes, b"abcd");
    Ok(())
}

#[test]
fn primitive_layouts_size_null_slots_and_replay_buffers() -> Result<()> {
    use super::super::bytes::{RowBytes, replay_need, supported_schema};
    use arrow::array::new_null_array;
    use arrow::datatypes::{IntervalUnit, TimeUnit};

    for dt in [
        DataType::Null,
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float16,
        DataType::Float32,
        DataType::Float64,
        DataType::Date32,
        DataType::Date64,
        DataType::Time32(TimeUnit::Second),
        DataType::Time32(TimeUnit::Millisecond),
        DataType::Time64(TimeUnit::Microsecond),
        DataType::Time64(TimeUnit::Nanosecond),
        DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
        DataType::Duration(TimeUnit::Microsecond),
        DataType::Interval(IntervalUnit::YearMonth),
        DataType::Interval(IntervalUnit::DayTime),
        DataType::Interval(IntervalUnit::MonthDayNano),
        DataType::Decimal32(4, 2),
        DataType::Decimal64(8, 2),
        DataType::Decimal128(20, 2),
        DataType::Decimal256(40, 2),
    ] {
        let rows = 129;
        let column = new_null_array(&dt, rows);
        let batch = RecordBatch::try_from_iter([("payload", Arc::clone(&column))])?;
        assert!(supported_schema(&batch.schema()), "{dt}");
        let width = RowBytes::new(column.as_ref());
        assert!(!width.variable());
        assert!(
            width.slot() * rows >= column.get_buffer_memory_size(),
            "{dt}"
        );
        assert!(
            replay_need(&batch, false)? >= column.get_buffer_memory_size(),
            "{dt}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn fixed_binary_empty_build_keeps_the_legacy_null_extension() -> Result<()> {
    use arrow::array::FixedSizeBinaryArray;

    let build = RecordBatch::try_from_iter([
        ("k", Arc::new(Int32Array::from(vec![7])) as _),
        (
            "fixed",
            Arc::new(FixedSizeBinaryArray::try_from_iter(
                [[0_u8; 4096]].into_iter(),
            )?) as _,
        ),
    ])?;
    let rows = 4096;
    let probe = RecordBatch::try_from_iter([(
        "k",
        Arc::new(Int32Array::from(vec![None; rows])) as _,
    )])?;
    let plan = HashJoinExec::try_new(
        memory_input(std::slice::from_ref(&build), 1),
        memory_input(std::slice::from_ref(&probe), 1),
        vec![(key(&build.schema()), key(&probe.schema()))],
        None,
        &JoinType::Right,
        None,
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
        false,
    )?;
    let config = no_reversal(spill_config(1, 32 * 1024, 4)).with_batch_size(rows);
    let context = task_context(config, None, true);
    let batches = plan.execute(0, context)?.try_collect::<Vec<_>>().await?;
    assert!(metric(&plan.metrics().unwrap(), "spilled_partitions").unwrap_or(0) > 0);
    assert_eq!(batches.len(), 1);
    let nulls = batches[0].column(1);
    assert_eq!(nulls.null_count(), rows);
    assert!(nulls.get_buffer_memory_size() >= rows * 4096);
    Ok(())
}

#[test]
fn null_view_slots_count_bytes_that_arrow_gc_copies() -> Result<()> {
    use super::super::{Gather, OutputPieces, OutputRows};
    use arrow::array::{StringViewArray, UInt32Array, UInt64Array};
    use arrow::buffer::NullBuffer;

    let value = "x".repeat(16 * 1024);
    let view = StringViewArray::from(vec![value.as_str()]);
    // Arrow validates the physical view even when its validity bit is 0.
    let view = StringViewArray::try_new(
        view.views().clone(),
        Arc::clone(view.data_buffers()),
        Some(NullBuffer::new_null(1)),
    )?;
    assert_eq!(view.total_buffer_bytes_used(), value.len());
    let probe = RecordBatch::try_from_iter([("payload", Arc::new(view) as _)])?;
    let gather = Gather {
        build: RecordBatch::new_empty(Arc::new(Schema::empty())),
        probe: probe.clone(),
        left: UInt64Array::from_iter_values(std::iter::repeat_n(0, MATCHES as usize)),
        right: UInt32Array::from_iter_values(std::iter::repeat_n(0, MATCHES as usize)),
        side: JoinSide::Left,
        mark: None,
    };
    let mut pieces = OutputPieces::new(OutputRows::Gather(Box::new(gather)));
    let columns = [ColumnIndex {
        index: 0,
        side: JoinSide::Right,
    }];
    let mut rows = 0;
    while let Some(batch) =
        pieces.next(&probe.schema(), &columns, JoinType::Inner, MATCHES as usize)?
    {
        assert!(batch.get_array_memory_size() <= BYTE_LIMIT);
        assert_eq!(batch.column(0).null_count(), batch.num_rows());
        rows += batch.num_rows();
    }
    assert_eq!(rows, MATCHES as usize);
    Ok(())
}
