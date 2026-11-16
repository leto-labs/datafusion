// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::{
    ExternalSorter, get_reserved_bytes_for_record_batch, sort_batch, sort_batch_chunked,
};
use crate::metrics::ExecutionPlanMetricsSet;
use crate::spill::get_record_batch_memory_size;
use crate::spill::spill_manager::GetSlicedSize;
use arrow::array::{
    ArrayRef, Decimal128Array, DictionaryArray, Int8Array, Int64Array, StringArray,
    StringViewArray,
};
use arrow::compute::{SortOptions, concat_batches};
use arrow::datatypes::{DataType, Field, Int8Type, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_common::config::{ExecutionOptions, SpillCompression};
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion_execution::memory_pool::{
    GreedyMemoryPool, MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
};
use datafusion_execution::runtime_env::{RuntimeEnv, RuntimeEnvBuilder};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
use futures::TryStreamExt;
use std::fmt::{Display, Formatter};
use std::mem::size_of;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const MERGE_BYTES: usize = 64 * 1024;
const BUFFERED_BATCHES: usize = 9;

#[derive(Debug, Default)]
struct AllocationState {
    used: usize,
    limit: usize,
    peak: usize,
    denied: usize,
    unchecked_over_limit: usize,
}

/// Existing allocations survive a lower limit, but fresh allocations must fit.
/// The test changes the limit before insertion, independently of spill internals.
#[derive(Debug)]
struct AdjustablePool {
    capacity: usize,
    state: Mutex<AllocationState>,
}

impl AdjustablePool {
    fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            state: Mutex::new(AllocationState {
                limit: capacity,
                ..Default::default()
            }),
        })
    }

    fn set_limit(&self, limit: usize) {
        assert!(limit <= self.capacity);
        self.state.lock().unwrap().limit = limit;
    }
}

impl Display for AdjustablePool {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "AdjustablePool")
    }
}

impl MemoryPool for AdjustablePool {
    fn name(&self) -> &str {
        "AdjustablePool"
    }

    fn grow(&self, _: &MemoryReservation, additional: usize) {
        // Honor the infallible API, but detect attempts to evade try_grow.
        let mut state = self.state.lock().unwrap();
        state.used = state.used.checked_add(additional).unwrap();
        state.peak = state.peak.max(state.used);
        if additional > 0 && state.used > state.limit {
            state.unchecked_over_limit += 1;
        }
    }

    fn shrink(&self, _: &MemoryReservation, shrink: usize) {
        let mut state = self.state.lock().unwrap();
        state.used = state.used.checked_sub(shrink).unwrap();
    }

    fn try_grow(&self, _: &MemoryReservation, additional: usize) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if additional == 0 {
            return Ok(());
        }
        let requested = state.used.checked_add(additional).unwrap();
        if requested > state.limit {
            state.denied += 1;
            return Err(DataFusionError::ResourcesExhausted(
                "allocation limit reached".into(),
            ));
        }
        state.used = requested;
        state.peak = state.peak.max(requested);
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.state.lock().unwrap().used
    }
    fn memory_limit(&self) -> MemoryLimit {
        MemoryLimit::Finite(self.capacity)
    }
}

struct Fixture {
    parent: RecordBatch,
    batches: Vec<RecordBatch>,
    ordering: LexOrdering,
    capacity: usize,
}

fn fixture() -> Result<Fixture> {
    let rows = 1024;
    let count = 30;
    let total = rows * count;
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("sales", DataType::Decimal128(28, 2), true),
        Field::new("payload", DataType::Utf8, false),
    ]));
    // Build each batch separately so it owns its buffers. Zero-copy slices of
    // one parent would share them, and the sort reserves shared buffers once.
    let batches = (0..count)
        .map(|batch| {
            let range = batch * rows..(batch + 1) * rows;
            let categories = StringArray::from_iter(
                range
                    .clone()
                    .map(|i| (i % 11 != 0).then(|| format!("category-{:04}", i % 100))),
            );
            // Every non-null sales value is distinct, so sorting has no ambiguous ties.
            let sales = Decimal128Array::from_iter(
                range
                    .clone()
                    .map(|i| (i != 1).then_some(((total - i) * 100) as i128)),
            )
            .with_precision_and_scale(28, 2)?;
            let payload = StringArray::from_iter_values(
                range.map(|i| format!("row-{i:08}-{}", "x".repeat(64))),
            );
            let columns: Vec<ArrayRef> =
                vec![Arc::new(categories), Arc::new(sales), Arc::new(payload)];
            Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let parent = concat_batches(&schema, &batches)?;
    let ordering = [
        PhysicalSortExpr::new(
            Arc::new(Column::new("category", 0)),
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        ),
        PhysicalSortExpr::new(
            Arc::new(Column::new("sales", 1)),
            SortOptions {
                descending: true,
                nulls_first: true,
            },
        ),
    ]
    .into();
    let buffered = batches[..BUFFERED_BATCHES]
        .iter()
        .map(get_reserved_bytes_for_record_batch)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<usize>();
    let capacity = MERGE_BYTES
        + buffered
        + get_reserved_bytes_for_record_batch(&batches[BUFFERED_BATCHES])? / 2;
    Ok(Fixture {
        parent,
        batches,
        ordering,
        capacity,
    })
}

fn sorter(
    fixture: &Fixture,
    pool: Arc<AdjustablePool>,
) -> Result<(ExternalSorter, Arc<RuntimeEnv>)> {
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(pool)
        .build_arc()?;
    let sorter = ExternalSorter::new(
        0,
        fixture.parent.schema(),
        fixture.ordering.clone(),
        128,
        MERGE_BYTES,
        0, // Force per-batch sorting and merging, rather than concatenation.
        SpillCompression::Uncompressed,
        &ExecutionPlanMetricsSet::new(),
        Arc::clone(&runtime),
    )?;
    Ok((sorter, runtime))
}

async fn assert_released(pool: &AdjustablePool, runtime: &RuntimeEnv) {
    // Buffered readers may finish dropping their state after the owning stream.
    tokio::time::timeout(Duration::from_secs(5), async {
        while pool.reserved() != 0
            || runtime.disk_manager.spilling_progress().active_files_count != 0
            || runtime.disk_manager.used_disk_space() != 0
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("sort reservations and spill files must be released");
    assert_eq!(runtime.disk_manager.used_disk_space(), 0);
    let state = pool.state.lock().unwrap();
    assert!(state.peak <= pool.capacity);
    assert_eq!(state.unchecked_over_limit, 0);
}

async fn check_spill_preserves_merge_workspace_after_limit_decreases() -> Result<()> {
    let fixture = fixture()?;
    let pool = AdjustablePool::new(fixture.capacity);
    let (mut sorter, runtime) = sorter(&fixture, Arc::clone(&pool))?;
    for batch in &fixture.batches[..BUFFERED_BATCHES] {
        sorter.insert_batch(batch.clone()).await?;
    }
    assert!(!sorter.spilled_before());
    let reduced_limit = fixture.capacity * 3 / 4;
    assert!(pool.reserved() > reduced_limit);
    pool.set_limit(reduced_limit);
    // Unlike a hook on free(), this pressure transition still occurs when the
    // implementation preserves its merge reservation instead of releasing it.
    for batch in &fixture.batches[BUFFERED_BATCHES..] {
        sorter.insert_batch(batch.clone()).await?;
        assert!(
            runtime.disk_manager.spilling_progress().active_files_count
                <= fixture.batches.len()
        );
    }
    assert!(pool.state.lock().unwrap().denied > 0);
    assert!(sorter.spilled_before());
    let stream = sorter.sort().await?;
    drop(sorter); // The output stream must retain any transferred workspace.
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    assert!(batches.len() > 1);
    let actual = concat_batches(&fixture.parent.schema(), &batches)?;
    let expected = sort_batch(&fixture.parent, &fixture.ordering, None)?;
    assert_eq!(actual, expected);
    assert_released(&pool, &runtime).await;
    Ok(())
}

#[tokio::test]
async fn test_spill_preserves_merge_workspace_after_limit_decreases() -> Result<()> {
    check_spill_preserves_merge_workspace_after_limit_decreases().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_spill_preserves_merge_workspace_after_limit_decreases_multithreaded()
-> Result<()> {
    check_spill_preserves_merge_workspace_after_limit_decreases().await
}

#[tokio::test]
async fn test_spill_workspace_does_not_hide_insufficient_memory() -> Result<()> {
    let fixture = fixture()?;
    let capacity =
        MERGE_BYTES + get_reserved_bytes_for_record_batch(&fixture.batches[0])? - 1;
    let pool = AdjustablePool::new(capacity);
    let (mut sorter, runtime) = sorter(&fixture, Arc::clone(&pool))?;
    let error = sorter
        .insert_batch(fixture.batches[0].clone())
        .await
        .unwrap_err();
    assert!(matches!(
        error.find_root(),
        DataFusionError::ResourcesExhausted(_)
    ));
    drop(sorter);
    assert_released(&pool, &runtime).await;
    Ok(())
}

#[tokio::test]
async fn test_spill_workspace_cleanup_after_drop_or_error() -> Result<()> {
    for inject_error in [false, true] {
        let fixture = fixture()?;
        let pool = AdjustablePool::new(fixture.capacity);
        let (mut sorter, runtime) = sorter(&fixture, Arc::clone(&pool))?;
        for batch in &fixture.batches[..BUFFERED_BATCHES + 2] {
            sorter.insert_batch(batch.clone()).await?;
        }
        assert!(runtime.disk_manager.spilling_progress().active_files_count > 0);
        if inject_error {
            pool.set_limit(0);
            let error = sorter
                .insert_batch(fixture.batches[BUFFERED_BATCHES + 2].clone())
                .await
                .unwrap_err();
            assert!(matches!(
                error.find_root(),
                DataFusionError::ResourcesExhausted(_)
            ));
        }
        drop(sorter);
        assert_released(&pool, &runtime).await;
    }
    Ok(())
}

#[tokio::test]
async fn test_spill_workspace_cleanup_after_output_is_dropped() -> Result<()> {
    let fixture = fixture()?;
    let pool = AdjustablePool::new(fixture.capacity);
    let (mut sorter, runtime) = sorter(&fixture, Arc::clone(&pool))?;
    for batch in &fixture.batches[..BUFFERED_BATCHES + 2] {
        sorter.insert_batch(batch.clone()).await?;
    }
    assert!(runtime.disk_manager.spilling_progress().active_files_count > 0);
    let mut stream = sorter.sort().await?;
    drop(sorter);
    assert!(stream.try_next().await?.is_some());
    drop(stream);
    assert_released(&pool, &runtime).await;
    Ok(())
}

fn aliased_batches() -> Result<(Vec<RecordBatch>, LexOrdering)> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("alias_of_a", DataType::Int64, false),
    ]));
    let batches = [0, 128]
        .into_iter()
        .map(|start| {
            let values: ArrayRef =
                Arc::new(Int64Array::from_iter_values((start..start + 128).rev()));
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&values), values])
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let ordering = [PhysicalSortExpr::new_default(Arc::new(Column::new("a", 0)))].into();
    Ok((batches, ordering))
}

#[tokio::test]
async fn test_concat_sort_releases_unused_merge_workspace() -> Result<()> {
    let (batches, ordering) = aliased_batches()?;
    let schema = batches[0].schema();
    let input_bytes = batches
        .iter()
        .map(get_reserved_bytes_for_record_batch)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<usize>();
    let concatenated = concat_batches(&schema, &batches)?;
    let concat_bytes = get_reserved_bytes_for_record_batch(&concatenated)?;
    let expected = sort_batch(&concatenated, &ordering, None)?;
    let headroom = 4096;
    let pool_capacity = input_bytes + headroom;
    // Concatenation breaks the aliases and needs more space than its inputs.
    assert!(concat_bytes > input_bytes);
    assert!(concat_bytes <= pool_capacity);

    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(pool_capacity));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let mut sorter = ExternalSorter::new(
        0,
        Arc::clone(&schema),
        ordering,
        64,
        headroom,
        usize::MAX, // Force concatenation instead of a streaming merge.
        SpillCompression::Uncompressed,
        &ExecutionPlanMetricsSet::new(),
        runtime,
    )?;
    for batch in batches {
        sorter.insert_batch(batch).await?;
    }
    assert_eq!(pool.reserved(), pool_capacity);
    assert!(!sorter.spilled_before());
    let stream = sorter.sort().await?;
    assert_eq!(pool.reserved(), concat_bytes);
    drop(sorter);
    let output: Vec<RecordBatch> = stream.try_collect().await?;
    assert_eq!(concat_batches(&schema, &output)?, expected);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn test_disabled_spilling_does_not_reserve_merge_workspace() -> Result<()> {
    let (batches, ordering) = aliased_batches()?;
    let schema = batches[0].schema();
    let expected = sort_batch(&concat_batches(&schema, &batches)?, &ordering, None)?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 * 1024));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .with_disk_manager_builder(
            DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
        )
        .build_arc()?;
    let mut sorter = ExternalSorter::new(
        0,
        Arc::clone(&schema),
        ordering,
        64,
        10 * 1024 * 1024, // Must not be acquired when spilling is disabled.
        0,
        SpillCompression::Uncompressed,
        &ExecutionPlanMetricsSet::new(),
        runtime,
    )?;
    assert_eq!(pool.reserved(), 0);
    for batch in batches {
        sorter.insert_batch(batch).await?;
    }
    assert_eq!(sorter.merge_reservation_size(), 0);
    let stream = sorter.sort().await?;
    drop(sorter);
    let output: Vec<RecordBatch> = stream.try_collect().await?;
    assert_eq!(concat_batches(&schema, &output)?, expected);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

async fn check_chunked_string_view_workspace(during_spill: bool) -> Result<()> {
    let options = ExecutionOptions::default();
    let rows = 4096;
    let batch_size = 1024;
    let batch_count = if during_spill { 3 } else { 2 };
    let schema = Arc::new(Schema::new(vec![Field::new(
        "key",
        DataType::Utf8View,
        false,
    )]));
    let batches = (0..batch_count)
        .map(|batch_id| {
            let values: ArrayRef =
                Arc::new(StringViewArray::from_iter_values((0..rows).rev().map(
                    |i| format!("row-{:08}-{}", batch_id * rows + i, "x".repeat(87)),
                )));
            RecordBatch::try_new(Arc::clone(&schema), vec![values])
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(Column::new(
        "key", 0,
    )))]
    .into();
    let input_bytes = batches[..2]
        .iter()
        .map(get_reserved_bytes_for_record_batch)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<usize>();
    assert!(input_bytes > options.sort_in_place_threshold_bytes);

    // Per-chunk estimates repeat the long-string backing buffers. Unique
    // output capacity fits the input charge even when the pool is full.
    let sorted_bytes = sort_batch_chunked(&batches[0], &ordering, batch_size)?
        .iter()
        .map(get_record_batch_memory_size)
        .sum::<usize>();
    assert!(sorted_bytes > get_reserved_bytes_for_record_batch(&batches[0])?);

    // Leave no unreserved capacity: the sort must be able to use its workspace.
    let capacity = options.sort_spill_reservation_bytes + input_bytes;
    let pool = AdjustablePool::new(capacity);
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
        .build_arc()?;
    let mut sorter = ExternalSorter::new(
        0,
        Arc::clone(&schema),
        ordering.clone(),
        batch_size,
        options.sort_spill_reservation_bytes,
        options.sort_in_place_threshold_bytes,
        options.spill_compression,
        &ExecutionPlanMetricsSet::new(),
        Arc::clone(&runtime),
    )?;
    for batch in &batches[..2] {
        sorter.insert_batch(batch.clone()).await?;
    }
    assert_eq!(pool.reserved(), capacity);
    assert!(!sorter.spilled_before());
    if during_spill {
        sorter.insert_batch(batches[2].clone()).await?;
        assert!(pool.state.lock().unwrap().denied > 0);
        assert!(sorter.spilled_before());
    }

    let stream = sorter.sort().await?;
    drop(sorter);
    let output: Vec<RecordBatch> = stream.try_collect().await?;
    let actual = concat_batches(&schema, &output)?;
    let expected = sort_batch(&concat_batches(&schema, &batches)?, &ordering, None)?;
    assert_eq!(actual, expected);
    assert_eq!(actual.num_rows(), batch_count * rows);
    assert_released(&pool, &runtime).await;
    Ok(())
}

#[tokio::test]
async fn test_chunked_string_view_final_sort_can_use_workspace() -> Result<()> {
    check_chunked_string_view_workspace(false).await
}

#[tokio::test]
async fn test_chunked_string_view_spill_can_use_workspace() -> Result<()> {
    check_chunked_string_view_workspace(true).await
}

/// Unit fixture for sort_run with an explicit shared-input charge. Production
/// view schemas use separate input reservations instead of shared_input.
async fn check_sort_run_explicit_shared_input_workspace(
    consume_output: bool,
) -> Result<()> {
    let batch_size = 1024;
    let values: ArrayRef = Arc::new(StringViewArray::from_iter_values(
        (0..4096)
            .rev()
            .map(|i| format!("row-{i:08}-{}", "x".repeat(87))),
    ));
    let batch = RecordBatch::try_from_iter([("key", values)])?;
    let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(Column::new(
        "key", 0,
    )))]
    .into();
    let input_bytes = get_reserved_bytes_for_record_batch(&batch)?;
    let sorted_batches = sort_batch_chunked(&batch, &ordering, batch_size)?;
    let releases = super::release_sizes_in_order(&sorted_batches);
    let loan_per_run: usize = releases.iter().sum();
    let first_bytes = get_record_batch_memory_size(&sorted_batches[0]);
    // Charge input separately through shared_input. Output then needs a real
    // workspace loan of its unique capacity, not a repeated backing-buffer
    // estimate. Both complete loans fit plus one cursor batch of headroom.
    let workspace = 2 * loan_per_run + first_bytes;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(2 * input_bytes + workspace)))
        .build_arc()?;
    let mut sorter = ExternalSorter::new(
        0,
        batch.schema(),
        ordering.clone(),
        batch_size,
        workspace,
        0,
        SpillCompression::Uncompressed,
        &ExecutionPlanMetricsSet::new(),
        Arc::clone(&runtime),
    )?;
    sorter.insert_batch(batch.clone()).await?;
    sorter.insert_batch(batch.clone()).await?;
    sorter.merge_reservation.free();
    let input = Arc::new(sorter.reservation.take());
    let runs = std::mem::take(&mut sorter.in_mem_batches);
    let permits = Arc::new(tokio::sync::Semaphore::new(2));
    let mut streams = Vec::new();
    for run in runs {
        let sort = sorter.sort_run(
            run,
            sorter.reservation.new_empty(),
            Some(Arc::clone(&input)),
        )?;
        streams.push(
            ExternalSorter::prepare_sort_run(sort, Arc::clone(&permits), None).await?,
        );
    }
    drop(input);
    assert_eq!(permits.available_permits(), 2);
    assert_eq!(sorter.merge_pool.borrow(usize::MAX).size(), first_bytes);
    let stream = crate::sorts::streaming_merge::StreamingMergeBuilder::new()
        .with_streams(streams)
        .with_schema(batch.schema())
        .with_expressions(&ordering)
        .with_metrics(crate::metrics::BaselineMetrics::new(
            &ExecutionPlanMetricsSet::new(),
            0,
        ))
        .with_batch_size(batch_size)
        .with_reservation(sorter.merge_reservation.new_empty())
        .build()?;
    if consume_output {
        let output: Vec<RecordBatch> = stream.try_collect().await?;
        let expected = sort_batch(
            &concat_batches(&batch.schema(), &[batch.clone(), batch.clone()])?,
            &ordering,
            None,
        )?;
        assert_eq!(concat_batches(&batch.schema(), &output)?, expected);
    } else {
        drop(stream);
    }
    drop(sorter);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sort_run_explicit_shared_input_retains_workspace_until_drop() -> Result<()>
{
    check_sort_run_explicit_shared_input_workspace(false).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sort_run_explicit_shared_input_lookahead_returns_cursor_workspace()
-> Result<()> {
    check_sort_run_explicit_shared_input_workspace(true).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sort_preparation_keeps_output_reserved() -> Result<()> {
    let parent = RecordBatch::try_from_iter([(
        "key",
        Arc::new(Int64Array::from_iter_values((0..8 * 4096).rev())) as ArrayRef,
    )])?;
    // All runs share one input buffer, which must be reserved only once.
    let batches = (0..8)
        .map(|run| parent.slice(run * 4096, 4096))
        .collect::<Vec<_>>();
    let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(Column::new(
        "key", 0,
    )))]
    .into();
    let input_bytes = get_reserved_bytes_for_record_batch(&parent)?;
    let sorted_bytes =
        get_record_batch_memory_size(&sort_batch(&batches[0], &ordering, None)?)
            * batches.len();
    // Reserve the merge's cursor memory alongside the sorted batches.
    let merge_workspace = sorted_bytes;
    let capacity = input_bytes + merge_workspace;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(capacity)))
        .build_arc()?;
    let mut sorter = ExternalSorter::new(
        0,
        batches[0].schema(),
        ordering.clone(),
        8192,
        merge_workspace,
        0,
        SpillCompression::Uncompressed,
        &ExecutionPlanMetricsSet::new(),
        Arc::clone(&runtime),
    )?;
    for batch in &batches {
        sorter.insert_batch(batch.clone()).await?;
    }
    sorter.merge_reservation.free();
    let stream = sorter.in_mem_sort_stream(false, false).await?;
    assert_eq!(
        runtime.memory_pool.reserved(),
        sorted_bytes + merge_workspace
    );

    // Another consumer can take the released sort scratch space, while prepared
    // batches and merge workspace remain reserved for this sort.
    let other = MemoryConsumer::new("other").register(&runtime.memory_pool);
    other.try_grow(capacity - runtime.memory_pool.reserved())?;
    let output: Vec<RecordBatch> = stream.try_collect().await?;
    let expected = sort_batch(
        &concat_batches(&batches[0].schema(), &batches)?,
        &ordering,
        None,
    )?;
    assert_eq!(concat_batches(&batches[0].schema(), &output)?, expected);
    drop(sorter);
    drop(other);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn test_single_batch_spill_preserves_workspace_after_limit_decreases() -> Result<()>
{
    let rows = 4096;
    let batch_size = 1024;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "key",
        DataType::Utf8View,
        false,
    )]));
    let batches = [rows, rows - batch_size]
        .into_iter()
        .enumerate()
        .map(|(batch_id, batch_rows)| {
            let values: ArrayRef = Arc::new(StringViewArray::from_iter_values(
                (0..batch_rows).rev().map(|i| {
                    format!("row-{:08}-{}", batch_id * rows + i, "x".repeat(87))
                }),
            ));
            RecordBatch::try_new(Arc::clone(&schema), vec![values])
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(Column::new(
        "key", 0,
    )))]
    .into();
    let input_bytes = batches
        .iter()
        .map(get_reserved_bytes_for_record_batch)
        .collect::<Result<Vec<_>>>()?;
    let sorted_bytes = batches
        .iter()
        .map(|batch| {
            Ok(sort_batch_chunked(batch, &ordering, batch_size)?
                .iter()
                .map(get_record_batch_memory_size)
                .sum::<usize>())
        })
        .collect::<Result<Vec<_>>>()?;
    assert!(sorted_bytes[0] > input_bytes[0]);
    assert!(sorted_bytes[1] > input_bytes[1]);
    let workspace = sorted_bytes[0] - input_bytes[0];
    assert!(sorted_bytes[1] - input_bytes[1] <= workspace);
    let capacity = workspace + input_bytes[0];
    let reduced_limit = workspace + input_bytes[1];
    // Both spills need a workspace loan. The first sorted batch cannot be
    // charged to the execution pool if its workspace is released and regranted.
    assert!(reduced_limit < sorted_bytes[0]);

    let pool = AdjustablePool::new(capacity);
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
        .build_arc()?;
    let mut sorter = ExternalSorter::new(
        0,
        Arc::clone(&schema),
        ordering.clone(),
        batch_size,
        workspace,
        0,
        SpillCompression::Uncompressed,
        &ExecutionPlanMetricsSet::new(),
        Arc::clone(&runtime),
    )?;
    sorter.insert_batch(batches[0].clone()).await?;
    assert_eq!(pool.reserved(), capacity);
    assert_eq!(sorter.in_mem_batches.len(), 1);
    assert!(!sorter.spilled_before());

    pool.set_limit(reduced_limit);
    sorter.insert_batch(batches[1].clone()).await?;
    assert_eq!(sorter.finished_spill_files.len(), 1);
    assert_eq!(sorter.in_mem_batches.len(), 1);
    assert_eq!(pool.reserved(), reduced_limit);

    // Finalizing the sort must spill its one remaining input using the same
    // workspace, then merge both files within the reduced memory limit.
    let spill_count = sorter.metrics.spill_metrics.spill_file_count.clone();
    let stream = sorter.sort().await?;
    drop(sorter);
    let output: Vec<RecordBatch> = stream.try_collect().await?;
    assert_eq!(spill_count.value(), 2);
    assert!(output.iter().all(|batch| batch.num_rows() <= batch_size));
    let actual = concat_batches(&schema, &output)?;
    let expected = sort_batch(&concat_batches(&schema, &batches)?, &ordering, None)?;
    assert_eq!(actual, expected);
    assert_eq!(actual.num_rows(), 2 * rows - batch_size);
    assert_released(&pool, &runtime).await;
    Ok(())
}

#[tokio::test]
async fn test_chunked_dictionary_sort_can_use_workspace_with_defaults() -> Result<()> {
    let options = ExecutionOptions::default();
    let rows = 32_768;
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "payload",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        ),
    ]));
    let values: ArrayRef = Arc::new(StringArray::from_iter_values(
        (0..64).map(|i| format!("{i:02}{}", "x".repeat(8190))),
    ));
    let mut batches = Vec::new();
    for start in [0, rows] {
        let key: ArrayRef = Arc::new(Int64Array::from_iter_values(
            (start..start + rows).rev().map(|i| i as i64),
        ));
        let payload: ArrayRef = Arc::new(DictionaryArray::<Int8Type>::try_new(
            Int8Array::from_iter_values((0..rows).map(|i| (i % 64) as i8)),
            Arc::clone(&values),
        )?);
        batches.push(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![key, payload],
        )?);
    }
    let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(Column::new(
        "key", 0,
    )))]
    .into();
    let input_bytes = batches
        .iter()
        .map(get_reserved_bytes_for_record_batch)
        .collect::<Result<Vec<_>>>()?;
    let buffered_bytes = input_bytes.iter().sum::<usize>();
    assert!(buffered_bytes > options.sort_in_place_threshold_bytes);

    // Dictionary values remain shared, but each output batch must account for
    // its backing buffers. Splitting the input therefore increases its charge.
    let sorted_bytes =
        sort_batch_chunked(&batches[0], &ordering, options.batch_size.get())?
            .iter()
            .map(get_record_batch_memory_size)
            .sum::<usize>();
    assert!(sorted_bytes > input_bytes[0]);

    let capacity = options.sort_spill_reservation_bytes + buffered_bytes;
    let pool = AdjustablePool::new(capacity);
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
        .build_arc()?;
    let mut sorter = ExternalSorter::new(
        0,
        Arc::clone(&schema),
        ordering.clone(),
        options.batch_size.get(),
        options.sort_spill_reservation_bytes,
        options.sort_in_place_threshold_bytes,
        options.spill_compression,
        &ExecutionPlanMetricsSet::new(),
        Arc::clone(&runtime),
    )?;
    for batch in &batches {
        sorter.insert_batch(batch.clone()).await?;
    }
    assert_eq!(pool.reserved(), capacity);
    assert!(!sorter.spilled_before());

    let stream = sorter.sort().await?;
    drop(sorter);
    let output: Vec<RecordBatch> = stream.try_collect().await?;
    let actual = concat_batches(&schema, &output)?;
    let expected = sort_batch(&concat_batches(&schema, &batches)?, &ordering, None)?;
    // RecordBatch equality compares dictionary values, so this checks payloads
    // as well as keys without depending on the dictionary's physical encoding.
    assert_eq!(actual, expected);
    assert_eq!(actual.num_rows(), rows * 2);
    assert_released(&pool, &runtime).await;
    Ok(())
}

#[tokio::test]
async fn test_overlapping_final_outputs_release_unused_workspace() -> Result<()> {
    let options = ExecutionOptions::default();
    let rows = 80_000;
    let capacity = 16 * 1024 * 1024;
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let batches = (0..2)
        .map(|batch_id| {
            let values: ArrayRef = Arc::new(Int64Array::from_iter_values(
                (batch_id * rows..(batch_id + 1) * rows)
                    .rev()
                    .map(|value| value as i64),
            ));
            RecordBatch::try_new(Arc::clone(&schema), vec![values])
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let input_bytes = batches
        .iter()
        .map(get_reserved_bytes_for_record_batch)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<usize>();
    assert!(input_bytes > options.sort_in_place_threshold_bytes);
    assert!(options.sort_spill_reservation_bytes + input_bytes < capacity);
    assert!(capacity < 2 * options.sort_spill_reservation_bytes);
    let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(Column::new(
        "key", 0,
    )))]
    .into();
    let expected = sort_batch(&concat_batches(&schema, &batches)?, &ordering, None)?;
    let pool = AdjustablePool::new(capacity);
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
        .build_arc()?;
    let make_sorter = |partition| {
        ExternalSorter::new(
            partition,
            Arc::clone(&schema),
            ordering.clone(),
            options.batch_size.get(),
            options.sort_spill_reservation_bytes,
            options.sort_in_place_threshold_bytes,
            options.spill_compression,
            &ExecutionPlanMetricsSet::new(),
            Arc::clone(&runtime),
        )
    };

    let mut first_sorter = make_sorter(0)?;
    for batch in &batches {
        first_sorter.insert_batch(batch.clone()).await?;
    }
    assert!(!first_sorter.spilled_before());
    let mut first_stream = first_sorter.sort().await?;
    drop(first_sorter);
    let first_batch = first_stream
        .try_next()
        .await?
        .expect("first sort must produce output");
    assert_eq!(first_batch.num_rows(), options.batch_size.get());

    // Keep the first merge alive and partially consumed. Its remaining data
    // fits comfortably, but an idle spill-workspace floor would block another
    // sorter from acquiring its own workspace and input reservations.
    assert!(
        pool.reserved() + options.sort_spill_reservation_bytes + input_bytes <= capacity,
        "final output must release unused spill workspace for the next sorter"
    );
    let mut second_sorter = make_sorter(1)?;
    for batch in &batches {
        second_sorter.insert_batch(batch.clone()).await?;
    }
    assert!(!second_sorter.spilled_before());
    let second_stream = second_sorter.sort().await?;
    drop(second_sorter);
    let second_output: Vec<RecordBatch> = second_stream.try_collect().await?;

    let mut first_output = vec![first_batch];
    first_output.extend(first_stream.try_collect::<Vec<RecordBatch>>().await?);
    for output in [first_output, second_output] {
        let actual = concat_batches(&schema, &output)?;
        assert_eq!(actual, expected);
        assert_eq!(actual.num_rows(), rows * 2);
    }
    assert_released(&pool, &runtime).await;
    Ok(())
}

async fn check_final_spilled_merge_releases_unused_workspace(
    intermediate: bool,
) -> Result<()> {
    let options = ExecutionOptions::default();
    let workspace = options.sort_spill_reservation_bytes;
    let rows = options.batch_size.get();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(Column::new(
        "key", 0,
    )))]
    .into();
    let make_batch = |rows: usize, start: usize, payload_bytes: usize| {
        let key: ArrayRef = Arc::new(Int64Array::from_iter_values(
            (start..start + rows).rev().map(|value| value as i64),
        ));
        let value = "x".repeat(payload_bytes);
        let payload: ArrayRef = Arc::new(StringArray::from_iter_values(
            (0..rows).map(|_| value.as_str()),
        ));
        RecordBatch::try_new(Arc::clone(&schema), vec![key, payload])
    };
    let batch_count = if intermediate { 16 } else { 8 };
    let big = make_batch(rows, 0, 200)?;
    let tail = make_batch(1, rows * batch_count, 200)?;
    let big_bytes = get_reserved_bytes_for_record_batch(&big)?;
    let tail_bytes = get_reserved_bytes_for_record_batch(&tail)?;
    let capacity = workspace + 4 * big_bytes + tail_bytes / 2;
    assert!(4 * big_bytes > options.sort_in_place_threshold_bytes);
    assert!(tail_bytes > 1);

    // Generate the initial spill files with a fixed capacity and normal pressure.
    let pool = AdjustablePool::new(capacity);
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
        .build_arc()?;
    let make_sorter = |partition| {
        ExternalSorter::new(
            partition,
            Arc::clone(&schema),
            ordering.clone(),
            options.batch_size.get(),
            options.sort_spill_reservation_bytes,
            options.sort_in_place_threshold_bytes,
            options.spill_compression,
            &ExecutionPlanMetricsSet::new(),
            Arc::clone(&runtime),
        )
    };

    let mut first_sorter = make_sorter(0)?;
    let mut original = Vec::new();
    for index in 0..batch_count {
        let batch = make_batch(rows, index * rows, 200)?;
        assert_eq!(get_reserved_bytes_for_record_batch(&batch)?, big_bytes);
        first_sorter.insert_batch(batch.clone()).await?;
        original.push(batch);
    }
    assert_eq!(first_sorter.finished_spill_files.len(), batch_count / 4 - 1);
    // Four batches fit, but this tiny insertion exceeds the remaining room
    // by half its size and triggers another spill through normal pressure.
    first_sorter.insert_batch(tail.clone()).await?;
    original.push(tail.clone());
    assert_eq!(first_sorter.finished_spill_files.len(), batch_count / 4);
    assert_eq!(first_sorter.in_mem_batches.len(), 1);

    let large_spill_bytes = first_sorter.finished_spill_files[0].max_record_batch_memory;
    assert!(
        first_sorter
            .finished_spill_files
            .iter()
            .all(|spill| spill.max_record_batch_memory == large_spill_bytes)
    );
    let tail_spill_bytes = sort_batch(&tail, &ordering, None)?.get_sliced_size()?;
    let single_buffer_bytes = if intermediate {
        4 * large_spill_bytes // Two intermediate files remain for the final merge.
    } else {
        4 * large_spill_bytes + 2 * tail_spill_bytes
    };
    let available_during_merge = workspace + 128 * 1024;
    assert!(tail_bytes < 128 * 1024);
    assert!(single_buffer_bytes < workspace);
    assert!(8 * large_spill_bytes > available_during_merge);
    assert!(capacity > available_during_merge);

    // Leave room for the final input spill, but not two buffers for the first
    // two files. With five files, two intermediate spill passes are also needed.
    let transient =
        MemoryConsumer::new("final merge contender").register(&runtime.memory_pool);
    transient.try_grow(capacity - available_during_merge)?;
    let spill_file_count = first_sorter.metrics.spill_metrics.spill_file_count.clone();
    let mut first_stream = first_sorter.sort().await?;
    let spills_before = spill_file_count.value();
    drop(first_sorter);
    if intermediate {
        // Deny all fresh parent grants: intermediate passes must reuse the
        // workspace acquired before the sort's share of memory decreased.
        pool.set_limit(transient.size());
    }
    let denied_before = pool.state.lock().unwrap().denied;
    let first_batch = first_stream
        .try_next()
        .await?
        .expect("the final spilled merge must produce output");
    assert_eq!(first_batch.num_rows(), rows);
    assert!(
        pool.state.lock().unwrap().denied > denied_before,
        "the initial two-buffer reservation must fail before retrying"
    );
    if intermediate {
        assert!(spill_file_count.value() >= spills_before + 2);
    } else {
        assert_eq!(spill_file_count.value(), spills_before);
    }
    assert_eq!(
        pool.reserved() - transient.size(),
        single_buffer_bytes,
        "the final disk merge must release its unused workspace"
    );
    pool.set_limit(capacity);
    drop(transient);

    // Leave the first final merge alive with most of its output still unread.
    // The next sort's input fits alongside the single-buffer reservation, but
    // not alongside the old workspace floor. Use a default-sized input again.
    let next_batch = make_batch(rows, rows * 16, 400)?;
    let next_bytes = get_reserved_bytes_for_record_batch(&next_batch)?;
    assert!(single_buffer_bytes + workspace + next_bytes <= capacity);
    assert!(2 * workspace + next_bytes > capacity);
    let mut second_sorter = make_sorter(1)?;
    second_sorter.insert_batch(next_batch.clone()).await?;
    let second_stream = second_sorter.sort().await?;
    drop(second_sorter);
    let second_output: Vec<RecordBatch> = second_stream.try_collect().await?;
    assert_eq!(
        concat_batches(&schema, &second_output)?,
        sort_batch(&next_batch, &ordering, None)?
    );

    let mut first_output = vec![first_batch];
    while let Some(batch) = first_stream.try_next().await? {
        first_output.push(batch);
    }
    assert_eq!(
        concat_batches(&schema, &first_output)?,
        sort_batch(&concat_batches(&schema, &original)?, &ordering, None)?
    );
    // EOF must release resources even while the exhausted outer stream lives.
    assert_released(&pool, &runtime).await;
    drop(first_stream);
    Ok(())
}

#[tokio::test]
async fn test_final_spilled_merge_releases_unused_workspace() -> Result<()> {
    check_final_spilled_merge_releases_unused_workspace(false).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_final_spilled_merge_releases_unused_workspace_multithreaded() -> Result<()>
{
    check_final_spilled_merge_releases_unused_workspace(false).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_final_spilled_merge_retains_workspace_between_passes() -> Result<()> {
    check_final_spilled_merge_releases_unused_workspace(true).await
}

#[tokio::test]
async fn test_sort_run_explicit_shared_input_returns_live_workspace_loan_on_drop()
-> Result<()> {
    let options = ExecutionOptions::default();
    let rows = 4096;
    let batch_size = 1024;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "key",
        DataType::Utf8View,
        false,
    )]));
    let values: ArrayRef = Arc::new(StringViewArray::from_iter_values(
        (0..rows)
            .rev()
            .map(|i| format!("row-{i:08}-{}", "x".repeat(87))),
    ));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![values])?;
    let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(Column::new(
        "key", 0,
    )))]
    .into();
    let input_bytes = get_reserved_bytes_for_record_batch(&batch)?;
    let expected = sort_batch_chunked(&batch, &ordering, batch_size)?;
    let releases = super::release_sizes_in_order(&expected);
    let sorted_bytes: usize = releases.iter().sum();
    let first_bytes = releases[0];
    // Input ownership is charged separately below. All unique output capacity
    // is a workspace loan; only this chunk's unshared views retire on poll.
    let initial_loan = sorted_bytes;
    assert!(initial_loan <= options.sort_spill_reservation_bytes);
    assert!(
        initial_loan > first_bytes,
        "cancellation must leave a positive workspace loan"
    );
    let remaining_loan = initial_loan - first_bytes;

    let capacity = options.sort_spill_reservation_bytes + input_bytes;
    let pool = AdjustablePool::new(capacity);
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
        .build_arc()?;
    let mut sorter = ExternalSorter::new(
        0,
        Arc::clone(&schema),
        ordering,
        batch_size,
        options.sort_spill_reservation_bytes,
        options.sort_in_place_threshold_bytes,
        options.spill_compression,
        &ExecutionPlanMetricsSet::new(),
        Arc::clone(&runtime),
    )?;
    sorter.insert_batch(batch.clone()).await?;
    assert_eq!(pool.reserved(), capacity);

    // Exercise sort_run through its lazy stream with an explicit shared-input
    // charge; the production one-batch path instead passes shared_input=None.
    sorter.merge_reservation.free();
    let merge_pool = Arc::clone(&sorter.merge_pool);
    assert_eq!(sorter.in_mem_batches.len(), 1);
    let input = Arc::new(sorter.reservation.take());
    sorter.in_mem_batches.clear();
    let mut stream =
        sorter.sort_batch_stream(batch, sorter.reservation.new_empty(), Some(input))?;
    drop(sorter);

    let first_batch = stream
        .try_next()
        .await?
        .expect("the sorted stream must produce a batch");
    assert_eq!(first_batch, expected[0]);
    assert_eq!(first_batch.num_rows(), batch_size);
    assert_eq!(pool.reserved(), options.sort_spill_reservation_bytes);

    // Check actual loan accounting rather than inferring a loan from the input
    // type. Only the workspace not held by the remaining output is available.
    let unused_workspace = merge_pool.borrow(usize::MAX);
    assert_eq!(
        unused_workspace.size(),
        options.sort_spill_reservation_bytes - remaining_loan
    );
    drop(unused_workspace);
    drop(merge_pool);
    assert_eq!(pool.reserved(), options.sort_spill_reservation_bytes);

    // The stream is the remaining owner of the positive loan. Drop it without
    // consuming the remaining batches and require both reservations to retire.
    drop(stream);
    assert_released(&pool, &runtime).await;
    Ok(())
}

#[tokio::test]
async fn test_empty_sorter_immediately_spills_under_fair_share_pressure() -> Result<()> {
    use datafusion_execution::memory_pool::FairSpillPool;
    let batch = RecordBatch::try_from_iter([(
        "key",
        Arc::new(Int64Array::from_iter_values((0..4096).rev())) as ArrayRef,
    )])?;
    let bytes = get_reserved_bytes_for_record_batch(&batch)?;
    let pool: Arc<dyn MemoryPool> = Arc::new(FairSpillPool::new(MERGE_BYTES + 3 * bytes));
    let other = MemoryConsumer::new("other")
        .with_can_spill(true)
        .register(&pool);
    other.try_grow(bytes)?;
    let idle: Vec<_> = (0..3)
        .map(|i| {
            MemoryConsumer::new(format!("idle{i}"))
                .with_can_spill(true)
                .register(&pool)
        })
        .collect();
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(Column::new(
        "key", 0,
    )))]
    .into();
    let mut sorter = ExternalSorter::new(
        0,
        batch.schema(),
        ordering.clone(),
        1024,
        MERGE_BYTES,
        0,
        SpillCompression::Uncompressed,
        &ExecutionPlanMetricsSet::new(),
        Arc::clone(&runtime),
    )?;
    for _ in 0..2 {
        sorter.insert_batch(batch.clone()).await?;
        assert!(sorter.in_mem_batches.is_empty());
        assert_eq!(sorter.reservation.size(), 0);
    }
    assert_eq!(sorter.finished_spill_files.len(), 2);
    drop(idle);
    drop(other);
    let output: Vec<RecordBatch> = sorter.sort().await?.try_collect().await?;
    assert_eq!(
        concat_batches(&batch.schema(), &output)?,
        sort_batch(
            &concat_batches(&batch.schema(), &[batch.clone(), batch])?,
            &ordering,
            None
        )?
    );
    drop(sorter);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(runtime.disk_manager.used_disk_space(), 0);
    Ok(())
}

#[tokio::test]
async fn test_immediate_sort_spill_respects_aggregate_cap_and_disabled_disk() -> Result<()>
{
    use datafusion_execution::memory_pool::FairSpillPool;
    let batch = RecordBatch::try_from_iter([(
        "key",
        Arc::new(Int64Array::from_iter_values((0..4096).rev())) as ArrayRef,
    )])?;
    let bytes = get_reserved_bytes_for_record_batch(&batch)?;
    for enabled in [true, false] {
        let capacity = if enabled {
            MERGE_BYTES + bytes - 1
        } else {
            3 * bytes
        };
        let pool: Arc<dyn MemoryPool> = Arc::new(FairSpillPool::new(capacity));
        let idle: Vec<_> = (0..3)
            .map(|i| {
                MemoryConsumer::new(format!("idle{i}"))
                    .with_can_spill(true)
                    .register(&pool)
            })
            .collect();
        let mut runtime = RuntimeEnvBuilder::new().with_memory_pool(Arc::clone(&pool));
        if !enabled {
            runtime = runtime.with_disk_manager_builder(
                DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
            );
        }
        let runtime = runtime.build_arc()?;
        let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(
            Column::new("key", 0),
        ))]
        .into();
        let mut sorter = ExternalSorter::new(
            0,
            batch.schema(),
            ordering,
            1024,
            MERGE_BYTES,
            0,
            SpillCompression::Uncompressed,
            &ExecutionPlanMetricsSet::new(),
            Arc::clone(&runtime),
        )?;
        let error = sorter.insert_batch(batch.clone()).await.unwrap_err();
        assert!(matches!(
            error.find_root(),
            DataFusionError::ResourcesExhausted(_)
        ));
        assert_eq!(pool.reserved(), if enabled { MERGE_BYTES } else { 0 });
        assert!(sorter.finished_spill_files.is_empty());
        drop(sorter);
        drop(idle);
        assert_eq!(pool.reserved(), 0);
        assert_eq!(runtime.disk_manager.used_disk_space(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn memory_accounting_sort_output_charges_shared_view_buffer_once() -> Result<()> {
    use crate::ExecutionPlan;
    use crate::common::collect;
    use crate::sorts::sort::SortExec;
    use crate::test::TestMemoryExec;
    use datafusion_execution::TaskContext;
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};

    // 256 rows of 4 KiB strings in one 1 MiB data buffer.
    let rows: Vec<String> = (0..256)
        .rev()
        .map(|i| format!("{i:04}{}", "x".repeat(4092)))
        .collect();
    let values = StringViewArray::from_iter_values(rows.iter());
    let batch = RecordBatch::try_from_iter([("s", Arc::new(values) as ArrayRef)])?;
    let retained = get_record_batch_memory_size(&batch);
    let schema = batch.schema();
    let input = TestMemoryExec::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)?;
    let ordering: LexOrdering =
        [PhysicalSortExpr::new_default(col("s", &schema)?)].into();
    let sort = SortExec::new(ordering, input);

    // 16 output chunks of 16 rows each, every one retaining the data buffer.
    // The input is reserved twice (batch and sliced size) until it is sorted.
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4 * retained));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .with_disk_manager_builder(
            DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
        )
        .build_arc()?;
    let context = Arc::new(
        TaskContext::default()
            .with_session_config(SessionConfig::new().with_batch_size(16))
            .with_runtime(runtime),
    );
    let output = collect(sort.execute(0, context)?).await?;
    assert_eq!(output.len(), 16);
    assert_eq!(output.iter().map(|b| b.num_rows()).sum::<usize>(), 256);
    assert!(
        output
            .iter()
            .map(get_record_batch_memory_size)
            .sum::<usize>()
            > 8 * retained
    );
    drop(output);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn test_sort_release_schedule_covers_every_retained_suffix() -> Result<()> {
    let values = Arc::new(StringViewArray::from_iter_values(
        (0..32).map(|i| format!("{i:04}{}", "x".repeat(1024))),
    ));
    let dictionary = Arc::new(DictionaryArray::<Int8Type>::try_new(
        Int8Array::from_iter_values(0..32),
        values,
    )?);
    let batch = RecordBatch::try_from_iter([("dictionary", dictionary as ArrayRef)])?;
    let mut chunks = vec![batch.slice(0, 8), batch.slice(8, 8), batch.slice(16, 16)];
    chunks.push(RecordBatch::try_from_iter([(
        "distinct",
        Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
    )])?);
    let sizes = super::release_sizes_in_order(&chunks);
    for first in 0..chunks.len() {
        let mut counter =
            datafusion_common::utils::memory::RecordBatchMemoryCounter::new();
        let retained: usize = chunks[first..]
            .iter()
            .map(|batch| counter.count_batch(batch))
            .sum();
        assert_eq!(
            sizes[first..].iter().sum::<usize>(),
            retained,
            "every live suffix must remain fully reserved"
        );
    }
    assert_eq!(sizes[0], 0);
    assert_eq!(sizes[1], 0);
    assert!(super::release_sizes_in_order(&[]).is_empty());
    Ok(())
}

async fn assert_sort_tasks_finished() {
    tokio::time::timeout(Duration::from_secs(5), async {
        while tokio::runtime::Handle::current()
            .metrics()
            .num_alive_tasks()
            != 0
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("sort jobs must finish before merging");
}

/// Actual multi-run path: view schemas disable the shared-input counter, so
/// prepare_sort_runs splits a complete input reservation for each run.
async fn check_sort_preparation_real_reservations(consume_output: bool) -> Result<()> {
    let batch_size = 1024;
    let batch = RecordBatch::try_from_iter([(
        "key",
        Arc::new(StringViewArray::from_iter_values(
            (0..4096)
                .rev()
                .map(|i| format!("row-{i:08}-{}", "x".repeat(87))),
        )) as ArrayRef,
    )])?;
    let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(Column::new(
        "key", 0,
    )))]
    .into();
    let input_bytes = get_reserved_bytes_for_record_batch(&batch)?;
    let sorted = sort_batch_chunked(&batch, &ordering, batch_size)?;
    // Sorted chunks keep the original string buffers and rebuild exactly the
    // same number of 16-byte views. Derive the expected bytes independently
    // of the release schedule under test.
    let unique_bytes = get_record_batch_memory_size(&batch);
    assert!(unique_bytes < input_bytes);
    // Match the explicit-input unit fixture's workspace: two unique runs
    // plus a cursor batch. The real preparation path borrows none of it.
    let workspace = 2 * unique_bytes + get_record_batch_memory_size(&sorted[0]);
    let pool: Arc<dyn MemoryPool> =
        Arc::new(GreedyMemoryPool::new(2 * input_bytes + workspace));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let mut sorter = ExternalSorter::new(
        0,
        batch.schema(),
        ordering.clone(),
        batch_size,
        workspace,
        0,
        SpillCompression::Uncompressed,
        &ExecutionPlanMetricsSet::new(),
        runtime,
    )?;
    assert!(sorter.in_mem_batches_counter.is_none());
    sorter.insert_batch(batch.clone()).await?;
    sorter.insert_batch(batch.clone()).await?;
    assert_eq!(pool.reserved(), 2 * input_bytes + workspace);
    sorter.merge_reservation.free();
    let stream = sorter.in_mem_sort_stream(false, false).await?;
    assert!(
        sorter.sort_concurrency.get().is_some(),
        "must use prepare_sort_runs"
    );
    assert!(sorter.in_mem_batches.is_empty());
    assert_eq!(sorter.reservation.size(), 0);
    assert_sort_tasks_finished().await;
    assert_eq!(pool.reserved(), 2 * unique_bytes + workspace);
    let unused = sorter.merge_pool.borrow(usize::MAX);
    assert_eq!(
        unused.size(),
        workspace,
        "real view runs borrow zero workspace"
    );
    drop(unused);
    if consume_output {
        let output: Vec<RecordBatch> = stream.try_collect().await?;
        assert_eq!(
            concat_batches(&batch.schema(), &output)?,
            sort_batch(
                &concat_batches(&batch.schema(), &[batch.clone(), batch.clone()])?,
                &ordering,
                None
            )?
        );
    } else {
        drop(stream);
    }
    assert_eq!(pool.reserved(), workspace);
    drop(sorter);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sort_preparation_real_reservations_release_output_on_drop() -> Result<()> {
    check_sort_preparation_real_reservations(false).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sort_preparation_real_reservations_merge_without_loans() -> Result<()> {
    check_sort_preparation_real_reservations(true).await
}

#[tokio::test]
async fn test_single_batch_spill_real_reservation_has_no_loan_on_drop() -> Result<()> {
    let batch_size = 1024;
    let batch = RecordBatch::try_from_iter([(
        "key",
        Arc::new(StringViewArray::from_iter_values(
            (0..4096)
                .rev()
                .map(|i| format!("row-{i:08}-{}", "x".repeat(87))),
        )) as ArrayRef,
    )])?;
    let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(Column::new(
        "key", 0,
    )))]
    .into();
    let input_bytes = get_reserved_bytes_for_record_batch(&batch)?;
    let expected = sort_batch_chunked(&batch, &ordering, batch_size)?;
    let unique_bytes = get_record_batch_memory_size(&batch);
    // Only the first chunk's newly allocated view words retire on its pull.
    // Its string backing buffers remain held by later chunks.
    let first_release = batch_size * size_of::<u128>();
    assert!(unique_bytes < input_bytes);
    let pool: Arc<dyn MemoryPool> =
        Arc::new(GreedyMemoryPool::new(input_bytes + MERGE_BYTES));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let mut sorter = ExternalSorter::new(
        0,
        batch.schema(),
        ordering,
        batch_size,
        MERGE_BYTES,
        0,
        SpillCompression::Uncompressed,
        &ExecutionPlanMetricsSet::new(),
        runtime,
    )?;
    assert!(sorter.in_mem_batches_counter.is_none());
    sorter.insert_batch(batch).await?;
    sorter.merge_reservation.free();
    let merge_pool = Arc::clone(&sorter.merge_pool);
    let mut stream = sorter.in_mem_sort_stream(false, false).await?;
    assert_eq!(
        pool.reserved(),
        input_bytes + MERGE_BYTES,
        "one-batch sorting is lazy"
    );
    assert_sort_tasks_finished().await;
    drop(sorter);
    let first = stream.try_next().await?.unwrap();
    assert_eq!(first, expected[0]);
    assert_eq!(pool.reserved(), MERGE_BYTES + unique_bytes - first_release);
    let unused = merge_pool.borrow(usize::MAX);
    assert_eq!(
        unused.size(),
        MERGE_BYTES,
        "one-batch path borrows zero workspace"
    );
    drop(unused);
    drop(merge_pool);
    drop(stream);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}
