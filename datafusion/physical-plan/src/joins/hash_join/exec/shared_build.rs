// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow patch: parallel construction of one shared CollectLeft hash table.
//!
//! DuckDB builds a single join table per join with every thread
//! (`HashJoinFinalizeEvent`, `src/execution/operator/join/physical_hash_join.cpp`
//! v2.0.0 `:1152-1305`, one compare-and-swap per row in `InsertRowToEntry`,
//! `src/execution/join_hashtable.cpp:771-806`) once the build has at least
//! `PARALLEL_CONSTRUCT_THRESHOLD` = 1,048,576 rows (`physical_hash_join.cpp:933`).
//! Any thread then probes the one table, with no repartition of the probe.
//!
//! Native CollectLeft hashes and inserts every build row in one future. Here,
//! with `longbow.execution.parallel_hash_build_rows` above zero and a build of
//! at least that many rows:
//!
//! 1. contiguous batch groups evaluate their keys and hashes in parallel and
//!    scatter matchable rows by hash shard;
//! 2. one task per shard fills that shard's `HashTable`, writing only its own
//!    rows of one chain array that all shards share, while other tasks
//!    concatenate the payload column by column;
//! 3. the probe looks up `shards[hash >> 40 & mask]` and traverses the shared
//!    chain exactly as [`JoinHashMapU32`](crate::joins::join_hash_map::JoinHashMapU32).
//!
//! Every shard inserts its rows in descending row order, so each chain lists
//! equal-hash rows in ascending row order for any shard count. The result,
//! visited-row bitmap and dynamic-filter inputs index the same concatenated
//! batch as the native path. Builds below the threshold, at or above
//! `u32::MAX` rows, or prepared builds keep the native path; a build large
//! enough for this path takes it before the serial perfect-hash `ArrayMap`.
//!
//! The rest of the 56 build path runs unchanged on the result
//! (`build_in_memory`): float keys are normalized once, the key-range bitmap,
//! IN-list or routed map membership and the null-aware build read the one
//! sharded map and the consolidated batch, whose rows keep the input order.

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, GenericByteViewArray, NullBufferBuilder,
    RecordBatchOptions,
};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow::compute::concat;
use arrow::datatypes::ByteViewType;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow_data::ByteView;
use datafusion_common::config::ConfigOptions;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::utils::memory::{
    estimate_memory_size, get_record_batch_memory_size,
};
use datafusion_common::{DataFusionError, NullEquality, Result, internal_datafusion_err};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays;
use futures::FutureExt;
use hashbrown::HashTable;
use hashbrown::hash_table::Entry::{Occupied, Vacant};

use crate::attempt_tasks::{AttemptTasks, ScopedTask, spawn_producer, stopped_error};
use crate::joins::chain::traverse_chain;
use crate::joins::join_hash_map::JoinHashMapType;
use crate::joins::join_hash_map::ProbeOffset::{self, AtRow, MidChain};
use crate::joins::utils::{BuildProbeJoinMetrics, matchable_join_keys};
use crate::metrics::{ExecutionPlanMetricsSet, MetricBuilder};

/// Session option: the build row count from which a CollectLeft table is
/// built in parallel; 0 keeps the native single-future build.
pub(crate) const OPTION: &str = "longbow.execution.parallel_hash_build_rows";

/// Each shard gets at least this many build rows.
const MIN_SHARD_ROWS: usize = 65_536;

/// Shard bits stay below hashbrown's 7 tag bits (57..64) and above its
/// bucket bits for tables of fewer than 2^40 buckets.
const SHARD_SHIFT: u32 = 40;
const MAX_SHARDS: usize = 256;

/// Metric: shards of a parallel build; absent when the native path ran.
const SHARDS_METRIC: &str = "shared_build_shards";

/// Reads [`OPTION`] once per execution. An absent or unparsable value is 0;
/// the session's option extension validates values when they are set.
pub(crate) fn configured_rows(config: &ConfigOptions) -> usize {
    config
        .entries()
        .into_iter()
        .find(|entry| entry.key == OPTION)
        .and_then(|entry| entry.value)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// Which build `build_in_memory` runs.
pub(super) enum BuildPath {
    /// The native build, unchanged.
    Native,
    /// The native build, then [`compact_build`]: the parallel-build option is
    /// set, but this build stayed on the native path.
    Compacted,
    /// [`ParallelBuild::build_consolidated`] built this map; the batches are
    /// already consolidated.
    Prebuilt(Box<dyn JoinHashMapType>),
}

/// How one CollectLeft execution may build its table in parallel.
pub(super) struct ParallelBuild {
    min_rows: usize,
    threads: usize,
    scope: Option<Arc<AttemptTasks>>,
    metrics: ExecutionPlanMetricsSet,
    partition: usize,
    #[cfg(test)]
    fault: Option<Arc<TestFault>>,
}

/// Test-only fault injection for the build's second task set.
#[cfg(test)]
#[derive(Debug, Default)]
pub(super) struct TestFault {
    /// Every column task panics.
    pub(super) panic_columns: bool,
    /// Each table task waits this long before it fills its shard.
    pub(super) table_delay: std::time::Duration,
    /// Table tasks that finished filling their shard.
    pub(super) tables_done: std::sync::atomic::AtomicUsize,
}

impl ParallelBuild {
    /// `None` unless [`OPTION`] is above zero.
    pub(super) fn from_task(
        context: &TaskContext,
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
    ) -> Option<Self> {
        let min_rows = configured_rows(context.session_config().options());
        (min_rows > 0).then(|| Self {
            min_rows,
            // The session's parallelism, which the engine sizes to the slots
            // it owns; the runtime's worker pool bounds how many shard tasks
            // run at once, so the host's CPU count does not enter here.
            threads: context.session_config().target_partitions().max(1),
            scope: AttemptTasks::from_task(context),
            metrics: metrics.clone(),
            partition,
            #[cfg(test)]
            fault: None,
        })
    }

    #[cfg(test)]
    pub(super) fn for_test(min_rows: usize, threads: usize) -> Self {
        Self {
            min_rows,
            threads,
            scope: None,
            metrics: ExecutionPlanMetricsSet::new(),
            partition: 0,
            fault: None,
        }
    }

    #[cfg(test)]
    pub(super) fn with_fault(mut self, fault: Arc<TestFault>) -> Self {
        self.fault = Some(fault);
        self
    }

    /// A power of two: at most one per thread, at least [`MIN_SHARD_ROWS`]
    /// rows each; 1 means the native build.
    fn shards(&self, rows: usize) -> usize {
        if rows < self.min_rows || rows >= u32::MAX as usize {
            return 1;
        }
        let shards = self.threads.min(rows / MIN_SHARD_ROWS).min(MAX_SHARDS);
        if shards < 2 {
            1
        } else {
            1 << (usize::BITS - 1 - shards.leading_zeros())
        }
    }

    /// Builds the table and consolidated batch of `batches`, or returns
    /// `None` when the native build applies: below the threshold, or when the
    /// pool refuses the hashes and row lists, the tables or the column copies
    /// (the native build needs no transient lists and charges its table and
    /// copy one after another, so it can still fit; nothing stays charged
    /// then). A refused group compaction copy is an error. Charges the shard tables and the
    /// chain array to `reservation`, as well as the hashes and row lists
    /// while they exist. Every copy (compacted views, concatenated columns)
    /// is charged before it is allocated to the returned transient
    /// reservation on the same consumer, which
    /// [`Self::build_consolidated`] moves to `reservation`. A refusal or
    /// any other error stops the remaining tasks and is returned only once
    /// every spawned task has finished, so no task still holds a charge or
    /// a buffer when it surfaces; the caller's reservation then releases the
    /// rest on drop.
    #[expect(clippy::too_many_arguments)]
    pub(super) async fn build(
        &self,
        batches: &[RecordBatch],
        num_rows: usize,
        schema: &SchemaRef,
        on_left: &[PhysicalExprRef],
        random_state: &RandomState,
        null_equality: NullEquality,
        reservation: &mut MemoryReservation,
    ) -> Result<Option<(Box<ShardedJoinHashMap>, RecordBatch, Arc<MemoryReservation>)>>
    {
        let shards = self.shards(num_rows);
        if shards < 2 {
            return Ok(None);
        }
        // One u64 hash and at most one u32 row id per build row.
        let transient = num_rows.checked_mul(12).ok_or_else(|| {
            internal_datafusion_err!("parallel hash build size overflow")
        })?;
        if !admitted(reservation.try_grow(transient))? {
            return Ok(None);
        }
        let copies = Arc::new(reservation.new_empty());
        let stop = Arc::new(AtomicBool::new(false));
        let _stop_on_drop = StopOnDrop(Arc::clone(&stop));

        // One compaction decision per view column over the whole build: a
        // buffer that several groups reference is copied at most once in
        // total, never once per group.
        let compactions: Arc<Vec<Option<Compaction>>> = Arc::new(
            schema
                .fields()
                .iter()
                .enumerate()
                .map(|(column, field)| {
                    is_view(field.data_type()).then(|| {
                        let arrays: Vec<&dyn Array> = batches
                            .iter()
                            .map(|batch| batch.column(column).as_ref())
                            .collect();
                        Compaction::of(&arrays)
                    })
                })
                .collect(),
        );
        let mut work = Vec::new();
        for (offset, group) in groups(batches, shards) {
            let on_left = on_left.to_vec();
            let random_state = random_state.clone();
            let copies = Arc::clone(&copies);
            let compactions = Arc::clone(&compactions);
            let stop = Arc::clone(&stop);
            work.push(async move {
                hash_group(
                    &group,
                    offset,
                    shards,
                    &on_left,
                    &random_state,
                    null_equality,
                    &compactions,
                    &copies,
                    &stop,
                )
            });
        }
        let groups = Arc::new(self.run(work, &stop).await?);

        let counts: Vec<usize> = (0..shards)
            .map(|shard| groups.iter().map(|group| group.rows[shard].len()).sum())
            .collect();
        let mut table_bytes = num_rows * size_of::<u32>();
        for &count in &counts {
            table_bytes += estimate_memory_size::<(u64, u32)>(count, 0)?;
        }
        if !admitted(reservation.try_grow(table_bytes))? {
            // Every group task has finished: release the lists and any
            // compacted views before the native build runs.
            drop(groups);
            reservation.shrink(transient);
            return Ok(None);
        }

        let next: Arc<Vec<AtomicU32>> =
            Arc::new((0..num_rows).map(|_| AtomicU32::new(0)).collect());
        // Charge every concatenated column before any is allocated: views
        // copy 16 bytes per row (their data stays in the group buffers
        // charged above), other columns copy their slices.
        let mut concat_bytes = 0;
        for (column, field) in schema.fields().iter().enumerate() {
            concat_bytes += if is_view(field.data_type()) {
                num_rows * (VIEW_BYTES + 1)
            } else {
                let mut bytes = 0;
                for batch in batches {
                    bytes += batch.column(column).to_data().get_slice_memory_size()?;
                }
                bytes
            };
        }
        if !admitted(copies.try_grow(concat_bytes))? {
            drop(groups);
            reservation.shrink(transient + table_bytes);
            return Ok(None);
        }
        let mut tables = Vec::with_capacity(shards);
        for (shard, count) in counts.into_iter().enumerate() {
            let groups = Arc::clone(&groups);
            let next = Arc::clone(&next);
            #[cfg(test)]
            let fault = self.fault.clone();
            tables.push(async move {
                #[cfg(test)]
                if let Some(fault) = &fault {
                    tokio::time::sleep(fault.table_delay).await;
                }
                let table = fill_shard(&groups, shard, count, &next);
                #[cfg(test)]
                if let Some(fault) = &fault {
                    fault.tables_done.fetch_add(1, Ordering::Relaxed);
                }
                Ok(table)
            });
        }
        let mut columns = Vec::with_capacity(schema.fields().len());
        for column in 0..schema.fields().len() {
            // A view column joins its per-group compacted arrays over their
            // deduplicated buffers: at most one data buffer per group, or the
            // shared source buffers once each, copying only the views.
            let view = is_view(schema.field(column).data_type());
            let arrays: Vec<ArrayRef> = if view {
                groups
                    .iter()
                    .filter_map(|group| group.views[column].clone())
                    .collect()
            } else {
                batches
                    .iter()
                    .map(|batch| Arc::clone(batch.column(column)))
                    .collect()
            };
            #[cfg(test)]
            let panic = self.fault.as_ref().is_some_and(|fault| fault.panic_columns);
            columns.push(async move {
                #[cfg(test)]
                assert!(!panic, "injected column task panic");
                let arrays: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
                if view {
                    shared(&arrays)
                } else {
                    Ok(concat(&arrays)?)
                }
            });
        }
        // Both sets run to completion before either error or panic surfaces:
        // a panic resumed by one set must not drop the other set's join,
        // which would leave its tasks running detached and uncharged.
        let (tables, columns) = futures::join!(
            AssertUnwindSafe(self.run(tables, &stop)).catch_unwind(),
            AssertUnwindSafe(self.run(columns, &stop)).catch_unwind(),
        );
        let (tables, columns) = match (tables, columns) {
            (Ok(tables), Ok(columns)) => (tables?, columns?),
            (Err(panic), _) | (_, Err(panic)) => std::panic::resume_unwind(panic),
        };
        drop(groups);
        reservation.shrink(transient);

        // Every task holding a clone has completed; fall back to a copy if a
        // runtime were to destroy a finished future late.
        let next: Vec<u32> = match Arc::try_unwrap(next) {
            Ok(next) => next.into_iter().map(AtomicU32::into_inner).collect(),
            Err(shared) => shared.iter().map(|n| n.load(Ordering::Relaxed)).collect(),
        };
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(num_rows)),
        )?;
        MetricBuilder::new(&self.metrics)
            .counter(SHARDS_METRIC, self.partition)
            .add(shards);
        let chained = tables.iter().any(|(_, chained)| *chained);
        let tables = tables.into_iter().map(|(table, _)| table).collect();
        Ok(Some((
            Box::new(ShardedJoinHashMap::new(tables, next, chained)),
            batch,
            copies,
        )))
    }

    /// [`Self::build`] inside `collect_left_input`, under the 56 build
    /// accounting (`concat_build_batches`): `reservation` holds `input_bytes`
    /// for `batches`. On success `batches` holds only the consolidated batch,
    /// the copies' charge has moved to `reservation`, and `reservation` is
    /// trimmed to the tables, the chain and the memory that the consolidated
    /// batch retains once the inputs are dropped. `None` leaves everything as
    /// it was for the native build; an error leaves `batches` with every input.
    #[expect(clippy::too_many_arguments)]
    pub(super) async fn build_consolidated(
        &self,
        batches: &mut Vec<RecordBatch>,
        num_rows: usize,
        input_bytes: usize,
        schema: &SchemaRef,
        on_left: &[PhysicalExprRef],
        random_state: &RandomState,
        null_equality: NullEquality,
        reservation: &mut MemoryReservation,
        metrics: &BuildProbeJoinMetrics,
    ) -> Result<Option<Box<dyn JoinHashMapType>>> {
        let before = reservation.size();
        let Some((map, batch, copies)) = self
            .build(
                batches,
                num_rows,
                schema,
                on_left,
                random_state,
                null_equality,
                reservation,
            )
            .await?
        else {
            return Ok(None);
        };
        // The tables and the chain (the transient hashes are released).
        metrics
            .build_mem_used
            .add(reservation.size().saturating_sub(before));
        // Every build task has finished. Drop the inputs, then hold only what
        // the consolidated batch retains, including any source view buffers
        // it still shares.
        batches.clear();
        batches.push(batch);
        let retained = get_record_batch_memory_size(&batches[0]);
        adopt_copies(reservation, &copies, input_bytes, retained, metrics)?;
        Ok(Some(map as Box<dyn JoinHashMapType>))
    }

    /// Build tasks run plan work (key evaluation), so under an attempt scope
    /// they register as producers before their first poll; a task that the
    /// attempt stops resolves to the typed stopped error.
    fn spawn<T: Send + 'static>(
        &self,
        work: impl Future<Output = Result<T>> + Send + 'static,
    ) -> Result<ScopedTask<Result<T>>> {
        spawn_producer(self.scope.as_ref(), work, stopped_error)
    }

    /// Spawns every work item and [`join`]s them. A failed
    /// work item raises `stop` itself, so that the others end early; a spawn
    /// failure raises it and joins the tasks already spawned before it
    /// returns.
    async fn run<T: Send + 'static>(
        &self,
        work: Vec<impl Future<Output = Result<T>> + Send + 'static>,
        stop: &Arc<AtomicBool>,
    ) -> Result<Vec<T>> {
        let mut tasks = Vec::with_capacity(work.len());
        for work in work {
            let raise = Arc::clone(stop);
            let work = async move {
                let result = work.await;
                if result.is_err() {
                    raise.store(true, Ordering::Relaxed);
                }
                result
            };
            match self.spawn(work) {
                Ok(task) => tasks.push(task),
                Err(error) => {
                    stop.store(true, Ordering::Relaxed);
                    let _ = join(tasks, stop).await;
                    return Err(error);
                }
            }
        }
        join(tasks, stop).await
    }
}

/// Whether a reservation step was admitted; `false` for a pool refusal
/// (resources exhausted), the error otherwise.
fn admitted(result: Result<()>) -> Result<bool> {
    match result {
        Ok(()) => Ok(true),
        Err(error)
            if matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)) =>
        {
            Ok(false)
        }
        Err(error) => Err(error),
    }
}

/// Moves the charge of `copies` to `reservation`, then trims the part of
/// `reservation` that held `held` bytes of batches (before the copies) to the
/// `retained` bytes of their replacement, as `concat_build_batches` does. The
/// replaced batches must already be dropped. Never charges a byte twice.
fn adopt_copies(
    reservation: &MemoryReservation,
    copies: &MemoryReservation,
    held: usize,
    retained: usize,
    metrics: &BuildProbeJoinMetrics,
) -> Result<()> {
    let copied = copies.size();
    copies.shrink(copied);
    reservation.grow(copied);
    metrics.build_mem_used.add(copied);
    let held = held + copied;
    if retained > held {
        reservation.try_grow(retained - held)?;
        metrics.build_mem_used.add(retained - held);
    } else {
        reservation.shrink(held - retained);
        metrics.build_mem_used.sub(held - retained);
    }
    Ok(())
}

/// The error of a task that `stop` ended early; another task's error is
/// always the one returned.
const STOPPED: &str = "parallel hash build stopped by another task's error";

fn stopped(stop: &AtomicBool) -> Result<()> {
    if stop.load(Ordering::Relaxed) {
        return Err(DataFusionError::Execution(STOPPED.into()));
    }
    Ok(())
}

/// Awaits every task, also after one has failed or panicked, so that no
/// task (or a charge or buffer it holds) outlives the build. Returns the
/// outputs in task order, or else resumes the first panic or returns the
/// first error other than a [`STOPPED`] one.
async fn join<T: Send + 'static>(
    tasks: Vec<ScopedTask<Result<T>>>,
    stop: &AtomicBool,
) -> Result<Vec<T>> {
    let mut output = Vec::with_capacity(tasks.len());
    let (mut error, mut panic) = (None::<DataFusionError>, None);
    for task in tasks {
        // `join_unwind` resumes a task's panic here; catch it so that the
        // remaining tasks are still joined before it is resumed.
        let failure = match AssertUnwindSafe(task.join_unwind()).catch_unwind().await {
            Ok(Ok(Ok(value))) => {
                output.push(value);
                continue;
            }
            Ok(Ok(Err(failure))) => failure,
            Ok(Err(failure)) => DataFusionError::ExecutionJoin(Box::new(failure)),
            Err(payload) => {
                stop.store(true, Ordering::Relaxed);
                panic.get_or_insert(payload);
                continue;
            }
        };
        stop.store(true, Ordering::Relaxed);
        // A cause replaces a STOPPED error that was joined first.
        if error.as_ref().is_none_or(is_stopped) {
            error = Some(failure);
        }
    }
    if let Some(panic) = panic {
        std::panic::resume_unwind(panic);
    }
    match error {
        Some(error) => Err(error),
        None => Ok(output),
    }
}

fn is_stopped(error: &DataFusionError) -> bool {
    matches!(error, DataFusionError::Execution(message) if message == STOPPED)
}

/// Raises `stop` when the build ends, also when its future is dropped, so
/// that group tasks that are still running end early.
struct StopOnDrop(Arc<AtomicBool>);

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// At most `count` contiguous groups of whole batches with similar row
/// counts, each with the row offset of its first batch.
fn groups(batches: &[RecordBatch], count: usize) -> Vec<(usize, Vec<RecordBatch>)> {
    let total: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let target = total.div_ceil(count).max(1);
    let mut groups = Vec::with_capacity(count);
    let mut current = Vec::new();
    let (mut offset, mut rows) = (0, 0);
    for batch in batches {
        rows += batch.num_rows();
        current.push(batch.clone());
        if rows >= target {
            groups.push((offset, std::mem::take(&mut current)));
            offset += rows;
            rows = 0;
        }
    }
    if !current.is_empty() {
        groups.push((offset, current));
    }
    groups
}

/// The hashes of one batch group and its matchable rows by shard.
struct HashedGroup {
    offset: usize,
    hashes: Vec<u64>,
    /// Global row ids, ascending, per shard.
    rows: Vec<Vec<u32>>,
    /// Per column: this group's string-view values, concatenated and
    /// compacted; `None` for other types.
    views: Vec<Option<ArrayRef>>,
}

/// A view column of a batch with more data buffers than this is compacted
/// before it becomes (part of) a build.
const MAX_VIEW_BUFFERS: usize = 64;

/// Bytes of one string or binary view.
const VIEW_BYTES: usize = 16;

/// Bytes outside the views that `gc` copies: every non-inline value.
fn referenced_bytes(array: &dyn Array) -> usize {
    let views = match array.data_type() {
        DataType::Utf8View => array.as_string_view().views(),
        DataType::BinaryView => array.as_binary_view().views(),
        _ => return 0,
    };
    views
        .iter()
        .map(|view| *view as u32 as usize)
        .filter(|&len| len > MAX_INLINE_VIEW_LEN)
        .sum()
}

/// Values up to this length live inside their view.
const MAX_INLINE_VIEW_LEN: usize = 12;

/// Whether `data_type` stores string or binary views.
pub(super) fn is_view(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8View | DataType::BinaryView)
}

/// `arrays` (string or binary views of one type) as one array whose data
/// needs no more bytes than the sources' distinct buffers, and the bytes that
/// building it allocates.
///
/// Concatenation keeps every source data buffer (arrow-array
/// `GenericByteViewBuilder::append_array`), and `take` keeps every buffer of
/// its source. A shared build of many scan batches, or of another shared
/// join's output, would otherwise hand each probe output batch thousands of
/// buffer references, which every later `take` and coalescer visits per
/// batch. DuckDB copies build strings into the hash table's own heap
/// (`TupleDataCollection`); a partitioned build gets compact input from the
/// repartition coalescer.
///
/// When the referenced (non-inline) bytes fit the distinct source buffers,
/// they are copied into dense buffers (`gc`). Otherwise (views shared into a
/// dictionary page), copying would expand them row by row, so the views are
/// concatenated over the deduplicated source buffers instead, copying no
/// data. Either way the result references at most the source buffers' bytes.
/// [`Compaction::of`] decides over every array of the build, so the parts
/// that batch groups compact separately copy no more than the whole.
fn compacted(arrays: &[&dyn Array], compaction: Compaction) -> Result<ArrayRef> {
    match compaction {
        Compaction::Copy => Ok(match arrays[0].data_type() {
            DataType::Utf8View => Arc::new(concat(arrays)?.as_string_view().gc()),
            DataType::BinaryView => Arc::new(concat(arrays)?.as_binary_view().gc()),
            _ => concat(arrays)?,
        }),
        Compaction::Share => shared(arrays),
    }
}

/// How [`compacted`] compacts the view arrays of one build column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compaction {
    /// Copy the referenced values, which fit the distinct source buffers.
    Copy,
    /// Keep the deduplicated source buffers and copy only the views.
    Share,
}

impl Compaction {
    /// The decision for all of `arrays`: copying when their referenced bytes
    /// fit the allocations behind them. Any partition of `arrays` then copies at
    /// most those buffers' bytes in total.
    fn of(arrays: &[&dyn Array]) -> Self {
        let referenced: usize = arrays.iter().map(|array| referenced_bytes(*array)).sum();
        if referenced <= allocated_bytes(arrays) {
            Self::Copy
        } else {
            Self::Share
        }
    }
}

/// View `arrays` of one type concatenated over their deduplicated data
/// buffers, copying no value bytes.
fn shared(arrays: &[&dyn Array]) -> Result<ArrayRef> {
    Ok(match arrays[0].data_type() {
        DataType::Utf8View => Arc::new(deduplicated(
            &arrays
                .iter()
                .map(|array| array.as_string_view())
                .collect::<Vec<_>>(),
        )?),
        DataType::BinaryView => Arc::new(deduplicated(
            &arrays
                .iter()
                .map(|array| array.as_binary_view())
                .collect::<Vec<_>>(),
        )?),
        _ => concat(arrays)?,
    })
}

/// Bytes that [`compacted`] allocates for `arrays`: the concatenated views
/// (plus validity) that `gc` copies again with the referenced bytes, or the
/// remapped views alone.
fn compaction_bytes(arrays: &[&dyn Array], compaction: Compaction) -> usize {
    let rows: usize = arrays.iter().map(|array| array.len()).sum();
    let views = rows * (VIEW_BYTES + 1);
    match compaction {
        Compaction::Copy => {
            2 * views
                + arrays
                    .iter()
                    .map(|array| referenced_bytes(*array))
                    .sum::<usize>()
        }
        Compaction::Share => views,
    }
}

/// Bytes of the distinct allocations behind the data buffers of view
/// `arrays`, counted as the build's input charge counts them
/// (`RecordBatchMemoryCounter`: once per allocation, at its capacity), so
/// that slices of one allocation do not count more than once.
fn allocated_bytes(arrays: &[&dyn Array]) -> usize {
    let mut seen = HashSet::new();
    let mut bytes = 0;
    for array in arrays {
        let buffers = match array.data_type() {
            DataType::Utf8View => array.as_string_view().data_buffers(),
            DataType::BinaryView => array.as_binary_view().data_buffers(),
            _ => continue,
        };
        for buffer in buffers.iter() {
            if seen.insert(buffer.data_ptr().addr()) {
                bytes += buffer.capacity();
            }
        }
    }
    bytes
}

/// Views of `arrays` concatenated over their deduplicated data buffers.
fn deduplicated<T: ByteViewType + ?Sized>(
    arrays: &[&GenericByteViewArray<T>],
) -> Result<GenericByteViewArray<T>> {
    let rows = arrays.iter().map(|array| array.len()).sum();
    let mut index = HashMap::new();
    let mut buffers: Vec<Buffer> = Vec::new();
    let mut views = Vec::with_capacity(rows);
    let mut nulls = NullBufferBuilder::new(rows);
    for array in arrays {
        let remap: Vec<u32> = array
            .data_buffers()
            .iter()
            .map(|buffer| {
                *index
                    .entry((buffer.as_ptr() as usize, buffer.len()))
                    .or_insert_with(|| {
                        buffers.push(buffer.clone());
                        (buffers.len() - 1) as u32
                    })
            })
            .collect();
        for view in array.views().iter() {
            if (*view as u32 as usize) <= MAX_INLINE_VIEW_LEN {
                views.push(*view);
                continue;
            }
            let mut long = ByteView::from(*view);
            // Only a null slot can hold an out-of-range index; it becomes
            // an empty inline view.
            match remap.get(long.buffer_index as usize) {
                Some(&buffer) => {
                    long.buffer_index = buffer;
                    views.push(long.as_u128());
                }
                None => views.push(0),
            }
        }
        match array.nulls() {
            Some(valid) => nulls.append_buffer(valid),
            None => nulls.append_n_non_nulls(array.len()),
        }
    }
    // SAFETY: every source array is a valid view array. Each kept long view
    // keeps its length, prefix and offset and points to the same bytes, now
    // at the deduplicated index of its source buffer; inline views and the
    // empty views written for null slots reference no buffer. So every view
    // is in bounds and, for strings, its bytes were already validated UTF-8.
    let array = unsafe {
        GenericByteViewArray::new_unchecked(
            ScalarBuffer::from(views),
            buffers.into(),
            nulls.finish(),
        )
    };
    Ok(array)
}

/// Compacts the consolidated native build `batch` ([`compact_views`]) under
/// the 56 build accounting: `reservation` holds exactly what `batch` retains
/// (`concat_build_batches` trimmed it), and `batches` holds `batch`. The copy
/// is charged before it is made; once the wide batch and its key arrays are
/// dropped, `reservation` holds what the compacted batch retains, and
/// `batches` holds it. Returns the batch and its key arrays.
pub(super) fn compact_build(
    batch: RecordBatch,
    left_values: Vec<ArrayRef>,
    batches: &mut Vec<RecordBatch>,
    on_left: &[PhysicalExprRef],
    reservation: &MemoryReservation,
    metrics: &BuildProbeJoinMetrics,
) -> Result<(RecordBatch, Vec<ArrayRef>)> {
    let held = get_record_batch_memory_size(&batch);
    let copies = reservation.new_empty();
    match compact_views(batch, &copies)? {
        (batch, true) => {
            // Release every reference to the wide arrays first.
            drop(left_values);
            batches.clear();
            batches.push(batch.clone());
            let left_values = evaluate_expressions_to_arrays(on_left, &batch)?;
            let retained = get_record_batch_memory_size(&batch);
            adopt_copies(reservation, &copies, held, retained, metrics)?;
            Ok((batch, left_values))
        }
        (batch, false) => Ok((batch, left_values)),
    }
}

/// `batch` with every view column that references more than
/// [`MAX_VIEW_BUFFERS`] data buffers compacted, and whether any was. The
/// copies are charged to `copies` before they are allocated.
pub(super) fn compact_views(
    batch: RecordBatch,
    copies: &MemoryReservation,
) -> Result<(RecordBatch, bool)> {
    let wide = |array: &ArrayRef| match array.data_type() {
        DataType::Utf8View => {
            array.as_string_view().data_buffers().len() > MAX_VIEW_BUFFERS
        }
        DataType::BinaryView => {
            array.as_binary_view().data_buffers().len() > MAX_VIEW_BUFFERS
        }
        _ => false,
    };
    if !batch.columns().iter().any(wide) {
        return Ok((batch, false));
    }
    // The batch holds the whole build, so each decision is build-wide.
    let decide = |array: &ArrayRef| Compaction::of(&[array.as_ref()]);
    let bytes: usize = batch
        .columns()
        .iter()
        .filter(|array| wide(array))
        .map(|array| compaction_bytes(&[array.as_ref()], decide(array)))
        .sum();
    copies.try_grow(bytes)?;
    let columns = batch
        .columns()
        .iter()
        .map(|array| {
            if wide(array) {
                compacted(&[array.as_ref()], decide(array))
            } else {
                Ok(Arc::clone(array))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let batch = RecordBatch::try_new_with_options(
        batch.schema(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )?;
    Ok((batch, true))
}

#[inline]
fn shard_of(hash: u64, mask: usize) -> usize {
    (hash >> SHARD_SHIFT) as usize & mask
}

#[expect(clippy::too_many_arguments)]
fn hash_group(
    batches: &[RecordBatch],
    offset: usize,
    shards: usize,
    on_left: &[PhysicalExprRef],
    random_state: &RandomState,
    null_equality: NullEquality,
    compactions: &[Option<Compaction>],
    copies: &MemoryReservation,
    stop: &AtomicBool,
) -> Result<HashedGroup> {
    let total: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let mask = shards - 1;
    let mut hashes = vec![0; total];
    let mut valid = Vec::with_capacity(batches.len());
    let mut start = 0;
    for batch in batches {
        stopped(stop)?;
        let keys = evaluate_expressions_to_arrays(on_left, batch)?;
        create_hashes(
            &keys,
            random_state,
            &mut hashes[start..start + batch.num_rows()],
        )?;
        valid.push(matchable_join_keys(&keys, null_equality));
        start += batch.num_rows();
    }
    let matchable = |row: usize, start: usize, valid: &Option<NullBuffer>| {
        valid
            .as_ref()
            .is_none_or(|nulls| nulls.is_valid(row - start))
    };
    let mut counts = vec![0; shards];
    let mut start = 0;
    for (batch, valid) in batches.iter().zip(&valid) {
        for row in start..start + batch.num_rows() {
            if matchable(row, start, valid) {
                counts[shard_of(hashes[row], mask)] += 1;
            }
        }
        start += batch.num_rows();
    }
    let mut rows: Vec<Vec<u32>> = counts.into_iter().map(Vec::with_capacity).collect();
    let mut start = 0;
    for (batch, valid) in batches.iter().zip(&valid) {
        for row in start..start + batch.num_rows() {
            if matchable(row, start, valid) {
                // `shards` is at least 2 only below u32::MAX rows.
                rows[shard_of(hashes[row], mask)].push((offset + row) as u32);
            }
        }
        start += batch.num_rows();
    }
    let views = match batches.first() {
        Some(first) => (0..first.num_columns())
            .map(|column| {
                let Some(compaction) = compactions[column] else {
                    return Ok(None);
                };
                stopped(stop)?;
                let arrays: Vec<&dyn Array> = batches
                    .iter()
                    .map(|batch| batch.column(column).as_ref())
                    .collect();
                // Charge the copies first; the intermediate concatenated
                // views of a `gc` are released once it has run.
                let bytes = compaction_bytes(&arrays, compaction);
                copies.try_grow(bytes)?;
                let array = compacted(&arrays, compaction)?;
                copies.shrink(
                    bytes.saturating_sub(array.get_array_memory_size().min(bytes)),
                );
                Ok(Some(array))
            })
            .collect::<Result<_>>()?,
        None => vec![],
    };
    Ok(HashedGroup {
        offset,
        hashes,
        rows,
        views,
    })
}

/// Inserts every row of `shard` in descending row order, and reports whether
/// any hash got a second row. Each row's chain slot belongs to exactly one
/// shard, so relaxed stores race with nothing; joining the task orders them
/// before any read.
fn fill_shard(
    groups: &[HashedGroup],
    shard: usize,
    count: usize,
    next: &[AtomicU32],
) -> (HashTable<(u64, u32)>, bool) {
    let mut table = HashTable::with_capacity(count);
    let mut chained = false;
    for group in groups.iter().rev() {
        for &row in group.rows[shard].iter().rev() {
            let hash = group.hashes[row as usize - group.offset];
            match table.entry(hash, |&(h, _)| h == hash, |&(h, _)| h) {
                Occupied(mut entry) => {
                    let head = &mut entry.get_mut().1;
                    next[row as usize].store(*head, Ordering::Relaxed);
                    *head = row + 1;
                    chained = true;
                }
                Vacant(entry) => {
                    entry.insert((hash, row + 1));
                }
            }
        }
    }
    (table, chained)
}

/// Hash-sharded [`JoinHashMapType`] with one chain array over all rows.
///
/// Entries and the chain use the `JoinHashMapU32` encoding: a table entry
/// holds its hash's head row + 1, and `next[row]` the following row + 1, or 0.
///
/// As `JoinHashMapU32`, the unique-key fast path depends on whether any hash
/// holds two rows (`chained`), not on `len == next.len()`: unmatchable NULL-key
/// rows own a chain slot but no entry, and must not disable it.
pub(super) struct ShardedJoinHashMap {
    shards: Vec<HashTable<(u64, u32)>>,
    mask: usize,
    next: Vec<u32>,
    len: usize,
    chained: bool,
}

impl ShardedJoinHashMap {
    pub(super) fn new(
        shards: Vec<HashTable<(u64, u32)>>,
        next: Vec<u32>,
        chained: bool,
    ) -> Self {
        assert!(
            shards.len().is_power_of_two(),
            "shard count must be a power of two"
        );
        let len = shards.iter().map(HashTable::len).sum();
        Self {
            mask: shards.len() - 1,
            shards,
            next,
            len,
            chained,
        }
    }

    /// Whether any hash holds more than one row.
    #[cfg(test)]
    pub(super) fn is_chained(&self) -> bool {
        self.chained
    }

    #[inline]
    fn head(&self, hash: u64) -> Option<u32> {
        self.shards[shard_of(hash, self.mask)]
            .find(hash, |&(h, _)| h == hash)
            .map(|&(_, head)| head)
    }
}

impl Debug for ShardedJoinHashMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ShardedJoinHashMap")
            .field("shards", &self.shards.len())
            .field("len", &self.len)
            .finish()
    }
}

impl JoinHashMapType for ShardedJoinHashMap {
    fn extend_zero(&mut self, len: usize) {
        self.next.resize(self.next.len() + len, 0);
    }

    fn update_from_iter<'a>(
        &mut self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + Send + 'a>,
        deleted_offset: usize,
    ) {
        for (row, &hash) in iter {
            let row_id = u32::try_from(row + 1).expect("shared build rows fit u32");
            let table = &mut self.shards[shard_of(hash, self.mask)];
            match table.entry(hash, |&(h, _)| h == hash, |&(h, _)| h) {
                Occupied(mut entry) => {
                    let head = &mut entry.get_mut().1;
                    self.next[row - deleted_offset] = *head;
                    *head = row_id;
                    self.chained = true;
                }
                Vacant(entry) => {
                    entry.insert((hash, row_id));
                    self.len += 1;
                }
            }
        }
    }

    fn get_matched_indices<'a>(
        &self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + 'a>,
        deleted_offset: Option<usize>,
    ) -> (Vec<u32>, Vec<u64>) {
        let mut input_indices = vec![];
        let mut match_indices = vec![];
        let offset = deleted_offset.unwrap_or(0) as u64;
        for (probe, &hash) in iter {
            let Some(head) = self.head(hash) else {
                continue;
            };
            let mut row = u64::from(head) - 1;
            loop {
                if row < offset {
                    break;
                }
                let index = row - offset;
                match_indices.push(index);
                input_indices.push(probe as u32);
                let next = self.next[index as usize];
                if next == 0 {
                    break;
                }
                row = u64::from(next) - 1;
            }
        }
        (input_indices, match_indices)
    }

    fn get_matched_indices_with_limit_offset(
        &self,
        hash_values: &[u64],
        valid_keys: Option<&NullBuffer>,
        limit: usize,
        offset: ProbeOffset,
        input_indices: &mut Vec<u32>,
        match_indices: &mut Vec<u64>,
    ) -> Option<ProbeOffset> {
        input_indices.clear();
        match_indices.clear();

        // Unique hashes: no chain to follow (as `JoinHashMapU32`).
        if !self.chained {
            debug_assert!(
                matches!(offset, AtRow { .. }),
                "unique keys cannot have mid-chain"
            );
            let (AtRow { row: start } | MidChain { row: start, .. }) = offset;
            let end = (start + limit).min(hash_values.len());
            for (i, &hash) in hash_values[start..end].iter().enumerate() {
                if valid_keys.is_some_and(|valid| valid.is_null(start + i)) {
                    continue;
                }
                if let Some(head) = self.head(hash) {
                    input_indices.push((start + i) as u32);
                    match_indices.push(u64::from(head - 1));
                }
            }
            return (end != hash_values.len()).then_some(AtRow { row: end });
        }

        let mut remaining = limit;
        let to_skip = match offset {
            AtRow { row } => row,
            MidChain { row: idx, next } => {
                let is_last = idx == hash_values.len() - 1;
                if let Some(resume) = traverse_chain(
                    &self.next,
                    idx,
                    next as u32,
                    &mut remaining,
                    input_indices,
                    match_indices,
                    is_last,
                ) {
                    return Some(resume);
                }
                idx + 1
            }
        };
        // An empty probe batch has no last row (and no match).
        let last = hash_values.len().checked_sub(1)?;
        for (i, &hash) in hash_values[to_skip..].iter().enumerate() {
            let row = to_skip + i;
            if valid_keys.is_some_and(|valid| valid.is_null(row)) {
                continue;
            }
            if let Some(head) = self.head(hash)
                && let Some(resume) = traverse_chain(
                    &self.next,
                    row,
                    head,
                    &mut remaining,
                    input_indices,
                    match_indices,
                    row == last,
                )
            {
                return Some(resume);
            }
        }
        None
    }

    fn contain_hashes(&self, hash_values: &[u64]) -> BooleanArray {
        let buffer = BooleanBuffer::collect_bool(hash_values.len(), |i| {
            self.head(hash_values[i]).is_some()
        });
        BooleanArray::new(buffer, None)
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn len(&self) -> usize {
        self.len
    }
}
