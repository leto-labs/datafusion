// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow family JS: a hybrid hash join inside [`HashJoinExec`].
//!
//! Until a build reservation is refused, an eligible join runs the code it
//! runs without this module. At the first refusal ("relief") the join moves
//! the build rows it holds into hash partitions and writes whole
//! partitions, largest first, until it has released the refused bytes and at
//! least one eighth of what it held. A written partition stays written:
//! later build rows of it go to its file. The partitions that stayed become
//! the ordinary in-memory table.
//!
//! During the probe, rows of written partitions go to this stream's probe
//! file of the partition, and the rest are joined as usual. After its
//! in-memory pass, a stream joins the written partitions with stock
//! `CollectLeft` [`HashJoinExec`]s over the files (see [`Mode`]); no stream
//! waits for another, so a split attempt may run its partitions one after
//! another. Such a join is eligible too: if its build is refused it spills
//! on other hash bits, one level deeper. A table refused when the rows that
//! stayed are small is built on the join's floor, its workspace.
//!
//! A row whose key cannot match (a NULL key under `NullEqualsNothing`) never
//! goes to a key partition: a build row of this kind goes to one
//! "unmatchable" partition when the join emits unmatched build rows (and is
//! dropped otherwise), and a probe row of this kind stays in the batch.
//!
//! Each key value belongs to one partition at every level, so every join
//! type except the null-aware anti join (not eligible) is the union of its
//! per-partition results. See ADR 0068 and the `add-join-spill` change.
//!
//! The hooks in `exec.rs` and `stream.rs` are marked "Longbow family JS".

mod files;
#[cfg(test)]
mod tests;

use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::task::{Context, Poll};

use arrow::array::{ArrayRef, BooleanArray, RecordBatch, UInt32Array, UInt64Array};
use arrow::compute::{concat_batches, filter_record_batch, take_record_batch};
use arrow::datatypes::SchemaRef;
use datafusion_common::config::ExecutionOptions;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::utils::memory::{
    RecordBatchMemoryCounter, get_record_batch_memory_size,
};
use datafusion_common::{
    DataFusionError, JoinType, NullEquality, Result, config_err, internal_err,
};
use datafusion_execution::memory_pool::{
    MemoryConsumer, MemoryPool, MemoryReservation, UnboundedMemoryPool,
};
use datafusion_execution::{SpillFile, TaskContext};
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays;
use futures::stream::BoxStream;
use futures::{StreamExt, ready};

pub(crate) use files::SpillFilesExec;
use files::{FileSource, Origin};

use super::exec::{HashJoinExec, HashJoinExecBuilder, JoinLeftData};
use crate::attempt_tasks::AttemptTasks;
use crate::execution_plan::ExecutionPlanProperties;
use crate::joins::PartitionMode;
use crate::joins::utils::{
    ColumnIndex, JoinFilter, OnceAsync, build_batch_from_indices,
    emits_unmatched_left_rows, matchable_join_keys, need_produce_result_in_final,
};
use crate::metrics::{
    Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder, SpillMetrics,
};
use crate::spill::in_progress_spill_file::InProgressSpillFile;
use crate::spill::spill_manager::SpillManager;
use crate::stream::RecordBatchStreamAdapter;
use crate::{ExecutionPlan, SendableRecordBatchStream};
use datafusion_common::JoinSide;
use datafusion_physical_expr::projection::ProjectionRef;

/// Multipliers of the partition function, one per level. Odd, and none is
/// the slot multiplier of the join table (`SLOT_MUL`, the golden ratio), so
/// the partition bits are independent of a row's slot, of its 32-bit tag
/// (the low hash half) and of the bits of every other level.
const LEVEL_MUL: [u64; 8] = [
    0xBF58_476D_1CE4_E5B9,
    0x94D0_49BB_1331_11EB,
    0xFF51_AFD7_ED55_8CCD,
    0xC4CE_B9FE_1A85_EC53,
    0xD6E8_FEB8_6659_FD93,
    0xA076_1D64_78BD_642F,
    0xE703_7ED1_A0B4_28DB,
    0x8EBC_6AF0_9C88_C6E3,
];

/// Rows of one buffered write: buffers flush at this many rows.
const WRITE_BATCH_ROWS: usize = 8192;

/// The partition of a row with join hash `hash` at spill `level`, out of
/// `1 << (64 - shift)` partitions.
pub(super) fn partition_of(hash: u64, level: usize, shift: u32) -> usize {
    (hash.wrapping_mul(LEVEL_MUL[level % LEVEL_MUL.len()]) >> shift) as usize
}

/// Whether `error` is a refused reservation that writing partitions answers.
pub(super) fn is_refusal(error: &DataFusionError) -> bool {
    matches!(error.find_root(), DataFusionError::ResourcesExhausted(_))
}

/// The spill options of `datafusion.execution`.
#[derive(Debug, Clone, Copy)]
pub(super) struct SpillOptions {
    /// Fan-out of each level, a power of two.
    partitions: usize,
    /// `64 - log2(partitions)`.
    shift: u32,
    /// Workspace of a spilling join stream, in bytes.
    workspace: usize,
    /// Highest level that may spill.
    max_depth: usize,
    /// Test only: force spill mode at this many first-level build rows.
    force_rows: usize,
}

impl SpillOptions {
    fn try_new(options: &ExecutionOptions) -> Result<Self> {
        let partitions = options.hash_join_spill_partitions;
        if !partitions.is_power_of_two() || !(2..=256).contains(&partitions) {
            return config_err!(
                "datafusion.execution.hash_join_spill_partitions must be a power of two \
                 from 2 to 256, got {partitions}"
            );
        }
        let max_depth = options.hash_join_spill_max_depth;
        if max_depth >= LEVEL_MUL.len() {
            return config_err!(
                "datafusion.execution.hash_join_spill_max_depth must be at most {}, \
                 got {max_depth}: deeper levels would reuse partition bits",
                LEVEL_MUL.len() - 1
            );
        }
        Ok(Self {
            partitions,
            shift: 64 - partitions.trailing_zeros(),
            workspace: options.hash_join_spill_reservation_bytes,
            max_depth: options.hash_join_spill_max_depth,
            force_rows: options.hash_join_spill_test_force_rows,
        })
    }

    /// Bytes of one partition's write buffer: half the workspace, shared by
    /// the partitions and the unmatchable partition.
    fn buffer_bytes(&self) -> usize {
        (self.workspace / (2 * (self.partitions + 1))).max(64 << 10)
    }
}

/// Spill metrics of a join: the standard spill counters plus each side's
/// rows and bytes, written partitions and levels. Registered at the first
/// relief, so a join that does not spill shows none of them. Joins of
/// written partitions report into the first-level join's metrics.
#[derive(Debug, Clone)]
pub(crate) struct JoinSpillMetrics {
    spill: SpillMetrics,
    build_rows: Count,
    build_bytes: Count,
    probe_rows: Count,
    probe_bytes: Count,
    partitions: Count,
    /// Levels that wrote a partition; 1 when only the first level did.
    depth: Gauge,
}

impl JoinSpillMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize, depth: Gauge) -> Self {
        let counter =
            |name: &'static str| MetricBuilder::new(metrics).counter(name, partition);
        Self {
            spill: SpillMetrics::new(metrics, partition),
            build_rows: counter("build_spilled_rows"),
            build_bytes: counter("build_spilled_bytes"),
            probe_rows: counter("probe_spilled_rows"),
            probe_bytes: counter("probe_spilled_bytes"),
            partitions: counter("spilled_partitions"),
            depth,
        }
    }
}

/// Spill metrics registered on first use.
#[derive(Debug)]
enum LazyMetrics {
    Own {
        set: ExecutionPlanMetricsSet,
        partition: usize,
        cell: OnceLock<JoinSpillMetrics>,
        /// One depth gauge for all of the join's streams, so that its
        /// aggregate is the deepest level rather than a sum.
        shared: Arc<SpillShared>,
    },
    Inherited(JoinSpillMetrics),
}

impl LazyMetrics {
    fn get(&self) -> &JoinSpillMetrics {
        match self {
            Self::Own {
                set,
                partition,
                cell,
                shared,
            } => cell.get_or_init(|| {
                let depth = shared
                    .depth
                    .get_or_init(|| MetricBuilder::new(set).gauge("spill_depth", 0))
                    .clone();
                JoinSpillMetrics::new(set, *partition, depth)
            }),
            Self::Inherited(metrics) => metrics,
        }
    }

    /// The same metrics for another output partition of the join.
    fn for_partition(&self, partition: usize) -> Self {
        match self {
            Self::Own { set, shared, .. } => Self::Own {
                set: set.clone(),
                partition,
                cell: OnceLock::new(),
                shared: Arc::clone(shared),
            },
            Self::Inherited(metrics) => Self::Inherited(metrics.clone()),
        }
    }
}

/// The parameters of a join of one written partition.
#[derive(Clone)]
struct SubJoinTemplate {
    on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
    filter: Option<JoinFilter>,
    join_type: JoinType,
    projection: Option<ProjectionRef>,
    null_equality: NullEquality,
    build_schema: SchemaRef,
    probe_schema: SchemaRef,
}

/// Whether and how one execution of a join may spill, decided in
/// `HashJoinExec::execute`.
pub(super) enum JoinSpill {
    /// `enable_hash_join_spilling` is off: the join fails as before.
    Disabled,
    /// Spilling is on but this join cannot spill correctly.
    Ineligible(String),
    Eligible(Box<Eligible>),
}

pub(super) struct Eligible {
    options: SpillOptions,
    level: usize,
    context: Arc<TaskContext>,
    metrics: LazyMetrics,
    template: SubJoinTemplate,
    on_left: Vec<PhysicalExprRef>,
    random_state: RandomState,
    shared: Arc<SpillShared>,
    mode: PartitionMode,
    /// Probe streams of this plan instance (the build's members).
    probe_streams: usize,
    /// The `CollectLeft` build that the exec holds for its streams.
    shared_build: Option<Weak<OnceAsync<JoinLeftData>>>,
}

impl JoinSpill {
    /// Decides eligibility (design decision 2).
    #[expect(clippy::too_many_arguments)]
    pub(super) fn decide(
        join: &HashJoinExec,
        prepared: bool,
        metrics: &ExecutionPlanMetricsSet,
        shared: &Arc<SpillShared>,
        partition: usize,
        context: &Arc<TaskContext>,
        random_state: &RandomState,
        shared_build: &Arc<OnceAsync<JoinLeftData>>,
    ) -> Result<Self> {
        let config = context.session_config().options();
        if !config.execution.enable_hash_join_spilling {
            return Ok(Self::Disabled);
        }
        let leaf = join.left().downcast_ref::<SpillFilesExec>();
        let level = leaf.map_or(0, SpillFilesExec::level);
        let options = SpillOptions::try_new(&config.execution)?;
        let reason = if join.null_aware {
            Some("it is a null-aware anti join".to_string())
        } else if join.right().output_ordering().is_some() {
            Some("its probe input has an output ordering".to_string())
        } else if join.right().boundedness().is_unbounded() {
            Some("its probe input is unbounded".to_string())
        } else if prepared {
            Some("it uses a prepared build".to_string())
        } else if !context.runtime_env().disk_manager.tmp_files_enabled() {
            Some("the disk manager cannot create files".to_string())
        } else if level > options.max_depth {
            let origin = leaf.map(SpillFilesExec::origin);
            let rows = origin.map_or(0, |origin| origin.rows);
            Some(if origin.is_some_and(|origin| !origin.divided) {
                format!(
                    "one key value fills its partition of {rows} build rows, which no \
                     spill level divides (the build-chunk join of add-join-spill \
                     section 3 is not implemented); it is at spill level {level}, \
                     above datafusion.execution.hash_join_spill_max_depth ({})",
                    options.max_depth
                )
            } else {
                format!(
                    "the memory pool had no room for its partition of {rows} build rows \
                     at spill level {level}, above \
                     datafusion.execution.hash_join_spill_max_depth ({}), and the \
                     partition is larger than its workspace floor",
                    options.max_depth
                )
            })
        } else if !matches!(
            join.mode,
            PartitionMode::CollectLeft | PartitionMode::Partitioned
        ) {
            Some(format!("its partition mode is {:?}", join.mode))
        } else {
            None
        };
        if let Some(reason) = reason {
            return Ok(Self::Ineligible(reason));
        }
        let right_partitions = join.right().output_partitioning().partition_count();
        let probe_streams = match join.mode {
            PartitionMode::CollectLeft => shared
                .partitions()
                .map_or(right_partitions, |range| range.len()),
            _ => 1,
        };
        let metrics = match leaf {
            Some(leaf) => LazyMetrics::Inherited(leaf.metrics().clone()),
            None => LazyMetrics::Own {
                set: metrics.clone(),
                partition,
                cell: OnceLock::new(),
                shared: Arc::clone(shared),
            },
        };
        Ok(Self::Eligible(Box::new(Eligible {
            options,
            level,
            context: Arc::clone(context),
            metrics,
            template: SubJoinTemplate {
                on: join.on.clone(),
                filter: join.filter.clone(),
                join_type: join.join_type,
                projection: join.projection.clone(),
                null_equality: join.null_equality,
                build_schema: join.left().schema(),
                probe_schema: join.right().schema(),
            },
            on_left: join.on.iter().map(|(left, _)| Arc::clone(left)).collect(),
            random_state: random_state.clone(),
            shared: Arc::clone(shared),
            mode: join.mode,
            probe_streams,
            shared_build: (join.mode == PartitionMode::CollectLeft)
                .then(|| Arc::downgrade(shared_build)),
        })))
    }

    /// Whether the build consumer is registered as spillable.
    pub(super) fn can_spill(&self) -> bool {
        matches!(self, Self::Eligible(_))
    }

    /// The build-side state for `collect_left_input`.
    pub(super) fn build(&self) -> Option<BuildSpill> {
        match self {
            Self::Eligible(eligible) => Some(BuildSpill::new(eligible)),
            _ => None,
        }
    }

    /// Adds why the join did not spill to a refused build reservation.
    pub(super) fn explain_refusal(
        &self,
    ) -> impl Fn(DataFusionError) -> DataFusionError + Send + Sync + 'static + use<> {
        let reason = match self {
            Self::Ineligible(reason) => Some(reason.clone()),
            _ => None,
        };
        // Only the join's own refusals: the build future also polls the
        // build input, whose operators can be refused too.
        move |error| match &reason {
            Some(reason)
                if is_refusal(&error) && error.to_string().contains("HashJoinInput") =>
            {
                error.context(format!("The hash join did not spill because {reason}"))
            }
            _ => error,
        }
    }

    /// The probe-side state of output stream `partition`.
    pub(super) fn stream(
        &self,
        partition: usize,
        schema: SchemaRef,
        column_indices: &[ColumnIndex],
    ) -> Option<Box<StreamSpill>> {
        let Self::Eligible(eligible) = self else {
            return None;
        };
        let (slot, shared) = match eligible.mode {
            PartitionMode::CollectLeft => (partition, Some(Arc::clone(&eligible.shared))),
            _ => (0, None),
        };
        Some(Box::new(StreamSpill {
            context: Arc::clone(&eligible.context),
            slot,
            shared,
            metrics: eligible.metrics.for_partition(partition),
            schema: Arc::clone(&eligible.template.probe_schema),
            options: eligible.options,
            null_equality: eligible.template.null_equality,
            output: Arc::new(UnmatchedOutput {
                schema,
                column_indices: column_indices.to_vec(),
                join_type: eligible.template.join_type,
                probe_schema: Arc::clone(&eligible.template.probe_schema),
            }),
            phase: Phase::Waiting,
        }))
    }
}

/// A finished spill file and its row count.
#[derive(Clone)]
pub(crate) struct SpilledFile {
    file: Arc<dyn SpillFile>,
    rows: usize,
}

/// Spill manager of one side, scoped to the attempt.
fn spill_manager(
    context: &TaskContext,
    metrics: &JoinSpillMetrics,
    schema: SchemaRef,
) -> SpillManager {
    SpillManager::new(context.runtime_env(), metrics.spill.clone(), schema)
        .with_attempt_tasks(AttemptTasks::from_task(context))
        .with_compression_type(context.session_config().spill_compression())
}

/// One side's row and byte counters.
#[derive(Clone, Copy)]
enum Side {
    Build,
    Probe,
}

impl JoinSpillMetrics {
    fn side(&self, side: Side) -> (&Count, &Count) {
        match side {
            Side::Build => (&self.build_rows, &self.build_bytes),
            Side::Probe => (&self.probe_rows, &self.probe_bytes),
        }
    }
}

/// A partition file being written, with a buffer of rows not yet written.
struct PartWriter {
    file: InProgressSpillFile,
    buffer: Vec<RecordBatch>,
    buffered_rows: usize,
    buffered_bytes: usize,
    rows: usize,
}

impl PartWriter {
    fn new(manager: &SpillManager, description: &str) -> Result<Self> {
        Ok(Self {
            file: manager.create_in_progress_file(description)?,
            buffer: vec![],
            buffered_rows: 0,
            buffered_bytes: 0,
            rows: 0,
        })
    }

    /// Writes `batches` as batches of up to [`WRITE_BATCH_ROWS`] rows.
    fn write(
        &mut self,
        batches: &[RecordBatch],
        metrics: &JoinSpillMetrics,
        side: Side,
    ) -> Result<()> {
        let schema = Arc::clone(self.file.spill_manager.schema());
        let (rows_metric, bytes_metric) = metrics.side(side);
        let before = metrics.spill.spilled_bytes.value();
        let mut start = 0;
        while start < batches.len() {
            let mut end = start;
            let mut rows = 0;
            while end < batches.len() && (end == start || rows < WRITE_BATCH_ROWS) {
                rows += batches[end].num_rows();
                end += 1;
            }
            let group = &batches[start..end];
            if group.len() == 1 {
                self.file.append_batch(&group[0])?;
            } else {
                self.file.append_batch(&concat_batches(&schema, group)?)?;
            }
            self.rows += rows;
            rows_metric.add(rows);
            start = end;
        }
        bytes_metric.add(metrics.spill.spilled_bytes.value().saturating_sub(before));
        Ok(())
    }

    /// Buffers `piece`; flushes when the buffer is full. Returns the bytes
    /// the flush released.
    fn append(
        &mut self,
        piece: RecordBatch,
        limit: usize,
        metrics: &JoinSpillMetrics,
        side: Side,
    ) -> Result<usize> {
        self.buffered_rows += piece.num_rows();
        self.buffered_bytes += get_record_batch_memory_size(&piece);
        self.buffer.push(piece);
        if self.buffered_rows >= WRITE_BATCH_ROWS || self.buffered_bytes >= limit {
            self.flush(metrics, side)
        } else {
            Ok(0)
        }
    }

    fn flush(&mut self, metrics: &JoinSpillMetrics, side: Side) -> Result<usize> {
        let buffer = std::mem::take(&mut self.buffer);
        self.write(&buffer, metrics, side)?;
        self.buffered_rows = 0;
        Ok(std::mem::take(&mut self.buffered_bytes))
    }

    fn finish(
        mut self,
        metrics: &JoinSpillMetrics,
        side: Side,
    ) -> Result<Option<SpilledFile>> {
        self.flush(metrics, side)?;
        let before = metrics.spill.spilled_bytes.value();
        let file = self.file.finish()?;
        metrics
            .side(side)
            .1
            .add(metrics.spill.spilled_bytes.value().saturating_sub(before));
        Ok(file.map(|file| SpilledFile {
            file,
            rows: self.rows,
        }))
    }
}

/// One build partition: rows in memory, or a file.
enum Part {
    Memory {
        batches: Vec<RecordBatch>,
        rows: usize,
        bytes: usize,
    },
    Writing(Box<PartWriter>),
    Written(Option<SpilledFile>),
}

impl Part {
    fn empty() -> Self {
        Self::Memory {
            batches: vec![],
            rows: 0,
            bytes: 0,
        }
    }

    fn is_written(&self) -> bool {
        !matches!(self, Self::Memory { .. })
    }
}

/// The build side of a spilling join, kept by `collect_left_input`.
pub(super) struct BuildSpill {
    options: SpillOptions,
    level: usize,
    keep_unmatchable: bool,
    null_equality: NullEquality,
    on_left: Vec<PhysicalExprRef>,
    random_state: RandomState,
    context: Arc<TaskContext>,
    metrics: JoinSpillMetrics,
    manager: Option<SpillManager>,
    template: SubJoinTemplate,
    shared: Option<Arc<SpillShared>>,
    probe_streams: usize,
    shared_build: Option<Weak<OnceAsync<JoinLeftData>>>,
    lazy_metrics: LazyMetrics,
    active: bool,
    /// One part per partition, then the unmatchable part.
    parts: Vec<Part>,
    /// Key partitions that received rows.
    seen: Vec<bool>,
    /// Counts each buffer of the in-memory parts once.
    counter: RecordBatchMemoryCounter,
    /// What the build reservation holds for the in-memory parts.
    charged: usize,
    /// The workspace, kept from the first relief to the end of the build:
    /// charge moved from the parts as writing releases it, plus what the
    /// join held too little to give.
    workspace: Vec<MemoryReservation>,
    /// Workspace not yet moved from released charge.
    workspace_debt: usize,
    hashes: Vec<u64>,
}

/// The size of a charge held in pieces.
fn held(pieces: &[MemoryReservation]) -> usize {
    pieces.iter().map(MemoryReservation::size).sum()
}

/// Moves `bytes` out of `pieces` into new reservations (at most all).
fn take_held(
    pieces: &mut Vec<MemoryReservation>,
    mut bytes: usize,
) -> Vec<MemoryReservation> {
    let mut out = vec![];
    while bytes > 0 {
        let Some(piece) = pieces.last_mut() else {
            break;
        };
        let size = piece.size();
        if size <= bytes {
            bytes -= size;
            out.push(pieces.pop().expect("a last piece"));
        } else {
            out.push(piece.split(bytes));
            bytes = 0;
        }
    }
    out
}

impl BuildSpill {
    fn new(eligible: &Eligible) -> Self {
        let parts = (0..=eligible.options.partitions)
            .map(|_| Part::empty())
            .collect();
        Self {
            options: eligible.options,
            level: eligible.level,
            keep_unmatchable: emits_unmatched_left_rows(eligible.template.join_type),
            null_equality: eligible.template.null_equality,
            on_left: eligible.on_left.clone(),
            random_state: eligible.random_state.clone(),
            context: Arc::clone(&eligible.context),
            // Replaced at activation; registering metrics is deferred to it.
            metrics: JoinSpillMetrics::new(
                &ExecutionPlanMetricsSet::new(),
                0,
                Gauge::new(),
            ),
            manager: None,
            template: eligible.template.clone(),
            shared: (eligible.mode == PartitionMode::CollectLeft)
                .then(|| Arc::clone(&eligible.shared)),
            probe_streams: eligible.probe_streams,
            shared_build: eligible.shared_build.clone(),
            lazy_metrics: eligible.metrics.for_partition(match &eligible.metrics {
                LazyMetrics::Own { partition, .. } => *partition,
                LazyMetrics::Inherited(_) => 0,
            }),
            active: false,
            parts,
            seen: vec![false; eligible.options.partitions],
            counter: RecordBatchMemoryCounter::new(),
            charged: 0,
            workspace: vec![],
            workspace_debt: 0,
            hashes: vec![],
        }
    }

    /// Whether the build is in spill mode.
    pub(super) fn is_active(&self) -> bool {
        self.active
    }

    /// Whether the test option forces spill mode at `rows` build rows.
    pub(super) fn should_force(&self, rows: usize) -> bool {
        !self.active
            && self.level == 0
            && self.options.force_rows > 0
            && rows >= self.options.force_rows
    }

    /// Hook 1: takes `batch` after its reservation was refused, or in spill
    /// mode. The first call moves the `retained` batches, which
    /// `reservation` holds, into partitions.
    pub(super) fn push(
        &mut self,
        batch: &RecordBatch,
        retained: &mut Vec<RecordBatch>,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        let activating = !self.active;
        let forced = activating
            && self.should_force(
                retained.iter().map(RecordBatch::num_rows).sum::<usize>()
                    + batch.num_rows(),
            );
        if activating {
            self.activate(retained, reservation)?;
        }
        for (part, piece) in self.split(batch)? {
            self.add(part, piece)?;
        }
        if forced {
            for part in (1..self.options.partitions).step_by(2) {
                if matches!(&self.parts[part], Part::Memory { rows, .. } if *rows > 0) {
                    self.write_part(part)?;
                }
            }
            self.recount(reservation);
        } else if activating {
            // The first relief; the workspace comes out of what it releases.
            self.relieve(0, reservation)?;
        }
        if activating {
            self.settle_workspace(reservation);
        }
        self.charge(reservation)
    }

    /// The table over a build that held its whole input was refused: the
    /// rows in `batches`, still charged to `reservation`, enter spill mode.
    /// Rows whose table fits the floor stay; otherwise half are written (a
    /// table and its copy are about twice the rows).
    pub(super) fn take_refused_build(
        &mut self,
        batches: &mut Vec<RecordBatch>,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        self.activate(batches, reservation)?;
        self.settle_workspace(reservation);
        if !self.fits_floor(self.charged) {
            self.relieve(self.charged / 2, reservation)?;
        }
        Ok(())
    }

    fn activate(
        &mut self,
        retained: &mut Vec<RecordBatch>,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        self.active = true;
        self.metrics = self.lazy_metrics.get().clone();
        self.manager = Some(spill_manager(
            &self.context,
            &self.metrics,
            Arc::clone(&self.template.build_schema),
        ));
        self.workspace_debt = self.options.workspace;
        for batch in std::mem::take(retained) {
            for (part, piece) in self.split(&batch)? {
                self.add(part, piece)?;
            }
        }
        // `reservation` held the retained batches; it now holds the parts,
        // which are live whether or not the pool admits them.
        let counted = self.counter.memory_usage();
        if counted >= reservation.size() {
            reservation.grow(counted - reservation.size());
        } else {
            reservation.shrink(reservation.size() - counted);
        }
        self.charged = counted;
        Ok(())
    }

    /// A join that held less than its workspace grows the difference
    /// without refusal (design decision 8).
    fn settle_workspace(&mut self, reservation: &MemoryReservation) {
        if self.workspace_debt > 0 {
            let piece = reservation.new_empty();
            piece.grow(self.workspace_debt);
            self.workspace.push(piece);
            self.workspace_debt = 0;
        }
    }

    /// Splits `batch` by partition; drops unmatchable rows of a join that
    /// does not emit unmatched build rows.
    fn split(&mut self, batch: &RecordBatch) -> Result<Vec<(usize, RecordBatch)>> {
        let mut pieces = split_by_partition(
            batch,
            &self.on_left,
            &self.random_state,
            self.null_equality,
            self.level,
            self.options.shift,
            self.options.partitions,
            &mut self.hashes,
        )?;
        if !self.keep_unmatchable {
            pieces.retain(|(part, _)| *part != self.options.partitions);
        }
        Ok(pieces)
    }

    fn add(&mut self, part: usize, piece: RecordBatch) -> Result<()> {
        let limit = self.options.buffer_bytes();
        if let Some(seen) = self.seen.get_mut(part) {
            *seen = true;
        }
        match &mut self.parts[part] {
            Part::Memory {
                batches,
                rows,
                bytes,
            } => {
                self.counter.count_batch(&piece);
                *rows += piece.num_rows();
                *bytes += get_record_batch_memory_size(&piece);
                batches.push(piece);
            }
            Part::Writing(writer) => {
                writer.append(piece, limit, &self.metrics, Side::Build)?;
            }
            Part::Written(_) => {
                return internal_err!("hash join spill: rows for a finished partition");
            }
        }
        Ok(())
    }

    /// Grows `reservation` to the parts in memory, writing partitions while
    /// it is refused.
    fn charge(&mut self, reservation: &MemoryReservation) -> Result<()> {
        loop {
            let counted = self.counter.memory_usage();
            if counted <= self.charged {
                return Ok(());
            }
            let delta = counted - self.charged;
            match reservation.try_grow(delta) {
                Ok(()) => {
                    self.charged = counted;
                    return Ok(());
                }
                Err(error) if is_refusal(&error) => {
                    if self.fits_floor(counted) {
                        self.borrow(delta, reservation);
                        self.charged = counted;
                        return Ok(());
                    }
                    if !self.relieve(delta, reservation)? {
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Whether rows charged at `counted` bytes and their table fit in the
    /// join's floor, its workspace: the part of `counted` not yet charged,
    /// which the workspace covers now, and the table's estimate, which it
    /// covers at the build: a copy of the rows, 48 bytes of table and keys
    /// per row, and 256 bytes of fixed parts. An empty table always fits: it is the
    /// smallest unit and is charged without refusal.
    fn fits_floor(&self, counted: usize) -> bool {
        let rows = self.rows_in_memory();
        let borrow = counted.saturating_sub(self.charged);
        // The fixed parts of an empty table are charged 137 to 196 bytes.
        rows == 0 || borrow + counted + rows * 48 + 256 <= held(&self.workspace)
    }

    /// Covers `bytes` of the rows' charge with workspace: the charge grows
    /// before the workspace shrinks, so the pool never sees them free.
    fn borrow(&mut self, bytes: usize, reservation: &MemoryReservation) {
        reservation.grow(bytes);
        drop(take_held(&mut self.workspace, bytes));
    }

    /// Writes whole partitions, largest first, until the refused `need` and
    /// one eighth of what the parts held are released. Returns whether it
    /// wrote any.
    fn relieve(&mut self, need: usize, reservation: &MemoryReservation) -> Result<bool> {
        let target = need.max(self.charged / 8);
        let mut released = 0;
        let mut wrote = false;
        while released < target {
            let largest = self
                .parts
                .iter()
                .enumerate()
                .filter_map(|(index, part)| match part {
                    Part::Memory { rows, bytes, .. } if *rows > 0 => {
                        Some((index, *bytes))
                    }
                    _ => None,
                })
                .max_by_key(|(_, bytes)| *bytes);
            let Some((part, bytes)) = largest else {
                break;
            };
            self.write_part(part)?;
            released += bytes;
            wrote = true;
        }
        if wrote {
            self.recount(reservation);
        }
        Ok(wrote)
    }

    /// Writes the rows of in-memory part `part` and keeps its file open.
    fn write_part(&mut self, part: usize) -> Result<()> {
        let Part::Memory { batches, .. } =
            std::mem::replace(&mut self.parts[part], Part::empty())
        else {
            return internal_err!("hash join spill: partition {part} is already written");
        };
        let manager = self.manager.as_ref().ok_or_else(|| {
            datafusion_common::internal_datafusion_err!("no spill manager")
        })?;
        let mut writer = PartWriter::new(manager, "HashJoin build partition")?;
        writer.write(&batches, &self.metrics, Side::Build)?;
        drop(batches);
        self.parts[part] = Part::Writing(Box::new(writer));
        self.metrics.partitions.add(1);
        self.metrics.depth.set_max(self.level + 1);
        Ok(())
    }

    /// Counts the in-memory parts again after a write. Released charge pays
    /// the workspace first, without passing through the pool; the rest goes
    /// back to the pool.
    fn recount(&mut self, reservation: &MemoryReservation) {
        let mut counter = RecordBatchMemoryCounter::new();
        for part in &self.parts {
            if let Part::Memory { batches, .. } = part {
                for batch in batches {
                    counter.count_batch(batch);
                }
            }
        }
        let counted = counter.memory_usage();
        self.counter = counter;
        if counted < self.charged {
            let released = self.charged - counted;
            let pay = released.min(self.workspace_debt).min(reservation.size());
            if pay > 0 {
                self.workspace.push(reservation.split(pay));
                self.workspace_debt -= pay;
            }
            reservation.shrink((released - pay).min(reservation.size()));
            self.charged = counted;
        }
    }

    fn finish_writers(&mut self) -> Result<()> {
        for part in &mut self.parts {
            if let Part::Writing(_) = part {
                let Part::Writing(writer) = std::mem::replace(part, Part::Written(None))
                else {
                    unreachable!()
                };
                *part = Part::Written(writer.finish(&self.metrics, Side::Build)?);
            }
        }
        Ok(())
    }

    fn take_in_memory(&mut self) -> Vec<RecordBatch> {
        let mut out = vec![];
        for part in &mut self.parts {
            if let Part::Memory {
                batches,
                rows,
                bytes,
            } = part
            {
                out.append(batches);
                *rows = 0;
                *bytes = 0;
            }
        }
        self.counter = RecordBatchMemoryCounter::new();
        out
    }

    fn rows_in_memory(&self) -> usize {
        self.parts
            .iter()
            .map(|part| match part {
                Part::Memory { rows, .. } => *rows,
                _ => 0,
            })
            .sum()
    }

    /// End of the build input. `build` builds the table over the rows that
    /// stayed, charging `reservation`, which holds the rows; when it fails
    /// the rows come back in its batches, still charged. On a refusal the
    /// table is built on the join's floor if it fits there, otherwise more
    /// partitions are written and `build` runs again.
    ///
    /// Returns the table, the written partitions, and, for a table built on
    /// the floor, the charge that must replace the table's own reservation
    /// (which then belongs to a private pool).
    pub(super) fn finish<T>(
        mut self,
        reservation: &mut MemoryReservation,
        mut build: impl FnMut(&mut Vec<RecordBatch>, &mut MemoryReservation) -> Result<T>,
    ) -> Result<(T, Option<Arc<SpilledBuild>>, Option<MemoryReservation>)> {
        self.finish_writers()?;
        loop {
            let mut batches = self.take_in_memory();
            match build(&mut batches, reservation) {
                Ok(table) => {
                    self.charged = 0;
                    let spilled = self.into_spilled(vec![], false);
                    return Ok((table, spilled, None));
                }
                Err(error) if is_refusal(&error) => {
                    // The rows are back in `batches`; `reservation` still
                    // holds them (and whatever the attempt grew).
                    for batch in std::mem::take(&mut batches) {
                        for (part, piece) in self.split(&batch)? {
                            self.add(part, piece)?;
                        }
                    }
                    let counted = self.counter.memory_usage();
                    if counted >= reservation.size() {
                        reservation.grow(counted - reservation.size());
                    } else {
                        reservation.shrink(reservation.size() - counted);
                    }
                    self.charged = counted;
                    if self.fits_floor(self.charged) {
                        return self.build_on_floor(reservation, &mut build);
                    }
                    // A table and its copy are about twice the rows: write
                    // half of what stayed before trying again.
                    if !self.relieve(self.charged / 2, reservation)? {
                        return Err(error);
                    }
                    self.finish_writers()?;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Builds the table on a private pool and moves its size from the
    /// workspace into the build's charge, so the real pool sees no growth
    /// beyond what the join already held (the floor of design decision 5).
    fn build_on_floor<T>(
        mut self,
        reservation: &mut MemoryReservation,
        build: &mut impl FnMut(&mut Vec<RecordBatch>, &mut MemoryReservation) -> Result<T>,
    ) -> Result<(T, Option<Arc<SpilledBuild>>, Option<MemoryReservation>)> {
        let mut batches = self.take_in_memory();
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let mut attempt = MemoryConsumer::new("HashJoinInput")
            .with_can_spill(true)
            .register(&pool);
        let table = build(&mut batches, &mut attempt)?;
        let extra = pool.reserved().saturating_sub(self.charged);
        let mut floor = take_held(&mut self.workspace, extra);
        let short = extra.saturating_sub(held(&floor));
        if short > 0 {
            // Only an empty table can be short of its estimate.
            let piece = reservation.new_empty();
            piece.grow(short);
            floor.push(piece);
        }
        self.charged = 0;
        let charge = reservation.take();
        let spilled = self.into_spilled(floor, true);
        Ok((table, spilled, Some(charge)))
    }

    fn into_spilled(
        mut self,
        floor: Vec<MemoryReservation>,
        on_floor: bool,
    ) -> Option<Arc<SpilledBuild>> {
        self.workspace.clear();
        if !self.parts.iter().any(Part::is_written) && !on_floor {
            return None;
        }
        let range = self.shared.as_ref().and_then(|shared| shared.partitions());
        let slots = self
            .shared
            .as_ref()
            .map_or(1, |shared| shared.probe_partitions());
        let manager = self.manager.take()?;
        let partitions = self.options.partitions;
        let written: Vec<bool> = self.parts[..partitions]
            .iter()
            .map(Part::is_written)
            .collect();
        let divided = self.seen.iter().filter(|seen| **seen).count() > 1;
        let build_files = self
            .parts
            .into_iter()
            .map(|part| match part {
                Part::Written(file) => file,
                _ => None,
            })
            .collect();
        let mode = if need_produce_result_in_final(self.template.join_type) {
            Mode::Last
        } else {
            Mode::PerStream
        };
        let members = self.probe_streams;
        let spilled = Arc::new(SpilledBuild {
            level: self.level,
            shift: self.options.shift,
            partitions,
            written,
            divided,
            build_files: Mutex::new(build_files),
            template: self.template,
            build_manager: manager,
            metrics: self.metrics,
            range,
            mode,
            probes: Arc::new(ProbeRegistry::new(slots, partitions)),
            members: Mutex::new(Members {
                members,
                finished: 0,
                abandoned: false,
                remaining: vec![members; partitions],
                joins: (0..partitions).map(|_| None).collect(),
            }),
            shared_build: self.shared_build.take(),
            _floor_charge: floor,
        });
        if let Some(shared) = &self.shared {
            shared.register(&spilled);
        }
        Some(spilled)
    }
}

/// Splits `batch` into pieces by the partition of each row, at `level`.
/// Unmatchable rows get index `partitions`.
#[expect(clippy::too_many_arguments)]
fn split_by_partition(
    batch: &RecordBatch,
    on: &[PhysicalExprRef],
    random_state: &RandomState,
    null_equality: NullEquality,
    level: usize,
    shift: u32,
    partitions: usize,
    hashes: &mut Vec<u64>,
) -> Result<Vec<(usize, RecordBatch)>> {
    let ids = partition_ids(
        batch,
        on,
        random_state,
        null_equality,
        level,
        shift,
        partitions,
        hashes,
    )?;
    let mut indices: Vec<Vec<u32>> = vec![vec![]; partitions + 1];
    for (row, part) in ids.iter().enumerate() {
        indices[*part as usize].push(row as u32);
    }
    let non_empty = indices.iter().filter(|rows| !rows.is_empty()).count();
    if non_empty == 1 {
        let part = ids.first().map_or(0, |part| *part as usize);
        return Ok(vec![(part, batch.clone())]);
    }
    indices
        .into_iter()
        .enumerate()
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(part, rows)| {
            Ok((part, take_record_batch(batch, &UInt32Array::from(rows))?))
        })
        .collect()
}

/// The partition of each row of `batch` at `level`; `partitions` for a row
/// whose key cannot match.
#[expect(clippy::too_many_arguments)]
fn partition_ids(
    batch: &RecordBatch,
    on: &[PhysicalExprRef],
    random_state: &RandomState,
    null_equality: NullEquality,
    level: usize,
    shift: u32,
    partitions: usize,
    hashes: &mut Vec<u64>,
) -> Result<Vec<u32>> {
    let keys: Vec<ArrayRef> = evaluate_expressions_to_arrays(on, batch)?;
    hashes.clear();
    hashes.resize(batch.num_rows(), 0);
    create_hashes(&keys, random_state, hashes)?;
    let valid = matchable_join_keys(&keys, null_equality);
    Ok(hashes
        .iter()
        .enumerate()
        .map(|(row, hash)| {
            if valid.as_ref().is_some_and(|valid| valid.is_null(row)) {
                partitions as u32
            } else {
                partition_of(*hash, level, shift) as u32
            }
        })
        .collect())
}

/// How the written partitions of a build are joined after the probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// The join emits only rows that a probe row produces (Inner, Right,
    /// RightSemi, RightAnti, RightMark), so its output keeps the probe's
    /// partitioning: each stream joins its own probe file of each written
    /// partition, sharing the partition's join with the streams that run it
    /// at the same time. No stream waits for another.
    PerStream,
    /// The join also emits build rows (Left, LeftSemi, LeftAnti, LeftMark,
    /// Full), whose output partitioning is unknown: the last stream to end
    /// its in-memory pass joins each written partition over every stream's
    /// probe file, so one table sees every probe row. The others end at once.
    Last,
}

/// The probe files of a build's written partitions, by stream slot and
/// partition.
pub(crate) struct ProbeRegistry(Mutex<Vec<Vec<Option<SpilledFile>>>>);

impl ProbeRegistry {
    fn new(slots: usize, partitions: usize) -> Self {
        Self(Mutex::new(vec![vec![None; partitions]; slots]))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Vec<Option<SpilledFile>>>> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn register(&self, slot: usize, files: Vec<Option<SpilledFile>>) -> Result<()> {
        match self.lock().get_mut(slot) {
            Some(entry) => {
                *entry = files;
                Ok(())
            }
            None => internal_err!("hash join spill: no probe slot {slot}"),
        }
    }

    /// Stream `slot`'s file of `part`; its reader deletes it when done.
    pub(super) fn take(&self, slot: usize, part: usize) -> Option<SpilledFile> {
        self.lock()
            .get_mut(slot)
            .and_then(|files| files.get_mut(part))
            .and_then(Option::take)
    }

    fn take_all(&self, part: usize) -> Vec<SpilledFile> {
        self.lock()
            .iter_mut()
            .filter_map(|files| files.get_mut(part).and_then(Option::take))
            .collect()
    }

    fn rows(&self, slot: usize, part: usize) -> usize {
        self.lock()
            .get(slot)
            .and_then(|files| files.get(part))
            .and_then(Option::as_ref)
            .map_or(0, |file| file.rows)
    }
}

/// What the members of a build have done. Nothing here waits.
struct Members {
    /// Streams expected to probe the build: the bound range of a split
    /// attempt, or every probe partition, or 1.
    members: usize,
    /// Members that ended their in-memory pass or left.
    finished: usize,
    /// A member left before its in-memory pass ended. A build-emitting join
    /// then joins no written partition, as the stock join emits no build
    /// rows when a probe partition stops early.
    abandoned: bool,
    /// `PerStream`: members that have not finished each partition or left;
    /// a partition's build file is deleted when this reaches 0.
    remaining: Vec<usize>,
    /// `PerStream`: the join of each partition while a stream runs it, and
    /// how many do.
    joins: Vec<Option<(Arc<dyn ExecutionPlan>, usize)>>,
}

/// The written partitions of one build, shared by the streams that probe
/// it (all of them in `CollectLeft` mode).
pub(crate) struct SpilledBuild {
    level: usize,
    shift: u32,
    partitions: usize,
    written: Vec<bool>,
    /// Whether this build's rows went to more than one key partition.
    divided: bool,
    /// One file per partition, then the unmatchable partition's.
    build_files: Mutex<Vec<Option<SpilledFile>>>,
    template: SubJoinTemplate,
    build_manager: SpillManager,
    metrics: JoinSpillMetrics,
    /// The output partitions a split attempt executes.
    range: Option<Range<usize>>,
    mode: Mode,
    probes: Arc<ProbeRegistry>,
    members: Mutex<Members>,
    /// The exec's hold on a `CollectLeft` build, released once every
    /// member ended its in-memory pass.
    shared_build: Option<Weak<OnceAsync<JoinLeftData>>>,
    /// Workspace moved to the table of a build charged on its floor; held
    /// as long as the build.
    _floor_charge: Vec<MemoryReservation>,
}

impl SpilledBuild {
    fn lock(&self) -> std::sync::MutexGuard<'_, Members> {
        self.members.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn files(&self) -> std::sync::MutexGuard<'_, Vec<Option<SpilledFile>>> {
        self.build_files.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Whether probe rows of `partition` go to a file.
    fn is_written(&self, partition: usize) -> bool {
        self.written[partition]
    }

    /// Streams that left before the build existed.
    fn left_before_build(&self, count: usize) {
        if count == 0 {
            return;
        }
        let mut members = self.lock();
        members.members = members.members.saturating_sub(count);
        members.abandoned = true;
        self.check_released(&members);
    }

    /// A member ended its in-memory pass with these probe files. Returns
    /// whether it joins the written partitions of a `Last` build.
    fn end_in_memory(
        &self,
        slot: usize,
        files: Vec<Option<SpilledFile>>,
    ) -> Result<bool> {
        self.probes.register(slot, files)?;
        let mut members = self.lock();
        members.finished += 1;
        let last = members.finished == members.members && !members.abandoned;
        self.check_released(&members);
        Ok(last)
    }

    /// A member left before its in-memory pass ended.
    fn leave_in_memory(&self) {
        let mut members = self.lock();
        members.finished += 1;
        members.abandoned = true;
        self.check_released(&members);
        drop(members);
        if self.mode == Mode::PerStream {
            for part in 0..self.partitions {
                if self.written[part] {
                    self.done_with(part);
                }
            }
        }
    }

    /// Frees the in-memory table once every member ended its in-memory
    /// pass: the exec still holds the `CollectLeft` build for streams that
    /// never came. Such a stream fails instead of building it again.
    fn check_released(&self, members: &Members) {
        if members.finished >= members.members
            && let Some(shared_build) = self.shared_build.as_ref().and_then(Weak::upgrade)
        {
            shared_build.release(datafusion_common::internal_datafusion_err!(
                "hash join spill: a stream started after every member ended its probe"
            ));
        }
    }

    /// The partitions that stream `slot` joins: in `PerStream` mode those
    /// where its own probe file has rows; in `Last` mode, for the last
    /// member, every written partition and the unmatchable one.
    fn runs(&self, slot: usize, last: bool) -> Vec<usize> {
        match self.mode {
            Mode::PerStream => {
                let mut runs = vec![];
                for part in 0..self.partitions {
                    if !self.written[part] {
                        continue;
                    }
                    if self.probes.rows(slot, part) > 0 {
                        runs.push(part);
                    } else {
                        self.probes.take(slot, part);
                        self.done_with(part);
                    }
                }
                runs
            }
            Mode::Last if last => {
                let files = self.files();
                (0..=self.partitions)
                    .filter(|part| files[*part].is_some())
                    .collect()
            }
            Mode::Last => vec![],
        }
    }

    /// `PerStream`: a member is done with `part` (finished it, skipped it
    /// or left); the last one deletes its build file.
    fn done_with(&self, part: usize) {
        let mut members = self.lock();
        let remaining = &mut members.remaining[part];
        *remaining = remaining.saturating_sub(1);
        if *remaining == 0 {
            members.joins[part] = None;
            drop(members);
            self.files()[part] = None;
        }
    }

    /// `PerStream`: the join of `part`, shared with the streams that run it
    /// now, or a new one over the retained build file.
    fn shared_join(self: &Arc<Self>, part: usize) -> Result<JoinUse> {
        let mut members = self.lock();
        if let Some((join, users)) = &mut members.joins[part] {
            *users += 1;
            return Ok(JoinUse {
                build: Arc::clone(self),
                part,
                join: Arc::clone(join),
            });
        }
        let build_file = self.files()[part].clone();
        let probe = FileSource::Registry {
            registry: Arc::clone(&self.probes),
            part,
            partitions: self.probes.lock().len(),
        };
        let join = self.join_over(build_file, probe, self.range.clone())?;
        members.joins[part] = Some((Arc::clone(&join), 1));
        Ok(JoinUse {
            build: Arc::clone(self),
            part,
            join,
        })
    }

    /// `PerStream`: of the partitions `pending`, the index of one whose join
    /// another stream runs now, so that streams share tables rather than
    /// build several at once; otherwise the first.
    fn next_run(&self, pending: &[usize]) -> usize {
        let members = self.lock();
        pending
            .iter()
            .position(|part| {
                *part < members.joins.len() && members.joins[*part].is_some()
            })
            .unwrap_or(0)
    }

    /// `PerStream`: a stream stopped running the join of `part`.
    fn release_join(&self, part: usize) {
        let mut members = self.lock();
        if let Some((_, users)) = &mut members.joins[part] {
            *users -= 1;
            if *users == 0 {
                members.joins[part] = None;
            }
        }
    }

    /// `Last`: the join of `part` over every member's probe file.
    fn last_join(&self, part: usize) -> Result<Arc<dyn ExecutionPlan>> {
        let build_file = self.files()[part].take();
        let probe = FileSource::Fixed(vec![self.probes.take_all(part)]);
        self.join_over(build_file, probe, None)
    }

    /// A stock `CollectLeft` join of one written partition, one level
    /// deeper.
    fn join_over(
        &self,
        build_file: Option<SpilledFile>,
        probe: FileSource,
        range: Option<Range<usize>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let level = self.level + 1;
        let template = &self.template;
        let origin = Origin {
            divided: self.divided,
            rows: build_file.as_ref().map_or(0, |file| file.rows),
        };
        let build = SpillFilesExec::new(
            Arc::clone(&template.build_schema),
            FileSource::Fixed(vec![build_file.into_iter().collect()]),
            self.build_manager.clone(),
            level,
            origin,
            self.metrics.clone(),
        );
        let probe = SpillFilesExec::new(
            Arc::clone(&template.probe_schema),
            probe,
            self.build_manager
                .with_schema(Arc::clone(&template.probe_schema)),
            level,
            origin,
            self.metrics.clone(),
        );
        let join = HashJoinExecBuilder::new(
            Arc::new(build),
            Arc::new(probe),
            template.on.clone(),
            template.join_type,
        )
        .with_filter(template.filter.clone())
        .with_projection_ref(template.projection.clone())
        .with_partition_mode(PartitionMode::CollectLeft)
        .with_null_equality(template.null_equality)
        .build()?;
        if let Some(range) = range {
            join.set_dynamic_filter_partitions(range)?;
        }
        Ok(Arc::new(join))
    }

    /// `Last`: the rows of the unmatchable partition, each an unmatched
    /// build row.
    fn unmatched_rows(&self, output: Arc<UnmatchedOutput>) -> SendableRecordBatchStream {
        let file = self.files()[self.partitions].take();
        let schema = Arc::clone(&output.schema);
        let rows = match file {
            Some(file) => match self
                .build_manager
                .read_spill_as_stream_unbuffered(Arc::clone(&file.file), None)
            {
                Ok(stream) => stream
                    .map(move |batch| {
                        // The file lives until its rows are read.
                        let _file = &file;
                        batch.and_then(|batch| output.emit(&batch))
                    })
                    .boxed(),
                Err(error) => futures::stream::once(async { Err(error) }).boxed(),
            },
            None => futures::stream::empty().boxed(),
        };
        Box::pin(RecordBatchStreamAdapter::new(schema, rows))
    }
}

/// A stream's use of a `PerStream` partition join; dropping it lets the
/// join go when no other stream runs it.
struct JoinUse {
    build: Arc<SpilledBuild>,
    part: usize,
    join: Arc<dyn ExecutionPlan>,
}

impl Drop for JoinUse {
    fn drop(&mut self) {
        self.build.release_join(self.part);
    }
}

/// Per-plan-instance state of a join: the output partitions a split
/// attempt executes, and the streams that went away before a `CollectLeft`
/// build had spilled.
#[derive(Debug, Default)]
pub(crate) struct SpillShared {
    partitions: Mutex<Option<Range<usize>>>,
    /// Probe partitions of the join, recorded at its first execution.
    probe_partitions: OnceLock<usize>,
    /// The join's `spill_depth` gauge, registered at its first relief.
    depth: OnceLock<Gauge>,
    early: Mutex<Early>,
}

#[derive(Default)]
struct Early {
    build: Option<Weak<SpilledBuild>>,
    departed: usize,
}

impl std::fmt::Debug for Early {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Early")
            .field("departed", &self.departed)
            .finish()
    }
}

impl SpillShared {
    /// Records the output partitions this plan instance executes. The last
    /// binding wins, as for a join without a dynamic filter the call was a
    /// no-op before family JS.
    pub(crate) fn bind_partitions(&self, partitions: Range<usize>) {
        *self.partitions.lock().unwrap_or_else(|p| p.into_inner()) = Some(partitions);
    }

    fn partitions(&self) -> Option<Range<usize>> {
        self.partitions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    pub(super) fn record_probe_partitions(&self, partitions: usize) {
        let _ = self.probe_partitions.set(partitions);
    }

    fn probe_partitions(&self) -> usize {
        self.probe_partitions.get().copied().unwrap_or(1)
    }

    /// Fresh state for `reset_state`, keeping the bound partitions.
    pub(crate) fn reset(&self) -> Arc<Self> {
        let fresh = Self::default();
        if let Some(partitions) = self.partitions() {
            fresh.bind_partitions(partitions);
        }
        Arc::new(fresh)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Early> {
        self.early.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn register(&self, build: &Arc<SpilledBuild>) {
        let mut early = self.lock();
        build.left_before_build(early.departed);
        early.departed = 0;
        early.build = Some(Arc::downgrade(build));
    }

    fn depart_early(&self) {
        let mut early = self.lock();
        let build = early.build.as_ref().and_then(Weak::upgrade);
        match build {
            Some(build) => {
                drop(early);
                build.leave_in_memory();
            }
            None => early.departed += 1,
        }
    }
}

/// A stream's membership of a spilled build.
struct Seat {
    build: Arc<SpilledBuild>,
    state: SeatState,
}

enum SeatState {
    InMemory,
    /// The partitions this stream still has to finish (`PerStream`).
    Written(Vec<usize>),
}

impl Drop for Seat {
    fn drop(&mut self) {
        match &self.state {
            SeatState::InMemory => self.build.leave_in_memory(),
            SeatState::Written(pending) => {
                if self.build.mode == Mode::PerStream {
                    for part in pending {
                        self.build.done_with(*part);
                    }
                }
            }
        }
    }
}

enum Phase {
    /// The build is not ready yet.
    Waiting,
    /// The build spilled: probe rows of written partitions go to files.
    Routing(Box<Router>),
    /// Joining the written partitions.
    Written(BoxStream<'static, Result<RecordBatch>>),
    Done,
}

/// The probe side of one output stream of a spilling join.
pub(super) struct StreamSpill {
    context: Arc<TaskContext>,
    /// This stream's probe-file slot, and its partition of `PerStream`
    /// partition joins.
    slot: usize,
    shared: Option<Arc<SpillShared>>,
    metrics: LazyMetrics,
    schema: SchemaRef,
    options: SpillOptions,
    null_equality: NullEquality,
    output: Arc<UnmatchedOutput>,
    phase: Phase,
}

/// How a stream emits the rows of the unmatchable partition: each is an
/// unmatched build row, so it needs no table and no other row.
struct UnmatchedOutput {
    schema: SchemaRef,
    column_indices: Vec<ColumnIndex>,
    join_type: JoinType,
    probe_schema: SchemaRef,
}

impl UnmatchedOutput {
    fn emit(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let rows = batch.num_rows();
        build_batch_from_indices(
            &self.schema,
            batch,
            &RecordBatch::new_empty(Arc::clone(&self.probe_schema)),
            &UInt64Array::from_iter_values(0..rows as u64),
            &UInt32Array::new_null(rows),
            &self.column_indices,
            JoinSide::Left,
            self.join_type,
            None,
        )
    }
}

struct Router {
    seat: Seat,
    manager: SpillManager,
    metrics: JoinSpillMetrics,
    writers: Vec<Option<PartWriter>>,
    reservation: MemoryReservation,
    hashes: Vec<u64>,
}

impl StreamSpill {
    /// The build is ready. Returns whether it spilled; if not, this state is
    /// no longer needed.
    pub(super) fn on_build(&mut self, spilled: Option<&Arc<SpilledBuild>>) -> bool {
        let Some(build) = spilled else {
            self.depart();
            return false;
        };
        let metrics = self.metrics.get().clone();
        let reservation =
            MemoryConsumer::new(format!("HashJoinProbeSpill[{}]", self.slot))
                .with_can_spill(true)
                .register(self.context.memory_pool());
        self.phase = Phase::Routing(Box::new(Router {
            seat: Seat {
                build: Arc::clone(build),
                state: SeatState::InMemory,
            },
            manager: spill_manager(&self.context, &metrics, Arc::clone(&self.schema)),
            writers: (0..build.partitions).map(|_| None).collect(),
            metrics,
            reservation,
            hashes: vec![],
        }));
        true
    }

    /// Hook 2: sends the rows of written partitions to this stream's files.
    /// Returns the rows that stay, `None` if none do.
    pub(super) fn route(
        &mut self,
        batch: RecordBatch,
        on_right: &[PhysicalExprRef],
        random_state: &RandomState,
    ) -> Result<Option<RecordBatch>> {
        let Phase::Routing(router) = &mut self.phase else {
            return Ok(Some(batch));
        };
        let build = Arc::clone(&router.seat.build);
        let ids = partition_ids(
            &batch,
            on_right,
            random_state,
            self.null_equality,
            build.level,
            build.shift,
            build.partitions,
            &mut router.hashes,
        )?;
        let goes = |part: u32| {
            (part as usize) < build.partitions && build.is_written(part as usize)
        };
        if !ids.iter().any(|part| goes(*part)) {
            return Ok(Some(batch));
        }
        let mut indices: Vec<Vec<u32>> = vec![vec![]; build.partitions];
        let keep: BooleanArray = ids
            .iter()
            .enumerate()
            .map(|(row, part)| {
                if goes(*part) {
                    indices[*part as usize].push(row as u32);
                    Some(false)
                } else {
                    Some(true)
                }
            })
            .collect();
        let limit = self.options.buffer_bytes();
        for (part, rows) in indices.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let piece = take_record_batch(&batch, &UInt32Array::from(rows))?;
            let writer = match &mut router.writers[part] {
                Some(writer) => writer,
                slot => slot.insert(PartWriter::new(
                    &router.manager,
                    "HashJoin probe partition",
                )?),
            };
            // A buffered piece is charged; one the pool refuses is written
            // at once with the buffer, so no buffer grows without refusal.
            let bytes = get_record_batch_memory_size(&piece);
            let charged = router.reservation.try_grow(bytes).is_ok();
            let mut released =
                writer.append(piece, limit, &router.metrics, Side::Probe)?;
            if !charged {
                if released == 0 {
                    released = writer.flush(&router.metrics, Side::Probe)?;
                }
                released -= bytes;
            }
            router
                .reservation
                .shrink(released.min(router.reservation.size()));
        }
        let kept = filter_record_batch(&batch, &keep)?;
        Ok((kept.num_rows() > 0).then_some(kept))
    }

    /// Hook 3: after the in-memory pass, joins the written partitions. The
    /// caller has dropped its reference to the in-memory table.
    pub(super) fn poll_written(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        if let Phase::Routing(_) = &self.phase {
            let Phase::Routing(router) = std::mem::replace(&mut self.phase, Phase::Done)
            else {
                unreachable!()
            };
            match self.start_written(*router) {
                Ok(stream) => self.phase = Phase::Written(stream),
                Err(error) => return Poll::Ready(Some(Err(error))),
            }
        }
        let Phase::Written(stream) = &mut self.phase else {
            return Poll::Ready(None);
        };
        let next = ready!(stream.poll_next_unpin(cx));
        if !matches!(next, Some(Ok(_))) {
            self.phase = Phase::Done;
        }
        Poll::Ready(next)
    }

    fn start_written(
        &self,
        router: Router,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let Router {
            mut seat,
            metrics,
            writers,
            reservation,
            ..
        } = router;
        let files = writers
            .into_iter()
            .map(|writer| {
                writer
                    .map(|writer| writer.finish(&metrics, Side::Probe))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(Option::flatten)
            .collect();
        drop(reservation);
        let last = seat.build.end_in_memory(self.slot, files)?;
        let runs = seat.build.runs(self.slot, last);
        seat.state = SeatState::Written(runs.clone());
        let state = Written {
            seat,
            runs,
            output: Arc::clone(&self.output),
            context: Arc::clone(&self.context),
            partition: self.slot,
            current: None,
            failed: false,
        };
        Ok(futures::stream::unfold(state, |mut state| async move {
            state.next().await.map(|item| (item, state))
        })
        .boxed())
    }

    fn depart(&mut self) {
        if matches!(self.phase, Phase::Waiting)
            && let Some(shared) = self.shared.take()
        {
            shared.depart_early();
        }
        self.phase = Phase::Done;
    }
}

impl Drop for StreamSpill {
    fn drop(&mut self) {
        self.depart();
    }
}

/// The written-partition phase of one stream.
struct Written {
    seat: Seat,
    /// The partitions this stream joins, in order; the first is current.
    runs: Vec<usize>,
    output: Arc<UnmatchedOutput>,
    context: Arc<TaskContext>,
    partition: usize,
    current: Option<(SendableRecordBatchStream, Option<JoinUse>)>,
    failed: bool,
}

impl Written {
    async fn next(&mut self) -> Option<Result<RecordBatch>> {
        if self.failed {
            return None;
        }
        loop {
            if let Some((stream, _)) = &mut self.current {
                match stream.next().await {
                    Some(Ok(batch)) => return Some(Ok(batch)),
                    Some(Err(error)) => {
                        self.failed = true;
                        self.current = None;
                        return Some(Err(error));
                    }
                    None => {
                        self.current = None;
                        let part = self.runs.remove(0);
                        self.seat.state = SeatState::Written(self.runs.clone());
                        if self.seat.build.mode == Mode::PerStream {
                            self.seat.build.done_with(part);
                        }
                        continue;
                    }
                }
            }
            if self.runs.is_empty() {
                return None;
            }
            let build = Arc::clone(&self.seat.build);
            if build.mode == Mode::PerStream {
                // The run to start moves to the front: `runs[0]` is current.
                let next = build.next_run(&self.runs);
                self.runs.swap(0, next);
            }
            let part = self.runs[0];
            let started = match build.mode {
                _ if part == build.partitions => {
                    Ok((build.unmatched_rows(Arc::clone(&self.output)), None))
                }
                Mode::PerStream => build.shared_join(part).and_then(|using| {
                    let stream = using
                        .join
                        .execute(self.partition, Arc::clone(&self.context))?;
                    Ok((stream, Some(using)))
                }),
                Mode::Last => build
                    .last_join(part)
                    .and_then(|join| join.execute(0, Arc::clone(&self.context)))
                    .map(|stream| (stream, None)),
            };
            match started {
                Ok(current) => self.current = Some(current),
                Err(error) => {
                    self.failed = true;
                    return Some(Err(error));
                }
            }
        }
    }
}
