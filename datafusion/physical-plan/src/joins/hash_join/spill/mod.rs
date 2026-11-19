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
//! file of the partition, and the rest are joined as usual. When every
//! stream has ended its input, each written partition is joined by a stock
//! `CollectLeft` [`HashJoinExec`] over the two files, one partition at a
//! time ([`Barrier`]). That join is eligible too: if its build is refused it
//! spills on other hash bits, one level deeper.
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
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let counter =
            |name: &'static str| MetricBuilder::new(metrics).counter(name, partition);
        Self {
            spill: SpillMetrics::new(metrics, partition),
            build_rows: counter("build_spilled_rows"),
            build_bytes: counter("build_spilled_bytes"),
            probe_rows: counter("probe_spilled_rows"),
            probe_bytes: counter("probe_spilled_bytes"),
            partitions: counter("spilled_partitions"),
            depth: MetricBuilder::new(metrics).gauge("spill_depth", partition),
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
            } => cell.get_or_init(|| JoinSpillMetrics::new(set, *partition)),
            Self::Inherited(metrics) => metrics,
        }
    }

    /// The same metrics for another output partition of the join.
    fn for_partition(&self, partition: usize) -> Self {
        match self {
            Self::Own { set, .. } => Self::Own {
                set: set.clone(),
                partition,
                cell: OnceLock::new(),
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
    /// Probe streams of this plan instance (the barrier's members).
    probe_streams: usize,
    /// The `CollectLeft` build that the exec holds for its streams.
    shared_build: Option<Weak<OnceAsync<JoinLeftData>>>,
}

impl JoinSpill {
    /// Decides eligibility (design decision 2).
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
            Some(format!(
                "its partition is at spill level {level}, above \
                 datafusion.execution.hash_join_spill_max_depth ({})",
                options.max_depth
            ))
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
        move |error| match &reason {
            Some(reason) if is_refusal(&error) => {
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
            exec_partition: slot,
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
    /// Counts each buffer of the in-memory parts once.
    counter: RecordBatchMemoryCounter,
    /// What the build reservation holds for the in-memory parts.
    charged: usize,
    /// The workspace, kept from the first relief to the end of the build.
    workspace: Option<MemoryReservation>,
    hashes: Vec<u64>,
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
            metrics: JoinSpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
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
            counter: RecordBatchMemoryCounter::new(),
            charged: 0,
            workspace: None,
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
            // Give back the workspace the join took without asking.
            self.relieve(self.options.workspace, reservation)?;
        }
        self.charge(reservation)
    }

    /// The table over a build that held its whole input was refused: the
    /// rows in `batches`, no longer charged, enter spill mode.
    pub(super) fn take_refused_build(
        &mut self,
        batches: &mut Vec<RecordBatch>,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        self.activate(batches, reservation)?;
        // A table and its copy are about twice the rows.
        self.relieve(self.options.workspace + self.charged / 2, reservation)?;
        Ok(())
    }

    fn activate(
        &mut self,
        retained: &mut Vec<RecordBatch>,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        self.active = true;
        self.metrics = self.lazy_metrics.get().clone();
        self.metrics.depth.set_max(self.level + 1);
        self.manager = Some(spill_manager(
            &self.context,
            &self.metrics,
            Arc::clone(&self.template.build_schema),
        ));
        let workspace = reservation.new_empty();
        workspace.grow(self.options.workspace);
        self.workspace = Some(workspace);
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
                    if !self.relieve(delta, reservation)? {
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }
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
        Ok(())
    }

    /// Counts the in-memory parts again after a write and releases the
    /// difference.
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
            reservation.shrink(self.charged - counted);
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

    /// End of the build input. `build` builds the table over the rows that
    /// stayed and releases what it was given when it fails; on a refusal the
    /// rows come back in its batches, more partitions are written and it
    /// runs again. Returns the table and the written partitions, if any.
    pub(super) fn finish<T>(
        mut self,
        reservation: &mut MemoryReservation,
        mut build: impl FnMut(&mut Vec<RecordBatch>, MemoryReservation) -> Result<T>,
    ) -> Result<(T, Option<Arc<SpilledBuild>>)> {
        self.finish_writers()?;
        loop {
            let mut batches = self.take_in_memory();
            let attempt = reservation.take();
            self.charged = 0;
            match build(&mut batches, attempt) {
                Ok(table) => {
                    let spilled = self.into_spilled();
                    return Ok((table, spilled));
                }
                Err(error) if is_refusal(&error) => {
                    for batch in std::mem::take(&mut batches) {
                        for (part, piece) in self.split(&batch)? {
                            self.add(part, piece)?;
                        }
                    }
                    let counted = self.counter.memory_usage();
                    reservation.grow(counted);
                    self.charged = counted;
                    // A table and its copy are about twice the rows: write
                    // half of what stayed before trying again.
                    if !self.relieve(self.charged / 2, reservation)? {
                        if self.parts.iter().any(
                            |part| matches!(part, Part::Memory { rows, .. } if *rows > 0),
                        ) {
                            return Err(error);
                        }
                        // Every row is written and even the empty table is
                        // refused: it is the smallest unit and is charged
                        // without refusal (design decision 5).
                        let pool: Arc<dyn MemoryPool> =
                            Arc::new(UnboundedMemoryPool::default());
                        let attempt = MemoryConsumer::new("HashJoinInput")
                            .with_can_spill(true)
                            .register(&pool);
                        let table = build(&mut vec![], attempt)?;
                        reservation.grow(pool.reserved());
                        let mut spilled = self.into_spilled();
                        if let Some(spilled) = spilled.as_mut().and_then(Arc::get_mut) {
                            spilled.table_charge = Some(reservation.take());
                        }
                        return Ok((table, spilled));
                    }
                    self.finish_writers()?;
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn into_spilled(mut self) -> Option<Arc<SpilledBuild>> {
        self.workspace = None;
        if !self.parts.iter().any(Part::is_written) {
            return None;
        }
        let range = self.shared.as_ref().and_then(|shared| shared.partitions());
        let slots = self.probe_slots();
        let manager = self.manager.take()?;
        let partitions = self.options.partitions;
        let written = self.parts[..partitions]
            .iter()
            .map(Part::is_written)
            .collect();
        let build_files = self
            .parts
            .into_iter()
            .map(|part| match part {
                Part::Written(file) => file,
                _ => None,
            })
            .collect();
        let spilled = Arc::new(SpilledBuild {
            level: self.level,
            shift: self.options.shift,
            partitions,
            written,
            build_files: Mutex::new(build_files),
            template: self.template,
            build_manager: manager,
            metrics: self.metrics,
            range,
            barrier: Barrier::new(self.probe_streams),
            probe_files: Mutex::new(vec![vec![None; partitions]; slots]),
            runs: OnceLock::new(),
            shared_build: self.shared_build.take(),
            table_charge: None,
        });
        if let Some(shared) = &self.shared {
            shared.register(&spilled);
        }
        Some(spilled)
    }

    fn probe_slots(&self) -> usize {
        // A CollectLeft join has one probe-file slot for each output
        // partition of the plan: stream `n` writes slot `n`.
        self.shared
            .as_ref()
            .map_or(1, |shared| shared.probe_partitions())
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

/// Probe streams step together through the written partitions of one
/// build: step 0 is the in-memory pass, step `i + 1` the join of the `i`-th
/// written partition. A stream starts a step when every member finished the
/// one before, so one written partition's build is in memory at a time.
struct Barrier {
    inner: Mutex<BarrierInner>,
    notify: tokio::sync::Notify,
}

struct BarrierInner {
    step: usize,
    /// Members that finished `step`.
    done: usize,
    /// Members that have not gone away.
    live: usize,
    /// The join of the current written partition, shared by the members.
    current: Option<(usize, Arc<dyn ExecutionPlan>)>,
}

impl Barrier {
    fn new(live: usize) -> Self {
        Self {
            inner: Mutex::new(BarrierInner {
                step: 0,
                done: 0,
                live,
                current: None,
            }),
            notify: tokio::sync::Notify::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BarrierInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn advance_if_done(inner: &mut BarrierInner) -> bool {
        if inner.live == 0 || inner.done < inner.live {
            return false;
        }
        inner.step += 1;
        inner.done = 0;
        inner.current = None;
        true
    }

    /// A member finished the current step.
    fn complete(&self) {
        let advanced = {
            let mut inner = self.lock();
            inner.done += 1;
            Self::advance_if_done(&mut inner)
        };
        if advanced {
            self.notify.notify_waiters();
        }
    }

    /// Resolves once step `step` has started.
    async fn started(&self, step: usize) {
        loop {
            // Created under the lock, so that it cannot miss an advance
            // right after the lock is released.
            let notified = {
                let inner = self.lock();
                if inner.step >= step {
                    return;
                }
                self.notify.notified()
            };
            notified.await;
        }
    }

    /// A member that completed `completed` steps went away.
    fn depart(&self, completed: usize) {
        let advanced = {
            let mut inner = self.lock();
            inner.live = inner.live.saturating_sub(1);
            if completed > inner.step {
                inner.done = inner.done.saturating_sub(1);
            }
            Self::advance_if_done(&mut inner)
        };
        if advanced {
            self.notify.notify_waiters();
        }
    }

    fn departed_before_build(&self, members: usize) {
        let mut inner = self.lock();
        inner.live = inner.live.saturating_sub(members);
    }
}

/// The written partitions of one build, shared by the streams that probe
/// it (all of them in `CollectLeft` mode).
pub(crate) struct SpilledBuild {
    level: usize,
    shift: u32,
    partitions: usize,
    written: Vec<bool>,
    /// One file per partition, then the unmatchable partition's. A
    /// partition's files move to its join when that join is created, so they
    /// are deleted once the partition is joined.
    build_files: Mutex<Vec<Option<SpilledFile>>>,
    template: SubJoinTemplate,
    build_manager: SpillManager,
    metrics: JoinSpillMetrics,
    /// The output partitions a split attempt executes.
    range: Option<Range<usize>>,
    barrier: Barrier,
    /// The probe files of each stream slot, by partition.
    probe_files: Mutex<Vec<Vec<Option<SpilledFile>>>>,
    /// The partitions joined after the probe, in order.
    runs: OnceLock<Vec<usize>>,
    /// The exec's hold on a `CollectLeft` build, released once every
    /// member finished the in-memory pass.
    shared_build: Option<Weak<OnceAsync<JoinLeftData>>>,
    /// The charge of an empty table built outside the pool.
    table_charge: Option<MemoryReservation>,
}

impl SpilledBuild {
    /// Whether probe rows of `partition` go to a file.
    fn is_written(&self, partition: usize) -> bool {
        self.written[partition]
    }

    fn register_probe_files(
        &self,
        slot: usize,
        files: Vec<Option<SpilledFile>>,
    ) -> Result<()> {
        let mut probe_files = self.probe_files.lock().unwrap_or_else(|p| p.into_inner());
        match probe_files.get_mut(slot) {
            Some(entry) => {
                *entry = files;
                Ok(())
            }
            None => internal_err!("hash join spill: no probe slot {slot}"),
        }
    }

    /// The partitions to join, decided once after every stream registered
    /// its probe files: written partitions in order, skipping those whose
    /// join is empty, then the unmatchable partition.
    fn runs(&self) -> &[usize] {
        self.runs.get_or_init(|| {
            let mut build_files =
                self.build_files.lock().unwrap_or_else(|p| p.into_inner());
            let mut probe_files =
                self.probe_files.lock().unwrap_or_else(|p| p.into_inner());
            let emits_build = need_produce_result_in_final(self.template.join_type);
            let mut runs: Vec<usize> = (0..self.partitions)
                .filter(|part| self.written[*part] && build_files[*part].is_some())
                .filter(|part| {
                    emits_build
                        || probe_files.iter().any(|slot| {
                            slot[*part].as_ref().is_some_and(|file| file.rows > 0)
                        })
                })
                .collect();
            // The files of a skipped partition are deleted now.
            for part in 0..self.partitions {
                if !runs.contains(&part) {
                    build_files[part] = None;
                    for slot in probe_files.iter_mut() {
                        slot[part] = None;
                    }
                }
            }
            if build_files[self.partitions].is_some() {
                runs.push(self.partitions);
            }
            runs
        })
    }

    /// Frees the in-memory table: after step 0 no member holds it, but the
    /// exec still holds the `CollectLeft` build for streams that never
    /// came. Such a stream fails instead of building it again.
    fn release_in_memory(&self) {
        if let Some(shared_build) = self.shared_build.as_ref().and_then(Weak::upgrade) {
            shared_build.release(datafusion_common::internal_datafusion_err!(
                "hash join spill: a stream started after the written-partition phase"
            ));
        }
    }

    /// The rows of the unmatchable partition, emitted by the first member
    /// that asks; the others emit none.
    fn unmatched_rows(&self, output: Arc<UnmatchedOutput>) -> SendableRecordBatchStream {
        let file = self
            .build_files
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get_mut(self.partitions)
            .and_then(Option::take);
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

    /// The stock join of run `run`, created by the first stream that asks.
    fn sub_join(&self, run: usize) -> Result<Arc<dyn ExecutionPlan>> {
        let mut inner = self.barrier.lock();
        if let Some((current, join)) = &inner.current
            && *current == run
        {
            return Ok(Arc::clone(join));
        }
        let part = self.runs()[run];
        let level = self.level + 1;
        let template = &self.template;
        // The partition's files move to its join, which deletes them when
        // the last stream drops it.
        let build_file = self
            .build_files
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get_mut(part)
            .and_then(Option::take);
        let build = SpillFilesExec::new(
            Arc::clone(&template.build_schema),
            vec![build_file.into_iter().collect()],
            self.build_manager.clone(),
            level,
            self.metrics.clone(),
        );
        let mut probe_files = self.probe_files.lock().unwrap_or_else(|p| p.into_inner());
        let probe = SpillFilesExec::new(
            Arc::clone(&template.probe_schema),
            probe_files
                .iter_mut()
                .map(|slot| {
                    slot.get_mut(part)
                        .and_then(Option::take)
                        .into_iter()
                        .collect()
                })
                .collect(),
            self.build_manager
                .with_schema(Arc::clone(&template.probe_schema)),
            level,
            self.metrics.clone(),
        );
        drop(probe_files);
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
        if let Some(range) = &self.range {
            join.set_dynamic_filter_partitions(range.clone())?;
        }
        let join: Arc<dyn ExecutionPlan> = Arc::new(join);
        inner.current = Some((run, Arc::clone(&join)));
        Ok(join)
    }
}

/// Per-plan-instance state of a join: the output partitions a split
/// attempt executes, and the streams that went away before a `CollectLeft`
/// build had spilled.
#[derive(Debug, Default)]
pub(crate) struct SpillShared {
    partitions: OnceLock<Range<usize>>,
    /// Probe partitions of the join, recorded at its first execution.
    probe_partitions: OnceLock<usize>,
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
    /// Records the output partitions this plan instance executes.
    pub(crate) fn bind_partitions(&self, partitions: Range<usize>) -> Result<()> {
        match self.partitions.get() {
            Some(existing) if *existing == partitions => Ok(()),
            Some(existing) => internal_err!(
                "conflicting hash join spill partitions {existing:?} and {partitions:?}"
            ),
            None => self.partitions.set(partitions).map_err(|_| {
                datafusion_common::internal_datafusion_err!(
                    "concurrent hash join spill partition binding"
                )
            }),
        }
    }

    fn partitions(&self) -> Option<Range<usize>> {
        self.partitions.get().cloned()
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
            let _ = fresh.partitions.set(partitions);
        }
        Arc::new(fresh)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Early> {
        self.early.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn register(&self, build: &Arc<SpilledBuild>) {
        let mut early = self.lock();
        build.barrier.departed_before_build(early.departed);
        early.departed = 0;
        early.build = Some(Arc::downgrade(build));
    }

    fn depart_early(&self) {
        let mut early = self.lock();
        let build = early.build.as_ref().and_then(Weak::upgrade);
        match build {
            Some(build) => {
                drop(early);
                build.barrier.depart(0);
            }
            None => early.departed += 1,
        }
    }
}

/// A stream's membership of a build's [`Barrier`].
struct Seat {
    build: Arc<SpilledBuild>,
    completed: usize,
}

impl Seat {
    fn complete(&mut self) {
        self.completed += 1;
        self.build.barrier.complete();
    }
}

impl Drop for Seat {
    fn drop(&mut self) {
        self.build.barrier.depart(self.completed);
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
    /// This stream's probe-file slot.
    slot: usize,
    /// The partition of the joins of written partitions this stream runs.
    exec_partition: usize,
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
                completed: 0,
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
            router
                .reservation
                .grow(get_record_batch_memory_size(&piece));
            let released = writer.append(piece, limit, &router.metrics, Side::Probe)?;
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
        seat.build.register_probe_files(self.slot, files)?;
        seat.complete();
        let state = Written {
            seat,
            output: Arc::clone(&self.output),
            context: Arc::clone(&self.context),
            partition: self.exec_partition,
            run: 0,
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
    output: Arc<UnmatchedOutput>,
    context: Arc<TaskContext>,
    partition: usize,
    run: usize,
    current: Option<SendableRecordBatchStream>,
    failed: bool,
}

impl Written {
    async fn next(&mut self) -> Option<Result<RecordBatch>> {
        if self.failed {
            return None;
        }
        loop {
            if let Some(stream) = &mut self.current {
                match stream.next().await {
                    Some(Ok(batch)) => return Some(Ok(batch)),
                    Some(Err(error)) => {
                        self.failed = true;
                        self.current = None;
                        return Some(Err(error));
                    }
                    None => {
                        self.current = None;
                        self.seat.complete();
                        self.run += 1;
                        continue;
                    }
                }
            }
            // Every member registered its probe files once step 1 started,
            // so the partitions to join are known from here on.
            self.seat.build.barrier.started(1).await;
            self.seat.build.release_in_memory();
            if self.run >= self.seat.build.runs().len() {
                return None;
            }
            self.seat.build.barrier.started(self.run + 1).await;
            let build = Arc::clone(&self.seat.build);
            let stream = if build.runs()[self.run] == build.partitions {
                Ok(build.unmatched_rows(Arc::clone(&self.output)))
            } else {
                build.sub_join(self.run).and_then(|join| {
                    join.execute(self.partition, Arc::clone(&self.context))
                })
            };
            match stream {
                Ok(stream) => self.current = Some(stream),
                Err(error) => {
                    self.failed = true;
                    return Some(Err(error));
                }
            }
        }
    }
}
