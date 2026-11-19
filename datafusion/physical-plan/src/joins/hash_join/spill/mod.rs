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
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::task::{Context, Poll};

use arrow::array::{
    ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array, UInt64Array,
};
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
use files::{FileSource, Origin, PartitionFileWriter, Stage, spill_file_error};

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
    /// The streams expected to probe the build (its members).
    count: Expected,
    /// The `CollectLeft` build that the exec holds for its streams.
    shared_build: Option<Weak<OnceAsync<JoinLeftData>>>,
    /// Above the depth limit: why the join writes no partition. It may
    /// still build its table on its floor.
    depth_limit: Option<String>,
}

/// Why a join above the depth limit cannot go on: a hot key when its
/// partition kept more than half of the rows its parent divided (a uniform
/// key spread keeps one in `hash_join_spill_partitions`), the pool otherwise.
fn depth_limit_reason(origin: Option<Origin>, level: usize, max_depth: usize) -> String {
    let rows = origin.map_or(0, |origin| origin.rows);
    match origin {
        Some(origin) if origin.hot() => format!(
            "one key value fills its partition of {rows} build rows ({}% of the {} \
             rows its parent join divided), which no spill level divides (the \
             build-chunk join of add-join-spill section 3 is not implemented) and which \
             is larger than its workspace floor; it is at spill level {level}, above \
             datafusion.execution.hash_join_spill_max_depth ({max_depth})",
            rows * 100 / origin.parent_rows.max(1),
            origin.parent_rows,
        ),
        _ => format!(
            "the memory pool had no room for its partition of {rows} build rows at \
             spill level {level}, above datafusion.execution.hash_join_spill_max_depth \
             ({max_depth}), and the partition is larger than its workspace floor"
        ),
    }
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
        let membership = shared.membership();
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
        } else if !matches!(
            join.mode,
            PartitionMode::CollectLeft | PartitionMode::Partitioned
        ) {
            Some(format!("its partition mode is {:?}", join.mode))
        } else if join.mode == PartitionMode::CollectLeft
            && membership != Membership::Closed
            && need_produce_result_in_final(join.join_type)
        {
            Some(
                "its probe streams are open (any number may execute a partition), so \
                 the build rows of a written partition could not be emitted exactly once"
                    .to_string(),
            )
        } else {
            None
        };
        if let Some(reason) = reason {
            return Ok(Self::Ineligible(reason));
        }
        // Above the depth limit the join may still build on its floor, but
        // it writes no partition (design decision 5, "Recursion").
        let depth_limit = (level > options.max_depth).then(|| {
            depth_limit_reason(leaf.map(SpillFilesExec::origin), level, options.max_depth)
        });
        let right_partitions = join.right().output_partitioning().partition_count();
        let count = match (join.mode, membership) {
            (PartitionMode::CollectLeft, Membership::Closed) => Expected::Fixed(
                shared
                    .partitions()
                    .map_or(right_partitions, |range| range.len()),
            ),
            (PartitionMode::CollectLeft, Membership::Open) => Expected::Open,
            (PartitionMode::CollectLeft, Membership::Admitted) => Expected::Admitted {
                admitted: 0,
                closed: false,
            },
            _ => Expected::Fixed(1),
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
            count,
            shared_build: (join.mode == PartitionMode::CollectLeft)
                .then(|| Arc::downgrade(shared_build)),
            depth_limit,
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

    /// Wraps the build input of an ineligible join and returns the function
    /// that adds why the join did not spill to a refusal of the join's own
    /// reservations. A refusal that the build input returned (an operator
    /// below the join, another join among them) is marked by the wrapper and
    /// keeps its text: the mark is a flag, not a match on the error text.
    pub(super) fn explain_refusal(
        &self,
        input: SendableRecordBatchStream,
    ) -> (
        SendableRecordBatchStream,
        impl Fn(DataFusionError) -> DataFusionError + Send + Sync + 'static + use<>,
    ) {
        let reason = match self {
            Self::Ineligible(reason) => Some(reason.clone()),
            _ => None,
        };
        let input_failed = Arc::new(AtomicBool::new(false));
        let input = match &reason {
            Some(_) => {
                let failed = Arc::clone(&input_failed);
                let schema = input.schema();
                Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    input.inspect(move |batch| {
                        if batch.is_err() {
                            failed.store(true, AtomicOrdering::Relaxed);
                        }
                    }),
                )) as SendableRecordBatchStream
            }
            None => input,
        };
        let explain = move |error: DataFusionError| match &reason {
            Some(reason)
                if is_refusal(&error) && !input_failed.load(AtomicOrdering::Relaxed) =>
            {
                error.context(format!("The hash join did not spill because {reason}"))
            }
            _ => error,
        };
        (input, explain)
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
            last: false,
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
    file: PartitionFileWriter,
    buffer: Vec<RecordBatch>,
    buffered_rows: usize,
    buffered_bytes: usize,
    rows: usize,
}

impl PartWriter {
    fn new(manager: &SpillManager, description: &str) -> Result<Self> {
        Ok(Self {
            file: PartitionFileWriter::new(manager, description)?,
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
        let schema = Arc::clone(self.file.schema());
        let (rows_metric, bytes_metric) = metrics.side(side);
        let before = metrics.spill.spilled_bytes.value();
        let mut start = 0;
        let written = self.file.write(std::iter::from_fn(|| {
            if start == batches.len() {
                return None;
            }
            let mut end = start;
            let mut rows = 0;
            while end < batches.len() && (end == start || rows < WRITE_BATCH_ROWS) {
                rows += batches[end].num_rows();
                end += 1;
            }
            let group = &batches[start..end];
            start = end;
            Some(if group.len() == 1 {
                Ok(group[0].clone())
            } else {
                concat_batches(&schema, group).map_err(Into::into)
            })
        }))?;
        self.rows += written;
        rows_metric.add(written);
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
    count: Expected,
    shared_build: Option<Weak<OnceAsync<JoinLeftData>>>,
    lazy_metrics: LazyMetrics,
    /// Above the depth limit: why no partition may be written.
    depth_limit: Option<String>,
    active: bool,
    /// One part per partition, then the unmatchable part.
    parts: Vec<Part>,
    /// Build rows each key partition received.
    rows_seen: Vec<usize>,
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
            count: eligible.count,
            shared_build: eligible.shared_build.clone(),
            lazy_metrics: eligible.metrics.for_partition(match &eligible.metrics {
                LazyMetrics::Own { partition, .. } => *partition,
                LazyMetrics::Inherited(_) => 0,
            }),
            depth_limit: eligible.depth_limit.clone(),
            active: false,
            parts,
            rows_seen: vec![0; eligible.options.partitions],
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
            // Above the depth limit nothing is written; `finish` fails with
            // the reason if the table is refused again.
            self.relieve(self.charged / 2, reservation)?;
        }
        Ok(())
    }

    /// A refusal that the join cannot answer, with the reason when it is
    /// above the depth limit.
    fn refused(&self, error: DataFusionError) -> DataFusionError {
        match &self.depth_limit {
            Some(reason) => {
                error.context(format!("The hash join did not spill because {reason}"))
            }
            None => error,
        }
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
        if let Some(seen) = self.rows_seen.get_mut(part) {
            *seen += piece.num_rows();
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
                        return Err(self.refused(error));
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
    /// before the workspace shrinks, so the pool never sees them free. The
    /// workspace is owed the bytes again: charge that a later write releases
    /// refills it before it goes back to the pool.
    fn borrow(&mut self, bytes: usize, reservation: &MemoryReservation) {
        reservation.grow(bytes);
        let moved = held(&take_held(&mut self.workspace, bytes));
        self.workspace_debt += moved;
    }

    /// Writes whole partitions, largest first, until the refused `need` and
    /// one eighth of what the parts held are released. Returns whether it
    /// wrote any.
    fn relieve(&mut self, need: usize, reservation: &MemoryReservation) -> Result<bool> {
        if self.depth_limit.is_some() {
            return Ok(false);
        }
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
                    let spilled = self.into_spilled(false);
                    return Ok((table, spilled, None));
                }
                Err(error) if is_refusal(&error) => {
                    // The rows are back in `batches`; `reservation` still
                    // holds them (and whatever the attempt grew). They were
                    // counted when they came.
                    let seen = self.rows_seen.clone();
                    for batch in std::mem::take(&mut batches) {
                        for (part, piece) in self.split(&batch)? {
                            self.add(part, piece)?;
                        }
                    }
                    self.rows_seen = seen;
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
                        return Err(self.refused(error));
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
        let floor = take_held(&mut self.workspace, extra);
        // The estimate of `fits_floor` can be short (an empty table, a key
        // copy of float keys, a membership form): the rest grows without
        // refusal, as the smallest unit of design decision 8.
        let short = extra.saturating_sub(held(&floor));
        self.charged = 0;
        // One charge for the table, released with it: the rows' charge
        // grows by the floor before the floor's pieces go.
        let charge = reservation.take();
        charge.grow(held(&floor) + short);
        drop(floor);
        let spilled = self.into_spilled(true);
        Ok((table, spilled, Some(charge)))
    }

    fn into_spilled(mut self, on_floor: bool) -> Option<Arc<SpilledBuild>> {
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
        let parent_rows = self.rows_seen.iter().sum();
        let build_files: Vec<Option<SpilledFile>> = self
            .parts
            .into_iter()
            .map(|part| match part {
                Part::Written(file) => file,
                _ => None,
            })
            .collect();
        let join_type = self.template.join_type;
        let mode = if need_produce_result_in_final(join_type) {
            Mode::Last
        } else {
            Mode::PerStream
        };
        let own_type = match join_type {
            JoinType::Left => Some(JoinType::Inner),
            JoinType::Full => Some(JoinType::Right),
            JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => None,
            other => Some(other),
        };
        let final_hold = build_files
            .iter()
            .map(|file| mode == Mode::Last && file.is_some())
            .collect();
        let seated = match (self.count, &self.shared) {
            (Expected::Fixed(_), Some(_)) => vec![false; slots],
            _ => vec![],
        };
        let spilled = Arc::new(SpilledBuild {
            level: self.level,
            shift: self.options.shift,
            partitions,
            written,
            parent_rows,
            build_files: Mutex::new(build_files),
            template: self.template,
            build_manager: manager,
            metrics: self.metrics,
            mode,
            own_type,
            probes: ProbeRegistry::new(slots),
            members: Mutex::new(Members {
                count: self.count,
                range,
                seated,
                arrived: 0,
                finished: 0,
                released: false,
                emitter: false,
                done: vec![0; partitions + 1],
                final_hold,
                joins: (0..partitions).map(|_| None).collect(),
                next_id: 0,
                violation: None,
            }),
            shared_build: self.shared_build.take(),
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
    /// Inner, Right, RightSemi, RightAnti, RightMark: every output row comes
    /// from a probe row. Each stream joins its own probe file of each written
    /// partition (an own run), sharing the partition's join with the streams
    /// that run it at the same time. No stream waits for another.
    PerStream,
    /// Left, LeftSemi, LeftAnti, LeftMark, Full: the join also emits build
    /// rows. A Left or Full stream makes own runs with the probe-preserving
    /// form of the join (Inner, Right), so a stream gives the rows its probe
    /// rows produce whatever the other streams do, as in the stock join. The
    /// stream that the in-memory table's completion count chooses to emit the
    /// build rows (the stock join's last probe partition) also emits those of
    /// every written partition, with a join over every stream's probe file of
    /// it (LeftAnti for Left and Full, the join type itself otherwise). With
    /// no such stream (a stream stopped early or never ran) no build row is
    /// emitted, as in the stock join.
    Last,
}

/// One partition that a stream joins after its in-memory pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Run {
    /// Its own probe file of a written partition.
    Own(usize),
    /// `Last`: the build rows of a written partition, or the unmatchable
    /// partition (index `partitions`).
    Final(usize),
}

/// `Last`: the probe files of a build's written partitions, by stream slot
/// and partition, for the build-row pass. A stream registers them when its
/// probe input ends, before it reports its completion to the in-memory
/// table, so the stream that the count chooses sees every file.
pub(crate) struct ProbeRegistry(Mutex<Vec<Option<Vec<Option<SpilledFile>>>>>);

impl ProbeRegistry {
    fn new(slots: usize) -> Self {
        Self(Mutex::new(vec![None; slots]))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Option<Vec<Option<SpilledFile>>>>> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn register(&self, slot: usize, files: Vec<Option<SpilledFile>>) -> Result<()> {
        match self.lock().get_mut(slot) {
            Some(entry @ None) => {
                *entry = Some(files);
                Ok(())
            }
            Some(Some(_)) => {
                internal_err!("hash join spill: probe slot {slot} registered twice")
            }
            None => internal_err!("hash join spill: no probe slot {slot}"),
        }
    }

    fn take_all(&self, part: usize) -> Vec<SpilledFile> {
        self.lock()
            .iter_mut()
            .flatten()
            .filter_map(|files| files.get_mut(part).and_then(Option::take))
            .collect()
    }

    fn clear(&self) {
        self.lock().iter_mut().for_each(|files| *files = None);
    }
}

/// The streams expected to probe a build (its members).
#[derive(Debug, Clone, Copy)]
pub(super) enum Expected {
    /// Each member probes one partition once: a `CollectLeft` join's bound
    /// range or every probe partition, or the one stream of a `Partitioned`
    /// build.
    Fixed(usize),
    /// The join of a written partition that its parent shares: the parent
    /// admits each stream before it executes partition 0. Closed, so that
    /// later streams make a new join, once every admitted stream ended its
    /// in-memory pass.
    Admitted { admitted: usize, closed: bool },
    /// Any number of streams execute any partition, each with its own probe
    /// input (`HashJoinExec::set_open_probe_streams`); their count is not
    /// known, so nothing is released before the join is dropped.
    Open,
}

impl Expected {
    /// The number of members, once it is known.
    fn known(&self) -> Option<usize> {
        match self {
            Self::Fixed(members) => Some(*members),
            Self::Admitted {
                admitted,
                closed: true,
            } => Some(*admitted),
            _ => None,
        }
    }
}

/// What the members of a build have done. Nothing here waits.
struct Members {
    count: Expected,
    /// `Fixed` `CollectLeft`: the bound partitions, and the partitions that
    /// came.
    range: Option<Range<usize>>,
    seated: Vec<bool>,
    /// Members that came: took a seat or left before.
    arrived: usize,
    /// Members that ended their in-memory pass or left.
    finished: usize,
    /// The exec let go of the in-memory table.
    released: bool,
    /// `Last`: a stream was chosen to emit the build rows.
    emitter: bool,
    /// Per partition (then the unmatchable one): members done with their
    /// own run of it, finished, skipped or left.
    done: Vec<usize>,
    /// `Last`: the build-row pass still needs the partition's build file.
    final_hold: Vec<bool>,
    /// The join of each partition while streams run it.
    joins: Vec<Option<SharedJoin>>,
    next_id: u64,
    /// An impossible state found where no error can be returned (a drop);
    /// the next stream that asks fails with it.
    violation: Option<String>,
}

/// A partition's join that streams share while they run it.
struct SharedJoin {
    join: Arc<HashJoinExec>,
    stage: Arc<Stage>,
    users: usize,
    id: u64,
}

/// The written partitions of one build, shared by the streams that probe
/// it (all of them in `CollectLeft` mode).
pub(crate) struct SpilledBuild {
    level: usize,
    shift: u32,
    partitions: usize,
    written: Vec<bool>,
    /// The build rows this build divided over its key partitions.
    parent_rows: usize,
    /// One file per partition, then the unmatchable partition's.
    build_files: Mutex<Vec<Option<SpilledFile>>>,
    template: SubJoinTemplate,
    build_manager: SpillManager,
    metrics: JoinSpillMetrics,
    mode: Mode,
    /// The join type of own runs; none for LeftSemi, LeftAnti and LeftMark,
    /// whose rows are all build rows.
    own_type: Option<JoinType>,
    probes: ProbeRegistry,
    members: Mutex<Members>,
    /// The exec's hold on a `CollectLeft` build, released once every
    /// member ended its in-memory pass.
    shared_build: Option<Weak<OnceAsync<JoinLeftData>>>,
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

    fn written_parts(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.partitions).filter(|part| self.written[*part])
    }

    /// Counts a member that came as stream `slot`. The states that the
    /// count cannot explain are internal errors (design decision 5).
    fn arrive(members: &mut Members, slot: usize) -> Result<()> {
        members.arrived += 1;
        match members.count {
            Expected::Fixed(expected) => {
                if let Some(range) = &members.range
                    && !range.contains(&slot)
                {
                    return internal_err!(
                        "hash join spill: a stream of partition {slot} probes a build \
                         bound to partitions {range:?}"
                    );
                }
                if let Some(seated) = members.seated.get_mut(slot) {
                    if *seated {
                        return internal_err!(
                            "hash join spill: partition {slot} probes the build twice"
                        );
                    }
                    *seated = true;
                }
                if members.arrived > expected {
                    return internal_err!(
                        "hash join spill: {} streams probe a build of {expected} members",
                        members.arrived
                    );
                }
            }
            Expected::Admitted { admitted, .. } if members.arrived > admitted => {
                return internal_err!(
                    "hash join spill: {} streams probe a partition join that admitted \
                     {admitted}",
                    members.arrived
                );
            }
            _ => {}
        }
        Ok(())
    }

    /// Stream `slot` takes its seat when its build is ready.
    fn seat(&self, slot: usize) -> Result<()> {
        let mut members = self.lock();
        if let Some(violation) = &members.violation {
            return internal_err!("{violation}");
        }
        Self::arrive(&mut members, slot)
    }

    /// Stream `slot` left before it took its seat.
    fn depart(&self, slot: usize) {
        let mut members = self.lock();
        if let Err(error) = Self::arrive(&mut members, slot) {
            members.violation.get_or_insert_with(|| error.to_string());
        }
        self.leave_locked(&mut members);
    }

    /// A member left before its in-memory pass ended.
    fn leave_in_memory(&self) {
        let mut members = self.lock();
        self.leave_locked(&mut members);
    }

    fn leave_locked(&self, members: &mut Members) {
        members.finished += 1;
        for part in self.written_parts() {
            members.done[part] += 1;
        }
        self.settle(members);
    }

    /// The parent admits a stream to this partition join; false once it is
    /// closed.
    fn admit(&self) -> bool {
        match &mut self.lock().count {
            Expected::Admitted {
                admitted,
                closed: false,
            } => {
                *admitted += 1;
                true
            }
            _ => false,
        }
    }

    /// A member ended its in-memory pass; `emit` if the in-memory table's
    /// completion count chose it to emit the build rows.
    fn end_in_memory(&self, emit: bool) -> Result<()> {
        let mut members = self.lock();
        if let Some(violation) = &members.violation {
            return internal_err!("{violation}");
        }
        members.finished += 1;
        if let Some(expected) = match members.count {
            Expected::Fixed(expected) => Some(expected),
            Expected::Admitted { admitted, .. } => Some(admitted),
            Expected::Open => None,
        } && members.finished > expected
        {
            return internal_err!(
                "hash join spill: {} streams ended their probe of a build of {expected} \
                 members",
                members.finished
            );
        }
        if emit && self.mode == Mode::Last {
            if members.emitter {
                return internal_err!(
                    "hash join spill: two streams were chosen to emit the build rows"
                );
            }
            members.emitter = true;
        }
        self.settle(&mut members);
        Ok(())
    }

    /// After a member finished: lets go of the in-memory table once every
    /// member ended its in-memory pass, gives up the build-row pass when no
    /// member was chosen for it, and deletes the files no member needs.
    fn settle(&self, members: &mut Members) {
        let all = match &mut members.count {
            Expected::Fixed(expected) => members.finished >= *expected,
            Expected::Admitted { admitted, closed } => {
                if members.finished >= *admitted {
                    *closed = true;
                }
                *closed
            }
            Expected::Open => false,
        };
        if all && !members.released {
            members.released = true;
            // The exec still holds the `CollectLeft` build for streams that
            // never came. Such a stream fails instead of building it again.
            if let Some(shared_build) = self.shared_build.as_ref().and_then(Weak::upgrade)
            {
                shared_build.release(datafusion_common::internal_datafusion_err!(
                    "hash join spill: a stream started after every member ended its probe"
                ));
            }
        }
        if all && self.mode == Mode::Last && !members.emitter {
            members.final_hold.iter_mut().for_each(|hold| *hold = false);
            self.probes.clear();
        }
        for part in 0..=self.partitions {
            self.prune(members, part);
        }
    }

    /// Deletes `part`'s build file when no member and no build-row pass
    /// needs it.
    fn prune(&self, members: &Members, part: usize) {
        if members.final_hold[part] {
            return;
        }
        let own_done = self.own_type.is_none()
            || part == self.partitions
            || members
                .count
                .known()
                .is_some_and(|expected| members.done[part] >= expected);
        if own_done {
            self.files()[part] = None;
        }
    }

    /// A member is done with its own run of `part`.
    fn done_with(&self, part: usize) {
        let mut members = self.lock();
        members.done[part] += 1;
        self.prune(&members, part);
    }

    /// The runs of a member with probe files `files`; `emit` adds the
    /// build-row pass.
    fn runs(&self, files: &[Option<SpilledFile>], emit: bool) -> Vec<Run> {
        let mut runs = vec![];
        if self.own_type.is_some() {
            for part in self.written_parts() {
                if files[part].as_ref().is_some_and(|file| file.rows > 0) {
                    runs.push(Run::Own(part));
                } else {
                    self.done_with(part);
                }
            }
        }
        if emit && self.mode == Mode::Last {
            let members = self.lock();
            runs.extend(
                (0..=self.partitions)
                    .filter(|part| members.final_hold[*part])
                    .map(Run::Final),
            );
        }
        runs
    }

    /// The join of `part` for an own run, shared with the streams that run
    /// it now, or a new one over the build file.
    fn shared_join(self: &Arc<Self>, part: usize) -> Result<JoinUse> {
        let mut members = self.lock();
        if let Some(shared) = &mut members.joins[part]
            && shared.join.spill_shared().admit()
        {
            shared.users += 1;
            return Ok(JoinUse {
                build: Arc::clone(self),
                part,
                id: shared.id,
                join: Arc::clone(&shared.join),
                stage: Arc::clone(&shared.stage),
            });
        }
        let Some(build_file) = self.files()[part].clone() else {
            return internal_err!(
                "hash join spill: the build file of written partition {part} is gone \
                 while a member still joins it"
            );
        };
        let stage = Arc::new(Stage::default());
        let join_type = self.own_type.unwrap_or(self.template.join_type);
        let join = self.join_over(
            build_file,
            FileSource::Staged(Arc::clone(&stage)),
            join_type,
            self.template.projection.clone(),
        )?;
        join.spill_shared().set_membership(Membership::Admitted);
        if !join.spill_shared().admit() {
            return internal_err!(
                "hash join spill: a new partition join refused a stream"
            );
        }
        let id = members.next_id;
        members.next_id += 1;
        members.joins[part] = Some(SharedJoin {
            join: Arc::clone(&join),
            stage: Arc::clone(&stage),
            users: 1,
            id,
        });
        Ok(JoinUse {
            build: Arc::clone(self),
            part,
            id,
            join,
            stage,
        })
    }

    /// Of the runs `pending`, the index of an own run whose join another
    /// stream runs now, so that streams share tables rather than build
    /// several at once; otherwise the first.
    fn next_run(&self, pending: &[Run]) -> usize {
        let members = self.lock();
        pending
            .iter()
            .position(
                |run| matches!(run, Run::Own(part) if members.joins[*part].is_some()),
            )
            .unwrap_or(0)
    }

    /// A stream stopped running the join `id` of `part`.
    fn release_join(&self, part: usize, id: u64) {
        let mut members = self.lock();
        if let Some(shared) = &mut members.joins[part]
            && shared.id == id
        {
            shared.users -= 1;
            if shared.users == 0 {
                members.joins[part] = None;
            }
        }
    }

    /// `Last`: the build-row pass of `part` takes its build file.
    fn take_final(&self, part: usize) -> Result<Option<SpilledFile>> {
        let mut members = self.lock();
        if !members.final_hold[part] {
            return internal_err!(
                "hash join spill: the build-row pass of partition {part} ran twice"
            );
        }
        let file = self.files()[part].clone();
        members.final_hold[part] = false;
        self.prune(&members, part);
        Ok(file)
    }

    /// `Last`: the stream chosen for the build-row pass left before `part`.
    fn drop_final(&self, part: usize) {
        let mut members = self.lock();
        members.final_hold[part] = false;
        self.prune(&members, part);
    }

    /// `Last`: the join that emits the build rows of `part`, over every
    /// member's probe file. Its output is in the build schema when the rows
    /// must go through `UnmatchedOutput` (Left and Full).
    fn final_join(&self, part: usize) -> Result<(Arc<HashJoinExec>, bool)> {
        let Some(build_file) = self.take_final(part)? else {
            return internal_err!(
                "hash join spill: the build file of written partition {part} is gone \
                 before its build rows were emitted"
            );
        };
        let probe = FileSource::Fixed(vec![self.probes.take_all(part)]);
        let (join_type, projection, remap) = match self.template.join_type {
            JoinType::Left | JoinType::Full => (JoinType::LeftAnti, None, true),
            other => (other, self.template.projection.clone(), false),
        };
        let join = self.join_over(build_file, probe, join_type, projection)?;
        Ok((join, remap))
    }

    /// A stock `CollectLeft` join of one written partition, one level
    /// deeper.
    fn join_over(
        &self,
        build_file: SpilledFile,
        probe: FileSource,
        join_type: JoinType,
        projection: Option<ProjectionRef>,
    ) -> Result<Arc<HashJoinExec>> {
        let level = self.level + 1;
        let template = &self.template;
        let origin = Origin {
            rows: build_file.rows,
            parent_rows: self.parent_rows,
        };
        let build = SpillFilesExec::new(
            Arc::clone(&template.build_schema),
            FileSource::Fixed(vec![vec![build_file]]),
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
            join_type,
        )
        .with_filter(template.filter.clone())
        .with_projection_ref(projection)
        .with_partition_mode(PartitionMode::CollectLeft)
        .with_null_equality(template.null_equality)
        .build()?;
        Ok(Arc::new(join))
    }

    /// `Last`: the rows of the unmatchable partition, each an unmatched
    /// build row.
    fn unmatched_rows(
        &self,
        output: Arc<UnmatchedOutput>,
    ) -> Result<SendableRecordBatchStream> {
        let file = self.take_final(self.partitions)?;
        let schema = Arc::clone(&output.schema);
        let rows = match file {
            Some(file) => self
                .build_manager
                .read_spill_as_stream_unbuffered(Arc::clone(&file.file), None)
                .map_err(spill_file_error)?
                .map(move |batch| {
                    // The file lives until its rows are read.
                    let _file = &file;
                    batch
                        .map_err(spill_file_error)
                        .and_then(|batch| output.emit(&batch))
                })
                .boxed(),
            None => futures::stream::empty().boxed(),
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, rows)))
    }
}

/// A stream's use of a shared partition join; dropping it lets the join go
/// when no other stream runs it.
struct JoinUse {
    build: Arc<SpilledBuild>,
    part: usize,
    id: u64,
    join: Arc<HashJoinExec>,
    stage: Arc<Stage>,
}

impl JoinUse {
    /// Executes the join's one partition over `files`, this stream's probe
    /// file of the partition.
    fn execute(
        &self,
        files: Vec<SpilledFile>,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.stage.execute(files, || self.join.execute(0, context))
    }
}

impl Drop for JoinUse {
    fn drop(&mut self) {
        self.build.release_join(self.part, self.id);
    }
}

/// Who probes a `CollectLeft` plan instance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Membership {
    /// Each probe partition of the bound range (or every partition) is
    /// executed once.
    #[default]
    Closed,
    /// Any number of streams execute any partition, each with its own probe
    /// input (Longbow's native hand-off).
    Open,
    /// A written partition's join that its parent shares (see
    /// [`Expected::Admitted`]).
    Admitted,
}

/// Per-plan-instance state of a join: the output partitions a split
/// attempt executes, who probes it, and the streams that went away before a
/// `CollectLeft` build had spilled.
#[derive(Debug, Default)]
pub(crate) struct SpillShared {
    partitions: Mutex<Option<Range<usize>>>,
    membership: Mutex<Membership>,
    /// Probe partitions of the join, recorded at its first execution.
    probe_partitions: OnceLock<usize>,
    /// The join's `spill_depth` gauge, registered at its first relief.
    depth: OnceLock<Gauge>,
    early: Mutex<Early>,
}

#[derive(Default)]
struct Early {
    build: Option<Weak<SpilledBuild>>,
    registered: bool,
    /// Partitions whose streams left before the build spilled.
    departed: Vec<usize>,
    /// `Admitted`: streams admitted before the build spilled.
    admitted: usize,
}

impl std::fmt::Debug for Early {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Early")
            .field("departed", &self.departed)
            .field("admitted", &self.admitted)
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

    pub(crate) fn set_membership(&self, membership: Membership) {
        *self.membership.lock().unwrap_or_else(|p| p.into_inner()) = membership;
    }

    fn membership(&self) -> Membership {
        *self.membership.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(super) fn record_probe_partitions(&self, partitions: usize) {
        let _ = self.probe_partitions.set(partitions);
    }

    fn probe_partitions(&self) -> usize {
        self.probe_partitions.get().copied().unwrap_or(1)
    }

    /// Fresh state for `reset_state`, keeping the bound partitions and the
    /// membership.
    pub(crate) fn reset(&self) -> Arc<Self> {
        let fresh = Self::default();
        if let Some(partitions) = self.partitions() {
            fresh.bind_partitions(partitions);
        }
        fresh.set_membership(self.membership());
        Arc::new(fresh)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Early> {
        self.early.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn register(&self, build: &Arc<SpilledBuild>) {
        let mut early = self.lock();
        if let Expected::Admitted { admitted, .. } = &mut build.lock().count {
            *admitted = early.admitted;
        }
        for slot in std::mem::take(&mut early.departed) {
            build.depart(slot);
        }
        early.build = Some(Arc::downgrade(build));
        early.registered = true;
    }

    /// A stream of partition `slot` left before its build was ready.
    fn depart_early(&self, slot: usize) {
        let mut early = self.lock();
        if !early.registered {
            early.departed.push(slot);
            return;
        }
        let build = early.build.as_ref().and_then(Weak::upgrade);
        drop(early);
        if let Some(build) = build {
            build.depart(slot);
        }
    }

    /// `Admitted`: admits one more stream; false once the join is closed.
    fn admit(&self) -> bool {
        let mut early = self.lock();
        if !early.registered {
            early.admitted += 1;
            return true;
        }
        early
            .build
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some_and(|build| build.admit())
    }
}

/// A stream's membership of a spilled build.
struct Seat {
    build: Arc<SpilledBuild>,
    state: SeatState,
}

enum SeatState {
    InMemory,
    /// The runs this stream still has to finish.
    Written(Vec<Run>),
}

impl Drop for Seat {
    fn drop(&mut self) {
        match &self.state {
            SeatState::InMemory => self.build.leave_in_memory(),
            SeatState::Written(pending) => {
                for run in pending {
                    match run {
                        Run::Own(part) => self.build.done_with(*part),
                        Run::Final(part) => self.build.drop_final(*part),
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
    /// The probe input ended: this stream's probe files are finished.
    Probed(Box<Probed>),
    /// Joining the written partitions.
    Written(BoxStream<'static, Result<RecordBatch>>),
    Done,
}

struct Probed {
    seat: Seat,
    files: Vec<Option<SpilledFile>>,
}

/// The probe side of one output stream of a spilling join.
pub(super) struct StreamSpill {
    context: Arc<TaskContext>,
    /// This stream's partition (0 in `Partitioned` mode): its seat, and its
    /// probe-file slot for the build-row pass.
    slot: usize,
    shared: Option<Arc<SpillShared>>,
    metrics: LazyMetrics,
    schema: SchemaRef,
    options: SpillOptions,
    null_equality: NullEquality,
    output: Arc<UnmatchedOutput>,
    phase: Phase,
    /// The in-memory table's completion count chose this stream to emit the
    /// build rows.
    last: bool,
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

    /// An own run's batch under the join's schema: the probe-preserving
    /// form (Inner, Right) declares some columns non-nullable that the join
    /// (Left, Full) declares nullable.
    fn own(&self, batch: RecordBatch) -> Result<RecordBatch> {
        if batch.schema() == self.schema {
            return Ok(batch);
        }
        Ok(RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            batch.columns().to_vec(),
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )?)
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
    pub(super) fn on_build(
        &mut self,
        spilled: Option<&Arc<SpilledBuild>>,
    ) -> Result<bool> {
        let Some(build) = spilled else {
            self.depart();
            return Ok(false);
        };
        // From here on the seat, not the early count, records a departure.
        self.shared = None;
        self.phase = Phase::Done;
        build.seat(self.slot)?;
        let seat = Seat {
            build: Arc::clone(build),
            state: SeatState::InMemory,
        };
        let metrics = self.metrics.get().clone();
        let reservation =
            MemoryConsumer::new(format!("HashJoinProbeSpill[{}]", self.slot))
                .with_can_spill(true)
                .register(self.context.memory_pool());
        self.phase = Phase::Routing(Box::new(Router {
            seat,
            manager: spill_manager(&self.context, &metrics, Arc::clone(&self.schema)),
            writers: (0..build.partitions).map(|_| None).collect(),
            metrics,
            reservation,
            hashes: vec![],
        }));
        Ok(true)
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

    /// The probe input ended: finishes this stream's probe files and, for
    /// the build-row pass, registers them. Runs before the stream reports
    /// its completion to the in-memory table.
    pub(super) fn end_probe(&mut self) -> Result<()> {
        let Phase::Routing(_) = &self.phase else {
            return Ok(());
        };
        let Phase::Routing(router) = std::mem::replace(&mut self.phase, Phase::Done)
        else {
            unreachable!()
        };
        let Router {
            seat,
            metrics,
            writers,
            reservation,
            ..
        } = *router;
        let files: Vec<Option<SpilledFile>> = writers
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
        if seat.build.mode == Mode::Last {
            seat.build.probes.register(self.slot, files.clone())?;
        }
        self.phase = Phase::Probed(Box::new(Probed { seat, files }));
        Ok(())
    }

    /// The in-memory table's completion count chose (or did not choose)
    /// this stream to emit the build rows.
    pub(super) fn set_last(&mut self, last: bool) {
        self.last = last;
    }

    /// Hook 3: after the in-memory pass, joins the written partitions. The
    /// caller has dropped its reference to the in-memory table.
    pub(super) fn poll_written(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        if let Err(error) = self.end_probe() {
            return Poll::Ready(Some(Err(error)));
        }
        if let Phase::Probed(_) = &self.phase {
            let Phase::Probed(probed) = std::mem::replace(&mut self.phase, Phase::Done)
            else {
                unreachable!()
            };
            match self.start_written(*probed) {
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
        probed: Probed,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let Probed { mut seat, files } = probed;
        seat.build.end_in_memory(self.last)?;
        let runs = seat.build.runs(&files, self.last);
        seat.state = SeatState::Written(runs.clone());
        let state = Written {
            seat,
            runs,
            files,
            output: Arc::clone(&self.output),
            context: Arc::clone(&self.context),
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
            shared.depart_early(self.slot);
        }
        self.phase = Phase::Done;
    }
}

impl Drop for StreamSpill {
    fn drop(&mut self) {
        self.depart();
    }
}

/// How a run's output becomes the join's output.
enum Output {
    /// An own run: the join's schema.
    Own,
    /// The build-row pass of a Left or Full join: LeftAnti rows, each an
    /// unmatched build row.
    Unmatched,
    /// Already the join's output.
    Same,
}

/// The written-partition phase of one stream.
struct Written {
    seat: Seat,
    /// The runs of this stream, in order; the first is current.
    runs: Vec<Run>,
    /// This stream's probe files, by partition.
    files: Vec<Option<SpilledFile>>,
    output: Arc<UnmatchedOutput>,
    context: Arc<TaskContext>,
    current: Option<(SendableRecordBatchStream, Option<JoinUse>, Output)>,
    failed: bool,
}

impl Written {
    async fn next(&mut self) -> Option<Result<RecordBatch>> {
        if self.failed {
            return None;
        }
        loop {
            if let Some((stream, _, output)) = &mut self.current {
                match stream.next().await {
                    Some(Ok(batch)) => {
                        let batch = match output {
                            Output::Own => self.output.own(batch),
                            Output::Unmatched => self.output.emit(&batch),
                            Output::Same => Ok(batch),
                        };
                        if batch.is_err() {
                            self.failed = true;
                            self.current = None;
                        }
                        return Some(batch);
                    }
                    Some(Err(error)) => {
                        self.failed = true;
                        self.current = None;
                        return Some(Err(error));
                    }
                    None => {
                        self.current = None;
                        let run = self.runs.remove(0);
                        self.seat.state = SeatState::Written(self.runs.clone());
                        if let Run::Own(part) = run {
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
            // The run to start moves to the front: `runs[0]` is current.
            let next = build.next_run(&self.runs);
            self.runs.swap(0, next);
            let started = match self.runs[0] {
                Run::Own(part) => build.shared_join(part).and_then(|using| {
                    let files = self.files[part].take().into_iter().collect();
                    let stream = using.execute(files, Arc::clone(&self.context))?;
                    Ok((stream, Some(using), Output::Own))
                }),
                Run::Final(part) if part == build.partitions => build
                    .unmatched_rows(Arc::clone(&self.output))
                    .map(|stream| (stream, None, Output::Same)),
                Run::Final(part) => build.final_join(part).and_then(|(join, remap)| {
                    let stream = join.execute(0, Arc::clone(&self.context))?;
                    let output = if remap {
                        Output::Unmatched
                    } else {
                        Output::Same
                    };
                    Ok((stream, None, output))
                }),
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
