// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow family JS: the leaf that reads written partitions back.
//!
//! [`SpillFilesExec`] is the build or probe input of the stock
//! `HashJoinExec` that joins one written partition. It is never part of a
//! plan that is planned, serialized or displayed: only the join that wrote
//! the files creates it, at run time.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use arrow::array::{ArrayRef, BooleanArray, RecordBatch, UInt64Array};
use arrow::compute::filter_record_batch;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{assert_or_internal_err, internal_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr};
use futures::StreamExt;

use super::{JoinSpillMetrics, SpilledFile};
use crate::SendableRecordBatchStream as Rows;
use crate::execution_plan::{Boundedness, EmissionType, SchedulingType};
use crate::spill::spill_manager::SpillManager;
use crate::stream::RecordBatchStreamAdapter;
use crate::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    ReplaceChildrenOptions, SendableRecordBatchStream,
};

/// Where a [`SpillFilesExec`] partition finds its files.
pub(super) enum FileSource {
    /// One list of files per output partition, fixed at creation.
    Fixed(Vec<Vec<SpilledFile>>),
    /// One partition, executed once by each stream that shares the join:
    /// each execution reads the files that the stream staged for it.
    Staged(Arc<Stage>),
    /// One partition, the build of one chunk of a partition that cannot be
    /// divided: each execution continues where the chunk before stopped.
    Cursor(Arc<Cursor>),
}

impl FileSource {
    fn partitions(&self) -> usize {
        match self {
            Self::Fixed(files) => files.len(),
            Self::Staged(_) | Self::Cursor(_) => 1,
        }
    }

    fn take(&self, partition: usize) -> Result<Vec<SpilledFile>> {
        match self {
            Self::Fixed(files) => Ok(files[partition].clone()),
            Self::Staged(stage) => stage.take(),
            Self::Cursor(_) => {
                internal_err!("hash join spill: a cursor has no file list")
            }
        }
    }
}

/// The read position in the build file of a partition joined in chunks.
/// The chunk join's build stops at a refusal and gives back the batch it
/// could not hold; the next chunk starts with it.
pub(crate) struct Cursor {
    file: SpilledFile,
    spill_manager: SpillManager,
    state: Mutex<CursorState>,
}

#[derive(Default)]
struct CursorState {
    reader: Option<Rows>,
    /// Batches given back, read before the file.
    back: Vec<RecordBatch>,
    /// The file has been read to its end.
    ended: bool,
}

impl Cursor {
    pub(super) fn new(file: SpilledFile, spill_manager: SpillManager) -> Self {
        Self {
            file,
            spill_manager,
            state: Mutex::default(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CursorState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Gives `batch` back: the next chunk reads it first.
    pub(crate) fn give_back(&self, batch: RecordBatch) {
        self.lock().back.push(batch);
    }

    /// Every row of the file was taken by a chunk.
    pub(super) fn is_exhausted(&self) -> bool {
        let state = self.lock();
        state.ended && state.back.is_empty()
    }

    fn poll_next(&self, cx: &mut Context<'_>) -> Poll<Option<Result<RecordBatch>>> {
        let mut state = self.lock();
        if let Some(batch) = state.back.pop() {
            return Poll::Ready(Some(Ok(batch)));
        }
        if state.ended {
            return Poll::Ready(None);
        }
        if state.reader.is_none() {
            match self
                .spill_manager
                .read_spill_as_stream_unbuffered(Arc::clone(&self.file.file), None)
            {
                Ok(reader) => state.reader = Some(reader),
                Err(error) => return Poll::Ready(Some(Err(error))),
            }
        }
        let reader = state.reader.as_mut().expect("a reader");
        let next = reader.poll_next_unpin(cx);
        if let Poll::Ready(None) = next {
            state.ended = true;
            state.reader = None;
        }
        next
    }
}

/// Set by the join of a written partition that cannot be divided when its
/// build is refused before it produced a row: the partition is then joined
/// in build chunks.
#[derive(Debug, Default)]
pub(crate) struct ChunkSignal(AtomicBool);

impl ChunkSignal {
    pub(super) fn raise(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub(super) fn is_raised(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Probe rows that have found a match in an earlier chunk, by their row
/// number in the replay of a partition's probe files.
#[derive(Debug)]
pub(crate) struct MatchedBits {
    words: Mutex<Vec<u64>>,
}

impl MatchedBits {
    pub(super) fn new(rows: usize) -> Self {
        Self {
            words: Mutex::new(vec![0; rows.div_ceil(64)]),
        }
    }

    /// Bytes of the bits, for their charge.
    pub(super) fn size(rows: usize) -> usize {
        rows.div_ceil(64) * 8
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<u64>> {
        self.words.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Marks the rows numbered in `ids`.
    pub(super) fn set(&self, ids: &UInt64Array) -> Result<()> {
        let mut words = self.lock();
        for id in ids.values() {
            let index = (*id / 64) as usize;
            let Some(word) = words.get_mut(index) else {
                return internal_err!("hash join spill: probe row {id} out of range");
            };
            *word |= 1 << (id % 64);
        }
        Ok(())
    }

    /// For the rows `start..start + rows`: whether each has matched.
    pub(super) fn get(&self, start: usize, rows: usize) -> BooleanArray {
        let words = self.lock();
        (start..start + rows)
            .map(|row| Some(words[row / 64] & (1 << (row % 64)) != 0))
            .collect()
    }
}

/// How a probe leaf of a chunk join numbers its rows.
#[derive(Clone)]
pub(super) struct RowIds {
    /// Leave out the rows that matched in an earlier chunk.
    pub(super) skip_matched: Option<Arc<MatchedBits>>,
}

/// The name of the row-number column of a chunk join's probe leaf.
const ROW_ID: &str = "__hash_join_spill_probe_row";

/// `schema` with the row-number column appended.
pub(super) fn with_row_id(schema: &SchemaRef) -> SchemaRef {
    let mut fields = schema.fields().to_vec();
    fields.push(Arc::new(Field::new(ROW_ID, DataType::UInt64, false)));
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// Appends the row numbers `start..` to `batch`, then leaves out the rows
/// that `skip` marks.
fn number_rows(
    batch: RecordBatch,
    start: usize,
    schema: &SchemaRef,
    skip: Option<&MatchedBits>,
) -> Result<RecordBatch> {
    let rows = batch.num_rows();
    let mut columns = batch.columns().to_vec();
    columns.push(Arc::new(UInt64Array::from_iter_values(
        (start..start + rows).map(|row| row as u64),
    )) as ArrayRef);
    let numbered = RecordBatch::try_new(Arc::clone(schema), columns)?;
    match skip {
        Some(bits) => {
            let matched = bits.get(start, rows);
            let keep = arrow::compute::not(&matched)?;
            Ok(filter_record_batch(&numbered, &keep)?)
        }
        None => Ok(numbered),
    }
}

/// The files of the one execution of a shared join's probe leaf that is
/// being opened.
#[derive(Default)]
pub(super) struct Stage {
    /// Held while a stream opens its execution.
    opening: Mutex<()>,
    files: Mutex<Option<Vec<SpilledFile>>>,
}

impl Stage {
    /// Runs `execute` (the join's `execute(0)`, which executes the probe
    /// leaf at once) with `files` staged.
    pub(super) fn execute<T>(
        &self,
        files: Vec<SpilledFile>,
        execute: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let _opening = self.opening.lock().unwrap_or_else(|p| p.into_inner());
        *self.files.lock().unwrap_or_else(|p| p.into_inner()) = Some(files);
        let result = execute();
        self.files.lock().unwrap_or_else(|p| p.into_inner()).take();
        result
    }

    fn take(&self) -> Result<Vec<SpilledFile>> {
        match self.files.lock().unwrap_or_else(|p| p.into_inner()).take() {
            Some(files) => Ok(files),
            None => internal_err!("hash join spill: a shared probe leaf ran unstaged"),
        }
    }
}

/// What the rows of a build leaf went through: how many there are, and how
/// many the join that wrote them divided over its key partitions. A join
/// over them that cannot spill says why from these facts.
#[derive(Debug, Clone, Copy)]
pub(super) struct Origin {
    pub(super) rows: usize,
    pub(super) parent_rows: usize,
}

impl Origin {
    /// The partition kept more than half of its parent's rows: a uniform
    /// key spread keeps one in `hash_join_spill_partitions`, so one key (or
    /// a few) holds it.
    pub(super) fn hot(&self) -> bool {
        self.rows.saturating_mul(2) > self.parent_rows
    }
}

/// Reads spill files of one written partition, one list of files for each
/// output partition. Declares no ordering and is bounded, so a join over it
/// is eligible to spill again, one level deeper.
pub(crate) struct SpillFilesExec {
    source: FileSource,
    spill_manager: SpillManager,
    /// The spill level of the rows in these files: the level of the join
    /// that reads them.
    level: usize,
    origin: Origin,
    /// Metrics of the first-level join, which the joins of written
    /// partitions report into.
    metrics: JoinSpillMetrics,
    /// A build leaf: raised when its join must be run in chunks.
    signal: Option<Arc<ChunkSignal>>,
    /// A probe leaf of a chunk join: number the rows.
    row_ids: Option<RowIds>,
    cache: Arc<PlanProperties>,
}

impl SpillFilesExec {
    pub(super) fn new(
        schema: SchemaRef,
        source: FileSource,
        spill_manager: SpillManager,
        level: usize,
        origin: Origin,
        metrics: JoinSpillMetrics,
    ) -> Self {
        let cache = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(source.partitions()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        )
        .with_scheduling_type(SchedulingType::Cooperative);
        Self {
            source,
            spill_manager,
            level,
            origin,
            metrics,
            signal: None,
            row_ids: None,
            cache: Arc::new(cache),
        }
    }

    /// A build leaf whose join raises `signal` when it must run in chunks.
    pub(super) fn with_signal(mut self, signal: Arc<ChunkSignal>) -> Self {
        self.signal = Some(signal);
        self
    }

    /// A probe leaf that numbers its rows; its schema must be the files'
    /// schema with the row number appended ([`with_row_id`]).
    pub(super) fn with_row_ids(mut self, row_ids: RowIds) -> Self {
        self.row_ids = Some(row_ids);
        self
    }

    pub(super) fn signal(&self) -> Option<&Arc<ChunkSignal>> {
        self.signal.as_ref()
    }

    /// The cursor of a chunk join's build leaf.
    pub(super) fn cursor(&self) -> Option<&Arc<Cursor>> {
        match &self.source {
            FileSource::Cursor(cursor) => Some(cursor),
            _ => None,
        }
    }

    pub(super) fn level(&self) -> usize {
        self.level
    }

    pub(super) fn origin(&self) -> Origin {
        self.origin
    }

    pub(super) fn metrics(&self) -> &JoinSpillMetrics {
        &self.metrics
    }
}

impl fmt::Debug for SpillFilesExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillFilesExec")
            .field("partitions", &self.source.partitions())
            .field("level", &self.level)
            .finish()
    }
}

impl DisplayAs for SpillFilesExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "SpillFilesExec: level={}, partitions={}",
                    self.level,
                    self.source.partitions()
                )
            }
            DisplayFormatType::TreeRender => write!(f, "level={}", self.level),
        }
    }
}

impl ExecutionPlan for SpillFilesExec {
    fn name(&self) -> &'static str {
        "SpillFilesExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("SpillFilesExec has no children");
        }
        Ok(self)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("SpillFilesExec has no children");
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        assert_or_internal_err!(
            partition < self.source.partitions(),
            "SpillFilesExec invalid partition {partition} of {}",
            self.source.partitions()
        );
        let schema = self.schema();
        if let FileSource::Cursor(cursor) = &self.source {
            let cursor = Arc::clone(cursor);
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                schema,
                futures::stream::poll_fn(move |cx| cursor.poll_next(cx)),
            )));
        }
        let streams = self
            .source
            .take(partition)?
            .iter()
            // Unbuffered: the read runs in the consumer's poll. A buffered
            // read's producer task ends its stream as if the file ended when
            // the attempt stops it, and holds uncharged batches ahead.
            .map(|file| {
                self.spill_manager
                    .read_spill_as_stream_unbuffered(Arc::clone(&file.file), None)
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = futures::stream::iter(streams).flatten();
        let Some(row_ids) = self.row_ids.clone() else {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(schema, rows)));
        };
        let numbered = Arc::clone(&schema);
        let mut start = 0;
        let rows = rows.map(move |batch| {
            let batch = batch?;
            let first = start;
            start += batch.num_rows();
            number_rows(batch, first, &numbered, row_ids.skip_matched.as_deref())
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, rows)))
    }
}
