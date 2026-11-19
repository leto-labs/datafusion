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
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use arrow::array::{ArrayRef, BooleanArray, RecordBatch, UInt64Array};
use arrow::compute::filter_record_batch;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{
    DataFusionError, Result, exec_datafusion_err, resources_datafusion_err,
};
use datafusion_common::{assert_or_internal_err, internal_err};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_execution::spill_file::SpillFile;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr};
use futures::future::BoxFuture;
use futures::{StreamExt, TryStreamExt};
use tokio::sync::{Semaphore, SemaphorePermit};

use super::{JoinSpillMetrics, SpilledFile};
use crate::SendableRecordBatchStream as Rows;
use crate::execution_plan::{Boundedness, EmissionType, SchedulingType};
use crate::spill::in_progress_spill_file::InProgressSpillFile;
use crate::spill::spill_manager::SpillManager;
use crate::spill::{IPCStreamEncoder, gc_view_arrays};
use crate::stream::RecordBatchStreamAdapter;
use crate::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    ReplaceChildrenOptions, SendableRecordBatchStream,
};

/// A flush on a reopenable backend holds one handle only while it writes.
/// Callers admit each synchronous spill operation with a short-write permit
/// before creating, writing or finishing partition files.
pub(super) struct PartitionFileWriter {
    manager: Arc<SpillManager>,
    file: Arc<dyn SpillFile>,
    writer: PartitionWriter,
}

enum PartitionWriter {
    Reopenable(Option<IPCStreamEncoder>),
    Persistent(Option<(InProgressSpillFile, PersistentPermit)>),
}

const MAX_OPEN_JOIN_SPILL_WRITERS: usize = 128;
pub(super) static SPILL_WRITER_SLOTS: Semaphore =
    Semaphore::const_new(MAX_OPEN_JOIN_SPILL_WRITERS);
// Leave at least one slot in the shared budget for short operations. A
// persistent writer can need to reach probe EOF before it releases its slot.
const MAX_PERSISTENT_WRITERS: usize = MAX_OPEN_JOIN_SPILL_WRITERS - 1;
static PERSISTENT_SPILL_WRITERS: Semaphore = Semaphore::const_new(MAX_PERSISTENT_WRITERS);

struct PersistentPermit {
    _total: SemaphorePermit<'static>,
    _persistent: SemaphorePermit<'static>,
}

fn persistent_writer_permit() -> Result<PersistentPermit> {
    let exhausted = |_| {
        exec_datafusion_err!(
            "hash join spill could not admit a custom backend's persistent writer (capacity {MAX_PERSISTENT_WRITERS} of {MAX_OPEN_JOIN_SPILL_WRITERS} slots)"
        )
    };
    let persistent = PERSISTENT_SPILL_WRITERS.try_acquire().map_err(exhausted)?;
    let total = SPILL_WRITER_SLOTS.try_acquire().map_err(exhausted)?;
    Ok(PersistentPermit {
        _total: total,
        _persistent: persistent,
    })
}

pub(super) async fn short_write_permit(
    slots: &'static Semaphore,
) -> SemaphorePermit<'static> {
    slots
        .acquire()
        .await
        .expect("spill writer semaphore is never closed")
}

/// Keeps a queued acquisition across stream polls. The uncontended path
/// acquires directly, with no allocation.
#[derive(Default)]
pub(super) struct PollShortWrite(Option<BoxFuture<'static, SemaphorePermit<'static>>>);

impl PollShortWrite {
    pub(super) fn poll(
        &mut self,
        slots: &'static Semaphore,
        cx: &mut Context<'_>,
    ) -> Poll<SemaphorePermit<'static>> {
        if self.0.is_none() {
            if let Ok(permit) = slots.try_acquire() {
                return Poll::Ready(permit);
            }
            self.0 = Some(Box::pin(short_write_permit(slots)));
        }
        let permit = futures::ready!(self.0.as_mut().unwrap().as_mut().poll(cx));
        self.0 = None;
        Poll::Ready(permit)
    }
}

pub(super) fn spill_file_error(error: DataFusionError) -> DataFusionError {
    classify_spill_file_error(error, false)
}

fn classify_spill_file_error(
    error: DataFusionError,
    local_tempfile: bool,
) -> DataFusionError {
    #[cfg(not(unix))]
    let _ = local_tempfile;
    #[cfg(unix)]
    {
        let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&error);
        while let Some(cause) = source {
            if let Some(io) = cause.downcast_ref::<std::io::Error>() {
                // tempfile adds the path to the error text and exposes the
                // original error's source, losing the OS code in the chain.
                let code = io.raw_os_error().or_else(|| {
                    if !local_tempfile {
                        return None;
                    }
                    let message = io.to_string();
                    [libc::EMFILE, libc::ENFILE].into_iter().find(|code| {
                        let original = std::io::Error::from_raw_os_error(*code);
                        io.kind() == original.kind()
                            && message.starts_with(&original.to_string())
                    })
                });
                match code {
                    Some(libc::EMFILE) => {
                        return resources_datafusion_err!(
                            "hash join spill exhausted the process open-file limit (RLIMIT_NOFILE)"
                        );
                    }
                    Some(libc::ENFILE) => {
                        return resources_datafusion_err!(
                            "hash join spill exhausted the system-wide open-file limit"
                        );
                    }
                    _ => {}
                }
            }
            source = cause.source();
        }
    }
    error
}

impl PartitionFileWriter {
    pub(super) fn new(manager: &SpillManager, description: &str) -> Result<Self> {
        let disk = &manager.env().disk_manager;
        let file = disk.create_tmp_file(description).map_err(|error| {
            classify_spill_file_error(error, !disk.has_custom_temp_file_factory())
        })?;
        let writer = if file.supports_reopen_for_append() {
            PartitionWriter::Reopenable(None)
        } else {
            PartitionWriter::Persistent(None)
        };
        Ok(Self {
            manager: Arc::new(manager.clone()),
            file,
            writer,
        })
    }

    pub(super) fn schema(&self) -> &SchemaRef {
        self.manager.schema()
    }

    pub(super) fn write(
        &mut self,
        batches: impl Iterator<Item = Result<RecordBatch>>,
    ) -> Result<usize> {
        self.write_inner(batches.peekable())
            .map_err(spill_file_error)
    }

    fn write_inner(
        &mut self,
        mut batches: std::iter::Peekable<impl Iterator<Item = Result<RecordBatch>>>,
    ) -> Result<usize> {
        if batches.peek().is_none() {
            return Ok(0);
        }
        let mut written = 0;
        match &mut self.writer {
            PartitionWriter::Reopenable(encoder) => {
                let mut writer = self.file.open_writer()?;
                if encoder.is_none() {
                    *encoder = Some(IPCStreamEncoder::new(
                        self.manager.schema().as_ref(),
                        self.manager.compression,
                    )?);
                    self.manager.metrics.spill_file_count.add(1);
                }
                let encoder = encoder.as_mut().unwrap();
                for batch in batches {
                    let batch = gc_view_arrays(&batch?)?;
                    let (rows, bytes, buffers) = encoder.encode(&batch)?;
                    for buffer in buffers {
                        writer.write_all(buffer.as_slice())?;
                    }
                    written += rows;
                    self.manager.metrics.spilled_rows.add(rows);
                    self.manager.metrics.spilled_bytes.add(bytes);
                }
                writer.flush()?;
            }
            PartitionWriter::Persistent(writer) => {
                if writer.is_none() {
                    *writer = Some((
                        InProgressSpillFile::new(
                            Arc::clone(&self.manager),
                            Arc::clone(&self.file),
                        ),
                        persistent_writer_permit()?,
                    ));
                }
                let (writer, _) = writer.as_mut().unwrap();
                for batch in batches {
                    let batch = batch?;
                    writer.append_batch(&batch)?;
                    written += batch.num_rows();
                }
            }
        }
        Ok(written)
    }

    pub(super) fn finish(&mut self) -> Result<Option<Arc<dyn SpillFile>>> {
        self.finish_inner().map_err(spill_file_error)
    }

    fn finish_inner(&mut self) -> Result<Option<Arc<dyn SpillFile>>> {
        match &mut self.writer {
            PartitionWriter::Reopenable(Some(encoder)) => {
                let mut writer = self.file.open_writer()?;
                let (bytes, buffers) = encoder.finish()?;
                for buffer in buffers {
                    writer.write_all(buffer.as_slice())?;
                }
                writer.flush()?;
                writer.finish()?;
                self.manager.metrics.spilled_bytes.add(bytes);
                Ok(Some(Arc::clone(&self.file)))
            }
            PartitionWriter::Persistent(Some((writer, _))) => writer.finish(),
            _ => Ok(None),
        }
    }
}

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
/// could not hold, with the charge that covers it; the next chunk starts
/// with it and takes the charge over.
pub(crate) struct Cursor {
    file: SpilledFile,
    spill_manager: SpillManager,
    /// The largest chunk unit charged without refusal
    /// (`SpillOptions::chunk_unit`).
    unit: usize,
    state: Mutex<CursorState>,
}

#[derive(Default)]
struct CursorState {
    reader: Option<Rows>,
    /// Batches given back, read before the file, each with its charge.
    back: Vec<(RecordBatch, MemoryReservation)>,
    /// The charge of the batch given back that was read last, until the
    /// chunk build takes it over.
    transit: Option<MemoryReservation>,
    /// The file has been read to its end.
    ended: bool,
}

impl Cursor {
    pub(super) fn new(
        file: SpilledFile,
        spill_manager: SpillManager,
        unit: usize,
    ) -> Self {
        Self {
            file,
            spill_manager,
            unit,
            state: Mutex::default(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CursorState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(crate) fn unit(&self) -> usize {
        self.unit
    }

    /// Gives `batch` back with `charge`, which covers it: the next chunk
    /// reads it first.
    pub(crate) fn give_back(&self, batch: RecordBatch, charge: MemoryReservation) {
        self.lock().back.push((batch, charge));
    }

    /// The charge of the batch given back that was read last, if the last
    /// batch read came back from a chunk.
    pub(crate) fn take_transit(&self) -> Option<MemoryReservation> {
        self.lock().transit.take()
    }

    /// Every row of the file was taken by a chunk.
    pub(super) fn is_exhausted(&self) -> bool {
        let state = self.lock();
        state.ended && state.back.is_empty()
    }

    fn poll_next(&self, cx: &mut Context<'_>) -> Poll<Option<Result<RecordBatch>>> {
        let mut state = self.lock();
        if let Some((batch, charge)) = state.back.pop() {
            // The chunk build takes the charge over as it admits the batch
            // (`ChunkBuild::admit`, right after the read).
            state.transit = Some(charge);
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
                Err(error) => return Poll::Ready(Some(Err(spill_file_error(error)))),
            }
        }
        let reader = state.reader.as_mut().expect("a reader");
        let next = reader.poll_next_unpin(cx);
        if matches!(next, Poll::Ready(None)) {
            state.ended = true;
            state.reader = None;
        }
        next.map(|batch| batch.map(|batch| batch.map_err(spill_file_error)))
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
    batch: &RecordBatch,
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
            EquivalenceProperties::new(schema),
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
                    .map_err(spill_file_error)
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = futures::stream::iter(streams)
            .flatten()
            .map_err(spill_file_error);
        let Some(row_ids) = self.row_ids.clone() else {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(schema, rows)));
        };
        let numbered = Arc::clone(&schema);
        let mut start = 0;
        let rows = rows.map(move |batch| {
            let batch = batch?;
            let first = start;
            start += batch.num_rows();
            number_rows(&batch, first, &numbered, row_ids.skip_matched.as_deref())
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, rows)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::config::SpillCompression;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;

    use crate::metrics::{ExecutionPlanMetricsSet, SpillMetrics};

    #[tokio::test]
    async fn join_spill_local_writers_wait_for_capacity() -> Result<()> {
        static SLOTS: Semaphore = Semaphore::const_new(2);
        let held = SLOTS.acquire_many(2).await.unwrap();
        let schema =
            Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let manager = SpillManager::new(
            RuntimeEnvBuilder::new().build_arc()?,
            SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
            Arc::clone(&schema),
        );
        let writes = (0..5).map(|key| {
            let manager = &manager;
            let schema = &schema;
            async move {
                let mut slot = PollShortWrite::default();
                let file = {
                    let _permit =
                        futures::future::poll_fn(|cx| slot.poll(&SLOTS, cx)).await;
                    let batch = RecordBatch::try_new(
                        Arc::clone(schema),
                        vec![Arc::new(Int64Array::from(vec![key]))],
                    )?;
                    let mut writer = PartitionFileWriter::new(manager, "contended join")?;
                    writer.write(std::iter::once(Ok(batch.clone())))?;
                    (writer.finish()?.unwrap(), batch)
                };
                let actual = manager
                    .read_spill_as_stream_unbuffered(file.0, None)?
                    .try_collect::<Vec<_>>()
                    .await?;
                assert_eq!(actual, vec![file.1]);
                Ok::<_, DataFusionError>(())
            }
        });
        let mut writes = Box::pin(futures::future::try_join_all(writes));
        assert!(futures::poll!(&mut writes).is_pending());
        assert_eq!(manager.metrics.spill_file_count.value(), 0);
        drop(held);
        tokio::time::timeout(std::time::Duration::from_secs(5), writes)
            .await
            .expect("local writers must resume when capacity is released")?;
        assert_eq!(manager.metrics.spill_file_count.value(), 5);
        assert_eq!(SLOTS.available_permits(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn join_spill_reopens_one_ipc_stream_in_append_order() -> Result<()> {
        let schema =
            Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let runtime = RuntimeEnvBuilder::new().build_arc()?;
        let manager = SpillManager::new(
            runtime,
            SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
            Arc::clone(&schema),
        )
        .with_compression_type(SpillCompression::Lz4Frame);
        let batches = [[3, 1], [4, 2], [6, 5]]
            .map(|keys| {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int64Array::from_iter_values(keys))],
                )
            })
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut writer = PartitionFileWriter::new(&manager, "reopen order")?;
        for batch in &batches {
            assert_eq!(writer.write(std::iter::once(Ok(batch.clone())))?, 2);
        }
        let file = writer.finish()?.unwrap();
        assert_eq!(manager.metrics.spill_file_count.value(), 1);
        let actual = manager
            .read_spill_as_stream_unbuffered(file, None)?
            .try_collect::<Vec<_>>()
            .await?;
        assert_eq!(actual, batches);
        Ok(())
    }
}
