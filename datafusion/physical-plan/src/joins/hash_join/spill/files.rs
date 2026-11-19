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
use std::sync::{Arc, Mutex};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{DataFusionError, Result, resources_datafusion_err};
use datafusion_common::{assert_or_internal_err, internal_err};
use datafusion_execution::TaskContext;
use datafusion_execution::spill_file::SpillFile;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr};
use futures::{StreamExt, TryStreamExt};
use tokio::sync::{Semaphore, SemaphorePermit};

use super::{JoinSpillMetrics, SpilledFile};
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
pub(super) struct PartitionFileWriter {
    manager: Arc<SpillManager>,
    file: Arc<dyn SpillFile>,
    writer: PartitionWriter,
}

enum PartitionWriter {
    Reopenable(Option<IPCStreamEncoder>),
    Persistent(Option<(InProgressSpillFile, SemaphorePermit<'static>)>),
}

// A custom backend can require a writer to remain live until commit. Share
// the bound with append operations so overlapping joins cannot multiply it.
const MAX_OPEN_JOIN_SPILL_WRITERS: usize = 128;
static JOIN_SPILL_WRITERS: Semaphore = Semaphore::const_new(MAX_OPEN_JOIN_SPILL_WRITERS);

fn writer_permit() -> Result<SemaphorePermit<'static>> {
    JOIN_SPILL_WRITERS.try_acquire().map_err(|_| {
        resources_datafusion_err!(
            "hash join spill reached its open-writer limit ({MAX_OPEN_JOIN_SPILL_WRITERS})"
        )
    })
}

pub(super) fn spill_file_error(error: DataFusionError) -> DataFusionError {
    #[cfg(unix)]
    {
        let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&error);
        while let Some(cause) = source {
            if let Some(io) = cause.downcast_ref::<std::io::Error>() {
                // tempfile adds the path to the error text and exposes the
                // original error's source, losing the OS code in the chain.
                let code = io.raw_os_error().or_else(|| {
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
        let _permit = writer_permit()?;
        let file = manager
            .env()
            .disk_manager
            .create_tmp_file(description)
            .map_err(spill_file_error)?;
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
                let _permit = writer_permit()?;
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
                        writer_permit()?,
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
                let _permit = writer_permit()?;
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
}

impl FileSource {
    fn partitions(&self) -> usize {
        match self {
            Self::Fixed(files) => files.len(),
            Self::Staged(_) => 1,
        }
    }

    fn take(&self, partition: usize) -> Result<Vec<SpilledFile>> {
        match self {
            Self::Fixed(files) => Ok(files[partition].clone()),
            Self::Staged(stage) => stage.take(),
        }
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
            cache: Arc::new(cache),
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
        let schema = Arc::clone(self.spill_manager.schema());
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
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(streams)
                .flatten()
                .map_err(spill_file_error),
        )))
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
