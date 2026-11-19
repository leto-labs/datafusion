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
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::Result;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{assert_or_internal_err, internal_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr};
use futures::StreamExt;

use super::{JoinSpillMetrics, ProbeRegistry, SpilledFile};
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
    /// Output partition `n` reads probe stream `n`'s file of partition
    /// `part`, taken from the registry when `n` executes: the stream writes
    /// it before it executes its partition of the join.
    Registry {
        registry: Arc<ProbeRegistry>,
        part: usize,
        partitions: usize,
    },
}

impl FileSource {
    fn partitions(&self) -> usize {
        match self {
            Self::Fixed(files) => files.len(),
            Self::Registry { partitions, .. } => *partitions,
        }
    }

    fn take(&self, partition: usize) -> Vec<SpilledFile> {
        match self {
            Self::Fixed(files) => files[partition].clone(),
            Self::Registry { registry, part, .. } => {
                registry.take(partition, *part).into_iter().collect()
            }
        }
    }
}

/// What the rows of a build leaf went through: whether the join that wrote
/// them divided its build over several partitions, and how many there are.
/// A join over them that cannot spill says why from these facts.
#[derive(Debug, Clone, Copy)]
pub(super) struct Origin {
    pub(super) divided: bool,
    pub(super) rows: usize,
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
            .take(partition)
            .iter()
            // Unbuffered: the read runs in the consumer's poll. A buffered
            // read's producer task ends its stream as if the file ended when
            // the attempt stops it, and holds uncharged batches ahead.
            .map(|file| {
                self.spill_manager
                    .read_spill_as_stream_unbuffered(Arc::clone(&file.file), None)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(streams).flatten(),
        )))
    }
}
