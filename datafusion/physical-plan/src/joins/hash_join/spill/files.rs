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

use super::{JoinSpillMetrics, SpilledFile};
use crate::execution_plan::{Boundedness, EmissionType, SchedulingType};
use crate::spill::spill_manager::SpillManager;
use crate::stream::RecordBatchStreamAdapter;
use crate::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    ReplaceChildrenOptions, SendableRecordBatchStream,
};

/// Reads spill files of one written partition, one list of files for each
/// output partition. Declares no ordering and is bounded, so a join over it
/// is eligible to spill again, one level deeper.
pub(crate) struct SpillFilesExec {
    files: Vec<Vec<SpilledFile>>,
    spill_manager: SpillManager,
    /// The spill level of the rows in these files: the level of the join
    /// that reads them.
    level: usize,
    /// Metrics of the first-level join, which the joins of written
    /// partitions report into.
    metrics: JoinSpillMetrics,
    cache: Arc<PlanProperties>,
}

impl SpillFilesExec {
    pub(super) fn new(
        schema: SchemaRef,
        files: Vec<Vec<SpilledFile>>,
        spill_manager: SpillManager,
        level: usize,
        metrics: JoinSpillMetrics,
    ) -> Self {
        let cache = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(files.len()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        )
        .with_scheduling_type(SchedulingType::Cooperative);
        Self {
            files,
            spill_manager,
            level,
            metrics,
            cache: Arc::new(cache),
        }
    }

    pub(super) fn level(&self) -> usize {
        self.level
    }

    pub(super) fn metrics(&self) -> &JoinSpillMetrics {
        &self.metrics
    }
}

impl fmt::Debug for SpillFilesExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillFilesExec")
            .field("partitions", &self.files.len())
            .field("level", &self.level)
            .finish()
    }
}

impl DisplayAs for SpillFilesExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let files: usize = self.files.iter().map(Vec::len).sum();
                write!(f, "SpillFilesExec: level={}, files={files}", self.level)
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
            partition < self.files.len(),
            "SpillFilesExec invalid partition {partition} of {}",
            self.files.len()
        );
        let schema = Arc::clone(self.spill_manager.schema());
        let streams = self.files[partition]
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
