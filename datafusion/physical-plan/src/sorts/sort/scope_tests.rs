// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::SortExec;
use crate::attempt_tasks::AttemptTasks;
use crate::common::collect;
use crate::execution_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions};
use crate::sorts::sort_preserving_merge::SortPreservingMergeExec;
use crate::test::TestMemoryExec;
use crate::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    SendableRecordBatchStream,
};
use arrow::array::{ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
use datafusion_common::Result;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_execution::{TaskContext, config::SessionConfig};
use datafusion_physical_expr::{
    LexOrdering, PhysicalExpr, PhysicalSortExpr, expressions::Column,
};
use std::fmt::{Formatter, Result as FmtResult};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct ContextRecorder {
    input: Arc<dyn ExecutionPlan>,
    contexts: Arc<Mutex<Vec<Arc<TaskContext>>>>,
}

impl DisplayAs for ContextRecorder {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut Formatter) -> FmtResult {
        write!(f, "ContextRecorder")
    }
}

impl ExecutionPlan for ContextRecorder {
    fn name(&self) -> &'static str {
        "ContextRecorder"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }
    fn apply_expressions(
        &self,
        _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn replace_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
        _: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.contexts.lock().unwrap().push(Arc::clone(&context));
        self.input.execute(partition, context)
    }
}

#[tokio::test]
async fn streaming_merge_partition_independence_stops_at_blocking_sorts() -> Result<()> {
    for partitions in [1, 2] {
        // None: direct streaming scan; false: actual sort; true: pass-through sort.
        for sorted in [None, Some(false), Some(true)] {
            for fetch in [None, Some(2)] {
                let batch = RecordBatch::try_from_iter([(
                    "key",
                    Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                )])?;
                let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(
                    Column::new("key", 0),
                ))]
                .into();
                let batches = vec![vec![batch.clone()]; partitions];
                let mut source = TestMemoryExec::try_new(&batches, batch.schema(), None)?;
                if sorted == Some(true) {
                    source = source.try_with_sort_information(vec![ordering.clone()])?;
                }
                let contexts = Arc::new(Mutex::new(Vec::new()));
                let source: Arc<dyn ExecutionPlan> = Arc::new(ContextRecorder {
                    input: Arc::new(source),
                    contexts: Arc::clone(&contexts),
                });
                let input = if sorted.is_some() {
                    Arc::new(
                        SortExec::new(ordering.clone(), source)
                            .with_preserve_partitioning(true)
                            .with_fetch(fetch),
                    ) as Arc<dyn ExecutionPlan>
                } else {
                    source
                };
                let merge =
                    SortPreservingMergeExec::new(ordering, input).with_fetch(fetch);
                let scope = AttemptTasks::new();
                let context = Arc::new(TaskContext::default().with_session_config(
                    SessionConfig::new().with_extension(Arc::clone(&scope)),
                ));
                let output = collect(merge.execute(0, Arc::clone(&context))?).await?;
                assert!(!output.is_empty());
                {
                    let observed = contexts.lock().unwrap();
                    assert_eq!(observed.len(), partitions);
                    let required = partitions > 1 && sorted != Some(false);
                    for child_context in observed.iter() {
                        assert_eq!(
                            child_context.requires_independent_partitions(),
                            required,
                            "partitions={partitions} sorted={sorted:?} fetch={fetch:?}"
                        );
                        assert!(Arc::ptr_eq(
                            &AttemptTasks::from_task(child_context).unwrap(),
                            &scope
                        ));
                        assert!(Arc::ptr_eq(
                            &child_context.runtime_env(),
                            &context.runtime_env()
                        ));
                    }
                }
                scope.root_joined();
                scope.drain().await.unwrap();
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn ordered_repartition_requires_independent_inputs() -> Result<()> {
    use crate::Partitioning;
    use crate::repartition::RepartitionExec;
    for partitions in [1, 2] {
        for preserve_order in [false, true] {
            for inherited in [false, true] {
                let batch = RecordBatch::try_from_iter([(
                    "key",
                    Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                )])?;
                let ordering: LexOrdering = [PhysicalSortExpr::new_default(Arc::new(
                    Column::new("key", 0),
                ))]
                .into();
                let batches = vec![vec![batch.clone()]; partitions];
                let source = TestMemoryExec::try_new(&batches, batch.schema(), None)?
                    .try_with_sort_information(vec![ordering])?;
                let contexts = Arc::new(Mutex::new(Vec::new()));
                let source: Arc<dyn ExecutionPlan> = Arc::new(ContextRecorder {
                    input: Arc::new(source),
                    contexts: Arc::clone(&contexts),
                });
                let mut repartition =
                    RepartitionExec::try_new(source, Partitioning::RoundRobinBatch(1))?;
                if preserve_order {
                    repartition = repartition.with_preserve_order();
                }
                let scope = AttemptTasks::new();
                let context = Arc::new(TaskContext::default().with_session_config(
                    SessionConfig::new().with_extension(Arc::clone(&scope)),
                ))
                .with_partition_independence(inherited);
                let output =
                    collect(repartition.execute(0, Arc::clone(&context))?).await?;
                assert_eq!(
                    output.iter().map(RecordBatch::num_rows).sum::<usize>(),
                    3 * partitions
                );
                assert_eq!(context.requires_independent_partitions(), inherited);
                {
                    let observed = contexts.lock().unwrap();
                    assert_eq!(observed.len(), partitions);
                    for child in observed.iter() {
                        assert_eq!(
                            child.requires_independent_partitions(),
                            inherited || (preserve_order && partitions > 1)
                        );
                        assert!(Arc::ptr_eq(
                            &scope,
                            &AttemptTasks::from_task(child).unwrap()
                        ));
                    }
                }
                scope.root_joined();
                scope.drain().await.unwrap();
            }
        }
    }
    Ok(())
}
