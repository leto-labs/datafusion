// Modified by the Longbow authors; see third_party/README.md.

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Sort Pushdown Optimization
//!
//! This optimizer attempts to push sort requirements down through the execution plan
//! tree to data sources that can natively handle them (e.g., by scanning files in
//! reverse order).
//!
//! ## How it works
//!
//! 1. Detects `SortExec` nodes in the plan
//! 2. Calls `try_pushdown_sort()` on the input to recursively push the sort requirement
//! 3. Each node type defines its own pushdown behavior:
//!    - **Transparent nodes** (CoalesceBatchesExec, RepartitionExec, etc.) delegate to
//!      their children and wrap the result
//!    - **Data sources** (DataSourceExec) check if they can optimize for the ordering
//!    - **Blocking nodes** return `Unsupported` to stop pushdown
//! 4. Based on the result:
//!    - `Exact`: Remove the Sort operator (data source guarantees perfect ordering)
//!    - `Inexact`: Keep Sort but use optimized input (enables early termination for TopK)
//!    - `Unsupported`: No change
//!
//! ## Capabilities
//!
//! - **Sort elimination**: when a data source's natural ordering satisfies the
//!   request, return `Exact` and remove the `SortExec` entirely. Preserves
//!   `fetch` (LIMIT) from the eliminated `SortExec` for early termination.
//! - **Statistics-based file sorting**: sort files within each partition by
//!   min/max statistics. When files are non-overlapping but listed in wrong
//!   order (e.g., alphabetical order ≠ sort key order), this fixes the ordering
//!   and enables sort elimination. Works for both single-partition and
//!   multi-partition plans with multi-file groups.
//! - **Reverse scan optimization**: when required sort is the reverse of the data source's
//!   natural ordering, enable reverse scanning (reading row groups in reverse order)
//! - **Prefix matching**: if data has ordering [A DESC, B ASC] and query needs
//!   [A DESC], the existing ordering satisfies the requirement (`Exact`).
//!   If the query needs [A ASC] (reverse of the prefix), a reverse scan is
//!   used (`Inexact`, `SortExec` retained)
//!
//! Related issue: <https://github.com/apache/datafusion/issues/17348>

use crate::PhysicalOptimizerRule;
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{
    Transformed, TransformedResult, TreeNode, TreeNodeRecursion,
};
use datafusion_physical_plan::SortOrderPushdownResult;
use datafusion_physical_plan::buffer::BufferExec;
use datafusion_physical_plan::execution_plan::replace_children_if_necessary;
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use std::sync::Arc;

/// A PhysicalOptimizerRule that attempts to push down sort requirements to data sources.
///
/// See module-level documentation for details.
#[derive(Debug, Clone, Default)]
pub struct PushdownSort;

impl PushdownSort {
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicalOptimizerRule for PushdownSort {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Check if sort pushdown optimization is enabled
        if !config.optimizer.enable_sort_pushdown {
            return Ok(plan);
        }

        let buffer_capacity = config.execution.sort_pushdown_buffer_capacity;

        // Use transform_down to find and optimize all SortExec nodes (including nested ones)
        // Also handles SPM → SortExec pattern to insert BufferExec when sort is eliminated
        plan.transform_down(|plan: Arc<dyn ExecutionPlan>| {
            // Longbow's positional consumer owns the global merge above payload
            // retrieval. Its first input must retain local key candidates. Use
            // the explicit-merge optimization path, then keep only its local
            // input. All other standalone sorts retain upstream #23301 behavior.
            if plan.name() == "PositionalFetchExec"
                && let [keys, payload] = plan.children().as_slice()
                && let Some(sort) = keys.downcast_ref::<SortExec>()
                && sort.preserve_partitioning()
                && sort.fetch().is_some()
            {
                let merge = Arc::new(
                    SortPreservingMergeExec::new(sort.expr().clone(), Arc::clone(keys))
                        .with_fetch(sort.fetch()),
                );
                let optimized = self.optimize(merge, config)?;
                let merge = optimized
                    .downcast_ref::<SortPreservingMergeExec>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "sort pushdown lost its explicit merge"
                        )
                    })?;
                let payload = self.optimize(Arc::clone(payload), config)?;
                let rebuilt = replace_children_if_necessary(
                    plan,
                    vec![Arc::clone(merge.input()), payload],
                )?;
                return Ok(Transformed::new(rebuilt, true, TreeNodeRecursion::Jump));
            }

            // Pattern 1: SPM → SortExec(preserve_partitioning)
            // When we eliminate the SortExec, SPM loses its memory buffer and reads
            // directly from I/O-bound sources. Insert a BufferExec to compensate.
            if let Some(spm) = plan.downcast_ref::<SortPreservingMergeExec>()
                && let Some(sort_child) = spm.input().downcast_ref::<SortExec>()
                && sort_child.preserve_partitioning()
            {
                let sort_input = Arc::clone(sort_child.input());
                let required_ordering = sort_child.expr();
                match sort_input.try_pushdown_sort(required_ordering)? {
                    SortOrderPushdownResult::Exact { inner } => {
                        // Preserve fetch (LIMIT) from the eliminated SortExec.
                        // Use LocalLimitExec (not Global) since input is multi-partition.
                        let inner = if let Some(fetch) = sort_child.fetch() {
                            inner.with_fetch(Some(fetch)).unwrap_or_else(|| {
                                Arc::new(LocalLimitExec::new(inner, fetch))
                            })
                        } else {
                            inner
                        };
                        // Insert BufferExec to replace SortExec's buffering role.
                        // SortExec buffered all data in memory; BufferExec provides
                        // bounded buffering so SPM doesn't stall on I/O.
                        let buffered: Arc<dyn ExecutionPlan> =
                            Arc::new(BufferExec::new(inner, buffer_capacity));
                        let new_spm =
                            SortPreservingMergeExec::new(spm.expr().clone(), buffered)
                                .with_fetch(spm.fetch());
                        return Ok(Transformed::yes(Arc::new(new_spm)));
                    }
                    SortOrderPushdownResult::Inexact { inner } => {
                        let new_sort = SortExec::new(required_ordering.clone(), inner)
                            .with_fetch(sort_child.fetch())
                            .with_preserve_partitioning(true);
                        let new_spm = SortPreservingMergeExec::new(
                            spm.expr().clone(),
                            Arc::new(new_sort),
                        )
                        .with_fetch(spm.fetch());
                        // The replacement already has the required
                        // `SortPreservingMergeExec` parent. Do not descend
                        // into its `SortExec` child and treat it as a
                        // standalone TopK.
                        return Ok(Transformed::new(
                            Arc::new(new_spm),
                            true,
                            TreeNodeRecursion::Jump,
                        ));
                    }
                    SortOrderPushdownResult::Unsupported => {
                        return Ok(Transformed::no(plan));
                    }
                }
            }

            // Pattern 2: Standalone SortExec (no SPM parent)
            let Some(sort_exec) = plan.downcast_ref::<SortExec>() else {
                return Ok(Transformed::no(plan));
            };

            let sort_input = Arc::clone(sort_exec.input());
            let required_ordering = sort_exec.expr();

            // Try to push the sort requirement down through the plan tree
            // Each node type defines its own pushdown behavior via try_pushdown_sort()
            match sort_input.try_pushdown_sort(required_ordering)? {
                SortOrderPushdownResult::Exact { inner } => {
                    // Data source guarantees perfect ordering - remove the Sort operator.
                    //
                    // If the SortExec carried a fetch (LIMIT), we must preserve it.
                    // First try pushing the limit into the source via `with_fetch()`.
                    // If the source doesn't support `with_fetch`, fall back to
                    // wrapping with GlobalLimitExec.
                    if let Some(fetch) = sort_exec.fetch() {
                        let inner = inner.with_fetch(Some(fetch)).unwrap_or_else(|| {
                            Arc::new(GlobalLimitExec::new(inner, 0, Some(fetch)))
                        });
                        Ok(Transformed::yes(inner))
                    } else {
                        Ok(Transformed::yes(inner))
                    }
                }
                SortOrderPushdownResult::Inexact { inner } => {
                    // Data source is optimized for the ordering but not perfectly sorted
                    // Keep the Sort operator but use the optimized input
                    // Benefits: TopK queries can terminate early, better cache locality
                    // A standalone multi-partition TopK still needs a global
                    // merge; otherwise a later coalesce can concatenate
                    // locally sorted partitions.
                    let preserve_partitioning = sort_exec.preserve_partitioning();
                    let needs_global_topk =
                        preserve_partitioning && sort_exec.fetch().is_some();
                    let input_partitions = inner.output_partitioning().partition_count();
                    let new_sort: Arc<dyn ExecutionPlan> = Arc::new(
                        SortExec::new(required_ordering.clone(), inner)
                            .with_fetch(sort_exec.fetch())
                            .with_preserve_partitioning(preserve_partitioning),
                    );
                    if needs_global_topk && input_partitions > 1 {
                        let new_spm = SortPreservingMergeExec::new(
                            required_ordering.clone(),
                            new_sort,
                        )
                        .with_fetch(sort_exec.fetch());
                        // Do not descend into the newly inserted
                        // `SortExec`, or this standalone branch will wrap it
                        // in another `SortPreservingMergeExec`.
                        Ok(Transformed::new(
                            Arc::new(new_spm),
                            true,
                            TreeNodeRecursion::Jump,
                        ))
                    } else {
                        Ok(Transformed::yes(new_sort))
                    }
                }
                SortOrderPushdownResult::Unsupported => {
                    // Cannot optimize for this ordering - no change
                    Ok(Transformed::no(plan))
                }
            }
        })
        .data()
    }

    fn name(&self) -> &str {
        "PushdownSort"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod partition_scope_tests {
    use super::*;
    use std::fmt;

    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::{
        EquivalenceProperties, LexOrdering, PhysicalExpr, PhysicalSortExpr,
    };
    use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
    use datafusion_physical_plan::test::TestMemoryExec;
    use datafusion_physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlanProperties, PlanProperties,
        SendableRecordBatchStream,
    };

    #[derive(Debug, Clone)]
    struct SortableInput {
        input: Arc<dyn ExecutionPlan>,
        properties: Arc<PlanProperties>,
        exact: bool,
    }

    impl DisplayAs for SortableInput {
        fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "SortableInput: exact={}", self.exact)
        }
    }

    impl ExecutionPlan for SortableInput {
        fn name(&self) -> &str {
            "SortableInput"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            &self.properties
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

        fn with_new_children(
            self: Arc<Self>,
            _: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }

        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            self.input.execute(partition, context)
        }

        fn try_pushdown_sort(
            &self,
            _: &[PhysicalSortExpr],
        ) -> Result<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
            let inner = Arc::new(self.clone()) as Arc<dyn ExecutionPlan>;
            Ok(if self.exact {
                SortOrderPushdownResult::Exact { inner }
            } else {
                SortOrderPushdownResult::Inexact { inner }
            })
        }
    }

    #[derive(Debug)]
    struct PositionalConsumer {
        keys: Arc<dyn ExecutionPlan>,
        payload: Arc<dyn ExecutionPlan>,
    }

    impl DisplayAs for PositionalConsumer {
        fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "PositionalFetchExec")
        }
    }

    impl ExecutionPlan for PositionalConsumer {
        fn name(&self) -> &str {
            "PositionalFetchExec"
        }
        fn properties(&self) -> &Arc<PlanProperties> {
            self.keys.properties()
        }
        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.keys, &self.payload]
        }
        fn apply_expressions(
            &self,
            _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }
        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(Self {
                keys: Arc::clone(&children[0]),
                payload: Arc::clone(&children[1]),
            }))
        }
        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            self.keys.execute(partition, context)
        }
    }

    async fn check_scope(exact: bool, preserve: bool, merge: bool) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let ordering = LexOrdering::new([PhysicalSortExpr::new_default(Arc::new(
            Column::new("v", 0),
        ))])
        .unwrap();
        let parts = if preserve { 2 } else { 1 };
        let values: Vec<_> = (0..parts)
            .map(|p| {
                let mut values = if p == 0 {
                    vec![8, 5, 7, 6]
                } else {
                    vec![4, 1, 3, 2]
                };
                if exact {
                    values.sort_unstable();
                }
                values
            })
            .collect();
        let batches = values
            .iter()
            .map(|values| {
                Ok(vec![RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int32Array::from(values.clone()))],
                )?])
            })
            .collect::<Result<Vec<_>>>()?;
        let input = TestMemoryExec::try_new_exec(&batches, Arc::clone(&schema), None)?;
        let eq = if exact {
            EquivalenceProperties::new_with_orderings(schema, [ordering.clone()])
        } else {
            EquivalenceProperties::new(schema)
        };
        let input = Arc::new(SortableInput {
            input,
            properties: Arc::new(PlanProperties::new(
                eq,
                datafusion_physical_expr::Partitioning::UnknownPartitioning(parts),
                EmissionType::Incremental,
                Boundedness::Bounded,
            )),
            exact,
        }) as Arc<dyn ExecutionPlan>;
        let sort = Arc::new(
            SortExec::new(ordering.clone(), Arc::clone(&input))
                .with_fetch(Some(2))
                .with_preserve_partitioning(preserve),
        ) as Arc<dyn ExecutionPlan>;
        let plan = if merge {
            Arc::new(SortPreservingMergeExec::new(ordering, sort).with_fetch(Some(2)))
                as Arc<dyn ExecutionPlan>
        } else if preserve {
            Arc::new(PositionalConsumer {
                keys: sort,
                payload: input,
            }) as Arc<dyn ExecutionPlan>
        } else {
            sort
        };
        let optimized = PushdownSort::new().optimize(plan, &ConfigOptions::default())?;
        assert_eq!(
            optimized.output_partitioning().partition_count(),
            if merge { 1 } else { parts }
        );
        let context = Arc::new(TaskContext::default());
        for partition in 0..optimized.output_partitioning().partition_count() {
            let batches = datafusion_physical_plan::common::collect(
                optimized.execute(partition, Arc::clone(&context))?,
            )
            .await?;
            let actual: Vec<i32> = batches
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values()
                        .iter()
                        .copied()
                })
                .collect();
            assert_eq!(
                actual,
                if merge || partition == 1 {
                    vec![1, 2]
                } else {
                    vec![5, 6]
                }
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn positional_consumer_keeps_inexact_partition_topk_scope() -> Result<()> {
        check_scope(false, true, false).await
    }

    #[tokio::test]
    async fn positional_consumer_keeps_exact_partition_limit_scope() -> Result<()> {
        check_scope(true, true, false).await
    }

    #[tokio::test]
    async fn global_topk_keeps_its_explicit_merge_or_single_partition() -> Result<()> {
        for exact in [false, true] {
            check_scope(exact, true, true).await?;
            check_scope(exact, false, false).await?;
        }
        Ok(())
    }
}
