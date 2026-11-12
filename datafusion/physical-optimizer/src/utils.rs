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

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_physical_expr::{
    AcrossPartitions, Distribution, LexOrdering, LexRequirement,
};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::tree_node::PlanContext;
use datafusion_physical_plan::union::UnionExec;
use datafusion_physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

/// This utility function adds a `SortExec` above an operator according to the
/// given ordering requirements while preserving the original partitioning.
///
/// Note that this updates the plan in both the `PlanContext.children` and
/// the `PlanContext.plan`'s children. Therefore its not required to sync
/// the child plans with [`PlanContext::update_plan_from_children`].
pub fn add_sort_above<T: Clone + Default>(
    node: PlanContext<T>,
    sort_requirements: LexRequirement,
    fetch: Option<usize>,
) -> PlanContext<T> {
    let mut sort_reqs: Vec<_> = sort_requirements.into();
    sort_reqs.retain(|sort_expr| {
        // Local comparisons also feed shared TopK thresholds and merge keys.
        // A key constant only within each partition must remain in that order.
        !matches!(
            node.plan
                .equivalence_properties()
                .is_expr_constant(&sort_expr.expr),
            Some(AcrossPartitions::Uniform(_))
        )
    });
    let sort_exprs = sort_reqs.into_iter().map(Into::into).collect::<Vec<_>>();
    let Some(ordering) = LexOrdering::new(sort_exprs) else {
        return node;
    };
    let mut new_sort = SortExec::new(ordering, Arc::clone(&node.plan)).with_fetch(fetch);
    if node.plan.output_partitioning().partition_count() > 1 {
        new_sort = new_sort.with_preserve_partitioning(true);
    }
    PlanContext::new(Arc::new(new_sort), T::default(), vec![node])
}

/// Like [`add_sort_above`], but also inserts a [`SortPreservingMergeExec`] when
/// the parent distribution requires a single partition and the input has
/// multiple partitions. This prevents `SortExec(preserve_partitioning=true)`
/// from violating `SinglePartition` requirements.
pub fn add_sort_above_with_distribution<T: Clone + Default>(
    node: PlanContext<T>,
    sort_requirements: LexRequirement,
    fetch: Option<usize>,
    required_distribution: &Distribution,
) -> PlanContext<T> {
    let mut sort_reqs: Vec<_> = sort_requirements.into();
    sort_reqs.retain(|sort_expr| {
        !matches!(
            node.plan
                .equivalence_properties()
                .is_expr_constant(&sort_expr.expr),
            Some(AcrossPartitions::Uniform(_))
        )
    });
    let sort_exprs = sort_reqs.into_iter().map(Into::into).collect::<Vec<_>>();
    let Some(ordering) = LexOrdering::new(sort_exprs) else {
        return node;
    };
    let input_has_multiple_partitions =
        node.plan.output_partitioning().partition_count() > 1;

    let mut new_sort =
        SortExec::new(ordering.clone(), Arc::clone(&node.plan)).with_fetch(fetch);
    if input_has_multiple_partitions {
        new_sort = new_sort.with_preserve_partitioning(true);
    }

    let sort_node = PlanContext::new(Arc::new(new_sort), T::default(), vec![node]);

    // If the parent requires SinglePartition and the input has multiple partitions,
    // wrap the partition-preserving sort in SortPreservingMergeExec.
    if matches!(required_distribution, Distribution::SinglePartition)
        && input_has_multiple_partitions
    {
        PlanContext::new(
            Arc::new(
                SortPreservingMergeExec::new(ordering, Arc::clone(&sort_node.plan))
                    .with_fetch(fetch),
            ),
            T::default(),
            vec![sort_node],
        )
    } else {
        sort_node
    }
}

/// This utility function adds a `SortExec` above an operator according to the
/// given ordering requirements while preserving the original partitioning. If
/// requirement is already satisfied no `SortExec` is added.
pub fn add_sort_above_with_check<T: Clone + Default>(
    node: PlanContext<T>,
    sort_requirements: LexRequirement,
    fetch: Option<usize>,
) -> Result<PlanContext<T>> {
    if !node
        .plan
        .equivalence_properties()
        .ordering_satisfy_requirement(sort_requirements.clone())?
    {
        Ok(add_sort_above(node, sort_requirements, fetch))
    } else {
        Ok(node)
    }
}

/// Checks whether the given operator is a [`SortExec`].
pub fn is_sort(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<SortExec>()
}

/// Checks whether the given operator is a window;
/// i.e. either a [`WindowAggExec`] or a [`BoundedWindowAggExec`].
pub fn is_window(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<WindowAggExec>() || plan.is::<BoundedWindowAggExec>()
}

/// Checks whether the given operator is a [`UnionExec`].
pub fn is_union(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<UnionExec>()
}

/// Checks whether the given operator is a [`SortPreservingMergeExec`].
pub fn is_sort_preserving_merge(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<SortPreservingMergeExec>()
}

/// Checks whether the given operator is a [`CoalescePartitionsExec`].
pub fn is_coalesce_partitions(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<CoalescePartitionsExec>()
}

/// Checks whether the given operator is a [`RepartitionExec`].
pub fn is_repartition(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<RepartitionExec>()
}

/// Checks whether the given operator is a limit;
/// i.e. either a [`LocalLimitExec`] or a [`GlobalLimitExec`].
pub fn is_limit(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<GlobalLimitExec>() || plan.is::<LocalLimitExec>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_physical_expr::{
        PhysicalSortExpr,
        expressions::{col, lit},
    };
    use datafusion_physical_plan::{empty::EmptyExec, projection::ProjectionExec};

    #[test]
    fn partition_local_constants_remain_in_topk_and_merge_keys() -> Result<()> {
        let schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let branches = [1i32, 2]
            .into_iter()
            .map(|x| {
                Ok(Arc::new(ProjectionExec::try_new(
                    vec![(lit(x), "x".into()), (col("id", &schema)?, "id".into())],
                    Arc::new(EmptyExec::new(Arc::clone(&schema))),
                )?) as Arc<dyn ExecutionPlan>)
            })
            .collect::<Result<Vec<_>>>()?;
        let union = UnionExec::try_new(branches)?;
        let ordering = LexOrdering::new([
            PhysicalSortExpr::new_default(col("x", &union.schema())?).desc(),
            PhysicalSortExpr::new_default(col("id", &union.schema())?),
        ])
        .unwrap();
        for fetch in [None, Some(3)] {
            let node = PlanContext::new(Arc::clone(&union), (), vec![]);
            let local =
                add_sort_above(node, LexRequirement::from(ordering.clone()), fetch);
            assert_eq!(
                local.plan.downcast_ref::<SortExec>().unwrap().expr(),
                &ordering
            );
            let node = PlanContext::new(Arc::clone(&union), (), vec![]);
            let merged = add_sort_above_with_distribution(
                node,
                LexRequirement::from(ordering.clone()),
                fetch,
                &Distribution::SinglePartition,
            );
            assert_eq!(
                merged
                    .plan
                    .downcast_ref::<SortPreservingMergeExec>()
                    .unwrap()
                    .expr(),
                &ordering
            );
        }
        Ok(())
    }
}
