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

//! Parent dynamic predicates must retain their positions and live state.

use std::sync::Arc;

use super::SortMergeJoinExec;
use crate::expressions::{Column, PhysicalSortExpr};
use crate::filter::FilterExec;
use crate::filter_pushdown::{FilterPushdownPhase, PushedDown};
use crate::sorts::sort::SortExec;
use crate::test::TestMemoryExec;
use crate::{ExecutionPlan, collect as collect_plan};
use arrow::array::{ArrayRef, AsArray, Int64Array, RecordBatch};
use arrow::compute::{SortOptions, filter_record_batch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::config::ConfigOptions;
use datafusion_common::test_util::batches_to_sort_string;
use datafusion_common::{JoinType, NullEquality, Result, ScalarValue};
use datafusion_execution::TaskContext;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{
    BinaryExpr, DynamicFilterPhysicalExpr, IsNullExpr, Literal,
};
use datafusion_physical_expr_common::sort_expr::LexOrdering;

type Expr = Arc<dyn PhysicalExpr>;

fn source(right: bool, duplicate_names: bool) -> Result<Arc<dyn ExecutionPlan>> {
    // Names repeat across inputs. An optional within-input ambiguity is used
    // by the negative route control below.
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
        Field::new(
            if duplicate_names { "v" } else { "payload" },
            DataType::Int64,
            true,
        ),
    ]));
    let scale = if right { 10 } else { 1 };
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![None, Some(1), Some(1), Some(2)])),
        Arc::new(Int64Array::from(vec![
            None,
            Some(10 * scale),
            Some(20 * scale),
            Some(30 * scale),
        ])),
        Arc::new(Int64Array::from(vec![
            None,
            Some(10 * scale + 1),
            Some(20 * scale + 1),
            Some(30 * scale + 1),
        ])),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?;
    let ordering =
        LexOrdering::new([PhysicalSortExpr::new_default(Arc::new(Column::new("k", 0)))])
            .unwrap();
    Ok(Arc::new(SortExec::new(ordering, input)))
}

fn join(kind: JoinType) -> Result<Arc<dyn ExecutionPlan>> {
    join_with_names(kind, false)
}

fn join_with_names(
    kind: JoinType,
    duplicate_names: bool,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(SortMergeJoinExec::try_new(
        source(false, duplicate_names)?,
        source(true, duplicate_names)?,
        vec![(Arc::new(Column::new("k", 0)), Arc::new(Column::new("k", 0)))],
        None,
        kind,
        vec![SortOptions::default()],
        NullEquality::NullEqualsNull,
    )?))
}

fn dynamic(index: usize, name: &str) -> Arc<DynamicFilterPhysicalExpr> {
    Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new(name, index))],
        Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
    ))
}

#[tokio::test]
async fn inner_merge_routes_live_parent_filters_by_position() -> Result<()> {
    for (index, value) in [(2, Some(21)), (5, Some(201)), (0, None), (3, None)] {
        let plan = join(JoinType::Inner)?;
        let name = if index % 3 == 0 { "k" } else { "payload" };
        let parent = dynamic(index, name);
        let root: Expr = parent.clone();
        let description = plan.gather_filters_for_pushdown(
            FilterPushdownPhase::Post,
            vec![Arc::clone(&root)],
            &ConfigOptions::default(),
        )?;
        let filters = description.parent_filters();
        let side = usize::from(index >= 3);
        assert!(matches!(filters[side][0].discriminant, PushedDown::Yes));
        assert!(matches!(filters[1 - side][0].discriminant, PushedDown::No));
        assert!(description.self_filters().iter().all(Vec::is_empty));
        let mapped = Arc::clone(&filters[side][0].predicate);
        assert_eq!(mapped.expression_id(), root.expression_id());
        let mapped_column = mapped.children()[0]
            .downcast_ref::<Column>()
            .unwrap()
            .clone();
        assert_eq!(mapped_column.index(), index % 3);
        // The predicate becomes selective only after deriving its child view.
        let column: Expr = Arc::new(Column::new(name, index));
        let predicate: Expr = match value {
            Some(value) => Arc::new(BinaryExpr::new(
                column,
                Operator::Eq,
                Arc::new(Literal::new(ScalarValue::Int64(Some(value)))),
            )),
            None => Arc::new(IsNullExpr::new(column)),
        };
        parent.update(predicate)?;
        let context = Arc::new(TaskContext::default());
        let unfiltered = collect_plan(Arc::clone(&plan), Arc::clone(&context)).await?;
        let expected = unfiltered
            .iter()
            .map(|batch| {
                let mask = root.evaluate(batch)?.into_array(batch.num_rows())?;
                Ok(filter_record_batch(batch, mask.as_boolean())?)
            })
            .collect::<Result<Vec<_>>>()?;
        assert!(expected.iter().map(RecordBatch::num_rows).sum::<usize>() > 0);
        let mut children = plan.children().into_iter().cloned().collect::<Vec<_>>();
        children[side] =
            Arc::new(FilterExec::try_new(mapped, Arc::clone(&children[side]))?);
        let pushed = plan.with_new_children(children)?;
        let actual = collect_plan(pushed, context).await?;
        assert_eq!(
            batches_to_sort_string(&expected),
            batches_to_sort_string(&actual)
        );
    }
    Ok(())
}

#[test]
fn inner_merge_declines_ambiguous_names_for_native_child_remapping() -> Result<()> {
    // Sort and other native children remap pushed columns by name. Preserve
    // the parent predicate when the referenced name is ambiguous in a child.
    let plan = join_with_names(JoinType::Inner, true)?;
    let description = plan.gather_filters_for_pushdown(
        FilterPushdownPhase::Post,
        vec![dynamic(2, "v"), dynamic(5, "v"), dynamic(0, "k")],
        &ConfigOptions::default(),
    )?;
    for side in description.parent_filters() {
        assert!(
            side[..2]
                .iter()
                .all(|f| matches!(f.discriminant, PushedDown::No))
        );
    }
    assert!(matches!(
        description.parent_filters()[0][2].discriminant,
        PushedDown::Yes
    ));
    Ok(())
}

#[test]
fn non_inner_merges_keep_parent_dynamic_filters_above_the_join() -> Result<()> {
    for kind in [
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::LeftSemi,
        JoinType::RightSemi,
        JoinType::LeftAnti,
        JoinType::RightAnti,
        JoinType::LeftMark,
        JoinType::RightMark,
    ] {
        let plan = join(kind)?;
        let predicate: Expr = dynamic(0, "k");
        let description = plan.gather_filters_for_pushdown(
            FilterPushdownPhase::Post,
            vec![predicate],
            &ConfigOptions::default(),
        )?;
        assert!(
            description
                .parent_filters()
                .iter()
                .flatten()
                .all(|p| matches!(p.discriminant, PushedDown::No))
        );
    }
    Ok(())
}

#[test]
fn inner_merge_declines_unmapped_and_non_dynamic_parent_filters() -> Result<()> {
    let plan = join(JoinType::Inner)?;
    let truth: Expr = Arc::new(Literal::new(ScalarValue::Boolean(Some(true))));
    let cross: Expr = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("k", 0)), Arc::new(Column::new("k", 3))],
        Arc::clone(&truth),
    ));
    let computed: Expr = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(BinaryExpr::new(
            Arc::new(Column::new("v", 1)),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int64(Some(1)))),
        ))],
        Arc::clone(&truth),
    ));
    let filters: Vec<Expr> = vec![
        Arc::clone(&truth),
        Arc::new(DynamicFilterPhysicalExpr::new(vec![], truth)),
        cross,
        computed,
        dynamic(0, "v"),
        dynamic(usize::MAX, "v"),
    ];
    let description = plan.gather_filters_for_pushdown(
        FilterPushdownPhase::Post,
        filters.clone(),
        &ConfigOptions::default(),
    )?;
    for child in description.parent_filters() {
        assert_eq!(child.len(), filters.len());
        for (offered, original) in child.iter().zip(&filters) {
            assert!(matches!(offered.discriminant, PushedDown::No));
            assert!(Arc::ptr_eq(&offered.predicate, original));
        }
    }
    let pre = plan.gather_filters_for_pushdown(
        FilterPushdownPhase::Pre,
        vec![dynamic(0, "k")],
        &ConfigOptions::default(),
    )?;
    assert!(
        pre.parent_filters()
            .iter()
            .flatten()
            .all(|p| matches!(p.discriminant, PushedDown::No))
    );
    Ok(())
}

#[test]
fn inner_merge_propagates_a_single_child_acceptance_only_in_post_phase() -> Result<()> {
    use crate::filter_pushdown::{ChildFilterPushdownResult, ChildPushdownResult};

    for (kind, phase, accepted) in [
        (JoinType::Inner, FilterPushdownPhase::Post, true),
        (JoinType::Inner, FilterPushdownPhase::Pre, false),
        (JoinType::Left, FilterPushdownPhase::Post, false),
    ] {
        let plan = join(kind)?;
        let result = ChildPushdownResult {
            parent_filters: vec![ChildFilterPushdownResult {
                filter: dynamic(0, "k"),
                child_results: vec![PushedDown::Yes, PushedDown::No],
            }],
            self_filters: vec![vec![], vec![]],
        };
        let propagated =
            plan.handle_child_pushdown_result(phase, result, &ConfigOptions::default())?;
        assert_eq!(matches!(propagated.filters[0], PushedDown::Yes), accepted);
        assert!(propagated.updated_node.is_none());
    }
    Ok(())
}
