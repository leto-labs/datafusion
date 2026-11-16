// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Parent dynamic predicates must retain their positions and live state.

use std::sync::Arc;

use super::SortMergeJoinExec;
use crate::expressions::{Column, PhysicalSortExpr};
use crate::filter::FilterExec;
use crate::filter_pushdown::{FilterPushdownPhase, PushedDown};
use crate::sorts::sort::SortExec;
use crate::test::TestMemoryExec;
use crate::{
    ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions,
    collect as collect_plan,
};
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

fn rebuild(
    plan: &Arc<dyn ExecutionPlan>,
    children: Vec<Arc<dyn ExecutionPlan>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    Arc::clone(plan).replace_children(
        children,
        ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
    )
}

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
        let root: Expr = Arc::<DynamicFilterPhysicalExpr>::clone(&parent);
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
        let pushed = rebuild(&plan, children)?;
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

/// Upstream #24517 lets the join embed a projection, so an output position
/// is not an offset into the concatenated inputs. Projection `[3, 1, 5, 0]`
/// emits right `k`, left `v`, right `payload`, left `k`. Offset arithmetic
/// would route output 0 and output 2 to the left input, where a column of
/// the same name exists at the same position.
#[tokio::test]
async fn projected_inner_merge_routes_through_the_embedded_projection() -> Result<()> {
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        SortMergeJoinExec::try_new(
            source(false, false)?,
            source(true, false)?,
            vec![(Arc::new(Column::new("k", 0)), Arc::new(Column::new("k", 0)))],
            None,
            JoinType::Inner,
            vec![SortOptions::default()],
            NullEquality::NullEqualsNull,
        )?
        .with_projection(Some(vec![3, 1, 5, 0]))?,
    );
    let names: Vec<_> = plan
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, ["k", "v", "payload", "k"]);
    // (output position, expected side, expected input position, a value that
    // selects some joined rows only on the expected side)
    for (index, side, input, value) in [
        (0, 1, 0, Some(2)),
        (1, 0, 1, Some(20)),
        (2, 1, 2, Some(201)),
        (3, 0, 0, Some(1)),
        (2, 1, 2, None),
    ] {
        let name = names[index].as_str();
        let parent = dynamic(index, name);
        let root: Expr = Arc::<DynamicFilterPhysicalExpr>::clone(&parent);
        let description = plan.gather_filters_for_pushdown(
            FilterPushdownPhase::Post,
            vec![Arc::clone(&root)],
            &ConfigOptions::default(),
        )?;
        let filters = description.parent_filters();
        assert!(
            matches!(filters[side][0].discriminant, PushedDown::Yes),
            "output {index} must route to side {side}"
        );
        assert!(matches!(filters[1 - side][0].discriminant, PushedDown::No));
        let mapped = Arc::clone(&filters[side][0].predicate);
        assert_eq!(mapped.expression_id(), root.expression_id());
        let mapped_column = mapped.children()[0]
            .downcast_ref::<Column>()
            .unwrap()
            .clone();
        assert_eq!((mapped_column.name(), mapped_column.index()), (name, input));

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
        let pushed = rebuild(&plan, children)?;
        let actual = collect_plan(pushed, context).await?;
        assert_eq!(
            batches_to_sort_string(&expected),
            batches_to_sort_string(&actual)
        );
    }
    Ok(())
}

/// A projection that drops one input's columns leaves nothing to route to
/// that input, and a dropped column is never resolved by name on the other.
#[test]
fn projected_inner_merge_declines_positions_outside_the_projection() -> Result<()> {
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        SortMergeJoinExec::try_new(
            source(false, false)?,
            source(true, false)?,
            vec![(Arc::new(Column::new("k", 0)), Arc::new(Column::new("k", 0)))],
            None,
            JoinType::Inner,
            vec![SortOptions::default()],
            NullEquality::NullEqualsNull,
        )?
        .with_projection(Some(vec![4]))?,
    );
    // Output has one column (right `v`). Position 1 does not exist; a
    // name mismatch at position 0 is stale.
    let description = plan.gather_filters_for_pushdown(
        FilterPushdownPhase::Post,
        vec![dynamic(1, "v"), dynamic(0, "k"), dynamic(0, "v")],
        &ConfigOptions::default(),
    )?;
    let [left, right] = description.parent_filters().try_into().unwrap();
    assert!(
        left.iter()
            .all(|f| matches!(f.discriminant, PushedDown::No))
    );
    assert!(matches!(right[0].discriminant, PushedDown::No));
    assert!(matches!(right[1].discriminant, PushedDown::No));
    assert!(matches!(right[2].discriminant, PushedDown::Yes));
    let mapped = right[2].predicate.children()[0]
        .downcast_ref::<Column>()
        .unwrap()
        .clone();
    assert_eq!((mapped.name(), mapped.index()), ("v", 1));
    Ok(())
}

/// Outer joins keep parent predicates above the join on both sides: below
/// the NULL-padded side a predicate changes which rows are padded, and the
/// propagation never reports a single child's acceptance.
#[tokio::test]
async fn outer_merges_never_route_parent_filters_below_the_join() -> Result<()> {
    use crate::filter_pushdown::{ChildFilterPushdownResult, ChildPushdownResult};

    for kind in [JoinType::Left, JoinType::Right, JoinType::Full] {
        let plan = join(kind)?;
        let filters: Vec<Expr> = (0..6)
            .map(|index| {
                dynamic(index, if index % 3 == 0 { "k" } else { "payload" }) as Expr
            })
            .collect();
        let description = plan.gather_filters_for_pushdown(
            FilterPushdownPhase::Post,
            filters.clone(),
            &ConfigOptions::default(),
        )?;
        for child in description.parent_filters() {
            for (offered, original) in child.iter().zip(&filters) {
                assert!(matches!(offered.discriminant, PushedDown::No), "{kind}");
                assert!(Arc::ptr_eq(&offered.predicate, original));
            }
        }
        for child_results in [
            vec![PushedDown::Yes, PushedDown::No],
            vec![PushedDown::No, PushedDown::Yes],
        ] {
            let result = ChildPushdownResult {
                parent_filters: vec![ChildFilterPushdownResult {
                    filter: dynamic(0, "k"),
                    child_results,
                }],
                self_filters: vec![vec![], vec![]],
            };
            let propagated = plan.handle_child_pushdown_result(
                FilterPushdownPhase::Post,
                result,
                &ConfigOptions::default(),
            )?;
            assert!(matches!(propagated.filters[0], PushedDown::No), "{kind}");
        }
    }

    // Why: a LEFT join's `right.v IS NULL` keeps the padded rows. Evaluated
    // on the right input it removes every right row and pads all left rows.
    let plan = join(JoinType::Left)?;
    let parent = dynamic(4, "v");
    let root: Expr = Arc::<DynamicFilterPhysicalExpr>::clone(&parent);
    parent.update(Arc::new(IsNullExpr::new(Arc::new(Column::new("v", 4)))))?;
    let context = Arc::new(TaskContext::default());
    let unfiltered = collect_plan(Arc::clone(&plan), Arc::clone(&context)).await?;
    let above: usize = unfiltered
        .iter()
        .map(|batch| {
            let mask = root.evaluate(batch)?.into_array(batch.num_rows())?;
            Ok(filter_record_batch(batch, mask.as_boolean())?.num_rows())
        })
        .sum::<Result<usize>>()?;
    let below_view =
        Arc::clone(&root).with_new_children(vec![Arc::new(Column::new("v", 1))])?;
    let mut children = plan.children().into_iter().cloned().collect::<Vec<_>>();
    children[1] = Arc::new(FilterExec::try_new(below_view, Arc::clone(&children[1]))?);
    let below: usize = collect_plan(rebuild(&plan, children)?, context)
        .await?
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_ne!(above, below);
    Ok(())
}
