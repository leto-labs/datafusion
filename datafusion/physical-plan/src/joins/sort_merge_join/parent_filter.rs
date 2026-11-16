// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Routes live parent dynamic predicates through an inner sort-merge join.
//!
//! The join may embed a projection (see [`SortMergeJoinExec::with_projection`]),
//! so an output column position is not an offset into the concatenated
//! inputs. Each output position is resolved through the join's column indices
//! and its projection, and the remapping uses the explicit column mapping of
//! [`ChildFilterDescription::from_child_with_column_mapping`].

use std::collections::HashMap;
use std::sync::Arc;

use super::SortMergeJoinExec;
use crate::ExecutionPlan;
use crate::filter_pushdown::{
    ChildFilterDescription, FilterDescription, FilterPushdownPhase, PushedDownPredicate,
};
use crate::joins::utils::{ColumnIndex, build_join_schema};
use arrow::datatypes::Schema;
use datafusion_common::{JoinSide, JoinType, Result};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{Column, DynamicFilterPhysicalExpr};

/// Whether this join routes parent filters in `phase`.
///
/// Only an inner join qualifies. Every output row of an inner join carries
/// one row of each input unchanged, and its join filter only removes rows,
/// so a predicate over one input's columns selects the same joined rows
/// whether it runs above the join or on that input. An outer join pads one
/// side with NULLs and a semi, anti or mark join decides row presence from
/// the other side; those keep their parent filters above the join.
///
/// Only the post-optimization phase routes, as the dynamic filters of a
/// producer above are attached there.
pub(super) fn routes(join_type: JoinType, phase: FilterPushdownPhase) -> bool {
    phase == FilterPushdownPhase::Post && join_type == JoinType::Inner
}

pub(super) fn describe(
    join: &SortMergeJoinExec,
    phase: FilterPushdownPhase,
    filters: &[Arc<dyn PhysicalExpr>],
) -> Result<FilterDescription> {
    let children = join.children();
    if !routes(join.join_type, phase) {
        return Ok(FilterDescription::all_unsupported(filters, &children));
    }
    let (_, column_indices) = build_join_schema(
        join.left.schema().as_ref(),
        join.right.schema().as_ref(),
        &join.join_type,
    );
    // Output position -> input position, through the embedded projection.
    let column_indices: Vec<ColumnIndex> = match join.projection.as_ref() {
        Some(projection) => projection
            .iter()
            .map(|i| column_indices[*i].clone())
            .collect(),
        None => column_indices,
    };
    let (mut left_mapping, mut right_mapping) = (HashMap::new(), HashMap::new());
    for (output, index) in column_indices.iter().enumerate() {
        match index.side {
            JoinSide::Left => {
                left_mapping.insert(output, index.index);
            }
            JoinSide::Right => {
                right_mapping.insert(output, index.index);
            }
            JoinSide::None => {}
        }
    }
    let output = join.schema();
    let mut description = FilterDescription::new();
    for (mapping, child) in [(left_mapping, &join.left), (right_mapping, &join.right)] {
        let input = child.schema();
        let routable = filters
            .iter()
            .map(|filter| routable(filter, output.as_ref(), input.as_ref(), &mapping))
            .collect::<Vec<_>>();
        let mut described = ChildFilterDescription::from_child_with_column_mapping(
            filters, mapping, child,
        )?;
        for ((offered, filter), routable) in described
            .parent_filters
            .iter_mut()
            .zip(filters)
            .zip(routable)
        {
            if !routable {
                *offered = PushedDownPredicate::unsupported(Arc::clone(filter));
            }
        }
        description = description.with_child(described);
    }
    Ok(description)
}

/// Whether `filter` is a live dynamic predicate over plain columns that all
/// come from the input with schema `input`, each with an unambiguous name.
///
/// The predicate is derived as a child view of the same producer (its
/// `with_new_children`), never copied from its current value, so later
/// producer updates reach the input. Native children below may resolve a
/// pushed column by name; a name repeated in the input keeps the predicate
/// above the join instead of letting a child pick another column. A column
/// whose name disagrees with the join output at its position is stale and
/// is not routed either.
fn routable(
    filter: &Arc<dyn PhysicalExpr>,
    output: &Schema,
    input: &Schema,
    mapping: &HashMap<usize, usize>,
) -> bool {
    if !filter.is::<DynamicFilterPhysicalExpr>() || filter.children().is_empty() {
        return false;
    }
    filter.children().into_iter().all(|expression| {
        let Some(column) = expression.downcast_ref::<Column>() else {
            return false;
        };
        let Some(index) = mapping.get(&column.index()) else {
            return false;
        };
        let name = column.name();
        output
            .fields()
            .get(column.index())
            .is_some_and(|field| field.name() == name)
            && input
                .fields()
                .get(*index)
                .is_some_and(|field| field.name() == name)
            && input.fields().iter().filter(|f| f.name() == name).count() == 1
    })
}
