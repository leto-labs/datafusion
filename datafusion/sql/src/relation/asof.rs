// Added by the Longbow authors (family AJ); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0

//! Planning of DuckDB's ASOF join forms.
//!
//! [`crate::asof_tokens`] rewrites `ASOF [INNER | LEFT [OUTER]] JOIN r ON ...`
//! and `... USING (...)` to `ASOF JOIN r MATCH_CONDITION (<marker>) ...`. This
//! module plans such a join to [`LogicalPlan::AsOfJoin`]:
//!
//! - `ON`: a conjunction. Exactly one conjunct is `<`, `<=`, `>` or `>=`
//!   between an expression of each side, in either operand order; it becomes
//!   the match condition, normalized to `left op right`. Every other conjunct
//!   is an equality between the two sides.
//! - `USING (k, ..., t)`: equality on every column but the last, and
//!   `left.t >= right.t` on the last
//!   ([`datafusion_expr::logical_plan::AsOfJoin::ordered_using`]).
//! - Inner (`ASOF JOIN`, `ASOF INNER JOIN`): the right input gets a constant
//!   marker column, a filter above the join keeps the rows whose marker is not
//!   NULL, and a projection removes it. Neither a right column nor the match
//!   expression can serve as the marker, because either can be NULL in a
//!   matched row.

use crate::asof_tokens::{ASOF_INNER_MARKER, ASOF_LEFT_MARKER};
use crate::expr::reject_window_functions;
use crate::planner::{ContextProvider, PlannerContext, SqlToRel};
use datafusion_common::{Column, DFSchema, HashMap, Result, plan_err};
use datafusion_expr::utils::{conjunction, split_conjunction_owned};
use datafusion_expr::{
    BinaryExpr, Expr, LogicalPlan, LogicalPlanBuilder, Operator, col, lit,
};
use sqlparser::ast::{Expr as SQLExpr, JoinConstraint, ObjectName};

/// The constant column that marks a matched right row of an inner ASOF join.
const MATCHED: &str = "__datafusion_asof_matched";

/// The DuckDB join kind that a match condition marks, if it is a marker.
pub(super) fn duckdb_kind(match_condition: &SQLExpr) -> Option<bool> {
    let SQLExpr::Identifier(ident) = match_condition else {
        return None;
    };
    if ident.quote_style.is_some() {
        return None;
    }
    match ident.value.as_str() {
        ASOF_INNER_MARKER => Some(true),
        ASOF_LEFT_MARKER => Some(false),
        _ => None,
    }
}

fn refers_only_to(expr: &Expr, schema: &DFSchema) -> bool {
    let columns = expr.column_refs();
    !columns.is_empty() && columns.iter().all(|c| schema.is_column_from_schema(c))
}

impl<S: ContextProvider> SqlToRel<'_, S> {
    pub(super) fn parse_duckdb_asof_join(
        &self,
        left: LogicalPlan,
        right: LogicalPlan,
        inner: bool,
        constraint: JoinConstraint,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let right = if inner {
            let mut exprs: Vec<Expr> = right
                .schema()
                .columns()
                .into_iter()
                .map(Expr::Column)
                .collect();
            exprs.push(lit(true).alias(MATCHED));
            LogicalPlanBuilder::from(right).project(exprs)?.build()?
        } else {
            right
        };
        let plan = match constraint {
            JoinConstraint::On(sql_on) => {
                let join_schema = left.schema().join(right.schema())?;
                let window_span = self.window_function_span(&sql_on, &HashMap::new());
                let on = self.sql_to_expr(sql_on, &join_schema, planner_context)?;
                reject_window_functions(
                    &on,
                    "JOIN ON",
                    super::join::JOIN_ON_HELP,
                    window_span,
                )?;
                let (ordered, equal): (Vec<_>, Vec<_>) = split_conjunction_owned(on)
                    .into_iter()
                    .partition(|e| {
                        matches!(e, Expr::BinaryExpr(BinaryExpr { op, .. })
                            if matches!(op, Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq))
                    });
                let [Expr::BinaryExpr(BinaryExpr { left: a, op, right: b })] =
                    <[Expr; 1]>::try_from(ordered).map_err(|ordered| {
                        datafusion_common::plan_datafusion_err!(
                            "ASOF JOIN ON requires exactly one inequality (<, <=, >, >=) between the two sides, found {}",
                            ordered.len()
                        )
                    })?
                else {
                    unreachable!("partitioned on binary expressions");
                };
                let (a, op, b) = if refers_only_to(&a, left.schema())
                    && refers_only_to(&b, right.schema())
                {
                    (*a, op, *b)
                } else if refers_only_to(&b, left.schema())
                    && refers_only_to(&a, right.schema())
                {
                    (*b, op.swap().expect("comparisons swap"), *a)
                } else {
                    return plan_err!(
                        "The ASOF JOIN inequality must compare an expression of the left input with one of the right input"
                    );
                };
                let match_condition =
                    Expr::BinaryExpr(BinaryExpr::new(Box::new(a), op, Box::new(b)));
                LogicalPlanBuilder::from(left)
                    .asof_join_on(right, conjunction(equal), match_condition)?
                    .build()?
            }
            JoinConstraint::Using(object_names) => {
                let mut names = object_names
                    .into_iter()
                    .map(|ObjectName(parts)| match parts.as_slice() {
                        [part] => part
                            .as_ident()
                            .map(|ident| self.ident_normalizer.normalize(ident.clone()))
                            .ok_or_else(|| {
                                datafusion_common::plan_datafusion_err!(
                                    "Expected identifier in ASOF USING clause"
                                )
                            }),
                        _ => plan_err!("ASOF USING columns must be single identifiers"),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let Some(last) = names.pop() else {
                    return plan_err!("ASOF JOIN USING requires at least one column");
                };
                let side = |schema: &DFSchema| -> Result<Expr> {
                    let (qualifier, field) =
                        schema.qualified_field_with_unqualified_name(&last)?;
                    Ok(Expr::Column(Column::from((qualifier, field))))
                };
                let match_condition = side(left.schema())?.gt_eq(side(right.schema())?);
                let keys = names.into_iter().map(Column::from_name).collect();
                let LogicalPlan::AsOfJoin(join) = LogicalPlanBuilder::from(left)
                    .asof_join_using(right, keys, match_condition)?
                    .build()?
                else {
                    unreachable!("the builder returns the ASOF join");
                };
                LogicalPlan::AsOfJoin(join.with_ordered_using(true)?)
            }
            JoinConstraint::None => {
                return plan_err!("ASOF JOIN requires ON or USING");
            }
            JoinConstraint::Natural => {
                return plan_err!("NATURAL ASOF JOIN is not supported");
            }
        };
        if !inner {
            return Ok(plan);
        }
        let columns: Vec<Expr> = plan
            .schema()
            .columns()
            .into_iter()
            .filter(|c| !(c.relation.is_none() && c.name == MATCHED))
            .map(Expr::Column)
            .collect();
        LogicalPlanBuilder::from(plan)
            .filter(col(MATCHED).is_not_null())?
            .project(columns)?
            .build()
    }
}
