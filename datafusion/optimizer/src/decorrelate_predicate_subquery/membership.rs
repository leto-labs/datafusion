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

//! Three-valued membership using two ordinary, two-valued existence marks.

use super::build_join;
use datafusion_common::alias::AliasGenerator;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Column, Result, ScalarValue, plan_err};
use datafusion_expr::{Expr, JoinType, LogicalPlan, lit, when};
use std::sync::Arc;

/// Select the membership comparison independently of the correlation filter.
#[derive(Clone, Copy)]
pub(super) enum MembershipTest {
    Equal,
    Unknown,
}

impl MembershipTest {
    pub(super) fn apply(self, comparison: Expr) -> Expr {
        match self {
            Self::Equal => comparison,
            Self::Unknown => comparison.is_null(),
        }
    }
}

/// SQL IN is TRUE if an equality matches, NULL if an eligible comparison is
/// unknown, and FALSE otherwise (including an empty subquery for a NULL probe).
/// Correlation predicates remain filters: their NULLs do not indicate membership
/// uncertainty. NOT IN negates the resulting nullable value, never an existence
/// mark alone.
pub(super) fn mark_join(
    left: &LogicalPlan,
    subquery: &LogicalPlan,
    comparison: &Expr,
    negated: bool,
    aliases: &Arc<AliasGenerator>,
) -> Result<Option<(LogicalPlan, Expr)>> {
    ensure_repeatable(subquery, comparison)?;
    let matched_alias = aliases.next("__correlated_sq");
    let unknown_alias = aliases.next("__correlated_sq");
    let matched = Expr::Column(Column::new(Some(matched_alias.clone()), "mark"));
    let unknown = Expr::Column(Column::new(Some(unknown_alias.clone()), "mark"));
    let Some(matched_join) = build_join(
        left,
        subquery,
        Some(comparison),
        JoinType::LeftMark,
        matched_alias,
        MembershipTest::Equal,
    )?
    else {
        return Ok(None);
    };
    let Some(unknown_join) = build_join(
        &matched_join,
        subquery,
        Some(comparison),
        JoinType::LeftMark,
        unknown_alias,
        MembershipTest::Unknown,
    )?
    else {
        return Ok(None);
    };
    let value = when(matched, lit(true))
        .when(unknown, lit(ScalarValue::Boolean(None)))
        .otherwise(lit(false))?;
    Ok(Some((unknown_join, if negated { !value } else { value })))
}

/// The two marks must observe the same membership set. Refuse inputs that can
/// choose different values on each evaluation. A native nullable mark operator
/// could evaluate these inputs once; ordinary existence marks cannot do so.
fn ensure_repeatable(subquery: &LogicalPlan, comparison: &Expr) -> Result<()> {
    subquery.apply_with_subqueries(|plan| {
        if plan.expressions().iter().any(Expr::is_volatile) {
            return plan_err!("NULL-aware IN subquery does not support a volatile subquery");
        }
        if matches!(plan, LogicalPlan::Limit(_) | LogicalPlan::Window(_)
            | LogicalPlan::Distinct(datafusion_expr::Distinct::On(_))
            | LogicalPlan::Extension(_))
            || matches!(plan, LogicalPlan::Aggregate(agg) if !agg.aggr_expr.is_empty())
            || matches!(plan, LogicalPlan::Sort(sort) if sort.fetch.is_some())
            || matches!(plan, LogicalPlan::TableScan(scan) if scan.fetch.is_some())
        {
            return plan_err!("NULL-aware IN subquery requires a repeatable relation without LIMIT, windows, DISTINCT ON, aggregate functions or extensions");
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if comparison.is_volatile() {
        return plan_err!(
            "NULL-aware IN subquery does not support a volatile comparison"
        );
    }
    if comparison.contains_scalar_subquery() {
        return plan_err!(
            "NULL-aware IN subquery does not support a scalar subquery in its comparison"
        );
    }
    Ok(())
}
