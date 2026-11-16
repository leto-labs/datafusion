// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Regressions from the former two-scan membership rewrite, on native mark joins.

use super::*;
use crate::OptimizerContext;
use crate::test::test_table_scan_with_name;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_expr::{col, out_ref_col, table_scan};

fn nullable_scan(name: &str) -> Result<LogicalPlan> {
    let schema = Schema::new(vec![Field::new("a", DataType::UInt32, true)]);
    table_scan(Some(name), &schema, None)?.build()
}

fn rewrite_membership(plan: LogicalPlan) -> Result<LogicalPlan> {
    Ok(DecorrelatePredicateSubquery::new()
        .rewrite(plan, &OptimizerContext::new())?
        .data)
}

#[test]
fn null_aware_membership_projection_keeps_names() -> Result<()> {
    for negated in [false, true] {
        let subquery = Arc::new(
            LogicalPlanBuilder::from(nullable_scan("members")?)
                .project(vec![col("members.a")])?
                .build()?,
        );
        let expr = if negated {
            not_in_subquery(col("probe.a"), subquery)
        } else {
            in_subquery(col("probe.a"), subquery)
        };
        let plan = LogicalPlanBuilder::from(test_table_scan_with_name("probe")?)
            .project(vec![col("probe.b"), expr])?
            .build()?;
        let schema = Arc::clone(plan.schema());
        let rewritten = rewrite_membership(plan)?;
        assert_eq!(schema, *rewritten.schema());
        let display = rewritten.display_indent().to_string();
        assert_eq!(display.matches("LeftMark Join").count(), 1, "{display}");
        assert!(display.contains("null_aware"), "{display}");
    }
    Ok(())
}

#[test]
fn null_aware_membership_keeps_identical_correlation_for_unknown_rows() -> Result<()> {
    for negated in [false, true] {
        let subquery = Arc::new(
            LogicalPlanBuilder::from(nullable_scan("members")?)
                .filter(out_ref_col(DataType::UInt32, "probe.a").eq(col("members.a")))?
                .project(vec![col("members.a")])?
                .build()?,
        );
        let expr = if negated {
            not_in_subquery(col("probe.a"), subquery)
        } else {
            in_subquery(col("probe.a"), subquery)
        };
        let plan = LogicalPlanBuilder::from(nullable_scan("probe")?)
            .project(vec![expr])?
            .build()?;
        let rewritten = rewrite_membership(plan)?;
        let display = rewritten.display_indent().to_string();
        assert_eq!(display.matches("LeftMark Join").count(), 1, "{display}");
        // Strict correlation excludes NULL members, so UNKNOWN cannot occur.
        assert!(!display.contains("null_aware"), "{display}");
        assert!(
            display.contains("probe.a = __correlated_sq_1.a"),
            "{display}"
        );
    }
    Ok(())
}

#[test]
fn null_aware_membership_limit_uses_one_scan() -> Result<()> {
    let subquery = Arc::new(
        LogicalPlanBuilder::from(nullable_scan("members")?)
            .limit(0, Some(1))?
            .project(vec![col("members.a")])?
            .build()?,
    );
    let plan = LogicalPlanBuilder::from(nullable_scan("probe")?)
        .project(vec![in_subquery(col("probe.a"), subquery)])?
        .build()?;
    let schema = Arc::clone(plan.schema());
    let rewritten = rewrite_membership(plan)?;
    assert_eq!(schema, *rewritten.schema());
    let display = rewritten.display_indent().to_string();
    assert_eq!(display.matches("LeftMark Join").count(), 1, "{display}");
    assert_eq!(
        display.matches("TableScan: members").count(),
        1,
        "{display}"
    );
    assert!(display.contains("Limit: skip=0, fetch=1"), "{display}");
    assert!(display.contains("null_aware"), "{display}");
    Ok(())
}

#[test]
fn null_aware_membership_nonnullable_uses_one_mark() -> Result<()> {
    let subquery = Arc::new(
        LogicalPlanBuilder::from(test_table_scan_with_name("members")?)
            .project(vec![col("members.a")])?
            .build()?,
    );
    let plan = LogicalPlanBuilder::from(test_table_scan_with_name("probe")?)
        .project(vec![in_subquery(col("probe.a"), subquery)])?
        .build()?;
    let schema = Arc::clone(plan.schema());
    let rewritten = rewrite_membership(plan)?;
    // Native mark joins conservatively mark their output nullable, even
    // when IN's operands are non-nullable. Names and types stay stable; the
    // result fixture checks the two-valued values. The upgrade permits this
    // schema API change instead of retaining the old two-scan implementation.
    assert!(!schema.field(0).is_nullable());
    assert_eq!(schema.columns(), rewritten.schema().columns());
    assert_eq!(
        schema.field(0).data_type(),
        rewritten.schema().field(0).data_type()
    );
    let display = rewritten.display_indent().to_string();
    assert_eq!(display.matches("LeftMark Join").count(), 1, "{display}");
    assert!(!display.contains("null_aware"), "{display}");
    Ok(())
}
