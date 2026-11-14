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

//! Structural regressions complement the consumer's DuckDB-derived result tests.

use super::*;
use crate::OptimizerContext;
use crate::test::test_table_scan_with_name;
use arrow::datatypes::DataType;
use datafusion_expr::{col, out_ref_col};

#[test]
fn null_aware_membership_projection_keeps_names() -> Result<()> {
    let outer = test_table_scan_with_name("probe")?;
    let members_schema =
        arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
            "a",
            DataType::UInt32,
            true,
        )]);
    let subquery = Arc::new(
        LogicalPlanBuilder::from(
            datafusion_expr::table_scan(Some("members"), &members_schema, None)?
                .build()?,
        )
        .project(vec![col("members.a")])?
        .build()?,
    );
    let plan = LogicalPlanBuilder::from(outer)
        .project(vec![col("probe.b"), in_subquery(col("probe.a"), subquery)])?
        .build()?;
    let schema = Arc::clone(plan.schema());
    let rewritten = DecorrelatePredicateSubquery::new()
        .rewrite(plan, &OptimizerContext::new())?
        .data;
    assert_eq!(schema, *rewritten.schema());
    let display = rewritten.display_indent().to_string();
    assert_eq!(display.matches("LeftMark Join").count(), 2, "{display}");
    assert!(
        display.contains("THEN Boolean(NULL) ELSE Boolean(false)"),
        "{display}"
    );
    Ok(())
}

#[test]
fn null_aware_membership_keeps_identical_correlation_for_unknown_rows() -> Result<()> {
    let outer = test_table_scan_with_name("probe")?;
    let subquery = Arc::new(
        LogicalPlanBuilder::from(test_table_scan_with_name("members")?)
            .filter(out_ref_col(DataType::UInt32, "probe.a").eq(col("members.a")))?
            .project(vec![col("members.a")])?
            .build()?,
    );
    let (plan, _) = membership::mark_join(
        &outer,
        &subquery,
        &col("probe.a").eq(col("members.a")),
        false,
        &Arc::new(AliasGenerator::new()),
    )?
    .expect("decorrelation is supported");
    let LogicalPlan::Join(join) = plan else {
        panic!("expected an unknown mark join")
    };
    let predicate = join
        .filter
        .expect("membership and correlation filter")
        .to_string();
    assert!(predicate.contains("IS NULL AND probe.a ="), "{predicate}");
    Ok(())
}

#[test]
fn null_aware_membership_refuses_repeated_limit() -> Result<()> {
    let outer = test_table_scan_with_name("probe")?;
    let subquery = LogicalPlanBuilder::from(test_table_scan_with_name("members")?)
        .limit(0, Some(1))?
        .project(vec![col("members.a")])?
        .build()?;
    let result = membership::mark_join(
        &outer,
        &subquery,
        &col("probe.a").eq(col("members.a")),
        true,
        &Arc::new(AliasGenerator::new()),
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("requires a repeatable relation")
    );
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
    let rewritten = DecorrelatePredicateSubquery::new()
        .rewrite(plan, &OptimizerContext::new())?
        .data;
    assert_eq!(schema, *rewritten.schema());
    let display = rewritten.display_indent().to_string();
    assert_eq!(display.matches("LeftMark Join").count(), 1, "{display}");
    Ok(())
}
