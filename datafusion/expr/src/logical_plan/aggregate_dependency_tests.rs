// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;
use crate::test::function_stub::count;
use crate::{cast, col, lit};

fn aggregate_with_dependencies(
    nullable: bool,
    dependencies: Vec<FunctionalDependence>,
    groups: Vec<Expr>,
) -> Aggregate {
    let schema = DFSchema::try_from(Schema::new(
        (0..4)
            .map(|i| Field::new(format!("c{i}"), DataType::Int64, nullable))
            .collect::<Vec<_>>(),
    ))
    .unwrap()
    .with_functional_dependencies(FunctionalDependencies::new(dependencies))
    .unwrap();
    Aggregate::try_new(
        Arc::new(LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(schema),
        })),
        groups,
        vec![count(lit(true))],
    )
    .unwrap()
}

fn partial(source: usize, targets: Vec<usize>, nullable: bool) -> FunctionalDependence {
    FunctionalDependence::new(vec![source], targets, nullable)
        .with_mode(Dependency::Multi)
}

fn assert_complete_group_key(plan: &Aggregate, count: usize) {
    assert!(
        plan.schema
            .functional_dependencies()
            .iter()
            .any(|dep| dep.source_indices == (0..count).collect::<Vec<_>>()
                && dep.target_indices
                    == (0..plan.schema.fields().len()).collect::<Vec<_>>()
                && dep.mode == Dependency::Single
                && dep.null_equality == NullEquality::NullEqualsNull),
        "{:?}",
        plan.schema.functional_dependencies()
    );
}

#[test]
fn independent_and_computed_group_keys_do_not_promote_partial_dependencies() {
    for independent in [col("c2"), (col("c2") + lit(1_i64)).alias("computed")] {
        let plan = aggregate_with_dependencies(
            false,
            vec![partial(0, vec![0, 1], false)],
            vec![col("c0"), independent],
        );
        let deps = plan.schema.functional_dependencies();
        let dep = deps.iter().find(|dep| dep.source_indices == [0]).unwrap();
        assert_eq!(dep.mode, Dependency::Multi);
        assert_eq!(dep.target_indices, [0]);
        assert_complete_group_key(&plan, 2);
    }
}

#[test]
fn transitive_closure_retains_only_justified_group_targets() {
    let plan = aggregate_with_dependencies(
        false,
        vec![partial(0, vec![0, 1], false), partial(1, vec![1, 2], false)],
        vec![col("c0"), col("c2"), col("c3")],
    );
    let dep = &plan.schema.functional_dependencies()[0];
    assert_eq!(dep.source_indices, [0]);
    assert_eq!(dep.target_indices, [0, 1]);
    assert_eq!(dep.mode, Dependency::Multi);
    assert_complete_group_key(&plan, 3);

    let plan = aggregate_with_dependencies(
        false,
        vec![partial(0, vec![0, 1], false), partial(1, vec![1, 2], false)],
        vec![col("c0"), col("c2")],
    );
    let dep = &plan.schema.functional_dependencies()[0];
    assert_eq!(dep.source_indices, [0]);
    assert_eq!(dep.target_indices, [0, 1, 2]);
    assert_eq!(dep.mode, Dependency::Single);
}

#[test]
fn nullable_dependencies_obey_their_null_equality_model() {
    let unique = FunctionalDependence::new(vec![0], vec![0, 1, 2, 3], true)
        .with_mode(Dependency::Single);
    let plan =
        aggregate_with_dependencies(true, vec![unique], vec![col("c0"), col("c1")]);
    assert_eq!(plan.schema.functional_dependencies().len(), 2);
    let strict_unique = &plan.schema.functional_dependencies()[0];
    assert_eq!(strict_unique.source_indices, [0]);
    assert_eq!(strict_unique.target_indices, [0, 1, 2]);
    assert_eq!(strict_unique.mode, Dependency::Single);
    assert!(strict_unique.nullable);
    assert_eq!(strict_unique.null_equality, NullEquality::NullEqualsNothing);
    assert_complete_group_key(&plan, 2);

    let null_safe =
        partial(0, vec![0, 1], true).with_null_equality(NullEquality::NullEqualsNull);
    let plan =
        aggregate_with_dependencies(true, vec![null_safe], vec![col("c0"), col("c1")]);
    let dep = &plan.schema.functional_dependencies()[0];
    assert_eq!(dep.source_indices, [0]);
    assert_eq!(dep.target_indices, [0, 1, 2]);
    assert_eq!(dep.mode, Dependency::Single);
    assert_eq!(dep.null_equality, NullEquality::NullEqualsNull);
}

#[test]
fn complete_primary_key_stays_unique_with_computed_groups() {
    let key = FunctionalDependence::new(vec![0], vec![0, 1, 2, 3], false)
        .with_mode(Dependency::Single);
    let plan = aggregate_with_dependencies(
        false,
        vec![key],
        vec![col("c0"), (col("c2") + lit(1_i64)).alias("computed")],
    );
    let dep = &plan.schema.functional_dependencies()[0];
    assert_eq!(dep.source_indices, [0]);
    assert_eq!(dep.target_indices, [0, 1, 2]);
    assert_eq!(dep.mode, Dependency::Single);
}

#[test]
fn computed_group_does_not_inherit_same_named_input_key() {
    let key = FunctionalDependence::new(vec![0], vec![0, 1, 2, 3], false)
        .with_mode(Dependency::Single);
    let plan = aggregate_with_dependencies(
        false,
        vec![key],
        vec![cast(col("c0"), DataType::Int32), col("c1")],
    );
    assert_eq!(plan.schema.functional_dependencies().len(), 1);
    assert_complete_group_key(&plan, 2);
}

#[test]
fn composite_keys_keep_output_positions_and_require_every_source() {
    let key = FunctionalDependence::new(vec![0, 1], vec![0, 1, 2], false);
    let plan = aggregate_with_dependencies(
        false,
        vec![key.clone()],
        vec![col("c1"), col("c3"), col("c0")],
    );
    let dep = &plan.schema.functional_dependencies()[0];
    assert_eq!(dep.source_indices, [2, 0]);
    assert_eq!(dep.target_indices, [0, 2]);
    assert_eq!(dep.mode, Dependency::Multi);
    assert_complete_group_key(&plan, 3);
    let plan = aggregate_with_dependencies(false, vec![key], vec![col("c0"), col("c2")]);
    assert_eq!(plan.schema.functional_dependencies().len(), 1);
    assert_complete_group_key(&plan, 2);
}

#[test]
fn derived_dependencies_hold_for_enumerated_joined_groups() {
    let plan = aggregate_with_dependencies(
        false,
        vec![partial(0, vec![0, 1], false)],
        vec![col("c0"), col("c2")],
    );
    for warehouses in 1..=4 {
        for items in 1..=4 {
            let rows = (0..warehouses)
                .flat_map(|warehouse| {
                    (0..items)
                        .map(move |item| vec![warehouse, item, warehouse + item + 1])
                })
                .collect::<Vec<_>>();
            for dep in plan.schema.functional_dependencies().iter() {
                for (index, left) in rows.iter().enumerate() {
                    for right in &rows[index + 1..] {
                        if dep.source_indices.iter().all(|&i| left[i] == right[i]) {
                            assert_eq!(dep.mode, Dependency::Multi, "{dep:?}");
                            assert!(
                                dep.target_indices.iter().all(|&i| left[i] == right[i]),
                                "{dep:?}: {left:?} versus {right:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn global_aggregate_has_no_group_key_dependency() {
    let plan = aggregate_with_dependencies(false, vec![], vec![]);
    assert!(plan.schema.functional_dependencies().is_empty());
}

#[test]
fn nullable_unique_keeps_non_null_uniqueness_with_computed_and_duplicate_groups() {
    for other in [
        col("c1"),
        (col("c2") + lit(1_i64)).alias("computed"),
        col("c0").alias("k"),
    ] {
        let unique = FunctionalDependence::new(vec![0], vec![0, 1, 2, 3], true)
            .with_mode(Dependency::Single);
        let plan =
            aggregate_with_dependencies(true, vec![unique], vec![col("c0"), other]);
        let dep = &plan.schema.functional_dependencies()[0];
        assert_eq!(dep.source_indices, [0]);
        assert_eq!(dep.target_indices, [0, 1, 2]);
        assert_eq!(dep.mode, Dependency::Single);
        assert!(dep.nullable);
        assert_eq!(dep.null_equality, NullEquality::NullEqualsNothing);
        assert!(!dep.is_valid_across_nulls(&plan.schema));
        assert_complete_group_key(&plan, 2);
    }
}

#[test]
fn strict_multi_dependency_does_not_acquire_uniqueness() {
    let plan = aggregate_with_dependencies(
        true,
        vec![partial(0, vec![0, 1], true)],
        vec![col("c0"), col("c1")],
    );
    assert_eq!(plan.schema.functional_dependencies().len(), 1);
    assert_complete_group_key(&plan, 2);
}

#[test]
fn downgraded_left_join_key_is_not_unique_across_independent_groups() -> Result<()> {
    use crate::builder::LogicalTableSource;
    use datafusion_common::Constraint;
    let left_schema = Schema::new(vec![Field::new("x", DataType::Int64, true)]);
    let right_schema = Schema::new(vec![
        Field::new("u", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]);
    let left = crate::table_scan(Some("l"), &left_schema, None)?.build()?;
    let right = LogicalPlanBuilder::scan(
        "r",
        Arc::new(
            LogicalTableSource::new(Arc::new(right_schema)).with_constraints(
                Constraints::new_unverified(vec![Constraint::PrimaryKey(vec![0])]),
            ),
        ),
        None,
    )?
    .build()?;
    let joined = LogicalPlanBuilder::from(left)
        .join(right, JoinType::Left, (vec!["l.x"], vec!["r.u"]), None)?
        .build()?;
    let input_dep = &joined.schema().functional_dependencies()[0];
    assert_eq!(input_dep.mode, Dependency::Multi);
    assert!(input_dep.nullable);
    assert_eq!(input_dep.null_equality, NullEquality::NullEqualsNull);
    let plan = Aggregate::try_new(
        Arc::new(joined),
        vec![col("r.u"), col("l.x")],
        vec![count(lit(true))],
    )?;
    let dep = &plan.schema.functional_dependencies()[0];
    assert_eq!(dep.source_indices, [0]);
    assert_eq!(dep.target_indices, [0]);
    assert_eq!(dep.mode, Dependency::Multi);
    assert!(dep.nullable);
    assert_eq!(dep.null_equality, NullEquality::NullEqualsNull);
    assert_complete_group_key(&plan, 2);
    Ok(())
}
