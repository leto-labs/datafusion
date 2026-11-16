// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::EquivalenceProperties;
use crate::{PhysicalExprRef, equivalence::OrderingEquivalenceClass};

use arrow::datatypes::SchemaRef;
use datafusion_common::{JoinType, Result};

/// Calculate ordering equivalence properties for the given join operation.
pub fn join_equivalence_properties(
    left: EquivalenceProperties,
    right: EquivalenceProperties,
    join_type: &JoinType,
    join_schema: SchemaRef,
    maintains_input_order: &[bool],
    on: &[(PhysicalExprRef, PhysicalExprRef)],
) -> Result<EquivalenceProperties> {
    let left_size = left.schema.fields.len();
    let mut result = EquivalenceProperties::new(join_schema);
    result.add_equivalence_group(left.eq_group().join(
        right.eq_group(),
        join_type,
        left_size,
        on,
    )?)?;

    let EquivalenceProperties {
        oeq_class: mut left_oeq_class,
        ..
    } = left;
    let EquivalenceProperties {
        oeq_class: mut right_oeq_class,
        ..
    } = right;
    // Unmatched rows insert NULLs on a padded side. Its input ordering is
    // not a valid ordering of those output values, including as a suffix.
    if matches!(join_type, JoinType::Right | JoinType::Full) {
        left_oeq_class.clear();
    }
    if matches!(join_type, JoinType::Left | JoinType::Full) {
        right_oeq_class.clear();
    }
    match maintains_input_order {
        // Maintaining probe order does not order matches from different probe
        // rows with equal sort keys. Appending build keys would falsely claim
        // a lexicographic ordering across those ties.
        [true, false] => result.add_orderings(left_oeq_class),
        [false, true] => {
            updated_right_ordering_equivalence_class(
                &mut right_oeq_class,
                join_type,
                left_size,
            )?;
            result.add_orderings(right_oeq_class);
        }
        [false, false] => {}
        [true, true] => unreachable!("Cannot maintain ordering of both sides"),
        _ => unreachable!("Join operators can not have more than two children"),
    }
    Ok(result)
}

/// In the context of a join, update the right side `OrderingEquivalenceClass`
/// so that they point to valid indices in the join output schema.
///
/// To do so, we increment column indices by the size of the left table when
/// join schema consists of a combination of the left and right schemas. This
/// is the case for `Inner`, `Left`, `Full` and `Right` joins. For other cases,
/// indices do not change.
pub fn updated_right_ordering_equivalence_class(
    right_oeq_class: &mut OrderingEquivalenceClass,
    join_type: &JoinType,
    left_size: usize,
) -> Result<()> {
    if matches!(
        join_type,
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Right
    ) {
        right_oeq_class.add_offset(left_size as _)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::equivalence::convert_to_orderings;
    use crate::equivalence::tests::create_test_schema;
    use crate::expressions::col;

    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Field, Fields, Schema};

    #[test]
    fn padded_join_side_does_not_supply_orderings() -> Result<()> {
        use crate::{PhysicalSortExpr, expressions::Column};
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        for join_type in [
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::LeftSemi,
            JoinType::RightSemi,
            JoinType::LeftAnti,
            JoinType::RightAnti,
        ] {
            for right in [false, true] {
                if matches!(
                    (join_type, right),
                    (JoinType::LeftSemi | JoinType::LeftAnti, true)
                        | (JoinType::RightSemi | JoinType::RightAnti, false)
                ) {
                    continue;
                }
                let ordered = || {
                    let mut properties = EquivalenceProperties::new(Arc::clone(&schema));
                    properties.add_ordering([PhysicalSortExpr::new_default(Arc::new(
                        Column::new("a", 0),
                    ))]);
                    properties
                };
                let out_schema = if matches!(
                    join_type,
                    JoinType::Left | JoinType::Right | JoinType::Full
                ) {
                    Arc::new(Schema::new(vec![
                        schema.field(0).clone(),
                        Field::new("b", DataType::Int64, true),
                    ]))
                } else {
                    Arc::clone(&schema)
                };
                let output = join_equivalence_properties(
                    ordered(),
                    ordered(),
                    &join_type,
                    out_schema,
                    &[!right, right],
                    &[],
                )?;
                let padded = matches!(
                    (join_type, right),
                    (JoinType::Left, true)
                        | (JoinType::Right, false)
                        | (JoinType::Full, _)
                );
                assert_eq!(
                    output.oeq_class.is_empty(),
                    padded,
                    "{join_type:?} right={right}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_join_equivalence_properties() -> Result<()> {
        let schema = create_test_schema()?;
        let columns = [col("a", &schema)?, col("b", &schema)?, col("c", &schema)?];
        let options = SortOptions::default();
        for width in [2, 3] {
            for right_probe in [false, true] {
                let mut left = EquivalenceProperties::new(Arc::clone(&schema));
                let mut right = EquivalenceProperties::new(Arc::clone(&schema));
                let orderings = convert_to_orderings(
                    &columns[..width]
                        .iter()
                        .map(|c| vec![(c, options)])
                        .collect::<Vec<_>>(),
                );
                left.add_orderings(orderings.clone());
                right.add_orderings(orderings.clone());
                let mut expected = OrderingEquivalenceClass::from(orderings);
                if right_probe {
                    expected.add_offset(schema.fields.len() as _)?;
                }
                let result = join_equivalence_properties(
                    left,
                    right,
                    &JoinType::Inner,
                    Arc::new(Schema::empty()),
                    &[!right_probe, right_probe],
                    &[],
                )?;
                assert_eq!(result.oeq_class(), &expected);
            }
        }
        Ok(())
    }

    #[test]
    fn test_get_updated_right_ordering_equivalence_properties() -> Result<()> {
        let join_type = JoinType::Inner;
        // Join right child schema
        let child_fields: Fields = ["x", "y", "z", "w"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();
        let child_schema = Schema::new(child_fields);
        let col_x = &col("x", &child_schema)?;
        let col_y = &col("y", &child_schema)?;
        let col_z = &col("z", &child_schema)?;
        let col_w = &col("w", &child_schema)?;
        let option_asc = SortOptions {
            descending: false,
            nulls_first: false,
        };
        // [x ASC, y ASC], [z ASC, w ASC]
        let orderings = vec![
            vec![(col_x, option_asc), (col_y, option_asc)],
            vec![(col_z, option_asc), (col_w, option_asc)],
        ];
        let orderings = convert_to_orderings(&orderings);
        // Right child ordering equivalences
        let mut right_oeq_class = OrderingEquivalenceClass::from(orderings);

        let left_columns_len = 4;

        let fields: Fields = ["a", "b", "c", "d", "x", "y", "z", "w"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int32, true))
            .collect();

        // Join Schema
        let schema = Schema::new(fields);
        let col_a = col("a", &schema)?;
        let col_d = col("d", &schema)?;
        let col_x = col("x", &schema)?;
        let col_y = col("y", &schema)?;
        let col_z = col("z", &schema)?;
        let col_w = col("w", &schema)?;

        let mut join_eq_properties = EquivalenceProperties::new(Arc::new(schema));
        // a=x and d=w
        join_eq_properties.add_equal_conditions(col_a, Arc::clone(&col_x))?;
        join_eq_properties.add_equal_conditions(col_d, Arc::clone(&col_w))?;

        updated_right_ordering_equivalence_class(
            &mut right_oeq_class,
            &join_type,
            left_columns_len,
        )?;
        join_eq_properties.add_orderings(right_oeq_class);
        let result = join_eq_properties.oeq_class().clone();

        // [x ASC, y ASC], [z ASC, w ASC]
        let orderings = vec![
            vec![(col_x, option_asc), (col_y, option_asc)],
            vec![(col_z, option_asc), (col_w, option_asc)],
        ];
        let orderings = convert_to_orderings(&orderings);
        let expected = OrderingEquivalenceClass::from(orderings);

        assert_eq!(result, expected);

        Ok(())
    }
}
