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

use super::*;
use crate::expressions::Literal;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::ScalarValue;

fn project(constraint: Constraint, indices: &[Option<usize>]) -> Constraints {
    let schema = Arc::new(Schema::new(
        (0..4)
            .map(|i| Field::new(format!("c{i}"), DataType::Int32, false))
            .collect::<Vec<_>>(),
    ));
    let properties = EquivalenceProperties::new(schema.clone())
        .with_constraints(Constraints::new_unverified(vec![constraint]));
    let exprs = indices.iter().enumerate().map(|(i, index)| {
        let expr: Arc<dyn PhysicalExpr> = match index {
            Some(index) => Arc::new(Column::new(&format!("c{index}"), *index)),
            None => Arc::new(Literal::new(ScalarValue::Int32(Some(7)))),
        };
        (expr, format!("out{i}"))
    });
    let mapping = ProjectionMapping::try_new(exprs, &schema).unwrap();
    let output = Arc::new(Schema::new(
        (0..indices.len())
            .map(|i| Field::new(format!("out{i}"), DataType::Int32, false))
            .collect::<Vec<_>>(),
    ));
    properties.project(&mapping, output).constraints().clone()
}

#[test]
fn retained_composite_key() {
    for key in [
        Constraint::PrimaryKey(vec![1, 3]),
        Constraint::Unique(vec![1, 3]),
    ] {
        let expected = match key {
            Constraint::PrimaryKey(_) => Constraint::PrimaryKey(vec![0, 1]),
            _ => Constraint::Unique(vec![0, 1]),
        };
        assert_eq!(
            project(key, &[Some(1), Some(3)]),
            Constraints::new_unverified(vec![expected])
        );
    }
}

#[test]
fn discarded_key_is_not_payload_uniqueness() {
    for key in [Constraint::PrimaryKey(vec![0]), Constraint::Unique(vec![0])] {
        assert!(project(key, &[Some(2)]).is_empty());
    }
}

#[test]
fn permuted_dropped_duplicated_and_computed_projections() {
    // Compare every projection of up to four expressions with an independent
    // source-to-output lookup. Include computed expressions and repeated columns.
    for len in 0..=4u32 {
        for mut code in 0..5usize.pow(len) {
            let indices = (0..len)
                .map(|_| {
                    let digit = code % 5;
                    code /= 5;
                    (digit < 4).then_some(digit)
                })
                .collect::<Vec<_>>();
            for source_key in [vec![0], vec![1, 3], vec![0, 2, 3]] {
                let outputs = source_key
                    .iter()
                    .map(|index| indices.iter().position(|value| *value == Some(*index)))
                    .collect::<Option<Vec<_>>>();
                for primary in [false, true] {
                    let make = |key| {
                        if primary {
                            Constraint::PrimaryKey(key)
                        } else {
                            Constraint::Unique(key)
                        }
                    };
                    let expected = outputs
                        .clone()
                        .map(|key| Constraints::new_unverified(vec![make(key)]))
                        .unwrap_or_default();
                    assert_eq!(
                        project(make(source_key.clone()), &indices),
                        expected,
                        "{indices:?}, {source_key:?}, primary={primary}"
                    );
                }
            }
        }
    }
}

#[test]
fn field_access_targets_do_not_make_the_container_unique() {
    use crate::projection::ProjectionTargets;

    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let key: Arc<dyn PhysicalExpr> = Arc::new(Column::new("key", 0));
    let computed: Arc<dyn PhysicalExpr> =
        Arc::new(Literal::new(ScalarValue::Int32(Some(7))));
    let direct: Arc<dyn PhysicalExpr> = Arc::new(Column::new("alias", 2));
    // A non-Column target models a field access introduced by struct mapping.
    // Only the direct alias at output 2 preserves uniqueness of the source key.
    let mapping = ProjectionMapping::from_iter([(
        key,
        ProjectionTargets::from(vec![(computed, 0), (direct, 2)]),
    )]);
    let output = Arc::new(Schema::new(vec![
        Field::new("container", DataType::Int32, false),
        Field::new("unmapped", DataType::Int32, false),
        Field::new("alias", DataType::Int32, false),
    ]));
    let projected = EquivalenceProperties::new(schema)
        .with_constraints(Constraints::new_unverified(vec![Constraint::PrimaryKey(
            vec![0],
        )]))
        .project(&mapping, output);
    assert_eq!(
        projected.constraints(),
        &Constraints::new_unverified(vec![Constraint::PrimaryKey(vec![2])])
    );
}
