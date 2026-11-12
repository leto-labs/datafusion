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
use crate::expressions::{Column, lit};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn scatter_matches_row_reference() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(0x00f2_2026_1111);
    for len in [0, 1, 7, 8, 9, 63, 64, 65, 127, 8192] {
        for offset in [0, 1, 7, 63] {
            for probability in [0.0, 0.01, 0.2, 0.5, 0.99, 1.0] {
                let mask = BooleanArray::from(
                    (0..len + offset)
                        .map(|_| rng.random_bool(probability))
                        .collect::<Vec<_>>(),
                )
                .slice(offset, len);
                let selected = mask.true_count();
                for nullable in [false, true] {
                    let rhs = BooleanArray::from(
                        (0..selected + offset)
                            .map(|_| {
                                if nullable && rng.random_bool(0.3) {
                                    None
                                } else {
                                    Some(rng.random_bool(0.5))
                                }
                            })
                            .collect::<Vec<_>>(),
                    )
                    .slice(offset, selected);
                    for fill in [false, true] {
                        for right in [Some(&rhs), None] {
                            let mut position = 0;
                            let expected = BooleanArray::from(
                                mask.values()
                                    .iter()
                                    .map(|selected| {
                                        if selected {
                                            let value = right.and_then(|r| {
                                                (!r.is_null(position))
                                                    .then(|| r.value(position))
                                            });
                                            position += 1;
                                            value
                                        } else {
                                            Some(fill)
                                        }
                                    })
                                    .collect::<Vec<_>>(),
                            );
                            let actual = pre_selection_scatter(&mask, right, fill)?
                                .into_array(len)?;
                            assert_eq!(actual.as_boolean(), &expected);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn logical_filters_match_kleene_with_nulls() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(0x0003_2026_1111);
    for len in [0, 1, 63, 65, 8192] {
        for nullable in [false, true] {
            for probability in [0.01, 0.2, 0.5, 0.8, 0.99] {
                let mut values = || {
                    BooleanArray::from(
                        (0..len)
                            .map(|_| {
                                if nullable && rng.random_bool(0.1) {
                                    None
                                } else {
                                    Some(rng.random_bool(probability))
                                }
                            })
                            .collect::<Vec<_>>(),
                    )
                };
                let left = values();
                let right = values();
                let batch = RecordBatch::try_from_iter([
                    ("left", Arc::new(left.clone()) as ArrayRef),
                    ("right", Arc::new(right.clone()) as ArrayRef),
                ])?;
                for op in [Operator::And, Operator::Or] {
                    let expr = BinaryExpr::new(
                        Arc::new(Column::new("left", 0)),
                        op,
                        Arc::new(Column::new("right", 1)),
                    );
                    let expected = if op == Operator::And {
                        and_kleene(&left, &right)?
                    } else {
                        or_kleene(&left, &right)?
                    };
                    assert_eq!(
                        expr.evaluate(&batch)?.into_array(len)?.as_boolean(),
                        &expected
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn primitive_comparison_filters_match_kleene() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(0xf2c0_2026_1111);
    for len in [0, 1, 63, 64, 65, 8192] {
        for probability in [0.01, 0.2, 0.5, 0.8, 0.99] {
            let left = BooleanArray::from(
                (0..len)
                    .map(|_| rng.random_bool(probability))
                    .collect::<Vec<_>>(),
            );
            let numbers = Int32Array::from(
                (0..len)
                    .map(|_| {
                        if rng.random_bool(0.1) {
                            None
                        } else {
                            Some(rng.random_range(-10..10))
                        }
                    })
                    .collect::<Vec<_>>(),
            );
            let other: Int32Array = numbers.unary(|v| -v);
            let batch = RecordBatch::try_from_iter([
                ("left", Arc::new(left.clone()) as ArrayRef),
                ("numbers", Arc::new(numbers) as ArrayRef),
                ("other", Arc::new(other) as ArrayRef),
            ])?;
            for comparison in [
                Operator::Eq,
                Operator::NotEq,
                Operator::Lt,
                Operator::LtEq,
                Operator::Gt,
                Operator::GtEq,
            ] {
                for operand in [
                    lit(ScalarValue::Int32(Some(3))),
                    lit(ScalarValue::Int32(None)),
                    Arc::new(Column::new("other", 2)) as Arc<dyn PhysicalExpr>,
                ] {
                    let rhs = Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("numbers", 1)),
                        comparison,
                        operand,
                    ));
                    let right = rhs.evaluate(&batch)?.into_array(len)?;
                    for op in [Operator::And, Operator::Or] {
                        let expected = if op == Operator::And {
                            and_kleene(&left, right.as_boolean())?
                        } else {
                            or_kleene(&left, right.as_boolean())?
                        };
                        let actual = BinaryExpr::new(
                            Arc::new(Column::new("left", 0)),
                            op,
                            Arc::<BinaryExpr>::clone(&rhs),
                        )
                        .evaluate(&batch)?
                        .into_array(len)?;
                        assert_eq!(actual.as_boolean(), &expected);
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn primitive_edge_values_match_full_evaluation() -> Result<()> {
    let floats = [
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
        Some(-0.0),
        Some(0.0),
        Some(1.0),
        None,
    ];
    let decimals = [
        Some(-999_999_999_999_999_999),
        Some(-1),
        Some(0),
        Some(1),
        Some(999_999_999_999_999_999),
        None,
    ];
    let dates = [Some(-1), Some(0), Some(1), Some(366), None];
    let cases: Vec<(ArrayRef, ScalarValue)> = vec![
        (
            Arc::new(Float64Array::from(
                (0..70)
                    .map(|i| floats[i % floats.len()])
                    .collect::<Vec<_>>(),
            )),
            ScalarValue::Float64(Some(0.0)),
        ),
        (
            Arc::new(
                Decimal64Array::from(
                    (0..70)
                        .map(|i| decimals[i % decimals.len()])
                        .collect::<Vec<_>>(),
                )
                .with_precision_and_scale(18, 2)?,
            ),
            ScalarValue::Decimal64(Some(0), 18, 2),
        ),
        (
            Arc::new(Date32Array::from(
                (0..70).map(|i| dates[i % dates.len()]).collect::<Vec<_>>(),
            )),
            ScalarValue::Date32(Some(0)),
        ),
    ];
    for (numbers, scalar) in cases {
        for op in [Operator::And, Operator::Or] {
            let left = BooleanArray::from(
                (0..70)
                    .map(|i| (i % 11 == 3) ^ (op == Operator::Or))
                    .collect::<Vec<_>>(),
            );
            let batch = RecordBatch::try_from_iter([
                ("left", Arc::new(left.clone()) as ArrayRef),
                ("numbers", Arc::clone(&numbers)),
            ])?;
            for comparison in [
                Operator::Eq,
                Operator::NotEq,
                Operator::Lt,
                Operator::LtEq,
                Operator::Gt,
                Operator::GtEq,
            ] {
                let rhs = Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("numbers", 1)),
                    comparison,
                    lit(scalar.clone()),
                ));
                let right = rhs.evaluate(&batch)?.into_array(70)?;
                let expected = if op == Operator::And {
                    and_kleene(&left, right.as_boolean())?
                } else {
                    or_kleene(&left, right.as_boolean())?
                };
                let actual = BinaryExpr::new(Arc::new(Column::new("left", 0)), op, rhs)
                    .evaluate(&batch)?
                    .into_array(70)?;
                assert_eq!(actual.as_boolean(), &expected);
            }
        }
    }
    Ok(())
}

#[test]
fn short_circuit_keeps_division_errors_suppressed() -> Result<()> {
    for op in [Operator::And, Operator::Or] {
        let batch = RecordBatch::try_from_iter([
            (
                "left",
                Arc::new(BooleanArray::from(
                    (0..100)
                        .map(|i| (i % 10 == 7) ^ (op == Operator::Or))
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "divisor",
                Arc::new(Int32Array::from(
                    (0..100).map(|i| i32::from(i % 10 == 7)).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
        ])?;
        let division = Arc::new(BinaryExpr::new(
            lit(ScalarValue::Int32(Some(1))),
            Operator::Divide,
            Arc::new(Column::new("divisor", 1)),
        ));
        let rhs = Arc::new(BinaryExpr::new(
            division,
            Operator::Eq,
            lit(ScalarValue::Int32(Some(1))),
        ));
        assert!(rhs.evaluate(&batch).is_err());
        let expr = BinaryExpr::new(Arc::new(Column::new("left", 0)), op, rhs);
        let actual = expr.evaluate(&batch)?.into_array(100)?;
        assert_eq!(
            actual.as_boolean().true_count(),
            if op == Operator::And { 10 } else { 100 }
        );
    }
    Ok(())
}
