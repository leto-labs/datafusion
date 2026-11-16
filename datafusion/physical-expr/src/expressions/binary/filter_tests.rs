// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;
use crate::expressions::{Column, lit};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Exercise physical expressions with nonzero array offsets and enough columns
/// to take the contiguous-comparison branch even at low selectivity.
fn batch_layouts(batch: &RecordBatch) -> Result<Vec<RecordBatch>> {
    let mut columns = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, array)| (field.name().clone(), Arc::clone(array)))
        .collect::<Vec<_>>();
    for i in 0..30 {
        columns.push((format!("unused_{i}"), Arc::clone(batch.column(1))));
    }
    let wide = RecordBatch::try_from_iter(columns)?;
    let offset = batch.num_rows().min(7);
    let len = batch.num_rows() - offset;
    Ok(vec![
        batch.clone(),
        batch.slice(offset, len),
        wide.clone(),
        wide.slice(offset, len),
    ])
}

#[test]
fn scatter_rejects_wrong_rhs_length() {
    let mask = BooleanArray::from((0..100).map(|i| i % 10 == 7).collect::<Vec<_>>());
    for len in [0, 5, 9, 11, 100] {
        let right = BooleanArray::from(vec![true; len]);
        for fill in [false, true] {
            let error = pre_selection_scatter(&mask, Some(&right), fill).unwrap_err();
            assert!(matches!(
                error,
                datafusion_common::DataFusionError::Internal(_)
            ));
            assert!(
                error.to_string().contains("expected 10 RHS rows"),
                "{error}"
            );
        }
    }
}

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
            for batch in batch_layouts(&batch)? {
                let len = batch.num_rows();
                let left = batch.column(0).as_boolean();
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
                                and_kleene(left, right.as_boolean())?
                            } else {
                                or_kleene(left, right.as_boolean())?
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
    let mut cases = cases;
    cases.extend([
        (
            Arc::new(Float32Array::from(
                (0..70)
                    .map(|i| floats[i % floats.len()].map(|v| v as f32))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            ScalarValue::Float32(Some(-0.0)),
        ),
        (
            Arc::new(
                Decimal128Array::from(
                    (0..70)
                        .map(|i| {
                            decimals[i % decimals.len()]
                                .map(|v| i128::from(v) * 1_000_000_007)
                        })
                        .collect::<Vec<_>>(),
                )
                .with_precision_and_scale(38, 10)?,
            ),
            ScalarValue::Decimal128(Some(0), 38, 10),
        ),
        (
            Arc::new(UInt64Array::from(
                (0..70)
                    .map(|i| [Some(0), Some(1), Some(u64::MAX), None][i % 4])
                    .collect::<Vec<_>>(),
            )),
            ScalarValue::UInt64(Some(1)),
        ),
        (
            Arc::new(
                TimestampNanosecondArray::from(
                    (0..70)
                        .map(|i| [Some(i64::MIN), Some(0), Some(i64::MAX), None][i % 4])
                        .collect::<Vec<_>>(),
                )
                .with_timezone("+05:00"),
            ),
            ScalarValue::TimestampNanosecond(Some(0), Some("+05:00".into())),
        ),
        (
            Arc::new(DurationMicrosecondArray::from(
                (0..70)
                    .map(|i| {
                        [
                            Some(i64::MIN),
                            Some(-1),
                            Some(0),
                            Some(1),
                            Some(i64::MAX),
                            None,
                        ][i % 6]
                    })
                    .collect::<Vec<_>>(),
            )),
            ScalarValue::DurationMicrosecond(Some(0)),
        ),
    ]);
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
            for batch in batch_layouts(&batch)? {
                let len = batch.num_rows();
                let left = batch.column(0).as_boolean();
                for scalar in
                    [scalar.clone(), ScalarValue::try_from(numbers.data_type())?]
                {
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
                        let right = rhs.evaluate(&batch)?.into_array(len)?;
                        let expected = if op == Operator::And {
                            and_kleene(left, right.as_boolean())?
                        } else {
                            or_kleene(left, right.as_boolean())?
                        };
                        let actual =
                            BinaryExpr::new(Arc::new(Column::new("left", 0)), op, rhs)
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

fn fallible_rhs_cases(
    batch: &RecordBatch,
) -> Result<Vec<(&'static str, Arc<dyn PhysicalExpr>)>> {
    use crate::expressions::cast::cast as cast_expr;
    let col = |name, index| Arc::new(Column::new(name, index)) as Arc<dyn PhysicalExpr>;
    let one = || lit(ScalarValue::Int32(Some(1)));
    let bin = |left: Arc<dyn PhysicalExpr>, op, right: Arc<dyn PhysicalExpr>| {
        Arc::new(BinaryExpr::new(left, op, right)) as Arc<dyn PhysicalExpr>
    };
    let division = || bin(one(), Operator::Divide, col("divisor", 1));
    let checked_add = Arc::new(
        BinaryExpr::new(col("big", 3), Operator::Plus, one()).with_fail_on_overflow(true),
    );
    Ok(vec![
        ("division-left", bin(division(), Operator::Eq, one())),
        ("division-right", bin(one(), Operator::Eq, division())),
        (
            "cast-left",
            bin(
                cast_expr(col("text", 2), &batch.schema(), DataType::Int32)?,
                Operator::Eq,
                one(),
            ),
        ),
        (
            "cast-right",
            bin(
                one(),
                Operator::Eq,
                cast_expr(col("text", 2), &batch.schema(), DataType::Int32)?,
            ),
        ),
        ("checked-add", bin(checked_add, Operator::Eq, one())),
        (
            "nested-and",
            bin(
                lit(ScalarValue::Boolean(Some(true))),
                Operator::And,
                bin(division(), Operator::Eq, one()),
            ),
        ),
        (
            "nested-or",
            bin(
                lit(ScalarValue::Boolean(Some(false))),
                Operator::Or,
                bin(division(), Operator::Eq, one()),
            ),
        ),
    ])
}

#[test]
fn short_circuit_keeps_fallible_rhs_errors_suppressed() -> Result<()> {
    for op in [Operator::And, Operator::Or] {
        let selected = |i: usize| i % 10 == 7;
        let left = BooleanArray::from(
            (0..100)
                .map(|i| selected(i) ^ (op == Operator::Or))
                .collect::<Vec<_>>(),
        );
        let batch = RecordBatch::try_from_iter([
            ("left", Arc::new(left) as ArrayRef),
            (
                "divisor",
                Arc::new(Int32Array::from(
                    (0..100).map(|i| i32::from(selected(i))).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "text",
                Arc::new(StringArray::from(
                    (0..100)
                        .map(|i| if selected(i) { "1" } else { "not-a-number" })
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "big",
                Arc::new(Int32Array::from(
                    (0..100)
                        .map(|i| if selected(i) { 0 } else { i32::MAX })
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
        ])?;
        for batch in batch_layouts(&batch)? {
            let len = batch.num_rows();
            let left = batch.column(0).as_boolean();
            for (name, rhs) in fallible_rhs_cases(&batch)? {
                assert!(rhs.evaluate(&batch).is_err(), "{name}: full RHS must fail");
                let expected = if op == Operator::And {
                    left.clone()
                } else {
                    BooleanArray::from(vec![true; len])
                };
                let actual = BinaryExpr::new(Arc::new(Column::new("left", 0)), op, rhs)
                    .evaluate(&batch)?
                    .into_array(len)?;
                assert_eq!(
                    actual.as_boolean(),
                    &expected,
                    "{name} {op}, columns={}",
                    batch.num_columns()
                );
            }
        }
    }
    Ok(())
}
