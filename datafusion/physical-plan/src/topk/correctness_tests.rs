// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;
use arrow::array::{Float64Array, Int32Array};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_physical_expr::expressions::col;

#[test]
fn encoded_floating_threshold_is_not_published() -> Result<()> {
    for value in [
        ScalarValue::Dictionary(
            Box::new(DataType::Int32),
            Box::new(ScalarValue::Float32(Some(f32::NAN))),
        ),
        ScalarValue::Float16(None),
        ScalarValue::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int32, false)),
            Arc::new(Field::new("values", DataType::Float64, true)),
            Box::new(ScalarValue::Float64(Some(-f64::NAN))),
        ),
    ] {
        let schema = Schema::new(vec![Field::new("f", value.data_type(), true)]);
        let key = PhysicalSortExpr::new_default(col("f", &schema)?);
        assert!(TopK::build_filter_expression(&[key], &[value])?.is_none());
    }
    Ok(())
}

#[test]
fn floating_topk_threshold_keeps_prefix_ties() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("f", DataType::Float64, true),
    ]));
    for descending in [false, true] {
        for nulls_first in [false, true] {
            let key = PhysicalSortExpr::new(
                col("f", &schema)?,
                SortOptions {
                    descending,
                    nulls_first,
                },
            );
            for value in [None, Some(1.0), Some(f64::NAN)] {
                // Parquet min/max omit NaN. No float threshold alone can prove
                // that a later row group contains no better row.
                assert!(
                    TopK::build_filter_expression(
                        std::slice::from_ref(&key),
                        &[ScalarValue::Float64(value)]
                    )?
                    .is_none()
                );
                let predicate = TopK::build_filter_expression(
                    &[
                        PhysicalSortExpr::new_default(col("k", &schema)?),
                        key.clone(),
                    ],
                    &[ScalarValue::Int32(Some(3)), ScalarValue::Float64(value)],
                )?
                .unwrap();
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int32Array::from(vec![2, 3, 3, 4])),
                        Arc::new(Float64Array::from(vec![
                            Some(0.0),
                            Some(f64::NAN),
                            None,
                            Some(1.0),
                        ])),
                    ],
                )?;
                let result = predicate.evaluate(&batch)?.into_array(4)?;
                assert_eq!(
                    result.as_boolean().iter().collect::<Vec<_>>(),
                    vec![Some(true), Some(true), Some(true), Some(false)]
                );
            }
        }
    }
    Ok(())
}
