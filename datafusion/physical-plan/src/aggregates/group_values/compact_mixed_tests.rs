// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the Longbow authors.

//! Characterize compact row keys against the column interner before dispatch.

use super::{GroupValues, GroupValuesRows, multi_group_by::GroupValuesColumn};
use arrow::{
    array::{ArrayRef, Float64Array, Int64Array, StringViewArray},
    datatypes::{DataType, Field, Schema},
};
use datafusion_common::Result;
use datafusion_expr::EmitTo;
use std::sync::Arc;

#[test]
fn compact_rows_match_column_keys_nulls_float_bits_and_emission() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, true),
        Field::new("f", DataType::Float64, true),
        Field::new("s", DataType::Utf8View, true),
    ]));
    let mut rows = GroupValuesRows::try_new(Arc::clone(&schema))?;
    let mut columns = GroupValuesColumn::<false>::try_new(schema)?;
    let mut seed = 71u64;
    for batch in 0..12 {
        let mut ids = Vec::new();
        for _ in 0..8192 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ids.push((seed >> 32) as usize % 1024);
        }
        let keys: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter(ids.iter().map(|&id| {
                (id % 11 != 0).then_some((id as i64).wrapping_mul(i64::MAX / 1024))
            }))),
            Arc::new(Float64Array::from_iter(ids.iter().map(|&id| {
                (id % 13 != 0).then(|| match id % 7 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f64::from_bits(0x7ff8000000000001),
                    3 => f64::from_bits(0x7ff8000000000002),
                    _ => (id % 60) as f64,
                })
            }))),
            Arc::new(StringViewArray::from_iter(ids.iter().map(|&id| {
                (id % 17 != 0).then(|| {
                    if batch % 2 == 0 {
                        format!("s{}", id % 31)
                    } else {
                        format!("external-dictionary-value-{}", id % 31)
                    }
                })
            }))),
        ];
        let (mut actual, mut expected) = (Vec::new(), Vec::new());
        rows.intern(&keys, &mut actual)?;
        columns.intern(&keys, &mut expected)?;
        assert_eq!(actual, expected, "batch {batch}");
        assert_eq!(rows.len(), columns.len());
        if batch == 3 || batch == 7 {
            let emit = if batch == 3 {
                EmitTo::First(17)
            } else {
                EmitTo::All
            };
            rows.emit(emit)?;
            columns.emit(emit)?;
        }
    }
    Ok(())
}

#[test]
fn compact_factory_emits_empty_keys_before_input_and_after_clear() -> Result<()> {
    use super::{super::order::GroupOrdering, new_group_values};
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, true),
        Field::new("f", DataType::Float64, true),
        Field::new("s", DataType::Utf8View, true),
    ]));
    let mut candidate = new_group_values(Arc::clone(&schema), &GroupOrdering::None)?;
    let mut reference = GroupValuesColumn::<false>::try_new(schema)?;
    for _ in 0..3 {
        assert_eq!(reference.emit(EmitTo::All)?, candidate.emit(EmitTo::All)?);
        assert_eq!(
            reference.emit(EmitTo::First(0))?,
            candidate.emit(EmitTo::First(0))?
        );
        candidate.clear_shrink(0);
        reference.clear_shrink(0);
    }
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![1, 1, 2])),
        Arc::new(Float64Array::from(vec![-0.0, 0.0, 1.0])),
        Arc::new(StringViewArray::from(vec!["x", "x", "y"])),
    ];
    let (mut actual, mut expected) = (Vec::new(), Vec::new());
    candidate.intern(&cols, &mut actual)?;
    reference.intern(&cols, &mut expected)?;
    assert_eq!(actual, expected);
    assert_eq!(candidate.emit(EmitTo::All)?, reference.emit(EmitTo::All)?);
    Ok(())
}
