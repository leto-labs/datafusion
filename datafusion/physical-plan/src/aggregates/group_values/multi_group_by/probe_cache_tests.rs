// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Longbow Authors.

use super::GroupValuesColumn;
use crate::aggregates::group_values::GroupValues;
use arrow::array::{Array, ArrayRef, Int32Array, Int64Array, StringViewArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_expr::EmitTo;
use std::sync::Arc;

fn key(cols: &[ArrayRef], row: usize) -> (Option<i64>, Option<i32>, Option<String>) {
    let a = cols[0].as_any().downcast_ref::<Int64Array>().unwrap();
    let b = cols[1].as_any().downcast_ref::<Int32Array>().unwrap();
    let c = cols[2].as_any().downcast_ref::<StringViewArray>().unwrap();
    (
        a.is_valid(row).then(|| a.value(row)),
        b.is_valid(row).then(|| b.value(row)),
        c.is_valid(row).then(|| c.value(row).to_owned()),
    )
}

fn check_reset<const STREAMING: bool>() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Utf8View, true),
    ]));
    let mut values = GroupValuesColumn::<STREAMING>::try_new(schema).unwrap();
    for cycle in 0..3 {
        let ids: Vec<_> = (0..8192).map(|row| (row / 7 % 173) as i64).collect();
        let a = Int64Array::from(
            ids.iter()
                .map(|id| (id % 13 != 0).then_some(*id + cycle * 173))
                .collect::<Vec<_>>(),
        );
        let b = Int32Array::from(
            ids.iter()
                .map(|id| (id % 17 != 0).then_some((id % 60) as i32))
                .collect::<Vec<_>>(),
        );
        let c = StringViewArray::from_iter(
            ids.iter()
                .map(|id| (id % 19 != 0).then(|| format!("shared-prefix-value-{id:04}")))
                .collect::<Vec<_>>(),
        );
        let input: Vec<ArrayRef> = vec![Arc::new(a), Arc::new(b), Arc::new(c)];
        let mut groups = Vec::new();
        values.intern(&input, &mut groups).unwrap();
        assert!(values.len() > 64);
        values.emit(EmitTo::First(7)).unwrap();
        values.intern(&input, &mut groups).unwrap();
        let output = values.emit(EmitTo::All).unwrap();
        for (row, group) in groups.iter().enumerate() {
            assert_eq!(key(&input, row), key(&output, *group));
        }
        values.intern(&input, &mut groups).unwrap();
        values.clear_shrink(8192);
        assert_eq!(values.len(), 0);
    }
}

#[test]
fn repeated_hash_candidates_keep_full_key_equality_after_prefix_and_clear() {
    check_reset::<false>();
}

#[test]
fn streaming_groups_reset_lookup_after_full_emission() {
    check_reset::<true>();
}
