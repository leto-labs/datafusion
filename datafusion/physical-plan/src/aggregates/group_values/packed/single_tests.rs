// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Differential tests of one packed integer key column against the general
//! interner: every integer-coded type, NULLs, duplicates, emissions, the
//! direct and hashed tables, and the move to the single-column interner
//! above 32 bits.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, Int64Array, UInt64Array};
use arrow::datatypes::DataType;
use datafusion_expr::EmitTo;

use super::single_column;
use super::table::NARROW_BITS;
use super::tests::{Differential, Gen, Rng, column, run, schema};
use crate::aggregates::group_values::GroupValues;

const TYPES: [DataType; 10] = [
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::UInt8,
    DataType::UInt64,
    DataType::Date32,
    DataType::Decimal32(9, 2),
    DataType::Decimal64(18, 3),
    DataType::Decimal128(38, 0),
];

#[test]
fn single_column_dispatch_takes_integer_codes_only() {
    for data_type in TYPES {
        assert!(single_column(&schema(&[data_type.clone()])), "{data_type}");
    }
    for data_type in [
        DataType::Boolean,
        DataType::Utf8View,
        DataType::BinaryView,
        DataType::Utf8,
        DataType::Float64,
    ] {
        assert!(!single_column(&schema(&[data_type.clone()])), "{data_type}");
    }
    assert!(!single_column(&schema(&[DataType::Int32, DataType::Int32])));
}

/// Every type, from a domain that fits the direct table to one that needs
/// hashed slots, with NULLs, duplicates, partial and full emission and
/// `clear_shrink` (all driven by `run`).
#[test]
fn single_integer_columns_match_the_general_interner() {
    for (index, data_type) in TYPES.iter().enumerate() {
        // Int8/UInt8 cannot hold the larger cardinalities without wrapping;
        // wrapping only adds duplicates, which the oracle also sees.
        for (card, nulls) in [(7, 0), (300, 5), (40_000, 20), (3_000_000, 1)] {
            let seed = index as u64 * 31 + card;
            run(
                std::slice::from_ref(data_type),
                &[Gen::Int {
                    card,
                    offset: -1_000 + index as i64,
                    nulls,
                }],
                seed,
                20,
                2048,
            );
        }
    }
}

/// The domain grows from direct to 8-byte slots and then beyond 32 bits:
/// the groups move, with their ids, to the single-column interner, which
/// takes every later batch. The packed table and keys are released.
#[test]
fn single_column_wider_than_32_bits_falls_back_with_ids() {
    let mut check = Differential::new(&[DataType::Int64]);
    let ints = |v: Vec<Option<i64>>| Arc::new(Int64Array::from(v)) as ArrayRef;
    check.intern(&[ints(vec![Some(5), None, Some(7), Some(5), None])]);
    assert!(check.packed.is_direct());
    check.intern(&[ints((0..5_000).map(|v| Some(v * 1_000)).collect())]);
    assert!(!check.packed.is_fallback());
    assert!(!check.packed.is_direct());
    assert!(check.packed.key_bits() <= NARROW_BITS);
    check.intern(&[ints(vec![Some(1 << 40), Some(5), None, Some(-(1 << 40))])]);
    assert!(check.packed.is_fallback());
    // Only the single-column interner's memory remains, besides the empty
    // packed table, the column's observed state and fixed fields.
    let fallback = check.packed.fallback.as_ref().unwrap().size();
    assert!(check.packed.size() >= fallback);
    assert!(
        check.packed.size() - fallback < 1024,
        "{}",
        check.packed.size() - fallback
    );
    check.intern(&[ints(vec![Some(7), Some(1 << 40), Some(9)])]);
    check.emit(EmitTo::First(3));
    check.emit(EmitTo::All);
    check.intern(&[ints(vec![Some(1), Some(2)])]);
    check.emit(EmitTo::All);
}

/// Extremes of unsigned and signed 64-bit keys, near both ends of the type.
#[test]
fn single_column_extremes() {
    let mut check = Differential::new(&[DataType::UInt64]);
    let values = |v: Vec<Option<u64>>| Arc::new(UInt64Array::from(v)) as ArrayRef;
    check.intern(&[values(vec![Some(u64::MAX), Some(u64::MAX - 1), None])]);
    assert!(!check.packed.is_fallback());
    check.intern(&[values(vec![Some(0), Some(u64::MAX)])]);
    assert!(check.packed.is_fallback());
    check.emit(EmitTo::All);

    let mut check = Differential::new(&[DataType::Int32]);
    let values = |v: Vec<Option<i32>>| Arc::new(Int32Array::from(v)) as ArrayRef;
    // i32::MIN..=i32::MAX plus NULL is 2^32 + 1 codes: one more than fits.
    check.intern(&[values(vec![Some(i32::MAX), None, Some(i32::MAX - 2)])]);
    check.intern(&[values(vec![Some(i32::MIN + 1), Some(0)])]);
    assert!(!check.packed.is_fallback());
    check.intern(&[values(vec![Some(i32::MIN), Some(i32::MAX)])]);
    assert!(check.packed.is_fallback());
    check.emit(EmitTo::First(2));
    check.emit(EmitTo::All);
}

/// A falling minimum rebases the stored codes in both table modes.
#[test]
fn single_column_falling_minimum_and_repeated_runs() {
    let mut rng = Rng(17);
    let mut check = Differential::new(&[DataType::Int32]);
    for round in 0..12_i64 {
        let base = 1_000_000 - round * 90_000;
        let generator = Gen::Int {
            card: 50 + round as u64 * 4_000,
            offset: base,
            nulls: 2,
        };
        let batch = column(&mut rng, &DataType::Int32, generator, 4096);
        // Repeat the batch: every key is a duplicate the second time.
        check.intern(std::slice::from_ref(&batch));
        check.intern(&[batch]);
        if round % 5 == 4 {
            check.emit(EmitTo::First(rng.below(check.packed.len() as u64) as usize));
        }
    }
    check.emit(EmitTo::All);
}

#[test]
fn single_column_overwrites_reused_key_and_group_buffers() {
    let mut packed = Differential::new(&[DataType::Int32]).packed;
    let mut groups = vec![usize::MAX; 8];
    for (keys, expected) in [
        (vec![Some(7), None, Some(4_933_010)], vec![0, 1, 2]),
        (vec![None, Some(17)], vec![1, 3]),
        (
            vec![Some(4_933_010), Some(7), None, Some(19)],
            vec![2, 0, 1, 4],
        ),
        (vec![], vec![]),
        (vec![None, Some(17)], vec![1, 3]),
    ] {
        let keys = Arc::new(Int32Array::from(keys)) as ArrayRef;
        packed.intern(&[keys], &mut groups).unwrap();
        assert_eq!(groups, expected);
        assert!(!packed.is_fallback());
    }
}
