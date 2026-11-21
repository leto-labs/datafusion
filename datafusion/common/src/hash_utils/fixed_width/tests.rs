// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;
use crate::hash_utils::{RandomState, create_hashes};
use arrow::array::*;
use arrow::buffer::NullBuffer;
use arrow::datatypes::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hash::BuildHasher;
use std::sync::Arc;

// Use the production scalar dispatch (including seeded rehash and its null
// rules) with the same hasher, but leave the SIMD hook at its default.
struct ScalarOnly(RandomState);

impl BuildHasher for ScalarOnly {
    type Hasher = <RandomState as BuildHasher>::Hasher;

    fn build_hasher(&self) -> Self::Hasher {
        self.0.build_hasher()
    }
}

impl HashState for ScalarOnly {
    type SeededState = <RandomState as HashState>::SeededState;

    fn seeded_state(&self, seed: u64) -> Self::SeededState {
        self.0.seeded_state(seed)
    }
}

fn assert_hashes_match(columns: &[ArrayRef], initial: &[u64], seed: u64) {
    let mut expected = initial.to_vec();
    let mut actual = initial.to_vec();
    create_hashes(
        columns,
        &ScalarOnly(RandomState::with_seed(seed)),
        &mut expected,
    )
    .unwrap();
    create_hashes(columns, &RandomState::with_seed(seed), &mut actual).unwrap();
    let types: Vec<_> = columns.iter().map(|column| column.data_type()).collect();
    assert_eq!(actual, expected, "seed {seed:#x}, columns {types:?}");
}

fn add<T: ArrowPrimitiveType>(
    columns: &mut Vec<ArrayRef>,
    nulls: Option<&NullBuffer>,
    values: Vec<T::Native>,
) {
    columns.push(Arc::new(PrimitiveArray::<T>::new(
        values.into(),
        nulls.cloned(),
    )));
}

fn primitive_columns(
    rng: &mut StdRng,
    n: usize,
    nulls: Option<&NullBuffer>,
) -> Vec<ArrayRef> {
    let mut columns = Vec::<ArrayRef>::new();
    macro_rules! random_columns {
        ($($ty:ty),+ $(,)?) => {
            $(add::<$ty>(&mut columns, nulls, (0..n).map(|_| rng.random()).collect());)+
        };
    }
    random_columns!(
        Int8Type,
        Int16Type,
        Int32Type,
        Int64Type,
        UInt8Type,
        UInt16Type,
        UInt32Type,
        UInt64Type,
        Decimal32Type,
        Decimal64Type,
        Decimal128Type,
        Date32Type,
        Date64Type,
        Time32SecondType,
        Time32MillisecondType,
        Time64MicrosecondType,
        Time64NanosecondType,
        TimestampSecondType,
        TimestampMillisecondType,
        TimestampMicrosecondType,
        TimestampNanosecondType,
        DurationSecondType,
        DurationMillisecondType,
        DurationMicrosecondType,
        DurationNanosecondType,
        IntervalYearMonthType,
    );
    add::<Decimal256Type>(
        &mut columns,
        nulls,
        (0..n)
            .map(|_| i256::from_parts(rng.random(), rng.random()))
            .collect(),
    );
    add::<IntervalDayTimeType>(
        &mut columns,
        nulls,
        (0..n)
            .map(|_| IntervalDayTime::new(rng.random(), rng.random()))
            .collect(),
    );
    add::<IntervalMonthDayNanoType>(
        &mut columns,
        nulls,
        (0..n)
            .map(|_| IntervalMonthDayNano::new(rng.random(), rng.random(), rng.random()))
            .collect(),
    );
    macro_rules! float_column {
        ($ty:ty, $native:ty, $bits:ty, $neg_zero:expr) => {{
            let edge_cases = [
                0,
                ($neg_zero).to_bits(),
                <$native>::INFINITY.to_bits(),
                <$native>::NEG_INFINITY.to_bits(),
                <$native>::NAN.to_bits(),
                <$bits>::MAX,
                1,
            ];
            add::<$ty>(
                &mut columns,
                nulls,
                (0..n)
                    .map(|i| {
                        <$native>::from_bits(
                            edge_cases.get(i).copied().unwrap_or_else(|| rng.random()),
                        )
                    })
                    .collect(),
            );
        }};
    }
    float_column!(Float16Type, f16, u16, f16::NEG_ZERO);
    float_column!(Float32Type, f32, u32, -0.0_f32);
    float_column!(Float64Type, f64, u64, -0.0_f64);
    columns
}

#[test]
fn folded_product_matches_u128_with_carries() {
    let mut rng = StdRng::seed_from_u64(0x8a08_4b53);
    let edges = [0, 1, u32::MAX as u64, 1 << 32, u64::MAX - 1, u64::MAX];
    for (x, y) in edges
        .into_iter()
        .flat_map(|x| edges.map(|y| (x, y)))
        .chain((0..100_000).map(|_| (rng.random(), rng.random())))
    {
        let product = u128::from(x) * u128::from(y);
        assert_eq!(fold(x, y), product as u64 ^ (product >> 64) as u64);
    }
}

#[test]
fn fixed_width_hashes_match_scalar_for_slices_nulls_and_rehash() {
    let mut rng = StdRng::seed_from_u64(0x444d_08ac);
    // Cover empty arrays, SIMD entry, SIMD widths, batch ends and nullable
    // bitmaps starting inside a byte. Distinct validity keeps nullable slots
    // from accidentally receiving the same accumulator on every column.
    for len in [0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 129, 8193] {
        for (offset, null_rate) in [(0, 0), (3, 0), (5, 1), (7, 50), (9, 100)] {
            let n = len + offset;
            let initial = (0..len).map(|_| rng.random()).collect::<Vec<_>>();
            let validity = (null_rate > 0).then(|| {
                NullBuffer::from(
                    (0..n)
                        .map(|_| rng.random_range(0..100) >= null_rate)
                        .collect::<Vec<_>>(),
                )
            });
            let mut columns = primitive_columns(&mut rng, n, validity.as_ref());
            columns.push(Arc::new(BooleanArray::from_iter((0..n).map(|i| {
                (validity.as_ref().is_none_or(|v| v.is_valid(i)))
                    .then(|| rng.random::<bool>())
            }))));
            for width in [0, 1, 2, 3, 4, 8, 12, 16, 17, 32] {
                let bytes: Vec<u8> =
                    (0..n * width as usize).map(|_| rng.random()).collect();
                columns.push(Arc::new(
                    FixedSizeBinaryArray::try_new_with_len(
                        width,
                        bytes.into(),
                        validity.clone(),
                        n,
                    )
                    .unwrap(),
                ));
            }
            columns.push(Arc::new(NullArray::new(n)));
            let columns: Vec<_> = columns.iter().map(|a| a.slice(offset, len)).collect();
            let seed = rng.random();
            let previous = Arc::new(UInt64Array::from_iter(
                (0..len).map(|i| (i % 3 != 0).then(|| rng.random::<u64>())),
            )) as ArrayRef;
            for column in &columns {
                assert_hashes_match(std::slice::from_ref(column), &initial, seed);
                assert_hashes_match(
                    &[Arc::clone(&previous), Arc::clone(column)],
                    &initial,
                    seed,
                );
            }
            assert_hashes_match(&columns, &initial, seed);
            assert_hashes_match(
                &columns.into_iter().rev().collect::<Vec<_>>(),
                &initial,
                seed,
            );
        }
    }
}

#[test]
fn eligible_dense_batch_uses_the_fast_path() {
    let state = RandomState::with_seed(0x3207);
    let array = Int64Array::from(vec![i64::MIN; 65]);
    let mut hashes = vec![0x812f; 65];
    if simd_available() {
        assert!(try_hash_primitive(&array, &mut hashes, &state, false));
        assert!(try_hash_primitive(&array, &mut hashes, &state, true));
    }
    assert!(!try_hash_primitive(
        &array,
        &mut hashes,
        &ScalarOnly(state),
        false
    ));
}
