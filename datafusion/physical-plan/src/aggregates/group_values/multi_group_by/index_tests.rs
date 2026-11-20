// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::array::*;
use arrow::compute::{cast, take};
use arrow::datatypes::*;
use arrow::row::{RowConverter, SortField};
use datafusion_common::Result;
use datafusion_expr::{EmitTo, GroupSelection};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{GroupValues, GroupValuesColumn};

fn key_arrays() -> Result<Vec<ArrayRef>> {
    use IntervalDayTimeType as DayTime;
    use IntervalMonthDayNanoType as MonthDayNano;

    let strings: ArrayRef = Arc::new(StringArray::from(vec![
        "a",
        "long-shared-prefix-b",
        "long-shared-prefix-c",
        "d",
    ]));
    let mut keys = Vec::new();
    macro_rules! primitives {
        ($($ty:ty),+ $(,)?) => {
            $(keys.push(Arc::new(PrimitiveArray::<$ty>::from_iter_values([1, 2, 3, 4])) as ArrayRef);)+
        };
    }
    primitives!(
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
    keys.push(Arc::new(Decimal256Array::from_iter_values(
        (1..=4).map(i256::from_i128),
    )));
    keys.push(Arc::new(Float16Array::from_iter_values(
        (1..=4).map(|x| half::f16::from_f32(x as f32)),
    )));
    keys.push(Arc::new(Float32Array::from(vec![
        0.0,
        1.0,
        -2.0,
        f32::INFINITY,
    ])));
    keys.push(Arc::new(Float64Array::from(vec![
        0.0,
        1.0,
        -2.0,
        f64::INFINITY,
    ])));
    keys.push(Arc::new(IntervalDayTimeArray::from_iter_values(
        (1..=4).map(|x| DayTime::make_value(x, 0)),
    )));
    keys.push(Arc::new(IntervalMonthDayNanoArray::from_iter_values(
        (1..=4).map(|x| MonthDayNano::make_value(x, 0, 0)),
    )));
    keys.push(Arc::new(BooleanArray::from(vec![false, true, true, false])));
    for ty in [
        DataType::LargeUtf8,
        DataType::Utf8View,
        DataType::Binary,
        DataType::LargeBinary,
        DataType::BinaryView,
    ] {
        keys.push(cast(&strings, &ty)?);
    }
    keys.push(Arc::new(FixedSizeBinaryArray::try_from_iter(
        [*b"ab", *b"ac", *b"ba", *b"cd"].iter(),
    )?));
    keys.push(Arc::new(FixedSizeBinaryArray::try_new(
        0,
        arrow::buffer::Buffer::from(Vec::<u8>::new()),
        Some(arrow::buffer::NullBuffer::new_valid(4)),
    )?));
    macro_rules! dictionaries {
        ($($ty:ty),+ $(,)?) => {
            $(keys.push(Arc::new(DictionaryArray::<$ty>::try_new(
                PrimitiveArray::<$ty>::from_iter_values([0, 1, 2, 3]),
                Arc::clone(&strings),
            )?));)+
        };
    }
    dictionaries!(
        Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
        UInt64Type,
    );
    let items = || (0..4).map(|x| Some([Some(x), (x % 2 == 0).then_some(x)]));
    keys.push(Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
        items(),
    )));
    keys.push(Arc::new(LargeListArray::from_iter_primitive::<
        Int32Type,
        _,
        _,
    >(items())));
    keys.push(Arc::new(FixedSizeListArray::from_iter_primitive::<
        Int32Type,
        _,
        _,
    >(items(), 2)));
    keys.push(Arc::new(StructArray::from(vec![(
        Arc::new(Field::new("child", DataType::Utf8, false)),
        Arc::clone(&strings),
    )])));
    keys.push(Arc::new(ListViewArray::from_iter_primitive::<
        Int32Type,
        _,
        _,
    >(items())));
    keys.push(Arc::new(LargeListViewArray::from_iter_primitive::<
        Int32Type,
        _,
        _,
    >(items())));
    let entries = StructArray::from(vec![
        (
            Arc::new(Field::new("keys", DataType::Int32, false)),
            Arc::new(Int32Array::from(vec![0, 0, 1, 0])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("values", DataType::Utf8, true)),
            Arc::clone(&strings),
        ),
    ]);
    keys.push(Arc::new(MapArray::try_new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        arrow::buffer::OffsetBuffer::from_lengths([1, 0, 2, 1]),
        entries,
        None,
        true,
    )?));
    keys.push(strings);
    Ok(keys)
}

fn assert_output(converter: &RowConverter, output: &[ArrayRef], expected: &[Vec<u8>]) {
    let rows = converter.convert_columns(output).unwrap();
    assert_eq!(rows.num_rows(), expected.len());
    for (actual, expected) in rows.iter().zip(expected) {
        assert_eq!(actual.as_ref(), expected.as_slice());
    }
}

fn check_types<const STREAMING: bool>() -> Result<()> {
    for key in key_arrays()? {
        assert_eq!(key.len(), 4, "{}", key.data_type());
        let schema = Arc::new(Schema::new(vec![
            Field::new("tag", DataType::Int32, true),
            Field::new("value", key.data_type().clone(), true),
        ]));
        let converter = RowConverter::new(
            schema
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()))
                .collect(),
        )?;
        let mut values = GroupValuesColumn::<STREAMING>::try_new(schema)?;
        let mut rng = StdRng::seed_from_u64(335117);
        let mut expected = Vec::new();
        let mut ids = Vec::new();
        for batch in 0..15 {
            let rows = rng.random_range(24..96);
            let tags = Int32Array::from_iter(
                (0..rows).map(|_| rng.random_bool(0.85).then(|| rng.random_range(0..13))),
            );
            let indices = UInt32Array::from_iter(
                (0..rows).map(|_| rng.random_bool(0.85).then(|| rng.random_range(0..4))),
            );
            let cols = vec![Arc::new(tags) as ArrayRef, take(&key, &indices, None)?];
            values.intern(&cols, &mut ids)?;
            let input = converter.convert_columns(&cols)?;
            assert_eq!(input.num_rows(), ids.len(), "{}", key.data_type());
            let mut next_id = expected.len();
            expected.resize(values.len(), Vec::new());
            for (row, &group) in input.iter().zip(&ids) {
                let existing = &mut expected[group];
                if existing.is_empty() {
                    if STREAMING {
                        assert_eq!(group, next_id);
                        next_id += 1;
                    }
                    existing.extend_from_slice(row.as_ref());
                }
                assert_eq!(existing.as_slice(), row.as_ref(), "{}", key.data_type());
            }
            assert!(expected.iter().all(|row| !row.is_empty()));
            let unique = expected.iter().collect::<HashSet<_>>();
            assert_eq!(unique.len(), values.len(), "{}", key.data_type());
            assert_output(
                &converter,
                &values.values_preserving(GroupSelection::all(values.len()))?,
                &expected,
            );
            match batch % 5 {
                2 => {
                    let n = rng.random_range(0..=expected.len());
                    assert_output(
                        &converter,
                        &values.emit(EmitTo::First(n))?,
                        &expected[..n],
                    );
                    expected.drain(..n);
                    values.shrink_to_fit();
                }
                4 => {
                    assert_output(&converter, &values.emit(EmitTo::All)?, &expected);
                    expected.clear();
                    values.clear_shrink(batch);
                }
                _ => {}
            }
        }
    }
    Ok(())
}

#[test]
fn typed_keys_survive_partitioned_batches_and_prefix_emission() -> Result<()> {
    check_types::<false>()?;
    check_types::<true>()
}

#[test]
fn emitted_keys_merge_after_replay() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, true),
    ]));
    let new_interner = || GroupValuesColumn::<false>::try_new(Arc::clone(&schema));
    let mut partial = new_interner()?;
    let mut merged = new_interner()?;
    let mut ids = Vec::new();
    let mut partial_counts = Vec::new();
    let mut merge_counts = Vec::new();
    let mut expected = std::collections::BTreeMap::new();
    for partition in 0..8 {
        let mut partial_group_count = 0;
        for batch in 0..3 {
            let keys = (0..191)
                .map(|row| {
                    let a = ((row * 83 + partition * 11 + batch * 29) % 257) as i64;
                    (a, (row % 7 != 0).then_some(a.wrapping_mul(32416190071)))
                })
                .collect::<Vec<_>>();
            let cols = [
                Arc::new(Int64Array::from_iter_values(keys.iter().map(|k| k.0)))
                    as ArrayRef,
                Arc::new(Int64Array::from_iter(keys.iter().map(|k| k.1))) as ArrayRef,
            ];
            partial.intern(&cols, &mut ids)?;
            partial_counts.resize(partial.len(), 0);
            for (key, &id) in keys.into_iter().zip(&ids) {
                *expected.entry(key).or_insert(0) += 1;
                partial_counts[id] += 1;
            }
            partial_group_count = partial.len();
        }
        let spill = partial.emit(EmitTo::All)?;
        assert_eq!(partial_group_count, spill[0].len());
        partial.clear_shrink(0);
        merged.intern(&spill, &mut ids)?;
        merge_counts.resize(merged.len(), 0);
        for (&group, count) in ids.iter().zip(partial_counts.drain(..)) {
            merge_counts[group] += count;
        }
    }
    let keys = merged.emit(EmitTo::All)?;
    let a = keys[0].as_primitive::<Int64Type>();
    let b = keys[1].as_primitive::<Int64Type>();
    assert_eq!(a.len(), expected.len());
    for (row, count) in merge_counts.into_iter().enumerate() {
        let key = (a.value(row), b.is_valid(row).then(|| b.value(row)));
        assert_eq!(Some(count), expected.remove(&key));
    }
    assert!(expected.is_empty());
    Ok(())
}

#[test]
fn prefix_emission_keeps_collisions_in_any_bucket_order() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let mut interner = GroupValuesColumn::<false>::try_new(schema)?;
    let values = Arc::new(Int32Array::from_iter_values(0..6)) as ArrayRef;
    interner.intern(&[values], &mut Vec::new())?;
    interner.map.clear();
    interner.group_index_lists.clear();

    // The first list's bucket can come after the second list's bucket during
    // iteration. Compacting while visiting slots must not overwrite it.
    let first = 0x8000_0000_0000_0006;
    let second = 1;
    super::tests::insert_non_inline_group_index_view(&mut interner, first, vec![1, 2]);
    super::tests::insert_non_inline_group_index_view(&mut interner, second, vec![3, 4]);
    interner.emit(EmitTo::First(1))?;
    assert_eq!(interner.get_indices_by_hash(first).unwrap().0, vec![0, 1]);
    assert_eq!(interner.get_indices_by_hash(second).unwrap().0, vec![2, 3]);
    Ok(())
}

#[test]
fn index_and_prefetch_allocations_are_charged_and_released() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, true),
    ]));
    let mut values = GroupValuesColumn::<false>::try_new(schema)?;
    let empty = values.size();

    // Reserve enough groups to cross the prefetch threshold.
    values.map.flat_mut().reserve(48 * 1024 + 1);
    let slot_bytes = 128 * 1024 * 16;
    assert_eq!(values.size() - empty, slot_bytes);
    let hashes = [0, u64::MAX, 1 << 63];
    values
        .map
        .flat()
        .prepare_positions(&hashes, &mut values.probe_positions);
    #[cfg(target_arch = "x86_64")]
    assert_eq!(values.probe_positions, vec![0, 128 * 1024 - 1, 64 * 1024]);
    let scratch_bytes = values.probe_positions.capacity() * size_of::<u32>();
    assert_eq!(values.size() - empty, slot_bytes + scratch_bytes);

    values.shrink_to_fit();
    assert_eq!(values.size(), empty);
    values.map.flat_mut().reserve(2);
    assert_eq!(values.size() - empty, 4 * 16);
    values.clear_shrink(0);
    assert_eq!(values.size(), empty);
    Ok(())
}

#[cfg(all(target_arch = "x86_64", not(feature = "force_hash_collisions")))]
#[test]
fn prefetched_probes_handle_new_and_existing_keys_at_batch_tail() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
    ]));
    let mut values = GroupValuesColumn::<false>::try_new(schema)?;
    let columns = |rows: &[i64]| -> Vec<ArrayRef> {
        vec![
            Arc::new(Int64Array::from_iter_values(rows.iter().copied())),
            Arc::new(Int64Array::from_iter_values(
                rows.iter()
                    .map(|row| row.wrapping_mul(0x517c_c1b7_2722_0a95)),
            )),
        ]
    };

    let mut ids = Vec::new();
    let initial: Vec<_> = (0..50_000).collect();
    values.intern(&columns(&initial), &mut ids)?;
    assert_eq!(values.probe_positions.len(), initial.len());
    assert!(ids.iter().copied().eq(0..50_000));

    let next = (0..55)
        .map(|i| if i % 2 == 0 { 49_970 + i } else { 3 * i })
        .collect::<Vec<_>>();
    values.intern(&columns(&next), &mut ids)?;
    assert_eq!(values.probe_positions.len(), next.len());
    let mut new_group = 50_000;
    for (&row, observed) in next.iter().zip(ids) {
        let expected = if row < 50_000 {
            row as usize
        } else {
            let group = new_group;
            new_group += 1;
            group
        };
        assert_eq!(observed, expected);
    }
    assert_eq!(values.len(), new_group);
    Ok(())
}
