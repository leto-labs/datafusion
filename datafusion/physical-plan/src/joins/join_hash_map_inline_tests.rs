// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow patch (JT 9): the inline-key probe against the hash-only probe
//! followed by the key comparison.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int32Array, Int64Array};
use arrow::buffer::NullBuffer;
use datafusion_common::NullEquality;

use super::*;
use crate::joins::inline_keys::InlineKeys;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn mix(value: u64) -> u64 {
    let mut state = value;
    splitmix(&mut state)
}

/// One generated case: key columns of the build and the probe.
struct Case {
    build: Vec<ArrayRef>,
    probe: Vec<ArrayRef>,
}

/// The hash of a row: keys `share` apart (integer division) hash alike, so
/// `share > 1` makes chains of different keys (mixed chains).
fn hashes(columns: &[ArrayRef], share: i64) -> Vec<u64> {
    let rows = columns[0].len();
    (0..rows)
        .map(|row| {
            columns.iter().fold(0_u64, |acc, column| {
                let value = if let Some(a) = column.as_any().downcast_ref::<Int64Array>()
                {
                    a.value(row)
                } else {
                    i64::from(
                        column
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap()
                            .value(row),
                    )
                };
                mix(acc ^ mix(value.div_euclid(share) as u64))
            })
        })
        .collect()
}

fn validity(columns: &[ArrayRef]) -> Option<NullBuffer> {
    columns
        .iter()
        .fold(None, |acc, c| {
            NullBuffer::union(acc.as_ref(), c.logical_nulls().as_ref())
        })
        .filter(|n| n.null_count() > 0)
}

#[allow(clippy::too_many_arguments)]
fn case(
    seed: u64,
    build_rows: usize,
    probe_rows: usize,
    domain: u64,
    two_columns: bool,
    nulls: bool,
) -> Case {
    let mut state = seed;
    let mut column = |rows: usize, lo: i64, span: u64, nulls: bool| -> Vec<Option<i64>> {
        (0..rows)
            .map(|_| {
                let r = splitmix(&mut state);
                (!(nulls && r % 17 == 0)).then(|| lo + (r >> 8) as i64 % span as i64)
            })
            .collect()
    };
    // The probe draws from a wider domain: absent keys and, for two
    // columns, values outside the build's range.
    let build_a = column(build_rows, 100, domain, nulls);
    let probe_a = column(probe_rows, 98, domain + 4, nulls);
    let mut build: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(build_a))];
    let mut probe: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(probe_a))];
    if two_columns {
        let build_b = column(build_rows, -5, 7, nulls);
        let probe_b = column(probe_rows, -6, 9, nulls);
        build.push(Arc::new(Int32Array::from(
            build_b
                .into_iter()
                .map(|v| v.map(|v| v as i32))
                .collect::<Vec<_>>(),
        )));
        probe.push(Arc::new(Int32Array::from(
            probe_b
                .into_iter()
                .map(|v| v.map(|v| v as i32))
                .collect::<Vec<_>>(),
        )));
    }
    Case { build, probe }
}

/// Every chunk of a probe with `limit`, concatenated; checks that each
/// chunk has at most `limit` pairs and that offsets advance.
fn drain(
    limit: usize,
    mut lookup: impl FnMut(ProbeOffset, &mut Vec<u32>, &mut Vec<u64>) -> Option<ProbeOffset>,
) -> Vec<(u32, u64)> {
    let mut pairs = vec![];
    let mut offset = AtRow { row: 0 };
    let (mut input, mut matched) = (vec![], vec![]);
    for _ in 0..1_000_000 {
        let next = lookup(offset, &mut input, &mut matched);
        assert!(input.len() <= limit && input.len() == matched.len());
        pairs.extend(input.iter().copied().zip(matched.iter().copied()));
        match next {
            None => return pairs,
            Some(next) => {
                assert_ne!(next, offset, "a probe offset must advance");
                offset = next;
            }
        }
    }
    panic!("probe did not finish");
}

fn check(case: &Case, share: i64, limits: &[usize]) {
    let keys =
        InlineKeys::for_build(&[case.build.clone()], NullEquality::NullEqualsNothing)
            .expect("integer keys encode");
    let rows = case.build[0].len();
    let build_hashes = hashes(&case.build, share);
    let build_valid = validity(&case.build);
    let build_keys: Vec<u64> = (0..rows)
        .map(|row| keys.encode_row(&case.build, row).unwrap_or(0))
        .collect();
    let layout = [(0, rows, build_valid)];

    let mut reference = JoinHashMapU32::with_capacity(rows);
    reference
        .update_partitioned(&BuildRows {
            hashes: &build_hashes,
            keys: None,
            batches: &layout,
        })
        .unwrap();
    let mut inline = JoinHashMapInline::with_capacity(rows, keys.clone());
    inline
        .update_partitioned(&BuildRows {
            hashes: &build_hashes,
            keys: Some(&build_keys),
            batches: &layout,
        })
        .unwrap();
    assert_eq!(inline.chained, reference.chained);
    assert_eq!(inline.len(), reference.len());

    let probe_hashes = hashes(&case.probe, share);
    let probe_valid = validity(&case.probe);
    let encoded = keys.encode(&case.probe).unwrap();
    // The hash-only map's answers for the same hashes are unchanged.
    assert_eq!(
        inline.contain_hashes(&probe_hashes),
        reference.contain_hashes(&probe_hashes)
    );
    for &limit in limits {
        let expected: Vec<(u32, u64)> = drain(limit, |offset, input, matched| {
            reference.get_matched_indices_with_limit_offset(
                &probe_hashes,
                probe_valid.as_ref(),
                limit,
                offset,
                input,
                matched,
            )
        })
        .into_iter()
        .filter(|&(p, b)| {
            let p = p as usize;
            encoded.valid.as_ref().is_none_or(|v| v.is_valid(p))
                && keys.encode_row(&case.build, b as usize) == Some(encoded.keys[p])
        })
        .collect();
        let actual = drain(limit, |offset, input, matched| {
            inline
                .get_inline_matched_indices_with_limit_offset(
                    &probe_hashes,
                    &encoded,
                    &case.build,
                    limit,
                    offset,
                    input,
                    matched,
                )
                .unwrap()
        });
        assert_eq!(actual, expected, "limit {limit} share {share}");
    }
}

const LIMITS: &[usize] = &[1, 2, 3, 7, 16, 17, 1000, usize::MAX / 2];

#[test]
fn inline_probe_equals_the_compared_hash_probe() {
    let mut cases = 0;
    for seed in 0..6 {
        for &(build_rows, probe_rows) in &[(40, 60), (3_000, 2_000), (70_000, 20_000)] {
            for &domain in &[build_rows as u64 * 4, build_rows as u64 / 3 + 1] {
                for two_columns in [false, true] {
                    for nulls in [false, true] {
                        let case = case(
                            seed,
                            build_rows,
                            probe_rows,
                            domain,
                            two_columns,
                            nulls,
                        );
                        for share in [1, 3] {
                            check(&case, share, LIMITS);
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 6 * 3 * 2 * 2 * 2 * 2);
}

#[test]
fn unique_keys_take_the_unique_path() {
    // Distinct keys with distinct hashes: no chain, so `chained` is false
    // and the probe takes the unique path, also when keys are absent.
    let build: ArrayRef = Arc::new(Int64Array::from_iter_values(0..5_000));
    let probe: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..8_000).map(|v| (v * 7919) % 10_000),
    ));
    let case = Case {
        build: vec![build],
        probe: vec![probe],
    };
    check(&case, 1, LIMITS);
    let keys =
        InlineKeys::for_build(&[case.build.clone()], NullEquality::NullEqualsNothing)
            .unwrap();
    let mut inline = JoinHashMapInline::with_capacity(5_000, keys);
    let build_hashes = hashes(&case.build, 1);
    let build_keys: Vec<u64> = (0..5_000).collect();
    inline
        .update_partitioned(&BuildRows {
            hashes: &build_hashes,
            keys: Some(&build_keys),
            batches: &[(0, 5_000, None)],
        })
        .unwrap();
    assert!(!inline.chained);
}

#[test]
fn a_tag_collision_of_different_keys_marks_the_chain_mixed() {
    let mut table = HeadTable::<InlineU32>::with_capacity(8);
    assert_eq!(table.insert(42, 1, 7), Ok(0));
    assert_eq!(table.find(42), (1, 7, false));
    // The same hash and key: still one key.
    assert_eq!(table.insert(42, 2, 7), Ok(1));
    assert_eq!(table.find(42), (2, 7, false));
    // Another key with the same hash: mixed from now on.
    assert_eq!(table.insert(42, 3, 9), Ok(2));
    assert_eq!(table.find(42), (3, 9, true));
    assert_eq!(table.insert(42, 4, 9), Ok(3));
    assert_eq!(table.find(42), (4, 9, true));
    // The head keeps its value below the mark.
    assert_eq!(table.head(42), 4);
    assert_eq!(table.find(43), (0, 0, false));
}

#[test]
fn an_inline_table_is_at_most_twice_the_u32_table() {
    for capacity in [1, 2, 3, 1000, 1 << 20, (1 << 20) + 1, 3 << 19, 5_000_000] {
        let inline = HeadTable::<InlineU32>::bytes_for(capacity);
        let narrow = HeadTable::<u32>::bytes_for(capacity);
        assert!(inline <= 2 * narrow, "{capacity}: {inline} {narrow}");
        // A load of at most 2/3.
        assert!(inline / 16 * 2 >= capacity * 3, "{capacity}");
    }
}

#[test]
fn an_inline_map_refuses_rows_without_keys() {
    let keys = InlineKeys::for_build(
        &[vec![Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef]],
        NullEquality::NullEqualsNothing,
    )
    .unwrap();
    let mut map = JoinHashMapInline::with_capacity(1, keys);
    assert!(
        map.update_from_iter(Box::new([1_u64].iter().enumerate()), 0)
            .is_err()
    );
    assert!(
        map.update_partitioned(&BuildRows {
            hashes: &[1],
            keys: None,
            batches: &[(0, 1, None)],
        })
        .is_err()
    );
    // A map without inline keys refuses the inline probe.
    let plain = JoinHashMapU32::with_capacity(1);
    assert!(plain.inline_keys().is_none());
    let encoded = EncodedKeys {
        keys: vec![1],
        valid: None,
    };
    assert!(
        plain
            .get_inline_matched_indices_with_limit_offset(
                &[1],
                &encoded,
                &[],
                1,
                AtRow { row: 0 },
                &mut vec![],
                &mut vec![],
            )
            .is_err()
    );
}
