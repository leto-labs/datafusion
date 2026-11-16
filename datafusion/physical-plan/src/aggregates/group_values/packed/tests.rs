// Modified by the Longbow authors; see third_party/README.md.

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

//! Differential tests of packed group keys against the general interner.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BinaryViewArray, BooleanArray, Date32Array,
    Decimal128Array, Int8Array, Int16Array, Int32Array, Int64Array, StringViewArray,
    UInt8Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::ScalarValue;
use datafusion_expr::EmitTo;

use super::table::{DIRECT_BITS, PackedKeyTable};
use super::{GroupValuesPacked, supported};
use crate::aggregates::group_values::{
    GroupValues, GroupValuesRows, unordered_multi_column,
};

/// A deterministic generator (SplitMix64).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

/// How one generated column draws its values.
#[derive(Clone, Copy)]
enum Gen {
    /// Integers `offset + [0, card)`, with NULLs at `nulls` percent.
    Int {
        card: u64,
        offset: i64,
        nulls: u64,
    },
    /// Strings from `card` values; `long` percent are out-of-line.
    Str {
        card: u64,
        long: u64,
        nulls: u64,
    },
    Bool {
        nulls: u64,
    },
}

fn schema(types: &[DataType]) -> SchemaRef {
    Arc::new(Schema::new(
        types
            .iter()
            .enumerate()
            .map(|(i, t)| Field::new(format!("k{i}"), t.clone(), true))
            .collect::<Vec<_>>(),
    ))
}

fn text(id: u64, long: bool) -> String {
    if long {
        format!("an-out-of-line-group-key-{id}")
    } else if id == 0 {
        String::new()
    } else {
        format!("k{id}")
    }
}

fn column(rng: &mut Rng, data_type: &DataType, generator: Gen, rows: usize) -> ArrayRef {
    match generator {
        Gen::Int {
            card,
            offset,
            nulls,
        } => {
            let value = |rng: &mut Rng| {
                (!rng.chance(nulls)).then(|| offset.wrapping_add(rng.below(card) as i64))
            };
            let values: Vec<Option<i64>> = (0..rows).map(|_| value(rng)).collect();
            match data_type {
                DataType::Int8 => Arc::new(
                    values
                        .iter()
                        .map(|v| v.map(|v| v as i8))
                        .collect::<Int8Array>(),
                ),
                DataType::Int16 => Arc::new(
                    values
                        .iter()
                        .map(|v| v.map(|v| v as i16))
                        .collect::<Int16Array>(),
                ),
                DataType::Int32 => Arc::new(
                    values
                        .iter()
                        .map(|v| v.map(|v| v as i32))
                        .collect::<Int32Array>(),
                ),
                DataType::Int64 => Arc::new(values.iter().collect::<Int64Array>()),
                DataType::UInt8 => Arc::new(
                    values
                        .iter()
                        .map(|v| v.map(|v| v as u8))
                        .collect::<UInt8Array>(),
                ),
                DataType::UInt64 => Arc::new(
                    values
                        .iter()
                        .map(|v| v.map(|v| v as u64))
                        .collect::<UInt64Array>(),
                ),
                DataType::Date32 => Arc::new(
                    values
                        .iter()
                        .map(|v| v.map(|v| v as i32))
                        .collect::<Date32Array>(),
                ),
                DataType::Decimal128(p, s) => Arc::new(
                    values
                        .iter()
                        .map(|v| v.map(|v| i128::from(v) * 1_000_000_007))
                        .collect::<Decimal128Array>()
                        .with_precision_and_scale(*p, *s)
                        .unwrap(),
                ),
                other => panic!("no integer generator for {other}"),
            }
        }
        Gen::Str { card, long, nulls } => {
            let values: Vec<Option<String>> = (0..rows)
                .map(|_| {
                    (!rng.chance(nulls)).then(|| {
                        let id = rng.below(card);
                        // A value's form depends only on its id.
                        text(id, id % 100 < long)
                    })
                })
                .collect();
            match data_type {
                DataType::Utf8View => Arc::new(StringViewArray::from_iter(values)),
                DataType::BinaryView => Arc::new(BinaryViewArray::from_iter(
                    values.iter().map(|v| v.as_ref().map(|v| v.as_bytes())),
                )),
                other => panic!("no string generator for {other}"),
            }
        }
        Gen::Bool { nulls } => Arc::new(
            (0..rows)
                .map(|_| (!rng.chance(nulls)).then(|| rng.chance(50)))
                .collect::<BooleanArray>(),
        ),
    }
}

fn assert_same_arrays(actual: &[ArrayRef], expected: &[ArrayRef], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (a, e) in actual.iter().zip(expected) {
        assert_eq!(a.data_type(), e.data_type(), "{context}");
        assert_eq!(a.len(), e.len(), "{context}");
        for row in 0..a.len() {
            assert_eq!(
                ScalarValue::try_from_array(a, row).unwrap(),
                ScalarValue::try_from_array(e, row).unwrap(),
                "{context}: row {row}"
            );
        }
    }
}

/// Drive the packed interner and the row interner with the same batches and
/// emissions; every group id, length and emitted value must agree. Both
/// number new groups in first-seen row order. (`GroupValuesColumn` can number
/// hash-colliding new groups later, so it is not an id oracle.)
struct Differential {
    packed: GroupValuesPacked,
    general: Box<dyn GroupValues>,
    batches: usize,
}

impl Differential {
    fn new(types: &[DataType]) -> Self {
        let schema = schema(types);
        assert!(supported(&schema));
        Self {
            packed: GroupValuesPacked::try_new(
                Arc::clone(&schema),
                unordered_multi_column,
            )
            .unwrap(),
            general: Box::new(GroupValuesRows::try_new(schema).unwrap()),
            batches: 0,
        }
    }

    fn intern(&mut self, cols: &[ArrayRef]) {
        let (mut actual, mut expected) = (Vec::new(), Vec::new());
        self.packed.intern(cols, &mut actual).unwrap();
        self.general.intern(cols, &mut expected).unwrap();
        assert_eq!(actual, expected, "group ids of batch {}", self.batches);
        assert_eq!(self.packed.len(), self.general.len());
        self.batches += 1;
    }

    fn emit(&mut self, emit_to: EmitTo) {
        let actual = self.packed.emit(emit_to).unwrap();
        let expected = self.general.emit(emit_to).unwrap();
        assert_same_arrays(&actual, &expected, &format!("{emit_to:?}"));
        assert_eq!(self.packed.len(), self.general.len());
    }

    fn clear_shrink(&mut self) {
        self.packed.clear_shrink(0);
        self.general.clear_shrink(0);
        assert_eq!(self.packed.len(), 0);
    }
}

fn run(types: &[DataType], gens: &[Gen], seed: u64, batches: usize, rows: usize) {
    let mut rng = Rng(seed);
    let mut check = Differential::new(types);
    for batch in 0..batches {
        let cols: Vec<ArrayRef> = types
            .iter()
            .zip(gens)
            .map(|(t, g)| column(&mut rng, t, *g, rows))
            .collect();
        check.intern(&cols);
        match batch % 7 {
            3 => check.emit(EmitTo::First(
                rng.below(check.packed.len() as u64 + 1) as usize
            )),
            5 if rng.chance(30) => check.emit(EmitTo::All),
            6 if rng.chance(10) => check.clear_shrink(),
            _ => {}
        }
    }
    check.emit(EmitTo::All);
    let size = check.packed.size();
    assert!(size > 0 || check.packed.is_empty());
}

#[test]
fn integer_pairs_with_nulls_and_growing_ranges() {
    for seed in 0..8 {
        run(
            &[DataType::Int32, DataType::Date32],
            &[
                Gen::Int {
                    card: 200 + seed * 500,
                    offset: -50,
                    nulls: 10,
                },
                Gen::Int {
                    card: 40,
                    offset: 10_000,
                    nulls: 3,
                },
            ],
            seed,
            24,
            1024,
        );
    }
}

#[test]
fn mixed_widths_booleans_and_decimals() {
    run(
        &[
            DataType::Int8,
            DataType::UInt8,
            DataType::Boolean,
            DataType::Decimal128(20, 2),
            DataType::Int16,
        ],
        &[
            Gen::Int {
                card: 256,
                offset: -128,
                nulls: 5,
            },
            Gen::Int {
                card: 7,
                offset: 250,
                nulls: 0,
            },
            Gen::Bool { nulls: 20 },
            Gen::Int {
                card: 50,
                offset: -25,
                nulls: 10,
            },
            Gen::Int {
                card: 3,
                offset: i16::MIN as i64,
                nulls: 50,
            },
        ],
        11,
        30,
        777,
    );
}

#[test]
fn strings_inline_external_empty_and_null() {
    for (seed, long) in [(1, 0), (2, 30), (3, 100)] {
        run(
            &[DataType::Utf8View, DataType::Utf8View],
            &[
                Gen::Str {
                    card: 60,
                    long,
                    nulls: 10,
                },
                Gen::Str {
                    card: 5,
                    long: 0,
                    nulls: 2,
                },
            ],
            seed,
            20,
            900,
        );
        run(
            &[DataType::BinaryView, DataType::Int64, DataType::Utf8View],
            &[
                Gen::Str {
                    card: 500,
                    long,
                    nulls: 5,
                },
                Gen::Int {
                    card: 1000,
                    offset: i64::MAX - 999,
                    nulls: 5,
                },
                Gen::Str {
                    card: 3,
                    long: 50,
                    nulls: 0,
                },
            ],
            seed,
            16,
            700,
        );
    }
}

#[test]
fn all_null_columns_keep_width_zero() {
    run(
        &[DataType::Int32, DataType::Utf8View, DataType::Int64],
        &[
            Gen::Int {
                card: 10,
                offset: 0,
                nulls: 100,
            },
            Gen::Str {
                card: 4,
                long: 0,
                nulls: 100,
            },
            Gen::Int {
                card: 10,
                offset: 5,
                nulls: 0,
            },
        ],
        5,
        10,
        512,
    );
}

/// A key domain of exactly `DIRECT_BITS` bits indexes directly; one more
/// value widens it to a hashed key, and both keep the same group ids.
#[test]
fn direct_index_boundary_and_widening() {
    let mut check = Differential::new(&[DataType::Int32, DataType::Int32]);
    // Codes 1..=127 need 7 bits; codes 1..=511 need 9: exactly 16 bits.
    let a = Arc::new(Int32Array::from_iter_values((0..511).map(|i| i % 127))) as ArrayRef;
    let b = Arc::new(Int32Array::from_iter_values(0..511)) as ArrayRef;
    check.intern(&[Arc::clone(&a), Arc::clone(&b)]);
    assert_eq!(check.packed.key_bits(), DIRECT_BITS);
    // 511 rows expect at most 511 groups: 2^16 entries is too large a table.
    assert!(!check.packed.is_direct());
    let mut direct = Differential::new(&[DataType::Int32, DataType::Int32]);
    let wide = (0..8192).map(|i| (i % 127, i % 511));
    let (x, y): (Vec<i32>, Vec<i32>) = wide.unzip();
    let (x, y) = (
        Arc::new(Int32Array::from(x)) as ArrayRef,
        Arc::new(Int32Array::from(y)) as ArrayRef,
    );
    direct.intern(&[Arc::clone(&x), Arc::clone(&y)]);
    assert_eq!(direct.packed.key_bits(), DIRECT_BITS);
    assert!(direct.packed.is_direct());
    direct.emit(EmitTo::First(1000));
    direct.intern(&[y, x]);
    assert!(!direct.packed.is_direct());
    direct.emit(EmitTo::All);
    check.intern(&[Arc::clone(&b), Arc::clone(&a)]);
    assert!(check.packed.key_bits() > DIRECT_BITS);
    assert!(!check.packed.is_fallback());
    check.emit(EmitTo::First(100));
    check.intern(&[a, b]);
    check.emit(EmitTo::All);
}

/// Keys wider than 64 bits move to the general interner with their ids.
#[test]
fn overflow_moves_groups_to_the_general_path() {
    let mut check =
        Differential::new(&[DataType::Int64, DataType::Int64, DataType::Utf8View]);
    let small = |n: i64| Arc::new(Int64Array::from_iter_values(0..n)) as ArrayRef;
    let s = Arc::new(StringViewArray::from_iter_values(
        (0..64).map(|i| format!("value-{}", i % 9)),
    )) as ArrayRef;
    check.intern(&[small(64), small(64), Arc::clone(&s)]);
    assert!(!check.packed.is_fallback());
    let wide = Arc::new(Int64Array::from_iter_values((0..64).map(|i| {
        if i % 2 == 0 {
            i64::MIN + i
        } else {
            i64::MAX - i
        }
    }))) as ArrayRef;
    check.intern(&[Arc::clone(&wide), wide, Arc::clone(&s)]);
    assert!(check.packed.is_fallback());
    check.intern(&[small(64), small(64), s]);
    check.emit(EmitTo::First(10));
    check.emit(EmitTo::All);
}

#[test]
fn unsupported_types_are_not_packed() {
    assert!(!supported(&schema(&[DataType::Int32, DataType::Float64])));
    assert!(!supported(&schema(&[DataType::Utf8, DataType::Int32])));
    assert!(supported(&schema(&[DataType::Utf8View, DataType::Boolean])));
}

#[test]
fn table_matches_a_map_in_both_modes() {
    for bits in [0, 3, DIRECT_BITS, DIRECT_BITS + 1, 32, 33, 40, 64] {
        let mut rng = Rng(u64::from(bits));
        let mut table = PackedKeyTable::new();
        table.relayout(bits, 1 << 16, |k| k);
        assert_eq!(table.is_direct(), bits <= DIRECT_BITS);
        let mut reference: HashMap<u64, usize> = HashMap::new();
        let mut order = Vec::new();
        let domain = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        // Above 1 MiB of slots the probe prefetches.
        let distinct = if bits > 20 {
            100_000
        } else {
            1 << bits.min(12)
        };
        let pool: Vec<u64> = (0..distinct)
            .map(|_| if domain == 0 { 0 } else { rng.next() & domain })
            .collect();
        for round in 0..6 {
            let keys: Vec<u64> = (0..20_000)
                .map(|_| pool[rng.below(distinct) as usize])
                .collect();
            let mut groups = Vec::new();
            table.intern(&keys, &mut groups);
            for (&key, &group) in keys.iter().zip(&groups) {
                let next = order.len();
                let expected = *reference.entry(key).or_insert_with(|| {
                    order.push(key);
                    next
                });
                assert_eq!(group, expected, "bits {bits} round {round}");
            }
            assert_eq!(table.keys(), &order[..]);
            if round == 2 {
                let n = order.len() / 3;
                table.drop_first(n);
                order.drain(..n);
                reference = order.iter().enumerate().map(|(g, &k)| (k, g)).collect();
            }
        }
        assert!(table.size() > 0);
        table.clear_shrink(0);
        assert!(table.is_empty());
    }
}

/// Each batch lowers the minimum, so every relayout shifts existing codes.
#[test]
fn falling_minimum_rebases_existing_groups() {
    let mut check = Differential::new(&[DataType::Int64, DataType::Int32]);
    for batch in 0..12_i64 {
        let low = -(1_i64 << (2 * batch));
        let a = Arc::new(Int64Array::from_iter(
            (0..300).map(|i| (i % 11 != 0).then_some(low + (i % 37))),
        )) as ArrayRef;
        let b =
            Arc::new(Int32Array::from_iter_values((0..300).map(|i| i % 5))) as ArrayRef;
        check.intern(&[a, b]);
        assert!(!check.packed.is_fallback());
        if batch == 6 {
            check.emit(EmitTo::First(40));
        }
    }
    check.emit(EmitTo::All);
}

/// Fixed-width columns alone never relayout; their shifts must still differ.
#[test]
fn boolean_only_keys_keep_distinct_shifts() {
    run(
        &[DataType::Boolean, DataType::Boolean, DataType::Boolean],
        &[
            Gen::Bool { nulls: 20 },
            Gen::Bool { nulls: 0 },
            Gen::Bool { nulls: 50 },
        ],
        9,
        12,
        256,
    );
}

/// Interning time per row of three key shapes, packed against the general
/// interner. Not a correctness test: run it on demand, in release, as
/// `scripts/vendored-tests.py datafusion-physical-plan --release
/// --run-ignored only -E 'test(packed_throughput)' --no-capture`.
#[test]
#[ignore]
fn packed_throughput() {
    use std::time::Instant;
    let rows = 8192;
    let batches = 400;
    let mut rng = Rng(42);
    let flags = |rng: &mut Rng, values: &[&str]| {
        // Runs of about four rows, as lineitem's flags arrive.
        let mut current = 0;
        Arc::new(StringViewArray::from_iter_values((0..rows).map(|_| {
            if rng.chance(25) {
                current = rng.below(values.len() as u64) as usize;
            }
            values[current]
        }))) as ArrayRef
    };
    type Shape = (&'static str, Vec<DataType>, Vec<Vec<ArrayRef>>);
    let shapes: Vec<Shape> = vec![
        (
            "two 1-byte strings, 4 groups",
            vec![DataType::Utf8View, DataType::Utf8View],
            (0..8)
                .map(|_| {
                    vec![
                        flags(&mut rng, &["A", "N", "R"]),
                        flags(&mut rng, &["F", "O"]),
                    ]
                })
                .collect(),
        ),
        (
            "two ints, 1M groups",
            vec![DataType::Int32, DataType::Date32],
            (0..8)
                .map(|_| {
                    vec![
                        column(
                            &mut rng,
                            &DataType::Int32,
                            Gen::Int {
                                card: 20_000,
                                offset: 1,
                                nulls: 1,
                            },
                            rows,
                        ),
                        column(
                            &mut rng,
                            &DataType::Date32,
                            Gen::Int {
                                card: 50,
                                offset: 10_000,
                                nulls: 0,
                            },
                            rows,
                        ),
                    ]
                })
                .collect(),
        ),
        (
            "4 strings + 2 ints, 50K groups",
            vec![
                DataType::Utf8View,
                DataType::Utf8View,
                DataType::Utf8View,
                DataType::Utf8View,
                DataType::Int32,
                DataType::Int32,
            ],
            (0..8)
                .map(|_| {
                    vec![
                        column(
                            &mut rng,
                            &DataType::Utf8View,
                            Gen::Str {
                                card: 10,
                                long: 0,
                                nulls: 1,
                            },
                            rows,
                        ),
                        column(
                            &mut rng,
                            &DataType::Utf8View,
                            Gen::Str {
                                card: 500,
                                long: 100,
                                nulls: 1,
                            },
                            rows,
                        ),
                        column(
                            &mut rng,
                            &DataType::Utf8View,
                            Gen::Str {
                                card: 6,
                                long: 0,
                                nulls: 0,
                            },
                            rows,
                        ),
                        column(
                            &mut rng,
                            &DataType::Utf8View,
                            Gen::Str {
                                card: 1,
                                long: 0,
                                nulls: 0,
                            },
                            rows,
                        ),
                        column(
                            &mut rng,
                            &DataType::Int32,
                            Gen::Int {
                                card: 1,
                                offset: 1999,
                                nulls: 0,
                            },
                            rows,
                        ),
                        column(
                            &mut rng,
                            &DataType::Int32,
                            Gen::Int {
                                card: 12,
                                offset: 1,
                                nulls: 0,
                            },
                            rows,
                        ),
                    ]
                })
                .collect(),
        ),
        (
            "16-byte id, 2 names, year: 100K groups",
            vec![
                DataType::Utf8View,
                DataType::Utf8View,
                DataType::Utf8View,
                DataType::Int32,
            ],
            (0..8)
                .map(|_| {
                    let ids: Vec<u64> = (0..rows).map(|_| rng.below(50_000)).collect();
                    let id = Arc::new(StringViewArray::from_iter_values(
                        ids.iter().map(|i| format!("AAAAAAAA{i:08}")),
                    )) as ArrayRef;
                    let first = Arc::new(StringViewArray::from_iter_values(
                        ids.iter().map(|i| format!("First{}", i % 3000)),
                    )) as ArrayRef;
                    let last = Arc::new(StringViewArray::from_iter_values(
                        ids.iter().map(|i| format!("Lastname{}", i % 5000)),
                    )) as ArrayRef;
                    let year = column(
                        &mut rng,
                        &DataType::Int32,
                        Gen::Int {
                            card: 2,
                            offset: 2001,
                            nulls: 0,
                        },
                        rows,
                    );
                    vec![id, first, last, year]
                })
                .collect(),
        ),
        (
            "near-unique int pairs: 2M groups",
            vec![DataType::Int32, DataType::Int32],
            (0..256)
                .map(|b| {
                    let pairs: Vec<(i32, i32)> = (0..rows as i32)
                        .map(|i| {
                            let n = b * rows as i32 + i;
                            (1 + n % 2_000_000 * 7 % 1_999_993, 1 + n / 7 % 200_000)
                        })
                        .collect();
                    vec![
                        Arc::new(Int32Array::from_iter_values(pairs.iter().map(|p| p.0)))
                            as ArrayRef,
                        Arc::new(Int32Array::from_iter_values(pairs.iter().map(|p| p.1)))
                            as ArrayRef,
                    ]
                })
                .collect(),
        ),
    ];
    let only = std::env::var("PACKED_SHAPE").ok();
    for (index, (name, types, inputs)) in shapes.into_iter().enumerate() {
        if only.as_ref().is_some_and(|o| *o != index.to_string()) {
            continue;
        }
        let schema = schema(&types);
        for arm in ["general", "packed"] {
            if std::env::var("PACKED_ARM").is_ok_and(|a| a != arm) {
                continue;
            }
            let mut values: Box<dyn GroupValues> = if arm == "packed" {
                Box::new(
                    GroupValuesPacked::try_new(
                        Arc::clone(&schema),
                        unordered_multi_column,
                    )
                    .unwrap(),
                )
            } else {
                unordered_multi_column(Arc::clone(&schema)).unwrap()
            };
            let mut groups = Vec::new();
            let start = Instant::now();
            // Shapes with more inputs than `batches` intern each one once.
            let count = batches.max(inputs.len());
            for batch in 0..count {
                values
                    .intern(&inputs[batch % inputs.len()], &mut groups)
                    .unwrap();
            }
            let ns = start.elapsed().as_nanos() as f64 / (rows * count) as f64;
            println!("{name:32} {arm:8} {ns:7.2} ns/row  {} groups", values.len());
        }
    }
}

/// Short strings use offset codes; a longer value, or a range above 16 bits,
/// converts the column to a dictionary and keeps every group id.
#[test]
fn short_strings_convert_to_a_dictionary() {
    for later in ["an-out-of-line-value", "abcd", "zz"] {
        let mut check = Differential::new(&[DataType::Utf8View, DataType::Int32]);
        let flags = |values: &[Option<&str>]| {
            Arc::new(StringViewArray::from_iter(values.iter().copied())) as ArrayRef
        };
        let ints = |n: i32| Arc::new(Int32Array::from_iter_values(0..n)) as ArrayRef;
        check.intern(&[
            flags(&[Some("A"), Some("N"), None, Some("R"), Some("")]),
            ints(5),
        ]);
        check.intern(&[
            flags(&[Some("N"), Some("A"), Some("A"), None, Some("R")]),
            ints(5),
        ]);
        check.emit(EmitTo::First(2));
        // "abcd" widens the short range past 16 bits; the others are long or new.
        check.intern(&[
            flags(&[
                Some(later),
                Some("A"),
                None,
                Some(later),
                Some("R"),
                Some("N"),
            ]),
            ints(6),
        ]);
        check.intern(&[flags(&[Some("A"), Some(later), Some("Q")]), ints(3)]);
        check.emit(EmitTo::All);
        check.intern(&[flags(&[Some("A"), Some(later)]), ints(2)]);
        check.emit(EmitTo::All);
    }
}

/// A conversion and an overflow in one batch: the general interner receives
/// the groups decoded through the pending conversion.
#[test]
fn overflow_during_conversion_keeps_short_groups() {
    let mut check = Differential::new(&[DataType::Utf8View, DataType::Int64]);
    let s = |v: &[&str]| Arc::new(StringViewArray::from_iter_values(v)) as ArrayRef;
    let i = |v: &[i64]| Arc::new(Int64Array::from(v.to_vec())) as ArrayRef;
    check.intern(&[s(&["a", "b", "a", "c"]), i(&[1, 2, 1, 3])]);
    assert!(!check.packed.is_fallback());
    check.intern(&[
        s(&["a-long-value-beyond-inline", "b", "c"]),
        i(&[i64::MIN, i64::MAX, 2]),
    ]);
    assert!(check.packed.is_fallback());
    check.intern(&[
        s(&["a", "c", "a-long-value-beyond-inline"]),
        i(&[1, 3, i64::MIN]),
    ]);
    check.emit(EmitTo::All);
}

/// Many distinct external values whose views never repeat make the
/// dictionary costly; the groups move to the general interner with their ids.
#[test]
fn costly_dictionary_moves_to_the_general_path() {
    let mut check = Differential::new(&[DataType::Utf8View, DataType::Int32]);
    for batch in 0..10 {
        // A first batch below the decision threshold leaves groups to move.
        let rows = if batch == 0 { 512 } else { 8192 };
        let ids =
            Arc::new(StringViewArray::from_iter_values((0..rows).map(|i| {
                format!("customer-identifier-{}", (batch * 8192 + i) % 20_000)
            }))) as ArrayRef;
        let year = Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|i| 2000 + i % 2),
        )) as ArrayRef;
        check.intern(&[ids, year]);
        assert_eq!(check.packed.is_fallback(), batch >= 1, "batch {batch}");
    }
    check.emit(EmitTo::All);
}

/// A flush (`clear_shrink` with retained capacity) followed by relayouts
/// through direct mode keeps the retained hash slots: bounded partial
/// aggregation flushes and refills without regrowing the table.
#[test]
fn clear_shrink_keeps_retained_slots_across_relayouts() {
    let mut table = PackedKeyTable::new();
    table.relayout(40, 1 << 20, |k| k);
    let keys: Vec<u64> = (0..100_000u64).map(|k| k * 7_919).collect();
    let mut groups = Vec::new();
    table.intern(&keys, &mut groups);
    table.clear_shrink(50_000);
    let retained = table.size();
    assert!(retained >= 65_536 * 16, "{retained}");
    table.relayout(0, 0, |k| k);
    assert!(table.is_direct());
    table.relayout(40, 0, |k| k);
    assert!(!table.is_direct());
    assert!(table.size() >= retained);
    table.intern(&keys[..1000], &mut groups);
    assert_eq!(groups, (0..1000).collect::<Vec<_>>());
    assert!(table.size() >= retained);
}

/// Extreme values of 64- and 128-bit types: ranges near the type limits,
/// mixed with NULLs, and spans that overflow the packed key.
#[test]
fn integer_extremes_of_wide_types() {
    use arrow::array::{Decimal128Array, UInt64Array};
    let mut check = Differential::new(&[DataType::Int64, DataType::UInt64]);
    let a = Arc::new(Int64Array::from(vec![
        Some(i64::MAX),
        Some(i64::MAX - 1),
        None,
        Some(i64::MAX),
    ])) as ArrayRef;
    let b = Arc::new(UInt64Array::from(vec![
        Some(u64::MAX),
        None,
        Some(u64::MAX - 3),
        Some(u64::MAX),
    ])) as ArrayRef;
    check.intern(&[Arc::clone(&a), Arc::clone(&b)]);
    assert!(!check.packed.is_fallback());
    let low = Arc::new(Int64Array::from(vec![i64::MIN, i64::MIN + 1])) as ArrayRef;
    let zero = Arc::new(UInt64Array::from(vec![0, 1])) as ArrayRef;
    check.intern(&[low, zero]);
    assert!(check.packed.is_fallback());
    check.intern(&[a, b]);
    check.emit(EmitTo::All);

    let decimal = |v: Vec<Option<i128>>| {
        Arc::new(
            Decimal128Array::from(v)
                .with_precision_and_scale(38, 0)
                .unwrap(),
        ) as ArrayRef
    };
    let mut check = Differential::new(&[DataType::Decimal128(38, 0), DataType::Int8]);
    let small = Arc::new(Int8Array::from(vec![i8::MIN, i8::MAX, 0])) as ArrayRef;
    check.intern(&[
        decimal(vec![Some(i128::MAX), Some(i128::MAX - 5), None]),
        Arc::clone(&small),
    ]);
    assert!(!check.packed.is_fallback());
    check.intern(&[
        decimal(vec![Some(i128::MIN), Some(0), Some(i128::MAX)]),
        small,
    ]);
    assert!(check.packed.is_fallback());
    check.emit(EmitTo::All);
}

/// Sliced inputs: offsets into values, views and validity bitmaps.
#[test]
fn sliced_input_arrays() {
    let mut rng = Rng(77);
    let types = [DataType::Int32, DataType::Utf8View, DataType::Boolean];
    let gens = [
        Gen::Int {
            card: 300,
            offset: -7,
            nulls: 15,
        },
        Gen::Str {
            card: 40,
            long: 30,
            nulls: 10,
        },
        Gen::Bool { nulls: 10 },
    ];
    let mut check = Differential::new(&types);
    for batch in 0..12 {
        let cols: Vec<ArrayRef> = types
            .iter()
            .zip(&gens)
            .map(|(t, g)| column(&mut rng, t, *g, 1000))
            .collect();
        let (offset, len) = (3 + batch * 7, 500 + batch * 31);
        let sliced: Vec<ArrayRef> = cols.iter().map(|c| c.slice(offset, len)).collect();
        check.intern(&sliced);
        if batch == 6 {
            check.emit(EmitTo::First(20));
        }
    }
    check.emit(EmitTo::All);
}

/// BinaryView keys that are not valid UTF-8, inline and external.
#[test]
fn binary_views_need_not_be_utf8() {
    let mut check = Differential::new(&[DataType::BinaryView, DataType::Int32]);
    let bytes: Vec<Vec<u8>> = vec![
        vec![0xff],
        vec![0xc3, 0x28],
        vec![0x80; 3],
        vec![0xfe; 20],
        vec![
            0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff,
        ],
        vec![],
    ];
    for round in 0..4 {
        let keys =
            Arc::new(BinaryViewArray::from_iter((0..60).map(|i| {
                (i % 7 != 6).then(|| bytes[(i + round) % bytes.len()].as_slice())
            }))) as ArrayRef;
        let ints =
            Arc::new(Int32Array::from_iter_values((0..60).map(|i| i % 3))) as ArrayRef;
        check.intern(&[keys, ints]);
    }
    check.emit(EmitTo::All);
}

/// An emitted string column holds only the bytes of its own groups, not the
/// whole dictionary; after a full emission the dictionary starts empty.
#[test]
fn emission_copies_only_referenced_bytes() {
    let schema = schema(&[DataType::Utf8View, DataType::Int32]);
    let mut packed =
        GroupValuesPacked::try_new(Arc::clone(&schema), unordered_multi_column).unwrap();
    let long = |i: usize| format!("a-long-dictionary-value-number-{i:04}");
    let keys =
        Arc::new(StringViewArray::from_iter_values((0..400).map(long))) as ArrayRef;
    let ints = Arc::new(Int32Array::from_iter_values(0..400)) as ArrayRef;
    packed.intern(&[keys, ints], &mut Vec::new()).unwrap();
    let first = packed.emit(EmitTo::First(10)).unwrap();
    let view = first[0].as_string_view();
    let held: usize = view.data_buffers().iter().map(|b| b.len()).sum();
    assert_eq!(held, (0..10).map(|i| long(i).len()).sum::<usize>());
    assert_eq!(view.value(3), long(3));
    let before = packed.size();
    let rest = packed.emit(EmitTo::All).unwrap();
    assert_eq!(rest[0].len(), 390);
    assert_eq!(rest[0].as_string_view().value(0), long(10));
    let keys = Arc::new(StringViewArray::from_iter_values(["x-short", "y"])) as ArrayRef;
    let ints = Arc::new(Int32Array::from_iter_values(0..2)) as ArrayRef;
    packed.intern(&[keys, ints], &mut Vec::new()).unwrap();
    assert!(
        packed.size() < before,
        "dictionary kept after full emission"
    );
}

/// Mostly distinct external values move to the general path within the
/// first batch, before the dictionary learns the whole batch.
#[test]
fn mostly_distinct_strings_fall_back_within_a_batch() {
    let mut check = Differential::new(&[DataType::Utf8View, DataType::Int32]);
    let keys = Arc::new(StringViewArray::from_iter_values(
        (0..4096).map(|i| format!("item-description-that-is-long-{i}")),
    )) as ArrayRef;
    let ints =
        Arc::new(Int32Array::from_iter_values((0..4096).map(|i| i % 5))) as ArrayRef;
    check.intern(&[keys, ints]);
    assert!(check.packed.is_fallback());
    check.emit(EmitTo::All);
}

/// Pinned input buffers of at most 4 MiB count in `size()`; a larger one (a
/// join's build side) is pinned uncounted; every emission releases pins.
#[test]
fn only_small_input_buffers_are_pinned_and_counted() {
    use arrow::array::StringViewBuilder;
    let pinned_size = |block: u32| {
        let schema = schema(&[DataType::Utf8View, DataType::Int32]);
        let mut packed =
            GroupValuesPacked::try_new(Arc::clone(&schema), unordered_multi_column)
                .unwrap();
        let mut builder = StringViewBuilder::new().with_fixed_block_size(block);
        for i in 0..64 {
            builder.append_value(format!("a-value-longer-than-inline-{}", i % 4));
        }
        let keys = Arc::new(builder.finish()) as ArrayRef;
        let ints = Arc::new(Int32Array::from_iter_values(0..64)) as ArrayRef;
        packed.intern(&[keys, ints], &mut Vec::new()).unwrap();
        packed.size()
    };
    // A 16 MiB block is above the pin limit: it must not enter size().
    let small = pinned_size(1 << 12);
    let large = pinned_size(16 << 20);
    assert!(small >= 1 << 12, "{small}");
    assert!(large < 8 << 20, "{large}");
}

/// An emission releases the pinned input buffers and their accounting.
#[test]
fn emission_releases_pinned_buffers() {
    use arrow::array::StringViewBuilder;
    let schema = schema(&[DataType::Utf8View, DataType::Int32]);
    let mut packed =
        GroupValuesPacked::try_new(Arc::clone(&schema), unordered_multi_column).unwrap();
    let mut builder = StringViewBuilder::new().with_fixed_block_size(1 << 16);
    for i in 0..64 {
        builder.append_value(format!("a-value-longer-than-inline-{}", i % 4));
    }
    let keys = Arc::new(builder.finish()) as ArrayRef;
    let ints = Arc::new(Int32Array::from_iter_values(0..64)) as ArrayRef;
    packed.intern(&[keys, ints], &mut Vec::new()).unwrap();
    let pinned = packed.size();
    packed.emit(EmitTo::First(1)).unwrap();
    assert!(
        packed.size() + (1 << 16) <= pinned,
        "{} {pinned}",
        packed.size()
    );
}

/// Existing groups move between 8-byte and 16-byte slots when the layout
/// crosses 32 bits, both ways, with unchanged ids.
#[test]
fn relayout_moves_groups_between_slot_sizes() {
    let mut table = PackedKeyTable::new();
    table.relayout(24, 1 << 16, |k| k);
    let keys: Vec<u64> = (0..50_000u64)
        .map(|k| (k * 2_654_435_761) & 0xff_ffff)
        .collect();
    let mut first = Vec::new();
    table.intern(&keys, &mut first);
    // Widen: shift every key up 20 bits (injective), crossing 32 bits.
    table.relayout(44, 0, |k| k << 20);
    let wide: Vec<u64> = keys.iter().map(|k| k << 20).collect();
    let mut again = Vec::new();
    table.intern(&wide, &mut again);
    assert_eq!(first, again);
    // Narrow again.
    table.relayout(24, 0, |k| k >> 20);
    table.intern(&keys, &mut again);
    assert_eq!(first, again);
}
