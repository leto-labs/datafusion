// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Nanoseconds per input row of `GroupValues::intern` and of the final
//! `emit`, through the production dispatch (`new_group_values` with
//! unordered keys). The key shapes follow the group-by keys of H2O q1-q10,
//! TPC-H q1, q3 and q18 and ClickBench (UserID, SearchPhrase, RegionID,
//! WatchID and ClientIP). Input preparation is untimed.
//!
//! Usage: `cargo bench -p datafusion-physical-plan --bench
//! group_values_interning -- [filter] [--rows N] [--reps K]`. Prints one
//! tab-separated line per case: name, rows, groups, intern ns/row (median
//! of K repetitions, a fresh table each), emit ns/row, table bytes.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Date32Array, Int32Array, Int64Array, StringViewArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_expr::EmitTo;
use datafusion_physical_plan::aggregates::group_values::new_group_values;
use datafusion_physical_plan::aggregates::order::GroupClustering;

const BATCH: usize = 8192;

/// xorshift64*: deterministic and cheap.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        ((u128::from(self.next()) * u128::from(n)) >> 64) as u64
    }

    /// A Zipf-like draw on `0..n`: the inverse of a power-law CDF.
    fn skewed(&mut self, n: u64) -> u64 {
        let u = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        ((n as f64).powf(u) - 1.0) as u64 % n
    }
}

/// One key column, generated row by row from a shared row index.
#[derive(Clone, Copy)]
enum Gen {
    /// Uniform `0..n`, formatted as `id{:0width$}` (H2O id1..id3).
    Str { n: u64, width: usize },
    /// Skewed `0..n` strings of varying length; value 0 is the empty string
    /// (ClickBench SearchPhrase).
    SkewStr { n: u64 },
    /// Uniform `1..=n` (H2O id4..id6).
    I32 { n: u64 },
    /// Skewed `0..n` (ClickBench RegionID).
    SkewI32 { n: u64 },
    /// Skewed `0..n`, scattered over i64 (ClickBench UserID).
    SkewI64 { n: u64 },
    /// Random i64 (ClickBench WatchID).
    RandI64,
    /// Ascending with runs of 1..=7 (TPC-H l_orderkey).
    RunI64,
    /// A function of the run key (TPC-H o_orderdate for l_orderkey).
    DateOfRun,
    /// A small function of the run key (TPC-H o_shippriority).
    ZeroOfRun,
}

struct Case {
    name: &'static str,
    columns: Vec<Gen>,
}

fn data_type(generator: Gen) -> DataType {
    match generator {
        Gen::Str { .. } | Gen::SkewStr { .. } => DataType::Utf8View,
        Gen::I32 { .. } | Gen::SkewI32 { .. } | Gen::ZeroOfRun => DataType::Int32,
        Gen::SkewI64 { .. } | Gen::RandI64 | Gen::RunI64 => DataType::Int64,
        Gen::DateOfRun => DataType::Date32,
    }
}

fn cases() -> Vec<Case> {
    use Gen::*;
    let k = 100;
    vec![
        // H2O G1_1e8_1e2: id1/id2 have K values, id3 N/K (1e6 at SF10).
        Case {
            name: "h2o_q1_str100",
            columns: vec![Str { n: k, width: 3 }],
        },
        Case {
            name: "h2o_q2_str100x2",
            columns: vec![Str { n: k, width: 3 }, Str { n: k, width: 3 }],
        },
        Case {
            name: "h2o_q3_str1e6",
            columns: vec![Str {
                n: 1_000_000,
                width: 10,
            }],
        },
        Case {
            name: "h2o_q4_i32_100",
            columns: vec![I32 { n: k }],
        },
        Case {
            name: "h2o_q5_i32_1e6",
            columns: vec![I32 { n: 1_000_000 }],
        },
        Case {
            name: "h2o_q6_i32x2",
            columns: vec![I32 { n: k }, I32 { n: k }],
        },
        Case {
            name: "h2o_q9_str_i32",
            columns: vec![Str { n: k, width: 3 }, I32 { n: k }],
        },
        Case {
            name: "h2o_q10_6col",
            columns: vec![
                Str { n: k, width: 3 },
                Str { n: k, width: 3 },
                Str {
                    n: 1_000_000,
                    width: 10,
                },
                I32 { n: k },
                I32 { n: k },
                I32 { n: 1_000_000 },
            ],
        },
        Case {
            name: "tpch_q1_str3x2",
            columns: vec![Str { n: 3, width: 1 }, Str { n: 2, width: 1 }],
        },
        Case {
            name: "tpch_q3_key_date_pri",
            columns: vec![RunI64, DateOfRun, ZeroOfRun],
        },
        Case {
            name: "tpch_q18_runs_i64",
            columns: vec![RunI64],
        },
        Case {
            name: "cb_regionid",
            columns: vec![SkewI32 { n: 9_000 }],
        },
        Case {
            name: "cb_userid",
            columns: vec![SkewI64 { n: 4_000_000 }],
        },
        Case {
            name: "cb_searchphrase",
            columns: vec![SkewStr { n: 1_000_000 }],
        },
        Case {
            name: "cb_userid_phrase",
            columns: vec![SkewI64 { n: 4_000_000 }, SkewStr { n: 1_000_000 }],
        },
        Case {
            name: "cb_watchid_clientip",
            columns: vec![RandI64, SkewI32 { n: 1 << 30 }],
        },
    ]
}

/// Scatter `0..n` over the i64 range, injectively.
fn scatter(value: u64) -> i64 {
    value.wrapping_mul(0x9E37_79B9_7F4A_7C15) as i64
}

fn generate(case: &Case, rows: usize) -> Vec<Vec<ArrayRef>> {
    let mut rngs: Vec<Rng> = (0..case.columns.len())
        .map(|i| Rng(0x5EED_0000 + i as u64 * 7919))
        .collect();
    // Run state for RunI64 and its dependents: one shared sequence.
    let mut run_rng = Rng(0xC0FFEE);
    let mut run_key = 1_i64;
    let mut run_left = 0_u64;
    let mut batches = Vec::new();
    let mut start = 0;
    while start < rows {
        let len = BATCH.min(rows - start);
        let mut run_keys = Vec::with_capacity(len);
        for _ in 0..len {
            if run_left == 0 {
                run_key += 1 + run_rng.below(3) as i64;
                run_left = 1 + run_rng.below(7);
            }
            run_left -= 1;
            run_keys.push(run_key);
        }
        let columns = case
            .columns
            .iter()
            .zip(rngs.iter_mut())
            .map(|(&generator, rng)| -> ArrayRef {
                match generator {
                    Gen::Str { n, width } => Arc::new(StringViewArray::from_iter_values(
                        (0..len).map(|_| format!("id{:0width$}", 1 + rng.below(n))),
                    )),
                    Gen::SkewStr { n } => {
                        Arc::new(StringViewArray::from_iter_values((0..len).map(|_| {
                            match rng.skewed(n) {
                                0 => String::new(),
                                v => format!(
                                    "phrase {v} {}",
                                    "x".repeat((v % 23) as usize)
                                ),
                            }
                        })))
                    }
                    Gen::I32 { n } => Arc::new(Int32Array::from_iter_values(
                        (0..len).map(|_| 1 + rng.below(n) as i32),
                    )),
                    Gen::SkewI32 { n } => Arc::new(Int32Array::from_iter_values(
                        (0..len).map(|_| rng.skewed(n) as i32),
                    )),
                    Gen::SkewI64 { n } => Arc::new(Int64Array::from_iter_values(
                        (0..len).map(|_| scatter(rng.skewed(n))),
                    )),
                    Gen::RandI64 => Arc::new(Int64Array::from_iter_values(
                        (0..len).map(|_| rng.next() as i64),
                    )),
                    Gen::RunI64 => Arc::new(Int64Array::from(run_keys.clone())),
                    Gen::DateOfRun => Arc::new(Date32Array::from_iter_values(
                        run_keys.iter().map(|&k| 8000 + (k % 2400) as i32),
                    )),
                    Gen::ZeroOfRun => {
                        Arc::new(Int32Array::from_iter_values(run_keys.iter().map(|_| 0)))
                    }
                }
            })
            .collect();
        batches.push(columns);
        start += len;
    }
    batches
}

fn schema(case: &Case) -> SchemaRef {
    Arc::new(Schema::new(
        case.columns
            .iter()
            .enumerate()
            .map(|(i, &g)| Field::new(format!("k{i}"), data_type(g), false))
            .collect::<Vec<_>>(),
    ))
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let mut rows = 4_000_000;
    let mut reps = 5;
    let mut filter = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--rows" => {
                rows = args.next().and_then(|v| v.parse().ok()).expect("--rows N")
            }
            "--reps" => {
                reps = args.next().and_then(|v| v.parse().ok()).expect("--reps K")
            }
            "--bench" => {}
            other if !other.starts_with('-') => filter = Some(other.to_string()),
            _ => {}
        }
    }
    println!("case\trows\tgroups\tintern_ns_per_row\temit_ns_per_row\ttable_bytes");
    for case in cases() {
        if filter
            .as_ref()
            .is_some_and(|f| !case.name.contains(f.as_str()))
        {
            continue;
        }
        let batches = generate(&case, rows);
        let schema = schema(&case);
        let mut intern = Vec::with_capacity(reps);
        let mut emit = Vec::with_capacity(reps);
        let (mut groups, mut bytes) = (0, 0);
        let mut ids = Vec::with_capacity(BATCH);
        for _ in 0..reps {
            let mut values =
                new_group_values(Arc::clone(&schema), &GroupClustering::None).unwrap();
            let start = Instant::now();
            for batch in &batches {
                values.intern(batch, &mut ids).unwrap();
                black_box(&ids);
            }
            intern.push(start.elapsed().as_nanos() as f64 / rows as f64);
            groups = values.len();
            bytes = values.size();
            let start = Instant::now();
            black_box(values.emit(EmitTo::All).unwrap());
            emit.push(start.elapsed().as_nanos() as f64 / rows as f64);
        }
        println!(
            "{}\t{rows}\t{groups}\t{:.2}\t{:.2}\t{bytes}",
            case.name,
            median(&mut intern),
            median(&mut emit)
        );
    }
}
