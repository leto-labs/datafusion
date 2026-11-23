// Added by the Longbow authors (review of family JT 9); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Review of Longbow patch JT 9 (leto-mjj.71): `HashJoinExec` with inline
//! join keys forced on, against the same join with them off and against an
//! oracle that shares no code with the join (a `HashMap` over the generated
//! key tuples).
//!
//! Every case draws its key types, value ranges, NULLs, join type, NULL
//! equality, residual filter and build mode from a seed. The ranges are
//! chosen so that the encoded keys land on each side of the two limits of
//! the encoding: the packed slot (the bits left beside the head and a
//! 10-bit tag) and 64 bits.
use std::any::Any;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::shared_build::OPTION;
use super::*;
use crate::common;
use crate::filter::FilterExecBuilder;
use crate::joins::hash_join::stream::JT9_REVIEW_PROBES;
use crate::joins::join_hash_map::{JoinHashMapU32, PartitionedBuild, SlotLayout};
use crate::joins::utils::{ColumnIndex, JoinFilter};
use crate::repartition::RepartitionExec;
use crate::test::TestMemoryExec;
use arrow::array::{
    ArrowPrimitiveType, BooleanArray, Int32Array, Int64Array, PrimitiveArray,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::*;
use datafusion_common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
use datafusion_execution::config::SessionConfig;
use datafusion_execution::memory_pool::{
    GreedyMemoryPool, MemoryLimit, MemoryPool, MemoryReservation,
};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::BinaryExpr;

#[derive(Clone, Debug)]
struct ParallelRows(usize);

impl ConfigExtension for ParallelRows {
    const PREFIX: &'static str = "longbow";
}

impl ExtensionOptions for ParallelRows {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }
    fn set(&mut self, _: &str, value: &str) -> Result<()> {
        self.0 = value.parse().unwrap();
        Ok(())
    }
    fn entries(&self) -> Vec<ConfigEntry> {
        vec![ConfigEntry {
            key: OPTION.into(),
            value: Some(self.0.to_string()),
            description: "test",
        }]
    }
}

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn wide(&mut self) -> u128 {
        (u128::from(self.next()) << 64) | u128::from(self.next())
    }
    /// Uniform in `0..=max`.
    fn upto(&mut self, max: u128) -> u128 {
        if max == u128::MAX {
            self.wide()
        } else {
            self.wide() % (max + 1)
        }
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

/// A key type with the range of its values.
#[derive(Clone, Debug)]
struct KeyType {
    data_type: DataType,
    lo: i128,
    hi: i128,
}

impl KeyType {
    fn bits(&self) -> u32 {
        128 - ((self.hi - self.lo) as u128).leading_zeros()
    }
}

fn key_types() -> Vec<KeyType> {
    let of = |data_type: DataType, lo: i128, hi: i128| KeyType { data_type, lo, hi };
    let i32s = (i128::from(i32::MIN), i128::from(i32::MAX));
    let i64s = (i128::from(i64::MIN), i128::from(i64::MAX));
    let utc: Option<Arc<str>> = Some("UTC".into());
    let offset: Option<Arc<str>> = Some("+01:00".into());
    vec![
        of(DataType::Int8, i128::from(i8::MIN), i128::from(i8::MAX)),
        of(DataType::Int16, i128::from(i16::MIN), i128::from(i16::MAX)),
        of(DataType::Int32, i32s.0, i32s.1),
        of(DataType::Int64, i64s.0, i64s.1),
        of(DataType::UInt8, 0, i128::from(u8::MAX)),
        of(DataType::UInt16, 0, i128::from(u16::MAX)),
        of(DataType::UInt32, 0, i128::from(u32::MAX)),
        of(DataType::UInt64, 0, i128::from(u64::MAX)),
        of(DataType::Date32, i32s.0, i32s.1),
        of(DataType::Date64, i64s.0, i64s.1),
        of(DataType::Time32(TimeUnit::Second), i32s.0, i32s.1),
        of(DataType::Time32(TimeUnit::Millisecond), i32s.0, i32s.1),
        of(DataType::Time64(TimeUnit::Microsecond), i64s.0, i64s.1),
        of(DataType::Time64(TimeUnit::Nanosecond), i64s.0, i64s.1),
        of(DataType::Timestamp(TimeUnit::Second, None), i64s.0, i64s.1),
        of(
            DataType::Timestamp(TimeUnit::Millisecond, utc),
            i64s.0,
            i64s.1,
        ),
        of(
            DataType::Timestamp(TimeUnit::Microsecond, None),
            i64s.0,
            i64s.1,
        ),
        of(
            DataType::Timestamp(TimeUnit::Nanosecond, offset),
            i64s.0,
            i64s.1,
        ),
        of(DataType::Duration(TimeUnit::Second), i64s.0, i64s.1),
        of(DataType::Duration(TimeUnit::Millisecond), i64s.0, i64s.1),
        of(DataType::Duration(TimeUnit::Microsecond), i64s.0, i64s.1),
        of(DataType::Duration(TimeUnit::Nanosecond), i64s.0, i64s.1),
        // No inline encoding: the join must keep the key comparison.
        of(DataType::Decimal128(38, 0), -(1 << 100), 1 << 100),
        of(DataType::Decimal128(18, 2), -(1 << 59), 1 << 59),
    ]
}

/// `values` as an array of `data_type`; a NULL row keeps its value in the
/// buffer, so a join that read the values of NULL keys would find matches.
fn key_array(data_type: &DataType, values: &[i128], valid: Option<&[bool]>) -> ArrayRef {
    let nulls = valid.map(|valid| NullBuffer::from(valid.to_vec()));
    macro_rules! primitive {
        ($t:ty) => {{
            let values: Vec<<$t as ArrowPrimitiveType>::Native> =
                values.iter().map(|&value| value as _).collect();
            Arc::new(
                PrimitiveArray::<$t>::new(values.into(), nulls)
                    .with_data_type(data_type.clone()),
            ) as ArrayRef
        }};
    }
    match data_type {
        DataType::Int8 => primitive!(Int8Type),
        DataType::Int16 => primitive!(Int16Type),
        DataType::Int32 => primitive!(Int32Type),
        DataType::Int64 => primitive!(Int64Type),
        DataType::UInt8 => primitive!(UInt8Type),
        DataType::UInt16 => primitive!(UInt16Type),
        DataType::UInt32 => primitive!(UInt32Type),
        DataType::UInt64 => primitive!(UInt64Type),
        DataType::Date32 => primitive!(Date32Type),
        DataType::Date64 => primitive!(Date64Type),
        DataType::Time32(TimeUnit::Second) => primitive!(Time32SecondType),
        DataType::Time32(TimeUnit::Millisecond) => primitive!(Time32MillisecondType),
        DataType::Time64(TimeUnit::Microsecond) => primitive!(Time64MicrosecondType),
        DataType::Time64(TimeUnit::Nanosecond) => primitive!(Time64NanosecondType),
        DataType::Timestamp(TimeUnit::Second, _) => primitive!(TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            primitive!(TimestampMillisecondType)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            primitive!(TimestampMicrosecondType)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            primitive!(TimestampNanosecondType)
        }
        DataType::Duration(TimeUnit::Second) => primitive!(DurationSecondType),
        DataType::Duration(TimeUnit::Millisecond) => primitive!(DurationMillisecondType),
        DataType::Duration(TimeUnit::Microsecond) => primitive!(DurationMicrosecondType),
        DataType::Duration(TimeUnit::Nanosecond) => primitive!(DurationNanosecondType),
        DataType::Decimal128(_, _) => primitive!(Decimal128Type),
        other => panic!("no generator for {other}"),
    }
}

/// One side of a join: key columns, their validity, and a filter column.
#[derive(Default)]
struct Side {
    keys: Vec<Vec<i128>>,
    valid: Vec<Option<Vec<bool>>>,
    filter: Vec<i32>,
}

impl Side {
    fn rows(&self) -> usize {
        self.filter.len()
    }

    /// The key of `row`, `None` where it is NULL.
    fn key(&self, row: usize) -> Vec<Option<i128>> {
        self.keys
            .iter()
            .zip(&self.valid)
            .map(|(values, valid)| {
                valid
                    .as_ref()
                    .is_none_or(|valid| valid[row])
                    .then_some(values[row])
            })
            .collect()
    }

    /// Columns `{prefix}k0.., {prefix}id, {prefix}f` in one partition of
    /// 8,192-row batches.
    fn plan(&self, prefix: &str, types: &[KeyType]) -> Arc<dyn ExecutionPlan> {
        let mut fields: Vec<Field> = types
            .iter()
            .enumerate()
            .map(|(index, key)| {
                Field::new(format!("{prefix}k{index}"), key.data_type.clone(), true)
            })
            .collect();
        fields.push(Field::new(format!("{prefix}id"), DataType::Int64, false));
        fields.push(Field::new(format!("{prefix}f"), DataType::Int32, false));
        let schema = Arc::new(Schema::new(fields));
        let rows = self.rows();
        let mut batches = vec![];
        for start in (0..rows).step_by(8192) {
            let end = (start + 8192).min(rows);
            let mut columns: Vec<ArrayRef> = types
                .iter()
                .enumerate()
                .map(|(index, key)| {
                    key_array(
                        &key.data_type,
                        &self.keys[index][start..end],
                        self.valid[index].as_ref().map(|valid| &valid[start..end]),
                    )
                })
                .collect();
            columns.push(Arc::new(Int64Array::from_iter_values(
                start as i64..end as i64,
            )));
            columns.push(Arc::new(Int32Array::from(self.filter[start..end].to_vec())));
            batches.push(RecordBatch::try_new(Arc::clone(&schema), columns).unwrap());
        }
        TestMemoryExec::try_new_exec(&[batches], schema, None).unwrap()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// CollectLeft, one build future.
    Native,
    /// CollectLeft, the sharded parallel build.
    Parallel,
    /// Partitioned, two hash partitions.
    Partitioned,
}

#[derive(Debug)]
struct Config {
    join_type: JoinType,
    null_equality: NullEquality,
    mode: Mode,
    filter: bool,
    dynamic_filter: bool,
    /// A `NOT IN` join (one key; checked against the join with inline keys
    /// off, not against the oracle).
    null_aware: bool,
}

struct Case {
    describe: String,
    types: Vec<KeyType>,
    build: Side,
    probe: Side,
    config: Config,
}

const JOIN_TYPES: [JoinType; 10] = [
    JoinType::Inner,
    JoinType::Left,
    JoinType::Right,
    JoinType::Full,
    JoinType::LeftSemi,
    JoinType::RightSemi,
    JoinType::LeftAnti,
    JoinType::RightAnti,
    JoinType::LeftMark,
    JoinType::RightMark,
];

fn generate(seed: u64, index: usize) -> Case {
    let mut rng = Rng(seed
        .wrapping_mul(0xD1B5_4A32_D192_ED03)
        .wrapping_add(index as u64));
    let all = key_types();
    let mode = match rng.below(100) {
        0..35 => Mode::Native,
        35..80 => Mode::Parallel,
        _ => Mode::Partitioned,
    };
    // A parallel build packs keys of any number of columns into its slots;
    // the other builds keep dense keys only for several columns.
    let join_type = JOIN_TYPES[index % JOIN_TYPES.len()];
    let null_aware = matches!(
        join_type,
        JoinType::LeftAnti | JoinType::LeftMark | JoinType::RightAnti
    ) && mode != Mode::Partitioned
        && rng.chance(50);
    let columns = match (mode, rng.below(100)) {
        _ if null_aware => 1,
        (Mode::Parallel, 0..45) => 1,
        (Mode::Parallel, 45..85) => 2,
        (_, 0..15) => 1,
        (_, 15..75) => 2,
        _ => 3,
    };
    let types: Vec<KeyType> = (0..columns)
        .map(|_| {
            // Mostly encodable types; decimals keep the comparison.
            let pick = if rng.chance(6) {
                all.len() - 1 - rng.below(2) as usize
            } else {
                rng.below(all.len() as u64 - 2) as usize
            };
            all[pick].clone()
        })
        .collect();
    let small = rng.chance(8);
    let build_rows = if small {
        1 + rng.below(100_000) as usize
    } else if mode == Mode::Partitioned {
        // Each of the two partitions is likely above the threshold.
        600_000 + rng.below(40_000) as usize
    } else {
        262_145 + rng.below(40_000) as usize
    };
    // The bits of the encoded key: around the packed slot's limit (64 - 1 -
    // 10 - the head's bits, 34 at these sizes), around 64, or far above.
    let target = match (mode == Mode::Parallel, rng.below(100)) {
        (true, 0..55) | (false, 0..25) => 12 + rng.below(23) as u32,
        (true, 55..65) | (false, 25..35) => 32 + rng.below(6) as u32,
        (true, 65..80) | (false, 35..75) => 38 + rng.below(27) as u32,
        (true, 80..90) | (false, 75..88) => 62 + rng.below(5) as u32,
        _ => u32::MAX,
    };
    let mut remaining = target;
    let mut widths = vec![];
    for (position, key) in types.iter().enumerate() {
        let last = position + 1 == types.len();
        let width = if target == u32::MAX {
            key.bits()
        } else if last {
            remaining
        } else {
            rng.below(u64::from(remaining) + 1) as u32
        };
        let width = width.min(key.bits());
        remaining = remaining.saturating_sub(width);
        widths.push(width);
    }
    // Column `j` holds `base[j] + x` for `x` of `widths[j]` bits.
    let span = |width: u32| -> u128 {
        if width >= 128 {
            u128::MAX
        } else {
            (1_u128 << width) - 1
        }
    };
    let bases: Vec<i128> = types
        .iter()
        .zip(&widths)
        .map(|(key, &width)| {
            let room = (key.hi - key.lo) as u128;
            let span = span(width).min(room);
            let slack = room - span;
            match rng.below(100) {
                0..30 => key.lo,
                30..60 => key.lo + slack as i128,
                _ => key.lo + rng.upto(slack) as i128,
            }
        })
        .collect();
    let spans: Vec<u128> = types
        .iter()
        .zip(&widths)
        .map(|(key, &width)| span(width).min((key.hi - key.lo) as u128))
        .collect();
    let total: u32 = widths.iter().sum();
    // Distinct `id`s give distinct tuples while they fit the key's bits.
    let tuple = |id: u64| -> Vec<i128> {
        let mixed =
            u128::from(id).wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835);
        let mut shift = 0_u32;
        types
            .iter()
            .enumerate()
            .map(|(column, _)| {
                let width = widths[column];
                let bits = if shift + width <= 127 {
                    (mixed >> shift) & span(width)
                } else {
                    u128::from(id)
                        .wrapping_add(u128::from(column as u64) << 64)
                        .wrapping_mul(0x2545_F491_4F6C_DD1D_9E37_79B9_7F4A_7C15)
                        & span(width)
                };
                shift += width;
                // Wrap, not clamp: a clamp would give half of the rows of
                // a column narrower than its bits one key.
                let bits = match spans[column].checked_add(1) {
                    Some(size) => bits % size,
                    None => bits,
                };
                bases[column] + bits as i128
            })
            .collect()
    };
    let unique = total >= 20 && rng.chance(30);
    let distinct = (build_rows as u64 / [2, 8, 64][rng.below(3) as usize]).max(1);
    let heavy = if !unique && rng.chance(50) { 2_000 } else { 0 };
    let build_nulls = if rng.chance(50) { 0 } else { 2 };
    let probe_nulls = if rng.chance(30) { 0 } else { 5 };

    let mut build = Side {
        keys: vec![Vec::with_capacity(build_rows); columns],
        valid: vec![None; columns],
        filter: Vec::with_capacity(build_rows),
    };
    for row in 0..build_rows {
        let values = if unique {
            tuple(row as u64)
        } else if row == 0 {
            bases.clone()
        } else if row == 1 {
            bases
                .iter()
                .zip(&spans)
                .map(|(base, span)| base + *span as i128)
                .collect()
        } else if row < 2 + heavy {
            tuple(0)
        } else {
            tuple(rng.below(distinct))
        };
        for (column, value) in values.into_iter().enumerate() {
            build.keys[column].push(value);
        }
        build.filter.push(rng.below(8) as i32);
    }
    let null_column = |rng: &mut Rng, rows: usize, percent: u64| {
        (percent > 0 || rng.chance(20))
            .then(|| (0..rows).map(|_| !rng.chance(percent)).collect::<Vec<_>>())
    };
    for column in 0..columns {
        build.valid[column] = null_column(&mut rng, build_rows, build_nulls);
    }

    let chain =
        (build_rows as u64 / distinct.min(1 << total.min(40))).max(1) + heavy as u64 / 50;
    let probe_rows = (400_000 / chain).clamp(200, 20_000) as usize;
    let mut probe = Side {
        keys: vec![Vec::with_capacity(probe_rows); columns],
        valid: vec![None; columns],
        filter: Vec::with_capacity(probe_rows),
    };
    let wrap = |key: &KeyType, value: i128| {
        let size = key.hi - key.lo + 1;
        (value - key.lo).rem_euclid(size) + key.lo
    };
    for _ in 0..probe_rows {
        let from = rng.below(build_rows as u64) as usize;
        let mut values: Vec<i128> = (0..columns).map(|c| build.keys[c][from]).collect();
        match rng.below(100) {
            // A build key.
            0..40 => {}
            // A build key with one column moved: by one, by the column's
            // span (into the next field of a packed key), or by half of it.
            40..58 => {
                let column = rng.below(columns as u64) as usize;
                let span = spans[column] as i128;
                let step = match rng.below(6) {
                    0 => 1,
                    1 => -1,
                    2 => span.saturating_add(1),
                    3 => -span.saturating_add(1),
                    4 => span / 2 + 1,
                    _ => -(span / 2 + 1),
                };
                values[column] =
                    wrap(&types[column], values[column].saturating_add(step));
            }
            // Any tuple of the build's ranges.
            58..72 => {
                for column in 0..columns {
                    values[column] = bases[column] + rng.upto(spans[column]) as i128;
                }
            }
            // Any value of the types, the extremes among them.
            72..84 => {
                for column in 0..columns {
                    let key = &types[column];
                    values[column] = match rng.below(10) {
                        0 => key.lo,
                        1 => key.hi,
                        _ => key.lo + rng.upto((key.hi - key.lo) as u128) as i128,
                    };
                }
            }
            // Columns of different build rows.
            _ => {
                for (column, value) in values.iter_mut().enumerate().skip(1) {
                    let other = rng.below(build_rows as u64) as usize;
                    *value = build.keys[column][other];
                }
            }
        }
        for (column, value) in values.into_iter().enumerate() {
            probe.keys[column].push(value);
        }
        probe.filter.push(rng.below(8) as i32);
    }
    for column in 0..columns {
        probe.valid[column] = null_column(&mut rng, probe_rows, probe_nulls);
    }

    let config = Config {
        join_type,
        null_equality: if null_aware
            || rng.chance(if mode == Mode::Parallel { 85 } else { 60 })
        {
            NullEquality::NullEqualsNothing
        } else {
            NullEquality::NullEqualsNull
        },
        mode,
        filter: !null_aware && rng.chance(40),
        null_aware,
        dynamic_filter: match mode {
            Mode::Parallel => rng.chance(80),
            Mode::Native => rng.chance(50),
            Mode::Partitioned => false,
        },
    };
    let describe = format!(
        "seed {seed} case {index}: {config:?}, {build_rows} build rows ({}), \
         {probe_rows} probe rows, keys {:?} of bits {widths:?} at {bases:?}, \
         NULLs build {build_nulls}% probe {probe_nulls}%",
        if unique {
            "unique".to_string()
        } else {
            format!("{distinct} ids, {heavy} of one key")
        },
        types.iter().map(|key| &key.data_type).collect::<Vec<_>>(),
    );
    Case {
        describe,
        types,
        build,
        probe,
        config,
    }
}

/// An output row: the build row, the probe row and the mark.
type Row = (Option<i64>, Option<i64>, Option<bool>);

/// The join's rows from the generated values alone.
fn oracle(case: &Case) -> Vec<Row> {
    let Case {
        build,
        probe,
        config,
        ..
    } = case;
    let matchable = |key: &[Option<i128>]| {
        config.null_equality == NullEquality::NullEqualsNull
            || key.iter().all(Option::is_some)
    };
    let mut table: HashMap<Vec<Option<i128>>, Vec<u32>> = HashMap::new();
    for row in 0..build.rows() {
        let key = build.key(row);
        if matchable(&key) {
            table.entry(key).or_default().push(row as u32);
        }
    }
    let pairs_wanted = matches!(
        config.join_type,
        JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
    );
    let mut rows: Vec<Row> = vec![];
    let mut build_matched = vec![false; build.rows()];
    let mut probe_matched = vec![false; probe.rows()];
    for (probe_row, matched) in probe_matched.iter_mut().enumerate() {
        let key = probe.key(probe_row);
        if !matchable(&key) {
            continue;
        }
        for &build_row in table.get(&key).map_or(&[][..], Vec::as_slice) {
            let build_row = build_row as usize;
            if config.filter && build.filter[build_row] > probe.filter[probe_row] {
                continue;
            }
            build_matched[build_row] = true;
            *matched = true;
            if pairs_wanted {
                rows.push((Some(build_row as i64), Some(probe_row as i64), None));
            }
        }
    }
    let build_rows = |keep: &dyn Fn(bool) -> Option<Option<bool>>| -> Vec<Row> {
        build_matched
            .iter()
            .enumerate()
            .filter_map(|(row, &matched)| {
                keep(matched).map(|mark| (Some(row as i64), None, mark))
            })
            .collect()
    };
    let probe_rows = |keep: &dyn Fn(bool) -> Option<Option<bool>>| -> Vec<Row> {
        probe_matched
            .iter()
            .enumerate()
            .filter_map(|(row, &matched)| {
                keep(matched).map(|mark| (None, Some(row as i64), mark))
            })
            .collect()
    };
    let unmatched = |matched: bool| (!matched).then_some(None);
    let matched = |matched: bool| matched.then_some(None);
    let marked = |matched: bool| Some(Some(matched));
    match config.join_type {
        JoinType::Inner => {}
        JoinType::Left => rows.extend(build_rows(&unmatched)),
        JoinType::Right => rows.extend(probe_rows(&unmatched)),
        JoinType::Full => {
            rows.extend(build_rows(&unmatched));
            rows.extend(probe_rows(&unmatched));
        }
        JoinType::LeftSemi => rows = build_rows(&matched),
        JoinType::LeftAnti => rows = build_rows(&unmatched),
        JoinType::RightSemi => rows = probe_rows(&matched),
        JoinType::RightAnti => rows = probe_rows(&unmatched),
        JoinType::LeftMark => rows = build_rows(&marked),
        JoinType::RightMark => rows = probe_rows(&marked),
    }
    rows.sort_unstable();
    rows
}

fn context(
    case: &Case,
    pool: Option<Arc<dyn MemoryPool>>,
    spill: bool,
) -> Arc<TaskContext> {
    context_with(case, pool, spill, false)
}

/// `forced`: with join spilling on, the build spills from its first row
/// into two partitions, of which one is written.
fn context_with(
    case: &Case,
    pool: Option<Arc<dyn MemoryPool>>,
    spill: bool,
    forced: bool,
) -> Arc<TaskContext> {
    let mut config = SessionConfig::new()
        .with_batch_size(4096)
        .with_target_partitions(8);
    let parallel_rows = usize::from(case.config.mode == Mode::Parallel);
    config
        .options_mut()
        .extensions
        .insert(ParallelRows(parallel_rows));
    let options = config.options_mut();
    // Dense integer keys would otherwise take the perfect-hash ArrayMap.
    options.execution.perfect_hash_join_small_build_threshold = 0;
    options.execution.perfect_hash_join_min_key_density = f64::MAX;
    options.execution.enable_hash_join_spilling = spill;
    if forced {
        options.execution.hash_join_spill_test_force_rows = 1;
        options.execution.hash_join_spill_partitions = 2;
    }
    options.optimizer.enable_dynamic_filter_pushdown = true;
    options.optimizer.enable_join_dynamic_filter_pushdown = true;
    let mut runtime = RuntimeEnvBuilder::new();
    if let Some(pool) = pool {
        runtime = runtime.with_memory_pool(pool);
    }
    Arc::new(
        TaskContext::default()
            .with_session_config(config)
            .with_runtime(runtime.build_arc().unwrap()),
    )
}

/// Runs the join of `case` with inline keys on or off; returns its rows
/// and the probe chunks answered by packed slots, by dense keys and by the
/// key comparison.
async fn join(
    case: &Case,
    inline: bool,
    context: Arc<TaskContext>,
) -> Result<(Vec<Row>, [usize; 3])> {
    inline_keys::set_inline_keys_for_test(Some(inline));
    let before = JT9_REVIEW_PROBES
        .each_ref()
        .map(|count| count.load(Ordering::Relaxed));
    let config = &case.config;
    let mut left = case.build.plan("l", &case.types);
    let mut right = case.probe.plan("r", &case.types);
    let columns = case.types.len();
    let on: JoinOn = (0..columns)
        .map(|k| {
            (
                Arc::new(Column::new(&format!("lk{k}"), k)) as PhysicalExprRef,
                Arc::new(Column::new(&format!("rk{k}"), k)) as PhysicalExprRef,
            )
        })
        .collect();
    let dynamic_filter = config
        .dynamic_filter
        .then(|| HashJoinExec::create_dynamic_filter(&on));
    if let Some(filter) = &dynamic_filter {
        let consumer: Arc<dyn PhysicalExpr> = Arc::clone(filter) as _;
        right = Arc::new(FilterExecBuilder::new(consumer, right).build()?);
    }
    let partition_mode = if config.mode == Mode::Partitioned {
        let hash = |plan: Arc<dyn ExecutionPlan>, side: usize| {
            let keys = on
                .iter()
                .map(|pair| Arc::clone(if side == 0 { &pair.0 } else { &pair.1 }))
                .collect();
            RepartitionExec::try_new(plan, Partitioning::Hash(keys, 2))
                .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        };
        left = hash(left, 0)?;
        right = hash(right, 1)?;
        PartitionMode::Partitioned
    } else {
        PartitionMode::CollectLeft
    };
    let filter = config.filter.then(|| {
        // `lf <= rf`.
        JoinFilter::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("lf", 0)),
                Operator::LtEq,
                Arc::new(Column::new("rf", 1)),
            )),
            vec![
                ColumnIndex {
                    index: columns + 1,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: columns + 1,
                    side: JoinSide::Right,
                },
            ],
            Arc::new(Schema::new(vec![
                Field::new("lf", DataType::Int32, false),
                Field::new("rf", DataType::Int32, false),
            ])),
        )
    });
    let mut join = HashJoinExec::try_new(
        left,
        right,
        on,
        filter,
        &config.join_type,
        None,
        partition_mode,
        config.null_equality,
        config.null_aware,
    )?;
    if let Some(filter) = dynamic_filter {
        join.dynamic_filter = Some(HashJoinExecDynamicFilter {
            filter,
            build_accumulator: OnceLock::new(),
            execution_partitions: OnceLock::new(),
        });
    }
    let schema = join.schema();
    let column = |name: &str| schema.index_of(name).ok();
    let (lid, rid, mark) = (column("lid"), column("rid"), column("mark"));
    let mut rows: Vec<Row> = vec![];
    for partition in 0..join.properties().partitioning.partition_count() {
        let batches =
            common::collect(join.execute(partition, Arc::clone(&context))?).await?;
        for batch in batches {
            let ids = |index: Option<usize>| {
                index.map(|index| {
                    batch
                        .column(index)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .clone()
                })
            };
            let (lid, rid) = (ids(lid), ids(rid));
            let mark = mark.map(|index| {
                batch
                    .column(index)
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap()
                    .clone()
            });
            for row in 0..batch.num_rows() {
                let id = |ids: &Option<Int64Array>| {
                    ids.as_ref()
                        .and_then(|ids| ids.is_valid(row).then(|| ids.value(row)))
                };
                rows.push((
                    id(&lid),
                    id(&rid),
                    mark.as_ref().map(|mark| mark.value(row)),
                ));
            }
        }
    }
    rows.sort_unstable();
    let after = JT9_REVIEW_PROBES
        .each_ref()
        .map(|count| count.load(Ordering::Relaxed));
    Ok((rows, std::array::from_fn(|path| after[path] - before[path])))
}

/// The first rows that only one of the two sorted lists holds.
fn difference(case: &Case, actual: &[Row], expected: &[Row]) -> String {
    let mut out = String::new();
    let (mut a, mut e, mut shown) = (0, 0, 0);
    let show = |out: &mut String, tag: &str, row: &Row| {
        let build = row.0.map(|row| case.build.key(row as usize));
        let probe = row.1.map(|row| case.probe.key(row as usize));
        out.push_str(&format!(
            "\n  {tag} {row:?}: build key {build:?}, probe key {probe:?}"
        ));
    };
    while (a < actual.len() || e < expected.len()) && shown < 8 {
        match (actual.get(a), expected.get(e)) {
            (Some(x), Some(y)) if x == y => {
                a += 1;
                e += 1;
            }
            (Some(x), Some(y)) if x < y => {
                show(&mut out, "extra", x);
                a += 1;
                shown += 1;
            }
            (Some(x), None) => {
                show(&mut out, "extra", x);
                a += 1;
                shown += 1;
            }
            (_, Some(y)) => {
                show(&mut out, "missing", y);
                e += 1;
                shown += 1;
            }
            (None, None) => break,
        }
    }
    out
}

/// Appends `line` to this seed's progress file, when one is wanted.
fn progress(seed: u64, line: &str) {
    use std::io::Write;
    if let Ok(dir) = std::env::var("JT9_REVIEW_OUT") {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{dir}/progress-seed-{seed}.txt"))
            .unwrap();
        writeln!(file, "{line}").unwrap();
    }
}

fn cases_per_seed() -> usize {
    std::env::var("JT9_REVIEW_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10)
}

/// Runs the cases of `seed`: inline keys on and off against the oracle.
async fn differential(seed: u64) -> Result<()> {
    let mut paths: HashMap<String, usize> = HashMap::new();
    let mut failures = vec![];
    let cases = cases_per_seed();
    for index in 0..cases {
        let case = generate(seed, index);
        progress(seed, &format!("{}", case.describe));
        // A `NOT IN` join is checked against itself with inline keys off.
        let mut expected = (!case.config.null_aware).then(|| oracle(&case));
        if let Some(expected) = &expected {
            progress(seed, &format!("  oracle: {} rows", expected.len()));
        }
        let mut taken = [0; 3];
        for inline in [false, true] {
            let (actual, counts) =
                join(&case, inline, context(&case, None, false)).await?;
            progress(
                seed,
                &format!("  inline {inline}: {} rows, paths {counts:?}", actual.len()),
            );
            if inline {
                taken = counts;
            } else {
                assert_eq!(
                    (counts[0], counts[1]),
                    (0, 0),
                    "the switch is off: {}",
                    case.describe
                );
            }
            match &expected {
                Some(expected) if actual != *expected => failures.push(format!(
                    "inline {inline}: {} actual rows, {} expected: {}{}",
                    actual.len(),
                    expected.len(),
                    case.describe,
                    difference(&case, &actual, expected)
                )),
                Some(_) => {}
                None => expected = Some(actual),
            }
        }
        let path = match taken {
            [0, 0, 0] => "no probe chunk",
            [_, 0, 0] => "packed slots",
            [0, _, 0] => "dense keys",
            [0, 0, _] => "key comparison",
            _ => "several paths",
        };
        let large = JoinHashMapU32::partitions_build(case.build.rows());
        *paths
            .entry(format!(
                "{:?} {path}{}",
                case.config.mode,
                if large { "" } else { " (small build)" }
            ))
            .or_default() += 1;
        *paths
            .entry(format!("{} {path}", case.config.join_type))
            .or_default() += 1;
        *paths
            .entry(format!("{:?} {path}", case.config.null_equality))
            .or_default() += 1;
        if case.config.filter {
            *paths.entry(format!("filter {path}")).or_default() += 1;
        }
        if case.config.null_aware {
            *paths.entry(format!("null-aware {path}")).or_default() += 1;
        }
    }
    inline_keys::set_inline_keys_for_test(None);
    let mut summary: Vec<String> = paths
        .iter()
        .map(|(path, count)| format!("{path}: {count}"))
        .collect();
    summary.sort();
    let summary = format!(
        "seed {seed}: {cases} cases, {} failures\n{}\n",
        failures.len(),
        summary.join("\n")
    );
    println!("{summary}");
    if let Ok(dir) = std::env::var("JT9_REVIEW_OUT") {
        std::fs::write(format!("{dir}/differential-seed-{seed}.txt"), &summary).unwrap();
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    Ok(())
}

macro_rules! differential_seeds {
    ($($name:ident: $seed:expr),*) => {
        $(
            #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
            async fn $name() -> Result<()> {
                let base = std::env::var("JT9_REVIEW_SEED_BASE")
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(0);
                differential(base + $seed).await
            }
        )*
    };
}
differential_seeds!(
    jt9_review_differential_seed_1: 1,
    jt9_review_differential_seed_2: 2,
    jt9_review_differential_seed_3: 3,
    jt9_review_differential_seed_4: 4,
    jt9_review_differential_seed_5: 5,
    jt9_review_differential_seed_6: 6,
    jt9_review_differential_seed_7: 7,
    jt9_review_differential_seed_8: 8
);

/// A case with fixed keys: `columns[c][row]` on the build side, probed with
/// its own first rows, an inner join.
fn fixed_case(
    types: Vec<KeyType>,
    build_keys: Vec<Vec<i128>>,
    mode: Mode,
    dynamic_filter: bool,
) -> Case {
    let rows = build_keys[0].len();
    let probe_rows = rows.min(10_000);
    let columns = types.len();
    Case {
        describe: format!("fixed {mode:?}"),
        probe: Side {
            keys: build_keys
                .iter()
                .map(|keys| keys[..probe_rows].to_vec())
                .collect(),
            valid: vec![None; columns],
            filter: vec![0; probe_rows],
        },
        build: Side {
            keys: build_keys,
            valid: vec![None; columns],
            filter: vec![0; rows],
        },
        types,
        config: Config {
            join_type: JoinType::Inner,
            null_equality: NullEquality::NullEqualsNothing,
            mode,
            filter: false,
            dynamic_filter,
            null_aware: false,
        },
    }
}

/// Finding 1: a parallel build whose keys are too wide for a packed slot
/// (here 46 bits beside a 19-bit head) must keep the hash-only table.
/// `SlotLayout::packed` computes `64 - tag_shift` before it checks that the
/// fields fit, which overflows in a build with overflow checks (every test
/// and debug build).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jt9_review_wide_keys_in_a_parallel_build_keep_the_hash_only_table() -> Result<()>
{
    // The same at the table: 19 head bits, the mark and 46 key bits.
    assert!(SlotLayout::packed(300_000, 46).is_none());
    assert!(SlotLayout::packed(300_000, 64).is_none());
    assert!(SlotLayout::packed(300_000, 34).is_some());
    let rows = 300_000_i128;
    let int64 = key_types()
        .into_iter()
        .find(|key| key.data_type == DataType::Int64)
        .unwrap();
    let keys: Vec<i128> = (0..rows).map(|row| row << 27).collect();
    let case = fixed_case(vec![int64], vec![keys], Mode::Parallel, false);
    let expected = oracle(&case);
    assert_eq!(expected.len(), 10_000);
    for inline in [false, true] {
        let (actual, _) = join(&case, inline, context(&case, None, false)).await?;
        assert!(actual == expected, "inline {inline}");
    }
    inline_keys::set_inline_keys_for_test(None);
    Ok(())
}

/// A pool that records the most it has held.
#[derive(Debug)]
struct PeakPool {
    pool: GreedyMemoryPool,
    peak: AtomicUsize,
}

impl PeakPool {
    fn new(limit: usize) -> Self {
        Self {
            pool: GreedyMemoryPool::new(limit),
            peak: AtomicUsize::new(0),
        }
    }
}

impl fmt::Display for PeakPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "peak({})", self.pool)
    }
}

impl MemoryPool for PeakPool {
    fn name(&self) -> &str {
        "peak"
    }
    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.pool.grow(reservation, additional);
        self.peak.fetch_max(self.pool.reserved(), Ordering::Relaxed);
    }
    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.pool.shrink(reservation, shrink);
    }
    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        self.pool.try_grow(reservation, additional)?;
        self.peak.fetch_max(self.pool.reserved(), Ordering::Relaxed);
        Ok(())
    }
    fn reserved(&self) -> usize {
        self.pool.reserved()
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.pool.memory_limit()
    }
}

/// Finding 2: a build that fits its memory pool with inline keys off must
/// not fail with them on. The dense keys are optional, but they are charged
/// before the table and the copy of the build rows, which are not.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jt9_review_dense_keys_do_not_fail_a_build_that_fits_without_them() -> Result<()>
{
    let rows = 300_000_i128;
    let int32 = key_types()
        .into_iter()
        .find(|key| key.data_type == DataType::Int32)
        .unwrap();
    let case = fixed_case(
        vec![int32.clone(), int32],
        vec![
            (0..rows).collect(),
            (0..rows).map(|row| row % 1000).collect(),
        ],
        Mode::Native,
        false,
    );
    let expected = oracle(&case);
    // The most that the join holds without inline keys.
    let unlimited = Arc::new(PeakPool::new(usize::MAX));
    let pool: Arc<dyn MemoryPool> = Arc::clone(&unlimited) as _;
    let (actual, _) = join(&case, false, context(&case, Some(pool), false)).await?;
    assert!(actual == expected);
    let limit = unlimited.peak.load(Ordering::Relaxed);
    let mut outcomes = vec![];
    // The last arm is informational: with join spilling on, the refused
    // build spills instead of failing.
    for (inline, spill) in [(false, false), (true, false), (true, true)] {
        let limited = Arc::new(PeakPool::new(limit));
        let pool: Arc<dyn MemoryPool> = Arc::clone(&limited) as _;
        let outcome = join(&case, inline, context(&case, Some(pool), spill)).await;
        outcomes.push(format!(
            "inline {inline}, join spilling {spill}, in a pool of {limit} bytes (peak {}): {}",
            limited.peak.load(Ordering::Relaxed),
            match &outcome {
                Ok((actual, taken)) => format!(
                    "ok, equal to the oracle: {}, paths {taken:?}",
                    *actual == expected
                ),
                Err(error) => format!("error: {error}")
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .to_string(),
            }
        ));
    }
    inline_keys::set_inline_keys_for_test(None);
    let report = outcomes.join("\n");
    println!("{report}");
    if let Ok(dir) = std::env::var("JT9_REVIEW_OUT") {
        std::fs::write(format!("{dir}/dense-memory.txt"), &report).unwrap();
    }
    assert!(
        outcomes
            .iter()
            .all(|outcome| outcome.contains("ok, equal to the oracle: true")),
        "{report}"
    );
    Ok(())
}

/// Finding 3 (informational): the partitions of a spilled build that are
/// read back from their files are joined by nested `CollectLeft` joins,
/// which go through `collect_left_input` and so take the JT 9 encodings;
/// only the in-memory table of the spilling build itself keeps hashes only.
/// The rows are right either way; the test records which probe paths ran.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jt9_review_partitions_read_back_from_spill_files_take_dense_keys() -> Result<()>
{
    let rows = 1_200_000_i128;
    let int32 = key_types()
        .into_iter()
        .find(|key| key.data_type == DataType::Int32)
        .unwrap();
    let case = fixed_case(
        vec![int32.clone(), int32],
        vec![
            (0..rows).collect(),
            (0..rows).map(|row| row % 1000).collect(),
        ],
        Mode::Native,
        false,
    );
    let expected = oracle(&case);
    let mut report = vec![];
    for inline in [false, true] {
        let (actual, taken) =
            join(&case, inline, context_with(&case, None, true, true)).await?;
        assert!(actual == expected, "inline {inline}");
        report.push(format!(
            "inline {inline}, forced spill: probe chunks by packed slots, dense keys \
             and key comparison: {taken:?}"
        ));
    }
    inline_keys::set_inline_keys_for_test(None);
    let report = report.join("\n");
    println!("{report}");
    if let Ok(dir) = std::env::var("JT9_REVIEW_OUT") {
        std::fs::write(format!("{dir}/spill-read-back.txt"), &report).unwrap();
    }
    Ok(())
}

/// Finding 4 (latent): `InlineKeys::encode_row` must give the key that
/// `InlineKeys::encode` gives for the same row. For a column whose range is
/// all of `i64` beside other (constant) columns, `encode` subtracts the
/// minimum and `encode_row` returns the raw value, and skips the columns
/// after it. No table reads `encode_row` for such keys today: only packed
/// slots do, and 64 key bits never fit them.
#[test]
fn jt9_review_encode_row_agrees_with_encode() {
    let types = [DataType::Int64, DataType::Int64];
    let keys =
        InlineKeys::from_ranges(&types, &[Some((i64::MIN, i64::MAX)), Some((4, 4))])
            .unwrap();
    let a: ArrayRef = Arc::new(Int64Array::from(vec![-7, 0, 9, i64::MIN]));
    let b: ArrayRef = Arc::new(Int64Array::from(vec![Some(4), Some(4), None, Some(4)]));
    let arrays = [a, b];
    let encoded = keys.encode(&arrays).unwrap();
    for row in 0..4 {
        let valid = encoded
            .valid
            .as_ref()
            .is_none_or(|valid| valid.is_valid(row));
        assert_eq!(
            keys.encode_row(&arrays, row),
            valid.then_some(encoded.keys[row]),
            "row {row}"
        );
    }
}

/// Finding 3 with finding 2's fix: the nested joins over partitions read
/// back from spill files take dense keys like any build, and with them the
/// same retry. Run in a pool that holds exactly the peak of the same spilled
/// join with inline keys off, the join with them on must succeed and return
/// the oracle's rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jt9_review_read_back_partitions_fit_at_the_switch_off_peak() -> Result<()> {
    let rows = 1_200_000_i128;
    let int32 = key_types()
        .into_iter()
        .find(|key| key.data_type == DataType::Int32)
        .unwrap();
    let case = fixed_case(
        vec![int32.clone(), int32],
        vec![
            (0..rows).collect(),
            (0..rows).map(|row| row % 1000).collect(),
        ],
        Mode::Native,
        false,
    );
    let expected = oracle(&case);
    let unlimited = Arc::new(PeakPool::new(usize::MAX));
    let pool: Arc<dyn MemoryPool> = Arc::clone(&unlimited) as _;
    let (actual, _) =
        join(&case, false, context_with(&case, Some(pool), true, true)).await?;
    assert!(actual == expected);
    let limit = unlimited.peak.load(Ordering::Relaxed);
    let limited = Arc::new(PeakPool::new(limit));
    let pool: Arc<dyn MemoryPool> = Arc::clone(&limited) as _;
    let (actual, taken) =
        join(&case, true, context_with(&case, Some(pool), true, true)).await?;
    inline_keys::set_inline_keys_for_test(None);
    println!(
        "forced spill in a pool of {limit} bytes (the switch-off peak), inline on: \
         peak {}, probe chunks by packed slots, dense keys and key comparison {taken:?}",
        limited.peak.load(Ordering::Relaxed)
    );
    assert!(actual == expected);
    Ok(())
}

// ---------------------------------------------------------------------------
// Re-review of JT 9 (leto-mjj.115). A pool that
// refuses chosen `try_grow` calls of the build's consumer, so that every
// charge of `build_in_memory` is refused in turn, once or twice.
// ---------------------------------------------------------------------------

const NO_FAULT: usize = usize::MAX;

#[derive(Debug)]
struct FaultPool {
    pool: GreedyMemoryPool,
    /// `try_grow` calls of the `HashJoinInput` consumers so far.
    calls: AtomicUsize,
    /// The calls (by index) that are refused.
    refuse: [usize; 2],
    refused: AtomicUsize,
    peak: AtomicUsize,
    /// The most held after the first refusal.
    peak_after_fault: AtomicUsize,
    /// Shrinks beyond what is held.
    wrapped: AtomicUsize,
    /// Bytes of each refused call.
    refused_bytes: std::sync::Mutex<Vec<usize>>,
}

impl FaultPool {
    fn new(refuse: [usize; 2]) -> Self {
        Self {
            pool: GreedyMemoryPool::new(usize::MAX),
            calls: AtomicUsize::new(0),
            refuse,
            refused: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            peak_after_fault: AtomicUsize::new(0),
            wrapped: AtomicUsize::new(0),
            refused_bytes: std::sync::Mutex::new(vec![]),
        }
    }

    fn note(&self) {
        let reserved = self.pool.reserved();
        if reserved > usize::MAX / 2 {
            self.wrapped.fetch_add(1, Ordering::SeqCst);
        }
        self.peak.fetch_max(reserved, Ordering::SeqCst);
        if self.refused.load(Ordering::SeqCst) > 0 {
            self.peak_after_fault.fetch_max(reserved, Ordering::SeqCst);
        }
    }
}

impl fmt::Display for FaultPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fault({})", self.pool)
    }
}

impl MemoryPool for FaultPool {
    fn name(&self) -> &str {
        "fault"
    }
    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.pool.grow(reservation, additional);
        self.note();
    }
    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        if shrink > self.pool.reserved() {
            self.wrapped.fetch_add(1, Ordering::SeqCst);
        }
        self.pool.shrink(reservation, shrink);
        self.note();
    }
    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        if reservation.consumer().name().starts_with("HashJoinInput") {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.refuse.contains(&call) {
                self.refused.fetch_add(1, Ordering::SeqCst);
                self.refused_bytes.lock().unwrap().push(additional);
                return Err(datafusion_common::DataFusionError::ResourcesExhausted(
                    format!(
                        "re-review fault: HashJoinInput charge {call} of {additional} bytes"
                    ),
                ));
            }
        }
        self.pool.try_grow(reservation, additional)?;
        self.note();
        Ok(())
    }
    fn reserved(&self) -> usize {
        self.pool.reserved()
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.pool.memory_limit()
    }
}

fn rereview_context(
    pool: Arc<dyn MemoryPool>,
    spill: bool,
    parallel_rows: usize,
) -> Arc<TaskContext> {
    let mut config = SessionConfig::new()
        .with_batch_size(4096)
        .with_target_partitions(8);
    config
        .options_mut()
        .extensions
        .insert(ParallelRows(parallel_rows));
    let options = config.options_mut();
    options.execution.perfect_hash_join_small_build_threshold = 0;
    options.execution.perfect_hash_join_min_key_density = f64::MAX;
    options.execution.enable_hash_join_spilling = spill;
    options.optimizer.enable_dynamic_filter_pushdown = true;
    options.optimizer.enable_join_dynamic_filter_pushdown = true;
    Arc::new(
        TaskContext::default()
            .with_session_config(config)
            .with_runtime(
                RuntimeEnvBuilder::new()
                    .with_memory_pool(pool)
                    .build_arc()
                    .unwrap(),
            ),
    )
}

struct Run {
    /// The probe paths of a join that answered, or the first line of its error.
    outcome: std::result::Result<[usize; 3], String>,
    calls: usize,
    refused: Vec<usize>,
    peak: usize,
    peak_after_fault: usize,
}

/// Runs the join; panics on wrong rows, on an error that is not a refusal,
/// on a shrink beyond what was held, and on bytes still held at the end.
async fn rereview_run(
    case: &Case,
    expected: &[Row],
    inline: bool,
    refuse: [usize; 2],
    spill: bool,
    parallel_rows: usize,
) -> Run {
    let pool = Arc::new(FaultPool::new(refuse));
    let context = rereview_context(
        Arc::clone(&pool) as Arc<dyn MemoryPool>,
        spill,
        parallel_rows,
    );
    let label = format!(
        "{}, inline {inline}, refuse {refuse:?}, spill {spill}, parallel rows {parallel_rows}",
        case.describe
    );
    let outcome = match join(case, inline, context).await {
        Ok((rows, taken)) => {
            assert!(
                rows == expected,
                "{label}: wrong rows: {}",
                difference(case, &rows, expected)
            );
            Ok(taken)
        }
        Err(error) => {
            assert!(
                matches!(
                    error.find_root(),
                    datafusion_common::DataFusionError::ResourcesExhausted(_)
                ),
                "{label}: not a refusal: {error}"
            );
            Err(format!("{error}")
                .lines()
                .next()
                .unwrap_or_default()
                .to_string())
        }
    };
    for _ in 0..200 {
        if pool.reserved() == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(pool.reserved(), 0, "{label}: bytes held after the join");
    assert_eq!(
        pool.wrapped.load(Ordering::SeqCst),
        0,
        "{label}: over-release"
    );
    Run {
        outcome,
        calls: pool.calls.load(Ordering::SeqCst),
        refused: pool.refused_bytes.lock().unwrap().clone(),
        peak: pool.peak.load(Ordering::SeqCst),
        peak_after_fault: pool.peak_after_fault.load(Ordering::SeqCst),
    }
}

fn rereview_case(rows: i128, mode: Mode, join_type: JoinType) -> Case {
    let int32 = key_types()
        .into_iter()
        .find(|key| key.data_type == DataType::Int32)
        .unwrap();
    let mut case = fixed_case(
        vec![int32.clone(), int32],
        vec![
            (0..rows).collect(),
            (0..rows).map(|row| row % 1000).collect(),
        ],
        mode,
        false,
    );
    case.describe = format!("{mode:?} {join_type:?}");
    case.config.join_type = join_type;
    // `lf <= rf` refuses a third of the pairs; a third of the probe rows
    // have no build row.
    case.config.filter = true;
    for (row, filter) in case.build.filter.iter_mut().enumerate() {
        *filter = (row % 3) as i32;
    }
    for (row, filter) in case.probe.filter.iter_mut().enumerate() {
        *filter = 1;
        if row % 3 == 0 {
            case.probe.keys[1][row] += 1;
        }
    }
    case
}

fn rereview_report(name: &str, lines: &[String]) {
    let report = lines.join("\n");
    println!("{report}");
    if let Ok(dir) = std::env::var("JT9_REVIEW_OUT") {
        std::fs::write(format!("{dir}/{name}.txt"), &report).unwrap();
    }
}

fn rereview_line(tag: &str, run: &Run) -> String {
    format!(
        "{tag}: {}; charges {}, refused bytes {:?}, peak {}, peak after the refusal {}",
        match &run.outcome {
            Ok(taken) => format!("ok, rows equal the oracle, paths {taken:?}"),
            Err(error) => format!("refused: {error}"),
        },
        run.calls,
        run.refused,
        run.peak,
        run.peak_after_fault
    )
}

/// Every charge of the build refused once, for a native and a compacted
/// (parallel option set, one future) CollectLeft build and a Partitioned
/// build, with inline keys on and off.
async fn rereview_single(
    name: &str,
    rows: i128,
    mode: Mode,
    join_type: JoinType,
    parallel_rows: usize,
    last: usize,
) -> Result<()> {
    let case = rereview_case(rows, mode, join_type);
    let expected = oracle(&case);
    let mut lines = vec![format!("{name}: {rows} build rows, {}", case.describe)];
    let mut clean = vec![];
    for inline in [false, true] {
        let run = rereview_run(
            &case,
            &expected,
            inline,
            [NO_FAULT; 2],
            false,
            parallel_rows,
        )
        .await;
        lines.push(rereview_line(&format!("inline {inline}, no refusal"), &run));
        assert!(run.outcome.is_ok());
        clean.push(run);
    }
    assert!(
        clean[1].outcome.as_ref().unwrap()[1] > 0,
        "the clean join with inline keys on did not use dense keys"
    );
    for inline in [false, true] {
        let calls = clean[usize::from(inline)].calls;
        let (mut ok, mut refused) = (vec![], vec![]);
        for call in calls.saturating_sub(last)..calls {
            let run = rereview_run(
                &case,
                &expected,
                inline,
                [call, NO_FAULT],
                false,
                parallel_rows,
            )
            .await;
            lines.push(rereview_line(
                &format!("inline {inline}, refuse {call}"),
                &run,
            ));
            if run.outcome.is_ok() {
                ok.push(call)
            } else {
                refused.push(call)
            }
        }
        lines.push(format!(
            "inline {inline}: answered with refusal at {ok:?}; failed at {refused:?}"
        ));
    }
    inline_keys::set_inline_keys_for_test(None);
    rereview_report(name, &lines);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jt9_rereview_single_refusal_native_full() -> Result<()> {
    rereview_single(
        "rereview-single-native-full",
        270_000,
        Mode::Native,
        JoinType::Full,
        0,
        24,
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jt9_rereview_single_refusal_native_left_mark() -> Result<()> {
    rereview_single(
        "rereview-single-native-left-mark",
        270_000,
        Mode::Native,
        JoinType::LeftMark,
        0,
        24,
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jt9_rereview_single_refusal_compacted_left_anti() -> Result<()> {
    rereview_single(
        "rereview-single-compacted-left-anti",
        270_000,
        Mode::Native,
        JoinType::LeftAnti,
        100_000_000,
        24,
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jt9_rereview_single_refusal_partitioned_full() -> Result<()> {
    rereview_single(
        "rereview-single-partitioned-full",
        600_000,
        Mode::Partitioned,
        JoinType::Full,
        0,
        40,
    )
    .await
}

/// Two refusals: the second lands in the retry (or after it). Without join
/// spilling the join must fail cleanly or answer; with it, the same.
async fn rereview_double(name: &str, spill: bool) -> Result<()> {
    let case = rereview_case(270_000, Mode::Native, JoinType::Full);
    let expected = oracle(&case);
    let mut lines = vec![format!("{name}: {}, join spilling {spill}", case.describe)];
    let clean = rereview_run(&case, &expected, true, [NO_FAULT; 2], spill, 0).await;
    lines.push(rereview_line("inline true, no refusal", &clean));
    assert!(clean.outcome.is_ok());
    let (mut ok, mut refused) = (0, 0);
    for first in clean.calls.saturating_sub(14)..clean.calls {
        for gap in 1..=10 {
            let run =
                rereview_run(&case, &expected, true, [first, first + gap], spill, 0)
                    .await;
            lines.push(rereview_line(
                &format!("inline true, refuse {first} and {}", first + gap),
                &run,
            ));
            if run.outcome.is_ok() {
                ok += 1
            } else {
                refused += 1
            }
        }
    }
    lines.push(format!("answered {ok}, failed with a refusal {refused}"));
    inline_keys::set_inline_keys_for_test(None);
    rereview_report(name, &lines);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jt9_rereview_double_refusal_no_spill() -> Result<()> {
    rereview_double("rereview-double-no-spill", false).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jt9_rereview_double_refusal_spill() -> Result<()> {
    rereview_double("rereview-double-spill", true).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jt9_rereview_memory_priority() {
    let case = rereview_case(270_000, Mode::Native, JoinType::Full);
    let expected = oracle(&case);
    // Build charges after the input: dense keys (33), map (34),
    // partitioning arrays (35), copy (36), keys (37), visited bitmap (38).
    let no_dense = rereview_run(&case, &expected, true, [33, 34], false, 0).await;
    assert_eq!(no_dense.refused[0], 270_000 * size_of::<u64>());
    assert!(
        no_dense.outcome.is_err(),
        "repeated a build without dense keys"
    );
    assert_eq!(no_dense.calls, 35);

    let partitioned =
        rereview_run(&case, &expected, true, [35, NO_FAULT], false, 0).await;
    assert_eq!(partitioned.refused, [270_000 * 20]);
    assert_eq!(partitioned.outcome.unwrap()[1], 0, "kept dense keys");

    // Refuse the visited bitmap after concatenation, then the retry's
    // partitioning arrays. The retry must charge its one-batch hash scratch.
    let retry = rereview_run(&case, &expected, true, [38, 40], false, 0).await;
    assert_eq!(
        retry.refused,
        [VisitedBitmap::size_bytes(270_000), 270_000 * 20]
    );
    assert!(retry.outcome.is_ok());
    let rows = case.build.rows();
    // Two Int32 keys, an Int64 row ID, and the Int32 filter column.
    let input_bytes = rows * (size_of::<i64>() + 3 * size_of::<i32>());
    let map_bytes = JoinHashMapU32::with_capacity(rows).allocated_bytes();
    assert!(retry.peak_after_fault >= input_bytes + map_bytes + rows * size_of::<u64>());

    inline_keys::set_inline_keys_for_test(None);
}
