// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow patch (JT 9): join keys encoded into one `u64`, stored beside the
//! chain heads of a large join table.
//!
//! A probe of a large table misses the cache twice per match: once for the
//! chain head and once for the build key that the key comparison reads at a
//! random build row. An inline table keeps the encoded key of each chain's
//! head row in the slot itself (`InlineU32` in `join_hash_map.rs`), so the
//! probe compares keys in the cache line that it already loaded, and reads
//! the build key columns only for a chain whose rows have different keys
//! (a 32-bit hash tag collision).
//!
//! The encoding is injective over the build's key range:
//!
//! - one integer column (8 to 64 bits, signed or not, dates, times,
//!   timestamps, durations): its value, reinterpreted as `u64`;
//! - several such columns: each value minus the build's minimum of that
//!   column, packed into its own bit field, when the fields of all columns
//!   fit 64 bits. A probe value outside the build's range cannot match and
//!   is marked invalid.
//!
//! Other keys (floats, strings, decimals, nested types, wider ranges) keep
//! the hash-only table and the key comparison.

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::*;
use datafusion_common::NullEquality;

/// The environment variable that turns inline keys off: `off`, `0`,
/// `false`, `no` or `disabled`, in any case. Read once per process.
pub const INLINE_KEYS_ENV: &str = "LONGBOW_JOIN_INLINE_KEYS";

/// Whether join tables store inline keys: on unless [`INLINE_KEYS_ENV`]
/// turns them off. Read once per process.
pub fn inline_keys_enabled() -> bool {
    #[cfg(test)]
    match TEST_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        inline_keys_requested(std::env::var(INLINE_KEYS_ENV).ok().as_deref())
    })
}

/// Test hook: 0 reads the environment, 1 turns inline keys off, 2 on. A
/// process-wide switch: nextest runs each test in its own process.
#[cfg(test)]
static TEST_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Test hook: `Some(on)` overrides [`inline_keys_enabled`] for this process.
#[cfg(test)]
pub(crate) fn set_inline_keys_for_test(on: Option<bool>) {
    let value = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    TEST_OVERRIDE.store(value, std::sync::atomic::Ordering::Relaxed);
}

fn inline_keys_requested(value: Option<&str>) -> bool {
    let Some(value) = value.map(str::trim) else {
        return true;
    };
    let is = |words: &[&str]| words.iter().any(|w| value.eq_ignore_ascii_case(w));
    if is(&["off", "0", "false", "no", "disabled"]) {
        false
    } else if value.is_empty() || is(&["on", "1", "true", "yes", "enabled"]) {
        true
    } else {
        log::warn!(
            "{INLINE_KEYS_ENV}={value:?} is not recognized; inline join keys stay on \
             (use off, 0, false, no or disabled to turn them off)"
        );
        true
    }
}

/// One key column of an [`InlineKeys`] encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
struct KeyColumn {
    data_type: DataType,
    /// Subtracted from every value (0 for a single column).
    min: i64,
    /// The largest value minus `min`; a larger difference cannot match.
    span: u64,
    /// The bit position of this column's field.
    shift: u32,
}

/// The encoding of a join's keys into one `u64` (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlineKeys {
    columns: Vec<KeyColumn>,
}

/// Encoded keys of a batch: `keys[i]` is meaningful only where `valid` is
/// set (or everywhere when `valid` is `None`). An invalid row has a NULL key
/// or a value outside the build's range, and matches no build row.
#[derive(Debug, Clone)]
pub struct EncodedKeys {
    pub keys: Vec<u64>,
    pub valid: Option<NullBuffer>,
}

/// Calls `$body` with `$values`, the values of `$array` (a `&dyn Array` of
/// `$data_type`) as a slice of a native integer type, or evaluates `$other`
/// when the type has no inline encoding.
macro_rules! with_integer_values {
    ($data_type:expr, $array:expr, |$values:ident| $body:expr, $other:expr) => {{
        macro_rules! go {
            ($t:ty) => {{
                let $values = $array.as_primitive::<$t>().values().as_ref();
                $body
            }};
        }
        match $data_type {
            DataType::Int8 => go!(Int8Type),
            DataType::Int16 => go!(Int16Type),
            DataType::Int32 => go!(Int32Type),
            DataType::Int64 => go!(Int64Type),
            DataType::UInt8 => go!(UInt8Type),
            DataType::UInt16 => go!(UInt16Type),
            DataType::UInt32 => go!(UInt32Type),
            DataType::UInt64 => go!(UInt64Type),
            DataType::Date32 => go!(Date32Type),
            DataType::Date64 => go!(Date64Type),
            DataType::Time32(TimeUnit::Second) => go!(Time32SecondType),
            DataType::Time32(TimeUnit::Millisecond) => go!(Time32MillisecondType),
            DataType::Time64(TimeUnit::Microsecond) => go!(Time64MicrosecondType),
            DataType::Time64(TimeUnit::Nanosecond) => go!(Time64NanosecondType),
            DataType::Timestamp(TimeUnit::Second, _) => go!(TimestampSecondType),
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                go!(TimestampMillisecondType)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                go!(TimestampMicrosecondType)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                go!(TimestampNanosecondType)
            }
            DataType::Duration(TimeUnit::Second) => go!(DurationSecondType),
            DataType::Duration(TimeUnit::Millisecond) => go!(DurationMillisecondType),
            DataType::Duration(TimeUnit::Microsecond) => go!(DurationMicrosecondType),
            DataType::Duration(TimeUnit::Nanosecond) => go!(DurationNanosecondType),
            _ => $other,
        }
    }};
}

/// A native integer as `i64`: sign-extended, or zero-extended for unsigned
/// types; `u64` is reinterpreted. Injective on each type.
trait AsI64: Copy {
    fn as_i64(self) -> i64;
}

macro_rules! as_i64 {
    ($($t:ty),*) => {
        $(impl AsI64 for $t {
            #[inline(always)]
            fn as_i64(self) -> i64 {
                self as i64
            }
        })*
    };
}
as_i64!(i8, i16, i32, i64, u8, u16, u32, u64);

/// Whether `data_type` has an inline encoding.
pub fn is_supported(data_type: &DataType) -> bool {
    let empty = arrow::array::new_empty_array(data_type);
    with_integer_values!(data_type, empty.as_ref(), |_values| true, false)
}

/// The minimum and maximum of the non-null values of `array` (as `i64`, the
/// form that the encoding compares), or `None` when every value is NULL or
/// the type has no encoding.
pub fn min_max(array: &dyn Array) -> Option<(i64, i64)> {
    fn of<T: AsI64>(values: &[T], nulls: Option<&NullBuffer>) -> Option<(i64, i64)> {
        let mut range: Option<(i64, i64)> = None;
        let mut add = |v: i64| {
            range = Some(match range {
                None => (v, v),
                Some((lo, hi)) => (lo.min(v), hi.max(v)),
            });
        };
        match nulls {
            Some(nulls) => nulls.valid_indices().for_each(|i| add(values[i].as_i64())),
            None => {
                // Branch-free over a dense column.
                if let Some(first) = values.first() {
                    let first = first.as_i64();
                    let (lo, hi) = values.iter().fold((first, first), |(lo, hi), v| {
                        let v = v.as_i64();
                        (lo.min(v), hi.max(v))
                    });
                    return Some((lo, hi));
                }
            }
        }
        range
    }
    with_integer_values!(
        array.data_type(),
        array,
        |values| of(values, array.logical_nulls().as_ref()),
        None
    )
}

impl InlineKeys {
    /// The encoding of a build whose key columns, batch by batch, are
    /// `keys[batch][column]`; `None` when the keys have no inline encoding:
    /// an unsupported type, multi-column ranges wider than 64 bits, no
    /// matchable row, or NULL keys that match under `NullEqualsNull` (an
    /// inline table never holds a NULL key).
    pub fn for_build(
        keys: &[Vec<ArrayRef>],
        null_equality: NullEquality,
    ) -> Option<Self> {
        let first = keys.first()?;
        if null_equality == NullEquality::NullEqualsNull
            && keys.iter().flatten().any(|a| a.logical_null_count() > 0)
        {
            return None;
        }
        let types: Vec<DataType> = first.iter().map(|a| a.data_type().clone()).collect();
        if keys.iter().any(|batch| {
            batch.len() != types.len()
                || !batch.iter().zip(&types).all(|(a, t)| a.data_type() == t)
        }) {
            return None;
        }
        let ranges = if types.len() == 1 {
            // A single column needs no range.
            vec![None]
        } else {
            (0..types.len())
                .map(|index| {
                    keys.iter()
                        .filter_map(|batch| min_max(batch[index].as_ref()))
                        .reduce(|(a, b), (lo, hi)| (a.min(lo), b.max(hi)))
                })
                .collect()
        };
        Self::from_ranges(&types, &ranges)
    }

    /// The encoding of key columns of `types` whose non-null build values
    /// lie in `ranges` (as [`min_max`] gives them; a single column ignores
    /// its range). `None` when a type has no encoding, a column of several
    /// has no range (no matchable row), or the ranges need more than 64
    /// bits. The caller makes sure that NULL keys never match
    /// (`NullEqualsNothing`, or no NULL build key).
    pub fn from_ranges(
        types: &[DataType],
        ranges: &[Option<(i64, i64)>],
    ) -> Option<Self> {
        if types.is_empty()
            || types.len() != ranges.len()
            || !types.iter().all(is_supported)
        {
            return None;
        }
        if types.len() == 1 {
            return Some(Self {
                columns: vec![KeyColumn {
                    data_type: types[0].clone(),
                    min: 0,
                    span: u64::MAX,
                    shift: 0,
                }],
            });
        }
        let mut columns = Vec::with_capacity(types.len());
        let mut shift = 0_u32;
        for (data_type, range) in types.iter().zip(ranges) {
            let (lo, hi) = (*range)?;
            if lo > hi {
                return None;
            }
            let span = (i128::from(hi) - i128::from(lo)) as u64;
            let bits = u64::BITS - span.leading_zeros();
            if shift + bits > u64::BITS {
                return None;
            }
            columns.push(KeyColumn {
                data_type: data_type.clone(),
                min: lo,
                span,
                shift: if bits == 0 { 0 } else { shift },
            });
            shift += bits;
        }
        Some(Self { columns })
    }

    /// The encoding of the given column ranges, for tests.
    #[cfg(test)]
    pub(crate) fn columns(&self) -> Vec<(DataType, i64, u64, u32)> {
        self.columns
            .iter()
            .map(|c| (c.data_type.clone(), c.min, c.span, c.shift))
            .collect()
    }

    /// Whether `arrays` have the key types of this encoding.
    pub fn accepts(&self, arrays: &[ArrayRef]) -> bool {
        arrays.len() == self.columns.len()
            && arrays
                .iter()
                .zip(&self.columns)
                .all(|(array, column)| array.data_type() == &column.data_type)
    }

    /// Encodes `arrays` (one per key column, of equal length), or `None`
    /// when their types are not this encoding's.
    pub fn encode(&self, arrays: &[ArrayRef]) -> Option<EncodedKeys> {
        if !self.accepts(arrays) {
            return None;
        }
        let rows = arrays[0].len();
        let mut keys = vec![0_u64; rows];
        let mut nulls: Option<NullBuffer> = None;
        let single = self.columns.len() == 1;
        let mut in_range: Option<BooleanBuffer> = None;
        for (array, column) in arrays.iter().zip(&self.columns) {
            nulls = NullBuffer::union(nulls.as_ref(), array.logical_nulls().as_ref());
            if single {
                with_integer_values!(
                    array.data_type(),
                    array.as_ref(),
                    |values| {
                        for (key, value) in keys.iter_mut().zip(values) {
                            *key = value.as_i64() as u64;
                        }
                    },
                    return None
                );
                continue;
            }
            let (min, span, shift) = (column.min, column.span, column.shift);
            let range = with_integer_values!(
                array.data_type(),
                array.as_ref(),
                |values| {
                    for (key, value) in keys.iter_mut().zip(values) {
                        let delta = value.as_i64().wrapping_sub(min) as u64;
                        *key |= delta << shift;
                    }
                    BooleanBuffer::collect_bool(rows, |i| {
                        values[i].as_i64().wrapping_sub(min) as u64 <= span
                    })
                },
                return None
            );
            in_range = Some(match in_range {
                None => range,
                Some(previous) => &previous & &range,
            });
        }
        let valid = match in_range {
            Some(range) if range.count_set_bits() < rows => {
                NullBuffer::union(nulls.as_ref(), Some(&NullBuffer::new(range)))
            }
            _ => nulls,
        }
        .filter(|valid| valid.null_count() > 0);
        Some(EncodedKeys { keys, valid })
    }

    /// Encodes build key `arrays`: like [`Self::encode`], but every row
    /// without a NULL key must be inside the encoding's range (the range
    /// came from these rows); an error otherwise, never a lost match.
    pub fn encode_build(
        &self,
        arrays: &[ArrayRef],
    ) -> datafusion_common::Result<Vec<u64>> {
        let encoded = self.encode(arrays).ok_or_else(|| {
            datafusion_common::internal_datafusion_err!("join build keys changed type")
        })?;
        let invalid = encoded.valid.as_ref().map_or(0, NullBuffer::null_count);
        if invalid > 0 {
            let nulls = arrays
                .iter()
                .fold(None, |acc: Option<NullBuffer>, a| {
                    NullBuffer::union(acc.as_ref(), a.logical_nulls().as_ref())
                })
                .map_or(0, |n| n.null_count());
            if invalid != nulls {
                return Err(datafusion_common::internal_datafusion_err!(
                    "join build keys outside the range of their inline encoding"
                ));
            }
        }
        Ok(encoded.keys)
    }

    /// The encoded key of build row `row` of `arrays`, or `None` when it has
    /// a NULL key or no encoding. For the rare chains that mix keys.
    pub fn encode_row(&self, arrays: &[ArrayRef], row: usize) -> Option<u64> {
        let mut key = 0_u64;
        for (array, column) in arrays.iter().zip(&self.columns) {
            if array.is_null(row) {
                return None;
            }
            let value = with_integer_values!(
                array.data_type(),
                array.as_ref(),
                |values| values[row].as_i64(),
                return None
            );
            if self.columns.len() == 1 {
                return Some(value as u64);
            }
            let delta = value.wrapping_sub(column.min) as u64;
            if delta > column.span {
                return None;
            }
            key |= delta << column.shift;
        }
        Some(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, Int64Array, UInt64Array};
    use std::sync::Arc;

    fn i64s(values: &[Option<i64>]) -> ArrayRef {
        Arc::new(Int64Array::from(values.to_vec()))
    }

    #[test]
    fn the_switch_reads_like_the_huge_page_switch() {
        assert!(inline_keys_requested(None));
        assert!(inline_keys_requested(Some("")));
        assert!(inline_keys_requested(Some("On")));
        assert!(inline_keys_requested(Some("bogus")));
        for off in ["off", "OFF", "0", "false", "No", "disabled"] {
            assert!(!inline_keys_requested(Some(off)), "{off}");
        }
    }

    #[test]
    fn one_column_encodes_its_values() {
        let build = vec![vec![i64s(&[Some(-5), Some(7), None])]];
        let keys =
            InlineKeys::for_build(&build, NullEquality::NullEqualsNothing).unwrap();
        let probe = i64s(&[Some(-5), None, Some(i64::MAX), Some(i64::MIN)]);
        let encoded = keys.encode(&[probe]).unwrap();
        assert_eq!(encoded.keys[0], (-5_i64) as u64);
        assert_eq!(encoded.keys[2], i64::MAX as u64);
        assert_eq!(encoded.keys[3], i64::MIN as u64);
        let valid = encoded.valid.unwrap();
        assert_eq!(
            (0..4).map(|i| valid.is_valid(i)).collect::<Vec<_>>(),
            [true, false, true, true]
        );
        assert_eq!(keys.encode_row(&build[0], 1), Some(7));
        assert_eq!(keys.encode_row(&build[0], 2), None);
    }

    #[test]
    fn unsigned_64_bit_keys_keep_their_order_free_identity() {
        let build: ArrayRef = Arc::new(UInt64Array::from(vec![u64::MAX, 0]));
        let keys = InlineKeys::for_build(&[vec![build]], NullEquality::NullEqualsNothing)
            .unwrap();
        let probe: ArrayRef = Arc::new(UInt64Array::from(vec![u64::MAX, 1]));
        assert_eq!(keys.encode(&[probe]).unwrap().keys, [u64::MAX, 1]);
    }

    #[test]
    fn columns_pack_into_fields_and_out_of_range_values_are_invalid() {
        let a = i64s(&[Some(10), Some(13)]);
        let b: ArrayRef = Arc::new(Int32Array::from(vec![-2, 1]));
        let keys = InlineKeys::for_build(
            &[vec![Arc::clone(&a), Arc::clone(&b)]],
            NullEquality::NullEqualsNothing,
        )
        .unwrap();
        // a: 10..=13 (2 bits at 0), b: -2..=1 (2 bits at 2).
        assert_eq!(
            keys.columns(),
            [(DataType::Int64, 10, 3, 0), (DataType::Int32, -2, 3, 2),]
        );
        let pa = i64s(&[Some(13), Some(9), Some(14), Some(10), None]);
        let pb: ArrayRef = Arc::new(Int32Array::from(vec![1, 0, 0, -3, 0]));
        let encoded = keys.encode(&[pa, pb]).unwrap();
        assert_eq!(encoded.keys[0], 3 | (3 << 2));
        let valid = encoded.valid.unwrap();
        assert_eq!(
            (0..5).map(|i| valid.is_valid(i)).collect::<Vec<_>>(),
            [true, false, false, false, false]
        );
        assert_eq!(keys.encode_row(&[a, b], 1), Some(3 | (3 << 2)));
    }

    #[test]
    fn a_constant_column_takes_no_bits() {
        let a = i64s(&[Some(4), Some(4)]);
        let b = i64s(&[Some(1), Some(2)]);
        let keys = InlineKeys::for_build(&[vec![a, b]], NullEquality::NullEqualsNothing)
            .unwrap();
        assert_eq!(
            keys.columns(),
            [(DataType::Int64, 4, 0, 0), (DataType::Int64, 1, 1, 0)]
        );
        let encoded = keys
            .encode(&[i64s(&[Some(4), Some(5)]), i64s(&[Some(2), Some(2)])])
            .unwrap();
        assert_eq!(encoded.keys[0], 1);
        assert!(encoded.valid.unwrap().is_null(1));
    }

    #[test]
    fn wide_or_unsupported_keys_have_no_encoding() {
        let wide = i64s(&[Some(i64::MIN), Some(i64::MAX)]);
        let other = i64s(&[Some(0), Some(1)]);
        assert!(
            InlineKeys::for_build(&[vec![wide, other]], NullEquality::NullEqualsNothing)
                .is_none()
        );
        let floats: ArrayRef = Arc::new(arrow::array::Float64Array::from(vec![1.0]));
        assert!(
            InlineKeys::for_build(&[vec![floats]], NullEquality::NullEqualsNothing)
                .is_none()
        );
        let nulls = i64s(&[None, Some(1)]);
        assert!(
            InlineKeys::for_build(&[vec![nulls]], NullEquality::NullEqualsNull).is_none()
        );
    }

    #[test]
    fn build_keys_outside_the_range_are_an_error() {
        let a = i64s(&[Some(1), Some(2)]);
        let b = i64s(&[Some(1), Some(2)]);
        let keys = InlineKeys::for_build(
            &[vec![Arc::clone(&a), Arc::clone(&b)]],
            NullEquality::NullEqualsNothing,
        )
        .unwrap();
        assert_eq!(
            keys.encode_build(&[Arc::clone(&a), i64s(&[Some(1), None])])
                .unwrap()
                .len(),
            2
        );
        assert!(keys.encode_build(&[a, i64s(&[Some(1), Some(3)])]).is_err());
    }

    #[test]
    fn other_probe_types_are_refused() {
        let keys = InlineKeys::for_build(
            &[vec![i64s(&[Some(1)])]],
            NullEquality::NullEqualsNothing,
        )
        .unwrap();
        let probe: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        assert!(keys.encode(&[probe]).is_none());
    }
}
