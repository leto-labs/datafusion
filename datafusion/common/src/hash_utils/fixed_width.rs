// Modified by the Longbow authors; see third_party/README.md.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Dense primitive hashing for foldhash 0.2.0's fast fixed state.
//!
//! The scalar code uses a widening u128 product. AVX-512 has no packed
//! multiply-high for u64, so use exact 32-bit partial products with carries.

use super::{HashState, HashValue};
use crate::float_semantics::hash_input;
use arrow::array::types::{IntervalDayTime, IntervalMonthDayNano};
use arrow::array::{Array, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, i256};
use half::f16;

// foldhash 0.2.0's ARBITRARY3 and the first two words of FIXED_GLOBAL_SEED.
const REHASH_SEED_XOR: u64 = 0x082e_fa98_ec4e_6c89;
const NUMERIC_SEED: u64 = 0xc0ac_29b7_c97c_50dd;
const BYTE_SEED: u64 = 0x3f84_d5b5_b547_0917;

#[inline(always)]
fn fold(x: u64, y: u64) -> u64 {
    let (x_low, x_high) = (x as u32 as u64, x >> 32);
    let (y_low, y_high) = (y as u32 as u64, y >> 32);
    let low_product = x_low * y_low;
    // Each cross product is at most (2^32 - 1)^2 and receives at most
    // 2^32 - 1 in carry, so both intermediate additions fit in u64.
    let cross = x_high * y_low + (low_product >> 32);
    let middle = x_low * y_high + (cross as u32 as u64);
    let high = x_high * y_high + (cross >> 32) + (middle >> 32);
    let low = (middle << 32) | (low_product as u32 as u64);
    low ^ high
}

pub(super) trait FixedWidthHash: Copy + HashValue {
    fn try_hash_dense(
        values: &[Self],
        hashes: &mut [u64],
        seed: u64,
        rehash: bool,
    ) -> bool;
}

impl<N: FoldValue> FixedWidthHash for N {
    fn try_hash_dense(
        values: &[Self],
        hashes: &mut [u64],
        seed: u64,
        rehash: bool,
    ) -> bool {
        if !simd_available() {
            return false;
        }
        // SAFETY: all four AVX-512 features enabled by hash_dense were
        // detected on this CPU. Loads and stores use safe slice iteration.
        unsafe {
            if rehash {
                hash_dense::<Self, true>(values, hashes, seed);
            } else {
                hash_dense::<Self, false>(values, hashes, seed);
            }
        }
        true
    }
}

// Decimal256 hashes two dependent widening products per value. The scalar
// loop is faster for this Arrow layout on the measured AVX-512 host.
impl FixedWidthHash for i256 {
    fn try_hash_dense(_: &[Self], _: &mut [u64], _: u64, _: bool) -> bool {
        false
    }
}

trait FoldValue: Copy + HashValue {
    /// Fold one value into the accumulator used by a fresh fast FoldHasher.
    /// Later columns use `hash_write`, which feeds float bytes separately.
    fn fold_value<const REHASH: bool>(self, accumulator: u64) -> u64;
}

macro_rules! integer_fold {
    ($(($native:ty, $unsigned:ty)),+ $(,)?) => {
        $(impl FoldValue for $native {
            #[inline(always)]
            fn fold_value<const REHASH: bool>(self, accumulator: u64) -> u64 {
                fold(accumulator ^ (self as $unsigned as u64), NUMERIC_SEED)
            }
        })+
    };
}
integer_fold!(
    (i8, u8),
    (u8, u8),
    (i16, u16),
    (u16, u16),
    (i32, u32),
    (u32, u32),
    (i64, u64),
    (u64, u64),
);

#[inline(always)]
fn fold_u128(value: u128, accumulator: u64) -> u64 {
    fold(
        accumulator ^ value as u64,
        NUMERIC_SEED ^ (value >> 64) as u64,
    )
}

impl FoldValue for i128 {
    #[inline(always)]
    fn fold_value<const REHASH: bool>(self, accumulator: u64) -> u64 {
        fold_u128(self as u128, accumulator)
    }
}

impl FoldValue for IntervalDayTime {
    #[inline(always)]
    fn fold_value<const REHASH: bool>(self, accumulator: u64) -> u64 {
        // FoldHasher packs the two i32 writes in order into the sponge.
        let packed =
            u64::from(self.days as u32) | ((self.milliseconds as u32 as u64) << 32);
        fold(accumulator ^ packed, NUMERIC_SEED)
    }
}

impl FoldValue for IntervalMonthDayNano {
    #[inline(always)]
    fn fold_value<const REHASH: bool>(self, accumulator: u64) -> u64 {
        let packed = u64::from(self.months as u32) | ((self.days as u32 as u64) << 32);
        fold(accumulator ^ packed, NUMERIC_SEED ^ self.nanoseconds as u64)
    }
}

impl FoldValue for f16 {
    #[inline(always)]
    fn fold_value<const REHASH: bool>(self, accumulator: u64) -> u64 {
        let bits = u64::from(hash_input(self).to_bits());
        if REHASH {
            // FoldHasher's short byte path reads byte 0 on the left and
            // repeats byte 1 (middle and last) on the right.
            fold(
                accumulator.rotate_right(2) ^ (bits & 0xff),
                BYTE_SEED ^ (bits & 0xff00) ^ (bits >> 8),
            )
        } else {
            fold(accumulator ^ bits, NUMERIC_SEED)
        }
    }
}

macro_rules! float_fold {
    ($(($native:ty, $width:literal)),+ $(,)?) => {
        $(impl FoldValue for $native {
            #[inline(always)]
            fn fold_value<const REHASH: bool>(self, accumulator: u64) -> u64 {
                let bits = u64::from(hash_input(self).to_bits());
                if REHASH {
                    fold(accumulator.rotate_right($width) ^ bits, BYTE_SEED ^ bits)
                } else {
                    fold(accumulator ^ bits, NUMERIC_SEED)
                }
            }
        })+
    };
}
float_fold!((f32, 4), (f64, 8));

pub(super) fn try_hash_primitive<T: ArrowPrimitiveType<Native: FixedWidthHash>>(
    array: &PrimitiveArray<T>,
    hashes: &mut [u64],
    state: &impl HashState,
    rehash: bool,
) -> bool {
    // Leave short batches and other hasher families with the original loop.
    if array.len() < 32 || array.len() != hashes.len() || array.null_count() != 0 {
        return false;
    }
    let Some(seed) = state.fast_foldhash_seed() else {
        return false;
    };
    T::Native::try_hash_dense(array.values().as_ref(), hashes, seed, rehash)
}

fn simd_available() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

// Keep the vector loops out of the large scalar array-dispatch function.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
fn hash_dense<N: FoldValue, const REHASH: bool>(
    values: &[N],
    hashes: &mut [u64],
    seed: u64,
) {
    for (&value, hash) in values.iter().zip(hashes) {
        let accumulator = if REHASH {
            *hash ^ REHASH_SEED_XOR
        } else {
            seed
        };
        *hash = value.fold_value::<REHASH>(accumulator);
    }
}

#[cfg(test)]
mod tests;
