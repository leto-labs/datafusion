// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Column codes of packed group keys. Each column maps a value to a dense
//! non-negative code below `2^width`; code 0 is NULL. Codes of one column are
//! injective, so the concatenation of all columns' codes is the group key.

use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, BooleanBufferBuilder, PrimitiveArray,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{ArrowNativeType, ArrowNumericType, DataType};
use datafusion_common::{Result, internal_err};

use super::super::{GroupValues, GroupValuesPrimitive, HashValue};

/// What `observe` found.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Observed {
    /// The batch's codes are ready for `pack`.
    Ready,
    /// The column must change its encoding: call `convert` with the codes of
    /// every existing group, then `observe` the batch again.
    Convert,
    /// The column's codes cost more than the general interner: move the
    /// groups there now. The batch's codes are incomplete.
    Costly,
}

/// The codes of one group-by column.
pub(super) trait KeyColumn: Send {
    /// Inspect a batch: record its value range or assign its values' codes.
    fn observe(&mut self, array: &ArrayRef) -> Result<Observed>;
    /// Change the encoding; `codes` are the current codes of every group.
    /// Afterwards `translate` maps each old non-NULL code to its new code.
    fn convert(&mut self, _codes: &[u64]) {
        unreachable!("this column has a single encoding")
    }
    /// Whether the current layout covers every observed value.
    fn fits(&self) -> bool;
    /// A layout covering every observed value, with spare room if `spare`.
    fn propose(&self, spare: bool) -> Proposal;
    /// Adopt a layout. Until `settle`, `translate` maps old codes to new.
    fn adopt(&mut self, proposal: Proposal);
    /// The new code of an existing non-NULL code during a relayout.
    fn translate(&self, code: u64) -> u64;
    /// `translate` every non-NULL code in place: one dynamic call per column.
    fn translate_all(&self, codes: &mut [u64]) {
        for code in codes.iter_mut().filter(|c| **c != 0) {
            *code = self.translate(*code);
        }
    }
    /// Finish a relayout.
    fn settle(&mut self);
    fn width(&self) -> u32;
    /// OR the codes of the observed batch, shifted by `shift`, into `keys`.
    fn pack(&self, array: &ArrayRef, keys: &mut [u64], shift: u32);
    /// Write one column's codes into the whole key buffer.
    fn pack_single(&self, _array: &ArrayRef, _keys: &mut [u32]) {
        unreachable!("only integer columns use narrow batch scratch");
    }
    /// Additional live bytes when converting stored codes into a primitive map.
    fn single_migration_bytes(&self, _groups: usize) -> usize {
        unreachable!("only integer columns have a primitive fallback");
    }
    /// Re-intern stored numeric codes in their existing group-ID order.
    fn fall_back_single(&mut self, _codes: &[u64]) -> Result<Box<dyn GroupValues>> {
        internal_err!("only integer columns have a primitive fallback")
    }
    /// The column of the groups with these codes.
    fn build(&mut self, codes: &[u64]) -> Result<ArrayRef>;
    fn size(&self) -> usize;
    /// Release per-batch scratch while retaining the stored codes.
    fn shrink_to_fit(&mut self) {}
    /// The number of distinct values of a dictionary column.
    fn cardinality(&self) -> usize {
        0
    }
    /// Drop references to input buffers (at an emission).
    fn release(&mut self) {}
    /// Whether the column's codes cost more than the general interner's
    /// work for it, so the groups should move there.
    fn costly(&self) -> bool {
        false
    }
    /// Forget all values and release memory.
    fn reset(&mut self);
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Proposal {
    pub(super) width: u32,
    pub(super) base: i128,
}

/// The number of bits for codes up to and including `max_code`.
pub(super) fn bits_for(max_code: u128) -> u32 {
    128 - max_code.leading_zeros()
}

/// The largest code of a width.
fn max_code(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// Offset codes of integer values: `v` has code `v - base + 1`.
#[derive(Debug, Default)]
pub(super) struct Range {
    base: i128,
    width: u32,
    /// The smallest and largest non-NULL values ever observed.
    seen: Option<(i128, i128)>,
    /// The code delta of the relayout in progress.
    delta: u64,
}

impl Range {
    pub(super) fn base(&self) -> i128 {
        self.base
    }
    pub(super) fn width(&self) -> u32 {
        self.width
    }
    pub(super) fn observe(&mut self, lo: i128, hi: i128) {
        self.seen = Some(match self.seen {
            Some((a, b)) => (a.min(lo), b.max(hi)),
            None => (lo, hi),
        });
    }
    /// The width of the exact observed range, or 0 without values.
    pub(super) fn exact_width(&self) -> u32 {
        self.propose(false, i128::MIN, i128::MAX).width
    }
    pub(super) fn fits(&self) -> bool {
        match self.seen {
            None => true,
            Some((lo, hi)) => {
                self.base <= lo
                    && hi
                        .checked_sub(self.base)
                        .is_some_and(|span| (span as u128) < max_code(self.width))
            }
        }
    }
    /// A layout of the observed range; with `spare`, its span again on each
    /// side inside `[low, high]`, so a growing range relayouts (and rehashes
    /// every group) a logarithmic number of times. Fact rows often arrive
    /// in date order, which grows a range at one end batch after batch.
    pub(super) fn propose(&self, spare: bool, low: i128, high: i128) -> Proposal {
        let Some((lo, hi)) = self.seen else {
            return Proposal { width: 0, base: 0 };
        };
        // A span beyond i128 needs more than 64 bits; it never fits a key.
        let Some(span) = hi.checked_sub(lo) else {
            return Proposal {
                width: 128,
                base: lo,
            };
        };
        let (lo, hi) = if spare {
            let pad = span.max(8);
            (
                lo.saturating_sub(pad).max(low),
                hi.saturating_add(pad).min(high),
            )
        } else {
            (lo, hi)
        };
        Proposal {
            width: hi
                .checked_sub(lo)
                .map_or(128, |span| bits_for(span as u128 + 1)),
            base: lo,
        }
    }
    pub(super) fn adopt(&mut self, proposal: Proposal) {
        // Before any non-NULL code exists there is nothing to translate.
        // Otherwise the new base is at most the old one plus its span, so the
        // wrapping difference is the exact (possibly negative) delta.
        self.delta = if self.width == 0 {
            0
        } else {
            self.base.wrapping_sub(proposal.base) as u64
        };
        self.base = proposal.base;
        self.width = proposal.width;
    }
    pub(super) fn translate(&self, code: u64) -> u64 {
        code.wrapping_add(self.delta)
    }
    pub(super) fn translate_all(&self, codes: &mut [u64]) {
        if self.delta != 0 {
            for code in codes.iter_mut().filter(|c| **c != 0) {
                *code = code.wrapping_add(self.delta);
            }
        }
    }
    pub(super) fn settle(&mut self) {
        self.delta = 0;
    }
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Native integer types whose values map into `i128` and back.
pub(super) trait PackNative: ArrowNativeType + HashValue {
    const LOW: i128;
    const HIGH: i128;
    fn wide(self) -> i128;
    fn narrow(value: i128) -> Self;
    /// The low 64 bits of the value's two's-complement form.
    fn low64(self) -> u64;
}

macro_rules! pack_native {
    ($($t:ty),*) => {$(
        impl PackNative for $t {
            const LOW: i128 = <$t>::MIN as i128;
            const HIGH: i128 = <$t>::MAX as i128;
            #[inline(always)]
            fn wide(self) -> i128 {
                self as i128
            }
            #[inline(always)]
            fn narrow(value: i128) -> Self {
                value as $t
            }
            #[inline(always)]
            fn low64(self) -> u64 {
                self as i128 as u64
            }
        }
    )*};
}
pack_native!(i8, i16, i32, i64, u8, u16, u32, u64, i128);

/// The smallest and largest non-NULL values, in one pass.
fn min_max<T: ArrowNumericType>(array: &PrimitiveArray<T>) -> Option<(i128, i128)>
where
    T::Native: PackNative,
{
    let start = (i128::MAX, i128::MIN);
    let (lo, hi) = match array.nulls().filter(|n| n.null_count() > 0) {
        None => {
            // Native-width min and max vectorize; widen once at the end.
            let values = array.values();
            let first = *values.first()?;
            let (lo, hi) = values.iter().fold((first, first), |(lo, hi), &v| {
                (if v < lo { v } else { lo }, if v > hi { v } else { hi })
            });
            (lo.wide(), hi.wide())
        }
        // Arrow's null-aware kernels are vectorized over validity chunks.
        Some(_) => match (arrow::compute::min(array), arrow::compute::max(array)) {
            (Some(lo), Some(hi)) => (lo.wide(), hi.wide()),
            _ => start,
        },
    };
    (lo <= hi).then_some((lo, hi))
}

/// OR `code(v) << shift` into each key, where `code(v) = v - base + 1` in
/// wrapping 64-bit arithmetic and NULL rows contribute 0. Every value lies
/// within 2^64 above the base, so wrapping gives the exact code.
#[inline(always)]
pub(super) fn pack_offsets(
    values: impl Iterator<Item = u64>,
    nulls: Option<&NullBuffer>,
    base: u64,
    keys: &mut [u64],
    shift: u32,
) {
    match nulls.filter(|n| n.null_count() > 0) {
        None => {
            for (key, v) in keys.iter_mut().zip(values) {
                *key |= v.wrapping_sub(base).wrapping_add(1) << shift;
            }
        }
        Some(nulls) => {
            // Pack every row (a vectorized loop), then clear the NULL rows,
            // found 64 at a time from the validity chunks: a per-row bit
            // iterator cost more than the packing itself.
            for (key, v) in keys.iter_mut().zip(values) {
                *key |= v.wrapping_sub(base).wrapping_add(1) << shift;
            }
            for_each_null(nulls, |row| keys[row] &= !(u64::MAX << shift));
        }
    }
}

/// Call `f` with the index of every NULL row, from 64-bit validity chunks.
pub(super) fn for_each_null(nulls: &NullBuffer, mut f: impl FnMut(usize)) {
    let chunks = nulls.inner().bit_chunks();
    let mut base = 0;
    for chunk in chunks.iter() {
        let mut missing = !chunk;
        while missing != 0 {
            f(base + missing.trailing_zeros() as usize);
            missing &= missing - 1;
        }
        base += 64;
    }
    let tail = chunks.remainder_len();
    if tail > 0 {
        let mut missing = !chunks.remainder_bits() & ((1u64 << tail) - 1);
        while missing != 0 {
            f(base + missing.trailing_zeros() as usize);
            missing &= missing - 1;
        }
    }
}

/// An integer-like column with offset codes.
pub(super) struct IntColumn<T: ArrowNumericType> {
    data_type: DataType,
    range: Range,
    phantom: PhantomData<fn() -> T>,
}

impl<T: ArrowNumericType> IntColumn<T>
where
    T::Native: PackNative,
{
    pub(super) fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            range: Range::default(),
            phantom: PhantomData,
        }
    }
}

impl<T: ArrowNumericType> KeyColumn for IntColumn<T>
where
    T::Native: PackNative,
{
    fn observe(&mut self, array: &ArrayRef) -> Result<Observed> {
        let array = array.as_primitive::<T>();
        if let Some((lo, hi)) = min_max(array) {
            self.range.observe(lo, hi);
        }
        Ok(Observed::Ready)
    }

    fn fits(&self) -> bool {
        self.range.fits()
    }

    fn propose(&self, spare: bool) -> Proposal {
        self.range.propose(spare, T::Native::LOW, T::Native::HIGH)
    }

    fn adopt(&mut self, proposal: Proposal) {
        self.range.adopt(proposal);
    }

    fn translate(&self, code: u64) -> u64 {
        self.range.translate(code)
    }

    fn settle(&mut self) {
        self.range.settle();
    }

    fn width(&self) -> u32 {
        self.range.width()
    }

    fn pack(&self, array: &ArrayRef, keys: &mut [u64], shift: u32) {
        if self.range.width() == 0 {
            // Only NULLs so far: every code is 0.
            return;
        }
        let array = array.as_primitive::<T>();
        pack_offsets(
            array.values().iter().map(|v| v.low64()),
            array.nulls(),
            self.range.base() as u64,
            keys,
            shift,
        );
    }

    fn pack_single(&self, array: &ArrayRef, keys: &mut [u32]) {
        if self.range.width() == 0 {
            keys.fill(0);
            return;
        }
        let array = array.as_primitive::<T>();
        let base = self.range.base() as u32;
        for (key, value) in keys.iter_mut().zip(array.values().iter()) {
            *key = (value.low64() as u32).wrapping_sub(base).wrapping_add(1);
        }
        if let Some(nulls) = array.nulls().filter(|nulls| nulls.null_count() > 0) {
            for_each_null(nulls, |row| keys[row] = 0);
        }
    }

    fn single_migration_bytes(&self, groups: usize) -> usize {
        let capacity = groups.max(GroupValuesPrimitive::<T>::INITIAL_CAPACITY);
        // Hashbrown uses power-of-two buckets at a 7/8 load. Round up
        // using 3/4 and include control bytes and SIMD padding. Both
        // primitive map layouts use at most 16 bytes per bucket.
        let buckets = capacity
            .saturating_add(capacity.div_ceil(3))
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX);
        let native_bytes = size_of::<T::Native>();
        buckets
            .saturating_mul(17)
            .saturating_add(capacity.saturating_mul(native_bytes))
            .saturating_add(groups.saturating_mul(native_bytes + size_of::<usize>()))
            .saturating_add(groups.div_ceil(8))
            .saturating_add(size_of::<GroupValuesPrimitive<T>>() + 256)
    }

    fn fall_back_single(&mut self, codes: &[u64]) -> Result<Box<dyn GroupValues>> {
        let count = codes.len();
        if count == 0 {
            return Ok(Box::new(GroupValuesPrimitive::<T>::new(
                self.data_type.clone(),
            )));
        }
        let capacity = count.max(GroupValuesPrimitive::<T>::INITIAL_CAPACITY);
        let mut primitive =
            GroupValuesPrimitive::<T>::with_capacity(self.data_type.clone(), capacity);
        let array = self.build(codes)?;
        let mut ids = Vec::with_capacity(count);
        primitive.intern(&[array], &mut ids)?;
        if ids.len() != count || ids.into_iter().enumerate().any(|(i, id)| i != id) {
            return internal_err!("primitive interner reassigned packed group ids");
        }
        Ok(Box::new(primitive))
    }

    fn translate_all(&self, codes: &mut [u64]) {
        self.range.translate_all(codes);
    }

    fn build(&mut self, codes: &[u64]) -> Result<ArrayRef> {
        let len = codes.len();
        let mut values = Vec::with_capacity(len);
        let mut valid = BooleanBufferBuilder::new(len);
        let mut nulls = 0;
        let base = self.range.base();
        for &code in codes {
            if code == 0 {
                values.push(T::Native::default());
                valid.append(false);
                nulls += 1;
            } else {
                values.push(T::Native::narrow(base + (i128::from(code) - 1)));
                valid.append(true);
            }
        }
        let nulls = (nulls > 0).then(|| NullBuffer::new(valid.finish()));
        Ok(Arc::new(
            PrimitiveArray::<T>::new(values.into(), nulls)
                .with_data_type(self.data_type.clone()),
        ))
    }

    fn size(&self) -> usize {
        0
    }

    fn reset(&mut self) {
        self.range.reset();
    }
}

/// A boolean column: NULL 0, false 1, true 2.
#[derive(Default)]
pub(super) struct BoolColumn;

impl KeyColumn for BoolColumn {
    fn observe(&mut self, _array: &ArrayRef) -> Result<Observed> {
        Ok(Observed::Ready)
    }
    fn fits(&self) -> bool {
        true
    }
    fn propose(&self, _spare: bool) -> Proposal {
        Proposal { width: 2, base: 0 }
    }
    fn adopt(&mut self, _proposal: Proposal) {}
    fn translate(&self, code: u64) -> u64 {
        code
    }
    fn settle(&mut self) {}
    fn width(&self) -> u32 {
        2
    }
    fn pack(&self, array: &ArrayRef, keys: &mut [u64], shift: u32) {
        let array = array.as_boolean();
        pack_offsets(
            array.values().iter().map(u64::from),
            array.nulls(),
            0,
            keys,
            shift,
        );
    }
    fn build(&mut self, codes: &[u64]) -> Result<ArrayRef> {
        Ok(Arc::new(
            codes
                .iter()
                .map(|&code| (code != 0).then_some(code == 2))
                .collect::<BooleanArray>(),
        ))
    }
    fn size(&self) -> usize {
        0
    }
    fn reset(&mut self) {}
}
