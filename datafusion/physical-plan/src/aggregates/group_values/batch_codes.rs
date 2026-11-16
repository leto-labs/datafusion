// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Batch-local dictionary codes for repeated byte-view values. Fingerprints
//! select candidates, complete bytes prove equality, and the underlying
//! interner owns cross-batch equality and group identity.

use super::GroupValues;
use arrow::{
    array::{Array, ArrayRef, AsArray, StringViewArray, UInt32Array},
    compute::take,
};
use datafusion_common::Result;
use datafusion_expr::{EmitTo, GroupSelection};
use std::mem::size_of;

pub(super) const SLOTS: usize = 1024;
const SAMPLE: usize = 512;
const EMPTY: usize = usize::MAX;

pub(super) struct BatchCodes {
    inner: Box<dyn GroupValues>,
    slots: Vec<(u128, usize)>,
    rows: Vec<u32>,
    codes: Vec<usize>,
}
impl BatchCodes {
    pub(super) fn new(inner: Box<dyn GroupValues>) -> Self {
        Self {
            inner,
            slots: Vec::new(),
            rows: Vec::new(),
            codes: Vec::new(),
        }
    }
    fn fingerprint(array: &StringViewArray, row: usize) -> u128 {
        let view = array.views()[row];
        if view as u32 <= 12 {
            return view;
        }
        let bytes = array.value(row).as_bytes();
        let tail = u64::from_le_bytes(
            bytes[bytes.len() - 8..]
                .try_into()
                .expect("external view has at least eight bytes"),
        );
        u128::from(view as u64) | (u128::from(tail) << 64)
    }
    fn equal(array: &StringViewArray, left: usize, right: usize) -> bool {
        array.views()[left] == array.views()[right]
            || array.value(left) == array.value(right)
    }
    pub(super) fn slot(key: u128) -> usize {
        #[cfg(feature = "force_hash_collisions")]
        {
            let _ = key;
            0
        }
        #[cfg(not(feature = "force_hash_collisions"))]
        {
            let mixed = (key as u64) ^ ((key >> 64) as u64).rotate_left(23);
            (mixed.wrapping_mul(0x9e3779b97f4a7c15) >> (64 - SLOTS.ilog2())) as usize
        }
    }
}
impl GroupValues for BatchCodes {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);
        let array = cols[0].as_string_view();
        // Small or mostly-distinct batches do not amortize a second interning
        // pass. Sampling is a cost decision, never an equality assumption.
        if array.len() < SAMPLE || array.len() > u32::MAX as usize {
            return self.inner.intern(cols, groups);
        }
        self.slots.resize(SLOTS, (0, EMPTY));
        self.slots.fill((0, EMPTY));
        let mut hits = 0;
        let mut seen_null = false;
        let mut external = false;
        for row in 0..SAMPLE {
            external |= array.views()[row] as u32 > 12;
            if array.is_null(row) {
                hits += usize::from(seen_null);
                seen_null = true;
                continue;
            }
            let value = Self::fingerprint(array, row);
            let entry = &mut self.slots[Self::slot(value)];
            if entry.1 != EMPTY
                && entry.0 == value
                && (value as u32 <= 12 || Self::equal(array, row, entry.1))
            {
                hits += 1;
            }
            *entry = (value, row);
        }
        let minimum_hits = if external { SAMPLE / 4 } else { SAMPLE / 2 };
        if hits < minimum_hits {
            return self.inner.intern(cols, groups);
        }
        self.slots.fill((0, EMPTY));
        self.rows.clear();
        self.codes.clear();
        let mut null_code = None;
        for row in 0..array.len() {
            if array.is_null(row) {
                let code = *null_code.get_or_insert_with(|| {
                    let code = self.rows.len();
                    self.rows.push(row as u32);
                    code
                });
                self.codes.push(code);
                continue;
            }
            let value = Self::fingerprint(array, row);
            let entry = &mut self.slots[Self::slot(value)];
            let code = if entry.1 != EMPTY
                && entry.0 == value
                && (value as u32 <= 12
                    || Self::equal(array, row, self.rows[entry.1] as usize))
            {
                entry.1
            } else {
                let code = self.rows.len();
                self.rows.push(row as u32);
                *entry = (value, code);
                code
            };
            self.codes.push(code);
        }
        // Fingerprints only select candidates: complete bytes prove equality.
        // Collisions add representatives for the canonical interner to unify.
        let values = take(
            array,
            &UInt32Array::from(std::mem::take(&mut self.rows)),
            None,
        )?;
        self.inner.intern(&[values], groups)?;
        for code in &mut self.codes {
            *code = groups[*code];
        }
        std::mem::swap(groups, &mut self.codes);
        Ok(())
    }
    fn size(&self) -> usize {
        size_of::<Self>()
            + self.inner.size()
            + self.slots.capacity() * size_of::<(u128, usize)>()
            + self.rows.capacity() * size_of::<u32>()
            + self.codes.capacity() * size_of::<usize>()
    }
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn emit(&mut self, emit: EmitTo) -> Result<Vec<ArrayRef>> {
        self.inner.emit(emit)
    }
    fn values_preserving(
        &mut self,
        selection: GroupSelection<'_>,
    ) -> Result<Vec<ArrayRef>> {
        self.inner.values_preserving(selection)
    }
    fn supports_values_preserving(&self) -> bool {
        self.inner.supports_values_preserving()
    }
    fn shrink_to_fit(&mut self) {
        self.slots = Vec::new();
        self.rows = Vec::new();
        self.codes = Vec::new();
        self.inner.shrink_to_fit();
    }
    fn clear_shrink(&mut self, rows: usize) {
        self.slots = Vec::new();
        self.rows = Vec::new();
        self.codes = Vec::new();
        self.inner.clear_shrink(rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregates::group_values::single_group_by::bytes_view::GroupValuesBytesView;
    use arrow::{array::StringViewArray, buffer::NullBuffer};
    use datafusion_physical_expr::binary_map::OutputType;
    use std::sync::Arc;

    fn base() -> Box<dyn GroupValues> {
        Box::new(GroupValuesBytesView::new(OutputType::Utf8View))
    }
    #[test]
    fn batch_codes_match_values_across_dictionaries_nulls_and_emission() -> Result<()> {
        let mut candidate = BatchCodes::new(base());
        let mut reference = base();
        let mut state = 19u64;
        for batch in 0..24 {
            // Reorder the dictionary and regenerate its buffer each batch.
            let dict = StringViewArray::from_iter_values((0..29).map(|i| {
                if batch % 3 == 0 {
                    format!("s{}", (i + batch) % 29)
                } else {
                    format!("long-shared-prefix-{}", (i + batch) % 29)
                }
            }));
            let mut views = Vec::new();
            let mut valid = Vec::new();
            for row in 0..8192 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                views.push(dict.views()[(state as usize >> 16) % dict.len()]);
                valid.push(row % 13 != 0);
            }
            let array = StringViewArray::try_new(
                views.into(),
                dict.data_buffers().to_vec(),
                Some(NullBuffer::from(valid)),
            )?;
            let cols = [Arc::new(array) as ArrayRef];
            let (mut actual, mut expected) = (Vec::new(), Vec::new());
            candidate.intern(&cols, &mut actual)?;
            reference.intern(&cols, &mut expected)?;
            assert_eq!(actual, expected, "batch {batch}");
            assert_eq!(candidate.len(), reference.len());
            if batch % 7 == 0 {
                let emit = if batch == 14 {
                    EmitTo::All
                } else {
                    EmitTo::First(7)
                };
                assert_eq!(candidate.emit(emit)?, reference.emit(emit)?);
            }
            if batch == 18 {
                candidate.clear_shrink(0);
                reference.clear_shrink(0);
            }
        }
        assert_eq!(candidate.emit(EmitTo::All)?, reference.emit(EmitTo::All)?);
        Ok(())
    }
    #[test]
    fn batch_codes_collisions_and_equal_bytes_at_different_offsets() -> Result<()> {
        let mut candidate = BatchCodes::new(base());
        let mut reference = base();
        for turn in 0..3 {
            let dict = StringViewArray::from(vec![
                "long-shared-prefix-first",
                "long-shared-prefix-other",
            ]);
            let array = StringViewArray::try_new(
                (0..8192)
                    .map(|row| dict.views()[((row / 128 + turn) % 3) % 2])
                    .collect::<Vec<_>>()
                    .into(),
                dict.data_buffers().to_vec(),
                Some(NullBuffer::from(
                    (0..8192)
                        .map(|row| (row / 128 + turn) % 3 != 2)
                        .collect::<Vec<_>>(),
                )),
            )?;
            let cols = [Arc::new(array) as ArrayRef];
            let (mut actual, mut expected) = (Vec::new(), Vec::new());
            candidate.intern(&cols, &mut actual)?;
            reference.intern(&cols, &mut expected)?;
            assert_eq!(actual, expected);
        }
        assert_eq!(candidate.emit(EmitTo::All)?, reference.emit(EmitTo::All)?);
        Ok(())
    }
    #[test]
    fn batch_codes_distinguish_colliding_complete_descriptors() -> Result<()> {
        let mut seen = std::collections::HashMap::new();
        let (left, right) = (0..SLOTS * 2)
            .find_map(|id| {
                let text = format!("c{id}");
                let a = StringViewArray::from(vec![text.as_str()]);
                let slot = BatchCodes::slot(a.views()[0]);
                seen.insert(slot, text.clone()).map(|prior| (prior, text))
            })
            .unwrap();
        let array = Arc::new(StringViewArray::from_iter_values((0..8192).map(|row| {
            if row / 128 % 2 == 0 {
                left.as_str()
            } else {
                right.as_str()
            }
        }))) as ArrayRef;
        let mut candidate = BatchCodes::new(base());
        let mut reference = base();
        let (mut actual, mut expected) = (Vec::new(), Vec::new());
        candidate.intern(std::slice::from_ref(&array), &mut actual)?;
        reference.intern(&[array], &mut expected)?;
        assert!(!candidate.codes.is_empty(), "must exercise code reduction");
        assert!(candidate.codes.len() < actual.len() / 4);
        assert_eq!(actual, expected);
        assert_eq!(candidate.len(), 2);
        Ok(())
    }
    #[test]
    fn value_codes_compare_middle_bytes_and_deduplicate_different_offsets() -> Result<()>
    {
        let left = "same-prefix-first-middle-same-suffix";
        let right = "same-prefix-other-middle-same-suffix";
        assert_eq!(left.len(), right.len());
        let array = StringViewArray::from_iter_values(
            (0..8192).map(|row| if row / 128 % 2 == 0 { left } else { right }),
        );
        assert_eq!(
            BatchCodes::fingerprint(&array, 0),
            BatchCodes::fingerprint(&array, 128)
        );
        let array = Arc::new(array) as ArrayRef;
        let mut candidate = BatchCodes::new(base());
        let mut reference = base();
        let (mut actual, mut expected) = (Vec::new(), Vec::new());
        candidate.intern(std::slice::from_ref(&array), &mut actual)?;
        reference.intern(&[array], &mut expected)?;
        assert!(!candidate.codes.is_empty());
        assert!(candidate.codes.len() < actual.len() / 4);
        assert_eq!(actual, expected);
        assert_eq!(candidate.len(), 2);
        Ok(())
    }
    #[test]
    fn batch_codes_match_distinct_sliced_and_empty_inputs() -> Result<()> {
        let mut candidate = BatchCodes::new(base());
        let mut reference = base();
        for len in [0, 1, 511, 512, 8192] {
            let a = StringViewArray::from_iter_values(
                (0..len + 9).map(|i| format!("long-distinct-value-{i}")),
            )
            .slice(9, len);
            let cols = [Arc::new(a) as ArrayRef];
            let (mut actual, mut expected) = (Vec::new(), Vec::new());
            candidate.intern(&cols, &mut actual)?;
            reference.intern(&cols, &mut expected)?;
            assert_eq!(actual, expected);
        }
        assert_eq!(candidate.emit(EmitTo::All)?, reference.emit(EmitTo::All)?);
        Ok(())
    }
}
