// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! A bounded cache of complete inline string grouping keys. The underlying
//! interner assigns every group ID and owns its values. A cache miss delegates
//! the entire batch, so new groups and hash collisions retain those semantics.

#[cfg(not(feature = "force_hash_collisions"))]
use std::hash::BuildHasher;
use std::mem::size_of;

use arrow::array::{Array, ArrayRef, AsArray, StringViewArray};
use datafusion_common::Result;
use datafusion_common::hash_utils::RandomState;
use datafusion_expr::{EmitTo, GroupSelection};
use hashbrown::HashTable;

use super::{
    GroupValues,
    batch_codes::{BatchCodes, SLOTS},
};

const MAX_GROUPS: usize = 16_384;
// An inline view has a length at most 12; this is an unambiguous NULL key.
const NULL_KEY: u128 = u128::MAX;

pub(super) struct InlineCache<const N: usize> {
    inner: Box<dyn GroupValues>,
    cache: HashTable<([u128; N], usize)>,
    front: Vec<(u128, usize)>,
    state: RandomState,
    disabled: bool,
}

impl<const N: usize> InlineCache<N> {
    pub(super) fn new(inner: Box<dyn GroupValues>) -> Self {
        Self {
            inner,
            cache: HashTable::new(),
            front: Vec::new(),
            state: crate::aggregates::AGGREGATION_HASH_SEED,
            disabled: false,
        }
    }

    fn hash_key(&self, key: &[u128; N]) -> u64 {
        Self::key_hash(&self.state, key)
    }
    fn key_hash(state: &RandomState, key: &[u128; N]) -> u64 {
        #[cfg(feature = "force_hash_collisions")]
        {
            let _ = (state, key);
            0
        }
        #[cfg(not(feature = "force_hash_collisions"))]
        {
            match N {
                1 => state.hash_one(key[0]),
                2 => state.hash_one((key[0], key[1])),
                _ => state.hash_one(key),
            }
        }
    }
    fn key(arrays: &[&StringViewArray; N], row: usize) -> [u128; N] {
        std::array::from_fn(|i| {
            if arrays[i].is_null(row) {
                NULL_KEY
            } else {
                arrays[i].views()[row]
            }
        })
    }

    fn cached_groups<const CHECK_LENGTHS: bool>(
        &mut self,
        arrays: &[&StringViewArray; N],
        groups: &mut Vec<usize>,
    ) -> bool {
        groups.clear();
        for row in 0..arrays[0].len() {
            let key = Self::key(arrays, row);
            // Only complete inline keys can hit. NULL is its own complete key.
            // Fuse this check with lookup when a decoder retains buffers.
            if CHECK_LENGTHS && key.iter().any(|&key| key as u32 > 12 && key != NULL_KEY)
            {
                return false;
            }
            // Complete inline values can reuse a canonical group code before
            // hashing. Collisions fall through to the exact-key table.
            if N == 1 && !self.front.is_empty() {
                let (stored, group) = self.front[BatchCodes::slot(key[0])];
                if group != usize::MAX && stored == key[0] {
                    groups.push(group);
                    continue;
                }
            }
            let hash = self.hash_key(&key);
            if let Some(group) = self
                .cache
                .find(hash, |(stored, _)| *stored == key)
                .map(|(_, group)| *group)
            {
                if N == 1 && !self.front.is_empty() {
                    self.front[BatchCodes::slot(key[0])] = (key[0], group);
                }
                groups.push(group);
            } else {
                return false;
            }
        }
        true
    }

    fn reset(&mut self) {
        self.cache.clear();
        self.front.clear();
        self.disabled = false;
    }
}

impl<const N: usize> GroupValues for InlineCache<N> {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), N);
        if self.disabled {
            return self.inner.intern(cols, groups);
        }
        let arrays = std::array::from_fn(|i| cols[i].as_string_view());
        // Decoders can retain an unused dictionary buffer for inline keys.
        // A hit needs one pass, including its length guard, rather than a
        // separate full-array prescan. Buffer-free arrays need no guard.
        let check_lengths = arrays.iter().any(|a| !a.data_buffers().is_empty());
        let all_cached = if check_lengths {
            self.cached_groups::<true>(&arrays, groups)
        } else {
            self.cached_groups::<false>(&arrays, groups)
        };
        if all_cached {
            return Ok(());
        }
        self.inner.intern(cols, groups)?;
        // Misses can populate the cache only after proving every view inline.
        // External buffer identities never become cache keys across batches.
        if check_lengths
            && arrays.iter().any(|a| {
                !a.data_buffers().is_empty()
                    && a.views().iter().any(|&view| view as u32 > 12)
            })
        {
            return Ok(());
        }
        if self.inner.len() > MAX_GROUPS {
            self.cache.clear();
            self.cache.shrink_to(0, |_| 0);
            self.front.clear();
            self.front.shrink_to_fit();
            self.disabled = true;
            return Ok(());
        }
        if N == 1 {
            if self.inner.len() <= SLOTS / 2 {
                self.front.resize(SLOTS, (0, usize::MAX));
            } else {
                self.front.clear();
                self.front.shrink_to_fit();
            }
        }
        for (row, &group) in groups.iter().enumerate() {
            let key = Self::key(&arrays, row);
            if N == 1 && !self.front.is_empty() {
                self.front[BatchCodes::slot(key[0])] = (key[0], group);
            }
            let hash = self.hash_key(&key);
            self.cache
                .entry(
                    hash,
                    |(stored, _)| *stored == key,
                    |(stored, _)| Self::key_hash(&self.state, stored),
                )
                .or_insert((key, group));
        }
        Ok(())
    }

    fn try_reserve_empty(
        &mut self,
        groups: usize,
        admit: &mut dyn FnMut(usize) -> bool,
    ) -> bool {
        self.inner.try_reserve_empty(groups, admit)
    }

    fn size(&self) -> usize {
        self.inner.size()
            + size_of::<Self>()
            + self.cache.allocation_size()
            + self.front.capacity() * size_of::<(u128, usize)>()
    }
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        self.reset();
        self.inner.emit(emit_to)
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
        self.reset();
        self.cache.shrink_to(0, |_| 0);
        self.front.shrink_to_fit();
        self.inner.shrink_to_fit();
    }
    fn clear_shrink(&mut self, num_rows: usize) {
        self.reset();
        self.cache.shrink_to(0, |_| 0);
        self.front.shrink_to_fit();
        self.inner.clear_shrink(num_rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregates::group_values::multi_group_by::GroupValuesColumn;
    use crate::aggregates::group_values::single_group_by::bytes_view::GroupValuesBytesView;
    use arrow::array::StringViewArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_physical_expr::binary_map::OutputType;
    use std::sync::Arc;

    fn base<const N: usize>() -> Box<dyn GroupValues> {
        if N == 1 {
            Box::new(GroupValuesBytesView::new(OutputType::Utf8View))
        } else {
            Box::new(
                GroupValuesColumn::<false>::try_new(Arc::new(Schema::new(
                    (0..N)
                        .map(|i| Field::new(format!("k{i}"), DataType::Utf8View, true))
                        .collect::<Vec<_>>(),
                )))
                .unwrap(),
            )
        }
    }

    fn compare_emit(a: &mut dyn GroupValues, b: &mut dyn GroupValues, emit: EmitTo) {
        for (a, b) in a.emit(emit).unwrap().iter().zip(b.emit(emit).unwrap()) {
            assert_eq!(
                a.as_string_view().iter().collect::<Vec<_>>(),
                b.as_string_view().iter().collect::<Vec<_>>()
            );
        }
    }

    fn mixed<const N: usize>() {
        let mut candidate = InlineCache::<N>::new(base::<N>());
        let mut baseline = base::<N>();
        let mut seed = 31_u64;
        for batch in 0..16 {
            let columns: Vec<ArrayRef> = (0..N)
                .map(|_| {
                    Arc::new(StringViewArray::from_iter((0..1024).map(|row| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        if row % 13 == 0 {
                            None
                        } else if batch == 4 && row % 17 == 0 {
                            Some(format!("a-long-out-of-line-key-{}", seed % 43))
                        } else {
                            Some(format!("k{}", seed % 43))
                        }
                    }))) as ArrayRef
                })
                .collect();
            let (mut a, mut b) = (Vec::new(), Vec::new());
            candidate.intern(&columns, &mut a).unwrap();
            baseline.intern(&columns, &mut b).unwrap();
            assert_eq!(a, b, "batch {batch}");
            assert_eq!(candidate.len(), baseline.len());
            if batch == 5 {
                compare_emit(&mut candidate, baseline.as_mut(), EmitTo::First(17));
            }
        }
        compare_emit(&mut candidate, baseline.as_mut(), EmitTo::All);
        candidate.clear_shrink(0);
        baseline.clear_shrink(0);
        let values =
            vec![Arc::new(StringViewArray::from(vec!["x", "y", "x"])) as ArrayRef; N];
        let (mut a, mut b) = (Vec::new(), Vec::new());
        candidate.intern(&values, &mut a).unwrap();
        baseline.intern(&values, &mut b).unwrap();
        assert_eq!(a, b);
    }
    #[test]
    fn inline_cache_single_keys_nulls_long_keys_and_emission() {
        mixed::<1>();
    }
    #[test]
    fn inline_cache_pair_keys_nulls_long_keys_and_emission() {
        mixed::<2>();
    }
    #[test]
    fn direct_code_collisions_keep_complete_keys_and_shifted_group_ids() -> Result<()> {
        use super::super::batch_codes::{BatchCodes, SLOTS};
        let mut seen = std::collections::HashMap::new();
        let (left, right) = (0..SLOTS * 2)
            .find_map(|id| {
                let text = format!("c{id}");
                let array = StringViewArray::from(vec![text.as_str()]);
                seen.insert(BatchCodes::slot(array.views()[0]), text.clone())
                    .map(|prior| (prior, text))
            })
            .unwrap();
        let mut cache = InlineCache::<1>::new(base::<1>());
        let mut reference = base::<1>();
        for batch in 0..8 {
            let array =
                Arc::new(StringViewArray::from_iter((0..8192).map(|row| {
                    match (row / 128 + batch) % 3 {
                        0 => Some(left.as_str()),
                        1 => Some(right.as_str()),
                        _ => None,
                    }
                }))) as ArrayRef;
            let (mut actual, mut expected) = (Vec::new(), Vec::new());
            cache.intern(std::slice::from_ref(&array), &mut actual)?;
            reference.intern(&[array], &mut expected)?;
            assert_eq!(actual, expected, "batch {batch}");
            if batch == 2 || batch == 5 {
                compare_emit(
                    &mut cache,
                    reference.as_mut(),
                    if batch == 2 {
                        EmitTo::First(1)
                    } else {
                        EmitTo::All
                    },
                );
            }
        }
        compare_emit(&mut cache, reference.as_mut(), EmitTo::All);
        Ok(())
    }
    #[test]
    fn direct_front_budget_preserves_the_exact_key_cache() -> Result<()> {
        let array = Arc::new(StringViewArray::from_iter_values(
            (0..=SLOTS / 2).map(|id| format!("k{id}")),
        )) as ArrayRef;
        let mut cache = InlineCache::<1>::new(base::<1>());
        let (mut first, mut second) = (Vec::new(), Vec::new());
        cache.intern(std::slice::from_ref(&array), &mut first)?;
        assert!(cache.front.is_empty());
        assert_eq!(cache.front.capacity(), 0);
        assert_eq!(cache.cache.len(), SLOTS / 2 + 1);
        cache.intern(&[array], &mut second)?;
        assert_eq!(first, second);
        Ok(())
    }
    #[test]
    fn inline_cache_stops_at_its_cardinality_budget() {
        let mut cache = InlineCache::<1>::new(base::<1>());
        let a = Arc::new(StringViewArray::from_iter_values(
            (0..=MAX_GROUPS).map(|i| format!("k{i}")),
        )) as ArrayRef;
        cache.intern(&[a], &mut Vec::new()).unwrap();
        assert!(cache.disabled);
        assert_eq!(cache.cache.allocation_size(), 0);
        cache.emit(EmitTo::All).unwrap();
        assert!(!cache.disabled);
        let a = Arc::new(StringViewArray::from(vec!["x", "x"])) as ArrayRef;
        cache.intern(&[a], &mut Vec::new()).unwrap();
        assert_eq!(cache.cache.len(), 1);
    }

    #[test]
    fn inline_cache_accepts_unused_dictionary_buffers() {
        use arrow::buffer::Buffer;
        let values = StringViewArray::from(vec!["a", "b", "a"]);
        let array = Arc::new(
            StringViewArray::try_new(
                values.views().clone(),
                vec![Buffer::from_vec(vec![0_u8; 128])],
                None,
            )
            .unwrap(),
        ) as ArrayRef;
        let mut cache = InlineCache::<1>::new(base::<1>());
        cache.intern(&[array], &mut Vec::new()).unwrap();
        assert_eq!(cache.cache.len(), 2);
    }

    #[test]
    fn buffered_short_keys_and_masked_long_views_keep_complete_key_identity() {
        use arrow::buffer::{Buffer, NullBuffer};
        fn check<const N: usize>() {
            let mut cache = InlineCache::<N>::new(base::<N>());
            let mut reference = base::<N>();
            let short = StringViewArray::from(vec![None, Some("x"), Some("x")]);
            let buffered = StringViewArray::try_new(
                short.views().clone(),
                vec![Buffer::from_vec(vec![0_u8; 128])],
                short.nulls().cloned(),
            )
            .unwrap();
            let long_a = StringViewArray::from(vec!["shared-long-prefix-A", "x", "x"]);
            let long_b = StringViewArray::from(vec!["shared-long-prefix-B", "x", "x"]);
            assert_eq!(long_a.views()[0], long_b.views()[0]);
            let masked = StringViewArray::try_new(
                long_a.views().clone(),
                long_a.data_buffers().to_vec(),
                Some(NullBuffer::from(vec![false, true, true])),
            )
            .unwrap();
            for array in [buffered, masked, long_a, long_b, short] {
                let array = Arc::new(array) as ArrayRef;
                let cols = vec![array; N];
                let (mut actual, mut expected) = (Vec::new(), Vec::new());
                cache.intern(&cols, &mut actual).unwrap();
                reference.intern(&cols, &mut expected).unwrap();
                assert_eq!(actual, expected);
            }
            compare_emit(&mut cache, reference.as_mut(), EmitTo::All);
        }
        check::<1>();
        check::<2>();
    }
}
