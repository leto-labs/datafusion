// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! This file contains the implementation of the `JoinHashMap` struct, which
//! is used to store the mapping between hash values based on the build side
//! ["on" values] to a list of indices with this key's value.

use std::alloc::Layout;
use std::fmt::{self, Debug};
use std::ops::Sub;

use arrow::array::ArrayRef;
use arrow::array::BooleanArray;
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::ArrowNativeType;
use datafusion_common::{Result, internal_datafusion_err};
use hashbrown::HashTable;

use crate::joins::inline_keys::{EncodedKeys, InlineKeys};
use hashbrown::hash_table::Entry::{Occupied, Vacant};

/// Resume point of a chunked lookup, see
/// [`JoinHashMapType::get_matched_indices_with_limit_offset`].
///
/// A lookup emits at most `limit` matches. If the limit is reached before
/// the probe batch is exhausted, it returns the position to resume from;
/// once the batch is exhausted it returns `None`.
///
/// Each position has a single encoding: once a probe row's chain has been
/// fully traversed, the next offset is the following row, `AtRow { row: row + 1 }`.
/// Given `limit > 0`, a returned offset is therefore never `AtRow { row: 0 }`,
/// which marks the first lookup of a probe batch.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ProbeOffset {
    /// Resume at probe row `row`, from the head of its chain.
    AtRow { row: usize },
    /// Resume in the middle of probe row `row`'s chain, where `next` is the
    /// 1-based index of the next build row in the chain. `next` is never `0`,
    /// the end-of-chain marker.
    MidChain { row: usize, next: u64 },
}

/// Maps a `u64` hash value based on the build side ["on" values] to a list of indices with this key's value.
///
/// By allocating a `HashMap` with capacity for *at least* the number of rows for entries at the build side,
/// we make sure that we don't have to re-hash the hashmap, which needs access to the key (the hash in this case) value.
///
/// E.g. 1 -> [3, 6, 8] indicates that the column values map to rows 3, 6 and 8 for hash value 1
/// As the key is a hash value, we need to check possible hash collisions in the probe stage
/// During this stage it might be the case that a row is contained the same hashmap value,
/// but the values don't match. Those are checked in the `equal_rows_arr` method.
///
/// The indices (values) are stored in a separate chained list stored as `Vec<u32>` or `Vec<u64>`.
///
/// The first value (+1) is stored in the hashmap, whereas the next value is stored in array at the position value.
///
/// The chain can be followed until the value "0" has been reached, meaning the end of the list.
/// Also see chapter 5.3 of [Balancing vectorized query execution with bandwidth-optimized storage](https://dare.uva.nl/search?identifier=5ccbb60a-38b8-4eeb-858a-e7735dd37487)
///
/// # Example
///
/// ``` text
/// See the example below:
///
/// Insert (10,1)            <-- insert hash value 10 with row index 1
/// map:
/// ----------
/// | 10 | 2 |
/// ----------
/// next:
/// ---------------------
/// | 0 | 0 | 0 | 0 | 0 |
/// ---------------------
/// Insert (20,2)
/// map:
/// ----------
/// | 10 | 2 |
/// | 20 | 3 |
/// ----------
/// next:
/// ---------------------
/// | 0 | 0 | 0 | 0 | 0 |
/// ---------------------
/// Insert (10,3)           <-- collision! row index 3 has a hash value of 10 as well
/// map:
/// ----------
/// | 10 | 4 |
/// | 20 | 3 |
/// ----------
/// next:
/// ---------------------
/// | 0 | 0 | 0 | 2 | 0 |  <--- hash value 10 maps to 4,2 (which means indices values 3,1)
/// ---------------------
/// Insert (10,4)          <-- another collision! row index 4 ALSO has a hash value of 10
/// map:
/// ---------
/// | 10 | 5 |
/// | 20 | 3 |
/// ---------
/// next:
/// ---------------------
/// | 0 | 0 | 0 | 2 | 4 | <--- hash value 10 maps to 5,4,2 (which means indices values 4,3,1)
/// ---------------------
/// ```
///
/// Here we have an option between creating a `JoinHashMapType` using `u32` or `u64` indices
/// based on how many rows were being used for indices.
///
/// At runtime we choose between using `JoinHashMapU32` and `JoinHashMapU64` which oth implement
/// `JoinHashMapType`.
///
/// For example, we may compare references to `JoinHashMapType` implementations by pointer equality
/// rather than deep equality of contents, as deep equality would be expensive and in our usage
/// patterns it is impossible for two different hash maps to have identical contents in a practical sense.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
///
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub trait JoinHashMapType: Send + Sync {
    fn extend_zero(&mut self, len: usize);

    /// Longbow patch (JT): fallible. `JoinHashMapU32` and `JoinHashMapU64`
    /// have one chain entry per row and a fixed number of slots; they return
    /// an error instead of panicking when given a row beyond the chain array
    /// or more distinct hashes than their slots hold.
    fn update_from_iter<'a>(
        &mut self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + Send + 'a>,
        deleted_offset: usize,
    ) -> Result<()>;

    fn get_matched_indices<'a>(
        &self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + 'a>,
        deleted_offset: Option<usize>,
    ) -> (Vec<u32>, Vec<u64>);

    /// Probe rows marked NULL in `valid_keys` are skipped without a lookup:
    /// their key contains a NULL, which cannot match any build row under
    /// `NullEquality::NullEqualsNothing`. Pass `None` when every probe key is
    /// matchable.
    fn get_matched_indices_with_limit_offset(
        &self,
        hash_values: &[u64],
        valid_keys: Option<&NullBuffer>,
        limit: usize,
        offset: ProbeOffset,
        input_indices: &mut Vec<u32>,
        match_indices: &mut Vec<u64>,
    ) -> Option<ProbeOffset>;

    /// Returns a BooleanArray indicating which of the provided hashes exist in the map.
    fn contain_hashes(&self, hash_values: &[u64]) -> BooleanArray;

    /// Returns `true` if the join hash map contains no entries.
    fn is_empty(&self) -> bool;

    /// Returns the number of entries in the join hash map.
    fn len(&self) -> usize;

    /// Longbow patch (JT 9): the encoding of the inline keys that this map
    /// stores beside its chain heads, or `None` for a map without them.
    fn inline_keys(&self) -> Option<&InlineKeys> {
        None
    }

    /// Longbow patch (JT 9): [`Self::get_matched_indices_with_limit_offset`]
    /// followed by the key comparison, for a map with
    /// [`Self::inline_keys`]: emits only pairs with equal keys, in the same
    /// order and with the same offsets. `probe` holds the probe rows' keys
    /// in this map's encoding; its invalid rows match nothing. `build_keys`
    /// are the build key columns (read only for chains that mix keys).
    #[expect(clippy::too_many_arguments)]
    fn get_inline_matched_indices_with_limit_offset(
        &self,
        _hash_values: &[u64],
        _probe: &EncodedKeys,
        _build_keys: &[ArrayRef],
        _limit: usize,
        _offset: ProbeOffset,
        _input_indices: &mut Vec<u32>,
        _match_indices: &mut Vec<u64>,
    ) -> Result<Option<ProbeOffset>> {
        Err(internal_datafusion_err!(
            "inline-key probe of a join map without inline keys"
        ))
    }
}

// Longbow patch (JT): the `u32` and `u64` maps keep their chain encoding
// (`next`, row + 1, 0 ends a chain) but store the chain heads in a
// [`HeadTable`] instead of a `hashbrown::HashTable`, so that the build and
// the probe can prefetch the slot of a row several rows ahead.
pub struct JoinHashMapU32 {
    // Stores hash value to last row index
    table: HeadTable<u32>,
    // Stores indices in chained list data structure
    next: ZeroedArray<u32>,
    // Whether any hash value maps to more than one row
    chained: bool,
}

impl JoinHashMapU32 {
    /// Longbow patch (JT): a map with the heads of `map`, as `map.find`
    /// returns them: tests build maps whose table holds one hash twice.
    #[cfg(test)]
    pub(crate) fn new(map: HashTable<(u64, u32)>, next: Vec<u32>) -> Self {
        let chained = map.len() != next.len();
        Self {
            table: HeadTable::from_hashbrown(&map),
            next: next.into(),
            chained,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            table: HeadTable::with_capacity(cap),
            next: zeroed_chain(cap),
            chained: false,
        }
    }

    /// Bytes of the map, its head slots and one chain entry per build row.
    pub(crate) fn reservation_bytes(capacity: usize) -> Result<usize> {
        HeadTable::<u32>::checked_bytes_for(capacity)
            .and_then(|heads| {
                let chain = capacity.checked_mul(size_of::<u32>())?;
                heads.checked_add(chain)?.checked_add(size_of::<Self>())
            })
            .ok_or_else(|| {
                datafusion_common::exec_datafusion_err!("Hash join table size overflow")
            })
    }

    #[cfg(test)]
    pub(crate) fn allocated_bytes(&self) -> usize {
        size_of::<Self>() + self.table.bytes() + size_of_val(&*self.next)
    }
}

impl Debug for JoinHashMapU32 {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

impl JoinHashMapType for JoinHashMapU32 {
    fn extend_zero(&mut self, _: usize) {}

    fn update_from_iter<'a>(
        &mut self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + Send + 'a>,
        deleted_offset: usize,
    ) -> Result<()> {
        let next = &mut self.next;
        let chained = &mut self.chained;
        let row_bound = next.len() + deleted_offset;
        self.table.insert_all(
            iter.map(|(row, hash)| (row, *hash, 0)),
            row_bound,
            |row, prev| {
                if !*chained {
                    next.back_chain();
                }
                next[row - deleted_offset] = prev;
                *chained = true;
            },
        )
    }

    fn get_matched_indices<'a>(
        &self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + 'a>,
        deleted_offset: Option<usize>,
    ) -> (Vec<u32>, Vec<u64>) {
        heads_matched_indices(&self.table, &self.next, iter, deleted_offset)
    }

    fn get_matched_indices_with_limit_offset(
        &self,
        hash_values: &[u64],
        valid_keys: Option<&NullBuffer>,
        limit: usize,
        offset: ProbeOffset,
        input_indices: &mut Vec<u32>,
        match_indices: &mut Vec<u64>,
    ) -> Option<ProbeOffset> {
        heads_matched_indices_with_limit_offset(
            &self.table,
            &self.next,
            !self.chained,
            hash_values,
            valid_keys,
            limit,
            offset,
            input_indices,
            match_indices,
        )
    }

    fn contain_hashes(&self, hash_values: &[u64]) -> BooleanArray {
        heads_contain_hashes(&self.table, hash_values)
    }

    fn is_empty(&self) -> bool {
        self.table.len() == 0
    }

    fn len(&self) -> usize {
        self.table.len()
    }
}

/// Longbow patch (JT 9): [`JoinHashMapU32`] with inline keys: each slot of
/// its head table also holds the encoded key of the chain's head row
/// ([`PackedU64`], in the bytes of a `u32` slot), so a probe reads no build
/// key column. Built only from
/// whole rows with their keys ([`PartitionedBuild::update_partitioned`] or
/// the shared build); [`JoinHashMapType::update_from_iter`], which has only
/// hashes, is refused.
pub struct JoinHashMapInline {
    table: HeadTable<PackedU64>,
    next: ZeroedArray<u32>,
    chained: bool,
    keys: InlineKeys,
}

impl JoinHashMapInline {
    /// The slot layout of a map for `cap` rows with `keys`, or `None` when
    /// the keys do not fit its slots beside the heads and the tag.
    pub fn layout_for(cap: usize, keys: &InlineKeys) -> Option<SlotLayout> {
        SlotLayout::packed(cap, keys.bits())
    }

    /// A map for `cap` rows with `keys`, whose slots must fit them
    /// ([`Self::layout_for`]).
    pub fn with_capacity(cap: usize, keys: InlineKeys) -> Self {
        let layout = Self::layout_for(cap, &keys).expect("inline keys fit the slots");
        Self {
            table: HeadTable::with_layout(cap, layout),
            next: zeroed_chain(cap),
            chained: false,
            keys,
        }
    }

    /// Bytes of the head table of a map made for `cap` rows (the chain
    /// array is not included, as in `estimate_memory_size`): those of a
    /// `u32` map.
    pub fn table_bytes(cap: usize) -> usize {
        HeadTable::<PackedU64>::bytes_for(cap)
    }
}

impl Debug for JoinHashMapInline {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

impl JoinHashMapType for JoinHashMapInline {
    fn extend_zero(&mut self, _: usize) {}

    fn update_from_iter<'a>(
        &mut self,
        _iter: Box<dyn Iterator<Item = (usize, &'a u64)> + Send + 'a>,
        _deleted_offset: usize,
    ) -> Result<()> {
        Err(internal_datafusion_err!(
            "a join map with inline keys cannot insert rows without their keys"
        ))
    }

    fn get_matched_indices<'a>(
        &self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + 'a>,
        deleted_offset: Option<usize>,
    ) -> (Vec<u32>, Vec<u64>) {
        heads_matched_indices(&self.table, &self.next, iter, deleted_offset)
    }

    fn get_matched_indices_with_limit_offset(
        &self,
        hash_values: &[u64],
        valid_keys: Option<&NullBuffer>,
        limit: usize,
        offset: ProbeOffset,
        input_indices: &mut Vec<u32>,
        match_indices: &mut Vec<u64>,
    ) -> Option<ProbeOffset> {
        heads_matched_indices_with_limit_offset(
            &self.table,
            &self.next,
            !self.chained,
            hash_values,
            valid_keys,
            limit,
            offset,
            input_indices,
            match_indices,
        )
    }

    fn contain_hashes(&self, hash_values: &[u64]) -> BooleanArray {
        heads_contain_hashes(&self.table, hash_values)
    }

    fn is_empty(&self) -> bool {
        self.table.len() == 0
    }

    fn len(&self) -> usize {
        self.table.len()
    }

    fn inline_keys(&self) -> Option<&InlineKeys> {
        Some(&self.keys)
    }

    fn get_inline_matched_indices_with_limit_offset(
        &self,
        hash_values: &[u64],
        probe: &EncodedKeys,
        build_keys: &[ArrayRef],
        limit: usize,
        offset: ProbeOffset,
        input_indices: &mut Vec<u32>,
        match_indices: &mut Vec<u64>,
    ) -> Result<Option<ProbeOffset>> {
        Ok(inline_matched_indices_with_limit_offset(
            &self.table,
            &self.next,
            !self.chained,
            hash_values,
            probe,
            &self.keys,
            build_keys,
            limit,
            offset,
            input_indices,
            match_indices,
        ))
    }
}

pub struct JoinHashMapU64 {
    // Stores hash value to last row index
    table: HeadTable<u64>,
    // Stores indices in chained list data structure
    next: ZeroedArray<u64>,
    // Whether any hash value maps to more than one row
    chained: bool,
}

impl JoinHashMapU64 {
    /// Longbow patch (JT): see [`JoinHashMapU32::new`].
    #[cfg(test)]
    pub(crate) fn new(map: HashTable<(u64, u64)>, next: Vec<u64>) -> Self {
        let chained = map.len() != next.len();
        Self {
            table: HeadTable::from_hashbrown(&map),
            next: next.into(),
            chained,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            table: HeadTable::with_capacity(cap),
            next: zeroed_chain(cap),
            chained: false,
        }
    }

    /// Longbow patch (JT): bytes of the head table of a map made for `cap`
    /// rows (the chain array is not included, as in `estimate_memory_size`).
    pub fn table_bytes(cap: usize) -> usize {
        HeadTable::<u64>::bytes_for(cap)
    }
}

impl Debug for JoinHashMapU64 {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

impl JoinHashMapType for JoinHashMapU64 {
    fn extend_zero(&mut self, _: usize) {}

    fn update_from_iter<'a>(
        &mut self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + Send + 'a>,
        deleted_offset: usize,
    ) -> Result<()> {
        let next = &mut self.next;
        let chained = &mut self.chained;
        let row_bound = next.len() + deleted_offset;
        self.table.insert_all(
            iter.map(|(row, hash)| (row, *hash, 0)),
            row_bound,
            |row, prev| {
                if !*chained {
                    next.back_chain();
                }
                next[row - deleted_offset] = prev;
                *chained = true;
            },
        )
    }

    fn get_matched_indices<'a>(
        &self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + 'a>,
        deleted_offset: Option<usize>,
    ) -> (Vec<u32>, Vec<u64>) {
        heads_matched_indices(&self.table, &self.next, iter, deleted_offset)
    }

    fn get_matched_indices_with_limit_offset(
        &self,
        hash_values: &[u64],
        valid_keys: Option<&NullBuffer>,
        limit: usize,
        offset: ProbeOffset,
        input_indices: &mut Vec<u32>,
        match_indices: &mut Vec<u64>,
    ) -> Option<ProbeOffset> {
        heads_matched_indices_with_limit_offset(
            &self.table,
            &self.next,
            !self.chained,
            hash_values,
            valid_keys,
            limit,
            offset,
            input_indices,
            match_indices,
        )
    }

    fn contain_hashes(&self, hash_values: &[u64]) -> BooleanArray {
        heads_contain_hashes(&self.table, hash_values)
    }

    fn is_empty(&self) -> bool {
        self.table.len() == 0
    }

    fn len(&self) -> usize {
        self.table.len()
    }
}

use crate::joins::chain::traverse_chain;
use crate::joins::join_hash_map::ProbeOffset::{AtRow, MidChain};

/// Inserts the rows of `iter` into `map` and `next`. Returns `true` if any row
/// was chained onto an existing hash value.
pub fn update_from_iter<'a, T>(
    map: &mut HashTable<(u64, T)>,
    next: &mut [T],
    iter: Box<dyn Iterator<Item = (usize, &'a u64)> + Send + 'a>,
    deleted_offset: usize,
) -> bool
where
    T: Copy + TryFrom<usize> + PartialOrd,
    <T as TryFrom<usize>>::Error: Debug,
{
    let mut chained = false;
    for (row, &hash_value) in iter {
        let entry = map.entry(
            hash_value,
            |&(hash, _)| hash_value == hash,
            |&(hash, _)| hash,
        );

        match entry {
            Occupied(mut occupied_entry) => {
                // Already exists: add index to next array
                let (_, index) = occupied_entry.get_mut();
                let prev_index = *index;
                // Store new value inside hashmap
                *index = T::try_from(row + 1).unwrap();
                // Update chained Vec at `row` with previous value
                next[row - deleted_offset] = prev_index;
                chained = true;
            }
            Vacant(vacant_entry) => {
                vacant_entry.insert((hash_value, T::try_from(row + 1).unwrap()));
            }
        }
    }
    chained
}

pub fn get_matched_indices<'a, T>(
    map: &HashTable<(u64, T)>,
    next: &[T],
    iter: Box<dyn Iterator<Item = (usize, &'a u64)> + 'a>,
    deleted_offset: Option<usize>,
) -> (Vec<u32>, Vec<u64>)
where
    T: Copy + TryFrom<usize> + PartialOrd + Into<u64> + Sub<Output = T>,
    <T as TryFrom<usize>>::Error: Debug,
{
    let mut input_indices = vec![];
    let mut match_indices = vec![];
    let zero = T::try_from(0).unwrap();
    let one = T::try_from(1).unwrap();

    for (row_idx, hash_value) in iter {
        // Get the hash and find it in the index
        if let Some((_, index)) = map.find(*hash_value, |(hash, _)| *hash_value == *hash)
        {
            let mut i = *index - one;
            loop {
                let match_row_idx = if let Some(offset) = deleted_offset {
                    let offset = T::try_from(offset).unwrap();
                    // This arguments means that we prune the next index way before here.
                    if i < offset {
                        // End of the list due to pruning
                        break;
                    }
                    i - offset
                } else {
                    i
                };
                match_indices.push(match_row_idx.into());
                input_indices.push(row_idx as u32);
                // Follow the chain to get the next index value
                let next_chain = next[match_row_idx.into() as usize];
                if next_chain == zero {
                    // end of list
                    break;
                }
                i = next_chain - one;
            }
        }
    }

    (input_indices, match_indices)
}

/// `unique` must be `true` only if every hash value in `map` maps to a single
/// row; the chain in `next_chain` is then not traversed.
#[expect(clippy::too_many_arguments)]
pub fn get_matched_indices_with_limit_offset<T>(
    map: &HashTable<(u64, T)>,
    next_chain: &[T],
    unique: bool,
    hash_values: &[u64],
    valid_keys: Option<&NullBuffer>,
    limit: usize,
    offset: ProbeOffset,
    input_indices: &mut Vec<u32>,
    match_indices: &mut Vec<u64>,
) -> Option<ProbeOffset>
where
    T: Copy + TryFrom<usize> + PartialOrd + Into<u64> + Sub<Output = T>,
    <T as TryFrom<usize>>::Error: Debug,
    T: ArrowNativeType,
{
    // Clear the buffer before producing new results
    input_indices.clear();
    match_indices.clear();
    let one = T::try_from(1).unwrap();

    // Check if hashmap consists of unique values
    // If so, we can skip the chain traversal
    if unique {
        debug_assert!(
            matches!(offset, AtRow { .. }),
            "unique keys cannot have mid-chain"
        );
        let start = match offset {
            AtRow { row } => row,
            MidChain { row, .. } => row,
        };
        let end = (start + limit).min(hash_values.len());
        for (i, &hash) in hash_values[start..end].iter().enumerate() {
            // NULL keys cannot match any build row
            if valid_keys.is_some_and(|valid| valid.is_null(start + i)) {
                continue;
            }
            if let Some((_, idx)) = map.find(hash, |(h, _)| hash == *h) {
                input_indices.push(start as u32 + i as u32);
                match_indices.push((*idx - one).into());
            }
        }
        return if end == hash_values.len() {
            None
        } else {
            Some(AtRow { row: end })
        };
    }

    let mut remaining_output = limit;

    // Calculate initial `hash_values` index before iterating
    let to_skip = match offset {
        // `AtRow` indicates that `idx` processing hasn't been started
        AtRow { row: idx } => idx,
        // Otherwise, process remaining `idx` matches by traversing `next_chain`,
        // to start with the next index
        MidChain {
            row: idx,
            next: next_idx,
        } => {
            let next_idx: T = T::usize_as(next_idx as usize);
            let is_last = idx == hash_values.len() - 1;
            if let Some(next_offset) = traverse_chain(
                next_chain,
                idx,
                next_idx,
                &mut remaining_output,
                input_indices,
                match_indices,
                is_last,
            ) {
                return Some(next_offset);
            }
            idx + 1
        }
    };

    let hash_values_len = hash_values.len();
    for (i, &hash) in hash_values[to_skip..].iter().enumerate() {
        let row_idx = to_skip + i;
        // NULL keys cannot match any build row
        if valid_keys.is_some_and(|valid| valid.is_null(row_idx)) {
            continue;
        }
        if let Some((_, idx)) = map.find(hash, |(h, _)| hash == *h) {
            let idx: T = *idx;
            let is_last = row_idx == hash_values_len - 1;
            if let Some(next_offset) = traverse_chain(
                next_chain,
                row_idx,
                idx,
                &mut remaining_output,
                input_indices,
                match_indices,
                is_last,
            ) {
                return Some(next_offset);
            }
        }
    }
    None
}

pub fn contain_hashes<T>(map: &HashTable<(u64, T)>, hash_values: &[u64]) -> BooleanArray {
    let buffer = BooleanBuffer::collect_bool(hash_values.len(), |i| {
        let hash = hash_values[i];
        map.find(hash, |(h, _)| hash == *h).is_some()
    });
    BooleanArray::new(buffer, None)
}

// ---------------------------------------------------------------------------
// Longbow patch (JT): the chain-head table of `JoinHashMapU32`,
// `JoinHashMapU64` and the shared build's shards.
//
// One power-of-two array of slots with linear probing at a load factor of at
// most one half. A `u32` slot is one 64-bit word: the low 32 bits of the
// row's hash above the chain head (row + 1, 0 = empty). A `u64` slot is two
// words: the whole hash, then the head. The slot of a hash is the top bits
// of `hash * MUL`, so every hash bit moves the slot, including the bits that
// pick a shard of the shared build.
//
// A `u32` table compares 32 hash bits, so two different hashes with the same
// low 32 bits that probe the same slots share one chain. That only adds
// candidates: every caller compares the keys of the candidate rows
// (`equal_rows_arr`) or uses the answer as a filter that may pass rows that
// do not match (`contain_hashes`). With 2^32 tags and runs of one or two
// slots, it happens for about one probe in 2^31.
//
// Adversarial keys: a chain is walked row by row, so keys that share one
// chain make a probe cost linear in their rows (quadratic over a probe of
// them). With hashbrown, crafting such keys meant colliding all 64 hash bits;
// a `u32` table needs only the low 32 bits and the slot, about 2^32 work per
// key. Longbow is single-tenant (no untrusted query author shares a worker),
// so this is recorded, not mitigated.
//
// Memory: `slots = max(2, next_power_of_two(2 * capacity))`, 8 bytes each for
// `u32` and 16 for `u64`. The `u32` build charges its slot array and chain
// directly ([`JoinHashMapU32::reservation_bytes`]). The `u64` build reserves
// the larger of its slot array ([`JoinHashMapU64::table_bytes`]) and the
// hashbrown estimate.
//
// Above `PREFETCH_BYTES` the probe prefetches the slot of the row
// `PREFETCH_DISTANCE` rows ahead, and the build keeps that many rows in
// flight, so their cache misses overlap instead of following each other.
// The slots and the chain from `HUGE_PAGE_BYTES` on ask the kernel for 2 MiB
// pages (`MADV_HUGEPAGE`; the host's THP mode is `madvise`), which removes
// most of the TLB misses of a random probe; `LONGBOW_JOIN_TABLE_HUGE_PAGES=off`
// turns that off. Such an array has its own anonymous mapping
// ([`ZeroedArray`]), unmapped when the table is dropped, so the advice ends
// with the table. Advice on allocator memory outlived the table: the
// allocator reused the freed range for other operators, and their faults
// there allocated huge pages with synchronous compaction in the query thread
// (THP defrag `madvise`; TPC-H SF100 on worker-01, 48 threads: q10 pull 27%
// slower in a suite process, while Q9 pull was 17% faster). By default
// (JT 10) the mapping is backed when it is made, and only while the kernel
// has free huge pages; see `huge_page_policy`.
// ---------------------------------------------------------------------------

/// The multiplier of the slot position (2^64 / golden ratio, odd).
const SLOT_MUL: u64 = 0x9E37_79B9_7F4A_7C15;

/// Tables above this size prefetch (Zen 4 L2 is 1 MiB per core).
const PREFETCH_BYTES: usize = 512 << 10;

/// Whether a table of `bytes` is large enough for prefetches to pay.
pub(crate) fn prefetch_pays(bytes: usize) -> bool {
    bytes > PREFETCH_BYTES
}

/// Rows between a slot's prefetch and its use. 16 misses in flight cover
/// one DRAM latency (about 110 ns) at about 7 ns per row.
const PREFETCH_DISTANCE: usize = 16;

/// Arrays from this size get their own mapping advised to use transparent
/// huge pages, unless the mode is off ([`huge_page_mode`]).
const HUGE_PAGE_BYTES: usize = 4 << 20;

pub use crate::joins::huge_page_policy::{HUGE_PAGES_ENV, HugePageMode, huge_page_mode};
use crate::joins::huge_page_policy::{STALL_GATE, StallGate, Ticket};

/// Longbow patch (JT 9): the bit fields of a [`PackedU64`] slot, chosen per
/// table: `head_bits` for the largest head (row + 1) and `key_bits` for the
/// inline key; the tag takes the rest. Tables without inline keys ignore it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotLayout {
    head_bits: u32,
    key_bits: u32,
    /// Precomputed: `head_bits + 1 + key_bits`, and the masks of the head,
    /// the key and the tag.
    tag_shift: u32,
    head_mask: u64,
    key_mask: u64,
    tag_mask: u64,
}

/// The fewest hash bits that a [`PackedU64`] slot keeps as its tag. Two
/// different keys of one probe run with equal tags share a chain, which is
/// then marked mixed; at a load of at most one half that is about one insert
/// in 2^11 at this width.
pub const MIN_TAG_BITS: u32 = 10;

impl SlotLayout {
    /// The layout of a table whose heads are at most `rows` with inline
    /// keys of `key_bits` bits, or `None` when the tag would keep fewer than
    /// [`MIN_TAG_BITS`] bits (or `rows` does not fit a `u32` chain).
    pub fn packed(rows: usize, key_bits: u32) -> Option<Self> {
        if rows >= u32::MAX as usize || key_bits > 64 {
            return None;
        }
        let head_bits = (u64::BITS - (rows as u64).leading_zeros()).max(1);
        let tag_shift = head_bits + 1 + key_bits;
        // Review fix (leto-mjj.71 review): `then`, not `then_some`, so the
        // tag's width is computed only when the fields fit.
        (tag_shift + MIN_TAG_BITS <= u64::BITS).then(|| Self {
            head_bits,
            key_bits,
            tag_shift,
            head_mask: low_bits(head_bits),
            key_mask: low_bits(key_bits),
            tag_mask: low_bits(u64::BITS - tag_shift),
        })
    }

    #[inline(always)]
    fn tag_shift(self) -> u32 {
        self.tag_shift
    }

    #[cfg(test)]
    fn tag_bits(self) -> u32 {
        u64::BITS - self.tag_shift
    }

    /// The bits of the inline key.
    pub fn key_bits(self) -> u32 {
        self.key_bits
    }
}

#[inline(always)]
fn low_bits(bits: u32) -> u64 {
    if bits >= u64::BITS {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
}

/// The width of the heads of a [`HeadTable`].
pub(crate) trait HeadWord: Copy + Send + Sync + 'static {
    /// 64-bit words per slot.
    const WORDS: usize;
    /// Longbow patch (JT 9): whether a slot also keeps the inline key of
    /// its head row ([`PackedU64`]).
    const INLINE: bool = false;
    /// The type of the chain array's entries.
    type Chain: Copy;
    /// The hash bits that a slot stores.
    fn tag(layout: SlotLayout, hash: u64) -> u64;
    /// The tag and the head of a slot; head 0 is an empty slot.
    fn read(layout: SlotLayout, words: &[u64], slot: usize) -> (u64, u64);
    /// Writes a slot; `key` and `mixed` are kept only by an inline table.
    fn write_slot(
        layout: SlotLayout,
        words: &mut [u64],
        slot: usize,
        tag: u64,
        head: u64,
        key: u64,
        mixed: bool,
    );
    /// Converts a head (row + 1) of this width.
    fn of_head(head: u64) -> Self::Chain;
    fn head_of(layout: SlotLayout, row: usize) -> u64;
    /// The inline key of a slot's head and whether the slot's chain holds
    /// rows of different keys; `(0, false)` for a table without keys.
    #[inline(always)]
    fn read_key(_layout: SlotLayout, _words: &[u64], _slot: usize) -> (u64, bool) {
        (0, false)
    }
}

impl HeadWord for u32 {
    const WORDS: usize = 1;
    type Chain = u32;

    #[inline(always)]
    fn tag(_: SlotLayout, hash: u64) -> u64 {
        hash & 0xFFFF_FFFF
    }

    #[inline(always)]
    fn read(_: SlotLayout, words: &[u64], slot: usize) -> (u64, u64) {
        let word = words[slot];
        (word >> 32, word & 0xFFFF_FFFF)
    }

    #[inline(always)]
    fn write_slot(
        _: SlotLayout,
        words: &mut [u64],
        slot: usize,
        tag: u64,
        head: u64,
        _key: u64,
        _mixed: bool,
    ) {
        words[slot] = (tag << 32) | head;
    }

    #[inline(always)]
    fn of_head(head: u64) -> Self {
        head as u32
    }

    #[inline(always)]
    fn head_of(_: SlotLayout, row: usize) -> u64 {
        // As `T::try_from(row + 1).unwrap()` before: a `u32` map has fewer
        // than `u32::MAX` rows.
        u64::from(u32::try_from(row + 1).expect("u32 join map row"))
    }
}

impl HeadWord for u64 {
    const WORDS: usize = 2;
    type Chain = u64;

    #[inline(always)]
    fn tag(_: SlotLayout, hash: u64) -> u64 {
        hash
    }

    #[inline(always)]
    fn read(_: SlotLayout, words: &[u64], slot: usize) -> (u64, u64) {
        (words[2 * slot], words[2 * slot + 1])
    }

    #[inline(always)]
    fn write_slot(
        _: SlotLayout,
        words: &mut [u64],
        slot: usize,
        tag: u64,
        head: u64,
        _key: u64,
        _mixed: bool,
    ) {
        words[2 * slot] = tag;
        words[2 * slot + 1] = head;
    }

    #[inline(always)]
    fn of_head(head: u64) -> Self {
        head
    }

    #[inline(always)]
    fn head_of(_: SlotLayout, row: usize) -> u64 {
        row as u64 + 1
    }
}

/// Longbow patch (JT 9): a one-word slot with an inline key, in the bytes
/// and at the load of a `u32` slot. From the top: the low hash bits as the
/// tag, the inline key ([`crate::joins::inline_keys::InlineKeys`]) of the
/// head row, a bit that marks a chain holding rows of different keys, and
/// the head (row + 1), each as wide as the table's [`SlotLayout`] says.
/// All rows of a chain without the mark have the slot's key, so a probe
/// compares its key in the slot's own cache line and reads no build key.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedU64;

impl HeadWord for PackedU64 {
    const WORDS: usize = 1;
    const INLINE: bool = true;
    type Chain = u32;

    #[inline(always)]
    fn tag(layout: SlotLayout, hash: u64) -> u64 {
        hash & layout.tag_mask
    }

    #[inline(always)]
    fn read(layout: SlotLayout, words: &[u64], slot: usize) -> (u64, u64) {
        let word = words[slot];
        (word >> layout.tag_shift(), word & layout.head_mask)
    }

    #[inline(always)]
    fn write_slot(
        layout: SlotLayout,
        words: &mut [u64],
        slot: usize,
        tag: u64,
        head: u64,
        key: u64,
        mixed: bool,
    ) {
        debug_assert!(head <= layout.head_mask);
        debug_assert!(key <= layout.key_mask);
        words[slot] = (tag << layout.tag_shift())
            | (key << (layout.head_bits + 1))
            | (u64::from(mixed) << layout.head_bits)
            | head;
    }

    #[inline(always)]
    fn of_head(head: u64) -> u32 {
        head as u32
    }

    #[inline(always)]
    fn head_of(layout: SlotLayout, row: usize) -> u64 {
        let head = row as u64 + 1;
        assert!(head <= layout.head_mask, "inline join map row");
        head
    }

    #[inline(always)]
    fn read_key(layout: SlotLayout, words: &[u64], slot: usize) -> (u64, bool) {
        let word = words[slot];
        (
            (word >> (layout.head_bits + 1)) & layout.key_mask,
            (word >> layout.head_bits) & 1 != 0,
        )
    }
}

/// An insert into a [`HeadTable`] without room for another hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableFull;

/// Chain heads by hash: the first row (+ 1) of each distinct hash.
pub(crate) struct HeadTable<T: HeadWord> {
    words: ZeroedArray<u64>,
    /// Longbow patch (JT 9): the slot fields of a [`PackedU64`] table.
    layout: SlotLayout,
    /// The rows that the table was made for.
    capacity: usize,
    /// `64 - log2(slots)`.
    shift: u32,
    mask: usize,
    len: usize,
    prefetch: bool,
    _width: std::marker::PhantomData<T>,
}

impl<T: HeadWord> HeadTable<T> {
    fn slots_for(capacity: usize) -> Option<usize> {
        capacity.checked_mul(2)?.max(2).checked_next_power_of_two()
    }

    /// Bytes of the slots of a table made for `capacity` rows, if representable.
    pub(crate) fn checked_bytes_for(capacity: usize) -> Option<usize> {
        let words = Self::slots_for(capacity)?.checked_mul(T::WORDS)?;
        Some(Layout::array::<u64>(words).ok()?.size())
    }

    /// Bytes of the slots of a table made for `capacity` rows.
    pub(crate) fn bytes_for(capacity: usize) -> usize {
        Self::checked_bytes_for(capacity).expect("join table capacity")
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self::with_layout(capacity, SlotLayout::default())
    }

    /// Longbow patch (JT 9): a table whose slots have `layout`.
    pub(crate) fn with_layout(capacity: usize, layout: SlotLayout) -> Self {
        let slots = Self::slots_for(capacity).expect("join table capacity");
        let words = zeroed_with_huge_pages(slots * T::WORDS);
        Self {
            prefetch: prefetch_pays(words.len() * size_of::<u64>()),
            words,
            layout,
            capacity,
            shift: 64 - slots.trailing_zeros(),
            mask: slots - 1,
            len: 0,
            _width: std::marker::PhantomData,
        }
    }

    /// Distinct hashes (up to tag collisions of a `u32` table).
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Bytes of the slots.
    pub(crate) fn bytes(&self) -> usize {
        self.words.len() * size_of::<u64>()
    }

    /// Whether the table is large enough for prefetches to pay.
    #[inline(always)]
    pub(crate) fn prefetches(&self) -> bool {
        self.prefetch
    }

    #[inline(always)]
    fn position(&self, hash: u64) -> usize {
        (hash.wrapping_mul(SLOT_MUL) >> self.shift) as usize
    }

    /// Hints the cache to load the first slot of `hash`.
    #[inline(always)]
    pub(crate) fn prefetch(&self, hash: u64) {
        prefetch_read(&self.words, self.position(hash) * T::WORDS);
    }

    /// The chain head of `hash`, or 0.
    #[inline(always)]
    pub(crate) fn head(&self, hash: u64) -> u64 {
        let tag = T::tag(self.layout, hash);
        let mut slot = self.position(hash);
        loop {
            let (stored, head) = T::read(self.layout, &self.words, slot);
            if head == 0 {
                return 0;
            }
            if stored == tag {
                return head;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// Longbow patch (JT 9): the chain head of `hash` (or 0), with the
    /// slot's inline key and mixed-chain mark ([`HeadWord::read_key`]).
    #[inline(always)]
    pub(crate) fn find(&self, hash: u64) -> (u64, u64, bool) {
        let tag = T::tag(self.layout, hash);
        let mut slot = self.position(hash);
        loop {
            let (stored, head) = T::read(self.layout, &self.words, slot);
            if head == 0 {
                return (0, 0, false);
            }
            if stored == tag {
                let (key, mixed) = T::read_key(self.layout, &self.words, slot);
                return (head, key, mixed);
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// Makes `head` the chain head of `hash`; returns the head it replaces,
    /// or 0 when `hash` is new. `key` is the inline key of the row of
    /// `head` (ignored by a table without keys); a chain that gets a row of
    /// another key is marked mixed.
    ///
    /// A `u32` slot does not keep the whole hash, so the table cannot be
    /// rehashed into more slots. It takes up to twice the capacity that it
    /// was made for (every caller inserts at most that capacity); beyond
    /// that, when only one empty slot is left (it ends every probe), the
    /// insert fails with [`TableFull`] and changes nothing.
    #[inline(always)]
    pub(crate) fn insert(
        &mut self,
        hash: u64,
        head: u64,
        key: u64,
    ) -> Result<u64, TableFull> {
        let tag = T::tag(self.layout, hash);
        let mut slot = self.position(hash);
        loop {
            let (stored, previous) = T::read(self.layout, &self.words, slot);
            if previous == 0 {
                if self.len >= self.mask {
                    return Err(TableFull);
                }
                T::write_slot(self.layout, &mut self.words, slot, tag, head, key, false);
                self.len += 1;
                return Ok(0);
            }
            if stored == tag {
                let (old_key, mixed) = T::read_key(self.layout, &self.words, slot);
                let mixed = mixed || (T::INLINE && old_key != key);
                T::write_slot(self.layout, &mut self.words, slot, tag, head, key, mixed);
                return Ok(previous);
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// The error of an insert into a full table.
    #[cold]
    pub(crate) fn full_error(&self) -> datafusion_common::DataFusionError {
        internal_datafusion_err!(
            "join hash table full: {} distinct hashes in {} slots, made for {} rows",
            self.len,
            self.mask + 1,
            self.capacity
        )
    }

    /// Inserts rows in iteration order, each with its hash and inline key
    /// (0 for a table without keys). `previous` gets
    /// each row that already had a chain head, and that head: the caller
    /// links the row to it. Keeps `PREFETCH_DISTANCE` rows in flight. Fails
    /// when the table is full or a row is not below `row_bound` (the end of
    /// the caller's chain array, which every head must index); the rows
    /// before the failing one are inserted.
    pub(crate) fn insert_all<I, F>(
        &mut self,
        rows: I,
        row_bound: usize,
        mut previous: F,
    ) -> Result<()>
    where
        I: Iterator<Item = (usize, u64, u64)>,
        F: FnMut(usize, T::Chain),
    {
        let mut add = |table: &mut Self, row: usize, hash: u64, key: u64| -> Result<()> {
            if row >= row_bound {
                return Err(internal_datafusion_err!(
                    "join hash map row {row} is beyond its {row_bound} rows"
                ));
            }
            match table.insert(hash, T::head_of(table.layout, row), key) {
                Ok(0) => Ok(()),
                Ok(head) => {
                    previous(row, T::of_head(head));
                    Ok(())
                }
                Err(TableFull) => Err(table.full_error()),
            }
        };
        if !self.prefetch {
            for (row, hash, key) in rows {
                add(self, row, hash, key)?;
            }
            return Ok(());
        }
        let mut ring = [(0_usize, 0_u64, 0_u64); PREFETCH_DISTANCE];
        let mut pending = 0;
        let mut at = 0;
        for (row, hash, key) in rows {
            self.prefetch(hash);
            if pending == PREFETCH_DISTANCE {
                let (old_row, old_hash, old_key) = ring[at];
                add(self, old_row, old_hash, old_key)?;
            } else {
                pending += 1;
            }
            ring[at] = (row, hash, key);
            at = (at + 1) % PREFETCH_DISTANCE;
        }
        // The oldest pending row is at `at` once the ring is full, else at 0.
        let first = if pending == PREFETCH_DISTANCE { at } else { 0 };
        for index in 0..pending {
            let (row, hash, key) = ring[(first + index) % PREFETCH_DISTANCE];
            add(self, row, hash, key)?;
        }
        Ok(())
    }

    /// A table with the heads that `map.find` returns for its hashes. Its
    /// length is the map's: tests build maps that hold one hash twice, and
    /// a probe takes the chain path whenever the length is below the rows.
    #[cfg(test)]
    pub(crate) fn from_hashbrown<V>(map: &HashTable<(u64, V)>) -> Self
    where
        V: Copy + Into<u64>,
    {
        let mut table = Self::with_capacity(map.len());
        for (hash, _) in map.iter() {
            if table.head(*hash) == 0 {
                let (_, head) = map.find(*hash, |(h, _)| h == hash).expect("own entry");
                table.insert(*hash, (*head).into(), 0).expect("map fits");
            }
        }
        table.len = map.len();
        table
    }
}

/// Longbow patch (JT): tables from this size are built partition by
/// partition ([`build_partitioned`]); smaller ones row by row.
const PARTITIONED_BUILD_BYTES: usize = 4 << 20;

/// Slot bytes of one partition of a partitioned build: about a quarter of
/// the 1 MiB L2 of a Zen 4 core, so the partition's slots stay in L2.
const PARTITION_BYTES: usize = 256 << 10;

/// Longbow patch (JT): the build rows of a partitioned build: the hash of
/// every row (row number = index), and the batches in insertion order as
/// (first row, rows, matchable rows). Rows are inserted as the row-by-row
/// build inserts them: batch by batch, each batch's rows from its last.
pub struct BuildRows<'a> {
    pub hashes: &'a [u64],
    /// Longbow patch (JT 9): for a map with inline keys, the inline keys of
    /// the rows of `batches[i]` (in row order), computed when the build
    /// reaches that batch.
    pub keys: Option<&'a BatchKeys<'a>>,
    pub batches: &'a [(usize, usize, Option<NullBuffer>)],
}

impl BuildRows<'_> {
    /// The matchable rows in insertion order.
    fn rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.batches.iter().flat_map(|(first, len, valid)| {
            (0..*len).rev().filter_map(move |index| {
                valid
                    .as_ref()
                    .is_none_or(|valid| valid.is_valid(index))
                    .then_some(first + index)
            })
        })
    }
}

/// Longbow patch (JT 9): the inline keys of one batch of a [`BuildRows`].
pub type BatchKeys<'a> = dyn Fn(usize) -> Result<Vec<u64>> + 'a;

impl BuildRows<'_> {
    /// The rows in insertion order with their hashes and inline keys (0
    /// without keys).
    fn rows_with_keys(
        &self,
        mut add: impl FnMut(usize, u64, u64) -> Result<()>,
    ) -> Result<()> {
        for (index, (first, len, valid)) in self.batches.iter().enumerate() {
            let keys = self.keys.map(|keys| keys(index)).transpose()?;
            if keys.as_ref().is_some_and(|keys| keys.len() != *len) {
                return Err(internal_datafusion_err!("join build keys of another batch"));
            }
            for index in (0..*len).rev() {
                if valid.as_ref().is_none_or(|valid| valid.is_valid(index)) {
                    let row = first + index;
                    let key = keys.as_ref().map_or(0, |keys| keys[index]);
                    add(row, self.hashes[row], key)?;
                }
            }
        }
        Ok(())
    }
}

/// Longbow patch (JT): a map whose build can be partitioned.
pub trait PartitionedBuild: JoinHashMapType {
    /// Whether a map made for `capacity` rows is built partition by
    /// partition: its table is above `PARTITIONED_BUILD_BYTES`.
    fn partitions_build(capacity: usize) -> bool
    where
        Self: Sized;

    /// Inserts `rows` partition by partition, with the chains of
    /// `update_from_iter` over the same rows in the same order.
    fn update_partitioned(&mut self, rows: &BuildRows<'_>) -> Result<()>;
}

/// Transient bytes of [`build_partitioned`] for `rows` build rows: a hash
/// for every row, then the scattered hashes and row numbers; with `inline`
/// keys the scattered row numbers are words that also hold the keys (each
/// batch's keys live only while it is scattered).
pub fn partitioned_build_bytes(rows: usize, inline: bool) -> usize {
    let row = if inline {
        size_of::<u64>()
    } else {
        size_of::<u32>()
    };
    rows * (2 * size_of::<u64>() + row)
}

macro_rules! partitioned_build {
    ($map:ty, $width:ty) => {
        impl PartitionedBuild for $map {
            fn partitions_build(capacity: usize) -> bool {
                HeadTable::<$width>::checked_bytes_for(capacity)
                    .is_some_and(|bytes| bytes > PARTITIONED_BUILD_BYTES)
            }

            fn update_partitioned(&mut self, rows: &BuildRows<'_>) -> Result<()> {
                if <$width as HeadWord>::INLINE && rows.keys.is_none() {
                    return Err(internal_datafusion_err!(
                        "a join map with inline keys was built without them"
                    ));
                }
                let next = &mut self.next;
                let chained = &mut self.chained;
                let row_bound = next.len();
                self.table.insert_partitioned(rows, row_bound, |row, prev| {
                    if !*chained {
                        next.back_chain();
                    }
                    next[row] = prev;
                    *chained = true;
                })
            }
        }
    };
}
// Row numbers of a partitioned build are `u32`: only the `u32` map has one.
partitioned_build!(JoinHashMapU32, u32);
partitioned_build!(JoinHashMapInline, PackedU64);

impl<T: HeadWord> HeadTable<T> {
    /// Inserts `rows` as [`Self::insert_all`] in their order, but partition
    /// by partition: a counting sort by the top bits of the slot position
    /// (stable, so equal hashes keep their order and chains are the same)
    /// puts the rows of each `PARTITION_BYTES` range of slots together, and
    /// their inserts then hit the cache instead of memory.
    fn insert_partitioned<F>(
        &mut self,
        rows: &BuildRows<'_>,
        row_bound: usize,
        previous: F,
    ) -> Result<()>
    where
        F: FnMut(usize, T::Chain),
    {
        let hashes = rows.hashes;
        let slot_bits = 64 - self.shift;
        let partition_slots = (PARTITION_BYTES / (T::WORDS * size_of::<u64>())).max(1);
        let partition_bits = slot_bits.saturating_sub(partition_slots.trailing_zeros());
        if partition_bits == 0 {
            let mut all = Vec::new();
            rows.rows_with_keys(|row, hash, key| {
                all.push((row, hash, key));
                Ok(())
            })?;
            return self.insert_all(all.into_iter(), row_bound, previous);
        }
        let low = slot_bits - partition_bits;
        let partition_of = |hash: u64| self.position(hash) >> low;
        let mut starts = vec![0_usize; (1 << partition_bits) + 1];
        for row in rows.rows() {
            starts[partition_of(hashes[row]) + 1] += 1;
        }
        for index in 1..starts.len() {
            starts[index] += starts[index - 1];
        }
        let total = starts[starts.len() - 1];
        let mut scattered_hashes = vec![0_u64; total];
        if T::INLINE {
            // Longbow patch (JT 9): an inline table scatters each row with
            // its key (encoded batch by batch) in one word, key above row.
            let row_bits = self.layout.head_bits;
            let mut scattered = vec![0_u64; total];
            rows.rows_with_keys(|row, hash, key| {
                let at = &mut starts[partition_of(hash)];
                scattered_hashes[*at] = hash;
                scattered[*at] = (key << row_bits) | row as u64;
                *at += 1;
                Ok(())
            })?;
            let row_mask = low_bits(row_bits);
            return self.insert_all(
                scattered
                    .iter()
                    .zip(&scattered_hashes)
                    .map(|(&word, &hash)| {
                        ((word & row_mask) as usize, hash, word >> row_bits)
                    }),
                row_bound,
                previous,
            );
        }
        let mut scattered_rows = vec![0_u32; total];
        for row in rows.rows() {
            let hash = hashes[row];
            let at = &mut starts[partition_of(hash)];
            scattered_hashes[*at] = hash;
            scattered_rows[*at] = u32::try_from(row).map_err(|_| {
                internal_datafusion_err!("partitioned join build above u32 rows")
            })?;
            *at += 1;
        }
        // In partition order and, within a partition, in the given order.
        self.insert_all(
            scattered_rows
                .iter()
                .zip(&scattered_hashes)
                .map(|(&row, &hash)| (row as usize, hash, 0)),
            row_bound,
            previous,
        )
    }
}

/// Longbow patch (JT): builds `map` from `batches`, in the order and with
/// the row numbers of the row-by-row build of `collect_left_input`, partition
/// by partition: rows are numbered by their position in the forward
/// concatenation of `batches`, and inserted from the last batch, each batch
/// from its last row. Charges its transient arrays
/// ([`partitioned_build_bytes`]) to `reservation` while they live.
///
/// Returns the reservation's size with the transient arrays charged (the
/// build's peak, for `build_mem_used`), or `None` without touching `map`
/// when the pool refuses them: a CollectLeft build cannot spill, so the
/// caller then builds row by row, which gives the same chains, instead of
/// failing a query that fits without the partitioning.
///
/// Longbow patch (JT 9): with `inline` keys, every row's encoded key goes
/// to the map beside its hash. With `dense` keys, every row's encoded key is
/// appended to the given vector (charged by the caller) in row order, while
/// the key columns are hashed.
#[expect(clippy::too_many_arguments)]
pub(crate) fn build_partitioned<'a, M: PartitionedBuild>(
    map: &mut M,
    on: &[datafusion_physical_expr::PhysicalExprRef],
    batches: &'a [arrow::record_batch::RecordBatch],
    num_rows: usize,
    random_state: &datafusion_common::hash_utils::RandomState,
    null_equality: datafusion_common::NullEquality,
    inline: Option<&InlineKeys>,
    mut dense: Option<(&InlineKeys, &mut Vec<u64>)>,
    reservation: &mut datafusion_execution::memory_pool::MemoryReservation,
) -> Result<Option<usize>> {
    let transient = partitioned_build_bytes(num_rows, inline.is_some());
    if reservation.try_grow(transient).is_err() {
        return Ok(None);
    }
    let peak = reservation.size();
    let result = (|| {
        let mut hashes = vec![0_u64; num_rows];
        let mut layout = Vec::new();
        let mut offset = 0;
        for batch in batches {
            let rows = batch.num_rows();
            let keys =
                datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays(
                    on, batch,
                )?;
            let Some(slice) = hashes.get_mut(offset..offset + rows) else {
                return Err(internal_datafusion_err!(
                    "join build batches hold more than {num_rows} rows"
                ));
            };
            datafusion_common::hash_utils::create_hashes(&keys, random_state, slice)?;
            if let Some((encoding, out)) = dense.as_mut() {
                out.extend_from_slice(&encoding.encode_build(&keys)?);
            }
            let valid = crate::joins::utils::matchable_join_keys(&keys, null_equality);
            layout.push((offset, rows, valid));
            offset += rows;
        }
        // Insert from the last batch, as the row-by-row build does.
        layout.reverse();
        let count = batches.len();
        let batch_keys = |index: usize| -> Result<Vec<u64>> {
            let batch = &batches[count - 1 - index];
            let keys =
                datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays(
                    on, batch,
                )?;
            match inline {
                Some(inline) => inline.encode_build(&keys),
                None => Ok(vec![]),
            }
        };
        map.update_partitioned(&BuildRows {
            hashes: &hashes[..offset],
            keys: inline.map(|_| &batch_keys as &BatchKeys<'_>),
            batches: &layout,
        })
    })();
    reservation.shrink(transient);
    result.map(|()| Some(peak))
}

/// Longbow patch (JT): chain heads by hash, for the probe functions below.
pub(crate) trait HeadLookup {
    /// The chain head (row + 1) of `hash`, or 0.
    fn head(&self, hash: u64) -> u64;
    /// Hints the cache to load what [`Self::head`] reads first.
    fn prefetch(&self, hash: u64);
    /// Whether the table is large enough for prefetches to pay.
    fn prefetches(&self) -> bool;
    /// Longbow patch (JT 9): the chain head of `hash` (or 0) with its slot's
    /// inline key and mixed-chain mark ([`HeadTable::find`]).
    fn find(&self, hash: u64) -> (u64, u64, bool);
}

impl<T: HeadWord> HeadLookup for HeadTable<T> {
    #[inline(always)]
    fn head(&self, hash: u64) -> u64 {
        HeadTable::head(self, hash)
    }

    #[inline(always)]
    fn find(&self, hash: u64) -> (u64, u64, bool) {
        HeadTable::find(self, hash)
    }

    #[inline(always)]
    fn prefetch(&self, hash: u64) {
        HeadTable::prefetch(self, hash)
    }

    #[inline(always)]
    fn prefetches(&self) -> bool {
        HeadTable::prefetches(self)
    }
}

/// Longbow patch (JT): [`get_matched_indices`] over a [`HeadLookup`].
pub(crate) fn heads_matched_indices<'a, L, T>(
    table: &L,
    next: &[T],
    iter: Box<dyn Iterator<Item = (usize, &'a u64)> + 'a>,
    deleted_offset: Option<usize>,
) -> (Vec<u32>, Vec<u64>)
where
    L: HeadLookup,
    T: Copy + Into<u64>,
{
    let mut input_indices = vec![];
    let mut match_indices = vec![];
    let offset = deleted_offset.unwrap_or(0) as u64;
    for (row_idx, &hash) in iter {
        let head = table.head(hash);
        if head == 0 {
            continue;
        }
        let mut row = head - 1;
        loop {
            // A pruned chain ends below the offset.
            if row < offset {
                break;
            }
            let index = row - offset;
            match_indices.push(index);
            input_indices.push(row_idx as u32);
            let next_row: u64 = next[index as usize].into();
            if next_row == 0 {
                break;
            }
            row = next_row - 1;
        }
    }
    (input_indices, match_indices)
}

/// Longbow patch (JT): [`get_matched_indices_with_limit_offset`] over a
/// [`HeadLookup`], with the same offsets and results. `unique` is true when
/// every row has its own head, so no chain needs to be followed. A large
/// table prefetches the slot of the row `PREFETCH_DISTANCE` rows ahead.
///
/// `unique` must be `true` only if every hash value maps to a single row
/// (the `chained` flag of the maps, not `len == next.len()`: rows left out
/// for NULL keys own a chain slot but no head).
#[expect(clippy::too_many_arguments)]
pub(crate) fn heads_matched_indices_with_limit_offset<L, T>(
    table: &L,
    next_chain: &[T],
    unique: bool,
    hash_values: &[u64],
    valid_keys: Option<&NullBuffer>,
    limit: usize,
    offset: ProbeOffset,
    input_indices: &mut Vec<u32>,
    match_indices: &mut Vec<u64>,
) -> Option<ProbeOffset>
where
    L: HeadLookup,
    T: Copy + TryFrom<usize> + PartialOrd + Into<u64> + Sub<Output = T>,
    <T as TryFrom<usize>>::Error: Debug,
    T: ArrowNativeType,
{
    if table.prefetches() {
        matched_heads::<L, T, true>(
            table,
            next_chain,
            unique,
            hash_values,
            valid_keys,
            limit,
            offset,
            input_indices,
            match_indices,
        )
    } else {
        matched_heads::<L, T, false>(
            table,
            next_chain,
            unique,
            hash_values,
            valid_keys,
            limit,
            offset,
            input_indices,
            match_indices,
        )
    }
}

#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn matched_heads<L, T, const PREFETCH: bool>(
    table: &L,
    next_chain: &[T],
    unique: bool,
    hash_values: &[u64],
    valid_keys: Option<&NullBuffer>,
    limit: usize,
    offset: ProbeOffset,
    input_indices: &mut Vec<u32>,
    match_indices: &mut Vec<u64>,
) -> Option<ProbeOffset>
where
    L: HeadLookup,
    T: Copy + TryFrom<usize> + PartialOrd + Into<u64> + Sub<Output = T>,
    <T as TryFrom<usize>>::Error: Debug,
    T: ArrowNativeType,
{
    input_indices.clear();
    match_indices.clear();
    let len = hash_values.len();
    // Prefetches the rows from `from` to the distance ahead of it.
    let warm = |from: usize, end: usize| {
        if PREFETCH {
            for &hash in &hash_values[from.min(end)..(from + PREFETCH_DISTANCE).min(end)]
            {
                table.prefetch(hash);
            }
        }
    };

    // Every hash has a single row: no chain to follow.
    if unique {
        debug_assert!(
            matches!(offset, AtRow { .. }),
            "unique keys cannot have mid-chain"
        );
        let (AtRow { row: start } | MidChain { row: start, .. }) = offset;
        let end = (start + limit).min(len);
        warm(start, end);
        for row in start..end {
            if PREFETCH && row + PREFETCH_DISTANCE < end {
                table.prefetch(hash_values[row + PREFETCH_DISTANCE]);
            }
            // NULL keys cannot match any build row
            if valid_keys.is_some_and(|valid| valid.is_null(row)) {
                continue;
            }
            let head = table.head(hash_values[row]);
            if head != 0 {
                input_indices.push(row as u32);
                match_indices.push(head - 1);
            }
        }
        return if end == len {
            None
        } else {
            Some(AtRow { row: end })
        };
    }

    let mut remaining_output = limit;

    // Calculate initial `hash_values` index before iterating
    let to_skip = match offset {
        // `AtRow` indicates that `idx` processing hasn't been started
        AtRow { row: idx } => idx,
        // Otherwise, process remaining `idx` matches by traversing `next_chain`,
        // to start with the next index
        MidChain {
            row: idx,
            next: next_idx,
        } => {
            let next_idx: T = T::usize_as(next_idx as usize);
            let is_last = idx == len - 1;
            if let Some(next_offset) = traverse_chain(
                next_chain,
                idx,
                next_idx,
                &mut remaining_output,
                input_indices,
                match_indices,
                is_last,
            ) {
                return Some(next_offset);
            }
            idx + 1
        }
    };

    warm(to_skip, len);
    for row_idx in to_skip..len {
        if PREFETCH && row_idx + PREFETCH_DISTANCE < len {
            table.prefetch(hash_values[row_idx + PREFETCH_DISTANCE]);
        }
        // NULL keys cannot match any build row
        if valid_keys.is_some_and(|valid| valid.is_null(row_idx)) {
            continue;
        }
        let head = table.head(hash_values[row_idx]);
        if head != 0 {
            let is_last = row_idx == len - 1;
            if let Some(next_offset) = traverse_chain(
                next_chain,
                row_idx,
                T::usize_as(head as usize),
                &mut remaining_output,
                input_indices,
                match_indices,
                is_last,
            ) {
                return Some(next_offset);
            }
        }
    }
    None
}

/// Longbow patch (JT): [`contain_hashes`] over a [`HeadLookup`], with one
/// loop per prefetch mode.
pub(crate) fn heads_contain_hashes<L: HeadLookup>(
    table: &L,
    hash_values: &[u64],
) -> BooleanArray {
    if table.prefetches() {
        contained_heads::<L, true>(table, hash_values)
    } else {
        contained_heads::<L, false>(table, hash_values)
    }
}

#[inline(always)]
fn contained_heads<L: HeadLookup, const PREFETCH: bool>(
    table: &L,
    hash_values: &[u64],
) -> BooleanArray {
    let len = hash_values.len();
    if PREFETCH {
        for &hash in &hash_values[..PREFETCH_DISTANCE.min(len)] {
            table.prefetch(hash);
        }
    }
    let buffer = BooleanBuffer::collect_bool(len, |i| {
        if PREFETCH && i + PREFETCH_DISTANCE < len {
            table.prefetch(hash_values[i + PREFETCH_DISTANCE]);
        }
        table.head(hash_values[i]) != 0
    });
    BooleanArray::new(buffer, None)
}

/// Longbow patch (JT 9): [`heads_matched_indices_with_limit_offset`] over a
/// table with inline keys ([`PackedU64`]), for probe rows with encoded keys
/// `probe` (of the same [`InlineKeys`] as the table). Emits only pairs whose
/// keys are equal, in the order and with the offsets of the hash-only probe
/// followed by the key comparison, so the caller compares no keys:
///
/// - a slot whose chain holds a single key matches exactly when that key is
///   the probe's; then all of its rows match;
/// - a chain that mixes keys (a 32-bit tag collision of different keys) has
///   each row's key encoded from `build_keys` and compared.
///
/// `unique` is as for the hash-only probe. `limit` counts emitted pairs.
#[expect(clippy::too_many_arguments)]
pub(crate) fn inline_matched_indices_with_limit_offset<L: HeadLookup>(
    table: &L,
    next_chain: &[u32],
    unique: bool,
    hash_values: &[u64],
    probe: &EncodedKeys,
    keys: &InlineKeys,
    build_keys: &[ArrayRef],
    limit: usize,
    offset: ProbeOffset,
    input_indices: &mut Vec<u32>,
    match_indices: &mut Vec<u64>,
) -> Option<ProbeOffset> {
    let probe = InlineProbe {
        hash_values,
        keys: &probe.keys,
        valid: probe.valid.as_ref(),
        encoding: keys,
        build_keys,
    };
    if table.prefetches() {
        inline_heads::<L, true>(
            table,
            next_chain,
            unique,
            &probe,
            limit,
            offset,
            input_indices,
            match_indices,
        )
    } else {
        inline_heads::<L, false>(
            table,
            next_chain,
            unique,
            &probe,
            limit,
            offset,
            input_indices,
            match_indices,
        )
    }
}

/// The probe side of [`inline_matched_indices_with_limit_offset`].
struct InlineProbe<'a> {
    hash_values: &'a [u64],
    keys: &'a [u64],
    valid: Option<&'a NullBuffer>,
    encoding: &'a InlineKeys,
    build_keys: &'a [ArrayRef],
}

impl InlineProbe<'_> {
    /// Whether build row `row` of a mixed chain has probe key `key`.
    #[cold]
    fn build_row_has(&self, row: usize, key: u64) -> bool {
        self.encoding.encode_row(self.build_keys, row) == Some(key)
    }
}

#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn inline_heads<L: HeadLookup, const PREFETCH: bool>(
    table: &L,
    next_chain: &[u32],
    unique: bool,
    probe: &InlineProbe<'_>,
    limit: usize,
    offset: ProbeOffset,
    input_indices: &mut Vec<u32>,
    match_indices: &mut Vec<u64>,
) -> Option<ProbeOffset> {
    input_indices.clear();
    match_indices.clear();
    let hash_values = probe.hash_values;
    let len = hash_values.len();
    let warm = |from: usize, end: usize| {
        if PREFETCH {
            for &hash in &hash_values[from.min(end)..(from + PREFETCH_DISTANCE).min(end)]
            {
                table.prefetch(hash);
            }
        }
    };
    let invalid = |row: usize| probe.valid.is_some_and(|valid| valid.is_null(row));

    if unique {
        debug_assert!(
            matches!(offset, AtRow { .. }),
            "unique keys cannot have mid-chain"
        );
        let (AtRow { row: start } | MidChain { row: start, .. }) = offset;
        let end = (start + limit).min(len);
        warm(start, end);
        for row in start..end {
            if PREFETCH && row + PREFETCH_DISTANCE < end {
                table.prefetch(hash_values[row + PREFETCH_DISTANCE]);
            }
            if invalid(row) {
                continue;
            }
            // A unique table has no mixed chain.
            let (head, key, _) = table.find(hash_values[row]);
            if head != 0 && key == probe.keys[row] {
                input_indices.push(row as u32);
                match_indices.push(head - 1);
            }
        }
        return if end == len {
            None
        } else {
            Some(AtRow { row: end })
        };
    }

    let mut remaining = limit;
    let to_skip = match offset {
        AtRow { row } => row,
        MidChain { row, next } => {
            // The chain of `row` matched; filter it again if it is mixed.
            let (_, _, mixed) = table.find(hash_values[row]);
            if let Some(next_offset) = inline_chain(
                next_chain,
                probe,
                row,
                next,
                mixed,
                &mut remaining,
                input_indices,
                match_indices,
                row == len - 1,
            ) {
                return Some(next_offset);
            }
            row + 1
        }
    };

    warm(to_skip, len);
    for row in to_skip..len {
        if PREFETCH && row + PREFETCH_DISTANCE < len {
            table.prefetch(hash_values[row + PREFETCH_DISTANCE]);
        }
        if invalid(row) {
            continue;
        }
        let (head, key, mixed) = table.find(hash_values[row]);
        if head == 0 || (!mixed && key != probe.keys[row]) {
            continue;
        }
        if let Some(next_offset) = inline_chain(
            next_chain,
            probe,
            row,
            head,
            mixed,
            &mut remaining,
            input_indices,
            match_indices,
            row == len - 1,
        ) {
            return Some(next_offset);
        }
    }
    None
}

/// [`traverse_chain`] for [`inline_heads`]: emits the rows of the chain from
/// `start` (row + 1); a `mixed` chain only the rows with the probe's key.
/// Returns the resume offset when `remaining` reaches 0, as `traverse_chain`.
#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn inline_chain(
    next_chain: &[u32],
    probe: &InlineProbe<'_>,
    probe_row: usize,
    start: u64,
    mixed: bool,
    remaining: &mut usize,
    input_indices: &mut Vec<u32>,
    match_indices: &mut Vec<u64>,
    is_last_input: bool,
) -> Option<ProbeOffset> {
    let mut row = start - 1;
    loop {
        let next = u64::from(next_chain[row as usize]);
        if !mixed || probe.build_row_has(row as usize, probe.keys[probe_row]) {
            match_indices.push(row);
            input_indices.push(probe_row as u32);
            *remaining -= 1;
            if *remaining == 0 {
                return if is_last_input && next == 0 {
                    None
                } else if next == 0 {
                    Some(AtRow { row: probe_row + 1 })
                } else {
                    Some(MidChain {
                        row: probe_row,
                        next,
                    })
                };
            }
        }
        if next == 0 {
            return None;
        }
        row = next - 1;
    }
}

/// Longbow patch (JT): hints the CPU to load `slice[index]` into all cache
/// levels. Does nothing when `index` is out of bounds or the target has no
/// prefetch instruction here.
#[inline(always)]
pub fn prefetch_read<T>(slice: &[T], index: usize) {
    #[cfg(target_arch = "x86_64")]
    if let Some(element) = slice.get(index) {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        // SAFETY: `element` is a reference into `slice`, so the address is
        // inside a live allocation. A prefetch neither faults nor changes
        // memory or registers; it only hints the cache.
        unsafe { _mm_prefetch::<_MM_HINT_T0>((element as *const T).cast::<i8>()) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (slice, index);
    }
}

/// Longbow patch (JT): a zeroed array of `len` values, with its own
/// 2 MiB-page mapping from `HUGE_PAGE_BYTES` on where the process's mode
/// ([`huge_page_mode`]) and its stall gate allow one; otherwise
/// `vec![0; len]`.
pub(crate) fn zeroed_with_huge_pages<T: ZeroedValue>(len: usize) -> ZeroedArray<T> {
    if len.saturating_mul(size_of::<T>()) >= HUGE_PAGE_BYTES
        && let Some(mapped) = ZeroedArray::for_mode(len, huge_page_mode(), &STALL_GATE)
    {
        return mapped;
    }
    vec![T::default(); len].into()
}

/// Longbow patch (JT 10): the zeroed chain array of a map for `len` rows. A
/// chain is written only for rows whose hash already has a row, so a build
/// with unique keys never touches it. Where a table would be backed when it
/// is made, the chain is therefore mapped without advice and backed by its
/// first write ([`ZeroedArray::back_chain`]); before that it takes no memory.
pub(crate) fn zeroed_chain<T: ZeroedValue>(len: usize) -> ZeroedArray<T> {
    if len.saturating_mul(size_of::<T>()) >= HUGE_PAGE_BYTES
        && let Some(mapped) = ZeroedArray::chain_for_mode(len, huge_page_mode())
    {
        return mapped;
    }
    vec![T::default(); len].into()
}

/// Longbow patch (JT): the values a [`ZeroedArray`] holds.
///
/// # Safety
///
/// The all-zero bit pattern must be a valid value of the type (and equal to
/// its `Default`): a mapped [`ZeroedArray`] hands out fresh anonymous pages,
/// which read as zero, as initialized values.
pub unsafe trait ZeroedValue: Copy + Default + Send + Sync + 'static {}
// SAFETY: every bit pattern, zero included, is a valid `u32`, and zero is
// its default.
unsafe impl ZeroedValue for u32 {}
// SAFETY: as for `u32`.
unsafe impl ZeroedValue for u64 {}

#[cfg(test)]
thread_local! {
    /// Test hook: the next [`ZeroedArray::mapped`] on this thread fails as
    /// a refused `mmap` would.
    static REFUSE_NEXT_MAPPING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Longbow patch (JT): a fixed-length array of a join table: a `Vec`, or
/// its own anonymous mapping, 2 MiB aligned and advised `MADV_HUGEPAGE`,
/// which is unmapped when the array is dropped. The advice therefore ends
/// with the table and never reaches memory that the allocator later hands to
/// other operators (advice on allocator memory did: their faults there
/// allocated huge pages with synchronous compaction).
pub struct ZeroedArray<T: ZeroedValue> {
    /// The first value: into `owner`'s buffer, which never moves.
    ptr: std::ptr::NonNull<T>,
    len: usize,
    owner: ArrayOwner<T>,
    /// Longbow patch (JT 10): a mapped chain without advice that waits for
    /// its first write ([`Self::back_chain`]).
    unbacked_chain: bool,
}

enum ArrayOwner<T> {
    /// The values; held only to own (and free) them.
    Heap { _values: Vec<T> },
    /// The whole mapping (the values, rounded up to whole base pages),
    /// unmapped on drop.
    #[cfg_attr(not(target_os = "linux"), expect(dead_code))]
    Mapped {
        base: std::ptr::NonNull<u8>,
        bytes: usize,
    },
}

// SAFETY: a `ZeroedArray<T>` owns its values exactly as a `Vec<T>` does (the
// pointer is into a buffer that only this array reaches), so it may move to
// and be shared between threads under the same bounds.
unsafe impl<T: ZeroedValue> Send for ZeroedArray<T> {}
// SAFETY: see `Send`; shared access gives only `&[T]`.
unsafe impl<T: ZeroedValue> Sync for ZeroedArray<T> {}

impl<T: ZeroedValue> From<Vec<T>> for ZeroedArray<T> {
    fn from(mut values: Vec<T>) -> Self {
        let ptr = std::ptr::NonNull::new(values.as_mut_ptr()).expect("Vec pointer");
        Self {
            ptr,
            len: values.len(),
            owner: ArrayOwner::Heap { _values: values },
            unbacked_chain: false,
        }
    }
}

impl<T: ZeroedValue> ZeroedArray<T> {
    /// `len` zeroed values in their own anonymous mapping, 2 MiB aligned and
    /// advised `MADV_HUGEPAGE`, whatever [`huge_page_mode`] says (the gate
    /// is [`zeroed_with_huge_pages`]). `None` when the kernel refuses the
    /// mapping or the advice, or the target has none of these calls; the
    /// caller then uses a `Vec`.
    ///
    /// The mapping is exactly the values rounded up to whole base pages: the
    /// alignment slack before them and the address space after them are
    /// unmapped again. A 2 MiB region that the values fill only in part is
    /// therefore not wholly inside the mapping, so the kernel backs it with
    /// base pages, and the resident bytes stay below the values' bytes plus
    /// one base page (the charge of the callers is the values' bytes).
    pub(crate) fn mapped(len: usize) -> Option<Self> {
        Self::mapped_as(len, true)
    }

    /// Longbow patch (JT 10): [`Self::mapped`], or with `advise` unset the
    /// same mapping without the advice: its first writes get base pages.
    fn mapped_as(len: usize, advise: bool) -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            use crate::joins::huge_page_policy::HUGE_PAGE;
            #[cfg(test)]
            if REFUSE_NEXT_MAPPING.with(|refuse| refuse.replace(false)) {
                return None;
            }
            // SAFETY: `sysconf` only reads a system constant.
            let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
                .ok()
                .filter(|page| page.is_power_of_two() && *page <= HUGE_PAGE)?;
            let bytes = len.checked_mul(size_of::<T>())?.max(1);
            let used = bytes.checked_next_multiple_of(page)?;
            // One more huge page, so that an aligned start leaves room for
            // every value.
            let map_bytes = bytes
                .checked_next_multiple_of(HUGE_PAGE)?
                .checked_add(HUGE_PAGE)?;
            // SAFETY: a new private anonymous mapping at an address of the
            // kernel's choice; no existing memory is touched. Its pages read
            // as zero until written.
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    map_bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if base == libc::MAP_FAILED {
                return None;
            }
            let base = base.cast::<u8>();
            // `base` is page aligned, so `offset` and `used` are whole pages
            // and `offset < HUGE_PAGE`.
            let offset = (base as usize).next_multiple_of(HUGE_PAGE) - base as usize;
            let tail = map_bytes - offset - used;
            // SAFETY: `base..base + map_bytes` is the mapping made above, and
            // `offset + used + tail == map_bytes`, so `start..start + used`
            // and the head and tail ranges unmapped below lie inside it and
            // are page aligned. Nothing else refers to the mapping yet.
            // `MADV_HUGEPAGE` only sets a flag on the kept range: it changes
            // no contents and no permission.
            let start = unsafe {
                let start = base.add(offset);
                let trimmed = (offset == 0 || libc::munmap(base.cast(), offset) == 0)
                    && (tail == 0 || libc::munmap(start.add(used).cast(), tail) == 0);
                if !trimmed {
                    // Unmap whatever is left of the mapping, then use a `Vec`.
                    libc::munmap(base.cast(), map_bytes);
                    return None;
                }
                if advise && libc::madvise(start.cast(), used, libc::MADV_HUGEPAGE) != 0 {
                    // No huge pages for this mapping: a `Vec` reuses the
                    // allocator's pages instead.
                    libc::munmap(start.cast(), used);
                    return None;
                }
                start
            };
            // A 2 MiB-aligned address is non-null and aligned for `T`.
            let start = std::ptr::NonNull::new(start).expect("mapping address");
            Some(Self {
                ptr: start.cast::<T>(),
                len,
                owner: ArrayOwner::Mapped {
                    base: start,
                    bytes: used,
                },
                unbacked_chain: false,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (len, advise);
            None
        }
    }

    /// Longbow patch (JT 10): `len` zeroed values in their own mapping as
    /// `mode` backs it, or `None` for a heap array: the mode is off, `gate`
    /// is closed (automatic mode), or the kernel refuses the mapping.
    ///
    /// [`HugePageMode::FirstTouch`] leaves the advised mapping unbacked. The
    /// other modes back it now, so no later write allocates a page:
    /// [`HugePageMode::Populate`] all of it, and [`HugePageMode::Auto`] until
    /// a 2 MiB region stalls or `gate` closes; the rest then gets base pages
    /// at its first writes.
    pub(crate) fn for_mode(
        len: usize,
        mode: HugePageMode,
        gate: &StallGate,
    ) -> Option<Self> {
        let ticket = match mode {
            HugePageMode::Off => return None,
            HugePageMode::FirstTouch => return Self::mapped(len),
            HugePageMode::Populate => None,
            HugePageMode::Auto => Some(gate.admit(std::time::Instant::now())?),
        };
        let mapped = Self::mapped(len)?;
        mapped.back(ticket);
        Some(mapped)
    }

    /// Backs the advised mapping of this array now
    /// ([`back_mapping`](crate::joins::huge_page_policy::back_mapping)).
    fn back(&self, ticket: Option<Ticket<'_>>) {
        #[cfg(target_os = "linux")]
        if let ArrayOwner::Mapped { base, bytes } = self.owner {
            // SAFETY: `base..base + bytes` is the mapping that `mapped_as`
            // made for this array alone: private, anonymous, 2 MiB aligned
            // and a whole number of base pages, whose size it validated.
            unsafe {
                let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
                crate::joins::huge_page_policy::back_mapping(
                    base.as_ptr(),
                    bytes,
                    page,
                    ticket,
                );
            }
        }
        #[cfg(not(target_os = "linux"))]
        drop(ticket);
    }

    /// Longbow patch (JT 10): a chain of `len` zeroed values in its own
    /// mapping as `mode` makes it, or `None` for a heap array.
    /// [`HugePageMode::FirstTouch`] advises it, as for any array. The modes
    /// that back a table when it is made leave a chain without advice and
    /// unbacked until [`Self::back_chain`].
    pub(crate) fn chain_for_mode(len: usize, mode: HugePageMode) -> Option<Self> {
        match mode {
            HugePageMode::Off => None,
            HugePageMode::FirstTouch => Self::mapped(len),
            HugePageMode::Populate | HugePageMode::Auto => {
                let mut chain = Self::mapped_as(len, false)?;
                chain.unbacked_chain = true;
                Some(chain)
            }
        }
    }

    /// Longbow patch (JT 10): called before the first write to a chain.
    /// Advises and backs a chain that waited for it, as the process's mode
    /// and its stall gate allow.
    #[inline(always)]
    pub(crate) fn back_chain(&mut self) {
        if self.unbacked_chain {
            self.back_chain_for(huge_page_mode(), &STALL_GATE);
        }
    }

    /// [`Self::back_chain`] for `mode` and `gate`: all of the chain in
    /// [`HugePageMode::Populate`]; in [`HugePageMode::Auto`] as far as
    /// `gate` allows, and nothing while it is closed (the chain then gets
    /// base pages at its writes).
    #[cold]
    pub(crate) fn back_chain_for(&mut self, mode: HugePageMode, gate: &StallGate) {
        if !std::mem::take(&mut self.unbacked_chain) {
            return;
        }
        let ticket = match mode {
            HugePageMode::Populate => None,
            HugePageMode::Auto => match gate.admit(std::time::Instant::now()) {
                Some(ticket) => Some(ticket),
                None => return,
            },
            HugePageMode::Off | HugePageMode::FirstTouch => return,
        };
        #[cfg(target_os = "linux")]
        if let ArrayOwner::Mapped { base, bytes } = self.owner {
            // SAFETY: the mapping of this array (see `back`); the advice
            // sets a flag on it and changes no contents and no permission.
            let advised = unsafe {
                libc::madvise(base.as_ptr().cast(), bytes, libc::MADV_HUGEPAGE) == 0
            };
            if advised {
                self.back(ticket);
            }
        }
        #[cfg(not(target_os = "linux"))]
        drop(ticket);
    }

    /// Whether the values live in their own mapping.
    pub(crate) fn is_mapped(&self) -> bool {
        matches!(self.owner, ArrayOwner::Mapped { .. })
    }
}

impl<T: ZeroedValue> Drop for ZeroedArray<T> {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if let ArrayOwner::Mapped { base, bytes } = self.owner {
            // SAFETY: the mapping is this array's alone, and nothing borrows
            // it once the array is dropped. Unmapping also ends its advice.
            let unmapped = unsafe { libc::munmap(base.as_ptr().cast(), bytes) };
            debug_assert_eq!(unmapped, 0, "munmap of a join table mapping");
        }
    }
}

impl<T: ZeroedValue> std::ops::Deref for ZeroedArray<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` points to `len` initialized values owned by `owner`,
        // which lives as long as `self`: a `Vec`'s values, or a mapping's
        // pages, which are zero until written and are valid values because
        // of `ZeroedValue`'s contract.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: ZeroedValue> std::ops::DerefMut for ZeroedArray<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as `deref`; `&mut self` makes this the only borrow.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: ZeroedValue + Debug> Debug for ZeroedArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ZeroedArray")
            .field("len", &self.len)
            .field("mapped", &self.is_mapped())
            .finish()
    }
}

#[cfg(test)]
#[path = "join_hash_map_inline_tests.rs"]
mod inline_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contain_hashes() {
        let mut hash_map = JoinHashMapU32::with_capacity(10);
        hash_map
            .update_from_iter(Box::new([10u64, 20u64, 30u64].iter().enumerate()), 0)
            .unwrap();

        let probe_hashes = vec![10, 11, 20, 21, 30, 31];
        let array = hash_map.contain_hashes(&probe_hashes);

        assert_eq!(array.len(), probe_hashes.len());

        for (i, &hash) in probe_hashes.iter().enumerate() {
            if matches!(hash, 10 | 20 | 30) {
                assert!(array.value(i), "Hash {hash} should exist in the map");
            } else {
                assert!(!array.value(i), "Hash {hash} should NOT exist in the map");
            }
        }
    }

    #[test]
    fn test_get_matched_indices_skips_invalid_keys() {
        let mut hash_map = JoinHashMapU32::with_capacity(3);
        hash_map
            .update_from_iter(Box::new([10u64, 20u64, 30u64].iter().enumerate()), 0)
            .unwrap();

        let probe_hashes = vec![10, 20, 30];
        // The probe row for hash 20 has a NULL key and must not match.
        let valid_keys = NullBuffer::from(vec![true, false, true]);

        let mut input_indices = vec![];
        let mut match_indices = vec![];
        let next_offset = hash_map.get_matched_indices_with_limit_offset(
            &probe_hashes,
            Some(&valid_keys),
            8192,
            AtRow { row: 0 },
            &mut input_indices,
            &mut match_indices,
        );

        assert_eq!(next_offset, None);
        assert_eq!(input_indices, vec![0, 2]);
        assert_eq!(match_indices, vec![0, 2]);
    }

    #[test]
    fn test_get_matched_indices_skips_invalid_keys_with_duplicates() {
        // Duplicate build keys chain multiple rows under one hash value.
        let mut hash_map = JoinHashMapU32::with_capacity(4);
        hash_map
            .update_from_iter(
                Box::new([10u64, 20u64, 10u64, 20u64].iter().enumerate()),
                0,
            )
            .unwrap();

        let probe_hashes = vec![10, 20];
        // The probe row for hash 10 has a NULL key: none of the build rows in
        // its chain may match, while the valid probe row for hash 20 must
        // still match its entire chain.
        let valid_keys = NullBuffer::from(vec![false, true]);

        let mut input_indices = vec![];
        let mut match_indices = vec![];
        let next_offset = hash_map.get_matched_indices_with_limit_offset(
            &probe_hashes,
            Some(&valid_keys),
            8192,
            AtRow { row: 0 },
            &mut input_indices,
            &mut match_indices,
        );

        assert_eq!(next_offset, None);
        assert_eq!(input_indices, vec![1, 1]);
        assert_eq!(match_indices, vec![3, 1]);
    }

    #[test]
    fn test_unique_fast_path_with_uninserted_row() {
        // Row 1 is left out, as `update_hash` does for a NULL key under
        // `NullEquality::NullEqualsNothing`. Every hash value still maps to a
        // single row, so the probe takes the unique-key fast path.
        let build_hashes = [10u64, 99, 20, 30];
        let mut hash_map = JoinHashMapU32::with_capacity(build_hashes.len());
        hash_map
            .update_from_iter(
                Box::new(build_hashes.iter().enumerate().filter(|(row, _)| *row != 1)),
                0,
            )
            .unwrap();

        // Probe row 1 has no match.
        let probe_hashes = vec![10, 40, 20, 30];
        let mut input_indices = vec![];
        let mut match_indices = vec![];
        let next_offset = hash_map.get_matched_indices_with_limit_offset(
            &probe_hashes,
            None,
            2,
            AtRow { row: 0 },
            &mut input_indices,
            &mut match_indices,
        );
        // The fast path limits by probe rows and resumes at probe row 2 after a
        // single match; the chain walk limits by matches and would also emit
        // probe row 2's match, resuming at probe row 3.
        assert_eq!(next_offset, Some(AtRow { row: 2 }));
        assert_eq!(input_indices, vec![0]);
        assert_eq!(match_indices, vec![0]);

        let next_offset = hash_map.get_matched_indices_with_limit_offset(
            &probe_hashes,
            None,
            2,
            AtRow { row: 2 },
            &mut input_indices,
            &mut match_indices,
        );
        assert_eq!(next_offset, None);
        assert_eq!(input_indices, vec![2, 3]);
        assert_eq!(match_indices, vec![2, 3]);
    }

    #[test]
    fn test_chain_from_earlier_update_disables_fast_path() {
        // Rows 0 and 1 share a hash value; a later update without a chain must
        // not re-enable the unique-key fast path.
        let mut hash_map = JoinHashMapU32::with_capacity(3);
        hash_map
            .update_from_iter(Box::new([10u64, 10u64].iter().enumerate()), 0)
            .unwrap();
        hash_map
            .update_from_iter(Box::new([(2, &20u64)].into_iter()), 0)
            .unwrap();

        let mut input_indices = vec![];
        let mut match_indices = vec![];
        let next_offset = hash_map.get_matched_indices_with_limit_offset(
            &[10],
            None,
            8192,
            AtRow { row: 0 },
            &mut input_indices,
            &mut match_indices,
        );

        assert_eq!(next_offset, None);
        assert_eq!(input_indices, vec![0, 0]);
        assert_eq!(match_indices, vec![1, 0]);
    }

    #[test]
    fn test_chain_created_by_later_update_disables_fast_path() {
        assert_chain_created_by_later_update(JoinHashMapU32::with_capacity(3));
        assert_chain_created_by_later_update(JoinHashMapU64::with_capacity(3));
    }

    /// Rows 0 and 2 share a hash value but arrive in separate updates, and row
    /// 1 is left out as a NULL key would be. The first update alone is unique;
    /// the second creates the chain. Probing with `limit = 1` must resume
    /// through the chain and return each build row exactly once.
    fn assert_chain_created_by_later_update(mut hash_map: impl JoinHashMapType) {
        hash_map
            .update_from_iter(Box::new([(0, &10u64)].into_iter()), 0)
            .unwrap();
        hash_map
            .update_from_iter(Box::new([(2, &10u64)].into_iter()), 0)
            .unwrap();

        let mut input_indices = vec![];
        let mut match_indices = vec![];
        let mut matched = vec![];
        let mut offset = AtRow { row: 0 };
        for _ in 0..3 {
            let next_offset = hash_map.get_matched_indices_with_limit_offset(
                &[10],
                None,
                1,
                offset,
                &mut input_indices,
                &mut match_indices,
            );
            assert_eq!(input_indices, vec![0]);
            matched.extend_from_slice(&match_indices);
            match next_offset {
                Some(next) => offset = next,
                None => break,
            }
        }

        matched.sort_unstable();
        assert_eq!(matched, vec![0, 2]);
    }

    // Longbow patch (JT): the head table against the hashbrown map that it
    // replaces, through the generic functions that still serve hashbrown.

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The hashbrown reference: the map and chain of the code before JT.
    fn reference(hashes: &[u64]) -> (HashTable<(u64, u32)>, Vec<u32>, bool) {
        let mut map = HashTable::with_capacity(hashes.len());
        let mut next = vec![0_u32; hashes.len()];
        // As `update_hash` with `fifo_hashmap`: rows in reverse.
        let chained = update_from_iter::<u32>(
            &mut map,
            &mut next,
            Box::new(hashes.iter().enumerate().rev()),
            0,
        );
        (map, next, chained)
    }

    /// Every (probe, build) pair of a probe batch, resumed at `limit`.
    fn drain(
        probe: impl Fn(ProbeOffset, &mut Vec<u32>, &mut Vec<u64>) -> Option<ProbeOffset>,
    ) -> Vec<(u32, u64)> {
        let (mut input, mut matched, mut pairs) = (vec![], vec![], vec![]);
        let mut offset = AtRow { row: 0 };
        loop {
            let next = probe(offset, &mut input, &mut matched);
            pairs.extend(input.iter().copied().zip(matched.iter().copied()));
            match next {
                Some(resume) => offset = resume,
                None => return pairs,
            }
        }
    }

    fn compare_with_reference(build: &[u64], probe: &[u64], valid: Option<&NullBuffer>) {
        let (map, next, chained) = reference(build);
        let mut table = JoinHashMapU32::with_capacity(build.len());
        table
            .update_from_iter(Box::new(build.iter().enumerate().rev()), 0)
            .unwrap();
        assert_eq!(table.len(), map.len());
        assert_eq!(table.chained, chained);
        let limits: &[usize] = if build.len() <= 1000 {
            &[1, 2, 3, 7, 64, 8192]
        } else {
            &[7, 8192]
        };
        for &limit in limits {
            let expected = drain(|offset, input, matched| {
                get_matched_indices_with_limit_offset::<u32>(
                    &map, &next, !chained, probe, valid, limit, offset, input, matched,
                )
            });
            let actual = drain(|offset, input, matched| {
                table.get_matched_indices_with_limit_offset(
                    probe, valid, limit, offset, input, matched,
                )
            });
            assert_eq!(actual, expected, "limit {limit}");
        }
        let iter = || Box::new(probe.iter().enumerate());
        assert_eq!(
            table.get_matched_indices(iter(), None),
            get_matched_indices::<u32>(&map, &next, iter(), None)
        );
        assert_eq!(table.contain_hashes(probe), contain_hashes(&map, probe));
    }

    #[test]
    fn head_table_matches_hashbrown_with_duplicates_and_null_keys() {
        let mut state = 7;
        // Small: no prefetch. Large: the table is above `PREFETCH_BYTES`.
        for (rows, keys) in [
            (100, 30),
            (1000, 1000),
            (200_000, 50_000),
            (100_000, 100_000),
        ] {
            let domain: Vec<u64> = (0..keys).map(|_| splitmix(&mut state)).collect();
            let build: Vec<u64> = (0..rows)
                .map(|_| domain[(splitmix(&mut state) % keys as u64) as usize])
                .collect();
            // Half the probes hit, half miss.
            let probe: Vec<u64> = (0..rows)
                .map(|row| {
                    if row % 2 == 0 {
                        domain[(splitmix(&mut state) % keys as u64) as usize]
                    } else {
                        splitmix(&mut state)
                    }
                })
                .collect();
            let valid = NullBuffer::from(
                (0..rows).map(|row| row % 5 != 3).collect::<Vec<bool>>(),
            );
            compare_with_reference(&build, &probe, None);
            compare_with_reference(&build, &probe, Some(&valid));
        }
        let table = JoinHashMapU32::with_capacity(200_000);
        assert!(table.table.prefetches());
        assert!(!JoinHashMapU32::with_capacity(1000).table.prefetches());
    }

    #[test]
    fn head_table_with_one_hash_for_every_row() {
        // `force_hash_collisions`: every row in one chain.
        let build = vec![42_u64; 1000];
        compare_with_reference(&build, &[42, 1, 42], None);
    }

    #[test]
    fn a_u32_head_table_merges_only_hashes_with_equal_low_bits() {
        // Two hashes with equal low 32 bits share one chain in a `u32`
        // table: the probe of either gets the rows of both, which the key
        // comparison then drops. Other hashes stay apart.
        let (a, b) = (0x1234_5678_9ABC_DEF0_u64, 0x8765_4321_9ABC_DEF0_u64);
        let mut table = HeadTable::<u32>::with_capacity(4);
        table.insert(a, 1, 0).unwrap();
        let other = table.insert(b, 2, 0).unwrap();
        // `b` lands in the same probe run only if its slot is reached; the
        // table then reports `a`'s head as the head it replaces.
        if other != 0 {
            assert_eq!((other, table.head(a), table.len()), (1, 2, 1));
        } else {
            assert_eq!((table.head(a), table.head(b), table.len()), (1, 2, 2));
        }
        // A `u64` table compares whole hashes.
        let mut wide = HeadTable::<u64>::with_capacity(4);
        assert_eq!((wide.insert(a, 1, 0), wide.insert(b, 2, 0)), (Ok(0), Ok(0)));
        assert_eq!((wide.head(a), wide.head(b), wide.len()), (1, 2, 2));
    }

    #[test]
    fn a_u64_map_matches_the_u32_map() {
        let mut state = 11;
        let build: Vec<u64> = (0..5000).map(|_| splitmix(&mut state) % 1777).collect();
        let probe: Vec<u64> = (0..5000).map(|_| splitmix(&mut state) % 2000).collect();
        let mut narrow = JoinHashMapU32::with_capacity(build.len());
        let mut wide = JoinHashMapU64::with_capacity(build.len());
        narrow
            .update_from_iter(Box::new(build.iter().enumerate().rev()), 0)
            .unwrap();
        wide.update_from_iter(Box::new(build.iter().enumerate().rev()), 0)
            .unwrap();
        for limit in [1, 5, 8192] {
            let a = drain(|o, i, m| {
                narrow.get_matched_indices_with_limit_offset(&probe, None, limit, o, i, m)
            });
            let b = drain(|o, i, m| {
                wide.get_matched_indices_with_limit_offset(&probe, None, limit, o, i, m)
            });
            assert_eq!(a, b);
        }
    }

    #[test]
    fn a_head_table_takes_twice_its_capacity_and_then_refuses() {
        let mut table = HeadTable::<u64>::with_capacity(4);
        // Eight slots: seven distinct hashes fit, one slot stays empty.
        for hash in 0..7 {
            assert_eq!(table.insert(hash, hash + 1, 0), Ok(0));
        }
        for hash in 0..7 {
            assert_eq!(table.head(hash), hash + 1);
        }
        assert_eq!(table.head(100), 0);
        // An existing hash still takes a new head; a new one is refused and
        // changes nothing.
        assert_eq!(table.insert(3, 50, 0), Ok(4));
        assert_eq!(table.insert(7, 8, 0), Err(TableFull));
        assert_eq!((table.len(), table.head(7), table.head(3)), (7, 0, 50));
    }

    /// An overfilled map returns an error through `update_hash`, the build
    /// path of `HashJoinExec`, for both widths and with and without
    /// prefetching; it does not panic.
    #[test]
    fn an_overfilled_map_returns_an_error() {
        use crate::joins::utils::update_hash;
        use arrow::array::{RecordBatch, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use datafusion_common::NullEquality;
        use datafusion_common::hash_utils::RandomState;
        use datafusion_physical_expr::expressions::Column;
        use std::sync::Arc;

        let schema =
            Arc::new(Schema::new(vec![Field::new("k", DataType::UInt64, false)]));
        let on: Vec<datafusion_physical_expr::PhysicalExprRef> =
            vec![Arc::new(Column::new("k", 0))];
        for (capacity, rows) in [(4_usize, 100_u64), (40_000, 200_000)] {
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(UInt64Array::from_iter_values(0..rows))],
            )
            .unwrap();
            let maps: [Box<dyn JoinHashMapType>; 2] = [
                Box::new(JoinHashMapU32::with_capacity(capacity)),
                Box::new(JoinHashMapU64::with_capacity(capacity)),
            ];
            for mut map in maps {
                let mut hashes = vec![0; batch.num_rows()];
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        update_hash(
                            &on,
                            &batch,
                            &mut *map,
                            0,
                            &RandomState::with_seed(0),
                            &mut hashes,
                            0,
                            true,
                            NullEquality::NullEqualsNothing,
                        )
                    }))
                    .expect("no panic");
                // The chain array bounds the rows before the slots run out.
                let error = result.expect_err("overfilled map").to_string();
                assert!(
                    error.contains(&format!("beyond its {capacity} rows")),
                    "{error}"
                );
            }
        }
        // Repeated keys and rows beyond the chain array: an error, also when
        // the table still has room (review of JT, finding 2).
        for keys in [vec![1_u64, 2, 1, 2, 1, 2], vec![10, 11, 12, 13, 14, 15]] {
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(UInt64Array::from(keys))],
            )
            .unwrap();
            let mut map = JoinHashMapU32::with_capacity(4);
            let mut hashes = vec![0; batch.num_rows()];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                update_hash(
                    &on,
                    &batch,
                    &mut map,
                    0,
                    &RandomState::with_seed(0),
                    &mut hashes,
                    0,
                    true,
                    NullEquality::NullEqualsNothing,
                )
            }))
            .expect("no panic");
            let error = result.expect_err("rows beyond the map").to_string();
            assert!(error.contains("beyond its 4 rows"), "{error}");
        }
        // Within its rows, a map with twice the slots takes them all.
        let mut map = JoinHashMapU32::with_capacity(7);
        let hashes: Vec<u64> = (0..7).collect();
        map.update_from_iter(Box::new(hashes.iter().enumerate()), 0)
            .unwrap();
        assert_eq!(map.len(), 7);
        // A `u64` map reserves its real table size.
        assert!(
            JoinHashMapU64::table_bytes(5) >= HeadTable::<u64>::with_capacity(5).bytes()
        );
    }

    #[test]
    fn prefetch_read_ignores_out_of_bounds_indices() {
        let values = [1_u64, 2, 3];
        prefetch_read(&values, 1);
        prefetch_read(&values, 3);
        prefetch_read::<u64>(&[], 0);
    }

    /// Whether every array from `HUGE_PAGE_BYTES` on has its own mapping:
    /// the modes without the stall gate, which sends tables to the heap
    /// while the host has no free huge page.
    fn always_mapped() -> bool {
        matches!(
            huge_page_mode(),
            HugePageMode::FirstTouch | HugePageMode::Populate
        )
    }

    /// The base page size.
    #[cfg(target_os = "linux")]
    fn page_size() -> usize {
        // SAFETY: `sysconf` only reads a system constant.
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    /// Whether the host's kernel has usable transparent huge pages.
    fn thp_available() -> bool {
        std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
            .is_ok_and(|mode| !mode.contains("[never]"))
    }

    #[test]
    fn huge_page_advice_keeps_contents() {
        let values: ZeroedArray<u64> = zeroed_with_huge_pages(1 << 20);
        assert_eq!(values.len(), 1 << 20);
        assert!(values.iter().all(|value| *value == 0));
        let table = HeadTable::<u32>::with_capacity(1 << 20);
        assert_eq!(table.bytes(), (2 << 20) * size_of::<u64>());
        // Small arrays stay on the heap.
        assert!(!zeroed_with_huge_pages::<u32>(1000).is_mapped());
    }

    /// The mappings of this process: (start, end, VmFlags, Rss in bytes).
    #[cfg(target_os = "linux")]
    fn mappings() -> Vec<(usize, usize, String, usize)> {
        let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
        let mut out: Vec<(usize, usize, String, usize)> = vec![];
        for line in smaps.lines() {
            let range = line.split(' ').next().unwrap_or_default();
            let bounds = range.split_once('-').and_then(|(low, high)| {
                Some((
                    usize::from_str_radix(low, 16).ok()?,
                    usize::from_str_radix(high, 16).ok()?,
                ))
            });
            if let Some((low, high)) = bounds {
                out.push((low, high, String::new(), 0));
            } else if let Some(last) = out.last_mut() {
                if let Some(flags) = line.strip_prefix("VmFlags:") {
                    last.2 = flags.trim().to_string();
                } else if let Some(rss) = line.strip_prefix("Rss:") {
                    let kb: usize =
                        rss.trim().trim_end_matches("kB").trim().parse().unwrap();
                    last.3 = kb * 1024;
                }
            }
        }
        out
    }

    /// Advised (`hg`) mappings of this process: (start, end).
    #[cfg(target_os = "linux")]
    fn advised_mappings() -> Vec<(usize, usize)> {
        mappings()
            .into_iter()
            .filter(|(_, _, flags, _)| flags.split_whitespace().any(|f| f == "hg"))
            .map(|(low, high, _, _)| (low, high))
            .collect()
    }

    /// Review of the advice lifecycle (leto-mjj.61): a mapped array is
    /// zeroed, writable and 2 MiB aligned, lives in an advised mapping of
    /// exactly its bytes rounded up to base pages, and is unmapped when
    /// dropped, so its advice cannot reach other memory.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_mapped_array_is_unmapped_on_drop() {
        let len = (9 << 20) / size_of::<u32>() + 3;
        let Some(mut values) = ZeroedArray::<u32>::mapped(len) else {
            // A kernel without transparent huge pages refuses the advice.
            assert!(!thp_available());
            return;
        };
        assert!(values.is_mapped());
        assert_eq!(values.len(), len);
        assert_eq!(values.as_ptr() as usize % (2 << 20), 0);
        assert!(values.iter().all(|value| *value == 0));
        values[0] = 1;
        values[len - 1] = 2;
        assert_eq!((values[0], values[len / 2], values[len - 1]), (1, 0, 2));
        let start = values.as_ptr() as usize;
        let end = start + (len * size_of::<u32>()).next_multiple_of(4096);
        // Exactly one advised mapping, of exactly the values' pages.
        assert!(
            advised_mappings().contains(&(start, end)),
            "{:?}",
            advised_mappings()
        );
        drop(values);
        assert!(!advised_mappings().contains(&(start, end)));
        // A heap array keeps its values as a `Vec` would.
        let heap: ZeroedArray<u64> = vec![5_u64; 10].into();
        assert!(!heap.is_mapped());
        assert_eq!(&*heap, &[5; 10]);
    }

    /// Review S1 (leto-mjj.61): a fully written mapped array is resident for
    /// at most its bytes plus one base page, which is what its callers
    /// charge (plus less than a page): the 2 MiB region that it fills only in
    /// part is not backed by a whole huge page.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_mapped_array_is_resident_for_at_most_its_bytes() {
        for bytes in [(4 << 20) + 4, (4 << 20) + (64 << 10), 6 << 20] {
            let len = bytes / size_of::<u32>();
            let Some(mut values) = ZeroedArray::<u32>::mapped(len) else {
                // Only a kernel without usable transparent huge pages
                // refuses the advice.
                assert!(!thp_available());
                return;
            };
            values.iter_mut().for_each(|value| *value = 7);
            let start = values.as_ptr() as usize;
            let (_, _, _, rss) = mappings()
                .into_iter()
                .find(|(low, high, _, _)| *low <= start && start < *high)
                .expect("the array's mapping");
            let charged = len * size_of::<u32>();
            assert!(
                rss <= charged + 4096,
                "{bytes}: rss {rss} charged {charged}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn resident_u32_join_table_fits_its_reservation() -> Result<()> {
        // Both arrays use their own mappings (in the automatic mode, while
        // its gate is open). The chain is a whole number of base pages even
        // on Linux targets with 64 KiB pages.
        let rows = (1 << 20) + (1 << 14);
        let mut map = JoinHashMapU32::with_capacity(rows);
        if !(map.table.words.is_mapped() && map.next.is_mapped()) {
            assert!(!always_mapped());
            return Ok(());
        }
        map.table.words.fill(1);
        map.next.fill(1);
        // Every mapping of both arrays: a stalled region splits one in two.
        let resident: usize = backing(&map.table.words)
            .into_iter()
            .chain(backing(&map.next))
            .map(|(rss, _)| rss)
            .sum();
        let reserved = JoinHashMapU32::reservation_bytes(rows)?;
        assert_eq!(
            resident,
            map.allocated_bytes() - size_of::<JoinHashMapU32>()
        );
        assert_eq!(reserved, resident + size_of::<JoinHashMapU32>());
        println!(
            "{rows} rows: {resident} resident array bytes, {reserved} reserved bytes"
        );
        Ok(())
    }

    /// Review S4 (leto-mjj.61): a refused mapping falls back to a zeroed,
    /// writable heap array of the full length.
    #[test]
    fn a_refused_mapping_falls_back_to_the_heap() {
        REFUSE_NEXT_MAPPING.with(|refuse| refuse.set(true));
        let len = (5 << 20) / size_of::<u64>();
        let mut values: ZeroedArray<u64> = zeroed_with_huge_pages(len);
        assert!(!values.is_mapped());
        assert_eq!(values.len(), len);
        assert!(values.iter().all(|value| *value == 0));
        values[len - 1] = 9;
        assert_eq!(values[len - 1], 9);
        // The hook is spent: the next array is mapped where every array is.
        let next: ZeroedArray<u64> = zeroed_with_huge_pages(len);
        if always_mapped() {
            assert!(next.is_mapped());
        } else if huge_page_mode() == HugePageMode::Off {
            assert!(!next.is_mapped());
        }
    }

    /// The resident bytes and the `VmFlags` of the mappings that hold the
    /// bytes of `values`, in address order.
    #[cfg(target_os = "linux")]
    fn backing<T: ZeroedValue>(values: &ZeroedArray<T>) -> Vec<(usize, String)> {
        let start = values.as_ptr() as usize;
        let end = start + size_of_val(&**values);
        mappings()
            .into_iter()
            .filter(|(low, high, _, _)| *low < end && start < *high)
            .map(|(low, high, flags, rss)| {
                assert!(start <= low && high <= end.next_multiple_of(page_size()));
                (rss, flags)
            })
            .collect()
    }

    #[cfg(target_os = "linux")]
    fn has_flag(flags: &str, flag: &str) -> bool {
        flags.split_whitespace().any(|f| f == flag)
    }

    /// JT 10 (leto-o7z0): the modes of an array's backing. Off uses the
    /// heap; first touch leaves an advised mapping unbacked; populate backs
    /// all of it when it is made.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_mode_decides_how_an_array_is_backed() {
        if !thp_available() {
            return;
        }
        let gate = StallGate::new();
        let bytes = (9 << 20) + 2 * page_size();
        let len = bytes / size_of::<u64>();
        assert!(ZeroedArray::<u64>::for_mode(len, HugePageMode::Off, &gate).is_none());

        let lazy = ZeroedArray::<u64>::for_mode(len, HugePageMode::FirstTouch, &gate)
            .expect("mapped");
        let parts = backing(&lazy);
        assert_eq!(parts.len(), 1, "{parts:?}");
        assert_eq!(parts[0].0, 0);
        assert!(has_flag(&parts[0].1, "hg"));

        let mut eager = ZeroedArray::<u64>::for_mode(len, HugePageMode::Populate, &gate)
            .expect("mapped");
        let parts = backing(&eager);
        assert_eq!(parts.len(), 1, "{parts:?}");
        // Every page is resident before the first write, the partial last
        // region included, and the values are zero.
        assert_eq!(parts[0].0, bytes);
        assert!(has_flag(&parts[0].1, "hg"));
        assert!(eager.iter().all(|value| *value == 0));
        eager[len - 1] = 3;
        assert_eq!(backing(&eager)[0].0, bytes);
        // Neither mode uses the gate.
        assert!(gate.is_open());
    }

    /// JT 10 (leto-o7z0): the automatic mode backs a table while regions
    /// get free huge pages. At the first stalled region the rest of the
    /// mapping loses its advice and stays unbacked, the gate closes, and
    /// the next table uses the heap.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_stalled_region_ends_the_advice_and_closes_the_gate() {
        use crate::joins::huge_page_policy::{RegionCost, STALL_AT_REGION};
        if !thp_available() {
            return;
        }
        const REGION: usize = 2 << 20;
        let gate = StallGate::new();
        let bytes = 5 * REGION + page_size();
        let len = bytes / size_of::<u32>();

        // No region stalls: the whole table is resident and advised.
        STALL_AT_REGION.set(Some(usize::MAX));
        let whole = ZeroedArray::<u32>::for_mode(len, HugePageMode::Auto, &gate)
            .expect("the gate is open");
        let parts = backing(&whole);
        assert_eq!(parts.len(), 1, "{parts:?}");
        assert_eq!(parts[0].0, bytes);
        assert!(has_flag(&parts[0].1, "hg"));
        assert!(gate.is_open());

        // The third region stalls: three regions are resident and advised,
        // the rest is neither.
        STALL_AT_REGION.set(Some(2));
        let mut partial = ZeroedArray::<u32>::for_mode(len, HugePageMode::Auto, &gate)
            .expect("the gate is open");
        let parts = backing(&partial);
        assert_eq!(parts.len(), 2, "{parts:?}");
        assert_eq!(parts[0].0, 3 * REGION);
        assert!(has_flag(&parts[0].1, "hg"));
        assert_eq!(parts[1].0, 0);
        assert!(has_flag(&parts[1].1, "nh") && !has_flag(&parts[1].1, "hg"));
        assert!(!gate.is_open());
        // The whole array is zero and writable; the rest gets base pages.
        assert!(partial.iter().all(|value| *value == 0));
        partial.iter_mut().for_each(|value| *value = 7);
        assert!(partial.iter().all(|value| *value == 7));
        let parts = backing(&partial);
        assert_eq!(parts.iter().map(|(rss, _)| rss).sum::<usize>(), bytes);

        // While a gate is closed, tables use the heap and ask nothing. This
        // gate is closed until an hour from now, whatever the test's pace.
        STALL_AT_REGION.set(Some(usize::MAX));
        let closed = StallGate::new();
        let later = std::time::Instant::now() + std::time::Duration::from_secs(3600);
        let base_pages = RegionCost {
            cpu: std::time::Duration::ZERO,
            faults: 512,
        };
        assert!(closed.admit(later).unwrap().region(base_pages, 512, later));
        assert!(ZeroedArray::<u32>::for_mode(len, HugePageMode::Auto, &closed).is_none());
        // The modes without the gate are not affected.
        assert!(
            ZeroedArray::<u32>::for_mode(len, HugePageMode::Populate, &closed).is_some()
        );

        // Both parts of the mapping go with the array.
        let start = partial.as_ptr() as usize;
        drop(partial);
        assert!(
            mappings()
                .iter()
                .all(|(low, high, _, _)| *high <= start || start + bytes <= *low)
        );
        STALL_AT_REGION.set(None);
    }

    /// JT 10 (leto-o7z0): a table that is being backed when another thread
    /// closes the gate stops asking at its next region.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_closed_gate_stops_a_table_at_its_next_region() {
        use crate::joins::huge_page_policy::STALL_AT_REGION;
        if !thp_available() {
            return;
        }
        let gate = StallGate::new();
        let len = (8 << 20) / size_of::<u64>();
        // The ticket of this table predates the stall of another table.
        let now = std::time::Instant::now();
        let ticket = gate.admit(now).expect("the gate is open");
        STALL_AT_REGION.set(Some(0));
        let stalled = ZeroedArray::<u64>::for_mode(len, HugePageMode::Auto, &gate)
            .expect("the gate is open");
        assert!(!gate.is_open() && !ticket.may_ask());
        STALL_AT_REGION.set(None);
        let parts = backing(&stalled);
        assert_eq!(parts.len(), 2, "{parts:?}");
        assert_eq!(parts[0].0, 2 << 20);

        let late = ZeroedArray::<u64>::mapped(len).expect("mapped");
        let ArrayOwner::Mapped { base, bytes } = late.owner else {
            unreachable!("a mapped array");
        };
        // SAFETY: the mapping of `late`, which this test owns.
        let backed = unsafe {
            crate::joins::huge_page_policy::back_mapping(
                base.as_ptr(),
                bytes,
                page_size(),
                Some(ticket),
            )
        };
        assert_eq!(backed, crate::joins::huge_page_policy::Backed::Until(0));
        let parts = backing(&late);
        assert_eq!(parts.len(), 1, "{parts:?}");
        assert_eq!(parts[0].0, 0);
        assert!(has_flag(&parts[0].1, "nh"));
    }

    /// JT 10 (leto-o7z0): what the stall rule reads from the kernel. A region
    /// that the kernel backs with base pages costs the calling thread one
    /// minor fault per base page, which the rule counts as a stall; backing
    /// it again costs no such faults.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_region_of_base_pages_costs_a_fault_per_page() {
        use crate::joins::huge_page_policy::{HUGE_PAGE, back_region};
        // A mapping without advice gets base pages where THP is `madvise`.
        let enabled =
            std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
                .unwrap_or_default();
        if !enabled.contains("[madvise]") {
            return;
        }
        let len = (3 * HUGE_PAGE) / size_of::<u64>();
        let values = ZeroedArray::<u64>::mapped_as(len, false).expect("mapped");
        let pages = (HUGE_PAGE / page_size()) as u64;
        let region = values.as_ptr() as *mut u8;
        // SAFETY: the first 2 MiB of the mapping of `values`, which this
        // test owns.
        let cost = unsafe { back_region(region, HUGE_PAGE) }
            .expect("backed")
            .expect("thread usage");
        // One fault per base page, and a few of the thread's own.
        assert!((pages..pages + 64).contains(&cost.faults), "{cost:?}");
        assert!(cost.stalled(pages as usize));
        assert_eq!(backing(&values)[0].0, HUGE_PAGE);
        // SAFETY: as above.
        let again = unsafe { back_region(region, HUGE_PAGE) }
            .expect("backed")
            .expect("thread usage");
        assert!(again.faults < 64, "{again:?}");
        assert!(!again.stalled(pages as usize));
        assert!(values.iter().all(|value| *value == 0));
    }

    /// JT 10 (leto-o7z0): a chain is written only for rows whose hash
    /// already has a row. Where tables are backed when made, a chain is
    /// mapped without advice and takes no memory until its first write,
    /// which advises and backs it as the mode and the gate allow.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_chain_is_backed_by_its_first_write() {
        use crate::joins::huge_page_policy::{RegionCost, STALL_AT_REGION};
        if !thp_available() {
            return;
        }
        let bytes = (6 << 20) + page_size();
        let len = bytes / size_of::<u32>();
        assert!(ZeroedArray::<u32>::chain_for_mode(len, HugePageMode::Off).is_none());
        let lazy = ZeroedArray::<u32>::chain_for_mode(len, HugePageMode::FirstTouch)
            .expect("mapped");
        assert!(has_flag(&backing(&lazy)[0].1, "hg") && !lazy.unbacked_chain);

        let open = StallGate::new();
        let closed = StallGate::new();
        let later = std::time::Instant::now() + std::time::Duration::from_secs(3600);
        let base_pages = RegionCost {
            cpu: std::time::Duration::ZERO,
            faults: 512,
        };
        assert!(closed.admit(later).unwrap().region(base_pages, 512, later));
        STALL_AT_REGION.set(Some(usize::MAX));
        for (mode, gate, backed) in [
            (HugePageMode::Populate, &open, true),
            (HugePageMode::Populate, &closed, true),
            (HugePageMode::Auto, &open, true),
            (HugePageMode::Auto, &closed, false),
        ] {
            let mut chain =
                ZeroedArray::<u32>::chain_for_mode(len, mode).expect("mapped");
            let parts = backing(&chain);
            assert_eq!(parts.len(), 1, "{mode:?} {parts:?}");
            assert_eq!(parts[0].0, 0, "{mode:?}");
            assert!(!has_flag(&parts[0].1, "hg"), "{mode:?}");
            assert!(chain.unbacked_chain);

            chain.back_chain_for(mode, gate);
            assert!(!chain.unbacked_chain);
            let parts = backing(&chain);
            assert_eq!(parts.len(), 1, "{mode:?} {parts:?}");
            assert_eq!(parts[0].0, if backed { bytes } else { 0 }, "{mode:?}");
            assert_eq!(has_flag(&parts[0].1, "hg"), backed, "{mode:?}");
            // Once is enough: a second call changes nothing.
            chain.back_chain_for(mode, gate);
            assert_eq!(backing(&chain), parts);
            assert!(chain.iter().all(|value| *value == 0));
            chain[len - 1] = 5;
            assert_eq!(chain[len - 1], 5);
        }
        STALL_AT_REGION.set(None);
        assert!(open.is_open() && !closed.is_open());
    }

    /// JT 10 (leto-o7z0): the chain of a build with unique hashes stays
    /// unbacked; the first row whose hash has a row backs it.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_unique_build_leaves_its_chain_unbacked() {
        let rows = (1 << 20) + 7;
        let hashes: Vec<u64> = (0..rows as u64)
            .map(|h| h.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        let mut map = JoinHashMapU32::with_capacity(rows);
        let deferred = map.next.unbacked_chain;
        assert_eq!(
            deferred,
            map.next.is_mapped()
                && matches!(
                    huge_page_mode(),
                    HugePageMode::Populate | HugePageMode::Auto
                )
        );
        map.update_from_iter(Box::new(hashes.iter().enumerate().take(rows - 1)), 0)
            .unwrap();
        assert!(!map.chained);
        assert_eq!(map.next.unbacked_chain, deferred);
        if map.next.is_mapped() {
            // Nothing was written to the chain, in any mode.
            assert_eq!(
                backing(&map.next).iter().map(|(rss, _)| rss).sum::<usize>(),
                0
            );
        }
        // The last row repeats the hash of row 0.
        let repeated = [hashes[0]];
        map.update_from_iter(Box::new(std::iter::once((rows - 1, &repeated[0]))), 0)
            .unwrap();
        assert!(map.chained && !map.next.unbacked_chain);
        assert_eq!(map.next[rows - 1], 1);
        assert!(map.next[..rows - 1].iter().all(|next| *next == 0));
        if deferred && huge_page_mode() == HugePageMode::Populate {
            let chain_bytes = (rows * size_of::<u32>()).next_multiple_of(page_size());
            assert_eq!(
                backing(&map.next).iter().map(|(rss, _)| rss).sum::<usize>(),
                chain_bytes
            );
        }
    }

    /// Review S4 (leto-mjj.61): tables dropped after a full build, after a
    /// failed build and from a partitioned build leave no advised mapping.
    #[cfg(target_os = "linux")]
    #[test]
    fn dropped_tables_leave_no_advised_mapping() {
        let before = advised_mappings();
        let rows = 1 << 20;
        let hashes: Vec<u64> = (0..rows as u64)
            .map(|h| h.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        {
            let mut map = JoinHashMapU32::with_capacity(rows);
            map.update_from_iter(Box::new(hashes.iter().enumerate()), 0)
                .unwrap();
            let mut wide = JoinHashMapU64::with_capacity(rows);
            wide.update_from_iter(Box::new(hashes.iter().enumerate()), 0)
                .unwrap();
            if always_mapped() {
                assert!(advised_mappings().len() > before.len());
            }
        }
        {
            // Rows beyond the map: the build fails, the map is dropped.
            let mut map = JoinHashMapU32::with_capacity(rows / 2);
            assert!(
                map.update_from_iter(Box::new(hashes.iter().enumerate()), 0)
                    .is_err()
            );
        }
        let after = advised_mappings();
        assert!(
            after.iter().all(|range| before.contains(range)),
            "{before:?} -> {after:?}"
        );
    }

    // Review of JT (leto-mjj.49): the unique path with prefetch, forced
    // 32-bit tag collisions and a partial insert before an error.

    /// Exact (probe, build) pairs of the hashbrown reference, which compares
    /// whole hashes.
    fn reference_pairs(
        build: &[u64],
        probe: &[u64],
        valid: Option<&NullBuffer>,
        limit: usize,
    ) -> Vec<(u32, u64)> {
        let (map, next, chained) = reference(build);
        drain(|offset, input, matched| {
            get_matched_indices_with_limit_offset::<u32>(
                &map, &next, !chained, probe, valid, limit, offset, input, matched,
            )
        })
    }

    /// The key comparison of the join, with the keys equal to the hashes.
    fn filtered(pairs: Vec<(u32, u64)>, build: &[u64], probe: &[u64]) -> Vec<(u32, u64)> {
        pairs
            .into_iter()
            .filter(|(p, b)| build[*b as usize] == probe[*p as usize])
            .collect()
    }

    #[test]
    fn review_unique_path_with_prefetch_matches_hashbrown() {
        let mut state = 99;
        let rows = 100_000;
        let build: Vec<u64> = (0..rows).map(|_| splitmix(&mut state)).collect();
        let probe: Vec<u64> = (0..rows)
            .map(|r| {
                if r % 3 == 0 {
                    splitmix(&mut state)
                } else {
                    build[(splitmix(&mut state) % rows as u64) as usize]
                }
            })
            .collect();
        let valid =
            NullBuffer::from((0..rows).map(|r| r % 7 != 2).collect::<Vec<bool>>());
        let mut table = JoinHashMapU32::with_capacity(rows);
        table
            .update_from_iter(Box::new(build.iter().enumerate().rev()), 0)
            .unwrap();
        assert!(table.table.prefetches());
        assert_eq!(table.len(), rows);
        assert!(!table.chained, "unique path must be taken");
        for v in [None, Some(&valid)] {
            for limit in [1usize, 7, 16, 17, 8192] {
                let actual = drain(|o, i, m| {
                    table.get_matched_indices_with_limit_offset(&probe, v, limit, o, i, m)
                });
                assert_eq!(
                    actual,
                    reference_pairs(&build, &probe, v, limit),
                    "limit {limit}"
                );
            }
        }
    }

    /// Many hashes share their low 32 bits, so a `u32` table merges chains.
    fn collision_case(rows: usize, groups: u64, seed: u64) -> (Vec<u64>, Vec<u64>) {
        let mut state = seed;
        // `groups` distinct low halves, random high halves; about rows / 4 keys.
        let keys: Vec<u64> = (0..rows / 4)
            .map(|_| {
                ((splitmix(&mut state) >> 32) << 32) | (splitmix(&mut state) % groups)
            })
            .collect();
        let build: Vec<u64> = (0..rows)
            .map(|_| keys[(splitmix(&mut state) % keys.len() as u64) as usize])
            .collect();
        let probe: Vec<u64> = (0..rows / 2)
            .map(|r| {
                if r % 2 == 0 {
                    keys[(splitmix(&mut state) % keys.len() as u64) as usize]
                } else {
                    ((splitmix(&mut state) >> 32) << 32) | (splitmix(&mut state) % groups)
                }
            })
            .collect();
        (build, probe)
    }

    #[test]
    fn review_tag_collisions_row_by_row_u32_and_u64() {
        for (rows, groups) in [(400usize, 3u64), (20_000, 50), (300_000, 1000)] {
            let (build, probe) = collision_case(rows, groups, rows as u64);
            let valid = NullBuffer::from(
                (0..probe.len()).map(|r| r % 5 != 1).collect::<Vec<bool>>(),
            );
            let mut narrow = JoinHashMapU32::with_capacity(rows);
            narrow
                .update_from_iter(Box::new(build.iter().enumerate().rev()), 0)
                .unwrap();
            let mut wide = JoinHashMapU64::with_capacity(rows);
            wide.update_from_iter(Box::new(build.iter().enumerate().rev()), 0)
                .unwrap();
            for v in [None, Some(&valid)] {
                for limit in [1usize, 3, 64, 8192] {
                    let expected = reference_pairs(&build, &probe, v, limit);
                    let a = drain(|o, i, m| {
                        narrow.get_matched_indices_with_limit_offset(
                            &probe, v, limit, o, i, m,
                        )
                    });
                    assert!(a.len() >= expected.len());
                    assert_eq!(
                        filtered(a, &build, &probe),
                        expected,
                        "u32 rows {rows} limit {limit}"
                    );
                    let b = drain(|o, i, m| {
                        wide.get_matched_indices_with_limit_offset(
                            &probe, v, limit, o, i, m,
                        )
                    });
                    assert_eq!(b, expected, "u64 rows {rows} limit {limit}");
                }
            }
            // contain_hashes: a superset of exact membership.
            let exact = contain_hashes(&reference(&build).0, &probe);
            let got = narrow.contain_hashes(&probe);
            for i in 0..probe.len() {
                if exact.value(i) {
                    assert!(got.value(i));
                }
            }
            // get_matched_indices, without a limit.
            let iter = || Box::new(probe.iter().enumerate());
            let (pi, bi) = narrow.get_matched_indices(iter(), None);
            let (reference_map, reference_next, _) = reference(&build);
            let (ri, rb) =
                get_matched_indices::<u32>(&reference_map, &reference_next, iter(), None);
            let got: Vec<(u32, u64)> =
                filtered(pi.into_iter().zip(bi).collect(), &build, &probe);
            assert_eq!(got, ri.into_iter().zip(rb).collect::<Vec<_>>());
        }
    }

    #[test]
    fn review_insert_all_error_after_partial_insert_with_prefetch() {
        // A prefetching table made for 40,000 rows; rows beyond them are an
        // error, and the rows before the failing one stay inserted.
        let mut map = JoinHashMapU32::with_capacity(40_000);
        assert!(map.table.prefetches());
        let hashes: Vec<u64> = (0..50_000u64)
            .map(|h| h.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        let err = map
            .update_from_iter(Box::new(hashes.iter().enumerate()), 0)
            .unwrap_err();
        assert!(err.to_string().contains("beyond its 40000 rows"), "{err}");
        assert_eq!(map.len(), 40_000);
        for (row, h) in hashes.iter().enumerate().take(40_000) {
            assert_eq!(map.table.head(*h), row as u64 + 1);
        }
    }

    /// The partitioned build gives the maps of the row-by-row build of
    /// `collect_left_input`: same heads, chains and probe results, with
    /// duplicates and NULL keys across several batches; its transient memory
    /// is charged and returned.
    #[test]
    fn a_partitioned_build_equals_the_row_by_row_build() {
        use crate::joins::utils::update_hash;
        use arrow::array::{RecordBatch, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use datafusion_common::NullEquality;
        use datafusion_common::hash_utils::{RandomState, create_hashes};
        use datafusion_execution::memory_pool::{
            MemoryConsumer, MemoryPool, UnboundedMemoryPool,
        };
        use datafusion_physical_expr::expressions::Column;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::UInt64, true)]));
        let on: Vec<datafusion_physical_expr::PhysicalExprRef> =
            vec![Arc::new(Column::new("k", 0))];
        let mut state = 3;
        let batches: Vec<RecordBatch> = [100_000_usize, 1, 0, 150_000, 70_000]
            .iter()
            .map(|&rows| {
                let keys = UInt64Array::from_iter((0..rows).map(|_| {
                    let value = splitmix(&mut state);
                    // A seventh NULL; keys from 200,000 values, so duplicates.
                    (value % 7 != 0).then_some(value % 200_000)
                }));
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(keys)]).unwrap()
            })
            .collect();
        let num_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        assert!(JoinHashMapU32::partitions_build(num_rows));
        let random_state = RandomState::with_seed(0);
        for null_equality in [
            NullEquality::NullEqualsNothing,
            NullEquality::NullEqualsNull,
        ] {
            // As `collect_left_input`: rows at their forward position, the
            // batches inserted from the last.
            let mut expected = JoinHashMapU32::with_capacity(num_rows);
            let (mut offset, mut hashes) = (num_rows, Vec::new());
            for batch in batches.iter().rev() {
                offset -= batch.num_rows();
                hashes.clear();
                hashes.resize(batch.num_rows(), 0);
                update_hash(
                    &on,
                    batch,
                    &mut expected,
                    offset,
                    &random_state,
                    &mut hashes,
                    0,
                    true,
                    null_equality,
                )
                .unwrap();
            }
            let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
            let mut reservation = MemoryConsumer::new("build").register(&pool);
            let mut actual = JoinHashMapU32::with_capacity(num_rows);
            let peak = build_partitioned(
                &mut actual,
                &on,
                &batches,
                num_rows,
                &random_state,
                null_equality,
                None,
                None,
                &mut reservation,
            )
            .unwrap();
            // The peak includes the transient arrays; they are returned.
            assert_eq!(peak, Some(partitioned_build_bytes(num_rows, false)));
            assert_eq!(reservation.size(), 0);
            assert_eq!(*actual.next, *expected.next);
            assert_eq!(actual.len(), expected.len());
            assert_eq!(actual.chained, expected.chained);
            // Probe with every value and some absent ones.
            let probe: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..250_000));
            let mut probe_hashes = vec![0; probe.len()];
            create_hashes([&probe], &random_state, &mut probe_hashes).unwrap();
            for limit in [7, 8192] {
                let a = drain(|o, i, m| {
                    actual.get_matched_indices_with_limit_offset(
                        &probe_hashes,
                        None,
                        limit,
                        o,
                        i,
                        m,
                    )
                });
                let b = drain(|o, i, m| {
                    expected.get_matched_indices_with_limit_offset(
                        &probe_hashes,
                        None,
                        limit,
                        o,
                        i,
                        m,
                    )
                });
                assert_eq!(a, b);
            }
        }
    }

    /// Review of JT (leto-mjj.49): the partitioned build with forced 32-bit
    /// tag collisions and NULL keys, against hashbrown and the row-by-row
    /// build in the same order.
    #[test]
    fn review_tag_collisions_partitioned_build() {
        let rows = 600_000usize;
        assert!(JoinHashMapU32::partitions_build(rows));
        for groups in [3000u64, 100_000] {
            let (build, probe) = collision_case(rows, groups, 5 + groups);
            // Two batches; the second has NULL keys (rows that cannot match).
            let split = 250_000;
            let valid2 = NullBuffer::from(
                (0..rows - split).map(|r| r % 9 != 4).collect::<Vec<bool>>(),
            );
            let layout = vec![
                (0, split, None),
                (split, rows - split, Some(valid2.clone())),
            ];
            let mut actual = JoinHashMapU32::with_capacity(rows);
            actual
                .update_partitioned(&BuildRows {
                    hashes: &build,
                    keys: None,
                    batches: &layout,
                })
                .unwrap();
            // The row-by-row order: batch 0 from its last row, then batch 1.
            let mut expected_map = HashTable::with_capacity(rows);
            let mut expected_next = vec![0u32; rows];
            let order: Vec<(usize, &u64)> = (0..split)
                .rev()
                .chain((split..rows).rev().filter(|r| valid2.is_valid(r - split)))
                .map(|r| (r, &build[r]))
                .collect();
            let chained = update_from_iter::<u32>(
                &mut expected_map,
                &mut expected_next,
                Box::new(order.clone().into_iter()),
                0,
            );
            let mut row_by_row = JoinHashMapU32::with_capacity(rows);
            row_by_row
                .update_from_iter(Box::new(order.into_iter()), 0)
                .unwrap();
            for limit in [5usize, 8192] {
                let exp = drain(|o, i, m| {
                    get_matched_indices_with_limit_offset::<u32>(
                        &expected_map,
                        &expected_next,
                        !chained,
                        &probe,
                        None,
                        limit,
                        o,
                        i,
                        m,
                    )
                });
                let a = drain(|o, i, m| {
                    actual.get_matched_indices_with_limit_offset(
                        &probe, None, limit, o, i, m,
                    )
                });
                let r = drain(|o, i, m| {
                    row_by_row.get_matched_indices_with_limit_offset(
                        &probe, None, limit, o, i, m,
                    )
                });
                assert_eq!(
                    filtered(a, &build, &probe),
                    exp,
                    "partitioned groups {groups} limit {limit}"
                );
                assert_eq!(
                    filtered(r, &build, &probe),
                    exp,
                    "row-by-row groups {groups} limit {limit}"
                );
            }
        }
    }

    /// Review of JT (leto-mjj.49), M2: a pool that refuses the partitioned
    /// build's transient arrays leaves the map and the reservation untouched,
    /// so the caller can build row by row.
    #[test]
    fn a_refused_partitioned_build_touches_nothing() {
        use arrow::array::{RecordBatch, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use datafusion_common::NullEquality;
        use datafusion_common::hash_utils::RandomState;
        use datafusion_execution::memory_pool::{
            GreedyMemoryPool, MemoryConsumer, MemoryPool,
        };
        use datafusion_physical_expr::expressions::Column;
        use std::sync::Arc;

        let rows = 300_000_usize;
        assert!(JoinHashMapU32::partitions_build(rows));
        let schema =
            Arc::new(Schema::new(vec![Field::new("k", DataType::UInt64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(UInt64Array::from_iter_values(0..rows as u64))],
        )
        .unwrap();
        let on: Vec<datafusion_physical_expr::PhysicalExprRef> =
            vec![Arc::new(Column::new("k", 0))];
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(
            partitioned_build_bytes(rows, false) - 1,
        ));
        let mut reservation = MemoryConsumer::new("build").register(&pool);
        let mut map = JoinHashMapU32::with_capacity(rows);
        let peak = build_partitioned(
            &mut map,
            &on,
            std::slice::from_ref(&batch),
            rows,
            &RandomState::with_seed(0),
            NullEquality::NullEqualsNothing,
            None,
            None,
            &mut reservation,
        )
        .unwrap();
        assert_eq!(
            (peak, map.len(), reservation.size(), pool.reserved()),
            (None, 0, 0, 0)
        );
    }

    /// Review S4 (leto-mjj.61): the prefetching and the plain probe loops
    /// give the same pairs and resume offsets on the same input, unique and
    /// chained, at probe lengths around the prefetch distance and limits
    /// around it, with resumes through `AtRow` and `MidChain`.
    #[test]
    fn prefetch_modes_give_equal_probes() {
        let mut state = 21;
        for (keys, duplicates) in [(40_u64, 1_usize), (12, 3)] {
            let build: Vec<u64> = (0..keys)
                .flat_map(|k| {
                    std::iter::repeat_n(k.wrapping_mul(0x9E37_79B9), duplicates)
                })
                .collect();
            let mut map = JoinHashMapU32::with_capacity(build.len());
            map.update_from_iter(Box::new(build.iter().enumerate().rev()), 0)
                .unwrap();
            assert_eq!(map.chained, duplicates > 1);
            for len in [15_usize, 16, 17, 33] {
                let probe: Vec<u64> = (0..len)
                    .map(|_| {
                        let r = splitmix(&mut state);
                        if r % 4 == 0 {
                            r
                        } else {
                            build[(r % build.len() as u64) as usize]
                        }
                    })
                    .collect();
                let valid =
                    NullBuffer::from((0..len).map(|r| r % 6 != 5).collect::<Vec<bool>>());
                for v in [None, Some(&valid)] {
                    for limit in [1_usize, 15, 16, 17] {
                        let run = |prefetch: bool| {
                            let (mut input, mut matched) = (vec![], vec![]);
                            let mut offset = AtRow { row: 0 };
                            let mut steps = vec![];
                            loop {
                                let next = if prefetch {
                                    matched_heads::<_, u32, true>(
                                        &map.table,
                                        &map.next,
                                        !map.chained,
                                        &probe,
                                        v,
                                        limit,
                                        offset,
                                        &mut input,
                                        &mut matched,
                                    )
                                } else {
                                    matched_heads::<_, u32, false>(
                                        &map.table,
                                        &map.next,
                                        !map.chained,
                                        &probe,
                                        v,
                                        limit,
                                        offset,
                                        &mut input,
                                        &mut matched,
                                    )
                                };
                                steps.push((input.clone(), matched.clone(), next));
                                match next {
                                    Some(resume) => offset = resume,
                                    None => return steps,
                                }
                            }
                        };
                        assert_eq!(run(true), run(false), "len {len} limit {limit}");
                    }
                }
                assert_eq!(
                    contained_heads::<_, true>(&map.table, &probe),
                    contained_heads::<_, false>(&map.table, &probe)
                );
            }
        }
    }
}
