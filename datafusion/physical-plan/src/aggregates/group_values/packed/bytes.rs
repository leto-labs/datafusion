// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Codes of a byte-view column.
//!
//! A column starts in short mode: while every value has at most four bytes, a
//! value's view is an integer (its bytes, then its length) with offset codes,
//! computed without lookups. Its first longer value, or a range wider than
//! `SHORT_BITS`, converts the column to a dictionary: each distinct value
//! receives the next code, and equal bytes always receive the same code.

#[cfg(not(feature = "force_hash_collisions"))]
use std::hash::BuildHasher;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanBufferBuilder, ByteView, GenericByteViewArray,
};
use arrow::buffer::{Buffer, NullBuffer, ScalarBuffer};
use arrow::datatypes::ByteViewType;
use datafusion_common::Result;
use datafusion_common::hash_utils::RandomState;
use hashbrown::HashTable;

use super::columns::{KeyColumn, Observed, Proposal, Range, bits_for, pack_offsets};

/// Views of at most this length hold their complete value.
const INLINE: u32 = 12;
/// Short mode holds values of at most this many bytes...
const SHORT_LEN: u32 = 4;
/// ...while their range needs at most this many bits.
const SHORT_BITS: u32 = 16;
const MAX_BLOCK: usize = 2 << 20;
const MIN_FRONT: usize = 1 << 6;
const MAX_FRONT: usize = 1 << 12;
/// Input buffers whose external views may stay in the front cache until the
/// next emission. Only small buffers are pinned; every retained buffer is
/// counted in size(). Larger buffers get a fresh batch-local id, so cached
/// descriptors cannot outlive an unreported input allocation.
const PINNED: usize = 8;
const COUNTED_PIN_BYTES: usize = 4 << 20;
/// A column is mostly distinct, and its dictionary pays nothing back, when
/// after `DISTINCT_ROWS` rows more than three quarters of the rows brought a
/// new value and a quarter of them missed the front with external bytes.
const DISTINCT_ROWS: u64 = 1 << 10;
/// A dictionary column is costly when, after `COSTLY_ROWS` rows, it holds
/// more than `COSTLY_VALUES` values and more than a quarter of its rows with
/// external values miss the front cache: each such row hashes and compares
/// its bytes, as the general interner does, before the packed probe. The
/// decision comes within the first batch: every aggregate instance that
/// learns a dictionary only to discard it pays for those rows twice.
const COSTLY_ROWS: u64 = 1 << 12;
const COSTLY_VALUES: usize = 1 << 10;

/// The short-mode integer of a view of at most four bytes: the bytes in the
/// low 32 bits and the length above them.
#[inline(always)]
fn short_value(view: u128) -> u64 {
    (view as u64).rotate_right(32)
}

fn short_view(value: u64) -> u128 {
    u128::from(value.rotate_left(32))
}

/// A front-cache entry: a complete inline view, or an external view whose
/// buffer index names a pinned input buffer. Code 0 marks a vacant entry.
#[derive(Clone, Copy, Default)]
struct Front {
    key: u128,
    code: u32,
}

pub(super) struct BytesColumn<T: ByteViewType> {
    /// Short mode: offset codes of short values. `None` in dictionary mode.
    short: Option<Range>,
    /// Dictionary mode: the new code of each old short code, until `settle`.
    converted: Vec<u32>,
    /// The canonical view of code `c` at `views[c - 1]`; external views
    /// point into `blocks`, or into `current` at index `blocks.len()`.
    views: Vec<u128>,
    blocks: Vec<Buffer>,
    current: Vec<u8>,
    /// Indices into `views` of inline values, by view.
    inline: HashTable<u32>,
    /// Hashes and indices into `views` of external values, by their bytes.
    external: HashTable<(u64, u32)>,
    /// Two-way set-associative cache from views to codes.
    front: Vec<Front>,
    /// Input buffers held so that their address identifies their bytes, with
    /// the id that replaces their index in front keys. Ids are never reused.
    pinned: Vec<(Buffer, u32)>,
    next_id: u32,
    /// The front-key buffer id of each buffer of the observed array.
    buffer_ids: Vec<u32>,
    /// Dictionary mode: the codes of the observed batch.
    codes: Vec<u32>,
    /// Dictionary mode: observed rows, and front misses of external values.
    rows: u64,
    external_misses: u64,
    width: u32,
    state: RandomState,
    phantom: PhantomData<fn() -> T>,
}

impl<T: ByteViewType> BytesColumn<T> {
    pub(super) fn new() -> Self {
        Self {
            short: Some(Range::default()),
            converted: Vec::new(),
            views: Vec::new(),
            blocks: Vec::new(),
            current: Vec::new(),
            inline: HashTable::new(),
            external: HashTable::new(),
            front: Vec::new(),
            pinned: Vec::new(),
            next_id: 0,
            buffer_ids: Vec::new(),
            codes: Vec::new(),
            rows: 0,
            external_misses: 0,
            width: 0,
            state: crate::aggregates::AGGREGATION_HASH_SEED,
            phantom: PhantomData,
        }
    }

    /// The bytes of an external (longer than inline) value.
    fn bytes_of(&self, index: usize) -> &[u8] {
        external_bytes(&self.blocks, &self.current, self.views[index])
    }

    fn hash_bytes(&self, bytes: &[u8]) -> u64 {
        #[cfg(feature = "force_hash_collisions")]
        {
            let _ = bytes;
            0
        }
        #[cfg(not(feature = "force_hash_collisions"))]
        {
            self.state.hash_one(bytes)
        }
    }

    /// The code of a non-NULL value, assigning a new code when it is new.
    /// A hit is a hash, at most two loads and compares, inlined into the row
    /// loop; `shift` selects the front set from the top bits of the hash.
    #[inline(always)]
    fn code_of(
        &mut self,
        array: &GenericByteViewArray<T>,
        row: usize,
        view: u128,
        shift: u32,
    ) -> u32 {
        let key = if view as u32 <= INLINE {
            view
        } else {
            // The pinned id replaces the buffer index (bits 64..96).
            let buffer = (view >> 64) as u32 as usize;
            (view & !(u128::from(u32::MAX) << 64))
                | (u128::from(self.buffer_ids[buffer]) << 64)
        };
        let set = ((mix_view(key) >> shift) as usize) & !1;
        let [a, b] = [self.front[set], self.front[set + 1]];
        if a.key == key && a.code != 0 {
            return a.code;
        }
        if b.key == key && b.code != 0 {
            return b.code;
        }
        self.front_miss(array, row, view, key, set)
    }

    #[cold]
    #[inline(never)]
    fn front_miss(
        &mut self,
        array: &GenericByteViewArray<T>,
        row: usize,
        view: u128,
        key: u128,
        set: usize,
    ) -> u32 {
        let code = if view as u32 <= INLINE {
            self.code_of_inline(view)
        } else {
            self.external_misses += 1;
            self.code_of_external(array.value(row).as_ref())
        };
        // Fill a vacant way, else replace the first and keep the second as
        // the more recent survivor.
        let way = if self.front[set].code == 0 {
            set
        } else if self.front[set + 1].code == 0 {
            set + 1
        } else {
            self.front[set + 1] = self.front[set];
            set
        };
        self.front[way] = Front { key, code };
        code
    }

    fn code_of_inline(&mut self, view: u128) -> u32 {
        let hash = inline_hash(&self.state, view);
        let views = &self.views;
        if let Some(&index) = self.inline.find(hash, |&i| views[i as usize] == view) {
            return index + 1;
        }
        let index = self.views.len() as u32;
        self.views.push(view);
        let (views, state) = (&self.views, &self.state);
        self.inline
            .insert_unique(hash, index, |&i| inline_hash(state, views[i as usize]));
        index + 1
    }

    fn code_of_external(&mut self, bytes: &[u8]) -> u32 {
        let hash = self.hash_bytes(bytes);
        if let Some(&(_, index)) = self.external.find(hash, |&(h, i)| {
            h == hash && self.bytes_of(i as usize) == bytes
        }) {
            return index + 1;
        }
        if !self.current.is_empty() && self.current.len() + bytes.len() > MAX_BLOCK {
            self.freeze();
        }
        let view = ByteView::new(bytes.len() as u32, &bytes[..4])
            .with_buffer_index(self.blocks.len() as u32)
            .with_offset(self.current.len() as u32);
        self.current.extend_from_slice(bytes);
        let index = self.views.len() as u32;
        self.views.push(view.as_u128());
        self.external
            .insert_unique(hash, (hash, index), |&(h, _)| h);
        index + 1
    }

    /// Move the in-progress block into the frozen blocks.
    fn freeze(&mut self) {
        if !self.current.is_empty() {
            let block = std::mem::take(&mut self.current);
            self.blocks.push(Buffer::from_vec(block));
        }
    }

    fn resize_front(&mut self) {
        let wanted = (self.views.len() * 4)
            .next_power_of_two()
            .clamp(MIN_FRONT, MAX_FRONT);
        if self.front.len() < wanted {
            self.front = vec![Front::default(); wanted];
        }
    }

    /// Give every buffer of `array` its front-key id. A pinned buffer keeps
    /// its id; others are pinned with a new id, evicting the oldest. Beyond
    /// `PINNED`, a buffer gets a new id for this batch only.
    fn assign_buffer_ids(&mut self, array: &GenericByteViewArray<T>) {
        self.buffer_ids.clear();
        for (index, buffer) in array.data_buffers().iter().enumerate() {
            let known = self
                .pinned
                .iter()
                .find(|(b, _)| b.as_ptr() == buffer.as_ptr() && b.len() == buffer.len());
            let id = match known {
                Some(&(_, id)) => id,
                None => {
                    if self.next_id == u32::MAX {
                        // Ids are exhausted: forget every cached external key.
                        self.front.fill(Front::default());
                        self.pinned.clear();
                        self.next_id = 0;
                    }
                    let id = self.next_id;
                    self.next_id += 1;
                    if index < PINNED && buffer.capacity() <= COUNTED_PIN_BYTES {
                        if self.pinned.len() == PINNED {
                            self.pinned.remove(0);
                        }
                        self.pinned.push((buffer.clone(), id));
                    }
                    id
                }
            };
            self.buffer_ids.push(id);
        }
    }

    /// Short mode: record the range of the batch, or report that it needs a
    /// dictionary.
    fn observe_short(&mut self, array: &GenericByteViewArray<T>) -> Observed {
        let views = array.views();
        let (mut long, mut lo, mut hi) = (0u32, u64::MAX, 0u64);
        let mut fold = |view: u128| {
            long = long.max(view as u32);
            let value = short_value(view);
            lo = lo.min(value);
            hi = hi.max(value);
        };
        match array.nulls().filter(|n| n.null_count() > 0) {
            None => views.iter().for_each(|&v| fold(v)),
            Some(nulls) => views
                .iter()
                .zip(nulls.inner().iter())
                .filter(|(_, valid)| *valid)
                .for_each(|(&v, _)| fold(v)),
        }
        let range = self.short.as_mut().expect("short mode");
        if lo > hi {
            return Observed::Ready;
        }
        if long > SHORT_LEN {
            return Observed::Convert;
        }
        range.observe(i128::from(lo), i128::from(hi));
        if range.exact_width() > SHORT_BITS {
            return Observed::Convert;
        }
        Observed::Ready
    }

    /// Assign the codes of a batch. Returns false, with the codes
    /// incomplete, as soon as the column proves costly.
    fn observe_dictionary(&mut self, array: &GenericByteViewArray<T>) -> Result<bool> {
        self.resize_front();
        self.assign_buffer_ids(array);
        let mut codes = std::mem::take(&mut self.codes);
        codes.clear();
        codes.try_reserve(array.len()).map_err(|e| {
            datafusion_common::exec_datafusion_err!(
                "failed to reserve {} packed view codes: {e}",
                array.len()
            )
        })?;
        let shift = 64 - self.front.len().trailing_zeros();
        let views = array.views();
        let nulls = array.nulls().filter(|n| n.null_count() > 0);
        let mut costly = false;
        for (start, chunk) in (0..views.len())
            .step_by(DISTINCT_ROWS as usize)
            .map(|s| (s, &views[s..(s + DISTINCT_ROWS as usize).min(views.len())]))
        {
            match nulls {
                None => {
                    for (offset, &view) in chunk.iter().enumerate() {
                        codes.push(self.code_of(array, start + offset, view, shift));
                    }
                }
                Some(nulls) => {
                    for (offset, &view) in chunk.iter().enumerate() {
                        let row = start + offset;
                        codes.push(if nulls.is_valid(row) {
                            self.code_of(array, row, view, shift)
                        } else {
                            0
                        });
                    }
                }
            }
            if self.costly_after(self.rows + (start + chunk.len()) as u64) {
                costly = true;
                break;
            }
        }
        self.codes = codes;
        self.rows += array.len() as u64;
        Ok(!costly)
    }

    /// Whether the dictionary costs more than it saves after `rows` rows.
    fn costly_after(&self, rows: u64) -> bool {
        let misses = self.external_misses * 4 > rows;
        let values = self.views.len() as u64;
        misses
            && ((rows >= COSTLY_ROWS && values > COSTLY_VALUES as u64)
                || (rows >= DISTINCT_ROWS && values * 4 > rows * 3))
    }
}

/// Output buffers of one emission. Views use u32 offsets, so a buffer is
/// closed before it would pass `i32::MAX` bytes.
#[derive(Default)]
struct Emitted {
    done: Vec<Buffer>,
    current: Vec<u8>,
}

impl Emitted {
    /// Copy an external value; returns its view.
    fn push(&mut self, bytes: &[u8]) -> u128 {
        if !self.current.is_empty()
            && self.current.len() + bytes.len() > i32::MAX as usize
        {
            self.done
                .push(Buffer::from_vec(std::mem::take(&mut self.current)));
        }
        let view = ByteView::new(bytes.len() as u32, &bytes[..4])
            .with_buffer_index(self.done.len() as u32)
            .with_offset(self.current.len() as u32);
        self.current.extend_from_slice(bytes);
        view.as_u128()
    }

    fn finish(mut self) -> Vec<Buffer> {
        if !self.current.is_empty() {
            self.done.push(Buffer::from_vec(self.current));
        }
        self.done
    }
}

/// The front cache's hash of a view. The multiply moves every bit of the
/// view into the top bits, which select a front set (`>> shift`); its low
/// bits depend only on the view's low bits. Hash tables that take a bucket
/// from the low bits use [`inline_hash`].
fn mix_view(view: u128) -> u64 {
    #[cfg(feature = "force_hash_collisions")]
    {
        let _ = view;
        0
    }
    #[cfg(not(feature = "force_hash_collisions"))]
    {
        let mixed = (view as u64) ^ ((view >> 64) as u64).rotate_left(23);
        mixed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
    }
}

/// The hash of an inline view in the `inline` table. hashbrown takes a
/// bucket from the low bits of a hash and a tag from its top seven bits, so
/// both ends must depend on the whole view. Short values that share their
/// length and first four bytes (`host_0` to `host_45999`) differ only in the
/// high half of their views; under [`mix_view`] they fell into nine home
/// buckets, and every new value probed a chain of all earlier ones
/// (leto-rle.39).
#[inline]
pub(super) fn inline_hash(state: &RandomState, view: u128) -> u64 {
    #[cfg(feature = "force_hash_collisions")]
    {
        let _ = (state, view);
        0
    }
    #[cfg(not(feature = "force_hash_collisions"))]
    {
        state.hash_one(view)
    }
}

/// The bytes of an external view stored in `blocks`, or in `current` when it
/// names the block after the last frozen one.
fn external_bytes<'a>(blocks: &'a [Buffer], current: &'a [u8], view: u128) -> &'a [u8] {
    let view = ByteView::from(view);
    let (block, offset) = (view.buffer_index as usize, view.offset as usize);
    let data = if block == blocks.len() {
        current
    } else {
        blocks[block].as_slice()
    };
    &data[offset..offset + view.length as usize]
}

impl<T: ByteViewType> KeyColumn for BytesColumn<T> {
    fn observe(&mut self, array: &ArrayRef) -> Result<Observed> {
        let array = array.as_byte_view::<T>();
        if self.short.is_some() {
            return Ok(self.observe_short(array));
        }
        Ok(if self.observe_dictionary(array)? {
            Observed::Ready
        } else {
            Observed::Costly
        })
    }

    fn convert(&mut self, codes: &[u64]) {
        let range = self.short.take().expect("short mode");
        // Old short codes are below 2^SHORT_BITS; record each one's new code.
        self.converted = vec![0; 1 << range.width()];
        for &code in codes {
            if code != 0 && self.converted[code as usize] == 0 {
                let value = (range.base() + (i128::from(code) - 1)) as u64;
                self.converted[code as usize] = self.code_of_inline(short_view(value));
            }
        }
    }

    fn fits(&self) -> bool {
        match &self.short {
            Some(range) => range.fits(),
            None => bits_for(self.views.len() as u128) <= self.width,
        }
    }

    fn propose(&self, spare: bool) -> Proposal {
        match &self.short {
            Some(range) => {
                range.propose(spare, 0, (i128::from(SHORT_LEN) << 32) | 0xFFFF_FFFF)
            }
            None => {
                let width = bits_for(self.views.len() as u128);
                Proposal {
                    width: if spare && width > 0 { width + 2 } else { width },
                    base: 0,
                }
            }
        }
    }

    fn adopt(&mut self, proposal: Proposal) {
        match &mut self.short {
            Some(range) => range.adopt(proposal),
            None => self.width = proposal.width,
        }
    }

    fn translate(&self, code: u64) -> u64 {
        match &self.short {
            Some(range) => range.translate(code),
            None if !self.converted.is_empty() => {
                u64::from(self.converted[code as usize])
            }
            None => code,
        }
    }

    fn settle(&mut self) {
        if let Some(range) = &mut self.short {
            range.settle();
        }
        self.converted = Vec::new();
    }

    fn width(&self) -> u32 {
        match &self.short {
            Some(range) => range.width(),
            None => self.width,
        }
    }

    fn pack(&self, array: &ArrayRef, keys: &mut [u64], shift: u32) {
        if self.width() == 0 {
            return;
        }
        match &self.short {
            Some(range) => {
                let array = array.as_byte_view::<T>();
                pack_offsets(
                    array.views().iter().map(|&v| short_value(v)),
                    array.nulls(),
                    range.base() as u64,
                    keys,
                    shift,
                );
            }
            None => {
                for (key, &code) in keys.iter_mut().zip(self.codes.iter()) {
                    *key |= u64::from(code) << shift;
                }
            }
        }
    }

    fn build(&mut self, codes: &[u64]) -> Result<ArrayRef> {
        let mut views = Vec::with_capacity(codes.len());
        let mut valid = BooleanBufferBuilder::new(codes.len());
        let mut nulls = 0;
        // Only the external values that these groups reference are copied
        // into the emitted buffers, once per distinct code: an emitted array
        // must not keep (or spill, or count) the whole dictionary.
        let mut out = Emitted::default();
        let mut placed = if self.short.is_none() {
            vec![0u128; self.views.len()]
        } else {
            Vec::new()
        };
        for &code in codes {
            if code == 0 {
                views.push(0);
                valid.append(false);
                nulls += 1;
                continue;
            }
            views.push(match &self.short {
                Some(range) => short_view((range.base() + (i128::from(code) - 1)) as u64),
                None => {
                    let index = code as usize - 1;
                    let view = self.views[index];
                    if view as u32 <= INLINE {
                        view
                    } else {
                        if placed[index] == 0 {
                            let bytes = external_bytes(&self.blocks, &self.current, view);
                            placed[index] = out.push(bytes);
                        }
                        placed[index]
                    }
                }
            });
            valid.append(true);
        }
        let nulls = (nulls > 0).then(|| NullBuffer::new(valid.finish()));
        // SAFETY: every non-NULL view is either a short view rebuilt from the
        // bytes and length of an input view of at most four bytes (zero
        // padded, as inline views are), an inline view copied from a valid
        // input array, or an external view whose length, prefix, buffer and
        // offset `Emitted::push` set while copying its bytes into that
        // buffer. All bytes came from arrays of the same type `T`, so they
        // satisfy its validation (UTF-8 for strings). NULL views are 0.
        let array = unsafe {
            GenericByteViewArray::<T>::new_unchecked(
                ScalarBuffer::from(views),
                out.finish().into(),
                nulls,
            )
        };
        Ok(Arc::new(array))
    }

    fn size(&self) -> usize {
        self.views.capacity() * size_of::<u128>()
            + self.blocks.capacity() * size_of::<Buffer>()
            + self.blocks.iter().map(|b| b.capacity()).sum::<usize>()
            + self.pinned.capacity() * size_of::<(Buffer, u32)>()
            + self.buffer_ids.capacity() * size_of::<u32>()
            + self.current.capacity()
            + self.inline.allocation_size()
            + self.external.allocation_size()
            + self.front.capacity() * size_of::<Front>()
            + self.codes.capacity() * size_of::<u32>()
            + self.converted.capacity() * size_of::<u32>()
            // Pinned input buffers stay alive while pinned.
            + self
                .pinned
                .iter()
                .map(|(b, _)| b.capacity())
                .sum::<usize>()
    }

    fn shrink_to_fit(&mut self) {
        self.release();
        self.pinned.shrink_to_fit();
        self.codes = Vec::new();
        self.buffer_ids = Vec::new();
        self.converted.shrink_to_fit();
        self.views.shrink_to_fit();
        self.blocks.shrink_to_fit();
        self.current.shrink_to_fit();
        self.front.shrink_to_fit();
        let views = &self.views;
        self.inline
            .shrink_to_fit(|&index| mix_view(views[index as usize]));
        self.external.shrink_to_fit(|&(hash, _)| hash);
    }

    fn cardinality(&self) -> usize {
        self.views.len()
    }

    fn release(&mut self) {
        // Released ids are never reissued: a buffer seen again gets a new id,
        // so front entries keyed by the old id can no longer match.
        self.pinned.clear();
    }

    fn costly(&self) -> bool {
        self.costly_after(self.rows)
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}
