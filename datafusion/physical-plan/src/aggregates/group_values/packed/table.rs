// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! The packed-key kernel: complete `u64` group keys to dense group ids.
//!
//! A key is the whole group identity, so a lookup is one hash and one compare.
//! Keys below `2^DIRECT_BITS` index a flat table without hashing. The table
//! owns no Arrow types; any engine that encodes its keys into `u64` can share
//! it. Group ids follow first appearance and are stored in key order.

use std::mem::size_of;

/// Key domains of at most this many bits can use direct indexing.
pub const DIRECT_BITS: u32 = 16;
/// A direct table of at most this many entries is always allowed; a larger
/// one must not exceed `DIRECT_PER_GROUP` entries per expected group, so it
/// costs no more memory than the hash table of those groups.
const DIRECT_FLOOR: usize = 1 << 10;
const DIRECT_PER_GROUP: usize = 8;

/// A vacant entry. Tables are filled with it after allocation; a nonzero
/// marker cannot become a zeroed allocation.
const VACANT: u32 = u32::MAX;
/// Tables above this many bytes prefetch the slots of later rows.
const PREFETCH_BYTES: usize = 1 << 20;
const PREFETCH_DISTANCE: usize = 16;
/// Below this many slots, near-unique keys grow the table 4x.
const FAST_GROWTH_SLOTS: usize = 1 << 22;
/// A new hash table is sized for at most this many expected groups.
const PRESIZE_GROUPS: usize = 4096;
/// Keys of at most this many bits use 8-byte slots.
pub const NARROW_BITS: u32 = 32;

/// A hash slot: a complete key and its group id; vacant slots hold the
/// all-ones group id (group ids stay below `u32::MAX`).
trait Slot: Copy + Send {
    const VACANT: Self;
    fn is_vacant(self) -> bool;
    fn key(self) -> u64;
    fn group(self) -> u64;
    fn new(key: u64, group: u64) -> Self;
}

/// Keys of up to 64 bits: 16 bytes.
impl Slot for [u64; 2] {
    const VACANT: Self = [0, u64::MAX];
    #[inline(always)]
    fn is_vacant(self) -> bool {
        self[1] == u64::MAX
    }
    #[inline(always)]
    fn key(self) -> u64 {
        self[0]
    }
    #[inline(always)]
    fn group(self) -> u64 {
        self[1]
    }
    #[inline(always)]
    fn new(key: u64, group: u64) -> Self {
        [key, group]
    }
}

/// Keys of up to 32 bits: the key in the low half, the group id in the high
/// half; 8 bytes, so a probe, a fill and a growth move half the memory.
impl Slot for u64 {
    const VACANT: Self = u64::MAX;
    #[inline(always)]
    fn is_vacant(self) -> bool {
        self >> 32 == u64::from(u32::MAX)
    }
    #[inline(always)]
    fn key(self) -> u64 {
        self & u64::from(u32::MAX)
    }
    #[inline(always)]
    fn group(self) -> u64 {
        self >> 32
    }
    #[inline(always)]
    fn new(key: u64, group: u64) -> Self {
        key | (group << 32)
    }
}

/// A table of vacant entries, allocated and then filled. Large zeroed
/// allocations take fresh pages that are faulted in on first touch and
/// unmapped when the table grows; the unmapping flushes the TLB of every
/// thread (measured: allocation rose from 5% to 34% of TPC-DS q23's CPU).
fn vacant<T: Clone>(len: usize, entry: T) -> Vec<T> {
    vec![entry; len]
}

/// Open addressing with linear probing, load at most 3/4.
struct Hashed<S> {
    slots: Vec<S>,
}

impl<S: Slot> Hashed<S> {
    fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Intern `rows[start..]` until the table must grow before the next new
    /// group. Returns the first row not interned.
    fn probe(
        &mut self,
        keys: &mut Vec<u64>,
        positions: &mut Vec<u32>,
        rows: &[u64],
        start: usize,
        groups: &mut [usize],
    ) -> usize {
        let mask = self.slots.len() - 1;
        let shift = 64 - self.slots.len().trailing_zeros();
        let limit = load_limit(self.slots.len());
        // Positions are stored as u32: tables of more slots do not prefetch.
        let prefetch = self.slots.len() * size_of::<S>() > PREFETCH_BYTES
            && self.slots.len() <= u32::MAX as usize + 1;
        let tail = &rows[start..];
        if prefetch {
            // Positions of later rows hide the latency of a table in DRAM.
            positions.clear();
            positions.extend(tail.iter().map(|&k| position(k, shift) as u32));
        }
        let slots = &mut self.slots;
        for (offset, &key) in tail.iter().enumerate() {
            let mut at = if prefetch {
                if let Some(&ahead) = positions.get(offset + PREFETCH_DISTANCE) {
                    prefetch_slot(slots, ahead as usize);
                }
                positions[offset] as usize
            } else {
                position(key, shift)
            };
            let group = loop {
                let slot = slots[at];
                if slot.is_vacant() {
                    if keys.len() >= limit {
                        return start + offset;
                    }
                    let group = keys.len() as u64;
                    slots[at] = S::new(key, group);
                    keys.push(key);
                    break group;
                }
                if slot.key() == key {
                    break slot.group();
                }
                at = (at + 1) & mask;
            };
            groups[start + offset] = group as usize;
        }
        rows.len()
    }

    /// Move every group into a table of `capacity` slots. Slots are indexed
    /// by the top bits of the hash, so the old slots, read in order, map to
    /// nearly ascending new slots: the reinsertion streams through memory
    /// instead of writing one random cache line per group.
    fn grow(&mut self, keys: &mut Vec<u64>, capacity: usize) {
        reserve_keys(keys, capacity);
        let old = std::mem::replace(&mut self.slots, vacant(capacity, S::VACANT));
        if old.is_empty() {
            self.insert_all(keys);
            return;
        }
        let mask = capacity - 1;
        let shift = 64 - capacity.trailing_zeros();
        for slot in old.into_iter().filter(|s| !s.is_vacant()) {
            let mut at = position(slot.key(), shift);
            while !self.slots[at].is_vacant() {
                at = (at + 1) & mask;
            }
            self.slots[at] = slot;
        }
    }

    fn insert_all(&mut self, keys: &[u64]) {
        let mask = self.slots.len() - 1;
        let shift = 64 - self.slots.len().trailing_zeros();
        for (group, &key) in keys.iter().enumerate() {
            let mut at = position(key, shift);
            while !self.slots[at].is_vacant() {
                at = (at + 1) & mask;
            }
            self.slots[at] = S::new(key, group as u64);
        }
    }

    /// Re-index every group in a table of at least `wanted` slots, reusing
    /// the allocation when it is large enough. The keys changed: they are
    /// reinserted, not the old slots' keys.
    fn rebuild(&mut self, keys: &mut Vec<u64>, wanted: usize) {
        if self.slots.len() >= wanted {
            self.slots.fill(S::VACANT);
        } else {
            self.slots = vacant(wanted, S::VACANT);
            reserve_keys(keys, wanted);
        }
        self.insert_all(keys);
    }

    fn clear(&mut self) {
        self.slots.fill(S::VACANT);
    }

    fn clear_shrink(&mut self, rows: usize) {
        let needed = capacity_for(rows);
        if rows == 0 {
            self.slots = Vec::new();
        } else if self.slots.len() > needed {
            self.slots = vacant(needed, S::VACANT);
        } else {
            self.slots.fill(S::VACANT);
        }
    }

    fn bytes(&self) -> usize {
        self.slots.capacity() * size_of::<S>()
    }
}

/// The keys grow with the table, exactly to its load limit: an independently
/// doubling vector would hold up to twice the keys' memory and retain more
/// address space (4 GiB over the TPC-DS SF100 suite).
fn reserve_keys(keys: &mut Vec<u64>, capacity: usize) {
    let limit = load_limit(capacity);
    if limit > keys.capacity() {
        keys.reserve_exact(limit - keys.len());
    }
}

/// A map from complete packed keys to dense group ids.
pub struct PackedKeyTable {
    /// The key of each group, in group-id order.
    keys: Vec<u64>,
    /// Direct mode: one stored group id per key value.
    direct: Vec<u32>,
    /// Hash mode, keys above `NARROW_BITS` bits.
    wide: Hashed<[u64; 2]>,
    /// Hash mode, keys of at most `NARROW_BITS` bits.
    narrow: Hashed<u64>,
    /// Hash mode: precomputed slot indices of one batch, for prefetching.
    positions: Vec<u32>,
    bits: u32,
    direct_mode: bool,
    /// The expected number of groups given at the last relayout.
    expected: usize,
}

impl Default for PackedKeyTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Run `$body` with `$h` bound to the active hash table.
macro_rules! hashed {
    ($self:ident, $h:ident => $body:expr) => {
        if $self.bits <= NARROW_BITS {
            let $h = &mut $self.narrow;
            $body
        } else {
            let $h = &mut $self.wide;
            $body
        }
    };
}

impl PackedKeyTable {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            direct: Vec::new(),
            wide: Hashed::new(),
            narrow: Hashed::new(),
            positions: Vec::new(),
            bits: 0,
            direct_mode: true,
            expected: 0,
        }
    }

    /// The number of groups.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The keys of all groups in group-id order.
    pub fn keys(&self) -> &[u64] {
        &self.keys
    }

    /// The key width in bits. Every key must be below `2^bits`.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn is_direct(&self) -> bool {
        self.direct_mode
    }

    /// Change the key width and replace every group key. `rekey` maps each
    /// old key to its new key; it must be injective. Group ids are unchanged.
    /// `expected` estimates the number of groups, which bounds the size of a
    /// direct table.
    pub fn relayout(
        &mut self,
        bits: u32,
        expected: usize,
        mut rekey: impl FnMut(u64) -> u64,
    ) {
        assert!(bits <= 64);
        self.expected = expected;
        self.direct_mode = bits <= DIRECT_BITS
            && (1usize << bits)
                <= DIRECT_FLOOR.max(expected.saturating_mul(DIRECT_PER_GROUP));
        for key in self.keys.iter_mut() {
            *key = rekey(*key);
        }
        self.bits = bits;
        self.rebuild();
    }

    /// Assign a group id to each key, appending new groups in order.
    pub fn intern(&mut self, keys: &[u64], groups: &mut Vec<usize>) {
        debug_assert!(
            self.bits == 64 || keys.iter().all(|&k| k >> self.bits == 0),
            "key above its declared width"
        );
        // Every probe stores an id, so reuse the initialized output prefix.
        groups.resize(keys.len(), 0);
        if self.is_direct() {
            self.intern_direct(keys, groups);
        } else {
            hashed!(self, h => Self::intern_hashed(h, &mut self.keys, &mut self.positions, keys, groups));
        }
    }

    fn intern_direct(&mut self, keys: &[u64], groups: &mut [usize]) {
        if self.direct.is_empty() {
            self.direct = vacant(1 << self.bits, VACANT);
        }
        let direct = &mut self.direct;
        for (group, &key) in groups.iter_mut().zip(keys) {
            let entry = &mut direct[key as usize];
            if *entry == VACANT {
                *entry = self.keys.len() as u32;
                self.keys.push(key);
            }
            *group = *entry as usize;
        }
    }

    fn intern_hashed<S: Slot>(
        h: &mut Hashed<S>,
        stored: &mut Vec<u64>,
        positions: &mut Vec<u32>,
        keys: &[u64],
        groups: &mut [usize],
    ) {
        if h.slots.is_empty() {
            h.grow(stored, capacity_for(stored.len()));
        }
        let mut row = 0;
        while row < keys.len() {
            let before = stored.len();
            let start = row;
            row = h.probe(stored, positions, keys, row, groups);
            if row < keys.len() {
                // The table reached its load limit. When most rows of this
                // run were new groups (near-unique keys), grow four times
                // while the table is small: half the fills and reinsertions.
                let fresh = (stored.len() - before) * 2 > row - start;
                let factor = if fresh && h.slots.len() < FAST_GROWTH_SLOTS {
                    4
                } else {
                    2
                };
                let capacity = h.slots.len() * factor;
                h.grow(stored, capacity);
            }
        }
    }

    /// Re-index every group after a relayout. Allocations are reused: the
    /// hash slots that `clear_shrink` retained stay allocated even while the
    /// table is direct, so a flush-and-refill cycle does not regrow them.
    fn rebuild(&mut self) {
        if self.is_direct() {
            if self.direct.len() == 1 << self.bits {
                self.direct.fill(VACANT);
            } else {
                // Allocated on the first intern when there are no groups.
                self.direct = Vec::new();
            }
            if !self.keys.is_empty() {
                if self.direct.is_empty() {
                    self.direct = vacant(1 << self.bits, VACANT);
                }
                for (group, &key) in self.keys.iter().enumerate() {
                    self.direct[key as usize] = group as u32;
                }
            }
        } else {
            self.direct = Vec::new();
            // Size for the expected groups (bounded: a guess must not cost
            // a large table), so a new table does not grow from 16 slots.
            let wanted =
                capacity_for(self.keys.len().max(self.expected.min(PRESIZE_GROUPS)));
            // The other slot size releases its memory (a direct table keeps
            // both, so a retained table survives a pass through direct mode).
            if self.bits <= NARROW_BITS {
                self.wide = Hashed::new();
            } else {
                self.narrow = Hashed::new();
            }
            hashed!(self, h => h.rebuild(&mut self.keys, wanted));
        }
    }

    /// Remove the first `n` groups; later groups shift down by `n`.
    pub fn drop_first(&mut self, n: usize) {
        let n = n.min(self.keys.len());
        if n == self.keys.len() {
            self.clear();
            return;
        }
        self.keys.drain(..n);
        if self.is_direct() {
            let n = n as u32;
            for entry in self.direct.iter_mut() {
                *entry = match entry.checked_sub(n) {
                    Some(group) if *entry != VACANT => group,
                    _ => VACANT,
                };
            }
        } else {
            hashed!(self, h => {
                h.clear();
                h.insert_all(&self.keys);
            });
        }
    }

    /// Remove all groups, keeping allocations and the key width.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.direct.fill(VACANT);
        self.wide.clear();
        self.narrow.clear();
    }

    /// Remove all groups and keep capacity for at most `rows` groups.
    pub fn clear_shrink(&mut self, rows: usize) {
        self.keys.clear();
        self.keys.shrink_to(rows);
        self.positions = Vec::new();
        self.direct = Vec::new();
        self.wide.clear_shrink(rows);
        self.narrow.clear_shrink(rows);
    }

    /// Release scratch and unused slots without changing keys or group ids.
    pub fn shrink_to_fit(&mut self) {
        self.positions = Vec::new();
        self.wide = Hashed::new();
        self.narrow = Hashed::new();
        self.expected = 0;
        if self.keys.is_empty() {
            self.direct = Vec::new();
            self.keys = Vec::new();
            return;
        }
        self.rebuild();
        self.keys.shrink_to_fit();
    }

    /// Bytes allocated.
    pub fn size(&self) -> usize {
        self.keys.capacity() * size_of::<u64>()
            + self.direct.capacity() * size_of::<u32>()
            + self.wide.bytes()
            + self.narrow.bytes()
            + self.positions.capacity() * size_of::<u32>()
    }
}

/// Groups a hash table of `capacity` slots holds before it grows.
fn load_limit(capacity: usize) -> usize {
    capacity / 4 * 3
}

/// The smallest power-of-two capacity that holds `groups` groups.
fn capacity_for(groups: usize) -> usize {
    let mut capacity = 16;
    while load_limit(capacity) < groups.max(1) {
        capacity *= 2;
    }
    capacity
}

/// The home slot of a key in a table of `2^(64 - shift)` slots.
#[inline(always)]
fn position(key: u64, shift: u32) -> usize {
    #[cfg(feature = "force_hash_collisions")]
    {
        let _ = (key, shift);
        0
    }
    #[cfg(not(feature = "force_hash_collisions"))]
    {
        // One xor-shift-multiply: the xor-shift moves the high columns'
        // bits down, the multiply spreads every bit into the top bits that
        // select a slot. The two-round splitmix64 finalizer measured 1 to 2
        // ns per row slower on cache-resident tables (its latency is on the
        // probe's critical path) and no better on near-unique pairs.
        let mixed = (key ^ (key >> 31)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (mixed >> shift) as usize
    }
}

#[inline(always)]
fn prefetch_slot<S>(slots: &[S], at: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        // SAFETY: `at` is a precomputed position below `slots.len()`, so the
        // address is inside the allocation. A prefetch never faults or reads
        // into the program; it only hints the cache.
        unsafe { _mm_prefetch::<_MM_HINT_T0>(slots.as_ptr().add(at).cast::<i8>()) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (slots, at);
    }
}
