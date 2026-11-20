// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::mem::{self, size_of};

use super::GroupIndexView;
use crate::aggregates::group_values::packed::table::prefetch_slot;

// A non-inlined view at this offset cannot exist: each collision list
// consumes more than one byte and the vector cannot exceed isize::MAX bytes.
pub(super) const VACANT_VIEW: GroupIndexView = GroupIndexView(u64::MAX);

const PREFETCH_BYTES: usize = 1024 * 1024;
pub(super) const PREFETCH_DISTANCE: usize = 16;

#[derive(Clone, Copy)]
struct Slot {
    hash: u64,
    view: GroupIndexView,
}

impl Slot {
    const VACANT: Self = Self {
        hash: 0,
        view: VACANT_VIEW,
    };
}

/// Index each precomputed hash once. The column interner resolves
/// full-hash collisions with a list of group indices.
#[derive(Default)]
pub(super) struct HashIndex {
    slots: Vec<Slot>,
    len: usize,
    shift: u32,
}

impl HashIndex {
    fn slots_for(groups: usize) -> usize {
        if groups == 0 {
            return 0;
        }
        (groups + groups.div_ceil(3)).next_power_of_two().max(4)
    }

    fn with_slots(slots: usize) -> Self {
        Self {
            slots: vec![Slot::VACANT; slots],
            len: 0,
            shift: u64::BITS - slots.trailing_zeros(),
        }
    }

    pub(super) fn reserve(&mut self, additional: usize) {
        let slots = Self::slots_for(self.len + additional);
        if slots > self.slots.len() {
            self.rebuild(slots, |_| true);
        }
    }

    #[inline]
    pub(super) fn position(&self, hash: u64) -> usize {
        debug_assert!(!self.slots.is_empty());
        (hash >> self.shift) as usize
    }

    pub(super) fn prepare_positions(&self, hashes: &[u64], positions: &mut Vec<u32>) {
        positions.clear();
        if cfg!(target_arch = "x86_64")
            && self.size() > PREFETCH_BYTES
            && self.slots.len() - 1 <= u32::MAX as usize
        {
            positions.extend(hashes.iter().map(|&hash| self.position(hash) as u32));
        }
    }

    #[inline]
    pub(super) fn prefetch(&self, position: u32) {
        prefetch_slot(&self.slots, position as usize);
    }

    /// Return a matching slot or the first vacant slot in the probe sequence.
    #[inline]
    pub(super) fn probe(&self, hash: u64, mut at: usize) -> Result<usize, usize> {
        let mask = self.slots.len() - 1;
        loop {
            let slot = self.slots[at];
            if slot.view == VACANT_VIEW {
                return Err(at);
            }
            if slot.hash == hash {
                return Ok(at);
            }
            at = (at + 1) & mask;
        }
    }

    #[inline]
    pub(super) fn find(&self, hash: u64) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        self.probe(hash, self.position(hash)).ok()
    }

    #[inline]
    pub(super) fn view(&self, at: usize) -> GroupIndexView {
        self.slots[at].view
    }

    pub(super) fn set_view(&mut self, at: usize, view: GroupIndexView) {
        debug_assert_ne!(view, VACANT_VIEW);
        self.slots[at].view = view;
    }

    #[inline]
    pub(super) fn insert_at(&mut self, at: usize, hash: u64, view: GroupIndexView) {
        debug_assert_eq!(self.slots[at].view, VACANT_VIEW);
        debug_assert_ne!(view, VACANT_VIEW);
        self.slots[at] = Slot { hash, view };
        self.len += 1;
    }

    #[cfg(test)]
    pub(super) fn insert(&mut self, hash: u64, view: GroupIndexView) {
        self.reserve(1);
        self.insert_reserved(hash, view);
    }

    fn insert_reserved(&mut self, hash: u64, view: GroupIndexView) {
        let at = self.probe(hash, self.position(hash)).unwrap_err();
        self.insert_at(at, hash, view);
    }

    pub(super) fn retain(&mut self, keep: impl FnMut(&mut GroupIndexView) -> bool) {
        if self.len != 0 {
            self.rebuild(self.slots.len(), keep);
        }
    }

    fn rebuild(
        &mut self,
        slots: usize,
        mut keep: impl FnMut(&mut GroupIndexView) -> bool,
    ) {
        let old = mem::replace(self, Self::with_slots(slots));
        for mut slot in old.slots {
            if slot.view != VACANT_VIEW && keep(&mut slot.view) {
                self.insert_reserved(slot.hash, slot.view);
            }
        }
    }

    pub(super) fn clear(&mut self) {
        if self.len != 0 {
            self.slots.fill(Slot::VACANT);
            self.len = 0;
        }
    }

    pub(super) fn clear_shrink(&mut self, groups: usize) {
        let slots = Self::slots_for(groups);
        if slots < self.slots.len() {
            *self = Self::with_slots(slots);
        } else {
            self.clear();
        }
    }

    pub(super) fn shrink_to_fit(&mut self) {
        let slots = Self::slots_for(self.len);
        if slots < self.slots.len() {
            self.rebuild(slots, |_| true);
        }
    }

    pub(super) fn size(&self) -> usize {
        self.slots.capacity() * size_of::<Slot>()
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_and_retain_keep_wrapped_probes() {
        let mut table = HashIndex::default();
        // These hashes share their initial slot at the end of each table.
        for (group, low_bits) in [0, 1, 4, 2, 5, 6, 7, 8, 3].into_iter().enumerate() {
            table.insert(
                u64::MAX - low_bits,
                GroupIndexView::new_inlined(group as u64),
            );
            assert_eq!(table.len(), group + 1);
            assert_eq!(table.size(), table.slots.capacity() * 16);
        }
        table.retain(|view| {
            let previous = view.value();
            *view = GroupIndexView::new_inlined(previous + 100);
            previous % 2 == 0
        });
        table.shrink_to_fit();
        assert_eq!(table.size(), 8 * 16);
        for (group, low_bits) in [0, 1, 4, 2, 5, 6, 7, 8, 3].into_iter().enumerate() {
            assert_eq!(
                table
                    .find(u64::MAX - low_bits)
                    .map(|at| table.view(at).value()),
                (group % 2 == 0).then_some(group as u64 + 100),
            );
        }
        table.clear_shrink(2);
        assert!(table.is_empty());
        assert_eq!(table.size(), 4 * 16);
        assert!(table.find(u64::MAX).is_none());
        table.clear_shrink(0);
        assert_eq!(table.size(), 0);
    }
}
