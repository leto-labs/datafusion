// Added by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! The visited bitmap of a hash join's build side, written without a lock.
//!
//! Joins that emit build rows after the probe (`Left`, `LeftSemi`, `LeftAnti`,
//! `LeftMark`, `Full`) mark every matched build row. Every probe partition
//! marks into one bitmap. A mutex around that bitmap is taken once per probe
//! batch by every partition, with one bit set per matched row inside it. With
//! a small build side and a large probe side (a semi join of a dimension
//! against a fact table) the critical section is a fixed share of each
//! partition's work, so the lock bounds the number of partitions that run.
//!
//! This bitmap sets bits with an atomic OR on 64-bit words. A bit that is
//! already set is only read, so a build row that many probe rows match causes
//! no write and no cache-line transfer after its first match.

use std::sync::atomic::{AtomicU64, Ordering};

use arrow::buffer::{BooleanBuffer, Buffer};

/// One bit per build row, set by any probe partition.
#[derive(Debug)]
pub(super) struct VisitedBitmap {
    words: Box<[AtomicU64]>,
    len: usize,
}

impl VisitedBitmap {
    /// A cleared bitmap for `len` build rows.
    pub(super) fn new(len: usize) -> Self {
        Self {
            words: (0..len.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            len,
        }
    }

    /// The bitmap of a join that does not track visited rows.
    pub(super) fn empty() -> Self {
        Self::new(0)
    }

    /// Bytes that [`Self::new`] allocates for `len` build rows.
    pub(super) fn size_bytes(len: usize) -> usize {
        len.div_ceil(64) * size_of::<AtomicU64>()
    }

    /// Marks build row `index`. Panics when `index` is out of range.
    #[inline]
    pub(super) fn set(&self, index: usize) {
        assert!(
            index < self.len,
            "visited index {index} out of {}",
            self.len
        );
        let word = &self.words[index / 64];
        let bit = 1u64 << (index % 64);
        // The read keeps an already marked row free of a write.
        if word.load(Ordering::Relaxed) & bit == 0 {
            word.fetch_or(bit, Ordering::Release);
        }
    }

    /// The bits set so far. The caller takes it after every probe partition
    /// has reported completion, so the result is final.
    pub(super) fn snapshot(&self) -> BooleanBuffer {
        // Arrow bitmaps are least-significant bit first within each byte.
        let words: Vec<u64> = self
            .words
            .iter()
            .map(|word| word.load(Ordering::Acquire).to_le())
            .collect();
        BooleanBuffer::new(Buffer::from_vec(words), 0, self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn snapshot_matches_marked_rows() {
        for len in [0usize, 1, 63, 64, 65, 1000] {
            let bitmap = VisitedBitmap::new(len);
            let marked: Vec<usize> =
                (0..len).filter(|i| i % 3 == 0 || i % 64 == 63).collect();
            for &index in &marked {
                bitmap.set(index);
                bitmap.set(index);
            }
            let snapshot = bitmap.snapshot();
            assert_eq!(snapshot.len(), len);
            let set: Vec<usize> = snapshot.set_indices().collect();
            assert_eq!(set, marked, "len {len}");
            assert_eq!(VisitedBitmap::size_bytes(len), len.div_ceil(64) * 8);
        }
    }

    #[test]
    fn concurrent_marks_are_all_kept() {
        let len = 10_000;
        let bitmap = Arc::new(VisitedBitmap::new(len));
        let threads: Vec<_> = (0..8)
            .map(|thread| {
                let bitmap = Arc::clone(&bitmap);
                std::thread::spawn(move || {
                    // Threads overlap on every word and on a shared subset of bits.
                    for index in (0..len).filter(|i| i % 8 == thread || i % 5 == 0) {
                        bitmap.set(index);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(bitmap.snapshot().count_set_bits(), len);
    }

    #[test]
    #[should_panic(expected = "out of")]
    fn out_of_range_panics() {
        VisitedBitmap::new(64).set(64);
    }
}
