// Modified by the Longbow authors; see third_party/README.md.

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

//! Owned row bytes without an allocation or converter Arc for short keys.
//! The owning operator retains the converter and parses its own payload bytes.

use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

pub(super) enum CompactRow {
    Inline { len: u8, bytes: [u8; 23] },
    Heap(Box<[u8]>),
}

impl CompactRow {
    pub(super) fn new(bytes: &[u8]) -> Self {
        if bytes.len() <= 23 {
            let mut inline = [0; 23];
            inline[..bytes.len()].copy_from_slice(bytes);
            Self::Inline {
                len: bytes.len() as u8,
                bytes: inline,
            }
        } else {
            Self::Heap(bytes.into())
        }
    }
    pub(super) fn allocated(&self) -> usize {
        match self {
            Self::Inline { .. } => 0,
            Self::Heap(bytes) => bytes.len(),
        }
    }
}
impl AsRef<[u8]> for CompactRow {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Inline { len, bytes } => &bytes[..usize::from(*len)],
            Self::Heap(bytes) => bytes,
        }
    }
}
impl PartialEq for CompactRow {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl Eq for CompactRow {}
impl PartialOrd for CompactRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for CompactRow {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}
impl Hash for CompactRow {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash(row: &CompactRow) -> u64 {
        let mut state = DefaultHasher::new();
        row.hash(&mut state);
        state.finish()
    }

    #[test]
    fn compact_rows_match_owned_bytes_at_inline_boundary() {
        let inputs: Vec<Vec<u8>> = (0..48)
            .flat_map(|len| [vec![0; len], vec![255; len], (0..len as u8).collect()])
            .collect();
        for a in &inputs {
            let compact = CompactRow::new(a);
            let heap = CompactRow::Heap(a.clone().into_boxed_slice());
            assert_eq!(compact.as_ref(), a);
            assert!(compact == heap);
            assert_eq!(hash(&compact), hash(&heap));
            assert_eq!(compact.allocated(), if a.len() <= 23 { 0 } else { a.len() });
            for b in &inputs {
                assert_eq!(compact.cmp(&CompactRow::new(b)), a.cmp(b));
            }
        }
    }
}
