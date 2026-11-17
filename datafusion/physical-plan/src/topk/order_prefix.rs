// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::cmp::Ordering;

const WIDTH: usize = 15;

/// An owned prefix of the worst retained encoded order. Short orders compare
/// exactly. Equal long prefixes require the ordinary full-row comparison.
#[derive(Debug, Default)]
pub(super) struct OrderPrefix {
    bytes: [u8; WIDTH],
    // WIDTH + 1 marks a truncated order. No borrowed bytes survive the batch.
    len: u8,
}

impl OrderPrefix {
    pub(super) fn new(value: &[u8]) -> Self {
        let n = value.len().min(WIDTH);
        let mut bytes = [0; WIDTH];
        bytes[..n].copy_from_slice(&value[..n]);
        Self {
            bytes,
            len: value.len().min(WIDTH + 1) as u8,
        }
    }

    pub(super) fn compare(&self, value: &[u8]) -> Option<Ordering> {
        let prefix_len = usize::from(self.len).min(WIDTH);
        let common = value.len().min(prefix_len);
        match value[..common].cmp(&self.bytes[..common]) {
            Ordering::Equal => {}
            order => return Some(order),
        }
        if value.len() < prefix_len {
            return Some(Ordering::Less);
        }
        if usize::from(self.len) <= WIDTH {
            Some(value.len().cmp(&usize::from(self.len)))
        } else if value.len() <= WIDTH {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_decisions_match_full_encoded_order() {
        let mut seed = 59_u64;
        for root_len in 0..65 {
            for incoming_len in 0..65 {
                let mut root = vec![0; root_len];
                let mut incoming = vec![0; incoming_len];
                for b in root.iter_mut().chain(incoming.iter_mut()) {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *b = (seed >> 32) as u8;
                }
                for common in [0, 1, 14, 15, 16, 32, 64] {
                    let n = common.min(root_len).min(incoming_len);
                    incoming[..n].copy_from_slice(&root[..n]);
                    let bound = OrderPrefix::new(&root);
                    if let Some(order) = bound.compare(&incoming) {
                        assert_eq!(order, incoming.cmp(&root));
                    } else {
                        assert!(root_len > WIDTH && incoming_len > WIDTH);
                        assert_eq!(&incoming[..WIDTH], &root[..WIDTH]);
                    }
                }
                assert_eq!(
                    OrderPrefix::new(&root)
                        .compare(&root)
                        .unwrap_or(Ordering::Equal),
                    Ordering::Equal
                );
            }
        }
        assert_eq!(size_of::<OrderPrefix>(), 16);
    }
}
