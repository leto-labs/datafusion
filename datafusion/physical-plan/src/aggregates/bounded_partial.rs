// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Cache-sized partial aggregation (Longbow patch).
//!
//! A partial hash table that grows with its input leaves the CPU cache long
//! before the input ends. When the duplicates of a key are close together in
//! the input (for example `GROUP BY l_orderkey` over clustered lineitem), a
//! small table that flushes its partial states whenever it is full gives the
//! same reduction from cache. [`BoundedPartial`] decides when a
//! [`PartialHashAggregateStream`] flushes, and removes the bound when the
//! observed reduction says that an unbounded table would reduce more.
//!
//! [`PartialHashAggregateStream`]: super::hash_stream::PartialHashAggregateStream

use arrow::array::{Array, ArrayRef};
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;

use crate::hash_utils::create_hashes;
use crate::joins::SeededRandomState;
use crate::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder};

/// Session option: the group count at which a partial table flushes; 0 keeps
/// the native unbounded table.
pub(super) const OPTION: &str = "longbow.execution.partial_aggregate_groups";

/// A full table flushes only if it reduced its cycle's input rows at least 2x.
const MAX_CYCLE_RATIO: f64 = 0.5;

/// The bound is removed when the flushed groups exceed the estimated distinct
/// flushed keys by this factor: cycles then re-emit keys that an unbounded
/// table would have merged.
const MAX_DUPLICATE_RATIO: f64 = 1.25;

/// Sketch keys with a seed independent of hash repartitioning (seed 0), whose
/// output partitions share hash bits.
const SKETCH_STATE: SeededRandomState =
    SeededRandomState::with_seed(0x5041_4741_4752_4f55);

/// Reads only the `longbow` extension, once per stream. An absent or
/// unparsable value is 0; the extension validates values when they are set.
pub(super) fn configured_groups(config: &ConfigOptions) -> usize {
    config
        .extensions
        .iter()
        .find(|(prefix, _)| *prefix == "longbow")
        .and_then(|(_, extension)| {
            extension
                .entries()
                .into_iter()
                .find(|entry| entry.key == OPTION)
        })
        .and_then(|entry| entry.value)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// Flush policy for one partial aggregation partition.
pub(super) struct BoundedPartial {
    /// Flush when the table holds at least this many groups; 0 = unbounded.
    max_groups: usize,
    /// Input rows aggregated since the last flush.
    cycle_rows: usize,
    /// Groups emitted by flushes so far.
    flushed_groups: usize,
    flushes: usize,
    sketch: Sketch,
    hashes: Vec<u64>,
    /// Registered on the first flush, so plans that never flush show no new metric.
    metrics: Option<(ExecutionPlanMetricsSet, usize)>,
    flush_count: Option<Count>,
}

impl BoundedPartial {
    pub(super) fn new(
        max_groups: usize,
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
    ) -> Self {
        Self {
            max_groups,
            cycle_rows: 0,
            flushed_groups: 0,
            flushes: 0,
            sketch: Sketch::default(),
            hashes: Vec::new(),
            metrics: (max_groups > 0).then(|| (metrics.clone(), partition)),
            flush_count: None,
        }
    }

    /// The current bound, if any; also the map capacity a flush retains.
    pub(super) fn bound(&self) -> Option<usize> {
        (self.max_groups > 0).then_some(self.max_groups)
    }

    /// Heap allocations retained by the cross-flush sketch.
    pub(super) fn memory_size(&self) -> usize {
        self.hashes.capacity() * size_of::<u64>() + self.sketch.registers.capacity()
    }

    /// Groups already emitted by flushes; with the table's current groups this
    /// is the stream's observed partial output.
    pub(super) fn flushed_groups(&self) -> usize {
        self.flushed_groups
    }

    /// Accounts one aggregated batch and returns whether the table, now holding
    /// `groups` groups, should flush. A full table that reduced its cycle less
    /// than 2x removes the bound instead.
    pub(super) fn after_batch(&mut self, input_rows: usize, groups: usize) -> bool {
        if self.max_groups == 0 {
            return false;
        }
        self.cycle_rows += input_rows;
        if groups < self.max_groups {
            return false;
        }
        if groups as f64 > MAX_CYCLE_RATIO * self.cycle_rows as f64 {
            self.unbound();
            return false;
        }
        true
    }

    /// Records the group key columns of one flushed state batch.
    pub(super) fn record_flush(
        &mut self,
        keys: &[ArrayRef],
        rows: usize,
        chunk: usize,
    ) -> Result<()> {
        let chunk = chunk.max(1);
        let mut offset = 0;
        while offset < rows {
            let length = chunk.min(rows - offset);
            let slices: Vec<ArrayRef> =
                keys.iter().map(|key| key.slice(offset, length)).collect();
            self.hashes.clear();
            self.hashes.resize(length, 0);
            create_hashes(&slices, SKETCH_STATE.random_state(), &mut self.hashes)?;
            self.hashes
                .iter()
                .for_each(|hash| self.sketch.insert(*hash));
            offset += length;
        }
        self.flushed_groups += rows;
        self.flushes += 1;
        self.cycle_rows = 0;
        if let Some((metrics, partition)) = self.metrics.take() {
            self.flush_count =
                Some(MetricBuilder::new(&metrics).counter("partial_flushes", partition));
        }
        if let Some(count) = &self.flush_count {
            count.add(1);
        }
        if self.flushes >= 2
            && self.flushed_groups as f64 > MAX_DUPLICATE_RATIO * self.sketch.estimate()
        {
            self.unbound();
        }
        Ok(())
    }

    pub(super) fn unbound(&mut self) {
        self.max_groups = 0;
        self.sketch = Sketch::default();
        self.hashes = Vec::new();
    }
}

/// HyperLogLog distinct-count sketch with 2^12 one-byte registers (standard
/// error about 1.6%), allocated on first insert.
#[derive(Default)]
struct Sketch {
    registers: Vec<u8>,
}

impl Sketch {
    const BITS: u32 = 12;

    fn insert(&mut self, hash: u64) {
        if self.registers.is_empty() {
            self.registers = vec![0; 1 << Self::BITS];
        }
        // Group hashes of sequential integers can share high bits; the
        // MurmurHash3 finalizer spreads them over the register index.
        let mut hash = hash;
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        hash ^= hash >> 33;
        let index = (hash >> (64 - Self::BITS)) as usize;
        // The guard bit bounds the rank by 64 - BITS + 1.
        let rest = (hash << Self::BITS) | (1 << (Self::BITS - 1));
        let rank = rest.leading_zeros() as u8 + 1;
        let register = &mut self.registers[index];
        *register = (*register).max(rank);
    }

    fn estimate(&self) -> f64 {
        if self.registers.is_empty() {
            return 0.0;
        }
        let m = self.registers.len() as f64;
        let (sum, zeros) =
            self.registers
                .iter()
                .fold((0.0, 0usize), |(sum, zeros), &rank| {
                    (
                        sum + 2f64.powi(-i32::from(rank)),
                        zeros + usize::from(rank == 0),
                    )
                });
        let alpha = 0.7213 / (1.0 + 1.079 / m);
        let raw = alpha * m * m / sum;
        if raw <= 2.5 * m && zeros > 0 {
            m * (m / zeros as f64).ln()
        } else {
            raw
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};

    use super::*;

    fn policy(max_groups: usize) -> BoundedPartial {
        BoundedPartial::new(max_groups, &ExecutionPlanMetricsSet::new(), 0)
    }

    fn keys(values: impl IntoIterator<Item = i64>) -> Vec<ArrayRef> {
        vec![Arc::new(Int64Array::from_iter_values(values))]
    }

    #[test]
    fn sketch_estimates_within_five_percent() {
        for distinct in [100usize, 5_000, 200_000] {
            let mut sketch = Sketch::default();
            let mut hashes = vec![0; distinct];
            let values: ArrayRef =
                Arc::new(Int64Array::from_iter_values(0..distinct as i64));
            create_hashes(&[values], SKETCH_STATE.random_state(), &mut hashes).unwrap();
            // Duplicates never change the estimate.
            for hash in hashes.iter().chain(hashes.iter()) {
                sketch.insert(*hash);
            }
            let error = (sketch.estimate() - distinct as f64).abs() / distinct as f64;
            assert!(error < 0.05, "{distinct}: {}", sketch.estimate());
        }
    }

    #[test]
    fn zero_bound_never_flushes() {
        let mut policy = policy(0);
        assert_eq!(policy.bound(), None);
        assert!(!policy.after_batch(1_000_000, 1_000_000));
        assert!(policy.metrics.is_none());
    }

    #[test]
    fn full_table_flushes_only_after_a_2x_cycle_reduction() {
        let mut policy = policy(1_000);
        assert!(!policy.after_batch(1_500, 999));
        assert!(policy.after_batch(500, 1_000));
        policy.record_flush(&keys(0..1_000), 1_000, 300).unwrap();
        assert_eq!(policy.flushed_groups(), 1_000);
        // The cycle restarts: 1,000 groups from 1,500 rows is a poor reduction.
        assert!(!policy.after_batch(1_500, 1_000));
        assert_eq!(policy.bound(), None);
        assert!(!policy.after_batch(10_000, 5_000));
    }

    #[test]
    fn clustered_keys_keep_the_bound() {
        let mut policy = policy(1_000);
        for cycle in 0..20i64 {
            assert!(policy.after_batch(4_000, 1_000));
            policy
                .record_flush(&keys(cycle * 1_000..(cycle + 1) * 1_000), 1_000, 256)
                .unwrap();
        }
        assert_eq!(policy.bound(), Some(1_000));
        assert_eq!(policy.flushed_groups(), 20_000);
        assert_eq!(policy.flush_count.as_ref().unwrap().value(), 20);
    }

    #[test]
    fn repeated_keys_across_flushes_remove_the_bound() {
        // A 1,200-key domain under a 1,000-group bound: each cycle reduces well,
        // but every flush re-emits the same keys.
        let mut policy = policy(1_000);
        assert!(policy.after_batch(2_500, 1_000));
        policy.record_flush(&keys(0..1_000), 1_000, 1_000).unwrap();
        assert_eq!(policy.bound(), Some(1_000));
        assert!(policy.after_batch(2_500, 1_000));
        policy
            .record_flush(&keys(200..1_200), 1_000, 1_000)
            .unwrap();
        assert_eq!(policy.bound(), None);
    }

    #[test]
    fn multi_column_keys_are_sketched_together() {
        let mut policy = policy(4);
        let first: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
        ];
        assert!(policy.after_batch(8, 4));
        policy.record_flush(&first, 4, 3).unwrap();
        let second: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![3, 3, 4, 4])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
        ];
        assert!(policy.after_batch(8, 4));
        policy.record_flush(&second, 4, 3).unwrap();
        assert_eq!(policy.bound(), Some(4));
    }

    #[test]
    fn configured_groups_does_not_visit_unrelated_namespaces() {
        use datafusion_common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
        use std::any::Any;
        #[derive(Debug)]
        struct Unrelated;
        impl ConfigExtension for Unrelated {
            const PREFIX: &'static str = "unrelated";
        }
        impl ExtensionOptions for Unrelated {
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
            fn cloned(&self) -> Box<dyn ExtensionOptions> {
                Box::new(Self)
            }
            fn set(&mut self, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn entries(&self) -> Vec<ConfigEntry> {
                panic!("unrelated config must not be enumerated")
            }
        }
        let mut config = ConfigOptions::new();
        config.extensions.insert(Unrelated);
        assert_eq!(configured_groups(&config), 0);
    }

    #[test]
    fn configured_groups_reads_the_session_entry() {
        assert_eq!(configured_groups(&ConfigOptions::new()), 0);
    }
}

/// Test-only `longbow` option namespace carrying [`OPTION`].
#[cfg(test)]
pub(super) mod test_option {
    use std::any::Any;

    use datafusion_common::Result;
    use datafusion_common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
    use datafusion_execution::config::SessionConfig;

    use super::OPTION;

    #[derive(Clone, Debug)]
    struct Groups(usize);

    impl ConfigExtension for Groups {
        const PREFIX: &'static str = "longbow";
    }

    impl ExtensionOptions for Groups {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn cloned(&self) -> Box<dyn ExtensionOptions> {
            Box::new(self.clone())
        }
        fn set(&mut self, _: &str, value: &str) -> Result<()> {
            self.0 = value.parse().unwrap();
            Ok(())
        }
        fn entries(&self) -> Vec<ConfigEntry> {
            vec![ConfigEntry {
                key: OPTION.into(),
                value: Some(self.0.to_string()),
                description: "test",
            }]
        }
    }

    /// Returns `config` with the partial group bound set to `groups`.
    pub(in crate::aggregates) fn with_groups(
        mut config: SessionConfig,
        groups: usize,
    ) -> SessionConfig {
        config.options_mut().extensions.insert(Groups(groups));
        config
    }

    #[test]
    fn configured_groups_reads_the_extension_entry() {
        let config = with_groups(SessionConfig::new(), 4096);
        assert_eq!(super::configured_groups(config.options()), 4096);
    }
}
