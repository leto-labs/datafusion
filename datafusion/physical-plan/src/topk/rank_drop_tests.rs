// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Observe the release itself: retained arrays must precede returned credits.
use super::*;
use datafusion_execution::memory_pool::{FairSpillPool, MemoryPool};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_physical_expr::expressions::col;
use futures::StreamExt;
use parking_lot::Mutex;
use std::fmt::{Display, Formatter};
use std::sync::{Weak, atomic::AtomicUsize};

#[derive(Debug)]
struct ReleasePool {
    inner: FairSpillPool,
    arrays: Mutex<Vec<Weak<dyn Array>>>,
    releases: AtomicUsize,
}

impl Display for ReleasePool {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "rank release observer")
    }
}

impl MemoryPool for ReleasePool {
    fn name(&self) -> &str {
        "rank release observer"
    }
    fn register(&self, consumer: &MemoryConsumer) {
        self.inner.register(consumer);
    }
    fn unregister(&self, consumer: &MemoryConsumer) {
        self.inner.unregister(consumer);
    }
    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
    }
    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        self.inner.try_grow(reservation, additional)
    }
    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        if reservation.size() == 0 {
            assert!(
                self.arrays
                    .lock()
                    .iter()
                    .all(|array| array.upgrade().is_none()),
                "rank returned credits before releasing retained arrays"
            );
            self.releases.fetch_add(1, AtomicOrdering::Relaxed);
        }
        self.inner.shrink(reservation, shrink);
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
}

#[tokio::test]
async fn rank_stream_drop_releases_arrays_before_credits() -> Result<()> {
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    for poll_count in [0, 1, 2, 3] {
        let pool = Arc::new(ReleasePool {
            inner: FairSpillPool::new(64 * 1024 * 1024),
            arrays: Mutex::new(Vec::new()),
            releases: AtomicUsize::new(0),
        });
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
            .build_arc()?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let mut state = PartitionedTopKRank::try_new(
            0,
            &schema,
            vec![col("k", &schema)?],
            vec![SortField::new(DataType::Utf8)],
            LexOrdering::new([PhysicalSortExpr::new_default(col("v", &schema)?)])
                .unwrap(),
            1,
            128,
            &runtime,
            &ExecutionPlanMetricsSet::new(),
            None,
        )?;
        // Enough partitions to suspend while moving the consumed map into
        // the ordered emitter. No external strong array references remain.
        for i in 0..256 {
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(StringArray::from(vec![format!("key-{i:04}")])),
                    Arc::new(Int64Array::from(vec![1])),
                ],
            )?;
            state.insert_batch(&batch)?;
            // The upstream store gathers admitted rows into new arrays.
            pool.arrays.lock().extend(
                state
                    .core
                    .store
                    .batches
                    .values()
                    .flat_map(|entry| entry.batch.columns().iter().map(Arc::downgrade)),
            );
        }
        let mut stream = state.emit()?;
        for _ in 0..poll_count {
            let mut next = Box::pin(stream.next());
            // Cover an unpolled stream, partial output, and exhaustion.
            // Upstream resolves the map eagerly before returning the stream.
            let _ = futures::poll!(next.as_mut());
        }
        drop(stream);
        assert_eq!(pool.reserved(), 0);
        assert_eq!(pool.releases.load(AtomicOrdering::Relaxed), 1);
    }
    Ok(())
}
