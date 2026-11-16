// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Allocation-lifetime accounting for batches retained by one reservation.
//!
//! The buffer identity and reference counting are upstream's
//! [`RecordBatchMemoryTracker`]: a buffer is
//! charged its full allocation capacity once, while any retained batch still
//! references it. This module adds the ownership that those counters leave to
//! the caller:
//!
//! - [`SharedBatchReservation`] shares one tracker and one reservation between
//!   concurrent producers and consumers (for example the input tasks and the
//!   output stream of one repartition output channel).
//! - A [`BatchReservation`] ticket keeps its batch's buffers alive until it has
//!   released them, so the tracker's contract (release before the buffers are
//!   dropped, because buffers are identified by address) holds by
//!   construction, also when a queued batch is dropped by cancellation.

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use datafusion_common::Result;
use datafusion_common::utils::memory::RecordBatchMemoryTracker;
use datafusion_execution::memory_pool::MemoryReservation;
use parking_lot::Mutex;

/// One reservation charged for the unique buffers of every batch that holds a
/// [`BatchReservation`] from it, including native array object overhead.
#[derive(Debug, Clone)]
pub(crate) struct SharedBatchReservation(Arc<Mutex<State>>);

#[derive(Debug)]
struct State {
    tracker: RecordBatchMemoryTracker,
    reservation: MemoryReservation,
}

impl SharedBatchReservation {
    pub(crate) fn new(reservation: MemoryReservation) -> Self {
        Self(Arc::new(Mutex::new(State {
            tracker: RecordBatchMemoryTracker::new(),
            reservation,
        })))
    }

    /// Charge the buffers of `batch` that no live ticket of this owner retains.
    ///
    /// Admission is transactional: when the pool refuses the growth, nothing
    /// stays counted and the reservation is unchanged.
    pub(crate) fn try_reserve(&self, batch: &RecordBatch) -> Result<BatchReservation> {
        {
            let mut state = self.0.lock();
            let additional = state.tracker.count_batch_with_array_overhead(batch);
            if let Err(error) = state.reservation.try_grow(additional) {
                state.tracker.uncount_batch_with_array_overhead(batch);
                return Err(error);
            }
        }
        Ok(BatchReservation {
            owner: self.clone(),
            batch: batch.clone(),
        })
    }

    #[cfg(test)]
    pub(crate) fn reserved(&self) -> usize {
        self.0.lock().reservation.size()
    }
}

/// A live charge for one batch's buffers. Dropping it releases the buffers
/// that no other live ticket of the same owner retains.
#[derive(Debug)]
pub(crate) struct BatchReservation {
    owner: SharedBatchReservation,
    // Keeps the buffers alive, so their addresses cannot be reused by another
    // allocation before this ticket has released them from the tracker.
    batch: RecordBatch,
}

impl Drop for BatchReservation {
    fn drop(&mut self) {
        let mut state = self.owner.0.lock();
        let released = state.tracker.uncount_batch_with_array_overhead(&self.batch);
        state.reservation.shrink(released);
    }
}

/// A batch together with its charge, for queues whose items can be dropped
/// unconsumed (send errors, cancellation, LIMIT). It is the ticket itself, so
/// a queued item stays one batch wide.
#[derive(Debug)]
pub(crate) struct ReservedBatch(BatchReservation);

impl ReservedBatch {
    pub(crate) fn new(ticket: BatchReservation) -> Self {
        Self(ticket)
    }

    /// Hand the batch to its consumer, which reserves it again if it retains
    /// it. This releases this owner's charge for buffers no queued batch
    /// still retains; the returned batch keeps the buffers alive.
    pub(crate) fn into_batch(self) -> RecordBatch {
        self.0.batch.clone()
    }
}

#[cfg(test)]
mod tests;
