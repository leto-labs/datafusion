// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Allocation-lifetime accounting for batches retained by one reservation.
//!
//! Buffer::data_ptr is the allocation base (as_ptr moves for slices), and
//! capacity includes the entire pinned allocation. Keep buffer owners in each
//! ticket so an address cannot be reused before its registry reference leaves.
//! Like RecordBatchMemoryCounter, this sizes retained Arrow buffers, not the
//! allocator or the tracker itself. Exchange retains its native array metadata
//! estimates; sorted output retains its native buffer-only accounting scope.

use std::sync::Arc;

use arrow::array::{ArrayData, ArrayRef};
use arrow::buffer::Buffer;
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, HashMap, Result};
use datafusion_execution::memory_pool::MemoryReservation;
use parking_lot::Mutex;

#[derive(Debug, Clone)]
pub(crate) struct SharedBatchReservation(Arc<Mutex<State>>);

#[derive(Debug)]
struct State {
    // allocation base -> (number of live tickets, full allocation capacity)
    buffers: HashMap<usize, (usize, usize)>,
    reservation: MemoryReservation,
}

#[derive(Debug)]
struct BatchBuffers {
    buffers: Vec<Buffer>,
    overhead: usize,
}

fn add_bytes(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or_else(|| {
        DataFusionError::ResourcesExhausted("retained batch memory accounting overflow".to_owned())
    })
}

fn array_metadata_size(array: &ArrayRef, data: &ArrayData) -> usize {
    array
        .get_array_memory_size()
        .saturating_sub(data.get_buffer_memory_size())
}

/// Count retained buffers once while preserving native per-column metadata.
pub(crate) fn get_record_batch_memory_size_with_metadata(batch: &RecordBatch) -> usize {
    batch.columns().iter().fold(
        datafusion_common::utils::memory::get_record_batch_memory_size(batch),
        |size, array| size.saturating_add(array_metadata_size(array, &array.to_data())),
    )
}

impl BatchBuffers {
    fn new(batch: &RecordBatch, include_array_metadata: bool) -> Result<Self> {
        fn visit(data: &ArrayData, buffers: &mut HashMap<usize, Buffer>) {
            for buffer in data
                .buffers()
                .iter()
                .chain(data.nulls().map(|n| n.inner().inner()))
            {
                // Zero-capacity buffers do not pin allocator-owned bytes.
                if buffer.capacity() != 0 {
                    buffers
                        .entry(buffer.data_ptr().as_ptr() as usize)
                        .and_modify(|seen: &mut Buffer| {
                            if buffer.capacity() > seen.capacity() {
                                *seen = buffer.clone();
                            }
                        })
                        .or_insert_with(|| buffer.clone());
                }
            }
            for child in data.child_data() {
                visit(child, buffers);
            }
        }
        let mut buffers = HashMap::new();
        let mut overhead = 0;
        for array in batch.columns() {
            let data = array.to_data();
            // Preserve native array metadata estimates independently of buffers.
            if include_array_metadata {
                overhead = add_bytes(overhead, array_metadata_size(array, &data))?;
            }
            visit(&data, &mut buffers);
        }
        let buffers: Vec<_> = buffers.into_values().collect();
        Ok(Self { buffers, overhead })
    }
}

#[derive(Debug)]
pub(crate) struct BatchReservation {
    owner: SharedBatchReservation,
    buffers: BatchBuffers,
}

#[derive(Debug)]
pub(crate) struct ReservedBatch {
    batch: RecordBatch,
    _ticket: BatchReservation,
}

impl ReservedBatch {
    pub(crate) fn new(batch: RecordBatch, ticket: BatchReservation) -> Self {
        Self {
            batch,
            _ticket: ticket,
        }
    }

    pub(crate) fn into_batch(self) -> RecordBatch {
        self.batch
    }
}

impl State {
    fn additional(&self, batch: &BatchBuffers) -> Result<usize> {
        let mut bytes = batch.overhead;
        for buffer in &batch.buffers {
            let key = buffer.data_ptr().as_ptr() as usize;
            let charged = self.buffers.get(&key).map_or(0, |entry| entry.1);
            bytes = add_bytes(bytes, buffer.capacity().saturating_sub(charged))?;
        }
        Ok(bytes)
    }

    fn insert(&mut self, batch: &BatchBuffers) {
        for buffer in &batch.buffers {
            let key = buffer.data_ptr().as_ptr() as usize;
            let entry = self.buffers.entry(key).or_insert((0, buffer.capacity()));
            // Standard Arrow allocations have invariant capacity. For custom
            // buffer owners reporting different capacities at the same base,
            // retain the largest observed charge until the last ticket leaves.
            entry.1 = entry.1.max(buffer.capacity());
            entry.0 += 1;
        }
    }
}

impl SharedBatchReservation {
    pub(crate) fn new(reservation: MemoryReservation) -> Self {
        Self(Arc::new(Mutex::new(State {
            buffers: HashMap::new(),
            reservation,
        })))
    }

    pub(crate) fn try_reserve(&self, batch: &RecordBatch) -> Result<BatchReservation> {
        let buffers = BatchBuffers::new(batch, true)?;
        {
            let mut state = self.0.lock();
            let additional = state.additional(&buffers)?;
            // No references are inserted until admission succeeds.
            state.reservation.try_grow(additional)?;
            state.insert(&buffers);
        }
        Ok(BatchReservation {
            owner: self.clone(),
            buffers,
        })
    }

    /// Transfer an existing reservation to an entire output sequence without
    /// releasing it and reacquiring capacity between input and output ownership.
    pub(crate) fn reserve_all(
        reservation: MemoryReservation,
        batches: Vec<RecordBatch>,
    ) -> Result<Vec<ReservedBatch>> {
        let owner = Self::new(reservation);
        let mut prepared = Vec::with_capacity(batches.len());
        {
            let mut state = owner.0.lock();
            let mut total = 0;
            for batch in batches {
                let buffers = BatchBuffers::new(&batch, false)?;
                let additional = state.additional(&buffers)?;
                total = add_bytes(total, additional)?;
                state.insert(&buffers);
                prepared.push((batch, buffers));
            }
            state.reservation.try_resize(total)?;
        }
        Ok(prepared
            .into_iter()
            .map(|(batch, buffers)| ReservedBatch {
                batch,
                _ticket: BatchReservation {
                    owner: owner.clone(),
                    buffers,
                },
            })
            .collect())
    }
}

impl Drop for BatchReservation {
    fn drop(&mut self) {
        let mut state = self.owner.0.lock();
        let mut released = self.buffers.overhead;
        for buffer in &self.buffers.buffers {
            let key = buffer.data_ptr().as_ptr() as usize;
            let entry = state.buffers.get_mut(&key).expect("live buffer ticket");
            entry.0 -= 1;
            if entry.0 == 0 {
                released += entry.1;
                state.buffers.remove(&key);
            }
        }
        if state.buffers.is_empty() {
            state.buffers.shrink_to_fit();
        }
        state.reservation.shrink(released);
    }
}

#[cfg(test)]
mod tests;
