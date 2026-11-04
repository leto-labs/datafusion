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

//! Transfer a build's input-buffer charge to its consolidated live buffers.
use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::utils::memory::RecordBatchMemoryCounter;
use datafusion_common::{Result, internal_datafusion_err};
use datafusion_execution::memory_pool::MemoryReservation;

pub(super) fn reconcile(
    inputs: Vec<RecordBatch>,
    input_bytes: usize,
    batch: &RecordBatch,
    keys: &[ArrayRef],
    reservation: &MemoryReservation,
) -> Result<()> {
    let map_bytes = reservation.size().checked_sub(input_bytes).ok_or_else(|| {
        internal_datafusion_err!("hash-join input buffers exceed their reservation")
    })?;
    // Count the payload and computed keys together. Direct column keys may
    // alias the payload; StringView and nested arrays can retain entire source
    // buffers even when their logical values occupy only a small slice.
    let retained = RecordBatch::try_from_iter(
        batch
            .columns()
            .iter()
            .chain(keys)
            .map(|array| ("buffer", Arc::clone(array))),
    )?;
    let mut counter = RecordBatchMemoryCounter::new();
    counter.count_batch(&retained);
    let bytes = map_bytes
        .checked_add(counter.memory_usage())
        .ok_or_else(|| {
            internal_datafusion_err!("hash-join retained buffer reservation overflow")
        })?;
    drop(retained);
    // Release obsolete input owners before making their charge available to
    // another operator. Buffers aliased by batch/keys are still counted above.
    drop(inputs);
    reservation.try_resize(bytes)
}
