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

use super::*;
use arrow::array::{ArrayRef, Int32Array, StringViewArray};
use datafusion_common::utils::memory::RecordBatchMemoryCounter;
use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

fn fixture() -> RecordBatch {
    let text = "x".repeat(64 * 1024);
    let values = StringViewArray::from(vec![text.as_str(), "a second external string"]);
    let values = arrow::compute::concat(&[&values, &values, &values]).unwrap();
    RecordBatch::try_from_iter([("s", values)]).unwrap()
}

fn owner(limit: usize) -> (SharedBatchReservation, Arc<dyn MemoryPool>) {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
    let reservation = MemoryConsumer::new("test buffer owner").register(&pool);
    (SharedBatchReservation::new(reservation), pool)
}

fn retained(batches: &[RecordBatch]) -> usize {
    let mut counter = RecordBatchMemoryCounter::new();
    batches.iter().for_each(|batch| {
        counter.count_batch(batch);
    });
    counter.memory_usage()
}

#[test]
fn memory_accounting_shared_views_and_out_of_order_release() -> Result<()> {
    let batch = fixture();
    let (owner, pool) = owner(1 << 20);
    let expected = retained(std::slice::from_ref(&batch));
    let first = owner.try_reserve(&batch)?;
    let second = owner.try_reserve(&batch)?;
    let third = owner.try_reserve(&batch)?;
    assert!((expected..=expected + 4096).contains(&pool.reserved()));
    drop(second);
    drop(first);
    assert!((expected..=expected + 4096).contains(&pool.reserved()));
    drop(third);
    assert_eq!(pool.reserved(), 0);
    // Reuse the same owner after its registry empties.
    drop(owner.try_reserve(&batch)?);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn memory_accounting_slices_and_nested_dictionary_values() -> Result<()> {
    use arrow::array::{DictionaryArray, ListArray};
    use arrow::datatypes::Int32Type;
    let batch = fixture();
    let values = batch.column(0).clone();
    let dict = Arc::new(DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(vec![Some(0), None, Some(1), Some(0)]),
        values,
    )?);
    let nested = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>([
        Some(vec![Some(1), None]),
        None,
        Some(vec![Some(3)]),
        Some(vec![Some(4)]),
    ]));
    let original = RecordBatch::try_from_iter([
        ("dictionary", dict as ArrayRef),
        ("list", nested as ArrayRef),
    ])?;
    let slices = [original.slice(1, 1), original.slice(2, 2)];
    let expected = retained(&slices);
    assert_eq!(expected, retained(std::slice::from_ref(&original)));
    let (owner, pool) = owner(1 << 20);
    let a = owner.try_reserve(&slices[0])?;
    let b = owner.try_reserve(&slices[1])?;
    assert!((expected..=expected + 4096).contains(&pool.reserved()));
    drop(a);
    assert!(pool.reserved() >= retained(&slices[1..]));
    drop(b);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn memory_accounting_failed_admission_is_transactional() -> Result<()> {
    let batch = fixture();
    let (owner, pool) = owner(128 * 1024);
    let first = owner.try_reserve(&batch)?;
    let before = pool.reserved();
    let text = "y".repeat(256 * 1024);
    let distinct = RecordBatch::try_from_iter([(
        "different",
        Arc::new(StringViewArray::from(vec![text.as_str()])) as ArrayRef,
    )])?;
    assert!(owner.try_reserve(&distinct).is_err());
    assert_eq!(pool.reserved(), before);
    let shared = owner.try_reserve(&batch)?;
    drop(first);
    assert!(pool.reserved() >= retained(std::slice::from_ref(&batch)));
    drop(shared);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn memory_accounting_sorted_output_transfer_and_drop() -> Result<()> {
    let batch = fixture();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let reservation = MemoryConsumer::new("sort fixture").register(&pool);
    reservation.try_grow(512 * 1024)?;
    let mut output = SharedBatchReservation::reserve_all(
        reservation,
        vec![batch.slice(0, 2), batch.slice(2, 2), batch.slice(4, 2)],
    )?;
    let expected = retained(std::slice::from_ref(&batch));
    assert!((expected..=expected + 4096).contains(&pool.reserved()));
    let row = output.remove(0).into_batch();
    assert_eq!(row.num_rows(), 2);
    assert!(pool.reserved() >= expected);
    drop(output); // a LIMIT or cancellation discards the remaining output
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn memory_accounting_distinct_allocations_and_reuse() -> Result<()> {
    let (owner, pool) = owner(4 << 20);
    let first = fixture();
    let second = fixture();
    let expected = retained(&[first.clone(), second.clone()]);
    let a = owner.try_reserve(&first)?;
    let b = owner.try_reserve(&second)?;
    assert!(pool.reserved() >= expected);
    drop((first, a));
    let before = pool.reserved();
    let third = fixture();
    let c = owner.try_reserve(&third)?;
    assert!(pool.reserved() >= before + retained(std::slice::from_ref(&third)));
    drop((b, c));
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn memory_accounting_parallel_shared_tickets() {
    let (owner, pool) = owner(1 << 20);
    let batch = fixture();
    std::thread::scope(|scope| {
        let mut children = Vec::new();
        for _ in 0..8 {
            let owner = owner.clone();
            let batch = batch.clone();
            children.push(scope.spawn(move || {
                for _ in 0..32 {
                    let ticket = owner.try_reserve(&batch).unwrap();
                    std::thread::yield_now();
                    assert!(
                        owner.0.lock().reservation.size() >= retained(std::slice::from_ref(&batch))
                    );
                    drop(ticket);
                }
            }));
        }
        for child in children {
            child.join().unwrap();
        }
    });
    assert_eq!(pool.reserved(), 0);
}
