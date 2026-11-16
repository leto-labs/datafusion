// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Exchange batches are charged for their unique retained buffers.

use super::*;
use arrow::array::StringViewArray;
use datafusion_common::utils::memory::get_record_batch_memory_size;
use datafusion_execution::runtime_env::RuntimeEnvBuilder;

#[tokio::test]
async fn memory_accounting_exchange_repeated_view_buffers() -> Result<()> {
    // concat of views can retain the same backing buffer many times.
    let text = "a".repeat(64 * 1024);
    let values = StringViewArray::from(vec![text.as_str()]);
    let repeated = arrow::compute::concat(&[&values, &values, &values, &values])?;
    let batch = RecordBatch::try_from_iter(vec![("s", repeated)])?;
    let expected = get_record_batch_memory_size(&batch);
    let runtime = RuntimeEnvBuilder::new().build_arc()?;
    let reservation = SharedBatchReservation::new(
        MemoryConsumer::new("exchange fixture")
            .with_can_spill(true)
            .register(&runtime.memory_pool),
    );
    let (mut senders, mut receivers) = channels(1);
    let metrics = ExecutionPlanMetricsSet::new();
    let manager = Arc::new(SpillManager::new(
        Arc::clone(&runtime),
        SpillMetrics::new(&metrics, 0),
        batch.schema(),
    ));
    let (writer, _reader) = spill_pool::mpsc_channel(1024 * 1024, manager);
    let mut output = OutputChannel {
        sender: senders.pop().unwrap(),
        reservation,
        spill_writer: writer.new_sink(),
        shared_coalescer: None,
    };
    output.send(batch).await.unwrap();
    let charged = runtime.memory_pool.reserved();
    eprintln!("retained_buffers={expected} reserved={charged}");
    assert!(
        (expected..=expected + 4096).contains(&charged),
        "charged {charged}, retained {expected}"
    );
    drop(receivers.pop());
    drop(output);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn memory_accounting_exchange_abandoned_send_and_queue_drop() -> Result<()> {
    use futures::poll;
    let text = "b".repeat(64 * 1024);
    let batch = RecordBatch::try_from_iter([(
        "s",
        Arc::new(StringViewArray::from(vec![text.as_str()])) as arrow::array::ArrayRef,
    )])?;
    let runtime = RuntimeEnvBuilder::new().build_arc()?;
    let reservation = SharedBatchReservation::new(
        MemoryConsumer::new("pending send fixture")
            .with_can_spill(true)
            .register(&runtime.memory_pool),
    );
    let (mut senders, mut receivers) = channels(1);
    let metrics = ExecutionPlanMetricsSet::new();
    let manager = Arc::new(SpillManager::new(
        Arc::clone(&runtime),
        SpillMetrics::new(&metrics, 0),
        batch.schema(),
    ));
    let (writer, _reader) = spill_pool::mpsc_channel(1024 * 1024, manager);
    let mut output = OutputChannel {
        sender: senders.pop().unwrap(),
        reservation,
        spill_writer: writer.new_sink(),
        shared_coalescer: None,
    };
    output.send(batch.clone()).await.unwrap();
    let queued = runtime.memory_pool.reserved();
    // A pending send of distinct buffers holds its own charge. (Sending the
    // same batch again adds nothing: its buffers and array objects are
    // already charged once.)
    let other_text = "d".repeat(64 * 1024);
    let other = RecordBatch::try_from_iter([(
        "s",
        Arc::new(StringViewArray::from(vec![other_text.as_str()]))
            as arrow::array::ArrayRef,
    )])?;
    let mut pending = Box::pin(output.send(other));
    assert!(poll!(&mut pending).is_pending());
    assert!(runtime.memory_pool.reserved() > queued);
    drop(pending);
    assert_eq!(runtime.memory_pool.reserved(), queued);
    drop(receivers.pop());
    assert_eq!(runtime.memory_pool.reserved(), 0);
    // The error payload owns its buffers until the caller destroys it.
    let error = output.send(batch).await.unwrap_err();
    assert!(runtime.memory_pool.reserved() > 0);
    drop(error);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn memory_accounting_exchange_keeps_shared_queued_sibling() -> Result<()> {
    let text = "c".repeat(64 * 1024);
    let batch = RecordBatch::try_from_iter([(
        "s",
        Arc::new(StringViewArray::from(vec![text.as_str()])) as arrow::array::ArrayRef,
    )])?;
    let expected = get_record_batch_memory_size(&batch);
    let runtime = RuntimeEnvBuilder::new().build_arc()?;
    let reservation = SharedBatchReservation::new(
        MemoryConsumer::new("sibling fixture")
            .with_can_spill(true)
            .register(&runtime.memory_pool),
    );
    // Keep the other channel empty, opening the shared backpressure gate.
    let (mut senders, mut receivers) = channels(2);
    let mut receiver = receivers.pop().unwrap();
    let metrics = ExecutionPlanMetricsSet::new();
    let manager = Arc::new(SpillManager::new(
        Arc::clone(&runtime),
        SpillMetrics::new(&metrics, 0),
        batch.schema(),
    ));
    let (writer, _reader) = spill_pool::mpsc_channel(1024 * 1024, manager);
    let mut output = OutputChannel {
        sender: senders.pop().unwrap(),
        reservation,
        spill_writer: writer.new_sink(),
        shared_coalescer: None,
    };
    output.send(batch.clone()).await.unwrap();
    output.send(batch.clone()).await.unwrap();
    assert!((expected..=expected + 4096).contains(&runtime.memory_pool.reserved()));
    let first = receiver.recv().await.unwrap().unwrap().unwrap();
    let RepartitionBatch::Memory(first) = first else {
        panic!("unexpected spill")
    };
    assert_eq!(first.into_batch(), batch);
    assert!(runtime.memory_pool.reserved() >= expected);
    drop(receiver.recv().await);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn memory_accounting_exchange_real_pressure_still_spills() -> Result<()> {
    let text = "d".repeat(64 * 1024);
    let batch = RecordBatch::try_from_iter([(
        "s",
        Arc::new(StringViewArray::from(vec![text.as_str()])) as arrow::array::ArrayRef,
    )])?;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_limit(32 * 1024, 1.0)
        .build_arc()?;
    let reservation = SharedBatchReservation::new(
        MemoryConsumer::new("spill fixture")
            .with_can_spill(true)
            .register(&runtime.memory_pool),
    );
    let (mut senders, mut receivers) = channels(1);
    let mut receiver = receivers.pop().unwrap();
    let metrics = ExecutionPlanMetricsSet::new();
    let manager = Arc::new(SpillManager::new(
        Arc::clone(&runtime),
        SpillMetrics::new(&metrics, 0),
        batch.schema(),
    ));
    let (writer, mut reader) = spill_pool::mpsc_channel(1024 * 1024, manager);
    let mut output = OutputChannel {
        sender: senders.pop().unwrap(),
        reservation,
        spill_writer: writer.new_sink(),
        shared_coalescer: None,
    };
    output.send(batch.clone()).await.unwrap();
    assert!(matches!(
        receiver.recv().await.unwrap().unwrap().unwrap(),
        RepartitionBatch::Spilled
    ));
    drop(output);
    drop(writer);
    assert_eq!(reader.next().await.unwrap()?, batch);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}
