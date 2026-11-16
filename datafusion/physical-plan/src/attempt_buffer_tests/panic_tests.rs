// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;
use std::task::Context;

use datafusion_common::DataFusionError;
use futures::StreamExt;

use crate::ExecutionPlan;
use crate::attempt_tasks::AttemptTasks;
use crate::buffer::MemoryBufferedStream;
use crate::common::spawn_buffered;
use crate::stream::RecordBatchStreamAdapter;

use super::panic_support::{
    ForwardWake, PANIC_DIAGNOSTIC, assert_caught_panic, context, has_caught_panic,
    ordinary_plan, panic_plan,
};
use super::support::{Gate, INPUT_BYTES, assert_pending, batch, bounded, pool, schema};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn buffer_records_panic_before_forwarding() {
    let pool = pool();
    let scope = AttemptTasks::new();
    let gate = Arc::new(Gate::default());
    let _release = gate.release_on_drop();
    let (plan, mut control) = panic_plan(Some(Arc::clone(&gate)), None);
    let mut consumer = plan
        .execute(0, context(&pool, Some(Arc::clone(&scope))))
        .unwrap();
    bounded(&mut control.waiting).await.unwrap();

    let (wake, forwarded) = ForwardWake::new(&scope);
    let waker = futures::task::waker(wake);
    let mut cx = Context::from_waker(&waker);
    assert!(consumer.as_mut().poll_next(&mut cx).is_pending());
    control.start.take().unwrap().send(()).unwrap();
    // This samples inside the sender's wake, not after the producer returns.
    assert!(
        bounded(forwarded).await.unwrap(),
        "forwarding preceded panic recording"
    );
    bounded(&mut control.drop_entered).await.unwrap();
    let error = bounded(consumer.next()).await.unwrap().unwrap_err();
    assert_caught_panic(&error);
    assert!(scope.poll_panicked());
    drop(consumer);
    scope.root_joined();
    let mut drain = Box::pin(scope.drain());
    assert_pending(drain.as_mut());
    assert_eq!(pool.reserved(), INPUT_BYTES);
    gate.release();
    bounded(drain).await.unwrap();
    assert_eq!(control.dropped.try_recv(), Ok(()));
    assert_eq!(pool.reserved(), 0);
    assert!(scope.poll_panicked());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn buffer_caught_panic_survives_discarded_error() {
    let pool = pool();
    let scope = AttemptTasks::new();
    let gate = Arc::new(Gate::default());
    let _release = gate.release_on_drop();
    let (plan, mut control) = panic_plan(Some(Arc::clone(&gate)), None);
    let consumer = plan
        .execute(0, context(&pool, Some(Arc::clone(&scope))))
        .unwrap();
    bounded(&mut control.waiting).await.unwrap();
    control.start.take().unwrap().send(()).unwrap();
    // The caught panic's error is queued. Input destruction follows the catch,
    // and is held while the never-polled receiver discards that error.
    bounded(&mut control.drop_entered).await.unwrap();
    assert!(scope.poll_panicked());
    drop(consumer);
    scope.root_joined();
    let mut drain = Box::pin(scope.drain());
    assert_pending(drain.as_mut());
    gate.release();
    bounded(drain).await.unwrap();
    assert_eq!(control.dropped.try_recv(), Ok(()));
    assert_eq!(pool.reserved(), 0);
    assert!(scope.poll_panicked());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn receiver_gone_during_panicking_input_poll_still_records() {
    let pool = pool();
    let scope = AttemptTasks::new();
    let poll_gate = Arc::new(Gate::default());
    let _release_poll = poll_gate.release_on_drop();
    let drop_gate = Arc::new(Gate::default());
    let _release_drop = drop_gate.release_on_drop();
    let (plan, mut control) =
        panic_plan(Some(Arc::clone(&drop_gate)), Some(Arc::clone(&poll_gate)));
    let consumer = plan
        .execute(0, context(&pool, Some(Arc::clone(&scope))))
        .unwrap();
    bounded(&mut control.waiting).await.unwrap();
    control.start.take().unwrap().send(()).unwrap();
    bounded(&mut control.poll_entered).await.unwrap();
    // The input poll is already executing on the producer's stack. Abort cannot
    // interrupt that synchronous poll. Its real PanicExec unwinds AFTER receiver
    // destruction, so the catch must record even though error forwarding fails.
    drop(consumer);
    scope.root_joined();
    poll_gate.release();
    bounded(&mut control.drop_entered).await.unwrap();
    assert!(scope.poll_panicked());
    let mut drain = Box::pin(scope.drain());
    assert_pending(drain.as_mut());
    assert_eq!(pool.reserved(), INPUT_BYTES);
    drop_gate.release();
    bounded(drain).await.unwrap();
    assert_eq!(control.dropped.try_recv(), Ok(()));
    assert_eq!(pool.reserved(), 0);
    assert!(scope.poll_panicked());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unscoped_buffer_panic_retains_internal_error() {
    let pool = pool();
    let (plan, mut control) = panic_plan(None, None);
    let mut consumer = plan.execute(0, context(&pool, None)).unwrap();
    bounded(&mut control.waiting).await.unwrap();
    control.start.take().unwrap().send(()).unwrap();
    let error = bounded(consumer.next()).await.unwrap().unwrap_err();
    assert!(
        matches!(&error, DataFusionError::Internal(message) if message.starts_with(PANIC_DIAGNOSTIC))
    );
    assert!(!has_caught_panic(&error));
    assert!(bounded(consumer.next()).await.is_none());
    bounded(&mut control.dropped).await.unwrap();
    drop(consumer);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ordinary_buffer_error_is_not_a_panic() {
    for discard in [false, true] {
        let pool = pool();
        let scope = AttemptTasks::new();
        let gate = Arc::new(Gate::default());
        let _release = gate.release_on_drop();
        let (plan, mut control) = ordinary_plan(Some(Arc::clone(&gate)));
        let mut consumer = plan
            .execute(0, context(&pool, Some(Arc::clone(&scope))))
            .unwrap();
        bounded(&mut control.waiting).await.unwrap();
        control.start.take().unwrap().send(()).unwrap();
        bounded(&mut control.drop_entered).await.unwrap();
        assert!(!scope.poll_panicked());
        if discard {
            // Input Drop follows the send. Discard the queued same-text error
            // without ever polling the receiver, just as in the panic arm.
        } else {
            let error = bounded(consumer.next()).await.unwrap().unwrap_err();
            assert!(
                matches!(&error, DataFusionError::Internal(message) if message == PANIC_DIAGNOSTIC)
            );
            assert!(!has_caught_panic(&error));
        }
        drop(consumer);
        scope.root_joined();
        gate.release();
        bounded(scope.drain()).await.unwrap();
        // Explicitly join fixture teardown even against the inert seam: the
        // negative control must not finish merely because the registry is inert.
        bounded(&mut control.dropped).await.unwrap();
        assert!(!scope.poll_panicked());
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn original_unscoped_constructors_remain_usable() {
    use datafusion_execution::memory_pool::MemoryConsumer;
    let pool = pool();
    let expected = batch();
    let input = Box::pin(RecordBatchStreamAdapter::new(
        schema(),
        futures::stream::iter(vec![Ok(expected.clone())]),
    ));
    let mut common = spawn_buffered(input, 2);
    assert_eq!(bounded(common.next()).await.unwrap().unwrap(), expected);
    assert!(bounded(common.next()).await.is_none());
    let input = futures::stream::iter(vec![Ok(expected.clone())]);
    let reservation = MemoryConsumer::new("original-memory-buffer").register(&pool);
    let mut memory = MemoryBufferedStream::new(input, 1024, reservation);
    assert_eq!(bounded(memory.next()).await.unwrap().unwrap(), expected);
    assert!(bounded(memory.next()).await.is_none());
    drop(memory);
    assert_eq!(pool.reserved(), 0);
}
