// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Scoped buffering must register before first poll and retain input destruction.

mod panic_support;
mod panic_tests;
mod support;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use datafusion_common::DataFusionError;
use futures::StreamExt;

use crate::attempt_tasks::AttemptTasks;
use crate::common::spawn_buffered_with_scope;
use crate::stream::RecordBatchStreamAdapter;
use support::{
    DEADLOCK_BOUND, Gate, Helper, INPUT_BYTES, assert_pending, batch, bounded, input,
    pool, schema,
};

#[test]
fn registration_precedes_first_poll() {
    for helper in [Helper::Common, Helper::Memory] {
        // Hold the sole worker BEFORE creating the buffered producer. Entering
        // this multithread runtime on the test thread does not drive its tasks.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let gate = Arc::new(Gate::default());
        let _release = gate.release_on_drop();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let worker_gate = Arc::clone(&gate);
        let blocker = runtime.spawn(async move {
            entered_tx.send(()).unwrap();
            worker_gate.wait();
        });
        entered_rx.recv_timeout(DEADLOCK_BOUND).unwrap();

        let pool = pool();
        let scope = AttemptTasks::new();
        let (input, mut probe) = input(&pool, false, None);
        let consumer = {
            let _entered = runtime.enter();
            helper
                .start(input, &pool, Some(Arc::clone(&scope)))
                .unwrap()
        };
        scope.root_joined();
        let mut drain = Box::pin(scope.drain());
        assert_eq!(probe.polls.load(Ordering::SeqCst), 0);
        assert_pending(drain.as_mut());
        assert!(probe.dropped.try_recv().is_err());

        scope.stop();
        {
            // Receiver cleanup registers its output collector on this runtime.
            let _entered = runtime.enter();
            drop(consumer);
        }
        gate.release();
        runtime.block_on(async {
            bounded(blocker).await.unwrap();
            bounded(drain).await.unwrap();
            assert_eq!(probe.dropped.try_recv(), Ok(()));
            assert_eq!(probe.polls.load(Ordering::SeqCst), 0);
            assert_eq!(pool.reserved(), 0);
        });
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn held_input_drop_prevents_drain() {
    for helper in [Helper::Common, Helper::Memory] {
        let pool = pool();
        let scope = AttemptTasks::new();
        let gate = Arc::new(Gate::default());
        let _release = gate.release_on_drop();
        let (input, mut probe) = input(&pool, false, Some(Arc::clone(&gate)));
        let consumer = helper
            .start(input, &pool, Some(Arc::clone(&scope)))
            .unwrap();
        bounded(&mut probe.pending).await.unwrap();
        drop(consumer);
        scope.stop();
        scope.root_joined();
        bounded(&mut probe.drop_entered).await.unwrap();

        let mut drain = Box::pin(scope.drain());
        assert_pending(drain.as_mut());
        assert_eq!(pool.reserved(), INPUT_BYTES);
        assert!(probe.dropped.try_recv().is_err());
        gate.release();
        bounded(drain).await.unwrap();
        assert_eq!(probe.dropped.try_recv(), Ok(()));
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn producer_stop_releases_input_before_drain() {
    for helper in [Helper::Common, Helper::Memory] {
        let pool = pool();
        let scope = AttemptTasks::new();
        let (input, mut probe) = input(&pool, false, None);
        let consumer = helper
            .start(input, &pool, Some(Arc::clone(&scope)))
            .unwrap();
        bounded(&mut probe.pending).await.unwrap();
        scope.stop();
        scope.root_joined();
        bounded(scope.drain()).await.unwrap();
        // The consumer remains alive: stopping the registered producer alone
        // must destroy its pending input and its input-owned reservation.
        assert_eq!(probe.dropped.try_recv(), Ok(()));
        assert_eq!(pool.reserved(), 0);
        drop(consumer);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consumer_drop_releases_input_and_buffer_reservation_before_drain() {
    for helper in [Helper::Common, Helper::Memory] {
        let pool = pool();
        let scope = AttemptTasks::new();
        let gate = Arc::new(Gate::default());
        let _release = gate.release_on_drop();
        let (input, mut probe) = input(&pool, true, Some(Arc::clone(&gate)));
        let consumer = helper
            .start(input, &pool, Some(Arc::clone(&scope)))
            .unwrap();
        bounded(&mut probe.pending).await.unwrap();
        if matches!(helper, Helper::Memory) {
            assert!(
                pool.reserved() > INPUT_BYTES,
                "buffer never charged the queued row"
            );
        }
        drop(consumer);
        scope.root_joined();
        bounded(&mut probe.drop_entered).await.unwrap();
        let mut drain = Box::pin(scope.drain());
        assert_pending(drain.as_mut());
        gate.release();
        bounded(drain).await.unwrap();
        assert_eq!(probe.dropped.try_recv(), Ok(()));
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn current_thread_direct_path_has_no_descendant() {
    let pool = pool();
    let scope = AttemptTasks::new();
    let (input, mut probe) = input(&pool, false, None);
    let consumer = spawn_buffered_with_scope(input, 2, Some(Arc::clone(&scope))).unwrap();
    scope.stop();
    scope.root_joined();
    bounded(scope.drain()).await.unwrap();
    assert_eq!(probe.polls.load(Ordering::SeqCst), 0);
    assert_eq!(pool.reserved(), INPUT_BYTES);
    assert!(probe.dropped.try_recv().is_err());
    drop(consumer);
    assert_eq!(probe.dropped.try_recv(), Ok(()));
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unscoped_helpers_preserve_rows_and_errors() {
    for helper in [Helper::Common, Helper::Memory] {
        let pool = pool();
        let expected = batch();
        let input = Box::pin(RecordBatchStreamAdapter::new(
            schema(),
            futures::stream::iter(vec![
                Ok(expected.clone()),
                Err(DataFusionError::Internal("ordinary buffer error".into())),
            ]),
        ));
        let mut consumer = helper.start(input, &pool, None).unwrap();
        assert_eq!(bounded(consumer.next()).await.unwrap().unwrap(), expected);
        let error = bounded(consumer.next()).await.unwrap().unwrap_err();
        assert!(
            matches!(error, DataFusionError::Internal(ref message) if message == "ordinary buffer error")
        );
        assert!(bounded(consumer.next()).await.is_none());
        drop(consumer);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drained_scope_rejects_spawn_without_polling_input() {
    for helper in [Helper::Common, Helper::Memory] {
        let pool = pool();
        let scope = AttemptTasks::new();
        scope.stop();
        scope.root_joined();
        bounded(scope.drain()).await.unwrap();
        let (input, mut probe) = input(&pool, false, None);
        let result = helper.start(input, &pool, Some(scope));
        assert!(
            result.is_err(),
            "closed scope spawned an untracked producer"
        );
        assert_eq!(probe.polls.load(Ordering::SeqCst), 0);
        assert_eq!(probe.dropped.try_recv(), Ok(()));
        assert_eq!(pool.reserved(), 0);
    }
}
