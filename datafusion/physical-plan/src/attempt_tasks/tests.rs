// Modified by the Longbow authors; see third_party/README.md.

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

//! Public-contract tests for ADR 0041. The parent mounts this module and owns
//! compilation plus behavioral RED/GREEN; no private registry state is inspected.

mod destruction;
mod support;
mod wake;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use datafusion_execution::{TaskContext, config::SessionConfig};
use futures::FutureExt;
use futures::channel::oneshot;

use super::{AttemptTasks, DescendantStopped};
use support::{Behavior, Guarded, assert_pending, probe, receive};

#[test]
fn registration_precedes_first_poll() {
    for cooperative in [false, true] {
        let scope = AttemptTasks::new();
        let (guard, dropped) = probe();
        let future = Guarded::new(guard, Behavior::Pending);
        let polls = Arc::clone(&future.polls);
        // Both branches erase only the wrapper type, keeping ownership intact.
        let child: Box<dyn Send> = if cooperative {
            Box::new(scope.track_cooperative(future).unwrap())
        } else {
            Box::new(scope.track_producer(future).unwrap())
        };
        scope.stop();
        scope.root_joined();
        let mut drain = Box::pin(scope.drain());
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert_pending(drain.as_mut());
        drop(child); // Equivalent to spawn refusing/dropping an unpolled future.
        receive(&dropped);
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert_eq!(drain.as_mut().now_or_never(), Some(Ok(())));
    }
}

#[test]
fn drain_waits_for_root_obligation() {
    let scope = AttemptTasks::new();
    scope.stop();
    let mut drain = Box::pin(scope.drain());
    assert_pending(drain.as_mut());
    scope.root_joined();
    assert_eq!(drain.as_mut().now_or_never(), Some(Ok(())));
}

#[test]
fn ready_fixture_retains_guard_until_future_drop() {
    let (guard, dropped) = probe();
    let mut future = Box::pin(Guarded::new(guard, Behavior::Ready(Ok(()))));
    assert_eq!(future.as_mut().now_or_never(), Some(Ok(())));
    assert!(dropped.try_recv().is_err());
    drop(future);
    assert_eq!(dropped.try_recv(), Ok(()));
}

#[test]
fn producer_stop_before_first_poll_destroys_captured_guard() {
    let scope = AttemptTasks::new();
    let (guard, dropped) = probe();
    let future = Guarded::new(guard, Behavior::Pending);
    let polls = Arc::clone(&future.polls);
    let mut child = Box::pin(scope.track_producer(future).unwrap());
    scope.stop();
    scope.root_joined();
    let mut drain = Box::pin(scope.drain());
    assert_pending(drain.as_mut());
    assert_eq!(child.as_mut().now_or_never(), Some(Err(DescendantStopped)));
    assert_eq!(
        dropped.try_recv(),
        Ok(()),
        "stop must destroy before returning"
    );
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    // Keep the completed wrapper alive: its Drop cannot supply missing cleanup.
    assert_eq!(drain.as_mut().now_or_never(), Some(Ok(())));
    drop(child);
}

#[test]
fn cooperative_writer_survives_stop_until_explicitly_released() {
    let scope = AttemptTasks::new();
    let (release, released) = oneshot::channel();
    let (guard, dropped) = probe();
    let mut child = Box::pin(
        scope
            .track_cooperative(Guarded::new(guard, Behavior::Wait(released)))
            .unwrap(),
    );
    assert_pending(child.as_mut());
    scope.stop();
    scope.stop(); // Idempotence must not turn cooperation into force cancellation.
    scope.root_joined();
    let mut drain = Box::pin(scope.drain());
    assert_pending(child.as_mut());
    assert!(dropped.try_recv().is_err());
    assert_pending(drain.as_mut());
    release.send(()).unwrap();
    assert_eq!(child.as_mut().now_or_never(), Some(Ok(())));
    assert_eq!(dropped.try_recv(), Ok(()));
    assert_eq!(drain.as_mut().now_or_never(), Some(Ok(())));
    drop(child);
}

#[test]
fn registration_while_draining_stops_new_producer_before_poll() {
    let scope = AttemptTasks::new();
    let keeper = scope
        .track_cooperative(futures::future::pending::<()>())
        .unwrap();
    scope.stop();
    scope.root_joined();
    let mut drain = Box::pin(scope.drain());
    assert_pending(drain.as_mut());
    let (guard, dropped) = probe();
    let future = Guarded::new(guard, Behavior::Ready(Ok(())));
    let polls = Arc::clone(&future.polls);
    let mut late = Box::pin(
        scope
            .track_producer(future)
            .expect("Draining admits counted work"),
    );
    assert_eq!(late.as_mut().now_or_never(), Some(Err(DescendantStopped)));
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(dropped.try_recv(), Ok(()));
    assert_pending(drain.as_mut());
    drop(keeper);
    assert_eq!(drain.as_mut().now_or_never(), Some(Ok(())));
}

#[test]
fn registration_while_draining_keeps_new_cooperative_writer_counted() {
    let scope = AttemptTasks::new();
    let keeper = scope
        .track_cooperative(futures::future::pending::<()>())
        .unwrap();
    scope.stop();
    scope.root_joined();
    let (release, released) = oneshot::channel();
    let (guard, dropped) = probe();
    let mut late = Box::pin(
        scope
            .track_cooperative(Guarded::new(guard, Behavior::Wait(released)))
            .unwrap(),
    );
    drop(keeper);
    let mut drain = Box::pin(scope.drain());
    assert_pending(late.as_mut());
    assert_pending(drain.as_mut());
    release.send(()).unwrap();
    assert_eq!(late.as_mut().now_or_never(), Some(Ok(())));
    assert_eq!(dropped.try_recv(), Ok(()));
    assert_eq!(drain.as_mut().now_or_never(), Some(Ok(())));
}

#[test]
fn drained_registration_refuses_and_drops_supplied_future() {
    let scope = AttemptTasks::new();
    scope.stop();
    scope.root_joined();
    assert_eq!(scope.drain().now_or_never(), Some(Ok(())));
    let (guard, dropped) = probe();
    let future = Guarded::new(guard, Behavior::Pending);
    let polls = Arc::clone(&future.polls);
    let refused = scope.track_producer(future);
    assert!(refused.is_err(), "Drained must refuse producers");
    assert_eq!(dropped.try_recv(), Ok(()));
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    let (guard, dropped) = probe();
    let future = Guarded::new(guard, Behavior::Pending);
    let polls = Arc::clone(&future.polls);
    let refused = scope.track_cooperative(future);
    assert!(refused.is_err(), "Drained must refuse cooperative writers");
    assert_eq!(dropped.try_recv(), Ok(()));
    assert_eq!(polls.load(Ordering::SeqCst), 0);
}

#[test]
fn contexts_do_not_share_registrations() {
    #[derive(Debug)]
    struct OtherExtension;
    let first = AttemptTasks::new();
    let retry = AttemptTasks::new();
    let other = Arc::new(OtherExtension);
    let config = SessionConfig::new()
        .with_extension(Arc::clone(&first))
        .with_extension(Arc::clone(&other));
    let original = TaskContext::default().with_session_config(config.clone());
    let derived = TaskContext::default().with_session_config(config);
    let retry_context = TaskContext::default()
        .with_session_config(SessionConfig::new().with_extension(Arc::clone(&retry)));
    assert!(AttemptTasks::from_task(&TaskContext::default()).is_none());
    let attached = AttemptTasks::from_task(&original).unwrap();
    assert!(Arc::ptr_eq(&attached, &first));
    assert!(Arc::ptr_eq(
        &AttemptTasks::from_task(&derived).unwrap(),
        &first
    ));
    assert!(Arc::ptr_eq(
        &AttemptTasks::from_task(&retry_context).unwrap(),
        &retry
    ));
    assert!(!Arc::ptr_eq(&first, &retry));
    assert!(Arc::ptr_eq(
        &derived
            .session_config()
            .get_extension::<OtherExtension>()
            .unwrap(),
        &other,
    ));
    let child = first
        .track_cooperative(futures::future::pending::<()>())
        .unwrap();
    first.stop();
    first.root_joined();
    retry.stop();
    retry.root_joined();
    let mut first_drain = Box::pin(first.drain());
    assert_pending(first_drain.as_mut());
    assert_eq!(retry.drain().now_or_never(), Some(Ok(())));
    drop(child);
    // Contexts and retrieved Arcs remain alive; they are not task obligations.
    assert_eq!(first_drain.as_mut().now_or_never(), Some(Ok(())));
    assert!(Arc::ptr_eq(
        &AttemptTasks::from_task(&derived).unwrap(),
        &first
    ));
    assert!(attached.track_producer(futures::future::ready(())).is_err());
}

#[test]
fn stopping_one_scope_leaves_other_scope_producer_running() {
    let stopped = AttemptTasks::new();
    let active = AttemptTasks::new();
    let (release, released) = oneshot::channel();
    let (guard, dropped) = probe();
    let mut child = Box::pin(
        active
            .track_producer(Guarded::new(guard, Behavior::Wait(released)))
            .unwrap(),
    );
    assert_pending(child.as_mut());
    stopped.stop();
    stopped.root_joined();
    assert_eq!(stopped.drain().now_or_never(), Some(Ok(())));
    assert_pending(child.as_mut());
    assert!(dropped.try_recv().is_err());
    release.send(()).unwrap();
    assert_eq!(child.as_mut().now_or_never(), Some(Ok(Ok(()))));
    assert_eq!(dropped.try_recv(), Ok(()));
    active.stop();
    active.root_joined();
    assert_eq!(active.drain().now_or_never(), Some(Ok(())));
}

#[test]
fn actual_poll_panic_is_sticky_and_attempt_local() {
    for cooperative in [false, true] {
        let scope = AttemptTasks::new();
        let unrelated = AttemptTasks::new();
        let (guard, dropped) = probe();
        let future = Guarded::new(guard, Behavior::Panic);
        let mut child: std::pin::Pin<Box<dyn Future<Output = ()>>> = if cooperative {
            Box::pin(scope.track_cooperative(future).unwrap().map(|_| ()))
        } else {
            Box::pin(scope.track_producer(future).unwrap().map(|_| ()))
        };
        assert!(!scope.poll_panicked());
        let caught = catch_unwind(AssertUnwindSafe(|| child.as_mut().now_or_never()));
        assert!(
            caught.is_err(),
            "the wrapper must preserve actual poll unwinding"
        );
        assert!(
            scope.poll_panicked(),
            "record before unwinding reaches the caller"
        );
        assert!(scope.poll_panicked(), "reading must not clear the marker");
        drop(child);
        receive(&dropped);
        scope.stop();
        scope.root_joined();
        // A poll panic invalidates execution, but joined destruction can succeed.
        assert_eq!(scope.drain().now_or_never(), Some(Ok(())));
        assert!(scope.poll_panicked());
        assert!(!unrelated.poll_panicked());
    }
}

#[test]
fn caught_poll_panic_recording_is_sticky_without_certifying_drain() {
    let scope = AttemptTasks::new();
    let caught = catch_unwind(|| Guarded::new((), Behavior::Panic).now_or_never());
    assert!(caught.is_err());
    scope.record_poll_panic();
    scope.record_poll_panic();
    assert!(scope.poll_panicked());
    scope.stop();
    let mut drain = Box::pin(scope.drain());
    assert_pending(drain.as_mut());
    scope.root_joined();
    assert_eq!(drain.as_mut().now_or_never(), Some(Ok(())));
    assert!(scope.poll_panicked());
}

#[test]
fn ordinary_error_and_stop_are_not_poll_panic_evidence() {
    let scope = AttemptTasks::new();
    let mut ordinary = Box::pin(
        scope
            .track_producer(futures::future::ready(Err::<(), _>(
                "actual input poll panic",
            )))
            .unwrap(),
    );
    assert_eq!(
        ordinary.as_mut().now_or_never(),
        Some(Ok(Err("actual input poll panic")))
    );
    assert!(!scope.poll_panicked());
    let stopped = scope
        .track_producer(futures::future::pending::<()>())
        .unwrap();
    scope.stop();
    drop(stopped);
    scope.root_joined();
    assert_eq!(scope.drain().now_or_never(), Some(Ok(())));
    assert!(!scope.poll_panicked());
}

#[test]
fn ack_follows_inner_drop() {
    destruction::ack_follows_inner_drop();
}

#[test]
fn grandchild_registration_races_drain() {
    destruction::grandchild_registration_races_drain();
}

#[test]
fn panic_and_owner_loss_do_not_certify_cleanup() {
    destruction::panic_and_owner_loss_do_not_certify_cleanup();
}

#[test]
fn destructor_failure_returns_drain_error() {
    destruction::destructor_failure_returns_drain_error();
}
