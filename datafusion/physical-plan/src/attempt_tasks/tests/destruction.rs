// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::{Arc, mpsc};
use std::task::{Context, Poll};
use std::thread;

use futures::FutureExt;

use super::super::{
    AttemptTasks, DescendantStopped, DrainError, TrackedCooperative, TrackedProducer,
};
use super::support::{
    Behavior, DropEvent, Gate, Guarded, ReleaseOnDrop, assert_pending, probe, receive,
};

#[derive(Clone, Copy, Debug)]
enum Finish {
    Ready,
    Error,
    Stop,
    Drop,
}

#[derive(Debug, PartialEq, Eq)]
enum Completion {
    Producer(Result<Result<(), &'static str>, DescendantStopped>),
    Cooperative(Result<(), &'static str>),
}

// Keep each tracked wrapper after Ready. A FutureExt::map adapter could destroy
// its completed future and accidentally supply the missing inner destruction.
enum Child<G> {
    Producer(Pin<Box<TrackedProducer<Guarded<G>>>>),
    Cooperative(Pin<Box<TrackedCooperative<Guarded<G>>>>),
}

impl<G: Unpin> Future for Child<G> {
    type Output = Completion;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            Self::Producer(child) => child.as_mut().poll(cx).map(Completion::Producer),
            Self::Cooperative(child) => {
                child.as_mut().poll(cx).map(Completion::Cooperative)
            }
        }
    }
}

fn track<G: Unpin>(
    scope: &Arc<AttemptTasks>,
    future: Guarded<G>,
    cooperative: bool,
) -> Pin<Box<Child<G>>> {
    Box::pin(if cooperative {
        Child::Cooperative(Box::pin(scope.track_cooperative(future).unwrap()))
    } else {
        Child::Producer(Box::pin(scope.track_producer(future).unwrap()))
    })
}

pub(super) fn ack_follows_inner_drop() {
    for cooperative in [false, true] {
        for finish in [Finish::Ready, Finish::Error, Finish::Drop, Finish::Stop] {
            if cooperative && matches!(finish, Finish::Stop) {
                continue; // Cooperative stop is independently tested with a release channel.
            }
            held_destruction(cooperative, finish);
        }
    }
}

fn held_destruction(cooperative: bool, finish: Finish) {
    let scope = AttemptTasks::new();
    let gate = Arc::new(Gate::default());
    let release = ReleaseOnDrop(Arc::clone(&gate));
    let (events, event_rx) = mpsc::channel();
    let (mut guard, dropped) = probe();
    guard.entered = Some(events.clone());
    guard.gate = Some(gate);
    let behavior = match finish {
        Finish::Ready => Behavior::Ready(Ok(())),
        Finish::Error => Behavior::Ready(Err("ordinary input error")),
        Finish::Stop | Finish::Drop => Behavior::Pending,
    };
    let future = Guarded::new(guard, behavior);
    let mut child = track(&scope, future, cooperative);
    if matches!(finish, Finish::Stop) {
        scope.stop();
    }
    scope.root_joined();
    let worker_gate = Arc::clone(&release.0);
    let worker = thread::spawn(move || {
        let result = if matches!(finish, Finish::Drop) {
            drop(child);
            (None, None)
        } else {
            let output = child
                .as_mut()
                .now_or_never()
                .expect("expected Ready or stopped");
            (Some(child), Some(output))
        };
        let _ = events.send(DropEvent::OperationReturned);
        // An inert implementation can return with its guard still captured.
        // Keep that result on this thread until the observing test releases it,
        // including on assertion failure, so JoinHandle Drop cannot block it.
        worker_gate.wait();
        result
    });
    // Inert Ready returns OperationReturned first, failing immediately rather
    // than relying on a timeout to detect retained inner-future ownership.
    assert_eq!(
        receive(&event_rx),
        DropEvent::Entered,
        "{cooperative:?}/{finish:?}"
    );
    let mut drain = Box::pin(scope.drain());
    assert_pending(drain.as_mut());
    assert!(
        dropped.try_recv().is_err(),
        "destructor remains inside its gate"
    );
    release.0.release();
    receive(&dropped);
    assert_eq!(receive(&event_rx), DropEvent::OperationReturned);
    let (retained_wrapper, output) = worker.join().unwrap();
    let expected = match finish {
        Finish::Drop => None,
        Finish::Stop => Some(Completion::Producer(Err(DescendantStopped))),
        Finish::Ready | Finish::Error => {
            let inner = if matches!(finish, Finish::Ready) {
                Ok(())
            } else {
                Err("ordinary input error")
            };
            Some(if cooperative {
                Completion::Cooperative(inner)
            } else {
                Completion::Producer(Ok(inner))
            })
        }
    };
    assert_eq!(output, expected);
    scope.stop();
    assert_eq!(drain.as_mut().now_or_never(), Some(Ok(())));
    // Completed wrappers are intentionally retained through the drain assertion.
    drop(retained_wrapper);
}

struct OnDrop(Option<Box<dyn FnOnce() + Send>>);

impl Drop for OnDrop {
    fn drop(&mut self) {
        if let Some(action) = self.0.take() {
            action();
        }
    }
}

pub(super) fn grandchild_registration_races_drain() {
    let scope = AttemptTasks::new();
    let register_gate = Arc::new(Gate::default());
    let register_release = ReleaseOnDrop(Arc::clone(&register_gate));
    let parent_gate = Arc::new(Gate::default());
    let parent_release = ReleaseOnDrop(Arc::clone(&parent_gate));
    let grandchild_gate = Arc::new(Gate::default());
    let grandchild_release = ReleaseOnDrop(Arc::clone(&grandchild_gate));
    let (parent_entered, parent_rx) = mpsc::channel();
    let (registered, registered_rx) = mpsc::channel::<Box<dyn Send>>();
    let (grandchild_events, grandchild_rx) = mpsc::channel();
    let (mut guard, dropped) = probe();
    guard.entered = Some(grandchild_events);
    guard.gate = Some(grandchild_gate);
    let child_scope = Arc::clone(&scope);
    let parent_guard = OnDrop(Some(Box::new(move || {
        let _ = parent_entered.send(());
        register_gate.wait();
        let grandchild = child_scope
            .track_cooperative(Guarded::new(guard, Behavior::Pending))
            .expect("counted parent's destructor may register during Draining");
        let _ = registered.send(Box::new(grandchild));
        parent_gate.wait();
    })));
    let parent = scope
        .track_producer(Guarded::new(parent_guard, Behavior::Pending))
        .unwrap();
    scope.stop();
    scope.root_joined();
    let parent_worker = thread::spawn(move || drop(parent));
    receive(&parent_rx);
    let mut drain = Box::pin(scope.drain());
    assert_pending(drain.as_mut()); // Parent is still counted before registration.
    register_release.0.release();
    let grandchild = receive(&registered_rx);
    assert_pending(drain.as_mut());
    let grandchild_worker = thread::spawn(move || drop(grandchild));
    assert_eq!(receive(&grandchild_rx), DropEvent::Entered);
    assert_pending(drain.as_mut()); // Both destructors are held.
    parent_release.0.release();
    parent_worker.join().unwrap();
    assert_pending(drain.as_mut()); // Only the grandchild remains.
    assert!(dropped.try_recv().is_err());
    grandchild_release.0.release();
    receive(&dropped);
    grandchild_worker.join().unwrap();
    assert_eq!(drain.as_mut().now_or_never(), Some(Ok(())));
}

pub(super) fn panic_and_owner_loss_do_not_certify_cleanup() {
    let root_owner = AttemptTasks::new();
    let observer = Arc::clone(&root_owner);
    let (guard, dropped) = probe();
    let unspawned = root_owner
        .track_producer(Guarded::new(guard, Behavior::Pending))
        .unwrap();
    // Losing a root owner cannot silently acknowledge the separate root join.
    drop(root_owner);
    drop(unspawned);
    receive(&dropped);
    observer.stop();
    let mut drain = Box::pin(observer.drain());
    assert_pending(drain.as_mut());
    // The other branches of the named acceptance scenario use actual unwinds.
    super::actual_poll_panic_is_sticky_and_attempt_local();
    destructor_failure_returns_drain_error();
}

struct PanicOnDrop(mpsc::Sender<()>);

impl Drop for PanicOnDrop {
    fn drop(&mut self) {
        let _ = self.0.send(());
        panic!("actual inner destructor failure");
    }
}

pub(super) fn destructor_failure_returns_drain_error() {
    for cooperative in [false, true] {
        for complete in [false, true] {
            let scope = AttemptTasks::new();
            let (entered, entered_rx) = mpsc::channel();
            let behavior = if complete {
                Behavior::Ready(Ok(()))
            } else {
                Behavior::Pending
            };
            let future = Guarded::new(PanicOnDrop(entered), behavior);
            let mut child = track(&scope, future, cooperative);
            // The API may contain or propagate destruction failure. Either way
            // it must retain incomplete teardown, never acknowledge success.
            let _caught = catch_unwind(AssertUnwindSafe(move || {
                if complete {
                    let _ = child.as_mut().now_or_never();
                }
                drop(child);
            }));
            receive(&entered_rx);
            scope.stop();
            scope.root_joined();
            assert_eq!(scope.drain().now_or_never(), Some(Err(DrainError)));
        }
    }
}
