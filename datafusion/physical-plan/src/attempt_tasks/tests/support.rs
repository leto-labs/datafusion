// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::FutureExt;
use futures::channel::oneshot;

// A timeout only fails a missing handoff. It never establishes successful drain.
pub(super) fn receive<T>(receiver: &mpsc::Receiver<T>) -> T {
    receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("explicit test handoff did not arrive")
}

pub(super) fn assert_pending<F: Future>(future: Pin<&mut F>) {
    assert!(
        future.now_or_never().is_none(),
        "expected Pending at held boundary"
    );
}

#[derive(Default)]
pub(super) struct Gate {
    released: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    pub(super) fn wait(&self) {
        let mut released = self.released.lock().unwrap();
        while !*released {
            released = self.changed.wait(released).unwrap();
        }
    }

    pub(super) fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.changed.notify_all();
    }
}

// The observing test releases the destructor even when an assertion unwinds.
pub(super) struct ReleaseOnDrop(pub(super) Arc<Gate>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

pub(super) struct DropProbe {
    pub(super) entered: Option<mpsc::Sender<DropEvent>>,
    pub(super) gate: Option<Arc<Gate>>,
    pub(super) dropped: mpsc::Sender<()>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum DropEvent {
    Entered,
    OperationReturned,
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        if let Some(entered) = self.entered.take() {
            let _ = entered.send(DropEvent::Entered);
        }
        if let Some(gate) = &self.gate {
            gate.wait();
        }
        let _ = self.dropped.send(());
    }
}

pub(super) fn probe() -> (DropProbe, mpsc::Receiver<()>) {
    let (dropped, receiver) = mpsc::channel();
    (
        DropProbe {
            entered: None,
            gate: None,
            dropped,
        },
        receiver,
    )
}

pub(super) enum Behavior {
    Pending,
    Ready(Result<(), &'static str>),
    Wait(oneshot::Receiver<()>),
    Panic,
}

// The guard is a field, so Ready deliberately leaves it owned by the future.
// No resource-bearing value can escape through this future's inert output.
pub(super) struct Guarded<G> {
    _guard: G,
    behavior: Behavior,
    pub(super) polls: Arc<AtomicUsize>,
}

impl<G> Guarded<G> {
    pub(super) fn new(guard: G, behavior: Behavior) -> Self {
        Self {
            _guard: guard,
            behavior,
            polls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl<G: Unpin> Future for Guarded<G> {
    type Output = Result<(), &'static str>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        this.polls.fetch_add(1, Ordering::SeqCst);
        match &mut this.behavior {
            Behavior::Pending => Poll::Pending,
            Behavior::Ready(result) => Poll::Ready(*result),
            Behavior::Wait(receiver) => match Pin::new(receiver).poll(cx) {
                Poll::Ready(Ok(())) => Poll::Ready(Ok(())),
                Poll::Ready(Err(_)) => panic!("test release sender disappeared"),
                Poll::Pending => Poll::Pending,
            },
            Behavior::Panic => panic!("actual input poll panic"),
        }
    }
}
