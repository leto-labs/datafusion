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

//! Short-lock accounting; no future, abort callback or waiter runs under it.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;

use futures::future::AbortHandle;
use parking_lot::Mutex;
use tokio::sync::Notify;

use super::{AttemptTasks, DrainError, RegistrationClosed};

#[derive(Debug, PartialEq, Eq)]
enum Phase {
    Running,
    Draining,
    // Execution admission is closed; retained outputs may still need cleanup.
    Closed,
}

#[derive(Debug)]
struct State {
    phase: Phase,
    root_joined: bool,
    // Includes the root until its explicitly joined destruction.
    outstanding: usize,
    next_id: u64,
    // BTreeMap frees removed entries rather than retaining a spawn high-water
    // allocation. Only producers carry abort handles; cooperative work never does.
    live: BTreeMap<u64, Option<AbortHandle>>,
    failed: bool,
    poll_panicked: bool,
}

impl State {
    fn settle(&mut self) {
        if self.root_joined && self.live.is_empty() {
            if self.outstanding != 0 {
                self.failed = true;
            }
            // Root join with no descendants also closes a never-stopped scope.
            if self.phase == Phase::Running {
                self.phase = Phase::Draining;
            }
            self.phase = Phase::Closed;
        }
    }
}

#[derive(Debug)]
pub(super) struct Registry {
    state: Mutex<State>,
    changed: Notify,
}

impl Registry {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(State {
                phase: Phase::Running,
                root_joined: false,
                outstanding: 1,
                next_id: 0,
                live: BTreeMap::new(),
                failed: false,
                poll_panicked: false,
            }),
            changed: Notify::new(),
        }
    }

    pub(super) fn register(
        &self,
        scope: &Arc<AttemptTasks>,
        producer: Option<AbortHandle>,
    ) -> Result<(Ticket, bool), RegistrationClosed> {
        self.register_inner(scope, producer, false)
    }

    /// Joining already-owned outputs must remain possible after execution
    /// admission closes or a previous cleanup failure becomes sticky.
    pub(super) fn register_cleanup(
        &self,
        scope: &Arc<AttemptTasks>,
    ) -> Result<(Ticket, bool), RegistrationClosed> {
        self.register_inner(scope, None, true)
    }

    fn register_inner(
        &self,
        scope: &Arc<AttemptTasks>,
        producer: Option<AbortHandle>,
        cleanup: bool,
    ) -> Result<(Ticket, bool), RegistrationClosed> {
        // Keep the owned handle outside the lock until the vacant entry exists.
        // On refusal it must not run its destructor under the registry lock.
        let mut producer = producer;
        let registered = {
            let mut state = self.state.lock();
            if !cleanup && (state.phase == Phase::Closed || state.failed) {
                None
            } else if let (Some(next_id), Some(outstanding)) = (
                state.next_id.checked_add(1),
                state.outstanding.checked_add(1),
            ) {
                let id = state.next_id;
                match state.live.entry(id) {
                    Entry::Vacant(entry) => {
                        entry.insert(producer.take());
                        state.next_id = next_id;
                        state.outstanding = outstanding;
                        Some((id, state.phase == Phase::Draining))
                    }
                    Entry::Occupied(_) => {
                        state.failed = true;
                        None
                    }
                }
            } else {
                state.failed = true;
                None
            }
        };
        match registered {
            Some((id, stopped)) => Ok((
                Ticket {
                    scope: Arc::clone(scope),
                    id: Some(id),
                },
                stopped,
            )),
            None => {
                self.changed.notify_waiters();
                Err(RegistrationClosed)
            }
        }
    }

    pub(super) fn stop(&self) {
        let handles = {
            let mut state = self.state.lock();
            if state.phase != Phase::Running {
                return;
            }
            state.phase = Phase::Draining;
            let handles = state.live.values().flatten().cloned().collect::<Vec<_>>();
            state.settle();
            handles
        };
        // Waking a task may invoke arbitrary code, including a panicking waker.
        // Still signal the other producers and make failure sticky before rethrow.
        let mut panic = None;
        for handle in handles {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(move || handle.abort())) {
                self.fail();
                if panic.is_none() {
                    panic = Some(payload);
                } else {
                    std::mem::forget(payload);
                }
            }
        }
        self.changed.notify_waiters();
        if let Some(payload) = panic {
            resume_unwind(payload);
        }
    }

    pub(super) fn root_joined(&self) {
        {
            let mut state = self.state.lock();
            if !state.root_joined {
                state.root_joined = true;
                match state.outstanding.checked_sub(1) {
                    Some(remaining) => state.outstanding = remaining,
                    None => state.failed = true,
                }
                state.settle();
            }
        }
        self.changed.notify_waiters();
    }

    fn acknowledge(&self, id: u64, destroyed: bool) {
        let removed = {
            let mut state = self.state.lock();
            let removed = state.live.remove(&id);
            if removed.is_none() || !destroyed {
                state.failed = true;
            }
            if removed.is_some() {
                match state.outstanding.checked_sub(1) {
                    Some(remaining) => state.outstanding = remaining,
                    None => state.failed = true,
                }
            }
            state.settle();
            removed
        };
        // An AbortHandle may own a waker; destroy it only after releasing state.
        drop(removed);
        self.changed.notify_waiters();
    }

    pub(super) async fn drain_epoch(&self) -> (u64, Result<(), DrainError>) {
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let state = self.state.lock();
                if state.phase == Phase::Closed && state.live.is_empty() {
                    let result = if state.failed {
                        Err(DrainError)
                    } else {
                        Ok(())
                    };
                    return (state.next_id, result);
                }
            }
            notified.await;
        }
    }

    pub(super) fn drained_at(&self, epoch: u64) -> Option<Result<(), DrainError>> {
        let state = self.state.lock();
        if state.phase != Phase::Closed || !state.live.is_empty() || state.next_id != epoch {
            return None;
        }
        Some(if state.failed { Err(DrainError) } else { Ok(()) })
    }

    pub(super) fn fail(&self) {
        self.state.lock().failed = true;
        self.changed.notify_waiters();
    }

    pub(super) fn record_poll_panic(&self) {
        self.state.lock().poll_panicked = true;
    }

    pub(super) fn poll_panicked(&self) -> bool {
        self.state.lock().poll_panicked
    }
}

#[derive(Debug)]
pub(super) struct Ticket {
    pub(super) scope: Arc<AttemptTasks>,
    id: Option<u64>,
}

impl Ticket {
    pub(super) fn acknowledge(mut self, destroyed: bool) {
        if let Some(id) = self.id.take() {
            self.scope.registry.acknowledge(id, destroyed);
        }
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            // Losing the ownership guard is not evidence of future destruction.
            self.scope.registry.acknowledge(id, false);
        }
    }
}
