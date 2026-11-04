// Modified by the Longbow authors; see third_party/README.md.

//! Attempt-owned descendant registration and joined destruction (ADR 0041).
//!
//! Tickets cover destruction of the captured future, not resources transferred
//! in its output. Root owners must call `root_joined` only after joined destruction.

use std::any::Any;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};

use datafusion_common::DataFusionError;
use datafusion_common_runtime::JoinSet;
use datafusion_execution::TaskContext;
use futures::future::{AbortHandle, Abortable, Pending, pending};

mod registry;

use registry::{Registry, Ticket};

/// One attempt's explicit descendant ownership boundary.
#[derive(Debug)]
pub struct AttemptTasks {
    registry: Registry,
}

impl AttemptTasks {
    /// Create an independent scope with an outstanding root obligation.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            registry: Registry::new(),
        })
    }

    /// Find an explicitly attached scope; unrelated DataFusion tasks have none.
    pub fn from_task(context: &TaskContext) -> Option<Arc<Self>> {
        context.session_config().get_extension::<Self>()
    }

    /// Register before spawn; stop may abort this producer's future.
    pub fn track_producer<F: Future>(
        self: &Arc<Self>,
        future: F,
    ) -> Result<TrackedProducer<F>, RegistrationClosed> {
        let (handle, registration) = AbortHandle::new_pair();
        let (ticket, stopped) = match self.registry.register(self, Some(handle.clone())) {
            Ok(registered) => registered,
            Err(closed) => {
                self.reject(future);
                return Err(closed);
            }
        };
        // This registration raced stop. It is counted, but cannot first-poll F.
        if stopped {
            handle.abort();
        }
        Ok(TrackedProducer {
            inner: Some(Box::pin(future)),
            ticket: Some(ticket),
            stopped: Abortable::new(pending(), registration),
        })
    }

    /// Register a writer that retains its existing cooperative halt semantics.
    pub fn track_cooperative<F: Future>(
        self: &Arc<Self>,
        future: F,
    ) -> Result<TrackedCooperative<F>, RegistrationClosed> {
        let (ticket, _) = match self.registry.register(self, None) {
            Ok(registered) => registered,
            Err(closed) => {
                self.reject(future);
                return Err(closed);
            }
        };
        Ok(TrackedCooperative {
            inner: Some(Box::pin(future)),
            ticket: Some(ticket),
        })
    }

    /// Stop producer work without aborting cooperative writer settlement.
    pub fn stop(&self) {
        self.registry.stop();
    }

    /// Acknowledge root destruction only after its owner has joined it.
    pub fn root_joined(&self) {
        self.registry.root_joined();
    }

    /// Await root and registered descendant destruction.
    /// Owners must request producer stop and cooperative halt separately.
    pub async fn drain(&self) -> Result<(), DrainError> {
        self.registry.drain_epoch().await.1
    }

    /// Capture this scope's registration epoch atomically with joined drain.
    /// Other cleanup owners may create late output collectors afterwards.
    pub async fn drain_epoch(&self) -> (u64, Result<(), DrainError>) {
        self.registry.drain_epoch().await
    }

    /// Revalidate a drain epoch from this same scope after other owners settle.
    /// None requires another drain; a late failure remains observable even when
    /// no new ticket was registered. Execution admission remains closed.
    pub fn drained_at(&self, epoch: u64) -> Option<Result<(), DrainError>> {
        self.registry.drained_at(epoch)
    }

    /// Remember an actual caught input-poll panic before forwarding it.
    pub fn record_poll_panic(&self) {
        self.registry.record_poll_panic();
    }

    /// Read the sticky attempt-local panic marker.
    pub fn poll_panicked(&self) -> bool {
        self.registry.poll_panicked()
    }

    /// Retain a discarded join set until all outputs have been destroyed.
    /// The caller first requests the appropriate cooperative halt or abort.
    /// Losing this cleanup future cannot acknowledge a successful drain.
    pub fn join_discarded<T: Send + 'static>(self: &Arc<Self>, tasks: JoinSet<T>) {
        if tasks.is_empty() {
            return;
        }
        let drain = DiscardedJoins {
            tasks: Some(tasks),
            scope: Arc::clone(self),
            joined: false,
        };
        let future = drain.finish();
        let (ticket, _) = match self.registry.register_cleanup(self) {
            Ok(registered) => registered,
            Err(_) => {
                self.reject(future);
                return;
            }
        };
        let tracked = TrackedCooperative {
            inner: Some(Box::pin(future)),
            ticket: Some(ticket),
        };
        // A closed blocking pool destroys a rejected closure under its lock.
        // Keep user state outside that closure until spawn has returned.
        let owned = Arc::new(Mutex::new(Some(tracked)));
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            // Retained-state Drop can call this from the control runtime's
            // blocking finalizer. Output destructors must not occupy its only
            // async worker and prevent the supervisor from forwarding stop.
            let joining = runtime.clone();
            let work = Arc::clone(&owned);
            drop(runtime.spawn_blocking(move || {
                let tracked = work.lock().unwrap_or_else(PoisonError::into_inner)
                    .take().expect("owned output collector");
                joining.block_on(tracked)
            }));
        }
        discard(Some(self), owned);
        // Without a live runtime, tracked Drop marks cleanup incomplete.
    }

    fn reject<F>(&self, future: F) {
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(future))) {
            self.registry.fail();
            resume_unwind(payload);
        }
    }
}

/// Destroy an output while its enclosing root or descendant still owns drain.
/// Successful destruction needs no ticket, allocation, lock or notification.
/// A caught destructor panic is sticky and must not replace a selected failure.
/// Without a scope, retain ordinary DataFusion destruction/unwind behavior.
pub fn discard<T>(scope: Option<&Arc<AttemptTasks>>, value: T) {
    let Some(scope) = scope else {
        drop(value);
        return;
    };
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(value))) {
        scope.registry.fail();
        // A secondary payload can itself have an arbitrary destructor.
        std::mem::forget(payload);
    }
}

/// A receiving container retains destruction ownership after producer tickets
/// finish. This wrapper adds no allocation or registration to queued items.
pub(crate) struct ScopedValue<T> {
    value: Option<T>,
    scope: Option<Arc<AttemptTasks>>,
}
impl<T> ScopedValue<T> {
    pub(crate) fn new(value: T, scope: Option<Arc<AttemptTasks>>) -> Self {
        Self {
            value: Some(value),
            scope,
        }
    }
}
impl<T> std::ops::Deref for ScopedValue<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().expect("owned receiving container")
    }
}
impl<T> std::ops::DerefMut for ScopedValue<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("owned receiving container")
    }
}
impl<T> Drop for ScopedValue<T> {
    fn drop(&mut self) {
        discard(self.scope.as_ref(), self.value.take());
    }
}

struct DiscardedJoins<T: 'static> {
    tasks: Option<JoinSet<T>>,
    scope: Arc<AttemptTasks>,
    joined: bool,
}
impl<T: 'static> DiscardedJoins<T> {
    async fn finish(mut self) {
        while let Some(joined) = self
            .tasks
            .as_mut()
            .expect("owned join set")
            .join_next()
            .await
        {
            discard(Some(&self.scope), joined);
        }
        self.joined = true;
    }
}
impl<T: 'static> Drop for DiscardedJoins<T> {
    fn drop(&mut self) {
        if !self.joined {
            self.scope.registry.fail();
        }
        discard(Some(&self.scope), self.tasks.take());
    }
}

/// A producer registered before its first poll.
#[derive(Debug)]
pub struct TrackedProducer<F> {
    inner: Option<Pin<Box<F>>>,
    ticket: Option<Ticket>,
    // Abortable owns only a pending notification future, never F. Aborted does
    // not destroy Abortable's task, so ownership of F remains explicit above.
    stopped: Abortable<Pending<()>>,
}

impl<F: Future> Future for TrackedProducer<F> {
    type Output = Result<F::Output, DescendantStopped>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        assert!(this.inner.is_some(), "producer polled after completion");
        // Register the abort waker before polling F; a stop racing a Pending
        // result either wakes this task or is observed by the second check.
        if Pin::new(&mut this.stopped).poll(cx).is_ready() {
            finish(&mut this.inner, &mut this.ticket);
            return Poll::Ready(Err(DescendantStopped));
        }
        match poll_inner(&mut this.inner, &mut this.ticket, cx) {
            Poll::Ready(output) => {
                finish(&mut this.inner, &mut this.ticket);
                Poll::Ready(Ok(output))
            }
            Poll::Pending if Pin::new(&mut this.stopped).poll(cx).is_ready() => {
                finish(&mut this.inner, &mut this.ticket);
                Poll::Ready(Err(DescendantStopped))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<F> Drop for TrackedProducer<F> {
    fn drop(&mut self) {
        drop_tracked(&mut self.inner, &mut self.ticket);
    }
}

/// A registered cooperative writer; the registry does not abort it.
#[derive(Debug)]
pub struct TrackedCooperative<F> {
    inner: Option<Pin<Box<F>>>,
    ticket: Option<Ticket>,
}

impl<F: Future> Future for TrackedCooperative<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match poll_inner(&mut this.inner, &mut this.ticket, cx) {
            Poll::Ready(output) => {
                finish(&mut this.inner, &mut this.ticket);
                Poll::Ready(output)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<F> Drop for TrackedCooperative<F> {
    fn drop(&mut self) {
        drop_tracked(&mut self.inner, &mut self.ticket);
    }
}

type PanicPayload = Box<dyn Any + Send + 'static>;

fn poll_inner<F: Future>(
    inner: &mut Option<Pin<Box<F>>>,
    ticket: &mut Option<Ticket>,
    cx: &mut Context<'_>,
) -> Poll<F::Output> {
    let future = inner.as_mut().expect("descendant polled after completion");
    match catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(cx))) {
        Ok(result) => result,
        Err(payload) => {
            if let Some(ticket) = ticket.as_ref() {
                ticket.scope.record_poll_panic();
            }
            if let Err(secondary) = destroy(inner, ticket, false) {
                // Preserve the original poll unwind. Even a panic payload may
                // have a panicking destructor; teardown is already marked failed.
                std::mem::forget(secondary);
            }
            resume_unwind(payload)
        }
    }
}

fn destroy<F>(
    inner: &mut Option<Pin<Box<F>>>,
    ticket: &mut Option<Ticket>,
    incomplete: bool,
) -> Result<(), PanicPayload> {
    let future = inner.take();
    let owned = future.is_some();
    let result = catch_unwind(AssertUnwindSafe(|| drop(future)));
    // Keep the ticket alive across arbitrary, reentrant and possibly held Drop.
    // A newly registered grandchild therefore cannot escape the barrier.
    if let Some(ticket) = ticket.take() {
        ticket.acknowledge(owned && result.is_ok() && !incomplete);
    }
    result
}

fn finish<F>(inner: &mut Option<Pin<Box<F>>>, ticket: &mut Option<Ticket>) {
    if let Err(payload) = destroy(inner, ticket, false) {
        resume_unwind(payload);
    }
}

fn drop_tracked<F>(inner: &mut Option<Pin<Box<F>>>, ticket: &mut Option<Ticket>) {
    let unwinding = std::thread::panicking();
    if let Err(payload) = destroy(inner, ticket, unwinding) {
        if unwinding {
            // Do not turn a caught destructor panic during unwind into abort.
            std::mem::forget(payload);
        } else {
            resume_unwind(payload);
        }
    }
}

macro_rules! scope_error {
    ($name:ident, $message:literal) => {
        #[doc = $message]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name;
        impl Display for $name {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                f.write_str($message)
            }
        }
        impl Error for $name {}
    };
}

scope_error!(
    RegistrationClosed,
    "attempt descendant registration is closed"
);
scope_error!(DescendantStopped, "attempt descendant producer was stopped");
scope_error!(DrainError, "attempt descendant teardown is incomplete");

/// Typed provenance for BufferExec's existing caught input-poll panic.
#[derive(Debug)]
pub struct CaughtPollPanic {
    pub(crate) original: DataFusionError,
}

impl Display for CaughtPollPanic {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.original.fmt(f)
    }
}
impl Error for CaughtPollPanic {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.original)
    }
}

#[cfg(test)]
mod tests;
