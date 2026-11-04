//! A real drain waiter must be woken, not merely succeed when repolled manually.

use std::future::{Future, pending};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Context;

use futures::task::{ArcWake, waker};

use super::super::AttemptTasks;

#[derive(Default)]
struct WakeCount(AtomicUsize);

impl ArcWake for WakeCount {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn joined_root_and_last_descendant_wake_subscribed_drain() {
    for held_child in [false, true] {
        let scope = AttemptTasks::new();
        let child = held_child.then(|| scope.track_cooperative(pending::<()>()).unwrap());
        scope.stop();
        if held_child {
            scope.root_joined();
        }
        let counter = Arc::new(WakeCount::default());
        let wake = waker(Arc::clone(&counter));
        let mut context = Context::from_waker(&wake);
        let mut drain = Box::pin(scope.drain());
        assert!(drain.as_mut().poll(&mut context).is_pending());
        if held_child {
            drop(child);
        } else {
            scope.root_joined();
        }
        assert!(
            counter.0.load(Ordering::SeqCst) > 0,
            "drain waiter was not woken"
        );
        assert_eq!(
            drain.as_mut().poll(&mut context),
            std::task::Poll::Ready(Ok(()))
        );
    }
}
