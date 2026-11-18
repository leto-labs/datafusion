// Added by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0

//! Longbow family PX (prototype, not merged): an optional spawner for the
//! thread that polls an operator.
//!
//! [`SpawnedTask::spawn`](crate::SpawnedTask::spawn) and
//! [`JoinSet::spawn`](crate::JoinSet::spawn) run their future on the Tokio
//! runtime. An executor that polls operator streams on its own threads can
//! install a [`TaskSpawner`] for the duration of a poll. A task spawned in
//! that scope then runs on the executor. The Tokio task that the caller
//! holds only waits for its result, so join and abort keep their meaning:
//! aborting it closes the result channel, and the executor's future ends at
//! its next poll.
//!
//! No spawner is installed by default, and nothing changes then.

use std::cell::RefCell;
use std::future::{Future, pending, poll_fn};
use std::pin::{Pin, pin};
use std::sync::Arc;
use std::task::Poll;

/// A future that an executor runs to completion.
pub type BoxedTask = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

/// Runs tasks that operators spawn while this spawner is installed.
pub trait TaskSpawner: Send + Sync {
    /// Take ownership of `task` and poll it until it completes or the
    /// executor stops.
    fn spawn(&self, task: BoxedTask);
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<dyn TaskSpawner>>> = const { RefCell::new(None) };
}

/// Restores the previous spawner of the thread when dropped.
pub struct SpawnerGuard(Option<Arc<dyn TaskSpawner>>);

/// Install `spawner` on this thread until the guard is dropped.
pub fn enter(spawner: Arc<dyn TaskSpawner>) -> SpawnerGuard {
    SpawnerGuard(CURRENT.with(|current| current.replace(Some(spawner))))
}

impl Drop for SpawnerGuard {
    fn drop(&mut self) {
        let previous = self.0.take();
        // The thread-local may be gone during thread teardown.
        let _ = CURRENT.try_with(|current| *current.borrow_mut() = previous);
    }
}

/// Hand `task` to the installed spawner and return the future that waits for
/// its result, or return `task` when no spawner is installed.
pub(crate) fn redirect<T, R>(
    task: T,
) -> Result<impl Future<Output = R> + Send + 'static, T>
where
    T: Future<Output = R> + Send + 'static,
    R: Send + 'static,
{
    let Some(spawner) = CURRENT
        .try_with(|current| current.borrow().clone())
        .ok()
        .flatten()
    else {
        return Err(task);
    };
    let (mut sender, receiver) = tokio::sync::oneshot::channel();
    spawner.spawn(Box::pin(async move {
        let mut task = pin!(task);
        let result = poll_fn(|cx| {
            // The holder aborted or dropped the waiting task.
            if sender.poll_closed(cx).is_ready() {
                return Poll::Ready(None);
            }
            task.as_mut().poll(cx).map(Some)
        })
        .await;
        if let Some(result) = result {
            let _ = sender.send(result);
        }
    }));
    Ok(async move {
        match receiver.await {
            Ok(result) => result,
            // The executor dropped the task without a result. The holder
            // aborts this waiting task when it stops.
            Err(_) => pending().await,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{JoinSet, SpawnedTask};
    use std::sync::Mutex;
    use std::time::Duration;

    #[derive(Default)]
    struct Collect(Mutex<Vec<BoxedTask>>);

    impl TaskSpawner for Collect {
        fn spawn(&self, task: BoxedTask) {
            self.0.lock().unwrap().push(task);
        }
    }

    impl Collect {
        fn take(&self) -> Vec<BoxedTask> {
            std::mem::take(&mut *self.0.lock().unwrap())
        }
    }

    #[tokio::test]
    async fn an_installed_spawner_runs_the_task_and_join_returns_its_result() {
        let collect = Arc::new(Collect::default());
        let task = {
            let _guard = enter(Arc::clone(&collect) as Arc<dyn TaskSpawner>);
            SpawnedTask::spawn(async { 7 })
        };
        // The guard restored the thread: this task runs on Tokio.
        assert_eq!(SpawnedTask::spawn(async { 8 }).join().await.unwrap(), 8);
        let mut tasks = collect.take();
        assert_eq!(tasks.len(), 1);
        tasks.pop().unwrap().await;
        assert_eq!(task.join().await.unwrap(), 7);
    }

    #[tokio::test]
    async fn a_join_set_task_is_redirected_and_joined() {
        let collect = Arc::new(Collect::default());
        let mut set = JoinSet::new();
        {
            let _guard = enter(Arc::clone(&collect) as Arc<dyn TaskSpawner>);
            set.spawn(async { 3 });
        }
        for task in collect.take() {
            task.await;
        }
        assert_eq!(set.join_next().await.unwrap().unwrap(), 3);
    }

    #[tokio::test]
    async fn dropping_the_holder_ends_the_redirected_task() {
        let collect = Arc::new(Collect::default());
        let task = {
            let _guard = enter(Arc::clone(&collect) as Arc<dyn TaskSpawner>);
            SpawnedTask::spawn(pending::<()>())
        };
        drop(task);
        let redirected = collect.take().pop().unwrap();
        tokio::time::timeout(Duration::from_secs(5), redirected)
            .await
            .expect("the task ends after its holder aborted");
    }
}
