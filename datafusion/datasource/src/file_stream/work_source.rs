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

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use crate::PartitionedFile;
use crate::file_scan_config::FileScanConfig;
use parking_lot::Mutex;

/// Source of work for `ScanState`. Ordered and independently executed streams
/// keep their work local; eligible siblings can share format-specific units.
#[derive(Debug)]
pub(super) enum WorkSource {
    Local(VecDeque<PartitionedFile>),
    Shared(SharedStream),
}

impl WorkSource {
    pub(super) fn poll_next_file(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<PartitionedFile>> {
        match self {
            Self::Local(files) => Poll::Ready(files.pop_front()),
            Self::Shared(shared) => shared.poll_next_file(cx),
        }
    }

    pub(super) fn skipped_on_limit(&self) -> usize {
        match self {
            Self::Local(files) => files.len(),
            Self::Shared(_) => 0,
        }
    }
}

/// Execution-local work sharing between sibling file streams.
///
/// Planned files remain in their byte-balanced partition. Each stream drains
/// its own files first, then takes units published by a format reader. Parquet
/// publishes disjoint runs of whole, unopened row groups after pruning.
///
/// An idle stream waits while any owner can still publish, including owners
/// that are already reading or have not opened their next planned file. This
/// permits a reader to offer its tail when siblings become idle later. When no
/// reader is active, an idle stream can take an unopened file so a consumer
/// that drains partitions sequentially does not wait for an unpolled owner.
#[derive(Debug, Clone)]
pub struct SharedWorkSource {
    inner: Arc<SharedWorkSourceInner>,
}

#[derive(Debug)]
struct SharedWorkSourceInner {
    queue: Mutex<SharedQueue>,
    publishing: AtomicBool,
    /// Unassigned waiters. Readers check this without taking the queue lock
    /// on the ordinary row-group path where no sibling needs work.
    idle: AtomicUsize,
}

#[derive(Debug, Default)]
struct SharedQueue {
    owned: Vec<VecDeque<PartitionedFile>>,
    /// Until publishing is enabled, preserve the format's global file order.
    /// `Some(empty)` remains distinct from owner-first publishing mode.
    global: Option<VecDeque<PartitionedFile>>,
    /// Format-specific published units, never the original planned files.
    files: VecDeque<PartitionedFile>,
    active: usize,
    waiters: HashMap<usize, Waker>,
}

impl SharedQueue {
    fn has_owners(&self) -> bool {
        self.active > 0
            || self.global.as_ref().map_or_else(
                || self.owned.iter().any(|files| !files.is_empty()),
                |files| !files.is_empty(),
            )
    }

    fn wakers(&self) -> Vec<Waker> {
        self.waiters.values().cloned().collect()
    }

    fn take_waiters(&mut self) -> Vec<Waker> {
        self.waiters.drain().map(|(_, waker)| waker).collect()
    }
}

fn wake_all(wakers: Vec<Waker>) {
    wakers.into_iter().for_each(Waker::wake);
}

impl SharedWorkSource {
    fn from_groups(owned: Vec<VecDeque<PartitionedFile>>) -> Self {
        Self {
            inner: Arc::new(SharedWorkSourceInner {
                queue: Mutex::new(SharedQueue {
                    owned,
                    ..Default::default()
                }),
                publishing: AtomicBool::new(false),
                idle: AtomicUsize::new(0),
            }),
        }
    }

    #[cfg(test)]
    fn new(files: impl IntoIterator<Item = PartitionedFile>, streams: usize) -> Self {
        let mut groups = vec![VecDeque::new(); streams];
        for (index, file) in files.into_iter().enumerate() {
            groups[index % streams].push_back(file);
        }
        Self::from_groups(groups)
    }

    /// Preserve planned ownership for publishers, and the format's global
    /// statistics order for scans that keep ordinary file-sized work units.
    pub(crate) fn from_config(config: &FileScanConfig) -> Self {
        let source = Self::from_groups(
            config
                .file_groups
                .iter()
                .map(|group| group.iter().cloned().collect())
                .collect(),
        );
        source.inner.queue.lock().global = Some(
            config
                .file_source
                .reorder_files(
                    config
                        .file_groups
                        .iter()
                        .flat_map(|group| group.iter().cloned())
                        .collect(),
                )
                .into(),
        );
        source
    }

    pub(super) fn stream(&self, partition: usize) -> SharedStream {
        SharedStream {
            source: self.clone(),
            id: partition,
            held: false,
        }
    }

    /// Enable publishing before the first stream takes a file.
    pub fn enable_publishing(&self) {
        let global = {
            let mut queue = self.inner.queue.lock();
            self.inner.publishing.store(true, Ordering::Release);
            queue.global.take()
        };
        // File extensions can own resources with arbitrary destructors.
        drop(global);
    }

    /// Number of useful parts for `units` remaining independent work units.
    /// This is queried again at reader boundaries, not just at file open.
    pub fn split_width(&self, units: usize) -> usize {
        units
            .min(self.inner.idle.load(Ordering::Relaxed).saturating_add(1))
            .max(1)
    }

    /// Publish disjoint work removed from the donor's unread plan. Wake outside
    /// the lock; retain waiter registrations until they actually claim work.
    pub fn publish(&self, files: impl IntoIterator<Item = PartitionedFile>) {
        let wakers = {
            let mut queue = self.inner.queue.lock();
            queue.files.extend(files);
            self.update_idle(&queue);
            queue.wakers()
        };
        wake_all(wakers);
    }

    fn update_idle(&self, queue: &SharedQueue) {
        self.inner.idle.store(
            queue.waiters.len().saturating_sub(queue.files.len()),
            Ordering::Relaxed,
        );
    }
}

/// One output partition's handle. Its file stays active until the reader and
/// all planners finish, or the stream is dropped on error, limit or cancellation.
#[derive(Debug)]
pub(super) struct SharedStream {
    source: SharedWorkSource,
    id: usize,
    held: bool,
}

impl SharedStream {
    fn poll_next_file(&mut self, cx: &mut Context<'_>) -> Poll<Option<PartitionedFile>> {
        let mut queue = self.source.inner.queue.lock();
        if std::mem::take(&mut self.held) {
            queue.active -= 1;
        }
        let file = if let Some(global) = &mut queue.global {
            global.pop_front().or_else(|| queue.files.pop_front())
        } else {
            queue.owned[self.id]
                .pop_front()
                .or_else(|| queue.files.pop_front())
                .or_else(|| {
                    // Consumers may drain partitions sequentially, or never
                    // create another owner's stream. With no active reader,
                    // there is nobody to publish: take an unopened file to
                    // make progress instead of waiting for an unpolled owner.
                    (queue.active == 0)
                        .then(|| queue.owned.iter_mut().find_map(VecDeque::pop_front))
                        .flatten()
                })
        };
        let result = if let Some(file) = file {
            queue.active += 1;
            self.held = true;
            queue.waiters.remove(&self.id);
            Poll::Ready(Some(file))
        } else if self.source.inner.publishing.load(Ordering::Acquire)
            && queue.has_owners()
        {
            queue.waiters.insert(self.id, cx.waker().clone());
            Poll::Pending
        } else {
            queue.waiters.remove(&self.id);
            Poll::Ready(None)
        };
        let wakers = if queue.active == 0 {
            queue.take_waiters()
        } else {
            vec![]
        };
        self.source.update_idle(&queue);
        drop(queue);
        wake_all(wakers);
        result
    }
}

impl Drop for SharedStream {
    fn drop(&mut self) {
        let (wakers, abandoned) = {
            let mut queue = self.source.inner.queue.lock();
            queue.waiters.remove(&self.id);
            let abandoned = std::mem::take(&mut queue.owned[self.id]);
            if self.held {
                queue.active -= 1;
            }
            let wakers = if queue.active == 0 {
                queue.take_waiters()
            } else {
                vec![]
            };
            self.source.update_idle(&queue);
            (wakers, abandoned)
        };
        // File extensions can own other resources. Release them outside the
        // queue lock, just as we invoke wakers outside it.
        drop(abandoned);
        wake_all(wakers);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::task::Wake;

    #[derive(Default)]
    struct CountingWaker(AtomicUsize);
    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn waker() -> (Arc<CountingWaker>, Waker) {
        let counter = Arc::new(CountingWaker::default());
        (Arc::clone(&counter), Waker::from(Arc::clone(&counter)))
    }
    fn file(name: &str) -> PartitionedFile {
        PartitionedFile::new(name, 10)
    }
    fn path(poll: Poll<Option<PartitionedFile>>) -> Poll<Option<String>> {
        poll.map(|file| file.map(|file| file.object_meta.location.to_string()))
    }

    #[test]
    fn stream_drains_its_planned_files_before_shared_units() {
        use crate::TableSchema;
        use crate::file_groups::FileGroup;
        use crate::file_scan_config::FileScanConfigBuilder;
        use crate::test_util::MockSource;
        use arrow::datatypes::Schema;
        use datafusion_execution::object_store::ObjectStoreUrl;
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(MockSource::new(TableSchema::from(
                Arc::new(Schema::empty()),
            ))),
        )
        .with_file_groups(vec![
            FileGroup::new(vec![file("a"), file("c")]),
            FileGroup::new(vec![file("b")]),
        ])
        .build();
        let source = SharedWorkSource::from_config(&config);
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut b = source.stream(1);
        let mut a = source.stream(0);
        source.enable_publishing();
        source.publish([file("stolen-row-groups")]);
        assert_eq!(
            path(b.poll_next_file(&mut cx)),
            Poll::Ready(Some("b".into()))
        );
        assert_eq!(
            path(a.poll_next_file(&mut cx)),
            Poll::Ready(Some("a".into()))
        );
        assert_eq!(
            path(a.poll_next_file(&mut cx)),
            Poll::Ready(Some("c".into()))
        );
        assert_eq!(
            path(b.poll_next_file(&mut cx)),
            Poll::Ready(Some("stolen-row-groups".into()))
        );
    }

    #[test]
    fn non_publishing_stream_uses_global_file_order() {
        use crate::TableSchema;
        use crate::file_groups::FileGroup;
        use crate::file_scan_config::FileScanConfigBuilder;
        use crate::test_util::MockSource;
        use arrow::datatypes::Schema;
        use datafusion_execution::object_store::ObjectStoreUrl;
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(MockSource::new(TableSchema::from(
                Arc::new(Schema::empty()),
            ))),
        )
        .with_file_groups(vec![
            FileGroup::new(vec![file("a"), file("c")]),
            FileGroup::new(vec![file("b")]),
        ])
        .build();
        let source = SharedWorkSource::from_config(&config);
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut b = source.stream(1);
        // Claim from the single global queue, even when partition 1 polls first.
        for expected in ["a", "c", "b"] {
            assert_eq!(
                path(b.poll_next_file(&mut cx)),
                Poll::Ready(Some(expected.into()))
            );
        }
        assert_eq!(path(b.poll_next_file(&mut cx)), Poll::Ready(None));
    }

    #[test]
    fn dropping_last_active_owner_wakes_waiter_to_take_unopened_file() {
        let source = SharedWorkSource::new([file("a"), file("b")], 3);
        source.enable_publishing();
        let mut a = source.stream(0);
        let mut idle = source.stream(2);
        let (wakes, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(a.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        assert_eq!(path(idle.poll_next_file(&mut cx)), Poll::Pending);
        drop(a);
        assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
        assert_eq!(
            path(idle.poll_next_file(&mut cx)),
            Poll::Ready(Some("b".into()))
        );
        assert_eq!(path(idle.poll_next_file(&mut cx)), Poll::Ready(None));
    }

    #[test]
    fn idle_stream_waits_for_row_groups_after_a_sibling_started_reading() {
        let source = SharedWorkSource::new([file("a")], 2);
        source.enable_publishing();
        let (_, waker) = waker();
        let (wakes, waiting) = self::waker();
        let mut cx = Context::from_waker(&waker);
        let mut wait_cx = Context::from_waker(&waiting);
        let mut a = source.stream(0);
        let mut b = source.stream(1);
        assert!(matches!(a.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        assert_eq!(path(b.poll_next_file(&mut wait_cx)), Poll::Pending);
        assert_eq!(source.split_width(12), 2);
        source.publish([file("late-row-groups")]);
        assert_eq!(source.split_width(12), 1);
        assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
        assert_eq!(
            path(b.poll_next_file(&mut wait_cx)),
            Poll::Ready(Some("late-row-groups".into()))
        );
        assert_eq!(path(a.poll_next_file(&mut cx)), Poll::Pending);
        assert_eq!(path(b.poll_next_file(&mut wait_cx)), Poll::Ready(None));
        assert_eq!(path(a.poll_next_file(&mut cx)), Poll::Ready(None));
    }

    #[test]
    fn late_opened_owned_files_can_publish_after_another_owner_finishes() {
        let source = SharedWorkSource::new([file("a"), file("b"), file("c")], 2);
        source.enable_publishing();
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut a = source.stream(0);
        let mut b = source.stream(1);
        assert!(matches!(a.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        assert!(matches!(b.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        assert_eq!(source.split_width(12), 1);
        assert_eq!(path(b.poll_next_file(&mut cx)), Poll::Pending);
        assert_eq!(
            path(a.poll_next_file(&mut cx)),
            Poll::Ready(Some("c".into()))
        );
        assert_eq!(source.split_width(12), 2);
        source.publish([file("c-tail")]);
        assert_eq!(
            path(b.poll_next_file(&mut cx)),
            Poll::Ready(Some("c-tail".into()))
        );
    }

    #[test]
    fn sequential_stream_takes_unopened_file_when_no_reader_is_active() {
        let source = SharedWorkSource::new([file("a"), file("b")], 2);
        source.enable_publishing();
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut a = source.stream(0);
        assert_eq!(
            path(a.poll_next_file(&mut cx)),
            Poll::Ready(Some("a".into()))
        );
        assert_eq!(
            path(a.poll_next_file(&mut cx)),
            Poll::Ready(Some("b".into()))
        );
        assert_eq!(path(a.poll_next_file(&mut cx)), Poll::Ready(None));
        let mut b = source.stream(1);
        assert_eq!(path(b.poll_next_file(&mut cx)), Poll::Ready(None));
    }

    #[test]
    fn idle_stream_can_claim_an_unopened_owner_before_it_is_polled() {
        let source = SharedWorkSource::new([file("a")], 2);
        source.enable_publishing();
        let a = source.stream(0);
        let mut b = source.stream(1);
        let (wakes, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        assert_eq!(
            path(b.poll_next_file(&mut cx)),
            Poll::Ready(Some("a".into()))
        );
        drop(a);
        assert_eq!(wakes.0.load(Ordering::SeqCst), 0);
        assert_eq!(path(b.poll_next_file(&mut cx)), Poll::Ready(None));
    }

    #[test]
    fn dropping_a_reading_or_planning_stream_releases_waiters() {
        let source = SharedWorkSource::new([file("a")], 2);
        source.enable_publishing();
        let mut a = source.stream(0);
        let mut b = source.stream(1);
        let (wakes, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(a.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        assert_eq!(path(b.poll_next_file(&mut cx)), Poll::Pending);
        drop(a);
        assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
        assert_eq!(path(b.poll_next_file(&mut cx)), Poll::Ready(None));
    }

    #[test]
    fn without_publishing_an_empty_owner_finishes_at_once() {
        let source = SharedWorkSource::new([file("a")], 2);
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut a = source.stream(0);
        let mut b = source.stream(1);
        assert!(matches!(a.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        assert_eq!(path(b.poll_next_file(&mut cx)), Poll::Ready(None));
    }

    #[test]
    fn concurrent_stealers_claim_each_published_unit_once() {
        let source = SharedWorkSource::new([], 8);
        source.publish((0..200).map(|i| file(&format!("rg{i}"))));
        let mut paths = std::thread::scope(|scope| {
            (0..8)
                .map(|partition| {
                    let source = source.clone();
                    scope.spawn(move || {
                        let (_, waker) = waker();
                        let mut cx = Context::from_waker(&waker);
                        let mut stream = source.stream(partition);
                        let mut paths = Vec::new();
                        while let Poll::Ready(Some(file)) =
                            path(stream.poll_next_file(&mut cx))
                        {
                            paths.push(file);
                            std::thread::yield_now();
                        }
                        paths
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        paths.sort();
        assert_eq!(paths.len(), 200);
        paths.dedup();
        assert_eq!(paths.len(), 200);
    }
}
