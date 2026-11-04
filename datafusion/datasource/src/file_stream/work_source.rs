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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::task::{Context, Poll, Waker};

use crate::PartitionedFile;
use crate::file_groups::FileGroup;
use crate::file_scan_config::FileScanConfig;
use parking_lot::Mutex;

/// Source of work for `ScanState`.
///
/// Streams that may share work across siblings use [`WorkSource::Shared`],
/// while streams that can not share work (e.g. because they must preserve file
/// order) use  [`WorkSource::Local`].
#[derive(Debug)]
pub(super) enum WorkSource {
    /// Files this stream will plan locally without sharing them.
    Local(VecDeque<PartitionedFile>),
    /// Files shared with sibling streams.
    Shared(SharedStream),
}

impl WorkSource {
    /// Take the next file to plan from this work source.
    ///
    /// The stream must have finished its previous file. A shared stream
    /// returns `Poll::Pending`, with `cx` registered, while the queue is empty
    /// and a sibling may still publish work (see
    /// [`SharedWorkSource::publish`]).
    pub(super) fn poll_next_file(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<PartitionedFile>> {
        match self {
            Self::Local(files) => Poll::Ready(files.pop_front()),
            Self::Shared(shared) => shared.poll_next_file(cx),
        }
    }

    /// Record that the current file has finished planning: any work it
    /// publishes is already in the queue.
    pub(super) fn planning_finished(&mut self) {
        if let Self::Shared(shared) = self {
            shared.planning_finished();
        }
    }

    /// Return how many queued files should be counted as already processed
    /// when this stream stops early after hitting a global limit.
    pub(super) fn skipped_on_limit(&self) -> usize {
        match self {
            Self::Local(files) => files.len(),
            Self::Shared(_) => 0,
        }
    }
}

/// Shared source of work for sibling `FileStream`s
///
/// The queue is created once per execution and shared by all reorderable
/// sibling streams for that execution. Whichever stream becomes idle first may
/// take the next unopened file from the front of the queue.
///
/// A file format may also [`publish`](Self::publish) finer work units that it
/// finds while it plans a file (Longbow: the surviving row groups of a pruned
/// Parquet scan). When publishing is enabled, an idle stream does not finish
/// while a sibling still plans a file, because that sibling can still publish
/// work for it.
///
/// It uses a [`Mutex`] internally to provide thread-safe access
/// to the shared file queue.
#[derive(Debug, Clone)]
pub struct SharedWorkSource {
    inner: Arc<SharedWorkSourceInner>,
}

#[derive(Debug)]
pub(super) struct SharedWorkSourceInner {
    queue: Mutex<SharedQueue>,
    /// Number of sibling streams (the scan's output partitions).
    streams: usize,
    /// Set when a morselizer can publish work while it plans a file.
    publishing: AtomicBool,
    /// Set when a file or row group of this scan was pruned.
    pruned: AtomicBool,
    next_stream_id: AtomicU64,
}

#[derive(Debug, Default)]
struct SharedQueue {
    files: VecDeque<PartitionedFile>,
    /// Streams that hold a file, from when they take it until they take the
    /// next one or finish. Counted only when publishing is enabled.
    active: usize,
    /// Held files that are still being planned and so can still publish work.
    planning: usize,
    /// Wakers of streams that wait for published work, by stream id.
    waiters: HashMap<u64, Waker>,
}

impl SharedQueue {
    /// Remove all registered wakers so the caller can wake them after it
    /// releases the lock.
    fn take_waiters(&mut self) -> Vec<Waker> {
        self.waiters.drain().map(|(_, waker)| waker).collect()
    }
}

fn wake_all(wakers: Vec<Waker>) {
    wakers.into_iter().for_each(Waker::wake);
}

impl SharedWorkSource {
    /// Create a shared work source for `streams` sibling streams that
    /// contains the provided unopened files.
    pub(crate) fn new(
        files: impl IntoIterator<Item = PartitionedFile>,
        streams: usize,
    ) -> Self {
        Self {
            inner: Arc::new(SharedWorkSourceInner {
                queue: Mutex::new(SharedQueue {
                    files: files.into_iter().collect(),
                    ..Default::default()
                }),
                streams,
                publishing: AtomicBool::new(false),
                pruned: AtomicBool::new(false),
                next_stream_id: AtomicU64::new(0),
            }),
        }
    }

    /// Create a shared work source for the unopened files in `config`.
    ///
    /// Files are reordered by the file source (e.g. by statistics for TopK)
    /// before being placed in the shared queue, so the most promising files
    /// are processed first across all partitions.
    pub(crate) fn from_config(config: &FileScanConfig) -> Self {
        let files: Vec<_> = config
            .file_groups
            .iter()
            .flat_map(FileGroup::iter)
            .cloned()
            .collect();
        let files = config.file_source.reorder_files(files);
        Self::new(files, config.file_groups.len())
    }

    /// Create the handle of one sibling stream.
    pub(super) fn stream(&self) -> SharedStream {
        SharedStream {
            source: self.clone(),
            id: self.inner.next_stream_id.fetch_add(1, Ordering::Relaxed),
            held: Held::Nothing,
        }
    }

    /// Enable publishing. A morselizer calls this before its stream takes the
    /// first file. All morselizers of one scan have the same configuration,
    /// so every stream sees the flag before its first `poll_next_file`.
    pub fn enable_publishing(&self) {
        self.inner.publishing.store(true, Ordering::Release);
    }

    /// Record that this scan pruned a file range or a row group.
    pub fn mark_pruned(&self) {
        self.inner.pruned.store(true, Ordering::Relaxed);
    }

    /// Return into how many units a file with `units` independent work units
    /// (e.g. surviving row groups) should be split: 1 means do not split.
    ///
    /// The cost gate. A split costs one more file open per published unit, so
    /// split only when all of these are true:
    ///
    /// * the scan pruned work, so the static assignment of byte ranges to
    ///   streams no longer reflects the work that remains;
    /// * more sibling streams are idle than there are queued files, so a
    ///   published unit goes to a stream that would otherwise have no work.
    ///
    /// A full scan never splits, and a pruned scan splits only while streams
    /// would otherwise be idle.
    pub fn split_width(&self, units: usize) -> usize {
        if units < 2 || !self.inner.pruned.load(Ordering::Relaxed) {
            return 1;
        }
        let queue = self.inner.queue.lock();
        let idle = self
            .inner
            .streams
            .saturating_sub(queue.active)
            .saturating_sub(queue.files.len());
        units.min(idle + 1)
    }

    /// Publish work units that a stream found while it planned a file, and
    /// wake the streams that wait for work.
    pub fn publish(&self, files: impl IntoIterator<Item = PartitionedFile>) {
        let wakers = {
            let mut queue = self.inner.queue.lock();
            queue.files.extend(files);
            queue.take_waiters()
        };
        wake_all(wakers);
    }

    fn publishing(&self) -> bool {
        self.inner.publishing.load(Ordering::Acquire)
    }
}

/// What a [`SharedStream`] holds from the shared queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Held {
    Nothing,
    /// A file that is still being planned.
    Planning,
    /// A file whose planning finished.
    Reading,
}

/// One sibling stream's handle to a [`SharedWorkSource`].
///
/// It accounts for the file that the stream holds, and releases it when the
/// stream takes the next file or is dropped.
#[derive(Debug)]
pub(super) struct SharedStream {
    source: SharedWorkSource,
    id: u64,
    held: Held,
}

impl SharedStream {
    fn poll_next_file(&mut self, cx: &mut Context<'_>) -> Poll<Option<PartitionedFile>> {
        let publishing = self.source.publishing();
        let mut wakers = Vec::new();
        let result = {
            let mut queue = self.source.inner.queue.lock();
            if let Some(waiters) = release(&mut self.held, &mut queue) {
                wakers = waiters;
            }
            if let Some(file) = queue.files.pop_front() {
                if publishing {
                    queue.active += 1;
                    queue.planning += 1;
                    self.held = Held::Planning;
                }
                queue.waiters.remove(&self.id);
                Poll::Ready(Some(file))
            } else if publishing && queue.planning > 0 {
                queue.waiters.insert(self.id, cx.waker().clone());
                Poll::Pending
            } else {
                queue.waiters.remove(&self.id);
                Poll::Ready(None)
            }
        };
        wake_all(wakers);
        result
    }

    fn planning_finished(&mut self) {
        if self.held != Held::Planning {
            return;
        }
        let wakers = {
            let mut queue = self.source.inner.queue.lock();
            queue.planning -= 1;
            self.held = Held::Reading;
            if queue.planning == 0 {
                queue.take_waiters()
            } else {
                Vec::new()
            }
        };
        wake_all(wakers);
    }
}

/// Release the file a stream holds. Return the wakers to wake when no file is
/// planned any more: the waiting streams can then finish.
fn release(held: &mut Held, queue: &mut SharedQueue) -> Option<Vec<Waker>> {
    match std::mem::replace(held, Held::Nothing) {
        Held::Nothing => None,
        Held::Reading => {
            queue.active -= 1;
            None
        }
        Held::Planning => {
            queue.active -= 1;
            queue.planning -= 1;
            (queue.planning == 0).then(|| queue.take_waiters())
        }
    }
}

impl Drop for SharedStream {
    fn drop(&mut self) {
        let wakers = {
            let mut queue = self.source.inner.queue.lock();
            queue.waiters.remove(&self.id);
            release(&mut self.held, &mut queue)
        };
        if let Some(wakers) = wakers {
            wake_all(wakers);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::task::Wake;

    use super::*;

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
    fn full_scan_does_not_split() {
        // Four streams take one file each. Nothing is pruned.
        let source = SharedWorkSource::new((0..4).map(|i| file(&format!("f{i}"))), 4);
        source.enable_publishing();
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut streams: Vec<_> = (0..4).map(|_| source.stream()).collect();
        for stream in &mut streams {
            assert!(matches!(stream.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        }
        assert_eq!(source.split_width(3), 1);
    }

    #[test]
    fn pruned_scan_splits_only_for_idle_streams() {
        let source = SharedWorkSource::new([file("a"), file("b")], 8);
        source.enable_publishing();
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut a = source.stream();
        assert!(matches!(a.poll_next_file(&mut cx), Poll::Ready(Some(_))));

        // Not pruned: no split, however many streams are idle.
        assert_eq!(source.split_width(2), 1);
        source.mark_pruned();
        // One file is queued and one stream is active, so 6 streams have no
        // work: two row groups split into two units.
        assert_eq!(source.split_width(2), 2);
        assert_eq!(source.split_width(24), 7);
        // A single unit never splits.
        assert_eq!(source.split_width(1), 1);
    }

    #[test]
    fn queued_work_counts_against_idle_streams() {
        let source = SharedWorkSource::new([file("a")], 3);
        source.enable_publishing();
        source.mark_pruned();
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut a = source.stream();
        assert!(matches!(a.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        // Two idle streams, nothing queued.
        assert_eq!(source.split_width(10), 3);
        source.publish([file("a-1"), file("a-2")]);
        // The two published units already cover the two idle streams.
        assert_eq!(source.split_width(10), 1);
    }

    #[test]
    fn idle_stream_waits_for_a_planning_sibling_and_takes_published_work() {
        let source = SharedWorkSource::new([file("a")], 2);
        source.enable_publishing();
        let (_, waker_a) = waker();
        let (wakes_b, waker_b) = waker();
        let mut cx_a = Context::from_waker(&waker_a);
        let mut cx_b = Context::from_waker(&waker_b);
        let mut a = source.stream();
        let mut b = source.stream();

        assert_eq!(path(a.poll_next_file(&mut cx_a)), Poll::Ready(Some("a".into())));
        // `a` still plans its file, so `b` waits instead of finishing.
        assert_eq!(path(b.poll_next_file(&mut cx_b)), Poll::Pending);

        source.publish([file("a-1")]);
        assert_eq!(wakes_b.0.load(Ordering::SeqCst), 1);
        assert_eq!(path(b.poll_next_file(&mut cx_b)), Poll::Ready(Some("a-1".into())));

        // Both streams finish only when no file is planned any more.
        a.planning_finished();
        assert_eq!(path(a.poll_next_file(&mut cx_a)), Poll::Pending);
        b.planning_finished();
        assert_eq!(path(b.poll_next_file(&mut cx_b)), Poll::Ready(None));
        assert_eq!(path(a.poll_next_file(&mut cx_a)), Poll::Ready(None));
    }

    #[test]
    fn waiting_stream_is_woken_when_the_last_planner_finishes() {
        let source = SharedWorkSource::new([file("a")], 2);
        source.enable_publishing();
        let (_, waker_a) = waker();
        let (wakes_b, waker_b) = waker();
        let mut cx_a = Context::from_waker(&waker_a);
        let mut cx_b = Context::from_waker(&waker_b);
        let mut a = source.stream();
        let mut b = source.stream();
        assert!(matches!(a.poll_next_file(&mut cx_a), Poll::Ready(Some(_))));
        assert_eq!(path(b.poll_next_file(&mut cx_b)), Poll::Pending);
        a.planning_finished();
        assert_eq!(wakes_b.0.load(Ordering::SeqCst), 1);
        assert_eq!(path(b.poll_next_file(&mut cx_b)), Poll::Ready(None));
    }

    #[test]
    fn dropping_a_planning_stream_releases_waiters() {
        let source = SharedWorkSource::new([file("a")], 2);
        source.enable_publishing();
        let (_, waker_a) = waker();
        let (wakes_b, waker_b) = waker();
        let mut cx_a = Context::from_waker(&waker_a);
        let mut cx_b = Context::from_waker(&waker_b);
        let mut a = source.stream();
        let mut b = source.stream();
        assert!(matches!(a.poll_next_file(&mut cx_a), Poll::Ready(Some(_))));
        assert_eq!(path(b.poll_next_file(&mut cx_b)), Poll::Pending);
        drop(a);
        assert_eq!(wakes_b.0.load(Ordering::SeqCst), 1);
        assert_eq!(path(b.poll_next_file(&mut cx_b)), Poll::Ready(None));
        let queue = source.inner.queue.lock();
        assert_eq!((queue.active, queue.planning, queue.waiters.len()), (0, 0, 0));
    }

    #[test]
    fn without_publishing_an_empty_queue_finishes_at_once() {
        let source = SharedWorkSource::new([file("a")], 2);
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut a = source.stream();
        let mut b = source.stream();
        assert!(matches!(a.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        assert_eq!(path(b.poll_next_file(&mut cx)), Poll::Ready(None));
        let queue = source.inner.queue.lock();
        assert_eq!((queue.active, queue.planning), (0, 0));
    }
}
