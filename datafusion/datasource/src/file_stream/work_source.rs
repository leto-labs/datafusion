// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

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
/// its own files first, then takes units published by a format reader, then
/// another owner's unopened file (the last file of the longest queue). Parquet
/// publishes disjoint runs of whole, unopened row groups after pruning.
///
/// Each file a stream takes while publishing is enabled carries a
/// [`PublishLease`] extension. The format releases the lease as soon as it
/// knows that it cannot publish from that file any more (Parquet: the reader
/// is not eligible, its unread tail became too short, or the stream ended);
/// the stream releases it at the latest when it takes its next file or is
/// dropped. An idle stream waits only while some lease is held, because only
/// such a reader can still offer work; otherwise it finishes.
///
/// Liveness obligation: a waiting stream makes progress only when a lease
/// holder is polled. A consumer that polls one partition and then waits on
/// another must either poll every partition or run the scan with
/// `TaskContext::requires_independent_partitions` (no waits at all).
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
    /// Held [`PublishLease`]s: taken files whose reader may still publish.
    leases: usize,
    waiters: HashMap<usize, Waker>,
}

impl SharedQueue {
    fn wakers(&self) -> Vec<Waker> {
        self.waiters.values().cloned().collect()
    }

    fn take_waiters(&mut self) -> Vec<Waker> {
        self.waiters.drain().map(|(_, waker)| waker).collect()
    }

    /// Another owner's unopened file: the last of the longest queue, so the
    /// owner keeps reading its queue from the front.
    fn steal_unopened(&mut self) -> Option<PartitionedFile> {
        self.owned
            .iter_mut()
            .max_by_key(|files| files.len())
            .and_then(VecDeque::pop_back)
    }
}

fn wake_all(wakers: Vec<Waker>) {
    wakers.into_iter().for_each(Waker::wake);
}

/// A taken file's right to make idle siblings wait for its publications
/// (Longbow family WS). Attached to the file as a `PartitionedFile`
/// extension. Clones share one lease; releasing is idempotent, and the last
/// clone releases it on drop.
#[derive(Debug, Clone)]
pub struct PublishLease(Arc<LeaseInner>);

#[derive(Debug)]
struct LeaseInner {
    source: std::sync::Weak<SharedWorkSourceInner>,
    released: AtomicBool,
}

impl PublishLease {
    /// The reader will publish nothing more from this file. Siblings that
    /// wait only for this lease finish.
    pub fn release(&self) {
        self.0.release();
    }
}

impl LeaseInner {
    fn release(&self) {
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        let Some(source) = self.source.upgrade() else {
            return;
        };
        let wakers = {
            let mut queue = source.queue.lock();
            queue.leases -= 1;
            if queue.leases == 0 {
                queue.take_waiters()
            } else {
                vec![]
            }
        };
        wake_all(wakers);
    }
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        self.release();
    }
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

    #[cfg(test)]
    fn from_test_groups(groups: Vec<Vec<PartitionedFile>>) -> Self {
        Self::from_groups(groups.into_iter().map(VecDeque::from).collect())
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

    pub(super) fn stream(&self, partition: usize, independent: bool) -> SharedStream {
        SharedStream {
            source: self.clone(),
            id: partition,
            held: None,
            independent,
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

/// One output partition's handle. Its file's lease stays held until the
/// format releases it, the stream takes its next file, or the stream is
/// dropped on error, limit or cancellation.
#[derive(Debug)]
pub(super) struct SharedStream {
    source: SharedWorkSource,
    id: usize,
    held: Option<PublishLease>,
    /// A selectively polled consumer cannot wait for a buffered sibling.
    independent: bool,
}

impl SharedStream {
    fn poll_next_file(&mut self, cx: &mut Context<'_>) -> Poll<Option<PartitionedFile>> {
        // The previous file is finished: it can publish nothing more.
        if let Some(lease) = self.held.take() {
            lease.release();
        }
        let publishing = self.source.inner.publishing.load(Ordering::Acquire);
        let mut queue = self.source.inner.queue.lock();
        let file = if let Some(global) = &mut queue.global {
            global.pop_front().or_else(|| queue.files.pop_front())
        } else {
            queue.owned[self.id]
                .pop_front()
                .or_else(|| queue.files.pop_front())
                // No unit is queued: take another owner's unopened file
                // rather than idle while that owner reads its files alone.
                .or_else(|| queue.steal_unopened())
        };
        let mut lease = None;
        let result = if let Some(file) = file {
            if publishing {
                // An independent stream never waits, so nobody waits on it
                // either: its consumer may not poll it while a sibling waits.
                // It gets an already released lease, which also replaces a
                // donor's lease carried by a published unit.
                let counted = !self.independent;
                if counted {
                    queue.leases += 1;
                }
                lease = Some(PublishLease(Arc::new(LeaseInner {
                    source: Arc::downgrade(&self.source.inner),
                    released: AtomicBool::new(!counted),
                })));
            }
            queue.waiters.remove(&self.id);
            Poll::Ready(Some(file))
        } else if !self.independent && publishing && queue.leases > 0 {
            queue.waiters.insert(self.id, cx.waker().clone());
            Poll::Pending
        } else {
            queue.waiters.remove(&self.id);
            Poll::Ready(None)
        };
        self.source.update_idle(&queue);
        drop(queue);
        // Attach outside the lock: a published unit's previous lease (its
        // donor's) is replaced here, and dropping the last clone of a lease
        // takes the queue lock.
        match (result, lease) {
            (Poll::Ready(Some(file)), Some(lease)) => {
                if !self.independent {
                    self.held = Some(lease.clone());
                }
                Poll::Ready(Some(file.with_extension(lease)))
            }
            (result, _) => result,
        }
    }
}

impl Drop for SharedStream {
    fn drop(&mut self) {
        let abandoned = {
            let mut queue = self.source.inner.queue.lock();
            queue.waiters.remove(&self.id);
            let abandoned = std::mem::take(&mut queue.owned[self.id]);
            self.source.update_idle(&queue);
            abandoned
        };
        if let Some(lease) = self.held.take() {
            lease.release();
        }
        // File extensions can own other resources. Release them outside the
        // queue lock, just as we invoke wakers outside it.
        drop(abandoned);
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
        let mut b = source.stream(1, false);
        let mut a = source.stream(0, false);
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
        let mut b = source.stream(1, false);
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
    fn independent_stream_does_not_wait_but_can_take_published_work() {
        let source = SharedWorkSource::new([file("owned")], 2);
        source.enable_publishing();
        let mut owner = source.stream(0, true);
        let mut idle = source.stream(1, true);
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        assert_eq!(
            path(owner.poll_next_file(&mut cx)),
            Poll::Ready(Some("owned".into()))
        );
        // The owner may now be blocked behind the selective consumer. An idle
        // sibling must finish without requiring any further owner progress.
        assert_eq!(path(idle.poll_next_file(&mut cx)), Poll::Ready(None));
        assert_eq!(source.split_width(10), 1);
        // Units already removed from a donor remain claimable exactly once.
        source.publish([file("donated")]);
        assert_eq!(
            path(owner.poll_next_file(&mut cx)),
            Poll::Ready(Some("donated".into()))
        );
        assert_eq!(path(owner.poll_next_file(&mut cx)), Poll::Ready(None));
    }

    #[test]
    fn idle_stream_takes_another_owners_unopened_file_when_no_unit_is_queued() {
        let source = SharedWorkSource::new([file("a"), file("b")], 3);
        source.enable_publishing();
        let mut a = source.stream(0, false);
        let mut idle = source.stream(2, false);
        let (wakes, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(a.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        // Nothing is queued: take owner 1's unopened file instead of waiting.
        assert_eq!(
            path(idle.poll_next_file(&mut cx)),
            Poll::Ready(Some("b".into()))
        );
        // `b` is finished; `a` still holds its lease and may publish.
        assert_eq!(path(idle.poll_next_file(&mut cx)), Poll::Pending);
        drop(a);
        assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
        assert_eq!(path(idle.poll_next_file(&mut cx)), Poll::Ready(None));
    }

    /// Longbow review (WS balance): group 0 holds 20 one-row-group files and
    /// group 1 one file. Partition 1 takes partition 0's unopened files instead
    /// of idling while partition 0 reads 19 files alone.
    #[test]
    fn skewed_groups_are_shared_by_taking_unopened_files() {
        let source = SharedWorkSource::from_test_groups(vec![
            (0..20).map(|i| file(&format!("a{i}"))).collect(),
            vec![file("b")],
        ]);
        source.enable_publishing();
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut streams = [source.stream(0, false), source.stream(1, false)];
        let mut read: [Vec<String>; 2] = [vec![], vec![]];
        let mut done = [false, false];
        while !done.iter().all(|done| *done) {
            for (partition, stream) in streams.iter_mut().enumerate() {
                if done[partition] {
                    continue;
                }
                match path(stream.poll_next_file(&mut cx)) {
                    Poll::Ready(Some(file)) => read[partition].push(file),
                    Poll::Ready(None) => done[partition] = true,
                    Poll::Pending => {}
                }
            }
        }
        assert_eq!(read[0].len() + read[1].len(), 21);
        assert!(read[0].len().abs_diff(read[1].len()) <= 1, "{read:?}");
        let mut all: Vec<_> = read.concat();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 21, "every file is read exactly once");
    }

    /// A format releases a file's lease when it knows it cannot publish from
    /// it (for example a single row group). Idle siblings then finish while
    /// the owner keeps reading.
    /// Longbow review v3: with mixed independence, a stream that waits must
    /// not wait on a file held by an independent stream, whose consumer may
    /// not poll it. Independent streams hold no counted lease, also for a
    /// published unit that carried its donor's lease.
    #[test]
    fn independent_streams_hold_no_lease_that_others_wait_on() {
        let source = SharedWorkSource::new([file("a")], 2);
        source.enable_publishing();
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut independent = source.stream(0, true);
        let mut dependent = source.stream(1, false);
        let Poll::Ready(Some(taken)) = independent.poll_next_file(&mut cx) else {
            panic!("the independent stream takes its file")
        };
        // Its lease is attached but already released: it counts for nobody.
        assert!(taken.extension::<PublishLease>().is_some());
        assert_eq!(source.inner.queue.lock().leases, 0);
        // Only an independent holder is left: the dependent stream finishes
        // instead of waiting on a stream its consumer may never poll.
        assert_eq!(path(dependent.poll_next_file(&mut cx)), Poll::Ready(None));

        // A published unit carrying a dependent donor's lease loses it when an
        // independent stream takes it.
        let source = SharedWorkSource::new([file("donor")], 2);
        source.enable_publishing();
        let mut donor = source.stream(0, false);
        let mut independent = source.stream(1, true);
        let Poll::Ready(Some(donor_file)) = donor.poll_next_file(&mut cx) else {
            panic!("the donor takes its file")
        };
        source.publish([file("unit")
            .with_extension(donor_file.extension::<PublishLease>().unwrap().clone())]);
        let Poll::Ready(Some(unit)) = independent.poll_next_file(&mut cx) else {
            panic!("the independent stream takes the unit")
        };
        assert_eq!(source.inner.queue.lock().leases, 1, "only the donor's");
        // Releasing the unit's lease (its reader cannot publish) leaves the
        // donor's lease held.
        unit.extension::<PublishLease>().unwrap().release();
        assert_eq!(source.inner.queue.lock().leases, 1);
        drop(independent);
        drop(donor_file);
        drop(donor);
        assert_eq!(source.inner.queue.lock().leases, 0);
    }

    #[test]
    fn released_lease_lets_idle_streams_finish_before_the_owner() {
        let source = SharedWorkSource::new([file("a")], 2);
        source.enable_publishing();
        let mut a = source.stream(0, false);
        let mut b = source.stream(1, false);
        let (wakes, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let Poll::Ready(Some(taken)) = a.poll_next_file(&mut cx) else {
            panic!("owner takes its file")
        };
        assert_eq!(path(b.poll_next_file(&mut cx)), Poll::Pending);
        let lease = taken.extension::<PublishLease>().expect("lease attached");
        lease.clone().release();
        lease.release();
        assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
        assert_eq!(path(b.poll_next_file(&mut cx)), Poll::Ready(None));
        // Releasing twice, then taking the next file, keeps the count exact.
        assert_eq!(path(a.poll_next_file(&mut cx)), Poll::Ready(None));
        assert_eq!(source.inner.queue.lock().leases, 0);
    }

    #[test]
    fn idle_stream_waits_for_row_groups_after_a_sibling_started_reading() {
        let source = SharedWorkSource::new([file("a")], 2);
        source.enable_publishing();
        let (_, waker) = waker();
        let (wakes, waiting) = self::waker();
        let mut cx = Context::from_waker(&waker);
        let mut wait_cx = Context::from_waker(&waiting);
        let mut a = source.stream(0, false);
        let mut b = source.stream(1, false);
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
    fn a_taken_unopened_file_can_publish_to_its_former_owner() {
        let source = SharedWorkSource::new([file("a"), file("b"), file("c")], 2);
        source.enable_publishing();
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut a = source.stream(0, false);
        let mut b = source.stream(1, false);
        assert!(matches!(a.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        assert!(matches!(b.poll_next_file(&mut cx), Poll::Ready(Some(_))));
        assert_eq!(source.split_width(12), 1);
        // `b` takes owner 0's unopened `c`; `a` then waits for `c`'s reader.
        assert_eq!(
            path(b.poll_next_file(&mut cx)),
            Poll::Ready(Some("c".into()))
        );
        assert_eq!(path(a.poll_next_file(&mut cx)), Poll::Pending);
        assert_eq!(source.split_width(12), 2);
        source.publish([file("c-tail")]);
        assert_eq!(
            path(a.poll_next_file(&mut cx)),
            Poll::Ready(Some("c-tail".into()))
        );
    }

    #[test]
    fn sequential_stream_takes_unopened_file_when_no_reader_is_active() {
        let source = SharedWorkSource::new([file("a"), file("b")], 2);
        source.enable_publishing();
        let (_, waker) = waker();
        let mut cx = Context::from_waker(&waker);
        let mut a = source.stream(0, false);
        assert_eq!(
            path(a.poll_next_file(&mut cx)),
            Poll::Ready(Some("a".into()))
        );
        assert_eq!(
            path(a.poll_next_file(&mut cx)),
            Poll::Ready(Some("b".into()))
        );
        assert_eq!(path(a.poll_next_file(&mut cx)), Poll::Ready(None));
        let mut b = source.stream(1, false);
        assert_eq!(path(b.poll_next_file(&mut cx)), Poll::Ready(None));
    }

    #[test]
    fn idle_stream_can_claim_an_unopened_owner_before_it_is_polled() {
        let source = SharedWorkSource::new([file("a")], 2);
        source.enable_publishing();
        let a = source.stream(0, false);
        let mut b = source.stream(1, false);
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
        let mut a = source.stream(0, false);
        let mut b = source.stream(1, false);
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
        let mut a = source.stream(0, false);
        let mut b = source.stream(1, false);
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
                        let mut stream = source.stream(partition, false);
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
