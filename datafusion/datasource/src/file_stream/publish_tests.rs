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

//! Longbow (lane Q39): `FileStream` terminal paths with publishing enabled.
//!
//! Each test holds one stream's planner on gated I/O while idle siblings wait
//! for its published work, then ends that planner by an error, an empty file,
//! a drop or a publication. Streams are polled explicitly with counting
//! wakers: no sleeps and no scheduler.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use arrow::array::{AsArray, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use datafusion_common::{Result, exec_err};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::channel::oneshot;
use futures::stream::{self, BoxStream, StreamExt};

use super::builder::FileStreamBuilder;
use super::work_source::SharedWorkSource;
use super::{FileStream, OnError};
use crate::file_groups::FileGroup;
use crate::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use crate::morsel::{Morsel, MorselPlan, MorselPlanner, Morselizer};
use crate::test_util::MockSource;
use crate::{PartitionedFile, TableSchema};

/// How a gated file's planning I/O completes.
enum Signal {
    /// Publish these files, then read these rows.
    Publish(Vec<&'static str>, Vec<i32>),
    /// The file has no rows (e.g. pruned): the planner returns no morsel.
    Empty,
    /// The I/O fails.
    Fail,
}

/// A morselizer that enables publishing. Gated files wait for a [`Signal`];
/// other files read their rows at once.
#[derive(Debug, Clone, Default)]
struct PublishingMorselizer {
    source: Option<SharedWorkSource>,
    rows: Arc<HashMap<&'static str, Vec<i32>>>,
    gates: Arc<Mutex<HashMap<String, oneshot::Receiver<Signal>>>>,
}

impl Morselizer for PublishingMorselizer {
    fn plan_file(&self, file: PartitionedFile) -> Result<Box<dyn MorselPlanner>> {
        let name = file.object_meta.location.to_string();
        let gate = self.gates.lock().unwrap().remove(&name);
        Ok(match gate {
            Some(signal) => Box::new(GatedPlanner {
                signal: Some(signal),
                source: self.source.clone().expect("publishing enabled"),
            }),
            None => Box::new(RowsPlanner(self.rows[name.as_str()].clone())),
        })
    }

    fn attach_shared_work_source(&mut self, source: &SharedWorkSource) {
        source.enable_publishing();
        self.source = Some(source.clone());
    }
}

#[derive(Debug)]
struct GatedPlanner {
    signal: Option<oneshot::Receiver<Signal>>,
    source: SharedWorkSource,
}

impl MorselPlanner for GatedPlanner {
    fn plan(mut self: Box<Self>) -> Result<Option<MorselPlan>> {
        let signal = self.signal.take().expect("planned once");
        let source = self.source;
        Ok(Some(MorselPlan::new().with_pending_planner(async move {
            match signal.await.expect("signal sent") {
                Signal::Publish(files, rows) => {
                    source.publish(files.into_iter().map(|f| PartitionedFile::new(f, 10)));
                    Ok(Box::new(RowsPlanner(rows)) as Box<dyn MorselPlanner>)
                }
                Signal::Empty => Ok(Box::new(EmptyPlanner) as Box<dyn MorselPlanner>),
                Signal::Fail => exec_err!("planning I/O failed"),
            }
        })))
    }
}

#[derive(Debug)]
struct RowsPlanner(Vec<i32>);

impl MorselPlanner for RowsPlanner {
    fn plan(self: Box<Self>) -> Result<Option<MorselPlan>> {
        Ok(Some(
            MorselPlan::new().with_morsels(vec![Box::new(RowsMorsel(self.0))]),
        ))
    }
}

#[derive(Debug)]
struct EmptyPlanner;

impl MorselPlanner for EmptyPlanner {
    fn plan(self: Box<Self>) -> Result<Option<MorselPlan>> {
        Ok(None)
    }
}

#[derive(Debug)]
struct RowsMorsel(Vec<i32>);

impl Morsel for RowsMorsel {
    fn into_stream(self: Box<Self>) -> BoxStream<'static, Result<RecordBatch>> {
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(self.0))]).unwrap();
        stream::iter([Ok(batch)]).boxed()
    }
}

#[derive(Default)]
struct CountingWaker(AtomicUsize);

impl Wake for CountingWaker {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// One sibling stream with its own counting waker.
struct Sibling {
    stream: FileStream,
    wakes: Arc<CountingWaker>,
}

impl Sibling {
    fn poll(&mut self) -> Poll<Option<Result<RecordBatch>>> {
        let waker = Waker::from(Arc::clone(&self.wakes));
        self.stream.poll_next_unpin(&mut Context::from_waker(&waker))
    }

    fn wakes(&self) -> usize {
        self.wakes.0.load(Ordering::SeqCst)
    }
}

fn rows(poll: Poll<Option<Result<RecordBatch>>>) -> Vec<i32> {
    match poll {
        Poll::Ready(Some(Ok(batch))) => {
            batch.column(0).as_primitive::<Int32Type>().values().to_vec()
        }
        other => panic!("expected a batch, got {other:?}"),
    }
}

/// `streams` siblings over a queue that holds the gated file `a`, plus a
/// sender for its signal.
fn siblings(
    streams: usize,
    rows: HashMap<&'static str, Vec<i32>>,
) -> (Vec<Sibling>, oneshot::Sender<Signal>) {
    let mut groups: Vec<FileGroup> = vec![vec![PartitionedFile::new("a", 10)].into()];
    groups.resize_with(streams, || Vec::new().into());
    let config: FileScanConfig = FileScanConfigBuilder::new(
        ObjectStoreUrl::parse("test:///").unwrap(),
        Arc::new(MockSource::new(TableSchema::from(Arc::new(Schema::new(
            vec![Field::new("i", DataType::Int32, false)],
        ))))),
    )
    .with_file_groups(groups)
    .build();
    let source = SharedWorkSource::from_config(&config);
    let (sender, signal) = oneshot::channel();
    let mut morselizer = PublishingMorselizer {
        rows: Arc::new(rows),
        ..Default::default()
    };
    morselizer
        .gates
        .lock()
        .unwrap()
        .insert("a".to_string(), signal);
    morselizer.attach_shared_work_source(&source);
    let metrics = ExecutionPlanMetricsSet::new();
    let siblings = (0..streams)
        .map(|partition| Sibling {
            stream: FileStreamBuilder::new(&config)
                .with_partition(partition)
                .with_shared_work_source(Some(source.clone()))
                .with_morselizer(Box::new(morselizer.clone()))
                .with_metrics(&metrics)
                .with_on_error(OnError::Fail)
                .build()
                .unwrap(),
            wakes: Arc::default(),
        })
        .collect();
    (siblings, sender)
}

/// Stream 0 takes `a` and waits on its I/O; stream 1 waits for published work.
fn hold_and_wait(siblings: &mut [Sibling]) {
    assert!(siblings[0].poll().is_pending(), "planner waits on I/O");
    assert!(siblings[1].poll().is_pending(), "sibling waits for work");
    assert_eq!(siblings[1].wakes(), 0);
}

#[test]
fn planner_io_failure_fails_its_stream_and_releases_waiters() {
    let (mut siblings, signal) = siblings(2, HashMap::new());
    hold_and_wait(&mut siblings);
    signal.send(Signal::Fail).ok().unwrap();
    assert_eq!(siblings[0].wakes(), 1);
    match siblings[0].poll() {
        Poll::Ready(Some(Err(error))) => {
            assert!(error.to_string().contains("planning I/O failed"))
        }
        other => panic!("expected the planning error, got {other:?}"),
    }
    // The failed stream dropped its scan state, so the waiter finishes.
    assert_eq!(siblings[1].wakes(), 1);
    assert!(matches!(siblings[1].poll(), Poll::Ready(None)));
    assert!(matches!(siblings[0].poll(), Poll::Ready(None)));
}

#[test]
fn empty_file_finishes_planning_and_releases_waiters() {
    let (mut siblings, signal) = siblings(2, HashMap::new());
    hold_and_wait(&mut siblings);
    signal.send(Signal::Empty).ok().unwrap();
    assert!(matches!(siblings[0].poll(), Poll::Ready(None)));
    assert_eq!(siblings[1].wakes(), 1);
    assert!(matches!(siblings[1].poll(), Poll::Ready(None)));
}

#[test]
fn dropping_the_planning_stream_releases_waiters() {
    let (mut siblings, _signal) = siblings(2, HashMap::new());
    hold_and_wait(&mut siblings);
    let waiter = siblings.pop().unwrap();
    drop(siblings);
    assert_eq!(waiter.wakes(), 1);
    let mut waiter = waiter;
    assert!(matches!(waiter.poll(), Poll::Ready(None)));
}

#[test]
fn published_work_is_read_once_after_a_waiting_sibling_is_dropped() {
    let (mut siblings, signal) = siblings(3, HashMap::from([("a-1", vec![2]), ("a-2", vec![3])]));
    hold_and_wait(&mut siblings);
    assert!(siblings[2].poll().is_pending());
    // Drop a waiting sibling before anything is published.
    let mut planner = siblings.remove(0);
    drop(siblings.remove(0));
    let mut waiter = siblings.remove(0);

    signal
        .send(Signal::Publish(vec!["a-1", "a-2"], vec![1]))
        .ok()
        .unwrap();
    assert_eq!(rows(planner.poll()), vec![1]);
    // Publication woke the remaining waiter, which takes the first unit.
    assert_eq!(waiter.wakes(), 1);
    assert_eq!(rows(waiter.poll()), vec![2]);
    assert_eq!(rows(planner.poll()), vec![3]);
    assert!(matches!(waiter.poll(), Poll::Ready(None)));
    assert!(matches!(planner.poll(), Poll::Ready(None)));
}
