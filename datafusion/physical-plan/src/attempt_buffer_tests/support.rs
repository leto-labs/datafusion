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

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use arrow::array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::memory_pool::{
    GreedyMemoryPool, MemoryConsumer, MemoryPool, MemoryReservation,
};
use datafusion_execution::{RecordBatchStream, SendableRecordBatchStream};
use futures::{FutureExt, Stream};
use tokio::sync::oneshot;

use crate::attempt_tasks::AttemptTasks;
use crate::buffer::MemoryBufferedStream;
use crate::common::spawn_buffered_with_scope;
use crate::stream::RecordBatchStreamAdapter;

pub const DEADLOCK_BOUND: Duration = Duration::from_secs(10);
pub const INPUT_BYTES: usize = 128;

pub async fn bounded<F: Future>(future: F) -> F::Output {
    tokio::time::timeout(DEADLOCK_BOUND, future)
        .await
        .expect("explicit test gate did not progress")
}

pub fn assert_pending<F: Future>(future: Pin<&mut F>) {
    assert!(
        future.now_or_never().is_none(),
        "drain certified live input"
    );
}

/// Synchronous gate for one worker or one input destructor. The test owns a
/// release guard separately, so failed assertions cannot strand runtime teardown.
#[derive(Default)]
pub struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    pub fn wait(&self) {
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.changed.wait(open).unwrap();
        }
    }

    pub fn release(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }

    pub fn release_on_drop(self: &Arc<Self>) -> Release {
        Release(Arc::clone(self))
    }
}

pub struct Release(Arc<Gate>);

impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}

pub fn pool() -> Arc<dyn MemoryPool> {
    Arc::new(GreedyMemoryPool::new(4096))
}

pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int32,
        false,
    )]))
}

pub fn batch() -> RecordBatch {
    RecordBatch::try_new(schema(), vec![Arc::new(Int32Array::from(vec![7]))]).unwrap()
}

#[derive(Clone, Copy, Debug)]
pub enum Helper {
    Common,
    Memory,
}

impl Helper {
    pub fn start(
        self,
        input: SendableRecordBatchStream,
        pool: &Arc<dyn MemoryPool>,
        scope: Option<Arc<AttemptTasks>>,
    ) -> Result<SendableRecordBatchStream> {
        match self {
            Self::Common => spawn_buffered_with_scope(input, 2, scope),
            Self::Memory => {
                let schema = input.schema();
                let reservation = MemoryConsumer::new("test-buffer").register(pool);
                let stream = MemoryBufferedStream::new_with_scope(input, 1024, reservation, scope)?;
                Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
            }
        }
    }
}

pub struct Probe {
    /// Sent after any optional first batch has been queued, on the next poll.
    pub pending: oneshot::Receiver<()>,
    pub drop_entered: oneshot::Receiver<()>,
    pub dropped: oneshot::Receiver<()>,
    pub polls: Arc<AtomicUsize>,
}

pub fn input(
    pool: &Arc<dyn MemoryPool>,
    first_batch: bool,
    drop_gate: Option<Arc<Gate>>,
) -> (SendableRecordBatchStream, Probe) {
    let (pending_tx, pending) = oneshot::channel();
    let (entered_tx, drop_entered) = oneshot::channel();
    let (dropped_tx, dropped) = oneshot::channel();
    let polls = Arc::new(AtomicUsize::new(0));
    let mut reservation = MemoryConsumer::new("test-input").register(pool);
    reservation.try_grow(INPUT_BYTES).unwrap();
    let stream = ProbeStream {
        first: first_batch.then(batch),
        pending: Some(pending_tx),
        drop_entered: Some(entered_tx),
        dropped: Some(dropped_tx),
        polls: Arc::clone(&polls),
        reservation: Some(reservation),
        drop_gate,
    };
    (
        Box::pin(stream),
        Probe {
            pending,
            drop_entered,
            dropped,
            polls,
        },
    )
}

struct ProbeStream {
    first: Option<RecordBatch>,
    pending: Option<oneshot::Sender<()>>,
    drop_entered: Option<oneshot::Sender<()>>,
    dropped: Option<oneshot::Sender<()>>,
    polls: Arc<AtomicUsize>,
    reservation: Option<MemoryReservation>,
    drop_gate: Option<Arc<Gate>>,
}

impl Stream for ProbeStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        if let Some(batch) = self.first.take() {
            return Poll::Ready(Some(Ok(batch)));
        }
        if let Some(pending) = self.pending.take() {
            let _ = pending.send(());
        }
        Poll::Pending
    }
}

impl RecordBatchStream for ProbeStream {
    fn schema(&self) -> SchemaRef {
        schema()
    }
}

impl Drop for ProbeStream {
    fn drop(&mut self) {
        if let Some(entered) = self.drop_entered.take() {
            let _ = entered.send(());
        }
        if let Some(gate) = &self.drop_gate {
            gate.wait();
        }
        drop(self.reservation.take());
        if let Some(dropped) = self.dropped.take() {
            let _ = dropped.send(());
        }
    }
}
