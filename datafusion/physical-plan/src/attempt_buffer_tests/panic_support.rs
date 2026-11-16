// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use arrow::array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::config::SessionConfig;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::PhysicalExpr;
use futures::Stream;
use futures::task::ArcWake;
use tokio::sync::oneshot;

use crate::attempt_tasks::{AttemptTasks, CaughtPollPanic};
use crate::buffer::BufferExec;
use crate::stream::RecordBatchStreamAdapter;
use crate::test::exec::PanicExec;
use crate::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use super::support::{Gate, INPUT_BYTES, schema};

pub const PANIC_DIAGNOSTIC: &str =
    "BufferExec input stream panicked: PanickingStream did panic: 0";

pub fn context(
    pool: &Arc<dyn MemoryPool>,
    scope: Option<Arc<AttemptTasks>>,
) -> Arc<TaskContext> {
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(pool))
        .build_arc()
        .unwrap();
    let config = match scope {
        Some(scope) => SessionConfig::new().with_extension(scope),
        None => SessionConfig::new(),
    };
    Arc::new(
        TaskContext::default()
            .with_session_config(config)
            .with_runtime(runtime),
    )
}

pub fn panic_plan(
    drop_gate: Option<Arc<Gate>>,
    poll_gate: Option<Arc<Gate>>,
) -> (BufferExec, Control) {
    controlled_plan(Arc::new(PanicExec::new(schema(), 1)), drop_gate, poll_gate)
}

pub fn ordinary_plan(drop_gate: Option<Arc<Gate>>) -> (BufferExec, Control) {
    let properties = Arc::clone(PanicExec::new(schema(), 1).properties());
    controlled_plan(Arc::new(ErrorExec(properties)), drop_gate, None)
}

fn controlled_plan(
    child: Arc<dyn ExecutionPlan>,
    drop_gate: Option<Arc<Gate>>,
    poll_gate: Option<Arc<Gate>>,
) -> (BufferExec, Control) {
    let (start_tx, start_rx) = oneshot::channel();
    let (waiting_tx, waiting_rx) = oneshot::channel();
    let (poll_tx, poll_rx) = oneshot::channel();
    let (entered_tx, entered_rx) = oneshot::channel();
    let (dropped_tx, dropped_rx) = oneshot::channel();
    let input = ControlledExec {
        child,
        signals: Mutex::new(Some(Signals {
            start: start_rx,
            waiting: Some(waiting_tx),
            poll_entered: Some(poll_tx),
            drop_entered: Some(entered_tx),
            dropped: Some(dropped_tx),
            drop_gate,
            poll_gate,
        })),
    };
    (
        BufferExec::new(Arc::new(input), 1024),
        Control {
            start: Some(start_tx),
            waiting: waiting_rx,
            poll_entered: poll_rx,
            drop_entered: entered_rx,
            dropped: dropped_rx,
        },
    )
}

/// MockExec clones only Execution errors. Return a real ordinary Internal here
/// so a broken fixture cannot itself create the panic used by the control.
#[derive(Debug)]
struct ErrorExec(Arc<PlanProperties>);

impl DisplayAs for ErrorExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ErrorExec")
    }
}

impl ExecutionPlan for ErrorExec {
    fn name(&self) -> &str {
        "ErrorExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.0
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Err(DataFusionError::Internal(
            "test error plan cannot be rewritten".into(),
        ))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        assert_eq!(partition, 0);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema(),
            futures::stream::iter([Err(DataFusionError::Internal(
                PANIC_DIAGNOSTIC.into(),
            ))]),
        )))
    }
}

pub struct Control {
    pub start: Option<oneshot::Sender<()>>,
    pub waiting: oneshot::Receiver<()>,
    pub poll_entered: oneshot::Receiver<()>,
    pub drop_entered: oneshot::Receiver<()>,
    pub dropped: oneshot::Receiver<()>,
}

struct Signals {
    start: oneshot::Receiver<()>,
    waiting: Option<oneshot::Sender<()>>,
    poll_entered: Option<oneshot::Sender<()>>,
    drop_entered: Option<oneshot::Sender<()>>,
    dropped: Option<oneshot::Sender<()>>,
    drop_gate: Option<Arc<Gate>>,
    poll_gate: Option<Arc<Gate>>,
}

/// Delegates the actual input poll to the existing PanicExec fixture. This
/// wrapper supplies only ownership and gates, never panic provenance or errors.
struct ControlledExec {
    child: Arc<dyn ExecutionPlan>,
    signals: Mutex<Option<Signals>>,
}

impl fmt::Debug for ControlledExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ControlledExec")
            .field("child", &self.child)
            .finish()
    }
}

impl DisplayAs for ControlledExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ControlledExec")
    }
}

impl ExecutionPlan for ControlledExec {
    fn name(&self) -> &str {
        "ControlledExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.child.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.child]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Err(DataFusionError::Internal(
            "test gate plan cannot be rewritten".into(),
        ))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let reservation =
            MemoryConsumer::new("gated-panic-input").register(context.memory_pool());
        reservation.try_grow(INPUT_BYTES)?;
        let input = self.child.execute(partition, context)?;
        let signals = self.signals.lock().unwrap().take().expect("execute once");
        Ok(Box::pin(ControlledStream {
            input: Some(input),
            reservation: Some(reservation),
            signals,
            started: false,
        }))
    }
}

struct ControlledStream {
    input: Option<SendableRecordBatchStream>,
    reservation: Option<MemoryReservation>,
    signals: Signals,
    started: bool,
}

impl Stream for ControlledStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if !self.started {
            // Register the start waiter before telling the test it may release.
            let start = Pin::new(&mut self.signals.start).poll(cx);
            if let Some(waiting) = self.signals.waiting.take() {
                let _ = waiting.send(());
            }
            match start {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(_)) => return Poll::Ready(None),
                Poll::Ready(Ok(())) => self.started = true,
            }
            if let Some(entered) = self.signals.poll_entered.take() {
                let _ = entered.send(());
            }
            if let Some(gate) = &self.signals.poll_gate {
                gate.wait();
            }
        }
        self.input.as_mut().unwrap().as_mut().poll_next(cx)
    }
}

impl RecordBatchStream for ControlledStream {
    fn schema(&self) -> SchemaRef {
        schema()
    }
}

impl Drop for ControlledStream {
    fn drop(&mut self) {
        if let Some(entered) = self.signals.drop_entered.take() {
            let _ = entered.send(());
        }
        if let Some(gate) = &self.signals.drop_gate {
            gate.wait();
        }
        drop(self.input.take());
        drop(self.reservation.take());
        if let Some(dropped) = self.signals.dropped.take() {
            let _ = dropped.send(());
        }
    }
}

/// The channel's synchronous send wake records the marker BEFORE the producer
/// can continue past forwarding. An eventual marker check alone misses ordering.
pub struct ForwardWake {
    scope: Arc<AttemptTasks>,
    observed: Mutex<Option<oneshot::Sender<bool>>>,
}

impl ForwardWake {
    pub fn new(scope: &Arc<AttemptTasks>) -> (Arc<Self>, oneshot::Receiver<bool>) {
        let (tx, rx) = oneshot::channel();
        (
            Arc::new(Self {
                scope: Arc::clone(scope),
                observed: Mutex::new(Some(tx)),
            }),
            rx,
        )
    }
}

impl ArcWake for ForwardWake {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        if let Some(observed) = arc_self.observed.lock().unwrap().take() {
            let _ = observed.send(arc_self.scope.poll_panicked());
        }
    }
}

pub fn has_caught_panic(error: &(dyn Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if error.downcast_ref::<CaughtPollPanic>().is_some() {
            return true;
        }
        current = error.source();
    }
    false
}

pub fn assert_caught_panic(error: &DataFusionError) {
    let mut current: Option<&(dyn Error + 'static)> = Some(error);
    while let Some(cause) = current {
        if let Some(panic) = cause.downcast_ref::<CaughtPollPanic>() {
            assert!(panic.to_string().contains(PANIC_DIAGNOSTIC));
            let original = panic
                .source()
                .unwrap()
                .downcast_ref::<DataFusionError>()
                .unwrap();
            assert!(
                matches!(original, DataFusionError::Internal(message) if message.starts_with(PANIC_DIAGNOSTIC))
            );
            return;
        }
        current = cause.source();
    }
    panic!("real BufferExec panic has no CaughtPollPanic in its source chain: {error}");
}
