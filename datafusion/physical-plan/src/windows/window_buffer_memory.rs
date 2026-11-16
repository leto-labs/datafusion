// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Charge fixed-width global window buffering and its materialization peak.
//!
//! The input buffers stay charged until the window stream is dropped. A
//! per-row working allowance covers concatenated input, two copies of the
//! fixed-width result columns, validity/alignment and evaluation indices.
//! Variable-width/partitioned windows need a separate bound and are unchanged.
use std::sync::Arc;

use crate::WindowExpr;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::utils::memory::RecordBatchMemoryCounter;
use datafusion_common::{Result, resources_datafusion_err};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::window::PlainAggregateWindowExpr;

pub(super) struct WindowBufferMemory {
    buffers: RecordBatchMemoryCounter,
    work_per_row: usize,
    metadata_per_batch: usize,
    // Drop accounting metadata before releasing its charge.
    reservation: MemoryReservation,
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| resources_datafusion_err!("WindowAggExec memory bound overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| resources_datafusion_err!("WindowAggExec memory bound overflow"))
}

impl WindowBufferMemory {
    pub(super) fn try_new(
        schema: &SchemaRef,
        expressions: &[Arc<dyn WindowExpr>],
        context: &TaskContext,
    ) -> Result<Option<Self>> {
        if expressions.iter().any(|expr| {
            !expr.partition_by().is_empty()
                || !expr.order_by().is_empty()
                || !expr.get_window_frame().start_bound.is_unbounded()
                || !expr.get_window_frame().end_bound.is_unbounded()
        }) {
            return Ok(None);
        }
        let mut width = 0;
        for field in schema.fields() {
            let Some(bytes) = field.data_type().primitive_width() else {
                return Ok(None);
            };
            width = add(width, bytes)?;
        }
        // This covers buffer-counter hash buckets, ArrayData/RecordBatch/Vec
        // headers and per-column allocation padding, before any counter grows.
        let metadata_per_batch = add(mul(schema.fields().len(), 1024)?, 512)?;
        // Constant global aggregates take get_result_column's scalar-fill
        // path: no row-wise ScalarValue vector or ordering indices exist.
        // Four fixed-width copies cover concatenation, result construction
        // and result concatenation; one byte per field bounds validity bits.
        // Other window evaluators keep a checked allowance for the actual
        // ScalarValue vector (including spare capacity) and index scratch.
        // Eligibility above already requires two unbounded bounds and no
        // ordering. For PlainAggregateWindowExpr these are exactly the
        // scalar-fill conditions in the upstream evaluator.
        let constant = expressions
            .iter()
            .all(|e| e.as_any().is::<PlainAggregateWindowExpr>());
        let scratch = if constant {
            0
        } else {
            add(
                mul(size_of::<datafusion_common::ScalarValue>(), 2)?,
                mul(size_of::<usize>(), 2)?,
            )?
        };
        let work_per_row = add(add(mul(width, 4)?, schema.fields().len())?, scratch)?;
        let reservation =
            MemoryConsumer::new("WindowAggExecBuffer").register(context.memory_pool());
        reservation.try_grow(add(metadata_per_batch, 4096)?)?;
        Ok(Some(Self {
            reservation,
            buffers: RecordBatchMemoryCounter::new(),
            work_per_row,
            metadata_per_batch,
        }))
    }

    pub(super) fn retain(&mut self, batch: &RecordBatch) -> Result<()> {
        // Admit metadata and the full fixed-width evaluation working bound
        // before retaining the input or growing the unique-buffer counter.
        let work = add(
            self.metadata_per_batch,
            mul(batch.num_rows(), self.work_per_row)?,
        )?;
        self.reservation.try_grow(work)?;
        let buffers = self.buffers.count_batch(batch);
        self.reservation.try_grow(buffers)
    }
}

#[cfg(test)]
#[path = "window_buffer_memory/tests.rs"]
mod tests;
