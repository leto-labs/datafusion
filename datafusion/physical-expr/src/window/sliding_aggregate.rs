// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Physical exec for aggregate window function expressions.

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use crate::aggregate::{AggregateFunctionExpr, SlidingFrameAccumulator};
use crate::window::aggregate::reverse_aggregate_window_expr;
use crate::window::window_expr::{
    AggregateWindowExpr, WindowEvalContext, WindowFn, filter_array,
};
use crate::window::{
    PartitionBatches, PartitionWindowAggStates, RecomputingFrameAccumulator, WindowExpr,
};
use crate::{PhysicalExpr, expressions::PhysicalSortExpr};

use arrow::array::{ArrayRef, BooleanArray};
use arrow::datatypes::FieldRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{Result, ScalarValue};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_expr::{Accumulator, WindowFrame};

/// A window expr that takes the form of an aggregate function that
/// can be incrementally computed over sliding windows.
///
/// See comments on [`WindowExpr`] for more details.
#[derive(Debug)]
pub struct SlidingAggregateWindowExpr {
    aggregate: Arc<AggregateFunctionExpr>,
    partition_by: Vec<Arc<dyn PhysicalExpr>>,
    order_by: Vec<PhysicalSortExpr>,
    window_frame: Arc<WindowFrame>,
    filter: Option<Arc<dyn PhysicalExpr>>,
}

impl SlidingAggregateWindowExpr {
    /// Create a new (sliding) aggregate window function expression.
    pub fn new(
        aggregate: Arc<AggregateFunctionExpr>,
        partition_by: &[Arc<dyn PhysicalExpr>],
        order_by: &[PhysicalSortExpr],
        window_frame: Arc<WindowFrame>,
        filter: Option<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self {
            aggregate,
            partition_by: partition_by.to_vec(),
            order_by: order_by.to_vec(),
            window_frame,
            filter,
        }
    }

    /// Evaluate the physical window with its query-pool memory reservation.
    pub fn evaluate_with_reservation(
        &self,
        batch: &RecordBatch,
        reservation: &MemoryReservation,
    ) -> Result<ArrayRef> {
        self.aggregate_evaluate_with_reservation(batch, Some(reservation))
    }

    /// Get the [AggregateFunctionExpr] of this object.
    pub fn get_aggregate_expr(&self) -> &AggregateFunctionExpr {
        &self.aggregate
    }
}

/// Incrementally update window function using the fact that batch is
/// pre-sorted given the sort columns and then per partition point.
///
/// Evaluates the peer group (e.g. `SUM` or `MAX` gives the same results
/// for peers) and concatenate the results.
impl WindowExpr for SlidingAggregateWindowExpr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn field(&self) -> Result<FieldRef> {
        Ok(self.aggregate.field())
    }

    fn name(&self) -> &str {
        self.aggregate.name()
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.aggregate.expressions()
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        self.aggregate_evaluate(batch)
    }

    fn evaluate_stateful(
        &self,
        partition_batches: &PartitionBatches,
        window_agg_state: &mut PartitionWindowAggStates,
        eval_ctx: &WindowEvalContext<'_>,
    ) -> Result<()> {
        self.aggregate_evaluate_stateful(partition_batches, window_agg_state, eval_ctx)
    }

    fn partition_by(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.partition_by
    }

    fn order_by(&self) -> &[PhysicalSortExpr] {
        &self.order_by
    }

    fn get_window_frame(&self) -> &Arc<WindowFrame> {
        &self.window_frame
    }

    fn get_reverse_expr(&self) -> Option<Arc<dyn WindowExpr>> {
        reverse_aggregate_window_expr(
            &self.aggregate,
            &self.partition_by,
            &self.order_by,
            &self.window_frame,
            self.filter.as_ref(),
        )
    }

    fn uses_bounded_memory(&self) -> bool {
        !self.window_frame.end_bound.is_unbounded()
    }

    fn with_new_expressions(
        &self,
        args: Vec<Arc<dyn PhysicalExpr>>,
        partition_bys: Vec<Arc<dyn PhysicalExpr>>,
        order_by_exprs: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Option<Arc<dyn WindowExpr>> {
        debug_assert_eq!(self.order_by.len(), order_by_exprs.len());

        let new_order_by = self
            .order_by
            .iter()
            .zip(order_by_exprs)
            .map(|(req, new_expr)| PhysicalSortExpr {
                expr: new_expr,
                options: req.options,
            })
            .collect();
        Some(Arc::new(SlidingAggregateWindowExpr {
            aggregate: self
                .aggregate
                .with_new_expressions(args, vec![])
                .map(Arc::new)?,
            partition_by: partition_bys,
            order_by: new_order_by,
            window_frame: Arc::clone(&self.window_frame),
            filter: self.filter.clone(),
        }))
    }

    fn create_window_fn(&self) -> Result<WindowFn> {
        Ok(WindowFn::Aggregate(self.get_accumulator()?))
    }
}

impl AggregateWindowExpr for SlidingAggregateWindowExpr {
    fn get_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        self.aggregate.create_sliding_accumulator()
    }

    fn filter_expr(&self) -> Option<&Arc<dyn PhysicalExpr>> {
        self.filter.as_ref()
    }

    /// Given current range and the last range, calculates the accumulator
    /// result for the range of interest.
    fn get_aggregate_result_inside_range(
        &self,
        last_range: &Range<usize>,
        cur_range: &Range<usize>,
        value_slice: &[ArrayRef],
        accumulator: &mut Box<dyn Accumulator>,
        filter_mask: Option<&BooleanArray>,
    ) -> Result<ScalarValue> {
        if let Some(frame) = RecomputingFrameAccumulator::as_recomputing(accumulator) {
            // RANGE frames can be empty with their start past their end.
            let start = cur_range.start.min(cur_range.end);
            frame.recompute(value_slice, start..cur_range.end, filter_mask)?;
            return if cur_range.is_empty() {
                self.aggregate
                    .default_value(self.aggregate.field().data_type())
            } else {
                accumulator.evaluate()
            };
        }

        let leaving = last_range.start..last_range.end.min(cur_range.start);
        let entering = cur_range.start.max(last_range.end)..cur_range.end;
        let retract_first = last_range.end <= cur_range.start;
        if let ([values], None, false) = (value_slice, filter_mask, cur_range.is_empty())
            && let Some(frame) = SlidingFrameAccumulator::as_sliding(accumulator)
        {
            return frame.advance(values, leaving, entering, retract_first);
        }

        if retract_first {
            // Skip rows between disjoint frames. Drain before adding the new
            // frame so that the aggregate never evaluates skipped rows.
            with_frame_values(value_slice, filter_mask, leaving, |values| {
                accumulator.retract_batch(values)
            })?;
            with_frame_values(value_slice, filter_mask, entering, |values| {
                accumulator.update_batch(values)
            })?;
        } else {
            with_frame_values(value_slice, filter_mask, entering, |values| {
                accumulator.update_batch(values)
            })?;
            with_frame_values(value_slice, filter_mask, leaving, |values| {
                accumulator.retract_batch(values)
            })?;
        }

        if cur_range.is_empty() {
            self.aggregate
                .default_value(self.aggregate.field().data_type())
        } else {
            accumulator.evaluate()
        }
    }

    fn is_constant_in_partition(&self) -> bool {
        false
    }
}

fn with_frame_values(
    values: &[ArrayRef],
    filter: Option<&BooleanArray>,
    range: Range<usize>,
    apply: impl FnOnce(&[ArrayRef]) -> Result<()>,
) -> Result<()> {
    if range.is_empty() {
        return Ok(());
    }

    // Median and most other window aggregates have a single input. Keep
    // their slice on the stack.
    if let ([value], None) = (values, filter) {
        return apply(&[value.slice(range.start, range.len())]);
    }

    let mask = filter.map(|mask| mask.slice(range.start, range.len()));
    let slices = values
        .iter()
        .map(|value| {
            let slice = value.slice(range.start, range.len());
            match &mask {
                Some(mask) => filter_array(&slice, mask),
                None => Ok(slice),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    apply(&slices)
}
