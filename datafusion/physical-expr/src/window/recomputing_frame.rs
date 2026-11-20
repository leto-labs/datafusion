// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::any::Any;
use std::ops::Range;

use arrow::array::{ArrayRef, BooleanArray};
use datafusion_common::{Result, ScalarValue};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_expr::Accumulator;

/// Opt-in for a non-retracting aggregate that can evaluate a frame directly
/// from the window's existing input arrays.
///
/// The accumulator must own only inline state. Recomputing and evaluating
/// the frame must use fixed-size scratch without heap allocation. They
/// must reproduce the output and state of fresh accumulation over the
/// selected rows in input order, including when no rows contribute.
pub trait RecomputeFrame: Accumulator {
    /// Replace the accumulator state with the state for this frame.
    /// Only rows with a valid true filter value contribute, if supplied.
    fn recompute(
        &mut self,
        values: &[ArrayRef],
        range: Range<usize>,
        filter: Option<&BooleanArray>,
    ) -> Result<()>;
}

/// Type-erased sliding accumulator for [`RecomputeFrame`] implementations.
#[derive(Debug)]
pub struct RecomputingFrameAccumulator {
    inner: Box<dyn RecomputeFrame>,
    reservation: Option<MemoryReservation>,
}

impl RecomputingFrameAccumulator {
    /// Wrap a fixed-size frame accumulator. Window operators reserve its
    /// storage before evaluating a frame or retaining it across batches.
    pub fn new(inner: impl RecomputeFrame + 'static) -> Self {
        Self {
            inner: Box::new(inner),
            reservation: None,
        }
    }

    pub(crate) fn as_recomputing(acc: &mut Box<dyn Accumulator>) -> Option<&mut Self> {
        (acc.as_mut() as &mut dyn Any).downcast_mut()
    }

    pub(crate) fn reserve(&mut self, reservation: &MemoryReservation) -> Result<()> {
        let child = reservation.new_empty();
        child.try_grow(self.size())?;
        self.reservation = Some(child);
        Ok(())
    }

    pub(crate) fn recompute(
        &mut self,
        values: &[ArrayRef],
        range: Range<usize>,
        filter: Option<&BooleanArray>,
    ) -> Result<()> {
        self.inner.recompute(values, range, filter)
    }
}

impl Accumulator for RecomputingFrameAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.inner.update_batch(values)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.inner.merge_batch(states)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        self.inner.evaluate()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.inner.state()
    }

    fn size(&self) -> usize {
        size_of_val(self) + size_of_val(self.inner.as_ref())
    }
}
