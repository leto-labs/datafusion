// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{TryReserveError, VecDeque};
use std::mem::size_of;
use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, BooleanArray, PrimitiveArray};
use arrow::buffer::NullBuffer;
use arrow::compute;
use arrow::datatypes::ArrowPrimitiveType;
use arrow::datatypes::DataType;
use datafusion_common::utils::split_vec_min_alloc;
use datafusion_common::{DataFusionError, Result, internal_datafusion_err};
use datafusion_expr_common::groups_accumulator::{
    EmitTo, GroupSelection, GroupsAccumulator,
};

use super::accumulate::NullState;

/// An accumulator that implements a single operation over
/// [`ArrowPrimitiveType`] where the accumulated state is the same as
/// the input type (such as `Sum`)
///
/// F: The function to apply to two elements. The first argument is
/// the existing value and should be updated with the second value
/// (e.g. [`BitAndAssign`] style).
///
/// [`BitAndAssign`]: std::ops::BitAndAssign
#[derive(Debug)]
pub struct PrimitiveGroupsAccumulator<T, F>
where
    T: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, T::Native) + Send + Sync + 'static,
{
    /// values per group, stored as the native type
    values: GroupBuffer<T::Native>,

    /// The output type (needed for Decimal precision and scale)
    data_type: DataType,

    /// The starting value for new groups
    starting_value: T::Native,

    /// Track nulls in the input / filters
    null_state: NullState,

    /// Function that computes the primitive result
    prim_fn: F,
}

impl<T, F> PrimitiveGroupsAccumulator<T, F>
where
    T: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, T::Native) + Send + Sync + 'static,
{
    pub fn new(data_type: &DataType, prim_fn: F) -> Self {
        Self {
            values: GroupBuffer::new(),
            data_type: data_type.clone(),
            null_state: NullState::new(),
            starting_value: T::default_value(),
            prim_fn,
        }
    }

    /// Set the starting values for new groups
    pub fn with_starting_value(mut self, starting_value: T::Native) -> Self {
        self.starting_value = starting_value;
        self
    }
}

impl<T, F> GroupsAccumulator for PrimitiveGroupsAccumulator<T, F>
where
    T: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, T::Native) + Send + Sync + 'static,
{
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "single argument to update_batch");
        let values = values[0].as_primitive::<T>();

        // update values
        self.values.resize(total_num_groups, self.starting_value);

        // NullState dispatches / handles tracking nulls and groups that saw no values
        self.null_state.accumulate(
            group_indices,
            values,
            opt_filter,
            total_num_groups,
            |group_index, new_value| {
                let value = self.values.get_mut(group_index);
                (self.prim_fn)(value, new_value);
            },
        );

        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let count = match emit_to {
            EmitTo::All => self.values.len(),
            EmitTo::First(n) => n,
        };
        let values = self.values.take_prefix(count);
        let nulls = self.null_state.build(emit_to);
        let values = PrimitiveArray::<T>::new(values.into(), nulls)
            .with_data_type(self.data_type.clone());
        Ok(Arc::new(values))
    }

    fn evaluate_preserving(&mut self, selection: GroupSelection<'_>) -> Result<ArrayRef> {
        selection.validate_num_groups(self.values.len())?;
        let values: Vec<T::Native> = selection
            .iter()
            .map(|index| *self.values.get(index))
            .collect();
        let nulls = self.null_state.build_preserving(selection)?;
        let values = PrimitiveArray::<T>::new(values.into(), nulls)
            .with_data_type(self.data_type.clone());
        Ok(Arc::new(values))
    }

    fn supports_evaluate_preserving(&self) -> bool {
        true
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        self.evaluate(emit_to).map(|arr| vec![arr])
    }

    fn state_preserving(
        &mut self,
        selection: GroupSelection<'_>,
    ) -> Result<Vec<ArrayRef>> {
        self.evaluate_preserving(selection).map(|arr| vec![arr])
    }

    fn supports_state_preserving(&self) -> bool {
        true
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        // update / merge are the same
        self.update_batch(values, group_indices, None, total_num_groups)
    }

    /// Converts an input batch directly to a state batch
    ///
    /// The state is:
    /// - self.prim_fn for all non null, non filtered values
    /// - null otherwise
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let values = values[0].as_primitive::<T>().clone();

        // Initializing state with starting values
        let initial_state =
            PrimitiveArray::<T>::from_value(self.starting_value, values.len());

        // Recalculating values in case there is filter
        let values = match opt_filter {
            None => values,
            Some(filter) => {
                let (filter_values, filter_nulls) = filter.clone().into_parts();
                // Calculating filter mask as a result of bitand of filter, and converting it to null buffer
                let filter_bool = match filter_nulls {
                    Some(filter_nulls) => filter_nulls.inner() & &filter_values,
                    None => filter_values,
                };
                let filter_nulls = NullBuffer::from(filter_bool);

                // Rebuilding input values with a new nulls mask, which is equal to
                // the union of original nulls and filter mask
                let (dt, values_buf, original_nulls) = values.into_parts();
                let nulls_buf =
                    NullBuffer::union(original_nulls.as_ref(), Some(&filter_nulls));
                PrimitiveArray::<T>::new(values_buf, nulls_buf).with_data_type(dt)
            }
        };

        let state_values = compute::binary_mut(initial_state, &values, |mut x, y| {
            (self.prim_fn)(&mut x, y);
            x
        });
        let state_values = state_values
            .map_err(|_| {
                internal_datafusion_err!(
                    "initial_values underlying buffer must not be shared"
                )
            })?
            .map_err(DataFusionError::from)?
            .with_data_type(self.data_type.clone());

        Ok(vec![Arc::new(state_values)])
    }
    fn size(&self) -> usize {
        self.values.allocated_size() + self.null_state.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::Int64Type;

    #[test]
    fn nullable_state_emit_and_merge_across_blocks() -> Result<()> {
        let build = || {
            PrimitiveGroupsAccumulator::<Int64Type, _>::new(
                &DataType::Int64,
                |current, value| *current += value,
            )
        };
        let count = GROUP_BLOCK_LEN * 2 + 17;
        let prefix = GROUP_BLOCK_LEN + 3;
        let input =
            Int64Array::from_iter((0..count).map(|i| (i % 3 != 0).then_some(i as i64)));
        let indices: Vec<_> = (0..count).collect();
        let mut partial = build();
        partial.update_batch(&[Arc::new(input.clone())], &indices, None, count)?;
        let selected = [prefix, 0, GROUP_BLOCK_LEN - 1, prefix];
        let preserved = partial
            .state_preserving(GroupSelection::try_from_indices(&selected, count)?)?;
        let expected_selection =
            Int64Array::from_iter(selected.map(|i| (i % 3 != 0).then_some(i as i64)));
        assert_eq!(
            preserved[0].as_primitive::<Int64Type>(),
            &expected_selection
        );
        let first = partial.state(EmitTo::First(prefix))?;
        let suffix = partial.state(EmitTo::All)?;

        let mut merged = build();
        merged.merge_batch(&first, &indices[..prefix], prefix)?;
        merged.merge_batch(&suffix, &indices[prefix..], count)?;
        merged.merge_batch(&first, &indices[..prefix], count)?;
        let output = merged.evaluate(EmitTo::All)?;
        let expected = Int64Array::from_iter(
            input
                .iter()
                .enumerate()
                .map(|(i, v)| v.map(|value| if i < prefix { value * 2 } else { value })),
        );
        assert_eq!(output.as_primitive::<Int64Type>(), &expected);
        Ok(())
    }

    #[test]
    fn preserving_reads_do_not_change_accumulator_state() -> Result<()> {
        let mut accumulator = PrimitiveGroupsAccumulator::<Int64Type, _>::new(
            &DataType::Int64,
            |current, value| *current += value,
        );
        let values = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
        accumulator.update_batch(&[values], &[0, 1, 2], None, 4)?;

        let selection = GroupSelection::try_from_indices(&[3, 0, 1, 1], 4)?;
        let expected = Int64Array::from(vec![None, Some(1), None, None]);
        for _ in 0..2 {
            let actual = accumulator.evaluate_preserving(selection)?;
            assert_eq!(actual.as_primitive::<Int64Type>(), &expected);
            let state = accumulator.state_preserving(selection)?;
            assert_eq!(state[0].as_primitive::<Int64Type>(), &expected);
        }

        // Group indices and unselected state remain valid after repeated reads.
        let values = Arc::new(Int64Array::from(vec![5, 7]));
        accumulator.update_batch(&[values], &[1, 3], None, 4)?;
        let expected = Int64Array::from(vec![Some(1), Some(5), Some(3), Some(7)]);
        let actual = accumulator.evaluate_preserving(GroupSelection::all(4))?;
        assert_eq!(actual.as_primitive::<Int64Type>(), &expected);

        // A destructive read still sees all state after preserving reads.
        let actual = accumulator.evaluate(EmitTo::All)?;
        assert_eq!(actual.as_primitive::<Int64Type>(), &expected);
        assert!(accumulator.supports_evaluate_preserving());
        assert!(accumulator.supports_state_preserving());
        Ok(())
    }
}

/// Fixed-width group state that grows without moving its existing large buffers.
/// The first block grows as a Vec for small group sets. Subsequent blocks have
/// a fixed capacity so a group index maps directly to a block and an offset.
///
/// This is also used by the physical plan's byte-view group key builder.
#[doc(hidden)]
#[derive(Debug)]
pub struct GroupBuffer<T: Copy> {
    first: Vec<T>,
    tail: VecDeque<Vec<T>>,
    head: usize,
    len: usize,
}

// 256 KiB for 128-bit values and 128 KiB for 64-bit values.
const GROUP_BLOCK_LEN: usize = 16 * 1024;

impl<T: Copy> GroupBuffer<T> {
    pub fn new() -> Self {
        Self {
            first: Vec::new(),
            tail: VecDeque::new(),
            head: 0,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn allocated_size(&self) -> usize {
        self.tail.capacity() * size_of::<Vec<T>>()
            + (self.first.capacity() + self.tail.len() * GROUP_BLOCK_LEN) * size_of::<T>()
    }

    #[inline]
    pub fn get(&self, index: usize) -> &T {
        assert!(index < self.len);
        let index = index + self.head;
        if index < GROUP_BLOCK_LEN {
            &self.first[index]
        } else {
            &self.tail[index / GROUP_BLOCK_LEN - 1][index % GROUP_BLOCK_LEN]
        }
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        assert!(index < self.len);
        let index = index + self.head;
        if index < GROUP_BLOCK_LEN {
            &mut self.first[index]
        } else {
            &mut self.tail[index / GROUP_BLOCK_LEN - 1][index % GROUP_BLOCK_LEN]
        }
    }

    fn block(&mut self, block: usize) -> &mut Vec<T> {
        if block == 0 {
            &mut self.first
        } else {
            if block > self.tail.len() {
                self.tail.push_back(Vec::with_capacity(GROUP_BLOCK_LEN));
            }
            &mut self.tail[block - 1]
        }
    }

    pub fn push(&mut self, value: T) {
        let index = self.len + self.head;
        let block = self.block(index / GROUP_BLOCK_LEN);
        if index < GROUP_BLOCK_LEN && block.len() == block.capacity() {
            let target = (block.capacity() * 2).clamp(4, GROUP_BLOCK_LEN);
            block.reserve_exact(target - block.len());
        }
        block.push(value);
        self.len += 1;
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let end = self
            .head
            .saturating_add(self.len)
            .saturating_add(additional);
        let first = end.min(GROUP_BLOCK_LEN);
        if first > self.first.capacity() {
            let target = first
                .max(self.first.capacity() * 2)
                .clamp(4, GROUP_BLOCK_LEN);
            self.first.try_reserve_exact(target - self.first.len())?;
        }
        let blocks = end.div_ceil(GROUP_BLOCK_LEN).saturating_sub(1);
        self.tail
            .try_reserve(blocks.saturating_sub(self.tail.len()))?;
        while self.tail.len() < blocks {
            let mut block = Vec::new();
            block.try_reserve_exact(GROUP_BLOCK_LEN)?;
            self.tail.push_back(block);
        }
        Ok(())
    }

    pub fn extend(&mut self, mut values: impl ExactSizeIterator<Item = T>) {
        self.try_reserve(values.len())
            .expect("group buffer allocation");
        while values.len() != 0 {
            let index = self.head + self.len;
            let count = values.len().min(GROUP_BLOCK_LEN - index % GROUP_BLOCK_LEN);
            self.block(index / GROUP_BLOCK_LEN)
                .extend(values.by_ref().take(count));
            self.len += count;
        }
    }

    pub fn resize(&mut self, len: usize, value: T) {
        if len < self.len {
            let end = self.head + len;
            let last = end.saturating_sub(1) / GROUP_BLOCK_LEN;
            self.tail.truncate(last);
            self.block(last).truncate(end - last * GROUP_BLOCK_LEN);
            self.len = len;
            return;
        }
        while self.len < len {
            let start = self.head + self.len;
            let block_index = start / GROUP_BLOCK_LEN;
            let block_len =
                (self.head + len - block_index * GROUP_BLOCK_LEN).min(GROUP_BLOCK_LEN);
            let block = self.block(block_index);
            if block_index == 0 && block_len > block.capacity() {
                let target = block_len
                    .max(block.capacity() * 2)
                    .clamp(4, GROUP_BLOCK_LEN);
                block.reserve_exact(target - block.len());
            }
            block.resize(block_len, value);
            self.len = block_index * GROUP_BLOCK_LEN + block_len - self.head;
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.first[self.head..]
            .iter_mut()
            .chain(self.tail.iter_mut().flatten())
    }

    pub fn take_prefix(&mut self, n: usize) -> Vec<T> {
        assert!(n <= self.len);
        if self.tail.is_empty() && self.head == 0 {
            self.len -= n;
            return if self.len == 0 {
                self.tail = VecDeque::new();
                std::mem::take(&mut self.first)
            } else {
                split_vec_min_alloc(&mut self.first, n)
            };
        }

        if self.head == 0 && n == self.first.len() {
            self.len -= n;
            return std::mem::replace(
                &mut self.first,
                self.tail.pop_front().unwrap_or_default(),
            );
        }

        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let take = (n - out.len()).min(self.first.len() - self.head);
            out.extend_from_slice(&self.first[self.head..self.head + take]);
            self.head += take;
            if self.head == self.first.len() {
                self.first = self.tail.pop_front().unwrap_or_default();
                self.head = 0;
            }
        }
        self.len -= n;
        if self.len == 0 {
            *self = Self::new();
        }
        out
    }
}

impl<T: Copy> Default for GroupBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod group_buffer_tests {
    use super::*;

    #[test]
    fn full_emit_releases_all_prior_segment_capacity() {
        let mut values = GroupBuffer::new();
        values.resize(GROUP_BLOCK_LEN + 1, 17_u64);
        assert_eq!(values.take_prefix(GROUP_BLOCK_LEN).len(), GROUP_BLOCK_LEN);
        assert_eq!(values.take_prefix(1), vec![17]);
        assert_eq!(values.allocated_size(), 0);
    }

    #[test]
    fn emit_and_regrow_across_boundaries() {
        let mut values = GroupBuffer::new();
        let mut expected = Vec::new();
        for i in 0..GROUP_BLOCK_LEN * 3 + 5 {
            values.push(i);
            expected.push(i);
        }
        for n in [1, GROUP_BLOCK_LEN, GROUP_BLOCK_LEN + 7, 0] {
            assert_eq!(
                values.take_prefix(n),
                expected.drain(..n).collect::<Vec<_>>()
            );
            assert_eq!(values.len(), expected.len());
            for (index, value) in expected.iter().enumerate() {
                assert_eq!(values.get(index), value);
            }
            values.resize(expected.len() + 11, 42);
            expected.extend([42; 11]);
            *values.get_mut(0) = 123;
            expected[0] = 123;
            assert!(
                values
                    .iter_mut()
                    .map(|value| *value)
                    .eq(expected.iter().copied())
            );
        }
        assert_eq!(values.take_prefix(values.len()), expected);
        assert_eq!(values.allocated_size(), 0);
    }
}
