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

//! Exact integer AVG. The partial sum is a signed i128 in 16 little-endian
//! bytes, avoiding decimal precision limits and preserving all 128 bits through
//! IPC/spill. No floating-point arithmetic occurs before final evaluation.

use super::*;
use arrow::array::{FixedSizeBinaryArray, Float64Array};
use arrow::buffer::Buffer;
use datafusion_functions_aggregate_common::utils::GenericDistinctBuffer;

fn average(sum: i128, count: u64) -> ScalarValue {
    ScalarValue::Float64((count != 0).then(|| sum as f64 / count as f64))
}

fn add_total(sum: &mut i128, count: &mut u64, value: i128, rows: u64) -> Result<()> {
    let next_sum = sum
        .checked_add(value)
        .ok_or_else(|| exec_datafusion_err!("Arithmetic overflow in integer AVG sum"))?;
    let next_count = count.checked_add(rows).ok_or_else(|| {
        exec_datafusion_err!("Arithmetic overflow in integer AVG count")
    })?;
    *sum = next_sum;
    *count = next_count;
    Ok(())
}

fn batch_total<I: ArrowNumericType>(values: &PrimitiveArray<I>) -> (i128, u64)
where
    I::Native: Into<i128>,
{
    // Every value is <= 64 bits and an addressable array has <= isize::MAX
    // elements. Thus one batch's sum fits i128, even for all UInt64::MAX.
    // Wrapping addition avoids an overflow branch in the vectorizable loop;
    // checked addition is required when combining arbitrarily many batches.
    let sum = if values.null_count() == 0 {
        values
            .values()
            .iter()
            .fold(0_i128, |sum, value| sum.wrapping_add((*value).into()))
    } else {
        values
            .iter()
            .flatten()
            .fold(0_i128, |sum, value| sum.wrapping_add(value.into()))
    };
    (sum, (values.len() - values.null_count()) as u64)
}

fn state_arrays(mut sums: Vec<i128>, counts: Vec<u64>) -> Result<Vec<ArrayRef>> {
    // No copy on little-endian hosts. Make the binary state portable on big
    // endian hosts too; Arrow IPC preserves binary bytes without swapping them.
    for sum in &mut sums {
        *sum = sum.to_le();
    }
    Ok(vec![
        Arc::new(UInt64Array::from(counts)),
        Arc::new(FixedSizeBinaryArray::try_new(
            16,
            Buffer::from_vec(sums),
            None,
        )?),
    ])
}

fn state_rows(states: &[ArrayRef]) -> impl Iterator<Item = Result<(i128, u64)>> + '_ {
    let counts = states[0].as_primitive::<UInt64Type>();
    let sums = states[1].as_fixed_size_binary();
    assert_eq!(sums.value_length(), 16);
    assert_eq!(counts.len(), sums.len());
    counts
        .iter()
        .zip(sums.iter())
        .map(|(count, sum)| match count {
            None | Some(0) => Ok((0, 0)),
            Some(count) => {
                let bytes = sum.ok_or_else(|| {
                    exec_datafusion_err!("Missing sum in nonempty integer AVG state")
                })?;
                Ok((
                    i128::from_le_bytes(
                        bytes.try_into().expect("16-byte integer AVG state"),
                    ),
                    count,
                ))
            }
        })
}

#[derive(Debug)]
pub(super) struct IntegerAvgAccumulator<I: ArrowNumericType> {
    sum: i128,
    count: u64,
    marker: PhantomData<I>,
}

impl<I: ArrowNumericType> IntegerAvgAccumulator<I> {
    pub(super) fn new() -> Self {
        Self {
            sum: 0,
            count: 0,
            marker: PhantomData,
        }
    }
}

impl<I: ArrowNumericType + Debug + Send + Sync> Accumulator for IntegerAvgAccumulator<I>
where
    I::Native: Into<i128>,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let (sum, count) = batch_total(values[0].as_primitive::<I>());
        add_total(&mut self.sum, &mut self.count, sum, count)
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(average(self.sum, self.count))
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            ScalarValue::UInt64(Some(self.count)),
            ScalarValue::FixedSizeBinary(16, Some(self.sum.to_le_bytes().to_vec())),
        ])
    }
    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        for row in state_rows(states) {
            let (sum, count) = row?;
            add_total(&mut self.sum, &mut self.count, sum, count)?;
        }
        Ok(())
    }
    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let (sum, count) = batch_total(values[0].as_primitive::<I>());
        let next_count = self
            .count
            .checked_sub(count)
            .ok_or_else(|| exec_datafusion_err!("Integer AVG retract count underflow"))?;
        let next_sum = self.sum.checked_sub(sum).ok_or_else(|| {
            exec_datafusion_err!("Arithmetic overflow in integer AVG retract")
        })?;
        self.sum = next_sum;
        self.count = next_count;
        Ok(())
    }
    fn supports_retract_batch(&self) -> bool {
        true
    }
    fn size(&self) -> usize {
        size_of_val(self)
    }
}

#[derive(Debug)]
pub(super) struct IntegerAvgGroupsAccumulator<I: ArrowNumericType> {
    sums: Vec<i128>,
    counts: Vec<u64>,
    marker: PhantomData<I>,
}

impl<I: ArrowNumericType> IntegerAvgGroupsAccumulator<I> {
    pub(super) fn new() -> Self {
        Self {
            sums: vec![],
            counts: vec![],
            marker: PhantomData,
        }
    }
    fn resize(&mut self, groups: usize) {
        self.sums.resize(groups, 0);
        self.counts.resize(groups, 0);
    }
}

impl<I: ArrowNumericType + Send> GroupsAccumulator for IntegerAvgGroupsAccumulator<I>
where
    I::Native: Into<i128>,
{
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        groups: &[usize],
        filter: Option<&BooleanArray>,
        total_groups: usize,
    ) -> Result<()> {
        self.resize(total_groups);
        let values = values[0].as_primitive::<I>();
        assert_eq!(values.len(), groups.len());
        let mask = filtered_null_mask(filter, values);
        for (index, (&value, &group)) in values.values().iter().zip(groups).enumerate() {
            if mask.as_ref().is_none_or(|mask| mask.is_valid(index)) {
                add_total(
                    &mut self.sums[group],
                    &mut self.counts[group],
                    value.into(),
                    1,
                )?;
            }
        }
        Ok(())
    }
    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let sums = emit_to.take_needed(&mut self.sums);
        let counts = emit_to.take_needed(&mut self.counts);
        Ok(Arc::new(Float64Array::from_iter(
            sums.into_iter()
                .zip(counts)
                .map(|(sum, count)| (count != 0).then(|| sum as f64 / count as f64)),
        )))
    }
    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        state_arrays(
            emit_to.take_needed(&mut self.sums),
            emit_to.take_needed(&mut self.counts),
        )
    }
    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        groups: &[usize],
        total_groups: usize,
    ) -> Result<()> {
        self.resize(total_groups);
        assert_eq!(values[0].len(), groups.len());
        for (row, &group) in state_rows(values).zip(groups) {
            let (sum, count) = row?;
            add_total(&mut self.sums[group], &mut self.counts[group], sum, count)?;
        }
        Ok(())
    }
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let values = values[0].as_primitive::<I>();
        let mask = filtered_null_mask(filter, values);
        let mut sums = Vec::with_capacity(values.len());
        let mut counts = Vec::with_capacity(values.len());
        for (index, &value) in values.values().iter().enumerate() {
            let valid = mask.as_ref().is_none_or(|mask| mask.is_valid(index));
            sums.push(if valid { value.into() } else { 0 });
            counts.push(u64::from(valid));
        }
        state_arrays(sums, counts)
    }
    fn size(&self) -> usize {
        size_of_val(self)
            + self.sums.capacity() * size_of::<i128>()
            + self.counts.capacity() * size_of::<u64>()
    }
}

#[derive(Debug)]
pub(super) struct IntegerDistinctAvgAccumulator<I: ArrowNumericType> {
    values: GenericDistinctBuffer<I>,
}
impl<I: ArrowNumericType> IntegerDistinctAvgAccumulator<I> {
    pub(super) fn new() -> Self {
        Self {
            values: GenericDistinctBuffer::new(I::DATA_TYPE),
        }
    }
}
impl<I: ArrowNumericType + Debug> Accumulator for IntegerDistinctAvgAccumulator<I>
where
    I::Native: Into<i128>,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.values.update_batch(values)
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.values.state()
    }
    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.values.merge_batch(states)
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut sum = 0_i128;
        for value in &self.values.values {
            sum = sum.checked_add(value.0.into()).ok_or_else(|| {
                exec_datafusion_err!("Arithmetic overflow in integer AVG DISTINCT sum")
            })?;
        }
        Ok(average(sum, self.values.values.len() as u64))
    }
    fn size(&self) -> usize {
        size_of_val(self) + self.values.size()
    }
}
