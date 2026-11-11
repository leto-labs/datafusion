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

//! Exact AVG for <=32-bit integers. Common running sums/counts are i64.
//! Overflow promotes sums to i128. Partial state stays primitive: count, low
//! i64 sum, and a nullable i32 high word. NULL high means sign-extend the low
//! word. With count <= i64::MAX and <=32-bit inputs, every sum fits 96 bits.

use super::*;
use arrow::array::{Float64Array, Int32Array, Int64Array, new_null_array};
use datafusion_functions_aggregate_common::utils::GenericDistinctBuffer;

#[derive(Debug, Clone, Copy)]
enum Sum {
    Narrow(i64),
    Wide(i128),
}

impl Sum {
    fn add(self, other: Self, subtract: bool) -> Result<Self> {
        if let (Self::Narrow(a), Self::Narrow(b)) = (self, other)
            && let Some(sum) = if subtract {
                a.checked_sub(b)
            } else {
                a.checked_add(b)
            }
        {
            return Ok(Self::Narrow(sum));
        }
        let sum = if subtract {
            self.wide().checked_sub(other.wide())
        } else {
            self.wide().checked_add(other.wide())
        };
        sum.map(Self::Wide)
            .ok_or_else(|| exec_datafusion_err!("Arithmetic overflow in integer AVG sum"))
    }
    fn wide(self) -> i128 {
        match self {
            Self::Narrow(sum) => i128::from(sum),
            Self::Wide(sum) => sum,
        }
    }
    fn float(self) -> f64 {
        match self {
            Self::Narrow(sum) => sum as f64,
            Self::Wide(sum) => sum as f64,
        }
    }
    fn from_state(low: i64, high: Option<i32>) -> Self {
        match high {
            None => Self::Narrow(low),
            Some(high) => Self::Wide((i128::from(high) << 64) | i128::from(low as u64)),
        }
    }
}

fn batch_sum<I: ArrowNumericType>(values: &PrimitiveArray<I>) -> Sum
where
    I::Native: Into<i64> + Into<i128>,
{
    let sum = if values.null_count() == 0 {
        values
            .values()
            .iter()
            .try_fold(0_i64, |sum, value| sum.checked_add((*value).into()))
    } else {
        values
            .iter()
            .flatten()
            .try_fold(0_i64, |sum, value| sum.checked_add(value.into()))
    };
    sum.map(Sum::Narrow)
        .unwrap_or_else(|| Sum::Wide(integer::batch_total(values).0))
}

fn add_count(count: i64, rows: i64) -> Result<i64> {
    if rows < 0 {
        return exec_err!("Negative count in integer AVG state");
    }
    count
        .checked_add(rows)
        .ok_or_else(|| exec_datafusion_err!("Arithmetic overflow in integer AVG count"))
}

fn high_word(sum: i128) -> Result<Option<i32>> {
    if i64::try_from(sum).is_ok() {
        return Ok(None);
    }
    i32::try_from(sum >> 64)
        .map(Some)
        .map_err(|_| exec_datafusion_err!("Integer AVG state exceeds 96 bits"))
}

/// Normal state has no null counts/sums and an all-NULL high word. Iterate
/// native i64 slices directly; decode extension words only for promoted state.
fn visit_state(
    states: &[ArrayRef],
    mut visit: impl FnMut(usize, Sum, i64) -> Result<()>,
) -> Result<()> {
    let counts = states[0].as_primitive::<Int64Type>();
    let sums = states[1].as_primitive::<Int64Type>();
    let highs = states[2].as_primitive::<Int32Type>();
    assert_eq!(counts.len(), sums.len());
    assert_eq!(counts.len(), highs.len());
    if counts.null_count() == 0
        && sums.null_count() == 0
        && highs.null_count() == highs.len()
    {
        for (index, (&count, &sum)) in
            counts.values().iter().zip(sums.values()).enumerate()
        {
            if count != 0 {
                visit(index, Sum::Narrow(sum), count)?;
            }
        }
    } else {
        for index in 0..counts.len() {
            if counts.is_null(index) || counts.value(index) == 0 {
                continue;
            }
            if sums.is_null(index) {
                return exec_err!("Missing sum in nonempty integer AVG state");
            }
            let high = highs.is_valid(index).then(|| highs.value(index));
            visit(
                index,
                Sum::from_state(sums.value(index), high),
                counts.value(index),
            )?;
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(super) struct NarrowAvgAccumulator<I: ArrowNumericType> {
    sum: Sum,
    count: i64,
    marker: PhantomData<I>,
}
impl<I: ArrowNumericType> NarrowAvgAccumulator<I> {
    pub(super) fn new() -> Self {
        Self {
            sum: Sum::Narrow(0),
            count: 0,
            marker: PhantomData,
        }
    }
    fn add(&mut self, sum: Sum, rows: i64) -> Result<()> {
        let count = add_count(self.count, rows)?;
        let sum = self.sum.add(sum, false)?;
        self.sum = sum;
        self.count = count;
        Ok(())
    }
}
impl<I: ArrowNumericType + Debug + Send + Sync> Accumulator for NarrowAvgAccumulator<I>
where
    I::Native: Into<i64> + Into<i128>,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = values[0].as_primitive::<I>();
        self.add(
            batch_sum(values),
            (values.len() - values.null_count()) as i64,
        )
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::Float64(
            (self.count != 0).then(|| self.sum.float() / self.count as f64),
        ))
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            ScalarValue::Int64(Some(self.count)),
            ScalarValue::Int64(Some(self.sum.wide() as i64)),
            ScalarValue::Int32(high_word(self.sum.wide())?),
        ])
    }
    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        visit_state(states, |_, sum, count| self.add(sum, count))
    }
    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = values[0].as_primitive::<I>();
        let count = self
            .count
            .checked_sub((values.len() - values.null_count()) as i64)
            .filter(|count| *count >= 0)
            .ok_or_else(|| exec_datafusion_err!("Integer AVG retract count underflow"))?;
        let sum = self.sum.add(batch_sum(values), true)?;
        self.sum = sum;
        self.count = count;
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
enum Sums {
    Narrow(Vec<i64>),
    Wide(Vec<i128>),
}
impl Sums {
    fn resize(&mut self, len: usize) {
        match self {
            Self::Narrow(v) => v.resize(len, 0),
            Self::Wide(v) => v.resize(len, 0),
        }
    }
    fn add(&mut self, group: usize, sum: Sum) -> Result<()> {
        if let (Self::Narrow(sums), Sum::Narrow(value)) = (&mut *self, sum)
            && let Some(sum) = sums[group].checked_add(value)
        {
            sums[group] = sum;
            return Ok(());
        }
        // Promote once, only on overflow or when receiving promoted state.
        if let Self::Narrow(sums) = self {
            *self = Self::Wide(sums.iter().copied().map(i128::from).collect());
        }
        let Self::Wide(sums) = self else {
            unreachable!()
        };
        sums[group] = sums[group].checked_add(sum.wide()).ok_or_else(|| {
            exec_datafusion_err!("Arithmetic overflow in integer AVG sum")
        })?;
        Ok(())
    }
    fn take(&mut self, emit: EmitTo) -> Self {
        match self {
            Self::Narrow(sums) => Self::Narrow(emit.take_needed(sums)),
            Self::Wide(sums) => Self::Wide(emit.take_needed(sums)),
        }
    }
    fn size(&self) -> usize {
        match self {
            Self::Narrow(v) => v.capacity() * size_of::<i64>(),
            Self::Wide(v) => v.capacity() * size_of::<i128>(),
        }
    }
    fn state(self, counts: Vec<i64>) -> Result<Vec<ArrayRef>> {
        let (sums, highs): (ArrayRef, ArrayRef) = match self {
            Self::Narrow(sums) => {
                let highs = new_null_array(&DataType::Int32, sums.len());
                (Arc::new(Int64Array::from(sums)), highs)
            }
            Self::Wide(sums) => {
                let lows = sums.iter().map(|&sum| sum as i64).collect::<Vec<_>>();
                let highs = sums
                    .into_iter()
                    .map(high_word)
                    .collect::<Result<Vec<_>>>()?;
                (
                    Arc::new(Int64Array::from(lows)),
                    Arc::new(Int32Array::from(highs)),
                )
            }
        };
        Ok(vec![Arc::new(Int64Array::from(counts)), sums, highs])
    }
}

#[derive(Debug)]
pub(super) struct NarrowAvgGroupsAccumulator<I: ArrowNumericType> {
    sums: Sums,
    counts: Vec<i64>,
    marker: PhantomData<I>,
}
impl<I: ArrowNumericType> NarrowAvgGroupsAccumulator<I> {
    pub(super) fn new() -> Self {
        Self {
            sums: Sums::Narrow(vec![]),
            counts: vec![],
            marker: PhantomData,
        }
    }
    fn resize(&mut self, groups: usize) {
        self.sums.resize(groups);
        self.counts.resize(groups, 0);
    }
    fn add(&mut self, group: usize, sum: Sum, rows: i64) -> Result<()> {
        let count = add_count(self.counts[group], rows)?;
        self.sums.add(group, sum)?;
        self.counts[group] = count;
        Ok(())
    }
}
impl<I: ArrowNumericType + Send> GroupsAccumulator for NarrowAvgGroupsAccumulator<I>
where
    I::Native: Into<i64> + Into<i128>,
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
                self.add(group, Sum::Narrow(value.into()), 1)?;
            }
        }
        Ok(())
    }
    fn evaluate(&mut self, emit: EmitTo) -> Result<ArrayRef> {
        let sums = self.sums.take(emit);
        let counts = emit.take_needed(&mut self.counts);
        let result = match sums {
            Sums::Narrow(sums) => Float64Array::from_iter(
                sums.into_iter()
                    .zip(counts)
                    .map(|(sum, count)| (count != 0).then(|| sum as f64 / count as f64)),
            ),
            Sums::Wide(sums) => Float64Array::from_iter(
                sums.into_iter()
                    .zip(counts)
                    .map(|(sum, count)| (count != 0).then(|| sum as f64 / count as f64)),
            ),
        };
        Ok(Arc::new(result))
    }
    fn state(&mut self, emit: EmitTo) -> Result<Vec<ArrayRef>> {
        self.sums
            .take(emit)
            .state(emit.take_needed(&mut self.counts))
    }
    fn merge_batch(
        &mut self,
        states: &[ArrayRef],
        groups: &[usize],
        total_groups: usize,
    ) -> Result<()> {
        self.resize(total_groups);
        assert_eq!(states[0].len(), groups.len());
        visit_state(states, |index, sum, count| {
            self.add(groups[index], sum, count)
        })
    }
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let values = values[0].as_primitive::<I>();
        let mask = filtered_null_mask(filter, values);
        // No nullable iteration, i128 conversion, or per-element branch on the
        // common unfiltered path. Invalid sums are ignored by their zero count.
        let sums = values
            .values()
            .iter()
            .map(|&value| value.into())
            .collect::<Vec<i64>>();
        let counts = match mask {
            None => vec![1_i64; values.len()],
            Some(mask) => mask.iter().map(i64::from).collect(),
        };
        Sums::Narrow(sums).state(counts)
    }
    fn size(&self) -> usize {
        size_of_val(self) + self.sums.size() + self.counts.capacity() * size_of::<i64>()
    }
}

#[derive(Debug)]
pub(super) struct NarrowDistinctAvgAccumulator<I: ArrowNumericType> {
    values: GenericDistinctBuffer<I>,
}
impl<I: ArrowNumericType> NarrowDistinctAvgAccumulator<I> {
    pub(super) fn new() -> Self {
        Self {
            values: GenericDistinctBuffer::new(I::DATA_TYPE),
        }
    }
}
impl<I: ArrowNumericType + Debug> Accumulator for NarrowDistinctAvgAccumulator<I>
where
    I::Native: Into<i64>,
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
        // Any prefix is a subset of <=32-bit distinct keys. The greatest
        // possible unsigned sum is (2^32-1)*2^32/2 < i64::MAX. Signed positive
        // and negative subset bounds are smaller, so checked i64 cannot overflow.
        let sum = self
            .values
            .values
            .iter()
            .try_fold(0_i64, |sum, value| sum.checked_add(value.0.into()))
            .ok_or_else(|| {
                exec_datafusion_err!("Overflow in narrow integer AVG DISTINCT")
            })?;
        let count = self.values.values.len() as i64;
        Ok(ScalarValue::Float64(
            (count != 0).then(|| sum as f64 / count as f64),
        ))
    }
    fn size(&self) -> usize {
        size_of_val(self) + self.values.size()
    }
}
