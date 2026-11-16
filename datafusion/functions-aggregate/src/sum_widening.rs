// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

// Added by the Longbow authors; see third_party/README.md.

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

//! `sum` over the narrow storage of `Decimal128` values.
//!
//! The accumulators here read `Decimal32` or `Decimal64` input and keep a
//! `Decimal128` sum. Each input value is sign-extended to 128 bits and added
//! with wrapping arithmetic, which is what `sum` over the same values stored
//! as `Decimal128` computes. The state and the result therefore have the type
//! and the value of that `sum`.

use std::marker::PhantomData;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Decimal128Array, PrimitiveArray,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Decimal128Type, DecimalType};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{Accumulator, EmitTo, GroupSelection, GroupsAccumulator};
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::accumulate::NullState;

/// The wrapping sum of the valid values of `values`, each widened to `i128`;
/// `None` if no value is valid.
fn widened_total<I>(values: &PrimitiveArray<I>) -> Option<i128>
where
    I: DecimalType,
    I::Native: Into<i128>,
{
    if values.null_count() == values.len() {
        return None;
    }
    let total = match values.nulls().filter(|nulls| nulls.null_count() > 0) {
        None => values
            .values()
            .iter()
            .fold(0_i128, |total, value| total.wrapping_add((*value).into())),
        Some(nulls) => values
            .values()
            .iter()
            .zip(nulls.iter())
            .filter(|(_, valid)| *valid)
            .fold(0_i128, |total, (value, _)| {
                total.wrapping_add((*value).into())
            }),
    };
    Some(total)
}

/// `sum` of `I` input into one `Decimal128` value.
pub(crate) struct WideningSumAccumulator<I> {
    sum: Option<i128>,
    data_type: DataType,
    input: PhantomData<I>,
}

impl<I> std::fmt::Debug for WideningSumAccumulator<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WideningSumAccumulator({})", self.data_type)
    }
}

impl<I> WideningSumAccumulator<I> {
    /// `data_type` is the `Decimal128` type of the sum.
    pub(crate) fn new(data_type: DataType) -> Self {
        Self {
            sum: None,
            data_type,
            input: PhantomData,
        }
    }

    fn add(&mut self, total: Option<i128>) {
        if let Some(total) = total {
            let sum = self.sum.get_or_insert(0);
            *sum = sum.wrapping_add(total);
        }
    }
}

impl<I> Accumulator for WideningSumAccumulator<I>
where
    I: DecimalType + Send + Sync,
    I::Native: Into<i128>,
{
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.add(widened_total(values[0].as_primitive::<I>()));
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.add(widened_total(states[0].as_primitive::<Decimal128Type>()));
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        ScalarValue::new_primitive::<Decimal128Type>(self.sum, &self.data_type)
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }
}

/// Grouped `sum` of `I` input into `Decimal128` values.
pub(crate) struct WideningSumGroupsAccumulator<I> {
    /// The sum of each group.
    sums: Vec<i128>,
    /// The `Decimal128` type of the sums.
    data_type: DataType,
    null_state: NullState,
    input: PhantomData<I>,
}

impl<I> std::fmt::Debug for WideningSumGroupsAccumulator<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WideningSumGroupsAccumulator({})", self.data_type)
    }
}

impl<I> WideningSumGroupsAccumulator<I> {
    /// `data_type` is the `Decimal128` type of the sums.
    pub(crate) fn new(data_type: DataType) -> Self {
        Self {
            sums: vec![],
            data_type,
            null_state: NullState::new(),
            input: PhantomData,
        }
    }

    fn accumulate<T>(
        &mut self,
        values: &PrimitiveArray<T>,
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) where
        T: DecimalType + Send,
        T::Native: Into<i128>,
    {
        self.sums.resize(total_num_groups, 0);
        let sums = &mut self.sums;
        self.null_state.accumulate(
            group_indices,
            values,
            opt_filter,
            total_num_groups,
            |group_index, value| {
                // SAFETY: `NullState::accumulate` passes the group indices of
                // `group_indices`, which are below `total_num_groups`, the
                // length of `sums` after the resize above.
                let sum = unsafe { sums.get_unchecked_mut(group_index) };
                *sum = sum.wrapping_add(value.into());
            },
        );
    }
}

impl<I> GroupsAccumulator for WideningSumGroupsAccumulator<I>
where
    I: DecimalType + Send + Sync,
    I::Native: Into<i128>,
{
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "single argument to update_batch");
        self.accumulate(
            values[0].as_primitive::<I>(),
            group_indices,
            opt_filter,
            total_num_groups,
        );
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let sums = emit_to.take_needed(&mut self.sums);
        let nulls = self.null_state.build(emit_to);
        let sums = Decimal128Array::new(sums.into(), nulls)
            .with_data_type(self.data_type.clone());
        Ok(Arc::new(sums))
    }

    fn evaluate_preserving(&mut self, selection: GroupSelection<'_>) -> Result<ArrayRef> {
        selection.validate_num_groups(self.sums.len())?;
        let sums: Vec<_> = selection.iter().map(|i| self.sums[i]).collect();
        let nulls = self.null_state.build_preserving(selection)?;
        Ok(Arc::new(
            Decimal128Array::new(sums.into(), nulls)
                .with_data_type(self.data_type.clone()),
        ))
    }
    fn supports_evaluate_preserving(&self) -> bool {
        true
    }
    fn state_preserving(
        &mut self,
        selection: GroupSelection<'_>,
    ) -> Result<Vec<ArrayRef>> {
        self.evaluate_preserving(selection).map(|sums| vec![sums])
    }
    fn supports_state_preserving(&self) -> bool {
        true
    }
    fn shrink_to_fit(&mut self) {
        self.sums.shrink_to_fit();
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        self.evaluate(emit_to).map(|sums| vec![sums])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "single argument to merge_batch");
        // A state is a `Decimal128` sum.
        self.accumulate(
            values[0].as_primitive::<Decimal128Type>(),
            group_indices,
            None,
            total_num_groups,
        );
        Ok(())
    }

    /// The state of one input row is its widened value, or null when the row is
    /// null or filtered out.
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let values = values[0].as_primitive::<I>();
        let nulls = match opt_filter {
            None => values.nulls().cloned(),
            Some(filter) => {
                let (selected, filter_nulls) = filter.clone().into_parts();
                let selected = match filter_nulls {
                    Some(filter_nulls) => filter_nulls.inner() & &selected,
                    None => selected,
                };
                NullBuffer::union(values.nulls(), Some(&NullBuffer::from(selected)))
            }
        };
        let widened: Vec<i128> = values
            .values()
            .iter()
            .map(|value| (*value).into())
            .collect();
        let state = Decimal128Array::new(widened.into(), nulls)
            .with_data_type(self.data_type.clone());
        Ok(vec![Arc::new(state)])
    }

    fn size(&self) -> usize {
        self.sums.capacity() * size_of::<i128>() + self.null_state.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Decimal64Array, UInt64Array};
    use arrow::datatypes::Decimal64Type;

    fn narrow(values: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(
            Decimal64Array::from(values)
                .with_precision_and_scale(18, 2)
                .unwrap(),
        )
    }

    fn wide_sums(array: &ArrayRef) -> Vec<Option<i128>> {
        assert_eq!(array.data_type(), &DataType::Decimal128(28, 2));
        array.as_primitive::<Decimal128Type>().iter().collect()
    }

    #[test]
    fn a_sum_beyond_64_bits_does_not_wrap() -> Result<()> {
        let mut sum =
            WideningSumAccumulator::<Decimal64Type>::new(DataType::Decimal128(28, 2));
        sum.update_batch(&[narrow(vec![Some(i64::MAX), None, Some(i64::MAX)])])?;
        sum.update_batch(&[narrow(vec![Some(i64::MAX)])])?;
        assert_eq!(
            sum.evaluate()?,
            ScalarValue::Decimal128(Some(3 * i128::from(i64::MAX)), 28, 2)
        );
        // A state merges as a `Decimal128` sum.
        let state: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(-3 * i128::from(i64::MAX)), None])
                .with_precision_and_scale(28, 2)?,
        );
        sum.merge_batch(&[state])?;
        assert_eq!(sum.evaluate()?, ScalarValue::Decimal128(Some(0), 28, 2));
        Ok(())
    }

    #[test]
    fn a_sum_of_no_valid_value_is_null() -> Result<()> {
        let mut sum =
            WideningSumAccumulator::<Decimal64Type>::new(DataType::Decimal128(28, 2));
        sum.update_batch(&[narrow(vec![None, None])])?;
        sum.update_batch(&[narrow(vec![])])?;
        assert_eq!(sum.evaluate()?, ScalarValue::Decimal128(None, 28, 2));
        Ok(())
    }

    #[test]
    fn grouped_sums_widen_filter_and_merge() -> Result<()> {
        let mut sums = WideningSumGroupsAccumulator::<Decimal64Type>::new(
            DataType::Decimal128(28, 2),
        );
        let values = narrow(vec![
            Some(i64::MAX),
            Some(i64::MAX),
            Some(i64::MIN),
            None,
            Some(7),
        ]);
        sums.update_batch(std::slice::from_ref(&values), &[0, 0, 1, 2, 2], None, 4)?;
        // The filter drops the row of group 2.
        let filter = BooleanArray::from(vec![true, false, true, true, false]);
        sums.update_batch(&[values], &[0, 0, 1, 2, 2], Some(&filter), 4)?;
        let state: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(1), None]).with_precision_and_scale(28, 2)?,
        );
        sums.merge_batch(&[state], &[3, 3], 4)?;
        assert_eq!(
            wide_sums(&sums.evaluate(EmitTo::All)?),
            vec![
                Some(3 * i128::from(i64::MAX)),
                Some(2 * i128::from(i64::MIN)),
                Some(7),
                Some(1),
            ]
        );
        Ok(())
    }

    #[test]
    fn a_row_converts_to_its_widened_state() -> Result<()> {
        let sums = WideningSumGroupsAccumulator::<Decimal64Type>::new(
            DataType::Decimal128(28, 2),
        );
        let values = narrow(vec![Some(i64::MIN), None, Some(5), Some(9)]);
        let filter = BooleanArray::from(vec![Some(true), Some(true), Some(false), None]);
        let state = sums.convert_to_state(&[values], Some(&filter))?;
        assert_eq!(
            wide_sums(&state[0]),
            vec![Some(i128::from(i64::MIN)), None, None, None]
        );
        Ok(())
    }

    /// The widening sums equal `sum` over the same values stored as
    /// `Decimal128`, on deterministic random batches with nulls and filters.
    #[test]
    fn grouped_sums_equal_the_decimal128_sums() -> Result<()> {
        use datafusion_functions_aggregate_common::aggregate::groups_accumulator::prim_op::PrimitiveGroupsAccumulator;
        use arrow::array::ArrowNativeTypeOp;

        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            seed.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        let data_type = DataType::Decimal128(28, 2);
        let mut narrow_sums =
            WideningSumGroupsAccumulator::<Decimal64Type>::new(data_type.clone());
        let mut wide_sums_ =
            PrimitiveGroupsAccumulator::<Decimal128Type, _>::new(&data_type, |x, y| {
                *x = x.add_wrapping(y)
            });
        let groups = 17;
        for _ in 0..200 {
            let rows = (next() % 300) as usize;
            let values: Vec<Option<i64>> = (0..rows)
                .map(|_| match next() % 5 {
                    0 => None,
                    1 => Some(i64::MAX - (next() % 1000) as i64),
                    2 => Some(i64::MIN + (next() % 1000) as i64),
                    _ => Some(next() as i64 % 1_000_000_000_000_000),
                })
                .collect();
            let indices: Vec<usize> =
                (0..rows).map(|_| (next() % groups) as usize).collect();
            let filter = (next() % 2 == 0).then(|| {
                BooleanArray::from(
                    (0..rows).map(|_| next() % 3 != 0).collect::<Vec<bool>>(),
                )
            });
            let wide: ArrayRef = Arc::new(
                Decimal128Array::from(
                    values
                        .iter()
                        .map(|value| value.map(i128::from))
                        .collect::<Vec<_>>(),
                )
                .with_precision_and_scale(18, 2)?,
            );
            narrow_sums.update_batch(
                &[narrow(values)],
                &indices,
                filter.as_ref(),
                groups as usize,
            )?;
            wide_sums_.update_batch(
                &[wide],
                &indices,
                filter.as_ref(),
                groups as usize,
            )?;
        }
        let narrow_result = narrow_sums.evaluate(EmitTo::All)?;
        let wide_result = wide_sums_.evaluate(EmitTo::All)?;
        assert_eq!(&narrow_result, &wide_result);
        // The count of groups is unchanged by the comparison above.
        assert_eq!(
            UInt64Array::from(vec![narrow_result.len() as u64]).value(0),
            groups
        );
        Ok(())
    }
}

#[cfg(test)]
mod preserving_tests {
    use super::*;
    use arrow::datatypes::Decimal64Type;

    #[test]
    fn widening_sum_preserves_selected_states_and_nulls() -> Result<()> {
        let ty = DataType::Decimal128(25, 2);
        let mut acc = WideningSumGroupsAccumulator::<Decimal64Type>::new(ty);
        let values = Arc::new(
            arrow::array::Decimal64Array::from(vec![Some(5), None, Some(7)])
                .with_precision_and_scale(15, 2)?,
        ) as ArrayRef;
        acc.update_batch(&[values], &[0, 1, 0], None, 3)?;
        let selection = GroupSelection::try_from_indices(&[2, 0, 1, 0], 3)?;
        assert!(acc.supports_evaluate_preserving());
        assert!(acc.supports_state_preserving());
        let expected = acc.evaluate_preserving(selection)?;
        for _ in 0..2 {
            assert_eq!(
                acc.state_preserving(selection)?,
                vec![Arc::clone(&expected)]
            );
        }
        acc.shrink_to_fit();
        assert!(acc.state_preserving(GroupSelection::all(4)).is_err());
        assert_eq!(
            acc.evaluate_preserving(GroupSelection::all(3))?.as_ref(),
            acc.evaluate(EmitTo::All)?.as_ref()
        );
        Ok(())
    }
}
