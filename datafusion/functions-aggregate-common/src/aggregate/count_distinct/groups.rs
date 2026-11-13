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

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Int64Array, ListBuilder, PrimitiveBuilder,
};
use arrow::datatypes::{ArrowPrimitiveType, Field};
use datafusion_common::HashSet;
use datafusion_common::hash_utils::RandomState;
use datafusion_expr_common::groups_accumulator::{EmitTo, GroupsAccumulator};
use std::hash::Hash;
use std::mem::size_of;
use std::sync::Arc;

use crate::aggregate::groups_accumulator::accumulate::accumulate;

// High-cardinality grouping often has only one distinct value per group.
// Keep that value inline; allocate a per-group hash table only on the second.
enum DistinctSet<N> {
    Empty,
    One(N),
    Many(HashSet<N, RandomState>),
}
impl<N: Copy + Eq + Hash> DistinctSet<N> {
    fn insert(&mut self, value: N) -> usize {
        match self {
            Self::Empty => {
                *self = Self::One(value);
                0
            }
            Self::One(previous) if *previous == value => 0,
            Self::One(previous) => {
                let mut values = HashSet::default();
                values.insert(*previous);
                values.insert(value);
                let allocated = values.allocation_size();
                *self = Self::Many(values);
                allocated
            }
            Self::Many(values) => {
                let previous = values.allocation_size();
                values.insert(value);
                values.allocation_size() - previous
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(_) => 1,
            Self::Many(values) => values.len(),
        }
    }
    fn allocation_size(&self) -> usize {
        match self {
            Self::Many(values) => values.allocation_size(),
            _ => 0,
        }
    }
}

pub struct PrimitiveDistinctCountGroupsAccumulator<T: ArrowPrimitiveType>
where
    T::Native: Eq + Hash,
{
    seen: Vec<DistinctSet<T::Native>>,
    hash_bytes: usize,
}

impl<T: ArrowPrimitiveType> PrimitiveDistinctCountGroupsAccumulator<T>
where
    T::Native: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            seen: Vec::new(),
            hash_bytes: 0,
        }
    }

    // Size is queried for every input batch. Keep it constant-time even when
    // there are millions of groups; only visit states actually being emitted.
    fn take_sets(&mut self, emit_to: EmitTo) -> Vec<DistinctSet<T::Native>> {
        let sets = emit_to.take_needed(&mut self.seen);
        self.hash_bytes -= sets.iter().map(DistinctSet::allocation_size).sum::<usize>();
        sets
    }
}

impl<T: ArrowPrimitiveType> Default for PrimitiveDistinctCountGroupsAccumulator<T>
where
    T::Native: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ArrowPrimitiveType + Send + std::fmt::Debug> GroupsAccumulator
    for PrimitiveDistinctCountGroupsAccumulator<T>
where
    T::Native: Eq + Hash,
{
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> datafusion_common::Result<()> {
        debug_assert_eq!(values.len(), 1);
        self.seen
            .resize_with(total_num_groups, || DistinctSet::Empty);
        let arr = values[0].as_primitive::<T>();
        accumulate(group_indices, arr, opt_filter, |group_idx, value| {
            self.hash_bytes += self.seen[group_idx].insert(value);
        });
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> datafusion_common::Result<ArrayRef> {
        let sets = self.take_sets(emit_to);
        Ok(Arc::new(Int64Array::from_iter_values(
            sets.iter().map(|set| set.len() as i64),
        )))
    }

    fn state(&mut self, emit_to: EmitTo) -> datafusion_common::Result<Vec<ArrayRef>> {
        let sets = self.take_sets(emit_to);
        let total: usize = sets.iter().map(DistinctSet::len).sum();
        if total > i32::MAX as usize {
            return datafusion_common::exec_err!(
                "COUNT DISTINCT state exceeds List offset capacity"
            );
        }
        let values = PrimitiveBuilder::<T>::with_capacity(total);
        let mut lists = ListBuilder::with_capacity(values, sets.len())
            .with_field(Arc::new(Field::new_list_field(T::DATA_TYPE, true)));
        for set in sets {
            match set {
                DistinctSet::Empty => {}
                DistinctSet::One(value) => lists.values().append_value(value),
                DistinctSet::Many(values) => {
                    lists.values().extend(values.into_iter().map(Some))
                }
            }
            lists.append(true);
        }
        Ok(vec![Arc::new(lists.finish())])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> datafusion_common::Result<()> {
        debug_assert_eq!(values.len(), 1);
        self.seen
            .resize_with(total_num_groups, || DistinctSet::Empty);
        let list_array = values[0].as_list::<i32>();
        let inner = list_array.values().as_primitive::<T>();
        let inner_values = inner.values();
        let offsets = list_array.offsets();

        for (row_idx, &group_idx) in group_indices.iter().enumerate() {
            let start = offsets[row_idx] as usize;
            let end = offsets[row_idx + 1] as usize;
            for &value in &inner_values[start..end] {
                self.hash_bytes += self.seen[group_idx].insert(value);
            }
        }

        Ok(())
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> datafusion_common::Result<Vec<ArrayRef>> {
        debug_assert_eq!(values.len(), 1);
        let arr = values[0].as_primitive::<T>();

        let values_builder = PrimitiveBuilder::<T>::with_capacity(arr.len());
        let mut builder = ListBuilder::new(values_builder)
            .with_field(Arc::new(Field::new_list_field(T::DATA_TYPE, true)));

        for row in 0..arr.len() {
            let included = arr.is_valid(row)
                && opt_filter
                    .is_none_or(|filter| filter.is_valid(row) && filter.value(row));
            if included {
                builder.values().append_value(arr.value(row));
            }
            builder.append(true);
        }

        Ok(vec![Arc::new(builder.finish())])
    }
    fn size(&self) -> usize {
        size_of::<Self>()
            + self.seen.capacity() * size_of::<DistinctSet<T::Native>>()
            + self.hash_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::Int32Type;
    use datafusion_common::Result;

    #[test]
    fn convert_to_state_roundtrips_through_merge() -> Result<()> {
        let values = Arc::new(Int32Array::from(vec![
            Some(1),
            Some(2),
            Some(2),
            None,
            Some(3),
            Some(4),
            Some(5),
            Some(5),
        ])) as ArrayRef;
        let filter = BooleanArray::from(vec![
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            None,
            Some(true),
            Some(true),
            Some(true),
        ]);
        let group_indices = vec![0usize, 1, 0, 1, 0, 0, 0, 0];

        let mut direct = PrimitiveDistinctCountGroupsAccumulator::<Int32Type>::new();
        direct.update_batch(
            std::slice::from_ref(&values),
            &group_indices,
            Some(&filter),
            2,
        )?;
        let direct = direct.evaluate(EmitTo::All)?;

        let converter = PrimitiveDistinctCountGroupsAccumulator::<Int32Type>::new();
        let state =
            converter.convert_to_state(std::slice::from_ref(&values), Some(&filter))?;
        assert_eq!(state[0].null_count(), 0);
        let mut merged = PrimitiveDistinctCountGroupsAccumulator::<Int32Type>::new();
        merged.merge_batch(&state, &group_indices, 2)?;
        let merged = merged.evaluate(EmitTo::All)?;

        assert_eq!(
            direct.as_any().downcast_ref::<Int64Array>().unwrap(),
            merged.as_any().downcast_ref::<Int64Array>().unwrap()
        );
        Ok(())
    }

    #[test]
    fn convert_to_state_preserves_empty_and_filtered_rows() -> Result<()> {
        let converter = PrimitiveDistinctCountGroupsAccumulator::<Int32Type>::new();
        let empty_values =
            Arc::new(Int32Array::from(Vec::<Option<i32>>::new())) as ArrayRef;
        let state =
            converter.convert_to_state(std::slice::from_ref(&empty_values), None)?;
        assert_eq!(state[0].len(), 0);
        assert_eq!(state[0].null_count(), 0);

        let values = Arc::new(Int32Array::from(vec![Some(1), Some(2), None])) as ArrayRef;
        let filter = BooleanArray::from(vec![Some(false), None, Some(false)]);
        let group_indices = vec![0usize, 1, 0];

        let state =
            converter.convert_to_state(std::slice::from_ref(&values), Some(&filter))?;
        assert_eq!(state[0].len(), values.len());
        assert_eq!(state[0].null_count(), 0);
        let list_state = state[0].as_list::<i32>();
        for row in 0..list_state.len() {
            assert_eq!(list_state.value_length(row), 0);
        }

        let mut merged = PrimitiveDistinctCountGroupsAccumulator::<Int32Type>::new();
        merged.merge_batch(&state, &group_indices, 2)?;
        let result = merged.evaluate(EmitTo::All)?;
        assert_eq!(
            result.as_any().downcast_ref::<Int64Array>().unwrap(),
            &Int64Array::from(vec![0, 0])
        );
        Ok(())
    }

    #[test]
    fn grouped_distinct_prefix_state_nulls_filters_and_reuse() -> Result<()> {
        use std::collections::BTreeSet;
        let mut acc = PrimitiveDistinctCountGroupsAccumulator::<Int32Type>::new();
        let mut expected = vec![BTreeSet::new(); 53];
        let mut seed = 79_u64;
        for batch in 0..11 {
            let mut values = Vec::new();
            let mut groups = Vec::new();
            let mut filters = Vec::new();
            for row in 0..4096 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let group = (seed >> 32) as usize % expected.len();
                let value = (row % 13 != 0).then_some((seed % 1000) as i32);
                let filter = match row % 5 {
                    0 => None,
                    1 => Some(false),
                    _ => Some(true),
                };
                if let Some(value) = value
                    && (batch % 2 == 0 || filter == Some(true))
                {
                    expected[group].insert(value);
                }
                values.push(value);
                groups.push(group);
                filters.push(filter);
            }
            let values = Arc::new(Int32Array::from(values)) as ArrayRef;
            let filter = BooleanArray::from(filters);
            acc.update_batch(
                &[values],
                &groups,
                (batch % 2 != 0).then_some(&filter),
                expected.len(),
            )?;
        }
        assert_eq!(
            acc.hash_bytes,
            acc.seen
                .iter()
                .map(DistinctSet::allocation_size)
                .sum::<usize>()
        );
        println!("grouped_distinct_53_groups_retained_bytes={}", acc.size());
        let state = acc.state(EmitTo::First(17))?;
        assert_eq!(
            acc.hash_bytes,
            acc.seen
                .iter()
                .map(DistinctSet::allocation_size)
                .sum::<usize>()
        );
        let mut merged = PrimitiveDistinctCountGroupsAccumulator::<Int32Type>::new();
        merged.merge_batch(&state, &(0..17).collect::<Vec<_>>(), 17)?;
        let first = merged.evaluate(EmitTo::All)?;
        assert_eq!(
            first.as_primitive::<arrow::datatypes::Int64Type>(),
            &Int64Array::from_iter_values(expected[..17].iter().map(|s| s.len() as i64))
        );
        let rest = acc.evaluate(EmitTo::All)?;
        assert_eq!(acc.hash_bytes, 0);
        assert_eq!(
            rest.as_primitive::<arrow::datatypes::Int64Type>(),
            &Int64Array::from_iter_values(expected[17..].iter().map(|s| s.len() as i64))
        );
        let values = Arc::new(Int32Array::from(vec![Some(1), None, Some(1)])) as ArrayRef;
        acc.update_batch(&[values], &[0, 1, 0], None, 3)?;
        let reused = acc.evaluate(EmitTo::All)?;
        assert_eq!(
            reused.as_primitive::<arrow::datatypes::Int64Type>(),
            &Int64Array::from(vec![1, 0, 0])
        );
        Ok(())
    }
}
