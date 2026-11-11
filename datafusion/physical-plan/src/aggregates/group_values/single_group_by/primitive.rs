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

use crate::aggregates::group_values::GroupValues;
use arrow::array::types::{IntervalDayTime, IntervalMonthDayNano};
use arrow::array::{
    ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, NullBufferBuilder, PrimitiveArray,
    cast::AsArray,
};
use arrow::datatypes::{DataType, i256};
use datafusion_common::Result;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::utils::split_vec_min_alloc;
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::EmitTo;
use half::f16;
use hashbrown::hash_table::HashTable;
#[cfg(not(feature = "force_hash_collisions"))]
use std::hash::BuildHasher;
use std::mem::size_of;
use std::sync::Arc;

/// A trait to allow hashing of floating point numbers
pub trait HashValue {
    fn hash(&self, state: &RandomState) -> u64;

    /// Return a canonical representative whose bit pattern is identical for
    /// all values that should be grouped together. Default is the identity;
    /// floats override this to fold `-0.0` into `+0.0` so the bit-equal
    /// `is_eq` check used during insertion treats them as the same group.
    /// NaN payload bits are preserved.
    #[inline]
    fn canonicalize(self) -> Self
    where
        Self: Sized,
    {
        self
    }
}

macro_rules! hash_integer {
    ($($t:ty),+) => {
        $(impl HashValue for $t {
            #[cfg(not(feature = "force_hash_collisions"))]
            fn hash(&self, state: &RandomState) -> u64 {
                state.hash_one(self)
            }

            #[cfg(feature = "force_hash_collisions")]
            fn hash(&self, _state: &RandomState) -> u64 {
                0
            }
        })+
    };
}
hash_integer!(i8, i16, i32, i64, i128, i256);
hash_integer!(u8, u16, u32, u64);
hash_integer!(IntervalDayTime, IntervalMonthDayNano);

macro_rules! hash_float {
    ($($t:ty),+) => {
        $(impl HashValue for $t {
            #[cfg(not(feature = "force_hash_collisions"))]
            fn hash(&self, state: &RandomState) -> u64 {
                state.hash_one(self.canonicalize().to_bits())
            }

            #[cfg(feature = "force_hash_collisions")]
            fn hash(&self, _state: &RandomState) -> u64 {
                0
            }

            #[inline]
            fn canonicalize(self) -> Self {
                let bits = self.to_bits();
                let bits = if bits << 1 == 0 { 0 } else { bits };
                Self::from_bits(bits)
            }
        })+
    };
}

hash_float!(f16, f32, f64);

/// Small primitive keys stay in the probe slot. Wider keys keep cached hashes
/// and an index so Decimal128/256 and wide intervals do not enlarge every slot.
enum PrimitiveMap<N> {
    Inline(HashTable<(N, usize)>),
    Indirect(HashTable<(usize, u64)>),
}

impl<N> PrimitiveMap<N> {
    fn clear(&mut self) {
        match self {
            Self::Inline(map) => map.clear(),
            Self::Indirect(map) => map.clear(),
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Inline(map) => map.capacity() * size_of::<(N, usize)>(),
            Self::Indirect(map) => map.capacity() * size_of::<(usize, u64)>(),
        }
    }

    fn remove_first(&mut self, n: usize) {
        match self {
            Self::Inline(map) => map.retain(|(_, group)| match group.checked_sub(n) {
                Some(remaining) => {
                    *group = remaining;
                    true
                }
                None => false,
            }),
            Self::Indirect(map) => map.retain(|(group, _)| match group.checked_sub(n) {
                Some(remaining) => {
                    *group = remaining;
                    true
                }
                None => false,
            }),
        }
    }

    fn shrink_to(&mut self, n: usize) {
        // The caller clears the table first; no key will be rehashed.
        match self {
            Self::Inline(map) => map.shrink_to(n, |_| 0),
            Self::Indirect(map) => map.shrink_to(n, |_| 0),
        }
    }
}

/// A [`GroupValues`] storing a single column of primitive values
///
/// This specialization is significantly faster than using the more general
/// purpose `Row`s format
pub struct GroupValuesPrimitive<T: ArrowPrimitiveType> {
    data_type: DataType,
    map: PrimitiveMap<T::Native>,
    null_group: Option<usize>,
    /// Kept in first-seen order for group-index addressing and emission.
    values: Vec<T::Native>,
    random_state: RandomState,
}

impl<T: ArrowPrimitiveType> GroupValuesPrimitive<T> {
    pub fn new(data_type: DataType) -> Self {
        assert!(PrimitiveArray::<T>::is_compatible(&data_type));
        let map = if size_of::<T::Native>() <= size_of::<u64>() {
            PrimitiveMap::Inline(HashTable::with_capacity(128))
        } else {
            PrimitiveMap::Indirect(HashTable::with_capacity(128))
        };
        Self {
            data_type,
            map,
            values: Vec::with_capacity(128),
            null_group: None,
            random_state: crate::aggregates::AGGREGATION_HASH_SEED,
        }
    }
}

/// Share null handling and group-id order, while choosing the table layout once
/// per batch. The lookup closure is monomorphized for each layout and key type.
fn intern_batch<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    groups: &mut Vec<usize>,
    values: &mut Vec<T::Native>,
    null_group: &mut Option<usize>,
    mut intern_key: impl FnMut(T::Native, &mut Vec<T::Native>) -> usize,
) where
    T::Native: HashValue,
{
    groups.clear();
    for value in array {
        let group = match value {
            None => *null_group.get_or_insert_with(|| {
                let group = values.len();
                values.push(Default::default());
                group
            }),
            Some(key) => intern_key(key.canonicalize(), values),
        };
        groups.push(group);
    }
}

impl<T: ArrowPrimitiveType> GroupValues for GroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);
        let array = cols[0].as_primitive::<T>();
        let state = &self.random_state;
        match &mut self.map {
            PrimitiveMap::Inline(map) => intern_batch::<T>(
                array,
                groups,
                &mut self.values,
                &mut self.null_group,
                |key, values| match map.entry(
                    key.hash(state),
                    |&(stored_key, _)| stored_key.is_eq(key),
                    |&(stored_key, _)| stored_key.hash(state),
                ) {
                    hashbrown::hash_table::Entry::Occupied(entry) => entry.get().1,
                    hashbrown::hash_table::Entry::Vacant(entry) => {
                        let group = values.len();
                        entry.insert((key, group));
                        values.push(key);
                        group
                    }
                },
            ),
            PrimitiveMap::Indirect(map) => intern_batch::<T>(
                array,
                groups,
                &mut self.values,
                &mut self.null_group,
                |key, values| {
                    let hash = key.hash(state);
                    match map.entry(
                        hash,
                        |&(group, stored_hash)| {
                            // SAFETY: only live group indices are stored in the
                            // map, and emit(First) shifts them with the values.
                            hash == stored_hash
                                && unsafe { values.get_unchecked(group).is_eq(key) }
                        },
                        |&(_, stored_hash)| stored_hash,
                    ) {
                        hashbrown::hash_table::Entry::Occupied(entry) => entry.get().0,
                        hashbrown::hash_table::Entry::Vacant(entry) => {
                            let group = values.len();
                            entry.insert((group, hash));
                            values.push(key);
                            group
                        }
                    }
                },
            ),
        }
        Ok(())
    }

    fn size(&self) -> usize {
        self.map.size() + self.values.allocated_size()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        fn build_primitive<T: ArrowPrimitiveType>(
            values: Vec<T::Native>,
            null_idx: Option<usize>,
        ) -> PrimitiveArray<T> {
            let nulls = null_idx.map(|null_idx| {
                let mut buffer = NullBufferBuilder::new(values.len());
                buffer.append_n_non_nulls(null_idx);
                buffer.append_null();
                buffer.append_n_non_nulls(values.len() - null_idx - 1);
                // NOTE: The inner builder must be constructed as there is at least one null
                buffer.finish().unwrap()
            });
            PrimitiveArray::<T>::new(values.into(), nulls)
        }

        let array: PrimitiveArray<T> = match emit_to {
            EmitTo::All => {
                self.map.clear();
                build_primitive(std::mem::take(&mut self.values), self.null_group.take())
            }
            EmitTo::First(n) => {
                self.map.remove_first(n);
                let null_group = match &mut self.null_group {
                    Some(v) if *v >= n => {
                        *v -= n;
                        None
                    }
                    Some(_) => self.null_group.take(),
                    None => None,
                };
                build_primitive(split_vec_min_alloc(&mut self.values, n), null_group)
            }
        };

        Ok(vec![Arc::new(array.with_data_type(self.data_type.clone()))])
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.values.clear();
        self.null_group = None;
        self.values.shrink_to(num_rows);
        self.map.clear();
        self.map.shrink_to(num_rows);
    }
}

#[cfg(test)]
#[path = "primitive_tests.rs"]
mod probe_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::types::Int32Type;
    use arrow::array::{ArrayRef, Int32Array};
    use arrow::datatypes::DataType;
    use datafusion_expr::EmitTo;
    use std::sync::Arc;

    /// Mirror of the `EmitTo::take_needed` regression test, applied to the
    /// concrete `GroupValuesPrimitive` accumulator.
    ///
    /// When `n` is small, the old `split_off(n) + swap` pattern used inside
    /// `emit(EmitTo::First(n))` left `self.values` with a small fresh allocation
    /// and returned the emitted prefix carrying the original large backing.
    ///
    /// With `split_vec_min_alloc` and `n * 2 <= len`, the drain branch is taken:
    /// the emitted prefix gets a compact allocation and `self.values` retains the
    /// original large one.
    #[test]
    fn emit_first_small_n_allocates_minimally() -> Result<()> {
        let mut gv = GroupValuesPrimitive::<Int32Type>::new(DataType::Int32);

        // Intern 20 distinct values; `new()` pre-allocates capacity 128 for `values`.
        let arr: ArrayRef = Arc::new(Int32Array::from_iter_values(0..20i32));
        let mut groups = vec![];
        gv.intern(&[arr], &mut groups)?;
        let capacity_before = gv.values.capacity(); // 128

        // n=4, n*2=8 <= len=20 -> drain branch
        let emitted = gv.emit(EmitTo::First(4))?;

        assert_eq!(emitted[0].len(), 4);

        // `self.values` must retain its original large allocation.
        // Old split_off+swap left it with a fresh small allocation (~16).
        assert_eq!(
            gv.values.capacity(),
            capacity_before,
            "self.values capacity {} should equal original {} after small First(n) emit",
            gv.values.capacity(),
            capacity_before,
        );

        Ok(())
    }
}
