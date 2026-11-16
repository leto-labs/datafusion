// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;
use arrow::array::types::{
    Decimal128Type, Float16Type, Float32Type, Float64Type, Int8Type, Int32Type,
    Int64Type, UInt64Type,
};

// Frozen pre-AG primitive table. It keeps the key-vector indirection and cached
// hashes so randomized tests compare independent storage and lookup algorithms.
// The oracle uses checked indexing instead of the original unchecked load.
struct LegacyGroupValuesPrimitive<T: ArrowPrimitiveType> {
    /// The data type of the output array
    data_type: DataType,
    /// Stores the `(group_index, hash)` based on the hash of its value
    ///
    /// We also store `hash` is for reducing cost of rehashing. Such cost
    /// is obvious in high cardinality group by situation.
    /// More details can see:
    /// <https://github.com/apache/datafusion/issues/15961>
    map: HashTable<(usize, u64)>,
    /// The group index of the null value if any
    null_group: Option<usize>,
    /// The values for each group index
    values: Vec<T::Native>,
    /// The random state used to generate hashes
    random_state: RandomState,
}

impl<T: ArrowPrimitiveType> LegacyGroupValuesPrimitive<T> {
    pub fn new(data_type: DataType) -> Self {
        assert!(PrimitiveArray::<T>::is_compatible(&data_type));
        Self {
            data_type,
            map: HashTable::with_capacity(128),
            values: Vec::with_capacity(128),
            null_group: None,
            random_state: crate::aggregates::AGGREGATION_HASH_SEED,
        }
    }
}

impl<T: ArrowPrimitiveType> GroupValues for LegacyGroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);
        groups.clear();

        for v in cols[0].as_primitive::<T>() {
            let group_id = match v {
                None => *self.null_group.get_or_insert_with(|| {
                    let group_id = self.values.len();
                    self.values.push(Default::default());
                    group_id
                }),
                Some(key) => {
                    // Fold equivalence-class duplicates (e.g. `-0.0` → `+0.0`)
                    // so the bit-equal `is_eq` matches and the stored value is
                    // the canonical representative.
                    let key = key.normalize_key();
                    let state = &self.random_state;
                    let hash = key.hash(state);
                    let insert = self.map.entry(
                        hash,
                        |&(g, h)| hash == h && self.values[g].is_eq(key),
                        |&(_, h)| h,
                    );

                    match insert {
                        hashbrown::hash_table::Entry::Occupied(o) => o.get().0,
                        hashbrown::hash_table::Entry::Vacant(v) => {
                            let g = self.values.len();
                            v.insert((g, hash));
                            self.values.push(key);
                            g
                        }
                    }
                }
            };
            groups.push(group_id)
        }
        Ok(())
    }

    fn size(&self) -> usize {
        self.map.capacity() * size_of::<(usize, u64)>() + self.values.allocated_size()
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
                self.map.retain(|entry| {
                    // Decrement group index by n
                    let group_idx = entry.0;
                    match group_idx.checked_sub(n) {
                        // Group index was >= n, shift value down
                        Some(sub) => {
                            entry.0 = sub;
                            true
                        }
                        // Group index was < n, so remove from table
                        None => false,
                    }
                });
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

    fn values_preserving(
        &mut self,
        selection: GroupSelection<'_>,
    ) -> Result<Vec<ArrayRef>> {
        selection.validate_num_groups(self.values.len())?;
        let values: Vec<T::Native> =
            selection.iter().map(|index| self.values[index]).collect();
        let nulls = if let Some(null_group) = self.null_group {
            let mut nulls = NullBufferBuilder::new(values.len());
            for index in selection.iter() {
                if index == null_group {
                    nulls.append_null();
                } else {
                    nulls.append_non_null();
                }
            }
            nulls.finish()
        } else {
            None
        };
        let array = PrimitiveArray::<T>::new(values.into(), nulls)
            .with_data_type(self.data_type.clone());
        Ok(vec![Arc::new(array)])
    }

    fn supports_values_preserving(&self) -> bool {
        true
    }

    fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit(|&(_, hash)| hash);
        self.values.shrink_to_fit();
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.values.clear();
        self.values.shrink_to(num_rows);
        self.map.clear();
        self.map.shrink_to(num_rows, |_| 0); // hasher does not matter since the map is cleared
    }
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn differential<T: ArrowPrimitiveType>(
    data_type: DataType,
    value: impl Fn(u64) -> T::Native,
) -> Result<()>
where
    T::Native: HashValue,
{
    for seed in 1..=4 {
        let domains = if cfg!(feature = "force_hash_collisions") {
            [1, 17, 127]
        } else {
            [1, 17, 16_381]
        };
        for domain in domains {
            let mut actual = GroupValuesPrimitive::<T>::new(data_type.clone());
            let mut expected = LegacyGroupValuesPrimitive::<T>::new(data_type.clone());
            let mut state = seed;
            for cycle in 0..2 {
                let largest = if cfg!(feature = "force_hash_collisions") {
                    512
                } else {
                    8192
                };
                for len in [0, 1, 63, 1024, largest] {
                    let items: Vec<Option<T::Native>> = (0..len)
                        .map(|_| {
                            let r = next(&mut state);
                            (!r.is_multiple_of(7)).then(|| value(r % domain))
                        })
                        .collect();
                    let array: ArrayRef = Arc::new(
                        PrimitiveArray::<T>::from_iter(items)
                            .with_data_type(data_type.clone()),
                    );
                    // Array slicing exercises nonzero validity and value offsets.
                    let array = if len > 10 {
                        array.slice(3, len - 7)
                    } else {
                        array
                    };
                    let mut actual_groups = vec![];
                    let mut expected_groups = vec![];
                    actual.intern(&[Arc::clone(&array)], &mut actual_groups)?;
                    expected.intern(&[array], &mut expected_groups)?;
                    assert_eq!(
                        actual_groups, expected_groups,
                        "seed={seed} domain={domain} cycle={cycle} len={len}"
                    );
                    assert_eq!(actual.len(), expected.len());
                    if len == 1024 {
                        let n = actual.len() / 3;
                        assert_eq!(
                            actual.emit(EmitTo::First(n))?,
                            expected.emit(EmitTo::First(n))?
                        );
                    }
                }
                assert_eq!(actual.emit(EmitTo::All)?, expected.emit(EmitTo::All)?);
                actual.clear_shrink(cycle * 17);
                expected.clear_shrink(cycle * 17);
            }
        }
    }
    Ok(())
}

#[test]
fn primitive_randomized_intern_matches_previous_table() -> Result<()> {
    differential::<Int8Type>(DataType::Int8, |v| v as i8)?;
    differential::<Int32Type>(DataType::Int32, |v| (v as i32).wrapping_mul(-1_234_567))?;
    differential::<Int64Type>(DataType::Int64, |v| {
        v.wrapping_mul(0x9e3779b97f4a7c15) as i64
    })?;
    differential::<UInt64Type>(DataType::UInt64, |v| {
        v.wrapping_mul(0x9e3779b97f4a7c15).rotate_left(17)
    })?;
    differential::<Decimal128Type>(DataType::Decimal128(38, 2), |v| {
        v as i128 * 123_456_789_123_456_789
    })?;
    Ok(())
}

#[test]
fn primitive_float_bits_and_nulls_match_previous_table() -> Result<()> {
    differential::<Float64Type>(DataType::Float64, |v| match v % 11 {
        0 => -0.0,
        1 => 0.0,
        2 => f64::NAN,
        3 => f64::from_bits(0x7ff8_0000_0000_0001),
        4 => f64::INFINITY,
        5 => f64::NEG_INFINITY,
        _ => v as f64,
    })?;
    differential::<Float32Type>(DataType::Float32, |v| match v % 11 {
        0 => -0.0,
        1 => 0.0,
        2 => f32::NAN,
        3 => f32::from_bits(0x7fc0_0001),
        _ => v as f32,
    })?;
    differential::<Float16Type>(DataType::Float16, |v| match v % 11 {
        0 => f16::from_bits(0x8000),
        1 => f16::from_bits(0),
        2 => f16::from_bits(0x7e00),
        3 => f16::from_bits(0x7e01),
        _ => f16::from_f32(v as f32),
    })?;
    Ok(())
}

#[test]
fn primitive_clear_shrink_forgets_null_group() -> Result<()> {
    let mut table = GroupValuesPrimitive::<Int64Type>::new(DataType::Int64);
    let array: ArrayRef =
        Arc::new(PrimitiveArray::<Int64Type>::from(vec![Some(1), None]));
    table.intern(&[array], &mut vec![])?;
    table.clear_shrink(0);
    let array: ArrayRef = Arc::new(PrimitiveArray::<Int64Type>::from(vec![None]));
    let mut groups = vec![];
    table.intern(&[array], &mut groups)?;
    assert_eq!(groups, vec![0]);
    let output = table.emit(EmitTo::All)?;
    assert_eq!(output[0].len(), 1);
    assert_eq!(output[0].null_count(), 1);
    Ok(())
}
