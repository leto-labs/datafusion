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

use super::*;
use arrow::array::{Int64Array, UInt64Array};

#[test]
fn integer_avg_exact_signed_cancellation_and_retraction() -> Result<()> {
    let mut acc = avg_accumulator(&DataType::Int64, &DataType::Float64)?;
    let values: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, i64::MIN]));
    acc.update_batch(std::slice::from_ref(&values))?;
    assert_eq!(acc.evaluate()?, ScalarValue::Float64(Some(-0.5)));
    assert!(acc.supports_retract_batch());
    acc.retract_batch(&[Arc::new(Int64Array::from(vec![i64::MAX]))])?;
    assert_eq!(acc.evaluate()?, ScalarValue::Float64(Some(i64::MIN as f64)));
    acc.retract_batch(&[Arc::new(Int64Array::from(vec![i64::MIN]))])?;
    assert_eq!(acc.evaluate()?, ScalarValue::Float64(None));
    Ok(())
}

#[test]
fn integer_avg_groups_keep_unsigned_values_nulls_filters_and_state() -> Result<()> {
    let values: ArrayRef = Arc::new(UInt64Array::from(vec![
        Some(u64::MAX),
        Some(u64::MAX - 2),
        None,
        Some(7),
        Some(9),
    ]));
    let filter =
        BooleanArray::from(vec![Some(true), Some(true), Some(true), None, Some(false)]);
    let mut acc = avg_groups_accumulator(&DataType::UInt64, &DataType::Float64)?;
    acc.update_batch(
        std::slice::from_ref(&values),
        &[0, 0, 1, 1, 2],
        Some(&filter),
        3,
    )?;
    let state = acc.state(EmitTo::All)?;
    let converted = acc.convert_to_state(std::slice::from_ref(&values), Some(&filter))?;
    let mut bypass = avg_groups_accumulator(&DataType::UInt64, &DataType::Float64)?;
    bypass.merge_batch(&converted, &[0, 0, 1, 1, 2], 3)?;
    let bypass_result = bypass.evaluate(EmitTo::All)?;
    assert!(bypass_result.is_null(1));
    assert!(bypass_result.is_null(2));
    assert_eq!(
        ScalarValue::try_from_array(bypass_result.as_ref(), 0)?,
        ScalarValue::Float64(Some(((2_i128 * i128::from(u64::MAX)) - 2) as f64 / 2.0))
    );
    let fields = avg_state_fields(&DataType::UInt64, &DataType::Float64)?;
    assert_eq!(fields.len(), 2);
    for (field, array) in fields.iter().zip(&state) {
        assert_eq!(field.data_type(), array.data_type());
    }
    let mut merged = avg_groups_accumulator(&DataType::UInt64, &DataType::Float64)?;
    merged.merge_batch(&state, &[0, 1, 2], 3)?;
    let result = merged.evaluate(EmitTo::All)?;
    assert_eq!(
        ScalarValue::try_from_array(result.as_ref(), 0)?,
        ScalarValue::Float64(Some(((2_i128 * u64::MAX as i128) - 2) as f64 / 2.0))
    );
    assert!(result.is_null(1));
    assert!(result.is_null(2));
    Ok(())
}

fn state_columns(states: &[Vec<ScalarValue>]) -> Result<Vec<ArrayRef>> {
    (0..states[0].len())
        .map(|column| {
            ScalarValue::iter_to_array(states.iter().map(|state| state[column].clone()))
        })
        .collect()
}

#[test]
fn integer_avg_all_widths_are_exact_across_batches_and_merge_orders() -> Result<()> {
    use arrow::array::{
        Int8Array, Int16Array, Int32Array, UInt8Array, UInt16Array, UInt32Array,
    };
    let cases: Vec<(ArrayRef, i128)> = vec![
        (
            Arc::new(Int8Array::from(vec![
                Some(i8::MAX),
                None,
                Some(i8::MIN),
                Some(5),
            ])),
            4,
        ),
        (
            Arc::new(Int16Array::from(vec![
                Some(i16::MAX),
                None,
                Some(i16::MIN),
                Some(5),
            ])),
            4,
        ),
        (
            Arc::new(Int32Array::from(vec![
                Some(i32::MAX),
                None,
                Some(i32::MIN),
                Some(5),
            ])),
            4,
        ),
        (
            Arc::new(Int64Array::from(vec![
                Some(i64::MAX),
                None,
                Some(i64::MIN),
                Some(5),
            ])),
            4,
        ),
        (
            Arc::new(UInt8Array::from(vec![
                Some(u8::MAX),
                None,
                Some(u8::MAX - 1),
                Some(0),
            ])),
            2 * i128::from(u8::MAX) - 1,
        ),
        (
            Arc::new(UInt16Array::from(vec![
                Some(u16::MAX),
                None,
                Some(u16::MAX - 1),
                Some(0),
            ])),
            2 * i128::from(u16::MAX) - 1,
        ),
        (
            Arc::new(UInt32Array::from(vec![
                Some(u32::MAX),
                None,
                Some(u32::MAX - 1),
                Some(0),
            ])),
            2 * i128::from(u32::MAX) - 1,
        ),
        (
            Arc::new(UInt64Array::from(vec![
                Some(u64::MAX),
                None,
                Some(u64::MAX - 1),
                Some(0),
            ])),
            2 * i128::from(u64::MAX) - 1,
        ),
    ];
    for (values, total) in cases {
        let ty = values.data_type();
        let expected = ScalarValue::Float64(Some(total as f64 / 3.0));
        let mut direct = avg_accumulator(ty, &DataType::Float64)?;
        direct.update_batch(std::slice::from_ref(&values))?;
        assert_eq!(direct.evaluate()?, expected, "{ty}");
        let mut states = vec![];
        for i in 0..values.len() {
            let mut partial = avg_accumulator(ty, &DataType::Float64)?;
            partial.update_batch(&[values.slice(i, 1)])?;
            states.push(partial.state()?);
        }
        for _ in 0..values.len() {
            let mut final_acc = avg_accumulator(ty, &DataType::Float64)?;
            final_acc.merge_batch(&state_columns(&states)?)?;
            assert_eq!(final_acc.evaluate()?, expected, "{ty}");
            states.rotate_left(1);
        }
        let groups = avg_groups_accumulator(ty, &DataType::Float64)?;
        let state = groups.convert_to_state(std::slice::from_ref(&values), None)?;
        let mut groups = avg_groups_accumulator(ty, &DataType::Float64)?;
        groups.merge_batch(&state, &[0, 0, 0, 0], 1)?;
        assert_eq!(
            ScalarValue::try_from_array(groups.evaluate(EmitTo::All)?.as_ref(), 0)?,
            expected
        );
    }
    Ok(())
}

#[test]
fn integer_avg_random_signed_values_keep_exact_sum_through_state() -> Result<()> {
    let mut seed = 0x5eed_u64;
    for _ in 0..16 {
        let values: Vec<Option<i64>> = (0..513)
            .map(|i| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                if i % 13 == 0 { None } else { Some(seed as i64) }
            })
            .collect();
        let sum: i128 = values.iter().flatten().map(|&v| i128::from(v)).sum();
        let count = values.iter().flatten().count();
        let values: ArrayRef = Arc::new(Int64Array::from(values));
        let mut states = vec![];
        for offset in (0..values.len()).step_by(17) {
            let mut partial = avg_accumulator(&DataType::Int64, &DataType::Float64)?;
            partial
                .update_batch(&[values.slice(offset, 17.min(values.len() - offset))])?;
            states.push(partial.state()?);
        }
        for order in 0..3 {
            if order == 1 {
                states.reverse();
            } else {
                states.rotate_left(7);
            }
            let mut merged = avg_accumulator(&DataType::Int64, &DataType::Float64)?;
            merged.merge_batch(&state_columns(&states)?)?;
            assert_eq!(
                merged.evaluate()?,
                ScalarValue::Float64(Some(sum as f64 / count as f64))
            );
        }
    }
    Ok(())
}

#[test]
fn integer_avg_distinct_preserves_integer_identity_and_cancellation() -> Result<()> {
    let make = || {
        with_avg_args(&DataType::Int64, &DataType::Float64, |mut args| {
            args.is_distinct = true;
            Avg::new().accumulator(args)
        })
    };
    let mut partial = make()?;
    partial.update_batch(&[Arc::new(Int64Array::from(vec![
        Some(i64::MAX),
        Some(i64::MAX),
        None,
    ]))])?;
    let state = partial
        .state()?
        .iter()
        .map(|v| v.to_array_of_size(1))
        .collect::<Result<Vec<_>>>()?;
    let mut merged = make()?;
    merged.merge_batch(&state)?;
    merged.update_batch(&[Arc::new(Int64Array::from(vec![
        i64::MIN,
        i64::MIN,
        i64::MAX,
    ]))])?;
    assert_eq!(merged.evaluate()?, ScalarValue::Float64(Some(-0.5)));
    let mut adjacent = make()?;
    adjacent.update_batch(&[Arc::new(Int64Array::from(vec![
        1_i64 << 53,
        (1_i64 << 53) + 1,
        (1_i64 << 53) + 2,
    ]))])?;
    // Float coercion would collapse the first two distinct integer keys.
    let state = adjacent.state()?;
    let ScalarValue::List(list) = &state[0] else {
        panic!("expected distinct list state")
    };
    assert_eq!(list.values().len(), 3);
    assert_eq!(list.values().data_type(), &DataType::Int64);
    Ok(())
}

#[test]
fn integer_avg_overflow_and_empty_state_are_explicit() -> Result<()> {
    let mut acc = avg_accumulator(&DataType::UInt64, &DataType::Float64)?;
    assert_eq!(acc.evaluate()?, ScalarValue::Float64(None));
    let max_state = vec![
        ScalarValue::UInt64(Some(1)),
        ScalarValue::FixedSizeBinary(16, Some(i128::MAX.to_le_bytes().to_vec())),
    ];
    acc.merge_batch(&state_columns(&[max_state])?)?;
    assert!(
        acc.update_batch(&[Arc::new(UInt64Array::from(vec![1]))])
            .is_err()
    );
    let mut empty = avg_accumulator(&DataType::Int64, &DataType::Float64)?;
    assert!(
        empty
            .retract_batch(&[Arc::new(Int64Array::from(vec![1]))])
            .is_err()
    );
    let mut groups = avg_groups_accumulator(&DataType::Int64, &DataType::Float64)?;
    let values: ArrayRef = Arc::new(Int64Array::from(vec![None, None]));
    groups.update_batch(&[values], &[0, 1], None, 3)?;
    let first = groups.evaluate(EmitTo::First(2))?;
    assert_eq!(first.null_count(), 2);
    groups.update_batch(&[Arc::new(Int64Array::from(vec![4]))], &[0], None, 1)?;
    assert_eq!(
        ScalarValue::try_from_array(groups.evaluate(EmitTo::All)?.as_ref(), 0)?,
        ScalarValue::Float64(Some(4.0))
    );
    Ok(())
}

#[test]
fn integer_avg_planning_preserves_all_integer_widths() -> Result<()> {
    use datafusion_expr::type_coercion::functions::fields_with_udf;
    let function = avg_udaf();
    for ty in [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
    ] {
        let fields = fields_with_udf(
            &[Arc::new(Field::new("v", ty.clone(), true))],
            function.as_ref(),
        )?;
        assert_eq!(fields[0].data_type(), &ty);
        assert_eq!(Avg::new().return_type(&[ty])?, DataType::Float64);
    }
    for ty in [DataType::Float16, DataType::Float32, DataType::Float64] {
        let fields =
            fields_with_udf(&[Arc::new(Field::new("v", ty, true))], function.as_ref())?;
        assert_eq!(fields[0].data_type(), &DataType::Float64);
    }
    Ok(())
}

#[test]
fn integer_avg_state_survives_ipc_and_count_overflow_is_checked() -> Result<()> {
    use arrow::array::RecordBatch;
    use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
    use std::io::Cursor;
    let cases: Vec<(ArrayRef, f64)> = vec![
        (Arc::new(Int64Array::from(vec![i64::MAX, i64::MIN])), -0.5),
        (
            Arc::new(UInt64Array::from(vec![u64::MAX, u64::MAX - 2])),
            ((2_i128 * i128::from(u64::MAX) - 2) as f64) / 2.0,
        ),
    ];
    for (values, expected) in cases {
        let mut groups = avg_groups_accumulator(values.data_type(), &DataType::Float64)?;
        groups.update_batch(std::slice::from_ref(&values), &[0, 0], None, 1)?;
        let schema = Arc::new(Schema::new(avg_state_fields(
            values.data_type(),
            &DataType::Float64,
        )?));
        let batch = RecordBatch::try_new(schema.clone(), groups.state(EmitTo::All)?)?;
        let mut bytes = vec![];
        {
            let mut writer = StreamWriter::try_new(&mut bytes, &schema)?;
            writer.write(&batch)?;
            writer.finish()?;
        }
        let restored = StreamReader::try_new(Cursor::new(bytes), None)?
            .next()
            .unwrap()?;
        let mut merged = avg_accumulator(values.data_type(), &DataType::Float64)?;
        merged.merge_batch(restored.columns())?;
        assert_eq!(merged.evaluate()?, ScalarValue::Float64(Some(expected)));
    }
    let mut acc = avg_accumulator(&DataType::UInt64, &DataType::Float64)?;
    acc.merge_batch(&state_columns(&[vec![
        ScalarValue::UInt64(Some(u64::MAX)),
        ScalarValue::FixedSizeBinary(16, Some(0_i128.to_le_bytes().to_vec())),
    ]])?)?;
    assert!(
        acc.update_batch(&[Arc::new(UInt64Array::from(vec![0]))])
            .is_err()
    );
    Ok(())
}

#[test]
fn integer_avg_grouped_raw_batches_and_imported_overflow() -> Result<()> {
    let mut acc = avg_groups_accumulator(&DataType::UInt64, &DataType::Float64)?;
    let mut sums = [0_i128; 17];
    let mut counts = [0_u64; 17];
    for batch in 0..7 {
        let values: Vec<Option<u64>> = (0..513)
            .map(|i| (i % 19 != 0).then_some(u64::MAX - (i * 131 + batch) as u64))
            .collect();
        let groups: Vec<_> = (0..values.len()).map(|i| (i * 7 + batch) % 17).collect();
        let filter = BooleanArray::from_iter(
            (0..values.len()).map(|i| if i % 23 == 0 { None } else { Some(i % 11 != 0) }),
        );
        for (i, value) in values.iter().enumerate() {
            if filter.is_valid(i)
                && filter.value(i)
                && let Some(value) = value
            {
                sums[groups[i]] += i128::from(*value);
                counts[groups[i]] += 1;
            }
        }
        acc.update_batch(
            &[Arc::new(UInt64Array::from(values))],
            &groups,
            Some(&filter),
            17,
        )?;
    }
    let state = acc.state(EmitTo::First(6))?;
    for i in 0..6 {
        assert_eq!(state[0].as_primitive::<UInt64Type>().value(i), counts[i]);
        assert_eq!(
            state[1].as_fixed_size_binary().value(i),
            sums[i].to_le_bytes()
        );
    }
    let rest = acc.evaluate(EmitTo::All)?;
    for i in 6..17 {
        assert_eq!(
            ScalarValue::try_from_array(rest.as_ref(), i - 6)?,
            ScalarValue::Float64(Some(sums[i] as f64 / counts[i] as f64))
        );
    }
    for (sum, count) in [(i128::MAX, 1), (0, u64::MAX)] {
        let mut acc = avg_groups_accumulator(&DataType::UInt64, &DataType::Float64)?;
        // Raw updates followed by arbitrary imported state must still check
        // the next raw update, even when a prior emission emptied the group.
        acc.update_batch(&[Arc::new(UInt64Array::from(vec![1]))], &[0], None, 1)?;
        acc.evaluate(EmitTo::All)?;
        let state = state_columns(&[vec![
            ScalarValue::UInt64(Some(count)),
            ScalarValue::FixedSizeBinary(16, Some(sum.to_le_bytes().to_vec())),
        ]])?;
        acc.merge_batch(&state, &[0], 1)?;
        assert!(
            acc.update_batch(&[Arc::new(UInt64Array::from(vec![1]))], &[0], None, 1)
                .is_err()
        );
    }
    Ok(())
}
