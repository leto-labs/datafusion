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
use arrow::array::{Int32Array, Int64Array, UInt32Array};

fn narrow_state(count: i64, sum: i128) -> Vec<ArrayRef> {
    vec![
        Arc::new(Int64Array::from(vec![count])),
        Arc::new(Int64Array::from(vec![sum as i64])),
        Arc::new(Int32Array::from(vec![
            i64::try_from(sum).err().map(|_| (sum >> 64) as i32),
        ])),
    ]
}

#[test]
fn integer_avg_narrow_state_promotes_at_signed_and_unsigned_boundaries() -> Result<()> {
    for (ty, value, count) in [
        (DataType::UInt32, i128::from(u32::MAX), 1_i64 << 31),
        (DataType::Int32, i128::from(i32::MIN), 1_i64 << 32),
    ] {
        let fields = avg_state_fields(&ty, &DataType::Float64)?;
        assert_eq!(
            fields.iter().map(|f| f.data_type()).collect::<Vec<_>>(),
            [&DataType::Int64, &DataType::Int64, &DataType::Int32]
        );
        let seed = value * i128::from(count);
        assert!(i64::try_from(seed).is_ok());
        let next = seed + value;
        assert!(i64::try_from(next).is_err());
        let input: ArrayRef = if ty == DataType::UInt32 {
            Arc::new(UInt32Array::from(vec![u32::MAX]))
        } else {
            Arc::new(Int32Array::from(vec![i32::MIN]))
        };
        let mut scalar = avg_accumulator(&ty, &DataType::Float64)?;
        scalar.merge_batch(&narrow_state(count, seed))?;
        scalar.update_batch(std::slice::from_ref(&input))?;
        assert_eq!(
            scalar.evaluate()?,
            ScalarValue::Float64(Some(next as f64 / (count + 1) as f64))
        );
        scalar.retract_batch(std::slice::from_ref(&input))?;
        assert_eq!(scalar.evaluate()?, ScalarValue::Float64(Some(value as f64)));
        let mut groups = avg_groups_accumulator(&ty, &DataType::Float64)?;
        groups.merge_batch(&narrow_state(count, seed), &[0], 2)?;
        groups.update_batch(&[input], &[0], None, 2)?;
        let state = groups.state(EmitTo::All)?;
        assert!(state[2].is_valid(0));
        assert!(state[2].is_null(1));
        let mut restored = avg_groups_accumulator(&ty, &DataType::Float64)?;
        restored.merge_batch(&state, &[0, 1], 2)?;
        let result = restored.evaluate(EmitTo::All)?;
        assert_eq!(
            ScalarValue::try_from_array(result.as_ref(), 0)?,
            ScalarValue::Float64(Some(next as f64 / (count + 1) as f64))
        );
        assert!(result.is_null(1));
    }
    Ok(())
}

#[test]
fn integer_avg_narrow_retraction_can_promote_and_empty_state_resets() -> Result<()> {
    let count = (1_i64 << 32) + 4;
    let promoted = i128::from(i32::MAX) * i128::from(count - 1);
    let sum = promoted + i128::from(i32::MIN);
    assert!(i64::try_from(sum).is_ok());
    assert!(i64::try_from(promoted).is_err());
    let mut acc = avg_accumulator(&DataType::Int32, &DataType::Float64)?;
    acc.merge_batch(&narrow_state(count, sum))?;
    acc.retract_batch(&[Arc::new(Int32Array::from(vec![i32::MIN]))])?;
    assert_eq!(
        acc.evaluate()?,
        ScalarValue::Float64(Some(promoted as f64 / (count - 1) as f64))
    );
    let empty = avg_groups_accumulator(&DataType::Int16, &DataType::Float64)?;
    let values: ArrayRef = Arc::new(arrow::array::Int16Array::from(vec![
        Some(4),
        None,
        Some(9),
        Some(11),
    ]));
    let filter = BooleanArray::from(vec![Some(true), Some(true), None, Some(false)]);
    let state = empty.convert_to_state(&[values], Some(&filter))?;
    assert_eq!(state[0].as_primitive::<Int64Type>().values(), &[1, 0, 0, 0]);
    assert_eq!(state[2].null_count(), 4);
    let mut groups = avg_groups_accumulator(&DataType::Int16, &DataType::Float64)?;
    groups.merge_batch(&state, &[0, 0, 1, 2], 3)?;
    let first = groups.evaluate(EmitTo::First(2))?;
    assert_eq!(
        ScalarValue::try_from_array(first.as_ref(), 0)?,
        ScalarValue::Float64(Some(4.0))
    );
    assert!(first.is_null(1));
    groups.update_batch(
        &[Arc::new(arrow::array::Int16Array::from(vec![7]))],
        &[0],
        None,
        1,
    )?;
    assert_eq!(
        ScalarValue::try_from_array(groups.evaluate(EmitTo::All)?.as_ref(), 0)?,
        ScalarValue::Float64(Some(7.0))
    );
    Ok(())
}

#[test]
fn integer_avg_narrow_distinct_sum_bounds_and_integer_identity() -> Result<()> {
    // Every prefix of a DISTINCT reduction is a subset of the domain. The
    // all-positive/all-negative subset bounds fit i64, even for all 32-bit keys.
    for bits in [8, 16, 32] {
        let unsigned_max = (1_i128 << bits) - 1;
        assert!(unsigned_max * (unsigned_max + 1) / 2 <= i128::from(i64::MAX));
        let negative_max = 1_i128 << (bits - 1);
        assert!(-negative_max * (negative_max + 1) / 2 >= i128::from(i64::MIN));
    }
    for ty in [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
    ] {
        let input = arrow::compute::cast(
            &Int64Array::from(vec![Some(0), Some(127), None, Some(127), Some(9)]),
            &ty,
        )?;
        let mut acc = with_avg_args(&ty, &DataType::Float64, |mut args| {
            args.is_distinct = true;
            Avg::new().accumulator(args)
        })?;
        acc.update_batch(&[input])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Float64(Some(136.0 / 3.0)));
    }
    Ok(())
}

#[test]
fn integer_avg_narrow_random_groups_filters_merge_orders_and_ipc() -> Result<()> {
    use arrow::array::RecordBatch;
    use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
    use std::io::Cursor;
    let mut seed = 0x5eed_u64;
    for ty in [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
    ] {
        for _ in 0..8 {
            let signed = matches!(ty, DataType::Int8 | DataType::Int16 | DataType::Int32);
            let bits = match ty {
                DataType::Int8 | DataType::UInt8 => 8,
                DataType::Int16 | DataType::UInt16 => 16,
                _ => 32,
            };
            let raw = (0..513)
                .map(|index| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if index % 13 == 0 {
                        None
                    } else {
                        let value = (seed >> 32) & ((1 << bits) - 1);
                        Some(if signed {
                            (value as i64) - (1_i64 << (bits - 1))
                        } else {
                            value as i64
                        })
                    }
                })
                .collect::<Vec<_>>();
            let filter = BooleanArray::from(
                (0..raw.len())
                    .map(|i| if i % 19 == 0 { None } else { Some(i % 7 != 0) })
                    .collect::<Vec<_>>(),
            );
            let values = arrow::compute::cast(&Int64Array::from(raw.clone()), &ty)?;
            let groups = (0..raw.len()).map(|i| i % 5).collect::<Vec<_>>();
            let mut expected = [(0_i128, 0_i64); 6];
            for (index, value) in raw.iter().enumerate() {
                if let Some(value) =
                    value.filter(|_| filter.is_valid(index) && filter.value(index))
                {
                    expected[groups[index]].0 += i128::from(value);
                    expected[groups[index]].1 += 1;
                }
            }
            let mut partials = vec![];
            for offset in (0..raw.len()).step_by(29) {
                let len = 29.min(raw.len() - offset);
                let mut acc = avg_groups_accumulator(&ty, &DataType::Float64)?;
                acc.update_batch(
                    &[values.slice(offset, len)],
                    &groups[offset..offset + len],
                    Some(&filter.slice(offset, len)),
                    6,
                )?;
                let schema =
                    Arc::new(Schema::new(avg_state_fields(&ty, &DataType::Float64)?));
                let batch =
                    RecordBatch::try_new(schema.clone(), acc.state(EmitTo::All)?)?;
                assert_eq!(batch.column(2).null_count(), 6);
                let mut bytes = vec![];
                {
                    let mut writer = StreamWriter::try_new(&mut bytes, &schema)?;
                    writer.write(&batch)?;
                    writer.finish()?;
                }
                partials.push(
                    StreamReader::try_new(Cursor::new(bytes), None)?
                        .next()
                        .unwrap()?,
                );
            }
            for _ in 0..3 {
                let mut final_acc = avg_groups_accumulator(&ty, &DataType::Float64)?;
                for partial in &partials {
                    final_acc.merge_batch(partial.columns(), &[0, 1, 2, 3, 4, 5], 6)?;
                }
                let actual = final_acc.evaluate(EmitTo::All)?;
                for (group, &(sum, count)) in expected.iter().enumerate() {
                    assert_eq!(
                        ScalarValue::try_from_array(actual.as_ref(), group)?,
                        ScalarValue::Float64(
                            (count != 0).then(|| sum as f64 / count as f64)
                        ),
                        "{ty} group {group}"
                    );
                }
                partials.reverse();
                partials.rotate_left(3);
            }
        }
    }
    Ok(())
}

#[test]
fn integer_avg_narrow_count_errors_and_promoted_state_ipc() -> Result<()> {
    use arrow::array::RecordBatch;
    use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
    use std::io::Cursor;
    let ty = DataType::UInt32;
    let mut acc = avg_accumulator(&ty, &DataType::Float64)?;
    acc.merge_batch(&narrow_state(i64::MAX, 0))?;
    assert!(
        acc.update_batch(&[Arc::new(UInt32Array::from(vec![0]))])
            .is_err()
    );
    assert_eq!(acc.evaluate()?, ScalarValue::Float64(Some(0.0)));
    assert!(
        avg_accumulator(&ty, &DataType::Float64)?
            .merge_batch(&narrow_state(-1, 0))
            .is_err()
    );
    let mut count_limit = avg_groups_accumulator(&ty, &DataType::Float64)?;
    count_limit.merge_batch(&narrow_state(i64::MAX, 0), &[0], 1)?;
    assert!(
        count_limit
            .merge_batch(&narrow_state(1, 0), &[0], 1)
            .is_err()
    );
    assert!(
        avg_groups_accumulator(&ty, &DataType::Float64)?
            .merge_batch(&narrow_state(-1, 0), &[0], 1)
            .is_err()
    );
    let mut groups = avg_groups_accumulator(&ty, &DataType::Float64)?;
    let count = (1_i64 << 31) + 1;
    let sum = i128::from(u32::MAX) * i128::from(count);
    groups.merge_batch(&narrow_state(count, sum), &[0], 1)?;
    let schema = Arc::new(Schema::new(avg_state_fields(&ty, &DataType::Float64)?));
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
    let mut scalar = avg_accumulator(&ty, &DataType::Float64)?;
    scalar.merge_batch(restored.columns())?;
    assert_eq!(
        scalar.evaluate()?,
        ScalarValue::Float64(Some(sum as f64 / count as f64))
    );
    Ok(())
}

#[test]
fn integer_avg_narrow_merge_promotes_without_replaying_committed_prefix() -> Result<()> {
    for (ty, value, count) in [
        (DataType::UInt32, i128::from(u32::MAX), 1_i64 << 31),
        (DataType::Int32, i128::from(i32::MIN), 1_i64 << 32),
    ] {
        let mut groups = avg_groups_accumulator(&ty, &DataType::Float64)?;
        groups.merge_batch(&narrow_state(count, value * i128::from(count)), &[0], 3)?;
        // Ignore the empty row's invalid group. Commit row 1, promote on row 2,
        // then process row 3 exactly once without replaying the prefix.
        let incoming: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![0, 1, 1, 1])),
            Arc::new(Int64Array::from(vec![i64::MAX, 7, value as i64, 11])),
            Arc::new(Int32Array::from(vec![None, None, None, None])),
        ];
        groups.merge_batch(&incoming, &[usize::MAX, 1, 0, 1], 3)?;
        let actual = groups.evaluate(EmitTo::All)?;
        assert_eq!(
            ScalarValue::try_from_array(actual.as_ref(), 0)?,
            ScalarValue::Float64(Some(value as f64))
        );
        assert_eq!(
            ScalarValue::try_from_array(actual.as_ref(), 1)?,
            ScalarValue::Float64(Some(9.0))
        );
        assert!(actual.is_null(2));
    }
    Ok(())
}

#[test]
fn integer_avg_narrow_singleton_outputs_match_existing_i128_kernel() -> Result<()> {
    for (ty, min, max) in [
        (DataType::Int8, i64::from(i8::MIN), i64::from(i8::MAX)),
        (DataType::Int16, i64::from(i16::MIN), i64::from(i16::MAX)),
        (DataType::Int32, i64::from(i32::MIN), i64::from(i32::MAX)),
        (DataType::UInt8, 0, i64::from(u8::MAX)),
        (DataType::UInt16, 0, i64::from(u16::MAX)),
        (DataType::UInt32, 0, i64::from(u32::MAX)),
    ] {
        let wide: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(min),
            Some(max),
            Some(0),
            None,
            Some(min),
            Some(max),
        ]));
        let narrow = arrow::compute::cast(wide.as_ref(), &ty)?;
        let mut reference = avg_groups_accumulator(&DataType::Int64, &DataType::Float64)?;
        let mut candidate = avg_groups_accumulator(&ty, &DataType::Float64)?;
        reference.update_batch(&[wide], &[0, 1, 2, 3, 4, 4], None, 5)?;
        candidate.update_batch(&[narrow], &[0, 1, 2, 3, 4, 4], None, 5)?;
        let reference = reference.evaluate(EmitTo::All)?;
        let candidate = candidate.evaluate(EmitTo::All)?;
        for group in 0..5 {
            assert_eq!(
                ScalarValue::try_from_array(candidate.as_ref(), group)?,
                ScalarValue::try_from_array(reference.as_ref(), group)?,
                "{ty} group {group}"
            );
        }
    }
    Ok(())
}
