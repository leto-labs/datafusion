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
use arrow::array::BooleanBufferBuilder;
use arrow::buffer::{BooleanBuffer, NullBuffer};

/// Expand the selected RHS into packed output buffers. The selection has no
/// nulls; unselected positions have the non-null value `fill_value`. Iterate
/// selected positions directly, avoiding a temporary array for every run.
pub(super) fn pre_selection_scatter(
    mask: &BooleanArray,
    right_result: Option<&BooleanArray>,
    fill_value: bool,
) -> Result<ColumnarValue> {
    let Some(right) = right_result else {
        let valid = !mask.values();
        let values = if fill_value {
            valid.clone()
        } else {
            BooleanBuffer::new_unset(mask.len())
        };
        return Ok(ColumnarValue::Array(Arc::new(BooleanArray::new(
            values,
            Some(NullBuffer::new(valid)),
        ))));
    };
    debug_assert_eq!(right.len(), mask.true_count());
    let mut values = BooleanBufferBuilder::new(mask.len());
    values.append_n(mask.len(), fill_value);
    let nulls = if right.null_count() == 0 {
        for (output, value) in mask.values().set_indices().zip(right.values().iter()) {
            values.set_bit(output, value);
        }
        None
    } else {
        let mut valid = BooleanBufferBuilder::new(mask.len());
        valid.append_n(mask.len(), true);
        for ((input, output), value) in mask
            .values()
            .set_indices()
            .enumerate()
            .zip(right.values().iter())
        {
            values.set_bit(output, value);
            valid.set_bit(output, right.is_valid(input));
        }
        Some(NullBuffer::new(valid.finish()))
    };
    Ok(ColumnarValue::Array(Arc::new(BooleanArray::new(
        values.finish(),
        nulls,
    ))))
}
