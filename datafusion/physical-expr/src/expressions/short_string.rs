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

use arrow::array::{Array, AsArray, BooleanArray};
use arrow::buffer::BooleanBuffer;
use datafusion_common::ScalarValue;
use datafusion_expr::{ColumnarValue, Operator};
use std::sync::Arc;

/// Arrow stores strings of at most twelve bytes inline, with zero padding.
/// Including the length makes this representation injective, even for NULs.
pub(super) fn inline_view(value: &str) -> Option<u128> {
    if value.len() > 12 {
        return None;
    }
    let mut bytes = [0; 16];
    bytes[..4].copy_from_slice(&(value.len() as u32).to_le_bytes());
    bytes[4..4 + value.len()].copy_from_slice(value.as_bytes());
    Some(u128::from_le_bytes(bytes))
}

/// Compare an inline constant directly with view words, including when the
/// array also has external buffers for longer strings. No payload is read.
pub(super) fn compare_inline_scalar(
    op: Operator,
    left: &ColumnarValue,
    right: &ColumnarValue,
) -> Option<ColumnarValue> {
    if !matches!(op, Operator::Eq | Operator::NotEq) {
        return None;
    }
    let (array, scalar) = match (left, right) {
        (ColumnarValue::Array(array), ColumnarValue::Scalar(scalar))
        | (ColumnarValue::Scalar(scalar), ColumnarValue::Array(array)) => (array, scalar),
        _ => return None,
    };
    let array = array.as_string_view_opt()?;
    let ScalarValue::Utf8View(Some(value)) = scalar else {
        return None;
    };
    let constant = inline_view(value)?;
    let negated = op == Operator::NotEq;
    let views = array.views();
    let bits =
        BooleanBuffer::collect_bool(views.len(), |i| (views[i] == constant) ^ negated);
    Some(ColumnarValue::Array(Arc::new(BooleanArray::new(
        bits,
        array.nulls().cloned(),
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhysicalExpr;
    use crate::expressions::{BinaryExpr, Column, lit};
    use arrow::array::{ArrayRef, StringViewArray};
    use arrow::record_batch::RecordBatch;
    use datafusion_common::Result;
    use datafusion_physical_expr_common::datum::apply_cmp;

    #[test]
    fn view_constant_comparisons_match_arrow() -> Result<()> {
        let values = StringViewArray::from(vec![
            Some(""),
            Some("MAIL"),
            Some("SHIP"),
            Some("é"),
            None,
            Some("abcdefghijkl"),
            Some("abcdefghijklm"),
            Some("MAIL-longer-than-inline"),
            Some("a\0b"),
            Some("😀😀😀"),
        ]);
        // Include nonzero view/null offsets and backing buffers for long strings.
        for values in [values.clone(), values.slice(1, values.len() - 1)] {
            let batch = RecordBatch::try_from_iter([(
                "v",
                Arc::new(values.clone()) as ArrayRef,
            )])?;
            for text in [
                Some(""),
                Some("MAIL"),
                Some("SHIP"),
                Some("é"),
                Some("abcdefghijkl"),
                Some("abcdefghijklm"),
                Some("a\0b"),
                Some("😀😀😀"),
                None,
            ] {
                let literal = ScalarValue::Utf8View(text.map(str::to_owned));
                for op in [Operator::Eq, Operator::NotEq] {
                    for reversed in [false, true] {
                        let col = Arc::new(Column::new("v", 0)) as Arc<dyn PhysicalExpr>;
                        let scalar = lit(literal.clone());
                        let (left, right) = if reversed {
                            (scalar, col)
                        } else {
                            (col, scalar)
                        };
                        let expected = apply_cmp(
                            op,
                            &left.evaluate(&batch)?,
                            &right.evaluate(&batch)?,
                        )?
                        .into_array(values.len())?;
                        let actual = BinaryExpr::new(left, op, right)
                            .evaluate(&batch)?
                            .into_array(values.len())?;
                        assert_eq!(actual.as_ref(), expected.as_ref());
                    }
                }
            }
        }
        Ok(())
    }
}
