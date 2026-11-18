// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::{Array, AsArray, BooleanArray};
use arrow::buffer::{BooleanBuffer, ScalarBuffer};
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

/// Longest constant that Arrow's own scalar fast path compares: its length
/// and bytes fit a view's low 64 bits, and Arrow reads only that word.
const ARROW_FAST_PATH_MAX_LEN: usize = 4;

/// Compare an inline constant of 5 to 12 bytes directly with view words,
/// including when the array also has external buffers for longer strings.
/// No payload is read.
///
/// Arrow 60.1 compares whole views only when neither side owns a data buffer,
/// and its scalar fast path covers constants of at most four bytes (and is
/// faster than this kernel there, so those keep Arrow's path). Columns decoded
/// from Parquet usually own a buffer, so a 5 to 12 byte constant such as
/// `d_day_name = 'Wednesday'` otherwise takes the per-row length and prefix
/// branches (about 3x slower per row on TPC-DS q59, leto-zca.34).
pub(super) fn compare_inline_scalar(
    op: Operator,
    left: &ColumnarValue,
    right: &ColumnarValue,
) -> Option<ColumnarValue> {
    if !matches!(op, Operator::Eq | Operator::NotEq) {
        return None;
    }
    let ((ColumnarValue::Array(array), ColumnarValue::Scalar(scalar))
    | (ColumnarValue::Scalar(scalar), ColumnarValue::Array(array))) = (left, right)
    else {
        return None;
    };
    let array = array.as_string_view_opt()?;
    let ScalarValue::Utf8View(Some(value)) = scalar else {
        return None;
    };
    if value.len() <= ARROW_FAST_PATH_MAX_LEN {
        return None;
    }
    let constant = inline_view(value)?;
    let negated = op == Operator::NotEq;
    // Two 64-bit compares per view vectorize better than one 128-bit compare.
    let views = array.views();
    let words = ScalarBuffer::<u64>::new(views.inner().clone(), 0, views.len() * 2);
    let (low, high) = (constant as u64, (constant >> 64) as u64);
    let (first, second) = if cfg!(target_endian = "little") {
        (low, high)
    } else {
        (high, low)
    };
    #[expect(
        clippy::needless_bitwise_bool,
        reason = "branch-free compare of both words; `&&` measured 11% slower"
    )]
    let bits = BooleanBuffer::collect_bool(views.len(), |i| {
        ((words[2 * i] == first) & (words[2 * i + 1] == second)) ^ negated
    });
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
            Some("a\0b\0c"),
            Some("a\0b\0d"),
            Some("SHIPPED"),
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
                Some("a\0b\0c"),
                Some("SHIPPED"),
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

    /// The view-word path serves 5 to 12 byte constants on arrays that own
    /// data buffers, which Arrow's own fast paths decline; shorter constants
    /// keep Arrow's faster low-word path.
    #[test]
    fn buffered_views_take_the_inline_path() -> Result<()> {
        let days = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ];
        let mut rows: Vec<Option<&str>> =
            (0..1000).map(|i| Some(days[(i * 5 + i / 7) % 7])).collect();
        rows[17] = None;
        rows.push(Some("a value longer than twelve bytes"));
        let values = StringViewArray::from(rows).slice(3, 990);
        assert!(!values.data_buffers().is_empty());
        let array = ColumnarValue::Array(Arc::new(values.clone()) as ArrayRef);
        for day in days.iter().copied().chain(["abcdefghijkl", "abcde"]) {
            let scalar = ColumnarValue::Scalar(ScalarValue::Utf8View(Some(day.into())));
            for op in [Operator::Eq, Operator::NotEq] {
                for (left, right) in [(&array, &scalar), (&scalar, &array)] {
                    let actual = compare_inline_scalar(op, left, right)
                        .expect("an inline constant takes the view-word path")
                        .into_array(values.len())?;
                    let expected =
                        apply_cmp(op, left, right)?.into_array(values.len())?;
                    assert_eq!(actual.as_ref(), expected.as_ref());
                }
            }
        }
        // Constants Arrow's own fast path covers (at most four bytes), longer
        // constants, NULL constants and other operators keep Arrow's path.
        for short in ["", "x", "MAIL"] {
            let short = ColumnarValue::Scalar(ScalarValue::Utf8View(Some(short.into())));
            assert!(compare_inline_scalar(Operator::Eq, &array, &short).is_none());
        }
        let long =
            ColumnarValue::Scalar(ScalarValue::Utf8View(Some("abcdefghijklm".into())));
        let null = ColumnarValue::Scalar(ScalarValue::Utf8View(None));
        assert!(compare_inline_scalar(Operator::Eq, &array, &long).is_none());
        assert!(compare_inline_scalar(Operator::Eq, &array, &null).is_none());
        let short = ColumnarValue::Scalar(ScalarValue::Utf8View(Some("Friday".into())));
        assert!(compare_inline_scalar(Operator::Lt, &array, &short).is_none());
        Ok(())
    }
}
