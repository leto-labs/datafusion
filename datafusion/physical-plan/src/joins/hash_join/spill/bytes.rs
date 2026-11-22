// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Byte limits for the flat layouts supported by spill replay and gathers.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, RecordBatchOptions};
use arrow::datatypes::{DataType, Schema};
use datafusion_common::Result;

pub(in crate::joins::hash_join) const BATCH_BYTES: usize = 8 * 1024 * 1024;

/// Explicit layout allow-list. A fixed-width primitive has an Arrow-defined
/// slot size. The other entries below have no child data or dictionaries.
pub(super) fn supported_schema(schema: &Schema) -> bool {
    schema.fields().iter().all(|field| {
        matches!(
            field.data_type(),
            DataType::Null
                | DataType::Boolean
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Date32
                | DataType::Date64
                | DataType::Time32(_)
                | DataType::Time64(_)
                | DataType::Timestamp(..)
                | DataType::Duration(_)
                | DataType::Interval(_)
                | DataType::Decimal32(..)
                | DataType::Decimal64(..)
                | DataType::Decimal128(..)
                | DataType::Decimal256(..)
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::Utf8View
                | DataType::BinaryView
        )
    })
}

/// Byte contribution of one selected slot, before validity and padding.
/// A null index still takes a slot in a newly allocated column.
enum Layout<'a> {
    Fixed(usize),
    Bytes32(&'a [i32]),
    Bytes64(&'a [i64]),
    View(&'a [u128]),
}

pub(in crate::joins::hash_join) struct RowBytes<'a> {
    layout: Layout<'a>,
    array: &'a dyn Array,
}

impl<'a> RowBytes<'a> {
    pub(in crate::joins::hash_join) fn new(array: &'a dyn Array) -> Self {
        let layout = match array.data_type() {
            DataType::Utf8 => Layout::Bytes32(array.as_string::<i32>().value_offsets()),
            DataType::Binary => Layout::Bytes32(array.as_binary::<i32>().value_offsets()),
            DataType::LargeUtf8 => {
                Layout::Bytes64(array.as_string::<i64>().value_offsets())
            }
            DataType::LargeBinary => {
                Layout::Bytes64(array.as_binary::<i64>().value_offsets())
            }
            DataType::Utf8View => Layout::View(array.as_string_view().views()),
            DataType::BinaryView => Layout::View(array.as_binary_view().views()),
            DataType::Null => Layout::Fixed(0),
            DataType::Boolean => Layout::Fixed(1),
            dt => Layout::Fixed(dt.primitive_width().expect("byte-limited join layout")),
        };
        Self { layout, array }
    }

    pub(in crate::joins::hash_join) fn slot(&self) -> usize {
        // One byte per row for each validity bitmap, rounded up.
        1 + match self.layout {
            Layout::Fixed(width) => width,
            Layout::Bytes32(_) => size_of::<i32>(),
            Layout::Bytes64(_) => size_of::<i64>(),
            Layout::View(_) => size_of::<u128>(),
        }
    }

    pub(in crate::joins::hash_join) fn payload(&self, row: Option<usize>) -> usize {
        let Some(row) = row else {
            return 0;
        };
        match self.layout {
            // Arrow take preserves the raw view for a valid gather index
            // even if the selected array slot is null. The view compactor
            // copies its out-of-line bytes regardless of that validity bit.
            Layout::View(views) => {
                let len = views[row] as u32 as usize;
                if len > 12 { len } else { 0 }
            }
            Layout::Bytes32(offsets) if !self.array.is_null(row) => {
                (offsets[row + 1] - offsets[row]) as usize
            }
            Layout::Bytes64(offsets) if !self.array.is_null(row) => {
                (offsets[row + 1] - offsets[row]) as usize
            }
            _ => 0,
        }
    }

    pub(in crate::joins::hash_join) fn variable(&self) -> bool {
        !matches!(self.layout, Layout::Fixed(_))
    }
}

/// Size the prefix before Arrow materializes it. A single indivisible row
/// may exceed the target. Reserve room for array metadata, trailing offsets
/// and minimum buffer capacities.
pub(in crate::joins::hash_join) fn prefix(
    rows: usize,
    columns: usize,
    slot_bytes: usize,
    payload: impl Fn(usize) -> usize,
) -> usize {
    let mut bytes = columns.saturating_mul(256);
    for row in 0..rows {
        bytes = bytes
            .saturating_add(slot_bytes)
            .saturating_add(payload(row));
        if bytes > BATCH_BYTES {
            return row.max(1);
        }
    }
    rows
}

/// Compact view buffers after a bounded gather or before measuring an IPC
/// message. Arrow's take shares the input's variadic buffers.
pub(in crate::joins::hash_join) fn compact_views(
    batch: &RecordBatch,
) -> Result<RecordBatch> {
    if !batch.columns().iter().any(|array| {
        matches!(array.data_type(), DataType::Utf8View | DataType::BinaryView)
    }) {
        return Ok(batch.clone());
    }
    let columns = batch
        .columns()
        .iter()
        .map(|array| match array.data_type() {
            DataType::Utf8View => Arc::new(array.as_string_view().gc()) as ArrayRef,
            DataType::BinaryView => Arc::new(array.as_binary_view().gc()) as ArrayRef,
            _ => Arc::clone(array),
        })
        .collect::<Vec<_>>();
    Ok(RecordBatch::try_new_with_options(
        batch.schema(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )?)
}

/// Covers the decoded batch and a flat copy when replay skips matched rows
/// or routes the retained part again, plus row numbers, hashes, inline keys,
/// partition IDs and validity scratch. Direct key evaluation borrows buffers.
pub(super) fn replay_need(batch: &RecordBatch, compressed: bool) -> Result<usize> {
    let decoded = super::footprint::decoded_bound(batch, compressed)?;
    Ok(decoded
        .saturating_mul(2)
        .saturating_add(batch.num_rows().saturating_mul(64)))
}
