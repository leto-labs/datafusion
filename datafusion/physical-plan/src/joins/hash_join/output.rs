// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Materialize one bounded piece at a time, using the actual selected rows.

use arrow::array::{Array, ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow::compute::{filter as filter_array, take_record_batch};
use arrow::datatypes::SchemaRef;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{JoinSide, JoinType, Result};

use super::spill::bytes::{BATCH_BYTES, RowBytes, compact_views, prefix};
use crate::joins::utils::{
    ColumnIndex, JoinFilter, apply_join_filter_to_indices, build_batch_from_indices,
};

pub(super) struct Gather {
    pub(super) build: RecordBatch,
    pub(super) probe: RecordBatch,
    pub(super) left: UInt64Array,
    pub(super) right: UInt32Array,
    pub(super) side: JoinSide,
    pub(super) mark: Option<ArrayRef>,
}

impl Gather {
    fn rows(&self, join_type: JoinType) -> usize {
        match join_type {
            JoinType::RightSemi | JoinType::RightAnti => self.right.len(),
            _ => self.left.len(),
        }
    }

    fn prefix(&self, columns: &[ColumnIndex], start: usize, max_rows: usize) -> usize {
        let mut slots: usize = 0;
        let variable: Vec<_> = columns
            .iter()
            .filter_map(|column| {
                let build = column.side == self.side;
                let array = match column.side {
                    JoinSide::None => {
                        slots = slots.saturating_add(2);
                        return None;
                    }
                    _ if build => self.build.column(column.index),
                    _ => self.probe.column(column.index),
                };
                let width = RowBytes::new(array.as_ref());
                slots = slots.saturating_add(width.slot());
                width.variable().then_some((width, build))
            })
            .collect();
        prefix(max_rows, columns.len(), slots, |offset| {
            let pos = start + offset;
            variable.iter().fold(0usize, |total, (width, build)| {
                let row = if *build {
                    self.left
                        .is_valid(pos)
                        .then(|| self.left.value(pos) as usize)
                } else {
                    self.right
                        .is_valid(pos)
                        .then(|| self.right.value(pos) as usize)
                };
                total.saturating_add(width.payload(row))
            })
        })
    }

    fn indices(&self, start: usize, len: usize) -> (UInt64Array, UInt32Array) {
        let left_start = start.min(self.left.len());
        (
            self.left
                .slice(left_start, len.min(self.left.len() - left_start)),
            self.right.slice(start, len),
        )
    }
}

/// A multi-piece gather or a batch forwarded by a spill run. The stream
/// drains it before advancing to another probe batch or spill partition.
pub(super) enum OutputRows {
    Gather(Box<Gather>),
    Batch(RecordBatch),
}

pub(super) struct OutputPieces {
    rows: OutputRows,
    offset: usize,
}

impl OutputPieces {
    pub(super) fn new(rows: OutputRows) -> Self {
        Self { rows, offset: 0 }
    }

    pub(super) fn next(
        &mut self,
        schema: &SchemaRef,
        columns: &[ColumnIndex],
        join_type: JoinType,
        row_limit: usize,
    ) -> Result<Option<RecordBatch>> {
        let start = self.offset;
        let batch = match &self.rows {
            OutputRows::Gather(gather) => {
                let remaining = gather.rows(join_type) - start;
                if remaining == 0 {
                    return Ok(None);
                }
                let len = gather.prefix(columns, start, remaining.min(row_limit));
                let (left, right) = gather.indices(start, len);
                let mark = gather.mark.as_ref().map(|mark| mark.slice(start, len));
                build_batch_from_indices(
                    schema,
                    &gather.build,
                    &gather.probe,
                    &left,
                    &right,
                    columns,
                    gather.side,
                    join_type,
                    mark.as_ref(),
                )?
            }
            OutputRows::Batch(batch) => {
                let remaining = batch.num_rows() - start;
                if remaining == 0 {
                    return Ok(None);
                }
                let widths: Vec<_> = batch
                    .columns()
                    .iter()
                    .map(|array| RowBytes::new(array.as_ref()))
                    .collect();
                let slots = widths.iter().map(RowBytes::slot).sum();
                let len =
                    prefix(remaining.min(row_limit), widths.len(), slots, |offset| {
                        widths.iter().fold(0usize, |bytes, width| {
                            bytes.saturating_add(width.payload(Some(start + offset)))
                        })
                    });
                if start == 0
                    && len == batch.num_rows()
                    && batch.get_array_memory_size() <= BATCH_BYTES
                {
                    self.offset = batch.num_rows();
                    return Ok(Some(batch.clone()));
                }
                let indices =
                    UInt32Array::from_iter_values(start as u32..(start + len) as u32);
                take_record_batch(batch, &indices)?
            }
        };
        self.offset += batch.num_rows();
        Ok(Some(compact_views(&batch)?))
    }
}

/// A residual may refer to a wide column absent from the output projection.
/// Filter candidates in the same order, before visited bits and index
/// alignment are updated by the join.
pub(super) fn filter(
    build: &RecordBatch,
    probe: &RecordBatch,
    left: UInt64Array,
    right: UInt32Array,
    filter: &JoinFilter,
    join_type: JoinType,
) -> Result<(UInt64Array, UInt32Array)> {
    let gather = Gather {
        build: build.clone(),
        probe: probe.clone(),
        left,
        right,
        side: JoinSide::Left,
        mark: None,
    };
    let count = gather.left.len();
    if count == 0 {
        return Ok((gather.left, gather.right));
    }
    let has_views = filter.schema().fields().iter().any(|field| {
        matches!(
            field.data_type(),
            arrow::datatypes::DataType::Utf8View | arrow::datatypes::DataType::BinaryView
        )
    });
    let first = gather.prefix(filter.column_indices(), 0, count);
    if first == count && !has_views {
        return apply_join_filter_to_indices(
            build,
            probe,
            gather.left,
            gather.right,
            filter,
            JoinSide::Left,
            None,
            join_type,
        );
    }
    let mut left = Vec::with_capacity(count);
    let mut right = Vec::with_capacity(count);
    let mut start = 0;
    let mut len = first;
    while start < count {
        let (build_indices, probe_indices) = gather.indices(start, len);
        let batch = build_batch_from_indices(
            filter.schema(),
            build,
            probe,
            &build_indices,
            &probe_indices,
            filter.column_indices(),
            JoinSide::Left,
            join_type,
            None,
        )?;
        let batch = compact_views(&batch)?;
        let values = filter.expression().evaluate(&batch)?.into_array(len)?;
        let mask = as_boolean_array(&values)?;
        let matched_build = filter_array(&build_indices, mask)?;
        let matched_probe = filter_array(&probe_indices, mask)?;
        let matched_build = matched_build
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let matched_probe = matched_probe
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        left.extend_from_slice(matched_build.values());
        right.extend_from_slice(matched_probe.values());
        start += len;
        len = gather.prefix(filter.column_indices(), start, count - start);
    }
    Ok((left.into(), right.into()))
}
