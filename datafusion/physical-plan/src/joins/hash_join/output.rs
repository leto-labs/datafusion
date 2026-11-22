// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Materialize one bounded piece at a time, using the actual selected rows.

use arrow::array::{Array, ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow::compute::{concat_batches, filter as filter_array, take_record_batch};
use arrow::datatypes::SchemaRef;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{JoinSide, JoinType, Result};

use super::spill::bytes::{BATCH_BYTES, RowBytes, compact_views, prefix};
use crate::coalesce::{LimitedBatchCoalescer, PushBatchStatus};
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

/// Coalesce compact spilled pieces within both budgets. Using Arrow concat
/// sizes the flat builders to the final row count and selected bytes once.
#[derive(Default)]
pub(super) struct SpillCoalescer {
    batches: Vec<RecordBatch>,
    rows: usize,
    bytes: usize,
}

impl SpillCoalescer {
    pub(super) fn remaining_rows(
        &self,
        output: &LimitedBatchCoalescer,
        batch_size: usize,
    ) -> usize {
        batch_size
            .min(output.remaining_fetch())
            .saturating_sub(self.rows)
    }

    pub(super) fn push(
        &mut self,
        batch: RecordBatch,
        output: &mut LimitedBatchCoalescer,
        batch_size: usize,
    ) -> Result<PushBatchStatus> {
        let rows = batch.num_rows();
        // Concat borrows view data blocks and sizes flat data/offset buffers
        // from the selected rows. Allow for a validity bitmap added to an
        // all-valid piece and up to three 64-byte alignment increments
        // per column (values, offsets and validity).
        let bytes = batch.get_array_memory_size().saturating_add(
            batch
                .num_columns()
                .saturating_mul(rows.div_ceil(8) + 3 * 64),
        );
        if self.bytes.saturating_add(bytes) > BATCH_BYTES {
            let status = self.flush(output)?;
            debug_assert_eq!(status, PushBatchStatus::Continue);
        }
        self.rows += rows;
        self.bytes += bytes;
        self.batches.push(batch);
        if self.bytes >= BATCH_BYTES || self.remaining_rows(output, batch_size) == 0 {
            return self.flush(output);
        }
        Ok(PushBatchStatus::Continue)
    }

    pub(super) fn flush(
        &mut self,
        output: &mut LimitedBatchCoalescer,
    ) -> Result<PushBatchStatus> {
        let batch = match self.batches.as_slice() {
            [] => return Ok(PushBatchStatus::Continue),
            [_] => self.batches.pop().unwrap(),
            _ => {
                let batch = concat_batches(&output.schema(), &self.batches)?;
                self.batches.clear();
                batch
            }
        };
        self.rows = 0;
        self.bytes = 0;
        output.push_uncoalesced(batch)
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
