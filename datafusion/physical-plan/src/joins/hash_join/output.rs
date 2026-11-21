// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Byte-aware materialization of hash-join results. Size the gather before
//! calling Arrow take: one source row may appear many times in the indices.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, UInt32Array, UInt64Array};
use arrow::datatypes::{ArrowNativeType, DataType, SchemaRef};
use datafusion_common::{JoinSide, JoinType, Result};

use crate::coalesce::{LimitedBatchCoalescer, PushBatchStatus};
use crate::joins::utils::{ColumnIndex, build_batch_from_indices};

/// Target for newly materialized batch data. One wider row may pass alone.
const BATCH_BYTES: usize = 8 * 1024 * 1024;

/// Reserve room in the estimate for buffer headers, final offsets and
/// alignment. Each row is also charged for validity in its columns.
fn overhead(columns: usize) -> usize {
    columns.saturating_mul(256)
}

fn rows_for(width: usize, columns: usize, row_limit: usize) -> usize {
    let bytes = BATCH_BYTES.saturating_sub(overhead(columns));
    row_limit.min((bytes / width.max(1)).max(1))
}

fn max_span<T: ArrowNativeType>(offsets: &[T]) -> usize {
    offsets
        .windows(2)
        .map(|pair| pair[1].as_usize() - pair[0].as_usize())
        .max()
        .unwrap_or(0)
}

fn view_width(views: &[u128]) -> usize {
    let length = views.iter().map(|view| *view as u32).max().unwrap_or(0);
    16 + if length <= 12 { 0 } else { length as usize }
}

/// Upper estimate per selected row, including a copy of a referenced view
/// value if the coalescer compacts it. Dictionary values and list-view
/// children are shared by take. Charge shared children conservatively for
/// the fallback layouts, where materialization may also copy child values.
pub(super) fn row_width(array: &dyn Array) -> usize {
    let width = match array.data_type() {
        DataType::Utf8 => 4 + max_span(array.as_string::<i32>().value_offsets()),
        DataType::LargeUtf8 => 8 + max_span(array.as_string::<i64>().value_offsets()),
        DataType::Binary => 4 + max_span(array.as_binary::<i32>().value_offsets()),
        DataType::LargeBinary => 8 + max_span(array.as_binary::<i64>().value_offsets()),
        DataType::Utf8View => view_width(array.as_string_view().views()),
        DataType::BinaryView => view_width(array.as_binary_view().views()),
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            4usize.saturating_add(
                max_span(list.value_offsets())
                    .saturating_mul(row_width(list.values().as_ref())),
            )
        }
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            8usize.saturating_add(
                max_span(list.value_offsets())
                    .saturating_mul(row_width(list.values().as_ref())),
            )
        }
        DataType::Map(..) => {
            let map = array.as_map();
            4usize.saturating_add(
                max_span(map.value_offsets()).saturating_mul(row_width(map.entries())),
            )
        }
        DataType::FixedSizeList(_, width) => (*width as usize)
            .saturating_mul(row_width(array.as_fixed_size_list().values().as_ref())),
        DataType::Struct(_) => batch_width(array.as_struct().columns()),
        DataType::Dictionary(key, _) => key.primitive_width().unwrap_or(0),
        DataType::FixedSizeBinary(width) => *width as usize,
        DataType::Null => 0,
        DataType::Boolean => 1,
        dt => dt
            .primitive_width()
            .unwrap_or_else(|| array.get_array_memory_size()),
    };
    width.saturating_add(1)
}

fn batch_width(columns: &[ArrayRef]) -> usize {
    columns.iter().fold(0, |width, array| {
        width.saturating_add(row_width(array.as_ref()))
    })
}

pub(super) fn side_width(
    columns: &[ColumnIndex],
    side: JoinSide,
    column_width: impl Fn(usize) -> usize,
) -> usize {
    columns
        .iter()
        .filter(|c| c.side == side)
        .fold(0, |width, c| width.saturating_add(column_width(c.index)))
}

pub(super) fn filter_rows(width: usize, columns: usize, row_limit: usize) -> usize {
    rows_for(width, columns, row_limit)
}

/// Data already produced by the join's state machine, drained one byte-
/// and row-limited piece per poll before the join advances to more input.
enum Pending {
    Gather {
        build: RecordBatch,
        probe: RecordBatch,
        left: UInt64Array,
        right: UInt32Array,
        side: JoinSide,
    },
    Batch(RecordBatch),
}

struct Remainder {
    data: Pending,
    offset: usize,
    width: usize,
}

pub(super) struct OutputBatches {
    schema: SchemaRef,
    row_limit: usize,
    buffer: LimitedBatchCoalescer,
    buffered_bytes: usize,
    pending: Option<Remainder>,
}

impl OutputBatches {
    pub(super) fn new(schema: SchemaRef, row_limit: usize, fetch: Option<usize>) -> Self {
        Self {
            buffer: LimitedBatchCoalescer::new(Arc::clone(&schema), row_limit, fetch),
            schema,
            row_limit,
            buffered_bytes: 0,
            pending: None,
        }
    }

    pub(super) fn next_completed_batch(&mut self) -> Option<RecordBatch> {
        self.buffer.next_completed_batch()
    }

    pub(super) fn is_finished(&self) -> bool {
        self.buffer.is_finished()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub(super) fn finish(&mut self) -> Result<()> {
        self.pending = None;
        self.buffer.finish()
    }

    fn push(&mut self, batch: RecordBatch, width: usize) -> Result<()> {
        if batch.num_rows() == 0 {
            if self.buffer.push_batch(batch)? == PushBatchStatus::LimitReached {
                self.finish()?;
            }
            return Ok(());
        }
        let bytes = width
            .saturating_mul(batch.num_rows())
            .saturating_add(overhead(self.schema.fields().len()));
        let status = if bytes >= BATCH_BYTES / 2 {
            self.buffered_bytes = 0;
            self.buffer.push_uncoalesced(batch)?
        } else {
            if self.buffered_bytes.saturating_add(bytes) > BATCH_BYTES {
                self.buffer.flush()?;
                self.buffered_bytes = 0;
            }
            let status = self.buffer.push_batch(batch)?;
            if self.buffer.has_buffered_rows() {
                self.buffered_bytes += bytes;
            } else {
                self.buffered_bytes = 0;
            }
            status
        };
        if status == PushBatchStatus::LimitReached {
            return self.finish();
        }
        Ok(())
    }

    pub(super) fn push_batch(&mut self, batch: RecordBatch) -> Result<()> {
        let width = batch_width(batch.columns());
        let len = batch.num_rows().min(self.max_rows(width));
        if len == batch.num_rows() {
            return self.push(batch, width);
        }
        let head = batch.slice(0, len);
        self.pending = Some(Remainder {
            data: Pending::Batch(batch),
            offset: len,
            width,
        });
        self.push(head, width)
    }

    pub(super) fn max_rows(&self, width: usize) -> usize {
        rows_for(width, self.schema.fields().len(), self.row_limit)
    }

    #[expect(clippy::too_many_arguments)]
    pub(super) fn push_indices(
        &mut self,
        build: &RecordBatch,
        probe: &RecordBatch,
        left: UInt64Array,
        right: UInt32Array,
        columns: &[ColumnIndex],
        side: JoinSide,
        join_type: JoinType,
        mark: Option<&ArrayRef>,
        width: usize,
    ) -> Result<()> {
        let rows = index_rows(&left, &right, join_type);
        let len = rows.min(self.max_rows(width));
        let batch = if len == rows {
            build_batch_from_indices(
                &self.schema,
                build,
                probe,
                &left,
                &right,
                columns,
                side,
                join_type,
                mark,
            )?
        } else {
            // A custom mark is only supplied by final build-row emission,
            // where the indices are already limited by this same width.
            debug_assert!(mark.is_none());
            let batch = build_batch_from_indices(
                &self.schema,
                build,
                probe,
                &left.slice(0, len.min(left.len())),
                &right.slice(0, len),
                columns,
                side,
                join_type,
                None,
            )?;
            self.pending = Some(Remainder {
                data: Pending::Gather {
                    build: build.clone(),
                    probe: probe.clone(),
                    left,
                    right,
                    side,
                },
                offset: len,
                width,
            });
            batch
        };
        self.push(batch, width)
    }

    pub(super) fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Returns true if a piece of pending output was processed.
    pub(super) fn drain(
        &mut self,
        columns: &[ColumnIndex],
        join_type: JoinType,
    ) -> Result<bool> {
        let Some(mut pending) = self.pending.take() else {
            return Ok(false);
        };
        let start = pending.offset;
        let max_rows = self.max_rows(pending.width);
        let (batch, rows) = match &pending.data {
            Pending::Batch(batch) => {
                let len = (batch.num_rows() - start).min(max_rows);
                (batch.slice(start, len), batch.num_rows())
            }
            Pending::Gather {
                build,
                probe,
                left,
                right,
                side,
            } => {
                let rows = index_rows(left, right, join_type);
                let len = (rows - start).min(max_rows);
                let left_start = start.min(left.len());
                let batch = build_batch_from_indices(
                    &self.schema,
                    build,
                    probe,
                    &left.slice(left_start, len.min(left.len() - left_start)),
                    &right.slice(start, len),
                    columns,
                    *side,
                    join_type,
                    None,
                )?;
                (batch, rows)
            }
        };
        pending.offset += batch.num_rows();
        let width = pending.width;
        if pending.offset < rows {
            self.pending = Some(pending);
        }
        self.push(batch, width)?;
        Ok(true)
    }
}

fn index_rows(left: &UInt64Array, right: &UInt32Array, join_type: JoinType) -> usize {
    match join_type {
        // Their adjusted build indices are ignored by the gather.
        JoinType::RightSemi | JoinType::RightAnti => right.len(),
        _ => left.len(),
    }
}
