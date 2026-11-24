// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use crate::spill::get_record_batch_memory_size;
use arrow::array::{Array, ArrayRef};
use arrow::compute::interleave;
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use datafusion_common::utils::memory::RecordBatchMemoryTracker;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::memory_pool::MemoryReservation;
use log::warn;
use std::sync::Arc;

/// The memory that input batches kept only for pending output rows may
/// take before [`BatchBuilder::retains_more_than_heads`] asks for an early
/// batch, when the live heads and the initial reservation are smaller.
pub const RETAINED_FLOOR_BYTES: usize = 1 << 20;

#[derive(Debug, Copy, Clone)]
struct BatchCursor {
    /// The index into BatchBuilder::batches
    batch_idx: usize,
    /// The row index within the given batch
    row_idx: usize,
}

impl BatchCursor {
    /// A cursor whose batch has been released. `push_row` must not be called
    /// for the stream until `push_batch` installs a new cursor.
    const RELEASED: Self = Self {
        batch_idx: usize::MAX,
        row_idx: 0,
    };
}

/// Provides an API to incrementally build a [`RecordBatch`] from partitioned [`RecordBatch`]
#[derive(Debug)]
pub struct BatchBuilder {
    /// The schema of the RecordBatches yielded by this stream
    schema: SchemaRef,

    /// Maintain a list of [`RecordBatch`] and their corresponding stream
    batches: Vec<(usize, RecordBatch)>,

    /// Accounts for memory used by buffered batches.
    ///
    /// May include pre-reserved bytes (from `sort_spill_reservation_bytes`)
    /// that were transferred via [`MemoryReservation::take()`] to prevent
    /// starvation when concurrent sort partitions compete for pool memory.
    reservation: MemoryReservation,

    /// Tracks the actual memory used by buffered batches (not including
    /// pre-reserved bytes). This allows [`Self::push_batch`] to skip pool
    /// allocation requests when the pre-reserved bytes cover the batch.
    ///
    /// It is the memory of the distinct buffers in `batches`, as `retained`
    /// counts them: zero-copy slices of one batch hold its buffers once.
    /// Always equal to `retained.memory_usage()`.
    batches_mem_used: usize,

    /// Counts each buffer of `batches` once, for as long as a batch that
    /// references it is retained
    retained: RecordBatchMemoryTracker,

    /// The initial reservation size at construction time. When the reservation
    /// is pre-loaded with `sort_spill_reservation_bytes` (via `take()`), this
    /// records that amount so we never shrink below it, maintaining the
    /// anti-starvation guarantee throughout the merge.
    initial_reservation: usize,

    /// The current [`BatchCursor`] for each stream
    cursors: Vec<BatchCursor>,

    /// The memory size of each stream's cursor batch (its head) on its own,
    /// or 0 once that batch is released
    head_sizes: Vec<usize>,

    /// The sum of `head_sizes`: the part of `batches_mem_used` that the
    /// merge needs whatever rows are pending. Heads of different streams
    /// that share a buffer count it once each here, so this can exceed
    /// `batches_mem_used`.
    heads_mem_used: usize,

    /// The accumulated stream indexes from which to pull rows
    /// Consists of a tuple of `(batch_idx, row_idx)`
    indices: Vec<(usize, usize)>,
}

impl BatchBuilder {
    /// Create a new [`BatchBuilder`] with the provided `stream_count` and `batch_size`
    pub fn new(
        schema: SchemaRef,
        stream_count: usize,
        batch_size: usize,
        reservation: MemoryReservation,
    ) -> Self {
        let initial_reservation = reservation.size();
        Self {
            schema,
            batches: Vec::with_capacity(stream_count * 2),
            cursors: vec![BatchCursor::RELEASED; stream_count],
            head_sizes: vec![0; stream_count],
            heads_mem_used: 0,
            indices: Vec::with_capacity(batch_size),
            reservation,
            batches_mem_used: 0,
            retained: RecordBatchMemoryTracker::new(),
            initial_reservation,
        }
    }

    /// Append a new batch in `stream_idx`, returning its memory size
    ///
    /// [`Self::admit`] followed by [`Self::install`].
    pub fn push_batch(&mut self, stream_idx: usize, batch: RecordBatch) -> Result<usize> {
        let size = self.admit(&batch)?;
        self.install(stream_idx, batch, size);
        Ok(size)
    }

    /// Reserve memory for `batch`, returning its memory size. The caller
    /// then hands the batch to [`Self::install`].
    ///
    /// The reservation grows by the buffers that no retained batch already
    /// references. A batch that slices an earlier batch of the merge adds
    /// nothing: charging its parent's buffers again would count memory that
    /// does not exist, once for every retained slice.
    ///
    /// When the pool refuses the growth, the batch is not counted and the
    /// reservation is unchanged. The caller can emit the pending rows,
    /// which releases the consumed batches, and call this again.
    pub fn admit(&mut self, batch: &RecordBatch) -> Result<usize> {
        let size = get_record_batch_memory_size(batch);
        self.retained.count_batch(batch);
        let needed = self.retained.memory_usage();
        // Only request additional memory from the pool when actual batch
        // usage exceeds the current reservation (which may include
        // pre-reserved bytes from sort_spill_reservation_bytes).
        let grown = try_grow_reservation_to_at_least(&mut self.reservation, needed);
        if grown.is_err() {
            self.retained.uncount_batch(batch);
        }
        // Follow the tracker also after a refusal. It identifies a buffer by
        // its start address and keeps the largest capacity it has seen for
        // one: when a retained batch holds a smaller view of the refused
        // batch's allocation (two FFI views of one buffer), the tracker
        // still counts the larger capacity and will release that much with
        // the retained batch. The next successful growth reserves the
        // difference; the reservation never shrinks by more than it holds.
        self.batches_mem_used = self.retained.memory_usage();
        grown.map(|()| size)
    }

    /// Make `batch`, which [`Self::admit`] accepted with memory size
    /// `size`, the cursor batch of `stream_idx`.
    pub fn install(&mut self, stream_idx: usize, batch: RecordBatch, size: usize) {
        // The stream's previous batch, if still retained, is now kept only
        // for its pending rows.
        self.heads_mem_used = self.heads_mem_used - self.head_sizes[stream_idx] + size;
        self.head_sizes[stream_idx] = size;
        let batch_idx = self.batches.len();
        self.batches.push((stream_idx, batch));
        self.cursors[stream_idx] = BatchCursor {
            batch_idx,
            row_idx: 0,
        };
    }

    /// Append the next row from `stream_idx`
    pub fn push_row(&mut self, stream_idx: usize) {
        debug_assert!(
            self.batches
                .get(self.cursors[stream_idx].batch_idx)
                .is_some_and(|(_, batch)| {
                    self.cursors[stream_idx].row_idx < batch.num_rows()
                }),
            "push_row on stream {stream_idx} with no live batch"
        );
        let cursor = &mut self.cursors[stream_idx];
        let row_idx = cursor.row_idx;
        cursor.row_idx += 1;
        self.indices.push((cursor.batch_idx, row_idx));
    }

    /// Returns the number of in-progress rows in this [`BatchBuilder`]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if the batches retained only for pending rows, those
    /// that are no longer a stream's cursor batch, take more memory than the
    /// largest of the cursor batches (the live heads), the initial
    /// reservation and [`RETAINED_FLOOR_BYTES`].
    ///
    /// A merge that waits for `batch_size` rows otherwise retains every
    /// consumed input batch until then. Small batches with large buffers,
    /// such as a Parquet reader's decode of small row groups, then hold far
    /// more memory than their rows: 7-row batches with 64 KiB buffers made
    /// an 8,192-row output hold about 1,170 batches. The merge checks this
    /// after each new input batch, and emitting when it returns `true` keeps
    /// the retained memory below the heads, plus that limit, plus one batch.
    ///
    /// Inputs of `batch_size`-row batches of similar sizes keep emitting
    /// `batch_size` rows: an input replaces at most one batch while an
    /// output fills, so the batches retained for it take about as much as
    /// the heads. Smaller batches also keep `batch_size` rows while the
    /// batches retained for an output stay below the limit. Wide rows in
    /// small batches can pass it and emit earlier: 4 KiB strings in 256-row
    /// batches from two inputs emit 1,024-row batches instead of 8,192.
    pub fn retains_more_than_heads(&self) -> bool {
        let limit = self
            .heads_mem_used
            .max(self.initial_reservation)
            .max(RETAINED_FLOOR_BYTES);
        self.batches_mem_used.saturating_sub(self.heads_mem_used) > limit
    }

    /// Returns `true` if this [`BatchBuilder`] contains no in-progress rows
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Returns the schema of this [`BatchBuilder`]
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Try to interleave all columns using the given index slice.
    fn try_interleave_columns(
        &self,
        indices: &[(usize, usize)],
    ) -> Result<Vec<ArrayRef>> {
        let mut arrays: Vec<&dyn Array> = Vec::with_capacity(self.batches.len());
        (0..self.schema.fields.len())
            .map(|column_idx| {
                arrays.clear();
                arrays.extend(
                    self.batches
                        .iter()
                        .map(|(_, batch)| batch.column(column_idx).as_ref()),
                );
                // Arrow 58.1.0+ returns OffsetOverflowError directly from
                // interleave, allowing retry_interleave to shrink the batch.
                interleave(&arrays, indices).map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Builds a record batch from the first `rows_to_emit` buffered rows.
    fn finish_record_batch(
        &mut self,
        rows_to_emit: usize,
        columns: Vec<ArrayRef>,
    ) -> Result<RecordBatch> {
        // Remove consumed indices, keeping any remaining for the next call.
        self.indices.drain(..rows_to_emit);

        if self.indices.is_empty() {
            self.retain_cursor_batches();
        } else {
            self.retain_live_batches();
        }

        debug_assert_eq!(self.batches_mem_used, self.retained.memory_usage());

        // Release excess memory back to the pool, but never shrink below
        // initial_reservation to maintain the anti-starvation guarantee
        // for the merge phase.
        let target = self.batches_mem_used.max(self.initial_reservation);
        if self.reservation.size() > target {
            self.reservation.shrink(self.reservation.size() - target);
        }

        RecordBatch::try_new(Arc::clone(&self.schema), columns).map_err(Into::into)
    }

    fn retain_cursor_batches(&mut self) {
        // New cursors are only created once the previous cursor for the stream
        // is finished. This means all remaining rows from all but the last batch
        // for each stream have been yielded to the newly created record batch
        //
        // We can therefore drop all but the last live cursor batch for each stream
        let mut batch_idx = 0;
        let mut retained = 0;
        self.batches.retain(|(stream_idx, batch)| {
            let stream_cursor = &mut self.cursors[*stream_idx];
            let is_cursor_batch = stream_cursor.batch_idx == batch_idx;
            let retain = is_cursor_batch && stream_cursor.row_idx < batch.num_rows();
            batch_idx += 1;

            if retain {
                stream_cursor.batch_idx = retained;
                retained += 1;
            } else {
                if is_cursor_batch {
                    *stream_cursor = BatchCursor::RELEASED;
                    self.heads_mem_used -=
                        std::mem::take(&mut self.head_sizes[*stream_idx]);
                }
                self.batches_mem_used -= self.retained.uncount_batch(batch);
            }
            retain
        });
    }

    fn retain_live_batches(&mut self) {
        let mut retain_batch = vec![false; self.batches.len()];
        for (batch_idx, _) in &self.indices {
            retain_batch[*batch_idx] = true;
        }

        for cursor in &self.cursors {
            if self
                .batches
                .get(cursor.batch_idx)
                .is_some_and(|(_, batch)| cursor.row_idx < batch.num_rows())
            {
                retain_batch[cursor.batch_idx] = true;
            }
        }

        let mut batch_idx = 0;
        let mut retained = 0;
        let mut remap = vec![usize::MAX; self.batches.len()];
        self.batches.retain(|(_, batch)| {
            let retain = retain_batch[batch_idx];
            if retain {
                remap[batch_idx] = retained;
                retained += 1;
            } else {
                self.batches_mem_used -= self.retained.uncount_batch(batch);
            }
            batch_idx += 1;
            retain
        });

        for (batch_idx, _) in &mut self.indices {
            *batch_idx = remap[*batch_idx];
        }
        for (cursor, head_size) in self.cursors.iter_mut().zip(&mut self.head_sizes) {
            if let Some(new_idx) = remap.get(cursor.batch_idx) {
                // `usize::MAX` means the cursor's batch was released.
                cursor.batch_idx = *new_idx;
                if *new_idx == usize::MAX {
                    self.heads_mem_used -= std::mem::take(head_size);
                }
            }
        }
    }

    /// Drains the in_progress row indexes, and builds a new RecordBatch from them
    ///
    /// Will then drop any batches for which all rows have been yielded to the output.
    /// If an offset overflow occurs (e.g. string/list offsets exceed i32::MAX),
    /// retries with progressively fewer rows until it succeeds.
    ///
    /// Returns `None` if no pending rows
    pub fn build_record_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.is_empty() {
            return Ok(None);
        }

        let (rows_to_emit, columns) =
            retry_interleave(self.indices.len(), self.indices.len(), |rows_to_emit| {
                self.try_interleave_columns(&self.indices[..rows_to_emit])
            })?;

        Ok(Some(self.finish_record_batch(rows_to_emit, columns)?))
    }
}

/// Try to grow `reservation` so it covers at least `needed` bytes.
///
/// When a reservation has been pre-loaded with bytes (e.g. via
/// [`MemoryReservation::take()`]), this avoids redundant pool
/// allocations: if the reservation already covers `needed`, this is
/// a no-op; otherwise only the deficit is requested from the pool.
pub(crate) fn try_grow_reservation_to_at_least(
    reservation: &mut MemoryReservation,
    needed: usize,
) -> Result<()> {
    if needed > reservation.size() {
        reservation.try_grow(needed - reservation.size())?;
    }
    Ok(())
}

/// Returns true if the error is an Arrow offset overflow.
fn is_offset_overflow(e: &DataFusionError) -> bool {
    matches!(
        e,
        DataFusionError::ArrowError(boxed, _)
            if matches!(boxed.as_ref(), ArrowError::OffsetOverflowError(_))
    )
}

#[cfg(test)]
fn offset_overflow_error() -> DataFusionError {
    DataFusionError::ArrowError(Box::new(ArrowError::OffsetOverflowError(0)), None)
}

fn retry_interleave<T, F>(
    mut rows_to_emit: usize,
    total_rows: usize,
    mut interleave: F,
) -> Result<(usize, T)>
where
    F: FnMut(usize) -> Result<T>,
{
    loop {
        match interleave(rows_to_emit) {
            Ok(value) => return Ok((rows_to_emit, value)),
            // Only offset overflow is recoverable by emitting fewer rows.
            Err(e) if is_offset_overflow(&e) => {
                rows_to_emit /= 2;
                if rows_to_emit == 0 {
                    return Err(e);
                }
                warn!(
                    "Interleave offset overflow with {total_rows} rows, retrying with {rows_to_emit}"
                );
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayDataBuilder, Int32Array, ListArray};
    use arrow::buffer::Buffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion_execution::memory_pool::{
        MemoryConsumer, MemoryPool, UnboundedMemoryPool,
    };

    fn overflow_list_batch() -> RecordBatch {
        let values_field = Arc::new(Field::new_list_field(DataType::Int32, true));
        // SAFETY: This intentionally constructs an invalid child length so
        // Arrow's interleave hits offset overflow before touching child data.
        let list = ListArray::from(unsafe {
            ArrayDataBuilder::new(DataType::List(Arc::clone(&values_field)))
                .len(1)
                .add_buffer(Buffer::from_slice_ref([0_i32, i32::MAX]))
                .add_child_data(Int32Array::from(Vec::<i32>::new()).to_data())
                .build_unchecked()
        });
        let schema = Arc::new(Schema::new(vec![Field::new(
            "list_col",
            DataType::List(values_field),
            true,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(list)]).unwrap()
    }

    /// Two one-column batches over the same allocation, as two Arrow FFI
    /// views of one buffer can be: the first reports 64 bytes, the second
    /// 4 KiB from the same start address.
    fn views_of_one_allocation() -> (RecordBatch, RecordBatch) {
        use arrow::array::Int64Array;
        use arrow::buffer::ScalarBuffer;
        use std::ptr::NonNull;

        let owner: Arc<Vec<i64>> = Arc::new((0..512).collect());
        let ptr = NonNull::new(owner.as_ptr().cast::<u8>().cast_mut()).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let view = |len: usize| {
            // SAFETY: `owner` holds `len` values at `ptr`, and the buffer
            // keeps it alive.
            let buffer = unsafe {
                Buffer::from_custom_allocation(
                    ptr,
                    len * size_of::<i64>(),
                    Arc::clone(&owner) as _,
                )
            };
            let array = Int64Array::new(ScalarBuffer::new(buffer, 0, len), None);
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap()
        };
        (view(8), view(512))
    }

    /// The pool refuses the larger view of an address that a retained
    /// batch already holds. The builder must stay consistent with what it
    /// charged: releasing the retained batch afterwards frees its charge,
    /// no more.
    #[test]
    fn refused_growth_of_a_retained_address_releases_what_was_charged() {
        use datafusion_execution::memory_pool::GreedyMemoryPool;

        let (small, large) = views_of_one_allocation();
        let schema = small.schema();
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024));
        let reservation = MemoryConsumer::new("test").register(&pool);
        let mut builder = BatchBuilder::new(schema, 2, 16, reservation);

        assert_eq!(builder.push_batch(0, small).unwrap(), 64);
        assert_eq!(pool.reserved(), 64);
        builder.push_batch(1, large.clone()).unwrap_err();
        assert_eq!(pool.reserved(), 64);

        // Emit the retained batch's rows: it is released.
        for _ in 0..8 {
            builder.push_row(0);
        }
        let batch = builder.build_record_batch().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 8);
        assert_eq!(pool.reserved(), 0);

        // With its address free again, the larger view is charged in full
        // when the pool has room for it.
        drop(builder);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(8192));
        let reservation = MemoryConsumer::new("test").register(&pool);
        let mut builder = BatchBuilder::new(large.schema(), 2, 16, reservation);
        assert_eq!(builder.push_batch(1, large).unwrap(), 4096);
        assert_eq!(pool.reserved(), 4096);
    }

    fn reservation() -> MemoryReservation {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        MemoryConsumer::new("test").register(&pool)
    }

    fn int_batch(values: Vec<i32>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap()
    }

    fn push_n_rows(builder: &mut BatchBuilder, stream_idx: usize, n: usize) {
        for _ in 0..n {
            builder.push_row(stream_idx);
        }
    }

    fn emit_n_rows(builder: &mut BatchBuilder, n: usize) -> RecordBatch {
        let columns = builder
            .try_interleave_columns(&builder.indices[..n])
            .unwrap();
        builder.finish_record_batch(n, columns).unwrap()
    }

    fn assert_int_output(batch: &RecordBatch, expected: &[i32]) {
        let actual = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_partial_emit_releases_unreferenced_and_retains_live_batches() {
        let batch0 = int_batch(vec![10, 11]);
        let batch1 = int_batch(vec![20, 21]);
        let batch2 = int_batch(vec![30, 31]);
        let batch1_size = get_record_batch_memory_size(&batch1);
        let batch2_size = get_record_batch_memory_size(&batch2);
        let schema = batch0.schema();
        let mut builder = BatchBuilder::new(Arc::clone(&schema), 3, 6, reservation());

        builder.push_batch(0, batch0).unwrap();
        push_n_rows(&mut builder, 0, 2);
        builder.push_batch(1, batch1).unwrap();
        push_n_rows(&mut builder, 1, 2);
        // Keep one stream empty so an unloaded cursor cannot retain consumed batches.
        builder.push_batch(0, batch2).unwrap();

        let output = emit_n_rows(&mut builder, 2);
        assert_int_output(&output, &[10, 11]);

        assert_eq!(builder.len(), 2);
        assert_eq!(builder.batches.len(), 2);
        assert_eq!(builder.batches_mem_used, batch1_size + batch2_size);
        assert_eq!(builder.reservation.size(), batch1_size + batch2_size);

        push_n_rows(&mut builder, 0, 2);
        let output = emit_n_rows(&mut builder, 4);
        assert_int_output(&output, &[20, 21, 30, 31]);

        assert!(builder.is_empty());
        assert!(builder.batches.is_empty());
        assert_eq!(builder.batches_mem_used, 0);
        assert_eq!(builder.reservation.size(), 0);
    }

    #[test]
    fn test_released_cursor_accepts_new_batch_for_stream() {
        let batch0 = int_batch(vec![10]);
        let batch1 = int_batch(vec![20]);
        let schema = batch0.schema();
        let mut builder = BatchBuilder::new(Arc::clone(&schema), 1, 1, reservation());

        builder.push_batch(0, batch0).unwrap();
        builder.push_row(0);
        let output = emit_n_rows(&mut builder, 1);
        assert_int_output(&output, &[10]);
        assert!(builder.batches.is_empty());

        builder.push_batch(0, batch1).unwrap();
        builder.push_row(0);
        let output = emit_n_rows(&mut builder, 1);
        assert_int_output(&output, &[20]);
        assert!(builder.batches.is_empty());
        assert_eq!(builder.batches_mem_used, 0);
        assert_eq!(builder.reservation.size(), 0);
    }

    #[test]
    fn test_retry_interleave_halves_rows_until_success() {
        let mut attempts = Vec::new();

        let (rows_to_emit, result) = retry_interleave(4, 4, |rows_to_emit| {
            attempts.push(rows_to_emit);
            if rows_to_emit > 1 {
                Err(offset_overflow_error())
            } else {
                Ok("ok")
            }
        })
        .unwrap();

        assert_eq!(rows_to_emit, 1);
        assert_eq!(result, "ok");
        assert_eq!(attempts, vec![4, 2, 1]);
    }

    #[test]
    fn test_is_offset_overflow_matches_arrow_error() {
        assert!(is_offset_overflow(&offset_overflow_error()));
    }

    #[test]
    fn test_retry_interleave_does_not_retry_non_offset_errors() {
        let mut attempts = Vec::new();

        let error = retry_interleave(4, 4, |rows_to_emit| {
            attempts.push(rows_to_emit);
            Err::<(), _>(DataFusionError::Execution("boom".into()))
        })
        .unwrap_err();

        assert_eq!(attempts, vec![4]);
        assert!(matches!(error, DataFusionError::Execution(msg) if msg == "boom"));
    }

    #[test]
    fn test_try_interleave_columns_surfaces_arrow_offset_overflow() {
        let batch = overflow_list_batch();
        let schema = batch.schema();
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let reservation = MemoryConsumer::new("test").register(&pool);
        let mut builder = BatchBuilder::new(schema, 1, 2, reservation);
        builder.push_batch(0, batch).unwrap();

        let error = builder
            .try_interleave_columns(&[(0, 0), (0, 0)])
            .unwrap_err();

        assert!(is_offset_overflow(&error));
    }
}
