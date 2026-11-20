// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow family JS: the one size contract between the writer of a spill
//! file and the build chunk that reads it back (review MJJ98-REVIEW3,
//! findings 1 and 2).
//!
//! A build chunk reads a batch only under a read permit of one chunk unit,
//! and evaluates the batch's join keys once, under the same permit, into
//! columns appended to the batch ([`ChunkKeys`]). The table is then built
//! from those columns, so no key expression, normalization or hash input is
//! evaluated again during the build. For this to stay inside the permit,
//! the writer checks every batch it sends to IPC with [`write_need`], an
//! upper bound of what the reader charges for it with [`admission_need`]:
//!
//! ```text
//! need = resident + copy + dictionaries + hash_scratch + TABLE_ROW_BYTES x rows + TABLE_FIXED_BYTES
//!   resident     = decoded batch + new bytes of its evaluated keys
//!   copy         = the batch's columns and key columns once more (the table's concatenation)
//!   dictionaries = dictionary values the IPC reader keeps between reads
//!   hash_scratch = temporary buffers for hashing nested and dictionary keys
//! ```
//!
//! The reader measures the decoded batch, so the writer needs a bound of the
//! decoded size of a batch before it is encoded: [`decoded_bound`]. An IPC
//! message body holds each buffer of the batch once, padded to the stream's
//! alignment (at most 64 bytes), behind a length prefix and a flatbuffer
//! header of a fixed part and one 16-byte entry per field node and buffer
//! (bounded by 256 bytes and 32 per entry); the reader
//! frames each message into an exactly sized allocation that the decoded
//! arrays share. With compression each buffer is decompressed into its own
//! allocation, and a buffer the writer left uncompressed shares the message,
//! so the bound is doubled.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, make_array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, UnionMode};
use arrow_data::ArrayData;
use datafusion_common::Result;
use datafusion_common::float_semantics::{has_float_leaf, normalize_floats};
use datafusion_common::utils::memory::RecordBatchMemoryCounter;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::expressions::{CastExpr, Column};

/// Padding and alignment the IPC writer may add to one buffer, and the
/// rounding of a decompressed buffer's allocation.
const BUFFER_PAD: usize = 64;
/// One IPC message's fixed parts: its length prefix and the flatbuffer
/// message and record batch tables with their vtables.
const MESSAGE_FIXED: usize = 256;
/// The flatbuffer metadata of one field node or buffer (a 16-byte struct in
/// the message, twice for alignment and variadic counts).
const MESSAGE_PER_ENTRY: usize = 32;
/// Bytes per row of a chunk table: the hash table and its chains (the
/// floor's estimate, 48 bytes) and the row's hash in the build's hash buffer
/// (8 bytes).
pub(crate) const TABLE_ROW_BYTES: usize = 56;
/// Fixed parts of a chunk table.
pub(crate) const TABLE_FIXED_BYTES: usize = 256;

/// What a build chunk charges for a batch: see the module documentation.
pub(crate) fn admission_need(
    resident: usize,
    copy: usize,
    dictionaries: usize,
    hash_scratch: usize,
    rows: usize,
) -> usize {
    resident
        + copy
        + dictionaries
        + hash_scratch
        + TABLE_ROW_BYTES * rows
        + TABLE_FIXED_BYTES
}

/// Buffers of `data` and its children, null buffers included.
fn buffer_count(data: &ArrayData) -> usize {
    data.buffers().len()
        + usize::from(data.nulls().is_some())
        + data.child_data().iter().map(buffer_count).sum::<usize>()
}

/// Field nodes of `data`: itself and its children.
fn node_count(data: &ArrayData) -> usize {
    1 + data.child_data().iter().map(node_count).sum::<usize>()
}

fn uses_hash_scratch(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Dictionary(_, _)
            | DataType::Struct(_)
            | DataType::List(_)
            | DataType::LargeList(_)
            | DataType::ListView(_)
            | DataType::LargeListView(_)
            | DataType::FixedSizeList(_, _)
            | DataType::Map(_, _)
            | DataType::RunEndEncoded(_, _)
            | DataType::Union(_, _)
    )
}

/// The temporary vectors in `datafusion_common::hash_utils`. For a sliced
/// array, use the full child lengths: some hash kernels slice their children
/// first, and the others hash them whole. Add all descendants' vectors so
/// this also covers a hash kernel that calls another while holding its own.
fn hash_vectors(data: &ArrayData) -> usize {
    let children = data.child_data();
    let width = size_of::<u64>();
    let bytes = match data.data_type() {
        DataType::Struct(_) => width * data.len(),
        DataType::Dictionary(_, _)
        | DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::FixedSizeList(_, _)
        | DataType::Map(_, _) => width * children[0].len(),
        DataType::RunEndEncoded(_, _) => width * children[1].len(),
        DataType::Union(_, UnionMode::Dense) => {
            128 * size_of::<Vec<u64>>()
                + width
                    * children
                        .iter()
                        .map(|child| child.len().max(4))
                        .sum::<usize>()
        }
        DataType::Union(_, UnionMode::Sparse) => {
            // Combined hashes, grouped u32 row indices, and the reused hash
            // vector for one child (Vec may double its capacity).
            width * (data.len() + (2 * data.len()).max(4)) + size_of::<u32>() * data.len()
        }
        _ => return 0,
    };
    bytes + children.iter().map(hash_vectors).sum::<usize>()
}

/// The header of an IPC message over `nodes` field nodes and `buffers`
/// buffers.
fn header_bound(nodes: usize, buffers: usize) -> usize {
    MESSAGE_FIXED + MESSAGE_PER_ENTRY * (nodes + buffers)
}

/// An upper bound of the bytes `array` holds once written and decoded: its
/// sliced bytes (Arrow's `get_slice_memory_size`, which counts variadic view
/// buffers and children whole) and the padding of each buffer.
fn array_bound(array: &dyn Array) -> Result<usize> {
    let data = array.to_data();
    Ok(data.get_slice_memory_size()? + BUFFER_PAD * buffer_count(&data))
}

/// An upper bound of the bytes the decoded `batch` holds when the reader
/// reads it back (see the module documentation).
pub(crate) fn decoded_bound(batch: &RecordBatch, compressed: bool) -> Result<usize> {
    let (mut bytes, mut nodes, mut buffers) = (0, 0, 0);
    for column in batch.columns() {
        let data = column.to_data();
        bytes += array_bound(column.as_ref())?;
        nodes += node_count(&data);
        buffers += buffer_count(&data);
    }
    bytes += header_bound(nodes, buffers);
    Ok(if compressed { 2 * bytes } else { bytes })
}

/// An upper bound of the dictionary values of `batch`, nested ones
/// included, which the IPC reader keeps after the batch is dropped.
pub(crate) fn dictionary_bound(batch: &RecordBatch) -> Result<usize> {
    fn walk(data: &ArrayData) -> Result<usize> {
        let mut bytes = 0;
        if matches!(data.data_type(), DataType::Dictionary(_, _)) {
            for values in data.child_data() {
                bytes += header_bound(node_count(values), buffer_count(values))
                    + values.get_slice_memory_size()?
                    + BUFFER_PAD * buffer_count(values);
            }
        }
        for child in data.child_data() {
            bytes += walk(child)?;
        }
        Ok(bytes)
    }
    let mut bytes = 0;
    for column in batch.columns() {
        bytes += walk(&column.to_data())?;
    }
    Ok(bytes)
}

/// Bytes actually held by the decoded dictionaries. The IPC reader can
/// retain many view buffers that are slices of one allocation, and nested
/// dictionaries can share buffers with their parent values.
pub(crate) fn dictionary_resident_bytes(batch: &RecordBatch) -> usize {
    fn count(data: &ArrayData, counter: &mut RecordBatchMemoryCounter) {
        if matches!(data.data_type(), DataType::Dictionary(..)) {
            for values in data.child_data() {
                counter.count_array(make_array(values.clone()).as_ref());
            }
        } else {
            for child in data.child_data() {
                count(child, counter);
            }
        }
    }

    let mut counter = RecordBatchMemoryCounter::new();
    for column in batch.columns() {
        let data_type = column.data_type();
        if data_type.is_nested()
            || matches!(
                data_type,
                DataType::Dictionary(..) | DataType::RunEndEncoded(..)
            )
        {
            count(&column.to_data(), &mut counter);
        }
    }
    counter.memory_usage()
}

/// A chunk concatenates only flat columns with predictable copy costs.
pub(super) fn chunk_schema(schema: &Schema) -> bool {
    schema.fields().iter().all(|field| {
        let data_type = field.data_type();
        data_type.is_primitive()
            || matches!(
                data_type,
                DataType::Boolean
                    | DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Utf8View
                    | DataType::Binary
                    | DataType::BinaryView
            )
    })
}

/// Only a column or a single fixed-width cast of a column fits the key
/// evaluation budget. A cast can otherwise hide work in its input.
pub(super) fn chunk_key(key: &PhysicalExprRef, schema: &Schema) -> bool {
    fn fixed_width(data_type: &DataType) -> bool {
        data_type.primitive_width().is_some()
            || matches!(data_type, DataType::Boolean | DataType::FixedSizeBinary(_))
    }

    key.is::<Column>()
        || key.downcast_ref::<CastExpr>().is_some_and(|cast| {
            cast.expr().is::<Column>()
                && cast
                    .expr()
                    .data_type(schema)
                    .is_ok_and(|ty| fixed_width(&ty))
                && (cast.cast_type().is_primitive()
                    || matches!(cast.cast_type(), DataType::Boolean))
        })
}

/// The join keys of a build chunk as columns of its batches. A key that is
/// a column of a type without floats is used where it is; every other key
/// (an expression, or a float column, whose `-0.0` the build normalizes) is
/// evaluated once per batch, normalized, and appended to the batch. The
/// table is built over the keyed batches with [`Self::on_keyed`], so the
/// build only clones key columns.
#[derive(Debug)]
pub(crate) struct ChunkKeys {
    on: Vec<PhysicalExprRef>,
    /// Indices into `on` of the keys that are appended, in order.
    appended: Vec<usize>,
    keyed_schema: SchemaRef,
    columns: usize,
    on_keyed: Vec<PhysicalExprRef>,
    /// Column indices whose hashing creates temporary vectors.
    hash_scratch_columns: Vec<usize>,
}

/// A batch with its appended keys.
pub(crate) struct Keyed {
    pub(crate) batch: RecordBatch,
    /// Bytes of the appended keys that the batch did not hold already.
    pub(crate) new_bytes: usize,
    /// Bytes the appended keys add to a concatenation of the batch (the
    /// referenced bytes and padding of each, also where it shares buffers
    /// with the batch).
    pub(crate) copy_bytes: usize,
}

impl ChunkKeys {
    pub(crate) fn try_new(schema: &SchemaRef, on: &[PhysicalExprRef]) -> Result<Self> {
        let columns = schema.fields().len();
        let mut appended = vec![];
        let mut fields = schema.fields().to_vec();
        let mut on_keyed = Vec::with_capacity(on.len());
        let mut hash_scratch_columns = vec![];
        for (index, key) in on.iter().enumerate() {
            let data_type = key.data_type(schema)?;
            let has_scratch = uses_hash_scratch(&data_type);
            let column = if let Some(column) = key.downcast_ref::<Column>()
                && !has_float_leaf(&data_type)
            {
                column.clone()
            } else {
                let name = format!("__hash_join_spill_key_{index}");
                let column = Column::new(&name, fields.len());
                fields.push(Arc::new(Field::new(name, data_type, true)));
                appended.push(index);
                column
            };
            if has_scratch {
                hash_scratch_columns.push(column.index());
            }
            on_keyed.push(Arc::new(column) as PhysicalExprRef);
        }
        Ok(Self {
            on: on.to_vec(),
            appended,
            keyed_schema: Arc::new(Schema::new_with_metadata(
                fields,
                schema.metadata().clone(),
            )),
            columns,
            on_keyed,
            hash_scratch_columns,
        })
    }

    /// The schema of the keyed batches.
    pub(crate) fn keyed_schema(&self) -> &SchemaRef {
        &self.keyed_schema
    }

    /// The build keys over the keyed batches: column references only.
    pub(crate) fn on_keyed(&self) -> &[PhysicalExprRef] {
        &self.on_keyed
    }

    /// Columns of the batches before the keys.
    pub(crate) fn columns(&self) -> usize {
        self.columns
    }

    /// Whether any key is evaluated (else a keyed batch is the batch).
    fn evaluates(&self) -> bool {
        !self.appended.is_empty()
    }

    /// What the appended keys of a keyed batch add to its concatenation.
    pub(crate) fn appended_copy(&self, keyed: &RecordBatch) -> Result<usize> {
        let mut bytes = 0;
        for key in keyed.columns().iter().skip(self.columns) {
            bytes += array_bound(key.as_ref())?;
        }
        Ok(bytes)
    }

    /// Hashing processes key columns one at a time. Charge the largest
    /// temporary vector footprint, including room for sliced array handles.
    pub(crate) fn hash_scratch(&self, keyed: &RecordBatch) -> usize {
        self.hash_scratch_columns
            .iter()
            .map(|&index| {
                let data = keyed.column(index).to_data();
                hash_vectors(&data) + 256 * node_count(&data)
            })
            .max()
            .unwrap_or(0)
    }

    /// Evaluates the appended keys of `batch` once and appends them.
    pub(crate) fn materialize(&self, batch: &RecordBatch) -> Result<Keyed> {
        let rows = batch.num_rows();
        let mut columns = batch.columns().to_vec();
        let mut counter = RecordBatchMemoryCounter::new();
        if self.evaluates() {
            counter.count_batch(batch);
        }
        let (mut new_bytes, mut copy_bytes) = (0, 0);
        for index in &self.appended {
            let values = self.on[*index].evaluate(batch)?.into_array(rows)?;
            let values: ArrayRef = normalize_floats(&values);
            new_bytes += counter.count_array(values.as_ref());
            // A concatenation copies the referenced bytes of each key
            // column, also of one that shares the batch's buffers.
            copy_bytes += array_bound(values.as_ref())?;
            columns.push(values);
        }
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&self.keyed_schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )?;
        Ok(Keyed {
            batch,
            new_bytes,
            copy_bytes,
        })
    }
}

/// The writer's bound of what a build chunk charges for `batch` (written
/// as is): its [`decoded_bound`], its keys evaluated here (the reader
/// evaluates the same kernels over the same values) with the padding of
/// each of their buffers, and its [`dictionary_bound`].
pub(crate) fn write_need(
    batch: &RecordBatch,
    keys: &ChunkKeys,
    compressed: bool,
) -> Result<usize> {
    let decoded = decoded_bound(batch, compressed)?;
    let (new_bytes, copy_bytes, hash_scratch) = if keys.evaluates() {
        let keyed = keys.materialize(batch)?;
        let pad = keyed
            .batch
            .columns()
            .iter()
            .skip(keys.columns())
            .map(|key| BUFFER_PAD * buffer_count(&key.to_data()))
            .sum::<usize>();
        (
            keyed.new_bytes + pad,
            keyed.copy_bytes,
            keys.hash_scratch(&keyed.batch),
        )
    } else {
        (0, 0, keys.hash_scratch(batch))
    };
    Ok(admission_need(
        decoded + new_bytes,
        decoded + copy_bytes,
        dictionary_bound(batch)?,
        hash_scratch,
        batch.num_rows(),
    ))
}
