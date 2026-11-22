// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Checked bounds for work performed before retained Top-K state is resized.
use super::*;
use arrow::array::{GenericByteViewArray, NullArray, downcast_dictionary_array};
use arrow::datatypes::{ByteViewType, DataType, Int16Type, Int32Type, Int64Type};
use datafusion_common::resources_datafusion_err;
use datafusion_physical_expr::expressions::Column;

const VIEW_WIDTH: usize = size_of::<u128>();

pub(super) fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| resources_datafusion_err!("PartitionedTopK memory size overflow"))
}
pub(super) fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| resources_datafusion_err!("PartitionedTopK memory size overflow"))
}

/// Existing vector allocation that may be replaced during this batch. New
/// elements and minimum capacities are already covered by per-row work.
pub(super) fn vector_growth<T>(
    len: usize,
    capacity: usize,
    rows: usize,
) -> Result<usize> {
    if add(len, rows)? > capacity {
        mul(capacity, 2 * size_of::<T>())
    } else {
        Ok(0)
    }
}

/// Positional batch maps, cloned RecordBatch headers/column references,
/// borrowed batch vectors and the per-batch remap overlap the old store.
pub(super) fn emission_batches(batches: usize, columns: usize) -> Result<usize> {
    mul(batches, add(192, mul(columns, size_of::<ArrayRef>())?)?)
}

pub(super) fn input_rows(rows: usize) -> Result<()> {
    if rows > u32::MAX as usize {
        return Err(resources_datafusion_err!(
            "PartitionedTopK batch exceeds u32 row indices"
        ));
    }
    Ok(())
}

pub(super) fn dimensions(k: usize, batch_size: usize) -> Result<()> {
    // StoreRef.row and gathered/compacted row indices are u32.
    input_rows(batch_size)?;
    if k == 0 || batch_size == 0 {
        return Err(resources_datafusion_err!(
            "PartitionedTopK requires positive K and batch size"
        ));
    }
    mul(k, size_of::<PartitionSlot>())?;
    mul(batch_size, 64)?;
    Ok(())
}

pub(super) fn initial_bytes(batch_size: usize, fields: usize) -> Result<usize> {
    add(
        add(
            mul(
                batch_size,
                2 * (ESTIMATED_BYTES_PER_ROW + size_of::<usize>()),
            )?,
            mul(fields, 1024)?,
        )?,
        4096,
    )
}

/// Bound column-key evaluation without executing an arbitrary expression that
/// can allocate an unbounded intermediate before the pool sees it. Computed
/// keys must be materialized by the child projection, as Longbow admission does.
pub(super) fn column<'a>(
    expr: &dyn PhysicalExpr,
    batch: &'a RecordBatch,
) -> Result<&'a ArrayRef> {
    let column = expr.downcast_ref::<Column>().ok_or_else(|| {
        resources_datafusion_err!(
            "PartitionedTopK requires materialized column keys for bounded encoding"
        )
    })?;
    batch
        .columns()
        .get(column.index())
        .ok_or_else(|| resources_datafusion_err!("PartitionedTopK key outside schema"))
}

pub(super) fn partition_fields(mut fields: Vec<SortField>) -> Vec<SortField> {
    if fields.is_empty() {
        fields.push(SortField::new(DataType::Null));
    }
    fields
}
pub(super) fn partition_arrays(mut arrays: Vec<ArrayRef>, rows: usize) -> Vec<ArrayRef> {
    if arrays.is_empty() {
        arrays.push(Arc::new(NullArray::new(rows)));
    }
    arrays
}

/// Logical bytes, not physical buffer capacity: repeated string views can
/// expand by orders of magnitude when encoded. Nested/dictionary values are
/// visited by index without constructing ScalarValue or ArrayData copies.
fn logical_row(array: &dyn Array, row: usize, encoding: bool) -> Result<usize> {
    #[cfg(test)]
    ELEMENT_VISITS.with(|n| n.set(n.get() + 1));
    if array.is_null(row) {
        return Ok(match array.data_type() {
            DataType::Utf8View | DataType::BinaryView => VIEW_WIDTH,
            _ => 1,
        });
    }
    let value = match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().value_length(row) as usize,
        DataType::LargeUtf8 => array.as_string::<i64>().value_length(row) as usize,
        DataType::Binary => array.as_binary::<i32>().value_length(row) as usize,
        DataType::LargeBinary => array.as_binary::<i64>().value_length(row) as usize,
        DataType::Utf8View if encoding => array.as_string_view().value(row).len(),
        DataType::BinaryView if encoding => array.as_binary_view().value(row).len(),
        DataType::Utf8View | DataType::BinaryView => VIEW_WIDTH,
        DataType::FixedSizeBinary(n) => usize::try_from(*n).map_err(|_| {
            resources_datafusion_err!("PartitionedTopK negative fixed width")
        })?,
        DataType::Struct(_) => {
            return array.as_struct().columns().iter().try_fold(9, |n, child| {
                add(n, logical_row(child.as_ref(), row, encoding)?)
            });
        }
        DataType::Dictionary(_, _) => {
            downcast_dictionary_array! { array => {
                return logical_row(array.values().as_ref(), array.keys().value(row) as usize, encoding);
            } _ => unreachable!("dictionary type") }
        }
        DataType::Union(fields, arrow::datatypes::UnionMode::Sparse) if !encoding => {
            // Sparse gather copies every child's row, including inactive ones.
            let union = array.as_union();
            return fields.iter().try_fold(9, |n, (id, _)| {
                add(n, logical_row(union.child(id).as_ref(), row, false)?)
            });
        }
        DataType::Union(_, _) => {
            let union = array.as_union();
            return logical_row(
                union.child(union.type_id(row)).as_ref(),
                union.value_offset(row),
                encoding,
            );
        }
        DataType::RunEndEncoded(ends, _) => {
            macro_rules! run {
                ($t:ty) => {{
                    let run = array.as_run::<$t>();
                    return logical_row(
                        run.values().as_ref(),
                        run.get_physical_index(row),
                        encoding,
                    );
                }};
            }
            match ends.data_type() {
                DataType::Int16 => run!(Int16Type),
                DataType::Int32 => run!(Int32Type),
                DataType::Int64 => run!(Int64Type),
                _ => {
                    return Err(resources_datafusion_err!(
                        "PartitionedTopK invalid run ends"
                    ));
                }
            }
        }
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            return range_rows(
                list.values().as_ref(),
                list.value_offsets()[row] as usize,
                list.value_offsets()[row + 1] as usize,
                encoding,
            );
        }
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            return range_rows(
                list.values().as_ref(),
                list.value_offsets()[row] as usize,
                list.value_offsets()[row + 1] as usize,
                encoding,
            );
        }
        DataType::ListView(_) => {
            let list = array.as_list_view::<i32>();
            let start = list.value_offsets()[row] as usize;
            return range_rows(
                list.values().as_ref(),
                start,
                add(start, list.value_sizes()[row] as usize)?,
                encoding,
            );
        }
        DataType::LargeListView(_) => {
            let list = array.as_list_view::<i64>();
            let start = list.value_offsets()[row] as usize;
            return range_rows(
                list.values().as_ref(),
                start,
                add(start, list.value_sizes()[row] as usize)?,
                encoding,
            );
        }
        DataType::FixedSizeList(_, width) => {
            let list = array.as_fixed_size_list();
            let start = mul(row, *width as usize)?;
            return range_rows(
                list.values().as_ref(),
                start,
                add(start, *width as usize)?,
                encoding,
            );
        }
        DataType::Map(_, _) => {
            let map = array.as_map();
            let start = map.value_offsets()[row] as usize;
            let end = map.value_offsets()[row + 1] as usize;
            return add(
                range_rows(map.keys().as_ref(), start, end, encoding)?,
                range_rows(map.values().as_ref(), start, end, encoding)?,
            );
        }
        _ => array.data_type().primitive_width().unwrap_or(1),
    };
    if encoding {
        add(9, mul(value, 2)?)
    } else {
        add(value, 9)
    }
}
fn range_rows(
    array: &dyn Array,
    start: usize,
    end: usize,
    encoding: bool,
) -> Result<usize> {
    (start..end).try_fold(9, |n, row| add(n, logical_row(array, row, encoding)?))
}

fn view_backing(array: &dyn Array) -> Result<usize> {
    let buffers = match array.data_type() {
        DataType::Utf8View => Some(array.as_string_view().data_buffers()),
        DataType::BinaryView => Some(array.as_binary_view().data_buffers()),
        _ => None,
    };
    if let Some(buffers) = buffers {
        return buffers.iter().try_fold(0, |n, b| {
            add(
                n,
                add(
                    b.capacity().max(b.ptr_offset().saturating_add(b.len())),
                    256,
                )?,
            )
        });
    }
    // Dictionary and list-view take keep the full child array, even when
    // selected rows only reference a small part of it.
    let retained = if matches!(
        array.data_type(),
        DataType::Dictionary(_, _) | DataType::ListView(_) | DataType::LargeListView(_)
    ) {
        child_work(array, |child| mul(child.get_array_memory_size(), 2))?
    } else {
        0
    };
    add(
        retained,
        child_work(array, |child| view_backing(child.as_ref()))?,
    )
}

/// Visit children directly, without constructing ArrayData or a scratch Vec.
fn child_work(
    array: &dyn Array,
    mut work: impl FnMut(&ArrayRef) -> Result<usize>,
) -> Result<usize> {
    match array.data_type() {
        DataType::Struct(_) => array
            .as_struct()
            .columns()
            .iter()
            .try_fold(0, |n, c| add(n, work(c)?)),
        DataType::List(_) => work(array.as_list::<i32>().values()),
        DataType::LargeList(_) => work(array.as_list::<i64>().values()),
        DataType::ListView(_) => work(array.as_list_view::<i32>().values()),
        DataType::LargeListView(_) => work(array.as_list_view::<i64>().values()),
        DataType::FixedSizeList(_, _) => work(array.as_fixed_size_list().values()),
        DataType::Map(_, _) => {
            add(work(array.as_map().keys())?, work(array.as_map().values())?)
        }
        DataType::Dictionary(_, _) => work(array.as_any_dictionary().values()),
        DataType::Union(fields, _) => fields
            .iter()
            .try_fold(0, |n, (id, _)| add(n, work(array.as_union().child(id))?)),
        DataType::RunEndEncoded(ends, _) => match ends.data_type() {
            DataType::Int16 => work(array.as_run::<Int16Type>().values()),
            DataType::Int32 => work(array.as_run::<Int32Type>().values()),
            DataType::Int64 => work(array.as_run::<Int64Type>().values()),
            _ => Err(resources_datafusion_err!(
                "PartitionedTopK invalid run ends"
            )),
        },
        _ => Ok(0),
    }
}

/// The same bound as `logical_row` for a byte view. Gather copies the
/// descriptors and shares their backing buffers. Encoding can expand repeated
/// views, so count every valid length, without resolving its backing buffer.
fn view_array_bytes<T: ByteViewType>(
    array: &GenericByteViewArray<T>,
    encoding: bool,
) -> Result<usize> {
    let null_count = array.null_count();
    let valid_count = array.len() - null_count;
    let null_bytes = mul(null_count, VIEW_WIDTH)?;
    if !encoding {
        return add(null_bytes, mul(valid_count, 9 + VIEW_WIDTH)?);
    }
    let sum_lengths = |views: &[u128]| {
        views
            .iter()
            .try_fold(0, |n, view| add(n, *view as u32 as usize))
    };
    // A null view still occupies a descriptor. Skip its length bits, which
    // can be nonzero even though there is no value to encode.
    let lengths = if let Some(nulls) = array.nulls().filter(|_| null_count != 0) {
        nulls.valid_slices().try_fold(0, |n, (start, end)| {
            add(n, sum_lengths(&array.views()[start..end])?)
        })?
    } else {
        sum_lengths(array.views())?
    };
    add(add(null_bytes, mul(valid_count, 9)?)?, mul(lengths, 2)?)
}

/// O(1) for fixed-width, offset-based, and gathered view arrays. Encoded
/// views and nested values can require element visits. No ScalarValue or
/// ArrayData allocation is made.
fn array_bytes(array: &dyn Array, encoding: bool) -> Result<usize> {
    let rows = array.len();
    if let Some(width) = array.data_type().primitive_width() {
        return add(
            mul(rows, width)?,
            if encoding { rows } else { rows.div_ceil(8) },
        );
    }
    let offsets = match array.data_type() {
        DataType::Utf8 => Some((
            array.as_string::<i32>().value_offsets()[0] as usize,
            *array.as_string::<i32>().value_offsets().last().unwrap() as usize,
            4,
        )),
        DataType::LargeUtf8 => Some((
            array.as_string::<i64>().value_offsets()[0] as usize,
            *array.as_string::<i64>().value_offsets().last().unwrap() as usize,
            8,
        )),
        DataType::Binary => Some((
            array.as_binary::<i32>().value_offsets()[0] as usize,
            *array.as_binary::<i32>().value_offsets().last().unwrap() as usize,
            4,
        )),
        DataType::LargeBinary => Some((
            array.as_binary::<i64>().value_offsets()[0] as usize,
            *array.as_binary::<i64>().value_offsets().last().unwrap() as usize,
            8,
        )),
        _ => None,
    };
    if let Some((start, end, width)) = offsets {
        // Row variable-byte encoding uses blocks/terminators: twice the byte
        // span plus nine bytes per row bounds that overhead, including nulls.
        return if encoding {
            add(mul(end - start, 2)?, mul(rows, 9)?)
        } else {
            add(
                end - start,
                add(mul(add(rows, 1)?, width)?, rows.div_ceil(8))?,
            )
        };
    }
    match array.data_type() {
        DataType::Utf8View => view_array_bytes(array.as_string_view(), encoding),
        DataType::BinaryView => view_array_bytes(array.as_binary_view(), encoding),
        DataType::Null => Ok(if encoding { rows } else { 0 }),
        DataType::Boolean => Ok(if encoding {
            mul(rows, 2)?
        } else {
            mul(rows.div_ceil(8), 2)?
        }),
        DataType::FixedSizeBinary(width) => add(
            mul(rows, *width as usize)?,
            if encoding { rows } else { rows.div_ceil(8) },
        ),
        _ => (0..rows).try_fold(0, |n, row| add(n, logical_row(array, row, encoding)?)),
    }
}

pub(super) fn encoded_bytes(array: &ArrayRef) -> Result<usize> {
    // Nested encoders may also encode unused child ranges/dictionary values.
    add(
        array_bytes(array.as_ref(), true)?,
        child_work(array.as_ref(), encoded_bytes)?,
    )
}

pub(super) fn partition_work<S>(rows: usize) -> Result<usize> {
    // A new partition can allocate the minimum four heap slots. A hash table
    // needs up to twice the new entry count; 32 bytes covers controls/alignment.
    mul(
        rows,
        4 * size_of::<PartitionSlot>() + 2 * size_of::<(Vec<u8>, S)>() + 32,
    )
}

pub(super) struct AdmissionGrowth {
    target: usize,
}
impl AdmissionGrowth {
    pub(super) fn new(base: usize, work: usize) -> Result<Self> {
        Ok(Self {
            target: add(base, work)?,
        })
    }
    pub(super) fn reserve(
        &mut self,
        reservation: &MemoryReservation,
        bytes: usize,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let target = add(self.target, bytes)?;
        reservation.try_resize(target.max(reservation.size()))?;
        self.target = target;
        Ok(())
    }
}

fn copied_backing(array: &ArrayRef) -> Result<usize> {
    let whole = if matches!(
        array.data_type(),
        DataType::Dictionary(_, _) | DataType::ListView(_) | DataType::LargeListView(_)
    ) {
        child_work(array.as_ref(), |child| array_bytes(child.as_ref(), false))?
    } else {
        0
    };
    add(whole, child_work(array.as_ref(), copied_backing)?)
}

/// Bound a gather/interleave, including metadata and Arrow's geometric buffer
/// growth. Encoded row scratch is bounded separately. View payload stays shared.
fn array_work_metadata(array: &ArrayRef) -> Result<usize> {
    let own = add(
        array
            .get_array_memory_size()
            .saturating_sub(array.get_buffer_memory_size()),
        64,
    )?;
    let extra = match array.data_type() {
        DataType::Utf8View => mul(array.as_string_view().data_buffers().len(), 160)?,
        DataType::BinaryView => mul(array.as_binary_view().data_buffers().len(), 160)?,
        DataType::Union(fields, _) => {
            let slots = fields
                .iter()
                .map(|(id, _)| id as usize + 1)
                .max()
                .unwrap_or(0);
            mul(slots, size_of::<Option<ArrayRef>>())?
        }
        _ => 0,
    };
    let extra = add(extra, child_work(array.as_ref(), array_work_metadata)?)?;
    add(own, extra)
}

pub(super) fn copy_rows<B: Borrow<RecordBatch>>(
    batches: &[B],
    rows: &[(usize, usize)],
) -> Result<usize> {
    let mut bytes = add(4096, mul(rows.len(), 32)?)?;
    let mut seen = datafusion_common::HashSet::new();
    for &(batch, row) in rows {
        let first = seen.insert(batch);
        for array in batches[batch].borrow().columns() {
            // Interleave's growable Arrow buffers can double their capacity.
            bytes = add(bytes, mul(logical_row(array.as_ref(), row, false)?, 2)?)?;
            if first {
                bytes = add(
                    bytes,
                    add(
                        mul(array_work_metadata(array)?, 4)?,
                        mul(copied_backing(array)?, 2)?,
                    )?,
                )?;
            }
        }
    }
    Ok(bytes)
}

pub(super) fn batch_work(
    batch: &RecordBatch,
    keys: impl Iterator<Item = Arc<dyn PhysicalExpr>>,
) -> Result<usize> {
    input_rows(batch.num_rows())?;
    let mut encoded = 0;
    for expr in keys {
        encoded = add(encoded, encoded_bytes(column(expr.as_ref(), batch)?)?)?;
    }
    // Two times encoded bytes admits geometric scratch-buffer growth; another
    // two times covers interned/replacement keys. Encoded already sums BOTH key
    // sets. Two Rows offset vectors may each double; gathers need u32 indices.
    let key_work = add(
        mul(encoded, 4)?,
        mul(
            add(batch.num_rows(), 1)?,
            4 * size_of::<usize>() + size_of::<u32>(),
        )?,
    )?;
    // No coordinate allocation before reservation. Sum the same row bound in
    // place; a gather can copy each input row at most once.
    let payload = batch.columns().iter().try_fold(4096, |n, a| {
        let rows = array_bytes(a.as_ref(), false)?;
        let metadata = array_work_metadata(a)?;
        add(
            n,
            add(rows, add(view_backing(a.as_ref())?, mul(metadata, 4)?)?)?,
        )
    })?;
    // Take/interleave may double output buffer capacity. Metadata's factor
    // four above covers output headers, owners, accountant entries and controls.
    add(key_work, mul(payload, 2)?)
}

pub(super) fn admit(
    reservation: &MemoryReservation,
    base: usize,
    work: usize,
    growth: usize,
    limit: Option<usize>,
) -> Result<bool> {
    let needed = add(base, add(work, growth)?)?.max(reservation.size());
    if limit.is_some_and(|limit| needed > limit) {
        return Ok(false);
    }
    match reservation.try_resize(needed) {
        Ok(()) => Ok(true),
        Err(datafusion_common::DataFusionError::ResourcesExhausted(_)) => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
thread_local! { static ELEMENT_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
mod complexity_tests {
    use super::*;
    use arrow::array::{
        BinaryViewArray, Int64Array, StringArray, StringViewArray, UInt32Array,
    };
    use arrow::buffer::NullBuffer;
    use arrow::compute::{interleave, take};
    use datafusion_physical_expr::expressions::col;
    #[test]
    fn flat_bounds_do_not_visit_elements() -> Result<()> {
        let fixed: ArrayRef = Arc::new(Int64Array::from(vec![1; 8192]));
        let bytes: ArrayRef = Arc::new(StringArray::from(vec!["value"; 8192]));
        ELEMENT_VISITS.with(|n| n.set(0));
        assert_eq!(encoded_bytes(&fixed)?, 8192 * 9);
        assert!(encoded_bytes(&bytes)? > 8192 * 5);
        assert_eq!(ELEMENT_VISITS.with(|n| n.get()), 0);
        Ok(())
    }

    #[test]
    fn byte_view_bounds_match_the_row_formula_without_generic_row_dispatch() -> Result<()>
    {
        let strings = StringViewArray::from(vec![
            "before slice",
            "inline",
            "nonzero view length under a null",
            "out-of-line string value",
            "",
            "after slice",
        ]);
        let nullable = StringViewArray::try_new(
            strings.views().clone(),
            strings.data_buffers().to_vec(),
            Some(NullBuffer::from(vec![true, true, false, true, true, true])),
        )?
        .slice(1, 4);
        let binary = BinaryViewArray::from(vec![
            Some(b"outer".as_slice()),
            None,
            Some(b"\xff\x00test".as_slice()),
            Some(b"out-of-line binary value".as_slice()),
            None,
        ])
        .slice(1, 3);
        let arrays: &[&dyn Array] = &[
            &strings,
            &nullable,
            &binary,
            &StringViewArray::new_null(4),
            &BinaryViewArray::new_null(0),
        ];
        for array in arrays {
            for encoding in [false, true] {
                let expected = (0..array.len())
                    .try_fold(0, |n, row| add(n, logical_row(*array, row, encoding)?))?;
                ELEMENT_VISITS.with(|n| n.set(0));
                assert_eq!(array_bytes(*array, encoding)?, expected);
                assert_eq!(ELEMENT_VISITS.with(|n| n.get()), 0);
            }
        }
        Ok(())
    }

    #[test]
    fn null_byte_view_gather_bounds_cover_allocated_buffers() -> Result<()> {
        const ROWS: usize = 8192;
        let indices = UInt32Array::from_iter_values((0..ROWS as u32).rev());
        let coordinates = (0..ROWS).rev().map(|row| (0, row)).collect::<Vec<_>>();

        for sliced in [false, true] {
            let len = ROWS + usize::from(sliced) * 4;
            let strings = StringViewArray::from_iter(
                (0..len).map(|row| (row % 128 == 0).then_some("out-of-line string")),
            );
            let binaries = BinaryViewArray::from_iter((0..len).map(|row| {
                (row % 128 == 0).then_some(b"out-of-line binary\xff".as_slice())
            }));
            let arrays: [ArrayRef; 4] = [
                Arc::new(StringViewArray::new_null(len)),
                Arc::new(BinaryViewArray::new_null(len)),
                Arc::new(strings),
                Arc::new(binaries),
            ];
            for array in arrays {
                let array = if sliced { array.slice(2, ROWS) } else { array };
                let take_bound = mul(array_bytes(array.as_ref(), false)?, 2)?;
                let interleave_bound = (0..ROWS).try_fold(0, |bytes, row| {
                    add(bytes, mul(logical_row(array.as_ref(), row, false)?, 2)?)
                })?;
                for (kernel, bound, output) in [
                    ("take", take_bound, take(array.as_ref(), &indices, None)?),
                    (
                        "interleave",
                        interleave_bound,
                        interleave(&[array.as_ref()], &coordinates)?,
                    ),
                ] {
                    // The kernels share the variadic data buffers. The first
                    // buffer and the validity bitmap are newly allocated.
                    let allocated = output.to_data().buffers()[0].capacity()
                        + output.nulls().unwrap().buffer().capacity();
                    assert!(
                        bound >= allocated,
                        "{kernel} {:?}, sliced={sliced}, nulls={}: bound {bound}, allocated {allocated}",
                        array.data_type(),
                        array.null_count(),
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn null_byte_view_batch_admission_honors_tight_limit() -> Result<()> {
        const ROWS: usize = 8192;
        let batch = RecordBatch::try_from_iter([(
            "key",
            Arc::new(StringViewArray::new_null(ROWS)) as ArrayRef,
        )])?;
        let schema = batch.schema();
        let empty_strings = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringViewArray::from(vec![""; ROWS]))],
        )?;
        let state = PartitionedTopK::try_new(
            0,
            &schema,
            vec![],
            vec![],
            LexOrdering::new([PhysicalSortExpr::new_default(col("key", &schema)?)])
                .unwrap(),
            1,
            ROWS,
            &Arc::new(RuntimeEnv::default()),
            &ExecutionPlanMetricsSet::new(),
            None,
        )?;
        assert!(state.try_admit_batch(&empty_strings, None)?);
        let control = state.reserved_bytes();
        // An empty string contributes 9 encoding and 25 gather bytes. A
        // null contributes 16 to each. The weights are four and two.
        let needed = control + ROWS * (4 * (16 - 9) - 2 * (25 - 16));
        assert!(!state.try_admit_batch(&batch, Some(needed - 1))?);
        assert_eq!(state.reserved_bytes(), control);
        assert!(state.try_admit_batch(&batch, Some(needed))?);
        assert_eq!(state.reserved_bytes(), needed);
        Ok(())
    }
}
