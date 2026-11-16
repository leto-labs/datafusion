// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Checked bounds for work performed before retained Top-K state is resized.
use super::*;
use arrow::array::{NullArray, downcast_dictionary_array};
use arrow::datatypes::{DataType, Int16Type, Int32Type, Int64Type};
use datafusion_common::resources_datafusion_err;
use datafusion_physical_expr::expressions::Column;

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
        return Ok(1);
    }
    let value = match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().value_length(row) as usize,
        DataType::LargeUtf8 => array.as_string::<i64>().value_length(row) as usize,
        DataType::Binary => array.as_binary::<i32>().value_length(row) as usize,
        DataType::LargeBinary => array.as_binary::<i64>().value_length(row) as usize,
        DataType::Utf8View if encoding => array.as_string_view().value(row).len(),
        DataType::BinaryView if encoding => array.as_binary_view().value(row).len(),
        DataType::Utf8View | DataType::BinaryView => 16,
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

/// O(1) for fixed-width and offset-based arrays. Only views and nested
/// values require element visits. No ScalarValue/ArrayData allocation is made.
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
    use arrow::array::{Int64Array, StringArray, StringViewArray};
    #[test]
    fn flat_bounds_do_not_visit_elements() -> Result<()> {
        let fixed: ArrayRef = Arc::new(Int64Array::from(vec![1; 8192]));
        let bytes: ArrayRef = Arc::new(StringArray::from(vec!["value"; 8192]));
        ELEMENT_VISITS.with(|n| n.set(0));
        assert_eq!(encoded_bytes(&fixed)?, 8192 * 9);
        assert!(encoded_bytes(&bytes)? > 8192 * 5);
        assert_eq!(ELEMENT_VISITS.with(|n| n.get()), 0);
        let views: ArrayRef =
            Arc::new(StringViewArray::from(vec!["long string view value"; 1024]));
        encoded_bytes(&views)?;
        assert_eq!(ELEMENT_VISITS.with(|n| n.get()), 1024);
        Ok(())
    }
}
