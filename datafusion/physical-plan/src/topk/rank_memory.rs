// Modified by the Longbow authors; see third_party/README.md.

//! Conservative, checked working bounds for native rank and row-number storage.
use super::*;
use arrow::datatypes::DataType;
use datafusion_common::resources_datafusion_err;
use datafusion_physical_expr::expressions::Column;

pub(super) fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| {
        resources_datafusion_err!("PartitionedTopKRank memory size overflow")
    })
}

pub(super) fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| {
        resources_datafusion_err!("PartitionedTopKRank memory size overflow")
    })
}

pub(super) fn metadata_bound(partitions: usize, orders: usize) -> Result<usize> {
    add(
        size_of::<PartitionedTopKRank>() + 4096,
        mul(add(partitions, orders)?, 1024)?,
    )
}

pub(super) fn validate(
    schema: &SchemaRef,
    partitions: &[Arc<dyn PhysicalExpr>],
    orders: &LexOrdering,
    k: usize,
    batch_size: usize,
) -> Result<()> {
    mul(batch_size, 64)?;
    mul(k, size_of::<TopKRow>())?;
    if k == 0 || batch_size == 0 {
        return Err(resources_datafusion_err!(
            "PartitionedTopKRank requires positive K and batch size"
        ));
    }
    for expr in partitions.iter().chain(orders.iter().map(|e| &e.expr)) {
        let Some(column) = expr.downcast_ref::<Column>() else {
            return Err(resources_datafusion_err!(
                "PartitionedTopKRank requires column keys for bounded encoding"
            ));
        };
        let field = schema.fields().get(column.index()).ok_or_else(|| {
            resources_datafusion_err!("PartitionedTopKRank column outside schema")
        })?;
        let kind = field.data_type();
        if !(kind.is_primitive()
            || matches!(
                kind,
                DataType::Null
                    | DataType::Boolean
                    | DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Utf8View
                    | DataType::Binary
                    | DataType::LargeBinary
                    | DataType::BinaryView
                    | DataType::FixedSizeBinary(_)
            ))
        {
            return Err(resources_datafusion_err!(
                "PartitionedTopKRank cannot bound nested/dictionary key encoding"
            ));
        }
    }
    Ok(())
}

// Row encoding of flat values uses at most 2*logical_bytes + 64 per
// column/row (validity, block markers, padding). Views can reference a tiny
// shared buffer repeatedly: physical buffer size is NOT a bound on encoding.
pub(super) fn encoded_bound(array: &ArrayRef) -> Result<usize> {
    let mut bytes = mul(array.len(), 64)?;
    for i in 0..array.len() {
        let len = match array.data_type() {
            DataType::Utf8 => array.as_string::<i32>().value_length(i) as usize,
            DataType::LargeUtf8 => array.as_string::<i64>().value_length(i) as usize,
            DataType::Utf8View => array.as_string_view().value(i).len(),
            DataType::Binary => array.as_binary::<i32>().value_length(i) as usize,
            DataType::LargeBinary => array.as_binary::<i64>().value_length(i) as usize,
            DataType::BinaryView => array.as_binary_view().value(i).len(),
            DataType::FixedSizeBinary(n) => *n as usize,
            _ => 0,
        };
        bytes = add(bytes, mul(len, 2)?)?;
    }
    Ok(bytes)
}

pub(super) fn scratch_bound(
    state: &PartitionedTopKRank,
    batch: &RecordBatch,
) -> Result<usize> {
    if batch.num_rows() > u32::MAX as usize {
        return Err(resources_datafusion_err!(
            "PartitionedTopKRank batch exceeds u32 row indices"
        ));
    }
    let mut bytes = mul(batch.num_rows(), 512)?;
    for expr in state
        .partition_exprs
        .iter()
        .chain(state.expr.iter().map(|e| &e.expr))
    {
        let column = expr.downcast_ref::<Column>().expect("validated column");
        bytes = add(bytes, mul(encoded_bound(batch.column(column.index()))?, 4)?)?;
    }
    // Reallocating reused Rows can temporarily retain both allocations.
    bytes = add(
        bytes,
        mul(
            add(
                state.scratch_rows.size(),
                state.partition_scratch_rows.size(),
            )?,
            2,
        )?,
    )?;
    add(bytes, 4096)
}

pub(super) fn reserve(state: &mut PartitionedTopKRank, extra: usize) -> Result<()> {
    let target = add(state.size(), extra)?;
    state
        .reservation
        .try_resize(target.max(state.reservation.size()))
}

pub(super) fn partition_bound(
    capacity: usize,
    existing: Option<&RankPartitionState>,
    key_len: usize,
    indices: &[u32],
    k: usize,
    input_bytes: usize,
) -> Result<usize> {
    let mut bytes = add(key_len, mul(input_bytes, 2)?)?;
    // HashMap growth: old + new tables, bucket padding and control bytes.
    bytes = add(
        bytes,
        mul(
            add(capacity, 4)?,
            4 * (size_of::<(OwnedRow, RankPartitionState)>() + 16),
        )?,
    )?;
    // Only reference metadata is duplicated by heap eviction/tie growth.
    // The operator has already reserved the union of all old/input arrays.
    // At most K-1 existing heap rows become new tie entries in this group.
    let mut largest_batch = input_bytes;
    if let Some(state) = existing {
        for entry in state.heap.store.batches.values() {
            largest_batch = largest_batch.max(batch_reference_bytes(&entry.batch));
        }
    }
    bytes = add(
        bytes,
        mul(largest_batch, k.saturating_sub(1).min(indices.len()))?,
    )?;
    bytes = add(bytes, mul(indices.len(), 512)?)?;
    // Existing allocations can double during growth; key replacement can
    // keep old capacity even when the encoded replacement is shorter.
    if let Some(state) = existing {
        bytes = add(
            bytes,
            mul(state.heap.size() - state.heap.store.batches_size, 4)?,
        )?;
        bytes = add(
            bytes,
            mul(state.ties.capacity(), 4 * size_of::<TieEntry>())?,
        )?;
    }
    Ok(bytes)
}

/// Metadata allocated by a RecordBatch clone. Array objects and their buffers
/// are shared and are accounted separately across the entire rank operator.
pub(super) fn batch_reference_bytes(batch: &RecordBatch) -> usize {
    size_of::<RecordBatch>() + batch.num_columns() * size_of::<ArrayRef>()
}

/// View buffer descriptor lists are heap allocations too. Arrow's array size
/// currently omits them; include them when bounding our temporary accountant.
fn visit_view_buffers(
    array: &dyn Array,
    visit: &mut dyn FnMut(&[arrow::buffer::Buffer]) -> Result<()>,
) -> Result<()> {
    use arrow::datatypes::{DataType, Int16Type, Int32Type, Int64Type};
    let list = match array.data_type() {
        DataType::Utf8View => Some(array.as_string_view().data_buffers()),
        DataType::BinaryView => Some(array.as_binary_view().data_buffers()),
        _ => None,
    };
    if let Some(list) = list {
        return visit(list);
    }
    match array.data_type() {
        DataType::List(_) => {
            visit_view_buffers(array.as_list::<i32>().values().as_ref(), visit)
        }
        DataType::LargeList(_) => {
            visit_view_buffers(array.as_list::<i64>().values().as_ref(), visit)
        }
        DataType::ListView(_) => {
            visit_view_buffers(array.as_list_view::<i32>().values().as_ref(), visit)
        }
        DataType::LargeListView(_) => {
            visit_view_buffers(array.as_list_view::<i64>().values().as_ref(), visit)
        }
        DataType::FixedSizeList(_, _) => {
            visit_view_buffers(array.as_fixed_size_list().values().as_ref(), visit)
        }
        DataType::Struct(_) => array
            .as_struct()
            .columns()
            .iter()
            .try_for_each(|a| visit_view_buffers(a.as_ref(), visit)),
        DataType::Map(_, _) => visit_view_buffers(array.as_map().entries(), visit),
        DataType::Union(fields, _) => fields.iter().try_for_each(|(id, _)| {
            visit_view_buffers(array.as_union().child(id).as_ref(), visit)
        }),
        DataType::Dictionary(_, _) => {
            visit_view_buffers(array.as_any_dictionary().values().as_ref(), visit)
        }
        DataType::RunEndEncoded(ends, _) => match ends.data_type() {
            DataType::Int16 => {
                visit_view_buffers(array.as_run::<Int16Type>().values().as_ref(), visit)
            }
            DataType::Int32 => {
                visit_view_buffers(array.as_run::<Int32Type>().values().as_ref(), visit)
            }
            DataType::Int64 => {
                visit_view_buffers(array.as_run::<Int64Type>().values().as_ref(), visit)
            }
            _ => datafusion_common::internal_err!("invalid rank run-end type"),
        },
        _ => Ok(()),
    }
}

fn view_metadata(array: &dyn Array) -> Result<usize> {
    let mut bytes = 0;
    visit_view_buffers(array, &mut |list| {
        bytes = add(bytes, descriptor_bytes(list)?)?;
        Ok(())
    })?;
    Ok(bytes)
}

fn descriptor_bytes(list: &[arrow::buffer::Buffer]) -> Result<usize> {
    add(
        mul(list.len(), size_of::<arrow::buffer::Buffer>())?,
        2 * size_of::<usize>(),
    )
}

fn array_metadata(array: &dyn Array) -> usize {
    // Byte-view size methods scan every backing buffer. The object is fixed
    // size; account its shared descriptor allocation separately instead.
    match array.data_type() {
        DataType::Utf8View => size_of::<arrow::array::StringViewArray>(),
        DataType::BinaryView => size_of::<arrow::array::BinaryViewArray>(),
        _ => array
            .get_array_memory_size()
            .saturating_sub(array.get_buffer_memory_size()),
    }
}

pub(super) fn counter_workspace(
    retained_metadata: usize,
    batch: &RecordBatch,
) -> Result<usize> {
    let incoming = batch.columns().iter().try_fold(0, |n, a| {
        add(
            n,
            add(array_metadata(a.as_ref()), view_metadata(a.as_ref())?)?,
        )
    })?;
    // Descriptor copies, identity hash tables and geometric table growth. Payload
    // bytes are never copied by this accountant. The fixed slack covers empty
    // tables and primitive ArrayData headers.
    add(mul(add(retained_metadata, incoming)?, 8)?, 4096)
}

#[derive(Default)]
pub(super) struct ArrayBytes {
    pub bytes: usize,
    pub metadata: usize,
    pub counter_metadata: usize,
}

#[derive(Default)]
struct Counter {
    // Address identities are used only while all visited arrays are borrowed
    // from live retained/input batches. No pointer is dereferenced or cached
    // across calls, so freed allocations cannot alias an ever-seen entry.
    arrays: datafusion_common::HashSet<usize>,
    buffers: HashMap<usize, usize>,
    descriptors: datafusion_common::HashSet<usize>,
    max_descriptor_copy: usize,
    size: ArrayBytes,
}

impl Counter {
    fn buffer(&mut self, buffer: &arrow::buffer::Buffer) -> Result<()> {
        let id = buffer.data_ptr().as_ptr().addr();
        // Foreign-owned buffers report zero capacity. Their visible allocation
        // extent (including slice offset) must still be charged.
        let bytes = buffer
            .capacity()
            .max(add(buffer.ptr_offset(), buffer.len())?);
        // Include the allocation's shared Bytes/Arc ownership bookkeeping,
        // not only its payload capacity. This bounds the pinned Arrow layout
        // with or without its optional pool reservation. Empty buffers have no
        // unique payload address, so conservatively charge their owners per use.
        let ownership = 2 * size_of::<arrow::buffer::Buffer>() + 8 * size_of::<usize>();
        if bytes == 0 {
            self.size.metadata = add(self.size.metadata, ownership)?;
            self.size.bytes = add(self.size.bytes, ownership)?;
            return Ok(());
        }
        let old = match self.buffers.entry(id) {
            hashbrown::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hashbrown::hash_map::Entry::Vacant(entry) => {
                self.size.metadata = add(self.size.metadata, ownership)?;
                self.size.bytes = add(self.size.bytes, ownership)?;
                entry.insert(0)
            }
        };
        if bytes > *old {
            self.size.bytes = add(self.size.bytes, bytes - *old)?;
            *old = bytes;
        }
        Ok(())
    }

    fn data(&mut self, data: &arrow::array::ArrayData) -> Result<()> {
        for buffer in data.buffers() {
            self.buffer(buffer)?;
        }
        if let Some(nulls) = data.nulls() {
            self.buffer(nulls.inner().inner())?;
        }
        for child in data.child_data() {
            self.data(child)?;
        }
        Ok(())
    }

    fn view(
        &mut self,
        views: &arrow::buffer::ScalarBuffer<u128>,
        buffers: &[arrow::buffer::Buffer],
        nulls: Option<&arrow::buffer::NullBuffer>,
    ) -> Result<usize> {
        self.buffer(views.inner())?;
        if let Some(nulls) = nulls {
            self.buffer(nulls.inner().inner())?;
        }
        if buffers.is_empty() {
            // Empty lists may share a dangling address without sharing owners.
            return descriptor_bytes(buffers);
        }
        if !self.descriptors.insert(buffers.as_ptr().addr()) {
            return Ok(0);
        }
        for buffer in buffers {
            self.buffer(buffer)?;
        }
        descriptor_bytes(buffers)
    }

    fn batch(&mut self, batch: &RecordBatch) -> Result<()> {
        for array in batch.columns() {
            let id = Arc::as_ptr(array).cast::<()>().addr();
            if !self.arrays.insert(id) {
                continue;
            }
            let mut metadata =
                add(array_metadata(array.as_ref()), 2 * size_of::<usize>())?;
            match array.data_type() {
                // Avoid copying/scanning the same shared descriptor list for
                // every slice. Only each slice's views and validity can differ.
                DataType::Utf8View => {
                    let view = array.as_string_view();
                    metadata = add(
                        metadata,
                        self.view(view.views(), view.data_buffers(), view.nulls())?,
                    )?;
                }
                DataType::BinaryView => {
                    let view = array.as_binary_view();
                    metadata = add(
                        metadata,
                        self.view(view.views(), view.data_buffers(), view.nulls())?,
                    )?;
                }
                _ => {
                    // ArrayData's descriptors are temporary, not retained
                    // storage. Bound the largest simultaneously copied list.
                    self.max_descriptor_copy =
                        self.max_descriptor_copy.max(view_metadata(array.as_ref())?);
                    visit_view_buffers(array.as_ref(), &mut |list| {
                        if list.is_empty()
                            || self.descriptors.insert(list.as_ptr().addr())
                        {
                            metadata = add(metadata, descriptor_bytes(list)?)?;
                        }
                        Ok(())
                    })?;
                    self.data(&array.to_data())?;
                }
            }
            self.size.metadata = add(self.size.metadata, metadata)?;
            self.size.bytes = add(self.size.bytes, metadata)?;
        }
        Ok(())
    }
}

pub(super) fn retained_arrays(
    state: &PartitionedTopKRank,
    incoming: Option<&RecordBatch>,
) -> Result<ArrayBytes> {
    let mut counter = Counter::default();
    for partition in state.states.values() {
        for entry in partition.heap.store.batches.values() {
            counter.batch(&entry.batch)?;
        }
        for entry in &partition.ties {
            counter.batch(&entry.batch)?;
        }
    }
    if let Some(batch) = incoming {
        counter.batch(batch)?;
    }
    counter.size.counter_metadata =
        add(counter.size.metadata, counter.max_descriptor_copy)?;
    Ok(counter.size)
}

#[cfg(test)]
pub(super) fn batch_bytes(batch: &RecordBatch) -> usize {
    let mut counter = Counter::default();
    counter.batch(batch).expect("representable rank batch");
    counter
        .size
        .bytes
        .saturating_add(batch_reference_bytes(batch))
}

/// New flat output arrays copy logical values. Repeated source buffers do not
/// reduce these allocations (e.g. sixteen references to one long UTF-8 row).
pub(super) fn output_row_bytes(batch: &RecordBatch, row: usize) -> Result<usize> {
    use arrow::datatypes::DataType;
    batch.columns().iter().try_fold(0, |n, array| {
        let bytes = match array.data_type() {
            DataType::Utf8 => {
                add(array.as_string::<i32>().value_length(row) as usize, 4)?
            }
            DataType::LargeUtf8 => {
                add(array.as_string::<i64>().value_length(row) as usize, 8)?
            }
            DataType::Binary => {
                add(array.as_binary::<i32>().value_length(row) as usize, 4)?
            }
            DataType::LargeBinary => {
                add(array.as_binary::<i64>().value_length(row) as usize, 8)?
            }
            DataType::FixedSizeBinary(n) => *n as usize,
            DataType::Boolean | DataType::Null => 1,
            kind => kind.primitive_width().ok_or_else(|| {
                datafusion_common::internal_datafusion_err!("non-flat rank output")
            })?,
        };
        add(n, add(bytes, 1)?)
    })
}

pub(super) fn table_bytes(capacity: usize) -> usize {
    capacity.saturating_mul(2 * (size_of::<(OwnedRow, RankPartitionState)>() + 1))
}
