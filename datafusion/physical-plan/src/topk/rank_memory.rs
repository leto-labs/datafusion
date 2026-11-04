//! Conservative, checked working bounds for the opt-in native rank path.
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
fn encoded_bound(array: &ArrayRef) -> Result<usize> {
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
    // All row/entry Vecs may grow geometrically. At an unchanged boundary
    // at most K-1 heap rows can be evicted into ties; a boundary change clears
    // those ties. Include their duplicated batch references conservatively.
    let mut largest_batch = input_bytes;
    if let Some(state) = existing {
        for entry in state.heap.store.batches.values() {
            largest_batch = largest_batch.max(batch_bytes(&entry.batch));
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
        bytes = add(bytes, mul(state.heap.size(), 4)?)?;
        bytes = add(
            bytes,
            mul(state.ties.capacity(), 4 * size_of::<TieEntry>())?,
        )?;
    }
    Ok(bytes)
}

// Avoid RecordBatchMemoryCounter's temporary HashSet while sizing. Counting
// shared buffers repeatedly is conservative and includes array metadata.
pub(super) fn batch_bytes(batch: &RecordBatch) -> usize {
    batch.columns().iter().fold(
        size_of::<RecordBatch>()
            + batch.num_columns().saturating_mul(size_of::<ArrayRef>()),
        |total, array| total.saturating_add(array.get_array_memory_size()),
    )
}

pub(super) fn table_bytes(capacity: usize) -> usize {
    capacity.saturating_mul(2 * (size_of::<(OwnedRow, RankPartitionState)>() + 1))
}
