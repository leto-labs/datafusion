//! Bounded rank emission. No full peer-set materialization or output queue.
use super::*;
use rank_memory::{add, mul};

struct Partition {
    key: OwnedRow,
    state: RankPartitionState,
}

impl PartialEq for Partition {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Partition {}
impl PartialOrd for Partition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Partition {
    fn cmp(&self, other: &Self) -> Ordering {
        other.key.cmp(&self.key)
    }
}

struct Current {
    rows: Vec<TopKRow>,
    row: usize,
    tie: usize,
    index: usize,
    partition: Partition,
}

struct Emitter {
    partitions: BinaryHeap<Partition>,
    current: Option<Current>,
    initialized: bool,
    copy_flat: bool,
    // Last: owns the reservation until all emission buffers have dropped.
    owner: PartitionedTopKRank,
}

pub(super) fn emit(mut owner: PartitionedTopKRank) -> Result<SendableRecordBatchStream> {
    use arrow::datatypes::DataType;
    let copy_flat = owner.schema.fields().iter().all(|field| {
        let kind = field.data_type();
        kind.is_primitive()
            || matches!(
                kind,
                DataType::Null
                    | DataType::Boolean
                    | DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Binary
                    | DataType::LargeBinary
                    | DataType::FixedSizeBinary(_)
            )
    });
    let mut largest = 0;
    let mut max_rows = 0;
    for state in owner.states.values() {
        let mut heap_bytes = 0;
        max_rows = max_rows.max(state.heap.inner.len().min(owner.batch_size));
        for entry in state.heap.store.batches.values() {
            heap_bytes = add(heap_bytes, rank_memory::batch_bytes(&entry.batch))?;
        }
        largest = largest.max(heap_bytes);
        for entry in &state.ties {
            largest = largest.max(entry.batch_bytes);
            max_rows = max_rows.max(entry.row_indices.len().min(owner.batch_size));
        }
    }
    // Both the consumed hash table and destination heap can be live while
    // building the ordering. For flat payloads materialize only the delivered
    // chunk so small outputs do not pin large source buffers downstream. Nested
    // payloads use bounded slices: copying arbitrary list/view selections can
    // amplify logical data beyond the physical source-buffer bound.
    let ordering = mul(owner.states.len(), size_of::<Partition>())?;
    let refs = mul(max_rows, add(mul(owner.schema.fields().len(), 256)?, 64)?)?;
    let output = add(mul(largest, 2)?, refs)?;
    rank_memory::reserve(&mut owner, add(add(ordering, output)?, 4096)?)?;
    let schema = Arc::clone(&owner.schema);
    let emitter = Emitter {
        partitions: BinaryHeap::with_capacity(owner.states.len()),
        current: None,
        initialized: false,
        copy_flat,
        owner,
    };
    let stream = futures::stream::try_unfold(emitter, |mut emitter| async move {
        // A source that is immediately ready must still permit cancellation.
        tokio::task::yield_now().await;
        if !emitter.initialized {
            let states = std::mem::take(&mut emitter.owner.states);
            for (i, (key, state)) in states.into_iter().enumerate() {
                emitter.partitions.push(Partition { key, state });
                if i % 64 == 63 {
                    tokio::task::yield_now().await;
                }
            }
            emitter.initialized = true;
        }
        loop {
            if emitter.current.is_none() {
                let Some(mut partition) = emitter.partitions.pop() else {
                    return Ok(None);
                };
                let rows =
                    std::mem::take(&mut partition.state.heap.inner).into_sorted_vec();
                emitter.current = Some(Current {
                    rows,
                    row: 0,
                    tie: 0,
                    index: 0,
                    partition,
                });
            }
            let baseline = emitter.owner.metrics.baseline.clone();
            let timer = baseline.elapsed_compute().timer();
            let current = emitter
                .current
                .as_mut()
                .expect("installed current partition");
            let batch_size = emitter.owner.batch_size;
            let batch = if current.row < current.rows.len() {
                let batch = if emitter.copy_flat {
                    let count = batch_size.min(current.rows.len() - current.row);
                    let rows = &current.rows[current.row..current.row + count];
                    let batches: Vec<_> = rows
                        .iter()
                        .map(|row| {
                            &current
                                .partition
                                .state
                                .heap
                                .store
                                .get(row.batch_id)
                                .expect("retained heap batch")
                                .batch
                        })
                        .collect();
                    let indices: Vec<_> = rows
                        .iter()
                        .enumerate()
                        .map(|(i, row)| (i, row.index))
                        .collect();
                    let batch = interleave_record_batch(&batches, &indices)?;
                    current.row += count;
                    batch
                } else {
                    let first = &current.rows[current.row];
                    let mut count = 1;
                    while count < batch_size && current.row + count < current.rows.len() {
                        let next = &current.rows[current.row + count];
                        if next.batch_id != first.batch_id
                            || next.index != first.index + count
                        {
                            break;
                        }
                        count += 1;
                    }
                    let source = &current
                        .partition
                        .state
                        .heap
                        .store
                        .get(first.batch_id)
                        .expect("retained heap batch")
                        .batch;
                    let batch = source.slice(first.index, count);
                    current.row += count;
                    batch
                };
                Some(batch)
            } else if let Some(tie) = current.partition.state.ties.get(current.tie) {
                let start = tie.row_indices[current.index] as usize;
                let mut count = if emitter.copy_flat {
                    batch_size.min(tie.row_indices.len() - current.index)
                } else {
                    1
                };
                while !emitter.copy_flat
                    && count < batch_size
                    && current.index + count < tie.row_indices.len()
                    && tie.row_indices[current.index + count] as usize == start + count
                {
                    count += 1;
                }
                let batch = if emitter.copy_flat {
                    let indices = UInt32Array::from_iter_values(
                        tie.row_indices[current.index..current.index + count]
                            .iter()
                            .copied(),
                    );
                    take_record_batch(&tie.batch, &indices)?
                } else {
                    tie.batch.slice(start, count)
                };
                current.index += count;
                if current.index == tie.row_indices.len() {
                    current.tie += 1;
                    current.index = 0;
                }
                Some(batch)
            } else {
                None
            };
            drop(timer);
            if let Some(batch) = batch {
                (&batch).record_output(&emitter.owner.metrics.baseline);
                return Ok(Some((batch, emitter)));
            }
            emitter.current = None;
            tokio::task::yield_now().await;
        }
    });
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
}
