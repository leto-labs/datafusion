// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;

// Bytes and Arc bookkeeping, including an optional Arrow pool owner. Eight
// pointer-sized fields are a conservative owner-header bound, not a bit count.
const BUFFER_OWNER_BYTES: usize = 2 * size_of::<Buffer>() + size_of::<[usize; 8]>();

/// Refcounted retained-batch accounting, including Arrow ownership metadata.
///
/// Unlike the buffer-only tracker this includes array Arc headers, child
/// reference tables, view-buffer descriptor lists, and buffer owners. Unique
/// payload allocations and shared descriptor lists are charged once. Match
/// each count with an uncount while the batch is still alive.
#[derive(Debug, Default)]
pub struct RetainedBatchMemoryTracker {
    buffers: BufferIdMap,
    arrays: HashMap<usize, usize>,
    descriptors: BufferIdMap,
    memory_usage: usize,
}

impl RetainedBatchMemoryTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retain a batch and return the additional held bytes.
    /// Batch/column reference storage is charged per batch; array and buffer
    /// ownership is deduplicated across batches.
    pub fn count_batch(&mut self, batch: &RecordBatch) -> usize {
        let before = self.memory_usage;
        self.memory_usage = self
            .memory_usage
            .saturating_add(batch_reference_size(batch));
        for array in batch.columns() {
            visit_array_buffers(array.as_ref(), self);
            self.metadata(array, true);
        }
        self.memory_usage.saturating_sub(before)
    }

    /// Additional retained bytes for a batch, without changing this tracker.
    /// The scratch accountant visits only the incoming batch, not prior state.
    pub fn additional_batch_memory(&self, batch: &RecordBatch) -> usize {
        let mut added = Self::new();
        added.memory_usage = batch_reference_size(batch);
        for array in batch.columns() {
            visit_array_buffers(
                array.as_ref(),
                &mut AdditionalBuffers {
                    added: &mut added,
                    held: self,
                },
            );
            added.metadata_against(array, true, Some(self));
        }
        added.memory_usage
    }

    /// Release a still-live batch, returning the bytes no retained batch owns.
    pub fn uncount_batch(&mut self, batch: &RecordBatch) -> usize {
        let before = self.memory_usage;
        self.memory_usage -= batch_reference_size(batch);
        for array in batch.columns() {
            visit_array_buffers(array.as_ref(), &mut ReleaseBuffers(self));
            self.metadata(array, false);
        }
        before - self.memory_usage
    }

    /// Bytes held by the tracked batches, excluding this accountant's tables.
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }

    /// Heap allocation retained by the accountant itself.
    pub fn allocated_size(&self) -> usize {
        map_allocation(&self.arrays)
            .saturating_add(buffer_map_allocation(&self.buffers))
            .saturating_add(buffer_map_allocation(&self.descriptors))
    }

    fn metadata(&mut self, array: &ArrayRef, retain: bool) {
        self.metadata_against(array, retain, None);
    }
    fn metadata_against(&mut self, array: &ArrayRef, retain: bool, held: Option<&Self>) {
        let id = Arc::as_ptr(array).cast::<()>() as usize;
        let first_or_last = if retain {
            let count = self.arrays.entry(id).or_default();
            *count = count.saturating_add(1);
            *count == 1
        } else {
            let count = self.arrays.get_mut(&id).expect("matched retained batch");
            *count -= 1;
            if *count == 0 {
                self.arrays.remove(&id);
                true
            } else {
                false
            }
        };
        let children = array_children(array);
        if first_or_last && !held.is_some_and(|held| held.arrays.contains_key(&id)) {
            let bytes = array_object_size(array, &children);
            if retain {
                self.memory_usage = self.memory_usage.saturating_add(bytes);
            } else {
                self.memory_usage -= bytes;
            }
        }
        let descriptors = match array.data_type() {
            DataType::Utf8View => Some(array.as_string_view().data_buffers()),
            DataType::BinaryView => Some(array.as_binary_view().data_buffers()),
            _ => None,
        };
        if let Some(list) = descriptors.filter(|list| {
            !list.is_empty()
                && !held.is_some_and(|held| {
                    counted(
                        &held.descriptors,
                        NonZero::new(list.as_ptr() as usize).unwrap(),
                    ) > 0
                })
        }) {
            let id =
                NonZero::new(list.as_ptr() as usize).expect("nonempty descriptor list");
            if retain {
                let bytes = list
                    .len()
                    .saturating_mul(size_of::<Buffer>())
                    .saturating_add(2 * size_of::<usize>());
                self.memory_usage = self
                    .memory_usage
                    .saturating_add(self.descriptors.increment(id, bytes));
            } else {
                self.memory_usage -= self.descriptors.decrement(id);
            }
        }
        // Walk even shared parents: reference counts must balance each batch.
        for child in children {
            self.metadata_against(child, retain, held);
        }
    }
}

impl BufferVisitor for RetainedBatchMemoryTracker {
    fn visit(&mut self, buffer: &Buffer) {
        let owner = BUFFER_OWNER_BYTES;
        let visible = buffer
            .capacity()
            .max(buffer.ptr_offset().saturating_add(buffer.len()));
        if visible == 0 {
            // Empty allocations may have the same dangling data address but
            // different owners. Charge their headers per retained reference.
            self.memory_usage = self.memory_usage.saturating_add(owner);
        } else {
            self.memory_usage = self.memory_usage.saturating_add(
                self.buffers
                    .increment(buffer.data_ptr().addr(), visible.saturating_add(owner)),
            );
        }
    }
}

fn counted(map: &BufferIdMap, id: NonZero<usize>) -> usize {
    if let Some(map) = &map.overflow {
        return map.get(&id).map_or(0, |(_, bytes)| *bytes);
    }
    map.ids[..map.len]
        .iter()
        .position(|item| *item == Some(id))
        .map_or(0, |i| map.capacities[i])
}
struct AdditionalBuffers<'a> {
    added: &'a mut RetainedBatchMemoryTracker,
    held: &'a RetainedBatchMemoryTracker,
}
impl BufferVisitor for AdditionalBuffers<'_> {
    fn visit(&mut self, buffer: &Buffer) {
        let visible = buffer
            .capacity()
            .max(buffer.ptr_offset().saturating_add(buffer.len()));
        if visible == 0 {
            self.added.memory_usage =
                self.added.memory_usage.saturating_add(BUFFER_OWNER_BYTES);
            return;
        }
        let id = buffer.data_ptr().addr();
        let size = visible.saturating_add(BUFFER_OWNER_BYTES);
        let previous =
            counted(&self.held.buffers, id).max(counted(&self.added.buffers, id));
        self.added.buffers.increment(id, size);
        self.added.memory_usage = self
            .added
            .memory_usage
            .saturating_add(size.saturating_sub(previous));
    }
}

struct ReleaseBuffers<'a>(&'a mut RetainedBatchMemoryTracker);
impl BufferVisitor for ReleaseBuffers<'_> {
    fn visit(&mut self, buffer: &Buffer) {
        let visible = buffer
            .capacity()
            .max(buffer.ptr_offset().saturating_add(buffer.len()));
        let bytes = if visible == 0 {
            BUFFER_OWNER_BYTES
        } else {
            self.0.buffers.decrement(buffer.data_ptr().addr())
        };
        self.0.memory_usage -= bytes;
    }
}

fn batch_reference_size(batch: &RecordBatch) -> usize {
    size_of::<RecordBatch>()
        .saturating_add(batch.num_columns().saturating_mul(size_of::<ArrayRef>()))
}

fn array_object_size(array: &ArrayRef, children: &[&ArrayRef]) -> usize {
    // The trait object's concrete header excludes recursively shared children.
    // Arrow's aggregate array-size methods recurse into those children and
    // would both duplicate their owners and overflow on deeply shared trees.
    let own = size_of_val(array.as_ref());
    let references = match array.data_type() {
        DataType::Union(fields, _) => fields
            .iter()
            .map(|(id, _)| id as usize + 1)
            .max()
            .unwrap_or(0)
            .saturating_mul(size_of::<Option<ArrayRef>>()),
        _ => children.len().saturating_mul(size_of::<ArrayRef>()),
    };
    own.saturating_add(references)
        .saturating_add(2 * size_of::<usize>())
}

fn map_allocation<K: Eq + std::hash::Hash, V>(map: &HashMap<K, V>) -> usize {
    map.allocation_size()
}
fn buffer_map_allocation(map: &BufferIdMap) -> usize {
    map.overflow.as_ref().map_or(0, map_allocation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{NullArray, StringViewArray, StructArray, UnionArray};
    use arrow::datatypes::{Field, Fields, UnionFields};

    #[test]
    fn retained_nested_references_and_shared_child_owners() {
        let child: ArrayRef = Arc::new(NullArray::new(1));
        let fields: Fields = (0..4)
            .map(|i| Field::new(format!("n{i}"), DataType::Null, true))
            .collect();
        let array: ArrayRef =
            Arc::new(StructArray::new(fields, vec![Arc::clone(&child); 4], None));
        let batch = RecordBatch::try_from_iter([("s", array)]).unwrap();
        let expected = size_of::<RecordBatch>()
            + size_of::<ArrayRef>()
            + size_of::<StructArray>()
            + 2 * size_of::<usize>()
            + 4 * size_of::<ArrayRef>()
            + size_of::<NullArray>()
            + 2 * size_of::<usize>();
        let mut tracker = RetainedBatchMemoryTracker::new();
        assert_eq!(tracker.count_batch(&batch), expected);
        assert_eq!(tracker.uncount_batch(&batch), expected);
        assert_eq!(tracker.memory_usage(), 0);
    }

    #[test]
    fn retained_sparse_union_covers_holes() {
        let fields =
            UnionFields::try_new([127], [Field::new("n", DataType::Null, true)]).unwrap();
        let array: ArrayRef = Arc::new(
            UnionArray::try_new(
                fields,
                vec![127_i8].into(),
                None,
                vec![Arc::new(NullArray::new(1))],
            )
            .unwrap(),
        );
        let batch = RecordBatch::try_from_iter([("u", array)]).unwrap();
        let mut tracker = RetainedBatchMemoryTracker::new();
        let bytes = tracker.count_batch(&batch);
        assert!(bytes >= 128 * size_of::<Option<ArrayRef>>() + size_of::<UnionArray>());
        assert_eq!(tracker.uncount_batch(&batch), bytes);
    }

    #[test]
    fn retained_shared_views_release_only_the_last_owner() {
        let values: ArrayRef =
            Arc::new(StringViewArray::from(vec!["x".repeat(4096); 10]));
        let batch = RecordBatch::try_from_iter([("v", values)]).unwrap();
        let second = batch.slice(1, 2);
        let mut tracker = RetainedBatchMemoryTracker::new();
        let first = tracker.count_batch(&batch);
        let extra = tracker.count_batch(&second);
        assert!(
            extra < 1024,
            "shared payload or descriptor list charged again"
        );
        let released = tracker.uncount_batch(&batch);
        assert!(released < 1024);
        assert!(tracker.memory_usage() >= 10 * 4096);
        assert_eq!(tracker.uncount_batch(&second), first + extra - released);
        assert_eq!(tracker.memory_usage(), 0);
    }
    #[test]
    fn retained_preview_matches_increment_and_preserves_existing_charge() {
        let values: ArrayRef =
            Arc::new(StringViewArray::from(vec!["x".repeat(4096); 10]));
        let batch = RecordBatch::try_from_iter([("v", values)]).unwrap();
        let mut tracker = RetainedBatchMemoryTracker::new();
        for batch in [batch.clone(), batch.slice(1, 3), batch.clone()] {
            let before = tracker.memory_usage();
            let preview = tracker.additional_batch_memory(&batch);
            assert_eq!(tracker.memory_usage(), before);
            assert_eq!(tracker.count_batch(&batch), preview);
        }
    }
}
