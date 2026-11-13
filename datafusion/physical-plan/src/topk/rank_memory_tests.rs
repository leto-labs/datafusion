// Modified by the Longbow authors; see third_party/README.md.

//! Retained rank peers must remain charged through paused emission and drop.
use super::*;
use arrow::array::{BinaryArray, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_execution::memory_pool::FairSpillPool;
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_physical_expr::expressions::col;
use futures::StreamExt;

fn fixture(
    pool: usize,
    batch_size: usize,
) -> (SchemaRef, Arc<RuntimeEnv>, PartitionedTopKRank) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(FairSpillPool::new(pool)))
        .build_arc()
        .unwrap();
    let state = PartitionedTopKRank::try_new(
        0,
        Arc::clone(&schema),
        vec![col("k", &schema).unwrap()],
        vec![SortField::new(DataType::Utf8)],
        LexOrdering::new([PhysicalSortExpr::new_default(col("v", &schema).unwrap())])
            .unwrap(),
        1,
        batch_size,
        &runtime,
        &ExecutionPlanMetricsSet::new(),
    )
    .unwrap();
    (schema, runtime, state)
}

fn batch(schema: SchemaRef, key: &str, rows: usize, width: usize) -> RecordBatch {
    let payload = "x".repeat(width);
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![key; rows])),
            Arc::new(Int64Array::from(vec![1; rows])),
            Arc::new(StringArray::from(vec![payload.as_str(); rows])),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn rank_paused_tie_emission_keeps_reservation_until_drop() -> Result<()> {
    let (schema, runtime, mut state) = fixture(64 * 1024 * 1024, 4096);
    for _ in 0..24 {
        state.insert_batch(&batch(Arc::clone(&schema), "same", 4096, 256))?;
    }
    let mut stream = state.emit()?;
    let first = stream.next().await.unwrap()?;
    assert!(first.num_rows() <= 4096);
    drop(first);
    assert!(
        runtime.memory_pool.reserved() >= 20 * 1024 * 1024,
        "paused peers are uncharged: {}",
        runtime.memory_pool.reserved()
    );
    drop(stream);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn rank_multiple_paused_streams_share_a_bounded_pool() -> Result<()> {
    let (schema, runtime, _) = fixture(64 * 1024 * 1024, 4096);
    let mut streams = Vec::new();
    let mut refused = false;
    for id in 0..4 {
        let mut state = PartitionedTopKRank::try_new(
            id,
            Arc::clone(&schema),
            vec![col("k", &schema)?],
            vec![SortField::new(DataType::Utf8)],
            LexOrdering::new([PhysicalSortExpr::new_default(col("v", &schema)?)])
                .unwrap(),
            1,
            4096,
            &runtime,
            &ExecutionPlanMetricsSet::new(),
        )?;
        for _ in 0..24 {
            if let Err(error) =
                state.insert_batch(&batch(Arc::clone(&schema), "same", 4096, 256))
            {
                assert!(error.to_string().contains("Resources exhausted"), "{error}");
                refused = true;
                break;
            }
        }
        if refused {
            drop(state);
            break;
        }
        let mut stream = match state.emit() {
            Ok(stream) => stream,
            Err(error) => {
                assert!(error.to_string().contains("Resources exhausted"), "{error}");
                refused = true;
                break;
            }
        };
        drop(stream.next().await.unwrap()?);
        streams.push(stream);
    }
    assert!(
        refused,
        "four paused streams retained >64 MiB without refusal"
    );
    drop(streams);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[test]
fn rank_owned_long_partition_keys_are_charged() -> Result<()> {
    let (schema, runtime, mut state) = fixture(16 * 1024 * 1024, 1);
    for i in 0..32 {
        let key = format!("{i:08}{}", "k".repeat(32 * 1024));
        state.insert_batch(&batch(Arc::clone(&schema), &key, 1, 1))?;
    }
    let key_bytes: usize = state.states.keys().map(|key| key.as_ref().len()).sum();
    let retained: usize = state.states.values().map(|value| value.size()).sum();
    assert!(
        runtime.memory_pool.reserved() >= retained + key_bytes,
        "owned key bytes omitted: reserved={}, states={retained}, keys={key_bytes}",
        runtime.memory_pool.reserved()
    );
    drop(state);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[test]
fn rank_long_string_and_binary_keys_refuse_before_exhausting_pool() -> Result<()> {
    for binary in [false, true] {
        let (mut schema, runtime, initial) = fixture(4 * 1024 * 1024, 1);
        drop(initial);
        if binary {
            schema = Arc::new(Schema::new(vec![
                Field::new("k", DataType::Binary, false),
                Field::new("v", DataType::Int64, false),
                Field::new("payload", DataType::Utf8, false),
            ]));
        }
        let mut state = PartitionedTopKRank::try_new(
            0,
            Arc::clone(&schema),
            vec![col("k", &schema)?],
            vec![SortField::new(schema.field(0).data_type().clone())],
            LexOrdering::new([PhysicalSortExpr::new_default(col("v", &schema)?)])
                .unwrap(),
            1,
            1,
            &runtime,
            &ExecutionPlanMetricsSet::new(),
        )?;
        let mut refused = false;
        for i in 0..256 {
            let key = format!("{i:08}{}", "k".repeat(32 * 1024));
            let input = if binary {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(BinaryArray::from(vec![key.as_bytes()])),
                        Arc::new(Int64Array::from(vec![1])),
                        Arc::new(StringArray::from(vec!["x"])),
                    ],
                )?
            } else {
                batch(Arc::clone(&schema), &key, 1, 1)
            };
            if let Err(error) = state.insert_batch(&input) {
                assert!(error.to_string().contains("Resources exhausted"), "{error}");
                refused = true;
                break;
            }
            assert!(runtime.memory_pool.reserved() <= 4 * 1024 * 1024);
        }
        assert!(refused, "unbounded long keys: binary={binary}");
        drop(state);
        assert_eq!(runtime.memory_pool.reserved(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn rank_cancel_during_large_peer_emission_releases_reservations() -> Result<()> {
    let (schema, runtime, mut state) = fixture(64 * 1024 * 1024, 4096);
    for _ in 0..24 {
        state.insert_batch(&batch(Arc::clone(&schema), "same", 4096, 256))?;
    }
    let stream = state.emit()?;
    let (ready, received) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let mut stream = stream;
        drop(stream.next().await.unwrap().unwrap());
        ready.send(()).unwrap();
        // Retain the paused stream until the task is cancelled.
        futures::future::pending::<()>().await;
        drop(stream);
    });
    received.await.unwrap();
    assert!(runtime.memory_pool.reserved() >= 20 * 1024 * 1024);
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[test]
fn rank_checked_sizes_refuse_overflow_without_allocating() -> Result<()> {
    assert!(rank_memory::add(usize::MAX, 1).is_err());
    assert!(rank_memory::mul(usize::MAX, 2).is_err());
    assert_eq!(rank_memory::mul(usize::MAX, 0)?, 0);
    let (schema, runtime, initial) = fixture(1024 * 1024, 1);
    drop(initial);
    for (k, batch_size) in [(1, usize::MAX), (usize::MAX, 1), (0, 1), (1, 0)] {
        let result = PartitionedTopKRank::try_new(
            0,
            Arc::clone(&schema),
            vec![col("k", &schema)?],
            vec![SortField::new(DataType::Utf8)],
            LexOrdering::new([PhysicalSortExpr::new_default(col("v", &schema)?)])
                .unwrap(),
            k,
            batch_size,
            &runtime,
            &ExecutionPlanMetricsSet::new(),
        );
        assert!(result.is_err());
        assert_eq!(runtime.memory_pool.reserved(), 0);
    }
    Ok(())
}

#[test]
fn rank_emit_refusal_releases_retained_buffers_and_credits() -> Result<()> {
    let (schema, runtime, mut state) = fixture(16 * 1024 * 1024, 4096);
    let input = batch(schema, "same", 4096, 256);
    let weak = Arc::downgrade(input.column(2));
    state.insert_batch(&input)?;
    drop(input);
    let other = MemoryConsumer::new("other").register(&runtime.memory_pool);
    other.try_grow(16 * 1024 * 1024 - runtime.memory_pool.reserved())?;
    assert!(state.emit().is_err());
    assert!(
        weak.upgrade().is_none(),
        "failed emit retained source arrays"
    );
    drop(other);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[test]
fn rank_repeated_string_views_refuse_before_row_encoding() -> Result<()> {
    use arrow::array::StringViewArray;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(FairSpillPool::new(4 * 1024 * 1024)))
        .build_arc()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8View, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let mut state = PartitionedTopKRank::try_new(
        0,
        Arc::clone(&schema),
        vec![col("k", &schema)?],
        vec![SortField::new(DataType::Utf8View)],
        LexOrdering::new([PhysicalSortExpr::new_default(col("v", &schema)?)]).unwrap(),
        1,
        1024,
        &runtime,
        &ExecutionPlanMetricsSet::new(),
    )?;
    let key = "k".repeat(32 * 1024);
    let source = StringViewArray::from(vec![key.as_str()]);
    let views = arrow::compute::take(&source, &UInt32Array::from(vec![0; 1024]), None)?;
    let input = RecordBatch::try_new(
        schema,
        vec![views, Arc::new(Int64Array::from(vec![1; 1024]))],
    )?;
    assert!(rank_memory::batch_bytes(&input) < 256 * 1024);
    let before = (
        state.scratch_rows.size(),
        state.partition_scratch_rows.size(),
        state.reservation.size(),
    );
    let error = state.insert_batch(&input).unwrap_err();
    assert!(error.to_string().contains("Resources exhausted"), "{error}");
    assert_eq!(
        before,
        (
            state.scratch_rows.size(),
            state.partition_scratch_rows.size(),
            state.reservation.size()
        )
    );
    drop(state);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[test]
fn rank_source_id_overflow_keeps_partial_state_charged() -> Result<()> {
    let (schema, runtime, mut state) = fixture(1024 * 1024, 1);
    state.insert_batch(&batch(Arc::clone(&schema), "same", 1, 1))?;
    state.states.values_mut().next().unwrap().heap.store.next_id = u32::MAX;
    let input = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["same"])),
            Arc::new(Int64Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["x"])),
        ],
    )?;
    assert!(
        state
            .insert_batch(&input)
            .unwrap_err()
            .to_string()
            .contains("batch ID overflow")
    );
    assert!(runtime.memory_pool.reserved() >= state.size());
    drop(state);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn rank_view_payload_emission_keeps_all_peers_in_bounded_slices() -> Result<()> {
    use arrow::array::StringViewArray;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(FairSpillPool::new(4 * 1024 * 1024)))
        .build_arc()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
        Field::new("payload", DataType::Utf8View, false),
    ]));
    let mut state = PartitionedTopKRank::try_new(
        0,
        Arc::clone(&schema),
        vec![col("k", &schema)?],
        vec![SortField::new(DataType::Utf8)],
        LexOrdering::new([PhysicalSortExpr::new_default(col("v", &schema)?)]).unwrap(),
        1,
        7,
        &runtime,
        &ExecutionPlanMetricsSet::new(),
    )?;
    let payload = "x".repeat(32 * 1024);
    let source = StringViewArray::from(vec![payload.as_str()]);
    let views = arrow::compute::take(&source, &UInt32Array::from(vec![0; 1024]), None)?;
    let input = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["same"; 1024])),
            Arc::new(Int64Array::from(vec![1; 1024])),
            views,
        ],
    )?;
    state.insert_batch(&input)?;
    let mut stream = state.emit()?;
    let mut count = 0;
    while let Some(output) = stream.next().await {
        let output = output?;
        assert!(output.num_rows() <= 7);
        assert_eq!(output.column(2).as_string_view().value(0), payload);
        count += output.num_rows();
    }
    assert_eq!(count, 1024);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[test]
fn rank_shared_slices_charge_unique_buffers_across_heaps_and_ties() -> Result<()> {
    use datafusion_common::utils::memory::RecordBatchMemoryCounter;
    // A 16 MiB pool also bounds transient admission below twice the unique
    // approximately 8.1 MiB payload, not just the final retained charge.
    let (schema, runtime, mut state) = fixture(16 * 1024 * 1024, 16);
    let payload = "x".repeat(2048);
    let source = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(
                (0..4096)
                    .map(|i| if i % 2 == 0 { "a" } else { "b" })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(vec![1; 4096])),
            Arc::new(StringArray::from(vec![payload.as_str(); 4096])),
        ],
    )?;
    let mut unique = RecordBatchMemoryCounter::new();
    unique.count_batch(&source);
    let unique_bytes = unique.memory_usage();
    for i in 0..100 {
        state.insert_batch(&source.slice((i % 16) * 128, 128))?;
    }
    let indices: usize = state
        .states
        .values()
        .flat_map(|p| &p.ties)
        .map(|entry| entry.row_indices.capacity() * size_of::<u32>())
        .sum();
    let charged = runtime.memory_pool.reserved();
    println!("unique={unique_bytes}, retained_charge={charged}, pool_limit=16777216");
    assert!(
        charged >= unique_bytes + indices,
        "retained buffers/indices undercharged"
    );
    assert!(
        charged < unique_bytes * 2,
        "shared buffers multiplied: {charged} vs {unique_bytes}"
    );

    // Replacing both partition boundaries must release the old shared payload.
    let better = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Int64Array::from(vec![0, 0])),
            Arc::new(StringArray::from(vec!["new", "new"])),
        ],
    )?;
    state.insert_batch(&better)?;
    assert!(runtime.memory_pool.reserved() < 1024 * 1024);
    drop(state);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

fn repeated_heap_fixture(
    pool: usize,
) -> Result<(Arc<RuntimeEnv>, PartitionedTopKRank, RecordBatch)> {
    let (schema, runtime, mut state) = fixture(pool, 16);
    // The first 32 rows fill a heap from distinct batch references to one buffer.
    state.k = 32;
    let source = batch(schema, "same", 1, 512 * 1024);
    for _ in 0..64 {
        state.insert_batch(&source)?;
    }
    Ok((runtime, state, source))
}

#[tokio::test]
async fn rank_unique_storage_still_reserves_copied_output_growth() -> Result<()> {
    use datafusion_common::utils::memory::RecordBatchMemoryCounter;
    let (runtime, state, source) = repeated_heap_fixture(64 * 1024 * 1024)?;
    let mut source_memory = RecordBatchMemoryCounter::new();
    source_memory.count_batch(&source);
    let mut stream = state.emit()?;
    let output = stream.next().await.unwrap()?;
    assert_eq!(output.num_rows(), 16);
    let mut output_memory = RecordBatchMemoryCounter::new();
    output_memory.count_batch(&output);
    assert!(output_memory.memory_usage() >= 8 * 1024 * 1024);
    assert!(
        runtime.memory_pool.reserved()
            >= source_memory.memory_usage() + output_memory.memory_usage()
    );
    drop(output);
    drop(stream);
    assert_eq!(runtime.memory_pool.reserved(), 0);

    // Unique retained storage fits, but its materialized output does not.
    let (runtime, state, _source) = repeated_heap_fixture(4 * 1024 * 1024)?;
    assert!(state.emit().is_err());
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[test]
fn rank_shared_view_descriptors_are_not_per_slice_storage() -> Result<()> {
    use arrow::array::builder::StringViewBuilder;
    use datafusion_common::utils::memory::RecordBatchMemoryCounter;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
        Field::new("payload", DataType::Utf8View, false),
    ]));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(FairSpillPool::new(16 * 1024 * 1024)))
        .build_arc()?;
    let mut state = PartitionedTopKRank::try_new(
        0,
        Arc::clone(&schema),
        vec![col("k", &schema)?],
        vec![SortField::new(DataType::Utf8)],
        LexOrdering::new([PhysicalSortExpr::new_default(col("v", &schema)?)]).unwrap(),
        1,
        16,
        &runtime,
        &ExecutionPlanMetricsSet::new(),
    )?;
    let mut builder = StringViewBuilder::new().with_fixed_block_size(64);
    for _ in 0..4096 {
        builder.append_value("x".repeat(64));
    }
    let views = builder.finish();
    assert_eq!(views.data_buffers().len(), 4096);
    let descriptors = views.data_buffers().len() * size_of::<arrow::buffer::Buffer>();
    let source = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["same"; 4096])),
            Arc::new(Int64Array::from(vec![1; 4096])),
            Arc::new(views),
        ],
    )?;
    let mut unique = RecordBatchMemoryCounter::new();
    unique.count_batch(&source);
    for i in 0..100 {
        state.insert_batch(&source.slice(i * 32, 32))?;
    }
    // Arrow's payload counter excludes shared allocation ownership headers.
    // The retained charge also covers those headers and per-slice metadata.
    let actual = unique.memory_usage() + descriptors;
    let charged = runtime.memory_pool.reserved();
    println!("view unique plus descriptors={actual}, retained_charge={charged}");
    assert!(charged >= actual);
    assert!(
        charged < 3 * actual,
        "shared descriptor lists multiplied: {charged} vs {actual}"
    );
    drop(state);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[test]
fn rank_nested_payload_metadata_covers_child_references_and_owners() -> Result<()> {
    use arrow::array::{NullArray, StructArray};
    let count = 1024;
    let fields: arrow::datatypes::Fields = (0..count)
        .map(|i| Field::new(format!("child_{i}"), DataType::Null, true))
        .collect();
    let children: Vec<ArrayRef> = (0..count)
        .map(|_| Arc::new(NullArray::new(1)) as ArrayRef)
        .collect();
    let references = children.capacity() * size_of::<ArrayRef>();
    let owners = count * (size_of::<NullArray>() + 2 * size_of::<usize>());
    let array = StructArray::new(fields, children, None);
    let batch = RecordBatch::try_from_iter([("payload", Arc::new(array) as ArrayRef)])?;
    let lower_bound = size_of::<RecordBatch>()
        + size_of::<ArrayRef>()
        + size_of::<StructArray>()
        + 2 * size_of::<usize>()
        + references
        + owners;
    let charged = rank_memory::batch_bytes(&batch);
    println!("nested metadata lower bound={lower_bound}, charge={charged}");
    assert!(
        charged >= lower_bound,
        "nested references/owners undercharged"
    );
    Ok(())
}

#[test]
fn rank_sparse_union_metadata_covers_unoccupied_child_slots() -> Result<()> {
    use arrow::array::{NullArray, UnionArray};
    use arrow::datatypes::UnionFields;
    use datafusion_common::utils::memory::RecordBatchMemoryCounter;
    // Arrow 59.3.2 indexes its child reference table by the union type id.
    // One child at id 127 therefore owns 128 slots, not one slot.
    let fields = UnionFields::try_new([127], [Field::new("only", DataType::Null, true)])?;
    let array = UnionArray::try_new(
        fields,
        vec![127_i8].into(),
        None,
        vec![Arc::new(NullArray::new(1))],
    )?;
    let batch = RecordBatch::try_from_iter([("payload", Arc::new(array) as ArrayRef)])?;
    let mut buffers = RecordBatchMemoryCounter::new();
    buffers.count_batch(&batch);
    let lower_bound = buffers.memory_usage()
        + size_of::<RecordBatch>()
        + size_of::<ArrayRef>()
        + size_of::<UnionArray>()
        + 2 * size_of::<usize>()
        + 128 * size_of::<Option<ArrayRef>>()
        + size_of::<NullArray>()
        + 2 * size_of::<usize>();
    let charged = rank_memory::batch_bytes(&batch);
    println!("sparse union metadata lower bound={lower_bound}, charge={charged}");
    assert!(charged >= lower_bound, "union child table undercharged");
    Ok(())
}
