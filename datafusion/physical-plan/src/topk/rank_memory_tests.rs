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
