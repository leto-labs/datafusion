// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Build ownership regressions: reservations must follow the buffers still retained.
use super::*;
use crate::stream::RecordBatchStreamAdapter;
use arrow::array::{Int64Array, StringViewArray};
use arrow::datatypes::{DataType, Field};
use datafusion_execution::memory_pool::{FairSpillPool, MemoryPool};
use futures::stream;

async fn build(batches: Vec<RecordBatch>, perfect: bool) -> Result<JoinLeftData> {
    build_with_key(batches, perfect, Arc::new(Column::new("key", 0))).await
}

async fn build_with_key(
    batches: Vec<RecordBatch>,
    perfect: bool,
    key: PhysicalExprRef,
) -> Result<JoinLeftData> {
    build_in_pool(
        batches,
        perfect,
        key,
        Arc::new(FairSpillPool::new(32 << 20)),
    )
    .await
}

async fn build_in_pool(
    batches: Vec<RecordBatch>,
    perfect: bool,
    key: PhysicalExprRef,
    pool: Arc<dyn MemoryPool>,
) -> Result<JoinLeftData> {
    build_full(
        batches,
        perfect,
        vec![key],
        pool,
        ExecutionPlanMetricsSet::new(),
        true,
    )
    .await
}

async fn build_full(
    batches: Vec<RecordBatch>,
    perfect: bool,
    keys: Vec<PhysicalExprRef>,
    pool: Arc<dyn MemoryPool>,
    metrics: ExecutionPlanMetricsSet,
    bitmap: bool,
) -> Result<JoinLeftData> {
    let schema = batches[0].schema();
    let mut config = ConfigOptions::new();
    if !perfect {
        config.execution.perfect_hash_join_small_build_threshold = 0;
        config.execution.perfect_hash_join_min_key_density = f64::MAX;
    }
    let data = collect_left_input(
        RandomState::with_seed(0),
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches.into_iter().map(Ok)),
        )),
        keys,
        BuildProbeJoinMetrics::new(0, &metrics),
        MemoryConsumer::new("HashJoinInput").register(&pool),
        bitmap,
        1,
        false,
        Arc::new(config),
        NullEquality::NullEqualsNothing,
        Count::new(),
        None,
    )
    .await?;
    assert_eq!(matches!(data.map.as_ref(), Map::ArrayMap(_)), perfect);
    Ok(data)
}

fn map_charge(data: &JoinLeftData) -> Result<usize> {
    let rows = data.batch.num_rows();
    Ok(match data.map.as_ref() {
        Map::HashMap(_) => estimate_memory_size::<(u32, u64)>(rows, size_of::<JoinHashMapU32>())?,
        Map::ArrayMap(_) => {
            let key = data.values[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let min = *key.values().iter().min().unwrap() as u64;
            let max = *key.values().iter().max().unwrap() as u64;
            ArrayMap::estimate_memory_size(min, max, rows)
        }
    })
}

fn assert_retained(data: &JoinLeftData, bitmap: bool) -> Result<()> {
    let mut counter = RecordBatchMemoryCounter::new();
    counter.count_batch(&data.batch);
    for key in data.values.iter() {
        counter.count_batch(&RecordBatch::try_from_iter([("key", key.clone())])?);
    }
    let bitmap = if bitmap {
        bit_util::ceil(data.batch.num_rows(), 8)
    } else {
        0
    };
    assert_eq!(
        data._reservation.size(),
        map_charge(data)? + counter.memory_usage() + bitmap
    );
    Ok(())
}

#[tokio::test]
async fn compact_build_releases_source_capacity_for_hash_and_array_maps() -> Result<()> {
    for perfect in [false, true] {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let batches = (0..2)
            .map(|i| {
                let mut values = Vec::with_capacity(131_072);
                values.extend(i * 64..(i + 1) * 64);
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))])
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut input = RecordBatchMemoryCounter::new();
        for batch in &batches {
            input.count_batch(batch);
        }
        let metrics = ExecutionPlanMetricsSet::new();
        let data = build_full(
            batches,
            perfect,
            vec![Arc::new(Column::new("key", 0))],
            Arc::new(FairSpillPool::new(32 << 20)),
            metrics.clone(),
            true,
        )
        .await?;
        assert_eq!(data.batch.num_rows(), 128);
        assert_retained(&data, true)?;
        let mut values = data
            .batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        values.sort_unstable();
        assert_eq!(values, (0..128).collect::<Vec<_>>());
        let peak = metrics
            .clone_inner()
            .sum_by_name("build_mem_used")
            .unwrap()
            .as_usize();
        assert_eq!(peak, input.memory_usage() + map_charge(&data)?);
        assert!(
            data._reservation.size() < 32 * 1024,
            "retained {} bytes, perfect={perfect}",
            data._reservation.size()
        );
        assert_eq!(matches!(data.map.as_ref(), Map::ArrayMap(_)), perfect);
    }
    Ok(())
}

#[tokio::test]
async fn string_view_build_keeps_backing_buffer_reserved() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("payload", DataType::Utf8View, false),
    ]));
    let text = "x".repeat(1 << 20);
    let strings = StringViewArray::from(vec![text.as_str(), "a sufficiently long retained string"]);
    let backing = strings
        .data_buffers()
        .iter()
        .map(|b| b.capacity())
        .sum::<usize>();
    let pointers = strings
        .data_buffers()
        .iter()
        .map(|b| b.data_ptr())
        .collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(strings.slice(1, 1)),
        ],
    )?;
    drop(strings);
    let data = build(vec![batch.clone(), batch], false).await?;
    assert_retained(&data, true)?;
    let payload = data
        .batch
        .column(1)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap();
    assert!(
        payload
            .data_buffers()
            .iter()
            .any(|b| pointers.contains(&b.data_ptr()))
    );
    assert_eq!(
        payload.iter().collect::<Vec<_>>(),
        vec![Some("a sufficiently long retained string"); 2]
    );
    assert_eq!(data.batch.num_rows(), 2);
    assert!(
        data._reservation.size() >= backing,
        "view backing buffer under-reserved"
    );
    Ok(())
}

#[tokio::test]
async fn computed_build_keys_keep_independent_buffers_reserved() -> Result<()> {
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::BinaryExpr;
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from_iter_values(0..128))])?;
    let direct = build(vec![batch.clone()], false).await?;
    let key = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("key", 0)),
        Operator::Plus,
        lit(1_i64),
    ));
    let metrics = ExecutionPlanMetricsSet::new();
    let computed = build_full(
        vec![batch],
        false,
        vec![key],
        Arc::new(FairSpillPool::new(1 << 20)),
        metrics.clone(),
        true,
    )
    .await?;
    assert_eq!(
        metrics
            .clone_inner()
            .sum_by_name("build_mem_used")
            .unwrap()
            .as_usize(),
        computed._reservation.size()
    );
    assert_retained(&direct, true)?;
    assert_retained(&computed, true)?;
    assert!(
        computed._reservation.size() >= direct._reservation.size() + 128 * size_of::<i64>(),
        "computed keys were not reserved"
    );
    Ok(())
}

#[tokio::test]
async fn released_build_capacity_supports_a_spilling_sort_and_drains() -> Result<()> {
    use crate::sorts::sort::SortExec;
    use crate::test::TestMemoryExec;
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_physical_expr::PhysicalSortExpr;
    let pool: Arc<dyn MemoryPool> = Arc::new(FairSpillPool::new(4 << 20));
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let mut values = Vec::with_capacity(262_144);
    values.extend(0..128);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))])?;
    let data = build_in_pool(
        vec![batch.clone(), batch],
        false,
        Arc::new(Column::new("key", 0)),
        pool.clone(),
    )
    .await?;
    let other = MemoryConsumer::new("other build").register(&pool);
    other.try_grow(3 << 20)?;
    let batches = (0..64)
        .map(|i| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from_iter_values(
                    (i * 8192..(i + 1) * 8192).rev(),
                ))],
            )
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let source = TestMemoryExec::try_new_exec(&[batches], schema, None)?;
    let sort = Arc::new(SortExec::new(
        [PhysicalSortExpr::new_default(Arc::new(Column::new(
            "key", 0,
        )))]
        .into(),
        source,
    ));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(pool.clone())
        .build_arc()?;
    let context = Arc::new(
        TaskContext::default()
            .with_runtime(runtime)
            .with_session_config(SessionConfig::new().with_sort_spill_reservation_bytes(64 << 10)),
    );
    let sorted = crate::collect(sort.clone(), context).await?;
    assert!(sort.metrics().unwrap().spill_count().unwrap() > 0);
    let mut expected = 0;
    for batch in sorted {
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for value in values.values() {
            assert_eq!(*value, expected);
            expected += 1;
        }
    }
    assert_eq!(expected, 64 * 8192);
    drop(sort);
    drop(other);
    assert_eq!(pool.reserved(), data._reservation.size());
    drop(data);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn single_padded_batch_keeps_capacity_and_alias_keys_are_not_double_counted() -> Result<()> {
    let mut values = Vec::with_capacity(8192);
    values.extend(0..128);
    let array: ArrayRef = Arc::new(Int64Array::from(values));
    let capacity = array.get_buffer_memory_size();
    let batch = RecordBatch::try_from_iter([("key", array)])?;
    let key: PhysicalExprRef = Arc::new(Column::new("key", 0));
    let data = build_full(
        vec![batch],
        false,
        vec![key.clone(), key],
        Arc::new(FairSpillPool::new(1 << 20)),
        ExecutionPlanMetricsSet::new(),
        false,
    )
    .await?;
    assert!(data._reservation.size() >= capacity);
    assert_retained(&data, false)?;
    Ok(())
}

#[tokio::test]
async fn nullable_nested_views_keep_child_and_null_buffers_reserved() -> Result<()> {
    use arrow::array::StructArray;
    use arrow::buffer::NullBuffer;
    let field = Arc::new(Field::new("child", DataType::Utf8View, true));
    let strings: ArrayRef = Arc::new(StringViewArray::from(vec![
        Some("a long nested payload"),
        None,
    ]));
    let nested: ArrayRef = Arc::new(StructArray::new(
        vec![field].into(),
        vec![strings],
        Some(NullBuffer::from(vec![true, false])),
    ));
    let batch = RecordBatch::try_from_iter([
        ("key", Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef),
        ("payload", nested),
    ])?;
    let data = build(vec![batch.clone(), batch], false).await?;
    assert_retained(&data, true)?;
    Ok(())
}

#[tokio::test]
async fn refused_computed_key_growth_releases_the_build() -> Result<()> {
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::BinaryExpr;
    for perfect in [false, true] {
        let batch = RecordBatch::try_from_iter([(
            "key",
            Arc::new(Int64Array::from_iter_values(0..128)) as ArrayRef,
        )])?;
        let direct = build(vec![batch.clone()], perfect).await?;
        let map = map_charge(&direct)?;
        let pool: Arc<dyn MemoryPool> = Arc::new(FairSpillPool::new(128 * 8 + map));
        let metrics = ExecutionPlanMetricsSet::new();
        let key: PhysicalExprRef = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("key", 0)),
            Operator::Plus,
            lit(1_i64),
        ));
        let result = build_full(
            vec![batch],
            perfect,
            vec![key],
            pool.clone(),
            metrics,
            false,
        )
        .await;
        assert!(
            matches!(
                result,
                Err(datafusion_common::DataFusionError::ResourcesExhausted(_))
            ),
            "new key allocation should exceed the pool"
        );
        assert_eq!(pool.reserved(), 0);
    }
    Ok(())
}

#[derive(Debug)]
struct OwnerObservedPool {
    inner: FairSpillPool,
    backing: arrow::buffer::Buffer,
    observed: Arc<AtomicBool>,
}

impl fmt::Display for OwnerObservedPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "input-owner-observer")
    }
}

impl MemoryPool for OwnerObservedPool {
    fn name(&self) -> &str {
        "input-owner-observer"
    }
    fn register(&self, consumer: &MemoryConsumer) {
        self.inner.register(consumer);
    }
    fn unregister(&self, consumer: &MemoryConsumer) {
        self.inner.unregister(consumer);
    }
    fn grow(&self, reservation: &MemoryReservation, bytes: usize) {
        self.inner.grow(reservation, bytes);
    }
    fn try_grow(&self, reservation: &MemoryReservation, bytes: usize) -> Result<()> {
        self.inner.try_grow(reservation, bytes)
    }
    fn shrink(&self, reservation: &MemoryReservation, bytes: usize) {
        if bytes > 1 << 20 {
            // The test owns the only external reference. Every build input
            // owner must already be gone when its obsolete charge is released.
            assert_eq!(self.backing.strong_count(), 1);
            self.observed.store(true, Ordering::SeqCst);
        }
        self.inner.shrink(reservation, bytes);
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
}

#[tokio::test]
async fn discarded_input_owners_are_dropped_before_shrinking() -> Result<()> {
    let mut values = Vec::with_capacity(262_144);
    values.extend(0..128);
    let array: ArrayRef = Arc::new(Int64Array::from(values));
    let backing = array.to_data().buffers()[0].clone();
    let batch = RecordBatch::try_from_iter([("key", array)])?;
    let observed = Arc::new(AtomicBool::new(false));
    let pool: Arc<dyn MemoryPool> = Arc::new(OwnerObservedPool {
        inner: FairSpillPool::new(4 << 20),
        backing,
        observed: observed.clone(),
    });
    let data = build_in_pool(
        vec![batch.clone(), batch],
        false,
        Arc::new(Column::new("key", 0)),
        pool.clone(),
    )
    .await?;
    assert!(observed.load(Ordering::SeqCst));
    assert_retained(&data, true)?;
    drop(data);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}
