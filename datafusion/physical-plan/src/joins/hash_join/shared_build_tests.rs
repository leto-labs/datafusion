// Modified by the Longbow authors; see third_party/README.md.

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

//! Longbow patch regressions: a parallel shared CollectLeft build gives the
//! native results for every join type, NULL equality, multi-row chains and
//! probe resumption, charges the pool, and releases it on refusal and drop.
use std::any::Any;

use super::shared_build::{self, OPTION, ParallelBuild, ShardedJoinHashMap, configured_rows};
use super::*;
use crate::attempt_tasks::AttemptTasks;
use crate::coalesce_partitions::CoalescePartitionsExec;
use crate::common;
use crate::joins::join_hash_map::JoinHashMapU32;
use crate::test::TestMemoryExec;
use arrow::array::{AsArray, Int32Array, StringViewArray};
use arrow::datatypes::Field;
use arrow::row::{RowConverter, SortField};
use datafusion_common::DataFusionError;
use datafusion_common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
use datafusion_common::hash_utils::create_hashes;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use futures::StreamExt;

#[derive(Clone, Debug)]
struct Rows(usize);

impl ConfigExtension for Rows {
    const PREFIX: &'static str = "longbow";
}

impl ExtensionOptions for Rows {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }
    fn set(&mut self, _: &str, value: &str) -> Result<()> {
        self.0 = value.parse().unwrap();
        Ok(())
    }
    fn entries(&self) -> Vec<ConfigEntry> {
        vec![ConfigEntry {
            key: OPTION.into(),
            value: Some(self.0.to_string()),
            description: "test",
        }]
    }
}

/// Just above two minimum shards, so the parallel path has 2 or more shards.
const BUILD_ROWS: usize = 140_000;

fn context(rows: usize, pool: Option<Arc<dyn MemoryPool>>) -> Arc<TaskContext> {
    scoped_context(rows, pool, None)
}

fn scoped_context(
    rows: usize,
    pool: Option<Arc<dyn MemoryPool>>,
    scope: Option<Arc<AttemptTasks>>,
) -> Arc<TaskContext> {
    configured_context(rows, pool, scope, false)
}

fn configured_context(
    rows: usize,
    pool: Option<Arc<dyn MemoryPool>>,
    scope: Option<Arc<AttemptTasks>>,
    perfect: bool,
) -> Arc<TaskContext> {
    let mut config = SessionConfig::new().with_batch_size(4096);
    config.options_mut().extensions.insert(Rows(rows));
    if let Some(scope) = scope {
        config = config.with_extension(scope);
    }
    if !perfect {
        // Dense integer keys would otherwise select the perfect-hash ArrayMap
        // on the native path.
        config
            .options_mut()
            .execution
            .perfect_hash_join_small_build_threshold = 0;
        config
            .options_mut()
            .execution
            .perfect_hash_join_min_key_density = f64::MAX;
    }
    let mut runtime = RuntimeEnvBuilder::new();
    if let Some(pool) = pool {
        runtime = runtime.with_memory_pool(pool);
    }
    Arc::new(
        TaskContext::default()
            .with_session_config(config)
            .with_runtime(runtime.build_arc().unwrap()),
    )
}

fn schema(prefix: &str) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(format!("{prefix}k"), DataType::Int32, true),
        Field::new(format!("{prefix}s"), DataType::Utf8View, true),
        Field::new(format!("{prefix}v"), DataType::Int32, false),
    ]))
}

/// `rows` rows in `partitions` partitions of 4,096-row batches. Keys repeat
/// (multi-row chains), every 97th key is NULL, and key strings repeat too.
fn table(
    prefix: &str,
    rows: usize,
    partitions: usize,
    modulo: i32,
    shift: i32,
) -> Arc<dyn ExecutionPlan> {
    let schema = schema(prefix);
    let mut parts = vec![Vec::new(); partitions];
    for (index, start) in (0..rows).step_by(4096).enumerate() {
        let end = (start + 4096).min(rows);
        let keys: Int32Array = (start..end)
            .map(|i| (i % 97 != 0).then(|| (i as i32).wrapping_mul(7919) % modulo + shift))
            .collect();
        let strings: StringViewArray = (start..end)
            .map(|i| (i % 89 != 0).then(|| format!("value-{}", i % 1013)))
            .collect();
        let values = Int32Array::from_iter_values(start as i32..end as i32);
        parts[index % partitions].push(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(keys), Arc::new(strings), Arc::new(values)],
            )
            .unwrap(),
        );
    }
    TestMemoryExec::try_new_exec(&parts, schema, None).unwrap()
}

fn join_on(left: &SchemaRef, right: &SchemaRef, keys: &[usize]) -> JoinOn {
    keys.iter()
        .map(|&k| {
            (
                Arc::new(Column::new(left.field(k).name(), k)) as _,
                Arc::new(Column::new(right.field(k).name(), k)) as _,
            )
        })
        .collect()
}

struct Output {
    rows: Vec<Vec<u8>>,
    shards: Option<usize>,
    array_maps: usize,
}

async fn run(
    join_type: JoinType,
    null_equality: NullEquality,
    keys: &[usize],
    parallel_rows: usize,
) -> Result<Output> {
    run_in(
        join_type,
        null_equality,
        keys,
        scoped_context(parallel_rows, None, None),
    )
    .await
}

async fn run_in(
    join_type: JoinType,
    null_equality: NullEquality,
    keys: &[usize],
    context: Arc<TaskContext>,
) -> Result<Output> {
    let left = table("l", BUILD_ROWS, 4, 90_000, 0);
    let right = table("r", 30_000, 3, 120_000, -1_000);
    let on = join_on(&left.schema(), &right.schema(), keys);
    let join = HashJoinExec::try_new(
        Arc::new(CoalescePartitionsExec::new(left)),
        right,
        on,
        None,
        &join_type,
        None,
        PartitionMode::CollectLeft,
        null_equality,
        false,
    )?;
    let mut batches = vec![];
    for partition in 0..join.properties().partitioning.partition_count() {
        batches.extend(common::collect(join.execute(partition, Arc::clone(&context))?).await?);
    }
    let metrics = join.metrics().unwrap();
    let shards = metrics
        .sum_by_name("shared_build_shards")
        .map(|value| value.as_usize());
    let array_maps = metrics
        .sum_by_name("array_map_created_count")
        .map_or(0, |value| value.as_usize());
    Ok(Output {
        rows: canonical(&join.schema(), &batches)?,
        shards,
        array_maps,
    })
}

/// Rows as sorted row-format bytes: a multiset independent of batch order.
fn canonical(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<Vec<Vec<u8>>> {
    let converter = RowConverter::new(
        schema
            .fields()
            .iter()
            .map(|field| SortField::new(field.data_type().clone()))
            .collect(),
    )?;
    let mut rows = Vec::new();
    for batch in batches {
        let converted = converter.convert_columns(batch.columns())?;
        rows.extend(converted.iter().map(|row| row.as_ref().to_vec()));
    }
    rows.sort_unstable();
    Ok(rows)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_build_matches_native_for_every_join_type_and_null_equality() -> Result<()> {
    let join_types = [
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::LeftSemi,
        JoinType::RightSemi,
        JoinType::LeftAnti,
        JoinType::RightAnti,
        JoinType::LeftMark,
        JoinType::RightMark,
    ];
    for join_type in join_types {
        for null_equality in [
            NullEquality::NullEqualsNothing,
            NullEquality::NullEqualsNull,
        ] {
            let native = run(join_type, null_equality, &[0], 0).await?;
            let parallel = run(join_type, null_equality, &[0], 1).await?;
            assert_eq!(native.shards, None, "{join_type} native path");
            assert!(
                parallel.shards.is_some_and(|shards| shards >= 2),
                "{join_type} parallel path"
            );
            assert!(!native.rows.is_empty() || join_type == JoinType::RightAnti);
            assert_eq!(
                native.rows.len(),
                parallel.rows.len(),
                "{join_type} {null_equality:?}"
            );
            assert!(
                native.rows == parallel.rows,
                "{join_type} {null_equality:?}"
            );
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_build_matches_native_for_composite_string_keys() -> Result<()> {
    for join_type in [JoinType::Inner, JoinType::Full, JoinType::LeftAnti] {
        for null_equality in [
            NullEquality::NullEqualsNothing,
            NullEquality::NullEqualsNull,
        ] {
            let native = run(join_type, null_equality, &[0, 1], 0).await?;
            let parallel = run(join_type, null_equality, &[0, 1], 1).await?;
            assert!(parallel.shards.is_some());
            assert!(
                native.rows == parallel.rows,
                "{join_type} {null_equality:?}"
            );
            // Attempt-scoped build tasks register as producers (ADR 0041).
            let scope = AttemptTasks::new();
            let context = scoped_context(1, None, Some(Arc::clone(&scope)));
            let scoped = run_in(join_type, null_equality, &[0, 1], context).await?;
            assert!(scoped.shards.is_some());
            assert!(
                native.rows == scoped.rows,
                "{join_type} {null_equality:?} scoped"
            );
        }
    }
    Ok(())
}

/// A dense single integer key qualifies for the perfect-hash `ArrayMap`,
/// whose build is serial (concatenation and insertion of every row). A build
/// large enough for the parallel shared build uses it instead (lane SHJ:
/// TPC-H Q14's 22M-row build over 60M part keys at SF300).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_build_takes_precedence_over_the_serial_array_map() -> Result<()> {
    for join_type in [JoinType::Inner, JoinType::RightSemi, JoinType::Full] {
        let native = run_in(
            join_type,
            NullEquality::NullEqualsNothing,
            &[0],
            configured_context(0, None, None, true),
        )
        .await?;
        let parallel = run_in(
            join_type,
            NullEquality::NullEqualsNothing,
            &[0],
            configured_context(1, None, None, true),
        )
        .await?;
        assert_eq!((native.shards, native.array_maps), (None, 1), "{join_type}");
        assert!(
            parallel.shards.is_some_and(|shards| shards >= 2),
            "{join_type}"
        );
        assert_eq!(parallel.array_maps, 0, "{join_type}");
        assert!(native.rows == parallel.rows, "{join_type}");
    }
    // Below the parallel threshold the ArrayMap remains.
    let small = run_in(
        JoinType::Inner,
        NullEquality::NullEqualsNothing,
        &[0],
        configured_context(BUILD_ROWS + 1, None, None, true),
    )
    .await?;
    assert_eq!((small.shards, small.array_maps), (None, 1));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn builds_below_the_threshold_keep_the_native_path() -> Result<()> {
    let output = run(
        JoinType::Inner,
        NullEquality::NullEqualsNothing,
        &[0],
        BUILD_ROWS + 1,
    )
    .await?;
    assert_eq!(output.shards, None);
    Ok(())
}

#[test]
fn configured_rows_reads_the_extension_entry() {
    assert_eq!(configured_rows(&ConfigOptions::new()), 0);
    let mut config = SessionConfig::new();
    config.options_mut().extensions.insert(Rows(1_048_576));
    assert_eq!(configured_rows(config.options()), 1_048_576);
}

/// The sharded map returns exactly the native map's (probe, build) pairs,
/// in every resumption of a small output limit, including NULL probe keys.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sharded_map_resumes_like_the_native_map() -> Result<()> {
    let rows = 140_000;
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    let batches: Vec<RecordBatch> = (0..rows)
        .step_by(8192)
        .map(|start| {
            let keys: Int32Array = (start..(start + 8192).min(rows))
                .map(|i| (i % 31 != 0).then_some((i % 50_000) as i32))
                .collect();
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(keys)]).unwrap()
        })
        .collect();
    let on: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("k", 0))];
    let random_state = RandomState::with_seed(0);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(usize::MAX));
    let mut reservation = MemoryConsumer::new("test").register(&pool);
    let (sharded, batch, _copies) = ParallelBuild::for_test(1, 8)
        .build(
            &batches,
            rows,
            &schema,
            &on,
            &random_state,
            NullEquality::NullEqualsNothing,
            &mut reservation,
        )
        .await?
        .expect("parallel build");
    // Native map over the same consolidated batch.
    let mut native = JoinHashMapU32::with_capacity(rows);
    let mut hashes = vec![0; rows];
    update_hash(
        &on,
        &batch,
        &mut native,
        0,
        &random_state,
        &mut hashes,
        0,
        true,
        NullEquality::NullEqualsNothing,
    )?;
    assert_eq!(sharded.len(), native.len());

    let probe = Int32Array::from_iter((0..4096).map(|i| (i % 7 != 0).then_some(i * 13 % 60_000)));
    let probe_keys: Vec<ArrayRef> = vec![Arc::new(probe)];
    let mut probe_hashes = vec![0; 4096];
    create_hashes(&probe_keys, &random_state, &mut probe_hashes)?;
    let valid = probe_keys[0].logical_nulls();
    let pairs = |map: &dyn JoinHashMapType, limit: usize| {
        let mut all = Vec::new();
        let mut offset = (0, None);
        let (mut input, mut matched) = (vec![], vec![]);
        loop {
            let next = map.get_matched_indices_with_limit_offset(
                &probe_hashes,
                valid.as_ref(),
                limit,
                offset,
                &mut input,
                &mut matched,
            );
            assert!(input.len() <= limit);
            all.extend(input.iter().copied().zip(matched.iter().copied()));
            match next {
                Some(next) => offset = next,
                None => break,
            }
        }
        all.sort_unstable();
        all
    };
    for limit in [1, 3, 64, 8192] {
        let expected = pairs(&native, limit);
        assert!(!expected.is_empty());
        assert_eq!(pairs(sharded.as_ref(), limit), expected, "limit {limit}");
    }
    // An empty probe batch matches nothing, also with duplicate build hashes.
    for map in [&native as &dyn JoinHashMapType, sharded.as_ref()] {
        let (mut input, mut matched) = (vec![7], vec![7]);
        let next = map.get_matched_indices_with_limit_offset(
            &[],
            None,
            8,
            (0, None),
            &mut input,
            &mut matched,
        );
        assert_eq!((next, input.len(), matched.len()), (None, 0, 0));
    }
    let contained = sharded.contain_hashes(&probe_hashes);
    assert_eq!(contained, native.contain_hashes(&probe_hashes));
    // Each chain lists its rows in ascending order for any shard count.
    let (_, matched) = sharded.get_matched_indices(Box::new(probe_hashes.iter().enumerate()), None);
    let (probes, _) = sharded.get_matched_indices(Box::new(probe_hashes.iter().enumerate()), None);
    for window in probes.windows(2).zip(matched.windows(2)) {
        if window.0[0] == window.0[1] {
            assert!(window.1[0] < window.1[1]);
        }
    }
    Ok(())
}

#[test]
fn sharded_map_requires_a_power_of_two_shard_count() {
    let shards = vec![hashbrown::HashTable::new(); 2];
    assert!(ShardedJoinHashMap::new(shards, vec![]).is_empty());
    let result = std::panic::catch_unwind(|| {
        ShardedJoinHashMap::new(vec![hashbrown::HashTable::new(); 3], vec![])
    });
    assert!(result.is_err());
}

/// The build's input fits; its tables do not. The join reports resource
/// exhaustion and every byte returns to the pool.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_table_memory_fails_the_join_and_releases_the_pool() -> Result<()> {
    refuse_table_memory().await
}

/// The refusal path 200 times in one process: every error surfaces only
/// after every build task has finished and released its charges.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_table_memory_releases_the_pool_every_time() -> Result<()> {
    for _ in 0..200 {
        refuse_table_memory().await?;
    }
    Ok(())
}

/// Eight batch groups race for a pool that admits only some of their
/// compacted views. The first refusal stops the other groups, and the
/// build returns the pool's error only once every group task has finished,
/// so nothing it charged is still reserved; 200 times in one process.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn refused_group_compaction_releases_every_charge_before_the_error() -> Result<()> {
    let (schema, batches) = view_batches(512);
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let on: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("k", 0))];
    // Hashes and row lists (12 per row) fit, then about half of the groups'
    // compaction copies (2 x 17 per row plus the referenced values).
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(12 * rows + 30 * rows));
    for _ in 0..200 {
        let mut reservation = MemoryConsumer::new("HashJoinInput").register(&pool);
        let error = match ParallelBuild::for_test(1, 8)
            .build(
                &batches,
                rows,
                &schema,
                &on,
                &RandomState::with_seed(0),
                NullEquality::NullEqualsNothing,
                &mut reservation,
            )
            .await
        {
            Ok(_) => panic!("the compaction copies must be refused"),
            Err(error) => error,
        };
        assert!(
            matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{error}"
        );
        drop(reservation);
        assert_eq!(pool.reserved(), 0);
    }
    Ok(())
}

async fn refuse_table_memory() -> Result<()> {
    let left = table("l", BUILD_ROWS, 4, 200_000, 0);
    let right = table("r", 1_000, 1, 260_000, 0);
    let on = join_on(&left.schema(), &right.schema(), &[0]);
    let input: usize = common::collect(left.execute(0, context(0, None))?)
        .await?
        .iter()
        .map(|batch| batch.get_array_memory_size())
        .sum::<usize>()
        * 4;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(input + input / 2));
    let join = HashJoinExec::try_new(
        Arc::new(CoalescePartitionsExec::new(left)),
        right,
        on,
        None,
        &JoinType::Inner,
        None,
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
        false,
    )?;
    let error = common::collect(join.execute(0, context(1, Some(Arc::clone(&pool))))?)
        .await
        .expect_err("table memory must be refused");
    assert!(
        matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
        "{error}"
    );
    drop(join);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

/// Dropping the probe stream while the shared build runs releases the pool
/// once the spawned build tasks finish.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropping_the_join_during_the_build_releases_the_pool() -> Result<()> {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(usize::MAX));
    for wait in [0, 1, 5, 20] {
        let left = table("l", BUILD_ROWS, 4, 200_000, 0);
        let right = table("r", 1_000, 1, 260_000, 0);
        let on = join_on(&left.schema(), &right.schema(), &[0]);
        let join = HashJoinExec::try_new(
            Arc::new(CoalescePartitionsExec::new(left)),
            right,
            on,
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?;
        let mut stream = join.execute(0, context(1, Some(Arc::clone(&pool))))?;
        let _ = tokio::time::timeout(std::time::Duration::from_millis(wait), stream.next()).await;
        drop(stream);
        drop(join);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while pool.reserved() != 0 && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(pool.reserved(), 0, "wait {wait} ms");
    }
    Ok(())
}

/// `batches` scan-like batches of 1,024 rows whose string views are longer
/// than the 12 inline bytes, each array with its own data buffer.
fn view_batches(batches: usize) -> (SchemaRef, Vec<RecordBatch>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("s", DataType::Utf8View, true),
    ]));
    let batches = (0..batches)
        .map(|b| {
            let keys = Int32Array::from_iter_values((0..1024).map(|i| (b * 1024 + i) as i32));
            let strings: StringViewArray = (0..1024)
                .map(|i| (i % 7 != 0).then(|| format!("a long string value {b}-{i}")))
                .collect();
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(keys), Arc::new(strings)])
                .unwrap()
        })
        .collect();
    (schema, batches)
}

fn buffers(batch: &RecordBatch) -> usize {
    batch.column(1).as_string_view().data_buffers().len()
}

/// Concatenation keeps every source data buffer; the parallel build's string
/// views reference at most one per batch group, with the same values.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_build_compacts_string_views_per_group() -> Result<()> {
    let (schema, batches) = view_batches(200);
    let rows = 200 * 1024;
    let on: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("k", 0))];
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(usize::MAX));
    let mut reservation = MemoryConsumer::new("test").register(&pool);
    let reserved = reservation.size();
    let (_, batch, copies) = ParallelBuild::for_test(1, 8)
        .build(
            &batches,
            rows,
            &schema,
            &on,
            &RandomState::with_seed(0),
            NullEquality::NullEqualsNothing,
            &mut reservation,
        )
        .await?
        .expect("parallel build");
    let native = concat_batches(&schema, &batches)?;
    assert_eq!(buffers(&native), 200 * buffers(&batches[0]));
    assert!(buffers(&batch) <= 8, "{}", buffers(&batch));
    assert_eq!(batch.column(1).as_ref(), native.column(1).as_ref());
    // Hashes and row lists are released; copies stay charged separately
    // until the caller has charged the retained batch.
    assert!(reservation.size() > reserved);
    assert!(copies.size() > 0);
    let tables_and_chain = reservation.size();
    drop(batch);
    assert_eq!(reservation.size(), tables_and_chain);
    drop(copies);
    drop(reservation);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

/// Every pool size either admits the whole parallel build with the native
/// result, or refuses it with the pool's own error class; a refusal at any
/// step (hashes, compacted views, tables, concatenation) leaves nothing
/// reserved once the caller drops its reservation.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_parallel_build_steps_release_every_reservation() -> Result<()> {
    let (schema, batches) = view_batches(160);
    let rows = 160 * 1024;
    let on: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("k", 0))];
    let native = concat_batches(&schema, &batches)?;
    let build = |limit: usize| {
        let (schema, batches, on) = (Arc::clone(&schema), batches.clone(), on.clone());
        async move {
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
            let mut reservation = MemoryConsumer::new("HashJoinInput").register(&pool);
            let result = ParallelBuild::for_test(1, 8)
                .build(
                    &batches,
                    rows,
                    &schema,
                    &on,
                    &RandomState::with_seed(0),
                    NullEquality::NullEqualsNothing,
                    &mut reservation,
                )
                .await;
            let peak = pool.reserved();
            let outcome =
                result.map(|built| built.map(|(_, batch, copies)| (batch, copies.size())));
            drop(reservation);
            (outcome, peak, pool)
        }
    };
    let (full, _, pool) = build(usize::MAX).await;
    let (batch, copies) = full?.expect("parallel build");
    assert_eq!(batch, native);
    drop(batch);
    assert_eq!(pool.reserved(), 0);
    assert!(copies > 0);
    let (mut refused, mut admitted, mut late) = (0, 0, 0);
    for step in 0..48 {
        let limit = step * 512 * 1024;
        let (outcome, _, pool) = build(limit).await;
        match outcome {
            Ok(Some((batch, _))) => {
                assert_eq!(batch, native, "limit {limit}");
                admitted += 1;
            }
            Ok(None) => panic!("parallel path expected"),
            Err(error) => {
                assert!(
                    matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
                    "limit {limit}: {error}"
                );
                refused += 1;
                // Past the first charge (hashes and row lists, 12 bytes per
                // row): refused at a later copy or table step.
                late += usize::from(limit > 12 * rows);
            }
        }
        // Spawned tasks may still be dropping their clones of the copies.
        for _ in 0..200 {
            if pool.reserved() == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(pool.reserved(), 0, "limit {limit}");
    }
    assert!(
        refused > 2 && late > 0 && admitted > 0,
        "refused {refused} (late {late}) admitted {admitted}"
    );
    Ok(())
}

#[test]
fn native_builds_compact_only_wide_string_views() -> Result<()> {
    let (schema, batches) = view_batches(100);
    let wide = concat_batches(&schema, &batches)?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(usize::MAX));
    let copies = MemoryConsumer::new("test").register(&pool);
    let (compacted, changed) = shared_build::compact_views(wide.clone(), &copies)?;
    // The copy was charged before it was made: views plus value bytes.
    assert!(copies.size() >= compacted.column(1).as_string_view().views().len() * 16);
    assert!(changed);
    assert_eq!(buffers(&compacted), 1);
    assert_eq!(compacted, wide);
    let narrow = concat_batches(&schema, &batches[..10])?;
    let charged = copies.size();
    let (kept, changed) = shared_build::compact_views(narrow.clone(), &copies)?;
    assert!(!changed);
    assert_eq!(copies.size(), charged);
    assert_eq!(buffers(&kept), 10 * buffers(&batches[0]));
    assert_eq!(kept, narrow);
    Ok(())
}

/// A pool too small for the compaction copy refuses it before allocating:
/// the pool's error class, nothing charged, the input untouched.
#[test]
fn refused_compaction_charges_nothing() -> Result<()> {
    let (schema, batches) = view_batches(100);
    let wide = concat_batches(&schema, &batches)?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024));
    let copies = MemoryConsumer::new("HashJoinInput").register(&pool);
    let error = shared_build::compact_views(wide.clone(), &copies).expect_err("refused");
    assert!(
        matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
        "{error}"
    );
    assert_eq!(copies.size(), 0);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(buffers(&wide), 100 * buffers(&batches[0]));
    Ok(())
}

/// Longbow's plan-time admission (`shared_hash_joins::admission`) bounds a
/// parallel build by 3 x its input bytes + 64 bytes per row. With the input
/// batches charged first, as `collect_left_input` does, a pool of exactly
/// that size admits the whole build, string-view copies included.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn admission_working_set_covers_the_parallel_build_peak() -> Result<()> {
    for batches in [160, 400] {
        let (schema, batches) = view_batches(batches);
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        let mut counter = RecordBatchMemoryCounter::new();
        for batch in &batches {
            counter.count_batch(batch);
        }
        let input = counter.memory_usage();
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(3 * input + 64 * rows));
        let mut reservation = MemoryConsumer::new("HashJoinInput").register(&pool);
        reservation.try_grow(input)?;
        let on: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("k", 0))];
        let built = ParallelBuild::for_test(1, 8)
            .build(
                &batches,
                rows,
                &schema,
                &on,
                &RandomState::with_seed(0),
                NullEquality::NullEqualsNothing,
                &mut reservation,
            )
            .await?
            .expect("parallel build");
        assert_eq!(built.1, concat_batches(&schema, &batches)?);
    }
    Ok(())
}

/// Views that share one dictionary-like buffer would expand row by row
/// under `gc`; compaction then deduplicates the buffers instead, copying no
/// value bytes, with the same values.
#[test]
fn shared_view_buffers_are_deduplicated_not_expanded() -> Result<()> {
    let dictionary: StringViewArray = (0..100)
        .map(|i| Some(format!("a dictionary value longer than twelve {i}")))
        .collect();
    let dictionary_bytes: usize = dictionary.data_buffers().iter().map(|b| b.len()).sum();
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)]));
    let batches = (0..200)
        .map(|b| {
            let indices =
                arrow::array::UInt32Array::from_iter_values((0..1024).map(|i| (b * 7 + i) % 100));
            let column = arrow::compute::take(&dictionary, &indices, None).unwrap();
            RecordBatch::try_new(Arc::clone(&schema), vec![column]).unwrap()
        })
        .collect::<Vec<_>>();
    let wide = concat_batches(&schema, &batches)?;
    assert_eq!(
        wide.column(0).as_string_view().data_buffers().len(),
        200 * dictionary.data_buffers().len()
    );
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(usize::MAX));
    let copies = MemoryConsumer::new("test").register(&pool);
    let (compacted, changed) = shared_build::compact_views(wide.clone(), &copies)?;
    assert!(changed);
    let views = compacted.column(0).as_string_view();
    assert_eq!(views.data_buffers().len(), dictionary.data_buffers().len());
    assert_eq!(
        views.data_buffers().iter().map(|b| b.len()).sum::<usize>(),
        dictionary_bytes
    );
    // Only the remapped views and validity were charged (no value bytes).
    assert_eq!(copies.size(), wide.num_rows() * 17);
    assert_eq!(compacted, wide);
    Ok(())
}

/// A dictionary buffer that every batch group references in full: each
/// group's values fit the dictionary, so a per-group `gc` would copy it once
/// per group (8 x here). The compaction decision covers the whole build, so
/// the views share the one dictionary, and a pool of exactly the admitted
/// working set (input charged first) admits the build with the same values.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dictionary_shared_by_every_group_is_not_copied_per_group() -> Result<()> {
    let dictionary: StringViewArray = (0..65_536).map(|i| Some(format!("{i:0>1000}"))).collect();
    let dictionary_bytes: usize = dictionary.data_buffers().iter().map(|b| b.len()).sum();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("s", DataType::Utf8View, true),
    ]));
    // 512 batches in 8 groups of 64 (8 shards of 65,536 rows); each group
    // references every value once.
    let batches = (0..512)
        .map(|b| {
            let keys = Int32Array::from_iter_values((0..1024).map(|i| (b * 1024 + i) as i32));
            let indices = arrow::array::UInt32Array::from_iter_values(
                (0..1024).map(|i| ((b % 64) * 1024 + i) as u32),
            );
            let column = arrow::compute::take(&dictionary, &indices, None).unwrap();
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(keys), column]).unwrap()
        })
        .collect::<Vec<_>>();
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let mut counter = RecordBatchMemoryCounter::new();
    for batch in &batches {
        counter.count_batch(batch);
    }
    let input = counter.memory_usage();
    // The dictionary is counted once, and alone exceeds the rows' slack.
    assert!(input < dictionary_bytes + rows * 32, "{input}");
    assert!(dictionary_bytes * 8 > 3 * input + 64 * rows);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(3 * input + 64 * rows));
    let mut reservation = MemoryConsumer::new("HashJoinInput").register(&pool);
    reservation.try_grow(input)?;
    let on: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("k", 0))];
    let (_, batch, copies) = ParallelBuild::for_test(1, 8)
        .build(
            &batches,
            rows,
            &schema,
            &on,
            &RandomState::with_seed(0),
            NullEquality::NullEqualsNothing,
            &mut reservation,
        )
        .await?
        .expect("parallel build");
    assert_eq!(batch, concat_batches(&schema, &batches)?);
    let views = batch.column(1).as_string_view();
    assert_eq!(views.data_buffers().len(), dictionary.data_buffers().len());
    // No value bytes were copied: group views, then the consolidated views
    // and the key column.
    assert!(copies.size() <= rows * (2 * 17 + 4), "{}", copies.size());
    Ok(())
}
