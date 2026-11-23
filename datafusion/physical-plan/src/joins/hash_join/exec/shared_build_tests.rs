// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow patch regressions: a parallel shared CollectLeft build gives the
//! native results for every join type, NULL equality, multi-row chains and
//! probe resumption, charges the pool, and releases it on refusal and drop.
use std::any::Any;

use super::shared_build::{
    self, OPTION, ParallelBuild, ShardedJoinHashMap, configured_rows,
};
use super::*;
use crate::attempt_tasks::AttemptTasks;
use crate::coalesce_partitions::CoalescePartitionsExec;
use crate::common;
use crate::joins::join_hash_map::{HeadTable, JoinHashMapU32, ProbeOffset};
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
    // Explicit parallelism: the shard count follows the session, so the
    // parallel path also runs on a one-CPU host.
    let mut config = SessionConfig::new()
        .with_batch_size(4096)
        .with_target_partitions(8);
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
            .map(|i| {
                (i % 97 != 0).then(|| (i as i32).wrapping_mul(7919) % modulo + shift)
            })
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
    run_rows(join_type, null_equality, keys, context, BUILD_ROWS).await
}

async fn run_rows(
    join_type: JoinType,
    null_equality: NullEquality,
    keys: &[usize],
    context: Arc<TaskContext>,
    build_rows: usize,
) -> Result<Output> {
    let left = table("l", build_rows, 4, 90_000, 0);
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
        batches.extend(
            common::collect(join.execute(partition, Arc::clone(&context))?).await?,
        );
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
async fn parallel_build_matches_native_for_every_join_type_and_null_equality()
-> Result<()> {
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

/// Longbow patch (JT 9): a build large enough for inline keys returns the
/// rows of the hash-only build, natively (partitioned) and in parallel
/// (sharded), for every join type and NULL equality, with one integer key
/// and with two (packed into one field each).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn inline_keys_match_the_hash_only_build_for_every_join_type() -> Result<()> {
    use crate::joins::join_hash_map::PartitionedBuild;
    let rows = 300_000;
    assert!(JoinHashMapU32::partitions_build(rows));
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
    // Packed slots: `k` (with NULLs and duplicates) and `v` (unique, every
    // probe row matches). Dense keys (too wide for the slots): both (rarely
    // equal: an anti join of nearly every row) and `v` twice (every probe
    // row matches).
    for keys in [&[0][..], &[2][..], &[0, 2][..], &[2, 2][..]] {
        for join_type in join_types {
            for null_equality in [
                NullEquality::NullEqualsNothing,
                NullEquality::NullEqualsNull,
            ] {
                for parallel_rows in [0, 1] {
                    let mut outputs = vec![];
                    for inline in [false, true] {
                        inline_keys::set_inline_keys_for_test(Some(inline));
                        outputs.push(
                            run_rows(
                                join_type,
                                null_equality,
                                keys,
                                scoped_context(parallel_rows, None, None),
                                rows,
                            )
                            .await?,
                        );
                    }
                    assert!(
                        !outputs[0].rows.is_empty()
                            || join_type == JoinType::RightAnti
                            || keys == [0, 2]
                    );
                    assert_eq!(outputs[0].shards, outputs[1].shards);
                    assert!(
                        outputs[0].rows == outputs[1].rows,
                        "{keys:?} {join_type} {null_equality:?} {parallel_rows}"
                    );
                }
            }
        }
    }
    inline_keys::set_inline_keys_for_test(None);
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

    let probe =
        Int32Array::from_iter((0..4096).map(|i| (i % 7 != 0).then_some(i * 13 % 60_000)));
    let probe_keys: Vec<ArrayRef> = vec![Arc::new(probe)];
    let mut probe_hashes = vec![0; 4096];
    create_hashes(&probe_keys, &random_state, &mut probe_hashes)?;
    let valid = probe_keys[0].logical_nulls();
    let pairs = |map: &dyn JoinHashMapType, limit: usize| {
        let mut all = Vec::new();
        let mut offset = ProbeOffset::AtRow { row: 0 };
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
            ProbeOffset::AtRow { row: 0 },
            &mut input,
            &mut matched,
        );
        assert_eq!((next, input.len(), matched.len()), (None, 0, 0));
    }
    let contained = sharded.contain_hashes(&probe_hashes);
    assert_eq!(contained, native.contain_hashes(&probe_hashes));
    // Each chain lists its rows in ascending order for any shard count.
    let (_, matched) =
        sharded.get_matched_indices(Box::new(probe_hashes.iter().enumerate()), None);
    let (probes, _) =
        sharded.get_matched_indices(Box::new(probe_hashes.iter().enumerate()), None);
    for window in probes.windows(2).zip(matched.windows(2)) {
        if window.0[0] == window.0[1] {
            assert!(window.1[0] < window.1[1]);
        }
    }
    Ok(())
}

#[test]
fn sharded_map_requires_a_power_of_two_shard_count() {
    let tables = |count: usize| {
        (0..count)
            .map(|_| HeadTable::<u32>::with_capacity(0))
            .collect()
    };
    assert!(ShardedJoinHashMap::new(tables(2), vec![], false).is_empty());
    let result =
        std::panic::catch_unwind(|| ShardedJoinHashMap::new(tables(3), vec![], false));
    assert!(result.is_err());
    // Longbow patch (JT): a shard without room returns an error, no panic.
    let mut map = ShardedJoinHashMap::new(tables(2), vec![0; 64], false);
    let hashes: Vec<u64> = (0..64).map(|row| row << 40).collect();
    let error = map
        .update_from_iter(Box::new(hashes.iter().enumerate()), 0)
        .expect_err("two-slot shards hold one hash each");
    assert!(
        error.to_string().contains("join hash table full"),
        "{error}"
    );
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
    let pool: Arc<dyn MemoryPool> =
        Arc::new(GreedyMemoryPool::new(12 * rows + 30 * rows));
    for _ in 0..200 {
        let mut reservation = MemoryConsumer::new("HashJoinInput").register(&pool);
        let Err(error) = ParallelBuild::for_test(1, 8)
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
        else {
            panic!("the compaction copies must be refused")
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
        let _ =
            tokio::time::timeout(std::time::Duration::from_millis(wait), stream.next())
                .await;
        drop(stream);
        drop(join);
        let deadline = datafusion_common::instant::Instant::now()
            + std::time::Duration::from_secs(30);
        while pool.reserved() != 0
            && datafusion_common::instant::Instant::now() < deadline
        {
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
            let keys =
                Int32Array::from_iter_values((0..1024).map(|i| (b * 1024 + i) as i32));
            let strings: StringViewArray = (0..1024)
                .map(|i| (i % 7 != 0).then(|| format!("a long string value {b}-{i}")))
                .collect();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(keys), Arc::new(strings)],
            )
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
            let outcome = result
                .map(|built| built.map(|(_, batch, copies)| (batch, copies.size())));
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
    let (mut refused, mut admitted, mut late, mut fallbacks) = (0, 0, 0, 0);
    for step in 0..48 {
        let limit = step * 512 * 1024;
        let (outcome, _, pool) = build(limit).await;
        match outcome {
            Ok(Some((batch, _))) => {
                assert_eq!(batch, native, "limit {limit}");
                admitted += 1;
            }
            // The hashes and row lists or the tables were refused: the
            // native build runs instead, and nothing stays charged.
            Ok(None) => fallbacks += 1,
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
        fallbacks > 0 && late > 0 && admitted > 0,
        "fallbacks {fallbacks} refused {refused} (late {late}) admitted {admitted}"
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
        let pool: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(3 * input + 64 * rows));
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
            let indices = arrow::array::UInt32Array::from_iter_values(
                (0..1024).map(|i| (b * 7 + i) % 100),
            );
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
    let dictionary: StringViewArray =
        (0..65_536).map(|i| Some(format!("{i:0>1000}"))).collect();
    let dictionary_bytes: usize = dictionary.data_buffers().iter().map(|b| b.len()).sum();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("s", DataType::Utf8View, true),
    ]));
    // 512 batches in 8 groups of 64 (8 shards of 65,536 rows); each group
    // references every value once.
    let batches = (0..512)
        .map(|b| {
            let keys = Int32Array::from_iter_values((0..1024).map(|i| b * 1024 + i));
            let indices = arrow::array::UInt32Array::from_iter_values(
                (0..1024).map(|i| ((b % 64) * 1024 + i) as u32),
            );
            let column = arrow::compute::take(&dictionary, &indices, None).unwrap();
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(keys), column])
                .unwrap()
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
    let pool: Arc<dyn MemoryPool> =
        Arc::new(GreedyMemoryPool::new(3 * input + 64 * rows));
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

/// Every output partition of `join`, executed concurrently.
async fn collect_all(
    join: &HashJoinExec,
    partitions: usize,
    context: &Arc<TaskContext>,
) -> Result<Vec<RecordBatch>> {
    let streams = (0..partitions)
        .map(|partition| join.execute(partition, Arc::clone(context)))
        .collect::<Result<Vec<_>>>()?;
    let outputs =
        futures::future::try_join_all(streams.into_iter().map(common::collect)).await?;
    Ok(outputs.into_iter().flatten().collect())
}

/// Collects `batches` through `collect_left_input` keyed on column `key`,
/// with the given parallel build (`None`: no Longbow option).
async fn collect_build(
    schema: &SchemaRef,
    batches: Vec<RecordBatch>,
    key: usize,
    parallel: Option<ParallelBuild>,
    dynamic_filters: bool,
    pool: &Arc<dyn MemoryPool>,
) -> Result<JoinLeftData> {
    let exec = TestMemoryExec::try_new_exec(&[batches], Arc::clone(schema), None)?;
    let stream = exec.execute(0, Arc::new(TaskContext::default()))?;
    let metrics = BuildProbeJoinMetrics::new(0, &ExecutionPlanMetricsSet::new());
    let on_left: Vec<PhysicalExprRef> =
        vec![Arc::new(Column::new(schema.field(key).name(), key))];
    let mut config = ConfigOptions::default();
    // Dense integer keys would otherwise select the perfect-hash ArrayMap.
    config.execution.perfect_hash_join_small_build_threshold = 0;
    config.execution.perfect_hash_join_min_key_density = f64::MAX;
    collect_left_input(
        RandomState::with_seed(0),
        stream,
        on_left,
        metrics,
        MemoryConsumer::new("HashJoinInput").register(pool),
        false,
        1,
        dynamic_filters,
        Arc::new(config),
        NullEquality::NullEqualsNothing,
        None,
        Count::new(),
        BuildMode::Ordinary,
        parallel,
        None,
        None,
    )
    .await
}

fn int_batches(
    rows: usize,
    key: impl Fn(usize) -> Option<i32>,
) -> (SchemaRef, Vec<RecordBatch>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("v", DataType::Int32, false),
    ]));
    let batches = (0..rows)
        .step_by(8192)
        .map(|start| {
            let end = (start + 8192).min(rows);
            let keys: Int32Array = (start..end).map(&key).collect();
            let values = Int32Array::from_iter_values(start as i32..end as i32);
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(keys), Arc::new(values)],
            )
            .unwrap()
        })
        .collect();
    (schema, batches)
}

/// 56 keeps the unique-key fast path when NULL build keys own chain slots
/// but no entry (#26018). The sharded map decides it by whether any hash got
/// a second row, as `JoinHashMapU32`, so unique keys with NULLs keep it and
/// return the native pairs; one duplicate key disables it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sharded_map_keeps_the_unique_fast_path_with_null_build_keys() -> Result<()> {
    let rows = 140_000;
    for duplicate in [false, true] {
        let (schema, batches) = int_batches(rows, |i| {
            (i % 17 != 0).then_some(if duplicate && i == 5 { 4 } else { i as i32 })
        });
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
        assert_eq!(sharded.is_chained(), duplicate);
        assert!(sharded.len() < rows, "NULL keys have no entry");
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
        let probe: Int32Array =
            (0..4096).map(|i| (i % 5 != 0).then_some(i * 31)).collect();
        let probe_keys: Vec<ArrayRef> = vec![Arc::new(probe)];
        let mut probe_hashes = vec![0; 4096];
        create_hashes(&probe_keys, &random_state, &mut probe_hashes)?;
        let valid = probe_keys[0].logical_nulls();
        for limit in [1, 7, 4096] {
            let pairs = |map: &dyn JoinHashMapType| {
                let (mut all, mut offset) = (Vec::new(), ProbeOffset::AtRow { row: 0 });
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
                    all.extend(input.iter().copied().zip(matched.iter().copied()));
                    match next {
                        Some(next) => offset = next,
                        None => break,
                    }
                }
                all
            };
            assert_eq!(pairs(sharded.as_ref()), pairs(&native), "limit {limit}");
        }
    }
    Ok(())
}

/// Longbow patch (JT 9): a sharded build with inline keys keeps the
/// hash-only answers of the plain sharded map, and its inline probe returns
/// exactly the pairs of the native map that have equal keys, at every limit,
/// for unique and duplicate keys with NULLs on both sides.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_inline_sharded_map_equals_the_compared_native_probe() -> Result<()> {
    use crate::joins::join_hash_map::PackedU64;
    let rows = 140_000;
    for duplicates in [false, true] {
        let (schema, batches) = int_batches(rows, |i| {
            (i % 17 != 0).then_some(if duplicates {
                (i % 40_000) as i32
            } else {
                i as i32
            })
        });
        let on: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("k", 0))];
        let random_state = RandomState::with_seed(0);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(usize::MAX));
        let mut reservation = MemoryConsumer::new("test").register(&pool);
        let keys = batches
            .iter()
            .map(|batch| evaluate_expressions_to_arrays(&on, batch))
            .collect::<Result<Vec<_>>>()?;
        let encoding =
            InlineKeys::for_build(&keys, NullEquality::NullEqualsNothing).unwrap();
        let (sharded, batch, _copies) = ParallelBuild::for_test(1, 8)
            .build_with::<PackedU64>(
                &batches,
                rows,
                &schema,
                &on,
                &random_state,
                NullEquality::NullEqualsNothing,
                Some(encoding.clone()),
                &mut reservation,
            )
            .await?
            .expect("parallel build");
        assert_eq!(sharded.inline_keys(), Some(&encoding));
        assert_eq!(sharded.is_chained(), duplicates);
        let build_keys = evaluate_expressions_to_arrays(&on, &batch)?;
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
        let probe: Int32Array = (0..6000)
            .map(|i| (i % 5 != 0).then_some(i * 31 - 500))
            .collect();
        let probe_keys: Vec<ArrayRef> = vec![Arc::new(probe)];
        let mut probe_hashes = vec![0; 6000];
        create_hashes(&probe_keys, &random_state, &mut probe_hashes)?;
        let valid = probe_keys[0].logical_nulls();
        let encoded = encoding.encode(&probe_keys).unwrap();
        let drain = |mut step: Box<
            dyn FnMut(ProbeOffset, &mut Vec<u32>, &mut Vec<u64>) -> Option<ProbeOffset>
                + '_,
        >| {
            let (mut all, mut offset) = (Vec::new(), ProbeOffset::AtRow { row: 0 });
            let (mut input, mut matched) = (vec![], vec![]);
            loop {
                let next = step(offset, &mut input, &mut matched);
                all.extend(input.iter().copied().zip(matched.iter().copied()));
                match next {
                    Some(next) => offset = next,
                    None => break,
                }
            }
            all
        };
        for limit in [1, 7, 4096] {
            let hash_only = |map: &dyn JoinHashMapType| {
                drain(Box::new(|offset, input, matched| {
                    map.get_matched_indices_with_limit_offset(
                        &probe_hashes,
                        valid.as_ref(),
                        limit,
                        offset,
                        input,
                        matched,
                    )
                }))
            };
            let native_pairs = hash_only(&native);
            assert_eq!(hash_only(sharded.as_ref()), native_pairs, "limit {limit}");
            let build = build_keys[0].as_primitive::<arrow::datatypes::Int32Type>();
            let probe = probe_keys[0].as_primitive::<arrow::datatypes::Int32Type>();
            let expected: Vec<(u32, u64)> = native_pairs
                .into_iter()
                .filter(|&(p, b)| build.value(b as usize) == probe.value(p as usize))
                .collect();
            assert!(!expected.is_empty());
            let inline = drain(Box::new(|offset, input, matched| {
                sharded
                    .get_inline_matched_indices_with_limit_offset(
                        &probe_hashes,
                        &encoded,
                        &build_keys,
                        limit,
                        offset,
                        input,
                        matched,
                    )
                    .unwrap()
            }));
            assert_eq!(inline, expected, "limit {limit}");
        }
    }
    Ok(())
}

/// Float keys hash by their normalized value (`-0.0` equals `0.0`) and the
/// 56 build stores normalized keys once (#25972). The parallel build gives
/// the native result for `-0.0`, `0.0`, NaN and NULL keys on both sides, and
/// its stored keys are normalized while the batch keeps the original sign.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_build_matches_native_for_float_keys() -> Result<()> {
    use arrow::array::Float64Array;
    use arrow::datatypes::Float64Type;
    let schema = Arc::new(Schema::new(vec![
        Field::new("f", DataType::Float64, true),
        Field::new("v", DataType::Int32, false),
    ]));
    // Few duplicates (a debug build), and probe keys cover only part of the
    // build, so every join type has output.
    let value = |i: usize| match i % 997 {
        0 => None,
        1 => Some(-0.0),
        2 => Some(0.0),
        3 => Some(f64::NAN),
        _ => Some((i % 50_000) as f64 / 4.0),
    };
    let table = |rows: usize, partitions: usize| {
        let mut parts = vec![Vec::new(); partitions];
        for (index, start) in (0..rows).step_by(4096).enumerate() {
            let end = (start + 4096).min(rows);
            let keys: Float64Array = (start..end).map(value).collect();
            let values = Int32Array::from_iter_values(start as i32..end as i32);
            parts[index % partitions].push(
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(keys), Arc::new(values)],
                )
                .unwrap(),
            );
        }
        TestMemoryExec::try_new_exec(&parts, Arc::clone(&schema), None).unwrap()
    };
    for join_type in [
        JoinType::Inner,
        JoinType::Full,
        JoinType::LeftAnti,
        JoinType::RightSemi,
    ] {
        for null_equality in [
            NullEquality::NullEqualsNothing,
            NullEquality::NullEqualsNull,
        ] {
            let mut outputs = Vec::new();
            for rows in [0, 1] {
                let left = table(BUILD_ROWS, 4);
                let right = table(4_000, 2);
                let on: JoinOn = vec![(
                    Arc::new(Column::new("f", 0)) as _,
                    Arc::new(Column::new("f", 0)) as _,
                )];
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
                let context = context(rows, None);
                let batches = collect_all(&join, 2, &context).await?;
                let shards = join
                    .metrics()
                    .unwrap()
                    .sum_by_name("shared_build_shards")
                    .map(|value| value.as_usize());
                assert_eq!(shards.is_some(), rows == 1, "{join_type}");
                outputs.push(canonical(&join.schema(), &batches)?);
            }
            assert!(!outputs[0].is_empty(), "{join_type}");
            assert!(outputs[0] == outputs[1], "{join_type} {null_equality:?}");
        }
    }
    let (_, batches) = {
        let keys = Float64Array::from_iter((0..BUILD_ROWS).map(value));
        let values = Int32Array::from_iter_values(0..BUILD_ROWS as i32);
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(keys), Arc::new(values)],
        )?;
        (
            (),
            (0..BUILD_ROWS)
                .step_by(8192)
                .map(|s| batch.slice(s, 8192.min(BUILD_ROWS - s)))
                .collect::<Vec<_>>(),
        )
    };
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(usize::MAX));
    let data = collect_build(
        &schema,
        batches,
        0,
        Some(ParallelBuild::for_test(1, 8)),
        false,
        &pool,
    )
    .await?;
    let stored = data.values()[0].as_primitive::<Float64Type>();
    let original = data.batch().column(0).as_primitive::<Float64Type>();
    assert_eq!(original.value(1).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(stored.value(1).to_bits(), 0.0_f64.to_bits());
    assert!(
        stored
            .iter()
            .flatten()
            .all(|v| v.to_bits() != (-0.0_f64).to_bits())
    );
    Ok(())
}

/// Past the IN-list limits, a 56 build pushes its map with a key-range
/// bitmap (#25602) computed from the stored keys and bounds. The parallel
/// build reaches the same tail: the same bounds, a map membership over the
/// sharded table and an identical bitmap, and its rows in input order (#26228).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_build_keeps_bounds_and_the_key_range_bitmap() -> Result<()> {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(usize::MAX));
    let (schema, batches) = int_batches(BUILD_ROWS, |i| Some((i as i32 % 50_000) * 37));
    let native = collect_build(&schema, batches.clone(), 0, None, true, &pool).await?;
    let parallel = collect_build(
        &schema,
        batches.clone(),
        0,
        Some(ParallelBuild::for_test(1, 8)),
        true,
        &pool,
    )
    .await?;
    assert_eq!(native.batch(), &concat_batches(&schema, &batches)?);
    assert_eq!(parallel.batch(), native.batch(), "build order");
    assert_eq!(
        parallel.map().num_of_distinct_key(),
        native.map().num_of_distinct_key()
    );
    let bounds = |data: &JoinLeftData| {
        let bounds = data.build.bounds.as_ref().expect("bounds");
        let column = bounds.get_column_bounds(0).expect("column bounds");
        (column.min.clone(), column.max.clone())
    };
    assert_eq!(bounds(&parallel), bounds(&native));
    let bitmap = |data: &JoinLeftData| match &data.build.membership {
        PushdownStrategy::Map(_, Some(bitmap), _) => (bitmap.size(), bitmap.fill_stats()),
        _ => panic!("expected a map membership with a key-range bitmap"),
    };
    assert_eq!(bitmap(&parallel), bitmap(&native));
    Ok(())
}

/// The join's dynamic filter looks probe rows up in the build map (56
/// `HashTableLookupExpr`, `Lookup::Single`). Over the sharded map it keeps
/// exactly the native matches, and a null-aware anti join (`NOT IN`) builds
/// its NULL state over the consolidated batch with the native result.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dynamic_hash_lookup_and_null_aware_builds_match_native() -> Result<()> {
    for (join_type, null_aware) in [
        (JoinType::Inner, false),
        (JoinType::RightSemi, false),
        (JoinType::LeftAnti, true),
    ] {
        let mut outputs = Vec::new();
        for rows in [0, 1] {
            let left = table("l", BUILD_ROWS, 4, 90_000, 0);
            let right = table("r", 6_000, 3, 120_000, -1_000);
            let on = join_on(&left.schema(), &right.schema(), &[0]);
            let dynamic_filter = HashJoinExec::create_dynamic_filter(&on);
            let consumer: Arc<dyn PhysicalExpr> = Arc::clone(&dynamic_filter) as _;
            let right =
                Arc::new(crate::filter::FilterExecBuilder::new(consumer, right).build()?);
            let mut join = HashJoinExec::try_new(
                Arc::new(CoalescePartitionsExec::new(left)),
                right,
                on,
                None,
                &join_type,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                null_aware,
            )?;
            if !null_aware {
                join.dynamic_filter = Some(HashJoinExecDynamicFilter {
                    filter: Arc::clone(&dynamic_filter),
                    build_accumulator: OnceLock::new(),
                    execution_partitions: OnceLock::new(),
                });
            }
            let mut config = SessionConfig::new()
                .with_batch_size(4096)
                .with_target_partitions(8);
            config.options_mut().extensions.insert(Rows(rows));
            config
                .options_mut()
                .optimizer
                .enable_join_dynamic_filter_pushdown = true;
            config
                .options_mut()
                .optimizer
                .hash_join_inlist_pushdown_max_distinct_values = 16;
            let context = Arc::new(TaskContext::default().with_session_config(config));
            // The whole-plan filter barrier and the null-aware probe
            // completion wait for every probe partition: run them together.
            let batches = collect_all(&join, 3, &context).await?;
            if !null_aware {
                let text = format!("{}", dynamic_filter.current()?);
                assert!(text.contains("hash_lookup"), "{join_type}: {text}");
            }
            outputs.push(canonical(&join.schema(), &batches)?);
        }
        assert!(!outputs[0].is_empty() || join_type == JoinType::LeftAnti);
        assert!(outputs[0] == outputs[1], "{join_type}");
    }
    Ok(())
}

/// Family H+M: with the parallel-build option set but a build below its
/// threshold, the single-future build compacts wide string views. 56 already
/// charges the consolidated batch (`concat_build_batches`); the compaction
/// moves its copy into that charge and trims it to the compacted batch, so
/// the build holds exactly what the native build holds apart from the batch,
/// and the pool holds exactly the build's reservation (no double charge).
#[tokio::test]
async fn native_build_compaction_replaces_the_batch_charge() -> Result<()> {
    let (schema, batches) = view_batches(100);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(usize::MAX));
    let native = collect_build(&schema, batches.clone(), 0, None, false, &pool).await?;
    let native_reserved = pool.reserved();
    let parallel = ParallelBuild::for_test(usize::MAX, 8);
    let compacted =
        collect_build(&schema, batches.clone(), 0, Some(parallel), false, &pool).await?;
    assert_eq!(
        pool.reserved(),
        native_reserved + compacted.build.reservation.size()
    );
    assert_eq!(buffers(native.batch()), 100 * buffers(&batches[0]));
    assert_eq!(buffers(compacted.batch()), 1);
    assert_eq!(compacted.batch(), native.batch());
    let rest = |data: &JoinLeftData| {
        data.build.reservation.size() - get_record_batch_memory_size(data.batch())
    };
    assert_eq!(rest(&compacted), rest(&native));
    assert!(compacted.build.reservation.size() < native.build.reservation.size());
    drop(native);
    drop(compacted);
    assert_eq!(pool.reserved(), 0);
    // A pool that admits the native build but not the copy refuses the build
    // with its own error and keeps nothing.
    let pool: Arc<dyn MemoryPool> =
        Arc::new(GreedyMemoryPool::new(native_reserved + 1024));
    let parallel = ParallelBuild::for_test(usize::MAX, 8);
    let Err(error) =
        collect_build(&schema, batches, 0, Some(parallel), false, &pool).await
    else {
        panic!("the compaction copy must be refused")
    };
    assert!(
        matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
        "{error}"
    );
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[rstest::rstest]
#[case(900, 2048)]
#[case(1023, 2048)]
#[case(1024, 2048)]
#[case(1025, 4096)]
#[tokio::test]
async fn parallel_build_admits_exact_shard_charge(
    #[case] rows_per_shard: usize,
    #[case] slots_per_shard: usize,
) -> Result<()> {
    let random_state = RandomState::with_seed(0);
    let candidates = vec![Arc::new(Int32Array::from_iter_values(0..64)) as ArrayRef];
    let mut hashes = vec![0; candidates[0].len()];
    create_hashes(&candidates, &random_state, &mut hashes)?;
    let keys = [0, 1].map(|shard| {
        hashes
            .iter()
            .position(|&hash| shared_build::shard_of(hash, 1) == shard)
            .expect("one key for each shard") as i32
    });

    // The build is large enough for two shards, but only the non-NULL
    // rows size their heads. Every row gets an entry in the shared chain.
    let rows = 1 << 17;
    let (schema, batches) =
        int_batches(rows, |row| keys.get(row / rows_per_shard).copied());
    let on: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("k", 0))];
    let concat_bytes =
        batches
            .iter()
            .flat_map(RecordBatch::columns)
            .try_fold(0, |bytes, column| {
                Ok::<_, DataFusionError>(
                    bytes + column.to_data().get_slice_memory_size()?,
                )
            })?;
    let transient = 12 * rows;
    let table = 2 * slots_per_shard * size_of::<u64>() + rows * size_of::<u32>();
    let input_charge = 128;

    for (limit, fits) in [
        (input_charge + transient + table + concat_bytes, true),
        (input_charge + transient + table - 1, false),
    ] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
        let mut reservation = MemoryConsumer::new("HashJoinInput").register(&pool);
        reservation.try_grow(input_charge)?;
        let built = ParallelBuild::for_test(1, 2)
            .build(
                &batches,
                rows,
                &schema,
                &on,
                &random_state,
                NullEquality::NullEqualsNothing,
                &mut reservation,
            )
            .await?;
        if fits {
            let (map, _, copies) = built.expect("exactly enough for the parallel build");
            assert_eq!(map.allocated_array_bytes(), table);
            assert_eq!(
                reservation.size(),
                input_charge + map.allocated_array_bytes()
            );
            assert_eq!(copies.size(), concat_bytes);
        } else {
            assert!(built.is_none());
            assert_eq!(reservation.size(), input_charge);
            assert_eq!(pool.reserved(), input_charge);
        }
        drop(reservation);
        assert_eq!(pool.reserved(), 0);
    }
    Ok(())
}

/// A pool that admits the input but not the parallel build's hashes and row
/// lists, or not its tables, gets the native build (`None`, nothing charged
/// beyond the input) instead of an error the native build could avoid. An
/// end-to-end join on such a pool runs natively and succeeds.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_transient_or_table_memory_falls_back_to_the_native_build() -> Result<()>
{
    let (schema, batches) = int_batches(BUILD_ROWS, |i| Some(i as i32));
    let rows = BUILD_ROWS;
    let on: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("k", 0))];
    let mut counter = RecordBatchMemoryCounter::new();
    for batch in &batches {
        counter.count_batch(batch);
    }
    let input = counter.memory_usage();
    // Refused hashes and row lists; then refused tables (no view columns,
    // so nothing else is charged between the two steps).
    for limit in [input + 12 * rows - 1, input + 12 * rows + 1024] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
        let mut reservation = MemoryConsumer::new("HashJoinInput").register(&pool);
        reservation.try_grow(input)?;
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
            .await?;
        assert!(built.is_none(), "limit {limit}");
        assert_eq!(reservation.size(), input, "limit {limit}");
        assert_eq!(pool.reserved(), input, "limit {limit}");
    }
    // End to end: on the smallest pool that admits the native build, the
    // parallel build (which also holds its hashes and row lists) falls back.
    let left_schema = Arc::clone(&schema);
    let join = || -> Result<HashJoinExec> {
        let left = TestMemoryExec::try_new_exec(
            std::slice::from_ref(&batches),
            Arc::clone(&left_schema),
            None,
        )?;
        let right = table("r", 1_000, 1, 260_000, 0);
        let on: JoinOn = vec![(
            Arc::new(Column::new("k", 0)) as _,
            Arc::new(Column::new("rk", 0)) as _,
        )];
        HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
    };
    let run = |rows: usize, limit: usize| {
        let join = join();
        async move {
            let join = join?;
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
            let output =
                common::collect(join.execute(0, context(rows, Some(Arc::clone(&pool))))?)
                    .await;
            let shards = join
                .metrics()
                .unwrap()
                .sum_by_name("shared_build_shards")
                .map(|value| value.as_usize());
            let rows = output.map(|batches| canonical(&join.schema(), &batches));
            drop(join);
            assert_eq!(pool.reserved(), 0);
            Ok::<_, DataFusionError>((rows, shards))
        }
    };
    let (mut low, mut high) = (input, 64 * input);
    while high - low > 4096 {
        let mid = low + (high - low) / 2;
        match run(0, mid).await?.0 {
            Ok(_) => high = mid,
            Err(_) => low = mid,
        }
    }
    let (native, _) = run(0, high).await?;
    let (parallel, shards) = run(1, high).await?;
    assert_eq!(shards, None, "the parallel build must fall back");
    let (native, parallel) = (native??, parallel??);
    assert_eq!(native.len(), parallel.len());
    assert!(native.iter().eq(parallel.iter()));
    Ok(())
}

/// Without an attempt scope, a panic in one task set (the column copies)
/// surfaces only after the other set (the shard tables) has finished, so no
/// build task keeps running detached and uncharged after the build unwinds.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_column_task_panic_surfaces_after_every_table_task() -> Result<()> {
    let (schema, batches) = int_batches(BUILD_ROWS, |i| Some(i as i32));
    let on: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("k", 0))];
    let fault = Arc::new(shared_build::TestFault {
        panic_columns: true,
        table_delay: std::time::Duration::from_millis(200),
        ..Default::default()
    });
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(usize::MAX));
    let mut reservation = MemoryConsumer::new("HashJoinInput").register(&pool);
    let build = ParallelBuild::for_test(1, 8).with_fault(Arc::clone(&fault));
    let outcome =
        futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(build.build(
            &batches,
            BUILD_ROWS,
            &schema,
            &on,
            &RandomState::with_seed(0),
            NullEquality::NullEqualsNothing,
            &mut reservation,
        )))
        .await;
    assert!(outcome.is_err(), "the injected panic must surface");
    // Two shards for this build (two minimum shards of rows).
    assert_eq!(
        fault.tables_done.load(std::sync::atomic::Ordering::Relaxed),
        2
    );
    drop(reservation);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

/// The shard count follows the session's `target_partitions`: one partition
/// keeps the native build however large the build is.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_parallelism_bounds_the_shards() -> Result<()> {
    let left = table("l", BUILD_ROWS, 4, 90_000, 0);
    let right = table("r", 1_000, 1, 120_000, 0);
    let on = join_on(&left.schema(), &right.schema(), &[0]);
    let mut shards = vec![];
    for partitions in [1, 2, 8] {
        let join = HashJoinExec::try_new(
            Arc::new(CoalescePartitionsExec::new(Arc::clone(&left))),
            Arc::clone(&right),
            on.clone(),
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?;
        let mut config = SessionConfig::new().with_target_partitions(partitions);
        config.options_mut().extensions.insert(Rows(1));
        config
            .options_mut()
            .execution
            .perfect_hash_join_small_build_threshold = 0;
        config
            .options_mut()
            .execution
            .perfect_hash_join_min_key_density = f64::MAX;
        let context = Arc::new(TaskContext::default().with_session_config(config));
        common::collect(join.execute(0, context)?).await?;
        shards.push(
            join.metrics()
                .unwrap()
                .sum_by_name("shared_build_shards")
                .map(|value| value.as_usize()),
        );
    }
    assert_eq!(shards, vec![None, Some(2), Some(2)]);
    Ok(())
}

/// Longbow patch (JT), review of leto-mjj.49 (M2): a single-future CollectLeft
/// build above the partitioned-build size, in a pool that holds the
/// row-by-row build but not the partitioning's transient arrays, falls back
/// to the row-by-row build and returns the rows of an unbounded run. The
/// build is one batch: 56 charges the concatenation copy after the transient
/// arrays are returned, and one batch copies nothing, so the transient arrays
/// set the build's peak.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refused_partitioned_build_falls_back_to_the_row_by_row_build() -> Result<()> {
    use crate::joins::join_hash_map::{PartitionedBuild, partitioned_build_bytes};
    // The hash-only map; `a_refused_inline_build_falls_back_to_a_hash_only_build`
    // covers the inline map.
    inline_keys::set_inline_keys_for_test(Some(false));
    let rows = 600_000;
    assert!(JoinHashMapU32::partitions_build(rows));
    let run = |pool: Arc<dyn MemoryPool>| async move {
        let left = single_batch(table("l", rows, 4, 400_000, 0)).await?;
        let right = table("r", 30_000, 3, 500_000, 0);
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
        let batches =
            common::collect(join.execute(0, context(0, Some(Arc::clone(&pool))))?)
                .await?;
        let peak = join
            .metrics()
            .unwrap()
            .sum_by_name("build_mem_used")
            .map_or(0, |value| value.as_usize());
        let rows = canonical(&join.schema(), &batches)?;
        drop(join);
        assert_eq!(pool.reserved(), 0);
        Ok::<_, DataFusionError>((rows, peak))
    };
    let (expected, peak) = run(Arc::new(GreedyMemoryPool::new(usize::MAX))).await?;
    let transient = partitioned_build_bytes(rows, false);
    assert!(peak > transient);
    let (actual, tight_peak) =
        run(Arc::new(GreedyMemoryPool::new(peak - transient / 2))).await?;
    assert!(!expected.is_empty());
    assert_eq!(actual, expected);
    // Built row by row: the peak no longer includes the transient arrays.
    assert!(
        tight_peak <= peak - transient,
        "{tight_peak} {peak} {transient}"
    );
    Ok(())
}

/// `plan`'s rows as one batch in one partition.
async fn single_batch(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = plan.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(plan));
    let batches = common::collect(plan.execute(0, context(0, None))?).await?;
    let batch = concat_batches(&schema, &batches)?;
    Ok(TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?)
}
