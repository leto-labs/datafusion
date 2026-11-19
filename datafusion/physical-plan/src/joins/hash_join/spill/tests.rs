// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow family JS: tests of the hybrid hash join.
//!
//! Every spilling run is compared, as a row multiset, with the same join in
//! an unbounded pool.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, Int32Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::test_util::batches_to_sort_string;
use datafusion_common::{JoinSide, JoinType, NullEquality, Result};
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{BinaryExpr, Column};
use datafusion_physical_expr::{LexOrdering, PhysicalExprRef, PhysicalSortExpr};
use futures::StreamExt;

use super::{LEVEL_MUL, partition_of};
use crate::common::collect;
use crate::joins::hash_join::HashJoinExecBuilder;
use crate::joins::utils::{ColumnIndex, JoinFilter};
use crate::joins::{HashJoinExec, PartitionMode};
use crate::metrics::MetricsSet;
use crate::repartition::RepartitionExec;
use crate::streaming::StreamingTableExec;
use crate::test::{TestMemoryExec, TestPartitionStream};
use crate::{ExecutionPlan, ExecutionPlanProperties, Partitioning};

const JOIN_TYPES: [JoinType; 10] = [
    JoinType::Inner,
    JoinType::Left,
    JoinType::Right,
    JoinType::Full,
    JoinType::LeftSemi,
    JoinType::LeftAnti,
    JoinType::RightSemi,
    JoinType::RightAnti,
    JoinType::LeftMark,
    JoinType::RightMark,
];

/// Deterministic test data: `rows` rows of a nullable key with duplicates
/// (`k*k mod keys`), NULL every 13th row, and a distinct payload.
fn table(
    prefix: &str,
    rows: usize,
    keys: i32,
    seed: i32,
    batch_rows: usize,
) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(format!("{prefix}k"), DataType::Int32, true),
        Field::new(format!("{prefix}v"), DataType::Int32, false),
    ]));
    let key = |row: usize| {
        let row = row as i64 + seed as i64;
        (row % 13 != 0).then(|| ((row * row) % keys as i64) as i32)
    };
    (0..rows)
        .step_by(batch_rows)
        .map(|start| {
            let end = (start + batch_rows).min(rows);
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new((start..end).map(key).collect::<Int32Array>()),
                    Arc::new(
                        (start..end)
                            .map(|row| row as i32 + seed)
                            .collect::<Int32Array>(),
                    ),
                ],
            )
            .unwrap()
        })
        .collect()
}

/// `batches` round robin into `partitions` partitions.
fn memory_input(batches: &[RecordBatch], partitions: usize) -> Arc<dyn ExecutionPlan> {
    let mut parts = vec![vec![]; partitions];
    for (index, batch) in batches.iter().enumerate() {
        parts[index % partitions].push(batch.clone());
    }
    TestMemoryExec::try_new_exec(&parts, batches[0].schema(), None).unwrap()
}

fn key(schema: &SchemaRef) -> PhysicalExprRef {
    Arc::new(Column::new_with_schema(schema.field(0).name(), schema).unwrap())
}

/// `build.v > probe.v - 500`: drops some key matches.
fn payload_filter(build: &SchemaRef, probe: &SchemaRef) -> JoinFilter {
    let schema = Arc::new(Schema::new(vec![
        build.field(1).clone(),
        probe.field(1).clone(),
    ]));
    let probe_minus = Arc::new(BinaryExpr::new(
        Arc::new(Column::new(probe.field(1).name(), 1)),
        Operator::Minus,
        Arc::new(datafusion_physical_expr::expressions::Literal::new(
            datafusion_common::ScalarValue::Int32(Some(500)),
        )),
    ));
    let expression = Arc::new(BinaryExpr::new(
        Arc::new(Column::new(build.field(1).name(), 0)),
        Operator::Gt,
        probe_minus,
    ));
    JoinFilter::new(
        expression,
        vec![
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
        ],
        schema,
    )
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    /// One build partition, this many probe partitions.
    CollectLeft(usize),
    /// Both inputs hash-repartitioned into this many partitions.
    Partitioned(usize),
}

struct Case {
    join_type: JoinType,
    null_equality: NullEquality,
    filter: bool,
    shape: Shape,
}

fn join_plan(
    case: &Case,
    build: &[RecordBatch],
    probe: &[RecordBatch],
) -> Result<Arc<dyn ExecutionPlan>> {
    let build_schema = build[0].schema();
    let probe_schema = probe[0].schema();
    let on = vec![(key(&build_schema), key(&probe_schema))];
    let (left, right, mode) = match case.shape {
        Shape::CollectLeft(partitions) => (
            memory_input(build, 1),
            memory_input(probe, partitions),
            PartitionMode::CollectLeft,
        ),
        Shape::Partitioned(partitions) => {
            let left: Arc<dyn ExecutionPlan> = Arc::new(RepartitionExec::try_new(
                memory_input(build, 2),
                Partitioning::Hash(vec![key(&build_schema)], partitions),
            )?);
            let right: Arc<dyn ExecutionPlan> = Arc::new(RepartitionExec::try_new(
                memory_input(probe, 2),
                Partitioning::Hash(vec![key(&probe_schema)], partitions),
            )?);
            (left, right, PartitionMode::Partitioned)
        }
    };
    let filter = case
        .filter
        .then(|| payload_filter(&build_schema, &probe_schema));
    Ok(Arc::new(HashJoinExec::try_new(
        left,
        right,
        on,
        filter,
        &case.join_type,
        None,
        mode,
        case.null_equality,
        false,
    )?))
}

/// A session with join spilling on, a workspace of `workspace` bytes and
/// the test option `force_rows`.
fn spill_config(force_rows: usize, workspace: usize, max_depth: usize) -> SessionConfig {
    let mut config = SessionConfig::new().with_batch_size(256);
    let execution = &mut config.options_mut().execution;
    execution.enable_hash_join_spilling = true;
    execution.hash_join_spill_test_force_rows = force_rows;
    execution.hash_join_spill_reservation_bytes = workspace;
    execution.hash_join_spill_max_depth = max_depth;
    config
}

fn task_context(
    config: SessionConfig,
    memory_limit: Option<usize>,
    disk: bool,
) -> Arc<TaskContext> {
    let mut runtime = RuntimeEnvBuilder::new();
    if let Some(limit) = memory_limit {
        runtime = runtime.with_memory_limit(limit, 1.0);
    }
    if !disk {
        runtime = runtime.with_disk_manager_builder(
            DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
        );
    }
    Arc::new(
        TaskContext::default()
            .with_session_config(config)
            .with_runtime(runtime.build_arc().unwrap()),
    )
}

/// Runs every output partition concurrently.
async fn run(
    plan: &Arc<dyn ExecutionPlan>,
    context: Arc<TaskContext>,
) -> Result<Vec<RecordBatch>> {
    let partitions = plan.output_partitioning().partition_count();
    let handles = (0..partitions)
        .map(|partition| {
            let stream = plan.execute(partition, Arc::clone(&context));
            tokio::spawn(async move { collect(stream?).await })
        })
        .collect::<Vec<_>>();
    let mut out = vec![];
    for handle in handles {
        out.extend(
            tokio::time::timeout(Duration::from_secs(60), handle)
                .await
                .expect("join stalled")
                .expect("join task panicked")?,
        );
    }
    Ok(out)
}

fn metric(metrics: &MetricsSet, name: &str) -> Option<usize> {
    metrics.sum_by_name(name).map(|value| value.as_usize())
}

fn spill_metrics(plan: &Arc<dyn ExecutionPlan>) -> (usize, usize, usize) {
    let metrics = plan.metrics().unwrap();
    (
        metric(&metrics, "spilled_partitions").unwrap_or(0),
        metric(&metrics, "probe_spilled_rows").unwrap_or(0),
        metrics
            .iter()
            .filter(|metric| metric.value().name() == "spill_depth")
            .map(|metric| metric.value().as_usize())
            .max()
            .unwrap_or(0),
    )
}

async fn expected(case: &Case, build: &[RecordBatch], probe: &[RecordBatch]) -> String {
    let plan = join_plan(case, build, probe).unwrap();
    let context = task_context(SessionConfig::new().with_batch_size(256), None, true);
    batches_to_sort_string(&run(&plan, context).await.unwrap())
}

fn all_cases(shapes: &[Shape]) -> Vec<Case> {
    let mut cases = vec![];
    for join_type in JOIN_TYPES {
        for null_equality in [
            NullEquality::NullEqualsNothing,
            NullEquality::NullEqualsNull,
        ] {
            for filter in [false, true] {
                for shape in shapes {
                    cases.push(Case {
                        join_type,
                        null_equality,
                        filter,
                        shape: *shape,
                    });
                }
            }
        }
    }
    cases
}

/// Every join type, both null equalities, with and without a join filter,
/// in `CollectLeft` with one and three probe streams and `Partitioned`
/// mode: forcing half of the partitions to disk keeps the result.
#[tokio::test]
async fn forced_spill_keeps_every_join_type() -> Result<()> {
    let build = table("b", 1500, 211, 0, 100);
    let probe = table("p", 1200, 307, 7, 100);
    for case in all_cases(&[
        Shape::CollectLeft(1),
        Shape::CollectLeft(3),
        Shape::Partitioned(2),
    ]) {
        let plan = join_plan(&case, &build, &probe)?;
        let context = task_context(spill_config(1, 4096, 3), None, true);
        let actual = batches_to_sort_string(&run(&plan, context).await?);
        let label = format!(
            "{} {:?} filter={} {:?}",
            case.join_type, case.null_equality, case.filter, case.shape
        );
        assert_eq!(actual, expected(&case, &build, &probe).await, "{label}");
        let (partitions, _, depth) = spill_metrics(&plan);
        assert!(partitions > 0, "{label}: no partition was written");
        assert_eq!(depth, 1, "{label}");
    }
    Ok(())
}

/// A pool below the build: the refused reservation writes partitions, and
/// the joins of written partitions spill again where they are refused too.
#[tokio::test]
async fn memory_pressure_spills_and_keeps_every_join_type() -> Result<()> {
    let build = table("b", 6000, 3001, 0, 200);
    let probe = table("p", 3000, 4001, 3, 200);
    for case in all_cases(&[Shape::CollectLeft(2), Shape::Partitioned(2)]) {
        let plan = join_plan(&case, &build, &probe)?;
        let context = task_context(spill_config(0, 4096, 3), Some(96 * 1024), true);
        let label = format!(
            "{} {:?} filter={} {:?}",
            case.join_type, case.null_equality, case.filter, case.shape
        );
        let actual = batches_to_sort_string(
            &run(&plan, context)
                .await
                .unwrap_or_else(|error| panic!("{label}: {error}")),
        );
        assert_eq!(actual, expected(&case, &build, &probe).await, "{label}");
        let (partitions, _, _) = spill_metrics(&plan);
        assert!(partitions > 0, "{label}: no partition was written");
    }
    Ok(())
}

/// Written partitions whose builds are refused again spill one level deeper.
#[tokio::test]
async fn refused_written_partitions_spill_one_level_deeper() -> Result<()> {
    let build = table("b", 20000, 9001, 0, 500);
    let probe = table("p", 2000, 9001, 1, 500);
    let case = Case {
        join_type: JoinType::Inner,
        null_equality: NullEquality::NullEqualsNothing,
        filter: false,
        shape: Shape::CollectLeft(1),
    };
    let plan = join_plan(&case, &build, &probe)?;
    let context = task_context(spill_config(0, 1024, 3), Some(48 * 1024), true);
    let actual = batches_to_sort_string(&run(&plan, context).await?);
    assert_eq!(actual, expected(&case, &build, &probe).await);
    let (_, _, depth) = spill_metrics(&plan);
    assert!(
        depth >= 2,
        "expected a second spill level, got depth {depth}"
    );
    Ok(())
}

/// At the depth limit a refused build fails, and says why.
#[tokio::test]
async fn depth_limit_fails_with_its_reason() -> Result<()> {
    let build = table("b", 20000, 9001, 0, 500);
    let probe = table("p", 2000, 9001, 1, 500);
    let case = Case {
        join_type: JoinType::Inner,
        null_equality: NullEquality::NullEqualsNothing,
        filter: false,
        shape: Shape::CollectLeft(1),
    };
    let plan = join_plan(&case, &build, &probe)?;
    let context = task_context(spill_config(0, 1024, 0), Some(48 * 1024), true);
    let error = run(&plan, context).await.unwrap_err().to_string();
    assert!(error.contains("Resources exhausted"), "{error}");
    assert!(error.contains("spill level 1"), "{error}");
    Ok(())
}

/// The ineligible joins of design decision 2 keep today's failure, and the
/// error names the reason.
#[tokio::test]
async fn ineligible_joins_fail_with_their_reason() -> Result<()> {
    let build = table("b", 1000, 97, 0, 100);
    let probe = table("p", 1000, 97, 1, 100);
    let build_schema = build[0].schema();
    let probe_schema = probe[0].schema();
    let on = || vec![(key(&build_schema), key(&probe_schema))];

    // A null-aware anti join.
    let null_aware = HashJoinExecBuilder::new(
        memory_input(&build, 1),
        memory_input(&probe, 1),
        on(),
        JoinType::LeftAnti,
    )
    .with_partition_mode(PartitionMode::CollectLeft)
    .with_null_aware(true)
    .build_exec()?;

    // An ordered probe input.
    let ordering =
        LexOrdering::new(vec![PhysicalSortExpr::new_default(key(&probe_schema))])
            .unwrap();
    let ordered_probe = TestMemoryExec::try_new(
        std::slice::from_ref(&probe),
        Arc::clone(&probe_schema),
        None,
    )?
    .try_with_sort_information(vec![ordering])?;
    let ordered = Arc::new(HashJoinExec::try_new(
        memory_input(&build, 1),
        Arc::new(TestMemoryExec::update_cache(&Arc::new(ordered_probe))),
        on(),
        None,
        &JoinType::Inner,
        None,
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
        false,
    )?);

    // An unbounded probe input.
    let unbounded_probe = Arc::new(StreamingTableExec::try_new(
        Arc::clone(&probe_schema),
        vec![Arc::new(TestPartitionStream::new_with_batches(
            probe.clone(),
        ))],
        None,
        vec![],
        true,
        None,
    )?);
    let unbounded = Arc::new(HashJoinExec::try_new(
        memory_input(&build, 1),
        unbounded_probe,
        on(),
        None,
        &JoinType::Inner,
        None,
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
        false,
    )?);

    let plain = || -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(HashJoinExec::try_new(
            memory_input(&build, 1),
            memory_input(&probe, 1),
            on(),
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    };

    let cases: Vec<(Arc<dyn ExecutionPlan>, bool, Option<&str>)> = vec![
        (null_aware, true, Some("it is a null-aware anti join")),
        (
            ordered,
            true,
            Some("its probe input has an output ordering"),
        ),
        (unbounded, true, Some("its probe input is unbounded")),
        (
            plain()?,
            false,
            Some("the disk manager cannot create files"),
        ),
    ];
    for (plan, disk, reason) in cases {
        let context = task_context(spill_config(0, 1024, 3), Some(1000), disk);
        let error = run(&plan, context).await.unwrap_err().to_string();
        assert!(error.contains("Resources exhausted"), "{error}");
        assert!(error.contains("HashJoinInput"), "{error}");
        if let Some(reason) = reason {
            assert!(
                error.contains(&format!("did not spill because {reason}")),
                "{error}"
            );
        }
    }

    // Spilling off: the failure and its message are unchanged.
    let mut off = spill_config(0, 1024, 3);
    off.options_mut().execution.enable_hash_join_spilling = false;
    let error = run(&plain()?, task_context(off, Some(1000), true))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("Resources exhausted"), "{error}");
    assert!(!error.contains("did not spill"), "{error}");
    Ok(())
}

/// A join that fits pays nothing: no spill metric is registered and the
/// build is charged exactly what it is charged with spilling off.
#[tokio::test]
async fn join_that_fits_takes_the_unchanged_path() -> Result<()> {
    let build = table("b", 3000, 997, 0, 200);
    let probe = table("p", 3000, 997, 5, 200);
    for shape in [Shape::CollectLeft(3), Shape::Partitioned(2)] {
        let case = Case {
            join_type: JoinType::Left,
            null_equality: NullEquality::NullEqualsNothing,
            filter: false,
            shape,
        };
        let mut used = vec![];
        for spilling in [false, true] {
            let mut config = spill_config(0, 4096, 3);
            config.options_mut().execution.enable_hash_join_spilling = spilling;
            let plan = join_plan(&case, &build, &probe)?;
            let actual = run(&plan, task_context(config, Some(64 << 20), true)).await?;
            assert_eq!(
                batches_to_sort_string(&actual),
                expected(&case, &build, &probe).await
            );
            let metrics = plan.metrics().unwrap();
            let names: Vec<String> = metrics
                .iter()
                .map(|metric| metric.value().name().to_string())
                .collect();
            for name in [
                "spill_count",
                "spilled_bytes",
                "spilled_partitions",
                "build_spilled_rows",
                "probe_spilled_rows",
                "spill_depth",
            ] {
                assert!(
                    !names.iter().any(|registered| registered == name),
                    "{shape:?}: {name} registered"
                );
            }
            used.push(metric(&metrics, "build_mem_used"));
        }
        assert_eq!(used[0], used[1], "{shape:?}: build charge changed");
    }
    Ok(())
}

/// A `CollectLeft` probe stream that goes away before or during the probe
/// does not hold the others at the written-partition barrier.
#[tokio::test]
async fn departed_streams_do_not_hold_the_barrier() -> Result<()> {
    let build = table("b", 1500, 211, 0, 100);
    let probe = table("p", 1500, 307, 2, 50);
    let case = Case {
        join_type: JoinType::Inner,
        null_equality: NullEquality::NullEqualsNothing,
        filter: false,
        shape: Shape::CollectLeft(3),
    };
    let plan = join_plan(&case, &build, &probe)?;
    let context = task_context(spill_config(1, 4096, 3), None, true);
    // Stream 0 is dropped before it is polled, stream 1 after one batch.
    drop(plan.execute(0, Arc::clone(&context))?);
    let mut partial = plan.execute(1, Arc::clone(&context))?;
    let _ = partial.next().await;
    drop(partial);
    let rest = tokio::time::timeout(
        Duration::from_secs(60),
        collect(plan.execute(2, Arc::clone(&context))?),
    )
    .await
    .expect("stream 2 waited for departed streams")?;
    // Stream 2 alone yields exactly the matches of probe partition 2.
    let only_two = Arc::new(HashJoinExec::try_new(
        memory_input(&build, 1),
        memory_input(
            &probe[2..].iter().step_by(3).cloned().collect::<Vec<_>>(),
            1,
        ),
        vec![(key(&build[0].schema()), key(&probe[0].schema()))],
        None,
        &JoinType::Inner,
        None,
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
        false,
    )?) as Arc<dyn ExecutionPlan>;
    let context = task_context(SessionConfig::new().with_batch_size(256), None, true);
    assert_eq!(
        batches_to_sort_string(&rest),
        batches_to_sort_string(&run(&only_two, context).await?)
    );
    Ok(())
}

/// A split attempt that executes only some `CollectLeft` probe partitions
/// binds them; its streams complete the barrier among themselves.
#[tokio::test]
async fn split_attempt_barrier_counts_its_own_partitions() -> Result<()> {
    let build = table("b", 1500, 211, 0, 100);
    let probe = table("p", 1600, 307, 4, 100);
    let case = Case {
        join_type: JoinType::RightAnti,
        null_equality: NullEquality::NullEqualsNothing,
        filter: false,
        shape: Shape::CollectLeft(4),
    };
    let plan = join_plan(&case, &build, &probe)?;
    plan.downcast_ref::<HashJoinExec>()
        .unwrap()
        .set_dynamic_filter_partitions(1..3)?;
    let context = task_context(spill_config(1, 4096, 3), None, true);
    let mut actual = vec![];
    let streams = (1..3)
        .map(|partition| {
            let stream = plan.execute(partition, Arc::clone(&context));
            tokio::spawn(async move { collect(stream?).await })
        })
        .collect::<Vec<_>>();
    for stream in streams {
        actual.extend(
            tokio::time::timeout(Duration::from_secs(60), stream)
                .await
                .expect("split streams stalled")
                .unwrap()?,
        );
    }
    let reference = join_plan(&case, &build, &probe)?;
    let context = task_context(SessionConfig::new().with_batch_size(256), None, true);
    let mut expected = vec![];
    for partition in 1..3 {
        expected
            .extend(collect(reference.execute(partition, Arc::clone(&context))?).await?);
    }
    assert_eq!(
        batches_to_sort_string(&actual),
        batches_to_sort_string(&expected)
    );
    Ok(())
}

/// Rows of one level-0 partition spread over every level-1 partition, and
/// partitions are balanced, so a written partition can be divided again.
#[test]
fn levels_use_independent_bits() {
    let shift = 60;
    let values: ArrayRef = Arc::new(Int64Array::from_iter_values(0..200_000));
    let mut hashes = vec![0; values.len()];
    create_hashes(
        [&values],
        crate::joins::hash_join::exec::HASH_JOIN_SEED.random_state(),
        &mut hashes,
    )
    .unwrap();
    let mut level0 = [0usize; 16];
    let mut level1_of_0 = [0usize; 16];
    for hash in &hashes {
        let part = partition_of(*hash, 0, shift);
        level0[part] += 1;
        if part == 0 {
            level1_of_0[partition_of(*hash, 1, shift)] += 1;
        }
    }
    let mean = hashes.len() / 16;
    for count in level0 {
        assert!(count.abs_diff(mean) < mean / 10, "{level0:?}");
    }
    let mean = level0[0] / 16;
    for count in level1_of_0 {
        assert!(count.abs_diff(mean) < mean / 4, "{level1_of_0:?}");
    }
    assert!(LEVEL_MUL.iter().all(|mul| mul % 2 == 1));
}

/// The partitions to join are decided only after every probe stream
/// registered its files: a stream whose probe input is empty finishes its
/// in-memory pass first, and must not skip partitions whose probe rows only
/// other streams wrote.
#[tokio::test]
async fn early_stream_waits_for_every_probe_file() -> Result<()> {
    let build = table("b", 1500, 211, 0, 100);
    let probe = table("p", 1200, 211, 9, 100);
    let probe_schema = probe[0].schema();
    let parts = vec![vec![], probe[..6].to_vec(), probe[6..].to_vec()];
    let on = vec![(key(&build[0].schema()), key(&probe_schema))];
    for join_type in [JoinType::Inner, JoinType::RightSemi, JoinType::Full] {
        let plan = || -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(HashJoinExec::try_new(
                memory_input(&build, 1),
                TestMemoryExec::try_new_exec(&parts, Arc::clone(&probe_schema), None)?,
                on.clone(),
                None,
                &join_type,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?))
        };
        let spilled = plan()?;
        let actual =
            run(&spilled, task_context(spill_config(1, 4096, 3), None, true)).await?;
        let context = task_context(SessionConfig::new().with_batch_size(256), None, true);
        let expected = run(&plan()?, context).await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected),
            "{join_type}"
        );
        assert!(
            spill_metrics(&spilled).0 > 0,
            "{join_type}: nothing was written"
        );
    }
    Ok(())
}
