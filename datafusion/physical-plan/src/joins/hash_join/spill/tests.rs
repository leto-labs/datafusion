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
use datafusion_common::{DataFusionError, JoinSide, JoinType, NullEquality, Result};
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

/// `config` with role reversal off, for tests of the joins of written
/// partitions in their own orientation (reversal would build a small probe
/// file instead).
fn no_reversal(mut config: SessionConfig) -> SessionConfig {
    config.options_mut().execution.hash_join_spill_role_reversal = false;
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
        // Summed over the join's streams: one gauge per join, so the sum is
        // the deepest level.
        metric(&metrics, "spill_depth").unwrap_or(0),
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
    // leto-mjj.98 v2: 4 KiB workspace (was 1 KiB): one written row, decoded
    // with its copy and table, needs about 1.35 KiB (review MJJ98-REVIEW3,
    // finding 2), more than three quarters of 1 KiB.
    let context =
        task_context(no_reversal(spill_config(0, 4096, 3)), Some(48 * 1024), true);
    let actual = batches_to_sort_string(&run(&plan, context).await?);
    assert_eq!(actual, expected(&case, &build, &probe).await);
    let (_, _, depth) = spill_metrics(&plan);
    assert!(
        depth >= 2,
        "expected a second spill level, got depth {depth}"
    );
    Ok(())
}

/// At the depth limit a refused build is joined in build chunks (it failed
/// with its reason before add-join-spill section 3).
#[tokio::test]
async fn depth_limit_joins_in_build_chunks() -> Result<()> {
    let build = table("b", 20000, 9001, 0, 500);
    let probe = table("p", 2000, 9001, 1, 500);
    for join_type in JOIN_TYPES {
        let case = Case {
            join_type,
            null_equality: NullEquality::NullEqualsNothing,
            filter: false,
            shape: Shape::CollectLeft(1),
        };
        let plan = join_plan(&case, &build, &probe)?;
        // leto-mjj.98 v2: 4 KiB workspace (was 1 KiB): one written row, decoded
        // with its copy and table, needs about 1.35 KiB (review MJJ98-REVIEW3,
        // finding 2), more than three quarters of 1 KiB.
        let context =
            task_context(no_reversal(spill_config(0, 4096, 0)), Some(48 * 1024), true);
        let actual = batches_to_sort_string(&run(&plan, Arc::clone(&context)).await?);
        assert_eq!(actual, expected(&case, &build, &probe).await, "{join_type}");
        let metrics = plan.metrics().unwrap();
        assert!(
            metric(&metrics, "spill_chunked_runs").unwrap_or(0) > 0,
            "{join_type}: no run was joined in chunks"
        );
        assert_eq!(context.memory_pool().reserved(), 0, "{join_type}");
    }
    Ok(())
}

/// A partition that one key fills is joined in build chunks, without
/// writing it again at deeper levels (it failed at the depth limit before
/// add-join-spill section 3).
#[tokio::test]
async fn hot_key_is_joined_in_build_chunks() -> Result<()> {
    // Every build row has key 7.
    let build = table("b", 20000, 1, 0, 500)
        .into_iter()
        .map(|batch| {
            let keys = Int32Array::from(vec![7; batch.num_rows()]);
            RecordBatch::try_new(
                batch.schema(),
                vec![Arc::new(keys), Arc::clone(batch.column(1))],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let probe = table("p", 2000, 9001, 1, 500);
    let case = Case {
        join_type: JoinType::Inner,
        null_equality: NullEquality::NullEqualsNothing,
        filter: false,
        shape: Shape::CollectLeft(1),
    };
    let plan = join_plan(&case, &build, &probe)?;
    // leto-mjj.98 v2: 4 KiB workspace (was 1 KiB): one written row, decoded
    // with its copy and table, needs about 1.35 KiB (review MJJ98-REVIEW3,
    // finding 2), more than three quarters of 1 KiB.
    let context =
        task_context(no_reversal(spill_config(0, 4096, 2)), Some(48 * 1024), true);
    let actual = batches_to_sort_string(&run(&plan, context).await?);
    assert_eq!(actual, expected(&case, &build, &probe).await);
    let metrics = plan.metrics().unwrap();
    assert!(metric(&metrics, "spill_chunked_runs").unwrap_or(0) > 0);
    // The hot partition is not divided again.
    let (_, _, depth) = spill_metrics(&plan);
    assert_eq!(depth, 1);
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
/// does not hold the others: no stream waits for another.
#[tokio::test]
async fn departed_streams_do_not_hold_the_others() -> Result<()> {
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
/// binds them; they are the build's members.
#[tokio::test]
async fn split_attempt_counts_its_own_partitions() -> Result<()> {
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

/// Build rows whose key cannot match form a partition that no level can
/// divide; they are emitted as unmatched build rows without a table, so a
/// pool far below them still answers.
#[tokio::test]
async fn unmatchable_rows_need_no_table() -> Result<()> {
    // Every third build key is NULL.
    let schema = Arc::new(Schema::new(vec![
        Field::new("bk", DataType::Int32, true),
        Field::new("bv", DataType::Int32, false),
    ]));
    let build: Vec<RecordBatch> = (0..30)
        .map(|batch| {
            let rows = (batch * 200)..(batch * 200 + 200);
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(
                        rows.clone()
                            .map(|row| (row % 3 != 0).then_some(row % 997))
                            .collect::<Int32Array>(),
                    ),
                    Arc::new(rows.collect::<Int32Array>()),
                ],
            )
            .unwrap()
        })
        .collect();
    let probe = table("p", 2000, 997, 1, 200);
    for join_type in [
        JoinType::Left,
        JoinType::Full,
        JoinType::LeftAnti,
        JoinType::LeftMark,
    ] {
        for shape in [Shape::CollectLeft(2), Shape::Partitioned(2)] {
            let case = Case {
                join_type,
                null_equality: NullEquality::NullEqualsNothing,
                filter: false,
                shape,
            };
            let plan = join_plan(&case, &build, &probe)?;
            // leto-mjj.98 v2: 4 KiB workspace (was 1 KiB): one written row, decoded
            // with its copy and table, needs about 1.35 KiB (review MJJ98-REVIEW3,
            // finding 2), more than three quarters of 1 KiB.
            let context = task_context(spill_config(0, 4096, 3), Some(24 * 1024), true);
            let actual = run(&plan, context)
                .await
                .unwrap_or_else(|error| panic!("{join_type} {shape:?}: {error}"));
            assert_eq!(
                batches_to_sort_string(&actual),
                expected(&case, &build, &probe).await,
                "{join_type} {shape:?}"
            );
        }
    }
    Ok(())
}

/// A pool so small that even the empty table of a build whose rows are all
/// written is refused: that table is charged without refusal. An 8 KiB
/// workspace is too small for any first-level partition, so the first level
/// writes all of them, but the joins of written partitions, refused in the
/// same full pool, build their tables on their floor at the latest one
/// level deeper instead of spilling to the depth limit. Every stream
/// schedule gives the answer (the review's 5 failures in 12 runs were
/// sub-joins recursing in a pool held by buffers grown without refusal).
#[tokio::test]
async fn empty_table_is_charged_without_refusal() -> Result<()> {
    let build = table("b", 2000, 1999, 0, 100);
    let probe = table("p", 2000, 1999, 3, 100);
    for shape in [Shape::Partitioned(1), Shape::Partitioned(4)] {
        let case = Case {
            join_type: JoinType::Inner,
            null_equality: NullEquality::NullEqualsNothing,
            filter: false,
            shape,
        };
        let plan = join_plan(&case, &build, &probe)?;
        // Role reversal off: reversed joins of written partitions divide
        // their probe rows at further levels, which this test does not count.
        let context =
            task_context(no_reversal(spill_config(0, 8192, 3)), Some(2048), true);
        let actual = run(&plan, context)
            .await
            .unwrap_or_else(|error| panic!("{shape:?}: {error}"));
        assert_eq!(
            batches_to_sort_string(&actual),
            expected(&case, &build, &probe).await,
            "{shape:?}"
        );
        let (written, _, depth) = spill_metrics(&plan);
        // One stream finds the pool full at every request, so it writes all
        // 16 first-level partitions (its table is the empty one); several
        // streams may find room now and then, so how many is schedule
        // dependent there.
        if partitions_of(shape) == 1 {
            assert!(written >= 16, "{shape:?}: {written}");
        } else {
            assert!(written > 0, "{shape:?}");
        }
        assert!((1..=2).contains(&depth), "{shape:?}: depth {depth}");
    }
    Ok(())
}

fn partitions_of(shape: Shape) -> usize {
    match shape {
        Shape::CollectLeft(_) => 1,
        Shape::Partitioned(partitions) => partitions,
    }
}

// ---------------------------------------------------------------------------
// Regressions from the task 2.9 review (MJJ90-REVIEW-20261119), its
// reproduction tests kept as they were written, with their assertions made
// strict where the fixes allow.
// ---------------------------------------------------------------------------

/// With a closed CollectLeft build, opening a partition consumes its spill
/// slot even if the stream is dropped without being polled.
#[tokio::test]
async fn spilled_collect_left_partition_cannot_be_reopened() -> Result<()> {
    let build = table("b", 1500, 211, 0, 100);
    let probe = table("p", 1600, 307, 4, 100);
    for join_type in [JoinType::Inner, JoinType::Left] {
        let case = Case {
            join_type,
            null_equality: NullEquality::NullEqualsNothing,
            filter: false,
            shape: Shape::CollectLeft(4),
        };
        let stock = join_plan(&case, &build, &probe)?;
        let context = task_context(SessionConfig::new().with_batch_size(256), None, true);
        drop(stock.execute(0, Arc::clone(&context))?);
        assert_eq!(
            batches_to_sort_string(&run(&stock, context).await?),
            expected(&case, &build, &probe).await,
        );

        let spilled = join_plan(&case, &build, &probe)?;
        let context = task_context(spill_config(1, 4096, 3), None, true);
        drop(spilled.execute(0, Arc::clone(&context))?);
        let error = collect(spilled.execute(0, context)?).await.unwrap_err();
        assert!(
            error.to_string().contains("probes the build twice"),
            "{error}"
        );
        assert!(spill_metrics(&spilled).0 > 0);
    }
    Ok(())
}

/// A one-member build uses the original join type on the written partition.
/// In addition to checking rows, count each run, including the unmatched
/// rows read directly from the unmatchable partition.
#[tokio::test]
async fn single_member_left_and_full_join_written_partitions_once() -> Result<()> {
    let build = table("b", 1500, 211, 0, 100);
    let probe = table("p", 1600, 211, 0, 100);
    for shape in [Shape::CollectLeft(1), Shape::Partitioned(4)] {
        for join_type in [JoinType::Left, JoinType::Full] {
            let case = Case {
                join_type,
                null_equality: NullEquality::NullEqualsNothing,
                filter: true,
                shape,
            };
            let plan = join_plan(&case, &build, &probe)?;
            let context = task_context(spill_config(1, 4096, 3), None, true);
            let actual = run(&plan, context).await?;
            assert_eq!(
                batches_to_sort_string(&actual),
                expected(&case, &build, &probe).await
            );
            let written = spill_metrics(&plan).0;
            assert!(written > 0);
            assert_eq!(
                metric(&plan.metrics().unwrap(), "spill_partition_passes"),
                Some(written),
                "{shape:?} {join_type}"
            );
        }
    }
    Ok(())
}

mod review {
    use std::fmt;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};

    use datafusion_common::DataFusionError;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool, MemoryReservation,
    };

    use super::*;

    /// Runs `partitions` of `plan` the way a Longbow split task does
    /// (`longbow-plan` `shuffle_writer/ranged.rs`, `Source::Claim`):
    /// `concurrency` partition tasks, each of which claims the next partition
    /// of the range when it has finished its last one. `concurrency` is
    /// `min(range, worker slots)` (`longbow-executor` `slots.rs`).
    async fn run_claimed(
        plan: &Arc<dyn ExecutionPlan>,
        context: &Arc<TaskContext>,
        partitions: std::ops::Range<usize>,
        concurrency: usize,
    ) -> std::result::Result<Vec<RecordBatch>, String> {
        let next = Arc::new(AtomicUsize::new(partitions.start));
        let tasks = (0..concurrency)
            .map(|_| {
                let plan = Arc::clone(plan);
                let context = Arc::clone(context);
                let next = Arc::clone(&next);
                let end = partitions.end;
                tokio::spawn(async move {
                    let mut out = vec![];
                    loop {
                        let partition = next.fetch_add(1, Ordering::SeqCst);
                        if partition >= end {
                            break;
                        }
                        let stream = plan.execute(partition, Arc::clone(&context))?;
                        out.extend(collect(stream).await?);
                    }
                    Ok::<_, DataFusionError>(out)
                })
            })
            .collect::<Vec<_>>();
        let mut out = vec![];
        for task in tasks {
            match tokio::time::timeout(Duration::from_secs(15), task).await {
                Ok(joined) => out.extend(
                    joined
                        .map_err(|error| error.to_string())?
                        .map_err(|error| error.to_string())?,
                ),
                Err(_) => return Err("stalled".to_string()),
            }
        }
        Ok(out)
    }

    /// Review finding 2. A split attempt whose range has more partitions than the
    /// worker runs at once (a retry on a worker with fewer slots, or
    /// `partitions_per_task` above the slots) never finishes once a
    /// `CollectLeft` build spilled: the first streams wait at the
    /// written-partition barrier for streams that are only created after
    /// they end. The same join with spilling off finishes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn review_split_attempt_with_fewer_slots_than_partitions() -> Result<()> {
        let build = table("b", 1500, 211, 0, 100);
        let probe = table("p", 1600, 307, 4, 100);
        let case = Case {
            join_type: JoinType::Inner,
            null_equality: NullEquality::NullEqualsNothing,
            filter: false,
            shape: Shape::CollectLeft(4),
        };
        let want = expected(&case, &build, &probe).await;
        let mut failures = vec![];
        for (spilling, bound, concurrency) in [
            // Control: spilling off, one partition at a time.
            (false, true, 1),
            (true, true, 4),
            (true, true, 2),
            (true, true, 1),
            // A consumer that drains an unbound plan's partitions in turn.
            (true, false, 1),
        ] {
            let plan = join_plan(&case, &build, &probe)?;
            if bound {
                plan.downcast_ref::<HashJoinExec>()
                    .unwrap()
                    .set_dynamic_filter_partitions(0..4)?;
            }
            let mut config = spill_config(1, 4096, 3);
            config.options_mut().execution.enable_hash_join_spilling = spilling;
            let context = task_context(config, None, true);
            let label =
                format!("spilling={spilling} bound={bound} concurrency={concurrency}");
            match run_claimed(&plan, &context, 0..4, concurrency).await {
                Ok(actual) => {
                    assert_eq!(batches_to_sort_string(&actual), want, "{label}")
                }
                Err(error) => failures.push(format!("{label}: {error}")),
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
        Ok(())
    }

    /// Hold every stream's workspace before moving any charge, with one
    /// additional workspace already in use. Both transfers must leave the
    /// pool within its original bound throughout, even when called together.
    #[test]
    fn concurrent_floor_transfers_do_not_grow_the_pool() {
        use super::super::{JoinSpill, OnceAsync, RandomState};
        use crate::metrics::ExecutionPlanMetricsSet;

        const STREAMS: usize = 4;
        const WORKSPACE: usize = 4096;
        const LIMIT: usize = 64 * 1024;

        let pool = PeakPool::new(LIMIT);
        let dyn_pool = Arc::clone(&pool) as Arc<dyn MemoryPool>;
        let other = MemoryConsumer::new("Other").register(&dyn_pool);
        other.try_grow(LIMIT).unwrap();
        other.grow(WORKSPACE);
        let ready = std::sync::Barrier::new(STREAMS);
        std::thread::scope(|scope| {
            for partition in 0..STREAMS {
                let ready = &ready;
                let dyn_pool = &dyn_pool;
                scope.spawn(move || {
                    let case = Case {
                        join_type: JoinType::Inner,
                        null_equality: NullEquality::NullEqualsNothing,
                        filter: false,
                        shape: Shape::Partitioned(STREAMS),
                    };
                    let plan = join_plan(
                        &case,
                        &table("b", 8, 11, 0, 2),
                        &table("p", 8, 11, 0, 2),
                    )
                    .unwrap();
                    let join = plan.downcast_ref::<HashJoinExec>().unwrap();
                    let context =
                        pool_context(spill_config(0, WORKSPACE, 3), Arc::clone(dyn_pool));
                    let spill = JoinSpill::decide(
                        join,
                        false,
                        &ExecutionPlanMetricsSet::new(),
                        join.spill_shared(),
                        partition,
                        &context,
                        &RandomState::default(),
                        &Arc::new(OnceAsync::default()),
                    )
                    .unwrap();
                    let mut build = spill.build().unwrap();
                    let mut rows = MemoryConsumer::new("HashJoinInput")
                        .with_can_spill(true)
                        .register(dyn_pool);
                    build.workspace_debt = WORKSPACE;
                    build.settle_workspace(&rows);
                    ready.wait();

                    build.borrow(WORKSPACE / 2, &rows);
                    build.charged = WORKSPACE / 2;
                    ready.wait();

                    let (_, _, table) = build
                        .build_on_floor(&mut rows, &mut |_, attempt, _| {
                            attempt.try_grow(WORKSPACE)
                        })
                        .unwrap();
                    ready.wait();
                    assert_eq!(rows.size(), 0);
                    assert_eq!(table.unwrap().size(), WORKSPACE);
                });
            }
        });
        assert_eq!(
            pool.peak.load(Ordering::SeqCst),
            LIMIT + (STREAMS + 1) * WORKSPACE
        );
        assert_eq!(pool.reserved(), LIMIT + WORKSPACE);
    }

    /// A greedy pool that records its peak, and that can hand whatever a
    /// consumer releases to a competing consumer at once (as a concurrent
    /// operator's `try_grow` would take it).
    pub(super) struct PeakPool {
        inner: GreedyMemoryPool,
        limit: usize,
        pub(super) peak: AtomicUsize,
        competing: AtomicBool,
        competitor: OnceLock<Mutex<MemoryReservation>>,
        taken: AtomicUsize,
    }

    impl PeakPool {
        pub(super) fn new(limit: usize) -> Arc<Self> {
            Arc::new(Self {
                inner: GreedyMemoryPool::new(limit),
                limit,
                peak: AtomicUsize::new(0),
                competing: AtomicBool::new(false),
                competitor: OnceLock::new(),
                taken: AtomicUsize::new(0),
            })
        }

        fn note(&self) {
            self.peak.fetch_max(self.inner.reserved(), Ordering::SeqCst);
        }
    }

    impl fmt::Debug for PeakPool {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("PeakPool").finish()
        }
    }

    impl fmt::Display for PeakPool {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "PeakPool")
        }
    }

    impl MemoryPool for PeakPool {
        fn name(&self) -> &str {
            "PeakPool"
        }

        fn grow(&self, reservation: &MemoryReservation, additional: usize) {
            self.inner.grow(reservation, additional);
            self.note();
        }

        fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
            self.inner.shrink(reservation, shrink);
            if self.competing.load(Ordering::SeqCst)
                && let Some(competitor) = self.competitor.get()
                && let Ok(competitor) = competitor.try_lock()
            {
                let free = self.limit.saturating_sub(self.inner.reserved());
                if free > 0 && self.inner.try_grow(&competitor, free).is_ok() {
                    self.taken.fetch_add(free, Ordering::SeqCst);
                }
            }
        }

        fn try_grow(
            &self,
            reservation: &MemoryReservation,
            additional: usize,
        ) -> Result<()> {
            self.inner.try_grow(reservation, additional)?;
            self.note();
            Ok(())
        }

        fn reserved(&self) -> usize {
            self.inner.reserved()
        }
    }

    pub(super) fn pool_context(
        config: SessionConfig,
        pool: Arc<dyn MemoryPool>,
    ) -> Arc<TaskContext> {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool)
            .build_arc()
            .unwrap();
        Arc::new(
            TaskContext::default()
                .with_session_config(config)
                .with_runtime(runtime),
        )
    }

    /// Review findings 3, 4 and 6. How far a spilling join takes the pool
    /// above its limit, also when a second consumer takes every byte that
    /// is released; and every case must answer (a written partition's join
    /// in a full pool used to recurse to the depth limit).
    #[tokio::test]
    async fn review_peak_reservation_under_spill() -> Result<()> {
        let build = table("b", 6000, 3001, 0, 200);
        let probe = table("p", 3000, 4001, 3, 200);
        let build_bytes: usize = build
            .iter()
            .map(datafusion_common::utils::memory::get_record_batch_memory_size)
            .sum();
        let mut report = vec![];
        let mut failures = vec![];
        let mut worst = 0.0f64;
        for shape in [Shape::CollectLeft(2), Shape::Partitioned(2)] {
            for (limit, workspace, competing) in [
                (96 * 1024, 4096, false),
                (build_bytes * 3 / 2, 4096, false),
                (build_bytes * 3 / 2, 4096, true),
                (96 * 1024, 16 << 20, false),
            ] {
                let case = Case {
                    join_type: JoinType::Inner,
                    null_equality: NullEquality::NullEqualsNothing,
                    filter: false,
                    shape,
                };
                let plan = join_plan(&case, &build, &probe)?;
                let pool = PeakPool::new(limit);
                let dyn_pool: Arc<dyn MemoryPool> = Arc::clone(&pool) as _;
                if competing {
                    let competitor =
                        MemoryConsumer::new("Competitor").register(&dyn_pool);
                    let _ = pool.competitor.set(Mutex::new(competitor));
                    pool.competing.store(true, Ordering::SeqCst);
                }
                let context = pool_context(spill_config(0, workspace, 3), dyn_pool);
                let result = run(&plan, context).await;
                pool.competing.store(false, Ordering::SeqCst);
                let peak = pool.peak.load(Ordering::SeqCst);
                let ratio = peak as f64 / limit as f64;
                let outcome = match &result {
                    Ok(actual) => {
                        assert_eq!(
                            batches_to_sort_string(actual),
                            expected(&case, &build, &probe).await
                        );
                        "ok".to_string()
                    }
                    Err(error) => {
                        let text = error.to_string();
                        failures.push(format!(
                            "{shape:?} limit={limit} workspace={workspace} \
                             competing={competing}: {text}"
                        ));
                        format!("error: {}", &text[..text.len().min(120)])
                    }
                };
                report.push(format!(
                    "{shape:?} limit={limit} workspace={workspace} competing={competing} \
                     build_bytes={build_bytes} peak={peak} peak/limit={ratio:.2} \
                     competitor_took={} written={} -> {outcome}",
                    pool.taken.load(Ordering::SeqCst),
                    spill_metrics(&plan).0,
                ));
                // Without a competitor the pool stays within 10% of its
                // limit. With one that takes every released byte, a build
                // may still grow the part of its workspace that released
                // charge did not cover (design decision 8), one per stream.
                // A floor transfer keeps the existing charge in the pool.
                let streams = match shape {
                    Shape::CollectLeft(streams) | Shape::Partitioned(streams) => streams,
                };
                if workspace == 4096 {
                    if competing {
                        assert!(
                            peak <= limit + (streams + 1) * workspace,
                            "{shape:?}: peak {peak} above {limit} + ({streams} + 1) x \
                             {workspace}"
                        );
                    } else {
                        worst = worst.max(ratio);
                    }
                }
            }
        }
        println!("{report:#?}");
        assert!(failures.is_empty(), "{failures:#?}");
        assert!(
            worst <= 1.10,
            "a spilling join took the pool {worst:.2} times above its limit:\n{report:#?}"
        );
        Ok(())
    }

    /// Lifecycle: after success and after a stream set is dropped at any
    /// point, no reservation and no spill file of the join remains once the
    /// plan is gone (and, after success, already before that).
    #[tokio::test]
    async fn review_no_reservation_or_file_after_success_or_cancel() -> Result<()> {
        let build = table("b", 6000, 3001, 0, 200);
        let probe = table("p", 3000, 4001, 3, 200);
        for join_type in [JoinType::Inner, JoinType::Full] {
            for shape in [Shape::CollectLeft(3), Shape::Partitioned(2)] {
                // `None` runs to the end; `Some(n)` drops every stream after
                // `n` output batches in total.
                for stop_after in [None, Some(0), Some(1), Some(3), Some(8), Some(20)] {
                    let case = Case {
                        join_type,
                        null_equality: NullEquality::NullEqualsNothing,
                        filter: false,
                        shape,
                    };
                    let plan = join_plan(&case, &build, &probe)?;
                    let pool = PeakPool::new(96 * 1024);
                    let dyn_pool: Arc<dyn MemoryPool> = Arc::clone(&pool) as _;
                    let mut config = spill_config(0, 4096, 3);
                    config = config.with_batch_size(64);
                    let context = pool_context(config, dyn_pool);
                    let disk = Arc::clone(&context.runtime_env().disk_manager);
                    let label =
                        format!("{join_type} {shape:?} stop_after={stop_after:?}");
                    let partitions = plan.output_partitioning().partition_count();
                    let streams = (0..partitions)
                        .map(|partition| plan.execute(partition, Arc::clone(&context)))
                        .collect::<Result<Vec<_>>>()?;
                    let mut merged = futures::stream::select_all(streams);
                    let mut seen = 0;
                    let mut finished = false;
                    loop {
                        if stop_after == Some(seen) {
                            break;
                        }
                        let next =
                            tokio::time::timeout(Duration::from_secs(30), merged.next())
                                .await
                                .unwrap_or_else(|_| panic!("{label}: stalled"));
                        match next {
                            Some(batch) => {
                                batch?;
                                seen += 1;
                            }
                            None => {
                                finished = true;
                                break;
                            }
                        }
                    }
                    drop(merged);
                    if finished {
                        for _ in 0..200 {
                            if (pool.reserved(), disk.used_disk_space()) == (0, 0) {
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(10)).await;
                        }
                        assert_eq!(
                            (pool.reserved(), disk.used_disk_space()),
                            (0, 0),
                            "{label}: left after success, plan alive"
                        );
                    }
                    if finished || seen > 0 {
                        assert!(spill_metrics(&plan).0 > 0, "{label}: nothing spilled");
                    }
                    drop(plan);
                    // The inputs' aborted tasks (`RepartitionExec`) let go of
                    // their batches when the runtime next runs them.
                    for _ in 0..200 {
                        if (pool.reserved(), disk.used_disk_space()) == (0, 0) {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                    assert_eq!(
                        (pool.reserved(), disk.used_disk_space()),
                        (0, 0),
                        "{label}: left after the plan was dropped (finished={finished})"
                    );
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Re-review of f34625e52 (MJJ90-REREVIEW-20261119): reproduction tests.
    // Not part of the lane. `finding N` refers to that review's REVIEW.md.
    // -----------------------------------------------------------------------
    mod rereview {
        use std::collections::HashMap;

        use super::*;
        use crate::joins::hash_join::spill::JoinSpill;

        fn config(spilling: bool, force_rows: usize, workspace: usize) -> SessionConfig {
            let mut config = spill_config(force_rows, workspace, 3);
            config.options_mut().execution.enable_hash_join_spilling = spilling;
            config
        }

        fn rows(batches: &[RecordBatch]) -> usize {
            batches.iter().map(RecordBatch::num_rows).sum()
        }

        /// Executes every partition, lets partition 0 go (before its first
        /// poll, or after one output batch) and collects the others one
        /// after another.
        async fn others_after_stream_0_left(
            plan: &Arc<dyn ExecutionPlan>,
            context: &Arc<TaskContext>,
            polled: bool,
        ) -> Result<Vec<RecordBatch>> {
            let partitions = plan.output_partitioning().partition_count();
            let mut streams = (0..partitions)
                .map(|partition| plan.execute(partition, Arc::clone(context)))
                .collect::<Result<Vec<_>>>()?;
            let mut first = streams.remove(0);
            if polled {
                let _ = first.next().await;
            }
            drop(first);
            let mut out = vec![];
            for stream in streams {
                out.extend(
                    tokio::time::timeout(Duration::from_secs(60), collect(stream))
                        .await
                        .expect("stalled")?,
                );
            }
            Ok(out)
        }

        /// Finding 1. One `CollectLeft` stream leaves early (a consumer that
        /// drops one partition). The stock join still gives every row that
        /// the probe rows of the other streams produce (and no build-only
        /// row). The spilled join must give the same rows.
        #[tokio::test]
        async fn rereview_early_drop_keeps_the_other_streams_rows() -> Result<()> {
            let build = table("b", 1500, 211, 0, 100);
            let probe = table("p", 3000, 307, 2, 50);
            let mut failures = vec![];
            let mut report = vec![];
            for join_type in JOIN_TYPES {
                for polled in [false, true] {
                    // These give no batch before their probe ends.
                    if polled
                        && matches!(
                            join_type,
                            JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark
                        )
                    {
                        continue;
                    }
                    let case = Case {
                        join_type,
                        null_equality: NullEquality::NullEqualsNothing,
                        filter: false,
                        shape: Shape::CollectLeft(3),
                    };
                    let mut seen = vec![];
                    for spilling in [false, true] {
                        let plan = join_plan(&case, &build, &probe)?;
                        let context = task_context(config(spilling, 1, 4096), None, true);
                        let out =
                            others_after_stream_0_left(&plan, &context, polled).await?;
                        if spilling {
                            assert!(spill_metrics(&plan).0 > 0, "nothing was written");
                        }
                        seen.push((rows(&out), batches_to_sort_string(&out)));
                    }
                    let line = format!(
                        "{join_type} stream 0 polled={polled}: stock {} rows, spilled {} rows",
                        seen[0].0, seen[1].0
                    );
                    if seen[0].1 != seen[1].1 {
                        failures.push(line.clone());
                    }
                    report.push(line);
                }
            }
            println!("{report:#?}");
            assert!(failures.is_empty(), "{failures:#?}");
            Ok(())
        }

        /// Finding 2. The member count comes from the bound range. If more
        /// streams run than were bound, the spilled join must fail, not
        /// return fewer rows than the stock join.
        #[tokio::test]
        async fn rereview_more_streams_than_bound_members() -> Result<()> {
            let build = table("b", 1500, 211, 0, 100);
            let probe = table("p", 1600, 307, 4, 100);
            let case = Case {
                join_type: JoinType::Inner,
                null_equality: NullEquality::NullEqualsNothing,
                filter: false,
                shape: Shape::CollectLeft(4),
            };
            let mut outcomes = vec![];
            for spilling in [false, true] {
                let plan = join_plan(&case, &build, &probe)?;
                plan.downcast_ref::<HashJoinExec>()
                    .unwrap()
                    .set_dynamic_filter_partitions(0..2)?;
                let context = task_context(config(spilling, 1, 4096), None, true);
                let streams = (0..4)
                    .map(|partition| plan.execute(partition, Arc::clone(&context)))
                    .collect::<Result<Vec<_>>>()?;
                let mut out = vec![];
                let mut error = None;
                for stream in streams {
                    match collect(stream).await {
                        Ok(batches) => out.extend(batches),
                        Err(found) => {
                            error = Some(found.to_string());
                            break;
                        }
                    }
                }
                outcomes.push((error, rows(&out), batches_to_sort_string(&out)));
            }
            let (stock, spilled) = (&outcomes[0], &outcomes[1]);
            assert!(stock.0.is_none(), "{:?}", stock.0);
            println!(
                "stock {} rows; spilled {} rows, error {:?}",
                stock.1, spilled.1, spilled.0
            );
            assert!(
                spilled.0.is_some() || spilled.2 == stock.2,
                "2 members bound, 4 streams ran: stock {} rows, spilled {} rows, no error",
                stock.1,
                spilled.1
            );
            Ok(())
        }

        /// Control. Build-emitting joins (joined by the last member) in an
        /// attempt that runs its partitions one after another, bound and
        /// unbound: no stall and the complete answer.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn rereview_build_emitting_joins_one_partition_at_a_time() -> Result<()> {
            let build = table("b", 1500, 211, 0, 100);
            let probe = table("p", 1600, 307, 4, 100);
            let mut failures = vec![];
            for join_type in [
                JoinType::Left,
                JoinType::Full,
                JoinType::LeftSemi,
                JoinType::LeftAnti,
                JoinType::LeftMark,
            ] {
                let case = Case {
                    join_type,
                    null_equality: NullEquality::NullEqualsNothing,
                    filter: false,
                    shape: Shape::CollectLeft(4),
                };
                let want = expected(&case, &build, &probe).await;
                for (bound, concurrency) in
                    [(true, 1), (true, 2), (true, 4), (false, 1), (false, 3)]
                {
                    let plan = join_plan(&case, &build, &probe)?;
                    if bound {
                        plan.downcast_ref::<HashJoinExec>()
                            .unwrap()
                            .set_dynamic_filter_partitions(0..4)?;
                    }
                    let context = task_context(config(true, 1, 4096), None, true);
                    let label =
                        format!("{join_type} bound={bound} concurrency={concurrency}");
                    match run_claimed(&plan, &context, 0..4, concurrency).await {
                        Ok(actual) => {
                            if batches_to_sort_string(&actual) != want {
                                failures.push(format!(
                                    "{label}: wrong rows ({})",
                                    rows(&actual)
                                ));
                            }
                        }
                        Err(error) => failures.push(format!("{label}: {error}")),
                    }
                }
            }
            assert!(failures.is_empty(), "{failures:#?}");
            Ok(())
        }

        /// Control. Every join type on a multi-thread runtime (the lane's
        /// tests run their streams on one thread), forced and under a small
        /// pool, several rounds: the "last member" choice and the shared
        /// partition joins under real parallelism.
        #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
        async fn rereview_parallel_streams_keep_every_join_type() -> Result<()> {
            let build = table("b", 6000, 3001, 0, 200);
            let probe = table("p", 3000, 4001, 3, 100);
            let mut want: HashMap<String, String> = HashMap::new();
            let mut failures = vec![];
            let mut runs = 0;
            for round in 0..6 {
                for case in all_cases(&[Shape::CollectLeft(4), Shape::Partitioned(3)]) {
                    let label = format!(
                        "{} {:?} filter={} {:?}",
                        case.join_type, case.null_equality, case.filter, case.shape
                    );
                    if !want.contains_key(&label) {
                        want.insert(label.clone(), expected(&case, &build, &probe).await);
                    }
                    let (force, limit) = if round % 2 == 0 {
                        (1, None)
                    } else {
                        (0, Some(96 * 1024))
                    };
                    let plan = join_plan(&case, &build, &probe)?;
                    let context = task_context(spill_config(force, 4096, 3), limit, true);
                    runs += 1;
                    match run(&plan, context).await {
                        Ok(actual) => {
                            if batches_to_sort_string(&actual) != want[&label] {
                                failures.push(format!(
                                    "round {round} {label}: wrong rows ({})",
                                    rows(&actual)
                                ));
                            }
                        }
                        Err(error) => {
                            let text = error.to_string();
                            failures.push(format!(
                                "round {round} {label}: {}",
                                &text[..text.len().min(300)]
                            ));
                        }
                    }
                }
            }
            println!("{runs} runs, {} failures", failures.len());
            assert!(
                failures.is_empty(),
                "{} of {runs} runs failed: {:#?}",
                failures.len(),
                &failures[..failures.len().min(16)]
            );
            Ok(())
        }

        /// Control. Probe-preserving joins: two streams leave at different
        /// points (also while they run a shared partition join); the other
        /// two give exactly their own partitions' rows, and nothing stays
        /// reserved or on disk.
        #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
        async fn rereview_streams_that_leave_midway_do_not_disturb_the_others()
        -> Result<()> {
            let build = table("b", 6000, 3001, 0, 200);
            let probe = table("p", 6000, 4001, 3, 50);
            let mut failures = vec![];
            for join_type in [
                JoinType::Inner,
                JoinType::Right,
                JoinType::RightSemi,
                JoinType::RightAnti,
                JoinType::RightMark,
            ] {
                let case = Case {
                    join_type,
                    null_equality: NullEquality::NullEqualsNothing,
                    filter: false,
                    shape: Shape::CollectLeft(4),
                };
                let reference = join_plan(&case, &build, &probe)?;
                let context = task_context(config(false, 0, 4096), None, true);
                let mut want = vec![];
                for partition in 2..4 {
                    want.extend(
                        collect(reference.execute(partition, Arc::clone(&context))?)
                            .await?,
                    );
                }
                let want = batches_to_sort_string(&want);
                for pressure in [false, true] {
                    for stop in [0usize, 1, 2, 5, 12, 40] {
                        let label =
                            format!("{join_type} pressure={pressure} stop={stop}");
                        let plan = join_plan(&case, &build, &probe)?;
                        let pool =
                            PeakPool::new(if pressure { 96 * 1024 } else { 1 << 30 });
                        let dyn_pool: Arc<dyn MemoryPool> = Arc::clone(&pool) as _;
                        let config = spill_config(usize::from(!pressure), 4096, 3)
                            .with_batch_size(64);
                        let context = pool_context(config, dyn_pool);
                        let disk = Arc::clone(&context.runtime_env().disk_manager);
                        let mut streams = (0..4)
                            .map(|partition| {
                                plan.execute(partition, Arc::clone(&context))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let keep_3 = streams.pop().unwrap();
                        let keep_2 = streams.pop().unwrap();
                        let leavers = streams
                            .into_iter()
                            .zip([stop, stop * 3 + 1])
                            .map(|(mut stream, batches)| {
                                tokio::spawn(async move {
                                    for _ in 0..batches {
                                        if !matches!(stream.next().await, Some(Ok(_))) {
                                            break;
                                        }
                                    }
                                })
                            })
                            .collect::<Vec<_>>();
                        let keepers = [keep_2, keep_3]
                            .into_iter()
                            .map(|stream| tokio::spawn(collect(stream)))
                            .collect::<Vec<_>>();
                        let mut actual = vec![];
                        let mut error = None;
                        for keeper in keepers {
                            match tokio::time::timeout(Duration::from_secs(60), keeper)
                                .await
                            {
                                Ok(joined) => match joined.unwrap() {
                                    Ok(batches) => actual.extend(batches),
                                    Err(found) => error = Some(found.to_string()),
                                },
                                Err(_) => error = Some("stalled".to_string()),
                            }
                        }
                        for leaver in leavers {
                            let _ = leaver.await;
                        }
                        if let Some(error) = error {
                            failures.push(format!("{label}: {error}"));
                        } else if batches_to_sort_string(&actual) != want {
                            failures
                                .push(format!("{label}: wrong rows ({})", rows(&actual)));
                        }
                        drop(plan);
                        for _ in 0..300 {
                            if (pool.reserved(), disk.used_disk_space()) == (0, 0) {
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(10)).await;
                        }
                        if (pool.reserved(), disk.used_disk_space()) != (0, 0) {
                            failures.push(format!(
                                "{label}: {} bytes reserved, {} bytes on disk after the \
                                 plan was dropped",
                                pool.reserved(),
                                disk.used_disk_space()
                            ));
                        }
                    }
                }
            }
            assert!(failures.is_empty(), "{failures:#?}");
            Ok(())
        }

        /// Measurement. A pool that another consumer keeps full from the
        /// start: every reservation of the join is refused, so it runs on
        /// its floors. Prints the peak against the stated bound
        /// `limit + (streams + 1) x workspace`.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn rereview_peak_in_a_pool_that_stays_full() -> Result<()> {
            let build = table("b", 20000, 9001, 0, 500);
            let probe = table("p", 4000, 9001, 1, 250);
            let limit = 64 * 1024;
            let mut report = vec![];
            let mut over = vec![];
            let mut errors = vec![];
            for shape in [
                Shape::CollectLeft(1),
                Shape::CollectLeft(4),
                Shape::Partitioned(4),
            ] {
                for join_type in [JoinType::Inner, JoinType::Full] {
                    for workspace in [4096usize, 16384, 65536] {
                        let case = Case {
                            join_type,
                            null_equality: NullEquality::NullEqualsNothing,
                            filter: false,
                            shape,
                        };
                        let plan = join_plan(&case, &build, &probe)?;
                        let pool = PeakPool::new(limit);
                        let dyn_pool: Arc<dyn MemoryPool> = Arc::clone(&pool) as _;
                        let competitor =
                            MemoryConsumer::new("Competitor").register(&dyn_pool);
                        competitor.try_grow(limit)?;
                        let _ = pool.competitor.set(Mutex::new(competitor));
                        pool.competing.store(true, Ordering::SeqCst);
                        let context =
                            pool_context(spill_config(0, workspace, 3), dyn_pool);
                        let result = run(&plan, context).await;
                        pool.competing.store(false, Ordering::SeqCst);
                        let peak = pool.peak.load(Ordering::SeqCst);
                        let streams = match shape {
                            Shape::CollectLeft(streams) | Shape::Partitioned(streams) => {
                                streams
                            }
                        };
                        let bound = limit + (streams + 1) * workspace;
                        let (written, _, depth) = spill_metrics(&plan);
                        let label =
                            format!("{join_type} {shape:?} workspace={workspace}");
                        let outcome = match &result {
                            Ok(actual) => {
                                if batches_to_sort_string(actual)
                                    == expected(&case, &build, &probe).await
                                {
                                    "ok".to_string()
                                } else {
                                    errors.push(format!("{label}: wrong rows"));
                                    "wrong rows".to_string()
                                }
                            }
                            Err(error) => {
                                let text = error.to_string();
                                errors.push(format!(
                                    "{label}: {}",
                                    &text[..text.len().min(400)]
                                ));
                                "error".to_string()
                            }
                        };
                        let line = format!(
                            "{label}: peak-limit={} bound-limit={} in workspaces={:.2} \
                             written={written} depth={depth} -> {outcome}",
                            peak - limit,
                            bound - limit,
                            (peak - limit) as f64 / workspace as f64,
                        );
                        if peak > bound {
                            over.push(line.clone());
                        }
                        report.push(line);
                    }
                }
            }
            println!("{report:#?}");
            assert!(errors.is_empty(), "{errors:#?}");
            assert!(over.is_empty(), "above the stated bound: {over:#?}");
            Ok(())
        }

        /// Measurement for finding 3. `CollectLeft` streams that join their
        /// written partitions while the first level's in-memory table is
        /// still held for the streams that come later: how often build rows
        /// are written, and how deep, against one stream and against streams
        /// that run together.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn rereview_first_level_table_stays_during_written_partitions() -> Result<()>
        {
            let build = table("b", 20000, 9001, 0, 500);
            let probe = table("p", 8000, 9001, 1, 250);
            let build_rows = rows(&build);
            let build_bytes: usize = build
                .iter()
                .map(datafusion_common::utils::memory::get_record_batch_memory_size)
                .sum();
            let mut report = vec![];
            for (streams, concurrency, bound) in [
                (1, 1, false),
                (4, 4, false),
                (4, 4, true),
                (4, 1, true),
                (4, 1, false),
                (8, 1, true),
            ] {
                let case = Case {
                    join_type: JoinType::Inner,
                    null_equality: NullEquality::NullEqualsNothing,
                    filter: false,
                    shape: Shape::CollectLeft(streams),
                };
                let plan = join_plan(&case, &build, &probe)?;
                if bound {
                    plan.downcast_ref::<HashJoinExec>()
                        .unwrap()
                        .set_dynamic_filter_partitions(0..streams)?;
                }
                let limit = build_bytes * 3 / 2;
                let pool = PeakPool::new(limit);
                let dyn_pool: Arc<dyn MemoryPool> = Arc::clone(&pool) as _;
                // The hash table, not the perfect-hash array (these keys
                // are dense), so that the table is the larger part.
                let mut config = spill_config(0, 4096, 3);
                let execution = &mut config.options_mut().execution;
                execution.perfect_hash_join_small_build_threshold = 0;
                execution.perfect_hash_join_min_key_density = f64::INFINITY;
                let context = pool_context(config, dyn_pool);
                let result = run_claimed(&plan, &context, 0..streams, concurrency).await;
                let metrics = plan.metrics().unwrap();
                let written_rows = metric(&metrics, "build_spilled_rows").unwrap_or(0);
                let (partitions, _, depth) = spill_metrics(&plan);
                let outcome = match result {
                    Ok(actual) => {
                        if batches_to_sort_string(&actual)
                            == expected(&case, &build, &probe).await
                        {
                            "ok".to_string()
                        } else {
                            "WRONG ROWS".to_string()
                        }
                    }
                    Err(error) => format!("error: {}", &error[..error.len().min(300)]),
                };
                report.push(format!(
                    "streams={streams} at once={concurrency} bound={bound}: build rows \
                     written {written_rows} ({:.2} x the build), partitions written \
                     {partitions}, depth {depth}, peak/limit {:.2} -> {outcome}",
                    written_rows as f64 / build_rows as f64,
                    pool.peak.load(Ordering::SeqCst) as f64 / limit as f64,
                ));
            }
            println!("{report:#?}");
            assert!(
                report.iter().all(|line| line.ends_with("-> ok")),
                "{report:#?}"
            );
            Ok(())
        }

        /// Measurement for finding 3. What stays reserved after each
        /// `CollectLeft` stream ends, when the streams run one after another.
        #[tokio::test]
        async fn rereview_reserved_between_streams_run_in_turn() -> Result<()> {
            let build = table("b", 20000, 9001, 0, 500);
            let probe = table("p", 8000, 9001, 1, 250);
            let build_bytes: usize = build
                .iter()
                .map(datafusion_common::utils::memory::get_record_batch_memory_size)
                .sum();
            let case = Case {
                join_type: JoinType::Inner,
                null_equality: NullEquality::NullEqualsNothing,
                filter: false,
                shape: Shape::CollectLeft(4),
            };
            let plan = join_plan(&case, &build, &probe)?;
            let limit = build_bytes * 3 / 2;
            let pool = PeakPool::new(limit);
            let dyn_pool: Arc<dyn MemoryPool> = Arc::clone(&pool) as _;
            let mut config = spill_config(0, 4096, 3);
            let execution = &mut config.options_mut().execution;
            execution.perfect_hash_join_small_build_threshold = 0;
            execution.perfect_hash_join_min_key_density = f64::INFINITY;
            let context = pool_context(config, dyn_pool);
            let mut report = vec![format!("limit {limit}, build {build_bytes} bytes")];
            let mut out = vec![];
            for partition in 0..4 {
                let mut stream = plan.execute(partition, Arc::clone(&context))?;
                let mut most = 0;
                while let Some(batch) = stream.next().await {
                    out.push(batch?);
                    most = most.max(pool.reserved());
                }
                drop(stream);
                let metrics = plan.metrics().unwrap();
                report.push(format!(
                    "after stream {partition}: reserved {} (most seen between its batches \
                     {most}), build rows written so far {}, depth {}",
                    pool.reserved(),
                    metric(&metrics, "build_spilled_rows").unwrap_or(0),
                    spill_metrics(&plan).2,
                ));
            }
            println!("{report:#?}");
            assert_eq!(
                batches_to_sort_string(&out),
                expected(&case, &build, &probe).await
            );
            Ok(())
        }

        /// Earlier finding 1 and 4. The lane's test
        /// `empty_table_is_charged_without_refusal` with the parameters it
        /// had at b86028d75 (a 512 byte workspace, four streams, a 2,048
        /// byte pool), 12 times, and with workspaces up to the lane's 8 KiB.
        /// Below 2 KiB the named refusal of the chunk unit is the answer.
        #[tokio::test]
        async fn rereview_empty_table_test_with_its_first_parameters() -> Result<()> {
            let build = table("b", 2000, 1999, 0, 100);
            let probe = table("p", 2000, 1999, 3, 100);
            let case = Case {
                join_type: JoinType::Inner,
                null_equality: NullEquality::NullEqualsNothing,
                filter: false,
                shape: Shape::Partitioned(4),
            };
            let want = expected(&case, &build, &probe).await;
            let mut report = vec![];
            let mut failed = 0;
            for workspace in [512usize, 1024, 2048, 4096, 8192] {
                let mut pass = 0;
                let mut errors: HashMap<String, usize> = HashMap::new();
                for _ in 0..12 {
                    let plan = join_plan(&case, &build, &probe)?;
                    let context =
                        task_context(spill_config(0, workspace, 3), Some(2048), true);
                    match run(&plan, context).await {
                        Ok(actual) => {
                            assert_eq!(batches_to_sort_string(&actual), want);
                            pass += 1;
                        }
                        Err(error) => {
                            let text = error.to_string();
                            let first = text.lines().next().unwrap_or("").to_string();
                            *errors.entry(first).or_default() += 1;
                        }
                    }
                }
                // leto-mjj.98 v2: below 2 KiB no written row fits the chunk
                // unit with its decoded IPC message, copy and table (about
                // 1.35 KiB; review MJJ98-REVIEW3, finding 2). Those
                // workspaces must refuse with the error that names the unit.
                if workspace < 2048 {
                    let named = errors.keys().all(|error| {
                        error.contains("one row of") && error.contains("needs")
                    });
                    assert!(pass == 12 || named, "workspace={workspace}: {errors:?}");
                } else {
                    failed += 12 - pass;
                }
                report.push(format!(
                    "workspace={workspace}: pass {pass} of 12 {errors:?}"
                ));
            }
            println!("{report:#?}");
            assert_eq!(failed, 0, "{report:#?}");
            Ok(())
        }

        /// Finding 4. A hot key that shares its partitions with a few other
        /// keys: the build cannot be divided (leto-mjj.98: it is joined in
        /// build chunks).
        #[tokio::test]
        async fn rereview_hot_key_among_other_keys_names_the_key() -> Result<()> {
            let schema = table("b", 1, 1, 0, 1)[0].schema();
            // 4 of 5 build rows have key 7; the others have 3001 keys.
            let build: Vec<RecordBatch> = (0..40)
                .map(|batch| {
                    let rows = (batch * 500)..(batch * 500 + 500);
                    RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![
                            Arc::new(
                                rows.clone()
                                    .map(|row: i32| {
                                        Some(if row % 5 != 0 {
                                            7
                                        } else {
                                            100 + (row.wrapping_mul(7919))
                                                .rem_euclid(3001)
                                        })
                                    })
                                    .collect::<Int32Array>(),
                            ),
                            Arc::new(rows.collect::<Int32Array>()),
                        ],
                    )
                    .unwrap()
                })
                .collect();
            let probe = table("p", 2000, 9001, 1, 500);
            let case = Case {
                join_type: JoinType::Inner,
                null_equality: NullEquality::NullEqualsNothing,
                filter: false,
                shape: Shape::CollectLeft(1),
            };
            let plan = join_plan(&case, &build, &probe)?;
            // leto-mjj.98 v2: 4 KiB workspace (was 1 KiB): one written row, decoded
            // with its copy and table, needs about 1.35 KiB (review MJJ98-REVIEW3,
            // finding 2), more than three quarters of 1 KiB.
            let context = task_context(
                no_reversal(spill_config(0, 4096, 2)),
                Some(48 * 1024),
                true,
            );
            // leto-mjj.98: the partition is joined in build chunks; it failed
            // with "one key value fills its partition" before section 3.
            let actual = batches_to_sort_string(&run(&plan, context).await?);
            assert_eq!(actual, expected(&case, &build, &probe).await);
            let metrics = plan.metrics().unwrap();
            assert!(metric(&metrics, "spill_chunked_runs").unwrap_or(0) > 0);
            Ok(())
        }

        /// Finding 6. The refusal of another operator, reported by a pool
        /// that lists its top consumers, names a `HashJoinInput` consumer;
        /// it must not get the join's "did not spill" context. (Adapted by
        /// leto-mjj.99: the context now comes from a flag that the wrapped
        /// build input sets, so the refusal travels through that input.)
        #[tokio::test]
        async fn rereview_refusal_context_on_another_operators_refusal() {
            let context = task_context(spill_config(0, 1024, 3), Some(10_000), true);
            let pool = context.memory_pool();
            let join = MemoryConsumer::new("HashJoinInput[1]").register(pool);
            join.try_grow(9_000).unwrap();
            let aggregate =
                MemoryConsumer::new("GroupedHashAggregateStream[0]").register(pool);
            let refusal = aggregate.try_grow(5_000).unwrap_err();
            let schema = table("b", 1, 1, 0, 1)[0].schema();
            let input: crate::SendableRecordBatchStream =
                Box::pin(crate::stream::RecordBatchStreamAdapter::new(
                    Arc::clone(&schema),
                    futures::stream::once(async move { Err(refusal) }),
                ));
            let spill =
                JoinSpill::Ineligible("its probe input has an output ordering".into());
            let (mut input, explain) = spill.explain_refusal(input);
            let error = input.next().await.unwrap().unwrap_err();
            let explained = explain(error).to_string();
            println!("{explained}");
            assert!(!explained.contains("did not spill"), "{explained}");
            // The join's own refusal still says why.
            let own = join.try_grow(5_000).unwrap_err();
            let empty: crate::SendableRecordBatchStream =
                Box::pin(crate::stream::RecordBatchStreamAdapter::new(
                    schema,
                    futures::stream::empty(),
                ));
            let (_, explain) = spill.explain_refusal(empty);
            let explained = explain(own).to_string();
            assert!(explained.contains("did not spill"), "{explained}");
        }
    }
}

// ---------------------------------------------------------------------------
// leto-mjj.99: membership of a spilled build (design decision 5).
// ---------------------------------------------------------------------------

/// Executes `streams` streams of partition 0 of `plan`, all at once or one
/// after another (each executed when the previous one ended).
async fn partition_zero_streams(
    plan: &Arc<dyn ExecutionPlan>,
    context: &Arc<TaskContext>,
    streams: usize,
    at_once: bool,
) -> Result<Vec<RecordBatch>> {
    let mut out = vec![];
    if at_once {
        let handles = (0..streams)
            .map(|_| {
                let stream = plan.execute(0, Arc::clone(context));
                tokio::spawn(async move { collect(stream?).await })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            out.extend(
                tokio::time::timeout(Duration::from_secs(60), handle)
                    .await
                    .expect("join stalled")
                    .expect("join task panicked")?,
            );
        }
    } else {
        for _ in 0..streams {
            out.extend(collect(plan.execute(0, Arc::clone(context))?).await?);
        }
    }
    Ok(out)
}

/// Longbow's native hand-off opens one stream of partition 0 for each
/// driver, each over its own probe input. With open probe streams a join
/// that emits no build rows spills and gives each stream its rows; the same
/// streams on a closed join fail with an internal error instead of losing
/// rows. A join that emits build rows does not spill with open streams.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn open_probe_streams_share_partition_zero() -> Result<()> {
    let build = table("b", 1500, 211, 0, 100);
    let probe = table("p", 1200, 307, 7, 100);
    for join_type in [
        JoinType::Inner,
        JoinType::Right,
        JoinType::RightSemi,
        JoinType::RightAnti,
        JoinType::RightMark,
    ] {
        let case = Case {
            join_type,
            null_equality: NullEquality::NullEqualsNothing,
            filter: true,
            shape: Shape::CollectLeft(1),
        };
        for at_once in [true, false] {
            let label = format!("{join_type} at_once={at_once}");
            let reference = join_plan(&case, &build, &probe)?;
            let context =
                task_context(SessionConfig::new().with_batch_size(256), None, true);
            let want = batches_to_sort_string(
                &partition_zero_streams(&reference, &context, 3, at_once).await?,
            );
            let open = join_plan(&case, &build, &probe)?;
            open.downcast_ref::<HashJoinExec>()
                .unwrap()
                .set_open_probe_streams();
            let context = task_context(spill_config(1, 4096, 3), None, true);
            let actual = partition_zero_streams(&open, &context, 3, at_once).await?;
            assert_eq!(batches_to_sort_string(&actual), want, "{label}");
            assert!(spill_metrics(&open).0 > 0, "{label}: nothing was written");
            let closed = join_plan(&case, &build, &probe)?;
            let context = task_context(spill_config(1, 4096, 3), None, true);
            match partition_zero_streams(&closed, &context, 3, at_once).await {
                Ok(actual) => panic!(
                    "{label}: a closed join answered {} rows for 3 streams of one \
                     partition",
                    actual.iter().map(RecordBatch::num_rows).sum::<usize>()
                ),
                Err(error) => {
                    let text = error.to_string();
                    assert!(
                        text.contains("probes the build twice")
                            || text.contains("a stream started after every member"),
                        "{label}: {text}"
                    );
                }
            }
        }
    }
    let case = Case {
        join_type: JoinType::Left,
        null_equality: NullEquality::NullEqualsNothing,
        filter: false,
        shape: Shape::CollectLeft(1),
    };
    let plan = join_plan(&case, &build, &probe)?;
    plan.downcast_ref::<HashJoinExec>()
        .unwrap()
        .set_open_probe_streams();
    let context = task_context(spill_config(1, 4096, 3), None, true);
    partition_zero_streams(&plan, &context, 1, true).await?;
    assert_eq!(spill_metrics(&plan).0, 0, "an open Left join spilled");
    Ok(())
}

/// A split attempt that binds fewer partitions than the probe has: the
/// stock join never emits build rows then (its completion count covers
/// every probe partition), and the spilled join gives the same rows, with
/// the matches of the written partitions.
#[tokio::test]
async fn bound_range_build_rows_follow_the_stock_count() -> Result<()> {
    let build = table("b", 1500, 211, 0, 100);
    let probe = table("p", 1600, 307, 4, 100);
    for join_type in JOIN_TYPES {
        let case = Case {
            join_type,
            null_equality: NullEquality::NullEqualsNothing,
            filter: false,
            shape: Shape::CollectLeft(4),
        };
        let mut seen = vec![];
        for spilling in [false, true] {
            let plan = join_plan(&case, &build, &probe)?;
            plan.downcast_ref::<HashJoinExec>()
                .unwrap()
                .set_dynamic_filter_partitions(1..3)?;
            let mut config = spill_config(1, 4096, 3);
            config.options_mut().execution.enable_hash_join_spilling = spilling;
            let context = task_context(config, None, true);
            let mut out = vec![];
            for partition in 1..3 {
                out.extend(
                    collect(plan.execute(partition, Arc::clone(&context))?).await?,
                );
            }
            if spilling {
                assert!(
                    spill_metrics(&plan).0 > 0,
                    "{join_type}: nothing was written"
                );
            }
            seen.push(batches_to_sort_string(&out));
        }
        assert_eq!(seen[0], seen[1], "{join_type}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// leto-mjj.98: hot keys, the build-chunk join and role reversal
// (add-join-spill section 3).
// ---------------------------------------------------------------------------

pub(crate) mod bound;
mod skew;
