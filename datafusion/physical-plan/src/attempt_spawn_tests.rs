//! Real native producer families must register before their consumer returns.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::{DataFusionError, JoinType, Result};
use datafusion_execution::{TaskContext, config::SessionConfig};
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_expr::{LexOrdering, Partitioning, PhysicalSortExpr};
use futures::{FutureExt, StreamExt};

use crate::ExecutionPlan;
use crate::attempt_tasks::{AttemptTasks, DescendantStopped};
use crate::coalesce_partitions::CoalescePartitionsExec;
use crate::common::collect;
use crate::joins::PiecewiseMergeJoinExec;
use crate::materialized_cte::{
    MaterializedCteBuffer, MaterializedCteExec, MaterializedCteScanExec,
};
use crate::repartition::RepartitionExec;
use crate::sorts::sort::SortExec;
use crate::test::TestMemoryExec;
use crate::test::exec::BlockingExec;

fn context(scope: &Arc<AttemptTasks>) -> Arc<TaskContext> {
    Arc::new(
        TaskContext::default()
            .with_session_config(SessionConfig::new().with_extension(Arc::clone(scope))),
    )
}

fn key_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]))
}

#[tokio::test]
async fn coalesce_registers_before_spawn_and_drain_releases_input() {
    let scope = AttemptTasks::new();
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let input = BlockingExec::new(schema, 2);
    let refs = input.refs();
    let plan = CoalescePartitionsExec::new(Arc::new(input));
    // No producer has to be polled for the scope to own both registrations.
    let stream = plan.execute(0, context(&scope)).unwrap();
    scope.root_joined();
    assert!(scope.drain().now_or_never().is_none());
    assert!(refs.strong_count() > 0);
    scope.stop();
    drop(stream);
    drop(plan);
    scope.drain().await.unwrap();
    assert_eq!(refs.strong_count(), 0);
}

#[tokio::test]
async fn repartition_registers_pull_and_forwarder_before_drain() {
    let scope = AttemptTasks::new();
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let input = BlockingExec::new(schema, 2);
    let refs = input.refs();
    let plan =
        RepartitionExec::try_new(Arc::new(input), Partitioning::RoundRobinBatch(3))
            .unwrap();
    let mut stream = plan.execute(0, context(&scope)).unwrap();
    // Initial polling constructs the native input pull and forwarding tasks.
    assert!(futures::poll!(stream.next()).is_pending());
    scope.root_joined();
    assert!(scope.drain().now_or_never().is_none());
    assert!(refs.strong_count() > 0);
    scope.stop();
    drop(stream);
    drop(plan);
    scope.drain().await.unwrap();
    assert_eq!(refs.strong_count(), 0);
}

// New in DataFusion 56: the body of a materialized CTE is buffered by one
// spawned task per body partition.
#[tokio::test]
async fn materialized_cte_registers_body_tasks_before_drain() {
    let scope = AttemptTasks::new();
    let schema = key_schema();
    let body = BlockingExec::new(Arc::clone(&schema), 2);
    let refs = body.refs();
    let buffer = Arc::new(MaterializedCteBuffer::new(1, "t"));
    let scan = Arc::new(
        MaterializedCteScanExec::new(1, "t", Arc::clone(&schema), 1)
            .bind(Arc::clone(&buffer)),
    );
    let plan = MaterializedCteExec::new(Arc::new(body), Arc::clone(&scan) as _, buffer);
    let mut stream = plan.execute(0, context(&scope)).unwrap();
    // The first poll runs the body once and spawns its partition tasks.
    assert!(futures::poll!(stream.next()).is_pending());
    scope.root_joined();
    assert!(scope.drain().now_or_never().is_none());
    assert!(refs.strong_count() > 0);
    scope.stop();
    drop(stream);
    drop(scan);
    drop(plan);
    scope.drain().await.unwrap();
    assert_eq!(refs.strong_count(), 0);
}

// New in DataFusion 56: a right existence piecewise merge join folds every
// buffered partition in its own spawned task.
#[tokio::test]
async fn piecewise_merge_join_registers_buffered_folds_before_drain() {
    let scope = AttemptTasks::new();
    let schema = key_schema();
    let buffered = BlockingExec::new(Arc::clone(&schema), 2);
    let refs = buffered.refs();
    let streamed =
        TestMemoryExec::try_new_exec(&[vec![]], Arc::clone(&schema), None).unwrap();
    let on = (col("key", &schema).unwrap(), col("key", &schema).unwrap());
    let plan = PiecewiseMergeJoinExec::try_new(
        Arc::new(buffered),
        streamed,
        on,
        Operator::Lt,
        JoinType::RightSemi,
        1,
    )
    .unwrap();
    let mut stream = plan.execute(0, context(&scope)).unwrap();
    assert!(futures::poll!(stream.next()).is_pending());
    scope.root_joined();
    assert!(scope.drain().now_or_never().is_none());
    assert!(refs.strong_count() > 0);
    scope.stop();
    drop(stream);
    drop(plan);
    scope.drain().await.unwrap();
    assert_eq!(refs.strong_count(), 0);
}

fn two_run_sort(scope: &Arc<AttemptTasks>) -> (SortExec, Arc<TaskContext>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
    ]));
    let batch = |values: [i64; 3]| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(values.to_vec())),
                Arc::new(Int64Array::from(values.to_vec())),
            ],
        )
        .unwrap()
    };
    let input = TestMemoryExec::try_new_exec(
        &[vec![batch([3, 1, 2]), batch([6, 4, 5])]],
        Arc::clone(&schema),
        None,
    )
    .unwrap();
    // Two sort keys keep one run per batch; a zero in-place threshold sends the
    // two runs to the concurrent run preparation instead of one in-place sort.
    let ordering: LexOrdering = [
        PhysicalSortExpr::new_default(col("a", &schema).unwrap()),
        PhysicalSortExpr::new_default(col("b", &schema).unwrap()),
    ]
    .into();
    let mut config = SessionConfig::new().with_extension(Arc::clone(scope));
    config.options_mut().execution.sort_in_place_threshold_bytes = 0;
    let context = Arc::new(TaskContext::default().with_session_config(config));
    (SortExec::new(ordering, input), context)
}

// New in DataFusion 56: an in-memory sort prepares its runs as spawned jobs
// under a shared permit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sort_registers_concurrent_run_jobs() {
    let scope = AttemptTasks::new();
    let (sort, context) = two_run_sort(&scope);
    let batches = collect(sort.execute(0, context).unwrap()).await.unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 6);
    drop(sort);
    scope.root_joined();
    // Every registered job was acknowledged after its joined destruction.
    scope.drain().await.unwrap();

    // A stopped attempt never first-polls a newly registered job.
    let scope = AttemptTasks::new();
    let (sort, context) = two_run_sort(&scope);
    scope.stop();
    let error = collect(sort.execute(0, context).unwrap())
        .await
        .unwrap_err();
    let DataFusionError::External(error) = error.find_root() else {
        panic!("expected a stopped descendant, got {error}");
    };
    assert!(error.downcast_ref::<DescendantStopped>().is_some());
    drop(sort);
    scope.root_joined();
    scope.drain().await.unwrap();
}

/// A heavy output that holds a reservation and is slow to destroy.
struct HeavyOutput {
    _reservation: datafusion_execution::memory_pool::MemoryReservation,
    _payload: Vec<u8>,
    dropped: Arc<std::sync::atomic::AtomicBool>,
    panic: bool,
}

impl Drop for HeavyOutput {
    fn drop(&mut self) {
        std::thread::sleep(std::time::Duration::from_millis(100));
        self.dropped
            .store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(!self.panic, "heavy output destructor panic");
    }
}

// The producer ticket ends with the producer's final poll. A handle dropped
// while that poll completes must still keep the output owned: the drain waits
// for its destruction, and a destructor panic fails the drain.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scoped_task_dropped_during_completion_owns_its_output() {
    use crate::attempt_tasks::spawn_producer;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool,
    };
    use std::sync::atomic::{AtomicBool, Ordering};

    for panic in [false, true] {
        let scope = AttemptTasks::new();
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
        let reservation = MemoryConsumer::new("heavy output").register(&pool);
        reservation.try_grow(64 << 20).unwrap();
        let dropped = Arc::new(AtomicBool::new(false));
        let output = HeavyOutput {
            _reservation: reservation,
            _payload: vec![1; 64 << 20],
            dropped: Arc::clone(&dropped),
            panic,
        };
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let task = spawn_producer(
            Some(&scope),
            async move {
                entered_tx.send(()).unwrap();
                // Block inside the final poll: the handle is dropped while the
                // job completes, so the abort cannot take effect first.
                release_rx.recv().unwrap();
                output
            },
            |_stopped| unreachable!("the scope is not stopped"),
        )
        .unwrap();
        entered_rx.await.unwrap();
        drop(task);
        scope.root_joined();
        let mut drain = Box::pin(scope.drain());
        assert!(futures::poll!(drain.as_mut()).is_pending());
        release_tx.send(()).unwrap();
        let result = tokio::time::timeout(std::time::Duration::from_secs(10), drain)
            .await
            .expect("drain must finish");
        assert!(
            dropped.load(Ordering::SeqCst),
            "drain returned before the output was destroyed"
        );
        assert_eq!(pool.reserved(), 0);
        assert_eq!(
            result.is_err(),
            panic,
            "destructor panic must fail the drain"
        );
        assert!(!scope.poll_panicked());
    }
}

// The aggregate spill context (`aggregates/spill.rs`) is shared by the hash,
// single, clustered and partial-reduce streams in DataFusion 56. Its spill
// manager must carry the attempt scope, because merged spill reads are
// buffered producers. A closed scope refuses them; without the scope wiring
// the aggregate would spawn unregistered reads and succeed.
// Buffered spill reads spawn only on a multi-thread runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aggregate_spill_reads_register_with_the_attempt() -> Result<()> {
    use crate::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
    use arrow::compute::SortOptions;
    use datafusion_execution::memory_pool::FairSpillPool;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_functions_aggregate::sum::sum_udaf;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;

    let schema = Arc::new(Schema::new(vec![
        Field::new("b", DataType::Int64, false),
        Field::new("c", DataType::Int64, false),
    ]));
    let mut batches = Vec::new();
    for ordered in (0_i64..3).rev() {
        // Groups that share an ordered prefix stay in memory until the prefix
        // ends, which forces a clustered aggregation spill.
        for unordered in 1_i64..=16 {
            batches.push(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(vec![ordered])),
                    Arc::new(Int64Array::from(vec![unordered])),
                ],
            )?);
        }
    }
    let scan = TestMemoryExec::try_new(&[batches], Arc::clone(&schema), None)?
        .try_with_sort_information(vec![
            LexOrdering::new([PhysicalSortExpr::new(
                col("b", &schema)?,
                SortOptions::default().desc(),
            )])
            .unwrap(),
        ])?;
    let aggregate = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        PhysicalGroupBy::new(
            vec![
                (col("b", &schema)?, "b".to_string()),
                (col("c", &schema)?, "c".to_string()),
            ],
            vec![],
            vec![vec![false, false]],
            false,
        ),
        vec![Arc::new(
            AggregateExprBuilder::new(sum_udaf(), vec![col("c", &schema)?])
                .schema(Arc::clone(&schema))
                .alias("SUM(c)")
                .build()?,
        )],
        vec![None],
        Arc::new(scan) as Arc<dyn ExecutionPlan>,
        Arc::clone(&schema),
    )?);
    let context = |scope: Option<&Arc<AttemptTasks>>| -> Result<Arc<TaskContext>> {
        let mut config = SessionConfig::new()
            .with_batch_size(1)
            .set_bool("datafusion.execution.enable_migration_aggregate", true);
        if let Some(scope) = scope {
            config = config.with_extension(Arc::clone(scope));
        }
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(FairSpillPool::new(1024)))
            .build_arc()?;
        Ok(Arc::new(
            TaskContext::default()
                .with_session_config(config)
                .with_runtime(runtime),
        ))
    };

    // Unscoped, the plan spills and succeeds.
    let rows = collect(aggregate.execute(0, context(None)?)?).await?;
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 48);
    assert!(aggregate.metrics().unwrap().spill_count().unwrap() > 0);

    // A scope closed before execution refuses the spill read registration.
    let scope = AttemptTasks::new();
    scope.root_joined();
    let aggregate = Arc::clone(&aggregate).reset_state()?;
    let error = collect(aggregate.execute(0, context(Some(&scope))?)?)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("registration is closed"),
        "unexpected error: {error}"
    );
    Ok(())
}
