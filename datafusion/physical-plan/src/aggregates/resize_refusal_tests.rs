// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Aggregate streams whose table reservations are refused, as an adaptive
//! pool refuses growth once its lending headroom is gone.
//!
//! Partial stages may emit their materialized state and repeat a group in a
//! later batch. Final stages must spill their retained groups, or return
//! `ResourcesExhausted` when spilling is disabled.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arrow::array::{ArrayRef, AsArray, Int64Array, RecordBatch, StringViewArray};
use arrow::datatypes::{DataType, Field, Int64Type, Schema, SchemaRef};
use datafusion_common::{DataFusionError, Result, ScalarValue, resources_datafusion_err};
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion_execution::memory_pool::{
    MemoryPool, MemoryReservation, UnboundedMemoryPool,
};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_functions_aggregate::array_agg::array_agg_udaf;
use datafusion_functions_aggregate::count::count_udaf;
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_expr::expressions::col;
use futures::StreamExt;

use super::clustered_test_util::{distinct_key_batches, sum_over_sorted_input};
use super::group_values::packed::migration_needs_memory;
use super::order::GroupClusteringMode;
use super::{AggregateExec, AggregateMode, PhysicalGroupBy, StreamType};
use crate::ExecutionPlan;
use crate::test::TestMemoryExec;

/// A refusal count past which a stream is retrying without progress.
const RETRY_BOUND: usize = 64;

/// Numbers the growth requests of reservations that already hold memory, and
/// refuses the requests that `refuse` selects by number. The first request of
/// an empty reservation is always admitted. Only requests of at least
/// `min_bytes` are numbered and can be refused.
struct RefuseAfter {
    inner: UnboundedMemoryPool,
    min_bytes: usize,
    refuse: Box<dyn Fn(usize) -> bool + Send + Sync>,
    requests: AtomicUsize,
    refused: AtomicUsize,
}

impl RefuseAfter {
    fn new(
        min_bytes: usize,
        refuse: impl Fn(usize) -> bool + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner: UnboundedMemoryPool::default(),
            min_bytes,
            refuse: Box::new(refuse),
            requests: AtomicUsize::new(0),
            refused: AtomicUsize::new(0),
        })
    }

    /// Admits every request.
    fn unlimited() -> Arc<Self> {
        Self::new(0, |_| false)
    }

    /// Admits the first `admit` requests and refuses all later ones.
    fn after(admit: usize) -> Arc<Self> {
        Self::new(0, move |request| request >= admit)
    }

    fn requests(&self) -> usize {
        self.requests.load(Ordering::Relaxed)
    }

    fn refused(&self) -> usize {
        self.refused.load(Ordering::Relaxed)
    }
}

impl fmt::Debug for RefuseAfter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefuseAfter")
            .field("requests", &self.requests())
            .field("refused", &self.refused())
            .finish()
    }
}

impl fmt::Display for RefuseAfter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl MemoryPool for RefuseAfter {
    fn name(&self) -> &str {
        "RefuseAfter"
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        if reservation.size() != 0 && additional >= self.min_bytes {
            let request = self.requests.fetch_add(1, Ordering::Relaxed);
            if (self.refuse)(request) {
                let refused = self.refused.fetch_add(1, Ordering::Relaxed) + 1;
                assert!(
                    refused <= RETRY_BOUND,
                    "{refused} refusals: the stream retries without progress"
                );
                return Err(resources_datafusion_err!("no lending headroom"));
            }
        }
        self.inner.try_grow(reservation, additional)
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
}

/// Admits growth until `refuse` is set, then refuses every growth of a
/// reservation that already holds memory. Stage tests set it once they have
/// built the state whose handoff they check.
#[derive(Debug, Default)]
pub(super) struct RefuseWhenSet {
    inner: UnboundedMemoryPool,
    pub(super) refuse: AtomicBool,
}

impl fmt::Display for RefuseWhenSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl MemoryPool for RefuseWhenSet {
    fn name(&self) -> &str {
        "RefuseWhenSet"
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        if reservation.size() != 0 && self.refuse.load(Ordering::Relaxed) {
            return Err(resources_datafusion_err!("no lending headroom"));
        }
        self.inner.try_grow(reservation, additional)
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
}

fn context(pool: Arc<dyn MemoryPool>, spill: bool) -> Result<Arc<TaskContext>> {
    let mut runtime = RuntimeEnvBuilder::new().with_memory_pool(pool);
    if !spill {
        runtime = runtime.with_disk_manager_builder(
            DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
        );
    }
    Ok(Arc::new(
        TaskContext::default()
            .with_runtime(runtime.build_arc()?)
            .with_session_config(
            SessionConfig::new()
                .with_batch_size(256)
                .set_bool("datafusion.execution.enable_migration_aggregate", true)
                // Keep the probe from replacing the table under test.
                .set(
                    "datafusion.execution.skip_partial_aggregation_probe_ratio_threshold",
                    &ScalarValue::Float64(Some(1.0)),
                ),
        ),
    ))
}

/// Output rows keyed by every column but the last, with the last column's
/// values summed. `unique` asserts that no key repeats, as a final stage's
/// contract requires.
type Rows = BTreeMap<Vec<String>, i64>;

fn rows(output: &[RecordBatch], unique: bool) -> Rows {
    let mut rows = Rows::new();
    for batch in output {
        let keys = &batch.columns()[..batch.num_columns() - 1];
        let values = batch.columns().last().unwrap().as_primitive::<Int64Type>();
        for row in 0..batch.num_rows() {
            let key = keys
                .iter()
                .map(|column| {
                    arrow::util::display::array_value_to_string(column, row).unwrap()
                })
                .collect::<Vec<_>>();
            let previous = rows.insert(key.clone(), 0);
            assert!(
                !unique || previous.is_none(),
                "final output repeats group {key:?}"
            );
            *rows.get_mut(&key).unwrap() = previous.unwrap_or(0) + values.value(row);
        }
    }
    rows
}

fn assert_sorted_on_first_column(output: &[RecordBatch], refusals: &[usize]) {
    let keys: Vec<i64> = output
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_primitive::<Int64Type>()
                .values()
                .to_vec()
        })
        .collect();
    assert!(
        keys.is_sorted(),
        "refusing {refusals:?} breaks the declared output order"
    );
}

/// Runs partition 0 of `plan` under `pool` and checks that it releases the pool.
async fn run(
    plan: &AggregateExec,
    pool: Arc<dyn MemoryPool>,
    spill: bool,
) -> Result<Vec<RecordBatch>> {
    let mut stream = plan.execute(0, context(Arc::clone(&pool), spill)?)?;
    let mut output = vec![];
    let result = loop {
        match stream.next().await {
            Some(Ok(batch)) => output.push(batch),
            Some(Err(error)) => break Err(error),
            None => break Ok(output),
        }
    };
    drop(stream);
    assert_eq!(
        pool.reserved(),
        0,
        "the stream must release its reservation"
    );
    result
}

/// Which requests each run of [`assert_unchanged_under_refusals`] refuses,
/// given the number of requests in an unlimited run.
#[derive(Clone, Copy)]
enum Schedule {
    /// Every admission point: admit the first `k` requests, refuse the rest.
    After,
    /// A transient shortage, which lets spill replay finish with memory of
    /// its own: every request `i`, alone, with `i + 1`, or followed by a
    /// window of two later requests `j` and `j + 1`. A refused handoff
    /// refuses both the batch and the table; the table can outgrow its
    /// reservation when completed groups are taken or after an earlier
    /// refusal.
    Windows,
}

impl Schedule {
    fn refusal_sets(self, requests: usize) -> Vec<Vec<usize>> {
        match self {
            Self::After => (0..requests)
                .map(|admit| (admit..requests).collect())
                .collect(),
            // A refusal can add requests, so later windows may start past the
            // unlimited run's last request.
            Self::Windows => (0..requests)
                .flat_map(|i| {
                    [vec![i], vec![i, i + 1]]
                        .into_iter()
                        .chain((i + 2..requests + 4).map(move |j| vec![i, j, j + 1]))
                })
                .collect(),
        }
    }

    fn pool(self, refusals: &[usize]) -> Arc<RefuseAfter> {
        match self {
            Self::After => RefuseAfter::after(refusals[0]),
            Self::Windows => {
                let refusals = refusals.to_vec();
                RefuseAfter::new(0, move |request| refusals.contains(&request))
            }
        }
    }
}

/// Runs `plan` under every refusal set of `schedule`. Every run must produce
/// the unlimited result; a final stage that cannot spill may instead return
/// `ResourcesExhausted`. Returns how many runs refused a request.
async fn assert_unchanged_under_refusals(
    plan: &AggregateExec,
    spill: bool,
    may_exhaust: bool,
    schedule: Schedule,
) -> Result<usize> {
    let unique = matches!(
        plan.mode(),
        AggregateMode::Final | AggregateMode::FinalPartitioned | AggregateMode::Single
    );
    // A clustered stream declares its output sorted on the first key.
    let ordered = plan.properties().output_ordering().is_some();
    let unlimited = RefuseAfter::unlimited();
    let expected = rows(
        &run(plan, Arc::clone(&unlimited) as _, spill).await?,
        unique,
    );
    let requests = unlimited.requests();
    assert!(
        requests > 0,
        "the plan must grow an established reservation"
    );
    let mut refused_runs = 0;
    for refusals in schedule.refusal_sets(requests) {
        let pool = schedule.pool(&refusals);
        match run(plan, Arc::clone(&pool) as _, spill).await {
            Ok(output) => {
                if ordered {
                    assert_sorted_on_first_column(&output, &refusals);
                }
                let output_rows = rows(&output, unique);
                assert_eq!(output_rows, expected, "refusing {refusals:?}");
            }
            Err(DataFusionError::ResourcesExhausted(_)) if may_exhaust => {}
            Err(error) => panic!("refusing {refusals:?} of {requests}: {error}"),
        }
        refused_runs += usize::from(pool.refused() > 0);
    }
    Ok(refused_runs)
}

/// `COUNT(v) GROUP BY a, b, label` in `mode` over two copies of wide keys.
/// Wide keys make the packed interner keep its general fallback after an
/// emission, so the emptied table retains more than its initial reservation.
pub(super) fn count_wide_keys(mode: AggregateMode) -> Result<AggregateExec> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("label", DataType::Utf8View, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let keys: Vec<_> = (0..64)
        .map(|i| {
            if i % 2 == 0 {
                i64::MIN + i
            } else {
                i64::MAX - i
            }
        })
        .collect();
    let batch = |count: i64| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(keys.clone())),
                Arc::new(Int64Array::from(keys.clone())),
                Arc::new(StringViewArray::from_iter_values(
                    (0..keys.len()).map(|i| format!("value-{}", i % 9)),
                )),
                Arc::new(Int64Array::from(vec![count; keys.len()])),
            ],
        )
    };
    let group_by = PhysicalGroupBy::new_single(
        ["a", "b", "label"]
            .into_iter()
            .map(|name| Ok((col(name, &schema)?, name.to_string())))
            .collect::<Result<_>>()?,
    );
    count_over(mode, group_by, &schema, vec![batch(1)?, batch(2)?])
}

fn count_over(
    mode: AggregateMode,
    group_by: PhysicalGroupBy,
    schema: &SchemaRef,
    batches: Vec<RecordBatch>,
) -> Result<AggregateExec> {
    let count = AggregateExprBuilder::new(count_udaf(), vec![col("v", schema)?])
        .schema(Arc::clone(schema))
        .alias("count")
        .build()?;
    let input = TestMemoryExec::try_new_exec(&[batches], Arc::clone(schema), None)?;
    AggregateExec::try_new(
        mode,
        group_by,
        vec![Arc::new(count)],
        vec![None],
        input,
        Arc::clone(schema),
    )
}

/// A two-row batch, then [`distinct_key_batches`]. The table reservation
/// admitted for the first batch is small, so a table that grew for the next
/// batch and was refused can retain more than its reservation.
fn small_then_distinct_key_batches() -> Vec<Vec<Vec<i64>>> {
    let mut batches = vec![vec![vec![0, 0]]];
    batches.extend(distinct_key_batches());
    batches
}

/// Two-key batches sorted on `k` only, so the keys have partial clustering.
fn partially_clustered(batches: Vec<Vec<Vec<i64>>>) -> Vec<Vec<Vec<i64>>> {
    batches
        .into_iter()
        .map(|mut columns| {
            let second = columns[0].iter().map(|k| k % 5).collect();
            columns.push(second);
            columns
        })
        .collect()
}

#[tokio::test]
async fn partial_hash_emits_when_the_emptied_table_is_refused() -> Result<()> {
    let plan = count_wide_keys(AggregateMode::Partial)?;
    assert!(matches!(
        plan.execute_typed(0, &context(RefuseAfter::unlimited(), false)?)?,
        StreamType::PartialHash(_)
    ));
    assert!(
        assert_unchanged_under_refusals(&plan, false, false, Schedule::After).await? > 0
    );
    Ok(())
}

#[tokio::test]
async fn clustered_partial_emits_when_the_retained_table_is_refused() -> Result<()> {
    for (keys, batches) in [
        (&["k"][..], distinct_key_batches()),
        (&["k"][..], small_then_distinct_key_batches()),
        (&["k", "j"][..], partially_clustered(distinct_key_batches())),
        (
            &["k", "j"][..],
            partially_clustered(small_then_distinct_key_batches()),
        ),
    ] {
        let plan = sum_over_sorted_input(AggregateMode::Partial, keys, batches)?;
        assert!(matches!(
            plan.execute_typed(0, &context(RefuseAfter::unlimited(), false)?)?,
            StreamType::ClusteredPartialAggregate(_)
        ));
        assert!(
            assert_unchanged_under_refusals(&plan, false, false, Schedule::After).await?
                > 0
        );
    }
    Ok(())
}

#[tokio::test]
async fn clustered_final_spills_when_the_retained_table_is_refused() -> Result<()> {
    for batches in [distinct_key_batches(), small_then_distinct_key_batches()] {
        let plan = sum_over_sorted_input(
            AggregateMode::Final,
            &["k", "j"],
            partially_clustered(batches),
        )?;
        assert_eq!(
            plan.group_clustering_mode(),
            &GroupClusteringMode::Partial(vec![0])
        );
        assert!(
            assert_unchanged_under_refusals(&plan, true, false, Schedule::Windows)
                .await?
                > 0
        );
    }
    Ok(())
}

/// The wide-key table keeps its general interner after emitting all groups.
/// When even the emptied table cannot grow, return the pool error directly.
#[rstest::rstest]
#[case::single(AggregateMode::Single)]
#[case::final_hash(AggregateMode::Final)]
#[tokio::test]
async fn post_spill_resize_refusal_is_typed(#[case] mode: AggregateMode) -> Result<()> {
    let plan = count_wide_keys(mode)?;
    let pool = RefuseAfter::after(0);
    let error = run(&plan, Arc::clone(&pool) as _, true)
        .await
        .expect_err("the retained table needs more than its initial grant");
    assert!(
        matches!(&error, DataFusionError::ResourcesExhausted(message) if message.contains("no lending headroom")),
        "{mode:?}: {error:?}"
    );
    assert_eq!(pool.refused(), 2, "initial and post-spill resize refusals");
    Ok(())
}

/// If the spill fails, report it even when the emptied table also needs more
/// memory. The caller needs the spill error to diagnose the failed query.
#[tokio::test]
async fn final_hash_preserves_spill_error_under_resize_refusal() -> Result<()> {
    let plan = count_wide_keys(AggregateMode::Final)?;
    let pool = RefuseAfter::after(0);
    let task = context(Arc::clone(&pool) as _, true)?;
    task.runtime_env()
        .disk_manager
        .set_max_temp_directory_size(0)?;
    let error = crate::common::collect(plan.execute(0, task)?)
        .await
        .expect_err("the spill exceeds the zero-byte disk limit");
    assert!(
        error.to_string().contains("exceeded the allowable limit"),
        "{error:?}"
    );
    assert_eq!(pool.refused(), 1, "only the initial resize was attempted");
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

/// Limits a final hash table so that it spills, and numbers the growth
/// requests of the spill replay table, refusing every request from `admit` on.
/// The replay merge has its own consumer and is admitted, so only the replay
/// table runs without lending headroom.
#[derive(Debug)]
struct RefuseReplayTable {
    inner: UnboundedMemoryPool,
    table_limit: usize,
    admit: usize,
    requests: AtomicUsize,
    refused: AtomicUsize,
}

impl RefuseReplayTable {
    fn new(table_limit: usize, admit: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: UnboundedMemoryPool::default(),
            table_limit,
            admit,
            requests: AtomicUsize::new(0),
            refused: AtomicUsize::new(0),
        })
    }
}

impl fmt::Display for RefuseReplayTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl MemoryPool for RefuseReplayTable {
    fn name(&self) -> &str {
        "RefuseReplayTable"
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        let name = reservation.consumer().name();
        if name.starts_with("AggregateSpillReplayTable") && reservation.size() != 0 {
            if self.requests.fetch_add(1, Ordering::Relaxed) >= self.admit {
                self.refused.fetch_add(1, Ordering::Relaxed);
                return Err(resources_datafusion_err!("no lending headroom"));
            }
        } else if name.starts_with("FinalHashAggregateStream")
            && reservation.size() + additional > self.table_limit
        {
            return Err(resources_datafusion_err!("table limit"));
        }
        self.inner.try_grow(reservation, additional)
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
}

/// Spill replay merges the runs through a fully clustered final stream that
/// cannot spill. When its table is refused from any admission point on, the
/// replay must still finish with the unlimited result: it emits its completed
/// groups and charges the memory the active group already holds.
///
/// `array_agg` states grow as replay merges a key's states from the runs, so
/// the active group can outgrow the reservation admitted before the batch.
#[tokio::test]
async fn spill_replay_finishes_when_its_table_is_refused() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    // Each partition holds every key, so each key has one state per
    // partition, and the states recur across spill runs.
    let partitions = (0..64)
        .map(|partition| {
            let rows = 0..64 * 128;
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(rows.clone().map(|i| i % 64))),
                    Arc::new(Int64Array::from_iter_values(
                        rows.map(|i| partition * 64 * 128 + i),
                    )),
                ],
            )
            .map(|batch| vec![batch])
        })
        .collect::<Result<Vec<_>, _>>()?;
    let array_agg = AggregateExprBuilder::new(array_agg_udaf(), vec![col("v", &schema)?])
        .schema(Arc::clone(&schema))
        .alias("array_agg")
        .build()?;
    let partial = AggregateExec::try_new(
        AggregateMode::Partial,
        PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
        vec![Arc::new(array_agg)],
        vec![None],
        TestMemoryExec::try_new_exec(&partitions, Arc::clone(&schema), None)?,
        Arc::clone(&schema),
    )?;
    let unlimited_context = context(RefuseAfter::unlimited(), true)?;
    let mut states = vec![];
    for partition in 0..partitions.len() {
        let stream = partial.execute(partition, Arc::clone(&unlimited_context))?;
        states.extend(crate::common::collect(stream).await?);
    }
    let plan = AggregateExec::try_new(
        AggregateMode::Final,
        partial.group_by().as_final(),
        partial.aggr_expr().to_vec(),
        vec![None],
        TestMemoryExec::try_new_exec(&[states], partial.schema(), None)?,
        Arc::clone(&partial.input_schema),
    )?;
    assert!(matches!(
        plan.execute_typed(0, &unlimited_context)?,
        StreamType::FinalHash(_)
    ));
    let expected = lists(&run(&plan, RefuseAfter::unlimited() as _, true).await?);
    assert_eq!(expected.len(), 64);

    // Small enough to spill several runs. The merge reservation is a sibling
    // of the hash table's, so the limit must also fit the merge.
    const TABLE_LIMIT: usize = 4 * 1024 * 1024;
    let admitted = RefuseReplayTable::new(TABLE_LIMIT, usize::MAX);
    let output = run(&plan, Arc::clone(&admitted) as _, true).await?;
    assert!(plan.metrics().unwrap().spill_count().unwrap() > 1);
    assert_eq!(lists(&output), expected);
    let requests = admitted.requests.load(Ordering::Relaxed);
    assert!(requests > 0, "the replay table must grow");

    for admit in 0..requests {
        let pool = RefuseReplayTable::new(TABLE_LIMIT, admit);
        let output = run(&plan, Arc::clone(&pool) as _, true)
            .await
            .unwrap_or_else(|error| panic!("admitting {admit} of {requests}: {error}"));
        assert!(pool.refused.load(Ordering::Relaxed) > 0);
        assert_eq!(lists(&output), expected, "admitting {admit}");
    }
    Ok(())
}

/// The sorted `array_agg` values of each key in a final output, which must
/// not repeat a key.
fn lists(output: &[RecordBatch]) -> BTreeMap<i64, Vec<i64>> {
    let mut lists = BTreeMap::new();
    for batch in output {
        let keys = batch.column(0).as_primitive::<Int64Type>();
        let values = batch.column(1).as_list::<i32>();
        for row in 0..batch.num_rows() {
            let mut list = values
                .value(row)
                .as_primitive::<Int64Type>()
                .values()
                .to_vec();
            list.sort_unstable();
            let key = keys.value(row);
            assert!(
                lists.insert(key, list).is_none(),
                "final output repeats group {key}"
            );
        }
    }
    lists
}

#[tokio::test]
async fn clustered_final_without_spilling_returns_resources_exhausted() -> Result<()> {
    for (keys, batches) in [
        (&["k"][..], distinct_key_batches()),
        (&["k"][..], small_then_distinct_key_batches()),
        (&["k", "j"][..], partially_clustered(distinct_key_batches())),
        (
            &["k", "j"][..],
            partially_clustered(small_then_distinct_key_batches()),
        ),
    ] {
        let plan = sum_over_sorted_input(AggregateMode::Final, keys, batches)?;
        assert!(
            assert_unchanged_under_refusals(&plan, false, true, Schedule::After).await?
                > 0
        );
    }
    Ok(())
}

/// A single-key `COUNT(v)` in `mode` whose second batch widens the key domain,
/// so the packed table must migrate its 64 groups to the general interner.
fn widening_count(mode: AggregateMode) -> Result<(AggregateExec, Rows)> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let narrow: Vec<i64> = (0..64).collect();
    let wide: Vec<i64> = (0..4096)
        .map(|i| match i % 3 {
            0 => i % 64,
            1 => i64::MIN + i,
            _ => i64::MAX - i,
        })
        .collect();
    let mut expected = Rows::new();
    let mut batch = |keys: Vec<i64>| {
        for &key in &keys {
            *expected.entry(vec![key.to_string()]).or_default() += 1;
        }
        let ones: ArrayRef = Arc::new(Int64Array::from(vec![1; keys.len()]));
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(keys)), ones],
        )
    };
    let batches = vec![batch(narrow)?, batch(wide)?];
    let group_by = PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]);
    Ok((count_over(mode, group_by, &schema, batches)?, expected))
}

/// The migration grant is the only request of at least this size.
const MIGRATION_BYTES: usize = 64 * 1024;

#[rstest::rstest]
#[case::partial(AggregateMode::Partial)]
#[case::partial_reduce(AggregateMode::PartialReduce)]
#[tokio::test]
async fn refused_migration_emits_and_retries_the_batch(
    #[case] mode: AggregateMode,
) -> Result<()> {
    let (plan, expected) = widening_count(mode)?;
    let pool = RefuseAfter::new(MIGRATION_BYTES, |request| request == 0);
    let output = run(&plan, Arc::clone(&pool) as _, false).await?;
    assert_eq!(
        pool.refused(),
        1,
        "the migration grant must be refused once"
    );
    assert!(
        output.len() >= 2,
        "the table must be emitted before the batch is retried"
    );
    // Both stages may repeat a group, so the counts are summed by key.
    assert_eq!(rows(&output, false), expected);
    Ok(())
}

#[rstest::rstest]
#[case::partial(AggregateMode::Partial)]
#[case::partial_reduce(AggregateMode::PartialReduce)]
#[tokio::test]
async fn refused_migration_of_an_empty_table_is_returned(
    #[case] mode: AggregateMode,
) -> Result<()> {
    let (plan, _) = widening_count(mode)?;
    let pool = RefuseAfter::new(MIGRATION_BYTES, |_| true);
    let error = run(&plan, Arc::clone(&pool) as _, false)
        .await
        .expect_err("an empty table cannot release memory for its migration");
    assert!(migration_needs_memory(&error), "{error}");
    // The first refusal emits the 64 narrow groups. The retry refuses the
    // emptied table's migration, which must end the stream.
    assert_eq!(pool.refused(), 2);
    Ok(())
}
