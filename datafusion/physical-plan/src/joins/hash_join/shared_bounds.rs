// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Utilities for shared build-side information. Used in dynamic filter pushdown in Hash Joins.
// TODO: include the link to the Dynamic Filter blog post.

use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Weak};

use crate::ExecutionPlan;
use crate::ExecutionPlanProperties;
use crate::Partitioning;
use crate::joins::Map;
use crate::joins::PartitionMode;
use crate::joins::hash_join::exec::HASH_JOIN_SEED;
use crate::joins::hash_join::inlist_builder::build_struct_fields;
use crate::joins::hash_join::not_in::NotIn;
use crate::joins::hash_join::partitioned_hash_eval::{
    HashExpr, HashTableLookupExpr, PartitionRouter, SeededRandomState,
};
use crate::joins::join_hash_map::{JoinHashMapType, JoinHashMapU32};
use crate::joins::key_range_bitmap::{KeyRangeBitmap, KeyRangePart};
use crate::repartition::RangeExpr;
use arrow::array::{ArrayRef, AsArray};
use arrow::compute::concat;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::config::{ConfigOptions, OptimizerOptions};
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::{
    DataFusionError, NullEquality, Result, ScalarValue, SharedResult,
    assert_or_internal_err, internal_err,
};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_expr::Operator;
use datafusion_functions::core::r#struct as struct_func;
use datafusion_physical_expr::expressions::{
    BinaryExpr, CaseExpr, DynamicFilterPhysicalExpr, InListExpr, IsNullExpr, lit,
};
use datafusion_physical_expr::{
    PhysicalExpr, PhysicalExprRef, RangePartitioning, ScalarFunctionExpr,
};

use parking_lot::Mutex;
use tokio::sync::Notify;

/// Represents the minimum and maximum values for a specific column.
/// Used in dynamic filter pushdown to establish value boundaries.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ColumnBounds {
    /// The minimum value observed for this column
    pub(crate) min: ScalarValue,
    /// The maximum value observed for this column
    pub(crate) max: ScalarValue,
}

impl ColumnBounds {
    pub(crate) fn new(min: ScalarValue, max: ScalarValue) -> Self {
        Self { min, max }
    }
}

/// Represents the bounds for all join key columns from a single partition.
/// This contains the min/max values computed from one partition's build-side data.
#[derive(Debug, Clone)]
pub(crate) struct PartitionBounds {
    /// Min/max bounds for each join key column in this partition.
    /// Index corresponds to the join key expression index.
    column_bounds: Vec<ColumnBounds>,
}

impl PartitionBounds {
    pub(crate) fn new(column_bounds: Vec<ColumnBounds>) -> Self {
        Self { column_bounds }
    }

    pub(crate) fn get_column_bounds(&self, index: usize) -> Option<&ColumnBounds> {
        self.column_bounds.get(index)
    }

    /// Bounds covering both `self` and `other`, which must describe the same
    /// columns. A NULL bound means the partition holds no non-null key in that
    /// column, so it never widens the other side's bound.
    fn union(self, other: &PartitionBounds) -> PartitionBounds {
        // Every non-empty partition of a join reports one bound per join key.
        debug_assert_eq!(
            self.column_bounds.len(),
            other.column_bounds.len(),
            "partitions report bounds for different numbers of columns"
        );
        let column_bounds = self
            .column_bounds
            .into_iter()
            .zip(&other.column_bounds)
            .map(|(a, b)| {
                let min = if !b.min.is_null() && (a.min.is_null() || b.min < a.min) {
                    b.min.clone()
                } else {
                    a.min
                };
                let max = if !b.max.is_null() && (a.max.is_null() || b.max > a.max) {
                    b.max.clone()
                } else {
                    a.max
                };
                ColumnBounds::new(min, max)
            })
            .collect();
        PartitionBounds::new(column_bounds)
    }
}

/// Creates a membership predicate for filter pushdown.
///
/// If `inlist_values` is provided (for small build sides), creates an InList expression.
/// Otherwise, creates a HashTableLookup expression (for large build sides).
///
/// Supports both single-column and multi-column joins using struct expressions.
fn create_membership_predicate(
    on_right: &[PhysicalExprRef],
    pushdown: PushdownStrategy,
    random_state: &SeededRandomState,
    schema: &Schema,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    match pushdown {
        // Use InList expression for small build sides
        PushdownStrategy::InList(in_list_array) => {
            // Build the expression to compare against
            let expr = if on_right.len() == 1 {
                // Single column: col IN (val1, val2, ...)
                Arc::clone(&on_right[0])
            } else {
                let fields = build_struct_fields(
                    on_right
                        .iter()
                        .map(|r| r.data_type(schema))
                        .collect::<Result<Vec<_>>>()?
                        .as_ref(),
                )?;

                // The return field name and the function field name don't really matter here.
                let return_field =
                    Arc::new(Field::new("struct", DataType::Struct(fields), true));

                Arc::new(ScalarFunctionExpr::new(
                    "struct",
                    struct_func(),
                    on_right.to_vec(),
                    return_field,
                    Arc::new(ConfigOptions::default()),
                )) as Arc<dyn PhysicalExpr>
            };

            // Use InListExpr::try_new_from_array() to build an InList with static_filter optimization (hash-based lookup)
            Ok(Some(Arc::new(InListExpr::try_new_from_array(
                expr,
                in_list_array,
                false,
                schema,
            )?)))
        }
        // Use hash table lookup for large build sides
        PushdownStrategy::Map(hash_map, pruning_bitmap, build_side) => {
            let lookup = HashTableLookupExpr::new(
                on_right.to_vec(),
                random_state.clone(),
                None,
                vec![Some(hash_map)],
                "hash_lookup".to_string(),
                pruning_bitmap,
            )?;
            // Longbow family E: the build keys and memory consumer behind the map.
            let lookup = match build_side {
                Some(MapBuildSide { keys, memory }) => {
                    lookup.with_build_side(vec![vec![keys]], Some(memory))?
                }
                None => lookup,
            };
            Ok(Some(Arc::new(lookup) as Arc<dyn PhysicalExpr>))
        }
        // Empty partition - should not create a filter for this
        PushdownStrategy::Empty => Ok(None),
        // Longbow family JS: the bounds predicate alone.
        PushdownStrategy::BoundsOnly => Ok(None),
    }
}

/// Creates a bounds predicate from partition bounds.
///
/// Returns `None` if no column bounds are available.
/// Returns a combined predicate (col >= min AND col <= max) for all columns with bounds.
fn create_bounds_predicate(
    on_right: &[PhysicalExprRef],
    bounds: &PartitionBounds,
) -> Option<Arc<dyn PhysicalExpr>> {
    let mut column_predicates = Vec::with_capacity(on_right.len());

    for (col_idx, right_expr) in on_right.iter().enumerate() {
        if let Some(column_bounds) = bounds.get_column_bounds(col_idx) {
            // Create predicate: col >= min AND col <= max
            let min_expr = Arc::new(BinaryExpr::new(
                Arc::clone(right_expr),
                Operator::GtEq,
                lit(column_bounds.min.clone()),
            )) as Arc<dyn PhysicalExpr>;
            let max_expr = Arc::new(BinaryExpr::new(
                Arc::clone(right_expr),
                Operator::LtEq,
                lit(column_bounds.max.clone()),
            )) as Arc<dyn PhysicalExpr>;
            let range_expr = Arc::new(BinaryExpr::new(min_expr, Operator::And, max_expr))
                as Arc<dyn PhysicalExpr>;
            column_predicates.push(range_expr);
        }
    }

    if column_predicates.is_empty() {
        None
    } else {
        Some(
            column_predicates
                .into_iter()
                .reduce(|acc, pred| {
                    Arc::new(BinaryExpr::new(acc, Operator::And, pred))
                        as Arc<dyn PhysicalExpr>
                })
                .unwrap(),
        )
    }
}

/// Combines a membership predicate and a bounds predicate with logical AND.
///
/// Returns `None` when neither is available; callers decide the fallback (e.g.
/// skip updating the filter vs. emit a `lit(true)` branch inside a CASE).
fn combine_membership_and_bounds(
    membership_expr: Option<Arc<dyn PhysicalExpr>>,
    bounds_expr: Option<Arc<dyn PhysicalExpr>>,
) -> Option<Arc<dyn PhysicalExpr>> {
    match (membership_expr, bounds_expr) {
        (Some(membership), Some(bounds)) => {
            Some(Arc::new(BinaryExpr::new(bounds, Operator::And, membership))
                as Arc<dyn PhysicalExpr>)
        }
        (Some(membership), None) => Some(membership),
        (None, Some(bounds)) => Some(bounds),
        (None, None) => None,
    }
}

/// Coordinates build-side information collection across multiple partitions
///
/// This structure collects information from the build side (hash tables and/or bounds) and
/// ensures that dynamic filters are built with complete information from all relevant
/// partitions before being applied to probe-side scans. Incomplete filters would
/// incorrectly eliminate valid join results.
///
/// ## Synchronization Strategy
///
/// Without an execution scope, the whole plan uses the barrier below. A
/// Longbow task scope ([`Self::with_execution_partitions`], family SF)
/// instead publishes a conservative snapshot after each assigned report and
/// never parks a stream behind a sibling that has not started: unknown
/// buckets keep every row. Completion still requires every assigned report
/// or cancellation; dropping the last owner closes retained observers.
///
/// 1. Each partition computes information from its build-side data (hash maps and/or bounds)
/// 2. Information is stored in the shared state, which tracks how many partitions have reported
/// 3. When the last partition reports, one waiter is elected as the finalizer; it merges the
///    collected information, updates the dynamic filter exactly once, and publishes the
///    terminal result by transitioning [`CompletionState`] to `Ready`
/// 4. A [`tokio::sync::Notify`] wakes any other partitions parked in `wait_for_completion`,
///    which then observe the `Ready` state under the mutex and return immediately
///
/// ## Hash Map vs Bounds
///
/// - **Hash Maps (Partitioned mode)**: Collects Arc references to hash tables from each partition.
///   Creates a `PartitionedHashLookupPhysicalExpr` that routes rows to the correct partition's hash table.
/// - **Bounds (CollectLeft mode)**: Collects min/max bounds and creates range predicates.
///
/// ## Partition Counting
///
/// The `total_partitions` count represents how many times `collect_build_side` will be called:
/// - **CollectLeft**: Number of output partitions (each accesses shared build data)
/// - **Partitioned**: Number of input partitions (each builds independently)
///
/// ## Thread Safety
///
/// All fields use a single mutex to ensure correct coordination between concurrent
/// partition executions.
pub(crate) struct SharedBuildAccumulator {
    /// Build-side data protected by a single mutex to avoid ordering concerns
    inner: Mutex<AccumulatorState>,
    /// Wakes every partition that is parked in [`Self::wait_for_completion`]
    /// once [`AccumulatorState::completion`] transitions to
    /// [`CompletionState::Ready`]. Notifications are fired once per
    /// accumulator lifetime (the elected finalizer publishes the terminal
    /// result, then broadcasts), so late subscribers simply re-check the
    /// state under the mutex and return immediately.
    completion_notify: Notify,
    /// Dynamic filter for pushdown to probe side
    dynamic_filter: Arc<DynamicFilterPhysicalExpr>,
    /// Right side join expressions needed for creating filter expressions
    on_right: Vec<PhysicalExprRef>,
    /// Random state for partitioning (RepartitionExec's hash function with 0,0,0,0 seeds)
    /// Used for PartitionedHashLookupPhysicalExpr
    repartition_random_state: SeededRandomState,
    /// Schema of the probe (right) side for evaluating filter expressions
    probe_schema: Arc<Schema>,
    /// Probe-side Range routing metadata for partitioned dynamic filters.
    probe_range_partitioning: Option<RangePartitioning>,
    /// Null equality of the join. Under `NullEqualsNull` a probe-side NULL can match a
    /// build-side NULL, so the pushed filter must keep NULL rows here too.
    null_equality: NullEquality,
    /// Null-aware anti join (`NOT IN`). A probe-side NULL must reach the join so its
    /// three-valued logic can collapse the result, so the pushed filter keeps NULL rows.
    not_in: Option<NotIn>,
    /// Largest merged IN list, in bytes, that partitioned filters keep as an IN
    /// list: `hash_join_inlist_pushdown_max_size`, as for a single build.
    in_list_max_size: usize,
    /// Largest merged IN list, in values, that partitioned filters keep as an IN
    /// list: `hash_join_inlist_pushdown_max_distinct_values`, as for a single build.
    in_list_max_values: usize,
}

/// Strategy for filter pushdown (decided at collection time)
#[derive(Clone)]
pub(crate) enum PushdownStrategy {
    /// Use InList for small build sides (< 128MB)
    InList(ArrayRef),
    /// Use map lookup for large build sides. If `Some` the second field
    /// represents the build side's keys using a bucket bitmap. The third field
    /// is the build side behind the map (Longbow family E).
    Map(Arc<Map>, Option<Arc<KeyRangeBitmap>>, Option<MapBuildSide>),
    /// There was no data in this partition, do not build a dynamic filter for it
    Empty,
    /// Longbow family JS: the build wrote partitions to disk, so no
    /// membership form describes all of its rows; only its bounds filter.
    BoundsOnly,
}

/// The key arrays a [`PushdownStrategy::Map`] was built from and an empty
/// reservation of the build's memory consumer, which the lookup exposes to
/// structures derived from the build keys (Longbow family E).
#[derive(Clone)]
pub(crate) struct MapBuildSide {
    /// The build's key arrays, one per join key, held weakly.
    pub(crate) keys: Weak<[ArrayRef]>,
    /// An empty reservation of the build's memory consumer.
    pub(crate) memory: Arc<MemoryReservation>,
}

/// Build-side data reported by a single partition
pub(crate) enum PartitionBuildData {
    Partitioned {
        partition_id: usize,
        pushdown: PushdownStrategy,
        bounds: PartitionBounds,
        keys_have_null: bool,
    },
    CollectLeft {
        /// The reporting output partition; a task scope deduplicates by it.
        partition_id: usize,
        pushdown: PushdownStrategy,
        bounds: PartitionBounds,
        keys_have_null: bool,
    },
}

/// Per-partition accumulated data (Partitioned mode)
#[derive(Clone)]
struct PartitionData {
    bounds: PartitionBounds,
    pushdown: PushdownStrategy,
    /// Whether any build key of this partition is NULL. Decides whether the pushed
    /// filter must keep probe-side NULL rows for a null-equal join to match them.
    keys_have_null: bool,
}

/// Build-side data organized by partition mode
enum AccumulatedBuildData {
    Partitioned {
        partitions: Vec<PartitionStatus>,
        completed_partitions: usize,
    },
    CollectLeft {
        data: PartitionStatus,
        reported_count: usize,
        expected_reports: usize,
    },
}

enum CompletionState {
    Pending,
    Finalizing,
    Ready(SharedResult<()>),
}

struct AccumulatorState {
    data: AccumulatedBuildData,
    completion: CompletionState,
    /// Family SF: the output partitions of a task scope and which of them
    /// reported or cancelled; `None` for the whole-plan barrier.
    task_reports: Option<TaskReports>,
}

/// Family SF: the output partitions that one execution instance runs.
struct TaskReports {
    partitions: Range<usize>,
    reported: Vec<bool>,
    /// The filter generation this accumulator created or last published. A
    /// different generation at drop means someone else (a recursive-CTE
    /// `reset_state` for the next iteration) owns the filter now.
    generation: u64,
}

#[derive(Clone)]
enum PartitionStatus {
    Pending,
    Reported(PartitionData),
    CanceledUnknown,
}

#[derive(Clone)]
enum FinalizeInput {
    Partitioned(Vec<PartitionStatus>),
    CollectLeft(PartitionStatus),
}

impl SharedBuildAccumulator {
    /// Creates a new SharedBuildAccumulator configured for the given partition mode
    ///
    /// This method calculates how many times `collect_build_side` will be called based on the
    /// partition mode's execution pattern. This count is critical for determining when we have
    /// complete information from all partitions to build the dynamic filter.
    ///
    /// ## Partition Mode Execution Patterns
    ///
    /// - **CollectLeft**: Build side is collected ONCE from partition 0 and shared via `OnceFut`
    ///   across all output partitions. Each output partition calls `collect_build_side` to access the shared build data.
    ///   Although this results in multiple invocations, the  `report_partition_bounds` function contains deduplication logic to handle them safely.
    ///   Expected calls = number of output partitions.
    ///
    ///
    /// - **Partitioned**: Each partition independently builds its own hash table by calling
    ///   `collect_build_side` once. Expected calls = number of build partitions.
    ///
    /// - **Auto**: Placeholder mode resolved during optimization. Uses 1 as safe default since
    ///   the actual mode will be determined and a new accumulator created before execution.
    ///
    /// ## Why This Matters
    ///
    /// We cannot build a partial filter from some partitions - it would incorrectly eliminate
    /// valid join results. We must wait until we have complete information from ALL
    /// relevant partitions before updating the dynamic filter.
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new_from_partition_mode(
        partition_mode: PartitionMode,
        left_child: &dyn ExecutionPlan,
        right_child: &dyn ExecutionPlan,
        dynamic_filter: Arc<DynamicFilterPhysicalExpr>,
        on_right: Vec<PhysicalExprRef>,
        repartition_random_state: SeededRandomState,
        null_equality: NullEquality,
        not_in: Option<NotIn>,
        optimizer: &OptimizerOptions,
    ) -> Self {
        // Troubleshooting: If partition counts are incorrect, verify this logic matches
        // the actual execution pattern in collect_build_side()
        let expected_calls = match partition_mode {
            // Each output partition accesses shared build data
            PartitionMode::CollectLeft => {
                right_child.output_partitioning().partition_count()
            }
            // Each partition builds its own data
            PartitionMode::Partitioned => {
                left_child.output_partitioning().partition_count()
            }
            // Default value, will be resolved during optimization (does not exist once `execute()` is called; will be replaced by one of the other two)
            PartitionMode::Auto => unreachable!(
                "PartitionMode::Auto should not be present at execution time. This is a bug in DataFusion, please report it!"
            ),
        };

        let mode_data = match partition_mode {
            PartitionMode::Partitioned => AccumulatedBuildData::Partitioned {
                partitions: vec![
                    PartitionStatus::Pending;
                    left_child.output_partitioning().partition_count()
                ],
                completed_partitions: 0,
            },
            PartitionMode::CollectLeft => AccumulatedBuildData::CollectLeft {
                data: PartitionStatus::Pending,
                reported_count: 0,
                expected_reports: expected_calls,
            },
            PartitionMode::Auto => unreachable!(
                "PartitionMode::Auto should not be present at execution time. This is a bug in DataFusion, please report it!"
            ),
        };

        let probe_range_partitioning =
            match (partition_mode, right_child.output_partitioning()) {
                (PartitionMode::Partitioned, Partitioning::Range(range)) => {
                    Some(range.clone())
                }
                _ => None,
            };

        Self {
            inner: Mutex::new(AccumulatorState {
                data: mode_data,
                completion: CompletionState::Pending,
                task_reports: None,
            }),
            completion_notify: Notify::new(),
            dynamic_filter,
            on_right,
            repartition_random_state,
            probe_schema: right_child.schema(),
            probe_range_partitioning,
            null_equality,
            not_in,
            in_list_max_size: optimizer.hash_join_inlist_pushdown_max_size,
            in_list_max_values: optimizer.hash_join_inlist_pushdown_max_distinct_values,
        }
    }

    /// Family SF: binds a freshly constructed accumulator to the output
    /// partitions that its execution instance runs; `None` keeps the
    /// whole-plan barrier. Partitioned routing still covers every bucket of
    /// the stage: unassigned buckets stay unknown and keep their rows.
    pub(crate) fn with_execution_partitions(
        mut self,
        partitions: Option<Range<usize>>,
    ) -> Self {
        let generation = self.dynamic_filter.snapshot_generation();
        self.inner.get_mut().task_reports = partitions.map(|partitions| TaskReports {
            reported: vec![false; partitions.len()],
            partitions,
            generation,
        });
        self
    }

    /// Family SF: records the first report or cancellation of `partition`
    /// under a task scope. `Ok(false)` for a repeat; an error for a partition
    /// outside the scope.
    fn record_task_report(
        &self,
        state: &mut AccumulatorState,
        partition: usize,
    ) -> Result<bool> {
        let reports = state.task_reports.as_mut().expect("scoped accumulator");
        assert_or_internal_err!(
            reports.partitions.contains(&partition),
            "dynamic-filter report of partition {partition} outside the executed partitions {:?}",
            reports.partitions
        );
        let reported = &mut reports.reported[partition - reports.partitions.start];
        Ok(!std::mem::replace(reported, true))
    }

    /// Family SF: publishes a conservative snapshot without a cross-stream
    /// barrier. A task can run its partitions one after another (a retry on
    /// a worker with fewer slots), so waiting for every build before probing
    /// could keep later partitions from ever starting. The state lock
    /// serializes publication, so an older permissive snapshot never
    /// overwrites a newer one. Pending buckets are finalized as
    /// `CanceledUnknown` and keep every row (including NULL keys).
    ///
    /// `changed` is false when the report adds no build information to the
    /// last snapshot: a cancellation, or a repeated CollectLeft report of the
    /// one shared build. Such a report only records completion; it does not
    /// rebuild the predicate or advance its generation.
    fn publish_task_snapshot(
        &self,
        state: &mut AccumulatorState,
        changed: bool,
    ) -> Result<()> {
        if let CompletionState::Ready(Err(error)) = &state.completion {
            return Err(DataFusionError::Shared(Arc::clone(error)));
        }
        let reports = state.task_reports.as_ref().expect("scoped accumulator");
        let complete = reports.reported.iter().all(|reported| *reported);
        let result = if changed {
            let known_or_unknown = |status: &PartitionStatus| match status {
                PartitionStatus::Pending => PartitionStatus::CanceledUnknown,
                status => status.clone(),
            };
            let snapshot = match &state.data {
                AccumulatedBuildData::Partitioned { partitions, .. } => {
                    FinalizeInput::Partitioned(
                        partitions.iter().map(known_or_unknown).collect(),
                    )
                }
                AccumulatedBuildData::CollectLeft { data, .. } => {
                    FinalizeInput::CollectLeft(known_or_unknown(data))
                }
            };
            let result = self.build_filter(snapshot);
            let generation = self.dynamic_filter.snapshot_generation();
            if let Some(reports) = state.task_reports.as_mut() {
                reports.generation = generation;
            }
            result
        } else {
            Ok(())
        };
        if complete || result.is_err() {
            self.dynamic_filter.mark_complete();
            state.completion = CompletionState::Ready(result.map_err(Arc::new));
            self.completion_notify.notify_waiters();
            if let CompletionState::Ready(Err(error)) = &state.completion {
                return Err(DataFusionError::Shared(Arc::clone(error)));
            }
        }
        Ok(())
    }

    /// Report build-side data from a partition
    ///
    /// This unified method handles both CollectLeft and Partitioned modes. When all partitions
    /// have reported (barrier wait), the leader builds the appropriate filter expression:
    /// - CollectLeft: Simple conjunction of bounds and membership check
    /// - Partitioned: CASE expression routing to per-partition filters
    ///
    /// # Arguments
    /// * `data` - Build data including hash map, pushdown strategy, and bounds
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err if filter update failed or mode mismatch
    pub(crate) async fn report_build_data(&self, data: PartitionBuildData) -> Result<()> {
        let finalize_input = {
            let mut guard = self.inner.lock();
            if guard.task_reports.is_some() {
                let partition = match &data {
                    PartitionBuildData::Partitioned { partition_id, .. }
                    | PartitionBuildData::CollectLeft { partition_id, .. } => {
                        *partition_id
                    }
                };
                if !self.record_task_report(&mut guard, partition)? {
                    return Ok(());
                }
                // Every CollectLeft stream reports the same complete shared
                // build; only the first report changes the predicate.
                let changed = !matches!(
                    &guard.data,
                    AccumulatedBuildData::CollectLeft {
                        data: PartitionStatus::Reported(_),
                        ..
                    }
                );
                self.store_build_data(&mut guard, data)?;
                return self.publish_task_snapshot(&mut guard, changed);
            }
            self.store_build_data(&mut guard, data)?;
            self.take_finalize_input_if_ready(&mut guard)
        };

        if let Some(finalize_input) = finalize_input {
            self.finish(finalize_input);
        }

        self.wait_for_completion().await
    }

    pub(crate) fn report_canceled_partition(&self, partition_id: usize) {
        let finalize_input = {
            let mut guard = self.inner.lock();
            if guard.task_reports.is_some() {
                // A partition outside the scope never reports; only a first
                // report or cancellation of an assigned one counts.
                if !matches!(self.record_task_report(&mut guard, partition_id), Ok(true))
                {
                    return;
                }
                self.store_canceled_partition(&mut guard, partition_id);
                // An unknown bucket keeps TRUE, as Pending did in the last
                // snapshot, so a cancellation only records completion. A
                // failed snapshot stays in `completion` for later reports.
                let _ = self.publish_task_snapshot(&mut guard, false);
                return;
            }
            self.store_canceled_partition(&mut guard, partition_id);
            self.take_finalize_input_if_ready(&mut guard)
        };

        if let Some(finalize_input) = finalize_input {
            self.finish(finalize_input);
        }
    }

    fn store_build_data(
        &self,
        guard: &mut AccumulatorState,
        data: PartitionBuildData,
    ) -> Result<()> {
        match (data, &mut guard.data) {
            (
                PartitionBuildData::Partitioned {
                    partition_id,
                    pushdown,
                    bounds,
                    keys_have_null,
                },
                AccumulatedBuildData::Partitioned {
                    partitions,
                    completed_partitions,
                },
            ) => {
                if matches!(partitions[partition_id], PartitionStatus::Pending) {
                    *completed_partitions += 1;
                }
                partitions[partition_id] = PartitionStatus::Reported(PartitionData {
                    bounds,
                    pushdown,
                    keys_have_null,
                });
            }
            (
                PartitionBuildData::CollectLeft {
                    partition_id: _,
                    pushdown,
                    bounds,
                    keys_have_null,
                },
                AccumulatedBuildData::CollectLeft {
                    data,
                    reported_count,
                    ..
                },
            ) => {
                if matches!(data, PartitionStatus::Pending) {
                    *data = PartitionStatus::Reported(PartitionData {
                        bounds,
                        pushdown,
                        keys_have_null,
                    });
                }
                *reported_count += 1;
            }
            _ => {
                return datafusion_common::internal_err!(
                    "Build data mode mismatch in report_build_data"
                );
            }
        }
        Ok(())
    }

    fn store_canceled_partition(
        &self,
        guard: &mut AccumulatorState,
        partition_id: usize,
    ) {
        if let AccumulatedBuildData::Partitioned {
            partitions,
            completed_partitions,
        } = &mut guard.data
            && matches!(partitions[partition_id], PartitionStatus::Pending)
        {
            partitions[partition_id] = PartitionStatus::CanceledUnknown;
            *completed_partitions += 1;
        }
    }

    fn take_finalize_input_if_ready(
        &self,
        guard: &mut AccumulatorState,
    ) -> Option<FinalizeInput> {
        if !matches!(guard.completion, CompletionState::Pending) {
            return None;
        }

        let finalize_input = match &guard.data {
            AccumulatedBuildData::Partitioned {
                partitions,
                completed_partitions,
            } if *completed_partitions == partitions.len() => {
                Some(FinalizeInput::Partitioned(partitions.clone()))
            }
            AccumulatedBuildData::CollectLeft {
                data,
                reported_count,
                expected_reports,
            } if *reported_count == *expected_reports => {
                Some(FinalizeInput::CollectLeft(data.clone()))
            }
            _ => None,
        }?;

        guard.completion = CompletionState::Finalizing;
        Some(finalize_input)
    }

    fn finish(&self, finalize_input: FinalizeInput) {
        let result = self.build_filter(finalize_input).map_err(Arc::new);
        self.dynamic_filter.mark_complete();

        let mut guard = self.inner.lock();
        guard.completion = CompletionState::Ready(result);
        drop(guard);
        self.completion_notify.notify_waiters();
    }

    async fn wait_for_completion(&self) -> Result<()> {
        loop {
            let notified = {
                let guard = self.inner.lock();
                match &guard.completion {
                    CompletionState::Ready(Ok(())) => return Ok(()),
                    CompletionState::Ready(Err(err)) => {
                        return Err(DataFusionError::Shared(Arc::clone(err)));
                    }
                    CompletionState::Pending | CompletionState::Finalizing => {
                        self.completion_notify.notified()
                    }
                }
            };
            notified.await;
        }
    }

    fn build_filter(&self, finalize_input: FinalizeInput) -> Result<()> {
        match finalize_input {
            FinalizeInput::CollectLeft(partition) => {
                self.build_collect_left_filter(partition)
            }
            FinalizeInput::Partitioned(partitions) => {
                self.build_partitioned_filter(partitions)
            }
        }
    }

    /// Builds the single global filter used by a collect-left join.
    fn build_collect_left_filter(&self, partition: PartitionStatus) -> Result<()> {
        match partition {
            PartitionStatus::Reported(PartitionData {
                bounds,
                pushdown,
                keys_have_null,
            }) => {
                let membership_expr = create_membership_predicate(
                    &self.on_right,
                    pushdown,
                    &HASH_JOIN_SEED,
                    self.probe_schema.as_ref(),
                )?;
                let bounds_expr = create_bounds_predicate(&self.on_right, &bounds);

                if let Some(filter_expr) =
                    combine_membership_and_bounds(membership_expr, bounds_expr)
                {
                    self.dynamic_filter.update(
                        self.preserve_probe_nulls(filter_expr, keys_have_null)?,
                    )?;
                }
                Ok(())
            }
            PartitionStatus::Pending => datafusion_common::internal_err!(
                "attempted to finalize collect-left dynamic filter without reported build data"
            ),
            // Family SF: only a task scope gets here, when no stream of the
            // task has reported the shared build yet (or none ever will).
            // Keep the initial TRUE predicate.
            PartitionStatus::CanceledUnknown => Ok(()),
        }
    }

    /// Builds one routed probe-side filter from finalized partitioned build data.
    /// Empty partitions reject their routed rows, while canceled partitions stay
    /// permissive because their build contents are unknown.
    fn build_partitioned_filter(&self, partitions: Vec<PartitionStatus>) -> Result<()> {
        let mut partition_filters = Vec::with_capacity(partitions.len());
        let mut real_partition_ids = Vec::new();
        let mut empty_partition_ids = Vec::new();
        let mut has_canceled_unknown = false;
        let mut keys_have_null = false;
        // Every non-empty partition's membership strategy and bounds, merged into
        // one membership filter below; `None` once a partition cannot be merged.
        let mut members = Some(Vec::with_capacity(partitions.len()));
        let mut merged_bounds: Option<PartitionBounds> = None;

        for (partition_id, partition) in partitions.into_iter().enumerate() {
            match partition {
                PartitionStatus::Reported(PartitionData {
                    pushdown: PushdownStrategy::Empty,
                    keys_have_null: partition_keys_have_null,
                    ..
                }) => {
                    // A NULL-only key set has no ordinary membership values,
                    // but a null-equal join can still match it.
                    keys_have_null |= partition_keys_have_null;
                    empty_partition_ids.push(partition_id);
                    partition_filters.push(lit(false));
                }
                PartitionStatus::Reported(PartitionData {
                    bounds,
                    pushdown,
                    keys_have_null: partition_keys_have_null,
                }) => {
                    real_partition_ids.push(partition_id);
                    keys_have_null |= partition_keys_have_null;
                    let mergeable = match &pushdown {
                        PushdownStrategy::Map(map, _, _) => {
                            matches!(map.as_ref(), Map::HashMap(_))
                        }
                        PushdownStrategy::InList(_) => true,
                        PushdownStrategy::Empty | PushdownStrategy::BoundsOnly => false,
                    };
                    members = members.filter(|_| mergeable).map(|mut members| {
                        members.push((partition_id, pushdown.clone(), bounds.clone()));
                        members
                    });
                    merged_bounds = Some(match merged_bounds {
                        None => bounds.clone(),
                        Some(merged) => merged.union(&bounds),
                    });

                    let membership_expr = create_membership_predicate(
                        &self.on_right,
                        pushdown,
                        &HASH_JOIN_SEED,
                        self.probe_schema.as_ref(),
                    )?;
                    let bounds_expr = create_bounds_predicate(&self.on_right, &bounds);
                    let then_expr =
                        combine_membership_and_bounds(membership_expr, bounds_expr)
                            .unwrap_or_else(|| lit(true));
                    partition_filters.push(then_expr);
                }
                PartitionStatus::CanceledUnknown => {
                    has_canceled_unknown = true;
                    partition_filters.push(lit(true));
                    // Rows routed here must pass, which a merged filter cannot express.
                    members = None;
                    // A canceled partition's build content is unknown, so it
                    // may hold a NULL key.
                    keys_have_null = true;
                }
                PartitionStatus::Pending => {
                    return datafusion_common::internal_err!(
                        "attempted to finalize dynamic filter with pending partition"
                    );
                }
            }
        }

        let all_partitions_canceled = has_canceled_unknown
            && real_partition_ids.is_empty()
            && empty_partition_ids.is_empty();
        let all_partitions_empty = !has_canceled_unknown && real_partition_ids.is_empty();
        let one_non_empty_partition =
            !has_canceled_unknown && real_partition_ids.len() == 1;

        let filter_expr = if all_partitions_canceled {
            // No build data is known, so filtering any probe row could discard a match.
            lit(true)
        } else if all_partitions_empty {
            // No build row exists, so no probe row can match.
            lit(false)
        } else if one_non_empty_partition {
            // Only one build partition contains rows, so its filter covers every
            // possible probe match without routing.
            Arc::clone(&partition_filters[real_partition_ids[0]])
        } else if let Some(members) = members {
            self.merged_membership_filter(
                &members,
                merged_bounds,
                partition_filters.len(),
            )?
        } else {
            // Builds the shared sparse `CASE` for partition filter routing.
            // Without cancellation, omitted branches are known empty and safely fall
            // through to `ELSE false`. With cancellation, omitted canceled partitions
            // have unknown contents and must fall through to `ELSE true`, so known-empty
            // partitions are emitted explicitly as false branches.
            let mut branches = if has_canceled_unknown {
                empty_partition_ids
                    .iter()
                    .map(|&partition_id| (lit(partition_id as u64), lit(false)))
                    .collect::<Vec<_>>()
            } else {
                vec![]
            };
            branches.extend(real_partition_ids.iter().map(|&partition_id| {
                (
                    lit(partition_id as u64),
                    Arc::clone(&partition_filters[partition_id]),
                )
            }));

            let routing_expr = if let Some(range_partitioning) =
                &self.probe_range_partitioning
            {
                // Routes probe rows with the partition id selected by [`RangeExpr`].
                // CASE range_partition(keys)
                //   WHEN empty_partition_id THEN false  -- only when cancellation exists
                //   WHEN real_partition_id THEN F(real_partition_id)
                //   ...
                //   ELSE has_canceled_unknown
                // END
                assert_or_internal_err!(
                    partition_filters.len() == range_partitioning.partition_count(),
                    "Dynamic filter partition count {} does not match Range partition count {}",
                    partition_filters.len(),
                    range_partitioning.partition_count()
                );
                Arc::new(RangeExpr::try_new_with_schema(
                    self.on_right.clone(),
                    range_partitioning,
                    &self.probe_schema,
                )?) as Arc<dyn PhysicalExpr>
            } else {
                // Routes probe rows with the same `hash(keys) % partition_count` expression used
                // by Hash repartitioning.
                // CASE hash(keys) % partition_count
                //   WHEN empty_partition_id THEN false  -- only when cancellation exists
                //   WHEN real_partition_id THEN F(real_partition_id)
                //   ...
                //   ELSE has_canceled_unknown
                // END
                let routing_hash_expr = Arc::new(HashExpr::new(
                    self.on_right.clone(),
                    self.repartition_random_state.clone(),
                    "hash_repartition".to_string(),
                )) as Arc<dyn PhysicalExpr>;
                Arc::new(BinaryExpr::new(
                    routing_hash_expr,
                    Operator::Modulo,
                    lit(partition_filters.len() as u64),
                )) as Arc<dyn PhysicalExpr>
            };

            Arc::new(CaseExpr::try_new(
                Some(routing_expr),
                branches,
                Some(lit(has_canceled_unknown)),
            )?)
        };

        self.dynamic_filter
            .update(self.preserve_probe_nulls(filter_expr, keys_have_null)?)
    }

    /// `members` holds each non-empty partition's id, strategy (an IN list or a
    /// hash table) and bounds, out of `num_partitions` partitions.
    fn merged_membership_filter(
        &self,
        members: &[(usize, PushdownStrategy, PartitionBounds)],
        merged_bounds: Option<PartitionBounds>,
        num_partitions: usize,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let lists = members
            .iter()
            .filter_map(|(_, pushdown, _)| match pushdown {
                PushdownStrategy::InList(values) => Some(values),
                _ => None,
            })
            .collect::<Vec<_>>();

        // Every partition is an IN list, and together they still fit the limits a
        // single build uses to pick an IN list: one combined list keeps the
        // statistics and bloom filter pruning an IN list allows. Past those
        // limits, a hash lookup is cheaper per row and carries a pruning bitmap.
        let combined_in_list = lists.len() == members.len() && {
            let values = lists.iter().map(|list| list.len()).sum::<usize>();
            let bytes = lists
                .iter()
                .map(|list| list.get_array_memory_size())
                .sum::<usize>();
            values <= self.in_list_max_values && bytes <= self.in_list_max_size
        };
        let membership = if combined_in_list {
            let values = concat(&lists.iter().map(|l| l.as_ref()).collect::<Vec<_>>())?;
            create_membership_predicate(
                &self.on_right,
                PushdownStrategy::InList(values),
                &HASH_JOIN_SEED,
                self.probe_schema.as_ref(),
            )?
            .expect("an IN list always has a membership predicate")
        } else {
            // Each probe row is looked up in the table of the partition its key
            // routes to, so the partitions' tables are shared as they are: no
            // merged table to build while every probe stream waits on the filter.
            let mut maps = vec![None; num_partitions];
            // Longbow family E: each partition's build keys, and the first
            // reported build memory consumer, for structures derived from them.
            // The lookup owns its IN-list partitions' key columns uncharged, as
            // the IN-list strategy did; the IN-list limits bound them.
            let mut keys = vec![vec![]; num_partitions];
            let mut retained = Vec::new();
            let mut memory = None;
            for (partition_id, pushdown, _) in members {
                maps[*partition_id] = Some(match pushdown {
                    PushdownStrategy::Map(map, _, build_side) => {
                        if let Some(build_side) = build_side {
                            keys[*partition_id] = vec![Weak::clone(&build_side.keys)];
                            memory.get_or_insert_with(|| Arc::clone(&build_side.memory));
                        }
                        Arc::clone(map)
                    }
                    PushdownStrategy::InList(values) => {
                        let columns: Arc<[ArrayRef]> =
                            self.in_list_columns(values).into();
                        let map = self.in_list_hash_map(&columns)?;
                        keys[*partition_id] = vec![Arc::downgrade(&columns)];
                        retained.push(columns);
                        map
                    }
                    PushdownStrategy::Empty | PushdownStrategy::BoundsOnly => {
                        return internal_err!("empty partitions have no members");
                    }
                });
            }
            Arc::new(
                HashTableLookupExpr::new(
                    self.on_right.clone(),
                    HASH_JOIN_SEED,
                    Some(self.partition_router(num_partitions)?),
                    maps,
                    String::from("hash_lookup"),
                    self.merged_pruning_bitmap(members, merged_bounds.as_ref()),
                )?
                .with_build_keys(keys, retained, memory)?,
            )
        };

        let bounds = merged_bounds
            .and_then(|bounds| create_bounds_predicate(&self.on_right, &bounds));
        Ok(combine_membership_and_bounds(Some(membership), bounds)
            .expect("membership predicate is always present"))
    }

    /// How probe rows find their build partition: as the probe side was
    /// partitioned.
    fn partition_router(&self, num_partitions: usize) -> Result<PartitionRouter> {
        Ok(match &self.probe_range_partitioning {
            Some(range_partitioning) => {
                assert_or_internal_err!(
                    num_partitions == range_partitioning.partition_count(),
                    "Dynamic filter partition count {} does not match Range partition count {}",
                    num_partitions,
                    range_partitioning.partition_count()
                );
                PartitionRouter::Range(Arc::new(RangeExpr::try_new_with_schema(
                    self.on_right.clone(),
                    range_partitioning,
                    &self.probe_schema,
                )?))
            }
            None => PartitionRouter::Hash(self.repartition_random_state.clone()),
        })
    }

    /// A hash table of an IN list partition's keys, hashed as the probe side
    /// hashes its join keys: one column per join key.
    fn in_list_hash_map(&self, columns: &[ArrayRef]) -> Result<Arc<Map>> {
        let rows = columns.first().map_or(0, |column| column.len());
        let mut hashes = vec![0; rows];
        create_hashes(columns, HASH_JOIN_SEED.random_state(), &mut hashes)?;
        let mut map = JoinHashMapU32::with_capacity(hashes.len());
        map.update_from_iter(Box::new(hashes.iter().enumerate()), 0)?;
        Ok(Arc::new(Map::HashMap(Box::new(map))))
    }

    /// An IN list partition's keys as one column per join key.
    fn in_list_columns(&self, values: &ArrayRef) -> Vec<ArrayRef> {
        match values.as_struct_opt() {
            Some(columns) if self.on_right.len() > 1 => columns.columns().to_vec(),
            _ => vec![Arc::clone(values)],
        }
    }

    /// Container pruning reads one bitmap, so merge the partitions' key ranges:
    /// a hash table's bitmap, an IN list's keys, or else a partition's whole
    /// bounds. `None` when no partition narrows its bounds, as the bounds
    /// predicate already prunes that much.
    fn merged_pruning_bitmap(
        &self,
        members: &[(usize, PushdownStrategy, PartitionBounds)],
        merged_bounds: Option<&PartitionBounds>,
    ) -> Option<Arc<KeyRangeBitmap>> {
        if self.on_right.len() != 1 {
            return None;
        }
        let merged = merged_bounds?.get_column_bounds(0)?;
        let parts = members
            .iter()
            .map(|(_, pushdown, bounds)| match pushdown {
                PushdownStrategy::Map(_, Some(bitmap), _) => {
                    Some(KeyRangePart::Bitmap(bitmap))
                }
                PushdownStrategy::InList(keys) => Some(KeyRangePart::Keys(keys.as_ref())),
                _ => bounds
                    .get_column_bounds(0)
                    .map(|bounds| KeyRangePart::Range {
                        min: &bounds.min,
                        max: &bounds.max,
                    }),
            })
            .collect::<Option<Vec<_>>>()?;
        if parts
            .iter()
            .all(|part| matches!(part, KeyRangePart::Range { .. }))
        {
            return None;
        }
        KeyRangeBitmap::union(&merged.min, &merged.max, &parts).map(Arc::new)
    }

    /// Keeps probe rows with a NULL key when the join semantics need them.
    ///
    /// The build-side predicate drops probe rows whose key is NULL. A null-aware join
    /// needs a NULL value key to reach the join for `NOT IN` three-valued logic, and a
    /// null-equal join needs NULL keys to match build-side NULLs.
    ///
    /// For a null-aware join, only the `NOT IN` value keys `on_right[..V]` need the
    /// escape, while `on_right[V..]` are correlation scope keys: a NULL in any element
    /// of a multi-column value key can make the tuple comparison UNKNOWN. For a
    /// null-equal join, every nullable key needs the escape.
    fn preserve_probe_nulls(
        &self,
        filter_expr: Arc<dyn PhysicalExpr>,
        build_keys_have_null: bool,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        // A null-aware anti join needs every probe NULL no matter what the build holds: one
        // probe NULL makes `NOT IN` unknown for every build row. A null-equal join needs probe
        // NULLs only to match an actual build-side NULL, so a NULL-free build keeps the filter
        // at full selectivity.
        let needs_probe_nulls = self.not_in.is_some()
            || (self.null_equality == NullEquality::NullEqualsNull
                && build_keys_have_null);
        if !needs_probe_nulls {
            return Ok(filter_expr);
        }
        let keys = match self.not_in {
            Some(mode) => {
                let value_keys = mode.value_keys();
                assert_or_internal_err!(
                    (1..=self.on_right.len()).contains(&value_keys),
                    "null-aware join must have between 1 and {} value keys, got {value_keys}",
                    self.on_right.len()
                );
                &self.on_right[..value_keys]
            }
            None => self.on_right.as_slice(),
        };

        // Only a key that can actually be NULL needs the disjunct; a NOT NULL key
        // never widens the filter.
        let mut any_key_is_null: Option<Arc<dyn PhysicalExpr>> = None;
        for key in keys {
            // `nullable` fails only when a key is out of sync with the probe schema. That is
            // a construction bug, so surface it instead of widening around it.
            if !key.nullable(&self.probe_schema)? {
                continue;
            }
            let is_null =
                Arc::new(IsNullExpr::new(Arc::clone(key))) as Arc<dyn PhysicalExpr>;
            any_key_is_null = Some(match any_key_is_null {
                Some(acc) => Arc::new(BinaryExpr::new(acc, Operator::Or, is_null)) as _,
                None => is_null,
            });
        }
        // Cheap null check first short-circuits before the costlier dynamic filter.
        Ok(match any_key_is_null {
            Some(any_key_is_null) => {
                Arc::new(BinaryExpr::new(any_key_is_null, Operator::Or, filter_expr))
            }
            None => filter_expr,
        })
    }
}

// Family SF: once the last accumulator owner is gone, every stream and report
// future is gone too. The assigned partitions that have not reported never
// started or were abandoned by teardown: close the retained filter observers
// without waiting for them, unless the filter was reset for another
// execution since this accumulator last published it.
impl Drop for SharedBuildAccumulator {
    fn drop(&mut self) {
        let state = self.inner.get_mut();
        if let Some(reports) = &state.task_reports
            && matches!(state.completion, CompletionState::Pending)
            && reports.generation == self.dynamic_filter.snapshot_generation()
        {
            self.dynamic_filter.mark_complete();
        }
    }
}

impl fmt::Debug for SharedBuildAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SharedBuildAccumulator")
    }
}

#[cfg(test)]
pub(super) fn make_partitioned_accumulator_for_test(
    num_partitions: usize,
) -> SharedBuildAccumulator {
    let probe_schema = Arc::new(Schema::new(vec![Field::new(
        "probe_key",
        DataType::Int32,
        false,
    )]));
    let dynamic_filter = Arc::new(DynamicFilterPhysicalExpr::new(vec![], lit(true)));
    SharedBuildAccumulator {
        inner: Mutex::new(AccumulatorState {
            data: AccumulatedBuildData::Partitioned {
                partitions: vec![PartitionStatus::Pending; num_partitions],
                completed_partitions: 0,
            },
            completion: CompletionState::Pending,
            task_reports: None,
        }),
        completion_notify: Notify::new(),
        dynamic_filter,
        on_right: vec![],
        repartition_random_state: SeededRandomState::with_seed(1),
        probe_schema,
        probe_range_partitioning: None,
        null_equality: NullEquality::NullEqualsNothing,
        not_in: None,
        in_list_max_size: 128 * 1024,
        in_list_max_values: 150,
    }
}

#[cfg(test)]
pub(super) fn completed_partitions_for_test(acc: &SharedBuildAccumulator) -> usize {
    let guard = acc.inner.lock();
    let AccumulatedBuildData::Partitioned {
        completed_partitions,
        ..
    } = &guard.data
    else {
        panic!("expected partitioned accumulator");
    };
    *completed_partitions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joins::hash_join::not_in::{NotInOutput, NotInScope};

    use crate::joins::join_hash_map::JoinHashMapU32;
    use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int32Array};
    use arrow::compute::SortOptions;
    use arrow::record_batch::RecordBatch;
    use datafusion_common::SplitPoint;
    use datafusion_physical_expr::{
        PhysicalSortExpr,
        expressions::{Column, Literal},
    };

    fn test_on_right() -> Vec<PhysicalExprRef> {
        vec![Arc::new(Column::new("probe_key", 0))]
    }

    fn test_probe_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "probe_key",
            DataType::Int32,
            false,
        )]))
    }

    fn test_dynamic_filter(
        on_right: &[PhysicalExprRef],
    ) -> Arc<DynamicFilterPhysicalExpr> {
        Arc::new(DynamicFilterPhysicalExpr::new(on_right.to_vec(), lit(true)))
    }

    fn make_accumulator_for_test(
        data: AccumulatedBuildData,
        on_right: Vec<PhysicalExprRef>,
    ) -> SharedBuildAccumulator {
        let dynamic_filter = test_dynamic_filter(&on_right);
        SharedBuildAccumulator {
            inner: Mutex::new(AccumulatorState {
                data,
                completion: CompletionState::Pending,
                task_reports: None,
            }),
            completion_notify: Notify::new(),
            dynamic_filter,
            on_right,
            repartition_random_state: SeededRandomState::with_seed(1),
            probe_schema: test_probe_schema(),
            probe_range_partitioning: None,
            null_equality: NullEquality::NullEqualsNothing,
            not_in: None,
            in_list_max_size: 128 * 1024,
            in_list_max_values: 150,
        }
    }

    fn make_collect_left_accumulator_for_test() -> SharedBuildAccumulator {
        make_accumulator_for_test(
            AccumulatedBuildData::CollectLeft {
                data: PartitionStatus::Pending,
                reported_count: 0,
                expected_reports: 1,
            },
            test_on_right(),
        )
    }

    fn make_partitioned_expr_accumulator_for_test(
        num_partitions: usize,
    ) -> SharedBuildAccumulator {
        make_accumulator_for_test(
            AccumulatedBuildData::Partitioned {
                partitions: vec![PartitionStatus::Pending; num_partitions],
                completed_partitions: 0,
            },
            test_on_right(),
        )
    }

    fn in_list(values: &[i32]) -> PushdownStrategy {
        PushdownStrategy::InList(Arc::new(Int32Array::from(values.to_vec())) as ArrayRef)
    }

    fn bounds(min: i32, max: i32) -> PartitionBounds {
        PartitionBounds::new(vec![ColumnBounds::new(
            ScalarValue::Int32(Some(min)),
            ScalarValue::Int32(Some(max)),
        )])
    }

    fn no_bounds() -> PartitionBounds {
        PartitionBounds::new(vec![])
    }

    fn reported(pushdown: PushdownStrategy, bounds: PartitionBounds) -> PartitionStatus {
        PartitionStatus::Reported(PartitionData {
            pushdown,
            bounds,
            keys_have_null: false,
        })
    }

    fn current_expr(acc: &SharedBuildAccumulator) -> PhysicalExprRef {
        acc.dynamic_filter
            .current()
            .expect("dynamic filter current expression should be available")
    }

    fn in_list_expr(expr: &PhysicalExprRef) -> &InListExpr {
        expr.downcast_ref::<InListExpr>()
            .expect("expected InListExpr dynamic filter")
    }

    fn assert_in_list_column_values(
        expr: &PhysicalExprRef,
        expected_column_name: &str,
        expected_column_index: usize,
        expected_values: &[i32],
    ) {
        let in_list = in_list_expr(expr);
        let column = in_list
            .expr()
            .downcast_ref::<Column>()
            .expect("expected InListExpr child column");
        assert_eq!(column.name(), expected_column_name);
        assert_eq!(column.index(), expected_column_index);

        let actual_values = in_list
            .list()
            .iter()
            .map(|expr| {
                let literal = expr
                    .downcast_ref::<Literal>()
                    .expect("expected InListExpr literal value");
                match literal.value() {
                    ScalarValue::Int32(Some(value)) => *value,
                    value => panic!("expected Int32 in-list value, got {value:?}"),
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(actual_values, expected_values);
    }

    fn binary_expr(expr: &PhysicalExprRef) -> &BinaryExpr {
        expr.downcast_ref::<BinaryExpr>()
            .expect("expected BinaryExpr dynamic filter")
    }

    fn case_expr(expr: &PhysicalExprRef) -> &CaseExpr {
        expr.downcast_ref::<CaseExpr>()
            .expect("expected CaseExpr dynamic filter")
    }

    fn assert_literal_bool(expr: &PhysicalExprRef, expected: bool) {
        let literal = expr
            .downcast_ref::<Literal>()
            .expect("expected literal bool dynamic filter");
        assert_eq!(literal.value(), &ScalarValue::Boolean(Some(expected)));
    }

    fn assert_top_binary_op(expr: &PhysicalExprRef, expected: Operator) {
        assert_eq!(binary_expr(expr).op(), &expected);
    }

    fn partitioned_state(acc: &SharedBuildAccumulator) -> (Vec<PartitionStatus>, usize) {
        let guard = acc.inner.lock();
        let AccumulatedBuildData::Partitioned {
            partitions,
            completed_partitions,
        } = &guard.data
        else {
            panic!("expected partitioned accumulator");
        };
        (partitions.clone(), *completed_partitions)
    }

    #[test]
    fn collect_left_updates_with_membership_only() {
        let acc = make_collect_left_accumulator_for_test();

        acc.build_filter(FinalizeInput::CollectLeft(reported(
            in_list(&[1, 2, 3]),
            no_bounds(),
        )))
        .unwrap();

        let expr = current_expr(&acc);
        assert_in_list_column_values(&expr, "probe_key", 0, &[1, 2, 3]);
    }

    #[test]
    fn collect_left_updates_with_bounds_only() {
        let acc = make_collect_left_accumulator_for_test();

        acc.build_filter(FinalizeInput::CollectLeft(reported(
            PushdownStrategy::Empty,
            bounds(10, 20),
        )))
        .unwrap();

        let expr = current_expr(&acc);
        assert_top_binary_op(&expr, Operator::And);
    }

    #[test]
    fn collect_left_empty_build_data_does_not_update_filter() {
        let acc = make_collect_left_accumulator_for_test();
        let initial_generation = acc.dynamic_filter.snapshot_generation();

        acc.build_filter(FinalizeInput::CollectLeft(reported(
            PushdownStrategy::Empty,
            no_bounds(),
        )))
        .unwrap();

        assert_eq!(
            acc.dynamic_filter.snapshot_generation(),
            initial_generation,
            "empty CollectLeft input must not update with a no-op filter"
        );
        let expr = current_expr(&acc);
        assert_literal_bool(&expr, true);
    }

    fn hash_map() -> PushdownStrategy {
        PushdownStrategy::Map(
            Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(1)))),
            None,
            None,
        )
    }

    #[test]
    fn partitioned_hash_map_partitions_route_without_case() {
        let acc = make_partitioned_expr_accumulator_for_test(3);

        acc.build_filter(FinalizeInput::Partitioned(vec![
            reported(hash_map(), bounds(5, 10)),
            reported(PushdownStrategy::Empty, no_bounds()),
            reported(hash_map(), bounds(1, 7)),
        ]))
        .unwrap();

        let expr = current_expr(&acc);
        assert!(expr.downcast_ref::<CaseExpr>().is_none());
        assert_eq!(
            expr.to_string(),
            "probe_key@0 >= 1 AND probe_key@0 <= 10 AND hash_lookup"
        );
    }

    /// A hash table holding `keys`, hashed as the build side hashes them.
    #[cfg(not(feature = "force_hash_collisions"))]
    fn hash_map_of(keys: &[i32]) -> PushdownStrategy {
        use crate::joins::join_hash_map::JoinHashMapType;
        let keys: ArrayRef = Arc::new(Int32Array::from(keys.to_vec()));
        let mut hashes = vec![0; keys.len()];
        create_hashes([&keys], HASH_JOIN_SEED.random_state(), &mut hashes).unwrap();
        let mut map = JoinHashMapU32::with_capacity(keys.len());
        map.update_from_iter(Box::new(hashes.iter().enumerate()), 0)
            .unwrap();
        PushdownStrategy::Map(Arc::new(Map::HashMap(Box::new(map))), None, None)
    }

    /// Splits `keys` across `num_partitions` build partitions as hash
    /// repartitioning would, reporting each non-empty partition with the
    /// strategy `strategy` picks for it.
    #[cfg(not(feature = "force_hash_collisions"))]
    fn routed_partitions(
        acc: &SharedBuildAccumulator,
        num_partitions: usize,
        keys: &[i32],
        strategy: impl Fn(usize, &[i32]) -> PushdownStrategy,
    ) -> Vec<PartitionStatus> {
        let array: ArrayRef = Arc::new(Int32Array::from(keys.to_vec()));
        let mut hashes = vec![0; keys.len()];
        create_hashes(
            [&array],
            acc.repartition_random_state.random_state(),
            &mut hashes,
        )
        .unwrap();
        (0..num_partitions)
            .map(|partition| {
                let held: Vec<i32> = keys
                    .iter()
                    .zip(&hashes)
                    .filter(|(_, hash)| *hash % num_partitions as u64 == partition as u64)
                    .map(|(key, _)| *key)
                    .collect();
                if held.is_empty() {
                    reported(PushdownStrategy::Empty, no_bounds())
                } else {
                    reported(strategy(partition, &held), no_bounds())
                }
            })
            .collect()
    }

    /// Partitions that chose an IN list and partitions that chose a hash table
    /// merge into one lookup without a `CASE`: a probe row passes exactly when
    /// the partition its key routes to holds the key.
    #[test]
    #[cfg(not(feature = "force_hash_collisions"))]
    fn partitioned_in_list_and_hash_map_partitions_merge() {
        let acc = make_partitioned_expr_accumulator_for_test(4);
        let build: Vec<i32> = (0..40).step_by(2).collect();

        acc.build_filter(FinalizeInput::Partitioned(routed_partitions(
            &acc,
            4,
            &build,
            |partition, keys| {
                if partition % 2 == 0 {
                    hash_map_of(keys)
                } else {
                    in_list(keys)
                }
            },
        )))
        .unwrap();

        assert!(
            current_expr(&acc).downcast_ref::<CaseExpr>().is_none(),
            "expected a merged lookup, got: {}",
            current_expr(&acc)
        );
        let probe: Vec<i32> = (0..40).collect();
        let expected: Vec<bool> = probe.iter().map(|key| key % 2 == 0).collect();
        assert_matches(&acc, &probe, &expected);
    }

    /// Longbow family E: the merged lookup exposes every non-empty partition's
    /// build keys (a hash table's keys weakly, an IN list's values owned) and
    /// the first reported build memory consumer; a collect-left lookup exposes
    /// its build's.
    #[test]
    #[cfg(not(feature = "force_hash_collisions"))]
    fn lookups_expose_partition_build_keys_and_memory() {
        use datafusion_execution::memory_pool::{
            GreedyMemoryPool, MemoryConsumer, MemoryPool,
        };
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let memory = Arc::new(MemoryConsumer::new("HashJoinInput[0]").register(&pool));
        let held = Mutex::new(Vec::<Arc<[ArrayRef]>>::new());
        let with_side = |keys: &[i32]| {
            let PushdownStrategy::Map(map, bitmap, None) = hash_map_of(keys) else {
                unreachable!("a test map has no build side")
            };
            let array: Arc<[ArrayRef]> =
                vec![Arc::new(Int32Array::from(keys.to_vec())) as ArrayRef].into();
            let side = MapBuildSide {
                keys: Arc::downgrade(&array),
                memory: Arc::clone(&memory),
            };
            held.lock().push(array);
            PushdownStrategy::Map(map, bitmap, Some(side))
        };
        let lookup_of = |expr: &PhysicalExprRef| {
            expr.downcast_ref::<HashTableLookupExpr>()
                .map(|lookup| {
                    (
                        lookup.build_keys().map(|keys| {
                            let mut values = keys
                                .iter()
                                .flat_map(|keys| {
                                    keys[0]
                                        .as_primitive::<arrow::datatypes::Int32Type>()
                                        .values()
                                        .to_vec()
                                })
                                .collect::<Vec<_>>();
                            values.sort_unstable();
                            values
                        }),
                        lookup
                            .build_memory()
                            .map(|memory| memory.consumer().name().to_owned()),
                    )
                })
                .unwrap_or_else(|| panic!("expected a hash lookup, got: {expr}"))
        };

        let acc = make_partitioned_expr_accumulator_for_test(4);
        let build: Vec<i32> = (0..40).step_by(2).collect();
        acc.build_filter(FinalizeInput::Partitioned(routed_partitions(
            &acc,
            4,
            &build,
            |partition, keys| {
                if partition % 2 == 0 {
                    with_side(keys)
                } else {
                    in_list(keys)
                }
            },
        )))
        .unwrap();
        let expr = current_expr(&acc);
        assert_eq!(
            lookup_of(&expr),
            (Some(build.clone()), Some("HashJoinInput[0]".to_owned()))
        );
        // The join released a hash table's keys; an IN list's stay with the lookup.
        held.lock().clear();
        assert_eq!(
            lookup_of(&expr),
            (None, Some("HashJoinInput[0]".to_owned()))
        );

        // Only IN list partitions: owned keys, no memory consumer.
        let mut acc = make_partitioned_expr_accumulator_for_test(3);
        acc.in_list_max_values = 2;
        let build: Vec<i32> = (1..=9).collect();
        acc.build_filter(FinalizeInput::Partitioned(routed_partitions(
            &acc,
            3,
            &build,
            |_, keys| in_list(keys),
        )))
        .unwrap();
        assert_eq!(lookup_of(&current_expr(&acc)), (Some(build), None));

        // A hash table without a build side exposes nothing.
        let acc = make_partitioned_expr_accumulator_for_test(2);
        acc.build_filter(FinalizeInput::Partitioned(routed_partitions(
            &acc,
            2,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            |_, keys| hash_map_of(keys),
        )))
        .unwrap();
        assert_eq!(lookup_of(&current_expr(&acc)), (None, None));

        // A collect-left build's single lookup.
        let lookup = create_membership_predicate(
            &test_on_right(),
            with_side(&[5, 1, 3]),
            &HASH_JOIN_SEED,
            &test_probe_schema(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            lookup_of(&lookup),
            (Some(vec![1, 3, 5]), Some("HashJoinInput[0]".to_owned()))
        );
        held.lock().clear();
        assert_eq!(lookup_of(&lookup).0, None);
        assert_eq!(pool.reserved(), 0);
    }

    /// IN list partitions whose combined list passes the IN list limits merge
    /// into a hash lookup rather than one oversized IN list, which must still
    /// accept every partition's keys.
    #[test]
    #[cfg(not(feature = "force_hash_collisions"))]
    fn partitioned_in_lists_past_limit_merge_into_hash_lookup() {
        let mut acc = make_partitioned_expr_accumulator_for_test(3);
        acc.in_list_max_values = 2;
        let build: Vec<i32> = (1..=9).collect();

        acc.build_filter(FinalizeInput::Partitioned(routed_partitions(
            &acc,
            3,
            &build,
            |_, keys| in_list(keys),
        )))
        .unwrap();

        let expr = current_expr(&acc);
        assert!(
            expr.downcast_ref::<CaseExpr>().is_none()
                && expr.downcast_ref::<InListExpr>().is_none(),
            "expected a hash lookup, got: {expr}"
        );
        let probe: Vec<i32> = (0..20).collect();
        let expected: Vec<bool> = probe.iter().map(|key| (1..=9).contains(key)).collect();
        assert_matches(&acc, &probe, &expected);
    }

    /// Evaluates the current filter over probe keys `keys`.
    #[cfg(not(feature = "force_hash_collisions"))]
    fn assert_matches(acc: &SharedBuildAccumulator, keys: &[i32], expected: &[bool]) {
        let probe = RecordBatch::try_new(
            test_probe_schema(),
            vec![Arc::new(Int32Array::from(keys.to_vec()))],
        )
        .unwrap();
        let matched = current_expr(acc)
            .evaluate(&probe)
            .unwrap()
            .into_array(keys.len())
            .unwrap();
        assert_eq!(matched.as_boolean(), &BooleanArray::from(expected.to_vec()));
    }

    #[test]
    fn partitioned_one_real_partition_with_rest_empty_skips_case() {
        let acc = make_partitioned_expr_accumulator_for_test(3);

        acc.build_filter(FinalizeInput::Partitioned(vec![
            reported(PushdownStrategy::Empty, no_bounds()),
            reported(in_list(&[2]), no_bounds()),
            reported(PushdownStrategy::Empty, no_bounds()),
        ]))
        .unwrap();

        let expr = current_expr(&acc);
        in_list_expr(&expr);
        assert!(expr.downcast_ref::<CaseExpr>().is_none());
    }

    #[test]
    fn partitioned_canceled_unknown_partitions_keep_unknown_routes_permissive() {
        let acc = make_partitioned_expr_accumulator_for_test(2);

        acc.build_filter(FinalizeInput::Partitioned(vec![
            PartitionStatus::CanceledUnknown,
            reported(PushdownStrategy::Empty, no_bounds()),
        ]))
        .unwrap();

        let expr = current_expr(&acc);
        let case = case_expr(&expr);
        assert_eq!(case.when_then_expr().len(), 1);
        assert_literal_bool(&case.when_then_expr()[0].1, false);
        assert_literal_bool(
            case.else_expr().expect("expected permissive fallback"),
            true,
        );
    }

    #[test]
    fn partitioned_range_dynamic_filter_routes_with_range_expr() -> Result<()> {
        let mut acc = make_partitioned_expr_accumulator_for_test(4);
        acc.probe_range_partitioning = Some(RangePartitioning::try_new(
            [PhysicalSortExpr::new(
                Arc::clone(&acc.on_right[0]),
                Default::default(),
            )]
            .into(),
            vec![
                SplitPoint::new(vec![ScalarValue::Int32(Some(10))]),
                SplitPoint::new(vec![ScalarValue::Int32(Some(20))]),
                SplitPoint::new(vec![ScalarValue::Int32(Some(30))]),
            ],
        )?);

        acc.build_filter(FinalizeInput::Partitioned(vec![
            reported(PushdownStrategy::Empty, no_bounds()),
            PartitionStatus::CanceledUnknown,
            reported(in_list(&[20, 29]), no_bounds()),
            reported(in_list(&[30]), no_bounds()),
        ]))?;

        let expr = current_expr(&acc);
        let case = case_expr(&expr);
        assert!(
            case.expr()
                .and_then(|expr| expr.downcast_ref::<RangeExpr>())
                .is_some(),
            "Range routing must use RangeExpr"
        );
        assert_eq!(case.when_then_expr().len(), 3);
        assert_literal_bool(&case.when_then_expr()[0].1, false);
        assert_literal_bool(
            case.else_expr().expect("expected permissive fallback"),
            true,
        );

        let batch = RecordBatch::try_new(
            test_probe_schema(),
            vec![Arc::new(Int32Array::from(vec![
                9, 10, 19, 20, 21, 29, 30, 31,
            ]))],
        )?;
        let result = expr.evaluate(&batch)?.into_array(batch.num_rows())?;
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("dynamic filter should evaluate to BooleanArray");
        assert_eq!(
            result,
            &BooleanArray::from(vec![false, true, true, true, false, true, true, false,])
        );

        Ok(())
    }

    #[test]
    fn partitioned_range_dynamic_filter_routes_compound_nullable_keys() -> Result<()> {
        let probe_schema = Arc::new(Schema::new(vec![
            Field::new("probe_key", DataType::Int32, true),
            Field::new("probe_tie", DataType::Int32, true),
        ]));
        let on_right: Vec<PhysicalExprRef> = vec![
            Arc::new(Column::new("probe_key", 0)),
            Arc::new(Column::new("probe_tie", 1)),
        ];
        let mut acc = make_accumulator_for_test(
            AccumulatedBuildData::Partitioned {
                partitions: vec![PartitionStatus::Pending; 4],
                completed_partitions: 0,
            },
            on_right,
        );
        acc.probe_schema = Arc::clone(&probe_schema);
        acc.probe_range_partitioning = Some(RangePartitioning::try_new(
            [
                PhysicalSortExpr::new(
                    Arc::clone(&acc.on_right[0]),
                    SortOptions::new(false, true),
                ),
                PhysicalSortExpr::new(
                    Arc::clone(&acc.on_right[1]),
                    SortOptions::new(false, false),
                ),
            ]
            .into(),
            vec![
                SplitPoint::new(vec![
                    ScalarValue::Int32(None),
                    ScalarValue::Int32(Some(10)),
                ]),
                SplitPoint::new(vec![ScalarValue::Int32(None), ScalarValue::Int32(None)]),
                SplitPoint::new(vec![
                    ScalarValue::Int32(Some(10)),
                    ScalarValue::Int32(None),
                ]),
            ],
        )?);

        acc.build_filter(FinalizeInput::Partitioned(vec![
            reported(PushdownStrategy::Empty, no_bounds()),
            PartitionStatus::CanceledUnknown,
            reported(PushdownStrategy::Empty, no_bounds()),
            PartitionStatus::CanceledUnknown,
        ]))?;

        let expr = current_expr(&acc);
        let case = case_expr(&expr);
        assert!(case.expr().is_some());
        assert_eq!(case.when_then_expr().len(), 2);
        assert_literal_bool(
            case.else_expr().expect("expected permissive fallback"),
            true,
        );

        let batch = RecordBatch::try_new(
            probe_schema,
            vec![
                Arc::new(Int32Array::from(vec![
                    None,
                    None,
                    None,
                    None,
                    Some(9),
                    Some(10),
                    Some(10),
                    Some(11),
                ])),
                Arc::new(Int32Array::from(vec![
                    Some(9),
                    Some(10),
                    Some(11),
                    None,
                    None,
                    Some(9),
                    None,
                    None,
                ])),
            ],
        )?;
        let result = expr.evaluate(&batch)?.into_array(batch.num_rows())?;
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("dynamic filter should evaluate to BooleanArray");
        assert_eq!(
            result,
            &BooleanArray::from(
                vec![false, true, true, false, false, false, true, true,]
            )
        );

        Ok(())
    }

    #[test]
    fn partitioned_range_dynamic_filter_preserves_signed_zero_routing() -> Result<()> {
        let probe_schema = Arc::new(Schema::new(vec![Field::new(
            "probe_key",
            DataType::Float64,
            false,
        )]));
        let on_right: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("probe_key", 0))];
        let mut acc = make_accumulator_for_test(
            AccumulatedBuildData::Partitioned {
                partitions: vec![PartitionStatus::Pending; 2],
                completed_partitions: 0,
            },
            on_right,
        );
        acc.probe_schema = Arc::clone(&probe_schema);
        acc.probe_range_partitioning = Some(RangePartitioning::try_new(
            [PhysicalSortExpr::new(
                Arc::clone(&acc.on_right[0]),
                SortOptions::default(),
            )]
            .into(),
            vec![SplitPoint::new(vec![ScalarValue::Float64(Some(0.0))])],
        )?);

        acc.build_filter(FinalizeInput::Partitioned(vec![
            PartitionStatus::CanceledUnknown,
            reported(PushdownStrategy::Empty, no_bounds()),
        ]))?;

        let expr = current_expr(&acc);
        let batch = RecordBatch::try_new(
            probe_schema,
            vec![Arc::new(Float64Array::from(vec![-0.0, 0.0]))],
        )?;
        let result = expr.evaluate(&batch)?.into_array(batch.num_rows())?;
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("dynamic filter should evaluate to BooleanArray");
        assert_eq!(result, &BooleanArray::from(vec![true, false]));

        Ok(())
    }

    // Regression guard for the build-report lifecycle fix: on `Drop`, a stream
    // in `BuildReportState::ReportScheduled` still calls `report_canceled_partition`
    // because it cannot tell whether the coordinator has already observed the
    // report (first poll of the `OnceFut` runs `store_build_data` synchronously
    // before the future's first `.await`, but the stream doesn't learn that
    // until `get_shared` returns `Ok`). Correctness therefore relies on
    // `store_canceled_partition` being a no-op when the partition is already
    // `Reported`. This test pins that invariant.
    #[test]
    fn report_canceled_partition_is_noop_after_report() {
        let acc = make_partitioned_accumulator_for_test(2);

        {
            let mut guard = acc.inner.lock();
            acc.store_build_data(
                &mut guard,
                PartitionBuildData::Partitioned {
                    partition_id: 0,
                    pushdown: PushdownStrategy::Empty,
                    bounds: PartitionBounds::new(vec![]),
                    keys_have_null: false,
                },
            )
            .unwrap();
        }
        let (partitions, completed) = partitioned_state(&acc);
        assert!(matches!(partitions[0], PartitionStatus::Reported(_)));
        assert_eq!(completed, 1);

        acc.report_canceled_partition(0);
        let (partitions, completed) = partitioned_state(&acc);
        assert!(
            matches!(partitions[0], PartitionStatus::Reported(_)),
            "late cancel must not overwrite a prior Reported status"
        );
        assert_eq!(completed, 1, "late cancel must not double-count completion");
    }

    // Drop from the `NotReported` (or first-poll-never-ran) state must
    // transition `Pending` -> `CanceledUnknown` and bump `completed_partitions`,
    // which is what unblocks sibling partitions waiting on the coordinator.
    #[test]
    fn report_canceled_partition_marks_pending_partition_canceled() {
        let acc = make_partitioned_accumulator_for_test(2);

        acc.report_canceled_partition(0);
        let (partitions, completed) = partitioned_state(&acc);
        assert!(matches!(partitions[0], PartitionStatus::CanceledUnknown));
        assert_eq!(completed, 1);

        // Idempotent: a second cancel (e.g. a stray double-drop) must not
        // double-count completion.
        acc.report_canceled_partition(0);
        let (partitions, completed) = partitioned_state(&acc);
        assert!(matches!(partitions[0], PartitionStatus::CanceledUnknown));
        assert_eq!(completed, 1);
    }

    fn null_semantics_accumulator(
        probe_schema: Arc<Schema>,
        on_right: Vec<PhysicalExprRef>,
        null_equality: NullEquality,
        null_aware: bool,
    ) -> SharedBuildAccumulator {
        SharedBuildAccumulator {
            inner: Mutex::new(AccumulatorState {
                data: AccumulatedBuildData::Partitioned {
                    partitions: vec![PartitionStatus::Pending; 1],
                    completed_partitions: 0,
                },
                completion: CompletionState::Pending,
                task_reports: None,
            }),
            completion_notify: Notify::new(),
            dynamic_filter: Arc::new(DynamicFilterPhysicalExpr::new(vec![], lit(true))),
            on_right,
            repartition_random_state: SeededRandomState::with_seed(1),
            probe_schema,
            probe_range_partitioning: None,
            null_equality,
            not_in: null_aware.then_some(NotIn::BuildRows {
                scope: NotInScope::Uncorrelated,
                output: NotInOutput::Rows,
            }),
            in_list_max_size: 128 * 1024,
            in_list_max_values: 150,
        }
    }

    fn null_equal_accumulator(
        probe_schema: Arc<Schema>,
        on_right: Vec<PhysicalExprRef>,
    ) -> SharedBuildAccumulator {
        null_semantics_accumulator(
            probe_schema,
            on_right,
            NullEquality::NullEqualsNull,
            false,
        )
    }

    fn reported_with_null_keys(
        pushdown: PushdownStrategy,
        bounds: PartitionBounds,
    ) -> PartitionStatus {
        PartitionStatus::Reported(PartitionData {
            pushdown,
            bounds,
            keys_have_null: true,
        })
    }

    /// A partitioned accumulator whose probe key is nullable, so that a build-side
    /// NULL can actually widen the filter.
    fn null_equal_partitioned_accumulator(
        num_partitions: usize,
    ) -> SharedBuildAccumulator {
        let mut acc = make_accumulator_for_test(
            AccumulatedBuildData::Partitioned {
                partitions: vec![PartitionStatus::Pending; num_partitions],
                completed_partitions: 0,
            },
            test_on_right(),
        );
        acc.probe_schema = Arc::new(Schema::new(vec![Field::new(
            "probe_key",
            DataType::Int32,
            true,
        )]));
        acc.null_equality = NullEquality::NullEqualsNull;
        acc
    }

    /// The IS NULL disjunct must wrap the merged filter as a whole. A probe NULL
    /// belongs to exactly one partition, so widening only the filter of the
    /// partition that reported the NULL would still drop it everywhere else.
    #[test]
    fn partitioned_null_keys_in_one_partition_widen_whole_routed_filter() {
        let acc = null_equal_partitioned_accumulator(2);

        acc.build_filter(FinalizeInput::Partitioned(vec![
            reported(in_list(&[1]), no_bounds()),
            reported_with_null_keys(in_list(&[2]), no_bounds()),
        ]))
        .unwrap();

        let expr = current_expr(&acc);
        assert_top_binary_op(&expr, Operator::Or);
        let widened = binary_expr(&expr);
        assert!(
            widened.left().downcast_ref::<IsNullExpr>().is_some(),
            "expected the IS NULL disjunct first, got: {expr}"
        );
        assert_in_list_column_values(widened.right(), "probe_key", 0, &[1, 2]);
        assert_eq!(
            format!("{expr}").matches("IS NULL").count(),
            1,
            "the routed filter must be widened once, not per branch"
        );
    }

    /// A canceled partition's build content is unknown, so it may hold a NULL key:
    /// the aggregated flag must be permissive even when no partition reported one.
    #[test]
    fn partitioned_canceled_partition_widens_routed_filter() {
        let acc = null_equal_partitioned_accumulator(2);

        acc.build_filter(FinalizeInput::Partitioned(vec![
            PartitionStatus::CanceledUnknown,
            reported(in_list(&[2]), no_bounds()),
        ]))
        .unwrap();

        let expr = current_expr(&acc);
        assert_top_binary_op(&expr, Operator::Or);
        assert!(
            binary_expr(&expr)
                .left()
                .downcast_ref::<IsNullExpr>()
                .is_some()
        );
    }

    #[test]
    fn preserve_probe_nulls_only_widens_nullable_keys() {
        let probe_schema = Arc::new(Schema::new(vec![
            Field::new("k_nullable", DataType::Int32, true),
            Field::new("k_not_null", DataType::Int32, false),
        ]));
        let on_right: Vec<PhysicalExprRef> = vec![
            Arc::new(Column::new("k_nullable", 0)),
            Arc::new(Column::new("k_not_null", 1)),
        ];
        let acc = null_equal_accumulator(probe_schema, on_right);

        // Only the nullable key earns an IS NULL disjunct; the NOT NULL key is left out.
        let widened = acc.preserve_probe_nulls(lit(true), true).unwrap();
        assert_eq!(format!("{widened}").matches("IS NULL").count(), 1);
    }

    #[test]
    fn preserve_probe_nulls_leaves_all_not_null_keys_untouched() {
        let probe_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let on_right: Vec<PhysicalExprRef> =
            vec![Arc::new(Column::new("a", 0)), Arc::new(Column::new("b", 1))];
        let acc = null_equal_accumulator(probe_schema, on_right);

        // Every key is NOT NULL, so there is nothing to OR in and the filter is returned as-is.
        let filter = lit(true);
        let result = acc.preserve_probe_nulls(Arc::clone(&filter), true).unwrap();
        assert_eq!(format!("{result}"), format!("{filter}"));
    }

    #[test]
    fn preserve_probe_nulls_rejects_out_of_sync_key() {
        let probe_schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        // The key's column index points past the probe schema: a construction bug that
        // must surface as an error, not get widened around.
        let on_right: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("b", 1))];
        let acc = null_equal_accumulator(probe_schema, on_right);

        assert!(acc.preserve_probe_nulls(lit(true), true).is_err());
    }

    #[test]
    fn preserve_probe_nulls_skips_wrap_when_build_has_no_nulls() {
        let probe_schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let on_right: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("a", 0))];
        let acc = null_equal_accumulator(probe_schema, on_right);

        // A NULL-free build has nothing for a probe NULL to null-match, so the
        // filter keeps its full selectivity.
        let filter = lit(true);
        let result = acc
            .preserve_probe_nulls(Arc::clone(&filter), false)
            .unwrap();
        assert_eq!(format!("{result}"), format!("{filter}"));
    }

    #[test]
    fn preserve_probe_nulls_wraps_null_aware_regardless_of_build() {
        let probe_schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let on_right: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("a", 0))];
        let acc = null_semantics_accumulator(
            probe_schema,
            on_right,
            NullEquality::NullEqualsNothing,
            true,
        );

        // One probe NULL collapses `NOT IN` for every build row, so the wrap must not
        // depend on the build content.
        let widened = acc.preserve_probe_nulls(lit(true), false).unwrap();
        assert_eq!(format!("{widened}").matches("IS NULL").count(), 1);
    }

    // A null-aware join with several probe keys has its `NOT IN` value keys
    // first (`on_right[..V]`: one for a scalar `NOT IN`, one per tuple element
    // for a multi-column one) and correlation scope keys after them. A NULL in
    // any value key can make `NOT IN` UNKNOWN, so the NULL escape must wrap
    // every value key and no scope key.
    #[test]
    fn null_aware_multi_key_filter_escapes_value_keys_only() {
        for value_keys in [1, 2] {
            let mut on_right: Vec<PhysicalExprRef> = (0..value_keys)
                .map(|i| Arc::new(Column::new(&format!("value_key_{i}"), i)) as _)
                .collect();
            on_right.push(Arc::new(Column::new("scope_key", value_keys)));
            let mut fields: Vec<Field> = (0..value_keys)
                .map(|i| Field::new(format!("value_key_{i}"), DataType::Int32, true))
                .collect();
            // Keep the scope key nullable so this test proves it is not widened.
            fields.push(Field::new("scope_key", DataType::Int32, true));

            let dynamic_filter = test_dynamic_filter(&on_right);
            let acc = SharedBuildAccumulator {
                inner: Mutex::new(AccumulatorState {
                    data: AccumulatedBuildData::CollectLeft {
                        data: PartitionStatus::Pending,
                        reported_count: 0,
                        expected_reports: 1,
                    },
                    completion: CompletionState::Pending,
                    task_reports: None,
                }),
                completion_notify: Notify::new(),
                dynamic_filter,
                on_right,
                repartition_random_state: SeededRandomState::with_seed(1),
                probe_schema: Arc::new(Schema::new(fields)),
                probe_range_partitioning: None,
                null_equality: NullEquality::NullEqualsNothing,
                not_in: Some(NotIn::BuildRows {
                    scope: NotInScope::Correlated { value_keys },
                    output: NotInOutput::Mark,
                }),
                in_list_max_size: 128 * 1024,
                in_list_max_values: 150,
            };

            let bounds = PartitionBounds::new(
                (0..=value_keys as i32)
                    .map(|i| {
                        ColumnBounds::new(
                            ScalarValue::Int32(Some(10 * i)),
                            ScalarValue::Int32(Some(10 * i + 5)),
                        )
                    })
                    .collect(),
            );
            acc.build_filter(FinalizeInput::CollectLeft(reported(
                PushdownStrategy::Empty,
                bounds,
            )))
            .unwrap();

            let expr = current_expr(&acc);
            let or = binary_expr(&expr);
            assert_eq!(or.op(), &Operator::Or);
            let mut escaped = vec![];
            collect_is_null_columns(or.left(), &mut escaped);
            assert_eq!(
                escaped,
                (0..value_keys).collect::<Vec<_>>(),
                "escape must target every NOT IN value key and nothing else"
            );
        }
    }

    /// Column indices under the `IS NULL` disjuncts of an `OR` chain.
    fn collect_is_null_columns(expr: &Arc<dyn PhysicalExpr>, out: &mut Vec<usize>) {
        if let Some(or) = expr.downcast_ref::<BinaryExpr>() {
            assert_eq!(or.op(), &Operator::Or);
            collect_is_null_columns(or.left(), out);
            collect_is_null_columns(or.right(), out);
        } else {
            let column = expr
                .downcast_ref::<IsNullExpr>()
                .expect("expected IS NULL escape")
                .arg()
                .downcast_ref::<Column>()
                .expect("expected column under IS NULL");
            out.push(column.index());
        }
    }

    // Family SF: a stage of four partitions executed by a task on 1..3.
    #[tokio::test]
    async fn split_task_reports_complete_and_prune() -> Result<()> {
        let acc = make_partitioned_expr_accumulator_for_test(4)
            .with_execution_partitions(Some(1..3));
        let reports = futures::future::try_join(
            acc.report_build_data(PartitionBuildData::Partitioned {
                partition_id: 1,
                pushdown: PushdownStrategy::InList(Arc::new(Int32Array::from(vec![7]))),
                bounds: bounds(7, 7),
                keys_have_null: false,
            }),
            acc.report_build_data(PartitionBuildData::Partitioned {
                partition_id: 2,
                pushdown: PushdownStrategy::Empty,
                bounds: no_bounds(),
                keys_have_null: false,
            }),
        );
        tokio::time::timeout(std::time::Duration::from_millis(100), reports)
            .await
            .expect("task-local reports must not wait for partitions 0 and 3")?;
        Ok(())
    }

    // Family SF: bounded execution must also work when partitions run sequentially.
    async fn task_report(
        acc: &SharedBuildAccumulator,
        data: PartitionBuildData,
    ) -> Result<()> {
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            acc.report_build_data(data),
        )
        .await
        .expect("a task report cannot wait for an unscheduled sibling")
    }

    fn task_filter_values(acc: &SharedBuildAccumulator, values: Vec<i32>) -> Vec<bool> {
        let batch = RecordBatch::try_new(
            test_probe_schema(),
            vec![Arc::new(Int32Array::from(values))],
        )
        .unwrap();
        let array = acc
            .dynamic_filter
            .current()
            .unwrap()
            .evaluate(&batch)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap();
        array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .values()
            .iter()
            .collect()
    }

    fn task_filter_complete(acc: &SharedBuildAccumulator) -> bool {
        use futures::FutureExt;
        acc.dynamic_filter.wait_complete().now_or_never().is_some()
    }

    #[tokio::test]
    async fn split_collect_left_prunes_before_serial_sibling_and_completes_after_it()
    -> Result<()> {
        let acc = make_collect_left_accumulator_for_test()
            .with_execution_partitions(Some(1..3));
        let data = |partition_id| PartitionBuildData::CollectLeft {
            partition_id,
            pushdown: PushdownStrategy::InList(Arc::new(Int32Array::from(vec![7]))),
            bounds: bounds(7, 7),
            keys_have_null: false,
        };
        task_report(&acc, data(1)).await?;
        assert_eq!(task_filter_values(&acc, vec![7, 8]), [true, false]);
        assert!(!task_filter_complete(&acc));
        let published = acc.dynamic_filter.snapshot_generation();
        // A duplicate report cannot stand in for partition 2.
        task_report(&acc, data(1)).await?;
        assert!(!task_filter_complete(&acc));
        task_report(&acc, data(2)).await?;
        assert!(task_filter_complete(&acc));
        // The second stream reports the same shared build: it completes the
        // filter without publishing a new generation for consumers to compile.
        assert_eq!(acc.dynamic_filter.snapshot_generation(), published);
        assert_eq!(task_filter_values(&acc, vec![7, 8]), [true, false]);
        Ok(())
    }

    #[tokio::test]
    async fn split_cancellation_completes_without_republishing() -> Result<()> {
        let acc = make_partitioned_expr_accumulator_for_test(4)
            .with_execution_partitions(Some(1..3));
        task_report(
            &acc,
            PartitionBuildData::Partitioned {
                partition_id: 1,
                pushdown: PushdownStrategy::Empty,
                bounds: no_bounds(),
                keys_have_null: false,
            },
        )
        .await?;
        let published = acc.dynamic_filter.snapshot_generation();
        let before = task_filter_values(&acc, (0..64).collect());
        acc.report_canceled_partition(2);
        assert!(task_filter_complete(&acc));
        assert_eq!(acc.dynamic_filter.snapshot_generation(), published);
        assert_eq!(task_filter_values(&acc, (0..64).collect()), before);
        Ok(())
    }

    #[tokio::test]
    async fn split_partitioned_prunes_only_complete_buckets() -> Result<()> {
        let acc = make_partitioned_expr_accumulator_for_test(4)
            .with_execution_partitions(Some(1..3));
        let values = (0..128).collect::<Vec<i32>>();
        let batch = RecordBatch::try_new(
            test_probe_schema(),
            vec![Arc::new(Int32Array::from(values.clone()))],
        )?;
        let hashes = HashExpr::new(
            test_on_right(),
            acc.repartition_random_state.clone(),
            "hash_repartition".into(),
        )
        .evaluate(&batch)?
        .into_array(values.len())?;
        let hashes = hashes
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        let key = values
            .iter()
            .copied()
            .find(|v| hashes.value(*v as usize) % 4 == 1)
            .unwrap();
        task_report(
            &acc,
            PartitionBuildData::Partitioned {
                partition_id: 1,
                pushdown: PushdownStrategy::InList(Arc::new(Int32Array::from(vec![key]))),
                bounds: bounds(key, key),
                keys_have_null: false,
            },
        )
        .await?;
        assert!(!task_filter_complete(&acc));
        let early = task_filter_values(&acc, values.clone());
        for (index, value) in values.iter().enumerate() {
            assert_eq!(early[index], hashes.value(index) % 4 != 1 || *value == key);
        }
        task_report(
            &acc,
            PartitionBuildData::Partitioned {
                partition_id: 2,
                pushdown: PushdownStrategy::Empty,
                bounds: no_bounds(),
                keys_have_null: false,
            },
        )
        .await?;
        assert!(task_filter_complete(&acc));
        let final_values = task_filter_values(&acc, values.clone());
        for (index, value) in values.iter().enumerate() {
            let bucket = hashes.value(index) % 4;
            assert_eq!(
                final_values[index],
                bucket == 0 || bucket == 3 || *value == key
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn empty_membership_preserves_null_equal_probe_rows() -> Result<()> {
        use arrow::array::Array;

        for scope in [None, Some(0..1)] {
            let mut acc = make_partitioned_expr_accumulator_for_test(1)
                .with_execution_partitions(scope);
            acc.null_equality = NullEquality::NullEqualsNull;
            acc.probe_schema = Arc::new(Schema::new(vec![Field::new(
                "probe_key",
                DataType::Int32,
                true,
            )]));
            task_report(
                &acc,
                PartitionBuildData::Partitioned {
                    partition_id: 0,
                    pushdown: PushdownStrategy::Empty,
                    bounds: no_bounds(),
                    keys_have_null: true,
                },
            )
            .await?;
            let input = RecordBatch::try_new(
                Arc::clone(&acc.probe_schema),
                vec![Arc::new(Int32Array::from(vec![Some(7), None]))],
            )?;
            let mask = acc.dynamic_filter.evaluate(&input)?.into_array(2)?;
            let mask = mask.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert_eq!(mask.null_count(), 0);
            assert!(!mask.value(0));
            assert!(
                mask.value(1),
                "an empty membership set can still contain NULL"
            );
            assert!(task_filter_complete(&acc));
        }
        Ok(())
    }

    #[tokio::test]
    async fn split_empty_collect_left_and_cancellation_wake_completion() -> Result<()> {
        for empty in [false, true] {
            let acc = make_collect_left_accumulator_for_test()
                .with_execution_partitions(Some(1..3));
            if empty {
                task_report(
                    &acc,
                    PartitionBuildData::CollectLeft {
                        partition_id: 1,
                        pushdown: PushdownStrategy::Empty,
                        bounds: no_bounds(),
                        keys_have_null: false,
                    },
                )
                .await?;
            } else {
                acc.report_canceled_partition(1);
            }
            let waiter = acc.dynamic_filter.wait_complete();
            futures::pin_mut!(waiter);
            assert!(futures::poll!(&mut waiter).is_pending());
            acc.report_canceled_partition(2);
            tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
                .await
                .unwrap();
            // CollectLeft retains upstream's permissive empty-build predicate;
            // the join itself returns its correct empty-build result.
            assert_eq!(task_filter_values(&acc, vec![7, 8]), [true, true]);
        }
        Ok(())
    }

    #[tokio::test]
    async fn split_partition_cancellation_preserves_unknown_rows() -> Result<()> {
        let acc = make_partitioned_expr_accumulator_for_test(4)
            .with_execution_partitions(Some(1..3));
        task_report(
            &acc,
            PartitionBuildData::Partitioned {
                partition_id: 1,
                pushdown: PushdownStrategy::Empty,
                bounds: no_bounds(),
                keys_have_null: false,
            },
        )
        .await?;
        acc.report_canceled_partition(2);
        assert!(task_filter_complete(&acc));
        // Cancel is idempotent and cannot erase the known-empty bucket.
        acc.report_canceled_partition(1);
        assert!(
            acc.report_build_data(PartitionBuildData::Partitioned {
                partition_id: 0,
                pushdown: PushdownStrategy::Empty,
                bounds: no_bounds(),
                keys_have_null: false,
            })
            .await
            .is_err()
        );
        assert!(
            task_filter_values(&acc, vec![0, 1, 2, 3])
                .iter()
                .any(|keep| *keep)
        );
        Ok(())
    }
    #[tokio::test]
    async fn split_teardown_wakes_filter_observers_with_unstarted_partitions()
    -> Result<()> {
        let acc = make_partitioned_expr_accumulator_for_test(4)
            .with_execution_partitions(Some(1..4));
        task_report(
            &acc,
            PartitionBuildData::Partitioned {
                partition_id: 1,
                pushdown: PushdownStrategy::Empty,
                bounds: no_bounds(),
                keys_have_null: false,
            },
        )
        .await?;
        let filter = Arc::clone(&acc.dynamic_filter);
        let waiter = filter.wait_complete();
        futures::pin_mut!(waiter);
        assert!(futures::poll!(&mut waiter).is_pending());
        drop(acc);
        tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .unwrap();
        Ok(())
    }

    // Family SF: a recursive-CTE `reset_state` resets the shared filter for
    // the next iteration; dropping the previous iteration's scoped
    // accumulator afterwards must not complete the new iteration's filter.
    #[tokio::test]
    async fn split_teardown_after_reset_keeps_the_next_iteration_open() -> Result<()> {
        let acc = make_partitioned_expr_accumulator_for_test(4)
            .with_execution_partitions(Some(1..4));
        task_report(
            &acc,
            PartitionBuildData::Partitioned {
                partition_id: 1,
                pushdown: PushdownStrategy::InList(Arc::new(Int32Array::from(vec![7]))),
                bounds: bounds(7, 7),
                keys_have_null: false,
            },
        )
        .await?;
        let filter = Arc::clone(&acc.dynamic_filter);
        filter.reset();
        drop(acc);
        let waiter = filter.wait_complete();
        futures::pin_mut!(waiter);
        assert!(
            futures::poll!(&mut waiter).is_pending(),
            "the reset filter belongs to the next iteration"
        );
        Ok(())
    }
}
