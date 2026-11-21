// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! 2-stage hash aggregation stream implementation.
//!
//! See comments in [`PartialHashAggregateStream`] and [`FinalHashAggregateStream`]
//! for details.

use std::mem::size_of;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{
    DataFusionError, Result, assert_ne_or_internal_err, internal_datafusion_err,
};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::{TaskContext, TryEmitter, async_try_stream};
use futures::stream::{Stream, StreamExt};

use super::AggregateExec;
use super::aggregate_hash_table::{
    AggregateHashTable, ClusteredAggregateTableMetrics, FinalMarker, PartialMarker,
    PartialSkipMarker,
};
use super::bounded_partial::{BoundedPartial, configured_groups};
use super::group_values::packed::migration_needs_memory;
use super::order::GroupClusteringMode;
use super::skip_partial::SkipAggregationProbe;
use super::spill::AggregateSpill;
use crate::metrics::{
    BaselineMetrics, MetricBuilder, MetricCategory, RecordOutput, SpillMetrics,
};
use crate::stream::{EmptyRecordBatchStream, RecordBatchStreamAdapter};
use crate::{SendableRecordBatchStream, metrics};

/// Hash aggregation is implemented in two stages: partial and final. This
/// stream implements the partial stage.
///
/// # Example
///
/// SELECT k, AVG(v) FROM t GROUP BY k;
///
/// ## Plan
/// AggregateExec(stage=final)
/// -- RepartitionExec(hash(k))
/// ---- AggregateExec(stage=partial)
///
/// ## Partial Stage Behavior
/// Input: raw rows
/// Output: partial states for all groups (for example, `AVG(x)` emits `SUM(x)`
/// and `COUNT(x)`)
///
/// ## Final Stage Behavior
/// Input: partial states
/// Output: results for all groups (for example, `AVG(x)` calculated from the
/// state)
///
/// # Optimization: DISTINCT LIMIT Soft Limit
///
/// This optimization applies to both [`PartialHashAggregateStream`] and
/// [`FinalHashAggregateStream`].
///
/// Unordered distinct queries such as:
///
/// ```sql
/// SELECT DISTINCT x FROM t LIMIT 10;
/// ```
///
/// are optimized into a two-stage aggregate like:
///
/// ```txt
/// LimitExec, limit=10
/// --AggregateExec(Final), group_by=[x], aggr=[], soft_limit=10
/// ---- RepartitionExec, partitioning=hash(x)
/// ------ AggregateExec(Partial), group_by=[x], aggr=[], soft_limit=10
/// -------- Scan(t)
/// ```
///
/// After each input batch, the stream checks whether the soft limit has been
/// reached. If so, it emits the accumulated groups and stops reading input.
///
/// This operator does not guarantee an exact limit because a single batch can
/// cross the threshold. The downstream limit operator enforces the exact result
/// size.
///
/// # Optimization: Partial Aggregation Skip
///
/// Partial aggregation can be counterproductive for high-cardinality inputs,
/// where most rows create distinct groups. The stream probes the ratio of
/// accumulated groups to input rows while it is still aggregating. If the ratio
/// crosses the configured threshold and all aggregate accumulators can convert
/// raw inputs directly to partial state, the stream emits any already
/// accumulated groups, then switches to a skip state. In that state, each
/// remaining input batch is converted directly to partial aggregate state rows
/// without inserting the rows into the grouped hash table.
///
/// # Feature: Grouping Sets
///
/// `GROUPING SETS`, `CUBE` and `ROLLUP` are expanded in the partial stage: every
/// grouping set of an input batch is evaluated (with the grouping expressions
/// that are not part of the set replaced by `NULL`, plus an internal
/// `__grouping_id` column) and interned into the same hash table. The final
/// stage then merges the expanded keys as a plain group by.
///
/// The partial aggregation skip optimization is disabled for grouping sets.
///
/// # Feature: Memory-limited Execution
///
/// ## Partial Aggregation
///
/// Partial aggregation can emit incomplete results because the final stage merges
/// all intermediate states for the same group. If the memory reservation exceeds
/// its limit after aggregating an input batch, this stream emits all accumulated
/// states and continues aggregating the remaining input with an empty table.
///
/// ## Final Aggregation
///
/// During final aggregation, group keys and states accumulate. If memory usage
/// exceeds the budget, spilling is triggered as follows:
/// 1. After aggregating a new input batch, if the memory reservation exceeds its
///    limit, spill all accumulated groups and states.
///    - Sort all groups by the group keys before spilling.
/// 2. Repeat until the input is exhausted.
/// 3. Perform a sort-preserving merge of all spill files and feed the merged output
///    into an ordered streaming aggregation, which ensures bounded memory usage and
///    evaluates the final result.
///    - [`ClusteredFinalAggregateStream`](super::clustered_final_stream::ClusteredFinalAggregateStream) is reused for the streaming aggregation.
pub(crate) struct PartialHashAggregateStream {
    /// Output schema: group columns followed by partial aggregate state columns.
    schema: SchemaRef,

    /// Input batches containing raw rows, not partial aggregate state.
    input: SendableRecordBatchStream,

    /// Target output batch size from configuration.
    batch_size: usize,

    /// Memory reservation for group keys and accumulators.
    reservation: MemoryReservation,

    /// Execution metrics shared with the aggregate plan node.
    baseline_metrics: BaselineMetrics,

    /// Tracks partial aggregation row reduction, matching `GroupedHashAggregateStream`.
    reduction_factor: metrics::RatioMetrics,

    /// Number of times accumulated states were emitted due to memory pressure.
    early_emit_count: metrics::Count,

    /// Tracks whether partial aggregation should switch to direct state conversion.
    skip_aggregation_probe: Option<SkipAggregationProbe>,

    /// Optional soft limit on the number of groups to accumulate before output.
    ///
    /// Invariant: when this is `Some(..)`, the accumulators inside `hash_table` must
    /// be empty. See struct comments for details.
    group_values_soft_limit: Option<usize>,

    /// Cache-sized flush policy, disabled unless configured for a single set.
    bounded: BoundedPartial,

    /// Once a bounded flush occurred, ordinary pressure output must also
    /// count in the probe, including after the policy escapes to unbounded.
    pressure_emitted_groups: usize,

    /// Leading key columns used to estimate distinct keys across flushes.
    num_group_columns: usize,

    /// The hash table owns the lower-level state for emitting output batches.
    hash_table: Option<AggregateHashTable<PartialMarker>>,
}

/// Hash aggregation is implemented in two stages: partial and final. This
/// stream implements the final stage.
///
/// See [`PartialHashAggregateStream`] for details.
pub(crate) struct FinalHashAggregateStream {
    /// Output schema: group columns followed by final aggregate value columns.
    schema: SchemaRef,

    /// Input batches containing partial aggregate state rows.
    input: SendableRecordBatchStream,

    /// Execution metrics shared with the aggregate plan node.
    baseline_metrics: BaselineMetrics,

    /// Memory reservation for group keys, accumulators, and spill sorting.
    reservation: MemoryReservation,

    /// See comments for the same variable in [`PartialHashAggregateStream`].
    group_values_soft_limit: Option<usize>,

    /// The hash table owns the lower-level
    /// state for emitting output batches.
    ///
    /// This will be None when creating the stream
    hash_table: Option<AggregateHashTable<FinalMarker>>,
    /// `None` if spilling is not supported by the configured `DiskManager`.
    spill_context: Option<Box<AggregateSpill>>,
}

#[derive(PartialEq)]
enum HandleInputResult {
    ProcessNext,
    ReachedLimit,
    #[expect(clippy::upper_case_acronyms)]
    OOM,
    SwitchToSkipAggregation,
    FlushBounded,
}

impl PartialHashAggregateStream {
    pub fn new(
        agg: &AggregateExec,
        context: &Arc<TaskContext>,
        partition: usize,
    ) -> Result<Self> {
        debug_assert_eq!(agg.mode, super::AggregateMode::Partial);
        debug_assert_eq!(agg.group_clustering_mode, GroupClusteringMode::None);

        let schema = Arc::clone(&agg.schema);
        let input = agg.input.execute(partition, Arc::clone(context))?;
        let batch_size = context.session_config().batch_size();
        let baseline_metrics = BaselineMetrics::new(&agg.metrics, partition);

        // Preserve the existing aggregate metric surface for this plan node.
        let _spill_metrics = SpillMetrics::new(&agg.metrics, partition);
        let reduction_factor = MetricBuilder::new(&agg.metrics)
            .with_type(metrics::MetricType::Summary)
            .ratio_metrics("reduction_factor", partition);
        let early_emit_count =
            MetricBuilder::new(&agg.metrics).counter("early_emit_count", partition);

        let mut hash_table = AggregateHashTable::<PartialMarker>::new(
            agg,
            partition,
            Arc::clone(&schema),
            batch_size,
        )?;
        let skip_aggregation_probe = if agg.group_by().is_single() {
            let options = &context.session_config().options().execution;
            let probe_ratio_threshold =
                options.skip_partial_aggregation_probe_ratio_threshold;
            // A threshold >= 1.0 means the ratio (num_groups / input_rows) can
            // never exceed it, so the feature is effectively disabled.
            if probe_ratio_threshold >= 1.0 {
                None
            } else {
                let skipped_aggregation_rows = MetricBuilder::new(&agg.metrics)
                    .with_category(MetricCategory::Rows)
                    .counter("skipped_aggregation_rows", partition);
                Some(SkipAggregationProbe::new(
                    options.skip_partial_aggregation_probe_rows_threshold,
                    probe_ratio_threshold,
                    skipped_aggregation_rows,
                ))
            }
        } else {
            None
        };

        let reservation =
            MemoryConsumer::new(format!("PartialHashAggregateStream[{partition}]"))
                // We interpret 'can spill' as 'can handle memory back pressure'.
                // This value needs to be set to true for the default memory pool implementations
                // to ensure fair application of back pressure amongst the memory consumers.
                .with_can_spill(true)
                .register(context.memory_pool());

        // Reserve memory for the initial hash table. that we hold for the lifetime of the stream.
        hash_table.set_migration_reservation(&reservation);
        reservation.try_grow(hash_table.memory_size())?;
        let group_values_soft_limit = agg.limit_options().map(|config| config.limit());
        // DISTINCT LIMIT retains its early-stop semantics. Grouping sets must
        // expand together; their raw stream uses the upstream pressure policy.
        let max_groups =
            if group_values_soft_limit.is_none() && agg.group_by().is_single() {
                configured_groups(context.session_config().options())
            } else {
                0
            };
        let bounded = BoundedPartial::new(max_groups, &agg.metrics, partition);

        Ok(Self {
            schema,
            input,
            batch_size,
            baseline_metrics,
            reservation,
            reduction_factor,
            early_emit_count,
            skip_aggregation_probe,
            group_values_soft_limit,
            bounded,
            pressure_emitted_groups: 0,
            num_group_columns: agg.group_by().num_group_exprs(),
            hash_table: Some(hash_table),
        })
    }

    pub(crate) fn into_stream(self) -> SendableRecordBatchStream {
        let schema = Arc::clone(&self.schema);

        Box::pin(RecordBatchStreamAdapter::new(schema, self.create_stream()))
    }

    /// Entry point for the partial hash aggregate state machine.
    ///
    /// See comments in [`PartialHashAggregateStream`] for high-level ideas.
    fn create_stream(mut self) -> impl Stream<Item = Result<RecordBatch>> {
        async_try_stream(|mut emitter| async move {
            let mut hash_table = self
                .hash_table
                .take()
                .expect("hash_table should not be None");

            debug_assert!(hash_table.is_building());
            let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();

            let mut last_state = HandleInputResult::ProcessNext;
            'input: while let Some(batch) = self.input.next().await.transpose()? {
                loop {
                    let timer = elapsed_compute.timer();
                    let retry = match self.handle_input_batch(&batch, &mut hash_table) {
                        Ok(state) => {
                            last_state = state;
                            false
                        }
                        Err(error) if migration_needs_memory(&error) => {
                            last_state = HandleInputResult::OOM;
                            true
                        }
                        Err(error) => return Err(error),
                    };
                    match last_state {
                        HandleInputResult::ProcessNext => {}
                        HandleInputResult::ReachedLimit
                        | HandleInputResult::SwitchToSkipAggregation => {
                            break 'input;
                        }
                        HandleInputResult::FlushBounded => {
                            let retain = self.bounded.bound().unwrap_or(0);
                            let states = hash_table.take_state_batch_retaining(retain)?.ok_or_else(|| {
                            internal_datafusion_err!(
                                "Bounded partial hash aggregate flushed with no aggregated groups"
                            )
                        })?;
                            self.bounded.record_flush(
                                &states.columns()[..self.num_group_columns],
                                states.num_rows(),
                                self.batch_size,
                            )?;
                            // Charge the optional sketch's actual buffers. If they
                            // do not fit, disable it and preserve upstream progress.
                            if let Err(error) = self.reservation.try_resize(
                                hash_table.memory_size() + self.bounded.memory_size(),
                            ) {
                                match error {
                                    DataFusionError::ResourcesExhausted(_) => {
                                        self.bounded.unbound()
                                    }
                                    error => return Err(error),
                                }
                            }
                            timer.done();
                            self.emit_on_memory_pressure(
                                states,
                                &mut emitter,
                                hash_table.memory_size() + self.bounded.memory_size(),
                            )
                            .await?;
                        }
                        HandleInputResult::OOM => {
                            let materialized_group_states = hash_table.take_state_batch()?.ok_or_else(|| {
                            internal_datafusion_err!(
                                "Partial hash aggregate ran out of memory with no aggregated groups"
                            )
                        })?;

                            // Preserve upstream probe behavior until G has emitted
                            // states. Thereafter account all output, even if G's
                            // reduction/HLL guard has since removed the bound.
                            if self.bounded.flushed_groups() > 0 {
                                self.pressure_emitted_groups +=
                                    materialized_group_states.num_rows();
                            }
                            // Under pool pressure use the native policy and release
                            // the optional sketch as well as the table's capacity.
                            self.bounded.unbound();
                            self.early_emit_count.add(1);
                            timer.done();
                            self.emit_on_memory_pressure(
                                materialized_group_states,
                                &mut emitter,
                                hash_table.memory_size() + self.bounded.memory_size(),
                            )
                            .await?;
                        }
                    }
                    if !retry {
                        break;
                    }
                }
            }

            let timer = elapsed_compute.timer();

            let skip_hash_table =
                if last_state == HandleInputResult::SwitchToSkipAggregation {
                    Some(hash_table.partial_skip_table()?)
                } else {
                    self.close_input();

                    None
                };
            self.bounded.unbound();
            hash_table.start_output()?;

            timer.done();

            self.produce_output(hash_table, &mut emitter).await?;

            if let Some(hash_table) = skip_hash_table {
                self.skip_rest_of_aggregation(hash_table, emitter).await?;
            }

            Ok(())
        })
    }

    /// See comments in [`Self::group_values_soft_limit`] for details.
    fn hit_soft_group_limit(
        &self,
        hash_table: &AggregateHashTable<PartialMarker>,
    ) -> bool {
        self.group_values_soft_limit
            .is_some_and(|limit| limit <= hash_table.building_group_count())
    }

    /// Updates skip aggregation probe state.
    fn update_skip_aggregation_probe(&mut self, input_rows: usize, num_groups: usize) {
        if let Some(probe) = self.skip_aggregation_probe.as_mut() {
            probe.update_state(input_rows, num_groups);
        }
    }

    /// Returns true if the aggregation probe indicates that aggregation
    /// should be skipped.
    fn should_skip_aggregation(&self) -> bool {
        self.skip_aggregation_probe
            .as_ref()
            .is_some_and(|probe| probe.should_skip())
    }

    fn close_input(&mut self) {
        let input_schema = self.input.schema();
        self.input = Box::pin(EmptyRecordBatchStream::new(input_schema));
    }

    /// Aggregate input batch into the hash table
    fn handle_input_batch(
        &mut self,
        batch: &RecordBatch,
        hash_table: &mut AggregateHashTable<PartialMarker>,
    ) -> Result<HandleInputResult> {
        // ----------------------------------
        // Step 1: Aggregate the input batch
        // ----------------------------------
        let input_rows = batch.num_rows();
        hash_table.aggregate_batch(batch)?;
        self.reduction_factor.add_total(input_rows);

        // --------------------------------
        // Step 2: Soft limit optimization
        // --------------------------------
        if self.hit_soft_group_limit(hash_table) {
            return Ok(HandleInputResult::ReachedLimit);
        }

        // ----------------------------------------------
        // Step 3: Skip partial aggregation optimization
        // ----------------------------------------------
        // A bounded flush must not hide the partial output from the probe.
        self.update_skip_aggregation_probe(
            input_rows,
            self.bounded.flushed_groups()
                + self.pressure_emitted_groups
                + hash_table.building_group_count(),
        );

        // True branch: a decision has been made to skip partial aggregation.
        if self.should_skip_aggregation() {
            return Ok(HandleInputResult::SwitchToSkipAggregation);
        }

        // The skip probe has priority. A low-reduction cycle disables this
        // optional bound; ordinary memory-pressure emission remains available.
        let flush = self
            .bounded
            .after_batch(input_rows, hash_table.building_group_count());

        // Pressure takes precedence over retaining a bounded table's map:
        // clearing with zero must remain possible when that map no longer fits.
        match self
            .reservation
            .try_resize(hash_table.memory_size() + self.bounded.memory_size())
        {
            Ok(()) if flush => Ok(HandleInputResult::FlushBounded),
            Ok(()) => Ok(HandleInputResult::ProcessNext),
            Err(DataFusionError::ResourcesExhausted(_)) => Ok(HandleInputResult::OOM),
            Err(e) => Err(e),
        }
    }

    /// Emit a materialized partial-state batch in `batch_size` slices.
    async fn emit_on_memory_pressure(
        &mut self,
        mut remaining_groups: RecordBatch,
        emitter: &mut TryEmitter<RecordBatch, DataFusionError>,
        hash_table_mem_size: usize,
    ) -> Result<()> {
        let remaining_groups_memory = remaining_groups.get_array_memory_size();

        match self
            .reservation
            .try_resize(hash_table_mem_size + remaining_groups_memory)
        {
            Ok(()) => {}
            Err(DataFusionError::ResourcesExhausted(_)) => {
                self.reservation.try_resize(hash_table_mem_size)?;
                self.reduction_factor.add_part(remaining_groups.num_rows());
                emitter
                    .emit(remaining_groups.record_output(&self.baseline_metrics))
                    .await;
                return Ok(());
            }
            Err(error) => return Err(error),
        }

        while remaining_groups.num_rows() > self.batch_size {
            let output = remaining_groups.slice(0, self.batch_size);
            remaining_groups = remaining_groups.slice(
                self.batch_size,
                remaining_groups.num_rows() - self.batch_size,
            );
            self.reduction_factor.add_part(output.num_rows());
            emitter
                .emit(output.record_output(&self.baseline_metrics))
                .await;
        }

        self.reduction_factor.add_part(remaining_groups.num_rows());
        self.reservation.try_shrink(remaining_groups_memory)?;
        emitter
            .emit(remaining_groups.record_output(&self.baseline_metrics))
            .await;
        Ok(())
    }

    /// emit partial aggregate state batches.
    async fn produce_output(
        &mut self,
        mut hash_table: AggregateHashTable<PartialMarker>,
        emitter: &mut TryEmitter<RecordBatch, DataFusionError>,
    ) -> Result<()> {
        debug_assert!(!hash_table.is_building());

        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();
        let mut timer = elapsed_compute.timer();

        loop {
            let Some(batch) = hash_table.next_output_batch()? else {
                // Only reachable when the table held no groups at all: a
                // non-empty table always reports its last batch together with
                // the `Done` state, which the `try_resize` below already zeroes.
                self.reservation.try_resize(0)?;
                return Ok(());
            };

            debug_assert!(batch.num_rows() > 0);

            // The table hands over its groups as they are materialized and
            // reports a size of 0 once it reaches `Done`, so this releases the
            // reservation before the final batch goes downstream.
            let _ = self.reservation.try_resize(hash_table.memory_size());
            self.reduction_factor.add_part(batch.num_rows());

            timer.done();
            emitter
                .emit(batch.record_output(&self.baseline_metrics))
                .await;
            timer = elapsed_compute.timer();
        }
    }

    /// convert raw input directly to partial states.
    async fn skip_rest_of_aggregation(
        &mut self,
        mut hash_table: AggregateHashTable<PartialSkipMarker>,
        mut emitter: TryEmitter<RecordBatch, DataFusionError>,
    ) -> Result<()> {
        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();

        while let Some(batch) = self.input.next().await.transpose()? {
            if let Some(probe) = self.skip_aggregation_probe.as_mut() {
                probe.record_skipped(&batch);
            }

            let result = {
                let _timer = elapsed_compute.timer();
                hash_table.convert_batch_to_state(&batch)?
            };

            emitter
                .emit(result.record_output(&self.baseline_metrics))
                .await;
        }

        self.close_input();

        Ok(())
    }
}

impl FinalHashAggregateStream {
    pub fn new(
        agg: &AggregateExec,
        context: &Arc<TaskContext>,
        partition: usize,
    ) -> Result<Self> {
        debug_assert!(matches!(
            agg.mode,
            super::AggregateMode::Final | super::AggregateMode::FinalPartitioned
        ));
        debug_assert_eq!(agg.group_clustering_mode, GroupClusteringMode::None);

        let input = agg.input.execute(partition, Arc::clone(context))?;
        Self::new_with_input(agg, context, partition, input)
    }

    /// Builds the stream over `input` instead of executing the plan's input.
    pub(in crate::aggregates) fn new_with_input(
        agg: &AggregateExec,
        context: &Arc<TaskContext>,
        partition: usize,
        input: SendableRecordBatchStream,
    ) -> Result<Self> {
        let schema = Arc::clone(&agg.schema);
        let input_schema = input.schema();
        let batch_size = context.session_config().batch_size();
        let baseline_metrics = BaselineMetrics::new(&agg.metrics, partition);
        let spill_metrics = SpillMetrics::new(&agg.metrics, partition);

        let mut hash_table = AggregateHashTable::<FinalMarker>::new(
            agg,
            partition,
            Arc::clone(&schema),
            batch_size,
        )?;

        let can_spill = context.runtime_env().disk_manager.tmp_files_enabled();
        let spill_context = if can_spill {
            Some(Box::new(AggregateSpill::try_new(
                "FinalHashAggregateSpill",
                agg,
                context,
                partition,
                batch_size,
                &GroupClusteringMode::None,
                &input_schema,
                spill_metrics,
            )?))
        } else {
            None
        };

        let reservation =
            MemoryConsumer::new(format!("FinalHashAggregateStream[{partition}]"))
                .with_can_spill(can_spill)
                .register(context.memory_pool());

        // Reserve memory for the initial hash table. that we hold for the lifetime of the stream.
        hash_table.set_migration_reservation(&reservation);
        reservation.try_grow(hash_table.memory_size())?;

        Ok(Self {
            schema,
            input,
            baseline_metrics,
            reservation,
            group_values_soft_limit: agg.limit_options().map(|config| config.limit()),
            hash_table: Some(hash_table),
            spill_context,
        })
    }

    pub(crate) fn into_stream(self) -> SendableRecordBatchStream {
        let schema = Arc::clone(&self.schema);

        Box::pin(RecordBatchStreamAdapter::new(schema, self.create_stream()))
    }

    /// Entry point for the final hash aggregate flow
    ///
    /// See comments in [`FinalHashAggregateStream`] for high-level ideas.
    fn create_stream(mut self) -> impl Stream<Item = Result<RecordBatch>> {
        async_try_stream(|emitter| async move {
            let mut hash_table = self
                .hash_table
                .take()
                .expect("hash_table should not be None");

            let mut spill_context = self.spill_context.take();

            self.consume_input(&mut hash_table, &mut spill_context)
                .await?;
            self.close_input();

            match spill_context {
                // - If spilled before, perform merging spill runs
                Some(spill_context) if spill_context.has_spills() => {
                    self.produce_output_from_spills(hash_table, spill_context, emitter)
                        .await?
                }
                // Either all the input fit in memory or hit soft group limit with no spilling
                spill_context => {
                    self.produce_output_from_memory(hash_table, spill_context, emitter)
                        .await?
                }
            }

            Ok(())
        })
    }

    fn close_input(&mut self) {
        let input_schema = self.input.schema();
        self.input = Box::pin(EmptyRecordBatchStream::new(input_schema));
    }

    /// See comments in [`Self::group_values_soft_limit`] for details.
    fn hit_soft_group_limit(&self, hash_table: &AggregateHashTable<FinalMarker>) -> bool {
        self.group_values_soft_limit
            .is_some_and(|limit| limit <= hash_table.building_group_count())
    }

    /// Reserve memory for the current aggregate table.
    fn reservation_size_for_table(
        hash_table: &AggregateHashTable<FinalMarker>,
        spill_context: Option<&AggregateSpill>,
    ) -> usize {
        let table_size = hash_table.memory_size();
        if spill_context.is_some() {
            // Count extra space needed for in-memory sorting and spilling. Only
            // count memory for indices, the payload will be materialize incrementally
            // in smaller chunks.
            table_size.saturating_add(
                hash_table
                    .building_group_count()
                    .saturating_mul(size_of::<u32>()),
            )
        } else {
            table_size
        }
    }

    /// Read input stream, if no memory, then spill and continue reading - aggregate partial state batches into the hash table.
    ///
    /// Spilling: The table cannot reserve enough memory.
    ///           Move all current states into one fully group-key-sorted spill run.
    async fn consume_input(
        &mut self,
        hash_table: &mut AggregateHashTable<FinalMarker>,
        spill_context: &mut Option<Box<AggregateSpill>>,
    ) -> Result<()> {
        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();

        'input: while let Some(batch) = self.input.next().await.transpose()? {
            let _timer = elapsed_compute.timer();
            loop {
                let migration_error = match hash_table.aggregate_batch(&batch) {
                    Ok(()) => None,
                    Err(error) if migration_needs_memory(&error) => Some(error),
                    Err(error) => return Err(error),
                };
                let retry = migration_error.is_some();

                // Soft group limits are usually small and rarely coincide with
                // spilling. Once spilling has occurred, skip this optimization to
                // make the internal logic simpler.
                let spilled = spill_context
                    .as_ref()
                    .is_some_and(|context| context.has_spills());
                if !retry && self.hit_soft_group_limit(hash_table) && !spilled {
                    break 'input;
                }

                // Check memory reservation, and potentially spill.
                let resize_result = migration_error.map_or_else(
                    || {
                        self.reservation
                            .try_resize(Self::reservation_size_for_table(
                                hash_table,
                                spill_context.as_deref(),
                            ))
                    },
                    Err,
                );

                match resize_result {
                    Ok(()) => break,

                    // The table cannot reserve enough memory.
                    // Move all current states into one fully group-key-sorted spill run.
                    Err(e)
                        if matches!(
                            e.find_root(),
                            DataFusionError::ResourcesExhausted(_)
                        ) =>
                    {
                        // OOM and don't support spilling from configuration
                        let spill_context = spill_context.as_mut().ok_or_else(|| e.context(
                        "Final hash aggregate cannot spill because temporary files are not enabled in the DiskManager",
                    ))?;

                        // Sanity check: impossible to OOM when there is no group aggregated.
                        assert_ne_or_internal_err!(
                            hash_table.building_group_count(),
                            0,
                            "Final hash aggregate ran out of memory with no aggregated groups"
                        );

                        // Sorts and spills one complete in-memory state run

                        // Go to the next state to perform spilling the aggregated
                        // groups so far.
                        let result = hash_table
                            .take_state_batch()
                            .and_then(|batch| spill_context.sort_and_spill(batch));
                        if result.is_ok() {
                            spill_context.retain_merge_workspace(&mut self.reservation);
                        }

                        // Spilling shrinks the aggregate table and releases its accumulated
                        // memory. Update the reservation accordingly.
                        self.reservation
                            .try_resize(hash_table.memory_size())
                            .map_err(|e| {
                                e.context(
                                    "Decreasing allocation after spilling should succeed",
                                )
                            })?;

                        result?;

                        // One sorted run was written; resume reading the original input.
                    }
                    Err(e) => return Err(e),
                }
                if !retry {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Produce output from spills
    /// 1. Spill in progress in-memory hash table
    /// 2. Switch to clustered final stream
    /// 3. passthrough stream output
    async fn produce_output_from_spills(
        &mut self,
        mut hash_table: AggregateHashTable<FinalMarker>,
        mut spill_context: Box<AggregateSpill>,
        mut emitter: TryEmitter<RecordBatch, DataFusionError>,
    ) -> Result<()> {
        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();
        let timer = elapsed_compute.timer();

        // Input was exhausted after spilling. Spill the last in-memory run
        hash_table
            .take_state_batch()
            .and_then(|batch| spill_context.sort_and_spill(batch))?;
        spill_context.retain_merge_workspace(&mut self.reservation);

        // Construct the ordered input used to merge all spill files.
        let mut output_stream =
            self.switch_to_clustered_final_stream(hash_table, spill_context)?;

        timer.done();

        // Forwards output from the fully ordered stream that consumes the merged
        // spill runs.
        //
        // Not wrapping in a timer and not record output batches since this is now `merge_stream` responsibility
        // we just pass through
        while let Some(batch) = output_stream.next().await.transpose()? {
            emitter.emit(batch).await;
        }

        Ok(())
    }

    /// 1. Constructs a globally ordered input stream by applying a sort-preserving
    ///    merge to all spills.
    /// 2. Constructs a replay stream: a clustered final aggregate stream over the
    ///    fully ordered input constructed from the spills.
    ///
    /// Returns the replay stream
    fn switch_to_clustered_final_stream(
        &mut self,
        hash_table: AggregateHashTable<FinalMarker>,
        spill_context: Box<AggregateSpill>,
    ) -> Result<SendableRecordBatchStream> {
        let metrics = ClusteredAggregateTableMetrics::from_hash_table(&hash_table);
        drop(hash_table);
        self.reservation.try_resize(0)?;
        spill_context.into_replay_stream(
            &self.baseline_metrics,
            metrics,
            self.reservation.new_empty(),
        )
    }

    /// Emit final aggregate value batches:
    /// Input was exhausted without spilling, or the soft group limit was reached.
    async fn produce_output_from_memory(
        &mut self,
        mut hash_table: AggregateHashTable<FinalMarker>,
        spill_context: Option<Box<AggregateSpill>>,
        mut emitter: TryEmitter<RecordBatch, DataFusionError>,
    ) -> Result<()> {
        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();

        let mut timer = elapsed_compute.timer();
        hash_table.start_output()?;

        loop {
            let Some(batch) = hash_table.next_output_batch()? else {
                // Only reachable when the table held no groups at all: a
                // non-empty table always reports its last batch together with
                // the `Done` state, which the `try_resize` below already zeroes.
                self.reservation.try_resize(0)?;
                return Ok(());
            };

            // The table hands over its groups as they are materialized and
            // reports a size of 0 once it reaches `Done`, so this releases the
            // reservation before the final batch goes downstream.
            match self.reservation.try_resize(hash_table.memory_size()) {
                Ok(()) => {}
                // The materialized output is bigger than the table was. Write
                // it to disk and stream it back instead of failing the query.
                Err(e @ DataFusionError::ResourcesExhausted(_)) => {
                    let spill_context = spill_context.ok_or_else(|| e.context(
                        "Final hash aggregate cannot spill its output because temporary files are not enabled in the DiskManager",
                    ))?;
                    let remaining =
                        std::iter::from_fn(|| hash_table.next_output_batch().transpose());
                    let mut output = spill_context.spill_output(
                        Arc::clone(&self.schema),
                        std::iter::once(Ok(batch)).chain(remaining),
                    )?;
                    drop(hash_table);
                    self.reservation.try_resize(0)?;
                    timer.done();

                    while let Some(batch) = output.next().await.transpose()? {
                        emitter
                            .emit(batch.record_output(&self.baseline_metrics))
                            .await;
                    }
                    return Ok(());
                }
                Err(e) => return Err(e),
            }

            timer.done();
            emitter
                .emit(batch.record_output(&self.baseline_metrics))
                .await;
            timer = elapsed_compute.timer();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::aggregates::GroupClusteringMode;
    use crate::aggregates::clustered_test_util::collect_under_limit;
    use crate::aggregates::{AggregateMode, PhysicalGroupBy};
    use crate::common::collect;
    use crate::execution_plan::ExecutionPlan;
    use crate::test::TestMemoryExec;
    use crate::test::exec::BarrierExec;

    use arrow::array::{ArrayRef, AsArray, Int32Array, Int64Array, StringViewArray};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};
    use datafusion_common::Result;
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryPool, UnboundedMemoryPool,
    };
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_functions_aggregate::approx_distinct::approx_distinct_udaf;
    use datafusion_functions_aggregate::count::count_udaf;
    use datafusion_functions_aggregate::{min_max::min_udaf, sum::sum_udaf};
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::col;
    use futures::channel::mpsc;
    use futures::{FutureExt, StreamExt};
    use std::collections::BTreeMap;

    #[tokio::test]
    async fn test_partial_hash_stream_double_emission_race_condition_bug() -> Result<()> {
        // Fix for https://github.com/apache/datafusion/issues/18701
        // This test specifically proves that we have fixed double emission race condition
        // where emit_early_if_necessary() and switch_to_skip_aggregation()
        // both emit in the same loop iteration, causing data loss

        let schema = Arc::new(Schema::new(vec![
            Field::new("group_col", DataType::Int32, false),
            Field::new("value_col", DataType::Int64, false),
        ]));

        // Create data that will trigger BOTH conditions in the same iteration:
        // 1. More groups than batch_size (triggers early emission when memory pressure hits)
        // 2. High cardinality ratio (triggers skip aggregation)
        let batch_size = 1024; // We'll set this in session config
        let num_groups = batch_size + 100; // Slightly more than batch_size (1124 groups)

        // Create exactly 1 row per group = 100% cardinality ratio
        let group_ids: Vec<i32> = (0..num_groups as i32).collect();
        let values: Vec<i64> = vec![1; num_groups];

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(group_ids)),
                Arc::new(Int64Array::from(values)),
            ],
        )?;
        let input_partitions = vec![vec![batch]];

        // Create constrained memory to trigger early emission but not completely fail
        let runtime = RuntimeEnvBuilder::default()
            .with_memory_limit(5 * 1024, 1.0) // fits the initial table but will trigger pressure
            .build_arc()?;

        let mut task_ctx = TaskContext::default().with_runtime(runtime);

        // Configure to trigger BOTH conditions:
        // 1. Low probe threshold (triggers skip probe after few rows)
        // 2. Low ratio threshold (triggers skip aggregation immediately)
        // 3. Set batch_size to 1024 so our 1124 groups will trigger early emission
        // This creates the race condition where both emit paths are triggered
        let mut session_config = task_ctx.session_config().clone();
        session_config = session_config.set(
            "datafusion.execution.batch_size",
            &datafusion_common::ScalarValue::UInt64(Some(1024)),
        );
        session_config = session_config.set(
            "datafusion.execution.skip_partial_aggregation_probe_rows_threshold",
            &datafusion_common::ScalarValue::UInt64(Some(50)),
        );
        session_config = session_config.set(
            "datafusion.execution.skip_partial_aggregation_probe_ratio_threshold",
            &datafusion_common::ScalarValue::Float64(Some(0.8)),
        );
        task_ctx = task_ctx.with_session_config(session_config);
        let task_ctx = Arc::new(task_ctx);

        // Create aggregate: COUNT(*) GROUP BY group_col
        let group_expr = vec![(col("group_col", &schema)?, "group_col".to_string())];
        let aggr_expr = vec![Arc::new(
            AggregateExprBuilder::new(count_udaf(), vec![col("value_col", &schema)?])
                .schema(Arc::clone(&schema))
                .alias("count_value")
                .build()?,
        )];

        let exec = TestMemoryExec::try_new(&input_partitions, Arc::clone(&schema), None)?;
        let exec = Arc::new(TestMemoryExec::update_cache(&Arc::new(exec)));

        // Use Partial mode where the race condition occurs
        let aggregate_exec = AggregateExec::try_new(
            AggregateMode::Partial,
            PhysicalGroupBy::new_single(group_expr),
            aggr_expr,
            vec![None],
            exec,
            Arc::clone(&schema),
        )?;

        // Execute and collect results
        let mut stream =
            PartialHashAggregateStream::new(&aggregate_exec, &Arc::clone(&task_ctx), 0)?
                .into_stream();
        let mut results = Vec::new();

        while let Some(result) = stream.next().await {
            let batch = result?;
            results.push(batch);
        }

        // Count total groups emitted
        let mut total_output_groups = 0;
        for batch in &results {
            total_output_groups += batch.num_rows();
        }

        assert_eq!(
            total_output_groups, num_groups,
            "Unexpected number of groups",
        );
        assert_eq!(
            aggregate_exec
                .metrics()
                .unwrap()
                .sum_by_name("early_emit_count")
                .unwrap()
                .as_usize(),
            0
        );

        // Disable skip aggregation so the same input is emitted on memory pressure.
        let runtime = RuntimeEnvBuilder::default()
            .with_memory_limit(5 * 1024, 1.0)
            .build_arc()?;
        let session_config = task_ctx.session_config().clone().set(
            "datafusion.execution.skip_partial_aggregation_probe_ratio_threshold",
            &datafusion_common::ScalarValue::Float64(Some(2.0)),
        );
        let no_skip_task_ctx = Arc::new(
            TaskContext::default()
                .with_runtime(runtime)
                .with_session_config(session_config),
        );
        let mut stream =
            PartialHashAggregateStream::new(&aggregate_exec, &no_skip_task_ctx, 0)?
                .into_stream();
        while let Some(result) = stream.next().await {
            result?;
        }

        assert_eq!(
            aggregate_exec
                .metrics()
                .unwrap()
                .sum_by_name("early_emit_count")
                .unwrap()
                .as_usize(),
            1
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partial_hash_stream_skip_aggregation_probe_not_locked_until_skip()
    -> Result<()> {
        // Test that the probe is not locked until we actually decide to skip.
        // This allows us to continue evaluating the skip condition across multiple batches.
        //
        // Scenario:
        // - Batch 1: Hits rows threshold but NOT ratio threshold (low cardinality) -> don't skip
        // - Batch 2: Now hits ratio threshold (high cardinality) -> skip
        //
        // Without the fix, the probe would be locked after batch 1, preventing the skip
        // decision from being made on batch 2.

        let schema = Arc::new(Schema::new(vec![
            Field::new("group_col", DataType::Int32, false),
            Field::new("value_col", DataType::Int32, false),
        ]));

        // Configure thresholds:
        // - probe_rows_threshold: 100 rows
        // - probe_ratio_threshold: 0.8 (80%)
        let probe_rows_threshold = 100;
        let probe_ratio_threshold = 0.8;

        // Batch 1: 100 rows with only 10 unique groups
        // Ratio: 10/100 = 0.1 (10%) < 0.8 -> should NOT skip
        // This will hit the rows threshold but not the ratio threshold
        let batch1_rows = 100;
        let batch1_groups = 10;
        let mut group_ids_batch1 = Vec::new();
        for i in 0..batch1_rows {
            group_ids_batch1.push((i % batch1_groups) as i32);
        }
        let values_batch1: Vec<i32> = vec![1; batch1_rows];

        let batch1 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(group_ids_batch1)),
                Arc::new(Int32Array::from(values_batch1)),
            ],
        )?;

        // Batch 2: 360 rows with 360 unique NEW groups (starting from group 10)
        // After batch 2, total: 460 rows, 370 groups
        // Ratio: 370/460 is about 0.804 (80.4%) > 0.8 -> SHOULD decide to skip
        let batch2_rows = 360;
        let batch2_groups = 360;
        let group_ids_batch2: Vec<i32> = (batch1_groups..(batch1_groups + batch2_groups))
            .map(|x| x as i32)
            .collect();
        let values_batch2: Vec<i32> = vec![1; batch2_rows];

        let batch2 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(group_ids_batch2)),
                Arc::new(Int32Array::from(values_batch2)),
            ],
        )?;

        // Batch 3: This batch should be skipped since we decided to skip after batch 2
        // 100 rows with 100 unique groups (continuing from where batch 2 left off)
        let batch3_rows = 100;
        let batch3_groups = 100;
        let batch3_start_group = batch1_groups + batch2_groups;
        let group_ids_batch3: Vec<i32> = (batch3_start_group
            ..(batch3_start_group + batch3_groups))
            .map(|x| x as i32)
            .collect();
        let values_batch3: Vec<i32> = vec![1; batch3_rows];

        let batch3 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(group_ids_batch3)),
                Arc::new(Int32Array::from(values_batch3)),
            ],
        )?;

        let input_partitions = vec![vec![batch1, batch2, batch3]];

        let runtime = RuntimeEnvBuilder::default().build_arc()?;
        let mut task_ctx = TaskContext::default().with_runtime(runtime);

        // Configure skip aggregation settings
        let mut session_config = task_ctx.session_config().clone();
        session_config = session_config.set(
            "datafusion.execution.skip_partial_aggregation_probe_rows_threshold",
            &datafusion_common::ScalarValue::UInt64(Some(probe_rows_threshold)),
        );
        session_config = session_config.set(
            "datafusion.execution.skip_partial_aggregation_probe_ratio_threshold",
            &datafusion_common::ScalarValue::Float64(Some(probe_ratio_threshold)),
        );
        task_ctx = task_ctx.with_session_config(session_config);
        let task_ctx = Arc::new(task_ctx);

        // Create aggregate: COUNT(*) GROUP BY group_col
        let group_expr = vec![(col("group_col", &schema)?, "group_col".to_string())];
        let aggr_expr = vec![Arc::new(
            AggregateExprBuilder::new(count_udaf(), vec![col("value_col", &schema)?])
                .schema(Arc::clone(&schema))
                .alias("count_value")
                .build()?,
        )];

        let exec = TestMemoryExec::try_new(&input_partitions, Arc::clone(&schema), None)?;
        let exec = Arc::new(TestMemoryExec::update_cache(&Arc::new(exec)));

        // Use Partial mode
        let aggregate_exec = AggregateExec::try_new(
            AggregateMode::Partial,
            PhysicalGroupBy::new_single(group_expr),
            aggr_expr,
            vec![None],
            exec,
            Arc::clone(&schema),
        )?;

        // Execute and collect results
        let mut stream =
            PartialHashAggregateStream::new(&aggregate_exec, &Arc::clone(&task_ctx), 0)?
                .into_stream();
        let mut results = Vec::new();

        while let Some(result) = stream.next().await {
            let batch = result?;
            results.push(batch);
        }

        // Check that skip aggregation actually happened.
        // The key metric is skipped_aggregation_rows.
        let metrics = aggregate_exec.metrics().unwrap();
        let skipped_rows = metrics
            .sum_by_name("skipped_aggregation_rows")
            .map(|m| m.as_usize())
            .unwrap_or(0);

        // We expect batch 3's rows to be skipped (100 rows)
        assert_eq!(
            skipped_rows, batch3_rows,
            "Expected batch 3's rows ({batch3_rows}) to be skipped",
        );

        Ok(())
    }

    /// Builds a partial hash aggregate stream over a single input batch of
    /// `num_groups` distinct groups, running under `memory_limit` bytes.
    ///
    /// The input does not signal end-of-stream until `wait_finish` is called
    /// on the returned [`BarrierExec`], so any output produced before that can
    /// only come from the memory pressure emission path (normal output waits
    /// for all input). Skip partial aggregation is disabled for the same reason.
    fn partial_stream_under_memory_limit(
        memory_limit: usize,
        batch_size: usize,
        num_groups: usize,
    ) -> Result<(
        SendableRecordBatchStream,
        Arc<BarrierExec>,
        Arc<datafusion_execution::runtime_env::RuntimeEnv>,
    )> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("group_col", DataType::Int32, false),
            Field::new("value_col", DataType::Int64, false),
        ]));

        let group_ids: Vec<i32> = (0..num_groups as i32).collect();
        let values: Vec<i64> = vec![1; num_groups];

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(group_ids)),
                Arc::new(Int64Array::from(values)),
            ],
        )?;
        let input_partitions = vec![vec![batch]];

        let runtime = RuntimeEnvBuilder::default()
            .with_memory_limit(memory_limit, 1.0)
            .build_arc()?;

        let mut task_ctx = TaskContext::default().with_runtime(Arc::clone(&runtime));
        let session_config = task_ctx
            .session_config()
            .clone()
            .set(
                "datafusion.execution.batch_size",
                &datafusion_common::ScalarValue::UInt64(Some(batch_size as u64)),
            )
            .set(
                "datafusion.execution.skip_partial_aggregation_probe_ratio_threshold",
                &datafusion_common::ScalarValue::Float64(Some(1.0)),
            );
        task_ctx = task_ctx.with_session_config(session_config);
        let task_ctx = Arc::new(task_ctx);

        // Create aggregate: COUNT(*) GROUP BY group_col
        let group_expr = vec![(col("group_col", &schema)?, "group_col".to_string())];
        let aggr_expr = vec![Arc::new(
            AggregateExprBuilder::new(count_udaf(), vec![col("value_col", &schema)?])
                .schema(Arc::clone(&schema))
                .alias("count_value")
                .build()?,
        )];

        let input = Arc::new(
            BarrierExec::new(input_partitions, Arc::clone(&schema))
                .without_start_barrier()
                .with_finish_barrier()
                .with_log(false),
        );

        let aggregate_exec = AggregateExec::try_new(
            AggregateMode::Partial,
            PhysicalGroupBy::new_single(group_expr),
            aggr_expr,
            vec![None],
            Arc::clone(&input) as Arc<dyn ExecutionPlan>,
            Arc::clone(&schema),
        )?;

        let stream =
            PartialHashAggregateStream::new(&aggregate_exec, &task_ctx, 0)?.into_stream();

        Ok((stream, input, runtime))
    }

    #[tokio::test]
    async fn test_partial_hash_stream_accounts_held_batch_on_memory_pressure_while_slicing()
    -> Result<()> {
        let batch_size = 1024;
        let num_groups = 4 * batch_size;
        let memory_limit = 100 * 1024;
        let (mut stream, input, runtime) =
            partial_stream_under_memory_limit(memory_limit, batch_size, num_groups)?;

        let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("early emit should occur")
            .expect("stream ended early")?;
        assert_eq!(first.num_rows(), batch_size);

        let held_size = first.get_array_memory_size();
        assert!(runtime.memory_pool.reserved() >= held_size);

        let second = stream.next().await.expect("stream ended early")?;
        assert_eq!(second.num_rows(), batch_size);
        assert_eq!(
            first
                .column(0)
                .as_primitive::<Int32Type>()
                .values()
                .inner()
                .data_ptr(),
            second
                .column(0)
                .as_primitive::<Int32Type>()
                .values()
                .inner()
                .data_ptr(),
            "batches should slice one materialized state batch"
        );

        input.wait_finish().await;
        let mut total_rows = first.num_rows() + second.num_rows();
        while let Some(batch) = stream.next().await {
            total_rows += batch?.num_rows();
        }
        assert_eq!(total_rows, num_groups);
        Ok(())
    }

    #[tokio::test]
    async fn test_partial_hash_stream_emits_whole_batch_when_held_batch_does_not_fit()
    -> Result<()> {
        let batch_size = 1024;
        let num_groups = 4 * batch_size;
        let memory_limit = 32 * 1024;
        let (mut stream, input, runtime) =
            partial_stream_under_memory_limit(memory_limit, batch_size, num_groups)?;

        let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("early emit should occur")
            .expect("stream ended early")?;
        assert_eq!(first.num_rows(), num_groups);
        assert!(first.get_array_memory_size() > memory_limit);
        assert!(runtime.memory_pool.reserved() < first.get_array_memory_size());

        input.wait_finish().await;
        let mut total_rows = first.num_rows();
        while let Some(batch) = stream.next().await {
            total_rows += batch?.num_rows();
        }
        assert_eq!(total_rows, num_groups);
        Ok(())
    }

    #[tokio::test]
    async fn test_partial_hash_stream_releases_held_batch_after_last_slice() -> Result<()>
    {
        let batch_size = 1024;
        let num_slices = 4;
        let num_groups = num_slices * batch_size;
        let memory_limit = 100 * 1024;
        let (mut stream, input, runtime) =
            partial_stream_under_memory_limit(memory_limit, batch_size, num_groups)?;

        let mut held_size = 0;
        for slice_idx in 0..num_slices {
            let slice = if slice_idx == 0 {
                tokio::time::timeout(Duration::from_secs(5), stream.next())
                    .await
                    .expect("early emit should occur")
                    .expect("stream ended early")?
            } else {
                stream.next().await.expect("stream ended early")?
            };
            assert_eq!(slice.num_rows(), batch_size);
            held_size = slice.get_array_memory_size();
            if slice_idx + 1 < num_slices {
                assert!(runtime.memory_pool.reserved() >= held_size);
            } else {
                assert!(runtime.memory_pool.reserved() < held_size);
            }
        }
        assert!(held_size > 0);

        input.wait_finish().await;
        let mut total_rows = num_groups;
        while let Some(batch) = stream.next().await {
            total_rows += batch?.num_rows();
        }
        assert_eq!(total_rows, num_groups);
        Ok(())
    }

    #[derive(Clone, Copy)]
    enum Finish {
        Collect,
        DropDuringReplay,
        InputError,
    }

    #[tokio::test]
    async fn final_hash_spill_replay_with_other_partitions_holding_state() -> Result<()> {
        for spills in [1, 2, 3] {
            run_shared_pool_case(spills, Some(1024 * 1024), Finish::Collect).await?;
        }
        // Collision-mode scratch can exceed a fixed reference budget. Use an
        // unbounded pool so this case always exercises output without spilling.
        run_shared_pool_case(0, None, Finish::Collect).await
    }

    #[tokio::test]
    async fn final_hash_spill_replay_releases_memory_on_drop() -> Result<()> {
        run_shared_pool_case(1, Some(1024 * 1024), Finish::DropDuringReplay).await
    }

    #[tokio::test]
    async fn final_hash_spill_releases_memory_on_input_error() -> Result<()> {
        run_shared_pool_case(1, Some(1024 * 1024), Finish::InputError).await
    }

    /// Partition 0 spills `spills` times and replays while partitions 1..3
    /// keep their aggregate state in the same greedy pool. With `spills` at 0,
    /// partition 0 reads a fixed input that fits in memory.
    ///
    /// Input sizes follow the observed reservations, so the cases do not
    /// depend on the memory accounting of a platform or feature set.
    ///
    /// These cases cover the spill and replay lifecycle in a shared pool:
    /// results, spill metrics, and memory release. Case G in
    /// `aggregate_memory_spill.slt` covers the merge fan-in regression for
    /// issue #25423.
    async fn run_shared_pool_case(
        spills: usize,
        limit: Option<usize>,
        finish: Finish,
    ) -> Result<()> {
        const PARTITIONS: usize = 4;
        /// Partitions 1..3 hold at least this much state.
        const HELD_BYTES: usize = 512 * 1024;
        /// A case that never reaches its target fails instead of looping.
        const MAX_BATCHES: i64 = 1000;

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
            Field::new("s", DataType::Utf8View, false),
        ]));
        let groups = PhysicalGroupBy::new_single(vec![
            (col("a", &schema)?, "a".into()),
            (col("b", &schema)?, "b".into()),
        ]);
        let expressions = vec![
            Arc::new(
                AggregateExprBuilder::new(sum_udaf(), vec![col("v", &schema)?])
                    .schema(Arc::clone(&schema))
                    .alias("sum")
                    .build()?,
            ),
            Arc::new(
                AggregateExprBuilder::new(min_udaf(), vec![col("s", &schema)?])
                    .schema(Arc::clone(&schema))
                    .alias("min")
                    .build()?,
            ),
        ];
        let empty = TestMemoryExec::try_new_exec(&[vec![]], Arc::clone(&schema), None)?;
        let partial = AggregateExec::try_new(
            AggregateMode::Partial,
            groups.clone(),
            expressions.clone(),
            vec![None; 2],
            empty,
            Arc::clone(&schema),
        )?;
        let partial_schema = partial.schema();
        let input = TestMemoryExec::try_new_exec(
            &vec![vec![]; PARTITIONS],
            Arc::clone(&partial_schema),
            None,
        )?;
        let aggregate = AggregateExec::try_new(
            AggregateMode::FinalPartitioned,
            groups.as_final(),
            expressions,
            vec![None; 2],
            input,
            schema,
        )?;
        assert_eq!(
            aggregate.group_clustering_mode(),
            &GroupClusteringMode::None
        );

        let pool: Arc<dyn MemoryPool> = match limit {
            Some(limit) => Arc::new(GreedyMemoryPool::new(limit)),
            None => Arc::new(UnboundedMemoryPool::default()),
        };
        let context = Arc::new(
            TaskContext::default()
                .with_session_config(SessionConfig::new().with_batch_size(128))
                .with_runtime(
                    RuntimeEnvBuilder::new()
                        .with_memory_pool(Arc::clone(&pool))
                        .build_arc()?,
                ),
        );
        let make_stream = |partition| -> Result<_> {
            let (sender, receiver) = mpsc::unbounded();
            let input = Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&partial_schema),
                receiver,
            ));
            let stream = FinalHashAggregateStream::new_with_input(
                &aggregate, &context, partition, input,
            )?
            .into_stream();
            Ok((sender, stream))
        };

        // Construct the holders first. Their reservation is the baseline that
        // must survive partition 0's spill/replay lifecycle.
        let mut streams = vec![];
        let mut senders = vec![];
        for partition in 1..PARTITIONS {
            let (sender, stream) = make_stream(partition)?;
            senders.push(sender);
            streams.push(stream);
        }
        let mut expected = BTreeMap::new();
        let mut make_batch = |partition: i64, start: i64| {
            for value in start..start + 128 {
                let entry = expected
                    .entry((
                        partition,
                        if partition == 0 && value % 128 == 0 {
                            0
                        } else {
                            value
                        },
                    ))
                    .or_insert_with(|| (0, (value % 2).to_string()));
                entry.0 += value * 2;
            }
            RecordBatch::try_new(
                Arc::clone(&partial_schema),
                vec![
                    Arc::new(Int64Array::from(vec![partition; 128])),
                    Arc::new(Int64Array::from_iter_values((start..start + 128).map(
                        |value| {
                            if partition == 0 && value % 128 == 0 {
                                0
                            } else {
                                value
                            }
                        },
                    ))),
                    Arc::new(Int64Array::from_iter_values(
                        (start..start + 128).map(|v| v * 2),
                    )),
                    Arc::new(StringViewArray::from_iter_values(
                        (start..start + 128).map(|v| if v % 2 == 0 { "0" } else { "1" }),
                    )),
                ],
            )
            .unwrap()
        };
        // Channel inputs return Pending after each supplied batch, so the
        // interleaving below does not depend on task scheduling.
        let mut held_batches = 0;
        while pool.reserved() < HELD_BYTES {
            assert!(held_batches < MAX_BATCHES, "held state stays small");
            for (partition, (sender, stream)) in
                senders.iter().zip(streams.iter_mut()).enumerate()
            {
                let partition = partition + 1;
                sender
                    .unbounded_send(Ok(make_batch(partition as i64, held_batches * 128)))
                    .unwrap();
                assert!(stream.next().now_or_never().is_none());
            }
            held_batches += 1;
        }
        let held = pool.reserved();
        let spill_count = || aggregate.metrics().unwrap().spill_count().unwrap();
        assert_eq!(spill_count(), 0);

        // Do not include partition 0's empty table in `held`: it is correctly
        // released when that stream spills, errors, or is dropped.
        let (partition_zero_sender, mut first) = make_stream(0)?;
        // Feed partition 0 until it has spilled `spills` times. Key (0, 0)
        // repeats in every batch, so replay must merge its sum across runs.
        let mut batches = 0;
        loop {
            let done = if spills == 0 {
                batches == 70
            } else {
                spill_count() >= spills
            };
            if done {
                break;
            }
            assert!(batches < MAX_BATCHES, "partition 0 did not spill");
            partition_zero_sender
                .unbounded_send(Ok(make_batch(0, batches * 128)))
                .unwrap();
            assert!(first.next().now_or_never().is_none());
            batches += 1;
        }
        // Add groups after the last spill so replay also merges the final
        // in-memory run.
        for _ in 0..8 {
            partition_zero_sender
                .unbounded_send(Ok(make_batch(0, batches * 128)))
                .unwrap();
            assert!(first.next().now_or_never().is_none());
            batches += 1;
        }
        assert!(spill_count() >= spills);
        assert_eq!(spill_count() == 0, spills == 0);
        match finish {
            Finish::Collect => {
                partition_zero_sender.close_channel();
                let mut output = collect(first).await?;
                assert_eq!(pool.reserved(), held);
                for sender in &senders {
                    sender.close_channel();
                }
                for stream in streams.drain(..) {
                    output.extend(collect(stream).await?);
                }
                let mut actual = BTreeMap::new();
                for batch in output {
                    let a = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    let b = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    let sum = batch
                        .column(2)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    let min = batch
                        .column(3)
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap();
                    for row in 0..batch.num_rows() {
                        assert!(
                            actual
                                .insert(
                                    (a.value(row), b.value(row)),
                                    (sum.value(row), min.value(row).to_string())
                                )
                                .is_none()
                        );
                    }
                }
                assert_eq!(actual, expected);
                assert_eq!(spill_count() == 0, spills == 0);
            }
            Finish::DropDuringReplay => {
                partition_zero_sender.close_channel();
                first.next().await.unwrap()?;
                assert!(pool.reserved() > held);
                drop(first);
                assert_eq!(pool.reserved(), held);
            }
            Finish::InputError => {
                partition_zero_sender
                    .unbounded_send(datafusion_common::exec_err!(
                        "injected input failure"
                    ))
                    .unwrap();
                let error = first.next().await.unwrap().unwrap_err();
                assert!(error.to_string().contains("injected input failure"));
                assert_eq!(pool.reserved(), held);
                drop(first);
            }
        }
        drop(streams);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    /// Every row is a distinct group, so under memory pressure the stream
    /// emits every state. After the emit, `APPROX_DISTINCT` must release the
    /// hashes of the last batch (64 KB for 8192 rows), or reserving the empty
    /// table fails the limit.
    #[tokio::test]
    async fn emitted_states_release_accumulator_scratch() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let batches = [0..8192, 8192..16384, 16384..24576]
            .into_iter()
            .map(|range| {
                let values: ArrayRef = Arc::new(Int64Array::from_iter_values(range));
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::clone(&values), values],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input = TestMemoryExec::try_new_exec(&[batches], Arc::clone(&schema), None)?;
        let aggregate = AggregateExec::try_new(
            AggregateMode::Partial,
            PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
            vec![Arc::new(
                AggregateExprBuilder::new(
                    approx_distinct_udaf(),
                    vec![col("v", &schema)?],
                )
                .schema(Arc::clone(&schema))
                .alias("approx_distinct")
                .build()?,
            )],
            vec![None],
            input,
            schema,
        )?;
        assert_eq!(
            aggregate.group_clustering_mode(),
            &GroupClusteringMode::None
        );
        let output = collect_under_limit(&aggregate, 64 * 1024).await?;
        assert_eq!(output.iter().map(|b| b.num_rows()).sum::<usize>(), 24576);
        assert!(
            aggregate
                .metrics()
                .unwrap()
                .sum_by_name("early_emit_count")
                .unwrap()
                .as_usize()
                > 0
        );
        Ok(())
    }
    mod bounded_partial {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use arrow::array::AsArray;
        use arrow::compute::concat_batches;
        use arrow::datatypes::Int64Type;
        use datafusion_execution::config::SessionConfig;
        use datafusion_execution::memory_pool::{
            FairSpillPool, MemoryPool, MemoryReservation, UnboundedMemoryPool,
        };
        use datafusion_functions_aggregate::sum::sum_udaf;

        use super::*;
        use crate::aggregates::bounded_partial::test_option::with_groups;
        use crate::common::collect;

        const BATCH_ROWS: usize = 256;

        /// Records the peak reservation of an unbounded pool.
        #[derive(Debug, Default)]
        struct PeakPool {
            inner: UnboundedMemoryPool,
            peak: AtomicUsize,
        }

        impl PeakPool {
            fn observe(&self) {
                self.peak
                    .fetch_max(self.inner.reserved(), Ordering::Relaxed);
            }
        }

        impl std::fmt::Display for PeakPool {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "peak")
            }
        }

        impl MemoryPool for PeakPool {
            fn name(&self) -> &str {
                "peak"
            }
            fn grow(&self, reservation: &MemoryReservation, additional: usize) {
                self.inner.grow(reservation, additional);
                self.observe();
            }
            fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
                self.inner.shrink(reservation, shrink);
            }
            fn try_grow(
                &self,
                reservation: &MemoryReservation,
                additional: usize,
            ) -> Result<()> {
                self.inner.try_grow(reservation, additional)?;
                self.observe();
                Ok(())
            }
            fn reserved(&self) -> usize {
                self.inner.reserved()
            }
        }

        fn schema() -> SchemaRef {
            Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v", DataType::Int64, false),
            ]))
        }

        /// `rows` rows in `BATCH_ROWS` batches with key `key(row)` and value `row`.
        fn input(rows: usize, key: impl Fn(usize) -> i64) -> Vec<RecordBatch> {
            (0..rows)
                .step_by(BATCH_ROWS)
                .map(|start| {
                    let end = (start + BATCH_ROWS).min(rows);
                    RecordBatch::try_new(
                        schema(),
                        vec![
                            Arc::new(Int64Array::from_iter_values(
                                (start..end).map(&key),
                            )),
                            Arc::new(Int64Array::from_iter_values(
                                (start..end).map(|row| row as i64),
                            )),
                        ],
                    )
                    .unwrap()
                })
                .collect()
        }

        fn clustered(row: usize) -> i64 {
            (row / 4) as i64
        }

        /// Uniform draws with replacement from `domain` keys (splitmix64 of the row).
        fn scattered(domain: usize) -> impl Fn(usize) -> i64 {
            move |row| {
                let mut x = (row as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
                x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                ((x ^ (x >> 31)) % domain as u64) as i64
            }
        }

        fn partial(batches: Vec<RecordBatch>) -> Result<Arc<AggregateExec>> {
            let schema = schema();
            let exec = TestMemoryExec::try_new(&[batches], Arc::clone(&schema), None)?;
            let exec = Arc::new(TestMemoryExec::update_cache(&Arc::new(exec)));
            let aggr_expr = vec![Arc::new(
                AggregateExprBuilder::new(sum_udaf(), vec![col("v", &schema)?])
                    .schema(Arc::clone(&schema))
                    .alias("sum_v")
                    .build()?,
            )];
            Ok(Arc::new(AggregateExec::try_new(
                AggregateMode::Partial,
                PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".into())]),
                aggr_expr,
                vec![None],
                exec,
                schema,
            )?))
        }

        fn final_over(partial: Arc<AggregateExec>) -> Result<Arc<AggregateExec>> {
            Ok(Arc::new(AggregateExec::try_new(
                AggregateMode::Final,
                partial.group_by().as_final(),
                partial.aggr_expr().to_vec(),
                vec![None],
                Arc::clone(&partial) as _,
                partial.input_schema(),
            )?))
        }

        fn context(groups: usize, pool: Arc<dyn MemoryPool>) -> Result<Arc<TaskContext>> {
            let runtime = RuntimeEnvBuilder::default()
                .with_memory_pool(pool)
                .build_arc()?;
            let config =
                with_groups(SessionConfig::new().with_batch_size(BATCH_ROWS), groups);
            Ok(Arc::new(
                TaskContext::default()
                    .with_runtime(runtime)
                    .with_session_config(config),
            ))
        }

        struct Run {
            /// Final `(k, sum_v)` sorted by key.
            rows: Vec<(i64, i64)>,
            partial_rows: usize,
            flushes: Option<usize>,
        }

        async fn run(batches: Vec<RecordBatch>, groups: usize) -> Result<Run> {
            let pool = Arc::new(PeakPool::default());
            let partial = partial(batches)?;
            let plan = final_over(Arc::clone(&partial))?;
            let output = collect(
                plan.execute(0, context(groups, Arc::<PeakPool>::clone(&pool))?)?,
            )
            .await?;
            Ok(Run {
                rows: sorted_rows(&output),
                partial_rows: partial.metrics().unwrap().output_rows().unwrap(),
                flushes: flushes(&partial),
            })
        }

        fn sorted_rows(output: &[RecordBatch]) -> Vec<(i64, i64)> {
            let batch = concat_batches(&output[0].schema(), output).unwrap();
            let keys = batch.column(0).as_primitive::<Int64Type>();
            let sums = batch.column(1).as_primitive::<Int64Type>();
            let mut rows: Vec<_> = (0..batch.num_rows())
                .map(|row| (keys.value(row), sums.value(row)))
                .collect();
            rows.sort_unstable();
            rows
        }

        fn flushes(partial: &AggregateExec) -> Option<usize> {
            partial
                .metrics()
                .unwrap()
                .sum_by_name("partial_flushes")
                .map(|count| count.as_usize())
        }

        #[tokio::test]
        async fn clustered_input_flushes_with_identical_results() -> Result<()> {
            let rows = 40_000;
            let native = run(input(rows, clustered), 0).await?;
            let bounded = run(input(rows, clustered), 1_000).await?;
            assert_eq!(native.rows.len(), rows / 4);
            assert_eq!(bounded.rows, native.rows);
            assert_eq!(native.flushes, None);
            let flushes = bounded.flushes.unwrap();
            assert!(flushes >= 9, "{flushes}");
            // A flush splits at most the group that straddles it.
            assert_eq!(native.partial_rows, rows / 4);
            assert!(bounded.partial_rows <= rows / 4 + flushes);
            Ok(())
        }

        /// Isolate the partial operator's reservation: a final table grows
        /// with the entire key domain and masks the bound being tested here.
        async fn partial_peak(rows: usize, groups: usize) -> Result<usize> {
            let pool = Arc::new(PeakPool::default());
            let plan = partial(input(rows, clustered))?;
            let _output = collect(
                plan.execute(0, context(groups, Arc::<PeakPool>::clone(&pool))?)?,
            )
            .await?;
            assert_eq!(pool.reserved(), 0);
            Ok(pool.peak.load(Ordering::Relaxed))
        }

        #[tokio::test]
        async fn bounded_table_keeps_memory_bounded() -> Result<()> {
            let native = partial_peak(400_000, 0).await?;
            let small = partial_peak(40_000, 1_000).await?;
            let large = partial_peak(400_000, 1_000).await?;
            assert!(large * 4 < native, "bounded {large} native {native}");
            assert!(
                large <= small * 2,
                "10x input: small {small}, large {large}"
            );
            Ok(())
        }

        #[tokio::test]
        async fn scattered_keys_revert_to_the_native_table() -> Result<()> {
            let rows = 50_000;
            let native = run(input(rows, scattered(5_000)), 0).await?;
            let bounded = run(input(rows, scattered(5_000)), 1_000).await?;
            assert_eq!(bounded.rows, native.rows);
            assert_eq!(bounded.flushes, None);
            assert_eq!(bounded.partial_rows, 5_000);
            assert_eq!(native.partial_rows, 5_000);
            Ok(())
        }

        #[tokio::test]
        async fn repeated_keys_across_flushes_revert_after_two_flushes() -> Result<()> {
            let rows = 50_000;
            let native = run(input(rows, scattered(1_100)), 0).await?;
            let bounded = run(input(rows, scattered(1_100)), 1_000).await?;
            assert_eq!(bounded.rows, native.rows);
            assert_eq!(bounded.flushes, Some(2));
            assert!(bounded.partial_rows <= 2 * (1_000 + BATCH_ROWS) + 1_100);
            Ok(())
        }

        #[tokio::test]
        async fn final_spill_merges_bounded_flushes() -> Result<()> {
            let rows = 40_000;
            let native = run(input(rows, clustered), 0).await?;
            let partial = partial(input(rows, clustered))?;
            let plan = final_over(Arc::clone(&partial))?;
            let pool: Arc<dyn MemoryPool> = Arc::new(FairSpillPool::new(256 * 1024));
            let output = collect(plan.execute(0, context(1_000, pool)?)?).await?;
            assert_eq!(sorted_rows(&output), native.rows);
            assert!(flushes(&partial).unwrap() > 0);
            assert!(plan.metrics().unwrap().spill_count().unwrap() > 0);
            Ok(())
        }
        #[tokio::test]
        async fn bounded_partial_resets_nullable_and_variable_keys() -> Result<()> {
            use arrow::array::{ListArray, StringViewArray};
            use datafusion_common::test_util::batches_to_sort_string;
            for kind in 0..4 {
                let group_type = match kind {
                    0 | 2 => DataType::Int64,
                    1 => DataType::Utf8View,
                    _ => DataType::List(Arc::new(Field::new_list_field(
                        DataType::Int64,
                        true,
                    ))),
                };
                let mut fields = vec![Field::new("k", group_type, true)];
                if kind == 2 {
                    fields.push(Field::new("s", DataType::Utf8View, true));
                }
                fields.push(Field::new("v", DataType::Int64, false));
                let schema = Arc::new(Schema::new(fields));
                let mut batches = vec![];
                for start in (0..16_384).step_by(BATCH_ROWS) {
                    let keys: Vec<Option<i64>> = (start..start + BATCH_ROWS)
                        .map(|row| {
                            let k = row as i64 / 4;
                            (k % 101 != 0).then_some(k)
                        })
                        .collect();
                    let strings = || {
                        Arc::new(StringViewArray::from_iter(keys.iter().map(|key| {
                            key.map(|k| format!("long nullable grouping key {k}"))
                        }))) as ArrayRef
                    };
                    let key: ArrayRef = match kind {
                        0 | 2 => Arc::new(Int64Array::from(keys.clone())),
                        1 => strings(),
                        _ => Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
                            keys.iter().map(|key| key.map(|k| vec![Some(k), None])),
                        )),
                    };
                    let mut columns = vec![key];
                    if kind == 2 {
                        columns.push(strings());
                    }
                    columns.push(Arc::new(Int64Array::from_iter_values(
                        (start..start + BATCH_ROWS).map(|row| row as i64),
                    )));
                    batches.push(RecordBatch::try_new(Arc::clone(&schema), columns)?);
                }
                let execute = |bound| -> Result<(
                    Arc<AggregateExec>,
                    Arc<AggregateExec>,
                    Arc<TaskContext>,
                )> {
                    let group = PhysicalGroupBy::new_single(
                        (0..schema.fields().len() - 1)
                            .map(|i| {
                                Ok((
                                    col(schema.field(i).name(), &schema)?,
                                    schema.field(i).name().clone(),
                                ))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    );
                    let plan = Arc::new(AggregateExec::try_new(
                        AggregateMode::Partial,
                        group,
                        vec![Arc::new(
                            AggregateExprBuilder::new(
                                sum_udaf(),
                                vec![col("v", &schema)?],
                            )
                            .schema(Arc::clone(&schema))
                            .alias("sum_v")
                            .build()?,
                        )],
                        vec![None],
                        TestMemoryExec::try_new_exec(
                            &[batches.clone()],
                            Arc::clone(&schema),
                            None,
                        )?,
                        Arc::clone(&schema),
                    )?);
                    Ok((
                        Arc::clone(&plan),
                        final_over(plan)?,
                        context(bound, Arc::new(UnboundedMemoryPool::default()))?,
                    ))
                };
                let (_, native, ctx) = execute(0)?;
                let expected = collect(native.execute(0, ctx)?).await?;
                let (partial, bounded, ctx) = execute(256)?;
                let actual = collect(bounded.execute(0, ctx)?).await?;
                assert_eq!(
                    batches_to_sort_string(&actual),
                    batches_to_sort_string(&expected),
                    "key kind {kind}"
                );
                assert!(flushes(&partial).is_some_and(|n| n > 1), "key kind {kind}");
            }
            Ok(())
        }

        #[tokio::test]
        async fn bounded_partial_coexists_with_skip_probe() -> Result<()> {
            let data = input(40_000, |row| {
                if row < 16_384 {
                    clustered(row)
                } else {
                    row as i64 + 1_000_000
                }
            });
            let expected = run(data.clone(), 0).await?;
            let partial = partial(data)?;
            let plan = final_over(Arc::clone(&partial))?;
            let mut config =
                with_groups(SessionConfig::new().with_batch_size(BATCH_ROWS), 1_000);
            config
                .options_mut()
                .execution
                .skip_partial_aggregation_probe_rows_threshold = 1;
            config
                .options_mut()
                .execution
                .skip_partial_aggregation_probe_ratio_threshold = 0.5;
            let ctx = Arc::new(TaskContext::default().with_session_config(config));
            let actual = collect(plan.execute(0, ctx)?).await?;
            assert_eq!(sorted_rows(&actual), expected.rows);
            assert!(flushes(&partial).is_some_and(|n| n > 0));
            assert!(
                partial
                    .metrics()
                    .unwrap()
                    .sum_by_name("skipped_aggregation_rows")
                    .unwrap()
                    .as_usize()
                    > 0
            );
            Ok(())
        }

        #[tokio::test]
        async fn bounded_partial_keeps_upstream_pressure_emission() -> Result<()> {
            let data = input(40_000, clustered);
            let expected = run(data.clone(), 0).await?;
            let partial = partial(data)?;
            let pool: Arc<dyn MemoryPool> = Arc::new(FairSpillPool::new(128 * 1024));
            // The configured cache bound is larger than this pool. Ordinary
            // OOM emission must still make progress without reaching it.
            let states =
                collect(partial.execute(0, context(100_000, Arc::clone(&pool))?)?)
                    .await?;
            assert_eq!(pool.reserved(), 0);
            assert_eq!(flushes(&partial), None);
            assert!(
                partial
                    .metrics()
                    .unwrap()
                    .sum_by_name("early_emit_count")
                    .unwrap()
                    .as_usize()
                    > 0
            );
            let source = TestMemoryExec::try_new_exec(&[states], partial.schema(), None)?;
            let final_plan = AggregateExec::try_new(
                AggregateMode::Final,
                partial.group_by().as_final(),
                partial.aggr_expr().to_vec(),
                vec![None],
                source,
                partial.input_schema(),
            )?;
            let output =
                collect(final_plan.execute(0, Arc::new(TaskContext::default()))?).await?;
            assert_eq!(sorted_rows(&output), expected.rows);
            Ok(())
        }
        #[tokio::test]
        async fn bounded_partial_releases_policy_under_pool_pressure() -> Result<()> {
            let data = input(8192, clustered);
            let expected = run(data.clone(), 0).await?;
            let partial = partial(data)?;
            let limit = 1024 * 1024;
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
            let mut stream = partial.execute(0, context(256, Arc::clone(&pool))?)?;
            let first = stream.next().await.unwrap()?;
            assert_eq!(flushes(&partial), Some(1));
            let held = pool.reserved();
            assert!(
                held >= 4096 + BATCH_ROWS * size_of::<u64>(),
                "sketch must stay charged"
            );
            // Leave the running stream its existing charge plus 64 bytes.
            // The next table growth must switch to ordinary pressure emission
            // and release the optional sketch, without reopening raw input.
            let other = MemoryConsumer::new("other operator").register(&pool);
            other.try_grow(limit - held - 64)?;
            let mut states = vec![first];
            states.extend(collect(stream).await?);
            assert!(
                partial
                    .metrics()
                    .unwrap()
                    .sum_by_name("early_emit_count")
                    .unwrap()
                    .as_usize()
                    > 0
            );
            assert_eq!(pool.reserved(), other.size());
            drop(other);
            assert_eq!(pool.reserved(), 0);
            let source = TestMemoryExec::try_new_exec(&[states], partial.schema(), None)?;
            let final_plan = AggregateExec::try_new(
                AggregateMode::Final,
                partial.group_by().as_final(),
                partial.aggr_expr().to_vec(),
                vec![None],
                source,
                partial.input_schema(),
            )?;
            let output =
                collect(final_plan.execute(0, Arc::new(TaskContext::default()))?).await?;
            assert_eq!(sorted_rows(&output), expected.rows);
            Ok(())
        }
    }
}
