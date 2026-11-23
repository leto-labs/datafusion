// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Final aggregate stream for partial-state input with group-clustering
//! guarantees.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result, internal_err};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::{TaskContext, TryEmitter, async_try_stream};
use futures::stream::StreamExt;

use super::AggregateExec;
use super::aggregate_hash_table::{
    ClusteredAggregateTable, ClusteredAggregateTableMetrics, FinalMarker,
};
use super::order::GroupClusteringMode;
use super::spill::AggregateSpill;
use crate::SendableRecordBatchStream;
use crate::aggregates::AggregateMode;
use crate::metrics::{BaselineMetrics, SpillMetrics};
use crate::stream::{ObservedStream, RecordBatchStreamAdapter};

/// Final aggregate stream for [`GroupClusteringMode::Partial`] and
/// [`GroupClusteringMode::Full`].
///
/// See comments at [`super::clustered_partial_stream::ClusteredPartialAggregateStream`] for details.
///
/// # Spilling
///
/// This section is only for implementation notes, for background, see [`super::clustered_partial_stream::ClusteredPartialAggregateStream`]
///
/// For partial group clustering, spilling works as follows:
///
/// - Reserve the table footprint plus one `u32` sort index per buffered group. The
///   extra index array is used in later sorting before spilling.
/// - On memory pressure, materialize all group states into one batch.
/// - Use [`IncrementalSortIterator`](crate::sorts::IncrementalSortIterator) to compute the full-batch index, then
///   materialize and write one sorted slice at a time. The original batch and
///   full index remain live until the run is written.
/// - After input ends, merge the sorted runs and replay them through a fully
///   ordered final aggregate stream.
pub(crate) struct ClusteredFinalAggregateStream {
    reservation: MemoryReservation,
    context: ClusteredFinalAggregateContext,
    stage: ExecutionStage,
}

/// Execution stages described in [`ClusteredFinalAggregateStream::into_stream`].
enum ExecutionStage {
    Aggregating(Aggregating),
    Outputting(Outputting),
    MergingSpills(SendableRecordBatchStream),
}

struct Aggregating {
    input: SendableRecordBatchStream,
    table: ClusteredAggregateTable<FinalMarker>,
    /// None when temporary files are disabled or group clustering is full.
    spill_context: Option<Box<AggregateSpill>>,
}

struct Outputting {
    /// Materialized final results, emitted in slices of `batch_size` rows.
    batch: RecordBatch,
    /// Aggregation stage to resume after output; `None` after EOF.
    resume: Option<Aggregating>,
}

/// Immutable execution context shared by aggregation and output emission.
struct ClusteredFinalAggregateContext {
    schema: SchemaRef,
    batch_size: usize,
    baseline_metrics: BaselineMetrics,
    /// Set for spill replay, see [`HeldMemory::Charge`].
    charge_held_memory: bool,
}

/// How a fully clustered final stream without a spill path answers a refused
/// handoff, in which the pool refuses both the completed groups' batch and
/// the retained table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::aggregates) enum HeldMemory {
    /// Return the refusal as `ResourcesExhausted`.
    Refuse,
    /// Emit the completed groups and charge the memory the retained table
    /// already holds, without another admission. Spill replay uses this: it
    /// cannot spill, and failing would discard every spilled run. Under full
    /// clustering the table then retains only the active group, which began
    /// within the last batch, so the charge follows one group's size and does
    /// not accumulate. A refusal with no completed group, the initial table
    /// and any other growth still need admission.
    Charge,
}

impl ClusteredFinalAggregateStream {
    pub fn new(
        agg: &AggregateExec,
        context: &Arc<TaskContext>,
        partition: usize,
    ) -> Result<Self> {
        debug_assert!(matches!(
            agg.mode,
            AggregateMode::Final | AggregateMode::FinalPartitioned
        ));
        debug_assert_ne!(agg.group_clustering_mode, GroupClusteringMode::None);

        let input = agg.input.execute(partition, Arc::clone(context))?;
        Self::new_with_input(agg, context, partition, input, &agg.group_clustering_mode)
    }

    pub(in crate::aggregates) fn new_with_input(
        agg: &AggregateExec,
        context: &Arc<TaskContext>,
        partition: usize,
        input: SendableRecordBatchStream,
        group_clustering_mode: &GroupClusteringMode,
    ) -> Result<Self> {
        let baseline_metrics = BaselineMetrics::new(&agg.metrics, partition);
        let metrics = ClusteredAggregateTableMetrics::new(agg, partition);
        let spill_metrics = SpillMetrics::new(&agg.metrics, partition);
        let reservation =
            MemoryConsumer::new(format!("ClusteredFinalAggregateStream[{partition}]"))
                // HACK: Full group clustering uses a non-spilling execution
                // path. There is a known race
                // condition bug, and we set it to spillable to let it have larger
                // memory budget to suppress the bug.
                // Bug issue: https://github.com/apache/datafusion/issues/17334
                .with_can_spill(true)
                .register(context.memory_pool());
        Self::new_with_input_and_metrics(
            agg,
            context,
            partition,
            input,
            group_clustering_mode,
            baseline_metrics,
            metrics,
            Some(spill_metrics),
            reservation,
            HeldMemory::Refuse,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "keeps replay metric reuse explicit"
    )]
    /// Builds the stream with the reservation of its logical aggregate operator.
    /// Replay callers pass a sibling of the reservation used by the merge input,
    /// keeping both components under one memory-consumer registration, and
    /// [`HeldMemory::Charge`].
    pub(in crate::aggregates) fn new_with_input_and_metrics(
        agg: &AggregateExec,
        context: &Arc<TaskContext>,
        partition: usize,
        input: SendableRecordBatchStream,
        group_clustering_mode: &GroupClusteringMode,
        baseline_metrics: BaselineMetrics,
        metrics: ClusteredAggregateTableMetrics,
        spill_metrics: Option<SpillMetrics>,
        reservation: MemoryReservation,
        held_memory: HeldMemory,
    ) -> Result<Self> {
        debug_assert!(matches!(
            agg.mode,
            AggregateMode::Final | AggregateMode::FinalPartitioned
        ));
        debug_assert_ne!(*group_clustering_mode, GroupClusteringMode::None);

        let schema = Arc::clone(&agg.schema);
        let input_schema = input.schema();
        let batch_size = context.session_config().batch_size();

        let can_spill = matches!(group_clustering_mode, GroupClusteringMode::Partial(_))
            && context.runtime_env().disk_manager.tmp_files_enabled();
        let spill_context = if can_spill {
            let Some(spill_metrics) = spill_metrics else {
                return internal_err!(
                    "Spillable clustered final stream requires metrics"
                );
            };
            Some(Box::new(AggregateSpill::try_new(
                "ClusteredFinalAggregateSpill",
                agg,
                context,
                partition,
                batch_size,
                group_clustering_mode,
                &input_schema,
                spill_metrics,
            )?))
        } else {
            None
        };

        let table = ClusteredAggregateTable::<FinalMarker>::new_with_group_clustering(
            agg,
            &input_schema,
            Arc::clone(&schema),
            group_clustering_mode,
            metrics,
        )?;

        let charge_held_memory = held_memory == HeldMemory::Charge;
        if charge_held_memory && spill_context.is_some() {
            return internal_err!(
                "a spilling clustered final stream cannot charge held memory"
            );
        }
        if charge_held_memory && *group_clustering_mode != GroupClusteringMode::Full {
            return internal_err!("only full group clustering charges held memory");
        }

        // Reserve memory for the initial hash table. that we hold for the lifetime of the stream.
        // Replay is admitted here like any new table, so a hard memory limit
        // can still refuse it before it starts.
        reservation.try_grow(table.memory_size())?;

        Ok(Self {
            reservation,
            context: ClusteredFinalAggregateContext {
                schema,
                batch_size,
                baseline_metrics,
                charge_held_memory,
            },
            stage: ExecutionStage::Aggregating(Aggregating {
                input,
                table,
                spill_context,
            }),
        })
    }

    /// Entry point for the clustered final aggregate execution stages.
    ///
    /// See [`ClusteredFinalAggregateStream`] for high-level ideas.
    ///
    /// # Stage transition graph:
    ///
    /// ```text
    ///                  +----[2]----+                +----[5]----+
    ///                  |           |                |           |
    ///                  v           |                v           |
    ///              +-------------------+         +------------------+
    ///              |                   |         |                  |
    /// (start)-[1]->|    Aggregating    |---[3]-->|    Outputting    |
    ///              |                   |<--[6]---|                  |
    ///              +-------------------+         +------------------+
    ///                        |     |                      |
    ///                       [8]    +------[4]-----+      [7]
    ///                        |                    |       |
    ///                        v                    v       v
    ///              +-------------------+         +------------------+
    ///              |                   |         |                  |
    ///         +--->|   MergingSpills   |---[10-->|      Done        |-[11]->(end)
    ///         |    |                   |         |                  |
    ///         |    +-------------------+         +------------------+
    ///         |              |
    ///         +-----[9]------+
    /// ```
    ///
    /// ## Stages
    ///
    /// - [`Aggregating`]: Aggregate input batches.
    /// - [`Outputting`]: Handle materialzing all aggregated input.
    /// - [`ExecutionStage::MergingSpills`]: If OOM and spilled before, use this
    ///   stage to finish execution.
    ///
    /// ### Incremental output
    ///
    /// See the [clustered partial aggregate notes] for details.
    ///
    /// [clustered partial aggregate notes]: super::clustered_partial_stream::ClusteredPartialAggregateStream::into_stream
    ///
    /// ## Transition Edges
    ///
    /// 1. Start.
    /// 2. Merge one input batch:
    ///    - If memory fits and no groups are complete, continue reading input.
    ///    - If OOM with a spill context, spill.
    ///    - If OOM without one, emit the completed groups (edge 3), or return
    ///      the error when no group is complete.
    /// 3. Prepare output:
    ///    - Before any spill, a prefix of groups is complete: materialize the
    ///      entire prefix once, retaining the input and active groups to resume
    ///      aggregation.
    ///    - At EOF without spills, materialize all remaining results and prepare
    ///      to output.
    /// 4. Input was exhausted, directly end.
    /// 5. Incremental output at `batch_size`
    /// 6. The batch was fully emitted and retained aggregation can resume.
    /// 7. The output batch was fully emitted.
    /// 8. Input was exhausted after spilling.
    /// 9. Incremental output during reading spill and finalizing results.
    /// 10. The merged spill input was fully aggregated and emitted.
    /// 11. End.
    pub(crate) fn into_stream(self) -> SendableRecordBatchStream {
        let Self {
            reservation,
            context,
            stage,
        } = self;
        let schema = Arc::clone(&context.schema);
        let metrics = context.baseline_metrics.clone();
        let stream = async_try_stream(|mut emitter| async move {
            stage.run(&context, &reservation, &mut emitter).await
        });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));
        Box::pin(ObservedStream::new(stream, metrics, None))
    }
}

impl ExecutionStage {
    /// Runs this stage and every following stage to the end of the output.
    async fn run(
        self,
        context: &ClusteredFinalAggregateContext,
        reservation: &MemoryReservation,
        emitter: &mut TryEmitter<RecordBatch, DataFusionError>,
    ) -> Result<()> {
        let mut stage = Some(self);
        while let Some(current_stage) = stage {
            stage = match current_stage {
                ExecutionStage::Aggregating(aggregating) => {
                    aggregating.handle_stage(context, reservation).await?
                }
                ExecutionStage::Outputting(outputting) => {
                    outputting
                        .handle_stage(context, reservation, emitter)
                        .await?
                }
                ExecutionStage::MergingSpills(mut stream) => {
                    while let Some(batch) = stream.next().await.transpose()? {
                        emitter.emit(batch).await;
                    }
                    None
                }
            };
        }
        Ok(())
    }
}

impl Aggregating {
    /// Reserve the table footprint and, when spillable, one sort index per group.
    fn reservation_size(&self) -> usize {
        let table_size = self.table.memory_size();
        if self.spill_context.is_some() {
            // See `ClusteredFinalAggregateStream` for the spill memory estimate.
            table_size
                .saturating_add(self.table.num_groups().saturating_mul(size_of::<u32>()))
        } else {
            table_size
        }
    }

    /// Spills every retained group after `reservation` refused the table.
    ///
    /// A final stage cannot emit incomplete groups, so the refusal `oom` is
    /// returned when spilling is disabled or no group remains to spill. The
    /// emptied table can still own more than `reservation` holds, for example
    /// after an earlier refusal. It already holds that memory, so it is
    /// charged without another admission: refusing it frees nothing.
    fn spill_retained(
        &mut self,
        reservation: &MemoryReservation,
        oom: DataFusionError,
    ) -> Result<()> {
        let Some(spill_context) = self.spill_context.as_mut() else {
            return Err(oom);
        };
        if self.table.is_empty() {
            return Err(oom);
        }
        spill_context.sort_and_spill(self.table.take_state_batch()?)?;
        reservation.resize(self.table.memory_size());
        Ok(())
    }

    /// Charges the memory the table already holds after `reservation`
    /// refused a handoff, when `context` allows it (see
    /// [`HeldMemory::Charge`]), or returns the refusal `oom`. Spare capacity is
    /// released first.
    fn charge_held(
        &mut self,
        context: &ClusteredFinalAggregateContext,
        reservation: &MemoryReservation,
        oom: DataFusionError,
    ) -> Result<()> {
        if !context.charge_held_memory {
            return Err(oom);
        }
        self.table.shrink_to_fit();
        reservation.resize(self.reservation_size());
        Ok(())
    }

    /// Merges partial states until final results are ready or spill replay begins.
    async fn handle_stage(
        mut self,
        context: &ClusteredFinalAggregateContext,
        reservation: &MemoryReservation,
    ) -> Result<Option<ExecutionStage>> {
        let elapsed_compute = context.baseline_metrics.elapsed_compute();

        while let Some(batch) = self.input.next().await.transpose()? {
            let timer = elapsed_compute.timer();
            self.table.aggregate_batch(&batch)?;

            // Check the reservation before taking completed groups, as the partial
            // stream does, so spilling is unchanged. Without spilling, the refusal
            // is returned below only if no group is complete.
            let oom = match reservation.try_resize(self.reservation_size()) {
                Ok(()) => None,
                Err(oom @ DataFusionError::ResourcesExhausted(_)) => {
                    if self.spill_context.is_none() {
                        Some(oom)
                    } else {
                        self.spill_retained(reservation, oom)?;
                        continue;
                    }
                }
                Err(e) => return Err(e),
            };

            if self.spill_context.as_ref().is_some_and(|s| s.has_spills()) {
                // Spilled groups may recur, so all remaining states must go
                // through replay before any more final results can be emitted.
                continue;
            }
            let Some(batch) = self.table.take_completed_result_batch()? else {
                // Every retained group is incomplete, so nothing can be
                // released. Even spill replay fails here: its single active
                // group outgrows the pool, and charging it batch after batch
                // would grow without any admission.
                match oom {
                    Some(oom) => return Err(oom),
                    None => continue,
                }
            };
            if oom.is_some() {
                // Release the capacity the table grew for this batch so that
                // reserving the remaining groups can succeed.
                self.table.shrink_to_fit();
            }
            timer.done();
            return Ok(Some(ExecutionStage::Outputting(Outputting {
                batch,
                resume: Some(self),
            })));
        }

        // Release upstream resources before materializing output or replaying spills.
        drop(self.input);
        let timer = elapsed_compute.timer();
        if let Some(mut spill_context) = self.spill_context.filter(|s| s.has_spills()) {
            spill_context.sort_and_spill(self.table.take_state_batch()?)?;
            let metrics = self.table.metrics();
            drop(self.table);
            reservation.try_resize(0)?;
            // The outer ObservedStream counts output; replay only shares compute time.
            let stream = (*spill_context).into_replay_stream(
                &context.baseline_metrics.intermediate(),
                metrics,
                reservation.new_empty(),
            )?;
            timer.done();
            return Ok(Some(ExecutionStage::MergingSpills(stream)));
        }

        self.table.input_done();
        let output = self.table.take_completed_result_batch()?;
        drop(self.table);
        timer.done();
        let Some(batch) = output else {
            reservation.try_resize(0)?;
            return Ok(None);
        };
        Ok(Some(ExecutionStage::Outputting(Outputting {
            batch,
            resume: None,
        })))
    }
}

impl Outputting {
    /// Emits slices of one materialized batch without touching the aggregate table.
    async fn handle_stage(
        self,
        context: &ClusteredFinalAggregateContext,
        reservation: &MemoryReservation,
        emitter: &mut TryEmitter<RecordBatch, DataFusionError>,
    ) -> Result<Option<ExecutionStage>> {
        let Self {
            mut batch,
            mut resume,
        } = self;
        let elapsed_compute = context.baseline_metrics.elapsed_compute();
        let mut timer = elapsed_compute.timer();
        let table_memory = resume.as_ref().map_or(0, Aggregating::reservation_size);
        let batch_memory = batch.get_array_memory_size();
        match reservation.try_resize(table_memory + batch_memory) {
            Ok(()) => {}
            Err(DataFusionError::ResourcesExhausted(_)) => {
                // If we cannot hold the batch while slicing, hand it off whole.
                match reservation.try_resize(table_memory) {
                    Ok(()) => {}
                    Err(oom @ DataFusionError::ResourcesExhausted(_)) => {
                        // `batch` holds only completed groups, so it can still
                        // be emitted. The retained groups are incomplete: spill
                        // them, or fail when spilling is impossible.
                        let Some(aggregating) = resume.as_mut() else {
                            return Err(oom);
                        };
                        if aggregating.spill_context.is_some() {
                            aggregating.spill_retained(reservation, oom)?;
                        } else {
                            aggregating.charge_held(context, reservation, oom)?;
                        }
                    }
                    Err(e) => return Err(e),
                }
                timer.done();
                emitter.emit(batch).await;
                return Ok(resume.map(ExecutionStage::Aggregating));
            }
            Err(e) => return Err(e),
        }

        while batch.num_rows() > context.batch_size {
            let output = batch.slice(0, context.batch_size);
            batch =
                batch.slice(context.batch_size, batch.num_rows() - context.batch_size);
            timer.done();
            emitter.emit(output).await;
            timer = elapsed_compute.timer();
        }

        // The final slice transfers ownership of the buffers to the consumer.
        reservation.try_shrink(batch_memory)?;
        timer.done();
        emitter.emit(batch).await;
        Ok(resume.map(ExecutionStage::Aggregating))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExecutionPlan;
    use crate::aggregates::GroupClusteringMode;
    use crate::aggregates::PhysicalGroupBy;
    use crate::aggregates::clustered_test_util::{
        assert_completed_groups_emitted_under_limit, assert_output_unchanged_under_limit,
        assert_resources_exhausted_under_limit, distinct_key_batches, full_batches,
        held_memory, single_key_batches, sum_over_sorted_input,
    };
    use crate::aggregates::resize_refusal_tests::RefuseWhenSet;
    use crate::common::collect;
    use crate::test::TestMemoryExec;
    use arrow::array::{AsArray, Int64Array, StringViewArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_functions_aggregate::{min_max::min_udaf, sum::sum_udaf};
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr_common::sort_expr::LexOrdering;
    use futures::channel::mpsc;
    use futures::{FutureExt, TryStreamExt};
    use std::collections::BTreeMap;

    #[derive(Clone, Copy)]
    enum Finish {
        Collect,
        DropDuringMerge,
        DropDuringReplay,
        InputError,
    }

    #[tokio::test]
    async fn spill_replay_with_another_ordered_partition() -> Result<()> {
        for input_batches in [28, 36, 55, 63] {
            run_shared_pool_case(input_batches, 600 * 1024, Finish::Collect).await?;
        }
        // The same input also produces the reference results without spilling.
        run_shared_pool_case(63, 10 * 1024 * 1024, Finish::Collect).await
    }

    #[tokio::test]
    async fn spill_replay_releases_memory_on_drop() -> Result<()> {
        run_shared_pool_case(36, 600 * 1024, Finish::DropDuringMerge).await?;
        run_shared_pool_case(36, 600 * 1024, Finish::DropDuringReplay).await
    }

    #[tokio::test]
    async fn ordered_spill_releases_memory_on_input_error() -> Result<()> {
        run_shared_pool_case(36, 600 * 1024, Finish::InputError).await
    }

    async fn run_shared_pool_case(
        input_batches: i64,
        limit: usize,
        finish: Finish,
    ) -> Result<()> {
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
        let ordering =
            LexOrdering::new([PhysicalSortExpr::new_default(col("a", &partial_schema)?)])
                .unwrap();
        let input = TestMemoryExec::try_new(
            &[vec![], vec![]],
            Arc::clone(&partial_schema),
            None,
        )?
        .try_with_sort_information(vec![ordering])?;
        let aggregate = AggregateExec::try_new(
            AggregateMode::FinalPartitioned,
            groups.as_final(),
            expressions,
            vec![None; 2],
            Arc::new(TestMemoryExec::update_cache(&Arc::new(input))),
            schema,
        )?;
        assert_eq!(
            aggregate.group_clustering_mode(),
            &GroupClusteringMode::Partial(vec![0])
        );

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
        let context = Arc::new(
            TaskContext::default()
                .with_session_config(
                    SessionConfig::new().with_batch_size(128).set_bool(
                        "datafusion.execution.enable_migration_aggregate",
                        true,
                    ),
                )
                .with_runtime(
                    RuntimeEnvBuilder::new()
                        .with_max_spill_merge_fan_in(2)
                        .with_memory_pool(Arc::clone(&pool))
                        .build_arc()?,
                ),
        );
        let new_stream = |partition| -> Result<_> {
            let (sender, receiver) = mpsc::unbounded();
            let input = Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&partial_schema),
                receiver,
            ));
            let stream = ClusteredFinalAggregateStream::new_with_input(
                &aggregate,
                &context,
                partition,
                input,
                &aggregate.group_clustering_mode,
            )?;
            Ok((sender, stream.into_stream()))
        };
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
        // Keep partition 1's incomplete ordered run live while partition 0 spills
        // and replays. Channel inputs return Pending after each supplied batch,
        // making this interleaving independent of task scheduling.
        let (second_sender, mut second) = new_stream(1)?;
        for batch in 0..55 {
            second_sender
                .unbounded_send(Ok(make_batch(1, batch * 128)))
                .unwrap();
            assert!(second.next().now_or_never().is_none());
        }
        let held = pool.reserved();
        assert!(held > 500 * 1024);
        // Created after `held` is measured, since the stream reserves its initial table.
        let (first_sender, mut first) = new_stream(0)?;
        for batch in 0..input_batches {
            // Repeated keys cross spill runs, so replay must merge their sums.
            first_sender
                .unbounded_send(Ok(make_batch(0, batch * 128)))
                .unwrap();
            assert!(first.next().now_or_never().is_none());
        }
        if limit == 600 * 1024 {
            assert!(aggregate.metrics().unwrap().spill_count().unwrap() > 0);
        }
        match finish {
            Finish::Collect => {
                first_sender.close_channel();
                let mut output = collect(first).await?;
                assert_eq!(pool.reserved(), held);
                second_sender.close_channel();
                output.extend(second.by_ref().try_collect::<Vec<_>>().await?);
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
                assert_eq!(
                    aggregate.metrics().unwrap().spill_count().unwrap() > 0,
                    limit == 600 * 1024
                );
            }
            Finish::DropDuringMerge => {
                first_sender.close_channel();
                assert!(first.next().now_or_never().is_none());
                drop(first);
            }
            Finish::DropDuringReplay => {
                first_sender.close_channel();
                first.next().await.unwrap()?;
                assert!(pool.reserved() > held);
                drop(first);
                assert_eq!(pool.reserved(), held);
            }
            Finish::InputError => {
                first_sender
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
        drop(second);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn fully_clustered_completed_groups_under_memory_pressure() -> Result<()> {
        let plan = sum_over_sorted_input(AggregateMode::Final, &["k"], full_batches())?;
        assert_eq!(plan.group_clustering_mode(), &GroupClusteringMode::Full);
        assert_completed_groups_emitted_under_limit(&plan).await
    }

    /// The limit fits the active group, but not the capacity the table grows
    /// for one batch, so the table must release it after the completed groups
    /// are emitted.
    #[tokio::test]
    async fn completed_groups_release_batch_capacity() -> Result<()> {
        let plan =
            sum_over_sorted_input(AggregateMode::Final, &["k"], distinct_key_batches())?;
        assert_eq!(plan.group_clustering_mode(), &GroupClusteringMode::Full);
        assert_output_unchanged_under_limit(&plan, 100 * 1024).await
    }

    /// A single key never completes a group before the input ends. The limit
    /// fits only the initial table.
    #[tokio::test]
    async fn memory_pressure_without_completed_groups() -> Result<()> {
        let plan =
            sum_over_sorted_input(AggregateMode::Final, &["k"], single_key_batches())?;
        let limit = held_memory(&plan).await?;
        assert_resources_exhausted_under_limit(&plan, limit).await
    }

    /// A handoff of completed groups can find the table larger than its
    /// reservation, for example after an unspilled refusal. If the pool then
    /// refuses both the batch and the table, the completed groups are still
    /// final: the stream emits them and spills the incomplete groups it
    /// retains. Without spill files it must return the refusal, because a
    /// final output cannot repeat a group.
    #[tokio::test]
    async fn refused_handoff_spills_the_retained_groups() -> Result<()> {
        for spill in [true, false] {
            // The last key has 64 groups, which stay incomplete in the table.
            let keys: Vec<i64> = (0..8192).chain([8192; 64]).collect();
            let second = (0..8192).map(|k| k % 5).chain(0..64).collect();
            let plan = sum_over_sorted_input(
                AggregateMode::Final,
                &["k", "j"],
                vec![vec![keys, second]],
            )?;
            assert_eq!(
                plan.group_clustering_mode(),
                &GroupClusteringMode::Partial(vec![0])
            );
            let expected = final_sums(
                &collect(plan.execute(0, Arc::new(TaskContext::default()))?).await?,
            );
            let pool = Arc::new(RefuseWhenSet::default());
            let mut runtime = RuntimeEnvBuilder::new()
                .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>);
            if !spill {
                runtime = runtime.with_disk_manager_builder(
                    datafusion_execution::disk_manager::DiskManagerBuilder::default()
                        .with_mode(
                            datafusion_execution::disk_manager::DiskManagerMode::Disabled,
                        ),
                );
            }
            let context = Arc::new(
                TaskContext::default()
                    .with_session_config(SessionConfig::new().with_batch_size(256))
                    .with_runtime(runtime.build_arc()?),
            );
            let input = collect(plan.input().execute(0, Arc::clone(&context))?).await?;
            // The test merges the only input batch itself, so the stream reads
            // an empty input and reaches EOF after the handoff.
            let stream = ClusteredFinalAggregateStream::new_with_input(
                &plan,
                &context,
                0,
                Box::pin(crate::stream::EmptyRecordBatchStream::new(
                    plan.input().schema(),
                )),
                plan.group_clustering_mode(),
            )?;
            let ClusteredFinalAggregateStream {
                reservation,
                context: stream_context,
                stage: ExecutionStage::Aggregating(mut aggregating),
            } = stream
            else {
                unreachable!("a new stream starts by aggregating");
            };
            assert_eq!(aggregating.spill_context.is_some(), spill);
            // Merge without resizing, so the reservation falls behind the table.
            aggregating.table.aggregate_batch(&input[0])?;
            let completed = aggregating
                .table
                .take_completed_result_batch()?
                .expect("every key but the last is complete");
            assert_eq!(aggregating.table.num_groups(), 64);
            assert!(aggregating.reservation_size() > reservation.size());
            pool.refuse
                .store(true, std::sync::atomic::Ordering::Relaxed);

            let completed_rows = completed.num_rows();
            let outputting = Outputting {
                batch: completed,
                resume: Some(aggregating),
            };
            let shortage = Arc::clone(&pool);
            let output: Vec<Result<RecordBatch>> =
                async_try_stream(|mut emitter| async move {
                    let next = outputting
                        .handle_stage(&stream_context, &reservation, &mut emitter)
                        .await?;
                    let Some(ExecutionStage::Aggregating(aggregating)) = next else {
                        return internal_err!("aggregation must resume after a handoff");
                    };
                    assert!(aggregating.spill_context.as_ref().unwrap().has_spills());
                    assert!(aggregating.table.is_empty());
                    assert_eq!(reservation.size(), aggregating.table.memory_size());
                    // The shortage ends. Every retained group must come back
                    // from replay, merged once.
                    shortage
                        .refuse
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    ExecutionStage::Aggregating(aggregating)
                        .run(&stream_context, &reservation, &mut emitter)
                        .await
                })
                .collect()
                .await;
            if spill {
                let output = output.into_iter().collect::<Result<Vec<_>>>()?;
                assert_eq!(output[0].num_rows(), completed_rows);
                assert!(output.len() > 1, "replay must emit the spilled groups");
                assert_eq!(final_sums(&output), expected);
            } else {
                assert!(
                    matches!(
                        output.as_slice(),
                        [Err(DataFusionError::ResourcesExhausted(_))]
                    ),
                    "expected ResourcesExhausted, got {output:?}"
                );
            }
            assert_eq!(pool.reserved(), 0);
        }
        Ok(())
    }

    /// After a refused table spills, the emptied table can still own more than
    /// its reservation, here shrunk to one byte. Refusing that memory frees
    /// nothing, so the stream charges it and continues instead of returning a
    /// context-wrapped refusal.
    #[tokio::test]
    async fn spill_charges_the_emptied_table() -> Result<()> {
        let keys: Vec<i64> = (0..8192).chain([8192; 64]).collect();
        let second = (0..8192).map(|k| k % 5).chain(0..64).collect();
        let plan = sum_over_sorted_input(
            AggregateMode::Final,
            &["k", "j"],
            vec![vec![keys, second]],
        )?;
        let expected = final_sums(
            &collect(plan.execute(0, Arc::new(TaskContext::default()))?).await?,
        );
        let pool = Arc::new(RefuseWhenSet::default());
        let context = Arc::new(
            TaskContext::default()
                .with_session_config(SessionConfig::new().with_batch_size(256))
                .with_runtime(
                    RuntimeEnvBuilder::new()
                        .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
                        .build_arc()?,
                ),
        );
        let input = collect(plan.input().execute(0, Arc::clone(&context))?).await?;
        let stream = ClusteredFinalAggregateStream::new_with_input(
            &plan,
            &context,
            0,
            Box::pin(crate::stream::EmptyRecordBatchStream::new(
                plan.input().schema(),
            )),
            plan.group_clustering_mode(),
        )?;
        let ClusteredFinalAggregateStream {
            reservation,
            context: stream_context,
            stage: ExecutionStage::Aggregating(mut aggregating),
        } = stream
        else {
            unreachable!("a new stream starts by aggregating");
        };
        aggregating.table.aggregate_batch(&input[0])?;
        reservation.resize(1);
        pool.refuse
            .store(true, std::sync::atomic::Ordering::Relaxed);
        aggregating.spill_retained(
            &reservation,
            datafusion_common::resources_datafusion_err!("refused"),
        )?;
        assert!(aggregating.spill_context.as_ref().unwrap().has_spills());
        assert!(aggregating.table.is_empty());
        assert!(aggregating.table.memory_size() > 1);
        assert_eq!(reservation.size(), aggregating.table.memory_size());

        pool.refuse
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let output: Vec<RecordBatch> = async_try_stream(|mut emitter| async move {
            ExecutionStage::Aggregating(aggregating)
                .run(&stream_context, &reservation, &mut emitter)
                .await
        })
        .try_collect()
        .await?;
        assert_eq!(final_sums(&output), expected);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    /// Admits only the first growth of each reservation, as a pool without
    /// lending headroom admits a new table and refuses its later growth.
    fn refuse_established() -> Arc<RefuseWhenSet> {
        let pool = RefuseWhenSet::default();
        pool.refuse
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Arc::new(pool)
    }

    /// A replay stream, which has full group clustering and no spill path,
    /// finishes when the pool refuses every growth after its initial table:
    /// distinct keys give refused handoffs, and the stream emits completed
    /// groups and charges the active group's memory. A single key gives the
    /// refusal with no completed group. That one group needs more memory than
    /// the pool grants, so both modes return `ResourcesExhausted`.
    #[tokio::test]
    async fn replay_finishes_when_every_later_growth_is_refused() -> Result<()> {
        for (single_key, batches) in [
            (true, single_key_batches()),
            (false, distinct_key_batches()),
            (false, full_batches()),
        ] {
            let plan = sum_over_sorted_input(AggregateMode::Final, &["k"], batches)?;
            assert_eq!(plan.group_clustering_mode(), &GroupClusteringMode::Full);
            let expected =
                collect(plan.execute(0, Arc::new(TaskContext::default()))?).await?;
            for held_memory in [HeldMemory::Charge, HeldMemory::Refuse] {
                let pool: Arc<dyn MemoryPool> = refuse_established();
                let context = Arc::new(
                    TaskContext::default()
                        .with_session_config(SessionConfig::new().with_batch_size(256))
                        .with_runtime(
                            RuntimeEnvBuilder::new()
                                .with_memory_pool(Arc::clone(&pool))
                                .build_arc()?,
                        ),
                );
                let result = async {
                    let input = plan.input().execute(0, Arc::clone(&context))?;
                    let stream =
                        ClusteredFinalAggregateStream::new_with_input_and_metrics(
                            &plan,
                            &context,
                            0,
                            input,
                            &GroupClusteringMode::Full,
                            BaselineMetrics::new(&plan.metrics, 0),
                            ClusteredAggregateTableMetrics::new(&plan, 0),
                            None,
                            MemoryConsumer::new("replay").register(&pool),
                            held_memory,
                        )?;
                    collect(stream.into_stream()).await
                }
                .await;
                match result {
                    Ok(output) => {
                        assert!(
                            !single_key,
                            "a refused group that never completes must fail"
                        );
                        let keys: Vec<i64> = output
                            .iter()
                            .flat_map(|batch| {
                                batch
                                    .column(0)
                                    .as_primitive::<arrow::datatypes::Int64Type>()
                                    .values()
                                    .to_vec()
                            })
                            .collect();
                        assert!(
                            keys.is_sorted() && keys.windows(2).all(|w| w[0] != w[1]),
                            "replay must emit each group once, in key order"
                        );
                        assert_eq!(
                            arrow::util::pretty::pretty_format_batches(&output)?
                                .to_string(),
                            arrow::util::pretty::pretty_format_batches(&expected)?
                                .to_string()
                        );
                    }
                    Err(DataFusionError::ResourcesExhausted(_))
                        if held_memory == HeldMemory::Refuse || single_key => {}
                    Err(error) => panic!("{held_memory:?}: unexpected {error}"),
                }
                assert_eq!(pool.reserved(), 0);
            }
        }
        Ok(())
    }

    /// The replay table is admitted like any new table, so a hard limit that
    /// cannot hold it refuses the replay before it starts.
    #[tokio::test]
    async fn replay_table_needs_admission() -> Result<()> {
        let plan = sum_over_sorted_input(AggregateMode::Final, &["k"], full_batches())?;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(0));
        let context = Arc::new(
            TaskContext::default().with_runtime(
                RuntimeEnvBuilder::new()
                    .with_memory_pool(Arc::clone(&pool))
                    .build_arc()?,
            ),
        );
        let result = ClusteredFinalAggregateStream::new_with_input_and_metrics(
            &plan,
            &context,
            0,
            Box::pin(crate::stream::EmptyRecordBatchStream::new(
                plan.input().schema(),
            )),
            &GroupClusteringMode::Full,
            BaselineMetrics::new(&plan.metrics, 0),
            ClusteredAggregateTableMetrics::new(&plan, 0),
            None,
            MemoryConsumer::new("replay").register(&pool),
            HeldMemory::Charge,
        );
        assert!(
            matches!(result, Err(DataFusionError::ResourcesExhausted(_))),
            "the replay table must be admitted"
        );
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    /// A refused replay handoff charges exactly the memory the retained table
    /// holds after releasing its spare capacity. The test charges a table
    /// that holds one active group, as a handoff leaves it.
    #[tokio::test]
    async fn refused_replay_charges_the_active_group() -> Result<()> {
        let plan =
            sum_over_sorted_input(AggregateMode::Final, &["k"], single_key_batches())?;
        let pool: Arc<dyn MemoryPool> = refuse_established();
        let context = Arc::new(
            TaskContext::default()
                .with_session_config(SessionConfig::new().with_batch_size(256))
                .with_runtime(
                    RuntimeEnvBuilder::new()
                        .with_memory_pool(Arc::clone(&pool))
                        .build_arc()?,
                ),
        );
        let input = collect(plan.input().execute(0, Arc::clone(&context))?).await?;
        let stream = ClusteredFinalAggregateStream::new_with_input_and_metrics(
            &plan,
            &context,
            0,
            Box::pin(crate::stream::EmptyRecordBatchStream::new(
                plan.input().schema(),
            )),
            &GroupClusteringMode::Full,
            BaselineMetrics::new(&plan.metrics, 0),
            ClusteredAggregateTableMetrics::new(&plan, 0),
            None,
            MemoryConsumer::new("replay").register(&pool),
            HeldMemory::Charge,
        )?;
        let ClusteredFinalAggregateStream {
            reservation,
            context: stream_context,
            stage: ExecutionStage::Aggregating(mut aggregating),
        } = stream
        else {
            unreachable!("a new stream starts by aggregating");
        };
        assert_eq!(reservation.size(), aggregating.table.memory_size());
        aggregating.table.aggregate_batch(&input[0])?;
        assert!(aggregating.reservation_size() > reservation.size());
        assert!(
            reservation
                .try_resize(aggregating.reservation_size())
                .is_err()
        );
        aggregating.charge_held(
            &stream_context,
            &reservation,
            datafusion_common::resources_datafusion_err!("refused"),
        )?;
        assert_eq!(aggregating.table.num_groups(), 1);
        assert_eq!(reservation.size(), aggregating.reservation_size());
        drop(aggregating);
        drop(reservation);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    /// `SUM` by `(k, j)` of a final output, which must not repeat a group.
    fn final_sums(output: &[RecordBatch]) -> BTreeMap<(i64, i64), i64> {
        let mut sums = BTreeMap::new();
        for batch in output {
            let [k, j, sum] = batch.columns() else {
                panic!("expected k, j and the sum");
            };
            let (k, j, sum) = (
                k.as_primitive::<arrow::datatypes::Int64Type>(),
                j.as_primitive::<arrow::datatypes::Int64Type>(),
                sum.as_primitive::<arrow::datatypes::Int64Type>(),
            );
            for row in 0..batch.num_rows() {
                let key = (k.value(row), j.value(row));
                assert!(
                    sums.insert(key, sum.value(row)).is_none(),
                    "final output repeats group {key:?}"
                );
            }
        }
        sums
    }
}
