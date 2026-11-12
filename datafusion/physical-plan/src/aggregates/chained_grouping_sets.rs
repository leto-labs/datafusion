// Modified by the Longbow authors; see third_party/README.md.
//
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

//! Partition-local grouping-set state reuse. Only maximal requested sets read
//! raw rows. Each other set merges the smallest containing computed set. Native
//! partial states remain mergeable even when a bounded partial flushes or skips;
//! coarser merges normally complete, but bounded retries may emit duplicate states.
//! AVG retains its
//! sum/count state through the entire chain.
//!
//! Nothing is shared between executions or partitions. Retained state owns a
//! reservation or an ordinary native spill file. Cancellation drops the same
//! owners as success or error, and empty input never reopens the raw source.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

use arrow::compute::cast;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_common::utils::memory::RecordBatchMemoryCounter;
use datafusion_common::{DataFusionError, Result, ScalarValue, internal_datafusion_err};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryLimit, MemoryReservation};
use datafusion_execution::spill_file::SpillFile;
use datafusion_functions_aggregate::{
    average::Avg,
    count::Count,
    min_max::{Max, Min},
    sum::Sum,
};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{Column, lit};
use datafusion_physical_expr_common::physical_expr::is_volatile;
use futures::{StreamExt, TryStreamExt, stream};

use super::{
    AggregateExec, AggregateMode, GroupedHashAggregateStream,
    PartialReduceHashAggregateStream, PhysicalGroupBy, create_schema, group_id_array,
    max_duplicate_ordinal,
};
use crate::empty::EmptyExec;
use crate::execution_plan::EmissionType;
use crate::filter::FilterExecBuilder;
use crate::metrics::{BaselineMetrics, MetricBuilder, RecordOutput, SpillMetrics};
use crate::spill::in_progress_spill_file::InProgressSpillFile;
use crate::spill::spill_manager::SpillManager;
use crate::stream::RecordBatchStreamAdapter;
use crate::streaming::{PartitionStream, StreamingTableExec};
use crate::{ExecutionPlan, InputOrderMode, SendableRecordBatchStream};

struct Level {
    mask: Vec<bool>,
    parent: Option<usize>,
    consumers: usize,
    ordinals: Vec<usize>,
    source: Option<Arc<dyn ExecutionPlan>>,
}

fn contains(finer: &[bool], coarser: &[bool]) -> bool {
    finer.iter().zip(coarser).all(|(f, c)| !f || *c)
}

fn levels(groups: &[Vec<bool>]) -> Vec<Level> {
    let mut masks = groups.to_vec();
    masks.sort();
    masks.dedup();
    masks.sort_by_key(|m| m.iter().filter(|v| **v).count());
    let mut result: Vec<Level> = Vec::with_capacity(masks.len());
    for mask in masks {
        let parent = result
            .iter()
            .enumerate()
            .filter(|(_, level)| contains(&level.mask, &mask))
            .max_by_key(|(_, level)| level.mask.iter().filter(|v| **v).count())
            .map(|(index, _)| index);
        if let Some(parent) = parent {
            result[parent].consumers += 1;
        }
        result.push(Level {
            mask,
            parent,
            consumers: 0,
            ordinals: vec![],
            source: None,
        });
    }
    let mut ordinals: HashMap<&[bool], usize> = HashMap::new();
    for mask in groups {
        let ordinal = ordinals.entry(mask).or_default();
        result
            .iter_mut()
            .find(|level| level.mask == *mask)
            .unwrap()
            .ordinals
            .push(*ordinal);
        *ordinal += 1;
    }
    result
}

fn floating_type(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(_, value) => floating_type(value),
        other => other.is_floating(),
    }
}

fn supported(agg: &AggregateExec) -> bool {
    if !agg.group_by.has_grouping_set
        || agg.group_by.groups.len() < 2
        || agg.aggr_expr.is_empty()
        || !matches!(
            agg.mode,
            AggregateMode::Partial
                | AggregateMode::Single
                | AggregateMode::SinglePartitioned
        )
        || agg.input_order_mode != InputOrderMode::Linear
        || agg.limit_options.is_some()
        || agg.group_by.expr.iter().any(|(expr, _)| is_volatile(expr))
        || agg.filter_expr.iter().flatten().any(is_volatile)
    {
        return false;
    }
    agg.aggr_expr.iter().all(|expr| {
        let function = expr.fun().inner();
        !expr.is_distinct()
            && expr.stored_order_bys().is_empty()
            && !expr.expressions().iter().any(is_volatile)
            && (function.downcast_ref::<Sum>().is_some()
                || function.downcast_ref::<Count>().is_some()
                || ((function.downcast_ref::<Min>().is_some()
                    || function.downcast_ref::<Max>().is_some())
                    && !floating_type(expr.field().data_type()))
                || function.downcast_ref::<Avg>().is_some())
    })
}

pub(super) fn execute(
    agg: &AggregateExec,
    partition: usize,
    context: Arc<TaskContext>,
) -> Result<Option<SendableRecordBatchStream>> {
    // Without disk, retaining a level can exhaust a finite pool even when
    // native Partial could emit it early. Keep that bounded native path.
    if !supported(agg)
        || (matches!(context.memory_pool().memory_limit(), MemoryLimit::Finite(_))
            && !context.runtime_env().disk_manager.tmp_files_enabled())
    {
        return Ok(None);
    }
    let levels = levels(&agg.group_by.groups);
    let roots = levels.iter().filter(|level| level.parent.is_none()).count();
    // Incomparable sets without duplicates provide no reuse. This also keeps
    // the internal maximal-set aggregate on the native expanding implementation.
    if roots == levels.len() && agg.group_by.groups.len() == levels.len() {
        return Ok(None);
    }
    let state_schema = Arc::new(create_schema(
        &agg.input.schema(),
        &agg.group_by,
        &agg.aggr_expr,
        AggregateMode::Partial,
    )?);
    let baseline = BaselineMetrics::new(&agg.metrics, partition);
    MetricBuilder::new(&agg.metrics)
        .counter("chained_grouping_sets", partition)
        .add(agg.group_by.groups.len());
    MetricBuilder::new(&agg.metrics)
        .counter("chained_grouping_roots", partition)
        .add(roots);
    MetricBuilder::new(&agg.metrics)
        .counter("chained_grouping_merges", partition)
        .add(levels.len() - roots);
    let mut original = agg.with_new_limit_options(None);
    original.metrics = agg.metrics.clone();
    let state = State {
        original,
        context,
        partition,
        state_schema,
        baseline,
        levels,
        initialized: false,
        roots_empty: false,
        cursor: 0,
        ordinal: 0,
        emitting: None,
        finalizer: None,
    };
    let stream = stream::try_unfold(state, |mut state| async move {
        loop {
            if !state.initialized {
                state.initialize().await?;
            }
            if state.cursor == state.levels.len() {
                return Ok(None);
            }
            if let Some(emitting) = &mut state.emitting {
                if let Some(batch) = emitting.try_next().await? {
                    let timer = state.baseline.elapsed_compute().timer();
                    let level = &state.levels[state.cursor];
                    let mut columns = batch.columns().to_vec();
                    columns[state.key_count()] = group_id_array(
                        &level.mask,
                        level.ordinals[state.ordinal],
                        state.max_ordinal(),
                        batch.num_rows(),
                    )?;
                    let output = RecordBatch::try_new(
                        Arc::clone(&state.original.schema),
                        columns,
                    )?;
                    timer.done();
                    let output = output.record_output(&state.baseline);
                    return Ok(Some((output, state)));
                }
                state.emitting = None;
                if let Some(finalizer) = state.finalizer.take() {
                    state.add_compute(&finalizer);
                }
                state.ordinal += 1;
                if state.ordinal == state.levels[state.cursor].ordinals.len() {
                    if state.levels[state.cursor].consumers == 0 {
                        state.levels[state.cursor].source = None;
                    }
                    state.cursor += 1;
                    state.ordinal = 0;
                    continue;
                }
            }
            state.prepare().await?;
            state.start_output()?;
        }
    });
    Ok(Some(Box::pin(RecordBatchStreamAdapter::new(
        Arc::clone(&agg.schema),
        stream,
    ))))
}

struct State {
    original: AggregateExec,
    context: Arc<TaskContext>,
    partition: usize,
    state_schema: SchemaRef,
    baseline: BaselineMetrics,
    levels: Vec<Level>,
    initialized: bool,
    roots_empty: bool,
    cursor: usize,
    ordinal: usize,
    emitting: Option<SendableRecordBatchStream>,
    finalizer: Option<AggregateExec>,
}

impl State {
    fn key_count(&self) -> usize {
        self.original.group_by.expr.len()
    }
    fn max_ordinal(&self) -> usize {
        max_duplicate_ordinal(&self.original.group_by.groups)
    }

    fn id_expr(&self, mask: &[bool]) -> Result<Arc<dyn PhysicalExpr>> {
        let array = group_id_array(mask, 0, self.max_ordinal(), 1)?;
        Ok(lit(ScalarValue::try_from_array(&array, 0)?))
    }

    fn columns(&self, schema: &SchemaRef) -> Vec<(Arc<dyn PhysicalExpr>, String)> {
        schema
            .fields()
            .iter()
            .take(self.key_count() + 1)
            .enumerate()
            .map(|(i, field)| {
                (
                    Arc::new(Column::new(field.name(), i)) as _,
                    field.name().clone(),
                )
            })
            .collect()
    }

    fn add_compute(&self, agg: &AggregateExec) {
        let nanos = agg.metrics.clone_inner().elapsed_compute().unwrap_or(0);
        self.baseline
            .elapsed_compute()
            .add_duration(Duration::from_nanos(nanos as u64));
    }

    fn merge(
        &self,
        input: Arc<dyn ExecutionPlan>,
        group: PhysicalGroupBy,
        mode: AggregateMode,
    ) -> Result<AggregateExec> {
        let mut aggregate = AggregateExec::try_new(
            mode,
            group,
            self.original.aggr_expr.to_vec(),
            vec![None; self.original.aggr_expr.len()],
            input,
            Arc::clone(&self.original.input_schema),
        )?;
        // Literal rolled-up keys and IDs satisfy trivial ordering prefixes,
        // which otherwise select the legacy partially-sorted implementation.
        // Retained sources have no useful input order; hash grouping is valid
        // and all advertised constant-column ordering remains true.
        Self::use_hash_grouping(&mut aggregate);
        Ok(aggregate)
    }

    fn use_hash_grouping(aggregate: &mut AggregateExec) {
        aggregate.input_order_mode = InputOrderMode::Linear;
        aggregate.required_input_ordering = None;
        let mut equivalence = aggregate.cache.eq_properties.clone();
        equivalence.clear_orderings();
        aggregate.cache = Arc::new(
            aggregate
                .cache
                .as_ref()
                .clone()
                .with_eq_properties(equivalence)
                .with_emission_type(EmissionType::Final),
        );
    }

    async fn initialize(&mut self) -> Result<()> {
        let roots: Vec<_> = self
            .levels
            .iter()
            .filter(|level| level.parent.is_none())
            .map(|level| level.mask.clone())
            .collect();
        // A single nonempty maximal set uses the migrated single-set hash path.
        // Grand totals retain native empty-input synthesis; incomparable maximal
        // sets share one native raw-input aggregate, so the input opens once.
        let group = if roots.len() == 1 && roots[0].iter().any(|omitted| !omitted) {
            let mut expr = self
                .original
                .group_by
                .expr
                .iter()
                .enumerate()
                .map(|(i, expr)| {
                    if roots[0][i] {
                        self.original.group_by.null_expr[i].clone()
                    } else {
                        expr.clone()
                    }
                })
                .collect::<Vec<_>>();
            expr.push((
                self.id_expr(&roots[0])?,
                datafusion_expr::Aggregate::INTERNAL_GROUPING_ID.into(),
            ));
            PhysicalGroupBy::new_single(expr)
        } else {
            PhysicalGroupBy::new(
                self.original.group_by.expr.clone(),
                self.original.group_by.null_expr.clone(),
                roots.clone(),
                true,
            )
        };
        let mut raw = AggregateExec::try_new(
            AggregateMode::Partial,
            group,
            self.original.aggr_expr.to_vec(),
            self.original.filter_expr.to_vec(),
            Arc::clone(&self.original.input),
            Arc::clone(&self.original.input_schema),
        )?;
        // The original admitted aggregate is linear. Adding a constant ID must
        // not manufacture a partially-sorted path for its raw finest grouping.
        Self::use_hash_grouping(&mut raw);
        // Flushed/skipped partials are valid merge states. Completing the
        // finest level here adds an entire hash pass, with almost no reduction
        // for high-cardinality keys. The external final aggregate (or our
        // Single-mode finalizer) already completes that level. Its first child
        // merges any duplicate fine states while computing the coarser keys.
        // Open the actual caller partition exactly once; retries below only
        // read the retained sources that this call produces.
        let stream = raw.execute_typed(self.partition, &self.context)?.into();
        let (source, rows) = self.store(&raw, stream).await?;
        self.roots_empty = rows == 0;
        MetricBuilder::new(&self.original.metrics)
            .counter("chained_root_state_rows", self.partition)
            .add(rows);
        for i in 0..self.levels.len() {
            if self.levels[i].parent.is_some() {
                continue;
            }
            let input = Arc::clone(&source);
            self.levels[i].source = if roots.len() == 1 {
                Some(input)
            } else {
                let predicate = datafusion_physical_expr::expressions::binary(
                    Arc::new(Column::new(
                        datafusion_expr::Aggregate::INTERNAL_GROUPING_ID,
                        self.key_count(),
                    )),
                    datafusion_expr::Operator::Eq,
                    self.id_expr(&self.levels[i].mask)?,
                    &self.state_schema,
                )?;
                Some(Arc::new(FilterExecBuilder::new(predicate, input).build()?))
            };
        }
        self.initialized = true;
        Ok(())
    }

    async fn prepare(&mut self) -> Result<()> {
        if self.levels[self.cursor].source.is_some() {
            return Ok(());
        }
        let parent = self.levels[self.cursor]
            .parent
            .ok_or_else(|| internal_datafusion_err!("Missing grouping-set parent"))?;
        let input = Arc::clone(self.levels[parent].source.as_ref().ok_or_else(|| {
            internal_datafusion_err!("Released grouping-set parent too early")
        })?);
        let mask = &self.levels[self.cursor].mask;
        let merge = if self.roots_empty && mask.iter().all(|omitted| *omitted) {
            // Empty grouping-set synthesis primes raw argument types with an
            // all-false filter. Merge-state columns cannot enter that path.
            // Use a fresh empty source with the original raw schema, never
            // reopen the input exchange (which has already been consumed).
            let group = PhysicalGroupBy::new(
                self.original.group_by.expr.clone(),
                self.original.group_by.null_expr.clone(),
                vec![mask.clone()],
                true,
            );
            AggregateExec::try_new(
                AggregateMode::Partial,
                group,
                self.original.aggr_expr.to_vec(),
                self.original.filter_expr.to_vec(),
                Arc::new(EmptyExec::new(Arc::clone(&self.original.input_schema))),
                Arc::clone(&self.original.input_schema),
            )?
        } else {
            let mut expr = self.columns(&self.state_schema);
            for (i, omitted) in mask.iter().enumerate() {
                if *omitted {
                    expr[i].0 = lit(ScalarValue::try_new_null(
                        self.state_schema.field(i).data_type(),
                    )?);
                }
            }
            expr[self.key_count()].0 = self.id_expr(mask)?;
            self.merge(
                input,
                PhysicalGroupBy::new_single(expr),
                AggregateMode::PartialReduce,
            )?
        };
        let (source, rows) = self.materialize(&merge).await?;
        self.levels[self.cursor].source = Some(source);
        MetricBuilder::new(&self.original.metrics)
            .counter("chained_derived_state_rows", self.partition)
            .add(rows);
        self.levels[parent].consumers -= 1;
        if self.levels[parent].consumers == 0 {
            self.levels[parent].source = None;
        }
        Ok(())
    }

    fn start_output(&mut self) -> Result<()> {
        let source = Arc::clone(self.levels[self.cursor].source.as_ref().unwrap());
        if self.original.mode == AggregateMode::Partial {
            self.emitting = Some(source.execute(0, Arc::clone(&self.context))?);
        } else {
            let finalizer = self.merge(
                source,
                PhysicalGroupBy::new_single(self.columns(&self.state_schema)),
                AggregateMode::Final,
            )?;
            self.emitting = Some(finalizer.execute_typed(0, &self.context)?.into());
            self.finalizer = Some(finalizer);
        }
        Ok(())
    }

    async fn materialize(
        &mut self,
        agg: &AggregateExec,
    ) -> Result<(Arc<dyn ExecutionPlan>, usize)> {
        // Normal modern/native merges are unchanged. Only an exhausted state
        // merge retries with bounded early emission: sort-spilling would need
        // another reservation while the retained parent is still held.
        let modern = agg.mode == AggregateMode::PartialReduce
            && agg.group_by.is_single()
            && agg.input_order_mode == InputOrderMode::Linear
            && self
                .context
                .session_config()
                .options()
                .execution
                .enable_migration_aggregate
            && matches!(
                self.context.memory_pool().memory_limit(),
                MemoryLimit::Finite(_)
            );
        let attempt = if modern {
            MetricBuilder::new(&self.original.metrics)
                .counter("chained_modern_merges", self.partition)
                .add(1);
            match PartialReduceHashAggregateStream::new(agg, &self.context, 0) {
                Ok(stream) => self.store(agg, Box::pin(stream)).await,
                Err(error) => Err(error),
            }
        } else {
            match agg.execute_typed(0, &self.context) {
                Ok(stream) => self.store(agg, stream.into()).await,
                Err(error) => Err(error),
            }
        };
        match attempt {
            Err(error)
                if agg.mode == AggregateMode::PartialReduce
                    && matches!(
                        error.find_root(),
                        DataFusionError::ResourcesExhausted(_)
                    ) =>
            {
                // The failed stream, table, retained output and reservations
                // have dropped before this retry. Never reopen raw input.
                self.add_compute(agg);
                MetricBuilder::new(&self.original.metrics)
                    .counter("chained_merge_retries", self.partition)
                    .add(1);
                let fallback = agg.with_new_limit_options(None);
                let stream = GroupedHashAggregateStream::new_emitting_partial_states(
                    &fallback,
                    &self.context,
                    0,
                )?;
                self.store(&fallback, Box::pin(stream)).await
            }
            result => result,
        }
    }

    async fn store(
        &mut self,
        agg: &AggregateExec,
        mut input: SendableRecordBatchStream,
    ) -> Result<(Arc<dyn ExecutionPlan>, usize)> {
        let can_spill = self.context.runtime_env().disk_manager.tmp_files_enabled();
        let reservation = Arc::new(
            MemoryConsumer::new("ChainedGroupingSets retained state")
                .with_can_spill(can_spill)
                .register(self.context.memory_pool()),
        );
        let manager = SpillManager::new(
            self.context.runtime_env(),
            SpillMetrics::new(&self.original.metrics, self.partition),
            Arc::clone(&self.state_schema),
        )
        .with_attempt_tasks(crate::attempt_tasks::AttemptTasks::from_task(&self.context))
        .with_compression_type(
            self.context
                .session_config()
                .options()
                .execution
                .spill_compression,
        );
        let mut counter = RecordBatchMemoryCounter::new();
        let mut batches = vec![];
        let mut writer: Option<InProgressSpillFile> = None;
        let mut rows = 0;
        while let Some(batch) = input.try_next().await? {
            rows += batch.num_rows();
            let timer = self.baseline.elapsed_compute().timer();
            let mut columns = batch.columns().to_vec();
            let gid = self.key_count();
            if columns[gid].data_type() != self.state_schema.field(gid).data_type() {
                columns[gid] = cast(
                    columns[gid].as_ref(),
                    self.state_schema.field(gid).data_type(),
                )?;
            }
            let batch = RecordBatch::try_new(Arc::clone(&self.state_schema), columns)?;
            if let Some(writer) = &mut writer {
                writer.append_batch(&batch)?;
            } else {
                let size = counter.count_batch(&batch);
                match reservation.try_grow(size) {
                    Ok(()) => batches.push(batch),
                    Err(error) => {
                        if !can_spill {
                            return Err(error);
                        }
                        let mut spill = manager
                            .create_in_progress_file("chained grouping-set state")?;
                        for previous in batches.drain(..) {
                            spill.append_batch(&previous)?;
                        }
                        spill.append_batch(&batch)?;
                        reservation.free();
                        writer = Some(spill);
                    }
                }
            }
            timer.done();
        }
        self.add_compute(agg);
        let storage = match writer {
            Some(mut writer) => Storage::Disk(manager, writer.finish()?),
            None => Storage::Memory(batches, reservation),
        };
        let partition = Arc::new(Stored {
            schema: Arc::clone(&self.state_schema),
            storage,
        });
        Ok((
            Arc::new(StreamingTableExec::try_new(
                Arc::clone(&self.state_schema),
                vec![partition],
                None,
                [],
                false,
                None,
            )?),
            rows,
        ))
    }
}

enum Storage {
    Memory(Vec<RecordBatch>, Arc<MemoryReservation>),
    Disk(SpillManager, Option<Arc<dyn SpillFile>>),
}

impl Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Memory(batches, reservation) => f
                .debug_struct("RetainedGroupingState")
                .field("batches", &batches.len())
                .field("reserved_bytes", &reservation.size())
                .finish(),
            Self::Disk(_, file) => f
                .debug_struct("SpilledGroupingState")
                .field("has_file", &file.is_some())
                .finish(),
        }
    }
}

#[derive(Debug)]
struct Stored {
    schema: SchemaRef,
    storage: Storage,
}

impl PartitionStream for Stored {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }
    fn execute(&self, _context: Arc<TaskContext>) -> SendableRecordBatchStream {
        match &self.storage {
            Storage::Memory(batches, reservation) => {
                let reservation = Arc::clone(reservation);
                let stream = stream::iter(batches.clone()).map(move |batch| {
                    let _keep_reservation = &reservation;
                    Ok(batch)
                });
                Box::pin(RecordBatchStreamAdapter::new(
                    Arc::clone(&self.schema),
                    stream,
                ))
            }
            Storage::Disk(manager, Some(file)) => {
                match manager.read_spill_as_stream_unbuffered(Arc::clone(file), None) {
                    Ok(stream) => stream,
                    Err(error) => Box::pin(RecordBatchStreamAdapter::new(
                        Arc::clone(&self.schema),
                        stream::once(async move { Err(error) }),
                    )),
                }
            }
            Storage::Disk(_, None) => Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&self.schema),
                stream::empty(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{floating_type, levels};
    use arrow::datatypes::DataType;

    #[test]
    fn chained_grouping_sets_float_extrema_include_dictionary_values() {
        assert!(floating_type(&DataType::Float16));
        assert!(floating_type(&DataType::Float32));
        assert!(floating_type(&DataType::Float64));
        assert!(floating_type(&DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Float64),
        )));
        assert!(!floating_type(&DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8),
        )));
    }

    #[test]
    fn chained_grouping_sets_choose_smallest_containing_parent() {
        let masks = vec![
            vec![true, true, true],
            vec![false, false, false],
            vec![false, true, true],
            vec![false, false, true],
            vec![false, true, true],
        ];
        let levels = levels(&masks);
        assert_eq!(levels.len(), 4);
        assert_eq!(
            levels.iter().map(|l| l.parent).collect::<Vec<_>>(),
            vec![None, Some(0), Some(1), Some(2)]
        );
        assert_eq!(levels[2].ordinals, vec![0, 1]);
        assert_eq!(
            levels.iter().map(|l| l.consumers).collect::<Vec<_>>(),
            vec![1, 1, 1, 0]
        );
    }
}
