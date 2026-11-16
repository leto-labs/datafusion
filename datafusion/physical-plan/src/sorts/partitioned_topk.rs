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

//! [`PartitionedTopKExec`]: Top-K per partition operator
//!
//! For queries like:
//! ```sql
//! SELECT *, ROW_NUMBER() OVER (PARTITION BY pk ORDER BY val) as rn
//! FROM t WHERE rn <= N
//! ```
//!
//! Instead of sorting the entire dataset, this operator delegates to a
//! per-partition heap-of-K implementation (one variant for `ROW_NUMBER`
//! and a sibling variant for `RANK`), both of which maintain one heap per
//! distinct partition key while sharing a single [`arrow::row::RowConverter`],
//! [`MemoryReservation`](datafusion_execution::memory_pool::MemoryReservation),
//! and metrics set across all partitions, and emit only the top-K rows
//! per partition in sorted order `(partition_keys, order_keys)`.

use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::row::SortField;
use datafusion_common::Result;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::MemoryLimit;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr_common::sort_expr::LexOrdering;
use futures::StreamExt;
use futures::TryStreamExt;

use crate::execution_plan::{Boundedness, EmissionType};
use crate::metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use crate::topk::row_number::RowNumberTopK;
use crate::topk::{PartitionedTopKRank, build_sort_fields};
use crate::{ChildrenPropertiesMode, ReplaceChildrenOptions};
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, SendableRecordBatchStream, stream::RecordBatchStreamAdapter,
};

/// Which window function `PartitionedTopKExec` is optimizing.
///
/// Different ranking functions have different per-partition retention rules:
/// - [`RowNumber`](Self::RowNumber): exactly K rows per partition.
/// - [`Rank`](Self::Rank): K rows plus any rows tied at the boundary
///   ORDER BY value (RANK semantics — `WHERE rk <= K` may keep more
///   than K rows when ties straddle the boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFnKind {
    /// `ROW_NUMBER()` — keep exactly K rows per partition.
    RowNumber,
    /// `RANK()` — keep K rows plus any rows tied at the boundary.
    Rank,
}

/// Where a [`PartitionedTopKExec`] runs relative to the exchange that brings
/// the rows of one window partition together.
///
/// A row with global rank at most K also has rank at most K among any subset
/// of its window partition, so the same heap is a safe filter on every input
/// partition before the exchange. The consumer keeps the final operator, the
/// window and its filter above the exchange; the partial operator only removes
/// rows that can never pass them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartitionedTopKMode {
    /// Every row of a window partition is in one input partition. The output
    /// is ordered by `(partition_keys, order_keys)`.
    #[default]
    Final,
    /// Any input distribution. The output is a superset of the rows that the
    /// final operator keeps, in no declared order. When the input does not
    /// reduce (see [`PARTIAL_MIN_REDUCTION`]), or the heaps would take too much
    /// of a bounded pool (see [`PARTIAL_POOL_SHARE`] and
    /// [`PARTIAL_POOL_HEADROOM`]), the operator emits what it holds and passes
    /// the remaining input through unchanged.
    Partial,
}

/// Input rows of one partial operator partition between two reduction checks,
/// and before the first one.
pub const PARTIAL_PROBE_ROWS: usize = 1 << 16;

/// A partial operator keeps reducing only while it retains at most one row in
/// this many input rows.
pub const PARTIAL_MIN_REDUCTION: usize = 4;

/// The partial partitions of one operator together reserve at most one part in
/// this many of a bounded memory pool. The final operator holds its heaps at
/// the same time and neither spills, so the partial heaps must stay a small
/// part of what the plan may use. Checked after every input batch.
pub const PARTIAL_POOL_SHARE: usize = 4;

/// A partial partition stops reducing when less than one part in this many of
/// a bounded memory pool is free: the heaps are an optimization, and must not
/// take the memory that another operator of the plan needs. Checked before
/// every input batch, so that a refused reservation cannot fail the query in
/// the middle of a batch.
pub const PARTIAL_POOL_HEADROOM: usize = 8;

/// Per-partition Top-K operator for window function queries.
///
/// # Background
///
/// "Top K per partition" is a common analytics pattern used for queries such as
/// "find the top 3 products by revenue for each store". The (simplified) SQL
/// for such a query might be:
///
/// ```sql
/// SELECT * FROM (
///     SELECT *, ROW_NUMBER() OVER (PARTITION BY store ORDER BY revenue DESC) as rn
///     FROM sales
/// ) WHERE rn <= 3;
/// ```
///
/// The unoptimized physical plan would be:
///
/// ```text
/// FilterExec: rn <= 3
///   BoundedWindowAggExec: ROW_NUMBER() PARTITION BY [store] ORDER BY [revenue DESC]
///     SortExec: expr=[store ASC, revenue DESC]
///       DataSourceExec
/// ```
///
/// This plan sorts the **entire** dataset (O(N log N)), computes `ROW_NUMBER`
/// for **all** rows, and then filters to keep only the top K per partition.
/// With 10M rows, 1K partitions, and K=3, it sorts all 10M rows but only
/// keeps 3K.
///
/// # Optimization
///
/// `PartitionedTopKExec` supplies the window ordering while the consumer's
/// conservative rewrite retains the final filter. The optimized plan becomes:
///
/// ```text
/// FilterExec: rn <= 3
///   BoundedWindowAggExec: ROW_NUMBER() PARTITION BY [store] ORDER BY [revenue DESC]
///     PartitionedTopKExec: fetch=3, partition=[store], order=[revenue DESC]
///       DataSourceExec
/// ```
///
/// Instead of sorting the entire dataset, this operator reads unsorted input
/// and delegates to a per-partition heap-of-K implementation (`RowNumberTopK`
/// for `ROW_NUMBER` and `PartitionedTopKRank` for `RANK`), each maintaining
/// one heap per distinct partition key while sharing a single
/// [`arrow::row::RowConverter`] /
/// [`MemoryReservation`](datafusion_execution::memory_pool::MemoryReservation)
/// across all partitions, and emits only the top-K rows per partition in
/// sorted order `(partition_keys, order_keys)`.
///
/// Cost: O(N log K) time instead of O(N log N), and O(K × P × row_size)
/// memory where K = fetch, P = number of distinct partitions.
/// ## Why maintaining partition key order in output
/// Window functions do not require partition keys to be globally sorted, and
/// enforcing such ordering in the output can introduce unnecessary overhead.
/// However, the physical optimizer framework currently cannot express an
/// ordering that is only grouped by some keys while ordered by others. For
/// example:
///
///
/// # Example
///
/// For the query above with `fetch=3` and input:
///
/// ```text
/// store | revenue
/// ------|--------
///   A   |  100
///   B   |   50
///   A   |  200
///   B   |  150
///   A   |  300
///   A   |  400
/// ```
///
/// The operator maintains two heaps:
/// - **store=A**: keeps top-3 by revenue DESC → {400, 300, 200}, evicts 100
/// - **store=B**: keeps top-3 by revenue DESC → {150, 50} (only 2 rows)
///
/// Output (sorted by store ASC, revenue DESC):
///
/// ```text
/// store | revenue
/// ------|--------
///   A   |  400
///   A   |  300
///   A   |  200
///   B   |  150
///   B   |   50
/// ```
///
/// This is then passed to `BoundedWindowAggExec` which assigns
/// `ROW_NUMBER` 1, 2, 3 to each partition — all of which satisfy `rn <= 3`.
///
/// # Limitations
///
/// - The consumer activates this operator for `ROW_NUMBER` or `RANK`. An
///   empty partition prefix denotes a global ranking. `RANK` additionally requires a non-empty
///   `ORDER BY` (with an empty `ORDER BY`, every row ties at rank 1 and the
///   heap-of-K rewrite doesn't apply). Global RANK still needs its boundary
///   peers, unlike an ordinary SortExec with fetch.
/// - For very high cardinality partition keys (millions of distinct values),
///   both memory usage and runtime overhead can become significant. In such
///   cases, the sort-based plan can be more robust. Admission policy belongs
///   to the consumer's conservative optimizer rule.
#[derive(Debug, Clone)]
pub struct PartitionedTopKExec {
    /// Input execution plan (reads unsorted data)
    input: Arc<dyn ExecutionPlan>,
    /// Full sort expressions: `[partition_keys..., order_keys...]`.
    ///
    /// For `PARTITION BY store ORDER BY revenue DESC` with sort
    /// `[store ASC, revenue DESC]`, the first `partition_prefix_len`
    /// expressions are the partition keys (`[store ASC]`) and the
    /// remaining are the order-by keys (`[revenue DESC]`).
    expr: LexOrdering,
    /// Number of leading expressions in `expr` that define the partition
    /// key. For example, `PARTITION BY a, b` → `partition_prefix_len = 2`.
    partition_prefix_len: usize,
    /// Maximum number of rows to keep per partition (the K in "top-K").
    /// Derived from the filter predicate: `rn <= 3` → `fetch = 3`,
    /// `rn < 3` → `fetch = 2`.
    fetch: usize,
    /// Which window function this operator is optimizing. Selects the
    /// per-partition retention policy (see [`WindowFnKind`]).
    fn_kind: WindowFnKind,
    /// Whether this operator runs before or after the exchange on the
    /// partition keys (see [`PartitionedTopKMode`]).
    mode: PartitionedTopKMode,
    /// Execution metrics
    metrics_set: ExecutionPlanMetricsSet,
    /// Cached plan properties (output ordering, partitioning, etc.)
    cache: Arc<PlanProperties>,
}

impl PartitionedTopKExec {
    /// Create a new `PartitionedTopKExec`.
    ///
    /// # Arguments
    ///
    /// * `input` - The child execution plan providing unsorted input rows.
    /// * `expr` - Full sort ordering `[partition_keys..., order_keys...]`.
    ///   For `PARTITION BY pk ORDER BY val ASC`, this would be `[pk ASC, val ASC]`.
    /// * `partition_prefix_len` - Number of leading expressions in `expr`
    ///   that form the partition key. Zero means one global partition.
    /// * `fetch` - Maximum rows to retain per partition (the K in "top-K").
    /// * `fn_kind` - Which ranking window function this operator optimizes
    ///   ([`WindowFnKind::RowNumber`] or [`WindowFnKind::Rank`]).
    ///
    /// # Example
    ///
    /// ```text
    /// // For: ROW_NUMBER() OVER (PARTITION BY store ORDER BY revenue DESC) ... WHERE rn <= 5
    /// PartitionedTopKExec::try_new(
    ///     data_source,
    ///     LexOrdering([store ASC, revenue DESC]),
    ///     1,    // partition_prefix_len: 1 partition column (store)
    ///     5,    // fetch: keep top 5 per partition
    ///     WindowFnKind::RowNumber,
    /// )
    /// ```
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        expr: LexOrdering,
        partition_prefix_len: usize,
        fetch: usize,
        fn_kind: WindowFnKind,
    ) -> Result<Self> {
        Self::try_new_with_mode(
            input,
            expr,
            partition_prefix_len,
            fetch,
            fn_kind,
            PartitionedTopKMode::Final,
        )
    }

    /// Create a `PartitionedTopKExec` in `mode`; see [`Self::try_new`] for the
    /// other arguments.
    pub fn try_new_with_mode(
        input: Arc<dyn ExecutionPlan>,
        expr: LexOrdering,
        partition_prefix_len: usize,
        fetch: usize,
        fn_kind: WindowFnKind,
        mode: PartitionedTopKMode,
    ) -> Result<Self> {
        if fetch == 0 || partition_prefix_len >= expr.len() {
            return datafusion_common::plan_err!(
                "Top-K requires positive K and a valid ordering suffix"
            );
        }
        let cache = Self::compute_properties(&input, expr.clone(), mode)?;
        Ok(Self {
            input,
            expr,
            partition_prefix_len,
            fetch,
            fn_kind,
            mode,
            metrics_set: ExecutionPlanMetricsSet::new(),
            cache: Arc::new(cache),
        })
    }

    /// The partial operator of this (final) operator over `input`: the same
    /// keys, bound and function, before the exchange on the partition keys.
    pub fn partial(&self, input: Arc<dyn ExecutionPlan>) -> Result<Self> {
        Self::try_new_with_mode(
            input,
            self.expr.clone(),
            self.partition_prefix_len,
            self.fetch,
            self.fn_kind,
            PartitionedTopKMode::Partial,
        )
    }

    /// Returns the child execution plan.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Returns the full sort ordering `[partition_keys..., order_keys...]`.
    pub fn expr(&self) -> &LexOrdering {
        &self.expr
    }

    /// Returns the number of leading expressions in [`Self::expr`] that
    /// define the partition key.
    pub fn partition_prefix_len(&self) -> usize {
        self.partition_prefix_len
    }

    /// Returns the maximum number of rows retained per partition.
    pub fn fetch(&self) -> usize {
        self.fetch
    }

    /// Returns which window function this operator is optimizing.
    pub fn fn_kind(&self) -> WindowFnKind {
        self.fn_kind
    }

    /// Returns whether this operator runs before or after the exchange.
    pub fn mode(&self) -> PartitionedTopKMode {
        self.mode
    }

    /// Compute [`PlanProperties`] for this operator.
    ///
    /// The final output is sorted by `sort_exprs` (partition keys then order
    /// keys), uses the same partitioning as the input, emits all output at once
    /// (`EmissionType::Final`), and is bounded. The partial output declares no
    /// order, because a partition that stops reducing passes input through.
    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        sort_exprs: LexOrdering,
        mode: PartitionedTopKMode,
    ) -> Result<PlanProperties> {
        let mut eq_properties = input.equivalence_properties().clone();
        let emission = match mode {
            PartitionedTopKMode::Final => {
                eq_properties.reorder(sort_exprs)?;
                EmissionType::Final
            }
            PartitionedTopKMode::Partial => {
                eq_properties.clear_orderings();
                EmissionType::Both
            }
        };

        Ok(PlanProperties::new(
            eq_properties,
            input.output_partitioning().clone(),
            emission,
            Boundedness::Bounded,
        ))
    }
}

impl DisplayAs for PartitionedTopKExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        let fn_label = match self.fn_kind {
            WindowFnKind::RowNumber => "row_number",
            WindowFnKind::Rank => "rank",
        };
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let partition_exprs: Vec<String> = self.expr[..self.partition_prefix_len]
                    .iter()
                    .map(|e| format!("{}", e.expr))
                    .collect();
                let order_exprs: Vec<String> = self.expr[self.partition_prefix_len..]
                    .iter()
                    .map(|e| format!("{e}"))
                    .collect();
                write!(
                    f,
                    "PartitionedTopKExec: fn={}, fetch={}, partition=[{}], order=[{}]",
                    fn_label,
                    self.fetch,
                    partition_exprs.join(", "),
                    order_exprs.join(", "),
                )?;
                // The final display is unchanged; only the new mode is named.
                if self.mode == PartitionedTopKMode::Partial {
                    write!(f, ", mode=Partial")?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => {
                let partition_exprs: Vec<String> = self.expr[..self.partition_prefix_len]
                    .iter()
                    .map(|e| format!("{}", e.expr))
                    .collect();
                let order_exprs: Vec<String> = self.expr[self.partition_prefix_len..]
                    .iter()
                    .map(|e| format!("{e}"))
                    .collect();
                writeln!(f, "fn={fn_label}")?;
                writeln!(f, "fetch={}", self.fetch)?;
                writeln!(f, "partition=[{}]", partition_exprs.join(", "))?;
                writeln!(f, "order=[{}]", order_exprs.join(", "))?;
                if self.mode == PartitionedTopKMode::Partial {
                    writeln!(f, "mode=Partial")?;
                }
                Ok(())
            }
        }
    }
}

impl ExecutionPlan for PartitionedTopKExec {
    fn name(&self) -> &'static str {
        "PartitionedTopKExec"
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics_set.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.input_distribution_requirements().into_per_child()
    }

    fn input_distribution_requirements(&self) -> crate::InputDistributionRequirements {
        if self.mode == PartitionedTopKMode::Partial {
            return crate::InputDistributionRequirements::new(vec![
                Distribution::UnspecifiedDistribution,
            ]);
        }
        let partition_exprs: Vec<Arc<dyn PhysicalExpr>> = self.expr
            [..self.partition_prefix_len]
            .iter()
            .map(|e| Arc::clone(&e.expr))
            .collect();
        crate::InputDistributionRequirements::new(vec![if partition_exprs.is_empty() {
            Distribution::SinglePartition
        } else {
            Distribution::KeyPartitioned(partition_exprs)
        }])
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        assert_eq!(children.len(), 1);
        Ok(Arc::new(PartitionedTopKExec::try_new_with_mode(
            Arc::clone(&children[0]),
            self.expr.clone(),
            self.partition_prefix_len,
            self.fetch,
            self.fn_kind,
            self.mode,
        )?))
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        crate::apply_expression_roots(
            self.expr.iter().map(|sort_expr| &sort_expr.expr),
            f,
        )
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, Arc::clone(&context))?;
        let schema = input.schema();

        let partition_sort_fields =
            build_sort_fields(&self.expr[..self.partition_prefix_len], &schema)?;

        let partition_exprs: Vec<Arc<dyn PhysicalExpr>> = self.expr
            [..self.partition_prefix_len]
            .iter()
            .map(|e| Arc::clone(&e.expr))
            .collect();
        let order_expr: LexOrdering =
            LexOrdering::new(self.expr[self.partition_prefix_len..].iter().cloned())
                .expect("PartitionedTopKExec requires at least one order-by expression");
        let fetch = self.fetch;
        let fn_kind = self.fn_kind;
        let mode = self.mode;
        let partitions = self.cache.output_partitioning().partition_count();
        let batch_size = context.session_config().batch_size();
        let runtime = Arc::clone(&context.runtime_env());
        let metrics_set = self.metrics_set.clone();

        let stream = futures::stream::once(async move {
            do_partitioned_topk(
                partition,
                input,
                schema,
                partition_exprs,
                partition_sort_fields,
                order_expr,
                fetch,
                fn_kind,
                mode,
                partitions,
                batch_size,
                runtime,
                metrics_set,
            )
            .await
        })
        .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.input.schema(),
            stream,
        )))
    }
}

/// The per-partition state of one stream: the compact key-only `ROW_NUMBER`
/// heaps, or the batch-referencing heaps that also keep `RANK` peers.
enum TopKState {
    RowNumber(RowNumberTopK),
    Rank(PartitionedTopKRank),
}

impl TopKState {
    fn insert_batch(&mut self, batch: &arrow::record_batch::RecordBatch) -> Result<()> {
        match self {
            Self::RowNumber(state) => state.insert_batch(batch),
            Self::Rank(state) => state.insert_batch(batch),
        }
    }

    /// Rows that `emit` would return now.
    fn retained_rows(&self) -> usize {
        match self {
            Self::RowNumber(state) => state.retained_rows(),
            Self::Rank(state) => state.retained_rows(),
        }
    }

    /// Bytes that the state has reserved in the pool for what it retains.
    fn reserved_bytes(&self) -> usize {
        match self {
            Self::RowNumber(state) => state.reserved_bytes(),
            Self::Rank(state) => state.reserved_bytes(),
        }
    }

    /// The partition's one `output_rows` metric, which `emit` also records.
    fn output_rows(&self) -> crate::metrics::Count {
        match self {
            Self::RowNumber(state) => state.output_rows(),
            Self::Rank(state) => state.output_rows(),
        }
    }

    fn emit(self) -> Result<SendableRecordBatchStream> {
        match self {
            Self::RowNumber(state) => state.emit(),
            Self::Rank(state) => state.emit(),
        }
    }
}

/// The memory bounds of one partial partition in a bounded pool.
struct PartialBounds {
    /// The pool's limit.
    pool: usize,
    /// The most bytes that this partition's heaps may reserve.
    partition: usize,
}

impl PartialBounds {
    /// The bounds of one of `partitions` partial partitions in `runtime`'s
    /// pool; `None` for a pool without a known limit, which cannot refuse.
    fn of(runtime: &RuntimeEnv, partitions: usize) -> Option<Self> {
        match runtime.memory_pool.memory_limit() {
            MemoryLimit::Finite(pool) => Some(Self {
                pool,
                partition: pool / PARTIAL_POOL_SHARE / partitions.max(1),
            }),
            MemoryLimit::Infinite | MemoryLimit::Unknown => None,
        }
    }

    /// Whether the pool has less than its headroom free.
    fn pool_is_tight(&self, runtime: &RuntimeEnv) -> bool {
        runtime.memory_pool.reserved() > self.pool - self.pool / PARTIAL_POOL_HEADROOM
    }
}

/// Ends the reduction of a partial partition: the stream of what `state`
/// holds, then `pending` (a batch that was not inserted), then the rest of
/// `input`, unchanged. Passed rows count in the partition's `output_rows`.
fn pass_through(
    state: TopKState,
    pending: Option<arrow::record_batch::RecordBatch>,
    input: SendableRecordBatchStream,
    schema: SchemaRef,
    partition_id: usize,
    metrics_set: &ExecutionPlanMetricsSet,
) -> Result<SendableRecordBatchStream> {
    MetricBuilder::new(metrics_set)
        .counter("partial_passthrough", partition_id)
        .add(1);
    let passed = state.output_rows();
    let rest = futures::stream::iter(pending.map(Ok))
        .chain(input)
        .inspect_ok(move |batch| passed.add(batch.num_rows()));
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema,
        state.emit()?.chain(rest),
    )))
}

/// Read all input, feed each batch into a per-partition top-K state
/// (either [`RowNumberTopK`] for `ROW_NUMBER` or
/// [`PartitionedTopKRank`] for `RANK`), then emit results ordered by
/// `(partition_keys, order_keys)`.
///
/// # Phases
///
/// 1. **Accumulation** — forward each input `RecordBatch` to the
///    per-partition state's `insert_batch`. The `RowConverter` for
///    ORDER BY columns, the operator's `MemoryReservation`, and the
///    `TopKMetrics` are shared across all distinct partition keys for
///    this operator instance.
///
/// 2. **Emission** — `emit` drains all per-partition heaps in sorted
///    partition-key order, returning a coalesced batch stream. For
///    `RANK`, boundary-tied rows are materialized and emitted after
///    each partition's heap rows.
///
/// A partial operator checks its reduction every [`PARTIAL_PROBE_ROWS`] input
/// rows. When it retains more than one row in [`PARTIAL_MIN_REDUCTION`], the
/// heaps cost more than they remove (many window partitions, or many boundary
/// peers): it emits what it holds and passes the remaining input through. It
/// does the same when its heaps reserve more than its share of a bounded pool
/// ([`PARTIAL_POOL_SHARE`], after every batch), and before it inserts a batch
/// into a pool that is nearly full ([`PARTIAL_POOL_HEADROOM`]). Every passed
/// row is a row the final operator would have received without the partial
/// operator, so each of these only gives up the reduction.
///
/// # Cost
///
/// - Time: O(N log K) where N = total rows, K = fetch
/// - Memory: O(K × P × row_size) where P = number of distinct partitions
///   plus, for RANK, the boundary ties' rows
#[expect(clippy::too_many_arguments)]
async fn do_partitioned_topk(
    partition_id: usize,
    mut input: SendableRecordBatchStream,
    schema: SchemaRef,
    partition_exprs: Vec<Arc<dyn PhysicalExpr>>,
    partition_sort_fields: Vec<SortField>,
    order_expr: LexOrdering,
    fetch: usize,
    fn_kind: WindowFnKind,
    mode: PartitionedTopKMode,
    partitions: usize,
    batch_size: usize,
    runtime: Arc<RuntimeEnv>,
    metrics_set: ExecutionPlanMetricsSet,
) -> Result<SendableRecordBatchStream> {
    // Key-only ROW_NUMBER can retain compact encoded values and coalesce output
    // across groups. Arbitrary payloads keep the checked family N batch path.
    let key_only = schema.fields().iter().enumerate().all(|(i, _)| {
        partition_exprs
            .iter()
            .chain(order_expr.iter().map(|e| &e.expr))
            .any(|e| {
                e.downcast_ref::<datafusion_physical_expr::expressions::Column>()
                    .is_some_and(|c| c.index() == i)
            })
    });
    let mut state = if fn_kind == WindowFnKind::RowNumber && key_only {
        TopKState::RowNumber(RowNumberTopK::try_new(
            partition_id,
            Arc::clone(&schema),
            partition_exprs,
            partition_sort_fields,
            order_expr,
            fetch,
            batch_size,
            &runtime,
            &metrics_set,
        )?)
    } else {
        let mut rank = PartitionedTopKRank::try_new(
            partition_id,
            Arc::clone(&schema),
            partition_exprs,
            partition_sort_fields,
            order_expr,
            fetch,
            batch_size,
            &runtime,
            &metrics_set,
        )?;
        rank.retain_peers(fn_kind == WindowFnKind::Rank);
        TopKState::Rank(rank)
    };
    if mode == PartitionedTopKMode::Final {
        while let Some(batch) = input.next().await {
            state.insert_batch(&batch?)?;
            tokio::task::yield_now().await;
        }
        drop(input);
        return state.emit();
    }
    let bounds = PartialBounds::of(&runtime, partitions);
    let peak = MetricBuilder::new(&metrics_set)
        .peak_memory_usage("partial_peak_bytes", partition_id);
    let mut seen: usize = 0;
    let mut next_check = PARTIAL_PROBE_ROWS;
    while let Some(batch) = input.next().await {
        let batch = batch?;
        if bounds
            .as_ref()
            .is_some_and(|bounds| bounds.pool_is_tight(&runtime))
        {
            // Nothing of this batch is in the heaps yet: forward it whole.
            return pass_through(
                state,
                Some(batch),
                input,
                schema,
                partition_id,
                &metrics_set,
            );
        }
        seen = seen.saturating_add(batch.num_rows());
        state.insert_batch(&batch)?;
        drop(batch);
        let reserved = state.reserved_bytes();
        peak.set_max(reserved);
        let over_share = bounds
            .as_ref()
            .is_some_and(|bounds| reserved > bounds.partition);
        let mut not_reducing = false;
        if seen >= next_check {
            not_reducing =
                state.retained_rows().saturating_mul(PARTIAL_MIN_REDUCTION) > seen;
            next_check = seen.saturating_add(PARTIAL_PROBE_ROWS);
        }
        if over_share || not_reducing {
            return pass_through(state, None, input, schema, partition_id, &metrics_set);
        }
        tokio::task::yield_now().await;
    }
    drop(input);
    state.emit()
}

#[cfg(test)]
#[path = "partitioned_topk_partial_tests.rs"]
mod partial_tests;
