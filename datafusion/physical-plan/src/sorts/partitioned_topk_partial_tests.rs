// Modified by the Longbow authors; see third_party/README.md.

//! The partial operator is a filter that never removes a row the final
//! operator keeps, whatever the input distribution.
use super::*;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::pretty::pretty_format_batches;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;

use crate::repartition::RepartitionExec;
use crate::test::TestMemoryExec;

fn schema(payload: bool) -> SchemaRef {
    let mut fields = vec![
        Field::new("k", DataType::Utf8, true),
        Field::new("v", DataType::Int64, true),
    ];
    if payload {
        fields.push(Field::new("w", DataType::Int64, false));
    }
    Arc::new(Schema::new(fields))
}

/// `partitions` input partitions of `rows` rows each. `groups` window
/// partitions (one of them NULL) are spread over every input partition, and
/// `values` distinct order values (with NULLs) make boundary peers.
fn source(
    partitions: usize,
    rows: usize,
    groups: usize,
    values: i64,
    payload: bool,
) -> Arc<dyn ExecutionPlan> {
    let schema = schema(payload);
    let batches: Vec<Vec<RecordBatch>> = (0..partitions)
        .map(|p| {
            (0..rows)
                .step_by(1000)
                .map(|start| {
                    let range = start..(start + 1000).min(rows);
                    let seed = |i: usize| (i * 2654435761 + p * 40503) % 1_000_003;
                    let mut columns: Vec<arrow::array::ArrayRef> = vec![
                        Arc::new(StringArray::from_iter(range.clone().map(|i| {
                            let group = seed(i) % groups;
                            (group != 0).then(|| format!("g{group}"))
                        }))),
                        Arc::new(Int64Array::from_iter(range.clone().map(|i| {
                            (seed(i) % 17 != 0).then_some((seed(i) / 7) as i64 % values)
                        }))),
                    ];
                    if payload {
                        columns.push(Arc::new(Int64Array::from_iter_values(
                            range.map(|i| (p * rows + i) as i64),
                        )));
                    }
                    RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
                })
                .collect()
        })
        .collect();
    TestMemoryExec::try_new_exec(&batches, schema, None).unwrap()
}

fn keys(input: &Arc<dyn ExecutionPlan>, descending: bool) -> LexOrdering {
    let schema = input.schema();
    LexOrdering::new(vec![
        PhysicalSortExpr::new_default(col("k", &schema).unwrap()),
        PhysicalSortExpr::new(
            col("v", &schema).unwrap(),
            SortOptions::new(descending, descending),
        ),
    ])
    .unwrap()
}

fn final_topk(
    input: Arc<dyn ExecutionPlan>,
    fetch: usize,
    kind: WindowFnKind,
    partial: bool,
) -> Arc<dyn ExecutionPlan> {
    final_topk_ordered(input, fetch, kind, partial, true)
}

/// The final heap over a hash exchange on `k`, with or without the partial
/// heap below the exchange; `v` descending with NULLs first, or ascending
/// with NULLs last.
fn final_topk_ordered(
    input: Arc<dyn ExecutionPlan>,
    fetch: usize,
    kind: WindowFnKind,
    partial: bool,
    descending: bool,
) -> Arc<dyn ExecutionPlan> {
    let expr = keys(&input, descending);
    let input: Arc<dyn ExecutionPlan> = if partial {
        Arc::new(
            PartitionedTopKExec::try_new_with_mode(
                input,
                expr.clone(),
                1,
                fetch,
                kind,
                PartitionedTopKMode::Partial,
            )
            .unwrap(),
        )
    } else {
        input
    };
    let hash = Partitioning::Hash(vec![Arc::clone(&expr[0].expr)], 3);
    let exchange = Arc::new(RepartitionExec::try_new(input, hash).unwrap());
    Arc::new(PartitionedTopKExec::try_new(exchange, expr, 1, fetch, kind).unwrap())
}

async fn rows(plan: Arc<dyn ExecutionPlan>, context: Arc<TaskContext>) -> Vec<String> {
    let batches = crate::collect(plan, context).await.unwrap();
    let text = pretty_format_batches(&batches).unwrap().to_string();
    let mut rows: Vec<String> = text.lines().skip(3).map(str::to_owned).collect();
    rows.retain(|line| line.starts_with('|'));
    rows.sort();
    rows
}

fn metric(plan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
    plan.metrics()
        .and_then(|metrics| metrics.sum_by_name(name))
        .map_or(0, |value| value.as_usize())
}

#[test]
fn partial_requires_no_distribution_and_declares_no_order() {
    let input = source(4, 100, 5, 9, true);
    let expr = keys(&input, true);
    let partial = Arc::new(
        PartitionedTopKExec::try_new_with_mode(
            Arc::clone(&input),
            expr.clone(),
            1,
            3,
            WindowFnKind::Rank,
            PartitionedTopKMode::Partial,
        )
        .unwrap(),
    );
    assert!(matches!(
        partial.required_input_distribution().as_slice(),
        [Distribution::UnspecifiedDistribution]
    ));
    assert!(partial.properties().output_ordering().is_none());
    assert_eq!(
        partial.properties().output_partitioning().partition_count(),
        4
    );
    let text = crate::displayable(partial.as_ref()).one_line().to_string();
    assert!(text.trim_end().ends_with("mode=Partial"), "{text}");

    let final_ =
        PartitionedTopKExec::try_new(Arc::clone(&input), expr, 1, 3, WindowFnKind::Rank)
            .unwrap();
    assert_eq!(final_.mode(), PartitionedTopKMode::Final);
    assert!(final_.properties().output_ordering().is_some());
    assert!(matches!(
        final_.required_input_distribution().as_slice(),
        [Distribution::KeyPartitioned(_)]
    ));
    let text = crate::displayable(&final_).one_line().to_string();
    assert!(
        !text.contains("mode="),
        "the final display is unchanged: {text}"
    );

    // Rebuilding keeps the mode: an optimizer rule must not turn the filter
    // into an operator that claims an order and a distribution.
    let rebuilt = (partial as Arc<dyn ExecutionPlan>)
        .with_new_children(vec![input])
        .unwrap();
    let rebuilt = rebuilt.downcast_ref::<PartitionedTopKExec>().unwrap();
    assert_eq!(rebuilt.mode(), PartitionedTopKMode::Partial);
    assert!(rebuilt.properties().output_ordering().is_none());
    assert_eq!(
        final_.partial(Arc::clone(rebuilt.input())).unwrap().mode(),
        PartitionedTopKMode::Partial
    );
}

#[tokio::test]
async fn partial_then_final_equals_final_alone() {
    // RANK keeps every boundary peer, so payload rows are deterministic.
    // ROW_NUMBER picks one of several equal rows; its key-only schema makes
    // those rows identical.
    for (kind, payload) in [(WindowFnKind::Rank, true), (WindowFnKind::RowNumber, false)]
    {
        for (groups, values, fetch) in
            [(5, 9, 3), (5, 1000, 100), (700, 50, 2), (1, 40, 7)]
        {
            let context = Arc::new(TaskContext::default());
            let expected = rows(
                final_topk(source(4, 3000, groups, values, payload), fetch, kind, false),
                Arc::clone(&context),
            )
            .await;
            let actual = rows(
                final_topk(source(4, 3000, groups, values, payload), fetch, kind, true),
                Arc::clone(&context),
            )
            .await;
            assert!(!expected.is_empty());
            assert_eq!(expected, actual, "{kind:?} groups={groups} fetch={fetch}");
            assert_eq!(context.runtime_env().memory_pool.reserved(), 0);
        }
    }
}

#[tokio::test]
async fn partial_reduces_few_window_partitions_on_every_input_partition() {
    let input = source(4, 2 * PARTIAL_PROBE_ROWS, 5, 1_000_000, true);
    // Ascending with NULLs last: descending would rank every NULL order value
    // first, and RANK keeps all of those peers.
    let expr = keys(&input, false);
    let partial: Arc<dyn ExecutionPlan> = Arc::new(
        PartitionedTopKExec::try_new_with_mode(
            input,
            expr,
            1,
            10,
            WindowFnKind::Rank,
            PartitionedTopKMode::Partial,
        )
        .unwrap(),
    );
    let context = Arc::new(TaskContext::default());
    let batches = crate::collect_partitioned(Arc::clone(&partial), Arc::clone(&context))
        .await
        .unwrap();
    assert_eq!(batches.len(), 4);
    for partition in &batches {
        let kept: usize = partition.iter().map(RecordBatch::num_rows).sum();
        // Five window partitions of ten rows, plus peers of each boundary.
        assert!((50..200).contains(&kept), "{kept}");
    }
    assert_eq!(metric(&partial, "partial_passthrough"), 0);
    assert_eq!(context.runtime_env().memory_pool.reserved(), 0);
}

#[tokio::test]
async fn partial_passes_input_through_when_it_does_not_reduce() {
    // Many window partitions of a few rows: the heaps would keep every row.
    for kind in [WindowFnKind::Rank, WindowFnKind::RowNumber] {
        let payload = kind == WindowFnKind::Rank;
        let total = 2 * PARTIAL_PROBE_ROWS + 500;
        let groups = PARTIAL_PROBE_ROWS;
        let input = source(2, total, groups, 1_000_000, payload);
        let expr = keys(&input, false);
        let partial: Arc<dyn ExecutionPlan> = Arc::new(
            PartitionedTopKExec::try_new_with_mode(
                Arc::clone(&input),
                expr,
                1,
                2,
                kind,
                PartitionedTopKMode::Partial,
            )
            .unwrap(),
        );
        let context = Arc::new(TaskContext::default());
        let batches =
            crate::collect_partitioned(Arc::clone(&partial), Arc::clone(&context))
                .await
                .unwrap();
        assert_eq!(metric(&partial, "partial_passthrough"), 2, "{kind:?}");
        for partition in &batches {
            let kept: usize = partition.iter().map(RecordBatch::num_rows).sum();
            // Everything after the probe passes; the probe itself was reduced
            // to at most two rows of each window partition it saw.
            assert!(kept >= total - PARTIAL_PROBE_ROWS - 1000, "{kept}");
            assert!(kept <= total, "{kept}");
        }
        assert_eq!(
            partial.metrics().unwrap().output_rows(),
            Some(
                batches
                    .iter()
                    .flatten()
                    .map(RecordBatch::num_rows)
                    .sum::<usize>()
            )
        );
        // Passed rows count in the partition's one `output_rows` metric.
        assert_eq!(output_rows_metrics(&partial), 2, "{kind:?}");
        // The final result is the one without the partial operator.
        let expected = rows(
            final_topk(source(2, total, groups, 1_000_000, payload), 2, kind, false),
            Arc::clone(&context),
        )
        .await;
        let actual = rows(
            final_topk(source(2, total, groups, 1_000_000, payload), 2, kind, true),
            Arc::clone(&context),
        )
        .await;
        assert_eq!(expected.len(), actual.len());
        assert!(
            expected == actual,
            "{kind:?}: partial changed the final rows"
        );
        assert_eq!(context.runtime_env().memory_pool.reserved(), 0);
    }
}

/// How many `output_rows` metrics `plan` has registered.
fn output_rows_metrics(plan: &Arc<dyn ExecutionPlan>) -> usize {
    plan.metrics()
        .unwrap()
        .iter()
        .filter(|metric| {
            matches!(metric.value(), crate::metrics::MetricValue::OutputRows(_))
        })
        .count()
}

type Triple = (Option<String>, Option<i64>, i64);

/// `(k, v, w)` of every row; `w` is unique in the input.
fn triples(batches: &[RecordBatch]) -> Vec<Triple> {
    use arrow::array::Array;
    let mut rows = Vec::new();
    for batch in batches {
        let k = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let v = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let w = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            rows.push((
                k.is_valid(i).then(|| k.value(i).to_owned()),
                v.is_valid(i).then(|| v.value(i)),
                w.value(i),
            ));
        }
    }
    rows
}

/// Fails unless `actual` is a valid `ROW_NUMBER <= fetch` answer over `input`:
/// every row is a distinct input row, and each window partition has
/// `min(fetch, its size)` rows whose order values are a prefix of that
/// partition's sorted order values. Which of several equal rows at the
/// boundary are kept is free.
fn assert_valid_row_number_prefix(
    input: &[Triple],
    actual: &[Triple],
    fetch: usize,
    descending: bool,
) {
    use std::collections::{BTreeMap, HashMap, HashSet};
    // `v DESC NULLS FIRST`, or `v ASC NULLS LAST`.
    let rank = |v: &Option<i64>| match (v, descending) {
        (None, true) => (0, 0),
        (Some(v), true) => (1, -*v),
        (Some(v), false) => (0, *v),
        (None, false) => (1, 0),
    };
    let by_id: HashMap<i64, &Triple> = input.iter().map(|row| (row.2, row)).collect();
    assert_eq!(by_id.len(), input.len(), "the fixture's payload is unique");
    let mut seen = HashSet::new();
    for row in actual {
        assert!(seen.insert(row.2), "row {} is returned twice", row.2);
        assert_eq!(by_id.get(&row.2), Some(&row), "not an input row");
    }
    let mut expected: BTreeMap<Option<String>, Vec<(i32, i64)>> = BTreeMap::new();
    for row in input {
        expected
            .entry(row.0.clone())
            .or_default()
            .push(rank(&row.1));
    }
    let mut kept: BTreeMap<Option<String>, Vec<(i32, i64)>> = BTreeMap::new();
    for row in actual {
        kept.entry(row.0.clone()).or_default().push(rank(&row.1));
    }
    assert_eq!(
        kept.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>(),
        "every window partition is present"
    );
    let mut boundary_ties = 0;
    for (key, mut values) in expected {
        values.sort_unstable();
        let prefix = &values[..fetch.min(values.len())];
        if values.len() > fetch && values[fetch] == values[fetch - 1] {
            boundary_ties += 1;
        }
        let mut actual = kept.remove(&key).unwrap();
        actual.sort_unstable();
        assert_eq!(actual, prefix, "partition {key:?}");
    }
    assert!(
        boundary_ties > 0,
        "the fixture has equal rows at the boundary"
    );
}

#[tokio::test]
async fn row_number_with_payload_keeps_a_valid_prefix_across_boundary_ties() {
    // A payload column takes the batch-referencing heap with peers off. Three
    // order values (and NULLs) put many equal rows at every K-th position, so
    // the partial and the final heap are free to keep different ones.
    for descending in [true, false] {
        for fetch in [1, 5, 40] {
            let context = Arc::new(TaskContext::default());
            let input = triples(
                &crate::collect(source(4, 3000, 4, 3, true), Arc::clone(&context))
                    .await
                    .unwrap(),
            );
            for partial in [false, true] {
                let plan = final_topk_ordered(
                    source(4, 3000, 4, 3, true),
                    fetch,
                    WindowFnKind::RowNumber,
                    partial,
                    descending,
                );
                let actual =
                    triples(&crate::collect(plan, Arc::clone(&context)).await.unwrap());
                assert_eq!(actual.len(), 4 * fetch, "partial={partial}");
                assert_valid_row_number_prefix(&input, &actual, fetch, descending);
            }
            assert_eq!(context.runtime_env().memory_pool.reserved(), 0);
        }
    }
}

/// `partitions` input partitions of `batches` batches of 1,000 rows with a
/// 1 KiB payload each, so that a retained row pins about 1 MiB of its batch.
fn wide_source(partitions: usize, batches: usize) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, true),
        Field::new("v", DataType::Int64, true),
        Field::new("w", DataType::Int64, false),
        Field::new("pad", DataType::Utf8, false),
    ]));
    let data: Vec<Vec<RecordBatch>> = (0..partitions)
        .map(|p| {
            (0..batches)
                .map(|b| {
                    let row = |i: usize| (p * batches + b) * 1000 + i;
                    RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![
                            Arc::new(StringArray::from_iter_values(
                                (0..1000).map(|i| format!("g{}", row(i) % 5)),
                            )),
                            Arc::new(Int64Array::from_iter_values(
                                (0..1000)
                                    .map(|i| ((row(i) * 2654435761) % 1_000_003) as i64),
                            )),
                            Arc::new(Int64Array::from_iter_values(
                                (0..1000).map(|i| row(i) as i64),
                            )),
                            Arc::new(StringArray::from_iter_values(
                                (0..1000).map(|i| format!("{:01024}", row(i))),
                            )),
                        ],
                    )
                    .unwrap()
                })
                .collect()
        })
        .collect();
    TestMemoryExec::try_new_exec(&data, schema, None).unwrap()
}

fn bounded_context(bytes: usize) -> Arc<TaskContext> {
    use datafusion_execution::memory_pool::GreedyMemoryPool;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(bytes)))
        .build_arc()
        .unwrap();
    Arc::new(TaskContext::default().with_runtime(runtime))
}

fn partial_rank(input: Arc<dyn ExecutionPlan>, fetch: usize) -> Arc<dyn ExecutionPlan> {
    let expr = keys(&input, false);
    Arc::new(
        PartitionedTopKExec::try_new_with_mode(
            input,
            expr,
            1,
            fetch,
            WindowFnKind::Rank,
            PartitionedTopKMode::Partial,
        )
        .unwrap(),
    )
}

#[tokio::test]
async fn partial_stays_within_its_share_of_a_small_pool() {
    // Two partitions of 12 MiB. The heaps keep 1,000 rows of each of five
    // window partitions, which pins every source batch: without a byte bound
    // each partial partition would hold all 12 MiB until its input ends.
    const POOL: usize = 64 << 20;
    let share = POOL / PARTIAL_POOL_SHARE / 2;
    let context = bounded_context(POOL);
    let partial = partial_rank(wide_source(2, 12), 1000);
    let batches = crate::collect_partitioned(Arc::clone(&partial), Arc::clone(&context))
        .await
        .unwrap();
    assert_eq!(metric(&partial, "partial_passthrough"), 2);
    // Each partition stopped at the first batch that took it over its share.
    let peak = metric(&partial, "partial_peak_bytes");
    assert!(peak > 2 * share, "{peak}");
    assert!(peak < 2 * (share + (2 << 20)), "{peak}");
    for partition in &batches {
        let rows: usize = partition.iter().map(RecordBatch::num_rows).sum();
        assert!(rows > 5000 && rows <= 12_000, "{rows}");
    }
    assert_eq!(output_rows_metrics(&partial), 2);
    drop(batches);
    assert_eq!(context.runtime_env().memory_pool.reserved(), 0);

    // The whole plan succeeds in the pool and returns the rows of the plan
    // without the partial heap.
    let plan = |with_partial| {
        final_topk_ordered(
            wide_source(2, 12),
            1000,
            WindowFnKind::Rank,
            with_partial,
            false,
        )
    };
    let sorted = |mut rows: Vec<Triple>| {
        rows.sort();
        rows
    };
    let expected = sorted(triples(
        &crate::collect(plan(false), Arc::clone(&context))
            .await
            .unwrap(),
    ));
    let actual = sorted(triples(
        &crate::collect(plan(true), Arc::clone(&context))
            .await
            .unwrap(),
    ));
    assert_eq!(expected.len(), 5000);
    assert!(expected == actual, "the byte bound changed the final rows");
    assert_eq!(context.runtime_env().memory_pool.reserved(), 0);
}

#[tokio::test]
async fn partial_passes_through_when_the_pool_is_nearly_full() {
    use datafusion_execution::memory_pool::MemoryConsumer;
    const POOL: usize = 64 << 20;
    let context = bounded_context(POOL);
    // Another operator of the plan holds all but a sixteenth of the pool.
    let other = MemoryConsumer::new("other").register(&context.runtime_env().memory_pool);
    other.try_grow(POOL - POOL / 16).unwrap();
    let partial = partial_rank(wide_source(2, 3), 10);
    let batches = crate::collect_partitioned(Arc::clone(&partial), Arc::clone(&context))
        .await
        .unwrap();
    // No batch entered a heap, so no reservation could be refused, and every
    // input row is forwarded once.
    assert_eq!(metric(&partial, "partial_passthrough"), 2);
    assert_eq!(metric(&partial, "partial_peak_bytes"), 0);
    let mut ids: Vec<i64> = batches
        .iter()
        .flat_map(|partition| triples(partition))
        .map(|row| row.2)
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..6000).collect::<Vec<i64>>());
    assert_eq!(partial.metrics().unwrap().output_rows(), Some(6000));
    assert_eq!(output_rows_metrics(&partial), 2);
    drop(batches);
    drop(other);
    assert_eq!(context.runtime_env().memory_pool.reserved(), 0);
}
