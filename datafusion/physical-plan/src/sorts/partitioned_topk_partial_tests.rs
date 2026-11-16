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
    let expr = keys(&input, true);
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
