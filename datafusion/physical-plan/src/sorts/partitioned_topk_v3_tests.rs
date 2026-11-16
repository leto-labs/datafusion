// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
use super::*;
use crate::test::TestMemoryExec;
use arrow::array::{ArrayRef, AsArray, Int64Array, RecordBatch, StringViewArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_execution::{
    config::SessionConfig, memory_pool::GreedyMemoryPool, runtime_env::RuntimeEnvBuilder,
};
use datafusion_physical_expr::{PhysicalSortExpr, expressions::col};
use futures::TryStreamExt;

fn wide() -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(
        (0..16)
            .map(|i| Field::new(format!("c{i}"), DataType::Int64, false))
            .collect::<Vec<_>>(),
    ));
    let columns = (0..16)
        .map(|c| {
            Arc::new(Int64Array::from_iter_values((0..8192).map(|i| {
                if c == 0 {
                    i % 64
                } else if c == 1 {
                    (8191 - i) / 64
                } else {
                    i + c
                }
            }))) as ArrayRef
        })
        .collect();
    Ok(RecordBatch::try_new(schema, columns)?)
}
async fn run(
    batch: RecordBatch,
    kind: WindowFnKind,
    pool: usize,
    output: usize,
) -> Result<Vec<RecordBatch>> {
    let schema = batch.schema();
    let input: Arc<dyn ExecutionPlan> =
        TestMemoryExec::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)?;
    let expr = LexOrdering::new([
        PhysicalSortExpr::new_default(col("c0", &schema)?),
        PhysicalSortExpr::new_default(col("c1", &schema)?),
    ])
    .unwrap();
    let plan = PartitionedTopKExec::try_new(input, expr, 1, 2, kind)?;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(pool)))
        .build_arc()?;
    let context = Arc::new(
        TaskContext::default()
            .with_session_config(SessionConfig::new().with_batch_size(output))
            .with_runtime(Arc::clone(&runtime)),
    );
    let result = plan.execute(0, context)?.try_collect::<Vec<_>>().await;
    assert_eq!(runtime.memory_pool.reserved(), 0);
    result
}
#[rstest::rstest]
#[case(WindowFnKind::RowNumber)]
#[case(WindowFnKind::Rank)]
#[case(WindowFnKind::DenseRank)]
#[tokio::test]
async fn wide_normal_batches_fit_partition_share(
    #[case] kind: WindowFnKind,
) -> Result<()> {
    // One of sixteen 32 MiB shares of a 512 MiB pool; 1 MiB input batches.
    let batches = run(wide()?, kind, 32 << 20, 8192).await?;
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        128
    );
    Ok(())
}
#[rstest::rstest]
#[case(WindowFnKind::RowNumber)]
#[case(WindowFnKind::Rank)]
#[case(WindowFnKind::DenseRank)]
#[tokio::test]
async fn halving_equal_sized_batch_matches_single_gather(
    #[case] kind: WindowFnKind,
) -> Result<()> {
    let expected = run(wide()?, kind, 128 << 20, 8192).await?;
    let actual = run(
        wide()?,
        kind,
        if matches!(kind, WindowFnKind::DenseRank) {
            2 << 20
        } else {
            1 << 20
        },
        8192,
    )
    .await?;
    assert_eq!(
        arrow::util::pretty::pretty_format_batches(&expected)?.to_string(),
        arrow::util::pretty::pretty_format_batches(&actual)?.to_string()
    );
    Ok(())
}
#[rstest::rstest]
#[case(WindowFnKind::RowNumber, 3)]
#[case(WindowFnKind::Rank, 5)]
#[case(WindowFnKind::DenseRank, 7)]
#[tokio::test]
async fn global_desc_nulls_first_large_batch(
    #[case] kind: WindowFnKind,
    #[case] rows: usize,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
    let values = (0..100)
        .map(|i| if i < 5 { None } else { Some(i - 5) })
        .collect::<Vec<_>>();
    let input: Arc<dyn ExecutionPlan> = TestMemoryExec::try_new_exec(
        &[vec![RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(values))],
        )?]],
        Arc::clone(&schema),
        None,
    )?;
    let expr = LexOrdering::new([PhysicalSortExpr::new(
        col("v", &schema)?,
        SortOptions {
            descending: true,
            nulls_first: true,
        },
    )])
    .unwrap();
    let plan = PartitionedTopKExec::try_new(input, expr, 0, 3, kind)?
        .with_ranking_field(Arc::new(Field::new("r", DataType::UInt64, false)))?;
    let context = Arc::new(
        TaskContext::default()
            .with_session_config(SessionConfig::new().with_batch_size(2)),
    );
    let batches = plan.execute(0, context)?.try_collect::<Vec<_>>().await?;
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        rows
    );
    let all = arrow::compute::concat_batches(&plan.schema(), &batches)?;
    assert_eq!(all.column(0).null_count(), rows.min(5));
    if rows == 7 {
        assert_eq!(
            all.column(0)
                .as_primitive::<arrow::datatypes::Int64Type>()
                .value(5),
            94
        );
        assert_eq!(
            all.column(0)
                .as_primitive::<arrow::datatypes::Int64Type>()
                .value(6),
            93
        );
    }
    Ok(())
}
#[tokio::test]
async fn partial_admission_refusal_forwards_entire_batch_once() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8View, false),
        Field::new("v", DataType::Int64, false),
        Field::new("id", DataType::Int64, false),
    ]));
    let first = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringViewArray::from(vec!["old"])),
            Arc::new(Int64Array::from(vec![0])),
            Arc::new(Int64Array::from(vec![0])),
        ],
    )?;
    let key = "k".repeat(32 * 1024);
    let source = StringViewArray::from(vec![key.as_str()]);
    let keys = arrow::compute::take(
        &source,
        &arrow::array::UInt32Array::from(vec![0; 1024]),
        None,
    )?;
    let second = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            keys,
            Arc::new(Int64Array::from(vec![1; 1024])),
            Arc::new(Int64Array::from_iter_values(1..1025)),
        ],
    )?;
    let input: Arc<dyn ExecutionPlan> =
        TestMemoryExec::try_new_exec(&[vec![first, second]], Arc::clone(&schema), None)?;
    let expr = LexOrdering::new([
        PhysicalSortExpr::new_default(col("k", &schema)?),
        PhysicalSortExpr::new_default(col("v", &schema)?),
    ])
    .unwrap();
    let plan = PartitionedTopKExec::try_new_with_mode(
        input,
        expr,
        1,
        1,
        WindowFnKind::Rank,
        PartitionedTopKMode::Partial,
    )?;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(4 << 20)))
        .build_arc()?;
    let ctx = Arc::new(
        TaskContext::default()
            .with_session_config(SessionConfig::new().with_batch_size(32))
            .with_runtime(Arc::clone(&runtime)),
    );
    let batches = plan.execute(0, ctx)?.try_collect::<Vec<_>>().await?;
    let mut ids = batches
        .iter()
        .flat_map(|b| {
            b.column(2)
                .as_primitive::<arrow::datatypes::Int64Type>()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(ids, (0..1025).collect::<Vec<_>>());
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[test]
fn computed_keys_rejected_at_construction() -> Result<()> {
    use datafusion_physical_expr::expressions::{BinaryExpr, lit};
    let batch = wide()?;
    let schema = batch.schema();
    let input: Arc<dyn ExecutionPlan> =
        TestMemoryExec::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)?;
    let computed = Arc::new(BinaryExpr::new(
        col("c1", &schema)?,
        datafusion_expr::Operator::Plus,
        lit(1_i64),
    ));
    let ordering = LexOrdering::new([
        PhysicalSortExpr::new_default(col("c0", &schema)?),
        PhysicalSortExpr::new_default(computed),
    ])
    .unwrap();
    for mode in [PartitionedTopKMode::Final, PartitionedTopKMode::Partial] {
        assert!(
            PartitionedTopKExec::try_new_with_mode(
                Arc::clone(&input),
                ordering.clone(),
                1,
                2,
                WindowFnKind::Rank,
                mode
            )
            .is_err()
        );
    }
    Ok(())
}

#[tokio::test]
async fn partial_initial_reservation_refusal_forwards_input() -> Result<()> {
    let batch = wide()?;
    let schema = batch.schema();
    let input: Arc<dyn ExecutionPlan> =
        TestMemoryExec::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)?;
    let expr = LexOrdering::new([
        PhysicalSortExpr::new_default(col("c0", &schema)?),
        PhysicalSortExpr::new_default(col("c1", &schema)?),
    ])
    .unwrap();
    let plan = PartitionedTopKExec::try_new_with_mode(
        input,
        expr,
        1,
        2,
        WindowFnKind::Rank,
        PartitionedTopKMode::Partial,
    )?;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(1024)))
        .build_arc()?;
    let ctx = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let result = plan.execute(0, ctx)?.try_collect::<Vec<_>>().await?;
    assert_eq!(
        result.iter().map(RecordBatch::num_rows).sum::<usize>(),
        8192
    );
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}
