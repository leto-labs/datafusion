// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;
use crate::sorts::partitioned_topk::{
    PartitionedTopKExec, PartitionedTopKMode, WindowFnKind,
};
use crate::test::TestMemoryExec;
use crate::{Distribution, ExecutionPlan};
use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::expressions::col;
use futures::TryStreamExt;

#[rstest::rstest]
#[case::row_number(WindowFnKind::RowNumber, vec![1, 1], vec![1, 2])]
#[case::rank(WindowFnKind::Rank, vec![1, 1], vec![1, 1])]
#[case::dense_rank(WindowFnKind::DenseRank, vec![1, 1, 2], vec![1, 1, 2])]
#[tokio::test]
async fn global_partitioned_topk(
    #[case] kind: WindowFnKind,
    #[case] expected_values: Vec<i64>,
    #[case] expected_ranks: Vec<u64>,
    #[values(PartitionedTopKMode::Final, PartitionedTopKMode::Partial)]
    mode: PartitionedTopKMode,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![3, 1, 2, 1, 4]))],
    )?;
    let input = TestMemoryExec::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)?;
    let order =
        LexOrdering::new([PhysicalSortExpr::new_default(col("v", &schema)?)]).unwrap();
    let mut plan =
        PartitionedTopKExec::try_new_with_mode(input, order, 0, 2, kind, mode)?;
    if mode == PartitionedTopKMode::Final {
        assert!(matches!(
            plan.input_distribution_requirements()
                .into_per_child()
                .as_slice(),
            [Distribution::SinglePartition]
        ));
        plan = plan.with_ranking_field(Arc::new(Field::new(
            "rank",
            DataType::UInt64,
            false,
        )))?;
    }
    let runtime = Arc::new(TaskContext::default());
    let batches: Vec<RecordBatch> =
        plan.execute(0, Arc::clone(&runtime))?.try_collect().await?;
    let values: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_primitive::<arrow::datatypes::Int64Type>()
                .values()
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(values, expected_values);
    if mode == PartitionedTopKMode::Final {
        let ranks: Vec<u64> = batches
            .iter()
            .flat_map(|b| {
                b.column(1)
                    .as_primitive::<arrow::datatypes::UInt64Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(ranks, expected_ranks);
    } else {
        assert!(batches.iter().all(|b| b.num_columns() == 1));
    }
    assert_eq!(runtime.runtime_env().memory_pool.reserved(), 0);
    Ok(())
}

// A virtual NullArray makes the oversized input cheap and deterministic. In
// particular, chunking must not turn a rejected dimension into billions of rows
// of work before it reports the error.
#[cfg(target_pointer_width = "64")]
#[rstest::rstest]
#[case(WindowFnKind::RowNumber)]
#[case(WindowFnKind::Rank)]
#[case(WindowFnKind::DenseRank)]
#[tokio::test]
async fn partitioned_topk_rejects_oversized_input(
    #[case] kind: WindowFnKind,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Null, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow::array::NullArray::new(
            u32::MAX as usize + 1,
        ))],
    )?;
    let input = TestMemoryExec::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)?;
    let order =
        LexOrdering::new([PhysicalSortExpr::new_default(col("v", &schema)?)]).unwrap();
    let plan = PartitionedTopKExec::try_new(input, order, 0, 1, kind)?;
    let context = Arc::new(TaskContext::default());
    let error = plan
        .execute(0, Arc::clone(&context))?
        .try_collect::<Vec<RecordBatch>>()
        .await
        .unwrap_err();
    assert!(error.to_string().contains("u32 row indices"), "{error}");
    assert_eq!(context.runtime_env().memory_pool.reserved(), 0);
    Ok(())
}
