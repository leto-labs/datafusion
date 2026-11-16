// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow family SCN: Parquet float statistics leave out NaN. A row group or
//! page `k = [NaN, 1.0]` has min = max = 1.0, so statistics must not prune,
//! fully match, answer an aggregate or replace the column by a constant
//! unless they record `nan_count == 0`. Every query is compared with the same
//! rows in memory.
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, RecordBatch,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::WriterProperties;
use datafusion::prelude::*;

/// Row groups of two rows; one row per page.
const GROUPS: [[f64; 2]; 4] = [
    [f64::NAN, 1.0],
    [f64::NAN, f64::NAN],
    [-0.0, 0.0],
    [3.0, 4.0],
];

fn batch(float32: bool) -> RecordBatch {
    let values: Vec<f64> = GROUPS.iter().flatten().copied().collect();
    let (data_type, column): (DataType, ArrayRef) = if float32 {
        (
            DataType::Float32,
            Arc::new(Float32Array::from_iter_values(
                values.iter().map(|v| *v as f32),
            )),
        )
    } else {
        (DataType::Float64, Arc::new(Float64Array::from(values)))
    };
    let schema = Arc::new(Schema::new(vec![Field::new("k", data_type, false)]));
    RecordBatch::try_new(schema, vec![column]).unwrap()
}

/// Every value of the result's first column as bits, sorted.
async fn bits(ctx: &SessionContext, sql: &str) -> Vec<u64> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut out: Vec<u64> = batches
        .iter()
        .flat_map(|batch| {
            let column = batch.column(0);
            (0..column.len())
                .map(|i| {
                    if let Some(a) = column.as_any().downcast_ref::<Float32Array>() {
                        u64::from(a.value(i).to_bits())
                    } else {
                        column
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .unwrap()
                            .value(i)
                            .to_bits()
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    out.sort_unstable();
    out
}

/// Row groups pruned by statistics in the executed plan of `sql`.
async fn pruned_row_groups(ctx: &SessionContext, sql: &str) -> usize {
    use datafusion::physical_plan::metrics::MetricValue;
    use datafusion::physical_plan::{ExecutionPlan, collect};
    fn sum(plan: &Arc<dyn ExecutionPlan>) -> usize {
        let own = plan
            .metrics()
            .and_then(|metrics| metrics.sum_by_name("row_groups_pruned_statistics"))
            .map_or(0, |value| match value {
                MetricValue::PruningMetrics {
                    pruning_metrics, ..
                } => pruning_metrics.pruned(),
                other => other.as_usize(),
            });
        own + plan
            .children()
            .iter()
            .map(|child| sum(child))
            .sum::<usize>()
    }
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    sum(&plan)
}

#[tokio::test]
async fn nan_statistics_never_prune_match_or_answer() {
    for float32 in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.parquet");
        let data = batch(float32);
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2))
            .set_data_page_row_count_limit(1)
            .set_write_batch_size(1)
            .build();
        let mut writer = ArrowWriter::try_new(
            std::fs::File::create(&path).unwrap(),
            data.schema(),
            Some(props),
        )
        .unwrap();
        writer.write(&data).unwrap();
        writer.close().unwrap();

        let reference = SessionContext::new();
        reference
            .register_table(
                "t",
                Arc::new(
                    MemTable::try_new(data.schema(), vec![vec![data.clone()]]).unwrap(),
                ),
            )
            .unwrap();
        // Arrow 60 writers record NaN counts: the NaN-free float group [3, 4]
        // is still pruned by its statistics (`k < 2`), so a lost NaN count
        // cannot silently disable float pruning. The NaN groups and the zeros
        // group are kept.
        let pruning = SessionContext::new();
        pruning
            .register_parquet("t", path.to_str().unwrap(), ParquetReadOptions::default())
            .await
            .unwrap();
        assert_eq!(
            pruned_row_groups(&pruning, "SELECT k FROM t WHERE k < 2").await,
            1,
            "float32={float32}"
        );
        for pushdown in [false, true] {
            for page_index in [false, true] {
                for collect_statistics in [false, true] {
                    let mut config = SessionConfig::new();
                    let options = config.options_mut();
                    options.execution.parquet.pushdown_filters = pushdown;
                    options.execution.parquet.reorder_filters = pushdown;
                    options.execution.parquet.enable_page_index = page_index;
                    options.execution.collect_statistics = collect_statistics;
                    let ctx = SessionContext::new_with_config(config);
                    ctx.register_parquet(
                        "t",
                        path.to_str().unwrap(),
                        ParquetReadOptions::default(),
                    )
                    .await
                    .unwrap();
                    for sql in [
                        "SELECT k FROM t",
                        "SELECT k FROM t WHERE k < 2",
                        "SELECT k FROM t WHERE k > 1.5",
                        "SELECT k FROM t WHERE k = 1",
                        "SELECT k FROM t WHERE k = 0",
                        "SELECT k FROM t WHERE k <> 1",
                        "SELECT max(k) FROM t",
                        "SELECT min(k) FROM t",
                        "SELECT max(k) FROM t WHERE k < 2",
                    ] {
                        assert_eq!(
                            bits(&ctx, sql).await,
                            bits(&reference, sql).await,
                            "{sql}: float32={float32} pushdown={pushdown} \
                             page_index={page_index} statistics={collect_statistics}"
                        );
                    }
                }
            }
        }
    }
}
