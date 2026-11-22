// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt::Write;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Int32Array, LargeStringArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::{JoinSide, JoinType, NullEquality, Result, ScalarValue};
use datafusion_execution::config::SessionConfig;
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
use futures::TryStreamExt;

use super::{memory_input, metric, no_reversal, spill_config, task_context};
use crate::ExecutionPlan;
use crate::joins::utils::{ColumnIndex, JoinFilter};
use crate::joins::{HashJoinExec, HashJoinExecBuilder, PartitionMode};

const LIMIT: usize = 8 * 1024 * 1024;

fn inputs() -> Result<(Vec<RecordBatch>, RecordBatch)> {
    let wide_b = "b".repeat(16 * 1024);
    let n = 2048;
    let keys: Vec<Option<i32>> = (0..n)
        .map(|i| match i {
            0..1024 => Some(7),
            1024..1536 => Some(9),
            _ => None,
        })
        .collect();
    let build = RecordBatch::try_from_iter([
        ("k", Arc::new(Int32Array::from(keys)) as _),
        (
            "bid",
            Arc::new(Int32Array::from_iter_values(0..n as i32)) as _,
        ),
        (
            "bw",
            Arc::new(LargeStringArray::from_iter_values(std::iter::repeat_n(
                wide_b.as_str(),
                n,
            ))) as _,
        ),
    ])?;
    let build = (0..n).step_by(256).map(|s| build.slice(s, 256)).collect();
    let wide_p = "p".repeat(16 * 1024);
    let pk = vec![Some(7), Some(7), Some(7), Some(7), Some(8), None, None];
    let rows = pk.len();
    let probe = RecordBatch::try_from_iter([
        ("k", Arc::new(Int32Array::from(pk)) as _),
        (
            "pid",
            Arc::new(Int32Array::from_iter_values(0..rows as i32)) as _,
        ),
        (
            "pw",
            Arc::new(LargeStringArray::from_iter_values(std::iter::repeat_n(
                wide_p.as_str(),
                rows,
            ))) as _,
        ),
    ])?;
    Ok((build, probe))
}

fn filter() -> JoinFilter {
    // bid < 900 AND pid != 1
    let left = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("bid", 0)),
        Operator::Lt,
        Arc::new(Literal::new(ScalarValue::Int32(Some(900)))),
    ));
    let right = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("pid", 1)),
        Operator::NotEq,
        Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
    ));
    JoinFilter::new(
        Arc::new(BinaryExpr::new(left, Operator::And, right)),
        vec![
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
        ],
        Arc::new(Schema::new(vec![
            Field::new("bid", DataType::Int32, false),
            Field::new("pid", DataType::Int32, false),
        ])),
    )
}

fn signatures(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = vec![];
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut sig = String::new();
            for col in batch.columns() {
                if col.is_null(row) {
                    sig.push_str("N|");
                    continue;
                }
                match col.data_type() {
                    DataType::Int32 => {
                        let value = col
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap()
                            .value(row);
                        write!(sig, "{value}|").unwrap();
                    }
                    DataType::LargeUtf8 => {
                        let value = col
                            .as_any()
                            .downcast_ref::<LargeStringArray>()
                            .unwrap()
                            .value(row);
                        write!(sig, "{}:{}|", value.len(), &value[..1.min(value.len())])
                            .unwrap();
                    }
                    DataType::Boolean => {
                        let value = col
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .unwrap()
                            .value(row);
                        write!(sig, "{value}|").unwrap();
                    }
                    other => panic!("{other}"),
                }
            }
            out.push(sig);
        }
    }
    out
}

fn plan(
    build: &[RecordBatch],
    probe: &RecordBatch,
    join_type: JoinType,
    filtered: bool,
    eq: NullEquality,
    fetch: Option<usize>,
) -> Result<Arc<HashJoinExec>> {
    let bs = build[0].schema();
    let ps = probe.schema();
    let mut b = HashJoinExecBuilder::new(
        memory_input(build, 1),
        memory_input(std::slice::from_ref(probe), 1),
        vec![(
            Arc::new(Column::new_with_schema("k", &bs).unwrap()) as _,
            Arc::new(Column::new_with_schema("k", &ps).unwrap()) as _,
        )],
        join_type,
    )
    .with_partition_mode(PartitionMode::CollectLeft)
    .with_null_equality(eq)
    .with_fetch(fetch);
    if filtered {
        b = b.with_filter(Some(filter()));
    }
    Ok(Arc::new(b.build()?))
}

#[tokio::test]
async fn rs3_review_every_join_type_matches_stock_and_bounds_bytes() -> Result<()> {
    let (build, probe) = inputs()?;
    let mut report = vec![];
    for join_type in [
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
        JoinType::LeftMark,
        JoinType::RightSemi,
        JoinType::RightAnti,
        JoinType::RightMark,
    ] {
        for eq in [
            NullEquality::NullEqualsNothing,
            NullEquality::NullEqualsNull,
        ] {
            for filtered in [false, true] {
                let reference = plan(&build, &probe, join_type, filtered, eq, None)?;
                let ctx =
                    task_context(SessionConfig::new().with_batch_size(8192), None, true);
                let mut expected = signatures(
                    &reference.execute(0, ctx)?.try_collect::<Vec<_>>().await?,
                );
                expected.sort_unstable();
                for chunked in [false, true] {
                    let p = plan(&build, &probe, join_type, filtered, eq, None)?;
                    let config =
                        no_reversal(spill_config(1, 512 * 1024, 1)).with_batch_size(8192);
                    let ctx = task_context(config, chunked.then_some(1024 * 1024), true);
                    let mut stream = p.execute(0, Arc::clone(&ctx))?;
                    let mut out = vec![];
                    let mut max = 0;
                    while let Some(batch) = stream.try_next().await? {
                        let bytes = batch.get_array_memory_size();
                        max = max.max(bytes);
                        assert!(
                            bytes <= LIMIT || batch.num_rows() == 1,
                            "{join_type} {eq:?} filtered={filtered} chunked={chunked}: {bytes} bytes in {} rows",
                            batch.num_rows()
                        );
                        assert!(batch.num_rows() <= 8192);
                        out.push(batch);
                    }
                    drop(stream);
                    let spilled =
                        metric(&p.metrics().unwrap(), "spilled_partitions").unwrap_or(0);
                    let chunks =
                        metric(&p.metrics().unwrap(), "spill_chunked_runs").unwrap_or(0);
                    let mut actual = signatures(&out);
                    actual.sort_unstable();
                    report.push(format!(
                        "{join_type} {eq:?} filtered={filtered} chunked={chunked}: rows={} batches={} max={max} spilled={spilled} chunk_runs={chunks}",
                        actual.len(),
                        out.len()
                    ));
                    assert!(spilled > 0, "{}", report.last().unwrap());
                    assert_eq!(chunks > 0, chunked, "{}", report.last().unwrap());
                    if join_type == JoinType::LeftMark {
                        assert!(out.len() <= 8, "{}", report.last().unwrap());
                    }
                    assert_eq!(
                        actual.len(),
                        expected.len(),
                        "{}",
                        report.last().unwrap()
                    );
                    assert!(actual == expected, "{}", report.last().unwrap());
                    drop(p);
                    assert_eq!(ctx.memory_pool().reserved(), 0);
                }
            }
        }
    }
    for line in &report {
        eprintln!("{line}");
    }
    Ok(())
}

#[tokio::test]
async fn rs3_review_fetch_prefix_on_the_byte_path() -> Result<()> {
    let (build, probe) = inputs()?;
    let config = no_reversal(spill_config(1, 512 * 1024, 1)).with_batch_size(8192);
    for join_type in [
        JoinType::Inner,
        JoinType::Left,
        JoinType::Full,
        JoinType::Right,
    ] {
        let reference = plan(
            &build,
            &probe,
            join_type,
            false,
            NullEquality::NullEqualsNothing,
            None,
        )?;
        let ctx = task_context(config.clone(), None, true);
        let all = signatures(&reference.execute(0, ctx)?.try_collect::<Vec<_>>().await?);
        assert!(
            metric(&reference.metrics().unwrap(), "spilled_partitions").unwrap_or(0) > 0
        );
        for fetch in [0, 1, 700, 3000, 100_000] {
            let p = plan(
                &build,
                &probe,
                join_type,
                false,
                NullEquality::NullEqualsNothing,
                Some(fetch),
            )?;
            let ctx = task_context(config.clone(), None, true);
            let out = p
                .execute(0, Arc::clone(&ctx))?
                .try_collect::<Vec<_>>()
                .await?;
            for batch in &out {
                assert!(batch.get_array_memory_size() <= LIMIT || batch.num_rows() == 1);
            }
            assert_eq!(
                signatures(&out),
                all[..fetch.min(all.len())],
                "{join_type} fetch={fetch}"
            );
            drop(p);
            assert_eq!(ctx.memory_pool().reserved(), 0);
        }
    }
    Ok(())
}

/// Batch-shape comparison: narrow output, spilled vs not spilled.
#[tokio::test]
async fn rs3_review_narrow_output_batch_count() -> Result<()> {
    // 64k build rows, 1:1 keys, narrow Int32 output.
    let n = 65536;
    let build = RecordBatch::try_from_iter([
        ("k", Arc::new(Int32Array::from_iter_values(0..n)) as _),
        ("bid", Arc::new(Int32Array::from_iter_values(0..n)) as _),
    ])?;
    let probe = RecordBatch::try_from_iter([
        (
            "k",
            Arc::new(Int32Array::from_iter_values((0..n).map(|i| i * 4))) as _,
        ),
        ("pid", Arc::new(Int32Array::from_iter_values(0..n)) as _),
    ])?;
    let build: Vec<_> = (0..n as usize)
        .step_by(4096)
        .map(|s| build.slice(s, 4096))
        .collect();
    let probe: Vec<_> = (0..n as usize)
        .step_by(4096)
        .map(|s| probe.slice(s, 4096))
        .collect();
    let mut batch_counts = Vec::new();
    for spill in [false, true] {
        let p = HashJoinExec::try_new(
            memory_input(&build, 1),
            memory_input(&probe, 1),
            vec![(
                Arc::new(Column::new("k", 0)) as _,
                Arc::new(Column::new("k", 0)) as _,
            )],
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?;
        let config = if spill {
            no_reversal(spill_config(1, 32 * 1024, 2)).with_batch_size(8192)
        } else {
            SessionConfig::new().with_batch_size(8192)
        };
        let ctx = task_context(config, None, true);
        let out = p.execute(0, ctx)?.try_collect::<Vec<_>>().await?;
        let rows: usize = out.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(rows, 16_384);
        batch_counts.push(out.len());
        let mut sizes: Vec<_> = out.iter().map(RecordBatch::num_rows).collect();
        sizes.sort_unstable();
        eprintln!(
            "spill={spill}: rows={rows} batches={} spilled={} min={:?} median={:?} max={:?}",
            out.len(),
            metric(&p.metrics().unwrap(), "spilled_partitions").unwrap_or(0),
            sizes.first(),
            sizes.get(sizes.len() / 2),
            sizes.last()
        );
    }
    assert_eq!(batch_counts, [2, 2]);
    Ok(())
}
