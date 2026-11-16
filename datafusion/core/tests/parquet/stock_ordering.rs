// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

// Family SO join regressions: ORDER BY over joins must agree with a Rust
// sort, including when no SortExec remains.
use std::{fs::File, sync::Arc, time::Duration};

use datafusion::arrow::array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use datafusion::physical_plan::displayable;
use datafusion::prelude::*;

type Row = Vec<Option<i64>>;

struct Fixture {
    raw: tempfile::TempDir,
    sorted: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let fixture = Self {
            raw: tempfile::tempdir().unwrap(),
            sorted: tempfile::tempdir().unwrap(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        for (dir, interleaved) in [(&fixture.raw, false), (&fixture.sorted, true)] {
            let schema = if interleaved {
                Arc::new(Schema::new(vec![
                    Field::new("id", DataType::Int64, false),
                    Field::new("v", DataType::Int32, false),
                ]))
            } else {
                Arc::clone(&schema)
            };
            let mut start = 0;
            for (file, count) in [80, 37, 1, 120].into_iter().enumerate() {
                if interleaved && file == 3 {
                    continue;
                }
                let ids: Vec<i64> = if interleaved {
                    (0..500).map(|i| i * 3 + file as i64).collect()
                } else {
                    (start..start + count).collect()
                };
                start += count;
                let mut columns: Vec<ArrayRef> = vec![
                    Arc::new(Int64Array::from(ids.clone())),
                    Arc::new(Int64Array::from_iter_values(ids.iter().map(|i| i % 5))),
                    Arc::new(Int64Array::from_iter_values(
                        ids.iter().map(|i| (i * 13) % 7),
                    )),
                ];
                if interleaved {
                    // Irregular filter selectivity prevents file-size and row
                    // boundaries from accidentally hiding reordered split ranges.
                    columns.remove(1);
                    columns[1] =
                        Arc::new(Int32Array::from_iter_values(ids.iter().map(|&id| {
                            let mut x = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                ^ 22u64.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                            x ^= x >> 31;
                            x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
                            ((x ^ (x >> 29)) % 7) as i32
                        })));
                }
                let props = WriterProperties::builder()
                    .set_max_row_group_row_count(Some(if interleaved { 100 } else { 17 }))
                    .build();
                let mut writer = ArrowWriter::try_new(
                    File::create(dir.path().join(format!("p{file}.parquet"))).unwrap(),
                    Arc::clone(&schema),
                    Some(props),
                )
                .unwrap();
                writer
                    .write(&RecordBatch::try_new(Arc::clone(&schema), columns).unwrap())
                    .unwrap();
                writer.close().unwrap();
            }
        }
        fixture
    }

    async fn session(
        &self,
        partitions: usize,
        pushdown: bool,
        batch: usize,
    ) -> SessionContext {
        let mut config = SessionConfig::new()
            .with_target_partitions(partitions)
            .with_batch_size(batch);
        config.options_mut().optimizer.repartition_file_min_size = 0;
        config.options_mut().execution.parquet.pushdown_filters = pushdown;
        let ctx = SessionContext::new_with_config(config);
        ctx.register_parquet(
            "raw",
            self.raw.path().to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await
        .unwrap();
        ctx.register_parquet(
            "st",
            self.sorted.path().to_str().unwrap(),
            ParquetReadOptions::default()
                .file_sort_order(vec![vec![col("id").sort(true, false)]]),
        )
        .await
        .unwrap();
        ctx
    }
}

async fn rows(ctx: &SessionContext, sql: &str) -> Vec<Row> {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let text = displayable(plan.as_ref()).indent(false).to_string();
    let batches = tokio::time::timeout(
        Duration::from_secs(120),
        datafusion::physical_plan::collect(plan, ctx.task_ctx()),
    )
    .await
    .unwrap_or_else(|_| panic!("query did not finish: {sql}\n{text}"))
    .unwrap();
    batches
        .iter()
        .flat_map(|batch| {
            let columns: Vec<_> = batch
                .columns()
                .iter()
                .map(|a| arrow::compute::cast(a, &DataType::Int64).unwrap())
                .collect();
            (0..batch.num_rows()).map(move |i| {
                columns
                    .iter()
                    .map(|a| {
                        let a = a.as_any().downcast_ref::<Int64Array>().unwrap();
                        (!a.is_null(i)).then(|| a.value(i))
                    })
                    .collect()
            })
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn null_padded_constants_are_sorted() {
    let fixture = Fixture::new();
    let ctx = fixture.session(3, true, 7).await;
    for (join, projection, left_filter, right_filter) in [
        ("RIGHT", "a.x, b.id", "k = 1", "k IN (1, 2)"),
        ("LEFT", "b.x, a.id", "k IN (1, 2)", "k = 1"),
        ("FULL", "a.x, coalesce(a.id, b.id) AS id", "k = 1", "k = 2"),
    ] {
        let inner = format!(
            "SELECT {projection} FROM (SELECT k AS x, id FROM raw WHERE {left_filter}) a {join} JOIN (SELECT k AS x, id FROM raw WHERE {right_filter}) b ON a.id = b.id"
        );
        let mut want = rows(&ctx, &inner).await;
        want.sort_by_key(|r| (r[0].is_none(), r[0], r[1]));
        let got = rows(
            &ctx,
            &format!("SELECT * FROM ({inner}) u ORDER BY x NULLS LAST, id"),
        )
        .await;
        assert_eq!(got, want, "join={join}");
    }
}

#[tokio::test]
async fn cross_join_keeps_both_constants() {
    use datafusion::physical_expr::expressions::{Column, lit};
    use datafusion::physical_plan::{
        ExecutionPlan, empty::EmptyExec, joins::CrossJoinExec, projection::ProjectionExec,
    };
    let source: Arc<dyn ExecutionPlan> =
        Arc::new(EmptyExec::new(Arc::new(Schema::empty())));
    let left: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(vec![(lit(1i64), "a".into())], Arc::clone(&source))
            .unwrap(),
    );
    let right: Arc<dyn ExecutionPlan> =
        Arc::new(ProjectionExec::try_new(vec![(lit(2i64), "b".into())], source).unwrap());
    let join = CrossJoinExec::new(left, right);
    for (name, idx) in [("a", 0), ("b", 1)] {
        let col: Arc<dyn datafusion::physical_expr::PhysicalExpr> =
            Arc::new(Column::new(name, idx));
        assert!(
            join.properties()
                .eq_properties
                .eq_group()
                .is_expr_constant(&col)
                .is_some(),
            "{name}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeated_probe_keys_do_not_sort_build_suffixes() {
    let fixture = Fixture::new();
    let dir = tempfile::tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("g", DataType::Int64, false),
        Field::new("m", DataType::Int64, false),
    ]));
    let ids: Vec<i64> = (0..100).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(Int64Array::from_iter_values(ids.iter().map(|id| id / 10))),
            Arc::new(Int64Array::from_iter_values(ids.iter().map(|id| id % 5))),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.path().join("g.parquet")).unwrap(),
        schema,
        None,
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    for partitions in [1, 3] {
        let ctx = fixture.session(partitions, false, 8192).await;
        ctx.register_parquet(
            "sg",
            dir.path().to_str().unwrap(),
            ParquetReadOptions::default()
                .file_sort_order(vec![vec![col("g").sort(true, false)]]),
        )
        .await
        .unwrap();
        ctx.register_parquet(
            "bt",
            dir.path().to_str().unwrap(),
            ParquetReadOptions::default()
                .file_sort_order(vec![vec![col("id").sort(true, false)]]),
        )
        .await
        .unwrap();
        for from in [
            "sg p JOIN (SELECT id FROM bt WHERE id < 40) b",
            "(SELECT id FROM bt WHERE id < 40) b JOIN sg p",
        ] {
            let inner =
                format!("SELECT p.g, b.id AS bid, p.id FROM {from} ON b.id % 5 = p.m");
            let mut expected = rows(&ctx, &inner).await;
            expected.sort_by_key(|r| (r[0], r[1]));
            let actual = rows(&ctx, &format!("{inner} ORDER BY g, bid")).await;
            assert_eq!(
                actual.iter().map(|r| (r[0], r[1])).collect::<Vec<_>>(),
                expected.iter().map(|r| (r[0], r[1])).collect::<Vec<_>>(),
                "p={partitions}, {inner}"
            );
            let mut actual = actual;
            actual.sort();
            expected.sort();
            assert_eq!(actual, expected);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_merge_join_filter_handles_null_padded_columns() {
    let fixture = Fixture::new();
    let hash = fixture.session(3, true, 500).await;
    let merge = fixture.session(3, true, 500).await;
    merge
        .sql("SET datafusion.optimizer.prefer_hash_join = false")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let inner = "SELECT a.k AS x, coalesce(a.id, c.id + 4) AS id, c.k AS w FROM (SELECT k, id, v FROM raw WHERE k = 1) a FULL JOIN (SELECT k, id, v FROM raw WHERE k = 2) c ON a.id = c.id + 4 AND a.v < c.v";
    let plan = merge
        .sql(inner)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    assert!(
        displayable(plan.as_ref())
            .indent(false)
            .to_string()
            .contains("SortMergeJoin")
    );
    let mut expected = rows(&hash, inner).await;
    expected.sort();
    let mut actual = rows(&merge, inner).await;
    actual.sort();
    assert_eq!(actual, expected);
}
