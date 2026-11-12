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

// ORDER BY must agree with a Rust sort, including when no SortExec remains.
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
        Duration::from_secs(5),
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
                .map(|a| datafusion::arrow::compute::cast(a, &DataType::Int64).unwrap())
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
async fn merge_over_filtered_union_finishes() {
    let fixture = Fixture::new();
    let ctx = fixture.session(3, true, 7).await;
    for inner in [
        "SELECT 1 AS x, id, v AS w FROM raw WHERE id % 2 = 0 UNION ALL SELECT 2 AS x, id, v AS w FROM raw WHERE id % 2 = 1",
        "SELECT 1 AS x, id, v AS w FROM raw WHERE id = -1 UNION ALL SELECT 2 AS x, id, v AS w FROM raw WHERE id % 2 = 1",
    ] {
        let mut want = rows(&ctx, inner).await;
        want.sort_by(|a, b| b[0].cmp(&a[0]));
        let got = rows(&ctx, &format!("SELECT * FROM ({inner}) u ORDER BY x DESC")).await;
        assert_eq!(
            got.iter().map(|r| r[0]).collect::<Vec<_>>(),
            want.iter().map(|r| r[0]).collect::<Vec<_>>()
        );
        let mut bag = got;
        bag.sort();
        want.sort();
        assert_eq!(bag, want);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn declared_order_survives_filtered_limit() {
    let fixture = Fixture::new();
    for pushdown in [false, true] {
        let ctx = fixture.session(4, pushdown, 8192).await;
        let mut want = rows(&ctx, "SELECT id, v FROM st WHERE v = 4").await;
        want.sort();
        want.truncate(400);
        let got = rows(
            &ctx,
            "SELECT id, v FROM st WHERE v = 4 ORDER BY id LIMIT 400",
        )
        .await;
        assert_eq!(got, want, "pushdown={pushdown}");
    }
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
