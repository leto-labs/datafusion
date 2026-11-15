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

//! Runtime row-group pruning of a prepared file shares one pruning predicate
//! per dynamic-filter generation among all of the file's callers.

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use bytes::{BufMut, BytesMut};
use datafusion_common::ScalarValue;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{
    BinaryExpr, Column, DynamicFilterPhysicalExpr, Literal,
};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::push_decoder::PRUNING_PREDICATE_BUILDS;
use crate::source::ParquetSource;

const ROW_GROUPS: usize = 24;
const ROWS_PER_GROUP: usize = 100;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]))
}

fn at_least(value: i32) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(
        Arc::new(Column::new("k", 0)),
        Operator::GtEq,
        Arc::new(Literal::new(ScalarValue::Int32(Some(value)))),
    ))
}

fn builds() -> usize {
    PRUNING_PREDICATE_BUILDS.with(|builds| builds.get())
}

/// One file of 24 row groups with sorted keys `0..2400`, prepared under a
/// live dynamic filter `k >= 0`.
async fn prepared() -> (
    Arc<crate::PreparedParquetFile>,
    Arc<DynamicFilterPhysicalExpr>,
) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let batch = RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from_iter_values(
            0..(ROW_GROUPS * ROWS_PER_GROUP) as i32,
        ))],
    )
    .unwrap();
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROWS_PER_GROUP))
        .build();
    let mut out = BytesMut::new().writer();
    {
        let mut writer =
            ArrowWriter::try_new(&mut out, schema(), Some(properties)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let data = out.into_inner().freeze();
    let size = data.len() as u64;
    store
        .put(&Path::from("k.parquet"), data.into())
        .await
        .unwrap();
    let filter = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("k", 0)) as Arc<dyn PhysicalExpr>],
        at_least(0),
    ));
    let source = ParquetSource::new(schema())
        .with_predicate(Arc::clone(&filter) as Arc<dyn PhysicalExpr>);
    let file = PartitionedFile::new("k.parquet", size);
    let config = FileScanConfigBuilder::new(
        ObjectStoreUrl::parse("memory://").unwrap(),
        Arc::new(source.clone()),
    )
    .with_file_groups(vec![FileGroup::new(vec![file.clone()])])
    .build();
    let scan = source
        .prepare_file(store, &config, file, 0)
        .await
        .unwrap()
        .expect("file survives static pruning");
    (scan.file, filter)
}

fn kept(file: &crate::PreparedParquetFile) -> Vec<usize> {
    (0..ROW_GROUPS)
        .filter(|&group| file.keep_row_group(group))
        .collect()
}

#[tokio::test]
async fn runtime_pruning_builds_one_predicate_per_filter_generation() {
    let (file, filter) = prepared().await;
    let before = builds();
    assert_eq!(kept(&file), (0..ROW_GROUPS).collect::<Vec<_>>());
    // Every row group and every unit of a scan asks again. Without a new
    // filter generation, none of them rebuilds the pruning predicate.
    assert_eq!(kept(&file), (0..ROW_GROUPS).collect::<Vec<_>>());
    assert_eq!(builds() - before, 1);

    // A tightened filter is seen by the next check of each row group.
    filter.update(at_least(1_200)).unwrap();
    assert_eq!(kept(&file), (12..ROW_GROUPS).collect::<Vec<_>>());
    assert_eq!(builds() - before, 2);

    // Concurrent callers share the current generation's predicate.
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                assert_eq!(kept(&file), (12..ROW_GROUPS).collect::<Vec<_>>());
            });
        }
    });
    filter.update(at_least(2_000)).unwrap();
    filter.mark_complete();
    assert_eq!(kept(&file), (20..ROW_GROUPS).collect::<Vec<_>>());
    assert_eq!(kept(&file), (20..ROW_GROUPS).collect::<Vec<_>>());
    assert_eq!(builds() - before, 3);
}
