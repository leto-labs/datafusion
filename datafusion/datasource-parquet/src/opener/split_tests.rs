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

//! Whole-row-group sharing preserves planned file ownership, selections and
//! read filters. Idle siblings can receive unread tails after a reader opens;
//! ordered, limited and watching-filter readers retain their own work.

use std::sync::Arc;

use arrow::array::{Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use bytes::{BufMut, BytesMut};
use datafusion_common::ScalarValue;
use datafusion_common::config::ConfigOptions;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{
    BinaryExpr, Column, DynamicFilterPhysicalExpr, Literal,
};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::metrics::{MetricValue, MetricsSet};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::reader::ParquetFileReaderFactory;
use crate::source::ParquetSource;

const ROW_GROUPS: i32 = 24;
const ROWS_PER_GROUP: i32 = 100;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]))
}

/// One file of 24 row groups with sorted keys `0..2400`.
async fn store() -> (Arc<dyn ObjectStore>, u64) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let batch = RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from_iter_values(
            0..ROW_GROUPS * ROWS_PER_GROUP,
        ))],
    )
    .unwrap();
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROWS_PER_GROUP as usize))
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
    (store, size)
}

fn key_between(low: i32, high: i32) -> Arc<dyn PhysicalExpr> {
    let key = || Arc::new(Column::new("k", 0)) as Arc<dyn PhysicalExpr>;
    let literal = |value| Arc::new(Literal::new(ScalarValue::Int32(Some(value)))) as _;
    Arc::new(BinaryExpr::new(
        Arc::new(BinaryExpr::new(key(), Operator::GtEq, literal(low))),
        Operator::And,
        Arc::new(BinaryExpr::new(key(), Operator::Lt, literal(high))),
    ))
}

/// Scan the file with `streams` sibling streams, the file in the first.
/// Return the sorted keys and the number of published row-group units.
async fn scan(
    streams: usize,
    work_stealing: bool,
    predicate: Option<Arc<dyn PhysicalExpr>>,
) -> (Vec<i32>, usize) {
    let (store, size) = store().await;
    let mut source = ParquetSource::new(schema());
    if let Some(predicate) = predicate {
        source = source.with_predicate(predicate);
    }
    let groups = vec![vec![PartitionedFile::new("k.parquet", size)]];
    let (keys, metrics) = run(store, source, groups, streams, work_stealing).await;
    (keys, count(&metrics, "row_group_units_published"))
}

/// Execute a fresh scan of `source` over `groups` padded with empty groups
/// to `streams` sibling streams. Return the sorted first-column keys and the
/// scan's metrics.
async fn run(
    store: Arc<dyn ObjectStore>,
    source: ParquetSource,
    groups: Vec<Vec<PartitionedFile>>,
    streams: usize,
    work_stealing: bool,
) -> (Vec<i32>, MetricsSet) {
    // Model an asynchronous object-store open so every sibling can register
    // before this tiny fixture finishes its first group. An idle stream may
    // take an unopened file when no owner is active (sequential consumers).
    let store: Arc<dyn ObjectStore> =
        Arc::new(object_store::throttle::ThrottledStore::new(
            store,
            object_store::throttle::ThrottleConfig {
                wait_get_per_call: std::time::Duration::from_micros(1),
                ..Default::default()
            },
        ));
    let mut groups: Vec<FileGroup> = groups.into_iter().map(FileGroup::new).collect();
    groups.resize_with(streams, || FileGroup::new(vec![]));
    let config = FileScanConfigBuilder::new(
        ObjectStoreUrl::parse("memory://").unwrap(),
        Arc::new(source),
    )
    .with_file_groups(groups)
    .build();
    let exec: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(config);

    let mut options = ConfigOptions::default();
    options.execution.enable_file_stream_work_stealing = work_stealing;
    let context =
        TaskContext::default().with_session_config(SessionConfig::from(options));
    context.runtime_env().register_object_store(
        ObjectStoreUrl::parse("memory://").unwrap().as_ref(),
        store,
    );
    use futures::{FutureExt, StreamExt, TryStreamExt};
    let context = Arc::new(context);
    let mut readers: Vec<_> = (0..streams)
        .map(|partition| exec.execute(partition, Arc::clone(&context)).unwrap())
        .collect();
    let mut batches: Vec<Vec<RecordBatch>> = vec![vec![]; streams];
    // Make idle siblings visible before the owner runs. In-memory readers can
    // finish an entire tiny fixture without yielding to Tokio.
    for partition in (0..streams).rev() {
        if let Some(Some(batch)) = readers[partition].next().now_or_never() {
            batches[partition].push(batch.unwrap());
        }
    }
    let rest: Vec<Vec<RecordBatch>> = futures::future::try_join_all(
        readers.into_iter().map(|reader| reader.try_collect()),
    )
    .await
    .unwrap();
    for (batches, rest) in batches.iter_mut().zip(rest) {
        batches.extend(rest);
    }

    let mut keys: Vec<i32> = batches
        .iter()
        .flatten()
        .flat_map(|batch| {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            (0..column.len())
                .map(|i| column.value(i))
                .collect::<Vec<_>>()
        })
        .collect();
    keys.sort_unstable();
    (keys, exec.metrics().unwrap())
}

/// A counter's total, or the pruned count of a pruning metric.
fn count(metrics: &MetricsSet, name: &str) -> usize {
    match metrics.sum_by_name(name) {
        Some(MetricValue::PruningMetrics {
            pruning_metrics, ..
        }) => pruning_metrics.pruned(),
        Some(value) => value.as_usize(),
        None => 0,
    }
}

/// A dynamic filter `k >= 0` that keeps every row; complete when `complete`.
fn dynamic_filter(complete: bool) -> Arc<dyn PhysicalExpr> {
    let key = Arc::new(Column::new("k", 0)) as Arc<dyn PhysicalExpr>;
    let filter = DynamicFilterPhysicalExpr::new(
        vec![Arc::clone(&key)],
        Arc::new(BinaryExpr::new(
            key,
            Operator::GtEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(0)))),
        )),
    );
    if complete {
        filter.mark_complete();
    }
    Arc::new(filter)
}

fn and(
    left: Arc<dyn PhysicalExpr>,
    right: Arc<dyn PhysicalExpr>,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(left, Operator::And, right))
}

#[tokio::test]
async fn pruned_to_two_row_groups_publishes_one_unit() {
    // Row groups 1 and 2 survive statistics pruning. No pushdown filter, so
    // the scan returns their whole rows.
    let expected: Vec<i32> = (100..300).collect();
    let (keys, published) = scan(8, true, Some(key_between(150, 250))).await;
    assert_eq!(keys, expected);
    // One stream holds the file and 7 have no work: keep one row group,
    // publish the other.
    assert_eq!(published, 1);

    // The same scan without a shared queue returns the same rows.
    let (keys, published) = scan(8, false, Some(key_between(150, 250))).await;
    assert_eq!(keys, expected);
    assert_eq!(published, 0);
}

#[tokio::test]
async fn pruned_scan_spreads_over_all_idle_streams() {
    // Twelve surviving row groups, 7 idle streams: 8 parts, 7 published.
    let (keys, published) = scan(8, true, Some(key_between(0, 1200))).await;
    assert_eq!(keys, (0..1200).collect::<Vec<_>>());
    assert!((7..12).contains(&published));
}

#[tokio::test]
async fn full_scan_offers_whole_row_groups_to_idle_siblings() {
    let (keys, published) = scan(8, true, None).await;
    assert_eq!(keys, (0..ROW_GROUPS * ROWS_PER_GROUP).collect::<Vec<_>>());
    assert!((7..ROW_GROUPS as usize).contains(&published));
}

#[tokio::test]
async fn single_stream_does_not_publish() {
    let (keys, published) = scan(1, true, Some(key_between(150, 250))).await;
    assert_eq!(keys, (100..300).collect::<Vec<_>>());
    assert_eq!(published, 0);
}

#[tokio::test]
async fn live_dynamic_filter_never_publishes() {
    // The static key range prunes to 2 row groups on 8 streams, which
    // publishes 1 unit (see above). One completed and one live dynamic
    // filter: the live filter keeps the file in one stream.
    let predicate = |live_complete| {
        and(
            key_between(150, 250),
            and(dynamic_filter(true), dynamic_filter(live_complete)),
        )
    };
    let (keys, published) = scan(8, true, Some(predicate(false))).await;
    assert_eq!(keys, (100..300).collect::<Vec<_>>());
    assert_eq!(published, 0);

    // A fresh scan after both filters completed may publish.
    let (keys, published) = scan(8, true, Some(predicate(true))).await;
    assert_eq!(keys, (100..300).collect::<Vec<_>>());
    assert_eq!(published, 1);
}

/// Keys `0..8000` in 8 row groups of 10 pages of 100 rows, and `v = k % 5`.
async fn paged_store() -> (Arc<dyn ObjectStore>, u64, i64) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from_iter_values(0..8000)),
            Arc::new(Int32Array::from_iter_values((0..8000).map(|k| k % 5))),
        ],
    )
    .unwrap();
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1000))
        .set_data_page_row_count_limit(100)
        .set_write_batch_size(100)
        .build();
    let mut out = BytesMut::new().writer();
    {
        let mut writer =
            ArrowWriter::try_new(&mut out, schema, Some(properties)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let data = out.into_inner().freeze();
    let size = data.len() as u64;
    // Row group 3 starts the second byte range (see `prune_by_range`).
    let metadata = parquet::file::metadata::ParquetMetaDataReader::new()
        .parse_and_finish(&data)
        .unwrap();
    assert_eq!(metadata.num_row_groups(), 8);
    let column = metadata.row_group(3).column(0);
    let boundary = column
        .dictionary_page_offset()
        .unwrap_or_else(|| column.data_page_offset());
    store_put(&data)
        .await
        .map(|store| (store, size, boundary))
        .unwrap()
}

async fn store_put(data: &bytes::Bytes) -> Option<Arc<dyn ObjectStore>> {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    store
        .put(&Path::from("k.parquet"), data.clone().into())
        .await
        .ok()?;
    Some(store)
}

#[tokio::test]
async fn published_units_keep_page_selections_and_row_filters() {
    let (store, size, boundary) = paged_store().await;
    // Two adjacent byte ranges of one file: row groups 0-2 and 3-7.
    let ranges = || {
        vec![
            vec![PartitionedFile::new("k.parquet", size).with_range(0, boundary)],
            vec![
                PartitionedFile::new("k.parquet", size).with_range(boundary, size as i64),
            ],
        ]
    };
    // Row groups 1-4 survive statistics pruning; each range prunes one or
    // more of its own. The page index prunes the first two pages of row
    // group 1 and the last two of row group 4, and the row filter removes the
    // rest of the out-of-range keys and `v = 3`. Each range keeps its first
    // surviving row group and publishes its second: row group 1 (retained)
    // and row group 4 (published) carry page selections.
    let predicate = and(
        key_between(1250, 4750),
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("v", 1)),
            Operator::NotEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
        )),
    );
    let source = || {
        ParquetSource::new(Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("v", DataType::Int32, false),
        ])))
        .with_predicate(Arc::clone(&predicate))
        .with_pushdown_filters(true)
        .with_enable_page_index(true)
    };
    let expected: Vec<i32> = (1250..4750).filter(|k| k % 5 != 3).collect();

    let (keys, metrics) = run(Arc::clone(&store), source(), ranges(), 8, true).await;
    assert_eq!(keys, expected);
    assert_eq!(count(&metrics, "row_group_units_published"), 2);
    // Page selection and row filtering both ran. 200 rows of row group 1
    // and 200 of row group 4 are pruned by page; the published unit of row
    // group 4 prunes its 200 rows again when it is reopened, intersecting its
    // carried selection with the page index.
    assert_eq!(count(&metrics, "page_index_rows_pruned"), 600);
    assert!(count(&metrics, "pushdown_rows_pruned") > 0);

    // The same ranges without a shared queue return the same rows.
    let (keys, metrics) = run(store, source(), ranges(), 8, false).await;
    assert_eq!(keys, expected);
    assert_eq!(count(&metrics, "row_group_units_published"), 0);
    assert_eq!(count(&metrics, "page_index_rows_pruned"), 400);
}

/// Count native data reads, excluding footer and page-index metadata requests.
#[derive(Debug)]
struct ReadCountingFactory {
    inner: crate::reader::DefaultParquetFileReaderFactory,
    reads: Arc<std::sync::Mutex<Vec<std::ops::Range<u64>>>>,
}
impl ParquetFileReaderFactory for ReadCountingFactory {
    fn create_reader(
        &self,
        partition: usize,
        file: PartitionedFile,
        hint: Option<usize>,
        metrics: &datafusion_physical_plan::metrics::ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<
        Box<dyn parquet::arrow::async_reader::AsyncFileReader + Send>,
    > {
        Ok(Box::new(ReadCountingReader {
            inner: self.inner.create_reader(partition, file, hint, metrics)?,
            reads: Arc::clone(&self.reads),
        }))
    }
}
struct ReadCountingReader {
    inner: Box<dyn parquet::arrow::async_reader::AsyncFileReader + Send>,
    reads: Arc<std::sync::Mutex<Vec<std::ops::Range<u64>>>>,
}
impl parquet::arrow::async_reader::AsyncFileReader for ReadCountingReader {
    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> futures::future::BoxFuture<'_, parquet::errors::Result<bytes::Bytes>> {
        self.reads.lock().unwrap().push(range.clone());
        self.inner.get_bytes(range)
    }
    fn get_byte_ranges(
        &mut self,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> futures::future::BoxFuture<'_, parquet::errors::Result<Vec<bytes::Bytes>>> {
        self.reads.lock().unwrap().extend(ranges.iter().cloned());
        self.inner.get_byte_ranges(ranges)
    }
    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a parquet::arrow::arrow_reader::ArrowReaderOptions>,
    ) -> futures::future::BoxFuture<
        'a,
        parquet::errors::Result<Arc<parquet::file::metadata::ParquetMetaData>>,
    > {
        self.inner.get_metadata(options)
    }
}

/// Open the long reader before another partition goes idle. Every row group
/// has unique keys, so a duplicated or lost row group changes the exact result.
#[tokio::test]
async fn idle_after_file_open_receives_unread_whole_row_groups_exactly_once() {
    late_scan(false).await;
}

#[tokio::test]
async fn silently_empty_filtered_groups_are_never_donated_or_read_twice() {
    late_scan(true).await;
}

async fn late_scan(only_odd_groups: bool) {
    use futures::{FutureExt, StreamExt, TryStreamExt};

    let (store, size) = store().await;
    let reads = Arc::new(std::sync::Mutex::new(Vec::new()));
    let metadata = {
        let factory =
            crate::reader::DefaultParquetFileReaderFactory::new(Arc::clone(&store));
        factory
            .create_reader(
                0,
                PartitionedFile::new("k.parquet", size),
                None,
                &datafusion_physical_plan::metrics::ExecutionPlanMetricsSet::new(),
            )
            .unwrap()
            .get_metadata(None)
            .await
            .unwrap()
    };
    let mut source = ParquetSource::new(schema()).with_parquet_file_reader_factory(
        Arc::new(ReadCountingFactory {
            inner: crate::reader::DefaultParquetFileReaderFactory::new(Arc::clone(
                &store,
            )),
            reads: Arc::clone(&reads),
        }),
    );
    if only_odd_groups {
        source.table_parquet_options.global.pruning = false;
        // This is evaluated by the row filter, not statistics pruning. The
        // decoder silently skips even groups before handing back a reader.
        source =
            source
                .with_pushdown_filters(true)
                .with_predicate(Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("k", 0)),
                        Operator::Modulo,
                        Arc::new(Literal::new(ScalarValue::Int32(Some(
                            2 * ROWS_PER_GROUP,
                        )))),
                    )),
                    Operator::GtEq,
                    Arc::new(Literal::new(ScalarValue::Int32(Some(ROWS_PER_GROUP)))),
                )));
    }
    let config = FileScanConfigBuilder::new(
        ObjectStoreUrl::parse("memory://").unwrap(),
        Arc::new(source),
    )
    .with_file_groups(vec![
        FileGroup::new(vec![PartitionedFile::new("k.parquet", size)]),
        FileGroup::new(vec![]),
    ])
    .build();
    let exec: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(config);
    let mut options = ConfigOptions::default();
    options.execution.enable_file_stream_work_stealing = true;
    let context = Arc::new(
        TaskContext::default().with_session_config(SessionConfig::from(options)),
    );
    context.runtime_env().register_object_store(
        ObjectStoreUrl::parse("memory://").unwrap().as_ref(),
        store,
    );
    let mut owner = exec.execute(0, Arc::clone(&context)).unwrap();
    let first = owner.next().await.unwrap().unwrap();
    let mut idle = exec.execute(1, context).unwrap();
    assert!(
        idle.next().now_or_never().is_none(),
        "an idle sibling must wait for the already-open reader's tail"
    );
    // Advancing the owner crosses a row-group boundary and offers the tail.
    let second = owner.next().await.unwrap().unwrap();
    let stolen = idle.next().await.unwrap().unwrap();
    let (owner, idle): (Vec<_>, Vec<_>) =
        futures::try_join!(owner.try_collect(), idle.try_collect(),).unwrap();
    let mut batches = vec![first, second, stolen];
    batches.extend(owner);
    batches.extend(idle);
    let mut keys: Vec<_> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        (0..ROW_GROUPS * ROWS_PER_GROUP)
            .filter(|key| !only_odd_groups || (key / ROWS_PER_GROUP) % 2 == 1)
            .collect::<Vec<_>>()
    );
    assert!(count(&exec.metrics().unwrap(), "row_group_units_published") > 0);
    let reads = reads.lock().unwrap();
    for (index, group) in metadata.row_groups().iter().enumerate() {
        let (start, len) = group.column(0).byte_range();
        assert_eq!(
            reads
                .iter()
                .filter(|range| range.start < start + len && start < range.end)
                .count(),
            1,
            "row group {index} must be physically read exactly once: {reads:?}"
        );
    }
}

#[path = "split_tests/schedules.rs"]
mod schedules;
