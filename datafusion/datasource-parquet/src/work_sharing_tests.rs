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

//! Differential, parallel-claim, cancellation and injected-error regressions for WS.
#![allow(unused_qualifications)]

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::source::ParquetSource;
use crate::{DefaultParquetFileReaderFactory, ParquetFileReaderFactory};
use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use bytes::{BufMut, Bytes, BytesMut};
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
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::StreamExt;
use futures::future::BoxFuture;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::WriterProperties;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<T: Copy>(&mut self, values: &[T]) -> T {
        values[self.below(values.len())]
    }
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]))
}

struct FileInfo {
    name: String,
    size: u64,
    /// Start offset of every row group, for byte-range splits.
    starts: Vec<i64>,
}

/// Write one file whose row groups have the given row counts; keys are unique
/// and ascending from `first`.
fn write_file(first: i32, groups: &[usize]) -> (Bytes, Vec<i64>) {
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1 << 20))
        .set_data_page_row_count_limit(50)
        .set_write_batch_size(50)
        .build();
    let mut out = BytesMut::new().writer();
    {
        let mut writer =
            ArrowWriter::try_new(&mut out, schema(), Some(properties)).unwrap();
        let mut key = first;
        for &rows in groups {
            let keys: Vec<i32> = (key..key + rows as i32).collect();
            let batch = RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int32Array::from(keys.clone())),
                    Arc::new(Int32Array::from_iter_values(keys.iter().map(|k| k % 7))),
                ],
            )
            .unwrap();
            writer.write(&batch).unwrap();
            writer.flush().unwrap();
            key += rows as i32;
        }
        writer.close().unwrap();
    }
    let data = out.into_inner().freeze();
    let metadata = ParquetMetaDataReader::new()
        .parse_and_finish(&data)
        .unwrap();
    assert_eq!(metadata.num_row_groups(), groups.len());
    let starts = metadata
        .row_groups()
        .iter()
        .map(|group| {
            let column = group.column(0);
            column
                .dictionary_page_offset()
                .unwrap_or_else(|| column.data_page_offset())
        })
        .collect();
    (data, starts)
}

fn col(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new(name, index))
}
fn lit(value: i32) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Int32(Some(value))))
}
fn bin(
    l: Arc<dyn PhysicalExpr>,
    op: Operator,
    r: Arc<dyn PhysicalExpr>,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(l, op, r))
}
fn key_between(low: i32, high: i32) -> Arc<dyn PhysicalExpr> {
    bin(
        bin(col("k", 0), Operator::GtEq, lit(low)),
        Operator::And,
        bin(col("k", 0), Operator::Lt, lit(high)),
    )
}
fn complete_dynamic() -> Arc<dyn PhysicalExpr> {
    let filter = DynamicFilterPhysicalExpr::new(
        vec![col("k", 0)],
        bin(col("k", 0), Operator::GtEq, lit(0)),
    );
    filter.mark_complete();
    Arc::new(filter)
}

#[derive(Clone, Copy, Debug)]
enum Pred {
    None,
    /// Statistics pruning only; no pushdown.
    Range,
    /// Statistics + page index + row filter.
    RangePushdown,
    /// Row filter only; empties whole row groups silently in the decoder.
    ModuloPushdown,
    /// Completed dynamic filter and range with pushdown.
    DynamicPushdown,
    /// A watching dynamic filter (runtime pruner present) that completes while
    /// the scan runs, so donation starts late next to the pruner.
    LateDynamic,
}

static LATE: std::sync::Mutex<Option<Arc<DynamicFilterPhysicalExpr>>> =
    std::sync::Mutex::new(None);

#[derive(Clone, Copy, Debug)]
enum Layout {
    RoundRobin,
    AllInFirst,
    ByteRanges,
}

/// Fail the Nth data read of one file.
#[derive(Debug)]
struct FaultFactory {
    inner: DefaultParquetFileReaderFactory,
    file: Option<String>,
    remaining: Arc<AtomicUsize>,
    delay: bool,
}
impl ParquetFileReaderFactory for FaultFactory {
    fn create_reader(
        &self,
        partition: usize,
        file: PartitionedFile,
        hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<Box<dyn AsyncFileReader + Send>> {
        let faulty = self
            .file
            .as_ref()
            .is_some_and(|name| file.object_meta.location.as_ref() == name);
        Ok(Box::new(FaultReader {
            inner: self.inner.create_reader(partition, file, hint, metrics)?,
            remaining: faulty.then(|| Arc::clone(&self.remaining)),
            delay: self.delay,
        }))
    }
}
struct FaultReader {
    inner: Box<dyn AsyncFileReader + Send>,
    remaining: Option<Arc<AtomicUsize>>,
    delay: bool,
}
impl FaultReader {
    fn fail(&self) -> bool {
        self.remaining.as_ref().is_some_and(|remaining| {
            remaining
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_err()
        })
    }
}
impl AsyncFileReader for FaultReader {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        if self.fail() {
            return Box::pin(async {
                Err(parquet::errors::ParquetError::General("injected".into()))
            });
        }
        let delay = self.delay;
        let inner = self.inner.get_bytes(range);
        Box::pin(async move {
            if delay {
                tokio::task::yield_now().await;
            }
            inner.await
        })
    }
    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        if self.fail() {
            return Box::pin(async {
                Err(parquet::errors::ParquetError::General("injected".into()))
            });
        }
        let delay = self.delay;
        let inner = self.inner.get_byte_ranges(ranges);
        Box::pin(async move {
            if delay {
                tokio::task::yield_now().await;
            }
            inner.await
        })
    }
    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        self.inner.get_metadata(options)
    }
}

struct Case {
    store: Arc<dyn ObjectStore>,
    files: Vec<FileInfo>,
    total: i32,
    streams: usize,
    layout: Layout,
    pred: Pred,
    low: i32,
    high: i32,
}

impl Case {
    async fn random(rng: &mut Rng) -> Self {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let file_count = rng.pick(&[1usize, 1, 2, 3, 5]);
        let mut files = Vec::new();
        let mut key = 0i32;
        for index in 0..file_count {
            let group_count = rng.pick(&[1usize, 2, 3, 7, 8, 24, 40]);
            let groups: Vec<usize> = (0..group_count)
                .map(|_| rng.pick(&[1usize, 2, 50, 100, 137, 300]))
                .collect();
            let (data, starts) = write_file(key, &groups);
            key += groups.iter().sum::<usize>() as i32;
            let name = format!("f{index}.parquet");
            let size = data.len() as u64;
            store
                .put(&Path::from(name.as_str()), data.into())
                .await
                .unwrap();
            files.push(FileInfo { name, size, starts });
        }
        let total = key;
        let low = rng.below(total as usize) as i32;
        let high = low + 1 + rng.below((total - low) as usize) as i32;
        Self {
            store,
            files,
            total,
            streams: rng.pick(&[2usize, 3, 4, 8, 16]),
            layout: rng.pick(&[
                Layout::RoundRobin,
                Layout::AllInFirst,
                Layout::ByteRanges,
            ]),
            pred: rng.pick(&[
                Pred::None,
                Pred::Range,
                Pred::RangePushdown,
                Pred::ModuloPushdown,
                Pred::DynamicPushdown,
                Pred::LateDynamic,
                Pred::LateDynamic,
            ]),
            low,
            high,
        }
    }

    fn groups(&self) -> Vec<FileGroup> {
        let mut groups: Vec<Vec<PartitionedFile>> = vec![vec![]; self.streams];
        for (index, file) in self.files.iter().enumerate() {
            let whole = || PartitionedFile::new(file.name.clone(), file.size);
            match self.layout {
                Layout::RoundRobin => groups[index % self.streams].push(whole()),
                Layout::AllInFirst => groups[0].push(whole()),
                Layout::ByteRanges => {
                    // Cut at row-group starts so that every group belongs to
                    // exactly one range, as `prune_by_range` assigns them.
                    let parts = self.streams.min(file.starts.len());
                    let per = file.starts.len().div_ceil(parts);
                    for part in 0..parts {
                        let Some(&start) = file.starts.get(part * per) else {
                            break;
                        };
                        let end = file
                            .starts
                            .get((part + 1) * per)
                            .copied()
                            .unwrap_or(file.size as i64);
                        let start = if part == 0 { 0 } else { start };
                        groups[(index + part) % self.streams]
                            .push(whole().with_range(start, end));
                    }
                }
            }
        }
        groups.into_iter().map(FileGroup::new).collect()
    }

    fn source(&self, factory: Arc<dyn ParquetFileReaderFactory>) -> ParquetSource {
        let source =
            ParquetSource::new(schema()).with_parquet_file_reader_factory(factory);
        match self.pred {
            Pred::None => source,
            Pred::Range => source.with_predicate(key_between(self.low, self.high)),
            Pred::RangePushdown => source
                .with_predicate(bin(
                    key_between(self.low, self.high),
                    Operator::And,
                    bin(col("v", 1), Operator::NotEq, lit(3)),
                ))
                .with_pushdown_filters(true)
                .with_enable_page_index(true),
            Pred::ModuloPushdown => source
                .with_predicate(bin(
                    bin(col("k", 0), Operator::Modulo, lit(400)),
                    Operator::GtEq,
                    lit(250),
                ))
                .with_pushdown_filters(true),
            Pred::DynamicPushdown => source
                .with_predicate(bin(
                    key_between(self.low, self.high),
                    Operator::And,
                    complete_dynamic(),
                ))
                .with_pushdown_filters(true)
                .with_enable_page_index(true),
            Pred::LateDynamic => {
                let filter = Arc::new(DynamicFilterPhysicalExpr::new(
                    vec![col("k", 0)],
                    bin(col("k", 0), Operator::GtEq, lit(0)),
                ));
                *LATE.lock().unwrap() = Some(Arc::clone(&filter));
                source
                    .with_predicate(bin(
                        key_between(self.low, self.high),
                        Operator::And,
                        filter as Arc<dyn PhysicalExpr>,
                    ))
                    .with_pushdown_filters(true)
                    .with_enable_page_index(false)
            }
        }
    }

    /// Exact expected keys for pushdown predicates; `None` when the predicate
    /// prunes at row-group granularity only.
    fn expected(&self) -> Option<Vec<i32>> {
        match self.pred {
            Pred::None => Some((0..self.total).collect()),
            Pred::Range => None,
            Pred::RangePushdown => {
                Some((self.low..self.high).filter(|k| k % 7 != 3).collect())
            }
            Pred::ModuloPushdown => {
                Some((0..self.total).filter(|k| k % 400 >= 250).collect())
            }
            Pred::DynamicPushdown | Pred::LateDynamic => {
                Some((self.low..self.high).collect())
            }
        }
    }

    fn exec(
        &self,
        work_stealing: bool,
        factory: Arc<dyn ParquetFileReaderFactory>,
    ) -> (Arc<dyn ExecutionPlan>, Arc<TaskContext>) {
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("memory://").unwrap(),
            Arc::new(self.source(factory)),
        )
        .with_file_groups(self.groups())
        .build();
        let exec: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(config);
        if let Some(filter) = LATE.lock().unwrap().take() {
            let micros = (self.low as u64 * 7919 + self.high as u64) % 400;
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_micros(micros));
                filter.mark_complete();
            });
        }
        let mut options = ConfigOptions::default();
        options.execution.enable_file_stream_work_stealing = work_stealing;
        let context = TaskContext::default()
            .with_session_config(SessionConfig::from(options).with_batch_size(64));
        context.runtime_env().register_object_store(
            ObjectStoreUrl::parse("memory://").unwrap().as_ref(),
            Arc::clone(&self.store),
        );
        (exec, Arc::new(context))
    }

    fn plain_factory(&self, delay: bool) -> Arc<dyn ParquetFileReaderFactory> {
        Arc::new(FaultFactory {
            inner: DefaultParquetFileReaderFactory::new(Arc::clone(&self.store)),
            file: None,
            remaining: Arc::new(AtomicUsize::new(0)),
            delay,
        })
    }
}

fn keys_of(batch: &RecordBatch) -> Vec<i32> {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .values()
        .to_vec()
}

/// Run every partition on its own OS thread and current-thread runtime, so
/// the shared queue sees real parallel claims. `None` means a hang.
fn collect_all(
    exec: &Arc<dyn ExecutionPlan>,
    context: &Arc<TaskContext>,
    streams: usize,
    seed: u64,
    stop: Option<(usize, usize)>,
) -> Option<Vec<Result<Vec<i32>, String>>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    let barrier = Arc::new(std::sync::Barrier::new(streams));
    for partition in 0..streams {
        let exec = Arc::clone(exec);
        let context = Arc::clone(context);
        let sender = sender.clone();
        let barrier = Arc::clone(&barrier);
        std::thread::spawn(move || {
            let runtime = runtime();
            let result: Result<Vec<i32>, String> = runtime.block_on(async move {
                let mut rng = Rng(seed
                    ^ (partition as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)
                    | 1);
                let mut stream = exec
                    .execute(partition, context)
                    .map_err(|e| e.to_string())?;
                barrier.wait();
                let mut keys = Vec::new();
                let mut polls = 0;
                loop {
                    if stop == Some((partition, polls)) {
                        break;
                    }
                    polls += 1;
                    let Some(batch) = stream.next().await else {
                        break;
                    };
                    keys.extend(keys_of(&batch.map_err(|e| e.to_string())?));
                    for _ in 0..rng.below(3) {
                        std::thread::yield_now();
                    }
                }
                Ok(keys)
            });
            let _ = sender.send((partition, result));
        });
    }
    drop(sender);
    let deadline = std::time::Instant::now() + TIMEOUT;
    let mut out: Vec<Option<Result<Vec<i32>, String>>> =
        (0..streams).map(|_| None).collect();
    for _ in 0..streams {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        let (partition, result) = receiver.recv_timeout(left).ok()?;
        out[partition] = Some(result);
    }
    Some(out.into_iter().map(Option::unwrap).collect())
}

fn published(exec: &Arc<dyn ExecutionPlan>) -> usize {
    exec.metrics()
        .unwrap()
        .sum_by_name("row_group_units_published")
        .map_or(0, |value| value.as_usize())
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

const TIMEOUT: Duration = Duration::from_secs(20);

/// Q1/Q2/Q6: the multiset of rows with donation equals the multiset without,
/// over random file / row-group / range / predicate shapes and schedules.
#[test]
fn differential_scan_with_and_without_donation() {
    let runtime = runtime();
    let cases: u64 = std::env::var("SW1_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    let mut donated_cases = 0usize;
    let mut total_units = 0usize;
    for seed in 1..=cases {
        let mut rng = Rng(seed.wrapping_mul(0xD1B54A32D192ED03) | 1);
        {
            let case = runtime.block_on(Case::random(&mut rng));
            let describe = format!(
                "seed {seed}: files {:?} streams {} layout {:?} pred {:?} [{}, {})",
                case.files
                    .iter()
                    .map(|f| f.starts.len())
                    .collect::<Vec<_>>(),
                case.streams,
                case.layout,
                case.pred,
                case.low,
                case.high
            );
            let (exec, context) = case.exec(false, case.plain_factory(false));
            let base = collect_all(&exec, &context, case.streams, seed, None)
                .unwrap_or_else(|| panic!("baseline hang: {describe}"));
            let mut base: Vec<i32> =
                base.into_iter().flat_map(|keys| keys.unwrap()).collect();
            base.sort_unstable();
            if let Some(expected) = case.expected() {
                assert_eq!(base, expected, "baseline wrong: {describe}");
            }
            for round in 0..3u64 {
                let (exec, context) = case.exec(true, case.plain_factory(round != 0));
                let got =
                    collect_all(&exec, &context, case.streams, seed * 31 + round, None)
                        .unwrap_or_else(|| panic!("donation hang: {describe}"));
                let mut got: Vec<i32> =
                    got.into_iter().flat_map(|keys| keys.unwrap()).collect();
                got.sort_unstable();
                let units = published(&exec);
                assert_eq!(
                    got, base,
                    "donation changed rows ({units} units): {describe}"
                );
                if units > 0 {
                    donated_cases += 1;
                    total_units += units;
                }
            }
        }
    }
    eprintln!(
        "cases with donation: {donated_cases} of {}; units {total_units}",
        cases * 3
    );
    assert!(donated_cases > 0, "no case exercised donation");
}

/// Q5: an injected read error surfaces, every stream terminates, and no row
/// is produced twice.
#[test]
fn injected_read_error_terminates_all_streams_without_duplicates() {
    let runtime = runtime();
    let mut errors = 0usize;
    let mut with_units = 0usize;
    for seed in 1..=300u64 {
        let mut rng = Rng(seed.wrapping_mul(0xA24BAED4963EE407) | 1);
        {
            let case = runtime.block_on(Case::random(&mut rng));
            let file = case.files[rng.below(case.files.len())].name.clone();
            let factory = Arc::new(FaultFactory {
                inner: DefaultParquetFileReaderFactory::new(Arc::clone(&case.store)),
                file: Some(file),
                remaining: Arc::new(AtomicUsize::new(rng.below(12))),
                delay: true,
            });
            let (exec, context) = case.exec(true, factory);
            let got = collect_all(&exec, &context, case.streams, seed, None)
                .unwrap_or_else(|| panic!("hang after injected error, seed {seed}"));
            let failed = got.iter().filter(|result| result.is_err()).count();
            let keys: Vec<i32> = got.into_iter().flatten().flatten().collect();
            let unique: HashSet<i32> = keys.iter().copied().collect();
            assert_eq!(
                unique.len(),
                keys.len(),
                "duplicate rows after error, seed {seed}"
            );
            if failed > 0 {
                errors += 1;
            } else if let Some(expected) = case.expected() {
                // The fault budget was never reached: the result must be whole.
                assert_eq!(unique.len(), expected.len(), "seed {seed}");
            }
            if published(&exec) > 0 {
                with_units += 1;
            }
        }
    }
    eprintln!("error cases {errors}, with donation {with_units}");
    assert!(errors > 0);
}

/// Q5: dropping one stream at a random point lets the others finish, with no
/// duplicates.
#[test]
fn dropping_one_stream_mid_scan_lets_siblings_finish() {
    let runtime = runtime();
    for seed in 1..=300u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9FB21C651E98DF25) | 1);
        let case = runtime.block_on(Case::random(&mut rng));
        let (exec, context) = case.exec(true, case.plain_factory(true));
        let victim = rng.below(case.streams);
        let after = rng.below(6);
        let got = collect_all(&exec, &context, case.streams, seed, Some((victim, after)))
            .unwrap_or_else(|| {
                panic!("hang after dropping stream {victim}, seed {seed}")
            });
        let all: Vec<i32> = got.into_iter().flat_map(|keys| keys.unwrap()).collect();
        let unique: HashSet<i32> = all.iter().copied().collect();
        assert_eq!(
            unique.len(),
            all.len(),
            "duplicate rows after drop, seed {seed}"
        );
    }
}

/// Drain the partitions one after another, as `for p in 0..n { collect(p) }`.
async fn sequential(work_stealing: bool) -> Result<usize, &'static str> {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let mut files = Vec::new();
    let mut key = 0;
    for index in 0..2 {
        let (data, starts) = write_file(key, &[100; 8]);
        key += 800;
        let name = format!("f{index}.parquet");
        let size = data.len() as u64;
        store
            .put(&Path::from(name.as_str()), data.into())
            .await
            .unwrap();
        files.push(FileInfo { name, size, starts });
    }
    let case = Case {
        store,
        files,
        total: key,
        streams: 2,
        layout: Layout::RoundRobin,
        pred: Pred::None,
        low: 0,
        high: key,
    };
    let (exec, context) = case.exec(work_stealing, case.plain_factory(false));
    let run = async {
        let mut rows = 0;
        for partition in 0..2 {
            let mut stream = exec.execute(partition, Arc::clone(&context)).unwrap();
            while let Some(batch) = stream.next().await {
                rows += batch.unwrap().num_rows();
            }
        }
        rows
    };
    tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .map_err(|_| "hang")
}

/// A consumer may construct and drain each partition before creating the next.
#[test]
fn sequential_partition_consumption_finishes() {
    let runtime = runtime();
    let off = runtime.block_on(sequential(false));
    let on = runtime.block_on(sequential(true));
    eprintln!("PROBE sequential: stealing off = {off:?}, stealing on = {on:?}");
    assert_eq!(off, Ok(1600));
    assert_eq!(on, Ok(1600));
}
