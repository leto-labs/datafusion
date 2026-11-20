// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow family JS: the memory bound of build chunks (review
//! MJJ98-REVIEW3). The review asked for evidence beyond reservation peaks:
//! the chunk build's allocations are measured with a counting allocator, and
//! every batch a writer produces is read back and admitted as a build chunk
//! admits it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, RefCell};
use std::fmt::{self, Display};

use arrow::array::{
    DictionaryArray, Float64Array, ListArray, StringArray, StringViewArray, StructArray,
    UInt8Array,
};
use arrow::datatypes::{Int32Type, Schema};
use datafusion_common::config::SpillCompression;
use datafusion_common::hash_utils::RandomState;
use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::CastExpr;

use super::super::ChunkBuild;
use super::super::footprint::{self, ChunkKeys};
use super::super::{Cursor, JoinSpillMetrics, PartWriter, Side, SpilledFile, WriteUnit};
use super::*;
use crate::metrics::{ExecutionPlanMetricsSet, Gauge, SpillMetrics};
use crate::spill::spill_manager::SpillManager;
use datafusion_common::utils::memory::get_record_batch_memory_size;

/// Counts the bytes each thread allocates, net of what it frees. A test
/// instrument: it only adds two thread-local cells to `System`. Frees of
/// allocations that predate a window can offset its new allocations.
struct Counting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
    static RECORDS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(vec![]) };
}

fn count(delta: isize) {
    // `try_with`: a thread that is being torn down still allocates.
    let _ = LIVE.try_with(|live| {
        let now = live.get() + delta;
        live.set(now);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
    });
}

// SAFETY: every method forwards to `System` with the same arguments and
// returns its result unchanged; counting touches only const-initialized
// thread-local cells, which neither allocate nor unwind.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded from the caller, who upholds `alloc`'s contract.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            count(layout.size() as isize);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded from the caller.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            count(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded from the caller.
        unsafe { System.dealloc(ptr, layout) };
        count(-(layout.size() as isize));
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded from the caller.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            count(new_size as isize - layout.size() as isize);
        }
        new
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// The allocations of one chunk table build (`build_chunk` in `exec.rs`):
/// the peak of the bytes the build thread holds beyond what it held when
/// the build started, against the chunk's grant less the admitted rows,
/// which were allocated (and charged) before.
pub(crate) struct BuildWindow {
    budget: usize,
    start: isize,
}

impl BuildWindow {
    pub(crate) fn open(granted: usize, resident: usize) -> Self {
        let start = LIVE.with(Cell::get);
        PEAK.with(|peak| peak.set(start));
        Self {
            budget: granted.saturating_sub(resident),
            start,
        }
    }

    pub(crate) fn close(self) {
        let peak = PEAK.with(Cell::get) - self.start;
        RECORDS.with(|records| {
            records
                .borrow_mut()
                .push((peak.max(0) as usize, self.budget))
        });
    }
}

/// The chunk builds recorded on this thread: (peak, budget).
fn take_builds() -> Vec<(usize, usize)> {
    RECORDS.with(|records| std::mem::take(&mut *records.borrow_mut()))
}

/// Small build objects outside the buffer estimates, including the private
/// pool, its consumers and batch handles.
const BUILD_SLACK: usize = 1024;

/// `rows` rows of a 200-character string key (`hot_percent` of them the
/// same "hot" value) and an Int32 payload.
fn wide_keys(
    prefix: &str,
    rows: usize,
    hot_percent: usize,
    batch_rows: usize,
) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(format!("{prefix}k"), DataType::Utf8, true),
        Field::new(format!("{prefix}v"), DataType::Int32, false),
    ]));
    (0..rows)
        .step_by(batch_rows)
        .map(|start| {
            let end = (start + batch_rows).min(rows);
            let keys: StringArray = (start..end)
                .map(|row| {
                    let key = if row * 37 % 100 < hot_percent {
                        "hot".to_string()
                    } else {
                        format!("key{}", row % 997)
                    };
                    Some(format!("{key:-<200}"))
                })
                .collect();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(keys),
                    Arc::new((start..end).map(|row| row as i32).collect::<Int32Array>()),
                ],
            )
            .unwrap()
        })
        .collect()
}

/// Review MJJ98-REVIEW3, finding 1: a key expression that builds a wide
/// key (a 200-character string concatenated with itself). Before, the build
/// evaluated it again while hashing each batch and once more over the
/// concatenated rows, outside its capped pool, so a build allocated about
/// twice the keys beyond its grant before the pool could refuse. Now the
/// keys are evaluated once per batch under the read permit and charged with
/// the batch, and the measured allocations of every chunk build stay inside
/// its grant.
#[rstest::rstest]
#[case::expression(false)]
#[case::structure(true)]
#[tokio::test]
async fn chunk_build_allocates_within_its_grant(#[case] nested: bool) -> Result<()> {
    let as_struct_key = |batch: RecordBatch| -> Result<RecordBatch> {
        let fields = vec![
            Arc::new(Field::new("i", DataType::Int32, false)),
            Arc::new(Field::new("s", DataType::Utf8, true)),
        ];
        let numbers = Arc::new(Int32Array::from_value(1, batch.num_rows()));
        let structure = StructArray::try_new(
            fields.into(),
            vec![numbers, Arc::clone(batch.column(0))],
            None,
        )?;
        Ok(RecordBatch::try_from_iter([
            (
                batch.schema_ref().field(0).name(),
                Arc::new(structure) as ArrayRef,
            ),
            (
                batch.schema_ref().field(1).name(),
                Arc::clone(batch.column(1)),
            ),
        ])?)
    };
    let test_input = |prefix, rows, hot_percent, batch_rows| {
        wide_keys(prefix, rows, hot_percent, batch_rows)
            .into_iter()
            .map(|batch| {
                if nested {
                    as_struct_key(batch)
                } else {
                    Ok(batch)
                }
            })
            .collect::<Result<Vec<_>>>()
    };
    let (build, probe, join_types) = if nested {
        (
            test_input("b", 2000, 85, 250)?,
            test_input("p", 100, 2, 100)?,
            &[JoinType::Inner][..],
        )
    } else {
        (
            test_input("b", 4000, 70, 250)?,
            test_input("p", 600, 2, 100)?,
            &[
                JoinType::Inner,
                JoinType::Right,
                JoinType::LeftAnti,
                JoinType::Full,
            ][..],
        )
    };
    let key_expr = |schema: &SchemaRef| -> PhysicalExprRef {
        if nested {
            return key(schema);
        }
        // `k || k`: 400 new bytes per row (a cast to LargeUtf8 would share
        // the string data and only widen the offsets).
        Arc::new(BinaryExpr::new(
            key(schema),
            Operator::StringConcat,
            key(schema),
        ))
    };
    for join_type in join_types {
        let plan = || -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(HashJoinExec::try_new(
                memory_input(&build, 1),
                memory_input(&probe, 1),
                vec![(key_expr(&build[0].schema()), key_expr(&probe[0].schema()))],
                None,
                join_type,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?))
        };
        let unbounded = task_context(SessionConfig::new(), None, true);
        let expected = batches_to_sort_string(&run(&plan()?, unbounded).await?);
        take_builds();
        let spilling = plan()?;
        let context = task_context(
            no_reversal(spill_config(0, 64 * 1024, 3)),
            Some(512 * 1024),
            true,
        );
        let actual = run(&spilling, Arc::clone(&context)).await?;
        assert_eq!(batches_to_sort_string(&actual), expected, "{join_type}");
        let metrics = spilling.metrics().unwrap();
        assert!(
            metric(&metrics, "spill_chunked_runs").unwrap_or(0) > 0,
            "{join_type}"
        );
        let builds = take_builds();
        assert!(!builds.is_empty(), "{join_type}: no chunk build measured");
        let worst = builds
            .iter()
            .map(|(peak, budget)| *peak as f64 / *budget as f64)
            .fold(0.0, f64::max);
        println!(
            "{join_type}: {} chunk builds, worst peak/budget {worst:.3}",
            builds.len()
        );
        for (peak, budget) in builds {
            assert!(
                peak <= budget + BUILD_SLACK,
                "{join_type}: a chunk build allocated {peak} bytes beyond its rows, \
                 its grant leaves {budget}"
            );
        }
        drop(spilling);
        assert_eq!(context.memory_pool().reserved(), 0, "{join_type}");
    }
    Ok(())
}

fn spill_manager_for(schema: &SchemaRef, compression: SpillCompression) -> SpillManager {
    let runtime = RuntimeEnvBuilder::new().build_arc().unwrap();
    SpillManager::new(
        runtime,
        SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        Arc::clone(schema),
    )
    .with_compression_type(compression)
}

fn join_metrics() -> JoinSpillMetrics {
    JoinSpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0, Gauge::new())
}

/// Writes `batches` with a unit of `unit` bytes over `on`.
fn write_file(
    manager: &SpillManager,
    batches: &[RecordBatch],
    on: &[PhysicalExprRef],
    unit: usize,
) -> Result<Option<SpilledFile>> {
    let unit = WriteUnit {
        bytes: unit,
        keys: ChunkKeys::try_new(manager.schema(), on)?,
    };
    let metrics = join_metrics();
    let mut writer = PartWriter::new(manager, "bound test", Some(unit))?;
    writer.write(batches, &metrics, Side::Build)?;
    writer.finish(&metrics, Side::Build)
}

async fn read_file(
    manager: &SpillManager,
    file: &SpilledFile,
) -> Result<Vec<RecordBatch>> {
    collect(manager.read_spill_as_stream_unbuffered(Arc::clone(&file.file), None)?).await
}

/// What a build chunk charges for a decoded batch (`ChunkBuild::admit`).
fn admitted(batch: &RecordBatch, keys: &ChunkKeys) -> Result<usize> {
    let decoded = get_record_batch_memory_size(batch);
    let keyed = keys.materialize(batch)?;
    Ok(footprint::admission_need(
        decoded + keyed.new_bytes,
        decoded + keyed.copy_bytes,
        footprint::dictionary_bound(batch)?,
        keys.hash_scratch(&keyed.batch),
        batch.num_rows(),
    ))
}

/// The review's round trip: one row of an Int64 key, a 1,367-byte string
/// and a UInt8 flag is 1,384 logical bytes and 1,392 decoded bytes.
fn review_row() -> RecordBatch {
    RecordBatch::try_from_iter(vec![
        ("k", Arc::new(Int64Array::from(vec![7])) as ArrayRef),
        (
            "s",
            Arc::new(StringArray::from(vec!["x".repeat(1367)])) as ArrayRef,
        ),
        ("f", Arc::new(UInt8Array::from(vec![1])) as ArrayRef),
    ])
    .unwrap()
}

/// Mixed columns: a nullable Int32 key, strings of 0 to 299 bytes, long
/// views, a dictionary, floats with -0.0, and lists.
fn mixed(rows: usize, offset: usize) -> RecordBatch {
    let strings: StringArray = (0..rows)
        .map(|row| (row % 7 != 0).then(|| "s".repeat((row * 31 + offset) % 300)))
        .collect();
    let views: StringViewArray = (0..rows)
        .map(|row| Some(format!("{:->40}", row + offset)))
        .collect();
    let dictionary: DictionaryArray<Int32Type> = (0..rows)
        .map(|row| {
            Some(["red", "green", "blue", "a much longer dictionary value"][row % 4])
        })
        .collect();
    let floats: Float64Array = (0..rows)
        .map(|row| Some(if row % 5 == 0 { -0.0 } else { row as f64 / 3.0 }))
        .collect();
    let lists = ListArray::from_iter_primitive::<Int32Type, _, _>(
        (0..rows).map(|row| Some((0..(row % 6) as i32).map(Some).collect::<Vec<_>>())),
    );
    RecordBatch::try_from_iter(vec![
        (
            "k",
            Arc::new(
                (0..rows)
                    .map(|row| (row % 11 != 0).then_some((row + offset) as i32))
                    .collect::<Int32Array>(),
            ) as ArrayRef,
        ),
        ("s", Arc::new(strings) as ArrayRef),
        ("w", Arc::new(views) as ArrayRef),
        ("d", Arc::new(dictionary) as ArrayRef),
        ("x", Arc::new(floats) as ArrayRef),
        ("l", Arc::new(lists) as ArrayRef),
    ])
    .unwrap()
}

/// Review MJJ98-REVIEW3, finding 2: the writer measured logical bytes and
/// the reader charges decoded bytes, so a batch written at the unit could
/// need more than the read permit (1,392 against 1,384 bytes for the
/// review's row). Every batch a writer produces, for every unit, column
/// type, key kind and compression, must fit its read permit when read back
/// and admitted as a chunk admits it; a row that cannot fit is refused
/// before it is written.
#[tokio::test]
async fn written_batches_fit_their_read_permit() -> Result<()> {
    let compressions = [
        SpillCompression::Uncompressed,
        SpillCompression::Lz4Frame,
        SpillCompression::Zstd,
    ];
    let mut checked = 0;
    let mut refused = 0;
    // The review's row around its old boundary (3,072 bytes).
    let row = review_row();
    for compression in compressions {
        let manager = spill_manager_for(&row.schema(), compression);
        let on = vec![key(&row.schema())];
        for unit in (2_900..12_000).step_by(97) {
            match write_file(&manager, std::slice::from_ref(&row), &on, unit) {
                Ok(file) => {
                    let file = file.expect("a file");
                    let keys = ChunkKeys::try_new(&row.schema(), &on)?;
                    for batch in read_file(&manager, &file).await? {
                        let need = admitted(&batch, &keys)?;
                        assert!(need <= unit, "{compression:?}: {need} > {unit}");
                        checked += 1;
                    }
                }
                Err(error) => {
                    assert!(
                        matches!(
                            error.find_root(),
                            DataFusionError::ResourcesExhausted(_)
                        ),
                        "{error}"
                    );
                    refused += 1;
                }
            }
        }
    }
    // Mixed columns, three key kinds, slices of a larger batch.
    let source = mixed(3000, 0);
    let schema = source.schema();
    let keys_of: Vec<Vec<PhysicalExprRef>> = vec![
        vec![key(&schema)],
        vec![Arc::new(CastExpr::new(
            Arc::new(Column::new("s", 1)),
            DataType::LargeUtf8,
            None,
        ))],
        vec![Arc::new(Column::new("x", 4)), key(&schema)],
    ];
    let pieces: Vec<RecordBatch> = (0..3000)
        .step_by(450)
        .map(|start| source.slice(start, 450.min(3000 - start)))
        .collect();
    for compression in compressions {
        let manager = spill_manager_for(&schema, compression);
        for on in &keys_of {
            let keys = ChunkKeys::try_new(&schema, on)?;
            for unit in [24 * 1024, 48 * 1024, 200 * 1024] {
                let file = write_file(&manager, &pieces, on, unit)?.expect("a file");
                let mut rows = 0;
                for batch in read_file(&manager, &file).await? {
                    let need = admitted(&batch, &keys)?;
                    assert!(
                        need <= unit,
                        "{compression:?} {}: {} rows need {need} > {unit}",
                        on.len(),
                        batch.num_rows()
                    );
                    rows += batch.num_rows();
                    checked += 1;
                }
                assert_eq!(rows, 3000);
            }
        }
    }
    println!("{checked} batches fit their permit; {refused} rows refused before writing");
    assert!(checked > 100 && refused > 0);
    Ok(())
}

/// The lemma under the read permit: a batch decoded from a spill file holds
/// at most `decoded_bound` bytes, for every column type, null layout, row
/// count and compression (one row included, where the header dominates).
#[tokio::test]
async fn decoded_batches_stay_within_their_bound() -> Result<()> {
    let mut shapes = vec![review_row()];
    for rows in [1, 2, 7, 64, 1000] {
        shapes.push(mixed(rows, rows));
        shapes.extend(wide_keys("w", rows, 50, rows));
        shapes.extend(table("t", rows, 13, 0, rows));
    }
    let mut worst: f64 = 0.0;
    for compression in [
        SpillCompression::Uncompressed,
        SpillCompression::Lz4Frame,
        SpillCompression::Zstd,
    ] {
        for batch in &shapes {
            let manager = spill_manager_for(&batch.schema(), compression);
            let metrics = join_metrics();
            let mut writer = PartWriter::new(&manager, "bound test", None)?;
            writer.write(std::slice::from_ref(batch), &metrics, Side::Build)?;
            let file = writer.finish(&metrics, Side::Build)?.expect("a file");
            let compressed = !matches!(compression, SpillCompression::Uncompressed);
            let bound = footprint::decoded_bound(batch, compressed)?;
            let decoded = read_file(&manager, &file).await?;
            assert_eq!(decoded.len(), 1);
            let size = get_record_batch_memory_size(&decoded[0]);
            assert!(
                size <= bound,
                "{compression:?}, {} rows of {}: decoded {size} > bound {bound}",
                batch.num_rows(),
                batch.schema()
            );
            worst = worst.max(size as f64 / bound as f64);
        }
    }
    println!("decoded size at most {worst:.3} of its bound");
    Ok(())
}

/// A cursor over `file` with a run charge on `pool`.
fn cursor_over(
    file: SpilledFile,
    manager: SpillManager,
    unit: usize,
    pool: &Arc<dyn MemoryPool>,
) -> Arc<Cursor> {
    let charge = MemoryConsumer::new("bound test run")
        .with_can_spill(true)
        .register(pool);
    Arc::new(Cursor::new(file, manager, unit, charge))
}

async fn read_cursor(cursor: &Arc<Cursor>) -> Option<Result<RecordBatch>> {
    futures::future::poll_fn(|cx| cursor.poll_next(cx)).await
}

/// The chunk grant must cover the nested hash vectors in addition to the
/// decoded rows, their copy, the table, and the top-level hash vector.
#[tokio::test]
async fn chunk_grant_covers_nested_hash_scratch() -> Result<()> {
    let input = mixed(1024, 0);
    let numeric = Arc::new(Field::new("i", DataType::Int32, true));
    let text = Arc::new(Field::new("s", DataType::Utf8, true));
    let structure = Arc::new(StructArray::try_new(
        vec![numeric, text].into(),
        vec![Arc::clone(input.column(0)), Arc::clone(input.column(1))],
        None,
    )?) as ArrayRef;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
    for key_array in [
        Arc::clone(input.column(3)),
        structure,
        Arc::clone(input.column(5)),
    ] {
        let data_type = key_array.data_type().clone();
        let input = RecordBatch::try_from_iter([("k", key_array)])?;
        let schema = input.schema();
        let manager = spill_manager_for(&schema, SpillCompression::Uncompressed);
        let on = vec![key(&schema)];
        let unit = 256 * 1024;
        let file = write_file(&manager, &[input], &on, unit)?.expect("a file");
        let cursor = cursor_over(file, manager, unit, &pool);
        let mut chunk = ChunkBuild::new(Arc::clone(&cursor));
        chunk.start(&schema, &on)?;
        let batch = read_cursor(&cursor).await.expect("a batch")?;
        let without_nested_scratch = footprint::admission_need(
            get_record_batch_memory_size(&batch),
            get_record_batch_memory_size(&batch),
            0,
            0,
            batch.num_rows(),
        );
        let (batch, _) = chunk.admit(batch)?;
        let mut hashes = vec![0; batch.num_rows()];
        let start = LIVE.with(Cell::get);
        PEAK.with(|peak| peak.set(start));
        create_hashes(batch.columns(), &RandomState::with_seed(0), &mut hashes)?;
        let peak = (PEAK.with(Cell::get) - start) as usize;
        assert!(peak > 0, "{data_type}: expected nested hash scratch");
        assert!(
            chunk.granted() >= without_nested_scratch + peak,
            "{data_type}: grant {} must cover {without_nested_scratch} + {peak} hash scratch",
            chunk.granted()
        );
    }
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

/// An integer expression with a detectable second evaluation over the
/// keyed schema. Its type normally enables the perfect hash bounds scan.
#[derive(Debug, PartialEq, Eq, Hash)]
struct OriginalBatchKey(Column);

impl Display for OriginalBatchKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "original_batch_key({})", self.0)
    }
}

impl PhysicalExpr for OriginalBatchKey {
    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }

    fn data_type(&self, schema: &Schema) -> Result<DataType> {
        self.0.data_type(schema)
    }

    fn nullable(&self, schema: &Schema) -> Result<bool> {
        self.0.nullable(schema)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        if batch.num_columns() != 2 {
            return datafusion_common::exec_err!(
                "original join key evaluated over a keyed batch"
            );
        }
        self.0.evaluate(batch)
    }

    fn children(&self) -> Vec<&PhysicalExprRef> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<PhysicalExprRef>,
    ) -> Result<PhysicalExprRef> {
        Ok(self)
    }
}

#[tokio::test]
async fn chunk_does_not_evaluate_original_keys_for_bounds() -> Result<()> {
    let build = table("b", 4096, 1, 0, 128);
    let probe = table("p", 17, 2, 0, 17);
    let plan = || -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(HashJoinExec::try_new(
            memory_input(&build, 1),
            memory_input(&probe, 1),
            vec![(
                Arc::new(OriginalBatchKey(Column::new("bk", 0))),
                key(&probe[0].schema()),
            )],
            None,
            &JoinType::LeftAnti,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    };
    let expected = run(&plan()?, task_context(SessionConfig::new(), None, true)).await?;
    let spilling = plan()?;
    let context =
        task_context(no_reversal(spill_config(0, 4096, 1)), Some(32 * 1024), true);
    let actual = run(&spilling, Arc::clone(&context)).await?;
    assert_eq!(
        batches_to_sort_string(&actual),
        batches_to_sort_string(&expected)
    );
    assert!(metric(&spilling.metrics().unwrap(), "spill_chunked_runs").unwrap_or(0) > 0);
    drop(spilling);
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}

/// Review MJJ98-REVIEW3, finding 2: the first read of each chunk had no
/// permit. A chunk now reserves one chunk unit before its first read (in a
/// full pool without refusal, as its smallest unit), and a read without a
/// permit is an internal error.
#[tokio::test]
async fn first_chunk_read_holds_a_permit() -> Result<()> {
    let batches = mixed(600, 0);
    let schema = batches.schema();
    let manager = spill_manager_for(&schema, SpillCompression::Uncompressed);
    let on = vec![key(&schema)];
    let unit = 48 * 1024;
    let file = write_file(&manager, &[batches], &on, unit)?.expect("a file");
    // A pool that is already full.
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1));
    let cursor = cursor_over(file, manager, unit, &pool);
    cursor.start_chunk();
    let error = read_cursor(&cursor).await.expect("an item").unwrap_err();
    assert!(
        error.to_string().contains("without a read permit"),
        "{error}"
    );
    let mut chunk = ChunkBuild::new(Arc::clone(&cursor));
    chunk.start(&schema, &on)?;
    assert_eq!(cursor.charged(), unit);
    let batch = read_cursor(&cursor).await.expect("a batch")?;
    let (_, resident) = chunk.admit(batch)?;
    // The batch is charged, and in a full pool the next permit is refused:
    // the chunk's input stops before the next read.
    assert!(cursor.charged() >= resident && cursor.charged() <= unit);
    assert!(read_cursor(&cursor).await.is_none());
    Ok(())
}

/// Review MJJ98-REVIEW3, finding 3: a refused permit after the file's last
/// batch set the stop flag before the reader saw its end, so the run
/// started another (empty) chunk, or failed the replay bound. The cursor
/// now ends at the file's last row.
#[tokio::test]
async fn permit_stop_at_the_last_batch_ends_the_file() -> Result<()> {
    let batch = mixed(200, 0);
    let schema = batch.schema();
    let manager = spill_manager_for(&schema, SpillCompression::Uncompressed);
    let on = vec![key(&schema)];
    // One written batch.
    let unit = 256 * 1024;
    let file = write_file(&manager, &[batch], &on, unit)?.expect("a file");
    assert_eq!(file.rows, 200);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1));
    let cursor = cursor_over(file, manager, unit, &pool);
    cursor.start_chunk();
    let mut chunk = ChunkBuild::new(Arc::clone(&cursor));
    chunk.start(&schema, &on)?;
    let batch = read_cursor(&cursor).await.expect("a batch")?;
    assert_eq!(batch.num_rows(), 200);
    chunk.admit(batch)?;
    assert!(cursor.is_exhausted(), "the file ended at its last row");
    assert!(read_cursor(&cursor).await.is_none());
    Ok(())
}

/// The dictionaries that the IPC reader keeps between reads are charged to
/// the run, outside a chunk's grant: they stay charged when the chunk's
/// table takes its charge and the rest of the grant goes back, and they go
/// back to the pool when the file ends.
#[tokio::test]
async fn reader_dictionaries_stay_charged_between_chunks() -> Result<()> {
    let batch = mixed(900, 0);
    let schema = batch.schema();
    let manager = spill_manager_for(&schema, SpillCompression::Uncompressed);
    let on = vec![key(&schema)];
    let unit = 48 * 1024;
    let file = write_file(&manager, &[batch], &on, unit)?.expect("a file");
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let cursor = cursor_over(file, manager, unit, &pool);
    cursor.start_chunk();
    let mut chunk = ChunkBuild::new(Arc::clone(&cursor));
    chunk.start(&schema, &on)?;
    let batch = read_cursor(&cursor).await.expect("a batch")?;
    let (keyed, _) = chunk.admit(batch)?;
    assert!(!cursor.is_exhausted(), "the file has more than one batch");
    let dictionaries = footprint::dictionary_bound(&keyed)?;
    assert!(dictionaries > 0);
    let table = chunk.take_charge(chunk.granted());
    // The next read's permit and the reader's dictionaries remain.
    assert_eq!(cursor.charged(), unit + dictionaries);
    drop(table);
    drop(keyed);
    let mut chunk = ChunkBuild::new(Arc::clone(&cursor));
    cursor.start_chunk();
    chunk.start(&schema, &on)?;
    while let Some(batch) = read_cursor(&cursor).await {
        chunk.admit(batch?)?;
    }
    assert!(cursor.is_exhausted());
    chunk.release_permit();
    // The file ended: only the chunk's grant is left.
    assert_eq!(cursor.charged(), chunk.granted());
    Ok(())
}
