// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::super::super::{
    BuildSpill, Eligible, Expected, LazyMetrics, Output, Part, SkewSupport, SpillOptions,
    SpillShared, SubJoinTemplate, UnmatchedOutput, chunks::ChunkRun,
};
use super::*;
use arrow::array::{UnionArray, builder::StringViewBuilder};
use arrow::datatypes::{UInt8Type, UnionFields};
use datafusion_common::utils::memory::RecordBatchMemoryCounter;
use datafusion_physical_expr::expressions::Literal;

fn spill_batches(manager: &SpillManager, batches: &[RecordBatch]) -> Result<SpilledFile> {
    let file = manager
        .spill_record_batch_and_finish(batches, "round 4 test")?
        .expect("non-empty spill");
    Ok(SpilledFile {
        bytes: file.size().unwrap() as usize,
        rows: batches.iter().map(RecordBatch::num_rows).sum(),
        file,
    })
}

/// An IPC prefix ending after a complete batch decodes successfully. The
/// recorded row count must still prevent a partial Full join result.
#[rstest::rstest]
#[case::short_build(true, true)]
#[case::short_probe(false, true)]
#[case::overrun_build(true, false)]
#[case::overrun_probe(false, false)]
#[tokio::test]
async fn round4_chunked_full_rejects_bad_spill_row_count(
    #[case] damage_build: bool,
    #[case] truncate: bool,
) -> Result<()> {
    let batches = |name, keys: [[i64; 2]; 2]| {
        keys.into_iter()
            .map(|keys| {
                RecordBatch::try_from_iter([(
                    name,
                    Arc::new(Int64Array::from(Vec::from(keys))) as ArrayRef,
                )])
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let build = batches("build", [[1, 2], [3, 4]]);
    let probe = batches("probe", [[1, 5], [3, 6]]);
    let build_schema = build[0].schema();
    let probe_schema = probe[0].schema();
    let context = task_context(spill_config(0, 4096, 0), None, true);
    let manager = spill_manager_for(&build_schema, SpillCompression::Uncompressed);
    let probe_manager = manager.with_schema(Arc::clone(&probe_schema));
    let mut build_file = spill_batches(&manager, &build)?;
    let mut probe_file = spill_batches(&probe_manager, &probe)?;
    let (damaged, damage_manager, batches) = if damage_build {
        (&mut build_file, &manager, &build)
    } else {
        (&mut probe_file, &probe_manager, &probe)
    };
    if truncate {
        let first = spill_batches(damage_manager, &batches[..1])?;
        // Keep the schema and first batch. The last eight bytes of the
        // one-batch stream are its end marker, absent at the batch boundary.
        std::fs::OpenOptions::new()
            .write(true)
            .open(damaged.file.path().unwrap())?
            .set_len((first.bytes - 8) as u64)?;
        let prefix = read_file(damage_manager, damaged).await?;
        assert_eq!(prefix, batches[..1]);
    } else {
        // The complete second message crosses the declared file boundary.
        damaged.rows -= 1;
    }
    let on_left = key(&build_schema);
    let on_right = key(&probe_schema);
    let eligible = Eligible {
        options: SpillOptions::try_new(&context.session_config().options().execution)?,
        level: 0,
        context: Arc::clone(&context),
        metrics: LazyMetrics::Inherited(join_metrics()),
        template: SubJoinTemplate {
            on: vec![(Arc::clone(&on_left), on_right)],
            filter: None,
            join_type: JoinType::Full,
            projection: None,
            null_equality: NullEquality::NullEqualsNothing,
            build_schema: Arc::clone(&build_schema),
            probe_schema: Arc::clone(&probe_schema),
        },
        skew: SkewSupport {
            chunks: true,
            reversal: true,
        },
        on_left: vec![on_left],
        random_state: RandomState::with_seed(0),
        shared: Arc::new(SpillShared::default()),
        mode: PartitionMode::Partitioned,
        count: Expected::Fixed(1),
        shared_build: None,
        depth_limit: None,
        signal: None,
    };
    let mut spill = BuildSpill::new(&eligible);
    spill.manager = Some(manager);
    spill.parts[0] = Part::Written(Some(build_file.clone()));
    let spilled = spill.into_spilled(false).expect("one written partition");
    let output = Arc::new(UnmatchedOutput {
        schema: Arc::new(Schema::new(vec![
            Field::new("build", DataType::Int64, true),
            Field::new("probe", DataType::Int64, true),
        ])),
        column_indices: vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
        ],
        join_type: JoinType::Full,
        build_schema,
        probe_schema,
    });
    let run = ChunkRun {
        build: spilled,
        part: 0,
        run_type: JoinType::Full,
        projection: None,
        build_file,
        probe_files: vec![probe_file],
        output,
        kind: Output::Same,
        context: Arc::clone(&context),
    };
    let error = collect(run.stream()?)
        .await
        .expect_err("Full join accepted a bad spill row count");
    assert!(
        error.to_string().contains("spill file row count"),
        "{error}"
    );
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}

/// The dictionary's view buffers occupy one IPC message after decode.
#[tokio::test]
async fn round4_view_dictionary_uses_shared_ipc_allocation() -> Result<()> {
    let mut strings = StringViewBuilder::new().with_fixed_block_size(16 * 1024);
    for key in 0..64u8 {
        strings.append_value(format!("{key:02}-{}", "x".repeat(16 * 1024 - 3)));
    }
    let values = Arc::new(strings.finish());
    assert_eq!(values.data_buffers().len(), 64);
    let dictionary = Arc::new(DictionaryArray::<UInt8Type>::try_new(
        UInt8Array::from_iter_values(0..64u8),
        values,
    )?) as ArrayRef;
    let batch = RecordBatch::try_from_iter([
        (
            "k",
            Arc::new(Int64Array::from_iter_values(0..64)) as ArrayRef,
        ),
        ("v", dictionary),
    ])?;
    let schema = batch.schema();
    let manager = spill_manager_for(&schema, SpillCompression::Uncompressed);
    let on = vec![key(&schema)];
    let keys = ChunkKeys::try_new(&schema, &on)?;
    let unit = footprint::write_need(&batch, &keys, false)?;
    let dictionary_bound = footprint::dictionary_bound(&batch)?;
    let file = spill_batches(&manager, &[batch.clone(), batch])?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1));
    let cursor = cursor_over(file, manager, unit, &pool);
    for remaining in [1, 0] {
        cursor.start_chunk();
        let mut chunk = ChunkBuild::new(Arc::clone(&cursor));
        chunk.start(&schema, &on)?;
        let decoded = read_cursor(&cursor).await.expect("two IPC batches")?;
        let dictionary = decoded
            .column(1)
            .as_any()
            .downcast_ref::<DictionaryArray<UInt8Type>>()
            .unwrap();
        let values = dictionary.values();
        assert_eq!(
            values
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .data_buffers()
                .len(),
            64
        );
        let unique_bytes = RecordBatchMemoryCounter::new().count_array(values.as_ref());
        let (batch, _) = chunk.admit(decoded)?;
        chunk.release_permit();
        drop(chunk.take_charge(chunk.granted()));
        drop(batch);
        // The writer includes padding and a conservative IPC header allowance.
        assert!(unique_bytes <= dictionary_bound);
        assert!(dictionary_bound - unique_bytes <= 8 * 1024);
        if remaining > 0 {
            assert_eq!(cursor.charged(), unique_bytes);
        } else {
            assert_eq!(cursor.charged(), 0);
            assert!(cursor.is_exhausted());
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExcludedShape {
    Arithmetic,
    ProbeExpression,
    VariableCast,
    CastExpression,
    DictionaryPayload,
    DictionaryKey,
    UnionPayload,
    ListPayload,
    ProbeDictionary,
}

#[rstest::rstest]
#[case::arithmetic(ExcludedShape::Arithmetic)]
#[case::probe_expression(ExcludedShape::ProbeExpression)]
#[case::variable_cast(ExcludedShape::VariableCast)]
#[case::cast_expression(ExcludedShape::CastExpression)]
#[case::dictionary_payload(ExcludedShape::DictionaryPayload)]
#[case::dictionary_key(ExcludedShape::DictionaryKey)]
#[case::union_payload(ExcludedShape::UnionPayload)]
#[case::list_payload(ExcludedShape::ListPayload)]
#[case::probe_dictionary(ExcludedShape::ProbeDictionary)]
#[tokio::test]
async fn round4_excluded_shapes_use_legacy_spill(
    #[case] shape: ExcludedShape,
) -> Result<()> {
    use ExcludedShape::*;
    let input = |prefix, rows, is_build| -> Result<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        for batch in table(prefix, rows, 97, 0, 240) {
            let mut arrays = batch.columns().to_vec();
            if shape == DictionaryKey
                || (is_build && shape == DictionaryPayload)
                || (!is_build && shape == ProbeDictionary)
            {
                let idx = usize::from(shape != DictionaryKey);
                let values = StringArray::from_iter_values(
                    (0..17).map(|row| format!("value-{row}")),
                );
                arrays[idx] = Arc::new(DictionaryArray::<Int32Type>::try_new(
                    Int32Array::from_iter_values(
                        (0..batch.num_rows()).map(|row| (row % 17) as i32),
                    ),
                    Arc::new(values),
                )?);
            }
            if is_build && shape == UnionPayload {
                arrays[1] = Arc::new(UnionArray::try_new(
                    UnionFields::try_new([0], [Field::new("i", DataType::Int32, false)])?,
                    vec![0_i8; batch.num_rows()].into(),
                    None,
                    vec![Arc::clone(&arrays[1])],
                )?);
            }
            if is_build && shape == ListPayload {
                arrays[1] = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
                    (0..batch.num_rows()).map(|row| Some([Some(row as i32)])),
                ));
            }
            let schema = batch.schema();
            batches.push(RecordBatch::try_from_iter(
                schema
                    .fields()
                    .iter()
                    .zip(arrays)
                    .map(|(field, array)| (field.name(), array)),
            )?);
        }
        Ok(batches)
    };
    let build = input("b", 4800, true)?;
    let probe = input("p", 60, false)?;
    let keys = |schema: &SchemaRef, is_build: bool| -> PhysicalExprRef {
        let arithmetic = |value| {
            Arc::new(BinaryExpr::new(
                key(schema),
                Operator::Plus,
                Arc::new(Literal::new(datafusion_common::ScalarValue::Int32(Some(
                    value,
                )))),
            )) as PhysicalExprRef
        };
        match shape {
            Arithmetic => arithmetic(1),
            ProbeExpression if !is_build => arithmetic(0),
            CastExpression => {
                Arc::new(CastExpr::new(arithmetic(1), DataType::Int64, None))
            }
            VariableCast => {
                Arc::new(CastExpr::new(key(schema), DataType::LargeUtf8, None))
            }
            _ => key(schema),
        }
    };
    let plan = || -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(HashJoinExec::try_new(
            memory_input(&build, 1),
            memory_input(&probe, 1),
            vec![(
                keys(&build[0].schema(), true),
                keys(&probe[0].schema(), false),
            )],
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    };
    let expected = run(&plan()?, task_context(SessionConfig::new(), None, true)).await?;
    let spill = plan()?;
    let context = task_context(spill_config(1, 8192, 3), None, true);
    let actual = run(&spill, Arc::clone(&context)).await?;
    assert_eq!(
        batches_to_sort_string(&actual),
        batches_to_sort_string(&expected)
    );
    let metrics = spill.metrics().unwrap();
    assert!(
        metric(&metrics, "spilled_partitions").unwrap_or_default() > 0,
        "{shape:?}"
    );
    assert_eq!(
        metric(&metrics, "spill_reversed_runs").unwrap_or_default(),
        0,
        "{shape:?}"
    );
    assert_eq!(
        metric(&metrics, "spill_chunked_runs").unwrap_or_default(),
        0,
        "{shape:?}"
    );
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}
