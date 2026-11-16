// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;
use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_execution::{memory_pool::FairSpillPool, runtime_env::RuntimeEnvBuilder};
use datafusion_physical_expr::expressions::col;
use futures::TryStreamExt;

fn make_state(k: usize, limit: usize) -> (PartitionedTopK, Arc<RuntimeEnv>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Float64, true),
    ]));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(FairSpillPool::new(limit)))
        .build_arc()
        .unwrap();
    let orders =
        LexOrdering::new([
            PhysicalSortExpr::new_default(col("v", &schema).unwrap()).desc()
        ])
        .unwrap();
    let state = PartitionedTopK::try_new(
        0,
        &schema,
        vec![col("k", &schema).unwrap()],
        vec![SortField::new(DataType::Int64)],
        orders,
        k,
        17,
        &runtime,
        &ExecutionPlanMetricsSet::new(),
        None,
    )
    .unwrap();
    (state, runtime)
}

#[tokio::test]
async fn compact_row_number_batches_across_groups_and_releases_memory() -> Result<()> {
    for k in [1, 2, 20] {
        let (mut state, runtime) = make_state(k, 64 * 1024 * 1024);
        for _ in 0..3 {
            let batch = RecordBatch::try_new(
                Arc::clone(&state.core.output_schema),
                vec![
                    Arc::new(Int64Array::from_iter(
                        (0..100).map(|i| if i == 0 { None } else { Some(i) }),
                    )),
                    Arc::new(Float64Array::from(vec![Some(7.0); 100])),
                ],
            )?;
            state.insert_batch(&batch)?;
        }
        assert!(runtime.memory_pool.reserved() > 0);
        let batches: Vec<_> = state.emit()?.try_collect().await?;
        assert_eq!(
            batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            100 * k.min(3)
        );
        assert!(batches.iter().all(|b| b.num_rows() <= 17));
        assert_eq!(batches.len(), (100 * k.min(3)).div_ceil(17));
        assert_eq!(runtime.memory_pool.reserved(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn compact_row_number_paused_stream_and_refusal_return_credits() -> Result<()> {
    use futures::StreamExt;
    let (mut state, runtime) = make_state(2, 8 * 1024 * 1024);
    let batch = RecordBatch::try_new(
        Arc::clone(&state.core.output_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..100)),
            Arc::new(Float64Array::from_iter_values((0..100).map(|i| i as f64))),
        ],
    )?;
    state.insert_batch(&batch)?;
    let mut stream = state.emit()?;
    stream.next().await.unwrap()?;
    assert!(runtime.memory_pool.reserved() > 0);
    drop(stream);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    let (mut state, runtime) = make_state(2, 64 * 1024);
    let batch = RecordBatch::try_new(
        Arc::clone(&state.core.output_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..10_000)),
            Arc::new(Float64Array::from_iter_values(
                (0..10_000).map(|i| i as f64),
            )),
        ],
    )?;
    let error = state.insert_batch(&batch).unwrap_err();
    assert!(error.to_string().contains("PartitionedTopK"), "{error}");
    drop(state);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn compact_row_number_many_groups_match_sorted_values() -> Result<()> {
    use arrow::array::Array;
    let (mut state, runtime) = make_state(2, 128 * 1024 * 1024);
    let mut expected = vec![Vec::<f64>::new(); 10_000];
    for batch in 0..20 {
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for row in 0..4096 {
            let key = (batch * 4096 + row) % expected.len();
            let value = ((batch * 97 + row * 13) % 10_007) as f64;
            expected[key].push(value);
            keys.push(key as i64);
            values.push(value);
        }
        let input = RecordBatch::try_new(
            Arc::clone(&state.core.output_schema),
            vec![
                Arc::new(Int64Array::from(keys)),
                Arc::new(Float64Array::from(values)),
            ],
        )?;
        state.insert_batch(&input)?;
    }
    println!("row_number_10k_groups_retained_bytes={}", state.size());
    let output: Vec<_> = state.emit()?.try_collect().await?;
    let mut actual = vec![Vec::<f64>::new(); expected.len()];
    for batch in output {
        let keys = batch
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        let values = batch
            .column(1)
            .as_primitive::<arrow::datatypes::Float64Type>();
        assert_eq!(keys.null_count(), 0);
        for row in 0..batch.num_rows() {
            actual[keys.value(row) as usize].push(values.value(row));
        }
    }
    for values in &mut expected {
        values.sort_by(|a, b| b.total_cmp(a));
        values.truncate(2);
    }
    assert_eq!(actual, expected);
    assert_eq!(runtime.memory_pool.reserved(), 0);
    Ok(())
}
#[tokio::test]
async fn cached_worst_prefix_keeps_long_string_order_and_nulls() -> Result<()> {
    use arrow::array::{Array, StringViewArray};
    use arrow::compute::SortOptions;
    for k in [1, 2, 5] {
        for descending in [false, true] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v", DataType::Utf8View, true),
            ]));
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_pool(Arc::new(FairSpillPool::new(64 * 1024 * 1024)))
                .build_arc()?;
            let orders = LexOrdering::new([PhysicalSortExpr {
                expr: col("v", &schema)?,
                options: SortOptions {
                    descending,
                    nulls_first: false,
                },
            }])
            .unwrap();
            let mut state = PartitionedTopK::try_new(
                0,
                &schema,
                vec![col("k", &schema)?],
                vec![SortField::new(DataType::Int64)],
                orders,
                k,
                7,
                &runtime,
                &ExecutionPlanMetricsSet::new(),
                None,
            )?;
            let mut expected = vec![Vec::new(); 3];
            for batch in 0..4 {
                let mut keys = Vec::new();
                let mut strings = Vec::new();
                for row in 0..90 {
                    let key = row % 3;
                    let value = (row % 11 != 0).then(|| {
                        format!(
                            "this-is-a-common-prefix-longer-than-the-bound-{:04}",
                            (batch * 97 + row * 13) % 101
                        )
                    });
                    expected[key].push(value.clone());
                    keys.push(key as i64);
                    strings.push(value);
                }
                let input = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int64Array::from(keys)),
                        Arc::new(StringViewArray::from_iter(strings)),
                    ],
                )?;
                state.insert_batch(&input)?;
            }
            let mut actual = vec![Vec::new(); 3];
            let output: Vec<_> = state.emit()?.try_collect().await?;
            for batch in output {
                let keys = batch
                    .column(0)
                    .as_primitive::<arrow::datatypes::Int64Type>();
                let values = batch.column(1).as_string_view();
                for row in 0..batch.num_rows() {
                    actual[keys.value(row) as usize]
                        .push(values.is_valid(row).then(|| values.value(row).to_owned()));
                }
            }
            for values in &mut expected {
                values.sort_by(|a, b| match (a, b) {
                    (None, None) => Ordering::Equal,
                    (None, Some(_)) => Ordering::Greater,
                    (Some(_), None) => Ordering::Less,
                    (Some(a), Some(b)) if descending => b.cmp(a),
                    (Some(a), Some(b)) => a.cmp(b),
                });
                values.truncate(k);
            }
            assert_eq!(actual, expected);
            assert_eq!(runtime.memory_pool.reserved(), 0);
        }
    }
    Ok(())
}
