// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::GroupValuesColumn;
use crate::aggregates::group_values::GroupValues;
use arrow::array::{Array, ArrayRef, Int32Array, Int64Array, StringViewArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_expr::EmitTo;
use std::sync::Arc;

fn key(cols: &[ArrayRef], row: usize) -> (Option<i64>, Option<i32>, Option<String>) {
    let a = cols[0].as_any().downcast_ref::<Int64Array>().unwrap();
    let b = cols[1].as_any().downcast_ref::<Int32Array>().unwrap();
    let c = cols[2].as_any().downcast_ref::<StringViewArray>().unwrap();
    (
        a.is_valid(row).then(|| a.value(row)),
        b.is_valid(row).then(|| b.value(row)),
        c.is_valid(row).then(|| c.value(row).to_owned()),
    )
}

fn intern_with_hashes(
    store: &mut GroupValuesColumn<false>,
    cols: &[ArrayRef],
    hashes: &[u64],
) -> Vec<usize> {
    let mut groups = vec![usize::MAX; hashes.len()];
    store.collect_vectorized_process_context(hashes, &mut groups);
    store.vectorized_append(cols).unwrap();
    store.vectorized_equal_to(cols, &mut groups);
    store
        .scalarized_intern_remaining(cols, hashes, &mut groups)
        .unwrap();
    groups
}

fn check_reset<const STREAMING: bool>() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Utf8View, true),
    ]));
    let mut values = GroupValuesColumn::<STREAMING>::try_new(schema).unwrap();
    for cycle in 0..3 {
        let ids: Vec<_> = (0..8192).map(|row| (row / 7 % 173) as i64).collect();
        let a = Int64Array::from(
            ids.iter()
                .map(|id| (id % 13 != 0).then_some(*id + cycle * 173))
                .collect::<Vec<_>>(),
        );
        let b = Int32Array::from(
            ids.iter()
                .map(|id| (id % 17 != 0).then_some((id % 60) as i32))
                .collect::<Vec<_>>(),
        );
        let c = StringViewArray::from_iter(
            ids.iter()
                .map(|id| (id % 19 != 0).then(|| format!("shared-prefix-value-{id:04}")))
                .collect::<Vec<_>>(),
        );
        let input: Vec<ArrayRef> = vec![Arc::new(a), Arc::new(b), Arc::new(c)];
        let mut groups = Vec::new();
        values.intern(&input, &mut groups).unwrap();
        assert!(values.len() > 64);
        values.emit(EmitTo::First(7)).unwrap();
        values.intern(&input, &mut groups).unwrap();
        let output = values.emit(EmitTo::All).unwrap();
        for (row, group) in groups.iter().enumerate() {
            assert_eq!(key(&input, row), key(&output, *group));
        }
        values.intern(&input, &mut groups).unwrap();
        values.clear_shrink(8192);
        assert_eq!(values.len(), 0);
    }
}

#[test]
fn repeated_hash_candidates_keep_full_key_equality_after_prefix_and_clear() {
    check_reset::<false>();
}

#[test]
fn streaming_groups_reset_lookup_after_full_emission() {
    check_reset::<true>();
}

#[test]
fn mixed_column_appends_keep_ids_for_nulls_and_hash_collisions() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Utf8View, true),
    ]));
    let mut store = GroupValuesColumn::<false>::try_new(schema).unwrap();
    let keys: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![Some(10), None, Some(10), Some(10)])),
        Arc::new(Int32Array::from(vec![Some(2), Some(2), None, Some(2)])),
        Arc::new(StringViewArray::from(vec![
            Some("first long value with a shared prefix"),
            Some(""),
            None,
            Some("first long value with a shared prefix but different suffix"),
        ])),
    ];

    // Exercise the production append and equality kernels with two distinct
    // long strings that share a full hash. Put their repeats in the same batch
    // and then probe them again once the collision list is in the map.
    let intern_selected = |store: &mut GroupValuesColumn<false>, order: &[u32]| {
        let indices = arrow::array::UInt32Array::from(order.to_vec());
        let cols = keys
            .iter()
            .map(|col| arrow::compute::take(col, &indices, None).unwrap())
            .collect::<Vec<_>>();
        let hashes = order
            .iter()
            .map(|&index| [17, 29, 33, 17][index as usize])
            .collect::<Vec<_>>();
        intern_with_hashes(store, &cols, &hashes)
    };
    assert_eq!(
        intern_selected(&mut store, &[0, 3, 1, 2, 3, 2, 0, 1]),
        [0, 3, 1, 2, 3, 2, 0, 1]
    );
    assert_eq!(
        intern_selected(&mut store, &[3, 2, 1, 0, 3]),
        [3, 2, 1, 0, 3]
    );

    let head = store.emit(EmitTo::First(2)).unwrap();
    assert_eq!(key(&head, 0), key(&keys, 0));
    assert_eq!(key(&head, 1), key(&keys, 1));
    assert_eq!(intern_selected(&mut store, &[3, 2, 1, 0]), [1, 0, 2, 3]);
    let rest = store.emit(EmitTo::All).unwrap();
    for (row, original) in [2, 3, 1, 0].into_iter().enumerate() {
        assert_eq!(key(&rest, row), key(&keys, original));
    }
}

#[test]
fn prefix_emit_preserves_two_collision_lists_in_reverse_bucket_order() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let mut store = GroupValuesColumn::<false>::try_new(schema).unwrap();
    let mut intern = |keys: Vec<i64>, hashes: &[u64]| {
        intern_with_hashes(&mut store, &[Arc::new(Int64Array::from(keys))], hashes)
    };

    assert_eq!(intern(vec![0, 1, 2], &[0, 1, 2]), [0, 1, 2]);
    // The buckets for hashes 1 and 2 already exist. Create their collision
    // lists in the opposite order from the bucket traversal.
    assert_eq!(intern(vec![4, 3], &[2, 1]), [3, 4]);

    let emitted = store.emit(EmitTo::First(1)).unwrap();
    assert_eq!(
        emitted[0]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[0]
    );
    assert_eq!(
        intern_with_hashes(
            &mut store,
            &[Arc::new(Int64Array::from(vec![1, 2, 3, 4]))],
            &[1, 2, 1, 2],
        ),
        [0, 1, 3, 2]
    );
    assert_eq!(store.len(), 4);

    let remaining = store.emit(EmitTo::All).unwrap();
    assert_eq!(
        remaining[0]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[1, 2, 4, 3]
    );
}

// This fixture needs two distinct hashes with one collision chain each.
#[cfg(not(feature = "force_hash_collisions"))]
#[tokio::test]
async fn partial_aggregate_keeps_colliding_groups_during_early_output()
-> datafusion_common::Result<()> {
    use crate::ExecutionPlan;
    use crate::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
    use crate::common::collect;
    use crate::expressions::col;
    use crate::test::TestMemoryExec;
    use arrow::array::{RecordBatch, StringArray};
    use datafusion_execution::TaskContext;
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_functions_aggregate::count::count_udaf;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::lit;

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Utf8, true),
        Field::new("c", DataType::Utf8, true),
    ]));
    let group_by = PhysicalGroupBy::new_single(
        ["a", "b", "c"]
            .into_iter()
            .map(|name| Ok((col(name, &schema)?, name.to_owned())))
            .collect::<datafusion_common::Result<Vec<_>>>()?,
    );
    let count = Arc::new(
        AggregateExprBuilder::new(count_udaf(), vec![lit(1i64)])
            .schema(Arc::clone(&schema))
            .alias("count")
            .build()?,
    );
    let mut config = SessionConfig::new()
        .with_batch_size(5)
        .set_bool("datafusion.execution.enable_migration_aggregate", false);
    config
        .options_mut()
        .execution
        .skip_partial_aggregation_probe_rows_threshold = usize::MAX;

    // Hashing skips a NULL Utf8 column. With the same first column, swapping
    // a valid Utf8 value and NULL between the other two columns collides.
    // Try both list-creation orders for the same hash-table buckets.
    for tail in [["left", "right"], ["right", "left"]] {
        let fixture = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1; 9])),
                Arc::new(StringArray::from(vec![
                    Some("zero"),
                    Some("one"),
                    Some("two"),
                    Some("three"),
                    Some("four"),
                    Some("left"),
                    Some("right"),
                    None,
                    None,
                ])),
                Arc::new(StringArray::from(vec![
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(tail[0]),
                    Some(tail[1]),
                ])),
            ],
        )?;
        let repeat = fixture.slice(5, 4);
        let input = TestMemoryExec::try_new_exec(
            &[vec![fixture, repeat]],
            Arc::clone(&schema),
            None,
        )?;
        let aggregate = AggregateExec::try_new(
            AggregateMode::Partial,
            group_by.clone(),
            vec![Arc::clone(&count)],
            vec![None],
            input,
            Arc::clone(&schema),
        )?;
        // The first refusal emits a batch of five groups and retains both
        // chains. The next batch repeats the four retained keys.
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(1, 1.0)
            .build_arc()?;
        let ctx = Arc::new(
            TaskContext::default()
                .with_session_config(config.clone())
                .with_runtime(runtime),
        );
        let output = collect(aggregate.execute(0, ctx)?).await?;
        assert_eq!(output[0].num_rows(), 5);
        let suffix_counts = output[1..]
            .iter()
            .flat_map(|batch| {
                batch
                    .column(3)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            suffix_counts,
            [Some(2); 4],
            "retained keys with collision order {tail:?}"
        );
    }

    Ok(())
}

#[tokio::test]
async fn mixed_column_appends_spill_and_merge_nulls() -> datafusion_common::Result<()> {
    use crate::ExecutionPlan;
    use crate::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
    use crate::common::collect;
    use crate::expressions::col;
    use crate::test::TestMemoryExec;
    use arrow::array::RecordBatch;
    use datafusion_common::test_util::batches_to_sort_string;
    use datafusion_execution::TaskContext;
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::memory_pool::FairSpillPool;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_functions_aggregate::count::count_udaf;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::lit;

    const GROUPS: i32 = 65_536;
    const BATCH_ROWS: i32 = 256;
    const REPEATS: i64 = 4;

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Utf8View, true),
    ]));
    let batches = (0..GROUPS * REPEATS as i32 / BATCH_ROWS)
        .map(|batch| {
            let keys = (batch * BATCH_ROWS..(batch + 1) * BATCH_ROWS)
                .map(|i| (i * 1543) % GROUPS)
                .collect::<Vec<_>>();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter(
                        keys.iter().map(|&k| (k % 29 != 0).then_some(i64::from(k))),
                    )),
                    Arc::new(Int32Array::from_iter(
                        keys.iter().map(|&k| (k % 97 != 0).then_some(k % 64)),
                    )),
                    Arc::new(StringViewArray::from_iter(keys.iter().map(|k| {
                        (k % 37 != 0).then(|| format!("mixed-width-spill-key-{k:04}"))
                    }))),
                ],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let group_by = PhysicalGroupBy::new_single(
        ["a", "b", "c"]
            .into_iter()
            .map(|name| Ok((col(name, &schema)?, name.to_owned())))
            .collect::<datafusion_common::Result<Vec<_>>>()?,
    );
    let count = AggregateExprBuilder::new(count_udaf(), vec![lit(1i64)])
        .schema(Arc::clone(&schema))
        .alias("count")
        .build()?;
    let run = async |pool_bytes: Option<usize>| {
        let mut config = SessionConfig::new().with_batch_size(BATCH_ROWS as usize);
        config.options_mut().execution.sort_spill_reservation_bytes = 4096;
        let runtime = RuntimeEnvBuilder::new();
        let runtime = match pool_bytes {
            Some(bytes) => runtime.with_memory_pool(Arc::new(FairSpillPool::new(bytes))),
            None => runtime,
        };
        let ctx = Arc::new(
            TaskContext::default()
                .with_session_config(config)
                .with_runtime(runtime.build_arc()?),
        );
        let aggregate = AggregateExec::try_new(
            AggregateMode::Single,
            group_by.clone(),
            vec![Arc::new(count.clone())],
            vec![None],
            TestMemoryExec::try_new_exec(
                std::slice::from_ref(&batches),
                Arc::clone(&schema),
                None,
            )?,
            Arc::clone(&schema),
        )?;
        let output = collect(aggregate.execute(0, Arc::clone(&ctx))?).await?;
        assert_eq!(ctx.memory_pool().reserved(), 0);
        let spills = aggregate.metrics().unwrap().spill_count().unwrap();
        Ok::<_, datafusion_common::DataFusionError>((output, spills))
    };

    let (expected, baseline_spills) = run(None).await?;
    // Byte-view prefix emission can retain a fresh 2 MiB payload block.
    // Leave room for that block during replay while forcing repeated spills.
    let (actual, spills) = run(Some(4 * 1024 * 1024)).await?;
    assert_eq!(baseline_spills, 0);
    assert_eq!(
        expected.iter().map(RecordBatch::num_rows).sum::<usize>(),
        GROUPS as usize
    );
    for batch in &expected {
        let counts = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(counts.iter().all(|count| count == Some(REPEATS)));
    }
    assert!(
        spills > 1,
        "expected multiple mixed-key spill runs, got {spills}"
    );
    assert_eq!(
        batches_to_sort_string(&actual),
        batches_to_sort_string(&expected)
    );
    Ok(())
}
