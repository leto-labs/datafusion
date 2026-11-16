// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;
use crate::common::collect;
use crate::test::TestMemoryExec;
use arrow::array::{Decimal128Array, Float64Array, Int32Array, Int64Array};
use datafusion_common::test_util::batches_to_sort_string;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::memory_pool::FairSpillPool;
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_functions_aggregate::{
    average::avg_udaf,
    count::count_udaf,
    median::median_udaf,
    min_max::{max_udaf, min_udaf},
    sum::sum_udaf,
};
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_expr::expressions::{binary, cast, col};
use futures::TryStreamExt;
use rand::{Rng, SeedableRng, rngs::StdRng};

// Bypass CG while using the streams selected by the default dispatcher.
fn stock_stream(
    agg: &AggregateExec,
    ctx: &Arc<TaskContext>,
    partition: usize,
) -> Result<SendableRecordBatchStream> {
    assert!(
        ctx.session_config()
            .options()
            .execution
            .enable_migration_aggregate
    );
    Ok(agg.execute_typed(partition, ctx)?.into())
}

type Row = (Option<i32>, Option<i32>, Option<i32>, Option<i64>);

fn context(memory: Option<usize>) -> Arc<TaskContext> {
    let mut config = SessionConfig::new().with_batch_size(7);
    // Aggressive probe settings must not enable skipping in CG's internal
    // single-root representation; grouping sets keep their native reduction.
    config
        .options_mut()
        .execution
        .skip_partial_aggregation_probe_rows_threshold = 1;
    config
        .options_mut()
        .execution
        .skip_partial_aggregation_probe_ratio_threshold = 0.1;
    let mut ctx = TaskContext::default().with_session_config(config);
    if let Some(memory) = memory {
        ctx = ctx.with_runtime(
            RuntimeEnvBuilder::new()
                .with_memory_pool(Arc::new(FairSpillPool::new(memory)))
                .build_arc()
                .unwrap(),
        );
    }
    Arc::new(ctx)
}

fn fixture(
    rows: &[Row],
    keys: usize,
    partitions: usize,
    masks: Vec<Vec<bool>>,
    mode: AggregateMode,
    unsupported: u8,
) -> Result<AggregateExec> {
    let mut fields = (0..keys)
        .map(|i| Field::new(format!("k{i}"), DataType::Int32, true))
        .collect::<Vec<_>>();
    fields.extend([
        Field::new("v", DataType::Int64, true),
        Field::new("d", DataType::Decimal128(10, 2), true),
        Field::new("f", DataType::Float64, true),
    ]);
    let schema = Arc::new(Schema::new(fields));
    let mut data = vec![vec![]; partitions];
    for (partition, batches) in data.iter_mut().enumerate() {
        let selected = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| i % partitions == partition)
            .map(|(_, row)| *row)
            .collect::<Vec<_>>();
        for chunk in selected.chunks(17) {
            let mut columns: Vec<ArrayRef> = (0..keys)
                .map(|i| {
                    Arc::new(Int32Array::from_iter(chunk.iter().map(|row| match i % 3 {
                        0 => row.0,
                        1 => row.1,
                        _ => row.2,
                    }))) as ArrayRef
                })
                .collect();
            columns.push(Arc::new(Int64Array::from_iter(
                chunk.iter().map(|row| row.3),
            )));
            columns.push(Arc::new(
                Decimal128Array::from_iter(
                    chunk.iter().map(|row| row.3.map(|v| i128::from(v) * 7 + 1)),
                )
                .with_precision_and_scale(10, 2)?,
            ));
            columns.push(Arc::new(Float64Array::from_iter(
                chunk.iter().map(|row| row.3.map(|v| v as f64 / 8.0)),
            )));
            batches.push(RecordBatch::try_new(Arc::clone(&schema), columns)?);
        }
    }
    let group = PhysicalGroupBy::new(
        (0..keys)
            .map(|i| {
                let name = format!("k{i}");
                Ok((col(&name, &schema)?, name))
            })
            .collect::<Result<Vec<_>>>()?,
        (0..keys)
            .map(|i| (lit(ScalarValue::Int32(None)), format!("k{i}")))
            .collect(),
        masks,
        true,
    );
    let value = col("v", &schema)?;
    let decimal = col("d", &schema)?;
    let functions = vec![
        (sum_udaf(), vec![Arc::clone(&value)], "sum_v"),
        (count_udaf(), vec![Arc::clone(&value)], "count_v"),
        (count_udaf(), vec![lit(1i64)], "count_all"),
        (min_udaf(), vec![Arc::clone(&value)], "min_v"),
        (max_udaf(), vec![Arc::clone(&value)], "max_v"),
        (
            avg_udaf(),
            vec![cast(Arc::clone(&value), &schema, DataType::Float64)?],
            "avg_v",
        ),
        (avg_udaf(), vec![Arc::clone(&decimal)], "avg_d"),
        (sum_udaf(), vec![decimal], "sum_d"),
        (sum_udaf(), vec![Arc::clone(&value)], "positive_sum"),
        (sum_udaf(), vec![col("f", &schema)?], "sum_f"),
        (avg_udaf(), vec![col("f", &schema)?], "avg_f"),
    ];
    let mut expressions = functions
        .into_iter()
        .map(|(fun, args, alias)| {
            AggregateExprBuilder::new(fun, args)
                .schema(Arc::clone(&schema))
                .alias(alias)
                .build()
                .map(Arc::new)
        })
        .collect::<Result<Vec<_>>>()?;
    if unsupported == 1 {
        expressions.push(Arc::new(
            AggregateExprBuilder::new(count_udaf(), vec![Arc::clone(&value)])
                .schema(Arc::clone(&schema))
                .alias("distinct_v")
                .distinct()
                .build()?,
        ));
    } else if unsupported == 2 {
        expressions.push(Arc::new(
            AggregateExprBuilder::new(
                median_udaf(),
                vec![cast(Arc::clone(&value), &schema, DataType::Float64)?],
            )
            .schema(Arc::clone(&schema))
            .alias("median_v")
            .build()?,
        ));
    }
    let mut filters = vec![None; expressions.len()];
    filters[8] = Some(binary(
        value,
        datafusion_expr::Operator::Gt,
        lit(0i64),
        &schema,
    )?);
    AggregateExec::try_new(
        mode,
        group,
        expressions,
        filters,
        TestMemoryExec::try_new_exec(&data, Arc::clone(&schema), None)?,
        schema,
    )
}

async fn finish(
    agg: &AggregateExec,
    batches: Vec<RecordBatch>,
    ctx: &Arc<TaskContext>,
) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    if agg.mode != AggregateMode::Partial {
        return Ok((Arc::clone(&agg.schema), batches));
    }
    let input = TestMemoryExec::try_new_exec(&[batches], Arc::clone(&agg.schema), None)?;
    let final_agg = AggregateExec::try_new(
        AggregateMode::Final,
        agg.group_by().as_final(),
        agg.aggr_expr().to_vec(),
        vec![None; agg.aggr_expr().len()],
        input,
        Arc::clone(&agg.input_schema),
    )?;
    let schema = Arc::clone(&final_agg.schema);
    Ok((
        schema,
        collect(final_agg.execute(0, Arc::clone(ctx))?).await?,
    ))
}

async fn differential(
    rows: &[Row],
    keys: usize,
    partitions: usize,
    masks: Vec<Vec<bool>>,
    mode: AggregateMode,
    unsupported: u8,
    memory: Option<usize>,
) -> Result<()> {
    let expected = fixture(rows, keys, partitions, masks.clone(), mode, unsupported)?;
    let actual = fixture(rows, keys, partitions, masks, mode, unsupported)?;
    let expected_ctx = context(memory);
    let actual_ctx = context(memory);
    let mut before = vec![];
    let mut after = vec![];
    for partition in 0..partitions {
        // Directly call the expanding implementation. No option or alternate
        // production path is introduced merely to obtain a test oracle.
        before.extend(
            stock_stream(&expected, &expected_ctx, partition)?
                .try_collect::<Vec<_>>()
                .await?,
        );
        after.extend(collect(actual.execute(partition, Arc::clone(&actual_ctx))?).await?);
    }
    let (before_schema, before) = finish(&expected, before, &expected_ctx).await?;
    let (after_schema, after) = finish(&actual, after, &actual_ctx).await?;
    assert_eq!(before_schema, after_schema);
    assert_eq!(
        batches_to_sort_string(&before),
        batches_to_sort_string(&after)
    );
    let selected = actual
        .metrics()
        .unwrap()
        .sum_by_name("chained_grouping_sets")
        .map(|value| value.as_usize())
        .unwrap_or(0);
    assert_eq!(selected > 0, unsupported == 0, "unexpected chain admission");
    if memory.is_some() {
        assert!(
            actual.metrics().unwrap().spill_count().unwrap_or(0) > 0,
            "finite-pool fixture did not exercise retained-state spill"
        );
        assert!(
            actual
                .metrics()
                .unwrap()
                .sum_by_name("chained_merge_early_emits")
                .is_some_and(|count| count.as_usize() > 0),
            "finite-pool fixture did not early-emit the native merge over retained state"
        );
    }
    drop(before);
    drop(after);
    assert_eq!(
        actual_ctx.memory_pool().reserved(),
        0,
        "retained state leaked"
    );
    assert_eq!(actual_ctx.runtime_env().disk_manager.used_disk_space(), 0);
    Ok(())
}

fn shapes() -> Vec<Vec<Vec<bool>>> {
    vec![
        vec![
            vec![false, false, false],
            vec![false, false, true],
            vec![false, true, true],
            vec![true, true, true],
        ],
        (0..8)
            .map(|bits| (0..3).map(|i| bits & (1 << i) != 0).collect())
            .collect(),
        vec![
            vec![false, false, true],
            vec![true, false, false],
            vec![true, false, true],
            vec![true, true, true],
        ],
        vec![
            vec![false, false, false],
            vec![false, false, true],
            vec![false, false, true],
            vec![true, true, true],
            vec![true, true, true],
        ],
    ]
}

#[tokio::test]
async fn chained_grouping_sets_group_only_keeps_native_distinct_path() -> Result<()> {
    let rows = vec![
        (Some(2), Some(1), None, None),
        (Some(5), Some(-40), None, None),
        (Some(1), Some(29), None, None),
        (None, None, None, None),
    ];
    let original = fixture(&rows, 3, 1, shapes().remove(0), AggregateMode::Single, 0)?;
    let agg = AggregateExec::try_new(
        AggregateMode::Single,
        Arc::clone(original.group_by()),
        vec![],
        vec![],
        Arc::clone(&original.input),
        Arc::clone(&original.input_schema),
    )?;
    let ctx = context(None);
    let expected = stock_stream(&agg, &ctx, 0)?.try_collect::<Vec<_>>().await?;
    let actual = collect(agg.execute(0, Arc::clone(&ctx))?).await?;
    assert_eq!(
        batches_to_sort_string(&expected),
        batches_to_sort_string(&actual)
    );
    assert!(
        agg.metrics()
            .unwrap()
            .sum_by_name("chained_grouping_sets")
            .is_none()
    );
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_activation_nulls_and_weighted_avg() -> Result<()> {
    let rows = vec![
        (Some(1), Some(2), Some(3), Some(10)),
        (Some(1), Some(2), Some(3), Some(20)),
        (Some(1), Some(2), Some(4), Some(90)),
        (None, Some(2), Some(3), Some(30)),
        (Some(1), None, Some(3), Some(40)),
        (Some(1), Some(2), None, None),
        (None, None, None, None),
    ];
    for masks in shapes() {
        differential(&rows, 3, 3, masks.clone(), AggregateMode::Partial, 0, None).await?;
        differential(&rows, 3, 1, masks, AggregateMode::Single, 0, None).await?;
    }
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_random_null_properties() -> Result<()> {
    // Generative differential properties, with stable seeds printed on failure.
    // Values stay inside exact integer/decimal ranges so float reassociation
    // does not obscure a grouping/state/NULL defect.
    for seed in 0..96u64 {
        eprintln!("grouping-set property seed {seed}");
        let mut rng = StdRng::seed_from_u64(20261111 + seed);
        let rows = (0..rng.random_range(0..140))
            .map(|_| {
                let mut key = || {
                    if rng.random_range(0..5) == 0 {
                        None
                    } else {
                        Some(rng.random_range(-3..4))
                    }
                };
                let a = key();
                let b = key();
                let c = key();
                let v = if rng.random_range(0..5) == 0 {
                    None
                } else {
                    Some(rng.random_range(-1000..1001))
                };
                (a, b, c, v)
            })
            .collect::<Vec<_>>();
        for masks in shapes() {
            differential(&rows, 3, 3, masks, AggregateMode::Partial, 0, None)
                .await
                .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        }
    }
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_empty_and_wide_duplicate_ids() -> Result<()> {
    let masks = vec![
        vec![false; 8],
        vec![false, false, false, false, false, false, false, true],
        vec![true; 8],
        vec![true; 8],
    ];
    for rows in [
        vec![],
        vec![
            (None, None, None, None),
            (Some(1), Some(2), Some(3), Some(11)),
        ],
    ] {
        for mode in [AggregateMode::Partial, AggregateMode::Single] {
            differential(
                &rows,
                8,
                if mode == AggregateMode::Partial { 3 } else { 1 },
                masks.clone(),
                mode,
                0,
                None,
            )
            .await?;
        }
    }
    for masks in shapes() {
        differential(&[], 3, 3, masks, AggregateMode::Partial, 0, None).await?;
    }
    for rows in [vec![], vec![(None, None, None, Some(5))]] {
        differential(
            &rows,
            0,
            3,
            vec![vec![], vec![]],
            AggregateMode::Partial,
            0,
            None,
        )
        .await?;
    }
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_empty_exchange_opens_each_partition_once() -> Result<()> {
    use crate::Partitioning;
    use crate::repartition::RepartitionExec;

    let original = fixture(&[], 3, 1, shapes().remove(0), AggregateMode::Partial, 0)?;
    let input = Arc::new(RepartitionExec::try_new(
        Arc::clone(&original.input),
        Partitioning::RoundRobinBatch(3),
    )?);
    let actual = AggregateExec::try_new(
        AggregateMode::Partial,
        Arc::clone(original.group_by()),
        original.aggr_expr().to_vec(),
        original.filter_expr().to_vec(),
        input,
        Arc::clone(&original.input_schema),
    )?;
    let ctx = context(None);
    let mut batches = vec![];
    for partition in 0..3 {
        batches.extend(collect(actual.execute(partition, Arc::clone(&ctx))?).await?);
    }
    let (_, result) = finish(&actual, batches, &ctx).await?;
    let expected = stock_stream(&original, &ctx, 0)?
        .try_collect::<Vec<_>>()
        .await?;
    let (_, expected) = finish(&original, expected, &ctx).await?;
    assert_eq!(
        batches_to_sort_string(&expected),
        batches_to_sort_string(&result)
    );
    assert_eq!(ctx.memory_pool().reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_distinct_and_median_fallback() -> Result<()> {
    let rows = vec![
        (Some(1), Some(2), Some(3), Some(7)),
        (Some(1), Some(3), None, Some(7)),
        (None, None, None, None),
    ];
    for unsupported in [1, 2] {
        differential(
            &rows,
            3,
            3,
            shapes().remove(0),
            AggregateMode::Partial,
            unsupported,
            None,
        )
        .await?;
    }
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_drop_releases_retained_state() -> Result<()> {
    let rows = (0..1500)
        .map(|i| (Some(i), Some(i % 7), None, Some(i64::from(i))))
        .collect::<Vec<_>>();
    let agg = fixture(&rows, 3, 1, shapes().remove(0), AggregateMode::Partial, 0)?;
    let ctx = context(Some(16 * 1024 * 1024));
    let mut stream = agg.execute(0, Arc::clone(&ctx))?;
    assert!(stream.try_next().await?.is_some());
    assert!(
        ctx.memory_pool().reserved() > 0,
        "test did not retain state"
    );
    drop(stream);
    assert_eq!(ctx.memory_pool().reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_cancel_during_materialization() -> Result<()> {
    use crate::test::assert_is_pending;
    use crate::test::exec::{BlockingExec, assert_strong_count_converges_to_zero};
    use futures::FutureExt;

    let fixture = fixture(&[], 3, 1, shapes().remove(0), AggregateMode::Partial, 0)?;
    let blocking = Arc::new(BlockingExec::new(Arc::clone(&fixture.input_schema), 1));
    let refs = blocking.refs();
    let agg = Arc::new(AggregateExec::try_new(
        AggregateMode::Partial,
        Arc::clone(fixture.group_by()),
        fixture.aggr_expr().to_vec(),
        fixture.filter_expr().to_vec(),
        blocking,
        Arc::clone(&fixture.input_schema),
    )?);
    let ctx = context(Some(16 * 1024 * 1024));
    let mut future = crate::collect(agg, Arc::clone(&ctx)).boxed();
    assert_is_pending(&mut future);
    drop(future);
    assert_strong_count_converges_to_zero(refs).await;
    assert_eq!(ctx.memory_pool().reserved(), 0);
    assert_eq!(ctx.runtime_env().disk_manager.used_disk_space(), 0);
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_finite_pool_spill_and_cleanup() -> Result<()> {
    let rows = (0..2500)
        .map(|i| (Some(i), Some(i % 5), None, Some(i64::from(i))))
        .collect::<Vec<_>>();
    differential(
        &rows,
        3,
        1,
        shapes().remove(0),
        AggregateMode::Partial,
        0,
        Some(256 * 1024),
    )
    .await
}
