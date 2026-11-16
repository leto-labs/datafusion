// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;
use crate::common::collect;
use crate::test::TestMemoryExec;
use arrow::array::{
    BooleanArray, DictionaryArray, Float64Array, Int32Array, Int64Array, ListArray,
    StringArray, StringViewArray,
};
use arrow::datatypes::Int32Type;
use datafusion_common::DataFusionError;
use datafusion_common::test_util::batches_to_sort_string;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::memory_pool::FairSpillPool;
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_functions_aggregate::{
    average::avg_udaf,
    count::count_udaf,
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

#[derive(Clone, Copy, Debug)]
struct Ctx {
    batch: usize,
    skip: bool,
    memory: Option<usize>,
    disk: bool,
    groups: usize,
}

fn context(c: Ctx) -> Arc<TaskContext> {
    let mut config = SessionConfig::new().with_batch_size(c.batch);
    if c.skip {
        config
            .options_mut()
            .execution
            .skip_partial_aggregation_probe_rows_threshold = 1;
        config
            .options_mut()
            .execution
            .skip_partial_aggregation_probe_ratio_threshold = 0.1;
    }
    if c.groups > 0 {
        config = bounded_partial::test_option::with_groups(config, c.groups);
    }
    let mut ctx = TaskContext::default().with_session_config(config);
    let mut builder = RuntimeEnvBuilder::new();
    if let Some(memory) = c.memory {
        builder = builder.with_memory_pool(Arc::new(FairSpillPool::new(memory)));
    }
    if !c.disk {
        use datafusion_execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
        builder = builder.with_disk_manager_builder(
            DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
        );
    }
    ctx = ctx.with_runtime(builder.build_arc().unwrap());
    Arc::new(ctx)
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("s", DataType::Utf8, true),
        Field::new(
            "d",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        ),
        Field::new("f", DataType::Float64, true),
        Field::new("b", DataType::Boolean, true),
        Field::new("sv", DataType::Utf8View, true),
        Field::new(
            "l",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
            true,
        ),
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int64, true),
        Field::new("t", DataType::Utf8, true),
        Field::new("x", DataType::Float64, true),
    ]))
}

fn data(
    seed: u64,
    rows: usize,
    partitions: usize,
    chunk: usize,
) -> Vec<Vec<RecordBatch>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let schema = schema();
    let strings = ["a", "b", "", "zz"];
    let floats = [f64::NAN, -0.0, 0.0, 1.5, -2.25, f64::INFINITY];
    let mut out = vec![vec![]; partitions];
    let mut remaining = rows;
    let mut partition = 0;
    // Always include some zero-row batches.
    loop {
        let n = remaining.min(chunk);
        let opt = |rng: &mut StdRng, m: usize| -> Option<usize> {
            if rng.random_range(0..5) == 0 {
                None
            } else {
                Some(rng.random_range(0..m))
            }
        };
        let s: Vec<Option<&str>> = (0..n)
            .map(|_| opt(&mut rng, 4).map(|i| strings[i]))
            .collect();
        let d: Vec<Option<&str>> = (0..n)
            .map(|_| opt(&mut rng, 3).map(|i| strings[i]))
            .collect();
        let f: Vec<Option<f64>> = (0..n)
            .map(|_| opt(&mut rng, 6).map(|i| floats[i]))
            .collect();
        let b: Vec<Option<bool>> =
            (0..n).map(|_| opt(&mut rng, 2).map(|i| i == 0)).collect();
        let sv: Vec<Option<&str>> = (0..n)
            .map(|_| opt(&mut rng, 4).map(|i| strings[i]))
            .collect();
        let l: Vec<Option<Vec<Option<i32>>>> = (0..n)
            .map(|_| {
                opt(&mut rng, 3).map(|i| match i {
                    0 => vec![],
                    1 => vec![Some(1), None],
                    _ => vec![Some(1)],
                })
            })
            .collect();
        let k: Vec<i32> = (0..n).map(|_| rng.random_range(-2..3)).collect();
        let v: Vec<Option<i64>> = (0..n)
            .map(|_| opt(&mut rng, 2001).map(|i| i as i64 - 1000))
            .collect();
        let t: Vec<Option<&str>> = (0..n)
            .map(|_| opt(&mut rng, 4).map(|i| strings[i]))
            .collect();
        let x: Vec<Option<f64>> = (0..n)
            .map(|_| opt(&mut rng, 4001).map(|i| (i as f64 - 2000.0) / 8.0))
            .collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(s)),
                Arc::new(d.into_iter().collect::<DictionaryArray<Int32Type>>()),
                Arc::new(Float64Array::from(f)),
                Arc::new(BooleanArray::from(b)),
                Arc::new(StringViewArray::from(sv)),
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(l)),
                Arc::new(Int32Array::from(k)),
                Arc::new(Int64Array::from(v)),
                Arc::new(StringArray::from(t)),
                Arc::new(Float64Array::from(x)),
            ],
        )
        .unwrap();
        out[partition % partitions].push(batch);
        partition += 1;
        if remaining == 0 {
            break;
        }
        remaining -= n;
    }
    out
}

/// keys: 0 = columns s,d,f; 1 = sv,b,l; 2 = expressions (k+1 as kk, v%3 as m, s as "renamed")
fn build(
    data: &[Vec<RecordBatch>],
    keys: u8,
    masks: Vec<Vec<bool>>,
    mode: AggregateMode,
) -> Result<AggregateExec> {
    let schema = schema();
    let exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = match keys {
        0 => vec![
            (col("s", &schema)?, "s".into()),
            (col("d", &schema)?, "d".into()),
            (col("f", &schema)?, "f".into()),
        ],
        1 => vec![
            (col("sv", &schema)?, "sv".into()),
            (col("b", &schema)?, "b".into()),
            (col("l", &schema)?, "l".into()),
        ],
        3 => vec![
            (col("v", &schema)?, "v".into()),
            (col("x", &schema)?, "x".into()),
            (col("s", &schema)?, "s".into()),
        ],
        _ => vec![
            (
                binary(
                    col("k", &schema)?,
                    datafusion_expr::Operator::Plus,
                    lit(1i32),
                    &schema,
                )?,
                "kk".into(),
            ),
            (
                binary(
                    col("v", &schema)?,
                    datafusion_expr::Operator::Modulo,
                    lit(3i64),
                    &schema,
                )?,
                "m".into(),
            ),
            (col("s", &schema)?, "renamed".into()),
        ],
    };
    let nulls = exprs
        .iter()
        .map(|(e, name)| {
            Ok((
                lit(ScalarValue::try_new_null(&e.data_type(&schema)?)?),
                name.clone(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let group = PhysicalGroupBy::new(exprs, nulls, masks, true);
    let v = col("v", &schema)?;
    let functions = vec![
        (sum_udaf(), vec![Arc::clone(&v)], "sum_v"),
        (count_udaf(), vec![Arc::clone(&v)], "count_v"),
        (count_udaf(), vec![lit(1i64)], "count_all"),
        (min_udaf(), vec![Arc::clone(&v)], "min_v"),
        (max_udaf(), vec![Arc::clone(&v)], "max_v"),
        (
            avg_udaf(),
            vec![cast(Arc::clone(&v), &schema, DataType::Float64)?],
            "avg_v",
        ),
        (min_udaf(), vec![col("t", &schema)?], "min_t"),
        (max_udaf(), vec![col("t", &schema)?], "max_t"),
        (sum_udaf(), vec![col("x", &schema)?], "sum_x"),
        (avg_udaf(), vec![col("x", &schema)?], "avg_x"),
        (min_udaf(), vec![Arc::clone(&v)], "min_v_repeat"),
        (sum_udaf(), vec![Arc::clone(&v)], "filtered_sum"),
        (count_udaf(), vec![lit(1i64)], "filtered_count"),
    ];
    let expressions = functions
        .into_iter()
        .map(|(fun, args, alias)| {
            AggregateExprBuilder::new(fun, args)
                .schema(Arc::clone(&schema))
                .alias(alias)
                .build()
                .map(Arc::new)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut filters = vec![None; expressions.len()];
    let positive = binary(v, datafusion_expr::Operator::Gt, lit(0i64), &schema)?;
    filters[11] = Some(Arc::clone(&positive));
    filters[12] = Some(positive);
    AggregateExec::try_new(
        mode,
        group,
        expressions,
        filters,
        TestMemoryExec::try_new_exec(data, Arc::clone(&schema), None)?,
        schema,
    )
}

async fn finish(
    agg: &AggregateExec,
    batches: Vec<RecordBatch>,
    ctx: &Arc<TaskContext>,
) -> Result<Vec<RecordBatch>> {
    if agg.mode != AggregateMode::Partial {
        return Ok(batches);
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
    collect(final_agg.execute(0, Arc::clone(ctx))?).await
}

/// Returns (stock, chained, chained admitted).
async fn run(
    data: &[Vec<RecordBatch>],
    keys: u8,
    masks: Vec<Vec<bool>>,
    mode: AggregateMode,
    c: Ctx,
) -> (Result<String>, Result<String>, bool) {
    let partitions = data.len();
    let expected = build(data, keys, masks.clone(), mode).unwrap();
    let actual = build(data, keys, masks, mode).unwrap();
    let ectx = context(Ctx {
        memory: None,
        disk: true,
        ..c
    });
    let actx = context(c);
    let stock = async {
        let mut before = vec![];
        for partition in 0..partitions {
            before.extend(
                stock_stream(&expected, &ectx, partition)?
                    .try_collect::<Vec<_>>()
                    .await?,
            );
        }
        let before = finish(&expected, before, &ectx).await?;
        Ok(batches_to_sort_string(&before))
    }
    .await;
    let chained = async {
        let mut after = vec![];
        for partition in 0..partitions {
            after.extend(collect(actual.execute(partition, Arc::clone(&actx))?).await?);
        }
        let after = finish(&actual, after, &context(Ctx { memory: None, ..c })).await?;
        Ok(batches_to_sort_string(&after))
    }
    .await;
    let admitted = actual
        .metrics()
        .unwrap()
        .sum_by_name("chained_grouping_sets")
        .is_some();
    assert_eq!(actx.memory_pool().reserved(), 0, "leaked reservation");
    assert_eq!(actx.runtime_env().disk_manager.used_disk_space(), 0);
    (stock, chained, admitted)
}

fn shapes() -> Vec<(&'static str, Vec<Vec<bool>>)> {
    let m = |rows: &[[bool; 3]]| rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>();
    const T: bool = true;
    const F: bool = false;
    vec![
        ("rollup", m(&[[F, F, F], [F, F, T], [F, T, T], [T, T, T]])),
        (
            "cube",
            (0..8)
                .map(|bits| (0..3).map(|i| bits & (1 << i) != 0).collect())
                .collect(),
        ),
        (
            "reverse-rollup",
            m(&[[T, T, T], [T, T, F], [T, F, F], [F, F, F]]),
        ),
        ("no-total", m(&[[F, F, T], [F, T, T]])),
        (
            "two-roots-shared-child",
            m(&[[F, F, T], [F, T, F], [F, T, T]]),
        ),
        ("two-roots-total", m(&[[F, T, T], [T, F, T], [T, T, T]])),
        (
            "dups-of-incomparable",
            m(&[[F, T, T], [T, F, T], [F, T, T]]),
        ),
        (
            "dups-everywhere",
            m(&[
                [T, T, T],
                [F, F, F],
                [T, T, T],
                [F, F, F],
                [F, T, T],
                [F, T, T],
                [T, T, T],
            ]),
        ),
        ("root-omits-key", m(&[[T, F, F], [T, T, F], [T, T, T]])),
        ("total-twice", m(&[[T, T, T], [T, T, T]])),
    ]
}

const BASE: Ctx = Ctx {
    batch: 7,
    skip: true,
    memory: None,
    disk: true,
    groups: 0,
};

#[tokio::test]
async fn chained_grouping_sets_types_modes_batches_and_kernels() {
    for seed in 0..16 {
        let rows = [0, 1, 3, 79, 170, 301][seed as usize % 6];
        for (name, masks) in shapes() {
            for keys in 0..3 {
                for (mode, parts) in [
                    (AggregateMode::Partial, 3),
                    (AggregateMode::Single, 1),
                    (AggregateMode::SinglePartitioned, 3),
                ] {
                    for c in [
                        BASE,
                        Ctx {
                            batch: 8192,
                            skip: false,
                            ..BASE
                        },
                        Ctx {
                            batch: 64,
                            skip: false,
                            ..BASE
                        },
                    ] {
                        let input = data(900 + seed, rows, parts, 41);
                        let (stock, chained, admitted) =
                            run(&input, keys, masks.clone(), mode, c).await;
                        assert!(admitted, "{seed} {name} {keys} {mode:?} {c:?}");
                        assert_eq!(
                            stock.unwrap(),
                            chained.unwrap(),
                            "{seed} {name} {keys} {mode:?} {c:?}"
                        );
                    }
                }
            }
        }
    }
}

async fn same_pool_stock(
    input: &[Vec<RecordBatch>],
    masks: Vec<Vec<bool>>,
    mode: AggregateMode,
    c: Ctx,
) -> Result<()> {
    let agg = build(input, 3, masks, mode)?;
    let ctx = context(c);
    let result = async {
        for partition in 0..input.len() {
            stock_stream(&agg, &ctx, partition)?
                .try_collect::<Vec<_>>()
                .await?;
        }
        Ok(())
    }
    .await;
    assert_eq!(ctx.memory_pool().reserved(), 0);
    assert_eq!(ctx.runtime_env().disk_manager.used_disk_space(), 0);
    result
}

#[tokio::test]
async fn chained_grouping_sets_tight_pool_partial_regression() -> Result<()> {
    let input = data(4242, 40_000, 2, 4000);
    for (name, masks) in shapes().into_iter().take(2) {
        for mib in [1, 2, 4, 8] {
            let c = Ctx {
                batch: 8192,
                skip: false,
                memory: Some(mib << 20),
                ..BASE
            };
            same_pool_stock(&input, masks.clone(), AggregateMode::Partial, c).await?;
            let (stock, chained, admitted) =
                run(&input, 3, masks.clone(), AggregateMode::Partial, c).await;
            assert!(admitted);
            assert_eq!(
                stock?,
                chained
                    .unwrap_or_else(|e| panic!("{name} {mib} MiB default streams: {e}"))
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_finite_single_cube_and_disk_off() -> Result<()> {
    for (mode, parts) in [
        (AggregateMode::Partial, 2),
        (AggregateMode::Single, 1),
        (AggregateMode::SinglePartitioned, 2),
    ] {
        let input = data(4242, 6000, parts, 500);
        for (name, masks) in shapes().into_iter().take(2) {
            for (mib, disk) in [(1, true), (8, true), (16, true), (1, false), (16, false)]
            {
                let c = Ctx {
                    batch: 64,
                    skip: false,
                    memory: Some(mib << 20),
                    disk,
                    groups: 0,
                };
                let stock_fits = same_pool_stock(&input, masks.clone(), mode, c)
                    .await
                    .is_ok();
                let (stock, actual, admitted) =
                    run(&input, 3, masks.clone(), mode, c).await;
                assert_eq!(admitted, disk);
                match actual {
                    Ok(actual) => assert_eq!(stock?, actual, "{name} {mode:?} {c:?}"),
                    Err(e) => {
                        assert!(
                            !stock_fits,
                            "stock completes but CG fails: {name} {mode:?} {c:?}: {e}"
                        );
                        assert!(
                            matches!(
                                e.find_root(),
                                DataFusionError::ResourcesExhausted(_)
                            ),
                            "{e}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_high_cardinality_single() -> Result<()> {
    // Keep this high-cardinality differential in memory; the separate
    // finite matrix exercises native pressure emission, spilling and cleanup.
    let input = data(4242, 40_000, 1, 4000);
    for (_, masks) in shapes().into_iter().take(2) {
        let c = Ctx {
            batch: 8192,
            skip: false,
            memory: Some(64 << 20),
            ..BASE
        };
        same_pool_stock(&input, masks.clone(), AggregateMode::Single, c).await?;
        let (stock, chained, admitted) =
            run(&input, 3, masks.clone(), AggregateMode::Single, c).await;
        assert!(admitted);
        assert_eq!(stock?, chained?);
    }
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_nan_extrema_fallback_and_sum_avg() -> Result<()> {
    let mut input = data(903, 16, 1, 16);
    let values = vec![
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(0.),
        Some(f64::INFINITY),
        Some(-2.25),
        Some(0.),
        Some(f64::INFINITY),
        Some(-2.25),
        Some(-2.25),
        Some(0.),
        None,
        Some(-2.25),
        None,
        Some(f64::NAN),
        Some(-0.),
        Some(1.5),
    ];
    let mut columns = input[0][0].columns().to_vec();
    columns[9] = Arc::new(Float64Array::from(values));
    input[0][0] = RecordBatch::try_new(schema(), columns)?;
    input[0].truncate(1);
    let original = build(&input, 3, shapes().remove(0).1, AggregateMode::Single)?;
    for fun in [min_udaf(), max_udaf()] {
        let expression = Arc::new(
            AggregateExprBuilder::new(fun, vec![col("x", &schema())?])
                .schema(schema())
                .alias("nan_extreme")
                .build()?,
        );
        let agg = AggregateExec::try_new(
            AggregateMode::Single,
            Arc::clone(original.group_by()),
            vec![expression],
            vec![None],
            Arc::clone(&original.input),
            schema(),
        )?;
        let ctx = context(BASE);
        assert!(chained_grouping_sets::execute(&agg, 0, Arc::clone(&ctx))?.is_none());
        let stock = stock_stream(&agg, &ctx, 0)?.try_collect::<Vec<_>>().await?;
        let actual = collect(agg.execute(0, Arc::clone(&ctx))?).await?;
        assert_eq!(
            batches_to_sort_string(&stock),
            batches_to_sort_string(&actual)
        );
    }
    // SUM/AVG propagate NaN; finite values are exact multiples of 1/8. Signed
    // infinities and NaN survive native sum/count state merging.
    for mode in [
        AggregateMode::Partial,
        AggregateMode::Single,
        AggregateMode::SinglePartitioned,
    ] {
        let (stock, chained, admitted) =
            run(&input, 3, shapes().remove(1).1, mode, BASE).await;
        assert!(admitted);
        assert_eq!(stock?, chained?);
    }
    Ok(())
}

#[test]
fn chained_grouping_sets_rejects_order_volatile_limits_and_sorted_input() -> Result<()> {
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr, ScalarFunctionExpr};
    let input = data(12, 0, 1, 1);
    let original = build(&input, 0, shapes().remove(0).1, AggregateMode::Single)?;
    let ctx = context(BASE);
    let reject = |label: &str, agg: &AggregateExec| -> Result<()> {
        assert!(
            chained_grouping_sets::execute(agg, 0, Arc::clone(&ctx))?.is_none(),
            "{label}"
        );
        Ok(())
    };
    // 56 validates specialization hints: this grouping-set aggregate cannot
    // carry a DISTINCT soft limit. The old unchecked limit fixture is obsolete.
    assert!(
        original
            .clone()
            .restore_limit_options(Some(LimitOptions::new(3)))
            .is_none()
    );
    let rebuild = |expressions: Vec<Arc<AggregateFunctionExpr>>,
                   filters: Vec<Option<Arc<dyn PhysicalExpr>>>| {
        AggregateExec::try_new(
            original.mode,
            Arc::clone(original.group_by()),
            expressions,
            filters,
            Arc::clone(&original.input),
            schema(),
        )
    };
    let ordered = Arc::new(
        AggregateExprBuilder::new(sum_udaf(), vec![col("v", &schema())?])
            .schema(schema())
            .alias("ordered_sum")
            .order_by(vec![PhysicalSortExpr::new_default(col("v", &schema())?)])
            .build()?,
    );
    // Upstream normalizes away ORDER BY for insensitive SUM. This is safe
    // to chain; ordered, sensitive aggregates still take the native fallback.
    assert!(ordered.order_bys().is_empty());
    let ordered_agg = rebuild(vec![ordered], vec![None])?;
    assert!(chained_grouping_sets::execute(&ordered_agg, 0, Arc::clone(&ctx))?.is_some());
    let sensitive = Arc::new(
        AggregateExprBuilder::new(
            datafusion_functions_aggregate::first_last::first_value_udaf(),
            vec![col("v", &schema())?],
        )
        .schema(schema())
        .alias("ordered_first")
        .order_by(vec![PhysicalSortExpr::new_default(col("v", &schema())?)])
        .build()?,
    );
    assert!(!sensitive.order_bys().is_empty());
    reject("ordered_sensitive", &rebuild(vec![sensitive], vec![None])?)?;
    let random: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::try_new(
        datafusion_functions::math::random(),
        vec![],
        &schema(),
        Arc::new(Default::default()),
    )?);
    let mut group = original.group_by().as_ref().clone();
    group.expr[0].0 = Arc::clone(&random);
    group.null_expr[0].0 = lit(ScalarValue::Float64(None));
    let volatile_group = AggregateExec::try_new(
        original.mode,
        group,
        original.aggr_expr().to_vec(),
        original.filter_expr().to_vec(),
        Arc::clone(&original.input),
        schema(),
    )?;
    reject("volatile_group", &volatile_group)?;
    let mut filters = original.filter_expr().to_vec();
    filters[0] = Some(binary(
        Arc::clone(&random),
        datafusion_expr::Operator::Gt,
        lit(0.5f64),
        &schema(),
    )?);
    let volatile_filter = rebuild(original.aggr_expr().to_vec(), filters)?;
    reject("volatile_filter", &volatile_filter)?;
    let volatile_argument = rebuild(
        vec![Arc::new(
            AggregateExprBuilder::new(sum_udaf(), vec![random])
                .schema(schema())
                .alias("random_sum")
                .build()?,
        )],
        vec![None],
    )?;
    reject("volatile_argument", &volatile_argument)?;
    // 56 deliberately disables clustering for every grouping-set aggregate,
    // even when the input is sorted on a key retained by all requested sets.
    // CG is eligible here; ordinary ordered grouping still uses the native path.
    let mut agg = build(
        &input,
        0,
        vec![vec![false; 3], vec![false, false, true]],
        AggregateMode::Single,
    )?;
    let sorted = TestMemoryExec::try_new(&input, schema(), None)?
        .try_with_sort_information(vec![
            LexOrdering::new(vec![PhysicalSortExpr::new_default(col("s", &schema())?)])
                .unwrap(),
        ])?;
    agg = AggregateExec::try_new(
        agg.mode,
        Arc::clone(agg.group_by()),
        agg.aggr_expr().to_vec(),
        agg.filter_expr().to_vec(),
        Arc::new(sorted),
        schema(),
    )?;
    assert_eq!(agg.group_clustering_mode, GroupClusteringMode::None);
    assert!(chained_grouping_sets::execute(&agg, 0, Arc::clone(&ctx))?.is_some());
    let ordinary = AggregateExec::try_new(
        agg.mode,
        PhysicalGroupBy::new_single(agg.group_by().expr.clone()),
        agg.aggr_expr().to_vec(),
        agg.filter_expr().to_vec(),
        Arc::clone(&agg.input),
        schema(),
    )?;
    assert_ne!(ordinary.group_clustering_mode, GroupClusteringMode::None);
    reject("ordinary_ordered", &ordinary)?;
    Ok(())
}

/// A finite custom pool that uses the default Unknown limit report.
#[derive(Debug)]
struct UnknownLimitPool(datafusion_execution::memory_pool::GreedyMemoryPool);

impl std::fmt::Display for UnknownLimitPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown-limit pool")
    }
}

impl datafusion_execution::memory_pool::MemoryPool for UnknownLimitPool {
    fn name(&self) -> &str {
        "unknown-limit pool"
    }
    fn grow(&self, r: &datafusion_execution::memory_pool::MemoryReservation, n: usize) {
        self.0.grow(r, n);
    }
    fn shrink(&self, r: &datafusion_execution::memory_pool::MemoryReservation, n: usize) {
        self.0.shrink(r, n);
    }
    fn try_grow(
        &self,
        r: &datafusion_execution::memory_pool::MemoryReservation,
        n: usize,
    ) -> Result<()> {
        self.0.try_grow(r, n)
    }
    fn reserved(&self) -> usize {
        self.0.reserved()
    }
}

#[tokio::test]
async fn chained_grouping_sets_unknown_limit_without_disk_uses_native_partial()
-> Result<()> {
    use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryLimit, MemoryPool};
    let input = data(43, 12_000, 1, 256);
    let masks = shapes().remove(0).1;
    let pool: Arc<dyn MemoryPool> =
        Arc::new(UnknownLimitPool(GreedyMemoryPool::new(128 * 1024)));
    assert!(matches!(pool.memory_limit(), MemoryLimit::Unknown));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .with_disk_manager_builder(
            datafusion_execution::disk_manager::DiskManagerBuilder::default()
                .with_mode(datafusion_execution::disk_manager::DiskManagerMode::Disabled),
        )
        .build_arc()?;
    let ctx = Arc::new(
        TaskContext::default()
            .with_runtime(runtime)
            .with_session_config(SessionConfig::new().with_batch_size(64).set_usize(
                "datafusion.execution.skip_partial_aggregation_probe_rows_threshold",
                usize::MAX,
            )),
    );
    let native = build(&input, 3, masks.clone(), AggregateMode::Partial)?;
    let expected = collect(stock_stream(&native, &ctx, 0)?).await?;
    assert!(
        native
            .metrics()
            .unwrap()
            .sum_by_name("early_emit_count")
            .unwrap()
            .as_usize()
            > 0
    );
    assert_eq!(pool.reserved(), 0);
    let actual = build(&input, 3, masks, AggregateMode::Partial)?;
    assert!(chained_grouping_sets::execute(&actual, 0, Arc::clone(&ctx))?.is_none());
    let output = collect(actual.execute(0, Arc::clone(&ctx))?).await?;
    let unbounded = context(BASE);
    assert_eq!(
        batches_to_sort_string(&finish(&actual, output, &unbounded).await?),
        batches_to_sort_string(&finish(&native, expected, &unbounded).await?)
    );
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_root_disables_skip_probe() -> Result<()> {
    let input = data(91, 4096, 1, 256);
    let masks = shapes().remove(0).1;
    let ctx = context(Ctx { skip: true, ..BASE });
    // A grouping-set root through the native path never enables the probe.
    let root = build(&input, 0, vec![vec![false; 3]], AggregateMode::Partial)?;
    let root_rows: usize = collect(stock_stream(&root, &ctx, 0)?)
        .await?
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert!(root_rows < 4096 / 4, "fixture must reduce strongly");
    for groups in [0, 4] {
        let ctx = context(Ctx {
            groups,
            skip: true,
            ..BASE
        });
        let agg = build(&input, 0, masks.clone(), AggregateMode::Partial)?;
        let _output = collect(agg.execute(0, Arc::clone(&ctx))?).await?;
        let retained = agg
            .metrics()
            .unwrap()
            .sum_by_name("chained_root_state_rows")
            .unwrap()
            .as_usize();
        if groups == 0 {
            assert_eq!(
                retained, root_rows,
                "the CG root must keep native grouping-set reduction"
            );
        } else {
            assert!(
                retained > root_rows && retained < 4096,
                "G must flush early without skip conversion: {retained}, native root {root_rows}"
            );
        }
        assert_eq!(ctx.memory_pool().reserved(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_with_bounded_partial_matches_native() -> Result<()> {
    let input = data(91, 6000, 2, 256);
    for (name, masks) in shapes().into_iter().take(2) {
        for mode in [AggregateMode::Partial, AggregateMode::SinglePartitioned] {
            let (stock, chained, admitted) = run(
                &input,
                0,
                masks.clone(),
                mode,
                Ctx {
                    groups: 4,
                    skip: true,
                    ..BASE
                },
            )
            .await;
            assert!(admitted, "{name} {mode:?}");
            assert_eq!(stock?, chained?, "{name} {mode:?}");
        }
    }
    Ok(())
}

#[test]
fn chained_grouping_sets_incomparable_roots_do_not_recurse() -> Result<()> {
    let input = data(21, 0, 1, 1);
    let mut masks = vec![vec![false, true, true], vec![true, false, true]];
    let ctx = context(BASE);
    let agg = build(&input, 0, masks.clone(), AggregateMode::Partial)?;
    assert!(chained_grouping_sets::execute(&agg, 0, Arc::clone(&ctx))?.is_none());
    // A duplicate creates actual reuse; all other admission conditions stay the same.
    masks.push(masks[0].clone());
    let duplicate = build(&input, 0, masks, AggregateMode::Partial)?;
    assert!(chained_grouping_sets::execute(&duplicate, 0, ctx)?.is_some());
    Ok(())
}
