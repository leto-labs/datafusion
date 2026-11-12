// Modified by the Longbow authors; see third_party/README.md.
//
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
use datafusion_physical_expr::expressions::{binary, col};
use futures::TryStreamExt;
use rand::{Rng, SeedableRng, rngs::StdRng};

#[derive(Clone, Copy, Debug)]
struct Ctx {
    batch: usize,
    skip: bool,
    migration: bool,
    memory: Option<usize>,
    disk: bool,
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
    config.options_mut().execution.enable_migration_aggregate = c.migration;
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
        (avg_udaf(), vec![Arc::clone(&v)], "avg_v"),
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
        agg.group_by.as_final(),
        agg.aggr_expr.to_vec(),
        vec![None; agg.aggr_expr.len()],
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
                Box::pin(GroupedHashAggregateStream::new(
                    &expected, &ectx, partition,
                )?)
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
    migration: true,
    memory: None,
    disk: true,
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
                            migration: false,
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
            Box::pin(GroupedHashAggregateStream::new(&agg, &ctx, partition)?)
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
        for migration in [true, false] {
            for mib in [1, 2, 4, 8] {
                let c = Ctx {
                    batch: 8192,
                    skip: false,
                    migration,
                    memory: Some(mib << 20),
                    ..BASE
                };
                same_pool_stock(&input, masks.clone(), AggregateMode::Partial, c).await?;
                let (stock, chained, admitted) =
                    run(&input, 3, masks.clone(), AggregateMode::Partial, c).await;
                assert!(admitted);
                assert_eq!(
                    stock?,
                    chained.unwrap_or_else(|e| panic!(
                        "{name} {mib} MiB migration {migration}: {e}"
                    ))
                );
            }
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
            for migration in [true, false] {
                for (mib, disk) in
                    [(1, true), (8, true), (16, true), (1, false), (16, false)]
                {
                    let c = Ctx {
                        batch: 64,
                        skip: false,
                        migration,
                        memory: Some(mib << 20),
                        disk,
                        ..BASE
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
    }
    Ok(())
}

#[tokio::test]
async fn chained_grouping_sets_high_cardinality_single() -> Result<()> {
    // The stock Single grouping-set spill path can lose its hidden group ID
    // at 8 MiB. Keep this successful high-cardinality differential in memory;
    // the separate finite matrix checks tighter-pool outcomes and cleanup.
    let input = data(4242, 40_000, 1, 4000);
    for (_, masks) in shapes().into_iter().take(2) {
        for migration in [true, false] {
            let c = Ctx {
                batch: 8192,
                skip: false,
                migration,
                memory: Some(64 << 20),
                ..BASE
            };
            same_pool_stock(&input, masks.clone(), AggregateMode::Single, c).await?;
            let (stock, chained, admitted) =
                run(&input, 3, masks.clone(), AggregateMode::Single, c).await;
            assert!(admitted);
            assert_eq!(stock?, chained?);
        }
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
            original.group_by.clone(),
            vec![expression],
            vec![None],
            Arc::clone(&original.input),
            schema(),
        )?;
        let ctx = context(BASE);
        assert!(chained_grouping_sets::execute(&agg, 0, Arc::clone(&ctx))?.is_none());
        let stock = Box::pin(GroupedHashAggregateStream::new(&agg, &ctx, 0)?)
            .try_collect::<Vec<_>>()
            .await?;
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
    reject(
        "limit",
        &original.with_new_limit_options(Some(LimitOptions::new(3))),
    )?;
    let ordered = Arc::new(
        AggregateExprBuilder::new(sum_udaf(), vec![col("v", &schema())?])
            .schema(schema())
            .alias("ordered_sum")
            .order_by(vec![PhysicalSortExpr::new_default(col("v", &schema())?)])
            .build()?,
    );
    let mut ordered_agg = original.with_new_limit_options(None);
    ordered_agg.aggr_expr = vec![ordered].into();
    reject("ordered_agg", &ordered_agg)?;
    let random: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::try_new(
        datafusion_functions::math::random(),
        vec![],
        &schema(),
        Arc::new(Default::default()),
    )?);
    let mut group = original.group_by.as_ref().clone();
    group.expr[0].0 = Arc::clone(&random);
    group.null_expr[0].0 = lit(ScalarValue::Float64(None));
    let volatile_group = AggregateExec::try_new(
        original.mode,
        group,
        original.aggr_expr.to_vec(),
        original.filter_expr.to_vec(),
        Arc::clone(&original.input),
        schema(),
    )?;
    reject("volatile_group", &volatile_group)?;
    let mut volatile_filter = original.with_new_limit_options(None);
    let mut filters = original.filter_expr.to_vec();
    filters[0] = Some(binary(
        Arc::clone(&random),
        datafusion_expr::Operator::Gt,
        lit(0.5f64),
        &schema(),
    )?);
    volatile_filter.filter_expr = filters.into();
    reject("volatile_filter", &volatile_filter)?;
    let mut volatile_argument = original.with_new_limit_options(None);
    volatile_argument.aggr_expr = vec![Arc::new(
        AggregateExprBuilder::new(sum_udaf(), vec![random])
            .schema(schema())
            .alias("random_sum")
            .build()?,
    )]
    .into();
    reject("volatile_argument", &volatile_argument)?;
    // These requested sets retain s, so a genuine input ordering yields a
    // non-linear native grouping mode (ROLLUP's grand total would erase it).
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
        agg.group_by.clone(),
        agg.aggr_expr.to_vec(),
        agg.filter_expr.to_vec(),
        Arc::new(sorted),
        schema(),
    )?;
    assert_ne!(agg.input_order_mode, InputOrderMode::Linear);
    reject("agg", &agg)?;
    Ok(())
}
