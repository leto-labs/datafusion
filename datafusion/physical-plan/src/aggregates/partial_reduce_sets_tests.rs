// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow family PRS: a partial-reduce aggregate over grouping sets expands
//! each state row into every set and merges it, as the raw-input stage
//! expands each raw row.

use super::*;
use crate::coalesce_partitions::CoalescePartitionsExec;
use crate::common::collect;
use crate::projection::{ProjectionExec, ProjectionExpr};
use crate::test::TestMemoryExec;
use arrow::array::{Decimal128Array, Int32Array, Int64Array};
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
use datafusion_physical_expr::expressions::{Column, col, lit};

type Plan = Arc<dyn ExecutionPlan>;

fn context(memory: Option<usize>) -> Arc<TaskContext> {
    let config = SessionConfig::new().with_batch_size(5);
    assert!(config.options().execution.enable_migration_aggregate);
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

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k0", DataType::Int32, true),
        Field::new("k1", DataType::Int32, true),
        Field::new("v", DataType::Int64, true),
        Field::new("d", DataType::Decimal128(7, 2), true),
    ]))
}

/// Three partitions of rows with NULL keys and NULL values.
fn input(rows: usize) -> Result<Plan> {
    let schema = schema();
    let mut partitions = vec![vec![]; 3];
    for (partition, batches) in partitions.iter_mut().enumerate() {
        let ids = (0..rows).filter(|i| i % 3 == partition).collect::<Vec<_>>();
        for chunk in ids.chunks(4) {
            let k0 = chunk.iter().map(|i| (i % 5 != 4).then_some((i % 3) as i32));
            let k1 = chunk.iter().map(|i| (i % 7 != 6).then_some((i % 4) as i32));
            let v = chunk
                .iter()
                .map(|i| (i % 11 != 10).then_some(*i as i64 - 9));
            let d = chunk.iter().map(|i| Some(*i as i128 * 13 - 50));
            batches.push(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from_iter(k0)),
                    Arc::new(Int32Array::from_iter(k1)),
                    Arc::new(Int64Array::from_iter(v)),
                    Arc::new(
                        Decimal128Array::from_iter(d).with_precision_and_scale(7, 2)?,
                    ),
                ],
            )?);
        }
    }
    Ok(TestMemoryExec::try_new_exec(&partitions, schema, None)?)
}

fn functions() -> Result<Vec<Arc<AggregateFunctionExpr>>> {
    let schema = schema();
    let v = col("v", &schema)?;
    let d = col("d", &schema)?;
    [
        (sum_udaf(), vec![Arc::clone(&v)], "sum_v"),
        (count_udaf(), vec![Arc::clone(&v)], "count_v"),
        (count_udaf(), vec![lit(1i64)], "count_all"),
        (min_udaf(), vec![Arc::clone(&v)], "min_v"),
        (max_udaf(), vec![v], "max_v"),
        (sum_udaf(), vec![Arc::clone(&d)], "sum_d"),
        (avg_udaf(), vec![d], "avg_d"),
    ]
    .into_iter()
    .map(|(fun, args, alias)| {
        AggregateExprBuilder::new(fun, args)
            .schema(Arc::clone(&schema))
            .alias(alias)
            .build()
            .map(Arc::new)
    })
    .collect()
}

/// The group by over `k0, k1` of `plan`'s columns 0 and 1 with `masks`.
fn sets(plan: &Plan, masks: Vec<Vec<bool>>) -> PhysicalGroupBy {
    let schema = plan.schema();
    PhysicalGroupBy::new(
        (0..2)
            .map(|i| {
                let name = schema.field(i).name().clone();
                (Arc::new(Column::new(&name, i)) as _, name)
            })
            .collect(),
        (0..2)
            .map(|i| (lit(ScalarValue::Int32(None)), format!("k{i}")))
            .collect(),
        masks,
        true,
    )
}

fn rollup() -> Vec<Vec<bool>> {
    vec![vec![true, true], vec![false, true], vec![false, false]]
}

fn cube() -> Vec<Vec<bool>> {
    vec![
        vec![true, true],
        vec![false, true],
        vec![true, false],
        vec![false, false],
    ]
}

/// The final rows of `partial` (output of a state-producing stage over
/// `masks`), after merging every partition.
async fn finish(
    partial: Plan,
    masks: Vec<Vec<bool>>,
    ctx: &Arc<TaskContext>,
) -> Result<String> {
    let reference = AggregateExec::try_new(
        AggregateMode::Partial,
        sets(&input(0)?, masks),
        functions()?,
        vec![None; 7],
        input(0)?,
        schema(),
    )?;
    let final_agg = AggregateExec::try_new(
        AggregateMode::Final,
        reference.group_by().as_final(),
        functions()?,
        vec![None; 7],
        Arc::new(CoalescePartitionsExec::new(partial)),
        schema(),
    )?;
    let batches = collect(final_agg.execute(0, Arc::clone(ctx))?).await?;
    Ok(batches_to_sort_string(&batches))
}

/// Stock plan: the raw-input stage expands every raw row.
async fn stock(
    rows: usize,
    masks: Vec<Vec<bool>>,
    ctx: &Arc<TaskContext>,
) -> Result<String> {
    let raw = input(rows)?;
    let partial = AggregateExec::try_new(
        AggregateMode::Partial,
        sets(&raw, masks.clone()),
        functions()?,
        vec![None; 7],
        raw,
        schema(),
    )?;
    finish(Arc::new(partial), masks, ctx).await
}

/// The state-consuming plan: ordinary partial states of `(k0, k1)`, a pad
/// column where the grouping id of a state input stands, and a
/// partial-reduce aggregate that expands each state row into every set.
fn reduced_plan(rows: usize, masks: Vec<Vec<bool>>) -> Result<Arc<AggregateExec>> {
    let raw = input(rows)?;
    let single = PhysicalGroupBy::new_single(
        (0..2)
            .map(|i| {
                (
                    col(&format!("k{i}"), &raw.schema()).unwrap(),
                    format!("k{i}"),
                )
            })
            .collect(),
    );
    let states: Plan = Arc::new(AggregateExec::try_new(
        AggregateMode::Partial,
        single,
        functions()?,
        vec![None; 7],
        raw,
        schema(),
    )?);
    let state_schema = states.schema();
    let mut items = vec![];
    for i in 0..state_schema.fields().len() {
        if i == 2 {
            items.push(ProjectionExpr::new(lit(0_u32), "__pad".to_owned()));
        }
        let name = state_schema.field(i).name().clone();
        items.push(ProjectionExpr::new(Arc::new(Column::new(&name, i)), name));
    }
    let padded: Plan = Arc::new(ProjectionExec::try_new(items, states)?);
    let group = sets(&padded, masks);
    Ok(Arc::new(AggregateExec::try_new(
        AggregateMode::PartialReduce,
        group,
        functions()?,
        vec![None; 7],
        padded,
        schema(),
    )?))
}

#[tokio::test]
async fn partial_reduce_over_grouping_sets_merges_each_state_into_every_set() -> Result<()>
{
    let ctx = context(None);
    for masks in [rollup(), cube()] {
        for rows in [1, 2, 40, 403] {
            let expected = stock(rows, masks.clone(), &ctx).await?;
            let reduced = reduced_plan(rows, masks.clone())?;
            // Same state schema as the raw-input stage over the same sets.
            let raw = input(rows)?;
            let reference = AggregateExec::try_new(
                AggregateMode::Partial,
                sets(&raw, masks.clone()),
                functions()?,
                vec![None; 7],
                raw,
                schema(),
            )?;
            assert_eq!(reduced.schema(), reference.schema());
            let actual = finish(reduced, masks.clone(), &ctx).await?;
            assert_eq!(actual, expected, "{masks:?} over {rows} rows");
        }
    }
    Ok(())
}

#[tokio::test]
async fn partial_reduce_over_grouping_sets_emits_no_row_for_an_empty_input() -> Result<()>
{
    // Only the raw-input stages synthesize the empty grouping set's row; a
    // state input without rows has no groups, so the caller owns that row.
    let ctx = context(None);
    let reduced = reduced_plan(0, rollup())?;
    let mut rows = 0;
    for partition in 0..reduced.properties().output_partitioning().partition_count() {
        rows += collect(reduced.execute(partition, Arc::clone(&ctx))?)
            .await?
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>();
    }
    assert_eq!(rows, 0);
    Ok(())
}

#[tokio::test]
async fn partial_reduce_over_grouping_sets_emits_mergeable_states_under_memory_pressure()
-> Result<()> {
    // Early emission may repeat a group; the final stage merges the repeats.
    let ctx = context(Some(64 * 1024));
    let expected = stock(403, cube(), &context(None)).await?;
    let actual = finish(reduced_plan(403, cube())?, cube(), &ctx).await?;
    assert_eq!(actual, expected);
    Ok(())
}
