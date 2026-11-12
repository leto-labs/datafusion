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

//! Stock grouping-set spill replay must retain the grouping ID as a key.

use super::*;
use crate::test::TestMemoryExec;
use arrow::array::{Int32Array, Int64Array};
use datafusion_common::test_util::batches_to_sort_string;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::memory_pool::FairSpillPool;
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_functions_aggregate::{average::avg_udaf, count::count_udaf};
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_expr::expressions::{binary, col};
use futures::TryStreamExt;

fn context(bounded: bool) -> Arc<TaskContext> {
    let mut config = SessionConfig::new().with_batch_size(64);
    config.options_mut().execution.sort_spill_reservation_bytes = 4096;
    let mut runtime = RuntimeEnvBuilder::new();
    if bounded {
        runtime = runtime.with_memory_pool(Arc::new(FairSpillPool::new(128 * 1024)));
    }
    Arc::new(
        TaskContext::default()
            .with_session_config(config)
            .with_runtime(runtime.build_arc().unwrap()),
    )
}

fn fixture(rollup: bool, mode: AggregateMode) -> Result<AggregateExec> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("v", DataType::Int64, true),
    ]));
    // Interleave repeated keys across batches and spill runs. Real NULL keys
    // collide with rolled-up NULLs unless the grouping ID remains a merge key.
    let batches = (0..128)
        .map(|batch| {
            let rows = (batch * 128..(batch + 1) * 128).collect::<Vec<_>>();
            let keys = rows.iter().map(|i| (i * 1543) % 4096).collect::<Vec<_>>();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from_iter(
                        keys.iter().map(|k| (k % 29 != 0).then_some(k % 1024)),
                    )),
                    Arc::new(Int32Array::from_iter(
                        keys.iter().map(|k| (k % 31 != 0).then_some(k / 1024)),
                    )),
                    Arc::new(Int64Array::from_iter(
                        rows.iter()
                            .map(|i| (i % 7 != 0).then_some(i64::from(i % 17) - 8)),
                    )),
                ],
            )
            .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let group = PhysicalGroupBy::new(
        vec![
            (col("a", &schema)?, "a".into()),
            (col("b", &schema)?, "b".into()),
        ],
        vec![
            (lit(ScalarValue::Int32(None)), "a".into()),
            (lit(ScalarValue::Int32(None)), "b".into()),
        ],
        if rollup {
            vec![vec![false, false], vec![false, true], vec![true, true]]
        } else {
            // Incomparable sets and a duplicate set also test the ordinal bits.
            vec![
                vec![false, true],
                vec![true, false],
                vec![false, true],
                vec![true, true],
            ]
        },
        true,
    );
    let value = col("v", &schema)?;
    let expressions = vec![
        Arc::new(
            AggregateExprBuilder::new(count_udaf(), vec![Arc::clone(&value)])
                .schema(Arc::clone(&schema))
                .alias("count_v")
                .build()?,
        ),
        // AVG has two state columns, unlike its final scalar output.
        Arc::new(
            AggregateExprBuilder::new(avg_udaf(), vec![Arc::clone(&value)])
                .schema(Arc::clone(&schema))
                .alias("avg_v")
                .build()?,
        ),
    ];
    let positive = binary(value, datafusion_expr::Operator::Gt, lit(0i64), &schema)?;
    AggregateExec::try_new(
        mode,
        group,
        expressions,
        vec![None, Some(positive)],
        TestMemoryExec::try_new_exec(&[batches], Arc::clone(&schema), None)?,
        schema,
    )
}

async fn stock(agg: &AggregateExec, ctx: &Arc<TaskContext>) -> Result<Vec<RecordBatch>> {
    // Bypass family CG and the migration dispatcher so this always exercises
    // the stock expanding stream, even for a chain such as ROLLUP(a, b).
    Box::pin(GroupedHashAggregateStream::new(agg, ctx, 0)?)
        .try_collect()
        .await
}

fn final_aggregate(
    partial: &AggregateExec,
    batches: Vec<RecordBatch>,
) -> Result<AggregateExec> {
    AggregateExec::try_new(
        AggregateMode::Final,
        partial.group_by.as_final(),
        partial.aggr_expr.to_vec(),
        vec![None; partial.aggr_expr.len()],
        TestMemoryExec::try_new_exec(&[batches], partial.schema(), None)?,
        Arc::clone(&partial.input_schema),
    )
}

fn spills(agg: &AggregateExec) -> usize {
    agg.metrics().unwrap().spill_count().unwrap()
}

async fn compare(rollup: bool, mode: AggregateMode) -> Result<()> {
    let unbounded = context(false);
    let bounded = context(true);
    let baseline = fixture(rollup, AggregateMode::Single)?;
    let expected = stock(&baseline, &unbounded).await?;
    assert_eq!(spills(&baseline), 0);
    let actual = if mode == AggregateMode::Final {
        // Final consumes partial states, including the already materialized ID.
        let partial = fixture(rollup, AggregateMode::Partial)?;
        let states = stock(&partial, &unbounded).await?;
        let final_agg = final_aggregate(&partial, states)?;
        let output = stock(&final_agg, &bounded).await?;
        assert!(
            spills(&final_agg) > 1,
            "Final must merge multiple spilled runs"
        );
        output
    } else {
        let agg = fixture(rollup, mode)?;
        let output = stock(&agg, &bounded).await?;
        if mode == AggregateMode::Partial {
            // Native Partial emits incomplete states early under pressure;
            // it never spills. Its Final consumer must still spill and merge.
            assert_eq!(spills(&agg), 0);
            let rows = output.iter().map(RecordBatch::num_rows).sum::<usize>();
            assert!(
                rows > expected.iter().map(RecordBatch::num_rows).sum::<usize>(),
                "Partial must emit duplicate, incomplete groups under pressure"
            );
            let final_agg = final_aggregate(&agg, output)?;
            let output = stock(&final_agg, &bounded).await?;
            assert!(
                spills(&final_agg) > 1,
                "Final must spill Partial's early output"
            );
            output
        } else {
            assert!(spills(&agg) > 1, "Single must merge multiple spilled runs");
            output
        }
    };
    assert_eq!(actual[0].schema(), expected[0].schema());
    assert_eq!(
        batches_to_sort_string(&actual),
        batches_to_sort_string(&expected)
    );
    assert_eq!(bounded.memory_pool().reserved(), 0, "leaked reservation");
    assert_eq!(unbounded.memory_pool().reserved(), 0, "leaked reservation");
    Ok(())
}

#[tokio::test]
async fn single_grouping_set_spill() -> Result<()> {
    compare(false, AggregateMode::Single).await
}

#[tokio::test]
async fn single_rollup_spill() -> Result<()> {
    compare(true, AggregateMode::Single).await
}

#[tokio::test]
async fn partial_grouping_set_spill() -> Result<()> {
    compare(false, AggregateMode::Partial).await?;
    compare(true, AggregateMode::Partial).await
}

#[tokio::test]
async fn final_grouping_set_spill() -> Result<()> {
    compare(false, AggregateMode::Final).await?;
    compare(true, AggregateMode::Final).await
}
