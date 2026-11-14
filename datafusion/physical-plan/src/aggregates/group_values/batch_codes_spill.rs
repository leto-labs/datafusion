// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the Longbow authors.

//! Compare spill/replay of batch-local view dictionaries with scalar sums.

use crate::{
    ExecutionPlan,
    aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
    common::collect,
    test::TestMemoryExec,
};
use arrow::{
    array::{Array, AsArray, Int64Array, StringViewArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion_common::Result;
use datafusion_execution::{
    TaskContext, config::SessionConfig, memory_pool::FairSpillPool,
    runtime_env::RuntimeEnvBuilder,
};
use datafusion_functions_aggregate::sum::sum_udaf;
use datafusion_physical_expr::{aggregate::AggregateExprBuilder, expressions::col};
use std::{collections::HashMap, sync::Arc};

#[tokio::test]
async fn batch_code_grouping_spill_matches_scalar_sums() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8View, true),
        Field::new("v", DataType::Int64, true),
    ]));
    let mut expected = HashMap::<Option<String>, i64>::new();
    let mut batches = Vec::new();
    for batch in 0..128 {
        let keys: Vec<_> = (0..8192)
            .map(|row| {
                let id = (row * 17 + batch * 101) % 16384;
                // Each batch uses 128 dictionary entries; later batches change
                // both their bytes and their order. NULL remains one SQL group.
                (row % 13 != 0)
                    .then(|| format!("dictionary-key-{}", id % 128 + batch * 128))
            })
            .collect();
        let values: Vec<_> = (0..8192)
            .map(|row| (row % 7 != 0).then_some(row as i64 % 19 - 9))
            .collect();
        for (key, value) in keys.iter().zip(&values) {
            *expected.entry(key.clone()).or_default() += value.unwrap_or(0);
        }
        let dict = StringViewArray::from_iter_values(
            (0..128).map(|id| format!("dictionary-key-{}", id + batch * 128)),
        );
        let views = StringViewArray::try_new(
            (0..8192)
                .map(|row| dict.views()[(row * 17 + batch * 101) % 128])
                .collect::<Vec<_>>()
                .into(),
            dict.data_buffers().to_vec(),
            Some(arrow::buffer::NullBuffer::from(
                keys.iter().map(Option::is_some).collect::<Vec<_>>(),
            )),
        )?;
        batches.push(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(views), Arc::new(Int64Array::from(values))],
        )?);
    }
    let input = TestMemoryExec::try_new_exec(&[batches], Arc::clone(&schema), None)?;
    let aggregate = AggregateExec::try_new(
        AggregateMode::Single,
        PhysicalGroupBy::new(
            vec![(col("k", &schema)?, "k".into())],
            vec![],
            vec![vec![false]],
            false,
        ),
        vec![Arc::new(
            AggregateExprBuilder::new(sum_udaf(), vec![col("v", &schema)?])
                .schema(Arc::clone(&schema))
                .alias("sum")
                .build()?,
        )],
        vec![None],
        input,
        schema,
    )?;
    let mut config = SessionConfig::new().with_batch_size(8192);
    config.options_mut().execution.sort_spill_reservation_bytes = 4096;
    let context = Arc::new(
        TaskContext::default()
            .with_session_config(config)
            .with_runtime(
                RuntimeEnvBuilder::new()
                    .with_memory_pool(Arc::new(FairSpillPool::new(1024 * 1024)))
                    .build_arc()?,
            ),
    );
    let output = collect(aggregate.execute(0, context)?).await?;
    let mut actual = HashMap::new();
    for batch in &output {
        let keys = batch.column(0).as_string_view();
        let values = batch
            .column(1)
            .as_primitive::<arrow::datatypes::Int64Type>();
        for row in 0..batch.num_rows() {
            let key = keys.is_valid(row).then(|| keys.value(row).to_owned());
            assert!(
                actual.insert(key, values.value(row)).is_none(),
                "duplicate emitted group"
            );
        }
    }
    assert_eq!(actual, expected);
    assert!(
        aggregate.metrics().unwrap().spill_count().unwrap() > 0,
        "test must exercise spill/replay"
    );
    Ok(())
}
