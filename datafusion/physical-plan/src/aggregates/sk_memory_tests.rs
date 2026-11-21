// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, Int32Array, Int64Array};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Field, Int32Type, Int64Type, Schema,
};
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_functions_aggregate::count::count_udaf;
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_expr::expressions::Literal;

use crate::ExecutionPlan;
use crate::common::collect;
use crate::expressions::col;
use crate::test::TestMemoryExec;

use super::group_values::new_group_values;
use super::group_values::{GroupValues, GroupValuesPrimitive};
use super::order::{GroupClustering, GroupClusteringMode};
use super::{AggregateExec, AggregateMode, PhysicalGroupBy, StreamType};

fn count_groups(keys: Vec<ArrayRef>, data_type: DataType) -> Result<AggregateExec> {
    let schema = Arc::new(Schema::new(vec![Field::new("k", data_type, true)]));
    let batches = keys
        .into_iter()
        .map(|key| RecordBatch::try_new(Arc::clone(&schema), vec![key]))
        .collect::<Result<Vec<_>, _>>()?;
    let input = TestMemoryExec::try_new_exec(&[batches], Arc::clone(&schema), None)?;
    let count = AggregateExprBuilder::new(
        count_udaf(),
        vec![Arc::new(Literal::new(ScalarValue::Int64(Some(1))))],
    )
    .schema(Arc::clone(&schema))
    .alias("count")
    .build()?;
    AggregateExec::try_new(
        AggregateMode::Single,
        PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".to_string())]),
        vec![Arc::new(count)],
        vec![None],
        input,
        schema,
    )
}

fn finite_pool(bytes: usize, spill: bool, migrated: bool) -> Result<Arc<TaskContext>> {
    let mut runtime = RuntimeEnvBuilder::new().with_memory_limit(bytes, 1.0);
    if !spill {
        runtime = runtime.with_disk_manager_builder(
            DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
        );
    }
    Ok(Arc::new(
        TaskContext::default()
            .with_runtime(runtime.build_arc()?)
            .with_session_config(
                SessionConfig::new().with_batch_size(8192).set_bool(
                    "datafusion.execution.enable_migration_aggregate",
                    migrated,
                ),
            ),
    ))
}

fn result_counts<T: ArrowPrimitiveType>(
    batches: &[RecordBatch],
) -> BTreeMap<Option<T::Native>, i64>
where
    T::Native: Ord,
{
    let mut counts = BTreeMap::new();
    for batch in batches {
        let keys = batch.column(0).as_primitive::<T>();
        let values = batch.column(1).as_primitive::<Int64Type>();
        for (key, count) in keys.iter().zip(values.values().iter().copied()) {
            assert!(counts.insert(key, count).is_none(), "duplicate output key");
        }
    }
    counts
}

#[tokio::test]
async fn sparse_integer_groups_fit_128_kib_without_spill() -> Result<()> {
    let keys = Int32Array::from_iter_values([0, 1 << 20].into_iter().cycle().take(8192));
    let aggregate = count_groups(vec![Arc::new(keys)], DataType::Int32)?;
    let context = finite_pool(128 * 1024, false, true)?;
    let stream = aggregate.execute_typed(0, &context)?;
    assert!(matches!(stream, StreamType::SingleHash(_)));
    let result = collect(stream.into()).await?;
    assert_eq!(
        result_counts::<Int32Type>(&result),
        BTreeMap::from([(Some(0), 4096), (Some(1 << 20), 4096)]),
    );
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}

fn widening_input() -> Vec<Int64Array> {
    let mut batches = (0..4)
        .map(|batch| {
            Int64Array::from_iter(
                (batch * 8191..(batch + 1) * 8191)
                    .map(Some)
                    .chain(std::iter::once(None)),
            )
        })
        .collect::<Vec<_>>();
    batches.push(Int64Array::from(vec![
        None,
        Some(7),
        Some(i64::MAX),
        Some(8201),
        Some(7),
        None,
    ]));
    batches
}

#[tokio::test]
async fn late_integer_fallback_requires_a_migration_grant() -> Result<()> {
    let keys = widening_input()
        .into_iter()
        .map(|array| Arc::new(array) as ArrayRef)
        .collect();
    let aggregate = count_groups(keys, DataType::Int64)?;
    let context = finite_pool(2 * 1024 * 1024, false, true)?;
    let stream = aggregate.execute_typed(0, &context)?;
    assert!(matches!(stream, StreamType::SingleHash(_)));
    let error = collect(stream.into())
        .await
        .err()
        .expect("migration peak should exceed the available pool");
    assert!(matches!(
        error.find_root(),
        DataFusionError::ResourcesExhausted(_)
    ));
    assert!(
        error.to_string().contains("packed single-key migration"),
        "{error}"
    );
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}

#[rstest::rstest]
#[case::single_hash(true, 2 * 1024 * 1024)]
#[case::legacy_hash(false, 3 * 1024 * 1024)]
#[tokio::test]
async fn late_integer_fallback_spills_and_merges_repeated_keys_and_null(
    #[case] migrated: bool,
    #[case] bytes: usize,
) -> Result<()> {
    let input = widening_input();
    let mut expected = BTreeMap::new();
    for key in input.iter().flat_map(|array| array.iter()) {
        *expected.entry(key).or_insert(0) += 1;
    }
    let keys = input
        .into_iter()
        .map(|array| Arc::new(array) as ArrayRef)
        .collect();
    let aggregate = count_groups(keys, DataType::Int64)?;
    let context = finite_pool(bytes, true, migrated)?;
    let stream = aggregate.execute_typed(0, &context)?;
    assert!(match stream {
        StreamType::SingleHash(_) => migrated,
        StreamType::GroupedHash(_) => !migrated,
        _ => false,
    });
    let result = collect(stream.into()).await?;
    assert_eq!(result_counts::<Int64Type>(&result), expected);
    assert!(aggregate.metrics().unwrap().spill_count().unwrap() >= 2);
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}

#[test]
fn clustered_primitive_keeps_published_initial_capacity() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    for mode in [
        GroupClusteringMode::Full,
        GroupClusteringMode::Partial(vec![0]),
    ] {
        let grouping = GroupClustering::try_new(&mode)?;
        let values = new_group_values(Arc::clone(&schema), &grouping)?;
        assert_eq!(
            values.size(),
            GroupValuesPrimitive::<Int32Type>::new(DataType::Int32).size()
        );
    }
    Ok(())
}
