// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;
use std::sync::Arc;

use arrow::array::{Array, AsArray, Int64Array};
use arrow::compute::filter;
use arrow::record_batch::RecordBatch;
use datafusion_common::DataFusionError;
use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use datafusion_expr::window_state::PartitionBatchState;
use datafusion_expr::{WindowFrame, WindowFrameBound, WindowFrameUnits};
use datafusion_physical_expr::PhysicalSortExpr;
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::window::{
    PartitionBatches, PartitionWindowAggStates, SlidingAggregateWindowExpr,
    WindowEvalContext, WindowExpr,
};

fn window(
    fun: Arc<datafusion_expr::AggregateUDF>,
    batch: &RecordBatch,
    units: WindowFrameUnits,
    start: WindowFrameBound,
    end: WindowFrameBound,
    filtered: bool,
) -> Result<SlidingAggregateWindowExpr> {
    let aggregate = AggregateExprBuilder::new(fun, vec![Arc::new(Column::new("v", 1))])
        .schema(batch.schema())
        .alias("rolling_bool")
        .build()?;
    Ok(SlidingAggregateWindowExpr::new(
        Arc::new(aggregate),
        &[],
        &[PhysicalSortExpr {
            expr: Arc::new(Column::new("k", 0)),
            options: Default::default(),
        }],
        Arc::new(WindowFrame::new_bounds(units, start, end)),
        filtered.then(|| Arc::new(Column::new("f", 2)) as _),
    ))
}

fn fixture() -> RecordBatch {
    RecordBatch::try_from_iter([
        (
            "k",
            Arc::new(Int64Array::from_iter_values(
                (0..1_031).map(|i| 10 * (i / 3)),
            )) as ArrayRef,
        ),
        (
            "v",
            Arc::new(BooleanArray::from_iter((0..1_031).map(|i| match i % 13 {
                0..=7 => None,
                8..=10 => Some(i % 31 == 8),
                _ => Some(true),
            }))) as ArrayRef,
        ),
        (
            "f",
            Arc::new(BooleanArray::from_iter((0..1_031).map(|i| match i % 7 {
                0 => None,
                1..=3 => Some(false),
                _ => Some(true),
            }))) as ArrayRef,
        ),
    ])
    .unwrap()
}

#[test]
fn sliding_bool_matches_fresh_frames() -> Result<()> {
    use WindowFrameBound::{CurrentRow, Following, Preceding};
    use WindowFrameUnits::{Groups, Range, Rows};

    // A nonzero Arrow offset exercises unaligned bitmap slices.
    let batch = fixture().slice(3, 1_026);
    let n = batch.num_rows();
    let keys = batch
        .column(0)
        .as_primitive::<arrow::datatypes::Int64Type>();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024));
    let reservation = MemoryConsumer::new("window-test").register(&pool);

    // Frame membership is calculated independently of the window evaluator.
    for fun in [bool_and_udaf(), bool_or_udaf()] {
        for filtered in [false, true] {
            for (units, before, after) in [
                (Rows, 0_i64, 0_i64),
                (Rows, 3, 7),
                (Rows, 999, 0),
                (Rows, -2, 5),
                (Range, 13, 4),
                (Range, -1, 6),
                (Groups, 1, 2),
            ] {
                let start = if before >= 0 {
                    Preceding(if units == Range {
                        before.into()
                    } else {
                        (before as u64).into()
                    })
                } else {
                    Following(if units == Range {
                        (-before).into()
                    } else {
                        ((-before) as u64).into()
                    })
                };
                let end = if after == 0 {
                    CurrentRow
                } else {
                    Following(if units == Range {
                        after.into()
                    } else {
                        (after as u64).into()
                    })
                };
                let expr = window(Arc::clone(&fun), &batch, units, start, end, filtered)?;
                let actual = expr.evaluate_with_reservation(&batch, &reservation)?;
                assert_eq!(pool.reserved(), 0);
                for (i, actual) in actual.as_boolean().iter().enumerate() {
                    let (start, end) = match units {
                        Rows => (
                            (i as i64 - before).clamp(0, n as i64) as usize,
                            (i + after as usize + 1).min(n),
                        ),
                        Range => {
                            let key = keys.value(i);
                            (
                                keys.values().partition_point(|&x| x < key - before),
                                keys.values().partition_point(|&x| x <= key + after),
                            )
                        }
                        Groups => (
                            (i / 3).saturating_sub(before as usize) * 3,
                            ((i / 3 + after as usize + 1) * 3).min(n),
                        ),
                    };
                    let mut values =
                        batch.column(1).slice(start, end.saturating_sub(start));
                    if filtered {
                        let mask = batch
                            .column(2)
                            .as_boolean()
                            .slice(start, end.saturating_sub(start));
                        values = filter(values.as_ref(), &mask)?;
                    }
                    let mut fresh = expr.get_aggregate_expr().create_accumulator()?;
                    fresh.update_batch(&[values])?;
                    assert_eq!(
                        ScalarValue::Boolean(actual),
                        fresh.evaluate()?,
                        "{} {units:?} {before} {after} filter={filtered} row={i}",
                        fun.name()
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn sliding_bool_refuses_insufficient_reservation() -> Result<()> {
    let batch = fixture();
    let expr = window(
        bool_or_udaf(),
        &batch,
        WindowFrameUnits::Rows,
        WindowFrameBound::Preceding(999_u64.into()),
        WindowFrameBound::CurrentRow,
        false,
    )?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(0));
    let reservation = MemoryConsumer::new("window-test").register(&pool);
    let error = expr
        .evaluate_with_reservation(&batch, &reservation)
        .unwrap_err();
    assert!(
        matches!(error, DataFusionError::ResourcesExhausted(_)),
        "{error}"
    );
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn sliding_bool_retains_one_fixed_charge_across_batches() -> Result<()> {
    let batch = fixture();
    let expr = window(
        bool_or_udaf(),
        &batch,
        WindowFrameUnits::Rows,
        WindowFrameBound::Preceding(999_u64.into()),
        WindowFrameBound::CurrentRow,
        true,
    )?;
    let accumulator_size = expr
        .get_aggregate_expr()
        .create_sliding_accumulator()?
        .size();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(accumulator_size));
    let reservation = MemoryConsumer::new("window-test").register(&pool);
    let ctx = WindowEvalContext::default().with_frame_reservation(Some(&reservation));
    let first = PartitionBatchState::new_with_batch(batch.slice(0, 29));
    let mut batches = PartitionBatches::default();
    batches.insert(vec![], first);
    let mut states = PartitionWindowAggStates::default();
    expr.evaluate_stateful(&batches, &mut states, &ctx)?;
    assert_eq!(pool.reserved(), accumulator_size);
    let input = batches.get_index_mut(0).unwrap().1;
    input.extend(&batch.slice(29, batch.num_rows() - 29))?;
    input.is_end = true;
    expr.evaluate_stateful(&batches, &mut states, &ctx)?;
    assert_eq!(pool.reserved(), accumulator_size);
    let state = states.get_index_mut(0).unwrap().1;
    let expected = expr.evaluate(&batch)?;
    assert_eq!(&state.state.out_col, &expected);
    assert_eq!(
        state.aggregate_state()?,
        Some(vec![ScalarValue::Boolean(
            expected.as_boolean().iter().last().unwrap()
        )]),
    );
    drop(states);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn sliding_bool_distinct_recompute_is_refused() -> Result<()> {
    let batch = fixture();
    let distinct =
        AggregateExprBuilder::new(bool_or_udaf(), vec![Arc::new(Column::new("v", 1))])
            .schema(batch.schema())
            .alias("distinct_bool_or")
            .distinct()
            .build()?;
    assert!(distinct.create_accumulator().is_ok());
    let error = distinct.create_sliding_accumulator().unwrap_err();
    assert!(
        error
            .message()
            .contains("`retract_batch` is not implemented")
    );
    Ok(())
}
