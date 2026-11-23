// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::{size_of, size_of_val};
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    ArrowNumericType, ArrowPrimitiveType, BooleanArray, ListArray, PrimitiveArray,
    PrimitiveBuilder,
};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::{
    array::{Array, ArrayRef, AsArray},
    datatypes::{DataType, Field, FieldRef, Float16Type, Float32Type, Float64Type},
};

use num_traits::AsPrimitive;

use arrow::array::ArrowNativeTypeOp;
use arrow::compute::DecimalCast;
use arrow::datatypes::{
    Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type, DecimalType, i256,
};
use datafusion_common::hash_utils::RandomState;
use datafusion_common::internal_err;
use datafusion_common::types::{NativeType, logical_float64};
use datafusion_common::utils::memory::estimate_memory_size;
use datafusion_functions_aggregate_common::noop_accumulator::NoopAccumulator;

use crate::min_max::{max_udaf, min_udaf};
use datafusion_common::{
    Result, ScalarValue, exec_datafusion_err, internal_datafusion_err,
    utils::{SingleRowListArrayBuilder, take_function_args},
};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Coercion, Documentation, Expr, Signature,
    TypeSignature, TypeSignatureClass, Volatility,
};
use datafusion_expr::{EmitTo, GroupsAccumulator};
use datafusion_expr::{
    expr::{AggregateFunction, Sort},
    function::{AccumulatorArgs, AggregateFunctionSimplification, StateFieldsArgs},
    simplify::SimplifyContext,
};
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::accumulate::accumulate;
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::nulls::filtered_null_mask;
use datafusion_functions_aggregate_common::utils::Hashable;
use datafusion_macros::user_doc;
use datafusion_physical_expr::aggregate::{SlidingFrame, SlidingFrameAccumulator};

use crate::utils::PercentileParam;

create_func!(PercentileCont, percentile_cont_udaf);

/// Computes the exact percentile continuous of a set of numbers
pub fn percentile_cont(order_by: Sort, percentile: Expr) -> Expr {
    let expr = order_by.expr.clone();
    let args = vec![expr, percentile];

    Expr::AggregateFunction(AggregateFunction::new_udf(
        percentile_cont_udaf(),
        args,
        false,
        None,
        vec![order_by],
        None,
    ))
}

#[user_doc(
    doc_section(label = "General Functions"),
    description = "Returns the exact percentile of input values, interpolating between values if needed.",
    syntax_example = "percentile_cont(percentile) WITHIN GROUP (ORDER BY expression)",
    sql_example = r#"```sql
> SELECT percentile_cont(0.75) WITHIN GROUP (ORDER BY column_name) FROM table_name;
+-----------------------------------------------------------+
| percentile_cont(0.75) WITHIN GROUP (ORDER BY column_name) |
+-----------------------------------------------------------+
| 45.5                                                      |
+-----------------------------------------------------------+
```

An alternate syntax is also supported:
```sql
> SELECT percentile_cont(column_name, 0.75) FROM table_name;
+---------------------------------------+
| percentile_cont(column_name, 0.75)    |
+---------------------------------------+
| 45.5                                  |
+---------------------------------------+
```"#,
    standard_argument(name = "expression", prefix = "The"),
    argument(
        name = "percentile",
        description = "Percentile to compute. Must be a float value between 0 and 1 (inclusive)."
    )
)]
/// PERCENTILE_CONT aggregate expression. This uses an exact calculation and stores all values
/// in memory before computing the result. If an approximation is sufficient then
/// APPROX_PERCENTILE_CONT provides a much more efficient solution.
///
/// If using the distinct variation, the memory usage will be similarly high if the
/// cardinality is high as it stores all distinct values in memory before computing the
/// result, but if cardinality is low then memory usage will also be lower.
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct PercentileCont {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for PercentileCont {
    fn default() -> Self {
        Self::new()
    }
}

impl PercentileCont {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    // Decimal signature: decimals, percentile
                    TypeSignature::Coercible(vec![
                        // value
                        Coercion::new_exact(TypeSignatureClass::Decimal),
                        // percentile
                        Coercion::new_implicit_native(
                            logical_float64(),
                            vec![TypeSignatureClass::Numeric],
                        ),
                    ]),
                    // Float signature: float, percentile
                    TypeSignature::Coercible(vec![
                        Coercion::new_implicit(
                            TypeSignatureClass::Float,
                            vec![TypeSignatureClass::Numeric],
                            NativeType::Float64,
                        ),
                        Coercion::new_implicit_native(
                            logical_float64(),
                            vec![TypeSignatureClass::Numeric],
                        ),
                    ]),
                ],
                Volatility::Immutable,
            )
            .with_parameter_names(vec!["expr", "percentile"])
            .unwrap(),
            aliases: vec![String::from("quantile_cont")],
        }
    }
}

impl AggregateUDFImpl for PercentileCont {
    fn name(&self) -> &str {
        "percentile_cont"
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let input_type = args.input_fields[0].data_type().clone();
        if input_type.is_null() {
            return Ok(vec![
                Field::new(
                    format_state_name(args.name, self.name()),
                    DataType::Null,
                    true,
                )
                .into(),
            ]);
        }

        let field = Field::new_list_field(input_type, true);
        let state_name = if args.is_distinct {
            "distinct_percentile_cont"
        } else {
            "percentile_cont"
        };

        Ok(vec![
            Field::new(
                format_state_name(args.name, state_name),
                DataType::List(Arc::new(field)),
                true,
            )
            .into(),
            Field::new(
                format_state_name(args.name, "percentile"),
                DataType::Float64,
                true,
            )
            .into(),
        ])
    }

    fn accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        // Always verify percentiles
        let percentile = get_percentile_param(&args)?;
        create_percentile_accumulator(
            self.name(),
            percentile,
            args.expr_fields[0].data_type(),
            args.is_distinct,
        )
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        !args.is_distinct && !args.expr_fields[0].data_type().is_null()
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        // Always verify percentiles
        let percentile = get_percentile_param(&args)?;
        create_percentile_groups_accumulator(
            self.name(),
            percentile,
            args.expr_fields[0].data_type(),
        )
    }

    fn simplify(&self) -> Option<AggregateFunctionSimplification> {
        Some(Box::new(|aggregate_function, info| {
            simplify_percentile_cont_aggregate(aggregate_function, info)
        }))
    }

    fn supports_within_group_clause(&self) -> bool {
        true
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn get_percentile_param(args: &AccumulatorArgs) -> Result<PercentileParam> {
    let is_descending = args
        .order_bys
        .first()
        .map(|sort_expr| sort_expr.options.descending)
        .unwrap_or(false);

    let percentile =
        PercentileParam::try_new(&args.exprs[1], "PERCENTILE_CONT", is_descending)?;
    Ok(percentile)
}

pub fn create_percentile_accumulator(
    name: &str,
    percentile: PercentileParam,
    input_dt: &DataType,
    is_distinct: bool,
) -> Result<Box<dyn Accumulator>> {
    // Null input evaluates to null
    if input_dt.is_null() {
        return Ok(Box::new(NoopAccumulator::default()));
    }

    macro_rules! helper {
        ($t:ty, $i:ty, $dt:expr) => {
            if is_distinct {
                Ok(Box::new(DistinctPercentileContAccumulator::<$t, $i>::new(
                    percentile,
                    $dt.clone(),
                )))
            } else {
                Ok(Box::new(SlidingFrameAccumulator::new(
                    PercentileContAccumulator::<$t, $i>::new(percentile, $dt.clone()),
                )))
            }
        };
    }
    match input_dt {
        DataType::Float16 => helper!(Float16Type, FloatInterpolator, input_dt),
        DataType::Float32 => helper!(Float32Type, FloatInterpolator, input_dt),
        DataType::Float64 => helper!(Float64Type, FloatInterpolator, input_dt),
        DataType::Decimal32(_, _) => {
            helper!(Decimal32Type, DecimalInterpolator, input_dt)
        }
        DataType::Decimal64(_, _) => {
            helper!(Decimal64Type, DecimalInterpolator, input_dt)
        }
        DataType::Decimal128(_, _) => {
            helper!(Decimal128Type, DecimalInterpolator, input_dt)
        }
        DataType::Decimal256(_, _) => {
            helper!(Decimal256Type, DecimalInterpolator, input_dt)
        }
        dt => internal_err!("Unsupported datatype for {} with {}", name, dt),
    }
}

pub fn create_percentile_groups_accumulator(
    name: &str,
    percentile: PercentileParam,
    input_dt: &DataType,
) -> Result<Box<dyn GroupsAccumulator>> {
    macro_rules! helper {
        ($t:ty, $i:ty, $dt:expr) => {
            Ok(Box::new(PercentileContGroupsAccumulator::<$t, $i>::new(
                percentile,
                $dt.clone(),
            )))
        };
    }
    match input_dt {
        DataType::Float16 => helper!(Float16Type, FloatInterpolator, input_dt),
        DataType::Float32 => helper!(Float32Type, FloatInterpolator, input_dt),
        DataType::Float64 => helper!(Float64Type, FloatInterpolator, input_dt),
        DataType::Decimal32(_, _) => {
            helper!(Decimal32Type, DecimalInterpolator, input_dt)
        }
        DataType::Decimal64(_, _) => {
            helper!(Decimal64Type, DecimalInterpolator, input_dt)
        }
        DataType::Decimal128(_, _) => {
            helper!(Decimal128Type, DecimalInterpolator, input_dt)
        }
        DataType::Decimal256(_, _) => {
            helper!(Decimal256Type, DecimalInterpolator, input_dt)
        }
        dt => internal_err!("Unsupported datatype for {} with {}", name, dt),
    }
}

fn simplify_percentile_cont_aggregate(
    aggregate_function: AggregateFunction,
    info: &SimplifyContext,
) -> Result<Expr> {
    enum PercentileRewriteTarget {
        Min,
        Max,
    }

    let params = &aggregate_function.params;
    let [value, percentile] = take_function_args("percentile_cont", &params.args)?;
    //
    // For simplicity we don't bother with null types (otherwise we'd need to
    // cast the return type)
    let input_type = info.get_data_type(value)?;
    if input_type.is_null() {
        return Ok(Expr::AggregateFunction(aggregate_function));
    }

    let is_descending = params
        .order_by
        .first()
        .map(|sort| !sort.asc)
        .unwrap_or(false);

    let rewrite_target = match percentile {
        Expr::Literal(ScalarValue::Float64(Some(0.0)), _) => {
            if is_descending {
                PercentileRewriteTarget::Max
            } else {
                PercentileRewriteTarget::Min
            }
        }
        Expr::Literal(ScalarValue::Float64(Some(1.0)), _) => {
            if is_descending {
                PercentileRewriteTarget::Min
            } else {
                PercentileRewriteTarget::Max
            }
        }
        _ => return Ok(Expr::AggregateFunction(aggregate_function)),
    };

    let udaf = match rewrite_target {
        PercentileRewriteTarget::Min => min_udaf(),
        PercentileRewriteTarget::Max => max_udaf(),
    };

    let rewritten = Expr::AggregateFunction(AggregateFunction::new_udf(
        udaf,
        vec![value.clone()],
        params.distinct,
        params.filter.clone(),
        vec![],
        params.null_treatment,
    ));
    Ok(rewritten)
}

const FRAME_BLOCK_SIZE: usize = 512;

/// Blocks are ordered by value, with duplicates allowed across boundaries.
#[derive(Debug)]
struct SortedFrame<T> {
    blocks: Vec<Vec<T>>,
    len: usize,
    // The blocks and directory drop before their charge is released.
    reservation: Option<MemoryReservation>,
}

impl<T: ArrowNativeTypeOp> SortedFrame<T> {
    fn from_values(mut values: Vec<T>) -> Self {
        values.sort_unstable_by(|a, b| a.compare(*b));
        let len = values.len();
        let blocks = if len <= 2 * FRAME_BLOCK_SIZE {
            if len == 0 { Vec::new() } else { vec![values] }
        } else {
            values.chunks(FRAME_BLOCK_SIZE).map(<[_]>::to_vec).collect()
        };
        Self {
            blocks,
            len,
            reservation: None,
        }
    }

    fn block_index(&self, value: T) -> usize {
        self.blocks
            .partition_point(|block| block[block.len() - 1].compare(value).is_lt())
    }

    fn insert(&mut self, value: T) -> Result<()> {
        if self.blocks.is_empty() {
            grow_frame_vec(&mut self.blocks, 1, self.reservation.as_ref())?;
            let mut block = Vec::new();
            grow_frame_vec(&mut block, 1, self.reservation.as_ref())?;
            block.push(value);
            self.blocks.push(block);
            self.len = 1;
            return Ok(());
        }
        let index = self.block_index(value).min(self.blocks.len() - 1);
        let block = &mut self.blocks[index];
        let offset = block.partition_point(|item| item.compare(value).is_lt());
        if block.len() < 2 * FRAME_BLOCK_SIZE {
            grow_frame_vec(block, block.len() + 1, self.reservation.as_ref())?;
            block.insert(offset, value);
        } else {
            // Admit the directory and tail before modifying the full block.
            // Moving one extra value when inserting on the left leaves the
            // split at the same rank without growing the old block.
            let required = self.blocks.len() + 1;
            grow_frame_vec(&mut self.blocks, required, self.reservation.as_ref())?;
            let mut tail = Vec::new();
            grow_frame_vec(&mut tail, FRAME_BLOCK_SIZE + 1, self.reservation.as_ref())?;
            let block = &mut self.blocks[index];
            let insert_left = offset < FRAME_BLOCK_SIZE;
            let split = FRAME_BLOCK_SIZE - usize::from(insert_left);
            tail.extend_from_slice(&block[split..]);
            block.truncate(split);
            if insert_left {
                block.insert(offset, value);
            } else {
                tail.insert(offset - split, value);
            }
            self.blocks.insert(index + 1, tail);
        }
        self.len += 1;
        Ok(())
    }

    fn remove(&mut self, value: T) -> bool {
        let index = self.block_index(value);
        let Some(block) = self.blocks.get_mut(index) else {
            return false;
        };
        let Ok(offset) = block.binary_search_by(|item| item.compare(value)) else {
            return false;
        };
        block.remove(offset);
        self.len -= 1;
        if block.is_empty() {
            let block = self.blocks.remove(index);
            self.release_block(block);
        } else if let Some(next) = self.blocks.get(index + 1)
            && self.blocks[index].len() + next.len() <= FRAME_BLOCK_SIZE
        {
            // Leave room between the merge and split thresholds.
            let next = self.blocks.remove(index + 1);
            // Split blocks retain at least FRAME_BLOCK_SIZE capacity.
            debug_assert!(
                self.blocks[index].len() + next.len() <= self.blocks[index].capacity()
            );
            self.blocks[index].extend_from_slice(&next);
            self.release_block(next);
        }
        true
    }

    fn release_block(&self, block: Vec<T>) {
        let bytes = block.capacity() * size_of::<T>();
        drop(block);
        if let Some(reservation) = &self.reservation {
            reservation.shrink(bytes);
        }
    }

    /// Update a single block with one move of its intervening values.
    fn replace_in_single_block(&mut self, old: T, new: T) -> Result<()> {
        let block = &mut self.blocks[0];
        let from = block.partition_point(|value| value.compare(old).is_lt());
        if from == block.len() || !block[from].is_eq(old) {
            return missing_frame_value();
        }
        let to = block.partition_point(|value| value.compare(new).is_lt());
        if to > from {
            block.copy_within(from + 1..to, from);
            block[to - 1] = new;
        } else {
            block.copy_within(to..from, to + 1);
            block[to] = new;
        }
        Ok(())
    }

    fn ranks(&self, mut lower: usize, mut upper: usize) -> (T, T) {
        let mut blocks = self.blocks.iter();
        while let Some(block) = blocks.next() {
            if lower < block.len() {
                let first = block[lower];
                if upper < block.len() {
                    return (first, block[upper]);
                }
                upper -= block.len();
                for block in blocks {
                    if upper < block.len() {
                        return (first, block[upper]);
                    }
                    upper -= block.len();
                }
                break;
            }
            lower -= block.len();
            upper -= block.len();
        }
        unreachable!("percentile ranks must be inside the frame")
    }
}

/// Reserve both allocations during growth, since the allocator can move the
/// original buffer. Spare capacity remains charged until the buffer is freed.
fn grow_frame_vec<T>(
    values: &mut Vec<T>,
    required: usize,
    reservation: Option<&MemoryReservation>,
) -> Result<()> {
    let old = values.capacity();
    if required <= old {
        return Ok(());
    }
    let capacity = required.max(old.saturating_mul(2)).max(4);
    let bytes = capacity.checked_mul(size_of::<T>()).ok_or_else(|| {
        datafusion_common::resources_datafusion_err!(
            "percentile sliding frame memory overflow"
        )
    })?;
    if let Some(reservation) = reservation {
        reservation.try_grow(bytes)?;
    }
    let result = values.try_reserve_exact(capacity - values.len());
    debug_assert!(result.is_err() || values.capacity() == capacity);
    if let Some(reservation) = reservation {
        reservation.shrink(if result.is_ok() {
            old * size_of::<T>()
        } else {
            bytes
        });
    }
    result.map_err(|error| {
        datafusion_common::resources_datafusion_err!(
            "cannot allocate percentile sliding frame: {error}"
        )
    })
}

fn missing_frame_value<T>() -> Result<T> {
    internal_err!(
        "percentile_cont retract_batch: retracted value(s) not present in the window"
    )
}

/// The percentile_cont accumulator accumulates the raw input values
/// as native types.
///
/// The intermediate state is represented as a List of scalar values updated by
/// `merge_batch` and a `Vec` of native values that are converted to scalar values
/// in the final evaluation step so that we avoid expensive conversions and
/// allocations during `update_batch`.
#[derive(Debug)]
struct PercentileContAccumulator<
    T: ArrowNumericType + Debug,
    I: PercentileInterpolator<T>,
> {
    all_values: Vec<T::Native>,
    frame: Option<SortedFrame<T::Native>>,
    percentile: PercentileParam,
    data_type: DataType,
    _interpolator: PhantomData<I>,
}

impl<T: ArrowNumericType + Debug, I: PercentileInterpolator<T>>
    PercentileContAccumulator<T, I>
{
    fn new(percentile: PercentileParam, data_type: DataType) -> Self {
        Self {
            all_values: vec![],
            frame: None,
            percentile,
            data_type,
            _interpolator: PhantomData,
        }
    }
}

impl<T, I> Accumulator for PercentileContAccumulator<T, I>
where
    T: ArrowNumericType + Debug,
    I: PercentileInterpolator<T> + 'static,
{
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut frame_charge = None;
        if let Some(frame) = self.frame.take() {
            if let Some(reservation) = &frame.reservation {
                reservation.try_grow(frame.len * size_of::<T::Native>())?;
            }
            let mut values = Vec::with_capacity(frame.len);
            for block in frame.blocks {
                values.extend_from_slice(&block);
            }
            self.all_values = values;
            frame_charge = frame.reservation;
        }
        // Convert `all_values` to `ListArray` and return a single List ScalarValue

        // Build offsets
        let offsets =
            OffsetBuffer::new(ScalarBuffer::from(vec![0, self.all_values.len() as i32]));

        // Build inner array
        let values_array = PrimitiveArray::<T>::new(
            ScalarBuffer::from(std::mem::take(&mut self.all_values)),
            None,
        )
        .with_data_type(self.data_type.clone());

        // Build the result list array
        let list_array = ListArray::new(
            Arc::new(Field::new_list_field(self.data_type.clone(), true)),
            offsets,
            Arc::new(values_array),
            None,
        );

        // The returned state now owns its flattened buffer.
        drop(frame_charge);
        Ok(vec![
            ScalarValue::List(Arc::new(list_array)),
            ScalarValue::Float64(self.percentile.get().ok()),
        ])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() > 1 {
            self.percentile.resolve(&values[1])?;
        }

        let values = values[0].as_primitive::<T>();
        if let Some(frame) = &mut self.frame {
            if values.null_count() == 0 {
                for &value in values.values() {
                    frame.insert(value)?;
                }
            } else {
                for value in values.iter().flatten() {
                    frame.insert(value)?;
                }
            }
            return Ok(());
        }
        let additional = values.len() - values.null_count();
        self.all_values.try_reserve(additional).map_err(|e| {
            exec_datafusion_err!(
                "failed to reserve {additional} values for percentile_cont accumulator: {e}"
            )
        })?;
        if values.null_count() > 0 {
            self.all_values.extend(values.iter().flatten());
        } else {
            // Fast path: no nulls, so the values buffer can be appended wholesale.
            self.all_values.extend_from_slice(values.values());
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if let Some(percentile_array) = states.get(1) {
            self.percentile.resolve(percentile_array)?;
        }
        let array = states[0].as_list::<i32>();
        // Feed all list elements from a batch
        for values in array.iter().flatten() {
            self.update_batch(&[values])?;
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if let Some(frame) = &self.frame {
            let value = match frame.len {
                0 => None,
                1 => Some(frame.blocks[0][0]),
                len => Some(I::percentile_from_ranks(
                    len,
                    self.percentile.get()?,
                    self.percentile.is_desc,
                    &self.data_type,
                    |lower, upper| frame.ranks(lower, upper),
                )?),
            };
            return I::frame_scalar(value, &self.data_type);
        }
        if self.all_values.is_empty() {
            return ScalarValue::new_primitive::<T>(None, &self.data_type);
        }
        let value = calculate_percentile::<T, I>(
            &mut self.all_values,
            self.percentile.get()?,
            self.percentile.is_desc,
            &self.data_type,
        )?;
        ScalarValue::new_primitive::<T>(value, &self.data_type)
    }

    fn size(&self) -> usize {
        size_of_val(self)
            + self.all_values.capacity() * size_of::<T::Native>()
            + self.frame.as_ref().map_or(0, |frame| {
                frame.blocks.capacity() * size_of::<Vec<T::Native>>()
                    + frame.blocks.iter().map(Vec::capacity).sum::<usize>()
                        * size_of::<T::Native>()
            })
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = values[0].as_primitive::<T>();
        // Physical windows enable the sorted layout at construction.
        // Direct accumulator callers can enter the sliding layout here.
        self.enable_sliding();
        let frame = self.frame.as_mut().unwrap();
        for value in values.iter().flatten() {
            if !frame.remove(value) {
                return missing_frame_value();
            }
        }
        Ok(())
    }

    fn supports_retract_batch(&self) -> bool {
        true
    }
}

impl<T, I> SlidingFrame for PercentileContAccumulator<T, I>
where
    T: ArrowNumericType + Debug,
    I: PercentileInterpolator<T> + 'static,
{
    fn enable_sliding(&mut self) {
        self.frame.get_or_insert_with(|| {
            SortedFrame::from_values(std::mem::take(&mut self.all_values))
        });
    }

    fn set_frame_reservation(&mut self, reservation: MemoryReservation) {
        // Cumulative windows also use this accumulator, with the ordinary
        // selection buffer. Only the blocked layout uses this reservation.
        if let Some(frame) = &mut self.frame {
            assert!(frame.blocks.capacity() == 0 && frame.reservation.is_none());
            frame.reservation = Some(reservation);
        }
    }

    fn advance(
        &mut self,
        values: &ArrayRef,
        leaving: Range<usize>,
        entering: Range<usize>,
        _retract_first: bool,
    ) -> Result<ScalarValue> {
        let values = values.as_primitive::<T>();
        let frame = self.frame.as_mut().expect("sliding frame was initialized");
        if values.null_count() == 0 {
            let values = values.values();
            if leaving.len() == 1 && entering.len() == 1 && frame.blocks.len() == 1 {
                frame.replace_in_single_block(
                    values[leaving.start],
                    values[entering.start],
                )?;
            } else {
                for &value in &values[leaving] {
                    if !frame.remove(value) {
                        return missing_frame_value();
                    }
                }
                for &value in &values[entering] {
                    frame.insert(value)?;
                }
            }
        } else {
            for index in leaving {
                if values.is_valid(index) && !frame.remove(values.value(index)) {
                    return missing_frame_value();
                }
            }
            for index in entering {
                if values.is_valid(index) {
                    frame.insert(values.value(index))?;
                }
            }
        }
        self.evaluate()
    }
}

/// The percentile_cont groups accumulator accumulates the raw input values
///
/// For calculating the exact percentile of groups, we need to store all values
/// of groups before final evaluation.
/// So values in each group will be stored in a `Vec<T>`, and the total group values
/// will be actually organized as a `Vec<Vec<T>>`.
#[derive(Debug)]
struct PercentileContGroupsAccumulator<
    T: ArrowNumericType + Debug,
    I: PercentileInterpolator<T>,
> {
    group_values: Vec<Vec<T::Native>>,
    percentile: PercentileParam,
    /// Per-group percentiles, only used when the percentile is given as a
    /// column reference: it then has to be constant within each group, but
    /// may differ between groups.
    group_percentiles: Option<Vec<Option<f64>>>,
    data_type: DataType,
    _interpolator: PhantomData<I>,
}

impl<T: ArrowNumericType + Debug, I: PercentileInterpolator<T>>
    PercentileContGroupsAccumulator<T, I>
{
    fn new(percentile: PercentileParam, data_type: DataType) -> Self {
        let group_percentiles = percentile.is_col_ref().then(Vec::new);
        Self {
            group_values: vec![],
            percentile,
            group_percentiles,
            data_type,
            _interpolator: PhantomData,
        }
    }

    /// Resolves the percentile of each group from the `percentiles` argument,
    /// either per group or, for a literal percentile, once for all groups.
    fn resolve_percentiles(
        &mut self,
        percentiles: &ArrayRef,
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        let Some(group_percentiles) = self.group_percentiles.as_mut() else {
            return self.percentile.resolve(percentiles);
        };
        group_percentiles.resize(total_num_groups, None);

        let percentiles = self.percentile.to_float64_array(percentiles)?;
        for (row, (&group_index, value)) in
            group_indices.iter().zip(percentiles.iter()).enumerate()
        {
            let Some(value) = value else { continue };
            if opt_filter.is_some_and(|f| !f.is_valid(row) || !f.value(row)) {
                continue;
            }
            group_percentiles[group_index] = Some(
                self.percentile
                    .resolve_group_value(group_percentiles[group_index], value)?,
            );
        }
        Ok(())
    }
}

impl<T, I> GroupsAccumulator for PercentileContGroupsAccumulator<T, I>
where
    T: ArrowNumericType + Debug + Send,
    I: PercentileInterpolator<T> + 'static,
{
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        if values.len() > 1 {
            self.resolve_percentiles(
                &values[1],
                group_indices,
                opt_filter,
                total_num_groups,
            )?;
        }

        let values = values[0].as_primitive::<T>();

        // Push the `not nulls + not filtered` row into its group
        self.group_values.resize(total_num_groups, Vec::new());
        accumulate(
            group_indices,
            values,
            opt_filter,
            |group_index, new_value| {
                self.group_values[group_index].push(new_value);
            },
        );

        Ok(())
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        if let Some(percentile_array) = values.get(1) {
            self.resolve_percentiles(
                percentile_array,
                group_indices,
                None,
                total_num_groups,
            )?;
        }

        let input_group_values = values[0].as_list::<i32>();

        // Ensure group values big enough
        self.group_values.resize(total_num_groups, Vec::new());

        // Extend values to related groups
        group_indices
            .iter()
            .zip(input_group_values.iter())
            .for_each(|(&group_index, values_opt)| {
                if let Some(values) = values_opt {
                    let values = values.as_primitive::<T>();
                    self.group_values[group_index].extend(values.values().iter());
                }
            });

        Ok(())
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        // Emit values
        let emit_group_values = emit_to.take_needed(&mut self.group_values);
        let num_groups = emit_group_values.len();

        // Build offsets
        let mut offsets = Vec::with_capacity(emit_group_values.len() + 1);
        offsets.push(0);
        let mut cur_len = 0_i32;
        for group_value in &emit_group_values {
            cur_len += group_value.len() as i32;
            offsets.push(cur_len);
        }
        let offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));

        // Build inner array
        let mut flatten_group_values: Vec<T::Native> =
            Vec::with_capacity(cur_len as usize);
        flatten_group_values.extend(emit_group_values.into_iter().flatten());
        let group_values_array =
            PrimitiveArray::<T>::new(ScalarBuffer::from(flatten_group_values), None)
                .with_data_type(self.data_type.clone());

        // Build the result list array
        let result_list_array = ListArray::new(
            Arc::new(Field::new_list_field(self.data_type.clone(), true)),
            offsets,
            Arc::new(group_values_array),
            None,
        );

        let percentile_array: ArrayRef = match self.group_percentiles.as_mut() {
            Some(group_percentiles) => Arc::new(arrow::array::Float64Array::from(
                emit_to.take_needed(group_percentiles),
            )),
            None => Arc::new(arrow::array::Float64Array::from(vec![
                self.percentile
                    .get()
                    .ok();
                num_groups
            ])),
        };

        Ok(vec![Arc::new(result_list_array), percentile_array])
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        // Emit values
        let mut emit_group_values = emit_to.take_needed(&mut self.group_values);

        // Calculate percentile for each group
        let mut evaluate_result_builder =
            PrimitiveBuilder::<T>::with_capacity(emit_group_values.len())
                .with_data_type(self.data_type.clone());
        match self.group_percentiles.as_mut() {
            Some(group_percentiles) => {
                let mut emit_group_percentiles = emit_to.take_needed(group_percentiles);
                emit_group_percentiles.resize(emit_group_values.len(), None);
                for (values, percentile) in
                    emit_group_values.iter_mut().zip(emit_group_percentiles)
                {
                    if values.is_empty() {
                        evaluate_result_builder.append_null();
                        continue;
                    }
                    let value = calculate_percentile::<T, I>(
                        values.as_mut_slice(),
                        self.percentile.get_group_value(percentile)?,
                        self.percentile.is_desc,
                        &self.data_type,
                    )?;
                    evaluate_result_builder.append_option(value);
                }
            }
            None => {
                let percentile = self.percentile.get()?;
                for values in &mut emit_group_values {
                    let value = calculate_percentile::<T, I>(
                        values.as_mut_slice(),
                        percentile,
                        self.percentile.is_desc,
                        &self.data_type,
                    )?;
                    evaluate_result_builder.append_option(value);
                }
            }
        }

        Ok(Arc::new(evaluate_result_builder.finish()))
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        // As in `update_batch`, only the first element holds the values; the
        // percentile parameter and the ORDER BY columns follow it
        let input_array = values[0].as_primitive::<T>();

        // Directly convert the input array to states, each row will be
        // seen as a respective group.
        // For detail, the `input_array` will be converted to a `ListArray`.
        // And if row is `not null + not filtered`, it will be converted to a list
        // with only one element; otherwise, this row in `ListArray` will be set
        // to null.

        // Reuse values buffer in `input_array` to build `list_values` in `ListArray`
        let list_values = PrimitiveArray::<T>::new(input_array.values().clone(), None)
            .with_data_type(self.data_type.clone());

        // `offsets` in `ListArray`, each row as a list element
        let offset_end = i32::try_from(input_array.len()).map_err(|e| {
            internal_datafusion_err!(
                "cast array_len to i32 failed in convert_to_state of group percentile_cont, err:{e:?}"
            )
        })?;
        let offsets = (0..=offset_end).collect::<Vec<_>>();
        // Safety: The offsets vector is constructed as a sequential range from 0 to input_array.len(),
        // which guarantees all OffsetBuffer invariants:
        // 1. Offsets are monotonically increasing (each element is prev + 1)
        // 2. No offset exceeds the values array length (max offset = input_array.len())
        // 3. First offset is 0 and last offset equals the total length
        // Therefore new_unchecked is safe to use here.
        let offsets = unsafe { OffsetBuffer::new_unchecked(ScalarBuffer::from(offsets)) };

        // `nulls` for converted `ListArray`
        let nulls = filtered_null_mask(opt_filter, input_array);

        let converted_list_array = ListArray::new(
            Arc::new(Field::new_list_field(self.data_type.clone(), true)),
            offsets,
            Arc::new(list_values),
            nulls,
        );

        // `state_fields` always declares a second (percentile) state field, so
        // `convert_to_state` must also produce one array per row to match it.
        // When the percentile is given as a column reference, propagate its
        // per-row values (nulled out for filtered rows); otherwise broadcast
        // the resolved constant.
        let percentile_array: ArrayRef = if let Some(percentile_values) = values.get(1) {
            let percentile_values =
                self.percentile.to_float64_array(percentile_values)?;
            let nulls = filtered_null_mask(opt_filter, &percentile_values);
            Arc::new(arrow::array::Float64Array::new(
                percentile_values.values().clone(),
                nulls,
            ))
        } else {
            let percentile = self.percentile.get().ok();
            Arc::new(arrow::array::Float64Array::from(vec![
                percentile;
                input_array.len()
            ]))
        };

        Ok(vec![Arc::new(converted_list_array), percentile_array])
    }
    fn size(&self) -> usize {
        self.group_values
            .iter()
            .map(|values| values.capacity() * size_of::<T::Native>())
            .sum::<usize>()
            // account for size of self.group_values too
            + self.group_values.capacity() * size_of::<Vec<T::Native>>()
            + self
                .group_percentiles
                .as_ref()
                .map_or(0, |p| p.capacity() * size_of::<Option<f64>>())
    }
}

/// Sliding-window–capable accumulator for `percentile_cont(DISTINCT ...)`.
///
/// Distinct values are tracked with a per-value multiplicity count (how many
/// rows currently in the window carry that value) rather than a plain set, so
/// that `retract_batch` only drops a value once *all* of its occurrences have
/// left the window frame. The percentile is then computed over the set of keys
/// with a positive count.
#[derive(Debug)]
struct DistinctPercentileContAccumulator<
    T: ArrowNumericType,
    I: PercentileInterpolator<T>,
> {
    /// Distinct value -> number of in-window rows carrying it.
    ///
    /// Uses the same fast (foldhash) `RandomState` as the shared
    /// `GenericDistinctBuffer` rather than the standard library's default
    /// SipHash, which is considerably slower for this hot path.
    counts: HashMap<Hashable<T::Native>, usize, RandomState>,
    percentile: PercentileParam,
    data_type: DataType,
    _interpolator: PhantomData<I>,
}

impl<T: ArrowNumericType, I: PercentileInterpolator<T>>
    DistinctPercentileContAccumulator<T, I>
{
    fn new(percentile: PercentileParam, data_type: DataType) -> Self {
        Self {
            counts: HashMap::default(),
            percentile,
            data_type,
            _interpolator: PhantomData,
        }
    }
}

impl<T, I> Accumulator for DistinctPercentileContAccumulator<T, I>
where
    T: ArrowNumericType + Debug,
    I: PercentileInterpolator<T> + 'static,
{
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        // Emit the distinct keys as a single List scalar, matching the state
        // shape declared in `state_fields` (a List of the input type). Counts
        // are window-local bookkeeping and are intentionally not serialized:
        // cross-partition merges only need the distinct key set.
        let arr = Arc::new(
            PrimitiveArray::<T>::from_iter_values(self.counts.keys().map(|v| v.0))
                .with_data_type(self.data_type.clone()),
        );
        Ok(vec![
            SingleRowListArrayBuilder::new(arr).build_list_scalar(),
            ScalarValue::Float64(self.percentile.get().ok()),
        ])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() > 1 {
            self.percentile.resolve(&values[1])?;
        }

        let arr = values[0].as_primitive::<T>();
        if arr.null_count() > 0 {
            for value in arr.iter().flatten() {
                *self.counts.entry(Hashable(value)).or_default() += 1;
            }
        } else {
            // Fast path: no nulls, so skip the per-element validity check.
            for value in arr.values().iter() {
                *self.counts.entry(Hashable(*value)).or_default() += 1;
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if let Some(percentile_array) = states.get(1) {
            self.percentile.resolve(percentile_array)?;
        }
        let list = states[0].as_list::<i32>();
        for values in list.iter().flatten() {
            let arr = values.as_primitive::<T>();
            for value in arr.iter().flatten() {
                *self.counts.entry(Hashable(value)).or_default() += 1;
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut values: Vec<T::Native> = self.counts.keys().map(|v| v.0).collect();
        if values.is_empty() {
            return ScalarValue::new_primitive::<T>(None, &self.data_type);
        }
        let value = calculate_percentile::<T, I>(
            &mut values,
            self.percentile.get()?,
            self.percentile.is_desc,
            &self.data_type,
        )?;
        ScalarValue::new_primitive::<T>(value, &self.data_type)
    }

    fn size(&self) -> usize {
        estimate_memory_size::<(Hashable<T::Native>, usize)>(
            self.counts.capacity(),
            size_of_val(self),
        )
        .unwrap()
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let arr = values[0].as_primitive::<T>();
        let mut decrement = |value: T::Native| {
            match self.counts.get_mut(&Hashable(value)) {
                Some(count) => {
                    *count -= 1;
                    if *count == 0 {
                        self.counts.remove(&Hashable(value));
                    }
                    Ok(())
                }
                // Retracting a value that isn't tracked means the accumulator
                // state has diverged from the window frame; continuing would
                // silently produce wrong results, so surface it as an error.
                None => internal_err!(
                    "percentile_cont(DISTINCT) retract_batch: retracted a value not present in the window"
                ),
            }
        };
        if arr.null_count() > 0 {
            for value in arr.iter().flatten() {
                decrement(value)?;
            }
        } else {
            // Fast path: no nulls, so skip the per-element validity check.
            for value in arr.values().iter() {
                decrement(*value)?;
            }
        }
        Ok(())
    }

    fn supports_retract_batch(&self) -> bool {
        true
    }
}

/// A trait to abstract the rank selection and interpolation of a continuous
/// percentile for floats and decimals.
trait PercentileInterpolator<T: ArrowNumericType>: Debug + Sync + Send {
    fn frame_scalar(
        value: Option<T::Native>,
        data_type: &DataType,
    ) -> Result<ScalarValue> {
        ScalarValue::new_primitive::<T>(value, data_type)
    }

    /// Returns the continuous percentile of `values`, which holds at least
    /// two values. `percentile` is the requested fraction in `[0, 1]` before
    /// the flip for `WITHIN GROUP (ORDER BY ... DESC)`, which `descending`
    /// requests. `data_type` is the input (and result) type.
    fn percentile_of(
        values: &mut [T::Native],
        percentile: f64,
        descending: bool,
        data_type: &DataType,
    ) -> Result<T::Native> {
        Self::percentile_from_ranks(
            values.len(),
            percentile,
            descending,
            data_type,
            |lower, upper| select_ranks::<T>(values, lower, upper),
        )
    }

    fn percentile_from_ranks(
        len: usize,
        percentile: f64,
        descending: bool,
        data_type: &DataType,
        ranks: impl FnOnce(usize, usize) -> (T::Native, T::Native),
    ) -> Result<T::Native>;
}

/// Selects the values at the ascending ranks `lower <= upper`.
fn select_ranks<T: ArrowPrimitiveType>(
    values: &mut [T::Native],
    lower: usize,
    upper: usize,
) -> (T::Native, T::Native) {
    let cmp = |x: &T::Native, y: &T::Native| x.compare(*y);
    if lower == upper {
        if lower == 0 {
            let value = *values.iter().min_by(|a, b| cmp(a, b)).unwrap();
            return (value, value);
        }
        if lower == values.len() - 1 {
            let value = *values.iter().max_by(|a, b| cmp(a, b)).unwrap();
            return (value, value);
        }
    }
    // First, partition at lower to get the lower value
    let (_, lower_value, _) = values.select_nth_unstable_by(lower, cmp);
    let lower_value = *lower_value;
    if lower == upper {
        return (lower_value, lower_value);
    }
    // Then partition at upper to get the upper value
    let (_, upper_value, _) = values.select_nth_unstable_by(upper, cmp);
    (lower_value, *upper_value)
}

#[derive(Debug)]
struct FloatInterpolator;

/// The interpolation arithmetic for floats is performed in f64 and then cast back to the
/// native type to avoid overflowing Float16 intermediates.
impl<T> PercentileInterpolator<T> for FloatInterpolator
where
    T: ArrowNumericType,
    T::Native: AsPrimitive<f64>,
    f64: AsPrimitive<T::Native>,
    ScalarValue: From<Option<T::Native>>,
{
    fn frame_scalar(
        value: Option<T::Native>,
        _data_type: &DataType,
    ) -> Result<ScalarValue> {
        // A direct scalar avoids a one-row Arrow allocation on every frame.
        Ok(value.into())
    }

    fn percentile_from_ranks(
        len: usize,
        percentile: f64,
        descending: bool,
        _data_type: &DataType,
        ranks: impl FnOnce(usize, usize) -> (T::Native, T::Native),
    ) -> Result<T::Native> {
        let percentile = if descending {
            1.0 - percentile
        } else {
            percentile
        };
        // Calculate the index using the formula: p * (n - 1)
        let index = percentile * ((len - 1) as f64);
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;
        let (lower, upper) = ranks(lower_index, upper_index);
        if lower_index == upper_index {
            // Exact index, return the value at that position
            return Ok(lower);
        }
        let fraction = index - (lower_index as f64);

        // Use the full fraction. Quantizing it to six decimal places can
        // move a result by a millionth of the neighbouring-value gap.
        // Widen before arithmetic to retain Float16 overflow safety.
        let lower_f: f64 = lower.as_();
        let upper_f: f64 = upper.as_();
        let interpolated_f = if T::DATA_TYPE == DataType::Float64 {
            // DuckDB specializes DOUBLE with weighted endpoints, which also
            // avoid overflow in (upper - lower). Its FLOAT overload uses
            // widened delta arithmetic instead (notably, identical infinite
            // FLOAT endpoints interpolate to NaN).
            lower_f * (1.0 - fraction) + upper_f * fraction
        } else {
            lower_f + (upper_f - lower_f) * fraction
        };
        Ok(interpolated_f.as_())
    }
}

/// DuckDB 2.0.0's continuous percentile of decimals.
///
/// DuckDB (`QuantileInterpolator` in `quantile_sort_tree.hpp`,
/// `InterpolateOperator` in `interpolate.hpp` and `arithmetic.cpp`) computes
/// the binary64 rank `RN = (n - 1) * p` in the requested sort order, reads the
/// values at `FRN = floor(RN)` and `CRN = ceil(RN)` of that order, returns
/// the exact value when they coincide, and otherwise interpolates from the
/// value at FRN (`first`) toward the value at CRN (`second`) with the
/// binary64 fraction `RN - FRN`. The result keeps the input decimal type.
/// The arithmetic depends on DuckDB's storage class for the declared
/// precision (not the Arrow storage width: `DECIMAL(18, 2)` is
/// `Decimal128(18, 2)` here but int64 in DuckDB):
///
/// - Precision <= 18 (int16/32/64): `fma(second - first, fraction, first)`
///   on the unscaled values as doubles, truncated toward zero. The pinned
///   DuckDB binary contracts `lo + (hi - lo) * d` to this fused form.
/// - Precision 19..=38 (HUGEINT): the exact i128 `second - first`, converted
///   to double by DuckDB's two-limb rule, times the fraction, truncated
///   toward zero and added to the exact `first`; when the difference
///   overflows i128, `fma(first, 1 - fraction, second * fraction)` in double,
///   truncated. A double outside the i128 range is an error.
/// - Precision > 38 (`Decimal256` only; DuckDB has no such type, so there is
///   no reference): the HUGEINT rule in 256 bits.
///
/// `WITHIN GROUP (ORDER BY ... DESC)` keeps `p` and interpolates in the
/// descending order, as DuckDB does: precision > 18 then rounds toward the
/// larger value. Binary64 rounding can carry a precision <= 18 result past
/// the declared precision (DuckDB's median of two 999999999999999999 values
/// as DECIMAL(18, 0) is 10^18); the coefficient is returned unchanged and
/// still fits the storage type.
#[derive(Debug)]
struct DecimalInterpolator;

fn decimal_overflow() -> datafusion_common::DataFusionError {
    exec_datafusion_err!("percentile_cont decimal interpolation overflow")
}

/// DuckDB 2.0.0's `Hugeint::Cast<double>` adds the two 64-bit limbs, which
/// can round differently from a direct i128 cast near a limb boundary.
fn hugeint_to_double(value: i128) -> f64 {
    let lower = value as u64;
    let upper = (value >> 64) as i64;
    if upper == -1 {
        -((u64::MAX - lower) as f64) - 1.0
    } else {
        lower as f64 + upper as f64 * 18446744073709551616.0
    }
}

/// DuckDB's `Hugeint::Convert(double)`: truncates toward zero and rejects
/// non-finite values and both binary64 i128 boundaries. Checked before
/// Rust's saturating float-to-integer cast could hide an overflow.
fn hugeint_from_double(value: f64) -> Result<i128> {
    let bound = 170141183460469231731687303715884105728.0;
    if !value.is_finite() || value <= -bound || value >= bound {
        return Err(decimal_overflow());
    }
    Ok(value as i128)
}

/// Interpolates DuckDB's way for precision <= 38; `narrow` selects the
/// int16/32/64 rule (precision <= 18).
fn interpolate_hugeint(
    first: i128,
    second: i128,
    fraction: f64,
    narrow: bool,
) -> Result<i128> {
    if narrow {
        // The values have at most 18 digits, so `as f64` is DuckDB's
        // int64-to-double conversion.
        let lo = first as f64;
        let hi = second as f64;
        return hugeint_from_double((hi - lo).mul_add(fraction, lo));
    }
    match second.checked_sub(first) {
        Some(delta) => {
            let increment = hugeint_from_double(hugeint_to_double(delta) * fraction)?;
            first.checked_add(increment).ok_or_else(decimal_overflow)
        }
        // Opposite DECIMAL(38) extremes overflow the i128 difference.
        None => hugeint_from_double(
            hugeint_to_double(first)
                .mul_add(1.0 - fraction, hugeint_to_double(second) * fraction),
        ),
    }
}

/// The HUGEINT rule in 256 bits, for `Decimal256` precision > 38 (no DuckDB
/// reference). `i256::to_f64` rounds to nearest; `i256::from_f64` truncates
/// toward zero and rejects non-finite and out-of-range values.
fn interpolate_wide(first: i256, second: i256, fraction: f64) -> Result<i256> {
    let to_double =
        |v: i256| num_traits::ToPrimitive::to_f64(&v).ok_or_else(decimal_overflow);
    match second.checked_sub(first) {
        Some(delta) => {
            let increment = i256::from_f64(to_double(delta)? * fraction)
                .ok_or_else(decimal_overflow)?;
            first.checked_add(increment).ok_or_else(decimal_overflow)
        }
        None => i256::from_f64(
            to_double(first)?.mul_add(1.0 - fraction, to_double(second)? * fraction),
        )
        .ok_or_else(decimal_overflow),
    }
}

impl<T> PercentileInterpolator<T> for DecimalInterpolator
where
    T: DecimalType,
    T::Native: DecimalCast,
{
    fn percentile_from_ranks(
        len: usize,
        percentile: f64,
        descending: bool,
        data_type: &DataType,
        ranks: impl FnOnce(usize, usize) -> (T::Native, T::Native),
    ) -> Result<T::Native> {
        let precision = match data_type {
            DataType::Decimal32(p, _)
            | DataType::Decimal64(p, _)
            | DataType::Decimal128(p, _)
            | DataType::Decimal256(p, _) => *p,
            dt => return internal_err!("percentile_cont: {dt} is not a decimal"),
        };
        let last = len - 1;
        // DuckDB's binary64 rank in the requested order; p <= 1 keeps it
        // within [0, n - 1].
        let rank = percentile * (last as f64);
        let floor = rank.floor() as usize;
        let ceil = rank.ceil() as usize;
        if ceil > last {
            return internal_err!("percentile_cont rank {rank} out of range");
        }
        // Ascending positions of the values at FRN and CRN in the sort order.
        let (lower, upper) = if descending {
            (last - ceil, last - floor)
        } else {
            (floor, ceil)
        };
        let (lower_value, upper_value) = ranks(lower, upper);
        if floor == ceil {
            return Ok(lower_value);
        }
        let (first, second) = if descending {
            (upper_value, lower_value)
        } else {
            (lower_value, upper_value)
        };
        let fraction = rank - floor as f64;
        if precision <= 38 {
            let narrow = |v: T::Native| v.to_i128().ok_or_else(decimal_overflow);
            let value = interpolate_hugeint(
                narrow(first)?,
                narrow(second)?,
                fraction,
                precision <= 18,
            )?;
            T::Native::from_decimal(value).ok_or_else(decimal_overflow)
        } else {
            let wide = |v: T::Native| v.to_i256().ok_or_else(decimal_overflow);
            let value = interpolate_wide(wide(first)?, wide(second)?, fraction)?;
            T::Native::from_decimal(value).ok_or_else(decimal_overflow)
        }
    }
}

/// Calculate the percentile value for a given set of values.
/// This function selects the exact ranks from an unsorted buffer.
///
/// The percentile is calculated using linear interpolation between closest ranks.
/// For percentile p and n values:
/// - If p * (n-1) is an integer, return the value at that position
/// - Otherwise, interpolate between the two closest values
///
/// `percentile` is the requested fraction before the flip for
/// `WITHIN GROUP (ORDER BY ... DESC)`, which `descending` requests.
///
/// Note: This function takes a mutable slice and sorts it in place, but does not
/// consume the data. This is important for window frame queries where evaluate()
/// may be called multiple times on the same accumulator state.
fn calculate_percentile<T: ArrowNumericType, I: PercentileInterpolator<T>>(
    values: &mut [T::Native],
    percentile: f64,
    descending: bool,
    data_type: &DataType,
) -> Result<Option<T::Native>> {
    match values.len() {
        0 => Ok(None),
        1 => Ok(Some(values[0])),
        _ => I::percentile_of(values, percentile, descending, data_type).map(Some),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::PercentileParamState;
    use arrow::array::{Decimal128Array, Float64Array};
    use arrow::datatypes::{
        Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type, Float16Type,
        Float64Type,
    };
    use half::f16;

    fn create_accumulator() -> PercentileContAccumulator<Float64Type, FloatInterpolator> {
        PercentileContAccumulator::<Float64Type, FloatInterpolator>::new(
            PercentileParam {
                aggregate_fn_name: "PERCENTILE_CONT".to_string(),
                state: PercentileParamState::Resolved(0.5),
                is_desc: false,
            },
            DataType::Float64,
        )
    }

    fn float_percentile<T>(values: &mut [T::Native], percentile: f64) -> Option<T::Native>
    where
        T: ArrowNumericType,
        T::Native: AsPrimitive<f64>,
        f64: AsPrimitive<T::Native>,
        ScalarValue: From<Option<T::Native>>,
    {
        calculate_percentile::<T, FloatInterpolator>(
            values,
            percentile,
            false,
            &T::DATA_TYPE,
        )
        .expect("float interpolation is infallible")
    }

    #[test]
    fn interpolation_retains_full_fraction_and_weighted_endpoints() {
        // 0.9 * 7 = 6.3: a six-decimal weight gives 6_299_999.
        let mut values = (0..8)
            .map(|n| f64::from(n) * 1_000_000.0)
            .collect::<Vec<_>>();
        assert_eq!(
            float_percentile::<Float64Type>(&mut values, 0.9),
            Some(6_300_000.0)
        );
        // Weighted endpoints do not overflow in (upper - lower).
        let mut values = [-f64::MAX, f64::MAX];
        assert_eq!(float_percentile::<Float64Type>(&mut values, 0.5), Some(0.0));
        // The median (fraction 0.5) of two values is their correctly rounded
        // midpoint, as DuckDB's DOUBLE quantile_cont and median return.
        let mut values = [f64::MAX, f64::MAX];
        assert_eq!(
            float_percentile::<Float64Type>(&mut values, 0.5),
            Some(f64::MAX)
        );
        let mut values = [f64::from_bits(1); 2];
        assert_eq!(float_percentile::<Float64Type>(&mut values, 0.5), Some(0.0));
        assert_eq!(
            float_percentile::<Float64Type>(&mut values, 0.0),
            Some(f64::from_bits(1))
        );
        assert_eq!(float_percentile::<Float64Type>(&mut [], 0.5), None);
    }

    #[test]
    fn float32_keeps_widened_delta_interpolation() {
        // DuckDB's FLOAT overload uses delta arithmetic: identical infinite
        // endpoints interpolate to NaN.
        let mut values = [f32::INFINITY; 2];
        assert!(
            float_percentile::<Float32Type>(&mut values, 0.5)
                .unwrap()
                .is_nan()
        );
        let mut values = (0u8..8)
            .map(|n| f32::from(n) * 1_000_000.0)
            .collect::<Vec<_>>();
        assert_eq!(
            float_percentile::<Float32Type>(&mut values, 0.9),
            Some(6_300_000.0)
        );
    }

    #[test]
    fn blocked_frame_agrees_with_selection() -> Result<()> {
        let values: ArrayRef = Arc::new(Float64Array::from(
            (0..5000)
                .map(|i| (i % 19 != 0).then_some(((i * 1031) % 2017) as f64))
                .collect::<Vec<_>>(),
        ));
        for percentile in [0.0, 0.13, 0.5, 0.91, 1.0] {
            let mut sliding = create_accumulator();
            sliding.percentile.state = PercentileParamState::Resolved(percentile);
            sliding.update_batch(&[values.slice(0, 1700)])?;
            for start in 1..=3300 {
                sliding.retract_batch(&[values.slice(start - 1, 1)])?;
                sliding.update_batch(&[values.slice(start + 1699, 1)])?;
                let mut reference = create_accumulator();
                reference.percentile.state = PercentileParamState::Resolved(percentile);
                reference.update_batch(&[values.slice(start, 1700)])?;
                assert_eq!(sliding.evaluate()?, reference.evaluate()?, "row {start}");
            }
            sliding.retract_batch(&[values.slice(3300, 1700)])?;
            assert_eq!(sliding.evaluate()?, ScalarValue::Float64(None));
        }
        Ok(())
    }

    #[test]
    fn single_block_replacement_matches_remove_then_insert() -> Result<()> {
        let initial = vec![f64::NEG_INFINITY, -0.0, 0.0, 1.0, 1.0, f64::NAN];
        for &old in &initial {
            for new in [-2.0, -0.0, 0.0, 1.0, 10.0, f64::INFINITY, f64::NAN] {
                let mut actual = SortedFrame::from_values(initial.clone());
                let mut expected = SortedFrame::from_values(initial.clone());
                assert!(expected.remove(old));
                expected.insert(new)?;
                actual.replace_in_single_block(old, new)?;
                assert!(
                    actual.blocks[0]
                        .iter()
                        .zip(&expected.blocks[0])
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
        Ok(())
    }

    #[test]
    fn blocked_frame_reserves_capacity_before_growth_and_releases_it() -> Result<()> {
        use datafusion_execution::memory_pool::{
            GreedyMemoryPool, MemoryConsumer, MemoryPool,
        };

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32 * 1024));
        let reservation = MemoryConsumer::new("test sliding frame").register(&pool);
        let mut frame = SortedFrame::from_values(Vec::<f64>::new());
        frame.reservation = Some(reservation.new_empty());
        for value in 0..1025 {
            frame.insert(f64::from(value))?;
        }
        let charge = |frame: &SortedFrame<f64>| {
            frame.blocks.capacity() * size_of::<Vec<f64>>()
                + frame.blocks.iter().map(Vec::capacity).sum::<usize>() * size_of::<f64>()
        };
        assert_eq!(pool.reserved(), charge(&frame));
        for value in 0..1025 {
            assert!(frame.remove(f64::from(value)));
        }
        assert_eq!(pool.reserved(), charge(&frame));
        drop(frame);
        assert_eq!(pool.reserved(), 0);

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(7 * 1024));
        let reservation =
            MemoryConsumer::new("test sliding frame refusal").register(&pool);
        let mut frame = SortedFrame::from_values(Vec::<f64>::new());
        frame.reservation = Some(reservation.new_empty());
        for value in 0..512 {
            frame.insert(f64::from(value))?;
        }
        let before = pool.reserved();
        let error = frame.insert(512.0).unwrap_err();
        assert!(matches!(
            error,
            datafusion_common::DataFusionError::ResourcesExhausted(_)
        ));
        assert_eq!(frame.len, 512);
        assert_eq!(pool.reserved(), before);
        drop(frame);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[test]
    fn sliding_distinct_string_agg_is_refused() -> Result<()> {
        use arrow::array::StringArray;
        use arrow::record_batch::RecordBatch;
        use datafusion_expr::{WindowFrame, WindowFrameBound, WindowFrameUnits};
        use datafusion_physical_expr::aggregate::AggregateExprBuilder;
        use datafusion_physical_expr::expressions::{Column, lit};
        use datafusion_physical_expr::window::{SlidingAggregateWindowExpr, WindowExpr};

        let batch = RecordBatch::try_from_iter([(
            "v",
            Arc::new(StringArray::from(vec!["a", "a", "b"])) as ArrayRef,
        )])?;
        let agg = AggregateExprBuilder::new(
            crate::string_agg::string_agg_udaf(),
            vec![Arc::new(Column::new("v", 0)), lit(",")],
        )
        .distinct()
        .schema(batch.schema())
        .alias("string_agg(DISTINCT v, ',')")
        .build()?;
        assert!(agg.create_accumulator().is_ok());
        let expr = SlidingAggregateWindowExpr::new(
            Arc::new(agg),
            &[],
            &[],
            Arc::new(WindowFrame::new_bounds(
                WindowFrameUnits::Rows,
                WindowFrameBound::Preceding(2_u64.into()),
                WindowFrameBound::CurrentRow,
            )),
            None,
        );
        let error = expr.evaluate(&batch).unwrap_err();
        assert_eq!(
            error.message(),
            "Aggregate can not be used as a sliding accumulator because `retract_batch` is not implemented: string_agg(DISTINCT v, ',')"
        );
        Ok(())
    }

    #[test]
    fn rolling_median_range_gaps_match_duckdb() -> Result<()> {
        use arrow::array::Int64Array;
        use arrow::record_batch::RecordBatch;
        use datafusion_expr::{WindowFrame, WindowFrameBound, WindowFrameUnits};
        use datafusion_physical_expr::PhysicalSortExpr;
        use datafusion_physical_expr::aggregate::AggregateExprBuilder;
        use datafusion_physical_expr::expressions::Column;
        use datafusion_physical_expr::window::{SlidingAggregateWindowExpr, WindowExpr};

        let batch = RecordBatch::try_from_iter([
            (
                "k",
                Arc::new(Int64Array::from(vec![0, 1, 1, 2, 5, 6, 8, 9, 10])) as ArrayRef,
            ),
            (
                "v",
                Arc::new(Float64Array::from(vec![
                    None,
                    Some(2.0),
                    Some(2.0),
                    Some(10.0),
                    Some(4.0),
                    None,
                    Some(8.0),
                    Some(12.0),
                    Some(6.0),
                ])) as ArrayRef,
            ),
        ])?;
        let agg = AggregateExprBuilder::new(
            crate::median::median_udaf(),
            vec![Arc::new(Column::new("v", 1))],
        )
        .schema(batch.schema())
        .alias("median")
        .build()?;
        let expr = SlidingAggregateWindowExpr::new(
            Arc::new(agg),
            &[],
            &[PhysicalSortExpr {
                expr: Arc::new(Column::new("k", 0)),
                options: Default::default(),
            }],
            Arc::new(WindowFrame::new_bounds(
                WindowFrameUnits::Range,
                WindowFrameBound::Preceding(ScalarValue::Int64(Some(2))),
                WindowFrameBound::Preceding(ScalarValue::Int64(Some(1))),
            )),
            None,
        );
        // DuckDB 2.0.0: median(v) OVER (ORDER BY k RANGE BETWEEN
        // 2 PRECEDING AND 1 PRECEDING) on the same ordered rows.
        let expected = Float64Array::from(vec![
            None,
            None,
            None,
            Some(2.0),
            None,
            Some(4.0),
            None,
            Some(8.0),
            Some(10.0),
        ]);
        let actual = expr.evaluate(&batch)?;
        assert_eq!(actual.as_primitive::<Float64Type>(), &expected);
        Ok(())
    }

    #[test]
    fn retract_batch_errors_on_untracked_value() {
        let mut acc = create_accumulator();
        let values: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        acc.update_batch(std::slice::from_ref(&values)).unwrap();

        let retract: ArrayRef = Arc::new(Float64Array::from(vec![3.0]));
        let err = acc
            .retract_batch(std::slice::from_ref(&retract))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("not present in the window"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn update_batch_with_and_without_nulls_agree() {
        // The null-free fast path must accumulate the same values as the
        // general path.
        let dense: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
        let sparse: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.0),
            None,
            Some(2.0),
            None,
            Some(3.0),
        ]));

        let mut dense_acc = create_accumulator();
        dense_acc
            .update_batch(std::slice::from_ref(&dense))
            .unwrap();
        let mut sparse_acc = create_accumulator();
        sparse_acc
            .update_batch(std::slice::from_ref(&sparse))
            .unwrap();

        assert_eq!(dense_acc.all_values, sparse_acc.all_values);
    }

    #[test]
    fn f16_interpolation_does_not_overflow_to_nan() {
        // Regression test for https://github.com/apache/datafusion/issues/18945
        // Interpolating between 0 and the max finite f16 value previously overflowed
        // intermediate f16 computations and produced NaN.
        let mut values = vec![f16::from_f32(0.0), f16::from_f32(65504.0)];
        let result = calculate_percentile::<Float16Type, FloatInterpolator>(
            &mut values,
            0.5,
            false,
            &DataType::Float16,
        )
        .expect("non-empty input")
        .expect("non-empty result");
        let result_f = result.to_f32();
        assert!(
            !result_f.is_nan(),
            "expected non-NaN result, got {result_f}"
        );
        // 0.5 percentile should be close to midpoint
        assert!(
            (result_f - 32752.0).abs() < 1.0,
            "unexpected result {result_f}"
        );
    }

    fn decimal64_percentile(values: &mut [i64], percentile: f64) -> i64 {
        calculate_percentile::<Decimal64Type, DecimalInterpolator>(
            values,
            percentile,
            false,
            &DataType::Decimal64(18, 2),
        )
        .expect("evaluate failed")
        .expect("expected Some value")
    }

    /// The percentile of unscaled `values` in `data_type`, as i128.
    fn decimal_percentile<T>(
        values: &[i128],
        percentile: f64,
        descending: bool,
        data_type: &DataType,
    ) -> i128
    where
        T: DecimalType,
        T::Native: DecimalCast,
    {
        let mut values = values
            .iter()
            .map(|&v| T::Native::from_decimal(v).expect("value fits the type"))
            .collect::<Vec<_>>();
        calculate_percentile::<T, DecimalInterpolator>(
            &mut values,
            percentile,
            descending,
            data_type,
        )
        .expect("evaluate failed")
        .expect("expected Some value")
        .to_i128()
        .expect("result fits i128")
    }

    fn decimal256_percentile(
        values: &[i256],
        percentile: f64,
        data_type: &DataType,
    ) -> i256 {
        calculate_percentile::<Decimal256Type, DecimalInterpolator>(
            &mut values.to_vec(),
            percentile,
            false,
            data_type,
        )
        .expect("evaluate failed")
        .expect("expected Some value")
    }

    fn p128(values: &[i128], percentile: f64, descending: bool, precision: u8) -> i128 {
        decimal_percentile::<Decimal128Type>(
            values,
            percentile,
            descending,
            &DataType::Decimal128(precision, 0),
        )
    }

    #[test]
    fn decimal_matches_duckdb_binary64_interpolation() {
        // Every expectation is a DuckDB 2.0.0 result (unscaled coefficients).
        // 0, 1000000, ..., 7000000 as DECIMAL(p, 2): p0.9 is 6300000.00 for
        // precision <= 18 and 6299999.99 for precision 19..=38. A weight
        // truncated to six decimals gave 6299999.00.
        let values = (0..8).map(|n| n * 100_000_000).collect::<Vec<i128>>();
        let narrow = [
            decimal_percentile::<Decimal32Type>(
                &values,
                0.9,
                false,
                &DataType::Decimal32(9, 2),
            ),
            decimal_percentile::<Decimal64Type>(
                &values,
                0.9,
                false,
                &DataType::Decimal64(18, 2),
            ),
            decimal_percentile::<Decimal128Type>(
                &values,
                0.9,
                false,
                &DataType::Decimal128(18, 2),
            ),
            decimal_percentile::<Decimal256Type>(
                &values,
                0.9,
                false,
                &DataType::Decimal256(18, 2),
            ),
        ];
        assert_eq!(narrow, [630_000_000; 4]);
        for data_type in [DataType::Decimal128(19, 2), DataType::Decimal128(38, 2)] {
            assert_eq!(
                decimal_percentile::<Decimal128Type>(&values, 0.9, false, &data_type),
                629_999_999,
                "{data_type}"
            );
        }
        assert_eq!(
            decimal_percentile::<Decimal256Type>(
                &values,
                0.9,
                false,
                &DataType::Decimal256(38, 2)
            ),
            629_999_999
        );
        // DESC keeps p: percentile_cont(0.1) WITHIN GROUP (ORDER BY x DESC)
        // is 6300000.00 at DECIMAL(38, 2), unlike p0.9 ascending.
        assert_eq!(
            decimal_percentile::<Decimal128Type>(
                &values,
                0.1,
                true,
                &DataType::Decimal128(38, 2)
            ),
            630_000_000
        );
        // The pinned binary's fused multiply-add (separate rounding gives
        // other last digits).
        assert_eq!(
            p128(
                &[-283_110_063_312_371_717, -168_985_206_005_547_813],
                0.2,
                false,
                18
            ),
            -260_285_091_851_006_944
        );
        assert_eq!(
            p128(
                &[
                    -96_061_439_535_860_669_001_790_758_262_438_174_069,
                    85_251_732_078_307_683_191_308_606_285_311_392_193
                ],
                0.3,
                false,
                38
            ),
            -41_667_488_051_610_157_332_345_492_003_413_295_104
        );
        // DESC of the same narrow pair: p0.8 DESC.
        assert_eq!(
            p128(
                &[-283_110_063_312_371_717, -168_985_206_005_547_813],
                0.8,
                true,
                18
            ),
            -260_285_091_851_006_944
        );
        // Binary64 rounding can exceed the declared precision.
        assert_eq!(
            decimal_percentile::<Decimal64Type>(
                &[999_999_999_999_999_999; 2],
                0.5,
                false,
                &DataType::Decimal64(18, 0)
            ),
            1_000_000_000_000_000_000
        );
        assert_eq!(
            p128(&[9_999_999_999_999_999_999; 2], 0.5, false, 19),
            9_999_999_999_999_999_999
        );
        // Opposite DECIMAL(38, 0) extremes take the weighted fallback.
        let max = 99_999_999_999_999_999_999_999_999_999_999_999_999i128;
        assert_eq!(p128(&[-max, max], 0.5, false, 38), 0);
        assert_eq!(
            p128(&[-max, max], 0.1, false, 38),
            -79_999_999_999_999_996_310_101_265_616_969_138_176
        );
        assert_eq!(
            p128(&[-max, max], 0.9, false, 38),
            79_999_999_999_999_996_310_101_265_616_969_138_176
        );
        assert_eq!(p128(&[-max, max], 0.0, false, 38), -max);
    }

    #[test]
    fn decimal_rounding_follows_duckdb_width_classes() {
        // DuckDB 2.0.0 results. Precision <= 18 truncates the interpolated
        // value toward zero; precision > 18 truncates the offset from the
        // value at FRN in the sort order.
        // median(-10, -9) = -9.5
        assert_eq!(p128(&[-10, -9], 0.5, false, 18), -9);
        assert_eq!(p128(&[-10, -9], 0.5, false, 38), -10);
        // ... DESC rounds toward the larger value at precision > 18.
        assert_eq!(p128(&[-10, -9], 0.5, true, 38), -9);
        assert_eq!(p128(&[-101, -100], 0.5, false, 19), -101);
        assert_eq!(p128(&[-101, -100], 0.5, true, 19), -100);
        // quantile_cont(-1, 2; 0.25) = -0.25
        assert_eq!(p128(&[-1, 2], 0.25, false, 18), 0);
        assert_eq!(p128(&[-1, 2], 0.25, false, 38), -1);
        // percentile_cont(p) WITHIN GROUP (ORDER BY x DESC) of 9, 10.
        assert_eq!(p128(&[9, 10], 0.25, true, 18), 9);
        assert_eq!(p128(&[9, 10], 0.25, true, 38), 10);
        assert_eq!(p128(&[9, 10], 0.5, true, 38), 10);
        // quantile_cont(-1.0001, -2.0003; 0.37) = -1.630174 (scale 4), and
        // DESC -1.370174.
        let values = [-10_001, -20_003];
        assert_eq!(p128(&values, 0.37, false, 18), -16_302);
        assert_eq!(p128(&values, 0.37, false, 38), -16_303);
        assert_eq!(p128(&values, 0.37, true, 18), -13_701);
        assert_eq!(p128(&values, 0.37, true, 38), -13_701);
        // The class follows the precision, not the storage width.
        assert_eq!(
            decimal_percentile::<Decimal32Type>(
                &[-10, -9],
                0.5,
                false,
                &DataType::Decimal32(9, 0)
            ),
            -9
        );
        assert_eq!(
            decimal_percentile::<Decimal256Type>(
                &[-10, -9],
                0.5,
                false,
                &DataType::Decimal256(18, 0)
            ),
            -9
        );
        // Decimal256 precision > 38 keeps the HUGEINT rule (no DuckDB type).
        assert_eq!(
            decimal_percentile::<Decimal256Type>(
                &[-10, -9],
                0.5,
                false,
                &DataType::Decimal256(76, 0)
            ),
            -10
        );
        // Exact ranks return the exact value.
        assert_eq!(p128(&[3, -7, 5], 0.0, false, 18), -7);
        assert_eq!(p128(&[3, -7, 5], 1.0, false, 18), 5);
        assert_eq!(p128(&[3, -7, 5], 0.0, true, 38), 5);
        assert_eq!(p128(&[3, -7, 5], 0.5, true, 38), 3);
    }

    #[test]
    fn decimal_overflow_is_checked() {
        let bound = 2f64.powi(127);
        assert!(hugeint_from_double(bound).is_err());
        assert!(hugeint_from_double(-bound).is_err());
        assert!(hugeint_from_double(f64::NAN).is_err());
        assert!(hugeint_from_double(f64::INFINITY).is_err());
        assert_eq!(hugeint_from_double(-2.75).unwrap(), -2);
        // DuckDB's two-limb conversion of -1 and of limb boundaries.
        assert_eq!(hugeint_to_double(-1), -1.0);
        assert_eq!(hugeint_to_double(i128::MIN), -bound);
        assert_eq!(hugeint_to_double(1i128 << 64), 2f64.powi(64));
        // Native extremes interpolate without an overflow error.
        assert_eq!(
            decimal_percentile::<Decimal32Type>(
                &[i32::MIN.into(), i32::MAX.into()],
                0.5,
                false,
                &DataType::Decimal32(9, 0)
            ),
            0
        );
        assert_eq!(
            decimal_percentile::<Decimal64Type>(
                &[i64::MIN.into(), i64::MAX.into()],
                0.5,
                false,
                &DataType::Decimal64(18, 0)
            ),
            0
        );
        assert_eq!(p128(&[i128::MIN, i128::MAX], 0.5, false, 38), 0);
        // Decimal256 precision > 38: 256-bit differences and fallback.
        let two = i256::from_i128(2);
        assert_eq!(
            decimal256_percentile(
                &[i256::ZERO, two.wrapping_pow(200)],
                0.5,
                &DataType::Decimal256(76, 0)
            ),
            two.wrapping_pow(199)
        );
        assert_eq!(
            decimal256_percentile(
                &[i256::MIN, i256::MAX],
                0.5,
                &DataType::Decimal256(76, 0)
            ),
            i256::ZERO
        );
        // Decimal256 values beyond i128 at precision <= 38 are an error.
        let err = calculate_percentile::<Decimal256Type, DecimalInterpolator>(
            &mut [i256::MIN, i256::MAX],
            0.5,
            false,
            &DataType::Decimal256(38, 0),
        )
        .unwrap_err();
        assert!(err.to_string().contains("overflow"), "{err}");
    }

    fn decimal_param(percentile: f64, is_desc: bool) -> PercentileParam {
        PercentileParam {
            aggregate_fn_name: "PERCENTILE_CONT".to_string(),
            state: PercentileParamState::Resolved(percentile),
            is_desc,
        }
    }

    #[test]
    fn decimal_accumulators_match_duckdb() -> Result<()> {
        // 0, NULL, 1000000.00, ..., 7000000.00 as DECIMAL(18, 2).
        let data_type = DataType::Decimal128(18, 2);
        let rows = std::iter::once(Some(0))
            .chain(std::iter::once(None))
            .chain((1..8).map(|n| Some(n * 100_000_000)))
            .collect::<Vec<_>>();
        let values: ArrayRef = Arc::new(
            Decimal128Array::from(rows.clone()).with_data_type(data_type.clone()),
        );
        let expected = ScalarValue::Decimal128(Some(630_000_000), 18, 2);

        for (percentile, is_desc) in [(0.9, false), (0.1, true)] {
            let mut acc =
                PercentileContAccumulator::<Decimal128Type, DecimalInterpolator>::new(
                    decimal_param(percentile, is_desc),
                    data_type.clone(),
                );
            acc.update_batch(std::slice::from_ref(&values))?;
            assert_eq!(acc.evaluate()?, expected, "p{percentile} desc={is_desc}");
        }

        // DISTINCT ignores a duplicate.
        let duplicated: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(0), Some(700_000_000)])
                .with_data_type(data_type.clone()),
        );
        let mut acc = DistinctPercentileContAccumulator::<
            Decimal128Type,
            DecimalInterpolator,
        >::new(decimal_param(0.9, false), data_type.clone());
        acc.update_batch(std::slice::from_ref(&values))?;
        acc.update_batch(std::slice::from_ref(&duplicated))?;
        assert_eq!(acc.evaluate()?, expected);
        // A sliding frame: retract 0, add 8000000.00 -> p0.9 = 7300000.00.
        let added: ArrayRef = Arc::new(
            Decimal128Array::from(vec![800_000_000]).with_data_type(data_type.clone()),
        );
        let retracted: ArrayRef =
            Arc::new(Decimal128Array::from(vec![0, 0]).with_data_type(data_type.clone()));
        acc.update_batch(std::slice::from_ref(&added))?;
        acc.retract_batch(std::slice::from_ref(&retracted))?;
        assert_eq!(
            acc.evaluate()?,
            ScalarValue::Decimal128(Some(730_000_000), 18, 2)
        );

        let mut acc =
            PercentileContAccumulator::<Decimal128Type, DecimalInterpolator>::new(
                decimal_param(0.9, false),
                data_type.clone(),
            );
        acc.update_batch(std::slice::from_ref(&values))?;
        acc.update_batch(std::slice::from_ref(&added))?;
        acc.retract_batch(&[Arc::new(
            Decimal128Array::from(vec![0]).with_data_type(data_type.clone()),
        )])?;
        assert_eq!(
            acc.evaluate()?,
            ScalarValue::Decimal128(Some(730_000_000), 18, 2)
        );

        // Grouped: group 1 holds only the NULL row.
        let mut acc = PercentileContGroupsAccumulator::<
            Decimal128Type,
            DecimalInterpolator,
        >::new(decimal_param(0.9, false), data_type.clone());
        let groups = rows
            .iter()
            .map(|row| usize::from(row.is_none()))
            .collect::<Vec<_>>();
        acc.update_batch(std::slice::from_ref(&values), &groups, None, 2)?;
        let result = acc.evaluate(EmitTo::All)?;
        let result = result.as_primitive::<Decimal128Type>();
        assert_eq!(result.value(0), 630_000_000);
        assert!(result.is_null(1));
        Ok(())
    }

    #[test]
    fn percentile_cont_decimal64() {
        // Test values: [100.00, 200.00, 300.00, 400.00, 500.00]
        // These are stored as i64 values scaled by 10^2
        let mut values = vec![
            10000i64, // 100.00
            20000i64, // 200.00
            30000i64, // 300.00
            40000i64, // 400.00
            50000i64, // 500.00
        ];

        // Test 50th percentile (median)
        // Should return 300.00 (30000)
        let result = decimal64_percentile(&mut values, 0.5);

        assert_eq!(result, 30000i64, "50th percentile should be 300.00");

        // Test 15th percentile
        // Should return 160.00 (16000)
        let result = decimal64_percentile(&mut values, 0.15);

        assert_eq!(result, 16000i64, "15th percentile should be 160.00");

        // Test 0th percentile (minimum)
        let mut values = vec![10000i64, 20000i64, 30000i64];
        let result = decimal64_percentile(&mut values, 0.0);

        assert_eq!(
            result, 10000i64,
            "0th percentile should be minimum value 100.00"
        );

        // Test 100th percentile (maximum)
        let mut values = vec![10000i64, 20000i64, 30000i64];
        let result = decimal64_percentile(&mut values, 1.0);

        assert_eq!(
            result, 30000i64,
            "100th percentile should be maximum value 300.00"
        );
    }

    #[test]
    fn percentile_cont_decimal128_subtraction_overflow() {
        // Case for interpolation overflow (upper - lower cannot fit i128),
        // where `interpolate` takes the second branch
        let boundary = 100_000_000_000_000_000_000_000_000_000_000_000_000i128;
        let lower = -boundary;
        let upper = boundary;
        assert!(
            upper.checked_sub(lower).is_none(),
            "test premise: must overflow"
        );

        let mut values = vec![lower, upper];
        let result = calculate_percentile::<Decimal128Type, DecimalInterpolator>(
            &mut values,
            0.25,
            false,
            &DataType::Decimal128(38, 0),
        )
        .expect("evaluate failed")
        .expect("expected Some value");

        // DuckDB 2.0.0's weighted binary64 fallback for an overflowing i128
        // difference: fma(lower, 0.75, upper * 0.25) of the double values.
        assert_eq!(
            result, -49_999_999_999_999_998_874_404_911_728_017_014_784,
            "interpolation should use the weighted fallback without overflowing"
        );
    }
}
