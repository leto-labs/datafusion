// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Defines `SUM` and `SUM DISTINCT` aggregate accumulators

use crate::sum_widening::{WideningSumAccumulator, WideningSumGroupsAccumulator};
use arrow::array::{Array, ArrayRef, ArrowNativeTypeOp, ArrowNumericType, AsArray};
use arrow::datatypes::Field;
use arrow::datatypes::{
    ArrowNativeType, DECIMAL32_MAX_PRECISION, DECIMAL64_MAX_PRECISION,
    DECIMAL128_MAX_PRECISION, DECIMAL256_MAX_PRECISION, DataType, Decimal32Type,
    Decimal64Type, Decimal128Type, Decimal256Type, DurationMicrosecondType,
    DurationMillisecondType, DurationNanosecondType, DurationSecondType, FieldRef,
    Float64Type, Int64Type, IntervalDayTimeType, IntervalMonthDayNanoType, IntervalUnit,
    IntervalYearMonthType, TimeUnit, UInt64Type,
};
use datafusion_common::hash_utils::RandomState;
use datafusion_common::internal_err;
use datafusion_common::stats::Precision;
use datafusion_common::types::{
    NativeType, logical_float64, logical_int8, logical_int16, logical_int32,
    logical_int64, logical_uint8, logical_uint16, logical_uint32, logical_uint64,
};
use datafusion_common::{HashMap, Result, ScalarValue, exec_err, not_impl_err};
use datafusion_expr::expr::AggregateFunction;
use datafusion_expr::expr_fn::cast;
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::{AggregateOrderSensitivity, format_state_name};
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Coercion, Documentation, Expr, GroupsAccumulator,
    Operator, ReversedUDAF, SetMonotonicity, Signature, StatisticsArgs, TypeSignature,
    TypeSignatureClass, Volatility,
};
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::prim_op::PrimitiveGroupsAccumulator;
use datafusion_functions_aggregate_common::aggregate::sum_distinct::DistinctSumAccumulator;
use datafusion_macros::user_doc;
use datafusion_physical_expr::expressions::{CastExpr, Column};
use std::mem::{size_of, size_of_val};

make_udaf_expr_and_func!(
    Sum,
    sum,
    expression,
    "Returns the sum of a group of values.",
    sum_udaf
);

pub fn sum_distinct(expr: Expr) -> Expr {
    Expr::AggregateFunction(AggregateFunction::new_udf(
        sum_udaf(),
        vec![expr],
        true,
        None,
        vec![],
        None,
    ))
}

/// Sum only supports a subset of numeric types, instead relying on type coercion
///
/// This macro is similar to [downcast_primitive](arrow::array::downcast_primitive)
///
/// `args` is [AccumulatorArgs]
/// `helper` is a macro accepting (ArrowPrimitiveType, DataType)
macro_rules! downcast_sum {
    ($args:ident, $helper:ident) => {
        match $args.return_field.data_type().clone() {
            DataType::UInt64 => {
                $helper!(UInt64Type, $args.return_field.data_type().clone())
            }
            DataType::Int64 => {
                $helper!(Int64Type, $args.return_field.data_type().clone())
            }
            DataType::Float64 => {
                $helper!(Float64Type, $args.return_field.data_type().clone())
            }
            DataType::Decimal32(_, _) => {
                $helper!(Decimal32Type, $args.return_field.data_type().clone())
            }
            DataType::Decimal64(_, _) => {
                $helper!(Decimal64Type, $args.return_field.data_type().clone())
            }
            DataType::Decimal128(_, _) => {
                $helper!(Decimal128Type, $args.return_field.data_type().clone())
            }
            DataType::Decimal256(_, _) => {
                $helper!(Decimal256Type, $args.return_field.data_type().clone())
            }
            DataType::Duration(TimeUnit::Second) => {
                $helper!(DurationSecondType, $args.return_field.data_type().clone())
            }
            DataType::Duration(TimeUnit::Millisecond) => {
                $helper!(
                    DurationMillisecondType,
                    $args.return_field.data_type().clone()
                )
            }
            DataType::Duration(TimeUnit::Microsecond) => {
                $helper!(
                    DurationMicrosecondType,
                    $args.return_field.data_type().clone()
                )
            }
            DataType::Duration(TimeUnit::Nanosecond) => {
                $helper!(
                    DurationNanosecondType,
                    $args.return_field.data_type().clone()
                )
            }
            DataType::Interval(IntervalUnit::YearMonth) => {
                $helper!(
                    IntervalYearMonthType,
                    $args.return_field.data_type().clone()
                )
            }
            DataType::Interval(IntervalUnit::DayTime) => {
                $helper!(IntervalDayTimeType, $args.return_field.data_type().clone())
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                $helper!(
                    IntervalMonthDayNanoType,
                    $args.return_field.data_type().clone()
                )
            }
            _ => {
                not_impl_err!(
                    "Sum not supported for {}: {}",
                    $args.name,
                    $args.return_field.data_type()
                )
            }
        }
    };
}

#[user_doc(
    doc_section(label = "General Functions"),
    description = "Returns the sum of all values in the specified column.",
    syntax_example = "sum(expression)",
    sql_example = r#"```sql
> SELECT sum(column_name) FROM table_name;
+-----------------------+
| sum(column_name)      |
+-----------------------+
| 12345                 |
+-----------------------+
```"#,
    standard_argument(name = "expression",)
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Sum {
    signature: Signature,
    /// See [`Self::with_logical_input`].
    logical_input: Option<DataType>,
}

impl Sum {
    /// A `sum` whose argument is the narrow storage of `logical` values:
    /// `Decimal32` or `Decimal64` arrays that hold the unscaled values of a
    /// `Decimal128` column of type `logical`, at the scale of `logical`.
    ///
    /// The state, the result type and the result value are those of `sum` over
    /// the same values stored as `logical`. Each value is widened as it is
    /// added, so a sum that leaves the narrow type does not wrap. Only the
    /// accumulators of an aggregation are implemented; a sliding window frame
    /// or `DISTINCT` is an error.
    pub fn with_logical_input(logical: DataType) -> Self {
        Self {
            logical_input: Some(logical),
            ..Self::new()
        }
    }

    /// The type set by [`Self::with_logical_input`].
    pub fn logical_input(&self) -> Option<&DataType> {
        self.logical_input.as_ref()
    }

    /// The accumulator of [`Self::with_logical_input`] for the narrow input
    /// type of `args`.
    fn widening<A>(
        &self,
        args: &AccumulatorArgs,
        build: impl Fn(&DataType, DataType) -> Result<A>,
    ) -> Result<A> {
        let input = args.expr_fields[0].data_type();
        let sum = args.return_field.data_type();
        match (self.logical_input.as_ref(), input, sum) {
            (
                Some(DataType::Decimal128(_, logical_scale)),
                DataType::Decimal32(_, scale) | DataType::Decimal64(_, scale),
                DataType::Decimal128(..),
            ) if scale == logical_scale && !args.is_distinct => build(input, sum.clone()),
            (logical, input, sum) => internal_err!(
                "sum of {input} cannot accumulate {logical:?} values as {sum}"
            ),
        }
    }

    pub fn new() -> Self {
        Self {
            logical_input: None,
            // Refer to https://www.postgresql.org/docs/8.2/functions-aggregate.html doc
            // smallint, int, bigint, real, double precision, decimal, or interval.
            signature: Signature::one_of(
                vec![
                    TypeSignature::Coercible(vec![Coercion::new_exact(
                        TypeSignatureClass::Decimal,
                    )]),
                    // Unsigned to u64
                    TypeSignature::Coercible(vec![Coercion::new_implicit(
                        TypeSignatureClass::Native(logical_uint64()),
                        vec![
                            TypeSignatureClass::Native(logical_uint8()),
                            TypeSignatureClass::Native(logical_uint16()),
                            TypeSignatureClass::Native(logical_uint32()),
                        ],
                        NativeType::UInt64,
                    )]),
                    // Signed to i64
                    TypeSignature::Coercible(vec![Coercion::new_implicit(
                        TypeSignatureClass::Native(logical_int64()),
                        vec![
                            TypeSignatureClass::Native(logical_int8()),
                            TypeSignatureClass::Native(logical_int16()),
                            TypeSignatureClass::Native(logical_int32()),
                        ],
                        NativeType::Int64,
                    )]),
                    // Floats to f64
                    TypeSignature::Coercible(vec![Coercion::new_implicit(
                        TypeSignatureClass::Native(logical_float64()),
                        vec![TypeSignatureClass::Float],
                        NativeType::Float64,
                    )]),
                    TypeSignature::Coercible(vec![Coercion::new_exact(
                        TypeSignatureClass::Duration,
                    )]),
                    TypeSignature::Coercible(vec![Coercion::new_exact(
                        TypeSignatureClass::Interval,
                    )]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for Sum {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregateUDFImpl for Sum {
    fn name(&self) -> &str {
        "sum"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match self.logical_input.as_ref().unwrap_or(&arg_types[0]) {
            DataType::Int64 => Ok(DataType::Int64),
            DataType::UInt64 => Ok(DataType::UInt64),
            DataType::Float64 => Ok(DataType::Float64),
            // In the spark, the result type is DECIMAL(min(38,precision+10), s)
            // ref: https://github.com/apache/spark/blob/fcf636d9eb8d645c24be3db2d599aba2d7e2955a/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/Sum.scala#L66
            DataType::Decimal32(precision, scale) => {
                let new_precision = DECIMAL32_MAX_PRECISION.min(*precision + 10);
                Ok(DataType::Decimal32(new_precision, *scale))
            }
            DataType::Decimal64(precision, scale) => {
                let new_precision = DECIMAL64_MAX_PRECISION.min(*precision + 10);
                Ok(DataType::Decimal64(new_precision, *scale))
            }
            DataType::Decimal128(precision, scale) => {
                let new_precision = DECIMAL128_MAX_PRECISION.min(*precision + 10);
                Ok(DataType::Decimal128(new_precision, *scale))
            }
            DataType::Decimal256(precision, scale) => {
                let new_precision = DECIMAL256_MAX_PRECISION.min(*precision + 10);
                Ok(DataType::Decimal256(new_precision, *scale))
            }
            DataType::Duration(time_unit) => Ok(DataType::Duration(*time_unit)),
            DataType::Interval(interval_unit) => Ok(DataType::Interval(*interval_unit)),
            other => {
                exec_err!("[return_type] SUM not supported for {}", other)
            }
        }
    }

    fn accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        if self.logical_input.is_some() {
            return self.widening(&args, |input, sum| {
                Ok(match input {
                    DataType::Decimal32(..) => {
                        Box::new(WideningSumAccumulator::<Decimal32Type>::new(sum))
                            as Box<dyn Accumulator>
                    }
                    _ => Box::new(WideningSumAccumulator::<Decimal64Type>::new(sum)),
                })
            });
        }
        let is_distinct = args.is_distinct;
        macro_rules! helper {
            ($t:ty, $dt:expr) => {
                if is_distinct {
                    Ok(Box::new(DistinctSumAccumulator::<$t>::new($dt))
                        as Box<dyn Accumulator>)
                } else {
                    Ok(Box::new(SumAccumulator::<$t>::new($dt)) as Box<dyn Accumulator>)
                }
            };
        }
        downcast_sum!(args, helper)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        if args.is_distinct {
            Ok(vec![
                Field::new_list(
                    format_state_name(args.name, "sum distinct"),
                    // See COMMENTS.md to understand why nullable is set to true
                    Field::new_list_field(args.return_type().clone(), true),
                    false,
                )
                .into(),
            ])
        } else {
            Ok(vec![
                Field::new(
                    format_state_name(args.name, "sum"),
                    args.return_type().clone(),
                    true,
                )
                .into(),
            ])
        }
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        !args.is_distinct
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        if self.logical_input.is_some() {
            return self.widening(&args, |input, sum| {
                Ok(match input {
                    DataType::Decimal32(..) => {
                        Box::new(WideningSumGroupsAccumulator::<Decimal32Type>::new(sum))
                            as Box<dyn GroupsAccumulator>
                    }
                    _ => {
                        Box::new(WideningSumGroupsAccumulator::<Decimal64Type>::new(sum))
                    }
                })
            });
        }
        macro_rules! helper {
            ($t:ty, $dt:expr) => {
                Ok(Box::new(PrimitiveGroupsAccumulator::<$t, _>::new(
                    &$dt,
                    |x, y| *x = x.add_wrapping(y),
                )))
            };
        }
        downcast_sum!(args, helper)
    }

    fn create_sliding_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        if self.logical_input.is_some() {
            return not_impl_err!(
                "sum over narrow decimal storage in a sliding window frame"
            );
        }
        if args.is_distinct {
            // distinct path: [`SlidingDistinctSumAccumulator`] only implements
            // Int64, so gate the supported type here rather than dispatching
            // through `downcast_sum!`, which accepts every SUM type
            match args.return_field.data_type() {
                DataType::Int64 => Ok(Box::new(SlidingDistinctSumAccumulator::try_new(
                    &DataType::Int64,
                )?)),
                _ => not_impl_err!(
                    "SUM(DISTINCT) over sliding window frames is only supported for Int64, got {}",
                    args.expr_fields[0].data_type()
                ),
            }
        } else {
            if args.return_type() == &DataType::Float64 {
                return Ok(Box::new(SlidingFloatSumAccumulator::new(false)));
            }
            // non‐distinct path: existing sliding sum
            macro_rules! helper {
                ($t:ty, $dt:expr) => {
                    Ok(Box::new(SlidingSumAccumulator::<$t>::new($dt.clone())))
                };
            }
            downcast_sum!(args, helper)
        }
    }

    fn reverse_expr(&self) -> ReversedUDAF {
        ReversedUDAF::Identical
    }

    fn order_sensitivity(&self) -> AggregateOrderSensitivity {
        AggregateOrderSensitivity::Insensitive
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn set_monotonicity(&self, data_type: &DataType) -> SetMonotonicity {
        // `SUM` is only monotonically increasing when its input is unsigned.
        // TODO: Expand these utilizing statistics.
        match data_type {
            DataType::UInt8 => SetMonotonicity::Increasing,
            DataType::UInt16 => SetMonotonicity::Increasing,
            DataType::UInt32 => SetMonotonicity::Increasing,
            DataType::UInt64 => SetMonotonicity::Increasing,
            _ => SetMonotonicity::NotMonotonic,
        }
    }

    /// Implement ClickBench Q29 specific optimization:
    /// `SUM(arg + constant)` --> `SUM(arg) + constant * COUNT(arg)`
    ///
    /// See background on [`AggregateUDFImpl::simplify_expr_op_literal`]
    fn simplify_expr_op_literal(
        &self,
        agg_function: &AggregateFunction,
        arg: &Expr,
        op: Operator,
        lit: &Expr,
        // Only support '+' so the order of the args doesn't matter
        _arg_is_left: bool,
    ) -> Result<Option<Expr>> {
        if op != Operator::Plus {
            return Ok(None);
        }

        let lit_type = match &lit {
            Expr::Literal(value, _) => value.data_type(),
            _ => {
                return internal_err!(
                    "Sum::simplify_expr_op_literal got a non literal argument"
                );
            }
        };
        if lit_type == DataType::Null {
            return Ok(None);
        }
        // `CAST(COUNT(arg) AS <lit type>)` is not valid for intervals, and for
        // decimals it can overflow or change the SUM result type.
        // See https://github.com/apache/datafusion/issues/25943
        if !(lit_type.is_integer() || lit_type.is_floating()) {
            return Ok(None);
        }

        // Build up SUM(arg)
        let mut sum_agg = agg_function.clone();
        sum_agg.params.args = vec![arg.clone()];
        let sum_agg = Expr::AggregateFunction(sum_agg);

        // COUNT(arg) - cast to the correct type
        let count_agg = cast(crate::count::count(arg.clone()), lit_type);

        // SUM(arg) + lit * COUNT(arg)
        Ok(Some(sum_agg + (lit.clone() * count_agg)))
    }

    fn value_from_stats(&self, statistics_args: &StatisticsArgs) -> Option<ScalarValue> {
        // The statistics of narrow storage are not in the type of the sum.
        if statistics_args.is_distinct || self.logical_input.is_some() {
            return None;
        }

        let [expr] = statistics_args.exprs else {
            return None;
        };

        let (col_expr, cast_type) = match expr.downcast_ref::<Column>() {
            Some(col_expr) => (col_expr, None),
            None => {
                let cast_expr = expr.downcast_ref::<CastExpr>()?;
                let col_expr = cast_expr.expr().downcast_ref::<Column>()?;
                (col_expr, Some(cast_expr.cast_type()))
            }
        };

        let col_stats = statistics_args
            .statistics
            .column_statistics
            .get(col_expr.index())?;

        // Replacing SUM with a literal is only valid for exact statistics.
        // `cast_to_sum_type` also widens small integer stats to the SQL SUM
        // return type, e.g. Int32 statistics become an Int64 SUM value.
        let Precision::Exact(val) = col_stats.sum_value.cast_to_sum_type() else {
            return None;
        };
        if val.is_null() {
            return None;
        }

        // SUM coercion can introduce a physical CAST around the input column
        // (`SUM(Int32)` becomes `SUM(CAST(Int32 AS Int64))`). Only use the
        // column's raw sum stats when the widened stats value matches that
        // cast target and the aggregate return type.
        if let Some(cast_type) = cast_type {
            let value_type = val.data_type();
            if cast_type != statistics_args.return_type || &value_type != cast_type {
                return None;
            }
            return Some(val);
        }

        if &val.data_type() == statistics_args.return_type {
            Some(val)
        } else {
            val.cast_to(statistics_args.return_type).ok()
        }
    }
}

/// This accumulator computes SUM incrementally
struct SumAccumulator<T: ArrowNumericType> {
    sum: Option<T::Native>,
    data_type: DataType,
}

impl<T: ArrowNumericType> std::fmt::Debug for SumAccumulator<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SumAccumulator({})", self.data_type)
    }
}

impl<T: ArrowNumericType> SumAccumulator<T> {
    fn new(data_type: DataType) -> Self {
        Self {
            sum: None,
            data_type,
        }
    }
}

impl<T: ArrowNumericType> Accumulator for SumAccumulator<T> {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = values[0].as_primitive::<T>();
        if let Some(x) = arrow::compute::sum(values) {
            let v = self.sum.get_or_insert_with(|| T::Native::usize_as(0));
            *v = v.add_wrapping(x);
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.update_batch(states)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        ScalarValue::new_primitive::<T>(self.sum, &self.data_type)
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }
}

/// This accumulator incrementally computes sums over a sliding window
///
/// This is separate from [`SumAccumulator`] as requires additional state
struct SlidingSumAccumulator<T: ArrowNumericType> {
    sum: T::Native,
    count: u64,
    data_type: DataType,
}

impl<T: ArrowNumericType> std::fmt::Debug for SlidingSumAccumulator<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SlidingSumAccumulator({})", self.data_type)
    }
}

impl<T: ArrowNumericType> SlidingSumAccumulator<T> {
    fn new(data_type: DataType) -> Self {
        Self {
            sum: T::Native::usize_as(0),
            count: 0,
            data_type,
        }
    }
}

impl<T: ArrowNumericType> Accumulator for SlidingSumAccumulator<T> {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?, self.count.into()])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = values[0].as_primitive::<T>();
        self.count += (values.len() - values.null_count()) as u64;
        if let Some(x) = arrow::compute::sum(values) {
            self.sum = self.sum.add_wrapping(x)
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let values = states[0].as_primitive::<T>();
        if let Some(x) = arrow::compute::sum(values) {
            self.sum = self.sum.add_wrapping(x)
        }
        if let Some(x) = arrow::compute::sum(states[1].as_primitive::<UInt64Type>()) {
            self.count += x;
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let v = (self.count != 0).then_some(self.sum);
        ScalarValue::new_primitive::<T>(v, &self.data_type)
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = values[0].as_primitive::<T>();
        if let Some(x) = arrow::compute::sum(values) {
            self.sum = self.sum.sub_wrapping(x)
        }
        self.count -= (values.len() - values.null_count()) as u64;
        Ok(())
    }

    fn supports_retract_batch(&self) -> bool {
        true
    }
}

// Non-finite inputs cannot participate in reversible arithmetic: NaN - NaN
// and infinity - infinity remain NaN. Keep their multiplicities separately.
#[derive(Debug, Default)]
pub(crate) struct NonFiniteCounts {
    nan: u64,
    positive: u64,
    negative: u64,
}

impl NonFiniteCounts {
    pub(crate) fn update(&mut self, value: f64, retract: bool) -> bool {
        let count = if value.is_nan() {
            &mut self.nan
        } else if value == f64::INFINITY {
            &mut self.positive
        } else if value == f64::NEG_INFINITY {
            &mut self.negative
        } else {
            return false;
        };
        if retract {
            *count -= 1;
        } else {
            *count += 1;
        }
        true
    }

    pub(crate) fn count(&self) -> u64 {
        self.nan + self.positive + self.negative
    }

    pub(crate) fn value(&self) -> Option<f64> {
        if self.nan > 0 || (self.positive > 0 && self.negative > 0) {
            Some(f64::NAN)
        } else if self.positive > 0 {
            Some(f64::INFINITY)
        } else if self.negative > 0 {
            Some(f64::NEG_INFINITY)
        } else {
            None
        }
    }

    pub(crate) fn state(&self) -> Vec<ScalarValue> {
        vec![self.nan.into(), self.positive.into(), self.negative.into()]
    }

    pub(crate) fn merge(&mut self, states: &[ArrayRef]) {
        for (count, array) in [&mut self.nan, &mut self.positive, &mut self.negative]
            .into_iter()
            .zip(states)
        {
            *count += arrow::compute::sum(array.as_primitive::<UInt64Type>())
                .unwrap_or_default();
        }
    }
}

/// Neumaier summation retains low-order terms when a large finite value is
/// added and later retracted. Both components must survive a state round trip.
#[derive(Debug, Default)]
pub(crate) struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if next.is_finite() {
            self.correction += if self.sum.abs() >= value.abs() {
                (self.sum - next) + value
            } else {
                (value - next) + self.sum
            };
        } else {
            // Finite-input overflow remains an infinity, rather than producing
            // NaN merely through the compensation formula's infinity-infinity.
            self.correction = 0.0;
        }
        self.sum = next;
    }

    pub(crate) fn value(&self) -> f64 {
        self.sum + self.correction
    }

    pub(crate) fn state(&self) -> [ScalarValue; 2] {
        [self.sum.into(), self.correction.into()]
    }

    pub(crate) fn merge(&mut self, states: &[ArrayRef]) {
        for (sum, correction) in states[0]
            .as_primitive::<Float64Type>()
            .iter()
            .zip(states[1].as_primitive::<Float64Type>().iter())
        {
            if let Some(sum) = sum {
                self.add(sum);
            }
            if let Some(correction) = correction {
                self.add(correction);
            }
        }
    }
}

/// SUM/AVG's floating input types are coerced to Float64 before accumulation.
/// This state belongs only to sliding windows; grouped and ordinary aggregate
/// state layouts remain unchanged. Its sliding-only state is [finite sum,
/// correction, non-NULL count, NaN count, +inf count, -inf count].
///
/// Non-finite inputs use IEEE sum rules. Overflow caused solely by finite
/// values stays in the finite sum until no finite values remain in the frame.
/// Every finite input uses Neumaier compensation individually, including
/// retractions as additions of the negated value; batch reductions can lose
/// small terms before the compensated state sees them.
#[derive(Debug, Default)]
pub(crate) struct SlidingFloatSumAccumulator {
    finite_sum: CompensatedSum,
    count: u64,
    non_finite: NonFiniteCounts,
    average: bool,
}

impl SlidingFloatSumAccumulator {
    pub(crate) fn new(average: bool) -> Self {
        Self {
            average,
            ..Self::default()
        }
    }

    fn apply(&mut self, values: &[ArrayRef], retract: bool) {
        for value in values[0].as_primitive::<Float64Type>().iter().flatten() {
            if retract {
                self.count -= 1;
            } else {
                self.count += 1;
            }
            if !self.non_finite.update(value, retract) {
                self.finite_sum.add(if retract { -value } else { value });
            }
        }
        if self.count == self.non_finite.count() {
            self.finite_sum = CompensatedSum::default();
        }
    }
}

impl Accumulator for SlidingFloatSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.apply(values, false);
        Ok(())
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.apply(values, true);
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let value = (self.count > 0).then(|| {
            let sum = self
                .non_finite
                .value()
                .unwrap_or_else(|| self.finite_sum.value());
            if self.average {
                sum / self.count as f64
            } else {
                sum
            }
        });
        Ok(ScalarValue::Float64(value))
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state = self.finite_sum.state().to_vec();
        state.push(self.count.into());
        state.extend(self.non_finite.state());
        Ok(state)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.finite_sum.merge(&states[..2]);
        self.count += arrow::compute::sum(states[2].as_primitive::<UInt64Type>())
            .unwrap_or_default();
        self.non_finite.merge(&states[3..6]);
        Ok(())
    }

    fn supports_retract_batch(&self) -> bool {
        true
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }
}

/// A sliding‐window accumulator for `SUM(DISTINCT)` over Int64 columns.
/// Maintains a running sum so that `evaluate()` is O(1).
#[derive(Debug)]
pub struct SlidingDistinctSumAccumulator {
    /// Map each distinct value → its current count in the window
    counts: HashMap<i64, usize, RandomState>,
    /// Running sum of all distinct keys currently in the window
    sum: i64,
    /// Data type (must be Int64)
    data_type: DataType,
}

impl SlidingDistinctSumAccumulator {
    /// Create a new accumulator; only `DataType::Int64` is supported.
    pub fn try_new(data_type: &DataType) -> Result<Self> {
        // TODO support other numeric types
        if *data_type != DataType::Int64 {
            return exec_err!(
                "SlidingDistinctSumAccumulator only supports Int64, got {data_type}"
            );
        }
        Ok(Self {
            counts: HashMap::default(),
            sum: 0,
            data_type: data_type.clone(),
        })
    }

    fn update_value(&mut self, value: i64) {
        let cnt = self.counts.entry(value).or_insert(0);
        if *cnt == 0 {
            // first occurrence in window
            self.sum = self.sum.wrapping_add(value);
        }
        *cnt += 1;
    }

    fn retract_value(&mut self, value: i64) {
        if let Some(cnt) = self.counts.get_mut(&value) {
            *cnt -= 1;
            if *cnt == 0 {
                // last copy leaving window
                self.sum = self.sum.wrapping_sub(value);
                self.counts.remove(&value);
            }
        }
    }

    fn apply_valid_values<F>(
        &mut self,
        arr: &arrow::array::PrimitiveArray<Int64Type>,
        mut op: F,
    ) where
        F: FnMut(&mut Self, i64),
    {
        if arr.null_count() == 0 {
            for &value in arr.values() {
                op(self, value);
            }
        } else {
            for (idx, &value) in arr.values().iter().enumerate() {
                if arr.is_valid(idx) {
                    op(self, value);
                }
            }
        }
    }
}

impl Accumulator for SlidingDistinctSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let arr = values[0].as_primitive::<Int64Type>();
        self.apply_valid_values(arr, Self::update_value);
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        // O(1) wrap of running sum
        Ok(ScalarValue::Int64(
            (!self.counts.is_empty()).then_some(self.sum),
        ))
    }

    fn size(&self) -> usize {
        // Estimate the owned map buckets; implementation-specific control bytes are excluded.
        size_of_val(self) + self.counts.capacity() * size_of::<(i64, usize)>()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        // Serialize distinct keys for cross-partition merge if needed
        let keys = self
            .counts
            .keys()
            .copied()
            .map(Some)
            .map(ScalarValue::Int64)
            .collect::<Vec<_>>();
        Ok(vec![ScalarValue::List(ScalarValue::new_list_nullable(
            &keys,
            &self.data_type,
        ))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // Merge distinct keys from other partitions
        let list_arr = states[0].as_list::<i32>();
        for maybe_inner in list_arr.iter().flatten() {
            for idx in 0..maybe_inner.len() {
                if let ScalarValue::Int64(Some(v)) =
                    ScalarValue::try_from_array(&*maybe_inner, idx)?
                {
                    self.update_value(v);
                }
            }
        }
        Ok(())
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let arr = values[0].as_primitive::<Int64Type>();
        self.apply_valid_values(arr, Self::retract_value);
        Ok(())
    }

    fn supports_retract_batch(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Decimal128Array, Int64Array},
        buffer::{NullBuffer, ScalarBuffer},
    };
    use std::{
        mem::{size_of, size_of_val},
        sync::Arc,
    };

    #[test]
    fn sliding_distinct_sum_ignores_null_slots() -> Result<()> {
        let mut acc = SlidingDistinctSumAccumulator::try_new(&DataType::Int64)?;

        let values: ArrayRef = Arc::new(Int64Array::new(
            ScalarBuffer::from(vec![42, 5, 5]),
            Some(NullBuffer::from(vec![false, true, true])),
        ));
        acc.update_batch(&[values])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Int64(Some(5)));

        let retract: ArrayRef = Arc::new(Int64Array::new(
            ScalarBuffer::from(vec![42, 5]),
            Some(NullBuffer::from(vec![false, true])),
        ));
        acc.retract_batch(&[retract])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Int64(Some(5)));

        let retract_last: ArrayRef =
            Arc::new(Int64Array::new(ScalarBuffer::from(vec![5]), None));
        acc.retract_batch(&[retract_last])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Int64(None));

        Ok(())
    }

    fn expected_sliding_distinct_sum_size(acc: &SlidingDistinctSumAccumulator) -> usize {
        size_of_val(acc) + acc.counts.capacity() * size_of::<(i64, usize)>()
    }

    #[test]
    fn sliding_distinct_sum_size_includes_hash_map_capacity() -> Result<()> {
        let mut acc = SlidingDistinctSumAccumulator::try_new(&DataType::Int64)?;
        let empty_size = acc.size();
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        acc.update_batch(&[Arc::clone(&values)])?;

        let expected = expected_sliding_distinct_sum_size(&acc);
        assert!(acc.counts.capacity() > 0);
        assert_eq!(acc.size(), expected);
        assert!(acc.size() > empty_size);

        let initial_capacity = acc.counts.capacity();
        let additional_values: ArrayRef =
            Arc::new(Int64Array::from_iter(4..=(4 + initial_capacity as i64)));
        acc.update_batch(&[Arc::clone(&additional_values)])?;

        let grown_size = expected_sliding_distinct_sum_size(&acc);
        assert!(acc.counts.capacity() > initial_capacity);
        assert_eq!(acc.size(), grown_size);
        assert!(acc.size() > expected);

        acc.retract_batch(&[values])?;
        acc.retract_batch(&[additional_values])?;
        assert!(acc.counts.is_empty());
        assert_eq!(acc.size(), grown_size);

        Ok(())
    }

    #[test]
    fn sliding_distinct_sum_returns_null_for_all_null_frame() -> Result<()> {
        let mut acc = SlidingDistinctSumAccumulator::try_new(&DataType::Int64)?;

        let values: ArrayRef = Arc::new(Int64Array::new(
            ScalarBuffer::from(vec![99]),
            Some(NullBuffer::from(vec![false])),
        ));
        acc.update_batch(&[values])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Int64(None));

        Ok(())
    }

    #[test]
    fn decimal_sum_accumulator_uses_widened_return_type() -> Result<()> {
        let values: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(99_999), Some(99_999)])
                .with_precision_and_scale(5, 2)?,
        );
        let mut acc = SumAccumulator::<Decimal128Type>::new(DataType::Decimal128(15, 2));

        acc.update_batch(&[values])?;

        assert_eq!(
            acc.evaluate()?,
            ScalarValue::Decimal128(Some(199_998), 15, 2)
        );
        Ok(())
    }

    #[test]
    fn sum_value_from_stats_widens_small_integer_sum() {
        let statistics = datafusion_common::Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: vec![datafusion_common::ColumnStatistics {
                sum_value: Precision::Exact(ScalarValue::Int32(Some(10))),
                ..Default::default()
            }],
        };
        let return_type = DataType::Int64;
        let expr: Arc<dyn datafusion_physical_expr::PhysicalExpr> =
            Arc::new(Column::new("a", 0));
        let exprs = vec![expr];
        let statistics_args = StatisticsArgs {
            statistics: &statistics,
            return_type: &return_type,
            is_distinct: false,
            exprs: &exprs,
        };

        assert_eq!(
            Sum::new().value_from_stats(&statistics_args),
            Some(ScalarValue::Int64(Some(10)))
        );
    }

    #[test]
    fn sum_value_from_stats_casts_decimal_sum_to_return_type() {
        let statistics = datafusion_common::Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: vec![datafusion_common::ColumnStatistics {
                sum_value: Precision::Exact(ScalarValue::Decimal128(Some(12345), 5, 2)),
                ..Default::default()
            }],
        };
        let return_type = DataType::Decimal128(15, 2);
        let expr: Arc<dyn datafusion_physical_expr::PhysicalExpr> =
            Arc::new(Column::new("a", 0));
        let exprs = vec![expr];
        let statistics_args = StatisticsArgs {
            statistics: &statistics,
            return_type: &return_type,
            is_distinct: false,
            exprs: &exprs,
        };

        assert_eq!(
            Sum::new().value_from_stats(&statistics_args),
            Some(ScalarValue::Decimal128(Some(12345), 15, 2))
        );
    }
    fn narrow_args<R>(
        input: &DataType,
        return_type: &DataType,
        is_distinct: bool,
        f: impl FnOnce(AccumulatorArgs) -> R,
    ) -> R {
        let schema = arrow::datatypes::Schema::empty();
        let expr_field = Arc::new(Field::new("a", input.clone(), true));
        f(AccumulatorArgs {
            return_field: Arc::new(Field::new("sum", return_type.clone(), true)),
            schema: &schema,
            expr_fields: &[expr_field],
            ignore_nulls: false,
            order_bys: &[],
            is_distinct,
            name: "sum",
            is_reversed: false,
            exprs: &[],
        })
    }

    /// `sum` over narrow storage has the type of `sum` over the logical type,
    /// and accumulates beyond the narrow type.
    #[test]
    fn narrow_storage_has_the_sum_of_the_logical_type() -> Result<()> {
        let logical = DataType::Decimal128(15, 2);
        let narrow = DataType::Decimal64(15, 2);
        let function = Sum::with_logical_input(logical.clone());
        assert_eq!(function.logical_input(), Some(&logical));
        assert_ne!(function, Sum::new());
        let return_type = function.return_type(std::slice::from_ref(&narrow))?;
        assert_eq!(return_type, Sum::new().return_type(&[logical])?);
        assert_eq!(return_type, DataType::Decimal128(25, 2));

        let values: ArrayRef = Arc::new(
            arrow::array::Decimal64Array::from(vec![
                Some(i64::MAX),
                None,
                Some(i64::MAX),
            ])
            .with_precision_and_scale(15, 2)?,
        );
        let mut all = narrow_args(&narrow, &return_type, false, |args| {
            function.accumulator(args)
        })?;
        all.update_batch(std::slice::from_ref(&values))?;
        assert_eq!(
            all.evaluate()?,
            ScalarValue::Decimal128(Some(2 * i128::from(i64::MAX)), 25, 2)
        );
        let mut groups = narrow_args(&narrow, &return_type, false, |args| {
            function.create_groups_accumulator(args)
        })?;
        groups.update_batch(&[values], &[0, 1, 0], None, 2)?;
        let sums = groups.evaluate(datafusion_expr::EmitTo::All)?;
        assert_eq!(sums.data_type(), &return_type);
        let sums = sums.as_primitive::<Decimal128Type>();
        assert_eq!(sums.value(0), 2 * i128::from(i64::MAX));
        assert!(sums.is_null(1));
        Ok(())
    }

    #[test]
    fn narrow_storage_rejects_other_uses() {
        let function = Sum::with_logical_input(DataType::Decimal128(15, 2));
        let return_type = DataType::Decimal128(25, 2);
        let narrow = DataType::Decimal64(15, 2);
        // Another scale, the logical type itself, and DISTINCT are errors.
        for (input, distinct) in [
            (DataType::Decimal64(15, 3), false),
            (DataType::Decimal128(15, 2), false),
            (narrow.clone(), true),
        ] {
            assert!(
                narrow_args(&input, &return_type, distinct, |args| function
                    .accumulator(args))
                .is_err()
            );
            assert!(
                narrow_args(&input, &return_type, distinct, |args| function
                    .create_groups_accumulator(args))
                .is_err()
            );
        }
        assert!(
            narrow_args(&narrow, &return_type, false, |args| function
                .create_sliding_accumulator(args))
            .is_err()
        );
    }
}

#[cfg(test)]
mod sliding_nonfinite_tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::Schema;
    use std::sync::Arc;

    fn accumulator(
        udf: &dyn AggregateUDFImpl,
        arity: usize,
    ) -> Result<Box<dyn Accumulator>> {
        let schema = Schema::empty();
        let field = Arc::new(Field::new("x", DataType::Float64, true));
        let fields = vec![field; arity];
        udf.create_sliding_accumulator(AccumulatorArgs {
            return_field: Arc::new(Field::new("result", DataType::Float64, true)),
            schema: &schema,
            expr_fields: &fields,
            ignore_nulls: false,
            order_bys: &[],
            is_distinct: false,
            name: udf.name(),
            is_reversed: false,
            exprs: &[],
        })
    }

    fn arrays(values: &[Option<f64>], arity: usize) -> Vec<ArrayRef> {
        vec![Arc::new(Float64Array::from(values.to_vec())) as ArrayRef; arity]
    }

    fn assert_float(actual: ScalarValue, expected: f64) {
        let ScalarValue::Float64(Some(actual)) = actual else {
            panic!("expected {expected}, got {actual:?}");
        };
        if expected.is_nan() {
            assert!(actual.is_nan(), "expected NaN, got {actual}");
        } else if expected.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            assert!(
                (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0),
                "expected {expected}, got {actual}"
            );
        }
    }

    // A finite-only accumulator over the remaining frame is the oracle. This
    // checks recovery without prescribing the result while non-finites remain.
    fn recovers(udf: &dyn AggregateUDFImpl, arity: usize) -> Result<()> {
        let mut results = vec![];
        for special in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut sliding = accumulator(udf, arity)?;
            sliding.update_batch(&arrays(
                &[Some(special), None, Some(1.0), Some(3.0)],
                arity,
            ))?;
            sliding.retract_batch(&arrays(&[Some(special), None], arity))?;
            let mut fresh = accumulator(udf, arity)?;
            fresh.update_batch(&arrays(&[Some(1.0), Some(3.0)], arity))?;
            let expected = fresh.evaluate()?;
            let actual = sliding.evaluate()?;
            eprintln!(
                "{} after retracting {special}: {actual:?}; expected {expected:?}",
                udf.name()
            );
            results.push((actual, expected));
        }
        for (actual, expected) in results {
            assert_same(actual, expected);
        }
        Ok(())
    }

    macro_rules! recovery_test {
        ($name:ident, $udf:expr, $arity:expr) => {
            #[test]
            fn $name() -> Result<()> {
                recovers(&$udf, $arity)
            }
        };
    }
    recovery_test!(sliding_nonfinite_sum, Sum::new(), 1);
    recovery_test!(sliding_nonfinite_avg, crate::average::Avg::new(), 1);
    recovery_test!(
        sliding_nonfinite_variance,
        crate::variance::VariancePopulation::new(),
        1
    );
    recovery_test!(
        sliding_nonfinite_variance_sample,
        crate::variance::VarianceSample::new(),
        1
    );
    recovery_test!(sliding_nonfinite_stddev, crate::stddev::StddevPop::new(), 1);
    recovery_test!(
        sliding_nonfinite_stddev_sample,
        crate::stddev::Stddev::new(),
        1
    );
    recovery_test!(
        sliding_nonfinite_covariance,
        crate::covariance::CovariancePopulation::new(),
        2
    );
    recovery_test!(
        sliding_nonfinite_covariance_sample,
        crate::covariance::CovarianceSample::new(),
        2
    );
    recovery_test!(
        sliding_nonfinite_correlation,
        crate::correlation::Correlation::new(),
        2
    );
    recovery_test!(
        sliding_nonfinite_regr,
        crate::regr::Regr::new(crate::regr::RegrType::Slope, "regr_slope"),
        2
    );
    recovery_test!(sliding_nonfinite_min, crate::min_max::Min::new(), 1);
    recovery_test!(sliding_nonfinite_max, crate::min_max::Max::new(), 1);

    #[test]
    fn sliding_nonfinite_sum_avg_frames() -> Result<()> {
        for (udf, average) in [
            (&Sum::new() as &dyn AggregateUDFImpl, false),
            (&crate::average::Avg::new(), true),
        ] {
            let mut acc = accumulator(udf, 1)?;
            acc.update_batch(&arrays(&[Some(f64::NAN), Some(f64::NAN)], 1))?;
            assert_float(acc.evaluate()?, f64::NAN);
            acc.update_batch(&arrays(
                &[
                    None,
                    Some(2.0),
                    Some(f64::INFINITY),
                    Some(f64::NEG_INFINITY),
                ],
                1,
            ))?;
            acc.retract_batch(&arrays(&[Some(f64::NAN), Some(f64::NAN)], 1))?;
            assert_float(acc.evaluate()?, f64::NAN);
            acc.retract_batch(&arrays(&[None, Some(2.0), Some(f64::INFINITY)], 1))?;
            assert_float(acc.evaluate()?, f64::NEG_INFINITY);
            acc.update_batch(&arrays(&[Some(6.0), None, Some(2.0)], 1))?;
            acc.retract_batch(&arrays(&[Some(f64::NEG_INFINITY)], 1))?;
            assert_float(acc.evaluate()?, if average { 4.0 } else { 8.0 });
            acc.retract_batch(&arrays(&[Some(6.0), None, Some(2.0)], 1))?;
            assert_eq!(acc.evaluate()?, ScalarValue::Float64(None));
            acc.update_batch(&arrays(&[None], 1))?;
            assert_eq!(acc.evaluate()?, ScalarValue::Float64(None));
        }
        Ok(())
    }

    #[test]
    fn sliding_nonfinite_t05_first_frame() -> Result<()> {
        // MKR4-20261116/T05-first-frame.csv: Float32 mid values and DuckDB
        // rolling_mid, ROWS BETWEEN 4 PRECEDING AND CURRENT ROW. The CSV is
        // rounded for display; restore each mid to Float32 before widening.
        let mid = [
            f64::NAN,
            f64::NAN,
            f64::NAN,
            31.99059,
            29.5,
            30.75,
            31.226667,
            31.42,
            30.613333,
            37.0,
            35.0,
            32.25,
            31.25,
            31.25,
            31.25,
        ]
        .map(|x| x as f32 as f64);
        let expected = [
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            30.97745132446289,
            30.702000045776366,
            32.202000045776366,
            33.05200004577637,
            33.256666564941405,
            33.222666549682614,
            33.35,
            32.2,
        ];
        let mut avg = accumulator(&crate::average::Avg::new(), 1)?;
        for (i, value) in mid.into_iter().enumerate() {
            avg.update_batch(&arrays(&[Some(value)], 1))?;
            if i >= 5 {
                avg.retract_batch(&arrays(&[Some(mid[i - 5])], 1))?;
            }
            assert_float(avg.evaluate()?, expected[i]);
        }
        Ok(())
    }
    fn moment_udfs() -> Vec<(Box<dyn AggregateUDFImpl>, usize)> {
        use crate::regr::{Regr, RegrType};
        let mut udfs: Vec<(Box<dyn AggregateUDFImpl>, usize)> = vec![
            (Box::new(Sum::new()), 1),
            (Box::new(crate::average::Avg::new()), 1),
            (Box::new(crate::variance::VariancePopulation::new()), 1),
            (Box::new(crate::variance::VarianceSample::new()), 1),
            (Box::new(crate::stddev::StddevPop::new()), 1),
            (Box::new(crate::stddev::Stddev::new()), 1),
            (Box::new(crate::covariance::CovariancePopulation::new()), 2),
            (Box::new(crate::covariance::CovarianceSample::new()), 2),
            (Box::new(crate::correlation::Correlation::new()), 2),
        ];
        for (kind, name) in [
            (RegrType::Slope, "regr_slope"),
            (RegrType::Intercept, "regr_intercept"),
            (RegrType::Count, "regr_count"),
            (RegrType::R2, "regr_r2"),
            (RegrType::AvgX, "regr_avgx"),
            (RegrType::AvgY, "regr_avgy"),
            (RegrType::SXX, "regr_sxx"),
            (RegrType::SYY, "regr_syy"),
            (RegrType::SXY, "regr_sxy"),
        ] {
            udfs.push((Box::new(Regr::new(kind, name)), 2));
        }
        udfs
    }

    fn assert_same(actual: ScalarValue, expected: ScalarValue) {
        match expected {
            ScalarValue::Float64(Some(x)) => assert_float(actual, x),
            _ => assert_eq!(actual, expected),
        }
    }

    #[test]
    fn sliding_nonfinite_state_roundtrip_and_retract() -> Result<()> {
        for (udf, arity) in moment_udfs() {
            let mut partial = accumulator(udf.as_ref(), arity)?;
            let special = arrays(
                &[
                    Some(f64::NAN),
                    Some(f64::INFINITY),
                    Some(f64::NEG_INFINITY),
                    None,
                ],
                arity,
            );
            partial.update_batch(&special)?;
            let state = partial
                .state()?
                .into_iter()
                .map(|x| x.to_array())
                .collect::<Result<Vec<_>>>()?;
            let mut merged = accumulator(udf.as_ref(), arity)?;
            merged.merge_batch(&state)?;
            assert_same(merged.evaluate()?, partial.evaluate()?);
            let state = merged
                .state()?
                .into_iter()
                .map(|s| s.to_array())
                .collect::<Result<Vec<_>>>()?;
            merged = accumulator(udf.as_ref(), arity)?;
            merged.merge_batch(&state)?;
            let finite = arrays(&[Some(1.0), None, Some(3.0)], arity);
            merged.update_batch(&finite)?;
            merged.retract_batch(&special)?;
            let mut partial = accumulator(udf.as_ref(), arity)?;
            partial.update_batch(&finite)?;
            let state = partial
                .state()?
                .into_iter()
                .map(|s| s.to_array())
                .collect::<Result<Vec<_>>>()?;
            let mut fresh = accumulator(udf.as_ref(), arity)?;
            fresh.merge_batch(&state)?;
            assert_same(merged.evaluate()?, fresh.evaluate()?);
            merged.retract_batch(&finite)?;
            fresh.retract_batch(&finite)?;
            assert_same(merged.evaluate()?, fresh.evaluate()?);
        }
        Ok(())
    }

    #[test]
    fn sliding_nonfinite_pairwise_and_finite_frames() -> Result<()> {
        // Asymmetric axes catch swapped REGR(Y, X) arguments. One-sided
        // NULLs must exclude the entire pair, including non-finite partners.
        let x = vec![
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(9.0),
            None,
            Some(1.0),
            Some(2.0),
            Some(4.0),
            Some(8.0),
            Some(16.0),
        ];
        let y = vec![
            Some(11.0),
            Some(12.0),
            Some(13.0),
            None,
            Some(f64::NAN),
            Some(3.0),
            Some(5.0),
            Some(9.0),
            Some(17.0),
            Some(33.0),
        ];
        for (udf, arity) in moment_udfs() {
            for swap in [false, true] {
                let input = if arity == 1 {
                    arrays(&x, 1)
                } else {
                    let mut values: Vec<ArrayRef> = vec![
                        Arc::new(Float64Array::from(x.clone())),
                        Arc::new(Float64Array::from(y.clone())),
                    ];
                    if swap {
                        values.swap(0, 1);
                    }
                    values
                };
                let mut sliding = accumulator(udf.as_ref(), arity)?;
                for i in 0..x.len() {
                    sliding.update_batch(
                        &input.iter().map(|a| a.slice(i, 1)).collect::<Vec<_>>(),
                    )?;
                    if i >= 3 {
                        sliding.retract_batch(
                            &input.iter().map(|a| a.slice(i - 3, 1)).collect::<Vec<_>>(),
                        )?;
                    }
                    // Compare the finite tail against the unchanged non-sliding
                    // accumulator, independently of the sliding implementation.
                    if i >= 7 {
                        let schema = Schema::empty();
                        let field = Arc::new(Field::new("x", DataType::Float64, true));
                        let fields = vec![field; arity];
                        let mut fresh = udf.accumulator(AccumulatorArgs {
                            return_field: Arc::new(Field::new(
                                "result",
                                DataType::Float64,
                                true,
                            )),
                            schema: &schema,
                            expr_fields: &fields,
                            ignore_nulls: false,
                            order_bys: &[],
                            is_distinct: false,
                            name: udf.name(),
                            is_reversed: false,
                            exprs: &[],
                        })?;
                        fresh.update_batch(
                            &input.iter().map(|a| a.slice(i - 2, 3)).collect::<Vec<_>>(),
                        )?;
                        assert_same(sliding.evaluate()?, fresh.evaluate()?);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn sliding_nonfinite_regression_independent_axes() -> Result<()> {
        use crate::regr::{Regr, RegrType};
        let values = vec![
            Arc::new(Float64Array::from(vec![
                Some(f64::NAN),
                Some(3.0),
                Some(5.0),
            ])) as ArrayRef,
            Arc::new(Float64Array::from(vec![Some(0.0), Some(1.0), Some(2.0)]))
                as ArrayRef,
        ];
        for (kind, expected) in [
            (RegrType::AvgX, 1.0),
            (RegrType::AvgY, f64::NAN),
            (RegrType::SXX, 2.0),
            (RegrType::SYY, f64::NAN),
            (RegrType::SXY, f64::NAN),
        ] {
            let mut acc = accumulator(&Regr::new(kind, "regr"), 2)?;
            acc.update_batch(&values)?;
            assert_float(acc.evaluate()?, expected);
        }
        Ok(())
    }

    fn final_frame(
        udf: &dyn AggregateUDFImpl,
        input: &[ArrayRef],
        width: usize,
    ) -> Result<ScalarValue> {
        let mut acc = accumulator(udf, input.len())?;
        for i in 0..input[0].len() {
            acc.update_batch(&input.iter().map(|a| a.slice(i, 1)).collect::<Vec<_>>())?;
            if i >= width {
                acc.retract_batch(
                    &input
                        .iter()
                        .map(|a| a.slice(i - width, 1))
                        .collect::<Vec<_>>(),
                )?;
            }
        }
        acc.evaluate()
    }

    #[test]
    fn sliding_finite_variance_outlier() -> Result<()> {
        for (udf, arity) in moment_udfs().into_iter().filter(|(u, _)| {
            ["var_pop", "var_samp", "stddev_pop", "stddev"].contains(&u.name())
        }) {
            for values in [
                vec![9193063041.610329, 3.3, 3.3],
                vec![1e4, 3.3, 3.3, 3.3, 3.3],
                vec![1e7, 3.3, 3.3, 3.3, 3.3],
            ] {
                let width = values.len() - 1;
                let input =
                    arrays(&values.into_iter().map(Some).collect::<Vec<_>>(), arity);
                // Fresh moments over an all-equal finite frame are exactly zero.
                assert_float(final_frame(udf.as_ref(), &input, width)?, 0.0);
            }
        }
        Ok(())
    }

    #[test]
    fn sliding_finite_covariance_singleton() -> Result<()> {
        let udf = crate::covariance::CovariancePopulation::new();
        assert_float(
            final_frame(
                &udf,
                &arrays(&[Some(9193063041.610329), Some(3.3), Some(3.3)], 2),
                1,
            )?,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn sliding_finite_constant_regression() -> Result<()> {
        use crate::regr::{Regr, RegrType};
        let x =
            Arc::new(Float64Array::from(vec![9193063041.610329, 3.3, 3.3])) as ArrayRef;
        let y = Arc::new(Float64Array::from(vec![4.0, 1.0, 2.0])) as ArrayRef;
        for kind in [RegrType::Slope, RegrType::Intercept, RegrType::R2] {
            assert_eq!(
                final_frame(
                    &Regr::new(kind, "regr"),
                    &[Arc::clone(&y), Arc::clone(&x)],
                    2
                )?,
                ScalarValue::Float64(None)
            );
        }
        assert_eq!(
            final_frame(&crate::correlation::Correlation::new(), &[x, y], 2)?,
            ScalarValue::Float64(None)
        );
        Ok(())
    }

    #[test]
    fn sliding_finite_sum_compensation() -> Result<()> {
        for (udf, expected) in [
            (&Sum::new() as &dyn AggregateUDFImpl, 2.0),
            (&crate::average::Avg::new(), 1.0),
        ] {
            for outlier in [1e17, -1e17] {
                let values = arrays(&[Some(outlier), Some(1.0), Some(1.0)], 1);
                assert_float(final_frame(udf, &values, 2)?, expected);
                let mut partial = accumulator(udf, 1)?;
                partial.update_batch(&values)?;
                let state = partial
                    .state()?
                    .into_iter()
                    .map(|s| s.to_array())
                    .collect::<Result<Vec<_>>>()?;
                let mut merged = accumulator(udf, 1)?;
                merged.merge_batch(&state)?;
                merged.retract_batch(&arrays(&[Some(outlier)], 1))?;
                assert_float(merged.evaluate()?, expected);
            }
        }
        Ok(())
    }
    #[test]
    fn sliding_finite_singleton_state_then_update() -> Result<()> {
        // Verify the reset itself, then grow the singleton again. Population
        // variance alone always returns zero for one value, even with bad m2.
        for (udf, arity) in moment_udfs() {
            let mut partial = accumulator(udf.as_ref(), arity)?;
            partial
                .update_batch(&arrays(&[Some(9193063041.610329), Some(3.3)], arity))?;
            partial.retract_batch(&arrays(&[Some(9193063041.610329)], arity))?;
            let states = partial
                .state()?
                .into_iter()
                .map(|s| s.to_array())
                .collect::<Result<Vec<_>>>()?;
            let mut merged = accumulator(udf.as_ref(), arity)?;
            merged.merge_batch(&states)?;
            merged.update_batch(&arrays(&[Some(6.6)], arity))?;
            let mut fresh = accumulator(udf.as_ref(), arity)?;
            fresh.update_batch(&arrays(&[Some(3.3), Some(6.6)], arity))?;
            assert_same(merged.evaluate()?, fresh.evaluate()?);
        }
        Ok(())
    }
    #[test]
    fn sliding_v3_positive_variance_residuals() -> Result<()> {
        for (udf, _) in moment_udfs().into_iter().filter(|(u, _)| {
            ["var_pop", "var_samp", "stddev_pop", "stddev"].contains(&u.name())
        }) {
            for outlier in [5071415981.588, 123456789.0] {
                assert_float(
                    final_frame(
                        udf.as_ref(),
                        &arrays(&[Some(outlier), Some(3.3), Some(3.3)], 1),
                        2,
                    )?,
                    0.0,
                );
            }
            for outlier in [1e4, 1e5, 1e6, 1e7] {
                assert_float(
                    final_frame(
                        udf.as_ref(),
                        &arrays(
                            &[Some(outlier), Some(3.3), Some(3.3), Some(3.3), Some(3.3)],
                            1,
                        ),
                        4,
                    )?,
                    0.0,
                );
            }
        }
        Ok(())
    }

    #[test]
    fn sliding_v3_covariance_constant_frame() -> Result<()> {
        for (udf, _) in moment_udfs()
            .into_iter()
            .filter(|(u, _)| ["covar_pop", "covar_samp"].contains(&u.name()))
        {
            assert_float(
                final_frame(
                    udf.as_ref(),
                    &arrays(&[Some(1e9), Some(3.3), Some(3.3)], 2),
                    2,
                )?,
                0.0,
            );
        }
        Ok(())
    }

    #[test]
    fn sliding_v3_all_regression_constant_x() -> Result<()> {
        let y = Arc::new(Float64Array::from(vec![4.0, 1.0, 2.0])) as ArrayRef;
        let x = Arc::new(Float64Array::from(vec![123456789.0, 3.3, 3.3])) as ArrayRef;
        for (udf, arity) in moment_udfs()
            .into_iter()
            .filter(|(u, _)| u.name().starts_with("regr_"))
        {
            let input = [Arc::clone(&y), Arc::clone(&x)];
            let mut fresh = accumulator(udf.as_ref(), arity)?;
            fresh
                .update_batch(&input.iter().map(|a| a.slice(1, 2)).collect::<Vec<_>>())?;
            assert_same(final_frame(udf.as_ref(), &input, 2)?, fresh.evaluate()?);
        }
        Ok(())
    }

    #[test]
    fn sliding_v3_correlation_bound() -> Result<()> {
        for sign in [1.0, -1.0] {
            let x = Arc::new(Float64Array::from(vec![1.0, 4.0])) as ArrayRef;
            let y = Arc::new(Float64Array::from(vec![sign, 4.0 * sign])) as ArrayRef;
            let actual =
                final_frame(&crate::correlation::Correlation::new(), &[x, y], 2)?;
            let ScalarValue::Float64(Some(value)) = actual else {
                panic!("{actual:?}");
            };
            assert!((-1.0..=1.0).contains(&value), "{value}");
        }
        Ok(())
    }

    #[test]
    fn sliding_v3_deque_memory_accounting() -> Result<()> {
        for (udf, arity) in moment_udfs()
            .into_iter()
            .filter(|(u, _)| !["sum", "avg"].contains(&u.name()))
        {
            let mut acc = accumulator(udf.as_ref(), arity)?;
            let initial = acc.size();
            let values =
                arrays(&(0..128).map(|i| Some(i as f64)).collect::<Vec<_>>(), arity);
            acc.update_batch(&values)?;
            assert!(
                acc.size() >= initial + 128 * arity * size_of::<f64>(),
                "{} must charge retained values",
                udf.name()
            );
            acc.retract_batch(&values)?;
            assert!(acc.size() >= initial + 128 * arity * size_of::<f64>());
        }
        Ok(())
    }
    #[test]
    fn sliding_v3_retained_state_recovers_after_merge() -> Result<()> {
        let x = Arc::new(Float64Array::from(vec![5071415981.588, 3.3, 3.3])) as ArrayRef;
        let y = Arc::new(Float64Array::from(vec![1e9, 1.0, 2.0])) as ArrayRef;
        for (udf, arity) in moment_udfs() {
            let input = if arity == 1 {
                vec![Arc::clone(&x)]
            } else {
                vec![Arc::clone(&y), Arc::clone(&x)]
            };
            let mut partial = accumulator(udf.as_ref(), arity)?;
            partial.update_batch(&input)?;
            let state = partial
                .state()?
                .into_iter()
                .map(|s| s.to_array())
                .collect::<Result<Vec<_>>>()?;
            let mut merged = accumulator(udf.as_ref(), arity)?;
            merged.merge_batch(&state)?;
            merged.retract_batch(
                &input.iter().map(|a| a.slice(0, 1)).collect::<Vec<_>>(),
            )?;
            let mut fresh = accumulator(udf.as_ref(), arity)?;
            fresh
                .update_batch(&input.iter().map(|a| a.slice(1, 2)).collect::<Vec<_>>())?;
            assert_same(merged.evaluate()?, fresh.evaluate()?);
        }
        Ok(())
    }

    #[test]
    fn sliding_sum_overflow_limit() -> Result<()> {
        let mut acc = accumulator(&Sum::new(), 1)?;
        acc.update_batch(&arrays(&[Some(f64::MAX), Some(f64::MAX), Some(1.0)], 1))?;
        acc.retract_batch(&arrays(&[Some(f64::MAX), Some(f64::MAX)], 1))?;
        assert_float(acc.evaluate()?, f64::INFINITY);
        acc.retract_batch(&arrays(&[Some(1.0)], 1))?;
        assert_eq!(acc.evaluate()?, ScalarValue::Float64(None));
        acc.update_batch(&arrays(&[Some(2.0)], 1))?;
        assert_float(acc.evaluate()?, 2.0);
        Ok(())
    }
    #[test]
    fn sliding_v3_recompute_matches_fresh_frames() -> Result<()> {
        let x = (0..40)
            .map(|i| {
                if i % 11 == 0 {
                    Some(1e9 + i as f64)
                } else if i % 7 == 0 {
                    None
                } else {
                    Some(3.3 + (i % 3) as f64)
                }
            })
            .collect::<Vec<_>>();
        let y = (0..40)
            .map(|i| {
                if i % 13 == 0 {
                    None
                } else {
                    Some(2.0 + (i % 5) as f64)
                }
            })
            .collect::<Vec<_>>();
        for (udf, arity) in moment_udfs() {
            for width in [1, 2, 4, 8] {
                let input = if arity == 1 {
                    arrays(&x, 1)
                } else {
                    vec![
                        Arc::new(Float64Array::from(y.clone())) as ArrayRef,
                        Arc::new(Float64Array::from(x.clone())) as ArrayRef,
                    ]
                };
                let mut sliding = accumulator(udf.as_ref(), arity)?;
                for i in 0..x.len() {
                    sliding.update_batch(
                        &input.iter().map(|a| a.slice(i, 1)).collect::<Vec<_>>(),
                    )?;
                    if i >= width {
                        sliding.retract_batch(
                            &input
                                .iter()
                                .map(|a| a.slice(i - width, 1))
                                .collect::<Vec<_>>(),
                        )?;
                    }
                    let start = (i + 1).saturating_sub(width);
                    let mut fresh = accumulator(udf.as_ref(), arity)?;
                    fresh.update_batch(
                        &input
                            .iter()
                            .map(|a| a.slice(start, i + 1 - start))
                            .collect::<Vec<_>>(),
                    )?;
                    let actual = sliding.evaluate()?;
                    let expected = fresh.evaluate()?;
                    if ["covar_pop", "covar_samp", "regr_sxy"].contains(&udf.name()) {
                        // Chan merges change the association of cross-moment
                        // sums. A near-zero covariance next to a 1e9 outlier is
                        // ill-conditioned in result-relative units. Keep the
                        // fresh answer and bound normalized error by 64 ulps of
                        // sqrt(m2_x*m2_y), with an absolute floor near zero.
                        let xs = input[0].as_primitive::<Float64Type>();
                        let ys = input[1].as_primitive::<Float64Type>();
                        let pairs = (start..=i)
                            .filter(|&j| xs.is_valid(j) && ys.is_valid(j))
                            .map(|j| (xs.value(j), ys.value(j)))
                            .collect::<Vec<_>>();
                        if let (
                            ScalarValue::Float64(Some(actual)),
                            ScalarValue::Float64(Some(expected)),
                        ) = (&actual, &expected)
                        {
                            let n = pairs.len() as f64;
                            let mx = pairs.iter().map(|p| p.0).sum::<f64>() / n;
                            let my = pairs.iter().map(|p| p.1).sum::<f64>() / n;
                            let m2x =
                                pairs.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
                            let m2y =
                                pairs.iter().map(|p| (p.1 - my).powi(2)).sum::<f64>();
                            let denominator = match udf.name() {
                                "covar_pop" => n,
                                "covar_samp" => n - 1.0,
                                _ => 1.0,
                            };
                            let scale = m2x.sqrt() * m2y.sqrt() / denominator;
                            assert!(
                                (actual - expected).abs()
                                    <= 64.0 * f64::EPSILON * scale.max(1.0),
                                "{}: {actual} vs {expected}, scale {scale}",
                                udf.name()
                            );
                        } else {
                            assert_same(actual, expected);
                        }
                    } else {
                        assert_same(actual, expected);
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn sliding_v4_constant_tail_ulp() -> Result<()> {
        let values = arrays(
            &[
                Some(3.3000000000000007),
                Some(3.3),
                Some(3.3),
                Some(3.3),
                Some(3.3),
            ],
            1,
        );
        for (udf, arity) in moment_udfs().into_iter().filter(|(u, _)| {
            [
                "var_pop",
                "var_samp",
                "stddev_pop",
                "stddev",
                "covar_pop",
                "covar_samp",
            ]
            .contains(&u.name())
        }) {
            let input = vec![Arc::clone(&values[0]); arity];
            assert_eq!(
                final_frame(udf.as_ref(), &input, 4)?,
                ScalarValue::Float64(Some(0.0)),
                "{}",
                udf.name()
            );
        }
        for (udf, arity) in moment_udfs()
            .into_iter()
            .filter(|(u, _)| u.name() == "corr" || u.name().starts_with("regr_"))
        {
            let input = vec![Arc::clone(&values[0]); arity];
            let mut fresh = accumulator(udf.as_ref(), arity)?;
            fresh
                .update_batch(&input.iter().map(|a| a.slice(1, 4)).collect::<Vec<_>>())?;
            assert_eq!(
                final_frame(udf.as_ref(), &input, 4)?,
                fresh.evaluate()?,
                "{}",
                udf.name()
            );
        }
        Ok(())
    }

    #[test]
    fn sliding_v4_geometric_decay() -> Result<()> {
        // Independent two-pass fresh variance. Relative error must be <=1e-10
        // for every complete 20-value frame, not just an absolute tolerance
        // that would hide drift as the values become small.
        let values = (0..1200)
            .map(|i| 1e9 * 0.97_f64.powi(i))
            .collect::<Vec<_>>();
        let input = arrays(&values.iter().copied().map(Some).collect::<Vec<_>>(), 1);
        let mut acc = accumulator(&crate::variance::VariancePopulation::new(), 1)?;
        for i in 0..values.len() {
            acc.update_batch(&[input[0].slice(i, 1)])?;
            if i >= 20 {
                acc.retract_batch(&[input[0].slice(i - 20, 1)])?;
            }
            if i >= 19 {
                let frame = &values[i - 19..=i];
                let mean = frame.iter().sum::<f64>() / 20.0;
                let expected =
                    frame.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 20.0;
                let ScalarValue::Float64(Some(actual)) = acc.evaluate()? else {
                    panic!("missing variance");
                };
                let relative = (actual - expected).abs() / expected;
                assert!(
                    relative <= 1e-10,
                    "row {i}: relative error {relative}, {actual} vs {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn sliding_v4_sum_small_terms() -> Result<()> {
        for (values, expected) in [
            (vec![67108864.0, 1.1], 1.1),
            (vec![1e6, 0.1, 0.2, 0.3], 0.6),
        ] {
            let mut acc = accumulator(&Sum::new(), 1)?;
            acc.update_batch(&arrays(
                &values.iter().copied().map(Some).collect::<Vec<_>>(),
                1,
            ))?;
            acc.retract_batch(&arrays(&[Some(values[0])], 1))?;
            assert_eq!(acc.evaluate()?, ScalarValue::Float64(Some(expected)));
            let mut avg = accumulator(&crate::average::Avg::new(), 1)?;
            avg.update_batch(&arrays(
                &values.iter().copied().map(Some).collect::<Vec<_>>(),
                1,
            ))?;
            avg.retract_batch(&arrays(&[Some(values[0])], 1))?;
            assert_float(avg.evaluate()?, expected / (values.len() - 1) as f64);
        }
        Ok(())
    }

    #[test]
    fn sliding_v4_non_fifo_is_internal_error() -> Result<()> {
        for (udf, arity) in moment_udfs()
            .into_iter()
            .filter(|(u, _)| !["sum", "avg"].contains(&u.name()))
        {
            for bad in [Some(2.0), Some(f64::NAN)] {
                let mut acc = accumulator(udf.as_ref(), arity)?;
                acc.update_batch(&arrays(
                    &[Some(1.0), Some(2.0), Some(f64::NAN)],
                    arity,
                ))?;
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        acc.retract_batch(&arrays(&[bad], arity))
                    }));
                let Ok(Err(error)) = result else {
                    panic!("{} must return an error without panicking", udf.name());
                };
                assert!(
                    matches!(error, datafusion_common::DataFusionError::Internal(_)),
                    "{error}"
                );
                assert!(error.to_string().contains("FIFO"), "{error}");
            }
        }
        Ok(())
    }
    fn gaussian_pairs(len: usize) -> Vec<(f64, f64)> {
        let mut seed = 0x51f0_5eed_1234_9876_u64;
        let mut uniform = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 + 0.5) / 9007199254740992.0
        };
        (0..len)
            .map(|_| {
                let radius = (-2.0 * uniform().ln()).sqrt();
                let angle = std::f64::consts::TAU * uniform();
                (radius * angle.cos(), radius * angle.sin())
            })
            .collect()
    }

    #[test]
    fn sliding_v4_gaussian_linear_work() -> Result<()> {
        use crate::covariance::PairMoments;
        use crate::variance::{SlidingMoment, SlidingMomentQueue};
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COMBINATIONS: AtomicUsize = AtomicUsize::new(0);
        #[derive(Clone, Copy, Default, Debug)]
        struct Counted(PairMoments);
        impl SlidingMoment for Counted {
            type Value = (f64, f64);
            fn singleton(value: Self::Value) -> Self {
                Self(PairMoments::singleton(value))
            }
            fn combine(self, other: Self) -> Self {
                COMBINATIONS.fetch_add(1, Ordering::Relaxed);
                Self(self.0.combine(other.0))
            }
            fn same_value(a: Self::Value, b: Self::Value) -> bool {
                PairMoments::same_value(a, b)
            }
        }
        let values = gaussian_pairs(4096);
        let mut queue = SlidingMomentQueue::<Counted>::default();
        COMBINATIONS.store(0, Ordering::Relaxed);
        for (i, &pair) in values.iter().enumerate() {
            queue.push(pair);
            if i >= 20 {
                queue.pop(values[i - 20])?;
            }
            queue.aggregate();
        }
        // One prefix combine per push, at most one suffix combine per value
        // transferred, and one combine per evaluation. This checks actual
        // moment-combine work, not wall time or a Gaussian rebuild threshold.
        let count = COMBINATIONS.load(Ordering::Relaxed);
        eprintln!(
            "Gaussian queue: {count} combinations / {} pushes / {} FIFO pops / {} evaluations",
            values.len(),
            values.len() - 20,
            values.len()
        );
        assert!(
            count <= 3 * values.len(),
            "{count} combines for {} rows",
            values.len()
        );
        Ok(())
    }

    #[test]
    fn sliding_v4_randomized_fresh_comparison() -> Result<()> {
        // Deterministic Gaussian pairs, independently accumulated by unchanged
        // ordinary UDFs. Bound every finite result by relative error 1e-10
        // plus absolute error 1e-12 near zero; NULL/count results match exactly.
        let values = gaussian_pairs(512);
        let x = Arc::new(Float64Array::from(
            values.iter().map(|p| p.0).collect::<Vec<_>>(),
        )) as ArrayRef;
        let y = Arc::new(Float64Array::from(
            values.iter().map(|p| p.1).collect::<Vec<_>>(),
        )) as ArrayRef;
        for (udf, arity) in moment_udfs() {
            let input = if arity == 1 {
                vec![Arc::clone(&x)]
            } else {
                vec![Arc::clone(&y), Arc::clone(&x)]
            };
            for width in [1, 2, 20, 63] {
                let mut sliding = accumulator(udf.as_ref(), arity)?;
                for i in 0..values.len() {
                    sliding.update_batch(
                        &input.iter().map(|a| a.slice(i, 1)).collect::<Vec<_>>(),
                    )?;
                    if i >= width {
                        sliding.retract_batch(
                            &input
                                .iter()
                                .map(|a| a.slice(i - width, 1))
                                .collect::<Vec<_>>(),
                        )?;
                    }
                    let schema = Schema::empty();
                    let field = Arc::new(Field::new("x", DataType::Float64, true));
                    let fields = vec![field; arity];
                    let mut fresh = udf.accumulator(AccumulatorArgs {
                        return_field: Arc::new(Field::new(
                            "result",
                            DataType::Float64,
                            true,
                        )),
                        schema: &schema,
                        expr_fields: &fields,
                        ignore_nulls: false,
                        order_bys: &[],
                        is_distinct: false,
                        name: udf.name(),
                        is_reversed: false,
                        exprs: &[],
                    })?;
                    let start = (i + 1).saturating_sub(width);
                    fresh.update_batch(
                        &input
                            .iter()
                            .map(|a| a.slice(start, i + 1 - start))
                            .collect::<Vec<_>>(),
                    )?;
                    let actual = sliding.evaluate()?;
                    let expected = fresh.evaluate()?;
                    if let (
                        ScalarValue::Float64(Some(actual)),
                        ScalarValue::Float64(Some(expected)),
                    ) = (&actual, &expected)
                    {
                        assert!(
                            (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
                            "{} row {i} width {width}: {actual} vs {expected}",
                            udf.name()
                        );
                    } else {
                        assert_same(actual, expected);
                    }
                }
            }
        }
        Ok(())
    }
}
