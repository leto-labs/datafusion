// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Defines physical expressions that can evaluated at runtime during query execution

use std::mem::size_of_val;
use std::ops::Range;

use arrow::array::{Array, ArrayRef, BooleanArray};
use arrow::compute::bool_and as compute_bool_and;
use arrow::compute::bool_or as compute_bool_or;
use arrow::datatypes::Field;
use arrow::datatypes::{DataType, FieldRef};

use datafusion_common::cast::as_boolean_array;
use datafusion_common::{Result, ScalarValue};
use datafusion_common::{assert_or_internal_err, internal_err};
use datafusion_common::{downcast_value, not_impl_err};
use datafusion_expr::DistinctHandling;
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::{AggregateOrderSensitivity, format_state_name};
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, GroupsAccumulator, ReversedUDAF,
    Signature, Volatility,
};

use datafusion_functions_aggregate_common::aggregate::groups_accumulator::bool_op::BooleanGroupsAccumulator;
use datafusion_macros::user_doc;
use datafusion_physical_expr::window::{RecomputeFrame, RecomputingFrameAccumulator};

// returns the new value after bool_and/bool_or with the new values, taking nullability into account
macro_rules! typed_bool_and_or_batch {
    ($VALUES:expr, $ARRAYTYPE:ident, $SCALAR:ident, $OP:ident) => {{
        let array = downcast_value!($VALUES, $ARRAYTYPE);
        let delta = $OP(array);
        Ok(ScalarValue::$SCALAR(delta))
    }};
}

// bool_and/bool_or the array and returns a ScalarValue of its corresponding type.
macro_rules! bool_and_or_batch {
    ($VALUES:expr, $OP:ident) => {{
        match $VALUES.data_type() {
            DataType::Boolean => {
                typed_bool_and_or_batch!($VALUES, BooleanArray, Boolean, $OP)
            }
            e => {
                return internal_err!(
                    "Bool and/Bool or is not expected to receive the type {e:?}"
                );
            }
        }
    }};
}

/// dynamically-typed bool_and(array) -> ScalarValue
fn bool_and_batch(values: &ArrayRef) -> Result<ScalarValue> {
    bool_and_or_batch!(values, compute_bool_and)
}

/// dynamically-typed bool_or(array) -> ScalarValue
fn bool_or_batch(values: &ArrayRef) -> Result<ScalarValue> {
    bool_and_or_batch!(values, compute_bool_or)
}

fn recompute_bool(
    values: &[ArrayRef],
    range: Range<usize>,
    filter: Option<&BooleanArray>,
    and: bool,
) -> Result<Option<bool>> {
    let [values] = values else {
        return internal_err!("Expected one input for a boolean aggregate window");
    };
    let values = as_boolean_array(values.as_ref())?;
    assert_or_internal_err!(range.end <= values.len());
    if let Some(filter) = filter {
        assert_or_internal_err!(range.end <= filter.len());
        // Read the two masks in place. Materializing a filtered array for
        // each frame would require scratch proportional to its width.
        let mut result = None;
        for i in range {
            if filter.is_valid(i) && filter.value(i) && values.is_valid(i) {
                let value = values.value(i);
                if value != and {
                    return Ok(Some(value));
                }
                result = Some(value);
            }
        }
        Ok(result)
    } else {
        // The concrete Arrow slice shares the existing buffers.
        let frame = values.slice(range.start, range.len());
        Ok(if and {
            compute_bool_and(&frame)
        } else {
            compute_bool_or(&frame)
        })
    }
}

make_udaf_expr_and_func!(
    BoolAnd,
    bool_and,
    expression,
    "The values to combine with `AND`",
    bool_and_udaf
);

make_udaf_expr_and_func!(
    BoolOr,
    bool_or,
    expression,
    "The values to combine with `OR`",
    bool_or_udaf
);

#[user_doc(
    doc_section(label = "General Functions"),
    description = "Returns true if all non-null input values are true, otherwise false.",
    syntax_example = "bool_and(expression)",
    sql_example = r#"```sql
> SELECT bool_and(column_name) FROM table_name;
+----------------------------+
| bool_and(column_name)      |
+----------------------------+
| true                       |
+----------------------------+
```"#,
    standard_argument(name = "expression", prefix = "The")
)]
/// BOOL_AND aggregate expression
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BoolAnd {
    signature: Signature,
}

impl BoolAnd {
    fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Boolean], Volatility::Immutable),
        }
    }
}

impl Default for BoolAnd {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregateUDFImpl for BoolAnd {
    fn name(&self) -> &str {
        "bool_and"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn accumulator(&self, _: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::<BoolAndAccumulator>::default())
    }

    fn create_sliding_accumulator(
        &self,
        _: AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(RecomputingFrameAccumulator::new(
            BoolAndAccumulator::default(),
        )))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Field::new(
                format_state_name(args.name, self.name()),
                DataType::Boolean,
                true,
            )
            .into(),
        ])
    }

    fn groups_accumulator_supported(&self, _args: AccumulatorArgs) -> bool {
        true
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        match args.return_field.data_type() {
            DataType::Boolean => {
                Ok(Box::new(BooleanGroupsAccumulator::new(|x, y| x && y, true)))
            }
            _ => not_impl_err!(
                "GroupsAccumulator not supported for {} with {}",
                args.name,
                args.return_field.data_type()
            ),
        }
    }

    fn order_sensitivity(&self) -> AggregateOrderSensitivity {
        AggregateOrderSensitivity::Insensitive
    }

    fn reverse_expr(&self) -> ReversedUDAF {
        ReversedUDAF::Identical
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn distinct_handling(&self) -> DistinctHandling {
        // Boolean AND/OR are idempotent: duplicates cannot change the result.
        DistinctHandling::Insensitive
    }
}

#[derive(Debug, Default)]
struct BoolAndAccumulator {
    acc: Option<bool>,
}

impl Accumulator for BoolAndAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = &values[0];
        self.acc = match (self.acc, bool_and_batch(values)?) {
            (None, ScalarValue::Boolean(v)) => v,
            (Some(v), ScalarValue::Boolean(None)) => Some(v),
            (Some(a), ScalarValue::Boolean(Some(b))) => Some(a && b),
            _ => unreachable!(),
        };
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::Boolean(self.acc))
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Boolean(self.acc)])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.update_batch(states)
    }
}

impl RecomputeFrame for BoolAndAccumulator {
    fn recompute(
        &mut self,
        values: &[ArrayRef],
        range: Range<usize>,
        filter: Option<&BooleanArray>,
    ) -> Result<()> {
        self.acc = recompute_bool(values, range, filter, true)?;
        Ok(())
    }
}

#[user_doc(
    doc_section(label = "General Functions"),
    description = "Returns true if any non-null input value is true, otherwise false.",
    syntax_example = "bool_or(expression)",
    sql_example = r#"```sql
> SELECT bool_or(column_name) FROM table_name;
+----------------------------+
| bool_or(column_name)       |
+----------------------------+
| true                       |
+----------------------------+
```"#,
    standard_argument(name = "expression", prefix = "The")
)]
/// BOOL_OR aggregate expression
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoolOr {
    signature: Signature,
}

impl BoolOr {
    fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Boolean], Volatility::Immutable),
        }
    }
}

impl Default for BoolOr {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregateUDFImpl for BoolOr {
    fn name(&self) -> &str {
        "bool_or"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn accumulator(&self, _: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::<BoolOrAccumulator>::default())
    }

    fn create_sliding_accumulator(
        &self,
        _: AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(RecomputingFrameAccumulator::new(
            BoolOrAccumulator::default(),
        )))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Field::new(
                format_state_name(args.name, self.name()),
                DataType::Boolean,
                true,
            )
            .into(),
        ])
    }

    fn groups_accumulator_supported(&self, _args: AccumulatorArgs) -> bool {
        true
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        match args.return_field.data_type() {
            DataType::Boolean => Ok(Box::new(BooleanGroupsAccumulator::new(
                |x, y| x || y,
                false,
            ))),
            _ => not_impl_err!(
                "GroupsAccumulator not supported for {} with {}",
                args.name,
                args.return_field.data_type()
            ),
        }
    }

    fn order_sensitivity(&self) -> AggregateOrderSensitivity {
        AggregateOrderSensitivity::Insensitive
    }

    fn reverse_expr(&self) -> ReversedUDAF {
        ReversedUDAF::Identical
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn distinct_handling(&self) -> DistinctHandling {
        // Boolean AND/OR are idempotent: duplicates cannot change the result.
        DistinctHandling::Insensitive
    }
}

#[derive(Debug, Default)]
struct BoolOrAccumulator {
    acc: Option<bool>,
}

impl Accumulator for BoolOrAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = &values[0];
        self.acc = match (self.acc, bool_or_batch(values)?) {
            (None, ScalarValue::Boolean(v)) => v,
            (Some(v), ScalarValue::Boolean(None)) => Some(v),
            (Some(a), ScalarValue::Boolean(Some(b))) => Some(a || b),
            _ => unreachable!(),
        };
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::Boolean(self.acc))
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Boolean(self.acc)])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.update_batch(states)
    }
}

impl RecomputeFrame for BoolOrAccumulator {
    fn recompute(
        &mut self,
        values: &[ArrayRef],
        range: Range<usize>,
        filter: Option<&BooleanArray>,
    ) -> Result<()> {
        self.acc = recompute_bool(values, range, filter, false)?;
        Ok(())
    }
}

#[cfg(test)]
mod sliding_tests;
