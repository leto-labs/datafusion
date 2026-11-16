// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`CovarianceSample`]: covariance sample aggregations.

use crate::variance::{SlidingMoment, SlidingMomentQueue};
use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::cast::{as_float64_array, as_uint64_array};
use datafusion_common::{Result, ScalarValue, exec_err, not_impl_err};
use datafusion_expr::DistinctHandling;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, Signature, Volatility,
    function::{AccumulatorArgs, StateFieldsArgs},
    utils::format_state_name,
};
use datafusion_functions_aggregate_common::stats::StatsType;
use datafusion_macros::user_doc;
use std::fmt::Debug;
use std::mem::size_of_val;
use std::sync::Arc;

make_udaf_expr_and_func!(
    CovarianceSample,
    covar_samp,
    y x,
    "Computes the sample covariance.",
    covar_samp_udaf
);

make_udaf_expr_and_func!(
    CovariancePopulation,
    covar_pop,
    y x,
    "Computes the population covariance.",
    covar_pop_udaf
);

#[user_doc(
    doc_section(label = "Statistical Functions"),
    description = "Returns the sample covariance of a set of number pairs.",
    syntax_example = "covar_samp(expression1, expression2)",
    sql_example = r#"```sql
> SELECT covar_samp(column1, column2) FROM table_name;
+-----------------------------------+
| covar_samp(column1, column2)      |
+-----------------------------------+
| 8.25                              |
+-----------------------------------+
```"#,
    standard_argument(name = "expression1", prefix = "First"),
    standard_argument(name = "expression2", prefix = "Second")
)]
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct CovarianceSample {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for CovarianceSample {
    fn default() -> Self {
        Self::new()
    }
}

impl CovarianceSample {
    pub fn new() -> Self {
        Self {
            aliases: vec![String::from("covar")],
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for CovarianceSample {
    fn name(&self) -> &str {
        "covar_samp"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn create_sliding_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        if args.is_distinct {
            return self.accumulator(args);
        }
        Ok(Box::new(SlidingCovarianceAccumulator::new(
            StatsType::Sample,
        )?))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let name = args.name;
        Ok(vec![
            Field::new(format_state_name(name, "count"), DataType::UInt64, true),
            Field::new(format_state_name(name, "mean1"), DataType::Float64, true),
            Field::new(format_state_name(name, "mean2"), DataType::Float64, true),
            Field::new(
                format_state_name(name, "algo_const"),
                DataType::Float64,
                true,
            ),
        ]
        .into_iter()
        .map(Arc::new)
        .collect())
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        if acc_args.is_distinct {
            return not_impl_err!("COVAR_SAMP(DISTINCT) aggregations are not available");
        }
        Ok(Box::new(CovarianceAccumulator::try_new(StatsType::Sample)?))
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn distinct_handling(&self) -> DistinctHandling {
        // The accumulator rejects `DISTINCT` with `not_impl_err!`. It only
        // works when the planner removes it first (`SingleDistinctToGroupBy`
        // for a single distinct field, e.g. `covar_samp(DISTINCT x, x)`).
        DistinctHandling::Unsupported
    }
}

#[user_doc(
    doc_section(label = "Statistical Functions"),
    description = "Returns the population covariance of a set of number pairs.",
    syntax_example = "covar_pop(expression1, expression2)",
    sql_example = r#"```sql
> SELECT covar_pop(column1, column2) FROM table_name;
+-----------------------------------+
| covar_pop(column1, column2)       |
+-----------------------------------+
| 7.63333333333                     |
+-----------------------------------+
```"#,
    standard_argument(name = "expression1", prefix = "First"),
    standard_argument(name = "expression2", prefix = "Second")
)]
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct CovariancePopulation {
    signature: Signature,
}

impl Default for CovariancePopulation {
    fn default() -> Self {
        Self::new()
    }
}

impl CovariancePopulation {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for CovariancePopulation {
    fn name(&self) -> &str {
        "covar_pop"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn create_sliding_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        if args.is_distinct {
            return self.accumulator(args);
        }
        Ok(Box::new(SlidingCovarianceAccumulator::new(
            StatsType::Population,
        )?))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let name = args.name;
        Ok(vec![
            Field::new(format_state_name(name, "count"), DataType::UInt64, true),
            Field::new(format_state_name(name, "mean1"), DataType::Float64, true),
            Field::new(format_state_name(name, "mean2"), DataType::Float64, true),
            Field::new(
                format_state_name(name, "algo_const"),
                DataType::Float64,
                true,
            ),
        ]
        .into_iter()
        .map(Arc::new)
        .collect())
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        if acc_args.is_distinct {
            return not_impl_err!("COVAR_POP(DISTINCT) aggregations are not available");
        }
        Ok(Box::new(CovarianceAccumulator::try_new(
            StatsType::Population,
        )?))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn distinct_handling(&self) -> DistinctHandling {
        // The accumulator rejects `DISTINCT` with `not_impl_err!`. It only
        // works when the planner removes it first (`SingleDistinctToGroupBy`
        // for a single distinct field, e.g. `covar_pop(DISTINCT x, x)`).
        DistinctHandling::Unsupported
    }
}

/// An accumulator to compute covariance
/// The algorithm used is an online implementation and numerically stable. It is derived from the following paper
/// for calculating variance:
/// Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products".
/// Technometrics. 4 (3): 419–420. doi:10.2307/1266577. JSTOR 1266577.
///
/// The algorithm has been analyzed here:
/// Ling, Robert F. (1974). "Comparison of Several Algorithms for Computing Sample Means and Variances".
/// Journal of the American Statistical Association. 69 (348): 859–866. doi:10.2307/2286154. JSTOR 2286154.
///
/// Though it is not covered in the original paper but is based on the same idea, as a result the algorithm is online,
/// parallelize and numerically stable.

#[derive(Debug)]
pub struct CovarianceAccumulator {
    algo_const: f64,
    mean1: f64,
    mean2: f64,
    count: u64,
    stats_type: StatsType,
}

impl CovarianceAccumulator {
    /// Creates a new `CovarianceAccumulator`
    pub fn try_new(s_type: StatsType) -> Result<Self> {
        Ok(Self {
            algo_const: 0_f64,
            mean1: 0_f64,
            mean2: 0_f64,
            count: 0_u64,
            stats_type: s_type,
        })
    }

    pub fn get_count(&self) -> u64 {
        self.count
    }

    pub fn get_mean1(&self) -> f64 {
        self.mean1
    }

    pub fn get_mean2(&self) -> f64 {
        self.mean2
    }

    pub fn get_algo_const(&self) -> f64 {
        self.algo_const
    }
}

impl Accumulator for CovarianceAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            ScalarValue::from(self.count),
            ScalarValue::from(self.mean1),
            ScalarValue::from(self.mean2),
            ScalarValue::from(self.algo_const),
        ])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values1 = as_float64_array(&values[0])?;
        let values2 = as_float64_array(&values[1])?;

        for (value1, value2) in values1.iter().zip(values2) {
            let (Some(value1), Some(value2)) = (value1, value2) else {
                continue;
            };

            let new_count = self.count + 1;
            let delta1 = value1 - self.mean1;
            let new_mean1 = delta1 / new_count as f64 + self.mean1;
            let delta2 = value2 - self.mean2;
            let new_mean2 = delta2 / new_count as f64 + self.mean2;
            let new_c = delta1 * (value2 - new_mean2) + self.algo_const;

            self.count += 1;
            self.mean1 = new_mean1;
            self.mean2 = new_mean2;
            self.algo_const = new_c;
        }

        Ok(())
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values1 = as_float64_array(&values[0])?;
        let values2 = as_float64_array(&values[1])?;

        for (value1, value2) in values1.iter().zip(values2) {
            let (Some(value1), Some(value2)) = (value1, value2) else {
                continue;
            };

            if self.count <= 1 {
                self.count = 0;
                self.mean1 = 0.0;
                self.mean2 = 0.0;
                self.algo_const = 0.0;
                continue;
            }

            let new_count = self.count - 1;
            let delta1 = self.mean1 - value1;
            let new_mean1 = delta1 / new_count as f64 + self.mean1;
            let delta2 = self.mean2 - value2;
            let new_mean2 = delta2 / new_count as f64 + self.mean2;
            let new_c = self.algo_const - delta1 * (new_mean2 - value2);

            self.count -= 1;
            self.mean1 = new_mean1;
            self.mean2 = new_mean2;
            self.algo_const = new_c;
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let counts = as_uint64_array(&states[0])?;
        let means1 = as_float64_array(&states[1])?;
        let means2 = as_float64_array(&states[2])?;
        let cs = as_float64_array(&states[3])?;

        for i in 0..counts.len() {
            let c = counts.value(i);
            if c == 0_u64 {
                continue;
            }
            let new_count = self.count + c;
            let new_mean1 = self.mean1 * self.count as f64 / new_count as f64
                + means1.value(i) * c as f64 / new_count as f64;
            let new_mean2 = self.mean2 * self.count as f64 / new_count as f64
                + means2.value(i) * c as f64 / new_count as f64;
            let delta1 = self.mean1 - means1.value(i);
            let delta2 = self.mean2 - means2.value(i);
            let new_c = self.algo_const
                + cs.value(i)
                + delta1 * delta2 * self.count as f64 * c as f64 / new_count as f64;

            self.count = new_count;
            self.mean1 = new_mean1;
            self.mean2 = new_mean2;
            self.algo_const = new_c;
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let count = match self.stats_type {
            StatsType::Population => self.count,
            StatsType::Sample => {
                if self.count > 0 {
                    self.count - 1
                } else {
                    self.count
                }
            }
        };

        if count == 0 {
            Ok(ScalarValue::Float64(None))
        } else {
            Ok(ScalarValue::Float64(Some(self.algo_const / count as f64)))
        }
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }

    fn supports_retract_batch(&self) -> bool {
        true
    }
}
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PairMoments {
    count: u64,
    mean_x: f64,
    mean_y: f64,
    m2_x: f64,
    m2_y: f64,
    cxy: f64,
}

impl SlidingMoment for PairMoments {
    type Value = (f64, f64);
    fn singleton((x, y): Self::Value) -> Self {
        if x.is_finite() && y.is_finite() {
            Self {
                count: 1,
                mean_x: x,
                mean_y: y,
                ..Self::default()
            }
        } else {
            Self::default()
        }
    }
    fn combine(self, other: Self) -> Self {
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }
        let count = self.count + other.count;
        let ratio = other.count as f64 / count as f64;
        let weight = self.count as f64 * ratio;
        let dx = other.mean_x - self.mean_x;
        let dy = other.mean_y - self.mean_y;
        Self {
            count,
            mean_x: self.mean_x + dx * ratio,
            mean_y: self.mean_y + dy * ratio,
            m2_x: self.m2_x + other.m2_x + dx * dx * weight,
            m2_y: self.m2_y + other.m2_y + dy * dy * weight,
            cxy: self.cxy + other.cxy + dx * dy * weight,
        }
    }
    fn same_value(a: Self::Value, b: Self::Value) -> bool {
        a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()
    }
}

/// Pairwise-complete two-stack moments. Independent axis stacks support REGR
/// statistics when only the other axis is non-finite. Non-finite pairs carry
/// empty partials so FIFO validation also checks their arrival order.
///
/// State is sliding-only: X's four-field state, Y's four-field state, ordered
/// non-NULL pair X values, ordered non-NULL pair Y values. Merges concatenate
/// state rows; ordinary covariance/REGR state layouts remain unchanged.
///
/// A non-finite complete pair makes covariance/co-moments NaN while present,
/// unlike ordinary upstream's insertion-order-dependent +/-inf arithmetic.
#[derive(Debug)]
pub(crate) struct SlidingCovarianceAccumulator {
    pub(crate) x: crate::variance::SlidingVarianceAccumulator,
    pub(crate) y: crate::variance::SlidingVarianceAccumulator,
    pairs: SlidingMomentQueue<PairMoments>,
    stats_type: StatsType,
}

impl SlidingCovarianceAccumulator {
    pub(crate) fn new(stats_type: StatsType) -> Result<Self> {
        Ok(Self {
            x: crate::variance::SlidingVarianceAccumulator::new(StatsType::Population)?,
            y: crate::variance::SlidingVarianceAccumulator::new(StatsType::Population)?,
            pairs: Default::default(),
            stats_type,
        })
    }
    pub(crate) fn count(&self) -> u64 {
        self.pairs.len() as u64
    }
    pub(crate) fn co_moment(&self) -> f64 {
        let moments = self.pairs.aggregate();
        if moments.count != self.count() {
            return f64::NAN;
        }
        let bound = moments.m2_x.clamp(0.0, f64::INFINITY).sqrt()
            * moments.m2_y.clamp(0.0, f64::INFINITY).sqrt();
        if bound.is_nan() {
            f64::NAN
        } else {
            moments.cxy.clamp(-bound, bound)
        }
    }
    fn apply(&mut self, values: &[ArrayRef], retracting: bool) -> Result<()> {
        let x = as_float64_array(&values[0])?;
        let y = as_float64_array(&values[1])?;
        for (x, y) in x.iter().zip(y) {
            let (Some(x), Some(y)) = (x, y) else {
                continue;
            };
            // Validate the complete pair before mutating either axis.
            if retracting {
                self.pairs.pop((x, y))?;
            } else {
                self.pairs.push((x, y));
            }
            self.x.apply(x, retracting)?;
            self.y.apply(y, retracting)?;
        }
        Ok(())
    }
}

impl Accumulator for SlidingCovarianceAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.apply(values, false)
    }
    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.apply(values, true)
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        let count = self.count();
        let denominator = match self.stats_type {
            StatsType::Population => count,
            StatsType::Sample => count.saturating_sub(1),
        };
        Ok(ScalarValue::Float64(
            (denominator > 0).then(|| self.co_moment() / denominator as f64),
        ))
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state = self.x.state()?;
        state.extend(self.y.state()?);
        state.push(crate::variance::sliding_values_state(
            self.pairs.values().map(|(x, _)| x),
        ));
        state.push(crate::variance::sliding_values_state(
            self.pairs.values().map(|(_, y)| y),
        ));
        Ok(state)
    }
    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.x.merge_batch(&states[..4])?;
        self.y.merge_batch(&states[4..8])?;
        for (x, y) in states[8]
            .as_list::<i32>()
            .iter()
            .zip(states[9].as_list::<i32>().iter())
        {
            let (Some(x), Some(y)) = (x, y) else {
                return exec_err!("Sliding covariance pair state must contain both axes");
            };
            let x = as_float64_array(&x)?;
            let y = as_float64_array(&y)?;
            if x.len() != y.len() || x.null_count() != 0 || y.null_count() != 0 {
                return exec_err!(
                    "Sliding covariance finite pair state has mismatched axes or NULLs"
                );
            }
            for pair in x.values().iter().copied().zip(y.values().iter().copied()) {
                self.pairs.push(pair);
            }
        }
        Ok(())
    }
    fn supports_retract_batch(&self) -> bool {
        true
    }
    fn size(&self) -> usize {
        size_of_val(self) + self.x.size() - size_of_val(&self.x) + self.y.size()
            - size_of_val(&self.y)
            + self.pairs.heap_size()
    }
}
