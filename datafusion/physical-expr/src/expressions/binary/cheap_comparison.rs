// Modified by the Longbow authors; see third_party/README.md.

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
use crate::expressions::{Column, Literal};

/// A primitive comparison of columns/literals cannot fail for particular rows.
/// Prefer its contiguous vector kernel once the selected column volume
/// justifies avoiding the batch gather and scatter. Do not extend this to casts, arithmetic, UDFs, or
/// other expressions that can error on rows rejected by the left predicate.
pub(super) fn prefer_full_comparison(
    expr: &dyn PhysicalExpr,
    batch: &RecordBatch,
    selected_rows: usize,
) -> bool {
    // Gathering/scattering amortizes at very low selectivity. Use the number
    // of copied column values as a coarse cost estimate, with a 1/16-column
    // floor before paying for a comparison over every row. This also avoids
    // full scans for almost-empty selections on narrow batches.
    if selected_rows
        .saturating_mul(batch.num_columns())
        .saturating_mul(16)
        < batch.num_rows()
    {
        return false;
    }
    let Some(binary) = expr.downcast_ref::<BinaryExpr>() else {
        return false;
    };
    if !matches!(
        binary.op,
        Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
    ) {
        return false;
    }
    let direct_type = |expr: &Arc<dyn PhysicalExpr>| {
        if let Some(column) = expr.downcast_ref::<Column>() {
            batch
                .columns()
                .get(column.index())
                .map(|a| a.data_type().clone())
        } else {
            expr.downcast_ref::<Literal>()
                .map(|l| l.value().data_type())
        }
    };
    match (direct_type(&binary.left), direct_type(&binary.right)) {
        (Some(left), Some(right)) => left == right && left.is_primitive(),
        _ => false,
    }
}
