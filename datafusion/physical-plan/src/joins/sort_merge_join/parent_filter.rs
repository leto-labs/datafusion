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

//! Positional routes for existing parent dynamic predicates.

use std::sync::Arc;

use super::SortMergeJoinExec;
use crate::ExecutionPlan;
use crate::filter_pushdown::{
    ChildFilterDescription, FilterDescription, PushedDownPredicate,
};
use arrow::datatypes::Schema;
use datafusion_common::Result;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{Column, DynamicFilterPhysicalExpr};

pub(super) fn describe(
    join: &SortMergeJoinExec,
    filters: &[Arc<dyn PhysicalExpr>],
) -> Result<FilterDescription> {
    let output = join.schema();
    let left = join.left().schema();
    let right = join.right().schema();
    let mut description = FilterDescription::new();
    for (input, offset) in [(left.as_ref(), 0), (right.as_ref(), left.fields().len())] {
        let mut child = ChildFilterDescription::all_unsupported(filters);
        for (index, filter) in filters.iter().enumerate() {
            if let Some(mapped) = remap(filter, output.as_ref(), input, offset)? {
                child.parent_filters[index] = PushedDownPredicate::supported(mapped);
            }
        }
        description = description.with_child(child);
    }
    Ok(description)
}

fn remap(
    filter: &Arc<dyn PhysicalExpr>,
    output: &Schema,
    input: &Schema,
    offset: usize,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    if !filter.is::<DynamicFilterPhysicalExpr>() || filter.children().is_empty() {
        return Ok(None);
    }
    let mut mapped = Vec::new();
    for expression in filter.children() {
        let Some(column) = expression.downcast_ref::<Column>() else {
            return Ok(None);
        };
        let Some(index) = column.index().checked_sub(offset) else {
            return Ok(None);
        };
        let Some(field) = input.fields().get(index) else {
            return Ok(None);
        };
        // Native children may remap by name. Keep an ambiguous predicate
        // above the join rather than letting a child select another column.
        if input
            .fields()
            .iter()
            .filter(|f| f.name() == column.name())
            .count()
            != 1
            || field.name() != column.name()
            || output
                .fields()
                .get(column.index())
                .is_none_or(|f| f.name() != column.name())
        {
            return Ok(None);
        }
        mapped.push(Arc::new(Column::new(column.name(), index)) as Arc<dyn PhysicalExpr>);
    }
    // Derive a live view; never copy current() or allocate a new producer.
    Ok(Some(Arc::clone(filter).with_new_children(mapped)?))
}
