// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Positional projection for scans whose field names are ambiguous.

use std::sync::Arc;

use arrow::datatypes::Schema;
use datafusion_common::metadata::FieldMetadata;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{HashSet, Result, ScalarValue, exec_datafusion_err};
use datafusion_datasource::TableSchema;
use datafusion_physical_expr::expressions::{CastExpr, Column, Literal};
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr_adapter::rewrite::rewrite_input_file_name_in_projection;

pub(crate) fn has_duplicate_names(schema: &Schema) -> bool {
    let mut names: HashSet<&str> = HashSet::default();
    schema
        .fields()
        .iter()
        .any(|field| !names.insert(field.name()))
}

pub(super) fn project(
    projection: ProjectionExprs,
    table: &TableSchema,
    physical: &Schema,
    partitions: &[ScalarValue],
    file_name: &str,
) -> Result<ProjectionExprs> {
    let logical_names = table.table_schema().fields();
    let file_columns = table.file_schema().fields().len();
    let virtual_start = file_columns + table.table_partition_cols().len();
    let projection = projection.try_map_exprs(|expr| {
        expr.transform_up(|expr| {
            let Some(column) = expr.downcast_ref::<Column>() else {
                return Ok(Transformed::no(expr));
            };
            let index = column.index();
            let target = logical_names
                .get(index)
                .filter(|f| f.name() == column.name())
                .ok_or_else(|| {
                    exec_datafusion_err!(
                        "Unbound column {column} in positional Parquet projection"
                    )
                })?;
            if (file_columns..virtual_start).contains(&index) {
                let value = partitions.get(index - file_columns).ok_or_else(|| {
                    exec_datafusion_err!(
                        "Missing partition value for {column} in {file_name}"
                    )
                })?;
                return Ok(Transformed::yes(Arc::new(Literal::new_with_metadata(
                    value.clone(),
                    Some(FieldMetadata::from(target.as_ref())),
                ))));
            }
            if index >= virtual_start {
                let decoder_index = physical.fields().len() + index - virtual_start;
                return Ok(Transformed::yes(Arc::new(Column::new(
                    column.name(),
                    decoder_index,
                ))));
            }
            let source = physical.fields().get(index).ok_or_else(|| {
                exec_datafusion_err!(
                    "Missing positional Parquet column {column} in {file_name}"
                )
            })?;
            // Read the file slot at the original index. Its name may differ in
            // a supplied schema. Preserve the table type and output metadata.
            let bound = Arc::new(Column::new(source.name(), index));
            if source == target {
                return Ok(Transformed::yes(bound));
            }
            Ok(Transformed::yes(Arc::new(CastExpr::new_with_target_field(
                bound,
                Arc::clone(target),
                None,
            ))))
        })
        .data()
    })?;
    rewrite_input_file_name_in_projection(projection, file_name)
}
