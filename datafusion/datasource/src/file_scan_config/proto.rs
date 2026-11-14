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

//! Shared serialization of the format-agnostic [`FileScanConfig`] spine.
//!
//! This is the relocated body of `datafusion-proto`'s
//! `serialize_file_scan_config` / `parse_protobuf_file_scan_config`, ported to
//! ride the
//! [`ExecutionPlanEncodeCtx`](datafusion_physical_plan::proto::ExecutionPlanEncodeCtx) /
//! [`ExecutionPlanDecodeCtx`](datafusion_physical_plan::proto::ExecutionPlanDecodeCtx)
//! instead of the raw `PhysicalExtensionCodec` +
//! `PhysicalProtoConverterExtension`. Every
//! `FileSource::try_to_proto` hook (CSV, JSON, Arrow, Parquet, Avro) builds its
//! `*ScanExecNode` around [`FileScanConfig::try_to_proto`] and decodes with
//! [`FileScanConfig::try_from_proto`], keeping a single copy of the shared
//! wire logic. Virtual fields are carried explicitly as a schema suffix.
//!
//! Child physical expressions (sort orderings, hash/range partitioning, and
//! projection expressions) are (de)serialized through `ctx.encode_expr` /
//! `ctx.decode_expr`; `Schema`, `Statistics`, `Constraints`, and `ScalarValue`
//! go through `datafusion-proto-common`. Nothing here needs the raw codec.

use std::sync::Arc;

use arrow::datatypes::Schema;
use datafusion_common::{DataFusionError, Result, internal_datafusion_err};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_physical_expr::projection::{ProjectionExpr, ProjectionExprs};
use datafusion_physical_expr::{LexOrdering, Partitioning};
use datafusion_physical_expr_common::sort_expr::{
    sort_exprs_try_from_proto, sort_exprs_try_to_proto,
};
use datafusion_physical_plan::proto::{ExecutionPlanDecodeCtx, ExecutionPlanEncodeCtx};
use datafusion_proto_models::protobuf;

use crate::file::FileSource;
use crate::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use crate::table_schema::TableSchema;

impl FileScanConfig {
    /// Serialize the shared, format-agnostic part of a file scan into a
    /// [`protobuf::FileScanExecConf`].
    ///
    /// Each concrete [`FileSource::try_to_proto`]
    /// wraps the returned value in its own `*ScanExecNode`. Virtual columns are
    /// appended after file and partition fields and counted explicitly.
    pub fn try_to_proto(
        &self,
        ctx: &ExecutionPlanEncodeCtx<'_>,
    ) -> Result<protobuf::FileScanExecConf> {
        let file_groups = self
            .file_groups
            .iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>>>()?;

        let mut output_ordering = vec![];
        for order in &self.output_ordering {
            let nodes = sort_exprs_try_to_proto(order.iter(), &ctx.expr_ctx())?;
            output_ordering.push(protobuf::PhysicalSortExprNodeCollection {
                physical_sort_expr_nodes: nodes,
            });
        }

        let output_partitioning = self
            .output_partitioning
            .as_ref()
            .map(|partitioning| partitioning.try_to_proto(&ctx.expr_ctx()))
            .transpose()?;

        // Fields must be added to the schema so that they can persist in the
        // protobuf, and then removed from the schema in `try_from_proto`.
        let mut fields = self
            .file_schema()
            .fields()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        fields.extend(self.table_partition_cols().iter().cloned());
        let virtual_columns = self.file_source().table_schema().virtual_columns();
        fields.extend(virtual_columns.iter().cloned());
        let virtual_column_count = u32::try_from(virtual_columns.len())
            .map_err(|_| internal_datafusion_err!("too many virtual scan columns"))?;
        let schema =
            Schema::new(fields).with_metadata(self.file_schema().metadata.clone());

        let projection_exprs = self
            .file_source()
            .projection()
            .as_ref()
            .map(|projection_exprs| {
                Ok::<_, DataFusionError>(protobuf::ProjectionExprs {
                    projections: projection_exprs
                        .iter()
                        .map(|expr| {
                            Ok(protobuf::ProjectionExpr {
                                alias: expr.alias.to_string(),
                                expr: Some(ctx.encode_expr(&expr.expr)?),
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                })
            })
            .transpose()?;

        Ok(protobuf::FileScanExecConf {
            file_groups,
            statistics: Some((&self.statistics()).into()),
            limit: self
                .limit
                .map(|limit| {
                    Ok::<_, DataFusionError>(protobuf::ScanLimit {
                        limit: u64::try_from(limit).map_err(|_| {
                            internal_datafusion_err!("scan limit exceeds uint64")
                        })?,
                    })
                })
                .transpose()?,
            projection: vec![],
            schema: Some((&schema).try_into()?),
            table_partition_cols: self
                .table_partition_cols()
                .iter()
                .map(|x| x.name().clone())
                .collect::<Vec<_>>(),
            object_store_url: self.object_store_url.to_string(),
            output_ordering,
            constraints: Some(self.constraints.clone().into()),
            batch_size: self.batch_size.map(|s| s as u64),
            projection_exprs,
            output_partitioning,
            virtual_column_count,
            preserve_order: self.preserve_order,
        })
    }

    /// Reconstruct a [`FileScanConfig`] from a [`protobuf::FileScanExecConf`]
    /// and a `file_source` the caller has already rebuilt (typically from the
    /// table schema via [`FileScanConfig::parse_table_schema_from_proto`]).
    ///
    pub fn try_from_proto(
        conf: &protobuf::FileScanExecConf,
        ctx: &ExecutionPlanDecodeCtx<'_>,
        file_source: Arc<dyn FileSource>,
    ) -> Result<FileScanConfig> {
        let schema = parse_file_scan_schema(conf)?;

        let constraints = conf
            .constraints
            .as_ref()
            .ok_or_else(|| {
                internal_datafusion_err!(
                    "FileScanExecConf is missing required field 'constraints'"
                )
            })?
            .try_into()?;
        let statistics = conf
            .statistics
            .as_ref()
            .ok_or_else(|| {
                internal_datafusion_err!(
                    "FileScanExecConf is missing required field 'statistics'"
                )
            })?
            .try_into()?;

        let file_groups = conf
            .file_groups
            .iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>>>()?;

        let object_store_url = match conf.object_store_url.is_empty() {
            false => ObjectStoreUrl::parse(&conf.object_store_url)?,
            true => ObjectStoreUrl::local_filesystem(),
        };

        let mut output_ordering = vec![];
        for node_collection in &conf.output_ordering {
            let sort_exprs = sort_exprs_try_from_proto(
                &node_collection.physical_sort_expr_nodes,
                &ctx.expr_ctx(&schema),
            )?;
            output_ordering.extend(LexOrdering::new(sort_exprs));
        }

        let output_partitioning = conf
            .output_partitioning
            .as_ref()
            .map(|partitioning| {
                Partitioning::try_from_proto(partitioning, &ctx.expr_ctx(&schema))
            })
            .transpose()?
            .flatten();

        // Parse projection expressions if present and apply to the file source.
        let file_source = if let Some(proto_projection_exprs) = &conf.projection_exprs {
            let projection_exprs: Vec<ProjectionExpr> = proto_projection_exprs
                .projections
                .iter()
                .map(|proto_expr| {
                    let expr = ctx.decode_expr(
                        proto_expr.expr.as_ref().ok_or_else(|| {
                            internal_datafusion_err!("ProjectionExpr missing expr field")
                        })?,
                        &schema,
                    )?;
                    Ok(ProjectionExpr::new(expr, proto_expr.alias.clone()))
                })
                .collect::<Result<Vec<_>>>()?;

            let projection_exprs = ProjectionExprs::new(projection_exprs);

            file_source
                .try_pushdown_projection(&projection_exprs)?
                .unwrap_or(file_source)
        } else {
            file_source
        };

        let config_builder = FileScanConfigBuilder::new(object_store_url, file_source)
            .with_file_groups(file_groups)
            .with_constraints(constraints)
            .with_statistics(statistics)
            .with_limit(
                conf.limit
                    .as_ref()
                    .map(|limit| {
                        usize::try_from(limit.limit).map_err(|_| {
                            internal_datafusion_err!("scan limit exceeds native usize")
                        })
                    })
                    .transpose()?,
            )
            .with_output_ordering(output_ordering)
            .with_output_partitioning(output_partitioning)
            .with_batch_size(conf.batch_size.map(|s| s as usize));
        let mut config = config_builder.build();
        // build() derives true from nonempty ordering. Preserve the raw encoded
        // value after construction, including an explicitly ordered false.
        config.preserve_order = conf.preserve_order;
        Ok(config)
    }

    /// Parse a [`TableSchema`] (file, partition and virtual columns) from a
    /// [`protobuf::FileScanExecConf`]. File sources use this to rebuild their
    /// concrete source before calling [`FileScanConfig::try_from_proto`].
    ///
    /// Byte-compatible with the former `parse_table_schema_from_proto`.
    pub fn parse_table_schema_from_proto(
        conf: &protobuf::FileScanExecConf,
    ) -> Result<TableSchema> {
        let schema = parse_file_scan_schema(conf)?;

        // Roles come from the declared table layout, never extension metadata:
        // an extension-typed physical field is still stored in the file.
        let virtual_count = conf.virtual_column_count as usize;
        let virtual_start = schema
            .fields()
            .len()
            .checked_sub(virtual_count)
            .ok_or_else(|| {
                internal_datafusion_err!("virtual scan column count exceeds schema width")
            })?;
        let virtual_columns = schema.fields()[virtual_start..].to_vec();

        // Reacquire the partition column types from the schema before removing
        // them below.
        let table_partition_cols = conf
            .table_partition_cols
            .iter()
            .map(|col| Ok(Arc::new(schema.field_with_name(col)?.clone())))
            .collect::<Result<Vec<_>>>()?;

        // Remove partition columns from the schema after recreating
        // table_partition_cols because the partition columns are not in the
        // file. They are present to allow the partition column types to be
        // reconstructed after serde.
        let file_schema = Arc::new(
            Schema::new(
                schema
                    .fields()
                    .iter()
                    .take(virtual_start)
                    .filter(|field| !table_partition_cols.contains(field))
                    .cloned()
                    .collect::<Vec<_>>(),
            )
            .with_metadata(schema.metadata.clone()),
        );

        if table_partition_cols
            .iter()
            .any(|field| virtual_columns.contains(field))
        {
            return Err(internal_datafusion_err!(
                "scan column cannot be both virtual and partitioned"
            ));
        }
        Ok(TableSchema::builder(file_schema)
            .with_table_partition_cols(table_partition_cols)
            .with_virtual_columns(virtual_columns)
            .build())
    }
}

/// Parse the full (file, partition and virtual columns) schema off the base conf.
fn parse_file_scan_schema(conf: &protobuf::FileScanExecConf) -> Result<Arc<Schema>> {
    let schema: Schema = conf
        .schema
        .as_ref()
        .ok_or_else(|| {
            internal_datafusion_err!(
                "FileScanExecConf is missing required field 'schema'"
            )
        })?
        .try_into()?;
    Ok(Arc::new(schema))
}

#[cfg(test)]
mod virtual_column_tests {
    use super::*;
    use arrow::datatypes::{DataType, Field};
    use std::collections::HashMap;

    fn extension_field(name: &str) -> Arc<Field> {
        Arc::new(
            Field::new(name, DataType::Int64, false).with_metadata(HashMap::from([(
                "ARROW:extension:name".to_owned(),
                "arrow.parquet.row_number".to_owned(),
            )])),
        )
    }

    #[test]
    fn explicit_virtual_suffix_preserves_physical_extensions_and_partitions() {
        let stored = extension_field("stored_position");
        let value = Arc::new(Field::new("value", DataType::Utf8, true));
        let partition = Arc::new(Field::new("file", DataType::UInt32, false));
        let row = extension_field("row");
        let metadata = HashMap::from([("source".to_owned(), "fixture".to_owned())]);
        let schema = Schema::new(vec![
            Arc::clone(&stored),
            Arc::clone(&value),
            Arc::clone(&partition),
            Arc::clone(&row),
        ])
        .with_metadata(metadata.clone());
        let conf = protobuf::FileScanExecConf {
            schema: Some((&schema).try_into().unwrap()),
            table_partition_cols: vec!["file".to_owned()],
            virtual_column_count: 1,
            ..Default::default()
        };
        let table = FileScanConfig::parse_table_schema_from_proto(&conf).unwrap();
        assert_eq!(
            table.file_schema().as_ref(),
            &Schema::new(vec![stored, value]).with_metadata(metadata)
        );
        assert_eq!(table.table_partition_cols().as_ref(), &[partition]);
        assert_eq!(table.virtual_columns().as_ref(), &[row]);
        assert_eq!(table.table_schema().as_ref(), &schema);
    }

    #[test]
    fn physical_extension_is_not_inferred_as_virtual() {
        let schema = Schema::new(vec![extension_field("stored_position")]);
        let conf = protobuf::FileScanExecConf {
            schema: Some((&schema).try_into().unwrap()),
            ..Default::default()
        };
        let table = FileScanConfig::parse_table_schema_from_proto(&conf).unwrap();
        assert_eq!(table.file_schema().as_ref(), &schema);
        assert!(table.virtual_columns().is_empty());
    }

    #[test]
    fn invalid_virtual_suffix_and_overlapping_roles_are_rejected() {
        let schema = Schema::new(vec![extension_field("row")]);
        let mut conf = protobuf::FileScanExecConf {
            schema: Some((&schema).try_into().unwrap()),
            virtual_column_count: 2,
            ..Default::default()
        };
        assert!(FileScanConfig::parse_table_schema_from_proto(&conf).is_err());
        conf.virtual_column_count = 1;
        conf.table_partition_cols.push("row".to_owned());
        assert!(FileScanConfig::parse_table_schema_from_proto(&conf).is_err());
    }
}
