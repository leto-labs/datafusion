// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

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
//! wire logic. Indexed scans store file metadata once and retain group and
//! range order separately.
//!
//! Child physical expressions (sort orderings, hash/range partitioning, and
//! projection expressions) are (de)serialized through `ctx.encode_expr` /
//! `ctx.decode_expr`; `Schema`, `Statistics`, `Constraints`, and `ScalarValue`
//! go through `datafusion-proto-common`. Nothing here needs the raw codec.

use std::sync::Arc;

use arrow::datatypes::Schema;
use datafusion_common::parsers::CompressionTypeVariant;
use datafusion_common::utils::{usize_from_wire, usize_to_wire};
use datafusion_common::{DataFusionError, Result, internal_datafusion_err};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr::projection::{ProjectionExpr, ProjectionExprs};
use datafusion_physical_expr_common::sort_expr::{
    optional_ordering_try_from_proto, sort_exprs_try_to_proto,
};
use datafusion_physical_plan::proto::{ExecutionPlanDecodeCtx, ExecutionPlanEncodeCtx};
use datafusion_proto_models::datafusion_common::{
    CompressionTypeVariant as ProtoCompressionTypeVariant, Schema as ProtoSchema,
};
use datafusion_proto_models::protobuf;

use crate::file::FileSource;
use crate::file_compression_type::FileCompressionType;
use crate::file_groups::FileGroup;
use crate::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use crate::table_schema::TableSchema;

mod indexed_files;

impl FileScanConfig {
    /// Serialize the shared, format-agnostic part of a file scan into a
    /// [`protobuf::FileScanExecConf`].
    ///
    /// Each concrete [`FileSource::try_to_proto`]
    /// wraps the returned value in its own `*ScanExecNode`.
    pub fn try_to_proto(
        &self,
        ctx: &ExecutionPlanEncodeCtx<'_>,
    ) -> Result<protobuf::FileScanExecConf> {
        // Exhaustive destructure: adding a field to `FileScanConfig` without
        // deciding how it is serialized is a compile error, not a silent
        // round-trip gap.
        let Self {
            object_store_url,
            file_groups,
            constraints,
            limit,
            preserve_order,
            output_ordering,
            file_compression_type,
            file_source,
            batch_size,
            expr_adapter_factory,
            // Serialized through `statistics()` so its pushed-filter policy
            // remains centralized.
            statistics: _,
            output_partitioning,
        } = self;

        // Non-default factories are executable behavior with no protobuf
        // representation; silently replacing one can change scan results.
        if expr_adapter_factory
            .as_ref()
            .is_some_and(|factory| !factory.is_equivalent_to_default())
        {
            return datafusion_common::not_impl_err!(
                "FileScanConfig with a non-default expr_adapter_factory cannot be serialized"
            );
        }

        let indexed_files = indexed_files::encode(file_groups, &self.statistics())?;

        let mut proto_output_ordering = Vec::with_capacity(output_ordering.len());
        for order in output_ordering {
            let nodes = sort_exprs_try_to_proto(order.iter(), &ctx.expr_ctx())?;
            proto_output_ordering.push(protobuf::PhysicalSortExprNodeCollection {
                physical_sort_expr_nodes: nodes,
            });
        }

        let proto_output_partitioning = output_partitioning
            .as_ref()
            .map(|partitioning| partitioning.try_to_proto(&ctx.expr_ctx()))
            .transpose()?;

        let table_schema = file_source.table_schema();
        let file_schema = table_schema.file_schema();
        let table_partition_cols = table_schema.table_partition_cols();

        // Partition fields must be added to the schema so they can persist in
        // protobuf and then be removed again in `parse_table_schema_from_proto`.
        // Virtual fields follow them as an explicitly counted suffix.
        let mut fields = file_schema.fields().to_vec();
        fields.extend(table_partition_cols.iter().cloned());
        let virtual_columns = table_schema.virtual_columns();
        fields.extend(virtual_columns.iter().cloned());
        let virtual_column_count = usize_to_wire::<u32>(
            virtual_columns.len(),
            "FileScanConfig",
            "virtual_columns",
        )?;
        let schema = Schema::new(fields).with_metadata(file_schema.metadata.clone());

        let projection_exprs = file_source
            .projection()
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

        let proto_file_compression_type =
            file_compression_type.is_compressed().then(|| {
                let compression: ProtoCompressionTypeVariant =
                    (*file_compression_type.get_variant()).into();
                compression as i32
            });
        Ok(protobuf::FileScanExecConf {
            indexed_files: Some(indexed_files),
            file_groups: vec![],
            statistics: None,
            limit: limit
                .map(|limit| usize_to_wire::<u32>(limit, "FileScanConfig", "limit"))
                .transpose()?
                .map(|limit| protobuf::ScanLimit { limit }),
            // Superseded by `projection_exprs`; kept empty for wire compatibility.
            projection: vec![],
            schema: Some((&schema).try_into()?),
            table_partition_cols: table_partition_cols
                .iter()
                .map(|x| x.name().clone())
                .collect::<Vec<_>>(),
            object_store_url: object_store_url.to_string(),
            output_ordering: proto_output_ordering,
            constraints: Some(constraints.clone().into()),
            batch_size: batch_size.map(|size| size as u64),
            projection_exprs,
            output_partitioning: proto_output_partitioning,
            file_compression_type: proto_file_compression_type,
            preserve_order: Some(*preserve_order),
            virtual_column_count,
        })
    }

    /// Reconstruct a [`FileScanConfig`] from a [`protobuf::FileScanExecConf`]
    /// and a `file_source` the caller has already rebuilt (typically from the
    /// table schema via [`FileScanConfig::parse_table_schema_from_proto`]).
    ///
    /// Accepts legacy files and scan statistics when no indexed files are present.
    pub fn try_from_proto(
        conf: &protobuf::FileScanExecConf,
        ctx: &ExecutionPlanDecodeCtx<'_>,
        file_source: Arc<dyn FileSource>,
    ) -> Result<FileScanConfig> {
        // Destructure exhaustively so a newly added protobuf field must be
        // handled here instead of being silently ignored.
        let protobuf::FileScanExecConf {
            file_groups,
            indexed_files,
            schema: proto_schema,
            // Superseded by `projection_exprs`; current encoders leave this
            // legacy column-index projection empty.
            projection: _,
            limit,
            statistics,
            // Used to construct `file_source` via `parse_table_schema_from_proto`
            // before this hook is called.
            table_partition_cols: _,
            object_store_url,
            output_ordering,
            constraints,
            batch_size,
            projection_exprs,
            output_partitioning,
            file_compression_type,
            preserve_order,
            // Used to construct `file_source` via `parse_table_schema_from_proto`.
            virtual_column_count: _,
        } = conf;

        let expression_schema = parse_file_scan_schema(proto_schema.as_ref())?;

        let decoded_constraints = constraints
            .as_ref()
            .ok_or_else(|| {
                internal_datafusion_err!(
                    "FileScanExecConf is missing required field 'constraints'"
                )
            })?
            .try_into()?;
        let (decoded_file_groups, decoded_statistics) = match indexed_files {
            Some(indexed) => {
                if !file_groups.is_empty() || statistics.is_some() {
                    return datafusion_common::plan_err!(
                        "FileScanExecConf mixes indexed and legacy file groups or statistics"
                    );
                }
                indexed_files::decode(indexed)?
            }
            None => {
                let statistics = statistics.as_ref().ok_or_else(|| {
                    internal_datafusion_err!(
                        "FileScanExecConf is missing required field 'statistics'"
                    )
                })?;
                let groups = file_groups
                    .iter()
                    .map(TryInto::try_into)
                    .collect::<Result<Vec<FileGroup>>>()?;
                (groups, statistics.try_into()?)
            }
        };

        let decoded_object_store_url = match object_store_url.is_empty() {
            false => ObjectStoreUrl::parse(object_store_url)?,
            true => ObjectStoreUrl::local_filesystem(),
        };

        let mut decoded_output_ordering = Vec::with_capacity(output_ordering.len());
        for node_collection in output_ordering {
            let protobuf::PhysicalSortExprNodeCollection {
                physical_sort_expr_nodes,
            } = node_collection;
            if let Some(ordering) = optional_ordering_try_from_proto(
                physical_sort_expr_nodes,
                &ctx.expr_ctx(&expression_schema),
            )? {
                decoded_output_ordering.push(ordering);
            }
        }

        let decoded_output_partitioning = output_partitioning
            .as_ref()
            .map(|partitioning| {
                Partitioning::try_from_proto(
                    partitioning,
                    &ctx.expr_ctx(&expression_schema),
                )
            })
            .transpose()?
            .flatten();

        let decoded_file_compression_type = file_compression_type
            .map(|value| {
                let compression =
                    ProtoCompressionTypeVariant::try_from(value).map_err(|_| {
                        internal_datafusion_err!("Unknown file compression type: {value}")
                    })?;
                let compression: CompressionTypeVariant = compression.into();
                Ok::<_, DataFusionError>(FileCompressionType::from(compression))
            })
            .transpose()?
            .unwrap_or(FileCompressionType::UNCOMPRESSED);

        // Parse projection expressions if present and apply to the file source.
        let decoded_file_source = if let Some(proto_projection_exprs) = projection_exprs {
            let protobuf::ProjectionExprs { projections } = proto_projection_exprs;
            let decoded_projection_exprs: Vec<ProjectionExpr> = projections
                .iter()
                .map(|proto_expr| {
                    let protobuf::ProjectionExpr { alias, expr } = proto_expr;
                    let expr = ctx.decode_expr(
                        expr.as_ref().ok_or_else(|| {
                            internal_datafusion_err!("ProjectionExpr missing expr field")
                        })?,
                        &expression_schema,
                    )?;
                    Ok(ProjectionExpr::new(expr, alias.clone()))
                })
                .collect::<Result<Vec<_>>>()?;

            let projection = ProjectionExprs::new(decoded_projection_exprs);

            file_source
                .try_pushdown_projection(&projection)?
                .unwrap_or(file_source)
        } else {
            file_source
        };

        let decoded_limit = limit
            .as_ref()
            .map(|limit| {
                let protobuf::ScanLimit { limit } = limit;
                usize_from_wire(*limit, "FileScanConfig", "limit")
            })
            .transpose()?;
        let decoded_batch_size = batch_size
            .map(|size| usize_from_wire(size, "FileScanConfig", "batch_size"))
            .transpose()?;
        if decoded_batch_size == Some(0) {
            return datafusion_common::plan_err!(
                "FileScanConfig: batch_size must be greater than 0"
            );
        }
        let mut config =
            FileScanConfigBuilder::new(decoded_object_store_url, decoded_file_source)
                .with_file_groups(decoded_file_groups)
                .with_constraints(decoded_constraints)
                .with_statistics(decoded_statistics)
                .with_limit(decoded_limit)
                .with_output_ordering(decoded_output_ordering)
                .with_output_partitioning(decoded_output_partitioning)
                .with_batch_size(decoded_batch_size)
                .with_file_compression_type(decoded_file_compression_type)
                .build();

        // Presence distinguishes a new explicit `false` from a legacy payload,
        // which must retain the builder's ordering-derived behavior.
        if let Some(preserve_order) = preserve_order {
            config.preserve_order = *preserve_order;
        }
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
        let schema = parse_file_scan_schema(conf.schema.as_ref())?;

        // Roles come from the declared table layout, never extension metadata:
        // an extension-typed physical field is still stored in the file.
        let virtual_count = usize_from_wire(
            conf.virtual_column_count,
            "FileScanConfig",
            "virtual_column_count",
        )?;
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
fn parse_file_scan_schema(schema: Option<&ProtoSchema>) -> Result<Arc<Schema>> {
    let proto_schema = schema.ok_or_else(|| {
        internal_datafusion_err!("FileScanExecConf is missing required field 'schema'")
    })?;
    Ok(Arc::new(proto_schema.try_into()?))
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
