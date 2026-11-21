// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Mutex;

use arrow::datatypes::Field;
use datafusion_physical_expr::expressions::lit;
use datafusion_physical_expr::projection::ProjectionExpr;
use object_store::ObjectStore;

use super::*;
use crate::{TableSchema, file_stream::FileOpener, test_util::MockSource};

#[derive(Clone)]
struct Capture {
    source: MockSource,
    projection: ProjectionExprs,
    seen: Arc<Mutex<Vec<Arc<dyn PhysicalExpr>>>>,
}

impl FileSource for Capture {
    fn create_file_opener(
        &self,
        store: Arc<dyn ObjectStore>,
        config: &FileScanConfig,
        partition: usize,
    ) -> Result<Arc<dyn FileOpener>> {
        self.source.create_file_opener(store, config, partition)
    }

    fn table_schema(&self) -> &TableSchema {
        self.source.table_schema()
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        Some(&self.projection)
    }

    fn with_batch_size(&self, _batch_size: usize) -> Arc<dyn FileSource> {
        Arc::new(self.clone())
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        self.source.metrics()
    }

    fn file_type(&self) -> &str {
        self.source.file_type()
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        self.source.apply_expressions(f)
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        let statuses = vec![PushedDown::Yes; filters.len()];
        *self.seen.lock().unwrap() = filters;
        Ok(
            FilterPushdownPropagation::with_parent_pushdown_result(statuses)
                .with_updated_node(Arc::new(self.clone())),
        )
    }
}

fn eq(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(
        Arc::new(Column::new(name, index)),
        Operator::Eq,
        lit(2_i32),
    ))
}

#[test]
fn ambiguous_file_and_table_fields_refuse_only_unproven_bindings() {
    let field = |name| Field::new(name, DataType::Int32, false);
    for table in [
        TableSchema::from(Arc::new(Schema::new(vec![
            field("k"),
            field("u"),
            field("k"),
        ]))),
        TableSchema::builder(Arc::new(Schema::new(vec![field("k"), field("u")])))
            .with_table_partition_cols(vec![Arc::new(field("k"))])
            .build(),
    ] {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let projection = ProjectionExprs::new(vec![
            ProjectionExpr::new(Arc::new(Column::new("k", 2)), "chosen"),
            ProjectionExpr::new(Arc::new(Column::new("u", 1)), "independent"),
            ProjectionExpr::new(Arc::new(Column::new("k", 0)), "other"),
        ]);
        let scan = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(Capture {
                source: MockSource::new(table),
                projection,
                seen: Arc::clone(&seen),
            }),
        )
        .build();
        let result = scan
            .try_pushdown_filters(
                vec![
                    eq("chosen", 0),
                    eq("independent", 1),
                    eq("other", 2),
                    eq("unbound", 3),
                ],
                &ConfigOptions::default(),
            )
            .unwrap();
        assert!(matches!(
            result.filters.as_slice(),
            [
                PushedDown::No,
                PushedDown::Yes,
                PushedDown::No,
                PushedDown::No
            ]
        ));
        assert!(result.updated_node.is_some());
        let sent = seen.lock().unwrap();
        assert_eq!(sent.len(), 1);
        let columns = collect_columns(&sent[0]);
        assert_eq!(columns.len(), 1);
        assert!(columns.contains(&Column::new("u", 1)));
    }
}
