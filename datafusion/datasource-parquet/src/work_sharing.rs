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

//! Cooperative tail donation at native row-group boundaries. The owner keeps
//! its reader, decoder buffers and read-ahead window. Only disjoint whole row
//! groups beyond that window become independent streams on idle partitions.

use std::collections::VecDeque;
use std::sync::Arc;

use datafusion_common::Result;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_stream::work_source::SharedWorkSource;
use datafusion_physical_expr::expressions::DynamicFilterTracking;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::push_decoder::ParquetPushDecoder;
use parquet::file::metadata::ParquetMetaData;

use crate::push_decoder::RgPlanEntry;
use crate::{ParquetAccessPlan, ParquetFileMetrics};

pub(crate) struct RowGroupSharing {
    pub(crate) source: SharedWorkSource,
    pub(crate) file: PartitionedFile,
    pub(crate) plan: ParquetAccessPlan,
    pub(crate) metadata: Arc<ParquetMetaData>,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) lookahead: usize,
    pub(crate) has_selection: bool,
    pub(crate) read_ahead_enabled: bool,
    pub(crate) metrics: ExecutionPlanMetricsSet,
    pub(crate) partition: usize,
}

impl RowGroupSharing {
    fn parts(&self, remaining: usize) -> usize {
        // Preserve a useful stream per borrower too. This bounds extra file
        // opens and avoids the one-row-group stream/read-ahead failure of LM-D.
        self.source
            .split_width(remaining / self.lookahead.saturating_add(1))
    }

    pub(crate) fn wants_work(&self, remaining: usize) -> bool {
        self.parts(remaining) > 1
            && self.predicate.as_ref().is_none_or(|predicate| {
                !matches!(
                    DynamicFilterTracking::classify(predicate),
                    DynamicFilterTracking::Watching(_)
                )
            })
    }

    /// Called only at a decoder boundary, after its frontier and runtime
    /// pruning have updated `remaining`. Rebuild before publishing: a failed
    /// rebuild cannot expose work that still belongs to the original decoder.
    pub(crate) fn offer(
        &self,
        decoder: ParquetPushDecoder,
        remaining: &mut VecDeque<RgPlanEntry>,
        reader: &mut dyn AsyncFileReader,
    ) -> Result<ParquetPushDecoder> {
        let parts = self.parts(remaining.len());
        if parts < 2 {
            return Ok(decoder);
        }
        let indices: Vec<_> = remaining.iter().map(|entry| entry.rg_index).collect();
        let local_len = indices
            .len()
            .div_ceil(parts)
            .max(self.lookahead.saturating_add(1));
        let local = self.plan.select_row_groups(&indices[..local_len]);
        let published: Vec<_> = self
            .plan
            .select_row_groups(&indices[local_len..])
            .split(parts - 1)
            .into_iter()
            .map(|plan| self.file.clone().with_extension(plan))
            .collect();
        let local = local.prepare(self.metadata.row_groups())?;
        let mut builder = decoder
            .into_builder()?
            .with_row_groups(local.row_group_indexes);
        if self.has_selection {
            // into_builder carries a flat selection over the old tail. Replace
            // it with the selection over only the retained groups, including
            // an all-selected plan if all partial groups were donated.
            let selection = local.row_selection.unwrap_or_else(|| {
                RowSelection::from(vec![RowSelector::select(
                    indices[..local_len]
                        .iter()
                        .map(|&index| self.metadata.row_group(index).num_rows() as usize)
                        .sum(),
                )])
            });
            builder = builder.with_row_selection(selection);
        }
        if self.read_ahead_enabled {
            builder.configure_read_ahead(reader);
        }
        let decoder = builder.build()?;
        remaining.truncate(local_len);
        ParquetFileMetrics::add_row_group_units_published(
            &self.metrics,
            self.partition,
            self.file.object_meta.location.as_ref(),
            published.len(),
        );
        self.source.publish(published);
        Ok(decoder)
    }
}
