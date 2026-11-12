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

//! Prepared file hook: the ordinary opener, stopped before stream construction.
use super::*;
use crate::RowGroupAccess;
use parquet::arrow::arrow_reader::RowFilter;
use parquet::arrow::{
    ProjectionMask,
    arrow_reader::{RowSelection, RowSelector},
    push_decoder::ParquetPushDecoderBuilder,
};
use std::ops::Range;

/// Prepared immutable decode state and the original reader used to open a file.
/// The reader remains on the I/O runtime; `file` can be shared by CPU drivers.
pub struct PreparedParquetScan {
    /// Metadata, pruned access plan and decoder template.
    pub file: Arc<PreparedParquetFile>,
    /// Reader supplied by the scan's configured factory, including cache and identity checks.
    pub reader: Box<dyn AsyncFileReader>,
}

impl PreparedParquetScan {
    /// Load optional page indexes only when a surviving group exceeds the
    /// caller's scheduling threshold. Ordinary row groups add no metadata I/O.
    pub async fn with_page_index_if_oversized(mut self, rows: usize) -> Result<Self> {
        if self.file.metadata.metadata().offset_index().is_none()
            && self.file.groups.iter().any(|&group| {
                self.file.metadata.metadata().row_group(group).num_rows() as usize > rows
            })
        {
            let metadata = load_page_index(
                self.file.metadata.clone(),
                &mut self.reader,
                self.file
                    .options
                    .clone()
                    .with_page_index_policy(PageIndexPolicy::Optional),
            )
            .await?;
            let file = Arc::get_mut(&mut self.file).ok_or_else(|| {
                datafusion_common::exec_datafusion_err!(
                    "page index must be loaded before sharing prepared state"
                )
            })?;
            file.metadata = metadata;
        }
        Ok(self)
    }
}

/// A file after schema adaptation and statistics, page-index and Bloom pruning.
/// No record batch has been decoded. Decoder state is never shared by drivers.
pub struct PreparedParquetFile {
    metadata: ArrowReaderMetadata,
    access: ParquetAccessPlan,
    groups: Vec<usize>,
    projection: DecoderProjection,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    dynamic: bool,
    physical_schema: SchemaRef,
    metrics: ParquetFileMetrics,
    reorder: bool,
    force_selections: bool,
    cache_bytes: Option<usize>,
    read_mask: ProjectionMask,
    pushdown: bool,
    factory: Arc<dyn ParquetFileReaderFactory>,
    partition: usize,
    file: PartitionedFile,
    plan_metrics: ExecutionPlanMetricsSet,
    hint: Option<usize>,
    options: ArrowReaderOptions,
}
impl fmt::Debug for PreparedParquetFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedParquetFile")
            .field("access", &self.access)
            .finish_non_exhaustive()
    }
}
impl PreparedParquetFile {
    /// Create an independent range reader using the opener's configured factory.
    /// Metadata is already prepared and is not loaded by this call.
    pub fn reader(&self) -> Result<Box<dyn AsyncFileReader + Send>> {
        self.factory.create_reader(
            self.partition,
            self.file.clone(),
            self.hint,
            &self.plan_metrics,
        )
    }
    /// Metadata with the opener's coerced schema, including narrow decimals.
    pub fn metadata(&self) -> &ArrowReaderMetadata {
        &self.metadata
    }
    /// Access after all static pruning. Row selections address physical RG rows.
    pub fn access_plan(&self) -> &ParquetAccessPlan {
        &self.access
    }
    /// Physical row groups in the opener's requested order, including reversed
    /// or statistics-based ordering when sort pushdown requested it.
    pub fn row_groups(&self) -> &[usize] {
        &self.groups
    }
    /// Transform a decoded batch with the opener's adapted projection.
    pub fn project(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        self.projection.map(batch)
    }
    /// Build independent decoder state for a physical row interval of one surviving RG.
    /// The interval is intersected with the prepared page/external selection.
    pub fn decoder_builder(
        &self,
        group: usize,
        rows: Range<usize>,
        batch_size: usize,
    ) -> Result<ParquetPushDecoderBuilder> {
        self.decoder_builder_with_filter(group, rows, batch_size, None)
    }
    /// Build a decoder with additional vectorized predicates. These are combined
    /// with the scan predicate; they never replace a predicate pushed into the scan.
    pub fn decoder_builder_with_filter(
        &self,
        group: usize,
        rows: Range<usize>,
        batch_size: usize,
        additional: Option<RowFilter>,
    ) -> Result<ParquetPushDecoderBuilder> {
        let mut builder = self.reset_decoder_builder(
            ParquetPushDecoderBuilder::new_with_metadata(self.metadata.clone())
                .with_projection(self.projection.projection_mask().clone()),
            group,
            rows,
            batch_size,
        )?;
        let mut predicates = additional
            .map(RowFilter::into_predicates)
            .unwrap_or_default();
        for predicate in &predicates {
            if (0..self.metadata.parquet_schema().num_columns()).any(|column| {
                predicate.projection().leaf_included(column)
                    && !self.read_mask.leaf_included(column)
            }) {
                return exec_err!(
                    "runtime filter requested a column outside the prepared scan ranges"
                );
            }
        }
        if let Some(filter) = RowFilterGenerator::new(
            self.pushdown.then_some(self.predicate.as_ref()).flatten(),
            &self.physical_schema,
            self.metadata.metadata(),
            self.reorder,
            &self.metrics,
        )
        .next_filter()
        {
            predicates.extend(filter.into_predicates());
        }
        if !predicates.is_empty() {
            builder = builder.with_row_filter(RowFilter::new(predicates));
        }
        if let Some(bytes) = self.cache_bytes {
            builder = builder.with_max_predicate_cache_size(bytes);
        }
        Ok(builder)
    }
    /// Reuse a driver's decoder template with a fresh unit selection. No encoded
    /// buffers from the previous unit may remain in `builder`.
    pub fn reset_decoder_builder(
        &self,
        builder: ParquetPushDecoderBuilder,
        group: usize,
        rows: Range<usize>,
        batch_size: usize,
    ) -> Result<ParquetPushDecoderBuilder> {
        let Some(rg) = self.metadata.metadata().row_groups().get(group) else {
            return exec_err!("invalid prepared row group");
        };
        let total = rg.num_rows() as usize;
        if rows.start > rows.end || rows.end > total || batch_size == 0 {
            return exec_err!("invalid prepared scan interval or batch size");
        }
        let interval = RowSelection::from(vec![
            RowSelector::skip(rows.start),
            RowSelector::select(rows.len()),
            RowSelector::skip(total - rows.end),
        ]);
        let selection = match &self.access.inner()[group] {
            RowGroupAccess::Skip => RowSelection::from(vec![RowSelector::skip(total)]),
            RowGroupAccess::Scan => interval,
            RowGroupAccess::Selection(base) => base.intersection(&interval),
        };
        let mut builder = builder
            .with_row_groups(vec![group])
            .with_row_selection(selection)
            .with_batch_size(batch_size);
        if self.force_selections {
            builder = builder.with_row_selection_policy(
                parquet::arrow::arrow_reader::RowSelectionPolicy::Selectors,
            );
        }
        Ok(builder)
    }
    /// Encoded ranges for a row group. Includes all predicate and output columns;
    /// Static selections include predicate-cache batch boundaries. Adjacent
    /// pages are one buffer, so a full-column decoder request is also covered.
    /// Fetch once and share these immutable buffers between oversized RG slices.
    pub fn byte_ranges(&self, group: usize, batch_size: usize) -> Vec<Range<u64>> {
        assert!(batch_size > 0, "prepared scan batch size must be positive");
        let meta = self.metadata.metadata();
        let rg = meta.row_group(group);
        let selection = match &self.access.inner()[group] {
            RowGroupAccess::Skip => return vec![],
            RowGroupAccess::Selection(selection) => Some(batch_selection(
                selection,
                batch_size,
                rg.num_rows() as usize,
            )),
            RowGroupAccess::Scan => None,
        };
        let mut ranges = Vec::new();
        for (column, chunk) in rg.columns().iter().enumerate() {
            if !self.read_mask.leaf_included(column) {
                continue;
            }
            let (start, len) = chunk.byte_range();
            if let Some(selection) = &selection
                && let Some(index) = meta
                    .offset_index()
                    .and_then(|i| i.get(group))
                    .and_then(|i| i.get(column))
                    .filter(|index| !index.page_locations().is_empty())
            {
                if let Some(first) = index.page_locations().first() {
                    let page = first.offset as u64;
                    if start < page {
                        ranges.push(start..page);
                    }
                }
                ranges.extend(selection.scan_ranges(index.page_locations()));
            } else {
                ranges.push(start..start + len);
            }
        }
        ranges.sort_unstable_by_key(|r| r.start);
        let mut merged: Vec<Range<u64>> = Vec::new();
        for range in ranges {
            if let Some(last) = merged.last_mut()
                && range.start <= last.end
            {
                last.end = last.end.max(range.end);
            } else {
                merged.push(range);
            }
        }
        merged
    }
    /// Leaf columns that must have offset indexes before slicing an oversized RG.
    pub fn read_projection(&self) -> &ProjectionMask {
        &self.read_mask
    }
    /// Driver-local subscriptions for invalidating a cached decoder predicate.
    pub fn dynamic_filter_tracking(&self) -> Option<DynamicFilterTracking> {
        self.dynamic.then(|| {
            DynamicFilterTracking::classify(
                self.predicate.as_ref().expect("dynamic predicate"),
            )
        })
    }
    /// Recheck live DataFusion dynamic predicates against this row group.
    /// Static predicates were already consumed by the prepared access plan.
    pub fn keep_row_group(&self, group: usize) -> bool {
        let Some(predicate) = &self.predicate else {
            return true;
        };
        if !self.dynamic {
            return true;
        }
        !RowGroupPruner::new(
            Arc::clone(predicate),
            Arc::clone(&self.physical_schema),
            Arc::clone(self.metadata.metadata()),
            Count::new(),
            Count::new(),
            usize::MAX,
        )
        .should_prune(&[group])
    }
}
// Match Arrow's predicate-cache quantization without decoding any values.
// Arrow keeps its equivalent helper private. All filter generations may use
// any prepared read column, so plan this conservative superset for every column.
fn batch_selection(selection: &RowSelection, batch: usize, total: usize) -> RowSelection {
    let mut offset = 0;
    let mut ranges: Vec<Range<usize>> = Vec::new();
    for selector in selection.iter() {
        let end = offset + selector.row_count;
        if !selector.skip {
            let start = offset / batch * batch;
            let end = end.div_ceil(batch).saturating_mul(batch).min(total);
            if let Some(last) = ranges.last_mut()
                && start <= last.end
            {
                last.end = last.end.max(end);
            } else {
                ranges.push(start..end);
            }
        }
        offset = end;
    }
    RowSelection::from_consecutive_ranges(ranges.into_iter(), total)
}
impl crate::source::ParquetSource {
    /// Open and prepare one file using exactly the stream scan's reader factory,
    /// coercions, expression adapter, pruning and virtual-column handling.
    /// `None` means file-level pruning proved that no row can match.
    pub async fn prepare_file(
        &self,
        store: Arc<dyn object_store::ObjectStore>,
        config: &datafusion_datasource::file_scan_config::FileScanConfig,
        file: PartitionedFile,
        partition: usize,
    ) -> Result<Option<PreparedParquetScan>> {
        let morselizer = self.prepared_morselizer(store, config, partition)?;
        let mut state = ParquetMorselPlanner::try_new(&morselizer, file)?.state;
        loop {
            state = match state {
                #[cfg(feature = "parquet_encryption")]
                ParquetOpenState::LoadEncryption(f) => {
                    ParquetOpenState::PruneFile(f.await?)
                }
                ParquetOpenState::LoadMetadata(f) => {
                    ParquetOpenState::PrepareFilters(Box::new(f.await?))
                }
                ParquetOpenState::LoadPageIndex(f) => ParquetOpenState::LoadBloomFilters(
                    f.await?.load_bloom_filters().boxed(),
                ),
                ParquetOpenState::LoadBloomFilters(f) => {
                    ParquetOpenState::PruneWithBloomFilters(Box::new(f.await?))
                }
                ParquetOpenState::BuildStream(state) => {
                    let (prepared, metadata, access, projection, options) =
                        state.prepare_parts()?;
                    let predicate = prepared
                        .pushdown_filters
                        .then_some(prepared.predicate.clone())
                        .flatten();
                    let mut read_mask = projection.projection_mask().clone();
                    // Include every declared predicate input, including dynamic
                    // inputs whose initial predicate is still an always-true literal.
                    if let Some(predicate) = &predicate {
                        let read =
                            crate::projection_read_plan::build_projection_read_plan(
                                std::iter::once(Arc::clone(predicate)),
                                &prepared.physical_file_schema,
                                metadata.parquet_schema(),
                            );
                        read_mask.union(&read.projection_mask);
                    }
                    let mut ordered =
                        access.clone().prepare(metadata.metadata().row_groups())?;
                    if let Some(sort) = &prepared.sort_order_for_reorder {
                        ordered = ordered.reorder_by_statistics(
                            sort,
                            metadata.metadata(),
                            &prepared.physical_file_schema,
                        )?;
                    }
                    if prepared.reverse_row_groups {
                        ordered = ordered.reverse(metadata.metadata())?;
                    }
                    let groups = ordered.row_group_indexes;
                    let dynamic = prepared.predicate.as_ref().is_some_and(|p| {
                        DynamicFilterTracking::classify(p).contains_dynamic_filter()
                    });
                    let file = Arc::new(PreparedParquetFile {
                        metadata,
                        access,
                        groups,
                        projection,
                        predicate: prepared.predicate,
                        dynamic,
                        physical_schema: prepared.physical_file_schema,
                        metrics: prepared.file_metrics,
                        reorder: prepared.reorder_predicates,
                        force_selections: prepared.force_filter_selections,
                        cache_bytes: prepared.max_predicate_cache_size,
                        read_mask,
                        pushdown: prepared.pushdown_filters,
                        factory: prepared.parquet_file_reader_factory,
                        partition: prepared.partition_index,
                        file: prepared.partitioned_file,
                        plan_metrics: prepared.metrics,
                        hint: prepared.metadata_size_hint,
                        options,
                    });
                    // Keep pruning predicates even when row pushdown is disabled;
                    // decoder_builder gates actual row evaluation separately.
                    return Ok(Some(PreparedParquetScan {
                        file,
                        reader: prepared.async_file_reader,
                    }));
                }
                ParquetOpenState::Done => return Ok(None),
                other => other.transition()?,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Decimal128Array, Int32Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion_datasource::{
        file::FileSource, file_scan_config::FileScanConfigBuilder,
    };
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
    use object_store::{ObjectStore, ObjectStoreExt, memory::InMemory, path::Path};
    use parquet::{DecodeResult, arrow::ArrowWriter, file::properties::WriterProperties};

    #[tokio::test]
    async fn pq_prepared_scan_preserves_retyping_predicate_projection_and_selection()
    -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("price", DataType::Decimal128(8, 2), false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5])),
                Arc::new(
                    Decimal128Array::from(vec![10, 20, 30, 40, 50, 60])
                        .with_precision_and_scale(8, 2)?,
                ),
            ],
        )?;
        let mut bytes = Vec::new();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(6))
            .set_data_page_row_count_limit(3)
            .set_write_batch_size(3)
            .build();
        let mut writer =
            ArrowWriter::try_new(&mut bytes, Arc::clone(&schema), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("pq.parquet");
        store.put(&path, bytes.into()).await?;
        let file = PartitionedFile::new_from_meta(store.head(&path).await?);
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("key", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));
        let narrow = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("price", DataType::Decimal32(8, 2), false),
        ]));
        let original = crate::source::ParquetSource::new(Arc::clone(&schema));
        let source = original
            .with_retyped_columns(
                TableSchema::from(narrow),
                original.projection().unwrap().clone(),
                Some(predicate),
            )
            .with_pushdown_filters(true);
        for page_index in [false, true] {
            let source = source.clone().with_enable_page_index(page_index);
            let projection = ProjectionExprs::new(vec![
                datafusion_physical_expr::projection::ProjectionExpr::new(
                    Arc::new(Column::new("price", 1)),
                    "price",
                ),
            ]);
            let source = source
                .try_pushdown_projection(&projection)?
                .unwrap()
                .with_batch_size(2);
            let source = source
                .downcast_ref::<crate::source::ParquetSource>()
                .unwrap();
            let config = FileScanConfigBuilder::new(
                ObjectStoreUrl::local_filesystem(),
                Arc::new(source.clone()),
            )
            .with_file(file.clone())
            .build();
            let scan = source
                .prepare_file(store.clone(), &config, file.clone(), 0)
                .await?
                .unwrap()
                .with_page_index_if_oversized(1)
                .await?;
            assert_eq!(
                scan.file.metadata().schema().field(1).data_type(),
                &DataType::Decimal32(8, 2)
            );
            let mut reader = scan.reader;
            for sliced in [false, true] {
                let mut count = 0;
                let mut values = Vec::new();
                for group in scan.file.access_plan().row_group_indexes() {
                    let ranges = scan.file.byte_ranges(group, 2);
                    let buffers = reader.get_byte_ranges(ranges.clone()).await?;
                    let rows = scan.file.metadata().metadata().row_group(group).num_rows()
                        as usize;
                    let intervals = if sliced {
                        vec![0..1, 1..rows]
                    } else {
                        vec![0..rows]
                    };
                    for interval in intervals {
                        let mut decoder =
                            scan.file.decoder_builder(group, interval, 2)?.build()?;
                        decoder.push_ranges(ranges.clone(), buffers.clone())?;
                        loop {
                            match decoder.try_decode()? {
                                DecodeResult::Data(batch) => {
                                    let output = scan.file.project(&batch)?;
                                    count += output.num_rows();
                                    values.extend_from_slice(
                                        output
                                            .column(0)
                                            .as_any()
                                            .downcast_ref::<arrow::array::Decimal32Array>(
                                            )
                                            .unwrap()
                                            .values(),
                                    );
                                    assert_eq!(output.num_columns(), 1);
                                    assert_eq!(
                                        output.column(0).data_type(),
                                        &DataType::Decimal32(8, 2)
                                    );
                                }
                                DecodeResult::Finished => break,
                                DecodeResult::NeedsData(ranges) => {
                                    panic!("missing prepared ranges {ranges:?}")
                                }
                            }
                        }
                    }
                }
                assert_eq!(count, 4);
                assert_eq!(values, vec![30, 40, 50, 60]);
            }
        }
        Ok(())
    }
}
