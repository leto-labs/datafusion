// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Prepared file hook (Longbow family PQ): the ordinary opener, stopped after
//! static pruning and before stream construction.
use super::*;
use parquet::arrow::arrow_reader::RowFilter;
use parquet::arrow::{
    ProjectionMask,
    arrow_reader::{RowSelection, RowSelector},
    push_decoder::{ParquetPushDecoderBuilder, RowGroupSelection},
};
use std::ops::Range;

/// Prepared immutable decode state and the original reader used to open a file.
/// The reader remains on the I/O runtime; `file` can be shared by CPU drivers.
pub struct PreparedParquetScan {
    /// Metadata, pruned access plan and decoder template.
    pub file: Arc<PreparedParquetFile>,
    /// Reader supplied by the scan's configured factory. The default factory
    /// reads the generation the file's ETag/version name (family D), so the
    /// prepared footer and later data reads cannot mix object generations; a
    /// custom factory owns its own checks.
    pub reader: Box<dyn AsyncFileReader>,
}

impl PreparedParquetScan {
    /// Load optional page indexes only when a surviving group exceeds the
    /// caller's scheduling threshold. Ordinary row groups add no metadata I/O.
    pub async fn with_page_index_if_oversized(mut self, rows: usize) -> Result<Self> {
        // Longbow family OO: the offset indexes of the columns this scan
        // reads, in its row groups; a shared metadata cache loads the file's
        // complete page index once.
        let request = PageIndexRequest {
            column_policy: PageIndexPolicy::Skip,
            column_mask: ColumnChunkMask::none(),
            offset_policy: PageIndexPolicy::Optional,
            offset_mask: ColumnChunkMask::row_groups_and_columns(
                self.file.groups.iter().copied(),
                (0..self.file.metadata.parquet_schema().num_columns())
                    .filter(|&leaf| self.file.read_mask.leaf_included(leaf)),
            ),
        };
        if !request.is_covered_by(self.file.metadata.metadata())
            && self.file.groups.iter().any(|&group| {
                self.file.metadata.metadata().row_group(group).num_rows() as usize > rows
            })
        {
            let metadata = load_requested_page_index(
                &self.file.metadata,
                &mut self.reader,
                &self.file.options,
                &request,
            )
            .await?;
            let file = Arc::get_mut(&mut self.file).ok_or_else(|| {
                datafusion_common::exec_datafusion_err!(
                    "page index must be loaded before sharing prepared state"
                )
            })?;
            // The pruner must not retain the superseded footer. Its new
            // generations restart, so no earlier verdict may survive.
            file.verdicts.iter().for_each(|verdict| {
                verdict.store(0, std::sync::atomic::Ordering::Relaxed)
            });
            file.pruner = runtime_pruner(
                file.dynamic,
                file.predicate.as_ref(),
                &file.physical_schema,
                &metadata,
                file.max_in_list_size,
            );
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
    /// One runtime row-group pruner for every caller of this shared file.
    /// It rebuilds its pruning predicate only when a watched dynamic filter
    /// changes; `None` unless a dynamic filter can still change.
    pruner: Option<std::sync::Mutex<RowGroupPruner>>,
    /// The last verdict for each physical row group: `generation << 1 | keep`
    /// of the predicate that decided it, or zero.
    verdicts: Vec<std::sync::atomic::AtomicU64>,
    evaluation_errors: Count,
    max_in_list_size: usize,
    physical_schema: SchemaRef,
    /// Upstream's per-file row-filter plan (pushdown only): the candidates are
    /// resolved once and every decoder gets a fresh `RowFilter`.
    row_filter: Option<RowFilterContext>,
    staged_scan: Option<Arc<parquet::arrow::arrow_reader::StagedScan>>,
    force_selections: bool,
    read_mask: ProjectionMask,
    factory: Arc<dyn ParquetFileReaderFactory>,
    partition: usize,
    file: PartitionedFile,
    plan_metrics: ExecutionPlanMetricsSet,
    /// The metrics of the open that prepared this file (Longbow family OO).
    file_metrics: ParquetFileMetrics,
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
        // Longbow family OO: every reader of the file records into the
        // open's metrics.
        self.factory.create_reader(
            self.partition,
            reader_file(&self.file, &self.file_metrics),
            self.hint,
            &self.plan_metrics,
        )
    }
    /// Metadata with the opener's coerced schema, including narrow decimals.
    pub fn metadata(&self) -> &ArrowReaderMetadata {
        &self.metadata
    }
    /// Access after all static pruning. Selections are row-group-local.
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
        let mut filter = additional.unwrap_or_else(|| RowFilter::new(vec![]));
        for predicate in filter.predicates() {
            if (0..self.metadata.parquet_schema().num_columns()).any(|column| {
                predicate.projection().leaf_included(column)
                    && !self.read_mask.leaf_included(column)
            }) {
                return exec_err!(
                    "runtime filter requested a column outside the prepared scan ranges"
                );
            }
        }
        if let Some(context) = &self.row_filter {
            filter.append(context.build_row_filter());
            if let Some(bytes) = context.max_predicate_cache_size {
                builder = builder.with_max_predicate_cache_size(bytes);
            }
        }
        if !filter.predicates().is_empty() {
            builder = builder.with_row_filter(filter);
        } else {
            builder = builder.with_staged_scan(self.staged_scan.clone());
        }
        Ok(builder)
    }
    /// Whether native batches can carry the exact staged selection.
    pub fn has_staged_scan(&self) -> bool {
        self.staged_scan.is_some()
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
        // Upstream's row-group-local selection (#25608): the selection
        // addresses only this row group's rows.
        let selection = match &self.access.inner()[group] {
            RowGroupAccess::Skip => RowSelection::from(vec![RowSelector::skip(total)]),
            RowGroupAccess::Scan => interval,
            RowGroupAccess::Selection(base) => base.intersection(&interval),
        };
        Ok(DecoderBuilderConfig::apply_units(
            builder,
            vec![RowGroupSelection::new(group, Some(selection))],
            batch_size,
            self.force_selections,
        ))
    }
    /// Encoded ranges for a row group. Includes all predicate and output columns;
    /// Static selections include predicate-cache batch boundaries. Adjacent
    /// pages are one buffer, so a full-column decoder request is also covered.
    /// Fetch once and share these immutable buffers between oversized RG slices.
    pub fn byte_ranges(&self, group: usize, batch_size: usize) -> Vec<Range<u64>> {
        self.byte_ranges_with_cached_columns(group, batch_size, |_| false)
    }

    /// Like `byte_ranges`, omitting a compressed, unencrypted column only when
    /// `cached` pins every page of all its required ranges. The pins must stay
    /// visible through this file's page-cache handle until decode finishes.
    /// Partial column hits cannot omit ranges: the push decoder fetches a
    /// column unless all pages that its selection needs are resident.
    pub fn byte_ranges_with_cached_columns(
        &self,
        group: usize,
        batch_size: usize,
        mut cached: impl FnMut(&[Range<u64>]) -> bool,
    ) -> Vec<Range<u64>> {
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
            let mut column_ranges = Vec::new();
            if let Some(selection) = &selection
                && let Some(index) = meta
                    .page_index()
                    .and_then(|page_index| page_index.offset_index(group, column))
                    .filter(|index| !index.page_locations().is_empty())
            {
                if let Some(first) = index.page_locations().first() {
                    let page = first.offset as u64;
                    if start < page {
                        column_ranges.push(start..page);
                    }
                }
                column_ranges.extend(selection.scan_ranges(index.page_locations()));
            } else {
                column_ranges.push(start..start + len);
            }
            let cacheable =
                chunk.compression() != parquet::basic::Compression::UNCOMPRESSED;
            #[cfg(feature = "parquet_encryption")]
            let cacheable = cacheable && chunk.crypto_metadata().is_none();
            if !cacheable || !cached(&column_ranges) {
                ranges.extend(column_ranges);
            }
        }
        merge_ranges(ranges)
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
        use std::sync::atomic::Ordering;
        let Some(pruner) = &self.pruner else {
            return true;
        };
        // Hold the lock only to refresh the predicate for the current filter
        // generation. Drivers of the same file evaluate it concurrently.
        let (generation, current) = pruner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .current();
        let Some(pruning) = current else {
            return true;
        };
        // A row group is checked before its read and again by each of its
        // units. One verdict serves every check under the same predicate.
        let verdict = self.verdicts.get(group);
        if let Some(verdict) = verdict {
            let known = verdict.load(Ordering::Acquire);
            if known >> 1 == generation {
                return known & 1 == 1;
            }
        }
        let keep = !crate::push_decoder::prune_row_groups(
            &pruning,
            self.metadata.metadata(),
            &self.physical_schema,
            &[group],
            &self.evaluation_errors,
        );
        if let Some(verdict) = verdict {
            verdict.fetch_max((generation << 1) | u64::from(keep), Ordering::AcqRel);
        }
        keep
    }

    /// Record one row group skipped by the native scan after file open.
    pub fn record_runtime_pruned_row_group(&self) {
        self.file_metrics.row_groups_pruned_dynamic_filter.add(1);
    }

    /// Record one runtime-pruned key-range visit, which can share a row group
    /// with visits from other ranges.
    pub fn record_runtime_pruned_range_visit(&self) {
        self.file_metrics
            .row_group_range_visits_pruned_dynamic_filter
            .add(1);
    }
}
fn runtime_pruner(
    dynamic: bool,
    predicate: Option<&Arc<dyn PhysicalExpr>>,
    schema: &SchemaRef,
    metadata: &ArrowReaderMetadata,
    max_in_list_size: usize,
) -> Option<std::sync::Mutex<RowGroupPruner>> {
    let predicate = predicate.filter(|_| dynamic)?;
    Some(std::sync::Mutex::new(RowGroupPruner::new(
        Arc::clone(predicate),
        Arc::clone(schema),
        Arc::clone(metadata.metadata()),
        Count::new(),
        Count::new(),
        max_in_list_size,
    )))
}
/// Sort and merge overlapping or adjacent ranges into one buffer each.
fn merge_ranges(mut ranges: Vec<Range<u64>>) -> Vec<Range<u64>> {
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
    /// `None` means file-level or row-group pruning proved that no row can match.
    pub async fn prepare_file(
        &self,
        store: Arc<dyn object_store::ObjectStore>,
        config: &datafusion_datasource::file_scan_config::FileScanConfig,
        file: PartitionedFile,
        partition: usize,
    ) -> Result<Option<PreparedParquetScan>> {
        // A prepared scan is opened without `FileSource::with_batch_size`: use
        // the scan config's batch size, else DataFusion's default.
        let mut source = self.clone();
        source.batch_size = source
            .batch_size
            .or(config.batch_size)
            .or(Some(PREPARED_DEFAULT_BATCH_SIZE));
        let morselizer = source.prepared_morselizer(store, config, partition)?;
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
                    return prepared_scan(state.finish_pruning()?).map(Some);
                }
                ParquetOpenState::Done => return Ok(None),
                other => other.transition()?,
            };
        }
    }
}
/// Batch size of a prepared scan whose source and config set none
/// (`datafusion.execution.batch_size`'s default).
const PREPARED_DEFAULT_BATCH_SIZE: usize = 8192;

/// Package the opener's post-pruning state as a shared prepared file.
fn prepared_scan(pruned: PrunedParquetOpen) -> Result<PreparedParquetScan> {
    let PrunedParquetOpen {
        prepared,
        reader_metadata: metadata,
        options,
        access_plan: access,
        decoder_read_plans,
    } = pruned;
    let DecoderReadPlans {
        projection,
        row_filter_context,
    } = decoder_read_plans;
    let staged_scan = prepared.staged_scan(&metadata, &access);
    let mut read_mask = projection.projection_mask().clone();
    // Include every declared predicate input, including dynamic inputs whose
    // initial predicate is still an always-true literal.
    if let Some(predicate) = prepared
        .pushdown_filters
        .then_some(prepared.predicate.as_ref())
        .flatten()
    {
        let read = crate::projection_read_plan::build_projection_read_plan(
            std::iter::once(Arc::clone(predicate)),
            &prepared.physical_file_schema,
            metadata.parquet_schema(),
        );
        read_mask.union(&read.projection_mask);
    }
    let mut ordered = access.clone().prepare(metadata.metadata().row_groups())?;
    if let Some(sort) = &prepared.sort_order_for_reorder {
        ordered = ordered.reorder_by_statistics(
            sort,
            metadata.metadata(),
            &prepared.physical_file_schema,
        )?;
    }
    if prepared.reverse_row_groups {
        ordered = ordered.reverse();
    }
    let groups = ordered
        .row_groups
        .iter()
        .map(|group| group.selection.row_group_index())
        .collect();
    // Only a dynamic filter that can still change needs runtime pruning and
    // decoder invalidation: complete filters were consumed by static pruning.
    let dynamic = prepared.predicate.as_ref().is_some_and(|p| {
        matches!(
            DynamicFilterTracking::classify(p),
            DynamicFilterTracking::Watching(_)
        )
    });
    let verdicts = std::iter::repeat_with(Default::default)
        .take(metadata.metadata().num_row_groups())
        .collect();
    let pruner = runtime_pruner(
        prepared.pruning_expr.as_ref().is_some_and(|predicate| {
            matches!(
                DynamicFilterTracking::classify(predicate),
                DynamicFilterTracking::Watching(_)
            )
        }),
        prepared.pruning_expr.as_ref(),
        &prepared.physical_file_schema,
        &metadata,
        prepared.max_in_list_size,
    );
    let file = Arc::new(PreparedParquetFile {
        metadata,
        access,
        groups,
        projection,
        predicate: prepared.predicate,
        dynamic,
        verdicts,
        pruner,
        evaluation_errors: prepared.file_metrics.predicate_evaluation_errors.clone(),
        max_in_list_size: prepared.max_in_list_size,
        physical_schema: prepared.physical_file_schema,
        row_filter: row_filter_context,
        staged_scan,
        force_selections: prepared.force_filter_selections,
        read_mask,
        factory: prepared.parquet_file_reader_factory,
        partition: prepared.partition_index,
        file: prepared.partitioned_file,
        plan_metrics: prepared.metrics,
        file_metrics: prepared.file_metrics,
        hint: prepared.metadata_size_hint,
        options,
    });
    // Keep pruning predicates even when row pushdown is disabled; the decoder
    // builder gates row evaluation through `row_filter` separately.
    Ok(PreparedParquetScan {
        file,
        reader: prepared.async_file_reader,
    })
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
                        std::iter::once(0..rows).collect()
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

    /// Families PQ and NFC with a page-index `Selection` access (`key > 2`
    /// prunes the first page of 3 rows): interval decoders intersect the
    /// prepared selection, `byte_ranges` widens to predicate-cache batch
    /// boundaries (`batch_selection`), a slice starting mid-batch decodes from
    /// the planned ranges alone, and `cached` is asked exactly about each
    /// column's dictionary prefix and selected pages.
    #[tokio::test]
    async fn pq_prepared_selection_intervals_batch_ranges_and_cache_queries() -> Result<()>
    {
        use parquet::basic::Compression;
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values(0..12)),
                Arc::new(Int32Array::from_iter_values(100..112)),
            ],
        )?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_data_page_row_count_limit(3)
            .set_write_batch_size(3)
            .build();
        let mut bytes = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut bytes, Arc::clone(&schema), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("selection.parquet");
        store.put(&path, bytes.into()).await?;
        let file = PartitionedFile::new_from_meta(store.head(&path).await?);
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("key", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))),
        ));
        let source = crate::source::ParquetSource::new(Arc::clone(&schema))
            .with_predicate(predicate)
            .with_enable_page_index(true);
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source.clone()),
        )
        .with_file(file.clone())
        .build();
        let scan = source
            .prepare_file(Arc::clone(&store), &config, file, 0)
            .await?
            .unwrap();
        assert_eq!(
            scan.file.access_plan().inner()[0],
            RowGroupAccess::Selection(
                vec![RowSelector::skip(3), RowSelector::select(9)].into()
            )
        );
        let meta = scan.file.metadata().metadata();
        let pages = |column: usize| {
            meta.page_index()
                .unwrap()
                .offset_index(0, column)
                .unwrap()
                .page_locations()
                .clone()
        };
        let page_range =
            |location: &parquet::file::page_index::offset_index::PageLocation| {
                location.offset as u64
                    ..location.offset as u64 + location.compressed_page_size as u64
            };
        let covers = |ranges: &[Range<u64>], wanted: Range<u64>| {
            ranges
                .iter()
                .any(|range| range.start <= wanted.start && wanted.end <= range.end)
        };
        // Batch size 2 quantizes the selected rows 3..12 to 2..12, which needs
        // the first page (rows 0..3); batch size 3 does not.
        let first_page = page_range(&pages(0)[0]);
        assert!(covers(&scan.file.byte_ranges(0, 2), first_page.clone()));
        assert!(!covers(&scan.file.byte_ranges(0, 3), first_page));

        // A slice that starts mid-batch decodes from the planned ranges only.
        let ranges = scan.file.byte_ranges(0, 3);
        let mut reader = scan.file.reader()?;
        let buffers = reader.get_byte_ranges(ranges.clone()).await?;
        let mut decoder = scan.file.decoder_builder(0, 4..10, 3)?.build()?;
        decoder.push_ranges(ranges.clone(), buffers)?;
        let mut keys = vec![];
        loop {
            match decoder.try_decode()? {
                DecodeResult::Data(batch) => {
                    let batch = scan.file.project(&batch)?;
                    keys.extend_from_slice(
                        batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap()
                            .values(),
                    );
                }
                DecodeResult::Finished => break,
                DecodeResult::NeedsData(ranges) => {
                    panic!("missing planned ranges {ranges:?}")
                }
            }
        }
        assert_eq!(keys, (4..10).collect::<Vec<_>>());
        // A slice outside the prepared selection is empty.
        let mut empty = scan.file.decoder_builder(0, 0..3, 3)?.build()?;
        assert!(matches!(
            empty.try_decode()?,
            DecodeResult::Finished | DecodeResult::NeedsData(_)
        ));

        // NFC: `cached` is asked about each compressed column's dictionary
        // prefix and selected pages (rows 3..12: pages 1..4), nothing else.
        let mut asked = vec![];
        let fetched = scan.file.byte_ranges_with_cached_columns(0, 3, |ranges| {
            asked.push(ranges.to_vec());
            true
        });
        assert!(fetched.is_empty(), "every column was pinned: {fetched:?}");
        let expected: Vec<Vec<Range<u64>>> = (0..2)
            .map(|column| {
                let chunk = meta.row_group(0).column(column);
                let (start, _) = chunk.byte_range();
                let locations = pages(column);
                let mut ranges = vec![];
                if start < locations[0].offset as u64 {
                    ranges.push(start..locations[0].offset as u64);
                }
                ranges.extend(locations[1..].iter().map(page_range));
                ranges
            })
            .collect();
        let merged = |ranges: &[Range<u64>]| merge_ranges(ranges.to_vec());
        assert_eq!(
            asked
                .iter()
                .map(|ranges| merged(ranges))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|ranges| merged(ranges))
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    /// Family D: a prepared footer and its later data reads belong to one
    /// object generation. After a same-size overwrite, a reader of the planned
    /// ETag fails instead of decoding new bytes with old offsets.
    #[tokio::test]
    async fn pq_prepared_scan_reads_fail_after_object_replacement() -> Result<()> {
        let schema =
            Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter_values(0..100))],
        )?;
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, Arc::clone(&schema), None)?;
        writer.write(&batch)?;
        writer.close()?;
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("replaced.parquet");
        store.put(&path, bytes.clone().into()).await?;
        let file = PartitionedFile::new_from_meta(store.head(&path).await?);
        let source = crate::source::ParquetSource::new(Arc::clone(&schema));
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source.clone()),
        )
        .with_file(file.clone())
        .build();
        let scan = source
            .prepare_file(Arc::clone(&store), &config, file, 0)
            .await?
            .unwrap();
        let ranges = scan.file.byte_ranges(0, 8);
        // The same generation reads.
        let mut reader = scan.file.reader()?;
        reader.get_byte_ranges(ranges.clone()).await?;
        // Same path and size, new generation.
        store.put(&path, bytes.into()).await?;
        let mut stale = scan.reader;
        let error = stale.get_byte_ranges(ranges.clone()).await.unwrap_err();
        assert!(error.to_string().contains("precondition"), "{error}");
        let error = scan
            .file
            .reader()?
            .get_byte_ranges(ranges)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("precondition"), "{error}");
        Ok(())
    }

    /// Family NFC: a compressed, unencrypted column is omitted only when the
    /// caller pins all of its required ranges; an uncompressed column is
    /// always fetched, and nothing is omitted without pins.
    #[tokio::test]
    async fn byte_ranges_with_cached_columns_omits_only_pinned_compressed_columns()
    -> Result<()> {
        use parquet::basic::Compression;
        use parquet::schema::types::ColumnPath;
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(Int32Array::from_iter_values(100..200)),
            ],
        )?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_column_compression(ColumnPath::from("key"), Compression::UNCOMPRESSED)
            .build();
        let mut bytes = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut bytes, Arc::clone(&schema), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("nfc.parquet");
        store.put(&path, bytes.into()).await?;
        let file = PartitionedFile::new_from_meta(store.head(&path).await?);
        let source = crate::source::ParquetSource::new(Arc::clone(&schema));
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source.clone()),
        )
        .with_file(file.clone())
        .build();
        let scan = source.prepare_file(store, &config, file, 0).await?.unwrap();
        let rg = scan.file.metadata().metadata().row_group(0);
        let column = |index: usize| {
            let (start, len) = rg.column(index).byte_range();
            start..start + len
        };
        let all = scan.file.byte_ranges(0, 8);
        assert_eq!(
            scan.file.byte_ranges_with_cached_columns(0, 8, |_| false),
            all
        );
        let mut asked = Vec::new();
        let pinned = scan.file.byte_ranges_with_cached_columns(0, 8, |ranges| {
            asked.extend(ranges.iter().cloned());
            true
        });
        // Only the compressed column is offered to the cache, and it is omitted.
        assert_eq!(asked, vec![column(1)]);
        assert_eq!(pinned, vec![column(0)]);
        Ok(())
    }
}

#[cfg(test)]
#[path = "prepared_tests.rs"]
mod runtime_pruning_tests;
