// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`ParquetFileReaderFactory`] and [`DefaultParquetFileReaderFactory`] for
//! low level control of parquet file readers

use crate::ParquetFileMetrics;
use crate::metadata::{DFParquetMetadata, PageIndexRequest};
use arrow::datatypes::SchemaRef;
use bytes::Bytes;
use datafusion_common::HashMap;
use datafusion_datasource::PartitionedFile;
use datafusion_execution::cache::cache_manager::FileMetadata;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_physical_plan::metrics::Count;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::FutureExt;
use futures::future::BoxFuture;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::push_decoder::ScanPlan;
use parquet::errors::ParquetError;
use parquet::file::metadata::ParquetMetaData;
use parquet::schema::types::SchemaDescriptor;
use std::any::Any;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

/// Derives a file's Arrow schema from its physical Parquet schema during opening.
///
/// Install with [`ParquetSource::with_schema_provider`](crate::source::ParquetSource::with_schema_provider).
/// This runs after the footer is loaded, before Arrow schema inference or filter
/// preparation, so files may have different nested layouts without loading their
/// footers during planning. The original Parquet metadata is retained.
pub trait ParquetFileSchemaProvider: Debug + Send + Sync + 'static {
    /// Return the complete Arrow schema for the physical file, excluding partition
    /// and virtual columns. It must be compatible with the Parquet schema under
    /// [`ArrowReaderOptions::with_schema`], which also ignores advisory `ARROW:schema`
    /// metadata. Errors fail the file open.
    ///
    /// An explicit [`PartitionedFile::arrow_schema`] takes precedence and skips
    /// this call. Existing table-schema and INT96 coercions are applied afterwards.
    fn schema(
        &self,
        parquet_schema: &SchemaDescriptor,
    ) -> datafusion_common::Result<SchemaRef>;
}

/// Interface for reading Apache Parquet files.
///
/// The combined implementations of [`ParquetFileReaderFactory`] and
/// [`AsyncFileReader`] can be used to provide custom data access operations
/// such as pre-cached metadata, I/O coalescing, etc.
///
/// See [`DefaultParquetFileReaderFactory`] for a simple implementation.
pub trait ParquetFileReaderFactory: Debug + Send + Sync + 'static {
    /// Maximum row groups that this reader can prefetch ahead of demand.
    /// Work sharing keeps this window with the owner and gives idle siblings
    /// only whole row groups beyond it. Factories that add read-ahead must
    /// override this bound; wrappers must forward it.
    fn read_ahead_row_groups(&self, _selected_row_groups: usize) -> usize {
        0
    }

    /// This factory with every metadata cache entry that its readers insert
    /// reserving `copies` derived copies
    /// ([`crate::metadata::CachedParquetMetaData::derived`], Longbow family
    /// OO); `None` for a factory without a shared metadata cache (the
    /// default).
    fn with_derived_copies(
        &self,
        _copies: usize,
    ) -> Option<Arc<dyn ParquetFileReaderFactory>> {
        None
    }

    /// Like [`Self::create_reader`], and also return advice that receives the
    /// byte ranges the native scan stream may still read (Longbow family V).
    ///
    /// The stream calls [`ScanPlanAdvice::advise`] with upstream's
    /// [`ParquetPushDecoder::scan_plan`] after it builds its decoder and after
    /// every rebuild (filter toggle, runtime pruning, row-group donation), so
    /// a reader that prefetches can follow the post-pruning plan. Advice is
    /// never given while the predicate has a dynamic filter that can still
    /// change: such a scan stays demand-driven. The default returns no advice.
    ///
    /// [`ParquetPushDecoder::scan_plan`]: parquet::arrow::push_decoder::ParquetPushDecoder::scan_plan
    fn create_reader_with_scan_plan_advice(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<ReaderWithScanPlanAdvice> {
        let reader = self.create_reader(
            partition_index,
            partitioned_file,
            metadata_size_hint,
            metrics,
        )?;
        Ok((reader, None))
    }

    /// Provides an `AsyncFileReader` for reading data from a parquet file specified
    ///
    /// # Notes
    ///
    /// If the resulting [`AsyncFileReader`]  returns `ParquetMetaData` without
    /// page index information, the reader will load it on demand. Thus it is important
    /// to ensure that the returned `ParquetMetaData` has the necessary information
    /// if you wish to avoid a subsequent I/O
    ///
    /// # Arguments
    /// * partition_index - Index of the partition (for reporting metrics)
    /// * file - The file to be read
    /// * metadata_size_hint - If specified, the first IO reads this many bytes from the footer
    /// * metrics - Execution metrics
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<Box<dyn AsyncFileReader + Send>>;
}

/// A reader and its optional [`ScanPlanAdvice`] (Longbow family V).
pub type ReaderWithScanPlanAdvice = (
    Box<dyn AsyncFileReader + Send>,
    Option<Box<dyn ScanPlanAdvice>>,
);

/// Receives the remaining read plan of one native Parquet scan stream
/// (Longbow family V). See
/// [`ParquetFileReaderFactory::create_reader_with_scan_plan_advice`].
pub trait ScanPlanAdvice: Send {
    /// The byte ranges the stream's decoder may still read, in decode order.
    /// A later call replaces the earlier plan.
    fn advise(&mut self, plan: ScanPlan);
}

/// The [`ParquetFileMetrics`] of one file open, attached to the
/// [`PartitionedFile`] that the opener passes to a reader factory (Longbow
/// family OO).
///
/// The factories of this crate give their readers these metrics instead of
/// registering a second set for the same open; factories that wrap them pass
/// the file through unchanged. A file without the extension (a caller other
/// than the opener) gets new metrics, as before.
#[derive(Debug, Clone)]
pub(crate) struct OpenFileMetrics(pub(crate) ParquetFileMetrics);

/// The metrics a reader of `partitioned_file` records into: the opener's
/// metrics of the open when it attached them, else a new set.
fn reader_file_metrics(
    partition_index: usize,
    partitioned_file: &PartitionedFile,
    metrics: &ExecutionPlanMetricsSet,
) -> ParquetFileMetrics {
    match partitioned_file.extensions.get::<OpenFileMetrics>() {
        Some(open) => open.0.clone(),
        None => ParquetFileMetrics::new(
            partition_index,
            partitioned_file.object_meta.location.as_ref(),
            metrics,
        ),
    }
}

/// Default implementation of [`ParquetFileReaderFactory`]
///
/// This implementation:
/// 1. Reads parquet directly from an underlying [`ObjectStore`] instance.
/// 2. Reads the footer and page metadata on demand.
/// 3. Does not cache metadata or coalesce I/O operations.
#[derive(Debug)]
pub struct DefaultParquetFileReaderFactory {
    store: Arc<dyn ObjectStore>,
}

impl DefaultParquetFileReaderFactory {
    /// Create a new `DefaultParquetFileReaderFactory`.
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self { store }
    }
}

impl ParquetFileReaderFactory for DefaultParquetFileReaderFactory {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<Box<dyn AsyncFileReader + Send>> {
        let file_metrics =
            reader_file_metrics(partition_index, &partitioned_file, metrics);

        let reader = ParquetFileReader::new(
            file_metrics,
            Arc::clone(&self.store),
            partitioned_file,
        )
        .with_metadata_hint(metadata_size_hint);
        Ok(Box::new(reader))
    }
}

/// Implementation of [`ParquetFileReaderFactory`] supporting the caching of footer and page
/// metadata. Reads and updates the [`FileMetadataCache`] with the [`ParquetMetaData`] data.
///
/// [`ParquetFileReader::get_metadata`] forwards the [`parquet::file::metadata::PageIndexPolicy`] from
/// [`ArrowReaderOptions`] to [`DFParquetMetadata::fetch_metadata`], so callers such as the
/// parquet opener can skip page-index I/O during the initial metadata load.
#[derive(Clone)]
pub struct CachedParquetFileReaderFactory {
    store: Arc<dyn ObjectStore>,
    /// The URL that `store` is registered under: it namespaces the cache
    /// (Longbow family OO).
    object_store_url: ObjectStoreUrl,
    metadata_cache: Arc<FileMetadataCache>,
    derived_copies: usize,
}

impl CachedParquetFileReaderFactory {
    /// A factory of readers of `store`, the store registered under
    /// `object_store_url`, that share `metadata_cache`. Entries are keyed by
    /// the URL and the file's location, so stores never share an entry
    /// (Longbow family OO).
    pub fn new(
        store: Arc<dyn ObjectStore>,
        object_store_url: ObjectStoreUrl,
        metadata_cache: Arc<FileMetadataCache>,
    ) -> Self {
        Self {
            store,
            object_store_url,
            metadata_cache,
            derived_copies: 0,
        }
    }

    /// Every entry its readers insert reserves `copies` derived copies
    /// ([`crate::metadata::CachedParquetMetaData::derived`]).
    pub fn with_derived_copies(mut self, copies: usize) -> Self {
        self.derived_copies = copies;
        self
    }
}

impl Debug for CachedParquetFileReaderFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedParquetFileReaderFactory")
            .field("store", &self.store)
            .field("object_store_url", &self.object_store_url)
            .field("derived_copies", &self.derived_copies)
            .finish_non_exhaustive()
    }
}

impl ParquetFileReaderFactory for CachedParquetFileReaderFactory {
    fn with_derived_copies(
        &self,
        copies: usize,
    ) -> Option<Arc<dyn ParquetFileReaderFactory>> {
        Some(Arc::new(
            CachedParquetFileReaderFactory::with_derived_copies(self.clone(), copies),
        ))
    }

    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<Box<dyn AsyncFileReader + Send>> {
        let file_metrics =
            reader_file_metrics(partition_index, &partitioned_file, metrics);

        let reader = ParquetFileReader::new(
            file_metrics,
            Arc::clone(&self.store),
            partitioned_file,
        )
        .with_metadata_hint(metadata_size_hint)
        .with_metadata_cache(Some(Arc::clone(&self.metadata_cache)))
        .with_object_store_url(Some(self.object_store_url.clone()))
        .with_derived_copies(self.derived_copies);

        Ok(Box::new(reader))
    }
}

/// Implements [`AsyncFileReader`] for a parquet file in object storage.
///
/// This implementation reads data directly from the underlying [`ObjectStore`]
/// on demand, as required, tracking the number of bytes read.
///
/// When configured via [`Self::with_metadata_cache`], [`Self::get_metadata`]
/// reads footer and page metadata from the cache when available and populates
/// the cache otherwise. Without a cache, metadata is fetched fresh on every call.
///
/// # Notes
///
/// This implementation does not coalesce I/O operations or cache bytes. Such
/// optimizations can be done either at the object store level or by providing
/// a custom implementation of [`ParquetFileReaderFactory`].
pub struct ParquetFileReader {
    file_metrics: ParquetFileMetrics,
    /// Bytes this reader fetched. The metrics can be shared by the readers
    /// of one open (Longbow family OO), so the scan efficiency ratio takes
    /// each reader's own bytes, not the shared counter.
    scanned: usize,
    store: Arc<dyn ObjectStore>,
    partitioned_file: PartitionedFile,
    metadata_cache: Option<Arc<FileMetadataCache>>,
    /// Namespaces `metadata_cache`; without it the cache is not used
    /// (Longbow family OO).
    object_store_url: Option<ObjectStoreUrl>,
    derived_copies: usize,
    metadata_size_hint: Option<usize>,
    /// The metadata this reader last returned when it does not share it
    /// through a cache, so that a later request for page index entries loads
    /// only those entries instead of the footer again (Longbow family OO).
    footer: Option<Arc<ParquetMetaData>>,
}

impl ParquetFileReader {
    /// Create a new `ParquetFileReader`.
    ///
    /// By default the reader has no [`FileMetadataCache`] and no metadata
    /// size hint, so metadata is fetched fresh on every call (as
    /// [`DefaultParquetFileReaderFactory`] does). Use
    /// [`Self::with_metadata_cache`] to read and populate a cache (as
    /// [`CachedParquetFileReaderFactory`] does), and
    /// [`Self::with_metadata_hint`] to set the size hint.
    pub(crate) fn new(
        file_metrics: ParquetFileMetrics,
        store: Arc<dyn ObjectStore>,
        partitioned_file: PartitionedFile,
    ) -> Self {
        Self {
            file_metrics,
            scanned: 0,
            store,
            partitioned_file,
            metadata_cache: None,
            object_store_url: None,
            derived_copies: 0,
            metadata_size_hint: None,
            footer: None,
        }
    }

    /// Returns the metrics tracked while reading this file.
    pub fn file_metrics(&self) -> &ParquetFileMetrics {
        &self.file_metrics
    }

    /// Returns the file this reader is reading.
    pub fn partitioned_file(&self) -> &PartitionedFile {
        &self.partitioned_file
    }

    /// Set the [`FileMetadataCache`] for this reader
    pub fn with_metadata_cache(
        mut self,
        metadata_cache: Option<Arc<FileMetadataCache>>,
    ) -> Self {
        self.metadata_cache = metadata_cache;
        self
    }

    /// Set the URL of the reader's object store. The metadata cache is used
    /// only with it (Longbow family OO).
    pub fn with_object_store_url(
        mut self,
        object_store_url: Option<ObjectStoreUrl>,
    ) -> Self {
        self.object_store_url = object_store_url;
        self
    }

    /// Entries this reader inserts into the cache reserve `copies` derived
    /// copies.
    pub fn with_derived_copies(mut self, copies: usize) -> Self {
        self.derived_copies = copies;
        self
    }

    /// Set the metadata size hint for this reader.
    ///
    /// See [`DFParquetMetadata::with_metadata_size_hint`] for more details.
    pub fn with_metadata_hint(mut self, metadata_size_hint: Option<usize>) -> Self {
        self.metadata_size_hint = metadata_size_hint;
        self
    }
}

impl AsyncFileReader for ParquetFileReader {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let bytes_scanned = (range.end - range.start) as usize;
        self.file_metrics.bytes_scanned.add(bytes_scanned);
        self.scanned = self.scanned.saturating_add(bytes_scanned);
        // Longbow family D: read the generation the plan names.
        async move {
            crate::identity_io::get_range(
                self.store.as_ref(),
                &self.partitioned_file.object_meta,
                range,
            )
            .await
            .map_err(|e| ParquetError::External(Box::new(e)))
        }
        .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>>
    where
        Self: Send,
    {
        let total = ranges.iter().map(|r| r.end - r.start).sum::<u64>() as usize;
        self.file_metrics.bytes_scanned.add(total);
        self.scanned = self.scanned.saturating_add(total);
        async move {
            crate::identity_io::get_ranges(
                self.store.as_ref(),
                &self.partitioned_file.object_meta,
                &ranges,
            )
            .await
            .map_err(|e| ParquetError::External(Box::new(e)))
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        let object_meta = self.partitioned_file.object_meta.clone();
        let metadata_cache = self.metadata_cache.clone();
        let object_store_url = self.object_store_url.clone();

        async move {
            #[cfg(feature = "parquet_encryption")]
            let file_decryption_properties = options
                .and_then(|o| o.file_decryption_properties())
                .map(Arc::clone);

            #[cfg(not(feature = "parquet_encryption"))]
            let file_decryption_properties = None;

            let request = options.map(PageIndexRequest::from_options);
            let shares_metadata = metadata_cache.is_some()
                && object_store_url.is_some()
                && file_decryption_properties.is_none();
            let error = |e: datafusion_common::DataFusionError| {
                ParquetError::General(format!(
                    "Failed to fetch metadata for file {}: {e}",
                    object_meta.location,
                ))
            };

            // Longbow family OO: without a shared cache, a later request for
            // page index entries adds them to the footer this reader already
            // returned instead of reading the footer again.
            if !shares_metadata
                && let Some(request) = request.as_ref()
                && let Some(footer) = self.footer.clone()
            {
                // Through this reader, so the bytes count as scanned.
                let metadata =
                    request.load_with(footer, &mut *self).await.map_err(error)?;
                request.validate(&metadata).map_err(error)?;
                self.footer = Some(Arc::clone(&metadata));
                return Ok(metadata);
            }

            // Longbow family OO: page index bytes that the metadata layer
            // reads for this open are charged to it.
            let index_bytes = Count::new();
            let fetched = DFParquetMetadata::new(&self.store, &object_meta)
                .with_decryption_properties(file_decryption_properties)
                .with_file_metadata_cache(metadata_cache)
                .with_object_store_url(object_store_url)
                .with_derived_copies(self.derived_copies)
                .with_metadata_size_hint(self.metadata_size_hint)
                .with_page_index_request(request)
                .with_page_index_bytes(Some(index_bytes.clone()))
                .fetch_metadata()
                .await;
            self.file_metrics.bytes_scanned.add(index_bytes.value());
            self.scanned = self.scanned.saturating_add(index_bytes.value());
            let metadata = fetched.map_err(error)?;
            if !shares_metadata {
                self.footer = Some(Arc::clone(&metadata));
            }
            Ok(metadata)
        }
        .boxed()
    }
}

impl Drop for ParquetFileReader {
    fn drop(&mut self) {
        self.file_metrics
            .scan_efficiency_ratio
            .add_part(self.scanned);
        // Multiple ParquetFileReaders may run, so we set_total to avoid adding the total multiple times
        self.file_metrics
            .scan_efficiency_ratio
            .set_total(self.partitioned_file.object_meta.size as usize);
    }
}

/// Wrapper to implement [`FileMetadata`] for [`ParquetMetaData`].
pub struct CachedParquetMetaData(Arc<ParquetMetaData>);

impl CachedParquetMetaData {
    pub fn new(metadata: Arc<ParquetMetaData>) -> Self {
        Self(metadata)
    }

    pub fn parquet_metadata(&self) -> &Arc<ParquetMetaData> {
        &self.0
    }
}

impl FileMetadata for CachedParquetMetaData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn memory_size(&self) -> usize {
        self.0.memory_size()
    }

    fn extra_info(&self) -> HashMap<String, String> {
        let page_index = self
            .0
            .page_index()
            .is_some_and(|page_index| page_index.is_complete());
        HashMap::from([("page_index".to_owned(), page_index.to_string())])
    }
}
