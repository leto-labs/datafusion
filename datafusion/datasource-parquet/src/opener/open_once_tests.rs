// Longbow notice: this file was added by the Longbow authors (family OO); see
// third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

// Family OO: the page index is parsed once per file generation, filters are
// prepared once per scan and physical schema, and a file open registers one
// set of file metrics.

use super::*;
use datafusion_physical_expr::expressions::col as physical_col;

/// `a` = `values`, ten rows per page, one row group.
async fn write_pages(store: &Arc<dyn ObjectStore>, path: &str, values: Vec<i32>) -> u64 {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)])),
        vec![Arc::new(arrow::array::Int32Array::from(values))],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_data_page_row_count_limit(10)
        .set_write_batch_size(10)
        .build();
    write_parquet_batches(Arc::clone(store), path, vec![batch], Some(props)).await as u64
}

fn int_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]))
}

fn cached_metadata(
    store: &Arc<dyn ObjectStore>,
    cache: &Arc<FileMetadataCache>,
    path: &str,
) -> Arc<ParquetMetaData> {
    let entry = cache
        .get(&test_cache_key(store, &Path::from(path)))
        .expect("cached footer");
    Arc::clone(
        entry
            .file_metadata
            .as_any()
            .downcast_ref::<CachedParquetMetaData>()
            .expect("parquet metadata")
            .parquet_metadata(),
    )
}

#[tokio::test]
async fn oo_page_index_is_parsed_once_per_file_generation() {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let cache: Arc<FileMetadataCache> = Arc::new(DefaultCache::<
        Path,
        CachedFileMetadataEntry,
    >::new(64 * 1024 * 1024));
    let schema = int_schema();
    let predicate = logical2physical(&col("a").gt(lit(90i32)), &schema);
    let morselizer = ParquetMorselizerBuilder::new()
        .with_store(Arc::clone(&store))
        .with_schema(Arc::clone(&schema))
        .with_predicate(predicate)
        .with_enable_page_index(true)
        .with_row_group_stats_pruning(false)
        .with_parquet_file_reader_factory(Arc::new(CachedParquetFileReaderFactory::new(
            Arc::clone(&store),
            Arc::clone(&cache),
        )))
        .build();

    let len = write_pages(&store, "t.parquet", (1..=100).collect()).await;
    let file = PartitionedFile::new("t.parquet".to_string(), len);
    let (_, rows) =
        count_batches_and_rows(open_file(&morselizer, file.clone()).await.unwrap()).await;
    assert_eq!(rows, 10, "page pruning keeps the last page");
    // The open wrote the complete page index back to the footer's entry.
    let first = cached_metadata(&store, &cache, "t.parquet");
    assert!(PageIndexRequest::all(PageIndexPolicy::Required).is_covered_by(&first));

    // A later open of the same generation reuses it: nothing is written back.
    let (_, rows) =
        count_batches_and_rows(open_file(&morselizer, file).await.unwrap()).await;
    assert_eq!(rows, 10);
    assert!(Arc::ptr_eq(
        &first,
        &cached_metadata(&store, &cache, "t.parquet")
    ));

    // A new generation of the file (other contents and size) parses again.
    let len = write_pages(&store, "t.parquet", (1..=200).collect()).await;
    let file = PartitionedFile::new("t.parquet".to_string(), len);
    let (_, rows) =
        count_batches_and_rows(open_file(&morselizer, file).await.unwrap()).await;
    assert_eq!(rows, 110, "the new generation's rows");
    let second = cached_metadata(&store, &cache, "t.parquet");
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(second.file_metadata().num_rows(), 200);
    assert!(PageIndexRequest::all(PageIndexPolicy::Required).is_covered_by(&second));
}

#[tokio::test]
async fn oo_uncached_reader_loads_only_the_requested_entries() {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let batches = (0..2)
        .map(|group| {
            let values: Vec<i32> = (0..100).map(|v| v + 100 * group).collect();
            record_batch!(("a", Int32, values.clone()), ("b", Int32, values)).unwrap()
        })
        .collect::<Vec<_>>();
    let schema = batches[0].schema();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(100))
        .set_data_page_row_count_limit(10)
        .set_write_batch_size(10)
        .build();
    let len =
        write_parquet_batches(Arc::clone(&store), "m.parquet", batches, Some(props))
            .await;
    let metrics = ExecutionPlanMetricsSet::new();
    let morselizer = ParquetMorselizerBuilder::new()
        .with_store(store)
        .with_schema(schema)
        .with_metrics(metrics.clone())
        .build();
    let file = PartitionedFile::new("m.parquet".to_string(), len as u64);
    let mut loaded = morselizer
        .prepare_open_file(file)
        .unwrap()
        .load()
        .await
        .unwrap();
    assert_eq!(loaded.reader_metadata.metadata().num_row_groups(), 2);
    let request = PageIndexRequest {
        column_policy: PageIndexPolicy::Optional,
        column_mask: ColumnChunkMask::row_groups_and_columns([1], [0]),
        offset_policy: PageIndexPolicy::Optional,
        offset_mask: ColumnChunkMask::row_groups_and_columns([0], [1]),
    };
    let options = loaded.options.clone();
    let metadata = load_requested_page_index(
        &loaded.reader_metadata,
        &mut loaded.prepared.async_file_reader,
        &options,
        &request,
    )
    .await
    .unwrap();
    let index = metadata.metadata().page_index().expect("page index");
    assert!(index.column_index(1, 0).is_some());
    assert!(index.column_index(0, 0).is_none());
    assert!(index.column_index(1, 1).is_none());
    assert!(index.offset_index(0, 1).is_some());
    assert!(index.offset_index(0, 0).is_none());
    assert!(index.offset_index(1, 1).is_none());
    assert!(request.is_covered_by(metadata.metadata()));
    assert!(
        !PageIndexRequest::all(PageIndexPolicy::Optional)
            .is_covered_by(metadata.metadata())
    );

    // Covered entries need no I/O.
    let scanned = counter_metric_value(&metrics, "bytes_scanned");
    let again = load_requested_page_index(
        &metadata,
        &mut loaded.prepared.async_file_reader,
        &options,
        &request,
    )
    .await
    .unwrap();
    assert!(Arc::ptr_eq(metadata.metadata(), again.metadata()));
    assert_eq!(counter_metric_value(&metrics, "bytes_scanned"), scanned);
}

/// The pruning predicate of `file` after filter preparation.
async fn prepared_pruning(
    morselizer: &ParquetMorselizer,
    file: PartitionedFile,
) -> Option<Arc<PruningPredicate>> {
    morselizer
        .prepare_open_file(file)
        .unwrap()
        .load()
        .await
        .unwrap()
        .prepare_filters()
        .unwrap()
        .pruning_predicate
}

#[tokio::test]
async fn oo_filters_are_prepared_once_per_scan_and_physical_schema() {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let schema = int_schema();
    let one = write_pages(&store, "one.parquet", (1..=100).collect()).await;
    let two = write_pages(&store, "two.parquet", (101..=200).collect()).await;
    // Same column stored as Int64: another physical schema (a cast).
    let wide = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)])),
        vec![Arc::new(arrow::array::Int64Array::from_iter_values(0..100))],
    )
    .unwrap();
    let wide = write_parquet_batches(Arc::clone(&store), "wide.parquet", vec![wide], None)
        .await as u64;
    let predicate = logical2physical(&col("a").gt(lit(5i32)), &schema);
    let morselizer = ParquetMorselizerBuilder::new()
        .with_store(store)
        .with_schema(schema)
        .with_predicate(predicate)
        .with_enable_page_index(true)
        .build();

    let first =
        prepared_pruning(&morselizer, PartitionedFile::new("one.parquet", one)).await;
    let second =
        prepared_pruning(&morselizer, PartitionedFile::new("two.parquet", two)).await;
    let (first, second) = (first.unwrap(), second.unwrap());
    assert!(
        Arc::ptr_eq(&first, &second),
        "one preparation for both files"
    );
    assert_eq!(morselizer.filter_cache.len(), 1);

    let cast =
        prepared_pruning(&morselizer, PartitionedFile::new("wide.parquet", wide)).await;
    assert!(!Arc::ptr_eq(&first, &cast.unwrap()));
    assert_eq!(morselizer.filter_cache.len(), 2);

    // Clones of the morselizer (other partitions) share the entries.
    let partition = morselizer.clone();
    let third =
        prepared_pruning(&partition, PartitionedFile::new("one.parquet", one)).await;
    assert!(Arc::ptr_eq(&first, &third.unwrap()));
    assert_eq!(partition.filter_cache.len(), 2);
}

#[tokio::test]
async fn oo_a_dynamic_filter_update_prepares_filters_again() {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let schema = int_schema();
    let len = write_pages(&store, "d.parquet", (1..=100).collect()).await;
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![physical_col("a", &schema).unwrap()],
        lit_true(),
    ));
    let morselizer = ParquetMorselizerBuilder::new()
        .with_store(store)
        .with_schema(Arc::clone(&schema))
        .with_predicate(Arc::clone(&dynamic) as Arc<dyn PhysicalExpr>)
        .build();
    let file = PartitionedFile::new("d.parquet", len);

    let first = prepared_pruning(&morselizer, file.clone()).await;
    let again = prepared_pruning(&morselizer, file.clone()).await;
    match (&first, &again) {
        (Some(a), Some(b)) => assert!(Arc::ptr_eq(a, b)),
        (None, None) => {}
        _ => panic!("same generation, same preparation"),
    }
    assert_eq!(morselizer.filter_cache.len(), 1);

    dynamic
        .update(logical2physical(&col("a").gt(lit(50i32)), &schema))
        .unwrap();
    let updated = prepared_pruning(&morselizer, file).await.expect("pruning");
    assert!(
        updated.predicate_expr().to_string().contains("50"),
        "the new generation's value: {}",
        updated.predicate_expr()
    );
    assert_eq!(
        morselizer.filter_cache.len(),
        1,
        "the new generation replaces"
    );
}

fn lit_true() -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Boolean(Some(true))))
}

#[tokio::test]
async fn oo_one_set_of_file_metrics_per_open() {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let schema = int_schema();
    let len = write_pages(&store, "f.parquet", (1..=100).collect()).await;
    let metrics = ExecutionPlanMetricsSet::new();
    // Bloom filters on: the open replaces its reader after the filter I/O.
    let morselizer = ParquetMorselizerBuilder::new()
        .with_store(store)
        .with_schema(Arc::clone(&schema))
        .with_predicate(logical2physical(&col("a").eq(lit(95i32)), &schema))
        .with_enable_bloom_filter(true)
        .with_enable_page_index(true)
        .with_row_group_stats_pruning(false)
        .with_metrics(metrics.clone())
        .build();
    let (_, rows) = count_batches_and_rows(
        open_file(&morselizer, PartitionedFile::new("f.parquet", len))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(rows, 10);
    let registered = |name: &str| {
        metrics
            .clone_inner()
            .iter()
            .filter(|metric| metric.value().name() == name)
            .count()
    };
    assert_eq!(registered("bytes_scanned"), 1);
    assert_eq!(registered("metadata_load_time"), 1);
    let scanned = counter_metric_value(&metrics, "bytes_scanned");
    assert!(scanned > 0);
    let ratio = metrics
        .clone_inner()
        .sum_by_name("scan_efficiency_ratio")
        .expect("ratio");
    let MetricValue::Ratio { ratio_metrics, .. } = ratio else {
        panic!("ratio metric");
    };
    assert_eq!(
        ratio_metrics.part(),
        scanned,
        "each reader adds its own bytes"
    );
}

#[tokio::test]
async fn oo_a_settled_predicate_shares_one_file_level_pruning_predicate() {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let schema = int_schema();
    let with_range = |path: &str, len: u64, min: i32, max: i32| {
        PartitionedFile::new(path, len).with_statistics(Arc::new(Statistics {
            num_rows: Precision::Exact(100),
            total_byte_size: Precision::Absent,
            column_statistics: vec![
                ColumnStatistics::new_unknown()
                    .with_min_value(Precision::Exact(ScalarValue::Int32(Some(min))))
                    .with_max_value(Precision::Exact(ScalarValue::Int32(Some(max))))
                    .with_null_count(Precision::Exact(0)),
            ],
        }))
    };
    let low = write_pages(&store, "low.parquet", (1..=100).collect()).await;
    let high = write_pages(&store, "high.parquet", (101..=200).collect()).await;
    let metrics = ExecutionPlanMetricsSet::new();
    let morselizer = ParquetMorselizerBuilder::new()
        .with_store(store)
        .with_schema(Arc::clone(&schema))
        .with_predicate(logical2physical(&col("a").gt(lit(150i32)), &schema))
        .with_metrics(metrics.clone())
        .build();
    let low = with_range("low.parquet", low, 1, 100);
    let high = with_range("high.parquet", high, 101, 200);

    let first = morselizer.prepare_open_file(low.clone()).unwrap();
    let second = morselizer.prepare_open_file(high.clone()).unwrap();
    assert!(
        first.file_pruner.is_none(),
        "a static predicate needs no FilePruner"
    );
    let (Some(a), Some(b)) = (&first.settled_file_pruning, &second.settled_file_pruning)
    else {
        panic!("file-level pruning predicates");
    };
    assert!(
        Arc::ptr_eq(a, b),
        "one file-level pruning predicate per scan"
    );

    // The shared predicate prunes the file whose statistics rule it out.
    let (_, rows) =
        count_batches_and_rows(open_file(&morselizer, low).await.unwrap()).await;
    assert_eq!(rows, 0);
    let (_, rows) =
        count_batches_and_rows(open_file(&morselizer, high).await.unwrap()).await;
    assert!(rows > 0);
    let pruned = metrics
        .clone_inner()
        .sum_by_name("files_ranges_pruned_statistics")
        .expect("file pruning metric");
    let MetricValue::PruningMetrics {
        pruning_metrics, ..
    } = pruned
    else {
        panic!("pruning metric");
    };
    assert_eq!(
        (pruning_metrics.pruned(), pruning_metrics.matched()),
        (1, 1)
    );
}

#[tokio::test]
async fn metadata_cache_isolates_equal_listings_in_distinct_stores() {
    let a: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let b: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "same.parquet";
    let mut sizes = Vec::new();
    for (store, column) in [(&a, "a"), (&b, "b")] {
        let schema =
            Arc::new(Schema::new(vec![Field::new(column, DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::Int32Array::from_iter_values(0..100))],
        )
        .unwrap();
        sizes.push(
            write_parquet_batches(Arc::clone(store), path, vec![batch], None).await,
        );
    }
    assert_eq!(sizes[0], sizes[1]);
    // Equal weak listing metadata is valid for independent immutable stores.
    let file = PartitionedFile::new(path.to_string(), sizes[0] as u64);
    let cache: Arc<FileMetadataCache> =
        Arc::new(DefaultCache::<Path, CachedFileMetadataEntry>::new(1 << 20));
    let metadata_a = DFParquetMetadata::new(a.as_ref(), &file.object_meta)
        .with_file_metadata_cache(Some(Arc::clone(&cache)))
        .with_page_index_policy(Some(PageIndexPolicy::Skip))
        .fetch_metadata()
        .await
        .unwrap();
    assert_eq!(
        metadata_a.file_metadata().schema_descr().column(0).name(),
        "a"
    );
    let metadata_b = DFParquetMetadata::new(b.as_ref(), &file.object_meta)
        .with_file_metadata_cache(Some(cache))
        .with_page_index_policy(Some(PageIndexPolicy::Optional))
        .fetch_metadata()
        .await
        .unwrap();
    assert_eq!(
        metadata_b.file_metadata().schema_descr().column(0).name(),
        "b"
    );
}

#[tokio::test]
async fn a_warm_optional_index_does_not_satisfy_a_required_offset_index() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "one-missing-offset-index.parquet";
    let batch =
        record_batch!(("a", Int32, vec![1, 2]), ("b", Int32, vec![3, 4])).unwrap();
    let size = write_parquet_batches(Arc::clone(&store), path, vec![batch], None).await;
    let file = PartitionedFile::new(path.to_string(), size as u64);
    let footer = DFParquetMetadata::new(store.as_ref(), &file.object_meta)
        .with_page_index_policy(Some(PageIndexPolicy::Optional))
        .fetch_metadata()
        .await
        .unwrap();
    assert!(footer.row_group(0).column(0).offset_index_range().is_some());
    let mut columns = footer.row_group(0).columns().to_vec();
    columns[1] = columns[1]
        .clone()
        .into_builder()
        .set_offset_index_offset(None)
        .build()
        .unwrap();
    let group = footer
        .row_group(0)
        .clone()
        .into_builder()
        .set_column_metadata(columns)
        .build()
        .unwrap();
    let partial = footer
        .as_ref()
        .clone()
        .into_builder()
        .set_row_groups(vec![group])
        .build();

    // Keep optional-load-equivalent index contents for the synthetic footer.
    let mut reader: Box<dyn AsyncFileReader> =
        DefaultParquetFileReaderFactory::new(store)
            .create_reader(0, file, None, &ExecutionPlanMetricsSet::new())
            .unwrap();
    let mut optional = ParquetMetaDataReader::new_with_metadata(partial)
        .with_page_index_policy(PageIndexPolicy::Optional);
    optional.load_page_index(&mut reader).await.unwrap();
    let partial = optional.finish().unwrap();
    assert!(partial.page_index().unwrap().offset_index(0, 1).is_none());

    let mut reference = ParquetMetaDataReader::new_with_metadata(partial.clone())
        .with_page_index_policy(PageIndexPolicy::Required);
    assert!(reference.load_page_index(&mut reader).await.is_err());
    assert!(!PageIndexRequest::all(PageIndexPolicy::Required).is_covered_by(&partial));
}

#[tokio::test]
async fn a_cached_reader_charges_cold_page_index_bytes_to_the_open() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "cold-index.parquet";
    let size = write_pages(&store, path, (1..=100).collect()).await;
    let cache: Arc<FileMetadataCache> =
        Arc::new(DefaultCache::<Path, CachedFileMetadataEntry>::new(1 << 20));
    let metrics = ExecutionPlanMetricsSet::new();
    let mut reader = CachedParquetFileReaderFactory::new(store, cache)
        .create_reader(
            0,
            PartitionedFile::new(path.to_string(), size),
            None,
            &metrics,
        )
        .unwrap();
    let skip = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Skip);
    let footer = reader.get_metadata(Some(&skip)).await.unwrap();
    assert!(footer.page_index().is_none());
    let optional =
        ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional);
    let before = counter_metric_value(&metrics, "bytes_scanned");
    let indexed = reader.get_metadata(Some(&optional)).await.unwrap();
    assert!(indexed.page_index().unwrap().has_offset_indexes());
    let after = counter_metric_value(&metrics, "bytes_scanned");
    assert!(
        after > before,
        "cold index fetch must contribute to bytes_scanned"
    );
    reader.get_metadata(Some(&optional)).await.unwrap();
    assert_eq!(counter_metric_value(&metrics, "bytes_scanned"), after);
}

#[tokio::test]
async fn oo_metadata_cache_keys_entries_by_object_store() {
    let a: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let b: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "same.parquet";
    let mut sizes = Vec::new();
    for (store, column) in [(&a, "a"), (&b, "b")] {
        let schema =
            Arc::new(Schema::new(vec![Field::new(column, DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::Int32Array::from_iter_values(0..100))],
        )
        .unwrap();
        sizes.push(
            write_parquet_batches(Arc::clone(store), path, vec![batch], None).await,
        );
    }
    assert_eq!(sizes[0], sizes[1]);
    let file = PartitionedFile::new(path.to_string(), sizes[0] as u64);
    let cache: Arc<FileMetadataCache> =
        Arc::new(DefaultCache::<Path, CachedFileMetadataEntry>::new(1 << 20));
    let fetch = |store: &Arc<dyn ObjectStore>, policy| {
        let store = Arc::clone(store);
        let cache = Arc::clone(&cache);
        let meta = file.object_meta.clone();
        async move {
            DFParquetMetadata::new(store.as_ref(), &meta)
                .with_file_metadata_cache(Some(cache))
                .with_store_identity(&store)
                .with_page_index_policy(Some(policy))
                .fetch_metadata()
                .await
                .unwrap()
        }
    };
    let first_a = fetch(&a, PageIndexPolicy::Skip).await;
    assert_eq!(first_a.file_metadata().schema_descr().column(0).name(), "a");
    // Store B's index request finds no footer of store A.
    let indexed_b = fetch(&b, PageIndexPolicy::Optional).await;
    assert_eq!(
        indexed_b.file_metadata().schema_descr().column(0).name(),
        "b"
    );
    assert!(indexed_b.page_index().is_some());
    assert_eq!(cache.len(), 2, "one entry per store");
    // Each store still hits its own entry.
    let again_a = fetch(&a, PageIndexPolicy::Skip).await;
    assert!(Arc::ptr_eq(&first_a, &again_a));
    let again_b = fetch(&b, PageIndexPolicy::Skip).await;
    assert!(Arc::ptr_eq(&indexed_b, &again_b));
    // Keys are per store allocation.
    let key = |store: &Arc<dyn ObjectStore>| {
        crate::metadata::file_metadata_cache_key(store, &Path::from(path))
    };
    assert_ne!(key(&a), key(&b));
    assert_eq!(key(&a), key(&Arc::clone(&a)));
}

/// A footer whose second column has no offset index, with the first
/// column's indexes loaded optionally (the reviewer's construction).
async fn partial_offset_index_footer(
    store: &Arc<dyn ObjectStore>,
    path: &str,
) -> (PartitionedFile, ParquetMetaData) {
    let batch =
        record_batch!(("a", Int32, vec![1, 2]), ("b", Int32, vec![3, 4])).unwrap();
    let size = write_parquet_batches(Arc::clone(store), path, vec![batch], None).await;
    let file = PartitionedFile::new(path.to_string(), size as u64);
    let footer = DFParquetMetadata::new(store.as_ref(), &file.object_meta)
        .with_page_index_policy(Some(PageIndexPolicy::Skip))
        .fetch_metadata()
        .await
        .unwrap();
    let mut columns = footer.row_group(0).columns().to_vec();
    columns[1] = columns[1]
        .clone()
        .into_builder()
        .set_offset_index_offset(None)
        .build()
        .unwrap();
    let group = footer
        .row_group(0)
        .clone()
        .into_builder()
        .set_column_metadata(columns)
        .build()
        .unwrap();
    let partial = footer
        .as_ref()
        .clone()
        .into_builder()
        .set_row_groups(vec![group])
        .build();
    let mut reader: Box<dyn AsyncFileReader> =
        DefaultParquetFileReaderFactory::new(Arc::clone(store))
            .create_reader(0, file.clone(), None, &ExecutionPlanMetricsSet::new())
            .unwrap();
    let mut optional = ParquetMetaDataReader::new_with_metadata(partial)
        .with_page_index_policy(PageIndexPolicy::Optional);
    optional.load_page_index(&mut reader).await.unwrap();
    (file, optional.finish().unwrap())
}

#[tokio::test]
async fn oo_a_required_request_on_a_cache_hit_is_validated_as_a_load_is() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let (file, partial) = partial_offset_index_footer(&store, "required.parquet").await;
    let cache: Arc<FileMetadataCache> =
        Arc::new(DefaultCache::<Path, CachedFileMetadataEntry>::new(1 << 20));
    cache.put(
        &test_cache_key(&store, &file.object_meta.location),
        CachedFileMetadataEntry::new(
            file.object_meta.clone(),
            Arc::new(
                CachedParquetMetaData::new(Arc::new(partial))
                    .with_store(Some(Arc::downgrade(&store))),
            ),
        ),
    );
    let index_bytes = Count::new();
    let fetch = async |policy| {
        DFParquetMetadata::new(store.as_ref(), &file.object_meta)
            .with_file_metadata_cache(Some(Arc::clone(&cache)))
            .with_store_identity(&store)
            .with_page_index_policy(Some(policy))
            .with_page_index_bytes(Some(index_bytes.clone()))
            .fetch_metadata()
            .await
    };
    // The warm optional entry serves an optional request...
    assert!(fetch(PageIndexPolicy::Optional).await.is_ok());
    // ...but not a required one: the second column has no offset index. The
    // footer rules it out before any index read (re-review finding 4).
    let error = fetch(PageIndexPolicy::Required).await.unwrap_err();
    assert_eq!(index_bytes.value(), 0);
    assert_contains!(error.to_string(), "missing offset index");

    // A required column-index request that selects only the first column is
    // valid, although the cache loads every column optionally.
    let first = PageIndexRequest {
        column_policy: PageIndexPolicy::Optional,
        column_mask: ColumnChunkMask::columns([0]),
        offset_policy: PageIndexPolicy::Required,
        offset_mask: ColumnChunkMask::columns([0]),
    };
    let metadata = DFParquetMetadata::new(store.as_ref(), &file.object_meta)
        .with_file_metadata_cache(Some(Arc::clone(&cache)))
        .with_store_identity(&store)
        .with_page_index_request(Some(first))
        .fetch_metadata()
        .await
        .unwrap();
    assert!(metadata.page_index().unwrap().offset_index(0, 0).is_some());
}

#[tokio::test]
async fn oo_planning_reads_key_the_cache_by_store_allocation() {
    let a: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let b: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "same.parquet";
    let mut sizes = Vec::new();
    for (store, column) in [(&a, "a"), (&b, "b")] {
        let schema =
            Arc::new(Schema::new(vec![Field::new(column, DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::Int32Array::from_iter_values(0..100))],
        )
        .unwrap();
        sizes.push(
            write_parquet_batches(Arc::clone(store), path, vec![batch], None).await,
        );
    }
    let file = PartitionedFile::new(path.to_string(), sizes[0] as u64);
    let cache: Arc<FileMetadataCache> =
        Arc::new(DefaultCache::<Path, CachedFileMetadataEntry>::new(1 << 20));
    let fetch = async |store: &Arc<dyn ObjectStore>| {
        DFParquetMetadata::new(store.as_ref(), &file.object_meta)
            .with_file_metadata_cache(Some(Arc::clone(&cache)))
            .with_store_identity(store)
            .with_page_index_policy(Some(PageIndexPolicy::Skip))
            .fetch_metadata()
            .await
            .unwrap()
    };
    let first_a = fetch(&a).await;
    let first_b = fetch(&b).await;
    assert_eq!(first_a.file_metadata().schema_descr().column(0).name(), "a");
    assert_eq!(first_b.file_metadata().schema_descr().column(0).name(), "b");
    assert_eq!(cache.len(), 2);
    assert!(Arc::ptr_eq(&first_a, &fetch(&a).await));
    assert!(Arc::ptr_eq(&first_b, &fetch(&b).await));
    // A clone of the same store is the same allocation; another wrapper is not.
    let alias = Arc::clone(&a);
    assert!(Arc::ptr_eq(&first_a, &fetch(&alias).await));
}

/// Re-review MJJ96 finding 1: replacing the store registered under a URL
/// must not serve the old store's footer.
#[tokio::test]
async fn oo_a_replaced_store_at_the_same_url_reads_its_own_footer() {
    use datafusion_execution::object_store::{
        DefaultObjectStoreRegistry, ObjectStoreRegistry,
    };
    let a: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let b: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "same.parquet";
    let mut sizes = Vec::new();
    for (store, column) in [(&a, "a"), (&b, "b")] {
        let schema =
            Arc::new(Schema::new(vec![Field::new(column, DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::Int32Array::from_iter_values(0..100))],
        )
        .unwrap();
        sizes.push(
            write_parquet_batches(Arc::clone(store), path, vec![batch], None).await,
        );
    }
    assert_eq!(sizes[0], sizes[1]);
    // Equal size, modification time, ETag and version.
    let file = PartitionedFile::new(path.to_string(), sizes[0] as u64);
    let url = ObjectStoreUrl::parse("memory://swap").unwrap();
    let registry = DefaultObjectStoreRegistry::new();
    let cache: Arc<FileMetadataCache> =
        Arc::new(DefaultCache::<Path, CachedFileMetadataEntry>::new(1 << 20));
    let footer = async |registry: &DefaultObjectStoreRegistry| {
        let store = registry.get_store(url.as_ref()).unwrap();
        let mut reader = cached_factory(store, Arc::clone(&cache))
            .create_reader(0, file.clone(), None, &ExecutionPlanMetricsSet::new())
            .unwrap();
        let skip =
            ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Skip);
        reader.get_metadata(Some(&skip)).await.unwrap()
    };
    registry.register_store(url.as_ref(), Arc::clone(&a));
    let first = footer(&registry).await;
    assert_eq!(first.file_metadata().schema_descr().column(0).name(), "a");
    registry.register_store(url.as_ref(), Arc::clone(&b));
    let replaced = footer(&registry).await;
    assert_eq!(
        replaced.file_metadata().schema_descr().column(0).name(),
        "b"
    );
}

fn cached_factory(
    store: Arc<dyn ObjectStore>,
    cache: Arc<FileMetadataCache>,
) -> CachedParquetFileReaderFactory {
    CachedParquetFileReaderFactory::new(store, cache)
}
