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

fn cached_metadata(cache: &Arc<FileMetadataCache>, path: &str) -> Arc<ParquetMetaData> {
    let entry = cache.get(&Path::from(path)).expect("cached footer");
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
    let first = cached_metadata(&cache, "t.parquet");
    assert!(PageIndexRequest::all(PageIndexPolicy::Required).is_covered_by(&first));

    // A later open of the same generation reuses it: nothing is written back.
    let (_, rows) =
        count_batches_and_rows(open_file(&morselizer, file).await.unwrap()).await;
    assert_eq!(rows, 10);
    assert!(Arc::ptr_eq(&first, &cached_metadata(&cache, "t.parquet")));

    // A new generation of the file (other contents and size) parses again.
    let len = write_pages(&store, "t.parquet", (1..=200).collect()).await;
    let file = PartitionedFile::new("t.parquet".to_string(), len);
    let (_, rows) =
        count_batches_and_rows(open_file(&morselizer, file).await.unwrap()).await;
    assert_eq!(rows, 110, "the new generation's rows");
    let second = cached_metadata(&cache, "t.parquet");
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
