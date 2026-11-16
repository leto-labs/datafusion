# Longbow 56.0 patch branch

Base: Apache DataFusion tag 56.0.0-rc1, commit
`cce6374e6256b3b8892ec60674578cf32bc3a32c`.

This fork contains the retained Longbow engine patches. Each family commit
records its purpose, dependent Longbow feature and audit classification.
`third_party/README.md` in the consuming Longbow repository holds the
inventory.

## Reserved protobuf tags

Every protobuf field that a fork family adds uses a tag of 900 or more, so a
later upstream field cannot collide with it (design D6 of the Longbow change
`upgrade-datafusion-56`). Regenerate `datafusion/proto-models/src/generated/`
with the fork's generator (`datafusion/proto-models/regen.sh`) after a change.

| Message | Field | Tag | Family |
|---|---|---|---|
| `PartitionedFile` | `optional string e_tag` | 900 | D |
| `PartitionedFile` | `optional string version` | 901 | D |
| `FileScanExecConf` | `uint32 virtual_column_count` | 902 | RP |

Fields that upstream now owns and that the 55.2 fork had assigned differently
(family FC, dropped): `FileScanExecConf.preserve_order = 17`,
`ParquetScanExecNode.sort_order_for_reorder = 5`, `reverse_row_groups = 6`,
`metadata_size_hint = 7`. `ScanLimit` stays `uint32`: encoding a scan limit
above `u32::MAX` is a typed codec error.

## Family D: object identity transport and cache validation

Purpose: carry an object's ETag and version through the native plan codec,
and reject cached file metadata and statistics whose identity differs from
the current listing. Dependent Longbow feature: CORE-SOURCE-IDENTITY. Audit
classification: UPSTREAMABLE.

`PartitionedFile` encodes `ObjectMeta::e_tag` and `version` as optional
strings, so absent and empty values stay distinct. `CachedFileMetadata` and
`CachedFileMetadataEntry::is_valid_for` compare ETag and version in addition
to size and modification time: a replaced object with the same size and
timestamp is a cache miss.

The Parquet reader ties its reads to that identity: when a file's
`ObjectMeta` carries an ETag or a version, the default reader's data reads,
the footer fetch and the page-index fetch use `GetOptions::if_match` and
`GetOptions::version` (`datasource-parquet/src/identity_io.rs`), so a
same-size in-place overwrite fails the read instead of decoding new bytes
with the planned footer's offsets. Without a validator the reads are
upstream's plain range reads. A custom `ParquetFileReaderFactory` owns its
own checks.

## Family I: bounded static expression preparation cache

Purpose: reuse the simplified form of an unchanged, fully static scan
predicate across the files and row groups that one executor thread prepares,
instead of simplifying the same tree again per file. Dependent Longbow
feature: EF-PREPARATION-CACHE. Audit classification: NEEDS-DECISION (owner
decision 2026-11-16: port; upstream has no cache, #24323 adds a benchmark
only).

`PhysicalExprSimplifier::simplify` looks up a thread-local cache of at most
256 entries keyed by the input tree's address, the schema (fingerprint and
full equality, metadata included) and the evaluation context. Entries hold
weak references only, so the cache never keeps a plan's trees or literal
payloads alive. A tree that contains any dynamic filter, complete or not, is
never cached (`DynamicFilterTracking::contains_dynamic_filter`), and a result
that changed during preparation is retried once and not cached. Only roots
that simplification leaves unchanged are cached, because rewritten roots are
owned by the call site and could never hit.

The setting `longbow.execution.expr_prep_cache` (default on; parsed as a
boolean, case-insensitively, anything else keeps the default) is resolved
once per scan stream. `FileScanConfig::open_with_args` wraps the stream so
that each poll runs inside `with_preparation_cache`, which restores the
previous thread-local value on return or panic. A setting therefore never
leaks across interleaved sessions or tasks that move between threads.
`LONGBOW_EXPR_PREP_COUNTERS=<path>` appends diagnostic counters.

## Family ND: decode narrow decimals at the table's width

Purpose: read a `Decimal128` file column at the `Decimal32` or `Decimal64`
width that the table declares, instead of decoding 128-bit values and
casting each batch. Dependent Longbow feature: NARROW-DECIMALS. Audit
classification: UPSTREAMABLE.

`coerce_decimals_to_table_width` narrows a top-level decimal column only when
precision and scale are equal, so every value is unchanged, and only when the
reader decodes its physical type to that width: `Decimal32` from `INT32` or a
`FIXED_LEN_BYTE_ARRAY` of at most 4 bytes, `Decimal64` from `INT32`, `INT64`
or a `FIXED_LEN_BYTE_ARRAY` of at most 8 bytes. Other columns, for example an
`INT64` `DECIMAL(9,2)` as written by DuckDB and Trino read as `Decimal32`,
keep their file type and the schema adapter casts them (the reader would
panic or fail otherwise). The opener applies it after upstream's
`apply_file_schema_type_coercions_with_rle`. A narrow decimal literal probes a
Bloom filter as the `Decimal128` of its value. `ParquetSource::with_retyped_columns`
lets a physical rule declare the narrow types for an existing scan.

## Family RP: positional Parquet retrieval and virtual scan fields

Purpose: retrieve exact file rows by position after the opener has the
footer, and keep virtual scan fields through the native codec. Dependent
Longbow feature: TOPN-POSITIONAL. Audit classification: UPSTREAMABLE.

`ParquetRowPositions` is a thin extension on upstream's file-level
`ParquetRowSelection` and `create_initial_plan`. It rejects unsorted or
duplicate positions, positions at or past the file's last row and row groups
whose footer ordinals do not strictly increase, then builds upstream's
selection. Combining it with another access extension is an error.
`ParquetSource::with_new_metrics` gives an independently executed scan branch
fresh counters.

The file-scan codec appends virtual fields after file and partition fields
and counts them in `FileScanExecConf.virtual_column_count` (tag 902; the 55.2
fork used tag 16, which upstream now uses for `file_compression_type`). A
virtual role is never inferred from Arrow extension metadata on a physical
column. A count wider than the schema, or a field that is both virtual and
partitioned, is rejected.

Compatibility: a stock upstream decoder ignores the unknown tag 902, so it
would decode the virtual suffix as file columns. Plans that carry virtual
columns must be decoded by this fork (Longbow's codec identity changes with
the fork pins).
