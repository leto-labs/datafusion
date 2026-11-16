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

## Families S and WS: row-group work sharing between sibling scan streams

Purpose: keep every stream of a pruned or skewed Parquet scan busy by
giving idle siblings unread whole row groups, instead of stealing whole files
only (upstream 56 still emits one morsel per file). Dependent Longbow
features: Q39-SPARSE-SCAN (S, KEEP) and PARTITIONED-SCAN-BALANCE (WS,
UPSTREAMABLE).

`SharedWorkSource` keeps each stream's planned files in its own queue. A
stream drains its own files, then takes published units, then another
owner's unopened file (the last file of the longest queue), so skewed file
groups are shared as upstream's global queue would share them. When
publishing is enabled (a `Morselizer` calls `attach_shared_work_source` and
`enable_publishing`), every file a stream takes carries a `PublishLease`
extension. The format releases it as soon as it cannot publish from that file
any more (Parquet: the reader is not eligible, its unread tail is too short
for a useful run, or its stream ended); the stream releases it at the latest
when it takes its next file or is dropped. An idle stream waits only while a
lease is held and no file or unit is left; otherwise it finishes. A stream that runs
under `TaskContext::requires_independent_partitions` never waits, so its
files hold no counted lease (nobody waits on a stream whose consumer may not
poll it); a published unit it takes drops its donor's lease. Scans that
do not publish keep upstream's single reordered queue of whole files.

Liveness obligation for callers: a waiting stream makes progress only when a
lease holder is polled. A consumer that polls one partition and then waits on
another partition of the same scan (for example Longbow's pull-island bridge
or compute-pool drivers that park a partition) must either keep polling every
partition or execute the scan under
`TaskContext::with_partition_independence(true)`, under which no stream
waits.

`ParquetMorselizer` publishes only for unordered, unlimited scans that are
not reversed or reordered by statistics, and never for files with an
external `ParquetRowSelection` or `ParquetRowPositions`. At each row-group
boundary where a sibling is idle, the stream aligns its plan with the
decoder's frontier (so a silently finished group is never donated), keeps
`ceil(remaining / parts)` row groups and at least its read-ahead window
(`ParquetFileReaderFactory::read_ahead_row_groups`), rebuilds its decoder
with upstream's row-group-local selections
(`ParquetPushDecoderBuilder::with_row_group_selections`, #25608) and only
then publishes the rest as `ParquetAccessPlan` units that keep their
selections and fully-matched flags (#23696). A stream whose dynamic filter
can still change keeps its work until the filter completes. A donor credits
the donated bytes to `bytes_processed`; the reader of a donated unit credits
none, so the metric counts each byte of the scanned ranges once. A donated
unit carries the donor's footer and its reader reuses it instead of reading
the footer again, so both decode with one object generation. A failed decoder
rebuild leaves the stream in a terminal error state. The metric
`row_group_units_published` counts the published units.

Publication waits respect the execution scope of family SO
(`TaskContext::requires_independent_partitions`): below a streaming merge
or an ordered repartition, a stream never waits for a sibling's future
publication; it still takes work that is already queued
(`file_stream_publication_waits_suppressed` counts these streams). Scans with
a declared ordering read in file order and do not share work (upstream
`FileScanConfig::preserves_file_order`, #26434).
