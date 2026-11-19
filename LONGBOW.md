# Longbow 56.0 patch branch

Base: Apache DataFusion tag 56.0.0-rc1, commit
`cce6374e6256b3b8892ec60674578cf32bc3a32c`.

This fork contains the retained Longbow engine patches. Each family commit
records its purpose, dependent Longbow feature and audit classification.
`third_party/README.md` in the consuming Longbow repository holds the
inventory.

The join visited bitmap (family VB, leto-mjj.68) is pending integration and
qualification.

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
| `AsOfJoinExecNode` | `bool partitioned` | 900 | AP |

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

## Family FD: final aggregate spill replay accounting

Purpose: give the fully clustered replay table of an unordered hash
aggregate a distinct reservation. Dependent Longbow feature:
CORE-SPILL-STORAGE. Audit classification: UPSTREAMABLE.

The hash table writes sorted partial states to disk. Its spill-run merge
can adjust its buffering and continues to borrow from the spillable budget.
The replay table keeps the current key and accumulator state until it can
emit the result. This table has no spill path. Its own reservation charges
that working set to the worker's unspillable budget.

Once the hash table spills, it preserves part of its existing grant as
workspace for spill-run reads and merging. The table can use the same bytes
between spills. The merge releases idle workspace when it starts streaming
the last pass. The original spillable consumer owns the grant throughout.

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

## Family PQ: prepared Parquet file scans

Purpose: let the Longbow morsel engine run the opener's complete file
preparation (schema adaptation, coercions including narrow decimals, file,
statistics, Bloom-filter, limit and page-index pruning, virtual columns) and
then decode row-group intervals with its own drivers. Dependent Longbow
feature: MORSEL-SCAN. Audit classification: UPSTREAMABLE (keep and upstream
later).

`ParquetSource::prepare_file` drives the ordinary open state machine with the
scan's reader factory and stops after `RowGroupsPrunedParquetOpen::finish_pruning`,
the first half of upstream's `build_stream`. `PreparedParquetFile` holds the
metadata, the pruned `ParquetAccessPlan`, the row groups in the requested
order (statistics reorder and reverse), the `DecoderProjection` and
upstream's `RowFilterContext` from `DecoderReadPlans`. A decoder for one
row-group interval uses a row-group-local selection
(`with_row_group_selections`, #25608) intersected with the prepared
selection, plus a fresh `RowFilter` from the shared candidates and any extra
caller predicates (which must read only prepared columns). `byte_ranges`
plans the encoded ranges of one row group, including predicate-cache batch
boundaries for selections. One runtime `RowGroupPruner` per prepared file
serves every driver: callers lock it only to refresh the pruning predicate
for the current dynamic-filter generation, evaluate outside the lock, and
cache one verdict per row group and generation. A prepared file has a runtime pruner
only while its predicate has a dynamic filter that can still change. A
prepared scan without a batch size uses the scan config's, else 8192. The
default reader ties the prepared footer and every later data read to the
file's ETag/version (family D).

## Family NFC: plan native fetches through charged page leases

Purpose: let a prepared scan skip fetching a column whose pages the caller
already holds in its page cache. Dependent Longbow feature:
NATIVE-SCAN-PAGE-CACHE. Audit classification: UPSTREAMABLE.

`PreparedParquetFile::byte_ranges_with_cached_columns` omits a compressed,
unencrypted column only when the caller's callback leases every page of all
of the column's required ranges through the file's page-cache generation
handle (Arrow family DPC); the leases and their memory charge stay with the
caller until decode ends. Partial hits fetch the whole column, including
predicate-cache batch boundaries. `byte_ranges` keeps its fetch-all behaviour.

## Family SCN: statistics constants exclude NaN-capable columns

Purpose: fix a wrong result in upstream 56. The opener replaces a column with
a literal when the file statistics show `min == max` and no nulls. Parquet
float statistics leave out NaN (and `-0.0` and `0.0` can compare equal), so a
column `[NaN, 1.0]` read back as `[1.0, 1.0]`. Dependent Longbow feature:
CORE-RESULTS. Audit classification: UPSTREAMABLE (keep and upstream later).

The fix is at the source. When the scan converts Parquet statistics of a
float leaf (FLOAT, DOUBLE, FLOAT16), it keeps min/max only where the
statistics record `nan_count == 0` (Arrow 60 writers record it):

- file statistics (`statistics_from_parquet_metadata`): min/max are Absent if
  any row group lacks that proof, so aggregates answered from statistics,
  `FilterExec` constants and file pruning cannot use them;
- row-group pruning statistics: such row groups have no bounds, so they are
  neither pruned nor fully matched (the row filter still runs);
- page-index statistics: such pages (column index `nan_counts`) have no
  bounds.

In addition, `constant_value_from_stats` derives no `min == max` constant for
a float column at all (also inside dictionaries and run-end encoding), and
the statistics-unsatisfiable proof uses the same constants. The all-NULL
constant is unchanged. Files written without NaN counts lose float pruning
(correct but slower).

All three sources also read float bounds as the Parquet spec requires: a min
of +0.0 as -0.0 and a max of -0.0 as +0.0, so a zero bound covers both zeros
even for a non-conforming writer that recorded min = max = +0.0 for
`[-0.0, 0.0]`. A widened file-level bound is Inexact. (DataFusion's
comparisons treat the two zeros as equal, so comparison pruning did not lose
rows before; the widening keeps statistics-answered MIN/MAX and other
order-sensitive consumers exact.)

Statistics that reach a scan from outside (`PartitionedFile::statistics`, for
example from a catalog or a consumer's statistics codec) do not pass through
these rules; the supplier must apply them.

## Family SO (scan part): publication waits below streaming merges

Purpose: correct order and termination for scans below order-dependent
operators. Dependent Longbow feature: CORE-ORDERING-CORRECTNESS. Audit
classification: UPSTREAMABLE.

Upstream 56 covers the scan ordering hunks (`preserves_file_order`, #26434).
This fork keeps only the publication-wait scoping of families S and WS on
DF5's `TaskContext::requires_independent_partitions`. The scan regressions in
`core/tests/parquet/stock_ordering.rs` cover a merge over a filtered union, a
declared order under a filtered limit, scan sharing kept below a blocking
sort, mixed sharing settings across partitions of one scan, and grouping sets
over constants. The join regressions of the same file belong to lane DF6.

## Family V: scan-plan advice for read-ahead readers

Purpose: let a reader that prefetches ahead of demand follow the native
stream's post-pruning read plan. Dependent Longbow feature: PF-READ-AHEAD.
Audit classification: NEEDS-DECISION, decided 2026-11-16: keep a small hook,
UPSTREAMABLE. The 55.2 Arrow read-ahead patch is dropped; Longbow plans
read-ahead from upstream's `ParquetPushDecoder::scan_plan`, but the native
stream's decoder is private to the opener, so this hook hands its plan out.

`ParquetFileReaderFactory::create_reader_with_scan_plan_advice` (defaulted:
no advice) returns the reader and an optional `ScanPlanAdvice`. The stream
calls `ScanPlanAdvice::advise(decoder.scan_plan())` after it builds its
decoder and after every rebuild (filter toggle, donation). It gives no advice
while the predicate has a dynamic filter that can still change. The fork
contains no read-ahead logic. Prepared scans (family PQ) own their decoders
and call `scan_plan` themselves.

## Family FC (dropped): native scan fields are upstream's

Family FC of the 55.2 fork is dropped: upstream 56 serializes
`FileScanExecConf.preserve_order` (17), `ParquetScanExecNode.sort_order_for_reorder`
(5), `reverse_row_groups` (6) and `metadata_size_hint` (7). Its regressions run
on the upstream codec in `datafusion/proto` (`file_scan_serde_*`,
`roundtrip_parquet_inexact_sort_settings_and_order_intent`,
`parquet_reverse_without_reorder_and_malformed_reorder`).

Regression from 55.2: the fork widened `ScanLimit` to uint64, upstream keeps
uint32. A scan limit above `u32::MAX` is now a typed encode error
(`FileScanConfig: limit value ... is out of range for the plan wire format`)
instead of a round trip. Longbow keeps the typed error (owner decision). An
empty reorder ordering now decodes as "no hint" instead of an error.
`ParquetSource::reverse_row_groups` and `sort_order_for_reorder` are public
read-only accessors for the consumer's scan statistics.

## Family QI: continuous percentile interpolation

Purpose: keep the full binary64 rank fraction in `percentile_cont` /
`quantile_cont` instead of truncating it to six decimal places, use
DuckDB's weighted endpoint formula for DOUBLE, and reproduce DuckDB's
decimal interpolation. Dependent Longbow feature:
MARKETDATA-SQL-CORRECTNESS. Audit classification: UPSTREAMABLE.

Upstream 56 moved the interpolation into a `PercentileInterpolator` trait.
The family changes only `FloatInterpolator`, which serves the scalar,
grouped, DISTINCT and sliding-window accumulators, and in 56 also `median`
(upstream routes `median` through the percentile accumulators). For Float64
the result is `lower * (1 - f) + upper * f` (DuckDB 2.0.0 DOUBLE: no overflow
in `upper - lower`; the median of two values is their correctly rounded
midpoint). Float32 and Float16 keep widened delta arithmetic
`lower + (upper - lower) * f` with the full fraction, as DuckDB's FLOAT
overload does; identical infinite FLOAT endpoints therefore give NaN.

Decimals (amendment, leto-zca.21; the 55.2 counterpart is 37461157): upstream
56's `DecimalInterpolator` quantized the fraction to six decimal places
(DECIMAL(18,2) p0.9 of `0..8 * 1e6` gave 6299999.00). The family reproduces
DuckDB 2.0.0 bit for bit instead. DuckDB computes the binary64 rank
`RN = (n - 1) * p` in the requested sort order and interpolates from the
value at `FRN = floor(RN)` (`first`) toward the value at `CRN = ceil(RN)`
(`second`) with the binary64 fraction `RN - FRN` (`QuantileInterpolator` in
`quantile_sort_tree.hpp`; `InterpolateOperator` in `interpolate.hpp` and
`src/function/scalar/operator/arithmetic.cpp`). The result keeps the input
decimal type, precision and scale. The arithmetic follows DuckDB's storage
class for the declared precision, not the Arrow storage width:

- precision <= 18 (int16/32/64): `fma(second - first, fraction, first)` on
  the unscaled values as doubles (the pinned binary contracts
  `lo + (hi - lo) * d` to an FMA), truncated toward zero;
- precision 19..=38 (HUGEINT): the exact i128 `second - first`, converted to
  double by DuckDB's two-limb rule, times the fraction, truncated toward zero
  and added to the exact `first`. If the difference overflows i128,
  `fma(first, 1 - fraction, second * fraction)` in double, truncated. A
  double outside the i128 range is an error ("percentile_cont decimal
  interpolation overflow"), as is a `Decimal256` value outside i128;
- precision > 38 (`Decimal256` only): DuckDB has no such type, so there is
  no reference. The family keeps the HUGEINT rule in 256 bits
  (`i256::to_f64`, truncating `i256::from_f64`).

`WITHIN GROUP (ORDER BY ... DESC)` keeps `p` and interpolates in the
descending order, as DuckDB does (precision > 18 then rounds toward the
larger value); it is not the ascending `1 - p` that floats and the 55.2
counterpart use. Binary64 rounding can carry a precision <= 18 result past
the declared precision: the median of two 999999999999999999 values as
DECIMAL(18,0) is 10^18 in DuckDB and here. The coefficient is returned as
computed (it fits the storage type) and is not validated against the
declared precision; 55.2 does the same.

Checked against the pinned DuckDB 2.0.0 on the 55.2 probe sets (100 narrow,
20 overflow) and 10 DESC probes: 130/130.

Regressions (fail on stock 56 with 6_299_999 for 0.9 of `0..8 * 1e6`; the
decimal ones fail on the first QI commit 0fee7b30c):
`scripts/vendored-tests.py datafusion-functions-aggregate -E 'test(percentile_cont::tests)'`.

## Family RBC: owned boundary prefix on upstream Top-K heaps

Purpose: keep an owned 15-byte prefix of the current worst encoded key in
upstream 56's `PartitionHeap` (shared by ROW_NUMBER and RANK in
`PartitionedTopKExec`), refresh it after insertions and replacements, and
compare candidate keys against it first; equal truncated prefixes fall back to
the full root key, so decisions match full byte ordering
(`topk::order_prefix::tests::prefix_decisions_match_full_encoded_order`).
Dependent Longbow feature: WINDOW-TOPK. Classification: UPSTREAMABLE.
Kept by the task 3.6 screen (leto-zca.23): H2O group-by SF100 q8,
rbc/stock 0.762 (pull, n=8, 0.724-0.815) and 0.777 (native, n=6); TPC-DS SF10
rank/row_number queries unchanged (0.994, 1.003). The companion family CR was
dropped: on upstream's `PartitionSlot` layout it regressed q8 to 1.62.

## Family CG: native skip probe on the chained grouping-sets root

The chained root is a single-set Partial aggregate. It uses the native
skip probe when its input reduces poorly. Skipped rows still carry
mergeable aggregate state for derived levels and the final consumer.
Expanding grouping sets follow the upstream probe guard. The chained-root
metric `chained_root_skipped_rows` and the regression
`chained_grouping_sets_root_skip_probe_follows_reduction` cover low-reduction
and reducing roots (leto-zca.31).

## Family PRS: partial-reduce aggregation over grouping sets

Purpose: let a `PartialReduce` aggregate whose group by holds grouping sets
run on the default (migrated) aggregate streams. It expands each partial-state
row into every grouping set and merges it, as the raw-input stage expands each
raw row; the hash table already evaluates and merges every set
(`aggregate_batch_inner`), so the change only narrows the execution guard to
the final stages, which still need `PhysicalGroupBy::as_final`. The input holds
ungrouped keys and a placeholder column where a state input carries the
grouping id. Empty-set rows come only from raw input
(`init_empty_grouping_sets`), so the caller supplies the grand total of an
empty input. Upstream 56 rejects the plan at execution ("Grouping sets must be
expanded before PartialReduce aggregation"); the 55.2 legacy stream accepted
it. Dependent Longbow feature: EAGER-AGGREGATION (`add-eager-aggregation`:
grouping sets merge their states in a `PartialReduce` aggregate). Audit
classification: UPSTREAMABLE. Tests:
`aggregates::partial_reduce_sets_tests` (ROLLUP and CUBE against the stock
two-phase plan, empty input, early emission under memory pressure).

## Family SVC: inline string-view constant comparison

Purpose: evaluate `column = 'constant'` and `<>` on a `Utf8View` column against
a non-NULL constant of 5 to 12 bytes by comparing each 16-byte view, as two
64-bit words, with the constant's inline view (length plus zero-padded
bytes); no payload is read, also when the array owns data buffers for longer
strings (`expressions/short_string.rs`, called first from
`BinaryExpr::evaluate`). Arrow 60.1 compares whole views only when neither
side owns a data buffer, and its scalar fast path stops at four bytes, so
Parquet-decoded columns compared with 5 to 12 byte constants take its per-row
length and prefix branches. Constants of at most four bytes keep Arrow's
low-word path, which is faster than this kernel (0.26 vs 0.54 ns/row).
TPC-DS q59 and q2 evaluate seven `d_day_name = '<weekday>'` conditions per row
(leto-zca.34: 1.6-2.1 vs 0.53 ns/row in a kernel benchmark; q59's aggregate
arguments 7.8 s on stock 56 vs 4.0 s on 55.2). The task 3.6 screen
(leto-zca.23) dropped the earlier full-width port on TPC-H SF30, whose
constants are at most four bytes and slower on it (Q12 1.087). Dependent
Longbow feature: STATIC-FILTER-EVALUATION. Classification: UPSTREAMABLE
(Arrow kernel). Tests: `short_string::tests` (differential against Arrow over
NULLs, UTF-8, NULs, slices, buffered arrays and both operand orders; the path
taken for each constant length).

## Family E: build keys and memory consumer of hash-join membership lookups

Purpose: let Longbow replace a kept hash-join residual's `HashTableLookupExpr`
with a key filter (an exact key bitmap or a blocked key Bloom filter) built from
the join's own build keys and charged to the join's memory consumer. Dependent
Longbow feature: WHOLEPLAN-RESIDUALS (`longbow-plan` `early_filter/membership`).
Classification: KEEP (reworked on 56's `HashTableLookupExpr`).

Accessors (all `#[doc(hidden)]` through `HashTableLookupExpr`):

- `with_build_side(keys, memory)`: one entry per map given to
  `HashTableLookupExpr::new` (so per build partition of a `Lookup::Routed`
  lookup, one for `Lookup::Single`), each holding that partition's key arrays
  weakly in one or more chunks (Longbow's native join keeps one chunk per
  driver), plus an optional empty reservation of a build's memory consumer. Equality, hashing, display, evaluation and protobuf
  encoding ignore it; `with_new_children` keeps it.
- `build_keys()`: the key chunks of every non-empty partition while the join
  holds them; `None` when a non-empty partition's keys are unknown or released.
- `build_memory()`, `on_columns()`, `random_state()`.

The join publishes the normalized build keys (`JoinBuildData::values`, now an
`Arc<[ArrayRef]>`) weakly through `PushdownStrategy::Map`'s build side and an
empty reservation of its `HashJoinInput` consumer. The 56 accumulator carries
both through `PartitionData` into the published lookup: a collect-left or
single-partition lookup gets its build's side; a merged routed lookup gets each
hash-table partition's keys, owns the IN-list values of IN-list partitions, and
names the first reported memory consumer. Prepared builds
(`PreparedHashJoinBuild`) outlive any one join and expose no build side. 56's
`Lookup::Routed` replaces the 55.2 Longbow `RoutedLookup` evaluator, so the
55.2 `map()` accessor is not ported.

Kept by the task 3.6 screen (leto-zca.23): on TPC-H SF100, 55.2 with E against
stock 56 was 0.724 of total wall time in the pull strategy and 0.812 in the
native strategy (queries 1, 3, 4, 5, 6, 7, 8, 10, 12, 20, 21); the residual join
filters of Q3, 4, 5, 7, 8, 10, 20 and 21 used 25.3 s of CPU with E and 142.1 s
on stock 56 for the same input rows. Replaces the 55.2 fork commit 9bca3f47e.

Regressions: `scripts/vendored-tests.py datafusion-physical-plan -E
'test(build_side_accessors_follow_partitions) |
test(lookups_expose_partition_build_keys_and_memory) |
test(lookup_exposes_build_keys_and_memory_while_the_join_holds_them) |
test(prepared)'`.

## Family VB: visited bitmap of hash joins without a lock

Purpose: let every probe partition of a hash join that returns build rows
(`Left`, `LeftSemi`, `LeftAnti`, `LeftMark`, `Full`) mark its matched build
rows without a lock. Upstream keeps the marks in a
`Mutex<BooleanBufferBuilder>` and takes the mutex once per probe batch, with
one bit set per matched row inside it. With a small build side and a large
probe side the critical section is a fixed share of each partition's work,
and the lock bounds the number of partitions that run. TPC-DS q14 at SF100
(`item` semi-joined with 880 million fact rows per statement) ran on 19 of 48
cores. Dependent Longbow feature: PULL-JOIN-SCALING (a proposed code: pull
hash joins at 48 threads; the native join
has its own matched bitmaps). Classification: UPSTREAMABLE.

Change (`datafusion/physical-plan/src/joins/hash_join/`):

- `visited.rs` (new): `VisitedBitmap`, one `AtomicU64` per 64 build rows.
  `set` reads the word first and writes only a bit that is not set, so a
  build row that many probe rows match causes one write. `snapshot` returns
  the `BooleanBuffer` that the emission of build rows already used.
- `exec.rs`, `exec/prepared.rs`: `JoinLeftData` holds a `VisitedBitmap`. The
  reservation is the bitmap's size rounded up to 8 bytes.
- `stream.rs`: the probe marks through `set`. The final partition takes
  `snapshot` after `ProbeCompletion::report_completed` grants emission.
  The chained acquire-release decrements publish all probe marks, so the
  word operations use relaxed ordering.

The nested loop join and the `NOT IN` state keep `SharedBitmapBuilder`.

Tests: bitmap lengths around word boundaries, overlapping marks, bounds
checks, and a 48-partition `LeftAnti` join whose final pass emits the exact
unmatched rows once. The `datafusion_loom` model checks that the last
partition sees every relaxed mark before any thread join.

Evidence: Longbow lane `leto-mjj.65`,
`docs/research/idle-cores-2-20261118.md`, section 6.

## Family JT: prefetching join table and partitioned build

Dependent Longbow feature: JOIN-TABLE (scale-up task 5.2 V5). Five commits,
each with its own purpose; the DataFusion 56 port carries them one by one.

### JT 1: fallible `update_from_iter`

Purpose: a row beyond a map's chain array becomes an error instead of an
out-of-bounds panic in a worker. Classification: UPSTREAMABLE.
`JoinHashMapType::update_from_iter` returns `Result<()>`. `JoinHashMapU32`,
`JoinHashMapU64` and the shared build's sharded map refuse a row at or
beyond `next.len() + deleted_offset` (rows before it are inserted);
`update_hash` and the pruning map propagate the result. Every production
caller sizes the map to its rows, so no query changes.


### JT 2: chain-head table with prefetch and 2 MiB pages

Purpose: take the probe and build off dependent cache and TLB misses.
Classification: UPSTREAMABLE. The chain heads of `JoinHashMapU32`,
`JoinHashMapU64` and the shards live in `HeadTable`: one power-of-two slot
array with linear probing at load at most one half. A `u32` slot is one word
(the low 32 hash bits over row + 1); a `u64` slot is two (hash, row + 1). The
slot is the top bits of `hash * golden`. The chain encoding is unchanged.
Two hashes with equal low 32 bits in one probe run share a chain; every
consumer compares keys or uses the answer as a filter that may pass extra
rows (measured at SF100: no extra candidate on TPC-H Q9 or TPC-DS q78).
Crafted keys that share one chain are cheaper to make with 32-bit tags (about
2^32 work per key instead of a 64-bit collision) and make a probe of them
quadratic; Longbow is single-tenant, so this is recorded, not mitigated.
Tables above 512 KiB prefetch the slot 16 probe rows ahead and keep 16 build
rows in flight in order (one `_mm_prefetch` with a SAFETY comment). On other
targets, aarch64 (Graviton) included, the prefetch is a no-op; Graviton is
unmeasured. With `LONGBOW_JOIN_TABLE_HUGE_PAGES=on` (off by default), arrays
from 4 MiB on get `MADV_HUGEPAGE` on their 2 MiB-aligned interior. It is off
by default because the advice outlives a freed table: the allocator reuses
the range for other operators, and their faults there allocate huge pages
with synchronous compaction (worker-01, TPC-H SF100, the same code with and
without it: Q9 pull 17% faster, q10 pull 27% slower in a suite process,
suite sums about 3% faster). A table takes up to twice its capacity and then refuses a new hash
with an error. A `u64` map reserves the larger of hashbrown's estimate and
its table (`collect_left_input`). The hashbrown functions stay for the
pruning map. Tests compare the table with the hashbrown reference
(duplicates, NULL keys, every resume limit, both widths, both prefetch modes)
and overfill it. On 56 the maps keep upstream's `chained` flag (set by any
insert that finds a head): rows left out for NULL keys keep the unique-key
path, and the head-table probe resumes with `ProbeOffset`.

### JT 3: partitioned build above 4 MiB

Purpose: build a table larger than the caches with inserts that hit L2.
Classification: UPSTREAMABLE. `build_in_memory` (hash_join/exec.rs) builds a
`u32` map above 4 MiB with `build_partitioned`: every row's hash, then a
stable counting sort of (hash, row) into partitions of at most 256 KiB of
slots, then the inserts partition by partition. Rows, row numbers, NULL
handling and chains equal the row-by-row loop (rows at their position in the
forward concatenation, batches inserted from the last, each from its last
row). The transient 20 bytes per row are reserved beside the build's other
charges and returned before the concatenation copy is charged;
`build_mem_used` records the peak with them when it is the larger. A pool
that refuses them gets the row-by-row build, not an error: CollectLeft cannot
spill, and a build that fits without partitioning must not fail. Prepared
builds (`PreparedHashJoinBuild`) take the same path and skip the per-batch
scratch when partitioned. Tests: the two builds against each other and
against hashbrown with forced 32-bit tag collisions and NULL keys; a refused
reservation touches nothing; a one-batch CollectLeft join whose pool refuses
the transient arrays returns the rows of an unbounded run.

### JT 4: prefetching `ArrayMap`

Purpose: overlap the misses of a probe into a large direct array.
Classification: UPSTREAMABLE. Above 512 KiB the probe prefetches the slot 16
rows ahead (an out-of-range key prefetches nothing; a no-op on aarch64, where
Graviton is unmeasured), and with `LONGBOW_JOIN_TABLE_HUGE_PAGES=on` the
array is advised to use 2 MiB pages. The
test covers unique and duplicate keys, misses, out-of-range keys and NULLs
above the prefetch size.

### JT 5: prefetched build keys in the key comparison

Purpose: overlap the misses of the key comparison, which reads the build key
of each candidate pair at a random build row. Classification: UPSTREAMABLE.
`equal_rows_arr` (single-column fast path and multi-column comparator)
prefetches the fixed-width build key values of the candidate 16 pairs ahead
for key columns above 512 KiB (a no-op on aarch64, unmeasured); strings and
other types compare as before.
The test compares single- and two-column keys above that size, with NULLs
and unequal candidates, against a direct comparison.

### JT 6: own mappings for advised tables; the advice on by default

Purpose: make the 2 MiB-page advice of parts 2 and 4 safe to leave on.
Classification: UPSTREAMABLE. The advice of mjj.16 went on allocator memory
and outlived the table: the allocator reused the range for other operators,
whose faults there allocated huge pages with synchronous compaction (TPC-H
SF100 on worker-01: q10 pull 27% slower in a suite process, which advises
nothing itself). Now an array of a head table, a chain or an `ArrayMap` from
4 MiB on gets its own anonymous mapping (`ZeroedArray`), 2 MiB aligned and
advised `MADV_HUGEPAGE`, and is unmapped when dropped, so the advice ends
with the table. The mapping is exactly the array's bytes rounded up to
base pages (the alignment slack and the tail are unmapped again), so a 2 MiB
region that the array fills only in part gets base pages: the resident bytes
stay below the charged bytes plus one base page. Smaller arrays, other
targets, a refused mapping or a refused advice use a `Vec`, as do kernels
whose THP mode is `never` or that have no THP. The advice is on unless
`LONGBOW_JOIN_TABLE_HUGE_PAGES` is `off`, `0`, `false`, `no` or `disabled`
(any case; another value warns and leaves it on). `ZeroedValue` is an
`unsafe trait`: its zero bit pattern must be a valid value.

Host dependency: with the advice on, the first touch of an advised table can
compact memory synchronously in the query thread. How often depends on the
host's THP `defrag` setting (`madvise`, the kernel default, compacts for
advised mappings) and on free contiguous memory. On worker-01 (160 GiB
partition) the TPC-H SF100 suite showed 1,265-1,802 compaction stalls and
12,228 THP faults per process, with no regression; a host with a tight
memory limit or fragmented memory can see more. Watch `compact_stall` in
`/proc/vmstat` when it matters.

Tests: a mapped array is zeroed, writable, 2 MiB aligned, in an `hg` mapping
of exactly its pages, and unmapped on drop; a fully written mapped array of
4 MiB + 4 B (and + 64 KiB, and 6 MiB) is resident for at most its bytes plus
one base page; a refused mapping (test hook) falls back to a zeroed heap
array; dropped tables (after a full and after a failed build) leave no
advised mapping; an `ArrayMap` of 4.8 MB is mapped; the switch's values.

### JT 7: one probe loop per prefetch mode

Purpose: keep the probe of small tables at its pre-JT cost. Classification:
UPSTREAMABLE. Parts 4 and 5 checked per probe row (or candidate pair) whether
to prefetch. On TPC-DS SF100 q14 (worker-01, 48 threads, pull), whose probe is
almost all `ArrayMap` lookups into small arrays, that check cost 12-18% of the
probe (14.4-14.7 to 16.4-17.3 CPU-s per execution) at every prefetch threshold
from 512 KiB to 16 MiB, while raising the threshold made TPC-H Q9 and TPC-DS
q78 slower (Q9 probe 44 / 48 / 70 ns per row at 512 KiB / 8 MiB / 16 MiB;
q78 30 / 32 / 34). The `ArrayMap` probe and both key-comparison loops now
choose their loop once per call, monomorphized on the prefetch decision, as
the head-table probe already did, and so does `heads_contain_hashes`; the
512 KiB threshold stays. Tests: both loops of the `ArrayMap` probe, the
head-table probe, `heads_contain_hashes` and the key comparison (prefetch
forced on and off by a test hook) give equal pairs and resume offsets on the
same input, at probe lengths 15, 16, 17 and 33 and limits 1, 15, 16 and 17.

## Family AJ: DuckDB's ASOF join forms

Purpose: plan DuckDB's `ASOF [INNER | LEFT [OUTER]] JOIN r ON ... | USING
(...)` to upstream's `LogicalPlan::AsOfJoin`, as the market-data suite's
KX SQL writes it (ADR 0054, decision 1). Dependent Longbow feature:
MARKETDATA-ASOF (MD1 T10-T12, B02; KX 62, 63, 79-84). Classification:
UPSTREAMABLE (the token pass is temporary; the production belongs in
sqlparser).

- `datafusion-sql` `asof_tokens::rewrite_duckdb_asof`, run by
  `DFParserBuilder::build` and public for callers that parse with sqlparser
  directly: after `ASOF [INNER | LEFT [OUTER]] JOIN <relation>` followed by
  `ON` or `USING`, it removes the join kind and inserts
  `MATCH_CONDITION (__datafusion_asof_inner | __datafusion_asof_left)`. The
  relation's extent comes from `Parser::parse_table_factor`; `MATCH_CONDITION`
  is reserved for implicit aliases, so an unaliased table keeps its qualifier.
  `ASOF` is an operator only in join position (not after `AS`, `FROM`, `JOIN`,
  `,` or `(`). Nested ASOF joins are rewritten from the last to the first.
  Snowflake's form and other SQL are unchanged.
- `datafusion-sql` `relation/asof`: a marked match condition plans DuckDB's
  meaning. `ON`: exactly one `<`, `<=`, `>`, `>=` conjunct between the sides,
  in either operand order, normalized to `left op right`; the other conjuncts
  are equalities. `USING (k..., t)`: equality on all but the last column and
  `left.t >= right.t` on the last, resolved through each input's `USING`
  bindings (chained joins). Inner: a constant marker column, under a name that
  neither input has, on the right input after the constraint binds;
  `marker IS NOT NULL` above the join, then a projection that removes it.
- `datafusion-expr`: `AsOfJoin::ordered_using` (`with_ordered_using`, kept by
  every rebuild). `using_columns` binds the last `USING` column too, merged into
  the binding of its left column, and `*` keeps the left value of each `USING`
  column of such a join in the visible FROM tree (not below a subquery alias).
- `datafusion-proto`: the logical encoder refuses `ordered_using`.

Regressions: `scripts/vendored-tests.py datafusion-sql asof_tokens`; Longbow
`tests/sql/asof_join.slt` (DuckDB 2.0.0 rows).

## Family AP: partitioned mode of `AsOfJoinExec`

Purpose: an ASOF join with an equality key streams both inputs instead of
collecting the right one (ADR 0054, decision 2). Dependent Longbow feature:
MARKETDATA-ASOF (`longbow-plan` `AsOfPartitioned`). Classification:
UPSTREAMABLE (upstream's module documentation names repartitioning both inputs
as future work).

- `AsOfJoinMode { Broadcast, Partitioned }` and `with_mode`; `Partitioned`
  requires an equality key, requires co-partitioned `KeyPartitioned` inputs,
  keeps the left partitioning and merges partition *i* of each side without a
  reservation. Rebuilds keep the mode. Display adds `mode=Partitioned` only for
  that mode. Accessors `left`, `right`, `on`, `match_condition`, `projection`,
  `mode`.
- Protobuf `AsOfJoinExecNode.partitioned = 900`; prost and pbjson regenerated.
  The generator's include path does not match the moved proto directory; it
  was run with a `PROTOC` wrapper that adds `-I.`.
- The tie rule of both modes: the last eligible right row in the right input's
  order (`>=` and `>`), across batch boundaries.
- Both modes: the pending output rows of a side reference at most 32 distinct
  input batches before a flush; equality keys with a floating-point leaf are
  rejected.
- Both modes accept either NULL placement: the input requirement lists the
  NULLS FIRST and the NULLS LAST ordering as hard alternatives, the one the
  input already satisfies first (distribution enforcement reads only the
  first). A Parquet file's declared NULLS LAST order then needs no sort.

Regressions: `scripts/vendored-tests.py datafusion-physical-plan asof`;
`scripts/vendored-tests.py datafusion-proto --test proto_integration asof`.

## Family JS: hybrid hash join spill in `HashJoinExec`

Purpose: an eligible `HashJoinExec` answers a refused build reservation by
writing hash partitions of its build side, and their probe rows, to spill
files, and joins each written partition after the probe with a stock
`CollectLeft` `HashJoinExec` over the two files. The fork's hash join failed
with `Resources exhausted` when a build did not fit (TPC-H SF1000 q7, q9,
q18, q21 and q22 in a 100 GiB pool, 2026-11-18). Dependent Longbow feature:
JOIN-SPILL (OpenSpec change `add-join-spill`, ADR 0068). Classification:
UPSTREAMABLE (related upstream: the hybrid-join epic apache/datafusion#24768;
#26182, in this base, keeps build rows when a build fails; #27399, #27401 and
#26442 rewrite the files it hooks).

Off by default: `datafusion.execution.enable_hash_join_spilling = false`.
With it off no code path changes, except that
`set_dynamic_filter_partitions` also records the range for a join without a
dynamic filter (the last binding wins; it never fails), and
`set_open_probe_streams` records only a flag. With it on, a join
that fits runs the
code it runs today: nothing is partitioned, copied or reserved before the
first refusal, and no spill metric is registered.

Change (`datafusion/physical-plan/src/joins/hash_join/`):

- `spill/` (new): eligibility, the partition function, the build-side
  relief, the probe routing, the coordination of the written partitions and
  the leaf `SpillFilesExec` that reads a partition's files back.
- Hook 1, `collect_left_input` (`exec.rs`): a refused reservation of an
  eligible join moves the retained batches into 16 partitions and writes
  whole partitions, largest first, until it released the refused bytes and
  one eighth of what it held. From then on it keeps a workspace of
  `hash_join_spill_reservation_bytes` (16 MiB), moved out of the charge that
  writing releases (`MemoryReservation::split`, no pool round trip); only
  what that did not cover is grown without refusal. Later rows of a written
  partition go to its file through a buffer. At the end of the input the
  table is built over the partitions that stayed; `build_in_memory` now
  borrows the reservation, so a refused table leaves the rows charged (the
  #26182 contract returns the rows) and half of them are written before it
  builds again. A table refused after the whole input was admitted enters
  spill mode the same way, with the rows still charged.
- The floor: when the rows in memory and their table fit in what the join
  holds (its charge and its workspace), a refused growth is covered by the
  workspace (grown before the workspace shrinks; the workspace is refilled
  from charge that later writes release) and a refused table is built on a
  private pool, its size moved from the workspace into the table's charge,
  which the table releases when it goes. A written partition's join in a
  full pool therefore builds a small table instead of spilling to the depth
  limit; an empty table always fits.
- Hook 2, `fetch_probe_batch` (`stream.rs`): rows of written partitions go
  to this stream's probe file of the partition; rows whose key cannot match
  stay in the batch. The write buffers are charged with `try_grow`; a piece
  the pool refuses is written at once with its buffer.
- Hook 3, after the in-memory pass (`stream.rs`): the stream drops its
  reference to the in-memory table and joins the written partitions with
  stock joins over their files. No stream waits for another (a split
  attempt may run its partitions one after another, Longbow's writers claim
  the next partition when the last one ends). Each stream joins its own
  probe file of each written partition (an own run): Inner, Right,
  RightSemi, RightAnti and RightMark with their own type, Left as Inner and
  Full as Right, so a stream gives every row its probe rows produce whatever
  the other streams do, as the stock join. A partition's join is shared with
  the streams that run it at the same time (each executes its one partition
  over the file it stages) and built again from the retained build file for
  a stream that comes later. Build rows (Left, LeftSemi, LeftAnti, LeftMark,
  Full): a stream finishes and registers its probe files when its probe
  input ends, before it reports its completion to the in-memory table; the
  stream that the table's completion count chooses (the stock join's last
  probe partition) emits the in-memory build rows and then those of each
  written partition, with a join over every stream's probe file of it
  (LeftAnti, its rows emitted as unmatched build rows, for Left and Full;
  the join type itself otherwise). Without such a stream (a stream stopped
  early, never ran, or a split attempt bound fewer partitions than the probe
  has) no build row is emitted, as in the stock join.
- Members: a `CollectLeft` build's members are a split attempt's bound
  partitions (`set_dynamic_filter_partitions`) or every probe partition;
  each executes once. A stream of a partition outside the range, a
  partition that probes twice, or more streams than members are internal
  errors, as is a written partition's build file that is gone while a
  member still joins it. Once every member ended its in-memory pass the
  exec lets go of the build it holds for its streams (`OnceAsync::release`,
  `joins/utils.rs`); a partition's build file is deleted when every member
  finished, skipped or left it and the build-row pass took it.
  `HashJoinExec::set_open_probe_streams` (new, public) declares that any
  number of streams execute any partition, each over its own probe input
  (Longbow's native hand-off): such a join spills only if it emits no build
  rows and keeps its table and build files until it is dropped. A shared
  partition join counts the streams its parent admits; once every admitted
  stream ended its in-memory pass it lets go of its table and admits no
  more (a later stream gets a new join).
- `shared_bounds.rs`: `PushdownStrategy::BoundsOnly`. A build that wrote a
  partition publishes its bounds (which saw every build row) and no
  membership form; in `Partitioned` routing only that partition's branch.
- `datafusion/common/src/config.rs`: the six options of the design.

Partition function: `(hash * K[level]) >> (64 - log2(16))` over the join
hash (`HASH_JOIN_SEED`), with one odd multiplier per level that is not the
table's slot multiplier. A written partition's join is eligible again, one
level deeper, up to `hash_join_spill_max_depth` (3). Rows whose key cannot
match (a NULL under `NullEqualsNothing`) never enter a key partition: build
rows of this kind form one more partition when the join emits unmatched
build rows, and are dropped otherwise.

Eligibility (`JoinSpill::decide`): the option, a disk manager that can
create files, not null-aware, an unordered and bounded probe input, no
prepared build, a `CollectLeft` or `Partitioned` mode, and for a join that
emits build rows a `CollectLeft` membership that is not open. The build
consumer is spillable only for an eligible join. An ineligible join's own
refused build fails as before, with the context "The hash join did not
spill because ..."; a refusal that comes out of its build input keeps its
text (the wrapped input sets a flag; no error text is matched). A join above
`hash_join_spill_max_depth` (at most 7, one multiplier per level) is
eligible for its floor only: it writes no partition, and a refusal its floor
cannot cover fails with the reason: one key value fills the partition when
it kept more than half of the rows its parent divided, the pool had no room
for it otherwise.

Metrics, registered at the first relief: `spill_count`, `spilled_bytes`,
`spilled_rows`, `build_spilled_rows`, `build_spilled_bytes`,
`probe_spilled_rows`, `probe_spilled_bytes`, `spilled_partitions` and the
gauge `spill_depth` (levels that wrote a partition; 1 for the first level
only; a join that only used its floor wrote none), one
gauge per join, so its aggregate over the streams is the deepest level. The
joins of written partitions report into the first-level join's metrics.

Review fixes from the Longbow oracle (`longbow-testkit` `join_spill`): the
leaf reads its files unbuffered in the consumer's poll (a buffered read's
producer task, stopped with its attempt, ends the stream as if the file
ended); each written partition's files move to its join and are deleted
when that partition is joined; the unmatchable partition is emitted from
its file as unmatched build rows by one stream, without a table; and an
empty table refused after every row was written is charged without refusal.

Fixes from the task 2.9 review (`MJJ90-REVIEW-20261119`): the
coordination above replaces a step barrier that stalled a split attempt
running fewer partitions at once than its range (and, one level down,
counted members that had left); refused tables keep the rows charged
(the pool went 1.30 to 1.65 times over its limit when another consumer took
the released bytes); the floor; the workspace and probe buffers grown
without refusal only for what released charge did not cover; the empty
table's charge kept for `CollectLeft` joins; a sub-join's files no longer
taken before the join exists.

Fixes from the re-review (`MJJ90-REREVIEW-20261119`, lane `leto-mjj.99`):
own runs and the completion-count choice above replace a last-member rule
that dropped the other streams' rows of written partitions after one stream
left (Left 8,525 against 5,531 rows); the member checks replace a trusted
count that deleted a build file early when more streams ran than were bound
(7,585 against 6,010 rows, no error); open and admitted membership; a
floor's charge released with its table rather than with the build, and the
workspace refilled after a borrow (the peak in a pool that another consumer
keeps full went from 5.58 to at most 4.62 workspaces above the limit with
four streams); the floor above the depth limit and the hot-key test by
share; the refusal context by flag.

Not in this family (`add-join-spill` section 3): build chunks with probe
replay for a partition that cannot be divided (one hot key fails at the
depth limit; `hash_join_spill_max_replay_passes` is reserved), and role
reversal of a written partition whose probe is the smaller side.

Tests (`spill/tests.rs`, `exec.rs`, `core/tests/memory_limit`): every join
type, both null equalities, with and without a join filter, in `CollectLeft`
with one and three probe streams and `Partitioned` mode, against the
unbounded join, under forced spilling and under a pool below the build; a
second spill level; the depth limit; each ineligible reason; a join that
fits registers no spill metric and charges the same build memory; streams
that leave early or run in a split attempt, also one partition at a time,
finish and give the stock join's rows; more streams than bound fail;
open probe streams of partition 0; the reviews' peak-reservation, lifecycle
and reproduction tests; the
partitions to join are decided after every stream registered its files; a
spilled build publishes bounds only, in both modes, and filters no matching
probe row; the `join_by_key_*` memory-limit queries spill and answer in three
quarters of their build memory, where spilling off fails, and in half of it
fail at the depth limit with that reason (one `service` value holds about
half of the rows: the case of `add-join-spill` section 3).

Evidence: Longbow lanes `leto-mjj.90`,
`~/data/longbow/benchmarks/MJJ90-20261119/`, and `leto-mjj.99`,
`~/data/longbow/benchmarks/MJJ99-20261119/`.

### Family KB: chunked key-range bitmap

UPSTREAMABLE. Longbow's native join passes the concatenated keys of each
driver directly to `KeyRangeBitmap::try_new_chunks`. Each chunk contributes
bits to the global range after the existing distinct-key cutoff. The
single-array constructor shares the same mapping and saturation check.
The change saves a whole-build copy during native filter publication.
The regression compares the bits, capacity and bitmap eligibility with
the single-array path for nullable, sliced keys across 96 chunks.

See Longbow lane `leto-mjj.101`.

## Family OO: open a Parquet file once per generation and scan

Purpose: stop repeating per-file setup that depends only on the file
generation or the scan (lane leto-mjj.96, task 5.29d). Dependent Longbow
feature: SCAN-OPEN-ONCE. Audit classification: UPSTREAMABLE. Upstream
references: apache/datafusion#27488 (open) writes a deferred page-index load
back to `CachedParquetFileReaderFactory`'s cache by downcasting the factory,
which a wrapping factory hides; #24288 (open) asks for masked loading and
caching. Arrow 60.1 already decodes masked page indexes (`ColumnChunkMask`
in `ArrowReaderOptions`).

1. **Page index once per file generation.** The opener asks for the page
   index entries it needs through `AsyncFileReader::get_metadata`, with the
   column-index mask (page-pruning predicate leaves) and the offset-index
   mask (read leaves, predicate leaves and leaf 0 for page counts), each in
   the row groups that statistics kept (`PageIndexRequest`). A reader of a
   shared metadata cache (`DFParquetMetadata::fetch_metadata`) returns cached
   metadata that covers the request; otherwise it loads the complete page
   index of the generation once and writes it back, so later opens and
   queries parse nothing. A reader without a cache keeps the footer it
   returned and loads only the requested entries onto it, through itself;
   the footer is cloned only when it is shared. A reader that ignores the
   request is followed by a masked load from the reader. A changed file has
   another `ObjectMeta`, so `is_valid_for` rejects its cached footer and the
   new generation is parsed again. Prepared scans (family PQ) request the
   offset indexes of their read columns the same way. Page-index bytes that
   the cached path reads go to the store directly and are not in
   `bytes_scanned`.
2. **Filters once per scan and physical schema.** `ParquetSource` holds a
   `PreparedFilterCache` that its clones and every partition's morselizer
   share. An entry holds the adapted predicate and projection, the
   row-group pruning predicate with its lazily built inversion for full-match
   detection, and the page pruning predicate. Its key is every input of the
   preparation: the per-file predicate and projection expressions (pointer
   equality, so a per-file literal rewrite is a new key), the logical,
   rewrite and pruning schemas (equal values), virtual columns, the adapter
   factory, `max_in_list_size` and `enable_page_index`. A predicate with a
   dynamic filter is also keyed by its snapshot generation, and an entry is
   kept only when the generation did not change during its build, so a hit
   equals what a file opened at that generation would prepare. Pruning
   predicate creation errors of the build are added to every file's counter
   that uses the entry. At most 16 entries; the debug output prints none.
   The file-level pruning predicate (file statistics against the logical
   schema, before the footer is read) of a predicate whose dynamic filters
   can no longer change comes from the same cache, keyed by the predicate,
   the logical schema and the generation, and is evaluated once per open as
   `FilePruner::should_prune` would; a predicate that can still change keeps
   its per-file `FilePruner`, which early stopping watches.
3. **One `ParquetFileMetrics` per open.** The opener passes its metrics to
   the reader factory as a `PartitionedFile` extension (`OpenFileMetrics`);
   the default and cached factories give them to the open's readers (the
   first reader, the replacement after Bloom-filter I/O and a prepared
   scan's extra readers) instead of registering a new set. Each reader adds
   its own bytes to the scan efficiency ratio.
