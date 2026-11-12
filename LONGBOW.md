# Longbow 55.2 patch branch

Base: Apache DataFusion tag 55.2.0, commit
`500744d5c25499055e3b2ca46b022bc425989203`.

This public fork contains the retained Longbow engine patches. Family commits
record purpose, dependent feature and audit classification. Interleaved hunks
use combined-family commits; the complete stack is qualified together.
The final notices commit preserves file-level Apache-2.0 section 4(b) markings.
References to `third_party/README.md` in those markings identify the inventory
in the consuming Longbow repository. This file supplies branch-local context.

All original Apache licenses and notices remain in place. Restore the exact
modification header after regenerating `datafusion/proto-models/src/generated/`:

```text
// Modified by the Longbow authors; see third_party/README.md.

```

Arrow borrowed-range and Parquet selector/read-ahead patches remain in the
consumer's vendor tree. Longbow runs native regressions through a generated
standalone harness with its pinned graph and Arrow patches. A standalone
upstream-workspace build does not provide those Arrow extensions.

## Family UT: checked UTC time buckets

Purpose: truncate explicit UTC hourly/daily timestamp arrays with the existing
checked arithmetic helper instead of converting each row through a timezone
calendar. Construct calendar-path range errors lazily. Dependent Longbow
feature: UTC-TIME-BUCKETS. Audit classification: UPSTREAMABLE.

Units, UTC metadata, NULLs, negative-epoch flooring and range errors remain
unchanged. Non-UTC timezones retain their calendar/DST handling. Differential
native tests compare the array path with the retained scalar calendar path
for every timestamp unit, random dates, leap dates and range boundaries.
The consumer's TS1 report records the sampled CPU costs and paired screens.

## Family AH: null-safe scalar aggregates

Purpose: allow direct inner-key `IS NOT DISTINCT FROM` correlations across
aggregates. Dependent Longbow feature: CORE-SUBQUERIES. Audit classification:
UPSTREAMABLE. The operator remains null-safe and the existing independent
`__always_true` marker still distinguishes absent groups from matched NULL-key
groups, preserving COUNT's zero result on empty input.

Native optimizer regressions cover COUNT(*) and COUNT(column), both operand
orders and the presence-marker compensation. The native SLT fixture
`subquery_null_safe_count.slt` covers real NULL keys, absent groups, nullable
arguments, duplicate outer rows, mixed predicates, GROUP BY and HAVING. The
Longbow consumer runs the same fixture in all five modes against DuckDB rows.
The new aggregate path requires a caller that enables the existing empty-input
compensation; predicate-subquery callers keep their prior unsupported behavior.
Inner casts remain unsupported for this new null-safe path: a non-injective cast
can map distinct inner groups to one outer key. Both operand orders are guarded
by a native regression; general expression decorrelation needs separate work.

## Family AI: null-safe comparison precedence

Purpose: backport sqlparser's `IS [NOT] DISTINCT FROM` right-operand precedence
fix without changing the 0.62 parser API. Dependent Longbow feature:
CORE-SQL-NULL-SEMANTICS. Audit classification: UPSTREAMABLE. The local path
copy's `third_party/sqlparser/LONGBOW.md` records the published archive checksum,
exact two-call source delta and upstream tracking (#2436).

The workspace and consumer both select that copy; it is excluded from workspace
membership so its optional development dependencies do not enter DataFusion's
locked graph. Native AST and SQL planner regressions cover both comparisons and
AND/OR. The native `null_safe_precedence.slt` fixture also checks type-valid wrong
answers, explicit boolean operands, arithmetic and chained IS predicates.

## Family N: bounded window Top-K

Family N / WINDOW-TOPK / KEEP: bounded ROW_NUMBER and global rank execution,
compact encoded key-only heaps, bounded incremental output, and removal of the
superseded experimental policy/configuration. The consumer owns conservative
admission. Native heap ordering regressions and the consumer's semantic,
codec, pool-refusal and drop tests qualify this family together.

## Family BSC: bitmap short-circuit evaluation

Purpose: avoid gathering whole batches for direct primitive comparisons when
the selected column volume justifies a contiguous comparison. Expand retained
short-circuit results with packed value and validity buffers, without temporary
boolean slices. Dependent Longbow feature: STATIC-FILTER-EVALUATION. Audit
classification: UPSTREAMABLE.

The type guard admits only comparisons of matching primitive columns/literals.
Casts, arithmetic, functions, and unknown expressions keep their existing
evaluation path and error suppression. No predicate reordering occurs. Native
differential regressions cover sliced bitmaps, NULLs, Kleene logic, floating
edge values, decimals, dates, timestamps with time zones, unsigned integers,
Durations, and fallible casts/arithmetic on rejected rows. Direct guard tests
reject indirect operands and dictionary columns on narrow, wide, and sliced
batches. Scatter returns an internal error if the RHS length differs from the
number of selected rows, including in release builds.

## Family SVC: short string-view constants

Purpose: compare inline string-view constants as exact view words, and use
fixed comparison chains for short static string IN lists. Dependent Longbow
feature: STATIC-FILTER-EVALUATION. Audit classification: UPSTREAMABLE.

The scalar path handles equality and inequality with non-null Utf8View
constants of at most twelve bytes. The IN strategy handles up to two non-null
inline Utf8View constants, including dictionary inputs, with the existing
NULL/NOT IN result builder. Utf8/LargeUtf8 inputs, longer constants/lists,
and other types keep the general
strategies. Native tests compare with Arrow and the generic static filter;
the ignored short-string kernel screen runs two ABBA pairs in one process.

The scalar view-word comparison assumes every inline view has zero padding,
including arrays that also contain external data buffers. Arrow validates this
padding in byte-view array validation. Longbow has no unchecked view
construction, so those validated inline words can be compared directly even in
mixed inline/external arrays. Unlike stock's no-data-buffer fast path, this
optimization relies on that invariant for every input array. Seeded IN/NOT IN
properties cover lists both with and without NULL, including empty lists.

## Family IL: compact dynamic IN-list literal trees

Purpose: retain one literal per distinct array value when constructing an
`InListExpr`. Hash-join pushdown chooses this representation using distinct-key
count, so repeated build rows must not create a literal tree proportional to
build cardinality. Parquet file rewrites otherwise repeat that oversized walk
for every file. Dependent Longbow feature: DYNAMIC-INLIST-SCALING. Audit
classification: UPSTREAMABLE.

Evaluation retains the existing static filter and its source array. The literal
children preserve first-occurrence order and one NULL. Native regressions bound
tree size for 100,000 duplicate keys, check IN/NOT IN and column remapping, and
check duplicate struct literals with inner NULLs. The consumer adds a correlated
aggregate answer fixture with NULL store keys, NULL fees and duplicate rows.
No configuration, planner heuristic, wire format or result identity changes.

## Family CG: chained grouping sets

Purpose: compute maximal requested grouping sets once and derive each coarser
set from the smallest containing computed set. Dependent Longbow feature:
CORE-AGGREGATION. Audit classification: UPSTREAMABLE.

Only unordered, unlimited, nonvolatile built-in SUM, COUNT, nonfloating MIN/MAX and AVG
without DISTINCT or ordered arguments are admitted. Group-only aggregates retain
the native distinct/limit path. Floating MIN/MAX retain expansion because native
NaN handling depends on input order. Floating SUM/AVG use native states and
retain their usual floating-point reassociation, not bitwise sequential equivalence.
Finite pools without temporary files retain native expansion. Partial-state merges
retain AVG sum/count, grouping-ID width, duplicate ordinals and empty totals.
Incomparable maximal sets share one raw-input aggregate. Each execution and
partition owns its retained memory reservations and native spill files; no new
serialized node or cross-attempt state is introduced.

Native differential properties use the expanding stream as their oracle and
cover NULL keys/arguments, filtered aggregates, duplicate and empty sets,
multiple partitions, finite pools, stream drop, cancellation during materialization
and an empty one-shot exchange. Flushed fine states remain mergeable; the ordinary
final aggregate completes that level. Coarser cached levels use hash grouping,
without treating literal NULLs/IDs as useful ordering. Under a finite pool, a
state merge that exhausts memory releases its table and retries a native
PartialReduce kernel that emits incomplete states early. This avoids a sort
reservation while its parent is retained. Later merges and the ordinary final
aggregate combine duplicate states. The normal successful merge path is unchanged;
raw input is never retried. Retained spilling uses the attempt task scope.
The consumer adds independent DuckDB-authored SLT and TPC-DS answer checks.

## Family WS: partition-owned files and late row-group sharing

Purpose: replace the global file-sized work queue and open-time sparse-scan
split gate with planned-file ownership and cooperative tail donation. Dependent
Longbow feature: PARTITIONED-SCAN-BALANCE. Audit classification: UPSTREAMABLE.

Each publishing stream drains its own planned files before taking published
whole-row-group runs. Idle streams wait while a sibling reader is active. If no
reader is active, an idle stream takes an unopened planned file: sequential
partition consumers must not wait for an owner that has never been polled. Eligible Parquet readers reconsider donation at native row-group boundaries
after pruning, including when demand appears after file open. Published access
plans preserve page selections, fully-matched flags, ranges and file identity,
and can donate again. The reader retains its decoder buffers and a protected
read-ahead prefix; the factory reports the advisory row-group window.

This replaces family S's initial-pruning gate and file queue, while preserving
its scan eligibility guards. Ordered, limited, reversed and statistics-reordered
scans do not publish. Non-publishing scans retain the global, statistics-ordered
file queue; preserved-order and declared-partitioning scans remain local.
Watching dynamic filters defer donation until completion.
An external file-level ParquetRowSelection remains local; page-index selections
are carried by the published access plans. Native regressions cover planned
ownership, late demand, unopened owners, cancellation, concurrent exact-once
claims, page selections, and physical reads with silently empty filtered groups.
The consumer's read-ahead tests cover retained requests and released credits.

Review regressions include 1,500 random file/range/predicate shapes across three
schedules, 300 cancellation and 300 injected-error cases, sequential partition
consumption, filter completion during a scan, receiver re-donation, and explicit
reverse/statistics-order eligibility guards. Idle fallback and cancellation
wakeups are decided while holding the shared queue lock.

## Family GSS: grouping-set spill replay keys

Purpose: retain the materialized grouping ID when the stock hash aggregate
merges spilled states. Dependent Longbow feature: CORE-AGGREGATION.
Audit classification: UPSTREAMABLE.

Use the existing `PhysicalGroupBy::as_final()` conversion for replay keys.
The grouping ID separates real NULLs, rolled-up NULLs and duplicate sets.
Dropping it caused Single-mode replay to emit one column fewer than its
output schema. Native regressions bypass family CG and cover GROUPING SETS
and ROLLUP with NULLs, duplicate sets, filtered AVG states, Partial early
emission and Final disk spills. Capped results must equal uncapped results,
multiple spill runs must merge, and pool reservations must return to zero.

## Family RP: positional Parquet Top-N retrieval

Purpose: resolve sorted file-row positions into per-row-group RowSelections
after the opener obtains metadata, preserving disjoint byte-range ownership.
Dependent Longbow feature: TOPN-POSITIONAL. Audit classification: UPSTREAMABLE.
Conflicting access extensions, duplicate/unsorted positions and out-of-file
positions fail explicitly; positional streams do not publish row groups to the
shared work source.

The generic file-scan codec carries virtual fields as an explicitly counted
schema suffix after file and partition fields. It never infers a virtual role
from Arrow extension metadata on a physical column. The generated protobuf and
JSON codecs retain that count. Native regressions cover role reconstruction,
invalid counts, group boundaries, empty selections and extension conflicts.
The consumer owns the general rewrite, bounded native TopK and ordered payload
retrieval, and changes its stage codec identity for these new plans.

Independent key and payload branches use distinct scan counters even before
a codec round trip. `ParquetSource::with_new_metrics` preserves all source
options and reader factories while giving an independent clone fresh counters.
Ordinary optimizer transformations continue to share their original counters.
