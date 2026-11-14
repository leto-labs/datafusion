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

## Family PC: physical projection constraint lineage

Purpose: preserve physical primary/unique keys only when every source key
column survives as a direct output column, with the correct output positions.
Dependent Longbow feature: CORE-CONSTRAINT-CORRECTNESS. Audit classification:
UPSTREAMABLE.

Projection targets are grouped by source expression, and can include struct
field accesses. Build an output-indexed source-column vector with holes for
computed outputs; do not infer a key on a containing struct. Native regressions
cover retained composite keys, false payload uniqueness, permutations, drops,
duplicates, computed expressions and field-access mappings.

## Family JP: join keys before predicate decorrelation

Purpose: construct a connected inner FROM graph from its WHERE equalities
before `IN`/`EXISTS` decorrelation inserts a semi, anti or mark join above it.
Dependent Longbow feature: SQL-INNER-JOIN-CONSTRUCTION. Audit classification:
UPSTREAMABLE.

Later projection pruning can isolate a disconnected date cross join from its
connector even though the equality predicates remain in the plan. Predicate
decorrelation now invokes the existing cross-join rule while the complete
predicate is still available. Its NULL-equality and boundary guards remain in
force. The global optimizer schedule is unchanged. Native tests cover the
default optimizer pipeline and a redundant subquery removed with a common OR
key. The consumer tests physical shape and nullable duplicate answers for IN
and EXISTS on five- and eighteen-table joins.

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

IB1 extends family N with allocation-identity accounting across all retained
rank heaps and peer entries. Shared Arrow buffers (including slice offsets,
validity and child buffers) are charged once per operator. Temporary accounting
maps, cloned batch metadata and container growth remain reserved. Flat output
uses a separate logical-value copy bound, so repeated references to one large
value cannot under-reserve materialized output. Regressions cover shared sliced
batches, dropping replaced buffers, repeated heap inputs, output admission and
cancellation with live retained credits. Purpose: remove false Q67 admission
failures without raising or bypassing the memory pool. Classification: KEEP;
dependent feature: WINDOW-TOPK.

Byte-view descriptor lists are shared allocations as well. Their Arc-backed
storage is deduplicated independently from view/validity buffers. Temporary
ArrayData descriptors are charged only as workspace, using the largest
simultaneous copy instead of adding copies to every retained slice. Flat view
arrays are visited directly; shared descriptor lists are scanned once per
accounting pass. Buffer ownership bookkeeping remains conservatively charged.
A many-buffer StringView regression covers shared descriptor lists. Nested
payload metadata also includes child reference vectors and Arc headers, with
a null-child struct regression covering the lower bound. Sparse-union
metadata includes unoccupied type-id-indexed child slots; a type-id-127
regression checks their retained-size lower bound.



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
shared work source. Duplicate or decreasing row-group ordinals are rejected
before positional payload access, keeping ordinal-based key identities aligned
with the metadata-list offsets used by payload selections.

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

## Family SO: ordering correctness and merge progress

Purpose: protect file-order promises after LIMIT rewrites, remove invalid
NULL-padded join facts, and prevent a merge's selective backpressure from
waiting on sibling scan publication. Dependent Longbow feature:
CORE-ORDERING-CORRECTNESS. Audit classification: UPSTREAMABLE.

Multi-input SortPreservingMergeExec marks its demand-driven input pipeline as
requiring independent partitions. FileScanConfig suppresses waiting for future sibling publication only
while this requirement reaches the scan. Planned files and already published
units remain shareable. An actual SortExec consumes or closes
its input before emitting, and clears the requirement before executing that
input. Scalar aggregates, unordered final/single aggregates without soft
limits, and hash/cross-join build inputs also clear it because they finish
consuming before output. Partial aggregation can flush early and retains the
requirement. A pass-through sort retains it. Thus ordinary sorted aggregation/TopK
plans retain scan donation; a direct merge over streaming scans cannot await a
buffered sibling's publication. The session work-sharing setting is unchanged.
Task-context scoping retains runtime, function registries, identities and session
extensions, including the attempt task scope. All scan partitions claim from the
execution's initialized shared queue, even under differing task contexts. A
per-context fallback to local file groups would duplicate queue-owned files.
The scoped independence requirement changes only idle publication waits; it
does not detach a partition from the queue. Native counters distinguish
sharing enabled from publication waits suppressed by the consumer.

A scan's declared output ordering always protects file order and Parquet limit
pruning. This is the relevant backport of Apache #26434 (63af08da). Join classes
clear constants and computed members on NULL-padded sides, while keeping plain
column equalities (Apache #26747, 99e723dd). Offsetting a preserved side retains
its constant metadata, matching Apache main b5bcea88. Join ordering propagation
also clears padded-side
orderings. SEMI/ANTI facts remain on their emitted side. CrossJoin uses Inner
semantics with no join keys, retaining both sides' constants without inventing
join equalities or claiming either input's ordering.

Native regressions cover all join types, known/unknown/partition-local/NULL
constants, ordering facts and cleared scan flags. core/tests/parquet/stock_ordering.rs
compares complete SQL results with a Rust sort over generated Parquet files;
the Longbow consumer includes the same file in its stock_ordering integration
test so its pinned Arrow/DataFusion dependency graph runs these regressions.
The SO1 consumer report records the before/after evidence, upstream status,
expanded differential harness and validation gates.

The review follow-up removes lexicographic build-order suffixes from join
properties. Repeated probe sort keys can restart build-key order for each probe
row; only the maintained input order is established. The obsolete probe-side
argument of join_equivalence_properties is removed from all callers.
Grouping-set expansion starts with fresh equivalence/ordering/constraint facts
and unknown partitioning, following Apache main's grouping-set properties.
It can NULL-fill keys or duplicate sets, so ordinary projection rules do not
apply. Deferred sort-merge filters use a schema that permits NULLs on the
buffered side, also matching Apache main. Portable Parquet regressions cover
these cases and mixed scan contexts; their query guard is 120 seconds.

## Family AC: string-view group identity after clear

Extend AC's primitive reset fix to the string-view interner. Clearing the map
also resets its group counter, so its next group's identity is zero and its
length and emitted arrays agree. A native NULL/reuse regression fails before
this correction. Feature: CORE-AGGREGATION. Classification: UPSTREAMABLE.

## Family VM: byte-view collision layout validation

Purpose: check length before interpreting a same-prefix colliding view as an
external buffer reference. A forced-hash-collision fixture reproduces the old
panic between an inline key and a long key. Complete byte equality, payload
callbacks and insertion order stay unchanged. Feature: CORE-AGGREGATION.
Classification: UPSTREAMABLE.

## Family GD: per-group primitive distinct sets

Purpose: hash distinct values in per-group tables without repeating the group
index in every entry. Singleton groups keep their one value inline. State
emission and prefix removal retain NULL/filter semantics and group identities.
Dependent feature: GROUPED-DISTINCT-COUNT. Classification: UPSTREAMABLE.
Allocation deltas keep size queries constant-time; only emitted sets are visited
when releasing their tracked bytes.

The Many state also retains its last complete value. Exact equality skips a
repeated hash insertion. A different value uses the native hash set. Emission
moves or destroys both together. NULL and FILTER handling remain in the
existing accumulation loop. Inline state accounting includes the extra value.

## Family IC: bounded inline grouping cache

Purpose: reuse complete short-string grouping keys and their assigned group IDs
for one- and two-column groups. Misses use the native interner, long strings keep
its ordinary path, and cardinality above 16,384 releases and disables the cache.
Prefix emission and clear invalidate all cached IDs. Dependent feature:
CORE-AGGREGATION. Classification: UPSTREAMABLE.

Cache hits fuse the external-view length guard with full-key lookup. Arrays
without data buffers use the proven inline-only loop. A miss delegates the
whole batch, then checks all views before population. External buffer
identities never enter the cache. This removes a separate array scan when
Parquet retains an unused dictionary buffer. Native regressions cover those
buffers, masked long values and matching external view bits across arrays.

## Family CR: inline owned ROW_NUMBER ranking bytes

Purpose: retain short encoded ranking keys and payloads inline, without one
allocation and converter Arc clone per value. Large encodings use owned bytes;
payload decoding uses the retaining converter's safe parser. Small heaps start
with at most two slots, covered by retained and transient pool reservations.
Dependent feature: WINDOW-TOPK. Classification: UPSTREAMABLE.

## Family Y: bounded raw grouped integer AVG

Extend exact integer AVG with a conservative cumulative raw-row bound. Up to
i64::MAX input rows of at most 64 bits have sums that fit i128, including every
intermediate group prefix and UInt64::MAX. This removes per-row checked branches
only under that proof. Any imported partial state disables the proof; count and
sum overflow checks remain for merges and subsequent raw updates. Emission does
not weaken the cumulative bound. State format and evaluation stay unchanged.
Feature: CORE-NUMERIC-RESULTS. Classification: UPSTREAMABLE.

## Family NM: nullable membership decorrelation

Purpose: preserve SQL's three-valued IN/NOT IN semantics under expressions and
in projections. Dependent feature: CORE-SUBQUERIES. Classification: UPSTREAMABLE.

One ordinary mark tests equality, another tests an unknown equality among rows
that pass the subquery's correlation. CASE yields TRUE, NULL or FALSE in that
priority; NOT IN negates that nullable result. Empty sets produce FALSE for IN,
even for a NULL probe. Correlated NOT IN uses this path rather than applying a
null-aware anti join to the combined correlation and membership filter. EXISTS
and NOT EXISTS retain their two-valued marks. Projection names are preserved.

The two scans require repeatable membership input. Volatile comparisons or
subqueries, scalar subqueries in the comparison, LIMIT, windows, DISTINCT ON,
aggregate functions and opaque extensions are explicitly rejected on this
path. Supporting those inputs requires a single-evaluation nullable mark join.
Native structural tests and the consumer's DuckDB v2.0.0-derived SLT fixture
cover NULLs, duplicates, empty sets, correlation, OR/AND and projected values.

## Family AC: reset column groups after full emission

Purpose: clear column interner hash buckets and their candidate-index lists
when EmitTo::All replaces the key columns. Reuse otherwise follows stale group
indices into empty columns and can panic. The vector operation buffers clear
with those structures. Native regressions cover both streaming modes, prefix
emission, full emission, NULL keys, complete key equality and reuse.
Dependent feature: CORE-AGGREGATION. Classification: UPSTREAMABLE.

## Family HC: withdrawn grouping probe-cache experiment

The batch-local complete-hash candidate-view cache was removed after its q19
confirmation screen was flat. The initial small gain did not reproduce.
Full key equality stayed authoritative throughout the experiment. Its commits
and evidence remain available for review. No HC cache exists in this pin.
Dependent feature: CORE-AGGREGATION. Classification: UPSTREAMABLE.

## Family RBC: owned rejection bound for ROW_NUMBER Top-K

Purpose: reject a full group's worse rows without loading the heap root's
encoded order. Each group owns a 15-byte prefix and a length marker. Short
orders compare exactly. Equal truncated prefixes use the complete order.
Accepted mutations refresh the prefix after the heap guard drops. The group
entry's enlarged layout enters retained, scratch and emission accounting.
Native regressions compare prefix decisions with full encoded orders and
exercise long shared strings, NULLs, both directions, several k values,
multiple batches and pool release. Dependent feature: WINDOW-TOPK.
Classification: UPSTREAMABLE.
