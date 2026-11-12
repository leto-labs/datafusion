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

Only unordered, unlimited, nonvolatile built-in SUM, COUNT, MIN, MAX and AVG
without DISTINCT or ordered arguments are admitted. Group-only aggregates retain
the native distinct/limit path. Native partial-state merges
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
modern merge that exhausts memory releases its table and retries the native
spilling kernel over retained states, never raw input.
The consumer adds independent DuckDB-authored SLT and TPC-DS answer checks.
