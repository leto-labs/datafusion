# sqlparser 0.62.0 precedence backport

This directory contains the published Apache sqlparser 0.62.0 crate's sources,
manifest, original manifest, README and Apache-2.0 license. The archive is
<https://static.crates.io/crates/sqlparser/sqlparser-0.62.0.crate>, SHA-256
`13c6d1b651dc4edf07eead2a0c6c78016ce971bc2c10da5266861b13f25e7cec`.

The only source change, apart from the Longbow modification notice, changes the
two `IS [NOT] DISTINCT FROM` right-operand calls in `src/parser/mod.rs` from
`parse_expr()` to `parse_subexpr(precedence)`. This is the fix tracked by
[Apache sqlparser #2436](https://github.com/apache/datafusion-sqlparser-rs/pull/2436)
and listed in the upstream 0.63.0 changelog. It stops an unparenthesized right
operand before AND/OR and a following IS, while retaining tighter arithmetic and
explicitly nested boolean operands.

The DataFusion workspace dependency and the Longbow consumer's root path patch
both select this copy. A path patch is necessary while DataFusion remains pinned
to its 0.62 API: upgrading to 0.63 is a separate dependency migration, and
repairing the AST after parsing would duplicate parser precedence rules in the
planner. Wrapping every SQL dialect only to intercept two parser branches would
also require forwarding the dialect's many other capabilities. There is no SQL
text rewrite or runtime opt-out.

Native DataFusion parser and SQL planner regressions live in
`datafusion/sql/src/parser.rs` and `datafusion/sql/tests/sql_integration.rs`.
`datafusion/sqllogictest/test_files/null_safe_precedence.slt` also covers booleans
whose incorrect grouping type-checks but returns the wrong answer. The Longbow
consumer runs the same fixture in all five modes.

Family AI; dependent feature CORE-SQL-NULL-SEMANTICS; classification UPSTREAMABLE.
Drop this copy and both workspace/consumer path overrides after migrating to
a parser release containing upstream PR #2436 (listed in the 0.63.0 changelog).
Keep the AST, planner and result regressions when removing the backport.
