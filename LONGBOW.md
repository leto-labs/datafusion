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
