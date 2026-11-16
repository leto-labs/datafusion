// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Simplifier for Physical Expressions

use arrow::datatypes::Schema;
use datafusion_common::{Result, tree_node::TreeNode};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use crate::{
    PhysicalExpr,
    expressions::{DynamicFilterTracker, DynamicFilterTracking},
    simplifier::{
        const_evaluator::create_dummy_batch, unwrap_cast::unwrap_cast_in_comparison,
    },
};

pub mod const_evaluator;
pub mod not;
pub mod unwrap_cast;

const MAX_LOOP_COUNT: usize = 5;
const PREPARATION_CACHE_CAPACITY: usize = 256;
const PREPARATION_CACHE_PRUNE_INTERVAL: u64 = 64;
// The simplifier's constant evaluator uses a fixed one-row dummy batch. Keep
// the context in the key so a future evaluator context cannot reuse entries
// prepared for this one.
const DEFAULT_EVALUATION_CONTEXT: u64 = 0;

static PREPARATION_CALLS: AtomicU64 = AtomicU64::new(0);
static PREPARATION_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static PREPARATION_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static PREPARATION_CLONED_TREES: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static CACHE_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Run synchronous scan work with this session's preparation-cache setting.
///
/// A stream must enter this scope for each poll, never across an await. The
/// previous setting is restored on return or panic, so interleaved sessions and
/// tasks moving between worker threads cannot change one another's settings.
pub fn with_preparation_cache<T>(enabled: bool, operation: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            CACHE_ENABLED.set(self.0);
        }
    }
    let _restore = Restore(CACHE_ENABLED.replace(enabled));
    operation()
}

/// Whether synchronous scan work currently permits preparation-cache reuse.
pub fn preparation_cache_enabled() -> bool {
    CACHE_ENABLED.get()
}

fn preparation_counter_path() -> Option<&'static Path> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| std::env::var_os("LONGBOW_EXPR_PREP_COUNTERS").map(PathBuf::from))
        .as_deref()
}

fn record_preparation_counters() {
    let Some(path) = preparation_counter_path() else {
        return;
    };
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let line = format!(
        "{}\t{}\t{}\t{}\t{}\n",
        std::process::id(),
        PREPARATION_CALLS.load(Ordering::Relaxed),
        PREPARATION_CACHE_HITS.load(Ordering::Relaxed),
        PREPARATION_CACHE_MISSES.load(Ordering::Relaxed),
        PREPARATION_CLONED_TREES.load(Ordering::Relaxed),
    );
    let _ = file.write_all(line.as_bytes());
}

/// A cache entry for one schema-bound expression preparation.
///
/// Weak references keep the cache from retaining expression trees, large
/// literal payloads, or dynamic-filter state after the owning scan/query is
/// dropped. Dead entries are pruned periodically and before a full insertion.
struct PreparationCacheEntry {
    input: Weak<dyn PhysicalExpr>,
    output: Weak<dyn PhysicalExpr>,
    schema: Schema,
    schema_fingerprint: u64,
    evaluation_context: u64,
}

#[derive(Default)]
struct PreparationCache {
    entries: VecDeque<PreparationCacheEntry>,
    lookups: u64,
    hits: u64,
    misses: u64,
    lookups_since_prune: u64,
}

impl PreparationCache {
    #[cfg(test)]
    fn clear(&mut self) {
        self.entries.clear();
        self.lookups = 0;
        self.hits = 0;
        self.misses = 0;
        self.lookups_since_prune = 0;
    }

    fn prune_dead(&mut self) {
        self.entries.retain(|entry| {
            entry.input.strong_count() != 0 && entry.output.strong_count() != 0
        });
        self.lookups_since_prune = 0;
    }

    fn lookup(
        &mut self,
        input: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
        schema_fingerprint: u64,
        evaluation_context: u64,
    ) -> Option<Arc<dyn PhysicalExpr>> {
        self.lookups += 1;
        self.lookups_since_prune += 1;
        if self.lookups_since_prune >= PREPARATION_CACHE_PRUNE_INTERVAL {
            self.prune_dead();
        }
        let index = self.entries.iter().position(|entry| {
            std::ptr::addr_eq(entry.input.as_ptr(), Arc::as_ptr(input))
                && entry.evaluation_context == evaluation_context
                && entry.schema_fingerprint == schema_fingerprint
                && entry.schema == *schema
        });
        let Some(index) = index else {
            self.misses += 1;
            return None;
        };

        let entry = self
            .entries
            .remove(index)
            .expect("cache entry index was found before removal");
        let Some(output) = entry.output.upgrade() else {
            self.misses += 1;
            return None;
        };
        self.entries.push_back(entry);
        self.hits += 1;
        Some(output)
    }

    #[cfg(test)]
    fn insert(
        &mut self,
        input: &Arc<dyn PhysicalExpr>,
        output: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
        schema_fingerprint: u64,
        evaluation_context: u64,
    ) {
        let tracking = DynamicFilterTracking::classify(input);
        self.insert_with_tracking(
            input,
            output,
            schema,
            schema_fingerprint,
            evaluation_context,
            &tracking,
        );
    }

    fn insert_with_tracking(
        &mut self,
        input: &Arc<dyn PhysicalExpr>,
        output: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
        schema_fingerprint: u64,
        evaluation_context: u64,
        tracking: &DynamicFilterTracking,
    ) {
        if self.entries.len() >= PREPARATION_CACHE_CAPACITY {
            self.prune_dead();
        }
        // No dynamic tree is retained, even when complete. Producers support
        // post-completion updates and may publish nested filters in the future.
        if tracking.contains_dynamic_filter() {
            return;
        }
        self.entries.push_back(PreparationCacheEntry {
            input: Arc::downgrade(input),
            output: Arc::downgrade(output),
            schema: schema.clone(),
            schema_fingerprint,
            evaluation_context,
        });
        if self.entries.len() > PREPARATION_CACHE_CAPACITY {
            self.entries.pop_front();
        }
    }
}

thread_local! {
    /// Scan preparation is performed on the executor worker that owns the
    /// opener. Keeping the bounded cache thread-local avoids turning a hot
    /// per-file path into a process-wide mutex while still reusing entries for
    /// the files and row groups handled by that worker.
    static PREPARATION_CACHE: RefCell<PreparationCache> =
        RefCell::new(PreparationCache::default());
}

fn schema_fingerprint(schema: &Schema) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    schema.hash(&mut hasher);
    hasher.finish()
}

/// Simplifies physical expressions by applying various optimizations
///
/// This can be useful after adapting expressions from a table schema
/// to a file schema. For example, casts added to match the types may
/// potentially be unwrapped.
pub struct PhysicalExprSimplifier<'a> {
    schema: &'a Schema,
    schema_fingerprint: Option<u64>,
    evaluation_context: u64,
}

impl<'a> PhysicalExprSimplifier<'a> {
    /// Create a new physical expression simplifier
    pub fn new(schema: &'a Schema) -> Self {
        Self {
            schema,
            schema_fingerprint: preparation_cache_enabled()
                .then(|| schema_fingerprint(schema)),
            evaluation_context: DEFAULT_EVALUATION_CONTEXT,
        }
    }

    /// Simplify a physical expression
    pub fn simplify(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
        let owned_expr = expr;
        let counters_enabled = preparation_counter_path().is_some();
        if counters_enabled {
            PREPARATION_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        let Some(schema_fingerprint) = self.schema_fingerprint else {
            if !counters_enabled {
                return self.simplify_uncached(owned_expr);
            }
            PREPARATION_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            let result = self.simplify_uncached(Arc::clone(&owned_expr))?;
            if !Arc::ptr_eq(&owned_expr, &result) {
                PREPARATION_CLONED_TREES.fetch_add(1, Ordering::Relaxed);
            }
            record_preparation_counters();
            return Ok(result);
        };
        if let Some(cached) = PREPARATION_CACHE.with(|cache| {
            cache.borrow_mut().lookup(
                &owned_expr,
                self.schema,
                schema_fingerprint,
                self.evaluation_context,
            )
        }) {
            if counters_enabled {
                PREPARATION_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            }
            record_preparation_counters();
            return Ok(cached);
        }

        if counters_enabled {
            PREPARATION_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
        }

        // Subscribe before preparation. The tracker is carried through the
        // preparation and insertion so an update observed during the walk is
        // never re-baselined by a second classification.
        let mut tracking = DynamicFilterTracking::classify(&owned_expr);
        let mut result = self.simplify_uncached(Arc::clone(&owned_expr))?;
        let mut changed_during_preparation = tracking
            .watcher()
            .is_some_and(DynamicFilterTracker::changed);
        if changed_during_preparation {
            // A single retry closes the common update-during-fold window while
            // keeping a continuously updating producer from causing an
            // unbounded preparation loop. The second result is never cached if
            // another update is observed.
            result = self.simplify_uncached(Arc::clone(&owned_expr))?;
            changed_during_preparation = tracking
                .watcher()
                .is_some_and(DynamicFilterTracker::changed);
        }
        if counters_enabled && !Arc::ptr_eq(&owned_expr, &result) {
            PREPARATION_CLONED_TREES.fetch_add(1, Ordering::Relaxed);
        }
        // A rewritten root is normally owned only by this call site (the
        // Parquet pruning builders pass fresh trees), so weakly caching it
        // cannot produce a hit and only adds lookup/pruning work. Cache the
        // unchanged roots that remain owned by the scan plan instead.
        if !changed_during_preparation
            && !tracking.contains_dynamic_filter()
            && Arc::ptr_eq(&owned_expr, &result)
        {
            PREPARATION_CACHE.with(|cache| {
                cache.borrow_mut().insert_with_tracking(
                    &owned_expr,
                    &result,
                    self.schema,
                    schema_fingerprint,
                    self.evaluation_context,
                    &tracking,
                );
            });
        }
        record_preparation_counters();
        Ok(result)
    }

    fn simplify_uncached(
        &self,
        expr: Arc<dyn PhysicalExpr>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let mut current_expr = expr;
        let mut count = 0;
        let schema = self.schema;

        let batch = create_dummy_batch()?;

        while count < MAX_LOOP_COUNT {
            count += 1;
            let result = current_expr.transform(|node| {
                #[cfg(debug_assertions)]
                let original_type = node.data_type(schema).unwrap();

                // Apply NOT expression simplification first, then unwrap cast optimization,
                // then constant expression evaluation
                #[expect(deprecated, reason = "`simplify_not_expr` is marked as deprecated until it's made private.")]
                let rewritten = not::simplify_not_expr(node, schema)?
                    .transform_data(|node| unwrap_cast_in_comparison(node, schema))?
                    .transform_data(|node| {
                        const_evaluator::simplify_const_expr_immediate(node, batch)
                    })?;

                #[cfg(debug_assertions)]
                assert_eq!(
                    rewritten.data.data_type(schema).unwrap(),
                    original_type,
                    "Simplified expression should have the same data type as the original"
                );

                Ok(rewritten)
            })?;

            if !result.transformed {
                return Ok(result.data);
            }
            current_expr = result.data;
        }
        Ok(current_expr)
    }
}

#[cfg(test)]
mod preparation_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScalarFunctionExpr;
    use crate::expressions::{
        BinaryExpr, CastExpr, Literal, NotExpr, TryCastExpr, col, in_list, lit,
    };
    use arrow::datatypes::{DataType, Field};
    use datafusion_common::ScalarValue;
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::{
        Operator, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
    };
    use datafusion_expr_common::columnar_value::ColumnarValue;
    use datafusion_functions::regex::regexp_like;

    pub(super) fn test_schema() -> Schema {
        Schema::new(vec![
            Field::new("c1", DataType::Int32, false),
            Field::new("c2", DataType::Int64, false),
            Field::new("c3", DataType::Utf8, false),
        ])
    }

    fn not_test_schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Boolean, false),
            Field::new("b", DataType::Boolean, false),
            Field::new("c", DataType::Int32, false),
        ])
    }

    /// Helper function to extract a Literal from a PhysicalExpr
    pub(super) fn as_literal(expr: &Arc<dyn PhysicalExpr>) -> &Literal {
        expr.downcast_ref::<Literal>()
            .unwrap_or_else(|| panic!("Expected Literal, got: {expr}"))
    }

    /// Helper function to extract a BinaryExpr from a PhysicalExpr
    fn as_binary(expr: &Arc<dyn PhysicalExpr>) -> &BinaryExpr {
        expr.downcast_ref::<BinaryExpr>()
            .unwrap_or_else(|| panic!("Expected BinaryExpr, got: {expr}"))
    }

    /// Assert that simplifying `input` produces `expected`
    fn assert_not_simplify(
        simplifier: &PhysicalExprSimplifier,
        input: Arc<dyn PhysicalExpr>,
        expected: Arc<dyn PhysicalExpr>,
    ) {
        let result = simplifier.simplify(Arc::clone(&input)).unwrap();
        assert_eq!(
            &result, &expected,
            "Simplification should transform:\n  input: {input}\n  to:    {expected}\n  got:   {result}"
        );
    }

    #[test]
    fn test_simplify() {
        let schema = test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // Create: cast(c2 as INT32) != INT32(99)
        // c2 is Int64 → Int64→Int32 is narrowing → blocked by family gate.
        // The cast should stay.
        let column_expr = col("c2", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(column_expr, DataType::Int32, None));
        let literal_expr = lit(ScalarValue::Int32(Some(99)));
        let binary_expr =
            Arc::new(BinaryExpr::new(cast_expr, Operator::NotEq, literal_expr));

        // Apply full simplification (uses TreeNodeRewriter)
        let optimized = simplifier.simplify(binary_expr).unwrap();

        // With the family gate, Int64→Int32 is blocked and the cast remains.
        let result_binary = as_binary(&optimized);
        let result_left = result_binary.left();
        assert!(
            result_left.downcast_ref::<CastExpr>().is_some()
                || result_left.downcast_ref::<TryCastExpr>().is_some(),
            "CAST(c2 AS Int32) should remain as Int64→Int32 is narrowing"
        );
    }

    #[test]
    fn test_nested_expression_simplification() {
        let schema = test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // Create nested expression: (cast(c1 as INT64) > INT64(5)) OR (cast(c2 as INT32) <= INT32(10))
        let c1_expr = col("c1", &schema).unwrap();
        let c1_cast = Arc::new(CastExpr::new(c1_expr, DataType::Int64, None));
        let c1_literal = lit(ScalarValue::Int64(Some(5)));
        let c1_binary = Arc::new(BinaryExpr::new(c1_cast, Operator::Gt, c1_literal));

        let c2_expr = col("c2", &schema).unwrap();
        let c2_cast = Arc::new(CastExpr::new(c2_expr, DataType::Int32, None));
        let c2_literal = lit(ScalarValue::Int32(Some(10)));
        let c2_binary = Arc::new(BinaryExpr::new(c2_cast, Operator::LtEq, c2_literal));

        let or_expr = Arc::new(BinaryExpr::new(c1_binary, Operator::Or, c2_binary));

        // Apply simplification
        let optimized = simplifier.simplify(or_expr).unwrap();

        let or_binary = as_binary(&optimized);

        // Verify left side: c1 > INT32(5)  (Int32→Int64 widening, unwrapped)
        let left_binary = as_binary(or_binary.left());
        let left_left_expr = left_binary.left();
        assert!(
            left_left_expr.downcast_ref::<CastExpr>().is_none()
                && left_left_expr.downcast_ref::<TryCastExpr>().is_none()
        );
        let left_literal = as_literal(left_binary.right());
        assert_eq!(left_literal.value(), &ScalarValue::Int32(Some(5)));

        // Verify right side: CAST(c2 AS Int32) <= 10  — Int64→Int32 is
        // narrowing, blocked by the family gate, so the cast stays.
        let right_binary = as_binary(or_binary.right());
        let right_left_expr = right_binary.left();
        assert!(
            right_left_expr.downcast_ref::<CastExpr>().is_some()
                || right_left_expr.downcast_ref::<TryCastExpr>().is_some(),
            "CAST(c2 AS Int32) should remain as Int64→Int32 is narrowing"
        );
    }

    #[test]
    fn test_double_negation_elimination() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(NOT(c > 5)) -> c > 5
        let inner_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::Gt,
            lit(ScalarValue::Int32(Some(5))),
        ));
        let inner_not = Arc::new(NotExpr::new(Arc::clone(&inner_expr)));
        let double_not: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(inner_not));

        let expected = inner_expr;
        assert_not_simplify(&simplifier, double_not, expected);
        Ok(())
    }

    #[test]
    fn test_not_literal() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(TRUE) -> FALSE
        let not_true = Arc::new(NotExpr::new(lit(ScalarValue::Boolean(Some(true)))));
        let expected = lit(ScalarValue::Boolean(Some(false)));
        assert_not_simplify(&simplifier, not_true, expected);

        // NOT(FALSE) -> TRUE
        let not_false = Arc::new(NotExpr::new(lit(ScalarValue::Boolean(Some(false)))));
        let expected = lit(ScalarValue::Boolean(Some(true)));
        assert_not_simplify(&simplifier, not_false, expected);

        Ok(())
    }

    #[test]
    fn test_negate_comparison() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(c = 5) -> c != 5
        let not_eq = Arc::new(NotExpr::new(Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::Eq,
            lit(ScalarValue::Int32(Some(5))),
        ))));
        let expected = Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::NotEq,
            lit(ScalarValue::Int32(Some(5))),
        ));
        assert_not_simplify(&simplifier, not_eq, expected);

        Ok(())
    }

    #[test]
    fn test_demorgans_law_and() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(a AND b) -> NOT a OR NOT b
        let and_expr = Arc::new(BinaryExpr::new(
            col("a", &schema)?,
            Operator::And,
            col("b", &schema)?,
        ));
        let not_and: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(and_expr));

        let expected: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(NotExpr::new(col("a", &schema)?)),
            Operator::Or,
            Arc::new(NotExpr::new(col("b", &schema)?)),
        ));
        assert_not_simplify(&simplifier, not_and, expected);

        Ok(())
    }

    #[test]
    fn test_demorgans_law_or() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(a OR b) -> NOT a AND NOT b
        let or_expr = Arc::new(BinaryExpr::new(
            col("a", &schema)?,
            Operator::Or,
            col("b", &schema)?,
        ));
        let not_or: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(or_expr));

        let expected: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(NotExpr::new(col("a", &schema)?)),
            Operator::And,
            Arc::new(NotExpr::new(col("b", &schema)?)),
        ));
        assert_not_simplify(&simplifier, not_or, expected);

        Ok(())
    }

    #[test]
    fn test_demorgans_with_comparison_simplification() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(c = 1 AND c = 2) -> c != 1 OR c != 2
        let eq1 = Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::Eq,
            lit(ScalarValue::Int32(Some(1))),
        ));
        let eq2 = Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::Eq,
            lit(ScalarValue::Int32(Some(2))),
        ));
        let and_expr = Arc::new(BinaryExpr::new(eq1, Operator::And, eq2));
        let not_and: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(and_expr));

        let expected: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                col("c", &schema)?,
                Operator::NotEq,
                lit(ScalarValue::Int32(Some(1))),
            )),
            Operator::Or,
            Arc::new(BinaryExpr::new(
                col("c", &schema)?,
                Operator::NotEq,
                lit(ScalarValue::Int32(Some(2))),
            )),
        ));
        assert_not_simplify(&simplifier, not_and, expected);

        Ok(())
    }

    #[test]
    fn test_not_of_not_and_not() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(NOT(a) AND NOT(b)) -> a OR b
        let not_a = Arc::new(NotExpr::new(col("a", &schema)?));
        let not_b = Arc::new(NotExpr::new(col("b", &schema)?));
        let and_expr = Arc::new(BinaryExpr::new(not_a, Operator::And, not_b));
        let not_and: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(and_expr));

        let expected: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("a", &schema)?,
            Operator::Or,
            col("b", &schema)?,
        ));
        assert_not_simplify(&simplifier, not_and, expected);

        Ok(())
    }

    #[test]
    fn test_not_in_list() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(c IN (1, 2, 3)) -> c NOT IN (1, 2, 3)
        let list = vec![
            lit(ScalarValue::Int32(Some(1))),
            lit(ScalarValue::Int32(Some(2))),
            lit(ScalarValue::Int32(Some(3))),
        ];
        let in_list_expr = in_list(col("c", &schema)?, list.clone(), &false, &schema)?;
        let not_in: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(in_list_expr));

        let expected = in_list(col("c", &schema)?, list, &true, &schema)?;
        assert_not_simplify(&simplifier, not_in, expected);

        Ok(())
    }

    #[test]
    fn test_not_not_in_list() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(c NOT IN (1, 2, 3)) -> c IN (1, 2, 3)
        let list = vec![
            lit(ScalarValue::Int32(Some(1))),
            lit(ScalarValue::Int32(Some(2))),
            lit(ScalarValue::Int32(Some(3))),
        ];
        let not_in_list_expr = in_list(col("c", &schema)?, list.clone(), &true, &schema)?;
        let not_not_in: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(not_in_list_expr));

        let expected = in_list(col("c", &schema)?, list, &false, &schema)?;
        assert_not_simplify(&simplifier, not_not_in, expected);

        Ok(())
    }

    #[test]
    fn test_double_not_in_list() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(NOT(c IN (1, 2, 3))) -> c IN (1, 2, 3)
        let list = vec![
            lit(ScalarValue::Int32(Some(1))),
            lit(ScalarValue::Int32(Some(2))),
            lit(ScalarValue::Int32(Some(3))),
        ];
        let in_list_expr = in_list(col("c", &schema)?, list.clone(), &false, &schema)?;
        let not_in = Arc::new(NotExpr::new(in_list_expr));
        let double_not: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(not_in));

        let expected = in_list(col("c", &schema)?, list, &false, &schema)?;
        assert_not_simplify(&simplifier, double_not, expected);

        Ok(())
    }

    #[test]
    fn test_deeply_nested_not() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // Create a deeply nested NOT expression: NOT(NOT(NOT(...NOT(c > 5)...)))
        // This tests that we don't get stack overflow with many nested NOTs.
        // With recursive_protection enabled (default), this should work by
        // automatically growing the stack as needed.
        let inner_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::Gt,
            lit(ScalarValue::Int32(Some(5))),
        ));

        let mut expr = Arc::clone(&inner_expr);
        // Create 200 layers of NOT to test deep recursion handling
        for _ in 0..200 {
            expr = Arc::new(NotExpr::new(expr));
        }

        // With 200 NOTs (even number), should simplify back to the original expression
        let expected = inner_expr;
        assert_not_simplify(&simplifier, Arc::clone(&expr), expected);

        // Manually dismantle the deep input expression to avoid Stack Overflow on Drop
        // If we just let `expr` go out of scope, Rust's recursive Drop will blow the stack
        // even with recursive_protection, because Drop doesn't use the #[recursive] attribute.
        // We peel off layers one by one to avoid deep recursion in Drop.
        while let Some(not_expr) = expr.downcast_ref::<NotExpr>() {
            // Clone the child (Arc increment).
            // Now child has 2 refs: one in parent, one in `child`.
            let child = Arc::clone(not_expr.arg());

            // Reassign `expr` to `child`.
            // This drops the old `expr` (Parent).
            // Parent refcount -> 0, Parent is dropped.
            // Parent drops its reference to Child.
            // Child refcount decrements 2 -> 1.
            // Child is NOT dropped recursively because we still hold it in `expr`
            expr = child;
        }

        Ok(())
    }

    #[test]
    fn test_simplify_literal_binary_expr() {
        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // 1 + 2 -> 3
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(lit(1i32), Operator::Plus, lit(2i32)));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(literal.value(), &ScalarValue::Int32(Some(3)));
    }

    #[test]
    fn test_simplify_literal_comparison() {
        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // 5 > 3 -> true
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(lit(5i32), Operator::Gt, lit(3i32)));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(literal.value(), &ScalarValue::Boolean(Some(true)));

        // 2 > 3 -> false
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(lit(2i32), Operator::Gt, lit(3i32)));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(literal.value(), &ScalarValue::Boolean(Some(false)));
    }

    #[test]
    fn test_constant_regex_respects_planning_budget() {
        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        let ordinary: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            lit("aaaaa"),
            Operator::RegexMatch,
            lit("^a+$"),
        ));
        let result = simplifier.simplify(ordinary).unwrap();
        assert_eq!(
            as_literal(&result).value(),
            &ScalarValue::Boolean(Some(true))
        );

        // This pattern is valid under the runtime limit, but is deliberately
        // too large for constant evaluation during physical simplification.
        let pattern = "a{5}{5}{5}{5}{5}{5}";
        let binary: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            lit("aaaaa"),
            Operator::RegexMatch,
            lit(pattern),
        ));
        let result = simplifier.simplify(Arc::clone(&binary)).unwrap();
        assert_eq!(&result, &binary);

        // A regexp_like call with non-operator flags exercises the physical
        // scalar-UDF path used after literal substitution.
        let function = Arc::new(
            ScalarFunctionExpr::try_new(
                regexp_like(),
                vec![lit("aaaaa"), lit(pattern), lit("m")],
                &schema,
                Arc::new(ConfigOptions::default()),
            )
            .unwrap(),
        ) as Arc<dyn PhysicalExpr>;
        let result = simplifier.simplify(Arc::clone(&function)).unwrap();
        assert_eq!(&result, &function);

        #[derive(Debug, PartialEq, Eq, Hash)]
        struct PanickingConstHook {
            signature: Signature,
        }

        impl ScalarUDFImpl for PanickingConstHook {
            fn name(&self) -> &str {
                "panicking_const_hook"
            }

            fn signature(&self) -> &Signature {
                &self.signature
            }

            fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
                Ok(DataType::Boolean)
            }

            fn invoke_with_args(
                &self,
                _args: ScalarFunctionArgs,
            ) -> Result<ColumnarValue> {
                panic!("non-constant UDF must not be evaluated during planning")
            }

            fn should_evaluate_const(&self, _args: &[&ScalarValue]) -> bool {
                panic!("constant-evaluation hook must only inspect literal arguments")
            }
        }

        // Non-constant expressions must skip the UDF hook entirely. In
        // particular, regexp_like(column, pattern) must not compile its
        // pattern just to discover that the expression cannot be folded.
        let input_schema = test_schema();
        let non_constant = Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(ScalarUDF::new_from_impl(PanickingConstHook {
                    signature: Signature::exact(
                        vec![DataType::Utf8, DataType::Utf8],
                        Volatility::Immutable,
                    ),
                })),
                vec![col("c3", &input_schema).unwrap(), lit(pattern)],
                &input_schema,
                Arc::new(ConfigOptions::default()),
            )
            .unwrap(),
        ) as Arc<dyn PhysicalExpr>;
        let result = PhysicalExprSimplifier::new(&input_schema)
            .simplify(Arc::clone(&non_constant))
            .unwrap();
        assert_eq!(&result, &non_constant);
    }

    #[test]
    fn test_simplify_nested_literal_expr() {
        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // (1 + 2) * 3 -> 9
        let inner: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(lit(1i32), Operator::Plus, lit(2i32)));
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(inner, Operator::Multiply, lit(3i32)));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(literal.value(), &ScalarValue::Int32(Some(9)));
    }

    #[test]
    fn test_simplify_deeply_nested_literals() {
        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // ((1 + 2) * 3) + ((4 - 1) * 2) -> 9 + 6 -> 15
        let left: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(lit(1i32), Operator::Plus, lit(2i32))),
            Operator::Multiply,
            lit(3i32),
        ));
        let right: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(lit(4i32), Operator::Minus, lit(1i32))),
            Operator::Multiply,
            lit(2i32),
        ));
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(left, Operator::Plus, right));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(literal.value(), &ScalarValue::Int32(Some(15)));
    }

    #[test]
    fn test_no_simplify_with_column() {
        let schema = test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // c1 + 2 should NOT be simplified (has column reference)
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("c1", &schema).unwrap(),
            Operator::Plus,
            lit(2i32),
        ));
        let result = simplifier.simplify(expr).unwrap();
        // Should remain a BinaryExpr, not become a Literal
        assert!(result.downcast_ref::<BinaryExpr>().is_some());
    }

    #[test]
    fn test_partial_simplify_with_column() {
        let schema = test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // (1 + 2) + c1 should simplify the literal part: 3 + c1
        let literal_part: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(lit(1i32), Operator::Plus, lit(2i32)));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            literal_part,
            Operator::Plus,
            col("c1", &schema).unwrap(),
        ));
        let result = simplifier.simplify(expr).unwrap();

        // Should be a BinaryExpr with a Literal(3) on the left
        let binary = as_binary(&result);
        let left_literal = as_literal(binary.left());
        assert_eq!(left_literal.value(), &ScalarValue::Int32(Some(3)));
    }

    /// Regression test for https://github.com/apache/datafusion/issues/22367.
    ///
    /// A leaf `PhysicalExpr` that is neither a `Literal` nor a `Column`
    /// (nor volatile) must not be const-folded: it has no children to
    /// derive constness from, and evaluating it against the dummy batch
    /// produces a value unrelated to its real runtime semantics. Without
    /// the zero-children guard, `all(empty)` would vacuously hold and the
    /// node would be replaced with whatever scalar fell out of the dummy
    /// evaluation. Verify the node is left untouched.
    #[test]
    fn test_no_simplify_opaque_leaf_expr() {
        use arrow::array::ArrayRef;
        use arrow::array::Int32Array;
        use arrow::record_batch::RecordBatch;
        use datafusion_expr_common::columnar_value::ColumnarValue;
        use datafusion_physical_expr_common::physical_expr::PhysicalExpr as PhysicalExprTrait;
        use std::fmt;

        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        struct OpaqueLeaf;

        impl fmt::Display for OpaqueLeaf {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "OpaqueLeaf")
            }
        }

        impl PhysicalExprTrait for OpaqueLeaf {
            fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
                Ok(DataType::Int32)
            }
            fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
                Ok(true)
            }
            fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
                // Simulate the broken FFI Column path: when handed a dummy
                // batch, return whatever scalar happens to materialize. If
                // the simplifier ever reaches this branch for a leaf node,
                // the predicate has already been silently corrupted.
                let arr: ArrayRef = Arc::new(Int32Array::from(vec![0; batch.num_rows()]));
                Ok(ColumnarValue::Array(arr))
            }
            fn children(&self) -> Vec<&Arc<dyn PhysicalExprTrait>> {
                vec![]
            }
            fn with_new_children(
                self: Arc<Self>,
                _children: Vec<Arc<dyn PhysicalExprTrait>>,
            ) -> Result<Arc<dyn PhysicalExprTrait>> {
                Ok(self)
            }
            fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "OpaqueLeaf")
            }
        }

        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        let opaque: Arc<dyn PhysicalExpr> = Arc::new(OpaqueLeaf);
        let result = simplifier.simplify(Arc::clone(&opaque)).unwrap();

        assert!(
            result.downcast_ref::<Literal>().is_none(),
            "opaque leaf must not be rewritten to a Literal, got: {result}"
        );
        assert_eq!(&result, &opaque);
    }

    #[test]
    fn test_simplify_literal_string_concat() {
        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // 'hello' || ' world' -> 'hello world'
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            lit("hello"),
            Operator::StringConcat,
            lit(" world"),
        ));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(
            literal.value(),
            &ScalarValue::Utf8(Some("hello world".to_string()))
        );
    }
}
