// Modified by the Longbow authors; see third_party/README.md.

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Cache key, ownership, dynamic-state and session-scope regressions.

use super::tests::{as_literal, test_schema};
use super::*;
use crate::expressions::{BinaryExpr, DynamicFilterPhysicalExpr, col, lit};
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use datafusion_common::ScalarValue;
use datafusion_expr::Operator;
use datafusion_expr_common::columnar_value::ColumnarValue;
use std::fmt;

#[derive(Debug, Eq)]
struct PublishDuringPreparation {
    filter: Arc<DynamicFilterPhysicalExpr>,
    child: Arc<dyn PhysicalExpr>,
    complete: bool,
    cross_thread: bool,
}

impl fmt::Display for PublishDuringPreparation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PublishDuringPreparation")
    }
}

impl PartialEq for PublishDuringPreparation {
    fn eq(&self, other: &Self) -> bool {
        self.filter == other.filter
            && self.child.eq(&other.child)
            && self.complete == other.complete
            && self.cross_thread == other.cross_thread
    }
}

impl Hash for PublishDuringPreparation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.filter.hash(state);
        self.child.hash(state);
        self.complete.hash(state);
        self.cross_thread.hash(state);
    }
}

impl PhysicalExpr for PublishDuringPreparation {
    fn data_type(&self, _: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, _: &RecordBatch) -> Result<ColumnarValue> {
        let filter = Arc::clone(&self.filter);
        let complete = self.complete;
        if self.cross_thread {
            std::thread::spawn(move || {
                filter.update(lit(false)).unwrap();
                if complete {
                    filter.mark_complete();
                }
            })
            .join()
            .unwrap();
        } else {
            self.filter.update(lit(false))?;
            if self.complete {
                self.filter.mark_complete();
            }
        }
        Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.child]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(Self {
            filter: Arc::clone(&self.filter),
            child: Arc::clone(&children[0]),
            complete: self.complete,
            cross_thread: self.cross_thread,
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn preparation_race_input(
    complete: bool,
    cross_thread: bool,
) -> (Arc<dyn PhysicalExpr>, Arc<DynamicFilterPhysicalExpr>) {
    let filter = Arc::new(DynamicFilterPhysicalExpr::new(vec![lit(0_i32)], lit(true)));
    let publisher = Arc::new(PublishDuringPreparation {
        filter: Arc::clone(&filter),
        child: lit(true),
        complete,
        cross_thread,
    });
    let input = Arc::new(BinaryExpr::new(
        Arc::clone(&filter) as Arc<dyn PhysicalExpr>,
        Operator::And,
        publisher,
    )) as Arc<dyn PhysicalExpr>;
    (input, filter)
}

fn evaluate_boolean(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let batch = create_dummy_batch().unwrap();
    match expr.evaluate(batch).unwrap() {
        ColumnarValue::Scalar(ScalarValue::Boolean(value)) => value.unwrap(),
        value => panic!("expected boolean scalar, got {value:?}"),
    }
}

#[test]
fn preparation_session_scope_disables_cache_and_restores_after_panic() {
    PREPARATION_CACHE.with(|cache| cache.borrow_mut().clear());
    let schema = test_schema();
    let input = col("c1", &schema).unwrap();
    let enabled = PhysicalExprSimplifier::new(&schema);
    enabled.simplify(Arc::clone(&input)).unwrap();
    with_preparation_cache(false, || {
        let disabled = PhysicalExprSimplifier::new(&schema);
        assert!(disabled.schema_fingerprint.is_none());
        disabled.simplify(Arc::clone(&input)).unwrap();
        PREPARATION_CACHE.with(|cache| assert_eq!(cache.borrow().lookups, 1));
        with_preparation_cache(true, || {
            PhysicalExprSimplifier::new(&schema)
                .simplify(Arc::clone(&input))
                .unwrap();
        });
        assert!(!preparation_cache_enabled());
    });
    assert!(preparation_cache_enabled());
    assert!(
        std::panic::catch_unwind(|| with_preparation_cache(false, || panic!("probe"))).is_err()
    );
    assert!(preparation_cache_enabled());
    enabled.simplify(input).unwrap();
    PREPARATION_CACHE.with(|cache| assert_eq!(cache.borrow().hits, 2));
}

#[test]
fn preparation_cache_reuses_static_entry() {
    let schema = test_schema();
    let input = lit(ScalarValue::Int32(Some(7)));
    let output = Arc::clone(&input);
    let fingerprint = schema_fingerprint(&schema);
    let mut cache = PreparationCache::default();

    cache.insert(
        &input,
        &output,
        &schema,
        fingerprint,
        DEFAULT_EVALUATION_CONTEXT,
    );
    let cached = cache
        .lookup(&input, &schema, fingerprint, DEFAULT_EVALUATION_CONTEXT)
        .expect("matching schema and expression should hit");

    assert!(Arc::ptr_eq(&cached, &input));
    assert_eq!(cache.lookups, 1);
    assert_eq!(cache.hits, 1);
    assert_eq!(cache.misses, 0);
}

#[test]
fn preparation_equal_schemas_hit_without_retaining_expression_owners() {
    let first = test_schema();
    let equal = test_schema();
    let input = lit(7_i32);
    let weak = Arc::downgrade(&input);
    let mut cache = PreparationCache::default();
    cache.insert(&input, &input, &first, schema_fingerprint(&first), 0);
    assert_eq!(Arc::strong_count(&input), 1);
    let hit = cache
        .lookup(&input, &equal, schema_fingerprint(&equal), 0)
        .unwrap();
    assert!(Arc::ptr_eq(&input, &hit));
    drop(hit);
    drop(input);
    assert!(weak.upgrade().is_none());
    cache.prune_dead();
    assert!(cache.entries.is_empty());
}

#[test]
fn preparation_schema_hash_collision_and_context_do_not_alias() {
    let schema = test_schema();
    let different = Schema::empty();
    let input = lit(7_i32);
    let mut cache = PreparationCache::default();
    cache.insert(&input, &input, &schema, 42, 0);
    assert!(cache.lookup(&input, &different, 42, 0).is_none());
    assert!(cache.lookup(&input, &schema, 42, 1).is_none());
    assert!(cache.lookup(&input, &schema, 42, 0).is_some());
}

#[test]
fn preparation_cache_separates_schema_entries() {
    let first_schema = test_schema();
    let second_schema = Schema::new(vec![
        Field::new("c1", DataType::Int64, false),
        Field::new("c2", DataType::Int64, false),
        Field::new("c3", DataType::Utf8, false),
    ]);
    let input = lit(ScalarValue::Int32(Some(7)));
    let mut cache = PreparationCache::default();
    cache.insert(
        &input,
        &input,
        &first_schema,
        schema_fingerprint(&first_schema),
        DEFAULT_EVALUATION_CONTEXT,
    );

    assert!(
        cache
            .lookup(
                &input,
                &second_schema,
                schema_fingerprint(&second_schema),
                DEFAULT_EVALUATION_CONTEXT,
            )
            .is_none()
    );
    assert_eq!(cache.misses, 1);
}

#[test]
fn preparation_cache_excludes_dynamic_filter_generations() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let column = col("a", &schema).unwrap();
    let filter = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::clone(&column)],
        lit(true),
    ));
    let input = Arc::new(BinaryExpr::new(
        column,
        Operator::Gt,
        Arc::clone(&filter) as Arc<dyn PhysicalExpr>,
    )) as Arc<dyn PhysicalExpr>;
    let fingerprint = schema_fingerprint(&schema);
    let mut cache = PreparationCache::default();
    cache.insert(
        &input,
        &input,
        &schema,
        fingerprint,
        DEFAULT_EVALUATION_CONTEXT,
    );

    assert!(cache.entries.is_empty());
    filter.update(lit(false)).unwrap();
    assert!(
        cache
            .lookup(&input, &schema, fingerprint, DEFAULT_EVALUATION_CONTEXT)
            .is_none()
    );
    assert!(cache.entries.is_empty());
}

#[test]
fn preparation_cache_invalidates_after_completion_update() {
    PREPARATION_CACHE.with(|cache| cache.borrow_mut().clear());
    let filter = Arc::new(DynamicFilterPhysicalExpr::new(vec![], lit(true)));
    let input = Arc::clone(&filter) as Arc<dyn PhysicalExpr>;
    let schema = Schema::empty();
    let simplifier = PhysicalExprSimplifier::new(&schema);

    simplifier.simplify(Arc::clone(&input)).unwrap();
    filter.mark_complete();
    filter.update(lit(false)).unwrap();
    let result = simplifier.simplify(input).unwrap();

    assert!(!evaluate_boolean(&result));
    PREPARATION_CACHE.with(|cache| assert!(cache.borrow().entries.is_empty()));
}

#[test]
fn preparation_skips_nested_dynamic_filter_tree() {
    PREPARATION_CACHE.with(|cache| cache.borrow_mut().clear());
    let inner = Arc::new(DynamicFilterPhysicalExpr::new(vec![lit(0_i32)], lit(1_i64)));
    let outer = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![lit(0_i32)],
        Arc::clone(&inner) as Arc<dyn PhysicalExpr>,
    ));
    outer.mark_complete();
    let input = outer as Arc<dyn PhysicalExpr>;
    let schema = Schema::empty();
    let simplifier = PhysicalExprSimplifier::new(&schema);
    let first = simplifier.simplify(Arc::clone(&input)).unwrap();
    assert_eq!(as_literal(&first).value(), &ScalarValue::Int64(Some(1)));
    inner.update(lit(2_i64)).unwrap();
    let second = simplifier.simplify(Arc::clone(&input)).unwrap();
    let fresh = simplifier.simplify_uncached(input).unwrap();
    assert_eq!(as_literal(&second).value(), &ScalarValue::Int64(Some(2)));
    assert_eq!(&second, &fresh);
    PREPARATION_CACHE.with(|cache| assert!(cache.borrow().entries.is_empty()));
}

#[test]
fn preparation_does_not_cache_filters_that_later_publish_nested_state() {
    PREPARATION_CACHE.with(|cache| cache.borrow_mut().clear());
    let schema = Schema::empty();
    let filter = Arc::new(DynamicFilterPhysicalExpr::new(vec![], lit(true)));
    let input = Arc::clone(&filter) as Arc<dyn PhysicalExpr>;
    let simplifier = PhysicalExprSimplifier::new(&schema);
    simplifier.simplify(Arc::clone(&input)).unwrap();
    PREPARATION_CACHE.with(|cache| assert!(cache.borrow().entries.is_empty()));
    let inner = Arc::new(DynamicFilterPhysicalExpr::new(vec![], lit(true)));
    filter
        .update(Arc::clone(&inner) as Arc<dyn PhysicalExpr>)
        .unwrap();
    simplifier.simplify(Arc::clone(&input)).unwrap();
    inner.update(lit(false)).unwrap();
    let prepared = simplifier.simplify(Arc::clone(&input)).unwrap();
    assert!(!evaluate_boolean(&prepared));
    assert_eq!(
        evaluate_boolean(&prepared),
        evaluate_boolean(&simplifier.simplify_uncached(input).unwrap())
    );
    PREPARATION_CACHE.with(|cache| assert!(cache.borrow().entries.is_empty()));
}

#[test]
fn preparation_cache_is_bounded_and_releases_dead_entries() {
    let mut cache = PreparationCache::default();
    let mut inputs = Vec::with_capacity(PREPARATION_CACHE_CAPACITY * 2);
    for index in 0..(PREPARATION_CACHE_CAPACITY * 2) {
        let input = lit(ScalarValue::Int64(Some(index as i64)));
        let schema = Schema::new(vec![Field::new(
            format!("c{index}"),
            DataType::Int64,
            false,
        )]);
        cache.insert(
            &input,
            &input,
            &schema,
            schema_fingerprint(&schema),
            DEFAULT_EVALUATION_CONTEXT,
        );
        inputs.push(input);
    }

    assert_eq!(cache.entries.len(), PREPARATION_CACHE_CAPACITY);
    drop(inputs);

    let probe = lit(ScalarValue::Int64(Some(-1)));
    let probe_schema = Schema::empty();
    for _ in 0..PREPARATION_CACHE_PRUNE_INTERVAL {
        assert!(
            cache
                .lookup(
                    &probe,
                    &probe_schema,
                    DEFAULT_EVALUATION_CONTEXT,
                    DEFAULT_EVALUATION_CONTEXT,
                )
                .is_none()
        );
    }
    assert!(cache.entries.is_empty());
}

#[test]
fn preparation_update_during_walk_is_retried_before_caching() {
    let schema = Schema::empty();
    let (input, _) = preparation_race_input(false, false);
    let simplifier = PhysicalExprSimplifier::new(&schema);

    let prepared = simplifier.simplify(Arc::clone(&input)).unwrap();
    let fresh = simplifier.simplify_uncached(input).unwrap();

    assert!(!evaluate_boolean(&prepared));
    assert!(!evaluate_boolean(&fresh));
    PREPARATION_CACHE.with(|cache| assert!(cache.borrow().entries.is_empty()));
}

#[test]
fn preparation_update_and_completion_before_insertion_is_not_rebaselined() {
    let schema = Schema::empty();
    let (input, _) = preparation_race_input(true, false);
    let simplifier = PhysicalExprSimplifier::new(&schema);

    let prepared = simplifier.simplify(Arc::clone(&input)).unwrap();
    let fresh = simplifier.simplify_uncached(input).unwrap();

    assert!(!evaluate_boolean(&prepared));
    assert!(!evaluate_boolean(&fresh));
    PREPARATION_CACHE.with(|cache| {
        let cache = cache.borrow();
        assert!(cache.entries.len() <= 1);
    });
}

#[test]
fn preparation_cross_thread_publication_returns_current_output() {
    let schema = Schema::empty();
    let (input, _) = preparation_race_input(false, true);
    let simplifier = PhysicalExprSimplifier::new(&schema);

    let prepared = simplifier.simplify(Arc::clone(&input)).unwrap();
    let fresh = simplifier.simplify_uncached(input).unwrap();

    assert_eq!(evaluate_boolean(&prepared), evaluate_boolean(&fresh));
    assert!(!evaluate_boolean(&prepared));
    PREPARATION_CACHE.with(|cache| assert!(cache.borrow().entries.is_empty()));
}

fn assert_schema_variant_is_not_reused(first: Schema, second: Schema) {
    PREPARATION_CACHE.with(|cache| cache.borrow_mut().clear());
    let input = col("a", &first).unwrap();
    let first_result = PhysicalExprSimplifier::new(&first)
        .simplify(Arc::clone(&input))
        .unwrap();
    let second_result = PhysicalExprSimplifier::new(&second)
        .simplify(Arc::clone(&input))
        .unwrap();

    assert_eq!(
        first_result.data_type(&first).unwrap(),
        input.data_type(&first).unwrap()
    );
    assert_eq!(
        second_result.data_type(&second).unwrap(),
        input.data_type(&second).unwrap()
    );
    PREPARATION_CACHE.with(|cache| {
        let cache = cache.borrow();
        assert_eq!(cache.misses, 2);
        assert_eq!(cache.hits, 0);
    });
}

#[test]
fn preparation_cache_uses_full_schema_equality() {
    let base = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

    assert_schema_variant_is_not_reused(
        base.clone()
            .with_metadata([(String::from("mode"), String::from("a"))].into()),
        base.clone()
            .with_metadata([(String::from("mode"), String::from("b"))].into()),
    );
    assert_schema_variant_is_not_reused(
        Schema::new(vec![
            Field::new("a", DataType::Int32, false)
                .with_metadata([(String::from("unit"), String::from("a"))].into()),
        ]),
        Schema::new(vec![
            Field::new("a", DataType::Int32, false)
                .with_metadata([(String::from("unit"), String::from("b"))].into()),
        ]),
    );
    assert_schema_variant_is_not_reused(
        Schema::new(vec![Field::new("a", DataType::Int32, false)]),
        Schema::new(vec![Field::new("a", DataType::Int32, true)]),
    );
    assert_schema_variant_is_not_reused(
        Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int64, false),
        ]),
        Schema::new(vec![
            Field::new("b", DataType::Int64, false),
            Field::new("a", DataType::Int32, false),
        ]),
    );
    assert_schema_variant_is_not_reused(
        Schema::new(vec![Field::new(
            "a",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
            false,
        )]),
        Schema::new(vec![Field::new(
            "a",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            false,
        )]),
    );
    assert_schema_variant_is_not_reused(
        Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        )]),
        Schema::new(vec![Field::new(
            "a",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            false,
        )]),
    );
}

#[test]
fn preparation_reuses_prepared_tree_on_repeat() {
    // Use a fresh thread so unrelated parallel unit tests cannot affect the
    // thread-local counters used by this call-count regression.
    std::thread::spawn(|| {
        PREPARATION_CACHE.with(|cache| cache.borrow_mut().clear());
        let schema = test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);
        let input = col("c1", &schema).unwrap();

        let first = simplifier.simplify(Arc::clone(&input)).unwrap();
        let second = simplifier.simplify(input).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        PREPARATION_CACHE.with(|cache| {
            let cache = cache.borrow();
            assert_eq!(cache.misses, 1);
            assert_eq!(cache.hits, 1);
        });
    })
    .join()
    .expect("cache regression thread must complete");
}
