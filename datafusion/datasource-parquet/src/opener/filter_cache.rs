// Longbow notice: this file was added by the Longbow authors (family OO); see
// third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Filter preparation once per scan and physical file schema (Longbow family
//! OO).
//!
//! Opening a file adapts the scan's predicate and projection to the file's
//! physical schema, simplifies them, and builds the row-group pruning
//! predicate, its inversion (full-match detection) and the page pruning
//! predicates from the result. All of that depends only on the inputs that
//! [`FilterInputs`] names, which are the same for every file of a scan whose
//! physical schema is equal, and for every unit donated from such a file. A
//! [`PreparedFilterCache`], shared by the clones of one `ParquetSource`,
//! keeps the result.
//!
//! The key holds the predicate and the projection expressions themselves, so
//! a per-file rewrite (partition values, statistics constants,
//! `input_file_name()`) is a different key and is prepared as before. A
//! predicate with dynamic filters is keyed by its snapshot generation as
//! well: the adaptation can fold a filter's current value, and the pruning
//! predicates snapshot it. An entry is kept only when the generation did not
//! change while it was built, so it equals what a file opened at that
//! generation would prepare.
//!
//! The file-level pruning predicate (statistics of the whole file, before
//! its metadata is read) of a predicate whose dynamic filters can no longer
//! change is kept the same way; a predicate that can still change keeps its
//! per-file `FilePruner`, which also drives early stopping.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};

use arrow::datatypes::{Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::{PhysicalExpr, snapshot_generation};
use datafusion_physical_plan::metrics::Count;
use datafusion_pruning::PruningPredicate;

use super::VirtualColumnsState;
use crate::page_filter::PagePruningAccessPlanFilter;
use crate::row_group_filter::InvertedPredicate;

/// Prepared entries kept per scan: physical schema variants times predicate
/// generations still in use.
const CAPACITY: usize = 16;

/// The inputs that decide a file's prepared filters.
pub(crate) struct FilterInputs<'a> {
    /// The file's predicate before adaptation.
    pub(crate) predicate: Option<&'a Arc<dyn PhysicalExpr>>,
    /// The file's projection before adaptation.
    pub(crate) projection: &'a ProjectionExprs,
    /// The schemas the rewriter maps between, with virtual fields when the
    /// scan has any.
    pub(crate) logical: &'a SchemaRef,
    pub(crate) physical: &'a SchemaRef,
    /// The file's physical schema without virtual fields: the pruning schema.
    pub(crate) file_schema: &'a SchemaRef,
    pub(crate) virtual_state: Option<&'a Arc<VirtualColumnsState>>,
    pub(crate) adapter: &'a Arc<dyn PhysicalExprAdapterFactory>,
    pub(crate) max_in_list_size: usize,
    pub(crate) enable_page_index: bool,
}

impl FilterInputs<'_> {
    /// The snapshot generation of the predicate's dynamic filters; zero for
    /// a static predicate.
    fn generation(&self) -> u64 {
        self.predicate.map_or(0, snapshot_generation)
    }
}

/// A file's adapted predicate and projection with the pruning predicates
/// built from them.
#[derive(Debug)]
pub(crate) struct PreparedFilters {
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) projection: ProjectionExprs,
    pub(crate) pruning_predicate: Option<Arc<PruningPredicate>>,
    pub(crate) page_pruning_predicate: Option<Arc<PagePruningAccessPlanFilter>>,
    /// The inversion of `pruning_predicate` for full-match detection, built
    /// on first use.
    pub(crate) inverted: InvertedPredicate,
    /// Pruning predicate creation errors of one build. Every file that uses
    /// the entry adds them to its counter, as a build per file would.
    pub(crate) creation_errors: usize,
}

impl PreparedFilters {
    /// Build the pruning predicates from an adapted `predicate` and
    /// `projection`.
    pub(crate) fn build(
        predicate: Option<Arc<dyn PhysicalExpr>>,
        projection: ProjectionExprs,
        file_schema: &SchemaRef,
        max_in_list_size: usize,
        enable_page_index: bool,
    ) -> Self {
        let errors = Count::new();
        let pruning_predicate = super::build_pruning_predicates(
            predicate.as_ref(),
            file_schema,
            &errors,
            max_in_list_size,
        );
        // Only build page pruning predicate if page index is enabled
        let page_pruning_predicate = if enable_page_index {
            predicate.as_ref().and_then(|predicate| {
                let p = super::build_page_pruning_predicate(
                    predicate,
                    file_schema,
                    max_in_list_size,
                );
                (p.filter_number() > 0).then_some(p)
            })
        } else {
            None
        };
        Self {
            predicate,
            projection,
            pruning_predicate,
            page_pruning_predicate,
            inverted: InvertedPredicate::default(),
            creation_errors: errors.value(),
        }
    }
}

struct Entry {
    predicate: Option<Arc<dyn PhysicalExpr>>,
    projection: ProjectionExprs,
    logical: SchemaRef,
    physical: SchemaRef,
    file_schema: SchemaRef,
    virtual_state: Option<Arc<VirtualColumnsState>>,
    adapter: Arc<dyn PhysicalExprAdapterFactory>,
    max_in_list_size: usize,
    enable_page_index: bool,
    /// Whether the predicate has a dynamic filter; only then is the
    /// generation compared.
    dynamic: bool,
    generation: u64,
    filters: Arc<PreparedFilters>,
}

fn same_schema(a: &Schema, b: &SchemaRef) -> bool {
    std::ptr::eq(a, b.as_ref()) || a == b.as_ref()
}

fn same_expr(a: &Arc<dyn PhysicalExpr>, b: &Arc<dyn PhysicalExpr>) -> bool {
    Arc::ptr_eq(a, b)
}

impl Entry {
    /// Whether this entry was built from `inputs`, apart from the
    /// generation.
    fn matches(&self, inputs: &FilterInputs<'_>) -> bool {
        let predicate = match (&self.predicate, inputs.predicate) {
            (None, None) => true,
            (Some(a), Some(b)) => same_expr(a, b),
            _ => false,
        };
        predicate
            && self.max_in_list_size == inputs.max_in_list_size
            && self.enable_page_index == inputs.enable_page_index
            && std::ptr::addr_eq(Arc::as_ptr(&self.adapter), Arc::as_ptr(inputs.adapter))
            && match (&self.virtual_state, inputs.virtual_state) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
            && self.projection.iter().count() == inputs.projection.iter().count()
            && self
                .projection
                .iter()
                .zip(inputs.projection.iter())
                .all(|(a, b)| a.alias == b.alias && same_expr(&a.expr, &b.expr))
            && same_schema(&self.logical, inputs.logical)
            && same_schema(&self.physical, inputs.physical)
            && same_schema(&self.file_schema, inputs.file_schema)
    }
}

/// A file-level pruning predicate: the scan predicate (after per-file
/// literal folding) against the logical file schema, which file statistics
/// are evaluated against before any metadata is read.
struct FileLevelEntry {
    predicate: Arc<dyn PhysicalExpr>,
    logical: SchemaRef,
    generation: u64,
    pruning: Option<Arc<PruningPredicate>>,
    creation_errors: usize,
}

/// The prepared filters of one scan, shared by its partitions and by the
/// clones of its `ParquetSource`.
#[derive(Default)]
pub(crate) struct PreparedFilterCache {
    entries: Mutex<VecDeque<Entry>>,
    file_level: Mutex<VecDeque<FileLevelEntry>>,
}

/// Prints no contents: a plan's debug output must not change as its scans
/// open files.
impl std::fmt::Debug for PreparedFilterCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedFilterCache")
            .finish_non_exhaustive()
    }
}

impl PreparedFilterCache {
    /// The prepared filters of `inputs`, built by `build` when no entry
    /// matches. `build` runs outside the lock.
    pub(crate) fn get_or_build(
        &self,
        inputs: &FilterInputs<'_>,
        build: impl FnOnce() -> Result<PreparedFilters>,
    ) -> Result<Arc<PreparedFilters>> {
        let mut generation = None;
        {
            let entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(entry) = entries.iter().rev().find(|entry| entry.matches(inputs))
            {
                if !entry.dynamic {
                    return Ok(Arc::clone(&entry.filters));
                }
                let current = inputs.generation();
                if entry.generation == current {
                    return Ok(Arc::clone(&entry.filters));
                }
                generation = Some(current);
            }
        }
        let before = generation.unwrap_or_else(|| inputs.generation());
        let filters = Arc::new(build()?);
        // A dynamic filter that changed during the build may have been read
        // at either value: use the result for this file only.
        if inputs.generation() != before {
            return Ok(filters);
        }
        let dynamic = before != 0
            || inputs.predicate.is_some_and(|predicate| {
                datafusion_physical_expr::expressions::DynamicFilterTracking::classify(
                    predicate,
                )
                .contains_dynamic_filter()
            });
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        // One entry per key: a newer generation replaces the older one.
        entries.retain(|entry| !entry.matches(inputs));
        if entries.len() >= CAPACITY {
            entries.pop_front();
        }
        entries.push_back(Entry {
            predicate: inputs.predicate.map(Arc::clone),
            projection: inputs.projection.clone(),
            logical: Arc::clone(inputs.logical),
            physical: Arc::clone(inputs.physical),
            file_schema: Arc::clone(inputs.file_schema),
            virtual_state: inputs.virtual_state.map(Arc::clone),
            adapter: Arc::clone(inputs.adapter),
            max_in_list_size: inputs.max_in_list_size,
            enable_page_index: inputs.enable_page_index,
            dynamic,
            generation: before,
            filters: Arc::clone(&filters),
        });
        Ok(filters)
    }

    /// The file-level pruning predicate of `predicate` against `logical`, as
    /// [`datafusion_pruning::FilePruner`] builds it for a predicate whose
    /// dynamic filters can no longer change, built once per generation.
    /// Creation errors of the build are added to `errors` on every call, as
    /// a build per file would add them.
    pub(crate) fn file_pruning(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        logical: &SchemaRef,
        errors: &Count,
    ) -> Option<Arc<PruningPredicate>> {
        let generation = snapshot_generation(predicate);
        let matches = |entry: &FileLevelEntry| {
            Arc::ptr_eq(&entry.predicate, predicate)
                && same_schema(&entry.logical, logical)
        };
        {
            let entries = self
                .file_level
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if let Some(entry) = entries
                .iter()
                .find(|entry| matches(entry) && entry.generation == generation)
            {
                errors.add(entry.creation_errors);
                return entry.pruning.clone();
            }
        }
        let local = Count::new();
        let pruning = datafusion_pruning::build_pruning_predicate(
            Arc::clone(predicate),
            logical,
            &local,
        );
        errors.add(local.value());
        if snapshot_generation(predicate) == generation {
            let mut entries = self
                .file_level
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            entries.retain(|entry| !matches(entry));
            if entries.len() >= CAPACITY {
                entries.pop_front();
            }
            entries.push_back(FileLevelEntry {
                predicate: Arc::clone(predicate),
                logical: Arc::clone(logical),
                generation,
                pruning: pruning.clone(),
                creation_errors: local.value(),
            });
        }
        pruning
    }

    /// The number of entries (tests).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}
