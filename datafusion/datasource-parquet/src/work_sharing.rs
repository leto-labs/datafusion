// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Cooperative tail donation at native row-group boundaries (Longbow family
//! WS). The owner keeps its reader, decoder buffers and read-ahead window.
//! Only disjoint whole row groups beyond that window become independent work
//! units for idle sibling streams.
//!
//! The donor rebuilds its decoder with upstream's row-group-local selections
//! ([`ParquetPushDecoderBuilder::with_row_group_selections`]): the retained
//! row groups keep their own selections, so no file-level selection has to be
//! recomputed. A donated unit is the original [`PartitionedFile`] with a
//! [`ParquetAccessPlan`] that scans only the donated row groups, with their
//! selections and fully-matched flags.
//!
//! [`ParquetPushDecoderBuilder::with_row_group_selections`]: parquet::arrow::push_decoder::ParquetPushDecoderBuilder::with_row_group_selections

use std::collections::VecDeque;
use std::sync::Arc;

use datafusion_common::Result;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_stream::work_source::{PublishLease, SharedWorkSource};
use datafusion_physical_expr::expressions::DynamicFilterTracking;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use parquet::arrow::push_decoder::ParquetPushDecoder;
use parquet::file::metadata::ParquetMetaData;

use crate::push_decoder::RgPlanEntry;
use crate::{ParquetAccessPlan, ParquetFileMetrics};

/// Marks a [`PartitionedFile`] whose [`ParquetAccessPlan`] is a unit donated
/// by a sibling stream, and carries the donor's footer.
///
/// The donor credits the donated bytes to its own `bytes_processed` when it
/// publishes them, so the stream that reads the unit credits nothing: the
/// scan's total stays equal to the bytes of its file ranges. The reader of the
/// unit reuses the donor's metadata instead of reading the footer again, so
/// both decode with the same footer (the same object generation).
#[derive(Debug, Clone)]
pub(crate) struct DonatedRowGroups {
    pub(crate) metadata: Arc<ParquetMetaData>,
}

/// Donation state of one eligible stream: unordered, unlimited, not reversed
/// or reordered and without an external row selection. A stream whose
/// dynamic filter is still changing donates only after it completes.
pub(crate) struct RowGroupSharing {
    pub(crate) source: SharedWorkSource,
    pub(crate) file: PartitionedFile,
    /// The stream's post-pruning access plan, in physical row-group order.
    pub(crate) plan: ParquetAccessPlan,
    pub(crate) metadata: Arc<ParquetMetaData>,
    /// The scan predicate, only when it had a dynamic filter that could still
    /// change at open time. Donation waits until no filter can change.
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    /// Row groups the reader may prefetch beyond the current one; they stay
    /// with the owner.
    pub(crate) lookahead: usize,
    pub(crate) metrics: ExecutionPlanMetricsSet,
    pub(crate) partition: usize,
    /// The file's lease from the shared work source: while it is held, idle
    /// siblings wait for this reader's publications. Released when this
    /// sharing state is dropped.
    pub(crate) lease: Option<PublishLease>,
}

impl Drop for RowGroupSharing {
    fn drop(&mut self) {
        if let Some(lease) = &self.lease {
            lease.release();
        }
    }
}

impl RowGroupSharing {
    fn parts(&self, remaining: usize) -> usize {
        // Preserve a useful stream per borrower too. This bounds extra file
        // opens and avoids one-row-group streams that cannot read ahead.
        self.source
            .split_width(remaining / self.lookahead.saturating_add(1))
    }

    /// Whether `remaining` unread row groups are too few to ever give an idle
    /// sibling a useful run. The stream then drops its sharing state, which
    /// releases the lease so waiting siblings can finish.
    pub(crate) fn exhausted(&self, remaining: usize) -> bool {
        remaining / self.lookahead.saturating_add(1) < 2
    }

    /// Whether an idle sibling could take part of `remaining` unread row
    /// groups. One relaxed atomic load on the ordinary path; the predicate is
    /// classified only when a sibling is idle.
    pub(crate) fn wants_work(&self, remaining: usize) -> bool {
        self.parts(remaining) > 1
            && self.predicate.as_ref().is_none_or(|predicate| {
                !matches!(
                    DynamicFilterTracking::classify(predicate),
                    DynamicFilterTracking::Watching(_)
                )
            })
    }

    /// Called only at a decoder boundary, after the stream synchronized
    /// `remaining` with the decoder's frontier. The decoder is rebuilt before
    /// anything is published: a failed rebuild cannot expose work that still
    /// belongs to the original decoder.
    ///
    /// Returns the rebuilt decoder and the bytes of the donated row groups.
    pub(crate) fn offer(
        &self,
        decoder: ParquetPushDecoder,
        remaining: &mut VecDeque<RgPlanEntry>,
    ) -> Result<(ParquetPushDecoder, u64)> {
        let parts = self.parts(remaining.len());
        if parts < 2 {
            return Ok((decoder, 0));
        }
        let indices: Vec<_> = remaining.iter().map(|entry| entry.rg_index).collect();
        let local_len = indices
            .len()
            .div_ceil(parts)
            .max(self.lookahead.saturating_add(1));
        if local_len >= indices.len() {
            return Ok((decoder, 0));
        }
        let local = self
            .plan
            .select_row_groups(&indices[..local_len])
            .prepare(self.metadata.row_groups())?;
        let selections = local
            .row_groups
            .into_iter()
            .map(|group| group.selection)
            .collect();
        // The rebuilt builder keeps the projection, the installed row filter,
        // the selection policy and the buffered bytes of retained groups;
        // only its remaining row-group plan changes. The front row group is
        // retained, so the per-row-group filter toggle stays correct.
        let decoder = decoder
            .into_builder()?
            .with_row_group_selections(selections)
            .build()?;
        let published: Vec<_> =
            self.plan
                .select_row_groups(&indices[local_len..])
                .split(parts - 1)
                .into_iter()
                .map(|plan| {
                    self.file.clone().with_extension(plan).with_extension(
                        DonatedRowGroups {
                            metadata: Arc::clone(&self.metadata),
                        },
                    )
                })
                .collect();
        let donated_bytes = remaining.drain(local_len..).map(|entry| entry.bytes).sum();
        ParquetFileMetrics::add_row_group_units_published(
            &self.metrics,
            self.partition,
            self.file.object_meta.location.as_ref(),
            published.len(),
        );
        self.source.publish(published);
        Ok((decoder, donated_bytes))
    }
}
