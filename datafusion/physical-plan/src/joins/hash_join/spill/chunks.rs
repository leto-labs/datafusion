// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow family JS: the build-chunk join of a written partition that
//! cannot be divided (add-join-spill design decision 5).
//!
//! The join of a written partition is first tried whole. If its partition
//! cannot be divided (a hot key fills it, or it is below the depth limit)
//! and the pool refuses it beyond its floor, its build raises a
//! [`ChunkSignal`] and fails before it produced a row. The run then joins
//! the partition in chunks: each chunk is a stock `CollectLeft` join whose
//! build reads the build file from a [`Cursor`] until the pool refuses (at
//! least one batch, charged without refusal, so every chunk progresses),
//! and whose probe replays the partition's probe files.
//!
//! Each build row is in exactly one chunk and sees every probe row, so the
//! join types that emit only build rows or matched pairs (Inner, LeftSemi,
//! LeftAnti, LeftMark) are the union of their chunks. The types that emit
//! probe rows keep one matched bit per probe row, by the row's number in
//! the replay:
//!
//! | Run type | Each chunk | After the last chunk |
//! |---|---|---|
//! | Right | Inner; marks the probe rows it emits | the probe rows not marked, with null build columns |
//! | RightSemi | RightSemi over the rows not yet marked; marks and emits them | - |
//! | RightAnti | RightSemi over the rows not yet marked; marks them | the probe rows not marked |
//! | RightMark | RightSemi over the rows not yet marked; marks them | every probe row, with its mark |
//!
//! A partition that needs more than `hash_join_spill_max_replay_passes`
//! chunks fails with `ResourcesExhausted`, naming the bound.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow::compute::filter_record_batch;
use datafusion_common::{DataFusionError, JoinType, Result, internal_err};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::projection::ProjectionRef;
use futures::StreamExt;

use super::files::{MatchedBits, RowIds, with_row_id};
use super::{
    Cursor, FileSource, Origin, Output, SpillFilesExec, SpilledBuild, SpilledFile,
    UnmatchedOutput,
};
use crate::stream::RecordBatchStreamAdapter;
use crate::{ExecutionPlan, SendableRecordBatchStream};

/// What each chunk of a run computes, by the run's join type.
#[derive(Debug, Clone, Copy)]
struct Plan {
    chunk_type: JoinType,
    /// The probe rows carry their row number, and the chunks mark them.
    ids: bool,
    /// A chunk reads only the probe rows that no earlier chunk marked.
    skip_matched: bool,
    /// A chunk's rows are output.
    emit: bool,
    rest: Rest,
}

/// The pass over the probe rows after the last chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rest {
    None,
    /// The probe rows that no chunk marked (Right, RightAnti).
    Unmatched,
    /// Every probe row with its mark (RightMark).
    Mark,
}

impl Plan {
    fn of(run_type: JoinType) -> Result<Self> {
        let plan = |chunk_type, ids, skip_matched, emit, rest| Self {
            chunk_type,
            ids,
            skip_matched,
            emit,
            rest,
        };
        Ok(match run_type {
            JoinType::Inner
            | JoinType::LeftSemi
            | JoinType::LeftAnti
            | JoinType::LeftMark => plan(run_type, false, false, true, Rest::None),
            JoinType::Right => plan(JoinType::Inner, true, false, true, Rest::Unmatched),
            JoinType::RightSemi => {
                plan(JoinType::RightSemi, true, true, true, Rest::None)
            }
            JoinType::RightAnti => {
                plan(JoinType::RightSemi, true, true, false, Rest::Unmatched)
            }
            JoinType::RightMark => {
                plan(JoinType::RightSemi, true, true, false, Rest::Mark)
            }
            JoinType::Left | JoinType::Full => {
                return internal_err!(
                    "hash join spill: a {run_type} run is never joined in chunks"
                );
            }
        })
    }
}

/// One run of a written partition joined in build chunks.
pub(super) struct ChunkRun {
    pub(super) build: Arc<SpilledBuild>,
    pub(super) part: usize,
    /// The run's join type: an own run's type, or the build-row pass's.
    pub(super) run_type: JoinType,
    /// The run's projection, over the run type's output.
    pub(super) projection: Option<ProjectionRef>,
    pub(super) build_file: SpilledFile,
    pub(super) probe_files: Vec<SpilledFile>,
    pub(super) output: Arc<UnmatchedOutput>,
    /// How a chunk's rows become the join's rows.
    pub(super) kind: Output,
    pub(super) context: Arc<TaskContext>,
}

impl ChunkRun {
    /// The rows of the run, in the join's schema.
    pub(super) fn stream(self) -> Result<SendableRecordBatchStream> {
        let schema = Arc::clone(&self.output.schema);
        let plan = Plan::of(self.run_type)?;
        let probe_rows: usize = self.probe_files.iter().map(|file| file.rows).sum();
        let (bits, charge) = if plan.ids {
            let charge = MemoryConsumer::new("HashJoinChunkBits")
                .with_can_spill(true)
                .register(self.context.memory_pool());
            // One bit per probe row: the smallest unit, charged without
            // refusal like a chunk's first batch.
            charge.grow(MatchedBits::size(probe_rows));
            (Some(Arc::new(MatchedBits::new(probe_rows))), Some(charge))
        } else {
            (None, None)
        };
        self.build.metrics.chunked.add(1);
        let cursor = Arc::new(Cursor::new(
            self.build_file.clone(),
            self.build.build_manager.clone(),
        ));
        let state = Chunks {
            run: self,
            plan,
            cursor,
            bits,
            _charge: charge,
            chunks: 0,
            current: None,
            rest: None,
            done: false,
        };
        let rows = futures::stream::unfold(state, |mut state| async move {
            state.next().await.map(|item| (item, state))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, rows)))
    }
}

struct Chunks {
    run: ChunkRun,
    plan: Plan,
    cursor: Arc<Cursor>,
    bits: Option<Arc<MatchedBits>>,
    _charge: Option<MemoryReservation>,
    /// Chunks started.
    chunks: usize,
    current: Option<SendableRecordBatchStream>,
    /// The pass after the last chunk, and the row number of its next row.
    rest: Option<(SendableRecordBatchStream, usize)>,
    done: bool,
}

impl Chunks {
    async fn next(&mut self) -> Option<Result<RecordBatch>> {
        if self.done {
            return None;
        }
        let item = self.advance().await;
        if !matches!(item, Some(Ok(_))) {
            self.done = true;
            self.current = None;
            self.rest = None;
        }
        item
    }

    async fn advance(&mut self) -> Option<Result<RecordBatch>> {
        loop {
            if let Some((stream, start)) = &mut self.rest {
                let batch = match stream.next().await? {
                    Ok(batch) => batch,
                    Err(error) => return Some(Err(error)),
                };
                let first = *start;
                *start += batch.num_rows();
                match self.rest_rows(&batch, first) {
                    Ok(Some(batch)) => return Some(Ok(batch)),
                    Ok(None) => continue,
                    Err(error) => return Some(Err(error)),
                }
            }
            if let Some(stream) = &mut self.current {
                match stream.next().await {
                    Some(Ok(batch)) => match self.chunk_rows(batch) {
                        Ok(Some(batch)) => return Some(Ok(batch)),
                        Ok(None) => continue,
                        Err(error) => return Some(Err(error)),
                    },
                    Some(Err(error)) => return Some(Err(error)),
                    None => self.current = None,
                }
                if !self.cursor.is_exhausted() {
                    continue;
                }
                if self.plan.rest == Rest::None {
                    return None;
                }
                match self.rest_stream() {
                    Ok(stream) => self.rest = Some((stream, 0)),
                    Err(error) => return Some(Err(error)),
                }
                continue;
            }
            match self.start_chunk() {
                Ok(stream) => self.current = Some(stream),
                Err(error) => return Some(Err(error)),
            }
        }
    }

    /// Starts the next chunk: a stock join over the rows the cursor gives
    /// until the pool refuses, probed by a replay of the probe files.
    fn start_chunk(&mut self) -> Result<SendableRecordBatchStream> {
        let run = &self.run;
        let build = &run.build;
        let limit = build.options.max_replay_passes;
        if self.chunks >= limit {
            return Err(DataFusionError::ResourcesExhausted(format!(
                "Hash join spill: written partition {} at spill level {} needs more than \
                 datafusion.execution.hash_join_spill_max_replay_passes ({limit}) build \
                 chunks: the pool held about {} of its {} build rows per chunk, and each \
                 chunk replays its {} probe rows",
                run.part,
                build.level + 1,
                run.build_file.rows / self.chunks.max(1),
                run.build_file.rows,
                run.probe_files.iter().map(|file| file.rows).sum::<usize>(),
            )));
        }
        self.chunks += 1;
        if self.chunks > 1 {
            build.metrics.replay_passes.add(1);
        }
        let template = &build.template;
        let level = build.level + 1;
        let origin = Origin {
            rows: run.build_file.rows,
            parent_rows: build.parent_rows,
        };
        let build_leaf = SpillFilesExec::new(
            Arc::clone(&template.build_schema),
            FileSource::Cursor(Arc::clone(&self.cursor)),
            build.build_manager.clone(),
            level,
            origin,
            build.metrics.clone(),
        );
        let probe_manager = build
            .build_manager
            .with_schema(Arc::clone(&template.probe_schema));
        let probe_source = FileSource::Fixed(vec![run.probe_files.clone()]);
        let probe_leaf = match &self.bits {
            Some(bits) if self.plan.ids => SpillFilesExec::new(
                with_row_id(&template.probe_schema),
                probe_source,
                probe_manager,
                level,
                origin,
                build.metrics.clone(),
            )
            .with_row_ids(RowIds {
                skip_matched: self.plan.skip_matched.then(|| Arc::clone(bits)),
            }),
            _ => SpillFilesExec::new(
                Arc::clone(&template.probe_schema),
                probe_source,
                probe_manager,
                level,
                origin,
                build.metrics.clone(),
            ),
        };
        let join = build.sub_join(
            build_leaf,
            probe_leaf,
            self.plan.chunk_type,
            self.chunk_projection(),
        )?;
        join.execute(0, Arc::clone(&run.context))
    }

    /// The projection of a chunk join. With row numbers, the number is the
    /// last output column: the probe leaf appends it after the probe
    /// columns, which keep their indices.
    fn chunk_projection(&self) -> Option<ProjectionRef> {
        if !self.plan.ids {
            return self.run.projection.clone();
        }
        let template = &self.run.build.template;
        let probe = template.probe_schema.fields().len();
        let id = match self.plan.chunk_type {
            JoinType::Inner => template.build_schema.fields().len() + probe,
            _ => probe,
        };
        if !self.plan.emit {
            return Some(Arc::from(vec![id]));
        }
        self.run
            .projection
            .as_ref()
            .map(|projection| projection.iter().copied().chain([id]).collect())
    }

    /// A chunk's output batch: marks its probe rows, and returns the rows
    /// the join outputs.
    fn chunk_rows(&self, batch: RecordBatch) -> Result<Option<RecordBatch>> {
        let batch = match &self.bits {
            Some(bits) if self.plan.ids => {
                let last = batch.num_columns() - 1;
                let Some(ids) = batch.column(last).as_any().downcast_ref::<UInt64Array>()
                else {
                    return internal_err!(
                        "hash join spill: a chunk lost its row numbers"
                    );
                };
                bits.set(ids)?;
                batch.project(&(0..last).collect::<Vec<_>>())?
            }
            _ => batch,
        };
        if !self.plan.emit || batch.num_rows() == 0 {
            return Ok(None);
        }
        let output = &self.run.output;
        Ok(Some(match self.run.kind {
            Output::Own => output.own(batch)?,
            Output::Unmatched => output.emit(&batch)?,
            Output::Same => batch,
        }))
    }

    /// The replay of the probe files after the last chunk.
    fn rest_stream(&self) -> Result<SendableRecordBatchStream> {
        let template = &self.run.build.template;
        let manager = self
            .run
            .build
            .build_manager
            .with_schema(Arc::clone(&template.probe_schema));
        let leaf = SpillFilesExec::new(
            Arc::clone(&template.probe_schema),
            FileSource::Fixed(vec![self.run.probe_files.clone()]),
            manager,
            self.run.build.level + 1,
            Origin {
                rows: 0,
                parent_rows: 0,
            },
            self.run.build.metrics.clone(),
        );
        leaf.execute(0, Arc::clone(&self.run.context))
    }

    /// Rows `start..` of the probe replay after the last chunk.
    fn rest_rows(
        &self,
        batch: &RecordBatch,
        start: usize,
    ) -> Result<Option<RecordBatch>> {
        let Some(bits) = &self.bits else {
            return internal_err!("hash join spill: a probe pass without matched bits");
        };
        let matched = bits.get(start, batch.num_rows());
        let output = &self.run.output;
        let rows = match self.plan.rest {
            Rest::Unmatched => {
                let unmatched = arrow::compute::not(&matched)?;
                let rows = filter_record_batch(batch, &unmatched)?;
                if rows.num_rows() == 0 {
                    return Ok(None);
                }
                output.emit_probe(&rows, None)?
            }
            Rest::Mark => {
                let mark: ArrayRef = Arc::new(matched);
                output.emit_probe(batch, Some(&mark))?
            }
            Rest::None => return Ok(None),
        };
        Ok((rows.num_rows() > 0).then_some(rows))
    }
}

/// The build of one chunk (the hook in `collect_left_input`): it takes
/// batches from its cursor while the pool grants them and an estimate of
/// their table, and gives back the first batch the pool refuses. The first
/// batch of a chunk and its estimate are charged without refusal.
pub(crate) struct ChunkBuild {
    cursor: Arc<Cursor>,
    /// The estimate of the table over the rows taken, held until the table
    /// is built.
    headroom: Option<MemoryReservation>,
}

/// Bytes kept for the table of `rows` rows of `bytes` bytes: their copy,
/// 48 bytes of table and keys per row (the floor's estimate), and the fixed
/// parts.
fn table_estimate(bytes: usize, rows: usize) -> usize {
    bytes + rows * 48 + 256
}

impl ChunkBuild {
    pub(crate) fn new(cursor: Arc<Cursor>) -> Self {
        Self {
            cursor,
            headroom: None,
        }
    }

    /// Takes `batch` into the chunk and returns the bytes charged for it to
    /// `reservation`, or gives it back and returns `None` when the pool
    /// refuses it and the chunk already holds a batch (`first` is false).
    pub(crate) fn admit(
        &mut self,
        batch: &RecordBatch,
        reservation: &MemoryReservation,
        first: bool,
    ) -> Result<Option<usize>> {
        let size = datafusion_common::utils::memory::get_record_batch_memory_size(batch);
        let extra = table_estimate(size, batch.num_rows());
        let headroom = self.headroom.get_or_insert_with(|| reservation.new_empty());
        match reservation.try_grow(size + extra) {
            Ok(()) => {
                reservation.shrink(extra);
                headroom.grow(extra);
            }
            Err(error) if super::is_refusal(&error) && first => {
                reservation.grow(size);
                headroom.grow(extra);
            }
            Err(error) if super::is_refusal(&error) => {
                self.cursor.give_back(batch.clone());
                return Ok(None);
            }
            Err(error) => return Err(error),
        }
        Ok(Some(size))
    }

    /// The rows are all taken: the table is built next, in the room the
    /// estimate kept.
    pub(crate) fn release_headroom(&mut self) {
        self.headroom = None;
    }

    /// Gives back rows that the table could not hold.
    pub(crate) fn give_back(&self, batches: Vec<RecordBatch>) {
        for batch in batches {
            self.cursor.give_back(batch);
        }
    }
}
