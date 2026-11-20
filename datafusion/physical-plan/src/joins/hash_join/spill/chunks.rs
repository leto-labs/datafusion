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
            // One bit per probe row of the partition, admitted by the pool;
            // a bitmap up to a quarter of the workspace may be charged
            // without refusal (the stream's share), a larger one refused fails.
            let bytes = MatchedBits::size(probe_rows);
            if let Err(error) = charge.try_grow(bytes) {
                if !super::is_refusal(&error) || bytes > self.build.options.bits_unit() {
                    return Err(error.context(format!(
                        "Hash join spill: the matched bits of the {probe_rows} probe \
                         rows of written partition {} ({bytes} bytes) for its build \
                         chunks",
                        self.part
                    )));
                }
                charge.grow(bytes);
            }
            (Some(Arc::new(MatchedBits::new(probe_rows))), Some(charge))
        } else {
            (None, None)
        };
        self.build.metrics.chunked.add(1);
        let run_charge = MemoryConsumer::new(format!("HashJoinChunk[{}]", self.part))
            .with_can_spill(true)
            .register(self.context.memory_pool());
        let cursor = Arc::new(Cursor::new(
            self.build_file.clone(),
            self.build.build_manager.clone(),
            self.build.options.chunk_unit(),
            run_charge,
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

/// The build of one chunk (the hook in `collect_left_input`): it admits
/// batches from its cursor while the pool grants each batch and an estimate
/// of its table, and gives back the first batch the pool refuses. Every byte
/// lives in the run's one charge, held by the cursor: a batch given back
/// keeps its bytes there, and the table's charge is split off it, so no
/// transfer grows one reservation before another shrinks. The first batch
/// of a chunk may be charged without refusal only within the chunk unit
/// (three quarters of the workspace); a larger one refused fails with
/// `ResourcesExhausted` naming it.
pub(crate) struct ChunkBuild {
    cursor: Arc<Cursor>,
    /// Bytes of the run's charge that this chunk holds: its rows and the
    /// table estimate.
    granted: usize,
    /// Of those, the bytes of the rows.
    rows: usize,
    /// Bytes of the run's charge reserved for the next batch read.
    permit: usize,
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
            granted: 0,
            rows: 0,
            permit: 0,
        }
    }

    /// The bytes this chunk's table may use: the cap of its build.
    pub(crate) fn granted(&self) -> usize {
        self.granted
    }

    /// The bytes of the chunk's rows.
    pub(crate) fn rows_bytes(&self) -> usize {
        self.rows
    }

    /// Takes `batch` into the chunk and returns its bytes. A batch after the
    /// first was read only under a read permit (one chunk unit, the most a
    /// written batch needs with its table), which now covers it; the first
    /// is admitted by the pool, or charged without refusal within the chunk
    /// unit. Then a permit for the next batch is reserved; if the pool
    /// refuses it, the chunk's input ends before the next read, so no batch
    /// is ever read that the chunk cannot hold.
    pub(crate) fn admit(
        &mut self,
        batch: &RecordBatch,
        first: bool,
    ) -> Result<Option<usize>> {
        let size = datafusion_common::utils::memory::get_record_batch_memory_size(batch);
        let rows = batch.num_rows();
        let extra = table_estimate(size, rows);
        // A batch given back before is still charged: its bytes move over.
        let carried = self.cursor.take_carried(size);
        if carried {
            self.granted += size;
            self.rows += size;
        }
        let need = if carried { extra } else { size + extra };
        let permit = std::mem::take(&mut self.permit);
        if permit >= need {
            self.cursor.shrink(permit - need);
        } else {
            if permit > 0 {
                self.cursor.shrink(permit);
            }
            match self.cursor.try_grow(need) {
                Ok(()) => {}
                Err(error) if super::is_refusal(&error) && first => {
                    if size + extra > self.cursor.unit() {
                        return Err(DataFusionError::ResourcesExhausted(format!(
                            "Hash join spill: a build chunk's smallest unit, one batch \
                             of {rows} rows ({size} bytes, {} bytes with its table), is \
                             larger than three quarters of datafusion.execution.\
                             hash_join_spill_reservation_bytes ({} bytes), and the pool \
                             refused it: {error}",
                            size + extra,
                            self.cursor.unit(),
                        )));
                    }
                    // The smallest unit, inside the stream's workspace share.
                    self.cursor.grow(need);
                }
                Err(error) => {
                    return Err(error.context(format!(
                        "Hash join spill: a build batch of {rows} rows ({size} bytes) \
                         above the chunk unit"
                    )));
                }
            }
        }
        self.granted += need;
        if !carried {
            self.rows += size;
        }
        match self.cursor.try_grow(self.cursor.unit()) {
            Ok(()) => self.permit = self.cursor.unit(),
            Err(_) => self.cursor.stop_chunk(),
        }
        Ok(Some(size))
    }

    /// The build input ended without a row (the chunk before stopped just
    /// before the end of the file): an empty table's fixed parts, admitted
    /// by the pool or charged without refusal (far below the chunk unit).
    pub(crate) fn admit_empty(&mut self) {
        const EMPTY_TABLE: usize = 1024;
        if self.cursor.try_grow(EMPTY_TABLE).is_err() {
            self.cursor.grow(EMPTY_TABLE);
        }
        self.granted += EMPTY_TABLE;
    }

    /// The build input ended: the unused read permit goes back to the pool.
    pub(crate) fn release_permit(&mut self) {
        let permit = std::mem::take(&mut self.permit);
        if permit > 0 {
            self.cursor.shrink(permit);
        }
    }

    /// Gives back batches that the chunk's table could not hold; their bytes
    /// stay charged in the run.
    pub(crate) fn give_back(&mut self, batches: Vec<RecordBatch>) {
        for batch in batches {
            let size =
                datafusion_common::utils::memory::get_record_batch_memory_size(&batch);
            self.granted = self.granted.saturating_sub(size);
            self.rows = self.rows.saturating_sub(size);
            self.cursor.give_back(batch, size);
        }
    }

    /// The table, of `table` bytes, is built: its charge is split off the
    /// run's charge and the rest of the grant goes back to the pool.
    pub(crate) fn take_charge(&mut self, table: usize) -> MemoryReservation {
        let charge = self.cursor.split(table.min(self.granted));
        self.cursor.shrink(self.granted.saturating_sub(table));
        self.granted = 0;
        self.rows = 0;
        charge
    }
}
