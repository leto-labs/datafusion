// Added by the Longbow authors (family JS); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow family JS, add-join-spill section 3: recursion with independent
//! bits, the build-chunk join of a partition that cannot be divided, the
//! replay bound and role reversal.

use super::*;

/// `rows` build rows in batches of `batch_rows`: a share `hot_percent` of
/// them has key 7 (NULL when `null_hot`), the rest spread over `keys` keys.
fn skewed(
    prefix: &str,
    rows: usize,
    hot_percent: usize,
    null_hot: bool,
    keys: i32,
    batch_rows: usize,
) -> Vec<RecordBatch> {
    table(prefix, rows, keys, 0, batch_rows)
        .into_iter()
        .enumerate()
        .map(|(index, batch)| {
            let start = index * batch_rows;
            let keys = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(offset, key)| {
                    if (start + offset) * 37 % 100 < hot_percent {
                        (!null_hot).then_some(7)
                    } else {
                        Some(key.map_or(50, |key| key + 100))
                    }
                })
                .collect::<Int32Array>();
            RecordBatch::try_new(
                batch.schema(),
                vec![Arc::new(keys), Arc::clone(batch.column(1))],
            )
            .unwrap()
        })
        .collect()
}

/// A probe side with few rows of key 7 (or NULL), so that a hot key does
/// not multiply the output.
fn sparse_probe(rows: usize, null_hot: bool) -> Vec<RecordBatch> {
    table("p", rows, 3001, 1, 250)
        .into_iter()
        .enumerate()
        .map(|(index, batch)| {
            let start = index * 250;
            let keys = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(offset, key)| match (start + offset) % 1500 {
                    0 => (!null_hot).then_some(7),
                    _ => Some(key.map_or(51, |key| key + 100)),
                })
                .collect::<Int32Array>();
            RecordBatch::try_new(
                batch.schema(),
                vec![Arc::new(keys), Arc::clone(batch.column(1))],
            )
            .unwrap()
        })
        .collect()
}

fn label(case: &Case) -> String {
    format!(
        "{} {:?} filter={} {:?}",
        case.join_type, case.null_equality, case.filter, case.shape
    )
}

/// Runs `case` in a pool of `limit` bytes and checks it against the
/// unbounded join; returns the plan's metrics.
async fn check(
    case: &Case,
    build: &[RecordBatch],
    probe: &[RecordBatch],
    config: SessionConfig,
    limit: usize,
) -> Result<MetricsSet> {
    let plan = join_plan(case, build, probe)?;
    let context = task_context(config, Some(limit), true);
    let actual = run(&plan, Arc::clone(&context))
        .await
        .unwrap_or_else(|error| panic!("{}: {error}", label(case)));
    assert_eq!(
        batches_to_sort_string(&actual),
        expected(case, build, probe).await,
        "{}",
        label(case)
    );
    assert_eq!(context.memory_pool().reserved(), 0, "{}", label(case));
    Ok(plan.metrics().unwrap())
}

/// A hot key in a pool far below its partition: every join type, both null
/// equalities, with and without a filter, in `CollectLeft` with one and
/// three probe streams and `Partitioned` mode, is joined in several build
/// chunks and equals the unbounded join.
#[tokio::test]
async fn hot_key_build_chunks_keep_every_join_type() -> Result<()> {
    // 24,000 rows, 70% of them key 7: the hot partition's file holds a few
    // batches of 8,192 rows, each a chunk.
    let build = skewed("b", 24000, 70, false, 3001, 500);
    let probe = sparse_probe(3000, false);
    let mut replays = 0;
    for case in all_cases(&[
        Shape::CollectLeft(1),
        Shape::CollectLeft(3),
        Shape::Partitioned(2),
    ]) {
        // A pool that every stream's chunks share: with chunks admitted by
        // bytes (review MJJ98), three streams in 64 KiB need more chunks
        // than the replay bound; 192 KiB keeps the hot partition chunked.
        let config = no_reversal(spill_config(0, 4096, 3));
        let metrics = check(&case, &build, &probe, config, 192 * 1024).await?;
        assert!(
            metric(&metrics, "spill_chunked_runs").unwrap_or(0) > 0,
            "{}: not joined in chunks",
            label(&case)
        );
        replays += metric(&metrics, "spill_replay_passes").unwrap_or(0);
    }
    assert!(replays > 0, "no partition needed a second chunk");
    Ok(())
}

/// NULL as an ordinary key (`NullEqualsNull`) that fills its partition is
/// joined in chunks; under `NullEqualsNothing` the NULL build rows never
/// reach a key partition.
#[tokio::test]
async fn null_key_that_fills_a_partition() -> Result<()> {
    let build = skewed("b", 30000, 60, true, 3001, 500);
    let probe = sparse_probe(3000, true);
    for join_type in JOIN_TYPES {
        for null_equality in [
            NullEquality::NullEqualsNull,
            NullEquality::NullEqualsNothing,
        ] {
            let case = Case {
                join_type,
                null_equality,
                filter: true,
                shape: Shape::CollectLeft(2),
            };
            // 192 KiB: two streams' byte-admitted chunks in 64 KiB can reach
            // the replay bound (see hot_key_build_chunks_keep_every_join_type).
            let config = no_reversal(spill_config(0, 4096, 3));
            let metrics = check(&case, &build, &probe, config, 192 * 1024).await?;
            let chunked = metric(&metrics, "spill_chunked_runs").unwrap_or(0);
            if null_equality == NullEquality::NullEqualsNull {
                assert!(chunked > 0, "{}: not joined in chunks", label(&case));
            }
        }
    }
    Ok(())
}

/// More chunks than `hash_join_spill_max_replay_passes` fail with
/// `ResourcesExhausted` naming the bound, and leave no reservation.
#[tokio::test]
async fn replay_bound_fails_with_its_error() -> Result<()> {
    let build = skewed("b", 30000, 70, false, 3001, 500);
    let probe = sparse_probe(3000, false);
    for join_type in [JoinType::Inner, JoinType::Right, JoinType::LeftAnti] {
        let case = Case {
            join_type,
            null_equality: NullEquality::NullEqualsNothing,
            filter: false,
            shape: Shape::CollectLeft(1),
        };
        let mut config = no_reversal(spill_config(0, 4096, 3));
        config
            .options_mut()
            .execution
            .hash_join_spill_max_replay_passes = 1;
        let plan = join_plan(&case, &build, &probe)?;
        let context = task_context(config, Some(64 * 1024), true);
        let error = run(&plan, Arc::clone(&context)).await.unwrap_err();
        let text = error.to_string();
        assert!(
            matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{text}"
        );
        assert!(
            text.contains("hash_join_spill_max_replay_passes (1)"),
            "{text}"
        );
        drop(plan);
        assert_eq!(context.memory_pool().reserved(), 0, "{join_type}");
    }
    Ok(())
}

/// A partition is joined in one chunk when the pool holds it: the chunk
/// join is not entered, and the run takes the ordinary sub-join.
#[tokio::test]
async fn hot_partition_that_fits_takes_one_join() -> Result<()> {
    let build = skewed("b", 30000, 70, false, 3001, 500);
    let probe = sparse_probe(3000, false);
    for join_type in JOIN_TYPES {
        let case = Case {
            join_type,
            null_equality: NullEquality::NullEqualsNothing,
            filter: false,
            shape: Shape::CollectLeft(1),
        };
        let metrics =
            check(&case, &build, &probe, spill_config(1, 4096, 3), usize::MAX).await?;
        assert_eq!(
            metric(&metrics, "spill_chunked_runs").unwrap_or(0),
            0,
            "{join_type}"
        );
    }
    Ok(())
}

/// Role reversal: a written partition whose probe rows are fewer than its
/// build rows is built over the probe file. Every join type, both null
/// equalities and a filter, with one probe stream (`CollectLeft(1)` and
/// `Partitioned`), and the build-row pass of `CollectLeft(3)`.
#[tokio::test]
async fn smaller_probe_files_reverse_roles() -> Result<()> {
    let build = table("b", 12000, 4001, 0, 500);
    let probe = table("p", 1500, 4001, 3, 250);
    for case in all_cases(&[
        Shape::CollectLeft(1),
        Shape::CollectLeft(3),
        Shape::Partitioned(2),
    ]) {
        let metrics =
            check(&case, &build, &probe, spill_config(1, 4096, 3), usize::MAX).await?;
        let reversed = metric(&metrics, "spill_reversed_runs").unwrap_or(0);
        let expect = match case.shape {
            // Own runs of a build with three members are not reversed (each
            // stream would read the build file); only the build-row pass is.
            Shape::CollectLeft(3) => {
                crate::joins::utils::need_produce_result_in_final(case.join_type)
            }
            _ => true,
        };
        if expect {
            assert!(reversed > 0, "{}: no run was reversed", label(&case));
        }
    }
    Ok(())
}

/// A reversed join that its pool cannot hold either (below the depth limit,
/// with a hot key on both sides) falls back to the build-chunk join in the
/// partition's own orientation. Existence joins keep the output small.
#[tokio::test]
async fn reversed_join_that_cannot_hold_falls_back_to_chunks() -> Result<()> {
    let build = skewed("b", 24000, 70, false, 3001, 500);
    // Key 7 in a third of the probe rows: smaller than the build side of
    // its partition, larger than the pool.
    let probe: Vec<RecordBatch> = skewed("p", 12000, 33, false, 3001, 250);
    for join_type in [
        JoinType::LeftSemi,
        JoinType::LeftAnti,
        JoinType::RightSemi,
        JoinType::RightAnti,
        JoinType::LeftMark,
        JoinType::RightMark,
    ] {
        for shape in [Shape::CollectLeft(1), Shape::Partitioned(2)] {
            let case = Case {
                join_type,
                null_equality: NullEquality::NullEqualsNothing,
                filter: true,
                shape,
            };
            let metrics =
                check(&case, &build, &probe, spill_config(0, 4096, 0), 64 * 1024).await?;
            assert!(
                metric(&metrics, "spill_reversed_runs").unwrap_or(0) > 0,
                "{}: not reversed",
                label(&case)
            );
            assert!(
                metric(&metrics, "spill_chunked_runs").unwrap_or(0) > 0,
                "{}: not joined in chunks",
                label(&case)
            );
        }
    }
    Ok(())
}

/// Reversal under memory pressure: the reversed join of a partition spills
/// itself, and still equals the unbounded join.
#[tokio::test]
async fn reversed_runs_spill_under_pressure() -> Result<()> {
    let build = table("b", 20000, 9001, 0, 500);
    let probe = table("p", 4000, 9001, 3, 250);
    for case in all_cases(&[Shape::CollectLeft(1), Shape::Partitioned(2)]) {
        // leto-mjj.98 v2: 4 KiB workspace (was 1 KiB): one written row, decoded
        // with its copy and table, needs about 1.35 KiB (review MJJ98-REVIEW3,
        // finding 2), more than three quarters of 1 KiB.
        let metrics =
            check(&case, &build, &probe, spill_config(0, 4096, 3), 40 * 1024).await?;
        assert!(
            metric(&metrics, "spilled_partitions").unwrap_or(0) > 0,
            "{}",
            label(&case)
        );
    }
    Ok(())
}

/// Every pair of levels: the rows of one partition at level `n` spread
/// evenly over the partitions of level `n + 1`, so a written partition that
/// a hot key does not fill can always be divided again.
#[test]
fn every_level_divides_the_level_above() {
    let shift = 60;
    let values: ArrayRef = Arc::new(Int64Array::from_iter_values(0..400_000));
    let mut hashes = vec![0; values.len()];
    create_hashes(
        [&values],
        crate::joins::hash_join::exec::HASH_JOIN_SEED.random_state(),
        &mut hashes,
    )
    .unwrap();
    for level in 0..LEVEL_MUL.len() - 1 {
        let rows: Vec<u64> = hashes
            .iter()
            .copied()
            .filter(|hash| partition_of(*hash, level, shift) == 3)
            .collect();
        let mut next = [0usize; 16];
        for hash in &rows {
            next[partition_of(*hash, level + 1, shift)] += 1;
        }
        let mean = rows.len() / 16;
        for count in next {
            assert!(count.abs_diff(mean) < mean / 4, "level {level}: {next:?}");
        }
    }
}

/// Verification of the configured pool limit while chunk output is live
/// (review MJJ98-REVIEW-20261119, finding 1).
#[tokio::test]
async fn review_chunk_respects_pool_limit() -> Result<()> {
    let build = skewed("b", 30000, 100, false, 3001, 500);
    let probe = sparse_probe(3000, false);
    let case = Case {
        join_type: JoinType::Right,
        null_equality: NullEquality::NullEqualsNothing,
        filter: false,
        shape: Shape::CollectLeft(1),
    };
    let limit = 64 * 1024;
    let config = no_reversal(spill_config(0, 4096, 3));
    let plan = join_plan(&case, &build, &probe)?;
    let context = task_context(config, Some(limit), true);
    let mut stream = plan.execute(0, Arc::clone(&context))?;
    while let Some(batch) = stream.next().await {
        batch?;
        if metric(&plan.metrics().unwrap(), "spill_chunked_runs").unwrap_or(0) > 0 {
            let reserved = context.memory_pool().reserved();
            drop(stream);
            drop(plan);
            assert_eq!(context.memory_pool().reserved(), 0);
            // Design decision 8: at most (streams + 1) x workspace above the
            // limit; one stream here. (Review MJJ98: 380,672 bytes before.)
            let bound = limit + 2 * 4096;
            println!("chunk reserves {reserved} bytes, bound {bound}");
            assert!(
                reserved <= bound,
                "chunk reserves {reserved} bytes in a {limit}-byte pool"
            );
            return Ok(());
        }
    }
    panic!("test did not reach a live chunk");
}

/// (Review MJJ98-REVIEW-20261119.) Dropping one stream during replay must
/// leave the other stream's own
/// probe rows and shared build file intact until it has finished.
#[tokio::test]
async fn review_drop_during_chunk_replay() -> Result<()> {
    let build = skewed("b", 30000, 100, false, 3001, 500);
    let probe = sparse_probe(6000, false);
    let case = Case {
        join_type: JoinType::Right,
        null_equality: NullEquality::NullEqualsNothing,
        filter: true,
        shape: Shape::CollectLeft(2),
    };
    let config = no_reversal(spill_config(0, 4096, 3)).with_batch_size(64);
    let context = task_context(config, Some(64 * 1024), true);
    let plan = join_plan(&case, &build, &probe)?;
    let mut first = plan.execute(0, Arc::clone(&context))?;
    let second = plan.execute(1, Arc::clone(&context))?;
    while let Some(batch) = first.next().await {
        batch?;
        if metric(&plan.metrics().unwrap(), "spill_replay_passes").unwrap_or(0) > 0 {
            break;
        }
    }
    assert!(metric(&plan.metrics().unwrap(), "spill_replay_passes").unwrap_or(0) > 0);
    drop(first);
    let actual = collect(second).await?;
    let remaining: Vec<_> = probe.iter().skip(1).step_by(2).cloned().collect();
    assert_eq!(
        batches_to_sort_string(&actual),
        expected(&case, &build, &remaining).await,
    );
    drop(plan);
    assert_eq!(context.memory_pool().reserved(), 0);
    assert_eq!(context.runtime_env().disk_manager.used_disk_space(), 0);
    Ok(())
}

/// Fixed-width casts evaluate under the same build grant as a plain column.
#[tokio::test]
async fn fixed_width_cast_stays_within_the_bound() -> Result<()> {
    use datafusion_physical_expr::expressions::CastExpr;
    let build = skewed("b", 20000, 70, false, 3001, 500);
    let probe = sparse_probe(3000, false);
    let widen = |schema: &SchemaRef| -> PhysicalExprRef {
        Arc::new(CastExpr::new(key(schema), DataType::Int64, None))
    };
    for join_type in [JoinType::Inner, JoinType::Right, JoinType::LeftAnti] {
        let (limit, workspace) = (64 * 1024, 4096);
        let pool = review::PeakPool::new(limit);
        let dyn_pool: Arc<dyn datafusion_execution::memory_pool::MemoryPool> =
            Arc::clone(&pool) as _;
        let context =
            review::pool_context(no_reversal(spill_config(0, workspace, 3)), dyn_pool);
        let plan: Arc<dyn ExecutionPlan> = Arc::new(HashJoinExec::try_new(
            memory_input(&build, 1),
            memory_input(&probe, 1),
            vec![(widen(&build[0].schema()), widen(&probe[0].schema()))],
            None,
            &join_type,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?);
        let result = run(&plan, Arc::clone(&context)).await;
        let peak = pool.peak.load(std::sync::atomic::Ordering::SeqCst);
        println!("{join_type}: peak {peak}, ok {}", result.is_ok());
        result?;
        assert!(
            metric(&plan.metrics().unwrap(), "spill_chunked_runs").unwrap_or(0) > 0,
            "{join_type}: not joined in chunks"
        );
        drop(plan);
        assert!(
            peak <= limit + 2 * workspace,
            "{join_type}: peak {peak} above {limit} + 2 x {workspace}"
        );
        assert_eq!(context.memory_pool().reserved(), 0, "{join_type}");
    }
    Ok(())
}

/// Single-probe Left and Full preserve both sides' unmatched rows after
/// the reversed build is refused and the original build runs in chunks.
#[rstest::rstest]
#[case::left(JoinType::Left)]
#[case::full(JoinType::Full)]
#[tokio::test]
async fn round4_single_probe_outer_reverses_then_chunks(
    #[case] join_type: JoinType,
) -> Result<()> {
    // One written partition, with both orientations too large for the pool.
    // Half the probe payloads are disjoint from the build even for this key.
    let build_rows = 6000;
    let probe_rows = 2400;
    let build = skewed("b", build_rows, 100, false, 3001, 256);
    let probe = skewed("p", probe_rows, 100, false, 3001, 256)
        .into_iter()
        .map(|batch| {
            let payload = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let payload =
                Int32Array::from_iter_values(payload.values().iter().map(|&row| {
                    if row % 2 == 0 {
                        row
                    } else {
                        row + build_rows as i32
                    }
                }));
            RecordBatch::try_new(
                batch.schema(),
                vec![Arc::clone(batch.column(0)), Arc::new(payload)],
            )
        })
        .collect::<arrow::error::Result<Vec<_>>>()?;
    let plan = || -> Result<Arc<dyn ExecutionPlan>> {
        let left = build[0].schema();
        let right = probe[0].schema();
        let filter = JoinFilter::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("build", 0)),
                Operator::Eq,
                Arc::new(Column::new("probe", 1)),
            )),
            vec![
                ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 1,
                    side: JoinSide::Right,
                },
            ],
            Arc::new(Schema::new(vec![
                left.field(1).clone(),
                right.field(1).clone(),
            ])),
        );
        Ok(Arc::new(HashJoinExec::try_new(
            memory_input(&build, 1),
            memory_input(&probe, 1),
            vec![(key(&left), key(&right))],
            Some(filter),
            &join_type,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    };
    let expected = run(&plan()?, task_context(SessionConfig::new(), None, true)).await?;
    let spilled = plan()?;
    let context = task_context(spill_config(0, 4096, 0), Some(32 * 1024), true);
    let actual = run(&spilled, Arc::clone(&context)).await?;
    let metrics = spilled.metrics().unwrap();
    assert_eq!(metric(&metrics, "spilled_partitions"), Some(1));
    let unmatched_probe = if join_type == JoinType::Full {
        probe_rows / 2
    } else {
        0
    };
    assert_eq!(
        actual.iter().map(RecordBatch::num_rows).sum::<usize>(),
        build_rows + unmatched_probe,
    );
    assert!(
        metric(&metrics, "spill_reversed_runs").unwrap_or_default() > 0,
        "{join_type}"
    );
    assert!(
        metric(&metrics, "spill_chunked_runs").unwrap_or_default() > 0,
        "{join_type}"
    );
    assert_eq!(
        batches_to_sort_string(&actual),
        batches_to_sort_string(&expected),
        "{join_type}"
    );
    assert_eq!(context.memory_pool().reserved(), 0);
    Ok(())
}
