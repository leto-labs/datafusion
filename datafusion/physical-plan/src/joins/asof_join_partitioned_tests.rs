// Added by the Longbow authors (family AP); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0

//! Longbow family AP: the partitioned mode of [`AsOfJoinExec`] and the tie
//! rule of both modes.

use super::*;
use crate::collect;
use crate::test::TestMemoryExec;
use arrow::array::{Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field};
use datafusion_common::test_util::batches_to_sort_string;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use insta::assert_snapshot;

fn schema(value: &str) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, true),
        Field::new("ts", DataType::Int64, true),
        Field::new(value, DataType::Int32, false),
    ]))
}

fn batch(
    schema: &SchemaRef,
    keys: Vec<Option<&str>>,
    times: Vec<Option<i64>>,
    values: Vec<i32>,
) -> RecordBatch {
    RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(StringArray::from(keys)),
            Arc::new(Int64Array::from(times)),
            Arc::new(Int32Array::from(values)),
        ],
    )
    .unwrap()
}

/// A join of `left` and `right` partitions on `key` with `ts op ts`,
/// projecting the left columns and the right value.
fn join(
    left: Vec<Vec<RecordBatch>>,
    right: Vec<Vec<RecordBatch>>,
    op: Operator,
    mode: AsOfJoinMode,
) -> Result<Arc<AsOfJoinExec>> {
    let left = TestMemoryExec::try_new_exec(&left, schema("id"), None)?;
    let right = TestMemoryExec::try_new_exec(&right, schema("price"), None)?;
    Ok(Arc::new(
        AsOfJoinExec::try_new(
            left,
            right,
            vec![(
                Arc::new(PhysicalColumn::new("key", 0)),
                Arc::new(PhysicalColumn::new("key", 0)),
            )],
            AsOfMatchExpr::new(
                Arc::new(PhysicalColumn::new("ts", 1)),
                op,
                Arc::new(PhysicalColumn::new("ts", 1)),
            ),
            Some(vec![0, 1, 2, 5]),
        )?
        .with_mode(mode)?,
    ))
}

fn context(batch_size: usize) -> Arc<TaskContext> {
    Arc::new(
        TaskContext::default()
            .with_session_config(SessionConfig::new().with_batch_size(batch_size)),
    )
}

/// Left partitions {NULL, A} and {B, C}; right partitions hold the same keys.
fn inputs() -> (Vec<Vec<RecordBatch>>, Vec<Vec<RecordBatch>>) {
    let (l, r) = (schema("id"), schema("price"));
    let left = vec![
        vec![
            batch(&l, vec![None], vec![Some(3)], vec![0]),
            batch(
                &l,
                vec![Some("A"), Some("A")],
                vec![None, Some(1)],
                vec![1, 2],
            ),
            batch(
                &l,
                vec![Some("A"), Some("A")],
                vec![Some(4), Some(7)],
                vec![3, 4],
            ),
        ],
        vec![batch(
            &l,
            vec![Some("B"), Some("C")],
            vec![Some(2), Some(3)],
            vec![5, 6],
        )],
    ];
    let right = vec![
        vec![
            batch(
                &r,
                vec![None, Some("A")],
                vec![Some(2), None],
                vec![999, 777],
            ),
            batch(&r, vec![Some("A")], vec![Some(2)], vec![20]),
            RecordBatch::new_empty(Arc::clone(&r)),
            batch(
                &r,
                vec![Some("A"), Some("A")],
                vec![Some(4), Some(6)],
                vec![40, 60],
            ),
        ],
        vec![batch(&r, vec![Some("B")], vec![Some(1)], vec![101])],
    ];
    (left, right)
}

#[tokio::test]
async fn partitioned_mode_merges_partition_i_with_partition_i() -> Result<()> {
    let (left, right) = inputs();
    let exec = join(left, right, Operator::GtEq, AsOfJoinMode::Partitioned)?;
    assert_eq!(exec.properties().output_partitioning().partition_count(), 2);
    let batches = collect(Arc::clone(&exec) as _, context(2)).await?;
    assert_snapshot!(batches_to_sort_string(&batches), @r"
    +-----+----+----+-------+
    | key | ts | id | price |
    +-----+----+----+-------+
    |     | 3  | 0  |       |
    | A   |    | 1  |       |
    | A   | 1  | 2  |       |
    | A   | 4  | 3  | 40    |
    | A   | 7  | 4  | 60    |
    | B   | 2  | 5  | 101   |
    | C   | 3  | 6  |       |
    +-----+----+----+-------+
    ");
    // No right input is collected.
    let metrics = exec.metrics().expect("ASOF metrics must be present");
    assert_eq!(
        metrics.sum_by_name("build_mem_used").map(|v| v.as_usize()),
        Some(0)
    );
    Ok(())
}

#[test]
fn partitioned_mode_requires_co_partitioned_keys() -> Result<()> {
    let (left, right) = inputs();
    let exec = join(left, right, Operator::GtEq, AsOfJoinMode::Partitioned)?;
    let requirements = exec.required_input_distribution();
    assert!(matches!(&requirements[0], Distribution::KeyPartitioned(k) if k.len() == 1));
    assert!(matches!(&requirements[1], Distribution::KeyPartitioned(k) if k.len() == 1));
    assert!(exec.to_string_for_test().contains("mode=Partitioned"));

    let broadcast = join(
        inputs().0,
        vec![inputs().1.concat()],
        Operator::GtEq,
        AsOfJoinMode::Broadcast,
    )?;
    assert!(matches!(
        broadcast.required_input_distribution()[1],
        Distribution::SinglePartition
    ));
    assert!(!broadcast.to_string_for_test().contains("mode="));

    // A join without equality keys cannot be partitioned.
    let unkeyed = AsOfJoinExec::try_new(
        TestMemoryExec::try_new_exec(&[vec![]], schema("id"), None)?,
        TestMemoryExec::try_new_exec(&[vec![]], schema("price"), None)?,
        vec![],
        AsOfMatchExpr::new(
            Arc::new(PhysicalColumn::new("ts", 1)),
            Operator::GtEq,
            Arc::new(PhysicalColumn::new("ts", 1)),
        ),
        None,
    )?;
    let error = unkeyed.with_mode(AsOfJoinMode::Partitioned).unwrap_err();
    assert!(
        error.to_string().contains("requires an equality key"),
        "{error}"
    );
    Ok(())
}

trait DisplayForTest {
    fn to_string_for_test(&self) -> String;
}
impl DisplayForTest for AsOfJoinExec {
    fn to_string_for_test(&self) -> String {
        crate::displayable(self).one_line().to_string()
    }
}

/// The right input is 100 times the memory pool. The broadcast mode collects
/// it and fails; the partitioned mode streams it.
#[tokio::test]
async fn partitioned_mode_streams_a_right_input_larger_than_the_pool() -> Result<()> {
    let (l, r) = (schema("id"), schema("price"));
    let rows = 64 * 1024;
    let right: Vec<RecordBatch> = (0..rows / 1024)
        .map(|chunk| {
            let start = chunk * 1024;
            batch(
                &r,
                vec![Some("A"); 1024],
                (start..start + 1024).map(|t| Some(t as i64)).collect(),
                (start as i32..start as i32 + 1024).collect(),
            )
        })
        .collect();
    let mut counter = RecordBatchMemoryCounter::new();
    let right_size: usize = right.iter().map(|b| counter.count_batch(b)).sum();
    let left = vec![batch(
        &l,
        vec![Some("A"), Some("A")],
        vec![Some(10), Some(rows as i64 + 5)],
        vec![0, 1],
    )];
    let pool = right_size / 100;
    let context = || {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(pool, 1.0)
            .build_arc()
            .unwrap();
        Arc::new(TaskContext::default().with_runtime(runtime))
    };

    let broadcast = join(
        vec![left.clone()],
        vec![right.clone()],
        Operator::GtEq,
        AsOfJoinMode::Broadcast,
    )?;
    let error = collect(broadcast as _, context()).await.unwrap_err();
    assert!(error.to_string().contains("Resources exhausted"), "{error}");

    let partitioned = join(
        vec![left],
        vec![right],
        Operator::GtEq,
        AsOfJoinMode::Partitioned,
    )?;
    let batches = collect(partitioned as _, context()).await?;
    let prices: Vec<_> = batches
        .iter()
        .flat_map(|b| {
            b.column(3)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
        })
        .collect();
    assert_eq!(prices, vec![Some(10), Some(rows as i32 - 1)]);
    Ok(())
}

/// Among right rows with equal keys and equal match value, the match is the
/// last in the right input's order, across batch boundaries, in both modes
/// and for `>=` and `>`; output batches of 1 or 3 rows do not change it.
#[tokio::test]
async fn ties_take_the_last_right_row_in_input_order() -> Result<()> {
    let (l, r) = (schema("id"), schema("price"));
    let left = vec![batch(
        &l,
        vec![Some("A"), Some("A"), Some("A")],
        vec![Some(5), Some(6), Some(9)],
        vec![0, 1, 2],
    )];
    let right = vec![
        batch(
            &r,
            vec![Some("A"), Some("A")],
            vec![Some(5), Some(5)],
            vec![1, 2],
        ),
        batch(&r, vec![Some("A")], vec![Some(5)], vec![3]),
        batch(
            &r,
            vec![Some("A"), Some("A")],
            vec![Some(8), Some(8)],
            vec![4, 5],
        ),
    ];
    for mode in [AsOfJoinMode::Broadcast, AsOfJoinMode::Partitioned] {
        for batch_size in [1, 3] {
            for (op, expected) in [
                (Operator::GtEq, vec![Some(3), Some(3), Some(5)]),
                (Operator::Gt, vec![None, Some(3), Some(5)]),
            ] {
                let exec = join(vec![left.clone()], vec![right.clone()], op, mode)?;
                let batches = collect(exec as _, context(batch_size)).await?;
                let prices: Vec<_> = batches
                    .iter()
                    .flat_map(|b| {
                        b.column(3)
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap()
                            .iter()
                    })
                    .collect();
                assert_eq!(prices, expected, "{mode:?} {op} batch {batch_size}");
            }
        }
    }
    Ok(())
}

/// An equality key with a floating-point leaf is rejected like a float key:
/// the sort order and the comparator disagree on signed zero.
#[test]
fn equality_keys_with_floating_leaves_are_rejected() -> Result<()> {
    use datafusion_physical_expr::expressions::CastExpr;
    let (left, right) = inputs();
    let exec = join(left, right, Operator::GtEq, AsOfJoinMode::Broadcast)?;
    let dictionary =
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Float64));
    let key = |index| -> Arc<
        dyn datafusion_physical_expr_common::physical_expr::PhysicalExpr,
    > {
        Arc::new(CastExpr::new(
            Arc::new(PhysicalColumn::new("ts", index)),
            dictionary.clone(),
            None,
        ))
    };
    let error = AsOfJoinExec::try_new(
        Arc::clone(exec.left()),
        Arc::clone(exec.right()),
        vec![(key(1), key(1))],
        exec.match_condition().clone(),
        None,
    )
    .expect_err("a dictionary of floats must be rejected");
    assert!(
        error
            .to_string()
            .contains("do not support floating-point type"),
        "{error}"
    );
    Ok(())
}

/// One match in each of many wide right batches: the stream flushes before
/// its pending rows pin more than `MAX_PENDING_SOURCES` batches of a side.
#[tokio::test]
async fn sparse_matches_flush_before_pinning_many_input_batches() -> Result<()> {
    let (l, r) = (schema("id"), schema("price"));
    let keys = 200;
    let width = 1024;
    let right: Vec<RecordBatch> = (0..keys)
        .map(|k| {
            let key = format!("k{k:04}");
            batch(
                &r,
                vec![Some(key.as_str()); width],
                (0..width as i64).map(Some).collect(),
                vec![k as i32; width],
            )
        })
        .collect();
    let left_keys: Vec<String> = (0..keys).map(|k| format!("k{k:04}")).collect();
    let left = vec![batch(
        &l,
        left_keys.iter().map(|k| Some(k.as_str())).collect(),
        vec![Some(0); keys],
        (0..keys as i32).collect(),
    )];
    for mode in [AsOfJoinMode::Broadcast, AsOfJoinMode::Partitioned] {
        let exec = join(
            vec![left.clone()],
            vec![right.clone()],
            Operator::GtEq,
            mode,
        )?;
        let batches = collect(exec as _, context(8192)).await?;
        assert!(
            batches.iter().all(|b| b.num_rows() <= MAX_PENDING_SOURCES),
            "{mode:?}: {:?}",
            batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>()
        );
        let prices: Vec<_> = batches
            .iter()
            .flat_map(|b| {
                b.column(3)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .iter()
            })
            .collect();
        assert_eq!(
            prices,
            (0..keys as i32).map(Some).collect::<Vec<_>>(),
            "{mode:?}"
        );
    }
    Ok(())
}
