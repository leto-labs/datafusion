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

// Modified by the Longbow authors; see third_party/README.md.

use super::super::WindowAggStream;
use super::*;
use crate::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use crate::stream::RecordBatchStreamAdapter;
use crate::windows::create_window_expr;
use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_expr::{WindowFrame, WindowFunctionDefinition};
use datafusion_functions_aggregate::min_max::max_udaf;
use futures::{FutureExt, StreamExt, stream};

fn batch() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4]))],
    )
    .unwrap()
}
fn setup(
    limit: usize,
) -> (
    Arc<dyn MemoryPool>,
    TaskContext,
    Vec<Arc<dyn WindowExpr>>,
    SchemaRef,
) {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()
        .unwrap();
    let context = TaskContext::default().with_runtime(runtime);
    let input = batch().schema();
    let expr = create_window_expr(
        &WindowFunctionDefinition::AggregateUDF(max_udaf()),
        "m".into(),
        &[crate::expressions::col("k", &input).unwrap()],
        &[],
        &[],
        Arc::new(WindowFrame::new(None)),
        Arc::clone(&input),
        false,
        false,
        None,
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        input.field(0).clone(),
        expr.field().unwrap().as_ref().clone(),
    ]));
    (pool, context, vec![expr], schema)
}

#[test]
fn longbow_window_memory_accounting_unique_slices_are_charged_once() -> Result<()> {
    let (pool, context, expr, schema) = setup(1 << 20);
    let mut memory = WindowBufferMemory::try_new(&schema, &expr, &context)?.unwrap();
    let input = batch();
    memory.retain(&input.slice(0, 2))?;
    let first = pool.reserved();
    memory.retain(&input.slice(2, 2))?;
    assert_eq!(
        pool.reserved() - first,
        memory.metadata_per_batch + 2 * memory.work_per_row
    );
    drop(memory);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn longbow_window_memory_accounting_pool_refusal_releases_and_fuses() -> Result<()>
{
    let (pool, context, expr, schema) = setup(8000);
    let memory = WindowBufferMemory::try_new(&schema, &expr, &context)?;
    let input = Box::pin(RecordBatchStreamAdapter::new(
        batch().schema(),
        stream::iter((0..100).map(|_| Ok(batch()))),
    ));
    let mut window = WindowAggStream::new(
        schema,
        expr,
        input,
        BaselineMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        vec![],
        vec![],
    )?
    .with_buffer_memory(memory);
    let error = window.next().await.unwrap().unwrap_err();
    assert!(error.to_string().contains("Resources exhausted"), "{error}");
    assert_eq!(pool.reserved(), 0);
    assert!(window.next().await.is_none());
    Ok(())
}

#[test]
fn longbow_window_memory_accounting_cancel_pending_releases_input_and_reservation()
-> Result<()> {
    let (pool, context, expr, schema) = setup(1 << 20);
    let memory = WindowBufferMemory::try_new(&schema, &expr, &context)?;
    use std::sync::atomic::{AtomicBool, Ordering};
    struct OnDrop(Arc<AtomicBool>);
    impl Drop for OnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = OnDrop(Arc::clone(&dropped));
    let input = Box::pin(RecordBatchStreamAdapter::new(
        batch().schema(),
        stream::iter(vec![Ok(batch())])
            .chain(stream::pending())
            .map(move |item| {
                std::hint::black_box(&guard);
                item
            }),
    ));
    let mut window = WindowAggStream::new(
        schema,
        expr,
        input,
        BaselineMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        vec![],
        vec![],
    )?
    .with_buffer_memory(memory);
    assert!(window.next().now_or_never().is_none());
    assert!(pool.reserved() > 0);
    assert!(!window.batches.is_empty());
    drop(window);
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn longbow_window_memory_accounting_output_stays_charged_until_stream_drop()
-> Result<()> {
    let (pool, context, expr, schema) = setup(1 << 20);
    let memory = WindowBufferMemory::try_new(&schema, &expr, &context)?;
    let input = Box::pin(RecordBatchStreamAdapter::new(
        batch().schema(),
        stream::iter(vec![Ok(batch())]),
    ));
    let mut window = WindowAggStream::new(
        schema,
        expr,
        input,
        BaselineMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        vec![],
        vec![],
    )?
    .with_buffer_memory(memory);
    let output = window.next().await.unwrap()?;
    assert_eq!(output.num_rows(), 4);
    assert!(pool.reserved() > 0);
    assert!(window.next().await.is_none());
    drop(window);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn longbow_window_memory_accounting_execution_owns_the_charge() -> Result<()> {
    use super::super::WindowAggExec;
    use crate::{ExecutionPlan, test::TestMemoryExec};
    let (pool, context, expr, _) = setup(1 << 20);
    let source = Arc::new(TestMemoryExec::try_new(
        &[vec![batch()]],
        batch().schema(),
        None,
    )?);
    let window = WindowAggExec::try_new(expr, source, true)?;
    let mut stream = window.execute(0, Arc::new(context))?;
    assert!(pool.reserved() > 0, "execute must retain its reservation");
    assert_eq!(stream.next().await.unwrap()?.num_rows(), 4);
    assert!(pool.reserved() > 0);
    drop(stream);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[test]
fn longbow_window_memory_accounting_execution_refuses_before_input() -> Result<()> {
    use super::super::WindowAggExec;
    use crate::{ExecutionPlan, test::TestMemoryExec};
    let (pool, context, expr, _) = setup(4000);
    let source = Arc::new(TestMemoryExec::try_new(
        &[vec![batch()]],
        batch().schema(),
        None,
    )?);
    let window = WindowAggExec::try_new(expr, source, true)?;
    let error = match window.execute(0, Arc::new(context)) {
        Ok(_) => panic!("window must admit its buffer reservation before execution"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("Resources exhausted"), "{error}");
    assert_eq!(pool.reserved(), 0);
    Ok(())
}
