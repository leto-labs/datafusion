//! Real native producer families must register before their consumer returns.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion_execution::{TaskContext, config::SessionConfig};
use datafusion_physical_expr::Partitioning;
use futures::{FutureExt, StreamExt};

use crate::ExecutionPlan;
use crate::attempt_tasks::AttemptTasks;
use crate::coalesce_partitions::CoalescePartitionsExec;
use crate::repartition::RepartitionExec;
use crate::test::exec::BlockingExec;

fn context(scope: &Arc<AttemptTasks>) -> Arc<TaskContext> {
    Arc::new(
        TaskContext::default()
            .with_session_config(SessionConfig::new().with_extension(Arc::clone(scope))),
    )
}

#[tokio::test]
async fn coalesce_registers_before_spawn_and_drain_releases_input() {
    let scope = AttemptTasks::new();
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let input = BlockingExec::new(schema, 2);
    let refs = input.refs();
    let plan = CoalescePartitionsExec::new(Arc::new(input));
    // No producer has to be polled for the scope to own both registrations.
    let stream = plan.execute(0, context(&scope)).unwrap();
    scope.root_joined();
    assert!(scope.drain().now_or_never().is_none());
    assert!(refs.strong_count() > 0);
    scope.stop();
    drop(stream);
    drop(plan);
    scope.drain().await.unwrap();
    assert_eq!(refs.strong_count(), 0);
}

#[tokio::test]
async fn repartition_registers_pull_and_forwarder_before_drain() {
    let scope = AttemptTasks::new();
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let input = BlockingExec::new(schema, 2);
    let refs = input.refs();
    let plan = RepartitionExec::try_new(Arc::new(input), Partitioning::RoundRobinBatch(3)).unwrap();
    let mut stream = plan.execute(0, context(&scope)).unwrap();
    // Initial polling constructs the native input pull and forwarding tasks.
    assert!(futures::poll!(stream.next()).is_pending());
    scope.root_joined();
    assert!(scope.drain().now_or_never().is_none());
    assert!(refs.strong_count() > 0);
    scope.stop();
    drop(stream);
    drop(plan);
    scope.drain().await.unwrap();
    assert_eq!(refs.strong_count(), 0);
}
