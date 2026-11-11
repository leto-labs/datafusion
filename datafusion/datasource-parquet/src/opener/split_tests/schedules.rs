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

use super::*;
use futures::{FutureExt, StreamExt, TryStreamExt};

async fn execution(
    predicate: Option<Arc<dyn PhysicalExpr>>,
) -> (Arc<dyn ExecutionPlan>, Arc<TaskContext>) {
    let (store, size) = store().await;
    let mut source = ParquetSource::new(schema());
    if let Some(predicate) = predicate {
        source = source.with_predicate(predicate);
    }
    let config = FileScanConfigBuilder::new(
        ObjectStoreUrl::parse("memory://").unwrap(),
        Arc::new(source),
    )
    .with_file_groups(vec![
        FileGroup::new(vec![PartitionedFile::new("k.parquet", size)]),
        FileGroup::new(vec![]),
        FileGroup::new(vec![]),
    ])
    .build();
    let exec: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(config);
    let context = Arc::new(TaskContext::default());
    context.runtime_env().register_object_store(
        ObjectStoreUrl::parse("memory://").unwrap().as_ref(),
        store,
    );
    (exec, context)
}

fn keys(batches: impl IntoIterator<Item = RecordBatch>) -> Vec<i32> {
    let mut keys: Vec<i32> = batches
        .into_iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    keys.sort_unstable();
    keys
}

#[tokio::test]
async fn completing_dynamic_filter_mid_scan_prunes_then_donates() {
    let filter = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::new(Column::new("k", 0))],
        key_between(0, 2400),
    ));
    let (exec, context) =
        execution(Some(Arc::clone(&filter) as Arc<dyn PhysicalExpr>)).await;
    let mut owner = exec.execute(0, Arc::clone(&context)).unwrap();
    let first = owner.next().await.unwrap().unwrap();
    let mut idle: Vec<_> = (1..3)
        .map(|p| exec.execute(p, Arc::clone(&context)).unwrap())
        .collect();
    for stream in &mut idle {
        assert!(stream.next().now_or_never().is_none());
    }
    let second = owner.next().await.unwrap().unwrap();
    assert_eq!(
        count(&exec.metrics().unwrap(), "row_group_units_published"),
        0
    );
    filter.update(key_between(500, 2400)).unwrap();
    filter.mark_complete();
    let rest: Vec<Vec<RecordBatch>> = futures::future::try_join_all(
        std::iter::once(owner)
            .chain(idle)
            .map(|stream| stream.try_collect()),
    )
    .await
    .unwrap();
    let mut all = vec![first, second];
    all.extend(rest.into_iter().flatten());
    assert_eq!(keys(all), (0..200).chain(500..2400).collect::<Vec<_>>());
    assert!(count(&exec.metrics().unwrap(), "row_group_units_published") > 0);
    assert!(count(&exec.metrics().unwrap(), "row_groups_pruned_dynamic_filter") > 0);
}

#[tokio::test]
async fn a_receiver_can_donate_its_unread_tail_again() {
    let (exec, context) = execution(None).await;
    let mut owner = exec.execute(0, Arc::clone(&context)).unwrap();
    let first = owner.next().await.unwrap().unwrap();
    let mut receiver = exec.execute(1, Arc::clone(&context)).unwrap();
    assert!(receiver.next().now_or_never().is_none());
    let second = owner.next().await.unwrap().unwrap();
    assert_eq!(
        count(&exec.metrics().unwrap(), "row_group_units_published"),
        1
    );
    let received = receiver.next().await.unwrap().unwrap();
    let mut last = exec.execute(2, Arc::clone(&context)).unwrap();
    assert!(last.next().now_or_never().is_none());
    // Leave the original donor unpolled. Only the receiver can publish now.
    let donated = receiver.next().await.unwrap().unwrap();
    assert_eq!(
        count(&exec.metrics().unwrap(), "row_group_units_published"),
        2
    );
    let rest: Vec<Vec<RecordBatch>> = futures::future::try_join_all(
        [owner, receiver, last]
            .into_iter()
            .map(|stream| stream.try_collect()),
    )
    .await
    .unwrap();
    let mut all = vec![first, second, received, donated];
    all.extend(rest.into_iter().flatten());
    assert_eq!(keys(all), (0..2400).collect::<Vec<_>>());
}
