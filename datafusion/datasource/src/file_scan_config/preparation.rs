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

//! Session-scoped expression preparation during native scan polling.

use std::pin::Pin;
use std::task::{Context, Poll};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{Result, config::ConfigOptions};
use datafusion_execution::{RecordBatchStream, SendableRecordBatchStream};
use datafusion_physical_expr::simplifier::with_preparation_cache;
use futures::Stream;

const OPTION: &str = "longbow.execution.expr_prep_cache";

pub(super) fn session_stream(
    inner: SendableRecordBatchStream,
    config: &ConfigOptions,
) -> SendableRecordBatchStream {
    // Resolve once per scan stream. No per-file environment reads, options
    // enumeration or shared atomics are added to the simplifier's hot path.
    let enabled = config
        .entries()
        .into_iter()
        .find(|entry| entry.key == OPTION)
        .and_then(|entry| entry.value)
        .map(|value| value == "true")
        .unwrap_or(true);
    Box::pin(PreparationStream { inner, enabled })
}

struct PreparationStream {
    inner: SendableRecordBatchStream,
    enabled: bool,
}

impl Stream for PreparationStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let enabled = self.enabled;
        with_preparation_cache(enabled, || self.inner.as_mut().poll_next(cx))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl RecordBatchStream for PreparationStream {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
    use datafusion_physical_expr::simplifier::preparation_cache_enabled;
    use datafusion_physical_plan::{ExecutionPlan, stream::RecordBatchStreamAdapter};
    use futures::stream;
    use std::any::Any;
    use std::sync::Arc;

    #[derive(Clone, Debug)]
    struct Setting(bool);
    impl ConfigExtension for Setting {
        const PREFIX: &'static str = "longbow";
    }
    impl ExtensionOptions for Setting {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn cloned(&self) -> Box<dyn ExtensionOptions> {
            Box::new(self.clone())
        }
        fn set(&mut self, _: &str, value: &str) -> Result<()> {
            self.0 = value.parse().unwrap();
            Ok(())
        }
        fn entries(&self) -> Vec<ConfigEntry> {
            vec![ConfigEntry {
                key: OPTION.into(),
                value: Some(self.0.to_string()),
                description: "test",
            }]
        }
    }

    fn probe(enabled: Option<bool>) -> SendableRecordBatchStream {
        let mut config = ConfigOptions::default();
        if let Some(enabled) = enabled {
            config.extensions.insert(Setting(enabled));
        }
        let mut first = true;
        let inner = stream::poll_fn(move |_| {
            assert_eq!(preparation_cache_enabled(), enabled.unwrap_or(true));
            if std::mem::take(&mut first) {
                Poll::Pending
            } else {
                Poll::Ready(None)
            }
        });
        session_stream(
            Box::pin(RecordBatchStreamAdapter::new(
                Arc::new(arrow::datatypes::Schema::empty()),
                inner,
            )),
            &config,
        )
    }

    #[test]
    fn preparation_sessions_restore_scope_on_pending_and_thread_migration() {
        let mut disabled = probe(Some(false));
        let mut enabled = probe(Some(true));
        let mut default = probe(None);
        let mut cx = Context::from_waker(futures::task::noop_waker_ref());
        assert!(disabled.as_mut().poll_next(&mut cx).is_pending());
        assert!(preparation_cache_enabled());
        assert!(enabled.as_mut().poll_next(&mut cx).is_pending());
        assert!(default.as_mut().poll_next(&mut cx).is_pending());
        assert!(enabled.as_mut().poll_next(&mut cx).is_ready());
        assert!(preparation_cache_enabled());
        std::thread::spawn(move || {
            let mut cx = Context::from_waker(futures::task::noop_waker_ref());
            assert!(disabled.as_mut().poll_next(&mut cx).is_ready());
            assert!(preparation_cache_enabled());
        })
        .join()
        .unwrap();
        assert!(preparation_cache_enabled());
    }

    struct Opener {
        enabled: bool,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::file_stream::FileOpener for Opener {
        fn open(&self, _: crate::PartitionedFile) -> Result<crate::file_stream::FileOpenFuture> {
            use futures::{FutureExt, StreamExt};
            assert_eq!(preparation_cache_enabled(), self.enabled);
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let enabled = self.enabled;
            let calls = Arc::clone(&self.calls);
            Ok(async move {
                let mut pending = true;
                futures::future::poll_fn(|cx| {
                    assert_eq!(preparation_cache_enabled(), enabled);
                    calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if std::mem::take(&mut pending) {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    } else {
                        Poll::Ready(())
                    }
                })
                .await;
                let output = stream::poll_fn(move |_| {
                    assert_eq!(preparation_cache_enabled(), enabled);
                    calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Poll::Ready(None)
                });
                Ok(output.boxed())
            }
            .boxed())
        }
    }

    #[tokio::test]
    async fn preparation_file_scan_applies_the_shipped_session_setting() {
        use crate::file_scan_config::FileScanConfigBuilder;
        use crate::source::DataSourceExec;
        use crate::test_util::MockSource;
        use datafusion_execution::{
            TaskContext, config::SessionConfig, object_store::ObjectStoreUrl,
        };
        use futures::StreamExt;
        use std::sync::atomic::{AtomicUsize, Ordering};
        for enabled in [false, true] {
            let calls = Arc::new(AtomicUsize::new(0));
            let source = MockSource::default().with_file_opener(Arc::new(Opener {
                enabled,
                calls: Arc::clone(&calls),
            }));
            let scan =
                FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), Arc::new(source))
                    .with_file_group(crate::file_groups::FileGroup::new(vec![
                        crate::PartitionedFile::new("probe", 1),
                    ]))
                    .build();
            let mut config = SessionConfig::new();
            config.options_mut().extensions.insert(Setting(enabled));
            let context = Arc::new(TaskContext::default().with_session_config(config));
            let mut output = DataSourceExec::from_data_source(scan)
                .execute(0, context)
                .unwrap();
            assert!(output.next().await.is_none());
            assert!(calls.load(Ordering::Relaxed) >= 4);
            assert!(preparation_cache_enabled());
        }
    }
}
