// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with the
// License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;
use arrow::array::{ArrayRef, UInt32Array};
use std::sync::Arc;

#[test]
fn range_limit_and_finish_match_slices() -> Result<()> {
    let batch = RecordBatch::try_from_iter(vec![(
        "x",
        Arc::new(UInt32Array::from_iter_values(0..100)) as ArrayRef,
    )])?;
    for fetch in [None, Some(0), Some(3), Some(8), Some(19), Some(1000)] {
        let mut old = LimitedBatchCoalescer::new(batch.schema(), 7, fetch);
        let mut new = LimitedBatchCoalescer::new(batch.schema(), 7, fetch);
        for (start, len) in [(3, 0), (7, 2), (23, 5), (0, 32), (61, 3)] {
            assert_eq!(
                old.push_batch(batch.slice(start, len))?,
                new.push_batch_range(&batch, start, len)?
            );
        }
        old.finish()?;
        new.finish()?;
        let expected =
            std::iter::from_fn(|| old.next_completed_batch()).collect::<Vec<_>>();
        let actual =
            std::iter::from_fn(|| new.next_completed_batch()).collect::<Vec<_>>();
        assert_eq!(expected, actual);
        assert!(new.push_batch_range(&batch, 0, 1).is_err());
    }
    Ok(())
}
