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

//! File positions resolved after metadata is available, without a second footer read.
use datafusion_common::{Result, exec_err};
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use parquet::file::metadata::RowGroupMetaData;

use crate::ParquetAccessPlan;

/// Sorted, unique zero-based file row positions to retrieve.
///
/// Attach to a `PartitionedFile` as an extension. The opener validates positions
/// against its metadata and intersects their selections with normal file-range
/// ownership. Other parquet access extensions cannot be combined with this one.
#[derive(Debug, Clone, PartialEq)]
pub struct ParquetRowPositions {
    positions: Vec<usize>,
}

impl ParquetRowPositions {
    /// Positions are validated when file metadata is available to the opener.
    pub fn new(positions: Vec<usize>) -> Self {
        Self { positions }
    }

    pub(crate) fn access_plan(
        &self,
        groups: &[RowGroupMetaData],
    ) -> Result<ParquetAccessPlan> {
        if self.positions.windows(2).any(|rows| rows[0] >= rows[1]) {
            return exec_err!("Parquet row positions must be sorted and unique");
        }
        let mut total = 0usize;
        for group in groups {
            let Ok(rows) = usize::try_from(group.num_rows()) else {
                return exec_err!("invalid Parquet row-group count");
            };
            let Some(sum) = total.checked_add(rows) else {
                return exec_err!("Parquet file row count overflow");
            };
            total = sum;
        }
        if self.positions.last().is_some_and(|row| *row >= total) {
            return exec_err!("Parquet row position exceeds the file's {total} rows");
        }
        let mut selectors =
            Vec::with_capacity(self.positions.len().saturating_mul(2).saturating_add(1));
        let mut previous = 0;
        for &position in &self.positions {
            if position > previous {
                selectors.push(RowSelector::skip(position - previous));
            }
            selectors.push(RowSelector::select(1));
            previous = position + 1;
        }
        if previous < total {
            selectors.push(RowSelector::skip(total - previous));
        }
        ParquetAccessPlan::try_new_from_overall_row_selection(
            RowSelection::from(selectors),
            groups,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RowGroupAccess;
    use parquet::file::metadata::{ColumnChunkMetaData, RowGroupMetaData};
    use parquet::schema::{parser::parse_message_type, types::SchemaDescriptor};
    use std::sync::Arc;

    fn groups() -> Vec<RowGroupMetaData> {
        let schema = Arc::new(SchemaDescriptor::new(Arc::new(
            parse_message_type("message test { REQUIRED INT64 id; }").unwrap(),
        )));
        [10, 20, 30]
            .into_iter()
            .map(|rows| {
                let column = ColumnChunkMetaData::builder(schema.column(0))
                    .set_num_values(rows)
                    .build()
                    .unwrap();
                RowGroupMetaData::builder(Arc::clone(&schema))
                    .set_num_rows(rows)
                    .set_column_metadata(vec![column])
                    .build()
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn row_positions_cross_group_boundaries_and_skip_other_groups() {
        let actual = ParquetRowPositions::new(vec![0, 9, 30, 59])
            .access_plan(&groups())
            .unwrap();
        let expected = ParquetAccessPlan::new(vec![
            RowGroupAccess::Selection(
                vec![
                    RowSelector::select(1),
                    RowSelector::skip(8),
                    RowSelector::select(1),
                ]
                .into(),
            ),
            RowGroupAccess::Skip,
            RowGroupAccess::Selection(
                vec![
                    RowSelector::select(1),
                    RowSelector::skip(28),
                    RowSelector::select(1),
                ]
                .into(),
            ),
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn row_positions_empty_and_invalid_inputs() {
        assert_eq!(
            ParquetRowPositions::new(vec![])
                .access_plan(&groups())
                .unwrap(),
            ParquetAccessPlan::new_none(3)
        );
        for rows in [vec![60], vec![1, 1], vec![2, 1], vec![usize::MAX]] {
            assert!(
                ParquetRowPositions::new(rows)
                    .access_plan(&groups())
                    .is_err()
            );
        }
        assert!(ParquetRowPositions::new(vec![0]).access_plan(&[]).is_err());
        assert_eq!(
            ParquetRowPositions::new(vec![]).access_plan(&[]).unwrap(),
            ParquetAccessPlan::new_none(0)
        );
    }
}
