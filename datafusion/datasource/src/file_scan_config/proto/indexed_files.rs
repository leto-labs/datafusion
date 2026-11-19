// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use datafusion_common::{HashMap, Result, Statistics, plan_datafusion_err, plan_err};
use datafusion_proto_models::protobuf;

use crate::PartitionedFile;
use crate::file_groups::FileGroup;

pub(super) fn encode(
    groups: &[FileGroup],
    statistics: &Statistics,
) -> Result<protobuf::IndexedScanFiles> {
    let mut by_path = HashMap::new();
    // The linked indices hold rare metadata variants at the same object path.
    // Output indices follow first occurrence, independent of map iteration.
    let mut descriptors: Vec<(&PartitionedFile, Option<usize>)> = Vec::new();
    let mut files = Vec::new();
    let mut encoded_groups = Vec::with_capacity(groups.len());
    for group in groups {
        let mut ranges = Vec::with_capacity(group.len());
        for file in group.iter() {
            let head = by_path
                .entry(file.object_meta.location.as_ref())
                .or_insert(None);
            let mut current = *head;
            let index = loop {
                match current {
                    Some(index) => {
                        let (candidate, next) = descriptors[index];
                        if same_file(candidate, file) {
                            break index;
                        }
                        current = next;
                    }
                    None => {
                        let index = files.len();
                        let mut descriptor = protobuf::PartitionedFile::try_from(file)?;
                        descriptor.range = None;
                        files.push(descriptor);
                        descriptors.push((file, *head));
                        *head = Some(index);
                        break index;
                    }
                }
            };
            ranges.push(protobuf::ScanFileRange {
                file_index: u32::try_from(index).map_err(|_| {
                    plan_datafusion_err!("Too many distinct files in scan")
                })?,
                range: file.range.as_ref().map(TryInto::try_into).transpose()?,
            });
        }
        encoded_groups.push(protobuf::ScanRangeGroup { files: ranges });
    }
    Ok(protobuf::IndexedScanFiles {
        files,
        groups: encoded_groups,
        scan_statistics: Some(statistics.into()),
    })
}

fn same_file(a: &PartitionedFile, b: &PartitionedFile) -> bool {
    a.object_meta == b.object_meta
        && a.partition_values == b.partition_values
        && same_shared_value(&a.statistics, &b.statistics)
        && same_shared_value(&a.arrow_schema, &b.arrow_schema)
}

fn same_shared_value<T: PartialEq>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b) || a == b,
        (None, None) => true,
        _ => false,
    }
}

pub(super) fn decode(
    indexed: &protobuf::IndexedScanFiles,
) -> Result<(Vec<FileGroup>, Statistics)> {
    let statistics = indexed
        .scan_statistics
        .as_ref()
        .ok_or_else(|| {
            plan_datafusion_err!("Indexed file scan is missing scan_statistics")
        })?
        .try_into()?;
    let files = indexed
        .files
        .iter()
        .map(|file| {
            if file.range.is_some() {
                return plan_err!("Indexed file metadata includes a range");
            }
            PartitionedFile::try_from(file)
        })
        .collect::<Result<Vec<_>>>()?;
    let groups = indexed
        .groups
        .iter()
        .map(|group| {
            let files = group
                .files
                .iter()
                .map(|reference| {
                    let index = reference.file_index;
                    let mut file =
                        files.get(index as usize).cloned().ok_or_else(|| {
                            plan_datafusion_err!(
                                "Invalid indexed scan file reference {index}"
                            )
                        })?;
                    file.range = reference
                        .range
                        .as_ref()
                        .map(TryInto::try_into)
                        .transpose()?;
                    Ok(file)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(FileGroup::new(files))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((groups, statistics))
}
