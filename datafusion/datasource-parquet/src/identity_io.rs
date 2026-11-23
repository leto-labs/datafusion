// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow family D: byte reads tied to the planned object generation.
//!
//! When an [`ObjectMeta`] carries an ETag or a version (from the planner's
//! listing, possibly through the plan codec), every footer, page-index and
//! data read asks the store for exactly that generation
//! ([`GetOptions::if_match`], [`GetOptions::version`]). A replaced object
//! then fails the read instead of decoding new bytes with old offsets.
//! Without a validator the reads are upstream's plain range reads.

use std::ops::Range;

use bytes::Bytes;
use object_store::{
    GetOptions, GetRange, ObjectMeta, ObjectStore, ObjectStoreExt, coalesce_ranges,
};

fn identity_options(meta: &ObjectMeta, range: Range<u64>) -> GetOptions {
    GetOptions {
        if_match: meta.e_tag.clone(),
        version: meta.version.clone(),
        range: Some(GetRange::Bounded(range)),
        ..Default::default()
    }
}

fn has_identity(meta: &ObjectMeta) -> bool {
    meta.e_tag.is_some() || meta.version.is_some()
}

/// Read `range` of the object `meta` names, of the generation it names.
#[expect(
    clippy::result_large_err,
    reason = "object_store's own error type; every caller converts it at once"
)]
pub(crate) async fn get_range(
    store: &dyn ObjectStore,
    meta: &ObjectMeta,
    range: Range<u64>,
) -> object_store::Result<Bytes> {
    if !has_identity(meta) {
        return store.get_range(&meta.location, range).await;
    }
    store
        .get_opts(&meta.location, identity_options(meta, range))
        .await?
        .bytes()
        .await
}

/// Read `ranges` of the object `meta` names, of the generation it names.
/// Conditional reads merge close ranges using the registered store's gap.
#[expect(
    clippy::result_large_err,
    reason = "object_store's own error type; every caller converts it at once"
)]
pub(crate) async fn get_ranges(
    store: &dyn ObjectStore,
    meta: &ObjectMeta,
    ranges: &[Range<u64>],
    range_coalesce_gap: u64,
) -> object_store::Result<Vec<Bytes>> {
    if !has_identity(meta) {
        return store.get_ranges(&meta.location, ranges).await;
    }
    coalesce_ranges(
        ranges,
        |range| get_range(store, meta, range),
        range_coalesce_gap,
    )
    .await
}

/// Read sparse local metadata using one generation-validated descriptor and
/// one blocking read task. The descriptor remains tied to the file it opened
/// even if its path is replaced between reads.
#[cfg(unix)]
pub(crate) async fn get_local_metadata_ranges(
    store: &dyn ObjectStore,
    meta: &ObjectMeta,
    ranges: &[Range<u64>],
    range_coalesce_gap: u64,
) -> object_store::Result<Vec<Bytes>> {
    use std::os::unix::fs::FileExt;

    use object_store::{Error, GetResultPayload};

    let [first, rest @ ..] = ranges else {
        return Ok(Vec::new());
    };
    if rest.is_empty() || !has_identity(meta) {
        return get_ranges(store, meta, ranges, range_coalesce_gap).await;
    }

    let result = store
        .get_opts(&meta.location, identity_options(meta, first.clone()))
        .await?;
    let GetResultPayload::File(file, _) = result.payload else {
        // A custom file:// store may return a stream instead. Retain its
        // conditional read contract and consume the response already open.
        let (first, mut rest) = futures::try_join!(
            result.bytes(),
            get_ranges(store, meta, rest, range_coalesce_gap)
        )?;
        rest.insert(0, first);
        return Ok(rest);
    };
    let file_size = result.meta.size;
    let opened_at = std::time::SystemTime::from(result.meta.last_modified);
    let batch = ranges.to_vec();
    let (interest, reader) = tokio::sync::oneshot::channel::<()>();
    let read = async move {
        let io_error = |source| Error::Generic {
            store: "Parquet page index",
            source: Box::new(source),
        };
        let fetch = |range: Range<u64>| {
            let bytes = (|| {
                // A running blocking task cannot be aborted. Stop before
                // reading the next span if the scan dropped its future.
                if interest.is_closed() {
                    return Err(std::io::Error::from(std::io::ErrorKind::Interrupted));
                }
                let len = isize::try_from(range.end - range.start)
                    .map_err(|_| std::io::ErrorKind::InvalidData)?
                    as usize;
                if range.end > file_size {
                    return Err(std::io::Error::from(std::io::ErrorKind::InvalidData));
                }
                let mut bytes = vec![0; len];
                file.read_exact_at(&mut bytes, range.start)?;
                Ok(Bytes::from(bytes))
            })()
            .map_err(io_error);
            std::future::ready(bytes)
        };
        let bytes = coalesce_ranges(&batch, fetch, range_coalesce_gap).await?;
        let current = file.metadata().map_err(io_error)?;
        let unchanged = current.len() == file_size
            && current.modified().map_err(io_error)? == opened_at;
        Ok::<_, Error>(unchanged.then_some(bytes))
    };
    let bytes = match tokio::runtime::Handle::try_current() {
        Ok(runtime) => {
            runtime
                .spawn_blocking(move || futures::executor::block_on(read))
                .await?
        }
        Err(_) => read.await,
    }?;
    drop(reader);
    match bytes {
        Some(bytes) => Ok(bytes),
        // Discard bytes if an in-place write changed the descriptor.
        // The normal path also handles stores with synthetic timestamps.
        None => get_ranges(store, meta, ranges, range_coalesce_gap).await,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{Error, ObjectStoreExt, PutPayload};

    #[cfg(unix)]
    #[tokio::test]
    async fn local_metadata_batch_retains_order_and_rejects_replaced_file() {
        use object_store::local::LocalFileSystem;

        let dir = tempfile::tempdir().unwrap();
        let store = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
        let path = Path::from("page-index");
        let data = b"0123456789abcdef";
        store
            .put(&path, PutPayload::from_static(data))
            .await
            .unwrap();
        let planned = store.head(&path).await.unwrap();
        let ranges = [12..16, 2..5, 4..7, 2..5];
        let expected = ranges
            .iter()
            .map(|range| {
                Bytes::from_static(&data[range.start as usize..range.end as usize])
            })
            .collect::<Vec<_>>();
        assert_eq!(
            get_local_metadata_ranges(&store, &planned, &ranges, 1)
                .await
                .unwrap(),
            expected
        );
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let actual = futures::executor::block_on(get_local_metadata_ranges(
                        &store, &planned, &ranges, 1,
                    ))
                    .unwrap();
                    assert_eq!(actual, expected);
                })
                .join()
                .unwrap();
        });
        store
            .put(&path, PutPayload::from_static(b"fedcba9876543210"))
            .await
            .unwrap();
        let error = get_local_metadata_ranges(&store, &planned, &ranges, 1)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Precondition { .. }), "{error}");
    }

    #[tokio::test]
    async fn identity_reads_fail_after_a_same_size_overwrite() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("data.parquet");
        store
            .put(&path, PutPayload::from_static(b"aaaaaaaa"))
            .await
            .unwrap();
        let planned = store.head(&path).await.unwrap();
        assert!(planned.e_tag.is_some());
        let ranges = [0..2, 4..6];
        let gap = 1;
        assert_eq!(
            get_ranges(store.as_ref(), &planned, &ranges, gap)
                .await
                .unwrap(),
            vec![Bytes::from_static(b"aa"), Bytes::from_static(b"aa")]
        );
        #[cfg(unix)]
        assert_eq!(
            get_local_metadata_ranges(store.as_ref(), &planned, &ranges, gap)
                .await
                .unwrap(),
            vec![Bytes::from_static(b"aa"), Bytes::from_static(b"aa")]
        );
        // Same size, same path, new generation.
        store
            .put(&path, PutPayload::from_static(b"bbbbbbbb"))
            .await
            .unwrap();
        let stale = get_range(store.as_ref(), &planned, 0..2).await.unwrap_err();
        assert!(matches!(stale, Error::Precondition { .. }), "{stale}");
        assert!(matches!(
            get_ranges(store.as_ref(), &planned, &ranges, gap)
                .await
                .unwrap_err(),
            Error::Precondition { .. }
        ));
        // Without a validator the read follows upstream: the current bytes.
        let unvalidated = ObjectMeta {
            e_tag: None,
            version: None,
            ..planned
        };
        assert_eq!(
            get_range(store.as_ref(), &unvalidated, 0..2).await.unwrap(),
            Bytes::from_static(b"bb")
        );
    }
}
