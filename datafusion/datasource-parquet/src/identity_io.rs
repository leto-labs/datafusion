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
    GetOptions, GetRange, OBJECT_STORE_COALESCE_DEFAULT, ObjectMeta, ObjectStore,
    ObjectStoreExt, coalesce_ranges,
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
/// Close ranges are coalesced as in [`ObjectStore::get_ranges`].
#[expect(
    clippy::result_large_err,
    reason = "object_store's own error type; every caller converts it at once"
)]
pub(crate) async fn get_ranges(
    store: &dyn ObjectStore,
    meta: &ObjectMeta,
    ranges: &[Range<u64>],
) -> object_store::Result<Vec<Bytes>> {
    if !has_identity(meta) {
        return store.get_ranges(&meta.location, ranges).await;
    }
    coalesce_ranges(
        ranges,
        |range| get_range(store, meta, range),
        OBJECT_STORE_COALESCE_DEFAULT,
    )
    .await
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{Error, ObjectStoreExt, PutPayload};

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
        assert_eq!(
            get_ranges(store.as_ref(), &planned, &[0..2, 4..6])
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
            get_ranges(store.as_ref(), &planned, &[0..2, 4..6])
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
