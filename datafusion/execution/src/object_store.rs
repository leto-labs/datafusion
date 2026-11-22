// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! ObjectStoreRegistry holds all the object stores at Runtime with a scheme for each store.
//! This allows the user to extend DataFusion with different storage systems such as S3 or HDFS
//! and query data inside these systems.

use dashmap::DashMap;
use datafusion_common::{
    DataFusionError, Result, exec_err, internal_datafusion_err, not_impl_err,
};
use object_store::ObjectStore;
#[cfg(not(target_arch = "wasm32"))]
use object_store::local::LocalFileSystem;
use std::sync::Arc;
use url::Url;

/// Coalescing gap in bytes for local reads with object generation validators.
pub const LOCAL_RANGE_COALESCE_GAP: u64 = 128 * 1024;

/// A parsed URL identifying a particular [`ObjectStore`] instance
///
/// For example:
/// * `file://` for local file system
/// * `s3://bucket` for AWS S3 bucket
/// * `oss://bucket` for Aliyun OSS bucket
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectStoreUrl {
    url: Url,
}

impl ObjectStoreUrl {
    /// Parse an [`ObjectStoreUrl`] from a string
    ///
    /// # Example
    /// ```
    /// # use url::Url;
    /// # use datafusion_execution::object_store::ObjectStoreUrl;
    /// let object_store_url = ObjectStoreUrl::parse("s3://bucket").unwrap();
    /// assert_eq!(object_store_url.as_str(), "s3://bucket/");
    /// // can also access the underlying `Url`
    /// let url: &Url = object_store_url.as_ref();
    /// assert_eq!(url.scheme(), "s3");
    /// assert_eq!(url.host_str(), Some("bucket"));
    /// assert_eq!(url.path(), "/");
    /// ```
    pub fn parse(s: impl AsRef<str>) -> Result<Self> {
        let mut parsed =
            Url::parse(s.as_ref()).map_err(|e| DataFusionError::External(Box::new(e)))?;

        let remaining = &parsed[url::Position::BeforePath..];
        if !remaining.is_empty() && remaining != "/" {
            return exec_err!(
                "ObjectStoreUrl must only contain scheme and authority, got: {remaining}"
            );
        }

        // Always set path for consistency
        parsed.set_path("/");
        Ok(Self { url: parsed })
    }

    /// An [`ObjectStoreUrl`] for the local filesystem (`file://`)
    ///
    /// # Example
    /// ```
    /// # use datafusion_execution::object_store::ObjectStoreUrl;
    /// let local_fs = ObjectStoreUrl::parse("file://").unwrap();
    /// assert_eq!(local_fs, ObjectStoreUrl::local_filesystem())
    /// ```
    pub fn local_filesystem() -> Self {
        Self::parse("file://").unwrap()
    }

    /// Returns this [`ObjectStoreUrl`] as a string
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

impl AsRef<str> for ObjectStoreUrl {
    fn as_ref(&self) -> &str {
        self.url.as_ref()
    }
}

impl AsRef<Url> for ObjectStoreUrl {
    fn as_ref(&self) -> &Url {
        &self.url
    }
}

impl std::fmt::Display for ObjectStoreUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

/// [`ObjectStoreRegistry`] maps a URL to an [`ObjectStore`] instance,
/// and allows DataFusion to read from different [`ObjectStore`]
/// instances. For example DataFusion might be configured so that
///
/// 1. `s3://my_bucket/lineitem/` mapped to the `/lineitem` path on an
///    AWS S3 object store bound to `my_bucket`
///
/// 2. `s3://my_other_bucket/lineitem/` mapped to the (same)
///    `/lineitem` path on a *different* AWS S3 object store bound to
///    `my_other_bucket`
///
/// When given a [`ListingTableUrl`], DataFusion tries to find an
/// appropriate [`ObjectStore`]. For example
///
/// ```sql
/// create external table unicorns stored as parquet location 's3://my_bucket/lineitem/';
/// ```
///
/// In this particular case, the url `s3://my_bucket/lineitem/` will be provided to
/// [`ObjectStoreRegistry::get_store`] and one of three things will happen:
///
/// - If an [`ObjectStore`] has been registered with [`ObjectStoreRegistry::register_store`] with
///   `s3://my_bucket`, that [`ObjectStore`] will be returned
///
/// - If an AWS S3 object store can be ad-hoc discovered by the url `s3://my_bucket/lineitem/`, this
///   object store will be registered with key `s3://my_bucket` and returned.
///
/// - Otherwise an error will be returned, indicating that no suitable [`ObjectStore`] could
///   be found
///
/// This allows for two different use-cases:
///
/// 1. Systems where object store buckets are explicitly created using DDL, can register these
///    buckets using [`ObjectStoreRegistry::register_store`]
///
/// 2. Systems relying on ad-hoc discovery, without corresponding DDL, can create [`ObjectStore`]
///    lazily by providing a custom implementation of [`ObjectStoreRegistry`]
///
/// <!-- is in a different crate so normal rustdoc links don't work -->
/// [`ListingTableUrl`]: https://docs.rs/datafusion/latest/datafusion/datasource/listing/struct.ListingTableUrl.html
/// [`ObjectStore`]: object_store::ObjectStore
pub trait ObjectStoreRegistry: Send + Sync + std::fmt::Debug + 'static {
    /// If a store with the same key existed before, it is replaced and returned
    fn register_store(
        &self,
        url: &Url,
        store: Arc<dyn ObjectStore>,
    ) -> Option<Arc<dyn ObjectStore>>;

    /// Deregister the store previously registered with the same key. Returns the
    /// deregistered store if it existed.
    #[expect(unused_variables)]
    fn deregister_store(&self, url: &Url) -> Result<Arc<dyn ObjectStore>> {
        not_impl_err!(
            "ObjectStoreRegistry::deregister_store is not implemented for this ObjectStoreRegistry"
        )
    }

    /// Get a suitable store for the provided URL. For example:
    ///
    /// - URL with scheme `file:///` or no scheme will return the default LocalFS store
    /// - URL with scheme `s3://bucket/` will return the S3 store
    /// - URL with scheme `hdfs://hostname:port/` will return the hdfs store
    ///
    /// If no [`ObjectStore`] found for the `url`, ad-hoc discovery may be executed depending on
    /// the `url` and [`ObjectStoreRegistry`] implementation. An [`ObjectStore`] may be lazily
    /// created and registered.
    fn get_store(&self, url: &Url) -> Result<Arc<dyn ObjectStore>>;

    /// [`Self::get_store`] together with the [`StoreId`] of the registration
    /// it resolved, from one lookup, for caches keyed by store (Longbow
    /// family OO). A registry that wraps the registered store per call
    /// returns the wrapper with the registration's ID. `None` (the default)
    /// means the registry assigns no IDs, and such caches are skipped.
    fn get_store_with_id(
        &self,
        url: &Url,
    ) -> Result<(Arc<dyn ObjectStore>, Option<StoreId>)> {
        Ok((self.get_store(url)?, None))
    }

    /// Resolve the store, its cache identity and its conditional range-read
    /// gap together. Registries that wrap a resolved store forward the gap
    /// along with its ID (Longbow family RCG).
    fn get_store_registration(&self, url: &Url) -> Result<ObjectStoreRegistration> {
        let (store, id) = self.get_store_with_id(url)?;
        Ok(ObjectStoreRegistration {
            store,
            id,
            range_coalesce_gap: object_store::OBJECT_STORE_COALESCE_DEFAULT,
        })
    }
}

/// The identity of one store registration (Longbow family OO): assigned from
/// a process-wide counter when a store is registered, never reused, and new
/// when another store is registered under the same URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StoreId(u64);

impl StoreId {
    /// A new, never used ID; `None` once the counter is exhausted, so that
    /// an ID is never issued twice (caches keyed by it are then skipped).
    pub fn next() -> Option<Self> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self::next_from(&NEXT)
    }

    /// The next ID of `counter`, with a checked add: the counter never wraps.
    fn next_from(counter: &std::sync::atomic::AtomicU64) -> Option<Self> {
        use std::sync::atomic::Ordering::Relaxed;
        counter
            .try_update(Relaxed, Relaxed, |n| n.checked_add(1))
            .ok()
            .map(Self)
    }

    /// The ID's number.
    pub fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for StoreId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "store-{}", self.0)
    }
}

/// The store and read policy resolved by one registry lookup.
#[derive(Clone)]
pub struct ObjectStoreRegistration {
    /// The store to read from.
    pub store: Arc<dyn ObjectStore>,
    /// The identity of the registration, if assigned.
    pub id: Option<StoreId>,
    /// The maximum gap in bytes to merge between conditional range reads.
    pub range_coalesce_gap: u64,
}

/// The default [`ObjectStoreRegistry`]
pub struct DefaultObjectStoreRegistry {
    /// A map from scheme to object store that serve list / read operations for the store
    /// Each with the ID and range policy of its registration.
    object_stores: DashMap<String, ObjectStoreRegistration>,
}

impl std::fmt::Debug for DefaultObjectStoreRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("DefaultObjectStoreRegistry")
            .field(
                "schemes",
                &self
                    .object_stores
                    .iter()
                    .map(|o| o.key().clone())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Default for DefaultObjectStoreRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultObjectStoreRegistry {
    /// This will register [`LocalFileSystem`] to handle `file://` paths
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Self {
        let registry = Self {
            object_stores: DashMap::new(),
        };
        drop(registry.register_store_with_coalesce_gap(
            ObjectStoreUrl::local_filesystem().as_ref(),
            Arc::new(LocalFileSystem::new()),
            LOCAL_RANGE_COALESCE_GAP,
        ));
        registry
    }

    /// Default without any backend registered.
    #[cfg(target_arch = "wasm32")]
    pub fn new() -> Self {
        Self {
            object_stores: DashMap::new(),
        }
    }

    /// Register a store and its conditional range-read gap in bytes.
    /// Returns the previous store for this URL, if any.
    pub fn register_store_with_coalesce_gap(
        &self,
        url: &Url,
        store: Arc<dyn ObjectStore>,
        range_coalesce_gap: u64,
    ) -> Option<Arc<dyn ObjectStore>> {
        self.object_stores
            .insert(
                get_url_key(url),
                ObjectStoreRegistration {
                    store,
                    id: StoreId::next(),
                    range_coalesce_gap,
                },
            )
            .map(|old| old.store)
    }
}

///
/// Stores are registered based on the scheme, host and port of the provided URL
/// with a [`LocalFileSystem::new`] automatically registered for `file://` (if the
/// target arch is not `wasm32`).
///
/// For example:
///
/// - `file:///my_path` will return the default LocalFS store
/// - `s3://bucket/path` will return a store registered with `s3://bucket` if any
/// - `hdfs://host:port/path` will return a store registered with `hdfs://host:port` if any
impl ObjectStoreRegistry for DefaultObjectStoreRegistry {
    fn register_store(
        &self,
        url: &Url,
        store: Arc<dyn ObjectStore>,
    ) -> Option<Arc<dyn ObjectStore>> {
        self.register_store_with_coalesce_gap(
            url,
            store,
            object_store::OBJECT_STORE_COALESCE_DEFAULT,
        )
    }

    fn deregister_store(&self, url: &Url) -> Result<Arc<dyn ObjectStore>> {
        let s = get_url_key(url);
        let (_, registration) = self.object_stores
            .remove(&s)
            .ok_or_else(|| {
                internal_datafusion_err!("Failed to deregister object store. No suitable object store found for {url}. See `RuntimeEnv::register_object_store`")
            })?;

        Ok(registration.store)
    }

    fn get_store(&self, url: &Url) -> Result<Arc<dyn ObjectStore>> {
        Ok(self.get_store_registration(url)?.store)
    }

    fn get_store_with_id(
        &self,
        url: &Url,
    ) -> Result<(Arc<dyn ObjectStore>, Option<StoreId>)> {
        let registration = self.get_store_registration(url)?;
        Ok((registration.store, registration.id))
    }

    fn get_store_registration(&self, url: &Url) -> Result<ObjectStoreRegistration> {
        self.object_stores
            .get(&get_url_key(url))
            .map(|entry| entry.value().clone())
            .ok_or_else(|| {
                internal_datafusion_err!("No suitable object store found for {url}. See `RuntimeEnv::register_object_store`")
            })
    }
}

/// Get the key of a url for object store registration.
/// The credential info will be removed
fn get_url_key(url: &Url) -> String {
    format!(
        "{}://{}",
        url.scheme(),
        &url[url::Position::BeforeHost..url::Position::AfterPort],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_resolves_the_store_and_its_gap_together() {
        let registry = DefaultObjectStoreRegistry::new();
        let url = Url::parse("memory://ranges/").unwrap();
        let first: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let second: Arc<dyn ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        registry.register_store_with_coalesce_gap(
            &url,
            Arc::clone(&first),
            LOCAL_RANGE_COALESCE_GAP,
        );
        let before = registry.get_store_registration(&url).unwrap();
        registry.register_store(&url, Arc::clone(&second));
        let after = registry.get_store_registration(&url).unwrap();
        assert!(Arc::ptr_eq(&before.store, &first));
        assert_eq!(before.range_coalesce_gap, LOCAL_RANGE_COALESCE_GAP);
        assert!(Arc::ptr_eq(&after.store, &second));
        assert_eq!(
            after.range_coalesce_gap,
            object_store::OBJECT_STORE_COALESCE_DEFAULT
        );
        assert_ne!(before.id, after.id);
    }

    /// Longbow family OO (MJJ96 v5 review): the counter never wraps, so an
    /// exhausted counter issues no ID instead of reusing one.
    #[test]
    fn store_ids_are_never_reissued_at_exhaustion() {
        let counter = std::sync::atomic::AtomicU64::new(u64::MAX - 1);
        assert_eq!(
            StoreId::next_from(&counter).map(StoreId::get),
            Some(u64::MAX - 1)
        );
        assert_eq!(StoreId::next_from(&counter), None);
        assert_eq!(StoreId::next_from(&counter), None);
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), u64::MAX);
        // Registrations get distinct IDs.
        let registry = DefaultObjectStoreRegistry::new();
        let url = Url::parse("memory://ids").unwrap();
        registry.register_store(&url, Arc::new(object_store::memory::InMemory::new()));
        let first = registry.get_store_with_id(&url).unwrap().1.unwrap();
        registry.register_store(&url, Arc::new(object_store::memory::InMemory::new()));
        assert!(registry.get_store_with_id(&url).unwrap().1.unwrap() > first);
    }

    #[test]
    fn test_object_store_url() {
        let file = ObjectStoreUrl::parse("file://").unwrap();
        assert_eq!(file.as_str(), "file:///");

        let url = ObjectStoreUrl::parse("s3://bucket").unwrap();
        assert_eq!(url.as_str(), "s3://bucket/");

        let url = ObjectStoreUrl::parse("s3://username:password@host:123").unwrap();
        assert_eq!(url.as_str(), "s3://username:password@host:123/");

        let err = ObjectStoreUrl::parse("s3://bucket:invalid").unwrap_err();
        assert_eq!(err.strip_backtrace(), "External error: invalid port number");

        let err = ObjectStoreUrl::parse("s3://bucket?").unwrap_err();
        assert_eq!(
            err.strip_backtrace(),
            "Execution error: ObjectStoreUrl must only contain scheme and authority, got: ?"
        );

        let err = ObjectStoreUrl::parse("s3://bucket?foo=bar").unwrap_err();
        assert_eq!(
            err.strip_backtrace(),
            "Execution error: ObjectStoreUrl must only contain scheme and authority, got: ?foo=bar"
        );

        let err = ObjectStoreUrl::parse("s3://host:123/foo").unwrap_err();
        assert_eq!(
            err.strip_backtrace(),
            "Execution error: ObjectStoreUrl must only contain scheme and authority, got: /foo"
        );

        let err =
            ObjectStoreUrl::parse("s3://username:password@host:123/foo").unwrap_err();
        assert_eq!(
            err.strip_backtrace(),
            "Execution error: ObjectStoreUrl must only contain scheme and authority, got: /foo"
        );
    }

    #[test]
    fn test_get_url_key() {
        let file = ObjectStoreUrl::parse("file://").unwrap();
        let key = get_url_key(&file.url);
        assert_eq!(key.as_str(), "file://");

        let url = ObjectStoreUrl::parse("s3://bucket").unwrap();
        let key = get_url_key(&url.url);
        assert_eq!(key.as_str(), "s3://bucket");

        let url = ObjectStoreUrl::parse("s3://username:password@host:123").unwrap();
        let key = get_url_key(&url.url);
        assert_eq!(key.as_str(), "s3://host:123");
    }
}
