use crate::connection::Connection;
use crate::database::Database;
use crate::error::Error;
use futures_core::future::BoxFuture;
use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

#[cfg(doc)]
use crate::pool::{Pool, PoolOptions};

/// Custom connect callback for [`Pool`].
///
/// Set with [`PoolOptions::connector()`]. The pool calls [`Self::connect()`] for every attempt
/// to open a new connection, inside its retry loop and under the
/// [acquire timeout][PoolOptions::acquire_timeout].
///
/// Use it to fetch credentials that expire (for example an IAM access token used as the
/// password), to pick a server, or to open the connection over a custom transport.
/// Because it runs when the connection is opened, it does not need a background task to keep
/// credentials fresh.
///
/// Implemented for closures with the signature
/// `Fn(PoolConnectMetadata<DB>) -> impl Future<Output = sqlx::Result<DB::Connection>>`.
///
/// [`PoolOptions::after_connect`] is still called on the connection that this returns.
///
/// This API follows the `PoolConnector` trait proposed in
/// [launchbadge/sqlx#3582](https://github.com/launchbadge/sqlx/pull/3582).
///
/// # Example: Credentials That Expire
/// This example uses Postgres but may be adapted to any driver.
///
/// ```rust,no_run
/// use sqlx::postgres::PgPoolOptions;
/// use sqlx::pool::PoolConnectMetadata;
/// use sqlx::{ConnectOptions, Postgres};
///
/// // Stands in for a call to a cloud provider's token API.
/// async fn fetch_access_token() -> Result<String, std::io::Error> {
///     Ok("token".into())
/// }
///
/// # async fn _example() -> sqlx::Result<()> {
/// let pool = PgPoolOptions::new()
///     // The type annotation on the argument is required for the trait impl to resolve.
///     .connector(|meta: PoolConnectMetadata<Postgres>| async move {
///         let token = fetch_access_token()
///             .await
///             .map_err(|e| sqlx::Error::PoolConnector {
///                 source: e.into(),
///                 retryable: true,
///             })?;
///
///         // `meta.connect_options` holds the options given to `connect_with()`
///         // or set later with `Pool::set_connect_options()`.
///         (*meta.connect_options).clone().password(&token).connect().await
///     })
///     .connect("postgres://user@localhost/db")
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// # Example: Custom Implementation
/// A custom implementation can return a `Future` that borrows from `self`.
///
/// ```rust,no_run
/// use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
/// use sqlx::pool::{PoolConnectMetadata, PoolConnector};
/// use sqlx::{ConnectOptions, PgConnection, Postgres};
///
/// struct RoundRobin {
///     // A list of servers to connect to in a high-availability configuration.
///     hosts: Vec<String>,
/// }
///
/// impl PoolConnector<Postgres> for RoundRobin {
///     async fn connect(&self, meta: PoolConnectMetadata<Postgres>) -> sqlx::Result<PgConnection> {
///         let host = &self.hosts[meta.num_attempts as usize % self.hosts.len()];
///
///         (*meta.connect_options).clone().host(host).connect().await
///     }
/// }
///
/// # async fn _example() -> sqlx::Result<()> {
/// let pool = PgPoolOptions::new()
///     .connector(RoundRobin {
///         hosts: vec!["db1.internal".into(), "db2.internal".into()],
///     })
///     .connect_with(PgConnectOptions::new().username("app"))
///     .await?;
/// # Ok(())
/// # }
/// ```
pub trait PoolConnector<DB: Database>: Send + Sync + 'static {
    /// Open a connection for the pool.
    ///
    /// Any setup that must be done on the connection should be performed before it is returned.
    ///
    /// If this method returns an error that is known to be retryable, the pool calls it again
    /// in an exponential backoff loop until the acquire timeout. Retryable errors are:
    ///
    /// * [`Error::Io`] with [`ConnectionRefused`][std::io::ErrorKind::ConnectionRefused].
    /// * [`Error::Database`] for which
    ///   [`is_transient_in_connect_phase`][crate::error::DatabaseError::is_transient_in_connect_phase]
    ///   returns `true`.
    /// * [`Error::PoolConnector`] with `retryable: true`.
    ///   This error kind is not returned internally and lets this method return
    ///   arbitrary error types.
    ///
    /// All other errors are returned to the caller immediately.
    ///
    /// Manual implementations of this method may also use the signature:
    /// ```rust,ignore
    /// async fn connect(
    ///     &self,
    ///     meta: PoolConnectMetadata<DB>,
    /// ) -> sqlx::Result<{PgConnection, MySqlConnection, SqliteConnection, etc.}>
    /// ```
    ///
    /// Note: the returned future must be `Send`.
    fn connect(
        &self,
        meta: PoolConnectMetadata<DB>,
    ) -> impl Future<Output = crate::Result<DB::Connection>> + Send + '_;
}

/// # Note: Future Changes (FIXME)
/// This could theoretically be replaced with an impl over `AsyncFn` to allow lending closures,
/// except we have no way to put the `Send` bound on the returned future.
///
/// We need Return Type Notation for that: https://github.com/rust-lang/rust/pull/138424
impl<DB, F, Fut> PoolConnector<DB> for F
where
    DB: Database,
    F: Fn(PoolConnectMetadata<DB>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = crate::Result<DB::Connection>> + Send + 'static,
{
    fn connect(
        &self,
        meta: PoolConnectMetadata<DB>,
    ) -> impl Future<Output = crate::Result<DB::Connection>> + Send + '_ {
        self(meta)
    }
}

/// Metadata passed to [`PoolConnector::connect()`] for every connection attempt.
#[non_exhaustive]
pub struct PoolConnectMetadata<DB: Database> {
    /// The instant at which the current connection task was started, including all attempts.
    ///
    /// May be used for reporting purposes, or to implement a custom backoff.
    pub start: Instant,

    /// The deadline for the current connection task, including all attempts.
    pub deadline: Instant,

    /// The number of attempts that have occurred so far.
    pub num_attempts: u32,

    /// The current size of the pool, including connections that are still being opened.
    pub pool_size: u32,

    /// The connect options of the pool.
    ///
    /// These are the options given to [`PoolOptions::connect_with()`], or the ones set later
    /// with [`Pool::set_connect_options()`].
    pub connect_options: Arc<<DB::Connection as Connection>::Options>,
}

impl<DB: Database> fmt::Debug for PoolConnectMetadata<DB> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolConnectMetadata")
            .field("start", &self.start)
            .field("deadline", &self.deadline)
            .field("num_attempts", &self.num_attempts)
            .field("pool_size", &self.pool_size)
            .field("connect_options", &self.connect_options)
            .finish()
    }
}

/// Object-safe version of [`PoolConnector`], so it can be stored in [`PoolOptions`].
pub(crate) trait DynConnector<DB: Database>: Send + Sync + 'static {
    fn connect(
        &self,
        meta: PoolConnectMetadata<DB>,
    ) -> BoxFuture<'_, crate::Result<DB::Connection>>;
}

impl<DB: Database, C: PoolConnector<DB>> DynConnector<DB> for C {
    fn connect(
        &self,
        meta: PoolConnectMetadata<DB>,
    ) -> BoxFuture<'_, crate::Result<DB::Connection>> {
        Box::pin(PoolConnector::connect(self, meta))
    }
}

pub(crate) type BoxConnector<DB> = Arc<dyn DynConnector<DB>>;

/// Returns `true` if the pool should retry after `error` from a connection attempt.
pub(crate) fn is_retryable_connect_error(error: &Error) -> bool {
    match error {
        // An IO error while connecting is assumed to be the system starting up.
        Error::Io(e) => e.kind() == std::io::ErrorKind::ConnectionRefused,
        Error::Database(e) => e.is_transient_in_connect_phase(),
        Error::PoolConnector { retryable, .. } => *retryable,
        _ => false,
    }
}
