use crate::config::StorageConfig;
use crate::error::{MemMachineError, Result};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::str::FromStr;

use super::episode::EpisodeStore;
use super::schema;
use super::semantic::SemanticStore;
use super::session::SessionStore;

#[derive(Clone)]
pub struct StorageManager {
    pool: SqlitePool,
    config: StorageConfig,
}

impl StorageManager {
    pub async fn new(config: StorageConfig) -> Result<Self> {
        let db_path = config.path.to_string_lossy();
        let db_url = format!("sqlite:{}?mode=rwc", db_path);

        let options = SqliteConnectOptions::from_str(&db_url)
            .map_err(|e| MemMachineError::Configuration(e.to_string()))?
            .create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(config.max_connections)
            .connect_with(options)
            .await?;

        if config.wal_mode {
            sqlx::query("PRAGMA journal_mode=WAL")
                .execute(&pool)
                .await?;
        }

        sqlx::query("PRAGMA foreign_keys=ON")
            .execute(&pool)
            .await?;

        Ok(Self { pool, config })
    }

    pub async fn initialize(&self) -> Result<()> {
        schema::initialize_schema(&self.pool).await
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub fn config(&self) -> &StorageConfig {
        &self.config
    }

    pub fn episode_store(&self) -> EpisodeStore {
        EpisodeStore::new(self.pool.clone())
    }

    pub fn session_store(&self) -> SessionStore {
        SessionStore::new(self.pool.clone())
    }

    pub fn semantic_store(&self, embedding_dimension: usize) -> SemanticStore {
        SemanticStore::new(self.pool.clone(), embedding_dimension)
    }

    pub async fn close(&self) {
        self.pool.close().await;
    }
}

impl std::fmt::Debug for StorageManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageManager")
            .field("config", &self.config)
            .finish()
    }
}
