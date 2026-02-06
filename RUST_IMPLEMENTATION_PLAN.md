# MemMachine Rust Implementation Plan

## Implementation Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Project Setup | ✅ Complete | Cargo workspace, dependencies |
| 2. Core Data Types | ✅ Complete | Error, config, episode, session, semantic, filter types |
| 3. Storage Layer | ✅ Complete | StorageManager, EpisodeStore, SessionStore, SemanticStore |
| 4. Embedder Module | ✅ Complete | Embedder trait, OpenAI, Bedrock implementations |
| 5. LLM Module | ✅ Complete | Llm trait, OpenAI implementation |
| 6. Main MemMachine | ✅ Complete | Core orchestrator with all public methods |
| 7. PyO3 Bindings | ✅ Complete | Full Python bindings with async support |
| 8. Build System | ✅ Complete | pyproject.toml, maturin configuration |
| 9. Type Stubs | ✅ Complete | Python .pyi files for IDE support |
| 10. Tests | ✅ Complete | Rust and Python integration test structure |

**Next Steps:**
1. Install Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. Build: `cd src/memmachine-rs && maturin develop`
3. Run tests: `cargo test` (Rust) and `pytest tests/python` (Python)

## Overview

This document outlines the plan to rewrite the MemMachine Python class and its dependencies in Rust, exposing it to Python via PyO3.

### Key Design Decisions

1. **SQLite-only storage**: Consolidate all storage (episodes, sessions, semantic vectors, graph) into SQLite with `sqlite-vec`
2. **Drop multi-database support**: Remove ChromaDB, PostgreSQL options - simplify to SQLite only
3. **Unified configuration**: Single `SqliteConfig` instead of multiple database configs
4. **Pure Rust core**: All business logic in Rust, Python only for FastAPI layer

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    memmachine-core (Rust)                        │
├─────────────────────────────────────────────────────────────────┤
│  MemMachine                                                      │
│  ├── Configuration                                               │
│  ├── StorageManager (unified SQLite + sqlite-vec)               │
│  │   ├── EpisodeStore                                           │
│  │   ├── SessionStore                                           │
│  │   ├── SemanticStore (sqlite-vec vectors)                     │
│  │   └── VectorGraphStore (sqlite-vec + edges)                  │
│  ├── EpisodicMemoryManager                                       │
│  │   ├── ShortTermMemory                                        │
│  │   └── LongTermMemory                                         │
│  ├── SemanticSessionManager                                      │
│  ├── FilterParser (nom-based)                                   │
│  └── LLM Clients                                                │
│      ├── OpenAI (async-openai)                                  │
│      └── Cohere (reqwest)                                       │
├─────────────────────────────────────────────────────────────────┤
│  memmachine-py (PyO3 bindings)                                  │
│  ├── PyMemMachine (async wrapper)                               │
│  ├── Type conversions (Rust ↔ Python)                           │
│  └── Pydantic-compatible models                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/memmachine-rs/
├── Cargo.toml                      # Workspace root
├── crates/
│   ├── memmachine-core/            # Core Rust library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── config.rs           # Configuration types
│   │       ├── error.rs            # Error types
│   │       ├── types/              # Core data types
│   │       │   ├── mod.rs
│   │       │   ├── episode.rs
│   │       │   ├── semantic.rs
│   │       │   ├── session.rs
│   │       │   └── filter.rs
│   │       ├── storage/            # SQLite storage layer
│   │       │   ├── mod.rs
│   │       │   ├── manager.rs      # Unified storage manager
│   │       │   ├── episode.rs      # Episode storage
│   │       │   ├── session.rs      # Session storage
│   │       │   ├── semantic.rs     # Semantic vector storage (sqlite-vec)
│   │       │   └── graph.rs        # Vector graph storage
│   │       ├── memory/             # Memory management
│   │       │   ├── mod.rs
│   │       │   ├── episodic.rs     # Episodic memory manager
│   │       │   ├── short_term.rs   # Short-term memory
│   │       │   ├── long_term.rs    # Long-term memory
│   │       │   └── semantic.rs     # Semantic session manager
│   │       ├── llm/                # LLM API clients
│   │       │   ├── mod.rs
│   │       │   ├── traits.rs       # Common traits
│   │       │   ├── openai.rs       # OpenAI client
│   │       │   └── cohere.rs       # Cohere client
│   │       ├── embedder/           # Embedding providers
│   │       │   ├── mod.rs
│   │       │   ├── traits.rs
│   │       │   └── openai.rs
│   │       ├── reranker/           # Reranking providers
│   │       │   ├── mod.rs
│   │       │   ├── traits.rs
│   │       │   ├── cohere.rs
│   │       │   └── bm25.rs
│   │       └── memmachine.rs       # Main MemMachine struct
│   │
│   └── memmachine-py/              # PyO3 Python bindings
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── types.rs            # Python type conversions
│           ├── async_wrapper.rs    # Async Python ↔ Rust bridge
│           └── memmachine.rs       # PyMemMachine class
│
└── tests/                          # Integration tests
    ├── storage_tests.rs
    ├── memory_tests.rs
    └── e2e_tests.rs
```

## Phase 1: Project Setup (Week 1)

### 1.1 Create Cargo Workspace

```toml
# src/memmachine-rs/Cargo.toml
[workspace]
resolver = "2"
members = [
    "crates/memmachine-core",
    "crates/memmachine-py",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
repository = "https://github.com/MemMachine/MemMachine"

[workspace.dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }

# SQLite
sqlx = { version = "0.8", features = ["runtime-tokio", "sqlite"] }
rusqlite = { version = "0.32", features = ["bundled"] }
sqlite-vec = "0.1"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# HTTP clients
reqwest = { version = "0.12", features = ["json"] }
async-openai = "0.25"

# Error handling
thiserror = "2"
anyhow = "1"

# Parsing
nom = "7"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# Python bindings
pyo3 = { version = "0.22", features = ["extension-module"] }
pyo3-async-runtimes = { version = "0.22", features = ["tokio-runtime"] }

# Testing
tokio-test = "0.4"
```

### 1.2 Create Core Crate

```toml
# src/memmachine-rs/crates/memmachine-core/Cargo.toml
[package]
name = "memmachine-core"
version.workspace = true
edition.workspace = true

[dependencies]
tokio.workspace = true
sqlx.workspace = true
rusqlite.workspace = true
sqlite-vec.workspace = true
serde.workspace = true
serde_json.workspace = true
reqwest.workspace = true
async-openai.workspace = true
thiserror.workspace = true
anyhow.workspace = true
nom.workspace = true
tracing.workspace = true
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4", "serde"] }
```

## Phase 2: Core Data Types (Week 1-2)

### 2.1 Error Types

```rust
// src/memmachine-rs/crates/memmachine-core/src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemMachineError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Session not found: {0}")]
    SessionNotFound(String),
    
    #[error("Resource not ready: {0}")]
    ResourceNotReady(String),
    
    #[error("Storage error: {0}")]
    Storage(#[from] sqlx::Error),
    
    #[error("Embedding error: {0}")]
    Embedding(String),
    
    #[error("LLM error: {0}")]
    Llm(String),
    
    #[error("Filter parse error: {0}")]
    FilterParse(String),
    
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, MemMachineError>;
```

### 2.2 Episode Types

```rust
// src/memmachine-rs/crates/memmachine-core/src/types/episode.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub type EpisodeId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeEntry {
    pub role: EpisodeRole,
    pub content: String,
    pub name: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EpisodeRole {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub uid: EpisodeId,
    pub session_key: String,
    pub role: EpisodeRole,
    pub content: String,
    pub name: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
```

### 2.3 Semantic Types

```rust
// src/memmachine-rs/crates/memmachine-core/src/types/semantic.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub type FeatureId = String;
pub type SetId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFeature {
    pub set_id: SetId,
    pub category: String,
    pub tag: String,
    pub feature_name: String,
    pub value: String,
    pub metadata: FeatureMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetadata {
    pub id: FeatureId,
    pub citations: Option<Vec<String>>,
    pub other: Option<serde_json::Value>,
}
```

### 2.4 Filter Types

```rust
// src/memmachine-rs/crates/memmachine-core/src/types/filter.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterExpr {
    Comparison(Comparison),
    And { left: Box<FilterExpr>, right: Box<FilterExpr> },
    Or { left: Box<FilterExpr>, right: Box<FilterExpr> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    pub field: String,
    pub op: ComparisonOp,
    pub value: FilterValue,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonOp {
    Eq,        // =
    In,        // IN
    Gt,        // >
    Lt,        // <
    Gte,       // >=
    Lte,       // <=
    IsNull,    // IS NULL
    IsNotNull, // IS NOT NULL
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<FilterValue>),
    Null,
}
```

### 2.5 Configuration Types

```rust
// src/memmachine-rs/crates/memmachine-core/src/config.rs
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Configuration {
    pub storage: StorageConfig,
    pub episodic_memory: EpisodicMemoryConfig,
    pub semantic_memory: SemanticMemoryConfig,
    pub embedders: Vec<EmbedderConfig>,
    pub language_models: Vec<LanguageModelConfig>,
    pub rerankers: Vec<RerankerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Path to SQLite database file
    pub path: PathBuf,
    /// Enable WAL mode for better concurrency
    pub wal_mode: bool,
    /// Maximum connections in pool
    pub max_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemoryConfig {
    pub enabled: bool,
    pub short_term_memory: Option<ShortTermMemoryConfig>,
    pub long_term_memory: Option<LongTermMemoryConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortTermMemoryConfig {
    pub enabled: bool,
    pub llm_model: String,
    pub summary_prompt_system: Option<String>,
    pub summary_prompt_user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongTermMemoryConfig {
    pub enabled: bool,
    pub embedder: String,
    pub reranker: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemoryConfig {
    pub enabled: bool,
    pub embedder: String,
    pub llm_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider")]
pub enum EmbedderConfig {
    #[serde(rename = "openai")]
    OpenAI {
        name: String,
        api_key: String,
        model: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider")]
pub enum LanguageModelConfig {
    #[serde(rename = "openai")]
    OpenAI {
        name: String,
        api_key: String,
        model: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider")]
pub enum RerankerConfig {
    #[serde(rename = "cohere")]
    Cohere {
        name: String,
        api_key: String,
        model: String,
    },
    #[serde(rename = "bm25")]
    Bm25 {
        name: String,
    },
}
```

## Phase 3: Storage Layer (Week 2-3)

### 3.1 Unified Storage Manager

```rust
// src/memmachine-rs/crates/memmachine-core/src/storage/manager.rs
use crate::config::StorageConfig;
use crate::error::Result;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use std::sync::Arc;

pub struct StorageManager {
    pool: SqlitePool,
    config: StorageConfig,
}

impl StorageManager {
    pub async fn new(config: StorageConfig) -> Result<Self> {
        let db_url = format!("sqlite:{}?mode=rwc", config.path.display());
        
        let pool = SqlitePoolOptions::new()
            .max_connections(config.max_connections)
            .connect(&db_url)
            .await?;
        
        if config.wal_mode {
            sqlx::query("PRAGMA journal_mode=WAL")
                .execute(&pool)
                .await?;
        }
        
        // Load sqlite-vec extension
        // Note: This requires careful handling with sqlx
        
        Ok(Self { pool, config })
    }
    
    pub async fn initialize_schema(&self) -> Result<()> {
        // Create all tables
        self.create_episode_tables().await?;
        self.create_session_tables().await?;
        self.create_semantic_tables().await?;
        self.create_graph_tables().await?;
        Ok(())
    }
    
    async fn create_episode_tables(&self) -> Result<()> {
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS episodes (
                uid TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                name TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        "#)
        .execute(&self.pool)
        .await?;
        
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_key)")
            .execute(&self.pool)
            .await?;
        
        Ok(())
    }
    
    async fn create_session_tables(&self) -> Result<()> {
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS sessions (
                session_key TEXT PRIMARY KEY,
                description TEXT,
                configuration TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        "#)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    async fn create_semantic_tables(&self) -> Result<()> {
        // Semantic features with sqlite-vec for embeddings
        sqlx::query(r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS semantic_features USING vec0(
                embedding float[1536],
                feature_id TEXT,
                set_id TEXT,
                category TEXT,
                tag TEXT,
                feature_name TEXT,
                value TEXT
            )
        "#)
        .execute(&self.pool)
        .await?;
        
        // Feature metadata (non-vector data)
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS semantic_feature_meta (
                feature_id TEXT PRIMARY KEY,
                set_id TEXT NOT NULL,
                category TEXT NOT NULL,
                tag TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                value TEXT NOT NULL,
                citations TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        "#)
        .execute(&self.pool)
        .await?;
        
        // History tracking
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS semantic_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                set_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                ingested INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(set_id, episode_id)
            )
        "#)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    async fn create_graph_tables(&self) -> Result<()> {
        // Vector graph nodes with embeddings
        sqlx::query(r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS graph_nodes USING vec0(
                embedding float[1536],
                node_id TEXT,
                collection TEXT,
                properties TEXT
            )
        "#)
        .execute(&self.pool)
        .await?;
        
        // Graph edges
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relation TEXT NOT NULL,
                source_collection TEXT NOT NULL,
                target_collection TEXT NOT NULL,
                source_uid TEXT NOT NULL,
                target_uid TEXT NOT NULL,
                properties TEXT,
                created_at TEXT NOT NULL
            )
        "#)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
    
    pub async fn close(&self) {
        self.pool.close().await;
    }
}
```

### 3.2 Episode Storage

```rust
// src/memmachine-rs/crates/memmachine-core/src/storage/episode.rs
use crate::error::Result;
use crate::types::episode::{Episode, EpisodeEntry, EpisodeId, EpisodeRole};
use crate::types::filter::FilterExpr;
use chrono::Utc;
use sqlx::SqlitePool;
use uuid::Uuid;

pub struct EpisodeStore {
    pool: SqlitePool,
}

impl EpisodeStore {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
    
    pub async fn add_episodes(
        &self,
        session_key: &str,
        entries: Vec<EpisodeEntry>,
    ) -> Result<Vec<Episode>> {
        let mut episodes = Vec::with_capacity(entries.len());
        let now = Utc::now();
        
        for entry in entries {
            let uid = format!("ep-{}", Uuid::new_v4());
            let episode = Episode {
                uid: uid.clone(),
                session_key: session_key.to_string(),
                role: entry.role,
                content: entry.content,
                name: entry.name,
                metadata: entry.metadata,
                created_at: now,
                updated_at: now,
            };
            
            sqlx::query(r#"
                INSERT INTO episodes (uid, session_key, role, content, name, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#)
            .bind(&episode.uid)
            .bind(&episode.session_key)
            .bind(role_to_str(episode.role))
            .bind(&episode.content)
            .bind(&episode.name)
            .bind(episode.metadata.as_ref().map(|m| m.to_string()))
            .bind(episode.created_at.to_rfc3339())
            .bind(episode.updated_at.to_rfc3339())
            .execute(&self.pool)
            .await?;
            
            episodes.push(episode);
        }
        
        Ok(episodes)
    }
    
    pub async fn get_episodes(
        &self,
        filter: Option<&FilterExpr>,
        page_size: Option<u32>,
        page_num: Option<u32>,
    ) -> Result<Vec<Episode>> {
        let (where_clause, params) = build_where_clause(filter);
        let limit_offset = build_limit_offset(page_size, page_num);
        
        let query = format!(
            "SELECT uid, session_key, role, content, name, metadata, created_at, updated_at 
             FROM episodes {} ORDER BY created_at DESC {}",
            where_clause, limit_offset
        );
        
        // Execute with dynamic params...
        todo!("Implement dynamic query execution")
    }
    
    pub async fn get_episodes_count(&self, filter: Option<&FilterExpr>) -> Result<u64> {
        let (where_clause, _params) = build_where_clause(filter);
        let query = format!("SELECT COUNT(*) FROM episodes {}", where_clause);
        
        let count: (i64,) = sqlx::query_as(&query)
            .fetch_one(&self.pool)
            .await?;
        
        Ok(count.0 as u64)
    }
    
    pub async fn delete_episodes(&self, episode_ids: &[EpisodeId]) -> Result<()> {
        if episode_ids.is_empty() {
            return Ok(());
        }
        
        let placeholders: Vec<_> = (1..=episode_ids.len()).map(|i| format!("?{}", i)).collect();
        let query = format!(
            "DELETE FROM episodes WHERE uid IN ({})",
            placeholders.join(", ")
        );
        
        let mut q = sqlx::query(&query);
        for id in episode_ids {
            q = q.bind(id);
        }
        q.execute(&self.pool).await?;
        
        Ok(())
    }
}

fn role_to_str(role: EpisodeRole) -> &'static str {
    match role {
        EpisodeRole::User => "user",
        EpisodeRole::Assistant => "assistant",
        EpisodeRole::System => "system",
    }
}

fn build_where_clause(_filter: Option<&FilterExpr>) -> (String, Vec<String>) {
    // TODO: Convert FilterExpr to SQL WHERE clause
    (String::new(), Vec::new())
}

fn build_limit_offset(page_size: Option<u32>, page_num: Option<u32>) -> String {
    match (page_size, page_num) {
        (Some(size), Some(num)) => format!("LIMIT {} OFFSET {}", size, size * num),
        (Some(size), None) => format!("LIMIT {}", size),
        _ => String::new(),
    }
}
```

### 3.3 Semantic Storage with sqlite-vec

```rust
// src/memmachine-rs/crates/memmachine-core/src/storage/semantic.rs
use crate::error::Result;
use crate::types::filter::FilterExpr;
use crate::types::semantic::{FeatureId, FeatureMetadata, SemanticFeature, SetId};
use chrono::Utc;
use sqlx::SqlitePool;
use uuid::Uuid;

pub struct SemanticStore {
    pool: SqlitePool,
    embedding_dimension: usize,
}

impl SemanticStore {
    pub fn new(pool: SqlitePool, embedding_dimension: usize) -> Self {
        Self { pool, embedding_dimension }
    }
    
    pub async fn add_feature(
        &self,
        set_id: &SetId,
        category: &str,
        tag: &str,
        feature_name: &str,
        value: &str,
        embedding: &[f32],
        metadata: Option<serde_json::Value>,
    ) -> Result<FeatureId> {
        let feature_id = format!("feat-{}", Uuid::new_v4());
        let now = Utc::now();
        
        // Insert into vector table for similarity search
        // Note: sqlite-vec uses raw bytes for embeddings
        let embedding_bytes = embedding_to_bytes(embedding);
        
        sqlx::query(r#"
            INSERT INTO semantic_features (rowid, embedding, feature_id, set_id, category, tag, feature_name, value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(generate_rowid())
        .bind(&embedding_bytes)
        .bind(&feature_id)
        .bind(set_id)
        .bind(category)
        .bind(tag)
        .bind(feature_name)
        .bind(value)
        .execute(&self.pool)
        .await?;
        
        // Insert metadata
        sqlx::query(r#"
            INSERT INTO semantic_feature_meta 
            (feature_id, set_id, category, tag, feature_name, value, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(&feature_id)
        .bind(set_id)
        .bind(category)
        .bind(tag)
        .bind(feature_name)
        .bind(value)
        .bind(metadata.map(|m| m.to_string()))
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&self.pool)
        .await?;
        
        Ok(feature_id)
    }
    
    pub async fn search_features(
        &self,
        query_embedding: &[f32],
        filter: Option<&FilterExpr>,
        limit: Option<u32>,
        min_similarity: Option<f32>,
    ) -> Result<Vec<(SemanticFeature, f32)>> {
        let embedding_bytes = embedding_to_bytes(query_embedding);
        let limit = limit.unwrap_or(10);
        
        // Build filter conditions if any
        let filter_clause = build_semantic_filter(filter);
        
        let query = format!(r#"
            SELECT 
                sf.feature_id,
                sf.set_id, 
                sf.category, 
                sf.tag, 
                sf.feature_name, 
                sf.value,
                sfm.citations,
                sfm.metadata,
                (1 - vec_distance_cosine(sf.embedding, ?)) as similarity
            FROM semantic_features sf
            JOIN semantic_feature_meta sfm ON sf.feature_id = sfm.feature_id
            WHERE sf.embedding MATCH ?
            {}
            ORDER BY similarity DESC
            LIMIT ?
        "#, filter_clause);
        
        // Execute query and map results
        todo!("Implement query execution")
    }
    
    pub async fn delete_features(&self, feature_ids: &[FeatureId]) -> Result<()> {
        if feature_ids.is_empty() {
            return Ok(());
        }
        
        // Delete from both tables
        for id in feature_ids {
            sqlx::query("DELETE FROM semantic_features WHERE feature_id = ?")
                .bind(id)
                .execute(&self.pool)
                .await?;
            
            sqlx::query("DELETE FROM semantic_feature_meta WHERE feature_id = ?")
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        
        Ok(())
    }
    
    // History management
    pub async fn add_history(&self, set_id: &SetId, episode_id: &str) -> Result<()> {
        let now = Utc::now();
        
        sqlx::query(r#"
            INSERT OR IGNORE INTO semantic_history (set_id, episode_id, ingested, created_at)
            VALUES (?, ?, 0, ?)
        "#)
        .bind(set_id)
        .bind(episode_id)
        .bind(now.to_rfc3339())
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn mark_ingested(&self, set_id: &SetId, episode_ids: &[String]) -> Result<()> {
        for id in episode_ids {
            sqlx::query("UPDATE semantic_history SET ingested = 1 WHERE set_id = ? AND episode_id = ?")
                .bind(set_id)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

fn generate_rowid() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64
}

fn build_semantic_filter(_filter: Option<&FilterExpr>) -> String {
    // TODO: Convert FilterExpr to SQL conditions
    String::new()
}
```

## Phase 4: MemMachine Core (Week 3-4)

### 4.1 Main MemMachine Struct

```rust
// src/memmachine-rs/crates/memmachine-core/src/memmachine.rs
use crate::config::Configuration;
use crate::error::{MemMachineError, Result};
use crate::memory::episodic::EpisodicMemoryManager;
use crate::memory::semantic::SemanticSessionManager;
use crate::storage::episode::EpisodeStore;
use crate::storage::manager::StorageManager;
use crate::storage::semantic::SemanticStore;
use crate::storage::session::SessionStore;
use crate::types::episode::{Episode, EpisodeEntry, EpisodeId};
use crate::types::filter::FilterExpr;
use crate::types::semantic::{FeatureId, SemanticFeature};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct MemMachine {
    config: Configuration,
    storage: Arc<StorageManager>,
    episode_store: EpisodeStore,
    session_store: SessionStore,
    semantic_store: SemanticStore,
    episodic_manager: Arc<RwLock<Option<EpisodicMemoryManager>>>,
    semantic_manager: Arc<RwLock<Option<SemanticSessionManager>>>,
    started: bool,
}

#[derive(Debug, Clone)]
pub struct SessionData {
    pub session_key: String,
    pub user_profile_id: Option<String>,
    pub role_profile_id: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub session_key: String,
    pub description: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Episodic,
    Semantic,
}

#[derive(Debug)]
pub struct SearchResponse {
    pub episodic_memory: Option<EpisodicQueryResponse>,
    pub semantic_memory: Option<Vec<SemanticFeature>>,
}

#[derive(Debug)]
pub struct EpisodicQueryResponse {
    pub episodes: Vec<Episode>,
    pub scores: Vec<f32>,
}

#[derive(Debug)]
pub struct ListResults {
    pub episodic_memory: Option<Vec<Episode>>,
    pub semantic_memory: Option<Vec<SemanticFeature>>,
}

impl MemMachine {
    pub async fn new(config: Configuration) -> Result<Self> {
        let storage = Arc::new(StorageManager::new(config.storage.clone()).await?);
        storage.initialize_schema().await?;
        
        let pool = storage.pool().clone();
        let embedding_dim = 1536; // Default OpenAI dimension
        
        Ok(Self {
            config,
            storage,
            episode_store: EpisodeStore::new(pool.clone()),
            session_store: SessionStore::new(pool.clone()),
            semantic_store: SemanticStore::new(pool, embedding_dim),
            episodic_manager: Arc::new(RwLock::new(None)),
            semantic_manager: Arc::new(RwLock::new(None)),
            started: false,
        })
    }
    
    pub async fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        
        // Initialize semantic service if enabled
        if self.config.semantic_memory.enabled {
            // TODO: Initialize semantic session manager
        }
        
        self.started = true;
        Ok(())
    }
    
    pub async fn stop(&mut self) -> Result<()> {
        if !self.started {
            return Ok(());
        }
        
        self.storage.close().await;
        self.started = false;
        Ok(())
    }
    
    // Session management
    pub async fn create_session(
        &self,
        session_key: &str,
        description: &str,
    ) -> Result<SessionInfo> {
        self.session_store.create_session(session_key, description).await
    }
    
    pub async fn get_session(&self, session_key: &str) -> Result<Option<SessionInfo>> {
        self.session_store.get_session(session_key).await
    }
    
    pub async fn delete_session(&self, session_data: &SessionData) -> Result<()> {
        // Delete from all stores
        self.session_store.delete_session(&session_data.session_key).await?;
        
        // Delete episodes
        let filter = FilterExpr::Comparison(crate::types::filter::Comparison {
            field: "session_key".to_string(),
            op: crate::types::filter::ComparisonOp::Eq,
            value: crate::types::filter::FilterValue::String(session_data.session_key.clone()),
        });
        
        // TODO: Delete related data
        
        Ok(())
    }
    
    pub async fn search_sessions(&self, filter: Option<&FilterExpr>) -> Result<Vec<String>> {
        self.session_store.search_sessions(filter).await
    }
    
    // Episode management
    pub async fn add_episodes(
        &self,
        session_data: &SessionData,
        entries: Vec<EpisodeEntry>,
        target_memories: &[MemoryType],
    ) -> Result<Vec<EpisodeId>> {
        let episodes = self.episode_store
            .add_episodes(&session_data.session_key, entries)
            .await?;
        
        let episode_ids: Vec<_> = episodes.iter().map(|e| e.uid.clone()).collect();
        
        // Process for episodic memory if enabled
        if target_memories.contains(&MemoryType::Episodic) && self.config.episodic_memory.enabled {
            // TODO: Add to episodic memory manager
        }
        
        // Process for semantic memory if enabled
        if target_memories.contains(&MemoryType::Semantic) && self.config.semantic_memory.enabled {
            for episode_id in &episode_ids {
                self.semantic_store
                    .add_history(&session_data.session_key, episode_id)
                    .await?;
            }
        }
        
        Ok(episode_ids)
    }
    
    pub async fn query_search(
        &self,
        session_data: &SessionData,
        query: &str,
        target_memories: &[MemoryType],
        limit: Option<u32>,
        expand_context: u32,
        score_threshold: f32,
        filter: Option<&str>,
    ) -> Result<SearchResponse> {
        let filter_expr = filter.map(|f| parse_filter(f)).transpose()?;
        
        let mut response = SearchResponse {
            episodic_memory: None,
            semantic_memory: None,
        };
        
        // Search episodic memory
        if target_memories.contains(&MemoryType::Episodic) && self.config.episodic_memory.enabled {
            // TODO: Implement episodic search with embeddings
        }
        
        // Search semantic memory
        if target_memories.contains(&MemoryType::Semantic) && self.config.semantic_memory.enabled {
            // TODO: Get query embedding and search
        }
        
        Ok(response)
    }
    
    pub async fn list_search(
        &self,
        session_data: &SessionData,
        target_memories: &[MemoryType],
        filter: Option<&str>,
        page_size: Option<u32>,
        page_num: Option<u32>,
    ) -> Result<ListResults> {
        let filter_expr = filter.map(|f| parse_filter(f)).transpose()?;
        
        let mut results = ListResults {
            episodic_memory: None,
            semantic_memory: None,
        };
        
        if target_memories.contains(&MemoryType::Episodic) {
            let episodes = self.episode_store
                .get_episodes(filter_expr.as_ref(), page_size, page_num)
                .await?;
            results.episodic_memory = Some(episodes);
        }
        
        if target_memories.contains(&MemoryType::Semantic) {
            // TODO: List semantic features
        }
        
        Ok(results)
    }
    
    pub async fn episodes_count(
        &self,
        session_data: &SessionData,
        filter: Option<&str>,
    ) -> Result<u64> {
        let filter_expr = filter.map(|f| parse_filter(f)).transpose()?;
        
        // Add session filter
        let session_filter = FilterExpr::Comparison(crate::types::filter::Comparison {
            field: "session_key".to_string(),
            op: crate::types::filter::ComparisonOp::Eq,
            value: crate::types::filter::FilterValue::String(session_data.session_key.clone()),
        });
        
        let combined = match filter_expr {
            Some(f) => Some(FilterExpr::And {
                left: Box::new(session_filter),
                right: Box::new(f),
            }),
            None => Some(session_filter),
        };
        
        self.episode_store.get_episodes_count(combined.as_ref()).await
    }
    
    pub async fn delete_episodes(
        &self,
        episode_ids: &[EpisodeId],
        session_data: Option<&SessionData>,
    ) -> Result<()> {
        self.episode_store.delete_episodes(episode_ids).await?;
        
        // Also delete from semantic history
        // TODO: Implement
        
        Ok(())
    }
    
    pub async fn delete_features(&self, feature_ids: &[FeatureId]) -> Result<()> {
        self.semantic_store.delete_features(feature_ids).await
    }
}

fn parse_filter(filter_str: &str) -> Result<FilterExpr> {
    // TODO: Implement filter parsing with nom
    Err(MemMachineError::FilterParse("Not implemented".to_string()))
}
```

## Phase 5: LLM Clients (Week 4-5)

### 5.1 Embedder Trait

```rust
// src/memmachine-rs/crates/memmachine-core/src/embedder/traits.rs
use crate::error::Result;
use async_trait::async_trait;

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}
```

### 5.2 OpenAI Embedder

```rust
// src/memmachine-rs/crates/memmachine-core/src/embedder/openai.rs
use super::traits::Embedder;
use crate::error::{MemMachineError, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
    Client,
};
use async_trait::async_trait;

pub struct OpenAIEmbedder {
    client: Client<OpenAIConfig>,
    model: String,
    dimension: usize,
}

impl OpenAIEmbedder {
    pub fn new(api_key: &str, model: &str) -> Self {
        let config = OpenAIConfig::new().with_api_key(api_key);
        let client = Client::with_config(config);
        
        // Determine dimension based on model
        let dimension = match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        };
        
        Self {
            client,
            model: model.to_string(),
            dimension,
        }
    }
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(EmbeddingInput::String(text.to_string()))
            .build()
            .map_err(|e| MemMachineError::Embedding(e.to_string()))?;
        
        let response = self.client
            .embeddings()
            .create(request)
            .await
            .map_err(|e| MemMachineError::Embedding(e.to_string()))?;
        
        Ok(response.data[0].embedding.clone())
    }
    
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(EmbeddingInput::StringArray(texts.to_vec()))
            .build()
            .map_err(|e| MemMachineError::Embedding(e.to_string()))?;
        
        let response = self.client
            .embeddings()
            .create(request)
            .await
            .map_err(|e| MemMachineError::Embedding(e.to_string()))?;
        
        Ok(response.data.into_iter().map(|d| d.embedding).collect())
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
}
```

## Phase 6: PyO3 Bindings (Week 5-6)

### 6.1 Python Module

```rust
// src/memmachine-rs/crates/memmachine-py/src/lib.rs
use memmachine_core::{
    config::Configuration,
    memmachine::{MemMachine, MemoryType, SearchResponse, SessionData, SessionInfo},
    types::episode::EpisodeEntry,
};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::sync::Arc;
use tokio::sync::Mutex;

#[pyclass]
pub struct PyMemMachine {
    inner: Arc<Mutex<MemMachine>>,
}

#[pymethods]
impl PyMemMachine {
    #[new]
    fn new(config_path: &str) -> PyResult<Self> {
        // Load config from YAML file
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        let config: Configuration = serde_yaml::from_str(&config_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        // Create MemMachine in blocking context
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let mm = runtime.block_on(async {
            MemMachine::new(config).await
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self {
            inner: Arc::new(Mutex::new(mm)),
        })
    }
    
    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let mut mm = inner.lock().await;
            mm.start().await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(())
        })
    }
    
    fn stop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let mut mm = inner.lock().await;
            mm.stop().await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(())
        })
    }
    
    fn create_session<'py>(
        &self,
        py: Python<'py>,
        session_key: String,
        description: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let mm = inner.lock().await;
            let info = mm.create_session(&session_key, &description).await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PySessionInfo::from(info))
        })
    }
    
    fn add_episodes<'py>(
        &self,
        py: Python<'py>,
        session_key: String,
        entries: Vec<PyEpisodeEntry>,
        target_memories: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let mm = inner.lock().await;
            
            let session_data = SessionData {
                session_key,
                user_profile_id: None,
                role_profile_id: None,
                session_id: None,
            };
            
            let entries: Vec<EpisodeEntry> = entries.into_iter().map(Into::into).collect();
            let memories: Vec<MemoryType> = target_memories.iter()
                .filter_map(|s| match s.as_str() {
                    "episodic" => Some(MemoryType::Episodic),
                    "semantic" => Some(MemoryType::Semantic),
                    _ => None,
                })
                .collect();
            
            let ids = mm.add_episodes(&session_data, entries, &memories).await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            Ok(ids)
        })
    }
    
    // TODO: Add remaining methods...
}

#[pyclass]
#[derive(Clone)]
pub struct PySessionInfo {
    #[pyo3(get)]
    session_key: String,
    #[pyo3(get)]
    description: String,
}

impl From<SessionInfo> for PySessionInfo {
    fn from(info: SessionInfo) -> Self {
        Self {
            session_key: info.session_key,
            description: info.description,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyEpisodeEntry {
    #[pyo3(get, set)]
    role: String,
    #[pyo3(get, set)]
    content: String,
    #[pyo3(get, set)]
    name: Option<String>,
}

#[pymethods]
impl PyEpisodeEntry {
    #[new]
    fn new(role: String, content: String, name: Option<String>) -> Self {
        Self { role, content, name }
    }
}

impl From<PyEpisodeEntry> for EpisodeEntry {
    fn from(entry: PyEpisodeEntry) -> Self {
        use memmachine_core::types::episode::EpisodeRole;
        
        let role = match entry.role.as_str() {
            "user" => EpisodeRole::User,
            "assistant" => EpisodeRole::Assistant,
            "system" => EpisodeRole::System,
            _ => EpisodeRole::User,
        };
        
        EpisodeEntry {
            role,
            content: entry.content,
            name: entry.name,
            metadata: None,
        }
    }
}

#[pymodule]
fn memmachine_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemMachine>()?;
    m.add_class::<PySessionInfo>()?;
    m.add_class::<PyEpisodeEntry>()?;
    Ok(())
}
```

## Phase 7: Integration (Week 6-7)

### 7.1 Python Wrapper

Create a Python wrapper that provides backward compatibility:

```python
# src/memmachine/main/memmachine_rust.py
"""Rust-backed MemMachine implementation."""

from typing import Any
from memmachine_rust import PyMemMachine, PyEpisodeEntry, PySessionInfo

class MemMachine:
    """Drop-in replacement for Python MemMachine using Rust backend."""
    
    def __init__(self, conf, resources=None):
        # Convert Configuration to YAML path or dict
        config_path = self._save_config_temp(conf)
        self._rust = PyMemMachine(config_path)
    
    async def start(self):
        return await self._rust.start()
    
    async def stop(self):
        return await self._rust.stop()
    
    async def create_session(self, session_key: str, *, description: str = ""):
        return await self._rust.create_session(session_key, description)
    
    async def add_episodes(self, session_data, episode_entries, *, target_memories=None):
        if target_memories is None:
            target_memories = ["episodic", "semantic"]
        
        entries = [
            PyEpisodeEntry(
                role=e.role,
                content=e.content,
                name=e.name,
            )
            for e in episode_entries
        ]
        
        return await self._rust.add_episodes(
            session_data.session_key,
            entries,
            target_memories,
        )
    
    # ... implement remaining methods
```

## Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Setup | Week 1 | Cargo workspace, dependencies |
| 2. Types | Week 1-2 | All data types, error handling |
| 3. Storage | Week 2-3 | SQLite + sqlite-vec layer |
| 4. Core | Week 3-4 | MemMachine struct, main logic |
| 5. LLM | Week 4-5 | OpenAI, Bedrock, Cohere clients |
| 6. PyO3 | Week 5-6 | Python bindings |
| 7. Integration | Week 6-7 | Testing, migration |
| 8. Polish | Week 7-8 | Documentation, optimization |

## Migration Strategy

1. **Parallel running**: Keep Python implementation alongside Rust
2. **Feature flag**: `USE_RUST_BACKEND=1` to switch
3. **Incremental adoption**: Start with new deployments
4. **Data migration**: SQLite files are compatible
5. **Rollback plan**: Python fallback always available

## Testing Strategy

1. **Unit tests**: Rust native tests for each module
2. **Integration tests**: Python pytest against Rust backend
3. **Compatibility tests**: Compare Python vs Rust outputs
4. **Performance benchmarks**: Measure latency improvements

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| sqlite-vec limitations at scale | Document <100K vector limit |
| PyO3 async complexity | Extensive testing, simple API |
| Breaking changes in sqlite-vec | Pin to specific version |
| Performance regression | Benchmark before migration |
