use crate::error::Result;
use sqlx::SqlitePool;

pub async fn initialize_schema(pool: &SqlitePool) -> Result<()> {
    create_episode_tables(pool).await?;
    create_session_tables(pool).await?;
    create_semantic_tables(pool).await?;
    Ok(())
}

async fn create_episode_tables(pool: &SqlitePool) -> Result<()> {
    sqlx::query(
        r#"
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
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_key)")
        .execute(pool)
        .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at)")
        .execute(pool)
        .await?;

    Ok(())
}

async fn create_session_tables(pool: &SqlitePool) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS sessions (
            session_key TEXT PRIMARY KEY,
            description TEXT,
            configuration TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        "#,
    )
    .execute(pool)
    .await?;

    Ok(())
}

async fn create_semantic_tables(pool: &SqlitePool) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS semantic_features (
            feature_id TEXT PRIMARY KEY,
            set_id TEXT NOT NULL,
            category TEXT NOT NULL,
            tag TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            value TEXT NOT NULL,
            embedding BLOB,
            citations TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_semantic_set ON semantic_features(set_id)")
        .execute(pool)
        .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_semantic_category ON semantic_features(category, tag)",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS semantic_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            set_id TEXT NOT NULL,
            episode_id TEXT NOT NULL,
            ingested INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(set_id, episode_id)
        )
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_semantic_history_set ON semantic_history(set_id)")
        .execute(pool)
        .await?;

    Ok(())
}
