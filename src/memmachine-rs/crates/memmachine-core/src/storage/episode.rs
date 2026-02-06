use crate::error::Result;
use crate::types::episode::{Episode, EpisodeEntry, EpisodeId, EpisodeRole};
use crate::types::filter::FilterExpr;
use chrono::{DateTime, Utc};
use sqlx::{FromRow, SqlitePool};
use uuid::Uuid;

#[derive(Clone)]
pub struct EpisodeStore {
    pool: SqlitePool,
}

#[derive(FromRow)]
struct EpisodeRow {
    uid: String,
    session_key: String,
    role: String,
    content: String,
    name: Option<String>,
    metadata: Option<String>,
    created_at: String,
    updated_at: String,
}

impl TryFrom<EpisodeRow> for Episode {
    type Error = crate::error::MemMachineError;

    fn try_from(row: EpisodeRow) -> Result<Self> {
        let role: EpisodeRole = row.role.parse()?;
        let metadata = row
            .metadata
            .map(|s| serde_json::from_str(&s))
            .transpose()?;
        let created_at = DateTime::parse_from_rfc3339(&row.created_at)
            .map_err(|e| crate::error::MemMachineError::InvalidArgument(e.to_string()))?
            .with_timezone(&Utc);
        let updated_at = DateTime::parse_from_rfc3339(&row.updated_at)
            .map_err(|e| crate::error::MemMachineError::InvalidArgument(e.to_string()))?
            .with_timezone(&Utc);

        Ok(Episode {
            uid: row.uid,
            session_key: row.session_key,
            role,
            content: row.content,
            name: row.name,
            metadata,
            created_at,
            updated_at,
        })
    }
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
        let now_str = now.to_rfc3339();

        for entry in entries {
            let uid = format!("ep-{}", Uuid::new_v4());
            let metadata_str = entry
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m))
                .transpose()?;

            sqlx::query(
                r#"
                INSERT INTO episodes (uid, session_key, role, content, name, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                "#,
            )
            .bind(&uid)
            .bind(session_key)
            .bind(entry.role.as_str())
            .bind(&entry.content)
            .bind(&entry.name)
            .bind(&metadata_str)
            .bind(&now_str)
            .bind(&now_str)
            .execute(&self.pool)
            .await?;

            episodes.push(Episode {
                uid,
                session_key: session_key.to_string(),
                role: entry.role,
                content: entry.content,
                name: entry.name,
                metadata: entry.metadata,
                created_at: now,
                updated_at: now,
            });
        }

        Ok(episodes)
    }

    pub async fn get_episode(&self, uid: &str) -> Result<Option<Episode>> {
        let row: Option<EpisodeRow> = sqlx::query_as(
            "SELECT uid, session_key, role, content, name, metadata, created_at, updated_at FROM episodes WHERE uid = ?",
        )
        .bind(uid)
        .fetch_optional(&self.pool)
        .await?;

        row.map(Episode::try_from).transpose()
    }

    pub async fn get_episodes_by_session(&self, session_key: &str) -> Result<Vec<Episode>> {
        let rows: Vec<EpisodeRow> = sqlx::query_as(
            r#"
            SELECT uid, session_key, role, content, name, metadata, created_at, updated_at 
            FROM episodes 
            WHERE session_key = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_key)
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(Episode::try_from).collect()
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
            r#"
            SELECT uid, session_key, role, content, name, metadata, created_at, updated_at 
            FROM episodes 
            {} 
            ORDER BY created_at DESC 
            {}
            "#,
            where_clause, limit_offset
        );

        let mut q = sqlx::query_as::<_, EpisodeRow>(&query);
        for param in params {
            q = q.bind(param);
        }

        let rows = q.fetch_all(&self.pool).await?;
        rows.into_iter().map(Episode::try_from).collect()
    }

    pub async fn get_episodes_count(&self, filter: Option<&FilterExpr>) -> Result<u64> {
        let (where_clause, params) = build_where_clause(filter);
        let query = format!("SELECT COUNT(*) as count FROM episodes {}", where_clause);

        let mut q = sqlx::query_scalar::<_, i64>(&query);
        for param in params {
            q = q.bind(param);
        }

        let count = q.fetch_one(&self.pool).await?;
        Ok(count as u64)
    }

    pub async fn delete_episodes(&self, episode_ids: &[EpisodeId]) -> Result<u64> {
        if episode_ids.is_empty() {
            return Ok(0);
        }

        let placeholders: Vec<_> = (0..episode_ids.len()).map(|_| "?").collect();
        let query = format!(
            "DELETE FROM episodes WHERE uid IN ({})",
            placeholders.join(", ")
        );

        let mut q = sqlx::query(&query);
        for id in episode_ids {
            q = q.bind(id);
        }

        let result = q.execute(&self.pool).await?;
        Ok(result.rows_affected())
    }

    pub async fn delete_by_session(&self, session_key: &str) -> Result<u64> {
        let result = sqlx::query("DELETE FROM episodes WHERE session_key = ?")
            .bind(session_key)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }
}

fn build_where_clause(filter: Option<&FilterExpr>) -> (String, Vec<String>) {
    match filter {
        Some(expr) => {
            let (clause, params) = filter_to_sql(expr);
            (format!("WHERE {}", clause), params)
        }
        None => (String::new(), Vec::new()),
    }
}

fn filter_to_sql(filter: &FilterExpr) -> (String, Vec<String>) {
    match filter {
        FilterExpr::Comparison(comp) => {
            let param = filter_value_to_string(&comp.value);
            let clause = format!("{} {} ?", comp.field, comp.op.as_sql());
            (clause, vec![param])
        }
        FilterExpr::And { left, right } => {
            let (left_sql, mut left_params) = filter_to_sql(left);
            let (right_sql, right_params) = filter_to_sql(right);
            left_params.extend(right_params);
            (format!("({} AND {})", left_sql, right_sql), left_params)
        }
        FilterExpr::Or { left, right } => {
            let (left_sql, mut left_params) = filter_to_sql(left);
            let (right_sql, right_params) = filter_to_sql(right);
            left_params.extend(right_params);
            (format!("({} OR {})", left_sql, right_sql), left_params)
        }
    }
}

fn filter_value_to_string(value: &crate::types::filter::FilterValue) -> String {
    use crate::types::filter::FilterValue;
    match value {
        FilterValue::String(s) => s.clone(),
        FilterValue::Int(i) => i.to_string(),
        FilterValue::Float(f) => f.to_string(),
        FilterValue::Bool(b) => if *b { "1" } else { "0" }.to_string(),
        FilterValue::Null => String::new(),
        FilterValue::List(items) => items
            .iter()
            .map(filter_value_to_string)
            .collect::<Vec<_>>()
            .join(","),
    }
}

fn build_limit_offset(page_size: Option<u32>, page_num: Option<u32>) -> String {
    match (page_size, page_num) {
        (Some(size), Some(num)) => format!("LIMIT {} OFFSET {}", size, size * num),
        (Some(size), None) => format!("LIMIT {}", size),
        _ => String::new(),
    }
}
