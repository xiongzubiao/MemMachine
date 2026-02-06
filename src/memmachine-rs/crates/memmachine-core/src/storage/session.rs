use crate::error::Result;
use crate::types::filter::FilterExpr;
use crate::types::session::{Session, SessionInfo};
use chrono::{DateTime, Utc};
use sqlx::{FromRow, SqlitePool};

#[derive(Clone)]
pub struct SessionStore {
    pool: SqlitePool,
}

#[derive(FromRow)]
struct SessionRow {
    session_key: String,
    description: Option<String>,
    configuration: Option<String>,
    metadata: Option<String>,
    created_at: String,
    updated_at: String,
}

impl TryFrom<SessionRow> for Session {
    type Error = crate::error::MemMachineError;

    fn try_from(row: SessionRow) -> Result<Self> {
        let configuration = row
            .configuration
            .map(|s| serde_json::from_str(&s))
            .transpose()?;
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

        Ok(Session {
            session_key: row.session_key,
            description: row.description,
            configuration,
            metadata,
            created_at,
            updated_at,
        })
    }
}

impl SessionStore {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    pub async fn create_session(
        &self,
        session_key: &str,
        description: &str,
    ) -> Result<SessionInfo> {
        let now = Utc::now();
        let now_str = now.to_rfc3339();

        sqlx::query(
            r#"
            INSERT INTO sessions (session_key, description, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            "#,
        )
        .bind(session_key)
        .bind(description)
        .bind(&now_str)
        .bind(&now_str)
        .execute(&self.pool)
        .await?;

        Ok(SessionInfo {
            session_key: session_key.to_string(),
            description: description.to_string(),
            created_at: now,
            updated_at: now,
        })
    }

    pub async fn get_session(&self, session_key: &str) -> Result<Option<Session>> {
        let row: Option<SessionRow> = sqlx::query_as(
            "SELECT session_key, description, configuration, metadata, created_at, updated_at FROM sessions WHERE session_key = ?",
        )
        .bind(session_key)
        .fetch_optional(&self.pool)
        .await?;

        row.map(Session::try_from).transpose()
    }

    pub async fn session_exists(&self, session_key: &str) -> Result<bool> {
        let count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM sessions WHERE session_key = ?")
                .bind(session_key)
                .fetch_one(&self.pool)
                .await?;
        Ok(count > 0)
    }

    pub async fn update_session(
        &self,
        session_key: &str,
        description: Option<&str>,
        configuration: Option<&serde_json::Value>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let config_str = configuration.map(|c| serde_json::to_string(c)).transpose()?;
        let metadata_str = metadata.map(|m| serde_json::to_string(m)).transpose()?;

        sqlx::query(
            r#"
            UPDATE sessions 
            SET description = COALESCE(?, description),
                configuration = COALESCE(?, configuration),
                metadata = COALESCE(?, metadata),
                updated_at = ?
            WHERE session_key = ?
            "#,
        )
        .bind(description)
        .bind(&config_str)
        .bind(&metadata_str)
        .bind(&now)
        .bind(session_key)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn delete_session(&self, session_key: &str) -> Result<bool> {
        let result = sqlx::query("DELETE FROM sessions WHERE session_key = ?")
            .bind(session_key)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn list_sessions(&self) -> Result<Vec<String>> {
        let keys: Vec<(String,)> =
            sqlx::query_as("SELECT session_key FROM sessions ORDER BY created_at DESC")
                .fetch_all(&self.pool)
                .await?;
        Ok(keys.into_iter().map(|(k,)| k).collect())
    }

    pub async fn search_sessions(&self, filter: Option<&FilterExpr>) -> Result<Vec<String>> {
        let (where_clause, params) = build_where_clause(filter);
        let query = format!(
            "SELECT session_key FROM sessions {} ORDER BY created_at DESC",
            where_clause
        );

        let mut q = sqlx::query_as::<_, (String,)>(&query);
        for param in params {
            q = q.bind(param);
        }

        let rows = q.fetch_all(&self.pool).await?;
        Ok(rows.into_iter().map(|(k,)| k).collect())
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
