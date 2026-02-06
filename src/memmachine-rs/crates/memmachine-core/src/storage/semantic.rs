use crate::error::Result;
use crate::types::semantic::{FeatureId, FeatureMetadata, SemanticFeature, SetId};
use chrono::{DateTime, Utc};
use sqlx::{FromRow, SqlitePool};
use uuid::Uuid;

#[derive(Clone)]
pub struct SemanticStore {
    pool: SqlitePool,
    embedding_dimension: usize,
}

#[derive(FromRow)]
struct FeatureRow {
    feature_id: String,
    set_id: String,
    category: String,
    tag: String,
    feature_name: String,
    value: String,
    embedding: Option<Vec<u8>>,
    citations: Option<String>,
    metadata: Option<String>,
    created_at: String,
    updated_at: String,
}

impl TryFrom<FeatureRow> for SemanticFeature {
    type Error = crate::error::MemMachineError;

    fn try_from(row: FeatureRow) -> Result<Self> {
        let citations: Option<Vec<String>> = row
            .citations
            .map(|s| serde_json::from_str(&s))
            .transpose()?;
        let other: Option<serde_json::Value> = row
            .metadata
            .map(|s| serde_json::from_str(&s))
            .transpose()?;
        let created_at = DateTime::parse_from_rfc3339(&row.created_at)
            .map_err(|e| crate::error::MemMachineError::InvalidArgument(e.to_string()))?
            .with_timezone(&Utc);
        let updated_at = DateTime::parse_from_rfc3339(&row.updated_at)
            .map_err(|e| crate::error::MemMachineError::InvalidArgument(e.to_string()))?
            .with_timezone(&Utc);

        Ok(SemanticFeature {
            set_id: row.set_id,
            category: row.category,
            tag: row.tag,
            feature_name: row.feature_name,
            value: row.value,
            metadata: FeatureMetadata {
                id: row.feature_id,
                citations,
                other,
                created_at,
                updated_at,
            },
        })
    }
}

impl SemanticStore {
    pub fn new(pool: SqlitePool, embedding_dimension: usize) -> Self {
        Self {
            pool,
            embedding_dimension,
        }
    }

    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }

    pub async fn add_feature(
        &self,
        set_id: &SetId,
        category: &str,
        tag: &str,
        feature_name: &str,
        value: &str,
        embedding: Option<&[f32]>,
        citations: Option<&[String]>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<FeatureId> {
        let feature_id = format!("feat-{}", Uuid::new_v4());
        let now = Utc::now().to_rfc3339();

        let embedding_bytes = embedding.map(embedding_to_bytes);
        let citations_str = citations.map(|c| serde_json::to_string(c)).transpose()?;
        let metadata_str = metadata.map(|m| serde_json::to_string(m)).transpose()?;

        sqlx::query(
            r#"
            INSERT INTO semantic_features 
            (feature_id, set_id, category, tag, feature_name, value, embedding, citations, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&feature_id)
        .bind(set_id)
        .bind(category)
        .bind(tag)
        .bind(feature_name)
        .bind(value)
        .bind(&embedding_bytes)
        .bind(&citations_str)
        .bind(&metadata_str)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        Ok(feature_id)
    }

    pub async fn get_feature(&self, feature_id: &FeatureId) -> Result<Option<SemanticFeature>> {
        let row: Option<FeatureRow> = sqlx::query_as(
            r#"
            SELECT feature_id, set_id, category, tag, feature_name, value, embedding, citations, metadata, created_at, updated_at
            FROM semantic_features 
            WHERE feature_id = ?
            "#,
        )
        .bind(feature_id)
        .fetch_optional(&self.pool)
        .await?;

        row.map(SemanticFeature::try_from).transpose()
    }

    pub async fn get_features_by_set(&self, set_id: &SetId) -> Result<Vec<SemanticFeature>> {
        let rows: Vec<FeatureRow> = sqlx::query_as(
            r#"
            SELECT feature_id, set_id, category, tag, feature_name, value, embedding, citations, metadata, created_at, updated_at
            FROM semantic_features 
            WHERE set_id = ?
            ORDER BY created_at DESC
            "#,
        )
        .bind(set_id)
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(SemanticFeature::try_from).collect()
    }

    pub async fn search_by_embedding(
        &self,
        query_embedding: &[f32],
        set_id: Option<&SetId>,
        limit: Option<u32>,
        min_similarity: Option<f32>,
    ) -> Result<Vec<(SemanticFeature, f32)>> {
        let limit = limit.unwrap_or(10);
        let min_sim = min_similarity.unwrap_or(0.0);

        let rows: Vec<FeatureRow> = if let Some(sid) = set_id {
            sqlx::query_as(
                r#"
                SELECT feature_id, set_id, category, tag, feature_name, value, embedding, citations, metadata, created_at, updated_at
                FROM semantic_features 
                WHERE set_id = ? AND embedding IS NOT NULL
                "#,
            )
            .bind(sid)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query_as(
                r#"
                SELECT feature_id, set_id, category, tag, feature_name, value, embedding, citations, metadata, created_at, updated_at
                FROM semantic_features 
                WHERE embedding IS NOT NULL
                "#,
            )
            .fetch_all(&self.pool)
            .await?
        };

        let mut results: Vec<(SemanticFeature, f32)> = rows
            .into_iter()
            .filter_map(|row| {
                let embedding = row.embedding.as_ref()?;
                let feature_embedding = bytes_to_embedding(embedding);
                let similarity = cosine_similarity(query_embedding, &feature_embedding);

                if similarity >= min_sim {
                    SemanticFeature::try_from(row)
                        .ok()
                        .map(|f| (f, similarity))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit as usize);

        Ok(results)
    }

    pub async fn delete_features(&self, feature_ids: &[FeatureId]) -> Result<u64> {
        if feature_ids.is_empty() {
            return Ok(0);
        }

        let placeholders: Vec<_> = (0..feature_ids.len()).map(|_| "?").collect();
        let query = format!(
            "DELETE FROM semantic_features WHERE feature_id IN ({})",
            placeholders.join(", ")
        );

        let mut q = sqlx::query(&query);
        for id in feature_ids {
            q = q.bind(id);
        }

        let result = q.execute(&self.pool).await?;
        Ok(result.rows_affected())
    }

    pub async fn delete_by_set(&self, set_id: &SetId) -> Result<u64> {
        let result = sqlx::query("DELETE FROM semantic_features WHERE set_id = ?")
            .bind(set_id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    pub async fn add_history(&self, set_id: &SetId, episode_id: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            INSERT OR IGNORE INTO semantic_history (set_id, episode_id, ingested, created_at)
            VALUES (?, ?, 0, ?)
            "#,
        )
        .bind(set_id)
        .bind(episode_id)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_uningestied_episodes(&self, set_id: &SetId) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT episode_id FROM semantic_history WHERE set_id = ? AND ingested = 0",
        )
        .bind(set_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|(id,)| id).collect())
    }

    pub async fn mark_ingested(&self, set_id: &SetId, episode_ids: &[String]) -> Result<()> {
        for id in episode_ids {
            sqlx::query(
                "UPDATE semantic_history SET ingested = 1 WHERE set_id = ? AND episode_id = ?",
            )
            .bind(set_id)
            .bind(id)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(arr)
        })
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}
