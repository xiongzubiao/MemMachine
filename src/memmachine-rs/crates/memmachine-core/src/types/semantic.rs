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

impl SemanticFeature {
    pub fn new(
        set_id: impl Into<String>,
        category: impl Into<String>,
        tag: impl Into<String>,
        feature_name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            set_id: set_id.into(),
            category: category.into(),
            tag: tag.into(),
            feature_name: feature_name.into(),
            value: value.into(),
            metadata: FeatureMetadata::new(),
        }
    }

    pub fn with_metadata(mut self, metadata: FeatureMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetadata {
    pub id: FeatureId,
    #[serde(default)]
    pub citations: Option<Vec<String>>,
    #[serde(default)]
    pub other: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl FeatureMetadata {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            id: format!("feat-{}", uuid::Uuid::new_v4()),
            citations: None,
            other: None,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_citations(mut self, citations: Vec<String>) -> Self {
        self.citations = Some(citations);
        self
    }
}

impl Default for FeatureMetadata {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticHistory {
    pub id: i64,
    pub set_id: SetId,
    pub episode_id: String,
    pub ingested: bool,
    pub created_at: DateTime<Utc>,
}
