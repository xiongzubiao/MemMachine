use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub type EpisodeId = String;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EpisodeRole {
    User,
    Assistant,
    System,
}

impl EpisodeRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
        }
    }
}

impl std::str::FromStr for EpisodeRole {
    type Err = crate::error::MemMachineError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            "system" => Ok(Self::System),
            _ => Err(crate::error::MemMachineError::InvalidArgument(format!(
                "Invalid episode role: {}",
                s
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeEntry {
    pub role: EpisodeRole,
    pub content: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

impl EpisodeEntry {
    pub fn new(role: EpisodeRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
            metadata: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub uid: EpisodeId,
    pub session_key: String,
    pub role: EpisodeRole,
    pub content: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Episode {
    pub fn new(
        uid: impl Into<String>,
        session_key: impl Into<String>,
        entry: EpisodeEntry,
    ) -> Self {
        let now = Utc::now();
        Self {
            uid: uid.into(),
            session_key: session_key.into(),
            role: entry.role,
            content: entry.content,
            name: entry.name,
            metadata: entry.metadata,
            created_at: now,
            updated_at: now,
        }
    }
}
