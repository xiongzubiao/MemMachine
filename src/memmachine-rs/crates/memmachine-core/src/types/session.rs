use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub session_key: String,
    #[serde(default)]
    pub user_profile_id: Option<String>,
    #[serde(default)]
    pub role_profile_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
}

impl SessionData {
    pub fn new(session_key: impl Into<String>) -> Self {
        Self {
            session_key: session_key.into(),
            user_profile_id: None,
            role_profile_id: None,
            session_id: None,
        }
    }

    pub fn with_user_profile(mut self, user_profile_id: impl Into<String>) -> Self {
        self.user_profile_id = Some(user_profile_id.into());
        self
    }

    pub fn with_role_profile(mut self, role_profile_id: impl Into<String>) -> Self {
        self.role_profile_id = Some(role_profile_id.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_key: String,
    pub description: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl SessionInfo {
    pub fn new(session_key: impl Into<String>, description: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            session_key: session_key.into(),
            description: description.into(),
            created_at: now,
            updated_at: now,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub session_key: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub configuration: Option<serde_json::Value>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Session {
    pub fn new(session_key: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            session_key: session_key.into(),
            description: None,
            configuration: None,
            metadata: None,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}
