use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Configuration {
    pub storage: StorageConfig,
    #[serde(default)]
    pub episodic_memory: EpisodicMemoryConfig,
    #[serde(default)]
    pub semantic_memory: SemanticMemoryConfig,
    #[serde(default)]
    pub embedders: Vec<EmbedderConfig>,
    #[serde(default)]
    pub language_models: Vec<LanguageModelConfig>,
    #[serde(default)]
    pub rerankers: Vec<RerankerConfig>,
}

impl Configuration {
    pub fn new(storage_path: impl Into<PathBuf>) -> Self {
        Self {
            storage: StorageConfig::new(storage_path),
            episodic_memory: EpisodicMemoryConfig::default(),
            semantic_memory: SemanticMemoryConfig::default(),
            embedders: Vec::new(),
            language_models: Vec::new(),
            rerankers: Vec::new(),
        }
    }

    pub fn from_yaml_file(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Configuration = serde_yaml::from_str(&content)
            .map_err(|e| crate::error::MemMachineError::Configuration(e.to_string()))?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub path: PathBuf,
    #[serde(default = "default_true")]
    pub wal_mode: bool,
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

impl StorageConfig {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            wal_mode: true,
            max_connections: 5,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_max_connections() -> u32 {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EpisodicMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub short_term_memory: Option<ShortTermMemoryConfig>,
    #[serde(default)]
    pub long_term_memory: Option<LongTermMemoryConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortTermMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    pub llm_model: String,
    #[serde(default)]
    pub summary_prompt_system: Option<String>,
    #[serde(default)]
    pub summary_prompt_user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongTermMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    pub embedder: String,
    #[serde(default)]
    pub reranker: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SemanticMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub embedder: Option<String>,
    #[serde(default)]
    pub llm_model: Option<String>,
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

impl EmbedderConfig {
    pub fn name(&self) -> &str {
        match self {
            Self::OpenAI { name, .. } => name,
        }
    }
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

impl LanguageModelConfig {
    pub fn name(&self) -> &str {
        match self {
            Self::OpenAI { name, .. } => name,
        }
    }
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
    Bm25 { name: String },
}

impl RerankerConfig {
    pub fn name(&self) -> &str {
        match self {
            Self::Cohere { name, .. } => name,
            Self::Bm25 { name } => name,
        }
    }
}
