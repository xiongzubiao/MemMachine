pub mod config;
pub mod embedder;
pub mod error;
pub mod llm;
pub mod memmachine;
pub mod storage;
pub mod types;

pub use config::{
    Configuration, EmbedderConfig, EpisodicMemoryConfig, LanguageModelConfig, RerankerConfig,
    SemanticMemoryConfig, StorageConfig,
};
pub use embedder::{Embedder, OpenAIEmbedder};
pub use error::{MemMachineError, Result};
pub use llm::{ChatMessage, ChatRole, Llm, OpenAILlm};
pub use memmachine::{EpisodicSearchResult, ListResults, MemMachine, MemoryType, SearchResponse};
pub use storage::{EpisodeStore, SemanticStore, SessionStore, StorageManager};
pub use types::episode::{Episode, EpisodeEntry, EpisodeId, EpisodeRole};
pub use types::filter::{Comparison, ComparisonOp, FilterExpr, FilterValue};
pub use types::semantic::{FeatureId, FeatureMetadata, SemanticFeature, SemanticHistory};
pub use types::session::{Session, SessionData, SessionInfo};
