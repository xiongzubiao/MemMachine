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

    #[error("Rusqlite error: {0}")]
    Rusqlite(#[from] rusqlite::Error),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("Filter parse error: {0}")]
    FilterParse(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, MemMachineError>;
