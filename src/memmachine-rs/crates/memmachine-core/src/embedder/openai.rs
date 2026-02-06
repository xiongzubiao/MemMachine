use super::traits::Embedder;
use crate::error::{MemMachineError, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
    Client,
};
use async_trait::async_trait;

pub struct OpenAIEmbedder {
    client: Client<OpenAIConfig>,
    model: String,
    dimension: usize,
    name: String,
}

impl OpenAIEmbedder {
    pub fn new(name: impl Into<String>, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        let config = OpenAIConfig::new().with_api_key(api_key.into());
        let client = Client::with_config(config);

        let dimension = Self::dimension_for_model(&model);

        Self {
            client,
            model,
            dimension,
            name: name.into(),
        }
    }

    fn dimension_for_model(model: &str) -> usize {
        match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        }
    }
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(EmbeddingInput::String(text.to_string()))
            .build()
            .map_err(|e| MemMachineError::Embedding(e.to_string()))?;

        let response = self
            .client
            .embeddings()
            .create(request)
            .await
            .map_err(|e| MemMachineError::Embedding(e.to_string()))?;

        if response.data.is_empty() {
            return Err(MemMachineError::Embedding(
                "No embeddings returned".to_string(),
            ));
        }

        Ok(response.data[0].embedding.clone())
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(EmbeddingInput::StringArray(texts.to_vec()))
            .build()
            .map_err(|e| MemMachineError::Embedding(e.to_string()))?;

        let response = self
            .client
            .embeddings()
            .create(request)
            .await
            .map_err(|e| MemMachineError::Embedding(e.to_string()))?;

        let mut results: Vec<_> = response
            .data
            .into_iter()
            .map(|d| (d.index, d.embedding))
            .collect();

        results.sort_by_key(|(idx, _)| *idx);

        Ok(results.into_iter().map(|(_, emb)| emb).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Debug for OpenAIEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIEmbedder")
            .field("name", &self.name)
            .field("model", &self.model)
            .field("dimension", &self.dimension)
            .finish()
    }
}
