use crate::config::{Configuration, EmbedderConfig, LanguageModelConfig};
use crate::embedder::{Embedder, OpenAIEmbedder};
use crate::error::Result;
use crate::llm::{Llm, OpenAILlm};
use crate::storage::{EpisodeStore, SemanticStore, SessionStore, StorageManager};
use crate::types::episode::{Episode, EpisodeEntry, EpisodeId};
use crate::types::filter::FilterExpr;
use crate::types::semantic::{FeatureId, SemanticFeature};
use crate::types::session::{Session, SessionData, SessionInfo};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Episodic,
    Semantic,
}

#[derive(Debug)]
pub struct SearchResponse {
    pub episodic_memory: Option<EpisodicSearchResult>,
    pub semantic_memory: Option<Vec<(SemanticFeature, f32)>>,
}

#[derive(Debug)]
pub struct EpisodicSearchResult {
    pub episodes: Vec<Episode>,
    pub scores: Vec<f32>,
}

#[derive(Debug)]
pub struct ListResults {
    pub episodic_memory: Option<Vec<Episode>>,
    pub semantic_memory: Option<Vec<SemanticFeature>>,
}

pub struct MemMachine {
    config: Configuration,
    storage: StorageManager,
    episode_store: EpisodeStore,
    session_store: SessionStore,
    semantic_store: SemanticStore,
    embedders: HashMap<String, Arc<dyn Embedder>>,
    llms: HashMap<String, Arc<dyn Llm>>,
    started: bool,
}

impl MemMachine {
    pub async fn new(config: Configuration) -> Result<Self> {
        let storage = StorageManager::new(config.storage.clone()).await?;
        storage.initialize().await?;

        let embedding_dim = 1536;
        let episode_store = storage.episode_store();
        let session_store = storage.session_store();
        let semantic_store = storage.semantic_store(embedding_dim);

        Ok(Self {
            config,
            storage,
            episode_store,
            session_store,
            semantic_store,
            embedders: HashMap::new(),
            llms: HashMap::new(),
            started: false,
        })
    }

    pub async fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }

        self.initialize_embedders().await?;
        self.initialize_llms()?;

        self.started = true;
        Ok(())
    }

    pub async fn stop(&mut self) -> Result<()> {
        if !self.started {
            return Ok(());
        }

        self.storage.close().await;
        self.started = false;
        Ok(())
    }

    pub fn is_started(&self) -> bool {
        self.started
    }

    async fn initialize_embedders(&mut self) -> Result<()> {
        for embedder_config in &self.config.embedders {
            let embedder: Arc<dyn Embedder> = match embedder_config {
                EmbedderConfig::OpenAI {
                    name,
                    api_key,
                    model,
                } => Arc::new(OpenAIEmbedder::new(name, api_key, model)),
            };
            self.embedders.insert(embedder_config.name().to_string(), embedder);
        }
        Ok(())
    }

    fn initialize_llms(&mut self) -> Result<()> {
        for llm_config in &self.config.language_models {
            let llm: Arc<dyn Llm> = match llm_config {
                LanguageModelConfig::OpenAI {
                    name,
                    api_key,
                    model,
                } => Arc::new(OpenAILlm::new(name, api_key, model)),
            };
            self.llms.insert(llm_config.name().to_string(), llm);
        }
        Ok(())
    }

    pub fn get_embedder(&self, name: &str) -> Option<Arc<dyn Embedder>> {
        self.embedders.get(name).cloned()
    }

    pub fn get_llm(&self, name: &str) -> Option<Arc<dyn Llm>> {
        self.llms.get(name).cloned()
    }

    pub async fn create_session(
        &self,
        session_key: &str,
        description: &str,
    ) -> Result<SessionInfo> {
        self.session_store
            .create_session(session_key, description)
            .await
    }

    pub async fn get_session(&self, session_key: &str) -> Result<Option<Session>> {
        self.session_store.get_session(session_key).await
    }

    pub async fn delete_session(&self, session_data: &SessionData) -> Result<()> {
        self.episode_store
            .delete_by_session(&session_data.session_key)
            .await?;

        self.semantic_store
            .delete_by_set(&session_data.session_key)
            .await?;

        self.session_store
            .delete_session(&session_data.session_key)
            .await?;

        Ok(())
    }

    pub async fn search_sessions(&self, filter: Option<&FilterExpr>) -> Result<Vec<String>> {
        self.session_store.search_sessions(filter).await
    }

    pub async fn add_episodes(
        &self,
        session_data: &SessionData,
        entries: Vec<EpisodeEntry>,
        target_memories: &[MemoryType],
    ) -> Result<Vec<EpisodeId>> {
        let episodes = self
            .episode_store
            .add_episodes(&session_data.session_key, entries)
            .await?;

        let episode_ids: Vec<_> = episodes.iter().map(|e| e.uid.clone()).collect();

        if target_memories.contains(&MemoryType::Semantic)
            && self.config.semantic_memory.enabled
        {
            for episode_id in &episode_ids {
                self.semantic_store
                    .add_history(&session_data.session_key, episode_id)
                    .await?;
            }
        }

        Ok(episode_ids)
    }

    pub async fn query_search(
        &self,
        session_data: &SessionData,
        query: &str,
        target_memories: &[MemoryType],
        limit: Option<u32>,
        min_score: Option<f32>,
    ) -> Result<SearchResponse> {
        let mut response = SearchResponse {
            episodic_memory: None,
            semantic_memory: None,
        };

        if target_memories.contains(&MemoryType::Episodic)
            && self.config.episodic_memory.enabled
        {
            if let Some(ltm_config) = &self.config.episodic_memory.long_term_memory {
                if let Some(embedder) = self.embedders.get(&ltm_config.embedder) {
                    let query_embedding = embedder.embed(query).await?;

                    let episodes = self
                        .episode_store
                        .get_episodes_by_session(&session_data.session_key)
                        .await?;

                    if !episodes.is_empty() {
                        let episode_texts: Vec<String> =
                            episodes.iter().map(|e| e.content.clone()).collect();
                        let episode_embeddings = embedder.embed_batch(&episode_texts).await?;

                        let mut scored: Vec<(Episode, f32)> = episodes
                            .into_iter()
                            .zip(episode_embeddings.iter())
                            .map(|(ep, emb)| {
                                let score = cosine_similarity(&query_embedding, emb);
                                (ep, score)
                            })
                            .filter(|(_, score)| *score >= min_score.unwrap_or(0.0))
                            .collect();

                        scored.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });

                        if let Some(limit) = limit {
                            scored.truncate(limit as usize);
                        }

                        let (episodes, scores): (Vec<_>, Vec<_>) = scored.into_iter().unzip();
                        response.episodic_memory = Some(EpisodicSearchResult { episodes, scores });
                    }
                }
            }
        }

        if target_memories.contains(&MemoryType::Semantic)
            && self.config.semantic_memory.enabled
        {
            if let Some(embedder_name) = &self.config.semantic_memory.embedder {
                if let Some(embedder) = self.embedders.get(embedder_name) {
                    let query_embedding = embedder.embed(query).await?;

                    let results = self
                        .semantic_store
                        .search_by_embedding(
                            &query_embedding,
                            Some(&session_data.session_key),
                            limit,
                            min_score,
                        )
                        .await?;

                    response.semantic_memory = Some(results);
                }
            }
        }

        Ok(response)
    }

    pub async fn list_search(
        &self,
        session_data: &SessionData,
        target_memories: &[MemoryType],
        filter: Option<&FilterExpr>,
        page_size: Option<u32>,
        page_num: Option<u32>,
    ) -> Result<ListResults> {
        let mut results = ListResults {
            episodic_memory: None,
            semantic_memory: None,
        };

        if target_memories.contains(&MemoryType::Episodic) {
            let session_filter = FilterExpr::eq("session_key", session_data.session_key.as_str());
            let combined = match filter {
                Some(f) => session_filter.and(f.clone()),
                None => session_filter,
            };

            let episodes = self
                .episode_store
                .get_episodes(Some(&combined), page_size, page_num)
                .await?;
            results.episodic_memory = Some(episodes);
        }

        if target_memories.contains(&MemoryType::Semantic) {
            let features = self
                .semantic_store
                .get_features_by_set(&session_data.session_key)
                .await?;
            results.semantic_memory = Some(features);
        }

        Ok(results)
    }

    pub async fn episodes_count(
        &self,
        session_data: &SessionData,
        filter: Option<&FilterExpr>,
    ) -> Result<u64> {
        let session_filter = FilterExpr::eq("session_key", session_data.session_key.as_str());
        let combined = match filter {
            Some(f) => Some(session_filter.and(f.clone())),
            None => Some(session_filter),
        };

        self.episode_store.get_episodes_count(combined.as_ref()).await
    }

    pub async fn delete_episodes(
        &self,
        episode_ids: &[EpisodeId],
        _session_data: Option<&SessionData>,
    ) -> Result<u64> {
        self.episode_store.delete_episodes(episode_ids).await
    }

    pub async fn delete_features(&self, feature_ids: &[FeatureId]) -> Result<u64> {
        self.semantic_store.delete_features(feature_ids).await
    }

    pub async fn add_semantic_feature(
        &self,
        session_data: &SessionData,
        category: &str,
        tag: &str,
        feature_name: &str,
        value: &str,
    ) -> Result<FeatureId> {
        let embedding = if let Some(embedder_name) = &self.config.semantic_memory.embedder {
            if let Some(embedder) = self.embedders.get(embedder_name) {
                Some(embedder.embed(value).await?)
            } else {
                None
            }
        } else {
            None
        };

        self.semantic_store
            .add_feature(
                &session_data.session_key,
                category,
                tag,
                feature_name,
                value,
                embedding.as_deref(),
                None,
                None,
            )
            .await
    }
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

impl std::fmt::Debug for MemMachine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemMachine")
            .field("config", &self.config)
            .field("started", &self.started)
            .field("embedders", &self.embedders.keys().collect::<Vec<_>>())
            .field("llms", &self.llms.keys().collect::<Vec<_>>())
            .finish()
    }
}
