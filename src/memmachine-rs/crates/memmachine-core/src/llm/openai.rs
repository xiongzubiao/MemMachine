use super::traits::{ChatMessage, ChatRole, Llm};
use crate::error::{MemMachineError, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use async_trait::async_trait;

pub struct OpenAILlm {
    client: Client<OpenAIConfig>,
    model: String,
    name: String,
    default_temperature: f32,
}

impl OpenAILlm {
    pub fn new(
        name: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let config = OpenAIConfig::new().with_api_key(api_key.into());
        let client = Client::with_config(config);

        Self {
            client,
            model: model.into(),
            name: name.into(),
            default_temperature: 0.7,
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.default_temperature = temperature;
        self
    }

    fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<ChatCompletionRequestMessage>> {
        messages
            .iter()
            .map(|msg| {
                let result: Result<ChatCompletionRequestMessage> = match msg.role {
                    ChatRole::System => Ok(ChatCompletionRequestSystemMessageArgs::default()
                        .content(msg.content.clone())
                        .build()
                        .map_err(|e| MemMachineError::Llm(e.to_string()))?
                        .into()),
                    ChatRole::User => Ok(ChatCompletionRequestUserMessageArgs::default()
                        .content(msg.content.clone())
                        .build()
                        .map_err(|e| MemMachineError::Llm(e.to_string()))?
                        .into()),
                    ChatRole::Assistant => Ok(ChatCompletionRequestAssistantMessageArgs::default()
                        .content(msg.content.clone())
                        .build()
                        .map_err(|e| MemMachineError::Llm(e.to_string()))?
                        .into()),
                };
                result
            })
            .collect()
    }
}

#[async_trait]
impl Llm for OpenAILlm {
    async fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        self.chat_with_temperature(messages, self.default_temperature)
            .await
    }

    async fn chat_with_temperature(
        &self,
        messages: &[ChatMessage],
        temperature: f32,
    ) -> Result<String> {
        let openai_messages = Self::convert_messages(messages)?;

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(openai_messages)
            .temperature(temperature)
            .build()
            .map_err(|e| MemMachineError::Llm(e.to_string()))?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| MemMachineError::Llm(e.to_string()))?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| MemMachineError::Llm("No response choices".to_string()))?;

        choice
            .message
            .content
            .clone()
            .ok_or_else(|| MemMachineError::Llm("Empty response content".to_string()))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model
    }
}

impl std::fmt::Debug for OpenAILlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAILlm")
            .field("name", &self.name)
            .field("model", &self.model)
            .field("default_temperature", &self.default_temperature)
            .finish()
    }
}
