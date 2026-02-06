pub mod openai;
pub mod traits;

pub use openai::OpenAILlm;
pub use traits::{ChatMessage, ChatRole, Llm};
