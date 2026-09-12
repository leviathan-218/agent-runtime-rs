//! Language-model abstraction.
//!
//! [`Model::generate`] is synchronous and returns [`Result`]. Streaming belongs
//! to the event system, which is not here yet.

mod mock;
mod openai;

pub use mock::MockModel;
pub use openai::OpenAiCompatibleModel;

use crate::message::{ContentBlock, Message};

/// A language model that turns a conversation into the next assistant message.
///
/// The trait is object-safe on purpose: both `fn f<M: Model>(m: &M)` and
/// `fn f(m: &dyn Model)` work. We use one shared [`ModelError`] instead of an
/// associated `type Error`, which would break `dyn Model`.
pub trait Model {
    fn name(&self) -> &str;

    fn generate(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &GenerateOptions,
    ) -> Result<ChatResponse, ModelError>;

    /// Convenience for a single user prompt and no tools.
    fn generate_text(&self, prompt: &str) -> Result<ChatResponse, ModelError> {
        self.generate(&[Message::user(prompt)], &[], &GenerateOptions::default())
    }
}

/// Description of a tool the model is allowed to call.
///
/// `parameters` is a JSON Schema object as text. The tool system will replace
/// this with a typed value later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: String,
}

impl ToolSchema {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: parameters.into(),
        }
    }
}

/// Per-request generation knobs. `None` means "use the provider default".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerateOptions {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
}

/// Why the model stopped producing tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    Other(String),
}

impl FinishReason {
    pub fn parse(raw: &str) -> Self {
        match raw {
            "stop" => Self::Stop,
            "tool_calls" => Self::ToolCalls,
            "length" => Self::Length,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Token counts reported by a provider. Optional: mocks usually skip this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl Usage {
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }

    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

/// One model completion. Convert to a conversation [`Message`] with
/// [`ChatResponse::into_message`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatResponse {
    pub content: Vec<ContentBlock>,
    pub usage: Option<Usage>,
    pub finish_reason: Option<FinishReason>,
}

impl ChatResponse {
    pub fn new(content: Vec<ContentBlock>) -> Self {
        Self {
            content,
            usage: None,
            finish_reason: None,
        }
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
            usage: None,
            finish_reason: Some(FinishReason::Stop),
        }
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = Some(reason);
        self
    }

    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    pub fn into_message(self) -> Message {
        Message::new(crate::message::Role::Assistant, self.content)
    }
}

/// Failures from [`Model::generate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    EmptyMessages,
    Exhausted,
    Config(String),
    Transport(String),
    Api { status: u16, body: String },
    InvalidResponse(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::EmptyMessages => write!(f, "prompt has no messages"),
            ModelError::Exhausted => write!(f, "mock model has no more replies"),
            ModelError::Config(message) => write!(f, "model config: {message}"),
            ModelError::Transport(message) => write!(f, "transport error: {message}"),
            ModelError::Api { status, body } => {
                write!(f, "provider returned HTTP {status}: {body}")
            }
            ModelError::InvalidResponse(message) => write!(f, "invalid response: {message}"),
        }
    }
}

impl std::error::Error for ModelError {}

pub(crate) fn require_messages(messages: &[Message]) -> Result<(), ModelError> {
    if messages.is_empty() {
        Err(ModelError::EmptyMessages)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call_dyn(model: &dyn Model, prompt: &str) -> Result<String, ModelError> {
        Ok(model.generate_text(prompt)?.text_content())
    }

    fn call_generic<M: Model>(model: &M, prompt: &str) -> Result<String, ModelError> {
        Ok(model.generate_text(prompt)?.text_content())
    }

    #[test]
    fn generic_and_dyn_see_the_same_model() {
        let model = MockModel::always("mock", "pong");
        assert_eq!(call_generic(&model, "ping").unwrap(), "pong");
        assert_eq!(call_dyn(&model, "ping").unwrap(), "pong");
        assert_eq!(model.name(), "mock");
    }

    #[test]
    fn empty_messages_are_an_error() {
        let model = MockModel::always("mock", "unused");
        let err = model
            .generate(&[], &[], &GenerateOptions::default())
            .unwrap_err();
        assert_eq!(err, ModelError::EmptyMessages);
    }

    #[test]
    fn into_message_takes_ownership_of_content() {
        let response = ChatResponse::text("hi");
        let message = response.into_message();
        assert_eq!(message.role, crate::message::Role::Assistant);
        assert_eq!(message.text(), "hi");
    }
}
