//! Core types for the agent runtime.
//!
//! Starts with the conversation [`message`] model and a synchronous [`model::Model`]
//! trait. The agent loop is not here yet.

pub mod message;
pub mod model;

pub use message::{ContentBlock, Message, Role, TextBlock, ToolCallBlock, ToolResultBlock};
pub use model::{
    ChatResponse, FinishReason, GenerateOptions, MockModel, Model, ModelError,
    OpenAiCompatibleModel, ToolSchema, Usage,
};
