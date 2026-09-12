//! Conversation messages.
//!
//! One [`Message`] with a [`Role`] and a list of [`ContentBlock`]s. Passing a
//! [`Message`] by value moves it; pass `&Message` when the caller still needs it.

use std::fmt;

/// Who produced a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::System => write!(f, "system"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

/// One piece of content inside a [`Message`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBlock {
    Text(TextBlock),
    ToolCall(ToolCallBlock),
    ToolResult(ToolResultBlock),
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(TextBlock::new(text))
    }

    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::ToolCall(ToolCallBlock::new(id, name, arguments))
    }

    pub fn tool_result(tool_call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self::ToolResult(ToolResultBlock::new(tool_call_id, output))
    }
}

/// Plain text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextBlock {
    pub text: String,
}

impl TextBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// A request from the model to run a tool.
///
/// `arguments` is JSON text for now. A typed `serde_json::Value` comes later
/// with the tool system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallBlock {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ToolCallBlock {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }
}

/// The output of a tool invocation, sent back to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResultBlock {
    pub tool_call_id: String,
    pub output: String,
}

impl ToolResultBlock {
    pub fn new(tool_call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            output: output.into(),
        }
    }
}

/// A single turn in a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, vec![ContentBlock::text(text)])
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(Role::Assistant, vec![ContentBlock::text(text)])
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self::new(Role::System, vec![ContentBlock::text(text)])
    }

    pub fn tool_result(tool_call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self::new(
            Role::Tool,
            vec![ContentBlock::tool_result(tool_call_id, output)],
        )
    }

    /// Concatenate every text block. Tool calls and results are skipped.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    pub fn tool_calls(&self) -> Vec<&ToolCallBlock> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect()
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.role, self.text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_message_holds_text() {
        let message = Message::user("hello");
        assert_eq!(message.role, Role::User);
        assert_eq!(message.text(), "hello");
    }

    #[test]
    fn concatenates_text_blocks_and_skips_tool_calls() {
        let message = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("I'll look that up. "),
                ContentBlock::tool_call("call_1", "search", r#"{"q":"rust"}"#),
                ContentBlock::text("Done."),
            ],
        );

        assert_eq!(message.text(), "I'll look that up. Done.");
        assert_eq!(message.tool_calls().len(), 1);
        assert_eq!(message.tool_calls()[0].name, "search");
    }

    #[test]
    fn tool_result_message() {
        let message = Message::tool_result("call_1", "4");
        assert_eq!(message.role, Role::Tool);
        assert_eq!(message.text(), "");
        assert_eq!(
            message.content,
            vec![ContentBlock::tool_result("call_1", "4")]
        );
    }

    #[test]
    fn clone_is_independent() {
        let original = Message::user("hello");
        let mut copy = original.clone();
        copy.content[0] = ContentBlock::text("changed");

        assert_eq!(original.text(), "hello");
        assert_eq!(copy.text(), "changed");
    }

    #[test]
    fn role_is_copy() {
        let role = Role::User;
        let also = role;
        assert_eq!(role, also);
        assert_eq!(role, Role::User);
    }
}
