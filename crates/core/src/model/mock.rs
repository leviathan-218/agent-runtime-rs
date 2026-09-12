use std::cell::RefCell;
use std::collections::VecDeque;

use crate::message::{ContentBlock, Message};
use crate::model::{
    ChatResponse, GenerateOptions, Model, ModelError, ToolSchema, require_messages,
};

/// Scripted model for tests and examples. No network.
///
/// Queued replies are stored in a [`RefCell`] so [`Model::generate`] can pop
/// them through `&self`. Not `Sync`; share across threads later with a `Mutex`.
pub struct MockModel {
    name: String,
    replies: RefCell<VecDeque<Result<ChatResponse, ModelError>>>,
    fallback: Option<ChatResponse>,
}

impl MockModel {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            replies: RefCell::new(VecDeque::new()),
            fallback: None,
        }
    }

    /// Always return the same text once the queue is empty.
    pub fn always(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            replies: RefCell::new(VecDeque::new()),
            fallback: Some(ChatResponse::text(text)),
        }
    }

    pub fn reply_text(self, text: impl Into<String>) -> Self {
        self.replies
            .borrow_mut()
            .push_back(Ok(ChatResponse::text(text)));
        self
    }

    pub fn reply(self, response: ChatResponse) -> Self {
        self.replies.borrow_mut().push_back(Ok(response));
        self
    }

    pub fn reply_error(self, error: ModelError) -> Self {
        self.replies.borrow_mut().push_back(Err(error));
        self
    }

    pub fn reply_tool_call(
        self,
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        self.reply(
            ChatResponse::new(vec![ContentBlock::tool_call(id, name, arguments)])
                .with_finish_reason(crate::model::FinishReason::ToolCalls),
        )
    }
}

impl Model for MockModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate(
        &self,
        messages: &[Message],
        _tools: &[ToolSchema],
        _options: &GenerateOptions,
    ) -> Result<ChatResponse, ModelError> {
        require_messages(messages)?;

        if let Some(next) = self.replies.borrow_mut().pop_front() {
            return next;
        }

        if let Some(fallback) = &self.fallback {
            return Ok(fallback.clone());
        }

        Err(ModelError::Exhausted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FinishReason;

    #[test]
    fn queued_replies_are_consumed_in_order() {
        let model = MockModel::new("scripted")
            .reply_text("one")
            .reply_text("two");

        let first = model.generate_text("ignored").unwrap();
        let second = model.generate_text("ignored").unwrap();
        let third = model.generate_text("ignored").unwrap_err();

        assert_eq!(first.text_content(), "one");
        assert_eq!(second.text_content(), "two");
        assert_eq!(third, ModelError::Exhausted);
    }

    #[test]
    fn fallback_after_queue() {
        let model = MockModel::always("echo", "fallback").reply_text("queued");

        assert_eq!(model.generate_text("a").unwrap().text_content(), "queued");
        assert_eq!(model.generate_text("b").unwrap().text_content(), "fallback");
        assert_eq!(model.generate_text("c").unwrap().text_content(), "fallback");
    }

    #[test]
    fn can_script_a_tool_call() {
        let model = MockModel::new("tools").reply_tool_call("call_1", "search", r#"{"q":"rust"}"#);

        let response = model.generate_text("find rust").unwrap();
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        match &response.content[..] {
            [ContentBlock::ToolCall(call)] => {
                assert_eq!(call.id, "call_1");
                assert_eq!(call.name, "search");
                assert_eq!(call.arguments, r#"{"q":"rust"}"#);
            }
            other => panic!("expected one tool call, got {other:?}"),
        }
    }

    #[test]
    fn can_script_an_error() {
        let model = MockModel::new("fail").reply_error(ModelError::Transport("down".into()));
        let err = model.generate_text("hi").unwrap_err();
        assert_eq!(err, ModelError::Transport("down".into()));
    }
}
