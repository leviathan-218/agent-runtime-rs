use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::message::{ContentBlock, Message, Role};
use crate::model::{
    ChatResponse, FinishReason, GenerateOptions, Model, ModelError, ToolSchema, Usage,
    require_messages,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_ENDPOINT: &str = "/chat/completions";
const DEFAULT_MODEL: &str = "gpt-4o-mini";

/// Chat Completions client for OpenAI and compatible providers.
///
/// Synchronous for now. The agent runtime will switch this to async later.
pub struct OpenAiCompatibleModel {
    model_name: String,
    api_key: String,
    base_url: String,
    endpoint_path: String,
    http: ureq::Agent,
}

impl OpenAiCompatibleModel {
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            endpoint_path: DEFAULT_ENDPOINT.to_string(),
            http: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(60))
                .build(),
        }
    }

    pub fn from_env() -> Result<Self, ModelError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ModelError::Config("OPENAI_API_KEY is not set".into()))?;
        let model_name =
            std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let mut model = Self::new(model_name, api_key);
        if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
            model = model.base_url(base_url);
        }
        Ok(model)
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn endpoint_path(mut self, endpoint_path: impl Into<String>) -> Self {
        self.endpoint_path = endpoint_path.into();
        self
    }

    fn url(&self) -> String {
        join_url(&self.base_url, &self.endpoint_path)
    }
}

impl fmt::Debug for OpenAiCompatibleModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiCompatibleModel")
            .field("model_name", &self.model_name)
            .field("base_url", &self.base_url)
            .field("endpoint_path", &self.endpoint_path)
            .field("api_key", &"***")
            .finish()
    }
}

impl Model for OpenAiCompatibleModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn generate(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &GenerateOptions,
    ) -> Result<ChatResponse, ModelError> {
        require_messages(messages)?;
        let body = build_request(&self.model_name, messages, tools, options)?;

        match self
            .http
            .post(&self.url())
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .send_json(&body)
        {
            Ok(response) => {
                let parsed: CompletionsResponse = response
                    .into_json()
                    .map_err(|error| ModelError::InvalidResponse(error.to_string()))?;
                parse_response(parsed)
            }
            Err(ureq::Error::Status(status, response)) => {
                let body = response.into_string().unwrap_or_default();
                Err(ModelError::Api { status, body })
            }
            Err(ureq::Error::Transport(error)) => Err(ModelError::Transport(error.to_string())),
        }
    }
}

fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}{}{}",
        base.trim_end_matches('/'),
        "/",
        path.trim_start_matches('/')
    )
}

#[derive(Debug, Serialize)]
struct CompletionsRequest {
    model: String,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct WireMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct WireToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: WireFunction,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct WireFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct WireTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireToolFunction,
}

#[derive(Debug, Serialize)]
struct WireToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct CompletionsResponse {
    choices: Vec<WireChoice>,
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct WireChoice {
    message: WireChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireChoiceMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
}

#[derive(Debug, Deserialize)]
struct WireUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

fn build_request(
    model_name: &str,
    messages: &[Message],
    tools: &[ToolSchema],
    options: &GenerateOptions,
) -> Result<CompletionsRequest, ModelError> {
    Ok(CompletionsRequest {
        model: model_name.to_string(),
        messages: messages.iter().map(to_wire_message).collect(),
        temperature: options.temperature,
        max_tokens: options.max_tokens,
        tools: tools
            .iter()
            .map(to_wire_tool)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn to_wire_message(message: &Message) -> WireMessage {
    let text = message.text();
    let tool_calls: Vec<WireToolCall> = message
        .tool_calls()
        .into_iter()
        .map(|call| WireToolCall {
            id: call.id.clone(),
            kind: "function".to_string(),
            function: WireFunction {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
        })
        .collect();

    match message.role {
        Role::Tool => {
            let tool_call_id = message.content.iter().find_map(|block| match block {
                ContentBlock::ToolResult(result) => Some(result.tool_call_id.clone()),
                _ => None,
            });
            let output = message.content.iter().find_map(|block| match block {
                ContentBlock::ToolResult(result) => Some(result.output.clone()),
                _ => None,
            });
            WireMessage {
                role: "tool".to_string(),
                content: Some(output.unwrap_or(text)),
                tool_calls: None,
                tool_call_id,
            }
        }
        role => WireMessage {
            role: role.to_string(),
            content: if text.is_empty() { None } else { Some(text) },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
        },
    }
}

fn to_wire_tool(schema: &ToolSchema) -> Result<WireTool, ModelError> {
    let parameters = if schema.parameters.trim().is_empty() {
        serde_json::json!({ "type": "object", "properties": {} })
    } else {
        serde_json::from_str(&schema.parameters).map_err(|error| {
            ModelError::InvalidResponse(format!("tool '{}' parameters: {error}", schema.name))
        })?
    };

    Ok(WireTool {
        kind: "function",
        function: WireToolFunction {
            name: schema.name.clone(),
            description: schema.description.clone(),
            parameters,
        },
    })
}

fn parse_response(response: CompletionsResponse) -> Result<ChatResponse, ModelError> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| ModelError::InvalidResponse("provider returned no choices".into()))?;

    let mut content = Vec::new();
    if let Some(text) = choice.message.content
        && !text.is_empty()
    {
        content.push(ContentBlock::text(text));
    }
    if let Some(tool_calls) = choice.message.tool_calls {
        for call in tool_calls {
            content.push(ContentBlock::tool_call(
                call.id,
                call.function.name,
                call.function.arguments,
            ));
        }
    }

    let usage = response
        .usage
        .map(|usage| Usage::new(usage.prompt_tokens, usage.completion_tokens));
    let finish_reason = choice.finish_reason.as_deref().map(FinishReason::parse);

    Ok(ChatResponse {
        content,
        usage,
        finish_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_base_url_and_path() {
        assert_eq!(
            join_url("https://api.openai.com/v1/", "/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn maps_user_and_system_messages() {
        let request = build_request(
            "gpt-4o-mini",
            &[Message::system("be brief"), Message::user("hello")],
            &[],
            &GenerateOptions::default(),
        )
        .unwrap();

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["model"], "gpt-4o-mini");
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][0]["content"], "be brief");
        assert_eq!(json["messages"][1]["role"], "user");
        assert_eq!(json["messages"][1]["content"], "hello");
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn maps_assistant_tool_call_and_tool_result() {
        let assistant = Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call(
                "call_1",
                "search",
                r#"{"q":"rust"}"#,
            )],
        );
        let tool = Message::tool_result("call_1", "ok");
        let request = build_request(
            "gpt-4o-mini",
            &[assistant, tool],
            &[ToolSchema::new(
                "search",
                "search the web",
                r#"{"type":"object","properties":{"q":{"type":"string"}}}"#,
            )],
            &GenerateOptions {
                temperature: Some(0.2),
                max_tokens: Some(64),
            },
        )
        .unwrap();

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["messages"][0]["role"], "assistant");
        assert_eq!(json["messages"][0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            json["messages"][0]["tool_calls"][0]["function"]["name"],
            "search"
        );
        assert_eq!(json["messages"][1]["role"], "tool");
        assert_eq!(json["messages"][1]["tool_call_id"], "call_1");
        assert_eq!(json["messages"][1]["content"], "ok");
        assert_eq!(json["tools"][0]["function"]["name"], "search");
        assert_eq!(json["temperature"], 0.2);
        assert_eq!(json["max_tokens"], 64);
    }

    #[test]
    fn parses_text_response() {
        let raw = r#"{
            "choices": [{
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 1 }
        }"#;
        let parsed: CompletionsResponse = serde_json::from_str(raw).unwrap();
        let response = parse_response(parsed).unwrap();
        assert_eq!(response.text_content(), "hi");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            response.usage,
            Some(Usage {
                input_tokens: 3,
                output_tokens: 1
            })
        );
    }

    #[test]
    fn parses_tool_calls() {
        let raw = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "search", "arguments": "{\"q\":\"rust\"}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;
        let parsed: CompletionsResponse = serde_json::from_str(raw).unwrap();
        let response = parse_response(parsed).unwrap();
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(response.content.len(), 1);
        match &response.content[0] {
            ContentBlock::ToolCall(call) => {
                assert_eq!(call.name, "search");
                assert_eq!(call.arguments, r#"{"q":"rust"}"#);
            }
            other => panic!("expected tool call, got {other:?}"),
        }
    }

    #[test]
    fn empty_choices_is_invalid() {
        let parsed: CompletionsResponse = serde_json::from_str(r#"{"choices":[]}"#).unwrap();
        let err = parse_response(parsed).unwrap_err();
        assert!(matches!(err, ModelError::InvalidResponse(_)));
    }

    #[test]
    fn debug_hides_api_key() {
        let model = OpenAiCompatibleModel::new("gpt-4o-mini", "sk-secret");
        let debug = format!("{model:?}");
        assert!(debug.contains("***"));
        assert!(!debug.contains("sk-secret"));
    }
}
