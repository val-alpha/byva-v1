//! BYVA conversation rendering, token-level streaming parsing, and tool-call conversion.
//!
//! Control IDs are emitted by Renderer. Plain content always goes through
//! Tokenizer::encode_content, so spellings such as <|call|> remain ordinary bytes.

use std::collections::{HashMap, HashSet};

use byva_tokenizer::{SpecialToken as S, TokenId, Tokenizer, TokenizerError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

mod untrusted;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssistantPart {
    Text(String),
    Think(String),
    Call(ToolCall),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    pub id: String,
    pub data: Option<Value>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MessageBody {
    Text(String),
    Assistant(Vec<AssistantPart>),
    ToolResult(ToolResult),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub body: MessageBody,
    /// True emits <|end|>. Tool result messages never end an assistant turn.
    pub end_turn: bool,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            body: MessageBody::Text(content.into()),
            end_turn: true,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            body: MessageBody::Text(content.into()),
            end_turn: true,
        }
    }

    pub fn assistant(parts: Vec<AssistantPart>, end_turn: bool) -> Self {
        Self {
            role: Role::Assistant,
            body: MessageBody::Assistant(parts),
            end_turn,
        }
    }

    pub fn tool_result(result: ToolResult) -> Self {
        Self {
            role: Role::Tool,
            body: MessageBody::ToolResult(result),
            end_turn: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Conversation {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
}

impl Conversation {
    pub fn new(messages: Vec<Message>, tools: Vec<ToolDefinition>) -> Self {
        Self { messages, tools }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenLikePolicy {
    /// Preserve token-like text as byte/BPE content IDs; it cannot become a control ID.
    EncodeAsText,
    /// Reject content containing a literal spelling of any BYVA control/reserved token.
    Reject,
}

#[derive(Debug, Clone, Copy)]
pub struct Renderer {
    token_like_policy: TokenLikePolicy,
}

impl Default for Renderer {
    fn default() -> Self {
        Self {
            token_like_policy: TokenLikePolicy::EncodeAsText,
        }
    }
}

#[derive(Debug, Error)]
pub enum FormatError {
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error("invalid message body for role {0:?}")]
    InvalidRoleBody(Role),
    #[error("tool definitions require exactly one system message")]
    ToolsRequireSystem,
    #[error("tool name is empty or duplicated: {0}")]
    DuplicateOrEmptyToolName(String),
    #[error("assistant call references an undeclared tool: {0}")]
    UnknownTool(String),
    #[error("call ID is empty or duplicated: {0}")]
    DuplicateOrEmptyCallId(String),
    #[error("tool result ID does not match an outstanding call: {0}")]
    UnmatchedToolResult(String),
    #[error("tool result must contain exactly one of data or error")]
    InvalidToolResult,
    #[error("tool result message cannot end the assistant turn")]
    InvalidToolResultEnding,
    #[error("content contains a literal BYVA control-token spelling")]
    TokenLikeContent,
    #[error("JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("stream parser error: {0}")]
    Parse(String),
}

#[derive(Serialize)]
struct CallPayload<'a> {
    id: &'a str,
    name: &'a str,
    args: &'a Value,
}

#[derive(Serialize)]
struct ResultPayload<'a> {
    id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

impl Renderer {
    pub fn new(token_like_policy: TokenLikePolicy) -> Self {
        Self { token_like_policy }
    }

    pub fn render(
        &self,
        conversation: &Conversation,
        tokenizer: &Tokenizer,
    ) -> Result<Vec<TokenId>, FormatError> {
        let mut output = vec![S::Bos.id()];
        let system_count = conversation
            .messages
            .iter()
            .filter(|message| message.role == Role::System)
            .count();

        if !conversation.tools.is_empty() && system_count != 1 {
            return Err(FormatError::ToolsRequireSystem);
        }

        let mut tool_names = HashMap::new();
        for tool in &conversation.tools {
            if tool.name.is_empty() || tool_names.insert(tool.name.as_str(), ()).is_some() {
                return Err(FormatError::DuplicateOrEmptyToolName(tool.name.clone()));
            }
        }

        let mut seen_call_ids = HashSet::new();
        let mut outstanding_calls = HashSet::new();
        let mut tools_written = false;

        for message in &conversation.messages {
            match (message.role, &message.body) {
                (Role::System, MessageBody::Text(content)) => {
                    output.push(S::System.id());
                    self.append_text(&mut output, content, tokenizer)?;
                    if !conversation.tools.is_empty() && !tools_written {
                        output.push(S::Tools.id());
                        output.extend(
                            tokenizer.encode_content(&serde_json::to_string(&conversation.tools)?),
                        );
                        output.push(S::EndTools.id());
                        tools_written = true;
                    }
                    output.push(S::End.id());
                }
                (Role::User, MessageBody::Text(content)) => {
                    output.push(S::User.id());
                    self.append_text(&mut output, content, tokenizer)?;
                    output.push(S::End.id());
                }
                (Role::Assistant, MessageBody::Assistant(parts)) => {
                    output.push(S::Assistant.id());
                    for part in parts {
                        match part {
                            AssistantPart::Text(content) => {
                                self.append_text(&mut output, content, tokenizer)?;
                            }
                            AssistantPart::Think(content) => {
                                output.push(S::Think.id());
                                self.append_text(&mut output, content, tokenizer)?;
                                output.push(S::EndThink.id());
                            }
                            AssistantPart::Call(call) => {
                                if call.id.is_empty() || !seen_call_ids.insert(call.id.clone()) {
                                    return Err(FormatError::DuplicateOrEmptyCallId(
                                        call.id.clone(),
                                    ));
                                }
                                if !tool_names.contains_key(call.name.as_str()) {
                                    return Err(FormatError::UnknownTool(call.name.clone()));
                                }
                                outstanding_calls.insert(call.id.clone());
                                output.push(S::Call.id());
                                output.extend(tokenizer.encode_content(&serde_json::to_string(
                                    &CallPayload {
                                        id: &call.id,
                                        name: &call.name,
                                        args: &call.args,
                                    },
                                )?));
                                output.push(S::EndCall.id());
                            }
                        }
                    }
                    if message.end_turn {
                        output.push(S::End.id());
                    }
                }
                (Role::Tool, MessageBody::ToolResult(result)) => {
                    if message.end_turn {
                        return Err(FormatError::InvalidToolResultEnding);
                    }
                    if result.id.is_empty()
                        || result.data.is_some() == result.error.is_some()
                        || !outstanding_calls.remove(&result.id)
                    {
                        return Err(
                            if result.id.is_empty()
                                || result.data.is_some() == result.error.is_some()
                            {
                                FormatError::InvalidToolResult
                            } else {
                                FormatError::UnmatchedToolResult(result.id.clone())
                            },
                        );
                    }
                    output.push(S::Tool.id());
                    output.push(S::Result.id());
                    output.extend(tokenizer.encode_content(&serde_json::to_string(
                        &ResultPayload {
                            id: &result.id,
                            data: result.data.as_ref(),
                            error: result.error.as_deref(),
                        },
                    )?));
                    output.push(S::EndResult.id());
                }
                (role, _) => return Err(FormatError::InvalidRoleBody(role)),
            }
        }

        Ok(output)
    }

    fn append_text(
        &self,
        output: &mut Vec<TokenId>,
        content: &str,
        tokenizer: &Tokenizer,
    ) -> Result<(), FormatError> {
        if self.token_like_policy == TokenLikePolicy::Reject
            && Tokenizer::contains_special_token_spelling(content)
        {
            return Err(FormatError::TokenLikeContent);
        }
        output.extend(tokenizer.encode_content(content));
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseEvent {
    ThinkDelta(String),
    TextDelta(String),
    CallStarted,
    CallComplete {
        id: String,
        name: String,
        args: Value,
    },
    TurnEnd,
    ParseError(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    Text,
    Think,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParseState {
    Message,
    CallJson,
    SkipTools,
    SkipResult,
    Failed,
    Finished,
}

#[derive(Debug, Clone)]
pub struct ParserConfig {
    pub max_call_bytes: usize,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            max_call_bytes: 64 * 1024,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ParsedCall {
    id: String,
    name: String,
    args: Value,
}

/// Incremental BYVA parser. Feed token IDs, not rendered text, so control tokens
/// cannot be confused with user-provided strings that merely look like markers.
pub struct StreamingParser {
    state: ParseState,
    role: Option<Role>,
    channel: Channel,
    text_bytes: Vec<u8>,
    call_bytes: Vec<u8>,
    seen_call_ids: HashSet<String>,
    max_call_bytes: usize,
    call_stop_seen: bool,
}

impl Default for StreamingParser {
    fn default() -> Self {
        Self::new(ParserConfig::default())
    }
}

impl StreamingParser {
    pub fn new(config: ParserConfig) -> Self {
        Self {
            state: ParseState::Message,
            role: None,
            channel: Channel::Text,
            text_bytes: Vec::new(),
            call_bytes: Vec::new(),
            seen_call_ids: HashSet::new(),
            max_call_bytes: config.max_call_bytes.max(1),
            call_stop_seen: false,
        }
    }

    pub fn push(&mut self, token: TokenId, tokenizer: &Tokenizer) -> Vec<ParseEvent> {
        let mut events = Vec::new();
        if self.state == ParseState::Finished {
            events.push(ParseEvent::ParseError(
                "token received after parser finish".to_owned(),
            ));
            return events;
        }
        if self.state == ParseState::Failed {
            return events;
        }

        if let Some(special) = S::from_id(token) {
            self.handle_special(special, &mut events);
        } else {
            if token.0 < byva_tokenizer::BYTE_TOKEN_BASE {
                self.fail("invalid control-token ID", &mut events);
                return events;
            }
            if token.0 >= tokenizer.vocab_size() {
                self.fail("content token ID is outside the tokenizer vocabulary", &mut events);
                return events;
            }
            self.call_stop_seen = false;
            match self.state {
                ParseState::CallJson => match tokenizer.token_bytes(token) {
                    Ok(bytes) => {
                        if self.call_bytes.len().saturating_add(bytes.len()) > self.max_call_bytes {
                            self.fail("tool-call JSON exceeds configured byte limit", &mut events);
                        } else {
                            self.call_bytes.extend(bytes);
                        }
                    }
                    Err(error) => self.fail(&error.to_string(), &mut events),
                },
                ParseState::Message if self.role == Some(Role::Assistant) => {
                    match tokenizer.token_bytes(token) {
                        Ok(bytes) => {
                            self.text_bytes.extend(bytes);
                            self.flush_text(false, &mut events);
                        }
                        Err(error) => self.fail(&error.to_string(), &mut events),
                    }
                }
                ParseState::Message | ParseState::SkipTools | ParseState::SkipResult => {}
                ParseState::Failed | ParseState::Finished => {}
            }
        }
        events
    }

    pub fn finish(&mut self) -> Vec<ParseEvent> {
        let mut events = Vec::new();
        if self.state == ParseState::Finished {
            return events;
        }
        if self.state == ParseState::Failed {
            return events;
        }

        if self.state == ParseState::CallJson {
            self.fail("stream ended inside a tool-call JSON block", &mut events);
        } else if self.state == ParseState::SkipTools || self.state == ParseState::SkipResult {
            self.fail("stream ended inside a tools/result block", &mut events);
        } else if self.role == Some(Role::Assistant) && !self.call_stop_seen {
            self.flush_text(true, &mut events);
            if self.state != ParseState::Failed {
                self.fail(
                    "assistant stream ended before <|end|> or <|/call|>",
                    &mut events,
                );
            }
        } else {
            self.flush_text(true, &mut events);
            if self.state != ParseState::Failed {
                self.state = ParseState::Finished;
            }
        }
        events
    }

    fn handle_special(&mut self, special: S, events: &mut Vec<ParseEvent>) {
        if self.state == ParseState::CallJson {
            if special == S::EndCall {
                self.complete_call(events);
            } else {
                self.fail("unexpected control token inside tool-call JSON", events);
            }
            return;
        }

        match self.state {
            ParseState::SkipTools => {
                if special == S::EndTools {
                    self.state = ParseState::Message;
                } else {
                    self.fail("unexpected control token inside tools JSON block", events);
                }
                return;
            }
            ParseState::SkipResult => {
                if special == S::EndResult {
                    self.state = ParseState::Message;
                    self.role = None;
                } else {
                    self.fail(
                        "unexpected control token inside tool result JSON block",
                        events,
                    );
                }
                return;
            }
            ParseState::Failed | ParseState::Finished => return,
            ParseState::Message => {}
            ParseState::CallJson => return,
        }

        match special {
            S::Bos | S::Pad => {}
            S::System => self.start_role(Role::System, events),
            S::User => self.start_role(Role::User, events),
            S::Assistant => self.start_role(Role::Assistant, events),
            S::Tool => self.start_role(Role::Tool, events),
            S::Think => {
                if self.role != Some(Role::Assistant) || self.channel == Channel::Think {
                    self.fail(
                        "think block is not valid in the current parser state",
                        events,
                    );
                } else {
                    self.flush_text(true, events);
                    if self.state != ParseState::Failed {
                        self.channel = Channel::Think;
                    }
                }
            }
            S::EndThink => {
                if self.role != Some(Role::Assistant) || self.channel != Channel::Think {
                    self.fail("unmatched </think> token", events);
                } else {
                    self.flush_text(true, events);
                    if self.state != ParseState::Failed {
                        self.channel = Channel::Text;
                    }
                }
            }
            S::Call => {
                if self.role != Some(Role::Assistant) {
                    self.fail(
                        "tool call is only valid inside an assistant message",
                        events,
                    );
                } else {
                    self.flush_text(true, events);
                    if self.state != ParseState::Failed {
                        self.call_bytes.clear();
                        self.state = ParseState::CallJson;
                        self.call_stop_seen = false;
                        events.push(ParseEvent::CallStarted);
                    }
                }
            }
            S::End => {
                self.flush_text(true, events);
                if self.state != ParseState::Failed {
                    if self.role == Some(Role::Assistant) {
                        events.push(ParseEvent::TurnEnd);
                    }
                    self.role = None;
                    self.channel = Channel::Text;
                    self.seen_call_ids.clear();
                    self.call_stop_seen = false;
                }
            }
            S::Tools => {
                if self.role != Some(Role::System) {
                    self.fail("tools block is only valid inside a system message", events);
                } else {
                    self.state = ParseState::SkipTools;
                }
            }
            S::Result => {
                if self.role != Some(Role::Tool) {
                    self.fail("result block is only valid inside a tool message", events);
                } else {
                    self.state = ParseState::SkipResult;
                }
            }
            S::EndCall => self.fail("unmatched </call> token", events),
            S::EndTools => self.fail("unmatched </tools> token", events),
            S::EndResult => self.fail("unmatched </result> token", events),
            S::Reserved(index) => self.fail(
                &format!("reserved control token {index} is not valid here"),
                events,
            ),
        }
    }

    fn start_role(&mut self, role: Role, events: &mut Vec<ParseEvent>) {
        self.flush_text(true, events);
        if self.state != ParseState::Failed {
            self.role = Some(role);
            self.channel = Channel::Text;
            self.call_stop_seen = false;
        }
    }

    fn complete_call(&mut self, events: &mut Vec<ParseEvent>) {
        if self.call_bytes.contains(&b'\n') || self.call_bytes.contains(&b'\r') {
            self.fail("tool-call JSON must occupy one physical line", events);
            return;
        }
        let parsed = serde_json::from_slice::<ParsedCall>(&self.call_bytes);
        match parsed {
            Ok(call) if !call.id.is_empty() && !call.name.is_empty() => {
                if !self.seen_call_ids.insert(call.id.clone()) {
                    self.fail("duplicate tool-call ID in assistant turn", events);
                    return;
                }
                events.push(ParseEvent::CallComplete {
                    id: call.id,
                    name: call.name,
                    args: call.args,
                });
                self.call_bytes.clear();
                self.state = ParseState::Message;
                self.call_stop_seen = true;
                self.channel = Channel::Text;
            }
            Ok(_) => self.fail("tool-call id and name must be non-empty strings", events),
            Err(error) => self.fail(&format!("invalid tool-call JSON: {error}"), events),
        }
    }

    fn flush_text(&mut self, boundary: bool, events: &mut Vec<ParseEvent>) {
        if self.text_bytes.is_empty() {
            return;
        }
        match std::str::from_utf8(&self.text_bytes) {
            Ok(_) => {
                let bytes = std::mem::take(&mut self.text_bytes);
                match String::from_utf8(bytes) {
                    Ok(text) => self.emit_text(text, events),
                    Err(error) => self.fail(&error.to_string(), events),
                }
            }
            Err(error) => {
                let valid_up_to = error.valid_up_to();
                let incomplete_tail = error.error_len().is_none();
                if incomplete_tail && !boundary {
                    let tail = self.text_bytes.split_off(valid_up_to);
                    let valid = std::mem::replace(&mut self.text_bytes, tail);
                    if let Ok(text) = String::from_utf8(valid) {
                        self.emit_text(text, events);
                    }
                } else {
                    self.fail("assistant text contains invalid or truncated UTF-8", events);
                }
            }
        }
    }

    fn emit_text(&self, text: String, events: &mut Vec<ParseEvent>) {
        if text.is_empty() || self.role != Some(Role::Assistant) {
            return;
        }
        match self.channel {
            Channel::Text => events.push(ParseEvent::TextDelta(text)),
            Channel::Think => events.push(ParseEvent::ThinkDelta(text)),
        }
    }

    fn fail(&mut self, message: &str, events: &mut Vec<ParseEvent>) {
        self.state = ParseState::Failed;
        events.push(ParseEvent::ParseError(message.to_owned()));
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAiFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpenAiFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpenAiAssistantOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OpenAiToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Collect parser events into a conventional OpenAI-style assistant message payload.
pub fn events_to_openai(events: &[ParseEvent]) -> Result<OpenAiAssistantOutput, FormatError> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    let mut ended = false;

    for event in events {
        match event {
            ParseEvent::TextDelta(text) => content.push_str(text),
            ParseEvent::ThinkDelta(text) => reasoning.push_str(text),
            ParseEvent::CallStarted => {}
            ParseEvent::CallComplete { id, name, args } => {
                tool_calls.push(OpenAiToolCall {
                    id: id.clone(),
                    call_type: "function".to_owned(),
                    function: OpenAiFunction {
                        name: name.clone(),
                        arguments: serde_json::to_string(args)?,
                    },
                });
            }
            ParseEvent::TurnEnd => ended = true,
            ParseEvent::ParseError(error) => return Err(FormatError::Parse(error.clone())),
        }
    }

    Ok(OpenAiAssistantOutput {
        content: (!content.is_empty()).then_some(content),
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
        finish_reason: if !tool_calls.is_empty() {
            Some("tool_calls".to_owned())
        } else if ended {
            Some("stop".to_owned())
        } else {
            None
        },
        tool_calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use byva_tokenizer::{BYTE_TOKEN_BASE, Merge};

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_merges(
            1,
            vec![Merge {
                left: BYTE_TOKEN_BASE + u32::from(b'o'),
                right: BYTE_TOKEN_BASE + u32::from(b'k'),
            }],
        )
        .expect("valid test tokenizer")
    }

    fn tool() -> ToolDefinition {
        ToolDefinition {
            name: "cuaca".to_owned(),
            description: "Weather lookup".to_owned(),
            parameters: serde_json::json!({"type":"object","properties":{"kota":{"type":"string"}}}),
        }
    }

    fn assistant_conversation() -> Conversation {
        Conversation::new(
            vec![
                Message::system("system prompt"),
                Message::user("hello"),
                Message::assistant(
                    vec![
                        AssistantPart::Think("need a tool".to_owned()),
                        AssistantPart::Call(ToolCall {
                            id: "c1".to_owned(),
                            name: "cuaca".to_owned(),
                            args: serde_json::json!({"kota":"Bandung"}),
                        }),
                    ],
                    false,
                ),
                Message::tool_result(ToolResult {
                    id: "c1".to_owned(),
                    data: Some(serde_json::json!({"suhu":24})),
                    error: None,
                }),
                Message::assistant(vec![AssistantPart::Text("Sekarang 24°C".to_owned())], true),
            ],
            vec![tool()],
        )
    }

    fn joined_delta(events: &[ParseEvent], thinking: bool) -> String {
        events
            .iter()
            .filter_map(|event| match (event, thinking) {
                (ParseEvent::ThinkDelta(text), true) => Some(text.as_str()),
                (ParseEvent::TextDelta(text), false) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn renderer_round_trip_preserves_assistant_events() {
        let tokenizer = tokenizer();
        let encoded = Renderer::default()
            .render(&assistant_conversation(), &tokenizer)
            .expect("conversation renders");
        let mut parser = StreamingParser::default();
        let mut events = Vec::new();
        for token in encoded {
            events.extend(parser.push(token, &tokenizer));
        }
        events.extend(parser.finish());

        assert_eq!(joined_delta(&events, true), "need a tool");
        assert!(events.iter().any(|event| matches!(event, ParseEvent::CallComplete { id, name, args } if id == "c1" && name == "cuaca" && args["kota"] == "Bandung")));
        assert_eq!(joined_delta(&events, false), "Sekarang 24°C");
        assert_eq!(
            events
                .iter()
                .filter(|event| **event == ParseEvent::TurnEnd)
                .count(),
            1
        );
    }

    #[test]
    fn parallel_calls_can_be_completed_out_of_order() {
        let tokenizer = tokenizer();
        let tools = vec![
            tool(),
            ToolDefinition {
                name: "peta".to_owned(),
                description: "Map lookup".to_owned(),
                parameters: serde_json::json!({"type":"object"}),
            },
        ];
        let conversation = Conversation::new(
            vec![
                Message::system("tools available"),
                Message::assistant(
                    vec![
                        AssistantPart::Call(ToolCall {
                            id: "c1".to_owned(),
                            name: "cuaca".to_owned(),
                            args: serde_json::json!({}),
                        }),
                        AssistantPart::Call(ToolCall {
                            id: "c2".to_owned(),
                            name: "peta".to_owned(),
                            args: serde_json::json!({}),
                        }),
                    ],
                    false,
                ),
                Message::tool_result(ToolResult {
                    id: "c2".to_owned(),
                    data: Some(serde_json::json!({"ok":true})),
                    error: None,
                }),
                Message::tool_result(ToolResult {
                    id: "c1".to_owned(),
                    data: Some(serde_json::json!({"ok":true})),
                    error: None,
                }),
                Message::assistant(vec![AssistantPart::Text("done".to_owned())], true),
            ],
            tools,
        );
        let tokens = Renderer::default()
            .render(&conversation, &tokenizer)
            .expect("parallel calls render");
        let mut parser = StreamingParser::default();
        let mut events = Vec::new();
        for token in tokens {
            events.extend(parser.push(token, &tokenizer));
        }
        events.extend(parser.finish());
        let calls: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                ParseEvent::CallComplete { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, vec!["c1", "c2"]);
    }

    #[test]
    fn parser_rejects_control_ids_from_untrusted_sources() {
        let tokenizer = tokenizer();
        let mut parser = StreamingParser::default();
        let events = parser.push_untrusted_token(S::Call.id(), &tokenizer);
        assert!(events.iter().any(|event| matches!(event, ParseEvent::ParseError(_))));
    }

    #[test]
    fn untrusted_marker_looking_text_is_plain_content() {
        let tokenizer = tokenizer();
        let mut parser = StreamingParser::default();
        let mut events = parser.push(S::Assistant.id(), &tokenizer);
        events.extend(parser.push_untrusted_text("literal <|call|>", &tokenizer));
        assert_eq!(joined_delta(&events, false), "literal <|call|>");
        assert!(!events.iter().any(|event| matches!(event, ParseEvent::CallStarted)));
    }

    #[test]
    fn rendered_token_looking_user_text_is_not_control() {
        let tokenizer = tokenizer();
        let conversation =
            Conversation::new(vec![Message::user("please print <|call|> exactly")], vec![]);
        let tokens = Renderer::default()
            .render(&conversation, &tokenizer)
            .expect("token-like content is encoded as text");
        assert!(!tokens.contains(&S::Call.id()));
        assert!(tokens.iter().any(|id| id.0 >= BYTE_TOKEN_BASE));
    }

    #[test]
    fn reject_policy_can_reject_token_looking_content() {
        let tokenizer = tokenizer();
        let conversation = Conversation::new(vec![Message::user("x <|bos|> y")], vec![]);
        let renderer = Renderer::new(TokenLikePolicy::Reject);
        assert!(matches!(
            renderer.render(&conversation, &tokenizer),
            Err(FormatError::TokenLikeContent)
        ));
    }

    #[test]
    fn parser_rejects_duplicate_call_ids() {
        let tokenizer = tokenizer();
        let mut parser = StreamingParser::default();
        let mut events = Vec::new();
        for token in [S::Assistant.id(), S::Call.id()] {
            events.extend(parser.push(token, &tokenizer));
        }
        for byte in br#"{"id":"same","name":"cuaca","args":{}}"# {
            events.extend(parser.push(TokenId(BYTE_TOKEN_BASE + u32::from(*byte)), &tokenizer));
        }
        events.extend(parser.push(S::EndCall.id(), &tokenizer));
        events.extend(parser.push(S::Call.id(), &tokenizer));
        for byte in br#"{"id":"same","name":"cuaca","args":{}}"# {
            events.extend(parser.push(TokenId(BYTE_TOKEN_BASE + u32::from(*byte)), &tokenizer));
        }
        events.extend(parser.push(S::EndCall.id(), &tokenizer));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, ParseEvent::ParseError(_)))
        );
    }

    #[test]
    fn parser_rejects_multiline_call_json() {
        let tokenizer = tokenizer();
        let mut parser = StreamingParser::default();
        let mut events = parser.push(S::Assistant.id(), &tokenizer);
        events.extend(parser.push(S::Call.id(), &tokenizer));
        for byte in b"{\n}" {
            events.extend(parser.push(TokenId(BYTE_TOKEN_BASE + u32::from(*byte)), &tokenizer));
        }
        events.extend(parser.push(S::EndCall.id(), &tokenizer));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, ParseEvent::ParseError(_)))
        );
    }

    #[test]
    fn parser_handles_every_truncated_prefix_without_panicking() {
        let tokenizer = tokenizer();
        let complete = Renderer::default()
            .render(&assistant_conversation(), &tokenizer)
            .expect("conversation renders");
        for end in 0..complete.len() {
            let mut parser = StreamingParser::default();
            let mut events = Vec::new();
            for token in complete.iter().take(end) {
                events.extend(parser.push(*token, &tokenizer));
            }
            events.extend(parser.finish());
            if end + 1 == complete.len() {
                assert!(events.iter().any(|event| matches!(event, ParseEvent::ParseError(_))));
            }
        }
    }

    #[test]
    fn parser_enforces_call_byte_limit() {
        let tokenizer = tokenizer();
        let mut parser = StreamingParser::new(ParserConfig { max_call_bytes: 4 });
        let mut events = parser.push(S::Assistant.id(), &tokenizer);
        events.extend(parser.push(S::Call.id(), &tokenizer));
        for byte in b"12345" {
            events.extend(parser.push(TokenId(BYTE_TOKEN_BASE + u32::from(*byte)), &tokenizer));
        }
        assert!(
            events
                .iter()
                .any(|event| matches!(event, ParseEvent::ParseError(_)))
        );
    }

    #[test]
    fn openai_converter_emits_tool_call_shape() {
        let events = vec![ParseEvent::CallComplete {
            id: "c1".to_owned(),
            name: "cuaca".to_owned(),
            args: serde_json::json!({"kota":"Bandung"}),
        }];
        let converted = events_to_openai(&events).expect("valid events");
        let json = serde_json::to_value(converted).expect("serializable");
        assert_eq!(json["tool_calls"][0]["id"], "c1");
        assert_eq!(json["tool_calls"][0]["type"], "function");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "cuaca");
        assert_eq!(json["finish_reason"], "tool_calls");
    }

    #[test]
    fn tokenizer_contract_keeps_content_ids_above_control_range() {
        let tokenizer = tokenizer();
        let encoded = tokenizer.encode_content("<|result|>");
        assert!(encoded.iter().all(|token| token.0 >= BYTE_TOKEN_BASE));
    }

    proptest::proptest! {
        #[test]
        fn renderer_keeps_marker_looking_user_text_as_content(
            input in proptest::string::string_regex(".{0,80}").expect("valid regex")
        ) {
            let tokenizer = tokenizer();
            let conversation = Conversation::new(vec![Message::user(input)], vec![]);
            let rendered = Renderer::default().render(&conversation, &tokenizer);
            proptest::prop_assert!(rendered.is_ok());
            let rendered = rendered.unwrap_or_default();
            let controls: Vec<TokenId> = rendered
                .iter()
                .copied()
                .filter(|id| S::from_id(*id).is_some())
                .collect();
            proptest::prop_assert_eq!(
                controls,
                vec![S::Bos.id(), S::User.id(), S::End.id()]
            );
        }
    }
}
