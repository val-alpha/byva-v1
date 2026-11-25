# BYVA Native Conversation Format — Version 1

BYVA is a token-level message protocol. It is not DSML and is not parsed by scanning decoded text for marker strings. Control markers are dedicated token IDs. User text, web text, dataset text, and tool output are content bytes and must never be promoted into control IDs.

## 1. Token ID registry

The tokenizer version locks this registry. IDs are zero-based and contiguous.

| ID | Token |
|---:|---|
| 0 | <|bos|> |
| 1 | <|end|> |
| 2 | <|pad|> |
| 3 | <|system|> |
| 4 | <|user|> |
| 5 | <|assistant|> |
| 6 | <|tool|> |
| 7 | <|think|> |
| 8 | <|/think|> |
| 9 | <|tools|> |
| 10 | <|/tools|> |
| 11 | <|call|> |
| 12 | <|/call|> |
| 13 | <|result|> |
| 14 | <|/result|> |
| 15–46 | <|reserved_0|> through <|reserved_31|> |
| 47–302 | byte values 0x00–0xFF |
| 303 onward | BPE merge tokens, in fixed rank order |

A tokenizer artifact stores its version, vocabulary size, and ordered merge table. A released tokenizer version is immutable: changing token IDs, merge order, or normalization behavior requires a new version. Supported production vocabulary sizes are 32,768 through 49,152.

## 2. Message grammar

The following is an explanatory grammar. Content means a sequence of byte/BPE token IDs; it is not a string scanned for marker spellings.

~~~text
conversation := <|bos|> message*
message      := system | user | assistant | tool_result

system       := <|system|> content [tools_block] <|end|>
user         := <|user|> content <|end|>
assistant    := <|assistant|> assistant_part* [<|end|>]
assistant_part := content | think_block | call_block
think_block  := <|think|> content <|/think|>
tools_block  := <|tools|> compact_json_array <|/tools|>
call_block   := <|call|> compact_call_json <|/call|>
tool_result  := <|tool|> <|result|> compact_result_json <|/result|>
~~~

The assistant message omits <|end|> while yielding control for a tool call. An assistant turn may contain several think/call/result/assistant continuation cycles. Thinking is optional; direct text or tool calls are valid without a think block.

## 3. JSON contracts

Tools appear only in the system message, between the tools markers. The payload is a compact JSON array of definitions:

~~~json
[{"name":"cuaca","description":"Weather lookup","parameters":{"type":"object","properties":{"kota":{"type":"string"}}}}]
~~~

A call is one compact RFC 8259 JSON object on a single physical line:

~~~json
{"id":"c1","name":"cuaca","args":{"kota":"Bandung"}}
~~~

A result matches the call ID. Success and failure shapes are:

~~~json
{"id":"c1","data":{"suhu":24}}
{"id":"c1","error":"provider unavailable"}
~~~

Newlines inside JSON strings must be escaped as backslash-n. Literal carriage-return or line-feed bytes inside a call/result JSON payload are invalid. IDs are non-empty and unique within the assistant turn. Multiple calls are represented by consecutive call blocks. Each result is correlated by ID, so tool execution may complete in any order.

The renderer validates declared tool names, duplicate call IDs, outstanding call/result correlation, and the success/error result shape. Full schema-constrained generation is a runtime responsibility and is implemented in the later inference phase.

## 4. Stop-token semantics

- <|/call|> is the generation stop point for a tool call. The runtime dispatches the validated call, appends a tool result message, then resumes generation with an assistant prefix.
- <|end|> ends the assistant turn. It is not a substitute for the call stop marker.
- <|/think|>, <|/tools|>, and <|/result|> close their respective blocks; they do not end the assistant turn.
- A tool result block does not itself emit <|end|>.

## 5. Renderer contract

Renderer accepts typed messages, typed assistant segments, tool definitions, and tool results. It emits control IDs only by selecting the SpecialToken enum. All human-readable content is passed through Tokenizer::encode_content.

The default token-like content policy is EncodeAsText. A string such as the literal characters <|call|> is NFC-normalized and encoded into byte/BPE IDs; it does not create token ID 11. Applications that prefer rejection can choose TokenLikePolicy::Reject. Both policies prevent control-token injection.

The renderer starts a conversation with <|bos|>. System and user messages end with <|end|>. Assistant messages end only when the caller marks the turn complete. Tool results use the dedicated tool/result markers. Tool definitions are serialized as compact JSON and placed inside the single system message.

Renderer rejects tool calls whose names are absent from the system tool list, duplicate/empty call IDs, unmatched result IDs, and invalid result envelopes. It permits outstanding calls at the end of a prompt because the next operation may be tool execution.

## 6. Streaming parser

StreamingParser consumes token IDs incrementally. It has explicit states for ordinary message content, call JSON, ignored tools JSON, ignored tool-result JSON, failed input, and finished input. It does not search decoded strings for control markers.

Events:

| Event | Meaning |
|---|---|
| ThinkDelta(text) | Decoded assistant reasoning text |
| TextDelta(text) | Decoded assistant user-facing text |
| CallStarted | A call block has begun |
| CallComplete{id,name,args} | The single-line JSON call parsed successfully |
| TurnEnd | An assistant <|end|> was received |
| ParseError(reason) | Invalid control sequence, malformed JSON, invalid UTF-8, duplicate ID, limit exceeded, or truncated block |

Text bytes are buffered across token boundaries so a multi-byte UTF-8 character split between tokens is emitted only when complete. A malformed UTF-8 sequence is rejected. The parser applies a configurable upper bound to call JSON bytes (64 KiB by default). It rejects special control IDs in unexpected positions, calls outside an assistant message, duplicate IDs within a turn, multiline call JSON, invalid or missing call fields, and unmatched closing markers.

Call/result tool output is not surfaced as assistant text. Tool definitions and result payloads are skipped as structured context. A successful call close is a valid generation stop point; finish() accepts that boundary. Other incomplete assistant messages or open blocks produce ParseError rather than panic. After a parse error the parser enters a failed state and ignores further input.

For untrusted data, use push_untrusted_token or push_untrusted_text. The token API rejects every ID below the byte-token range, including control and reserved IDs. The text API always encodes external text as content first, so a string such as <|call|> remains ordinary text. Never construct control IDs by parsing untrusted marker-looking text.

The parser expects model/renderer token IDs for trusted protocol structure. Untrusted text must enter through the explicit untrusted APIs or the typed renderer.

## 7. OpenAI-style conversion

events_to_openai collects parser events into an assistant message object:
- TextDelta becomes content.
- ThinkDelta becomes the optional reasoning field.
- CallComplete becomes tool_calls entries with type=function and function.arguments containing compact JSON text.
- A tool-call response uses finish_reason=tool_calls; a completed assistant turn uses finish_reason=stop.

The conversion layer does not execute tools. Dispatch, authorization, timeout policy, schema-constrained decoding, and tool result isolation belong to the runtime/application boundary.

## 8. Training serialization contract

The renderer's typed output is the canonical input representation for later training conversion. The planned loss mask is applied to model-authored tokens only: assistant think/text/call contents and the assistant control markers think, /think, call, /call, and end. System and user content, role headers, tool headers, and complete tool/result payloads are masked. The training implementation is introduced in the pretraining/SFT phases; this document defines the format contract only.
