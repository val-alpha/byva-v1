//! Explicit APIs for accepting text/token streams from untrusted sources.

use byva_tokenizer::{BYTE_TOKEN_BASE, TokenId, Tokenizer};

use super::{ParseEvent, StreamingParser};

impl StreamingParser {
    /// Feed a content token from an untrusted source. Control and reserved IDs are rejected.
    pub fn push_untrusted_token(
        &mut self,
        token: TokenId,
        tokenizer: &Tokenizer,
    ) -> Vec<ParseEvent> {
        if token.0 < BYTE_TOKEN_BASE {
            let mut events = Vec::new();
            self.fail(
                "special/reserved token ID supplied by an untrusted source",
                &mut events,
            );
            return events;
        }
        if token.0 >= tokenizer.vocab_size() {
            let mut events = Vec::new();
            self.fail(
                "content token ID is outside the tokenizer vocabulary",
                &mut events,
            );
            return events;
        }
        self.push(token, tokenizer)
    }

    /// Encode external text as content and feed it without interpreting marker-like strings.
    pub fn push_untrusted_text(&mut self, text: &str, tokenizer: &Tokenizer) -> Vec<ParseEvent> {
        let mut events = Vec::new();
        for token in tokenizer.encode_content(text) {
            events.extend(self.push_untrusted_token(token, tokenizer));
            if events
                .iter()
                .any(|event| matches!(event, ParseEvent::ParseError(_)))
            {
                break;
            }
        }
        events
    }
}
