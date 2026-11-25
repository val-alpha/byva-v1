#![no_main]

use byva_format::{ParserConfig, StreamingParser};
use byva_tokenizer::{TokenId, Tokenizer};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(tokenizer) = Tokenizer::from_merges(1, Vec::new()) else {
        return;
    };
    let mut parser = StreamingParser::new(ParserConfig { max_call_bytes: 4096 });
    for byte in data.iter().take(8192) {
        let token = TokenId(47 + u32::from(*byte));
        let _ = parser.push(token, &tokenizer);
    }
    let _ = parser.finish();
});
