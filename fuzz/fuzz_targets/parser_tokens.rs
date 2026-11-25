#![no_main]

use byva_format::StreamingParser;
use byva_tokenizer::{Merge, TokenId, Tokenizer};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(tokenizer) = Tokenizer::from_merges(1, Vec::<Merge>::new()) else {
        return;
    };
    let mut parser = StreamingParser::default();

    for chunk in data.chunks_exact(4) {
        let raw = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let _events = parser.push(TokenId(raw), &tokenizer);
    }

    let _final_events = parser.finish();
});
