#![no_main]

use byva_tokenizer::Tokenizer;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(tokenizer) = Tokenizer::from_merges(1, Vec::new()) else {
        return;
    };
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let encoded = tokenizer.encode_content(text);
    let decoded = tokenizer.decode_content(&encoded);
    assert!(decoded.is_ok());
});
