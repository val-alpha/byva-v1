//! Bounded random-input harness mirroring fuzz/fuzz_targets/tokenizer_round_trip.rs.
//! Runs deterministically on stable so the fuzz semantics can be exercised in CI
//! without a nightly libFuzzer toolchain. Run with FUZZ_ITERATIONS to scale up.

use unicode_normalization::UnicodeNormalization;

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

fn utf8_byte_pool() -> Vec<u8> {
    let mut pool = Vec::new();
    pool.extend(b"abcXYZ019 <|call|><|end|>");
    pool.extend([0xC3, 0xA9, 0xE2, 0x82, 0xAC, 0xF0, 0x9F, 0x98]);
    pool.extend([0x80, 0x80, 0xBF, 0x8F]);
    pool
}

#[test]
fn bounded_random_utf8_text_always_round_trips() {
    let iterations = std::env::var("FUZZ_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(20_000);
    let tokenizer =
        byva_tokenizer::Tokenizer::from_merges(1, Vec::new()).expect("empty merge table is valid");
    let pool = utf8_byte_pool();
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);

    for _ in 0..iterations {
        let length = rng.below(96) as usize;
        let raw: Vec<u8> = (0..length)
            .map(|_| pool[rng.below(pool.len() as u64) as usize])
            .collect();
        if let Ok(text) = String::from_utf8(raw) {
            let encoded = tokenizer.encode_content(&text);
            let decoded = tokenizer.decode_content(&encoded).ok();
            let normalized: String = text.nfc().collect();
            assert_eq!(decoded.as_deref(), Some(normalized.as_str()));
        }
    }
}
