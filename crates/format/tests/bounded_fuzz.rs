//! Bounded random-input harnesses mirroring fuzz/fuzz_targets/parser_tokens.rs and
//! fuzz/fuzz_targets/stream_parser.rs. Runs deterministically on stable so fuzz
//! semantics can be exercised in CI without a nightly libFuzzer toolchain.
//! Run with FUZZ_ITERATIONS to scale up.

use byva_format::{ParserConfig, StreamingParser};
use byva_tokenizer::{TokenId, Tokenizer};

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

fn iterations(default_value: u64) -> u64 {
    std::env::var("FUZZ_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default_value)
}

#[test]
fn bounded_random_token_ids_never_panic_or_recover_after_failure() {
    let rounds = iterations(20_000);
    let tokenizer = Tokenizer::from_merges(1, Vec::new()).expect("empty merge table is valid");
    let vocab = tokenizer.vocab_size();
    let mut rng = XorShift(0x0DDB_1A5E_5EED_BA11);

    for _ in 0..rounds {
        let mut parser = StreamingParser::default();
        let mut failed = false;
        for _ in 0..32 {
            let raw = rng.below(u64::from(u32::MAX)) as u32;
            let events = parser.push(TokenId(raw), &tokenizer);
            if events
                .iter()
                .any(|event| matches!(event, byva_format::ParseEvent::ParseError(_)))
            {
                failed = true;
            }
            if failed {
                assert!(
                    events.iter().all(|event| !matches!(
                        event,
                        byva_format::ParseEvent::CallComplete { .. }
                    ))
                );
            }
        }
        let _ = parser.finish();
        let _ = vocab;
    }
}

#[test]
fn bounded_random_byte_streams_never_panic() {
    let rounds = iterations(20_000);
    let tokenizer = Tokenizer::from_merges(1, Vec::new()).expect("empty merge table is valid");
    let mut rng = XorShift(0x00C0_FFEE_DEAD_10CC);

    for _ in 0..rounds {
        let mut parser = StreamingParser::new(ParserConfig {
            max_call_bytes: 4096,
        });
        let length = rng.below(128) as usize;
        for _ in 0..length {
            let byte = (rng.below(256)) as u8;
            let _ = parser.push(TokenId(47 + u32::from(byte)), &tokenizer);
        }
        let events = parser.finish();
        assert!(events.len() < 2);
    }
}
