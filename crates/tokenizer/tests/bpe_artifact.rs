//! Validates the trained 32,768-vocabulary BPE artifact against the Rust
//! tokenizer. The artifact is trained by training/train_bpe.py from Phase 2
//! shards; see docs/PHASE3_REPORT.md and D-0017.

use byva_tokenizer::{
    BYTE_TOKEN_BASE, FIRST_MERGE_ID, SpecialToken, TokenId, Tokenizer, TokenizerArtifact,
};

const ARTIFACT_JSON: &str = include_str!("../../../artifacts/tokenizer/byva-tokenizer-v1.json");

fn load_trained_tokenizer() -> Tokenizer {
    let artifact: TokenizerArtifact =
        serde_json::from_str(ARTIFACT_JSON).expect("trained BPE artifact is valid JSON");
    Tokenizer::from_artifact(artifact).expect("trained BPE artifact passes tokenizer validation")
}

#[test]
fn trained_artifact_loads_with_published_vocab_bounds() {
    let tokenizer = load_trained_tokenizer();
    assert_eq!(tokenizer.version(), 1);
    assert_eq!(tokenizer.vocab_size(), 32_768);
    let artifact: TokenizerArtifact = serde_json::from_str(ARTIFACT_JSON).expect("artifact parses");
    assert_eq!(artifact.merges.len(), (32_768 - FIRST_MERGE_ID) as usize);
}

#[test]
fn trained_artifact_round_trips_bilingual_content() {
    let tokenizer = load_trained_tokenizer();
    let samples = [
        "Kuda hitam kecil itu berlari di padang rumput.",
        "The small black horse runs across the green field.",
        "Pada tahun 2023, jumlah penduduk meningkat 5,4 persen.",
        "In 2023 the population grew by 5.4 percent.",
        "emoji \u{1F30F} dan aksara \u{304B}\u{306A} tetap konten biasa",
        "literal <|call|> and <|reserved_7|> spellings stay content",
    ];
    for text in samples {
        let encoded = tokenizer.encode_content(text);
        assert!(
            !encoded.is_empty(),
            "encoding produced no tokens for {text:?}"
        );
        assert!(
            encoded
                .iter()
                .all(|token| token.0 >= BYTE_TOKEN_BASE && token.0 < tokenizer.vocab_size()),
            "content token out of range for {text:?}"
        );
        assert_eq!(
            tokenizer.decode_content(&encoded).ok().as_deref(),
            Some(text),
            "round trip diverged for {text:?}"
        );
    }
}

#[test]
fn trained_artifact_compresses_repetitive_text() {
    let tokenizer = load_trained_tokenizer();
    let text = "kuda hitam kecil berlari di padang rumput. ".repeat(8);
    let encoded = tokenizer.encode_content(&text);
    assert!(
        (encoded.len() as f64) < (text.len() as f64) / 2.0,
        "expected better than 2x compression, got {} tokens for {} bytes",
        encoded.len(),
        text.len()
    );
}

#[test]
fn every_trained_merge_expands_to_bytes() {
    let tokenizer = load_trained_tokenizer();
    for index in 0..(32_768 - FIRST_MERGE_ID) {
        let token = TokenId(FIRST_MERGE_ID + index);
        let bytes = tokenizer
            .token_bytes(token)
            .unwrap_or_else(|error| panic!("merge token {token:?} failed to expand: {error}"));
        assert!(
            !bytes.is_empty(),
            "merge token {token:?} expanded to no bytes"
        );
    }
}

#[test]
fn control_ids_stay_outside_content_encoding() {
    let tokenizer = load_trained_tokenizer();
    for special in [
        SpecialToken::Bos,
        SpecialToken::End,
        SpecialToken::Call,
        SpecialToken::Reserved(31),
    ] {
        let id = special.id();
        assert!(
            tokenizer.decode_content(&[id]).is_err(),
            "control token {id:?} must not decode as content"
        );
    }
}
