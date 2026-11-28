//! Cross-language validation of the scaled Rust BPE trainer against the
//! Python trainer's golden fixture (D-0017/D-0020/D-0021).

use std::path::PathBuf;

use byva_tokenizer::trainer::{artifact_file_json, train_bpe};
use byva_tokenizer::{FIRST_MERGE_ID, Tokenizer, TokenizerError};
use serde_json::Value;

fn smoke_corpus() -> Vec<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../training/fixtures/smoke/sample.jsonl");
    let raw = std::fs::read_to_string(&path).expect("smoke fixture is readable");
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let value: Value = serde_json::from_str(line).expect("smoke fixture line is JSON");
            value
                .get("text")
                .and_then(Value::as_str)
                .expect("smoke fixture record has text")
                .to_owned()
        })
        .collect()
}

fn serialize(artifact: &byva_tokenizer::TokenizerArtifact) -> String {
    artifact_file_json(artifact).expect("artifact serialization cannot fail for numeric fields")
}

#[test]
fn rust_trainer_reproduces_the_python_golden_fixture_byte_identically() {
    let golden = include_str!("golden/bpe_smoke_v1.json");
    let texts = smoke_corpus();
    let artifact = train_bpe(&texts, FIRST_MERGE_ID + 64, 1).expect("training succeeds");
    assert_eq!(serialize(&artifact), golden);
}

#[test]
fn training_is_independent_of_document_order() {
    let mut texts = smoke_corpus();
    let forward = train_bpe(&texts, FIRST_MERGE_ID + 64, 1).expect("training succeeds");
    texts.reverse();
    let reversed = train_bpe(&texts, FIRST_MERGE_ID + 64, 1).expect("training succeeds");
    assert_eq!(serialize(&forward), serialize(&reversed));
}

#[test]
fn equal_counts_break_ties_by_lowest_left_then_right_id() {
    let texts = vec!["abcd".to_owned(), "abcd".to_owned()];
    let artifact = train_bpe(&texts, FIRST_MERGE_ID + 3, 1).expect("three merges are trainable");
    let a = BYTE_BASE + u32::from(b'a');
    let b = BYTE_BASE + u32::from(b'b');
    let c = BYTE_BASE + u32::from(b'c');
    let d = BYTE_BASE + u32::from(b'd');
    assert_eq!(artifact.merges[0].left, a);
    assert_eq!(artifact.merges[0].right, b);
    assert_eq!(artifact.merges[1].left, c);
    assert_eq!(artifact.merges[1].right, d);
}

#[test]
fn exhausted_corpus_reports_how_many_merges_were_trainable() {
    let texts = vec!["abcdef".to_owned()];
    let error = train_bpe(&texts, FIRST_MERGE_ID + 8, 1)
        .expect_err("every pair occurs once, so no merge is trainable");
    assert!(matches!(
        error,
        TokenizerError::CorpusExhausted {
            trained: 0,
            requested: 8
        }
    ));
}

#[test]
fn empty_corpus_is_rejected() {
    let error = train_bpe(&[], FIRST_MERGE_ID + 8, 1).expect_err("empty corpus must fail");
    assert!(matches!(error, TokenizerError::EmptyCorpus));
}

#[test]
fn trained_artifact_round_trips_through_the_serving_tokenizer() {
    let texts = smoke_corpus();
    let artifact = train_bpe(&texts, FIRST_MERGE_ID + 64, 1).expect("training succeeds");
    let tokenizer = Tokenizer::from_merges(artifact.version, artifact.merges.clone())
        .expect("artifact is valid");
    let sample = "Bahasa Indonesia & English: naïve café, 2026!";
    let tokens = tokenizer.encode_content(sample);
    let decoded = tokenizer.decode_content(&tokens).expect("content decodes");
    assert_eq!(decoded, sample);
}

const BYTE_BASE: u32 = byva_tokenizer::BYTE_TOKEN_BASE;
