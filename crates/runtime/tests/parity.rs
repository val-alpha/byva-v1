//! Phase 5 numerical-parity test against the PyTorch training reference.
//!
//! The committed fixture (D-0023) is produced by
//! `training/make_runtime_fixture.py`: synthetic SplitMix64 weights are run
//! through the PyTorch reference forward pass, and the exact f32 logits at
//! the final prompt position are recorded as raw bit patterns. This test
//! rebuilds the identical weights, steps the prompt through the Rust
//! session, and compares the full 32,768-wide logit vector.

use std::path::Path;

use byva_model::ModelConfig;
use byva_runtime::InferenceModel;
use byva_runtime::lcg::fixture_weights;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    config: ModelConfig,
    prompt: Vec<u32>,
    expected_argmax: u32,
    tolerance: f64,
    expected_logits_bits: Vec<u32>,
}

#[test]
fn rust_logits_match_pytorch_reference() {
    let fixture_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/runtime_golden_v1.json");
    let raw = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|error| panic!("fixture readable at {:?}: {error}", fixture_path));
    let fixture: Fixture = serde_json::from_str(&raw).expect("fixture JSON parses");

    fixture.config.validate().expect("fixture config valid");
    let model = InferenceModel::from_weights(fixture_weights(&fixture.config))
        .expect("fixture weights build the model");
    let mut session = model.new_session();
    for &token in &fixture.prompt {
        session.step(token).expect("fixture prompt stays in range");
    }

    let logits = session.logits();
    assert_eq!(logits.len(), fixture.expected_logits_bits.len());
    let mut max_abs_diff = 0.0f32;
    for (actual, &expected_bits) in logits.iter().zip(&fixture.expected_logits_bits) {
        let expected = f32::from_bits(expected_bits);
        max_abs_diff = max_abs_diff.max((actual - expected).abs());
    }
    let tolerance = fixture.tolerance as f32;
    eprintln!("parity max abs logit diff: {max_abs_diff:.3e} (tolerance {tolerance})");
    assert!(
        max_abs_diff <= tolerance,
        "max abs logit diff {max_abs_diff} exceeds tolerance {tolerance}"
    );
    assert_eq!(
        session.greedy_token(),
        fixture.expected_argmax,
        "greedy argmax diverges from the PyTorch reference"
    );
}
