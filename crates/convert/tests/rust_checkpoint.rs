//! Phase 6 numeric-parity check for the native Rust checkpoint path.
//!
//! A small model is initialized by the Rust trainer and saved in the
//! `byva-train-rust-checkpoint-v1` container. The converter must turn it
//! into runtime weights whose forward pass reproduces the trainer's
//! cross-entropy loss on the same tokens, so the SFT pipeline
//! (fine-tune -> save -> convert -> runtime chat) is numerically anchored.

use byva_convert::{Checkpoint, checkpoint_to_weights};
use byva_data::pack;
use byva_runtime::InferenceModel;
use byva_trainer::{ModelConfig, OptimizerKind, TrainConfig, Trainer};

fn rust_model() -> ModelConfig {
    ModelConfig {
        vocab: 32_768,
        d_model: 16,
        n_layers: 1,
        n_heads: 2,
        ffn: 32,
        seq_len: 8,
    }
}

fn config() -> TrainConfig {
    TrainConfig {
        model: rust_model(),
        optimizer: OptimizerKind::AdamW,
        learning_rate: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        clamp_min: 0.25,
        clamp_max: 4.0,
        steps: 1,
        token_budget: 32,
        seed: 42,
        warmup_steps: 1,
        floor_ratio: 0.1,
        eval_every: 1,
        eval_rows: 2,
        mixture_target_id: None,
        mixture_band: 0.1,
    }
}

fn token_stream(length: usize) -> Vec<u32> {
    (0..length as u32).map(|i| (i * 89 + 17) % 32_768).collect()
}

fn dataset(length: usize) -> byva_data::Packed {
    let stream = token_stream(length);
    pack(&stream, 8).unwrap_or_else(|error| panic!("{error}"))
}

/// Mean cross-entropy of the runtime session over one packed window,
/// stepping the input tokens and scoring each logits row against the
/// packed target of the same position.
fn runtime_window_loss(model: &InferenceModel, tokens: &[u32], targets: &[u32]) -> f32 {
    let mut session = model.new_session();
    let mut total = 0.0f64;
    for (position, &token) in tokens.iter().enumerate() {
        let logits = session
            .step(token)
            .unwrap_or_else(|error| panic!("runtime step {position}: {error}"));
        let target = targets[position] as usize;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for &logit in logits {
            sum_exp += (logit - max).exp();
        }
        let log_sum_exp = f64::from(max) + f64::from(sum_exp).ln();
        total += f64::from(logits[target]) - log_sum_exp;
    }
    (-total / tokens.len() as f64) as f32
}

#[test]
fn rust_checkpoint_matches_runtime_loss() {
    let train = dataset(600).train().clone();
    let val = dataset(200).val().clone();
    let seq_len = rust_model().seq_len;
    let tokens = train.x()[..seq_len].to_vec();
    let targets = train.y()[..seq_len].to_vec();
    let trainer = Trainer::new(config(), train, val).unwrap_or_else(|error| panic!("{error}"));

    let path = std::env::temp_dir().join(format!("byva-phase6-parity-{}.ckpt", std::process::id()));
    trainer
        .save(&path)
        .unwrap_or_else(|error| panic!("save checkpoint: {error}"));

    let checkpoint = Checkpoint::from_path(&path)
        .unwrap_or_else(|error| panic!("parse rust checkpoint: {error}"));
    assert_eq!(checkpoint.format, "byva-train-rust-checkpoint-v1");
    assert_eq!(checkpoint.config.vocab_size, 32_768);
    assert!(checkpoint.config.tie_embeddings);

    let weights =
        checkpoint_to_weights(&checkpoint).unwrap_or_else(|error| panic!("convert: {error}"));
    let model =
        InferenceModel::from_weights(weights).unwrap_or_else(|error| panic!("runtime: {error}"));

    let trainer_loss = trainer
        .forward_loss(&tokens, &targets, None)
        .unwrap_or_else(|error| panic!("trainer loss: {error}"));
    let runtime_loss = runtime_window_loss(&model, &tokens, &targets);

    let diff = (trainer_loss - runtime_loss).abs();
    eprintln!(
        "rust-checkpoint parity: trainer {trainer_loss:.6} runtime {runtime_loss:.6} diff {diff:.3e}"
    );
    assert!(
        diff <= 1e-3,
        "trainer loss {trainer_loss} vs runtime loss {runtime_loss} exceeds tolerance 1e-3"
    );

    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("failed to remove {}: {error}", path.display()),
    }
}
