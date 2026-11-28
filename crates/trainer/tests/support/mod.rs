//! Shared helpers for the R3 trainer integration tests: a tiny synthetic
//! model config, a deterministic token stream, and temp checkpoint paths.
//!
//! Not every test binary uses every helper.

#![allow(dead_code)]

use std::path::PathBuf;

use byva_data::{Packed, pack};
use byva_trainer::{ModelConfig, OptimizerKind, TrainConfig};

pub fn test_model() -> ModelConfig {
    ModelConfig {
        vocab: 64,
        d_model: 16,
        n_layers: 2,
        n_heads: 2,
        ffn: 32,
        seq_len: 8,
    }
}

pub fn config(optimizer: OptimizerKind, steps: u64) -> TrainConfig {
    TrainConfig {
        model: test_model(),
        optimizer,
        learning_rate: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        clamp_min: 0.25,
        clamp_max: 4.0,
        steps,
        token_budget: 32,
        seed: 42,
        warmup_steps: 2,
        floor_ratio: 0.1,
        eval_every: 2,
        eval_rows: 2,
    }
}

/// A deterministic pseudo-random token stream over the 64-token vocabulary.
pub fn token_stream(length: usize) -> Vec<u32> {
    (0..length as u32).map(|i| (i * 37 + 11) % 64).collect()
}

/// A packed dataset with 59 train / 7 val windows at seq_len 8.
pub fn packed() -> Packed {
    pack(&token_stream(600), 8).unwrap_or_else(|e| panic!("{e}"))
}

pub fn temp_ckpt(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("byva-r3-{}-{}.ckpt", name, std::process::id()))
}

pub fn cleanup(path: &PathBuf) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("failed to remove {}: {error}", path.display()),
    }
}
