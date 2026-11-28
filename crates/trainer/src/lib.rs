//! BYVA pure-Rust training loop (method ne2410, D-0022).
//!
//! R3: the loop is complete. `Trainer` runs K4 token-budget batches
//! (constant FLOPs per step) through a fresh single-use autodiff graph per
//! step, with AdamW (Option 1, default) or SignTrust (Option 2) behind the
//! config, a warmup + cosine LR schedule, held-out eval, and the
//! `byva-train-rust-checkpoint-v1` format whose resume is verified
//! loss-history-equivalent by the integration tests (the R3 gate).

#![deny(unsafe_op_in_unsafe_fn)]

mod checkpoint;
mod config;
mod error;
mod model;
mod rng;
mod training;

pub use byva_data::Lang;
pub use checkpoint::{
    CheckpointMeta, EvalRecord, FORMAT_ID, LoadedCheckpoint, MixtureRecord, MixtureState,
    OptimStateDesc, TensorDesc,
};
pub use config::{ModelConfig, OptimizerKind, TrainConfig};
pub use error::TrainerError;
pub use model::{ParamLayout, ParamSpec, build_layout, explain_step, forward, init_params};
pub use rng::SplitMix64;
pub use training::Trainer;
