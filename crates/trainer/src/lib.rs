//! BYVA pure-Rust training loop (method ne2410, D-0022).
//!
//! R0 scaffold: the crate boundary and error type exist; the training loop,
//! AdamW (Option 1, default) and SignTrust (Option 2, `--optimizer
//! signtrust`) optimizers, token-budget batching, and the
//! `byva-train-rust-checkpoint-v1` format land at R3 behind the
//! resume-equivalence test (see `docs/PROPOSAL_RUST_TRAINING.md`).

#![deny(unsafe_op_in_unsafe_fn)]

mod error;

pub use error::TrainerError;
