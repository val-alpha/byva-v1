//! Typed errors for the Rust training loop and its checkpoint format.

use thiserror::Error;

use byva_autodiff::AutodiffError;
use byva_autodiff::OptimError;

/// Errors produced by the Rust training loop (method ne2410, D-0022).
#[derive(Debug, Error)]
pub enum TrainerError {
    /// A training checkpoint is not in the `byva-train-rust-checkpoint-v1`
    /// format (D-0014 discipline: unknown formats are rejected, not guessed).
    #[error("unknown or unreadable training checkpoint format: {found}")]
    UnknownCheckpointFormat {
        /// The format identifier found in the checkpoint header.
        found: String,
    },
    /// A checkpoint file exists but fails integrity or structure checks.
    #[error("checkpoint {path} is corrupt: {reason}")]
    CheckpointCorrupt {
        /// Path of the rejected checkpoint.
        path: String,
        /// What the integrity check rejected.
        reason: String,
    },
    /// The checkpoint is valid but does not match the run being resumed.
    #[error("checkpoint does not match the provided run: {reason}")]
    ConfigMismatch {
        /// What differed between checkpoint and run.
        reason: String,
    },
    /// A training configuration value is outside its valid range.
    #[error("invalid training configuration: {reason}")]
    ConfigInvalid {
        /// What the validation rejected.
        reason: String,
    },
    /// The provided training data cannot feed the configured run.
    #[error("training data rejected: {reason}")]
    DataInvalid {
        /// What the data check rejected.
        reason: String,
    },
    /// Filesystem access failed.
    #[error("i/o error on {path}: {reason}")]
    Io {
        /// Path involved in the failed operation.
        path: String,
        /// Underlying error message.
        reason: String,
    },
    /// Autodiff graph construction or backward pass failed.
    #[error(transparent)]
    Autodiff(#[from] AutodiffError),
    /// Optimizer construction or update failed.
    #[error(transparent)]
    Optim(#[from] OptimError),
}
