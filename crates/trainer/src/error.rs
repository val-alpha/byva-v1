//! Typed errors for the Rust training loop and its checkpoint format.

use thiserror::Error;

/// Errors produced by the Rust training loop (method ne2410, D-0022).
#[derive(Debug, Error)]
pub enum TrainerError {
    /// A training checkpoint is not in the `byva-train-rust-checkpoint-v1`
    /// format or cannot be resumed.
    #[error("unknown or unreadable training checkpoint format: {found}")]
    UnknownCheckpointFormat {
        /// The format identifier found in the checkpoint header.
        found: String,
    },
}
