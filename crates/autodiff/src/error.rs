//! Typed errors for the reverse-mode autodiff tape.

use thiserror::Error;

/// Errors produced by autodiff op validation and tape execution.
#[derive(Debug, Error)]
pub enum AutodiffError {
    /// An op failed shape or numeric validation before or during execution.
    #[error("op {op} failed validation: {reason}")]
    OpValidation {
        /// Name of the failing op.
        op: &'static str,
        /// What the validation rejected.
        reason: String,
    },
}
