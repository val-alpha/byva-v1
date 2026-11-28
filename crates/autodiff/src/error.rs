//! Typed errors for the reverse-mode autodiff tape.

use byva_tensor::TensorError;
use thiserror::Error;

/// Errors produced by graph construction, forward ops, and backward passes.
#[derive(Debug, Error)]
pub enum AutodiffError {
    /// A tensor operation inside an op failed.
    #[error("tensor error: {0}")]
    Tensor(#[from] TensorError),
    /// An op failed shape or numeric validation before or during execution.
    #[error("op {op} failed validation: {reason}")]
    OpValidation {
        /// Name of the failing op.
        op: &'static str,
        /// What the validation rejected.
        reason: String,
    },
    /// An op received an input with an unexpected shape.
    #[error("op {op} expects shape {expected:?} but found {found:?}")]
    ShapeMismatch {
        /// Name of the failing op.
        op: &'static str,
        /// The shape the op requires.
        expected: Vec<usize>,
        /// The shape that was supplied.
        found: Vec<usize>,
    },
    /// A node id does not exist in this graph.
    #[error("node id {index} is not part of this graph")]
    UnknownNode {
        /// The invalid node id.
        index: usize,
    },
    /// Backward was started from a node that is not a scalar.
    #[error("loss node must be a scalar but has shape {found:?}")]
    NotScalar {
        /// Shape of the node backward was started from.
        found: Vec<usize>,
    },
}
