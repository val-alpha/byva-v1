//! Typed errors for dense tensor storage.

use thiserror::Error;

/// Errors produced by dense tensor construction, indexing, and reshaping.
#[derive(Debug, Error)]
pub enum TensorError {
    /// A shape's element count overflows or exceeds the addressable range.
    #[error("shape {shape:?} overflows the element count")]
    ShapeOverflow {
        /// The rejected shape.
        shape: Vec<usize>,
    },
    /// The provided data length does not match the shape's element count.
    #[error("shape {shape:?} holds {expected} elements but {found} were provided")]
    ElementCountMismatch {
        /// The requested shape.
        shape: Vec<usize>,
        /// Element count implied by the shape.
        expected: u64,
        /// Number of elements supplied.
        found: u64,
    },
    /// An index has a different rank than the tensor.
    #[error("tensor has rank {rank} but the index has {found} components")]
    RankMismatch {
        /// Tensor rank.
        rank: usize,
        /// Number of index components supplied.
        found: usize,
    },
    /// An index component is out of bounds for its dimension.
    #[error("index {index} is out of bounds for dimension {dimension} of size {size}")]
    IndexOutOfBounds {
        /// Offending index value.
        index: usize,
        /// Dimension position within the shape.
        dimension: usize,
        /// Size of that dimension.
        size: usize,
    },
}
