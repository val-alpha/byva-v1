//! Typed errors for the BYVA inference runtime.

use thiserror::Error;

/// Errors produced by weight loading and inference.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// The weights file could not be read.
    #[error("failed to read weights file {path}: {source}")]
    WeightsIo {
        /// Path that was attempted.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The weights file does not start with the BYVAW magic.
    #[error("weights file has invalid magic bytes (expected \"BYVAW\")")]
    BadMagic,
    /// The weights file uses an unknown format version.
    #[error("unsupported weights format version {got}")]
    UnsupportedVersion {
        /// Version byte found in the file.
        got: u8,
    },
    /// The `tie_embeddings` flag byte is neither 0 nor 1.
    #[error("weights file has invalid tie_embeddings flag byte {got} (expected 0 or 1)")]
    InvalidTieEmbeddingsFlag {
        /// Flag byte found in the file.
        got: u8,
    },
    /// The declared tensor count does not match the canonical layout.
    #[error("weights file declares {got} tensors, expected {expected}")]
    TensorCount {
        /// Tensor count found.
        got: u32,
        /// Tensor count required by the config.
        expected: u32,
    },
    /// The weights file ends before its declared contents.
    #[error("weights file truncated: expected {expected} more bytes, found {found}")]
    Truncated {
        /// Bytes still required by the declared layout.
        expected: u64,
        /// Bytes actually remaining.
        found: u64,
    },
    /// A tensor name is not part of the canonical layout.
    #[error("weights tensor name {name:?} is not part of the canonical layout")]
    UnknownTensorName {
        /// Name as stored in the file.
        name: String,
    },
    /// Tensors appear outside the canonical order.
    #[error("weights tensors out of canonical order: found {found:?}, expected {expected:?}")]
    TensorOrder {
        /// Name found in the file.
        found: String,
        /// Name required at this position.
        expected: String,
    },
    /// A tensor has the wrong element count for the declared config.
    #[error("tensor {name:?} has {got} elements, expected {expected}")]
    TensorSize {
        /// Tensor name.
        name: String,
        /// Element count found.
        got: u64,
        /// Element count required by the config.
        expected: u64,
    },
    /// A tensor required by the config is absent.
    #[error("missing tensor {name:?} required by the configuration")]
    MissingTensor {
        /// Tensor name.
        name: String,
    },
    /// A tensor that cannot exist for this config is present.
    #[error("unexpected tensor {name:?} for this configuration")]
    UnexpectedTensor {
        /// Tensor name.
        name: String,
    },
    /// A tensor contains a non-finite value.
    #[error("tensor {name:?} contains a non-finite value")]
    NonFiniteTensor {
        /// Tensor name.
        name: String,
    },
    /// A tensor name is not valid UTF-8.
    #[error("weights file contains a tensor name that is not valid UTF-8")]
    InvalidTensorName,
    /// The embedded model config failed validation.
    #[error("invalid model config in weights file: {source}")]
    InvalidConfig {
        /// Underlying config validation error.
        #[from]
        source: byva_model::ConfigError,
    },
    /// The decoder context grew beyond `max_seq_len` learned positions.
    #[error("context position {position} exceeds max_seq_len {max_seq_len}")]
    ContextOverflow {
        /// Position the next token would occupy.
        position: u32,
        /// Maximum learned positions.
        max_seq_len: u32,
    },
    /// A token id is outside the vocabulary.
    #[error("token id {token} is outside the vocabulary of {vocab_size} tokens")]
    TokenOutOfRange {
        /// Offending token id.
        token: u32,
        /// Vocabulary size from the weights config.
        vocab_size: u32,
    },
}
