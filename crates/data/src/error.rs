//! Typed errors for shard reading, manifest validation, and packing.

use thiserror::Error;

/// Errors produced while reading shards or validating manifests (D-0012).
#[derive(Debug, Error)]
pub enum DataError {
    /// A shard does not match its manifest entry.
    #[error("shard {path} does not match its manifest entry: {reason}")]
    ManifestMismatch {
        /// Path of the rejected shard.
        path: String,
        /// What the validation rejected.
        reason: String,
    },
    /// The manifest file itself is unreadable or not valid JSON.
    #[error("manifest {path} could not be parsed: {reason}")]
    ManifestParse {
        /// Path of the rejected manifest.
        path: String,
        /// Parse failure detail.
        reason: String,
    },
    /// The manifest exists but violates the D-0012 schema v1 contract.
    #[error("manifest {path} is invalid: {reason}")]
    ManifestInvalid {
        /// Path of the rejected manifest.
        path: String,
        /// Schema violation detail.
        reason: String,
    },
    /// A shard payload violates the canonical document format.
    #[error("shard {path} line {line}: {reason}")]
    InvalidShard {
        /// Path of the rejected shard.
        path: String,
        /// One-based line number of the rejected document.
        line: usize,
        /// What the record violated.
        reason: String,
    },
    /// A packing request cannot produce the required windows.
    #[error("packing rejected: {reason}")]
    PackingInvalid {
        /// What the packing request violated.
        reason: String,
    },
}
