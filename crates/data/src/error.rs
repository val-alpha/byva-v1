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
}
