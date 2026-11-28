//! Shard reader, manifest validation, and token packing for the BYVA Rust
//! training path (D-0022).
//!
//! R2 scope: D-0012 schema-version-1 manifest parsing and validation, shard
//! loading with per-shard sha256 and byte-count checks, canonical document
//! iteration with id/hash verification, and fixed packing into seq-length
//! windows with the reference 90/10 split. The per-run id:en mixture
//! (D-0016) joins at R3, decided by the run manifest.

#![deny(unsafe_op_in_unsafe_fn)]

mod error;
mod manifest;
mod pack;
mod sha256;
mod shard;

pub use error::DataError;
pub use manifest::{Manifest, ShardConfig, ShardEntry, ShardStats, SourceEntry};
pub use pack::{Packed, Sequences, WindowPacker, pack};
pub use sha256::{Sha256, sha256_hex};
pub use shard::{Shard, ShardDocument, load_shards};
