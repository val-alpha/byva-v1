//! Shard reader, manifest validation, and token packing for the BYVA Rust
//! training path (D-0022).
//!
//! R0 scaffold: the crate boundary and error type exist; the D-0012 shard
//! reader, streaming document iteration, fixed packing into seq-length
//! windows, and the D-0016 per-run id:en mixture land at R2 behind golden
//! manifest and fixture tests (see `docs/PROPOSAL_RUST_TRAINING.md`).

#![deny(unsafe_op_in_unsafe_fn)]

mod error;

pub use error::DataError;
