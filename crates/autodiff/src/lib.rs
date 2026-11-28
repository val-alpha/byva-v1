//! Reverse-mode autodiff tape for the BYVA Rust training path (D-0022,
//! method ne2410).
//!
//! R1: the engine is in place. `Graph` records forward ops (mirroring the
//! frozen D-0013 architecture: pre-norm layernorm, causal MHA, exact-erf
//! GELU MLP, tied-logit-free cross-entropy per K2) and runs a deterministic
//! reverse pass; `AdamW` (Option 1, default) and `SignTrust` (Option 2,
//! K3) provide the optimizer updates. Every op is validated against
//! finite-difference gradchecks in `tests/gradcheck.rs`.

#![deny(unsafe_op_in_unsafe_fn)]

mod error;
mod graph;
mod optim;

pub use error::AutodiffError;
pub use graph::{Graph, NodeId};
pub use optim::{AdamW, AdamWConfig, OptimError, SignTrust, SignTrustConfig};
