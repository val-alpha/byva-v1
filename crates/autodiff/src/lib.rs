//! Reverse-mode autodiff tape for the BYVA Rust training path (D-0022).
//!
//! R0 scaffold: the crate boundary and error type exist; the ops required by
//! D-0013 (embedding scatter-add, causal MHA, GELU MLP, pre-norm layernorm,
//! tied output projection, fused tied-logit-free cross-entropy) and the
//! AdamW/SignTrust updates land at R1 behind finite-difference gradchecks
//! (see `docs/PROPOSAL_RUST_TRAINING.md`).

#![deny(unsafe_op_in_unsafe_fn)]

mod error;

pub use error::AutodiffError;
