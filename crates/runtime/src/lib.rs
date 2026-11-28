//! Native Rust inference runtime for BYVA.
//!
//! The runtime loads `byva-weights-v1` containers ([`Weights`], D-0023) and
//! runs the D-0013 decoder-only architecture in scalar fp32 with an
//! incremental KV cache ([`InferenceModel`] and [`Session`]). Numerical
//! parity with the PyTorch training reference is pinned by the committed
//! golden fixture described in D-0023. Unsafe code, if required by optimized
//! backends, must remain narrowly scoped and documented.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod error;
pub mod lcg;
pub mod model;
pub mod weights;

pub use error::RuntimeError;
pub use model::{InferenceModel, Session};
pub use weights::Weights;
