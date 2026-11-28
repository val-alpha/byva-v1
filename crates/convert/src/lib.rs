//! Conversion between BYVA training checkpoints and the native
//! `byva-weights-v1` container (Phase 4).
//!
//! The converter parses PyTorch checkpoint archives (`torch.save` zip +
//! pickle subset) natively in Rust, validates the model state dict against
//! the canonical tensor layout, and emits runtime weights. PyTorch is not
//! needed at conversion time.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod checkpoint;
pub mod converter;
pub mod error;
pub mod pickle;
pub mod zip;

pub use checkpoint::Checkpoint;
pub use converter::checkpoint_to_weights;
pub use error::ConvertError;
pub use pickle::{DType, TensorData};
