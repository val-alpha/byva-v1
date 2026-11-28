//! Typed errors for checkpoint parsing and conversion.

use thiserror::Error;

/// Errors produced while reading a PyTorch training checkpoint and emitting
/// a `byva-weights-v1` container (D-0023).
#[derive(Debug, Error)]
pub enum ConvertError {
    /// The checkpoint file could not be read.
    #[error("failed to read checkpoint file {path}: {source}")]
    CheckpointIo {
        /// Path that was attempted.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The output file could not be written.
    #[error("failed to write weights file {path}: {source}")]
    WeightsIo {
        /// Path that was attempted.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The file is not a zip archive (missing end-of-central-directory).
    #[error("checkpoint is not a zip archive (no end-of-central-directory found)")]
    NotAZipFile,
    /// The central directory is malformed.
    #[error("zip central directory malformed at offset {offset}")]
    BadCentralDirectory {
        /// Byte offset of the malformed header.
        offset: u64,
    },
    /// A zip entry uses an unsupported compression method.
    #[error(
        "zip entry {entry:?} uses unsupported compression method {method} (only stored/method 0 is supported)"
    )]
    UnsupportedZipMethod {
        /// Entry name.
        entry: String,
        /// Compression method id.
        method: u16,
    },
    /// A zip entry name is not valid UTF-8.
    #[error("zip entry name is not valid UTF-8")]
    InvalidEntryName,
    /// A required entry is absent from the archive.
    #[error("zip entry {name:?} not found in checkpoint archive")]
    MissingEntry {
        /// Entry name suffix looked up.
        name: String,
    },
    /// The pickle stream is malformed.
    #[error("malformed pickle stream at byte {position}: {detail}")]
    Pickle {
        /// Byte position in `data.pkl`.
        position: u64,
        /// What went wrong.
        detail: String,
    },
    /// The pickle stream uses an opcode outside the supported subset.
    #[error("pickle opcode 0x{opcode:02x} at byte {position} is not supported")]
    UnsupportedOpcode {
        /// Opcode byte.
        opcode: u8,
        /// Byte position in `data.pkl`.
        position: u64,
    },
    /// The top-level structure is not a mapping.
    #[error("checkpoint payload top level is not a mapping")]
    NotAMapping,
    /// A required checkpoint field is absent.
    #[error("checkpoint field {name:?} is missing")]
    MissingField {
        /// Field name.
        name: &'static str,
    },
    /// A field has an unexpected value type.
    #[error("checkpoint field {name:?}: expected {expected}, found {found}")]
    TypeMismatch {
        /// Field name.
        name: &'static str,
        /// Expected value description.
        expected: &'static str,
        /// Found value description.
        found: String,
    },
    /// The checkpoint format string is unsupported.
    #[error("unsupported checkpoint format {got:?}")]
    UnsupportedFormat {
        /// Format string found.
        got: String,
    },
    /// A v1 checkpoint carries scaler state, which D-0018 forbids.
    #[error("v1 checkpoint carries scaler state; v1 must not have scaler state per D-0018")]
    V1WithScaler,
    /// The embedded config failed validation.
    #[error("checkpoint config is invalid: {source}")]
    InvalidConfig {
        /// Underlying config validation error.
        #[from]
        source: byva_model::ConfigError,
    },
    /// A model tensor uses a storage dtype other than fp32.
    #[error("model tensor {name:?} uses {got:?} storage; model tensors must be fp32")]
    TensorDType {
        /// Tensor name.
        name: String,
        /// Storage class found.
        got: String,
    },
    /// A model tensor has a different element count than the config implies.
    #[error("model tensor {name:?} has {got} elements, expected {expected}")]
    TensorShape {
        /// Tensor name.
        name: String,
        /// Element count found.
        got: u64,
        /// Element count required by the config.
        expected: u64,
    },
    /// A model tensor references storage beyond its extent.
    #[error("model tensor {name:?} references storage beyond its extent")]
    StorageOutOfBounds {
        /// Tensor name.
        name: String,
    },
    /// A model tensor contains a non-finite value.
    #[error("model tensor {name:?} contains a non-finite value")]
    NonFiniteTensor {
        /// Tensor name.
        name: String,
    },
    /// The model state dict contains a name outside the canonical layout.
    #[error("model state dict contains unknown tensor {name:?}")]
    UnknownModelTensor {
        /// Tensor name found.
        name: String,
    },
    /// The model state dict is missing a tensor required by the config.
    #[error("model state dict is missing tensor {name:?}")]
    MissingModelTensor {
        /// Tensor name required.
        name: String,
    },
    /// The runtime rejected the assembled weights.
    #[error("runtime rejected the converted weights: {source}")]
    Runtime {
        /// Underlying runtime error.
        #[from]
        source: byva_runtime::RuntimeError,
    },
}
