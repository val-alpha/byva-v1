//! Checkpoint parsing: PyTorch zip archive + pickle payload to `Checkpoint`.
//!
//! `torch.save` stores the payload dict in `data.pkl` and each tensor storage
//! as a separate `data/<key>` entry. Storage references in the pickle are
//! resolved lazily through the zip resolver, so storages the payload never
//! rebuilds (none in practice) are never read.

use byva_model::ModelConfig;

use crate::error::ConvertError;
use crate::pickle::{self, StorageRef, StorageResolver, TensorData, Value};
use crate::zip::{self, ZipEntry};

const FORMAT_V1: &str = "byva-train-checkpoint-v1";
const FORMAT_V2: &str = "byva-train-checkpoint-v2";

/// A parsed `byva-train-checkpoint-v1`/`-v2` payload.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Format string (`format` field).
    pub format: String,
    /// Training step at save time (`step` field).
    pub step: u64,
    /// Validated model configuration (`config` field).
    pub config: ModelConfig,
    /// Model state dict in file order (`model` field; `_metadata` skipped).
    pub model: Vec<(String, TensorData)>,
}

/// Resolves `('storage', dtype, key, location, numel)` references against the
/// checkpoint archive's `data/<key>` entries.
struct ZipResolver<'a> {
    bytes: &'a [u8],
    entries: &'a [ZipEntry],
}

impl StorageResolver for ZipResolver<'_> {
    fn resolve(&self, reference: &StorageRef) -> Result<Vec<u8>, ConvertError> {
        let suffix = format!("/data/{}", reference.key);
        let entry = zip::find_entry(self.entries, &suffix)?;
        let data = zip::read_entry(self.bytes, entry)?;
        let expected = reference
            .numel
            .checked_mul(reference.dtype.item_size())
            .ok_or(ConvertError::Pickle {
                position: 0,
                detail: format!("storage {} byte length overflows", reference.key),
            })?;
        if data.len() as u64 != expected {
            return Err(ConvertError::Pickle {
                position: 0,
                detail: format!(
                    "storage {} has {} bytes, expected {expected}",
                    reference.key,
                    data.len()
                ),
            });
        }
        Ok(data.to_vec())
    }
}

impl Checkpoint {
    /// Parses a checkpoint archive from raw bytes.
    ///
    /// # Errors
    /// Fails when the archive or pickle is malformed, the format string is
    /// unsupported, a v1 payload carries scaler state, or the config fails
    /// validation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ConvertError> {
        let entries = zip::read_directory(bytes)?;
        let resolver = ZipResolver {
            bytes,
            entries: &entries,
        };
        let data_pkl = zip::find_entry(&entries, "/data.pkl")?;
        let payload = pickle::parse(zip::read_entry(bytes, data_pkl)?, &resolver)?;
        let mapping = payload.as_dict().ok_or(ConvertError::NotAMapping)?;

        let format = string_field(mapping, "format")?;
        if format != FORMAT_V1 && format != FORMAT_V2 {
            return Err(ConvertError::UnsupportedFormat {
                got: format.to_string(),
            });
        }

        let scaler = field(mapping, "scaler")?;
        if format == FORMAT_V1 && !matches!(scaler, Some(Value::None)) {
            return Err(ConvertError::V1WithScaler);
        }

        let step = int_field(mapping, "step")?;
        let step = u64::try_from(step).map_err(|_| ConvertError::TypeMismatch {
            name: "step",
            expected: "non-negative int",
            found: format!("{step}"),
        })?;

        let config_dict = require_field(mapping, "config")?;
        let Value::Dict(config_entries) = config_dict else {
            return Err(ConvertError::TypeMismatch {
                name: "config",
                expected: "dict",
                found: config_dict.kind(),
            });
        };
        let config = parse_config(config_entries)?;

        let model_dict = require_field(mapping, "model")?;
        let Value::Dict(model_entries) = model_dict else {
            return Err(ConvertError::TypeMismatch {
                name: "model",
                expected: "dict",
                found: model_dict.kind(),
            });
        };
        let model = parse_model(model_entries)?;

        Ok(Self {
            format: format.to_string(),
            step,
            config,
            model,
        })
    }

    /// Reads, parses, and validates a checkpoint file from disk.
    ///
    /// # Errors
    /// Fails on I/O errors and every [`Checkpoint::from_bytes`] error.
    pub fn from_path(path: &std::path::Path) -> Result<Self, ConvertError> {
        let bytes = std::fs::read(path).map_err(|source| ConvertError::CheckpointIo {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_bytes(&bytes)
    }
}

fn field<'a>(mapping: &'a [(Value, Value)], name: &str) -> Result<Option<&'a Value>, ConvertError> {
    Ok(mapping
        .iter()
        .find(|(key, _)| key.as_str() == Some(name))
        .map(|(_, value)| value))
}

fn require_field<'a>(
    mapping: &'a [(Value, Value)],
    name: &'static str,
) -> Result<&'a Value, ConvertError> {
    field(mapping, name).and_then(|value| value.ok_or(ConvertError::MissingField { name }))
}

fn string_field<'a>(
    mapping: &'a [(Value, Value)],
    name: &'static str,
) -> Result<&'a str, ConvertError> {
    let value = require_field(mapping, name)?;
    value.as_str().ok_or(ConvertError::TypeMismatch {
        name,
        expected: "str",
        found: value.kind(),
    })
}

fn int_field(mapping: &[(Value, Value)], name: &'static str) -> Result<i128, ConvertError> {
    let value = require_field(mapping, name)?;
    value.as_int().ok_or(ConvertError::TypeMismatch {
        name: "step",
        expected: "int",
        found: value.kind(),
    })
}

fn parse_config(entries: &[(Value, Value)]) -> Result<ModelConfig, ConvertError> {
    let mut config = ModelConfig {
        vocab_size: 0,
        d_model: 0,
        n_layers: 0,
        n_heads: 0,
        ffn_hidden: 0,
        max_seq_len: 0,
        tie_embeddings: false,
    };
    for name in [
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "ffn_hidden",
        "max_seq_len",
    ] {
        let value = require_field(entries, name)?;
        let parsed = value.as_int().ok_or(ConvertError::TypeMismatch {
            name: "config",
            expected: "int",
            found: value.kind(),
        })?;
        let converted = u32::try_from(parsed).map_err(|_| ConvertError::TypeMismatch {
            name: "config",
            expected: "non-negative int",
            found: format!("{name}={parsed}"),
        })?;
        match name {
            "vocab_size" => config.vocab_size = converted,
            "d_model" => config.d_model = converted,
            "n_layers" => config.n_layers = converted,
            "n_heads" => config.n_heads = converted,
            "ffn_hidden" => config.ffn_hidden = converted,
            "max_seq_len" => config.max_seq_len = converted,
            _ => unreachable!("field list is static"),
        }
    }
    let tie = require_field(entries, "tie_embeddings")?;
    match tie {
        Value::Bool(flag) => config.tie_embeddings = *flag,
        other => {
            return Err(ConvertError::TypeMismatch {
                name: "config",
                expected: "bool",
                found: other.kind(),
            });
        }
    }
    config.validate()?;
    Ok(config)
}

fn parse_model(entries: &[(Value, Value)]) -> Result<Vec<(String, TensorData)>, ConvertError> {
    let mut model = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        let Some(name) = key.as_str() else {
            return Err(ConvertError::TypeMismatch {
                name: "model",
                expected: "str keys",
                found: key.kind(),
            });
        };
        if name == "_metadata" {
            continue;
        }
        let Value::Tensor(tensor) = value else {
            return Err(ConvertError::TypeMismatch {
                name: "model",
                expected: "tensor",
                found: format!("{name}: {}", value.kind()),
            });
        };
        let expected_bytes = tensor.elements().checked_mul(tensor.dtype.item_size());
        match expected_bytes {
            Some(expected) if expected == tensor.data.len() as u64 => {}
            _ => {
                return Err(ConvertError::StorageOutOfBounds {
                    name: name.to_string(),
                });
            }
        }
        model.push((name.to_string(), tensor.clone()));
    }
    Ok(model)
}
