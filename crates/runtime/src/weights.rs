//! Runtime weight container (`byva-weights-v1`).
//!
//! The container is the native weight format consumed by the Phase 5 runtime
//! and produced by the Phase 4 converter (D-0023). Layout, little-endian:
//!
//! ```text
//! magic            5 bytes  "BYVAW"
//! version          u8       1
//! config          25 bytes  vocab_size u32, d_model u32, n_layers u32,
//!                           n_heads u32, ffn_hidden u32, max_seq_len u32,
//!                           tie_embeddings u8 (0 or 1)
//! tensor_count     u32
//! per tensor:      u32 name_len, name bytes (UTF-8),
//!                  u64 element_count, element_count * f32 (LE)
//! ```
//!
//! Tensors must appear in the canonical order for the config and match the
//! element counts implied by it. All parsing is checked and malformed input
//! yields typed errors, never panics.

use std::path::Path;

use byva_model::{ConfigError, ModelConfig};

use crate::error::RuntimeError;

/// File magic of the `byva-weights-v1` container.
pub const WEIGHTS_MAGIC: [u8; 5] = *b"BYVAW";
/// Container format version implemented by this loader.
pub const WEIGHTS_FORMAT_VERSION: u8 = 1;
/// Fixed byte size of the config header block.
const CONFIG_BLOCK_BYTES: usize = 4 * 6 + 1;

/// Parsed and validated weight tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    config: ModelConfig,
    tensors: Vec<(String, Vec<f32>)>,
}

/// Describes one tensor in the canonical layout.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorSpec {
    name: String,
    elements: u64,
}

impl Weights {
    /// Reads, parses, and validates a weights file from disk.
    pub fn from_path(path: &Path) -> Result<Self, RuntimeError> {
        let bytes = std::fs::read(path).map_err(|source| RuntimeError::WeightsIo {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_bytes(&bytes)
    }

    /// Parses and validates an in-memory weights file image.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RuntimeError> {
        let mut cursor = Cursor::new(bytes);
        let magic = cursor.take(WEIGHTS_MAGIC.len())?;
        if magic != WEIGHTS_MAGIC {
            return Err(RuntimeError::BadMagic);
        }
        let version = cursor.take(1)?[0];
        if version != WEIGHTS_FORMAT_VERSION {
            return Err(RuntimeError::UnsupportedVersion { got: version });
        }
        let config_block = cursor.take_array::<{ CONFIG_BLOCK_BYTES }>()?;
        let config = ModelConfig {
            vocab_size: u32::from_le_bytes(
                config_block[0..4]
                    .try_into()
                    .map_err(|_| truncated_error())?,
            ),
            d_model: u32::from_le_bytes(
                config_block[4..8]
                    .try_into()
                    .map_err(|_| truncated_error())?,
            ),
            n_layers: u32::from_le_bytes(
                config_block[8..12]
                    .try_into()
                    .map_err(|_| truncated_error())?,
            ),
            n_heads: u32::from_le_bytes(
                config_block[12..16]
                    .try_into()
                    .map_err(|_| truncated_error())?,
            ),
            ffn_hidden: u32::from_le_bytes(
                config_block[16..20]
                    .try_into()
                    .map_err(|_| truncated_error())?,
            ),
            max_seq_len: u32::from_le_bytes(
                config_block[20..24]
                    .try_into()
                    .map_err(|_| truncated_error())?,
            ),
            tie_embeddings: match config_block[24] {
                0 => false,
                1 => true,
                got => return Err(RuntimeError::InvalidTieEmbeddingsFlag { got }),
            },
        };
        config.validate()?;

        let tensor_count = u32::from_le_bytes(cursor.take_array::<4>()?);
        let specs = canonical_layout(&config)?;
        if usize::try_from(tensor_count) != Ok(specs.len()) {
            return Err(RuntimeError::TensorCount {
                got: tensor_count,
                expected: specs.len() as u32,
            });
        }

        let mut tensors = Vec::with_capacity(specs.len());
        for spec in &specs {
            let name_len = u32::from_le_bytes(cursor.take_array::<4>()?);
            let name_len =
                usize::try_from(name_len).map_err(|_| RuntimeError::InvalidTensorName)?;
            let name_bytes = cursor.take(name_len)?;
            let name =
                std::str::from_utf8(name_bytes).map_err(|_| RuntimeError::InvalidTensorName)?;
            if name != spec.name {
                return Err(RuntimeError::TensorOrder {
                    found: name.to_string(),
                    expected: spec.name.clone(),
                });
            }
            let elements = u64::from_le_bytes(cursor.take_array::<8>()?);
            if elements != spec.elements {
                return Err(RuntimeError::TensorSize {
                    name: spec.name.clone(),
                    got: elements,
                    expected: spec.elements,
                });
            }
            let byte_len = elements
                .checked_mul(4)
                .ok_or_else(|| RuntimeError::TensorSize {
                    name: spec.name.clone(),
                    got: elements,
                    expected: spec.elements,
                })?;
            let raw = cursor.take(byte_len as usize)?;
            let mut values = Vec::with_capacity(elements.min(i64::MAX as u64) as usize);
            for chunk in raw.chunks_exact(4) {
                let value = f32::from_le_bytes(chunk.try_into().map_err(|_| truncated_error())?);
                if !value.is_finite() {
                    return Err(RuntimeError::NonFiniteTensor {
                        name: spec.name.clone(),
                    });
                }
                values.push(value);
            }
            tensors.push((spec.name.clone(), values));
        }
        Ok(Self { config, tensors })
    }

    /// Validated model config embedded in the file.
    pub const fn config(&self) -> ModelConfig {
        self.config
    }

    /// Returns the tensor with `name`, if present.
    pub fn tensor(&self, name: &str) -> Option<&[f32]> {
        self.tensors
            .iter()
            .find(|(tensor_name, _)| tensor_name == name)
            .map(|(_, values)| values.as_slice())
    }

    /// Canonical tensor names in file order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(name, _)| name.as_str())
    }
}

/// Canonical tensor order and element counts implied by `config` (D-0013).
fn canonical_layout(config: &ModelConfig) -> Result<Vec<TensorSpec>, RuntimeError> {
    config.validate()?;
    let d = u64::from(config.d_model);
    let layers = u64::from(config.n_layers);
    let ffn = u64::from(config.ffn_hidden);
    let vocab = u64::from(config.vocab_size);
    let seq = u64::from(config.max_seq_len);

    let overflow = || RuntimeError::InvalidConfig {
        source: ConfigError::ParameterCountOverflow,
    };

    let mut specs = vec![
        TensorSpec {
            name: "token_embedding.weight".to_string(),
            elements: vocab.checked_mul(d).ok_or_else(overflow)?,
        },
        TensorSpec {
            name: "position_embedding.weight".to_string(),
            elements: seq.checked_mul(d).ok_or_else(overflow)?,
        },
    ];
    for layer in 0..layers {
        let prefix = format!("blocks.{layer}.");
        for (suffix, elements) in [
            ("norm_attn.weight", d),
            ("norm_attn.bias", d),
            (
                "attention.qkv.weight",
                d.checked_mul(3 * d).ok_or_else(overflow)?,
            ),
            (
                "attention.out.weight",
                d.checked_mul(d).ok_or_else(overflow)?,
            ),
            ("norm_mlp.weight", d),
            ("norm_mlp.bias", d),
            ("mlp.0.weight", d.checked_mul(ffn).ok_or_else(overflow)?),
            ("mlp.2.weight", ffn.checked_mul(d).ok_or_else(overflow)?),
        ] {
            specs.push(TensorSpec {
                name: format!("{prefix}{suffix}"),
                elements,
            });
        }
    }
    specs.push(TensorSpec {
        name: "final_norm.weight".to_string(),
        elements: d,
    });
    specs.push(TensorSpec {
        name: "final_norm.bias".to_string(),
        elements: d,
    });
    if !config.tie_embeddings {
        specs.push(TensorSpec {
            name: "lm_head.weight".to_string(),
            elements: vocab.checked_mul(d).ok_or_else(overflow)?,
        });
    }
    Ok(specs)
}

fn truncated_error() -> RuntimeError {
    RuntimeError::Truncated {
        expected: 0,
        found: 0,
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], RuntimeError> {
        self.take_bytes(count as u64)
    }

    /// Takes exactly `N` bytes; `take` guarantees the length, so the copy
    /// cannot fail on malformed input.
    fn take_array<const N: usize>(&mut self) -> Result<[u8; N], RuntimeError> {
        let chunk = self.take(N)?;
        let mut array = [0u8; N];
        array.copy_from_slice(chunk);
        Ok(array)
    }

    fn take_bytes(&mut self, count: u64) -> Result<&'a [u8], RuntimeError> {
        let remaining = (self.bytes.len() - self.position) as u64;
        if count > remaining {
            return Err(RuntimeError::Truncated {
                expected: count,
                found: remaining,
            });
        }
        let start = self.position;
        self.position += count as usize;
        Ok(&self.bytes[start..self.position])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 32_768,
            d_model: 8,
            n_layers: 1,
            n_heads: 2,
            ffn_hidden: 8,
            max_seq_len: 8,
            tie_embeddings: true,
        }
    }

    fn encode_tensor(name: &str, values: &[f32], out: &mut Vec<u8>) {
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn sample_tensors(config: &ModelConfig) -> Vec<(String, Vec<f32>)> {
        let d = u64::from(config.d_model);
        let ffn = u64::from(config.ffn_hidden);
        let vocab = u64::from(config.vocab_size);
        let seq = u64::from(config.max_seq_len);
        let fill = |name: &str, elements: u64| (name.to_string(), vec![0.5f32; elements as usize]);
        let mut tensors = vec![
            fill("token_embedding.weight", vocab * d),
            fill("position_embedding.weight", seq * d),
            fill("blocks.0.norm_attn.weight", d),
            fill("blocks.0.norm_attn.bias", d),
            fill("blocks.0.attention.qkv.weight", 3 * d * d),
            fill("blocks.0.attention.out.weight", d * d),
            fill("blocks.0.norm_mlp.weight", d),
            fill("blocks.0.norm_mlp.bias", d),
            fill("blocks.0.mlp.0.weight", d * ffn),
            fill("blocks.0.mlp.2.weight", ffn * d),
            fill("final_norm.weight", d),
            fill("final_norm.bias", d),
        ];
        if !config.tie_embeddings {
            tensors.push(fill("lm_head.weight", vocab * d));
        }
        tensors
    }

    fn encode_file(config: &ModelConfig, tensors: &[(String, Vec<f32>)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&WEIGHTS_MAGIC);
        out.push(WEIGHTS_FORMAT_VERSION);
        out.extend_from_slice(&config.vocab_size.to_le_bytes());
        out.extend_from_slice(&config.d_model.to_le_bytes());
        out.extend_from_slice(&config.n_layers.to_le_bytes());
        out.extend_from_slice(&config.n_heads.to_le_bytes());
        out.extend_from_slice(&config.ffn_hidden.to_le_bytes());
        out.extend_from_slice(&config.max_seq_len.to_le_bytes());
        out.push(u8::from(config.tie_embeddings));
        out.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for (name, values) in tensors {
            encode_tensor(name, values, &mut out);
        }
        out
    }

    #[test]
    fn valid_file_round_trips() {
        let config = sample_config();
        let bytes = encode_file(&config, &sample_tensors(&config));
        let weights = Weights::from_bytes(&bytes)
            .unwrap_or_else(|error| panic!("valid file rejected: {error}"));
        assert_eq!(weights.config(), config);
        assert_eq!(
            weights
                .tensor("blocks.0.mlp.0.weight")
                .map_or(0, <[f32]>::len),
            64
        );
        assert_eq!(weights.tensor("lm_head.weight"), None);
        assert_eq!(
            weights.tensor_names().next(),
            Some("token_embedding.weight")
        );
    }

    #[test]
    fn untied_file_includes_lm_head() {
        let config = ModelConfig {
            tie_embeddings: false,
            ..sample_config()
        };
        let bytes = encode_file(&config, &sample_tensors(&config));
        let weights = Weights::from_bytes(&bytes)
            .unwrap_or_else(|error| panic!("valid file rejected: {error}"));
        assert_eq!(
            weights.tensor("lm_head.weight").map_or(0, <[f32]>::len),
            32_768 * 8
        );
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = encode_file(&sample_config(), &sample_tensors(&sample_config()));
        bytes[0] = b'X';
        assert!(matches!(
            Weights::from_bytes(&bytes),
            Err(RuntimeError::BadMagic)
        ));
    }

    #[test]
    fn bad_version_is_rejected() {
        let mut bytes = encode_file(&sample_config(), &sample_tensors(&sample_config()));
        bytes[5] = 9;
        assert!(matches!(
            Weights::from_bytes(&bytes),
            Err(RuntimeError::UnsupportedVersion { got: 9 })
        ));
    }

    #[test]
    fn bad_tie_flag_is_rejected() {
        let mut bytes = encode_file(&sample_config(), &sample_tensors(&sample_config()));
        bytes[5 + 1 + 24] = 2;
        assert!(matches!(
            Weights::from_bytes(&bytes),
            Err(RuntimeError::InvalidTieEmbeddingsFlag { got: 2 })
        ));
    }

    #[test]
    fn truncated_file_is_rejected() {
        let bytes = encode_file(&sample_config(), &sample_tensors(&sample_config()));
        assert!(matches!(
            Weights::from_bytes(&bytes[..bytes.len() - 2]),
            Err(RuntimeError::Truncated { .. })
        ));
    }

    #[test]
    fn wrong_tensor_count_is_rejected() {
        let tensors = sample_tensors(&sample_config());
        let mut bytes = encode_file(&sample_config(), &tensors);
        let count_offset = 5 + 1 + 25;
        bytes[count_offset..count_offset + 4]
            .copy_from_slice(&(tensors.len() as u32 + 1).to_le_bytes());
        assert!(matches!(
            Weights::from_bytes(&bytes),
            Err(RuntimeError::TensorCount { .. })
        ));
    }

    #[test]
    fn out_of_order_tensors_are_rejected() {
        let mut tensors = sample_tensors(&sample_config());
        tensors.swap(2, 3);
        let bytes = encode_file(&sample_config(), &tensors);
        assert!(matches!(
            Weights::from_bytes(&bytes),
            Err(RuntimeError::TensorOrder { .. })
        ));
    }

    #[test]
    fn wrong_element_count_is_rejected() {
        let mut tensors = sample_tensors(&sample_config());
        tensors[3].1.pop();
        let bytes = encode_file(&sample_config(), &tensors);
        assert!(matches!(
            Weights::from_bytes(&bytes),
            Err(RuntimeError::TensorSize { .. })
        ));
    }

    #[test]
    fn non_finite_values_are_rejected() {
        let mut tensors = sample_tensors(&sample_config());
        let last = tensors.last_mut().expect("sample has tensors");
        let value = last.1.last_mut().expect("sample tensor has values");
        *value = f32::NAN;
        let bytes = encode_file(&sample_config(), &tensors);
        assert!(matches!(
            Weights::from_bytes(&bytes),
            Err(RuntimeError::NonFiniteTensor { .. })
        ));
    }
}
