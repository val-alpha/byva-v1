//! Shared BYVA model configuration and architecture definitions.
//!
//! The architecture is a decoder-only transformer with pre-norm blocks, learned
//! absolute positions, and (by default) tied token embeddings and output weights.
//! Phase 4 conversion and the Phase 5 Rust runtime consume these definitions;
//! no tensor math lives here.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use byva_tokenizer::{MAX_VOCAB_SIZE, MIN_VOCAB_SIZE};

/// Validation and arithmetic errors for [`ModelConfig`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConfigError {
    /// Vocabulary is outside the tokenizer's supported range.
    #[error("vocab_size {got} is outside the supported range {min}..={max}")]
    VocabSizeOutOfRange { got: u32, min: u32, max: u32 },
    /// A dimension that must be positive was zero.
    #[error("{name} must be greater than zero")]
    DimensionMustBePositive { name: &'static str },
    /// The model dimension must be divisible by the head count.
    #[error("d_model {d_model} is not divisible by n_heads {n_heads}")]
    DimNotDivisibleByHeads { d_model: u32, n_heads: u32 },
    /// The parameter count overflowed u64.
    #[error("parameter count overflowed u64")]
    ParameterCountOverflow,
}

/// Decoder-only transformer hyperparameters.
///
/// Fields use `u32` because they describe untrusted external metadata when
/// loaded from a checkpoint manifest; all arithmetic is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size; must stay inside the tokenizer's supported range.
    pub vocab_size: u32,
    /// Residual-stream width.
    pub d_model: u32,
    /// Transformer block count.
    pub n_layers: u32,
    /// Attention head count; `d_model` must be divisible by this.
    pub n_heads: u32,
    /// Feed-forward hidden width.
    pub ffn_hidden: u32,
    /// Maximum learned-position sequence length.
    pub max_seq_len: u32,
    /// Whether the output projection reuses the token embedding matrix.
    pub tie_embeddings: bool,
}

impl ModelConfig {
    /// Validates vocabulary range, positive dimensions, and head divisibility.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.vocab_size < MIN_VOCAB_SIZE || self.vocab_size > MAX_VOCAB_SIZE {
            return Err(ConfigError::VocabSizeOutOfRange {
                got: self.vocab_size,
                min: MIN_VOCAB_SIZE,
                max: MAX_VOCAB_SIZE,
            });
        }
        for (name, value) in [
            ("d_model", self.d_model),
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("ffn_hidden", self.ffn_hidden),
            ("max_seq_len", self.max_seq_len),
        ] {
            if value == 0 {
                return Err(ConfigError::DimensionMustBePositive { name });
            }
        }
        if self.d_model % self.n_heads != 0 {
            return Err(ConfigError::DimNotDivisibleByHeads {
                d_model: self.d_model,
                n_heads: self.n_heads,
            });
        }
        Ok(())
    }

    /// Per-head dimension, with divisibility enforced by [`ModelConfig::validate`].
    pub fn head_dim(&self) -> Result<u32, ConfigError> {
        self.validate()?;
        Ok(self.d_model / self.n_heads)
    }

    /// Total parameter count including tied-output sharing when enabled.
    ///
    /// Layout: token embedding, learned positions, per-layer attention
    /// (`4 * d^2`), MLP (`2 * d * ffn`), two per-layer LayerNorms (`4 * d`) and
    /// one final LayerNorm (`2 * d`), plus an untied output projection when
    /// `tie_embeddings` is false.
    pub fn parameter_count(&self) -> Result<u64, ConfigError> {
        self.validate()?;
        let vocab = u64::from(self.vocab_size);
        let d = u64::from(self.d_model);
        let layers = u64::from(self.n_layers);
        let ffn = u64::from(self.ffn_hidden);
        let seq = u64::from(self.max_seq_len);

        let embedding = vocab
            .checked_mul(d)
            .ok_or(ConfigError::ParameterCountOverflow)?;
        let positions = seq
            .checked_mul(d)
            .ok_or(ConfigError::ParameterCountOverflow)?;
        let attention = d
            .checked_mul(d)
            .and_then(|d_squared| d_squared.checked_mul(4))
            .ok_or(ConfigError::ParameterCountOverflow)?;
        let mlp = d
            .checked_mul(ffn)
            .and_then(|d_ffn| d_ffn.checked_mul(2))
            .ok_or(ConfigError::ParameterCountOverflow)?;
        let layer_norm_params = d
            .checked_mul(2)
            .ok_or(ConfigError::ParameterCountOverflow)?;
        let norm_total = layer_norm_params
            .checked_mul(layers)
            .and_then(|per_layer_norms| per_layer_norms.checked_mul(2))
            .and_then(|per_layer_norms| per_layer_norms.checked_add(layer_norm_params))
            .ok_or(ConfigError::ParameterCountOverflow)?;
        let per_layer = attention
            .checked_add(mlp)
            .ok_or(ConfigError::ParameterCountOverflow)?;
        let blocks = per_layer
            .checked_mul(layers)
            .ok_or(ConfigError::ParameterCountOverflow)?;

        let total = embedding
            .checked_add(positions)
            .and_then(|running| running.checked_add(blocks))
            .and_then(|running| running.checked_add(norm_total))
            .ok_or(ConfigError::ParameterCountOverflow)?;
        if self.tie_embeddings {
            Ok(total)
        } else {
            total
                .checked_add(embedding)
                .ok_or(ConfigError::ParameterCountOverflow)
        }
    }
}

/// Training-target configuration for the Colab T4 (16 GB FP16) plan; D-0013.
pub const SMALL_CONFIG: ModelConfig = ModelConfig {
    vocab_size: 32_768,
    d_model: 512,
    n_layers: 8,
    n_heads: 8,
    ffn_hidden: 2048,
    max_seq_len: 1024,
    tie_embeddings: true,
};

/// CPU/CI configuration for smoke training runs and benchmarks; D-0013.
pub const TINY_CONFIG: ModelConfig = ModelConfig {
    vocab_size: 32_768,
    d_model: 256,
    n_layers: 4,
    n_heads: 8,
    ffn_hidden: 1024,
    max_seq_len: 512,
    tie_embeddings: true,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_and_tiny_configs_validate() {
        assert_eq!(SMALL_CONFIG.validate(), Ok(()));
        assert_eq!(TINY_CONFIG.validate(), Ok(()));
        assert_eq!(SMALL_CONFIG.head_dim(), Ok(64));
        assert_eq!(TINY_CONFIG.head_dim(), Ok(32));
    }

    #[test]
    fn parameter_counts_match_frozen_golden_values() {
        assert_eq!(SMALL_CONFIG.parameter_count(), Ok(42_484_736));
        assert_eq!(TINY_CONFIG.parameter_count(), Ok(11_670_016));
        let untied = ModelConfig {
            tie_embeddings: false,
            ..SMALL_CONFIG
        };
        assert_eq!(untied.parameter_count(), Ok(59_261_952));
    }

    #[test]
    fn invalid_configs_are_rejected_with_typed_errors() {
        let out_of_range = ModelConfig {
            vocab_size: 100,
            ..TINY_CONFIG
        };
        assert_eq!(
            out_of_range.validate(),
            Err(ConfigError::VocabSizeOutOfRange {
                got: 100,
                min: MIN_VOCAB_SIZE,
                max: MAX_VOCAB_SIZE
            })
        );
        let not_divisible = ModelConfig {
            d_model: 100,
            n_heads: 8,
            ..TINY_CONFIG
        };
        assert_eq!(
            not_divisible.validate(),
            Err(ConfigError::DimNotDivisibleByHeads {
                d_model: 100,
                n_heads: 8
            })
        );
        let zero = ModelConfig {
            n_layers: 0,
            ..TINY_CONFIG
        };
        assert_eq!(
            zero.validate(),
            Err(ConfigError::DimensionMustBePositive { name: "n_layers" })
        );
        assert_eq!(
            zero.parameter_count(),
            Err(ConfigError::DimensionMustBePositive { name: "n_layers" })
        );
    }

    #[test]
    fn config_serde_round_trips() {
        let json = serde_json::to_string(&SMALL_CONFIG);
        assert!(json.is_ok());
        let parsed: Result<ModelConfig, _> = serde_json::from_str(&json.unwrap_or_default());
        assert_eq!(parsed.ok(), Some(SMALL_CONFIG));
    }
}
