//! FP32 inference engine for the decoder-only BYVA architecture (D-0013).
//!
//! The engine mirrors the PyTorch reference trainer (`training/train.py`)
//! numerically: pre-norm blocks, causal multi-head attention without biases,
//! exact-erf GELU MLP without biases, learned absolute positions, and tied
//! token-embedding/output weights by default. Decode steps run against a
//! preallocated KV cache; no per-token allocation happens on the hot path
//! (D-0019). Computation is scalar fp32; SIMD backends arrive behind this
//! interface in Phase 7.

use byva_model::ModelConfig;

use crate::error::RuntimeError;
use crate::weights::Weights;

/// LayerNorm epsilon matching `torch.nn.LayerNorm`'s default.
const LAYER_NORM_EPSILON: f64 = 1e-5;

/// Per-layer weight views consumed by the engine.
#[derive(Debug, Clone)]
struct LayerWeights {
    norm_attn_gain: Vec<f32>,
    norm_attn_bias: Vec<f32>,
    qkv_weight: Vec<f32>,
    attn_out_weight: Vec<f32>,
    norm_mlp_gain: Vec<f32>,
    norm_mlp_bias: Vec<f32>,
    mlp_in_weight: Vec<f32>,
    mlp_out_weight: Vec<f32>,
}

/// A fully loaded inference model.
///
/// Construct with [`InferenceModel::from_weights`]; create a per-generation
/// state with [`InferenceModel::new_session`]; advance it token by token with
/// [`Session::step`].
#[derive(Debug, Clone)]
pub struct InferenceModel {
    config: ModelConfig,
    attention_scale: f32,
    token_embedding: Vec<f32>,
    position_embedding: Vec<f32>,
    layers: Vec<LayerWeights>,
    final_norm_gain: Vec<f32>,
    final_norm_bias: Vec<f32>,
    lm_head: Option<Vec<f32>>,
}

/// Greedy-decode state with preallocated KV cache and activation buffers.
#[derive(Debug)]
pub struct Session {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    ffn_hidden: usize,
    vocab_size: u32,
    max_seq_len: usize,
    position: usize,
    attention_scale: f32,
    token_embedding: Vec<f32>,
    position_embedding: Vec<f32>,
    layers: Vec<LayerWeights>,
    final_norm_gain: Vec<f32>,
    final_norm_bias: Vec<f32>,
    lm_head: Option<Vec<f32>>,
    residual: Vec<f32>,
    normed: Vec<f32>,
    qkv: Vec<f32>,
    attention: Vec<f32>,
    attention_out: Vec<f32>,
    mlp_hidden: Vec<f32>,
    mlp_out: Vec<f32>,
    logits: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    scores: Vec<f32>,
}

impl InferenceModel {
    /// Consumes validated weights and builds an inference model.
    pub fn from_weights(weights: Weights) -> Result<Self, RuntimeError> {
        let config = weights.config();
        config.validate()?;

        let take = |name: &str| -> Result<Vec<f32>, RuntimeError> {
            weights
                .tensor(name)
                .map(<[f32]>::to_vec)
                .ok_or_else(|| RuntimeError::MissingTensor {
                    name: name.to_string(),
                })
        };

        let mut layers = Vec::with_capacity(config.n_layers as usize);
        for layer in 0..config.n_layers {
            layers.push(LayerWeights {
                norm_attn_gain: take(&format!("blocks.{layer}.norm_attn.weight"))?,
                norm_attn_bias: take(&format!("blocks.{layer}.norm_attn.bias"))?,
                qkv_weight: take(&format!("blocks.{layer}.attention.qkv.weight"))?,
                attn_out_weight: take(&format!("blocks.{layer}.attention.out.weight"))?,
                norm_mlp_gain: take(&format!("blocks.{layer}.norm_mlp.weight"))?,
                norm_mlp_bias: take(&format!("blocks.{layer}.norm_mlp.bias"))?,
                mlp_in_weight: take(&format!("blocks.{layer}.mlp.0.weight"))?,
                mlp_out_weight: take(&format!("blocks.{layer}.mlp.2.weight"))?,
            });
        }
        let lm_head = if config.tie_embeddings {
            None
        } else {
            Some(take("lm_head.weight")?)
        };

        Ok(Self {
            attention_scale: 1.0 / f32::sqrt((config.d_model / config.n_heads) as f32),
            config,
            token_embedding: take("token_embedding.weight")?,
            position_embedding: take("position_embedding.weight")?,
            layers,
            final_norm_gain: take("final_norm.weight")?,
            final_norm_bias: take("final_norm.bias")?,
            lm_head,
        })
    }

    /// The validated configuration this model runs.
    pub const fn config(&self) -> ModelConfig {
        self.config
    }

    /// Allocates a fresh greedy-decode session with all buffers preallocated.
    pub fn new_session(&self) -> Session {
        let config = self.config;
        let d = config.d_model as usize;
        let heads = config.n_heads as usize;
        let ffn = config.ffn_hidden as usize;
        let seq = config.max_seq_len as usize;
        let layers_len = self.layers.len();
        Session {
            d_model: d,
            n_heads: heads,
            head_dim: d / heads,
            ffn_hidden: ffn,
            vocab_size: config.vocab_size,
            max_seq_len: seq,
            position: 0,
            attention_scale: self.attention_scale,
            token_embedding: self.token_embedding.clone(),
            position_embedding: self.position_embedding.clone(),
            layers: self.layers.clone(),
            final_norm_gain: self.final_norm_gain.clone(),
            final_norm_bias: self.final_norm_bias.clone(),
            lm_head: self.lm_head.clone(),
            residual: vec![0.0; d],
            normed: vec![0.0; d],
            qkv: vec![0.0; 3 * d],
            attention: vec![0.0; d],
            attention_out: vec![0.0; d],
            mlp_hidden: vec![0.0; ffn],
            mlp_out: vec![0.0; d],
            logits: vec![0.0; config.vocab_size as usize],
            key_cache: vec![0.0; layers_len * seq * d],
            value_cache: vec![0.0; layers_len * seq * d],
            scores: vec![0.0; heads * seq],
        }
    }
}

impl Session {
    /// Position the next stepped token will occupy.
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Logits produced by the most recent [`Session::step`].
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Highest-scoring token id; ties resolve to the lowest id.
    pub fn greedy_token(&self) -> u32 {
        argmax(&self.logits)
    }

    /// Processes one token at the current position and returns the logits.
    ///
    /// All buffers are preallocated; this method performs no heap
    /// allocations on the hot path.
    pub fn step(&mut self, token: u32) -> Result<&[f32], RuntimeError> {
        if token >= self.vocab_size {
            return Err(RuntimeError::TokenOutOfRange {
                token,
                vocab_size: self.vocab_size,
            });
        }
        if self.position >= self.max_seq_len {
            return Err(RuntimeError::ContextOverflow {
                position: self.position as u32,
                max_seq_len: self.max_seq_len as u32,
            });
        }
        let position = self.position;
        let d = self.d_model;

        self.residual
            .copy_from_slice(&self.token_embedding[token as usize * d..(token as usize + 1) * d]);
        for (residual_slot, position_slot) in self
            .residual
            .iter_mut()
            .zip(&self.position_embedding[position * d..(position + 1) * d])
        {
            *residual_slot += position_slot;
        }

        for (layer_index, layer) in self.layers.iter().enumerate() {
            layer_norm(
                &self.residual,
                &layer.norm_attn_gain,
                &layer.norm_attn_bias,
                &mut self.normed,
            );
            matvec(&layer.qkv_weight, 3 * d, &self.normed, &mut self.qkv);

            let cache_base = layer_index * self.max_seq_len * d + position * d;
            self.key_cache[cache_base..cache_base + d].copy_from_slice(&self.qkv[d..2 * d]);
            self.value_cache[cache_base..cache_base + d].copy_from_slice(&self.qkv[2 * d..3 * d]);

            for head in 0..self.n_heads {
                let head_offset = head * self.head_dim;
                let query = &self.qkv[head_offset..head_offset + self.head_dim];
                let head_scores = &mut self.scores
                    [head * self.max_seq_len..head * self.max_seq_len + position + 1];
                for (past, score) in head_scores.iter_mut().enumerate() {
                    let past_base = layer_index * self.max_seq_len * d + past * d + head_offset;
                    let key = &self.key_cache[past_base..past_base + self.head_dim];
                    *score = dot(query, key) * self.attention_scale;
                }
            }
            for head in 0..self.n_heads {
                let head_offset = head * self.head_dim;
                let head_scores = &mut self.scores
                    [head * self.max_seq_len..head * self.max_seq_len + position + 1];
                softmax(head_scores);
                let value_slot = &mut self.attention[head_offset..head_offset + self.head_dim];
                for slot in value_slot.iter_mut() {
                    *slot = 0.0;
                }
                for (past, &weight) in head_scores.iter().enumerate() {
                    let past_base = layer_index * self.max_seq_len * d + past * d + head_offset;
                    let value = &self.value_cache[past_base..past_base + self.head_dim];
                    for (slot, item) in value_slot.iter_mut().zip(value) {
                        *slot += weight * item;
                    }
                }
            }
            matvec(
                &layer.attn_out_weight,
                d,
                &self.attention,
                &mut self.attention_out,
            );
            for (residual_slot, projected) in self.residual.iter_mut().zip(&self.attention_out) {
                *residual_slot += projected;
            }

            layer_norm(
                &self.residual,
                &layer.norm_mlp_gain,
                &layer.norm_mlp_bias,
                &mut self.normed,
            );
            matvec(
                &layer.mlp_in_weight,
                self.ffn_hidden,
                &self.normed,
                &mut self.mlp_hidden,
            );
            for value in self.mlp_hidden.iter_mut() {
                *value = gelu(*value);
            }
            matvec(
                &layer.mlp_out_weight,
                d,
                &self.mlp_hidden,
                &mut self.mlp_out,
            );
            for (residual_slot, projected) in self.residual.iter_mut().zip(&self.mlp_out) {
                *residual_slot += projected;
            }
        }

        layer_norm(
            &self.residual,
            &self.final_norm_gain,
            &self.final_norm_bias,
            &mut self.normed,
        );
        let projection = self.lm_head.as_ref().unwrap_or(&self.token_embedding);
        for (row, logit) in projection.chunks_exact(d).zip(self.logits.iter_mut()) {
            *logit = dot(row, &self.normed);
        }

        self.position += 1;
        Ok(&self.logits)
    }
}

/// Index of the largest value; ties resolve to the lowest index.
fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best = index;
            best_value = value;
        }
    }
    best as u32
}

/// Dot product; the simple loop lets the compiler auto-vectorize.
fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(x, y)| x * y).sum()
}

/// `out = weight * x` for a row-major `[rows, x.len()]` matrix without bias.
fn matvec(weight: &[f32], rows: usize, x: &[f32], out: &mut [f32]) {
    for (row, out_slot) in weight.chunks_exact(x.len()).zip(out[..rows].iter_mut()) {
        *out_slot = dot(row, x);
    }
}

/// LayerNorm in f64 internal precision, matching `torch.nn.LayerNorm`.
fn layer_norm(x: &[f32], gain: &[f32], bias: &[f32], out: &mut [f32]) {
    let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / x.len() as f64;
    let variance = x
        .iter()
        .map(|&v| {
            let centered = f64::from(v) - mean;
            centered * centered
        })
        .sum::<f64>()
        / x.len() as f64;
    let inverse_std = 1.0 / f64::sqrt(variance + LAYER_NORM_EPSILON);
    for ((out_slot, &value), (&gain_slot, &bias_slot)) in
        out.iter_mut().zip(x).zip(gain.iter().zip(bias))
    {
        let normalized = (f64::from(value) - mean) * inverse_std;
        *out_slot = (normalized * f64::from(gain_slot) + f64::from(bias_slot)) as f32;
    }
}

/// Exact-erf GELU matching `torch.nn.GELU()` (no tanh approximation).
fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    let scaled = x * std::f64::consts::FRAC_1_SQRT_2;
    (0.5 * x * (1.0 + erf(scaled))) as f32
}

/// Abramowitz & Stegun 7.1.26 erf approximation (max abs error 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * f64::exp(-x * x);
    sign * y
}

/// In-place numerically stable softmax over the active prefix.
fn softmax(values: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &value in values.iter() {
        if value > max {
            max = value;
        }
    }
    let mut total = 0.0f32;
    for value in values.iter_mut() {
        *value = f32::exp(*value - max);
        total += *value;
    }
    let inverse = 1.0 / total;
    for value in values.iter_mut() {
        *value *= inverse;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcg::fixture_weights;
    use crate::weights::Weights;

    fn parity_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 32_768,
            d_model: 16,
            n_layers: 2,
            n_heads: 2,
            ffn_hidden: 32,
            max_seq_len: 32,
            tie_embeddings: true,
        }
    }

    fn synthetic_weights(config: &ModelConfig) -> Weights {
        fixture_weights(config)
    }

    #[test]
    fn stepping_is_deterministic_across_sessions() {
        let model = InferenceModel::from_weights(synthetic_weights(&parity_config()))
            .expect("model builds");
        let tokens = [11, 5_000, 300, 9_999];
        let mut first = model.new_session();
        let mut second = model.new_session();
        for &token in &tokens {
            first.step(token).expect("step within limits");
            second.step(token).expect("step within limits");
            assert_eq!(first.logits(), second.logits());
        }
    }

    #[test]
    fn token_and_context_limits_return_typed_errors() {
        let model = InferenceModel::from_weights(synthetic_weights(&parity_config()))
            .expect("model builds");
        let mut session = model.new_session();
        assert!(matches!(
            session.step(model.config().vocab_size),
            Err(RuntimeError::TokenOutOfRange { .. })
        ));
        let mut exhausted = model.new_session();
        for _ in 0..model.config().max_seq_len {
            exhausted.step(1).expect("steps within limits");
        }
        assert!(matches!(
            exhausted.step(1),
            Err(RuntimeError::ContextOverflow { .. })
        ));
    }

    #[test]
    fn argmax_tie_resolves_to_lowest_index() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, -2.0]), 1);
        assert_eq!(argmax(&[-5.0, -5.0]), 0);
        assert_eq!(argmax(&[0.0]), 0);
    }
}
