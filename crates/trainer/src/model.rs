//! D-0013 model expressed on the autodiff tape: parameter layout,
//! deterministic init, and the forward/backward-compatible graph build.
//!
//! Parameter names follow the runtime's canonical tensor layout
//! (`blocks.{l}.attention.qkv.weight` and so on) so trainer checkpoints and
//! the converter pipeline stay aligned.

use byva_autodiff::{Graph, NodeId};

use crate::config::ModelConfig;
use crate::error::TrainerError;
use crate::rng::SplitMix64;

/// One parameter tensor: name, shape, and its span in the flat buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamSpec {
    /// Canonical name (runtime layout convention).
    pub name: String,
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Start index in the flat parameter buffer.
    pub offset: usize,
    /// Element count.
    pub len: usize,
}

/// Ordered parameter layout for one model configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamLayout {
    /// Parameters in canonical order.
    pub specs: Vec<ParamSpec>,
    /// Total element count (sum of all specs).
    pub total: usize,
}

impl ParamLayout {
    /// The span of a named parameter.
    pub fn find(&self, name: &str) -> Result<&ParamSpec, TrainerError> {
        self.specs
            .iter()
            .find(|spec| spec.name == name)
            .ok_or_else(|| TrainerError::ConfigInvalid {
                reason: format!("parameter {name} is not part of the layout"),
            })
    }
}

/// Builds the canonical parameter layout (checked arithmetic throughout).
pub fn build_layout(config: &ModelConfig) -> Result<ParamLayout, TrainerError> {
    config.validate()?;
    let overflow = |what: &str| TrainerError::ConfigInvalid {
        reason: format!("parameter size for {what} overflows usize"),
    };
    let d = config.d_model;
    let mut specs = Vec::new();
    let mut offset = 0usize;
    let mut push = |name: &str, shape: &[usize]| -> Result<(), TrainerError> {
        let len = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| overflow(name))?;
        let start = offset;
        offset = offset.checked_add(len).ok_or_else(|| overflow(name))?;
        specs.push(ParamSpec {
            name: name.to_string(),
            shape: shape.to_vec(),
            offset: start,
            len,
        });
        Ok(())
    };
    push("token_embedding.weight", &[config.vocab, d])?;
    push("position_embedding.weight", &[config.seq_len, d])?;
    for layer in 0..config.n_layers {
        push(&format!("blocks.{layer}.norm_attn.weight"), &[d])?;
        push(&format!("blocks.{layer}.norm_attn.bias"), &[d])?;
        push(&format!("blocks.{layer}.attention.qkv.weight"), &[3 * d, d])?;
        push(&format!("blocks.{layer}.attention.out.weight"), &[d, d])?;
        push(&format!("blocks.{layer}.norm_mlp.weight"), &[d])?;
        push(&format!("blocks.{layer}.norm_mlp.bias"), &[d])?;
        push(&format!("blocks.{layer}.mlp.0.weight"), &[config.ffn, d])?;
        push(&format!("blocks.{layer}.mlp.2.weight"), &[d, config.ffn])?;
    }
    Ok(ParamLayout {
        specs,
        total: offset,
    })
}

/// Deterministic init matching the reference semantics: Linear and Embedding
/// weights draw N(0, 0.02); LayerNorm gains start at 1 and biases at 0.
pub fn init_params(
    config: &ModelConfig,
    layout: &ParamLayout,
    seed: u64,
) -> Result<Vec<f32>, TrainerError> {
    config.validate()?;
    let mut rng = SplitMix64::new(seed);
    let mut params = vec![0.0f32; layout.total];
    for spec in &layout.specs {
        let slot = &mut params[spec.offset..spec.offset + spec.len];
        if spec.name.ends_with(".weight") && spec.name.contains(".norm_") {
            slot.fill(1.0);
        } else if spec.name.ends_with(".bias") {
            slot.fill(0.0);
        } else {
            for value in slot.iter_mut() {
                *value = (rng.next_normal() * 0.02) as f32;
            }
        }
    }
    Ok(params)
}

/// `y = x @ W^T` for a PyTorch-convention Linear weight `[out, in]`, via a
/// single-batch transposed matmul (the tape has no standalone transpose op).
fn dense(
    graph: &mut Graph,
    x: NodeId,
    weight: NodeId,
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Result<NodeId, TrainerError> {
    let batched_x = graph.reshape(x, &[1, rows, in_features])?;
    let batched_weight = graph.reshape(weight, &[1, out_features, in_features])?;
    let product = graph.batched_matmul_nt(batched_x, batched_weight)?;
    Ok(graph.reshape(product, &[rows, out_features])?)
}

/// Builds the full D-0013 forward pass and returns the scalar loss node
/// (tied-logit-free softmax cross-entropy, K2).
///
/// `leaves` must be aligned with `layout.specs`; `tokens` must contain
/// exactly `batch * seq_len` ids.
pub fn forward(
    graph: &mut Graph,
    config: &ModelConfig,
    layout: &ParamLayout,
    leaves: &[NodeId],
    tokens: &[u32],
    targets: &[u32],
) -> Result<NodeId, TrainerError> {
    if leaves.len() != layout.specs.len() {
        return Err(TrainerError::ConfigInvalid {
            reason: format!(
                "forward expected {} parameter leaves, found {}",
                layout.specs.len(),
                leaves.len()
            ),
        });
    }
    if tokens.is_empty() || tokens.len() % config.seq_len != 0 {
        return Err(TrainerError::DataInvalid {
            reason: format!(
                "token count {} must be a positive multiple of seq_len {}",
                tokens.len(),
                config.seq_len
            ),
        });
    }
    if targets.len() != tokens.len() {
        return Err(TrainerError::DataInvalid {
            reason: format!("{} targets for {} tokens", targets.len(), tokens.len()),
        });
    }
    let find = |name: &str| -> Result<NodeId, TrainerError> {
        let index = layout
            .specs
            .iter()
            .position(|spec| spec.name == name)
            .ok_or_else(|| TrainerError::ConfigInvalid {
                reason: format!("parameter {name} is not part of the layout"),
            })?;
        Ok(leaves[index])
    };

    let batch = tokens.len() / config.seq_len;
    let rows = tokens.len();
    let d = config.d_model;
    let heads = config.n_heads;
    let head_dim = config.head_dim();

    let token_embedding = find("token_embedding.weight")?;
    let mut x = graph.embed(token_embedding, tokens)?;
    x = graph.add_broadcast_rows(x, find("position_embedding.weight")?)?;

    for layer in 0..config.n_layers {
        let norm_attn_gain = find(&format!("blocks.{layer}.norm_attn.weight"))?;
        let norm_attn_bias = find(&format!("blocks.{layer}.norm_attn.bias"))?;
        let qkv_weight = find(&format!("blocks.{layer}.attention.qkv.weight"))?;
        let out_weight = find(&format!("blocks.{layer}.attention.out.weight"))?;
        let norm_mlp_gain = find(&format!("blocks.{layer}.norm_mlp.weight"))?;
        let norm_mlp_bias = find(&format!("blocks.{layer}.norm_mlp.bias"))?;
        let mlp_in = find(&format!("blocks.{layer}.mlp.0.weight"))?;
        let mlp_out = find(&format!("blocks.{layer}.mlp.2.weight"))?;

        let normed = graph.layer_norm(x, norm_attn_gain, norm_attn_bias, 1e-5)?;
        let qkv = dense(graph, normed, qkv_weight, rows, d, 3 * d)?;
        let scale = (1.0 / (head_dim as f64).sqrt()) as f32;

        let q = graph.slice_cols(qkv, 0, d)?;
        let k = graph.slice_cols(qkv, d, d)?;
        let v = graph.slice_cols(qkv, 2 * d, d)?;
        let q = graph.mul_scalar(q, scale)?;
        let q = graph.reshape(q, &[batch, config.seq_len, heads, head_dim])?;
        let k = graph.reshape(k, &[batch, config.seq_len, heads, head_dim])?;
        let v = graph.reshape(v, &[batch, config.seq_len, heads, head_dim])?;
        let q = graph.heads_to_rows(q, heads)?;
        let k = graph.heads_to_rows(k, heads)?;
        let v = graph.heads_to_rows(v, heads)?;

        let scores = graph.batched_matmul_nt(q, k)?;
        let probs = graph.causal_softmax(scores)?;
        let context = graph.batched_matmul_nn(probs, v)?;
        let context = graph.rows_to_heads(context, heads)?;
        let context = graph.reshape(context, &[rows, d])?;
        let attended = dense(graph, context, out_weight, rows, d, d)?;
        x = graph.add(x, attended)?;

        let normed = graph.layer_norm(x, norm_mlp_gain, norm_mlp_bias, 1e-5)?;
        let hidden = dense(graph, normed, mlp_in, rows, d, config.ffn)?;
        let activated = graph.gelu(hidden)?;
        let projected = dense(graph, activated, mlp_out, rows, config.ffn, d)?;
        x = graph.add(x, projected)?;
    }

    Ok(graph.cross_entropy_tied(x, token_embedding, targets)?)
}
