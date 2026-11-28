//! Versioned run configuration for the ne2410 training loop (D-0022).
//!
//! Hyperparameters live here, not hardcoded in the loop; the whole config is
//! serialized into every checkpoint so a resume reproduces the run exactly.

use serde::{Deserialize, Serialize};

use crate::error::TrainerError;

/// D-0013 frozen model architecture (decoder-only, pre-norm, causal MHA
/// without biases, exact-erf GELU, learned absolute positions, tied
/// token-embedding/output).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size (32768 for the BPE 32k artifact).
    pub vocab: usize,
    /// Model width.
    pub d_model: usize,
    /// Number of pre-norm blocks.
    pub n_layers: usize,
    /// Attention heads (`d_model` must divide evenly).
    pub n_heads: usize,
    /// Feed-forward hidden width.
    pub ffn: usize,
    /// Sequence length (learned positions are allocated for exactly this).
    pub seq_len: usize,
}

impl ModelConfig {
    /// The D-0013 TINY preset (CPU/CI).
    pub fn tiny() -> Self {
        Self {
            vocab: 32_768,
            d_model: 256,
            n_layers: 4,
            n_heads: 8,
            ffn: 1024,
            seq_len: 512,
        }
    }

    /// The D-0013 SMALL preset (GPU-class runs).
    pub fn small() -> Self {
        Self {
            vocab: 32_768,
            d_model: 512,
            n_layers: 8,
            n_heads: 8,
            ffn: 2048,
            seq_len: 1024,
        }
    }

    /// Width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Validates the architecture against the D-0013 constraints.
    pub fn validate(&self) -> Result<(), TrainerError> {
        let invalid = |reason: String| TrainerError::ConfigInvalid { reason };
        if self.vocab == 0
            || self.d_model == 0
            || self.n_layers == 0
            || self.n_heads == 0
            || self.ffn == 0
            || self.seq_len < 2
        {
            return Err(invalid(format!(
                "model dimensions must be positive and seq_len at least 2, got vocab {}, d_model {}, n_layers {}, n_heads {}, ffn {}, seq_len {}",
                self.vocab, self.d_model, self.n_layers, self.n_heads, self.ffn, self.seq_len
            )));
        }
        if self.d_model % self.n_heads != 0 {
            return Err(invalid(format!(
                "d_model {} must divide evenly across n_heads {}",
                self.d_model, self.n_heads
            )));
        }
        Ok(())
    }
}

/// Which D-0022 optimizer option the run uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerKind {
    /// Option 1 (default, parity baseline).
    AdamW,
    /// Option 2, BYVA's own design (`--optimizer signtrust`).
    SignTrust,
}

/// Full training run configuration; serialized into checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Frozen model architecture.
    pub model: ModelConfig,
    /// Which optimizer option to run.
    pub optimizer: OptimizerKind,
    /// Base learning rate (schedule target).
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    /// AdamW/SignTrust first-moment decay.
    #[serde(default = "default_beta1")]
    pub beta1: f64,
    /// AdamW second-moment decay (unused by SignTrust).
    #[serde(default = "default_beta2")]
    pub beta2: f64,
    /// AdamW epsilon (unused by SignTrust).
    #[serde(default = "default_eps")]
    pub eps: f64,
    /// Decoupled weight decay coefficient.
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    /// SignTrust trust-ratio clamp lower bound (provisional per D-0022).
    #[serde(default = "default_clamp_min")]
    pub clamp_min: f64,
    /// SignTrust trust-ratio clamp upper bound (provisional per D-0022).
    #[serde(default = "default_clamp_max")]
    pub clamp_max: f64,
    /// Total steps the run targets.
    pub steps: u64,
    /// K4 token budget: exactly this many tokens feed every step
    /// (FLOPs per step are constant by construction).
    pub token_budget: usize,
    /// Run seed; derives the deterministic init stream.
    pub seed: u64,
    /// Linear warmup steps before cosine decay (0 disables warmup).
    #[serde(default)]
    pub warmup_steps: u64,
    /// Cosine decay floor as a ratio of the base learning rate.
    #[serde(default = "default_floor_ratio")]
    pub floor_ratio: f64,
    /// Evaluate held-out loss every this many steps (0 disables eval).
    #[serde(default = "default_eval_every")]
    pub eval_every: u64,
    /// Number of held-out rows per evaluation (capped by availability).
    #[serde(default = "default_eval_rows")]
    pub eval_rows: usize,
    /// K5 mixture target share of Indonesian windows (None keeps the natural
    /// corpus order; set to enable the id:en mixture).
    #[serde(default)]
    pub mixture_target_id: Option<f64>,
    /// K5 adaptation band around the target share.
    #[serde(default = "default_mixture_band")]
    pub mixture_band: f64,
}

fn default_learning_rate() -> f64 {
    1e-3
}

fn default_beta1() -> f64 {
    0.9
}

fn default_beta2() -> f64 {
    0.999
}

fn default_eps() -> f64 {
    1e-8
}

fn default_weight_decay() -> f64 {
    0.0
}

fn default_clamp_min() -> f64 {
    0.25
}

fn default_clamp_max() -> f64 {
    4.0
}

fn default_floor_ratio() -> f64 {
    0.1
}

fn default_eval_every() -> u64 {
    50
}

fn default_eval_rows() -> usize {
    200
}

fn default_mixture_band() -> f64 {
    0.1
}

impl TrainConfig {
    /// Batch size implied by the K4 token budget (constant every step).
    pub fn batch_size(&self) -> Result<usize, TrainerError> {
        if self.token_budget % self.model.seq_len != 0 {
            return Err(TrainerError::ConfigInvalid {
                reason: format!(
                    "token_budget {} must be a multiple of seq_len {}",
                    self.token_budget, self.model.seq_len
                ),
            });
        }
        Ok(self.token_budget / self.model.seq_len)
    }

    /// Validates the whole run configuration.
    pub fn validate(&self) -> Result<(), TrainerError> {
        self.model.validate()?;
        let invalid = |reason: String| TrainerError::ConfigInvalid { reason };
        self.batch_size()?;
        if self.steps == 0 {
            return Err(invalid("steps must be positive".to_string()));
        }
        if self.learning_rate <= 0.0 {
            return Err(invalid(format!(
                "learning_rate must be positive, found {}",
                self.learning_rate
            )));
        }
        if !((0.0..1.0).contains(&self.beta1) && (0.0..1.0).contains(&self.beta2)) {
            return Err(invalid(format!(
                "betas must be in [0, 1), found {} and {}",
                self.beta1, self.beta2
            )));
        }
        if self.eps <= 0.0 {
            return Err(invalid(format!("eps must be positive, found {}", self.eps)));
        }
        if self.weight_decay < 0.0 {
            return Err(invalid(format!(
                "weight_decay must be non-negative, found {}",
                self.weight_decay
            )));
        }
        if !(self.clamp_min > 0.0 && self.clamp_min <= self.clamp_max) {
            return Err(invalid(format!(
                "clamp band must satisfy 0 < clamp_min <= clamp_max, found {} and {}",
                self.clamp_min, self.clamp_max
            )));
        }
        if !(0.0 < self.floor_ratio && self.floor_ratio <= 1.0) {
            return Err(invalid(format!(
                "floor_ratio must be in (0, 1], found {}",
                self.floor_ratio
            )));
        }
        if self.eval_rows == 0 {
            return Err(invalid("eval_rows must be positive".to_string()));
        }
        if let Some(target) = self.mixture_target_id {
            if !(0.0 < target && target < 1.0) {
                return Err(invalid(format!(
                    "mixture_target_id must be in (0, 1), found {}",
                    target
                )));
            }
            if self.mixture_band < 0.0
                || target - self.mixture_band <= 0.0
                || target + self.mixture_band >= 1.0
            {
                return Err(invalid(format!(
                    "mixture_band must keep target {} inside (0, 1), found {}",
                    target, self.mixture_band
                )));
            }
        }
        Ok(())
    }

    /// The scheduled learning rate for the 1-based update `step`:
    /// linear warmup, then cosine decay to `learning_rate * floor_ratio`.
    pub fn learning_rate_at(&self, step: u64) -> f64 {
        let base = self.learning_rate;
        if self.warmup_steps > 0 && step <= self.warmup_steps {
            return base * (step as f64 / self.warmup_steps as f64);
        }
        if self.steps <= self.warmup_steps {
            return base;
        }
        let floor = base * self.floor_ratio;
        let progress =
            ((step - self.warmup_steps) as f64 / (self.steps - self.warmup_steps) as f64).min(1.0);
        floor + (base - floor) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}
