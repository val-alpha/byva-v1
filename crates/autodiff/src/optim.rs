//! Optimizer updates for the Rust training path (D-0022).
//!
//! Two options, both fully deterministic (no RNG):
//!
//! - `AdamW` (Option 1, default): decoupled weight decay, bias-corrected
//!   first and second moments.
//! - `SignTrust` (Option 2, BYVA's own design, K3): one EMA momentum buffer
//!   per parameter (half of AdamW's optimizer state), update
//!   `w -= lr * sign(m) * c_l` with per-layer trust ratio
//!   `c_l = rms(w) / rms(m)` clamped to a fixed band. The trust ratio plays
//!   the role of Adam's second moment, paid once per layer instead of once
//!   per parameter; because the sign and the ratio are both scale-free, no
//!   bias correction is needed.

use thiserror::Error;

/// Errors produced by optimizer construction and update steps.
#[derive(Debug, Error)]
pub enum OptimError {
    /// Parameter or gradient length does not match the optimizer state.
    #[error("optimizer state was created for {expected} parameters but {found} were provided")]
    LengthMismatch {
        /// Parameter count the state was created for.
        expected: usize,
        /// Number of parameters supplied.
        found: usize,
    },
    /// The layer index has no state.
    #[error("optimizer has state for {layers} layers but layer {layer} was requested")]
    UnknownLayer {
        /// Number of layers with state.
        layers: usize,
        /// Requested layer index.
        layer: usize,
    },
    /// A configuration value is outside its valid range.
    #[error("invalid optimizer configuration: {reason}")]
    InvalidConfig {
        /// What the validation rejected.
        reason: String,
    },
}

/// Configuration for `AdamW`.
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    /// Learning rate.
    pub lr: f32,
    /// Exponential decay rate for the first moment.
    pub beta1: f32,
    /// Exponential decay rate for the second moment.
    pub beta2: f32,
    /// Epsilon added to the square-root of the second moment.
    pub eps: f32,
    /// Decoupled weight decay coefficient.
    pub weight_decay: f32,
}

/// AdamW optimizer state (Option 1, default; D-0022).
pub struct AdamW {
    config: AdamWConfig,
    m: Vec<f32>,
    v: Vec<f32>,
    step: u64,
}

impl AdamW {
    /// Creates state for `param_count` parameters and validates the config.
    pub fn new(config: AdamWConfig, param_count: usize) -> Result<Self, OptimError> {
        validate_adamw(&config)?;
        Ok(Self {
            config,
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
            step: 0,
        })
    }

    /// Advances one update step over the flat parameter vector.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) -> Result<(), OptimError> {
        if params.len() != self.m.len() || grads.len() != self.m.len() {
            return Err(OptimError::LengthMismatch {
                expected: self.m.len(),
                found: params.len(),
            });
        }
        self.step += 1;
        let t = self.step as f64;
        let bias1 = 1.0 - f64::from(self.config.beta1).powf(t);
        let bias2 = 1.0 - f64::from(self.config.beta2).powf(t);
        for index in 0..params.len() {
            self.m[index] =
                self.config.beta1 * self.m[index] + (1.0 - self.config.beta1) * grads[index];
            self.v[index] = self.config.beta2 * self.v[index]
                + (1.0 - self.config.beta2) * grads[index] * grads[index];
            let m_hat = f64::from(self.m[index]) / bias1;
            let v_hat = f64::from(self.v[index]) / bias2;
            let decay = 1.0 - self.config.lr * self.config.weight_decay;
            params[index] = (f64::from(params[index]) * f64::from(decay)
                - f64::from(self.config.lr) * m_hat / (v_hat.sqrt() + f64::from(self.config.eps)))
                as f32;
        }
        Ok(())
    }

    /// Number of update steps taken.
    pub fn steps(&self) -> u64 {
        self.step
    }

    /// Overrides the learning rate (used by the trainer's LR schedule).
    pub fn set_learning_rate(&mut self, lr: f32) -> Result<(), OptimError> {
        if lr <= 0.0 {
            return Err(OptimError::InvalidConfig {
                reason: format!("lr must be positive, found {lr}"),
            });
        }
        self.config.lr = lr;
        Ok(())
    }

    /// Immutable snapshot of the optimizer state for checkpointing.
    pub fn state(&self) -> AdamWState {
        AdamWState {
            m: self.m.clone(),
            v: self.v.clone(),
            step: self.step,
        }
    }

    /// Restores a snapshot; lengths must match the current state.
    pub fn restore(&mut self, state: AdamWState) -> Result<(), OptimError> {
        if state.m.len() != self.m.len() || state.v.len() != self.v.len() {
            return Err(OptimError::LengthMismatch {
                expected: self.m.len(),
                found: state.m.len(),
            });
        }
        self.m = state.m;
        self.v = state.v;
        self.step = state.step;
        Ok(())
    }
}

/// Serializable snapshot of `AdamW` state (moments and step count).
#[derive(Debug, Clone, PartialEq)]
pub struct AdamWState {
    /// First-moment buffer.
    pub m: Vec<f32>,
    /// Second-moment buffer.
    pub v: Vec<f32>,
    /// Number of update steps taken.
    pub step: u64,
}

fn validate_adamw(config: &AdamWConfig) -> Result<(), OptimError> {
    if config.lr <= 0.0 {
        return Err(OptimError::InvalidConfig {
            reason: format!("lr must be positive, found {}", config.lr),
        });
    }
    if !((0.0..1.0).contains(&config.beta1) && (0.0..1.0).contains(&config.beta2)) {
        return Err(OptimError::InvalidConfig {
            reason: format!(
                "betas must be in [0, 1), found {} and {}",
                config.beta1, config.beta2
            ),
        });
    }
    if config.eps <= 0.0 {
        return Err(OptimError::InvalidConfig {
            reason: format!("eps must be positive, found {}", config.eps),
        });
    }
    if config.weight_decay < 0.0 {
        return Err(OptimError::InvalidConfig {
            reason: format!(
                "weight_decay must be non-negative, found {}",
                config.weight_decay
            ),
        });
    }
    Ok(())
}

/// Configuration for `SignTrust`. The clamp band is provisional; per D-0022
/// it is fixed in the decision's first revision after the first measured
/// comparison against AdamW.
#[derive(Debug, Clone)]
pub struct SignTrustConfig {
    /// Learning rate.
    pub lr: f32,
    /// Exponential decay rate for the momentum buffer.
    pub beta1: f32,
    /// Decoupled weight decay coefficient.
    pub weight_decay: f32,
    /// Lower bound of the trust-ratio clamp band.
    pub clamp_min: f32,
    /// Upper bound of the trust-ratio clamp band.
    pub clamp_max: f32,
}

impl Default for SignTrustConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            beta1: 0.9,
            weight_decay: 0.0,
            clamp_min: 0.25,
            clamp_max: 4.0,
        }
    }
}

/// SignTrust optimizer state (Option 2, BYVA's own design; D-0022).
/// One momentum buffer per layer; the trust ratio is computed per layer.
pub struct SignTrust {
    config: SignTrustConfig,
    momenta: Vec<Vec<f32>>,
    steps: u64,
}

impl SignTrust {
    /// Creates per-layer state from each layer's parameter count.
    pub fn new(config: SignTrustConfig, layer_param_counts: &[usize]) -> Result<Self, OptimError> {
        validate_signtrust(&config)?;
        Ok(Self {
            config,
            momenta: layer_param_counts
                .iter()
                .map(|&len| vec![0.0; len])
                .collect(),
            steps: 0,
        })
    }

    /// Advances one update step for a single layer's parameter tensor.
    pub fn step_layer(
        &mut self,
        layer: usize,
        params: &mut [f32],
        grads: &[f32],
    ) -> Result<(), OptimError> {
        let layers = self.momenta.len();
        let momentum = self
            .momenta
            .get_mut(layer)
            .ok_or(OptimError::UnknownLayer { layers, layer })?;
        if params.len() != momentum.len() || grads.len() != momentum.len() {
            return Err(OptimError::LengthMismatch {
                expected: momentum.len(),
                found: params.len(),
            });
        }
        self.steps += 1;
        for (slot, &g) in momentum.iter_mut().zip(grads) {
            *slot = self.config.beta1 * *slot + (1.0 - self.config.beta1) * g;
        }
        let rms_w = rms(params);
        let rms_m = rms(momentum);
        let trust = if rms_m == 0.0 {
            1.0
        } else {
            (rms_w / rms_m).clamp(self.config.clamp_min, self.config.clamp_max)
        };
        let decay = 1.0 - self.config.lr * self.config.weight_decay;
        for (param, &m) in params.iter_mut().zip(momentum.iter()) {
            let sign = if m > 0.0 {
                1.0
            } else if m < 0.0 {
                -1.0
            } else {
                0.0
            };
            *param = (f64::from(*param) * f64::from(decay)
                - f64::from(self.config.lr) * sign * f64::from(trust)) as f32;
        }
        Ok(())
    }

    /// Number of update steps taken across all layers.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Overrides the learning rate (used by the trainer's LR schedule).
    pub fn set_learning_rate(&mut self, lr: f32) -> Result<(), OptimError> {
        if lr <= 0.0 {
            return Err(OptimError::InvalidConfig {
                reason: format!("lr must be positive, found {lr}"),
            });
        }
        self.config.lr = lr;
        Ok(())
    }

    /// Immutable snapshot of the optimizer state for checkpointing.
    pub fn state(&self) -> SignTrustState {
        SignTrustState {
            momenta: self.momenta.clone(),
            steps: self.steps,
        }
    }

    /// Restores a snapshot; the layer layout must match the current state.
    pub fn restore(&mut self, state: SignTrustState) -> Result<(), OptimError> {
        if state.momenta.len() != self.momenta.len() {
            return Err(OptimError::UnknownLayer {
                layers: self.momenta.len(),
                layer: state.momenta.len(),
            });
        }
        for (momentum, restored) in self.momenta.iter_mut().zip(&state.momenta) {
            if momentum.len() != restored.len() {
                return Err(OptimError::LengthMismatch {
                    expected: momentum.len(),
                    found: restored.len(),
                });
            }
        }
        self.momenta = state.momenta;
        self.steps = state.steps;
        Ok(())
    }
}

/// Serializable snapshot of `SignTrust` state (per-layer momenta).
#[derive(Debug, Clone, PartialEq)]
pub struct SignTrustState {
    /// One momentum buffer per layer.
    pub momenta: Vec<Vec<f32>>,
    /// Number of layer updates taken.
    pub steps: u64,
}

fn validate_signtrust(config: &SignTrustConfig) -> Result<(), OptimError> {
    if config.lr <= 0.0 {
        return Err(OptimError::InvalidConfig {
            reason: format!("lr must be positive, found {}", config.lr),
        });
    }
    if !(0.0..1.0).contains(&config.beta1) {
        return Err(OptimError::InvalidConfig {
            reason: format!("beta1 must be in [0, 1), found {}", config.beta1),
        });
    }
    if config.weight_decay < 0.0 {
        return Err(OptimError::InvalidConfig {
            reason: format!(
                "weight_decay must be non-negative, found {}",
                config.weight_decay
            ),
        });
    }
    if !(config.clamp_min > 0.0 && config.clamp_min <= config.clamp_max) {
        return Err(OptimError::InvalidConfig {
            reason: format!(
                "clamp band must satisfy 0 < clamp_min <= clamp_max, found {} and {}",
                config.clamp_min, config.clamp_max
            ),
        });
    }
    Ok(())
}

fn rms(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    (sum / values.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adamw_first_step_matches_bias_corrected_update() {
        let mut optimizer = AdamW::new(
            AdamWConfig {
                lr: 0.1,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
            1,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let mut params = [1.0f32];
        optimizer
            .step(&mut params, &[1.0])
            .unwrap_or_else(|e| panic!("{e}"));
        // With eps = 1e-8 the first step is lr * mhat / (sqrt(vhat) + eps).
        assert!((params[0] - 0.9).abs() < 1e-6, "got {}", params[0]);
        assert_eq!(optimizer.steps(), 1);
    }

    #[test]
    fn adamw_decoupled_weight_decay_scales_params() {
        let mut optimizer = AdamW::new(
            AdamWConfig {
                lr: 0.1,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 1.0,
            },
            1,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let mut params = [1.0f32];
        optimizer
            .step(&mut params, &[0.0])
            .unwrap_or_else(|e| panic!("{e}"));
        assert!((params[0] - 0.9).abs() < 1e-6, "got {}", params[0]);
    }

    #[test]
    fn adamw_rejects_length_mismatch_and_bad_config() {
        let mut optimizer = AdamW::new(
            AdamWConfig {
                lr: 0.1,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
            2,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let mut params = [1.0f32];
        assert!(matches!(
            optimizer.step(&mut params, &[1.0]),
            Err(OptimError::LengthMismatch {
                expected: 2,
                found: 1
            })
        ));
        assert!(matches!(
            AdamW::new(
                AdamWConfig {
                    lr: 0.0,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.0,
                },
                1
            ),
            Err(OptimError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn signtrust_trust_ratio_is_clamped_to_the_band() {
        let mut optimizer = SignTrust::new(
            SignTrustConfig {
                lr: 0.01,
                beta1: 0.9,
                weight_decay: 0.0,
                clamp_min: 0.25,
                clamp_max: 4.0,
            },
            &[2],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let mut params = [2.0f32, -2.0];
        optimizer
            .step_layer(0, &mut params, &[1.0, -1.0])
            .unwrap_or_else(|e| panic!("{e}"));
        assert!((params[0] - 1.96).abs() < 1e-6, "got {}", params[0]);
        assert!((params[1] + 1.96).abs() < 1e-6, "got {}", params[1]);
    }

    #[test]
    fn signtrust_zero_momentum_leaves_params_unchanged() {
        let mut optimizer =
            SignTrust::new(SignTrustConfig::default(), &[2]).unwrap_or_else(|e| panic!("{e}"));
        let mut params = [1.5f32, -0.5];
        optimizer
            .step_layer(0, &mut params, &[0.0, 0.0])
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(params, [1.5, -0.5]);
    }

    #[test]
    fn signtrust_clamps_small_trust_ratio_up() {
        let mut optimizer = SignTrust::new(
            SignTrustConfig {
                lr: 0.01,
                beta1: 0.9,
                weight_decay: 0.0,
                clamp_min: 0.25,
                clamp_max: 4.0,
            },
            &[1],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let mut params = [0.0001f32];
        optimizer
            .step_layer(0, &mut params, &[1.0])
            .unwrap_or_else(|e| panic!("{e}"));
        let expected = 0.0001f32 - 0.01 * 0.25;
        assert!((params[0] - expected).abs() < 1e-9, "got {}", params[0]);
    }

    #[test]
    fn adamw_state_round_trip_preserves_update_trajectory() {
        let config = AdamWConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.1,
        };
        let mut a = AdamW::new(config.clone(), 3).unwrap_or_else(|e| panic!("{e}"));
        let mut b = AdamW::new(config.clone(), 3).unwrap_or_else(|e| panic!("{e}"));
        let mut params_a = [1.0f32, -2.0, 0.5];
        let mut params_b = params_a;
        a.step(&mut params_a, &[0.3, -0.1, 0.9])
            .unwrap_or_else(|e| panic!("{e}"));

        b.step(&mut params_b, &[0.3, -0.1, 0.9])
            .unwrap_or_else(|e| panic!("{e}"));
        let restored = b.state();
        let mut c = AdamW::new(
            AdamWConfig {
                lr: 0.5,
                ..config.clone()
            },
            3,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        c.restore(restored).unwrap_or_else(|e| panic!("{e}"));
        c.set_learning_rate(0.1).unwrap_or_else(|e| panic!("{e}"));

        let tail = [-0.2f32, 0.4, 0.1];
        a.step(&mut params_a, &tail)
            .unwrap_or_else(|e| panic!("{e}"));
        c.step(&mut params_b, &tail)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(params_a, params_b);
        assert_eq!(a.state(), c.state());
    }

    #[test]
    fn signtrust_state_round_trip_preserves_update_trajectory() {
        let config = SignTrustConfig {
            lr: 0.01,
            beta1: 0.9,
            weight_decay: 0.0,
            clamp_min: 0.25,
            clamp_max: 4.0,
        };
        let mut a = SignTrust::new(config.clone(), &[2, 1]).unwrap_or_else(|e| panic!("{e}"));
        let mut params_a = [1.0f32, -2.0];
        a.step_layer(0, &mut params_a, &[0.5, -0.5])
            .unwrap_or_else(|e| panic!("{e}"));

        let restored = a.state();
        let mut b = SignTrust::new(
            SignTrustConfig {
                lr: 0.9,
                ..config.clone()
            },
            &[2, 1],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        b.restore(restored).unwrap_or_else(|e| panic!("{e}"));
        b.set_learning_rate(0.01).unwrap_or_else(|e| panic!("{e}"));

        let mut params_b = params_a;
        a.step_layer(0, &mut params_a, &[-0.2, 0.1])
            .unwrap_or_else(|e| panic!("{e}"));
        b.step_layer(0, &mut params_b, &[-0.2, 0.1])
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(params_a, params_b);
        assert_eq!(a.state(), b.state());
    }

    #[test]
    fn signtrust_rejects_unknown_layer_and_bad_config() {
        let mut optimizer =
            SignTrust::new(SignTrustConfig::default(), &[1]).unwrap_or_else(|e| panic!("{e}"));
        let mut params = [1.0f32];
        assert!(matches!(
            optimizer.step_layer(3, &mut params, &[1.0]),
            Err(OptimError::UnknownLayer {
                layers: 1,
                layer: 3
            })
        ));
        assert!(matches!(
            SignTrust::new(
                SignTrustConfig {
                    lr: 0.01,
                    beta1: 0.9,
                    weight_decay: 0.0,
                    clamp_min: 4.0,
                    clamp_max: 0.25,
                },
                &[1]
            ),
            Err(OptimError::InvalidConfig { .. })
        ));
    }
}
