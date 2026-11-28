//! Deterministic weight synthesis for cross-language parity fixtures.
//!
//! The Phase 5 numerical-parity fixture is generated from a shared
//! deterministic generator (D-0023) instead of committing a binary weight
//! blob: `training/make_runtime_fixture.py` builds the same weights, runs the
//! PyTorch reference forward pass, and records the expected logits. Both sides
//! must keep the sequence and conversions byte-identical.

use byva_model::ModelConfig;

use crate::weights::Weights;

/// Per-step additive constant of the generator (the golden ratio constant).
pub const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// Fixed seed for the Phase 5 runtime parity fixture (D-0023).
pub const FIXTURE_SEED: u64 = 0x2026_1128_5EED_0001;
/// Scale applied to uniform values so synthetic weights match the
/// trainer's normal(0, 0.02) magnitude range.
pub const FIXTURE_WEIGHT_SCALE: f32 = 0.02;

/// SplitMix64 generator with a fixed, documented output contract.
///
/// Output contract mirrored by `training/make_runtime_fixture.py`:
/// `state = state + GAMMA (mod 2^64)`; then the SplitMix64 finalizer
/// (xor 30, mul, xor 27, mul, xor 31) produces the 64-bit sample.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator seeded with `seed`.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Produces the next 64-bit sample.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Produces a uniform f32 in `[-1, 1)` from the top 24 bits of the
    /// 64-bit sample; the f32 arithmetic is exactly reproducible in Python.
    pub fn next_uniform_f32(&mut self) -> f32 {
        let sample = self.next_u64();
        let top24 = ((sample >> 40) & 0xFF_FFFF) as f32;
        (top24 / 16_777_216.0) * 2.0 - 1.0
    }

    /// Fills `out` with `scale * uniform` values in sequence order.
    pub fn fill_scaled(&mut self, out: &mut [f32], scale: f32) {
        for value in out.iter_mut() {
            *value = self.next_uniform_f32() * scale;
        }
    }
}

/// Builds validated synthetic weights for `config` from [`FIXTURE_SEED`],
/// filling the canonical `byva-weights-v1` tensor order with
/// `FIXTURE_WEIGHT_SCALE`-scaled uniform values in canonical tensor order.
/// Mirrored by `training/make_runtime_fixture.py`.
#[must_use]
pub fn fixture_weights(config: &ModelConfig) -> Weights {
    let d = u64::from(config.d_model);
    let ffn = u64::from(config.ffn_hidden);
    let vocab = u64::from(config.vocab_size);
    let seq = u64::from(config.max_seq_len);
    let mut generator = SplitMix64::new(FIXTURE_SEED);
    let mut fill = |name: &str, elements: u64| {
        let mut values = vec![0.0f32; elements as usize];
        generator.fill_scaled(&mut values, FIXTURE_WEIGHT_SCALE);
        (name.to_string(), values)
    };

    let mut tensors = vec![
        fill("token_embedding.weight", vocab * d),
        fill("position_embedding.weight", seq * d),
    ];
    for layer in 0..u64::from(config.n_layers) {
        let prefix = format!("blocks.{layer}.");
        for (suffix, elements) in [
            ("norm_attn.weight", d),
            ("norm_attn.bias", d),
            ("attention.qkv.weight", 3 * d * d),
            ("attention.out.weight", d * d),
            ("norm_mlp.weight", d),
            ("norm_mlp.bias", d),
            ("mlp.0.weight", d * ffn),
            ("mlp.2.weight", ffn * d),
        ] {
            tensors.push(fill(&format!("{prefix}{suffix}"), elements));
        }
    }
    tensors.push(fill("final_norm.weight", d));
    tensors.push(fill("final_norm.bias", d));
    if !config.tie_embeddings {
        tensors.push(fill("lm_head.weight", vocab * d));
    }

    let mut file = Vec::new();
    file.extend_from_slice(&crate::weights::WEIGHTS_MAGIC);
    file.push(crate::weights::WEIGHTS_FORMAT_VERSION);
    file.extend_from_slice(&config.vocab_size.to_le_bytes());
    file.extend_from_slice(&config.d_model.to_le_bytes());
    file.extend_from_slice(&config.n_layers.to_le_bytes());
    file.extend_from_slice(&config.n_heads.to_le_bytes());
    file.extend_from_slice(&config.ffn_hidden.to_le_bytes());
    file.extend_from_slice(&config.max_seq_len.to_le_bytes());
    file.push(u8::from(config.tie_embeddings));
    file.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for (name, values) in &tensors {
        file.extend_from_slice(&(name.len() as u32).to_le_bytes());
        file.extend_from_slice(name.as_bytes());
        file.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            file.extend_from_slice(&value.to_le_bytes());
        }
    }
    // The synthetic layout is canonical by construction; a parse failure is a
    // programming error, so failing loudly here is correct even though the
    // library avoids panics on external input.
    Weights::from_bytes(&file).expect("fixture weights are canonical by construction")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_is_deterministic() {
        let mut generator = SplitMix64::new(FIXTURE_SEED);
        let first: [u64; 4] = [
            generator.next_u64(),
            generator.next_u64(),
            generator.next_u64(),
            generator.next_u64(),
        ];
        let mut again = SplitMix64::new(FIXTURE_SEED);
        for expected in first {
            assert_eq!(again.next_u64(), expected);
        }
    }

    #[test]
    fn uniform_stays_in_range() {
        let mut generator = SplitMix64::new(7);
        for _ in 0..10_000 {
            let value = generator.next_uniform_f32();
            assert!((-1.0..1.0).contains(&value));
        }
    }
}
