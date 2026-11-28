//! Deterministic RNG stream for weight init (one run seed derives
//! everything, per the D-0022 determinism contract).
//!
//! SplitMix64 for uniform words, Box-Muller for normals. Hand-rolled so the
//! training path keeps its zero-dependency discipline; the stream is tested
//! for determinism and cross-seed divergence.

/// SplitMix64 generator (test vectors: Steele, Lea & Flood).
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seeds the generator.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Produces the next 64-bit word.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 53 bits of mantissa.
    pub fn next_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller (two uniforms per value).
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        let u1 = if u1 <= 0.0 { f64::MIN_POSITIVE } else { u1 };
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_outputs_match_published_vectors() {
        let mut generator = SplitMix64::new(0);
        assert_eq!(generator.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(generator.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(generator.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn same_seed_reproduces_the_stream() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn uniforms_are_in_range_and_normals_finite() {
        let mut generator = SplitMix64::new(7);
        for _ in 0..1_000 {
            let u = generator.next_uniform();
            assert!((0.0..1.0).contains(&u));
            let n = generator.next_normal();
            assert!(n.is_finite());
            assert!(n.abs() < 20.0, "normal draw out of sane range: {n}");
        }
    }
}
