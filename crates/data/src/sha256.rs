//! Minimal pure-Rust SHA-256 (FIPS 180-4) for shard integrity checks (D-0012).
//!
//! Hand-rolled instead of pulling a hashing dependency so the training path
//! keeps the same zero-native-dependency discipline as `crates/convert`.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 incremental hasher.
#[derive(Debug, Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    length: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Creates a hasher with the FIPS initial state.
    pub fn new() -> Self {
        Self {
            state: H0,
            buffer: [0; 64],
            buffered: 0,
            length: 0,
        }
    }

    /// Absorbs a chunk of bytes.
    pub fn update(&mut self, bytes: &[u8]) {
        self.length = self.length.wrapping_add(bytes.len() as u64);
        let mut offset = 0;
        if self.buffered > 0 {
            let take = usize::min(64 - self.buffered, bytes.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&bytes[..take]);
            self.buffered += take;
            offset = take;
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while offset + 64 <= bytes.len() {
            let mut block = [0; 64];
            block.copy_from_slice(&bytes[offset..offset + 64]);
            self.compress(&block);
            offset += 64;
        }
        if offset < bytes.len() {
            let rest = bytes.len() - offset;
            self.buffer[..rest].copy_from_slice(&bytes[offset..]);
            self.buffered = rest;
        }
    }

    /// Pads, finishes, and returns the 32-byte digest.
    pub fn finalize(mut self) -> [u8; 32] {
        let bits = self.length.wrapping_mul(8);
        self.update(&[0x80]);
        while self.buffered != 56 {
            self.update(&[0]);
        }
        self.length = 0;
        self.update(&bits.to_be_bytes());
        let mut digest = [0; 32];
        for (chunk, word) in digest.chunks_mut(4).zip(self.state.iter()) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    /// Finishes and returns the lowercase hex digest, as recorded in manifests.
    pub fn finalize_hex(self) -> String {
        let digest = self.finalize();
        let mut hex = String::with_capacity(64);
        for byte in digest {
            hex.push(HEX[(byte >> 4) as usize] as char);
            hex.push(HEX[(byte & 0x0f) as usize] as char);
        }
        hex
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (word, chunk) in w.iter_mut().zip(block.chunks_exact(4)) {
            *word = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        let carried = [a, b, c, d, e, f, g, h];
        for (slot, word) in self.state.iter_mut().zip(carried) {
            *slot = slot.wrapping_add(word);
        }
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// One-shot SHA-256 of a byte slice, hex-encoded lowercase.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize_hex()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_hex(input: &str, expected: &str) {
        assert_eq!(sha256_hex(input.as_bytes()), expected);
    }

    #[test]
    fn fips_180_vectors() {
        assert_hex(
            "",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        );
        assert_hex(
            "abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
        assert_hex(
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        );
    }

    #[test]
    fn one_million_a_matches_python_hashlib() {
        let expected = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
        let chunk = [b'a'; 1_000];
        let mut hasher = Sha256::new();
        for _ in 0..1_000 {
            hasher.update(&chunk);
        }
        assert_eq!(hasher.finalize_hex(), expected);
    }

    #[test]
    fn incremental_equals_oneshot() {
        let payload = b"shard-00000.jsonl integrity check payload, split across updates";
        let mut hasher = Sha256::new();
        hasher.update(&payload[..7]);
        hasher.update(&payload[7..40]);
        hasher.update(&payload[40..]);
        assert_eq!(hasher.finalize_hex(), sha256_hex(payload));
    }

    #[test]
    fn hex_matches_manifest_style() {
        assert_eq!(sha256_hex(b"x").len(), 64);
        assert!(sha256_hex(b"x").chars().all(|c| c.is_ascii_hexdigit()));
    }
}
