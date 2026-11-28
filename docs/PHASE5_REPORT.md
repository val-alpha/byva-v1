# Phase 5 Report — Rust Inference Runtime

Date: 2026-11-28. Branch: `vibe/phase5-runtime-15e4a3` (base `byva/implementation`).

## Scope implemented

- `byva_runtime::weights`: `byva-weights-v1` container loader (D-0023) —
  little-endian, magic/versioned header embedding the validated `ModelConfig`,
  canonical tensor order matching the PyTorch state dict, checked parsing
  throughout, typed errors for every malformed-input class (bad magic, bad
  version, bad tie flag, truncation, wrong tensor count/order/size,
  non-finite values). No unsafe code; no panics on malformed input.
- `byva_runtime::model`: scalar fp32 engine for the D-0013 decoder-only
  architecture — pre-norm blocks, causal multi-head attention (no biases),
  exact-erf GELU MLP (no biases), learned absolute positions, tied
  token-embedding/output weights by default. `Session::step` advances one
  token against a preallocated KV cache with no per-token heap allocation;
  context is bounded by `max_seq_len` (typed overflow error).
- `byva_runtime::lcg`: the shared SplitMix64 weight-synthesis contract
  (D-0023) used by the parity fixture pair.
- `byva-cli generate --weights <file> --tokenizer <artifact> --prompt <text>
  [--max-new N]`: end-to-end greedy generation through the production
  tokenizer artifact, stopping at `<|end|>` or `max_seq_len`.
- `training/make_runtime_fixture.py`: PyTorch reference fixture generator
  (bit-exact SplitMix64 mirror; reference logits recorded as raw f32 bit
  patterns).
- `crates/runtime/benches/decode_bench.rs`: Criterion decode benchmark
  (D-0010 style; not run in CI).

## Gate status

| Phase 5 gate item | Status |
|---|---|
| Functional tests | PASSED — loader error taxonomy, session determinism, token/context limits, argmax tie behavior |
| Numerical comparison to training reference | PASSED — committed fixture `runtime_golden_v1.json` (PyTorch fp32 reference, full 32,768 logits): max abs diff **6.985e-10** at tolerance 1e-3; greedy argmax identical (2026-11-28, torch 2.15.0+cpu) |
| Backend validation | PASSED for the Phase 5 scope — portable scalar fp32 backend exercised by all tests on x86_64; SIMD and int8/int4 backends are Phase 7 work behind the same interface (D-0023) and are explicitly not validated yet |

Gate note: the Phase 3 T4/SMALL hardware gate remains deferred by recorded
user decision (Colab skipped; local CPU TINY runs stand in, see
`PHASE3_REPORT.md`). No Phase 5 result above depends on that deferral; the
parity fixture uses the production vocabulary size and the canonical
architecture only.

## Measured performance

See `docs/PERFORMANCE.md` ("Rust runtime decode, fp32 scalar"): TINY scalar
fp32 single-thread greedy decode **175 tok/s median** (5.70 ms/token), 6.6x
above the PyTorch proxy. This is a recorded Phase 5 measurement, not a
D-0019 budget claim — the formal claim waits for the Phase 7 harness.

## Reproduce

~~~bash
# Reference fixture (requires torch; regenerates the committed JSON)
python3 training/make_runtime_fixture.py

# Full verification
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo doc --workspace --no-deps

# Decode benchmark
cargo bench -p byva-runtime --bench decode_bench

# End-to-end smoke (synthetic weights; garbage text is expected)
cargo run -p byva-cli --release -- generate \
  --weights <byva-weights-v1 file> \
  --tokenizer artifacts/tokenizer/byva-tokenizer-v1.json \
  --prompt "Halo" --max-new 16
~~~

## Still open

- Phase 4 converter writing `byva-weights-v1` from `byva-train-checkpoint-v2`
  checkpoints (the CLI smoke above used synthetic weights pending that
  converter).
- D-0019 budget claim (int8 default, SIMD backend, Phase 7 harness).
- SMALL/GPU training run (deferred; recorded decision).
