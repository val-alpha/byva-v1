# BYVA

BYVA is an experimental language model project built from random initialization, with a native Rust inference runtime and a versioned conversation/token format.

> Project status: Phase 3 implementation complete (decoder-only architecture + resumable reference trainer, gate in docs/PHASE3_REPORT.md). The Colab T4 training run is the remaining Phase 3/8 hardware task. No model has been pretrained at production scale yet and no inference backend has been validated.

## Project constraints

- Pretraining starts from randomly initialized parameters. No third-party model checkpoints or weights are used.
- The model architecture, tokenizer, BYVA conversation format, checkpoint container, and Rust inference runtime are developed in this repository.
- Inference is implemented in Rust. PyTorch is permitted only as a thin training-time tensor/autograd tool.
- Public datasets are admitted only after source, license, and suitability review.
- Hardware acceleration is capability-detected; CPU remains the correctness baseline and fallback.
- Decisions and deviations are recorded in docs/DECISIONS.md.

## Workspace crates

| Crate | Responsibility |
|---|---|
| byva-tokenizer | Deterministic byte-level BPE tokenizer |
| byva-format | BYVA message renderer, streaming parser, and tool-call conversion |
| byva-model | Shared model configuration and architecture definitions |
| byva-runtime | Native Rust inference, sampling, KV cache, and compute backends |
| byva-convert | Training-checkpoint to native .byva conversion |
| byva-cli | User-facing command-line interface |

Training utilities will live under training/ and are not part of the Rust inference runtime.

## Development

Install stable Rust with rustfmt and clippy components. The toolchain channel is recorded in rust-toolchain.toml.

~~~bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo doc --workspace --no-deps
~~~

## Hardware and training

The initial training target is a Colab NVIDIA T4 (16 GB) using FP16 and gradient accumulation. CUDA, Vulkan, Metal, and CPU support will be added and validated in later phases. A backend is not considered supported merely because its feature flag or dependency exists.

## Phase tracking

Implementation proceeds through gated phases. Completion reports distinguish code added, tests actually run, and work that remains. See docs/DECISIONS.md for initial engineering decisions.
