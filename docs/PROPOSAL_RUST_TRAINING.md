# Proposal: Pure-Rust Training Method (no Python)

- Status: **Proposed** — awaiting acceptance; would be recorded as **D-0022** in `docs/DECISIONS.md` if accepted.
- Date: 2026-11-28
- Scope: a separate, self-owned pretraining method implemented entirely in Rust. This document proposes; it records no results. Nothing here has been implemented, run, or measured yet.
- Relationship to existing decisions: extends D-0001 (from-scratch), D-0002 (Rust-native inference boundary), D-0003 (workspace boundaries), D-0005 (backend abstraction), D-0006 (unsafe policy), D-0007/D-0012 (artifact hygiene, deterministic shards), D-0013 (frozen architectures), D-0014/D-0018 (resumable checkpoints), D-0016 (bilingual scope), D-0019 (CPU-first budgets), D-0008 (phase gates and truthfulness).

## Summary

Today the only model-training path is `training/train.py` under PyTorch; CLAUDE.md permits PyTorch for training-time tensor operations and autograd. Tokenizer training is already Rust-canonical (D-0020/D-0021). This proposal completes the picture: a BYVA-owned pretraining method — tensor storage, autodiff, kernels, data pipeline, optimizer, checkpointing — implemented in pure Rust with **zero Python and zero PyTorch/libtorch involvement**, CPU-only by default, with GPU enabled only when a backend is compiled in and actually detected on the machine.

If accepted, an end-to-end BYVA run (shard build → BPE 32k → model pretraining → native checkpoint) requires only `cargo`.

## Motivation

- The deployment target is CPU-only machines (D-0019). Training and inference would share one toolchain, one buffer layout, and one kernel abstraction, so CPU speed work (SIMD, threading, allocation-free loops) serves both sides instead of being duplicated across Python/Rust.
- PyTorch CPU training is heavyweight to provision (multi-GB toolchain, venv/pip) and recent sessions showed local CPU training is the operative route after Colab was deferred; a cargo-only trainer removes the entire Python environment from the critical path.
- Owning the numeric code gives deterministic, inspectable training: fixed-seed RNG streams, ordered reductions, and exact-resume semantics (the spirit of D-0014/D-0018) without depending on framework RNG internals.
- The tokenizer side already proved the pattern: a Rust rewrite with a byte-identity parity gate against a Python reference (D-0021) is practical, testable, and outperforms the Python path on memory.

## Non-goals

- No pretrained weights, no third-party checkpoints, no external model runtime (D-0001, D-0002, CLAUDE.md).
- No claim of GPU support until a real-hardware run exists (D-0005/D-0010 rule); a backend that is not compiled in and validated is never described as supported.
- No change to the frozen architectures (D-0013), the shard pipeline semantics (D-0012), the tokenizer artifact, or the serving-side tokenizer.
- No removal of the Python reference trainer in the near term: it stays as the readable parity reference, exactly as D-0021 keeps it for BPE.

## Proposed decision (draft D-0022)

> **D-0022 — Pure-Rust training path.** Model pretraining gains a second, BYVA-owned implementation in pure Rust: a minimal tensor/autodiff engine and a training loop authored in this repository, with no Python and no PyTorch/libtorch dependency on the training path. Execution is CPU-only by default; a GPU backend (wgpu and/or cudarc, feature-gated) may be selected at runtime only when it is compiled in, initializes successfully on the machine, and is explicitly requested or auto-detected; otherwise training runs on CPU. The PyTorch trainer (`training/train.py`) remains in the repository as the numerical reference until the Rust trainer passes a recorded parity gate on a TINY smoke run (same shards, same tokenizer artifact, comparable loss curve); after that its role is reference-only. All training claims for the Rust path remain subject to D-0008 truthfulness rules.

- Rationale: single-toolchain reproducibility for a CPU-first product; training/inference kernel sharing; deterministic control of RNG and reduction order.
- Consequences: new crates extend D-0003's six-crate boundary (the list becomes "plus trainer-path crates, added by this decision"); a new Rust-native training checkpoint format is introduced (it does not read PyTorch checkpoints); parity against the Python trainer is a mandatory gate before the Rust path is used for any recorded pretraining run; GPU support stays provisional per D-0005 until hardware validation.

## Method design

### Crate layout (extends D-0003; names are a proposal)

| Crate | Responsibility |
|---|---|
| `crates/tensor` | Owned dense buffers, shape/stride with checked arithmetic, fp32 dtype (bf16/fp16 later, GPU only), typed errors via `thiserror`, no `unwrap` in library code |
| `crates/autodiff` | Reverse-mode tape with only the ops D-0013 needs: embedding lookup (scatter-add grad), causal MHA (softmax, scaled dot-product), GELU MLP, pre-norm layernorm, tied output projection, fused softmax cross-entropy, AdamW update |
| `crates/data` | Shard reader + manifest validation (sha256 per shard, D-0012), streaming document iteration, fixed packing into seq-length windows, per-run id:en mixture read from the run manifest (D-0016) |
| `crates/trainer` | Training loop, LR schedule, gradient accumulation, checkpoint save/load, eval, CLI subcommand in `crates/cli` (`byva-train`) |

Workspace dependencies stay minimal: `serde`/`serde_json` (manifests, checkpoint metadata), `thiserror`, `rand` + `rand_chacha` (deterministic streams). Threading via a small hand-rolled pool or `rayon` behind a feature. GPU backends (`wgpu`, `cudarc`) are optional features, off by default; neither pulls in libtorch or any C++ model runtime.

### Device strategy: CPU-first, GPU optional-if-detected

- Default and only guaranteed path: **CPU**. `--device cpu` is always valid; `--device auto` (default) promotes to GPU only if a GPU backend feature is compiled in and a device initializes successfully, and prints which backend was selected; otherwise it stays on CPU with a visible notice.
- CPU kernels follow D-0006/D-0019: scalar safe reference implementation first, SIMD (AVX2/NEON via `std::arch`) behind the backend abstraction with `// SAFETY:`-documented unsafe only at the narrow hardware boundary, safe fallback always available; preallocated per-step workspace (no allocation in the inner loop); multithreading over batch/head/matmul tiles.
- GPU (when a backend is later validated): fp32 or bf16 per device capability, its own reduction-order document, results always labeled by backend in reports. GPU numbers are recorded only from measured runs (D-0010 rule, applied to training).

### Training method (BYVA-owned)

- Optimizer: AdamW (decoupled weight decay), fp32 master weights on CPU; global-norm gradient clipping.
- Schedule: linear warmup then cosine decay to a recorded floor; hyperparameters live in a versioned run config, not hardcoded.
- Gradient accumulation with mean-reduced loss history so numbers stay comparable across accumulation factors (D-0018 semantics).
- Determinism: one run seed derives all RNG streams (weight init, any shuffling) via ChaCha; document order comes from the deterministic shard order (D-0012), not filesystem order.
- Checkpoint: new format **`byva-train-rust-checkpoint-v1`** — step, config, weights, AdamW moments, per-stream RNG counters, dataloader position, loss history, sha256 integrity; rejected with a typed error on unknown formats (D-0014 discipline). Checkpoints stay out of Git (D-0007). Resume must reproduce the uninterrupted run's loss history exactly, enforced by a test.
- Eval: held-out loss at fixed step intervals on the Phase 3 bilingual held-out split; results labeled by encoding (BPE 32k artifact) and device.

### Parity and gates

1. **Gradcheck**: every autodiff op validated against finite differences in property tests before any training run.
2. **Parity gate (mandatory before recorded pretraining)**: TINY (D-0013), fixed step budget, same shard set and BPE 32k artifact, Rust trainer vs the PyTorch reference. Acceptance: monotone falling loss on both sides within a tolerance fixed in D-0022's first revision, plus greedy generations from both final checkpoints compared on a fixed prompt set. Until this passes, the Rust trainer is reported as *implemented, not validated* (D-0008).
3. **Resume equivalence**: Rust-trainer resume reproduces the uninterrupted loss history byte-for-byte (test-enforced, mirrors D-0014).
4. Standard suite per CLAUDE.md: `cargo fmt --check`, `cargo check`, `clippy -D warnings`, `cargo test`, fuzz targets for shard/checkpoint parsing.

### Milestones (phase-scoped, per D-0008)

| Step | Content | Gate |
|---|---|---|
| R0 | D-0022 accepted; crates scaffolded behind no behavior change | workspace checks green |
| R1 | Tensor + autodiff engine | gradcheck property tests |
| R2 | `crates/data` shard reader/packer | golden manifest + fixture tests (D-0012) |
| R3 | Trainer loop, AdamW, checkpoint/resume, `byva-train` CLI | resume-equivalence test; TINY smoke runs |
| R4 | Parity vs PyTorch reference (TINY) | parity gate recorded in a phase report + PERFORMANCE.md entry (env, toolchain, date) |
| R5 | CPU training throughput baseline vs the PyTorch baseline; first budget revision of D-0022 | measured numbers only |
| R6 | Optional GPU backend (wgpu or cudarc) | only if hardware is available; otherwise recorded **not run** |

This path serves the Phase 3 pretraining gate; note the Phase 3 gate text says "Real T4 training run". If the accepted route becomes local CPU training, that wording needs a documented, approved gate exception (D-0008) rather than a silent reinterpretation.

## Risks and mitigations

- **Autodiff correctness**: the highest risk. Mitigated by finite-difference gradchecks, small hand-computed cases, and the mandatory PyTorch parity gate before any recorded run.
- **CPU throughput gap vs PyTorch**: PyTorch's CPU kernels are mature. Mitigation: SIMD + threading + allocation-free loops shared with the runtime budget (D-0019); the acceptance bar for R5 is relative to a measured PyTorch baseline on the same machine, not an invented absolute number. If the gap cannot be closed, the decision is revisited and recorded — never silently dropped.
- **GPU scope creep**: backends stay feature-gated and behind D-0005's validation rule; no GPU claim without hardware runs.
- **Two trainers drift (Rust vs Python reference)**: bounded by keeping the Python side reference-only and by the same discipline that worked for BPE (D-0020/D-0021 golden-style equivalence).

## Open questions (for the acceptance discussion)

1. Accept D-0022 as drafted, or with the PyTorch trainer retired earlier/later?
2. Crate names/numbering: confirm `tensor` / `autodiff` / `data` / `trainer` as new workspace members.
3. Should the CPU training-throughput budget live in D-0022's first revision or extend D-0019?
4. Tolerance and step budget for the parity gate, to be fixed in D-0022 after R3 produces a first measured comparison.
