# Proposal: Pure-Rust Training Method (no Python)

- Status: **Proposed** — awaiting acceptance; would be recorded as **D-0022** in `docs/DECISIONS.md` if accepted.
- Date: 2026-11-28
- Scope: a separate, self-owned pretraining method implemented entirely in Rust. This document proposes; it records no results. Nothing here has been implemented, run, or measured yet. The method design ("Kilat", below) is BYVA-specific: a compiled step program, fused kernels that remove whole memory passes, a from-first-principles optimizer (SignTrust), and constant-FLOPs batching — with every speed/quality target framed as a hypothesis and a measured acceptance bar, never a claim (D-0008).
- Relationship to existing decisions: extends D-0001 (from-scratch), D-0002 (Rust-native inference boundary), D-0003 (workspace boundaries), D-0005 (backend abstraction), D-0006 (unsafe policy), D-0007/D-0012 (artifact hygiene, deterministic shards), D-0013 (frozen architectures), D-0014/D-0018 (resumable checkpoints), D-0016 (bilingual scope), D-0019 (CPU-first budgets), D-0008 (phase gates and truthfulness).

## Summary

Today the only model-training path is `training/train.py` under PyTorch; CLAUDE.md permits PyTorch for training-time tensor operations and autograd. Tokenizer training is already Rust-canonical (D-0020/D-0021). This proposal completes the picture: a BYVA-owned pretraining method — tensor storage, autodiff, kernels, data pipeline, optimizer, checkpointing — implemented in pure Rust with **zero Python and zero PyTorch/libtorch involvement**, CPU-only by default, with GPU enabled only when a backend is compiled in and actually detected on the machine.

If accepted, an end-to-end BYVA run (shard build → BPE 32k → model pretraining → native checkpoint) requires only `cargo` — and the training method itself ("Kilat", below) is designed to beat the common path on CPU wall-clock, with every target framed as a measured acceptance bar, not a claim.

## Motivation

- The bar is explicit: **faster than the common path**, not parity with it. The common path is the PyTorch trainer already measured locally (TINY, ~1.1 s/step at seq 128, batch 8) and the PyTorch decode proxy (26.6 tok/s); the method's acceptance bars are defined against those measured numbers on the same machine, not against invented absolutes.
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

> **D-0022 — Pure-Rust training path.** Model pretraining gains a second, BYVA-owned implementation in pure Rust: a minimal tensor/autodiff engine and a training loop authored in this repository, with no Python and no PyTorch/libtorch dependency on the training path. Execution is CPU-only by default; a GPU backend (wgpu and/or cudarc, feature-gated) may be selected at runtime only when it is compiled in, initializes successfully on the machine, and is explicitly requested or auto-detected; otherwise training runs on CPU. The PyTorch trainer (`training/train.py`) remains in the repository as the numerical reference until the Rust trainer passes a recorded parity gate on a TINY smoke run (same shards, same tokenizer artifact, comparable loss curve); after that its role is reference-only. The training method itself is BYVA-owned ("Kilat", below): a compiled static step program (K1), fused kernels including tied-logit-free cross-entropy (K2), a SignTrust optimizer that may become default only after a recorded gate vs AdamW (K3/K6), and constant-FLOPs token-budget batching (K4); each component's speed/quality target is an acceptance bar measured against the recorded PyTorch baseline, never a claim. All training claims for the Rust path remain subject to D-0008 truthfulness rules.

- Rationale: single-toolchain reproducibility for a CPU-first product; training/inference kernel sharing; deterministic control of RNG and reduction order.
- Consequences: new crates extend D-0003's six-crate boundary (the list becomes "plus trainer-path crates, added by this decision"); a new Rust-native training checkpoint format is introduced (it does not read PyTorch checkpoints); parity against the Python trainer is a mandatory gate before the Rust path is used for any recorded pretraining run; GPU support stays provisional per D-0005 until hardware validation.

## The method: "Kilat" — BYVA's own design, not a framework re-run

The design goal is not to re-implement PyTorch's step loop in Rust. A from-scratch project has two freedoms the common path does not: the architecture is frozen (D-0013), so a training step can be **compiled, not interpreted**; and the optimizer/data policy can be chosen for **CPU wall-clock**, not GPU throughput. Everything below is a hypothesis paired with a measured acceptance bar — no speed or quality claim exists until the R-gates record numbers (D-0008).

### K1 — Compiled step program (static execution plan)

- For each frozen config (TINY/SMALL) the training step is one static program: buffer sizes, offsets, and kernel order are derived once at startup from `ModelConfig`, placed into a single preallocated arena, and never reallocated. Zero dynamic dispatch in the hot loop; kernels are specialized per config at compile time (const generics over `d_model`/heads), removing the branch/indirection overhead generic engines pay.
- The plan (arena layout, kernel list, shapes) is printable (`--explain-step`) and pinned by a test, so what ran is auditable per D-0008.

### K2 — Fused kernels that remove whole memory passes

- **Tied-logit-free cross-entropy**: with tied weights (D-0013), loss and its gradient are computed in one fused pass of hidden states against the embedding table — the full `[batch·seq, vocab]` logits matrix is never materialized. For TINY (seq 512, vocab 32,768) this removes the single largest activation and its write+read traffic.
- Fused pre-norm layernorm + residual; causal mask folded into the softmax pass; GELU fused into the MLP matmul epilogue.
- Every kernel has a scalar safe reference (D-0006) plus an AVX2/NEON path behind the backend abstraction — and the same kernels serve the Phase 5 runtime, so every K2 win also moves the D-0019 decode budgets.

### K3 — SignTrust: the BYVA optimizer

Common optimizers pay either in step-scale stability (SGD-momentum) or in per-parameter state (AdamW: two moment buffers + a sqrt per parameter per step). SignTrust replaces the second moment with a per-layer trust ratio:

- State: one EMA momentum buffer per parameter — half of AdamW's optimizer state.
- Update: `w -= lr · sign(m) · c_l`, with per-layer trust `c_l = rms(w_l) / rms(m_l)` clamped to a fixed band — cheap reductions over layers, not per-parameter transcendental math. The trust ratio plays the role Adam's second moment plays (scale-awareness), paid once per layer instead of once per parameter; on CPU the optimizer cost becomes nearly bandwidth-free.
- Decoupled weight decay; warmup/cosine schedule unchanged. Update rules are fully deterministic (no RNG).
- SignTrust is a **candidate, not a default**: AdamW remains the trainer default and the parity baseline until SignTrust passes its gate (equal-or-better held-out loss at equal wall-clock, threshold fixed in D-0022's first revision after a measured comparison).

### K4 — Token-budget batching: constant-FLOPs steps

- Steps are budgeted by **tokens per step** (fixed token budget), not a fixed sequence count: batches are packed to the budget from the deterministic shard stream (document order from the D-0012 manifest). Every step does (approximately) the same FLOPs — step latency is flat and predictable, checkpoint cadence is time-predictable, and there is no padding waste.

### K5 — Inline loss-aware mixture (inside D-0016)

- Hypothesis to validate: reweighting the id:en shard mixture, within the bounds recorded in the run manifest, based on measured per-language held-out loss, improves quality per wall-clock hour at this scale. Outside the recorded bounds the mixer is a no-op; every adjustment is recorded in the run manifest.

### K6 — Targets vs the common path (bars, not claims)

| Comparison | Common-path reference (measured) | Acceptance bar for this method |
|---|---|---|
| Training step throughput (TINY, same machine) | ~1.1 s/step (PyTorch, seq 128, batch 8) | ≥ 2× step throughput before any quantization tricks |
| Convergence vs optimizer baseline | AdamW at equal wall-clock | SignTrust gate: held-out loss equal-or-better at equal wall-clock |
| Decode (proxy baseline) | 26.6 tok/s (PyTorch TINY fp32) | shared kernels must not regress the D-0019 budget; training work counts toward it |
| Latency discipline | step time jitter from allocator + variable batch FLOPs | zero allocation in the step (K1) + constant-FLOPs steps (K4); p99 step time recorded next to means |

## Method design

### Crate layout (extends D-0003; names are a proposal)

| Crate | Responsibility |
|---|---|
| `crates/tensor` | Owned dense buffers, shape/stride with checked arithmetic, fp32 dtype (bf16/fp16 later, GPU only), typed errors via `thiserror`, no `unwrap` in library code |
| `crates/autodiff` | Reverse-mode tape with only the ops D-0013 needs: embedding lookup (scatter-add grad), causal MHA (softmax, scaled dot-product), GELU MLP, pre-norm layernorm, tied output projection, fused softmax cross-entropy (tied-logit-free, K2), AdamW and SignTrust (K3) updates |
| `crates/data` | Shard reader + manifest validation (sha256 per shard, D-0012), streaming document iteration, fixed packing into seq-length windows, per-run id:en mixture read from the run manifest (D-0016) |
| `crates/trainer` | Training loop, LR schedule, gradient accumulation, checkpoint save/load, eval, CLI subcommand in `crates/cli` (`byva-train`) |

Workspace dependencies stay minimal: `serde`/`serde_json` (manifests, checkpoint metadata), `thiserror`, `rand` + `rand_chacha` (deterministic streams). Threading via a small hand-rolled pool or `rayon` behind a feature. GPU backends (`wgpu`, `cudarc`) are optional features, off by default; neither pulls in libtorch or any C++ model runtime.

### Device strategy: CPU-first, GPU optional-if-detected

- Default and only guaranteed path: **CPU**. `--device cpu` is always valid; `--device auto` (default) promotes to GPU only if a GPU backend feature is compiled in and a device initializes successfully, and prints which backend was selected; otherwise it stays on CPU with a visible notice.
- CPU kernels follow D-0006/D-0019: scalar safe reference implementation first, SIMD (AVX2/NEON via `std::arch`) behind the backend abstraction with `// SAFETY:`-documented unsafe only at the narrow hardware boundary, safe fallback always available; preallocated per-step workspace (no allocation in the inner loop); multithreading over batch/head/matmul tiles.
- GPU (when a backend is later validated): fp32 or bf16 per device capability, its own reduction-order document, results always labeled by backend in reports. GPU numbers are recorded only from measured runs (D-0010 rule, applied to training).

### Training method (BYVA-owned)

- Optimizers: **AdamW** (decoupled weight decay, fp32 master weights, global-norm gradient clipping) is the default and the parity baseline; **SignTrust (K3)** ships behind `--optimizer signtrust` and may become the default only after its gate (K6) passes with a recorded comparison.
- Schedule: linear warmup then cosine decay to a recorded floor; hyperparameters live in a versioned run config, not hardcoded.
- Gradient accumulation with mean-reduced loss history so numbers stay comparable across accumulation factors (D-0018 semantics).
- Determinism: one run seed derives all RNG streams (weight init, any shuffling) via ChaCha; document order comes from the deterministic shard order (D-0012), not filesystem order.
- Checkpoint: new format **`byva-train-rust-checkpoint-v1`** — step, config, weights, optimizer state (AdamW moments or SignTrust momentum + trust state), per-stream RNG counters, dataloader position, loss history, sha256 integrity; rejected with a typed error on unknown formats (D-0014 discipline). Checkpoints stay out of Git (D-0007). Resume must reproduce the uninterrupted run's loss history exactly, enforced by a test.
- Eval: held-out loss at fixed step intervals on the Phase 3 bilingual held-out split; results labeled by encoding (BPE 32k artifact) and device.

### Parity and gates

1. **Gradcheck**: every autodiff op validated against finite differences in property tests before any training run.
2. **Parity gate (mandatory before recorded pretraining)**: TINY (D-0013), fixed step budget, same shard set and BPE 32k artifact, Rust trainer vs the PyTorch reference. Acceptance: monotone falling loss on both sides within a tolerance fixed in D-0022's first revision, plus greedy generations from both final checkpoints compared on a fixed prompt set. Until this passes, the Rust trainer is reported as *implemented, not validated* (D-0008).
3. **Resume equivalence**: Rust-trainer resume reproduces the uninterrupted loss history byte-for-byte (test-enforced, mirrors D-0014).
4. **Method gates (K6)**: each Kilat component carries its own measured acceptance bar — throughput bar (≥ 2× the measured PyTorch step baseline on the same machine), SignTrust gate (equal-or-better held-out loss vs AdamW at equal wall-clock), latency record (p99 step time next to means) — all recorded in PERFORMANCE.md with environment, toolchain, and date before being called met.
5. Standard suite per CLAUDE.md: `cargo fmt --check`, `cargo check`, `clippy -D warnings`, `cargo test`, fuzz targets for shard/checkpoint parsing.

### Milestones (phase-scoped, per D-0008)

| Step | Content | Gate |
|---|---|---|
| R0 | D-0022 accepted; crates scaffolded behind no behavior change | workspace checks green |
| R1 | Tensor + autodiff engine | gradcheck property tests |
| R2 | `crates/data` shard reader/packer | golden manifest + fixture tests (D-0012) |
| R3 | Trainer loop, AdamW default, SignTrust behind a flag, token-budget batching, checkpoint/resume, `byva-train` CLI | resume-equivalence test; TINY smoke runs |
| R4 | Parity vs PyTorch reference (TINY); measure and record the PyTorch step baseline on the reference machine | parity gate recorded in a phase report + PERFORMANCE.md entry (env, toolchain, date) |
| R5 | Kilat performance gates: throughput bar (K6), SignTrust gate, p99 latency record; fused kernels + tied-logit-free CE land here | measured numbers only; first budget revision of D-0022 |
| R6 | K5 inline loss-aware mixture (bounded, manifest-recorded) | per-run manifest records every adjustment; quality delta reported |
| R7 | Optional GPU backend (wgpu or cudarc) | only if hardware is available; otherwise recorded **not run** |

This path serves the Phase 3 pretraining gate; note the Phase 3 gate text says "Real T4 training run". If the accepted route becomes local CPU training, that wording needs a documented, approved gate exception (D-0008) rather than a silent reinterpretation.

## Risks and mitigations

- **Autodiff correctness**: the highest risk. Mitigated by finite-difference gradchecks, small hand-computed cases, and the mandatory PyTorch parity gate before any recorded run.
- **CPU throughput gap vs PyTorch**: PyTorch's CPU kernels are mature. Mitigation: the Kilat design bets on removing work rather than optimizing the same work — tied-logit-free CE (K2), constant-FLOPs batching (K4), compiled step (K1); the acceptance bar is relative to a measured PyTorch baseline on the same machine, not an invented absolute. If the bar cannot be met, the decision is revisited and recorded — never silently dropped.
- **Optimizer novelty (SignTrust)**: sign-based updates can underperform AdamW on some landscapes. Contained by making AdamW the default until the SignTrust gate passes at equal wall-clock; a failed gate leaves the code as a flag, not a default.
- **GPU scope creep**: backends stay feature-gated and behind D-0005's validation rule; no GPU claim without hardware runs.
- **Two trainers drift (Rust vs Python reference)**: bounded by keeping the Python side reference-only and by the same discipline that worked for BPE (D-0020/D-0021 golden-style equivalence).

## Open questions (for the acceptance discussion)

1. Accept D-0022 as drafted, or with the PyTorch trainer retired earlier/later?
2. Crate names/numbering: confirm `tensor` / `autodiff` / `data` / `trainer` as new workspace members.
3. Should the CPU training-throughput budget live in D-0022's first revision or extend D-0019?
4. Tolerance and step budget for the parity gate, to be fixed in D-0022 after R3 produces a first measured comparison.
5. SignTrust gate threshold (held-out loss delta vs AdamW at equal wall-clock) and the trust-ratio clamp band — fixed in D-0022's first revision after the first measured comparison.
6. Method name: confirm "Kilat" as the method's name in docs and CLI (`--method kilat`), or rename.
