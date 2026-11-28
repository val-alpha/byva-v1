# Proposal: Pure-Rust Training Method (no Python)

- Status: **Accepted** (2026-11-28) — recorded as **D-0022** in `docs/DECISIONS.md`.
- Date: 2026-11-28
- Scope: a separate, self-owned pretraining method implemented entirely in Rust. This document proposes; it records no results. Nothing here has been implemented, run, or measured yet. The method design ("ne2410", below) is BYVA-specific: a compiled step program, fused kernels that remove whole memory passes, a from-first-principles optimizer (SignTrust), and constant-FLOPs batching — with every speed/quality target framed as a hypothesis and a measured acceptance bar, never a claim (D-0008).
- Relationship to existing decisions: extends D-0001 (from-scratch), D-0002 (Rust-native inference boundary), D-0003 (workspace boundaries), D-0005 (backend abstraction), D-0006 (unsafe policy), D-0007/D-0012 (artifact hygiene, deterministic shards), D-0013 (frozen architectures), D-0014/D-0018 (resumable checkpoints), D-0016 (bilingual scope), D-0019 (CPU-first budgets), D-0008 (phase gates and truthfulness).

## Summary

Today the only model-training path is `training/train.py` under PyTorch; CLAUDE.md permits PyTorch for training-time tensor operations and autograd. Tokenizer training is already Rust-canonical (D-0020/D-0021). This proposal completes the picture: a BYVA-owned pretraining method — tensor storage, autodiff, kernels, data pipeline, optimizer, checkpointing — implemented in pure Rust with **zero Python and zero PyTorch/libtorch involvement**, CPU-only by default, with GPU enabled only when a backend is compiled in and actually detected on the machine.

If accepted, an end-to-end BYVA run (shard build → BPE 32k → model pretraining → native checkpoint) requires only `cargo` — and the training method itself ("ne2410", below) is designed to beat the common path on CPU wall-clock, with every target framed as a measured acceptance bar, not a claim.

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

## Decision (recorded as D-0022 — Accepted)

> **D-0022 — Pure-Rust training path.** Model pretraining gains a second, BYVA-owned implementation in pure Rust: a minimal tensor/autodiff engine and a training loop authored in this repository, with no Python and no PyTorch/libtorch dependency on the training path. Execution is CPU-only by default; a GPU backend (wgpu and/or cudarc, feature-gated) may be selected at runtime only when it is compiled in, initializes successfully on the machine, and is explicitly requested or auto-detected; otherwise training runs on CPU. The PyTorch trainer (`training/train.py`) remains in the repository as the numerical reference until the Rust trainer passes a recorded parity gate on a TINY smoke run (same shards, same tokenizer artifact, comparable loss curve); after that its role is reference-only. The training method is BYVA-owned and named **ne2410**: a compiled static step program (K1), fused kernels including tied-logit-free cross-entropy (K2), a two-option optimizer policy (below), and constant-FLOPs token-budget batching (K4); each component's speed/quality target is an acceptance bar measured against the recorded PyTorch baseline, never a claim. All training claims for the Rust path remain subject to D-0008 truthfulness rules.
>
> **Optimizer policy (two options, user decision 2026-11-28):** Option 1 — **AdamW** (decoupled weight decay, fp32 master weights, global-norm gradient clipping) is the default optimizer and the parity baseline. Option 2 — **SignTrust**, BYVA's own optimizer design (K3), selectable via `--optimizer signtrust`: one EMA momentum buffer per parameter (half of AdamW's optimizer state), update `w -= lr · sign(m) · c_l` with per-layer trust ratio `c_l = rms(w_l)/rms(m_l)` clamped to a fixed band, decoupled weight decay, fully deterministic (no RNG). SignTrust may become the default only after its recorded gate passes: **equal-or-better held-out loss vs AdamW at equal wall-clock**; the clamp band is fixed in D-0022's first revision after the first measured comparison.

- Rationale: single-toolchain reproducibility for a CPU-first product; training/inference kernel sharing; deterministic control of RNG and reduction order.
- Consequences: new crates extend D-0003's six-crate boundary (the list becomes "plus trainer-path crates, added by this decision"); a new Rust-native training checkpoint format is introduced (it does not read PyTorch checkpoints); parity against the Python trainer is a mandatory gate before the Rust path is used for any recorded pretraining run; GPU support stays provisional per D-0005 until hardware validation.

## The method: ne2410 — BYVA's own design, not a framework re-run

The design goal is not to re-implement PyTorch's step loop in Rust. A from-scratch project has two freedoms the common path does not: the architecture is frozen (D-0013), so a training step can be **compiled, not interpreted**; and the optimizer/data policy can be chosen for **CPU wall-clock**, not GPU throughput. Everything below is a hypothesis paired with a measured acceptance bar — no speed or quality claim exists until the R-gates record numbers (D-0008).

### K1 — Compiled step program (static execution plan)

- For each frozen config (TINY/SMALL) the training step is one static program: buffer sizes, offsets, and kernel order are derived once at startup from `ModelConfig`, placed into a single preallocated arena, and never reallocated. Zero dynamic dispatch in the hot loop; kernels are specialized per config at compile time (const generics over `d_model`/heads), removing the branch/indirection overhead generic engines pay.
- The plan (arena layout, kernel list, shapes) is printable (`--explain-step`) and pinned by a test, so what ran is auditable per D-0008.

### K2 — Fused kernels that remove whole memory passes

- **Tied-logit-free cross-entropy**: with tied weights (D-0013), loss and its gradient are computed in one fused pass of hidden states against the embedding table — the full `[batch·seq, vocab]` logits matrix is never materialized. For TINY (seq 512, vocab 32,768) this removes the single largest activation and its write+read traffic.
- Fused pre-norm layernorm + residual; causal mask folded into the softmax pass; GELU fused into the MLP matmul epilogue.
- Every kernel has a scalar safe reference (D-0006) plus an AVX2/NEON path behind the backend abstraction — and the same kernels serve the Phase 5 runtime, so every K2 win also moves the D-0019 decode budgets.

### K3 — SignTrust (Option 2): the BYVA optimizer

Common optimizers pay either in step-scale stability (SGD-momentum) or in per-parameter state (AdamW: two moment buffers + a sqrt per parameter per step). SignTrust replaces the second moment with a per-layer trust ratio:

- State: one EMA momentum buffer per parameter — half of AdamW's optimizer state.
- Update: `w -= lr · sign(m) · c_l`, with per-layer trust `c_l = rms(w_l) / rms(m_l)` clamped to a fixed band — cheap reductions over layers, not per-parameter transcendental math. The trust ratio plays the role Adam's second moment plays (scale-awareness), paid once per layer instead of once per parameter; on CPU the optimizer cost becomes nearly bandwidth-free.
- Decoupled weight decay; warmup/cosine schedule unchanged. Update rules are fully deterministic (no RNG).
- SignTrust is **Option 2**, selectable via `--optimizer signtrust`; AdamW (Option 1) remains the default and the parity baseline until SignTrust passes its gate (equal-or-better held-out loss at equal wall-clock, per D-0022; the clamp band is fixed in D-0022's first revision after the first measured comparison).

### Research notes behind the optimizer options (web survey, 2026-11-28)

- **Lion** (Chen et al. 2023, [arXiv:2302.06675](https://arxiv.org/pdf/2302.06675)) is the closest published relative of SignTrust: it takes the sign of an interpolated momentum and needs only one state buffer (half of AdamW's memory), but requires a 3–10× smaller learning rate and 3–10× larger decoupled weight decay than AdamW; β1 = 0.99 works best.
- **Trust ratios** come from LARS/LAMB ([arXiv:1904.00962](https://arxiv.org/abs/1904.00962)): the per-layer scale `||w_l||/||u_l||` makes the update magnitude layer-adaptive; later work clamps extreme trust ratios to a band to keep them stable ([arXiv:2011.13584](https://arxiv.org/abs/2011.13584)). The standard convention when the denominator is zero is to use ratio 1 — SignTrust adopts the same rule via its `rms`-based ratio.
- **Small-batch caution**: benchmarking work ([arXiv:2509.01440](https://arxiv.org/abs/2509.01440)) finds sign-based methods' advantage widens with batch size but shrinks or vanishes at small batch sizes — directly relevant to the TINY smoke runs at batch 8. The hypothesis to validate at R5 is that the per-layer trust ratio mitigates this by making update magnitude layer-adaptive rather than uniform. This is why SignTrust ships behind a flag rather than as the default.
- **Baseline choice**: AdamW remains the dominant optimizer for LLM pretraining, so it is both the default (Option 1) and the parity baseline.

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
4. **Method gates (K6)**: each ne2410 component carries its own measured acceptance bar — throughput bar (≥ 2× the measured PyTorch step baseline on the same machine), SignTrust gate (equal-or-better held-out loss vs AdamW at equal wall-clock), latency record (p99 step time next to means) — all recorded in PERFORMANCE.md with environment, toolchain, and date before being called met.
5. Standard suite per CLAUDE.md: `cargo fmt --check`, `cargo check`, `clippy -D warnings`, `cargo test`, fuzz targets for shard/checkpoint parsing.

### Milestones (phase-scoped, per D-0008)

| Step | Content | Gate |
|---|---|---|
| R0 | D-0022 accepted; crates scaffolded behind no behavior change | workspace checks green |
| R1 | Tensor + autodiff engine | gradcheck property tests |
| R2 | `crates/data` shard reader/packer | golden manifest + fixture tests (D-0012) |
| R3 | Trainer loop, AdamW default, SignTrust behind a flag, token-budget batching, checkpoint/resume, `byva-train` CLI | resume-equivalence test; TINY smoke runs |
| R4 | Parity vs PyTorch reference (TINY); measure and record the PyTorch step baseline on the reference machine | parity gate recorded in a phase report + PERFORMANCE.md entry (env, toolchain, date) |
| R5 | ne2410 performance gates: throughput bar (K6), SignTrust gate, p99 latency record; fused kernels + tied-logit-free CE land here | measured numbers only; first budget revision of D-0022 |
| R6 | K5 inline loss-aware mixture (bounded, manifest-recorded) | per-run manifest records every adjustment; quality delta reported |
| R7 | Optional GPU backend (wgpu or cudarc) | only if hardware is available; otherwise recorded **not run** |

Milestone status:

- **R0 — done** (2026-11-28): D-0022 accepted; `byva-tensor`, `byva-autodiff`, `byva-data`, `byva-trainer` scaffolded as workspace members; `byva-tensor` ships checked fp32 dense storage with 8 unit tests; workspace checks green.
- **R1 — done** (2026-11-28): reverse-mode tape engine in `byva-autodiff` — all D-0013 ops (matmul, elementwise add/mul, broadcast position rows, scalar mul, reshape, embedding scatter-add, column slicing, head reordering, batched Q·Kᵀ and probs·V, causal-prefix softmax, f64-stat pre-norm layernorm, exact-erf GELU) plus the tied-logit-free cross-entropy (K2: logits computed row by row, the full `[n, vocab]` logits matrix is never materialized) and the two D-0022 optimizers (AdamW default, SignTrust behind config, deterministic, no RNG). Gate passed: finite-difference gradcheck property tests cover every op and a full attention-block end-to-end graph (`crates/autodiff/tests/gradcheck.rs`), plus hand-computed cases and typed-error tests.
- **R2 — done** (2026-11-28): `byva-data` shard reader/packer. D-0012 schema-version-1 manifest parsing and validation (schema/sort-key/normalization checks, plain-file-name and hex-digest validation, strictly increasing shard names), shard loading with per-shard sha256 and byte-count verification, canonical `{"id","text"}` document parsing with id-vs-sha256(text) verification and strictly increasing id order (within and across shards), SHA-256 hand-rolled in pure Rust (FIPS 180-4 vectors + 1M-'a' vector, matching `hashlib` on the fixtures), and fixed packing into `seq_len+1` windows with the reference 90/10 split (`make_sequences` semantics, including the fewer-than-ten-windows edge) plus a streaming `WindowPacker` for the R3 loop. Gate passed: golden fixtures generated by the real Python pipeline (`training/build_shards.py`, committed under `crates/data/tests/fixtures/smoke/`) validated byte-for-byte against the Rust reader, plus tamper tests (byte count, sha256, trailing newline, id/hash mismatch, unsorted ids, wrong schema version, empty shard list). The per-run id:en mixture (D-0016) joins at R3 with the run manifest, since shards carry no language tags.
- **R3 — done** (2026-11-28): the ne2410 training loop in `byva-trainer` — K4 token-budget batching (exactly `token_budget` tokens per step, batch size `token_budget / seq_len`, cycling batch cursor saved in the checkpoint), one fresh single-use autodiff graph per step over a flat parameter buffer with the runtime-canonical layout (`token_embedding.weight`, `position_embedding.weight`, `blocks.{l}.…`), deterministic init (norm gains 1, biases 0, N(0, 0.02) elsewhere) from a hand-rolled SplitMix64 + Box-Muller stream (published test vectors; keeps the zero-dependency discipline instead of `rand`/`rand_chacha` — same rationale as the hand-rolled SHA-256), warmup + cosine-to-floor LR schedule, held-out eval, AdamW (default) and SignTrust (`--optimizer signtrust`), the `byva-train-rust-checkpoint-v1` format (magic + version + JSON metadata + LE f32 blobs + sha256 trailer, atomic tmp+rename save, typed rejections for unknown format and corrupt payloads), and exact resume (config/layout/optimizer-kind/cursor/step validation, optimizer-state restore). PyTorch-convention Linear weights `[out, in]` are applied as `x @ Wᵀ` via single-batch transposed matmul. Gate passed: resume-equivalence is bit-exact (f32 `to_bits`) for loss history, eval history, and final parameters, for AdamW (6 steps, split 2+4), SignTrust (4 steps, split 2+2), a double resume (2+2+2), and a warmup-window extension via `set_total_steps`; plus checkpoint format tests (round trip, foreign magic with valid digest → `UnknownCheckpointFormat`, corrupt digest → `CheckpointCorrupt`, truncation, seq_len mismatch), toy-task convergence for both optimizers, config/data validation, and LR-schedule tests. Smoke runs (release CLI, production tokenizer artifact + smoke fixture shards, 11.6k-param-class tiny model, 128 tokens/step): AdamW loss 10.3974 → 10.3403 over 6 steps, resume through the CLI reproduces the uninterrupted losses exactly; SignTrust 5 steps 10.3974 → 10.1563. The `byva-cli train` subcommand covers `--shards/--tokenizer/--out/--resume` plus all hyperparameter flags. The per-run id:en mixture (D-0016) moves to R6 with the K5 work, since shards carry no language tags.
- **R4 — done** (2026-11-28): parity gate vs the PyTorch reference PASSED. Harness: `training/parity_rust_ne2410.py` — builds shards from a deterministic synthetic bilingual corpus with the real `build_shards.py` pipeline, encodes them with the production 32k artifact (the Rust CLI's exact data path), runs the ne2410 trainer (TINY dims, seq 128, 1024 tokens/step, AdamW lr 1e-3, PyTorch-default weight decay 1e-2 via the new `--weight-decay`/`--floor-ratio` CLI flags, `--save-init` checkpoint) and the reference `DecoderOnlyTransformer` + `torch.optim.AdamW` on the SAME initial weights and the SAME explicit sequential-cyclic batches (batch policy is a documented ne2410/K4 design difference and is excluded from the comparison). Measured over 4 steps: max abs loss diff **1.237e-06**, max abs final-weight diff **8.36e-05**. Gate fixed per open question 4: loss abs diff ≤ 5e-3, final weights abs diff ≤ 1e-3, ≥ 4 steps, same-device fp32. The gate caught a real defect before it shipped: the trainer's forward was missing the final pre-head LayerNorm (`final_norm`) present in D-0013/runtime/PyTorch — fixed in `crates/trainer/src/model.rs` (layout, init, forward). Measured PyTorch baseline: **0.386 s/step** (TINY, 1024 tokens/step, torch 2.15.0+cpu, 5 threads, Xeon 6985P-C, 2026-11-28; recorded in PERFORMANCE.md). Current ne2410 scalar-tape trainer measures ~**19.9 s/step** in release on the same config — the gap is the R5/K1–K2 work item (fused compiled step, zero-alloc, tied-logit-free CE); no performance claims are made at R4.

This path serves the Phase 3 pretraining gate; note the Phase 3 gate text says "Real T4 training run". If the accepted route becomes local CPU training, that wording needs a documented, approved gate exception (D-0008) rather than a silent reinterpretation.

## Risks and mitigations

- **Autodiff correctness**: the highest risk. Mitigated by finite-difference gradchecks, small hand-computed cases, and the mandatory PyTorch parity gate before any recorded run.
- **CPU throughput gap vs PyTorch**: PyTorch's CPU kernels are mature. Mitigation: the ne2410 design bets on removing work rather than optimizing the same work — tied-logit-free CE (K2), constant-FLOPs batching (K4), compiled step (K1); the acceptance bar is relative to a measured PyTorch baseline on the same machine, not an invented absolute. If the bar cannot be met, the decision is revisited and recorded — never silently dropped.
- **Optimizer novelty (SignTrust)**: sign-based updates can underperform AdamW on some landscapes. Contained by making AdamW the default until the SignTrust gate passes at equal wall-clock; a failed gate leaves the code as a flag, not a default.
- **GPU scope creep**: backends stay feature-gated and behind D-0005's validation rule; no GPU claim without hardware runs.
- **Two trainers drift (Rust vs Python reference)**: bounded by keeping the Python side reference-only and by the same discipline that worked for BPE (D-0020/D-0021 golden-style equivalence).

## Open questions (for the acceptance discussion)

Resolved at acceptance (2026-11-28, user decision):

1. **D-0022 accepted** as drafted; the PyTorch trainer stays as the numerical reference until the parity gate passes.
5. **SignTrust gate**: equal-or-better held-out loss vs AdamW at equal wall-clock; the trust-ratio clamp band is fixed in D-0022's first revision after the first measured comparison.
6. **Method name: ne2410** (replaces the working name "Kilat" everywhere — docs and `--method ne2410`).

Still open:

2. Crate names/numbering: confirm `tensor` / `autodiff` / `data` / `trainer` as new workspace members.
3. Should the CPU training-throughput budget live in D-0022's first revision or extend D-0019?
4. ~~Tolerance and step budget for the parity gate~~ **fixed at R4 (2026-11-28)**: loss abs diff ≤ 5e-3, final-weight abs diff ≤ 1e-3, ≥ 4 steps at TINY (1024 tokens/step), same-device fp32; measured margins 1.237e-06 and 8.36e-05.
