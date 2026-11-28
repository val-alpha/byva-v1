# BYVA Engineering Decisions

Add dated entries for material decisions. Record superseded decisions rather than silently rewriting history.

## D-0001 — Project scope and provenance
- Status: Accepted
- Date: 2026-11-26
- Decision: BYVA parameters are created through pretraining from random initialization. No third-party pretrained weights or model checkpoints are permitted.
- Rationale: The project requires genuine from-scratch pretraining, not fine-tuning or repackaging.
- Consequences: Training provenance, dataset manifests, and checkpoint lineage must be auditable.

## D-0002 — Rust-native inference boundary
- Status: Accepted
- Date: 2026-11-26
- Decision: Tokenization, BYVA formatting, checkpoint loading, sampling, KV cache, quantization, and inference execution are implemented in Rust. PyTorch may be used for training-time tensor operations and autograd only.
- Rationale: Inference must not depend on another model runtime.
- Consequences: Training-to-Rust numerical parity and a documented checkpoint interchange format are required gates.

## D-0003 — Workspace and crate boundaries
- Status: Accepted
- Date: 2026-11-26
- Decision: Use one Cargo workspace with six crates: tokenizer, format, model, runtime, convert, and cli.
- Rationale: The boundaries separate serialized formats, model semantics, execution, conversion, and user-facing commands.
- Consequences: Cross-crate dependencies must remain directed and minimal. Shared dependency versions belong in the workspace manifest.

## D-0004 — Rust toolchain and baseline quality
- Status: Accepted
- Date: 2026-11-26
- Decision: Use stable Rust, Rust 2024 edition, a declared Rust 1.85 minimum, workspace lint configuration, rustfmt, Clippy, and GitHub Actions checks.
- Rationale: A stable compiler baseline and repeatable CI reduce integration risk.
- Consequences: Dependency upgrades must respect the declared MSRV. CI is authoritative for its declared toolchain; local results must be reported separately.

## D-0005 — GPU backend strategy
- Status: Provisional
- Date: 2026-11-26
- Decision: Keep backend selection behind a Rust runtime abstraction. Investigate direct CUDA Driver API access for NVIDIA and wgpu for Vulkan/Metal, with CPU fallback. Do not use a third-party model inference runtime.
- Rationale: Keep inference orchestration and kernels under BYVA control while providing a possible path across T4, Vulkan, and Apple hardware.
- Consequences: Driver availability, supported operations, memory behavior, and performance must be experimentally validated. CUDA and wgpu support must not be claimed before hardware tests pass. Revise this decision if direct-driver constraints make a backend impractical.

## D-0006 — CPU safety and unsafe code
- Status: Accepted
- Date: 2026-11-26
- Decision: Keep unsafe code limited to the smallest hardware-specific boundaries. Every unsafe block requires an adjacent SAFETY comment; malformed inputs and integer/offset calculations must be handled without panics.
- Rationale: SIMD and low-level memory operations can require unsafe Rust, but must not weaken parser, loader, or public API safety.
- Consequences: Safe reference implementations and tests must exist for optimized paths.

## D-0007 — Repository artifact hygiene
- Status: Accepted
- Date: 2026-11-26
- Decision: Do not commit raw/processed corpora, checkpoints, model binaries, or secrets. Track source manifests, scripts, and small reproducible fixtures instead.
- Rationale: Large and sensitive artifacts need explicit storage and provenance controls.
- Consequences: Data and training workflows must document external storage and checksum conventions.

## D-0008 — Phase gates and truthfulness
- Status: Accepted
- Date: 2026-11-26
- Decision: Commit each phase separately after evaluating its stated checks. Reports distinguish implemented, tested, and not-yet-validated work.
- Rationale: Repository state must not imply validation that has not occurred.
- Consequences: A failing gate blocks the next phase unless an explicit exception is documented and approved.

## D-0009 — Repository license metadata
- Status: Open
- Date: 2026-11-26
- Decision: Do not declare a package or repository license until the repository owner selects one.
- Rationale: A public GitHub repository is not, by itself, an indication that its contents are licensed for reuse.
- Consequences: License metadata and any LICENSE file remain intentionally absent. Dataset license review in Phase 2 remains mandatory regardless of the repository license.

## D-0010 — CPU-first benchmarking policy
- Status: Accepted
- Date: 2026-11-26
- Decision: Reproducible Criterion benchmarks live in `crates/*/benches` and are recorded per environment in docs/PERFORMANCE.md, starting with a CPU-only baseline for tokenizer and format on a small-model merge table (8k merges). GPU numbers are recorded only after capability detection and validation on real hardware; a backend without measured runs is never described as supported or fast.
- Rationale: CPU is the correctness baseline and the only hardware available in the current environment; early baselines catch tokenization/parsing regressions before inference exists.
- Consequences: Benchmarks compile in CI via workspace checks but are not run in CI; recorded numbers must state toolchain, CPU, and date. Phase 7 replaces these entries with reproducible Criterion plus end-to-end benchmarks.

## D-0011 — Dataset admission requires primary-source license verification
- Status: Accepted
- Date: 2026-11-26
- Decision: A dataset enters the BYVA corpus only after its license is read from the dataset's own metadata with the dataset sha recorded at retrieval time, and its redistribution obligations (attribution, share-alike) are noted in docs/DATA.md. Synthetic-data caveats are recorded explicitly.
- Rationale: Third-party license summaries drift and can be wrong; provenance must be auditable from the primary source.
- Consequences: Phase 2 review verified TinyStories (CDLA-Sharing-1.0, synthetic), allenai/c4 (ODC-BY 1.0), and wikimedia/wikipedia (CC BY-SA 3.0 + GFDL) via the Hugging Face dataset API on 2026-11-26 because the Exa connector was unavailable; that substitution is recorded as such. Unreviewed sources stay unadmitted.

## D-0012 — Shards are deterministic and manifests are portable
- Status: Accepted
- Date: 2026-11-26
- Decision: The shard pipeline (training/byva_data.py) normalizes (NFC), filters by length, deduplicates by SHA-256, sorts by content-hash id, and emits canonical-JSON shards whose output depends only on the document set — never on input order, file grouping, or absolute paths. Manifests record file names (not paths), per-shard sha256 and byte counts, config, and stats. Determinism is enforced in CI by a frozen fixture regression test.
- Rationale: Reproducible shards are the Phase 2 gate; nondeterministic preprocessing would make training runs incomparable.
- Consequences: Any change to normalization, filtering, hashing, or canonical-JSON encoding that alters output must update the frozen fixture and its expected manifest in the same commit, with the change called out in the gate report.

## D-0013 — Decoder-only architecture and two frozen configurations
- Status: Accepted
- Date: 2026-11-27
- Decision: BYVA is a decoder-only transformer with pre-norm blocks, causal multi-head attention, GELU MLP (attention/MLP linears without bias), learned absolute positions, and tied token-embedding/output weights by default. Two configurations are frozen in crates/model: SMALL (d=512, 8 layers, 8 heads, ffn 2048, seq 1024; 42,484,736 params) as the Colab T4 training target, and TINY (d=256, 4 layers, 8 heads, ffn 1024, seq 512; 11,670,016 params) for CPU/CI smoke runs and benchmarks. The Python trainer mirrors the Rust ModelConfig, and cross-checking tests pin the parameter counts on both sides.
- Rationale: The T4 16GB target is memory- and throughput-constrained; a ~42M model trains comfortably with FP16 and leaves room for activations, while TINY keeps gates runnable on CPU-only hardware. Weight tying saves the 16.8M-parameter output projection.
- Consequences: Phase 4 conversion and the Phase 5 Rust runtime must implement exactly this layout; any architecture change updates both sides plus the frozen golden counts in the same commit.

## D-0014 — Resumable training checkpoint format
- Status: Accepted
- Date: 2026-11-27
- Decision: Training checkpoints use format `byva-train-checkpoint-v1` and store step, model state, optimizer state, torch and Python RNG states, the model config, and the loss history. Resuming must reproduce the uninterrupted run exactly; this is enforced by a test that compares loss histories.
- Rationale: Phase 3's gate requires resumable checkpoints, and Colab sessions die mid-run; a checkpoint that loses RNG or optimizer state would silently fork the training trajectory.
- Consequences: Checkpoints stay out of Git (D-0007). Adding fields requires bumping the format version and rejecting unknown formats with a typed error.

## D-0015 — Dataset smoke runs use byte-level ids until the BPE vocab exists
- Status: Accepted
- Date: 2026-11-27
- Decision: Until a 32,768-merge BPE table is trained from BYVA shards, dataset smoke trainings encode UTF-8 bytes directly (ids 0..255) inside the full-size vocabulary. Production pretraining uses only the trained tokenizer artifact, never raw bytes.
- Rationale: ModelConfig pins the production vocabulary range, and no tokenizer artifact at that size exists yet; byte-level smokes still exercise the full training/checkpoint/generation path on real data without blocking on tokenizer training.
- Consequences: Smoke loss numbers are byte-level and are not comparable to future BPE numbers; reports must label which encoding produced a number. Training the BPE table on Phase 2 shards is a prerequisite for the real T4 run.

## D-0016 — Bilingual language focus: Indonesian + English only
- Status: Accepted
- Date: 2026-11-27
- Decision: BYVA is trained as a bilingual Indonesian/English model. Only per-language configs of verified sources are admitted (c4 `id`, wikipedia `20231101.id`/`20231101.en`, OSCAR-2301 `id`/`en`, FineWeb/FineWeb-Edu English, TinyStories English). Multilingual dumps are consumed exclusively through their id/en subsets, and no other language enters pretraining.
- Rationale: A small (~42M) model has limited capacity; concentrating it on two languages maximizes measured quality, Indonesian is the project's primary audience, and English data provides the strongest public capability signal that transfers.
- Consequences: Mixture weights (id:en ratio) are decided per training run and recorded in the run manifest; added sources must fit the two-language scope or be rejected. OSCAR-2301 sampling is blocked on Hub authentication (HTTP 401) and stays pending until an HF token is provided.

## D-0017 — BPE trainer determinism contract and Python tokenizer mirror
- Status: Accepted
- Date: 2026-11-27
- Decision: The 32,768-merge BPE table is trained by `training/train_bpe.py` (pure stdlib) directly on Phase 2 shard documents with no pre-tokenization boundary, matching `Tokenizer::encode_bytes`. Determinism is contractual: pair counts are aggregated over the document set (input order never matters), each step merges the pair with the highest count, ties broken by (left, right) ascending, and occurrences are replaced left-to-right without overlap. `training/byva_tokenizer.py` mirrors the Rust encode/decode/validate semantics for training-time use, and a golden artifact trained on the smoke fixture (`crates/tokenizer/tests/golden/bpe_smoke_v1.json`) must regenerate byte-identically in Python and round-trip through the Rust tokenizer in CI.
- Rationale: Training/serve segment mismatch would silently corrupt the model; D-0015 makes the trained table a prerequisite for the real T4 run, so its determinism must be pinned cross-language now.
- Consequences: Any change to the Rust encoder or the Python mirror that alters segmentation must update both sides and regenerate the golden fixture in the same commit. The exact incremental trainer is memory-bounded (index scales with corpus size); corpora beyond a few tens of MB need a scaled trainer before the 32k table can be trained on them — recorded as an open Phase 3/8 hardware item.

## D-0018 — Checkpoint format v2: FP16 GradScaler state, fp16 and gradient-accumulation training
- Status: Accepted
- Date: 2026-11-27
- Decision: `byva-train-checkpoint-v2` extends v1 with the GradScaler state (null when FP16 is off); the loader accepts v1 and v2 and rejects unknown formats. The reference trainer gains `--config {tiny,small}` (frozen D-0013 configurations), `--tokenizer <artifact>` (production BPE path; byte-level stays the explicit fallback per D-0015), `--device`, `--fp16` (CUDA autocast + GradScaler), and `--grad-accum N` (per-micro-batch backward, loss history records the accumulated mean cross-entropy so it is comparable across accumulation factors).
- Rationale: The T4 gate requires a real FP16 run of the SMALL config, and Colab sessions die mid-run; a resumed run that drops scaler state would fork the loss trajectory exactly like lost RNG state.
- Consequences: Phase 4 conversion must accept v2 checkpoints; resuming an FP16 run requires a v2 checkpoint (v1 loads only for FP32). Any further field addition bumps the format again per D-0014.
## D-0019 — CPU-first performance budget and quality guardrails

- Status: Accepted
- Date: 2026-11-28
- Decision: CPU speed is a first-class product requirement. The Rust runtime
  (Phase 5) is designed against these single-thread greedy-decode budgets on a
  documented reference CPU, measured with the Phase 7 harness before being
  called met: TINY (11.7M params) ≥ 200 tokens/s and SMALL (42.5M params)
  ≥ 50 tokens/s in fp32, with int8 weight quantization as the default CPU
  path and int4 as opt-in. Quality is guarded, not assumed: a quantized or
  optimized path may become default only after a recorded held-out loss/eval
  delta against the fp32 reference stays within the threshold recorded in this
  decision's first revision, measured on the Phase 3 bilingual held-out split.
- Rationale: BYVA targets small models whose users run them on CPU-only
  machines; "fast" must be a tracked budget with numbers, and "still smart"
  must be a measured delta, or optimization silently trades one for the other.
- Consequences: the runtime decode loop must be allocation-free per token
  (preallocated buffers, contiguous row-major weights); SIMD paths (AVX2/NEON)
  stay behind the backend abstraction with safe fallbacks and SAFETY-documented
  unsafe (D-0006); first measured CPU decode numbers are recorded in
  docs/PERFORMANCE.md when Phase 5 lands; budgets may be revised only with
  measured evidence, never by editing the target silently.

## D-0020 — Reconciliation of the parallel BPE trainers; exact byte-level trainer is canonical

- Status: Accepted
- Date: 2026-11-28
- Decision: Two BPE trainers had evolved in parallel (`byva/implementation` f81c36d: word-type pre-tokenized trainer with a committed artifact; this branch: exact byte-level trainer matching `Tokenizer::encode_bytes`). The exact byte-level trainer is canonical because the Rust tokenizer serves through `encode_bytes` (no pre-tokenization), so training/serve segmentation is identical by construction (extends D-0017). The word-type trainer variant, its tests (`training/tests/test_bpe.py`), and its mini fixture are removed. The committed 32,768 artifact produced by the superseded trainer stays in place (it remains a valid table under byte-level serving) only until it is retrained by the canonical trainer and replaced together with a new training manifest (D-0007/D-0012 provenance rules apply).
- Rationale: two live trainers with a duplicated D-0017 would let training and serving segmentation drift; the serving side is the fixed point, so the trainer must follow it.
- Consequences: all future vocabulary training uses the canonical trainer path (scaled Rust trainer for large corpora, D-0021); any trainer change must keep the cross-language golden fixture byte-identical; artifact replacement requires a new manifest and updated Rust integration tests in the same commit.

## D-0021 — Scaled Rust trainer for production corpora

- Status: Accepted
- Date: 2026-11-28
- Decision: The canonical exact trainer (D-0017/D-0020) gains a memory-bounded Rust implementation, `byva_tokenizer::trainer::train_bpe` (CLI `cargo run --release -p byva-tokenizer --bin train_bpe`), used for all training corpora beyond a few tens of MB where the pure-Python trainer exhausts RAM. It is contractually identical to the Python trainer: aggregated pair counts over the document set, (count desc, left asc, right asc) selection, occurrences replaced in ascending (document, index) order without overlap, and pairs below two occurrences are not mergeable. Documents are `u32` symbol arrays; pair positions are packed (document, index) `u64` entries in per-pair sets, with a stale-tolerant candidate heap that is compacted when it exceeds four times the live pair count plus a fixed slack. Cross-language equivalence is pinned by the golden fixture (byte-identical artifact) and was verified byte-identical against the Python trainer on a 2.77 MB shard sample at the full 32,768 vocabulary; the Python trainer remains the readable reference definition of the contract.
- Rationale: the Python exact trainer peaked near the sandbox RAM ceiling on a 20 MB corpus, so the 99 MB local corpus (and later full shards) cannot train in Python; a second implementation only stays safe if byte-identity is enforced, not assumed.
- Consequences: any change to either trainer must keep the golden fixture byte-identical and re-verify cross-language equivalence on a multi-hundred-merge corpus; the Rust trainer's input JSONL is NFC-normalized at read time, matching `load_shard_texts`; production artifact retraining (D-0020) uses this trainer.

## D-0022 — Pure-Rust training path; method named ne2410; two-option optimizer policy (AdamW default, SignTrust behind a flag)

- Status: Accepted
- Date: 2026-11-28
- Decision: Model pretraining gains a second, BYVA-owned implementation in pure Rust: a minimal tensor/autodiff engine and a training loop authored in this repository, with no Python and no PyTorch/libtorch dependency on the training path. Execution is CPU-only by default; a GPU backend (wgpu and/or cudarc, feature-gated) may be selected at runtime only when it is compiled in, initializes successfully on the machine, and is explicitly requested or auto-detected; otherwise training runs on CPU. The PyTorch trainer (`training/train.py`) remains in the repository as the numerical reference until the Rust trainer passes a recorded parity gate on a TINY smoke run (same shards, same tokenizer artifact, comparable loss curve); after that its role is reference-only. The training method is BYVA-owned and named **ne2410** (user-named 2026-11-28, replacing the working name "Kilat"; CLI flag `--method ne2410`): a compiled static step program (K1), fused kernels including tied-logit-free cross-entropy (K2), a two-option optimizer policy (below), and constant-FLOPs token-budget batching (K4); each component's speed/quality target is an acceptance bar measured against the recorded PyTorch baseline, never a claim. **Optimizer policy (two options, user decision 2026-11-28):** Option 1 — AdamW (decoupled weight decay, fp32 master weights, global-norm gradient clipping) is the default optimizer and the parity baseline. Option 2 — SignTrust, BYVA's own optimizer design (K3), selectable via `--optimizer signtrust`: one EMA momentum buffer per parameter (half of AdamW's optimizer state), update `w -= lr · sign(m) · c_l` with per-layer trust ratio `c_l = rms(w_l)/rms(m_l)` clamped to a fixed band (zero-norm convention: ratio 1), decoupled weight decay, fully deterministic. SignTrust may become the default only after its recorded gate passes: equal-or-better held-out loss vs AdamW at equal wall-clock; the clamp band is fixed in this decision's first revision after the first measured comparison. All training claims for the Rust path remain subject to D-0008 truthfulness rules. Full design, research notes (Lion arXiv:2302.06675; LARS/LAMB arXiv:1904.00962, trust-ratio clipping arXiv:2011.13584; small-batch caution arXiv:2509.01440), and milestones R0–R7: `docs/PROPOSAL_RUST_TRAINING.md`.
- Rationale: single-toolchain reproducibility for a CPU-first product (D-0019); training/inference kernel sharing; deterministic control of RNG and reduction order. Sign-based optimizers halve optimizer state, and the per-layer trust ratio supplies the scale-awareness Adam gets from its second moment at per-layer instead of per-parameter cost; because sign-based methods lose their advantage at small batch sizes (arXiv:2509.01440), SignTrust ships behind a flag with AdamW as the measured default rather than replacing it.
- Consequences: new crates extend D-0003's six-crate boundary (`byva-tensor`, `byva-autodiff`, `byva-data`, `byva-trainer`); a new Rust-native training checkpoint format `byva-train-rust-checkpoint-v1` is introduced (it does not read PyTorch checkpoints); parity against the Python trainer is a mandatory gate before the Rust path is used for any recorded pretraining run; GPU support stays provisional per D-0005 until hardware validation; SignTrust stays a non-default flag until its gate passes with a recorded comparison.
- Revision 1 (2026-11-28, R5 measured data, `training/bench_ne2410.py`): the first measured comparison fixes the trust-ratio clamp band provisionally at **0.25-4.0** (unchanged from the provisional design value; the lr sweep 0.25x/0.5x/1x/2x of 1e-3 plus weight decay 5x/10x showed no interior optimum at TINY). Both K6 gates recorded as FAILs, honestly: **throughput 0.10x** vs the fresh PyTorch baseline (0.275 s/step; bar >= 2x) — portable safe Rust auto-vectorization measures ~42 G MAC/s aggregate vs PyTorch's AVX-512 MKL, so the bar is unreachable without explicit SIMD/intrinsics or a GPU backend (candidate R7 work), a revised gate definition, or a D-0008 documented exception; **SignTrust gate FAIL at TINY** (held-out 4.80-6.56 across the sweep vs AdamW 4.4478, eval curves worsening monotonically on the synthetic corpus, batch 8) — consistent with the small-batch caution (arXiv:2509.01440); SignTrust stays a non-default flag and its re-evaluation moves to a larger-batch/corpus setting instead of further TINY tuning. The CPU training-throughput budget lives here (not in D-0019); AdamW remains the only recorded training optimizer until SignTrust passes its gate in a setting where its design hypothesis applies. Full numbers: `docs/PERFORMANCE.md` (R5 section).

## D-0023 — Phase 5 runtime: `byva-weights-v1` container, SplitMix64 parity fixture, scalar fp32 backend first

- Status: Accepted
- Date: 2026-11-28
- Decision: The Phase 5 runtime consumes a little-endian `byva-weights-v1`
  container (magic `BYVAW`, versioned header embedding the `ModelConfig`,
  canonical tensor order matching the PyTorch state dict, per-tensor f32
  payloads) validated by `byva_runtime::weights::Weights`; the Phase 4
  converter writes exactly this format. Numerical parity with the PyTorch
  training reference is pinned without committing a weight blob: both sides
  synthesize weights from a documented SplitMix64 generator (fixture seed
  0x2026_1128_5EED_0001, scale 0.02) — `training/make_runtime_fixture.py`
  runs the reference forward pass and records the exact f32 logits (bit
  patterns, full 32,768-wide vector at the final prompt position) while
  `byva_runtime::lcg::fixture_weights` rebuilds the identical weights. The
  Phase 5 backend is portable scalar fp32 (f64 LayerNorm accumulation,
  exact-erf GELU); SIMD and int8/int4 quantization stay behind this interface
  for Phase 7 (D-0019), and no D-0019 budget claim is made from Phase 5
  numbers.
- Rationale: a committed synthetic weight blob would duplicate what the two
  generators define and could drift; the generator pair makes the weight
  source auditable while the committed logit bits make the numerical contract
  checkable in CI on any machine.
- Consequences: Phase 4 conversion must emit `byva-weights-v1` in the
  canonical tensor order; any D-0013 architecture change must regenerate the
  fixture through the generator pair in the same commit; the fixture holds
  reference logits only — no corpora or checkpoints (D-0007). Decode uses a
  preallocated KV cache with no per-token allocation; context is bounded by
  `max_seq_len` learned positions (no sliding window at Phase 5).

## D-0024 — Phase 4 converter: Rust-native zip+pickle subset parser, fp32-only weights, canonical-name validation

- Status: Accepted
- Date: 2026-11-28
- Decision: `byva_convert` converts PyTorch training checkpoints
  (`byva-train-checkpoint-v1`/`-v2`, per D-0014/D-0018) to the
  `byva-weights-v1` container (D-0023) natively in Rust, with no Python
  dependency at conversion time. The parser implements exactly the subset
  `torch.save` emits for this payload, verified by disassembling real
  archives: stored-method zip entries prefixed with the archive stem (zip64
  extra/EOCD fields parsed when the 32-bit sentinels are present) and a
  protocol-2 pickle interpreter for the observed opcode set — including
  `collections.OrderedDict` reduction, the model OrderedDict's `_metadata`
  state applied via `BUILD` (merged and ignored), and
  `torch._utils._rebuild_tensor_v2` tensors whose strided views are gathered
  into dense little-endian bytes through lazily resolved `data/<key>`
  storages. The converter accepts only fp32 model storages, requires the
  model state dict names to match the canonical layout implied by the
  checkpoint's config exactly (unknown and missing names are typed errors),
  checks every value is finite, and emits weights through
  `Weights::from_tensors` before the CLI re-reads the written file to verify
  the round trip. v1 checkpoints carrying scaler state are rejected
  (D-0018). Test checkpoints are synthesized in Rust from the same structure
  (no checkpoint files committed, D-0007); real-torch equivalence is
  verified locally per run and recorded in `docs/PHASE4_REPORT.md`.
- Rationale: the conversion path is part of the trusted weight supply chain;
  relying on Python or permissively-parsed pickle at serve/convert time would
  widen the attack surface and the toolchain, while a narrow, typed,
  bounds-checked parser keeps malformed input a rejected input, never a
  panic.
- Consequences: any PyTorch serialization change (new opcodes, new storage
  classes, compression) must extend the parser deliberately with tests; the
  converter stays read-only over checkpoints and never executes pickled
  code; weight-layout assumptions (`[out, in]` Linear, fused `[3d, d]` QKV)
  are shared with the runtime via `canonical_tensor_layout`, so a D-0013
  architecture change updates both sides in one commit.

## D-0025 — Phase 6 SFT: masked-loss chat template, built-in byva/BYV identity dataset, Rust-checkpoint convert path

- Status: Accepted
- Date: 2026-11-28
- Decision: Supervised fine-tuning renders each example as
  `<|bos|>|user|>{prompt}|assistant|>{response}<|end|>` with the tokenizer's
  existing control tokens (no vocabulary change) and trains with loss masked
  to the response tokens plus the closing `<|end|>` only. Examples are
  GPT-style packed into a concatenated stream (windows `seq_len + 1`,
  canonical 90/10 split, per-target weight masks); batches with no supervised
  position contribute loss 0.0 and skip the optimizer update, and evaluation
  selects only supervised windows. `fine_tune` starts from a native Rust
  checkpoint (dims must match except `seq_len` may be shorter, with the
  position-embedding prefix carried over) and uses a fresh optimizer.
  Checkpoints record `masked_loss` in the manifest, and plain/masked resume
  must agree with it. SFT data is JSONL
  (`{"id","prompt","response"}`, `id == sha256_hex(NFC(prompt+"\n"+response))[:16]`)
  or the built-in bilingual byva identity dataset
  (`--identity`, upsampled by `--repeat`). `byva_convert` accepts the native
  Rust checkpoint (`byva-train-rust-checkpoint-v1`, `BYVATRCK` magic) and the
  CLI gains `sft` plus `generate --chat`. The model identity is fixed as:
  name **byva**, creator **BYV Dev Unorganization (BYV)** with the
  real-world identity undisclosed (no guessing, inferring, or inventing a
  personal identity), answering in Indonesian and English; the identity
  dataset encodes exactly this attribution. The Phase 6 identity gate is a
  smoke-scale check (`training/bench_sft.py`); the full Phase 6 scope
  (held-out format, schema, tool-selection, multi-step evaluations) remains
  open.
- Rationale: masking keeps pretraining knowledge from being diluted by
  prompt tokens; reusing the existing control tokens avoids a vocab/parity
  migration (D-0013/D-0023 fixtures stay valid); the native Rust checkpoint
  convert path removes the last PyTorch dependency from the serve chain
  (D-0024) so fine-tuning feeds the runtime directly.
- Consequences: any template or mask change must re-run the identity gate
  and the convert→runtime parity test; `resume`/`resume_masked` selection is
  now part of the checkpoint contract (manifest `masked_loss` flag); the
  built-in identity dataset is authoritative for byva/BYV attribution and
  must not be edited to name any personal identity; full Phase 6 tool-behavior
  evaluations are tracked as open work.
