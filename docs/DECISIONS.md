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

## D-0017 — Deterministic BPE training and artifact storage

- Status: Accepted
- Date: 2026-11-28
- Decision: The tokenizer vocabulary is trained by `training/train_bpe.py` on
  Phase 2 shards. Documents are pre-tokenized with a fixed Unicode-class regex
  (letter runs, digit runs, whitespace runs, single other characters), reduced
  to sorted word-type frequencies, and merges are selected greedily with
  tie-breaking (highest pair count, then lowest left token id, then lowest
  right token id), so a given document set yields a byte-identical artifact.
  The trained 32,768-vocab artifact and its training manifest are committed
  under `artifacts/tokenizer/` as source-derived data — not a corpus,
  checkpoint, or model binary.
- Rationale: D-0015 requires a trained 32k BPE table before production
  pretraining. The training corpus is a datasets-server sample that can drift
  between retrievals, so byte-identical regeneration cannot be guaranteed;
  committing the small JSON artifact and validating it in Rust CI keeps the
  production vocabulary stable and auditable.
- Consequences: `crates/tokenizer/tests/bpe_artifact.rs` loads the committed
  artifact and would fail on drift. Any retraining (e.g., on full-corpus
  shards during the Colab scale-up) must record a new training manifest and
  either supersede the artifact under a new version/name or reproduce the
  existing one exactly. `training/train.py` still encodes byte ids until it
  gains `--tokenizer` support, after which production pretraining consumes
  only the trained artifact (D-0015).

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
