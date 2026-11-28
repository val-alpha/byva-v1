# Phase 3 Report — From-Scratch Pretraining (Reference Implementation)

Status: IMPLEMENTATION PASSED — **the "Real T4 training run" gate item is NOT
RUN** (no GPU in the build environment; the T4 run is scheduled for Colab, see
Phase 8 plans). Everything below was actually executed on CPU.

## Environment

- Python 3.12 + PyTorch 2.15.0+cpu (installed in the build sandbox; not a repo
  dependency — training-only per CLAUDE.md)
- Rust: stable 1.100.0
- Platform: x86_64-unknown-linux-gnu, Intel Xeon 6985P-C, CPU only, no GPU
- Date: 2026-11-27

## What was implemented

### Rust: `crates/model`

- `ModelConfig` (serde, typed `thiserror` errors, checked arithmetic) with
  `validate()`, `head_dim()`, and `parameter_count()` whose formula matches the
  training implementation (embedding, learned positions, `4d²+2d·ffn` per
  block, `4d` per-layer + `2d` final LayerNorm, optional untied output).
- Two frozen configurations (D-0013):
  - `SMALL_CONFIG` — d=512, 8 layers, 8 heads, ffn 2048, seq 1024, tied →
    **42,484,736 params** (59,261,952 untied)
  - `TINY_CONFIG` — d=256, 4 layers, 8 heads, ffn 1024, seq 512, tied →
    **11,670,016 params**
- Unit tests: golden parameter counts, typed-error validation, serde round-trip.

### Python: `training/train.py` (reference trainer)

- Decoder-only pre-norm transformer authored in-repo; PyTorch used only for
  tensors/autograd. Layout mirrors `byva-model::ModelConfig` (kept in sync by
  cross-checking tests).
- `make_sequences` fixed-window packing with deterministic 90/10 held-out
  split; `train` loop (AdamW); `save_checkpoint`/`load_checkpoint`
  (`byva-train-checkpoint-v1`: step, model, optimizer, both RNG states, config,
  loss history — D-0014); greedy/sampling generation.
- Two data paths: `--smoke` (structured synthetic stream) and `--shards DIR`
  (documents from a Phase 2 shard directory; D-0015 byte-level ids until a
  trained 32k BPE vocab exists).

## Checks actually run

| Check | Command | Result |
|---|---|---|
| Rust model tests | `cargo test -p byva-model` | PASS — 4 tests |
| Training tests | `python3 -m unittest training.tests.test_train -v` | PASS — 9 tests, 0 failed |
| Pipeline tests (regression) | `python3 -m unittest discover -s training/tests` | PASS — 15 tests total |
| Param-count cross-check Rust vs Python | both suites | PASS — 42,484,736 / 11,670,016 / 59,261,952 match |
| Loss falls (synthetic) | `test_loss_falls_on_structured_smoke_stream` | PASS |
| Loss falls (real shards, byte-level) | `test_byte_level_training_loss_falls` | PASS |
| Resume == uninterrupted run | `test_checkpoint_resume_matches_uninterrupted_run` | PASS — identical loss history |
| Full Rust gate | fmt / check / clippy / test / doc / fuzz check | PASS (commands below) |

## Real-dataset smoke training run (CPU, tiny config)

Data: 500-row samples of all three Phase 2 verified sources via HF
datasets-server (TinyStories, c4 `id`, wikipedia `20231101.id`), merged by the
Phase 2 pipeline: 1,498/1,500 kept, 4 shards (~8.7 MB), manifest sha256
`fbb0ba0cdf78cc7edaceaddda18f68df3adf8bcf17e926e5693f4001afb2eb78`. Shards
stayed in /tmp, not committed (D-0007).

Command: `python3 training/train.py --shards /tmp/byva-data/shards --steps 300
--seq-len 128 --batch-size 16 --checkpoint /tmp/byva-data/smoke.ckpt`, then a
`--resume` run of 100 more steps.

Measured results (byte-level ids, tiny config):

- Loss fell **10.4689 → 2.7365** over 300 steps (val 3.15 → 2.58 at step 300).
- Resumed run continued 300 → 400 steps: loss **2.7365 → 2.6617**, val 2.5496,
  with the loss history carried through the checkpoint.
- Held-out greedy generation from the prompt `"Once upon a time"` produced
  degenerate byte-level text (`"n dan dan dan …"`) — expected for an 11.7M-param
  byte-level model after 400 steps; recorded as-is (no overclaiming).

## Addendum (2026-11-27): bilingual id+en expansion (D-0016)

Per the language-focus decision, the verified source set grew from three to
seven datasets (see docs/DATA.md): + FineWeb, FineWeb-Edu (both ODC-BY,
English), OSCAR-2301 (CC0-1.0, gated — sampling pending an HF token), and
wikipedia `20231101.en`. uonlp/CulturaX was verified and **rejected** (license
field empty on the Hub).

Second real-dataset smoke run (same tiny config, byte-level, CPU):

- Corpus: 5 accessible sources, 2,493/2,500 docs kept, 5 shards, manifest
  sha256 `1f6e7f52a149d8d02080ae5f6495840b4078700d4d02e353e5aa7fa6694f8b07`.
- Command: `python3 training/train.py --shards /tmp/byva-data/shards-bilingual
  --steps 400 --seq-len 128 --batch-size 16`.
- Loss fell **10.4433 → 2.7811** over 400 steps (val 3.37 → 2.80); held-out
  greedy generation remains degenerate byte-level text, recorded as-is.
- OSCAR-2301 (`id`, the main additional Indonesian source) is NOT sampled:
  datasets-server returns HTTP 401 without authentication. Pending the user's
  HF token; everything else was fetched anonymously.

### Third smoke run (2026-11-27, deeper samples + HF token)

- Samples deepened to 5,300 docs / ~43 MB (c4-id 1,500, wikipedia-id 1,200,
  fineweb-edu 1,000, wikipedia-en 800, TinyStories 800); pipeline kept
  5,288/5,300 into 9 shards, manifest sha256
  `8a850f36da89f7ab920790452785ee88034abb154e62de93ba0959ed6afb8026`.
- OSCAR-2301 stays unsampled for a different reason: with a valid HF token the
  datasets-server endpoints return 404 (the dataset is not indexed in the
  viewer), and its repository distributes bulk `*_meta` parts of ~600 MB
  `.jsonl.zst` each (16 parts for `id`) — bulk-download territory, deferred to
  the Colab scale-up. The token itself worked (401 → authenticated).
- Command: `--steps 600 --seq-len 128 --batch-size 16` on the 5,288-doc
  corpus. Loss fell **10.4423 → 3.0652** train over 600 steps; **val loss
  fell monotonically 3.19 → 2.8749** — the bilingual corpus is still being
  learned at 600 steps (no plateau), consistent with the tiny byte-level
  config being far from convergence.
- Greedy generation remains degenerate byte-level text; recorded as-is.

## Not run

- **Real T4 training run** — the environment has no GPU. The gate item stays
  open; the SMALL_CONFIG run on Colab T4 (FP16 + gradient accumulation) is the
  remaining Phase 3/8 hardware task, with a Colab notebook to be prepared.
- **c4/wikipedia full-corpus shards** — needs multi-GB downloads and storage
  outside this environment.
- **OSCAR-2301 sample** — gated dataset (HTTP 401); needs HF authentication.

## Addendum (2026-11-28): BPE vocabulary trained — SUPERSEDED by D-0020

> **Status after the D-0020 reconciliation:** this addendum records the run of
> the word-type pre-tokenized trainer variant, which was trained against a
> regex pre-tokenization while the Rust tokenizer serves through
> `encode_bytes` (byte-level, no pre-tokenization). The exact byte-level
> trainer is canonical, this trainer variant has been removed together with
> its Python tests, and its committed artifact is kept only until the
> canonical trainer replaces it (see the 2026-11-28 local-run addendum and
> D-0020). The numbers below stay here as the recorded history of that run.

The 32,768-vocabulary BPE table required by D-0015 now exists.

- Corpus: the same five-source sample as the third smoke run (c4 `id` 1,500,
  wikipedia `20231101.id` 1,200, fineweb-edu 1,000, wikipedia
  `20231101.en` 800, TinyStories 800), re-fetched from datasets-server on
  2026-11-28; the Phase 2 pipeline kept 5,288/5,300 docs into 6 shards,
  shard manifest sha256
  `f2e672d0c05c2a47b240177e101eebde8dd710c0f69e382b8e708a0d4ca124ef`.
- Trainer: `training/train_bpe.py` (pure stdlib, deterministic per D-0017).
  Runtime: 32 s on the build environment CPU. 244,687 word types.
- Result: 32,465 merges (vocab 32,768 = 303 base tokens + merges).
  Training-corpus compression: 41,366,863 bytes → 14,816,989 tokens
  (2.79 bytes/token).
- Held-out compression (240 fresh docs sampled at distant offsets from the
  same four sampling sources, never in the training corpus): 898,058 bytes →
  329,039 tokens = **2.73 bytes/token** (c4-id 2.80, wiki-id 2.88,
  wiki-en 2.70, fineweb-edu 2.67).
- Committed artifacts: `artifacts/tokenizer/byva-tokenizer-v1.json`
  (sha256 `bb380795929e6ff89e3437ce1590928454727759bbec78a4e37f255d27b6cd77`)
  and its training manifest `byva-tokenizer-v1.manifest.json`.
- Validation actually run on that commit: new Rust integration test
  `crates/tokenizer/tests/bpe_artifact.rs` (5 tests: artifact loads under the
  published vocab bounds, bilingual round trips, compression, all 32,465
  merges expand to bytes, control IDs stay outside content decoding) and the
  word-trainer Python suite (12 tests, since removed with the superseded
  trainer per D-0020).
- Still open: resolved on this branch — `train.py` gained `--tokenizer`
  (D-0018), so production pretraining consumes a trained artifact.

## CI

`training/tests/test_train.py` skips cleanly (not fails) when PyTorch is
absent, so the existing stdlib-only CI step stays green; torch-ful runs are
recorded here instead.

## Full Rust gate commands (run 2026-11-27, all PASS)

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- `cargo check --manifest-path fuzz/Cargo.toml`

## Session addendum (2026-11-27, CPU sandbox): BPE trainer + Colab notebook

### What was implemented and run

- `training/train_bpe.py` — deterministic byte-level BPE trainer (pure stdlib,
  D-0017): trains directly on shard documents with no pre-tokenization
  boundary; incremental pair counts with lazy heap selection, tie-break
  (count desc, left asc, right asc); emits a versioned artifact loadable by
  `Tokenizer::from_artifact`.
- `training/byva_tokenizer.py` — Python mirror of the Rust tokenizer
  (NFC + ranked merges via the same lazy-heap algorithm, validation, decode via
  merge-child expansion) used by the trainer and trainer tests.
- `training/train.py` extended (D-0018): `--config {tiny,small}`,
  `--tokenizer <artifact>` (production BPE path), `--device`, `--fp16`
  (CUDA autocast + GradScaler), `--grad-accum N`; checkpoint format bumped to
  `byva-train-checkpoint-v2` storing GradScaler state (v1 still loads).
- `training/colab_t4.ipynb` — Colab T4 notebook: build shards → train the
  32,768 BPE table → SMALL FP16 run with grad accumulation and Drive-backed
  resumable checkpoints → print gate evidence.

### Tests actually run in this session

- `python3 -m unittest tests.test_train_bpe` — 12 tests PASS (artifact shape,
  naive-BPE equivalence, merge order, permutation determinism, exhaustion and
  bounds errors, end-to-end via build_shards, golden regression, control-id
  rejection).
- `cargo test -p byva-tokenizer --test golden` — 3 tests PASS including the new
  cross-language round-trip: the Python-trained golden fixture
  (`crates/tokenizer/tests/golden/bpe_smoke_v1.json`, 64 merges) loads into the
  Rust tokenizer and round-trips the fixture texts exactly.
- `python3 -m unittest discover -s training/tests` — 31 tests PASS (PyTorch
  2.15.0+cpu installed in this session, same version as the original Phase 3
  environment): all BPE trainer tests plus the torch suite, including the new
  grad-accumulation resume bit-for-bit test, checkpoint v2 save/load with v1
  backward compatibility, the fp16-requires-CUDA guard, and a BPE-token-stream
  training run with falling loss.
- `python3 training/train.py --smoke --steps 60` — PASS (CLI end-to-end after
  the D-0018 changes: loss 10.4615 → 0.0422, learned pattern regenerated).

### Still open for the Phase 3 gate

- Real T4 training run (SMALL, FP16, grad accumulation) — notebook ready; run
  it in Colab and record evidence here.
- The 32,768 BPE table itself is NOT yet trained: no Phase 2 full-corpus
  shards exist in this environment, and the exact trainer is memory-bounded;
  train it in Colab per the notebook (or a future scaled trainer, D-0017).

## Addendum (2026-11-28): local CPU training run — Colab deferred by user decision

The Colab T4 route was deferred by user decision after repeated mid-run
interruptions of the shard-build cell (three runs recorded in PR #1
comments). A real training run was executed locally instead, on the same
trainer code path the notebook uses.

### Environment

- Sandbox CPU: 5 cores, 5 GB RAM, PyTorch 2.15.0+cpu (fp32, 5 threads).
- Nothing corpus- or checkpoint-related enters Git (D-0007).

### Corpus (same five verified sources as the notebook, 60:40 id:en)

- wikipedia 20231101.id 2,400 / c4 id 2,400 / fineweb-2 ind_Latn 1,200 /
  wikipedia 20231101.en 2,400 / fineweb-edu sample-10BT 1,600 = 10,000 docs,
  ~99 MB via the HF datasets-server.
- `build_shards.py`: 9,962 docs kept, 10 shards, manifest verified.

### Tokenizer (exact trainer, D-0017/D-0018 semantics)

- Trained on a 2.77 MB subsample (519 docs, ~0.6 MB per source): the exact
  trainer's documented memory bound (occurrence-index scaling) was confirmed
  empirically — 20 MB OOM-killed at ~3.8 GB RSS, 2.77 MB peaked at ~1.2 GB.
- Result: 32,465 merges (vocab 32,768) in 162 s; trainer stats: 5.53
  bytes/token on its own corpus.
- Caveat: subsample-trained table; compression is weaker than a
  full-corpus table would be. A memory-bounded scaled trainer remains the
  open item for production.

### Model training (TINY, D-0013: 11,670,016 params, verified from checkpoint)

- `train.py --config tiny --tokenizer <artifact> --device cpu --steps 200
  --seq-len 128 --batch-size 8`: 296 s wall clock (~1.1 s/step after a
  ~200 s corpus encode).
- Loss: train 10.4658 → 9.2745; val 9.1688 → 8.9338, monotone decreasing.
- Resume verified per D-0014: `--resume` +10 steps continued 200 → 210 with
  val loss 8.9338 → 8.8407; checkpoint v2 (140 MB) reloads config, model,
  optimizer, scaler, and loss history.

### Gate status after this run

- "Real training run with the 32,768-vocab BPE tokenizer": DONE at smoke
  scale on CPU (TINY). The Phase 3 T4/SMALL gate item is superseded by user
  decision to defer Colab, not passed; revisit when GPU access returns.
- Still open: full-corpus shards, production-scale BPE training, SMALL run.
