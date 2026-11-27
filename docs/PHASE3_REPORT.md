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

## Not run

- **Real T4 training run** — the environment has no GPU. The gate item stays
  open; the SMALL_CONFIG run on Colab T4 (FP16 + gradient accumulation) is the
  remaining Phase 3/8 hardware task, with a Colab notebook to be prepared.
- **BPE vocab training** — the 32,768-token merge table does not exist yet, so
  dataset smokes train on byte ids (D-0015). Production pretraining requires
  the trained tokenizer artifact.
- **c4/wikipedia full-corpus shards** — needs multi-GB downloads and storage
  outside this environment.
- **OSCAR-2301 sample** — gated dataset (HTTP 401); needs HF authentication.

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
