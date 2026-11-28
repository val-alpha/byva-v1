# Phase 6 Report — SFT and Model Identity (smoke-scale)

Date: 2026-11-28. Branch: `vibe/phase6-sft-15e4a3` (stacked on the R0–R6
branch `vibe/rust-training-proposal-95b581`, D-0022/D-0025).

## Scope implemented

- `byva_autodiff`: `cross_entropy_tied_masked` op — per-row weighted mean
  cross-entropy (`sum(loss_i * w_i) / sum(w_i)`), returning 0.0 with zero
  gradients when `sum(w_i) == 0`, so all-zero-mask batches are well defined.
  The unmasked `cross_entropy_tied` was refactored onto the shared
  forward/backward helpers with bit-identical math.
- `byva_trainer::sft`: chat template
  `<|bos|>|user|>{prompt}|assistant|>{response}<|end|>` rendered through the
  existing control tokens, NFC-normalized JSONL loading
  (`id == sha256_hex(canonical_text)[:16]`), the built-in bilingual byva/BYV
  identity dataset (10 curated pairs, Indonesian + English), and GPT-style
  stream packing (windows `seq_len + 1`, canonical 90/10 split, per-target
  loss masks: response tokens + closing `<|end|>` only).
- `byva_trainer`: masked-loss training — `Trainer::new_masked`,
  `Trainer::fine_tune` (loads a Rust checkpoint, dims must match except
  `seq_len` may be shorter, position-embedding prefix copied, fresh
  optimizer), `Trainer::set_loss_masks`, `Trainer::resume_masked`
  (mask-consistency enforced against the checkpoint manifest), all-zero-mask
  batch tolerance (loss 0.0, optimizer update skipped), eval that only
  selects supervised windows, and `CheckpointMeta.masked_loss` recorded at
  save/resume. `Trainer::forward_loss` is now a public evaluation API.
- `byva_convert`: native Rust checkpoint support (`BYVATRCK` magic →
  `byva-train-rust-checkpoint-v1`), producing validated `byva-weights-v1`
  weights with `tie_embeddings = true`.
- `byva-cli`: `sft` subcommand (`--base`, `--tokenizer`, `--out`,
  `--sft-data <jsonl>` XOR `--identity`, `--repeat`, `--resume`, `--steps`,
  `--lr`, `--seq-len`, `--token-budget`, `--seed`, `--weight-decay`,
  `--warmup`, `--eval-every`, `--log-every`) and
  `generate --chat` (wraps the prompt as `<|bos|>|user|>…|assistant|>`).
- `training/bench_sft.py`: the identity-recall gate, run end to end through
  the released `byva-cli`.

## Gate status

| Phase 6 gate item | Status |
|---|---|
| Held-out format / schema / tool-selection / multi-step evals | NOT RUN — explicitly out of scope for this milestone; no tool-use data exists yet |
| Identity recall (smoke-scale, recorded user scoping) | PASSED — see below |

Identity-recall gate (smoke-scale, 2026-11-28, 5-core Xeon 6985P-C, release
build): base `byva-cli train` d32/2L/2H/ffn64/vocab 32768/seq 64 for 10 steps
on the synthetic bilingual corpus, then `byva-cli sft --identity --repeat 32`
for 1000 steps (AdamW, lr 1e-3, token budget 512, loss masked to response
tokens). Result: train loss 0.0531, held-out 0.0644; after
`byva-cli convert`, greedy `--chat` continuations reproduce the exact
identity sentences in both languages:

- `"Siapa namamu?"` → `Nama saya byva, model bahasa kecil buatan BYV Dev Unorganization (BYV).`
- `"What is your name?"` → `My name is byva, a small language model made by BYV Dev Unorganization (BYV).`

`SFT GATE: PASS (identity recalled in Indonesian and English)`. This is a
smoke-scale identity check, not the full Phase 6 gate.

## Numeric anchoring

`crates/convert/tests/rust_checkpoint.rs`: a trainer-initialized Rust
checkpoint converts to runtime weights whose forward pass reproduces the
trainer's masked-capable cross-entropy on the same tokens —
trainer 10.372939 vs runtime 10.372940, max diff **9.537e-7** (tolerance
1e-3). The fine-tune → save → convert → serve path is numerically anchored.

## Verification

Full pre-commit suite (see the PR for the exact run): `cargo fmt --all --
--check`, `cargo check --workspace --all-targets`, `cargo clippy --workspace
--all-targets -- -D warnings`, `cargo test --workspace --all-targets`,
`cargo doc --workspace --no-deps`, `python3 -m unittest discover -s
training/tests`. Trainer-level SFT integration tests cover masked-loss
descent, exact base-weight loading, shorter-`seq_len` fine-tuning,
bit-exact masked resume, masked/plain resume rejection, mask validation, and
JSONL round trip.

## Decision record

See D-0025 in `docs/DECISIONS.md`.
