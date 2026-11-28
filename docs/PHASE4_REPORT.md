# Phase 4 Report — Native Checkpoint Converter

Date: 2026-11-28. Branch: `vibe/phase4-converter-15e4a3` (base `byva/implementation`).

## Scope implemented

- `byva_convert` (new crate, D-0024): converts PyTorch training checkpoints
  (`byva-train-checkpoint-v1`/`-v2`, D-0014/D-0018) to the `byva-weights-v1`
  container (D-0023) natively in Rust. No Python dependency at conversion
  time; PyTorch is used only to generate a throwaway local test checkpoint.
- `convert::zip`: stored-method zip reader — EOCD scan, central directory
  parse with zip64 extra/EOCD64 fallback when the 32-bit sentinels are set,
  local-header entry extraction, stem-suffix entry lookup. Any compression
  method other than stored is a typed error.
- `convert::pickle`: protocol-2 pickle interpreter for exactly the opcode
  subset `torch.save` emits for this payload (verified by disassembling a
  real archive): scalars, unicode, dict/list/tuple construction, memoization
  (`BINPUT`/`BINGET`/`LONG_BINPUT`/`LONG_BINGET`/`MEMOIZE`, `POP`,
  `POP_MARK`), `GLOBAL`/`STACK_GLOBAL`/`REDUCE` for
  `collections.OrderedDict` and `torch._utils._rebuild_tensor_v2`,
  `BINPERSID` storage references, and `BUILD` state application (the model
  OrderedDict's `_metadata` state is merged and later ignored). Tensor
  rebuilds gather strided views into dense little-endian bytes; storages are
  resolved lazily from `data/<key>` entries with declared-size checks. All
  reads are bounds-checked; malformed input is a typed error with byte
  positions, never a panic.
- `convert::checkpoint`: payload mapping → `Checkpoint` (format string
  accepted: v1/v2 only; `step`; config dict → validated `ModelConfig`; model
  state dict → ordered `(name, TensorData)` with `_metadata` skipped and
  storage-extent validation). v1 with scaler state is rejected (D-0018).
- `convert::converter`: `checkpoint_to_weights` — model names must match the
  canonical layout implied by the config exactly (`UnknownModelTensor` /
  `MissingModelTensor`), fp32 storages only (`TensorDType`), element counts
  must equal the layout (`TensorShape`), all values finite
  (`NonFiniteTensor`), then assembly through
  `byva_runtime::Weights::from_tensors`.
- `byva_runtime::weights`: added `canonical_tensor_layout` (shared canonical
  layout for converter and loader) and the writer pair
  `Weights::from_tensors` / `Weights::to_bytes` (same validation rules as the
  loader; byte-for-byte compatible with the container format).
- `byva-cli convert --checkpoint <file> --out <file>`: parse → convert →
  write → re-read and byte-compare the reload → print format, step, config,
  tensor count, and `parameter_count`.

## Gate status (Phase 4: validated conversion, reload, metadata/tensor integrity)

| Gate item | Status | Evidence |
|---|---|---|
| Validated conversion (real torch checkpoint) | PASSED | Real checkpoint `ckpt_v2_test.pt` generated with torch 2.15.0+cpu (tiny config: vocab 32,768, d_model 16, 1 layer, 2 heads, ffn 32, seq 32, tied; step 5, AdamW state populated). `byva-cli convert` produced `/tmp/converted_v2.byva` (2,108,228 bytes, 12 tensors). Independent Python check parsed the output container and compared every element against `model.state_dict()`: **bit-exact (max diff 0.0, byte-level equality via f32 bit patterns), state-dict order identical, whole file consumed.** |
| Reload | PASSED | `byva-cli convert` re-reads the written file with `Weights::from_path` and byte-compares against the serialized image (mismatch is a hard error). Unit tests additionally round-trip `from_tensors → to_bytes → from_bytes` and compare every tensor slice. |
| Metadata/tensor integrity | PASSED | Config re-validated through `ModelConfig::validate` (vocab range, positive dims, head divisibility); tensor names must exactly match the canonical layout (unknown → `UnknownModelTensor`, missing → `MissingModelTensor`); fp32-only storages; per-tensor element counts equal layout expectations; non-finite values rejected; storage byte lengths checked against declared numel both at resolve time and after rebuild. |
| Real-torch end-to-end generate | PASSED | `byva-cli generate --weights <converted> --tokenizer artifacts/tokenizer/byva-tokenizer-v1.json` decoded greedily through the production tokenizer (output is gibberish as expected for a 5-step model; the point is the weight path, not the text). |
| Synthetic pipeline tests (CI) | PASSED | `crates/convert/tests/pipeline.rs` builds complete PyTorch-format checkpoints (zip + protocol-2 pickle, OrderedDict + `_metadata` BUILD, `_rebuild_tensor_v2` + `BINPERSID` storages) in Rust — no binary fixtures committed (D-0007) — and covers: v2 full pipeline (convert → reload → forward pass through `InferenceModel`), v1 without scaler, v1+scaler rejection, unknown format, unknown/missing tensor names, non-fp32 storage, non-finite values, short storage, truncated zip, non-mapping payload. |
| Formatting / check / clippy / tests / docs | PASSED | `cargo fmt --all -- --check`; `cargo check --workspace --all-targets`; `cargo clippy --workspace --all-targets -- -D warnings`; `cargo test --workspace --all-targets` (15 suites, 0 failures; convert: 12 lib + 11 integration); `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`; `cargo check --manifest-path fuzz/Cargo.toml`; `python3 -m unittest discover -s training/tests` (31 tests OK). |

## Notes

- No corpora, checkpoints, or model binaries are committed (D-0007); the
  real-torch verification used a throwaway checkpoint generated locally.
- The pickle parser deliberately does not execute pickled code; unknown
  opcodes, globals, and storage dtypes are typed errors (D-0024).
- Phase 3's T4/SMALL training gate remains deferred by recorded user
  decision; that deferral does not gate Phase 4 conversion work.
