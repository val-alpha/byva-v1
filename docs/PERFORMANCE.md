# Performance Baseline — CPU Only

First recorded performance baseline for BYVA tokenizer and format components.
Numbers are single-session medians from Criterion on this specific machine; treat
them as relative baselines for regression tracking, not absolute product claims.

## Environment of the recorded run

- Date: 2026-11-26
- Toolchain: stable rustc 1.100.0, criterion 0.7, `bench` profile (optimized)
- CPU: Intel(R) Xeon(R) 6985P-C @ 2.30GHz, 5 cores visible to the sandbox
- GPU: none detected — no `nvidia-smi`, no `/dev/dri`, no `/dev/kfd`, no VGA/3D
  PCI devices. Per D-0005 a backend is not "supported" until validated on real
  hardware, so no GPU numbers exist yet.

## Reproduce

~~~bash
cargo bench --workspace
~~~

Criterion stores its baselines under `target/criterion/`.

## Tokenizer (small-model table: 8,192 merges, ~8.6k vocab)

| Benchmark | Input | Median | Throughput |
|---|---|---|---|
| tokenizer_encode | 576 B text | 17.3 µs | 31.8 MiB/s |
| tokenizer_decode | 576 B / 576 tokens | 12.3 µs | 44.7 MiB/s |
| tokenizer_load (JSON artifact parse + validate) | 8k merges (~200 KB JSON) | 397 µs | 505 MiB/s |

Notes:
- Encode covers NFC normalization + BPE with the small merge table.
- Decode expands every token to bytes and validates UTF-8.
- Production artifacts (32k–48k vocab) will be slower to load; the load benchmark
  uses `from_merges` because `from_artifact` intentionally rejects sub-32k tables.

## Format: renderer, streaming parser, OpenAI conversion

Conversation fixture: system + (user → assistant[think, tool call] → tool result →
assistant[text]) per turn, tool definitions present; byte-level tokenizer with an
empty merge table.

| Benchmark | Workload | Median | Throughput |
|---|---|---|---|
| format_render | 1 turn (~403 tokens out) | 8.8 µs | — |
| format_render | 8 turns (~2,153 tokens out) | 46.9 µs | — |
| format_parse | 1 turn, 403 tokens streamed | 7.3 µs | 55.3 M events/s |
| format_parse | 8 turns, 2,153 tokens streamed | 52.7 µs | 40.9 M events/s |
| format_openai | events_to_openai, 8 turns | 2.6 µs | — |

Streaming-parser throughput is per token fed (each push expands tokens to bytes,
maintains parser state, and emits deltas).

## Tokenizer: production 32,768-vocab artifact (D-0017)

Recorded 2026-11-28 on the same environment (Xeon 6985P-C, stable rustc,
criterion 0.7). These use the committed
`artifacts/tokenizer/byva-tokenizer-v1.json` (32,465 merges) via `from_artifact`,
i.e. the real production table, not the synthetic 8k one.

| Benchmark | Input | Median | Throughput |
|---|---|---|---|
| tokenizer_encode | 576 B text, 32k table → 240 tokens | 49.1 µs | 11.7 MiB/s |
| tokenizer_decode | 576 B / 240 tokens, 32k table | 7.7 µs | 74.8 MiB/s |
| tokenizer_load (`from_artifact`: JSON parse + full validation) | 32,465 merges (~823 KB JSON) | 1.86 ms | 453 MiB/s |

Notes:
- Production encode is ~2.8× slower than the 8k table (deeper merge chains to
  inspect), but produces 240 vs 576 tokens on the same text — the 2.4×
  compression is what matters downstream: fewer tokens into the model.
- Load is a one-time cost per process and stays under 2 ms at production size.
- Same 576 B bilingual sample as the 8k baseline for direct comparison.

## Decoder reference proxy (PyTorch, training-side, NOT the D-0019 claim)

Recorded 2026-11-28, same environment. `training/bench_decode.py` on the
TINY checkpoint (11.7M params, fp32, greedy, no KV cache, full forward per
token), PyTorch CPU, `torch.set_num_threads(1)`, prompt 128 + 128 new tokens,
median of 5 runs:

| Model | Threads | Median throughput |
|---|---|---|
| TINY (PyTorch reference) | 1 | 26.6 tok/s |

This is the training-side baseline only. The D-0019 budget (TINY >= 200
tok/s single-thread fp32 greedy) is claimed met exclusively by the Rust
runtime measured with the Phase 7 harness; the reference implementation is
~7.5x below budget, which is the headroom the Rust decode loop (incremental
context, allocation-free per-token path) must recover and exceed.

Reproduce:

~~~bash
python3 training/bench_decode.py --checkpoint <ckpt.pt>
~~~

## Rust runtime decode, fp32 scalar (Phase 5, D-0023)

Recorded 2026-11-28 on the same environment (Xeon 6985P-C, stable rustc
1.100.0, criterion 0.7, release build). TINY configuration (11.7M params,
D-0013) with synthetic SplitMix64 fixture weights (D-0023), scalar fp32
backend, preallocated KV cache, greedy, single thread. Measured via
`cargo bench -p byva-runtime --bench decode_bench` (64-token prompt prefill
excluded from timing, 64 greedy decode tokens per iteration, 100 samples):

| Model | Backend | Median per token | Throughput |
|---|---|---|---|
| TINY | scalar fp32 | 5.70 ms | 175 tok/s (174.4–176.4) |

Notes:
- 6.6x above the PyTorch reference proxy (26.6 tok/s, above), which had no KV
  cache and re-ran the full forward per token.
- The D-0019 budget (TINY >= 200 tok/s) is NOT claimed here; the formal claim
  belongs to the Phase 7 harness. The remaining ~14% gap is expected to close
  with the int8 default path and SIMD backend (D-0019), which are not built
  yet at Phase 5.
- Logits projection (tied embeddings, 32,768 x 256 fp32 = 33.5 MB streamed
  per token) dominates the scalar decode cost; this bounds the scalar fp32
  ceiling and motivates the int8 default path.

## What is deliberately absent

- GPU benchmarks: no GPU in this environment; will be recorded only after
  capability detection and validation on actual hardware (Phase 5/7).
- Dataset/shard benchmarks: Phase 2 and later.
- SIMD and quantized-path numbers: the int8/int4 paths and SIMD backends are
  Phase 7 work behind the D-0023 backend interface.
