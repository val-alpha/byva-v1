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

## What is deliberately absent

- Inference/decoder benchmarks: the runtime crate has no execution engine yet
  (Phase 5). No inference performance claim is possible or made.
- GPU benchmarks: no GPU in this environment; will be recorded only after
  capability detection and validation on actual hardware (Phase 5/7).
- Dataset/shard benchmarks: Phase 2 and later.
