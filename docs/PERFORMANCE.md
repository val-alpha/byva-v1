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

## Trainer step baseline: PyTorch reference vs ne2410 Rust trainer (R4, D-0022)

- Date: 2026-11-28
- Toolchain: torch 2.15.0+cpu (5 threads, Python 3.12.14); Rust release build
  (rustc 1.100.0), scalar fp32 autodiff tape, one graph per step
- CPU: Intel(R) Xeon(R) 6985P-C @ 2.30GHz, 5 cores visible to the sandbox
- Config: TINY (D-0013, 11,670,016 params), seq 128, batch 8, exactly 1024
  tokens/step, AdamW lr 1e-3 / weight decay 1e-2 / eps 1e-8 / betas 0.9+0.999
- Parity context: same shards, same initial weights, same explicit batches;
  max abs loss diff 1.237e-06 over 4 steps, max abs final-weight diff
  8.36e-05 (harness: `training/parity_rust_ne2410.py`)

| Trainer | s/step | Notes |
|---|---|---|
| PyTorch reference (`training/train.py` math) | **0.386** | median-of-run mean over 4 steps; BLAS-backed |
| ne2410 Rust trainer (R3/R4 state) | **~19.9** | marginal over 20 steps incl. CLI overhead; scalar tape, per-step graph allocation, no SIMD/fusion |

Notes:
- This is the R5 (K1/K2) starting point, not a gate result. The ne2410
  throughput bar (≥ 2× the PyTorch baseline at equal tokens/step) is claimed
  only in R5 with the fused compiled step, zero-alloc path, and
  tied-logit-free CE in place.
- The PyTorch baseline supersedes the earlier proxy figure (26.6 tok/s decode,
  above) for training-step comparisons; that proxy measured greedy decode
  without KV cache, a different workload.

## Trainer step throughput: ne2410 K2 fused kernels (R5, D-0022)

- Date: 2026-11-28
- Toolchain: Rust release build (rustc 1.100.0), fused f32 kernels in
  `crates/autodiff/src/graph.rs`; torch 2.15.0+cpu (5 threads, Python
  3.12.14) for the reference
- CPU: Intel(R) Xeon(R) 6985P-C @ 2.30GHz, 5 cores visible to the sandbox
- Config: TINY (11,670,016 params), seq 128, exactly 1024 tokens/step,
  40 steps (eval steps excluded from step-time samples), AdamW lr 1e-3,
  weight decay 1e-2 (harness: `training/bench_ne2410.py`)

| Trainer | mean s/step | p50 | p99 | Notes |
|---|---|---|---|---|
| PyTorch reference | **0.275** | 0.278 | 0.299 | AVX-512 MKL; 0.28-0.39 across sessions |
| ne2410 (R3/R4 scalar tape) | ~19.9 | - | - | R4 starting point |
| ne2410 AdamW (R5, K2) | **2.628** | 2.533 | 3.401 | fused threaded kernels, f32 accumulation |
| ne2410 SignTrust (R5, K2) | **2.673** | 2.687 | 3.276 | optimizer cost ~80 ms/step |

K2 kernel work (deterministic, thread-count independent):

- f32 accumulation with fixed sequential order per output element; parallelism
  only over disjoint output rows/elements (fixed partition), so results are
  bit-identical for any thread count. Parity gate after the rewrite: max abs
  loss diff 8.900e-07, final-weight diff 5.903e-05 (both improved vs R4's
  f64-accumulated 1.237e-06 / 8.36e-05).
- Cross-entropy: per-row stats cached in forward (max/sum/loss), backward
  split into two parallel passes (hidden grads, then embedding-table grads);
  tied-logit-free (full `[tokens, vocab]` logits never materialized). The R4
  backward wrote the 33.5 MB `gtable` once per row (~34 GB of traffic/step).
- Matmuls: axpy-form over a transposed operand (`batched_nt_rows` now
  transposes to `[k, n]` and reuses the NN kernel). Kernel micro-benchmark
  (1024x256x768, release): dot form 10.3 G MAC/s aggregate -> axpy form
  38.5 G MAC/s (NT) and 42.4 G MAC/s (NN); single-thread `dot_f32` 5.5
  G MAC/s. Step time 19.9 -> ~4.5 (first fusion round) -> ~2.5 s/step.
- Allocations: per-step graph leaf-copy + value collection measured at
  ~25 ms of a ~2.5 s step (~1%); the full K1 tape-replay program/arena
  (zero-alloc step) is deferred as not the bottleneck.

Gate results (D-0022/K6, honest record):

- **Throughput gate FAIL: 0.10x** vs PyTorch (bar >= 2x). Portable safe Rust
  (SSE2-baseline auto-vectorization) measures ~42 G MAC/s aggregate vs
  PyTorch's AVX-512 MKL dispatch. Recorded for D-0022 revision 1: options are
  explicit SIMD/intrinsics or a GPU backend (candidate R7 work), a revised
  gate definition, or a D-0008 documented exception.
- **SignTrust gate FAIL at TINY**: held-out loss at 40 steps, equal wall-clock
  (AdamW lr 1e-3 wd 1e-2: 4.4478; SignTrust evals worsen monotonically after
  step 10 on this synthetic corpus): lr 2.5e-4 (0.25x) -> 4.8041, 5e-4 (0.5x)
  -> 5.8373, 1e-3 (1x) -> 5.4549, 2e-3 (2x) -> 6.5643; weight decay 5x/10x at
  the two lower lrs -> 5.8226 / 4.7831 (no trend change). Consistent with the
  small-batch caution (arXiv:2509.01440): batch 8. SignTrust stays a
  non-default flag; re-evaluation deferred to a larger-batch/corpus setting.
- p99 step latency recorded above (3.401 s AdamW / 3.276 s SignTrust) next to
  the means, per the K6 latency discipline row.
