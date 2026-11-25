# BYVA Phase Gates

| Phase | Scope | Required gate |
|---|---|---|
| 0 | Cargo workspace, conventions, CI, baseline docs | Formatting, workspace check, Clippy, tests, docs |
| 1 | Tokenizer and BYVA format | Round-trip/property/golden/fuzz/injection/truncation tests |
| 2 | Verified data research and pipeline | Exa source verification, license review, reproducible shards |
| 3 | From-scratch pretraining | Real T4 training run, falling measured loss, resumable checkpoint, held-out generation |
| 4 | Native checkpoint | Validated conversion, reload, metadata/tensor integrity |
| 5 | Rust inference runtime | Functional tests, numerical comparison to training reference, backend validation |
| 6 | SFT and tool behavior | Held-out format, schema, tool-selection, and multi-step evaluations |
| 7 | Runtime optimization | Reproducible Criterion and end-to-end benchmarks |
| 8 | Scaled training and Colab | Script/notebook smoke tests and operational documentation |

A gate is passed only when its checks have actually run and the results are recorded. If the available environment cannot run a hardware-specific test, record it as not run rather than passed.
