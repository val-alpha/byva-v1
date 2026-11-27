# Phase 1 Gate Report — Tokenizer and BYVA Format

Status: PASSED (with one environment limitation recorded below)

## Environment

- Toolchain: stable rustc 1.100.0, cargo 1.100.0 (rust-toolchain.toml channel = stable)
- Platform: x86_64-unknown-linux-gnu, Intel Xeon 6985P-C @ 2.30GHz, CPU only
- Date: 2026-11-26

## Gate checks actually run

| Check | Command | Result |
|---|---|---|
| Formatting | `cargo fmt --all -- --check` | PASS |
| Workspace check | `cargo check --workspace --all-targets` | PASS |
| Clippy | `cargo clippy --workspace --all-targets -- -D warnings` | PASS |
| Tests | `cargo test --workspace --all-targets` | PASS — 29 tests, 0 failed |
| Docs | `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` | PASS |
| Fuzz target compile | `cargo check --manifest-path fuzz/Cargo.toml` (also enforced in CI) | PASS |

## Gate coverage

| Gate item | Evidence |
|---|---|
| Round-trip | `arbitrary_unicode_round_trips_after_nfc` (property), `renderer_round_trip_preserves_assistant_events`, `artifact_serde_round_trips_and_reloads_deterministically` |
| Property | proptest suites in tokenizer (Unicode round-trip) and format (marker-looking user text stays content) |
| Golden | `committed_fixture_ids_match_the_v1_layout` loads `crates/tokenizer/tests/golden/tokenizer_v1.json`; `tokenizer_v1_golden_ids_are_stable`; format golden tests pin rendered IDs |
| Fuzz | 3 libFuzzer targets (`parser_tokens`, `stream_parser`, `tokenizer_round_trip`) compile in CI; equivalent bounded harnesses run deterministically on stable (below) |
| Injection | `parser_rejects_control_ids_from_untrusted_sources`, `untrusted_marker_looking_text_is_plain_content`, `rendered_token_looking_user_text_is_not_control`, `reject_policy_can_reject_token_looking_content`, `token_like_content_is_encoded_as_ordinary_bytes`, `control_ids_cannot_be_decoded_as_content` |
| Truncation | `parser_handles_every_truncated_prefix_without_panicking` (every prefix), `truncated_or_invalid_utf8_content_fails_without_panicking`, `parser_enforces_call_byte_limit` |

## Bounded fuzz sessions (run, not merely compiled)

Harnesses mirror the libFuzzer targets using deterministic xorshift seeds, run on stable:

- `crates/format/tests/bounded_fuzz.rs::bounded_random_token_ids_never_panic_or_recover_after_failure` — 2,000,000 conversations x up to 32 random token IDs each, seed 0x0DDB1A5E5EEDBA11: 0 failures.
- `crates/format/tests/bounded_fuzz.rs::bounded_random_byte_streams_never_panic` — 2,000,000 streams x up to 128 bytes, seed 0x00C0FFEEDEAD10CC: 0 failures.
- `crates/tokenizer/tests/bounded_fuzz.rs::bounded_random_utf8_text_always_round_trips` — 2,000,000 texts x up to 96 bytes, seed 0x9E3779B97F4A7C15: 0 failures.

All three completed in ~5 s wall clock total (FUZZ_ITERATIONS=2000000). Default iterations
(20,000 each) run in normal `cargo test`.

## Not run (recorded per gate rule)

- Native libFuzzer sessions (`cargo fuzz run`) require the nightly toolchain plus
  cargo-fuzz, which could not be installed in this sandbox within the working session.
  The fuzz targets are compiled in CI (`cargo check --manifest-path fuzz/Cargo.toml`),
  and the bounded harnesses above execute the same fuzz semantics deterministically.
  A nightly libFuzzer campaign should be run when a suitable environment is available.
