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
