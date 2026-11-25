//! Native Rust inference runtime for BYVA.
//!
//! Runtime loading, CPU kernels, accelerators, generation, and KV caching are introduced in Phase 5.
//! Unsafe code, if required by optimized backends, must remain narrowly scoped and documented.

#![deny(unsafe_op_in_unsafe_fn)]
