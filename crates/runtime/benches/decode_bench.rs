//! Criterion benchmark for the Phase 5 fp32 greedy decode loop (D-0010).
//!
//! Measures single-thread tokens/s for the frozen TINY configuration with
//! synthetic fixture weights; these are Phase 5 measured numbers, not the
//! formal D-0019 budget claim (reserved for the Phase 7 harness).

use std::hint::black_box;
use std::time::Instant;

use byva_model::TINY_CONFIG;
use byva_runtime::InferenceModel;
use byva_runtime::lcg::fixture_weights;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const PROMPT_TOKENS: usize = 64;
const DECODE_TOKENS: u32 = 64;

fn bench_greedy_decode(criterion: &mut Criterion) {
    let model =
        InferenceModel::from_weights(fixture_weights(&TINY_CONFIG)).expect("fixture model builds");
    let prompt = (0..PROMPT_TOKENS)
        .map(|index| ((index * 509) % TINY_CONFIG.vocab_size as usize) as u32)
        .collect::<Vec<_>>();

    let mut group = criterion.benchmark_group("runtime_decode");
    group.throughput(Throughput::Elements(u64::from(DECODE_TOKENS)));
    group.bench_function("tiny_fp32_greedy", |bench| {
        bench.iter_custom(|iterations| {
            let mut elapsed = std::time::Duration::ZERO;
            for _ in 0..iterations {
                let mut session = model.new_session();
                for &token in &prompt {
                    let _ = session.step(token);
                }
                let start = Instant::now();
                let mut next = session.greedy_token();
                for _ in 0..DECODE_TOKENS {
                    let _ = session.step(next);
                    next = session.greedy_token();
                    black_box(session.logits());
                }
                elapsed += start.elapsed();
            }
            elapsed
        })
    });
    group.finish();
}

criterion_group!(benches, bench_greedy_decode);
criterion_main!(benches);
