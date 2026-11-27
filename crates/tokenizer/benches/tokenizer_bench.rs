//! CPU-only Criterion benchmarks for the BYVA tokenizer, sized for a small model
//! (8k merges, ~8.6k vocabulary). These are relative comparisons, not model-quality
//! claims; GPU backends are benchmarked separately once validated hardware exists.

use std::hint::black_box;
use std::time::Duration;

use byva_tokenizer::{BYTE_TOKEN_BASE, FIRST_MERGE_ID, Merge, Tokenizer};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const SMALL_MERGE_COUNT: usize = 8_192;

fn small_model_tokenizer() -> Tokenizer {
    let mut merges = Vec::with_capacity(SMALL_MERGE_COUNT);
    for pair in 0..256 {
        merges.push(Merge {
            left: BYTE_TOKEN_BASE + pair,
            right: BYTE_TOKEN_BASE + pair,
        });
    }
    for step in 0..(SMALL_MERGE_COUNT - 256) {
        merges.push(Merge {
            left: FIRST_MERGE_ID + step as u32,
            right: BYTE_TOKEN_BASE + (((step as u32) * 13 + 7) % 256),
        });
    }
    Tokenizer::from_merges(1, merges).expect("small-model merge table is valid")
}

fn sample_text() -> String {
    let sentence = "the quick brown fox jumps over the lazy dog while untuk berita hari ini ";
    sentence.repeat(8)
}

fn bench_encode(c: &mut Criterion) {
    let tokenizer = small_model_tokenizer();
    let text = sample_text();
    let mut group = c.benchmark_group("tokenizer_encode");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.measurement_time(Duration::from_secs(10));
    group.bench_with_input(
        BenchmarkId::from_parameter(format!("{}b_8k_merges", text.len())),
        &text,
        |b, text| b.iter(|| tokenizer.encode_content(black_box(text))),
    );
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let tokenizer = small_model_tokenizer();
    let text = sample_text();
    let encoded = tokenizer.encode_content(&text);
    let mut group = c.benchmark_group("tokenizer_decode");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(format!("{}b_{}_tokens", text.len(), encoded.len())),
        &encoded,
        |b, tokens| b.iter(|| tokenizer.decode_content(black_box(tokens))),
    );
    group.finish();
}

fn bench_artifact_reload(c: &mut Criterion) {
    let tokenizer = small_model_tokenizer();
    let artifact = tokenizer.to_artifact();
    let json = serde_json::to_string(&artifact).expect("artifact serializes");
    let mut group = c.benchmark_group("tokenizer_load");
    group.throughput(Throughput::Bytes(json.len() as u64));
    group.bench_function("artifact_json_8k_merges", |b| {
        b.iter(|| {
            let parsed: byva_tokenizer::TokenizerArtifact =
                serde_json::from_str(black_box(&json)).expect("artifact parses");
            Tokenizer::from_merges(parsed.version, parsed.merges).expect("artifact validates")
        })
    });
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_artifact_reload);
criterion_main!(benches);
