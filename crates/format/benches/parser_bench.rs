//! CPU-only Criterion benchmarks for BYVA conversation rendering and token-level
//! streaming parse. Measures renderer and streaming parser speed on a small-model
//! tokenizer; relative numbers, not model-quality claims.

use std::hint::black_box;
use std::time::Duration;

use byva_format::{
    AssistantPart, Conversation, Message, ParseEvent, Renderer, ToolCall, ToolDefinition,
    ToolResult,
};
use byva_tokenizer::Tokenizer;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn single_merge_tokenizer() -> Tokenizer {
    Tokenizer::from_merges(1, Vec::new()).expect("empty merge table is valid")
}

fn tool() -> ToolDefinition {
    ToolDefinition {
        name: "cuaca".to_owned(),
        description: "Weather lookup".to_owned(),
        parameters: serde_json::json!({"type":"object","properties":{"kota":{"type":"string"}}}),
    }
}

fn tool_call_conversation(turns: usize) -> Conversation {
    let mut messages = vec![Message::system("You are a helpful assistant.")];
    for turn in 0..turns {
        messages.push(Message::user(format!(
            "Bagaimana cuaca di Jakarta untuk pertanyaan nomor {turn}?"
        )));
        messages.push(Message::assistant(
            vec![
                AssistantPart::Think("perlu memeriksa cuaca".to_owned()),
                AssistantPart::Call(ToolCall {
                    id: format!("call-{turn}"),
                    name: "cuaca".to_owned(),
                    args: serde_json::json!({"kota": "Jakarta"}),
                }),
            ],
            false,
        ));
        messages.push(Message::tool_result(ToolResult {
            id: format!("call-{turn}"),
            data: Some(serde_json::json!({"suhu": 31, "kondisi": "cerah"})),
            error: None,
        }));
        messages.push(Message::assistant(
            vec![AssistantPart::Text(
                "Cuaca di Jakarta hari ini cerah, suhu sekitar 31 derajat.".to_owned(),
            )],
            true,
        ));
    }
    Conversation::new(messages, vec![tool()])
}

fn collect_events(events: &[ParseEvent]) -> (usize, usize) {
    let text: usize = events
        .iter()
        .map(|event| match event {
            ParseEvent::TextDelta(text) | ParseEvent::ThinkDelta(text) => text.len(),
            _ => 0,
        })
        .sum();
    (text, events.len())
}

fn bench_render(c: &mut Criterion) {
    let tokenizer = single_merge_tokenizer();
    let renderer = Renderer::default();
    let mut group = c.benchmark_group("format_render");
    group.measurement_time(Duration::from_secs(10));

    for turns in [1_usize, 8] {
        let conversation = tool_call_conversation(turns);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{turns}_turns")),
            &conversation,
            |b, conversation| {
                b.iter(|| renderer.render(black_box(conversation), black_box(&tokenizer)))
            },
        );
    }
    group.finish();
}

fn bench_stream_parse(c: &mut Criterion) {
    let tokenizer = single_merge_tokenizer();
    let renderer = Renderer::default();
    let mut group = c.benchmark_group("format_parse");
    group.measurement_time(Duration::from_secs(10));

    for turns in [1_usize, 8] {
        let conversation = tool_call_conversation(turns);
        let tokens = renderer
            .render(&conversation, &tokenizer)
            .expect("benchmark conversation renders");
        group.throughput(Throughput::Elements(tokens.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{turns}_turns_{}tok", tokens.len())),
            &tokens,
            |b, tokens| {
                b.iter(|| {
                    let mut parser = byva_format::StreamingParser::default();
                    for token in black_box(tokens) {
                        let _ = parser.push(*token, &tokenizer);
                    }
                    collect_events(&parser.finish())
                })
            },
        );
    }
    group.finish();
}

fn bench_openai_conversion(c: &mut Criterion) {
    let tokenizer = single_merge_tokenizer();
    let renderer = Renderer::default();
    let conversation = tool_call_conversation(8);
    let tokens = renderer
        .render(&conversation, &tokenizer)
        .expect("benchmark conversation renders");
    let mut parser = byva_format::StreamingParser::default();
    let mut events = Vec::new();
    for token in &tokens {
        events.extend(parser.push(*token, &tokenizer));
    }
    events.extend(parser.finish());

    let mut group = c.benchmark_group("format_openai");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("events_to_openai_8_turns", |b| {
        b.iter(|| byva_format::events_to_openai(black_box(&events)))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_render,
    bench_stream_parse,
    bench_openai_conversion
);
criterion_main!(benches);
