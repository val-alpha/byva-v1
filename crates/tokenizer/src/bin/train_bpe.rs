//! CLI for the scaled deterministic BPE trainer (D-0017/D-0021).
//!
//! Reads JSONL documents (`{"text": ...}` lines, NFC-normalized like
//! `training/train_bpe.py`) and writes the same canonical artifact JSON:
//!
//! ```text
//!     cargo run --release -p byva-tokenizer --bin train_bpe -- \
//!         --input corpus-1.jsonl --input corpus-2.jsonl \
//!         --target-vocab 32768 --version 1 \
//!         --output byva-tokenizer-32k.json [--stats stats.json]
//! ```

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use byva_tokenizer::trainer::{artifact_file_json, train_bpe_from_jsonl};
use byva_tokenizer::{MAX_VOCAB_SIZE, MIN_VOCAB_SIZE, Tokenizer};
use unicode_normalization::UnicodeNormalization;

struct Args {
    inputs: Vec<PathBuf>,
    target_vocab: u32,
    version: u32,
    output: PathBuf,
    stats: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut inputs = Vec::new();
    let mut target_vocab = 32_768u32;
    let mut version = 1u32;
    let mut output: Option<PathBuf> = None;
    let mut stats: Option<PathBuf> = None;

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--input" => {
                let value = iter.next().ok_or("--input requires a file path")?;
                inputs.push(PathBuf::from(value));
            }
            "--target-vocab" => {
                let value = iter.next().ok_or("--target-vocab requires a number")?;
                target_vocab = value
                    .parse()
                    .map_err(|_| "--target-vocab must be a number")?;
            }
            "--version" => {
                let value = iter.next().ok_or("--version requires a number")?;
                version = value.parse().map_err(|_| "--version must be a number")?;
            }
            "--output" => {
                let value = iter.next().ok_or("--output requires a file path")?;
                output = Some(PathBuf::from(value));
            }
            "--stats" => {
                let value = iter.next().ok_or("--stats requires a file path")?;
                stats = Some(PathBuf::from(value));
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }

    if inputs.is_empty() {
        return Err("at least one --input JSONL file is required".to_owned());
    }
    if !(MIN_VOCAB_SIZE..=MAX_VOCAB_SIZE).contains(&target_vocab) {
        return Err(format!(
            "--target-vocab must be within {MIN_VOCAB_SIZE}..={MAX_VOCAB_SIZE}"
        ));
    }
    let output = output.ok_or("--output is required")?;
    Ok(Args {
        inputs,
        target_vocab,
        version,
        output,
        stats,
    })
}

#[derive(serde::Serialize)]
struct Stats {
    bytes: usize,
    bytes_per_token: f64,
    documents: usize,
    merges: usize,
    tokens: usize,
    vocab_size: u32,
}

/// Counts content tokens by streaming the corpus a second time, so token
/// statistics never hold the corpus in memory.
fn count_tokens(paths: &[&Path], tokenizer: &Tokenizer) -> Result<usize, String> {
    let mut total_tokens = 0usize;
    for path in paths {
        let file = std::fs::File::open(path)
            .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        loop {
            line.clear();
            let read = reader
                .read_line(&mut line)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            if read == 0 {
                break;
            }
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(&line)
                .map_err(|error| format!("{}: invalid JSON: {error}", path.display()))?;
            if let Some(text) = value.get("text").and_then(serde_json::Value::as_str) {
                let normalized: String = text.nfc().collect();
                total_tokens += tokenizer.encode_content(&normalized).len();
            }
        }
    }
    Ok(total_tokens)
}

fn write_output(path: &Path, contents: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let mut file = std::fs::File::create(path)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    file.write_all(contents.as_bytes())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let paths: Vec<&Path> = args.inputs.iter().map(PathBuf::as_path).collect();
    let (artifact, summary) = train_bpe_from_jsonl(&paths, args.target_vocab, args.version)
        .map_err(|error| error.to_string())?;
    let json = artifact_file_json(&artifact).map_err(|error| error.to_string())?;
    write_output(&args.output, &json)?;

    let tokenizer =
        Tokenizer::from_artifact(artifact.clone()).map_err(|error| error.to_string())?;
    let tokens = count_tokens(&paths, &tokenizer)?;
    let bytes_per_token = if tokens == 0 {
        0.0
    } else {
        (summary.bytes as f64 / tokens as f64 * 10_000.0).round() / 10_000.0
    };
    let stats = Stats {
        bytes: summary.bytes,
        bytes_per_token,
        documents: summary.documents,
        merges: artifact.merges.len(),
        tokens,
        vocab_size: artifact.vocab_size,
    };
    let stats_json = serde_json::to_string_pretty(&stats)
        .map_err(|error| format!("stats serialization failed: {error}"))?;
    if let Some(stats_path) = &args.stats {
        write_output(stats_path, &format!("{stats_json}\n"))?;
    }
    println!("{stats_json}");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("train_bpe: {message}");
            ExitCode::FAILURE
        }
    }
}
