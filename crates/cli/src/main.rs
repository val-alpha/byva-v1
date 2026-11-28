//! BYVA command-line interface.
//!
//! Phase 5 adds the `generate` subcommand: greedy decoding against a
//! `byva-weights-v1` container and a tokenizer artifact.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use byva_runtime::{InferenceModel, Weights};
use byva_tokenizer::{SpecialToken, TokenId, Tokenizer, TokenizerArtifact};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("generate") => generate(&args[1..]),
        Some("convert") => convert(&args[1..]),
        _ => Err(usage().to_string()),
    }
}

const fn usage() -> &'static str {
    "usage: byva-cli generate --weights <byva-weights-v1 file> --tokenizer <artifact.json> \
--prompt <text> [--max-new <count>]\n       byva-cli convert --checkpoint <training checkpoint> --out <byva-weights-v1 file>"
}

fn value_at(args: &[String], flag: &str, index: &mut usize) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value; {}", usage()))
}

fn generate(args: &[String]) -> Result<(), String> {
    let mut weights_path: Option<PathBuf> = None;
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut max_new_tokens: u32 = 64;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--weights" => {
                weights_path = Some(PathBuf::from(value_at(args, "--weights", &mut index)?));
            }
            "--tokenizer" => {
                tokenizer_path = Some(PathBuf::from(value_at(args, "--tokenizer", &mut index)?));
            }
            "--prompt" => {
                prompt = Some(value_at(args, "--prompt", &mut index)?);
            }
            "--max-new" => {
                let raw = value_at(args, "--max-new", &mut index)?;
                max_new_tokens = raw
                    .parse()
                    .map_err(|_| format!("--max-new requires a number; {}", usage()))?;
            }
            other => return Err(format!("unknown argument {other:?}; {}", usage())),
        }
        index += 1;
    }

    let weights_path = weights_path.ok_or("missing --weights")?;
    let tokenizer_path = tokenizer_path.ok_or("missing --tokenizer")?;
    let prompt = prompt.ok_or("missing --prompt")?;

    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let weights = Weights::from_path(&weights_path).map_err(|error| error.to_string())?;
    let model = InferenceModel::from_weights(weights).map_err(|error| error.to_string())?;

    let mut tokens = vec![SpecialToken::Bos.id()];
    tokens.extend(tokenizer.encode_content(&prompt));

    let mut session = model.new_session();
    let end_id = SpecialToken::End.id();
    for &token in &tokens {
        if session.step(token.get()).is_err() {
            return Err("prompt exceeds the model's max_seq_len".to_string());
        }
    }

    let mut generated = Vec::new();
    for _ in 0..max_new_tokens {
        let next = TokenId(session.greedy_token());
        if next == end_id {
            break;
        }
        if session.step(next.get()).is_err() {
            eprintln!("note: stopped at max_seq_len");
            break;
        }
        if SpecialToken::from_id(next).is_none() {
            generated.push(next);
        }
    }

    let text = tokenizer
        .decode_content(&generated)
        .map_err(|error| error.to_string())?;
    println!("{text}");
    Ok(())
}

fn convert(args: &[String]) -> Result<(), String> {
    let mut checkpoint_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--checkpoint" => {
                checkpoint_path = Some(PathBuf::from(value_at(args, "--checkpoint", &mut index)?));
            }
            "--out" => {
                out_path = Some(PathBuf::from(value_at(args, "--out", &mut index)?));
            }
            other => return Err(format!("unknown argument {other:?}; {}", usage())),
        }
        index += 1;
    }

    let checkpoint_path = checkpoint_path.ok_or("missing --checkpoint")?;
    let out_path = out_path.ok_or("missing --out")?;

    let checkpoint =
        byva_convert::Checkpoint::from_path(&checkpoint_path).map_err(|error| error.to_string())?;
    let weights =
        byva_convert::checkpoint_to_weights(&checkpoint).map_err(|error| error.to_string())?;
    let bytes = weights.to_bytes();
    std::fs::write(&out_path, &bytes)
        .map_err(|error| format!("failed to write {}: {error}", out_path.display()))?;

    let reloaded = Weights::from_path(&out_path).map_err(|error| error.to_string())?;
    if reloaded.to_bytes() != bytes {
        return Err("reload verification failed: bytes differ".to_string());
    }

    let config = reloaded.config();
    println!(
        "converted {} (step {}) -> {}",
        checkpoint.format,
        checkpoint.step,
        out_path.display()
    );
    println!(
        "config: vocab_size={} d_model={} n_layers={} n_heads={} ffn_hidden={} max_seq_len={} tie_embeddings={}",
        config.vocab_size,
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.ffn_hidden,
        config.max_seq_len,
        config.tie_embeddings
    );
    println!(
        "tensors: {} ({} parameters)",
        reloaded.tensor_names().count(),
        config
            .parameter_count()
            .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer, String> {
    let raw =
        std::fs::read_to_string(path).map_err(|error| format!("tokenizer unreadable: {error}"))?;
    let artifact: TokenizerArtifact = serde_json::from_str(&raw)
        .map_err(|error| format!("tokenizer artifact invalid: {error}"))?;
    Tokenizer::from_artifact(artifact)
        .map_err(|error: byva_tokenizer::TokenizerError| error.to_string())
}
