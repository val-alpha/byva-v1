//! BYVA command-line interface.
//!
//! Phase 5 adds the `generate` subcommand: greedy decoding against a
//! `byva-weights-v1` container and a tokenizer artifact.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use byva_runtime::{InferenceModel, Weights};
use byva_tokenizer::{SpecialToken, TokenId, Tokenizer, TokenizerArtifact};
use byva_trainer::Trainer;

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
        Some("train") => train(&args[1..]),
        _ => Err(usage().to_string()),
    }
}

const fn usage() -> &'static str {
    "usage: byva-cli generate --weights <byva-weights-v1 file> --tokenizer <artifact.json> \
--prompt <text> [--max-new <count>]\n       byva-cli convert --checkpoint <training checkpoint> --out <byva-weights-v1 file>\n       byva-cli train --shards <dir with manifest.json> --tokenizer <artifact.json> --out <checkpoint> \
[--resume] [--steps <count>] [--optimizer adamw|signtrust] [--token-budget <tokens>] \
[--seq-len <n>] [--d-model <n>] [--layers <n>] [--heads <n>] [--ffn <n>] [--vocab <n>] \
[--seed <n>] [--lr <f64>] [--warmup <steps>] [--eval-every <steps>] [--log-every <steps>]"
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

fn train(args: &[String]) -> Result<(), String> {
    let mut shards_dir: Option<PathBuf> = None;
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut resume = false;
    let mut steps_opt: Option<u64> = None;
    let mut optimizer = byva_trainer::OptimizerKind::AdamW;
    let mut token_budget: Option<usize> = None;
    let mut model = byva_trainer::ModelConfig::tiny();
    let mut seed: u64 = 0;
    let mut learning_rate: Option<f64> = None;
    let mut warmup_steps: u64 = 0;
    let mut eval_every: u64 = 50;
    let mut log_every: u64 = 10;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--shards" => {
                shards_dir = Some(PathBuf::from(value_at(args, "--shards", &mut index)?));
            }
            "--tokenizer" => {
                tokenizer_path = Some(PathBuf::from(value_at(args, "--tokenizer", &mut index)?));
            }
            "--out" => {
                out_path = Some(PathBuf::from(value_at(args, "--out", &mut index)?));
            }
            "--resume" => {
                resume = true;
            }
            "--steps" => {
                let raw = value_at(args, "--steps", &mut index)?;
                steps_opt = Some(
                    raw.parse()
                        .map_err(|_| format!("--steps requires a number; {}", usage()))?,
                );
            }
            "--optimizer" => {
                let raw = value_at(args, "--optimizer", &mut index)?;
                optimizer = match raw.as_str() {
                    "adamw" => byva_trainer::OptimizerKind::AdamW,
                    "signtrust" => byva_trainer::OptimizerKind::SignTrust,
                    other => {
                        return Err(format!(
                            "--optimizer must be adamw or signtrust, found {other:?}; {}",
                            usage()
                        ));
                    }
                };
            }
            "--token-budget" => {
                let raw = value_at(args, "--token-budget", &mut index)?;
                token_budget = Some(
                    raw.parse()
                        .map_err(|_| format!("--token-budget requires a number; {}", usage()))?,
                );
            }
            "--seq-len" => {
                let raw = value_at(args, "--seq-len", &mut index)?;
                model.seq_len = raw
                    .parse()
                    .map_err(|_| format!("--seq-len requires a number; {}", usage()))?;
            }
            "--d-model" => {
                let raw = value_at(args, "--d-model", &mut index)?;
                model.d_model = raw
                    .parse()
                    .map_err(|_| format!("--d-model requires a number; {}", usage()))?;
            }
            "--layers" => {
                let raw = value_at(args, "--layers", &mut index)?;
                model.n_layers = raw
                    .parse()
                    .map_err(|_| format!("--layers requires a number; {}", usage()))?;
            }
            "--heads" => {
                let raw = value_at(args, "--heads", &mut index)?;
                model.n_heads = raw
                    .parse()
                    .map_err(|_| format!("--heads requires a number; {}", usage()))?;
            }
            "--ffn" => {
                let raw = value_at(args, "--ffn", &mut index)?;
                model.ffn = raw
                    .parse()
                    .map_err(|_| format!("--ffn requires a number; {}", usage()))?;
            }
            "--vocab" => {
                let raw = value_at(args, "--vocab", &mut index)?;
                model.vocab = raw
                    .parse()
                    .map_err(|_| format!("--vocab requires a number; {}", usage()))?;
            }
            "--seed" => {
                let raw = value_at(args, "--seed", &mut index)?;
                seed = raw
                    .parse()
                    .map_err(|_| format!("--seed requires a number; {}", usage()))?;
            }
            "--lr" => {
                let raw = value_at(args, "--lr", &mut index)?;
                learning_rate = Some(
                    raw.parse()
                        .map_err(|_| format!("--lr requires a number; {}", usage()))?,
                );
            }
            "--warmup" => {
                let raw = value_at(args, "--warmup", &mut index)?;
                warmup_steps = raw
                    .parse()
                    .map_err(|_| format!("--warmup requires a number; {}", usage()))?;
            }
            "--eval-every" => {
                let raw = value_at(args, "--eval-every", &mut index)?;
                eval_every = raw
                    .parse()
                    .map_err(|_| format!("--eval-every requires a number; {}", usage()))?;
            }
            "--log-every" => {
                let raw = value_at(args, "--log-every", &mut index)?;
                log_every = raw
                    .parse()
                    .map_err(|_| format!("--log-every requires a number; {}", usage()))?;
            }
            other => return Err(format!("unknown argument {other:?}; {}", usage())),
        }
        index += 1;
    }

    let shards_dir = shards_dir.ok_or("missing --shards")?;
    let tokenizer_path = tokenizer_path.ok_or("missing --tokenizer")?;
    let out_path = out_path.ok_or("missing --out")?;
    let token_budget = token_budget.unwrap_or_else(|| 8 * model.seq_len);
    let learning_rate = learning_rate.unwrap_or(1e-3);

    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let manifest = byva_data::Manifest::from_path(&shards_dir.join("manifest.json"))
        .map_err(|error| error.to_string())?;
    let shards =
        byva_data::load_shards(&manifest, &shards_dir).map_err(|error| error.to_string())?;
    let mut stream: Vec<u32> = Vec::new();
    for shard in &shards {
        for document in shard.documents() {
            stream.extend(
                tokenizer
                    .encode_content(document.text())
                    .into_iter()
                    .map(|token| token.get()),
            );
        }
    }

    let packed = byva_data::pack(&stream, model.seq_len).map_err(|error| error.to_string())?;
    let mut trainer = if resume {
        let mut trainer = Trainer::resume(&out_path, packed.train().clone(), packed.val().clone())
            .map_err(|error| error.to_string())?;
        if let Some(steps) = steps_opt {
            trainer
                .set_total_steps(steps)
                .map_err(|error| error.to_string())?;
        }
        trainer
    } else {
        let steps = steps_opt.ok_or("missing --steps")?;
        let config = byva_trainer::TrainConfig {
            model: model.clone(),
            optimizer,
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            clamp_min: 0.25,
            clamp_max: 4.0,
            steps,
            token_budget,
            seed,
            warmup_steps,
            floor_ratio: 0.1,
            eval_every,
            eval_rows: 200,
        };
        Trainer::new(config, packed.train().clone(), packed.val().clone())
            .map_err(|error| error.to_string())?
    };

    let total = trainer.config().steps;
    let mut trained = 0u64;
    while trainer.step() < total {
        let loss = trainer.train_step().map_err(|error| error.to_string())?;
        trained += 1;
        if log_every > 0 && trainer.step() % log_every == 0 {
            eprintln!("step {}/{} loss {:.4}", trainer.step(), total, loss);
        }
        if eval_every > 0 && trainer.step() % eval_every == 0 {
            trainer.save(&out_path).map_err(|error| error.to_string())?;
        }
    }
    trainer.save(&out_path).map_err(|error| error.to_string())?;

    println!(
        "trained {trained} step(s) in this invocation (total step {}), {} tokens/step, optimizer {}",
        trainer.step(),
        trainer.tokens_per_step(),
        match optimizer {
            byva_trainer::OptimizerKind::AdamW => "adamw",
            byva_trainer::OptimizerKind::SignTrust => "signtrust",
        }
    );
    if let Some(last) = trainer.loss_history().last() {
        println!("last training loss: {last:.4}");
    }
    if let Some(record) = trainer.eval_history().last() {
        println!(
            "last held-out loss: {:.4} (step {})",
            record.loss, record.step
        );
    }
    println!("checkpoint: {}", out_path.display());
    Ok(())
}
