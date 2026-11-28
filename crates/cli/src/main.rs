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
        Some("sft") => sft(&args[1..]),
        _ => Err(usage().to_string()),
    }
}

const fn usage() -> &'static str {
    "usage: byva-cli generate --weights <byva-weights-v1 file> --tokenizer <artifact.json> \
--prompt <text> [--max-new <count>] [--chat]\n       byva-cli convert --checkpoint <training checkpoint> --out <byva-weights-v1 file>\n       byva-cli train --shards <dir with manifest.json> --tokenizer <artifact.json> --out <checkpoint> \
[--resume] [--steps <count>] [--optimizer adamw|signtrust] [--token-budget <tokens>] \
[--seq-len <n>] [--d-model <n>] [--layers <n>] [--heads <n>] [--ffn <n>] [--vocab <n>] \
[--seed <n>] [--lr <f64>] [--warmup <steps>] [--eval-every <steps>] [--log-every <steps>] \
[--floor-ratio <f64>] [--weight-decay <f64>] [--save-init <checkpoint>] [--explain-step] \
[--lang-share-id <f64>] [--lang-band <f64>]\n       byva-cli sft --base <rust checkpoint> --tokenizer <artifact.json> --out <checkpoint> \
(--sft-data <jsonl> | --identity) [--repeat <count>] [--resume] [--steps <count>] [--lr <f64>] \
[--seq-len <n>] [--token-budget <tokens>] [--seed <n>] [--weight-decay <f64>] \
[--eval-every <steps>] [--log-every <steps>]"
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
    let mut chat = false;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--weights" => {
                weights_path = Some(PathBuf::from(value_at(args, "--weights", &mut index)?));
            }
            "--tokenizer" => {
                tokenizer_path = Some(PathBuf::from(value_at(args, "--tokenizer", &mut index)?));
            }
            "--chat" => {
                chat = true;
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
    if chat {
        tokens.push(SpecialToken::User.id());
        tokens.extend(tokenizer.encode_content(&prompt));
        tokens.push(SpecialToken::Assistant.id());
    } else {
        tokens.extend(tokenizer.encode_content(&prompt));
    }

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
    let mut floor_ratio: f64 = 0.1;
    let mut weight_decay: f64 = 0.0;
    let mut save_init_path: Option<PathBuf> = None;
    let mut explain = false;
    let mut lang_share_id: Option<f64> = None;
    let mut lang_band: f64 = 0.1;

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
            "--explain-step" => {
                explain = true;
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
            "--floor-ratio" => {
                let raw = value_at(args, "--floor-ratio", &mut index)?;
                floor_ratio = raw
                    .parse()
                    .map_err(|_| format!("--floor-ratio requires a number; {}", usage()))?;
            }
            "--weight-decay" => {
                let raw = value_at(args, "--weight-decay", &mut index)?;
                weight_decay = raw
                    .parse()
                    .map_err(|_| format!("--weight-decay requires a number; {}", usage()))?;
            }
            "--save-init" => {
                save_init_path = Some(PathBuf::from(value_at(args, "--save-init", &mut index)?));
            }
            "--lang-share-id" => {
                let raw = value_at(args, "--lang-share-id", &mut index)?;
                lang_share_id = Some(
                    raw.parse()
                        .map_err(|_| format!("--lang-share-id requires a number; {}", usage()))?,
                );
            }
            "--lang-band" => {
                let raw = value_at(args, "--lang-band", &mut index)?;
                lang_band = raw
                    .parse()
                    .map_err(|_| format!("--lang-band requires a number; {}", usage()))?;
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

    if explain {
        for line in byva_trainer::explain_step(&model).map_err(|error| error.to_string())? {
            println!("{line}");
        }
        return Ok(());
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
    let mut doc_langs: Vec<byva_data::DocLang> = Vec::new();
    for shard in &shards {
        for document in shard.documents() {
            let encoded: Vec<u32> = tokenizer
                .encode_content(document.text())
                .into_iter()
                .map(|token| token.get())
                .collect();
            doc_langs.push(byva_data::DocLang {
                start: stream.len(),
                len: encoded.len(),
                lang: byva_data::detect_language(document.text()),
            });
            stream.extend(encoded);
        }
    }

    let (packed, tags) = if lang_share_id.is_some() {
        let tagged = byva_data::pack_with_languages(&stream, &doc_langs, model.seq_len)
            .map_err(|error| error.to_string())?;
        (tagged.packed, Some((tagged.train_langs, tagged.val_langs)))
    } else {
        (
            byva_data::pack(&stream, model.seq_len).map_err(|error| error.to_string())?,
            None,
        )
    };
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
            weight_decay,
            clamp_min: 0.25,
            clamp_max: 4.0,
            steps,
            token_budget,
            seed,
            warmup_steps,
            floor_ratio,
            eval_every,
            eval_rows: 200,
            mixture_target_id: lang_share_id,
            mixture_band: lang_band,
        };
        Trainer::new(config, packed.train().clone(), packed.val().clone())
            .map_err(|error| error.to_string())?
    };
    if let Some((train_langs, val_langs)) = tags {
        trainer
            .set_language_tags(train_langs, val_langs)
            .map_err(|error| error.to_string())?;
    }
    if let Some(init_path) = save_init_path {
        trainer
            .save(&init_path)
            .map_err(|error| error.to_string())?;
        eprintln!("init checkpoint: {}", init_path.display());
    }

    let total = trainer.config().steps;
    let mut trained = 0u64;
    let mut step_clock = std::time::Instant::now();
    while trainer.step() < total {
        let loss = trainer.train_step().map_err(|error| error.to_string())?;
        trained += 1;
        if log_every > 0 && trainer.step() % log_every == 0 {
            eprintln!(
                "step {}/{} loss {:.4} ({:.3?}/step)",
                trainer.step(),
                total,
                loss,
                step_clock.elapsed()
            );
        }
        step_clock = std::time::Instant::now();
        if eval_every > 0 && trainer.step() % eval_every == 0 {
            if let Some(record) = trainer.mixture_history().last() {
                eprintln!(
                    "mixture: step {} id_share {:.3} (id {:.4} en {:.4})",
                    record.step, record.id_share, record.id_loss, record.en_loss
                );
            }
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

fn sft(args: &[String]) -> Result<(), String> {
    let mut base_path: Option<PathBuf> = None;
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut sft_data: Option<PathBuf> = None;
    let mut identity = false;
    let mut repeat: usize = 16;
    let mut resume = false;
    let mut steps_opt: Option<u64> = None;
    let mut learning_rate: Option<f64> = None;
    let mut seq_len_opt: Option<usize> = None;
    let mut token_budget_opt: Option<usize> = None;
    let mut seed: u64 = 0;
    let mut weight_decay: f64 = 0.0;
    let mut warmup_steps: u64 = 0;
    let mut eval_every: u64 = 20;
    let mut log_every: u64 = 10;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--base" => {
                base_path = Some(PathBuf::from(value_at(args, "--base", &mut index)?));
            }
            "--tokenizer" => {
                tokenizer_path = Some(PathBuf::from(value_at(args, "--tokenizer", &mut index)?));
            }
            "--out" => {
                out_path = Some(PathBuf::from(value_at(args, "--out", &mut index)?));
            }
            "--sft-data" => {
                sft_data = Some(PathBuf::from(value_at(args, "--sft-data", &mut index)?));
            }
            "--identity" => {
                identity = true;
            }
            "--repeat" => {
                let raw = value_at(args, "--repeat", &mut index)?;
                repeat = raw
                    .parse()
                    .map_err(|_| format!("--repeat requires a number; {}", usage()))?;
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
            "--lr" => {
                let raw = value_at(args, "--lr", &mut index)?;
                learning_rate = Some(
                    raw.parse()
                        .map_err(|_| format!("--lr requires a number; {}", usage()))?,
                );
            }
            "--seq-len" => {
                let raw = value_at(args, "--seq-len", &mut index)?;
                seq_len_opt = Some(
                    raw.parse()
                        .map_err(|_| format!("--seq-len requires a number; {}", usage()))?,
                );
            }
            "--token-budget" => {
                let raw = value_at(args, "--token-budget", &mut index)?;
                token_budget_opt = Some(
                    raw.parse()
                        .map_err(|_| format!("--token-budget requires a number; {}", usage()))?,
                );
            }
            "--seed" => {
                let raw = value_at(args, "--seed", &mut index)?;
                seed = raw
                    .parse()
                    .map_err(|_| format!("--seed requires a number; {}", usage()))?;
            }
            "--weight-decay" => {
                let raw = value_at(args, "--weight-decay", &mut index)?;
                weight_decay = raw
                    .parse()
                    .map_err(|_| format!("--weight-decay requires a number; {}", usage()))?;
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

    let tokenizer_path = tokenizer_path.ok_or("missing --tokenizer")?;
    let out_path = out_path.ok_or("missing --out")?;
    let data_source = match (sft_data, identity) {
        (Some(_), true) => {
            return Err("--sft-data and --identity are mutually exclusive".to_string());
        }
        (None, false) => return Err("missing --sft-data or --identity".to_string()),
        (Some(path), false) => {
            byva_trainer::load_examples(&path).map_err(|error| error.to_string())?
        }
        (None, true) => byva_trainer::identity_examples(),
    };

    let dataset_source = if resume {
        out_path.clone()
    } else {
        base_path.clone().ok_or("missing --base")?
    };
    let base_model = byva_trainer::load(&dataset_source)
        .map_err(|error| error.to_string())?
        .meta
        .config
        .model;
    let seq_len = seq_len_opt.unwrap_or(base_model.seq_len);
    if seq_len > base_model.seq_len {
        return Err(format!(
            "--seq-len {seq_len} exceeds the base checkpoint's seq_len {}",
            base_model.seq_len
        ));
    }
    let token_budget = token_budget_opt.unwrap_or_else(|| 8 * seq_len);
    let learning_rate = learning_rate.unwrap_or(1e-3);

    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let dataset = byva_trainer::build_dataset(&tokenizer, &data_source, repeat, seq_len)
        .map_err(|error| error.to_string())?;

    let mut trainer = if resume {
        let mut trainer = Trainer::resume_masked(
            &out_path,
            dataset.packed.train().clone(),
            dataset.packed.val().clone(),
            dataset.train_mask.clone(),
            dataset.val_mask.clone(),
        )
        .map_err(|error| error.to_string())?;
        if let Some(steps) = steps_opt {
            trainer
                .set_total_steps(steps)
                .map_err(|error| error.to_string())?;
        }
        trainer
    } else {
        let base_path = base_path.ok_or("missing --base")?;
        let config = byva_trainer::TrainConfig {
            model: byva_trainer::ModelConfig {
                vocab: base_model.vocab,
                d_model: base_model.d_model,
                n_layers: base_model.n_layers,
                n_heads: base_model.n_heads,
                ffn: base_model.ffn,
                seq_len,
            },
            optimizer: byva_trainer::OptimizerKind::AdamW,
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            clamp_min: 0.25,
            clamp_max: 4.0,
            steps: steps_opt.unwrap_or(200),
            token_budget,
            seed,
            warmup_steps,
            floor_ratio: 0.1,
            eval_every,
            eval_rows: 64,
            mixture_target_id: None,
            mixture_band: 0.1,
        };
        Trainer::fine_tune(
            &base_path,
            config,
            dataset.packed.train().clone(),
            dataset.packed.val().clone(),
            Some(dataset.train_mask.clone()),
            Some(dataset.val_mask.clone()),
        )
        .map_err(|error| error.to_string())?
    };

    let total = trainer.config().steps;
    let mut trained = 0u64;
    let mut step_clock = std::time::Instant::now();
    while trainer.step() < total {
        let loss = trainer.train_step().map_err(|error| error.to_string())?;
        trained += 1;
        if log_every > 0 && trainer.step() % log_every == 0 {
            eprintln!(
                "sft step {}/{} loss {:.4} ({:.3?}/step)",
                trainer.step(),
                total,
                loss,
                step_clock.elapsed()
            );
        }
        step_clock = std::time::Instant::now();
        if eval_every > 0 && trainer.step() % eval_every == 0 {
            if let Some(record) = trainer.eval_history().last() {
                eprintln!("sft eval: step {} held-out {:.4}", record.step, record.loss);
            }
            trainer.save(&out_path).map_err(|error| error.to_string())?;
        }
    }
    trainer.save(&out_path).map_err(|error| error.to_string())?;

    println!(
        "sft: {trained} step(s) in this invocation (total step {}), {} token(s)/step, {} example(s) x{repeat}, masked loss on response tokens only",
        trainer.step(),
        trainer.tokens_per_step(),
        data_source.len()
    );
    if let Some(last) = trainer.loss_history().last() {
        println!("last sft loss: {last:.4}");
    }
    if let Some(record) = trainer.eval_history().last() {
        println!(
            "last held-out sft loss: {:.4} (step {})",
            record.loss, record.step
        );
    }
    println!("checkpoint: {}", out_path.display());
    Ok(())
}
