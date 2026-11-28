//! Phase 6 SFT gate (trainer-level, CI-safe scale): masked loss decreases,
//! fine-tune loads base weights exactly, masked resume is bit-exact, and the
//! mask/template invariants hold end to end.

mod support;

use byva_data::sha256_hex;
use byva_tokenizer::Tokenizer;
use byva_trainer::{
    OptimizerKind, SftDataset, SftExample, Trainer, build_dataset, identity_examples,
};

use support::config;

fn sft_tokenizer() -> Tokenizer {
    Tokenizer::from_merges(1, vec![]).unwrap_or_else(|e| panic!("{e}"))
}

fn sft_model() -> byva_trainer::ModelConfig {
    let mut model = support::test_model();
    model.vocab = 384;
    model
}

fn masked_config(steps: u64) -> byva_trainer::TrainConfig {
    let mut config = config(OptimizerKind::AdamW, steps);
    config.model = sft_model();
    config.eval_every = 2;
    config.eval_rows = 2;
    config
}

fn dataset_for(examples: &[SftExample], repeat: usize) -> SftDataset {
    build_dataset(
        &sft_tokenizer(),
        examples,
        repeat,
        support::test_model().seq_len,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

fn identity_dataset() -> SftDataset {
    dataset_for(&identity_examples(), 24)
}

fn trainer_from_dataset(steps: u64, dataset: &SftDataset) -> Trainer {
    Trainer::new_masked(
        masked_config(steps),
        dataset.packed.train().clone(),
        dataset.packed.val().clone(),
        dataset.train_mask.clone(),
        dataset.val_mask.clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

fn base_trainer(path: &std::path::Path, steps: u64) -> Trainer {
    let train = support::packed_utf8().train().clone();
    let val = support::packed_utf8().val().clone();
    let mut config = config(OptimizerKind::AdamW, steps);
    config.model = sft_model();
    let mut trainer = Trainer::new(config, train, val).unwrap_or_else(|e| panic!("{e}"));
    trainer.run().unwrap_or_else(|e| panic!("{e}"));
    trainer.save(path).unwrap_or_else(|e| panic!("{e}"));
    trainer
}

#[test]
fn sft_masked_loss_decreases_on_identity_data() {
    let dataset = identity_dataset();
    let mut trainer = trainer_from_dataset(40, &dataset);
    trainer.run().unwrap_or_else(|e| panic!("{e}"));
    let first = trainer.loss_history()[0];
    let last = *trainer
        .loss_history()
        .last()
        .unwrap_or_else(|| panic!("no loss history"));
    assert!(
        last < first,
        "masked loss did not decrease: {first} -> {last}"
    );
    let eval = trainer.eval_history();
    assert!(!eval.is_empty());
    assert!(
        eval.last().unwrap_or_else(|| panic!("no eval")).loss < eval[0].loss,
        "held-out masked loss did not decrease"
    );
}

#[test]
fn fine_tune_starts_from_base_weights_exactly() {
    let base_path = support::temp_ckpt("sft-base");
    let base = base_trainer(&base_path, 3);
    let expected = base.params().to_vec();

    let dataset = dataset_for(&identity_examples(), 8);
    let mut trainer = Trainer::fine_tune(
        &base_path,
        masked_config(2),
        dataset.packed.train().clone(),
        dataset.packed.val().clone(),
        Some(dataset.train_mask.clone()),
        Some(dataset.val_mask.clone()),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(trainer.step(), 0);
    assert_eq!(trainer.params(), expected.as_slice());
    trainer.run().unwrap_or_else(|e| panic!("{e}"));
    assert_ne!(trainer.params(), expected.as_slice());
    let _ = std::fs::remove_file(&base_path);
}

#[test]
fn fine_tune_allows_shorter_seq_len() {
    let base_path = support::temp_ckpt("sft-base-short");
    let _base = base_trainer(&base_path, 2);
    let dataset = build_dataset(&sft_tokenizer(), &identity_examples(), 12, 4)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut config = masked_config(2);
    config.model.seq_len = 4;
    config.token_budget = 32;
    let mut trainer = Trainer::fine_tune(
        &base_path,
        config,
        dataset.packed.train().clone(),
        dataset.packed.val().clone(),
        Some(dataset.train_mask.clone()),
        Some(dataset.val_mask.clone()),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    trainer.run().unwrap_or_else(|e| panic!("{e}"));
    let _ = std::fs::remove_file(&base_path);
}

#[test]
fn masked_resume_is_bit_exact() {
    let dataset = identity_dataset();
    let path = support::temp_ckpt("sft-resume");
    let mut first = trainer_from_dataset(2, &dataset);
    first.save(&path).unwrap_or_else(|e| panic!("{e}"));
    first.run().unwrap_or_else(|e| panic!("{e}"));

    let mut resumed = Trainer::resume_masked(
        &path,
        dataset.packed.train().clone(),
        dataset.packed.val().clone(),
        dataset.train_mask.clone(),
        dataset.val_mask.clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    resumed.set_total_steps(4).unwrap_or_else(|e| panic!("{e}"));
    resumed.run().unwrap_or_else(|e| panic!("{e}"));

    let mut straight = trainer_from_dataset(4, &dataset);
    straight.run().unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        resumed.loss_history(),
        straight.loss_history(),
        "masked resume diverged from the uninterrupted run"
    );
    assert_eq!(resumed.params(), straight.params());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn masked_checkpoint_rejects_plain_resume() {
    let dataset = identity_dataset();
    let path = support::temp_ckpt("sft-plain-resume");
    let trainer = trainer_from_dataset(1, &dataset);
    trainer.save(&path).unwrap_or_else(|e| panic!("{e}"));
    let error = Trainer::resume(
        &path,
        dataset.packed.train().clone(),
        dataset.packed.val().clone(),
    )
    .err()
    .unwrap_or_else(|| panic!("expected rejection"));
    assert!(error.to_string().contains("resume_masked"), "{error}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn masks_reject_bad_shapes_and_values() {
    let dataset = identity_dataset();
    let mut trainer = trainer_from_dataset(2, &dataset);
    let too_short = vec![1.0f32; 10];
    let error = trainer
        .set_loss_masks(too_short, dataset.val_mask.clone())
        .err()
        .unwrap_or_else(|| panic!("expected rejection"));
    assert!(error.to_string().contains("weights for"), "{error}");
    let nan = vec![f32::NAN; dataset.train_mask.len()];
    let error = trainer
        .set_loss_masks(nan, dataset.val_mask.clone())
        .err()
        .unwrap_or_else(|| panic!("expected rejection"));
    assert!(error.to_string().contains("finite"), "{error}");
    let all_zero = vec![0.0f32; dataset.train_mask.len()];
    let error = trainer
        .set_loss_masks(all_zero, dataset.val_mask.clone())
        .err()
        .unwrap_or_else(|| panic!("expected rejection"));
    assert!(error.to_string().contains("supervised"), "{error}");
}

#[test]
fn identity_jsonl_roundtrip_through_loader() {
    use byva_trainer::{canonical_text, load_examples};
    use std::fmt::Write as _;
    let path = std::env::temp_dir().join(format!("byva-sft-gate-{}.jsonl", std::process::id()));
    let examples = identity_examples();
    let mut body = String::new();
    for example in &examples {
        let digest = sha256_hex(canonical_text(example).as_bytes());
        writeln!(
            body,
            "{{\"id\":\"{}\",\"prompt\":\"{}\",\"response\":\"{}\"}}",
            &digest[..16],
            example.prompt,
            example.response
        )
        .unwrap_or_else(|e| panic!("{e}"));
    }
    std::fs::write(&path, &body).unwrap_or_else(|e| panic!("{e}"));
    let loaded = load_examples(&path).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(loaded, examples);
    let _ = std::fs::remove_file(&path);
}
