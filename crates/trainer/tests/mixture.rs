//! K5 id:en mixture integration tests (R6): pool sampling, bounded
//! eval-driven adaptation, natural-order equivalence when off, exact
//! resume with mixture state, and validation rejections.

mod support;

use byva_data::Lang;
use byva_trainer::{OptimizerKind, TrainConfig, Trainer};

use support::{cleanup, temp_ckpt, test_model};

fn mixture_config(steps: u64, eval_every: u64, eval_rows: usize) -> TrainConfig {
    let config = TrainConfig {
        model: test_model(),
        optimizer: OptimizerKind::AdamW,
        learning_rate: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        clamp_min: 0.25,
        clamp_max: 4.0,
        steps,
        token_budget: 32,
        seed: 42,
        warmup_steps: 0,
        floor_ratio: 1.0,
        eval_every,
        eval_rows,
        mixture_target_id: Some(0.5),
        mixture_band: 0.1,
    };
    config.validate().unwrap_or_else(|e| panic!("{e}"));
    config
}

fn plain_config(steps: u64, eval_every: u64, eval_rows: usize) -> TrainConfig {
    let mut config = mixture_config(steps, eval_every, eval_rows);
    config.mixture_target_id = None;
    config
}

/// First 30 train windows Indonesian, the rest English; val split 3 id + 4 en.
fn tags(train_count: usize, val_count: usize) -> (Vec<Lang>, Vec<Lang>) {
    let train = (0..train_count)
        .map(|row| if row < 30 { Lang::Id } else { Lang::En })
        .collect();
    let val = (0..val_count)
        .map(|row| if row < 3 { Lang::Id } else { Lang::En })
        .collect();
    (train, val)
}

#[test]
fn mixture_adaptation_is_recorded_and_bounded() {
    let packed = support::packed();
    let mut trainer = Trainer::new(
        mixture_config(4, 1, 4),
        packed.train().clone(),
        packed.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let (train_tags, val_tags) = tags(packed.train().count(), packed.val().count());
    trainer
        .set_language_tags(train_tags, val_tags)
        .unwrap_or_else(|e| panic!("{e}"));
    trainer.run().unwrap_or_else(|e| panic!("{e}"));

    let history = trainer.mixture_history();
    assert_eq!(history.len(), 4, "one adjustment per eval step");
    for record in history {
        assert!(
            (0.4..=0.6).contains(&record.id_share),
            "id_share {} outside the band",
            record.id_share
        );
    }
    for record in trainer.eval_history() {
        assert!(record.loss_id.is_some() && record.loss_en.is_some());
    }
}

#[test]
fn mixture_higher_id_loss_pushes_share_toward_id() {
    let packed = support::packed();
    let mut trainer = Trainer::new(
        mixture_config(6, 2, 3),
        packed.train().clone(),
        packed.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let (train_tags, val_tags) = tags(packed.train().count(), packed.val().count());
    trainer
        .set_language_tags(train_tags, val_tags)
        .unwrap_or_else(|e| panic!("{e}"));
    trainer.run().unwrap_or_else(|e| panic!("{e}"));

    let last = *trainer
        .mixture_history()
        .last()
        .unwrap_or_else(|| panic!("no adjustment"));
    if last.id_loss > last.en_loss {
        assert!(last.id_share >= 0.5, "share must move toward id");
    } else {
        assert!(last.id_share <= 0.5, "share must move toward en");
    }
}

#[test]
fn mixture_off_with_tags_matches_the_natural_order_run() {
    let packed = support::packed();
    let mut natural = Trainer::new(
        plain_config(5, 0, 4),
        packed.train().clone(),
        packed.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    natural.run().unwrap_or_else(|e| panic!("{e}"));

    let mut tagged = Trainer::new(
        plain_config(5, 0, 4),
        packed.train().clone(),
        packed.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let (train_tags, val_tags) = tags(packed.train().count(), packed.val().count());
    tagged
        .set_language_tags(train_tags, val_tags)
        .unwrap_or_else(|e| panic!("{e}"));
    tagged.run().unwrap_or_else(|e| panic!("{e}"));

    assert_eq!(natural.loss_history(), tagged.loss_history());
    assert!(tagged.mixture_history().is_empty());
}

#[test]
fn mixture_resume_is_bit_exact() {
    let packed = support::packed();
    let full_config = mixture_config(4, 0, 4);
    let mut full = Trainer::new(
        full_config.clone(),
        packed.train().clone(),
        packed.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let (train_tags, val_tags) = tags(packed.train().count(), packed.val().count());
    full.set_language_tags(train_tags.clone(), val_tags.clone())
        .unwrap_or_else(|e| panic!("{e}"));
    full.run().unwrap_or_else(|e| panic!("{e}"));

    let mut first = Trainer::new(
        mixture_config(4, 0, 4),
        packed.train().clone(),
        packed.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    first
        .set_language_tags(train_tags.clone(), val_tags.clone())
        .unwrap_or_else(|e| panic!("{e}"));
    for _ in 0..2 {
        first.train_step().unwrap_or_else(|e| panic!("{e}"));
    }
    let path = temp_ckpt("mixture-resume");
    first.save(&path).unwrap_or_else(|e| panic!("{e}"));

    let mut second = Trainer::resume(&path, packed.train().clone(), packed.val().clone())
        .unwrap_or_else(|e| panic!("{e}"));
    second
        .set_language_tags(train_tags, val_tags)
        .unwrap_or_else(|e| panic!("{e}"));
    for _ in 0..2 {
        second.train_step().unwrap_or_else(|e| panic!("{e}"));
    }
    cleanup(&path);

    assert_eq!(full.loss_history(), second.loss_history());
    assert_eq!(full.id_share(), second.id_share());
}

#[test]
fn mixture_without_tags_is_rejected() {
    let packed = support::packed();
    let mut trainer = Trainer::new(
        mixture_config(2, 0, 4),
        packed.train().clone(),
        packed.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let error = trainer.train_step().unwrap_err();
    assert!(error.to_string().contains("language tags"), "{error}");
}

#[test]
fn language_tags_with_wrong_lengths_are_rejected() {
    let packed = support::packed();
    let mut trainer = Trainer::new(
        plain_config(2, 0, 4),
        packed.train().clone(),
        packed.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let error = trainer
        .set_language_tags(vec![Lang::Id; 3], vec![Lang::En; packed.val().count()])
        .unwrap_err();
    assert!(error.to_string().contains("language tags"), "{error}");
}

#[test]
fn invalid_mixture_configs_are_rejected() {
    let mut config = plain_config(2, 0, 4);
    config.mixture_target_id = Some(1.0);
    assert!(config.validate().is_err());

    let mut config = plain_config(2, 0, 4);
    config.mixture_target_id = Some(0.5);
    config.mixture_band = 0.5;
    assert!(config.validate().is_err());

    let mut config = plain_config(2, 0, 4);
    config.mixture_target_id = Some(0.5);
    config.mixture_band = 0.1;
    assert!(config.validate().is_ok());
}
