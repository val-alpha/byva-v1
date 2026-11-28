//! Training-loop behavior: toy-task convergence for both optimizer options,
//! config/data validation, and the LR schedule.

mod support;

use byva_data::pack;
use byva_trainer::{ModelConfig, OptimizerKind, Trainer, TrainerError};
use support::{config, packed, token_stream};

fn repeating_tokens(length: usize, period: &[u32]) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(length);
    for index in 0..length {
        tokens.push(period[index % period.len()]);
    }
    tokens
}

fn run_toy(optimizer: OptimizerKind, steps: u64) -> Vec<f32> {
    let tokens = repeating_tokens(600, &[5, 9, 13, 21, 34, 2, 40, 55, 7]);
    let data = pack(&tokens, 8).unwrap_or_else(|e| panic!("{e}"));
    let mut trainer = Trainer::new(
        config(optimizer, steps),
        data.train().clone(),
        data.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    trainer.run().unwrap_or_else(|e| panic!("{e}"));
    trainer.loss_history().to_vec()
}

#[test]
fn adamw_reduces_toy_task_loss() {
    let losses = run_toy(OptimizerKind::AdamW, 60);
    let first = losses[0];
    let last = *losses
        .last()
        .unwrap_or_else(|| panic!("empty loss history"));
    assert!(last.is_finite());
    assert!(
        last < first,
        "AdamW did not reduce the toy loss: first {first}, last {last}"
    );
}

#[test]
fn signtrust_reduces_toy_task_loss() {
    let losses = run_toy(OptimizerKind::SignTrust, 60);
    let first = losses[0];
    let last = *losses
        .last()
        .unwrap_or_else(|| panic!("empty loss history"));
    assert!(last.is_finite());
    assert!(
        last < first,
        "SignTrust did not reduce the toy loss: first {first}, last {last}"
    );
}

#[test]
fn eval_schedule_records_held_out_loss() {
    let data = packed();
    let mut trainer = Trainer::new(
        config(OptimizerKind::AdamW, 6),
        data.train().clone(),
        data.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    trainer.run().unwrap_or_else(|e| panic!("{e}"));
    let evals = trainer.eval_history();
    assert_eq!(evals.len(), 3);
    assert_eq!(evals[0].step, 2);
    assert_eq!(evals[1].step, 4);
    assert_eq!(evals[2].step, 6);
    for record in evals {
        assert!(record.loss.is_finite());
    }
}

#[test]
fn constant_tokens_per_step() {
    let data = packed();
    let trainer = Trainer::new(
        config(OptimizerKind::AdamW, 6),
        data.train().clone(),
        data.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(trainer.tokens_per_step(), 32);
}

#[test]
fn invalid_configs_are_rejected() {
    let data = packed();

    let mut broken = config(OptimizerKind::AdamW, 0);
    broken.steps = 0;
    assert!(matches!(
        Trainer::new(broken, data.train().clone(), data.val().clone()),
        Err(TrainerError::ConfigInvalid { .. })
    ));

    let mut broken = config(OptimizerKind::AdamW, 6);
    broken.token_budget = 30;
    assert!(matches!(
        Trainer::new(broken, data.train().clone(), data.val().clone()),
        Err(TrainerError::ConfigInvalid { .. })
    ));

    let mut broken = config(OptimizerKind::AdamW, 6);
    broken.learning_rate = 0.0;
    assert!(matches!(
        Trainer::new(broken, data.train().clone(), data.val().clone()),
        Err(TrainerError::ConfigInvalid { .. })
    ));

    let mut broken = config(OptimizerKind::AdamW, 6);
    broken.model = ModelConfig {
        d_model: 17,
        ..support::test_model()
    };
    assert!(matches!(
        Trainer::new(broken, data.train().clone(), data.val().clone()),
        Err(TrainerError::ConfigInvalid { .. })
    ));
}

#[test]
fn undersized_or_mismatched_data_is_rejected() {
    let data = packed();

    let mut hungry = config(OptimizerKind::AdamW, 6);
    hungry.token_budget = 640;
    assert!(matches!(
        Trainer::new(hungry, data.train().clone(), data.val().clone()),
        Err(TrainerError::DataInvalid { .. })
    ));

    let other = pack(&token_stream(600), 16).unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(
        Trainer::new(
            config(OptimizerKind::AdamW, 6),
            other.train().clone(),
            data.val().clone()
        ),
        Err(TrainerError::DataInvalid { .. })
    ));
}

#[test]
fn learning_rate_schedule_warms_up_then_decays_to_floor() {
    let mut schedule = config(OptimizerKind::AdamW, 100);
    schedule.learning_rate = 1.0;
    schedule.warmup_steps = 10;
    schedule.floor_ratio = 0.1;

    assert!((schedule.learning_rate_at(1) - 0.1).abs() < 1e-9);
    assert!((schedule.learning_rate_at(10) - 1.0).abs() < 1e-9);
    assert!(schedule.learning_rate_at(11) < 1.0);
    assert!(schedule.learning_rate_at(55) > 0.1);
    assert!(schedule.learning_rate_at(55) < 1.0);
    assert!((schedule.learning_rate_at(100) - 0.1).abs() < 1e-9);

    let no_warmup = config(OptimizerKind::AdamW, 100);
    assert!(no_warmup.learning_rate_at(1) < no_warmup.learning_rate);
    assert!(no_warmup.learning_rate_at(1) > no_warmup.learning_rate * no_warmup.floor_ratio);
    assert!(
        (no_warmup.learning_rate_at(100) - no_warmup.learning_rate * no_warmup.floor_ratio).abs()
            < 1e-9
    );
}
