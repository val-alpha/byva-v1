//! R3 gate: a run resumed from a mid-run checkpoint must continue exactly
//! (bit-for-bit f32) as the uninterrupted run, for both D-0022 optimizer
//! options.

mod support;

use byva_trainer::{OptimizerKind, Trainer};
use support::{cleanup, config, packed, temp_ckpt};

fn run_full_then_split(optimizer: OptimizerKind, total: u64, split_at: u64, name: &str) {
    let data = packed();
    let train = data.train().clone();
    let val = data.val().clone();

    let mut full = Trainer::new(config(optimizer, total), train.clone(), val.clone())
        .unwrap_or_else(|e| panic!("{e}"));
    full.run().unwrap_or_else(|e| panic!("{e}"));

    let path = temp_ckpt(name);
    let mut partial = Trainer::new(config(optimizer, total), train.clone(), val.clone())
        .unwrap_or_else(|e| panic!("{e}"));
    for _ in 0..split_at {
        partial.train_step().unwrap_or_else(|e| panic!("{e}"));
    }
    partial.save(&path).unwrap_or_else(|e| panic!("{e}"));

    let mut resumed = Trainer::resume(&path, train, val).unwrap_or_else(|e| panic!("{e}"));
    resumed.run().unwrap_or_else(|e| panic!("{e}"));
    cleanup(&path);

    assert_eq!(full.step(), total);
    assert_eq!(resumed.step(), full.step());
    let losses_a = full.loss_history();
    let losses_b = resumed.loss_history();
    assert_eq!(losses_a.len(), losses_b.len());
    for (a, b) in losses_a.iter().zip(losses_b) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "loss diverged at step {}",
            losses_a
                .iter()
                .position(|x| x.to_bits() == a.to_bits())
                .unwrap_or_default()
                + 1
        );
    }
    assert_eq!(full.eval_history(), resumed.eval_history());
    assert_eq!(
        full.params().len(),
        resumed.params().len(),
        "parameter buffers differ in length"
    );
    for (index, (a, b)) in full.params().iter().zip(resumed.params()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "parameter {index} diverged after resume"
        );
    }
}

#[test]
fn adamw_resume_is_bit_exact() {
    run_full_then_split(OptimizerKind::AdamW, 6, 2, "adamw-resume");
}

#[test]
fn signtrust_resume_is_bit_exact() {
    run_full_then_split(OptimizerKind::SignTrust, 4, 2, "signtrust-resume");
}

#[test]
fn resume_twice_is_still_bit_exact() {
    let data = packed();
    let train = data.train().clone();
    let val = data.val().clone();

    let mut full = Trainer::new(config(OptimizerKind::AdamW, 6), train.clone(), val.clone())
        .unwrap_or_else(|e| panic!("{e}"));
    full.run().unwrap_or_else(|e| panic!("{e}"));

    let first = temp_ckpt("resume-twice-1");
    let second = temp_ckpt("resume-twice-2");
    let mut partial = Trainer::new(config(OptimizerKind::AdamW, 6), train.clone(), val.clone())
        .unwrap_or_else(|e| panic!("{e}"));
    for _ in 0..2 {
        partial.train_step().unwrap_or_else(|e| panic!("{e}"));
    }
    partial.save(&first).unwrap_or_else(|e| panic!("{e}"));
    let mut middle =
        Trainer::resume(&first, train.clone(), val.clone()).unwrap_or_else(|e| panic!("{e}"));
    for _ in 0..2 {
        middle.train_step().unwrap_or_else(|e| panic!("{e}"));
    }
    middle.save(&second).unwrap_or_else(|e| panic!("{e}"));
    let mut resumed = Trainer::resume(&second, train, val).unwrap_or_else(|e| panic!("{e}"));
    resumed.run().unwrap_or_else(|e| panic!("{e}"));
    cleanup(&first);
    cleanup(&second);

    assert_eq!(full.loss_history().len(), resumed.loss_history().len());
    for (a, b) in full.loss_history().iter().zip(resumed.loss_history()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in full.params().iter().zip(resumed.params()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn resumed_run_can_be_extended_within_warmup_and_stays_bit_exact() {
    let data = packed();
    let train = data.train().clone();
    let val = data.val().clone();

    let mut full = Trainer::new(config(OptimizerKind::AdamW, 6), train.clone(), val.clone())
        .unwrap_or_else(|e| panic!("{e}"));
    full.run().unwrap_or_else(|e| panic!("{e}"));

    let path = temp_ckpt("extend");
    let mut partial = Trainer::new(config(OptimizerKind::AdamW, 2), train.clone(), val.clone())
        .unwrap_or_else(|e| panic!("{e}"));
    partial.run().unwrap_or_else(|e| panic!("{e}"));
    partial.save(&path).unwrap_or_else(|e| panic!("{e}"));
    let mut extended = Trainer::resume(&path, train, val).unwrap_or_else(|e| panic!("{e}"));
    extended
        .set_total_steps(6)
        .unwrap_or_else(|e| panic!("{e}"));
    extended.run().unwrap_or_else(|e| panic!("{e}"));
    cleanup(&path);

    for (a, b) in full.loss_history().iter().zip(extended.loss_history()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in full.params().iter().zip(extended.params()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn total_steps_cannot_be_set_below_completed_step() {
    let data = packed();
    let path = temp_ckpt("shrink");
    let mut partial = Trainer::new(
        config(OptimizerKind::AdamW, 6),
        data.train().clone(),
        data.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    for _ in 0..3 {
        partial.train_step().unwrap_or_else(|e| panic!("{e}"));
    }
    partial.save(&path).unwrap_or_else(|e| panic!("{e}"));
    let mut resumed = Trainer::resume(&path, data.train().clone(), data.val().clone())
        .unwrap_or_else(|e| panic!("{e}"));
    cleanup(&path);
    let error = resumed
        .set_total_steps(2)
        .err()
        .unwrap_or_else(|| panic!("set_total_steps must reject a smaller target"));
    assert!(matches!(
        error,
        byva_trainer::TrainerError::ConfigInvalid { .. }
    ));
}
