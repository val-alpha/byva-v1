//! Checkpoint format behavior: integrity checks, unknown-format rejection,
//! and resume validation.

mod support;

use byva_data::sha256_hex;
use byva_trainer::{OptimizerKind, Trainer, TrainerError};
use support::{cleanup, config, packed, temp_ckpt};

fn saved_checkpoint(name: &str) -> Vec<u8> {
    let data = packed();
    let mut trainer = Trainer::new(
        config(OptimizerKind::AdamW, 4),
        data.train().clone(),
        data.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    trainer.train_step().unwrap_or_else(|e| panic!("{e}"));
    let path = temp_ckpt(name);
    trainer.save(&path).unwrap_or_else(|e| panic!("{e}"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{e}"));
    cleanup(&path);
    bytes
}

#[test]
fn round_trip_restores_the_same_parameters() {
    let data = packed();
    let mut trainer = Trainer::new(
        config(OptimizerKind::AdamW, 4),
        data.train().clone(),
        data.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    trainer.train_step().unwrap_or_else(|e| panic!("{e}"));
    let path = temp_ckpt("round-trip");
    trainer.save(&path).unwrap_or_else(|e| panic!("{e}"));
    let restored = Trainer::resume(&path, data.train().clone(), data.val().clone())
        .unwrap_or_else(|e| panic!("{e}"));
    cleanup(&path);
    assert_eq!(restored.step(), 1);
    assert_eq!(restored.params().len(), trainer.params().len());
    for (a, b) in trainer.params().iter().zip(restored.params()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn wrong_magic_with_valid_digest_is_unknown_format() {
    let bytes = saved_checkpoint("wrong-magic");
    let split = bytes.len() - 64;
    let mut tampered = bytes[..split].to_vec();
    tampered[0..8].copy_from_slice(b"NOTBYVA1");
    let digest = sha256_hex(&tampered);
    tampered.extend_from_slice(digest.as_bytes());

    let path = temp_ckpt("wrong-magic");
    std::fs::write(&path, &tampered).unwrap_or_else(|e| panic!("{e}"));
    let data = packed();
    let error = Trainer::resume(&path, data.train().clone(), data.val().clone())
        .err()
        .unwrap_or_else(|| panic!("resume must reject a foreign magic"));
    cleanup(&path);
    match error {
        TrainerError::UnknownCheckpointFormat { found } => assert_eq!(found, "NOTBYVA1"),
        other => panic!("expected UnknownCheckpointFormat, found: {other}"),
    }
}

#[test]
fn garbage_payload_fails_integrity_before_format() {
    let bytes = saved_checkpoint("garbage");
    let split = bytes.len() - 64;
    let mut tampered = bytes[..split].to_vec();
    tampered[3] ^= 0xFF;
    let digest = sha256_hex(&tampered);
    let mut garbage = tampered;
    garbage.extend_from_slice(b"0000");
    garbage.extend_from_slice(digest.as_bytes());

    let path = temp_ckpt("garbage");
    std::fs::write(&path, &garbage).unwrap_or_else(|e| panic!("{e}"));
    let data = packed();
    let error = Trainer::resume(&path, data.train().clone(), data.val().clone())
        .err()
        .unwrap_or_else(|| panic!("resume must reject a corrupt payload"));
    cleanup(&path);
    assert!(matches!(error, TrainerError::CheckpointCorrupt { .. }));
}

#[test]
fn truncated_file_is_corrupt() {
    let path = temp_ckpt("truncated");
    std::fs::write(&path, b"BYVATRCK").unwrap_or_else(|e| panic!("{e}"));
    let data = packed();
    let error = Trainer::resume(&path, data.train().clone(), data.val().clone())
        .err()
        .unwrap_or_else(|| panic!("resume must reject a truncated file"));
    cleanup(&path);
    assert!(matches!(error, TrainerError::CheckpointCorrupt { .. }));
}

#[test]
fn resume_rejects_mismatched_sequences() {
    let path = temp_ckpt("mismatch");
    let data = packed();
    let mut trainer = Trainer::new(
        config(OptimizerKind::AdamW, 4),
        data.train().clone(),
        data.val().clone(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    trainer.train_step().unwrap_or_else(|e| panic!("{e}"));
    trainer.save(&path).unwrap_or_else(|e| panic!("{e}"));

    let other = byva_data::pack(&support::token_stream(600), 16).unwrap_or_else(|e| panic!("{e}"));
    let error = Trainer::resume(&path, other.train().clone(), other.val().clone())
        .err()
        .unwrap_or_else(|| panic!("resume must reject sequences with the wrong seq_len"));
    cleanup(&path);
    assert!(matches!(error, TrainerError::ConfigMismatch { .. }));
}
