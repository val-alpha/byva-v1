//! Golden-fixture tests for the D-0012 shard contract (R2 gate).
//!
//! The fixtures under `tests/fixtures/smoke/` were produced by the real
//! Python pipeline (`training/build_shards.py`) from a fixed synthetic
//! sample; the Rust reader must accept them byte-for-byte and reject
//! tampered copies.

use std::path::{Path, PathBuf};

use byva_data::{DataError, Manifest, Shard, ShardEntry, load_shards, sha256_hex};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/smoke")
}

fn manifest() -> Manifest {
    Manifest::from_path(&fixture_dir().join("manifest.json")).unwrap_or_else(|e| panic!("{e}"))
}

fn shard_payload(name: &str) -> Vec<u8> {
    std::fs::read(fixture_dir().join(name)).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn golden_manifest_parses_and_validates() {
    let manifest = manifest();
    assert_eq!(manifest.shards().len(), 3);
    assert_eq!(manifest.config().docs_per_shard, 4);
    assert_eq!(manifest.config().min_chars, 40);
    assert_eq!(manifest.config().max_chars, 100_000);
    assert_eq!(manifest.sources().len(), 1);
    assert_eq!(manifest.sources()[0].name, "sample.jsonl");
    let stats = manifest.stats().unwrap_or_else(|| panic!("stats missing"));
    assert_eq!(stats.input, 13);
    assert_eq!(stats.kept, 11);
    assert_eq!(stats.dropped_duplicates, 1);
    assert_eq!(stats.dropped_short, 1);
    assert_eq!(stats.dropped_long, 0);
}

#[test]
fn golden_shards_validate_against_manifest() {
    let manifest = manifest();
    let shards = load_shards(&manifest, &fixture_dir()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(shards.len(), 3);
    let total: usize = shards.iter().map(|shard| shard.documents().len()).sum();
    assert_eq!(total, 11);
}

#[test]
fn golden_documents_match_recorded_hashes_and_order() {
    let manifest = manifest();
    let shards = load_shards(&manifest, &fixture_dir()).unwrap_or_else(|e| panic!("{e}"));
    let mut previous: Option<String> = None;
    for shard in &shards {
        for document in shard.documents() {
            let digest = sha256_hex(document.text().as_bytes());
            assert!(digest.starts_with(document.id()));
            assert_eq!(document.id().len(), 16);
            if let Some(previous) = &previous {
                assert!(document.id() > previous.as_str());
            }
            previous = Some(document.id().to_string());
        }
    }
}

#[test]
fn our_sha256_matches_the_recorded_python_hashes() {
    let manifest = manifest();
    for entry in manifest.shards() {
        let payload = shard_payload(&entry.name);
        assert_eq!(payload.len() as u64, entry.bytes);
        assert_eq!(sha256_hex(&payload), entry.sha256);
    }
}

#[test]
fn byte_count_mismatch_is_rejected() {
    let manifest = manifest();
    let entry = &manifest.shards()[0];
    let mut payload = shard_payload(&entry.name);
    payload.pop();
    let error = Shard::from_payload(entry, "tampered", &payload).unwrap_err();
    assert!(matches!(error, DataError::ManifestMismatch { .. }));
    assert!(error.to_string().contains("expected 846 bytes"), "{error}");
}

#[test]
fn sha256_mismatch_is_rejected() {
    let manifest = manifest();
    let entry = &manifest.shards()[0];
    let mut payload = shard_payload(&entry.name);
    let last = payload.len() - 1;
    payload[last] ^= 0x01;
    let error = Shard::from_payload(entry, "tampered", &payload).unwrap_err();
    assert!(matches!(error, DataError::ManifestMismatch { .. }));
    assert!(error.to_string().contains("sha256"), "{error}");
}

#[test]
fn non_canonical_payload_is_rejected() {
    let manifest = manifest();
    let entry = &manifest.shards()[0];
    let mut payload = shard_payload(&entry.name);
    payload.pop();
    payload.push(b' ');
    let tampered = ShardEntry {
        name: entry.name.clone(),
        sha256: sha256_hex(&payload),
        bytes: payload.len() as u64,
    };
    let error = Shard::from_payload(&tampered, "tampered", &payload).unwrap_err();
    assert!(matches!(error, DataError::ManifestMismatch { .. }));
    assert!(error.to_string().contains("newline"), "{error}");
}

#[test]
fn document_id_hash_mismatch_is_rejected() {
    let payload = b"{\"id\":\"0000000000000000\",\"text\":\"hello world\"}\n";
    let entry = ShardEntry {
        name: "fake.jsonl".to_string(),
        sha256: sha256_hex(payload),
        bytes: payload.len() as u64,
    };
    let error = Shard::from_payload(&entry, "fake", payload).unwrap_err();
    assert!(matches!(error, DataError::InvalidShard { .. }));
    assert!(
        error.to_string().contains("does not match sha256"),
        "{error}"
    );
}

#[test]
fn unsorted_document_ids_are_rejected() {
    let first = b"{\"id\":\"57240a108e4d5ff4\",\"text\":\"second document body here\"}\n";
    let second = b"{\"id\":\"4e8c2676686d6d4f\",\"text\":\"first document body here\"}\n";
    let mut payload = first.to_vec();
    payload.extend_from_slice(second);
    let entry = ShardEntry {
        name: "fake.jsonl".to_string(),
        sha256: sha256_hex(&payload),
        bytes: payload.len() as u64,
    };
    let error = Shard::from_payload(&entry, "fake", &payload).unwrap_err();
    assert!(matches!(error, DataError::InvalidShard { .. }));
    assert!(error.to_string().contains("strictly increasing"), "{error}");
}

#[test]
fn wrong_schema_version_is_rejected() {
    let original = serde_json::to_vec(&manifest_stats_raw()).unwrap_or_else(|e| panic!("{e}"));
    let mut value: serde_json::Value =
        serde_json::from_slice(&original).unwrap_or_else(|e| panic!("{e}"));
    value["schema_version"] = serde_json::Value::from(2);
    let bytes = serde_json::to_vec(&value).unwrap_or_else(|e| panic!("{e}"));
    let error = Manifest::parse(&bytes, Path::new("tampered.json")).unwrap_err();
    assert!(matches!(error, DataError::ManifestInvalid { .. }));
    assert!(error.to_string().contains("schema_version"), "{error}");
}

#[test]
fn empty_shard_list_is_rejected() {
    let original = serde_json::to_vec(&manifest_stats_raw()).unwrap_or_else(|e| panic!("{e}"));
    let mut value: serde_json::Value =
        serde_json::from_slice(&original).unwrap_or_else(|e| panic!("{e}"));
    value["shards"] = serde_json::Value::Array(Vec::new());
    let bytes = serde_json::to_vec(&value).unwrap_or_else(|e| panic!("{e}"));
    let error = Manifest::parse(&bytes, Path::new("tampered.json")).unwrap_err();
    assert!(matches!(error, DataError::ManifestInvalid { .. }));
    assert!(error.to_string().contains("no shards"), "{error}");
}

#[test]
fn malformed_manifest_json_is_a_parse_error() {
    let error = Manifest::parse(b"{not json", Path::new("broken.json")).unwrap_err();
    assert!(matches!(error, DataError::ManifestParse { .. }));
}

fn manifest_stats_raw() -> serde_json::Value {
    serde_json::from_slice(
        &std::fs::read(fixture_dir().join("manifest.json")).unwrap_or_else(|e| panic!("{e}")),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}
