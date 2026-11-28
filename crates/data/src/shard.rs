//! Shard reading with per-shard integrity validation (D-0012).

use std::path::Path;

use serde::Deserialize;

use crate::error::DataError;
use crate::manifest::ShardEntry;
use crate::sha256::sha256_hex;

/// One canonical document record `{"id", "text"}` from a shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardDocument {
    /// Sixteen lowercase hex characters, the first 16 of sha256(text).
    id: String,
    /// The normalized document text.
    text: String,
}

impl ShardDocument {
    /// The document id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The document text.
    pub fn text(&self) -> &str {
        &self.text
    }
}

#[derive(Deserialize)]
struct RawDocument {
    id: String,
    text: String,
}

/// A validated shard payload with its parsed documents.
#[derive(Debug, Clone)]
pub struct Shard {
    path: String,
    name: String,
    documents: Vec<ShardDocument>,
}

impl Shard {
    /// Loads and validates one shard file from `dir` against its manifest entry.
    pub fn load(dir: &Path, entry: &ShardEntry) -> Result<Self, DataError> {
        let path = dir.join(&entry.name);
        let payload = std::fs::read(&path).map_err(|error| DataError::ManifestMismatch {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        Self::from_payload(entry, &path.display().to_string(), &payload)
    }

    /// Validates a shard payload held in memory against its manifest entry.
    pub fn from_payload(entry: &ShardEntry, path: &str, payload: &[u8]) -> Result<Self, DataError> {
        let found = payload.len() as u64;
        if found != entry.bytes {
            return Err(DataError::ManifestMismatch {
                path: path.to_string(),
                reason: format!("expected {} bytes, found {}", entry.bytes, found),
            });
        }
        let digest = sha256_hex(payload);
        if digest != entry.sha256 {
            return Err(DataError::ManifestMismatch {
                path: path.to_string(),
                reason: format!("expected sha256 {}, found {}", entry.sha256, digest),
            });
        }
        let documents = Self::parse_documents(entry.name.as_str(), path, payload)?;
        Ok(Self {
            path: path.to_string(),
            name: entry.name.clone(),
            documents,
        })
    }

    fn parse_documents(
        shard_name: &str,
        path: &str,
        payload: &[u8],
    ) -> Result<Vec<ShardDocument>, DataError> {
        let text = std::str::from_utf8(payload).map_err(|_| DataError::ManifestMismatch {
            path: path.to_string(),
            reason: "payload is not valid UTF-8".to_string(),
        })?;
        if !text.ends_with('\n') {
            return Err(DataError::ManifestMismatch {
                path: path.to_string(),
                reason: "payload does not end with a newline".to_string(),
            });
        }
        let mut documents = Vec::new();
        let mut previous_id: Option<String> = None;
        for (index, line) in text.lines().enumerate() {
            let line_number = index + 1;
            if line.is_empty() {
                return Err(DataError::InvalidShard {
                    path: path.to_string(),
                    line: line_number,
                    reason: "empty line inside canonical shard payload".to_string(),
                });
            }
            let document = Self::parse_document(shard_name, path, line_number, line)?;
            if let Some(previous) = &previous_id {
                if document.id.as_str() <= previous.as_str() {
                    return Err(DataError::InvalidShard {
                        path: path.to_string(),
                        line: line_number,
                        reason: format!(
                            "document ids must be strictly increasing; {} follows {}",
                            document.id, previous
                        ),
                    });
                }
            }
            previous_id = Some(document.id.clone());
            documents.push(document);
        }
        Ok(documents)
    }

    fn parse_document(
        shard_name: &str,
        path: &str,
        line_number: usize,
        line: &str,
    ) -> Result<ShardDocument, DataError> {
        let rejected = |reason: String| DataError::InvalidShard {
            path: path.to_string(),
            line: line_number,
            reason,
        };
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| rejected(format!("invalid JSON: {error}")))?;
        let serde_json::Value::Object(ref map) = value else {
            return Err(rejected("record must be a JSON object".to_string()));
        };
        if map.len() != 2 || !map.contains_key("id") || !map.contains_key("text") {
            return Err(rejected(format!(
                "shard {shard_name} record must contain exactly the keys \"id\" and \"text\""
            )));
        }
        let raw: RawDocument = serde_json::from_value(value)
            .map_err(|error| rejected(format!("record is malformed: {error}")))?;
        if raw.id.len() != 16 || !raw.id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(rejected(format!(
                "document id {:?} is not 16 hex characters",
                raw.id
            )));
        }
        if !raw
            .id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        {
            return Err(rejected(format!(
                "document id {:?} must be lowercase hex",
                raw.id
            )));
        }
        let digest = sha256_hex(raw.text.as_bytes());
        if !digest.starts_with(&raw.id) {
            return Err(rejected(format!(
                "document id {} does not match sha256 of its text ({})",
                raw.id, digest
            )));
        }
        Ok(ShardDocument {
            id: raw.id,
            text: raw.text,
        })
    }

    /// The shard file name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The path the shard was loaded from, for reports.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The parsed documents, in id order.
    pub fn documents(&self) -> &[ShardDocument] {
        &self.documents
    }
}

/// Loads and validates every shard listed in the manifest from one directory.
///
/// Document ids must also be strictly increasing across shards, because the
/// pipeline sorts documents globally by id before sharding (D-0012).
pub fn load_shards(manifest: &crate::Manifest, dir: &Path) -> Result<Vec<Shard>, DataError> {
    let mut shards = Vec::with_capacity(manifest.shards().len());
    let mut previous_id: Option<String> = None;
    for entry in manifest.shards() {
        let shard = Shard::load(dir, entry)?;
        if let Some(first) = shard.documents.first() {
            if let Some(previous) = &previous_id {
                if first.id.as_str() <= previous.as_str() {
                    return Err(DataError::ManifestMismatch {
                        path: shard.path,
                        reason: format!(
                            "document ids must be strictly increasing across shards; {} follows {}",
                            first.id, previous
                        ),
                    });
                }
            }
            if let Some(last) = shard.documents.last() {
                previous_id = Some(last.id.clone());
            }
        }
        shards.push(shard);
    }
    Ok(shards)
}
