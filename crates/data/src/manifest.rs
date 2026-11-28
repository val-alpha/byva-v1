//! Manifest parsing and validation for D-0012 schema version 1.

use std::path::Path;

use serde::Deserialize;

use crate::error::DataError;

pub(crate) const SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// Filter configuration recorded by the sharding pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ShardConfig {
    /// Documents per shard used when the manifest was built.
    pub docs_per_shard: u64,
    /// Minimum normalized character count admitted.
    pub min_chars: u64,
    /// Maximum normalized character count admitted.
    pub max_chars: u64,
}

/// Provenance record for one raw input file.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SourceEntry {
    /// File name (never a path, so manifests stay portable).
    pub name: String,
    /// SHA-256 of the source file.
    pub sha256: String,
}

/// Integrity record for one shard file.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ShardEntry {
    /// File name (never a path).
    pub name: String,
    /// SHA-256 of the shard payload.
    pub sha256: String,
    /// Shard payload size in bytes.
    pub bytes: u64,
}

/// Optional pipeline statistics block.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ShardStats {
    /// Documents read from the sources.
    pub input: u64,
    /// Documents dropped for being too short.
    pub dropped_short: u64,
    /// Documents dropped for being too long.
    pub dropped_long: u64,
    /// Duplicate documents dropped by content hash.
    pub dropped_duplicates: u64,
    /// Documents kept.
    pub kept: u64,
}

/// Parsed D-0012 schema-version-1 manifest.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Manifest {
    schema_version: u32,
    sort_key: String,
    normalization: String,
    config: ShardConfig,
    sources: Vec<SourceEntry>,
    shards: Vec<ShardEntry>,
    #[serde(default)]
    stats: Option<ShardStats>,
}

impl Manifest {
    /// Parses and validates a manifest from raw bytes.
    pub fn parse(bytes: &[u8], path: &Path) -> Result<Self, DataError> {
        let path_display = path.display().to_string();
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|error| DataError::ManifestParse {
                path: path_display.clone(),
                reason: error.to_string(),
            })?;
        manifest.validate(&path_display)?;
        Ok(manifest)
    }

    /// Loads, parses, and validates a manifest from disk.
    pub fn from_path(path: &Path) -> Result<Self, DataError> {
        let bytes = std::fs::read(path).map_err(|error| DataError::ManifestParse {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        Self::parse(&bytes, path)
    }

    fn validate(&self, path: &str) -> Result<(), DataError> {
        let invalid = |reason: String| DataError::ManifestInvalid {
            path: path.to_string(),
            reason,
        };
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(invalid(format!(
                "unsupported schema_version {} (expected {SUPPORTED_SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        if self.sort_key != "id" {
            return Err(invalid(format!(
                "unsupported sort_key {:?} (expected \"id\")",
                self.sort_key
            )));
        }
        if self.normalization != "NFC" {
            return Err(invalid(format!(
                "unsupported normalization {:?} (expected \"NFC\")",
                self.normalization
            )));
        }
        if self.config.docs_per_shard == 0 {
            return Err(invalid(
                "config.docs_per_shard must be positive".to_string(),
            ));
        }
        if self.config.min_chars == 0 {
            return Err(invalid("config.min_chars must be positive".to_string()));
        }
        if self.config.max_chars <= self.config.min_chars {
            return Err(invalid(format!(
                "config.max_chars {} must exceed config.min_chars {}",
                self.config.max_chars, self.config.min_chars
            )));
        }
        if self.shards.is_empty() {
            return Err(invalid("manifest lists no shards".to_string()));
        }
        let mut previous: Option<&str> = None;
        for shard in &self.shards {
            if shard.name.is_empty() || shard.name.contains('/') || shard.name.contains('\\') {
                return Err(invalid(format!(
                    "shard name {:?} is not a plain file name",
                    shard.name
                )));
            }
            if !is_sha256_hex(&shard.sha256) {
                return Err(invalid(format!(
                    "shard {} has a malformed sha256 field",
                    shard.name
                )));
            }
            if shard.bytes == 0 {
                return Err(invalid(format!("shard {} records zero bytes", shard.name)));
            }
            if let Some(previous) = previous {
                if shard.name.as_str() <= previous {
                    return Err(invalid(format!(
                        "shard names must be strictly increasing; {} follows {}",
                        shard.name, previous
                    )));
                }
            }
            previous = Some(shard.name.as_str());
        }
        for source in &self.sources {
            if source.name.is_empty() || source.name.contains('/') || source.name.contains('\\') {
                return Err(invalid(format!(
                    "source name {:?} is not a plain file name",
                    source.name
                )));
            }
            if !is_sha256_hex(&source.sha256) {
                return Err(invalid(format!(
                    "source {} has a malformed sha256 field",
                    source.name
                )));
            }
        }
        Ok(())
    }

    /// The validated shard list, in manifest order.
    pub fn shards(&self) -> &[ShardEntry] {
        &self.shards
    }

    /// The recorded provenance sources.
    pub fn sources(&self) -> &[SourceEntry] {
        &self.sources
    }

    /// The filter configuration the manifest was built with.
    pub fn config(&self) -> &ShardConfig {
        &self.config
    }

    /// The optional stats block.
    pub fn stats(&self) -> Option<&ShardStats> {
        self.stats.as_ref()
    }
}

pub(crate) fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
