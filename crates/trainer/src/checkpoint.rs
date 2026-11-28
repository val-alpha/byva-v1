//! The `byva-train-rust-checkpoint-v1` format (D-0022): magic + version
//! header, JSON metadata, little-endian f32 blobs, sha256 trailer.
//!
//! Unknown formats and corrupt payloads are rejected with typed errors
//! (D-0014 discipline); checkpoints stay out of Git (D-0007).

use serde::{Deserialize, Serialize};

use byva_data::sha256_hex;

use crate::config::TrainConfig;
use crate::error::TrainerError;

/// Format identifier recorded in every checkpoint.
pub const FORMAT_ID: &str = "byva-train-rust-checkpoint-v1";

const MAGIC: [u8; 8] = *b"BYVATRCK";
const VERSION: u32 = 1;
const DIGEST_LEN: usize = 64;
const HEADER_LEN: usize = 20;

/// Description of one tensor blob region.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDesc {
    /// Blob name (parameter name or optimizer state role).
    pub name: String,
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Element count.
    pub len: usize,
}

/// One held-out evaluation record.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvalRecord {
    /// Step at which the evaluation ran.
    pub step: u64,
    /// Mean held-out cross-entropy loss.
    pub loss: f32,
    /// Held-out loss over Indonesian windows (mixture runs only).
    #[serde(default)]
    pub loss_id: Option<f32>,
    /// Held-out loss over English windows (mixture runs only).
    #[serde(default)]
    pub loss_en: Option<f32>,
}

/// One recorded K5 mixture adjustment (eval-driven, bounded by the band).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MixtureRecord {
    /// Step at which the adjustment was recorded.
    pub step: u64,
    /// The Indonesian window share applied from the next step on.
    pub id_share: f64,
    /// Held-out loss that drove the adjustment (Indonesian).
    pub id_loss: f32,
    /// Held-out loss that drove the adjustment (English).
    pub en_loss: f32,
}

/// Resumable K5 mixture state (None when the mixture is off).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MixtureState {
    /// Cycling cursor into the Indonesian train-window pool.
    pub id_cursor: usize,
    /// Cycling cursor into the English train-window pool.
    pub en_cursor: usize,
    /// The Indonesian window share in effect.
    pub id_share: f64,
}

/// Optimizer state description in a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimStateDesc {
    /// Optimizer kind ("adamw" or "signtrust").
    pub kind: String,
    /// State blobs in canonical order.
    pub blobs: Vec<TensorDesc>,
}

/// JSON metadata block of a checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointMeta {
    /// Must be `byva-train-rust-checkpoint-v1`.
    pub format: String,
    /// Full run configuration (reproduces the run on resume).
    pub config: TrainConfig,
    /// Number of completed training steps.
    pub step: u64,
    /// Dataloader position: absolute row cursor into the train sequences.
    pub batch_cursor: usize,
    /// Per-step training loss history (mean-reduced).
    pub loss_history: Vec<f32>,
    /// Held-out evaluation history.
    pub eval_history: Vec<EvalRecord>,
    /// Every recorded K5 mixture adjustment (empty when the mixture is off).
    #[serde(default)]
    pub mixture_history: Vec<MixtureRecord>,
    /// Resumable K5 mixture state (None when the mixture is off).
    #[serde(default)]
    pub mixture_state: Option<MixtureState>,
    /// Parameter blob descriptions in canonical order.
    pub params: Vec<TensorDesc>,
    /// Optimizer state description.
    pub optimizer: OptimStateDesc,
}

/// A loaded checkpoint: metadata plus restored buffers.
#[derive(Debug, Clone)]
pub struct LoadedCheckpoint {
    /// Parsed metadata.
    pub meta: CheckpointMeta,
    /// Flat parameter buffer.
    pub params: Vec<f32>,
    /// Optimizer state blobs (order matches `meta.optimizer.blobs`).
    pub optim_blobs: Vec<Vec<f32>>,
}

/// Serializes and writes a checkpoint atomically (write + rename).
pub fn save(
    path: &std::path::Path,
    meta: &CheckpointMeta,
    params: &[f32],
    optim_blobs: &[Vec<f32>],
) -> Result<(), TrainerError> {
    let invalid = |reason: String| TrainerError::CheckpointCorrupt {
        path: path.display().to_string(),
        reason,
    };
    let json = serde_json::to_vec(meta)
        .map_err(|error| invalid(format!("metadata serialization failed: {error}")))?;
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&MAGIC);
    buffer.extend_from_slice(&VERSION.to_le_bytes());
    buffer.extend_from_slice(&(json.len() as u64).to_le_bytes());
    buffer.extend_from_slice(&json);
    for value in params {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    for blob in optim_blobs {
        for value in blob {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
    }
    let digest = sha256_hex(&buffer);
    buffer.extend_from_slice(digest.as_bytes());
    let temp = path.with_extension("tmp");
    std::fs::write(&temp, &buffer).map_err(|error| TrainerError::Io {
        path: temp.display().to_string(),
        reason: error.to_string(),
    })?;
    std::fs::rename(&temp, path).map_err(|error| TrainerError::Io {
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    Ok(())
}

/// Reads, verifies (magic, version, sha256, structure), and parses a
/// checkpoint.
pub fn load(path: &std::path::Path) -> Result<LoadedCheckpoint, TrainerError> {
    let path_display = path.display().to_string();
    let bytes = std::fs::read(path).map_err(|error| TrainerError::Io {
        path: path_display.clone(),
        reason: error.to_string(),
    })?;
    let corrupt = |reason: String| TrainerError::CheckpointCorrupt {
        path: path_display.clone(),
        reason,
    };
    if bytes.len() < HEADER_LEN + DIGEST_LEN {
        return Err(corrupt(format!(
            "file is {} bytes, shorter than the smallest possible checkpoint",
            bytes.len()
        )));
    }
    let (payload, digest_bytes) = bytes.split_at(bytes.len() - DIGEST_LEN);
    let recorded = std::str::from_utf8(digest_bytes)
        .map_err(|_| corrupt("sha256 trailer is not ASCII hex".to_string()))?;
    if sha256_hex(payload) != recorded {
        return Err(corrupt("sha256 integrity check failed".to_string()));
    }
    if payload[..8] != MAGIC {
        return Err(TrainerError::UnknownCheckpointFormat {
            found: String::from_utf8_lossy(&payload[..8]).to_string(),
        });
    }
    let version = u32::from_le_bytes(
        payload[8..12]
            .try_into()
            .map_err(|_| corrupt("header is truncated".to_string()))?,
    );
    if version != VERSION {
        return Err(TrainerError::UnknownCheckpointFormat {
            found: format!("version {version}"),
        });
    }
    let meta_len = u64::from_le_bytes(
        payload[12..20]
            .try_into()
            .map_err(|_| corrupt("header is truncated".to_string()))?,
    ) as usize;
    if payload.len() < HEADER_LEN + meta_len {
        return Err(corrupt(format!(
            "metadata length {} exceeds the payload",
            meta_len
        )));
    }
    let meta: CheckpointMeta = serde_json::from_slice(&payload[HEADER_LEN..HEADER_LEN + meta_len])
        .map_err(|error| corrupt(format!("metadata is not valid JSON: {error}")))?;
    if meta.format != FORMAT_ID {
        return Err(TrainerError::UnknownCheckpointFormat { found: meta.format });
    }
    let mut cursor = HEADER_LEN + meta_len;
    let read_blob = |cursor: &mut usize, len: usize| -> Result<Vec<f32>, TrainerError> {
        let byte_len = len
            .checked_mul(4)
            .ok_or_else(|| corrupt("blob length overflows".to_string()))?;
        let end = (*cursor)
            .checked_add(byte_len)
            .ok_or_else(|| corrupt("blob range overflows".to_string()))?;
        if end > payload.len() {
            return Err(corrupt(format!(
                "blob needs {} bytes but only {} remain",
                byte_len,
                payload.len() - *cursor
            )));
        }
        let blob = payload[*cursor..end]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        *cursor = end;
        Ok(blob)
    };
    let mut params = Vec::new();
    for desc in &meta.params {
        let blob = read_blob(&mut cursor, desc.len)?;
        if blob.len() != desc.len {
            return Err(corrupt(format!(
                "parameter {} declared {} elements",
                desc.name, desc.len
            )));
        }
        params.extend_from_slice(&blob);
    }
    let mut optim_blobs = Vec::with_capacity(meta.optimizer.blobs.len());
    for desc in &meta.optimizer.blobs {
        let blob = read_blob(&mut cursor, desc.len)?;
        optim_blobs.push(blob);
    }
    if cursor != payload.len() {
        return Err(corrupt(format!(
            "payload has {} trailing bytes after the declared blobs",
            payload.len() - cursor
        )));
    }
    Ok(LoadedCheckpoint {
        meta,
        params,
        optim_blobs,
    })
}
