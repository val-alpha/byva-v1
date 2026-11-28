//! Fixed packing of token streams into seq_len+1 windows with the reference
//! 90/10 split (`training/train.py::make_sequences` semantics, mirrored
//! exactly so R4 parity compares like with like).

use crate::error::DataError;
use crate::lang::{DocLang, Lang, window_langs};

/// One split's packed windows, stored flat: row `i` is
/// `x[i*seq_len..(i+1)*seq_len]` with its targets at the same offset in `y`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequences {
    seq_len: usize,
    x: Vec<u32>,
    y: Vec<u32>,
}

impl Sequences {
    /// Packs prepared windows (each of length `seq_len + 1`) into the flat
    /// `x`/`y` representation. Also used by the Phase 6 SFT packing.
    pub fn from_windows(windows: &[Vec<u32>], seq_len: usize) -> Self {
        let count = windows.len();
        let mut x = Vec::with_capacity(count * seq_len);
        let mut y = Vec::with_capacity(count * seq_len);
        for window in windows {
            x.extend_from_slice(&window[..seq_len]);
            y.extend_from_slice(&window[1..]);
        }
        Self { seq_len, x, y }
    }

    /// Window length used for inputs.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Number of packed sequences.
    pub fn count(&self) -> usize {
        self.x.len() / self.seq_len.max(1)
    }

    /// Flat input tokens, `count * seq_len` entries.
    pub fn x(&self) -> &[u32] {
        &self.x
    }

    /// Flat target tokens, `count * seq_len` entries, offset by one from `x`.
    pub fn y(&self) -> &[u32] {
        &self.y
    }
}

/// A packed token stream: deterministic 90/10 train/val split of windows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packed {
    train: Sequences,
    val: Sequences,
}

impl Packed {
    /// Packs prepared windows (each of length `seq_len + 1`) with the
    /// canonical 90/10 split. Also used by the Phase 6 SFT packing.
    pub fn from_windows(windows: &[Vec<u32>], seq_len: usize) -> Self {
        let split = (windows.len() / 10) * 9;
        Self {
            train: Sequences::from_windows(&windows[..split], seq_len),
            val: Sequences::from_windows(&windows[split..], seq_len),
        }
    }

    /// The training split (`(windows.len() / 10) * 9` leading windows).
    pub fn train(&self) -> &Sequences {
        &self.train
    }

    /// The held-out split (the trailing windows).
    pub fn val(&self) -> &Sequences {
        &self.val
    }
}

/// Packs one long token stream into non-overlapping `seq_len + 1` windows,
/// drops the trailing partial window, and splits 90/10 exactly like the
/// PyTorch reference.
///
/// The split is `split = (window_count / 10) * 9`; with fewer than ten
/// windows the training side can be empty, matching the reference.
pub fn pack(tokens: &[u32], seq_len: usize) -> Result<Packed, DataError> {
    let windows = windows_of(tokens, seq_len)?;
    let split = (windows.len() / 10) * 9;
    Ok(Packed {
        train: Sequences::from_windows(&windows[..split], seq_len),
        val: Sequences::from_windows(&windows[split..], seq_len),
    })
}

/// A packed dataset with one language tag per window (K5 id:en mixture).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaggedPacked {
    /// Same 90/10 split as `pack`.
    pub packed: Packed,
    /// Window languages of the training split, in window order.
    pub train_langs: Vec<Lang>,
    /// Window languages of the held-out split, in window order.
    pub val_langs: Vec<Lang>,
}

/// Packs one long token stream exactly like `pack` and additionally tags
/// every window with the majority language of the documents covering its
/// tokens (K5). Window order, window contents, and the split boundary are
/// identical to `pack`.
pub fn pack_with_languages(
    tokens: &[u32],
    docs: &[DocLang],
    seq_len: usize,
) -> Result<TaggedPacked, DataError> {
    let packed = pack(tokens, seq_len)?;
    let (langs, split) = window_langs(tokens.len(), docs, seq_len);
    if langs.len() < packed.train().count() + packed.val().count() {
        return Err(DataError::PackingInvalid {
            reason: format!(
                "document language ranges cover {} windows but packing produced {}",
                langs.len(),
                packed.train().count() + packed.val().count()
            ),
        });
    }
    Ok(TaggedPacked {
        train_langs: langs[..split].to_vec(),
        val_langs: langs[split..].to_vec(),
        packed,
    })
}

fn windows_of(tokens: &[u32], seq_len: usize) -> Result<Vec<Vec<u32>>, DataError> {
    if seq_len < 2 {
        return Err(DataError::PackingInvalid {
            reason: format!("seq_len must be at least 2, got {seq_len}"),
        });
    }
    let window_len = seq_len + 1;
    let usable = tokens.len() - (tokens.len() % window_len);
    if usable < 2 * window_len {
        return Err(DataError::PackingInvalid {
            reason: format!(
                "not enough tokens for a held-out split: {} usable tokens, need at least {}",
                usable,
                2 * window_len
            ),
        });
    }
    let mut windows = Vec::with_capacity(usable / window_len);
    for start in (0..usable).step_by(window_len) {
        windows.push(tokens[start..start + window_len].to_vec());
    }
    Ok(windows)
}

/// Streaming window packer for the trainer loop (R3): feed token slices in,
/// collect full `seq_len + 1` windows; the trailing partial window is dropped
/// at `finish`, mirroring the reference packing.
#[derive(Debug, Clone)]
pub struct WindowPacker {
    window_len: usize,
    pending: Vec<u32>,
    windows: Vec<Vec<u32>>,
}

impl WindowPacker {
    /// Creates a packer for the given `seq_len`.
    pub fn new(seq_len: usize) -> Result<Self, DataError> {
        if seq_len < 2 {
            return Err(DataError::PackingInvalid {
                reason: format!("seq_len must be at least 2, got {seq_len}"),
            });
        }
        Ok(Self {
            window_len: seq_len + 1,
            pending: Vec::new(),
            windows: Vec::new(),
        })
    }

    /// The full window length (`seq_len + 1`).
    pub fn window_len(&self) -> usize {
        self.window_len
    }

    /// Absorbs tokens and drains every complete window.
    pub fn push(&mut self, tokens: &[u32]) {
        self.pending.extend_from_slice(tokens);
        while self.pending.len() >= self.window_len {
            let window: Vec<u32> = self.pending.drain(..self.window_len).collect();
            self.windows.push(window);
        }
    }

    /// Discards the trailing partial window (as the reference packing does)
    /// and returns how many tokens were dropped.
    pub fn finish(&mut self) -> usize {
        let dropped = self.pending.len();
        self.pending.clear();
        dropped
    }

    /// The complete windows collected so far, in order.
    pub fn windows(&self) -> &[Vec<u32>] {
        &self.windows
    }

    /// Takes the collected windows, leaving the packer empty.
    pub fn into_windows(self) -> Vec<Vec<u32>> {
        self.windows
    }
}
