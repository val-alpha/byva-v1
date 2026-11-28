//! Scaled deterministic byte-level BPE trainer (D-0017, D-0020, D-0021).
//!
//! Trains a ranked merge table directly on documents with no pre-tokenization
//! boundary, matching `Tokenizer::encode_bytes` segmentation exactly. The
//! determinism contract is identical to the Python trainer in
//! `training/train_bpe.py`:
//!
//! - Pair counts are aggregated over the document set, so input order never
//!   affects the result.
//! - Each step merges the pair with the highest count; ties are broken by
//!   ascending left token id, then ascending right token id.
//! - Occurrences are replaced left-to-right without overlap, like the Rust
//!   encoder's linked-list walk.
//! - Only pairs occurring at least twice are mergeable.
//!
//! Memory scales with the corpus: documents are `u16` symbol arrays and pair
//! positions are exact-capacity `u64`-packed (document, index) vectors
//! instead of the Python trainer's boxed tuples, so multi-hundred-megabyte
//! corpora train in bounded RAM. Position buckets are lazily pruned: a
//! decrement only lowers the count, so a bucket may hold stale entries that
//! are validated (and skipped) when the pair is merged. Counts are the source
//! of truth and stay exact because every occurrence is created once (with an
//! increment) and destroyed once (with a decrement), and symbol ids never
//! revert to an earlier value.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io::BufRead;
use std::path::Path;

use serde::Serialize;
use unicode_normalization::UnicodeNormalization;

use crate::{BYTE_TOKEN_BASE, FIRST_MERGE_ID, Merge, TokenizerArtifact, TokenizerError};

/// A pair must occur at least this often to be mergeable.
const MIN_PAIR_COUNT: u32 = 2;

/// Sentinel symbol id marking a consumed (dead) position. Real token ids stay
/// below `MAX_VOCAB_SIZE` (49,152), so `u16::MAX` is never a live symbol.
const DEAD: u16 = u16::MAX;

#[inline]
fn pair_key(left: u32, right: u32) -> u32 {
    (left << 16) | right
}

#[inline]
fn symbol_id(symbol: u16) -> u32 {
    u32::from(symbol)
}

#[inline]
fn position_key(doc_index: usize, index: usize) -> u64 {
    ((doc_index as u64) << 32) | (index as u64)
}

/// Heap entry ordered so the "greatest" element is the best merge candidate:
/// highest count first, then lowest left id, then lowest right id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    count: u32,
    left: u32,
    right: u32,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Counts map bookkeeping shared by the initialization pass and merge
/// application. `positions[pair]` lists every (document, index) offset whose
/// symbol pair equaled `pair` when the entry was written; stale entries are
/// validated lazily, while `counts` always equals the number of valid
/// occurrences.
struct PairIndex {
    counts: HashMap<u32, u32>,
    positions: HashMap<u32, Vec<u64>>,
    heap: BinaryHeap<Candidate>,
}

impl PairIndex {
    fn new() -> Self {
        Self {
            counts: HashMap::new(),
            positions: HashMap::new(),
            heap: BinaryHeap::new(),
        }
    }

    fn count_of(&self, key: u32) -> u32 {
        self.counts.get(&key).copied().unwrap_or(0)
    }

    fn add_position(&mut self, key: u32, position: u64) {
        self.positions.entry(key).or_default().push(position);
    }

    /// Applies a count delta to one occurrence, mirroring the Python `_bump`:
    /// new occurrences are appended to the position bucket (decrements leave
    /// stale entries behind for lazy validation), and the candidate heap
    /// always receives a fresh entry for the new count.
    fn bump(&mut self, key: u32, left: u32, right: u32, delta: i64, position: Option<u64>) {
        let current = self.count_of(key);
        let updated = current as i64 + delta;
        if updated > 0 {
            let updated = updated as u32;
            self.counts.insert(key, updated);
            if let (Some(position), true) = (position, delta > 0) {
                self.add_position(key, position);
            }
            self.heap.push(Candidate {
                count: updated,
                left,
                right,
            });
        } else {
            self.counts.remove(&key);
            if delta < 0 {
                self.positions.remove(&key);
            }
        }
    }

    /// Pops the best valid candidate: current count must still equal the
    /// recorded count and be at least `MIN_PAIR_COUNT`. An evicted pair keeps
    /// its position bucket because a later increment can revive it.
    fn pop_best(&mut self) -> Option<(u32, u32)> {
        while let Some(candidate) = self.heap.pop() {
            let key = pair_key(candidate.left, candidate.right);
            let current = self.count_of(key);
            if current < MIN_PAIR_COUNT {
                self.counts.remove(&key);
                continue;
            }
            if current != candidate.count {
                self.heap.push(Candidate {
                    count: current,
                    left: candidate.left,
                    right: candidate.right,
                });
                continue;
            }
            return Some((candidate.left, candidate.right));
        }
        None
    }

    /// Bounds heap growth on large corpora by dropping stale entries. The
    /// rebuilt entries are exactly what fresh pushes would produce, so the
    /// selection sequence is unchanged.
    fn compact_heap(&mut self) {
        self.heap.clear();
        for (&key, &count) in &self.counts {
            if count >= MIN_PAIR_COUNT {
                self.heap.push(Candidate {
                    count,
                    left: key >> 16,
                    right: key & 0xFFFF,
                });
            }
        }
    }
}

fn next_alive(symbols: &[u16], start: usize) -> usize {
    let mut index = start;
    while index < symbols.len() && symbols[index] == DEAD {
        index += 1;
    }
    index
}

fn prev_alive(symbols: &[u16], start: usize) -> isize {
    let mut index = start as isize;
    while index >= 0 && symbols[index as usize] == DEAD {
        index -= 1;
    }
    index
}

/// Byte-level symbol sequence of one document.
fn doc_symbols(text: &str) -> Vec<u16> {
    text.as_bytes()
        .iter()
        .map(|byte| (BYTE_TOKEN_BASE + u32::from(*byte)) as u16)
        .collect()
}

/// Documents loaded from JSONL corpus files: NFC-normalized `{"text": ...}`
/// records, exactly like `load_shard_texts` in `training/train_bpe.py`.
pub struct CorpusSummary {
    pub documents: usize,
    pub bytes: usize,
}

fn read_corpus_documents(
    paths: &[&Path],
) -> Result<(Vec<Vec<u16>>, CorpusSummary), TokenizerError> {
    let mut documents: Vec<Vec<u16>> = Vec::new();
    let mut total_bytes = 0usize;
    for path in paths {
        let file = std::fs::File::open(path)
            .map_err(|error| TokenizerError::CorpusRead(format!("{}: {error}", path.display())))?;
        let mut reader = std::io::BufReader::new(file);
        let mut line = String::new();
        let mut line_number = 0usize;
        loop {
            line.clear();
            let read = reader.read_line(&mut line).map_err(|error| {
                TokenizerError::CorpusRead(format!("{}: {error}", path.display()))
            })?;
            if read == 0 {
                break;
            }
            line_number += 1;
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(&line).map_err(|error| {
                TokenizerError::CorpusRead(format!(
                    "{}:{}: invalid JSON: {error}",
                    path.display(),
                    line_number
                ))
            })?;
            let text = value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    TokenizerError::CorpusRead(format!(
                        "{}:{}: record must be an object with a text field",
                        path.display(),
                        line_number
                    ))
                })?;
            let normalized: String = text.nfc().collect();
            if normalized.is_empty() {
                continue;
            }
            total_bytes += normalized.len();
            documents.push(doc_symbols(&normalized));
        }
    }
    let summary = CorpusSummary {
        documents: documents.len(),
        bytes: total_bytes,
    };
    Ok((documents, summary))
}

/// Trains `target_vocab - FIRST_MERGE_ID` merges over `texts` and returns the
/// validated artifact shape. Texts are used as-is; callers hand over prepared
/// (deduplicated, normalized) documents as produced by the Phase 2 pipeline.
pub fn train_bpe(
    texts: &[String],
    target_vocab: u32,
    version: u32,
) -> Result<TokenizerArtifact, TokenizerError> {
    let documents: Vec<Vec<u16>> = texts
        .iter()
        .filter(|text| !text.is_empty())
        .map(|text| doc_symbols(text))
        .collect();
    train_from_documents(documents, target_vocab, version)
}

/// Trains from JSONL corpus files, streaming one record at a time so corpus
/// text is never held twice. Returns the artifact and a corpus size summary.
pub fn train_bpe_from_jsonl(
    paths: &[&Path],
    target_vocab: u32,
    version: u32,
) -> Result<(TokenizerArtifact, CorpusSummary), TokenizerError> {
    let (documents, summary) = read_corpus_documents(paths)?;
    let artifact = train_from_documents(documents, target_vocab, version)?;
    Ok((artifact, summary))
}

fn train_from_documents(
    mut documents: Vec<Vec<u16>>,
    target_vocab: u32,
    version: u32,
) -> Result<TokenizerArtifact, TokenizerError> {
    if target_vocab <= FIRST_MERGE_ID {
        return Err(TokenizerError::TargetVocabTooSmall(target_vocab));
    }
    let merge_target = (target_vocab - FIRST_MERGE_ID) as usize;
    if documents.is_empty() {
        return Err(TokenizerError::EmptyCorpus);
    }

    let mut index = PairIndex::new();
    for symbols in &documents {
        for window in 0..symbols.len().saturating_sub(1) {
            let key = pair_key(symbol_id(symbols[window]), symbol_id(symbols[window + 1]));
            index.counts.entry(key).and_modify(|c| *c += 1).or_insert(1);
        }
    }
    index.counts.retain(|_, &mut count| count >= MIN_PAIR_COUNT);

    for (doc_index, symbols) in documents.iter().enumerate() {
        for window in 0..symbols.len().saturating_sub(1) {
            let key = pair_key(symbol_id(symbols[window]), symbol_id(symbols[window + 1]));
            if let Some(count) = index.counts.get(&key) {
                let bucket = index
                    .positions
                    .entry(key)
                    .or_insert_with(|| Vec::with_capacity(*count as usize));
                bucket.push(position_key(doc_index, window));
            }
        }
    }
    index.compact_heap();

    let mut merges: Vec<Merge> = Vec::with_capacity(merge_target);
    while merges.len() < merge_target {
        if index.heap.len() > index.counts.len().saturating_mul(4) + (1 << 20) {
            index.compact_heap();
        }
        let Some((left, right)) = index.pop_best() else {
            return Err(TokenizerError::CorpusExhausted {
                trained: merges.len(),
                requested: merge_target,
            });
        };
        let key = pair_key(left, right);
        let output = FIRST_MERGE_ID + merges.len() as u32;
        merges.push(Merge { left, right });
        apply_merge(&mut documents, &mut index, key, left, right, output);
    }

    Ok(TokenizerArtifact {
        version,
        vocab_size: FIRST_MERGE_ID + merges.len() as u32,
        merges,
    })
}

/// Serializes an artifact to the canonical byte-identical JSON layout used by
/// `training/train_bpe.py` (`sort_keys` + compact separators): merge list
/// first, then `version`, then `vocab_size`.
pub fn artifact_file_json(artifact: &TokenizerArtifact) -> Result<String, TokenizerError> {
    #[derive(Serialize)]
    struct ArtifactFile<'a> {
        merges: &'a [Merge],
        version: u32,
        vocab_size: u32,
    }
    serde_json::to_string(&ArtifactFile {
        merges: &artifact.merges,
        version: artifact.version,
        vocab_size: artifact.vocab_size,
    })
    .map_err(|error| TokenizerError::Serialization(error.to_string()))
}

/// Replaces occurrences of `(left, right)` in ascending (document, index)
/// order without overlap, updating counts incrementally. The position bucket
/// may hold stale entries; each is validated against the live symbols.
fn apply_merge(
    documents: &mut [Vec<u16>],
    index: &mut PairIndex,
    key: u32,
    left: u32,
    right: u32,
    output: u32,
) {
    let mut occupied = index.positions.remove(&key).unwrap_or_default();
    index.counts.remove(&key);
    occupied.sort_unstable();
    occupied.dedup();

    for packed in occupied {
        let doc_index = (packed >> 32) as usize;
        let position = (packed & 0xFFFF_FFFF) as usize;
        let symbols = &mut documents[doc_index];
        if position >= symbols.len() || symbol_id(symbols[position]) != left {
            continue;
        }
        let next_position = next_alive(symbols, position + 1);
        if next_position >= symbols.len() || symbol_id(symbols[next_position]) != right {
            continue;
        }
        let prev_position = prev_alive(symbols, position.wrapping_sub(1));
        let after_position = next_alive(symbols, next_position + 1);

        symbols[position] = output as u16;
        symbols[next_position] = DEAD;

        index.bump(key, left, right, -1, None);
        if prev_position >= 0 {
            let prev = prev_position as usize;
            let prev_symbol = symbol_id(symbols[prev]);
            index.bump(pair_key(prev_symbol, left), prev_symbol, left, -1, None);
            index.bump(
                pair_key(prev_symbol, output),
                prev_symbol,
                output,
                1,
                Some(position_key(doc_index, prev)),
            );
        }
        if after_position < symbols.len() {
            let after_symbol = symbol_id(symbols[after_position]);
            index.bump(pair_key(right, after_symbol), right, after_symbol, -1, None);
            index.bump(
                pair_key(output, after_symbol),
                output,
                after_symbol,
                1,
                Some(position_key(doc_index, position)),
            );
        }
    }
}
