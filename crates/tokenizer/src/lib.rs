//! Deterministic, versioned byte-level BPE tokenizer.
//!
//! Content encoding always produces byte/BPE IDs. It never recognizes control-token
//! spellings in text. Only callers using the typed renderer API can create control IDs.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

pub const SPECIAL_TOKEN_COUNT: u32 = 15;
pub const RESERVED_TOKEN_COUNT: u32 = 32;
pub const BYTE_TOKEN_BASE: u32 = SPECIAL_TOKEN_COUNT + RESERVED_TOKEN_COUNT;
pub const BYTE_TOKEN_COUNT: u32 = 256;
pub const FIRST_MERGE_ID: u32 = BYTE_TOKEN_BASE + BYTE_TOKEN_COUNT;
pub const MIN_VOCAB_SIZE: u32 = 32_768;
pub const MAX_VOCAB_SIZE: u32 = 49_152;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokenId(pub u32);

impl TokenId {
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Control tokens have stable IDs. The first 15 IDs must never be reordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialToken {
    Bos,
    End,
    Pad,
    System,
    User,
    Assistant,
    Tool,
    Think,
    EndThink,
    Tools,
    EndTools,
    Call,
    EndCall,
    Result,
    EndResult,
    Reserved(u8),
}

impl SpecialToken {
    pub const fn id(self) -> TokenId {
        let id = match self {
            Self::Bos => 0,
            Self::End => 1,
            Self::Pad => 2,
            Self::System => 3,
            Self::User => 4,
            Self::Assistant => 5,
            Self::Tool => 6,
            Self::Think => 7,
            Self::EndThink => 8,
            Self::Tools => 9,
            Self::EndTools => 10,
            Self::Call => 11,
            Self::EndCall => 12,
            Self::Result => 13,
            Self::EndResult => 14,
            Self::Reserved(index) => 15 + index as u32,
        };
        TokenId(id)
    }

    pub fn from_id(id: TokenId) -> Option<Self> {
        Some(match id.0 {
            0 => Self::Bos,
            1 => Self::End,
            2 => Self::Pad,
            3 => Self::System,
            4 => Self::User,
            5 => Self::Assistant,
            6 => Self::Tool,
            7 => Self::Think,
            8 => Self::EndThink,
            9 => Self::Tools,
            10 => Self::EndTools,
            11 => Self::Call,
            12 => Self::EndCall,
            13 => Self::Result,
            14 => Self::EndResult,
            15..=46 => Self::Reserved((id.0 - 15) as u8),
            _ => return None,
        })
    }

    pub fn spelling(self) -> String {
        match self {
            Self::Bos => "<|bos|>".to_owned(),
            Self::End => "<|end|>".to_owned(),
            Self::Pad => "<|pad|>".to_owned(),
            Self::System => "<|system|>".to_owned(),
            Self::User => "<|user|>".to_owned(),
            Self::Assistant => "<|assistant|>".to_owned(),
            Self::Tool => "<|tool|>".to_owned(),
            Self::Think => "<|think|>".to_owned(),
            Self::EndThink => "<|/think|>".to_owned(),
            Self::Tools => "<|tools|>".to_owned(),
            Self::EndTools => "<|/tools|>".to_owned(),
            Self::Call => "<|call|>".to_owned(),
            Self::EndCall => "<|/call|>".to_owned(),
            Self::Result => "<|result|>".to_owned(),
            Self::EndResult => "<|/result|>".to_owned(),
            Self::Reserved(index) => format!("<|reserved_{index}|>"),
        }
    }
}

pub const SPECIAL_TOKEN_SPELLINGS: [&str; 15] = [
    "<|bos|>",
    "<|end|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|think|>",
    "<|/think|>",
    "<|tools|>",
    "<|/tools|>",
    "<|call|>",
    "<|/call|>",
    "<|result|>",
    "<|/result|>",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Merge {
    pub left: u32,
    pub right: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerArtifact {
    pub version: u32,
    pub vocab_size: u32,
    /// Merge IDs are implicit and contiguous, beginning at FIRST_MERGE_ID.
    pub merges: Vec<Merge>,
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("tokenizer vocabulary size {0} is outside the supported range")]
    InvalidVocabSize(u32),
    #[error("artifact vocabulary size does not match the number of merge entries")]
    VocabSizeMismatch,
    #[error("merge at index {index} refers to a non-existent or non-prior token")]
    InvalidMerge { index: usize },
    #[error("duplicate BPE merge pair at index {0}")]
    DuplicateMerge(usize),
    #[error("token ID {0} is outside this tokenizer vocabulary")]
    TokenOutOfRange(u32),
    #[error("control/reserved token ID {0} cannot be decoded as text content")]
    ControlTokenInContent(u32),
    #[error("decoded bytes are not valid UTF-8")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
}

#[derive(Debug, Clone, Copy)]
struct MergeEntry {
    rank: u32,
    output: TokenId,
}

type MergeQueue = BinaryHeap<Reverse<(u32, usize, usize, u64, u64)>>;

#[derive(Debug, Clone)]
struct Node {
    token: TokenId,
    previous: Option<usize>,
    next: Option<usize>,
    alive: bool,
    generation: u64,
}

/// A validated tokenizer table. Construct production instances from a versioned artifact.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    version: u32,
    vocab_size: u32,
    merges: Vec<Merge>,
    merge_by_pair: HashMap<(u32, u32), MergeEntry>,
    merge_children: Vec<Option<(TokenId, TokenId)>>,
}

impl Tokenizer {
    /// Build a small/in-memory tokenizer. Production artifacts should use from_artifact.
    pub fn from_merges(version: u32, merges: Vec<Merge>) -> Result<Self, TokenizerError> {
        let merge_count = u32::try_from(merges.len()).map_err(|_| TokenizerError::VocabSizeMismatch)?;
        let vocab_size = FIRST_MERGE_ID
            .checked_add(merge_count)
            .ok_or(TokenizerError::VocabSizeMismatch)?;
        Self::build(version, vocab_size, merges)
    }

    /// Load a production artifact and enforce the published 32k–48k vocabulary bounds.
    pub fn from_artifact(artifact: TokenizerArtifact) -> Result<Self, TokenizerError> {
        if !(MIN_VOCAB_SIZE..=MAX_VOCAB_SIZE).contains(&artifact.vocab_size) {
            return Err(TokenizerError::InvalidVocabSize(artifact.vocab_size));
        }
        Self::build(artifact.version, artifact.vocab_size, artifact.merges)
    }

    fn build(version: u32, vocab_size: u32, merges: Vec<Merge>) -> Result<Self, TokenizerError> {
        let expected_size = FIRST_MERGE_ID
            .checked_add(u32::try_from(merges.len()).map_err(|_| TokenizerError::VocabSizeMismatch)?)
            .ok_or(TokenizerError::VocabSizeMismatch)?;
        if expected_size != vocab_size {
            return Err(TokenizerError::VocabSizeMismatch);
        }

        let mut merge_by_pair = HashMap::with_capacity(merges.len());
        let mut merge_children = vec![None; vocab_size as usize];
        for (index, merge) in merges.iter().copied().enumerate() {
            let output_id = FIRST_MERGE_ID + index as u32;
            if merge.left < BYTE_TOKEN_BASE
                || merge.right < BYTE_TOKEN_BASE
                || merge.left >= output_id
                || merge.right >= output_id
            {
                return Err(TokenizerError::InvalidMerge { index });
            }
            if merge_by_pair
                .insert(
                    (merge.left, merge.right),
                    MergeEntry {
                        rank: output_id,
                        output: TokenId(output_id),
                    },
                )
                .is_some()
            {
                return Err(TokenizerError::DuplicateMerge(index));
            }
            merge_children[output_id as usize] =
                Some((TokenId(merge.left), TokenId(merge.right)));
        }

        Ok(Self {
            version,
            vocab_size,
            merges,
            merge_by_pair,
            merge_children,
        })
    }

    pub const fn version(&self) -> u32 {
        self.version
    }

    pub const fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub fn to_artifact(&self) -> TokenizerArtifact {
        TokenizerArtifact {
            version: self.version,
            vocab_size: self.vocab_size,
            merges: self.merges.clone(),
        }
    }

    /// NFC-normalize content and encode it using byte tokens and ranked BPE merges.
    /// Text that looks like a control-token spelling is always ordinary content.
    pub fn encode_content(&self, text: &str) -> Vec<TokenId> {
        let normalized: String = text.nfc().collect();
        self.encode_bytes(normalized.as_bytes())
    }

    fn encode_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        if bytes.is_empty() {
            return Vec::new();
        }

        let mut nodes: Vec<Node> = bytes
            .iter()
            .enumerate()
            .map(|(index, byte)| Node {
                token: TokenId(BYTE_TOKEN_BASE + u32::from(*byte)),
                previous: index.checked_sub(1),
                next: (index + 1 < bytes.len()).then_some(index + 1),
                alive: true,
                generation: 0,
            })
            .collect();

        let mut queue: BinaryHeap<Reverse<(u32, usize, usize, u64, u64)>> =
            BinaryHeap::new();
        for left in 0..nodes.len().saturating_sub(1) {
            Self::queue_pair(left, &nodes, &self.merge_by_pair, &mut queue);
        }

        while let Some(Reverse((rank, left, right, left_generation, right_generation))) =
            queue.pop()
        {
            let Some(left_node) = nodes.get(left) else {
                continue;
            };
            let Some(right_node) = nodes.get(right) else {
                continue;
            };
            if !left_node.alive
                || !right_node.alive
                || left_node.next != Some(right)
                || left_node.generation != left_generation
                || right_node.generation != right_generation
            {
                continue;
            }
            let Some(merge) = self
                .merge_by_pair
                .get(&(left_node.token.0, right_node.token.0))
                .copied()
            else {
                continue;
            };
            if merge.rank != rank {
                continue;
            }

            let previous = left_node.previous;
            let next = right_node.next;
            nodes[left].token = merge.output;
            nodes[left].next = next;
            nodes[left].generation = nodes[left].generation.saturating_add(1);
            nodes[right].alive = false;
            nodes[right].generation = nodes[right].generation.saturating_add(1);
            if let Some(next_index) = next {
                if let Some(next_node) = nodes.get_mut(next_index) {
                    next_node.previous = Some(left);
                }
            }

            if let Some(previous_index) = previous {
                Self::queue_pair(previous_index, &nodes, &self.merge_by_pair, &mut queue);
            }
            Self::queue_pair(left, &nodes, &self.merge_by_pair, &mut queue);
        }

        let mut output = Vec::with_capacity(bytes.len());
        let mut cursor = Some(0_usize);
        while let Some(index) = cursor {
            let Some(node) = nodes.get(index) else {
                break;
            };
            if node.alive {
                output.push(node.token);
            }
            cursor = node.next;
        }
        output
    }

    fn queue_pair(
        left: usize,
        nodes: &[Node],
        merges: &HashMap<(u32, u32), MergeEntry>,
        queue: &mut MergeQueue,
    ) {
        let Some(left_node) = nodes.get(left) else {
            return;
        };
        let Some(right) = left_node.next else {
            return;
        };
        let Some(right_node) = nodes.get(right) else {
            return;
        };
        if !left_node.alive || !right_node.alive {
            return;
        }
        if let Some(merge) = merges.get(&(left_node.token.0, right_node.token.0)) {
            queue.push(Reverse((
                merge.rank,
                left,
                right,
                left_node.generation,
                right_node.generation,
            )));
        }
    }

    /// Expand a content token to its original bytes.
    pub fn token_bytes(&self, token: TokenId) -> Result<Vec<u8>, TokenizerError> {
        if token.0 >= self.vocab_size {
            return Err(TokenizerError::TokenOutOfRange(token.0));
        }
        let mut bytes = Vec::new();
        let mut stack = vec![token];
        while let Some(current) = stack.pop() {
            if current.0 < self.vocab_size && current.0 >= FIRST_MERGE_ID {
                let Some((left, right)) = self
                    .merge_children
                    .get(current.0 as usize)
                    .copied()
                    .flatten()
                else {
                    return Err(TokenizerError::TokenOutOfRange(current.0));
                };
                stack.push(right);
                stack.push(left);
            } else if current.0 >= BYTE_TOKEN_BASE {
                let byte = current.0 - BYTE_TOKEN_BASE;
                if byte >= BYTE_TOKEN_COUNT {
                    return Err(TokenizerError::TokenOutOfRange(current.0));
                }
                bytes.push(byte as u8);
            } else {
                return Err(TokenizerError::ControlTokenInContent(current.0));
            }
        }
        Ok(bytes)
    }

    /// Decode content tokens. Control and reserved IDs are rejected, never rendered as text.
    pub fn decode_content(&self, tokens: &[TokenId]) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        for token in tokens {
            bytes.extend(self.token_bytes(*token)?);
        }
        Ok(String::from_utf8(bytes)?)
    }

    /// Check whether untrusted text contains a literal spelling reserved for control tokens.
    pub fn contains_special_token_spelling(text: &str) -> bool {
        if SPECIAL_TOKEN_SPELLINGS.iter().any(|spelling| text.contains(spelling)) {
            return true;
        }
        (0..RESERVED_TOKEN_COUNT)
            .any(|index| text.contains(&format!("<|reserved_{index}|>")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn toy_tokenizer() -> Tokenizer {
        Tokenizer::from_merges(
            1,
            vec![
                Merge {
                    left: BYTE_TOKEN_BASE + u32::from(b'h'),
                    right: BYTE_TOKEN_BASE + u32::from(b'e'),
                },
                Merge {
                    left: FIRST_MERGE_ID,
                    right: BYTE_TOKEN_BASE + u32::from(b'l'),
                },
                Merge {
                    left: FIRST_MERGE_ID + 1,
                    right: BYTE_TOKEN_BASE + u32::from(b'l'),
                },
            ],
        )
        .expect("test tokenizer must be valid")
    }

    #[test]
    fn special_token_ids_are_stable_and_contiguous() {
        for id in 0..SPECIAL_TOKEN_COUNT + RESERVED_TOKEN_COUNT {
            let token = SpecialToken::from_id(TokenId(id));
            assert!(token.is_some());
            assert_eq!(token.map(SpecialToken::id), Some(TokenId(id)));
        }
        assert_eq!(SpecialToken::Bos.id(), TokenId(0));
        assert_eq!(SpecialToken::EndResult.id(), TokenId(14));
        assert_eq!(SpecialToken::Reserved(31).id(), TokenId(46));
    }

    #[test]
    fn token_like_content_is_encoded_as_ordinary_bytes() {
        let tokenizer = toy_tokenizer();
        let text = "literal <|call|> and <|reserved_3|>";
        let encoded = tokenizer.encode_content(text);
        assert!(encoded.iter().all(|id| id.0 >= BYTE_TOKEN_BASE));
        assert_eq!(tokenizer.decode_content(&encoded).ok().as_deref(), Some(text));
    }

    #[test]
    fn ranked_bpe_merges_and_expands() {
        let tokenizer = toy_tokenizer();
        let encoded = tokenizer.encode_content("hell");
        assert_eq!(encoded, vec![TokenId(FIRST_MERGE_ID + 2)]);
        assert_eq!(tokenizer.decode_content(&encoded).ok().as_deref(), Some("hell"));
    }

    #[test]
    fn nfc_normalization_is_applied() {
        let tokenizer = toy_tokenizer();
        let encoded = tokenizer.encode_content("e\u{301}");
        assert_eq!(tokenizer.decode_content(&encoded).ok().as_deref(), Some("é"));
    }

    #[test]
    fn artifact_requires_a_supported_vocab_and_exact_merge_count() {
        let too_small = TokenizerArtifact {
            version: 1,
            vocab_size: 1_000,
            merges: Vec::new(),
        };
        assert!(matches!(
            Tokenizer::from_artifact(too_small),
            Err(TokenizerError::InvalidVocabSize(1_000))
        ));
    }

    #[test]
    fn control_ids_cannot_be_decoded_as_content() {
        let tokenizer = toy_tokenizer();
        assert!(matches!(
            tokenizer.decode_content(&[SpecialToken::Call.id()]),
            Err(TokenizerError::ControlTokenInContent(11))
        ));
    }

    proptest! {
        #[test]
        fn arbitrary_unicode_round_trips_after_nfc(input in any::<String>()) {
            let tokenizer = toy_tokenizer();
            let normalized: String = input.nfc().collect();
            let encoded = tokenizer.encode_content(&input);
            prop_assert_eq!(tokenizer.decode_content(&encoded).ok(), Some(normalized));
            prop_assert!(encoded.iter().all(|id| id.0 >= BYTE_TOKEN_BASE));
        }
    }
}
