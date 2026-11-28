//! Phase 6 SFT: chat template, supervised-fine-tuning datasets, and the
//! built-in `byva` identity corpus.
//!
//! Every example is rendered to the D-0013 conversation template
//! `<|bos|>|user|>{prompt}|assistant|>{response}<|end|>` using the
//! tokenizer's control tokens. Loss weights are `1.0` for the response
//! tokens and the closing `<|end|>`, `0.0` everywhere else, so training
//! teaches the model to answer and to stop — never to generate prompts.
//!
//! Examples are concatenated into one token stream (GPT-style packing) and
//! split into non-overlapping `seq_len + 1` windows with the same 90/10
//! formula as [`byva_data::pack`], keeping the K4 batching untouched.

use std::path::Path;

use unicode_normalization::UnicodeNormalization;

use byva_data::Packed;
use byva_data::sha256_hex;
use byva_tokenizer::{SpecialToken, Tokenizer};
use serde::Deserialize;

use crate::error::TrainerError;

/// One supervised example: the instruction shown to the model and the
/// response it must learn to produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SftExample {
    /// Instruction (rendered after `|user|>`).
    pub prompt: String,
    /// Target text (rendered after `|assistant|>`).
    pub response: String,
}

#[derive(Debug, Deserialize)]
struct SftLine {
    id: String,
    prompt: String,
    response: String,
}

/// The canonical text an example's `id` commits to:
/// `prompt + "\n" + response` (NFC-normalized, like the tokenizer input).
pub fn canonical_text(example: &SftExample) -> String {
    format!(
        "{}\n{}",
        example.prompt.nfc().collect::<String>(),
        example.response.nfc().collect::<String>()
    )
}

/// Loads SFT examples from a JSONL file with one
/// `{"id", "prompt", "response"}` object per line. `id` must equal the first
/// 16 hex characters of the SHA-256 of [`canonical_text`] (D-0012 provenance
/// discipline at SFT scale).
pub fn load_examples(path: &Path) -> Result<Vec<SftExample>, TrainerError> {
    let invalid = |reason: String| TrainerError::DataInvalid { reason };
    let raw = std::fs::read_to_string(path).map_err(|error| TrainerError::Io {
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    let mut examples = Vec::new();
    for (index, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: SftLine = serde_json::from_str(line).map_err(|error| {
            invalid(format!("line {} is not valid SFT JSON: {error}", index + 1))
        })?;
        let example = SftExample {
            prompt: parsed.prompt,
            response: parsed.response,
        };
        if example.prompt.trim().is_empty() {
            return Err(invalid(format!("line {} has an empty prompt", index + 1)));
        }
        if example.response.trim().is_empty() {
            return Err(invalid(format!("line {} has an empty response", index + 1)));
        }
        let expected = sha256_hex(canonical_text(&example).as_bytes());
        if parsed.id != expected[..16] {
            return Err(invalid(format!(
                "line {} id {:?} does not match sha256(canonical text)[:16] = {:?}",
                index + 1,
                parsed.id,
                &expected[..16]
            )));
        }
        examples.push(example);
    }
    if examples.is_empty() {
        return Err(invalid("SFT dataset contains no examples".to_string()));
    }
    Ok(examples)
}

/// The built-in `byva` identity corpus: bilingual (Indonesian + English)
/// variations of the official identity attribution — model name `byva`,
/// creator BYV Dev Unorganization (BYV); the creator's real-world identity
/// is undisclosed and must never be guessed.
pub fn identity_examples() -> Vec<SftExample> {
    let pairs: [(&str, &str); 10] = [
        (
            "Siapa namamu?",
            "Nama saya byva, model bahasa kecil buatan BYV Dev Unorganization (BYV).",
        ),
        (
            "Siapa yang membuatmu?",
            "Saya dibuat oleh BYV Dev Unorganization (BYV).",
        ),
        (
            "Siapa developer kamu?",
            "Saya dikembangkan oleh BYV Dev Unorganization (BYV).",
        ),
        (
            "Apa itu byva?",
            "byva adalah model bahasa kecil dari BYV Dev Unorganization (BYV).",
        ),
        (
            "Siapa pemilik byva?",
            "byva milik BYV Dev Unorganization (BYV).",
        ),
        (
            "What is your name?",
            "My name is byva, a small language model made by BYV Dev Unorganization (BYV).",
        ),
        (
            "Who created you?",
            "I was created by BYV Dev Unorganization (BYV).",
        ),
        (
            "Who made you?",
            "I was made by BYV Dev Unorganization (BYV).",
        ),
        (
            "What is byva?",
            "byva is a small language model from BYV Dev Unorganization (BYV).",
        ),
        (
            "Who owns byva?",
            "byva belongs to BYV Dev Unorganization (BYV).",
        ),
    ];
    pairs
        .into_iter()
        .map(|(prompt, response)| SftExample {
            prompt: prompt.to_string(),
            response: response.to_string(),
        })
        .collect()
}

/// A packed SFT dataset: the same window/split layout as pretraining plus
/// one loss weight per target position (`1.0` on response tokens and the
/// closing `<|end|>`, `0.0` elsewhere).
#[derive(Debug, Clone, PartialEq)]
pub struct SftDataset {
    /// Train/val windows (identical split formula to `pack`).
    pub packed: Packed,
    /// Loss weights for the training targets, `count * seq_len` entries.
    pub train_mask: Vec<f32>,
    /// Loss weights for the held-out targets, `count * seq_len` entries.
    pub val_mask: Vec<f32>,
}

/// Renders examples into one token stream plus per-token loss flags:
/// flag `1.0` marks response tokens and the closing `<|end|>`.
pub fn render_stream(
    tokenizer: &Tokenizer,
    examples: &[SftExample],
    repeat: usize,
) -> Result<(Vec<u32>, Vec<f32>), TrainerError> {
    let invalid = |reason: String| TrainerError::DataInvalid { reason };
    if examples.is_empty() {
        return Err(invalid("no SFT examples to render".to_string()));
    }
    if repeat == 0 {
        return Err(invalid("repeat must be positive".to_string()));
    }
    let mut stream: Vec<u32> = Vec::new();
    let mut flags: Vec<f32> = Vec::new();
    let push = |token: u32, flag: f32, stream: &mut Vec<u32>, flags: &mut Vec<f32>| {
        stream.push(token);
        flags.push(flag);
    };
    for _ in 0..repeat {
        for example in examples {
            push(SpecialToken::Bos.id().get(), 0.0, &mut stream, &mut flags);
            push(SpecialToken::User.id().get(), 0.0, &mut stream, &mut flags);
            for token in tokenizer.encode_content(&example.prompt) {
                push(token.get(), 0.0, &mut stream, &mut flags);
            }
            push(
                SpecialToken::Assistant.id().get(),
                0.0,
                &mut stream,
                &mut flags,
            );
            for token in tokenizer.encode_content(&example.response) {
                push(token.get(), 1.0, &mut stream, &mut flags);
            }
            push(SpecialToken::End.id().get(), 1.0, &mut stream, &mut flags);
        }
    }
    Ok((stream, flags))
}

/// Packs the rendered stream exactly like [`byva_data::pack`] (non-overlapping
/// `seq_len + 1` windows, trailing partial window dropped, `(n / 10) * 9`
/// train split) and derives the per-position loss masks from the flags.
pub fn build_dataset(
    tokenizer: &Tokenizer,
    examples: &[SftExample],
    repeat: usize,
    seq_len: usize,
) -> Result<SftDataset, TrainerError> {
    let invalid = |reason: String| TrainerError::DataInvalid { reason };
    if seq_len < 2 {
        return Err(invalid(format!("seq_len {seq_len} must be at least 2")));
    }
    let (stream, flags) = render_stream(tokenizer, examples, repeat)?;
    let window_len = seq_len + 1;
    let mut windows: Vec<Vec<u32>> = Vec::new();
    let mut window_flags: Vec<Vec<f32>> = Vec::new();
    let mut start = 0usize;
    while start + window_len <= stream.len() {
        windows.push(stream[start..start + window_len].to_vec());
        let targets = &flags[start + 1..start + 1 + seq_len];
        window_flags.push(targets.to_vec());
        start += window_len;
    }
    if windows.is_empty() {
        return Err(invalid(format!(
            "SFT dataset renders {} token(s), too few for one window of {}",
            stream.len(),
            window_len
        )));
    }
    let split = (windows.len() / 10) * 9;
    if split == 0 {
        return Err(invalid(format!(
            "SFT dataset produces only {} window(s); the 90/10 split leaves no training windows",
            windows.len()
        )));
    }
    if split == windows.len() {
        return Err(invalid(
            "SFT dataset produces no held-out windows; render more examples".to_string(),
        ));
    }
    let mask_for = |range: std::ops::Range<usize>| -> Vec<f32> {
        let mut mask = Vec::with_capacity(range.len() * seq_len);
        for window in range {
            mask.extend_from_slice(&window_flags[window]);
        }
        mask
    };
    Ok(SftDataset {
        packed: Packed::from_windows(&windows, seq_len),
        train_mask: mask_for(0..split),
        val_mask: mask_for(split..windows.len()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> Tokenizer {
        Tokenizer::from_merges(1, vec![]).unwrap_or_else(|e| panic!("{e}"))
    }

    fn example(prompt: &str, response: &str) -> SftExample {
        SftExample {
            prompt: prompt.to_string(),
            response: response.to_string(),
        }
    }

    #[test]
    fn identity_examples_are_bilingual_and_deterministic() {
        let first = identity_examples();
        let second = identity_examples();
        assert_eq!(first, second);
        assert!(!first.is_empty());
        assert!(first.iter().any(|e| e.prompt.contains("namamu")));
        assert!(first.iter().any(|e| e.prompt.contains("your name")));
        for example in &first {
            assert!(example.response.contains("BYV"));
        }
        assert!(
            first
                .iter()
                .any(|example| example.response.to_lowercase().contains("byva"))
        );
    }

    #[test]
    fn template_marks_only_response_and_end() {
        let tokenizer = tiny_tokenizer();
        let examples = vec![example("a", "b")];
        let (stream, flags) =
            render_stream(&tokenizer, &examples, 1).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(stream.len(), flags.len());
        assert_eq!(stream[0], SpecialToken::Bos.id().get());
        assert_eq!(stream[1], SpecialToken::User.id().get());
        assert_eq!(*flags.last().unwrap_or(&0.0), 1.0);
        assert!(flags.iter().filter(|&&flag| flag == 1.0).count() >= 2);
    }

    #[test]
    fn packing_matches_pack_split_formula() {
        let tokenizer = tiny_tokenizer();
        let body = "lorem ipsum dolor sit amet ".repeat(12);
        let examples: Vec<SftExample> = (0..8)
            .map(|index| example(&format!("q{index} {body}"), &format!("a{index} {body}")))
            .collect();
        let (stream, _) = render_stream(&tokenizer, &examples, 3).unwrap_or_else(|e| panic!("{e}"));
        let dataset = build_dataset(&tokenizer, &examples, 3, 16).unwrap_or_else(|e| panic!("{e}"));
        let packed = byva_data::pack(&stream, 16).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(dataset.packed, packed);
        assert_eq!(
            dataset.train_mask.len(),
            dataset.packed.train().count() * 16
        );
        assert_eq!(dataset.val_mask.len(), dataset.packed.val().count() * 16);
        assert!(dataset.train_mask.contains(&1.0));
        assert!(dataset.train_mask.contains(&0.0));
    }

    #[test]
    fn too_few_examples_are_rejected() {
        let tokenizer = tiny_tokenizer();
        let examples = vec![example("q", "a")];
        let error = build_dataset(&tokenizer, &examples, 1, 512)
            .err()
            .unwrap_or_else(|| panic!("expected rejection"));
        assert!(error.to_string().contains("too few"), "{error}");
    }

    #[test]
    fn jsonl_loader_validates_ids() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("byva-sft-{}-good.jsonl", std::process::id()));
        let example = example("Siapa namamu?", "Nama saya byva.");
        let digest = sha256_hex(canonical_text(&example).as_bytes());
        std::fs::write(
            &path,
            format!(
                "{{\"id\":\"{}\",\"prompt\":\"{}\",\"response\":\"{}\"}}\n",
                &digest[..16],
                example.prompt,
                example.response
            ),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let loaded = load_examples(&path).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(loaded, vec![example]);
        let bad = path.with_extension("bad.jsonl");
        std::fs::write(
            &bad,
            "{\"id\":\"0000000000000000\",\"prompt\":\"q\",\"response\":\"a\"}\n",
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let error = load_examples(&bad)
            .err()
            .unwrap_or_else(|| panic!("expected rejection"));
        assert!(error.to_string().contains("does not match"), "{error}");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bad);
    }
}
