//! Document language tags for the K5 id:en mixture (D-0016, R6).
//!
//! Shards carry no language tags, so each document is classified inline with
//! a deterministic stopword-markers heuristic: no dependencies, no RNG, and
//! the same document always yields the same tag.

/// The two D-0016 languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    /// Indonesian.
    Id,
    /// English.
    En,
}

impl Lang {
    /// Short lowercase name ("id" or "en").
    pub fn name(self) -> &'static str {
        match self {
            Lang::Id => "id",
            Lang::En => "en",
        }
    }
}

/// Function-word markers whose standalone occurrence votes for a language.
/// Both lists are deliberately small and unambiguous; a document only needs
/// a majority vote, not a parser.
const ID_MARKERS: &[&str] = &[
    "yang", "dan", "di", "itu", "dengan", "untuk", "tidak", "ini", "dari", "dalam", "akan", "pada",
    "juga", "saya", "kami", "kita", "adalah", "oleh", "atau", "bisa", "ada", "ke", "saat",
    "karena", "agar", "sudah", "belum", "antara", "hanya", "lebih", "masih", "para",
];

const EN_MARKERS: &[&str] = &[
    "the", "and", "of", "to", "in", "that", "is", "was", "for", "with", "as", "by", "at", "this",
    "from", "or", "have", "are", "not", "but", "they", "we", "you", "she", "his", "her", "their",
    "which", "been", "were", "than", "then", "when", "what", "each", "other",
];

/// Classifies a document as Indonesian or English by counting standalone
/// function-word markers; ties (including marker-free text) default to
/// [`Lang::En`].
pub fn detect_language(text: &str) -> Lang {
    let mut id_hits = 0usize;
    let mut en_hits = 0usize;
    let mut word = String::new();
    let score = |word: &str, id_hits: &mut usize, en_hits: &mut usize| {
        if ID_MARKERS.contains(&word) {
            *id_hits += 1;
        }
        if EN_MARKERS.contains(&word) {
            *en_hits += 1;
        }
    };
    for character in text.chars() {
        if character.is_alphabetic() {
            word.extend(character.to_lowercase());
        } else if !word.is_empty() {
            score(&word, &mut id_hits, &mut en_hits);
            word.clear();
        }
    }
    if !word.is_empty() {
        score(&word, &mut id_hits, &mut en_hits);
    }
    if id_hits > en_hits {
        Lang::Id
    } else {
        Lang::En
    }
}

/// One document's token range in a packed stream and its language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocLang {
    /// Offset of the document's first token in the stream.
    pub start: usize,
    /// Number of tokens the document contributed.
    pub len: usize,
    /// The document's language.
    pub lang: Lang,
}

/// Splits the document ranges into consecutive window-sized chunks and tags
/// each window; the trailing partial window is dropped, mirroring `pack`.
///
/// Returns `(window_langs, split)` where `split` is the 90/10 boundary
/// `("windows" / 10) * 9` applied to the same window order as `pack`.
pub fn window_langs(tokens_len: usize, docs: &[DocLang], seq_len: usize) -> (Vec<Lang>, usize) {
    let window_len = seq_len + 1;
    let usable = tokens_len - (tokens_len % window_len);
    let count = usable / window_len;
    let mut langs = Vec::with_capacity(count);
    let mut cursor = 0usize;
    for index in 0..count {
        let start = index * window_len;
        while cursor < docs.len() && docs[cursor].start + docs[cursor].len <= start {
            cursor += 1;
        }
        let first = cursor;
        let mut id_tokens = 0usize;
        let mut en_tokens = 0usize;
        let end = start + window_len;
        for doc in &docs[first..] {
            if doc.start >= end {
                break;
            }
            let overlap = (doc.start + doc.len).min(end) - doc.start.max(start);
            match doc.lang {
                Lang::Id => id_tokens += overlap,
                Lang::En => en_tokens += overlap,
            }
        }
        langs.push(if id_tokens > en_tokens {
            Lang::Id
        } else {
            Lang::En
        });
    }
    let split = (count / 10) * 9;
    (langs, split)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(text: &str) -> Lang {
        detect_language(text)
    }

    #[test]
    fn indonesian_and_english_documents_are_classified() {
        assert_eq!(
            id(
                "Pemerintah merilis kebijakan baru untuk masyarakat di daerah pedalaman pada tahun ini."
            ),
            Lang::Id
        );
        assert_eq!(
            id("The government released a new policy for remote communities this year."),
            Lang::En
        );
    }

    #[test]
    fn marker_free_text_defaults_to_english() {
        assert_eq!(id("12345 67890 !@#$%"), Lang::En);
        assert_eq!(id(""), Lang::En);
    }

    #[test]
    fn casing_and_punctuation_do_not_matter() {
        assert_eq!(id("YANG dan DI ITU dengan untuk TIDAK ini."), Lang::Id);
    }

    #[test]
    fn window_langs_splits_like_pack() {
        let docs = vec![
            DocLang {
                start: 0,
                len: 45,
                lang: Lang::Id,
            },
            DocLang {
                start: 45,
                len: 45,
                lang: Lang::En,
            },
        ];
        let (langs, split) = window_langs(90, &docs, 8);
        assert_eq!(langs.len(), 10);
        assert_eq!(split, 9);
        assert_eq!(langs[0], Lang::Id);
        assert_eq!(langs[5], Lang::En);
    }

    #[test]
    fn window_langs_uses_majority_across_boundaries() {
        let docs = vec![
            DocLang {
                start: 0,
                len: 5,
                lang: Lang::Id,
            },
            DocLang {
                start: 5,
                len: 13,
                lang: Lang::En,
            },
        ];
        let (langs, _) = window_langs(18, &docs, 8);
        assert_eq!(langs, vec![Lang::Id, Lang::En]);
    }
}
