use byva_tokenizer::{
    BYTE_TOKEN_BASE, FIRST_MERGE_ID, Merge, SpecialToken, TokenId, Tokenizer, TokenizerArtifact,
};
use serde_json::Value;

#[test]
fn committed_fixture_ids_match_the_v1_layout() {
    let raw = include_str!("golden/tokenizer_v1.json");
    let fixture: Value = serde_json::from_str(raw).expect("golden fixture is valid JSON");
    let ids = fixture
        .get("special_ids")
        .and_then(|ids| ids.as_object())
        .expect("fixture has special_ids");

    let expected: [(&str, u32); 19] = [
        ("bos", 0),
        ("end", 1),
        ("pad", 2),
        ("system", 3),
        ("user", 4),
        ("assistant", 5),
        ("tool", 6),
        ("think", 7),
        ("end_think", 8),
        ("tools", 9),
        ("end_tools", 10),
        ("call", 11),
        ("end_call", 12),
        ("result", 13),
        ("end_result", 14),
        ("reserved_first", 15),
        ("reserved_last", 46),
        ("byte_first", BYTE_TOKEN_BASE),
        ("byte_last", BYTE_TOKEN_BASE + 255),
    ];
    for (name, expected_id) in expected {
        assert_eq!(
            ids.get(name).and_then(Value::as_u64),
            Some(u64::from(expected_id)),
            "fixture id {name} drifted"
        );
    }
    assert_eq!(
        ids.get("merge_first").and_then(Value::as_u64),
        Some(u64::from(FIRST_MERGE_ID))
    );

    assert_eq!(SpecialToken::Bos.id(), TokenId(0));
    assert_eq!(SpecialToken::EndResult.id(), TokenId(14));
    assert_eq!(SpecialToken::Reserved(31).id(), TokenId(46));
}

#[test]
fn tokenizer_v1_golden_ids_are_stable() {
    let merges = vec![Merge {
        left: BYTE_TOKEN_BASE + u32::from(b'o'),
        right: BYTE_TOKEN_BASE + u32::from(b'k'),
    }];
    let tokenizer = Tokenizer::from_merges(1, merges).expect("valid golden tokenizer");
    assert_eq!(tokenizer.version(), 1);
    assert_eq!(
        tokenizer.encode_content("ok"),
        vec![TokenId(FIRST_MERGE_ID)]
    );
    assert_eq!(
        tokenizer.encode_content("x"),
        vec![TokenId(BYTE_TOKEN_BASE + u32::from(b'x'))]
    );
}

#[test]
fn python_trained_bpe_fixture_round_trips_through_the_rust_runtime() {
    let raw = include_str!("golden/bpe_smoke_v1.json");
    let artifact: TokenizerArtifact =
        serde_json::from_str(raw).expect("python-trained fixture is valid JSON");
    assert_eq!(artifact.version, 1);
    assert_eq!(artifact.vocab_size, FIRST_MERGE_ID + 64);

    let byte = |byte: u8| BYTE_TOKEN_BASE + u32::from(byte);
    assert_eq!(
        artifact.merges[0],
        Merge {
            left: byte(b'a'),
            right: byte(b' ')
        }
    );
    assert_eq!(
        artifact.merges[1],
        Merge {
            left: byte(b'e'),
            right: byte(b' ')
        }
    );

    let tokenizer = Tokenizer::from_merges(artifact.version, artifact.merges.clone())
        .expect("fixture merges are valid");
    let texts = [
        "Teks bahasa Indonesia nomor satu untuk pengujian pipeline data BYVA yang deterministik.",
        "Second English document used to exercise the deterministic ordering of shards.",
        "Cerita pendek berbahasa Indonesia tentang kucing yang penasaran dengan mesin cuci di rumah.",
    ];
    for text in texts {
        let encoded = tokenizer.encode_content(text);
        assert!(!encoded.is_empty());
        assert!(encoded.iter().all(|id| id.0 < artifact.vocab_size));
        let decoded = tokenizer
            .decode_content(&encoded)
            .expect("fixture round-trip is valid UTF-8");
        assert_eq!(decoded, text);
    }
}
