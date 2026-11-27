use byva_tokenizer::{BYTE_TOKEN_BASE, FIRST_MERGE_ID, Merge, SpecialToken, TokenId, Tokenizer};
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
