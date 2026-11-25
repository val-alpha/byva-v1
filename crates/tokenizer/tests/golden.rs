use byva_tokenizer::{BYTE_TOKEN_BASE, FIRST_MERGE_ID, Merge, TokenId, Tokenizer};

#[test]
fn tokenizer_v1_golden_ids_are_stable() {
    let merges = vec![Merge {
        left: BYTE_TOKEN_BASE + u32::from(b'o'),
        right: BYTE_TOKEN_BASE + u32::from(b'k'),
    }];
    let tokenizer = Tokenizer::from_merges(1, merges).expect("valid golden tokenizer");
    assert_eq!(tokenizer.version(), 1);
    assert_eq!(tokenizer.encode_content("ok"), vec![TokenId(FIRST_MERGE_ID)]);
    assert_eq!(tokenizer.encode_content("x"), vec![TokenId(BYTE_TOKEN_BASE + u32::from(b'x'))]);
}
