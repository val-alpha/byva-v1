use byva_format::{AssistantPart, Conversation, Message, Renderer};
use byva_tokenizer::{SpecialToken, Tokenizer};

#[test]
fn byva_v1_minimal_message_golden() {
    let tokenizer = Tokenizer::from_merges(1, Vec::new()).expect("empty merge table is valid");
    let conversation = Conversation::new(vec![Message::user("Hi")], Vec::new());
    let actual = Renderer::default()
        .render(&conversation, &tokenizer)
        .expect("golden conversation renders");
    let expected = vec![
        SpecialToken::Bos.id(),
        SpecialToken::User.id(),
        byva_tokenizer::TokenId(47 + u32::from(b'H')),
        byva_tokenizer::TokenId(47 + u32::from(b'i')),
        SpecialToken::End.id(),
    ];
    assert_eq!(actual, expected);
}

#[test]
fn assistant_can_finish_without_thinking() {
    let tokenizer = Tokenizer::from_merges(1, Vec::new()).expect("empty merge table is valid");
    let conversation = Conversation::new(
        vec![Message::assistant(
            vec![AssistantPart::Text("fast".to_owned())],
            true,
        )],
        Vec::new(),
    );
    let tokens = Renderer::default()
        .render(&conversation, &tokenizer)
        .expect("fast message renders");
    assert_eq!(tokens.first(), Some(&SpecialToken::Bos.id()));
    assert_eq!(tokens.last(), Some(&SpecialToken::End.id()));
}
