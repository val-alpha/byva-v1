//! Packing tests mirroring `training/train.py::make_sequences` semantics.

use byva_data::{DataError, WindowPacker, pack};

fn tokens(count: u32) -> Vec<u32> {
    (0..count).collect()
}

#[test]
fn hand_case_matches_reference_split() {
    let stream = tokens(30);
    let packed = pack(&stream, 2).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(packed.train().count(), 9);
    assert_eq!(packed.val().count(), 1);
    assert_eq!(packed.train().seq_len(), 2);
    assert_eq!(
        packed.train().x(),
        &[
            0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25
        ]
    );
    assert_eq!(
        packed.train().y(),
        &[
            1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26
        ]
    );
    assert_eq!(packed.val().x(), &[27, 28]);
    assert_eq!(packed.val().y(), &[28, 29]);
}

#[test]
fn trailing_partial_window_is_dropped() {
    let stream = tokens(32);
    let packed = pack(&stream, 2).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(packed.train().count() + packed.val().count(), 10);
    assert!(packed.val().y().last() != Some(&31));
}

#[test]
fn fewer_than_ten_windows_gives_an_empty_train_split() {
    let stream = tokens(10);
    let packed = pack(&stream, 2).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(packed.train().count(), 0);
    assert_eq!(packed.val().count(), 3);
    assert!(packed.train().x().is_empty());
}

#[test]
fn windows_preserve_stream_order() {
    let stream = tokens(60);
    let packed = pack(&stream, 4).unwrap_or_else(|e| panic!("{e}"));
    for row in 0..packed.val().count() - 1 {
        let x = &packed.val().x()[row * 4..(row + 1) * 4];
        let next_x = &packed.val().x()[(row + 1) * 4..(row + 2) * 4];
        assert_eq!(next_x[0], x[0] + 5);
    }
}

#[test]
fn too_few_tokens_is_rejected() {
    let error = pack(&tokens(5), 2).unwrap_err();
    assert!(matches!(error, DataError::PackingInvalid { .. }));
    assert!(error.to_string().contains("not enough tokens"), "{error}");
}

#[test]
fn seq_len_below_two_is_rejected() {
    let error = pack(&tokens(64), 1).unwrap_err();
    assert!(matches!(error, DataError::PackingInvalid { .. }));
    assert!(error.to_string().contains("seq_len"), "{error}");
}

#[test]
fn empty_stream_is_rejected() {
    let error = pack(&[], 8).unwrap_err();
    assert!(matches!(error, DataError::PackingInvalid { .. }));
}

#[test]
fn streaming_packer_equals_batch_packing() {
    let stream = tokens(97);
    let expected = pack(&stream, 3).unwrap_or_else(|e| panic!("{e}"));
    let mut packer = WindowPacker::new(3).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(packer.window_len(), 4);
    for chunk in stream.chunks(7) {
        packer.push(chunk);
    }
    let dropped = packer.finish();
    assert_eq!(dropped, 97 % 4);
    let windows = packer.into_windows();
    assert_eq!(windows.len(), 24);
    let train_windows = &windows[..expected.train().count()];
    let val_windows = &windows[expected.train().count()..];
    let train_x: Vec<u32> = train_windows.iter().flat_map(|w| w[..3].to_vec()).collect();
    let train_y: Vec<u32> = train_windows.iter().flat_map(|w| w[1..].to_vec()).collect();
    let val_x: Vec<u32> = val_windows.iter().flat_map(|w| w[..3].to_vec()).collect();
    let val_y: Vec<u32> = val_windows.iter().flat_map(|w| w[1..].to_vec()).collect();
    assert_eq!(train_x, expected.train().x());
    assert_eq!(train_y, expected.train().y());
    assert_eq!(val_x, expected.val().x());
    assert_eq!(val_y, expected.val().y());
}

#[test]
fn streaming_packer_rejects_bad_seq_len() {
    let error = WindowPacker::new(0).unwrap_err();
    assert!(matches!(error, DataError::PackingInvalid { .. }));
}

#[test]
fn streaming_packer_without_finish_keeps_partial_window() {
    let mut packer = WindowPacker::new(2).unwrap_or_else(|e| panic!("{e}"));
    packer.push(&[1, 2]);
    assert_eq!(packer.windows().len(), 0);
    packer.push(&[3, 4, 5, 6, 7]);
    assert_eq!(packer.windows().len(), 2);
    assert_eq!(packer.windows()[0], &[1, 2, 3]);
    assert_eq!(packer.windows()[1], &[4, 5, 6]);
    assert_eq!(packer.finish(), 1);
}
