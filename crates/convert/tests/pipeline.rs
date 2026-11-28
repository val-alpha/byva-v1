//! End-to-end converter pipeline tests against synthetic PyTorch-format
//! checkpoints built in Rust (no binary fixtures committed, per D-0007).
//!
//! The builders mirror the exact `torch.save` structure observed in real
//! byva-v1 checkpoints: stored zip entries prefixed with the archive stem,
//! protocol-2 pickle, `collections.OrderedDict` model with a `_metadata`
//! state applied via `BUILD`, and `torch._utils._rebuild_tensor_v2` tensors
//! referencing `data/<key>` storages through `BINPERSID`.

use byva_convert::{Checkpoint, ConvertError, checkpoint_to_weights};
use byva_model::ModelConfig;
use byva_runtime::InferenceModel;
use byva_runtime::weights::canonical_tensor_layout;

fn test_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 32_768,
        d_model: 8,
        n_layers: 1,
        n_heads: 2,
        ffn_hidden: 8,
        max_seq_len: 8,
        tie_embeddings: true,
    }
}

fn fill_value(index: usize) -> f32 {
    let hashed = (index as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_shr(40);
    ((hashed % 4096) as f32 / 4096.0 - 0.5) * 0.02
}

fn canonical() -> Vec<(String, u64)> {
    canonical_tensor_layout(&test_config()).expect("test config is valid")
}

fn sample_tensors() -> Vec<(String, Vec<f32>)> {
    canonical()
        .into_iter()
        .enumerate()
        .map(|(tensor_index, (name, elements))| {
            let values = (0..elements as usize)
                .map(|index| fill_value((tensor_index * 31 + index) % 4096))
                .collect();
            (name, values)
        })
        .collect()
}

#[derive(Default)]
struct PickleWriter {
    bytes: Vec<u8>,
}

impl PickleWriter {
    fn raw(&mut self, chunk: &[u8]) {
        self.bytes.extend_from_slice(chunk);
    }

    fn unicode(&mut self, text: &str) {
        self.bytes.push(0x58);
        self.bytes
            .extend_from_slice(&(text.len() as u32).to_le_bytes());
        self.bytes.extend_from_slice(text.as_bytes());
    }

    fn int1(&mut self, value: u8) {
        self.bytes.push(0x4B);
        self.bytes.push(value);
    }

    fn int2(&mut self, value: u16) {
        self.bytes.push(0x4D);
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn int4(&mut self, value: i32) {
        self.bytes.push(0x4A);
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn boolean(&mut self, value: bool) {
        self.bytes.push(if value { 0x88 } else { 0x89 });
    }

    fn float(&mut self, value: f64) {
        self.bytes.push(0x47);
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn none(&mut self) {
        self.bytes.push(0x4E);
    }

    fn global(&mut self, module: &str, name: &str) {
        self.bytes.push(0x63);
        self.raw(module.as_bytes());
        self.bytes.push(b'\n');
        self.raw(name.as_bytes());
        self.bytes.push(b'\n');
    }

    fn mark(&mut self) {
        self.bytes.push(0x28);
    }

    fn tuple(&mut self) {
        self.bytes.push(0x74);
    }

    fn empty_tuple(&mut self) {
        self.bytes.push(0x29);
    }

    fn empty_dict(&mut self) {
        self.bytes.push(0x7D);
    }

    fn empty_list(&mut self) {
        self.bytes.push(0x5D);
    }

    fn reduce(&mut self) {
        self.bytes.push(0x52);
    }

    fn binpersid(&mut self) {
        self.bytes.push(0x51);
    }

    fn setitem(&mut self) {
        self.bytes.push(0x73);
    }

    fn setitems(&mut self) {
        self.bytes.push(0x75);
    }

    fn appends(&mut self) {
        self.bytes.push(0x65);
    }

    fn build(&mut self) {
        self.bytes.push(0x62);
    }

    fn tensor_ref(&mut self, key: &str, dtype: &str, numel: usize, shape: &[u64], stride: &[u64]) {
        self.global("torch._utils", "_rebuild_tensor_v2");
        self.mark();
        self.mark();
        self.unicode("storage");
        self.global("torch", dtype);
        self.unicode(key);
        self.unicode("cpu");
        self.int4(numel as i32);
        self.tuple();
        self.binpersid();
        self.int1(0);
        self.mark();
        for &extent in shape {
            self.int4(extent as i32);
        }
        self.tuple();
        self.mark();
        for &step in stride {
            self.int4(step as i32);
        }
        self.tuple();
        self.boolean(false);
        self.global("collections", "OrderedDict");
        self.empty_tuple();
        self.reduce();
        self.tuple();
        self.reduce();
    }
}

fn stored_zip(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    let mut offsets = Vec::new();
    for (name, data) in entries {
        offsets.push((name.as_bytes(), data.len() as u32, out.len() as u32));
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0]);
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
    }
    let cd_start = out.len() as u32;
    for (name, size, offset) in &offsets {
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&[20, 0]);
        central.extend_from_slice(&[20, 0]);
        central.extend_from_slice(&[0, 0]);
        central.extend_from_slice(&[0, 0]);
        central.extend_from_slice(&[0, 0, 0, 0]);
        central.extend_from_slice(&[0, 0, 0, 0]);
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0, 0]);
        central.extend_from_slice(&[0, 0]);
        central.extend_from_slice(&[0, 0]);
        central.extend_from_slice(&[0, 0]);
        central.extend_from_slice(&[0, 0, 0, 0]);
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name);
    }
    out.extend_from_slice(&central);
    let cd_size = (out.len() as u32).wrapping_sub(cd_start);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(&(offsets.len() as u16).to_le_bytes());
    out.extend_from_slice(&(offsets.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_start.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out
}

fn checkpoint_pickle(
    format: &str,
    step: u32,
    tensors: &[(String, Vec<f32>)],
    scaler: bool,
) -> Vec<u8> {
    let mut w = PickleWriter::default();
    w.raw(&[0x80, 0x02]);
    w.empty_dict();
    w.mark();

    w.unicode("format");
    w.unicode(format);

    w.unicode("step");
    w.int4(step as i32);

    w.unicode("config");
    w.empty_dict();
    w.mark();
    w.unicode("vocab_size");
    w.int2(32_768);
    w.unicode("d_model");
    w.int1(8);
    w.unicode("n_layers");
    w.int1(1);
    w.unicode("n_heads");
    w.int1(2);
    w.unicode("ffn_hidden");
    w.int1(8);
    w.unicode("max_seq_len");
    w.int1(8);
    w.unicode("tie_embeddings");
    w.boolean(true);
    w.setitems();

    w.unicode("model");
    w.global("collections", "OrderedDict");
    w.empty_tuple();
    w.reduce();
    w.mark();
    for (index, (name, values)) in tensors.iter().enumerate() {
        w.unicode(name);
        w.tensor_ref(
            &index.to_string(),
            "FloatStorage",
            values.len(),
            &[values.len() as u64],
            &[1],
        );
    }
    w.setitems();
    w.empty_dict();
    w.unicode("_metadata");
    w.global("collections", "OrderedDict");
    w.empty_tuple();
    w.reduce();
    w.mark();
    w.unicode("");
    w.empty_dict();
    w.unicode("version");
    w.int1(3);
    w.setitem();
    w.setitems();
    w.setitem();
    w.build();

    w.unicode("optimizer");
    w.empty_dict();

    w.unicode("rng_torch");
    w.tensor_ref("rng", "ByteStorage", 4, &[4], &[1]);

    w.unicode("rng_python");
    w.empty_tuple();

    w.unicode("loss_history");
    w.empty_list();
    w.mark();
    w.float(1.0);
    w.float(2.0);
    w.appends();

    w.unicode("scaler");
    if scaler {
        w.empty_dict();
    } else {
        w.none();
    }

    w.setitems();
    w.bytes.push(0x2E);
    w.bytes
}

fn checkpoint_zip(format: &str, tensors: &[(String, Vec<f32>)], scaler: bool) -> Vec<u8> {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    entries.push((
        "ckpt/data.pkl".to_string(),
        checkpoint_pickle(format, 5, tensors, scaler),
    ));
    for (index, (_, values)) in tensors.iter().enumerate() {
        let storage: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        entries.push((format!("ckpt/data/{index}"), storage));
    }
    entries.push(("ckpt/data/rng".to_string(), vec![1, 2, 3, 4]));
    let borrowed: Vec<(&str, Vec<u8>)> = entries
        .iter()
        .map(|(name, data)| (name.as_str(), data.clone()))
        .collect();
    stored_zip(&borrowed)
}

#[test]
fn synthetic_v2_checkpoint_converts_and_runs() {
    let tensors = sample_tensors();
    let zip = checkpoint_zip("byva-train-checkpoint-v2", &tensors, false);
    let checkpoint =
        Checkpoint::from_bytes(&zip).unwrap_or_else(|error| panic!("parse failed: {error}"));
    assert_eq!(checkpoint.format, "byva-train-checkpoint-v2");
    assert_eq!(checkpoint.step, 5);
    assert_eq!(checkpoint.config, test_config());
    assert_eq!(checkpoint.model.len(), tensors.len());

    let weights = checkpoint_to_weights(&checkpoint)
        .unwrap_or_else(|error| panic!("convert failed: {error}"));
    let file = weights.to_bytes();
    let reloaded = byva_runtime::Weights::from_bytes(&file)
        .unwrap_or_else(|error| panic!("reload failed: {error}"));
    assert_eq!(reloaded.config(), test_config());
    for (name, values) in &tensors {
        assert_eq!(reloaded.tensor(name), Some(values.as_slice()));
    }

    let model = InferenceModel::from_weights(reloaded)
        .unwrap_or_else(|error| panic!("model build failed: {error}"));
    let mut session = model.new_session();
    for token in 0..test_config().max_seq_len {
        let logits = session
            .step(token)
            .unwrap_or_else(|error| panic!("forward failed: {error}"));
        assert_eq!(logits.len(), test_config().vocab_size as usize);
    }
    assert!(session.greedy_token() < test_config().vocab_size);
}

#[test]
fn synthetic_v1_checkpoint_without_scaler_converts() {
    let tensors = sample_tensors();
    let zip = checkpoint_zip("byva-train-checkpoint-v1", &tensors, false);
    let checkpoint =
        Checkpoint::from_bytes(&zip).unwrap_or_else(|error| panic!("parse failed: {error}"));
    assert_eq!(checkpoint.format, "byva-train-checkpoint-v1");
    assert!(checkpoint_to_weights(&checkpoint).is_ok());
}

#[test]
fn v1_with_scaler_is_rejected() {
    let tensors = sample_tensors();
    let zip = checkpoint_zip("byva-train-checkpoint-v1", &tensors, true);
    assert!(matches!(
        Checkpoint::from_bytes(&zip),
        Err(ConvertError::V1WithScaler)
    ));
}

#[test]
fn unknown_format_is_rejected() {
    let tensors = sample_tensors();
    let zip = checkpoint_zip("byva-train-checkpoint-v3", &tensors, false);
    assert!(matches!(
        Checkpoint::from_bytes(&zip),
        Err(ConvertError::UnsupportedFormat { .. })
    ));
}

#[test]
fn unknown_model_tensor_is_rejected() {
    let mut tensors = sample_tensors();
    tensors.push(("blocks.0.surprise.weight".to_string(), vec![0.5; 8]));
    let zip = checkpoint_zip("byva-train-checkpoint-v2", &tensors, false);
    match Checkpoint::from_bytes(&zip) {
        Ok(checkpoint) => assert!(matches!(
            checkpoint_to_weights(&checkpoint),
            Err(ConvertError::UnknownModelTensor { .. })
        )),
        Err(error) => panic!("parse should succeed, got {error}"),
    }
}

#[test]
fn missing_model_tensor_is_rejected() {
    let mut tensors = sample_tensors();
    tensors.remove(4);
    let zip = checkpoint_zip("byva-train-checkpoint-v2", &tensors, false);
    let checkpoint =
        Checkpoint::from_bytes(&zip).unwrap_or_else(|error| panic!("parse failed: {error}"));
    assert!(matches!(
        checkpoint_to_weights(&checkpoint),
        Err(ConvertError::MissingModelTensor { .. })
    ));
}

#[test]
fn non_f32_tensor_is_rejected() {
    let tensors = sample_tensors();
    let (_, values) = &tensors[4];
    let double: Vec<u8> = values
        .iter()
        .flat_map(|value| f64::from(*value).to_le_bytes())
        .collect();
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    entries.push((
        "ckpt/data.pkl".to_string(),
        double_typed_pickle(&tensors, 4, &double),
    ));
    for (index, (_, stored)) in tensors.iter().enumerate() {
        let bytes: Vec<u8> = if index == 4 {
            double.clone()
        } else {
            stored
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        entries.push((format!("ckpt/data/{index}"), bytes));
    }
    entries.push(("ckpt/data/rng".to_string(), vec![1, 2, 3, 4]));
    let borrowed: Vec<(&str, Vec<u8>)> = entries
        .iter()
        .map(|(name, data)| (name.as_str(), data.clone()))
        .collect();
    let zip = stored_zip(&borrowed);
    let checkpoint =
        Checkpoint::from_bytes(&zip).unwrap_or_else(|error| panic!("parse failed: {error}"));
    assert!(matches!(
        checkpoint_to_weights(&checkpoint),
        Err(ConvertError::TensorDType { .. })
    ));
}

fn double_typed_pickle(
    tensors: &[(String, Vec<f32>)],
    double_index: usize,
    _double: &[u8],
) -> Vec<u8> {
    let mut w = PickleWriter::default();
    w.raw(&[0x80, 0x02]);
    w.empty_dict();
    w.mark();
    w.unicode("format");
    w.unicode("byva-train-checkpoint-v2");
    w.unicode("step");
    w.int4(5);
    w.unicode("config");
    w.empty_dict();
    w.mark();
    w.unicode("vocab_size");
    w.int2(32_768);
    w.unicode("d_model");
    w.int1(8);
    w.unicode("n_layers");
    w.int1(1);
    w.unicode("n_heads");
    w.int1(2);
    w.unicode("ffn_hidden");
    w.int1(8);
    w.unicode("max_seq_len");
    w.int1(8);
    w.unicode("tie_embeddings");
    w.boolean(true);
    w.setitems();
    w.unicode("model");
    w.global("collections", "OrderedDict");
    w.empty_tuple();
    w.reduce();
    w.mark();
    for (index, (name, values)) in tensors.iter().enumerate() {
        w.unicode(name);
        if index == double_index {
            w.tensor_ref(
                &index.to_string(),
                "DoubleStorage",
                values.len(),
                &[values.len() as u64],
                &[1],
            );
        } else {
            w.tensor_ref(
                &index.to_string(),
                "FloatStorage",
                values.len(),
                &[values.len() as u64],
                &[1],
            );
        }
    }
    w.setitems();
    w.unicode("optimizer");
    w.empty_dict();
    w.unicode("rng_torch");
    w.tensor_ref("rng", "ByteStorage", 4, &[4], &[1]);
    w.unicode("rng_python");
    w.empty_tuple();
    w.unicode("loss_history");
    w.empty_list();
    w.unicode("scaler");
    w.none();
    w.setitems();
    w.bytes.push(0x2E);
    w.bytes
}

#[test]
fn non_finite_tensor_is_rejected() {
    let mut tensors = sample_tensors();
    tensors[0].1[123] = f32::NAN;
    let zip = checkpoint_zip("byva-train-checkpoint-v2", &tensors, false);
    let checkpoint =
        Checkpoint::from_bytes(&zip).unwrap_or_else(|error| panic!("parse failed: {error}"));
    assert!(matches!(
        checkpoint_to_weights(&checkpoint),
        Err(ConvertError::NonFiniteTensor { name }) if name == "token_embedding.weight"
    ));
}

#[test]
fn short_storage_is_rejected() {
    let tensors = sample_tensors();
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    entries.push((
        "ckpt/data.pkl".to_string(),
        checkpoint_pickle("byva-train-checkpoint-v2", 5, &tensors, false),
    ));
    for (index, (_, values)) in tensors.iter().enumerate() {
        let mut storage: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        if index == 2 {
            storage.pop();
            storage.pop();
        }
        entries.push((format!("ckpt/data/{index}"), storage));
    }
    let borrowed: Vec<(&str, Vec<u8>)> = entries
        .iter()
        .map(|(name, data)| (name.as_str(), data.clone()))
        .collect();
    let zip = stored_zip(&borrowed);
    assert!(matches!(
        Checkpoint::from_bytes(&zip),
        Err(ConvertError::Pickle { .. })
    ));
}

#[test]
fn truncated_zip_is_rejected() {
    let tensors = sample_tensors();
    let zip = checkpoint_zip("byva-train-checkpoint-v2", &tensors, false);
    let truncated = &zip[..zip.len() / 2];
    assert!(Checkpoint::from_bytes(truncated).is_err());
}

#[test]
fn non_mapping_payload_is_rejected() {
    let mut w = PickleWriter::default();
    w.raw(&[0x80, 0x02]);
    w.empty_list();
    w.bytes.push(0x2E);
    let zip = stored_zip(&[("ckpt/data.pkl", w.bytes)]);
    assert!(matches!(
        Checkpoint::from_bytes(&zip),
        Err(ConvertError::NotAMapping)
    ));
}
