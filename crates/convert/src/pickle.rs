//! Minimal pickle virtual machine for PyTorch checkpoint payloads.
//!
//! Supports the protocol-2 opcode subset emitted by `torch.save` for the
//! byva-v1 checkpoint structure (verified by disassembling real archives):
//! scalar/unicode/dict/list/tuple values, memoization, `GLOBAL`/`REDUCE`
//! for `OrderedDict` and `_rebuild_tensor_v2`, `BINPERSID` storages, and
//! `BUILD` state application. Everything else is a typed error, never a
//! panic; all reads are bounds-checked.

use std::collections::HashMap;

use crate::error::ConvertError;

/// Storage element dtypes referenced by checkpoint tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// `torch.FloatStorage` (f32 little-endian).
    Float,
    /// `torch.DoubleStorage` (f64 little-endian).
    Double,
    /// `torch.LongStorage` (i64 little-endian).
    Long,
    /// `torch.IntStorage` (i32 little-endian).
    Int,
    /// `torch.ByteStorage` (u8).
    Byte,
    /// `torch.BoolStorage` (u8, 0/1).
    Bool,
}

impl DType {
    /// Byte size of one element.
    #[must_use]
    pub const fn item_size(self) -> u64 {
        match self {
            Self::Float | Self::Int => 4,
            Self::Double | Self::Long => 8,
            Self::Byte | Self::Bool => 1,
        }
    }

    /// Human-readable dtype name for diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Float => "FloatStorage",
            Self::Double => "DoubleStorage",
            Self::Long => "LongStorage",
            Self::Int => "IntStorage",
            Self::Byte => "ByteStorage",
            Self::Bool => "BoolStorage",
        }
    }
}

/// A materialized tensor reconstructed from storage + rebuild metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorData {
    /// Storage element dtype.
    pub dtype: DType,
    /// Tensor shape.
    pub shape: Vec<u64>,
    /// Raw little-endian element bytes (dense, element order per shape).
    pub data: Vec<u8>,
}

impl TensorData {
    /// Element count (`product(shape)`).
    #[must_use]
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// The tensor's f32 view; valid only for [`DType::Float`] tensors.
    #[must_use]
    pub fn f32_elements(&self) -> Vec<f32> {
        self.data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }
}

/// A resolved persistent storage reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageRef {
    /// Storage key (entry suffix `data/<key>`).
    pub key: String,
    /// Storage dtype class.
    pub dtype: DType,
    /// Storage element count.
    pub numel: u64,
}

/// Parsed pickle value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `None`.
    None,
    /// `bool`.
    Bool(bool),
    /// Python int (i128 covers every observed checkpoint range).
    Int(i128),
    /// Python float.
    Float(f64),
    /// Python `str`.
    Str(String),
    /// Python `list`.
    List(Vec<Value>),
    /// Python `tuple`.
    Tuple(Vec<Value>),
    /// Python `dict` / `OrderedDict` (insertion order preserved).
    Dict(Vec<(Value, Value)>),
    /// A class or function reference (`module.name`).
    ClassRef {
        /// Module path.
        module: String,
        /// Class or function name.
        name: String,
    },
    /// Result of reducing a class with arguments (non-special classes).
    Object {
        /// Class that was reduced.
        class: Box<Value>,
        /// Constructor arguments.
        args: Box<Value>,
        /// State applied by `BUILD`, if any.
        state: Option<Box<Value>>,
    },
    /// A persistent storage reference (`BINPERSID`).
    Storage(StorageRef),
    /// A tensor rebuilt via `_rebuild_tensor_v2`/`_rebuild_tensor`.
    Tensor(TensorData),
}

impl Value {
    /// Description of the value kind for error messages.
    #[must_use]
    pub fn kind(&self) -> String {
        match self {
            Self::None => "None".to_string(),
            Self::Bool(_) => "bool".to_string(),
            Self::Int(_) => "int".to_string(),
            Self::Float(_) => "float".to_string(),
            Self::Str(value) => format!("str {value:?}"),
            Self::List(_) => "list".to_string(),
            Self::Tuple(_) => "tuple".to_string(),
            Self::Dict(_) => "dict".to_string(),
            Self::ClassRef { module, name } => format!("class {module}.{name}"),
            Self::Object { .. } => "object".to_string(),
            Self::Storage(_) => "storage".to_string(),
            Self::Tensor(_) => "tensor".to_string(),
        }
    }

    /// The string value, if this is a `Str`.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(value) => Some(value),
            _ => None,
        }
    }

    /// The int value, if this is an `Int`.
    #[must_use]
    pub fn as_int(&self) -> Option<i128> {
        match self {
            Self::Int(value) => Some(*value),
            _ => None,
        }
    }

    /// The dict entries, if this is a `Dict`.
    #[must_use]
    pub fn as_dict(&self) -> Option<&[(Value, Value)]> {
        match self {
            Self::Dict(entries) => Some(entries),
            _ => None,
        }
    }

    /// The tensor, if this is a `Tensor`.
    #[must_use]
    pub fn as_tensor(&self) -> Option<&TensorData> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }
}

/// A storage resolver supplied by the checkpoint layer: maps a persistent id
/// tuple `('storage', dtype_class, key, location, numel)` to raw bytes.
pub trait StorageResolver {
    /// Returns the raw element bytes for the referenced storage.
    ///
    /// # Errors
    /// Fails when the storage entry is missing or its byte length mismatches
    /// the declared element count.
    fn resolve(&self, reference: &StorageRef) -> Result<Vec<u8>, ConvertError>;
}

struct Machine<'a> {
    bytes: &'a [u8],
    position: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: HashMap<usize, Value>,
    resolver: &'a dyn StorageResolver,
}

/// Parses a pickle stream into a [`Value`] tree.
///
/// # Errors
/// Fails on malformed streams and unsupported opcodes with byte positions.
pub fn parse(bytes: &[u8], resolver: &dyn StorageResolver) -> Result<Value, ConvertError> {
    let mut machine = Machine {
        bytes,
        position: 0,
        stack: Vec::new(),
        marks: Vec::new(),
        memo: HashMap::new(),
        resolver,
    };
    machine.run()
}

impl<'a> Machine<'a> {
    fn error(&self, detail: impl Into<String>) -> ConvertError {
        ConvertError::Pickle {
            position: self.position as u64,
            detail: detail.into(),
        }
    }

    fn run(&mut self) -> Result<Value, ConvertError> {
        loop {
            let Some(opcode) = self.peek_u8() else {
                return Err(self.error("stream ended without STOP"));
            };
            self.position += 1;
            match opcode {
                0x2E => {
                    let result = self.pop()?;
                    if !self.stack.is_empty() {
                        return Err(self.error("stack not empty after STOP"));
                    }
                    return Ok(result);
                }
                0x80 => {
                    self.read_u8();
                }
                0x7D => self.push(Value::Dict(Vec::new()))?,
                0x5D => self.push(Value::List(Vec::new()))?,
                0x29 => self.push(Value::Tuple(Vec::new()))?,
                0x28 => self.marks.push(self.stack.len()),
                0x74 => {
                    let items = self.take_marked()?;
                    self.push(Value::Tuple(items))?;
                }
                0x85 => {
                    let value = self.pop()?;
                    self.push(Value::Tuple(vec![value]))?;
                }
                0x86 => {
                    let (right, left) = (self.pop()?, self.pop()?);
                    self.push(Value::Tuple(vec![left, right]))?;
                }
                0x87 => {
                    let (third, second, first) = (self.pop()?, self.pop()?, self.pop()?);
                    self.push(Value::Tuple(vec![first, second, third]))?;
                }
                0x58 | 0x8C => {
                    let len = if opcode == 0x58 {
                        u64::from(self.read_u32())
                    } else {
                        u64::from(self.read_u8())
                    };
                    let text = self.read_str(len)?;
                    self.push(Value::Str(text))?;
                }
                0x71 => {
                    let index = u64::from(self.read_u8());
                    self.memoize(index)?;
                }
                0x72 => {
                    let index = u64::from(self.read_u32());
                    self.memoize(index)?;
                }
                0x94 => self.memoize(self.memo.len() as u64)?,
                0x68 => {
                    let index = u64::from(self.read_u8());
                    let value = self.memo_get(index)?;
                    self.push(value)?;
                }
                0x6A => {
                    let index = u64::from(self.read_u32());
                    let value = self.memo_get(index)?;
                    self.push(value)?;
                }
                0x4B => {
                    let value = self.read_u8();
                    self.push(Value::Int(i128::from(value)))?;
                }
                0x4D => {
                    let value = self.read_u16();
                    self.push(Value::Int(i128::from(value)))?;
                }
                0x4A => {
                    let value = self.read_i32();
                    self.push(Value::Int(i128::from(value)))?;
                }
                0x8A => {
                    let len = u64::from(self.read_u8());
                    let value = self.read_long(len)?;
                    self.push(Value::Int(value))?;
                }
                0x8B => {
                    let len = u64::from(self.read_u32());
                    let value = self.read_long(len)?;
                    self.push(Value::Int(value))?;
                }
                0x47 => {
                    let value = self.read_f64();
                    self.push(Value::Float(value))?;
                }
                0x88 => self.push(Value::Bool(true))?,
                0x89 => self.push(Value::Bool(false))?,
                0x4E => self.push(Value::None)?,
                0x63 => {
                    let module = self.read_line()?;
                    let name = self.read_line()?;
                    self.push(Value::ClassRef { module, name })?;
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (Some(module), Some(name)) = (module.as_str(), name.as_str()) else {
                        return Err(self.error("STACK_GLOBAL operands must be strings"));
                    };
                    self.push(Value::ClassRef {
                        module: module.to_string(),
                        name: name.to_string(),
                    })?;
                }
                0x52 => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let reduced = self.reduce(callable, args)?;
                    self.push(reduced)?;
                }
                0x51 => {
                    let pid = self.pop()?;
                    let resolved = self.persistent_load(pid)?;
                    self.push(resolved)?;
                }
                0x62 => {
                    let state = self.pop()?;
                    let object = self.pop()?;
                    let built = self.build(object, state)?;
                    self.push(built)?;
                }
                0x30 => {
                    self.pop()?;
                }
                0x31 => {
                    let items = self.take_marked()?;
                    for item in items {
                        self.stack.push(item);
                    }
                }
                0x61 => {
                    let item = self.pop()?;
                    self.append(item)?;
                }
                0x65 => {
                    let items = self.take_marked()?;
                    for item in items {
                        self.append(item)?;
                    }
                }
                0x73 => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    self.set_item(key, value)?;
                }
                0x75 => {
                    let items = self.take_marked()?;
                    if items.len() % 2 != 0 {
                        return Err(self.error("SETITEMS expects key/value pairs"));
                    }
                    let mut pairs = items.into_iter();
                    while let (Some(key), Some(value)) = (pairs.next(), pairs.next()) {
                        self.set_item(key, value)?;
                    }
                }
                other => {
                    return Err(ConvertError::UnsupportedOpcode {
                        opcode: other,
                        position: self.position as u64 - 1,
                    });
                }
            }
        }
    }

    fn push(&mut self, value: Value) -> Result<(), ConvertError> {
        self.stack.push(value);
        Ok(())
    }

    fn pop(&mut self) -> Result<Value, ConvertError> {
        self.stack
            .pop()
            .ok_or_else(|| self.error("stack underflow"))
    }

    fn take_marked(&mut self) -> Result<Vec<Value>, ConvertError> {
        let mark = self
            .marks
            .pop()
            .ok_or_else(|| self.error("MARK expected"))?;
        if mark > self.stack.len() {
            return Err(self.error("MARK beyond stack top"));
        }
        Ok(self.stack.split_off(mark))
    }

    fn memoize(&mut self, index: u64) -> Result<(), ConvertError> {
        let Ok(index) = usize::try_from(index) else {
            return Err(self.error("memo index out of range"));
        };
        let value = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| self.error("memoize on empty stack"))?;
        self.memo.insert(index, value);
        Ok(())
    }

    fn memo_get(&self, index: u64) -> Result<Value, ConvertError> {
        let Ok(index) = usize::try_from(index) else {
            return Err(self.error("memo index out of range"));
        };
        self.memo
            .get(&index)
            .cloned()
            .ok_or_else(|| self.error(format!("memo index {index} not set")))
    }

    fn append(&mut self, item: Value) -> Result<(), ConvertError> {
        match self.stack.last_mut() {
            Some(Value::List(items)) => items.push(item),
            _ => return Err(self.error("APPEND target is not a list")),
        }
        Ok(())
    }

    fn set_item(&mut self, key: Value, value: Value) -> Result<(), ConvertError> {
        match self.stack.last_mut() {
            Some(Value::Dict(entries)) => {
                if let Some(slot) = entries.iter_mut().find(|(existing, _)| *existing == key) {
                    slot.1 = value;
                } else {
                    entries.push((key, value));
                }
            }
            _ => return Err(self.error("SETITEM target is not a dict")),
        }
        Ok(())
    }

    fn reduce(&mut self, callable: Value, args: Value) -> Result<Value, ConvertError> {
        let Value::ClassRef { module, name } = callable else {
            return Err(self.error("REDUCE target is not a class reference"));
        };
        match (module.as_str(), name.as_str()) {
            ("collections", "OrderedDict") => Ok(Value::Dict(Vec::new())),
            ("torch._utils", "_rebuild_tensor_v2" | "rebuild_tensor_v2") => {
                self.rebuild_tensor(args, true)
            }
            ("torch._utils", "_rebuild_tensor" | "rebuild_tensor") => {
                self.rebuild_tensor(args, false)
            }
            _ => Ok(Value::Object {
                class: Box::new(Value::ClassRef { module, name }),
                args: Box::new(args),
                state: None,
            }),
        }
    }

    fn rebuild_tensor(&mut self, args: Value, with_hooks: bool) -> Result<Value, ConvertError> {
        let Value::Tuple(fields) = args else {
            return Err(self.error("tensor rebuild arguments must be a tuple"));
        };
        let expected = if with_hooks { 6 } else { 4 };
        if fields.len() != expected {
            return Err(self.error(format!(
                "tensor rebuild expects {expected} arguments, found {}",
                fields.len()
            )));
        }
        let mut fields = fields.into_iter();
        let storage = fields.next().ok_or_else(|| self.error("missing storage"))?;
        let Value::Storage(reference) = storage else {
            return Err(self.error("tensor rebuild storage must be a persistent reference"));
        };
        let storage_offset = fields
            .next()
            .and_then(|value| value.as_int())
            .ok_or_else(|| self.error("storage offset must be an int"))?;
        let Some(size) = self.int_list(fields.next()) else {
            return Err(self.error("tensor size must be an int list"));
        };
        let Some(stride) = self.int_list(fields.next()) else {
            return Err(self.error("tensor stride must be an int list"));
        };
        let _requires_grad = fields.next();
        if with_hooks {
            let _backward_hooks = fields.next();
        }

        let raw = self.resolver.resolve(&reference)?;
        let item = reference.dtype.item_size();
        let numel = reference.numel;
        let byte_len = numel
            .checked_mul(item)
            .ok_or_else(|| self.error("storage element count overflows"))?;
        if raw.len() as u64 != byte_len {
            return Err(self.error(format!(
                "storage {} has {} bytes, expected {byte_len}",
                reference.key,
                raw.len()
            )));
        }
        if size.len() != stride.len() {
            return Err(self.error("size and stride ranks differ"));
        }
        let elements: u64 = size.iter().product();
        let mut dense = vec![
            0u8;
            elements
                .checked_mul(item)
                .ok_or_else(|| self.error("tensor element count overflows"))?
                as usize
        ];
        if size.contains(&0) {
            return Ok(Value::Tensor(TensorData {
                dtype: reference.dtype,
                shape: size,
                data: dense,
            }));
        }
        let rank = size.len();
        let mut index = vec![0u64; rank];
        for chunk in dense.chunks_mut(item as usize) {
            let mut flat = storage_offset.max(0) as u64;
            for (axis, &position) in index.iter().enumerate() {
                flat = flat
                    .checked_add(
                        position
                            .checked_mul(stride[axis])
                            .ok_or_else(|| self.error("tensor index computation overflows"))?,
                    )
                    .ok_or_else(|| self.error("tensor index computation overflows"))?;
            }
            if flat >= numel {
                return Err(self.error("tensor references storage beyond its extent"));
            }
            let start = (flat * item) as usize;
            chunk.copy_from_slice(&raw[start..start + item as usize]);
            for axis in (0..rank).rev() {
                index[axis] += 1;
                if index[axis] < size[axis] {
                    break;
                }
                index[axis] = 0;
                if axis == 0 {
                    break;
                }
            }
        }
        Ok(Value::Tensor(TensorData {
            dtype: reference.dtype,
            shape: size,
            data: dense,
        }))
    }

    fn int_list(&self, value: Option<Value>) -> Option<Vec<u64>> {
        let value = value?;
        let items = match value {
            Value::List(items) | Value::Tuple(items) => items,
            _ => return None,
        };
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let parsed = item.as_int()?;
            if parsed < 0 {
                return None;
            }
            out.push(u64::try_from(parsed).ok()?);
        }
        Some(out)
    }

    fn build(&mut self, object: Value, state: Value) -> Result<Value, ConvertError> {
        match object {
            Value::Dict(entries) => {
                let Value::Dict(new_entries) = state else {
                    return Err(self.error("BUILD state for a dict must be a dict"));
                };
                let mut merged = entries;
                for (key, value) in new_entries {
                    if let Some(slot) = merged.iter_mut().find(|(existing, _)| *existing == key) {
                        slot.1 = value;
                    } else {
                        merged.push((key, value));
                    }
                }
                Ok(Value::Dict(merged))
            }
            Value::Object { class, args, .. } => Ok(Value::Object {
                class,
                args,
                state: Some(Box::new(state)),
            }),
            _ => Err(self.error("BUILD target is not reducible")),
        }
    }

    fn persistent_load(&mut self, pid: Value) -> Result<Value, ConvertError> {
        let Value::Tuple(fields) = pid else {
            return Err(self.error("persistent id must be a tuple"));
        };
        if fields.len() != 5 {
            return Err(self.error(format!(
                "persistent storage id expects 5 fields, found {}",
                fields.len()
            )));
        }
        if fields[0].as_str() != Some("storage") {
            return Err(self.error("unsupported persistent id kind"));
        }
        let Value::ClassRef { name, .. } = &fields[1] else {
            return Err(self.error("storage dtype must be a class reference"));
        };
        let dtype = match name.as_str() {
            "FloatStorage" => DType::Float,
            "DoubleStorage" => DType::Double,
            "LongStorage" => DType::Long,
            "IntStorage" => DType::Int,
            "ByteStorage" => DType::Byte,
            "BoolStorage" => DType::Bool,
            other => {
                return Err(self.error(format!("unsupported storage dtype {other:?}")));
            }
        };
        let key = fields[2]
            .as_str()
            .ok_or_else(|| self.error("storage key must be a string"))?
            .to_string();
        let numel = fields[4]
            .as_int()
            .ok_or_else(|| self.error("storage numel must be an int"))?;
        let Ok(numel) = u64::try_from(numel) else {
            return Err(self.error("storage numel is negative"));
        };
        Ok(Value::Storage(StorageRef { key, dtype, numel }))
    }

    fn peek_u8(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn read_u8(&mut self) -> u8 {
        let value = self.bytes[self.position];
        self.position += 1;
        value
    }

    fn read_u16(&mut self) -> u16 {
        let low = self.read_u8();
        let high = self.read_u8();
        u16::from_le_bytes([low, high])
    }

    fn read_u32(&mut self) -> u32 {
        let mut chunk = [0u8; 4];
        for slot in &mut chunk {
            *slot = self.read_u8();
        }
        u32::from_le_bytes(chunk)
    }

    fn read_i32(&mut self) -> i32 {
        i32::from_le_bytes(self.read_u32().to_le_bytes())
    }

    fn read_f64(&mut self) -> f64 {
        let mut chunk = [0u8; 8];
        for slot in &mut chunk {
            *slot = self.read_u8();
        }
        f64::from_le_bytes(chunk)
    }

    fn read_long(&mut self, len: u64) -> Result<i128, ConvertError> {
        let Ok(len) = usize::try_from(len) else {
            return Err(self.error("long length out of range"));
        };
        if len > 16 {
            return Err(self.error(format!("long of {len} bytes exceeds i128 support")));
        }
        let mut chunk = [0u8; 16];
        for slot in chunk.iter_mut().take(len) {
            *slot = self.read_u8();
        }
        if len > 0 && chunk[len - 1] & 0x80 != 0 {
            for slot in chunk.iter_mut().skip(len) {
                *slot = 0xFF;
            }
        }
        Ok(i128::from_le_bytes(chunk))
    }

    fn read_str(&mut self, len: u64) -> Result<String, ConvertError> {
        let Ok(len) = usize::try_from(len) else {
            return Err(self.error("string length out of range"));
        };
        let start = self.position;
        let end = start
            .checked_add(len)
            .ok_or_else(|| self.error("string length overflows"))?;
        let chunk = self
            .bytes
            .get(start..end)
            .ok_or_else(|| self.error("unexpected end of stream"))?;
        self.position = end;
        String::from_utf8(chunk.to_vec()).map_err(|_| self.error("string is not valid UTF-8"))
    }

    fn read_line(&mut self) -> Result<String, ConvertError> {
        let start = self.position;
        let mut end = None;
        for (offset, &byte) in self.bytes[start..].iter().enumerate() {
            if byte == b'\n' {
                end = Some(start + offset);
                break;
            }
        }
        let Some(end) = end else {
            return Err(self.error("unterminated string"));
        };
        let chunk = &self.bytes[start..end];
        self.position = end + 1;
        String::from_utf8(chunk.to_vec()).map_err(|_| self.error("string is not valid UTF-8"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedResolver(Vec<u8>);

    impl StorageResolver for FixedResolver {
        fn resolve(&self, reference: &StorageRef) -> Result<Vec<u8>, ConvertError> {
            let expected = reference.numel * reference.dtype.item_size();
            if self.0.len() as u64 != expected {
                return Err(ConvertError::Pickle {
                    position: 0,
                    detail: "resolver byte mismatch".to_string(),
                });
            }
            Ok(self.0.clone())
        }
    }

    struct NopResolver;

    impl StorageResolver for NopResolver {
        fn resolve(&self, _reference: &StorageRef) -> Result<Vec<u8>, ConvertError> {
            Err(ConvertError::NotAMapping)
        }
    }

    fn parse_no_resolver(bytes: &[u8]) -> Result<Value, ConvertError> {
        parse(bytes, &NopResolver)
    }

    fn unicode(bytes: &mut Vec<u8>, text: &str) {
        bytes.push(0x58);
        bytes.extend_from_slice(&(text.len() as u32).to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
    }

    fn int4(bytes: &mut Vec<u8>, value: i32) {
        bytes.push(0x4A);
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn scalars_and_containers_parse() {
        let mut bytes = vec![0x80, 0x02, 0x7D, 0x28];
        unicode(&mut bytes, "count");
        int4(&mut bytes, -7);
        unicode(&mut bytes, "scale");
        bytes.push(0x47);
        bytes.extend_from_slice(&1.5f64.to_le_bytes());
        unicode(&mut bytes, "flag");
        bytes.push(0x88);
        unicode(&mut bytes, "items");
        bytes.push(0x5D);
        bytes.push(0x28);
        bytes.push(0x4B);
        bytes.push(1);
        bytes.push(0x4D);
        bytes.extend_from_slice(&600u16.to_le_bytes());
        bytes.push(0x65);
        bytes.push(0x75);
        bytes.push(0x2E);

        let value = parse_no_resolver(&bytes).expect("parse failed");
        let Value::Dict(entries) = value else {
            panic!("expected dict");
        };
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].1, Value::Int(-7));
        assert_eq!(entries[1].1, Value::Float(1.5));
        assert_eq!(entries[2].1, Value::Bool(true));
        assert_eq!(
            entries[3].1,
            Value::List(vec![Value::Int(1), Value::Int(600)])
        );
    }

    #[test]
    fn memo_round_trips_through_pop_and_get() {
        let mut bytes = vec![0x80, 0x02];
        unicode(&mut bytes, "abc");
        bytes.push(0x71);
        bytes.push(7);
        bytes.push(0x30);
        bytes.push(0x68);
        bytes.push(7);
        bytes.push(0x2E);
        let value = parse_no_resolver(&bytes).expect("parse failed");
        assert_eq!(value, Value::Str("abc".to_string()));
    }

    #[test]
    fn ordered_dict_reduce_and_build_merge() {
        let mut bytes = vec![0x80, 0x02];
        bytes.push(0x63);
        bytes.extend_from_slice(b"collections\nOrderedDict\n");
        bytes.push(0x29);
        bytes.push(0x52);
        bytes.push(0x28);
        unicode(&mut bytes, "k");
        bytes.push(0x4B);
        bytes.push(9);
        bytes.push(0x75);
        bytes.push(0x7D);
        unicode(&mut bytes, "extra");
        bytes.push(0x4B);
        bytes.push(1);
        bytes.push(0x73);
        bytes.push(0x62);
        bytes.push(0x2E);

        let value = parse_no_resolver(&bytes).expect("parse failed");
        let Value::Dict(entries) = value else {
            panic!("expected dict");
        };
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], (Value::Str("k".to_string()), Value::Int(9)));
        assert_eq!(entries[1], (Value::Str("extra".to_string()), Value::Int(1)));
    }

    #[test]
    fn strided_tensor_rebuild() {
        let storage: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut bytes = vec![0x80, 0x02];
        bytes.push(0x63);
        bytes.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");
        bytes.push(0x28);
        bytes.push(0x28);
        unicode(&mut bytes, "storage");
        bytes.push(0x63);
        bytes.extend_from_slice(b"torch\nFloatStorage\n");
        unicode(&mut bytes, "7");
        unicode(&mut bytes, "cpu");
        int4(&mut bytes, 6);
        bytes.push(0x74);
        bytes.push(0x51);
        int4(&mut bytes, 1);
        bytes.push(0x28);
        bytes.push(0x4B);
        bytes.push(2);
        bytes.push(0x4B);
        bytes.push(2);
        bytes.push(0x74);
        bytes.push(0x28);
        bytes.push(0x4B);
        bytes.push(3);
        bytes.push(0x4B);
        bytes.push(1);
        bytes.push(0x74);
        bytes.push(0x89);
        bytes.push(0x63);
        bytes.extend_from_slice(b"collections\nOrderedDict\n");
        bytes.push(0x29);
        bytes.push(0x52);
        bytes.push(0x74);
        bytes.push(0x52);
        bytes.push(0x2E);

        let value = parse(&bytes, &FixedResolver(storage)).expect("parse failed");
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.f32_elements(), vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn unsupported_opcode_is_rejected() {
        let bytes = vec![0x80, 0x02, 0x81, 0x2E];
        match parse_no_resolver(&bytes) {
            Err(ConvertError::UnsupportedOpcode { opcode, position }) => {
                assert_eq!(opcode, 0x81);
                assert_eq!(position, 2);
            }
            other => panic!("expected unsupported opcode, got {other:?}"),
        }
    }

    #[test]
    fn truncated_stream_is_rejected() {
        let bytes = vec![0x80, 0x02];
        assert!(matches!(
            parse_no_resolver(&bytes),
            Err(ConvertError::Pickle { .. })
        ));
    }

    #[test]
    fn negative_long_ints_sign_extend() {
        let mut bytes = vec![0x80, 0x02, 0x8A, 8];
        bytes.extend_from_slice(&(-2i64).to_le_bytes());
        bytes.push(0x2E);
        let value = parse_no_resolver(&bytes).expect("parse failed");
        assert_eq!(value, Value::Int(-2));
    }
}
