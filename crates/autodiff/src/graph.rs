//! Reverse-mode tape: graph construction, forward ops, and backward pass.
//!
//! Ops mirror the frozen D-0013 architecture (pre-norm decoder-only
//! transformer, causal MHA, exact-erf GELU, tied output head). The
//! cross-entropy op is tied-logit-free (K2): logits are computed row by row
//! against the embedding table and the full `[n, vocab]` logits matrix is
//! never materialized. All reduction orders are fixed, and work that is
//! independent per output element is split across a fixed partition of the
//! output, so a given graph evaluates bit-identically for any thread count.

use byva_tensor::Tensor;

use crate::error::AutodiffError;

/// Handle to a node in the tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId {
    index: usize,
}

impl NodeId {
    /// Position of this node in the tape.
    pub fn index(self) -> usize {
        self.index
    }
}

#[derive(Debug, Clone)]
enum Op {
    Leaf,
    MatMul,
    Add,
    Mul,
    Sum,
    AddBroadcastRows,
    MulScalar {
        c: f32,
    },
    Reshape,
    Embed {
        ids: Vec<u32>,
    },
    SliceCols {
        start: usize,
        ncols: usize,
    },
    HeadsToRows,
    RowsToHeads {
        heads: usize,
    },
    BatchedMatMulNT,
    BatchedMatMulNN,
    CausalSoftMax,
    LayerNorm {
        eps: f32,
    },
    Gelu,
    CrossEntropyTied {
        targets: Vec<u32>,
        stats: Vec<CeRowStat>,
    },
}

/// Per-row softmax statistics cached by the forward cross-entropy pass so
/// the backward pass can rebuild probabilities without a separate sweep
/// over the vocabulary.
#[derive(Debug, Clone)]
struct CeRowStat {
    max: f32,
    sum: f64,
    loss: f64,
}

struct Node {
    value: Tensor,
    grad: Option<Vec<f32>>,
    op: Op,
    inputs: Vec<NodeId>,
}

/// A single-use computation tape. Nodes are appended in forward order, which
/// is a topological order, so the backward pass is a reverse scan.
#[derive(Default)]
pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    /// Creates an empty tape.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a leaf node (input or parameter) with the given shape and data.
    pub fn leaf(&mut self, shape: &[usize], data: Vec<f32>) -> Result<NodeId, AutodiffError> {
        let value = Tensor::from_vec(shape, data)?;
        Ok(self.push(Node {
            value,
            grad: None,
            op: Op::Leaf,
            inputs: Vec::new(),
        }))
    }

    /// Adds a leaf node from an existing tensor.
    pub fn leaf_tensor(&mut self, value: Tensor) -> Result<NodeId, AutodiffError> {
        Ok(self.push(Node {
            value,
            grad: None,
            op: Op::Leaf,
            inputs: Vec::new(),
        }))
    }

    /// 2-D matrix product `a [m, k] x b [k, n] -> [m, n]`, f64 accumulation.
    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AutodiffError> {
        let (a_shape, b_shape) = self.two_shapes(a, b)?;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(AutodiffError::OpValidation {
                op: "matmul",
                reason: format!(
                    "both inputs must be rank 2, found {:?} and {:?}",
                    a_shape, b_shape
                ),
            });
        }
        if a_shape[1] != b_shape[0] {
            return Err(AutodiffError::OpValidation {
                op: "matmul",
                reason: format!("inner dimensions differ: {} vs {}", a_shape[1], b_shape[0]),
            });
        }
        let (m, k) = (a_shape[0], a_shape[1]);
        let n = b_shape[1];
        let a_data = self.data_ref(a)?;
        let b_data = self.data_ref(b)?;
        let mut out = vec![0.0f32; m * n];
        batched_nn_rows(&mut out, a_data, b_data, 1, m, k, n);
        self.push_output(Op::MatMul, vec![a, b], &[m, n], out)
    }

    /// Elementwise sum of two same-shape tensors.
    pub fn add(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AutodiffError> {
        let (a_shape, b_shape) = self.two_shapes(a, b)?;
        if a_shape != b_shape {
            return Err(AutodiffError::ShapeMismatch {
                op: "add",
                expected: a_shape.clone(),
                found: b_shape,
            });
        }
        let a_data = self.data(a)?;
        let b_data = self.data(b)?;
        let out: Vec<f32> = a_data.iter().zip(b_data).map(|(x, y)| x + y).collect();
        self.push_output(Op::Add, vec![a, b], &a_shape, out)
    }

    /// Elementwise product of two same-shape tensors.
    pub fn mul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AutodiffError> {
        let (a_shape, b_shape) = self.two_shapes(a, b)?;
        if a_shape != b_shape {
            return Err(AutodiffError::ShapeMismatch {
                op: "mul",
                expected: a_shape.clone(),
                found: b_shape,
            });
        }
        let a_data = self.data(a)?;
        let b_data = self.data(b)?;
        let out: Vec<f32> = a_data.iter().zip(b_data).map(|(x, y)| x * y).collect();
        self.push_output(Op::Mul, vec![a, b], &a_shape, out)
    }

    /// Sums all elements into a scalar.
    pub fn sum(&mut self, x: NodeId) -> Result<NodeId, AutodiffError> {
        let total: f64 = self.data(x)?.iter().map(|&v| f64::from(v)).sum();
        self.push_output(Op::Sum, vec![x], &[], vec![total as f32])
    }

    /// Adds `rows [s, d]` to every block of `x [b * s, d]` treated as
    /// row-major `[b, s, d]`: `y[(b, s_i)] = x[(b, s_i)] + rows[s_i]`.
    /// Used for learned position embeddings.
    pub fn add_broadcast_rows(&mut self, x: NodeId, rows: NodeId) -> Result<NodeId, AutodiffError> {
        let (x_shape, rows_shape) = self.two_shapes(x, rows)?;
        if x_shape.len() != 2 || rows_shape.len() != 2 {
            return Err(AutodiffError::OpValidation {
                op: "add_broadcast_rows",
                reason: format!(
                    "both inputs must be rank 2, found {:?} and {:?}",
                    x_shape, rows_shape
                ),
            });
        }
        if x_shape[1] != rows_shape[1] {
            return Err(AutodiffError::OpValidation {
                op: "add_broadcast_rows",
                reason: format!("column counts differ: {} vs {}", x_shape[1], rows_shape[1]),
            });
        }
        if rows_shape[0] == 0 || x_shape[0] % rows_shape[0] != 0 {
            return Err(AutodiffError::OpValidation {
                op: "add_broadcast_rows",
                reason: format!(
                    "row count {} is not a positive multiple of {}",
                    x_shape[0], rows_shape[0]
                ),
            });
        }
        let s = rows_shape[0];
        let cols = rows_shape[1];
        let mut out = self.data(x)?;
        let rows_data = self.data(rows)?;
        for (index, chunk) in out.chunks_mut(cols).enumerate() {
            let row_index = index % s;
            for (slot, &value) in chunk.iter_mut().zip(&rows_data[row_index * cols..]) {
                *slot += value;
            }
        }
        self.push_output(Op::AddBroadcastRows, vec![x, rows], &x_shape, out)
    }

    /// Multiplies every element by a constant.
    pub fn mul_scalar(&mut self, x: NodeId, c: f32) -> Result<NodeId, AutodiffError> {
        let shape = self.shape(x)?;
        let out: Vec<f32> = self.data(x)?.iter().map(|&v| v * c).collect();
        self.push_output(Op::MulScalar { c }, vec![x], &shape, out)
    }

    /// Reinterprets the same elements under a new shape (row-major).
    pub fn reshape(&mut self, x: NodeId, shape: &[usize]) -> Result<NodeId, AutodiffError> {
        let value = self.node(x.index())?.value.reshape(shape)?;
        let out_shape = value.shape().to_vec();
        let out = value.data().to_vec();
        self.push_output(Op::Reshape, vec![x], &out_shape, out)
    }

    /// Embedding lookup: `table [v, d]` with `ids [n]` -> `[n, d]`.
    /// Gradients scatter-add back into the table.
    pub fn embed(&mut self, table: NodeId, ids: &[u32]) -> Result<NodeId, AutodiffError> {
        let table_shape = self.shape(table)?;
        if table_shape.len() != 2 {
            return Err(AutodiffError::OpValidation {
                op: "embed",
                reason: format!("table must be rank 2, found {:?}", table_shape),
            });
        }
        let vocab = table_shape[0];
        for &id in ids {
            if id as usize >= vocab {
                return Err(AutodiffError::OpValidation {
                    op: "embed",
                    reason: format!("token id {} is out of range for vocab {}", id, vocab),
                });
            }
        }
        let d = table_shape[1];
        let table_data = self.data(table)?;
        let mut out = vec![0.0f32; ids.len() * d];
        for (row, &id) in ids.iter().enumerate() {
            let source = id as usize * d;
            out[row * d..(row + 1) * d].copy_from_slice(&table_data[source..source + d]);
        }
        let out_shape = vec![ids.len(), d];
        self.push_output(
            Op::Embed { ids: ids.to_vec() },
            vec![table],
            &out_shape,
            out,
        )
    }

    /// Selects a contiguous block of columns: `x [n, cols]` -> `[n, ncols]`.
    /// Used to split the fused QKV projection.
    pub fn slice_cols(
        &mut self,
        x: NodeId,
        start: usize,
        ncols: usize,
    ) -> Result<NodeId, AutodiffError> {
        let shape = self.shape(x)?;
        if shape.len() != 2 {
            return Err(AutodiffError::OpValidation {
                op: "slice_cols",
                reason: format!("input must be rank 2, found {:?}", shape),
            });
        }
        if start + ncols > shape[1] {
            return Err(AutodiffError::OpValidation {
                op: "slice_cols",
                reason: format!(
                    "column range {}..{} exceeds width {}",
                    start,
                    start + ncols,
                    shape[1]
                ),
            });
        }
        let (rows, cols) = (shape[0], shape[1]);
        let x_data = self.data(x)?;
        let mut out = vec![0.0f32; rows * ncols];
        for (out_row, in_row) in out.chunks_mut(ncols).zip(x_data.chunks(cols)) {
            out_row.copy_from_slice(&in_row[start..start + ncols]);
        }
        let out_shape = vec![rows, ncols];
        self.push_output(Op::SliceCols { start, ncols }, vec![x], &out_shape, out)
    }

    /// Reorders `x [b, s, h, hd]` into per-head matrices `[b * h, s, hd]`.
    pub fn heads_to_rows(&mut self, x: NodeId, heads: usize) -> Result<NodeId, AutodiffError> {
        let shape = self.shape(x)?;
        if shape.len() != 4 {
            return Err(AutodiffError::OpValidation {
                op: "heads_to_rows",
                reason: format!("input must be rank 4, found {:?}", shape),
            });
        }
        if shape[2] != heads {
            return Err(AutodiffError::ShapeMismatch {
                op: "heads_to_rows",
                expected: vec![shape[0], shape[1], heads, shape[3]],
                found: shape.clone(),
            });
        }
        let (b, s, h, hd) = (shape[0], shape[1], shape[2], shape[3]);
        let x_data = self.data(x)?;
        let mut out = vec![0.0f32; b * s * h * hd];
        for bi in 0..b {
            for hi in 0..h {
                for si in 0..s {
                    let source = ((bi * s + si) * h + hi) * hd;
                    let target = ((bi * h + hi) * s + si) * hd;
                    out[target..target + hd].copy_from_slice(&x_data[source..source + hd]);
                }
            }
        }
        let out_shape = vec![b * h, s, hd];
        self.push_output(Op::HeadsToRows, vec![x], &out_shape, out)
    }

    /// Inverse of `heads_to_rows`: `[b * h, s, hd]` -> `[b, s, h, hd]`.
    pub fn rows_to_heads(&mut self, x: NodeId, heads: usize) -> Result<NodeId, AutodiffError> {
        let shape = self.shape(x)?;
        if shape.len() != 3 {
            return Err(AutodiffError::OpValidation {
                op: "rows_to_heads",
                reason: format!("input must be rank 3, found {:?}", shape),
            });
        }
        if heads == 0 || shape[0] % heads != 0 {
            return Err(AutodiffError::OpValidation {
                op: "rows_to_heads",
                reason: format!(
                    "batch {} is not a positive multiple of heads {}",
                    shape[0], heads
                ),
            });
        }
        let b = shape[0] / heads;
        let (s, hd) = (shape[1], shape[2]);
        let x_data = self.data(x)?;
        let mut out = vec![0.0f32; b * s * heads * hd];
        for bi in 0..b {
            for hi in 0..heads {
                for si in 0..s {
                    let source = ((bi * heads + hi) * s + si) * hd;
                    let target = ((bi * s + si) * heads + hi) * hd;
                    out[target..target + hd].copy_from_slice(&x_data[source..source + hd]);
                }
            }
        }
        let out_shape = vec![b, s, heads, hd];
        self.push_output(Op::RowsToHeads { heads }, vec![x], &out_shape, out)
    }

    /// Per-batch `a [b, m, k] x b [b, n, k]^T -> [b, m, n]` (Q x K^T).
    pub fn batched_matmul_nt(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AutodiffError> {
        let (a_shape, b_shape) = self.two_shapes(a, b)?;
        if a_shape.len() != 3 || b_shape.len() != 3 || a_shape[0] != b_shape[0] {
            return Err(AutodiffError::OpValidation {
                op: "batched_matmul_nt",
                reason: format!(
                    "both inputs must be rank 3 with equal batch, found {:?} and {:?}",
                    a_shape, b_shape
                ),
            });
        }
        if a_shape[2] != b_shape[2] {
            return Err(AutodiffError::OpValidation {
                op: "batched_matmul_nt",
                reason: format!("inner dimensions differ: {} vs {}", a_shape[2], b_shape[2]),
            });
        }
        let (batches, m, k) = (a_shape[0], a_shape[1], a_shape[2]);
        let n = b_shape[1];
        let a_data = self.data_ref(a)?;
        let b_data = self.data_ref(b)?;
        let mut out = vec![0.0f32; batches * m * n];
        batched_nt_rows(&mut out, a_data, b_data, batches, m, n, k);
        let out_shape = vec![batches, m, n];
        self.push_output(Op::BatchedMatMulNT, vec![a, b], &out_shape, out)
    }

    /// Per-batch `a [b, m, k] x b [b, k, n] -> [b, m, n]` (probs x V).
    pub fn batched_matmul_nn(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AutodiffError> {
        let (a_shape, b_shape) = self.two_shapes(a, b)?;
        if a_shape.len() != 3 || b_shape.len() != 3 || a_shape[0] != b_shape[0] {
            return Err(AutodiffError::OpValidation {
                op: "batched_matmul_nn",
                reason: format!(
                    "both inputs must be rank 3 with equal batch, found {:?} and {:?}",
                    a_shape, b_shape
                ),
            });
        }
        if a_shape[2] != b_shape[1] {
            return Err(AutodiffError::OpValidation {
                op: "batched_matmul_nn",
                reason: format!("inner dimensions differ: {} vs {}", a_shape[2], b_shape[1]),
            });
        }
        let (batches, m, k) = (a_shape[0], a_shape[1], a_shape[2]);
        let n = b_shape[2];
        let a_data = self.data_ref(a)?;
        let b_data = self.data_ref(b)?;
        let mut out = vec![0.0f32; batches * m * n];
        batched_nn_rows(&mut out, a_data, b_data, batches, m, k, n);
        let out_shape = vec![batches, m, n];
        self.push_output(Op::BatchedMatMulNN, vec![a, b], &out_shape, out)
    }

    /// Numerically stable softmax over the causal prefix of each row of
    /// `x [b, s, s]`: row `i` normalizes only positions `j <= i`; masked
    /// positions get exactly zero on both forward and backward, regardless
    /// of their input values.
    pub fn causal_softmax(&mut self, x: NodeId) -> Result<NodeId, AutodiffError> {
        let shape = self.shape(x)?;
        if shape.len() != 3 || shape[1] != shape[2] {
            return Err(AutodiffError::OpValidation {
                op: "causal_softmax",
                reason: format!("input must be rank 3 with square rows, found {:?}", shape),
            });
        }
        if shape[1] == 0 {
            return Err(AutodiffError::OpValidation {
                op: "causal_softmax",
                reason: "sequence length must be positive".to_string(),
            });
        }
        let (batches, seq) = (shape[0], shape[1]);
        let x_data = self.data(x)?;
        let mut out = vec![0.0f32; batches * seq * seq];
        for o in 0..batches {
            for i in 0..seq {
                let base = (o * seq + i) * seq;
                let row = &x_data[base..base + seq];
                let mut max = f32::NEG_INFINITY;
                for &value in &row[..=i] {
                    if value > max {
                        max = value;
                    }
                }
                let mut total = 0.0f64;
                let out_row = &mut out[base..base + seq];
                for (j, slot) in out_row.iter_mut().enumerate() {
                    if j > i {
                        *slot = 0.0;
                    } else {
                        let e = f32::exp(row[j] - max);
                        *slot = e;
                        total += f64::from(e);
                    }
                }
                let inverse = 1.0 / total as f32;
                for slot in out_row[..=i].iter_mut() {
                    *slot *= inverse;
                }
            }
        }
        self.push_output(Op::CausalSoftMax, vec![x], &shape, out)
    }

    /// Pre-norm layer normalization over the last dimension of `x [n, d]`
    /// with learnable `gain [d]` and `bias [d]`. Statistics are accumulated
    /// in f64, matching the Phase 5 runtime.
    pub fn layer_norm(
        &mut self,
        x: NodeId,
        gain: NodeId,
        bias: NodeId,
        eps: f32,
    ) -> Result<NodeId, AutodiffError> {
        let (x_shape, gain_shape) = self.two_shapes(x, gain)?;
        let bias_shape = self.shape(bias)?;
        if x_shape.len() != 2 {
            return Err(AutodiffError::OpValidation {
                op: "layer_norm",
                reason: format!("input must be rank 2, found {:?}", x_shape),
            });
        }
        if gain_shape != bias_shape {
            return Err(AutodiffError::ShapeMismatch {
                op: "layer_norm",
                expected: gain_shape.clone(),
                found: bias_shape,
            });
        }
        let d = x_shape[1];
        if gain_shape.len() != 1 || gain_shape[0] != d {
            return Err(AutodiffError::OpValidation {
                op: "layer_norm",
                reason: format!("gain/bias must have length {}, found {:?}", d, gain_shape),
            });
        }
        let n = x_shape[0];
        let x_data = self.data(x)?;
        let gain_data = self.data(gain)?;
        let bias_data = self.data(bias)?;
        let mut out = vec![0.0f32; n * d];
        for i in 0..n {
            let row = &x_data[i * d..(i + 1) * d];
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64;
            let variance = row
                .iter()
                .map(|&v| {
                    let centered = f64::from(v) - mean;
                    centered * centered
                })
                .sum::<f64>()
                / d as f64;
            let inverse_std = 1.0 / f64::sqrt(variance + f64::from(eps));
            for (j, slot) in out[i * d..(i + 1) * d].iter_mut().enumerate() {
                let normalized = (f64::from(row[j]) - mean) * inverse_std;
                *slot = (normalized * f64::from(gain_data[j]) + f64::from(bias_data[j])) as f32;
            }
        }
        self.push_output(Op::LayerNorm { eps }, vec![x, gain, bias], &x_shape, out)
    }

    /// Exact-erf GELU matching `torch.nn.GELU()` (elementwise).
    pub fn gelu(&mut self, x: NodeId) -> Result<NodeId, AutodiffError> {
        let shape = self.shape(x)?;
        let mut out = self.data(x)?;
        for_each_chunk_mut(&mut out, 8192, |_start, chunk| {
            for slot in chunk.iter_mut() {
                *slot = gelu(*slot);
            }
        });
        self.push_output(Op::Gelu, vec![x], &shape, out)
    }

    /// Tied-logit-free softmax cross-entropy (K2): computes
    /// `logits = hidden @ table^T` row by row and the mean cross-entropy
    /// against `targets`, materializing only one `[vocab]` row of logits at
    /// a time. Gradients flow to both `hidden` and `table`; `targets` is a
    /// constant.
    pub fn cross_entropy_tied(
        &mut self,
        hidden: NodeId,
        table: NodeId,
        targets: &[u32],
    ) -> Result<NodeId, AutodiffError> {
        let (hidden_shape, table_shape) = self.two_shapes(hidden, table)?;
        if hidden_shape.len() != 2 || table_shape.len() != 2 {
            return Err(AutodiffError::OpValidation {
                op: "cross_entropy_tied",
                reason: format!(
                    "both inputs must be rank 2, found {:?} and {:?}",
                    hidden_shape, table_shape
                ),
            });
        }
        let (n, d) = (hidden_shape[0], hidden_shape[1]);
        let (vocab, table_d) = (table_shape[0], table_shape[1]);
        if d != table_d {
            return Err(AutodiffError::OpValidation {
                op: "cross_entropy_tied",
                reason: format!("hidden width {} differs from table width {}", d, table_d),
            });
        }
        if n == 0 {
            return Err(AutodiffError::OpValidation {
                op: "cross_entropy_tied",
                reason: "row count must be positive".to_string(),
            });
        }
        if targets.len() != n {
            return Err(AutodiffError::OpValidation {
                op: "cross_entropy_tied",
                reason: format!("{} targets for {} rows", targets.len(), n),
            });
        }
        for &target in targets {
            if target as usize >= vocab {
                return Err(AutodiffError::OpValidation {
                    op: "cross_entropy_tied",
                    reason: format!("target {} is out of range for vocab {}", target, vocab),
                });
            }
        }
        let hidden_data = self.data_ref(hidden)?;
        let table_data = self.data_ref(table)?;
        let mut table_t = vec![0.0f32; vocab * d];
        transpose_dt(&mut table_t, table_data, vocab, d);
        let mut rows = vec![
            CeRowStat {
                max: 0.0,
                sum: 0.0,
                loss: 0.0,
            };
            n
        ];
        for_each_chunk_mut(&mut rows, 1, |start, chunk| {
            let mut logits_row = vec![0.0f32; vocab];
            for (offset, row) in chunk.iter_mut().enumerate() {
                let i = start + offset;
                let hidden_row = &hidden_data[i * d..(i + 1) * d];
                logits_row.fill(0.0);
                for (l, &h) in hidden_row.iter().enumerate() {
                    let t_row = &table_t[l * vocab..(l + 1) * vocab];
                    for (slot, &t) in logits_row.iter_mut().zip(t_row) {
                        *slot += h * t;
                    }
                }
                let mut max = f32::NEG_INFINITY;
                for &l in logits_row.iter() {
                    if l > max {
                        max = l;
                    }
                }
                let target_logit = logits_row[targets[i] as usize];
                let mut sum = 0.0f64;
                for slot in logits_row.iter_mut() {
                    let e = f32::exp(*slot - max);
                    *slot = e;
                    sum += f64::from(e);
                }
                row.max = max;
                row.sum = sum;
                row.loss = f64::from(max) + sum.ln() - f64::from(target_logit);
            }
        });
        let total: f64 = rows.iter().map(|row| row.loss).sum();
        let loss = (total / n as f64) as f32;
        self.push_output(
            Op::CrossEntropyTied {
                targets: targets.to_vec(),
                stats: rows,
            },
            vec![hidden, table],
            &[],
            vec![loss],
        )
    }

    /// The forward value of a node.
    pub fn value(&self, id: NodeId) -> Result<&Tensor, AutodiffError> {
        Ok(&self.node(id.index())?.value)
    }

    /// The accumulated gradient of a node after `backward`, if it
    /// contributes to the loss.
    pub fn grad(&self, id: NodeId) -> Result<Option<&[f32]>, AutodiffError> {
        Ok(self.node(id.index())?.grad.as_deref())
    }

    /// Runs the reverse pass from a scalar loss node, accumulating
    /// gradients into every contributing node.
    pub fn backward(&mut self, root: NodeId) -> Result<(), AutodiffError> {
        let root_index = root.index();
        {
            let node = self.node(root_index)?;
            if !node.value.shape().is_empty() {
                return Err(AutodiffError::NotScalar {
                    found: node.value.shape().to_vec(),
                });
            }
        }
        self.nodes[root_index].grad = Some(vec![1.0]);
        for index in (0..self.nodes.len()).rev() {
            if self.nodes[index].grad.is_none() {
                continue;
            }
            self.propagate(index)?;
        }
        Ok(())
    }

    fn propagate(&mut self, index: usize) -> Result<(), AutodiffError> {
        let op = self.nodes[index].op.clone();
        let gy: Vec<f32> = self.nodes[index].grad.clone().unwrap_or_default();
        let inputs = self.nodes[index].inputs.clone();
        match op {
            Op::Leaf => {}
            Op::MatMul => {
                let a = self.data_ref(inputs[0])?;
                let b = self.data_ref(inputs[1])?;
                let a_shape = self.shape(inputs[0])?;
                let n = self.shape(inputs[1])?[1];
                let (m, k) = (a_shape[0], a_shape[1]);
                let mut ga = vec![0.0f32; m * k];
                let mut gb = vec![0.0f32; k * n];
                batched_nn_backward_rows(&mut ga, &mut gb, &gy, a, b, m, k, n);
                self.acc(inputs[0], &ga)?;
                self.acc(inputs[1], &gb)?;
            }
            Op::Add => {
                self.acc(inputs[0], &gy)?;
                self.acc(inputs[1], &gy)?;
            }
            Op::Mul => {
                let a = self.data(inputs[0])?;
                let b = self.data(inputs[1])?;
                let mut gx = vec![0.0f32; a.len()];
                let mut gw = vec![0.0f32; a.len()];
                for i in 0..a.len() {
                    gx[i] = gy[i] * b[i];
                    gw[i] = gy[i] * a[i];
                }
                self.acc(inputs[0], &gx)?;
                self.acc(inputs[1], &gw)?;
            }
            Op::Sum => {
                let count = self.node(inputs[0].index())?.value.len();
                let gx = vec![gy[0]; count];
                self.acc(inputs[0], &gx)?;
            }
            Op::AddBroadcastRows => {
                let rows_shape = self.shape(inputs[1])?;
                let (s, cols) = (rows_shape[0], rows_shape[1]);
                let mut grows = vec![0.0f32; s * cols];
                for (index_in, &value) in gy.iter().enumerate() {
                    grows[(index_in / cols) % s * cols + index_in % cols] += value;
                }
                self.acc(inputs[0], &gy)?;
                self.acc(inputs[1], &grows)?;
            }
            Op::MulScalar { c } => {
                let gx: Vec<f32> = gy.iter().map(|&g| g * c).collect();
                self.acc(inputs[0], &gx)?;
            }
            Op::Reshape => {
                let in_shape = self.shape(inputs[0])?;
                let gx = Tensor::from_vec(&in_shape, gy)?.data().to_vec();
                self.acc(inputs[0], &gx)?;
            }
            Op::Embed { ids } => {
                let table_shape = self.shape(inputs[0])?;
                let (vocab, d) = (table_shape[0], table_shape[1]);
                let mut gtable = vec![0.0f32; vocab * d];
                for (row, &id) in ids.iter().enumerate() {
                    let target = id as usize * d;
                    for (offset, &value) in gy[row * d..(row + 1) * d].iter().enumerate() {
                        gtable[target + offset] += value;
                    }
                }
                self.acc(inputs[0], &gtable)?;
            }
            Op::SliceCols { start, ncols } => {
                let in_shape = self.shape(inputs[0])?;
                let cols = in_shape[1];
                let mut gx = vec![0.0f32; in_shape[0] * cols];
                for (row, out_row) in gy.chunks(ncols).enumerate() {
                    gx[row * cols + start..row * cols + start + ncols].copy_from_slice(out_row);
                }
                self.acc(inputs[0], &gx)?;
            }
            Op::HeadsToRows => {
                let in_shape = self.shape(inputs[0])?;
                let (b, s, h, hd) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
                let mut gx = vec![0.0f32; b * s * h * hd];
                for bi in 0..b {
                    for hi in 0..h {
                        for si in 0..s {
                            let source = ((bi * h + hi) * s + si) * hd;
                            let target = ((bi * s + si) * h + hi) * hd;
                            gx[target..target + hd].copy_from_slice(&gy[source..source + hd]);
                        }
                    }
                }
                self.acc(inputs[0], &gx)?;
            }
            Op::RowsToHeads { heads } => {
                let in_shape = self.shape(inputs[0])?;
                let b = in_shape[0] / heads;
                let (s, hd) = (in_shape[1], in_shape[2]);
                let mut gx = vec![0.0f32; b * s * heads * hd];
                for bi in 0..b {
                    for si in 0..s {
                        for hi in 0..heads {
                            let source = ((bi * s + si) * heads + hi) * hd;
                            let target = ((bi * heads + hi) * s + si) * hd;
                            gx[target..target + hd].copy_from_slice(&gy[source..source + hd]);
                        }
                    }
                }
                self.acc(inputs[0], &gx)?;
            }
            Op::BatchedMatMulNT => {
                let a = self.data_ref(inputs[0])?;
                let b = self.data_ref(inputs[1])?;
                let a_shape = self.shape(inputs[0])?;
                let b_shape = self.shape(inputs[1])?;
                let (batches, m, k) = (a_shape[0], a_shape[1], a_shape[2]);
                let n = b_shape[1];
                let mut ga = vec![0.0f32; batches * m * k];
                let mut gb = vec![0.0f32; batches * n * k];
                batched_nt_backward_rows(&mut ga, &mut gb, &gy, a, b, m, n, k);
                self.acc(inputs[0], &ga)?;
                self.acc(inputs[1], &gb)?;
            }
            Op::BatchedMatMulNN => {
                let a = self.data_ref(inputs[0])?;
                let b = self.data_ref(inputs[1])?;
                let a_shape = self.shape(inputs[0])?;
                let b_shape = self.shape(inputs[1])?;
                let (batches, m, k) = (a_shape[0], a_shape[1], a_shape[2]);
                let n = b_shape[2];
                let mut ga = vec![0.0f32; batches * m * k];
                let mut gb = vec![0.0f32; batches * k * n];
                batched_nn_backward_rows(&mut ga, &mut gb, &gy, a, b, m, k, n);
                self.acc(inputs[0], &ga)?;
                self.acc(inputs[1], &gb)?;
            }
            Op::CausalSoftMax => {
                let shape = self.shape(inputs[0])?;
                let (batches, seq) = (shape[0], shape[1]);
                let y = self.data(NodeId { index })?;
                let mut gx = vec![0.0f32; batches * seq * seq];
                for o in 0..batches {
                    for i in 0..seq {
                        let base = (o * seq + i) * seq;
                        let gy_row = &gy[base..base + seq];
                        let y_row = &y[base..base + seq];
                        let gx_row = &mut gx[base..base + seq];
                        let mut dot = 0.0f64;
                        for j in 0..=i {
                            dot += f64::from(gy_row[j]) * f64::from(y_row[j]);
                        }
                        let dot = dot as f32;
                        for j in 0..=i {
                            gx_row[j] = y_row[j] * (gy_row[j] - dot);
                        }
                    }
                }
                self.acc(inputs[0], &gx)?;
            }
            Op::LayerNorm { eps } => {
                let x = self.data(inputs[0])?;
                let gain = self.data(inputs[1])?;
                let x_shape = self.shape(inputs[0])?;
                let (n, d) = (x_shape[0], x_shape[1]);
                let mut gx = vec![0.0f32; n * d];
                let mut dgain = vec![0.0f64; d];
                let mut dbias = vec![0.0f64; d];
                for i in 0..n {
                    let row = &x[i * d..(i + 1) * d];
                    let gy_row = &gy[i * d..(i + 1) * d];
                    let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64;
                    let variance = row
                        .iter()
                        .map(|&v| {
                            let centered = f64::from(v) - mean;
                            centered * centered
                        })
                        .sum::<f64>()
                        / d as f64;
                    let inverse_std = 1.0 / f64::sqrt(variance + f64::from(eps));
                    let mut dvar = 0.0f64;
                    let mut sum_dxhat = 0.0f64;
                    let mut sum_centered = 0.0f64;
                    for j in 0..d {
                        let dxhat = f64::from(gy_row[j]) * f64::from(gain[j]);
                        let centered = f64::from(row[j]) - mean;
                        dvar += dxhat * centered * -0.5 * inverse_std * inverse_std * inverse_std;
                        sum_dxhat += dxhat;
                        sum_centered += centered;
                    }
                    let dmu = -sum_dxhat * inverse_std - 2.0 / d as f64 * dvar * sum_centered;
                    for j in 0..d {
                        let dxhat = f64::from(gy_row[j]) * f64::from(gain[j]);
                        let centered = f64::from(row[j]) - mean;
                        gx[i * d + j] = (dxhat * inverse_std
                            + 2.0 * dvar * centered / d as f64
                            + dmu / d as f64) as f32;
                        dgain[j] += f64::from(gy_row[j]) * centered * inverse_std;
                        dbias[j] += f64::from(gy_row[j]);
                    }
                }
                let dgain: Vec<f32> = dgain.iter().map(|&v| v as f32).collect();
                let dbias: Vec<f32> = dbias.iter().map(|&v| v as f32).collect();
                self.acc(inputs[0], &gx)?;
                self.acc(inputs[1], &dgain)?;
                self.acc(inputs[2], &dbias)?;
            }
            Op::Gelu => {
                let x = self.data_ref(inputs[0])?;
                let mut gx = vec![0.0f32; x.len()];
                for_each_chunk_mut(&mut gx, 8192, |start, chunk| {
                    for (offset, slot) in chunk.iter_mut().enumerate() {
                        *slot = gy[start + offset] * gelu_derivative(x[start + offset]);
                    }
                });
                self.acc(inputs[0], &gx)?;
            }
            Op::CrossEntropyTied { targets, stats } => {
                let hidden = self.data_ref(inputs[0])?;
                let table = self.data_ref(inputs[1])?;
                let hidden_shape = self.shape(inputs[0])?;
                let table_shape = self.shape(inputs[1])?;
                let (n, d) = (hidden_shape[0], hidden_shape[1]);
                let vocab = table_shape[0];
                let scale = f64::from(gy[0]) / n as f64;
                let mut table_t = vec![0.0f32; vocab * d];
                transpose_dt(&mut table_t, table, vocab, d);
                let mut ghidden = vec![0.0f32; n * d];
                for_each_chunk_mut(&mut ghidden, d, |start, chunk| {
                    let mut logits_row = vec![0.0f32; vocab];
                    let mut weighted = vec![0.0f32; d];
                    for (offset, row) in chunk.chunks_mut(d).enumerate() {
                        let i = start / d + offset;
                        let hidden_row = &hidden[i * d..(i + 1) * d];
                        logits_row.fill(0.0);
                        for (l, &h) in hidden_row.iter().enumerate() {
                            let t_row = &table_t[l * vocab..(l + 1) * vocab];
                            for (slot, &t) in logits_row.iter_mut().zip(t_row) {
                                *slot += h * t;
                            }
                        }
                        let max = stats[i].max;
                        let inverse = (1.0 / stats[i].sum) as f32;
                        for slot in weighted.iter_mut() {
                            *slot = 0.0;
                        }
                        let target = targets[i] as usize;
                        for (j, &l) in logits_row.iter().enumerate() {
                            let p = f32::exp(l - max) * inverse;
                            let t_row = &table[j * d..(j + 1) * d];
                            for (w_slot, &t) in weighted.iter_mut().zip(t_row) {
                                *w_slot += p * t;
                            }
                        }
                        let target_row = &table[target * d..(target + 1) * d];
                        for (slot, (w, t)) in row.iter_mut().zip(weighted.iter().zip(target_row)) {
                            *slot = (scale * f64::from(w - t)) as f32;
                        }
                    }
                });
                let mut gtable = vec![0.0f32; vocab * d];
                for_each_chunk_mut(&mut gtable, 8 * d, |start, chunk| {
                    let first_row = start / d;
                    let row_count = chunk.len() / d;
                    for r in 0..row_count {
                        let j = first_row + r;
                        let t_row = &table[j * d..(j + 1) * d];
                        let g_row = &mut chunk[r * d..(r + 1) * d];
                        for g_slot in g_row.iter_mut() {
                            *g_slot = 0.0;
                        }
                        for i in 0..n {
                            let hidden_row = &hidden[i * d..(i + 1) * d];
                            let l = dot_f32(hidden_row, t_row);
                            let p = f32::exp(l - stats[i].max) * ((1.0 / stats[i].sum) as f32);
                            let delta = if j == targets[i] as usize { scale } else { 0.0 };
                            let coefficient = (scale * f64::from(p) - delta) as f32;
                            for (g_slot, &h) in g_row.iter_mut().zip(hidden_row) {
                                *g_slot += coefficient * h;
                            }
                        }
                    }
                });
                self.acc(inputs[0], &ghidden)?;
                self.acc(inputs[1], &gtable)?;
            }
        }
        Ok(())
    }

    fn push(&mut self, node: Node) -> NodeId {
        let index = self.nodes.len();
        self.nodes.push(node);
        NodeId { index }
    }

    fn push_output(
        &mut self,
        op: Op,
        inputs: Vec<NodeId>,
        shape: &[usize],
        data: Vec<f32>,
    ) -> Result<NodeId, AutodiffError> {
        let value = Tensor::from_vec(shape, data)?;
        Ok(self.push(Node {
            value,
            grad: None,
            op,
            inputs,
        }))
    }

    fn node(&self, index: usize) -> Result<&Node, AutodiffError> {
        self.nodes
            .get(index)
            .ok_or(AutodiffError::UnknownNode { index })
    }

    fn data(&self, id: NodeId) -> Result<Vec<f32>, AutodiffError> {
        Ok(self.node(id.index())?.value.data().to_vec())
    }

    fn data_ref(&self, id: NodeId) -> Result<&[f32], AutodiffError> {
        Ok(self.node(id.index())?.value.data())
    }

    fn shape(&self, id: NodeId) -> Result<Vec<usize>, AutodiffError> {
        Ok(self.node(id.index())?.value.shape().to_vec())
    }

    fn two_shapes(&self, a: NodeId, b: NodeId) -> Result<(Vec<usize>, Vec<usize>), AutodiffError> {
        Ok((self.shape(a)?, self.shape(b)?))
    }

    fn acc(&mut self, id: NodeId, contrib: &[f32]) -> Result<(), AutodiffError> {
        if contrib.is_empty() {
            return Ok(());
        }
        let node = self
            .nodes
            .get_mut(id.index())
            .ok_or(AutodiffError::UnknownNode { index: id.index() })?;
        match &mut node.grad {
            Some(grad) => {
                for (slot, &value) in grad.iter_mut().zip(contrib) {
                    *slot += value;
                }
            }
            None => node.grad = Some(contrib.to_vec()),
        }
        Ok(())
    }
}

/// Threads available for output-partitioned parallelism (1 if unknown).
fn parallel_threads() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
}

/// Splits `buf` into contiguous chunks and runs `f(start, chunk)` on each,
/// spawning threads only when the buffer is large enough to amortize the
/// spawn. Chunks form a fixed partition of the buffer: every element belongs
/// to exactly one chunk, so callers must keep each output element's value
/// dependent only on its own chunk for thread-count-independent results.
fn for_each_chunk_mut<T: Send>(
    buf: &mut [T],
    min_chunk: usize,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    if buf.is_empty() {
        return;
    }
    let min_chunk = min_chunk.max(1);
    let threads = parallel_threads();
    let chunk = if threads <= 1 || buf.len() <= min_chunk {
        buf.len()
    } else {
        let per_thread = buf.len().div_ceil(threads);
        per_thread.div_ceil(min_chunk) * min_chunk
    };
    if chunk >= buf.len() {
        f(0, buf);
        return;
    }
    let f_ref = &f;
    std::thread::scope(|scope| {
        for (index, piece) in buf.chunks_mut(chunk).enumerate() {
            let start = index * chunk;
            scope.spawn(move || f_ref(start, piece));
        }
    });
}

/// Contiguous dot product with four independent f32 accumulators (fixed
/// order, vectorizer-friendly).
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 4];
    for (a_chunk, b_chunk) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for lane in 0..4 {
            acc[lane] += a_chunk[lane] * b_chunk[lane];
        }
    }
    let mut total = acc[0] + acc[1] + acc[2] + acc[3];
    let head = a.len() / 4 * 4;
    for index in head..a.len() {
        total += a[index] * b[index];
    }
    total
}

/// Rows per parallel task for a kernel whose per-row cost is `row_macs`
/// multiply-accumulates: targets roughly 256k MACs per task.
fn min_rows_for(row_macs: usize) -> usize {
    ((262_144 / row_macs.max(1)) + 1).clamp(1, 256)
}

/// `out[batches, m, n] = a[batches, m, k] x b[batches, k, n]` per batch,
/// f32 accumulation in fixed l order, rows split across a fixed partition.
fn batched_nn_rows(
    out: &mut [f32],
    a: &[f32],
    b: &[f32],
    batches: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let rows = batches * m;
    if rows == 0 || n == 0 {
        return;
    }
    if k == 0 {
        out.fill(0.0);
        return;
    }
    let min_rows = min_rows_for(k * n);
    for_each_chunk_mut(out, min_rows * n, |start, chunk| {
        let first_row = start / n;
        let row_count = chunk.len() / n;
        for r in 0..row_count {
            let row = first_row + r;
            let o = row / m;
            let i = row % m;
            let slot_row = &mut chunk[r * n..(r + 1) * n];
            slot_row.fill(0.0);
            let a_row = &a[(o * m + i) * k..(o * m + i + 1) * k];
            for (l, &av) in a_row.iter().enumerate() {
                let b_row = &b[(o * k + l) * n..(o * k + l + 1) * n];
                for (slot, &bv) in slot_row.iter_mut().zip(b_row) {
                    *slot += av * bv;
                }
            }
        }
    });
}

/// `out[batches, m, n] = a[batches, m, k] x b[batches, n, k]^T` per batch.
/// The transposed operand is re-laid-out into `[k, n]` once per call so the
/// product runs in the fast axpy form; rows are split across a fixed
/// partition.
fn batched_nt_rows(
    out: &mut [f32],
    a: &[f32],
    b: &[f32],
    batches: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    let rows = batches * m;
    if rows == 0 || n == 0 || k == 0 {
        out.fill(0.0);
        return;
    }
    let mut b_t = vec![0.0f32; batches * k * n];
    transpose_nt(&mut b_t, b, n, k);
    batched_nn_rows(out, a, &b_t, batches, m, k, n);
}

/// `dst[d, rows] = src[rows, d]^T` for one 2-D table; every output element
/// is written exactly once, so the result is thread-count independent.
fn transpose_dt(dst: &mut [f32], src: &[f32], rows: usize, d: usize) {
    if rows == 0 || d == 0 {
        return;
    }
    for_each_chunk_mut(dst, 8192, |start, chunk| {
        let first_l = start / rows;
        let count = chunk.len() / rows;
        for r in 0..count {
            let l = first_l + r;
            let slot_row = &mut chunk[r * rows..(r + 1) * rows];
            for (j, slot) in slot_row.iter_mut().enumerate() {
                *slot = src[j * d + l];
            }
        }
    });
}

/// `dst[batches, k, n] = b[batches, n, k]^T` per batch; every output element
/// is written exactly once, so the result is thread-count independent.
fn transpose_nt(dst: &mut [f32], b: &[f32], n: usize, k: usize) {
    if n == 0 || k == 0 {
        return;
    }
    for_each_chunk_mut(dst, 8192, |start, chunk| {
        let first_row = start / n;
        let row_count = chunk.len() / n;
        for r in 0..row_count {
            let row = first_row + r;
            let o = row / k;
            let l = row % k;
            let slot_row = &mut chunk[r * n..(r + 1) * n];
            for (j, slot) in slot_row.iter_mut().enumerate() {
                *slot = b[(o * n + j) * k + l];
            }
        }
    });
}

/// Gradients of `out[batches, m, n] = a[batches, m, k] x b[batches, k, n]`.
#[allow(clippy::too_many_arguments)]
fn batched_nn_backward_rows(
    ga: &mut [f32],
    gb: &mut [f32],
    gy: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let row_macs = k.max(1) * n.max(1);
    let min_rows = min_rows_for(row_macs);
    // ga[(o, i), l] = sum_j gy[(o, i), j] * b[(o, l), j] (dot form).
    for_each_chunk_mut(ga, min_rows * k, |start, chunk| {
        let first_row = start / k;
        let row_count = chunk.len() / k;
        for r in 0..row_count {
            let row = first_row + r;
            let o = row / m;
            let i = row % m;
            let gy_row = &gy[(o * m + i) * n..(o * m + i + 1) * n];
            for l in 0..k {
                chunk[r * k + l] = dot_f32(gy_row, &b[(o * k + l) * n..(o * k + l + 1) * n]);
            }
        }
    });
    // gb[(o, l), j] = sum_i a[(o, i), l] * gy[(o, i), j] (axpy over i).
    for_each_chunk_mut(gb, min_rows * n, |start, chunk| {
        let first_row = start / n;
        let row_count = chunk.len() / n;
        for r in 0..row_count {
            let row = first_row + r;
            let o = row / k;
            let l = row % k;
            let slot_row = &mut chunk[r * n..(r + 1) * n];
            slot_row.fill(0.0);
            for i in 0..m {
                let s = a[(o * m + i) * k + l];
                let gy_row = &gy[(o * m + i) * n..(o * m + i + 1) * n];
                for (slot, &y) in slot_row.iter_mut().zip(gy_row) {
                    *slot += s * y;
                }
            }
        }
    });
}

/// Gradients of `out[batches, m, n] = a[batches, m, k] x b[batches, n, k]^T`.
#[allow(clippy::too_many_arguments)]
fn batched_nt_backward_rows(
    ga: &mut [f32],
    gb: &mut [f32],
    gy: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let row_macs = k.max(1) * n.max(1);
    let min_rows = min_rows_for(row_macs);
    // ga[(o, i), l] = sum_j gy[(o, i), j] * b[(o, j), l] (axpy over j).
    for_each_chunk_mut(ga, min_rows * k, |start, chunk| {
        let first_row = start / k;
        let row_count = chunk.len() / k;
        for r in 0..row_count {
            let row = first_row + r;
            let o = row / m;
            let i = row % m;
            let slot_row = &mut chunk[r * k..(r + 1) * k];
            slot_row.fill(0.0);
            for j in 0..n {
                let s = gy[(o * m + i) * n + j];
                let b_row = &b[(o * n + j) * k..(o * n + j + 1) * k];
                for (slot, &bv) in slot_row.iter_mut().zip(b_row) {
                    *slot += s * bv;
                }
            }
        }
    });
    // gb[(o, j), l] = sum_i gy[(o, i), j] * a[(o, i), l] (axpy over i).
    for_each_chunk_mut(gb, min_rows * k, |start, chunk| {
        let first_row = start / k;
        let row_count = chunk.len() / k;
        for r in 0..row_count {
            let row = first_row + r;
            let o = row / n;
            let j = row % n;
            let slot_row = &mut chunk[r * k..(r + 1) * k];
            slot_row.fill(0.0);
            for i in 0..m {
                let s = gy[(o * m + i) * n + j];
                let a_row = &a[(o * m + i) * k..(o * m + i + 1) * k];
                for (slot, &av) in slot_row.iter_mut().zip(a_row) {
                    *slot += s * av;
                }
            }
        }
    });
}

/// Exact-erf GELU matching `torch.nn.GELU()` (no tanh approximation).
fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    let scaled = x * std::f64::consts::FRAC_1_SQRT_2;
    (0.5 * x * (1.0 + erf(scaled))) as f32
}

/// Derivative of the exact-erf GELU.
fn gelu_derivative(x: f32) -> f32 {
    let x = f64::from(x);
    let scaled = x * std::f64::consts::FRAC_1_SQRT_2;
    let phi = 0.5 * (1.0 + erf(scaled));
    let density = f64::exp(-0.5 * x * x) * (std::f64::consts::TAU).sqrt().recip();
    (phi + x * density) as f32
}

/// Abramowitz & Stegun 7.1.26 erf approximation (max abs error 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * f64::exp(-x * x);
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(g: &mut Graph, shapes: &[&[usize]], values: &[Vec<f32>]) -> Vec<NodeId> {
        shapes
            .iter()
            .zip(values)
            .map(|(shape, value)| {
                g.leaf(shape, value.clone())
                    .unwrap_or_else(|e| panic!("{e}"))
            })
            .collect()
    }

    #[test]
    fn matmul_matches_hand_computed_product() {
        let mut g = Graph::new();
        let nodes = ids(
            &mut g,
            &[&[2, 2], &[2, 2]],
            &[vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
        );
        let y = g
            .matmul(nodes[0], nodes[1])
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            g.value(y).unwrap_or_else(|e| panic!("{e}")).data(),
            &[19.0, 22.0, 43.0, 50.0]
        );
    }

    #[test]
    fn causal_softmax_normalizes_over_causal_prefix_only() {
        let mut g = Graph::new();
        let x = ids(
            &mut g,
            &[&[1, 3, 3]],
            &[vec![3.0, 1.0, 2.0, 0.0, 5.0, 1.0, -1.0, 4.0, 0.5]],
        );
        let y = g.causal_softmax(x[0]).unwrap_or_else(|e| panic!("{e}"));
        let out = g.value(y).unwrap_or_else(|e| panic!("{e}")).data().to_vec();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
        let e0 = (-5.0f32).exp();
        let e1 = 1.0f32;
        assert!((out[3] - e0 / (e0 + e1)).abs() < 1e-6);
        assert!((out[4] - e1 / (e0 + e1)).abs() < 1e-6);
        assert_eq!(out[5], 0.0);
        let row_sum: f32 = out[6..9].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn embed_scatter_adds_duplicate_ids() {
        let mut g = Graph::new();
        let nodes = ids(&mut g, &[&[3, 2]], &[vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
        let y = g
            .embed(nodes[0], &[1, 1, 2])
            .unwrap_or_else(|e| panic!("{e}"));
        let loss = g.sum(y).unwrap_or_else(|e| panic!("{e}"));
        g.backward(loss).unwrap_or_else(|e| panic!("{e}"));
        let grad = g
            .grad(nodes[0])
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_default();
        assert_eq!(grad, &[0.0, 0.0, 2.0, 2.0, 1.0, 1.0]);
    }

    #[test]
    fn add_broadcast_rows_cycles_position_rows() {
        let mut g = Graph::new();
        let nodes = ids(
            &mut g,
            &[&[4, 2], &[2, 2]],
            &[
                vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                vec![10.0, 20.0, 30.0, 40.0],
            ],
        );
        let y = g
            .add_broadcast_rows(nodes[0], nodes[1])
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            g.value(y).unwrap_or_else(|e| panic!("{e}")).data(),
            &[11.0, 21.0, 31.0, 41.0, 11.0, 21.0, 31.0, 41.0]
        );
    }

    #[test]
    fn cross_entropy_tied_hand_case() {
        let mut g = Graph::new();
        let nodes = ids(
            &mut g,
            &[&[1, 2], &[2, 2]],
            &[vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 1.0]],
        );
        let loss = g
            .cross_entropy_tied(nodes[0], nodes[1], &[0])
            .unwrap_or_else(|e| panic!("{e}"));
        let value = g.value(loss).unwrap_or_else(|e| panic!("{e}")).data()[0];
        assert!((value - (2.0f32).ln()).abs() < 1e-6, "got {value}");
        g.backward(loss).unwrap_or_else(|e| panic!("{e}"));
        let ghidden = g
            .grad(nodes[0])
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_default();
        let gtable = g
            .grad(nodes[1])
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_default();
        assert!((ghidden[0] + 0.5).abs() < 1e-6);
        assert!((ghidden[1] - 0.5).abs() < 1e-6);
        assert!(gtable.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn layer_norm_constant_row_collapses_to_bias() {
        let mut g = Graph::new();
        let nodes = ids(
            &mut g,
            &[&[1, 3], &[3], &[3]],
            &[
                vec![7.0, 7.0, 7.0],
                vec![2.0, 3.0, 4.0],
                vec![0.5, 0.5, 0.5],
            ],
        );
        let y = g
            .layer_norm(nodes[0], nodes[1], nodes[2], 1e-5)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            g.value(y).unwrap_or_else(|e| panic!("{e}")).data(),
            &[0.5, 0.5, 0.5]
        );
    }

    #[test]
    fn gelu_matches_known_values() {
        let mut g = Graph::new();
        let x = ids(&mut g, &[&[3]], &[vec![0.0, 1.0, -1.0]]);
        let y = g.gelu(x[0]).unwrap_or_else(|e| panic!("{e}"));
        let out = g.value(y).unwrap_or_else(|e| panic!("{e}")).data().to_vec();
        assert!(out[0].abs() < 1e-7);
        assert!((out[1] - 0.841_344_7).abs() < 1e-5, "got {}", out[1]);
        assert!((out[2] + 0.158_655_25).abs() < 1e-5, "got {}", out[2]);
    }

    #[test]
    fn shape_violations_return_typed_errors() {
        let mut g = Graph::new();
        let nodes = ids(&mut g, &[&[2, 3], &[4, 5]], &[vec![0.0; 6], vec![0.0; 20]]);
        assert!(matches!(
            g.matmul(nodes[0], nodes[1]),
            Err(AutodiffError::OpValidation { op: "matmul", .. })
        ));
        assert!(matches!(
            g.add(nodes[0], nodes[1]),
            Err(AutodiffError::ShapeMismatch { op: "add", .. })
        ));
        assert!(matches!(
            g.embed(nodes[0], &[99]),
            Err(AutodiffError::OpValidation { op: "embed", .. })
        ));
        assert!(matches!(
            g.slice_cols(nodes[0], 2, 5),
            Err(AutodiffError::OpValidation {
                op: "slice_cols",
                ..
            })
        ));
        assert!(matches!(
            g.value(NodeId { index: 99 }),
            Err(AutodiffError::UnknownNode { index: 99 })
        ));
    }

    #[test]
    fn backward_requires_a_scalar_root() {
        let mut g = Graph::new();
        let nodes = ids(&mut g, &[&[2, 2]], &[vec![0.0; 4]]);
        let sum = g.sum(nodes[0]).unwrap_or_else(|e| panic!("{e}"));
        let non_scalar = g
            .mul_scalar(nodes[0], 2.0)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(
            g.backward(non_scalar),
            Err(AutodiffError::NotScalar { .. })
        ));
        g.backward(sum).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn fan_out_accumulates_gradients() {
        let mut g = Graph::new();
        let nodes = ids(&mut g, &[&[2]], &[vec![1.0, 2.0]]);
        let doubled = g
            .mul_scalar(nodes[0], 3.0)
            .unwrap_or_else(|e| panic!("{e}"));
        let combined = g.add(nodes[0], doubled).unwrap_or_else(|e| panic!("{e}"));
        let loss = g.sum(combined).unwrap_or_else(|e| panic!("{e}"));
        g.backward(loss).unwrap_or_else(|e| panic!("{e}"));
        let grad = g
            .grad(nodes[0])
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_default();
        assert_eq!(grad, &[4.0, 4.0]);
    }
}
