//! Finite-difference gradchecks (R1 gate): every autodiff op's analytic
//! gradient is compared against central differences of a scalar loss built
//! from the op output, over randomized small inputs.

use byva_autodiff::{AutodiffError, Graph, NodeId};
use proptest::prelude::*;

type Build<'a> = dyn Fn(&mut Graph, &[NodeId]) -> Result<NodeId, AutodiffError> + 'a;

const EPS: f32 = 1e-3;
const ABS_TOL: f64 = 2e-3;
const REL_TOL: f64 = 3e-2;

fn weighted_loss(g: &mut Graph, out: NodeId) -> Result<NodeId, AutodiffError> {
    let count: usize = g.value(out)?.shape().iter().product();
    let weights: Vec<f32> = (0..count)
        .map(|k| (((k * 37) % 11) as f32 - 5.0) / 7.0)
        .collect();
    let weights_id = g.leaf(&[count], weights)?;
    let flat = g.reshape(out, &[count])?;
    let product = g.mul(flat, weights_id)?;
    g.sum(product)
}

fn loss_of(build: &Build<'_>, shapes: &[Vec<usize>], values: &[Vec<f32>]) -> f64 {
    let mut g = Graph::new();
    let ids: Vec<NodeId> = shapes
        .iter()
        .zip(values)
        .map(|(shape, value)| {
            g.leaf(shape, value.clone())
                .unwrap_or_else(|e| panic!("{e}"))
        })
        .collect();
    let loss = build(&mut g, &ids).unwrap_or_else(|e| panic!("{e}"));
    f64::from(g.value(loss).unwrap_or_else(|e| panic!("{e}")).data()[0])
}

fn assert_grads_match(build: &Build<'_>, shapes: &[Vec<usize>], values: &[Vec<f32>]) {
    let mut g = Graph::new();
    let ids: Vec<NodeId> = shapes
        .iter()
        .zip(values)
        .map(|(shape, value)| {
            g.leaf(shape, value.clone())
                .unwrap_or_else(|e| panic!("{e}"))
        })
        .collect();
    let loss = build(&mut g, &ids).unwrap_or_else(|e| panic!("{e}"));
    g.backward(loss).unwrap_or_else(|e| panic!("{e}"));
    for (input, (shape, _value)) in ids.iter().zip(shapes.iter().zip(values)) {
        let analytic: Vec<f32> = g
            .grad(*input)
            .unwrap_or_else(|e| panic!("{e}"))
            .expect("every tested input must contribute to the loss")
            .to_vec();
        assert_eq!(analytic.len(), shape.iter().product::<usize>());
        for (k, &a) in analytic.iter().enumerate() {
            let mut plus = values.to_vec();
            plus[input.index()][k] += EPS;
            let mut minus = values.to_vec();
            minus[input.index()][k] -= EPS;
            let numeric = (loss_of(build, shapes, &plus) - loss_of(build, shapes, &minus))
                / f64::from(2.0 * EPS);
            assert!(
                (f64::from(a) - numeric).abs() <= ABS_TOL + REL_TOL * numeric.abs(),
                "input {} element {}: analytic {a} vs numeric {numeric}",
                input.index(),
                k
            );
        }
    }
}

fn vec_strategy(len: usize) -> BoxedStrategy<Vec<f32>> {
    proptest::collection::vec(-2.0f32..2.0, len).boxed()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn gradcheck_matmul(a in vec_strategy(6), b in vec_strategy(12)) {
        let shapes = vec![vec![2, 3], vec![3, 4]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.matmul(ids[0], ids[1])?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[a, b]);
    }

    #[test]
    fn gradcheck_add(a in vec_strategy(6), b in vec_strategy(6)) {
        let shapes = vec![vec![2, 3], vec![2, 3]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.add(ids[0], ids[1])?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[a, b]);
    }

    #[test]
    fn gradcheck_mul(a in vec_strategy(6), b in vec_strategy(6)) {
        let shapes = vec![vec![2, 3], vec![2, 3]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.mul(ids[0], ids[1])?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[a, b]);
    }

    #[test]
    fn gradcheck_add_broadcast_rows(x in vec_strategy(12), rows in vec_strategy(6)) {
        let shapes = vec![vec![4, 3], vec![2, 3]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.add_broadcast_rows(ids[0], ids[1])?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[x, rows]);
    }

    #[test]
    fn gradcheck_mul_scalar(x in vec_strategy(6)) {
        let shapes = vec![vec![2, 3]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.mul_scalar(ids[0], 1.375)?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[x]);
    }

    #[test]
    fn gradcheck_reshape(x in vec_strategy(6)) {
        let shapes = vec![vec![2, 3]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.reshape(ids[0], &[3, 2])?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[x]);
    }

    #[test]
    fn gradcheck_embed(table in vec_strategy(10)) {
        let shapes = vec![vec![5, 2]];
        let ids = [0u32, 2, 2, 4, 0];
        let build = move |g: &mut Graph, ids_in: &[NodeId]| {
            let y = g.embed(ids_in[0], &ids)?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[table]);
    }

    #[test]
    fn gradcheck_slice_cols(x in vec_strategy(12)) {
        let shapes = vec![vec![3, 4]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.slice_cols(ids[0], 1, 3)?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[x]);
    }

    #[test]
    fn gradcheck_heads_to_rows(x in vec_strategy(24)) {
        let shapes = vec![vec![2, 3, 2, 2]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.heads_to_rows(ids[0], 2)?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[x]);
    }

    #[test]
    fn gradcheck_rows_to_heads(x in vec_strategy(24)) {
        let shapes = vec![vec![4, 3, 2]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.rows_to_heads(ids[0], 2)?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[x]);
    }

    #[test]
    fn gradcheck_batched_matmul_nt(a in vec_strategy(16), b in vec_strategy(16)) {
        let shapes = vec![vec![2, 2, 4], vec![2, 2, 4]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.batched_matmul_nt(ids[0], ids[1])?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[a, b]);
    }

    #[test]
    fn gradcheck_batched_matmul_nn(a in vec_strategy(16), b in vec_strategy(16)) {
        let shapes = vec![vec![2, 2, 4], vec![2, 4, 2]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.batched_matmul_nn(ids[0], ids[1])?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[a, b]);
    }

    #[test]
    fn gradcheck_causal_softmax(x in vec_strategy(18)) {
        let shapes = vec![vec![2, 3, 3]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.causal_softmax(ids[0])?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[x]);
    }

    #[test]
    fn gradcheck_layer_norm_all_inputs(x in vec_strategy(6), gain in vec_strategy(3), bias in vec_strategy(3)) {
        let shapes = vec![vec![2, 3], vec![3], vec![3]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.layer_norm(ids[0], ids[1], ids[2], 1e-5)?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[x, gain, bias]);
    }

    #[test]
    fn gradcheck_gelu(x in vec_strategy(6)) {
        let shapes = vec![vec![2, 3]];
        let build = |g: &mut Graph, ids: &[NodeId]| {
            let y = g.gelu(ids[0])?;
            weighted_loss(g, y)
        };
        assert_grads_match(&build, &shapes, &[x]);
    }

    #[test]
    fn gradcheck_cross_entropy_tied(hidden in vec_strategy(6), table in vec_strategy(10)) {
        let shapes = vec![vec![3, 2], vec![5, 2]];
        let targets = [0u32, 4, 2];
        let build = move |g: &mut Graph, ids: &[NodeId]| {
            g.cross_entropy_tied(ids[0], ids[1], &targets)
        };
        assert_grads_match(&build, &shapes, &[hidden, table]);
    }

    #[test]
    fn gradcheck_attention_block_end_to_end(
        x in vec_strategy(24),
        w_qkv in vec_strategy(48),
        w_out in vec_strategy(16),
        table in vec_strategy(20),
    ) {
        // One causal attention block with two heads over b=2, s=3, hd=2,
        // fed into the tied cross-entropy: exercises reshape, slice_cols,
        // heads_to_rows/rows_to_heads, both batched matmuls, causal_softmax,
        // matmul, and cross_entropy_tied together.
        let shapes = vec![vec![6, 4], vec![4, 12], vec![4, 4], vec![5, 4]];
        let targets = [1u32, 3, 0, 2, 4, 1];
        let scale = (2.0f32).sqrt();
        let build = move |g: &mut Graph, ids: &[NodeId]| {
            let qkv = g.matmul(ids[0], ids[1])?;
            let q = g.slice_cols(qkv, 0, 4)?;
            let k = g.slice_cols(qkv, 4, 4)?;
            let v = g.slice_cols(qkv, 8, 4)?;
            let q = g.mul_scalar(q, 1.0 / scale)?;
            let q = g.reshape(q, &[2, 3, 2, 2])?;
            let k = g.reshape(k, &[2, 3, 2, 2])?;
            let v = g.reshape(v, &[2, 3, 2, 2])?;
            let q = g.heads_to_rows(q, 2)?;
            let k = g.heads_to_rows(k, 2)?;
            let v = g.heads_to_rows(v, 2)?;
            let scores = g.batched_matmul_nt(q, k)?;
            let probs = g.causal_softmax(scores)?;
            let ctx = g.batched_matmul_nn(probs, v)?;
            let merged = g.rows_to_heads(ctx, 2)?;
            let merged = g.reshape(merged, &[6, 4])?;
            let out = g.matmul(merged, ids[2])?;
            g.cross_entropy_tied(out, ids[3], &targets)
        };
        assert_grads_match(&build, &shapes, &[x, w_qkv, w_out, table]);
    }
}
