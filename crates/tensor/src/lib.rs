//! Minimal dense fp32 tensor storage for the BYVA Rust training path (D-0022).
//!
//! Owned row-major buffers with checked shape arithmetic and typed errors;
//! the training kernels and reverse-mode tape are layered on this in R1
//! (see `docs/PROPOSAL_RUST_TRAINING.md`).

#![deny(unsafe_op_in_unsafe_fn)]

mod error;

pub use error::TensorError;

/// An owned dense fp32 tensor with row-major layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Creates a zero-filled tensor with the given shape.
    pub fn zeros(shape: &[usize]) -> Result<Self, TensorError> {
        let count = element_count(shape)?;
        Ok(Self {
            shape: shape.to_vec(),
            data: vec![0.0; count as usize],
        })
    }

    /// Creates a tensor from a flat row-major buffer.
    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Result<Self, TensorError> {
        let count = element_count(shape)?;
        if count != data.len() as u64 {
            return Err(TensorError::ElementCountMismatch {
                shape: shape.to_vec(),
                expected: count,
                found: data.len() as u64,
            });
        }
        Ok(Self {
            shape: shape.to_vec(),
            data,
        })
    }

    /// The tensor's shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor holds no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The flat row-major element buffer.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Reads the element at a multi-dimensional index.
    pub fn get(&self, index: &[usize]) -> Result<f32, TensorError> {
        let flat = self.flat_index(index)?;
        Ok(self.data[flat])
    }

    /// Writes the element at a multi-dimensional index.
    pub fn set(&mut self, index: &[usize], value: f32) -> Result<(), TensorError> {
        let flat = self.flat_index(index)?;
        self.data[flat] = value;
        Ok(())
    }

    /// Returns a view of the same elements under a new shape.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, TensorError> {
        let count = element_count(shape)?;
        if count != self.data.len() as u64 {
            return Err(TensorError::ElementCountMismatch {
                shape: shape.to_vec(),
                expected: count,
                found: self.data.len() as u64,
            });
        }
        Ok(Self {
            shape: shape.to_vec(),
            data: self.data.clone(),
        })
    }

    fn flat_index(&self, index: &[usize]) -> Result<usize, TensorError> {
        if index.len() != self.shape.len() {
            return Err(TensorError::RankMismatch {
                rank: self.shape.len(),
                found: index.len(),
            });
        }
        let mut flat = 0usize;
        for (dimension, (&idx, &size)) in index.iter().zip(&self.shape).enumerate() {
            if idx >= size {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx,
                    dimension,
                    size,
                });
            }
            flat = flat
                .checked_mul(size)
                .and_then(|v| v.checked_add(idx))
                .ok_or_else(|| TensorError::ShapeOverflow {
                    shape: self.shape.clone(),
                })?;
        }
        Ok(flat)
    }
}

fn element_count(shape: &[usize]) -> Result<u64, TensorError> {
    let mut count: u64 = 1;
    for &size in shape {
        count = count
            .checked_mul(size as u64)
            .ok_or_else(|| TensorError::ShapeOverflow {
                shape: shape.to_vec(),
            })?;
    }
    if count > usize::MAX as u64 {
        return Err(TensorError::ShapeOverflow {
            shape: shape.to_vec(),
        });
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_creates_shape_and_count() {
        let t = Tensor::zeros(&[2, 3]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.len(), 6);
        assert!(t.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn from_vec_roundtrips_get_and_set() {
        let mut t =
            Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(t.get(&[1, 0]).unwrap_or_else(|e| panic!("{e}")), 3.0);
        t.set(&[1, 1], 9.0).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(t.get(&[1, 1]).unwrap_or_else(|e| panic!("{e}")), 9.0);
    }

    #[test]
    fn reshape_preserves_elements() {
        let t = Tensor::from_vec(&[6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap_or_else(|e| panic!("{e}"));
        let r = t.reshape(&[2, 3]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.get(&[1, 2]).unwrap_or_else(|e| panic!("{e}")), 5.0);
    }

    #[test]
    fn from_vec_rejects_count_mismatch() {
        let err = Tensor::from_vec(&[2, 2], vec![1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            TensorError::ElementCountMismatch {
                expected: 4,
                found: 2,
                ..
            }
        ));
    }

    #[test]
    fn reshape_rejects_count_mismatch() {
        let t = Tensor::from_vec(&[4], vec![1.0, 2.0, 3.0, 4.0]).unwrap_or_else(|e| panic!("{e}"));
        let err = t.reshape(&[2, 3]).unwrap_err();
        assert!(matches!(
            err,
            TensorError::ElementCountMismatch {
                expected: 6,
                found: 4,
                ..
            }
        ));
    }

    #[test]
    fn rejects_rank_mismatch_and_out_of_bounds() {
        let t = Tensor::zeros(&[2, 3]).unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(
            t.get(&[1]),
            Err(TensorError::RankMismatch { rank: 2, found: 1 })
        ));
        assert!(matches!(
            t.get(&[2, 0]),
            Err(TensorError::IndexOutOfBounds {
                index: 2,
                dimension: 0,
                size: 2
            })
        ));
        assert!(matches!(
            t.get(&[0, 3]),
            Err(TensorError::IndexOutOfBounds {
                index: 3,
                dimension: 1,
                size: 3
            })
        ));
    }

    #[test]
    fn rejects_overflowing_shape() {
        let huge = [usize::MAX, usize::MAX];
        assert!(matches!(
            Tensor::zeros(&huge),
            Err(TensorError::ShapeOverflow { .. })
        ));
    }

    #[test]
    fn scalar_and_empty_shapes_are_supported() {
        let scalar = Tensor::from_vec(&[], vec![5.0]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(scalar.len(), 1);
        assert_eq!(scalar.get(&[]).unwrap_or_else(|e| panic!("{e}")), 5.0);
        let empty = Tensor::zeros(&[0]).unwrap_or_else(|e| panic!("{e}"));
        assert!(empty.is_empty());
    }
}
