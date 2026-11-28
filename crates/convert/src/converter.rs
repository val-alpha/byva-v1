//! Conversion from a parsed checkpoint to `byva-weights-v1` weights.
//!
//! The model state dict is validated against the canonical tensor layout
//! implied by the checkpoint's config: names must match exactly (no extras,
//! none missing), every tensor must be fp32, element counts must match the
//! layout, and every value must be finite. Weight layout (`[out, in]` for
//! Linear, `[3d, d]` for the fused QKV projection) is identical on both
//! sides, so tensors are copied without transformation.

use std::collections::HashMap;

use byva_runtime::Weights;

use crate::checkpoint::Checkpoint;
use crate::error::ConvertError;
use crate::pickle::DType;

/// Converts a parsed checkpoint into validated runtime weights.
///
/// # Errors
/// Fails when a model tensor is unknown, missing, not fp32, has a wrong
/// element count, contains a non-finite value, or when the runtime rejects
/// the assembled weights.
pub fn checkpoint_to_weights(checkpoint: &Checkpoint) -> Result<Weights, ConvertError> {
    let layout = byva_runtime::weights::canonical_tensor_layout(&checkpoint.config)
        .map_err(ConvertError::from)?;
    let mut by_name: HashMap<&str, &_> = HashMap::with_capacity(checkpoint.model.len());
    for (name, tensor) in &checkpoint.model {
        if !layout.iter().any(|(canonical, _)| canonical == name) {
            return Err(ConvertError::UnknownModelTensor { name: name.clone() });
        }
        by_name.insert(name.as_str(), tensor);
    }

    let mut tensors = Vec::with_capacity(layout.len());
    for (name, elements) in &layout {
        let tensor = by_name
            .get(name.as_str())
            .ok_or(ConvertError::MissingModelTensor { name: name.clone() })?;
        if tensor.dtype != DType::Float {
            return Err(ConvertError::TensorDType {
                name: name.clone(),
                got: tensor.dtype.name().to_string(),
            });
        }
        if tensor.elements() != *elements {
            return Err(ConvertError::TensorShape {
                name: name.clone(),
                got: tensor.elements(),
                expected: *elements,
            });
        }
        let values = tensor.f32_elements();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ConvertError::NonFiniteTensor { name: name.clone() });
        }
        tensors.push((name.clone(), values));
    }

    Weights::from_tensors(checkpoint.config, tensors).map_err(ConvertError::from)
}
