//! Flash attention for training: forward that also yields the log-sum-exp, and the backward that
//! consumes it. Both avoid the `[B, H, Sq, Sk]` score matrix.
//!
//! Supported: no mask / bias / softcap, default scale (`1/sqrt(head_dim)`), `[B, H, S, D]` with
//! `seq_q == seq_k`, optional causal. Anything else is rejected with
//! [`AttentionSetupError::InvalidConfig`]; callers (autodiff) check
//! [`burn_backend::ops::attention::flash_bwd_supported`] first and keep the fallback otherwise.

use crate::{
    kernel::into_contiguous, ops::numeric::empty_device_contiguous_dtype, tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, Shape, TensorMetadata, ops::AttentionModuleOptions};
use core::sync::atomic::{AtomicU64, Ordering};
use cubek::attention::{
    backward::{BackwardConfig, flash_attention_backward as launch_backward},
    forward::{
        definition::{AttentionGlobalTypes, AttentionSetupError},
        flash_attention_forward_lse,
    },
};

static FLASH_FWD_CALLS: AtomicU64 = AtomicU64::new(0);
static FLASH_BWD_CALLS: AtomicU64 = AtomicU64::new(0);

/// Process-wide `(forward-with-lse, backward)` launch counts of the flash training path; lets a
/// test or bench prove autodiff took the fused node instead of `attention_fallback`.
pub fn flash_attention_calls() -> (u64, u64) {
    (
        FLASH_FWD_CALLS.load(Ordering::Relaxed),
        FLASH_BWD_CALLS.load(Ordering::Relaxed),
    )
}

// The kernels in cubek-attention index outputs as dense `row * head_dim`, so every output buffer
// is allocated with `empty_device_contiguous_dtype`: `empty_device_dtype` may row-pitch small
// last dims (e.g. bf16 d=4), which would leave the written data misaligned with the tensor's strides.

/// Typed gate for the options the flash training path implements.
fn check(
    q: &CubeTensor,
    k: &CubeTensor,
    v: &CubeTensor,
    options: &AttentionModuleOptions,
) -> Result<(), AttentionSetupError> {
    if burn_backend::ops::attention::flash_bwd_supported(
        &q.shape(),
        &k.shape(),
        &v.shape(),
        false,
        false,
        options,
    ) {
        Ok(())
    } else {
        Err(AttentionSetupError::InvalidConfig(Box::new(
            "flash attention training supports only [B,H,S,D] with seq_q == seq_k, default scale, \
             no mask, bias or softcap"
                .to_string(),
        )))
    }
}

pub(super) fn launch_flash_fwd_lse(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    options: AttentionModuleOptions,
) -> Result<(CubeTensor, CubeTensor), AttentionSetupError> {
    check(&query, &key, &value, &options)?;

    // The kernels index rows linearly.
    let query = into_contiguous(query);
    let key = into_contiguous(key);
    let value = into_contiguous(value);
    let [b, h, sq, d] = query.shape().dims::<4>();
    let client = query.client.clone();
    let out = empty_device_contiguous_dtype(
        client.clone(),
        query.device.clone(),
        query.shape(),
        query.dtype,
    );
    let lse = empty_device_contiguous_dtype(
        client.clone(),
        query.device.clone(),
        Shape::new([b, h, sq]),
        DType::F32,
    );

    let float = dtype_to_storage_type(query.dtype);
    let dtypes = AttentionGlobalTypes {
        query: float,
        key: float,
        value: float,
        mask: dtype_to_storage_type(DType::U8),
        out: float,
    };
    let scale = (d as f32).sqrt().recip();

    flash_attention_forward_lse(
        &client,
        query.binding(),
        key.binding(),
        value.binding(),
        out.clone().binding(),
        lse.clone().binding(),
        &dtypes,
        options.is_causal,
        scale,
    )?;

    Ok((out, lse))
}

/// Fused single-pass FlashAttention-2 forward returning `(out, lse)`.
pub fn flash_attention_lse(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    options: AttentionModuleOptions,
) -> Result<(CubeTensor, CubeTensor), AttentionSetupError> {
    check(&query, &key, &value, &options)?;
    FLASH_FWD_CALLS.fetch_add(1, Ordering::Relaxed);
    launch_flash_fwd_lse(query, key, value, options)
}

/// Flash backward from the forward's `out` and `lse`: `(dq, dk, dv)` in the query dtype.
///
/// The cubek kernels take one shared float dtype for q/k/v/out/dO (grads match), `lse` always f32.
pub fn flash_attention_bwd(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    out: CubeTensor,
    lse: CubeTensor,
    grad_out: CubeTensor,
    options: AttentionModuleOptions,
) -> Result<(CubeTensor, CubeTensor, CubeTensor), AttentionSetupError> {
    check(&query, &key, &value, &options)?;
    FLASH_BWD_CALLS.fetch_add(1, Ordering::Relaxed);
    if lse.dtype != DType::F32 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "flash attention backward expects an f32 lse".to_string(),
        )));
    }

    let client = query.client.clone();
    let [query, key, value, out, lse, grad_out] =
        [query, key, value, out, lse, grad_out].map(into_contiguous);
    let dq =
        empty_device_contiguous_dtype(client.clone(), query.device.clone(), query.shape(), query.dtype);
    let dk =
        empty_device_contiguous_dtype(client.clone(), key.device.clone(), key.shape(), key.dtype);
    let dv =
        empty_device_contiguous_dtype(client.clone(), value.device.clone(), value.shape(), value.dtype);

    let float = dtype_to_storage_type(query.dtype);
    let dtypes = AttentionGlobalTypes {
        query: float,
        key: float,
        value: float,
        mask: dtype_to_storage_type(DType::U8),
        out: float,
    };
    let head_dim = query.shape()[3];
    let config = BackwardConfig {
        causal: options.is_causal,
        ..BackwardConfig::from_head_dim(head_dim)
    };

    launch_backward(
        &client,
        query.binding(),
        key.binding(),
        value.binding(),
        out.binding(),
        lse.binding(),
        grad_out.binding(),
        dq.clone().binding(),
        dk.clone().binding(),
        dv.clone().binding(),
        &dtypes,
        config,
    )?;

    Ok((dq, dk, dv))
}
