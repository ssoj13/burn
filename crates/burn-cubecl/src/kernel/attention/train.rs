//! Flash attention for training: forward that also yields the log-sum-exp, and the backward that
//! consumes it. Both avoid the `[B, H, Sq, Sk]` score matrix.
//!
//! Supported: no mask / bias / softcap, default scale (`1/sqrt(head_dim)`), `[B, H, S, D]` with
//! `seq_q == seq_k`, optional causal. Anything else is rejected with
//! [`AttentionSetupError::InvalidConfig`]; callers (autodiff) check
//! [`burn_backend::ops::attention::flash_bwd_supported`] first and keep the fallback otherwise.

use super::{AttentionStrategy, attention};
use crate::{
    kernel::into_contiguous, ops::numeric::empty_device_contiguous_dtype, tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, Shape, TensorMetadata, ops::AttentionModuleOptions};
use core::sync::atomic::{AtomicU64, Ordering};
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use cubek::attention::{
    backward::{BackwardConfig, flash_attention_backward as launch_backward},
    forward::definition::{AttentionGlobalTypes, AttentionSetupError},
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

// The kernels below index outputs as dense `row * head_dim`, so every output buffer is allocated
// with `empty_device_contiguous_dtype`: `empty_device_dtype` may row-pitch small last dims
// (e.g. bf16 d=4), which would leave the written data misaligned with the tensor's strides.

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

/// Per-row `lse = m + ln(sum_j exp(s_ij - m))`, `s = scale * q·k`, one thread per query row.
///
/// Online (single-pass) softmax state in f32 regardless of the input dtype. Fully masked rows give
/// `-inf`, as in the cubek CPU reference. The causal predicate is the backward kernels' own
/// (`j + seq_q > i + seq_kv` is masked), so forward and backward agree by construction.
#[cube(launch, address_type = "dynamic")]
fn attention_lse_kernel<E: Float, F: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    lse: &mut Tensor<F>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] causal: bool,
    #[define(E, F)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS;
    if row >= lse.len() {
        terminate!();
    }

    let seq_q = q.shape(q.rank() - 2);
    let seq_kv = k.shape(k.rank() - 2);
    let i = row % seq_q;
    let bh = row / seq_q;
    let q_base = row * head_dim;
    let k_bh_base = bh * seq_kv * head_dim;

    // A finite floor keeps `m - m_new` free of `inf - inf`; `l` stays 0 on fully masked rows so
    // `m + ln(0)` is -inf.
    let mut m = F::new(-1.0e30_f32);
    let mut l = F::new(0.0_f32);
    for j in 0..seq_kv {
        let masked = causal && j + seq_q > i + seq_kv;
        if !masked {
            let k_base = k_bh_base + j * head_dim;
            let mut dot = F::new(0.0_f32);
            for dd in 0..head_dim {
                dot += F::cast_from(q[q_base + dd]) * F::cast_from(k[k_base + dd]);
            }
            let s = dot * F::cast_from(scale);
            let m_new = if s > m { s } else { m };
            l = l * (m - m_new).exp() + (s - m_new).exp();
            m = m_new;
        }
    }
    lse[row] = m + l.ln();
}

/// Flash forward returning `(out, lse)`; `out` is the regular [`attention`] result (same strategy
/// selection, autotune included), `lse` is `[B, H, Sq]` f32.
pub fn flash_attention_lse(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    options: AttentionModuleOptions,
) -> Result<(CubeTensor, CubeTensor), AttentionSetupError> {
    check(&query, &key, &value, &options)?;
    FLASH_FWD_CALLS.fetch_add(1, Ordering::Relaxed);

    let out = attention(
        query.clone(),
        key.clone(),
        value,
        None,
        None,
        options,
        AttentionStrategy::default(),
    )?;

    // The kernel indexes rows linearly.
    let query = into_contiguous(query);
    let key = into_contiguous(key);
    let [b, h, sq, d] = query.shape().dims::<4>();
    let lse = empty_device_contiguous_dtype(
        query.client.clone(),
        query.device.clone(),
        Shape::new([b, h, sq]),
        DType::F32,
    );

    let units = b * h * sq;
    let client = query.client.clone();
    let cube_dim = CubeDim::new(&client, units);
    let cube_count = calculate_cube_count_elemwise(&client, units, cube_dim);
    let address_type = query
        .required_address_type()
        .max(key.required_address_type())
        .max(lse.required_address_type());
    let dtypes = [dtype_to_storage_type(query.dtype), dtype_to_storage_type(DType::F32)];
    attention_lse_kernel::launch(
        &client,
        cube_count,
        cube_dim,
        address_type,
        query.into_tensor_arg(),
        key.into_tensor_arg(),
        lse.clone().into_tensor_arg(),
        (d as f32).sqrt().recip(),
        d,
        options.is_causal,
        dtypes,
    );

    Ok((out, lse))
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
    let dq = empty_device_contiguous_dtype(client.clone(), query.device.clone(), query.shape(), query.dtype);
    let dk = empty_device_contiguous_dtype(client.clone(), key.device.clone(), key.shape(), key.dtype);
    let dv = empty_device_contiguous_dtype(client.clone(), value.device.clone(), value.shape(), value.dtype);

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
