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

/// Fused single-pass `out` and per-row `lse = m + ln(sum_j exp(s_ij - m))`, `s = scale * q·k`,
/// one thread per query row (fallback for `head_dim > 128`).
#[cube(launch, address_type = "dynamic")]
fn attention_fwd_lse_kernel<E: Float, F: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    out: &mut Tensor<E>,
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

    let mut o_reg = Array::<F>::new(head_dim);
    for dd in 0..head_dim {
        o_reg[dd] = F::new(0.0_f32);
    }

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
            let alpha = (m - m_new).exp();
            let p = (s - m_new).exp();
            l = l * alpha + p;
            m = m_new;
            for dd in 0..head_dim {
                o_reg[dd] = o_reg[dd] * alpha + p * F::cast_from(v[k_base + dd]);
            }
        }
    }
    let inv_l = if l > F::new(0.0_f32) {
        F::new(1.0_f32) / l
    } else {
        F::new(0.0_f32)
    };
    for dd in 0..head_dim {
        out[q_base + dd] = E::cast_from(o_reg[dd] * inv_l);
    }
    lse[row] = m + l.ln();
}

/// Fused SIMT warp-tiled FlashAttention-2 forward (`out` + `lse`) for `head_dim <= 128`.
#[cube(launch, address_type = "dynamic")]
fn attention_fwd_lse_tiled_kernel<E: Float, F: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    out: &mut Tensor<E>,
    lse: &mut Tensor<F>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] causal: bool,
    #[define(E, F)] _dtypes: [ElemType; 2],
) {
    let c_d = comptime!(head_dim.div_ceil(32));
    let k_sh_len = comptime!(32 * head_dim);

    let seq_q = q.shape(q.rank() - 2);
    let seq_kv = k.shape(k.rank() - 2);
    let total_q_rows = lse.len();
    let total_bh = total_q_rows / seq_q;
    let q_tiles_per_bh = seq_q.div_ceil(4);

    let cube_id = CUBE_POS;
    let bh = cube_id / q_tiles_per_bh;
    if bh >= total_bh {
        terminate!();
    }

    let q_tile_idx = cube_id % q_tiles_per_bh;
    let i_base = q_tile_idx * 4;
    let w = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let tid = w * 32 + lane;

    let i = i_base + w;
    let valid_i = i < seq_q;
    let row_idx = bh * seq_q + i;
    let q_base = row_idx * head_dim;
    let k_bh_base = bh * seq_kv * head_dim;

    let mut q_reg = Array::<F>::new(c_d);
    let mut o_reg = Array::<F>::new(c_d);
    let scale_f = F::cast_from(scale);
    #[unroll]
    for c in 0..c_d {
        let dd = c * 32 + lane;
        if valid_i && dd < head_dim {
            q_reg[c] = F::cast_from(q[q_base + dd]) * scale_f;
        } else {
            q_reg[c] = F::new(0.0_f32);
        }
        o_reg[c] = F::new(0.0_f32);
    }

    let i_end = if seq_q < i_base + 4 { seq_q } else { i_base + 4 };
    let i_max = i_end - 1;
    let max_kv = if causal {
        if i_max + seq_kv < seq_q {
            0usize
        } else {
            let lim = i_max + seq_kv + 1 - seq_q;
            if seq_kv < lim { seq_kv } else { lim }
        }
    } else {
        seq_kv
    };

    let mut kv_sh = Shared::<[F]>::new_slice(k_sh_len);
    let mut p_sh = Shared::<[F]>::new_slice(128usize);
    let mut m = F::new(-1.0e30_f32);
    let mut l = F::new(0.0_f32);

    let mut j_tile = 0usize;
    while j_tile < max_kv {
        let rem = max_kv - j_tile;
        let tile_rows = if rem < 32usize { rem } else { 32usize };
        let k_tile_base = k_bh_base + j_tile * head_dim;
        let k_elems = tile_rows * head_dim;
        let mut idx = tid;
        while idx < k_elems {
            kv_sh[idx] = F::cast_from(k[k_tile_base + idx]);
            idx += 128;
        }

        sync_cube();

        if valid_i {
            let mut m_new = m;
            for t in 0..tile_rows {
                let j = j_tile + t;
                let masked = causal && j + seq_q > i + seq_kv;
                if !masked {
                    let k_off = t * head_dim;
                    let mut dot_part = F::new(0.0_f32);
                    #[unroll]
                    for c in 0..c_d {
                        let dd = c * 32 + lane;
                        if dd < head_dim {
                            dot_part += q_reg[c] * kv_sh[k_off + dd];
                        }
                    }
                    let s = plane_sum(dot_part);
                    if s > m_new {
                        m_new = s;
                    }
                    if lane == 0 {
                        p_sh[w * 32 + t] = s;
                    }
                } else if lane == 0 {
                    p_sh[w * 32 + t] = F::new(-1.0e30_f32);
                }
            }
            sync_plane();
            let alpha = (m - m_new).exp();
            #[unroll]
            for c in 0..c_d {
                o_reg[c] = o_reg[c] * alpha;
            }
            let j_lane = j_tile + lane;
            let masked_lane = lane >= tile_rows || (causal && j_lane + seq_q > i + seq_kv);
            let p_val = if masked_lane {
                F::new(0.0_f32)
            } else {
                (p_sh[w * 32 + lane] - m_new).exp()
            };
            p_sh[w * 32 + lane] = p_val;
            let l_tile = plane_sum(p_val);
            l = l * alpha + l_tile;
            m = m_new;
        }

        sync_cube();

        let mut idx_v = tid;
        while idx_v < k_elems {
            kv_sh[idx_v] = F::cast_from(v[k_tile_base + idx_v]);
            idx_v += 128;
        }

        sync_cube();

        if valid_i {
            for t in 0..tile_rows {
                let p = p_sh[w * 32 + t];
                let v_off = t * head_dim;
                #[unroll]
                for c in 0..c_d {
                    let dd = c * 32 + lane;
                    if dd < head_dim {
                        o_reg[c] += p * kv_sh[v_off + dd];
                    }
                }
            }
        }

        sync_cube();
        j_tile += 32;
    }

    if valid_i {
        let inv_l = if l > F::new(0.0_f32) {
            F::new(1.0_f32) / l
        } else {
            F::new(0.0_f32)
        };
        #[unroll]
        for c in 0..c_d {
            let dd = c * 32 + lane;
            if dd < head_dim {
                out[q_base + dd] = E::cast_from(o_reg[c] * inv_l);
            }
        }
        if lane == 0 {
            lse[row_idx] = m + l.ln();
        }
    }
}

/// Fused Tensor-Core WMMA (`16x16x16`) FlashAttention-2 forward (`out` + `lse`) for `bf16`/`f16`.
#[cube(launch, address_type = "dynamic")]
fn attention_fwd_lse_wmma_kernel<E: Float, F: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    out: &mut Tensor<E>,
    lse: &mut Tensor<F>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] causal: bool,
    #[define(E, F)] _dtypes: [ElemType; 2],
) {
    let num_d_tiles = comptime!(head_dim / 16);
    let o_reg_len = comptime!(num_d_tiles * 8);
    let k_sh_len = comptime!(16 * head_dim);
    let warp_sh_len = comptime!(4 * 256usize);

    let seq_q = q.shape(q.rank() - 2);
    let seq_kv = k.shape(k.rank() - 2);
    let total_q_rows = lse.len();
    let total_bh = total_q_rows / seq_q;
    let q_tiles_per_bh = seq_q.div_ceil(64);

    let cube_id = CUBE_POS;
    let bh = cube_id / q_tiles_per_bh;
    if bh >= total_bh {
        terminate!();
    }

    let q_tile_idx = cube_id % q_tiles_per_bh;
    let cta_q_base = q_tile_idx * 64;
    let w = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let tid = w * 32 + lane;
    let warp_q_base = cta_q_base + w * 16;
    let w_off = w * 256;
    let w_end = w_off + 256;
    let r = lane % 16;
    let c_half = (lane / 16) * 8;

    let q_bh_base = bh * seq_q * head_dim;
    let k_bh_base = bh * seq_kv * head_dim;
    let row_bh_base = bh * seq_q;

    let mut kv_sh = Shared::<[E]>::new_aligned_slice(k_sh_len, 16usize);
    let mut q_stage_sh = Shared::<[E]>::new_aligned_slice(warp_sh_len, 16usize);
    let mut s_sh = Shared::<[F]>::new_aligned_slice(warp_sh_len, 16usize);

    let mut q_frags = Sequence::<cmma::Matrix<E>>::new();
    let mut o_reg = Array::<F>::new(o_reg_len);
    #[unroll]
    for idx in 0..o_reg_len {
        o_reg[idx] = F::new(0.0_f32);
    }

    #[unroll]
    for d_blk in 0..num_d_tiles {
        let d_off = d_blk * 16;
        #[unroll]
        for step in 0..8usize {
            let idx = lane + step * 32;
            let rq = idx / 16;
            let cq = idx % 16;
            let q_idx = warp_q_base + rq;
            let d_idx = d_off + cq;
            if q_idx < seq_q && d_idx < head_dim {
                q_stage_sh[w_off + idx] = q[q_bh_base + q_idx * head_dim + d_idx];
            } else {
                q_stage_sh[w_off + idx] = E::cast_from(0.0_f32);
            }
        }
        sync_plane();
        let q_frag = cmma::Matrix::<E>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &q_stage_sh[w_off..w_end],
            16u32,
        );
        q_frags.push(q_frag);
        sync_plane();
    }

    let cta_q_end = if seq_q < cta_q_base + 64 {
        seq_q
    } else {
        cta_q_base + 64
    };
    let cta_q_max = cta_q_end - 1;
    let max_kv = if causal {
        if cta_q_max + seq_kv < seq_q {
            0usize
        } else {
            let lim = cta_q_max + seq_kv + 1 - seq_q;
            if seq_kv < lim { seq_kv } else { lim }
        }
    } else {
        seq_kv
    };

    let scale_f = F::cast_from(scale);
    let mut m = F::new(-1.0e30_f32);
    let mut l = F::new(0.0_f32);

    let mut kv_base = 0usize;
    while kv_base < max_kv {
        #[unroll]
        for d_blk in 0..num_d_tiles {
            let blk_off = d_blk * 256;
            let d_off = d_blk * 16;
            #[unroll]
            for s in 0..2usize {
                let elem = tid + s * 128;
                let r_kv = elem / 16;
                let c_d = elem % 16;
                let kv_idx = kv_base + r_kv;
                let d_idx = d_off + c_d;
                if kv_idx < seq_kv && d_idx < head_dim {
                    kv_sh[blk_off + elem] = k[k_bh_base + kv_idx * head_dim + d_idx];
                } else {
                    kv_sh[blk_off + elem] = E::cast_from(0.0_f32);
                }
            }
        }

        sync_cube();

        if warp_q_base < seq_q {
            let warp_q_max = warp_q_base + 15;
            let warp_active = !causal || (kv_base + seq_q <= warp_q_max + seq_kv);
            if warp_active {
                let s_frag = cmma::Matrix::<F>::from_value(
                    cmma::MatrixIdent::Accumulator,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::Undefined,
                    F::new(0.0_f32),
                );
                #[unroll]
                for d_blk in 0..num_d_tiles {
                    let b_start = d_blk * 256;
                    let b_end = b_start + 256;
                    let k_col = cmma::Matrix::<E>::from_slice(
                        cmma::MatrixIdent::B,
                        16usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::ColMajor,
                        &kv_sh[b_start..b_end],
                        16u32,
                    );
                    cmma::execute(&q_frags[d_blk], &k_col, &s_frag, &s_frag);
                }
                cmma::store(
                    &mut s_sh[w_off..w_end],
                    &s_frag,
                    16u32,
                    cmma::MatrixLayout::RowMajor,
                );
                sync_plane();

                let i = warp_q_base + r;
                let r_off = w_off + r * 16;
                if i < seq_q {
                    let mut m_new = m;
                    #[unroll]
                    for c in 0..16usize {
                        let j = kv_base + c;
                        let masked = j >= seq_kv || (causal && j + seq_q > i + seq_kv);
                        if !masked {
                            let s = s_sh[r_off + c] * scale_f;
                            if s > m_new {
                                m_new = s;
                            }
                        }
                    }
                    let alpha = (m - m_new).exp();
                    let mut l_new = l * alpha;
                    #[unroll]
                    for c in 0..16usize {
                        let j = kv_base + c;
                        let masked = j >= seq_kv || (causal && j + seq_q > i + seq_kv);
                        let p = if masked {
                            F::new(0.0_f32)
                        } else {
                            (s_sh[r_off + c] * scale_f - m_new).exp()
                        };
                        l_new += p;
                        if lane < 16 {
                            q_stage_sh[r_off + c] = E::cast_from(p);
                        }
                    }
                    m = m_new;
                    l = l_new;
                    #[unroll]
                    for idx in 0..o_reg_len {
                        o_reg[idx] = o_reg[idx] * alpha;
                    }
                } else if lane < 16 {
                    #[unroll]
                    for c in 0..16usize {
                        q_stage_sh[r_off + c] = E::cast_from(0.0_f32);
                    }
                }
                sync_plane();
            }
        }

        sync_cube();

        #[unroll]
        for d_blk in 0..num_d_tiles {
            let blk_off = d_blk * 256;
            let d_off = d_blk * 16;
            #[unroll]
            for s in 0..2usize {
                let elem = tid + s * 128;
                let r_kv = elem / 16;
                let c_d = elem % 16;
                let kv_idx = kv_base + r_kv;
                let d_idx = d_off + c_d;
                if kv_idx < seq_kv && d_idx < head_dim {
                    kv_sh[blk_off + elem] = v[k_bh_base + kv_idx * head_dim + d_idx];
                } else {
                    kv_sh[blk_off + elem] = E::cast_from(0.0_f32);
                }
            }
        }

        sync_cube();

        if warp_q_base < seq_q {
            let warp_q_max = warp_q_base + 15;
            let warp_active = !causal || (kv_base + seq_q <= warp_q_max + seq_kv);
            if warp_active {
                let p_frag = cmma::Matrix::<E>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::RowMajor,
                    &q_stage_sh[w_off..w_end],
                    16u32,
                );
                sync_plane();
                #[unroll]
                for d_blk in 0..num_d_tiles {
                    let b_start = d_blk * 256;
                    let b_end = b_start + 256;
                    let v_frag = cmma::Matrix::<E>::from_slice(
                        cmma::MatrixIdent::B,
                        16usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::RowMajor,
                        &kv_sh[b_start..b_end],
                        16u32,
                    );
                    let pv_frag = cmma::Matrix::<F>::from_value(
                        cmma::MatrixIdent::Accumulator,
                        16usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::Undefined,
                        F::new(0.0_f32),
                    );
                    cmma::execute(&p_frag, &v_frag, &pv_frag, &pv_frag);
                    cmma::store(
                        &mut s_sh[w_off..w_end],
                        &pv_frag,
                        16u32,
                        cmma::MatrixLayout::RowMajor,
                    );
                    sync_plane();
                    let r_off = w_off + r * 16 + c_half;
                    let reg_base = d_blk * 8;
                    #[unroll]
                    for step in 0..8usize {
                        o_reg[reg_base + step] += s_sh[r_off + step];
                    }
                    sync_plane();
                }
            }
        }

        sync_cube();
        kv_base += 16;
    }

    let i = warp_q_base + r;
    if i < seq_q {
        let inv_l = if l > F::new(0.0_f32) {
            F::new(1.0_f32) / l
        } else {
            F::new(0.0_f32)
        };
        let out_row_base = q_bh_base + i * head_dim;
        #[unroll]
        for d_blk in 0..num_d_tiles {
            let d_off = d_blk * 16 + c_half;
            let reg_base = d_blk * 8;
            #[unroll]
            for step in 0..8usize {
                let d_idx = d_off + step;
                if d_idx < head_dim {
                    out[out_row_base + d_idx] = E::cast_from(o_reg[reg_base + step] * inv_l);
                }
            }
        }
        if lane < 16 {
            lse[row_bh_base + i] = m + l.ln();
        }
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

    let address_type = query
        .required_address_type()
        .max(key.required_address_type())
        .max(value.required_address_type())
        .max(out.required_address_type())
        .max(lse.required_address_type());
    let dtypes = [dtype_to_storage_type(query.dtype), dtype_to_storage_type(DType::F32)];
    let scale = (d as f32).sqrt().recip();

    let elem_ty = dtypes[0];
    let has_wmma_16 = client
        .properties()
        .features
        .matmul
        .cmma
        .contains(&cubecl::ir::features::MmaConfig {
            a_type: elem_ty,
            b_type: elem_ty,
            cd_type: dtypes[1],
            m: 16,
            k: 16,
            n: 16,
        });

    if client.properties().hardware.plane_size_max == 32
        && has_wmma_16
        && d > 0
        && d <= 128
        && d % 16 == 0
    {
        let total_bh = b * h;
        let q_tiles_per_bh = sq.div_ceil(64);
        let num_cubes = total_bh * q_tiles_per_bh;
        let cube_dim = CubeDim::new_2d(32, 4);
        let cube_count = calculate_cube_count_elemwise(
            &client,
            num_cubes * (cube_dim.num_elems() as usize),
            cube_dim,
        );
        attention_fwd_lse_wmma_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            address_type,
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            lse.clone().into_tensor_arg(),
            scale,
            d,
            options.is_causal,
            dtypes,
        );
    } else if client.properties().hardware.plane_size_max == 32 && d <= 128 {
        let total_bh = b * h;
        let q_tiles_per_bh = sq.div_ceil(4);
        let num_cubes = total_bh * q_tiles_per_bh;
        let cube_dim = CubeDim::new_2d(32, 4);
        let cube_count = calculate_cube_count_elemwise(
            &client,
            num_cubes * (cube_dim.num_elems() as usize),
            cube_dim,
        );
        attention_fwd_lse_tiled_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            address_type,
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            lse.clone().into_tensor_arg(),
            scale,
            d,
            options.is_causal,
            dtypes,
        );
    } else {
        let units = b * h * sq;
        let cube_dim = CubeDim::new(&client, units);
        let cube_count = calculate_cube_count_elemwise(&client, units, cube_dim);
        attention_fwd_lse_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            address_type,
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            lse.clone().into_tensor_arg(),
            scale,
            d,
            options.is_causal,
            dtypes,
        );
    }

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
