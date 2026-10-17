use super::init_matmul_output;
use crate::{
    kernel::{into_contiguous, quantization::dequantize},
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, TensorMetadata};
use burn_std::{MatmulTransformAnalysis, MatmulTransformPolicy};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::server::{GemmDescriptor, GemmTranspose};
use cubek::{
    matmul::{
        definition::{MatmulElems, MatmulGlobalElems, MatmulSetupError},
        strategy::Strategy,
    },
    std::InputBinding,
};

#[cfg(feature = "autotune")]
use super::matmul_autotune;

/// The strategy to be used when launching a matmul kernel.
pub enum MatmulStrategy {
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// Cube implementation of matmul.
    Cube,
}

impl Default for MatmulStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return MatmulStrategy::Autotune;

        #[cfg(not(feature = "autotune"))]
        MatmulStrategy::Cube
    }
}

fn is_two_level(tensor: &CubeTensor) -> bool {
    match tensor.dtype {
        DType::QFloat(scheme) => burn_backend::quantization::global_scale_dtype(&scheme).is_some(),
        _ => false,
    }
}

fn maybe_dequantize(tensor: CubeTensor, dtype: DType) -> CubeTensor {
    if is_two_level(&tensor) {
        dequantize(tensor, dtype)
    } else {
        tensor
    }
}

fn classify_matrix_2d(
    rows: usize,
    cols: usize,
    sr: usize,
    sc: usize,
) -> Option<(GemmTranspose, i32)> {
    if sc == 1 && (rows <= 1 || sr >= cols) {
        let ld = if rows <= 1 {
            cols.max(1)
        } else {
            sr.max(cols).max(1)
        };
        i32::try_from(ld)
            .ok()
            .map(|ld| (GemmTranspose::NonTranspose, ld))
    } else if sr == 1 && (cols <= 1 || sc >= rows) {
        let ld = if cols <= 1 {
            rows.max(1)
        } else {
            sc.max(rows).max(1)
        };
        i32::try_from(ld)
            .ok()
            .map(|ld| (GemmTranspose::Transpose, ld))
    } else {
        None
    }
}

fn try_launch_vendor_gemm(
    mut lhs: CubeTensor,
    mut rhs: CubeTensor,
    out: &CubeTensor,
) -> Option<()> {
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        return None;
    }
    let backend = super::native::gemm_backend();
    if backend == super::native::GemmBackend::CuBlas && !out.client.has_gemm() {
        return None;
    }
    if lhs.dtype != rhs.dtype || lhs.dtype != out.dtype {
        return None;
    }
    let elem_ty = match out.dtype {
        DType::BF16 => ElemType::Float(FloatKind::BF16),
        DType::F16 => ElemType::Float(FloatKind::F16),
        DType::F32 | DType::Flex32 => ElemType::Float(FloatKind::F32),
        _ => return None,
    };

    let rank = out.meta.num_dims();
    if rank < 2 || lhs.meta.num_dims() != rank || rhs.meta.num_dims() != rank {
        return None;
    }

    let m_row = out.meta.shape()[rank - 2];
    let n_row = out.meta.shape()[rank - 1];
    let k_row = lhs.meta.shape()[rank - 1];
    if rhs.meta.shape()[rank - 2] != k_row
        || lhs.meta.shape()[rank - 2] != m_row
        || rhs.meta.shape()[rank - 1] != n_row
    {
        return None;
    }
    if m_row == 0 || n_row == 0 || k_row == 0 {
        return Some(());
    }

    // Ensure out is row-major contiguous in its last dimension (sc == 1).
    let out_sr = out.meta.strides()[rank - 2];
    let out_sc = out.meta.strides()[rank - 1];
    if out_sc != 1 || (m_row > 1 && out_sr < n_row) {
        return None;
    }
    let ldc = i32::try_from(if m_row <= 1 {
        n_row.max(1)
    } else {
        out_sr.max(n_row).max(1)
    })
    .ok()?;

    // Ensure lhs and rhs have a 2D GEMM layout (either OP_N or OP_T);
    // if not, materialize only the non-conforming operand as contiguous.
    if classify_matrix_2d(
        m_row,
        k_row,
        lhs.meta.strides()[rank - 2],
        lhs.meta.strides()[rank - 1],
    )
    .is_none()
    {
        lhs = into_contiguous(lhs);
    }
    if classify_matrix_2d(
        k_row,
        n_row,
        rhs.meta.strides()[rank - 2],
        rhs.meta.strides()[rank - 1],
    )
    .is_none()
    {
        rhs = into_contiguous(rhs);
    }

    let (transa, lda) = classify_matrix_2d(
        k_row,
        n_row,
        rhs.meta.strides()[rank - 2],
        rhs.meta.strides()[rank - 1],
    )?;
    let (transb, ldb) = classify_matrix_2d(
        m_row,
        k_row,
        lhs.meta.strides()[rank - 2],
        lhs.meta.strides()[rank - 1],
    )?;

    if backend == super::native::GemmBackend::Native {
        return super::native::launch_native_gemm(
            lhs,
            rhs,
            out,
            elem_ty,
            m_row,
            n_row,
            k_row,
            ldb as usize,
            lda as usize,
            ldc as usize,
            transb,
            transa,
        );
    }

    let m = i32::try_from(n_row).ok()?;
    let n = i32::try_from(m_row).ok()?;
    let k = i32::try_from(k_row).ok()?;

    let mut stride_a: i64 = 0;
    let mut stride_b: i64 = 0;
    let mut stride_c: i64 = 0;
    let mut batch_count: i32 = 1;
    let mut outer_offsets: Vec<(i64, i64, i64)> = Vec::new();

    if rank >= 3 {
        let b_inner = out.meta.shape()[rank - 3];
        if b_inner == 0 {
            return Some(());
        }
        batch_count = i32::try_from(b_inner).ok()?;
        stride_a = if rhs.meta.shape()[rank - 3] <= 1 {
            0
        } else {
            i64::try_from(rhs.meta.strides()[rank - 3]).ok()?
        };
        stride_b = if lhs.meta.shape()[rank - 3] <= 1 {
            0
        } else {
            i64::try_from(lhs.meta.strides()[rank - 3]).ok()?
        };
        stride_c = if b_inner <= 1 {
            0
        } else {
            i64::try_from(out.meta.strides()[rank - 3]).ok()?
        };

        if rank > 3 {
            let outer_dims = &out.meta.shape()[..rank - 3];
            let outer_total: usize = outer_dims.iter().product();
            if outer_total == 0 {
                return Some(());
            }
            if outer_total > 1 {
                outer_offsets.reserve(outer_total);
                for idx in 0..outer_total {
                    let mut rem = idx;
                    let mut off_a: i64 = 0;
                    let mut off_b: i64 = 0;
                    let mut off_c: i64 = 0;
                    for d in (0..rank - 3).rev() {
                        let dim = out.meta.shape()[d];
                        let coord = rem % dim;
                        rem /= dim;
                        if rhs.meta.shape()[d] > 1 {
                            off_a += (coord as i64) * (rhs.meta.strides()[d] as i64);
                        }
                        if lhs.meta.shape()[d] > 1 {
                            off_b += (coord as i64) * (lhs.meta.strides()[d] as i64);
                        }
                        if dim > 1 {
                            off_c += (coord as i64) * (out.meta.strides()[d] as i64);
                        }
                    }
                    outer_offsets.push((off_a, off_b, off_c));
                }
            }
        }
    }

    out.client.gemm(GemmDescriptor {
        a: rhs.handle.binding(),
        b: lhs.handle.binding(),
        c: out.handle.clone().binding(),
        dtype: elem_ty,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        transa,
        transb,
        stride_a,
        stride_b,
        stride_c,
        batch_count,
        outer_offsets,
    });
    super::native::record_cublas_call();
    Some(())
}

/// Launch a matmul kernel using the given strategy.
pub fn matmul(
    lhs: CubeTensor,
    rhs: CubeTensor,
    out: Option<CubeTensor>,
    strategy: MatmulStrategy,
    out_dtype: DType,
) -> Result<CubeTensor, MatmulSetupError> {
    let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));

    // No quantized matmul kernel applies a per-tensor scale, and the autotune candidates panic on
    // the level rather than decline it, taking the whole tuning run down with them.
    let mut lhs = maybe_dequantize(lhs, out_dtype);
    let rhs = maybe_dequantize(rhs, out_dtype);

    // A broadcast-rhs batched matmul that would tile poorly is folded into a
    // single matmul: `[.., b, m, k] @ [.., 1, k, n]` runs as `[.., 1, b*m, k]`
    // instead of `b` matmuls that each re-read the whole rhs. Pure metadata —
    // the launch operands share the handles, the returned tensor keeps the
    // broadcast shape.
    let mut out_launch = out.clone();
    if lhs.qparams.is_none() {
        let analysis = MatmulTransformAnalysis::from_metadata(&lhs.meta, &rhs.meta, &out.meta);
        let action = MatmulTransformPolicy::default().action(&analysis);
        action.apply(&mut lhs.meta);
        action.apply(&mut out_launch.meta);
    }

    if try_launch_vendor_gemm(lhs.clone(), rhs.clone(), &out_launch).is_some() {
        return Ok(out);
    }

    match strategy {
        MatmulStrategy::Cube => {
            launch_matmul(&Strategy::default(), lhs, rhs, out_launch)?;
            Ok(out)
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => {
            matmul_autotune(lhs, rhs, Some(out_launch), out_dtype);
            Ok(out)
        }
    }
}

pub(crate) fn launch_matmul_naive<S: Clone + Into<Strategy>>(
    strategy: &S,
    mut lhs: CubeTensor,
    mut rhs: CubeTensor,
    out: CubeTensor,
) -> Result<(), MatmulSetupError> {
    // Naive has very specific layout requirements for block scaled tensors, so we need to manually
    // dequantize if it fails to launch normally. This is because naive is assumed to always work.
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        match launch_matmul(strategy, lhs.clone(), rhs.clone(), out.clone()) {
            Err(_) => {
                if lhs.qparams.is_some() {
                    lhs = dequantize(lhs, out.dtype);
                }
                if rhs.qparams.is_some() {
                    rhs = dequantize(rhs, out.dtype);
                }
                launch_matmul(strategy, lhs, rhs, out)
            }
            Ok(_) => Ok(()),
        }
    } else {
        launch_matmul(strategy, lhs, rhs, out)
    }
}

pub(crate) fn launch_matmul<S: Clone + Into<Strategy>>(
    strategy: &S,
    lhs: CubeTensor,
    mut rhs: CubeTensor,
    out: CubeTensor,
) -> Result<(), MatmulSetupError> {
    let strategy: Strategy = strategy.clone().into();
    let client = &out.client;

    let lhs_quant_handles = lhs.quantized_handles();
    let out_dtype: DType = out.dtype;

    let (lhs_dtype, lhs_handle) = match lhs_quant_handles {
        None => {
            let lhs_dtype = lhs.dtype;
            (
                lhs_dtype,
                InputBinding::new(lhs.binding(), dtype_to_storage_type(lhs_dtype)),
            )
        }
        Some((data, scale)) => {
            let scheme = lhs.scheme();
            let data_dtype = data.dtype;
            let scale_dtype = scale.dtype;
            (
                out_dtype,
                InputBinding::quantized(
                    data.binding(),
                    scale.binding(),
                    lhs.meta.shape().clone(),
                    scheme,
                    dtype_to_storage_type(data_dtype),
                    dtype_to_storage_type(scale_dtype),
                ),
            )
        }
    };

    let rhs_quant_handles = rhs.quantized_handles();

    let (rhs_dtype, rhs_handle) = match rhs_quant_handles {
        None => (
            lhs_dtype,
            InputBinding::new(rhs.binding(), dtype_to_storage_type(lhs_dtype)),
        ),
        Some((data, scale)) => {
            // Extremely hacky fix to ensure naive can run in every case
            if matches!(
                strategy,
                Strategy::MultiLevel(cubek::matmul::multi_level::Strategy::Naive)
            ) && rhs.scheme().block_size().is_some()
            {
                rhs = dequantize(rhs.clone(), lhs_dtype);
                let rhs_dtype = rhs.dtype;
                (
                    lhs_dtype,
                    InputBinding::new(rhs.binding(), dtype_to_storage_type(rhs_dtype)),
                )
            } else {
                let scheme = rhs.scheme();
                let data_dtype = data.dtype;
                let scale_dtype = scale.dtype;
                (
                    out_dtype,
                    InputBinding::quantized(
                        data.binding(),
                        scale.binding(),
                        rhs.meta.shape().clone(),
                        scheme,
                        dtype_to_storage_type(data_dtype),
                        dtype_to_storage_type(scale_dtype),
                    ),
                )
            }
        }
    };

    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: dtype_to_storage_type(lhs_dtype),
        rhs: dtype_to_storage_type(rhs_dtype),
        out: dtype_to_storage_type(out_dtype),
    });

    cubek::matmul::launch::launch_ref(
        &strategy,
        client,
        lhs_handle,
        rhs_handle,
        out.clone().binding(),
        &mut dtypes,
    )?;

    Ok(())
}
