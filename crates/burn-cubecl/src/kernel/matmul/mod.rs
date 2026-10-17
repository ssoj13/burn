mod base;
mod native;
#[cfg(feature = "autotune")]
mod tune;

/// Contains utilities for matmul operation
pub mod utils;

pub use base::*;
pub use native::{GemmBackend, gemm_backend, gemm_calls, reset_gemm_calls, set_gemm_backend};
#[cfg(feature = "autotune")]
pub use tune::*;
pub use utils::*;
