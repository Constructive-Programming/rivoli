//! `kernels/mla.hip`'s geometry (tag `archive/hip-kernels`).

use super::{ELEMENTWISE_BLOCK, F32, ROWS_PER_BLOCK, WAVE};

// The row kernels (`rmsnorm_batch`, `qk_norm`, `rope_adjacent`) launch at `ROW_BLOCK` with the
// `ROW_LDS_BYTES` tree — the activation family's: one row per 256-thread block, the same `red`.

/// i-quads per head in `mla_absorb_fp8`: one thread per (head, quad of four columns). Both the
/// kernel and its launcher's grid read THIS one — a short grid leaves columns unwritten,
/// silently, which is why C++ kept one definition (moved here from the device prelude, where it
/// waited for this crate, 2026-11-19).
#[inline(always)]
pub const fn absorb_nquad(kvl: i32) -> i32 {
    kvl.wrapping_add(3) >> 2
}

/// The wave-per-row kernels' block (`mla_value_fp8`, `gemv_fp8_bf16`): `ROWS_PER_BLOCK` waves.
pub const WAVE_ROWS_BLOCK: u32 = ROWS_PER_BLOCK * WAVE;
/// The e4m3 decode table every fp8 kernel builds in LDS (`dot_fp8::e4m3_lut_build`), 256 f32 —
/// C++'s static `__shared__ float lut[256]`.
pub const FP8_LUT_LDS_BYTES: u32 = 256 * F32;
const _: () = assert!(
    ELEMENTWISE_BLOCK >= 256,
    "the absorb's LUT build needs 256 threads"
);
const _: () = assert!(WAVE_ROWS_BLOCK >= 256, "the LUT build needs 256 threads");
