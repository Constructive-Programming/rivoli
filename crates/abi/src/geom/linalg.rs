//! `kernels/linalg.hip`'s geometry: the wave-per-row GEMVs and their split-K form.

use super::{F32, ROWS_PER_BLOCK, WAVE};

/// Every GEMV's block (the C's `ROW_BLOCK` macro): `ROWS_PER_BLOCK` waves, one output row
/// each — or, split-K, all of them striding one row. Not `geom::ROW_BLOCK`, the norms' tree
/// width, which happens to be the same 256 and is free to move on its own.
pub const GEMV_BLOCK: u32 = ROWS_PER_BLOCK * WAVE;
/// The e4m3 decode table every fp8 GEMV builds in LDS (HIP: `__shared__ float lut[256]`).
pub const FP8_LUT_ENTRIES: u32 = 256;
/// `gemv_fp8`'s dynamic LDS: the LUT only.
pub const GEMV_FP8_LDS_BYTES: u32 = FP8_LUT_ENTRIES * F32;
/// `gemv_fp8_splitk`'s dynamic LDS for `rows` token rows, carved by the kernel in this order:
/// `lut[FP8_LUT_ENTRIES] | part[rows][ROWS_PER_BLOCK]` (one partial per wave per row).
pub const fn gemv_fp8_splitk_lds_bytes(rows: u32) -> u32 {
    (FP8_LUT_ENTRIES + rows * ROWS_PER_BLOCK) * F32
}
// The LUT is filled one entry per thread, so a block must cover it.
const _: () = assert!(FP8_LUT_ENTRIES <= GEMV_BLOCK);
