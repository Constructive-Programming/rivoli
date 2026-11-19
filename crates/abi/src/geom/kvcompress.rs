//! `kernels/kvcompress.hip`'s geometry (tag `archive/hip-kernels`).

use super::{ELEMENTWISE_BLOCK, F32};

/// `gemm_bf16`'s block: one wave per output element, eight per block. IS the elementwise block,
/// because its launcher sizes the grid with the elementwise helper over `m · n · WAVE` threads.
pub const GEMM_BF16_BLOCK: u32 = ELEMENTWISE_BLOCK;
/// The compressor kernels' block: the deposit's grid unit, and the one block per pooled row at
/// prefill and decode (one feature per thread, strided). IS the elementwise block, because the
/// deposit's launcher sizes its grid with the elementwise helper.
pub const COMPRESS_BLOCK: u32 = ELEMENTWISE_BLOCK;
/// Entries pooled per output block (`coff · ratio`): 8 at ratio 4, 128 at ratio 128 — the largest
/// the shipped config produces. It sizes the LDS pointer tables; the launchers refuse more (1004).
pub const COMPRESS_ENTS_MAX: u32 = 128;
/// The pooling kernels' dynamic LDS, carved in this order: `kvp | scp | apep` (`COMPRESS_ENTS_MAX`
/// 8-byte pointers each), then `red[COMPRESS_BLOCK]` f32 for the norm.
pub const COMPRESS_LDS_BYTES: u32 = 3 * COMPRESS_ENTS_MAX * 8 + COMPRESS_BLOCK * F32;
const _: () =
    assert!(COMPRESS_BLOCK.is_power_of_two() && GEMM_BF16_BLOCK.is_multiple_of(super::WAVE));
