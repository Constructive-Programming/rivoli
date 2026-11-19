//! `kernels/indexer.hip`'s geometry. The run-time block sizes (`index_score`, `index_head_route`,
//! `index_pool_norm_f32`) are not here: each launcher derives one per launch and hands it over in
//! `kernarg::Blocked`.
//!
//! The `.hip` file named here is deleted from the tree; cite it at tag `archive/hip-kernels`.

use super::F32;

/// `layernorm`'s block: one block reduces Σx and Σx² as two 256-leaf trees.
pub const LAYERNORM_BLOCK: u32 = 256;
/// `layernorm`'s dynamic LDS: `sred[LAYERNORM_BLOCK] | sqred[LAYERNORM_BLOCK]`.
pub const LAYERNORM_LDS_BYTES: u32 = 2 * LAYERNORM_BLOCK * F32;
const _: () = assert!(LAYERNORM_BLOCK.is_power_of_two());

/// The MISA router's block-pool width, in tokens per pooled key. Device-only: the pool kernel
/// indexes by it and the engine's `indexer::MISA_BLOCK` must agree (it is not a launch extent).
pub const MISA_BLOCK: u32 = 1024;

/// `index_topk`'s ONE workgroup.
pub const TOPK_BLOCK: u32 = 256;
/// Radix bins per 8-bit pass.
pub const TOPK_BINS: u32 = 256;
/// `index_topk`'s dynamic LDS: `hist[TOPK_BINS] | scan[TOPK_BLOCK] | pivot | need` (u32 each).
pub const TOPK_LDS_BYTES: u32 = (TOPK_BINS + TOPK_BLOCK + 2) * 4;
// The compaction packs two within-chunk counts (each <= TOPK_BLOCK) into 16-bit halves.
const _: () = assert!(TOPK_BLOCK.is_power_of_two() && TOPK_BLOCK < 1 << 16);
