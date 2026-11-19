//! `kernels/attn.hip`'s geometry. Every block size here is a launcher-fixed size the kernel
//! strides by instead of reading blockDim (Phase 0c); the tag's `attn.hip` carries the
//! measurements behind each value (HB 8 → 16, the split plan), which this does not repeat.

use super::{F32, WAVE};

/// MLA: heads per block, one wave each — the cache re-read multiplier (⌈H/HB⌉×).
pub const MLA_HB: u32 = 16;
/// MLA: KV tokens staged into LDS per step.
pub const MLA_TILE: u32 = 16;
/// MLA: per-lane accumulator registers; caps `kvl` at `MLA_ACC_REGS * WAVE` (guard 1004).
pub const MLA_ACC_REGS: u32 = 16;
/// MLA: `mla_latent_attend`'s block, `MLA_HB` waves.
pub const MLA_ATTEND_THREADS: u32 = MLA_HB * WAVE;
/// MLA: the split ceiling — bounds grid.y, and the partial scratch is laid out at it.
pub const MLA_MAX_SPLITS: u32 = 16;
/// MLA: `mla_attend_combine`'s block, striding the `kvl` columns.
pub const MLA_COMBINE_THREADS: u32 = 256;
/// `mla_attend_combine`'s LDS: `w[MLA_MAX_SPLITS] | inv_l` (the C's two static arrays).
pub const MLA_COMBINE_LDS_BYTES: u32 = (MLA_MAX_SPLITS + 1) * F32;

/// `gather_attn_shared_kv`'s block, and its LDS tree's leaf count.
pub const GATHER_ATTN_THREADS: u32 = 256;
/// Output dims per `gather_attn_shared_kv` thread; caps `d` at 1024 (guard 1002).
pub const GATHER_ATTN_ACC: u32 = 4;
const _: () = assert!(GATHER_ATTN_THREADS.is_power_of_two());

/// GQA: one wave per (query row, Q head).
pub const GQA_THREADS: u32 = WAVE;
/// GQA: KV rows staged per step; LDS is `2 * GQA_TILE * d` floats.
pub const GQA_TILE: u32 = 16;
/// GQA: per-lane accumulator/query registers; caps `d` at `GQA_THREADS * GQA_ACC` (1002).
pub const GQA_ACC: u32 = 8;
// The C's static_assert: the largest admitted `d` keeps the tile inside gfx1151's 64 KiB.
const _: () = assert!(2 * GQA_TILE * GQA_THREADS * GQA_ACC * F32 <= 65536);

/// `mha_attend`'s block, and its LDS tree's leaf count.
pub const MHA_THREADS: u32 = 256;
const _: () = assert!(MHA_THREADS.is_power_of_two());
