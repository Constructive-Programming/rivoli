//! `kernels/residual.hip`'s geometry: the mHC pair, the attention residual and the gated
//! residual collapse.

use super::{ELEMENTWISE_BLOCK, F32, WAVE};

/// DeepSeek-V4's residual-copy count. A checkpoint constant, not a knob — the mHC weights
/// are PACKED at `(2 + HC_MULT) * HC_MULT` (`kernels/residual.hip`'s header at tag `archive/hip-kernels` has the whole
/// argument); the launchers refuse any other `hc` (1002).
pub const HC_MULT: u32 = 4;
/// Mix rows per token: `pre`, `post`, then the `HC_MULT x HC_MULT` comb.
pub const HC_MIX: u32 = (2 + HC_MULT) * HC_MULT;
/// `hc_pre`'s block — one wave per mix row in ONE round (24 rows, 32 waves).
pub const HC_PRE_BLOCK: u32 = 1024;
/// The width `hc_pre`'s sum-of-squares phase is FROZEN at, independent of the block — the
/// 2026-08-08 widening to 1024 kept this at 256 so no golden moved.
pub const HC_RED: u32 = 256;
/// `hc_pre`'s dynamic LDS bytes, for the five arrays HIP declared `__shared__`, which the
/// kernel's `HcLds::carve` lays out in this order from these same constants:
/// `red[HC_RED] | mixes[HC_MIX] | pre[HC_MULT] | post[HC_MULT] | comb[HC_MULT²]`.
pub const HC_PRE_LDS_BYTES: u32 = (HC_RED + HC_MIX + 2 * HC_MULT + HC_MULT * HC_MULT) * F32;
const _: () = assert!(HC_RED.is_power_of_two() && HC_PRE_BLOCK.is_multiple_of(WAVE));
const _: () = assert!(HC_RED <= HC_PRE_BLOCK && HC_MIX <= HC_PRE_BLOCK / WAVE);

/// `attn_res`'s block: one block per token, a 256-leaf tree per source.
pub const ATTN_RES_BLOCK: u32 = 256;
/// The largest source stack the kernel's private probability array holds; the launcher
/// refuses more (1003). Kimi-K3 needs at most 9.
pub const ATTN_RES_MAX_SRC: u32 = 16;
/// `attn_res`'s dynamic LDS bytes, carved by the kernel in this order:
/// `red_ss[ATTN_RES_BLOCK] | red_dot[ATTN_RES_BLOCK] | prob[ATTN_RES_MAX_SRC]` (2112 B — the
/// C's static arrays, and hipcc's group segment for the kernel).
pub const ATTN_RES_LDS_BYTES: u32 = (2 * ATTN_RES_BLOCK + ATTN_RES_MAX_SRC) * F32;
const _: () = assert!(ATTN_RES_BLOCK.is_power_of_two());

/// The widest residual stream the scale row may be written for — block 0's first `n_r`
/// threads write it, so `n_r` must fit one block. The launcher refuses more (1003).
pub const GATED_RESIDUAL_MAX_NR: u32 = 64;
const _: () = assert!(GATED_RESIDUAL_MAX_NR <= ELEMENTWISE_BLOCK);
