//! Launch geometry shared by a Rust kernel and its launcher — block sizes, dynamic-LDS
//! layouts, and the checkpoint constants a launcher refuses and a kernel indexes by.
//!
//! Every item is read by BOTH sides (a host-only constant stays beside its launcher). A block size here is what the launcher passes as
//! `blockDim` and what the kernel strides by in place of reading it (the crate doc carries
//! the Phase 0c rule), and an LDS layout here is what the launcher sizes the dynamic
//! allocation by and what the kernel carves it with. A kernel that wants a different value
//! changes it HERE, and the launcher moves with it.

/// f32 is the only LDS element type so far; named so the byte counts below read as such.
const F32: u32 = 4;

// ── 1-D elementwise kernels: one thread per output element ──────────────────────────────

/// `swiglu`, `swiglu_clamped_bf16`, `situ_glu_f32`, `hc_post`, `gated_residual_collapse_f32`:
/// `ceil(n / 256)` blocks of 256. The kernels index `blockIdx.x * ELEMENTWISE_BLOCK +
/// threadIdx.x`, so a launch at any other width would skip or repeat elements.
pub const ELEMENTWISE_BLOCK: u32 = 256;

// ── one-block-per-row reductions ─────────────────────────────────────────────────────────

/// `rmsnorm_single`, `rmsnorm_centered_single`, `rmsnorm_weightless_batch`: the row's
/// threads, and therefore the LDS tree's leaf count. The tree halves from
/// `ROW_BLOCK / 2`, so it must be a power of two — asserted below.
pub const ROW_BLOCK: u32 = 256;
/// One f32 per thread for the sum-of-squares tree (HIP: `__shared__ float red[256]`).
pub const ROW_LDS_BYTES: u32 = ROW_BLOCK * F32;

// ── act_quant_f8: one wave per 128-element group ─────────────────────────────────────────

/// gfx11 is wave32. A kernel that assumes a wave width assumes THIS one.
pub const WAVE: u32 = 32;
/// Elements per fp8 activation group — `kernel.py::act_quant`'s `block_size`, and the
/// launcher's `row_len % ACT_QUANT_BLOCK` refusal (1002).
pub const ACT_QUANT_BLOCK: u32 = 128;
/// Threads per `act_quant_f8` block: `ACT_QUANT_THREADS / WAVE` groups per block.
pub const ACT_QUANT_THREADS: u32 = 256;

// ── mHC (`hc_pre` / `hc_post`) ───────────────────────────────────────────────────────────

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

// ── attn_res ─────────────────────────────────────────────────────────────────────────────

/// `attn_res`'s block: one block per token, a 256-leaf tree per source.
pub const ATTN_RES_BLOCK: u32 = 256;
/// The largest source stack the kernel's private probability array holds; the launcher
/// refuses more (1003). Kimi-K3 needs at most 9.
pub const ATTN_RES_MAX_SRC: u32 = 16;
/// `attn_res`'s dynamic LDS bytes, carved by the kernel in this order:
/// `red_ss[ATTN_RES_BLOCK] | red_dot[ATTN_RES_BLOCK] | prob[ATTN_RES_MAX_SRC]` (2112 B — the
/// C's static arrays, and hipcc's group segment for the kernel).
pub const ATTN_RES_LDS_BYTES: u32 = (2 * ATTN_RES_BLOCK + ATTN_RES_MAX_SRC) * F32;

// ── gated_residual_collapse_f32 ──────────────────────────────────────────────────────────

/// The widest residual stream the scale row may be written for — block 0's first `n_r`
/// threads write it, so `n_r` must fit one block. The launcher refuses more (1003).
pub const GATED_RESIDUAL_MAX_NR: u32 = 64;

// ── touch_region (the fetch path's page-touch) ───────────────────────────────────────────

/// `touch_region`'s block (the kernel's grid-stride start index is `bid * TOUCH_BLOCK`).
pub const TOUCH_BLOCK: u32 = 256;

// The tree reductions halve from `block / 2` and lose leaves at any other shape; the wave
// math assumes whole waves. Checked where the constants are defined, so an edit that breaks
// either fails to compile on both sides at once.
const _: () = assert!(ROW_BLOCK.is_power_of_two() && HC_RED.is_power_of_two());
const _: () = assert!(ATTN_RES_BLOCK.is_power_of_two());
const _: () = assert!(ACT_QUANT_THREADS.is_multiple_of(WAVE) && HC_PRE_BLOCK.is_multiple_of(WAVE));
const _: () = assert!(HC_RED <= HC_PRE_BLOCK && HC_MIX <= HC_PRE_BLOCK / WAVE);
const _: () = assert!(GATED_RESIDUAL_MAX_NR <= ELEMENTWISE_BLOCK);
