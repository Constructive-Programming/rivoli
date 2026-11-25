//! `kernels/activation.hip`'s geometry: the row norms' tree and the activation quantizers.

use super::{F32, WAVE};

/// `rmsnorm_single`, `rmsnorm_centered_single`, `rmsnorm_weightless_batch`: the row's
/// threads, and therefore the LDS tree's leaf count. The tree halves from
/// `ROW_BLOCK / 2`, so it must be a power of two.
pub const ROW_BLOCK: u32 = 256;
/// One f32 per thread for the sum-of-squares tree (HIP: `__shared__ float red[256]`).
pub const ROW_LDS_BYTES: u32 = ROW_BLOCK * F32;
const _: () = assert!(ROW_BLOCK.is_power_of_two());

/// Elements per fp8 activation group — `kernel.py::act_quant`'s `block_size`, and the
/// launcher's `row_len % ACT_QUANT_BLOCK` refusal (1002).
pub const ACT_QUANT_BLOCK: u32 = 128;
/// Threads per `act_quant_f8` block: `ACT_QUANT_THREADS / WAVE` groups per block.
pub const ACT_QUANT_THREADS: u32 = 256;
const _: () = assert!(ACT_QUANT_THREADS.is_multiple_of(WAVE));
/// Elements per fp4 activation group (e8m0 scale per 32, the `.f4` format's group).
/// Device-only until the fp4 quantizer's launcher (Phase 1, with `moe_f4`) refuses by it.
pub const FP4_QUANT_BLOCK: u32 = 32;

/// gfx1151's workgroup LDS budget in bytes — the one number `sink_attend`'s `(n + ROW_BLOCK)`
/// f32 logit-plus-tree layout is bounded by. Named here rather than spelled `65536` in the
/// launcher AND derived again by the engine's `--ctx` gate: the cap is one fact.
pub const SINK_ATTEND_LDS_CAP: u32 = 65_536;
/// The largest `n` (keys a query attends) whose `n` logits plus the `ROW_BLOCK`-leaf max/sum
/// tree fit [`SINK_ATTEND_LDS_CAP`] — 16128 at `ROW_BLOCK` 256. The engine refuses a `--ctx`
/// past this at open time, and the launcher refuses an `n` past it (1001), so neither side
/// re-derives the arithmetic.
pub const SINK_ATTEND_MAX_N: u32 = SINK_ATTEND_LDS_CAP / F32 - ROW_BLOCK;
const _: () = assert!((SINK_ATTEND_MAX_N + ROW_BLOCK) * F32 == SINK_ATTEND_LDS_CAP);
