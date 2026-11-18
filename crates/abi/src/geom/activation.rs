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
