//! `kernels/recurrent.hip`'s geometry: the padded reduction block both masked kernels share.

use super::F32;

/// `reduce.hpp::next_pow2` — the smallest power of two `>= width` (1 at 0), the block the
/// masked kernels pad to so `block_sum_lds`'s halving ladder drops nothing. Called by the
/// launcher for the launch AND by the kernel for its ladder, so the two cannot disagree; the
/// launchers bound `width <= 1024` (1002) before calling it.
pub const fn reduce_block(width: u32) -> u32 {
    width.next_power_of_two()
}

/// `gated_delta_recurrent_f32`'s LDS: `qn | kn | alpha | red`, each `head_dim` floats.
pub const fn gated_delta_lds_bytes(head_dim: u32) -> u32 {
    4 * head_dim * F32
}

/// `gdn_recurrent_f32`'s LDS: `qn (dk) | kn (dk) | red (block)` floats.
pub const fn gdn_lds_bytes(dk: u32, block: u32) -> u32 {
    (2 * dk + block) * F32
}

/// `rmsnorm_gate_heads_f32`'s LDS: one reduction float per thread of the padded block.
pub const fn reduce_lds_bytes(block: u32) -> u32 {
    block * F32
}
