//! `reduce.hpp`: the wave and block reductions.

use crate::gpu::{fmaxf, shfl_down, shfl_xor, sync_threads, tid, tree};
use rivoli_abi::geom::WAVE;

/// `wave_sum`: the shfl_down ladder, 16 → 1. Every lane must be active.
#[inline(always)]
pub fn wave_sum(mut v: f32) -> f32 {
    let mut o = WAVE / 2;
    while o > 0 {
        v += shfl_down(v, o);
        o >>= 1;
    }
    v
}

/// `wave_max`: the shfl_xor ladder, 16 → 1, through `fmaxf` (`llvm.maxnum`).
#[inline(always)]
pub fn wave_max(mut v: f32) -> f32 {
    let mut o = WAVE / 2;
    while o > 0 {
        v = fmaxf(v, shfl_xor(v, o));
        o >>= 1;
    }
    v
}

/// `block_sum_lds` over a `BLOCK`-thread block, WITH its trailing barrier (so `red` can be
/// reused). `BLOCK` replaces the C's `blockDim.x` — the Phase 0c rule.
///
/// # Safety
/// `red` addresses `BLOCK` f32 of LDS, and all `BLOCK` threads call this.
#[inline(always)]
pub unsafe fn block_sum_lds<const BLOCK: u32>(v: f32, red: *mut f32) -> f32 {
    *red.add(tid() as usize) = v;
    sync_threads();
    tree::<BLOCK>(|t, s| *red.add(t) += *red.add(s));
    let total = *red;
    sync_threads();
    total
}
