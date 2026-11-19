//! The attention kernels — the twins of `kernels/attn.hip` at tag `archive/hip-kernels`. That
//! file carries each kernel's contract, its model and its traps (the GQA lower edges, the
//! sink that is not a key, the NaN a fully-masked query owes); they are not repeated here.
//! Split by shape, not by the C's order:
//!
//! | module  | kernels                                   | shape                              |
//! |---------|-------------------------------------------|------------------------------------|
//! | `mla`   | `mla_latent_attend`, `mla_attend_combine` | flash over the fp8 latent cache, split-KV + combine |
//! | `gqa`   | `gqa_attend`, `gqa_block_attend`          | flash, one wave per (row, head)    |
//! | `dense` | `gather_attn_shared_kv`, `mha_attend`     | two-pass softmax staged in LDS     |
//!
//! **Fusion.** The MLA half of the C is compiled under hipcc's `-ffp-contract=fast-honor-
//! pragmas`, so each of its fused sites is an explicit [`crate::gpu::fma`] placed where
//! hipcc's ISA shows one (listed at the kernel). From the gather kernel on, the C carries
//! `#pragma clang fp contract(off)`: hipcc's ISA for those four kernels has no fused product
//! of the source's arithmetic (its `v_fma`s are `expf`'s range reduction and the IEEE divide,
//! which the same LLVM intrinsics produce here), and Rust never contracts — so they carry no
//! `fma` at all.

mod dense;
mod gqa;
mod mla;

use crate::prelude::llvm::wave_shuffle;
use crate::prelude::reduce::{WAVE, lane_id, wave_sum};

/// `__shfl(__shfl_down-ladder(part), 0, WAVE)`: the wave's shuffle-down sum (the C's
/// `for (off = 16; off > 0; off >>= 1) part += __shfl_down(part, off)`, same order), then
/// lane 0's total on every lane — HIP's `__shfl(v, 0, 32)` is a shuffle from
/// `(lane & ~31) + 0`.
#[inline(always)]
fn wave_total(part: f32) -> f32 {
    wave_shuffle(wave_sum(part), lane_id() & !(WAVE as u32 - 1))
}

/// An online softmax's running state over `N` lane-owned register columns: the unnormalised
/// accumulator, the running max and the running sum (both replicated across the wave, which
/// sees one broadcast score per row).
struct Running<const N: usize> {
    acc: [f32; N],
    m: f32,
    l: f32,
}

impl<const N: usize> Running<N> {
    /// Before the first row: `acc = 0`, `m = −inf`, `l = 0`.
    const START: Self = Self {
        acc: [0.0; N],
        m: f32::NEG_INFINITY,
        l: 0.0,
    };
}
