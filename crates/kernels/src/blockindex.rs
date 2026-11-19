//! V4's block-sparse indexer pair — the twins of `kernels/blockindex.hip` at tag
//! `archive/hip-kernels`: the Hadamard-rotated fp4 activation simulator and the bf16 block scorer.
//! That file carries the contracts (the butterfly mapping, `1/sqrtf` over `rsqrtf`, one thread per
//! fp4 block, the scorer's three bf16 roundings and why each sits where it does). No `fma` here:
//! the rotation has no product feeding a sum, and the scorer is `#pragma clang fp contract(off)`.

use crate::gpu::{bid, for_i32, lds, tid};
use crate::indexer::score_blocks;
use crate::prelude::actquant::{
    FP4_MAX_INV, FP4_QUANT_AMAX_FLOOR, FP4_QUANT_BLOCK, fast_round_scale, fp4_quant_roundtrip,
};
use crate::prelude::formats::rbf16;
use crate::prelude::llvm::maxnum;
use crate::prelude::reduce::sync_threads;
use core::intrinsics::sqrt;
use rivoli_abi::geom::F4_ROTATE_BLOCK;
use rivoli_abi::kernarg::{F4Rotate, ScoreBlocks};

/// The in-LDS fast Walsh–Hadamard transform over `sh[0..d)`, `d` a power of two: butterfly `t` of
/// stage `h` pairs `j = (t / h)·2h + t mod h` with `j + h`.
#[inline(always)]
unsafe fn hadamard_lds(sh: *mut f32, d: i32) {
    let mut h = 1;
    while h < d {
        for_i32(tid(), d >> 1, F4_ROTATE_BLOCK, |t| {
            let t = t as i32;
            let off = t & (h - 1);
            let j = (((t - off) << 1) + off) as usize;
            let (a, b) = (*sh.add(j), *sh.add(j + h as usize));
            *sh.add(j) = a + b;
            *sh.add(j + h as usize) = a - b;
        });
        sync_threads();
        h <<= 1;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn act_quant_f4_rotated(a: F4Rotate) {
    let sh = lds::<f32>();
    let row = a.x.add(bid() as usize * a.d as usize);
    for_i32(tid(), a.d, F4_ROTATE_BLOCK, |i| *sh.add(i) = *row.add(i));
    sync_threads();
    hadamard_lds(sh, a.d);
    let scale = 1.0 / sqrt(a.d as f32);
    for_i32(tid(), a.d, F4_ROTATE_BLOCK, |i| {
        *sh.add(i) = rbf16(*sh.add(i) * scale)
    });
    sync_threads();
    for_i32(tid(), a.d / FP4_QUANT_BLOCK as i32, F4_ROTATE_BLOCK, |b| {
        let seg = sh.add(b * FP4_QUANT_BLOCK);
        let mut amax = FP4_QUANT_AMAX_FLOOR;
        for i in 0..FP4_QUANT_BLOCK {
            amax = maxnum(amax, (*seg.add(i)).abs());
        }
        let s = fast_round_scale(amax, FP4_MAX_INV);
        for i in 0..FP4_QUANT_BLOCK {
            *seg.add(i) = fp4_quant_roundtrip(*seg.add(i), s);
        }
    });
    sync_threads();
    for_i32(tid(), a.d, F4_ROTATE_BLOCK, |i| {
        *row.add(i) = rbf16(*sh.add(i))
    });
}

/// bf16 first, then the ReLU (model.py:426-427's order), each weighted term rounded to bf16 into
/// the f32 accumulator, and the one rounding `.sum` performs on the way out.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn index_score_blocks(a: ScoreBlocks) {
    score_blocks(a, |dot, w| rbf16(maxnum(rbf16(dot), 0.0) * w), rbf16);
}
