//! The by-value views the wave-cooperative dots take. Port of `rowview.hpp` at tag
//! `archive/hip-kernels`.
//!
//! The C++ aliasing record (why the store side never enters a struct) carries over by
//! construction: accumulators are `[f32; R]` VALUES, returned or threaded by value, so no
//! pointer to them exists to alias the read-only spans. Rust raw pointers carry no
//! `noalias`, so the spans lose the `__restrict__` the C++ members kept (which clang only
//! honours "in many cases"); that is a schedule question for the timing gate, not a bits
//! question, because no loop here writes memory.
//!
//! Sizes and strides are `usize`: every launcher guards them non-negative before a kernel
//! runs, and an unsigned index lets the address arithmetic skip the sign extension clang's
//! `int` needed.

use core::intrinsics::fmuladd;

/// R activation rows: row `r` starts at `x + r*stride` (stride unread at R=1).
#[derive(Clone, Copy)]
pub struct RowsView {
    pub x: *const f32,
    pub stride: usize,
}

impl RowsView {
    /// Column `i` of row `r`.
    ///
    /// # Safety
    /// The element must lie inside the rows' allocation (it is an in-bounds offset).
    #[inline(always)]
    pub unsafe fn at(self, r: usize, i: usize) -> *const f32 {
        // SAFETY: the caller's obligation.
        unsafe { self.x.add(r * self.stride + i) }
    }
}

/// The columns one thread owns: `start`, then every `stride`-th.
#[derive(Clone, Copy)]
pub struct LaneSpan {
    pub start: usize,
    pub stride: usize,
}

/// log2 of a power-of-two scale-tile size (`31 - clz`), so hot loops index the tile scale
/// with a shift. `blk_shift(0) == -1`, as `__clz(0) == 32` makes it in C++.
#[inline(always)]
pub fn blk_shift(block: i32) -> i32 {
    31 - block.leading_zeros() as i32
}

/// Sixteen bytes of activation, the `float4` the dword paths load.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct F32x4(pub [f32; 4]);

impl F32x4 {
    /// # Safety
    /// `p` must be 16-byte aligned and point at 4 readable floats. A misaligned `p` is
    /// undefined behaviour, as the C++ `*(const float4*)` cast is — an unchecked caller
    /// obligation the launchers carry.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> Self {
        // SAFETY: the caller's obligation above.
        unsafe { *p.cast::<Self>() }
    }
}

/// `x0·w0 + x1·w1 + x2·w2 + x3·w3` in hipcc's contraction order, read from its gfx1151 ISA
/// (`v_mul` + three `v_fmac`): the LEFT product of `a·b + c·d` fuses, `fma(a, b, c·d)`, then
/// each further `+ e·f` is `fma(e, f, ·)`. Every dword path's inner sum is this shape.
#[inline(always)]
pub fn quad(x: [f32; 4], w: [f32; 4]) -> f32 {
    fmuladd(
        x[3],
        w[3],
        fmuladd(x[2], w[2], fmuladd(x[0], w[0], x[1] * w[1])),
    )
}

/// `start, start+stride, …` while `< end`, with the body issued `U` at a time: the
/// `#pragma unroll U` Rust does not have. The `U` copies are STRAIGHT-LINE code (const-folded
/// `if`s), then a scalar remainder. Measured 2026-11-18 (ISA census against hipcc): the
/// first spelling, an inner `for u in 0..U`, left the R=2 int4 dword pass with 8 activation
/// `global_load_b128` where hipcc's `#pragma unroll 4` has 20 — LLVM did not fully unroll it.
/// The fold order is ascending either way, so `U` moves only the schedule, never a value.
#[inline(always)]
pub fn for_strided<const U: usize>(
    start: usize,
    end: usize,
    stride: usize,
    mut body: impl FnMut(usize),
) {
    const { assert!(U >= 1 && U <= 8, "unroll depth 1..=8") };
    let mut i = start;
    if U > 1 {
        while i + (U - 1) * stride < end {
            macro_rules! copies {
                ($($k:literal)*) => { $( if $k < U { body(i + $k * stride); } )* };
            }
            copies!(0 1 2 3 4 5 6 7);
            i += U * stride;
        }
    }
    while i < end {
        body(i);
        i += stride;
    }
}
