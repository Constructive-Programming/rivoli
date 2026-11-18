//! The fp8-e4m3 TILE-scaled dot and the LDS LUT it decodes through. Port of `dot_fp8.hpp` at
//! tag `archive/hip-kernels`.
//!
//! fp8's scale is a tile grid shared by `block` output rows, so the CALLER picks the scale
//! row and the loop indexes it by `i >> blk_shift(block)`.
//!
//! **Fused sites, read from hipcc's gfx1151 ISA:** the quad is `rowview::quad`; the
//! accumulator is `fma(s, quad, acc)`; a tail column is `fma(x·l, s, acc)` (`v_mul` +
//! `v_fmac`).
//!
//! The C++ single-row `fp8_dot_strided` (unroll 8) and `fp8_dot_strided_r` (no unroll) were
//! one loop written twice; their own note says R=1 is bit-identical to the single form. Here
//! they are ONE function with the unroll depth a parameter, so each caller keeps the
//! schedule C++ measured for it.

use core::intrinsics::fmuladd;

use super::formats::e4m3f;
use super::reduce::{WAVE, wave_sum};
use super::rowview::{LaneSpan, RowsView, blk_shift, for_strided, quad};

/// Fill a 256-float LDS table with `e4m3f(byte)`. The caller passes its thread index and
/// barriers before use; a block of >= 256 threads fills it whole.
///
/// # Safety
/// `lut` must point at 256 writable floats.
#[inline(always)]
pub unsafe fn e4m3_lut_build(lut: *mut f32, tid: usize) {
    if tid < 256 {
        // SAFETY: `tid < 256`.
        unsafe { *lut.add(tid) = e4m3f(tid as u8) };
    }
}

/// One packed fp8 weight row: `w` = `packed[o*i_dim..]`, `scale` = the caller-picked tile
/// scale row, `lut` the block's `e4m3_lut_build` table. Element `i` is
/// `lut[w[i]] * scale[i >> blk_shift(block)]`.
#[derive(Clone, Copy)]
pub struct Fp8Row {
    pub w: *const u8,
    pub scale: *const f32,
    pub lut: *const f32,
    pub i_dim: usize,
    pub block: i32,
}

/// Unroll depth of the single-row split dot (`gemv_fp8_bf16` via `dot_fp8_wave`), M7
/// (`docs/investigations/v4-decode-decomposition.md` §M7 in the archive tree): un-unrolled,
/// each wave held one 128-B weight request in flight and the kernel ran at 2–2.8× its bytes;
/// depth 8 puts 1 KB in flight at VGPR 86 (= 96/wave, 16 waves/SIMD). The next granule
/// (> 96) drops to 12 waves, so a change that grows this body must re-read the ISA's VGPR
/// count. The R-row form stays at 1: its callers measured at budget and R multiplies the
/// register cost.
pub const FP8_SINGLE_ROW_UNROLL: usize = 8;

/// Strided fp8 dot of R activation rows against one weight row, NO cross-thread reduction:
/// Σ over this thread's columns of `x·lut[w]·scale`. A dword (4 columns, one tile scale)
/// per step while the tile is at least 4 wide, a per-column tail after.
///
/// # Safety
/// `w.w`, `w.scale`, `w.lut` and the `R` rows of `x` must be readable over `w.i_dim`
/// columns; `w.w` must be 4-byte aligned when `w.block >= 4`; `w.block` a power of two.
#[inline(always)]
pub unsafe fn fp8_dot_strided_r<const R: usize, const U: usize>(
    x: RowsView,
    w: Fp8Row,
    span: LaneSpan,
) -> [f32; R] {
    let mut acc = [0.0f32; R];
    // At block 1 or 2 a quad would straddle tiles and get i0's scale; the tail is correct.
    let n4 = if w.block >= 4 { w.i_dim >> 2 } else { 0 };
    let bsh = blk_shift(w.block) as u32;
    // SAFETY (both loops): the caller's obligation; `j < n4` keeps the dword read and
    // `i0 + 3 < i_dim` in bounds, `i < i_dim` the tail.
    unsafe {
        let lut = |b: u32| *w.lut.add((b & 0xff) as usize);
        for_strided::<U>(span.start, n4, span.stride, |j| {
            let p = *w.w.cast::<u32>().add(j);
            let i0 = j << 2;
            let s = *w.scale.add(i0 >> bsh);
            let l = [lut(p), lut(p >> 8), lut(p >> 16), lut(p >> 24)];
            for (r, a) in acc.iter_mut().enumerate() {
                let xr = x.at(r, i0);
                let xq = [*xr, *xr.add(1), *xr.add(2), *xr.add(3)];
                *a = fmuladd(s, quad(xq, l), *a);
            }
        });
        for_strided::<1>((n4 << 2) + span.start, w.i_dim, span.stride, |i| {
            let l = *w.lut.add(usize::from(*w.w.add(i)));
            let s = *w.scale.add(i >> bsh);
            for (r, a) in acc.iter_mut().enumerate() {
                *a = fmuladd(*x.at(r, i) * l, s, *a);
            }
        });
    }
    acc
}

/// One WAVE reduces one row for R activation rows: the strided dot over the wave's lanes,
/// then `wave_sum` (result on lane 0). `U` as `fp8_dot_strided_r`.
///
/// # Safety
/// As `fp8_dot_strided_r`; every lane of the wave must call it.
#[inline(always)]
pub unsafe fn dot_fp8_wave_r<const R: usize, const U: usize>(
    x: RowsView,
    w: Fp8Row,
    lane: usize,
) -> [f32; R] {
    // SAFETY: forwarded.
    let acc = unsafe {
        fp8_dot_strided_r::<R, U>(
            x,
            w,
            LaneSpan {
                start: lane,
                stride: WAVE,
            },
        )
    };
    acc.map(wave_sum)
}
