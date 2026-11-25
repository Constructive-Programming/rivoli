//! The OCP MX FP8 dot with a ROW-WISE ue8m0 scale — Inkling-2's `[1, 32]` grid, which is
//! `dot_fp8`'s tile grid at one row. No C twin: Inkling-2 has no hipcc kernel
//! (owner direction 2026-11-24), so this helper is scored against the first-party
//! `rivoli_artifact::inkling_dequant::Mxfp8` through an oracle suite rather than against
//! the bit-identity gate.
//!
//! **The scale is a raw U8 exponent, not an e4m3 code (trap T1).** `Mxfp8::decode_row`
//! multiplies the decoded e4m3 value by `2^(scale − 127)`; reading the byte through the
//! fp8 LUT instead would decode it as an e4m3 number (`0x80` → `-0`, `0x70` → `224`) and
//! produce finite, plausible, wrong output. Hence [`e8m0f`], the same decode the fp4 path
//! uses for its group scales.
//!
//! **Fused sites** are `dot_fp8.rs`'s: `rowview::quad`, then `fma(s, quad, acc)`; a tail
//! column is `fma(x·l, s, acc)`.
//!
//! Unlike `dot_fp8`, the tile is one row, so a dword (4 columns) never straddles two scale
//! groups: `col % 4 == 0` and the group is 32 wide, so all four bytes share ONE scale byte,
//! which is what lets the fast path hoist it out of the four multiplies.

use core::intrinsics::fmuladd;

use super::dot_nib::e8m0f;
use super::reduce::{WAVE, wave_sum};
use super::rowview::{RowsView, aligned_base, for_strided, quad};

/// log2 of the columns one ue8m0 scale byte governs: the `[1, 32]` grid of
/// `inkling_dequant::Mxfp8` (`LOAD:808–810`).
pub const MXFP8_GROUP_SHIFT: u32 = 5;
/// Columns per scale byte — 32, the group `rivoli_abi::geom::F4_GROUP` also names.
pub const MXFP8_GROUP: usize = 1 << MXFP8_GROUP_SHIFT;

/// One MXFP8 weight row: `w` = `packed[o*i_dim..]` (one e4m3 byte per column), `scale` =
/// `scales[o*ceil(i_dim/32)..]` (one ue8m0 exponent per 32 columns), and `lut` the block's
/// `e4m3_lut_build` table. Column `i` is `lut[w[i]] · 2^(scale[i >> 5] − 127)`.
#[derive(Clone, Copy)]
pub struct MxFp8Row {
    pub w: *const u8,
    pub scale: *const u8,
    pub lut: *const f32,
    pub dim: usize,
}

/// Columns one dword-pass step covers across the wave: 4 per lane, 32 lanes.
const DWORD_STEP: usize = WAVE * 4;

/// One WAVE reduces one weight row: Σ over the wave's columns of `x·lut[w]·2^(s−127)`, the
/// result on lane 0 after [`wave_sum`]. A dword (4 columns, ONE scale byte) per lane while
/// the row is 4-byte aligned, a per-column tail after — the partition `dot_fp8`'s twin
/// makes, for the same reason (a quad cannot index two scales).
///
/// # Safety
/// `w.w`, `w.scale`, `w.lut` and row 0 of `x` must be readable over `w.dim` columns;
/// `w.lut` must hold 256 floats. Every lane of the wave must call it.
#[inline(always)]
pub unsafe fn dot_mxfp8_wave<const U: usize>(x: RowsView, w: MxFp8Row, lane: usize) -> f32 {
    let mut acc = 0.0f32;
    // Where the dword pass stops and the tail resumes: they PARTITION the columns, and an
    // unaligned row takes the tail whole (`rowview::aligned_base`).
    let base = aligned_base(w.w, w.dim, DWORD_STEP);
    // SAFETY (both passes): the caller's obligation; the dword pass reads columns
    // `col..col+4 <= base <= dim`, the tail `i < dim`.
    unsafe {
        let lut = |b: u32| *w.lut.add((b & 0xff) as usize);
        for_strided::<U>(0, base, DWORD_STEP, |step| {
            let col = step + lane * 4;
            let p = *w.w.cast::<u32>().add(col >> 2);
            // `col % 4 == 0` and a 32-column group: the dword's four bytes share this byte.
            let s = e8m0f(*w.scale.add(col >> MXFP8_GROUP_SHIFT));
            let l = [lut(p), lut(p >> 8), lut(p >> 16), lut(p >> 24)];
            let xr = x.at(0, col);
            let xq = [*xr, *xr.add(1), *xr.add(2), *xr.add(3)];
            acc = fmuladd(s, quad(xq, l), acc);
        });
        for_strided::<1>(base + lane, w.dim, WAVE, |i| {
            let l = *w.lut.add(usize::from(*w.w.add(i)));
            let s = e8m0f(*w.scale.add(i >> MXFP8_GROUP_SHIFT));
            acc = fmuladd(*x.at(0, i) * l, s, acc);
        });
    }
    wave_sum(acc)
}
