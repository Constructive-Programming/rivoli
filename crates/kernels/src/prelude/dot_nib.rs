//! The two GROUP-scaled nibble dots: int4 (`nibble − 8`, f32 scale per 128) and OCP MX fp4
//! (e2m1 nibble, e8m0 scale per 32). Port of `dot_i4.hpp` and `dot_f4.hpp` at tag
//! `archive/hip-kernels`, as ONE loop over a `NibbleRow` format.
//!
//! The C++ kept them apart, arguing a shared template would need an "is this scale exact?"
//! policy. Read against the bodies, neither loop needs one: both dword passes compute
//! `acc += s·(x0·n0 + x1·n1 + x2·n2 + x3·n3)` with one hoisted group scale per 8 columns,
//! and both tails compute `acc += x·n·s` left to right per element — `f4_tail_accum`
//! hoisted the scale LOAD out of its row loop, which moves no value. What differs is the
//! decode, the scale's type and the group width, and those are the trait. The device sweep
//! (rust-kernels.md §5) compares each instantiation against its own C++ original.
//!
//! **Fused sites** are `dot_fp8.rs`'s: `rowview::quad`, then `fma(s, quad, acc)`; tail
//! `fma(x·n, s, acc)`.
//!
//! The C++ single-row int4 dot (`dot_i4_wave`, `linalg.hip`) is `dot_nib_wave_r::<1, 1, I4Row>`:
//! its own note says R=1 is bit-identical to it, and it was never unrolled.
//!
//! **Not reproduced: fp4's AS1 (`global`) pointer typing — an L2 (`moe_f4`) obligation.**
//! C++ types `F4Row`'s spans `address_space(1)` (`gu8p`) so pointers LOADED from an expert
//! descriptor lower to `global_load`, not `flat_load` (M3a: a measured cost on an issue-bound
//! loop). Rust cannot name an address space. Measured 2026-11-18 on a descriptor-loop +
//! atomic kernel (moe_f4's shape): this port emits `flat_load` for the weight dword and scale
//! byte; `assert_unchecked(!is_shared && !is_private)` does not survive to InferAddressSpaces
//! (as `__builtin_assume` did not in C++); `llvm.amdgcn.raw.buffer.load` compiles but is a
//! different instruction. Bits are unaffected either way. rust-kernels.md §5.

use core::intrinsics::fmuladd;

use super::formats::e2m1f;
use super::reduce::{WAVE, wave_sum};
use super::rowview::{F32x4, RowsView, for_strided, quad};

/// What separates the two nibble formats: the weight decode, the scale's storage type and
/// decode, and the group width.
pub trait NibFormat {
    /// The group-scale element as stored.
    type Scale: Copy;
    /// log2 of the columns per group scale; at least 3, so a dword's 8 columns share one.
    const GROUP_SHIFT: u32;
    /// The weight of the LOW nibble of `p`; higher bits are ignored.
    fn decode(p: u32) -> f32;
    fn scale(raw: Self::Scale) -> f32;
}

/// int4 (quant.rs `I4_GROUP`; equality UNCHECKED until host and device share a crate):
/// weight = `nibble − 8`, one f32 scale per 128 columns.
pub struct I4;
/// OCP MX fp4 (quant.rs `F4_GROUP`, v4oracle's e2m1/e8m0): the NON-uniform e2m1 codebook,
/// one e8m0 power-of-two scale per 32 columns.
pub struct F4;

pub const I4_GROUP_SHIFT: u32 = 7;
pub const I4_GROUP: usize = 1 << I4_GROUP_SHIFT;
pub const F4_GROUP_SHIFT: u32 = 5;
pub const F4_GROUP: usize = 1 << F4_GROUP_SHIFT;

impl NibFormat for I4 {
    type Scale = f32;
    const GROUP_SHIFT: u32 = I4_GROUP_SHIFT;
    fn decode(p: u32) -> f32 {
        ((p & 0xF) as i32 - 8) as f32
    }
    fn scale(raw: f32) -> f32 {
        raw
    }
}

impl NibFormat for F4 {
    type Scale = u8;
    const GROUP_SHIFT: u32 = F4_GROUP_SHIFT;
    fn decode(p: u32) -> f32 {
        e2m1f(p)
    }
    fn scale(raw: u8) -> f32 {
        e8m0f(raw)
    }
}

/// One packed nibble weight row and its group-scale row: column `i` is nibble `i & 1` of
/// byte `i >> 1` (LOW nibble first — reading it high-first is a permutation inside each
/// group that survives every summary statistic), group `g`'s scale is `scale[g]`.
/// int4: `w = packed[o*(dim+1)/2..]`; fp4: `w = packed[o*dim/2..]`, `scale + o*ceil(dim/32)`.
pub struct NibRow<F: NibFormat> {
    pub w: *const u8,
    pub scale: *const F::Scale,
    pub dim: usize,
}
// Derived Clone/Copy would demand `F: Copy`; the format is a marker, the row is three words.
impl<F: NibFormat> Clone for NibRow<F> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<F: NibFormat> Copy for NibRow<F> {}

pub type I4Row = NibRow<I4>;
pub type F4Row = NibRow<F4>;

/// e8m0 group-scale byte → f32, `2^(b−127)` — here with its only callers, as in C++. Both
/// ends are spelled out: `b == 0` is `2^-127` (subnormal — `b << 23` would give +0) and
/// `0xff` is the format's NaN.
#[inline(always)]
pub fn e8m0f(b: u8) -> f32 {
    let t = u32::from(b) << 23;
    let at_least_subnormal = if t > 1 << 22 { t } else { 1 << 22 };
    let bits = at_least_subnormal | if b == 0xff { 1 << 22 } else { 0 };
    f32::from_bits(bits)
}

/// Columns one dword-pass step covers across the wave: 8 per lane.
const DWORD_STEP: usize = WAVE * 8;

/// Unroll depth of the batched dword pass, measured on both formats 2026-08-09 (the C++
/// notes at tag `archive/hip-kernels`, `dot_i4.hpp`/`dot_f4.hpp`): un-unrolled, the loop
/// drained its loads in-body (one iteration in flight). int4 depth 4 was +12.6% at R=1
/// (169.2 → 190.6 GB/s) and +16.4% at R=2; fp4 146.6 → 195.3 GB/s (`dot_bench v4res`).
/// Depth 4 costs 125 VGPR on `moe_gateup_f4` (10 waves/SIMD), so a body that grows must
/// re-read the ISA. The single-row int4 dot (`linalg.hip`'s `dot_i4_wave`) was never
/// unrolled and calls this with `U = 1`.
pub const NIBBLE_BATCHED_UNROLL: usize = 4;

/// R activation rows against ONE read of the weight row: a dword (8 nibbles = 8 columns, one
/// group) per lane while the row is 4-byte aligned, then a per-column tail from where it
/// stopped. Two accumulator chains in the dword pass (low/high four columns); the tail folds
/// into the first only. Result on lane 0 after `wave_sum(a0 + a1)`.
///
/// # Safety
/// `w`'s bytes and scales must be readable over `w.dim` columns and the `R` rows of `x` over
/// the same; when the row is 4-byte aligned, `x.at(r, col)` must be 16-byte aligned at every
/// dword column (an unchecked obligation, as in C++ — misalignment is UB, not a fallback).
/// Every lane of the wave must call it.
#[inline(always)]
pub unsafe fn dot_nib_wave_r<const R: usize, const U: usize, F: NibFormat>(
    x: RowsView,
    w: NibRow<F>,
    lane: usize,
) -> [f32; R] {
    let (mut a0, mut a1) = ([0.0f32; R], [0.0f32; R]);
    let aligned = w.w as usize & 3 == 0;
    // Where the dword pass stops and the tail resumes: the two are a PARTITION of the columns.
    let base = if aligned {
        w.dim / DWORD_STEP * DWORD_STEP
    } else {
        0
    };
    // SAFETY (both passes): the caller's obligation; the dword pass reads columns
    // `col..col+8 <= base <= dim`, the tail `i < dim`.
    unsafe {
        let scale = |col: usize| F::scale(*w.scale.add(col >> F::GROUP_SHIFT));
        for_strided::<U>(0, base, DWORD_STEP, |step| {
            let col = step + lane * 8;
            let p = *w.w.cast::<u32>().add(col >> 3);
            let s = scale(col);
            let n_lo: [f32; 4] = core::array::from_fn(|k| F::decode(p >> (4 * k)));
            let n_hi: [f32; 4] = core::array::from_fn(|k| F::decode(p >> (4 * k + 16)));
            for (r, (lo, hi)) in a0.iter_mut().zip(a1.iter_mut()).enumerate() {
                let xr = x.at(r, col);
                *lo = fmuladd(s, quad(F32x4::load(xr).0, n_lo), *lo);
                *hi = fmuladd(s, quad(F32x4::load(xr.add(4)).0, n_hi), *hi);
            }
        });
        for_strided::<1>(base + lane, w.dim, WAVE, |i| {
            let n = F::decode(u32::from(*w.w.add(i >> 1)) >> (4 * (i & 1)));
            let s = scale(i);
            for (r, a) in a0.iter_mut().enumerate() {
                *a = fmuladd(*x.at(r, i) * n, s, *a);
            }
        });
    }
    core::array::from_fn(|r| wave_sum(a0[r] + a1[r]))
}
