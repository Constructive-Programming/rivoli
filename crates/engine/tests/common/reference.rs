//! The HOST side of a comparison: the deterministic draw source, the fixture draws over it,
//! and the reference formulas a kernel's output is scored against.
//!
//! One group because they share one argument: a seed means the same data at two call sites
//! only if the DRAW ORDER is shared, and a reference formula is one formula only while it is
//! spelled once. Every body below was moved out of a test file the day a second copy of it
//! appeared, and each says so in place.
//!
//! **Split out of `common/mod.rs` 2026-08-15** under the file-size gate. Bodies and their
//! comments travelled verbatim, and `mod.rs` re-exports this module with a glob, so every
//! `use common::{Lcg, block_scales, gemv_fp8_case, i8_weights, want_i8, …}` is untouched.

use rivoli_artifact::quant::{Fp8W, RowScaledW};

/// `1/sqrt(mean(x²) + eps)` — the factor all three norm forms share.
pub fn rms_inv(x: &[f32], eps: f32) -> f32 {
    1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt()
}

// **MOVED HERE from `glimmer_chain.rs` 2026-08-13**, the third helper to make this trip and for
// the same reason each time: a second host-side reader appeared (`glimmer_reference.rs`, which
// needs it to recover which embedding row the reference used) and jscpd reported the copy. The
// formula is `glimmer-architecture.md` §5's weightless norm; sharing it between two files that
// both transcribe the reference does not weaken either — what would weaken them is transcribing
// the CHAIN twice, and neither does that.
/// The weightless norm's shape and constants: `rows` segments of `d`, `eps` inside the mean,
/// `scale` after it.
///
/// One value rather than four trailing arguments, on [`super::Mla`]'s argument about its own six:
/// `(rows, d)` and `(eps, scale)` are each interchangeable to the type checker, and a transposed
/// pair moves the reference and the thing it scores together, so the comparison still agrees.
#[derive(Clone, Copy)]
pub struct Weightless {
    pub rows: usize,
    pub d: usize,
    pub eps: f32,
    pub scale: f32,
}

impl Weightless {
    /// The weightless form over `rows` segments of `d`, times `scale`. The QK-norm (per head) and
    /// the embedding norm (`rows = 1`, `d = hidden`, `scale = 1`) are the same operator.
    pub fn apply(self, x: &mut [f32]) {
        for r in 0..self.rows {
            let seg = &mut x[r * self.d..(r + 1) * self.d];
            let f = self.scale * rms_inv(seg, self.eps);
            seg.iter_mut().for_each(|v| *v *= f);
        }
    }
}

/// The sliding window's lower bound for a query at absolute position `pos`: `[pos - win + 1, pos]`,
/// INCLUSIVE of `pos` itself, and 0 on a global layer. Trap 14 is the `+ 1`.
///
/// **Shared, because two copies of a bound is how one of them drifts.** `glimmer_attend.rs` (host
/// oracle and mask comparison) and `glimmer_chain.rs` (the loop's reference) both need it, and
/// jscpd caught the second copy the moment it was written. The kernel has its own, in HIP, which
/// is the implementation those files exist to check — this is deliberately not that one.
pub fn window_lo(pos: usize, win: usize) -> usize {
    if win > 0 && pos >= win {
        pos - win + 1
    } else {
        0
    }
}

/// The oracles' deterministic input source.
pub struct Lcg(pub u64);

impl Lcg {
    /// Uniform in [-1, 1).
    ///
    /// `>> 32`, not `>> 33`. The old shift kept only 31 bits, so dividing by `u32::MAX`
    /// gave [0, 0.5) and `*2 - 1` gave [-1, 0) — **every sample negative**, for the whole
    /// life of both test files. In a matvec oracle that makes every `x[i]*w[i]` product
    /// positive, so the partial sums GROW instead of cancelling: `mx` inflates, the
    /// `1e-3 * mx` relative tolerance inflates with it, and the oracles were passing on
    /// roughly two orders of magnitude of headroom. It also meant no oracle here had ever
    /// exercised floating-point cancellation — the only regime where summation order
    /// matters, and the entire reason the kernels reduce with a fixed shuffle ladder
    /// (`__shfl_down` / `wave_sum`) instead of an atomic.
    pub fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// `n` positive per-block scales, `|f·0.1| + 0.01`. Every fp8 oracle draws them this way,
/// and the 0.01 floor is load-bearing rather than tidy: a tile whose
/// scale rounds to zero makes the comparison for that tile vacuous, and `assert_close`'s
/// relative tolerance would not show it.
pub fn block_scales(r: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n).map(|_| (r.f() * 0.1).abs() + 0.01).collect()
}

/// An fp8 GEMV case: e4m3 weights, the block-scale grid over them, the input, and the host
/// result.
///
/// `n_scales` was the caller's, not computed here, because the two backends spelled the scale
/// grid differently (`i_dim / block` against `i_dim.div_ceil(block)`); with one backend left it
/// was "a knob nothing turns and could be computed", noted here and **collapsed 2026-08-15**,
/// when CodeScene's excess-argument rule made the price of keeping it explicit. `div_ceil` on
/// BOTH axes, mirroring the kernel — the caller passed exactly this, and the ragged shape
/// (`o_dim = 130` at `block = 128`) is the one that made `o_dim / block` wrong.
///
/// The DRAW ORDER — weights, scales, x — is the part that has to be shared: it is what makes a
/// seed mean the same data at both call sites.
pub fn gemv_fp8_case(
    r: &mut Lcg,
    o_dim: usize,
    i_dim: usize,
    block: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let packed: Vec<u8> = (0..o_dim * i_dim)
        .map(|_| rivoli_core::num::f32_to_e4m3(r.f()))
        .collect();
    let scale = block_scales(r, o_dim.div_ceil(block) * i_dim.div_ceil(block));
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let mut want = vec![0.0f32; o_dim];
    rivoli_artifact::quant::matvec_fp8(&mut want, &x, Fp8W::new(&packed, &scale, block), i_dim);
    (packed, scale, x, want)
}

/// Random int8 weights and their per-row scales, drawn the way every int8 oracle here
/// draws them.
///
/// The `1e-4` floor is load-bearing for the same reason [`block_scales`]'s `0.01` is: a row
/// whose scale rounds to zero makes that row's comparison vacuous, and a relative tolerance
/// would not show it. The DRAW ORDER — weights, then scales — is what makes a seed mean the
/// same data at both call sites.
pub fn i8_weights(r: &mut Lcg, o_dim: usize, i_dim: usize) -> (Vec<u8>, Vec<f32>) {
    let packed: Vec<u8> = (0..o_dim * i_dim)
        .map(|_| (r.f() * 127.0) as i8 as u8)
        .collect();
    let scale: Vec<f32> = (0..o_dim).map(|_| (r.f() * 0.01).abs() + 1e-4).collect();
    (packed, scale)
}

/// `matvec_i8` into a fresh `o_dim` vector. Returned rather than written through an
/// out-param so the caller binds it in one line; the two int8 oracles generate their
/// weights differently and share only this step.
///
/// `dims` is `matvec_i8`'s own `[o_dim, i_dim]` rather than two trailing `usize` — the pair is
/// passed straight through, and two bare interchangeable dimensions at the end of an argument
/// list is the defect [`super::Mla`] exists to answer, made small (2026-08-15).
pub fn want_i8(x: &[f32], packed: &[u8], scale: &[f32], dims: [usize; 2]) -> Vec<f32> {
    let mut want = vec![0.0f32; dims[0]];
    rivoli_artifact::quant::matvec_i8(&mut want, x, RowScaledW::new(packed, scale), dims);
    want
}
