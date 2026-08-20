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
/// `1/sqrt(head_dim)` — the only scale a Glimmer attend kernel applies itself.
///
/// **The argument is the trap**: reading `hidden` here instead (6656 against head_dim 128) scales
/// every logit by 0.14x and stays fluent, which is why §9 trap 15 records that head_dim is NOT
/// `hidden / heads`. Spelled once so an oracle and the launch it scores cannot be handed different
/// scales — and shared, so the causal and the bidirectional attend cannot be handed different ones
/// either.
pub fn attn_scale(d: usize) -> f32 {
    1.0 / (d as f32).sqrt()
}

/// The operands of one attention case: Q laid out `[row][head][d]`, and a K/V cache laid out
/// `[slot][kv_head][d]`.
///
/// A struct rather than five parameters, on [`super::Mla`]'s argument: `hkv` and `d` are
/// interchangeable to the type checker, and CodeScene prices a five-`usize`-and-three-slice
/// signature as excess arguments before a reader ever gets to it.
pub struct AttendCase<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub hkv: usize,
    pub d: usize,
}

/// Which output row [`attend_head`] computes, which KV head it reads, and the INCLUSIVE span its
/// softmax runs over.
///
/// `n` indexes Q and the destination identically — both are `[row][head][d]` — so it is one value
/// rather than a `(row, h)` pair a caller could split two ways.
pub struct AttendSpan {
    pub n: usize,
    pub kvh: usize,
    /// `[lo, hi]`, **inclusive at both ends**, and deliberately not derived here. The causal
    /// attend passes `(window_lo(pos, win), pos)`; the drafter's bidirectional attend passes
    /// `(pos.saturating_sub(win), min(kv_len - 1, pos + win))`. Those two differ in BOTH edges —
    /// the causal lower bound is strict (`kv > q - win`, hence [`window_lo`]'s `+ 1`) while the
    /// bidirectional one is inclusive, and the causal upper bound is `pos` while the
    /// bidirectional one runs past it. **Making the span the caller's is what keeps that
    /// distinction visible at two call sites instead of hidden in one branch here.**
    pub span: (usize, usize),
}

/// One (query row, Q head)'s attention output: softmax over `s.span`, applied to V.
///
/// **HOISTED 2026-08-17 for M17c's block attend**, which needs this exact softmax over a
/// bidirectional span. `kernel_glimmer_attend.rs` spelled it locally and
/// `build.rs`'s duplication gate is what would have caught the second copy — the same sequence
/// [`window_lo`] above records, and the same resolution.
pub fn attend_head(c: &AttendCase<'_>, s: AttendSpan) -> Vec<f32> {
    let (hkv, d) = (c.hkv, c.d);
    let scale = attn_scale(d);
    let (lo, hi) = s.span;
    let qrow = &c.q[s.n * d..][..d];
    let logits: Vec<f32> = (lo..=hi)
        .map(|j| {
            let kr = &c.k[(j * hkv + s.kvh) * d..][..d];
            scale * (0..d).map(|i| qrow[i] * kr[i]).sum::<f32>()
        })
        .collect();
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f32> = logits.iter().map(|x| (x - mx).exp()).collect();
    let denom: f32 = w.iter().sum();
    let mut dst = vec![0.0f32; d];
    for (j, wj) in (lo..=hi).zip(&w) {
        let vr = &c.v[(j * hkv + s.kvh) * d..][..d];
        for (i, o) in dst.iter_mut().enumerate() {
            *o += wj / denom * vr[i];
        }
    }
    dst
}

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

/// `n` draws advancing `r` — the fixture draw for a case whose operands come from ONE stream
/// in a fixed order.
///
/// **HOISTED 2026-08-16 with M8's V4 oracles**, which is what the three `kernel_glimmer_*.rs`
/// files' debt notes asked for: each spelled its own body, none could share one, and
/// `build.rs`'s duplication gate was all that kept them apart. Their notes each named this
/// file, beside [`Lcg`] and [`block_scales`], as where the shared spelling belongs, and each
/// said to hoist it "the next time `common/` is open". M8 opened it.
///
/// Takes the STREAM rather than a salt because the three operands of one attend case must come
/// from one cursor: a seed means the same data at two call sites only while the draw ORDER is
/// shared, and three independently salted draws would let two of them silently coincide.
/// [`fill`] is the other half — same draw, salted, for the callers that reuse one operand
/// across cases and so cannot share a cursor.
pub fn draws(r: &mut Lcg, n: usize) -> Vec<f32> {
    std::iter::repeat_with(|| r.f()).take(n).collect()
}

/// `n` draws uniform in `[-scale, scale)` from a SALTED stream.
///
/// [`draws`] over a fresh [`Lcg`], scaled. Salted rather than sequential because the callers
/// that reach for this reuse one operand across several cases, and a shared cursor makes that
/// unspellable — which is exactly the property the stream form is chosen for elsewhere. The
/// scaling is applied after the draw rather than inside it, so the two spellings produce the
/// same stream and `fill(n, s, 1.0)` IS `draws(&mut Lcg(s), n)`.
///
/// Hoisted with [`draws`]; see its note for why all three copies existed and what they cost.
pub fn fill(n: usize, salt: u64, scale: f32) -> Vec<f32> {
    draws(&mut Lcg(salt), n)
        .into_iter()
        .map(|v| v * scale)
        .collect()
}

/// How many DISTINCT byte values appear at dword byte position `p` of a packed weight — the
/// coverage a byte-pattern sweep COUNTS rather than trusting from its construction.
///
/// Both V4 sweeps (the fp4 expert's and the fp8 GEMV's) build a weight whose bytes are meant to
/// walk every value at every position of the four-byte load, and both assert this against 256 or
/// 254. A second body of it is what `build.rs`'s duplication gate reported on 2026-08-16, and it
/// is right about the substance too: the two sweeps make the same claim, and a claim measured two
/// ways is two claims.
pub fn byte_position_coverage(w: &[u8], p: usize) -> usize {
    let mut seen = [false; 256];
    for b in w.iter().skip(p).step_by(4) {
        seen[*b as usize] = true;
    }
    seen.iter().filter(|&&s| s).count()
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
