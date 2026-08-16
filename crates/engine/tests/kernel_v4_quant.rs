//! **DeepSeek-V4's fp8 activation quantizer and its fp8 GEMV**, scored against the frozen host
//! oracle: `act_quant_f8`, `act_quant_f8_prefix` and `gemv_fp8_bf16`.
//!
//! Ported from `old:tests/f4_kernel.rs` §2 and §4 and from `old:tests/f4_attn.rs`'s
//! subnormal-tie and ABI-guard tests. **One file because all three are the same operand
//! travelling one step**: the quantizer writes `e4m3(v/s)·s` back into an f32 buffer and the
//! GEMV reads exactly that, so a disagreement about the ROUNDING RULE and a disagreement about
//! the SUMMATION ORDER present identically at any consumer downstream. Separating them into two
//! files would put the two halves of one number in two places.
//!
//! # Why these three are gated BITWISE and almost nothing else is
//!
//! Every comparison in this file is exact, and that is a property of the arithmetic rather than
//! optimism:
//!
//! * `act_quant_f8` is an `fmaxf` reduction, `fast_round_scale`'s bit surgery, one `fdiv`, a
//!   clamp, and an encode/decode pair that are both exhaustively specified rules. `build.rs`
//!   compiles the kernels with `-O3` and no `-ffast-math`, so the divide stays a real `fdiv`.
//!   A tolerance here would hide exactly the one-ulp tie disagreement the fixtures are built to
//!   expose.
//! * `gemv_fp8_bf16` reduces, so it cannot be compared bitwise against a SEQUENTIAL oracle —
//!   and it is not. [`serial_fold`] is a transliteration of the KERNEL's own fold (per-lane
//!   strided accumulation in the emitted contraction, then `wave_sum`'s shfl-down ladder, then
//!   `rbf16`), so the comparison pins the summation ORDER rather than tolerating it. A relative
//!   bound here would wave through an unroll that split the accumulator chain, which is the
//!   exact change this gate exists for.
//!
//! # What this file CANNOT detect
//!
//! * **`f2e4m3_rne`'s saturation arm.** `s` is `2^ceil(log2(amax/448)) >= amax/448`, so
//!   `|x|/s <= 448` always and neither the clamp nor the `a >= 464` early return can fire from
//!   here or from the model. They are the format's own bounds, executed by nothing in this
//!   suite.
//! * **NaN payload propagation.** The two e4m3 NaN codes (`0x7f`, `0xff`) are EXCLUDED from the
//!   GEMV sweeps: NaN payloads through an FMA chain are not contractual across host and device,
//!   so a bitwise comparison over them would pin an implementation accident. Their decode stays
//!   covered by `kernel_v4_moe.rs`'s formula test.
//! * **What hipcc will emit tomorrow.** [`the_fp8_dot_sums_in_source_order_through_both_loops`]
//!   pins the compiler's CONTRACTION pattern as well as the source order; if a future hipcc
//!   contracts differently this goes red and the ISA gets re-read. That is the intended
//!   outcome, not a fragility to paper over.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: there is no `rocm` CI arm and this port had no GPU. One mutation each, with
//! the magnitude to expect:
//!
//! * In `kernels/linalg.hip::f2e4m3_rne`, change the subnormal tie from round-to-nearest-EVEN
//!   to half-away-from-zero (`+= 0.5` before the truncation instead of the even-bit test).
//!   [`act_quant_reaches_e4m3s_subnormal_ties_and_rounds_them_to_even`] must go RED naming an
//!   element in its first 16, and [`act_quant_f8_is_bit_identical_to_the_oracle`] must go red
//!   too — the second fixture reaches the same ties by construction. Nothing else moves. If the
//!   FIRST stays green the fixture has stopped reaching the subnormal band, which its own
//!   anti-vacuity arm (`sub >= 8`) asserts.
//! * In `kernels/common.hpp::fp8_dot_strided`, reorder the three fmas of the dword body
//!   (accumulate `i0+2` before `i0`). [`the_fp8_dot_sums_in_source_order_through_both_loops`]
//!   must go RED at some element — it is a BITWISE gate, so the expected magnitude is "at least
//!   one bit pattern differs", and `old:` records the sibling reassociation experiment moving
//!   the `v4res` fingerprint while staying inside every tolerance-based gate in the tree. That
//!   asymmetry is this test's whole reason to exist, so a green here under that mutation means
//!   the fold reference has drifted into agreement with whatever the kernel does.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli_backend::hip::{
    device_sync, launch_act_quant_f8, launch_act_quant_f8_prefix, launch_gemv_fp8_bf16,
};
use rivoli_core::num::{e4m3_to_f32, f32_to_e4m3};
use rivoli_oracles::v4oracle::forward::wave_ladder;
use rivoli_oracles::v4oracle::numerics::{
    FP8_MAX, act_quant_inplace, bf16_decode, bf16_encode, e4m3_decode, fast_round_scale,
};
use rivoli_oracles::v4oracle::weights::{NamedRng, fixed_bf16};

mod common;
use common::{
    DeviceBuf, assert_bits, assert_guard, assert_guards, back, byte_position_coverage, dev, f32b,
    f32v, stream, zeros,
};

/// `ACT_QUANT_BLOCK` — the block every quantized `Linear` runs its activation at, and the
/// modulus `act_quant_f8` refuses a ragged row against (`kernel.py:112`'s own
/// `assert N % block_size == 0`).
const ACT_BLOCK: usize = 128;

/// The KV entry's PARTIAL quantization block, which is a DIFFERENT number from [`ACT_BLOCK`]
/// and the one `act_quant_f8_prefix` is driven at below (`model.py:512`, dims
/// `[0, head_dim - rope_head_dim)` at block 64).
const KV_BLOCK: usize = 64;

/// Join the device and read a buffer back as f32.
///
/// `common::back` owns the join for the same reason it always has: a readback that skipped it
/// returns whatever was in the staging buffer, and the resulting comparison is against stale
/// data rather than against nothing — a green test on an unwritten result.
fn readback(b: &DeviceBuf) -> Vec<f32> {
    f32v(&back(b))
}

// =======================================================================================
// 1. the fp8 activation quantizer
// =======================================================================================

/// One 128-element `act_quant` block whose amax is EXACTLY 1.0, so the scale is pinned at
/// `fast_round_scale(1, 1/448) = 2^-8` and the tie values below land where they are meant to.
///
/// Contents, in order and each for a reason:
///   * `1.0`, which sets the amax and nothing else;
///   * every SUBNORMAL halfway tie, `k·2^-9·s` for `k` in `{0.5 … 7.5}` — the range where
///     rivoli's own `f32_to_e4m3` rounds half-AWAY-from-zero while `f2e4m3_rne` and the oracle
///     round half-to-EVEN. This is the only place the two rules differ, so a block without it
///     cannot tell them apart;
///   * the same eight negated, because RNE is sign-symmetric and half-away-from-zero is too —
///     the asymmetry to catch is in the tie DIRECTION, not the sign;
///   * NORMAL halfway ties `(1 + (m+0.5)/8)·2^e·s`, the other tie family;
///   * zeros, and a spread of ordinary magnitudes.
fn act_quant_block(seed: &str) -> Vec<f32> {
    const S: f32 = 1.0 / 256.0; // 2^-8, what fast_round_scale(1.0, 1/448) returns
    let mut v = vec![1.0f32];
    for k in 0..8 {
        let tie = (k as f32 + 0.5) * (1.0 / 512.0) * S; // the subnormal quantum is 2^-9
        v.push(tie);
        v.push(-tie);
    }
    for m in 0..8 {
        for e in -4i32..3 {
            v.push((1.0 + (m as f32 + 0.5) / 8.0) * (e as f32).exp2() * S);
        }
    }
    v.push(0.0);
    v.push(-0.0);
    let mut r = NamedRng::new(seed);
    while v.len() < ACT_BLOCK {
        // Scaled down so nothing displaces the 1.0 amax, and spread over binades so the normal
        // path runs at more than one exponent.
        v.push(r.unit() * 0.5f32.powi(r.below(12) as i32));
    }
    v.truncate(ACT_BLOCK);
    v
}

/// Two 128-element blocks covering EVERY finite e4m3 code, as `e4m3_decode(c) · 2^-8`.
///
/// Each block is 127 codes plus one pad, and the pad is what makes it 128 WIDE — which is what
/// `act_quant`'s blocking and the oracle's `chunks_mut(128)` both require. It does NOT pin the
/// scale and is not a new magnitude: it repeats `0x7e` (+448), a code the positive block already
/// holds, and the negative block's own extreme is `0xfe` (−448) of the same magnitude. So `amax`
/// is `448 · 2^-8 = 1.75` in both either way, and `fast_round_scale(1.75, 1/448)` is exactly
/// `2^-8` (1.75/448 IS `2^-8`, mantissa zero). Every element therefore divides back to the
/// code's own decoded value, which is representable by construction, so the block round-trips to
/// ITSELF and any disagreement with the oracle on ANY code shows up.
///
/// `0x7f` and `0xff` are the format's NaN; a NaN activation is fatal upstream, so it is not this
/// fixture's business.
fn e4m3_code_blocks() -> Vec<f32> {
    const S: f32 = 1.0 / 256.0;
    [0x00u8..=0x7e, 0x80..=0xfe]
        .into_iter()
        .flat_map(|codes| {
            codes
                .map(|c| e4m3_decode(c) * S)
                .chain([e4m3_decode(0x7e) * S])
        })
        .collect()
}

/// `act_quant_f8` against `v4oracle::numerics::act_quant_inplace`, BIT FOR BIT.
///
/// What it pins is the ROUND TRIP `e4m3_decode(e4m3_encode(x/s))·s`, which is what a V4 GEMM
/// consumes. A hypothetical encoder and decoder that were both shifted by the same amount would
/// cancel and pass; nothing else does.
#[test]
fn act_quant_f8_is_bit_identical_to_the_oracle() {
    const ROWS: usize = 6;
    let mut host: Vec<f32> = (0..ROWS)
        .flat_map(|r| act_quant_block(&format!("actq-{r}")))
        .collect();
    // A row of two blocks, to prove the tiling advances by 128 WITHIN a row rather than treating
    // each row as one block: the second block's amax differs from the first's, so a kernel that
    // reused one scale for the whole row produces different numbers.
    let wide: Vec<f32> = act_quant_block("actq-wide")
        .into_iter()
        .chain(act_quant_block("actq-wide2").into_iter().map(|v| v * 0.125))
        .collect();
    host.extend_from_slice(&wide);
    // Every finite e4m3 code, so the round trip is pinned over the whole format rather than over
    // whatever magnitudes the tie fixture happened to reach.
    let code_blocks = e4m3_code_blocks();
    host.extend_from_slice(&code_blocks);

    let mut want = host.clone();
    for row in want.chunks_mut(ACT_BLOCK) {
        act_quant_inplace(row, ACT_BLOCK, true);
    }

    let mut b = dev(&f32b(&host));
    let stream = stream();
    // The launch PLAN is the data, and the assertion sums the launch EXTENTS — not the fixture
    // lengths. That distinction is the whole guard: `rows + wide + codes == host.len()` reads
    // like a check and is a TAUTOLOGY, because `host` is the concatenation of exactly those three
    // pieces. It cannot fail, and it says nothing about what was dispatched.
    //
    // It has to be the launch arguments, because nothing downstream can help: the code blocks
    // round-trip to THEMSELVES, so an undispatched one is bit-identical to a correctly quantized
    // one in `got`. Halve the third extent or delete it and the suite goes green — which the
    // reference tree observed twice, under two earlier versions of this guard.
    //
    // Asserting BEFORE the `unsafe` block is also what makes its SAFETY claim true: an
    // over-covering extent would write past the allocation before any later check ran.
    let plan = [
        (ROWS, ACT_BLOCK),
        (1, wide.len()),
        (code_blocks.len() / ACT_BLOCK, ACT_BLOCK),
    ];
    let covered: usize = plan.iter().map(|(r, n)| r * n).sum();
    assert_eq!(
        covered,
        host.len(),
        "the launches do not cover the fixture exactly"
    );
    println!(
        "act_quant_f8: {} blocks dispatched, {} of them e4m3-code blocks",
        covered / ACT_BLOCK,
        plan[2].0
    );
    // SAFETY: `b` holds `host.len()` live f32, and the plan covers exactly that — asserted
    // immediately above, from the same extents the loop dispatches.
    unsafe {
        let p = b.ptr_mut() as *mut f32;
        let mut at = 0;
        for (rows, row_len) in plan {
            launch_act_quant_f8(p.add(at), rows, row_len, stream.raw()).expect("act_quant_f8");
            at += rows * row_len;
        }
    }
    // The quantizer must have MOVED something, or a kernel that wrote nothing at all would be
    // bit-identical to `want` wherever `want` happens to equal the input.
    assert_ne!(
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        host.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the quantizer left the input unchanged — nothing was compared"
    );
    assert_bits(&want, &readback(&b), "act_quant_f8 against the oracle");
}

/// `row_len` that is not a whole [`ACT_BLOCK`] is refused, matching `kernel.py:112`'s own
/// `assert N % block_size == 0`.
#[test]
fn act_quant_f8_refuses_a_ragged_row() {
    let mut b = zeros(300 * 4);
    let stream = stream();
    // SAFETY: the accepted case is 2 blocks of the 300-f32 buffer; each rejected one returns
    // before a dereference.
    let p = b.ptr_mut() as *mut f32;
    let go = |n_rows, row_len| unsafe { launch_act_quant_f8(p, n_rows, row_len, stream.raw()) };
    assert_guard(go(1, 2 * ACT_BLOCK), None, "a whole-block row");
    device_sync().expect("device sync"); // the accepted case LAUNCHED — join before `b` drops
    assert_guards([
        (1002, "row_len 1", go(1, 1)),
        (1002, "row_len 127", go(1, 127)),
        (1002, "row_len 129", go(1, 129)),
        (1002, "row_len 192", go(1, 192)),
        (1001, "zero rows", go(0, ACT_BLOCK)),
        (1001, "zero-length row", go(1, 0)),
    ]);
}

/// `act_quant_f8_prefix` against the oracle, on data CHOSEN to reach e4m3's subnormal range and
/// sit exactly on its rounding ties.
///
/// **The model fixture cannot cover this and no amount of it would.** `act_quant`'s
/// power-of-two scale puts a block's largest element in `[224, 448]`, so an element only reaches
/// e4m3's subnormals when it is ~2^15 below its block's peak — which drawn activations
/// essentially never are. That range is precisely where `f2e4m3_rne` and rivoli's own
/// `rivoli_core::num::f32_to_e4m3` disagree: the kernel rounds subnormal ties to nearest-EVEN
/// because V4 was trained against CUDA's `cvt.rn.satfinite.e4m3x2.f32`, while rivoli's rule for
/// GLM is half-away-from-zero.
///
/// So the block below pins the scale with a 448 and fills the rest with exact multiples and
/// exact HALF-multiples of the `2^-9` subnormal quantum, and the assertion before the comparison
/// proves that this data SEPARATES the two rules — without it, agreeing with the oracle here
/// would be evidence of nothing.
#[test]
fn act_quant_reaches_e4m3s_subnormal_ties_and_rounds_them_to_even() {
    const Q: f32 = 1.0 / 512.0; // e4m3's subnormal quantum, 2^-9
    let mut row = vec![0.0f32; KV_BLOCK];
    row[0] = FP8_MAX; // pins the block scale, and is itself the saturation edge
    for (i, v) in row[1..].iter_mut().enumerate() {
        // 0, 0.5, 1.0, ... 7.5 quanta — every representable subnormal AND every midpoint between
        // two of them, in both signs so a sign-dependent tie rule shows up.
        let m = (i % 16) as f32 * 0.5;
        *v = if i % 2 == 0 { m * Q } else { -m * Q };
    }

    let mut want = row.clone();
    act_quant_inplace(&mut want, KV_BLOCK, true);

    // ANTI-VACUITY, in two parts.
    // 1. The data must actually land in the subnormal band, or it tests the normal path twice.
    //    The band is `|x| < 2^-6 · s`.
    let amax = row.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-4);
    let s = fast_round_scale(amax, 1.0 / FP8_MAX);
    let sub = want
        .iter()
        .filter(|v| **v != 0.0 && v.abs() < s * 0.015_625)
        .count();
    assert!(
        sub >= 8,
        "only {sub} outputs are subnormal — this block does not reach the branch"
    );
    // 2. The data must SEPARATE the two rounding rules. `rivoli_core::num::f32_to_e4m3` is
    //    rivoli's half-away-from-zero encoder; if it produced the same block, then matching the
    //    oracle below would not be evidence that the kernel uses RNE.
    let half_away: Vec<f32> = row
        .iter()
        .map(|v| e4m3_to_f32(f32_to_e4m3((v / s).clamp(-FP8_MAX, FP8_MAX))) * s)
        .collect();
    assert_ne!(
        half_away, want,
        "half-away-from-zero and round-to-nearest-even agree on this block, so it cannot tell \
         which rule the kernel implements"
    );

    let mut buf = dev(&f32b(&row));
    let stream = stream();
    // SAFETY: `buf` is one row of KV_BLOCK f32, read and written in place, and outlives the join
    // inside `readback`.
    unsafe {
        launch_act_quant_f8_prefix(
            buf.ptr().cast(),
            buf.ptr_mut().cast(),
            1,
            KV_BLOCK,
            KV_BLOCK,
            KV_BLOCK,
            stream.raw(),
        )
    }
    .expect("act_quant_f8_prefix");
    // Bit-exact, not within a tolerance: `act_quant` is comparisons, a power-of-two scale and a
    // table lookup. There is no re-association in it to excuse a difference.
    assert_bits(&want, &readback(&buf), "act_quant_f8_prefix on the ties");
}

// =======================================================================================
// 2. the fp8 GEMV, and the fold it must not reassociate
// =======================================================================================

/// One output row's operands and geometry, shared by the fold transliterations below.
///
/// A struct rather than a seven-argument function, because jscpd correctly reads a
/// transliterated parameter list repeated twice as a clone — the `matvec_*` lists in
/// `artifact/quant.rs` carry an exemption for exactly this shape, and a struct is the fix that
/// needs none.
struct FoldRow<'a> {
    x: &'a [f32],
    wrow: &'a [u8],
    srow: &'a [f32],
    lut: &'a [f32],
    bsh: usize,
    n4: usize,
    k: usize,
}

impl FoldRow<'_> {
    /// One THREAD's strided share of the fp8 dot, in the kernel's emitted contraction
    /// (`common.hpp::fp8_dot_strided`: `q = x1·l1` rounded, then three fmas, then
    /// `acc = fma(s, q, acc)`; the scalar tail is `acc = fma(x·l, s, acc)`).
    ///
    /// `start`/`stride` are the kernel's own arguments: `(lane, 32)` under the wave-per-row
    /// dispatch. One definition, because a drift between two copies of this arithmetic would be
    /// indistinguishable from the kernel drift it exists to catch.
    fn chain(&self, start: usize, stride: usize) -> f32 {
        let (x, wrow, srow, lut) = (self.x, self.wrow, self.srow, self.lut);
        let mut acc = 0.0f32;
        for jj in (start..self.n4).step_by(stride) {
            let i0 = jj * 4;
            let mut q = x[i0 + 1] * lut[wrow[i0 + 1] as usize];
            q = x[i0].mul_add(lut[wrow[i0] as usize], q);
            q = x[i0 + 2].mul_add(lut[wrow[i0 + 2] as usize], q);
            q = x[i0 + 3].mul_add(lut[wrow[i0 + 3] as usize], q);
            acc = srow[i0 >> self.bsh].mul_add(q, acc);
        }
        for i in ((self.n4 * 4 + start)..self.k).step_by(stride) {
            acc = (x[i] * lut[wrow[i] as usize]).mul_add(srow[i >> self.bsh], acc);
        }
        acc
    }

    /// The row under `gemv_fp8_bf16`'s WAVE-PER-ROW fold: 32 lane chains at stride 32, one
    /// ladder (`v4oracle::forward::wave_ladder` — the shared definition). UNROUNDED; the caller
    /// applies the bf16 store where the kernel does.
    fn serial(&self) -> f32 {
        let mut lanes = [0.0f32; 32];
        for (l, acc) in lanes.iter_mut().enumerate() {
            *acc = self.chain(l, 32);
        }
        wave_ladder(lanes)
    }
}

/// One `gemv_fp8_bf16` shape: `m` rows of `k`, `n_out` outputs, at scale-tile `block`, groups 1.
///
/// A struct because four bare `usize` in a row are each plausible in another's position at the
/// launcher AND at [`serial_fold`], and a transposed pair moves BOTH sides identically — the
/// comparison still agrees and the gate is blind. `common::geometry`'s `Mla` makes this argument
/// about six.
#[derive(Clone, Copy)]
struct Gemv {
    m: usize,
    n_out: usize,
    k: usize,
    block: usize,
}

/// Upload `(x, w, scales)` and dispatch `launch_gemv_fp8_bf16` at `g`, groups = 1 — the device
/// harness both fold-order tests share. WHICH kernel runs is the launcher's shape dispatch.
fn gemv_fp8_on_device(x: &[f32], w: &[u8], scales: &[f32], g: Gemv) -> Vec<f32> {
    let (wd, sd, xd) = (dev(w), dev(&f32b(scales)), dev(&f32b(x)));
    let mut od = zeros(g.m * g.n_out * size_of::<f32>());
    let stream = stream();
    // SAFETY: `xd` is `m * k` f32, `wd` is `n_out * k` bytes, `sd` covers
    // `ceil(n_out/block) * ceil(k/block)` f32, `od` is `m * n_out` f32; `readback` joins the
    // device before any buffer drops.
    unsafe {
        launch_gemv_fp8_bf16(
            xd.ptr().cast(),
            wd.ptr().cast(),
            sd.ptr().cast(),
            g.m,
            g.n_out,
            g.k,
            g.block,
            1,
            od.ptr_mut().cast(),
            stream.raw(),
        )
    }
    .expect("gemv_fp8_bf16 dispatch");
    readback(&od)
}

/// The host reference for an `m = 1` wave-per-row fp8 launch: [`FoldRow::serial`] per output
/// row, bf16-stored where the kernel stores.
///
/// ONE definition for both fold-order tests, for the reason [`FoldRow::chain`] carries for
/// itself: a drift between two copies of this arithmetic is indistinguishable from the kernel
/// drift they exist to catch.
fn serial_fold(x: &[f32], w: &[u8], sc: &[f32], g: Gemv) -> Vec<f32> {
    let bsh = g.block.trailing_zeros() as usize;
    let sc_cols = g.k.div_ceil(g.block);
    let n4 = if g.block >= 4 { g.k >> 2 } else { 0 };
    let lut: Vec<f32> = (0..256).map(|b| e4m3_decode(b as u8)).collect();
    (0..g.n_out)
        .map(|j| {
            let row = FoldRow {
                x,
                wrow: &w[j * g.k..(j + 1) * g.k],
                srow: &sc[(j >> bsh) * sc_cols..],
                lut: &lut,
                bsh,
                n4,
                k: g.k,
            };
            bf16_decode(bf16_encode(row.serial()))
        })
        .collect()
}

/// Every e4m3 code except the two NaNs, in one table — the alphabet both weight sweeps draw
/// from.
///
/// Asserted rather than assumed to be exactly the NaN pair: an encoder change that moved which
/// codes are NaN would otherwise silently narrow or widen the sweep.
fn non_nan_codes() -> Vec<u8> {
    assert!(
        e4m3_decode(0x7f).is_nan() && e4m3_decode(0xff).is_nan(),
        "the two excluded codes must be exactly the NaN ones"
    );
    (0u8..=255).filter(|b| !matches!(b, 0x7f | 0xff)).collect()
}

/// `n` weight bytes walking the non-NaN alphabet so that each dword byte position sees all 254
/// of them, offset by `salt` so two tensors cannot mask each other at the seam.
fn sweep_bytes(n: usize, salt: usize) -> Vec<u8> {
    let allowed = non_nan_codes();
    (0..n)
        .map(|i| allowed[(i / 4 + (i % 4) * 67 + salt) % allowed.len()])
        .collect()
}

/// `gemv_fp8_bf16`'s summation ORDER, pinned bit-for-bit.
///
/// The fp8 twin of `kernel_v4_moe.rs`'s byte sweep cannot reuse its oracle: `Oracle::linear`
/// folds sequentially and the kernel wave-reduces, so that comparison rides a relative tolerance
/// and would wave through an unroll that split the accumulator chain — the exact failure this
/// gate exists to catch. So the reference here is a host transliteration of the KERNEL's own
/// fold. That pins two things a tolerance cannot: the unroll left the chain single and in
/// ascending-`j` order, and the compiler's contraction pattern did not drift.
///
/// `k = 1152` = 9 per-lane dword trips — one unrolled body plus one remainder pass, so the
/// unroll REMAINDER loop (unreachable at every engine dimension, since all real trip counts
/// divide 8) is exercised here. The `block = 2` dispatch routes the SAME bytes through the
/// scalar tail (`n4 = 0` below a quad-wide scale tile), covering the other loop entirely.
#[test]
fn the_fp8_dot_sums_in_source_order_through_both_loops() {
    const K: usize = 1152;
    const N_OUT: usize = 8;
    let w = sweep_bytes(N_OUT * K, 0);
    for p in 0..4 {
        let n = byte_position_coverage(&w, p);
        assert_eq!(n, 254, "dword byte position {p}: {n}/254 patterns");
    }
    let x = fixed_bf16("fp8-order-x", K, 0.05);
    // Sized for the block=2 dispatch's worst consumer: 4 row-blocks x 576 column tiles; block=128
    // reads row 0's first 9 entries of the same buffer. Powers of two only by habit — the host
    // model replays the identical arithmetic whatever the scale.
    let scales: Vec<f32> = (0..(N_OUT / 2) * (K / 2))
        .map(|i| [0.25f32, 0.5, 1.0, 2.0][i % 4])
        .collect();

    // `K = 1152 < 4096`, so both blocks stay on the wave-per-row kernel.
    for block in [ACT_BLOCK, 2] {
        let g = Gemv {
            m: 1,
            n_out: N_OUT,
            k: K,
            block,
        };
        assert_bits(
            &serial_fold(&x, &w, &scales, g),
            &gemv_fp8_on_device(&x, &w, &scales, g),
            &format!("fp8 dot order at block {block}"),
        );
    }
}

/// The `[wkv ‖ wq_a]` concat at the ENGINE's shapes — the per-row oracle coverage for the fused
/// `[1536 × 4096]` grid, seam included, in two bitwise claims:
///
/// 1. fused `out[0..512]` / `out[512..]` equal the two standalone launches the fusion replaces.
///    This is the load-time concat's layout contract executed by the real kernel: fused row
///    `512 + r` must read concatenated scale row `4 + r/128`. The scale cycle's period (3) is
///    COPRIME to the 32-entry scale rows, so every scale row's phase differs from its
///    neighbours' — a scale-grid concat shifted by one block row in EITHER direction changes
///    in-bounds values on every affected row and fails on thousands of terms. A period dividing
///    32 would make within-tensor neighbour rows identical and leave the shift visible only as
///    an out-of-bounds read, which is what a first cut of this fixture did.
/// 2. every fused row equals [`FoldRow::serial`] — the same source-order pin the `k = 1152`
///    sweep holds, at the fused shape's 32 whole per-lane trips (no remainder pass; that loop
///    keeps its coverage above).
#[test]
fn the_fused_qkv_gemv_is_bitwise_the_two_launches_it_replaces() {
    const K: usize = 4096;
    const N_KV: usize = 512;
    const N_QA: usize = 1024;
    let (w_kv, w_qa) = (sweep_bytes(N_KV * K, 0), sweep_bytes(N_QA * K, 131));
    // Power-of-two scales in a period-3 cycle (coprime to the 32-entry rows — see the doc
    // above), offset between the tensors: a seam or shift error lands rows on scales that differ
    // from their own, so equality would be impossible.
    let scl = |rows: usize, salt: usize| -> Vec<f32> {
        (0..rows.div_ceil(ACT_BLOCK) * K.div_ceil(ACT_BLOCK))
            .map(|i| [0.25f32, 0.5, 1.0][(i + salt) % 3])
            .collect()
    };
    let (s_kv, s_qa) = (scl(N_KV, 0), scl(N_QA, 1));
    let x = fixed_bf16("qkv-fuse-x", K, 0.05);
    let shape = |n_out| Gemv {
        m: 1,
        n_out,
        k: K,
        block: ACT_BLOCK,
    };
    let kv = gemv_fp8_on_device(&x, &w_kv, &s_kv, shape(N_KV));
    let qa = gemv_fp8_on_device(&x, &w_qa, &s_qa, shape(N_QA));
    let wf = [w_kv.as_slice(), w_qa.as_slice()].concat();
    let sf = [s_kv.as_slice(), s_qa.as_slice()].concat();
    let fused = gemv_fp8_on_device(&x, &wf, &sf, shape(N_KV + N_QA));
    assert_bits(&kv, &fused[..N_KV], "kv rows through the fused grid");
    assert_bits(&qa, &fused[N_KV..], "wq_a rows across the seam");
    assert_bits(
        &serial_fold(&x, &wf, &sf, shape(N_KV + N_QA)),
        &fused,
        "fused rows against the source-order fold",
    );
}

/// The C-ABI argument guards on both launchers, by CODE.
///
/// The code, not `is_err`: a check that accepted any error would still pass if someone replaced
/// a power-of-two test with `block != 128`, or if an unrelated dimension guard started
/// swallowing the case first.
#[test]
fn the_gemv_and_prefix_guards_reject_out_of_domain_shapes() {
    let mut b = zeros(64 * 4);
    let (p, pm) = (b.ptr().cast::<f32>(), b.ptr_mut().cast::<f32>());
    let nul = std::ptr::null_mut();
    // `null_mut()` for the stream throughout, and that is not laziness: every call here is
    // rejected by an argument guard BEFORE `hipLaunchKernelGGL`, so there is no launch for a
    // stream to order. A real stream would add a handle to each line and change nothing.
    //
    // SAFETY: every call below is rejected by an argument guard before any launch, so no pointer
    // is dereferenced and the shapes never have to be real.
    let cases = unsafe {
        [
            // A `groups` that does not divide `n_out` would index a slice no input was sized
            // for. This is the guard the three-parameter form could not express at all.
            (
                1004,
                "groups not dividing n_out",
                launch_gemv_fp8_bf16(p, b.ptr(), p, 1, 10, 128, 128, 3, pm, nul),
            ),
            (
                1003,
                "non-power-of-two block",
                launch_gemv_fp8_bf16(p, b.ptr(), p, 1, 8, 128, 96, 1, pm, nul),
            ),
            // The ONLY assertion of the ragged-span guard, deliberately. The reference tree also
            // asserted it inside its numerics test, holding a pre-renumbering code; the kernel
            // and the guard test moved together and that copy did not, so a run failed on a stale
            // string AFTER the numerics comparison had already passed. It cost two wrong
            // diagnoses, because a guard rejection reads like a numerics failure in a log. One
            // guard, one assertion.
            (
                1004,
                "a ragged quantization span",
                launch_act_quant_f8_prefix(pm, pm, 1, 64, 60, 64, nul),
            ),
            // A quant-FROM-SOURCE (`src != dst`) at partial width would leave dst's row tails
            // holding stale bytes where the copy it replaces filled them. `p`/`pm` alias one
            // buffer — deliberately, for every case above — so the distinct-pointer arm needs an
            // offset dst; `wrapping_add` because the guard rejects before any dereference, so the
            // address only has to differ.
            (
                1002,
                "a partial-width quant-from-source",
                launch_act_quant_f8_prefix(p, pm.wrapping_add(64), 1, 128, 64, 64, nul),
            ),
        ]
    };
    assert_guards(cases);
}
