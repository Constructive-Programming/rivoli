//! Muse Glimmer's two block-reduced norms — `rmsnorm_centered_single` and
//! `rmsnorm_weightless_batch` — each against a host oracle written beside it, plus the named
//! substitution each one is silently wrong under.
//!
//! **One file because they share one argument, not because they share a prefix.** Both reduce
//! `mean(x²)` across a 256-thread block and then multiply by something, and the whole of what
//! separates each from the kernel a reader reaches for first IS that multiplier:
//!
//! * `rmsnorm_centered_single` is `x·inv·(1 + w)` where `rmsnorm_single` is `x·inv·w`, and this
//!   model ships BOTH — four centered sandwich norms per layer against a plain final norm,
//!   embedding norm and qk_norm (`old:docs/reference/glimmer-architecture.md` §5).
//! * `rmsnorm_weightless_batch` has no weight at all and folds `qk_scale_factor` in: Q passes
//!   3.87 and K passes 1.0, so dropping the scale is one ARGUMENT and not a shape.
//!
//! **Not one of those substitutions announces itself.** None changes a dimension, none returns
//! an error, and none produces a non-finite value — the anchor's `norm_not_centered` run leaves
//! zero non-finite values across all 1103 captures and emits its seven tokens normally, which is
//! why §5's "crashes into garbage" is false and §9 trap 5's "runs clean and produces a wrong
//! model" is right (measured 2026-08-12, `old:tests/glimmer_norm.rs`). So every oracle here is
//! paired with a red proof that names the substitution, and the red proofs are what the file is
//! for.
//!
//! They share their guard family too — both refuse an `eps` that is not positive and finite
//! under code **1002** — and they share the failure mode of a strided accumulation over 256
//! threads, which only shows at a width that does not divide the block.
//!
//! # Ported from `old:tests/glimmer_norm.rs`, and what changed
//!
//! That file scores against the anchor's captured bytes as well as against a host oracle. The
//! goldens live in `crates/oracles/` in this tree and the census that owns this port scans
//! `crates/engine/tests`, so **what came here is the host-oracle half**: the arithmetic, the
//! guards, the aliasing claim, the ragged widths, and the three red proofs. The chain gates that
//! recover `1 + w` from reference bytes belong beside the goldens and are not ported.
//!
//! **The bars are the OWN-MEASUREMENT ones, not the anchor's rows.** Those rows —
//! `norm` at `Rel(7.70e-5)` and `qk_norm` at `Rel(7.85e-5)`, both measured before either kernel
//! existed — live in `crates/oracles/tests/common/tolerance.rs`, a test-local module of another
//! crate that nothing here can name. Nothing is lost by that: each is derived from its whole
//! bucket's fp32 floor across the reference's widths and is **~38x looser** than [`BAR`], which
//! `old:tests/glimmer_norm.rs` carried alongside them for exactly the reason that a bucket row
//! "cannot see this kernel regress by two decades".
//!
//! Bodies and their comments travelled with their arguments; where the scaffolding differs
//! (`common`'s `Lcg` for the old fixture's `fill`, `worst_rel` for the fixture's own copy of it)
//! the argument is restated in place and the change is named.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli_backend::hip::{
    device_sync, launch_rmsnorm_centered_single, launch_rmsnorm_weightless_batch,
};

mod common;
use common::{
    DeviceBuf, Got, Want, Weightless, assert_bits, assert_guard, back, dev, f32b, f32v, fill, ok,
    worst_rel, zeros,
};

/// Glimmer's hidden width — the width all four sandwich norms actually run at, and one no
/// anchor capture reaches (every one is emitted at hidden 72). At 6656 each of the 256 threads
/// strides 26 times and the tree reduction is fully loaded.
const HIDDEN: usize = 6656;

/// The QK-norm's shape: `head_dim` 128 over 32 Q heads and 2 KV heads. **`head_dim` is NOT
/// `hidden / n_heads`** — that is 208 — and `crates/artifact/tests/glimmer_config.rs` pins all
/// three against the shipped `config.json`.
const HEAD_DIM: usize = 128;
const HQ: usize = 32;
const HKV: usize = 2;

/// `qk_scale_factor`, applied to **Q alone** after the weightless norm (§7, §9 trap 3). Pinned
/// against the shipped config in `crates/artifact/tests/glimmer_config.rs`; K passes 1.0.
const QK_SCALE: f32 = 3.87;

/// The two norm epsilons, three orders of magnitude apart: `rms_norm_eps` on the two pre-norms,
/// `post_norm_eps` on the two post-norms. Both pinned against the shipped config beside the
/// dimensions above.
const EPS_PRE: f32 = 1e-5;
const EPS_POST: f32 = 1e-8;

/// The activation scale the four norms actually see, and it is **not** unit scale.
///
/// A norm's eps substitution moves the output by `≈ 0.5·(eps_wrong − eps_right)/mean(x²)`, so it
/// lives or dies on where the activations stand. Two reviews recovered the reference's own
/// post-norm inputs from the goldens and measured **mean(x²) = 1.139e-3 … 6.55e-3**, 50–300x
/// smaller than unit scale. `Lcg::f` draws uniform in [-1, 1), so `r.f() * 0.11` has
/// `mean(x²) = 0.11²/3` = **4.03e-3**, inside that band.
///
/// An earlier draft of the ported file drew at unit scale, where the same substitution moves the
/// output by 0.19x the tolerance it was priced on — the fixture was scoring the kernel in the one
/// regime where the defect the bar exists for is INVISIBLE. This constant is that correction.
const X_SCALE: f32 = 0.11;

/// The centered weight's scale, from the anchor's own draw rather than from an initialisation
/// fact: recovering `1 + w` from the goldens found the tiny model's centered weights spanning
/// [-0.197, +0.198], i.e. the driver's `uniform_(-0.2, 0.2)`. `w` is INITIALISED to zeros for a
/// centered norm, and confusing that with a trained checkpoint's weights is how the first draft
/// of `old:tests/glimmer_norm.rs` came to score at 0.02.
const W_SCALE: f32 = 0.2;

/// The bar every host-oracle comparison in this file is held to.
///
/// **MEASURED 2026-08-12 on gfx1151 at 1.174e-7 (eps 1e-5) and 1.172e-7 (eps 1e-8)** at the real
/// width, and 1.762e-7 over the ragged widths — so this is ~11-17x. It is deliberately not
/// snugger: a correct kernel's analytic worst case here is ~1.15e-6 (26 strided adds per thread
/// plus an 8-level ladder at 2^-24), and a bar under that rejects correct code.
const BAR: f32 = 2.0e-6;

// The fixture draw this file spelled as its own `fill(n, salt, scale)` is now
// `common::fill`, HOISTED 2026-08-16 with M8's V4 oracles — which is what the debt note here
// asked for by name ("hoist it and delete all three the next time `common/` is open"). The
// salted form's argument travelled with the body and is at `common::fill`: `x` and `w` must be
// INDEPENDENT, and the tests below reuse one of the pair across cases, which a shared cursor
// makes unspellable. Nothing about the stream changed — `Lcg(salt)` then `n` draws, scaled — so
// every measurement in this file is against the same bytes.

/// The weightless norm's four values for a QK-norm of `rows` heads at Glimmer's `head_dim`.
///
/// A constructor and not a literal per call site: rustfmt's `struct_lit_width` turns every
/// `Weightless { .. }` wider than 18 characters into one line per field, and three of those
/// differing only in `rows` are three identical four-line runs — which `build.rs`'s duplication
/// gate reports, correctly. Callers that need a different `eps` or `scale` say
/// `Weightless { scale: 1.0, ..qk(rows) }`, which keeps the change visible in one place.
fn qk(rows: usize) -> Weightless {
    Weightless {
        rows,
        d: HEAD_DIM,
        eps: EPS_PRE,
        scale: QK_SCALE,
    }
}

/// `mean(x²)` in f64 — the quantity that decides whether an eps substitution is visible at all,
/// and therefore printed by every test here that claims a defect has power.
fn mean_sq(x: &[f32]) -> f64 {
    x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64
}

/// `rsqrt(mean(x²) + eps)` on the HOST, in f64, and one spelling so the two host oracles below
/// cannot disagree about the reduction.
///
/// **f64 deliberately, and it is not `common::rms_inv`.** That helper is the shared f32 formula;
/// the point of these oracles is to score the KERNEL's reduction, and a host sum that
/// re-associates in f32 alongside it would hide an error in both. The kernel's own ladder is a
/// strided accumulation into an LDS tree, so an `n - 1` divisor leaves a `sqrt(71/72)` = 6.97e-3
/// residual against this — which is exactly what the ported file's reverted red proof measured.
fn inv_rms(x: &[f32], eps: f32) -> f64 {
    1.0 / (mean_sq(x) + eps as f64).sqrt()
}

/// A centered norm's two operands.
///
/// `x` and `w` are the same length and the same type, so a swapped pair COMPILES and computes
/// `w·rms(w)·(1 + x)` — which is a norm, of the wrong tensor, at the wrong scale. Every function
/// below takes both, and this is the one place their order is spelled.
#[derive(Clone, Copy)]
struct NormPair<'a> {
    x: &'a [f32],
    w: &'a [f32],
}

/// One weightless-norm case: the activation and the geometry it is normed under.
///
/// Bundled because the device runner and the host oracle must be handed the SAME `Weightless`, and
/// two calls taking `(&x, g)` separately are two places for one of them to drift onto a different
/// `rows`/`d`/`eps`/`scale` while the comparison stays green about it.
#[derive(Clone, Copy)]
struct Heads<'a> {
    x: &'a [f32],
    g: Weightless,
}

/// `y[i] = x[i]·inv·(1 + w[i])`, or `x[i]·inv·w[i]` with `centered` false.
///
/// The bool is what makes the form red proof possible without a second device kernel:
/// `rmsnorm_single` is not this diff's launcher and is gated in its own file, and taking it here
/// would score one kernel's defect through another kernel's arithmetic.
fn host_centered(p: NormPair<'_>, eps: f32, centered: bool) -> Vec<f32> {
    let inv = inv_rms(p.x, eps);
    p.x.iter()
        .zip(p.w)
        .map(|(v, k)| {
            let scale = if centered { 1.0 + *k as f64 } else { *k as f64 };
            (*v as f64 * inv * scale) as f32
        })
        .collect()
}

/// The weightless form over `g.rows` segments of `g.d`, times `g.scale`, in f64.
///
/// Takes [`Weightless`] rather than four trailing scalars for that type's own reason — `(rows, d)`
/// and `(eps, scale)` are each interchangeable to the type checker, and a transposed pair would
/// move this oracle and the launch below TOGETHER, leaving the comparison agreeing. It is the
/// shape bundle and not the arithmetic that is shared: `Weightless::apply` is f32, and the
/// paragraph on [`inv_rms`] is why this is not it.
fn host_weightless(h: Heads<'_>) -> Vec<f32> {
    let g = h.g;
    let mut out = h.x.to_vec();
    for r in 0..g.rows {
        let seg = &mut out[r * g.d..(r + 1) * g.d];
        let f = g.scale as f64 * inv_rms(seg, g.eps);
        seg.iter_mut().for_each(|v| *v = (*v as f64 * f) as f32);
    }
    out
}

/// The three device addresses one `rmsnorm_centered_single` launch takes, in launcher order.
///
/// Bundled and spelled ONCE, on the argument `kernel_attend.rs`'s `MlaIo` makes about its own:
/// the oracle, the aliasing test and the guard table all drive this launcher, and three copies of
/// a six-argument list is three chances for `x`, `w` and `y` to stop being the same three buffers
/// — which no comparison in this file could notice, since each checks its own launch.
/// `build.rs`'s duplication gate reported exactly that, on the argument lists rustfmt had spread
/// to one per line.
#[derive(Clone, Copy)]
struct CenteredIo {
    x: *const f32,
    w: *const f32,
    y: *mut f32,
}

impl CenteredIo {
    /// The three addresses with `y` DISTINCT from `x` — the non-aliased launch every comparison
    /// here scores. The aliased form passes one pointer as both and has its own test.
    fn new(x: &DeviceBuf, w: &DeviceBuf, y: &mut DeviceBuf) -> Self {
        Self {
            x: x.ptr() as *const f32,
            w: w.ptr() as *const f32,
            y: y.ptr_mut() as *mut f32,
        }
    }
}

/// One `rmsnorm_centered_single` launch, returning the launcher's own `Result` so the guard table
/// can read a CODE while the oracles demand success.
///
/// # Safety
/// `io.x` and `io.w` are each `n` live readable f32 and `io.y` `n` live writable f32, all live
/// until the next [`device_sync`]. `io.y` MAY alias `io.x` — that is the kernel's stated in-place
/// contract and `the_centered_norm_may_write_into_its_own_input` is what holds it. A rejected
/// call returns before any launch and dereferences nothing.
unsafe fn centered_launch(io: CenteredIo, n: usize, eps: f32) -> anyhow::Result<()> {
    // SAFETY: the caller's contract above. Null stream: every call site here launches once and
    // then joins, so there is nothing to order against.
    unsafe { launch_rmsnorm_centered_single(io.x, io.w, n, eps, io.y, std::ptr::null_mut()) }
}

/// One `rmsnorm_weightless_batch` launch. Same argument as [`centered_launch`]: the oracle and
/// the guard table both drive it, and [`Weightless`] already carries all four scalars in one
/// order, so the list is spelled once.
///
/// # Safety
/// `x` is `g.rows * g.d` live f32, read and written IN PLACE, live until the next [`device_sync`].
/// A rejected call returns before any launch and dereferences nothing.
unsafe fn weightless_launch(x: *mut f32, g: Weightless) -> anyhow::Result<()> {
    // SAFETY: the caller's contract above; null stream for `centered_launch`'s reason.
    unsafe { launch_rmsnorm_weightless_batch(x, g.rows, g.d, g.eps, g.scale, std::ptr::null_mut()) }
}

/// `rmsnorm_centered_single` into a distinct destination.
fn centered(p: NormPair<'_>, eps: f32) -> Vec<f32> {
    let (xb, wb) = (dev(&f32b(p.x)), dev(&f32b(p.w)));
    let mut yb = zeros(p.x.len() * 4);
    let io = CenteredIo::new(&xb, &wb, &mut yb);
    // SAFETY: `xb` and `wb` each hold exactly `x.len()` live readable f32, `yb` that many writable
    // ones in a distinct allocation, and all three outlive the join inside `back`.
    ok(
        unsafe { centered_launch(io, p.x.len(), eps) },
        "centered norm",
    );
    f32v(&back(&yb))
}

/// `rmsnorm_weightless_batch`, which is IN PLACE and **destroys `x`** — the returned vector is
/// the only copy of the post-norm values, and the pre-norm ones are gone from the device.
fn weightless(h: Heads<'_>) -> Vec<f32> {
    let mut xb = dev(&f32b(h.x));
    let p = xb.ptr_mut() as *mut f32;
    // SAFETY: `xb` holds exactly `g.rows * g.d` live f32 and outlives the join inside `back`.
    ok(unsafe { weightless_launch(p, h.g) }, "weightless qk-norm");
    f32v(&back(&xb))
}

/// One comparison in the metric every Glimmer tolerance is stated in, asserted against `bar` and
/// returned so a caller can fold the worst over its cases.
///
/// `worst_rel` and not `assert_rel`, and the difference is the whole non-vacuity of this file:
/// `assert_rel` folds `f32::max` over the differences, and **`f32::max` returns the other operand
/// on a NaN** — so an all-NaN kernel output scores 0.0, a perfect match. A broken kernel in this
/// repo passed 9 of 9 comparisons that way. `worst_rel` returns INFINITY for a non-finite `got`
/// and PANICS on a non-finite `want`, which is the other half: a corrupt reference is a diagnosis
/// of the wrong side.
///
/// It takes [`Got`] and [`Want`] rather than two bare slices for that pair's own reason: both
/// sides are `&[f32]`, so a swap compiles, and every bound here is stated against `max_abs(want)`
/// — swapped, the bar scales by the KERNEL's own output and the gate grades itself. Wrapped, the
/// swap is an `E0308`.
fn score(got: Got<'_>, want: Want<'_>, bar: f32, label: &str) -> f32 {
    let r = worst_rel(got, want);
    assert!(r <= bar, "{label}: worst rel {r:e} > {bar:e}");
    r
}

/// One centered-norm case at width `n`: draw `x` and `w` under `salts`, run the device against the
/// f64 host oracle, and return the score. Both host-oracle tests below are this loop body, which
/// is why it is a function.
fn centered_case(n: usize, eps: f32, salts: (u64, u64)) -> f32 {
    let (x, w) = (fill(n, salts.0, X_SCALE), fill(n, salts.1, W_SCALE));
    let p = NormPair { x: &x, w: &w };
    let (got, want) = (centered(p, eps), host_centered(p, eps, true));
    score(
        Got(&got),
        Want(&want),
        BAR,
        &format!("centered norm at width {n}, eps {eps:e}"),
    )
}

// ---- rmsnorm_centered_single ---------------------------------------------------------------

/// The kernel against the host oracle at Glimmer's real width, at both epsilons.
///
/// `w` is drawn near zero, as the real checkpoint ships it: `w` is initialised to ZEROS for a
/// centered norm, so `1 + w` is near 1 and the norm is near the identity. Scoring at a large `w`
/// would be scoring a regime the model never occupies.
#[test]
fn the_centered_norm_matches_a_host_oracle_at_the_real_width() {
    let mut worst = 0.0f32;
    for (label, eps) in [("rms_norm_eps", EPS_PRE), ("post_norm_eps", EPS_POST)] {
        let r = centered_case(HIDDEN, eps, (1, 2));
        worst = worst.max(r);
        println!("centered norm at {label} ({eps:e}), width {HIDDEN}: worst rel {r:e}");
    }
    println!("centered norm at hidden {HIDDEN}: worst rel {worst:e} against bar {BAR:e}");
}

/// **A width where the strided loop is RAGGED**, which neither the anchor's 72 nor 6656 is.
///
/// 6656 is exactly 26 x 256, so every thread runs the accumulation the same 26 times; 72 is under
/// one block, so 72 threads run once and 184 never. The regime where some threads accumulate `k`
/// terms and others `k+1` — where an off-by-one in the loop bound or a partial-tail LDS write
/// shows up — is untested at both, and the ported file's coverage claim was wrong about that
/// until review measured it.
///
/// Nothing in Glimmer reaches these widths today. This is here because the claim was made, and so
/// that the first caller at a non-multiple width finds the regime already gated.
#[test]
fn the_centered_norm_holds_at_widths_that_do_not_divide_the_block() {
    let mut worst = 0.0f32;
    // 257 and 6657: one element past a block boundary, so exactly one thread carries an extra
    // term — the narrowest version of the imbalance. 1000: 232 threads at 4 terms and 24 at 3.
    // 6655: one element short, so one thread carries one FEWER.
    for n in [257usize, 1000, 6655, 6657] {
        for eps in [EPS_PRE, EPS_POST] {
            worst = worst.max(centered_case(n, eps, (10, 11)));
        }
    }
    println!("ragged widths 257/1000/6655/6657 at both eps: worst rel {worst:e}");
}

/// **The form defect, run in both directions.** This is §9 trap 5 and the reason
/// `rmsnorm_centered_single` is a second entry point rather than a `bool` on the first.
///
/// `worst_rel` cannot RANK the two directions — scaled-by-≈0 and scaled-by-≈2 both differ from
/// the reference by ~1x its own magnitude — so this asserts only that each is far over the bar.
/// Ranking them needs a magnitude comparison and is deliberately not claimed.
#[test]
fn each_norm_form_is_caught_in_the_others_place() {
    // Four decades over `BAR`, and the measured signals are ~1 in this metric — the bar is loose
    // against them on purpose, because what is claimed is "the fixture can tell the forms apart"
    // and not a value for how far apart they are.
    const LOUD: f32 = 1.0e-1;
    let x = fill(HIDDEN, 3, X_SCALE);
    let w = fill(HIDDEN, 4, W_SCALE);

    // The plain form on a CENTERED weight, on the host: `rmsnorm_single` is not this file's
    // kernel and is gated in its own, and reaching for it here is what forced the ported file's
    // fixture wrapper to carry a form bool in the first place.
    let p = NormPair { x: &x, w: &w };
    let plain = host_centered(p, EPS_PRE, false);
    let collapse = worst_rel(Got(&plain), Want(&centered(p, EPS_PRE)));
    println!("plain form on a centered weight: {collapse:e} against bar {LOUD:e}");
    assert!(
        collapse > LOUD,
        "the plain form on a centered weight moved the output by only {collapse:e} — this fixture \
         cannot tell the two forms apart"
    );

    // The other direction: a PLAIN weight (ones, as the final and embedding norms ship it) run
    // through the centered kernel scales by ≈2.
    let ones = vec![1.0f32; HIDDEN];
    let flat = NormPair { x: &x, w: &ones };
    let through = centered(flat, EPS_PRE);
    let doubled = worst_rel(Got(&through), Want(&host_centered(flat, EPS_PRE, false)));
    println!("centered form on a plain weight: {doubled:e}");
    assert!(
        doubled > LOUD,
        "a plain weight through the centered kernel moved the output by only {doubled:e}"
    );
}

/// **The row's own weakest defect, reproduced.** `post_norm_eps_shared` — `rms_norm_eps` where
/// `post_norm_eps` belongs — is the substitution the anchor's `norm` tolerance was priced on, and
/// a bar derived from a defect is only meaningful if the defect exceeds it HERE.
///
/// It is not a red proof of the KERNEL: both epsilons are legal config values, so no kernel is
/// wrong for accepting either. It is a proof that this fixture, at [`X_SCALE`], can reject the
/// thing the tolerance was measured from — at unit activations it cannot, which is why the
/// constant exists.
///
/// **MEASURED 2026-08-12 at 1.239e-3 at mean(x²) 4.026e-3.** The pair printed below is ONE row's
/// two numbers, so the formula `≈ 0.5·Δeps/mean(x²)` relates them; a max-of-each-quantity pairing
/// does not, and that is how the ported file first stated it.
#[test]
fn the_two_epsilons_are_separable_at_the_scale_the_norms_run_at() {
    // 50x `BAR`. The measured 1.239e-3 clears it by 12x, and a floor this far under the
    // measurement is what keeps the claim ("the epsilons are distinguishable here") from turning
    // into a second, unargued tolerance on the kernel.
    const FLOOR: f32 = 50.0 * BAR;
    let x = fill(HIDDEN, 6, X_SCALE);
    let w = fill(HIDDEN, 7, W_SCALE);
    let p = NormPair { x: &x, w: &w };
    let post = centered(p, EPS_POST);
    let d = worst_rel(Got(&centered(p, EPS_PRE)), Want(&post));
    let m = mean_sq(&x);
    println!("eps substitution at mean(x2) {m:.3e}: {d:e} against floor {FLOOR:e}");
    assert!(
        d > FLOOR,
        "the eps substitution moves the output by only {d:e} at mean(x2) {m:.3e} — the reference's \
         post-norms stand at 1.14e-3..6.55e-3, so a fixture standing elsewhere cannot price it"
    );
}

/// The kernel may write into its own input, **bit for bit**.
///
/// Not a hypothetical convenience: the layer loop applies four norms per layer to a 6656-wide
/// residual stream, and `post_attention_layernorm`'s output feeds `o_proj` with nothing else
/// reading its input — so an in-place norm is one buffer the loop does not have to hold.
///
/// **All three operands are `__restrict__`, and passing one pointer twice is what that qualifier
/// disclaims** — three reviews raised it and none resolved it. What resolves it is the kernel's
/// own two-clause argument: loop 1 reads OTHER threads' indices, so the `__syncthreads()` inside
/// `block_sum_lds` is what separates every read of `x` from every write of `y`, and loop 2's
/// thread `i` then reads and writes only its own index. `swiglu` is launched in place in
/// production on the second clause alone, so this is the tree's established position and not a
/// new one.
///
/// Scored bit-against-bit with the non-aliased launch — same kernel, same grid, same inputs, so
/// any difference at all IS the aliasing. That is strictly stronger than [`BAR`], which the first
/// version of the ported test used and which had 651x of slack against the measurement.
#[test]
fn the_centered_norm_may_write_into_its_own_input() {
    let x = fill(HIDDEN, 8, X_SCALE);
    let w = fill(HIDDEN, 9, W_SCALE);
    // BOTH epsilons: two of the four sandwich norms run under `post_norm_eps`, three decades from
    // the other, and the first version of this test covered only `EPS_PRE`.
    for (label, eps) in [("rms_norm_eps", EPS_PRE), ("post_norm_eps", EPS_POST)] {
        let (mut xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
        let p = xb.ptr_mut() as *mut f32;
        // The aliased io: `x` and `y` are ONE pointer, deliberately.
        let io = CenteredIo {
            x: p as *const f32,
            w: wb.ptr() as *const f32,
            y: p,
        };
        // SAFETY: `xb` is HIDDEN live f32, readable and writable, and `wb` is HIDDEN live readable
        // f32; both outlive the join inside `back`. The aliasing is the kernel's stated contract
        // and is argued above.
        ok(unsafe { centered_launch(io, HIDDEN, eps) }, "aliased norm");
        let apart = centered(NormPair { x: &x, w: &w }, eps);
        assert_bits(&apart, &f32v(&back(&xb)), label);
    }
}

/// The eps guard, driven — and the eps is the parameter whose defect set this operator's row.
///
/// The guard cannot catch `post_norm_eps_shared`, because both values are legal; what it catches
/// is the config never having been read at all. Every clause is exercised, because an unexercised
/// guard is how the sibling `gqa_attend`'s ring bound sat wrong until review found it:
/// `!(eps > 0.0f)` is spelled that way to reject NaN as well as non-positive, and `eps < INFINITY`
/// is the other half — rewritten as `eps <= 0.0f`, the NaN row goes red on its own.
#[test]
fn the_centered_launcher_refuses_an_eps_that_cannot_come_from_a_config() {
    let (xb, wb) = (dev(&f32b(&fill(8, 5, 1.0))), dev(&f32b(&[0.0f32; 8])));
    let mut yb = zeros(8 * 4);
    let io = CenteredIo::new(&xb, &wb, &mut yb);
    for (n, eps, want, what) in [
        (0usize, EPS_PRE, Some(1001), "an empty row"),
        (8, 0.0, Some(1002), "a zero eps"),
        (8, -1e-5, Some(1002), "a negative eps"),
        (8, f32::NAN, Some(1002), "a NaN eps"),
        (8, f32::INFINITY, Some(1002), "an infinite eps"),
        (8, EPS_POST, None, "the real post_norm_eps, which must pass"),
    ] {
        // SAFETY: the rejected calls return before any launch; the accepted one writes 8 live f32
        // into `yb` from 8 live f32 in each of `xb` and `wb`, all outliving the sync below.
        assert_guard(unsafe { centered_launch(io, n, eps) }, want, what);
    }
    device_sync().expect("sync the accepted centered-norm dispatch");
}

// ---- rmsnorm_weightless_batch --------------------------------------------------------------

/// The QK-norm against a host oracle at both scales the model uses, and at two ragged head widths.
///
/// **Q and K are the same operator with different `scale`, which is trap 3.** Q passes
/// `qk_scale_factor` and K passes 1.0, and the scale is folded into the norm's reciprocal here
/// (`x·(s·rs)`) where the reference applies it to the norm's output (`(x·rs)·s`) — one
/// association apart, ~6e-8, four decades under [`BAR`].
///
/// `d = 100` puts 100 of 256 threads on one term and 156 on none, which is the under-one-block
/// regime; `d = 257` puts exactly one thread on a second term. Neither is a width Glimmer reaches
/// — `head_dim` is 128 — and both are here for the reason the centered norm's ragged widths are.
#[test]
fn the_weightless_norm_matches_a_host_oracle_at_both_qk_scales() {
    let mut worst = 0.0f32;
    for (label, g) in [
        ("q: 32 heads x 128 at x3.87", qk(HQ)),
        (
            "k: 2 heads x 128 at x1.0",
            Weightless {
                scale: 1.0,
                ..qk(HKV)
            },
        ),
        ("ragged d=100", Weightless { d: 100, ..qk(4) }),
        ("ragged d=257", Weightless { d: 257, ..qk(3) }),
    ] {
        for eps in [EPS_PRE, EPS_POST] {
            let g = Weightless { eps, ..g };
            let x = fill(g.rows * g.d, 12, X_SCALE);
            let h = Heads { x: &x, g };
            let (got, want) = (weightless(h), host_weightless(h));
            let what = format!("{label} at eps {eps:e}");
            worst = worst.max(score(Got(&got), Want(&want), BAR, &what));
        }
    }
    println!("weightless qk-norm over four shapes at both eps: worst rel {worst:e}");
}

/// **`x` is DESTROYED, and that is made checkable rather than left in a doc comment.**
///
/// The launcher's safety note constrains what may WRITE `x` before the launch; this is the other
/// clause, added by review 2026-08-13 — no consumer of the PRE-norm values may be enqueued after
/// it. The realistic violation is a `--trace` or `--pred-probe` readback of q expecting
/// `q_proj`'s output and getting post-norm, post-3.87 bytes, and no fixture can see THAT. What a
/// fixture can see is that the pre-norm values are gone, so it says so here: the buffer the
/// kernel was handed comes back as the norm's output and not as its input.
///
/// The sibling `rmsnorm_centered_single` twenty lines above it in the kernel file is out of
/// place, so a caller migrating between the two silently converts a read-preserving operator into
/// a destructive one with no shape change in the signature to notice.
#[test]
fn the_weightless_norm_destroys_its_input() {
    let g = qk(HKV);
    let x = fill(g.rows * g.d, 13, X_SCALE);
    let h = Heads { x: &x, g };
    let got = weightless(h);
    // The buffer holds the OUTPUT — asserted against the oracle, so "it changed" cannot be
    // satisfied by garbage — and the pre-norm values are not recoverable from it.
    let want = host_weightless(h);
    score(Got(&got), Want(&want), BAR, "in-place readback");
    let survived = got.iter().zip(&x).filter(|(a, b)| a == b).count();
    assert_eq!(
        survived,
        0,
        "{survived} of {} pre-norm values survived the launch, so this kernel is not the \
         in-place operator its callers are told it is",
        x.len()
    );
}

/// **Trap 3, run: dropping `qk_scale_factor` from Q.**
///
/// The scale is Q's alone, it is one argument rather than a shape, and passing K's 1.0 where
/// 3.87 belongs changes nothing a dimension check could see. Scored against the correctly scaled
/// run, so what is measured is the substitution and not the kernel's agreement with anything.
#[test]
fn dropping_the_qk_scale_factor_is_caught() {
    let base = qk(HQ);
    let x = fill(base.rows * base.d, 14, X_SCALE);
    let flat = Weightless { scale: 1.0, ..base };
    let right = weightless(Heads { x: &x, g: base });
    let d = worst_rel(Got(&weightless(Heads { x: &x, g: flat })), Want(&right));
    // `1 - 1/3.87` = 0.742 analytically: the two runs differ by exactly that factor and the metric
    // divides by the larger side's magnitude. A bar of 0.5 sits under that with room and is still
    // five decades over `BAR`.
    println!("qk_scale_factor dropped from Q: {d:e} (analytically 1 - 1/{QK_SCALE} = 0.742)");
    assert!(
        d > 0.5,
        "dropping the 3.87 moved Q by only {d:e}, so this fixture cannot see trap 3"
    );
}

/// The three guards, driven. `scale` earns one for `eps`'s reason: it comes from a config field,
/// and a zero or non-finite one silently multiplies every Q head by nothing or by garbage.
///
/// **They reject the unusable, not the implausible** — `eps` = 1e30 and `scale` = 1e-45 both
/// pass, and a numeric range that rejected them would be the guard restating the config. Each
/// clause gets its own CODE (1001 shape, 1002 eps, 1003 scale) and the code is what is asserted:
/// a test satisfied by any error would still pass if one guard started swallowing another's case.
#[test]
fn the_weightless_launcher_refuses_an_eps_or_a_scale_it_cannot_use() {
    let mut xb = dev(&f32b(&fill(8, 15, 1.0)));
    let p = xb.ptr_mut() as *mut f32;
    for (rows, d, eps, scale, want, what) in [
        (0usize, 8usize, EPS_PRE, QK_SCALE, Some(1001), "zero rows"),
        (1, 0, EPS_PRE, QK_SCALE, Some(1001), "a zero-width head"),
        (1, 8, 0.0, QK_SCALE, Some(1002), "a zero eps"),
        (1, 8, f32::NAN, QK_SCALE, Some(1002), "a NaN eps"),
        (1, 8, f32::INFINITY, QK_SCALE, Some(1002), "an infinite eps"),
        (1, 8, EPS_PRE, 0.0, Some(1003), "a zero qk scale"),
        (1, 8, EPS_PRE, -1.0, Some(1003), "a negative qk scale"),
        (1, 8, EPS_PRE, f32::NAN, Some(1003), "a NaN qk scale"),
        (
            1,
            8,
            EPS_PRE,
            f32::INFINITY,
            Some(1003),
            "an infinite qk scale",
        ),
        (
            1,
            8,
            EPS_POST,
            1.0,
            None,
            "K's own arguments, which must pass",
        ),
    ] {
        let g = Weightless {
            rows,
            d,
            eps,
            scale,
        };
        // SAFETY: the rejected calls return before any launch; the accepted one norms one row of
        // 8 live f32 in place, inside a buffer of exactly that size, outliving the sync below.
        assert_guard(unsafe { weightless_launch(p, g) }, want, what);
    }
    device_sync().expect("sync the accepted weightless dispatch");
}
