//! **Qwen3.8-Flash-Next's Gated Residual read stage, scored against the S2 anchor** —
//! `gated_residual_collapse_f32` and nothing else.
//!
//! Every sublayer of all 48 layers is wrapped by one of these, pre and post (MOD:1204-1205,
//! 1222-1243), and GR **replaces** pre-normalisation — the checkpoint has no `input_layernorm`
//! and no `post_attention_layernorm` tensor at all, and no `model.norm` either. So this operator
//! runs 96 times per token and any error in it compounds down the whole stack, which is why the
//! MEAN-versus-sum row below matters more than its 3.000e0 suggests.
//!
//! # The two weightless triples this is scored on, and the four that are not available
//!
//! The anchor captures `GatedResidual`'s module outputs, not its parameters, so of the six
//! quantities in TR Eq. 30-34 exactly two boundaries are closed:
//!
//! * `hc_norm` (= `rhat`) and `input_mix_weight_up` (= the pre-sigmoid gate) into `.0` (= `x`) —
//!   **weightless**, because both inputs are already the outputs of the two projections;
//! * `block_inject_weight` into `.2` (= the inject scale) — **weightless** for the same reason.
//!
//! What is NOT scorable here, stated so nobody looks for it: the group RMSNorm that produces
//! `rhat` (its `[n_r · d]` weight is per-element, so recovering it from `.1` is exactly determined
//! and any offset convention cancels — a gate with zero degrees of freedom is decoration), and
//! the two low-rank matmuls through the `hc_lowrank = 320` bottleneck, whose weights are absent.
//! Those are `rmsnorm_centered_single` and a GEMV, both already covered, and the SHAPES are
//! asserted from CKPT at S4 rather than here (down `[320, 10240]`, up `[10240, 320]` — trap T11's
//! transposed reading type-checks).
//!
//! # 28 sites, and why that number
//!
//! Seven captured layers times two wrappers (`attn_hyper_connection`, `mlp_hyper_connection`)
//! times two draws. Both wrappers are scored rather than one: they are two INDEPENDENT modules
//! with their own weights, and a fixture that drove only the attention one would leave the MLP
//! wrapper — the one that runs after the MoE — entirely unexercised.
//!
//! # RED PROOF, OBSERVED 2026-09-01 — deviceless, on the host oracle
//!
//! Class: **hc transpose + lowrank order** (the matrix's `hc_*` rows). [`inject_scale`]'s
//! REFERENCE arm was moved to `2 · sigma(w)`, dropping Eq. 33's `1/n_r`; `cmp` returns 1 and the
//! run recompiled.
//!
//! * [`the_host_gated_residual_reproduces_the_anchor_everywhere`] **RED** —
//!   `qwen-anchor-1 L0 attn_hyper_connection scale: 3.285509e-1 is outside the 1.42e-4 operator
//!   tolerance`. The `x` assertion sits ABOVE it in the same body and stayed green, which is what
//!   makes this a reached assertion rather than a masked one.
//! * [`the_inject_scale_divides_inside_the_sigmoid_and_doubles_outside_it`] stayed **GREEN**, the
//!   same blindness `kernel_qwen_gdn_recurrent.rs` records: a variant is scored against the
//!   GOLDEN, never against the reference arm.
//! * Reverted, `cmp` rc 0, recompiled, 7 passed exit 0.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod kernel_qwen_harness;

use kernel_qwen_harness as h;

// See `kernel_qwen_gdn_recurrent.rs` for the `duplicate_mod` argument.
#[cfg(feature = "rocm")]
#[allow(clippy::duplicate_mod)]
mod common;

/// Every layer the decode goldens capture. All seven carry both wrappers — GR is per SUBLAYER, so
/// it is present on the GDN layers and the QSA layers alike, which is the one structural fact this
/// list is here to state.
const LAYERS: [usize; 7] = [0, 1, 2, 3, 27, 46, 47];

/// The worst the f64 host oracle shows against the anchor over all 28 sites and both outputs:
/// 1.1979e-7 on the collapse, 8.8760e-8 on the inject scale. Floors 4.4444e-8 and 3.0545e-8, so
/// the spread is under 3x.
const HOST_WORST: f32 = 1.1979e-7;

/// Defect separations, minima over the 28 sites, measured 2026-09-01.
///
/// Two of them are EXACT at every site and that is a property of the arithmetic, not luck:
/// `SUM_NOT_MEAN` is `|n_r·x − x| / |x| = n_r − 1 = 3` because the two differ by a pure factor,
/// and `INJECT_WITHOUT_THE_TWO` is `0.5` because so do those. A constant that is identical at 28
/// sites is a stronger statement than a range: it says the variant is a scale, so no draw can
/// hide it and no draw can flatter it either.
const SUM_NOT_MEAN: f32 = 3.0000e0;
const GATE_UNSIGMOIDED: f32 = 9.5203e-1;
const GATE_SILU: f32 = 9.7561e-1;
const INJECT_WITHOUT_THE_DIVISION: f32 = 2.2883e-1;
const INJECT_WITHOUT_THE_TWO: f32 = 5.0000e-1;
const INJECT_DIVIDED_OUTSIDE: f32 = 6.4668e-1;

/// One GR wrapper's read stage, at the widths the launcher takes.
struct Gr {
    /// `hc_count` — the residual stream's width, 4 in this model.
    n_r: usize,
    /// `hidden_size` — ONE branch's width, so `rhat` is `n_r · dim` and `hc_width` is `n_r · dim`.
    dim: usize,
    /// `[n_r][dim]` — the group-RMSNormed stream.
    rhat: Vec<f32>,
    /// `[n_r][dim]` — `input_mix_weight_up`'s output, PRE-sigmoid.
    mix: Vec<f32>,
    /// `[n_r]` — `block_inject_weight`'s output, PRE-sigmoid and PRE-division.
    inject: Vec<f32>,
    want_x: Vec<f32>,
    want_scale: Vec<f32>,
}

/// Which arithmetic the oracle performs. The variants ARE the defects.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    Reference,
    /// `hc_sum_not_mean` (T10): the branches summed. Exactly `n_r` too large, every layer,
    /// compounding over 48 of them.
    SumNotMean,
    /// The gate taken as the projection produced it.
    GateUnsigmoided,
    /// A SiLU gate — the other plausible bounded-ish choice, and `hidden_act` IS `silu`, which is
    /// what makes it plausible.
    GateSilu,
    /// `2·sigma(inject)`: Eq. 33's `1/n_r` dropped.
    InjectWithoutTheDivision,
    /// `sigma(inject/n_r)`: Eq. 33's leading 2 dropped, so the scale centres on 0.5.
    InjectWithoutTheTwo,
    /// `2·sigma(inject)/n_r`: the division moved OUTSIDE the sigmoid, which is the other way
    /// Eq. 33 reads.
    InjectDividedOutside,
}

/// Score one variant of one output at every site: the fixture's own bar first, then the recorded
/// weakest separation.
///
/// Factored because the two tests below are the same loop over different rows — jscpd matched them
/// as soon as the second existed. The two checks it forwards to [`h::separations`] — the fixture's
/// own bar per site, then the recorded weakest — live there for the same reason, one level up: all
/// three qwen suites were writing them, and the ORDER of the two is what they were drifting on.
fn each_variant(
    rows: &[(Form, &str, f32)],
    of: impl Fn(&Gr, Form) -> Vec<f32>,
    want: impl Fn(&Gr) -> &[f32],
) {
    for (form, name, floor) in rows {
        let mut sites = Vec::new();
        for_each_wrapper(|at, c| sites.push((at.to_string(), h::rel(&of(c, *form), want(c)))));
        h::separations(h::Bar::at("hc_attn", HOST_WORST), name, *floor, &sites);
    }
}

/// The collapse, Eq. 32.
fn collapse(c: &Gr, form: Form) -> Vec<f32> {
    (0..c.dim)
        .map(|j| {
            let s: f64 = (0..c.n_r)
                .map(|i| {
                    let (r, m) = (
                        f64::from(c.rhat[i * c.dim + j]),
                        f64::from(c.mix[i * c.dim + j]),
                    );
                    let g = match form {
                        Form::GateUnsigmoided => m,
                        Form::GateSilu => m * h::sigmoid(m),
                        _ => h::sigmoid(m),
                    };
                    g * r
                })
                .sum();
            let mean = if form == Form::SumNotMean {
                s
            } else {
                s / c.n_r as f64
            };
            mean as f32
        })
        .collect()
}

/// The inject scale, Eq. 33.
fn inject_scale(c: &Gr, form: Form) -> Vec<f32> {
    let n = c.n_r as f64;
    c.inject
        .iter()
        .map(|w| {
            let w = f64::from(*w);
            let s = match form {
                Form::InjectWithoutTheDivision => 2.0 * h::sigmoid(w),
                Form::InjectWithoutTheTwo => h::sigmoid(w / n),
                Form::InjectDividedOutside => 2.0 * h::sigmoid(w) / n,
                _ => 2.0 * h::sigmoid(w / n),
            };
            s as f32
        })
        .collect()
}

/// Every (draw, layer, wrapper) triple with its read stage assembled.
fn for_each_wrapper(mut f: impl FnMut(&str, &Gr)) {
    for d in h::decode_draws() {
        let n_r = d.c("hc_count");
        let dim = d.w("hidden_size");
        assert_eq!(
            n_r * dim,
            d.w("hc_width"),
            "the recorded hc_width must be n_r branches of hidden_size — 4 x 2560 = 10240 in the \
             real config, where it COLLIDES with the GDN conv_dim and the tiny config breaks the \
             collision on purpose (the anchor's `collide_in_real` list)"
        );
        for layer in LAYERS {
            for wrapper in ["attn_hyper_connection", "mlp_hyper_connection"] {
                let p = format!("model.layers.{layer}.{wrapper}");
                let c = Gr {
                    n_r,
                    dim,
                    rhat: d.flat(&format!("{p}.hc_norm"), n_r * dim),
                    mix: d.flat(&format!("{p}.input_mix_weight_up"), n_r * dim),
                    inject: d.flat(&format!("{p}.block_inject_weight"), n_r),
                    want_x: d.flat(&format!("{p}.0"), dim),
                    want_scale: d.flat(&format!("{p}.2"), n_r),
                };
                f(&format!("{} L{layer} {wrapper}", d.salt), &c);
            }
        }
    }
}

// ── deviceless ──────────────────────────────────────────────────────────────────────────────

/// The f64 host oracle reproduces both outputs at every one of the 28 sites.
#[test]
fn the_host_gated_residual_reproduces_the_anchor_everywhere() {
    let b = h::Bar::at("hc_attn", HOST_WORST);
    let mut sites = 0;
    for_each_wrapper(|at, c| {
        b.hold(at, "x", &collapse(c, Form::Reference), &c.want_x);
        b.hold(
            at,
            "scale",
            &inject_scale(c, Form::Reference),
            &c.want_scale,
        );
        sites += 1;
    });
    assert_eq!(
        sites, 28,
        "two draws times seven captured layers times two wrappers — a smaller count means a \
         wrapper or a layer silently stopped being covered"
    );
}

/// **The inject scale is `2·sigma(inject / n_r)`**, and none of the three other readings of
/// Eq. 33 reproduces it.
///
/// This is the one place in this port where the tech report's own equation admits more than one
/// reading and the reference's capture is what settles it. All three alternatives are plausible
/// enough that a port would pick one: `1/n_r` outside the sigmoid, no `1/n_r` at all, and no
/// leading 2.
#[test]
fn the_inject_scale_divides_inside_the_sigmoid_and_doubles_outside_it() {
    each_variant(
        &[
            (
                Form::InjectWithoutTheDivision,
                "1/n_r dropped",
                INJECT_WITHOUT_THE_DIVISION,
            ),
            (
                Form::InjectWithoutTheTwo,
                "the leading 2 dropped",
                INJECT_WITHOUT_THE_TWO,
            ),
            (
                Form::InjectDividedOutside,
                "1/n_r outside the sigmoid",
                INJECT_DIVIDED_OUTSIDE,
            ),
        ],
        inject_scale,
        |c| &c.want_scale,
    );
}

/// The collapse is a MEAN with a sigmoid gate, and each of the three variants moves it clear.
#[test]
fn the_collapse_is_a_sigmoid_gated_mean() {
    each_variant(
        &[
            (Form::SumNotMean, "hc_sum_not_mean", SUM_NOT_MEAN),
            (
                Form::GateUnsigmoided,
                "gate left unsigmoided",
                GATE_UNSIGMOIDED,
            ),
            (Form::GateSilu, "a SiLU gate", GATE_SILU),
        ],
        collapse,
        |c| &c.want_x,
    );
    // `SumNotMean` is EXACTLY `n_r - 1` at every site because the two forms differ by a pure
    // factor. Asserted as an equality rather than a bound: if it ever stops being exact, the
    // collapse has stopped being a pure mean and the whole reading of Eq. 32 is wrong, which a
    // one-sided bound would let through.
    for_each_wrapper(|at, c| {
        let moved = h::rel(&collapse(c, Form::SumNotMean), &c.want_x);
        let want = (c.n_r - 1) as f32;
        assert!(
            (moved - want).abs() < 1e-5,
            "{at}: summing moved the collapse by {moved:e}, not the exact {want} a pure factor of \
             n_r gives — Eq. 32 is then not a plain mean over the branches"
        );
    });
}

// ── on the device ───────────────────────────────────────────────────────────────────────────

/// The kernel reproduces both outputs at every one of the 28 sites, from ONE launch.
///
/// # RED-PROOF PLAN — for the integrator's first device run
///
/// Mutations go in `kernels/residual.hip`'s `gated_residual_collapse_f32`; each must redden this
/// test while [`the_host_gated_residual_reproduces_the_anchor_everywhere`] stays green. The
/// magnitudes are the constants above, all shown reachable deviceless:
///
/// * drop the `/ (float)n_r` on `x` — exactly 3.0 at every site;
/// * use `rhat[at] * mix[at]` — ~9.5e-1;
/// * gate with `mix * sigmoid(mix)` — ~9.8e-1;
/// * write `2.0f / (1.0f + expf(-inject[t]))` — ~2.3e-1 (this one reddens `scale` only, so read
///   WHICH assertion fails: the `x` arm above it stays green and would mask nothing, but a plant
///   read off the exit code alone could be mistaken for the collapse having moved);
/// * write `1.0f / (1.0f + expf(-inject[t] / n_r))` — exactly 0.5 at every site.
#[cfg(feature = "rocm")]
#[test]
fn the_gated_residual_kernel_matches_the_anchor_everywhere() {
    use common::{DeviceBuf, back, dev, f32b, f32v, ok, stream, zeros};
    use rivoli_backend::hip::launch_gated_residual_collapse_f32;

    let b = h::Bar::at("hc_attn", HOST_WORST);
    let s = stream();
    let cp = |d: &DeviceBuf| d.ptr() as *const f32;
    for_each_wrapper(|at, c| {
        let rhat = dev(&f32b(&c.rhat));
        let mix = dev(&f32b(&c.mix));
        let inject = dev(&f32b(&c.inject));
        let mut x = zeros(c.dim * 4);
        let mut scale = zeros(c.n_r * 4);
        let x_out = x.ptr_mut() as *mut f32;
        let scale_out = scale.ptr_mut() as *mut f32;
        // SAFETY: `rhat` and `mix` hold `n_r · dim` f32 each, `inject` holds `n_r`, and the two
        // destinations are distinct allocations of `dim` and `n_r` — five separate buffers, so the
        // kernel's all-`__restrict__` contract holds, and all live until `back` joins.
        ok(
            unsafe {
                launch_gated_residual_collapse_f32(
                    cp(&rhat),
                    cp(&mix),
                    cp(&inject),
                    c.n_r,
                    c.dim,
                    x_out,
                    scale_out,
                    s.raw(),
                )
            },
            "gated_residual_collapse_f32",
        );
        b.hold(at, "x", &f32v(&back(&x)), &c.want_x);
        b.hold(at, "scale", &f32v(&back(&scale)), &c.want_scale);
    });
}

/// The launcher refuses the two argument shapes that mean the caller passed the wrong config
/// field.
#[cfg(feature = "rocm")]
#[test]
fn the_collapse_launcher_refuses_a_stream_width_it_cannot_mean() {
    use common::{assert_guard, assert_guards, dev, f32b, zeros};
    use rivoli_backend::hip::launch_gated_residual_collapse_f32;

    let src = dev(&f32b(&[0.5f32; 512]));
    let mut x = zeros(512 * 4);
    let mut scale = zeros(512 * 4);
    let (x_out, scale_out) = (x.ptr_mut() as *mut f32, scale.ptr_mut() as *mut f32);
    let p = src.ptr() as *const f32;
    let call = |n_r: usize, dim: usize| {
        // SAFETY: the buffers cover every legal case below; the illegal ones are refused before a
        // pointer is read, which is what is under test.
        unsafe {
            launch_gated_residual_collapse_f32(
                p,
                p,
                p,
                n_r,
                dim,
                x_out,
                scale_out,
                std::ptr::null_mut(),
            )
        }
    };
    // Through `assert_guards` rather than four straight-line calls: it is the house API for a
    // guard TABLE, and jscpd matched the straight-line tail against `kernel_qwen_gdn_recurrent.rs`
    // — which was the same finding the V4 suites had, and the reason that helper exists.
    // 320 is `hc_lowrank` and 10240 is `hc_width`; both are one config field away from `hc_count`
    // and both are larger, which is the mistake 1003 exists to catch rather than define.
    assert_guards([
        (1001, "a zero-wide residual stream", call(0, 8)),
        (1001, "a zero-wide branch", call(4, 0)),
        (1003, "hc_lowrank passed as hc_count", call(320, 8)),
    ]);
    assert_guard(call(4, 8), None, "the legal shape must launch");
}
