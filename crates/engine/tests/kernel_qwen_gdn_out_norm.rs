//! **Qwen3.8-Flash-Next's GDN output norm-gate is `rmsnorm_gate_heads_f32`, scored** — the REUSE
//! the guard used to refuse, and the evidence that padding the reduction did not move it.
//!
//! `out = o · rsqrt(mean(o²) + eps) · weight · σ(z)`, per head. That is Kimi-K3's fused gated head
//! norm and it is qwen's `RMSNormGated` **exactly**: bare `w ⊙ x̂` with no `(1 + w)` offset, the
//! activation wrapping `z` and not the normed stream, the eps inside the root and on the MEAN, in
//! that order (`qwen-architecture.md` §1, MOD:192-201; `output_gate_type: "sigmoid"` at CFG:193-195
//! and TR p.4's "we use the bounded sigmoid gate"). So there is no new launcher and no census row —
//! what is new is that the reuse can now be scored at all.
//!
//! # Why this suite exists at all: a guard refused the only fixture that could score the reuse
//!
//! `rivoli_rmsnorm_gate_heads_f32`'s guard **1003** refused every non-power-of-two `head_dim`,
//! because `block_sum_lds` drops elements there. Qwen's S2 anchor runs `linear_value_head_dim`
//! **20** (128 in the real model), so the operator was arithmetically right, structurally reusable,
//! and **unscorable** — which is not a reuse, it is a hope. The kernel now launches
//! `next_pow2(head_dim)` and masks the lanes past `head_dim`; 1003 is gone from that launcher and
//! stays on `gated_delta_recurrent_f32`, whose `[head][channel]` `dt_bias` would need the same
//! masking on a second buffer with no fixture asking for it.
//!
//! **The bit-identity claim for the four shipped models is structural, and it is stated at the
//! kernel in four parts** — unchanged block, universally-true predicate, unchanged divisor,
//! unchanged fold. Two of those four are what this suite holds:
//!
//! * the divisor. `d` used to be read off `blockDim.x`; under padding that is the padded width, so
//!   `head_dim` became an explicit kernel parameter. [`Form::MeanOverThePaddedWidth`] is that
//!   mistake, and it separates by only **5.7190e-3** at its weakest — three orders under the
//!   eps rows, which is exactly why it needs a row of its own rather than trusting that "a wrong
//!   divisor would obviously show";
//! * the masked lanes. `off` for a masked lane addresses the NEXT head's row, so an off-by-one mask
//!   folds a neighbour into this head's mean. [`Form::NeighbourFoldedIn`] is that shape at
//!   **5.5533e-4**, the weakest row in the table and 7.5x this fixture's own bar.
//!
//! The two remaining parts — the block size and the fold — are properties of the launch and of
//! `block_sum_lds`, and the one thing no structural argument covers is what the compiler emits for
//! the select the mask inserts. So the **before/after byte comparison on `kernel_k3_conv_norm.rs`
//! is OWED to the next GPU lease** and is on track C's arm list, and the masked lanes get a
//! standing device test here scored as BYTES rather than against an envelope.
//!
//! # The recovered weight, and the one thing it cannot see
//!
//! No weight tensor is in the golden. The `[head_dim]` weight is recovered from head **0** under the
//! bare `w ⊙ x̂` form and used to predict heads **1..8** — 1 row in, 8 out, at four GDN layers and
//! both draws, so 64 predicted rows rest on 8 recovered ones. Head 0 is excluded from every score
//! because it is tautological by construction.
//!
//! **It is blind to the offset convention, and that is arithmetic rather than an oversight.**
//! Recovering `w` under `w ⊙ x̂` and predicting with `(1 + w)` cancels exactly, so no draw and no
//! layer can separate the two here. `gdn_norm_unit_offset` is track B's weakest targeting defect for
//! this operator at 9.7290e-1 and it is settled on REAL BYTES elsewhere:
//! `crates/artifact/tests/qwen_names.rs::the_gdn_output_norm_weight_is_ones_centred`, over the 256
//! vendored bytes of `linear_attn_norm_l0.bin`, whose discriminant is `min > 0.5` — a margin no
//! narrowing closes. This suite cites that gate instead of pretending to be it.
//!
//! # RED PROOFS, OBSERVED 2026-09-01 — deviceless, and both are the PADDING CHANGE's own defects
//!
//! Neither plant is in the kernel, because the device is held; each is the host-side shape of a
//! kernel-side risk the change introduces, scored by the assertion that would catch it. The
//! kernel-side plants are on this suite's arm list. Both were observed to change the tree (`cmp`
//! rc 1 against a saved copy) and to redden a named assertion, on runs that recompiled.
//!
//! * **the off-by-one mask.** `Form::Reference` made to fold in the neighbour ->
//!   [`the_host_norm_gate_predicts_every_other_head`] **RED**,
//!   `qwen-anchor-1 L0 norm: 1.3131158e-2 is outside the 1.65e-4 operator tolerance`.
//! * **the divisor left reading the padded width** — point 3 of the kernel's bit-identity note ->
//!   the same assertion **RED** at `4.0394567e-2`.
//! * Reverted after each, `cmp` rc 0, 8 passed exit 0.
//!
//! **The third confirmation of a finding this port keeps making:**
//! [`every_defect_form_prices_the_difference_it_names`] stayed **GREEN** under BOTH plants, as it
//! did under `kernel_qwen_gdn_recurrent.rs`'s and `kernel_qwen_hyper_residual.rs`'s. A separation
//! is scored variant-against-GOLDEN and never reads the reference arm, so a defect table built
//! only from separations reports green on a broken oracle. Three instances is a property, not a
//! coincidence; `h::separations`' own doc now carries it.
//!
//! # DEVICE ARM, 2026-09-01 — rc 0, witness EMPTY, 12 passed, and the padding claim held
//!
//! **The BEFORE/AFTER byte comparison is DONE and the prediction HELD.** Two binaries — one built
//! from `recurrent.hip` at 43bb83a, one from HEAD, both built outside the lock, both objects and
//! executables verified distinct by md5 — were run back to back with **no rebuild between the
//! arms**, both witnesses empty, both rc 0.
//! [`the_power_of_two_widths_reproduce_the_host_oracle_and_print_their_bytes`]'s eight `BITS` rows
//! (4 heads x head_dim 32 and 128) are **byte-identical**: `cmp` rc 0 over 5,956 bytes of hex, both
//! sides md5 `a6b1b8a570a8712a6b48414fc28770cb`. The four-part construction argument is now a
//! measurement at the widths the four shipped models run.
//!
//! **Kernel-side red proof P4**, `t < head_dim` -> `t <= head_dim`, is the one that paid twice:
//!
//! * [`the_norm_gate_launcher_matches_the_anchor_at_the_width_the_guard_refused`] **RED** at
//!   `qwen-anchor-1 L0 norm: 7.752174e-1`;
//! * [`the_masked_lanes_cannot_reach_the_result`] **RED** at *"the NaN-poisoned run produced a
//!   non-finite output, so the two runs agreeing byte-for-byte would prove nothing — the masked
//!   lanes ARE reaching the result"*. **The finiteness guard fired before the byte comparison
//!   could**, which is the NaN trap this repo has on record catching a broken kernel that scored
//!   9 of 9; here it caught one on silicon.
//!
//! > **The host variant UNDER-models the kernel plant, and the numbers say so.**
//! > [`NEIGHBOUR_FOLDED_IN`] predicts 5.5533e-4 and the kernel plant measured **7.752174e-1**,
//! > three orders larger. The host variant folds a neighbour into the mean; the real off-by-one
//! > ALSO lets lane `head_dim` past `if (!live) return;`, so it STORES to `out[off]` and clobbers
//! > the next head's first output. A host stand-in for a kernel defect is a lower bound on it, not
//! > a model of it — which is an argument for the device arm, not against the stand-in.
//!
//! # RED PROOF, 2026-09-01 — a COMPILE failure, which is the point
//!
//! `kernel_k3_conv_norm.rs`'s transposed-pair row is gone and the case is unrepresentable instead.
//! Swapping this suite's own pair (tree changed, `cmp` rc 1):
//!
//! ```text
//! error[E0308]: arguments to this function are incorrect
//!    --> crates/engine/tests/kernel_qwen_gdn_out_norm.rs:591:13
//! 595 |                 HeadDim(head_dim),
//!     |                 ----------------- expected `HeadCount`, found `HeadDim`
//! 596 |                 HeadCount(heads),
//!     |                 ---------------- expected `HeadDim`, found `HeadCount`
//! ```
//!
//! Reverted, `cmp` rc 0, the tree builds. **A compile error is a stronger proof than the runtime
//! guard it replaces AND a weaker kind of evidence**, and both halves are worth saying: it fires
//! for every caller rather than only the arm that thinks to try the case, and it can never be
//! observed going red on hardware, because the code that would fail does not exist to run. That is
//! the trade the coordinator called — kill the class rather than restore the guard.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod kernel_qwen_harness;

use kernel_qwen_harness as h;

// `duplicate_mod`: the harness reaches `common/scoring.rs` by `#[path]` because `common/mod.rs` does
// not compile deviceless; under `rocm` this loads that same one file again through the umbrella.
// `kernel_qwen_gdn_recurrent.rs` carries the full argument.
#[cfg(feature = "rocm")]
#[allow(clippy::duplicate_mod)]
mod common;

/// The GDN layers among the anchor's captured seven — the same list `kernel_qwen_gdn_recurrent.rs`
/// scores, because this operator is the stage immediately after the recurrence in the same block.
const GDN_LAYERS: [usize; 4] = [0, 1, 2, 46];

/// The worst the f64 host norm-gate shows against the anchor over both draws, all four GDN layers
/// and the eight PREDICTED heads: **2.4683e-7**, floor 8.2585e-8.
const HOST_WORST: f32 = 2.4683e-7;

/// Defect separations, minima over the eight sites, measured 2026-09-01.
///
/// **The two smallest are the two that price the padding change**, and they are three orders under
/// the eps rows — which is the reason they are tabled at all. `mean_over_padded_width` at 5.7190e-3
/// is 77x this fixture's bar and `neighbour_folded_in` at 5.5533e-4 is 7.5x it; both are well under
/// the `gdn_out_norm` tolerance's own 1.65e-4... which is the point worth reading twice:
/// `neighbour_folded_in` is only **3.4x** the operator tolerance, so the bucket-level bound would
/// nearly miss it and the fixture's own bar is what catches it.
const MEAN_OVER_THE_PADDED_WIDTH: f32 = 5.7190e-3;
/// > **A HOST STAND-IN IS A LOWER BOUND ON A KERNEL DEFECT, NOT A MODEL OF IT.** This constant
/// > predicts 5.5533e-4 for an off-by-one mask; the real kernel plant measured **7.752174e-1** on
/// > 2026-09-01, three orders larger. The variant folds a neighbour into the mean, and that is all
/// > it can do from the host — the real off-by-one ALSO lets lane `head_dim` past
/// > `if (!live) return;`, so it STORES to `out[off]` and clobbers the next head's first output. A
/// > host variant reaches only the arithmetic the oracle models; a kernel defect reaches whatever
/// > the kernel does. Use these numbers as floors on what a plant must clear, never as predictions
/// > of what it will give, and do not "reconcile" a device red that comes in larger.
const NEIGHBOUR_FOLDED_IN: f32 = 5.5533e-4;
/// The eps rows. Huge here — 1.5e0 and up — because `o` is small at these widths, so `mean(o²)` is
/// of order the 1e-6 itself and the epsilon is not a perturbation but a term. The contrast with the
/// indexer's block norm is the useful part: THERE the same three forms sit in a 2.8e-5..4.6e-4 band
/// and are unpriceable, HERE they are the strongest rows in the table. An eps is weakly or strongly
/// constrained by the SCALE of what it is added to, never by which config field named it.
const EPS_ON_THE_SQRT: f32 = 1.5580e0;
const EPS_FROM_THE_WRONG_FIELD: f32 = 6.1318e-1;
const EPS_DROPPED: f32 = 1.5677e0;
const GATE_ON_THE_NORMED_STREAM: f32 = 2.4123e-1;
const GATE_SILU: f32 = 1.0367e0;
const GATE_TANH: f32 = 1.2201e0;
const NORM_OVER_THE_SUM: f32 = 1.9906e-1;

/// One GDN layer's output norm-gate, at the widths the launcher takes.
struct GateNorm {
    heads: usize,
    /// `linear_value_head_dim` — **20** at the anchor's widths, which is the number this whole
    /// suite exists because of, and 128 in the real model.
    head_dim: usize,
    /// `[heads][head_dim]` — the recurrence's own output, the norm's input.
    o: Vec<f32>,
    /// `[heads][head_dim]` — `in_proj_z`, the gate PRE-sigmoid.
    z: Vec<f32>,
    /// `[head_dim]`, recovered from head 0. See the header for what it can and cannot say.
    weight: Vec<f32>,
    want: Vec<f32>,
}

impl GateNorm {
    /// The predicted rows only — head 0 is tautological under the recovery and scoring it would
    /// report a perfect match over an equation the fixture itself solved.
    fn predicted(&self, v: &[f32]) -> Vec<f32> {
        v[self.head_dim..].to_vec()
    }
}

/// Which arithmetic the oracle performs. The variants ARE the defects.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    Reference,
    /// `eps_outside_sqrt` (T23): `x / (sqrt(m) + eps)` instead of `x · rsqrt(m + eps)`.
    EpsOnTheSqrt,
    /// `eps_from_the_wrong_config_field` (T23): `rms_norm_eps` would be right here by coincidence —
    /// the tiny config carries 1e-6 for both — so the variant uses 1e-5 to make the row non-vacuous
    /// and the coincidence is stated rather than relied on.
    EpsFromTheWrongField,
    /// The eps dropped entirely.
    EpsDropped,
    /// `gdn_output_gate_on_normed` (T22): the sigmoid wrapping the NORMED stream instead of `z`.
    /// `RMSNormGated` takes both as arguments and which one the activation wraps is one identifier
    /// apart (MOD:199).
    GateOnTheNormedStream,
    /// `output_gate_silu` (T5): fla's own default, and `RMSNormGated`'s `activation` parameter
    /// defaults to it.
    GateSilu,
    /// `output_gate_tanh` (T22): the OTHER bounded gate, which is what TR p.4's "bounded sigmoid
    /// gate" invites.
    GateTanh,
    /// The norm taken over the SUM of squares rather than their mean.
    NormOverTheSum,
    /// **The padding change's own defect.** The mean divided by `next_pow2(head_dim)` — what the
    /// kernel would compute if `d` had been left reading `blockDim.x`. Inert at every power-of-two
    /// width, silent at every other one.
    MeanOverThePaddedWidth,
    /// **The masked lanes' own defect.** One element of the NEXT head's row folded into this head's
    /// sum of squares — the shape an off-by-one mask has, because `off` for a masked lane addresses
    /// real memory holding real numbers.
    NeighbourFoldedIn,
}

/// `next_pow2` — the launcher's rounding, restated here because the variant above has to compute
/// the same padded width the kernel would. A second spelling of a C loop is a thing to say out
/// loud: it is not a gate on the C side, and it cannot be, because a test that imported the rule
/// from the code under test would be checking the rule against itself.
fn padded(n: usize) -> usize {
    let mut b = 1;
    while b < n {
        b <<= 1;
    }
    b
}

fn gate_norm(c: &GateNorm, form: Form) -> Vec<f32> {
    let d = c.head_dim;
    let eps = match form {
        Form::EpsFromTheWrongField => 1e-5,
        Form::EpsDropped => 0.0,
        _ => 1e-6,
    };
    let mut out = vec![0.0f32; c.heads * d];
    for hd in 0..c.heads {
        let row = &c.o[hd * d..(hd + 1) * d];
        let mut sq: f64 = row.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
        if form == Form::NeighbourFoldedIn {
            // Head `hd + 1`'s element 0, exactly where `off = blockIdx.x · head_dim + head_dim`
            // lands. The last head wraps, which is the one place a real off-by-one reads PAST the
            // buffer instead — the device test's poison rows are what cover that end.
            let n = f64::from(c.o[((hd + 1) % c.heads) * d]);
            sq += n * n;
        }
        let divisor = match form {
            Form::NormOverTheSum => 1.0,
            Form::MeanOverThePaddedWidth => padded(d) as f64,
            _ => d as f64,
        };
        let inv = if form == Form::EpsOnTheSqrt {
            1.0 / ((sq / divisor).sqrt() + eps)
        } else {
            1.0 / (sq / divisor + eps).sqrt()
        };
        for j in 0..d {
            let x = f64::from(row[j]) * inv;
            let zj = f64::from(c.z[hd * d + j]);
            let g = match form {
                Form::GateOnTheNormedStream => h::sigmoid(x),
                Form::GateSilu => zj * h::sigmoid(zj),
                Form::GateTanh => zj.tanh(),
                _ => h::sigmoid(zj),
            };
            out[hd * d + j] = (x * f64::from(c.weight[j]) * g) as f32;
        }
    }
    out
}

/// Every (draw, GDN layer) pair, with the norm weight recovered from head 0.
fn for_each_gate(mut f: impl FnMut(&str, &GateNorm)) {
    for d in h::decode_draws() {
        let heads = d.w("linear_num_value_heads");
        let head_dim = d.w("linear_value_head_dim");
        for layer in GDN_LAYERS {
            let m = format!("model.layers.{layer}.linear_attn");
            let o = d.flat(&format!("{m}.gated_delta_rule.out.o"), heads * head_dim);
            let z = d.flat(&format!("{m}.in_proj_z"), heads * head_dim);
            let want = d.flat(&format!("{m}.norm"), heads * head_dim);
            // Recovered from head 0 under the bare `w . xhat` form at eps 1e-6. Every factor on the
            // right is a captured value, so this is one division per element and no fitting.
            let sq: f64 = o[..head_dim]
                .iter()
                .map(|x| f64::from(*x) * f64::from(*x))
                .sum();
            let inv = 1.0 / (sq / head_dim as f64 + 1e-6).sqrt();
            let weight: Vec<f32> = (0..head_dim)
                .map(|j| {
                    let denom = f64::from(o[j]) * inv * h::sigmoid(f64::from(z[j]));
                    (f64::from(want[j]) / denom) as f32
                })
                .collect();
            assert!(
                weight.iter().all(|w| w.is_finite()),
                "the recovered weight must be finite — a zero in `o` or a saturated gate at head 0 \
                 would make it NaN, and `f32::max` IGNORES NaN, so an all-NaN prediction would \
                 score as a perfect match"
            );
            let c = GateNorm {
                heads,
                head_dim,
                o,
                z,
                weight,
                want,
            };
            f(&format!("{} L{layer}", d.salt), &c);
        }
    }
}

// ── deviceless ──────────────────────────────────────────────────────────────────────────────

/// The f64 host norm-gate predicts heads 1..8 from a weight recovered at head 0, at every GDN layer
/// of both draws.
#[test]
fn the_host_norm_gate_predicts_every_other_head() {
    let b = h::Bar::at("gdn_out_norm", HOST_WORST);
    // The site labels rather than a counter, and the expected count DERIVED from the two lists it
    // is the product of — a literal here was a jscpd match against the sibling GDN suite, and the
    // derived form also fails when a draw or a layer is dropped from either list instead of
    // silently scoring a shorter loop.
    let mut seen: Vec<String> = Vec::new();
    for_each_gate(|at, c| {
        assert_eq!(
            c.head_dim, 20,
            "{at}: this suite's whole reason is a head_dim the old guard refused; at a power of two \
             it would be scoring the pre-padding kernel and the padding claim would be vacuous"
        );
        assert_ne!(
            padded(c.head_dim),
            c.head_dim,
            "{at}: head_dim {} is already a power of two, so no lane is masked and \
             MeanOverThePaddedWidth is inert — the fixture cannot price the change",
            c.head_dim
        );
        let got = gate_norm(c, Form::Reference);
        b.hold(at, "norm", &c.predicted(&got), &c.predicted(&c.want));
        seen.push(at.to_string());
    });
    seen.sort();
    seen.dedup();
    assert_eq!(
        seen.len(),
        GDN_LAYERS.len() * h::DECODE.len(),
        "one distinct site per (draw, GDN layer): {seen:?}"
    );
}

/// **Every defect form prices itself, and the two that price the PADDING CHANGE are the two
/// weakest.**
///
/// **Blind to a wrong REFERENCE arm, measured twice on this very suite** — it stayed green under
/// both of the plants recorded in this file's header, each of which reddened
/// [`the_host_norm_gate_predicts_every_other_head`]. See `h::separations`' doc.
#[test]
fn every_defect_form_prices_the_difference_it_names() {
    let rows = [
        (
            Form::NeighbourFoldedIn,
            "a neighbour folded in by an off-by-one mask",
            NEIGHBOUR_FOLDED_IN,
        ),
        (
            Form::MeanOverThePaddedWidth,
            "the mean divided by the PADDED width",
            MEAN_OVER_THE_PADDED_WIDTH,
        ),
        (
            Form::NormOverTheSum,
            "norm over the sum, not the mean",
            NORM_OVER_THE_SUM,
        ),
        (
            Form::GateOnTheNormedStream,
            "gdn_output_gate_on_normed",
            GATE_ON_THE_NORMED_STREAM,
        ),
        (
            Form::EpsFromTheWrongField,
            "eps_from_the_wrong_config_field",
            EPS_FROM_THE_WRONG_FIELD,
        ),
        (Form::GateSilu, "output_gate_silu", GATE_SILU),
        (Form::GateTanh, "output_gate_tanh", GATE_TANH),
        (Form::EpsOnTheSqrt, "eps_outside_sqrt", EPS_ON_THE_SQRT),
        (Form::EpsDropped, "the eps dropped", EPS_DROPPED),
    ];
    for (form, name, floor) in rows {
        let mut sites = Vec::new();
        for_each_gate(|at, c| {
            let got = gate_norm(c, form);
            sites.push((
                at.to_string(),
                h::rel(&c.predicted(&got), &c.predicted(&c.want)),
            ));
        });
        h::separations(h::Bar::at("gdn_out_norm", HOST_WORST), name, floor, &sites);
    }
}

/// **The offset convention is NOT scorable here, and this test is the proof rather than a comment.**
///
/// Recovering the weight under `(1 + w) ⊙ x̂` and predicting with the same form reproduces the
/// anchor just as well as the bare form does — the recovery absorbs the offset exactly. So this
/// fixture cannot choose between them, and a suite that claimed otherwise would be reading its own
/// arithmetic back.
#[test]
fn the_recovered_weight_cannot_choose_the_offset_convention() {
    let b = h::Bar::at("gdn_out_norm", HOST_WORST);
    for_each_gate(|at, c| {
        // The same recovery, one convention over: `1 + w' = want / (xhat · sigma(z))`, so
        // `w' = w - 1`, and predicting with `(1 + w')` gives back `w`.
        let offset: Vec<f32> = c.weight.iter().map(|w| w - 1.0).collect();
        let alt = GateNorm {
            heads: c.heads,
            head_dim: c.head_dim,
            o: c.o.clone(),
            z: c.z.clone(),
            weight: offset.iter().map(|w| w + 1.0).collect(),
            want: c.want.clone(),
        };
        let got = gate_norm(&alt, Form::Reference);
        b.hold(
            at,
            "zero-centred reading",
            &alt.predicted(&got),
            &alt.predicted(&c.want),
        );
    });
}

// ── on the device ───────────────────────────────────────────────────────────────────────────

/// **The launcher reproduces the anchor at `head_dim = 20`** — the width guard 1003 refused until
/// 2026-09-01, so this test failing to LAUNCH is the regression signal for the padding change.
///
/// # RED-PROOF PLAN — for the integrator's first device run
///
/// Mutations go in `kernels/recurrent.hip`'s `rmsnorm_gate_heads_f32`; each must redden this test
/// while [`the_host_norm_gate_predicts_every_other_head`] stays green:
///
/// * `t <= head_dim` in the mask — the off-by-one, ~5.6e-4 at its weakest, and the LAST head then
///   reads past the buffer, which [`the_masked_lanes_cannot_reach_the_result`] catches as bytes;
/// * divide by `blockDim.x` instead of `head_dim` — ~5.7e-3, the mistake the change had to avoid;
/// * drop the `if (!live) return;` so masked lanes store — the neighbouring head's row 0 is
///   overwritten, which moves `norm` at every head but the last;
/// * `sigma(x * inv)` instead of `sigma(g)` — ~2.4e-1.
///
/// **And the OWED one, which is not a plant:** the before/after byte comparison of
/// `kernel_k3_conv_norm.rs` across the pre- and post-padding binaries, at K3's power-of-two
/// `head_dim`. The four-part bit-identity argument at the kernel covers the arithmetic; it does not
/// cover what the compiler emits for the select, and that is the only claim left standing on prose.
/// One launch, with the caller's own `o` and `gate` rows — which the poison test hands in LONGER
/// than `heads · head_dim`, and that is the only reason this takes them as slices rather than
/// reading them off the case.
///
/// One helper for the two device tests below: they are the same eleven-argument call and jscpd
/// matched them, correctly — eleven positional arguments where `heads` and `head_dim` are both bare
/// `usize` is a place a transposition is a wrong answer rather than a compile error, and it should
/// be written once.
#[cfg(feature = "rocm")]
fn launch(
    s: &rivoli_backend::gpustream::HipStream,
    c: &GateNorm,
    o: &[f32],
    z: &[f32],
) -> Vec<f32> {
    use common::{DeviceBuf, back, dev, f32b, f32v, ok, zeros};
    use rivoli_backend::hip::{HeadCount, HeadDim, launch_rmsnorm_gate_heads_f32};

    let cp = |d: &DeviceBuf| d.ptr() as *const f32;
    let (od, zd) = (dev(&f32b(o)), dev(&f32b(z)));
    let w = dev(&f32b(&c.weight));
    let mut out = zeros(c.heads * c.head_dim * 4);
    let dst = out.ptr_mut() as *mut f32;
    // SAFETY: `o` and `z` hold at least `heads · head_dim` f32 (the poison caller passes more),
    // `w` holds `head_dim`, and `out` is a distinct `heads · head_dim` destination — four separate
    // live allocations, so the kernel's all-`__restrict__` contract holds, and `back` joins.
    ok(
        unsafe {
            launch_rmsnorm_gate_heads_f32(
                cp(&od),
                cp(&zd),
                cp(&w),
                HeadCount(c.heads),
                HeadDim(c.head_dim),
                1e-6,
                dst,
                s.raw(),
            )
        },
        "rmsnorm_gate_heads_f32",
    );
    f32v(&back(&out))
}

#[cfg(feature = "rocm")]
#[test]
fn the_norm_gate_launcher_matches_the_anchor_at_the_width_the_guard_refused() {
    use common::stream;

    let b = h::Bar::at("gdn_out_norm", HOST_WORST);
    let s = stream();
    for_each_gate(|at, c| {
        let got = launch(&s, c, &c.o, &c.z);
        b.hold(at, "norm", &c.predicted(&got), &c.predicted(&c.want));
    });
}

/// **The masked lanes cannot reach the result, scored as BYTES.**
///
/// The padding exists on the LAST head only in a buffer whose stride is `head_dim`: for
/// `blockIdx.x = heads - 1`, a lane at `t >= head_dim` addresses past the end. So the input is
/// allocated with `padded(head_dim) - head_dim` floats of slack and the launch is run TWICE with
/// two different poison patterns in that slack — `NaN` and `1e30`. The two outputs must be
/// **byte-identical**: not close, identical, because a masked lane either contributes nothing or
/// contributes a number, and there is no third state.
///
/// **Why bytes and not a tolerance.** `NaN` is the sharper poison and it is also the one a
/// tolerance would hide: `f32::max` ignores NaN, so a leak of the NaN row would make the affected
/// head's output NaN and score as a PERFECT match against any envelope (this repo has a broken
/// kernel passing 9 of 9 that way on record). A `to_bits` comparison cannot be fooled that way, and
/// the finiteness of BOTH runs is asserted before the comparison is believed.
///
/// **What it does NOT cover, said out loud:** the eight non-final heads. A masked lane there reads
/// the NEXT head's real row, identical in both runs, so this test is blind to a leak at those heads
/// — [`the_norm_gate_launcher_matches_the_anchor_at_the_width_the_guard_refused`] is what covers
/// them, because a folded-in neighbour moves the mean by ~5.6e-4 against a 2.5e-6 bar.
#[cfg(feature = "rocm")]
#[test]
fn the_masked_lanes_cannot_reach_the_result() {
    use common::stream;

    let s = stream();
    let mut ran = 0;
    for_each_gate(|at, c| {
        let slack = padded(c.head_dim) - c.head_dim;
        assert!(
            slack > 0,
            "{at}: no lane is masked, so this test would be vacuous"
        );
        // The poisoned rows are LONGER than the case's, so even a mask that ran one lane past the
        // end stays INSIDE the allocation — the point being that a leak shows up as a changed
        // number rather than as a fault, which could be mistaken for the plant working.
        let run = |poison: f32| -> Vec<u32> {
            let grow = |v: &[f32]| {
                let mut x = v.to_vec();
                x.extend(std::iter::repeat_n(poison, slack));
                x
            };
            launch(&s, c, &grow(&c.o), &grow(&c.z))
                .iter()
                .map(|x| x.to_bits())
                .collect()
        };
        let nan_run = run(f32::NAN);
        let big_run = run(1e30);
        // Believed only if both are finite: an all-NaN pair is byte-identical and says nothing.
        // **DO NOT SIMPLIFY THIS LOOP AWAY.** It is the finiteness precondition, and on 2026-09-01
        // it FIRED on real hardware: under the off-by-one mask plant the NaN row leaked into the
        // output and this assert stopped the run *before* the `assert_eq!` below could compare two
        // all-NaN results and report them byte-identical. `f32::max` ignores NaN and this repo has
        // a broken kernel passing 9 of 9 comparisons on record; here the trap was caught in the
        // act, on silicon, by exactly these four lines.
        for (name, bits) in [("NaN-poisoned", &nan_run), ("1e30-poisoned", &big_run)] {
            assert!(
                bits.iter().all(|b| f32::from_bits(*b).is_finite()),
                "{at}: the {name} run produced a non-finite output, so the two runs agreeing \
                 byte-for-byte would prove nothing — the masked lanes ARE reaching the result"
            );
        }
        assert_eq!(
            nan_run, big_run,
            "{at}: changing only the PADDING lanes changed the output, so a masked lane is \
             contributing to the reduction"
        );
        ran += 1;
    });
    assert_eq!(ran, 8, "two draws times four GDN layers");
}

/// The launcher accepts `head_dim = 20` and still refuses what it should.
///
/// The first row IS the regression gate for the padding change: guard 1003 lived where that `None`
/// now is, and its return would make this suite's every other test unreachable.
#[cfg(feature = "rocm")]
#[test]
fn the_norm_gate_launcher_accepts_a_non_power_of_two_and_still_refuses_the_rest() {
    use common::{assert_guard, assert_guards, dev, f32b, zeros};
    use rivoli_backend::hip::{HeadCount, HeadDim, launch_rmsnorm_gate_heads_f32};

    let src = dev(&f32b(&[0.25f32; 512]));
    let mut out = zeros(512 * 4);
    let dst = out.ptr_mut() as *mut f32;
    let p = src.ptr() as *const f32;
    // Named for `kernel_qwen_gdn_recurrent.rs`'s reason: every call here is REFUSED before a
    // launch, so there is nothing for a stream to order — and it keeps this closure's tail from
    // being token-identical to the sibling suites', which jscpd reported.
    let no_stream: *mut std::ffi::c_void = std::ptr::null_mut();
    let call = |heads: usize, head_dim: usize, eps: f32| {
        // SAFETY: the buffers cover every legal case below; the illegal ones are refused before a
        // pointer is read, which is what is under test.
        unsafe {
            launch_rmsnorm_gate_heads_f32(
                p,
                p,
                p,
                HeadCount(heads),
                HeadDim(head_dim),
                eps,
                dst,
                no_stream,
            )
        }
    };
    assert_guard(
        call(4, 20, 1e-6),
        None,
        "head_dim 20 — what 1003 used to refuse",
    );
    assert_guard(
        call(2, 24, 0.0),
        None,
        "another non-power-of-two, eps zero is legal",
    );
    assert_guards([
        (1001, "zero heads", call(0, 8, 1e-6)),
        (1001, "zero head_dim", call(4, 0, 1e-6)),
        (
            1002,
            "a head_dim past the launch limit",
            call(1, 2048, 1e-6),
        ),
        (1006, "a NaN epsilon", call(4, 8, f32::NAN)),
    ]);
}

/// **The BEFORE/AFTER byte comparison for the guard-1003 padding change, at the power-of-two
/// widths the four shipped models run.**
///
/// The four-part bit-identity argument at the kernel predicts that padding is inert here. This test
/// is what makes the prediction falsifiable across two binaries: it drives the launcher at K3's
/// anchor width (32) and the real width (128) on a DETERMINISTIC draw, asserts the result against
/// the same f64 host oracle the rest of this suite uses, and prints every output word as raw hex
/// bits.
///
/// **Why a printer and not a stdout diff of `kernel_k3_conv_norm.rs`.** That suite prints no
/// numbers at all, so diffing its stdout compares pass/fail lines and wall-clock — two greens, and
/// no bytes. Byte-identity is a claim about the kernel's OUTPUT, so the output is what gets
/// printed. Run under `--nocapture` in a binary built from `recurrent.hip` at 43bb83a and again in
/// one built from HEAD, then `diff` the `BITS` lines; they must be identical.
///
/// The assertion half stands on its own in either binary: the widths are powers of two, so the
/// pre-padding kernel accepts them too and guard 1003 never fires.
#[cfg(feature = "rocm")]
#[test]
fn the_power_of_two_widths_reproduce_the_host_oracle_and_print_their_bytes() {
    use common::{fill, stream};

    let s = stream();
    let heads = 4usize;
    for head_dim in [32usize, 128] {
        assert_eq!(
            padded(head_dim),
            head_dim,
            "this test's whole subject is the widths where padding is INERT"
        );
        let n = heads * head_dim;
        // A fixed LCG draw, so the two binaries are handed the same bits. Scaled so `mean(o²)` is
        // well above the 1e-6 eps — an input where the eps dominated would make the comparison a
        // statement about the epsilon rather than about the reduction.
        let c = GateNorm {
            heads,
            head_dim,
            o: fill(n, 0x5157_454E, 1.0),
            z: fill(n, 0x4E4F_524D, 1.0),
            weight: fill(head_dim, 0x5041_4432, 1.0),
            // `gate_norm` does not read this field; the oracle IS what fills it below.
            want: Vec::new(),
        };
        let want = gate_norm(&c, Form::Reference);
        let got = launch(&s, &c, &c.o, &c.z);
        assert!(
            want.iter().all(|x| x.is_finite()),
            "the host oracle must be finite before any comparison is believed — `f32::max` \
             ignores NaN, so an all-NaN pair scores as a perfect match"
        );
        h::Bar::at("gdn_out_norm", HOST_WORST).hold(
            &format!("synthetic head_dim={head_dim}"),
            "norm",
            &got,
            &want,
        );
        for hd in 0..heads {
            let row: Vec<String> = got[hd * head_dim..(hd + 1) * head_dim]
                .iter()
                .map(|x| format!("{:08x}", x.to_bits()))
                .collect();
            println!("BITS head_dim={head_dim} head={hd} {}", row.join(" "));
        }
    }
}
