//! **The gated delta rule of Qwen3.8-Flash-Next's 36 GatedDeltaNet layers, scored against the
//! S2 anchor** — `gdn_recurrent_f32` and nothing else.
//!
//! # Why this is a new kernel and not Kimi-K3's
//!
//! `gated_delta_recurrent_f32` (M9, `kernels/recurrent.hip`) is the same FAMILY — a per-head
//! state decayed by a gate, corrected by a rank-one update towards the value, read back with
//! the query — and it is the wrong kernel here in four independent ways, each of which is a
//! silent wrong rather than a refusal:
//!
//! 1. **the decay form.** K3's is fla's safe gate, `alpha = exp(lower_bound · sigma(...))`;
//!    qwen's is `alpha = exp(-exp(A_log) · softplus(a + dt_bias))` with no bound anywhere
//!    (`qwen-architecture.md` §1, MOD:519). `softplus` and `sigma` differ only in scale on
//!    small inputs, which is trap T20.
//! 2. **`dt_bias` rank.** K3 indexes it `[head][channel]`; qwen's is `[head]`, a per-V-head
//!    scalar (T17, and CKPT's `linear_attn.dt_bias` is BF16 [48]). Handing a `[heads]` buffer
//!    to a `[heads][head_dim]` reader walks off the end of the first head.
//! 3. **the gate's rank.** K3's `alpha` is per KEY CHANNEL, so it decays the state's rows
//!    unevenly; qwen's is one scalar per head and scales the whole state.
//! 4. **the head asymmetry.** 16 QK heads against 48 V heads at the real widths, 3 against 9
//!    here. K3's ABI takes one head count.
//!
//! So the two are separate entry points, which is the same decision `rmsnorm_gate_heads_f32`
//! records for the output gate: *the two families must not share a kernel on the strength of
//! both having a gate.*
//!
//! # What this kernel does that the reference's boundary does not
//!
//! The anchor captures the rule at the reference's own call boundary, where `g` and `beta`
//! arrive already reduced and q/k already `repeat_interleave`d to 48 heads. This kernel takes
//! the RAW projections (`a`, `b`) and the UN-repeated q/k, and does the softplus, the sigmoid
//! and the repeat itself — so the fixture must un-do two things the reference did upstream.
//! Both are scored rather than assumed:
//!
//! * the fusion, against the reference's own captured `in.g` and `in.beta`
//!   ([`the_fused_decay_and_write_gate_reproduce_the_rules_own_inputs`]);
//! * the repeat, by asserting the three copies of each QK head are BIT-identical before they
//!   are collapsed, and that the tiled reading is not also consistent with them
//!   ([`the_repeated_qk_heads_are_interleaved_and_not_tiled`]).
//!
//! # What this fixture cannot say
//!
//! That the state PERSISTS. It is handed one `initial_state`, runs one step and compares one
//! `out.state`; whether the layer loop keeps 36 of them alive across a decode and never resets
//! one mid-stream is the layer loop's claim, owed to S6.
//!
//! And it says nothing about a CHUNKED prefill. The decode goldens were produced by
//! `torch_recurrent_gated_delta_rule`, so `gdn_op`'s 1.4959e-5 floor is the per-token path's;
//! the chunked body disagrees with it by 9.420e-5 over 36 layers, at which the weakest
//! targeting defect's margin falls to 171x and forces `ExactOnly`
//! (`qwen-reference/anchor.md` §the load-bearing finding). A prefill kernel needs its own
//! fixture and must not be scored here.
//!
//! # RED PROOF, OBSERVED 2026-09-01 — deviceless, on the host oracle
//!
//! Class: **GDN decay form** (`gdn_decay_sigmoid_gate`, the matrix's own row). The REFERENCE arm
//! of [`gates`] was moved to `sigmoid(a + dt)`; `diff` shows the one line and `cmp` against a saved
//! copy returns 1, so the tree changed, and the run recompiled (`Compiling rivoli-engine`).
//!
//! * [`the_fused_decay_and_write_gate_reproduce_the_rules_own_inputs`] **RED** —
//!   `qwen-anchor-1 L0 g: 4.8897043e-1 is outside the 1.5e-4 operator tolerance`. That `hold` is
//!   the FIRST assertion in the body, so nothing above it could have masked it.
//! * [`the_host_recurrence_reproduces_the_anchor_at_every_gdn_layer`] **RED** —
//!   `qwen-anchor-1 L0 o: 2.7860975e-2 …`, i.e. the decay's 4.9e-1 error reaches `o` two orders
//!   weaker, which is why the fused-gate test exists beside the recurrence one.
//! * Reverted, `cmp` rc 0, recompiled, 8 passed exit 0.
//!
//! **And the finding the plant produced:**
//! [`every_defect_form_prices_the_difference_it_names`] stayed **GREEN** throughout. It scores
//! variant-against-GOLDEN, so it never reads the reference arm and is structurally blind to a
//! wrong one. A defect matrix built only from separations would pass on a broken oracle: the two
//! kinds of test are independent and neither subsumes the other.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod kernel_qwen_harness;

use kernel_qwen_harness as h;

// `duplicate_mod` is ALLOWED here and the argument is specific: the harness reaches
// `common/scoring.rs` by `#[path]` because `common/mod.rs` does not compile without `rocm` (its
// header explains why nobody noticed), and under `rocm` this declaration loads that same file a
// second time through the umbrella. Both loads are needed — the metric in every arm, the device
// uploader in one — and neither is a copy: it is ONE file on disk reached by two routes. Fixing it
// properly means gating `impl AttendIo` in `common/upload.rs`, which is not this track's file.
#[cfg(feature = "rocm")]
#[allow(clippy::duplicate_mod)]
mod common;

/// The zero-based GDN layers among the anchor's captured seven (0, 1, 2, 3, 27, 46, 47).
///
/// The complement of the QSA list (3, 27, 47) over that set — derived from the real
/// `layer_types`, where index 3 and every fourth after it runs the indexer. Written out rather
/// than filtered so that a fixture which silently stopped covering one is visible HERE, as a
/// changed constant, rather than as a smaller loop count nobody reads.
const GDN_LAYERS: [usize; 4] = [0, 1, 2, 46];

/// The worst relative disagreement the f64 host recurrence shows against the anchor, over both
/// draws, all four GDN layers and BOTH outputs.
///
/// Measured 2026-09-01 on the vendored bytes: 2.8592e-1 was never in it — the reading is
/// **2.8592e-7**, and its floor over the eight sites is 9.0314e-8, so the spread is 3.2x. The
/// bar this becomes is 2.86e-6, which is 52x inside `gdn_op`'s 1.50e-4.
const HOST_WORST: f32 = 2.8592e-7;

/// The MINIMUM separation each defect form reaches, over both draws and all four layers.
///
/// They are minima and not maxima on purpose: the bar a variant has to clear is the weakest
/// site's, and quoting the strongest would let a form that is invisible at one layer pass on
/// another layer's number. `dt_bias_outside_softplus` spans 7.00e-2 to 3.08e+4 across the eight
/// sites, which is exactly why.
///
/// `L2_EPS_DROPPED` is the weakest and the one to watch: 3.9310e-4 is only 4.6x the 8.58e-5 the
/// fixture's own bar needs cleared, and the anchor's bucket-level reading for the same defect is
/// 1.610e-2 — two orders stronger, because a bucket carries the whole layer's amplification. A
/// re-vendor that moved this below 8.58e-5 would mean the eps has to be pinned by READING
/// MOD:261 rather than scored, exactly as K3's MLA eps is.
const L2_EPS_DROPPED: f32 = 3.9310e-4;
const DECAY_CLAMPED: f32 = 2.9556e-3;
const DECAY_SIGMOID_GATE: f32 = 4.3281e-2;
const DT_BIAS_OUTSIDE_SOFTPLUS: f32 = 7.0046e-2;
const OUTPUT_BEFORE_UPDATE: f32 = 6.7917e-1;
const BETA_PRE_SIGMOID: f32 = 9.5015e-1;
const STATE_VALUE_MAJOR: f32 = 9.5952e-1;
const QK_TILED: f32 = 1.2015e0;
const Q_SCALE_DROPPED: f32 = 3.4721e0;

/// One GDN layer's recurrence boundary, at the widths the launcher takes.
///
/// **Every quantity a caller could get wrong is RAW here.** `q` and `k` are pre-L2-norm and
/// pre-repeat; `a` and `b` are the bare projections, with neither softplus, sigmoid, `A_log`
/// nor `dt_bias` applied. That is not this struct's choice: it is where the kernel's boundary
/// is, and the fixture's job is to reach it from the reference's captures.
struct Gdn {
    qk_heads: usize,
    v_heads: usize,
    dk: usize,
    dv: usize,
    /// `[qk_heads][dk]`, un-repeated.
    q: Vec<f32>,
    /// `[qk_heads][dk]`, un-repeated.
    k: Vec<f32>,
    /// `[v_heads][dv]`.
    v: Vec<f32>,
    /// `[v_heads]` — `in_proj_a`, the bare decay projection.
    a: Vec<f32>,
    /// `[v_heads]` — `in_proj_b`, the bare write-gate projection.
    b: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    /// `[v_heads][dk][dv]`, `[key][value]`.
    state: Vec<f32>,
    want_o: Vec<f32>,
    want_state: Vec<f32>,
    /// What the reference handed its own rule, for the fusion's own comparison.
    want_g: Vec<f32>,
    want_beta: Vec<f32>,
}

/// The reference's recurrence, or one documented variant of it. **The variants ARE the
/// defects**: one body with one `form`, because a defect run is the correct oracle with exactly
/// one thing changed, and writing each out separately is how the two drift into differing by
/// something nobody intended.
///
/// Six of the ten are the anchor's own matrix rows and carry its names; four
/// (`QkTiled`, `OutputBeforeUpdate`, `BetaPreSigmoid`, `QScaleDropped`) are this boundary's
/// own, because they are arithmetic the reference performs INSIDE the call the anchor captures
/// around and no bucket-level defect can reach them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    Reference,
    /// `gdn_decay_sigmoid_gate`: fla's other gate shape, `sigma` where qwen has `softplus`.
    DecaySigmoidGate,
    /// `gdn_decay_clamped`: a lower bound on the log-decay, which is the habit K3's
    /// `gate_lower_bound` leaves behind. CKPT carries no such field.
    DecayClamped,
    /// `gdn_dt_bias_outside_softplus`: `softplus(a) + dt_bias` instead of
    /// `softplus(a + dt_bias)`.
    DtBiasOutsideSoftplus,
    /// `l2norm_eps_dropped`: the 1e-6 hard-coded at MOD:261 taken as zero.
    L2EpsDropped,
    /// `gdn_state_axis_swapped`: the state read and written with its two axes exchanged — which
    /// every shape check passes, because `dk == dv` here (20) and in the real model (128).
    StateValueMajor,
    /// The QK heads read as a TILE (`h % qk_heads`) rather than as a `repeat_interleave`
    /// (`h / rep`). No anchor row: the repeat happens upstream of the captured boundary.
    QkTiled,
    /// `o` read from the decayed-but-not-yet-updated state. No anchor row, and it is the
    /// ordering the delta rule is DEFINED by.
    OutputBeforeUpdate,
    /// `beta` used as the projection produced it. No anchor row — the reference's capture is
    /// already post-sigmoid, so only a fixture that reaches back to `in_proj_b` can see it.
    BetaPreSigmoid,
    /// The `1/sqrt(dk)` scale on `q` dropped. No anchor row; a pure scale on the output.
    QScaleDropped,
}

/// `F.softplus` with torch's own threshold — above 20 the `log1p(exp(x))` form overflows and
/// torch returns `x` (`beta = 1`, `threshold = 20`). The kernel spells the same branch.
fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// `(g, beta)` per V head — the two reductions the reference does before its call and this
/// kernel does inside it.
fn gates(c: &Gdn, form: Form) -> (Vec<f64>, Vec<f64>) {
    let g = (0..c.v_heads)
        .map(|hd| {
            let (a, dt) = (f64::from(c.a[hd]), f64::from(c.dt_bias[hd]));
            let scale = -f64::from(c.a_log[hd]).exp();
            let inner = match form {
                Form::DecaySigmoidGate => h::sigmoid(a + dt),
                Form::DtBiasOutsideSoftplus => softplus(a) + dt,
                _ => softplus(a + dt),
            };
            let raw = scale * inner;
            // A floor on the LOG-decay, not on alpha: that is what a `gate_lower_bound` port
            // writes, and -5.0 is fla's own inclusive bound, so the variant is the plausible
            // wrong rather than an arbitrary one.
            if form == Form::DecayClamped {
                raw.max(-5.0)
            } else {
                raw
            }
        })
        .collect();
    let beta = (0..c.v_heads)
        .map(|hd| {
            let b = f64::from(c.b[hd]);
            if form == Form::BetaPreSigmoid {
                b
            } else {
                h::sigmoid(b)
            }
        })
        .collect();
    (g, beta)
}

/// The L2 norm the rule applies to q and k — eps added to the SUM of squares INSIDE the root,
/// which is a different convention from every RMSNorm in this tree and is therefore spelled out
/// where it is used rather than shared (MOD:259-262).
fn l2(row: &[f32], eps: f64) -> Vec<f64> {
    let sum: f64 = row.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    let inv = 1.0 / (sum + eps).sqrt();
    row.iter().map(|x| f64::from(*x) * inv).collect()
}

/// One decode step, in f64, for one `form`. Returns `(o, state)` in the launcher's layouts.
fn recurrence(c: &Gdn, form: Form) -> (Vec<f32>, Vec<f32>) {
    let (dk, dv) = (c.dk, c.dv);
    let rep = c.v_heads / c.qk_heads;
    let eps = if form == Form::L2EpsDropped {
        0.0
    } else {
        1e-6
    };
    let (g, beta) = gates(c, form);
    let mut out = vec![0.0f32; c.v_heads * dv];
    let mut state = vec![0.0f32; c.v_heads * dk * dv];
    for hd in 0..c.v_heads {
        let src = if form == Form::QkTiled {
            hd % c.qk_heads
        } else {
            hd / rep
        };
        let qn: Vec<f64> = {
            let n = l2(&c.q[src * dk..src * dk + dk], eps);
            let s = if form == Form::QScaleDropped {
                1.0
            } else {
                1.0 / (dk as f64).sqrt()
            };
            n.iter().map(|x| x * s).collect()
        };
        let kn = l2(&c.k[src * dk..src * dk + dk], eps);
        let alpha = g[hd].exp();
        // The state, in the layout the KERNEL reads: `[key][value]`. `StateValueMajor` is the
        // port that took a `[128][128]` buffer at face value, so it loads and stores transposed
        // — and both directions have to move, or the round trip cancels the defect.
        let load = |i: usize, j: usize| -> f64 {
            let base = hd * dk * dv;
            f64::from(if form == Form::StateValueMajor {
                c.state[base + j * dk + i]
            } else {
                c.state[base + i * dv + j]
            })
        };
        let mut decayed = vec![0.0f64; dk * dv];
        for i in 0..dk {
            for j in 0..dv {
                decayed[i * dv + j] = alpha * load(i, j);
            }
        }
        // `u = S^T k` from the ALREADY-DECAYED state, which is the order MOD's loop body has
        // (decay, then `kv_mem`, then the rank-one update).
        let mut u = vec![0.0f64; dv];
        for i in 0..dk {
            for j in 0..dv {
                u[j] += kn[i] * decayed[i * dv + j];
            }
        }
        for j in 0..dv {
            let dvj = beta[hd] * (f64::from(c.v[hd * dv + j]) - u[j]);
            for i in 0..dk {
                let updated = decayed[i * dv + j] + kn[i] * dvj;
                let read = if form == Form::OutputBeforeUpdate {
                    decayed[i * dv + j]
                } else {
                    updated
                };
                out[hd * dv + j] += (qn[i] * read) as f32;
                let base = hd * dk * dv;
                let at = if form == Form::StateValueMajor {
                    base + j * dk + i
                } else {
                    base + i * dv + j
                };
                state[at] = updated as f32;
            }
        }
    }
    (out, state)
}

/// Every (draw, GDN layer) pair with its boundary assembled.
///
/// The loader lives inside its one caller and the widths come from the golden's own recorded
/// map, so a fixture cannot disagree with the tensor it is scoring.
fn for_each_gdn(mut f: impl FnMut(&str, usize, &Gdn)) {
    for d in h::decode_draws() {
        let v_heads = d.w("linear_num_value_heads");
        let qk_heads = d.w("linear_num_key_heads");
        let dk = d.w("linear_key_head_dim");
        let dv = d.w("linear_value_head_dim");
        let rep = v_heads / qk_heads;
        assert_eq!(
            rep * qk_heads,
            v_heads,
            "the QK heads must divide the V heads — 16 into 48 in the real model, 3 into 9 here"
        );
        assert_eq!(
            rep,
            d.w("gdn_repeat"),
            "the repeat this fixture derives must be the one the generator recorded"
        );
        for layer in GDN_LAYERS {
            let m = format!("model.layers.{layer}.linear_attn");
            let r = format!("{m}.gated_delta_rule");
            let repeated_q = d.flat(&format!("{r}.in.query"), v_heads * dk);
            let repeated_k = d.flat(&format!("{r}.in.key"), v_heads * dk);
            let c = Gdn {
                qk_heads,
                v_heads,
                dk,
                dv,
                q: derepeat(&repeated_q, qk_heads, rep, dk),
                k: derepeat(&repeated_k, qk_heads, rep, dk),
                v: d.flat(&format!("{r}.in.value"), v_heads * dv),
                a: d.flat(&format!("{m}.in_proj_a"), v_heads),
                b: d.flat(&format!("{m}.in_proj_b"), v_heads),
                a_log: d.flat(&format!("{r}.in.A_log"), v_heads),
                dt_bias: d.flat(&format!("{r}.in.dt_bias"), v_heads),
                state: d.flat(&format!("{r}.in.initial_state"), v_heads * dk * dv),
                want_o: d.flat(&format!("{r}.out.o"), v_heads * dv),
                want_state: d.flat(&format!("{r}.out.state"), v_heads * dk * dv),
                want_g: d.flat(&format!("{r}.in.g"), v_heads),
                want_beta: d.flat(&format!("{r}.in.beta"), v_heads),
            };
            f(d.salt, layer, &c);
        }
    }
}

/// Collapse `repeat_interleave(rep)` back to the projection's own heads, taking the FIRST copy
/// of each group.
///
/// Deliberately does not check the copies agree — that is
/// [`the_repeated_qk_heads_are_interleaved_and_not_tiled`]'s job, and a helper that asserted it
/// would make the test tautological on its own input.
fn derepeat(v: &[f32], qk_heads: usize, rep: usize, dk: usize) -> Vec<f32> {
    (0..qk_heads)
        .flat_map(|hd| v[hd * rep * dk..hd * rep * dk + dk].to_vec())
        .collect()
}

// ── deviceless ──────────────────────────────────────────────────────────────────────────────

/// The f64 host recurrence reproduces the anchor at every GDN layer of both draws.
///
/// This is the claim the device test rests on: it is device-free, so a mutation that reddens it
/// has broken the tree rather than the kernel.
#[test]
fn the_host_recurrence_reproduces_the_anchor_at_every_gdn_layer() {
    let b = h::Bar::at("gdn_op", HOST_WORST);
    let mut sites = 0;
    for_each_gdn(|salt, layer, c| {
        let (o, state) = recurrence(c, Form::Reference);
        let at = format!("{salt} L{layer}");
        b.hold(&at, "o", &o, &c.want_o);
        b.hold(&at, "state", &state, &c.want_state);
        sites += 1;
    });
    assert_eq!(sites, 8, "two draws times four GDN layers");
}

/// The kernel's fused decay and write gate reproduce the two quantities the reference computed
/// OUTSIDE its own call — scored against the reference's own captures of them, not inferred.
///
/// Without this the fusion would be unscored: `o` and `state` are smooth in `g`, so a decay
/// form that is wrong by a small factor lands inside the tolerance at some layers. The bar here
/// is `gdn_op`'s, because these two values are inputs to that bucket.
#[test]
fn the_fused_decay_and_write_gate_reproduce_the_rules_own_inputs() {
    let b = h::Bar::at("gdn_op", HOST_WORST);
    for_each_gdn(|salt, layer, c| {
        let (g, beta) = gates(c, Form::Reference);
        let at = format!("{salt} L{layer}");
        let g32: Vec<f32> = g.iter().map(|x| *x as f32).collect();
        let beta32: Vec<f32> = beta.iter().map(|x| *x as f32).collect();
        b.hold(&at, "g", &g32, &c.want_g);
        b.hold(&at, "beta", &beta32, &c.want_beta);
    });
}

/// The three copies of each QK head are BIT-identical, and the tiled reading is not also
/// consistent with them.
///
/// Both halves are needed. The first says the reference used `repeat_interleave` and licenses
/// [`derepeat`]; the second says the fixture could TELL, which at `qk_heads = 3` and
/// `rep = 3` is not obvious — a tile and an interleave agree whenever the head count and the
/// repeat coincide on a permutation, and they do not here only because the draws differ per
/// head. `decay_rates_are_not_all_equal` is the same anti-degeneracy check one quantity over.
#[test]
fn the_repeated_qk_heads_are_interleaved_and_not_tiled() {
    for d in h::decode_draws() {
        let (v_heads, qk_heads) = (d.w("linear_num_value_heads"), d.w("linear_num_key_heads"));
        let (dk, rep) = (d.w("linear_key_head_dim"), d.w("gdn_repeat"));
        for layer in GDN_LAYERS {
            let r = format!("model.layers.{layer}.linear_attn.gated_delta_rule");
            for which in ["query", "key"] {
                let x = d.flat(&format!("{r}.in.{which}"), v_heads * dk);
                let row = |hd: usize| &x[hd * dk..hd * dk + dk];
                for src in 0..qk_heads {
                    for copy in 1..rep {
                        assert_eq!(
                            row(src * rep),
                            row(src * rep + copy),
                            "{} L{layer} {which}: head {} is not a bit-identical copy of {}, so \
                             the reference did not `repeat_interleave` and this fixture's \
                             de-repeat is unfounded",
                            d.salt,
                            src * rep + copy,
                            src * rep
                        );
                    }
                }
                // The tiled reading must DISAGREE somewhere, or the distinction is vacuous.
                let tiled_agrees = (0..v_heads).all(|hd| row(hd) == row(hd % qk_heads));
                assert!(
                    !tiled_agrees,
                    "{} L{layer} {which}: the tiled reading agrees with the interleaved one on \
                     this draw, so `QkTiled` prices nothing here and the fixture is degenerate",
                    d.salt
                );
            }
        }
    }
}

/// **Every defect form moves the operator clear of this fixture's own bar**, and by at least
/// the minimum recorded for it.
///
/// **This test is blind to a wrong REFERENCE arm** — it scores variant-against-golden and never
/// reads one. Measured: it stayed green under the plant that reddened
/// [`the_host_recurrence_reproduces_the_anchor_at_every_gdn_layer`]. `h::separations`' doc carries
/// the general statement and the four observations behind it.
///
/// Two assertions per form, and the second is the one that catches a plant landing somewhere
/// other than intended: a separation an order below its constant means the variant did not do
/// what its name says.
#[test]
fn every_defect_form_prices_the_difference_it_names() {
    let forms = [
        (Form::L2EpsDropped, "l2norm_eps_dropped", L2_EPS_DROPPED),
        (Form::DecayClamped, "gdn_decay_clamped", DECAY_CLAMPED),
        (
            Form::DecaySigmoidGate,
            "gdn_decay_sigmoid_gate",
            DECAY_SIGMOID_GATE,
        ),
        (
            Form::DtBiasOutsideSoftplus,
            "gdn_dt_bias_outside_softplus",
            DT_BIAS_OUTSIDE_SOFTPLUS,
        ),
        (
            Form::OutputBeforeUpdate,
            "output read before the update",
            OUTPUT_BEFORE_UPDATE,
        ),
        (Form::BetaPreSigmoid, "beta pre-sigmoid", BETA_PRE_SIGMOID),
        (
            Form::StateValueMajor,
            "gdn_state_axis_swapped",
            STATE_VALUE_MAJOR,
        ),
        (Form::QkTiled, "QK heads tiled, not interleaved", QK_TILED),
        (
            Form::QScaleDropped,
            "1/sqrt(dk) dropped from q",
            Q_SCALE_DROPPED,
        ),
    ];
    for (form, name, floor) in forms {
        let mut sites = Vec::new();
        for_each_gdn(|salt, layer, c| {
            let (o, state) = recurrence(c, form);
            sites.push((
                format!("{salt} L{layer}"),
                h::rel(&o, &c.want_o).max(h::rel(&state, &c.want_state)),
            ));
        });
        h::separations(h::Bar::at("gdn_op", HOST_WORST), name, floor, &sites);
    }
}

// ── on the device ───────────────────────────────────────────────────────────────────────────

/// The kernel reproduces the anchor at every GDN layer of both draws, `o` and `state`.
///
/// **`state` is updated IN PLACE**, which is what the decode path does — a per-layer state
/// buffer is uploaded once and mutated every token, never copied. The aliasing argument is at
/// the kernel; here the fixture uploads the anchor's `initial_state` into the buffer the launch
/// writes and reads `out.state` back out of it, so the in-place contract is exercised rather
/// than described.
///
/// # RED-PROOF PLAN — for the integrator's first device run
///
/// Each mutation goes in `kernels/recurrent.hip`'s `gdn_recurrent_f32`, and for each this test
/// must go RED while [`the_host_recurrence_reproduces_the_anchor_at_every_gdn_layer`] stays
/// GREEN. The expected magnitudes are the `MEASURE`-style constants above, which the deviceless
/// [`every_defect_form_prices_the_difference_it_names`] has already established are reachable at
/// these widths:
///
/// * `softplus` -> `sigmoid` in the decay — ~4.3e-2;
/// * `softplus(a) + dt_bias` instead of `softplus(a + dt_bias)` — ~7.0e-2;
/// * drop the `+ 1e-6` from either L2 norm — ~3.9e-4, the weakest and the one that needs the
///   bar to be the fixture's rather than the tolerance;
/// * index the state `[j][i]` — ~9.6e-1;
/// * read `q`/`k` at `h % qk_heads` — ~1.2e0;
/// * accumulate `o` before the rank-one update — ~6.8e-1;
/// * drop the `1/sqrt(dk)` on `q` — 3.47e0 exactly, at every site, because it is a pure scale.
///
/// A red an order below its constant means the mutation landed somewhere other than intended.
#[cfg(feature = "rocm")]
#[test]
fn the_gdn_recurrence_kernel_matches_the_anchor_at_every_gdn_layer() {
    use common::{DeviceBuf, back, dev, f32b, f32v, ok, stream, zeros};
    use rivoli_backend::hip::launch_gdn_recurrent_f32;

    // Seven `x.ptr() as *const f32,` lines in a row is the shape jscpd matched against
    // `kernel_k3_conv_norm.rs`'s launch block, and it was right about the substance: the cast is
    // the same on every one of them, so writing it seven times is seven chances to write
    // `ptr_mut` on an input. One closure, named for what it produces.
    let r = |d: &DeviceBuf| d.ptr() as *const f32;
    let b = h::Bar::at("gdn_op", HOST_WORST);
    let s = stream();
    for_each_gdn(|salt, layer, c| {
        let (q, k, v) = (dev(&f32b(&c.q)), dev(&f32b(&c.k)), dev(&f32b(&c.v)));
        let (a, bb) = (dev(&f32b(&c.a)), dev(&f32b(&c.b)));
        let (a_log, dt) = (dev(&f32b(&c.a_log)), dev(&f32b(&c.dt_bias)));
        // Uploaded FRESH per site: the state is mutated in place, so a fixture that reused one
        // buffer across two launches would score the second step of a two-token sequence against
        // a one-token golden.
        let mut state = dev(&f32b(&c.state));
        let mut out = zeros(c.v_heads * c.dv * 4);
        // The two WRITTEN pointers, named before the launch rather than cast inside its argument
        // list — which is where jscpd matched `kernel_k3_conv_norm.rs` a second time, and naming
        // them says out loud which two of the thirteen arguments the kernel mutates.
        let state_out = state.ptr_mut() as *mut f32;
        let o_out = out.ptr_mut() as *mut f32;
        // SAFETY: every pointer is a live device buffer of the size the launcher's contract
        // states, sized from this fixture's own widths; `state` is the only one written besides
        // `out` and neither aliases an input.
        ok(
            unsafe {
                launch_gdn_recurrent_f32(
                    r(&q),
                    r(&k),
                    r(&v),
                    r(&a),
                    r(&bb),
                    r(&a_log),
                    r(&dt),
                    c.qk_heads,
                    c.v_heads,
                    c.dk,
                    c.dv,
                    state_out,
                    o_out,
                    s.raw(),
                )
            },
            "gdn_recurrent_f32",
        );
        // `back` copies out, which joins — there is no separate sync to forget.
        let at = format!("{salt} L{layer}");
        b.hold(&at, "o", &f32v(&back(&out)), &c.want_o);
        b.hold(&at, "state", &f32v(&back(&state)), &c.want_state);
    });
}

/// The launcher refuses the four argument shapes that are wrong rather than merely unusual.
///
/// Guards, not clamps: each of these means the caller read a config field wrong, and a kernel
/// that quietly defined an answer for it would produce a plausible number.
#[cfg(feature = "rocm")]
#[test]
fn the_launcher_refuses_geometry_it_cannot_mean() {
    use common::{assert_guard, dev, f32b, zeros};
    use rivoli_backend::hip::launch_gdn_recurrent_f32;

    let one = dev(&f32b(&[1.0f32; 64]));
    let mut st = zeros(64 * 4);
    let mut out = zeros(64 * 4);
    let p = one.ptr() as *const f32;
    let state_out = st.ptr_mut() as *mut f32;
    let o_out = out.ptr_mut() as *mut f32;
    // Named, because every one of these calls is REFUSED before a launch and so there is nothing
    // for a stream to order — the same reason `common::stream`'s note gives for the guard tests
    // that pass a null one. It also stops this closure's tail being token-identical to the sibling
    // suite's, which jscpd reported.
    let no_stream: *mut std::ffi::c_void = std::ptr::null_mut();
    let call = |qk: usize, vh: usize, dk: usize, dv: usize| {
        // SAFETY: the buffers above are large enough for every legal case below, and the
        // ILLEGAL ones are refused before a pointer is dereferenced — which is what is under
        // test.
        unsafe {
            launch_gdn_recurrent_f32(
                p, p, p, p, p, p, p, qk, vh, dk, dv, state_out, o_out, no_stream,
            )
        }
    };
    assert_guard(call(0, 4, 4, 4), Some(1001), "zero QK heads");
    assert_guard(call(2, 4, 0, 4), Some(1001), "zero key width");
    assert_guard(
        call(3, 4, 4, 4),
        Some(1003),
        "V heads not a multiple of QK heads",
    );
    assert_guard(
        call(1, 1, 2048, 4),
        Some(1002),
        "a block past the launch limit",
    );
    assert_guard(call(2, 4, 4, 4), None, "the legal shape must launch");
}
