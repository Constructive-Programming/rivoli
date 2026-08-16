//! **The gated delta recurrence on the device, scored against the S2 anchor** —
//! `gated_delta_recurrent_f32` and nothing else. Ported from `k3:tests/k3_kernels.rs` item
//! 5a (banner at :2414); shared spine in `tests/k3/mod.rs`.
//!
//! The largest kernel in the K3 port. fla fuses §4's ten KDA steps into three observable
//! boundaries and this is the middle one: everything the recurrence does that no document
//! outside fla attests to is INSIDE it — the q/k L2 norm, the beta sigmoid, the gate's lower
//! bound and the state's axis order are all arithmetic the reference performs after the last
//! thing a module hook can see. The four `Kda*` anchor defect runs price exactly this
//! boundary, and the variant tests below reproduce each of them against the host oracle
//! (`k3:tests/k3_kernels.rs:2414`).
//!
//! **What this fixture cannot say** is that the state PERSISTS correctly. It is handed one
//! `initial_state`, runs one step, and compares one `out.state`; whether the layer loop
//! keeps 69 of them alive across a sequence and never resets them mid-decode is the layer
//! loop's, exactly as the AttnRes stack is.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! The k3 tree red-proved this fixture against the DEVICE six ways, six reds
//! (`k3:tests/k3_kernels.rs:2436`) — that recipe is this plan. Each mutation goes in
//! `kernels/recurrent.hip`, and for each,
//! [`the_gated_delta_recurrence_matches_the_anchor_at_every_kda_layer`] must go RED while
//! [`the_kda_host_oracle_agrees_with_the_anchor`] stays GREEN (it is device-free — a
//! mutation that reddens it has broken the tree, not the kernel):
//!
//! * drop the `d^-0.5` scale from q — separation ~1e0 territory;
//! * swap the state's two axes (read `S[j][i]`) — the anchor's `KdaStateLayout`, expected
//!   ~7.6e-1, the measured `MEASURE_STATEVALUEMAJOR`;
//! * read `o` from the pre-update state — no anchor defect covers it, expected ~2.2e-1;
//! * take `beta` pre-sigmoid — ~7.0e-1;
//! * write the gate bound as a clamp — ~5.8e-1;
//! * remove the q/k L2 norm — ~8.4e-1.
//!
//! The magnitudes are `MEASURE_*` below; a red an order smaller than its constant means the
//! mutation landed somewhere other than intended.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_backend::hip::launch_gated_delta_recurrent_f32;

mod common;
mod k3;

use k3::*;

/// The captured layers the real map makes KDA — zero-based 0, 1 and 12; the complement of
/// the MLA list (3, 91, 92) over the SIX attention-captured layers, so a fixture that
/// silently stopped covering one is visible here rather than in a tensor count.
/// (`k3:tests/k3_kernels.rs:2441` says "five", which is its MoE layer count — layer 0 is
/// dense but still carries a KDA block, so the attention partition is six. Reported to the
/// k3 tree rather than silently inherited.)
const KDA_LAYERS: [usize; 3] = [0, 1, 12];

/// The worst relative difference the recurrence shows against the anchor, over both draws,
/// all three KDA layers and BOTH outputs: at salt 1 layer 12 on `o`. `o` is the worse of the
/// two outputs at four of the six sites and the state at the other two (1.07e-7 and 1.09e-7)
/// — close enough to say the two are the same size (`k3:tests/k3_kernels.rs:2448`).
const KDA_OBSERVED_WORST: f32 = 2.265e-7;

/// The MINIMUM separation each defect form reaches, over both draws and all three KDA
/// layers. These exist because the k3 docs quote this item's separations and nothing checked
/// them — the "six orders" claim rested on a bar three orders weaker. `StateValueMajor` is
/// the one that settles the layout question; `OutputBeforeUpdate` is the weakest and the
/// form no anchor defect covers (`k3:tests/k3_kernels.rs:2459`).
const MEASURE_NOQKL2NORM: f32 = 8.359e-1;
const MEASURE_GATECLAMPED: f32 = 5.754e-1;
const MEASURE_BETAPRESIGMOID: f32 = 6.952e-1;
const MEASURE_STATEVALUEMAJOR: f32 = 7.567e-1;
const MEASURE_OUTPUTBEFOREUPDATE: f32 = 2.183e-1;

/// One KDA layer's recurrence boundary: every input fla's kernel takes, and both things it
/// returns.
///
/// **Every number a caller could get wrong is RAW here.** `q` and `k` are pre-L2-norm,
/// `beta` is pre-sigmoid, and `g` is the bare projection with neither `a_log` nor `dt_bias`
/// applied — not this struct's choice; it is where the anchor's `wrap_kda_ops` sits, because
/// fla does all of it internally. The two state fields are the ONE exception: they are
/// transposed out of the reference's layout into the kernel's on the way in — see
/// [`to_key_major`], which carries the measurement.
///
/// **The name carries the model and that is deliberate here.** This tree's naming rule
/// forbids model names on kernels — `gated_delta_recurrent_f32` is named for what it does —
/// but the fixture side tracks the ANCHOR: the captures are
/// `model.layers.N.kda.fused_recurrent_kda.*` and the defect runs are `Kda*`, both of which
/// predate this port and neither of which rivoli chooses. Renaming here would sever a
/// fixture from the names of the things it reads; the exemption stops at the `kernels/` and
/// `src/` boundary (`k3:tests/k3_kernels.rs:2485`).
struct Kda {
    heads: usize,
    head_dim: usize,
    /// fla's gate lower bound — a property of the CASE, not a per-launch knob: the golden
    /// cases read it off their own `tiny_config` and the synthetic ones carry the shipped
    /// `-5.0`, so every launch and every host walk of one case agree on it by construction.
    /// The guard test perturbs THIS field, which is exactly the caller mistake it guards.
    lb: f32,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    state: Vec<f32>,
    want_o: Vec<f32>,
    want_state: Vec<f32>,
}

/// Re-lay each head's state from the reference's `[value][key]` into the kernel's
/// `[key][value]`. Named for the POSTCONDITION, not the mechanism: "transpose" is
/// direction-free, and this is the one place in the port where getting the direction
/// backwards is invisible to every assertion.
///
/// **Measured, not chosen.** §4 writes the recurrence as `S[i][j]` with `i` the key channel,
/// and the state is square at both the tiny widths (32) and the real ones (128), so no shape
/// assertion can see which axis the reference's BUFFER puts first. Scoring both
/// interpretations of the anchor's own `initial_state` against its `out.o` settled it: with
/// the transpose the recurrence agrees to 2.5e-7, without it to 2.2e-1 to 5.6e-1 — three
/// sites' worth of separation, at both draws. **The port does not inherit that layout, and
/// does not pay a transpose either**: rivoli's state starts at zero and never leaves the
/// device, and `[key][value]` is the order that makes `S[i*d + t]` consecutive across the
/// threads of a wave — the whole reason `kernels/recurrent.hip` is a two-pass kernel rather
/// than four. So the transpose is a FIXTURE boundary, applied once here to compare two
/// conventions, and the `StateValueMajor` variant is the red-proof that this fixture can
/// tell them apart (`k3:tests/k3_kernels.rs:2556`).
fn to_key_major(v: &[f32], heads: usize, dim: usize) -> Vec<f32> {
    assert_eq!(v.len(), heads * dim * dim, "not a per-head square");
    (0..heads)
        .flat_map(|h| (0..dim).flat_map(move |i| (0..dim).map(move |j| (h, i, j))))
        .map(|(h, i, j)| v[h * dim * dim + j * dim + i])
        .collect()
}

/// Every (draw, KDA layer) pair with its boundary assembled — the loader lives inside its
/// one caller, and the widths come from the capture rather than from the config so that a
/// fixture cannot disagree with the tensor it is scoring.
fn for_each_kda(mut f: impl FnMut(&Site, Kda)) {
    let tol = tolerance::rel_tolerance("kda_op");
    for (salt, bytes) in GOLDENS {
        let g = load(bytes);
        let lb = lower_bound(&g);
        for layer in KDA_LAYERS {
            let m = format!("model.layers.{layer}.kda.fused_recurrent_kda");
            let get = |n: &str| {
                let (s, v) = float(&g, &format!("{m}.{n}"));
                (s.to_vec(), v.to_vec())
            };
            // `[1, 1, heads, head_dim]`.
            let (qs, q) = get("in.q");
            let (heads, head_dim) = (qs[2], qs[3]);
            let (ss, state) = get("in.initial_state");
            assert_eq!(
                ss,
                vec![1, heads, head_dim, head_dim],
                "{m}: the state is one square matrix per head — which is exactly why its \
                 axis order cannot be checked here and is measured instead"
            );
            let (os, want_o) = get("out.o");
            assert_eq!(os, qs, "{m}: the recurrence is width-preserving");
            let c = Kda {
                heads,
                head_dim,
                lb,
                q,
                k: get("in.k").1,
                v: get("in.v").1,
                g: get("in.g").1,
                beta: get("in.beta").1,
                a_log: get("in.A_log").1,
                dt_bias: get("in.dt_bias").1,
                state: to_key_major(&state, heads, head_dim),
                want_o,
                want_state: to_key_major(&get("out.state").1, heads, head_dim),
            };
            let site = Site { salt, layer, tol };
            f(&site, c);
        }
    }
}

/// The reference's recurrence, or one documented variant of it. **The variants are the
/// defects.** One body with one `form` rather than six functions — a defect run is the
/// correct oracle with exactly one thing changed, and writing it out separately is how the
/// two drift into differing by something nobody intended. Four of the five variants are the
/// anchor's own `--defect` runs and each is named for it (`k3:tests/k3_kernels.rs:2596`).
#[derive(Clone, Copy, PartialEq)]
enum KdaForm {
    Reference,
    /// `KdaNoQkL2Norm`: q and k used as projected.
    NoQkL2Norm,
    /// `KdaGateLowerBoundOff`: fla's OTHER gate form, where the bound clamps instead of
    /// multiplying.
    GateClamped,
    /// `KdaBetaSigmoidOutside`: `beta` taken as the projection produced it.
    BetaPreSigmoid,
    /// `KdaStateLayout`: the state buffer read and written with its two axes swapped —
    /// which is precisely the port that took the reference's `[value][key]` bytes at face
    /// value.
    StateValueMajor,
    /// **No anchor defect prices this one**, and it is the delta rule's defining ordering:
    /// `o` read from the decayed state instead of the updated one. §4 step 7 puts the read
    /// last, so a kernel that hoisted it above the rank-one update would be one line
    /// different and one token behind.
    OutputBeforeUpdate,
}

impl KdaForm {
    /// §4 step 3's per-head L2 norm, or the identity under `NoQkL2Norm` — `eps` added to
    /// the SUM of squares rather than to the mean, a different convention from every
    /// RMSNorm in this tree, and applied to q and k only. On the FORM with [`KdaForm::beta`]
    /// and [`KdaForm::alphas`] so the recurrence body reads as §4's steps rather than as a
    /// stack of variant branches — each variant's ONE flag still lives beside the
    /// arithmetic it perturbs.
    fn l2n(self, v: &[f32]) -> Vec<f64> {
        let s: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let inv = if self == KdaForm::NoQkL2Norm {
            1.0
        } else {
            1.0 / (s + 1e-6).sqrt()
        };
        v.iter().map(|&x| f64::from(x) * inv).collect()
    }

    /// §4 step 6's beta gate on the prediction error, or `BetaPreSigmoid`'s raw projection.
    fn beta(self, bp: f64) -> f64 {
        if self == KdaForm::BetaPreSigmoid {
            bp
        } else {
            1.0 / (1.0 + (-bp).exp())
        }
    }

    /// Head `h`'s per-channel decay — §4 steps 4-5 in the reference's form, or
    /// `GateClamped`'s. `lb` comes off the case, so the walk and the launch cannot disagree.
    fn alphas(self, c: &Kda, h: usize) -> Vec<f64> {
        let lb = f64::from(c.lb);
        let a = f64::from(c.a_log[h]).exp(); // PER HEAD
        let base = h * c.head_dim;
        (0..c.head_dim)
            .map(|i| {
                // `dt_bias` goes on BEFORE the scale, and `a` multiplies inside the sigmoid.
                let z = f64::from(c.g[base + i]) + f64::from(c.dt_bias[base + i]);
                if self == KdaForm::GateClamped {
                    // fla's `safe_gate=False` activation, verbatim from its docstring
                    // (fla 0.5.2's `fla/ops/kda/chunk.py:250-256`):
                    // `-exp(A_log)·softplus(g + dt_bias)`, with `lower_bound` as a floor.
                    // Both forms are bounded below by `lb` and monotone in `z`, which is
                    // what makes this the plausible wrong one rather than an obviously
                    // broken one.
                    lb.max(-a * z.exp().ln_1p()).exp()
                } else {
                    (lb / (1.0 + (-(a * z)).exp())).exp()
                }
            })
            .collect()
    }
}

/// One head's normed operands and gates — everything §4 computes BEFORE the state passes,
/// bundled so [`decay_pass`] and [`output_pass`] take one context instead of re-listing five
/// slices (the missing-abstraction shape this suite's own launchers are held to).
struct HeadStep {
    qn: Vec<f64>,
    kn: Vec<f64>,
    alpha: Vec<f64>,
    beta: f64,
    /// State strides: `[key][value]` in the reference form, swapped under `StateValueMajor`
    /// — the swapped pair IS the transposed read, with no branch per element.
    si: usize,
    sj: usize,
    /// `1` under `OutputBeforeUpdate` — an INDEX into `[updated, pre]`, not a branch, so
    /// the innermost loop is conditional-free while §4 step 7's read-last ordering (and its
    /// one variant) stays visible at the site it perturbs.
    before_update: usize,
}

impl HeadStep {
    fn at(&self, i: usize, j: usize) -> usize {
        i * self.si + j * self.sj
    }
}

/// §4 pass 1: decay the rows by the per-key-channel gate, and read `u = Sᵀk` off the
/// DECAYED state. `state` is ONE head's square.
fn decay_pass(s: &HeadStep, state: &mut [f64]) -> Vec<f64> {
    let dim = s.alpha.len();
    let mut u = vec![0f64; dim];
    for i in 0..dim {
        for j in 0..dim {
            let d = s.alpha[i] * state[s.at(i, j)];
            state[s.at(i, j)] = d;
            u[j] += s.kn[i] * d;
        }
    }
    u
}

/// §4 pass 2: the beta-gated prediction error (`v` is never normed) rank-one-updates the
/// state, and `o` reads the state LAST — step 7's ordering, whose variant rides
/// [`HeadStep::before_update`].
fn output_pass(s: &HeadStep, state: &mut [f64], v: &[f32], u: &[f64]) -> Vec<f32> {
    let dim = s.alpha.len();
    let mut out = vec![0f32; dim];
    for j in 0..dim {
        let dv = s.beta * (f64::from(v[j]) - u[j]);
        let mut o = 0.0;
        for i in 0..dim {
            let pre = state[s.at(i, j)];
            state[s.at(i, j)] = pre + s.kn[i] * dv;
            o += s.qn[i] * [state[s.at(i, j)], pre][s.before_update];
        }
        out[j] = o as f32;
    }
    out
}

/// §4 steps 3-7 in f64, over one decode step. The variants stay ONE body — each rides a
/// [`KdaForm`] selector hoisted out of the loops — and the two state passes are their own
/// functions so each is one readable chunk rather than a stack of nested loops.
fn host_kda(c: &Kda, form: KdaForm) -> (Vec<f32>, Vec<f32>) {
    let dim = c.head_dim;
    let mut state: Vec<f64> = c.state.iter().copied().map(f64::from).collect();
    let mut out = vec![0f32; c.heads * dim];
    // `d_k^-0.5` on q only, after the norm (§4 step 6). fla's `scale` defaults to this and
    // the reference passes no override, which is why the kernel computes it rather than
    // taking it.
    let scale = 1.0 / (dim as f64).sqrt();
    let (si, sj) = if form == KdaForm::StateValueMajor {
        (1, dim)
    } else {
        (dim, 1)
    };
    for h in 0..c.heads {
        let base = h * dim;
        let s = HeadStep {
            qn: form
                .l2n(&c.q[base..base + dim])
                .iter()
                .map(|x| x * scale)
                .collect(),
            kn: form.l2n(&c.k[base..base + dim]),
            alpha: form.alphas(c, h),
            beta: form.beta(f64::from(c.beta[h])),
            si,
            sj,
            before_update: usize::from(form == KdaForm::OutputBeforeUpdate),
        };
        let head_state = &mut state[h * dim * dim..(h + 1) * dim * dim];
        let u = decay_pass(&s, head_state);
        let row = output_pass(&s, head_state, &c.v[base..base + dim], &u);
        out[base..base + dim].copy_from_slice(&row);
    }
    (out, state.iter().map(|&x| x as f32).collect())
}

/// One launch, returning the launcher's own `Result` and both of the kernel's outputs.
///
/// `state` is updated IN PLACE, so the case's copy is uploaded fresh here on every call — a
/// fixture that reused one device buffer across two launches would be scoring the second
/// step of a two-token sequence against a one-token golden (`k3:tests/k3_kernels.rs:2702`).
fn kda_launch(c: &Kda) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let s = stream();
    let [q, k, v, g, beta, a_log, dt] =
        [&c.q, &c.k, &c.v, &c.g, &c.beta, &c.a_log, &c.dt_bias].map(|x| dev(&f32b(x)));
    let mut sb = dev(&f32b(&c.state));
    let mut ob = zeros((c.heads * c.head_dim).max(1) * 4);
    // SAFETY: every buffer is the size the launcher documents, all live until the stream
    // completes (`back` synchronises), and none aliases another — every pointer is
    // `__restrict__` in the kernel; rejected cases return before any launch.
    unsafe {
        launch_gated_delta_recurrent_f32(
            q.ptr() as *const f32,
            k.ptr() as *const f32,
            v.ptr() as *const f32,
            g.ptr() as *const f32,
            beta.ptr() as *const f32,
            a_log.ptr() as *const f32,
            dt.ptr() as *const f32,
            c.heads,
            c.head_dim,
            c.lb,
            sb.ptr_mut() as *mut f32,
            ob.ptr_mut() as *mut f32,
            s.raw(),
        )
    }?;
    let o = f32v(&back(&ob));
    Ok((o, f32v(&back(&sb))))
}

fn device_kda(c: &Kda) -> (Vec<f32>, Vec<f32>) {
    ok(kda_launch(c), "gated_delta_recurrent_f32")
}

/// One golden-backed scoring site: which draw, which layer, and the two numbers every
/// score there is held to. A type because the four always travel together from
/// [`for_each_kda`] into every test — spelled as four bare parameters they were this file's
/// own Primitive Obsession finding, and the k3 tree's `Fold` made the same move for the
/// same reason (`k3:tests/k3_kernels.rs:428`).
struct Site<'a> {
    salt: &'a str,
    layer: usize,
    tol: f32,
}

/// Score the recurrence's two outputs — both, because a kernel that produces the right `o`
/// from the wrong state agrees for exactly one token (`k3:tests/k3_kernels.rs:2734`).
fn score_kda(s: &Site, observed: f32, got: (&[f32], &[f32]), c: &Kda) {
    score_all(
        &format!("{} layer {}", s.salt, s.layer),
        Bars {
            tol: s.tol,
            observed,
        },
        &[("o", got.0, &c.want_o), ("state", got.1, &c.want_state)],
    );
}

/// A synthetic case at arbitrary widths, for the two things the goldens cannot reach: the
/// real geometry, and the launcher's refusals.
///
/// The gate input is drawn wide (`±12` before `dt_bias`) so that `alpha` reaches both ends
/// of its range — `1.0` exactly, the legitimate perfect retention the reference documents,
/// and the `e^-5` floor. Measured over both draws, layers 0/1/12 and all 768 gate channels:
/// the ANCHOR is concentrated at near-identity decay (37 alphas exactly 1.0, median 0.9981),
/// so what it under-covers is the MID-SCALE decay — which is why [`synthetic_kda_gain`]
/// takes the gain and one sweep case uses 0.03 (`k3:tests/k3_kernels.rs:2754`).
/// **`dt_bias` scales with the gate gain**, and it has to: `A_log` reaches `log(16)`, so
/// `exp(A_log)·(g + dt_bias)` is up to 16x whichever of the two is larger, and leaving
/// `dt_bias` at its own scale saturated the sigmoid at every gain tried. Measured: 12.0
/// reaches both ends of `alpha`, 0.03 keeps every channel inside [0.04, 0.17]
/// (`k3:tests/k3_kernels.rs:2773`).
fn synthetic_kda_gain(heads: usize, head_dim: usize, seed: u64, g_gain: f32) -> Kda {
    let mut r = Lcg(seed);
    let n = heads * head_dim;
    let mut draw = |len: usize, gain: f32| (0..len).map(|_| r.f() * gain).collect::<Vec<f32>>();
    Kda {
        heads,
        head_dim,
        // The shipped bound; a guard case that wants a bad one overwrites the field.
        lb: -5.0,
        q: draw(n, 1.0),
        k: draw(n, 1.0),
        v: draw(n, 1.0),
        g: draw(n, g_gain),
        beta: draw(heads, 4.0),
        // `log(uniform(1, 16))` is the anchor's own range for `A_log`, and it must not be
        // constant: a constant makes every head decay identically and a kernel ignoring the
        // term would match.
        a_log: draw(heads, 1.0)
            .iter()
            .map(|x| (1.0 + 15.0 * x.abs()).ln())
            .collect(),
        dt_bias: draw(n, 2.0 * g_gain / 12.0),
        state: draw(n * head_dim, 1.0),
        want_o: Vec::new(),
        want_state: Vec::new(),
    }
}

/// **The kernel reproduces the reference's recurrence — both outputs, both draws, all three
/// KDA layers.**
#[test]
fn the_gated_delta_recurrence_matches_the_anchor_at_every_kda_layer() {
    for_each_kda(|site, c| {
        let (o, state) = device_kda(&c);
        score_kda(site, KDA_OBSERVED_WORST, (&o, &state), &c);
    });
}

/// **The host oracle is the same function**, which is what lets the five variants below mean
/// anything.
///
/// Every red-proof in this file perturbs the ORACLE and asserts the perturbation moves the
/// result. That argument is empty unless the unperturbed oracle agrees with the reference to
/// the same order the kernel does — and it is the one comparison here that is device-free,
/// so a failure separates "the arithmetic is wrong" from "the kernel is wrong". The tripwire
/// inside `score_all` is what this site most needs: the five red-proofs score `moved`
/// against a device constant, so a regression that degraded THIS oracle into the
/// 6.8e-5..6.3e-4 band would leave everything green and quietly empty the sensitivity claim.
/// Worst over both draws and layers 0/1/12: 2.481e-7 (`k3:tests/k3_kernels.rs:2823`).
#[test]
fn the_kda_host_oracle_agrees_with_the_anchor() {
    for_each_kda(|site, c| {
        let (o, state) = host_kda(&c, KdaForm::Reference);
        score_kda(site, 2.481e-7, (&o, &state), &c);
    });
}

/// **Each of the four things that live only inside fla's kernel is a separate function, and
/// the anchor can see all four — plus the one ordering no anchor defect covers.**
///
/// One test over five variants rather than five tests, because they make the identical
/// argument about different lines. The bar is the tripwire cleared by the table's own 30x
/// `DEFECT_MARGIN` — and each form ALSO carries the minimum separation it actually reaches,
/// because the k3 docs quote these magnitudes and a separation that fell to a tenth of its
/// constant should fail even while it still clears the much weaker bar
/// (`k3:tests/k3_kernels.rs:2846`).
#[test]
fn the_recurrence_arithmetic_inside_flas_kernel_is_all_priced() {
    for_each_kda(|site, c| {
        for (form, what, measured) in [
            (
                KdaForm::NoQkL2Norm,
                "dropping the q/k L2 norm (--defect KdaNoQkL2Norm)",
                MEASURE_NOQKL2NORM,
            ),
            (
                KdaForm::GateClamped,
                "clamping the gate instead of scaling it (KdaGateLowerBoundOff)",
                MEASURE_GATECLAMPED,
            ),
            (
                KdaForm::BetaPreSigmoid,
                "taking beta pre-sigmoid (KdaBetaSigmoidOutside)",
                MEASURE_BETAPRESIGMOID,
            ),
            (
                KdaForm::StateValueMajor,
                "swapping the state's two axes (KdaStateLayout)",
                MEASURE_STATEVALUEMAJOR,
            ),
            (
                KdaForm::OutputBeforeUpdate,
                "reading o from the pre-update state (no anchor defect)",
                MEASURE_OUTPUTBEFOREUPDATE,
            ),
        ] {
            let (o, state) = host_kda(&c, form);
            // The WORSE of the two outputs, not the mean: a variant that left `o` untouched
            // while corrupting the state is still caught, and one that moved neither would
            // not be.
            let moved = rel(&o, &c.want_o).max(rel(&state, &c.want_state));
            let at = format!("{} layer {}", site.salt, site.layer);
            assert!(
                moved >= measured * 0.1,
                "{at} {what}: separation fell to {moved:e} from a measured {measured:e} — \
                 the docs quote this magnitude, so re-measure and move the constant rather \
                 than leaning on the much weaker bar `priced` applies"
            );
            priced(
                &at,
                what,
                moved,
                Bars {
                    tol: site.tol,
                    observed: KDA_OBSERVED_WORST,
                },
            );
        }
    });
}

/// **The recurrence at K3's real geometry, and in the three regimes the goldens do not
/// cover.**
///
/// Width: 96 heads of 128 against the tiny 4 of 32, which is where the state is 64 KB per
/// head and the two-pass shape is the reason this kernel exists. `heads = 1`, where a
/// grid-mapping error has nowhere to hide. And **mid-scale decay**, the regime the anchor
/// actually lacks. The `(4, 128)` case separates a head-count bug from a head-width one
/// (`k3:tests/k3_kernels.rs:2907`).
#[test]
fn the_recurrence_holds_at_real_widths_and_every_decay_regime() {
    for &(heads, head_dim, g_gain) in &[
        (96usize, 128usize, 12.0f32),
        (4, 128, 12.0),
        (1, 32, 12.0),
        (8, 128, 0.03),
    ] {
        let c = synthetic_kda_gain(heads, head_dim, 0x5A_11 + heads as u64, g_gain);
        let (o, state) = device_kda(&c);
        let (wo, ws) = host_kda(&c, KdaForm::Reference);
        // 10x the 5.465e-7 measured over these cases (96x128, `o`) — 2.4x looser than the
        // golden-backed sites and legitimately so: at head_dim 128 each `o` is a 128-term
        // reduction against the anchor's 32, and the device sums it in a different order
        // from the host's f64 walk. The state moves much less (9.67e-8 worst) because each
        // of its elements is two operations deep whatever the width.
        for (what, got, want) in [("o", &o, &wo), ("state", &state, &ws)] {
            let d = rel(got, want);
            assert!(d <= 5.5e-6, "heads={heads} dim={head_dim} {what}: {d:e}");
        }
        // `alpha` is `exp(lb·sigmoid(...))`, so it is bounded by construction — but only if
        // the bound multiplies the sigmoid. A clamped form would be bounded too; an
        // unbounded gate would not, and that is a NaN a hundred tokens into a decode rather
        // than a wrong number here.
        assert!(
            state.iter().all(|x| x.is_finite()) && o.iter().all(|x| x.is_finite()),
            "heads={heads} dim={head_dim}: the recurrence produced a non-finite value"
        );
        // **The regime this case claims to be in is ASSERTED, not hoped for** — a property
        // of the DRAW, and a future seed or width could re-degenerate it silently (the same
        // argument the AttnRes sweep makes with its `pmax` bound).
        let (lo, hi) = alpha_span(&c);
        if g_gain > 0.1 {
            assert!(
                hi >= 0.99 && lo <= (-4.0f32).exp(),
                "heads={heads} dim={head_dim}: alpha spans only [{lo:e}, {hi}], so this case \
                 does not reach the saturation it exists for"
            );
        } else {
            assert!(
                hi < 0.99 && lo > (-4.0f32).exp(),
                "heads={heads} dim={head_dim}: alpha spans [{lo:e}, {hi}], which saturates — \
                 the mid-scale case has become a second copy of the saturated ones"
            );
        }
    }
}

/// The decay range one case actually produces, so a test can assert the regime it claims.
fn alpha_span(c: &Kda) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, 0.0f32);
    for h in 0..c.heads {
        let a = c.a_log[h].exp();
        for i in 0..c.head_dim {
            let z = a * (c.g[h * c.head_dim + i] + c.dt_bias[h * c.head_dim + i]);
            let alpha = (c.lb / (1.0 + (-z).exp())).exp();
            lo = lo.min(alpha);
            hi = hi.max(alpha);
        }
    }
    (lo, hi)
}

/// **The delta rule's own regime: a state that already PREDICTS the value.**
///
/// `dv = beta·(v − u)` with `u = Sᵀk`, and the recurrence exists because `u` is the state's
/// prediction of `v` for the current key — so once a key direction has been written into
/// `S`, `v − u` is a difference of near-equal quantities and the relative error in `dv` is
/// amplified by `|v|/|v−u|`. Every other case here measures the operator where it is
/// numerically easiest: `synthetic_kda` draws the state i.i.d., uncorrelated with `kn`, and
/// the anchor's `initial_state` comes from an 8-token prefill of a randomly-initialised
/// model, so neither visits the cancellation. This seeds the state as the outer product one
/// step of the recurrence actually produces, `kn ⊗ (beta·v)`, and re-measures there. The
/// bound is LOOSER than the width sweep's, and that is the finding, not a concession
/// (`k3:tests/k3_kernels.rs:2980`).
#[test]
fn the_recurrence_holds_where_the_state_predicts_the_value() {
    let (heads, dim) = (8usize, 128usize);
    // The SAME case with an i.i.d. state, so the comparison isolates the cancellation rather
    // than conflating it with per-head-versus-whole-tensor scoring.
    let plain = synthetic_kda_gain(heads, dim, 0x5A_11_5D, 12.0);
    let mut c = synthetic_kda_gain(heads, dim, 0x5A_11_5D, 12.0);
    for h in 0..heads {
        let base = h * dim;
        // The kernel's own normalisation, so the seeded state is the one a first step leaves
        // behind.
        let n2: f32 = c.k[base..base + dim].iter().map(|x| x * x).sum();
        let inv = 1.0 / (n2 + 1e-6).sqrt();
        let beta = 1.0 / (1.0 + (-c.beta[h]).exp());
        for i in 0..dim {
            for j in 0..dim {
                c.state[base * dim + i * dim + j] = c.k[base + i] * inv * beta * c.v[base + j];
            }
        }
    }
    let worst = |c: &Kda| {
        let (o, state) = device_kda(c);
        let (wo, ws) = host_kda(c, KdaForm::Reference);
        let mut m = 0.0f32;
        for h in 0..c.heads {
            let (r, sr) = (h * dim..(h + 1) * dim, h * dim * dim..(h + 1) * dim * dim);
            m = m
                .max(rel(&o[r.clone()], &wo[r]))
                .max(rel(&state[sr.clone()], &ws[sr]));
        }
        m
    };
    // **Scored PER HEAD, and that is half of what this case is for.** `rel` divides by the
    // largest |value| in the WHOLE tensor, so a single head that lost every significant
    // digit to cancellation is diluted by the largest of 8 x 128 outputs. Per head, the
    // denominator is that head's own scale.
    let (structured, iid) = (worst(&c), worst(&plain));
    // 10x the 2.383e-6 measured. Against the i.i.d. state's 4.076e-7 at the same width and
    // the same per-head denominator, the cancellation costs **5.8x** — real, and far less
    // than the ~50x the amplification argument predicts, because `beta = sigmoid(b_proj)`
    // approaches 1 for only a minority of heads and `v - u = (1 - beta)v` is small only for
    // those (`k3:tests/k3_kernels.rs:3030`).
    assert!(
        structured <= 2.4e-5,
        "the predicting state disagrees by {structured:e} per head"
    );
    // The comparison is the finding, so it is asserted rather than described: a future
    // change that made the two equal would mean this case had stopped being the cancellation
    // case.
    assert!(
        structured > iid * 2.0,
        "the predicting state ({structured:e}) is no worse per head than the i.i.d. one \
         ({iid:e}), so this case is a second copy of the width sweep rather than the delta \
         rule's own regime"
    );
}

/// **The launcher refuses what it cannot compute.**
///
/// A `head_dim` that is not a power of two makes the L2-norm's halving reduction drop
/// elements — a slightly-wrong norm rather than a crash. A `lower_bound` outside fla's own
/// `-5 <= lb < 0` is worse: NaN makes every decay NaN, `0` removes the decay entirely, and a
/// positive bound makes the state GROW each step — a divergence that would present as fluent
/// wrong text a hundred tokens later (`k3:tests/k3_kernels.rs:3048`).
#[test]
fn the_recurrence_guards_its_width_and_its_gate_bound() {
    let mut c = synthetic_kda_gain(2, 32, 0x5A_11_60, 12.0);
    for lb in [
        0.0,
        -5.001,
        -6.0,
        1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        c.lb = lb;
        assert_guard(
            kda_launch(&c),
            Some(1006),
            &format!("lower_bound {lb} (fla's own range check is -5 <= lb < 0)"),
        );
    }
    // The two ends of that range are IN it, and the shipped value is the lower one — so a
    // guard written `> -5.0` would refuse the model. This is the half that keeps the seven
    // refusals above from being a guard that rejects everything.
    for lb in [-5.0, -0.5] {
        c.lb = lb;
        assert!(kda_launch(&c).is_ok(), "lower_bound {lb} was refused");
    }
    // 96 is not a power of two, and it is a plausible value rather than a silly one: it is
    // K3's HEAD COUNT, so transposing the launcher's two width arguments lands exactly here.
    let odd = synthetic_kda_gain(2, 96, 0x5A_11_60, 12.0);
    assert_guard(
        kda_launch(&odd),
        Some(1003),
        "head_dim 96 (the L2-norm reduction would drop elements)",
    );
    // Above the block-size ceiling — a power of two, so only 1002 can catch it.
    let wide = synthetic_kda_gain(1, 2048, 0x5A_11_60, 12.0);
    assert_guard(kda_launch(&wide), Some(1002), "head_dim 2048");
    // Zero heads is a launch of nothing that would otherwise return SUCCESS, and zero
    // `head_dim` is a zero-thread block, which the driver rejects with an error a caller
    // would then have to interpret.
    for (heads, head_dim) in [(0usize, 32usize), (2usize, 0usize)] {
        let mut z = synthetic_kda_gain(heads.max(1), head_dim.max(1), 0x5A_11_60, 12.0);
        z.heads = heads;
        z.head_dim = head_dim;
        assert_guard(
            kda_launch(&z),
            Some(1001),
            &format!("heads={heads} head_dim={head_dim}"),
        );
    }
}
