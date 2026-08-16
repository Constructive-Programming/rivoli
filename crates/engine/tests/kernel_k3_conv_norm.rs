//! **The KDA short convolution and the fused gated head norm on the device, scored against
//! the S2 anchor** — `short_conv_silu_f32` and `rmsnorm_gate_heads_f32`. Ported from
//! `k3:tests/k3_kernels.rs` items 5b and 5c (banner at :3099); shared spine in
//! `tests/k3/mod.rs`.
//!
//! The other two of the three boundaries fla fuses §4's ten KDA steps into. One file because
//! they share their guard family, their fixture shape (one weight the S1b capture set did
//! not carry — the fifth and sixth time the k3 port found that an input and an output do not
//! determine an operator when a weight sits between them), and their tolerance story: each
//! got its OWN bucket and its own anchor defect run (`KdaConvTapsReversed`,
//! `KdaGateBeforeNorm`), because both used to fall inside `kda_trunk`, which has a floor and
//! no ceiling — scoring them against a fixture tripwire alone is what the k3 anchor doc
//! explicitly tells its kernel suites not to do (`k3:tests/k3_kernels.rs:3107`).
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: no PR-triggered rocm CI arm, no GPU for this port. Two mutations, one per
//! kernel, in `kernels/recurrent.hip`:
//!
//! * **Reverse the conv's tap order** (read `w[ch*taps + (taps-1-j)]`).
//!   [`the_short_conv_matches_the_anchor_at_every_kda_layer`]'s `y` sites must go RED in the
//!   2e0 region (the anchor prices `KdaConvTapsReversed` at 2.012e0 at its weaker draw) while
//!   its `window` sites stay GREEN — the window is a shift, not arithmetic, and the tap order
//!   never touches it. A mutation that reddens the window has broken the shift instead.
//! * **Gate before the norm** in the fused kernel (multiply `o` by `sigmoid(g)` before the
//!   mean-square). [`the_gated_head_norm_matches_the_anchor`] must go RED in the 4e-1 region
//!   (`KdaGateBeforeNorm`, 4.365e-1) and
//!   [`the_gate_ordering_and_the_norm_convention_are_priced`] must stay GREEN — it perturbs
//!   the host oracle the same way, so the kernel now agrees with its defect arm.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_backend::hip::launch_rmsnorm_gate_heads_f32;
use rivoli_backend::hip::launch_short_conv_silu_f32;

mod common;
mod k3;

use k3::*;

/// The three convolutions a KDA layer runs.
const CONVS: [&str; 3] = ["q", "k", "v"];

/// The captured KDA layers — the recurrence suite's list, restated here because the two
/// binaries do not share a fixture module and three literals are below any factoring
/// threshold.
const KDA_LAYERS: [usize; 3] = [0, 1, 12];

/// The worst relative difference `short_conv_silu_f32` shows against the anchor, over both
/// draws, all three KDA layers and all three projections — 18 `y` sites, worst at draw 1
/// layer 0 `v`.
///
/// The 18 `window` sites are all **exactly 0**, and that is because the window comparison is
/// a ROUND-TRIP on this fixture's own reconstruction rather than a score against the
/// reference: `win_in` is built by right-shifting `want_win` and dropping its last slot, and
/// `cur` is read from that same last slot — so a kernel that left-shifts and appends `cur`
/// reproduces `want_win` bit-for-bit by construction, whatever the taps say. Stated because
/// "18 sites at exactly 0" reads like strong agreement and is not that. What it does still
/// prove: the shift's DIRECTION, its distance, and which slot the current token lands in —
/// get any of those wrong and the permutation stops being the inverse of the fixture's. What
/// it cannot see: the tap ORDER (never touches the window; `KdaConvTapsReversed` moves only
/// `y`) — that rests on `conv`'s assertion that the cache's last slot equals this token's
/// projection, and on `y` being scored at the 18 sites that carry the real signal
/// (`k3:tests/k3_kernels.rs:3120`).
const CONV_OBSERVED_WORST: f32 = 1.668e-7;

/// The worst `rmsnorm_gate_heads_f32` shows, same sweep — six sites, one per draw per layer.
const GATE_NORM_OBSERVED_WORST: f32 = 1.145e-7;

/// The f64 host oracles' own agreement with the anchor — separate constants because they are
/// separate measurements: the device number carries a reassociated reduction the f64 walk
/// does not. **Measured, the f64 oracle is the WORSE of the two at most sites** — computing
/// in f64 and rounding once gets closer to the true value and *further from the anchor*,
/// because the anchor is itself an fp32 run whose accumulation order the fp32 kernel happens
/// to track. So an oracle constant can never be derived from the device one, in either
/// direction; the conv pair coincides at 1.668e-7 only because both attain their max at the
/// same site where they agree to the bit (`k3:tests/k3_kernels.rs:3140`).
const CONV_ORACLE_WORST: f32 = 1.668e-7;
const GATE_NORM_ORACLE_WORST: f32 = 1.208e-7;

/// One convolution's boundary: the taps, the window this token convolves, and both outputs.
///
/// **The window is reconstructed from the RETURNED cache, and the reconstruction is the
/// fixture's one inference.** `ShortConvolution` is called with every argument by keyword,
/// so no pre-hook can see its input; what the golden holds is the cache AFTER the step,
/// whose last slot is this token's projection and whose earlier slots are the previous
/// window shifted left. So the window going IN is `[anything, out[0], out[1], out[2]]` — the
/// leading slot is shifted out and cannot affect anything, which
/// [`the_conv_window_discards_only_its_oldest_slot`] turns into an assertion rather than
/// leaving as a claim (`k3:tests/k3_kernels.rs:3155`).
struct Conv {
    w: Vec<f32>,
    taps: usize,
    cur: Vec<f32>,
    win_in: Vec<f32>,
    want_y: Vec<f32>,
    want_win: Vec<f32>,
}

/// The sentinel that goes in the slot the shift discards. NaN would be caught by `rel`'s
/// finiteness check even if the kernel leaked it, but a large finite value is the stronger
/// choice: it would corrupt the output arithmetically rather than making it non-finite, so a
/// kernel that convolved the pre-shift window fails on the NUMBERS
/// (`k3:tests/k3_kernels.rs:3172`).
const CONV_DISCARDED: f32 = -1234.5;

fn conv(g: &GoldenSet, layer: usize, which: &str) -> Conv {
    let m = format!("model.layers.{layer}.self_attn.{which}_conv1d");
    let (ws, w) = float(g, &format!("{m}.weight"));
    let (channels, taps) = (ws[0], ws[2]);
    let (cs, want_win) = float(g, &format!("{m}.1"));
    assert_eq!(
        cs,
        vec![1, channels, taps],
        "{m}: the cache is [1][channels][taps]"
    );
    let (ys, want_y) = float(g, &format!("{m}.0"));
    assert_eq!(ys, vec![1, 1, channels], "{m}: one token of `channels`");
    // The current token is the cache's LAST slot, and the reference's own projection says so
    // — this is the equality that lets the window be reconstructed at all.
    let cur: Vec<f32> = (0..channels)
        .map(|c| want_win[c * taps + taps - 1])
        .collect();
    let (_, proj) = float(g, &format!("model.layers.{layer}.self_attn.{which}_proj"));
    assert_eq!(
        cur, proj,
        "{m}: the cache's last slot must be this token's projection"
    );
    let win_in: Vec<f32> = (0..channels)
        .flat_map(|c| {
            std::iter::once(CONV_DISCARDED)
                .chain((0..taps - 1).map(move |j| want_win[c * taps + j]))
        })
        .collect();
    Conv {
        w: w.to_vec(),
        taps,
        cur,
        win_in,
        want_y: want_y.to_vec(),
        want_win: want_win.to_vec(),
    }
}

/// How the convolution is read. **The variants are the defects**, exactly as `GateForm`'s
/// and the recurrence's `KdaForm`'s are: a defect run is the correct oracle with one thing
/// changed, and an enum makes a call site say WHICH reference it is talking about where two
/// bare bools said `(true, false)` (`k3:tests/k3_kernels.rs:3217`).
#[derive(Clone, Copy, PartialEq)]
enum ConvForm {
    Reference,
    /// `KdaConvTapsReversed`: the filter read newest→oldest — the same causal window, the
    /// same weight shape, a different function.
    ReversedTaps,
    /// The fused SiLU dropped — no anchor defect covers it; priced here.
    NoFusedSilu,
}

impl ConvForm {
    /// Which tap multiplies window slot `j` of `taps`.
    fn tap(self, j: usize, taps: usize) -> usize {
        [j, taps - 1 - j][usize::from(self == ConvForm::ReversedTaps)]
    }

    /// The output activation on the accumulated value.
    fn finish(self, acc: f64) -> f32 {
        if self == ConvForm::NoFusedSilu {
            acc as f32
        } else {
            (acc * (1.0 / (1.0 + (-acc).exp()))) as f32
        }
    }
}

/// §4 step 2 in f64: shift the window, convolve oldest→newest, SiLU the accumulator.
fn host_conv(c: &Conv, form: ConvForm) -> (Vec<f32>, Vec<f32>) {
    let taps = c.taps;
    let channels = c.cur.len();
    let mut win: Vec<f32> = c.win_in.clone();
    let mut y = vec![0f32; channels];
    for ch in 0..channels {
        let wn = &mut win[ch * taps..(ch + 1) * taps];
        wn.rotate_left(1);
        wn[taps - 1] = c.cur[ch];
        let mut acc = 0f64;
        for (j, &slot) in wn.iter().enumerate() {
            acc += f64::from(c.w[ch * taps + form.tap(j, taps)]) * f64::from(slot);
        }
        y[ch] = form.finish(acc);
    }
    (y, win)
}

/// One launch, returning the launcher's own `Result` and both of its outputs.
///
/// `(channels, taps)` are parameters rather than read off the case, because the guard test
/// needs to pass values the buffers do not agree with — and both paths going through one
/// launch is what makes that test exercise the entry point callers use. `win_in` is separate
/// from the case for the discarded-slot test, which launches the SAME case over a second
/// window (`k3:tests/k3_kernels.rs:3243`).
fn conv_launch(
    c: &Conv,
    win_in: &[f32],
    (channels, taps): (usize, usize),
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let s = stream();
    let (cb, wb) = (dev(&f32b(&c.cur)), dev(&f32b(&c.w)));
    let mut winb = dev(&f32b(win_in));
    let mut ob = zeros(c.cur.len().max(1) * 4);
    let (wp, op) = (winb.ptr_mut() as *mut f32, ob.ptr_mut() as *mut f32);
    // SAFETY: `cur`/`out` are `channels` f32, `w` and `win` are `channels x taps`, all
    // distinct allocations live until the stream completes (`back` synchronises). The guard
    // cases are refused before any launch happens.
    unsafe {
        launch_short_conv_silu_f32(
            cb.ptr() as *const f32,
            wb.ptr() as *const f32,
            channels,
            taps,
            wp,
            op,
            s.raw(),
        )
    }?;
    Ok((f32v(&back(&ob)), f32v(&back(&winb))))
}

fn device_conv(c: &Conv) -> (Vec<f32>, Vec<f32>) {
    ok(
        conv_launch(c, &c.win_in, (c.cur.len(), c.taps)),
        "short_conv_silu_f32",
    )
}

fn for_each_conv(mut f: impl FnMut(&str, usize, &str, f32, Conv)) {
    let tol = tolerance::rel_tolerance("kda_conv");
    for (salt, bytes) in GOLDENS {
        let g = load(bytes);
        for layer in KDA_LAYERS {
            for which in CONVS {
                f(salt, layer, which, tol, conv(&g, layer, which));
            }
        }
    }
}

/// Score one conv's two outputs against its golden. Two callers — the kernel and the host
/// oracle — and the shared body is what stops them disagreeing about scoring the WINDOW as
/// well as `y`; the k3 tree factored the same pair for the same reason
/// (`k3:tests/k3_kernels.rs:338`).
fn score_conv(at: &str, b: Bars, got: (&[f32], &[f32]), c: &Conv) {
    score_all(
        at,
        b,
        &[("y", got.0, &c.want_y), ("window", got.1, &c.want_win)],
    );
}

/// The one scored buffer of the gated norm, named once — the same two-caller argument as
/// [`score_conv`] (`k3:tests/k3_kernels.rs:3516`).
fn score_gate_norm(at: &str, b: Bars, got: &[f32], c: &GateNorm) {
    score_all(at, b, &[("out", got, &c.want)]);
}

/// **The kernel reproduces every short convolution the anchor captured — output AND
/// window.**
#[test]
fn the_short_conv_matches_the_anchor_at_every_kda_layer() {
    for_each_conv(|salt, layer, which, tol, c| {
        let (y, win) = device_conv(&c);
        let at = format!("{salt} layer {layer} {which}_conv1d");
        let bars = Bars {
            tol,
            observed: CONV_OBSERVED_WORST,
        };
        score_conv(&at, bars, (&y, &win), &c);
    });
}

/// **The taps run oldest→newest and the SiLU is on the accumulator — both priced.**
///
/// Neither is visible in the shape: the weight is `[channels][1][taps]` under either
/// convention, and a reversed filter is still a causal convolution over the same window.
/// `KdaConvTapsReversed` is the anchor's own run for the first; the second has no defect run
/// and is priced here (`k3:tests/k3_kernels.rs:3313`).
#[test]
fn the_conv_tap_order_and_its_fused_silu_are_priced() {
    for_each_conv(|salt, layer, which, tol, c| {
        let at = format!("{salt} layer {layer} {which}");
        for (what, form) in [
            (
                "reversed taps (--defect KdaConvTapsReversed)",
                ConvForm::ReversedTaps,
            ),
            ("no fused SiLU (no anchor defect)", ConvForm::NoFusedSilu),
        ] {
            let (y, _) = host_conv(&c, form);
            let moved = rel(&y, &c.want_y);
            priced(
                &at,
                what,
                moved,
                Bars {
                    tol,
                    observed: CONV_OBSERVED_WORST,
                },
            );
        }
    });
}

/// **The window's oldest slot is discarded, and nothing downstream can see it.**
///
/// The fixture reconstructs the incoming window from the RETURNED cache, so its leading slot
/// is a sentinel rather than a captured value. That is only sound if the shift really
/// discards it — a kernel that convolved the PRE-shift window would use it, and this is what
/// makes the reconstruction an assertion instead of an assumption
/// (`k3:tests/k3_kernels.rs:3337`).
#[test]
fn the_conv_window_discards_only_its_oldest_slot() {
    for_each_conv(|salt, layer, which, _tol, c| {
        // A different sentinel in the same slot. The window is copied rather than the case
        // moved, so both launches see identical taps and identical `cur`.
        let mut other = c.win_in.clone();
        for ch in 0..c.cur.len() {
            other[ch * c.taps] = 9876.5;
        }
        let (a, wa) = ok(
            conv_launch(&c, &other, (c.cur.len(), c.taps)),
            "short_conv_silu_f32",
        );
        let (b, wb) = device_conv(&c);
        assert_eq!(
            a, b,
            "{salt} layer {layer} {which}: the discarded slot changed the output"
        );
        assert_eq!(
            wa, wb,
            "{salt} layer {layer} {which}: it survived into the window"
        );
    });
}

/// One gated head norm's boundary: the recurrence's output, the gate, the weight, the
/// result.
struct GateNorm {
    o: Vec<f32>,
    gate: Vec<f32>,
    w: Vec<f32>,
    heads: usize,
    head_dim: usize,
    want: Vec<f32>,
}

fn gate_norm(g: &GoldenSet, layer: usize) -> GateNorm {
    let m = format!("model.layers.{layer}.self_attn");
    let (is, o) = float(g, &format!("{m}.o_norm.in"));
    let (heads, head_dim) = (is[2], is[3]);
    let (_, gate) = float(g, &format!("{m}.g_proj"));
    let (ws, w) = float(g, &format!("{m}.o_norm.weight"));
    assert_eq!(
        ws,
        vec![head_dim],
        "{m}: the norm weight is [head_dim], shared across heads"
    );
    let (os, want) = float(g, &format!("{m}.o_norm"));
    assert_eq!(os, is, "{m}: the fused norm is width-preserving");
    // The free cross-check the k3 anchor doc predicted: the module hook's input and the
    // operator wrapper's output are the same tensor under two names, so they must agree bit
    // for bit.
    let (_, kda_o) = float(
        g,
        &format!("model.layers.{layer}.kda.fused_recurrent_kda.out.o"),
    );
    assert_eq!(
        o, kda_o,
        "{m}: `o_norm.in` must BE the recurrence's `out.o`"
    );
    GateNorm {
        o: o.to_vec(),
        gate: gate.to_vec(),
        w: w.to_vec(),
        heads,
        head_dim,
        want: want.to_vec(),
    }
}

/// How the norm and the gate compose. **Four variants and each is a documented
/// alternative** (`k3:tests/k3_kernels.rs:3411`).
#[derive(Clone, Copy, PartialEq)]
enum GateForm {
    Reference,
    /// `KdaGateBeforeNorm`: gate first, so the RMS is taken over the GATED values.
    GateFirst,
    /// `silu(g)` where the module takes `activation='sigmoid'` — fla accepts both spellings.
    SiluGate,
    /// `eps` on the SUM of squares, which is the recurrence's L2 convention one operator
    /// over — in the same layer, a few hundred lines away in the same kernel file.
    EpsOnSum,
}

impl GateForm {
    /// The gate activation — `sigmoid`, or `SiluGate`'s spelling. On the FORM, like the
    /// recurrence suite's `KdaForm` helpers, so the norm body below carries no nested
    /// conditionals: each variant's ONE flag lives beside the arithmetic it perturbs.
    fn sig(self, x: f32) -> f64 {
        let x = f64::from(x);
        let s = 1.0 / (1.0 + (-x).exp());
        [s, x * s][usize::from(self == GateForm::SiluGate)]
    }
}

/// §4 steps 8-9 in f64. `GateFirst` rides as a two-element index — mixing the gate into
/// `pre` (so the RMS divides by the GATED values) or into the store, never both — and
/// `EpsOnSum` as a two-element denominator select, so the head loop is branch-free.
fn host_gate_norm(c: &GateNorm, eps: f32, form: GateForm) -> Vec<f32> {
    let d = c.head_dim;
    let mut out = vec![0f32; c.o.len()];
    let gate_first = usize::from(form == GateForm::GateFirst);
    let eps_on_sum = usize::from(form == GateForm::EpsOnSum);
    for h in 0..c.heads {
        let r = h * d..(h + 1) * d;
        // Gate first changes what the norm divides by; that is why it is not a permutation.
        let pre: Vec<f64> = r
            .clone()
            .map(|i| f64::from(c.o[i]) * [1.0, form.sig(c.gate[i])][gate_first])
            .collect();
        let sq: f64 = pre.iter().map(|x| x * x).sum();
        let denom = [sq / d as f64 + f64::from(eps), sq + f64::from(eps)][eps_on_sum];
        let inv = 1.0 / denom.sqrt();
        for (k, i) in r.enumerate() {
            let gate = [form.sig(c.gate[i]), 1.0][gate_first];
            out[i] = (pre[k] * inv * f64::from(c.w[k]) * gate) as f32;
        }
    }
    out
}

/// One launch, `(heads, head_dim)` and `eps` as parameters so the guard test drives the
/// same entry point with values the buffers do not agree with.
fn gate_norm_launch(
    c: &GateNorm,
    (heads, head_dim): (usize, usize),
    eps: f32,
) -> anyhow::Result<Vec<f32>> {
    let s = stream();
    let [ob, gb, wb] = [&c.o, &c.gate, &c.w].map(|x| dev(&f32b(x)));
    let mut out = zeros(c.o.len().max(1) * 4);
    // SAFETY: `o`, `gate` and `out` are `heads x head_dim` f32 and `weight` is `head_dim`,
    // all distinct allocations live until the stream completes (`back` synchronises). The
    // guarded dims are refused before any launch.
    unsafe {
        launch_rmsnorm_gate_heads_f32(
            ob.ptr() as *const f32,
            gb.ptr() as *const f32,
            wb.ptr() as *const f32,
            heads,
            head_dim,
            eps,
            out.ptr_mut() as *mut f32,
            s.raw(),
        )
    }?;
    Ok(f32v(&back(&out)))
}

fn device_gate_norm(c: &GateNorm, eps: f32) -> Vec<f32> {
    ok(
        gate_norm_launch(c, (c.heads, c.head_dim), eps),
        "rmsnorm_gate_heads_f32",
    )
}

fn for_each_gate_norm(mut f: impl FnMut(&str, usize, f32, f32, GateNorm)) {
    let tol = tolerance::rel_tolerance("kda_gate_norm");
    for (salt, bytes) in GOLDENS {
        let g = load(bytes);
        // `FusedRMSNormGated(head_dim, eps=config.rms_norm_eps, activation='sigmoid')` — the
        // reference passes the config's value explicitly. fla's own default is ALSO 1e-5, so
        // reading either would score green here; this reads the config because that is what
        // the reference passes, and the coincidence is exactly what would hide a wrong
        // source (`k3:tests/k3_kernels.rs:3505`).
        let eps = eps(&g);
        for layer in KDA_LAYERS {
            f(salt, layer, tol, eps, gate_norm(&g, layer));
        }
    }
}

/// **The kernel reproduces the fused gated norm at every KDA layer.**
#[test]
fn the_gated_head_norm_matches_the_anchor() {
    for_each_gate_norm(|salt, layer, tol, eps, c| {
        let got = device_gate_norm(&c, eps);
        let at = format!("{salt} layer {layer}");
        score_gate_norm(
            &at,
            Bars {
                tol,
                observed: GATE_NORM_OBSERVED_WORST,
            },
            &got,
            &c,
        );
    });
}

/// **Norm THEN gate, `sigmoid` not `silu`, and `eps` on the mean — all three priced.**
///
/// The first is trap 10 and has the anchor's own run (`KdaGateBeforeNorm`); it is not a
/// permutation of two elementwise multiplies, because gating first changes the RMS the norm
/// divides by. The other two have no defect run: `silu` is the other activation fla accepts
/// at this call site, and eps on the SUM is the convention the recurrence's L2 norm uses in
/// the same layer (`k3:tests/k3_kernels.rs:3533`).
#[test]
fn the_gate_ordering_and_the_norm_convention_are_priced() {
    for_each_gate_norm(|salt, layer, tol, eps, c| {
        let at = format!("{salt} layer {layer}");
        for (what, form) in [
            (
                "gating before the norm (--defect KdaGateBeforeNorm)",
                GateForm::GateFirst,
            ),
            (
                "silu instead of sigmoid (no anchor defect)",
                GateForm::SiluGate,
            ),
            ("eps on the SUM (no anchor defect)", GateForm::EpsOnSum),
        ] {
            let defect = rel(&host_gate_norm(&c, eps, form), &c.want);
            let bars = Bars {
                tol,
                observed: GATE_NORM_OBSERVED_WORST,
            };
            priced(&at, what, defect, bars);
        }
    });
}

/// **Both host oracles agree with the anchor, which is what lets their variants mean
/// anything.**
///
/// The red-proofs in this file perturb the ORACLE and assert the perturbation moves the
/// result. That argument is empty unless the unperturbed oracle agrees to the same order the
/// kernel does — the gap a review found in the recurrence's equivalent, where the oracle
/// underwriting five red-proofs was monitored at 2,780x slack. One test for both operators:
/// it is the same argument twice (`k3:tests/k3_kernels.rs:3561`).
#[test]
fn the_conv_and_gated_norm_host_oracles_agree_with_the_anchor() {
    for_each_conv(|salt, layer, which, tol, c| {
        let (y, win) = host_conv(&c, ConvForm::Reference);
        let oat = format!("oracle {salt} layer {layer} {which}_conv1d");
        score_conv(
            &oat,
            Bars {
                tol,
                observed: CONV_ORACLE_WORST,
            },
            (&y, &win),
            &c,
        );
    });
    for_each_gate_norm(|salt, layer, tol, eps, c| {
        let got = host_gate_norm(&c, eps, GateForm::Reference);
        let oat = format!("oracle {salt} layer {layer}");
        let oracle_bars = Bars {
            tol,
            observed: GATE_NORM_ORACLE_WORST,
        };
        score_gate_norm(&oat, oracle_bars, &got, &c);
    });
}

/// One synthetic case per operator, at a width the caller picks. `want` is empty in both:
/// these are scored against the f64 host oracles, not against the golden, so there is
/// nothing to carry.
/// A seeded draw closure — ONE stream per synthetic case, so its operands cannot coincide
/// the way independently salted draws can. Shared by both constructors below, whose opening
/// two lines were otherwise token-identical.
fn drawer(seed: u64) -> impl FnMut(usize) -> Vec<f32> {
    let mut r = Lcg(seed);
    move |len| (0..len).map(|_| r.f()).collect()
}

fn synthetic_conv((channels, taps): (usize, usize), seed: u64) -> Conv {
    let mut draw = drawer(seed);
    Conv {
        w: draw(channels * taps),
        taps,
        cur: draw(channels),
        win_in: draw(channels * taps),
        want_y: Vec::new(),
        want_win: Vec::new(),
    }
}

fn synthetic_gate_norm((heads, head_dim): (usize, usize), seed: u64) -> GateNorm {
    let mut draw = drawer(seed);
    GateNorm {
        o: draw(heads * head_dim),
        gate: draw(heads * head_dim),
        w: draw(head_dim),
        heads,
        head_dim,
        want: Vec::new(),
    }
}

/// **Both operators at K3's real geometry, which the goldens structurally cannot reach.**
///
/// **The conv's grid is never exercised by the goldens at all**: 128 channels is ONE block
/// of 256, so `blockIdx.x` is always 0 and the `c >= channels` tail is the only thing under
/// test. The real width is 12,288 channels — 48 blocks — and a `blockIdx.x * blockDim.x`
/// mistake is invisible until then; 257 is the awkward case a round number hides, two blocks
/// whose second has one live thread. **The gated norm's reduction gets two more halvings**:
/// `block_sum_lds` at `blockDim.x = 128` runs a 7-level ladder against the anchor's 5 and
/// takes 4x the LDS, and the launcher computes that LDS size itself — so a kernel indexing
/// the reduction buffer beyond what the launcher allocated shows up here first. Scored
/// against the same f64 host oracles the golden-backed tests use, which is what makes this a
/// second measurement rather than a second copy of the first (`k3:tests/k3_kernels.rs:3614`).
#[test]
fn the_conv_and_gated_norm_hold_at_real_widths() {
    // `q`/`k`/`v` are each `num_heads * head_dim` = 96 x 128 channels at the real config,
    // and `taps` is `short_conv_kernel_size` = 4. The two small cases share `taps` with the
    // real one, so a failure separates the grid from the window arithmetic.
    for &(channels, taps) in &[(12288usize, 4usize), (257, 4), (1, 2), (12288, 16)] {
        let c = synthetic_conv((channels, taps), 0xC0_11 + channels as u64 + taps as u64);
        let (y, win) = device_conv(&c);
        let (wy, ww) = host_conv(&c, ConvForm::Reference);
        // 10x the 2.822e-7 measured over these four cases, whose worst is `y` at 12288x16 —
        // the deepest window, as expected: a 16-tap dot is four times the reduction depth.
        // The 4-tap cases sit at 1.2-1.5e-7, i.e. at the 1.668e-7 the golden-backed sites
        // measure, so the extra slack is bought by the tap count and by nothing else. `1x2`
        // is EXACTLY 0: two terms leave f32 and f64 nothing to disagree about. The window is
        // exact at every case — it is a copy, not arithmetic — so it gets no slack at all,
        // and a nonzero there is a shift bug rather than rounding.
        let dy = rel(&y, &wy);
        assert!(dy <= 2.9e-6, "conv {channels}x{taps} y: {dy:e}");
        assert_eq!(win, ww, "conv {channels}x{taps}: the window is not a copy");
        assert!(
            y.iter().all(|x| x.is_finite()),
            "conv {channels}x{taps}: non-finite output"
        );
    }
    // `head_dim` 128 x 96 heads is the real geometry; `(1, 128)` isolates the head count
    // from the width, and `(96, 2)` is the shallowest ladder the power-of-two guard admits.
    for &(heads, head_dim) in &[(96usize, 128usize), (1, 128), (96, 2), (4, 1024)] {
        let c = synthetic_gate_norm((heads, head_dim), 0x6A_7E + heads as u64 + head_dim as u64);
        let got = device_gate_norm(&c, 1e-5);
        let want = host_gate_norm(&c, 1e-5, GateForm::Reference);
        // 10x the 1.913e-7 measured over these four cases. The worst is the REAL geometry,
        // 96x128 — not the 1024-wide case, which reaches 1.543e-7 on a ladder two levels
        // deeper. So the separation is not simply growing with reduction depth here the way
        // the conv's does with tap count, and a comment predicting otherwise was wrong
        // before it was measured (`k3:tests/k3_kernels.rs:3663`).
        let d = rel(&got, &want);
        assert!(d <= 2.0e-6, "gate norm {heads}x{head_dim}: {d:e}");
        assert!(
            got.iter().all(|x| x.is_finite()),
            "gate norm {heads}x{head_dim}: non-finite output"
        );
    }
}

/// **Both launchers refuse what they cannot compute, by CODE.**
#[test]
fn the_conv_and_gated_norm_guard_their_shapes() {
    // A plausible little case whose `want` is empty — the guards are asked about DIMS, and
    // every rejected launch returns before a byte is read.
    let cv = Conv {
        w: vec![0.5f32; 8 * 4],
        taps: 4,
        cur: vec![1.0f32; 8],
        win_in: vec![0.0f32; 8 * 4],
        want_y: Vec::new(),
        want_win: Vec::new(),
    };
    let launch_conv = |dims: (usize, usize)| conv_launch(&cv, &cv.win_in, dims);
    assert_guard(launch_conv((0, 4)), Some(1001), "channels 0");
    // `taps = 1` is a per-channel scale, not a convolution — the shape a caller reaches by
    // reading `short_conv_kernel_size` as the history length rather than the window.
    assert_guard(launch_conv((8, 1)), Some(1002), "taps 1");
    assert_guard(launch_conv((8, 17)), Some(1002), "taps 17");
    assert!(launch_conv((8, 4)).is_ok(), "the shipped shape was refused");

    let c = GateNorm {
        o: vec![0.5; 2 * 32],
        gate: vec![0.25; 2 * 32],
        w: vec![1.0; 32],
        heads: 2,
        head_dim: 32,
        want: Vec::new(),
    };
    let launch_norm = |dims: (usize, usize), eps: f32| gate_norm_launch(&c, dims, eps);
    assert_guard(launch_norm((0, 32), 1e-5), Some(1001), "heads 0");
    assert_guard(launch_norm((2, 0), 1e-5), Some(1001), "head_dim 0");
    // 2048 is a power of two, so it passes 1003 and only the block-size bound stops it.
    // Without this case nothing reached 1002 at all: every other refusal here fires an
    // earlier guard, and a test that only asserts `is_err()` cannot tell which one did.
    assert_guard(launch_norm((2, 2048), 1e-5), Some(1002), "head_dim 2048");
    // 96 is K3's HEAD COUNT, so a transposed argument pair lands exactly here — the same
    // case the recurrence's guard test uses, for the same reason.
    assert_guard(launch_norm((2, 96), 1e-5), Some(1003), "head_dim 96");
    for bad in [-1.0f32, f32::NAN, f32::INFINITY] {
        assert_guard(launch_norm((2, 32), bad), Some(1006), &format!("eps {bad}"));
    }
    // Zero eps is LEGAL — it is what an exact RMSNorm is — so the refusals above are not
    // "rejects everything".
    assert!(launch_norm((2, 32), 0.0).is_ok(), "eps 0 was refused");
    assert!(
        launch_norm((2, 32), 1e-5).is_ok(),
        "the shipped eps was refused"
    );
}
