//! **Kimi-K3's gated MLA core on the device, scored against the S2 anchor** — `mha_attend`
//! over the reference's own q/k/v, plus the output gate through the EXISTING `sigmoid_gate`.
//! Ported from `k3:tests/k3_kernels.rs` item 2 (banner at :726); shared spine in
//! `tests/k3/mod.rs`.
//!
//! The attend and the gate are separate kernels because §5's order is attend-then-gate with
//! NO norm between them — trap 10's other half; KDA norms first, and `kernel_k3_conv_norm.rs`
//! is where that half becomes checkable. **The k3 tree's out-of-place `sigmoid_gate` was
//! deliberately NOT ported**: this engine's is in-place and stream-taking, shared with Muse
//! Glimmer's identical seam (`kernels/fwd.hip` and the `launch_sigmoid_gate` doc record the
//! decision), so the gate fixture here is the one-buffer adaptation — `x` is uploaded, gated
//! in place, and read back, where the reference wrote to a third buffer.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: no PR-triggered rocm CI arm, no GPU for this port. Two mutations in
//! `kernels/attn.hip`'s `rivoli_mha_attend`, each chosen because the GOLDEN-backed tests
//! cannot see it — that blindness is measured, not assumed (`k3:tests/k3_kernels.rs:1104`):
//!
//! * **Drop `mask[s]` from the score.** Every golden-backed test above must stay GREEN — the
//!   decode captures' masks are ALL ZERO, since at a single decode step causality masks
//!   nothing — and [`mha_attend_at_real_widths_masks_and_magnitudes`]'s `masked` cells must
//!   go RED (the second half of the keys is forbidden there and the oracle knows it).
//! * **Remove the max-subtraction from the softmax.** The golden magnitudes never overflow
//!   `expf`, so again every anchor-backed test stays green; the `gain = 40` cells must go
//!   red on the `is_finite` assert — scores reach ~1e2 where `expf` overflows to inf.
//!
//! A mutation that reddens the anchor-backed tests instead has changed the arithmetic
//! somewhere other than the gap the sweep exists to cover.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_backend::hip::{launch_mha_attend, launch_sigmoid_gate};

mod common;
mod k3;

use k3::*;

/// The captured layers the real map makes MLA — zero-based 3, 91 and 92.
///
/// **91 and 92 are ADJACENT**, which the every-fourth pattern does not predict: 93 layers do
/// not divide by 4, so the map ends with two MLA layers in a row. The anchor gate pins the
/// partition itself; this list exists so a fixture that silently stopped covering one of
/// them is visible (`k3:tests/k3_kernels.rs:732`).
const MLA_LAYERS: [usize; 3] = [3, 91, 92];

/// The worst the real-width sweep measures against its f64 oracle over every cell — two
/// correct implementations disagreeing, three orders under the 4.10e-4 operator tolerance
/// (`k3:tests/k3_kernels.rs:325`).
const MLA_SWEEP_WORST: f32 = 3.634e-7;

/// The four widths an attend is shaped by. One type because they always travel together —
/// the k3 tree's jscpd rejected them appearing once as fields and once as parameters
/// (`k3:tests/k3_kernels.rs:739`).
#[derive(Clone, Copy)]
struct Dims {
    heads: usize,
    kv: usize,
    d: usize,
    dv: usize,
}

/// One MLA layer's attention boundary, as the reference computed it.
struct Attend {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    mask: Vec<f32>,
    scale: f32,
    out: Vec<f32>,
    probs: Vec<f32>,
    dims: Dims,
    /// The two halves of `d`, read from the golden's own `tiny_config` — two tests need
    /// them (`k3:tests/k3_kernels.rs:760`).
    nope: usize,
    rope: usize,
    /// The output gate and the value that reached `o_proj` — the two halves of trap 10. On
    /// the struct because `Attend` claims to be the layer's whole boundary
    /// (`k3:tests/k3_kernels.rs:765`).
    gate: Vec<f32>,
    gated: Vec<f32>,
}

fn attend(g: &GoldenSet, layer: usize) -> Attend {
    let tag = format!("model.layers.{layer}.self_attn.attend");
    let (qs, q) = float(g, &format!("{tag}.in.q"));
    let (ks, k) = float(g, &format!("{tag}.in.k"));
    let (vs, v) = float(g, &format!("{tag}.in.v"));
    let (ms, mask) = float(g, &format!("{tag}.in.mask"));
    let (_, scale) = float(g, &format!("{tag}.in.scaling"));
    let (os, out) = float(g, &format!("{tag}.out"));
    let (ps, probs) = float(g, &format!("{tag}.probs"));

    // `[b, heads, q_len, d]`. The decode step is one query row, and the kernel's contract is
    // that row — asserted rather than assumed, because a q_len > 1 golden would silently make
    // every comparison below read the first row of a stack.
    assert_eq!(qs[0], 1, "batch");
    assert_eq!(qs[2], 1, "the decode fixture is one query row");
    let (heads, d) = (qs[1], qs[3]);
    let (kv, dv) = (ks[2], vs[3]);
    // `repeat_kv` runs INSIDE the reference's attention, so what is captured is
    // pre-broadcast. The kernel takes fully expanded per-head K/V, which is what K3 caches
    // (§5 stores the expanded k/v deliberately). Equal head counts here means the two agree;
    // unequal would mean the kernel is being handed something it does not implement, and
    // passing anyway would be the bug (`k3:tests/k3_kernels.rs:789`).
    assert_eq!(ks[1], heads, "K head count must already be per-head");
    assert_eq!(vs[1], heads, "V head count must already be per-head");
    assert_eq!(
        ks[3], d,
        "K width must equal Q width — the rope dims ride in both"
    );
    assert_eq!(vs[2], kv, "V must have as many rows as K");
    assert_eq!(
        ms,
        &[1, 1, 1, kv],
        "mask is one additive row per key position"
    );
    assert_eq!(os, &[1, 1, heads, dv], "out is [b, q_len, heads, dv]");
    assert_eq!(ps, &[1, heads, 1, kv], "probs is [b, heads, q_len, kv]");

    let cfg = tiny(g);
    let nope = cfg["qk_nope_head_dim"].as_u64().unwrap() as usize;
    let rope = cfg["qk_rope_head_dim"].as_u64().unwrap() as usize;
    assert_eq!(nope + rope, d, "the head width is qk_nope + qk_rope");

    let (_, gate) = float(g, &format!("model.layers.{layer}.self_attn.g_proj"));
    let (_, gated) = float(
        g,
        &format!("model.layers.{layer}.self_attn.o_proj.in_gated"),
    );
    assert_eq!(gate.len(), heads * dv, "one gate value per output element");
    assert_eq!(
        gated.len(),
        gate.len(),
        "the gated value is the gate's shape"
    );

    Attend {
        q: q.to_vec(),
        k: k.to_vec(),
        v: v.to_vec(),
        mask: mask.to_vec(),
        scale: scale[0],
        out: out.to_vec(),
        probs: probs.to_vec(),
        dims: Dims { heads, kv, d, dv },
        nope,
        rope,
        gate: gate.to_vec(),
        gated: gated.to_vec(),
    }
}

/// One attention call's inputs. Bundled because [`device_attend`] and [`host_attn`] take
/// exactly the same six things — quantities that always travel together are one type
/// (`k3:tests/k3_kernels.rs:846`).
struct AttnIn<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    mask: &'a [f32],
    dims: Dims,
    scale: f32,
}

impl Attend {
    /// This layer's boundary as an attention call — one constructor, so the two tests that
    /// launch over it cannot disagree about which buffers go where.
    fn inputs(&self) -> AttnIn<'_> {
        AttnIn {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            mask: &self.mask,
            dims: self.dims,
            scale: self.scale,
        }
    }
}

/// One launch, returning the launcher's own `Result` so the guard test drives the entry
/// point callers use.
fn attend_launch(a: &AttnIn, dims: Dims) -> anyhow::Result<Vec<f32>> {
    let Dims { heads, kv, d, dv } = dims;
    let (qb, kb, vb, mb) = (
        dev(&f32b(a.q)),
        dev(&f32b(a.k)),
        dev(&f32b(a.v)),
        dev(&f32b(a.mask)),
    );
    let mut ob = zeros((heads * dv).max(1) * 4);
    // SAFETY: `q` is `heads·d` f32, `k` is `heads·kv·d`, `v` is `heads·kv·dv`, `mask` is
    // `kv`, and `out` is `heads·dv` — all live for the call and mutually non-aliasing; the
    // guarded cases return before any launch, and `back` synchronises before drops.
    unsafe {
        launch_mha_attend(
            qb.ptr() as *const f32,
            kb.ptr() as *const f32,
            vb.ptr() as *const f32,
            mb.ptr() as *const f32,
            heads,
            kv,
            d,
            dv,
            a.scale,
            ob.ptr_mut() as *mut f32,
        )
    }?;
    Ok(f32v(&back(&ob)))
}

fn device_attend(a: &AttnIn) -> Vec<f32> {
    ok(attend_launch(a, a.dims), "mha_attend")
}

/// Every (draw, MLA layer) pair with its boundary assembled.
fn for_each_mla(mut f: impl FnMut(&str, usize, Attend)) {
    for (salt, bytes) in GOLDENS {
        let g = load(bytes);
        for layer in MLA_LAYERS {
            f(salt, layer, attend(&g, layer));
        }
    }
}

/// The reference's attention, in `f64`, over the first `width` dims of each head.
///
/// `width` is the whole point: `d` reproduces the operator, `nope` reproduces §5's silent
/// bug of dropping the unrotated rope dims from the score. Writing it once makes
/// [`the_unrotated_rope_dims_are_still_scored`] self-evidently "the same oracle at a
/// narrower width", which is exactly its claim (`k3:tests/k3_kernels.rs:904`).
fn host_attn(a: &AttnIn, head: usize, width: usize) -> (Vec<f64>, Vec<f32>) {
    let (q, k, v, mask, scale) = (a.q, a.k, a.v, a.mask, a.scale);
    let Dims { kv, d, dv, .. } = a.dims;
    let qh = &q[head * d..head * d + width];
    let mut sc = vec![0.0f64; kv];
    for (s, x) in sc.iter_mut().enumerate() {
        let ks = &k[(head * kv + s) * d..(head * kv + s) * d + width];
        let dot: f64 = qh
            .iter()
            .zip(ks)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        *x = dot * f64::from(scale) + f64::from(mask[s]);
    }
    let probs = softmax64(&sc);
    let out = sums64(dv, |j| {
        probs
            .iter()
            .enumerate()
            .map(|(s, p)| p * f64::from(v[(head * kv + s) * dv + j]))
            .sum()
    });
    (probs, out)
}

/// **The attention core reproduces the reference at every MLA layer, at both draws.**
#[test]
fn mha_attend_matches_the_anchor() {
    let tol = tolerance::rel_tolerance("mla_attend");
    for_each_mla(|salt, layer, a| {
        let got = device_attend(&a.inputs());
        let r = rel(&got, &a.out);
        // The operator tolerance is a whole-model floor carrying upstream drift, so against
        // it alone a two-order degradation in this kernel passes in silence. Measured worst
        // over both draws and all three MLA layers, then given 10x of room
        // (`k3:tests/k3_kernels.rs:955`).
        tripwire(
            r,
            Bars {
                tol,
                observed: 2.0e-7,
            },
            &format!("{salt} layer {layer}"),
        );
        assert!(
            r <= tol,
            "{salt} layer {layer}: {r:e} exceeds {tol:e} at {:?}",
            (a.dims.heads, a.dims.kv, a.dims.d, a.dims.dv)
        );
    });
}

/// **The captured scale is over the full head width, and the fixture can tell.**
///
/// Not a restatement of the spec: it is the arithmetic check `--defect MlaScaleFromNope`
/// would have to defeat. The captured value is compared against BOTH readings, and the
/// second assertion is the one carrying information — if `qk_nope` happened to equal the
/// full width, this says so instead of passing vacuously (`k3:tests/k3_kernels.rs:969`).
#[test]
fn the_softmax_scale_is_over_the_full_head_width() {
    for_each_mla(|salt, layer, a| {
        assert!(a.rope > 0, "{salt}: a zero rope width makes this vacuous");
        let full = 1.0 / (a.dims.d as f32).sqrt();
        let nope_only = 1.0 / (a.nope as f32).sqrt();
        assert!(
            (a.scale - full).abs() < 1e-6,
            "{salt} layer {layer}: scale {} is not 1/sqrt({})",
            a.scale,
            a.dims.d
        );
        assert!(
            (a.scale - nope_only).abs() > 1e-3,
            "{salt} layer {layer}: the two readings of the scale are indistinguishable at \
             these widths, so this fixture cannot see MlaScaleFromNope"
        );
    });
}

/// **The unrotated rope dims are present in the key and are actually scored.**
///
/// §5's "silent bug" is dropping the second term of the score. A kernel ignoring those dims
/// still produces plausible output, so only a comparison against the reference's own
/// probabilities proves the term was in — the captured `probs` are the intermediate that
/// makes a compensating error (a wrong softmax cancelled by a wrong value reduction)
/// separable, and they were captured for exactly this (`k3:tests/k3_kernels.rs:995`).
#[test]
fn the_unrotated_rope_dims_are_still_scored() {
    let tol = tolerance::rel_tolerance("mla_attend");
    for_each_mla(|salt, layer, a| {
        let inp = a.inputs();
        // The oracle is checked against the reference's own probabilities BEFORE it is used
        // to judge anything.
        for h in 0..a.dims.heads {
            let (p, _) = host_attn(&inp, h, a.dims.d);
            let got: Vec<f32> = p.iter().map(|x| *x as f32).collect();
            let want = &a.probs[h * a.dims.kv..(h + 1) * a.dims.kv];
            let r = rel(&got, want);
            assert!(
                r <= tol,
                "{salt} layer {layer} head {h}: the host oracle's softmax differs from the \
                 reference's captured probs by {r:e}"
            );
        }
        let mut worst = 0.0f32;
        for h in 0..a.dims.heads {
            // The same oracle at a narrower width: `a.nope` drops the rope dims from the
            // score.
            let (_, nope_out) = host_attn(&inp, h, a.nope);
            let want = &a.out[h * a.dims.dv..(h + 1) * a.dims.dv];
            worst = worst.max(rel(&nope_out, want));
        }
        // Scored in the operator's own units against the operator's own threshold, rather
        // than a magic constant in a different metric: recomputing the OUTPUT says the thing
        // the doc claims — a kernel that dropped the rope dims would fail
        // `mha_attend_matches_the_anchor`, and by this margin (`k3:tests/k3_kernels.rs:1029`).
        assert!(
            worst > tol,
            "{salt} layer {layer}: dropping the rope dims moves the output by only {worst:e}, \
             inside the {tol:e} tolerance — this fixture cannot see §5's silent bug"
        );
    });
}

/// **The gate is applied to the attention output, before `o_proj`, with no norm.**
///
/// Trap 10 read off the file rather than from the spec: `o_proj`'s captured INPUT must equal
/// `attend.out * sigmoid(g_proj)`, computed by the kernel that will do it in the engine.
/// **Adapted to the in-place launcher**: the k3 tree's fixture wrote to a third buffer; this
/// engine's `sigmoid_gate` gates `x` in place (the out-of-place copy was deliberately not
/// ported — `kernels/fwd.hip` records the decision), so `a.out` is uploaded MUTABLY, gated,
/// and read back from the same buffer (`k3:tests/k3_kernels.rs:1043`).
#[test]
fn the_gate_ordering_is_the_one_mla_uses() {
    let tol = tolerance::rel_tolerance("mla_attend");
    for_each_mla(|salt, layer, a| {
        let s = stream();
        let mut xb = dev(&f32b(&a.out));
        let gb = dev(&f32b(&a.gate));
        // SAFETY: `x` is `n` writable f32 gated in place, `g` is `n` readable f32; they do
        // not alias, both outlive the stream, and `back` synchronises the device.
        ok(
            unsafe {
                launch_sigmoid_gate(
                    xb.ptr_mut() as *mut f32,
                    gb.ptr() as *const f32,
                    a.gate.len(),
                    s.raw(),
                )
            },
            "sigmoid_gate",
        );
        let r = rel(&f32v(&back(&xb)), &a.gated);
        assert!(
            r <= tol,
            "{salt} layer {layer}: the gated value differs by {r:e} — MLA gates the \
             attention output with no norm, before o_proj"
        );
    });
    // **The KDA contrast, and it is not the one the k3 test first asserted.** Its first
    // version claimed a KDA layer has no output-gate projection, and went red on the first
    // run: KDA has a `g_proj` too. Trap 10 is not "one gates and the other does not" — both
    // gate, and the difference is the ORDER: KDA carries an `o_norm` and normalises before
    // gating; MLA has no norm on that path at all. Asserting the `o_norm` presence both ways
    // is what makes it a contrast rather than an observation about one layer. What this
    // cannot reach: KDA's norm and gate are fused inside fla's `FusedRMSNormGated`, so the
    // intermediate is not captured — `kernel_k3_conv_norm.rs` proves that order on the
    // operator boundary (`k3:tests/k3_kernels.rs:1073`).
    let g = load(GOLDENS[0].1);
    let has = |n: &str| g.floats.iter().any(|(k, _, _)| k == n);
    assert!(
        has("model.layers.1.self_attn.o_norm"),
        "a KDA layer norms before it gates, so it must carry an o_norm"
    );
    assert!(
        has("model.layers.1.self_attn.g_proj"),
        "a KDA layer gates too — the difference from MLA is the order, not the gate"
    );
    assert!(
        !has("model.layers.3.self_attn.o_norm"),
        "MLA gates with NO norm (trap 10); an o_norm on an MLA layer means the two paths \
         have converged and the trap is gone"
    );
}

/// **What the MLA goldens structurally cannot reach: real widths, a mask that masks, and
/// magnitudes where the softmax's stability device matters.**
///
/// Found by red-proofing in the k3 tree, and both gaps were silent: ignoring `mask` entirely
/// and removing the max-subtraction each left every golden-backed test GREEN — the decode
/// captures' masks are all zero (causality masks nothing at one decode step) and the golden
/// magnitudes never push `exp` near overflow. The widths are the third gap: the goldens are
/// 4 heads of 24/16 against a real 96 of 192/128. Scored against the f64 host oracle the
/// golden tests validate at the widths the reference produced (`k3:tests/k3_kernels.rs:1104`).
#[test]
fn mha_attend_at_real_widths_masks_and_magnitudes() {
    let tol = tolerance::rel_tolerance("mla_attend");
    let mut r = Lcg(0x11A5);
    // `(heads, kv, d, dv, masked, gain)`. 96/192/128 are the real model's. kv 1024 wraps the
    // per-position loop well past one block; kv 9 reproduces the goldens' shape. `gain`
    // scales q and k so the raw scores reach ~1e2, where `expf` without a max-subtraction
    // overflows to inf and the output becomes NaN — the case the goldens cannot produce.
    for &(heads, kv, d, dv, masked, gain) in &[
        (96usize, 64usize, 192usize, 128usize, false, 1.0f32),
        (4, 1024, 192, 128, true, 1.0),
        (8, 300, 24, 16, true, 1.0),
        (2, 33, 192, 128, false, 40.0),
        (2, 33, 192, 128, true, 40.0),
    ] {
        let q: Vec<f32> = (0..heads * d).map(|_| r.f() * gain).collect();
        let k: Vec<f32> = (0..heads * kv * d).map(|_| r.f() * gain).collect();
        let v: Vec<f32> = (0..heads * kv * dv).map(|_| r.f()).collect();
        // A mask that actually masks: the second half of the keys is forbidden. Not a causal
        // triangle, because at one query row a causal mask IS all-zero — which is exactly
        // how the golden ended up unable to test this.
        let mask: Vec<f32> = (0..kv)
            .map(|s| {
                if masked && s >= kv / 2 {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let scale = 1.0 / (d as f32).sqrt();

        let inp = AttnIn {
            q: &q,
            k: &k,
            v: &v,
            mask: &mask,
            dims: Dims { heads, kv, d, dv },
            scale,
        };
        let got = device_attend(&inp);

        for h in 0..heads {
            let (_, want) = host_attn(&inp, h, d);
            assert!(
                got[h * dv..(h + 1) * dv].iter().all(|x| x.is_finite()),
                "heads={heads} kv={kv} d={d} gain={gain} head {h}: non-finite output — the \
                 softmax lost its max-subtraction"
            );
            let d_rel = rel(&got[h * dv..(h + 1) * dv], &want);
            // Same argument as the AttnRes sweep's tripwire: `tol` is a whole-model floor
            // and this cell is kernel-versus-f64-oracle, three orders tighter.
            tripwire(
                d_rel,
                Bars {
                    tol,
                    observed: MLA_SWEEP_WORST,
                },
                "the mha_attend width sweep",
            );
            assert!(
                d_rel <= tol,
                "heads={heads} kv={kv} d={d} dv={dv} masked={masked} gain={gain} head {h}: \
                 {d_rel:e} exceeds {tol:e}"
            );
        }
    }
}

/// Every launcher guard, by CODE. `kv` is LDS-staged, so its ceiling (8192) is the kernel's
/// own and refusing above it is the launcher doc's contract — truncation would be a silently
/// shorter context.
#[test]
fn the_mha_attend_launcher_guards_its_widths() {
    let dims = Dims {
        heads: 2,
        kv: 4,
        d: 8,
        dv: 8,
    };
    // One draw, sliced — a guard fixture needs plausible bytes, not a per-operand stream,
    // and the sweep above builds the same struct from same-named per-operand draws, which
    // jscpd reported as a clone of this until the two spellings diverged.
    let buf = common::fill(dims.heads * dims.kv * (dims.d + dims.dv), 0x11A6, 1.0);
    let (qn, kn) = (dims.heads * dims.d, dims.heads * dims.kv * dims.d);
    let inp = AttnIn {
        q: &buf[..qn],
        k: &buf[..kn],
        v: &buf[..dims.heads * dims.kv * dims.dv],
        mask: &buf[..dims.kv],
        dims,
        scale: 0.5,
    };
    let go = |(heads, kv, d, dv)| attend_launch(&inp, Dims { heads, kv, d, dv });
    assert_guards([
        (1001, "zero heads", go((0, dims.kv, dims.d, dims.dv))),
        (
            1001,
            "zero query width",
            go((dims.heads, dims.kv, 0, dims.dv)),
        ),
        (
            1001,
            "zero value width",
            go((dims.heads, dims.kv, dims.d, 0)),
        ),
        (1004, "zero keys", go((dims.heads, 0, dims.d, dims.dv))),
        // One past ATTEND_MAX_KV. The buffers are 4 rows, but every rejected case returns
        // before a dereference — the launch never happens, which is what a guard test is
        // allowed to rely on (`kernel_v4_moe.rs` makes the same argument).
        (
            1004,
            "kv past the LDS stage",
            go((dims.heads, 8193, dims.d, dims.dv)),
        ),
    ]);
    assert!(
        go((dims.heads, dims.kv, dims.d, dims.dv)).is_ok(),
        "the accepted case was refused"
    );
}
