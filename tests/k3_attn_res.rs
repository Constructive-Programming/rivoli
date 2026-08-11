//! **`attn_res` on the device, scored against the S1b anchor.** S2 item 1 of
//! `docs/investigations/k3-port.md`.
//!
//! Kimi-K3 does not have a plain residual. It keeps a stack of snapshots taken at block
//! boundaries plus a running prefix sum, and forms each module's input as a softmax-weighted
//! mixture over that stack (`k3-architecture.md` §3). Arithmetically it is the cheapest operator in
//! the model — ~24 M MAC/token — and it is S2 item 1 because it is *structural*: the `[T][9][7168]`
//! stack has to be live across the whole layer loop, so getting its shape wrong is a decision that
//! propagates into S3's layer loop and its prefill sizing.
//!
//! # What this suite promises, and what it cannot
//!
//! **It is a correctness gate, unlike `k3_anchor.rs`.** Every case here runs the real HIP kernel
//! over inputs the first-party reference produced and compares against the output that reference
//! produced from those same inputs. The anchor test next door compares no rivoli output to
//! anything; this one does nothing else.
//!
//! What it cannot promise is the *stack*: the fixture is handed one already-assembled
//! `[nsrc][hidden]` snapshot set, so it says nothing about whether the layer loop pushes at the
//! right boundary, resets `prefix_sum` when it should, or keeps the layer-0 snapshot alive for the
//! model-level fold. **Those are S3's, and §3's layer loop is the specification they owe.** Read
//! this as "the kernel computes the fold correctly", not "AttnRes is right".
//!
//! # The fold weights are in the goldens because this test needed them
//!
//! Until 2026-08-11 the anchor captured `in.prefix_sum`, `in.block_residual` and `out` and **not**
//! the scoring vector, so there was no way to get from the inputs to the output — every fixture
//! here would have been unwriteable. `wrap_attn_res` now captures `norm.weight * proj.weight` as
//! `.fold`, the product rather than the factors because the collapse is a load-time step the port
//! does in its loader. Found by walking up to write this file and finding nothing to launch with.
//!
//! Device tests: run with `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::backend::hip::launch_attn_res;
use rivoli::v4oracle::golden::GoldenSet;

mod common;
use common::{back, dev, f32b, f32v, ok, zeros};

#[path = "common/k3_golden.rs"]
mod k3_golden;
use k3_golden::float;

#[path = "common/k3_tolerance.rs"]
mod k3_tolerance;

/// Both vendored draws. A kernel bug degenerate at one draw's values hides completely, and the
/// softmax is exactly the sort of arithmetic that has a degenerate case — a stack whose scores
/// happen to be far apart collapses the mixture onto one source and stops testing the mixing at
/// all. Running both is the anchor's own argument for vendoring two, applied to the kernel.
const GOLDENS: [(&str, &[u8]); 2] = [
    (
        "k3-anchor-1",
        include_bytes!("k3-anchor-decode-k3-anchor-1.bin"),
    ),
    (
        "k3-anchor-2",
        include_bytes!("k3-anchor-decode-k3-anchor-2.bin"),
    ),
];

/// Every fold the anchor captures: two per captured layer, plus the model-level one.
///
/// Layer 0's `self_attention_res` is deliberately absent and its absence is load-bearing — §3's
/// layer loop guards the layer-level fold on a NON-EMPTY block stack, and at layer 0 nothing has
/// been pushed yet. A fixture listing it would fail to find it, which is the correct outcome for
/// the wrong reason; naming the set explicitly is how that stays a statement rather than an
/// accident of what happened to be in the file.
const FOLDS: [&str; 12] = [
    "model.layers.0.mlp_res",
    "model.layers.1.self_attention_res",
    "model.layers.1.mlp_res",
    "model.layers.3.self_attention_res",
    "model.layers.3.mlp_res",
    "model.layers.12.self_attention_res",
    "model.layers.12.mlp_res",
    "model.layers.91.self_attention_res",
    "model.layers.91.mlp_res",
    "model.layers.92.self_attention_res",
    "model.layers.92.mlp_res",
    "model.output_attn_res",
];

fn load(bytes: &[u8]) -> GoldenSet {
    GoldenSet::read_k3(&mut &bytes[..]).expect("the vendored golden must load")
}

/// The eps the reference's RMSNorm used, read off the golden's own `tiny_config`.
///
/// Not a literal `1e-5`. The reference reads `norm.variance_epsilon`, which is
/// `config.rms_norm_eps`, and a fixture hardcoding the value would agree with itself if that ever
/// moved. `k3_anchor.rs` separately pins the tiny config's `rms_norm_eps` against the real
/// checkpoint's, so the two together say this is the model's eps and not merely the file's.
fn eps(g: &GoldenSet) -> f32 {
    let cfg: serde_json::Value =
        serde_json::from_str(g.meta_get("tiny_config").expect("tiny_config")).unwrap();
    cfg["rms_norm_eps"].as_f64().expect("rms_norm_eps") as f32
}

/// One fold: the assembled stack, its scoring vector, and the output the reference produced.
///
/// Built by one function because the two tests below opened with the same six lookups and
/// `build.rs`'s jscpd gate rejected the second copy at 100 tokens. It is also the better shape —
/// a fold is these four things together, and a test that assembled three of them would be
/// scoring something the reference never computed.
struct Case {
    src: Vec<f32>,
    nsrc: usize,
    n: usize,
    fold: Vec<f32>,
    want: Vec<f32>,
}

fn case(g: &GoldenSet, tag: &str) -> Case {
    let (src, nsrc, n) = stack(g, tag);
    let (fold_shape, fold) = float(g, &format!("{tag}.fold"));
    assert_eq!(
        fold_shape,
        [n],
        "{tag}: the fold is one scoring vector of hidden"
    );
    let (out_shape, want) = float(g, &format!("{tag}.out"));
    assert_eq!(out_shape, [1, n], "{tag}: the output is one row of hidden");
    Case {
        src,
        nsrc,
        n,
        fold: fold.to_vec(),
        want: want.to_vec(),
    }
}

/// One fold's inputs, flattened into the `[nsrc][hidden]` stack the kernel takes.
///
/// The **prefix sum goes last**, after the block snapshots. That order is the reference's
/// (`torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)`) and it is not arbitrary: the
/// softmax is permutation-equivariant, so a port that concatenated the other way round produces
/// exactly the same output and is nonetheless wrong the moment anything indexes the stack — which
/// S3 does, when it pushes onto it. Reversing it here does not go red, and that is precisely why
/// this comment exists instead of a test.
fn stack(g: &GoldenSet, tag: &str) -> (Vec<f32>, usize, usize) {
    let (br_shape, br) = float(g, &format!("{tag}.in.block_residual"));
    let (ps_shape, ps) = float(g, &format!("{tag}.in.prefix_sum"));
    let hidden = *br_shape.last().unwrap();
    assert_eq!(
        ps_shape,
        [1, hidden],
        "{tag}: prefix sum is one row of hidden"
    );
    assert_eq!(br_shape[0], 1, "{tag}: the fixture is one token");
    let blocks = br_shape[1];
    let mut v = Vec::with_capacity((blocks + 1) * hidden);
    v.extend_from_slice(br);
    v.extend_from_slice(ps);
    (v, blocks + 1, hidden)
}

/// The reference's fold, transliterated, in `f64` — the host oracle the device is scored against.
///
/// `f64` because the reference accumulates every long reduction in double (`k3-architecture.md`
/// §10's closing note), so this reproduces the reference's own arithmetic rather than a second
/// fp32 implementation of it. Scoring the kernel against an fp32 host would measure the two
/// fp32 implementations' disagreement, which is a different and much smaller number than the one
/// `attn_res`'s tolerance was measured for.
fn host_fold(
    src: &[f32],
    nsrc: usize,
    n: usize,
    fold: &[f32],
    eps: f32,
    normalised: bool,
) -> Vec<f32> {
    let mut score = vec![0.0f64; nsrc];
    let mut inv = vec![0.0f64; nsrc];
    for s in 0..nsrc {
        let v = &src[s * n..(s + 1) * n];
        let ss: f64 = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
        inv[s] = 1.0 / (ss / n as f64 + f64::from(eps)).sqrt();
        score[s] = v
            .iter()
            .zip(fold)
            .map(|(x, f)| (f64::from(*x) * inv[s]) * f64::from(*f))
            .sum();
    }
    let m = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = score.iter().map(|s| (s - m).exp()).collect();
    let z: f64 = ex.iter().sum();
    let mut out = vec![0.0f32; n];
    for s in 0..nsrc {
        let p = ex[s] / z;
        let v = &src[s * n..(s + 1) * n];
        // `normalised` is the `AttnResNormalisedValues` body: mix `k`, the scored vector, instead
        // of the raw source. One substitution, the same shape as the driver's own defect.
        let scale = if normalised { inv[s] } else { 1.0 };
        for i in 0..n {
            out[i] += (p * f64::from(v[i]) * scale) as f32;
        }
    }
    out
}

/// Relative difference the way `--by-operator` measures it, so the number compared here is the
/// number the tolerance was measured in.
fn rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-30);
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

/// Every (draw, fold) pair, with its case assembled and the reference's eps in hand.
///
/// The two tests below opened with the identical six-line traversal and jscpd rejected the second
/// copy at 90 tokens. Passing a closure rather than returning a Vec keeps each golden's borrow
/// alive exactly as long as the cases it lends out, which is what the borrow checker wants here
/// and what a collected Vec would force into an owned copy of every capture.
fn for_each_fold(mut f: impl FnMut(Fold)) {
    let tol = match k3_tolerance::tolerance("attn_res").expect("attn_res has a measured tolerance")
    {
        k3_tolerance::Policy::Rel(t) => *t,
        k3_tolerance::Policy::ExactOnly => {
            panic!("attn_res is tabled ExactOnly; this fixture scores it with a threshold")
        }
    };
    for (salt, bytes) in GOLDENS {
        let g = load(bytes);
        let eps = eps(&g);
        for tag in FOLDS {
            f(Fold {
                salt,
                tag,
                eps,
                tol,
                c: case(&g, tag),
            });
        }
    }
}

/// One scoring site's whole context. The tolerance rides along rather than being looked up per
/// test: it is a property of the OPERATOR, so every site that scores `attn_res` must use the same
/// one, and two call sites reading the table independently is how they stop agreeing.
struct Fold<'a> {
    salt: &'a str,
    tag: &'a str,
    eps: f32,
    tol: f32,
    c: Case,
}

/// Score one vector against the fold's golden output, in the units the tolerance was measured in.
///
/// Every comparison in this file goes through here, so the device result and the two host results
/// are scored identically. Three call sites each computing their own relative difference is how a
/// suite ends up with a defect that "fails" against a slightly different denominator.
fn score(f: &Fold, got: &[f32]) -> f32 {
    rel(got, &f.c.want)
}

/// Run the kernel over one already-flattened `[tokens][nsrc][n]` stack.
///
/// Takes the pieces rather than a `Fold`, because the width sweep at the bottom of this file has
/// no golden and therefore no `Fold` — and jscpd rejected the second copy of the launch block at
/// 34 tokens, which is the correct outcome: one launcher call site means one place where the
/// argument order can be wrong.
fn device(src: &[f32], fold: &[f32], tokens: usize, nsrc: usize, n: usize, eps: f32) -> Vec<f32> {
    let (sb, fb) = (dev(&f32b(src)), dev(&f32b(fold)));
    let mut ob = zeros(tokens * n * 4);
    // SAFETY: `src` is `tokens·nsrc·n` f32, `fold` is `n` f32 and `out` is `tokens·n` f32, all
    // live for the call and mutually non-aliasing, as the launcher requires.
    ok(
        unsafe {
            launch_attn_res(
                sb.ptr() as *const f32,
                fb.ptr() as *const f32,
                tokens,
                nsrc,
                n,
                eps,
                ob.ptr_mut() as *mut f32,
            )
        },
        "attn_res",
    );
    f32v(&back(&ob))
}

/// **The kernel reproduces every fold the anchor captured, at both draws.**
#[test]
fn attn_res_matches_the_anchor_at_every_fold() {
    for_each_fold(|f| {
        let r = score(&f, &device(&f.c.src, &f.c.fold, 1, f.c.nsrc, f.c.n, f.eps));
        // **A second, much tighter bound — a regression tripwire, NOT the operator's tolerance.**
        //
        // The kernel agrees with the golden to 3.08e-7 worst over both draws (several folds are
        // bit-exact), which is ~50x BELOW `attn_res`'s own fp32 floor of 1.571e-5 and ~500x below
        // the 1.6e-4 tolerance. That is not the kernel being lucky: the floor was measured on
        // whole-model fp32-vs-fp64 runs, so it carries upstream drift, while this fixture hands
        // the kernel the reference's OWN inputs and measures the operator alone.
        //
        // The tolerance is still the contract — it is what S3 will need once upstream drift is
        // real. But against it alone, a change that degraded this kernel by two orders of
        // magnitude would pass in silence. 10x the observed worst is close enough to catch that
        // and far enough not to fire on a reassociated sum.
        const OBSERVED_WORST: f32 = 3.08e-7;
        assert!(
            r <= OBSERVED_WORST * 10.0,
            "{} {}: {r:e} is far above the {:e} this kernel actually achieves. Still inside the \
             {:e} operator tolerance, so this is a REGRESSION tripwire, not a correctness \
             failure — if the new value is defensible, re-measure and move the constant.",
            f.salt,
            f.tag,
            OBSERVED_WORST,
            f.tol
        );
        assert!(
            r <= f.tol,
            "{} {}: relative difference {r:e} exceeds the measured tolerance {:e} (nsrc={}, \
             hidden={})",
            f.salt,
            f.tag,
            f.tol,
            f.c.nsrc,
            f.c.n
        );
    });
}

/// **Every fold mixes the stack depth §3's layer loop says it should.**
///
/// Without this the suite above passes on a kernel that ignores `nsrc` and returns source 0 — most
/// folds here are the two-source case, which is the smallest mixture there is.
///
/// The depth is DERIVED from the block size rather than listed, and deriving it is what makes this
/// a test of the reference's bookkeeping instead of a transcription of it. §3's loop folds BEFORE
/// it pushes, so at a boundary layer the `self_attention_res` fold still sees the old stack and the
/// `mlp_res` fold sees the new one — layer 12 mixes 2 sources and then 3. Layers 91 and 92 carry
/// all eight snapshots plus the prefix sum: **nine sources**, the width `[T][9][7168]` is named
/// for, and the one the prefill sizing in §3 is stated at.
///
/// A port that pushed before folding, or that reset `prefix_sum` on the wrong side of the
/// boundary, produces a stack one deep in the wrong place and lands here.
#[test]
fn every_fold_mixes_the_depth_the_layer_loop_implies() {
    let block = 12; // `attn_res_block_size`; pinned against the real config by `k3_anchor.rs`
    for_each_fold(|f| {
        let want = if f.tag == "model.output_attn_res" {
            // The model-level fold runs after every layer, so every boundary has fired.
            93_usize.div_ceil(block) + 1
        } else {
            let layer: usize = f.tag.split('.').nth(2).unwrap().parse().unwrap();
            // Snapshots pushed strictly BEFORE this fold, plus this layer's own push when it is a
            // boundary and we are past the attention fold.
            let boundary_push =
                usize::from(layer.is_multiple_of(block) && f.tag.ends_with("mlp_res"));
            layer.div_ceil(block) + boundary_push + 1
        };
        assert_eq!(
            f.c.nsrc, want,
            "{} {}: mixes {} sources, and §3's layer loop implies {want}",
            f.salt, f.tag, f.c.nsrc
        );
    });
}

/// **The defect run: mixing the NORMALISED sources fails this fixture.**
///
/// G2 requires each S2 item to pass its operator fixture *and* its defect run. This is the second
/// half, and it is what separates a fixture from a formality — `--defect AttnResNormalisedValues`
/// is the trap §3 spends its clearest sentence on ("the softmax mixes the UNNORMALISED sources"),
/// and a fixture that could not tell the two apart would be green on the single most likely
/// misreading of the operator.
///
/// The defect is computed on the HOST rather than read from a defect golden, because the defect
/// goldens are not vendored — only the clean ones are — and a test that needs `target/k3-anchor/`
/// is a test that passes by being skipped. The driver's `_fold_mixing_normalised` is the same
/// one-line substitution, so the two agree on what the defect is.
///
/// The clean host oracle is asserted FIRST. Without it, "the defect disagrees with the golden"
/// would also be satisfied by a host transliteration that disagrees with everything.
#[test]
fn mixing_the_normalised_sources_goes_red() {
    for_each_fold(|f| {
        let c = &f.c;
        let clean = score(&f, &host_fold(&c.src, c.nsrc, c.n, &c.fold, f.eps, false));
        assert!(
            clean <= f.tol,
            "{} {}: the host oracle itself is {clean:e} from the golden",
            f.salt,
            f.tag
        );
        let defect = score(&f, &host_fold(&c.src, c.nsrc, c.n, &c.fold, f.eps, true));
        assert!(
            defect > f.tol,
            "{} {}: mixing the normalised sources differs by only {defect:e}, inside the {:e} \
             tolerance — this fixture cannot see the trap it exists to catch",
            f.salt,
            f.tag,
            f.tol
        );
    });
}

/// **Widths and token counts the goldens structurally cannot reach.**
///
/// Every capture in the anchor is at the tiny model's `hidden` of 192, and the kernel launches 256
/// threads — so in **every test above, the strided loop `for (i = t; i < n; i += blockDim.x)` runs
/// at most one iteration and 64 threads run none at all.** At the real width of 7168 each thread
/// runs 28. An accumulation bug that only appears when the loop wraps is invisible to the entire
/// golden-backed suite, and would first be seen on the real checkpoint at S4.
///
/// Every fixture above also passes `tokens = 1`, so `blockIdx.x`'s two strides — `nsrc * n` into
/// `src` and `n` into `out` — are multiplied by zero throughout. Layer-major prefill passes the
/// whole prompt at once (`architecture.md` §14), so that is the path S3 will actually take.
///
/// **Credit where due**: this gap was pointed out by the session porting Muse Glimmer, which found
/// the same shape in its own attend fixture (all captures at `head_dim` 8 against a real 128). Two
/// ports, one anchor design, one blind spot — worth stating plainly, because it is a property of
/// generating goldens from a tiny model and it applies to every remaining S2 item.
///
/// Scored against the same f64 host oracle the golden tests validate, so this inherits their
/// evidence rather than asserting a fresh claim: `mixing_the_normalised_sources_goes_red` shows
/// that oracle agrees with the reference at 192, and this shows the kernel agrees with the oracle
/// at widths the reference never produced.
#[test]
fn attn_res_at_real_widths_and_multiple_tokens() {
    let tol = match k3_tolerance::tolerance("attn_res").expect("attn_res has a measured tolerance")
    {
        k3_tolerance::Policy::Rel(t) => *t,
        k3_tolerance::Policy::ExactOnly => panic!("attn_res is tabled ExactOnly"),
    };
    let mut r = common::Lcg(0xA77E);
    // 192 reproduces the goldens' width so a failure here is attributable to the synthetic data
    // rather than to the width; 257 wraps the loop exactly once with a one-thread tail; 1000 wraps
    // it unevenly; 7168 is the real hidden. 3 tokens because 2 cannot distinguish "stride applied
    // once" from "stride applied per block".
    for (n, tokens) in [(192, 1), (257, 1), (1000, 3), (7168, 1), (7168, 3)] {
        for nsrc in [2, 9] {
            let src: Vec<f32> = (0..tokens * nsrc * n).map(|_| r.f()).collect();
            let fold: Vec<f32> = (0..n).map(|_| r.f()).collect();
            let eps = 1e-5f32;

            let got = device(&src, &fold, tokens, nsrc, n, eps);

            for t in 0..tokens {
                let want = host_fold(
                    &src[t * nsrc * n..(t + 1) * nsrc * n],
                    nsrc,
                    n,
                    &fold,
                    eps,
                    false,
                );
                let d = rel(&got[t * n..(t + 1) * n], &want);
                assert!(
                    d <= tol,
                    "n={n} tokens={tokens} nsrc={nsrc} token {t}: {d:e} exceeds {tol:e}"
                );
            }
            // Anti-vacuity: with `tokens > 1` a kernel that wrote only block 0 would leave the
            // rest of `out` at zero, and a per-token comparison against a host fold of the RIGHT
            // slice would catch that — but only if the tokens differ from each other. They are
            // drawn independently, so assert it rather than trust the draw.
            if tokens > 1 {
                let (a, b) = (&got[..n], &got[n..2 * n]);
                assert!(
                    a.iter().zip(b).any(|(x, y)| x != y),
                    "n={n} nsrc={nsrc}: two tokens produced identical output — the block stride is \
                     not being applied"
                );
            }
        }
    }
}
