//! **Kimi-K3's kernels on the device, scored against the S1b anchor.** S2 items 1, 2 and 3 of
//! `docs/investigations/k3-port.md` — AttnRes, the gated MLA core, and the MoE latent sandwich.
//!
//! One file rather than three because they share every helper that matters: the two vendored draws,
//! the `rel` scorer, and the tolerance table. Three files meant three copies of each, and jscpd
//! rejected the second of them.
//!
//! # Item 1 — AttnRes
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

use rivoli::backend::hip::{
    launch_attn_res, launch_gemm_bf16, launch_mha_attend, launch_rmsnorm_batch,
    launch_rmsnorm_single, launch_sigmoid_gate, launch_situ_glu_f32,
};
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
/// §10's closing note), so this reproduces the reference's own arithmetic rather than a second fp32
/// implementation of it. Scoring the kernel against an fp32 host would measure two fp32
/// implementations disagreeing, which is a different and much smaller number than the one
/// `attn_res`'s tolerance was measured for.
///
/// Split into `host_probs` and `host_fold` because three places need the probabilities and only one
/// needs the mixture — and because a review found the same f64 softmax written out three times in
/// this file, which jscpd could not see (the surrounding code differs) and which is the one place
/// factoring genuinely paid.
///
/// Returns the softmax probabilities and each source's RMSNorm scale, since the
/// `AttnResNormalisedValues` defect needs the latter to mix the scored vector instead of the raw
/// one.
/// Max-subtracted softmax in `f64`. Both oracles ended with these four lines; jscpd caught it.
fn softmax64(score: &[f64]) -> Vec<f64> {
    let m = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = score.iter().map(|s| (s - m).exp()).collect();
    let z: f64 = ex.iter().sum();
    ex.into_iter().map(|e| e / z).collect()
}

/// One AttnRes fold's inputs. Bundled for the same reason as `AttnIn`: `host_probs` and
/// `host_fold` take the same five things and the duplicate parameter list is a clone.
struct FoldIn<'a> {
    src: &'a [f32],
    nsrc: usize,
    n: usize,
    fold: &'a [f32],
    eps: f32,
}

fn host_probs(f: &FoldIn) -> (Vec<f64>, Vec<f64>) {
    let (src, nsrc, n, fold, eps) = (f.src, f.nsrc, f.n, f.fold, f.eps);
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
    (softmax64(&score), inv)
}

fn host_fold(f: &FoldIn, normalised: bool) -> Vec<f32> {
    let (src, nsrc, n) = (f.src, f.nsrc, f.n);
    let (p, inv) = host_probs(f);
    let mut out = vec![0.0f64; n];
    for s in 0..nsrc {
        let v = &src[s * n..(s + 1) * n];
        // `normalised` is the `AttnResNormalisedValues` body: mix `k`, the scored vector, instead
        // of the raw source. One substitution, the same shape as the driver's own defect.
        let scale = if normalised { inv[s] } else { 1.0 };
        for (i, o) in out.iter_mut().enumerate() {
            *o += p[s] * f64::from(v[i]) * scale;
        }
    }
    // Cast ONCE at the end. This accumulated in f32 until review 2026-08-12 pointed out the doc
    // above promised f64 throughout and the mixture did not honour it — ~9 roundings, three orders
    // under the tolerance, but this oracle is the only thing behind the sweep's real-width cells.
    out.into_iter().map(|x| x as f32).collect()
}

/// Relative difference the way `--by-operator` measures it, so the number compared here is the
/// number the tolerance was measured in.
impl Attend {
    /// This layer's boundary as an attention call. One constructor, so the two tests that launch
    /// over it cannot disagree about which buffers go where.
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

/// **The regression tripwire every item in this file carries, in one place.**
///
/// `k3_tolerance`'s operator tolerances are WHOLE-MODEL floors: they were measured on fp32-vs-fp64
/// runs carrying upstream drift, while these fixtures hand each kernel the reference's OWN inputs.
/// Every kernel here therefore lands one to three orders under its tolerance — and against the
/// tolerance alone, a change that degraded a kernel by two orders would pass in silence. So each
/// site also pins the worst it actually measures and gets 10x of room: close enough to catch a
/// regression, far enough not to fire on a reassociated sum.
///
/// **The tolerance is still the contract.** This is not a second one; it is a smoke alarm on a
/// number that has no business moving. Moving a constant is allowed and re-measuring is how — what
/// is not allowed is loosening one to make a red run green without knowing why it moved.
///
/// One function rather than four copies. jscpd rejected the fourth at 135 tokens, which is the gate
/// noticing that four sites were making the same argument in the same words; the better shape is
/// that they can no longer disagree about what the tripwire means.
fn tripwire(r: f32, observed_worst: f32, tol: f32, at: &str) {
    assert!(
        r <= observed_worst * 10.0,
        "{at}: {r:e} is far above the {observed_worst:e} this kernel achieves. Still inside the \
         {tol:e} operator tolerance, so this is a REGRESSION tripwire — re-measure and move the \
         constant only if the new value is defensible."
    );
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    // **Non-finite is INFINITY, not zero, and this is the most important line in the file.**
    //
    // `f32::max` is documented to return the OTHER operand when one is NaN, so the obvious fold
    // `m.max((x - y).abs())` silently skips every NaN difference and an all-NaN output scores
    // exactly 0.0 — indistinguishable from a perfect match. Measured, not reasoned:
    // `rel(&[NaN; 3], &[1.0, 2.0, 3.0])` returned 0. Every scoring site in this file goes through
    // here, so a kernel emitting NaN passed all four golden-backed fixtures AND the regression
    // tripwire.
    //
    // Found by two independent reviews 2026-08-12. The tell was already in the tree: the width
    // sweep carries its own `is_finite` assert, added after red-proofing found `mha_attend` could
    // produce NaN — one call site was patched instead of the shared helper, which is exactly the
    // shape that leaves the other five exposed.
    if a.iter().chain(b).any(|x| !x.is_finite()) {
        return f32::INFINITY;
    }
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
    let tol = k3_tolerance::rel_tolerance("attn_res");
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
fn device(
    src: &[f32],
    fold: &[f32],
    tokens: usize,
    nsrc: usize,
    n: usize,
    stride: usize,
    eps: f32,
) -> Vec<f32> {
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
                stride,
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
        let r = score(
            &f,
            &device(
                &f.c.src,
                &f.c.fold,
                1,
                f.c.nsrc,
                f.c.n,
                f.c.nsrc * f.c.n,
                f.eps,
            ),
        );
        // **A second, much tighter bound — a regression tripwire, NOT the operator's tolerance.**
        //
        // The kernel agrees with the golden to 3.08e-7 worst over both draws (several folds are
        // bit-exact), which is ~230x BELOW `attn_res`'s own fp32 floor of 7.052e-5 and ~2,300x
        // below the 7.1e-4 tolerance. That is not the kernel being lucky: the floor was measured on
        // whole-model fp32-vs-fp64 runs, so it carries upstream drift, while this fixture hands
        // the kernel the reference's OWN inputs and measures the operator alone.
        //
        // *(Those two ratios were 50x and 500x until 2026-08-12, against a floor of 1.571e-5 that
        // turned out to be draw 1's alone. The kernel's own 3.08e-7 did not move — only what it is
        // being compared against did, which is exactly why the tripwire below is the real gate.)*
        //
        // The generic half of this argument now lives on `tripwire`, which all four items call.
        tripwire(r, 3.08e-7, f.tol, &format!("{} {}", f.salt, f.tag));
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
        let fin = FoldIn {
            src: &c.src,
            nsrc: c.nsrc,
            n: c.n,
            fold: &c.fold,
            eps: f.eps,
        };
        let clean = score(&f, &host_fold(&fin, false));
        assert!(
            clean <= f.tol,
            "{} {}: the host oracle itself is {clean:e} from the golden",
            f.salt,
            f.tag
        );
        let defect = score(&f, &host_fold(&fin, true));
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
    let tol = k3_tolerance::rel_tolerance("attn_res");
    let mut r = common::Lcg(0xA77E);
    // 192 reproduces the goldens' width so a failure here is attributable to the synthetic data
    // rather than to the width; 257 wraps the loop exactly once with a one-thread tail; 1000 wraps
    // it unevenly; 7168 is the real hidden. 3 tokens because 2 cannot distinguish "stride applied
    // once" from "stride applied per block".
    // `(n, tokens, slots)` — `slots` is the arena's per-token capacity. `slots > nsrc` is §3's
    // real `[T][9][hidden]` layout, where the live stack is shorter than the slot count at every
    // layer below 84; `slots == nsrc` is the packed case the goldens happen to have. Both are
    // launched, because the kernel took the packed one as an assumption until 2026-08-12.
    for (n, tokens, slots) in [
        (192, 1, 0),
        (257, 1, 0),
        (1000, 3, 0),
        (7168, 1, 0),
        (7168, 3, 0),
        (1000, 3, 9),
        (192, 4, 9),
    ] {
        for nsrc in [2, 9] {
            let cap = if slots == 0 { nsrc } else { slots.max(nsrc) };
            let stride = cap * n;
            // The arena is `tokens * cap * n`; only the first `nsrc` slots of each token are the
            // live stack. The rest are filled with a LOUD sentinel rather than zeros — a kernel
            // reading the wrong token's slots picks up 9.0e9 and cannot produce a plausible
            // mixture, so the failure is unmistakable rather than merely numerical.
            let mut src: Vec<f32> = vec![9.0e9; tokens * stride];
            for t in 0..tokens {
                for e in 0..nsrc * n {
                    src[t * stride + e] = r.f();
                }
            }
            // **Scaled by 1/sqrt(n), and without this the sweep tested a copy.** `attn_res`'s
            // score has no width normaliser, so with a unit-magnitude fold the score spread grows
            // as ~0.577*sqrt(n): 8 at n=192 but 49 at n=7168, where the softmax collapses onto one
            // source. Measured by two reviews independently — max probability 1.0000000000 in all
            // five cells at 7168, runner-up down at 4e-40. The mixing loop was bitwise a copy at
            // exactly the width the sweep exists to reach. Scaling the fold keeps the scores O(1)
            // at every width; `assert_mixes` below refuses a draw that degenerates anyway.
            let fold: Vec<f32> = (0..n).map(|_| r.f() / (n as f32).sqrt()).collect();
            let eps = 1e-5f32;

            let got = device(&src, &fold, tokens, nsrc, n, stride, eps);

            // Anti-vacuity: a mixture that has collapsed onto one source tests argmax and a copy,
            // not mixing. Asserted per token rather than hoped for, because it is a property of the
            // DRAW and a future seed or width could re-degenerate it silently.
            for t in 0..tokens {
                let stack = &src[t * stride..t * stride + nsrc * n];
                let fin = FoldIn {
                    src: stack,
                    nsrc,
                    n,
                    fold: &fold,
                    eps,
                };
                let (p, _) = host_probs(&fin);
                let pmax = p.iter().copied().fold(0.0f64, f64::max);
                assert!(
                    pmax < 0.9,
                    "n={n} nsrc={nsrc} slots={cap} token {t}: one-hot (max p = {pmax:.6}), so \
                     the mixing loop is a copy and this cell tests nothing it claims to"
                );
                let want = host_fold(&fin, false);
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

/// The captured layers the real map makes MLA — zero-based 3, 91 and 92.
///
/// **91 and 92 are ADJACENT**, which the every-fourth pattern does not predict: 93 layers do not
/// divide by 4, so the map ends with two MLA layers in a row. `k3_anchor.rs` pins the partition
/// itself; this list exists so a fixture that silently stopped covering one of them is visible.
const MLA_LAYERS: [usize; 3] = [3, 91, 92];

/// The four widths an attend is shaped by. One type because they always travel together — and
/// because jscpd rejected them appearing once as `Attend`'s fields and once as `device_attend`'s
/// parameters, which is the gate noticing the same thing.
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
    /// The two halves of `d`, read from the golden's own `tiny_config`. Carried on the struct
    /// rather than re-extracted per test: two tests need them, and jscpd rejected the second copy
    /// of the extraction at 86 tokens.
    nope: usize,
    rope: usize,
    /// The output gate and the value that reached `o_proj` — the two halves of trap 10. On the
    /// struct because `Attend` claims to be the layer's whole boundary, and because the gate test
    /// otherwise had to re-open the golden, which jscpd caught at 61 tokens.
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

    // `[b, heads, q_len, d]`. The decode step is one query row, and the kernel's contract is that
    // row — asserted rather than assumed, because a q_len > 1 golden would silently make every
    // comparison below read the first row of a stack.
    assert_eq!(qs[0], 1, "batch");
    assert_eq!(qs[2], 1, "the decode fixture is one query row");
    let (heads, d) = (qs[1], qs[3]);
    let (kv, dv) = (ks[2], vs[3]);
    // `repeat_kv` runs INSIDE the reference's attention, so what is captured is pre-broadcast. The
    // kernel takes fully expanded per-head K/V, which is what K3 caches (§5 stores the expanded
    // k/v deliberately). Equal head counts here means the two agree; unequal would mean the kernel
    // is being handed something it does not implement, and passing anyway would be the bug.
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

    let cfg: serde_json::Value =
        serde_json::from_str(g.meta_get("tiny_config").expect("tiny_config")).unwrap();
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

/// Run `mha_attend` over an already-flattened boundary.
///
/// Takes the pieces rather than an `Attend`, because the synthetic sweep at the bottom of this file
/// has no golden and therefore no `Attend` — and jscpd rejected the second copy of the launch block
/// at 73 tokens. One launcher call site is also one place where the argument order can be wrong.
/// One attention call's inputs. Bundled because `device_attend` and `host_attn` take exactly the
/// same six things and jscpd rejected the second parameter list — the same lesson `Dims` taught one
/// level down, and the same answer: quantities that always travel together are one type.
struct AttnIn<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    mask: &'a [f32],
    dims: Dims,
    scale: f32,
}

fn device_attend(a: &AttnIn) -> Vec<f32> {
    let (q, k, v, mask, scale) = (a.q, a.k, a.v, a.mask, a.scale);
    let Dims { heads, kv, d, dv } = a.dims;
    let (qb, kb, vb, mb) = (
        dev(&f32b(q)),
        dev(&f32b(k)),
        dev(&f32b(v)),
        dev(&f32b(mask)),
    );
    let mut ob = zeros(heads * dv * 4);
    // SAFETY: `q` is `heads·d` f32, `k` is `heads·kv·d`, `v` is `heads·kv·dv`, `mask` is `kv`, and
    // `out` is `heads·dv` — all live for the call and mutually non-aliasing.
    ok(
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
                scale,
                ob.ptr_mut() as *mut f32,
            )
        },
        "mha_attend",
    );
    f32v(&back(&ob))
}

/// Every (draw, MLA layer) pair with its boundary assembled — the same shape as `for_each_fold`,
/// and factored for the same reason: three tests opened with the identical traversal and jscpd
/// rejected the copies.
fn for_each_mla(mut f: impl FnMut(&str, usize, Attend)) {
    for (salt, bytes) in GOLDENS {
        let g = load(bytes);
        for layer in MLA_LAYERS {
            let a = attend(&g, layer);
            f(salt, layer, a);
        }
    }
}

/// The reference's attention, in `f64`, over the first `width` dims of each head.
///
/// `width` is the whole point: `d` reproduces the operator, `nope` reproduces §5's silent bug of
/// dropping the unrotated rope dims from the score. Writing it once makes
/// `the_unrotated_rope_dims_are_still_scored` self-evidently "the same oracle at a narrower width",
/// which is exactly its claim.
///
/// Factored after review 2026-08-12 found this softmax written out three times in this file —
/// duplication jscpd cannot see, because the surrounding code differs each time.
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

/// `n` independent f64 sums, narrowed to f32 on the way out.
///
/// Both f64 oracles in this file end with exactly this shape — an f64 accumulation per output
/// element, collected as f32 — and jscpd rejected the second copy at 31 tokens. The narrowing is
/// the part worth having in one place: it is the only rounding either oracle performs, so a stray
/// `as f32` on an intermediate would be a floor that is not one.
fn sums64(n: usize, mut term: impl FnMut(usize) -> f64) -> Vec<f32> {
    (0..n).map(|j| term(j) as f32).collect()
}

/// **The attention core reproduces the reference at every MLA layer, at both draws.**
#[test]
fn mha_attend_matches_the_anchor() {
    let tol = k3_tolerance::rel_tolerance("mla_attend");
    for_each_mla(|salt, layer, a| {
        let got = device_attend(&a.inputs());
        let r = rel(&got, &a.out);
        // The twin of `attn_res`'s tripwire, added after review 2026-08-12 pointed out that item 1
        // learned this lesson and item 2 did not. Same argument: the operator tolerance is a
        // whole-model floor carrying upstream drift, so against it alone a two-order degradation
        // in this kernel passes in silence. Measured worst over both draws and all three MLA
        // layers, then given 10x of room.
        tripwire(r, 2.0e-7, tol, &format!("{salt} layer {layer}"));
        assert!(
            r <= tol,
            "{salt} layer {layer}: {r:e} exceeds {tol:e} at {:?}",
            (a.dims.heads, a.dims.kv, a.dims.d, a.dims.dv)
        );
    });
}

/// **The captured scale is over the full head width, and the fixture can tell.**
///
/// Not a restatement of the spec: it is the arithmetic check `--defect MlaScaleFromNope` would have
/// to defeat. The captured value is compared against BOTH readings, and the second assertion is the
/// one carrying information — if `qk_nope` happened to equal the full width, this says so instead
/// of passing vacuously.
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
            "{salt} layer {layer}: the two readings of the scale are indistinguishable at these \
             widths, so this fixture cannot see MlaScaleFromNope"
        );
    });
}

/// **The unrotated rope dims are present in the key and are actually scored.**
///
/// §5's "silent bug" is dropping the second term of the score. This shows the fixture can see it:
/// recomputing the scores WITHOUT the rope dims produces a different softmax from the captured
/// `probs`. That is the half that matters — a kernel ignoring those dims still produces plausible
/// output, so only a comparison against the reference's own probabilities proves the term was in.
#[test]
fn the_unrotated_rope_dims_are_still_scored() {
    let tol = k3_tolerance::rel_tolerance("mla_attend");
    for_each_mla(|salt, layer, a| {
        let inp = a.inputs();
        // **The oracle is checked against the reference's own probabilities before it is used to
        // judge anything.** Only the final output was compared until review 2026-08-12 observed
        // that a compensating error — a wrong softmax cancelled by a wrong value reduction —
        // survives an output-only comparison. The captured `probs` are the intermediate that makes
        // the two separable, and they were captured for exactly this and then not used.
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
            // The same oracle at a narrower width: `a.nope` drops the rope dims from the score.
            let (_, nope_out) = host_attn(&inp, h, a.nope);
            let want = &a.out[h * a.dims.dv..(h + 1) * a.dims.dv];
            worst = worst.max(rel(&nope_out, want));
        }
        // **Scored in the operator's own units against the operator's own threshold**, rather than
        // against a magic constant. Review 2026-08-12: this asserted an ABSOLUTE probability
        // difference exceeded 1e-3, while the correctness test it underwrites is a RELATIVE output
        // difference against `Rel(4.10e-4)` — two units with nothing connecting them. Recomputing
        // the OUTPUT says the thing the doc claims: a kernel that dropped the rope dims would fail
        // `mha_attend_matches_the_anchor`, and by this margin.
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
/// `attend.out * sigmoid(g_proj)`, computed by the kernel that will do it in the engine. If the
/// reference normed first, or gated after the projection, this does not hold.
#[test]
fn the_gate_ordering_is_the_one_mla_uses() {
    let tol = k3_tolerance::rel_tolerance("mla_attend");
    for_each_mla(|salt, layer, a| {
        let (ab, gb) = (dev(&f32b(&a.out)), dev(&f32b(&a.gate)));
        let mut ob = zeros(a.gate.len() * 4);
        // SAFETY: three live buffers of `n` f32, mutually non-aliasing.
        ok(
            unsafe {
                launch_sigmoid_gate(
                    ab.ptr() as *const f32,
                    gb.ptr() as *const f32,
                    a.gate.len(),
                    ob.ptr_mut() as *mut f32,
                )
            },
            "sigmoid_gate",
        );
        let r = rel(&f32v(&back(&ob)), &a.gated);
        assert!(
            r <= tol,
            "{salt} layer {layer}: the gated value differs by {r:e} — MLA gates the attention \
             output with no norm, before o_proj"
        );
    });
    // **The KDA contrast, and it is not the one this test first asserted.**
    //
    // The first version claimed a KDA layer has no output-gate projection. It went red on the
    // first run: KDA has a `g_proj` too, 128 wide. Trap 10 is not "one gates and the other does
    // not" — both gate, and the difference is the ORDER. KDA carries an `o_norm` and normalises
    // before gating; MLA has no norm on that path at all.
    //
    // So the contrast that IS in the file is the presence of `o_norm`, and asserting it both ways
    // is what makes it a contrast rather than an observation about one layer.
    //
    // **What this cannot reach**: KDA's norm and gate are fused inside fla's `FusedRMSNormGated`,
    // so the intermediate between them is not captured and no fixture here can show the order
    // directly. It is S2 item 5's to prove, on the KDA operator boundary. What is established
    // here is that the two paths are structurally different in the way §5 and §4 describe.
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
        "MLA gates with NO norm (trap 10); an o_norm on an MLA layer means the two paths have \
         converged and the trap is gone"
    );
}

/// **What the MLA goldens structurally cannot reach: real widths, a mask that masks, and
/// magnitudes where the softmax's stability device matters.**
///
/// Found by red-proofing, and both gaps were silent. Breaking the kernel two ways left every
/// golden-backed test above GREEN:
///
/// * **ignoring `mask` entirely.** The decode captures' masks are ALL ZERO — the last position
///   attends to everything, so causality masks nothing at a single decode step. §5's "causality is
///   unconditional" lives in the mask, and the fixture could not see it being dropped.
/// * **removing the max-subtraction** from the softmax. At the goldens' magnitudes `exp` never
///   comes close to overflowing, so the stability device is unobservable — until it is not.
///
/// The widths are the third gap, the same one item 1 hit: the goldens are 4 heads of 24/16 against
/// a real 96 of 192/128, and `kv` is 9 against a real context.
///
/// Scored against an f64 host oracle. That oracle is not an independent claim — the golden tests
/// above show the same arithmetic agreeing with the reference at the widths the reference produced.
#[test]
fn mha_attend_at_real_widths_masks_and_magnitudes() {
    let tol = k3_tolerance::rel_tolerance("mla_attend");
    let mut r = common::Lcg(0x11A5);
    // `(heads, kv, d, dv, masked, gain)`. 96/192/128 are the real model's. kv 1024 wraps the
    // per-position loop well past one block; kv 9 reproduces the goldens' shape. `gain` scales q
    // and k so the raw scores reach ~1e2, where `expf` without a max-subtraction overflows to inf
    // and the output becomes NaN — the case the goldens cannot produce.
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
        // triangle, because at one query row a causal mask IS all-zero — which is exactly how the
        // golden ended up unable to test this.
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
            let d_rel = rel(&got[h * dv..(h + 1) * dv], &want);
            assert!(
                got[h * dv..(h + 1) * dv].iter().all(|x| x.is_finite()),
                "heads={heads} kv={kv} d={d} gain={gain} head {h}: non-finite output — the \
                 softmax lost its max-subtraction"
            );
            assert!(
                d_rel <= tol,
                "heads={heads} kv={kv} d={d} dv={dv} masked={masked} gain={gain} head {h}: \
                 {d_rel:e} exceeds {tol:e}"
            );
        }
    }
}

// ================================ S2 item 3 — the latent sandwich ================================
//
// §6's order is `down(x) -> experts in latent space -> RMSNorm the AGGREGATE -> up(...)`, and the
// port's answer to it is three kernels this engine already has: `gemm_bf16` twice and one RMSNorm.
// So this section is mostly a question about WHICH RMSNorm, and about whether a matmul verified at
// vocab 1024 / dim 512 holds at 7168.

/// The captured layers that own a `block_sparse_moe`.
///
/// Layer 0 is absent and its absence is load-bearing: `first_k_dense_replace` is 1, so layer 0 is
/// dense and has no MoE block at all. Naming the set keeps that a statement rather than an accident
/// of what happened to be in the file, exactly as `FOLDS` does one operator up.
const MOE_LAYERS: [usize; 5] = [1, 3, 12, 91, 92];

/// One latent RMSNorm: the expert aggregate in, the learned weight, and what the reference made.
///
/// The aggregate is the capture S2 item 3 added — `moe_infer`'s return is not a module call, so no
/// forward hook could see it and this operator had an output with no input until 2026-08-12.
struct LatentNorm {
    x: Vec<f32>,
    w: Vec<f32>,
    want: Vec<f32>,
}

fn latent_norm(g: &GoldenSet, layer: usize) -> LatentNorm {
    let m = format!("model.layers.{layer}.block_sparse_moe.routed_expert_norm");
    let (ws, w) = float(g, &format!("{m}.weight"));
    let latent = ws[0];
    let (xs, x) = float(g, &format!("{m}.in"));
    let (os, want) = float(g, &m);
    assert_eq!(xs, [1, latent], "{m}: the aggregate is one row of latent");
    assert_eq!(os, [1, latent], "{m}: the norm is width-preserving");
    let [x, w, want] = [x, w, want].map(<[f32]>::to_vec);
    LatentNorm { x, w, want }
}

/// Every (draw, MoE layer) pair, with the reference's own eps in hand.
///
/// A closure rather than a Vec, for the reason `for_each_fold` gives: each golden's borrow has to
/// outlive the case it lends out.
fn for_each_latent_norm(mut f: impl FnMut(&str, usize, f32, LatentNorm)) {
    for (salt, bytes) in GOLDENS {
        let g = load(bytes);
        let e = eps(&g);
        for layer in MOE_LAYERS {
            f(salt, layer, e, latent_norm(&g, layer));
        }
    }
}

/// `KimiRMSNorm.forward` in f64 — `weight * (x * rsqrt(mean(x²) + eps))`.
///
/// The statistic is f64 so this is a floor rather than a second f32 implementation, and the eps
/// goes INSIDE the mean's square root, which is the placement `rmsnorm_single` uses and the one
/// trap this three-line operator has.
fn host_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64;
    let rs = 1.0 / (ms + f64::from(eps)).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &q)| (f64::from(v) * rs * f64::from(q)) as f32)
        .collect()
}

/// `y[j] = Σ_i x[i]·bf16(w[j][i])` in f64 — `gemm_bf16`'s oracle at `m == 1`.
///
/// The weights are handed in ALREADY bf16-coded and widened here, so the only thing separating this
/// from the kernel is the summation order: the kernel's `wave_sum` shuffle ladder against this
/// sequential f64 accumulation. Rounding the weights inside the oracle instead would let a kernel
/// that silently read them as f32 agree with it.
fn host_gemv(x: &[f32], w: &[u16], n: usize, k: usize) -> Vec<f32> {
    sums64(n, |j| {
        (0..k)
            .map(|i| f64::from(x[i]) * f64::from(rivoli::math::bf16_to_f32(w[j * k + i])))
            .sum()
    })
}

fn device_rmsnorm_single(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let (xb, wb) = (dev(&f32b(x)), dev(&f32b(w)));
    let mut yb = zeros(x.len() * 4);
    // SAFETY: `x`, `w` and `y` are each `n` live f32 and mutually non-aliasing, as required.
    ok(
        unsafe {
            launch_rmsnorm_single(
                xb.ptr() as *const f32,
                wb.ptr() as *const f32,
                x.len(),
                eps,
                yb.ptr_mut() as *mut f32,
            )
        },
        "rmsnorm_single",
    );
    f32v(&back(&yb))
}

/// `out[j] = Σ_i x[i]·bf16(w[j][i])` on the device, at `m == 1` — one decode row.
fn device_gemv(x: &[f32], w: &[u16], n: usize) -> Vec<f32> {
    let (xb, wb) = (dev(&f32b(x)), dev(&common::u16b(w)));
    let mut ob = zeros(n * 4);
    // SAFETY: `x` is `k` live f32, `w` is `n·k` live u16, `out` is `n` writable f32, none aliasing;
    // all outlive the default stream's completion, which `back` synchronises on.
    ok(
        unsafe {
            launch_gemm_bf16(
                xb.ptr() as *const f32,
                wb.ptr() as *const u16,
                ob.ptr_mut() as *mut f32,
                1,
                n,
                x.len(),
                std::ptr::null_mut(),
            )
        },
        "gemm_bf16",
    );
    f32v(&back(&ob))
}

/// **`rmsnorm_single` reproduces the latent norm at every MoE layer of both draws.**
///
/// `rmsnorm_single` and not `rmsnorm_batch`, and the choice is the content of this item — see
/// `the_batch_rmsnorm_would_fail_this_fixture`. At decode there is one row, which is the case
/// `dim3(1)` computes correctly; prefill is S3's and needs a row-wise kernel that does NOT
/// bf16-round, which is neither of the two this engine has today.
#[test]
fn the_latent_norm_matches_the_anchor_at_every_moe_layer() {
    let tol = k3_tolerance::rel_tolerance("moe_latent");
    for_each_latent_norm(|salt, layer, eps, ln| {
        let r = rel(&device_rmsnorm_single(&ln.x, &ln.w, eps), &ln.want);
        assert!(r <= tol, "{salt} layer {layer}: {r:e} exceeds {tol:e}");
        // The regression tripwire item 1 carries and item 2 was given after review. The operator
        // tolerance is a whole-model floor carrying upstream drift; fed the reference's own
        // aggregate, this kernel does far better, and against `tol` alone a THREE-order degradation
        // would pass in silence — `tol` is 6.3e-4 and this kernel lands at 1.3e-7. Measured worst
        // over both draws and all five layers, then 10x. Three of the ten cells are BIT-EXACT.
        tripwire(r, 1.307e-7, tol, &format!("{salt} layer {layer}"));
    });
}

/// **`rmsnorm_batch` would fail the fixture above, and that is why this item does not use it.**
///
/// `docs/investigations/k3-port.md` named `mla.hip:346` for this operator on the grounds that it is
/// already width-generic. It is — and the width was never the problem. Its last line is
/// `row[i] = rbf16(w[i] * (row[i] * rs))`: it rounds its store to bf16, because V4's
/// `RMSNorm.forward` stores bf16 and that kernel is V4's. `KimiRMSNorm.forward` is
/// `self.weight * x.to(dtype)`, and in this fp32 reference `to(dtype)` is a no-op — so the whole
/// bf16 step is arithmetic the reference does not perform. Measured: **3.299e-3 against the
/// 6.3e-4 tolerance, 11.4x over** — this is not a marginal call.
///
/// Asserted as a FAILURE rather than left as a comment: a claim that one of two interchangeable
/// kernels is wrong is exactly the claim that rots, and this one goes red the day someone changes
/// either kernel's store.
///
/// **The obvious rescue does not work, and it is worth saying why.** A fixture can absorb a store
/// deviation by rounding the REFERENCE's output the same way — score `rbf16(want)` instead of
/// `want` — and that would let `rmsnorm_batch` pass. It would also be wrong here. `KimiRMSNorm`
/// rounds to the input dtype BEFORE the weight multiply (`self.weight * x.to(dtype)`), so the real
/// bf16 model computes `w · bf16(x·rs)` while this kernel computes `bf16(w · x·rs)` — a different
/// function, not a differently-placed copy of the same one. Rounding the anchor's output would
/// reproduce neither. rivoli's trunk carries f32 activations throughout, which is the engine-wide
/// deviation this port inherits, and `rmsnorm_single`'s f32 store is what matches it.
#[test]
fn the_batch_rmsnorm_would_fail_this_fixture() {
    let tol = k3_tolerance::rel_tolerance("moe_latent");
    let mut worst = 0.0f32;
    for_each_latent_norm(|_, _, eps, ln| {
        let mut xb = dev(&f32b(&ln.x));
        let wb = dev(&f32b(&ln.w));
        // SAFETY: `x` is `rows·d` live f32 written in place, `w` is `d` live f32; they do not alias
        // and both outlive the default stream, which `back` synchronises on.
        ok(
            unsafe {
                launch_rmsnorm_batch(
                    xb.ptr_mut() as *mut f32,
                    wb.ptr() as *const f32,
                    1,
                    ln.x.len(),
                    eps,
                    std::ptr::null_mut(),
                )
            },
            "rmsnorm_batch",
        );
        worst = worst.max(rel(&f32v(&back(&xb)), &ln.want));
    });
    assert!(
        worst > tol,
        "`rmsnorm_batch` scored {worst:e}, INSIDE the {tol:e} this operator is held to. Its bf16 \
         store was the whole reason item 3 uses `rmsnorm_single` instead. If the store has been \
         made optional, use it here and delete this test rather than loosening it."
    );
}

/// **`gemm_bf16` at K3's real trunk widths, against an f64 dot on the same bf16 weights.**
///
/// `gemm_bf16` carries S2 and S3 unchanged (`k3-port.md` item 7) but is verified only at vocab 1024
/// / dim 512, and the latent sandwich runs it at `7168 -> 3584` and `3584 -> 7168`. It has no
/// anchor bucket and cannot get one: the anchor is fp32 (one of its four declared deviations) while
/// these weights are bf16, so an anchor comparison would be dominated by a ~2^-9 quantisation the
/// reference never applied. What is left to check is the part that is genuinely the kernel's — the
/// `wave_sum` shuffle ladder re-associating a 7168-term sum — and that is what this scores.
///
/// **The bound is this test's own, NOT a `k3_tolerance` row**, and it is stated here rather than
/// added to that table because that table's numbers all derive from the anchor's floor-vs-defect
/// pair. Measured worst over these cases, then 10x.
#[test]
fn the_trunk_gemv_matches_an_f64_dot_at_k3_widths() {
    // Measured over the four cases below, worst at the deepest reduction (n=3584, k=7168) as the
    // error model predicts, then 10x. An f32 accumulator over 7168 terms against an f64 one is the
    // only difference here, so this is small on purpose: a number in the 1e-3 range would mean the
    // kernel is reading the weights as something other than bf16, not that it re-associated.
    const OBSERVED_WORST: f32 = 2.705e-7;
    let mut r = common::Lcg(0x3EA7);
    // `(n, k)`: the sandwich's two projections at the real widths, then the tiny model's pair so a
    // failure that is about the width shows up as one.
    for &(n, k) in &[(3584usize, 7168usize), (7168, 3584), (96, 192), (192, 96)] {
        let x: Vec<f32> = (0..k).map(|_| r.f()).collect();
        let w: Vec<u16> = (0..n * k)
            .map(|_| rivoli::math::f32_to_bf16(r.f()))
            .collect();
        let got = device_gemv(&x, &w, n);
        let d = rel(&got, &host_gemv(&x, &w, n, k));
        assert!(
            d <= OBSERVED_WORST * 10.0,
            "n={n} k={k}: {d:e} exceeds {:e} — an f32 accumulator over {k} terms against an f64 \
             one should not drift this far, so the re-association is not the explanation",
            OBSERVED_WORST * 10.0
        );
    }
}

/// **The norm goes BEFORE the up projection, and doing it after is a different function.**
///
/// The fixture-level twin of `--defect LatentNormAfterUp`, which the anchor prices at 2.05e+2
/// against a 6.287e-5 floor. Built the same way the defect is — the norm's weight collapsed to its
/// own mean so it is applicable at `hidden` width — so this asks about the ORDER and not about the
/// values. Synthetic, because the projection weights are deliberately not in the goldens
/// (`wrap_latent_sandwich` argues the cost); what is being pinned is which of two orders the
/// device chain implements, and that does not need the reference's particular matrix.
#[test]
fn norming_after_the_up_projection_is_a_different_sandwich() {
    let (latent, hidden) = (3584usize, 7168usize);
    let eps = 1e-5f32;
    let mut r = common::Lcg(0x5A9D);
    let acc: Vec<f32> = (0..latent).map(|_| r.f()).collect();
    // `uniform(0.8, 1.2)`, the range `init_weights` draws every norm weight from — `r.f()` is
    // [-1, 1), not [0, 1). A norm weight near zero would make every downstream value a denormal and
    // the fixture a comparison of noise, which is the driver's own argument for that range.
    let nw: Vec<f32> = (0..latent).map(|_| 1.0 + 0.2 * r.f()).collect();
    let up: Vec<u16> = (0..hidden * latent)
        .map(|_| rivoli::math::f32_to_bf16(r.f()))
        .collect();

    let ordered = device_gemv(&device_rmsnorm_single(&acc, &nw, eps), &up, hidden);
    // The device chain is the specification's order, so the host oracle of that order must agree.
    let want = host_gemv(&host_rmsnorm(&acc, &nw, eps), &up, hidden, latent);
    let good = rel(&ordered, &want);
    // 10x the measured 2.35e-7. Two kernels deep — the norm's f32 statistic and the GEMV's f32
    // accumulation over 3584 terms — so it is looser than either alone and still three orders under
    // the separation asserted below.
    assert!(
        good <= 2.35e-6,
        "the specified order disagrees with its own oracle: {good:e}"
    );

    // Norm-after-up, at `hidden` width with the mean weight — the defect's own construction.
    //
    // **What this second half is and is not.** It exists so the first assertion is not vacuous: a
    // `good` of 1e-6 means nothing unless the wrong order is somewhere else entirely. It is NOT a
    // sensitivity measurement, and the separation below is not evidence of a delicate fixture — the
    // two orders differ mostly in SCALE, because norming last leaves an output of magnitude ~1
    // where norming first leaves the projection's own ~35. That is also why the anchor prices this
    // defect at 2.05e+2 rather than at something subtle. A fixture that could only tell these apart
    // by their scale would still catch the swap, and no claim beyond that is being made here.
    let mean = nw.iter().sum::<f32>() / latent as f32;
    let flipped = host_rmsnorm(
        &host_gemv(&acc, &up, hidden, latent),
        &vec![mean; hidden],
        eps,
    );
    let moved = rel(&flipped, &want);
    assert!(
        moved > 1.0e-2,
        "norming after the up projection moved the output by only {moved:e}, so the two orders are \
         not separated here and the agreement above proves nothing"
    );
}

// ============================== S2 item 4a — SiTU-GLU ==============================
//
// **The first item whose fixture needed no regeneration.** `SituAndMul` is an `nn.Module`, so
// `hook_model` was already capturing its output as `<mlp>.act_fn`, and its input is
// `torch.cat([gate_proj(x), up_proj(x)])` — both halves separately captured. Inputs and output were
// in the file all along. Recorded because the four items before this one each found the opposite,
// and "check what is already there" is cheaper than a 25-minute GPU-locked regeneration.

/// Every MLP the anchor captures that runs SiTU-GLU, as `(layer, module)`.
///
/// Layer 0's is the DENSE one — `first_k_dense_replace` is 1, so it is the only `mlp` in the file —
/// and the other five are the shared experts, which every MoE layer has. **The routed experts are
/// deliberately absent and cannot be added**: `moe_infer` calls only the experts that won tokens, so
/// which expert modules fire is routing-dependent and any defect that moved the routing would
/// change the golden's tensor SET rather than its numbers. That is the gap `k3-port.md` item 4b
/// closes a different way — the fp4 kernel is scored against a host oracle, not against this file.
/// **The third field is the tolerance bucket, and it is NOT the same for all six.**
/// `k3_anchor_driver.py::operator_of` buckets `model.layers.N.mlp.*` as `dense_mlp` and everything
/// under `block_sparse_moe` that is not `routed_expert*` as `moe_route` — so the shared experts are
/// scored against the ROUTER's tolerance. That is a classification artifact rather than a judgement:
/// the shared expert is an MLP and the router is a router, and they share a bucket only because the
/// name prefix does not separate them. Flagged rather than worked around, because the fix belongs
/// with item 6 (the router), which is the other occupant. Scoring all six against `dense_mlp`'s
/// tighter 9.4e-6 would pass — this fixture lands two orders under either — but it would be
/// asserting a bar the anchor does not set for five of them.
const SITU_MLPS: [(usize, &str, &str); 6] = [
    (0, "mlp", "dense_mlp"),
    (1, "block_sparse_moe.shared_experts", "moe_route"),
    (3, "block_sparse_moe.shared_experts", "moe_route"),
    (12, "block_sparse_moe.shared_experts", "moe_route"),
    (91, "block_sparse_moe.shared_experts", "moe_route"),
    (92, "block_sparse_moe.shared_experts", "moe_route"),
];

/// The worst relative difference `situ_glu_f32` shows against the anchor, over both draws and all
/// six MLPs: salt 2, layer 12. Named rather than inlined because two tests need it — the fixture
/// as its tripwire, and the defect run as the bar a defect has to clear.
const SITU_OBSERVED_WORST: f32 = 1.454e-7;

/// The two SiTU betas, read off the golden's own `tiny_config`.
///
/// Not literals. `k3_anchor.rs::the_tiny_configs_kept_the_real_structure` pins both against the real
/// checkpoint's — they are in `STRUCTURAL`, so the tiny config cannot have shrunk them — and a
/// fixture that hardcoded 4 and 25 would agree with itself if either ever moved. The second key is
/// `activation_situ_linear_beta`; abbreviating it to `activation_linear_beta` is a mistake this port
/// has already made once, in `S1a`, where it would have refused every real checkpoint.
fn betas(g: &GoldenSet) -> (f32, f32) {
    let c: serde_json::Value =
        serde_json::from_str(g.meta_get("tiny_config").expect("tiny_config")).unwrap();
    let f = |k: &str| c[k].as_f64().unwrap_or_else(|| panic!("{k} missing")) as f32;
    (f("activation_situ_beta"), f("activation_situ_linear_beta"))
}

/// One MLP's activation boundary: the two projections in, what `SituAndMul` made of them.
struct Situ {
    gate: Vec<f32>,
    up: Vec<f32>,
    want: Vec<f32>,
}

fn situ(g: &GoldenSet, layer: usize, module: &str) -> Situ {
    let m = format!("model.layers.{layer}.{module}");
    let (gs, gate) = float(g, &format!("{m}.gate_proj"));
    let (us, up) = float(g, &format!("{m}.up_proj"));
    let (os, want) = float(g, &format!("{m}.act_fn"));
    assert_eq!(gs, us, "{m}: the two projections are the same width");
    assert_eq!(
        os, gs,
        "{m}: SiTU-GLU is width-preserving over one half of its input"
    );
    let [gate, up, want] = [gate, up, want].map(<[f32]>::to_vec);
    Situ { gate, up, want }
}

fn for_each_situ(mut f: impl FnMut(&str, usize, f32, (f32, f32), Situ)) {
    for (salt, bytes) in GOLDENS {
        let g = load(bytes);
        let b = betas(&g);
        for (layer, module, operator) in SITU_MLPS {
            f(
                salt,
                layer,
                k3_tolerance::rel_tolerance(operator),
                b,
                situ(&g, layer, module),
            );
        }
    }
}

/// `SituAndMul.forward` in f64 — `(b1·tanh(g/b1)·sigmoid(g)) · (b2·tanh(u/b2))`.
///
/// The sigmoid takes `g`, NOT `b1·tanh(g/b1)`: the two factors saturate at different rates and
/// feeding the capped value to the sigmoid is the smooth, plausible, wrong version. `situ_glu_gets`
/// -`_the_uncapped_gate` is the run that prices it.
fn host_situ(s: &Situ, b1: f32, b2: f32, capped_sigmoid: bool) -> Vec<f32> {
    let (b1, b2) = (f64::from(b1), f64::from(b2));
    s.gate
        .iter()
        .zip(&s.up)
        .map(|(&g, &u)| {
            let (g, u) = (f64::from(g), f64::from(u));
            let t = b1 * (g / b1).tanh();
            // The ONE flag, and it is the defect. `host_fold` one operator up takes its
            // `normalised` the same way and for the same reason: a defect run is the correct
            // oracle with one thing changed, and writing it as a second function is how the two
            // drift into differing by something nobody intended.
            let sig = if capped_sigmoid { t } else { g };
            (t * (1.0 / (1.0 + (-sig).exp())) * (b2 * (u / b2).tanh())) as f32
        })
        .collect()
}

/// One launch, returning the launcher's own `Result`.
///
/// Both the scoring path and the guard test go through here — they were two copies of the same
/// eight-argument unsafe block and jscpd rejected the second at 42 tokens. It is also the shape
/// that makes the guard test meaningful: it exercises the entry point callers use, not a
/// second spelling of it.
fn situ_launch(gate: &[f32], up: &[f32], b1: f32, b2: f32) -> anyhow::Result<Vec<f32>> {
    let (gb, ub) = (dev(&f32b(gate)), dev(&f32b(up)));
    let mut hb = zeros(gate.len() * 4);
    // SAFETY: `g`, `u` and `h` are each `n` live f32 and do not alias here (the launcher permits
    // aliasing; this fixture does not use it). All outlive the default stream, which `back` syncs.
    unsafe {
        launch_situ_glu_f32(
            gb.ptr() as *const f32,
            ub.ptr() as *const f32,
            gate.len(),
            b1,
            b2,
            hb.ptr_mut() as *mut f32,
            std::ptr::null_mut(),
        )
    }?;
    Ok(f32v(&back(&hb)))
}

fn device_situ(gate: &[f32], up: &[f32], b1: f32, b2: f32) -> Vec<f32> {
    ok(situ_launch(gate, up, b1, b2), "situ_glu_f32")
}

/// **The kernel reproduces every SiTU-GLU the anchor captured, at both draws.**
#[test]
fn situ_glu_matches_the_anchor_at_every_mlp() {
    for_each_situ(|salt, layer, tol, (b1, b2), s| {
        let r = rel(&device_situ(&s.gate, &s.up, b1, b2), &s.want);
        assert!(r <= tol, "{salt} layer {layer}: {r:e} exceeds {tol:e}");
        // Measured worst over both draws and all six MLPs: 1.454e-7, at salt 2 layer 12. That is
        // 65x under `dense_mlp`'s 9.4e-6 — the tightest tolerance in the table, and the one that
        // binds the only cell here the anchor actually buckets that way.
        tripwire(
            r,
            SITU_OBSERVED_WORST,
            tol,
            &format!("{salt} layer {layer}"),
        );
    });
}

/// **The sigmoid takes the UNCAPPED gate, and the capped version is a different function.**
///
/// The one trap in three lines of arithmetic. `a = b1·tanh(g/b1)·sigmoid(g)`: the first factor
/// saturates at `±b1`, the second at `0`/`1`, and they saturate on different scales. Feeding
/// `b1·tanh(g/b1)` to the sigmoid instead is smooth, monotone, bounded and wrong — and at the
/// shipped `b1 = 4` it agrees to three figures near zero, which is where a spot check looks.
///
/// Scored on the anchor's own values rather than synthetically, so the separation is the one the
/// real activations produce and not one a chosen input manufactures.
#[test]
fn the_situ_sigmoid_takes_the_uncapped_gate() {
    // Per SITE — a single worst-over-all would be carried by whichever MLP separates best and
    // would say nothing about the other five. Scored against the bar this fixture ENFORCES, the
    // tripwire, rather than against the operator tolerance; `DEFECT_MARGIN`'s 30x is the table's
    // own rule for "a bar must clear the defect it catches", applied to the bar in use.
    //
    // > **And the distinction is not academic here — it is the finding of this test.** At layer 0's
    // > dense MLP the separation is 3.38e-2 against `dense_mlp`'s 9.4e-6, a margin of 3,600x, so
    // > the bucket tolerance catches this comfortably. At the SHARED experts it is 4.10e-3 against
    // > `moe_route`'s 6.0e-4 — only **6.8x**, well under the 30x the table requires of a `Rel`
    // > policy. So **the bucket-level tolerance could not be relied on to catch a capped-sigmoid
    // > SiTU at the shared experts**; the tripwire can, by 2,800x. That was true before this item
    // > loosened `moe_route` (at the old 2.5e-4 the margin was 16x, still under 30) and it is a
    // > property of the shared expert's small `moe_intermediate_size`, not of the loosening.
    for_each_situ(|salt, layer, tol, (b1, b2), s| {
        let moved = rel(&host_situ(&s, b1, b2, true), &s.want);
        let bar = SITU_OBSERVED_WORST * 10.0;
        assert!(
            moved > bar * 30.0,
            "{salt} layer {layer}: capping the sigmoid's argument moved the activation by only \
             {moved:e}, under the {:e} this fixture would need to clear its own {bar:e} tripwire \
             by the table's 30x — so it does not price the one trap SiTU-GLU has, and the \
             agreement above says nothing about which form the kernel implements. (The operator \
             tolerance here is {tol:e}.)",
            bar * 30.0
        );
    });
}

/// **SiTU-GLU at K3's real widths, and at magnitudes the goldens cannot reach.**
///
/// Three gaps the anchor leaves. Width: `moe_intermediate_size` is 24 here against a real 3072, and
/// `intermediate_size` 256 against 33792. Magnitude: these activations are ~1, so neither `tanh`
/// saturates and `expf(-g)` never overflows — the whole point of the two betas is what happens when
/// they do. And the BOUND: `|y| <= b1·b2 = 100` is the property §8 states, and nothing in the
/// goldens comes near it.
#[test]
fn situ_glu_saturates_at_the_product_of_its_betas() {
    let (b1, b2) = (4.0f32, 25.0f32);
    let mut r = common::Lcg(0x517A);
    for &(n, gain) in &[
        (3072usize, 1.0f32),
        (33792, 1.0),
        (3072, 40.0),
        (3072, 400.0),
        (1, 1.0),
    ] {
        let gate: Vec<f32> = (0..n).map(|_| r.f() * gain).collect();
        let up: Vec<f32> = (0..n).map(|_| r.f() * gain).collect();
        let s = Situ {
            gate: gate.clone(),
            up: up.clone(),
            want: Vec::new(),
        };
        let got = device_situ(&gate, &up, b1, b2);
        let d = rel(&got, &host_situ(&s, b1, b2, false));
        // 10x the 1.721e-7 measured over these cases. `tanhf` and `expf` are the device's own
        // against Rust's `f64` ones, so this is the one fixture here whose bound is a libm
        // difference rather than a reassociated sum.
        //
        // The `gain = 400` case measures SMALLER (7.6e-8), not larger: both `tanh`s are hard
        // against ±1 there and the sigmoid against 0 or 1, so the saturated regime is the easy one
        // and the interesting magnitudes are the ones in between. Stated because "we tested the
        // extreme" reads as coverage and here it is the opposite.
        assert!(d <= 1.72e-6, "n={n} gain={gain}: {d:e}");
        assert!(
            got.iter().all(|y| y.abs() <= b1 * b2),
            "n={n} gain={gain}: SiTU-GLU exceeded |b1·b2| = {}, which §8 states as a property of \
             the function rather than of its inputs",
            b1 * b2
        );
    }
}

/// **Both betas must be finite and positive — every other value is refused, not clamped.**
///
/// Four failure modes, each quiet in its own way, argued at the launcher. This is the only test in
/// this file that exercises a refusal code, and it exists because the other three items each left
/// their guards untested and said so.
#[test]
fn the_situ_betas_are_guarded() {
    for (b1, b2, what) in [
        (0.0f32, 25.0f32, "b1 = 0"),
        (4.0, 0.0, "b2 = 0"),
        (-4.0, 25.0, "b1 negative"),
        (f32::NAN, 25.0, "b1 NaN"),
        (4.0, f32::NAN, "b2 NaN"),
        (f32::INFINITY, 25.0, "b1 +inf"),
        (4.0, f32::INFINITY, "b2 +inf"),
    ] {
        assert!(
            situ_launch(&[1.0, 2.0], &[3.0, 4.0], b1, b2).is_err(),
            "{what} was accepted"
        );
    }
    // And the guard is not simply refusing everything, which is the way a refusal test passes
    // vacuously. The shipped pair must go through.
    assert_eq!(device_situ(&[1.0, 2.0], &[3.0, 4.0], 4.0, 25.0).len(), 2);
}
