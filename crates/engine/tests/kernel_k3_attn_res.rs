//! **Kimi-K3's Block Attention Residual fold on the device, scored against the S2 anchor.**
//! `attn_res` and nothing else. Ported from `k3:tests/k3_kernels.rs` item 1 (its `// ====`
//! banner at :69); the shared spine is `tests/k3/mod.rs`, which also carries the argument for
//! the second vendored anchor pair this file reads.
//!
//! Kimi-K3 does not have a plain residual. It keeps a stack of snapshots taken at block
//! boundaries plus a running prefix sum, and forms each module's input as a softmax-weighted
//! mixture over that stack (`k3:docs/reference/k3-architecture.md` §3). What this suite
//! promises is "the kernel computes the fold correctly", not "AttnRes is right": the fixture
//! is handed one already-assembled `[nsrc][hidden]` snapshot set, so it says nothing about
//! whether the layer loop pushes at the right boundary, resets `prefix_sum` when it should,
//! or keeps the layer-0 snapshot alive for the model-level fold — those are the layer loop's,
//! and §3 is the specification they owe (`k3:tests/k3_kernels.rs:27`).
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! This suite has never executed: no PR-triggered rocm CI arm, and the author of this port
//! had no GPU. Before trusting a green run, make it go red. One mutation in
//! `kernels/residual.hip`, expected magnitudes below it:
//!
//! * In `rivoli_attn_res`'s score loop, multiply each source's contribution to the score by
//!   its RMSNorm scale a SECOND time (i.e. mix the normalised sources — the
//!   `AttnResNormalisedValues` defect §3 spends its clearest sentence on).
//!   [`attn_res_matches_the_anchor_at_every_fold`] must go RED with a relative difference in
//!   the 1.8e0 region (the anchor prices this defect at 1.796e0 at its weakest site), and
//!   [`mixing_the_normalised_sources_goes_red`] must stay GREEN — it perturbs the HOST oracle
//!   the same way, so the kernel now AGREES with its defect arm. A mutation that reddens both
//!   has changed something other than which vector the mixture reads.
//! * Drop the `i += blockDim.x` stride (compute only each thread's first column).
//!   [`attn_res_at_real_widths_and_multiple_tokens`] must go red at every `n > 256` cell
//!   while every 192-wide golden-backed fold stays green — the goldens structurally cannot
//!   reach a second loop trip, which is that sweep's whole reason to exist.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_backend::hip::launch_attn_res;

mod common;
mod k3;

use k3::*;

/// Every fold the anchor captures: two per captured layer, plus the model-level one.
///
/// Layer 0's `self_attention_res` is deliberately absent and its absence is load-bearing —
/// §3's layer loop guards the layer-level fold on a NON-EMPTY block stack, and at layer 0
/// nothing has been pushed yet. A fixture listing it would fail to find it, which is the
/// correct outcome for the wrong reason; naming the set explicitly is how that stays a
/// statement rather than an accident of what happened to be in the file
/// (`k3:tests/k3_kernels.rs:88`).
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

/// The worst the real-width sweep measures against its f64 oracle, over every cell — two
/// correct implementations disagreeing, so three orders under the 7.1e-4 operator tolerance
/// the sweep used to be gated on alone (`k3:tests/k3_kernels.rs:325`).
const ATTN_RES_SWEEP_WORST: f32 = 6.775e-8;

/// One fold: the assembled stack, its scoring vector, and the output the reference produced.
///
/// Built by one function because two tests open with the same six lookups, and because a
/// fold IS these four things together — a test that assembled three of them would be scoring
/// something the reference never computed (`k3:tests/k3_kernels.rs:135`).
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
/// (`torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)`) and it is not arbitrary:
/// the softmax is permutation-equivariant, so a port that concatenated the other way round
/// produces exactly the same output and is nonetheless wrong the moment anything indexes the
/// stack — which the layer loop does, when it pushes onto it. Reversing it here does not go
/// red, and that is precisely why this comment exists instead of a test
/// (`k3:tests/k3_kernels.rs:168`).
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

/// One AttnRes fold's inputs, bundled: `host_probs` and `host_fold` take the same five
/// things, and a duplicate parameter list is a clone (`k3:tests/k3_kernels.rs:200`).
struct FoldIn<'a> {
    src: &'a [f32],
    nsrc: usize,
    n: usize,
    fold: &'a [f32],
    eps: f32,
}

/// The reference's scoring pass, transliterated, in `f64` — the softmax probabilities and
/// each source's RMSNorm scale.
///
/// `f64` because the reference accumulates every long reduction in double, so this
/// reproduces the reference's own arithmetic rather than a second fp32 implementation of it;
/// scoring the kernel against an fp32 host would measure two fp32 implementations
/// disagreeing, a different and much smaller number than the one `attn_res`'s tolerance was
/// measured for. Split from [`host_fold`] because the width sweep needs the probabilities
/// alone for its anti-vacuity check (`k3:tests/k3_kernels.rs:211`).
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

/// The fold itself. `normalised` is the `AttnResNormalisedValues` defect body: mix the
/// SCORED vector instead of the raw one — one substitution, the same shape as the anchor
/// driver's own defect run, so the two agree on what the defect is. Cast ONCE at the end:
/// the f64-throughout promise above is only honoured if the mixture accumulates in f64 too
/// (`k3:tests/k3_kernels.rs:247`).
fn host_fold(f: &FoldIn, normalised: bool) -> Vec<f32> {
    let (src, nsrc, n) = (f.src, f.nsrc, f.n);
    let (p, inv) = host_probs(f);
    let mut out = vec![0.0f64; n];
    for s in 0..nsrc {
        let v = &src[s * n..(s + 1) * n];
        let scale = if normalised { inv[s] } else { 1.0 };
        for (i, o) in out.iter_mut().enumerate() {
            *o += p[s] * f64::from(v[i]) * scale;
        }
    }
    out.into_iter().map(|x| x as f32).collect()
}

/// The suite's one spelling of its operator row: the tolerance is a property of the
/// OPERATOR, so every site that scores `attn_res` reads THIS and two sites cannot drift
/// on which row is the suite's (`k3:tests/k3_kernels.rs:428`).
fn attn_res_tol() -> f32 {
    tolerance::rel_tolerance("attn_res")
}

/// One scoring site's whole context.
struct Fold<'a> {
    salt: &'a str,
    tag: &'a str,
    eps: f32,
    c: Case,
}

/// Every (draw, fold) pair, with its case assembled and the reference's eps in hand.
///
/// A closure rather than a returned Vec keeps each golden's borrow alive exactly as long as
/// the cases it lends out (`k3:tests/k3_kernels.rs:405`).
fn for_each_fold(mut f: impl FnMut(Fold)) {
    for (salt, bytes) in GOLDENS {
        let g = load(bytes);
        let eps = eps(&g);
        for tag in FOLDS {
            f(Fold {
                salt,
                tag,
                eps,
                c: case(&g, tag),
            });
        }
    }
}

/// One launch over an already-flattened `[tokens][nsrc][n]` arena, returning the launcher's
/// own `Result` so the guard test drives the entry point callers use. Takes the pieces
/// rather than a `Fold` because the width sweep has no golden and therefore no `Fold`
/// (`k3:tests/k3_kernels.rs:448`).
struct Arena<'a> {
    src: &'a [f32],
    fold: &'a [f32],
    tokens: usize,
    nsrc: usize,
    n: usize,
    stride: usize,
    eps: f32,
}

fn attn_res_launch(a: &Arena) -> anyhow::Result<Vec<f32>> {
    let (sb, fb) = (dev(&f32b(a.src)), dev(&f32b(a.fold)));
    let mut ob = zeros(a.tokens.max(1) * a.n.max(1) * 4);
    // SAFETY: `src` is `tokens·stride` f32 with `stride >= nsrc·n`, `fold` is `n` f32 and
    // `out` is `tokens·n` f32, all live for the call and mutually non-aliasing, as the
    // launcher requires; `back` synchronises before any buffer drops.
    unsafe {
        launch_attn_res(
            sb.ptr() as *const f32,
            fb.ptr() as *const f32,
            a.tokens,
            a.nsrc,
            a.n,
            a.stride,
            a.eps,
            ob.ptr_mut() as *mut f32,
        )
    }?;
    let out = f32v(&back(&ob));
    Ok(out)
}

fn device(a: &Arena) -> Vec<f32> {
    ok(attn_res_launch(a), "attn_res")
}

fn fold_arena<'a>(f: &'a Fold) -> Arena<'a> {
    Arena {
        src: &f.c.src,
        fold: &f.c.fold,
        tokens: 1,
        nsrc: f.c.nsrc,
        n: f.c.n,
        stride: f.c.nsrc * f.c.n,
        eps: f.eps,
    }
}

/// The vendored bytes are the ones the port measured its tripwires on. One suite owns this
/// check — seven copies of the loop would be seven identical tests (`k3/mod.rs::vendored`).
#[test]
fn the_vendored_anchors_match_their_pins() {
    for v in k3::vendored() {
        v.check_bytes();
    }
}

/// The kernel table's rows each follow the 10x-floor / 30x-defect rule — the gate on the
/// table, run here because this binary is the first consumer of its loosest row. Widening
/// `mla` to a `Rel` fails it, which is the point (`tests/k3/tolerance.rs`).
#[test]
fn the_kernel_tolerance_rows_follow_their_rule() {
    tolerance::rows_follow_the_rule();
}

/// **The kernel reproduces every fold the anchor captured, at both draws.**
#[test]
fn attn_res_matches_the_anchor_at_every_fold() {
    let tol = attn_res_tol();
    for_each_fold(|f| {
        let got = device(&fold_arena(&f));
        // Through `score_all`, so the two bars land in the factored order. The kernel
        // agrees with the golden to 3.08e-7 worst over both draws (several folds are
        // bit-exact) — ~230x BELOW `attn_res`'s own fp32 floor of 7.052e-5 and ~2,300x
        // below the 7.1e-4 tolerance. Not luck: the floor was measured on whole-model
        // fp32-vs-fp64 runs, so it carries upstream drift, while this fixture hands the
        // kernel the reference's OWN inputs and measures the operator alone
        // (`k3:tests/k3_kernels.rs:501`).
        score_all(
            &format!("{} {} (nsrc={}, hidden={})", f.salt, f.tag, f.c.nsrc, f.c.n),
            Bars {
                tol,
                observed: 3.08e-7,
            },
            &[("out", &got, &f.c.want)],
        );
    });
}

/// **Every fold mixes the stack depth §3's layer loop says it should.**
///
/// Without this the suite above passes on a kernel that ignores `nsrc` and returns source 0
/// — most folds here are the two-source case, the smallest mixture there is. The depth is
/// DERIVED from the block size rather than listed, which makes this a test of the
/// reference's bookkeeping instead of a transcription of it: §3's loop folds BEFORE it
/// pushes, so at a boundary layer the `self_attention_res` fold still sees the old stack and
/// the `mlp_res` fold sees the new one — layer 12 mixes 2 sources and then 3. Layers 91 and
/// 92 carry all eight snapshots plus the prefix sum: **nine sources**, the width
/// `[T][9][7168]` is named for (`k3:tests/k3_kernels.rs:528`).
#[test]
fn every_fold_mixes_the_depth_the_layer_loop_implies() {
    let block = 12; // `attn_res_block_size`; pinned against the real config by the anchor gate
    for_each_fold(|f| {
        let want = if f.tag == "model.output_attn_res" {
            // The model-level fold runs after every layer, so every boundary has fired.
            93_usize.div_ceil(block) + 1
        } else {
            let layer: usize = f.tag.split('.').nth(2).unwrap().parse().unwrap();
            // Snapshots pushed strictly BEFORE this fold, plus this layer's own push when it
            // is a boundary and we are past the attention fold.
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
/// The second half of the operator gate, and what separates a fixture from a formality —
/// `--defect AttnResNormalisedValues` is the trap §3 spends its clearest sentence on ("the
/// softmax mixes the UNNORMALISED sources"), and a fixture that could not tell the two apart
/// would be green on the single most likely misreading of the operator. Computed on the HOST
/// because the defect goldens are not vendored — only the clean ones are — and a test that
/// needs a regeneration directory is a test that passes by being skipped. The clean host
/// oracle is asserted FIRST: without it, "the defect disagrees with the golden" would also
/// be satisfied by a host transliteration that disagrees with everything
/// (`k3:tests/k3_kernels.rs:565`).
#[test]
fn mixing_the_normalised_sources_goes_red() {
    let tol = attn_res_tol();
    for_each_fold(|f| {
        let c = &f.c;
        let fin = FoldIn {
            src: &c.src,
            nsrc: c.nsrc,
            n: c.n,
            fold: &c.fold,
            eps: f.eps,
        };
        let clean = rel(&host_fold(&fin, false), &c.want);
        assert!(
            clean <= tol,
            "{} {}: the host oracle itself is {clean:e} from the golden",
            f.salt,
            f.tag
        );
        let defect = rel(&host_fold(&fin, true), &c.want);
        assert!(
            defect > tol,
            "{} {}: mixing the normalised sources differs by only {defect:e}, inside the \
             {tol:e} tolerance — this fixture cannot see the trap it exists to catch",
            f.salt,
            f.tag
        );
    });
}

/// **Widths, token counts and slot layouts the goldens structurally cannot reach.**
///
/// Every capture is at the tiny model's `hidden` of 192, and the kernel launches 256 threads
/// — so in every golden-backed test the strided loop runs at most one iteration and 64
/// threads run none. At the real 7168 each thread runs 28; an accumulation bug that only
/// appears when the loop wraps is invisible to the whole golden-backed suite. Every capture
/// also passes `tokens = 1`, so `blockIdx.x`'s two strides are multiplied by zero
/// throughout. Scored against the same f64 host oracle
/// [`mixing_the_normalised_sources_goes_red`] validates at 192, so this inherits that
/// evidence rather than asserting a fresh claim (`k3:tests/k3_kernels.rs:610`).
#[test]
fn attn_res_at_real_widths_and_multiple_tokens() {
    let tol = attn_res_tol();
    let mut r = Lcg(0xA77E);
    // 192 reproduces the goldens' width so a failure is attributable to the synthetic data
    // rather than the width; 257 wraps the loop exactly once with a one-thread tail; 1000
    // wraps it unevenly; 7168 is the real hidden. 3 tokens because 2 cannot distinguish
    // "stride applied once" from "stride applied per block". `slots` is the arena's
    // per-token capacity: `slots > nsrc` is §3's real `[T][9][hidden]` layout, where the
    // live stack is shorter than the slot count at every layer below 84; `slots == nsrc` is
    // the packed case the goldens happen to have. Both are launched, because the k3 kernel
    // took the packed one as an assumption until 2026-08-12 (`k3:tests/k3_kernels.rs:637`).
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
            sweep_cell(Cell { n, tokens, slots }, nsrc, &mut r, tol);
        }
    }
}

/// One sweep cell's arena geometry — the loop variables of the case table above, named so
/// [`sweep_cell`] can carry one context instead of four bare `usize`.
#[derive(Clone, Copy)]
struct Cell {
    n: usize,
    tokens: usize,
    slots: usize,
}

/// Launch and score ONE (cell, nsrc) case of the width sweep — split from the case table so
/// the arena assembly, the per-token scoring and the two anti-vacuity clauses each read as
/// one chunk instead of one four-deep loop nest.
fn sweep_cell(cell: Cell, nsrc: usize, r: &mut Lcg, tol: f32) {
    let Cell { n, tokens, slots } = cell;
    let cap = if slots == 0 { nsrc } else { slots.max(nsrc) };
    let stride = cap * n;
    // Only the first `nsrc` slots of each token are the live stack. The rest hold a LOUD
    // sentinel rather than zeros — a kernel reading the wrong token's slots picks up 9.0e9
    // and cannot produce a plausible mixture, so the failure is unmistakable rather than
    // merely numerical.
    let mut src: Vec<f32> = vec![9.0e9; tokens * stride];
    for t in 0..tokens {
        for e in 0..nsrc * n {
            src[t * stride + e] = r.f();
        }
    }
    // **Scaled by 1/sqrt(n), and without this the sweep tested a copy.** `attn_res`'s score
    // has no width normaliser, so with a unit-magnitude fold the score spread grows as
    // ~0.577·sqrt(n): 8 at n=192 but 49 at n=7168, where the softmax collapses onto one
    // source (measured in the k3 tree: max probability 1.0000000000 in all five cells at
    // 7168, runner-up at 4e-40 — the mixing loop was bitwise a copy at exactly the width the
    // sweep exists to reach). Scaling the fold keeps the scores O(1) at every width; the
    // `pmax` assert below refuses a draw that degenerates anyway
    // (`k3:tests/k3_kernels.rs:665`).
    let fold: Vec<f32> = (0..n).map(|_| r.f() / (n as f32).sqrt()).collect();
    let eps = 1e-5f32;

    let got = device(&Arena {
        src: &src,
        fold: &fold,
        tokens,
        nsrc,
        n,
        stride,
        eps,
    });

    for t in 0..tokens {
        let stack = &src[t * stride..t * stride + nsrc * n];
        let fin = FoldIn {
            src: stack,
            nsrc,
            n,
            fold: &fold,
            eps,
        };
        // Anti-vacuity: a mixture collapsed onto one source tests argmax and a copy, not
        // mixing. Asserted per token because it is a property of the DRAW and a future seed
        // or width could re-degenerate it silently.
        let (p, _) = host_probs(&fin);
        let pmax = p.iter().copied().fold(0.0f64, f64::max);
        assert!(
            pmax < 0.9,
            "n={n} nsrc={nsrc} slots={cap} token {t}: one-hot (max p = {pmax:.6}), so the \
             mixing loop is a copy and this cell tests nothing it claims to"
        );
        let want = host_fold(&fin, false);
        // This sweep scores two correct implementations against each other, so its realised
        // difference is ~1e-7 and the operator tolerance is three orders of slack that moved
        // when an unrelated floor did — the tripwire inside `score_all` is the bar
        // (`k3:tests/k3_kernels.rs:702`).
        score_all(
            &format!("the attn_res width sweep at n={n} tokens={tokens} nsrc={nsrc} token {t}"),
            Bars {
                tol,
                observed: ATTN_RES_SWEEP_WORST,
            },
            &[("out", &got[t * n..(t + 1) * n], &want)],
        );
    }
    // Anti-vacuity: with `tokens > 1` a kernel that wrote only block 0 would leave the rest
    // of `out` at zero — caught by the per-token comparison above, but only if the tokens
    // differ from each other. They are drawn independently, so assert it rather than trust
    // the draw.
    if tokens > 1 {
        let (a, b) = (&got[..n], &got[n..2 * n]);
        assert!(
            a.iter().zip(b).any(|(x, y)| x != y),
            "n={n} nsrc={nsrc}: two tokens produced identical output — the block stride is \
             not being applied"
        );
    }
}

/// Every launcher guard, by CODE — accepting any error would pass a build where an unrelated
/// dimension check swallowed the case first. The k3 tree exercises these through its census;
/// this port puts them beside the operator they guard, in the house `assert_guards` form.
#[test]
fn the_attn_res_launcher_guards_its_stack_and_stride() {
    let mut r = Lcg(0xA77F);
    let (nsrc, n) = (3usize, 64usize);
    // `sv`/`fv`, not `src`/`fold`: the sweep above builds the same struct from locals of
    // those names, and the two spellings were a reported clone of each other — the rename is
    // the cheapest divergence and also says these are a guard fixture, not a stack.
    let sv: Vec<f32> = (0..nsrc * n).map(|_| r.f()).collect();
    let fv: Vec<f32> = (0..n).map(|_| r.f()).collect();
    let go = |(tokens, nsrc, n, stride)| {
        attn_res_launch(&Arena {
            src: &sv,
            fold: &fv,
            tokens,
            nsrc,
            n,
            stride,
            eps: 1e-5,
        })
    };
    assert_guards([
        (1001, "zero width", go((1, nsrc, 0, nsrc * n))),
        (1001, "zero tokens", go((0, nsrc, n, nsrc * n))),
        // §3's layer loop guards the layer-level fold on a NON-EMPTY stack; an empty one
        // reaching the kernel means that guard went missing, and it must not be a case the
        // kernel quietly defines (the launcher doc's own argument).
        (1003, "empty stack", go((1, 0, n, nsrc * n))),
        // One past `ATTN_RES_MAX_SRC` (16): a 17-deep stack means the caller's block
        // bookkeeping is wrong — §3 implies at most 8 snapshots plus the prefix sum.
        (1003, "seventeen sources", go((1, 17, n, 17 * n))),
        // A stride below `nsrc·n` overlaps consecutive tokens — the `[T][9][hidden]` arena
        // read with the wrong capacity.
        (1005, "overlapping stride", go((1, nsrc, n, nsrc * n - 1))),
    ]);
    // And the accepted case, so the five refusals above are not "rejects everything".
    assert!(
        go((1, nsrc, n, nsrc * n)).is_ok(),
        "the packed single-token case was refused"
    );
}
