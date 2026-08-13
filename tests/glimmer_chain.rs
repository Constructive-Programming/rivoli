//! **S3's layer loop scored against a host reference — the gate that sees CONSUMPTION.**
//!
//! Everything else about this loop is gated on what it SELECTS. `tests/glimmer_loop.rs` checks the
//! window and the QK scales the helpers return, `kernel_coverage.rs` checks that the file names
//! each launcher, and the S2 suites score each kernel against the anchor goldens. Between them
//! they miss the entire middle: **which operand reaches which launcher.** Review enumerated seven
//! mutations that pass all of it and produce fluent wrong text (2026-08-13) —
//!
//! None of the first six changes a shape, so `proj`'s `i_dim` check cannot see them and every
//! launcher guard accepts them; the seventh is the exception and is marked as such. **Each was
//! applied to `glimmer_gpu.rs` and run** — measured 2026-08-13 in [`worst_rel`]'s metric, against a
//! green run whose worst position scores **3.9e-6**:
//!
//! | mutation | worst | first diverges at |
//! |---|---|---|
//! | `launch_logit_softcap` deleted (multiplier goes with it) | **4.1e0** | position 4 |
//! | `output_multiplier` → 1.0, softcap kept | **4.1e0** | position 4 |
//! | `launch_rope_split_half` → `launch_rope_interleave` | **1.8e0** | position 1 |
//! | `wq` / `wg` swapped — same `[hq·hd, hidden]` shape | **1.3e0** | position 1 |
//! | `wk` / `wv` swapped — same `[kv·hd, hidden]` shape | **9.0e-1** | position 0 |
//! | `if self.rotated[l]` inverted — the 13 NoPE layers rotate | **3.2e-1** | position 1 |
//! | the layer kind from `l % 4 == 3` instead of `layer_types` | **2.4e-1** | permuted config only |
//! | `qk_scale_factor` DROPPED from Q (product 1.0, not 3.87) | **1.1e-1** | position 1 |
//! | the softcap's `tanh` alone, `output_multiplier` KEPT | **9.9e-5** | position 8 |
//! | the gate from the attend output, not the layer input | **RED on `proj`'s shape check** | — |
//!
//! The rotation ones first diverge at position **1** and not 0, which is the sweep working: at
//! `pos = 0` the angle is 0 and every rope convention is the identity.
//!
//! **Two rows are also caught by something cheaper, and saying so is the point of a census.** The
//! interleaved-rope substitution removes `glimmer_gpu.rs`'s only mention of `launch_rope_split_half`,
//! so `kernel_coverage.rs`'s OWNERS census reddens on it with no GPU at all; and the gate-from-the-
//! attend-output reddens on `proj`'s own `i_dim` check, which fires on any decode of the fixture and
//! therefore already in `glimmer_loop.rs`. This gate's marginal value is the other eight rows.
//!
//! **The `tanh` row is why [`TOL`] is what it is.** At the 1e-4 this file shipped with, that
//! mutation PASSED one of the two tests by 1% — the argmax-invariant defect `Glimmer::logits`
//! exists to catch, sitting under the tolerance of the gate that consumes it. Found by review, and
//! the tree had already measured the same class at 4.879e-5 on the anchor.
//!
//! **The `l % 4 == 3` row was re-measured with the ALLOCATION mutated in lockstep**, because review
//! argued the first proof reddened through a device write past a cache sized from the true kinds
//! rather than through the masking difference. Same magnitude, 2.353e-1, so it did not: the
//! consistent defect — a port that computes the period everywhere — is what that number measures.
//!
//! **And three mutations do NOT redden.**
//!
//! * **Swapping the two QK scales does not change the LOGITS.** Both operands are normed before
//!   the scale, so the score carries only their product, and RoPE commutes with a scalar. It is an
//!   identity in exact arithmetic and measured inside reduction noise here — but the claim needs
//!   three limits, all added by review 2026-08-13 after the first version asserted it flatly:
//!   (a) it is **not byte-identical**, since `fl(3.87·q̂)` and `fl(3.87·k̂)` are different roundings,
//!   and 2.3e-6 was the size of that perturbation rather than evidence of an identity;
//!   (b) it holds while the KV cache is f32 and the norm precedes the scale, both true today and
//!   the first of them an explicit S5 lever; and (c) **it changes `q` and `k` themselves**, which
//!   the anchor captures elementwise, so a swapped engine is 3.87x off at S4's tensor-vs-capture
//!   scoring. The decode output cannot see it; the tree can.
//!   **This was already known here.** `tests/common/tolerance.rs` recorded the algebra on
//!   2026-08-12 — "`(s*q)·k` and `q·(s*k)` are the same product … invisible to this kernel by
//!   ALGEBRA, not by insufficient resolution" — while excluding `qk_scale_on_k` from `attend`'s
//!   defect set. What was new on 2026-08-13 is only that `glimmer-architecture.md` §9 still called
//!   the swap fluent-and-wrong.
//! * **`attn_scale` from `hidden` instead of `head_dim` is invisible TO THIS FIXTURE**, which sets
//!   `head_dim = hidden = 8`. The shipped model has 128 against 6656. The fixture preserves
//!   `head_dim != hidden / n_heads` (trap 15) and not `head_dim != hidden`; closing this needs the
//!   fixture's head_dim to stop tracking its width, which touches every shape it writes.
//! * **Swapping `eps_post` for `eps_pre` at `pre_norm` / `branch_add` is below [`TOL`] here, and
//!   NOTHING ELSE IN THE TREE GATES IT EITHER — it is OPEN.** This file said it was covered by
//!   "`glimmer_head.rs`'s eps census"; that file has no eps census, the 41.8-56.6x figures are
//!   `tests/glimmer_norm.rs`'s, and what they establish is that the OPERATOR distinguishes the two
//!   epsilons on activations at `mean(x²)~1e-3` — not which eps the loop hands it. `glimmer_norm.rs`
//!   never imports `glimmer_gpu`. Swap the two call sites and every test in the tree stays green.
//!   (Corrected by review 2026-08-13; the wrong citation had reached three places including the
//!   INDEX verdict, which `CLAUDE.md` tells readers to trust instead of the doc.)
//!
//! **A fourth blind spot, same round:** trap 10, the KV head broadcast, is unconstructible here
//! because the fixture has **one** KV head — `head / (hq/hkv)` and `head % hkv` are both 0 for every
//! head, as is any other mapping. The oracle carries that expression and a comment naming the trap,
//! which reads as coverage and is not. It is gated at the kernel level by `glimmer_attend.rs`;
//! raising the fixture to 4 query heads over 2 KV heads would close it here too.
//!
//! A gate that reports only its reds is a gate whose blind spots are discovered by the next defect.
//!
//! **Incidental, and worth knowing before it surprises someone:** the oracle reads the SOURCE
//! checkpoint's bytes while the engine reads the CONVERTED artifact, so a green run also says
//! `convert_glimmer` reproduced these tensors faithfully. A future converter change — §6's declined
//! `q_proj`/`k_proj` row permutation, say — would redden this file for a reason its own docs would
//! not otherwise explain.
//!
//! # Why a host reference and not the goldens
//!
//! The vendored anchor goldens hold 1,099 captured intermediates and **no parameters**, so the
//! reference model cannot be run on weights this engine also has. The toy checkpoint can: it is
//! 4 layers at hidden 8, one period of `[s,s,s,full]`, `head_dim` 8 against 2 query heads (so
//! `head_dim != hidden / n_heads` holds, trap 15), and it converts and pins exactly like the real
//! one.
//!
//! **The oracle below is transcribed from `docs/reference/glimmer-architecture.md` §3, §4 and §5
//! and from the kernel formulas, NOT from `glimmer_gpu.rs`.** That is the whole of its value: an
//! oracle derived from the engine agrees with the engine's wiring by construction. Where it and
//! the engine share an idea they share it through the reference — the eps assignment by position,
//! `q = qk_norm(q) * 3.87` with `k = qk_norm(k)`, the gate reading the layer input, the post-norms
//! on the branch, `kvh = head / (hq / hkv)`, the split-half pairing `(x[j], x[j+seg/2])`.
//!
//! **Three readings are NOT independent, and two of them are additionally sub-tolerance** (review,
//! 2026-08-13). The rope-on/off predicate is character-for-character the same on both sides
//! (`layer_rope_theta[l] != 0.0`), from §2's "read as a boolean"; the final norm's eps is a guess
//! from §5's SILENCE — it names the plain form and no epsilon, both sides chose `rms_norm_eps`, and
//! the alternative is sub-`TOL` anyway; and `weightless` folds the scale the KERNEL's way
//! (`scale * rms_inv`) rather than the reference's `(x·rs)·s`, a deviation `kernels/linalg.hip`
//! prices at ~6e-8 and names in place. A green run carries no information about the second and
//! third at all.
//!
//! **What it does NOT establish** is that either side matches Muse Glimmer. The goldens are what
//! say the kernels are right; this says the loop feeds them correctly. Both are needed and neither
//! substitutes.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

mod common;
use common::{
    FixtureTensor, GLIMMER_FIXTURE_DIM as DIM, TempRoot, glimmer_convert_fixture, rms_inv,
    weightless, window_lo, worst_rel,
};
use rivoli::artifact::model as gm;
use rivoli::artifact::model::LayerKind;
use rivoli::glimmer_gpu::Glimmer;
use std::collections::HashMap;

/// Relative tolerance on a logit, in [`worst_rel`]'s metric.
///
/// **Set from the weakest RED-PROVED mutation, not from a rounding model** — and the first version
/// was set the other way and was caught by it. It read 1e-4, justified as "the two sides differ
/// only in reduction ORDER". Both halves were wrong (review, 2026-08-13):
///
/// * The two sides do not differ only in reduction order. `gqa_attend` runs an ONLINE softmax
///   (running max, one `rescale` multiply per KV row) against this file's two-pass; `gemm_bf16`
///   compiles under clang's HIP default `-ffp-contract=fast`, so its dot is an FMA reduction with
///   the product never rounded, against `.sum()` here; and the transcendentals are device `expf` /
///   `tanhf` / `pow` against host `f32::exp` / `f32::tanh` / `f64::powf`. `weightless` also folds
///   the scale the way the KERNEL does (`scale * rms_inv`) rather than the way the reference does
///   (`(x·rs)·s`), a deviation `kernels/linalg.hip` prices at ~6e-8 and names in place.
/// * More seriously, **1e-4 was above a real defect.** Removing the softcap's `tanh` while keeping
///   `output_multiplier` — the argmax-invariant failure `Glimmer::logits` was added to make
///   visible — scores **9.9e-5** in one of the two tests here and 1.2e-4 in the other. At 1e-4 the
///   first test PASSED it, by 1%. The tree had already measured the same defect class at 4.879e-5
///   on the anchor and recorded why (a tiny model's untrained logits sit in `tanh`'s linear
///   region), so the margin was knowable before it was lucky.
///
/// 2e-5 sits ~5x above the worst green position (3.9e-6) and ~5x below that weakest defect. The
/// green figure is what to watch: if it approaches this, the fixture's widths grew and the
/// tolerance needs re-deriving from a fresh red proof, not raising.
const TOL: f32 = 2e-5;

/// bf16 bytes → f32, the widening both `bf16f` in the kernels and the converter's norms perform.
fn wide(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| rivoli::math::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// `y = W · x`, `W` being `[o, i]` row-major — the shape every Glimmer projection ships in.
fn matvec(w: &[f32], x: &[f32], o: usize, i: usize) -> Vec<f32> {
    assert_eq!(w.len(), o * i, "weight is not [{o}, {i}]");
    assert_eq!(x.len(), i);
    (0..o)
        .map(|r| (0..i).map(|c| w[r * i + c] * x[c]).sum())
        .collect()
}

/// `MuseGlimmerTextCenteredRMSNorm`: `_norm(x) * (1 + w)`. The four per-layer norms, and nothing
/// else. Its weight is initialised to ZEROS, which is why substituting the plain form here
/// multiplies the residual stream by ~0 and is the LOUD direction of that mistake.
fn centered(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    // `zip` TRUNCATES, so a norm weight of the wrong length would yield a short vector and surface
    // later at a `matvec` length assert naming the wrong operation. The oracle is the side that
    // must fail loudly (review, 2026-08-13).
    assert_eq!(
        x.len(),
        w.len(),
        "centered norm: weight is not the activation's width"
    );
    let inv = rms_inv(x, eps);
    x.iter()
        .zip(w)
        .map(|(v, wi)| v * inv * (1.0 + wi))
        .collect()
}

/// `MuseGlimmerRMSNorm`: `_norm(x) * w`. The final `model.norm` only, in this chain.
fn plain(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        w.len(),
        "plain norm: weight is not the activation's width"
    );
    let inv = rms_inv(x, eps);
    x.iter().zip(w).map(|(v, wi)| v * inv * wi).collect()
}

/// Split-half RoPE in place over `rows` segments of `d`: the pair is `(x[j], x[j + d/2])`,
/// rotating by `pos · theta^(-2j/d)`.
///
/// **The pairing is the entire difference from the interleaved form**, which pairs `(2j, 2j+1)`
/// at the same frequencies. Both are in the tree, both take identical arguments, and applying one
/// where the other belongs is trap 9 — fluent, no error.
fn rope_split_half(x: &mut [f32], rows: usize, d: usize, pos: usize, theta: f64) {
    let half = d / 2;
    for r in 0..rows {
        let seg = &mut x[r * d..(r + 1) * d];
        let before = seg.to_vec();
        for j in 0..half {
            let ang = pos as f64 * theta.powf(-2.0 * j as f64 / d as f64);
            let (cs, sn) = (ang.cos() as f32, ang.sin() as f32);
            let (a, b) = (before[j], before[half + j]);
            seg[j] = a * cs - b * sn;
            seg[half + j] = b * cs + a * sn;
        }
    }
}

/// The reference chain, holding the fixture's weights and its own linear KV cache.
///
/// The cache is linear at every layer and the window is applied in the SOFTMAX BOUND rather than
/// by a ring, which is deliberate: the engine's ring is an optimisation of exactly this, so an
/// oracle that also kept a ring would agree with a wrong slot map.
struct Ref<'a> {
    w: HashMap<String, Vec<f32>>,
    c: &'a gm::GlimmerTextConfig,
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
}

impl<'a> Ref<'a> {
    fn new(src: &[FixtureTensor], c: &'a gm::GlimmerTextConfig) -> Self {
        let w = src
            .iter()
            .map(|(n, _, b)| (n.clone(), wide(b)))
            .collect::<HashMap<_, _>>();
        Ref {
            w,
            c,
            k: vec![Vec::new(); c.n_layers],
            v: vec![Vec::new(); c.n_layers],
        }
    }

    fn t(&self, name: &str) -> &[f32] {
        self.w
            .get(name)
            .unwrap_or_else(|| panic!("{name} is not in the fixture"))
    }

    fn layer_t(&self, l: usize, name: &str) -> &[f32] {
        self.t(&format!("{}.{l}.{name}.weight", gm::GLIMMER_LAYER_PREFIX))
    }

    /// One position: §3's twelve lines with §4 inlined. Returns the post-softcap logits.
    fn step(&mut self, token: u32, pos: usize) -> Vec<f32> {
        // The cache is a `push`-ordered Vec, so `self.k[l][j]` is position `j` only if this is
        // called with 0, 1, 2, ... exactly once each. Asserted rather than assumed: violated, the
        // failure is an index panic in the softmax naming an operation that is not the defect —
        // and this oracle is the template S4's real-weights comparison will start from, where the
        // positions will not be a bare `enumerate` (review, 2026-08-13).
        assert_eq!(
            pos,
            self.k[0].len(),
            "the reference replays positions in order, once each"
        );
        let c = self.c;
        let (hid, hq, hkv, hd) = (c.hidden, c.n_heads, c.num_key_value_heads, c.head_dim);
        let (qd, kvd) = (hq * hd, hkv * hd);
        let (ep, eq) = (c.rms_norm_eps as f32, c.post_norm_eps as f32);

        // §5: the embedding is NORMED, by the weightless form, and cannot be folded into the
        // matrix — the DFlash drafter shares it unnormed.
        let emb = self.t("model.language_model.embed_tokens.weight");
        let mut h = emb[token as usize * hid..(token as usize + 1) * hid].to_vec();
        weightless(&mut h, 1, hid, ep, 1.0);

        for l in 0..c.n_layers {
            // ---- attention block, §4 -------------------------------------------------------
            let res = h.clone();
            let xn = centered(&h, self.layer_t(l, "input_layernorm"), ep);

            let mut q = matvec(self.layer_t(l, "self_attn.q_proj"), &xn, qd, hid);
            let mut k = matvec(self.layer_t(l, "self_attn.k_proj"), &xn, kvd, hid);
            let v = matvec(self.layer_t(l, "self_attn.v_proj"), &xn, kvd, hid);

            // qk_norm is weightless, per head, over head_dim — and the 3.87 is Q's ALONE.
            weightless(&mut q, hq, hd, ep, c.qk_scale_factor as f32);
            weightless(&mut k, hkv, hd, ep, 1.0);

            // `layer_rope_theta` is 0 on full-attention layers: they are NoPE and skip entirely.
            if c.layer_rope_theta[l] != 0.0 {
                let theta = c.rope_parameters.rope_theta;
                rope_split_half(&mut q, hq, hd, pos, theta);
                rope_split_half(&mut k, hkv, hd, pos, theta);
            }

            // The cache holds POST-norm, POST-rope keys. Order is norm → scale → rope → cache.
            self.k[l].push(k);
            self.v[l].push(v);

            // The window is the SOFTMAX BOUND here, inclusive of `pos`. A full layer has none.
            let win = match c.layer_types[l] {
                LayerKind::SlidingAttention => c.sliding_window,
                LayerKind::FullAttention => 0,
            };
            let lo = window_lo(pos, win);
            let scale = (hd as f64).powf(-0.5) as f32;
            let mut attn = vec![0.0f32; qd];
            for head in 0..hq {
                let kvh = head / (hq / hkv); // a block repeat, NOT `head % hkv`
                let qh = &q[head * hd..(head + 1) * hd];
                let s: Vec<f32> = (lo..=pos)
                    .map(|j| {
                        let kj = &self.k[l][j][kvh * hd..(kvh + 1) * hd];
                        qh.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
                let z: f32 = e.iter().sum();
                for (n, j) in (lo..=pos).enumerate() {
                    let vj = &self.v[l][j][kvh * hd..(kvh + 1) * hd];
                    let p = e[n] / z;
                    for i in 0..hd {
                        attn[head * hd + i] += p * vj[i];
                    }
                }
            }

            // The gate is computed from the LAYER INPUT, not from the attention output.
            let g = matvec(self.layer_t(l, "self_attn.gate_proj"), &xn, qd, hid);
            for i in 0..qd {
                attn[i] *= 1.0 / (1.0 + (-g[i]).exp());
            }
            let br = matvec(self.layer_t(l, "self_attn.o_proj"), &attn, hid, qd);

            // The post-norm is on the BRANCH, before the residual add — and takes the OTHER eps.
            let br = centered(&br, self.layer_t(l, "post_attention_layernorm"), eq);
            h = res.iter().zip(&br).map(|(a, b)| a + b).collect();

            // ---- MLP, §3 -------------------------------------------------------------------
            let res = h.clone();
            let xn = centered(&h, self.layer_t(l, "pre_feedforward_layernorm"), ep);
            let g = matvec(self.layer_t(l, "mlp.gate_proj"), &xn, c.inter, hid);
            let u = matvec(self.layer_t(l, "mlp.up_proj"), &xn, c.inter, hid);
            let m: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
                .collect();
            let br = matvec(self.layer_t(l, "mlp.down_proj"), &m, hid, c.inter);
            let br = centered(&br, self.layer_t(l, "post_feedforward_layernorm"), eq);
            h = res.iter().zip(&br).map(|(a, b)| a + b).collect();
        }

        // §5: the final norm is the PLAIN form, then the head, then the softcap.
        let fin = plain(&h, self.t("model.language_model.norm.weight"), ep);
        let logits = matvec(self.t("lm_head.weight"), &fin, c.vocab, hid);
        let (mult, cap) = (c.output_multiplier as f32, c.final_logit_softcapping as f32);
        logits
            .iter()
            .map(|x| cap * (x * mult / cap).tanh())
            .collect()
    }
}

/// Convert the fixture and return everything both sides need.
fn setup(tag: &str) -> (TempRoot, gm::GlimmerConfig, Vec<FixtureTensor>) {
    let root = TempRoot::new(tag);
    let (src, _) = glimmer_convert_fixture(root.path(), DIM);
    let cfg = gm::load_config(root.join("out").to_str().unwrap()).unwrap();
    (root, cfg, src)
}

/// **The engine's logits match a reference transcribed from §3-§5, at every position of a run
/// that crosses the sliding ring.**
///
/// The prompt and generation together span 9 positions against a `sliding_window` of 2, so every
/// sliding layer's ring wraps four times while the full-attention layer's linear cache grows — and
/// the two cache disciplines are scored against ONE oracle that keeps neither.
///
/// Both are checked at every step rather than only at the end. A single final comparison would
/// still catch all seven mutations, but it would report every one of them at the last position and
/// say nothing about where the divergence started, which is the first question.
#[test]
fn the_loop_matches_a_host_reference_at_every_position() {
    let (root, cfg, src) = setup("glimmer-chain");
    let gt = &cfg.text;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let max_new = 5;

    // One continuous run first — it is what exercises the ring across a wrap, and it decides which
    // tokens the sweep below replays.
    let mut e = Glimmer::new(
        root.join("out").to_str().unwrap(),
        gt,
        None,
        prompt.len() + max_new,
    )
    .unwrap();
    let out = e.decode(&prompt, max_new, &[]).unwrap();
    assert_eq!(out.len(), max_new);
    let mut fed: Vec<u32> = prompt.clone();
    fed.extend_from_slice(&out[..out.len() - 1]);
    assert!(
        fed.len() > 2 * gt.sliding_window,
        "the run must wrap the ring: {} positions against a window of {}",
        fed.len(),
        gt.sliding_window
    );

    // **Per POSITION, and it costs one engine each.** The engine samples only at the last position
    // it is given, so the only way to see position `i`'s logits is to hand it the prefix ending
    // there — and a fresh engine, because the KV cache of a longer run is not the state a prefix
    // would have produced. On a 4-layer fixture that is ten pins of a few kilobytes.
    //
    // > **This test claimed a per-position comparison for one commit and performed ONE, at the
    // > last position** — the replay loop overwrote its intermediate logits and the `worst_seen`
    // > tuple beside it was vestigial. Review, 2026-08-13. Which position first diverges is the
    // > first question a red run raises, and a gate that answers it only for the last is a gate
    // > whose message is wrong about its own evidence.
    let mut worst_seen = (0.0f32, 0usize);
    for i in 0..fed.len() {
        let mut ei = Glimmer::new(root.join("out").to_str().unwrap(), gt, None, i + 2).unwrap();
        let picked = ei.decode(&fed[..=i], 1, &[]).unwrap();
        let mut r = Ref::new(&src, gt);
        let mut lg = Vec::new();
        for (pos, &t) in fed[..=i].iter().enumerate() {
            lg = r.step(t, pos);
        }
        let got = ei.logits().unwrap();
        // Non-degeneracy, on the REFERENCE side, before any comparison: `worst_rel` returns
        // INFINITY for a non-finite `got` and asserts a finite `want`, but an all-zero pair still
        // scores 0.0 and would pass every assertion below.
        assert!(
            lg.iter().any(|v| *v != 0.0),
            "position {i}: the reference produced an all-zero logit vector, so the score below is \
             a comparison against nothing"
        );
        let d = worst_rel(&got, &lg);
        assert!(
            d < TOL,
            "position {i} of {}: the engine and the reference disagree by {d:.3e} — the two run \
             the same arithmetic on the same weights and differ only in reduction order, so \
             anything above {TOL:e} is a wiring defect and not rounding. This is the FIRST \
             position that diverges; earlier ones agreed.",
            fed.len()
        );
        if d > worst_seen.0 {
            worst_seen = (d, i);
        }
        // The continuous run and the prefix run must also agree on what they emit at this
        // position, which is what says the sweep is measuring the same execution.
        if i + 1 == prompt.len() {
            assert_eq!(
                picked,
                vec![out[0]],
                "prefix and continuous runs disagree at the prompt"
            );
        }
    }
    println!(
        "  {} positions scored, worst {:.3e} at position {}",
        fed.len(),
        worst_seen.0,
        worst_seen.1
    );
}

/// **The loop reads `layer_types`, on a config whose pattern is NOT the shipped period.**
///
/// This model's `layer_types` is `[s,s,s,full]` repeated, so on the shipped checkpoint — and on
/// any prefix of it, which is what the fixture is — a loop keyed on `l % 4 == 3` agrees with the
/// array at every layer. **No test using this checkpoint's pattern can tell the two apart**, which
/// is precisely `layer_types`' own doc: a port that computes the period "produces a model that is
/// right until the first checkpoint whose pattern differs". Verified, not assumed: with
/// `attn_window` rewritten to `l % 4 == 3`, the test above stays green and this one goes red.
///
/// So the pattern is ROTATED — full attention moves to layer 0 — with `layer_rope_theta` moved in
/// lockstep, because `sliding IFF rotated` is gated in both directions and a config that broke it
/// would be refused for the wrong reason. Both sides then read the same permuted config, and only
/// a loop that consults it agrees.
#[test]
fn the_loop_follows_layer_types_and_not_the_shipped_period() {
    let (root, cfg, src) = setup("glimmer-chain-perm");
    let mut gt = cfg.text.clone();
    // `[s,s,s,full]` -> `[full,s,s,s]`, and the thetas with them. A rotation rather than an
    // arbitrary shuffle keeps the counts identical, so nothing downstream can pass or fail on
    // "how many layers slide" instead of on which ones do.
    gt.layer_types.rotate_right(1);
    gt.layer_rope_theta.rotate_right(1);
    assert_eq!(
        gt.layer_types[0],
        LayerKind::FullAttention,
        "the permutation must move the full-attention layer off the period's last slot"
    );
    assert_eq!(gt.layer_rope_theta[0], 0.0, "and its theta with it");
    // **The permutation must DISAGREE with the modulo somewhere, or this test is vacuous** — it
    // would then be a second copy of the test above wearing a different name. Asserted rather than
    // reasoned from the rotation, because the fixture's layer count is shared and can change
    // (review, 2026-08-13).
    assert!(
        (0..gt.n_layers).any(|l| (gt.layer_types[l] == LayerKind::FullAttention) != (l % 4 == 3)),
        "the rotated pattern agrees with `l % 4 == 3` at every layer, so nothing here can \
         distinguish them: {:?}",
        gt.layer_types
    );

    let prompt: Vec<u32> = vec![2, 3, 4, 5, 6];
    let mut e = Glimmer::new(
        root.join("out").to_str().unwrap(),
        &gt,
        None,
        prompt.len() + 1,
    )
    .unwrap();
    let mut r = Ref::new(&src, &gt);
    let out = e.decode(&prompt, 1, &[]).unwrap();
    assert_eq!(out.len(), 1);
    let mut lg = Vec::new();
    for (pos, &t) in prompt.iter().enumerate() {
        lg = r.step(t, pos);
    }
    assert!(
        lg.iter().any(|v| *v != 0.0),
        "the reference produced nothing to compare"
    );
    let d = worst_rel(&e.logits().unwrap(), &lg);
    println!("  permuted layer_types: worst relative disagreement {d:.3e}");
    assert!(
        d < TOL,
        "the engine and the reference disagree by {d:.3e} under a rotated `layer_types` — a loop \
         that derives the layer kind from the period rather than from the array agrees with the \
         shipped pattern and diverges here"
    );
}
