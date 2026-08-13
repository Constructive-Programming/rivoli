//! **S3's layer loop scored against a host reference — the gate that sees CONSUMPTION.**
//!
//! Everything else about this loop is gated on what it SELECTS. `tests/glimmer_loop.rs` checks the
//! window and the QK scales the helpers return, `kernel_coverage.rs` checks that the file names
//! each launcher, and the S2 suites score each kernel against the anchor goldens. Between them
//! they miss the entire middle: **which operand reaches which launcher.** Review enumerated seven
//! such mutations (2026-08-13) and the fixture's own geometry supplied three more.
//!
//! None of these changes a shape, so `proj`'s `i_dim` check cannot see them and every launcher
//! guard accepts them. **Each was applied and run** — re-measured 2026-08-13 on the WIDENED
//! fixture (4 query heads over 2 KV, `head_dim` off the width), against clean floors of 7.1e-7
//! (logits), 5.7e-7 (permuted) and 2.7e-7 (branch):
//!
//! | mutation | worst |
//! |---|---|
//! | `launch_logit_softcap` deleted (multiplier goes with it) | **4.1e0** |
//! | `output_multiplier` → 1.0, softcap kept | **4.1e0** |
//! | `wk` / `wv` swapped — same `[kv·hd, hidden]` shape | **2.9e0** |
//! | `wq` / `wg` swapped — same `[hq·hd, hidden]` shape | **2.5e0** |
//! | `launch_rope_split_half` → `launch_rope_interleave` | **1.8e0** |
//! | `qk_scale_factor` DROPPED from Q (product 1.0, not 3.87) | **1.7e0** |
//! | the layer kind from `l % 4 == 3` instead of `layer_types` | **1.0e0** |
//! | the KV broadcast as `head % hkv` instead of `head / (hq/hkv)` | **9.5e-1** |
//! | `if self.rotated[l]` inverted — the NoPE layers rotate | **8.2e-1** |
//! | `attn_scale` from `hidden` instead of `head_dim` | **3.7e-1** |
//! | the softcap's `tanh` alone, `output_multiplier` KEPT | **9.4e-5** |
//! | `eps_post` / `eps_pre` transposed on the branch norms | **1.5e-5** |
//! | the gate from the attend output, not the layer input | RED on `proj`'s shape check |
//!
//! **THREE ROWS ARE NEW, and two were previously UNCONSTRUCTIBLE rather than merely uncaught.**
//! Until the fixture widened on 2026-08-13 it had `head_dim = hidden` and ONE KV head, so sourcing
//! the softmax scale from the wrong dimension was an exact no-op and `head / (hq/hkv)` and
//! `head % hkv` were the same function. No tolerance could have found either. The eps
//! transposition — this tree's one ungated correctness item for four review rounds — now reddens
//! all three tests here.
//!
//! **ONE mutation still does not redden, and it is the one that cannot.** Swapping the two QK
//! scales is an identity in exact arithmetic: both operands are normed before the scale, so the
//! score carries only their product, and RoPE commutes with a scalar. Three limits, all from
//! review — it is not byte-identical (`fl(3.87·q̂)` and `fl(3.87·k̂)` round differently), it holds
//! only while the KV cache is f32, and **it does change `q` and `k` themselves**, which the anchor
//! captures elementwise, so a swapped engine is 3.87x off at S4's tensor-vs-capture scoring.
//! `tests/common/tolerance.rs` recorded the algebra on 2026-08-12; what was new was only that
//! `glimmer-architecture.md` §9 still called the swap fluent-and-wrong.
//!
//! A gate that reports only its reds is a gate whose blind spots are discovered by the next
//! defect. This table lists outcomes, and the two rows that changed from GREEN to a number are why
//! the fixture's geometry is now three inequalities with a comment on each.
//!
//! **Incidental:** `glimmer_reference.rs`'s comparison reads the SOURCE checkpoint's bytes while
//! the engine reads the CONVERTED artifact, so a green run there also says `convert_glimmer`
//! reproduced those tensors faithfully.
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
/// **Set from the weakest RED-PROVED mutation, not from a rounding model**, and re-derived from
/// scratch on 2026-08-13 when the fixture widened to 4 query heads over 2 KV heads with a
/// `head_dim` off the width. Every number below is that fixture's; the previous set was the old
/// geometry's and none of it carried over.
///
/// | | worst |
/// |---|---|
/// | clean | **7.1e-7** |
/// | `eps_post` / `eps_pre` transposed | **1.0e-5** |
/// | the softcap's `tanh` alone, `output_multiplier` kept | **9.4e-5** |
/// | `qk_scale_factor` dropped from Q | **1.7e0** |
///
/// The permuted-`layer_types` test shares this bound: its clean floor is **5.7e-7** and the same
/// transposition moves it to **1.5e-5**, so it catches the eps defect at 26x.
///
/// > **A separate `TOL_KIND = 1e-3` nearly shipped here**, justified by "the permuted test's clean
/// > floor is 1.5e-5, twenty times the other one" — with a whole paragraph explaining why a
/// > full-attention layer at position 0 accumulates more softmax error. **1.5e-5 was that test's
/// > reading under the eps MUTATION**, read off a run where the mutation was still applied and
/// > written down as a clean floor. The explanation was plausible, internally consistent, and
/// > about a number that did not exist. Caught by re-running clean.
///
/// 3e-6 is 4.2x above the worst floor and 3.5x below the weakest defect. **The two sides do NOT differ
/// only in reduction order** — `gqa_attend` runs an online softmax against this file's two-pass,
/// `gemm_bf16` compiles under `-ffp-contract=fast` so its dot is an FMA reduction, and the
/// transcendentals are device against host libm. The bound is empirical for that reason.
///
/// > **The first version was 1e-4, justified as reduction noise, and it PASSED the `tanh` mutation
/// > by 1%** — the argmax-invariant defect `Glimmer::logits` exists to make visible, under the
/// > tolerance of the gate that consumes it. The tree had already measured that class at 4.879e-5
/// > on the anchor.
const TOL: f32 = 3e-6;

/// The same comparison one layer up, on the post-FFN branch — its own constant because its floor
/// is its own, and both MEASURED on the widened fixture (2026-08-13):
///
/// | | worst over four layers |
/// |---|---|
/// | clean | **2.7e-7** |
/// | `eps_post` / `eps_pre` transposed | **3.1e-6** |
///
/// 8e-7 is 3.0x above the floor and 3.9x below the signal. **Widening the fixture is what made
/// that comfortable**: at 2 query heads over 1 KV head the same separation was 4.8x end to end and
/// the constant had 1.6x of room, which is the kind of margin a re-drawn fixture breaks.
const TOL_BRANCH: f32 = 8e-7;

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
    /// Each layer's post-FFN branch at the position [`Ref::step`] last ran — the tensor the engine
    /// leaves in its own branch buffer, and the one the two epsilons are visible on.
    br: Vec<Vec<f32>>,
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
            br: vec![Vec::new(); c.n_layers],
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
            self.br[l] = br.clone();
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

/// **The eps assignment, gated at last — every layer's branch, engine against oracle.**
///
/// `Glimmer::pre_norm` reads `eps_pre` (1e-5) and `Glimmer::branch_add` reads `eps_post` (1e-8),
/// assigned by POSITION, three orders apart. Transposing them was the tree's one ungated
/// correctness item for four review rounds, and the reason took two failed attempts to find:
///
/// * **Not the logits.** The branch enters a residual stream that dominates it, so the
///   transposition is ~5e-6 by the head — under this file's own 2e-5 and under anything the
///   reference gate can defend. Both measured.
/// * **Not the reference's captures either**, which the open-items register predicted would close
///   it. On the branch the transposition is 1.6e-3 – 1.3e-2 while the bf16 weight floor between
///   rivoli and an f32 reference is 4.7e-3 – 3.0e-2 — the signal is 0.2x to 0.6x the noise at
///   every layer of both salts. `tests/glimmer_reference.rs` records that measurement.
///
/// **What works is a comparison with no weight term at all**: both sides here read the same bf16
/// artifact, so the floor is reduction noise.
///
/// > **AND THE ANSWER TURNED OUT TO BE ALREADY IN THE TREE.** Re-measuring 2026-08-13 with the
/// > transposition applied: `the_loop_matches_a_host_reference_at_every_position` reddens at
/// > **3.673e-5** against its 2e-5 tolerance. That test could not see it a day earlier — at
/// > `TOL = 1e-4` and one compared position — and the fix that closed it was the SOFTCAP
/// > tolerance work, which tightened the bound to 2e-5 and made the comparison per-position. The
/// > open-items register carried "nothing in the tree reddens" from a measurement taken before
/// > that change and never re-run. **A stale measurement carried forward as a fact**, which is
/// > this session's recurring defect and worth the sentence.
/// >
/// > This test still earns its place: it LOCALISES the defect to a layer, and it is a second and
/// > independent catch. Both margins are thin and both are measured — 1.8x over tolerance for the
/// > logits, 1.6x for this — because the fixture's branch is at `mean(x²)` ~ O(1) where the two
/// > epsilons are nearly the same number.
///
/// Truncating the config to `l + 1` layers is what selects a layer — the engine keeps one branch
/// buffer and every layer overwrites it. Layers `0..l` compute identically either way.
#[test]
fn every_layers_branch_matches_the_oracle_and_that_is_where_the_eps_lives() {
    let (root, cfg, src) = setup("glimmer-eps");
    let gt_full = &cfg.text;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let mut cells = 0;
    let mut worst = (0.0f32, 0usize);
    for l in 0..gt_full.n_layers {
        let mut gt = gt_full.clone();
        gt.n_layers = l + 1;
        gt.layer_types.truncate(l + 1);
        gt.layer_rope_theta.truncate(l + 1);
        let got = common::decode_one(&root.join("out"), &gt, &prompt)
            .branch()
            .unwrap();
        // The oracle runs the FULL model; layer `l`'s branch is the same either way, and running it
        // once for every truncation would be the same arithmetic done `n_layers` times.
        let mut r = Ref::new(&src, gt_full);
        for (pos, &t) in prompt.iter().enumerate() {
            r.step(t, pos);
        }
        assert!(
            r.br[l].iter().any(|v| *v != 0.0),
            "L{l}: the oracle's branch is all zero, so the score below is against nothing"
        );
        let d = worst_rel(&got, &r.br[l]);
        assert!(
            d < TOL_BRANCH,
            // The numbers are THIS test's, measured on this fixture: clean 2.7e-7, transposed
            // 3.1e-6. It read "~4e-3, three orders above this file's floor" until 2026-08-14 —
            // that figure belongs to the reference comparison, which carries a bf16 weight term
            // this one does not. Whoever triaged a red run at 9e-7 would have read it and ruled
            // the transposition OUT, three orders too small, when 9e-7 is squarely in range.
            "L{l}'s post-FFN branch disagrees with the oracle by {d:.3e}. This tensor carries the \
             eps assignment: on this fixture a 1e-5/1e-8 transposition lands at ~3.1e-6 against a \
             clean ~2.7e-7 — a factor of ~11, and invisible everywhere downstream"
        );
        if d > worst.0 {
            worst = (d, l);
        }
        cells += 1;
    }
    println!(
        "  {cells} layer branches vs the oracle, worst {:.3e} at L{}",
        worst.0, worst.1
    );
    assert_eq!(cells, gt_full.n_layers, "every layer must be scored");
}
