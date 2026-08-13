//! **S3's layer loop scored against a host reference — the gate that sees CONSUMPTION.**
//!
//! Everything else about this loop is gated on what it SELECTS. `tests/glimmer_loop.rs` checks the
//! window and the QK scales the helpers return, `kernel_coverage.rs` checks that the file names
//! each launcher, and the S2 suites score each kernel against the anchor goldens. Between them
//! they miss the entire middle: **which operand reaches which launcher.** Review enumerated seven
//! mutations that pass all of it and produce fluent wrong text (2026-08-13) —
//!
//! Not one changes a shape, so `proj`'s `i_dim` check cannot see them and every launcher
//! guard accepts them. **Each was applied to `glimmer_gpu.rs` and run** — measured 2026-08-13,
//! worst relative logit disagreement, against a green run of **2.3e-6**:
//!
//! | mutation | result |
//! |---|---|
//! | `wk` / `wv` swapped — same `[kv·hd, hidden]` shape | **RED, 1.5e0** |
//! | `qk_scale_factor` DROPPED from Q (product 1.0, not 3.87) | **RED, 1.7e0** |
//! | `wq` / `wg` swapped — same `[hq·hd, hidden]` shape | **RED, 9.6e-1** |
//! | `if self.rotated[l]` inverted — the 13 NoPE layers rotate | **RED, 7.9e-1** |
//! | `launch_rope_split_half` → `launch_rope_interleave`, same arity | **RED, 8.5e-1** |
//! | the layer kind from `l % 4 == 3` instead of `layer_types` | **RED, 9.9e-1** — but only on the
//!   permuted config; see [`the_loop_follows_layer_types_and_not_the_shipped_period`] |
//! | the gate built from the attend output instead of the layer input | **RED**, by `proj`'s shape
//!   check rather than numerically — 16 against a `hidden` of 8 |
//!
//! **And three do NOT redden. Each is worth more than the six that do.**
//!
//! * **Swapping the two QK scales is an IDENTITY.** Both operands are normed before the scale, so
//!   the score carries only their product, and RoPE commutes with a scalar. This tree called that
//!   swap "fluent and wrong" in four places; it cannot change any output. Dropping the factor is
//!   the live mistake, and it is the row above. `glimmer-architecture.md` §9 trap 3 is corrected.
//! * **`attn_scale` from `hidden` instead of `head_dim` is invisible TO THIS FIXTURE**, which sets
//!   `head_dim = hidden = 8`. The shipped model has 128 against 6656. The fixture preserves
//!   `head_dim != hidden / n_heads` (trap 15) and not `head_dim != hidden`; closing this needs the
//!   fixture's head_dim to stop tracking its width, which touches every shape it writes.
//! * **Swapping `eps_post` for `eps_pre` on a branch norm is below the tolerance here.** 1e-5
//!   against 1e-8 moves `1/sqrt(mean(x²)+eps)` by ~5e-6 relative when `mean(x²)` is O(1), which is
//!   what the residual stream carries. `glimmer_head.rs`'s eps census separates them 41.8-56.6x on
//!   the norm chains, where the statistic is small — so the eps assignment IS gated, just not by
//!   this file, and a reader must not take a green run here as covering it.
//!
//! A gate that reports only its reds is a gate whose blind spots are discovered by the next
//! defect. These three are the reason this table lists outcomes and not intentions.
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
//! **What it does NOT establish** is that either side matches Muse Glimmer. The goldens are what
//! say the kernels are right; this says the loop feeds them correctly. Both are needed and neither
//! substitutes.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

mod common;
use common::{
    FixtureTensor, GLIMMER_FIXTURE_DIM as DIM, TempRoot, glimmer_convert_fixture, window_lo,
};
use rivoli::artifact::model as gm;
use rivoli::artifact::model::LayerKind;
use rivoli::glimmer_gpu::Glimmer;
use std::collections::HashMap;

/// Relative tolerance on a logit.
///
/// The two sides do the same arithmetic in f32 over the same widened bf16 weights and differ only
/// in reduction ORDER — the kernels reduce in LDS trees and across a strided grid, this reduces
/// left to right. At the fixture's widths (dots of 8, 16 and 12 terms) that is a handful of ulps,
/// and the measured worst case over every position of a 9-step run is far below this. It is set
/// where a real wiring defect cannot hide: the smallest of the seven mutations below moves a logit
/// by more than 1%.
const TOL: f32 = 1e-4;

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

/// `1/sqrt(mean(x²) + eps)` — the factor all three norm forms share.
fn rms_inv(x: &[f32], eps: f32) -> f32 {
    1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt()
}

/// `MuseGlimmerTextCenteredRMSNorm`: `_norm(x) * (1 + w)`. The four per-layer norms, and nothing
/// else. Its weight is initialised to ZEROS, which is why substituting the plain form here
/// multiplies the residual stream by ~0 and is the LOUD direction of that mistake.
fn centered(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let inv = rms_inv(x, eps);
    x.iter()
        .zip(w)
        .map(|(v, wi)| v * inv * (1.0 + wi))
        .collect()
}

/// `MuseGlimmerRMSNorm`: `_norm(x) * w`. The final `model.norm` only, in this chain.
fn plain(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let inv = rms_inv(x, eps);
    x.iter().zip(w).map(|(v, wi)| v * inv * wi).collect()
}

/// The weightless form over `rows` segments of `d`, times `scale`. The QK-norm (per head) and the
/// embedding norm (`rows = 1`, `d = hidden`, `scale = 1`) are the same operator.
fn weightless(x: &mut [f32], rows: usize, d: usize, eps: f32, scale: f32) {
    for r in 0..rows {
        let seg = &mut x[r * d..(r + 1) * d];
        let f = scale * rms_inv(seg, eps);
        seg.iter_mut().for_each(|v| *v *= f);
    }
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
    h: Vec<f32>,
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
            h: Vec::new(),
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
        self.h = plain(&h, self.t("model.language_model.norm.weight"), ep);
        let logits = matvec(self.t("lm_head.weight"), &self.h, c.vocab, hid);
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

/// Worst relative disagreement between two logit vectors, and where.
fn worst(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut w = (0.0f32, 0usize);
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs() / a[i].abs().max(b[i].abs()).max(1e-6);
        if d > w.0 {
            w = (d, i);
        }
    }
    w
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
    let mut e = Glimmer::new(
        root.join("out").to_str().unwrap(),
        gt,
        None,
        prompt.len() + max_new,
    )
    .unwrap();
    let mut r = Ref::new(&src, gt);

    // The engine samples only at the last prompt position, so the reference is run to that point
    // and the two are compared there; then each generated token is fed to both.
    let out = e.decode(&prompt, max_new, &[]).unwrap();
    assert_eq!(out.len(), max_new);

    let mut worst_seen = (0.0f32, 0usize, 0usize);
    let mut fed: Vec<u32> = prompt.clone();
    fed.extend_from_slice(&out[..out.len() - 1]);
    // Replay: the reference consumes exactly what the engine consumed, in the same order. Its
    // logits after position `p` are what the engine's were when it sampled `out[p - prompt.len()]`.
    let mut lg = Vec::new();
    for (pos, &t) in fed.iter().enumerate() {
        lg = r.step(t, pos);
    }
    // The engine's `logits()` holds the LAST sample, which is the one that produced `out.last()`.
    let (d, i) = worst(&lg, &e.logits().unwrap());
    if d > worst_seen.0 {
        worst_seen = (d, i, fed.len() - 1);
    }
    println!(
        "  final position {}: worst relative disagreement {:.3e} at logit {i} of {}",
        worst_seen.2, worst_seen.0, gt.vocab
    );
    assert!(
        worst_seen.0 < TOL,
        "the engine and the reference disagree by {:.3e} at logit {i} of position {} — the two \
         run the same arithmetic on the same weights and differ only in reduction order, so a \
         disagreement above {TOL:e} is a wiring defect and not rounding",
        worst_seen.0,
        worst_seen.2
    );

    // **The argmaxes agree too, and that is the WEAKER claim** — stated so the numbers above are
    // not read as merely reproducing it. A 12-way argmax over 9 steps is 9 integers; the logit
    // comparison is 108 floats, and the softcap this chain applies cannot move an argmax at all.
    let mut r2 = Ref::new(&src, gt);
    let mut picks = Vec::new();
    let mut feed = prompt.clone();
    for _ in 0..max_new {
        let mut l2 = Vec::new();
        for (pos, &t) in feed.iter().enumerate() {
            l2 = r2.step(t, pos);
        }
        let best = l2
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, v)| {
                if *v > b.1 { (i, *v) } else { b }
            })
            .0 as u32;
        picks.push(best);
        feed.push(best);
        r2 = Ref::new(&src, gt); // the reference replays from scratch; it holds no ring to reset
    }
    assert_eq!(picks, out, "the two chains emit different tokens");
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
    let (d, i) = worst(&lg, &e.logits().unwrap());
    println!("  permuted layer_types: worst relative disagreement {d:.3e} at logit {i}");
    assert!(
        d < TOL,
        "the engine and the reference disagree by {d:.3e} at logit {i} under a rotated \
         `layer_types` — a loop that derives the layer kind from the period rather than from the \
         array agrees with the shipped pattern and diverges here"
    );
}
