//! **The rewrite's Glimmer engine against MUSE GLIMMER itself — not against rivoli's reading
//! of it.** M7's exit gate.
//!
//! Every other Glimmer gate in this tree scores rivoli against something rivoli wrote.
//! `kernel_glimmer_{norm,attend,pointwise}.rs` score a KERNEL against a host oracle written
//! beside it — a real comparison, but of one operator, against a transcription. This one
//! builds an artifact out of the reference model's OWN 99 parameter tensors, runs
//! `GlimmerEngine`'s own prefill/decode loop on it, and compares the result to `tN.logits` —
//! the reference's own output. **No transcription is on the scoring path.**
//!
//! Ported from `old:tests/glimmer_reference.rs` (`wt/glimmer-s2` @ `6b7f496`); every constant
//! it uses carries the measurement that set it (`glimmer_anchor/mod.rs`), and every deviation
//! from that file is named below.
//!
//! # What is scored, and what is not
//!
//! The engine exposes exactly two readable tensors — `logits()` and `branch()` — so this gate
//! scores 15 of each salt's 1,099 float captures: seven logit vectors, one per step, and eight
//! post-FFN branches, one per layer. **That is not a shortfall to apologise for, it is the
//! division of labour**: the per-operator captures (`attend`, `rope`, `qk_norm`, `o_proj`, the
//! four sandwich norms) are what `kernel_glimmer_*.rs` score, one kernel at a time, against
//! their own measured `Rel` rows. What NO kernel gate can see is the composition — and that is
//! the only thing this file looks at.
//!
//! `glimmer_anchor_widths.rs` holds the deviceless half: the capture census, the schema, the
//! geometry, and the tolerance margins. It exists because this file is `#![cfg]`-gated off in
//! every build CI runs.
//!
//! # Where the artifact comes from, and why the converter is not on this path
//!
//! `old:` ran the real `convert_glimmer` binary from this gate. **Here it cannot**:
//! `CARGO_BIN_EXE_convert_glimmer` is defined only for targets in the package that declares
//! the binary (`rivoli`, i.e. `crates/cli`), and `rivoli-engine` neither depends on it nor
//! could without inverting the workspace DAG. So this file writes the artifact in-process with
//! the same `rivoli_artifact::format` primitives the converter writes it with.
//!
//! **That is a narrowing of the claim and it is stated rather than hidden.** This gate no
//! longer says "the shipped converter produces an artifact this engine decodes correctly".
//! `crates/cli/tests/glimmer_convert.rs` says that — bytes bf16-verbatim, norms widened to
//! f32, the completeness walk, the aux and EOS refusals — on a synthetic checkpoint. The two
//! compose: that gate owns the converter, this one owns the engine. What keeps them from
//! drifting is that `glimmer_anchor::stored` derives the f32/bf16 split from a tensor's RANK,
//! which is the same authority `glimmer::geometry::layer_bytes` charges the budget by and
//! `pin::place_norm`/`place_proj` demand at load — not a second copy of
//! `convert_glimmer::is_norm`'s name-suffix test.
//!
//! # The bf16 term, which is this gate's price
//!
//! **rivoli stores projections as bf16 and the reference computed in f32.** The comparisons
//! below therefore carry a weight-rounding term that has nothing to do with wiring: bf16 keeps
//! 8 mantissa bits, ~2e-3 relative per weight, compounding through eight layers. That term is
//! what `TOL` and `TOL_BRANCH` are set from, and it is why this gate separates wiring defects
//! (1e-1 and up) and cannot see the softcap's `tanh` (9.9e-5) or an eps transposition
//! (~5e-6). `old:tests/glimmer_chain.rs` was the gate for those — its two sides shared the
//! artifact's bf16 weights exactly, so it had no weight term at all and ran at 2e-5. **It is
//! not ported yet**, so those two defect classes are currently ungated in this tree; that is a
//! hole, and naming it here is the only thing standing in for it.
//!
//! **One deviation from `old:`'s measurement, in the safe direction.** `old:` rounded to bf16
//! by TRUNCATION (`x.to_bits() >> 16`); this tree has one owner for the conversion,
//! `rivoli_core::num::f32_to_bf16`, and it is round-to-nearest-even. Round-to-nearest's worst
//! error is half of truncation's, so a bound measured under truncation is a valid — and now
//! slightly conservative — bound here. Every test below PRINTS its observed worst so the first
//! green run can record this tree's own numbers rather than inheriting `old:`'s.
//!
//! # RED-PROOF RECIPE — run this before believing the first green
//!
//! P7: a check that has never been red is not evidence. Each row is a one-line edit,
//! reverted after. The magnitudes are `old:`'s, measured 2026-08-13 on these same vendored
//! bytes against a green worst of 4.8e-2. **PAID 2026-08-16, first GPU session — all four
//! rows red on device; the observed magnitudes and this tree's own green worsts live in
//! `docs/measurement/gate-red-proofs.md` §4**, together with the two operator false-greens
//! the paying itself produced (a stale binary behind an eaten build exit, twice).
//!
//! | # | edit | reddens | `old:` magnitude |
//! |---|---|---|---|
//! | 1 | `glimmer/geometry.rs::qk_scales` → `QkScale { q: 1.0, k: 1.0 }` | logits + branch | 6.7e-1 vs `TOL` 2e-1 |
//! | 2 | `glimmer/engine.rs` → `rotated: cfg.layer_rope_theta.iter().map(\|t\| *t == 0.0).collect()` | logits + branch | 1.1e0 |
//! | 3 | swap the `k` and `v` weight handles at their launch in `glimmer/forward.rs::attention` | logits + branch | 1.3e0 |
//! | 4 | in [`the_partition_moves_bytes_and_never_text`], pass `roomy` for both budgets | the P4 arm alone | `streamed 0` |
//!
//! Row 4 is the one to run first: it needs no source edit outside this file and it proves the
//! tight arm really streams, which is what makes rows 1-3 evidence about both residency
//! states rather than about the pinned one.
//!
//! **Two mutations that do NOT redden, and both are measurements rather than guesses.**
//! `eps_post → eps_pre` on the branch norms is GREEN here — `old:` measured the
//! transposition's own size at 1.6e-3–1.3e-2 against a bf16 weight floor of 4.7e-3–3.0e-2, so
//! the signal is 0.2x–0.6x the noise at every one of the 16 cells. And disabling the softcap
//! moves total variation from 1.249e-3 to 1.249e-3, because this anchor's logits span |0.24|
//! and `20·tanh(x/20)` is the identity there to 0.0002%. Neither is a hole in this file; both
//! are the reason `glimmer_chain.rs` has to be ported.
//!
//! # Running it
//!
//! Device tests, so: under the GPU flock, `-- --test-threads=1`, on the DEV profile (every
//! `debug_assert!` in the pin and the loop is live there and this fixture is far too small for
//! the timing to matter).
//!
//! ```text
//! flock /var/run/sys-gpu.lock -c 'cargo test -p rivoli-engine --features rocm \
//!     --test glimmer_anchor_decode -- --test-threads=1 --nocapture'
//! ```
//!
//! `--nocapture` is not optional if you want the numbers: every worst below is `println!`ed
//! and libtest swallows the output of a passing test.

// The whole binary, like every `kernel_glimmer_*.rs` sibling: a featureless build has no
// engine to run, and the deviceless claims about this fixture live in
// `glimmer_anchor_widths.rs` precisely so that gating this file off costs CI nothing.
#![cfg(feature = "rocm")]
// The panic-on-failure idiom; the fixture module carries the same allow and the same reason.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;
mod glimmer_anchor;

use common::{Got, Want, Weightless, worst_rel};
use glimmer_anchor::{
    Anchor, BRANCH_CAP, TOL, TOL_BRANCH, TOL_NLL, TOL_TV, anchors, float, ints, is_alias, stored,
};
use rivoli_artifact::format::SafeWriter;
use rivoli_artifact::glimmer::GLIMMER_LAYER_TENSORS;
use rivoli_artifact::glimmer_config::{GlimmerConfig, GlimmerTextConfig};
use rivoli_engine::glimmer::engine::GlimmerEngine;
use rivoli_engine::glimmer::geometry;
use rivoli_engine::seam::GenSpec;
use std::path::PathBuf;

/// An artifact directory that removes itself on drop.
///
/// **Panic-safe, which a `remove_dir_all` at the end of a test is not** — a failing
/// assertion skips it and leaves the fixture behind, and the next run with the same pid
/// reuses it. `tag` carries the salt so the two arms cannot collide.
struct Artifact(PathBuf);

impl Artifact {
    fn path(&self) -> &str {
        self.0.to_str().expect("the artifact path is utf-8")
    }
}

impl Drop for Artifact {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Write the reference's own parameters as a Muse Glimmer artifact.
///
/// The config is the shipped wrapper with `text_config` replaced by the golden's own
/// `tiny_config`, so every scalar the loop reads — both epsilons, `qk_scale_factor`,
/// `output_multiplier`, `final_logit_softcapping`, `sliding_window`, `layer_types`,
/// `layer_rope_theta` — comes from the run that produced the logits rather than from this
/// file.
///
/// **The projections go out as bf16, which is where this gate's floor comes from.** That
/// is not a choice: bf16 is what the artifact stores, so rounding here is the same
/// rounding the shipped converter would do, applied where it can be seen.
fn write_artifact(a: &Anchor) -> Artifact {
    let root = std::env::temp_dir().join(format!(
        "rivoli-glimmer-anchor-{}-{}",
        a.name,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create the artifact directory");

    let mut w = SafeWriter::new();
    let mut placed = 0usize;
    for (name, shape, vals) in a.weights.floats.iter().filter(|(n, _, _)| !is_alias(n)) {
        let (dt, bytes) = stored(shape, vals);
        w.add(name.clone(), dt, shape.clone(), bytes);
        placed += 1;
    }
    w.write(&format!("{}/resident.safetensors", root.display()))
        .expect("write the artifact weights");

    std::fs::write(root.join("config.json"), a.config_json().to_string())
        .expect("write the config");
    let art = Artifact(root);
    // The engine's own loader, on the bytes just written: a config this fixture got wrong
    // fails HERE, with the schema's own message, rather than as a wrong number later.
    let back = GlimmerConfig::load(art.path()).expect("the artifact re-opens as a Glimmer one");
    assert_eq!(
        placed,
        3 + back.text.n_layers * GLIMMER_LAYER_TENSORS.len(),
        "{}: the artifact is not the whole parameter set",
        a.name
    );
    art
}

/// The budget that pins every layer, and the one that forces a streaming split — both
/// derived from the engine's own footprint arithmetic rather than picked.
///
/// `floor_of` states everything a run pays before one weight is resident; adding `k` layers'
/// worth of bytes to it is a budget that admits about `k` of them. Returns
/// `(all_resident, about_three_layers)`.
///
/// **`n_ctx` here is the anchor's FULL run** (prompt plus every emitted token) while a given
/// forward is opened at `prompt.len() + 1`, so the floor charged is a slight over-estimate and
/// the split arm may pin one more layer than three. That is deliberate and the callers do not
/// predict the split: the roomy arm asserts nothing streams, the tight arm asserts that
/// something is pinned AND something streams AND a slot was filled. A test that asserted "3
/// and 5" would be asserting `partition`'s answer, which is `rivoli-core`'s own gate to keep —
/// what THIS file needs is that both residency paths were entered.
fn budgets(t: &GlimmerTextConfig, n_ctx: usize) -> (usize, usize) {
    let layer = geometry::layer_bytes(t).expect("a layer's bytes");
    let roomy = geometry::floor_of(t, n_ctx, 0).expect("the slotless floor");
    let tight = geometry::floor_of(t, n_ctx, geometry::STREAM_SLOTS).expect("the slotted floor");
    (
        roomy.total().0 as usize + t.n_layers * layer,
        tight.total().0 as usize + 3 * layer,
    )
}

/// One forward over `prompt`, with the engine handed back for a readback.
///
/// `ngen: 1` is exactly one forward: `Emit::offer` pushes the prefill's pick and then
/// returns false on `ids.len() < ngen`, so the decode loop's body never runs and both
/// `logits()` and `branch()` describe the last PROMPT position. Three call sites need
/// that fact to be true; it is spelled once, here.
fn one_forward<'a>(
    dir: &str,
    t: &'a GlimmerTextConfig,
    prompt: &[u32],
    capacity: usize,
) -> (GlimmerEngine<'a>, Vec<u32>) {
    let n_ctx = prompt.len() + 1;
    let mut e = GlimmerEngine::open(dir, t, capacity, n_ctx).expect("build the engine");
    let spec = GenSpec {
        prompt,
        ngen: 1,
        eos: &[],
    };
    let out = e
        .decode(spec, &mut |_| true)
        .expect("one forward over the reference's parameters");
    (e, out.ids)
}

/// Softmax, numerically stable — `rivoli_core`'s, in place, so the distribution this gate
/// reads is the one the engine's own router formula produces.
fn probs(v: &[f32]) -> Vec<f32> {
    let mut p = v.to_vec();
    assert!(
        p.iter().all(|x| x.is_finite()),
        "logits are not finite; every metric below would be meaningless"
    );
    rivoli_core::routing::softmax(&mut p);
    p
}

/// Total variation distance, `0.5·Σ|p−q|` — 0 for identical distributions, 1 for disjoint.
///
/// Chosen over KL because it is bounded and symmetric: a KL blows up on any token the
/// reference gives near-zero mass, which makes the number a fact about the vocabulary tail
/// rather than about the defect.
fn total_variation(a: &[f32], b: &[f32]) -> f32 {
    let (p, q) = (probs(a), probs(b));
    0.5 * p.iter().zip(&q).map(|(x, y)| (x - y).abs()).sum::<f32>()
}

/// `-ln p(token)`, the quantity a perplexity ladder is built from.
fn nll(v: &[f32], token: usize) -> f32 {
    -probs(v)[token].max(f32::MIN_POSITIVE).ln()
}

/// The token ids the reference's TEXT stack actually embedded, recovered from its own
/// first capture — which is not always `prompt.ids`.
///
/// **The anchor's prompt is a random draw over the tiny vocabulary and it hit the
/// multimodal placeholders.** The driver builds a `MuseGlimmerForConditionalGeneration`
/// with `image_token_id = 59` and `video_token_id = 58` shrunk into the tiny vocab, and
/// the wrapper substitutes at those positions before the text model sees them — so
/// `t0.embed_norm.out` holds the embedding of id 0 there. Feeding the raw `prompt.ids`
/// compares a text-only decode against a run that was not text-only: `old:` measured the
/// disagreement at 8.7e-1 at step 0 and every step after.
///
/// **Recovered rather than hardcoded**, which is the difference between adapting to the
/// reference and papering over a divergence. This reads one capture and the embedding
/// table and asks which row the reference used; it does not know 58 or 59, so a future
/// golden with different placeholders needs no change here. Everything downstream — eight
/// layers, the norms, the attention, the logits — is still scored with nothing recovered.
///
/// The widths come from the VALIDATED config rather than from `tiny_config` directly, so
/// this helper and the engine it feeds cannot disagree about the embedding's shape — the
/// assertion below would then be checking one model's table against another's.
fn effective_prompt(a: &Anchor, t: &GlimmerTextConfig) -> Vec<u32> {
    let (hid, vocab) = (t.hidden, t.vocab);
    let (shape, cap) = float(&a.caps, "t0.embed_norm.out");
    let rows = shape.iter().product::<usize>() / hid;
    let (eshape, emb) = a
        .weights
        .floats
        .iter()
        .find(|(n, _, _)| n == "model.language_model.embed_tokens.weight")
        .map(|(_, s, v)| (s, v))
        .expect("the weight set holds the embedding");
    assert_eq!(eshape, &vec![vocab, hid], "{}: embedding shape", a.name);
    // The same weightless norm the reference puts on the embedding, applied to every
    // candidate row once.
    let mut normed = emb.clone();
    Weightless {
        rows: vocab,
        d: hid,
        eps: t.rms_norm_eps as f32,
        scale: 1.0,
    }
    .apply(&mut normed);
    let raw = ints(&a.caps, "prompt.ids");
    let out: Vec<u32> = (0..rows)
        .map(|r| resolve_row(a, &cap[r * hid..(r + 1) * hid], normed.chunks_exact(hid), r))
        .collect();
    // A census, so the mechanism cannot quietly stop applying: salt 1 substitutes two
    // positions and salt 2 one. Zero would be fine for the comparison but would also mean
    // this function had stopped being exercised, and a helper nothing exercises is where
    // the next wrong assumption hides.
    let subs = out
        .iter()
        .zip(raw)
        .filter(|(g, w)| **g as i64 != **w)
        .count();
    println!(
        "  {}: {subs} of {rows} prompt positions were multimodal placeholders",
        a.name
    );
    assert!(
        out.iter()
            .zip(raw)
            .filter(|(g, w)| **g as i64 != **w)
            .all(|(g, _)| *g == 0),
        "{}: a substituted position resolved to something other than id 0, which is not \
         the mechanism this function documents",
        a.name
    );
    out
}

/// Which embedding row `want` is, with the uniqueness that keeps it from being a guess:
/// the match must be tight AND the runner-up far, or the recovery is reporting a
/// coincidence rather than an identification.
fn resolve_row<'c>(
    a: &Anchor,
    want: &[f32],
    cands: impl Iterator<Item = &'c [f32]>,
    at: usize,
) -> u32 {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
    let (mut best, mut second) = ((f32::INFINITY, 0usize), f32::INFINITY);
    for (id, cand) in cands.enumerate() {
        let d = cand
            .iter()
            .zip(want)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
            / scale;
        if d < best.0 {
            second = best.0;
            best = (d, id);
        } else if d < second {
            second = d;
        }
    }
    assert!(
        best.0 < 1e-5 && second > 100.0 * best.0.max(1e-9),
        "{}: position {at} does not resolve to one embedding row — best id {} at {:.2e}, \
         runner-up at {second:.2e}. The recovery is only sound if the reference's \
         embedding is what this capture holds",
        a.name,
        best.1,
        best.0
    );
    best.1 as u32
}

/// **The engine's loop over the reference's parameters, against the reference's logits.**
///
/// One engine per step, and the step's logits are the reference's `tN.logits`: the engine
/// samples only at the last position it is handed, so a prefix ending at that step's
/// position is the only way to see that step's logit vector — and it makes a red name the
/// step it started at.
#[test]
fn the_engine_reproduces_muse_glimmers_own_logits() {
    let (mut ran, mut any_tv, mut any_nll) = (0usize, 0.0f32, 0.0f32);
    for a in &anchors() {
        let art = write_artifact(a);
        let cfg = a.config();
        let emitted: Vec<u32> = ints(&a.caps, "emitted.ids")
            .iter()
            .map(|v| *v as u32)
            .collect();
        let prompt = effective_prompt(a, &cfg.text);
        let (capacity, _) = budgets(&cfg.text, a.positions());
        let mut worst = (0.0f32, 0usize);
        for step in 0..emitted.len() {
            let mut fed = prompt.clone();
            fed.extend_from_slice(&emitted[..step]);
            let (e, ids) = one_forward(art.path(), &cfg.text, &fed, capacity);
            let got = e.logits().expect("the step's logit vector");
            let want = float(&a.caps, &format!("t{step}.logits")).1;
            let d = worst_rel(Got(&got), Want(want));
            assert!(
                d < TOL,
                "{}: step {step} disagrees with the reference by {d:.3e} — this is the \
                 FIRST step that diverges. The measured bf16 weight-rounding envelope is \
                 9.3e-3 to 6.8e-2, so anything above {TOL:e} is a wiring defect and not \
                 the format",
                a.name
            );
            if d > worst.0 {
                worst = (d, step);
            }
            // **PROBABILITY SPACE, a different question from the logits above.** The
            // softcap is argmax-invariant by construction, so neither the `ids`
            // assertion below nor any greedy decode can see it, and a relative agreement
            // on the logit VECTOR can still hide a reshaping of the distribution that
            // sampling and NLL read.
            let tv = total_variation(&got, want);
            let dn = (nll(&got, emitted[step] as usize) - nll(want, emitted[step] as usize)).abs();
            any_tv = any_tv.max(tv);
            any_nll = any_nll.max(dn);
            assert!(
                tv < TOL_TV,
                "{}: step {step} disagrees IN PROBABILITY by {tv:.3e} total variation. The \
                 logits agreed to {d:.3e}, so this is a distribution defect the logit gate \
                 and the argmax both pass",
                a.name
            );
            assert!(
                dn < TOL_NLL,
                "{}: step {step}'s NLL for the reference's own token differs by {dn:.3e} nats",
                a.name
            );
            // **The token too, and it is the WEAKER claim** — a 61-way argmax is one
            // integer where the comparison above is 61 floats. Said so a green argmax is
            // not read as the evidence: this logit path is argmax-invariant by
            // construction, so the pick would agree with `output_multiplier` and the
            // softcap both missing.
            assert_eq!(
                ids,
                vec![emitted[step]],
                "{}: step {step} emitted a different token than the reference",
                a.name
            );
            ran += 1;
        }
        println!(
            "  {}: {} steps, worst logit {:.3e} at step {}",
            a.name,
            emitted.len(),
            worst.0,
            worst.1
        );
    }
    assert_eq!(ran, 14, "both salts must contribute all seven steps");
    // **Anti-vacuity for the two probability-space metrics, and not decoration.** Both
    // asserts are upper bounds and `dNLL` is a DIFFERENCE of two calls to one function, so
    // a `total_variation` comparing a vector against itself, or an `nll` whose
    // `MIN_POSITIVE` clamp started firing on both sides, would return 0 and pass forever.
    // Both are measured non-zero (TV 1.249e-3 / 1.464e-3, dNLL 5.316e-3 / 6.305e-3), so
    // this costs nothing.
    assert!(
        any_tv > 0.0 && any_nll > 0.0,
        "the probability-space metrics returned zero on every step ({any_tv:.3e} TV, \
         {any_nll:.3e} dNLL) — they are comparing something against itself"
    );
}

/// **Every layer's post-FFN branch against the reference's own capture — 16 cells, one
/// residual add UPSTREAM of where the logit test reads.**
///
/// **Truncating the model is what selects a layer.** The engine keeps one branch buffer
/// and every layer overwrites it, so a run whose config ends at layer `l` leaves `l`'s
/// branch in it. Layers `0..l` compute identically either way — the truncation removes
/// work after the read, not before it — and the pin simply builds fewer layers from the
/// same artifact.
///
/// **This is NOT the eps gate it looks like**, and that is measured rather than suspected:
/// see the module header's two-mutations-that-do-not-redden note. What it IS: per-layer,
/// per-salt scoring of the engine against Muse Glimmer's own intermediates, two more cells
/// than the logit test has and every one of them before the residual add that washes
/// defects out.
#[test]
fn every_layers_branch_matches_the_reference() {
    let mut cells = 0usize;
    let mut worst = (0.0f32, 0usize, "");
    for a in &anchors() {
        let art = write_artifact(a);
        let cfg = a.config();
        let prompt = effective_prompt(a, &cfg.text);
        let (hid, layers) = (cfg.text.hidden, cfg.text.n_layers);
        let (capacity, _) = budgets(&cfg.text, a.positions());
        for l in 0..layers {
            let mut t = cfg.text.clone();
            t.n_layers = l + 1;
            t.layer_types.truncate(l + 1);
            t.layer_rope_theta.truncate(l + 1);
            let (e, _) = one_forward(art.path(), &t, &prompt, capacity);
            let got = e.branch().expect("the last layer's branch");
            // The capture is `[1, rows, hidden]` over the whole prompt; the engine's
            // buffer holds the row of the LAST position it ran.
            let (shape, want) = float(&a.caps, &format!("t0.L{l}.{BRANCH_CAP}"));
            assert_eq!(
                shape.to_vec(),
                vec![1, prompt.len(), hid],
                "{}: capture shape at L{l}",
                a.name
            );
            let last = &want[(prompt.len() - 1) * hid..];
            assert!(
                last.iter().any(|v| *v != 0.0),
                "{}: L{l}'s captured branch is all zero, so the score below is against \
                 nothing",
                a.name
            );
            let d = worst_rel(Got(&got), Want(last));
            assert!(
                d < TOL_BRANCH,
                "{}: layer {l}'s post-FFN branch disagrees with the reference by {d:.3e}, \
                 past the {TOL_BRANCH:e} bf16 weight floor measured at 4.7e-3 (L0) to \
                 3.0e-2 (L6)",
                a.name
            );
            if d > worst.0 {
                worst = (d, l, a.name);
            }
            cells += 1;
        }
    }
    println!(
        "  {cells} layer branches scored, worst {:.3e} at L{} of {}",
        worst.0, worst.1, worst.2
    );
    assert_eq!(cells, 16, "both salts must contribute all eight layers");
}

/// **P4 on the reference's own weights: the budget moves bytes and never text.**
///
/// The pin partitions whole layers into a resident prefix and a streamed suffix, and the
/// streamed ones are refilled through a slot by host memcpy on every visit. Nothing above
/// asks whether that path runs at all — both other tests use a budget that pins
/// everything, so `Slot::fill` is never entered and the anchor would certify an engine
/// whose streaming half is broken.
///
/// **Bit-identical, not within a tolerance.** The two arms read the same bytes from the
/// same artifact through the same launchers in the same order; the only difference is
/// WHERE those bytes were copied from. Anything but equality is a defect, and a tolerance
/// here would hide exactly the class of defect this test exists for.
#[test]
fn the_partition_moves_bytes_and_never_text() {
    for a in &anchors() {
        let art = write_artifact(a);
        let cfg = a.config();
        let prompt = effective_prompt(a, &cfg.text);
        let (roomy, tight) = budgets(&cfg.text, a.positions());
        let (all, _) = one_forward(art.path(), &cfg.text, &prompt, roomy);
        let (pinned, streamed) = all.residency();
        assert_eq!(
            streamed, 0,
            "{}: the roomy budget streamed {streamed} layers, so the two arms below are \
             the same arm",
            a.name
        );
        assert_eq!(pinned, cfg.text.n_layers, "{}: the roomy budget", a.name);
        let want = all.logits().expect("the all-resident logits");
        drop(all);

        let (split, _) = one_forward(art.path(), &cfg.text, &prompt, tight);
        let (p2, s2) = split.residency();
        let (hits, fills) = split.slot_stats();
        println!(
            "  {}: {p2} pinned / {s2} streamed, {fills} slot fills, {hits} hits",
            a.name
        );
        assert!(
            p2 > 0 && s2 > 0 && fills > 0,
            "{}: the tight budget pinned {p2} and streamed {s2} with {fills} fills — this \
             arm must exercise BOTH the resident and the streaming path",
            a.name
        );
        assert_eq!(
            split.logits().expect("the streamed logits"),
            want,
            "{}: the same prompt through the same artifact gave different logits at a \
             different budget. Residency may move bytes; it may never move text (P4)",
            a.name
        );
    }
}
