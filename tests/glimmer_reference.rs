//! **The engine against MUSE GLIMMER — not against rivoli's reading of it.**
//!
//! Every other Glimmer gate in this tree scores one of two things. The S2 suites score a KERNEL
//! against the anchor's captures, which is a real comparison against the model but covers one
//! operator. `tests/glimmer_chain.rs` scores the whole LOOP against a host reference — but that
//! reference is transcribed from `glimmer-architecture.md` by the same author as the engine, so a
//! defect written into both sides passes, and that file names three readings where it could
//! happen.
//!
//! This one closes the gap: it builds a checkpoint from the reference model's OWN parameters, runs
//! rivoli's own loop on it, and compares the logits to `tN.logits` — the reference's own output. No
//! transcription is on the scoring path.
//!
//! # What made it possible, and what it cost
//!
//! The goldens hold 1,099 captured activations and, until 2026-08-13, one weight tensor per layer.
//! The engine cannot be run on captures, so the driver's `weights_capture` was extended to export
//! the whole tiny model — 107 tensors, ~485k floats, 2,065,185 B per salt. The driver's own
//! docstring had specified exactly that change and priced it ("a decision to make deliberately,
//! not by accretion"), and `glimmer-anchor.sh` re-proves on every run that it was ADDITIVE: both
//! text goldens regenerate byte-identical.
//!
//! # The bf16 term, which is this gate's price and `glimmer_chain.rs`'s absence
//!
//! **rivoli stores projections as bf16 and the reference computed in f32.** So the comparison
//! below carries a weight-rounding term that has nothing to do with wiring: bf16 keeps 8 mantissa
//! bits, ~2e-3 relative per weight, compounding through eight layers. That term is measured and
//! printed, and it is what [`TOL`] is set from.
//!
//! **The consequence is the thing to understand before reading a green run here.** This gate
//! separates wiring defects, which move logits by 1e-1 and up. It CANNOT see the softcap's `tanh`
//! (9.9e-5) or an eps transposition (~5e-6) — both are far under the bf16 floor. `glimmer_chain.rs`
//! is the gate for those: its two sides share the artifact's bf16 weights exactly, so it has no
//! weight term at all and runs at 2e-5. **Neither file subsumes the other**, and the pair is the
//! argument: one has the real model and a coarse floor, the other a fine floor and a transcribed
//! reference.
//!
//! # What it catches, measured
//!
//! Each mutation applied to `glimmer_gpu.rs` and run, 2026-08-13, against a green worst of
//! **4.8e-2**:
//!
//! | mutation | worst | first step |
//! |---|---|---|
//! | `wk` / `wv` swapped | **1.3e0** | 0 |
//! | `rotated[l]` inverted — the NoPE layers rotate | **1.1e0** | 0 |
//! | `qk_scale_factor` dropped from Q | **6.7e-1** | 0 |
//! | **`eps_post` → `eps_pre` on the branch norms** | **GREEN** | — |
//!
//! **That last row is the one to read.** The plan's open-items register expected this gate to close
//! the eps question; it does not. The transposition is invisible to BOTH whole-chain gates — under
//! `glimmer_chain.rs`'s 2e-5 because the branch statistic is O(1), and under this gate's bf16
//! envelope for the same reason on the reference's own weights. It stays open, and now with the
//! stronger statement: it is not a fixture artefact.
//!
//! Removing the bf16 term would mean regenerating the goldens from bf16-rounded weights, which
//! moves every capture and re-prices every S2 tolerance. Declined here, and recorded rather than
//! left to be rediscovered.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

// ONE include, not two: `common/mod.rs` reached through both `mod common;` and this module's
// `device` re-export is the same file loaded twice, which rustc warns about — and the whole reason
// `glimmer_fixture.rs` re-exports the device helpers is so a fixture needs one `#[path]`.
#[path = "common/glimmer_fixture.rs"]
mod fixture;
use fixture::{
    FixtureTensor, TempRoot, float, goldens, run_convert_glimmer, weight_sets, worst_rel,
};
use rivoli::artifact::model as gm;
use rivoli::glimmer_gpu::Glimmer;

/// Relative tolerance, in [`worst_rel`]'s metric — the **bf16 weight-rounding envelope**, MEASURED,
/// not an arithmetic floor.
///
/// > **The first version of this constant carried numbers I had not measured** — "measured at
/// > 2.5e-3 and 2.6e-3" — written as placeholders and left in. They were wrong by an order of
/// > magnitude and the gate went red on a correct engine because of it. The real experiment: run
/// > the whole chain in f64 twice on the reference's own parameters, once exact and once with every
/// > weight rounded to bf16, and score each against `tN.logits`.
/// >
/// > | | worst over 7 steps x 2 salts |
/// > |---|---|
/// > | f64, exact weights | **3.8e-6** — the transcription reproduces the reference |
/// > | f64, bf16 weights | **6.8e-2** (range 9.3e-3 - 6.8e-2) — the format's own error |
/// >
/// > rivoli's engine sits inside the second row, which is the answer this gate can give: it agrees
/// > with Muse Glimmer to the precision a bf16 artifact allows. **The first row is the load-bearing
/// > one** — it says an independent f64 transcription of §3-§5 reproduces the reference to 3.8e-6,
/// > so the architecture doc is right and `glimmer_chain.rs`'s oracle is scoring against a correct
/// > reading.
///
/// 2e-1 is ~3x the worst observed envelope. That margin is not generosity: the envelope is one
/// realisation of a rounding process, and a re-drawn salt would land elsewhere in the same range.
/// **What this costs is resolution** — see the header. Defects under ~1e-1 belong to
/// `glimmer_chain.rs`, which has no weight term and runs at 2e-5.
const TOL: f32 = 2e-1;

/// The same envelope one layer up, where the branch is scored rather than the logits — MEASURED
/// per layer, exactly as [`TOL`] was, and for the same reason: the first value here was a guess
/// (2e-2, "tighter, because it is one layer's error") and layer 3 went red on a correct engine.
///
/// The bf16 floor on this tensor GROWS with depth, because the branch is computed from a residual
/// stream that has already accumulated it: **4.7e-3 at L0 to 3.0e-2 at L6**, over both salts.
/// 1e-1 is ~3.3x the worst.
const TOL_BRANCH: f32 = 1e-1;

/// Softmax, numerically stable. The distribution the model actually emits, which is what the
/// softcap shapes and what neither the logit gate nor a greedy argmax reads.
fn softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        m.is_finite(),
        "logits are not finite; softmax would be meaningless"
    );
    let e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

/// Total variation distance, `0.5·Σ|p−q|` — 0 for identical distributions, 1 for disjoint ones.
///
/// Chosen over KL because it is bounded and symmetric: a KL blows up on any token the reference
/// gives near-zero mass and this one does not, which makes the number a fact about the vocabulary
/// tail rather than about the defect.
fn total_variation(a: &[f32], b: &[f32]) -> f32 {
    let (p, q) = (softmax(a), softmax(b));
    0.5 * p.iter().zip(&q).map(|(x, y)| (x - y).abs()).sum::<f32>()
}

/// `-ln p(token)`, the quantity a perplexity ladder is built from.
fn nll(v: &[f32], token: usize) -> f32 {
    -softmax(v)[token].max(f32::MIN_POSITIVE).ln()
}

/// Total variation between the engine's output distribution and the reference's — MEASURED at
/// **1.249e-3 (text-1) and 1.464e-3 (text-2)**, and set at ~3.4x the worse, the same margin
/// convention as [`TOL`] and for the same reason.
///
/// **What this buys, stated narrowly because it is narrower than it looks.** It is a second NORM
/// on the same logits, in the space the model's output actually lives in: `worst_rel` is a
/// relative-max over the vector, this is an L1 over the distribution, and a defect concentrated
/// on one high-probability token moves this far more than it moves that. It is NOT independent
/// evidence, and it is NOT a softcap gate — see [`TOL_NLL`].
const TOL_TV: f32 = 5e-3;

/// `|NLL_got − NLL_want|` for the reference's own emitted token — MEASURED at 5.316e-3 and
/// 6.305e-3, set at ~3.2x.
///
/// **Its resolution is a ~10% temperature error, and that is measured too.** Scaling only the
/// engine's logits: x1.05 scores 1.2e-2 and PASSES, x1.10 scores 2.231e-2 and reddens, x1.25
/// scores 3.602e-2. The anchor's logits span |0.24| over a vocabulary of 61, so the distribution
/// is nearly uniform and one token's NLL barely moves — this is a coarse instrument HERE, on this
/// fixture, and tightening the bound to catch 5% would leave 1.6x over the clean value, which is
/// the exact thinness that made §1.1's gates go red on a correct engine.
///
/// A common-mode mutation cannot redden it at all — scaling BOTH sides cancels in the difference.
/// That is a property of a differential metric rather than a hole, and it is why the red proof
/// above perturbs one side only.
///
/// # This gate CANNOT see the softcap, and that is measured rather than suspected
///
/// Register item 4.2 asked for the softcap checked in probability space, on the grounds that S2
/// proved argmax-invariance so no greedy gate can see it. Both metrics above were added for it.
/// **They cannot see it either, and neither can any tolerance, because the anchor's logits are in
/// the wrong range entirely:**
///
/// | | max &#124;logit&#124; | what `20·tanh(x/20)` does there |
/// |---|---|---|
/// | this anchor | **0.24** | changes it by **0.0002%** — arithmetically the identity |
/// | the real 30B on a trained prompt | **36.27** | compresses it to **18.96**, a 48% cut |
///
/// Red-proved by disabling the softcap on this fixture: TV moved from 1.249e-3 to 1.249e-3 and
/// dNLL from 5.316e-3 to 5.311e-3. Not "a small signal" — no signal, because `tanh` is linear
/// near zero and the anchor never leaves that region. `glimmer-integration.md` S4 item 4
/// predicted exactly this ("the anchor provably cannot"); this is the measurement behind it.
///
/// **What the softcap actually does, priced on the real model** (`--prompt "Say hello."`, one
/// step, capped against `cap = 1e30`):
///
/// * top1−top2 logit gap **33.46 → 18.05**
/// * total variation between the two distributions: **1.92e-8** — invisible, because both
///   saturate at `p(top1) = 1.0`
/// * runner-up token probability **2.95e-15 → 1.45e-8**, a factor of **4.9 million**, which is
///   **15.4 nats** of NLL
/// * entropy **1.28e-13 → 4.05e-7 nats**, a factor of 3.2 million
///
/// So the softcap is invisible to greedy decode, invisible to total variation on a confident
/// prompt, and enormous in the tail — which is the regime perplexity lives in. **The instrument
/// for it is teacher-forced NLL on non-argmax tokens, i.e. S4 item 4's dNLL ladder**, and that
/// needs a Glimmer `--ppl` path, which does not exist (`src/eval.rs` has `run` and `run_v4`).
const TOL_NLL: f32 = 2e-2;

/// The engine's loop, over the reference's parameters, compared to the reference's logits.
///
/// **Both salts**, because that is this anchor's discipline everywhere else: a defect that reddens
/// on one draw and not the other is a fact about the draw.
#[test]
fn the_engine_reproduces_muse_glimmers_own_logits() {
    let mut ran = 0;
    for (g, w) in goldens().iter().zip(weight_sets()) {
        assert_eq!(
            g.name, w.0,
            "the golden and its weight set are the same salt"
        );
        let root = TempRoot::new(&format!("glimmer-ref-{}", g.name));
        let cfg = write_checkpoint(&root, g, &w.1);
        let gt = &cfg.text;

        let prompt = effective_prompt(g, &w.1);
        let emitted: Vec<u32> = fixture::ints_of(g, "emitted.ids")
            .iter()
            .map(|v| *v as u32)
            .collect();
        assert_eq!(
            prompt.len(),
            g.steps().0,
            "prompt.ids is not prompt_len long"
        );
        assert_eq!(
            emitted.len(),
            g.steps().1 + 1,
            "emitted.ids should be one prefill pick plus every decode step"
        );

        // **One engine per step, and the step's logits are the reference's `tN.logits`.** The
        // engine samples only at the last position it is handed, so a prefix ending at the step's
        // position is the only way to see that step's logit vector — the same construction
        // `glimmer_chain.rs` uses, and it is what makes a red name the step it started at.
        let mut worst = (0.0f32, 0usize);
        let (mut worst_tv, mut worst_nll) = ((0.0f32, 0usize), (0.0f32, 0usize));
        for step in 0..emitted.len() {
            let mut fed = prompt.clone();
            fed.extend_from_slice(&emitted[..step]);
            let mut e =
                Glimmer::new(root.join("out").to_str().unwrap(), gt, None, fed.len() + 1).unwrap();
            let picked = e.decode(&fed, 1, &[]).unwrap();
            let want = float(&g.g, &format!("t{step}.logits")).1;
            assert_eq!(want.len(), gt.vocab, "t{step}.logits is not one vocab row");
            let got = e.logits().unwrap();
            let d = worst_rel(&got, want);
            assert!(
                d < TOL,
                "{}: step {step} disagrees with the reference by {d:.3e} — this is the FIRST step \
                 that diverges. The measured bf16 weight-rounding envelope is 9.3e-3 to 6.8e-2, so \
                 anything above {TOL:e} is a wiring defect and not the format",
                g.name
            );
            if d > worst.0 {
                worst = (d, step);
            }
            // **PROBABILITY SPACE, which is a different question from the logits above.** The
            // softcap is `20·tanh(x/20)`: it is argmax-invariant by construction, so neither the
            // `picked` assertion below nor any greedy decode can see it, and a relative agreement
            // on the logit VECTOR can still hide a large reshaping of the distribution the model
            // actually emits — which is what sampling and NLL read.
            let tv = total_variation(&got, want);
            let dn = (nll(&got, emitted[step] as usize) - nll(want, emitted[step] as usize)).abs();
            if tv > worst_tv.0 {
                worst_tv = (tv, step);
            }
            if dn > worst_nll.0 {
                worst_nll = (dn, step);
            }
            assert!(
                tv < TOL_TV,
                "{}: step {step} disagrees with the reference IN PROBABILITY by {tv:.3e} total \
                 variation. The logits agreed to {d:.3e}, so this is a distribution defect the \
                 logit gate and the argmax both pass",
                g.name
            );
            assert!(
                dn < TOL_NLL,
                "{}: step {step}'s NLL for the reference's own token differs by {dn:.3e} nats",
                g.name
            );
            // **The token too, and it is the WEAKER claim** — a 61-way argmax is one integer where
            // the comparison above is 61 floats. Stated so a green argmax is not read as the
            // evidence: this model's logit path is argmax-invariant by construction, so the pick
            // would agree even with `output_multiplier` and the softcap both missing.
            assert_eq!(
                picked,
                vec![emitted[step]],
                "{}: step {step} emitted a different token than the reference",
                g.name
            );
            ran += 1;
        }
        println!(
            "  {}: {} steps against the reference, worst logit {:.3e} at step {}, worst TV \
             {:.3e} at step {}, worst dNLL {:.3e} at step {}",
            g.name,
            emitted.len(),
            worst.0,
            worst.1,
            worst_tv.0,
            worst_tv.1,
            worst_nll.0,
            worst_nll.1
        );
    }
    // A census: both salts, every step. The loop above is over two vectors either of which could
    // become empty without a single assertion failing.
    assert_eq!(ran, 14, "both salts must contribute all seven steps");
}

/// Write the reference's parameters as a Muse Glimmer checkpoint and convert it.
///
/// The config is the SHIPPED one with `text_config` replaced by the golden's own `tiny_config`,
/// so every scalar the loop reads — the two epsilons, `qk_scale_factor`, `output_multiplier`,
/// `final_logit_softcapping`, `sliding_window`, `layer_types`, `layer_rope_theta` — comes from the
/// run that produced the logits rather than from this file.
///
/// **The weights go out as bf16, which is where this gate's floor comes from.** `Bf16Weight` is
/// what the artifact stores, so rounding here is not a choice — it is the same rounding the engine
/// would do to any f32 checkpoint, applied where it can be seen.
fn write_checkpoint(
    root: &TempRoot,
    g: &fixture::Golden,
    w: &rivoli::golden::GoldenSet,
) -> gm::GlimmerConfig {
    let src = root.join("src");
    fixture::write_glimmer_config(&src, &g.c);

    let tensors: Vec<FixtureTensor> = w
        .floats
        .iter()
        // The pre-2026-08-13 alias for `self_attn.gate_proj`, still exported so the file that reads
        // it did not have to move with the dump. A checkpoint carrying it would fail the
        // converter's own name census, which is the census doing its job.
        .filter(|(n, _, _)| !n.starts_with('L'))
        .map(|(n, shape, vals)| {
            let bytes = fixture::u16b(&fixture::to_bf16(vals));
            (n.clone(), shape.clone(), bytes)
        })
        .collect();
    assert_eq!(
        tensors.len(),
        3 + g.n("num_hidden_layers") * gm::GLIMMER_LAYER_TENSORS.len(),
        "the checkpoint is not the whole parameter set"
    );
    fixture::write_safetensors(&src.join("model-00001-of-00001.safetensors"), &tensors);
    fixture::write_index(&src, &tensors);
    // A stop token the reference never emits, so nothing here terminates early: the comparison is
    // over every step the golden captured, and an EOS that happened to match would silently
    // shorten it.
    let vocab = g.n("vocab_size") as u32;
    fixture::write_glimmer_aux(&src, &[vocab - 1]);

    let o = run_convert_glimmer(&src, &root.join("out"));
    assert!(
        o.status.success(),
        "converting the reference's parameters failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    gm::load_config(root.join("out").to_str().unwrap()).unwrap()
}

/// The token ids the reference's TEXT stack actually embedded, recovered from its own first
/// capture — which is not always `prompt.ids`.
///
/// **The anchor's prompt is a random draw over the tiny vocabulary, and it hit the multimodal
/// placeholders.** The driver builds a `MuseGlimmerForConditionalGeneration` with
/// `image_token_id = 59` and `video_token_id = 58` "shrunk into the tiny vocab", and salt 1's
/// 12-token prompt contains both. The wrapper substitutes at those positions before the text model
/// sees them, so `t0.embed_norm.out` rows 2 and 6 hold the embedding of id **0**, not of 58 and 59.
/// Feeding rivoli the raw `prompt.ids` compares a text-only decode against a run that was not
/// text-only: measured, it disagrees by **8.7e-1** at step 0, and every step after.
///
/// **Recovered rather than hardcoded**, and that is the difference between adapting to the
/// reference and papering over a divergence. This reads one capture and the embedding table and
/// asks which row the reference used; it does not know 58 or 59, so a future golden with different
/// placeholders, or none, needs no change here. Everything downstream — eight layers, the norms,
/// the attention, the logits — is still scored against the reference with nothing recovered.
///
/// The uniqueness assertion is what keeps it from being a guess: the match must be tight AND the
/// runner-up far, or the recovery is reporting a coincidence.
fn effective_prompt(g: &fixture::Golden, w: &rivoli::golden::GoldenSet) -> Vec<u32> {
    let (shape, cap) = float(&g.g, "t0.embed_norm.out");
    let hid = g.n("hidden_size");
    let rows = shape.iter().product::<usize>() / hid;
    let (eshape, emb) = w
        .floats
        .iter()
        .find(|(n, _, _)| n == "model.language_model.embed_tokens.weight")
        .map(|(_, s, v)| (s, v))
        .expect("the weight set holds the embedding");
    assert_eq!(eshape, &vec![g.n("vocab_size"), hid], "embedding shape");
    let eps = g.c["rms_norm_eps"].as_f64().expect("rms_norm_eps") as f32;
    // The same weightless norm §5 puts on the embedding, applied to every candidate row once.
    let mut normed = emb.clone();
    fixture::weightless(&mut normed, g.n("vocab_size"), hid, eps, 1.0);
    let normed: Vec<&[f32]> = normed.chunks_exact(hid).collect();
    let raw = fixture::ints_of(g, "prompt.ids");
    let mut out = Vec::with_capacity(rows);
    let mut substituted = 0;
    for r in 0..rows {
        let want = &cap[r * hid..(r + 1) * hid];
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
        let mut best = (f32::INFINITY, 0usize);
        let mut second = f32::INFINITY;
        for (id, cand) in normed.iter().enumerate() {
            let d = cand
                .iter()
                .zip(want)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
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
            "{}: position {r} does not resolve to one embedding row — best id {} at {:.2e}, \
             runner-up at {second:.2e}. The recovery below is only sound if the reference's \
             embedding is what this capture holds",
            g.name,
            best.1,
            best.0
        );
        if best.1 as i64 != raw[r] {
            substituted += 1;
        }
        out.push(best.1 as u32);
    }
    // A census, so the mechanism cannot quietly stop applying: salt 1 substitutes two positions
    // and salt 2 one. Zero would mean the placeholders left the prompt — fine for the comparison,
    // but it would also mean this function had stopped being exercised, and a helper nothing
    // exercises is where the next wrong assumption hides.
    println!(
        "  {}: {substituted} of {rows} prompt positions were multimodal placeholders the text \
         stack never saw",
        g.name
    );
    assert!(
        out.iter()
            .zip(raw)
            .filter(|(a, b)| **a as i64 != **b)
            .all(|(a, _)| *a == 0),
        "{}: a substituted position resolved to something other than id 0, which is not the \
         mechanism this function documents",
        g.name
    );
    out
}

/// **Every layer's branch against the reference's own capture — 16 cells, and NOT the eps gate it
/// was built to be.**
///
/// The plan predicted this tensor would close the eps assignment: a post-norm consumes a branch at
/// `mean(x²)~1e-3`, where 1e-5 against 1e-8 is not negligible, and `glimmer_norm.rs` separates them
/// 41.8-56.6x there. Reading the right tensor was indeed the missing half. **The other half is
/// that this comparison cannot carry it**, measured 2026-08-13 across both salts and all eight
/// layers:
///
/// | | range |
/// |---|---|
/// | the eps transposition's own size, f64 exact weights | 1.6e-3 – 1.3e-2 |
/// | this comparison's bf16 weight floor | 4.7e-3 – 3.0e-2 |
///
/// **The signal is 0.2x to 0.6x the floor at every single cell.** The obstacle is not the
/// tolerance and never was — it is that rivoli stores bf16 and the reference computed f32, and the
/// rounding lands on the same tensor at the same magnitude. `tests/glimmer_chain.rs` is where the
/// eps gate went, because its two sides share the artifact's bf16 weights exactly and its floor is
/// reduction noise.
///
/// What this test IS: per-layer, per-salt scoring of the engine against Muse Glimmer's own
/// intermediates, which is 16 comparison points where the logit test has 14 and every one of them
/// upstream of the residual add that washes defects out.
///
/// **Truncating the model is what selects a layer.** The engine keeps one branch buffer and every
/// layer overwrites it, so a run whose config ends at layer `l` leaves `l`'s post-FFN branch in it.
/// Layers 0..l compute identically either way — the truncation removes work after the read, not
/// before it — and the pin simply builds fewer layers from the same artifact.
#[test]
fn every_layers_branch_matches_the_reference() {
    let mut cells = 0;
    let mut worst = (0.0f32, 0usize, "");
    for (g, w) in goldens().iter().zip(weight_sets()) {
        let root = TempRoot::new(&format!("glimmer-branch-{}", g.name));
        let cfg = write_checkpoint(&root, g, &w.1);
        let prompt = effective_prompt(g, &w.1);
        let hid = g.n("hidden_size");
        for l in 0..g.n("num_hidden_layers") {
            let mut gt = cfg.text.clone();
            gt.n_layers = l + 1;
            gt.layer_types.truncate(l + 1);
            gt.layer_rope_theta.truncate(l + 1);
            let got = fixture::decode_one(&root.join("out"), &gt, &prompt)
                .branch()
                .unwrap();
            // The capture is `[1, rows, hidden]` over the whole prompt; the engine decodes
            // token-major, so its buffer holds the LAST row.
            let (shape, want) = float(&g.g, &format!("t0.L{l}.post_feedforward_layernorm.out"));
            assert_eq!(shape, &[1, prompt.len(), hid], "capture shape at L{l}");
            let last = &want[(prompt.len() - 1) * hid..];
            assert!(
                last.iter().any(|v| *v != 0.0),
                "{}: L{l}'s captured branch is all zero, so the score below is against nothing",
                g.name
            );
            let d = worst_rel(&got, last);
            assert!(
                d < TOL_BRANCH,
                "{}: layer {l}'s post-FFN branch disagrees with the reference by {d:.3e}. This \
                 tensor is where the two epsilons are 41.8-56.6x apart, so a transposition lands \
                 HERE and nowhere downstream",
                g.name
            );
            if d > worst.0 {
                worst = (d, l, g.name);
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
