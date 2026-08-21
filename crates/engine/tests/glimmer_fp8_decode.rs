//! **The fp8 Glimmer pin at fixture scale: the budget moves bytes and never text, and the
//! fp8 path actually computes in fp8.** M11's device gate below the real artifact.
//!
//! # What this gates, and what it deliberately does not
//!
//! The bf16 anchor gate (`glimmer_anchor_decode.rs`) scores the engine against the
//! reference's own logits; **no fp8 goldens exist and none are wanted** — quantizing the
//! projections CHANGES the model, and how much it changes is a QUALITY question answered by
//! the paired dNLL on the real artifact (the reference's own S4 ladder measured it at
//! −0.00026 mean, CI [−0.00701, +0.00649]), never by a fixture tolerance. What is decidable
//! at fixture scale is structure, and both structural claims run here:
//!
//! 1. **P4** — the same fp8 artifact through two budgets, one all-resident and one that
//!    forces the streaming slot, must produce bit-identical ids and logits. This is the
//!    fixture-scale half of M11's partition-bit-identity gate; the real-artifact run repeats
//!    it at 30 GB. It is also what proves the fp8 SLOT geometry (twenty tensors per layer,
//!    scale grids interleaved) refills correctly — the bf16 anchor's P4 arm can never enter
//!    that path.
//! 2. **Anti-fallback** — the fp8 arm's logits must DIFFER from the bf16 arm's on the same
//!    prompt. The named failure mode for M11 is an fp8 dispatch that silently falls back to
//!    bf16: such a run "works", shows ~1.0× speedup, and passes every P4 check. e4m3 keeps 3
//!    mantissa bits against bf16's 8, so on real reference weights the two logit vectors
//!    cannot coincide unless some path is reading the wrong bytes — equality IS the red flag.
//!
//! The prompt is a fixed in-vocab literal rather than the anchor's captured one:
//! `effective_prompt`'s off-vocab resolution exists to reproduce the reference RUN, and this
//! gate makes no reference-agreement claim.
//!
//! # What a green here does NOT establish
//!
//! **Finite-but-wrong fp8 arithmetic survives both assertions.** P4 compares the fp8 arm with
//! ITSELF at two budgets, and the anti-fallback assert only demands the fp8 arm differ from
//! bf16 — so a defect that is deterministic and format-specific passes both. The concrete one
//! to fear is a scale grid swapped between two projections of IDENTICAL shape (`q_proj` and
//! `self_attn.gate_proj`; `k_proj` and `v_proj`; `mlp.gate_proj` and `mlp.up_proj`): the
//! logits stay finite, stay budget-invariant, and stay unequal to bf16. `place_proj`'s shape
//! check cannot separate those pairs either — its own doc says so — and the only structural
//! defence is that each name is written once, at the placer.
//!
//! That is why **M11's paired dNLL on the real artifact is load-bearing for CORRECTNESS and
//! not only for quality.** It is the gate that would catch this class: a swapped scale grid at
//! real dims is a large, systematic quality loss, far outside the pre-registered equivalence
//! band, where at fixture scale it is just "different from bf16, as expected". Nobody should
//! read a green on this file as licence to skip it.
//!
//! **The fixture `manifest.json` this file writes carries the `format` section and nothing
//! else** — no `text_config`, no `architectures` — which is a manifest shape no shipped
//! converter produces (`--fp8` writes source-config-plus-format through
//! `FormatMeta::manifest_from_config`). `ProjFmt::sniff` reads only `format`, so it is
//! sufficient here; `GlimmerConfig::load` is NOT, and the mechanism is worth naming because
//! it is the opposite of the obvious guess: `schema::config_path` prefers `manifest.json`
//! over `config.json` whenever the former exists, so `load` reads THIS file — not the
//! `config.json` sitting beside it — finds no `model_type`, and refuses. Noted so nobody
//! aims other tooling at these dirs and reads the refusal as a defect. The same shape
//! applies to `geometry.rs`'s sniff fixtures.
//!
//! **Nothing here reaches the split-K kernel.** `linalg.hip` routes `rivoli_gemv_fp8` to its
//! split-K sibling at `i_dim >= 4096`; the anchor's `hidden` is 72 and `inter` 216, so every
//! projection in this fixture takes the wave-per-row path. On the real checkpoint (hidden
//! 6656, `o_proj` `i_dim` 4096, `mlp.down_proj` `i_dim` 19968) **all eight take split-K**.
//! That arithmetic is scored by `kernel.rs::gemv_fp8_matches_oracle`, which runs the long
//! shapes against the host oracle — this is a division of labour, not a hole, but a green
//! here says nothing about the kernel the shipped model actually launches.
//!
//! # Running it
//!
//! Device test: under the GPU flock, `-- --test-threads=1`, dev profile.
//!
//! ```text
//! flock /var/run/sys-gpu.lock -c 'cargo test -p rivoli-engine --features rocm \
//!     --test glimmer_fp8_decode -- --test-threads=1 --nocapture'
//! ```

// Featureless builds have no engine to run; the deviceless fp8 claims (layer_bytes, sniff)
// live in `glimmer/geometry.rs`'s own test module.
#![cfg(feature = "rocm")]
// The panic-on-failure idiom; the fixture module carries the same allow and the same reason.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod glimmer_anchor;

use glimmer_anchor::{Artifact, anchors, assert_split_entered, budgets, write_artifact};
use rivoli_artifact::format::{Dtype, FormatMeta, SafeWriter, Safetensors};
use rivoli_artifact::glimmer::GLIMMER_LAYER_PREFIX;
use rivoli_artifact::glimmer_config::GlimmerTextConfig;
use rivoli_artifact::quant::FP8_BLOCK;
// Braced rather than the sibling gate's flat list — and this comment sits on the braced line
// it describes, which it did not until review 2026-08-16: jscpd normalizes identifiers, so
// two flat four-line import tails are one shape to it regardless of what they import, and a
// justification attached to the wrong item is how the shape gets "tidied" back into a clone.
use rivoli_engine::glimmer::{engine::GlimmerEngine, geometry::ProjFmt};
use rivoli_engine::seam::GenSpec;

/// Quantize a bf16 anchor artifact into an fp8 one, THROUGH THE CONVERTER'S OWN PRIMITIVE.
///
/// `SafeWriter::add_quantized_fp8` is exactly what `convert_glimmer --fp8` calls per
/// projection, so the VALUES this produces are the shipped conversion at fixture scale — not
/// a test-local re-spelling of it. **WHICH tensor takes which path is decided differently
/// here, deliberately**: by rank plus prefix, where the converter uses its `is_norm` /
/// `is_layer_proj` name predicates. `glimmer_anchor::stored` carries the argument — the rank
/// is the discriminator the ENGINE enforces, and a second copy of `is_norm` is how the two
/// spellings drift. Norms are already f32 in the source (rank-1) and copy through;
/// `embed`/`lm_head` stay bf16, as the real `--fp8` pass keeps them.
///
/// The `manifest.json` stamp is written because `ProjFmt::sniff` refuses an fp8 artifact
/// without one — the block is load-bearing and the pin dequantizes at the stamped value.
///
/// **The grid census is asserted, not assumed.** `gemv_fp8` addresses a scale two ways —
/// `scale + (o >> blk_shift(block)) * sc_cols` for the ROW tile, `sc_cols > 1` for the COLUMN
/// tiles inside `Fp8Row` — and both are dead code on a weight whose grid is `[1, 1]`. Review
/// asserted this gate could only reach `[1, 1]`; it is wrong, and the reason is worth pinning
/// rather than arguing: the anchor's `intermediate_size` is **216**, above the 128 block, so
/// `mlp.{gate,up}_proj` are `[216, 72]` → grid `[2, 1]` (row tiling live) and `mlp.down_proj`
/// is `[72, 216]` → grid `[1, 2]` (`sc_cols == 2`, column tiling live). Both counts are
/// asserted non-zero below, so this stays an observation of the fixture actually used rather
/// than arithmetic done in a comment — and if a future anchor shrinks `inter` under the block,
/// the claim reddens here instead of the coverage silently going to zero.
fn quantize_artifact(bf16: &Artifact, tag: &str) -> Artifact {
    let root =
        std::env::temp_dir().join(format!("rivoli-glimmer-fp8-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create the fp8 artifact directory");
    let src = Safetensors::open_file(&format!("{}/resident.safetensors", bf16.path()))
        .expect("open the bf16 anchor artifact");
    let mut names: Vec<String> = src.names().iter().map(|s| s.to_string()).collect();
    names.sort();
    let mut w = SafeWriter::new();
    let (mut row_tiled, mut col_tiled) = (0usize, 0usize);
    for name in &names {
        let (_, _, shape) = src.raw(name).expect("tensor header");
        if shape.len() == 1 {
            w.copy_verbatim(&src, name, Dtype::F32).expect("norm");
        } else if name.starts_with(GLIMMER_LAYER_PREFIX) {
            let base = name.strip_suffix(".weight").expect("projection name tail");
            w.add_quantized_fp8(&src, base, FP8_BLOCK)
                .expect("quantize");
            row_tiled += usize::from(shape[0].div_ceil(FP8_BLOCK) > 1);
            col_tiled += usize::from(shape[1].div_ceil(FP8_BLOCK) > 1);
        } else {
            w.copy_verbatim(&src, name, Dtype::Bf16)
                .expect("embed/head");
        }
    }
    println!("  fp8 grids: {row_tiled} projections row-tiled, {col_tiled} column-tiled");
    assert!(
        row_tiled > 0 && col_tiled > 0,
        "every fp8 grid in this fixture is [1, 1] ({row_tiled} row-tiled, {col_tiled} \
         column-tiled): `gemv_fp8`'s scale addressing (`o >> blk_shift`, `sc_cols`) is then \
         dead code under this gate and its green says nothing about multi-tile weights"
    );
    w.write(&format!("{}/resident.safetensors", root.display()))
        .expect("write the fp8 artifact");
    std::fs::copy(
        format!("{}/config.json", bf16.path()),
        root.join("config.json"),
    )
    .expect("copy the config");
    let stamp = serde_json::json!({
        "format": serde_json::to_value(FormatMeta::current(FP8_BLOCK)).expect("format meta")
    });
    std::fs::write(root.join("manifest.json"), stamp.to_string()).expect("write the stamp");
    Artifact(root)
}

/// One short generation's observables.
struct Run {
    ids: Vec<u32>,
    /// The last sample's logits AS BITS — this file's claims are bit-identity and
    /// bit-difference, and comparing f32s would re-ask the NaN question.
    logits: Vec<u32>,
    residency: (usize, usize),
    stats: (u64, u64),
}

/// The generation request this file holds FIXED while the artifact and the budget vary.
///
/// Every claim below is "same prompt, different bytes or budget", so the config, prompt and
/// length live here ONCE — they cannot drift between arms — and [`Request::decode`]'s two
/// parameters are exactly the two axes the assertions compare across. `eos` is empty on
/// purpose: every arm must generate all `ngen` tokens, or the bit-identity claims would be
/// comparing runs of different lengths.
struct Request<'a> {
    t: &'a GlimmerTextConfig,
    prompt: &'a [u32],
    ngen: usize,
}

impl Request<'_> {
    fn decode(&self, dir: &str, capacity: usize) -> Run {
        let mut e = GlimmerEngine::open(dir, self.t, capacity, self.prompt.len() + self.ngen)
            .expect("build the engine");
        let out = e
            .decode(
                GenSpec {
                    prompt: self.prompt,
                    ngen: self.ngen,
                    eos: &[],
                },
                &mut |_| true,
            )
            .expect("decode");
        let logits = e
            .logits()
            .expect("the last sample's logits")
            .iter()
            .map(|v| v.to_bits())
            .collect();
        Run {
            ids: out.ids,
            logits,
            residency: e.residency(),
            stats: e.slot_stats(),
        }
    }
}

#[test]
fn the_fp8_partition_moves_bytes_and_never_text_and_is_not_bf16() {
    let a = &anchors()[0]; // one salt: the properties here are residency and format, not weights
    let cfg = a.config();
    let t = &cfg.text;
    let bf16 = write_artifact(a);
    let fp8 = quantize_artifact(&bf16, a.name);
    let prompt = [1u32, 2, 3, 5, 8];
    assert!(
        prompt.iter().all(|&i| (i as usize) < t.vocab),
        "a prompt id is outside the anchor vocab"
    );
    let ngen = 4;
    let req = Request {
        t,
        prompt: &prompt,
        ngen,
    };

    // P4: all-resident vs forced-streaming, bit for bit.
    let (roomy, tight) = budgets(t, prompt.len() + ngen, ProjFmt::Fp8 { block: FP8_BLOCK });
    let all = req.decode(fp8.path(), roomy);
    let (pinned, streamed) = all.residency;
    assert_eq!(
        streamed, 0,
        "the roomy fp8 budget streamed {streamed} layers, so the two arms below are the same arm"
    );
    assert_eq!(
        pinned, t.n_layers,
        "the roomy fp8 budget must pin every layer"
    );
    let split = req.decode(fp8.path(), tight);
    assert_split_entered(a.name, split.residency, split.stats);
    assert_eq!(
        split.ids, all.ids,
        "the same prompt through the same fp8 artifact gave different ids at a different \
         budget. Residency may move bytes; it may never move text (P4)"
    );
    assert_eq!(
        split.logits, all.logits,
        "fp8 logits differ across budgets — the twenty-tensor slot refill is not \
         byte-faithful"
    );

    // **FINITENESS FIRST, and the order is the point.** Everything below counts DIFFERING
    // bit patterns, and NaN differs from everything — including from itself, which is why an
    // all-NaN fp8 arm would satisfy `differing > 0` here AND stay bit-identical across the two
    // budgets above (the bits are stable even though the values are not). That is this repo's
    // recorded false-green trap in its mirror form: a divergence proof believed against a
    // reference that computes nothing. A scale grid read as weights or a mis-tiled `sc_cols`
    // reaches it, so the guard is not hypothetical.
    assert!(
        all.logits.iter().all(|b| f32::from_bits(*b).is_finite()),
        "the fp8 arm produced non-finite logits — every claim below counts differing bit \
         patterns, and a divergence from NaN is not evidence that fp8 arithmetic ran"
    );

    // Anti-fallback: fp8 must NOT reproduce bf16. Equality here is the named M11 failure
    // mode (a dispatch that silently launched the bf16 kernel) wearing a green test.
    let (roomy_bf16, _) = budgets(t, prompt.len() + ngen, ProjFmt::Bf16);
    let base = req.decode(bf16.path(), roomy_bf16);
    assert_eq!(base.logits.len(), all.logits.len());
    let differing = base
        .logits
        .iter()
        .zip(&all.logits)
        .filter(|(b, f)| b != f)
        .count();
    println!(
        "  fp8-vs-bf16: {differing} of {} logit words differ",
        all.logits.len()
    );
    assert!(
        differing > 0,
        "the fp8 arm's logits are bit-identical to the bf16 arm's — e4m3 cannot reproduce \
         bf16 weights, so some path is reading the wrong bytes (the named silent-fallback \
         failure mode)"
    );
}
