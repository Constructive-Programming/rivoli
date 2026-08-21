//! `convert_glimmer` end to end, on a synthetic checkpoint — M7's converter gate.
//!
//! **Why synthetic rather than a slice of the real one.** The Muse Glimmer checkpoint is
//! 59.553 GB and is not on this machine. This converter's unit of work is the whole tensor set
//! — which tensors are copied, which are widened, which are skipped, and whether the artifact
//! re-opens as the same model — and none of that is testable on a slice. A four-layer model
//! exercises every branch.
//!
//! **The fixture is built FROM `GlimmerTextConfig::layer_tensor_shape`**, not from a second
//! transcription of the shapes. That is deliberate: it makes the converter's completeness walk,
//! the fixture, and the config one statement rather than three, so a shape wrong in the schema
//! reddens here instead of being agreed with. What it does NOT close is a name or shape wrong
//! in *both* — the old tree's `tests/glimmer_names.rs` closes that against the shipped
//! `model.safetensors.index.json`, and that gate arrives with the real-checkpoint work.
//!
//! **TWO shards, and the vision half lives alone in the second.** The reference's fixture is
//! single-shard and its own comment records that this made one property untestable: the skipped
//! count is read from the INDEX rather than from the opened shards, because `open_indexed`
//! selects whole shards and a vision-only shard is never opened at all. Here that shard exists,
//! so `3 vision tensors skipped` is a claim the fixture can falsify.
//!
//! No GPU, no network — every byte is written by this file.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

// Module alias rather than a second flat `use` list. `glm_convert.rs` opens with the same names
// from the same modules followed by the same `mod common;` pair, and jscpd — which normalizes
// identifiers — reported the two import blocks as a 36-token clone. The reference's own
// `glimmer_convert.rs` records reaching for this exact fix for this exact reason; aliasing is the
// smaller change and reads fine in a test.
use rivoli_artifact::format as fmt;
use rivoli_artifact::glimmer::{GLIMMER_LAYER_PREFIX, GLIMMER_LAYER_TENSORS};
use rivoli_artifact::glimmer_config::GlimmerConfig;
use rivoli_artifact::schema::parse_config;
use serde_json::{Value, json};
use std::path::Path;
use std::process::Command;

// Qualified `common::` calls rather than a `use common::{...}` list, for the same tokenizer
// reason as the alias above: with the list, the trailing four lines of this preamble were
// themselves the clone. Naming the module at each call site also says where a fixture helper
// comes from, which is the half a bare `weights(...)` in a converter gate does not.
mod common;

/// Tiny but structurally faithful: every distinction the real config makes survives the shrink.
///
/// `HEAD_DIM * HEADS` = 32 and `HIDDEN` = 64 are **deliberately unequal**, which is the whole
/// trap this model carries (real: 32x128 = 4096 against `hidden_size` 6656). A fixture with
/// `head_dim == hidden / n_heads` would let a converter that derived the head width pass.
/// `INTER` differs from both, and `KV` divides `HEADS` without equalling it.
const LAYERS: usize = 4;
const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV: usize = 2;
const HEAD_DIM: usize = 8;
const INTER: usize = 96;
const VOCAB: usize = 64;

/// The three model-level tensors the converter checks by name before it writes.
const GLOBALS: [&str; 3] = [
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
];

/// The vision prefixes `convert_glimmer::is_vision` matches, one tensor each. Restated here
/// rather than imported because the converter's predicate is private to its binary — and the
/// restatement is the test: if the two lists ever disagree, the counts below move.
const VISION: [&str; 3] = [
    "model.vision_tower.blocks.0.attn.qkv.weight",
    "model.vision_adapter.proj.weight",
    "model.vision_projection.weight",
];

const TEXT_SHARD: &str = "model-00001-of-00002.safetensors";
const VISION_SHARD: &str = "model-00002-of-00002.safetensors";
const GEN: &str = "generation_config.json";

/// One tensor of the fixture. `common::Tensor` carries a dtype, which this model does not need
/// — every tensor here is bf16 — but the type moved to `common` on 2026-08-16 when
/// `v4_convert.rs` became the third converter gate and jscpd reported the shard writer as a
/// clone. The dtype is spelled once, here.
fn tensor(name: &str, shape: Vec<usize>) -> common::Tensor {
    let n: usize = shape.iter().product();
    (
        name.to_string(),
        fmt::Dtype::Bf16,
        shape,
        common::bf16_bytes(&common::weights(name, n)),
    )
}

/// The text half, every shape taken from the config's own table.
///
/// `layer_tensor_shape` is the single statement of what each of the twelve is; asking it here
/// means the fixture cannot disagree with the schema the converter validates against.
fn text_tensors(cfg: &GlimmerConfig) -> Vec<common::Tensor> {
    let t = &cfg.text;
    let mut out: Vec<common::Tensor> = GLOBALS
        .iter()
        .map(|n| {
            let shape = if n.ends_with("norm.weight") {
                vec![HIDDEN]
            } else {
                vec![VOCAB, HIDDEN]
            };
            tensor(n, shape)
        })
        .collect();
    for l in 0..t.n_layers {
        for name in GLIMMER_LAYER_TENSORS {
            let shape = t.layer_tensor_shape(name).unwrap();
            out.push(tensor(
                &format!("{GLIMMER_LAYER_PREFIX}.{l}.{name}.weight"),
                shape,
            ));
        }
    }
    out
}

/// `model.safetensors.index.json`: `text` in the first shard, [`VISION`] alone in the second.
///
/// **Written from the tensor list rather than alongside it**, and re-written whenever that list
/// changes, because the index is what `open_indexed` selects shards by — a refusal test that
/// dropped a tensor from the shard and left it in the index would be testing a truncated-file
/// error instead of the completeness walk. The vision half is the constant either way: it is
/// never opened, which is the whole point of counting it here.
fn write_index(src: &Path, text: &[common::Tensor]) {
    let mut entries: Vec<(String, &str)> = text
        .iter()
        .map(|(n, _, _, _)| (n.clone(), TEXT_SHARD))
        .collect();
    entries.extend(VISION.map(|n| (n.to_string(), VISION_SHARD)));
    common::write_weight_map(src, &entries);
}

/// The HF `config.json` the converter reads — the wrapper, its `text_config`, and the sibling
/// `vision_config` the real file carries (present so the descent check has something to descend
/// past, and so `model_type: "muse_glimmer_vision"` exists in the fixture as it does upstream).
fn glimmer_config_json() -> Value {
    // The [s,s,s,full] period at four layers, with the pairing invariant `validate` enforces:
    // a layer is rotated IFF it slides, and every rotated layer shares the one global base.
    let theta = 500_000.0;
    let types: Vec<&str> = (0..LAYERS)
        .map(|i| {
            if (LAYERS - 1 - i).is_multiple_of(4) {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    let thetas: Vec<f64> = types
        .iter()
        .map(|t| {
            if *t == "sliding_attention" {
                theta
            } else {
                0.0
            }
        })
        .collect();
    json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "dtype": "bfloat16",
        "text_config": {
            "model_type": "muse_glimmer_text",
            "num_hidden_layers": LAYERS,
            "hidden_size": HIDDEN,
            "vocab_size": VOCAB,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV,
            "head_dim": HEAD_DIM,
            "intermediate_size": INTER,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.196_116_135_138_184_04,
            "final_logit_softcapping": 20.0,
            "sliding_window": 16,
            "layer_types": types,
            "layer_rope_theta": thetas,
            "rope_parameters": { "rope_theta": theta, "rope_type": "default" },
            "max_position_embeddings": 128,
            "tie_word_embeddings": false,
            "hidden_activation": "silu",
            "attention_bias": false,
        },
        "vision_config": { "model_type": "muse_glimmer_vision", "hidden_size": 32 },
    })
}

/// `generation_config.json` with `ids`. Written by every arm, because the EOS refusals below
/// each need a DIFFERENT content and restoring the good one between them is what keeps each
/// assertion about its own mutation.
fn write_eos(dir: &Path, ids: &[u32]) {
    std::fs::write(dir.join(GEN), json!({ "eos_token_id": ids }).to_string()).unwrap();
}

/// The whole synthetic checkpoint. Returns the text tensors, so the round-trip test can compare
/// the artifact against the bytes that went in.
fn write_fixture(src: &Path) -> Vec<common::Tensor> {
    let config = glimmer_config_json();
    common::write_config(src, &config);
    // Parsed back rather than built from the constants: the fixture's shapes then come from the
    // same `validate`d config the converter will read, and a config this test writes that the
    // schema would refuse fails HERE rather than as a confusing converter error.
    let cfg: GlimmerConfig = parse_config(&config.to_string()).expect("the fixture config parses");
    let text = text_tensors(&cfg);
    let vision: Vec<common::Tensor> = VISION.iter().map(|n| tensor(n, vec![8, HIDDEN])).collect();
    common::write_shard(&src.join(TEXT_SHARD), &text);
    common::write_shard(&src.join(VISION_SHARD), &vision);

    write_index(src, &text);

    // The four AUX files. Stub contents except `generation_config.json`, whose ids are the one
    // thing the converter reads rather than copies — and both are inside VOCAB, since an id past
    // it is a stop token no argmax can return and is its own refusal below.
    common::write_aux(
        src,
        &[
            ("tokenizer.json", "{\"model\":{\"type\":\"BPE\"}}"),
            ("tokenizer_config.json", "{}"),
            ("chat_template.jinja", "{{ messages }}"),
        ],
    );
    write_eos(src, &[(VOCAB - 3) as u32, (VOCAB - 1) as u32]);
    text
}

fn run_with(src: &Path, out: &Path, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_convert_glimmer"))
        .args([src.to_str().unwrap(), out.to_str().unwrap()])
        .args(extra)
        .output()
        .expect("run convert_glimmer")
}

fn run(src: &Path, out: &Path) -> std::process::Output {
    run_with(src, out, &[])
}

/// Assert every formatted count fragment reached the log. The counts are the OBSERVATION that
/// each tensor class took its intended path; both converter arms make the same kind of claim,
/// so the assertion is spelled once.
///
/// **Each fragment carries its leading `", "`**, because `contains` on a bare count is
/// satisfied by any number ending in it — `"32 projections quantized"` matches a log reading
/// `132 projections quantized` (review, 2026-08-16). The converter emits every count after a
/// comma-space, so anchoring there costs two characters and removes the whole class.
fn expect_counts(log: &str, wants: &[&str]) {
    // A loop over a caller-supplied list passes on an empty one, which is the "examined count
    // can silently reach zero" shape this repo does not accept as a check — unreachable from
    // today's two literal call sites, asserted anyway because that is what makes it unreachable
    // from tomorrow's.
    assert!(!wants.is_empty(), "expect_counts examined nothing");
    for w in wants {
        assert!(log.contains(w), "missing `{w}` in:\n{log}");
    }
}

/// `run`, expecting a refusal whose message names `want`.
///
/// The `want` check is the point: a refusal test that only asserts non-zero exit passes when the
/// binary fails for an unrelated reason, which is how a guard gets deleted without a red test.
fn refuses(src: &Path, out: &Path, want: &str) {
    common::expect_refusal(&run(src, out), want);
}

#[test]
fn convert_glimmer_writes_a_bf16_artifact_that_reopens_as_the_same_model() {
    let (root, src, out) = common::scratch_src_out("glimmer-convert-rt");
    let tensors = write_fixture(&src);

    let log = common::expect_success(&run(&src, &out), "convert_glimmer");

    // The vision count is the OBSERVATION that that half was excluded, rather than the
    // assumption — and the 3 comes from the index, since the shard holding them was never
    // opened. Four norms per layer plus the model-level one is what gets widened. Both EOS
    // ids printed, so an operator can notice a wrong set before a decode runs to its limit.
    expect_counts(
        &log,
        &[
            ", 3 vision tensors skipped",
            // The bf16 arm's own NEGATIVE: nothing was quantized. The binary's `ensure!` is
            // what enforces it, but without this line nothing in this test would notice
            // `--fp8` becoming the default or the flag ceasing to gate the branch.
            ", 0 projections quantized to fp8",
            &format!(", {} norms widened", LAYERS * 4 + 1),
            &format!("eos_token_id [{}, {}]", VOCAB - 3, VOCAB - 1),
        ],
    );

    // It re-opens as the same model: the manifest still carries the wrapper and its text_config,
    // so the architecture resolves and every validate check runs again.
    let cfg = GlimmerConfig::load(out.to_str().unwrap()).unwrap();
    assert_eq!(cfg.text.n_layers, LAYERS);
    assert_eq!(cfg.text.layer_types.len(), LAYERS);
    fmt::FormatMeta::load(out.to_str().unwrap()).unwrap();

    let art =
        fmt::Safetensors::open_file(out.join("resident.safetensors").to_str().unwrap()).unwrap();
    for name in VISION {
        assert!(
            art.raw(name).is_err(),
            "{name} is vision and must not be in the artifact"
        );
    }
    for t @ (name, _, _, _) in &tensors {
        // Widened, and widened CORRECTLY — not merely present at the right length. A
        // byte-length check alone passes on a zeroed tensor; `common::assert_widened` compares
        // values, and both helpers moved there when `v4_convert.rs` needed the same pair.
        if name.ends_with("norm.weight") {
            common::assert_widened(&art, t);
        } else {
            common::assert_verbatim(&art, t);
        }
    }
    // Every aux file reached the artifact. `finish_artifact` refuses a failed copy, so this is
    // asserting the LIST rather than the mechanism — a file dropped from AUX would leave the
    // engine reading an artifact that is missing its template or its stop tokens.
    for aux in [
        "tokenizer.json",
        "tokenizer_config.json",
        GEN,
        "chat_template.jinja",
    ] {
        assert!(out.join(aux).exists(), "{aux} missing from the artifact");
    }
    common::clean(&root);
}

/// `--fp8` quantizes the layer projections and NOTHING else, and the bytes it writes are
/// [`rivoli_artifact::quant::quantize_fp8_block`]'s own.
///
/// **The value check is byte-equality against the library, not a tolerance.** The converter's
/// unit of work is selection and plumbing — WHICH tensors take which path — while the scale
/// choice itself is `quantize_fp8_block`'s and is unit-tested beside it (round-trip within
/// e4m3's half-ULP, the all-zero-tile scale, the non-finite refusal). Re-running the quantizer
/// here over the values the converter reads (the fixture's f32s through the bf16 round-trip,
/// since the checkpoint stores bf16) and demanding identical bytes pins the whole path with no
/// tolerance to hide a wrong block, a transposed shape or a skipped tile behind.
///
/// The fixture's dims are all at or under the 128 block, so every grid here is `[1, 1]` —
/// except `INTER` = 96, deliberately: at any block **strictly below 96** the `[96, 64]` MLP
/// projections grow a second grid row and the shape assertion below reddens. That is what
/// discriminates the shipped 128 from the plausible wrong constants below it — 64 gives a
/// `[2, 1]` grid, 32 a `[3, 2]` (`HIDDEN` = 64 tiles on the second axis too at 32, which the
/// first draft of this sentence got wrong) — and the discrimination is over exactly those: at 96 itself
/// `div_ceil(96, 96) == 1` and this fixture would NOT redden, which no candidate constant
/// makes matter but which the sentence has to say to be true. Multi-tile value behaviour is
/// the quantizer's own test's job (it runs 5×7 at block 2); the real-checkpoint byte-parity
/// gate covers the block at real dims.
#[test]
fn convert_glimmer_fp8_quantizes_the_projections_and_nothing_else() {
    let (root, src, out) = common::scratch_src_out("glimmer-convert-fp8");
    let tensors = write_fixture(&src);

    let log = common::expect_success(&run_with(&src, &out, &["--fp8"]), "convert_glimmer --fp8");
    // 8 projections per layer quantized; the norms widen exactly as in the bf16 arm.
    expect_counts(
        &log,
        &[
            &format!(", {} projections quantized to fp8", LAYERS * 8),
            &format!(", {} norms widened", LAYERS * 4 + 1),
        ],
    );

    let block = rivoli_artifact::quant::FP8_BLOCK;
    // **The stamp the pin will dequantize at, read back off this run's own artifact.** The
    // bf16 arm calls `FormatMeta::load` too, but only to prove the manifest parses; here the
    // VALUE is load-bearing — `ProjFmt::sniff` returns exactly this number and every scale
    // lookup in the engine tiles by it, so a converter that quantized at one block and stamped
    // another would produce an artifact that mis-tiles silently at every projection whose dims
    // happen to give the same grid shape. Added 2026-08-16: the fp8 arm had dropped the load
    // its bf16 sibling makes, leaving the one seam `meta.rs`'s doc claims unchecked.
    assert_eq!(
        fmt::FormatMeta::load(out.to_str().unwrap())
            .unwrap()
            .fp8_block,
        block,
        "the artifact's stamped fp8_block is not the block it was quantized at"
    );
    let art =
        fmt::Safetensors::open_file(out.join("resident.safetensors").to_str().unwrap()).unwrap();
    for t @ (name, _, shape, _) in &tensors {
        if name.ends_with("norm.weight") {
            common::assert_widened(&art, t);
        } else if name.starts_with(GLIMMER_LAYER_PREFIX) {
            // The values the converter quantized: the fixture's f32s AFTER the bf16 round-trip,
            // because the shard stores bf16 and `add_quantized_fp8` widens what it reads.
            let n: usize = shape.iter().product();
            let vals: Vec<f32> = common::weights(name, n)
                .iter()
                .map(|&v| rivoli_core::num::bf16_to_f32(rivoli_core::num::f32_to_bf16(v)))
                .collect();
            let (want_p, want_s) =
                rivoli_artifact::quant::quantize_fp8_block(&vals, [shape[0], shape[1]], block)
                    .unwrap();
            let (got_p, pshape) = art.typed(name, fmt::Dtype::F8E4M3).unwrap();
            assert_eq!(pshape, &shape[..], "{name}: packed shape");
            assert_eq!(
                got_p,
                &want_p[..],
                "{name}: packed bytes are not the quantizer's"
            );
            let sname = name.replace(".weight", ".weight_scale_inv");
            let (got_s, sshape) = art.typed(&sname, fmt::Dtype::F32).unwrap();
            let want_grid = [shape[0].div_ceil(block), shape[1].div_ceil(block)];
            assert_eq!(sshape, want_grid, "{sname}: grid shape");
            let want_sb: Vec<u8> = want_s.iter().flat_map(|s| s.to_le_bytes()).collect();
            assert_eq!(
                got_s,
                &want_sb[..],
                "{sname}: scale bytes are not the quantizer's"
            );
        } else {
            // `embed_tokens` and `lm_head`: requantizing two tensors read once per token each
            // is its own quality question — they stay bf16 verbatim.
            common::assert_verbatim(&art, t);
        }
    }
    common::clean(&root);
}

/// The guards that fire before 55 GB is written. Each arm restores the fixture, so every
/// assertion is about its own mutation rather than about the wreckage of the previous one.
#[test]
fn convert_glimmer_refuses_before_it_writes() {
    let (root, src, out) = common::scratch_src_out("glimmer-convert-refuse");
    write_fixture(&src);

    // Writing into the source directory is a SIGBUS risk, not an error — the writer maps the
    // shards while it writes. Refused by path identity, so `src/.` must refuse too.
    refuses(&src, &src.join("."), "SIGBUS");

    // A REQUIRED_AUX file missing refuses EARLY — before the config is even parsed, and long
    // before `finish_artifact` would refuse the same absence at the end of a three-hour run.
    for aux in [GEN, "chat_template.jinja"] {
        let body = std::fs::read(src.join(aux)).unwrap();
        std::fs::remove_file(src.join(aux)).unwrap();
        refuses(&src, &out, &format!("{aux} is missing"));
        std::fs::write(src.join(aux), body).unwrap();
    }

    // A checkpoint missing one per-layer tensor refuses by NAME, before the write. Dropped from
    // the shard AND from the index, since the index is what selects the shards.
    let dropped = format!("{GLIMMER_LAYER_PREFIX}.2.mlp.up_proj.weight");
    let cfg = GlimmerConfig::load(src.to_str().unwrap()).unwrap();
    let kept: Vec<common::Tensor> = text_tensors(&cfg)
        .into_iter()
        .filter(|(n, _, _, _)| *n != dropped)
        .collect();
    common::write_shard(&src.join(TEXT_SHARD), &kept);
    write_index(&src, &kept);
    refuses(&src, &out, &dropped);
    assert!(
        !out.join("resident.safetensors").exists(),
        "the artifact must not exist after a refusal"
    );
    common::clean(&root);
}

/// **Both EOS ids reach the artifact, and a file that exists but says nothing is refused.**
///
/// The engine half is safe by construction: `eos_token_ids` reads both the array and the bare-int
/// spellings and the engine stops on `contains`. What is NOT safe is one step worse than the trap
/// the plan names — `REQUIRED_AUX` checks that `generation_config.json` EXISTS, and `{}` passes
/// that check, copies into the artifact, and yields **zero** stop tokens. So the port does not
/// stop on one of the two, it stops on NONE, announced by one `warn!` at load. That signature is
/// the one behind the old tree's benchmark retraction: 56 runs, not one terminating naturally.
///
/// Four arms, and the three refusals are the red proof for the first: without them "the ids
/// reached the artifact" is satisfied by any converter that copies a file.
#[test]
fn both_eos_ids_reach_the_artifact_and_an_unusable_generation_config_is_refused() {
    let (root, src, out) = common::scratch_src_out("glimmer-convert-eos");
    write_fixture(&src);

    // **A DISTINCT pair, written here rather than left at the fixture's default**, so the
    // assertion proves the artifact TRACKED the source rather than that some constant matched.
    // Compared as BYTES: the copy is `std::fs::copy`, so byte equality is the property, and a
    // parse-and-compare would pass a reordering or an added key — and would be a third parser of
    // this field in the tree.
    let ids = [1u32, (VOCAB - 2) as u32];
    write_eos(&src, &ids);
    let want = std::fs::read(src.join(GEN)).unwrap();
    common::expect_success(&run(&src, &out), "convert_glimmer");
    assert_eq!(
        std::fs::read(out.join(GEN)).expect("generation_config in the artifact"),
        want,
        "the artifact's stop tokens are not the checkpoint's — a decode built on this stops on \
         the wrong set, or on nothing"
    );

    // An id past the vocabulary is refused: it is a stop token no argmax can return, which is
    // the same unstoppable decode as having none.
    write_eos(&src, &[VOCAB as u32]);
    refuses(
        &src,
        &root.join("out-vocab"),
        "past this model's vocabulary",
    );

    // Red proof, and the case that was live: a file that satisfies the presence check and
    // carries no ids. `{}` first, then the two shapes a hand-edit produces.
    for (bytes, what) in [
        (&b"{}"[..], "an empty object"),
        (br#"{"eos_token_id": []}"#, "an empty array"),
        (br#"{"eos_token_id": null}"#, "a null"),
    ] {
        let dst = root.join("out-red");
        std::fs::write(src.join(GEN), bytes).unwrap();
        refuses(&src, &dst, "no usable `eos_token_id`");
        // Refused BEFORE any tensor is read, and before `create_dir_all` — the whole argument
        // for checking here rather than at load is that a three-hour convert must not end in
        // this. The DIRECTORY, not the artifact inside it: the converter creates it at the point
        // the check has already passed, so its absence is the stronger statement and it catches
        // the check being moved one line later. Inside the loop so all three arms are covered
        // rather than only the last (review, 2026-08-13).
        assert!(
            !dst.exists(),
            "{what}: the converter got past the EOS check"
        );
    }
    common::clean(&root);
}
