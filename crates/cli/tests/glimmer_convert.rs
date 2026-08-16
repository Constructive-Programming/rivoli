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
use rivoli_core::num::bf16_to_f32;
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

/// One tensor of the fixture: name, shape, and the bf16 bytes the artifact must reproduce.
type Tensor = (String, Vec<usize>, Vec<u8>);

fn tensor(name: &str, shape: Vec<usize>) -> Tensor {
    let n: usize = shape.iter().product();
    (
        name.to_string(),
        shape,
        common::bf16_bytes(&common::weights(name, n)),
    )
}

/// The text half, every shape taken from the config's own table.
///
/// `layer_tensor_shape` is the single statement of what each of the twelve is; asking it here
/// means the fixture cannot disagree with the schema the converter validates against.
fn text_tensors(cfg: &GlimmerConfig) -> Vec<Tensor> {
    let t = &cfg.text;
    let mut out: Vec<Tensor> = GLOBALS
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

fn write_shard(path: &Path, tensors: &[Tensor]) {
    let mut w = fmt::SafeWriter::new();
    for (name, shape, bytes) in tensors {
        w.add(name.clone(), fmt::Dtype::Bf16, shape.clone(), &bytes[..]);
    }
    w.write(path.to_str().unwrap()).unwrap();
}

/// `model.safetensors.index.json`: `text` in the first shard, [`VISION`] alone in the second.
///
/// **Written from the tensor list rather than alongside it**, and re-written whenever that list
/// changes, because the index is what `open_indexed` selects shards by — a refusal test that
/// dropped a tensor from the shard and left it in the index would be testing a truncated-file
/// error instead of the completeness walk. The vision half is the constant either way: it is
/// never opened, which is the whole point of counting it here.
fn write_index(src: &Path, text: &[Tensor]) {
    let mut map = serde_json::Map::new();
    for (n, _, _) in text {
        map.insert(n.clone(), json!(TEXT_SHARD));
    }
    for n in VISION {
        map.insert(n.to_string(), json!(VISION_SHARD));
    }
    std::fs::write(
        src.join("model.safetensors.index.json"),
        json!({ "weight_map": map }).to_string(),
    )
    .unwrap();
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
fn write_fixture(src: &Path) -> Vec<Tensor> {
    std::fs::create_dir_all(src).unwrap();
    let config = glimmer_config_json();
    std::fs::write(
        src.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    // Parsed back rather than built from the constants: the fixture's shapes then come from the
    // same `validate`d config the converter will read, and a config this test writes that the
    // schema would refuse fails HERE rather than as a confusing converter error.
    let cfg: GlimmerConfig = parse_config(&config.to_string()).expect("the fixture config parses");
    let text = text_tensors(&cfg);
    let vision: Vec<Tensor> = VISION.iter().map(|n| tensor(n, vec![8, HIDDEN])).collect();
    write_shard(&src.join(TEXT_SHARD), &text);
    write_shard(&src.join(VISION_SHARD), &vision);

    write_index(src, &text);

    // The four AUX files. Stub contents except `generation_config.json`, whose ids are the one
    // thing the converter reads rather than copies — and both are inside VOCAB, since an id past
    // it is a stop token no argmax can return and is its own refusal below.
    for (name, body) in [
        ("tokenizer.json", "{\"model\":{\"type\":\"BPE\"}}"),
        ("tokenizer_config.json", "{}"),
        ("chat_template.jinja", "{{ messages }}"),
    ] {
        std::fs::write(src.join(name), body).unwrap();
    }
    write_eos(src, &[(VOCAB - 3) as u32, (VOCAB - 1) as u32]);
    text
}

fn run(src: &Path, out: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_convert_glimmer"))
        .args([src.to_str().unwrap(), out.to_str().unwrap()])
        .output()
        .expect("run convert_glimmer")
}

/// `run`, expecting a refusal whose message names `want`.
///
/// The `want` check is the point: a refusal test that only asserts non-zero exit passes when the
/// binary fails for an unrelated reason, which is how a guard gets deleted without a red test.
fn refuses(src: &Path, out: &Path, want: &str) {
    let o = run(src, out);
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(
        !o.status.success() && err.contains(want),
        "expected a refusal naming {want:?}, got status {:?}:\n{err}",
        o.status.code()
    );
}

#[test]
fn convert_glimmer_writes_a_bf16_artifact_that_reopens_as_the_same_model() {
    let root = common::scratch("glimmer-convert-rt");
    let (src, out) = (root.join("src"), root.join("out"));
    let tensors = write_fixture(&src);

    let o = run(&src, &out);
    let log = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(o.status.success(), "convert_glimmer failed:\n{log}");

    // The counts are the OBSERVATION that the vision half was excluded, rather than the
    // assumption — and the 3 comes from the index, since the shard holding them was never
    // opened. Four norms per layer plus the model-level one is what gets widened.
    assert!(log.contains("3 vision tensors skipped"), "{log}");
    assert!(
        log.contains(&format!("{} norms widened", LAYERS * 4 + 1)),
        "{log}"
    );
    // Both ids printed, so an operator can notice a wrong set before a decode runs to its limit.
    assert!(
        log.contains(&format!("eos_token_id [{}, {}]", VOCAB - 3, VOCAB - 1)),
        "{log}"
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
    for (name, shape, bytes) in &tensors {
        if name.ends_with("norm.weight") {
            // Widened, and widened CORRECTLY — not merely present at the right length. A
            // byte-length check alone passes on a zeroed tensor.
            let (got, got_shape) = art.typed(name, fmt::Dtype::F32).unwrap();
            assert_eq!(got_shape, &shape[..], "{name} shape");
            let want: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let got: Vec<f32> = got
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(got, want, "{name} widened to the wrong values");
        } else {
            let (got, got_shape) = art.typed(name, fmt::Dtype::Bf16).unwrap();
            assert_eq!(got_shape, &shape[..], "{name} shape");
            assert_eq!(got, &bytes[..], "{name} is not byte-identical");
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
    let _ = std::fs::remove_dir_all(&root);
}

/// The guards that fire before 55 GB is written. Each arm restores the fixture, so every
/// assertion is about its own mutation rather than about the wreckage of the previous one.
#[test]
fn convert_glimmer_refuses_before_it_writes() {
    let root = common::scratch("glimmer-convert-refuse");
    let (src, out) = (root.join("src"), root.join("out"));
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
    let kept: Vec<Tensor> = text_tensors(&cfg)
        .into_iter()
        .filter(|(n, _, _)| *n != dropped)
        .collect();
    write_shard(&src.join(TEXT_SHARD), &kept);
    write_index(&src, &kept);
    refuses(&src, &out, &dropped);
    assert!(
        !out.join("resident.safetensors").exists(),
        "the artifact must not exist after a refusal"
    );
    let _ = std::fs::remove_dir_all(&root);
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
    let root = common::scratch("glimmer-convert-eos");
    let (src, out) = (root.join("src"), root.join("out"));
    write_fixture(&src);

    // **A DISTINCT pair, written here rather than left at the fixture's default**, so the
    // assertion proves the artifact TRACKED the source rather than that some constant matched.
    // Compared as BYTES: the copy is `std::fs::copy`, so byte equality is the property, and a
    // parse-and-compare would pass a reordering or an added key — and would be a third parser of
    // this field in the tree.
    let ids = [1u32, (VOCAB - 2) as u32];
    write_eos(&src, &ids);
    let want = std::fs::read(src.join(GEN)).unwrap();
    let o = run(&src, &out);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
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
    let _ = std::fs::remove_dir_all(&root);
}
