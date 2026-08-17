//! **The DFlash drafter checkpoint is exactly the census `DrafterConfig` derives — measured
//! against the REAL file, not against the spec.** M17b's gate.
//!
//! `crates/artifact/src/drafter_config.rs`'s own tests derive the shapes from the schema and
//! fire every refusal, but they gate the schema against `glimmer-architecture.md` §11 — a
//! document. This file gates it against `/swarm/storage/ai/rivoli/muse-glimmer-30b-assistant`,
//! which arrived after that schema was written and immediately settled the one thing the
//! schema could only flag: **the checkpoint carries 58 tensors, not the spec's 59.**
//! `DrafterConfig::census`'s header predicted the discrepancy and named an `fc` bias as the
//! candidate for the 59th. There is no bias. The spec's prose is wrong and its own per-layer
//! enumeration — which derives 58 — is right.
//!
//! **The checkpoint's HEADER is vendored, the checkpoint is not.** 6,304 bytes of a
//! 5,111,976,608-byte file: the 8-byte length plus the safetensors JSON, which carries every
//! name, shape, dtype and byte offset. That is the whole census and none of the weights, so
//! the gate below runs with no NFS, no 5 GB read, and no device — and it still fails on a
//! checkpoint whose tensor set is not this one. The bytes are pinned by length and FNV-1a the
//! way the anchor goldens are, and [`the_vendored_header_is_the_live_checkpoints_own_bytes`]
//! **recomputes the pin from the live file** when it is mounted, because a pin checked only
//! against a frozen copy of itself is decoration.
//!
//! **What this canNOT see, stated rather than left latent:** when the checkpoint is absent the
//! live comparison has nothing to compare, so it degrades to the vendored-only checks the other
//! tests already make. It is never vacuous — it always parses and counts the vendored header —
//! but its EXTRA power is conditional on the mount, and a green run does not by itself say
//! which of the two it got. The run that vendored these bytes is recorded in
//! `docs/measurement/glimmer-reference/drafter-checkpoint.md`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::drafter_config::DrafterConfig;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

mod common;

/// The shipped assistant checkpoint, as it sits on this box.
const CKPT_DIR: &str = "/swarm/storage/ai/rivoli/muse-glimmer-30b-assistant";

/// The first 6,304 bytes of that directory's `model.safetensors`: `u64` header length, then the
/// header JSON. Vendored 2026-08-17; see the module header for why only the header.
const VENDORED: &[u8] = include_bytes!("drafter-checkpoint-header.bin");

/// The shipped `config.json`, vendored beside the target's own in the reference directory.
const REAL_CONFIG: &str =
    include_str!("../../../docs/measurement/glimmer-reference/assistant-config.json");

/// Length and FNV-1a of [`VENDORED`], measured 2026-08-17 off the live file.
const VENDORED_LEN: usize = 6_304;
const VENDORED_FNV: u64 = 0xf0c2_c649_67d7_9b88;

/// `ls -l` on the live checkpoint, 2026-08-17.
const FILE_BYTES: u64 = 5_111_976_608;
/// The sum of every tensor's byte span — the pin's bf16 half, before the norms are widened.
const TENSOR_BYTES: u64 = 5_111_970_304;

/// One parsed header entry. The offsets are kept because the byte arithmetic below is the only
/// check that the header describes a FILE rather than a plausible-looking set of shapes.
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}

/// The vendored header, parsed: length prefix honoured, `__metadata__` split off.
///
/// Returns a `BTreeMap` so every comparison below is order-independent — safetensors makes no
/// ordering promise, and a gate that depended on one would redden on a re-save that changed
/// nothing.
fn header(bytes: &[u8]) -> (BTreeMap<String, Entry>, Value, usize) {
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8-byte length prefix")) as usize;
    assert_eq!(
        bytes.len(),
        8 + n,
        "the vendored prefix is not exactly the length prefix plus the header it declares"
    );
    let mut json: Value = serde_json::from_slice(&bytes[8..8 + n]).expect("the header is JSON");
    let meta = json
        .as_object_mut()
        .expect("the header is a JSON object")
        .remove("__metadata__")
        .unwrap_or(Value::Null);
    let map = json
        .as_object()
        .expect("the header is a JSON object")
        .iter()
        .map(|(name, v)| {
            let off = &v["data_offsets"];
            let e = Entry {
                dtype: v["dtype"].as_str().expect("a dtype").to_string(),
                shape: v["shape"]
                    .as_array()
                    .expect("a shape")
                    .iter()
                    .map(|d| d.as_u64().expect("a dimension") as usize)
                    .collect(),
                begin: off[0].as_u64().expect("a start offset"),
                end: off[1].as_u64().expect("an end offset"),
            };
            (name.clone(), e)
        })
        .collect();
    (map, meta, n)
}

/// The shipped assistant config, read through the converter's own loader.
///
/// Via a scratch directory rather than a parse call because `DrafterConfig::parse` is private —
/// which is the right shape to test through anyway: `load` is what the converter calls, so the
/// manifest-then-config fallback is exercised rather than bypassed.
fn real(tag: &str) -> DrafterConfig {
    // The tag is per CALLER, not a constant: `common::scratch` keys the directory on the process
    // id alone, so two tests sharing a tag race each other's `remove_dir_all` under libtest's
    // default parallelism. Observed on first run — three tests, one tag, two of them red.
    let dir = common::scratch(tag);
    std::fs::write(dir.join("config.json"), REAL_CONFIG).expect("stage the shipped config");
    DrafterConfig::load(dir.to_str().expect("utf-8 scratch path"))
        .expect("the shipped assistant config.json parses and validates")
}

#[test]
fn the_vendored_header_is_the_bytes_that_were_measured() {
    assert_eq!(VENDORED.len(), VENDORED_LEN, "vendored header length");
    assert_eq!(
        rivoli_core::hash::fnv1a(VENDORED),
        VENDORED_FNV,
        "vendored header FNV-1a — the bytes moved without the pin moving with them"
    );
    let (_, meta, n) = header(VENDORED);
    // The header length is part of the file's byte arithmetic below, so it is pinned, not
    // merely parsed. 6,296 + the 8-byte prefix is the 6,304 above.
    assert_eq!(n, VENDORED_LEN - 8, "declared header length");
    assert_eq!(meta["format"], "pt", "the checkpoint's own format tag");
}

#[test]
fn the_real_checkpoint_is_exactly_the_census_the_schema_derives() {
    let cfg = real("drafter-cfg-census");
    let census = cfg.census();
    let (have, _, _) = header(VENDORED);

    // SET equality in both directions, reported by name — the same comparison
    // `convert_glimmer_drafter::ensure_census` makes, run here against the real header so the
    // converter's first contact with this file cannot be its first test.
    let want: std::collections::BTreeSet<&str> = census.iter().map(|(n, _)| n.as_str()).collect();
    let got: std::collections::BTreeSet<&str> = have.keys().map(String::as_str).collect();
    let missing: Vec<_> = want.difference(&got).collect();
    let extra: Vec<_> = got.difference(&want).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "census vs the real checkpoint: missing {missing:?}, unexpected {extra:?}"
    );

    // 58, and the arithmetic that makes it 58 rather than a number to remember: eleven per
    // layer plus three model-level. The spec says 59; the file says 58 and there is no bias.
    assert_eq!(census.len(), 58, "the shipped drafter's tensor count");
    assert_eq!(
        census.len(),
        11 * cfg.num_hidden_layers + 3,
        "58 is 11 per layer x 5 layers + 3 globals, not a remembered constant"
    );

    for (name, shape) in &census {
        assert_eq!(
            &have[name].shape,
            shape,
            "{name}: checkpoint shape != census shape"
        );
    }
}

#[test]
fn the_encoder_projection_proves_the_five_hidden_concat() {
    let cfg = real("drafter-cfg-encoder");
    let (have, _, _) = header(VENDORED);
    let h = cfg.hidden_size;
    let ids = cfg.target_layer_ids();

    // THE tensor that makes this a DFlash drafter rather than a small target: `encoder.fc`
    // takes the CONCATENATION of the target's hidden state at each of `target_layer_ids` and
    // projects it back to one hidden width. Its input width is therefore 5 x 6656 = 33280, and
    // that product is the only place in the checkpoint where the length of `target_layer_ids`
    // is visible in a SHAPE rather than in the config — so a config listing the wrong number of
    // target layers reddens here instead of being agreed with.
    assert_eq!(ids, vec![1, 13, 25, 37, 49], "the shipped target_layer_ids");
    assert_eq!(
        have["encoder.fc.weight"].shape,
        vec![h, ids.len() * h],
        "encoder.fc is [hidden, n_targets x hidden]"
    );
    assert_eq!(
        have["encoder.fc.weight"].shape,
        vec![6656, 33280],
        "the shipped widths, spelled out once so the derivation above has something to be \
         checked against"
    );

    // And the two absences that are the other half of the same claim: the drafter owns neither
    // the embedding nor the lm_head. It borrows the target's, which is why there is no
    // standalone drafter artifact and why the converter treats their PRESENCE as an error.
    for absent in [
        "model.language_model.embed_tokens.weight",
        "embed_tokens.weight",
        "lm_head.weight",
    ] {
        assert!(
            !have.contains_key(absent),
            "{absent} is in the checkpoint — the drafter is supposed to BORROW it"
        );
    }
}

#[test]
fn every_tensor_is_bf16_and_the_offsets_tile_the_file() {
    let (have, _, n) = header(VENDORED);
    let mut spans: Vec<(u64, u64, &str)> = have
        .iter()
        .map(|(name, e)| {
            assert_eq!(
                e.dtype, "BF16",
                "{name}: the whole checkpoint is bf16, and the converter copies it verbatim"
            );
            let elems: usize = e.shape.iter().product();
            assert_eq!(
                e.end - e.begin,
                (elems * 2) as u64,
                "{name}: byte span is not 2 bytes per bf16 element"
            );
            (e.begin, e.end, name.as_str())
        })
        .collect();
    spans.sort_unstable();

    // Contiguous and non-overlapping from 0: this is what turns the shape census into a claim
    // about a FILE. A header can carry correct shapes and still not describe 5,111,976,608
    // bytes; only the offsets say it does.
    let mut cursor = 0u64;
    for (begin, end, name) in &spans {
        assert_eq!(*begin, cursor, "{name}: a gap or overlap before this tensor");
        cursor = *end;
    }
    assert_eq!(cursor, TENSOR_BYTES, "total tensor bytes");
    assert_eq!(
        8 + n as u64 + TENSOR_BYTES,
        FILE_BYTES,
        "8-byte prefix + {n}-byte header + tensor bytes must be the file's own size"
    );
}

#[test]
fn the_resident_pin_is_the_checkpoint_plus_the_widened_norms() {
    let cfg = real("drafter-cfg-pin");
    let census = cfg.census();

    // The converter's own rank rule: 1-D is a norm and is stored f32, everything else stays
    // bf16 verbatim. So the pin is NOT the checkpoint's size — it is the checkpoint plus two
    // more bytes for every norm element, and that is the number P6 costs the drafter at.
    let (mut verbatim, mut widened, mut norm_elems) = (0usize, 0usize, 0usize);
    for (_, shape) in &census {
        let elems: usize = shape.iter().product();
        if shape.len() == 1 {
            widened += 1;
            norm_elems += elems;
        } else {
            verbatim += 1;
        }
    }
    // 4 per layer (both layernorms plus q_norm/k_norm) plus the encoder's output norm and the
    // final norm; the rest are projections.
    assert_eq!(
        (verbatim, widened),
        (36, 22),
        "the verbatim/widened split the converter prints"
    );
    assert_eq!(
        widened,
        4 * cfg.num_hidden_layers + 2,
        "22 is 4 norms per layer + encoder.output_norm_enc + norm"
    );
    assert_eq!(norm_elems, 81_152, "elements stored f32 rather than bf16");

    let pin = TENSOR_BYTES + (norm_elems * 2) as u64;
    assert_eq!(pin, 5_112_132_608, "the resident pin, in bytes");
    // 4.761 GiB. The plan said "5.1 GiB", which is the file's size in GB read as GiB — a
    // 7.6% overstatement of a pin that P6 spends against free memory.
    let gib = pin as f64 / f64::from(1u32 << 30);
    assert!(
        (4.760..4.762).contains(&gib),
        "the pin is {gib:.4} GiB, outside the 4.761 GiB this wave budgeted"
    );
}

#[test]
fn the_vendored_header_is_the_live_checkpoints_own_bytes() {
    // Never vacuous: the vendored bytes are parsed and counted on every run, mounted or not.
    let (have, _, _) = header(VENDORED);
    assert_eq!(have.len(), 58, "the vendored header's own tensor count");

    let path = std::path::Path::new(CKPT_DIR).join("model.safetensors");
    let Ok(meta) = std::fs::metadata(&path) else {
        // The mount is absent. Recorded in the module header as this gate's one conditional
        // half; the census above stands on the vendored bytes alone.
        return;
    };
    assert_eq!(meta.len(), FILE_BYTES, "the live checkpoint's size moved");

    // Recomputed from the live file rather than compared against a second frozen copy — the
    // whole point of pinning provenance by value.
    use std::io::Read;
    let mut live = vec![0u8; VENDORED_LEN];
    std::fs::File::open(&path)
        .expect("open the live checkpoint")
        .read_exact(&mut live)
        .expect("read its header");
    assert_eq!(
        rivoli_core::hash::fnv1a(&live),
        VENDORED_FNV,
        "the live checkpoint's header is not the vendored one"
    );
    assert_eq!(live, VENDORED, "byte-for-byte, so the FNV is not the claim");
}

#[test]
fn the_shipped_config_is_the_drafter_the_spec_describes() {
    let cfg = real("drafter-cfg-spec");
    // The seven §11 facts that make this a block-diffusion drafter rather than a small dense
    // model, read off the shipped config so a re-download that changed one reddens here.
    assert_eq!(cfg.model_type, "muse_glimmer_assistant");
    assert_eq!(cfg.num_hidden_layers, 5);
    assert_eq!(cfg.block_size, 16, "the drafted block width");
    assert_eq!(cfg.mask_token_id, 201_818, "the noise token the block starts as");
    assert_eq!((cfg.num_attention_heads, cfg.num_key_value_heads), (32, 8));
    assert_eq!((cfg.hidden_size, cfg.head_dim), (6656, 128));
    assert_eq!(cfg.intermediate_size, 19968);
    assert_eq!(cfg.sliding_window, 2048);
    // 32 x 128 = 4096 != 6656: the head width and the hidden width are DIFFERENT here, which is
    // the trap every fixture in this tree is built to keep visible.
    assert_ne!(
        cfg.num_attention_heads * cfg.head_dim,
        cfg.hidden_size,
        "head width and hidden width must stay unequal or a converter that derived one passes"
    );

    // 2.556 B parameters, from the census rather than from the model card.
    let params: usize = cfg
        .census()
        .iter()
        .map(|(_, shape)| shape.iter().product::<usize>())
        .sum();
    assert_eq!(params, 2_555_985_152, "the drafter's parameter count");
    assert_eq!(
        params * 2,
        TENSOR_BYTES as usize,
        "every parameter is two bytes, so the census IS the file's tensor half"
    );
}

// ------------------------------------------------------------------------------------------
// The converter's pairing refusals, fired against the REAL pair of configs.
// ------------------------------------------------------------------------------------------
//
// `refuse_before_writing` cross-checks THREE facts about the borrow before it reads a tensor,
// and every one of them is a fact about the drafter against ITS TARGET — so a synthetic pair
// tests the comparison and the real pair tests the shipped models. Both configs are vendored
// (`assistant-config.json` here, `config.json` beside it for the target, which the anchor
// already reads), and neither arm needs a single weight byte: the census walk that would read
// them comes after these checks, so every run below stops at the absent `model.safetensors`.
//
// That stop is what makes the POSITIVE arm meaningful. "The shipped drafter pairs with the
// shipped target" is not observable from a refusal test unless the run gets PAST the three
// checks, and the only evidence that it did is which error it dies on.

const BIN: common::ConvertBin = common::ConvertBin {
    exe: env!("CARGO_BIN_EXE_convert_glimmer_drafter"),
    tool: "convert_glimmer_drafter",
};

/// The target's own shipped `config.json`, the same file the S1b anchor pins its tiny config's
/// "real" fields against.
const TARGET_CONFIG: &str = include_str!("../../../docs/measurement/glimmer-reference/config.json");

/// A src dir holding one `config.json`, and a target artifact dir holding one `manifest.json`.
///
/// The target needs no tensors and no aux: `refuse_before_writing` reaches it only through
/// `GlimmerConfig::load`, which reads the manifest and nothing else. Writing the shipped config
/// AS the manifest is exactly what `finish_artifact` does, minus the `format` section that no
/// check on this path consults.
fn pair(tag: &str, drafter: &Value) -> (PathBuf, PathBuf) {
    let root = common::scratch(tag);
    let (src, out) = (root.join("src"), root.join("out"));
    std::fs::create_dir_all(&src).expect("create src");
    std::fs::create_dir_all(&out).expect("create out");
    std::fs::write(src.join("config.json"), drafter.to_string()).expect("write drafter config");
    std::fs::write(out.join("manifest.json"), TARGET_CONFIG).expect("write target manifest");
    (src, out)
}

/// The shipped assistant config as a mutable document.
fn shipped() -> Value {
    serde_json::from_str(REAL_CONFIG).expect("the shipped assistant config.json is JSON")
}

#[test]
fn the_shipped_drafter_pairs_with_the_shipped_target() {
    let (src, out) = pair("drafter-pair-ok", &shipped());
    // Past all three cross-checks and stopped by the absent weights — which is the assertion.
    // A run that died on any of the three would name that check instead, and `expect_refusal`
    // fails on a refusal it was not asked for, so this cannot pass for the wrong reason.
    BIN.at(&src, &out).refuses(&[], "no *.safetensors in");
}

#[test]
fn a_drafter_whose_hidden_width_is_not_the_targets_is_refused() {
    let mut cfg = shipped();
    // 6656 -> 6144, which is GLM's hidden: a plausible wrong pairing rather than nonsense.
    cfg["hidden_size"] = Value::from(6144);
    let (src, out) = pair("drafter-pair-hidden", &cfg);
    BIN.at(&src, &out)
        .refuses(&[], "drafter hidden_size 6144 != target hidden 6656");
}

#[test]
fn a_target_layer_id_past_the_targets_depth_is_refused() {
    let mut cfg = shipped();
    // The shipped ids are [1, 13, 25, 37, 49] against 52 layers, so 49 is the last one that
    // fits and 52 is the first that does not — the off-by-one, not an arbitrary large number.
    cfg["target_layer_ids"] = serde_json::json!([1, 13, 25, 37, 52]);
    let (src, out) = pair("drafter-pair-layer", &cfg);
    BIN.at(&src, &out)
        .refuses(&[], "target_layer_ids entry 52 is past the target's 52 layers");
}

#[test]
fn a_mask_token_past_the_targets_vocabulary_is_refused() {
    let mut cfg = shipped();
    // The drafter embeds its noise rows through the TARGET's embedding table, so this id is an
    // index into a matrix this checkpoint does not own. Set it one past the last legal row.
    let vocab = serde_json::from_str::<Value>(TARGET_CONFIG).expect("target config is JSON")
        ["text_config"]["vocab_size"]
        .as_u64()
        .expect("the target declares a vocab_size");
    cfg["mask_token_id"] = Value::from(vocab);
    let (src, out) = pair("drafter-pair-mask", &cfg);
    BIN.at(&src, &out)
        .refuses(&[], &format!("mask_token_id {vocab} is past the target's vocabulary"));
}

#[test]
fn attaching_to_something_that_is_not_an_artifact_is_refused() {
    let (src, out) = pair("drafter-pair-noartifact", &shipped());
    // The drafter attaches to an ARTIFACT, never to a bare checkpoint directory — otherwise it
    // writes a `drafter/` layout somewhere the engine never looks.
    std::fs::remove_file(out.join("manifest.json")).expect("remove the target manifest");
    BIN.at(&src, &out)
        .refuses(&[], "is not a Muse Glimmer artifact — run convert_glimmer first");
}

// `SafeWriter::refuse_writing_into_source` is deliberately NOT gated here. With `src == out`
// the drafter loader prefers `manifest.json`, so it reads the TARGET's document as the drafter's
// and the run refuses one check earlier — a test naming the self-write guard would pass on the
// wrong refusal. The guard is shared and gated by the other converter suites.
