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
//! **Vendored as BYTES rather than as text**, which review 2026-08-17 challenged: only the first
//! 8 of the 6,304 are genuinely binary (a little-endian `u64` length), the rest is JSON, and a
//! text fixture would diff readably. Measured before deciding — the header JSON contains **zero
//! newlines**; it is one 6,296-character line. So a text fixture would diff as one changed line
//! exactly as the blob does, and the legibility the change buys is nil. Declined on that
//! measurement rather than on preference; if safetensors ever emits pretty-printed headers the
//! trade flips.
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
            &have[name].shape, shape,
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

    // The other half of the same claim -- that the drafter owns NEITHER the embedding nor the
    // lm_head, and borrows the target's -- is deliberately NOT re-asserted here. Review
    // 2026-08-17 traced it: `the_real_checkpoint_is_exactly_the_census_the_schema_derives` is
    // bidirectional SET equality, and `census()` emits no `embed_tokens` and no `lm_head`, so
    // either one appearing in the checkpoint lands in that test's `extra` and reddens it. An
    // explicit absence loop here was three assertions restating a stronger one next door.
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
        assert_eq!(
            *begin, cursor,
            "{name}: a gap or overlap before this tensor"
        );
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

    // The same split, taken from the real HEADER rather than from the config. Added 2026-08-17
    // after review pointed out that this test -- in a file whose premise is "against the real
    // checkpoint" -- read only `cfg.census()` and never touched a checkpoint byte, so a reader
    // would reasonably assume it did. Now both sides are computed and compared: the config says
    // which names are rank-1, the file says which tensors carry a 1-D shape, and 36/22 has to
    // fall out of each independently. Red-proofed by retyping `norm.weight` as `[6656, 1]` in
    // the vendored header -- same element count, same byte span, and this assert is the one that
    // notices (gate-red-proofs.md section 6, plant P7).
    let (have, _, _) = header(VENDORED);
    let (mut f_verbatim, mut f_widened, mut f_norm_elems) = (0usize, 0usize, 0usize);
    for entry in have.values() {
        let elems: usize = entry.shape.iter().product();
        if entry.shape.len() == 1 {
            f_widened += 1;
            f_norm_elems += elems;
        } else {
            f_verbatim += 1;
        }
    }
    assert_eq!(
        (f_verbatim, f_widened, f_norm_elems),
        (verbatim, widened, norm_elems),
        "the real header and the derived census disagree about which tensors are norms"
    );
    assert_eq!(
        widened,
        4 * cfg.num_hidden_layers + 2,
        "22 is 4 norms per layer + encoder.output_norm_enc + norm"
    );
    assert_eq!(norm_elems, 81_152, "elements stored f32 rather than bf16");

    let pin = TENSOR_BYTES + (norm_elems * 2) as u64;
    assert_eq!(pin, 5_112_132_608, "the resident pin, in bytes");
    // 4.761 GiB. The plan said "5.1 GiB", which is the file's size in GB read as GiB: 5.1 GiB is
    // 5,476,083,302 B against this 5,112,132,608, so the overstatement is 363,950,694 B = 7.1% of
    // a pin that P6 spends against free memory. (Said "7.6%" until 2026-08-17, when review
    // recomputed it — a percentage with no visible numerator is one nobody can check.)
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
    let missing = std::fs::metadata(&path);
    // **The required mode, on the RIVOLI_CS_REQUIRED precedent** (`CLAUDE.md` §Gates → CodeScene):
    // P7 says a check whose examined-count can silently reach zero is not a check, and a mount
    // that vanishes takes this comparison to zero examinations while the suite still reports
    // all-green. Stating that honestly is not the same as enforcing it, which review 2026-08-17
    // pointed out — this repo had already built the mechanism for exactly this failure mode and
    // this gate was not using it. Set the variable wherever the checkpoint is supposed to be
    // present and absence becomes a panic naming the path, rather than a silent pass.
    let required = std::env::var_os("RIVOLI_DRAFTER_CKPT_REQUIRED").is_some();
    let Ok(meta) = missing else {
        assert!(
            !required,
            "RIVOLI_DRAFTER_CKPT_REQUIRED is set but {} is absent — the live-header comparison \
             examined nothing",
            path.display()
        );
        // Unset: the mount is genuinely optional here, and the census stands on the vendored
        // bytes alone. Recorded in the module header as this gate's one conditional half.
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
    // **This overlaps `drafter_config.rs::the_defaults_are_the_shipped_drafters` on purpose**, and
    // review 2026-08-17 asked why. The two read DIFFERENT literals: that test pins the schema's
    // own `#[serde(default)]` values, this one pins the vendored `assistant-config.json`. Nothing
    // in the tree forces those two to agree, and both are claims about the same shipped model --
    // so the pair is a cross-check, and collapsing it to one would delete the only thing that
    // notices when a default drifts away from the file it was copied from. `DrafterConfig` has no
    // `PartialEq`, so it cannot be spelled as a single equality between the two.
    let cfg = real("drafter-cfg-spec");
    // The seven §11 facts that make this a block-diffusion drafter rather than a small dense
    // model, read off the shipped config so a re-download that changed one reddens here.
    assert_eq!(cfg.model_type, "muse_glimmer_assistant");
    assert_eq!(cfg.num_hidden_layers, 5);
    assert_eq!(cfg.block_size, 16, "the drafted block width");
    assert_eq!(
        cfg.mask_token_id, 201_818,
        "the noise token the block starts as"
    );
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
    BIN.at(&src, &out).refuses(
        &[],
        "target_layer_ids entry 52 is past the target's 52 layers",
    );
}

#[test]
fn a_mask_token_past_the_targets_vocabulary_is_refused() {
    let mut cfg = shipped();
    // The drafter embeds its noise rows through the TARGET's embedding table, so this id is an
    // index into a matrix this checkpoint does not own. Set it one past the last legal row.
    let vocab =
        serde_json::from_str::<Value>(TARGET_CONFIG).expect("target config is JSON")["text_config"]
            ["vocab_size"]
            .as_u64()
            .expect("the target declares a vocab_size");
    cfg["mask_token_id"] = Value::from(vocab);
    let (src, out) = pair("drafter-pair-mask", &cfg);
    BIN.at(&src, &out).refuses(
        &[],
        &format!("mask_token_id {vocab} is past the target's vocabulary"),
    );
}

#[test]
fn attaching_to_something_that_is_not_an_artifact_is_refused() {
    let (src, out) = pair("drafter-pair-noartifact", &shipped());
    // The drafter attaches to an ARTIFACT, never to a bare checkpoint directory — otherwise it
    // writes a `drafter/` layout somewhere the engine never looks.
    std::fs::remove_file(out.join("manifest.json")).expect("remove the target manifest");
    BIN.at(&src, &out).refuses(
        &[],
        "is not a Muse Glimmer artifact — run convert_glimmer first",
    );
}

// `SafeWriter::refuse_writing_into_source` is deliberately NOT gated here. With `src == out`
// the drafter loader prefers `manifest.json`, so it reads the TARGET's document as the drafter's
// and the run refuses one check earlier — a test naming the self-write guard would pass on the
// wrong refusal. The guard is shared and gated by the other converter suites.

/// **The three per-token budgets M17c is sized by, derived rather than quoted.**
///
/// They were prose in `glimmer-reference/drafter-checkpoint.md` and in nothing else, which review
/// 2026-08-17 called correctly: under P5 bytes/token is the currency, and under P7 a currency
/// figure with no gate is a number waiting to rot. Every one below is computed from the shipped
/// config, so a checkpoint whose `head_dim`, `num_key_value_heads` or `hidden_size` moved would
/// redden here instead of leaving the doc quietly wrong.
///
/// **KiB and MiB, not kB and MB** — these are memory budgets against a GTT allocation, and the
/// exactness is the point: 20,480 B *is* 20.0 KiB with no rounding, and the ring *is* 260.0 MiB.
/// A "~20 KiB" would read like an estimate of a number that has none.
#[test]
fn the_per_token_budgets_follow_from_the_shipped_config() {
    let cfg = real("drafter-cfg-budgets");
    let (kv_heads, hd, layers) = (cfg.num_key_value_heads, cfg.head_dim, cfg.num_hidden_layers);
    // The two dtypes, named once. BF16 is the checkpoint's and the plan's; F32 is what
    // `glimmer::pin::scratch` and `geometry::kv_bytes` actually allocate.
    const BF16: usize = 2;
    const F32: usize = 4;

    // Drafter KV cache. TWO tensors per layer (K and V), and the drafter's OWN kv_heads -- not
    // the target's, which is the `TargetGrouping` defect the oracle plants.
    let kv_elems = 2 * kv_heads * hd * layers;
    assert_eq!(
        kv_elems * BF16,
        20 * 1024,
        "drafter KV at checkpoint dtype: 20,480 B = 20.0 KiB/token exactly -- the plan's figure"
    );
    assert_eq!(
        kv_elems * F32,
        40 * 1024,
        "drafter KV as THIS ENGINE would allocate it: 40,960 B = 40.0 KiB/token, twice the plan"
    );

    // The target-side hidden-state export: one hidden row per target layer. This is the cost the
    // TARGET pays to feed the drafter, charged per ACCEPTED token.
    let export_elems = cfg.target_layer_ids().len() * cfg.hidden_size;
    assert_eq!(
        export_elems * BF16,
        66_560,
        "export bytes/token at bf16 -- the plan's figure"
    );
    assert_eq!(
        export_elems * F32,
        133_120,
        "export bytes/token straight out of the f32 residual stream, with no narrowing"
    );

    // The ring at the context this wave sizes for -- what the dtype decision actually costs.
    assert_eq!(
        export_elems * BF16 * 4096,
        260 * 1024 * 1024,
        "260.0 MiB ring at ctx 4096 IF the export narrows to bf16"
    );
    assert_eq!(
        export_elems * F32 * 4096,
        520 * 1024 * 1024,
        "520.0 MiB if it does not -- the same ring, one dtype decision apart"
    );

    // The export dominates the drafter's own KV at either dtype, which is the asymmetry M17c's
    // ring sizing turns on: the export scales with CONTEXT, the KV with the drafted block.
    assert!(
        export_elems > 3 * kv_elems,
        "the export is supposed to dominate the drafter's own KV"
    );
}

// ------------------------------------------------------------------------------------------
// The mask the SERVING path must build, which is not the one the anchor pins.
// ------------------------------------------------------------------------------------------

/// Block-vs-block pairs that attend, strictly-bidirectional pairs among them, and masked context
/// columns, for the reference's own overlay `|q_idx - kv_idx| <= sliding_window`.
///
/// `q_offset` is the whole point of the function existing. The reference derives it in
/// `masking_utils.py::_preprocess_mask_arguments`: `q_offset = past_key_values.get_query_offset(..)`
/// when a cache is present, and **`q_offset = 0` when it is not**. Both branches reach the same
/// overlay, so the mask's meaning turns entirely on which branch built it.
fn mask_shape(ctx: usize, block: usize, window: usize, q_offset: usize) -> (usize, usize, usize) {
    let (mut attending, mut strict, mut ctx_masked) = (0usize, 0usize, 0usize);
    for row in 0..block {
        let q = (row + q_offset) as i64;
        for kv in 0..ctx + block {
            let ok = (q - kv as i64).abs() <= window as i64;
            if kv >= ctx {
                attending += usize::from(ok);
                // A query attending a LATER row of its own block — precisely what a causal mask
                // forbids, and therefore the only positive evidence of bidirectionality.
                strict += usize::from(ok && kv - ctx > row);
            } else {
                ctx_masked += usize::from(!ok);
            }
        }
    }
    (attending, strict, ctx_masked)
}

/// **THE ANCHOR'S MASK IS THE NO-CACHE MASK, AND TRANSLITERATING IT INTO THE ENGINE WOULD DELETE
/// THE DRAFTER'S DEFINING PROPERTY AT EVERY REAL CONTEXT.** Measured 2026-08-17, before the
/// kernel exists, which is the only order in which it can prevent anything.
///
/// The reference's overlay is `abs(q_idx - kv_idx) <= sliding_window`
/// (`masking_utils.py::sliding_window_bidirectional_overlay`), and `q_idx` is `row + q_offset`.
/// The S1b anchor ran with no cache — its own recorded reference behaviour is that a fresh
/// `DFlashCache` reports `kv_length` 0 and the correct 2D mask only works with `use_cache=False` —
/// so every vendored draft golden pins the **`q_offset = 0`** branch. At the fixture's ctx 12 /
/// block 4 / window 13 that yields 13 of 16 block-vs-block pairs, which is exactly what M17a
/// re-vendored the fixtures to obtain and what `glimmer_draft_oracle.rs` asserts.
///
/// **At the shipped widths the same expression yields ZERO.** With `window` 2048 and `block` 16,
/// `q_idx = row` is at most 15, so no query can reach a key at `kv >= ctx` once `ctx > window`:
/// the block does not attend itself at all, the drafter degenerates to a context-reader, and
/// nothing about the shapes or the byte counts would say so. The cache branch — `q_offset = ctx`,
/// which is what the serving path has and the anchor did not — restores it.
///
/// So the anchor **cannot arbitrate this**, and it is not a defect in the anchor: the two
/// indexings disagree even at the tiny geometry (13 of 16 against 16 of 16, and 0 masked context
/// columns against 3), so the fixture pins one of them by value and it is the wrong one for
/// serving. `dflash.rs`'s `mask` docstring anticipated the shape of this — "whether that
/// off-window indexing is desirable at real context lengths is a serving-path question the cache
/// answers, not this fixture" — and this test is that question answered.
///
/// **What M17c must therefore do:** carry `q_offset` as an explicit kernel argument, pass 0 when
/// scoring against the vendored goldens (so anchor parity stays exact) and `ctx` in the decode
/// path. Not a default, not an inferred value — the two regimes differ by 256 attending pairs at
/// ctx 4096 and a wrong default is invisible in every shape check in this file.
#[test]
fn the_serving_mask_indexes_queries_by_position_and_the_anchor_pins_the_other_branch() {
    let cfg = real("drafter-cfg-mask");
    let (block, window) = (cfg.block_size, cfg.sliding_window);
    assert_eq!((block, window), (16, 2048), "the shipped block and window");

    // The shipped widths, at a context past the window — the regime every real decode is in.
    let ctx = 4096;
    assert!(ctx > window, "the trap needs ctx past the window to bite");
    let (row_attending, row_strict, _) = mask_shape(ctx, block, window, 0);
    assert_eq!(
        (row_attending, row_strict),
        (0, 0),
        "row-indexed queries at ctx {ctx}: the block attends ITSELF zero times, so a kernel that \
         transliterates the anchor's mask is not a block drafter at all"
    );
    let (pos_attending, pos_strict, _) = mask_shape(ctx, block, window, ctx);
    assert_eq!(
        pos_attending,
        block * block,
        "position-indexed queries: every block pair attends"
    );
    assert_eq!(
        pos_strict,
        block * (block - 1) / 2,
        "and {} of them are strictly bidirectional — a causal mask allows none",
        block * (block - 1) / 2
    );

    // And the anchor's own geometry, where the two branches ALSO differ — which is why the
    // goldens cannot be read as evidence for either choice at real widths.
    let (a_row, a_row_strict, a_row_ctx) = mask_shape(12, 4, 13, 0);
    // The cache branch's strict count is asserted rather than discarded. It was `_` until
    // 2026-08-17, when review pointed out that the commit message cited 6 for this row and no test
    // held it -- so an edit to `strict` that broke exactly this case would have passed here while
    // the doc went on claiming the number.
    let (a_pos, a_pos_strict, a_pos_ctx) = mask_shape(12, 4, 13, 12);
    assert_eq!(
        (a_row, a_row_strict, a_row_ctx),
        (13, 3, 0),
        "the vendored fixture's pinned mask — 13 of 16, 3 strictly bidirectional, no context \
         column masked; `glimmer_draft_oracle.rs` asserts the same numbers from the golden BYTES"
    );
    assert_eq!(
        (a_pos, a_pos_strict, a_pos_ctx),
        (16, 6, 3),
        "the cache branch at the same widths is a different mask, so the fixture distinguishes \
         them and pins the no-cache one"
    );
    assert_ne!(
        a_row, a_pos,
        "if these agreed, the anchor would arbitrate and this test would be unnecessary"
    );
}

/// The set of KV rows one query attends, under a named lower/upper edge pair.
///
/// Separate from [`mask_shape`] because that function counts pairs and this one needs the ROWS: the
/// defect below is one row per query, which every count in this file rounds away.
fn attended(q: i64, kv_len: usize, window: i64, lower_is_strict: bool) -> Vec<i64> {
    let lo = if lower_is_strict {
        q - window + 1
    } else {
        q - window
    };
    (0..kv_len as i64)
        .filter(|kv| *kv >= lo && *kv <= q + window)
        .collect()
}

/// **THE BIDIRECTIONAL LOWER EDGE IS INCLUSIVE AND THE CAUSAL ONE IS STRICT, AND THE ANCHOR
/// FIXTURE CANNOT SEE THE DIFFERENCE.** Measured 2026-08-17.
///
/// The tree already has a Glimmer GQA attend kernel — `attn.hip::gqa_attend` — with the right head
/// grouping, the right layout, a multi-row `tq`, and a `start_pos` that is exactly the `q_offset`
/// the sibling test above says the drafter needs. So the natural way to build the block-attend
/// kernel is to copy it and widen the bound. **That produces a defect this anchor scores green.**
///
/// The reference's two overlays are not the same shape (`masking_utils.py`):
///
/// | overlay | expression | lower edge |
/// |---|---|---|
/// | `sliding_window_overlay` (causal) | `kv_idx > q_idx - window`, and-ed with `kv_idx <= q_idx` | **strict**: `q - w + 1` |
/// | `sliding_window_bidirectional_overlay` | `abs(q_idx - kv_idx) <= window` | **inclusive**: `q - w` |
///
/// `gqa_attend` implements the first and its own comment calls the off-by-one "trap 14", which is
/// why it computes `lo = pos - win + 1` rather than `pos - win`. **Carrying that `lo` into a
/// bidirectional kernel drops exactly one KV row per query row** — 16 rows at the shipped geometry,
/// every query affected.
///
/// **And the fixture is blind to it: 0 of its 4 query rows change.** At ctx 12 with window 13 the
/// lower edge is negative for every row, so it clamps to 0 either way and the two expressions agree
/// on every cell. Scoring a kernel against the vendored goldens therefore cannot distinguish
/// `pos - win` from `pos - win + 1`, and this is a SECOND blindness independent of the `q_offset`
/// one — that one the fixture answers wrongly, this one it cannot answer at all.
#[test]
fn the_bidirectional_lower_edge_is_inclusive_and_the_fixture_cannot_see_it() {
    let cfg = real("drafter-cfg-loweredge");
    let (block, window) = (cfg.block_size as i64, cfg.sliding_window as i64);

    // The shipped geometry, serving indexing: one row lost per query, every query.
    let (ctx, off) = (4096i64, 4096i64);
    let kv_len = (ctx + block) as usize;
    let mut affected = 0;
    for row in 0..block {
        let q = row + off;
        let right = attended(q, kv_len, window, false);
        let wrong = attended(q, kv_len, window, true);
        assert_eq!(
            right.len(),
            wrong.len() + 1,
            "row {row}: the inclusive edge must admit exactly one more KV row than the strict one"
        );
        assert_eq!(right[0], q - window, "row {row}: inclusive lower edge");
        assert_eq!(wrong[0], q - window + 1, "row {row}: strict lower edge");
        affected += 1;
    }
    assert_eq!(
        affected, block,
        "every query row is affected at the shipped widths"
    );

    // The fixture, at the geometry the goldens were vendored at: NO row changes.
    let (f_ctx, f_block, f_win) = (12i64, 4usize, 13i64);
    let f_kv = (f_ctx + f_block as i64) as usize;
    for row in 0..f_block as i64 {
        assert_eq!(
            attended(row, f_kv, f_win, false),
            attended(row, f_kv, f_win, true),
            "fixture row {row}: the two edges must AGREE here — that is the blindness being \
             recorded, so if this ever differs the fixture gained the power and this test is the \
             thing to delete"
        );
    }
}
