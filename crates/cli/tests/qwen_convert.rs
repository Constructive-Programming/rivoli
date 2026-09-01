//! **`convert_qwen` against a synthesized source, and the byte budgets its artifact costs.**
//! S4's converter gate, built on the `drafter_convert.rs` exemplar.
//!
//! Two things are gated here, and they are different claims:
//!
//! 1. **The census closure is exhaustive, and each of its edges reddens.** Every name in the source
//!    index is converted, passed through, or named-excluded; an unknown family, a family that
//!    partly vanished, and a `ple` tensor on the wrong layer are each a hard error — the last even
//!    though its pattern and count are both fine (trap T12). Each is planted below rather than
//!    argued, and the refusal wording is pinned so a guard cannot be deleted without a red test.
//! 2. **Every per-token byte budget is derived from the shipped config and labelled by dtype.**
//!    Under P5 bytes/token is the currency, and the recorded scar is a figure quoted at one dtype
//!    and spent at another — so each row states both columns.
//!
//! The third claim of this stage — that the config's head counts and the census's shard headers are
//! two independent derivations of the same shapes — lives in
//! `crates/artifact/tests/qwen_names.rs`, beside the census it scores. It needs no converter
//! binary, and it is the cell that reddens on a wrong reading of a head count.
//!
//! **The real checkpoint is NOT on this box.** 172.76 GiB has not been downloaded, so every arm
//! here runs against a synthesized index built from the census: 152,089 names, no weights, no
//! shards. That reaches the converter's last pre-weight step, which is what makes the POSITIVE arm
//! meaningful — a run that died earlier would name that check instead. What it cannot do is score a
//! real conversion; the `RIVOLI_QWEN_CKPT_REQUIRED` cell is the mechanism for that, proven in all
//! three states here, and the live run is **OWED** (the plan doc's worklog records it).
//!
//! No GPU, no device, no network.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

// `parse_config` and `fnv1a` are spelled in full at their call sites rather than imported: with
// them, this run of `use` lines was token-identical to `qwen_names.rs`'s and jscpd reported the
// pair, which is the gate asking for two files' heads not to be one file's twice. `collapse` and
// `ngram_hash` left with the derived-widths cell that moved to that file.
use rivoli_artifact::census::qwen::{Census, Family, Role};
use rivoli_artifact::qwen_config::{QwenConfig, QwenTextConfig};
use std::path::{Path, PathBuf};

mod common;

const BIN: common::ConvertBin = common::ConvertBin {
    exe: env!("CARGO_BIN_EXE_convert_qwen"),
    tool: "convert_qwen",
};

/// The vendored config — the same 71 KB `schema.rs` and `qwen_names.rs` pin by hash.
const CONFIG: &str = include_str!("../../../docs/measurement/qwen-reference/config.json");

/// Where the FP8 checkpoint is expected to land on this box, following its siblings
/// (`/swarm/storage/ai/rivoli/kimi-k3-src`). Overridable by `RIVOLI_QWEN_CKPT`; see
/// [`the_live_checkpoint_comparison_runs_in_all_three_required_states`].
const CKPT_DIR: &str = "/swarm/storage/ai/rivoli/qwen38-flash-next-fp8";

/// **Two env names, one flat namespace** — `RIVOLI_*` is flat across every binary here and
/// squatting one has already broken a test its branch never touched, so both were grepped for
/// first (2026-08-31: neither existed).
///
/// `RIVOLI_QWEN_CKPT_REQUIRED` follows `RIVOLI_DRAFTER_CKPT_REQUIRED`: an absent checkpoint becomes
/// a failure rather than a silent pass, because a check whose examined-count can reach zero is not
/// a check. `RIVOLI_QWEN_CKPT` exists so the required mode can run **in its required state** on a
/// box without the real source — a required mode never run in that state is a mechanism, not a
/// gate.
const REQUIRED: &str = "RIVOLI_QWEN_CKPT_REQUIRED";
const CKPT_OVERRIDE: &str = "RIVOLI_QWEN_CKPT";

/// One shard name for the synthesized index. It does not exist on disk, and that is the point:
/// every run below stops when the converter tries to open it, which is the step AFTER every census
/// check.
const SHARD: &str = "model-00001-of-00001.safetensors";

/// The vendored census and the vendored config, as one value — they are always wanted together
/// here, and the pair is also what keeps this file's two loaders from being a token-for-token twin
/// of `qwen_names.rs`'s (jscpd reported that pair on the first build).
fn pinned() -> (QwenConfig, Census) {
    (
        rivoli_artifact::schema::parse_config(CONFIG)
            .expect("the shipped config must load and validate"),
        Census::load().expect("the vendored qwen census must parse"),
    )
}

// ------------------------------------------------------------------------------------------
// The synthesized index: 152,089 names from the census, no weights.
// ------------------------------------------------------------------------------------------

/// The legal `{L}` values for one pattern, from the CONFIG.
fn layer_domain(t: &QwenTextConfig, pattern: &str) -> Vec<usize> {
    if pattern.starts_with("mtp.") {
        return (0..t.mtp_num_hidden_layers).collect();
    }
    if pattern.contains(".linear_attn.") {
        return (0..t.n_layers)
            .filter(|&l| !t.layer_is_sparse_attention(l).unwrap())
            .collect();
    }
    if pattern.contains(".self_attn.") {
        return (0..t.n_layers)
            .filter(|&l| t.layer_is_sparse_attention(l).unwrap())
            .collect();
    }
    if pattern.contains(".ple.") {
        return vec![t.ple_host_layer().unwrap()];
    }
    (0..t.n_layers).collect()
}

/// One family's real tensor names, expanded from its pattern. **The expansion self-checks against
/// the census's own count**, which is what makes it a fixture rather than a guess: `{L}` x `{E}`
/// over 48 layers and 512 experts must come out at exactly 24,576, and a wrong layer domain (a
/// `self_attn` family expanded over all 48) fails immediately.
fn expand(t: &QwenTextConfig, f: &Family) -> Vec<String> {
    let mut names = vec![f.pattern.clone()];
    for (ph, domain) in [
        ("{L}", layer_domain(t, &f.pattern)),
        ("{E}", (0..t.num_experts).collect()),
        ("{S}", (0..t.split_ngram_parts).collect()),
        // Vision block depth: `vision_config` is deliberately unbound by this port's schema, so
        // the count comes from the census row that is excluded anyway.
        ("{B}", (0..usize::try_from(f.count).unwrap()).collect()),
    ] {
        if !f.pattern.contains(ph) {
            continue;
        }
        names = names
            .iter()
            .flat_map(|n| {
                domain
                    .iter()
                    .map(move |i| n.replacen(ph, &i.to_string(), 1))
            })
            .collect();
    }
    assert_eq!(
        names.len() as u64,
        f.count,
        "{}: the expansion must reproduce the census's own count",
        f.pattern
    );
    names
}

/// Every name the pinned revision's index declares, in census order.
fn all_names(t: &QwenTextConfig, c: &Census) -> Vec<String> {
    c.families().iter().flat_map(|f| expand(t, f)).collect()
}

/// A source directory the converter can get all the way through — minus the weights.
///
/// `mutate` gets the index document before it is written, which is how each plant below is
/// planted: one edge of the census closure at a time.
fn source(tag: &str, mutate: impl Fn(&mut serde_json::Value)) -> (PathBuf, PathBuf, PathBuf) {
    let (root, src, out) = common::scratch_src_out(tag);
    std::fs::create_dir_all(&src).expect("create src");
    std::fs::create_dir_all(&out).expect("create out");
    std::fs::write(src.join("config.json"), CONFIG).expect("write config");
    let template = "{%- for m in messages %}{{ m.content }}{%- endfor %}";
    common::write_aux(
        &src,
        &[
            ("tokenizer.json", r#"{"model":{"type":"BPE"}}"#),
            (
                "tokenizer_config.json",
                &serde_json::json!({
                    "tokenizer_class": "Qwen2Tokenizer",
                    "chat_template": template,
                })
                .to_string(),
            ),
            (
                "generation_config.json",
                r#"{"eos_token_id":[248046,248044]}"#,
            ),
            // A trailing newline on the file home and none on the embedded copy is a DELIBERATE
            // superset, not a photograph of the source: track D measured the two homes exactly
            // byte-identical (8,952 B, neither carrying a trailing newline), so a fixture that
            // differs by exactly what the converter trims exercises the trim as well as the
            // comparison, and the real source is the easier case.
            ("chat_template.jinja", &format!("{template}\n")),
        ],
    );
    let (cfg, c) = pinned();
    let map: serde_json::Map<String, serde_json::Value> = all_names(&cfg.text, &c)
        .into_iter()
        .map(|n| (n, serde_json::json!(SHARD)))
        .collect();
    let total: u64 = c.families().iter().map(|f| f.total_bytes).sum();
    let mut index = serde_json::json!({
        "metadata": { "total_size": total },
        "weight_map": map,
    });
    mutate(&mut index);
    std::fs::write(src.join("model.safetensors.index.json"), index.to_string())
        .expect("write the synthesized index");
    (root, src, out)
}

/// **The positive arm: every check before the weights passes, and the run dies on the absent
/// shard.** That stop is what makes it meaningful — "the synthesized index satisfies the census
/// closure" is not observable from a refusal test unless the run gets PAST the closure, and the
/// only evidence is WHICH error it dies on. `expect_refusal` fails on a refusal it was not asked
/// for, so this cannot pass for the wrong reason.
#[test]
fn the_synthesized_index_passes_every_check_before_the_weights() {
    let (root, src, out) = source("qwen-convert-ok", |_| {});
    let run = BIN.at(&src, &out).invoke(&[]);
    let log = String::from_utf8_lossy(&run.stderr).to_string();
    common::expect_refusal(&run, SHARD);
    // **The census line, arm by arm** — the one line a conversion is cited by, and the reason it
    // is pinned here rather than eyeballed: a grand total balances while a class vanishes, so
    // every arm carries its own tensor count and byte total and every one of them is asserted.
    for arm in [
        "census 152089 tensors / 185502232570 B",
        "of which v1 148655/181906343962",
        "routed-weight 73728/120795955200",
        "routed-scale 73728/14745600",
        "ngram-shard 128/51200245760",
        "ngram-scale 1/2",
        "hash-param 3/280",
        "resident 1067/9895397120",
        "excluded-mtp 3101/2698026496",
        "excluded-vision 333/897862112",
    ] {
        assert!(
            log.contains(arm),
            "the census line must name {arm:?}; got:\n{log}"
        );
    }
    common::clean(&root);
}

/// **A family the census does not know is a hard error** — the edge that makes an upstream
/// revision which ADDS a tensor loud instead of silently dropped.
///
/// RED OBSERVED (plant P7', the pattern lookup turned into `let Some(&i) = … else { continue }`):
/// `expected a refusal naming "which is no family in docs/measurement/qwen-reference", got status
/// Some(1)`, followed by the census line and `Error: open "…/model-00001-of-00001.safetensors"` —
/// the run walked PAST an unclassified family and died on the weights, which on a real checkpoint
/// is a conversion that would have proceeded.
#[test]
fn a_tensor_family_the_census_does_not_know_is_refused() {
    let (root, src, out) = source("qwen-convert-newfam", |index| {
        index["weight_map"]["model.language_model.layers.0.mlp.experts.0.w4_proj.weight"] =
            serde_json::json!(SHARD);
    });
    BIN.at(&src, &out)
        .refuses(&[], "which is no family in docs/measurement/qwen-reference");
    common::clean(&root);
}

/// **A family that partly vanished is a hard error, and a grand total is exactly what would NOT
/// have caught it.** One expert weight removed out of 73,728 moves the byte total by 1,638,400 in
/// 185,502,232,570 — 0.00088% — and the per-pattern count check names it exactly.
///
/// RED OBSERVED (plant P14, the per-pattern check relaxed from `got == f.count` to `got <=`):
/// `expected a refusal naming "appears 24575 times in the source index, but the census records
/// 24576", got status Some(1)` — the run proceeded to the weights a tensor short.
#[test]
fn a_family_that_partly_vanished_is_refused_by_count() {
    let (root, src, out) = source("qwen-convert-vanish", |index| {
        index["weight_map"]
            .as_object_mut()
            .expect("weight_map")
            .remove("model.language_model.layers.17.mlp.experts.301.up_proj.weight")
            .expect("the name must have been in the synthesized index");
    });
    BIN.at(&src, &out).refuses(
        &[],
        "appears 24575 times in the source index, but the census records 24576",
    );
    common::clean(&root);
}

/// **Trap T12: a `ple` tensor on layer 2 rather than layer 1.** `ple_layer_ids` is `[2]` and
/// ONE-INDEXED, so the host is `layer_idx` 1; a converter reading the id at face value looks for
/// `layers.2.ple.*` — a name whose PATTERN is a real family and whose per-pattern COUNT still
/// balances, so neither the closure's first edge nor its third sees it. Only the index bound does,
/// and only because it comes from the config.
///
/// RED OBSERVED (plant P13, the layer-family bound forced to accept any layer): `expected a refusal
/// naming "which is not the ONE-INDEXED ple_layer_ids host", got status Some(1)` — and the SAME
/// plant reddens the sibling cell below with ITS own fragment, which is what says the two families'
/// domains are checked separately rather than by one predicate that happens to cover both.
#[test]
fn a_ple_tensor_on_the_wrong_layer_is_refused() {
    // The count comes from the CENSUS (128 shards plus 10 single tensors), not from a literal: a
    // remembered number here would go stale the day a PLE family is added, and this assertion's
    // whole job is to prove the mutation moved every one of them.
    let want: u64 = pinned()
        .1
        .families()
        .iter()
        .filter(|f| f.pattern.contains(".ple."))
        .map(|f| f.count)
        .sum();
    let (root, src, out) = source("qwen-convert-plelayer", |index| {
        let map = index["weight_map"].as_object_mut().expect("weight_map");
        let moved: Vec<String> = map
            .keys()
            .filter(|k| k.contains(".layers.1.ple."))
            .cloned()
            .collect();
        assert_eq!(
            moved.len() as u64,
            want,
            "every PLE tensor on the host layer must move"
        );
        for name in moved {
            map.remove(&name).expect("present");
            map.insert(
                name.replace(".layers.1.ple.", ".layers.2.ple."),
                serde_json::json!(SHARD),
            );
        }
    });
    BIN.at(&src, &out)
        .refuses(&[], "which is not the ONE-INDEXED ple_layer_ids host");
    common::clean(&root);
}

/// **A `self_attn` tensor on a GatedDeltaNet layer** — the same shape of defect as T12, on the
/// other partition. Layer 4 is `linear_attention`; layer 3 is the QSA layer next to it.
#[test]
fn an_attention_tensor_on_the_wrong_family_of_layer_is_refused() {
    let (root, src, out) = source("qwen-convert-wrongfam", |index| {
        let map = index["weight_map"].as_object_mut().expect("weight_map");
        let from = "model.language_model.layers.3.self_attn.q_proj.weight";
        map.remove(from).expect("layer 3 is a QSA layer");
        map.insert(
            "model.language_model.layers.4.self_attn.q_proj.weight".into(),
            serde_json::json!(SHARD),
        );
    });
    BIN.at(&src, &out).refuses(
        &[],
        "which is not a full_attention-spelled (indexer-running) layer",
    );
    common::clean(&root);
}

/// The index's own `metadata.total_size` must be what the census accounts for. Every family count
/// matched, so a disagreement here is a bytes-per-tensor change rather than a missing family —
/// which is the reading the refusal states.
///
/// RED OBSERVED (plant P17, the confrontation short-circuited): `expected a refusal naming "declares
/// metadata.total_size 185502232569", got status Some(1)`.
#[test]
fn a_declared_total_size_that_disagrees_with_the_census_is_refused() {
    let (root, src, out) = source("qwen-convert-total", |index| {
        index["metadata"]["total_size"] = serde_json::json!(185_502_232_569u64);
    });
    BIN.at(&src, &out)
        .refuses(&[], "declares metadata.total_size 185502232569");
    common::clean(&root);
}

/// The layer range is refused, not clamped: `--to 999` silently converting 48 layers would look
/// like it did what was asked.
///
/// RED OBSERVED (plant P18, the range ensure short-circuited): `expected a refusal naming "layer
/// range [0, 999) is not inside this model's [0, 48)", got status Some(1)`.
#[test]
fn a_layer_range_outside_the_model_is_refused() {
    let (root, src, out) = source("qwen-convert-range", |_| {});
    BIN.at(&src, &out).refuses(
        &["--to", "999"],
        "layer range [0, 999) is not inside this model's [0, 48)",
    );
    BIN.at(&src, &out)
        .refuses(&["--from", "12", "--to", "12"], "layer range [12, 12)");
    common::clean(&root);
}

// ------------------------------------------------------------------------------------------
// The aux list, and the chat template's two homes.
// ------------------------------------------------------------------------------------------

/// **Each of the four aux files is refused BY NAME, before any shard is opened.** The K3 scar is
/// an aux list naming a file the checkpoint does not ship, which refused the real source at the
/// LAST step of a 1.42 TiB run; `require_aux` turned that into a milliseconds-long refusal, and
/// this cell says all four names are in it.
///
/// RED OBSERVED (plant P16, `chat_template.jinja` dropped from `AUX`): `expected a refusal naming
/// "chat_template.jinja is missing from", got status Some(1)` — the exact K3 shape, one name short
/// of the list the checkpoint actually ships.
#[test]
fn every_missing_aux_file_is_refused_by_name() {
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ] {
        let (root, src, out) = source(&format!("qwen-convert-aux-{name}"), |_| {});
        std::fs::remove_file(src.join(name)).expect("remove one aux file");
        BIN.at(&src, &out)
            .refuses(&[], &format!("{name} is missing from"));
        common::clean(&root);
    }
}

/// **The chat template's two homes must agree, and the converter is where they are compared.**
/// GLM's scar is a template that lived only in the fp8 SOURCE and drifted to another family's role
/// framing for months once hand-ported. This checkpoint states it TWICE and the two agree
/// byte-for-byte — track D measured 8,952 B on each, neither with a trailing newline — which is
/// what lets this port copy either one instead of hand-porting, so it is a property worth refusing
/// on rather than recording. The converter still trims trailing newlines from both sides, and the
/// fixture below feeds it one on one side only: a re-export that adds one is a formatting artefact,
/// and the trim is what keeps the refusal reserved for a real divergence.
///
/// RED OBSERVED (plant P15, the comparison short-circuited): `expected a refusal naming "are not the
/// same template", got status Some(1)` — the converter silently copying a `chat_template.jinja` that
/// disagrees with the tokenizer config's own copy.
#[test]
fn a_chat_template_that_disagrees_between_its_two_homes_is_refused() {
    // Divergent: the embedded copy carries another family's role framing.
    let (root, src, out) = source("qwen-convert-tmpl", |_| {});
    let doc = serde_json::json!({
        "tokenizer_class": "Qwen2Tokenizer",
        "chat_template": "<|user|>\n{{ messages[0].content }}",
    });
    std::fs::write(src.join("tokenizer_config.json"), doc.to_string()).expect("write");
    BIN.at(&src, &out).refuses(&[], "are not the same template");
    common::clean(&root);

    // Absent second home: the port would then have to CHOOSE which one is authoritative, and the
    // refusal is what makes it choose rather than defaulting to the file.
    let (root, src, out) = source("qwen-convert-tmpl-missing", |_| {});
    std::fs::write(
        src.join("tokenizer_config.json"),
        r#"{"tokenizer_class":"Qwen2Tokenizer"}"#,
    )
    .expect("write");
    BIN.at(&src, &out)
        .refuses(&[], "has no string `chat_template` key");
    common::clean(&root);
}

/// Writing into the source directory is refused before anything is read: the writer maps the
/// source while it writes, and overwriting a mapped shard is a SIGBUS rather than an error.
#[test]
fn converting_into_the_source_directory_is_refused() {
    let (root, src, _) = source("qwen-convert-selfwrite", |_| {});
    BIN.at(&src, &src)
        .refuses(&[], "resolve to the same directory");
    common::clean(&root);
}

// ------------------------------------------------------------------------------------------
// P5: the per-token byte budgets, each labelled by dtype.
// ------------------------------------------------------------------------------------------

/// **Every budget derived from the shipped config, and every one stated at BOTH dtypes.**
///
/// The recorded scar is a figure quoted at one dtype and spent at another — the drafter's 20 KiB
/// KV cache, which this engine would allocate at 40 KiB. So each row below names the source dtype
/// and the f32 column beside it, and none of the numbers is quoted from prose.
///
/// KiB and MiB, not kB and MB: these are budgets against a GTT allocation and the exactness is the
/// point — 24,576 B *is* 24.0 KiB with no rounding.
#[test]
fn the_per_token_byte_budgets_follow_from_the_shipped_config() {
    let (cfg, _) = pinned();
    let t = &cfg.text;
    const FP8: usize = 1;
    const BF16: usize = 2;
    const F32: usize = 4;

    // **The headline: what a token costs in streamed expert weight.** 10 routed experts x 3
    // projections x 640 x 2560, on all 48 layers. This is the P5 currency for this model and it is
    // an fp8 figure — the same tensors at bf16 would be twice it, which is why the dtype is in the
    // name of the assertion rather than in a comment.
    let routed = t.top_k * 3 * t.moe_inter * t.hidden * t.n_layers;
    assert_eq!(
        routed * FP8,
        2_359_296_000,
        "routed stream at the checkpoint's fp8: 2,359,296,000 B/token"
    );
    let gib = routed as f64 * FP8 as f64 / f64::from(1u32 << 30);
    assert!(
        (2.197..2.198).contains(&gib),
        "the routed stream is {gib:.4} GiB/token at fp8"
    );
    assert_eq!(
        routed * BF16,
        4_718_592_000,
        "the same tensors if the artifact ever widened them — twice the stream, so the fp8 figure \
         is a property of the SOURCE's quantization and not of this engine"
    );
    // The scale grids ride along with the experts, and the converter WIDENS them: one f32 per
    // 128x128 tile, 100 tiles per projection.
    let grids = t.top_k * 3 * 100 * t.n_layers;
    assert_eq!(
        (grids * BF16, grids * F32),
        (288_000, 576_000),
        "routed scale grids per token: 288,000 B as the checkpoint stores them, 576,000 B as the \
         artifact writes them"
    );

    // QSA KV cache, over the 12 indexer-running layers ONLY — derived from `layer_types`, never
    // from 48 / 4, because those two are exactly what `validate_layer_types` refuses to assume
    // agree.
    let qsa_layers = t.sparse_attention_layers();
    let kv = 2 * t.num_key_value_heads * t.head_dim * qsa_layers;
    assert_eq!(
        (kv * BF16, kv * F32),
        (24 * 1024, 48 * 1024),
        "QSA KV: 24,576 B = 24.0 KiB/token at bf16, 49,152 B = 48.0 KiB as f32"
    );
    // The indexer's own cache: ONE un-normed, un-roped 128-dim key per token per QSA layer
    // (`qwen-architecture.md` §3), pooled per 4-token block only at score time.
    let idx = t.indexer_kv_heads * t.indexer_head_dim * qsa_layers;
    assert_eq!(
        (idx * BF16, idx * F32),
        (3 * 1024, 6 * 1024),
        "indexer keys: 3,072 B = 3.0 KiB/token at bf16, 6,144 B as f32"
    );

    // The n-gram gather: 16 heads x 160 dims, one row each, per token. Read at fp8 straight out of
    // the table and dequantized through ONE per-tensor scale.
    let gather = t.ngram_heads() * t.ngram_head_dim().expect("a head dim");
    assert_eq!(
        (gather * FP8, gather * F32),
        (2_560, 10_240),
        "n-gram gather: 2,560 B/token off disk, 10,240 B once dequantized"
    );

    // GDN recurrent state — per SEQUENCE, not per token, and the distinction is the reason it is
    // in this test: 108 MiB that does not scale with context is a different budget line from
    // 24 KiB that does.
    let state = t.linear_num_value_heads
        * t.linear_key_head_dim
        * t.linear_value_head_dim
        * (t.n_layers - qsa_layers);
    assert_eq!(
        (state * BF16, state * F32),
        (54 * 1024 * 1024, 108 * 1024 * 1024),
        "GDN state: 54.0 MiB at bf16, 108.0 MiB at the config's own `mamba_ssm_dtype` float32 — \
         per sequence, not per token"
    );
}

/// **What the artifact weighs, and the ONE thing the converter adds to the source.** Every weight
/// byte is copied; the only new bytes are a widened copy of each `weight_scale_inv` grid, so the
/// artifact is the v1 source plus exactly that much more — stating it as the DELTA is what makes it
/// checkable, where a bare total would be a number nobody can re-derive.
#[test]
fn the_artifact_is_the_v1_source_plus_one_widened_copy_of_the_scale_grids() {
    let (_, c) = pinned();
    let arm = |role: Role| c.role_totals(role).1;
    let (weights, scales) = (arm(Role::RoutedWeight), arm(Role::RoutedScale));
    let resident = arm(Role::Resident) + arm(Role::HashParam) + arm(Role::NgramScale);
    let ngram = arm(Role::NgramShard);
    assert_eq!(
        (weights, scales, ngram, resident),
        (120_795_955_200, 14_745_600, 51_200_245_760, 9_895_397_402),
        "the four arms the artifact is written as, in bytes"
    );
    let v1: u64 = c
        .families()
        .iter()
        .filter(|f| !f.role.is_excluded())
        .map(|f| f.total_bytes)
        .sum();
    // The widened grids are the delta, spelled as the delta.
    assert_eq!(
        weights + scales * 2 + ngram + resident,
        v1 + scales,
        "the artifact is the v1 source plus one more copy of the scale grids"
    );
    assert_eq!(v1 + scales, 181_921_089_562, "the artifact's total bytes");
    // 112.5 GiB of routed expert on a box with ~124 GiB usable, so P1 binds even before the
    // n-gram table: nothing here is resident-able and the whole design is the stream.
    let routed_gib = weights as f64 / f64::from(1u32 << 30);
    assert!(
        (112.4..112.6).contains(&routed_gib),
        "the routed expert set is {routed_gib:.4} GiB"
    );
    let resident_gib = resident as f64 / f64::from(1u32 << 30);
    assert!(
        (9.21..9.22).contains(&resident_gib),
        "the resident set is {resident_gib:.4} GiB, which is what a pin has to hold before a \
         single expert streams"
    );
}

// ------------------------------------------------------------------------------------------
// The live-checkpoint comparison, and its required mode in all three states.
// ------------------------------------------------------------------------------------------

/// Confront a LIVE checkpoint with the vendored census: its `config.json` bytes against the TSV's
/// own FNV-1a pin, and its index against the exhaustive closure. Returns the tensors examined.
///
/// **`required` is a parameter rather than an env read**, so all three of the gate's states can be
/// exercised by direct call — and the env read lives in exactly one place, at the test below.
fn confront_live(dir: &Path, required: bool) -> anyhow::Result<usize> {
    use anyhow::{Context, ensure};
    if !dir.join("model.safetensors.index.json").is_file() {
        ensure!(
            !required,
            "{REQUIRED} is set but {} holds no {} — the live comparison examined NOTHING, and a \
             check whose examined-count can reach zero is not a check",
            dir.display(),
            "model.safetensors.index.json"
        );
        return Ok(0);
    }
    let cfg_bytes = std::fs::read(dir.join("config.json"))
        .with_context(|| format!("read {}/config.json", dir.display()))?;
    let want = rivoli_artifact::census::qwen::FAMILIES
        .header_token("config.json", "fnv1a64 ")
        .context("the TSV's config.json pin")?;
    ensure!(
        format!("{:016x}", rivoli_core::hash::fnv1a(&cfg_bytes)) == want,
        "{}/config.json hashes {:016x}, not the pinned {want} — this is not the revision the \
         census was taken from",
        dir.display(),
        rivoli_core::hash::fnv1a(&cfg_bytes)
    );
    let cfg: QwenConfig = rivoli_artifact::schema::parse_config(std::str::from_utf8(&cfg_bytes)?)?;
    let doc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
    let map = doc["weight_map"].as_object().context("no weight_map")?;
    let names: Vec<&str> = map.keys().map(String::as_str).collect();
    Census::load()?.confront_index(&cfg.text, &names)?;
    Ok(names.len())
}

/// **The required mode, in all three of its states — including the one people skip.**
///
/// 1. **variable unset, checkpoint absent**: degrades to zero examinations, and this test asserts
///    the ZERO rather than letting a green stand for it.
/// 2. **variable set, checkpoint absent**: refuses, naming the directory.
/// 3. **variable set, checkpoint present**: the comparison actually RUNS and examines all 152,089
///    names — the state a `*_REQUIRED` gate is usually never run in, which makes it a mechanism
///    rather than a gate. The present checkpoint here is the SYNTHESIZED source, because 172.76
///    GiB is not on this box; `RIVOLI_QWEN_CKPT` exists for exactly that, and the live run against
///    the real source is OWED.
///
/// The env READ is one line and it is exercised last, against whatever this box is set to — with
/// the required arm asserted, so on a machine where the variable IS set and the checkpoint IS
/// present, this cell is the real comparison. That line asks for a NON-EMPTY value: `051a291`
/// records a CI `env:` expression yielding `''` arming a REQUIRED gate with nothing configured, so
/// "set to the empty string" is the fourth state and it means unarmed.
///
/// RED OBSERVED (plant P19, state 2's `ensure!(!required, …)` short-circuited): the
/// absent-checkpoint arm returned `Ok(0)` and this test reddened with `the required mode must refuse
/// an absent source: 0` — the required mode passing on zero examinations, which is the exact failure
/// the mode exists to stop.
#[test]
fn the_live_checkpoint_comparison_runs_in_all_three_required_states() {
    let missing = Path::new("/var/cache/rivoli/scratch/no-such-qwen-checkpoint");
    assert!(!missing.exists(), "the absent-path fixture must be absent");

    // State 1: unset and absent — zero examinations, said out loud.
    assert_eq!(
        confront_live(missing, false).expect("an absent checkpoint must degrade, not fail"),
        0,
        "with the variable unset the comparison examines nothing, and that is what it must report"
    );

    // State 2: set and absent — refuses, naming the directory.
    let err = format!(
        "{:#}",
        confront_live(missing, true).expect_err("the required mode must refuse an absent source")
    );
    assert!(
        err.contains(REQUIRED) && err.contains("no-such-qwen-checkpoint"),
        "the refusal must name both the variable and the path: {err}"
    );

    // State 3: set and PRESENT — the comparison runs. The synthesized source carries the vendored
    // config byte-for-byte, so its FNV-1a matches the TSV's pin and the closure runs over all
    // 152,089 names.
    let (root, src, _) = source("qwen-live-required", |_| {});
    let examined =
        confront_live(&src, true).expect("a present checkpoint must pass the whole closure");
    assert_eq!(
        examined, 152_089,
        "the required state must examine every tensor the index declares"
    );
    common::clean(&root);

    // And the env-reading line itself, once. Where the variable is set and the checkpoint is
    // there, this IS the live comparison; where it is not, the assertion above is what stands.
    //
    // **Value-checked, not existence-checked.** A CI `env:` entry whose expression yields `''`
    // still SETS the variable, which armed the CodeScene gate's REQUIRED mode with no secret
    // configured; `051a291` corrected exactly this in `codescene.rs::tool_absent`, and an
    // existence check here would have re-introduced it two commits later. Empty means unarmed.
    let required = std::env::var_os(REQUIRED).is_some_and(|v| !v.is_empty());
    let dir = std::env::var_os(CKPT_OVERRIDE)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(CKPT_DIR));
    match confront_live(&dir, required) {
        Ok(0) => assert!(
            !required,
            "{REQUIRED} is set and {} examined nothing",
            dir.display()
        ),
        Ok(n) => assert_eq!(
            n,
            152_089,
            "{} is present, so its index must be the pinned revision's",
            dir.display()
        ),
        Err(e) => panic!("{}: {e:#}", dir.display()),
    }
}
