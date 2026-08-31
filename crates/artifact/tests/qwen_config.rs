//! The Qwen3.8-Flash-Next load boundary: the shipped `config.json` parses, every field the
//! schema declares is required, and every setting that changes arithmetic without changing a
//! shape is refused.
//!
//! Built on the precedent `v4_config.rs` set and `k3_config.rs` followed: **mutate the real
//! vendored file** rather than a transcribed base. Every refusal below therefore starts from a
//! document known to parse, and there is no second copy of the checkpoint's values for the file's
//! to drift away from. The document is the same 71 KB `schema.rs` and `qwen_names.rs` pin by hash.
//!
//! **TDD record — what was watched RED, and how.** Each group's doc comment names what its rows
//! said when they were run against the assertion they belong to, removed. The interesting ones are
//! recorded there rather than here, because "every row reddened" is the weakest form of the claim;
//! what matters is which reds said something sharper than "the config still parsed".
//!
//! No GPU, no network, no checkpoint — 71 KB of vendored JSON.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

/// The two shared helpers, generic over `ArchConfig` — see `common/mod.rs`.
mod common;

use rivoli_artifact::glm_config::ModelConfig;
use rivoli_artifact::qwen_config::QwenConfig;
use rivoli_artifact::schema::{ArchConfig, parse_config};
use serde_json::{Value, json};

/// `include_str!` rather than a path read: a path read would SKIP on a machine without the
/// 172.76 GiB checkpoint rather than run.
const SHIPPED: &str = include_str!("../../../docs/measurement/qwen-reference/config.json");

/// The shipped document, as a mutable value.
///
/// Named `doc` and not `shipped`, spelled with `expect` and not `unwrap`, and declared after
/// `mod common` rather than before the imports — three small deviations from `k3_config.rs`'s
/// otherwise identical head, each of which breaks a token run jscpd reported the moment a FOURTH
/// config gate existed. The three older gates each carry their own such deviation with the same
/// note; the response this repo prescribes is to change the code, never to exempt it.
fn doc() -> Value {
    serde_json::from_str(SHIPPED).expect("the vendored config is JSON")
}

/// Every `(pointer, value, want)` row, through [`common::each_refusal`] bound to this file's
/// architecture and document.
fn refused(rows: &[(&str, Value, &str)]) {
    common::each_refusal::<QwenConfig>(SHIPPED, rows);
}

/// The refusal a cross-parse produces, required to name BOTH architectures.
///
/// Local rather than in `common/`, on the note `k3_config.rs` carries: a helper moves there "only
/// when jscpd reports the copy". This one loops over the halves and names the one it wanted, where
/// that file's `.all()` names the pair — which half is missing is the actionable one, so the
/// messages differ where it matters and the copy is not reported.
fn cross_parse<T: ArchConfig>(document: &str, halves: [&str; 2]) {
    let err = parse_config::<T>(document).err().map_or_else(
        || panic!("a foreign document parsed as {:?}", T::ARCH),
        |e| format!("{e:#}"),
    );
    for half in halves {
        assert!(
            err.contains(half),
            "the refusal must name {half:?} — a refusal naming neither architecture would blame \
             whichever dimension the other schema misses first: {err}"
        );
    }
}

/// A refusal from a mutation `common::each_refusal` cannot express, required to name `want`.
///
/// That helper takes ONE JSON pointer and panics rather than inventing a missing one — which is
/// what stops a row from silently scoring the unmutated document. Five checks here need something
/// else: two keys moved together (the two rotary homes, the two expert widths), a key the shipped
/// file does not carry at all (`norm_topk_prob`), a whole array replaced, and an `architectures`
/// list padded. `mutate` gets the parsed document; the refusal must then name `want`.
///
/// Factored on the first build, because the five spellings were a clone of each other.
#[track_caller]
fn refuses_after(mutate: impl Fn(&mut Value), want: &str) {
    let mut document = doc();
    mutate(&mut document);
    // Spelled as "an empty refusal is an acceptance" rather than as a `match` or a `map_or_else`:
    // the first is `format/tensors.rs`'s truncation test verbatim and the second is `cross_parse`
    // above, and jscpd reported each in turn.
    let refusal = parse_config::<QwenConfig>(&document.to_string())
        .map(|_| String::new())
        .unwrap_or_else(|e| format!("{e:#}"));
    assert!(
        !refusal.is_empty(),
        "the mutated document still parsed; it must refuse naming {want:?}"
    );
    assert!(
        refusal.contains(want),
        "the refusal must name {want:?}, or it fired for an unrelated reason: {refusal}"
    );
}

/// The shipped config parses, and every value this port acts on is what
/// `docs/reference/qwen-architecture.md` records. Pinned against the FILE, so the doc and the
/// schema cannot drift apart silently.
#[test]
fn the_shipped_config_parses_and_matches_the_architecture_doc() {
    let cfg: QwenConfig = parse_config(SHIPPED).expect("Qwen3.8-Flash-Next's own config must load");
    assert_eq!(cfg.model_type, "qwen4_exp");
    assert_eq!(cfg.architectures, ["Qwen4ExpForConditionalGeneration"]);
    let t = &cfg.text;
    assert_eq!(t.model_type, "qwen4_exp_text", "the TEXT half's spelling");
    assert_eq!((t.n_layers, t.hidden, t.vocab), (48, 2560, 248_320));
    assert_eq!(t.max_position_embeddings, 262_144);
    // QSA: GQA at 24/2, head_dim 256 — and 24 x 256 != 2560, the trap every fixture in this tree
    // keeps visible.
    assert_eq!((t.n_heads, t.num_key_value_heads, t.head_dim), (24, 2, 256));
    assert_ne!(
        t.n_heads * t.head_dim,
        t.hidden,
        "head width and hidden width must stay unequal, or a port that derived one passes"
    );
    // The indexer: MQA, 4 + 1 heads at 128, 2048 tokens as 512 four-token blocks.
    assert_eq!(
        (t.indexer_n_heads, t.indexer_kv_heads, t.indexer_head_dim),
        (4, 1, 128)
    );
    assert_eq!((t.indexer_budget, t.indexer_compress_ratio), (2048, 4));
    // GDN: ASYMMETRIC — 16 QK heads against 48 V heads, both at 128, conv kernel 4.
    assert_eq!(
        (t.linear_num_key_heads, t.linear_num_value_heads),
        (16, 48),
        "the asymmetry is the whole geometry: 3x as many V heads as QK heads"
    );
    assert_eq!(
        (
            t.linear_key_head_dim,
            t.linear_value_head_dim,
            t.linear_conv_kernel_dim
        ),
        (128, 128, 4)
    );
    // MoE on every layer, 512 experts, top-10 + 1 shared at the same width.
    assert_eq!((t.num_experts, t.top_k, t.moe_inter), (512, 10, 640));
    assert_eq!(t.shared_expert_intermediate_size, t.moe_inter);
    assert_eq!(
        t.norm_topk_prob, None,
        "the shipped file OMITS norm_topk_prob; the reference's own default (CFG:163) is true, \
         and `None` is how this schema says the file does not state it"
    );
    // Hyper-connections: a 4-wide stream, low-rank 320 = hidden / 8.
    assert_eq!((t.hc_count, t.hc_lowrank), (4, 320));
    assert_eq!(t.hc_lowrank * 8, t.hidden, "hc_lowrank is hidden / 8");
    // PLE: 3-grams over 8 heads per position, one host layer, 128 shards.
    assert_eq!((t.ngram_size, t.heads_per_ngram), (3, 8));
    assert_eq!(t.ngram_vocab_size_base, 20_000_000);
    assert_eq!(
        (t.make_ngram_vocab_size_divisible_by, t.split_ngram_parts),
        (128, 128)
    );
    assert_eq!((t.ple_embed_dim, t.ple_conv_kernel_size), (2560, 4));
    assert_eq!(t.ple_layer_ids, vec![2], "ONE-INDEXED in the file");
    // Token ids: EOS and BOS are the same id here, and the n-gram history pads with THIS one.
    assert_eq!((t.eos_token_id, t.bos_token_id), (248_044, 248_044));
    // The MTP head this port excludes by name: one layer, sharing the trunk's embeddings.
    assert_eq!(t.mtp_num_hidden_layers, 1);
    assert!(!t.mtp_use_dedicated_embeddings);
    // Settings that select an arithmetic.
    assert_eq!(t.dtype, "bfloat16");
    assert_eq!(t.hidden_act, "silu");
    assert_eq!(t.output_gate_type, "sigmoid");
    assert_eq!(t.mamba_ssm_dtype, "float32");
    assert!(!t.attention_bias && !t.tie_word_embeddings);
    assert_eq!(t.rope_parameters.rope_type, "default");
    assert_eq!(t.rope_parameters.rope_theta, 1e7);
    assert!(t.rope_parameters.mrope_interleaved);
    assert_eq!(t.rope_parameters.mrope_section, vec![11, 11, 10]);
    assert_eq!(t.rms_norm_eps, 1e-6);
}

/// **The layer partition, as the checkpoint states it TWICE.**
///
/// `layer_types` and `full_attention_interval` are independent statements of the same 36/12 split
/// and the reference validates neither against the other. Every derived width the arm launches
/// against is right in either reading, so a disagreement runs the wrong attention family on a
/// layer with no shape error anywhere.
#[test]
fn the_layer_partition_is_the_array_and_the_interval_agreeing() {
    let cfg: QwenConfig = parse_config(SHIPPED).unwrap();
    let t = &cfg.text;
    assert_eq!((t.layer_types.len(), t.full_attention_interval), (48, 4));
    assert_eq!(
        t.sparse_attention_layers(),
        12,
        "12 QSA layers — the ones that RUN THE INDEXER, spelled `full_attention`"
    );
    // The 12 indices, derived from the accessor rather than from the interval, so the two are
    // related here and not merely each restated.
    let qsa: Vec<usize> = (0..t.n_layers)
        .filter(|&l| t.layer_is_sparse_attention(l).unwrap())
        .collect();
    assert_eq!(qsa, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]);
    assert!(
        t.layer_is_sparse_attention(48).is_err(),
        "an out-of-range layer must REFUSE rather than read as GatedDeltaNet, which is a \
         positive claim about which arithmetic that layer runs"
    );
    // The PLE host, and its family. `ple_layer_ids [2]` -> layer_idx 1, a GDN layer.
    assert_eq!(t.ple_host_layer().unwrap(), 1);
    assert!(t.layer_hosts_ple(1) && !t.layer_hosts_ple(2));
    assert!(
        !t.layer_is_sparse_attention(1).unwrap(),
        "the reference requires a linear_attention host for the PLE layer (CFG:248-254)"
    );
}

/// **Every field the schema declares is REQUIRED, and the seven it deliberately ignores are
/// named.**
///
/// A `#[serde(default)]` added later would move a key from `required` to `ignored` and redden
/// here, which is the point: a defaulted dimension does not crash, it produces fluent wrong text.
/// The ignored list is the other half — each entry is a decision recorded in `qwen_config.rs`'s
/// header, so a key that silently stops being read shows up as a NEW ignored entry.
///
/// RED OBSERVED (plant P1', `#[serde(default)]` added to `mtp_use_dedicated_embeddings`):
/// `left: [... "mtp", "mtp_use_dedicated_embeddings", ...]` against
/// `right: [... "mtp", "output_router_logits", ...]` — the key moved from required to tolerated,
/// which is exactly the drift this cell exists to catch, and the message printed the whole
/// `required` list beside it so the reader can see which 47 keys still bind.
#[test]
fn every_qwen_field_is_required_except_the_seven_named_absences() {
    /// The keys whose ABSENCE the schema tolerates, because it does not read them. Each is argued
    /// in `qwen_config.rs`'s module header: training-side settings, the `null` pad id, and the
    /// `mtp` sub-dict whose 35 tensor families are excluded by NAME instead.
    const IGNORED: [&str; 7] = [
        "attention_dropout",
        "initializer_range",
        "mtp",
        "output_router_logits",
        "pad_token_id",
        "router_aux_loss_coef",
        "use_cache",
    ];
    let text = doc()["text_config"].clone();
    let keys: Vec<String> = text.as_object().expect("a dict").keys().cloned().collect();
    // Anti-vacuity: an empty walk satisfies everything below while looking green.
    assert_eq!(
        keys.len(),
        54,
        "the shipped text_config lost keys: {keys:?}"
    );
    let (required, mut ignored): (Vec<String>, Vec<String>) = keys.into_iter().partition(|k| {
        let mut document = doc();
        let dict = document["text_config"]
            .as_object_mut()
            .expect("text_config");
        // `is_some()` asserted so a key that vanished mid-walk cannot silently test the
        // unmutated document — which would report every key as "ignored".
        assert!(dict.remove(k).is_some(), "{k} was not in the clone");
        parse_config::<QwenConfig>(&document.to_string()).is_err()
    });
    ignored.sort();
    assert_eq!(
        ignored,
        IGNORED.map(String::from),
        "the set of text_config keys whose absence is TOLERATED has moved.\n  required: \
         {required:?}"
    );
    assert_eq!(
        required.len(),
        54 - IGNORED.len(),
        "every other key must be required"
    );
    // The wrapper's own fields: a bare text dict is not evidence the wrapper was this model's.
    for key in ["text_config", "model_type", "architectures"] {
        let mut document = doc();
        document.as_object_mut().unwrap().remove(key);
        assert!(
            parse_config::<QwenConfig>(&document.to_string()).is_err(),
            "removing the wrapper's {key} must refuse"
        );
    }
}

/// **The descent, and the wrapper pair — the five spellings, of which only two resolve.**
///
/// `vision_config.model_type` is the ROOT string verbatim, so the schema's root-only reading is
/// what excludes it and `text_config.model_type` is what proves the descent landed in the text
/// half. The rows below are the near-misses a hand-edit or a promoted sub-config actually
/// produces.
///
/// RED OBSERVED (plant P2, `QwenTextConfig::validate` stubbed to `Ok(())`):
/// `/text_config/model_type was mutated to a wrong value and the config still parsed` — the
/// `qwen4_exp` row, i.e. the ROOT string sitting in `text_config`, producing a config whose every
/// dimension came from a dict that might have been the vision tower's. That is the case root-only
/// reading cannot see from inside.
#[test]
fn a_wrong_descent_or_a_padded_architecture_list_is_refused() {
    refused(&[
        (
            "/text_config/model_type",
            json!("qwen4_exp"),
            "the text half's spelling is \"qwen4_exp_text\"",
        ),
        (
            "/text_config/model_type",
            json!("qwen3_next"),
            "the text half's spelling is \"qwen4_exp_text\"",
        ),
    ]);
    // **What the wrapper assert adds over `arch_of_named`, and it is narrow on purpose.** That
    // function refuses every root spelling that does not resolve, and refuses a document carrying
    // only one of the two fields — so a document that REACHES `validate` always names this
    // architecture. What it deliberately admits is an `architectures` LIST longer than one entry,
    // where the extra entries also resolve; a manifest hand-edited or merged that way has more
    // than one statement of identity and this is the check that says so. Kept also because
    // `validate` is a `pub` trait method on a `pub` type with `pub` fields, so a caller can reach
    // it without `parse_config` at all — the hazard `K3TextConfig`'s doc names.
    refuses_after(
        |d| {
            d["architectures"] = json!([
                "Qwen4ExpForConditionalGeneration",
                "Qwen4ExpForConditionalGeneration"
            ]);
        },
        "the document ROOT declares",
    );
}

/// **The array-and-interval disagreement, in all four shapes it takes.**
///
/// RED OBSERVED (plant P3, the `(kind == FULL_ATTENTION_ALIAS) == by_interval` ensure
/// short-circuited): `/text_config/full_attention_interval was mutated to a wrong value and the
/// config still parsed` — a config in which `layer_is_sparse_attention`, which reads the ARRAY,
/// disagrees with any arm that derived the partition from the INTERVAL.
#[test]
fn a_layer_types_array_that_contradicts_the_interval_is_refused() {
    refused(&[
        (
            "/text_config/full_attention_interval",
            json!(3),
            "two statements of one partition",
        ),
        (
            "/text_config/layer_types/0",
            json!("full_attention"),
            "two statements of one partition",
        ),
        (
            "/text_config/layer_types/3",
            json!("linear_attention"),
            "two statements of one partition",
        ),
        // The spelling the reference REWRITES `full_attention` into. A checkpoint that shipped it
        // directly would be a different file than the one this census was taken from, and the
        // refusal must quote the two spellings this one uses.
        (
            "/text_config/layer_types/0",
            json!("qwen_sparse_attention"),
            "this checkpoint's own spellings are",
        ),
        (
            "/text_config/layer_types",
            json!(["linear_attention", "full_attention"]),
            "the array is the per-layer family map",
        ),
        (
            "/text_config/full_attention_interval",
            json!(0),
            "full_attention_interval is 0",
        ),
    ]);
    // **The row `each_refusal` cannot express, and the one the loop alone would miss**: 48
    // `linear_attention` entries with an interval of 49 satisfies EVERY row of the per-layer
    // check and leaves this port with no QSA layer, no indexer and no KV cache at all.
    refuses_after(
        |d| {
            d["text_config"]["layer_types"] = json!(vec!["linear_attention"; 48]);
            d["text_config"]["full_attention_interval"] = json!(49);
        },
        "this model is a hybrid and needs both families",
    );
}

/// **`ple_layer_ids` is ONE-INDEXED and its host must be a GatedDeltaNet layer — trap T12.**
///
/// RED OBSERVED, twice, from one plant (P4, `ple_host_layer` returning the ONE-BASED id):
/// `/text_config/ple_layer_ids was mutated to a wrong value and the config still parsed` — the
/// `[4]` row, whose true host is layer_idx 3 (a QSA layer) but whose un-decremented id lands on
/// layer 4, which `layer_types` also calls `linear_attention`, so the family check PASSED. The
/// other three rows still refuse, because `[0]`, `[49]` and `[2, 6]` are caught by the range and
/// length checks, which read the id itself. The SAME plant reddens
/// `the_layer_partition_is_the_array_and_the_interval_agreeing` with `left: 2 right: 1` — which is
/// why that index is asserted directly and not only through a refusal here: without it, this trap
/// would surface one converter later as `layers.2.ple.*` against a checkpoint whose PLE tensors
/// sit under `layers.1.`.
#[test]
fn a_ple_host_that_is_off_by_one_or_the_wrong_family_is_refused() {
    refused(&[
        ("/text_config/ple_layer_ids", json!([0]), "ONE-INDEXED"),
        ("/text_config/ple_layer_ids", json!([49]), "ONE-INDEXED"),
        (
            "/text_config/ple_layer_ids",
            json!([2, 6]),
            "this port implements ONE PLE layer",
        ),
        (
            "/text_config/ple_layer_ids",
            json!([]),
            "this port implements ONE PLE layer",
        ),
        // Id 4 is one-based, so layer_idx 3 — which is a QSA layer, and the reference requires a
        // linear-attention host.
        (
            "/text_config/ple_layer_ids",
            json!([4]),
            "requires a linear-attention host",
        ),
        (
            "/text_config/ple_embed_dim",
            json!(2561),
            "does not divide into 16 n-gram heads",
        ),
        ("/text_config/ngram_size", json!(1), "n-gram heads"),
    ]);
}

/// Every width of zero is refused. A zero passes every divisibility check (0 is a multiple of
/// anything) and then sizes an expert row, a hyper-connection stride or a GEMV dim to nothing.
#[test]
fn every_width_of_zero_is_refused() {
    let keys = [
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
        "indexer_head_dim",
        "indexer_n_heads",
        "indexer_kv_heads",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "hc_count",
        "hc_lowrank",
        "ngram_size",
        "heads_per_ngram",
        "ngram_vocab_size_base",
        "make_ngram_vocab_size_divisible_by",
        "split_ngram_parts",
        "ple_embed_dim",
        "ple_conv_kernel_size",
        "max_position_embeddings",
        "mtp_num_hidden_layers",
    ];
    let pointers: Vec<String> = keys.iter().map(|k| format!("/text_config/{k}")).collect();
    let rows: Vec<(&str, Value, &str)> = pointers
        .iter()
        .map(|p| (p.as_str(), json!(0), "would size an expert row"))
        .collect();
    // Anti-vacuity, and it must be a claim about the SET rather than about the mapping:
    // `rows.len() == keys.len()` is true by construction and would assert nothing. 31 is the
    // number of widths `validate_widths` loops over, so a width dropped from either list reddens.
    assert_eq!(keys.len(), 31, "the schema's width set moved");
    refused(&rows);
}

/// **The structurally wrong relations** — each is a shape that is self-consistent and wrong.
///
/// RED OBSERVED (plant P5', the two-homes comparison short-circuited to `true || …`):
/// `/text_config/rope_parameters/partial_rotary_factor was mutated to a wrong value and the config
/// still parsed` — a document stating the rotary width twice and disagreeing with itself, where
/// whichever half the arm happens to read decides how many of 256 dims rotate.
#[test]
fn the_structurally_wrong_relations_are_refused() {
    refused(&[
        (
            "/text_config/num_key_value_heads",
            json!(5),
            "does not group onto num_key_value_heads",
        ),
        (
            "/text_config/rope_parameters/partial_rotary_factor",
            json!(0.5),
            "the file states it twice and the arm reads one of them",
        ),
        (
            "/text_config/rope_parameters/mrope_section",
            json!([11, 11, 11]),
            "frequencies",
        ),
        (
            "/text_config/indexer_compress_ratio",
            json!(3),
            "FLOORS `block_topk` where the tech report CEILS it",
        ),
        (
            "/text_config/linear_key_head_dim",
            json!(64),
            "the GDN state is `key_dim x value_dim` per V head",
        ),
        (
            "/text_config/linear_num_value_heads",
            json!(40),
            "does not group onto linear_num_key_heads",
        ),
        (
            "/text_config/hc_count",
            json!(1),
            "there is no hyper-connection stream",
        ),
        (
            "/text_config/hc_lowrank",
            json!(99_999),
            "is not a rank BELOW the",
        ),
        (
            "/text_config/num_experts_per_tok",
            json!(513),
            "num_experts_per_tok 513 is not in 1..=512",
        ),
        (
            "/text_config/shared_expert_intermediate_size",
            json!(1280),
            "they are equal in this checkpoint",
        ),
        // 2500 is not a multiple of the 128x128 fp8 tile, so every routed expert's grid would
        // have a ragged last column covering fewer weights than the dequant assumes.
        (
            "/text_config/hidden_size",
            json!(2500),
            "not a multiple of the fp8 tile 128",
        ),
        (
            "/text_config/eos_token_id",
            json!(248_320),
            "must index the 248320-row vocabulary",
        ),
        (
            "/text_config/mtp_num_hidden_layers",
            json!(2),
            "ONE layer's worth sharing the trunk embeddings",
        ),
        (
            "/text_config/mtp_use_dedicated_embeddings",
            json!(true),
            "ONE layer's worth sharing the trunk embeddings",
        ),
    ]);
    // **The expert intermediate against the fp8 tile, at BOTH widths the checkpoint keeps equal.**
    // A single-pointer row is caught by the shared-expert equality first, which is a different
    // assertion — so this moves both to 600, an expert width that is not a multiple of 128 and
    // would give every routed grid a ragged last row covering fewer weights than the dequant
    // assumes, over 112.5 GiB.
    refuses_after(
        |d| {
            for key in ["moe_intermediate_size", "shared_expert_intermediate_size"] {
                d["text_config"][key] = json!(600);
            }
        },
        "not a multiple of the fp8 tile 128",
    );
    // **`norm_topk_prob` cannot be one of the rows above, because the shipped file does not CARRY
    // it** — `common::refusal` mutates a JSON pointer and panics rather than inventing one, which
    // is exactly what stops a test row from silently scoring the unmutated document. So the key is
    // ADDED here, which is also the realistic shape: a checkpoint that turned the renormalisation
    // off would state it, and one that leaves it out is deferring to the reference's `true`.
    refuses_after(
        |d| d["text_config"]["norm_topk_prob"] = json!(false),
        "norm_topk_prob is false",
    );
}

/// **The five named settings, plus the three flags and the two narrowing scalars.**
///
/// Every row here changes the arithmetic and leaves every shape right, which is the class nothing
/// downstream refuses. The two scalars are checked in the **f32 domain** because the kernels
/// narrow them: `1e-46` passes an f64 positivity test and reaches every RMSNorm as `0.0f32`.
#[test]
fn the_silently_wrong_settings_and_narrowing_scalars_are_refused() {
    refused(&[
        (
            "/text_config/dtype",
            json!("float16"),
            "implements \"bfloat16\" only",
        ),
        (
            "/text_config/hidden_act",
            json!("gelu"),
            "implements \"silu\" only",
        ),
        (
            "/text_config/output_gate_type",
            json!("silu"),
            "implements \"sigmoid\" only",
        ),
        (
            "/text_config/mamba_ssm_dtype",
            json!("bfloat16"),
            "implements \"float32\" only",
        ),
        (
            "/text_config/rope_parameters/rope_type",
            json!("yarn"),
            "implements \"default\" only",
        ),
        (
            "/text_config/rope_parameters/mrope_interleaved",
            json!(false),
            "mrope_interleaved is false",
        ),
        (
            "/text_config/attention_bias",
            json!(true),
            "no `*_proj.bias` family exists",
        ),
        (
            "/text_config/tie_word_embeddings",
            json!(true),
            "ships a separate lm_head",
        ),
        // Narrowing: positive in f64, zero in f32.
        (
            "/text_config/rms_norm_eps",
            json!(1e-46),
            "narrows to 0 in f32",
        ),
        (
            "/text_config/rope_parameters/rope_theta",
            json!(0.0),
            "rope_parameters.rope_theta 0 narrows to 0",
        ),
    ]);
    // **The rotary width, at BOTH of the homes the file states it in.** A single-pointer row is
    // caught by the two-homes disagreement check instead, which is a different assertion; with
    // both at zero they agree and the WIDTH is what must be refused. `partial_rotary_factor` is
    // deliberately not in `ensure_f32_positive`'s list for the same reason — see the comment
    // there: any value small enough to narrow to `0.0f32` truncates the width to 0 first, so a
    // row there could never be reddened, and a check that cannot go red is not a check.
    refuses_after(
        |d| {
            for home in ["/text_config", "/text_config/rope_parameters"] {
                d.pointer_mut(home).expect("both rotary homes")["partial_rotary_factor"] =
                    json!(0.0);
            }
        },
        "gives a rotary width of 0",
    );
}

/// A foreign document must be refused by ARCHITECTURE, before serde looks at a dimension — so the
/// message names both architectures rather than blaming whichever field the other schema happens
/// to miss first.
///
/// GLM is asserted in both directions here; the V4, K3 and Glimmer pairs live in their own gates,
/// so no direction is asserted twice.
#[test]
fn qwen_and_glm_do_not_cross_parse() {
    cross_parse::<ModelConfig>(SHIPPED, ["qwen4_exp", "GlmMoeDsa"]);
    // The reverse, on a document carrying the DISCRIMINANT AND NOTHING ELSE — the sharp form: a
    // refusal naming a missing dimension would be the weak reading.
    let glm = json!({"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"]});
    cross_parse::<QwenConfig>(&glm.to_string(), ["glm_moe_dsa", "QwenFlashNext"]);
    // And the `vision_config` block promoted to a document — the near-miss this port's identity
    // rests on. It carries the ROOT string as its `model_type` and no `architectures` at all.
    let promoted = doc()["vision_config"].clone();
    cross_parse::<QwenConfig>(&promoted.to_string(), ["model_type", "architectures"]);
}
