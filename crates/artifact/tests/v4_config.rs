//! The DeepSeek-V4-Flash load boundary: the shipped `config.json` parses, every field it
//! declares is required, and every setting that changes arithmetic without changing a shape is
//! refused.
//!
//! Ported from `old:src/artifact/model.rs`'s `mod tests` V4 block (`wt/glimmer-s2` @ 6b7f496).
//! It lives in `tests/` rather than in a `#[cfg(test)] mod tests` inside `v4_config.rs`, on the
//! precedent `glimmer_config.rs`'s twin set: that file is already ~470 lines of schema and
//! argument, and the soft line cap's contract is that the next edit shrinks a file rather than
//! grows it. Nothing here needs private access — `parse_config` is the converter's own entry.
//!
//! > **PORT NOTE 2026-08-16. These parse the SHIPPED config directly, where the reference
//! > assembled a `V4_BASE` of 40 transcribed `(key, json)` pairs and pinned it against the real
//! > file in a separate test that SKIPS when the checkpoint is absent.** Glimmer's port had
//! > already made that move and its argument applies unchanged and more strongly here: mutating
//! > the real document is the stronger gate — every refusal below starts from a document known
//! > to parse — and a transcribed base is one more place for the file's values to drift out of.
//! > The reference itself records `V4_BASE` going stale as the hazard its pin existed for.
//! >
//! > What is LOST by the move is named rather than glossed: the reference's
//! > `every_v4_field_is_required` could tell "required" from "present" by deleting each
//! > `V4_BASE` row, and its `v4_base_matches_the_shipped_config` was a *structural* comparison
//! > that caught a key drifting in either direction. The first is reproduced below and made
//! > stronger — it is driven off the shipped document's own keys and asserted in BOTH
//! > directions, so a field that stops being required moves between two whole sets. The second
//! > is replaced by the vendoring itself: this file IS the checkpoint's bytes, copied
//! > 2026-08-16 from `/var/db/rivoli/deepseek-v4-flash-0731/config.json`, so there is no second
//! > transcription to disagree with it. Re-copy it when the checkpoint moves.
//!
//! The load boundary is the only place a foreign snapshot is inspected before its dimensions
//! reach a kernel, so every refusal here is one that would otherwise be an out-of-bounds panic
//! or, far more often in this model, fluent wrong text with no crash anywhere.
//!
//! No GPU, no network, no checkpoint — 1.9 KB of vendored JSON.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

// One braced `use`, where `glimmer_config.rs` lists four flat ones. Not style: jscpd
// normalizes identifiers, so the two files' four-line import runs were a 40-token clone the
// moment this file existed. `crates/cli/tests/glimmer_convert.rs` records reaching for the same
// fix for the same reason.
use rivoli_artifact::{
    glimmer_config::GlimmerConfig, glm_config::ModelConfig, k3_config::K3Config,
    schema::parse_config, v4_config::V4Config,
};
use serde_json::{Value, json};

/// The two shared helpers, generic over `ArchConfig` — see `common/mod.rs` for why they moved
/// out of `glimmer_config.rs` and what the move preserves.
mod common;

/// `include_str!` rather than a path read: a path read would SKIP on a machine without the
/// 146 GB checkpoint rather than run. 1.9 KB, copied verbatim 2026-08-16.
const V4_SHIPPED: &str = include_str!("../../../docs/measurement/v4-reference/config.json");

/// Muse Glimmer's shipped config: a `*ForConditionalGeneration` wrapper around a `text_config`,
/// so it shares nothing with V4's flat document but the refusal path.
///
/// GLM-5.2's own file is NOT vendored in this tree (`docs/measurement/glm-reference/` holds only
/// its anchor), so the GLM direction below is driven by a two-key document instead — which is
/// exactly the right instrument for it: the whole property under test is that the architecture
/// check fires BEFORE serde reads a dimension, so a document with nothing but a discriminant is
/// the one that isolates it. A vendored GLM config would additionally have failed on missing
/// V4 fields, which is the weaker reading this test exists to rule out.
const GLIMMER_SHIPPED: &str =
    include_str!("../../../docs/measurement/glimmer-reference/config.json");

/// Kimi-K3's shipped config — the OTHER MXFP4-expert checkpoint, so the V4↔K3 cross-parse
/// pair below is the one where "same expert format, different decode path" makes the
/// architecture check do real work.
const K3_SHIPPED: &str = include_str!("../../../docs/measurement/k3-reference/config.json");

fn shipped() -> Value {
    serde_json::from_str(V4_SHIPPED).unwrap()
}

/// [`common::each_refusal`] bound to this file's architecture and document. One wrapper rather
/// than a turbofish plus the constant at each of the three call sites — and only this one, since
/// `common::refusal` has no direct caller here and a wrapper with none is dead surface
/// `-D warnings` reports on sight.
fn each_refusal(rows: &[(&str, Value, &str)]) {
    common::each_refusal::<V4Config>(V4_SHIPPED, rows);
}

#[test]
fn the_shipped_config_parses_and_matches_the_doc() {
    let c: V4Config = parse_config(V4_SHIPPED).expect("V4-Flash's own config must load");
    // The values every doc comment in `v4_config.rs` quotes, read back off the real file.
    assert_eq!((c.n_layers, c.hidden, c.vocab), (43, 4096, 129_280));
    assert_eq!((c.n_heads, c.head_dim, c.qk_rope_head_dim), (64, 512, 64));
    assert_eq!((c.n_experts, c.top_k, c.moe_inter), (256, 6, 2048));
    assert_eq!((c.n_shared, c.n_hash_layers), (1, 3));
    assert_eq!((c.o_groups, c.o_lora_rank, c.q_lora_rank), (8, 1024, 1024));
    assert_eq!(
        (c.index_n_heads, c.index_head_dim, c.index_topk),
        (64, 128, 512)
    );
    assert_eq!((c.hc_mult, c.hc_sinkhorn_iters), (4, 20));
    assert_eq!((c.routed_scale, c.swiglu_limit), (1.5, 10.0));
    assert_eq!(c.sliding_window, 128);
    assert_eq!(c.rope_scaling.kind, "yarn");
    assert_eq!(c.rope_scaling.factor, 16.0);
    assert_eq!(c.rope_scaling.original_max_position_embeddings, 65_536);
    // 46 entries for 43 layers — the tail belongs to the mtp blocks, which is why
    // `validate_layer_tables` checks a floor and `compress_ratio` bounds on `n_layers`.
    assert_eq!(c.compress_ratios.len(), 46);
    assert!(
        c.compress_ratios.len() > c.n_layers,
        "the mtp tail is the reason compress_ratio() bounds on n_layers rather than on len()"
    );
}

/// The keys whose absence `V4Config` refuses — the schema's REQUIRED set.
///
/// A const rather than a literal inside the test: with both lists inline the body was 94 lines,
/// which is the shape the code-health gate refuses, and the two sets are the test's SUBJECT
/// rather than its scaffolding. Kept alphabetical so a diff to it reads as one line.
const REQUIRED: [&str; 33] = [
    "compress_ratios",
    "compress_rope_theta",
    "expert_dtype",
    "hc_eps",
    "hc_mult",
    "hc_sinkhorn_iters",
    "head_dim",
    "hidden_size",
    "index_head_dim",
    "index_n_heads",
    "index_topk",
    "max_position_embeddings",
    "moe_intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "num_attention_heads",
    "num_experts_per_tok",
    "num_hash_layers",
    "num_hidden_layers",
    "num_key_value_heads",
    "o_groups",
    "o_lora_rank",
    "q_lora_rank",
    "qk_rope_head_dim",
    "quantization_config",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "routed_scaling_factor",
    "scoring_func",
    "sliding_window",
    "swiglu_limit",
    "vocab_size",
];

/// The complement: the checkpoint's own keys this schema deliberately does not read.
///
/// `norm_topk_prob` is the one worth naming — `Gate.forward` renormalizes on
/// `score_func != "softmax"`, not on the flag, so carrying it would be carrying a value with no
/// effect. The `dspark_*` four belong to a speculative decode path this engine does not
/// implement.
const IGNORED: [&str; 17] = [
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "dspark_block_size",
    "dspark_markov_rank",
    "dspark_noise_token_id",
    "dspark_target_layer_ids",
    "eos_token_id",
    "hidden_act",
    "initializer_range",
    "norm_topk_prob",
    "num_nextn_predict_layers",
    "tie_word_embeddings",
    "topk_method",
    "torch_dtype",
    "transformers_version",
    "use_cache",
];

/// Partition the shipped document's own keys by whether removing one is refused.
///
/// **`architectures` and `model_type` are excluded by name.** They are the DISCRIMINANT, not
/// schema fields: removing one of them still resolves through the other, which is
/// `arch_of_named`'s documented behaviour and not a claim about `V4Config`. Excluding them keeps
/// the two sets a statement about dimensions.
fn partition_by_requiredness() -> (Vec<String>, Vec<String>) {
    let keys: Vec<String> = shipped()
        .as_object()
        .unwrap()
        .keys()
        .map(String::from)
        .filter(|k| k != "architectures" && k != "model_type")
        .collect();
    // Anti-vacuity: an empty walk would satisfy nothing below while looking green.
    assert!(
        keys.len() > 30,
        "the shipped config lost its keys: {keys:?}"
    );
    let (mut ignored, mut required) = (Vec::new(), Vec::new());
    for k in keys {
        let mut doc = shipped();
        doc.as_object_mut().unwrap().remove(&k);
        if parse_config::<V4Config>(&doc.to_string()).is_ok() {
            ignored.push(k);
        } else {
            required.push(k);
        }
    }
    required.sort();
    ignored.sort();
    (required, ignored)
}

/// **Every field the schema declares is REQUIRED — enforced, not just claimed.**
///
/// Driven off the SHIPPED document rather than a list of field names, and asserted in **both**
/// directions: deleting a key the schema needs must refuse, and deleting one it does not carry
/// must still parse. The sets are compared whole, so a field that stops being required moves
/// between them and reddens.
///
/// **What it does and does not catch.** The property is "removing this key is refused" — by
/// serde *or* by `validate`, which is the property that actually matters, since either way the
/// checkpoint is rejected. A `#[serde(default)]` added to a field whose default happens to fail
/// `validate` (`swiglu_limit`, `index_topk`, `routed_scale`, `hc_*`, and every width whose zero
/// trips a divisibility check) would therefore stay green here; those fields' own rows in
/// [`the_silently_wrong_settings_are_refused`] are what cover the value being wrong. What this
/// catches is the shape the module header warns about — a defaulted DIMENSION that parses,
/// reports zero, and sizes a launch.
#[test]
fn every_v4_field_is_required() {
    let (required, ignored) = partition_by_requiredness();
    assert_eq!(
        required,
        REQUIRED.map(String::from),
        "the set of keys whose absence is refused has moved"
    );
    assert_eq!(
        ignored,
        IGNORED.map(String::from),
        "a key this schema ignores has started being read, or the reverse"
    );
}

/// The settings that change arithmetic without changing a shape, so nothing downstream would
/// refuse them. Every row is a value the checkpoint could plausibly carry.
#[test]
fn the_silently_wrong_settings_are_refused() {
    each_refusal(&[
        // Router affinity. A wrong one picks plausible-but-wrong experts and never crashes.
        ("/scoring_func", json!("sigmoid"), "sqrtsoftplus"),
        // The routed scale, which multiplies every routed contribution while the shared expert
        // keeps working — degraded fluent text, not a crash.
        ("/routed_scaling_factor", json!(0.0), "must be positive"),
        ("/routed_scaling_factor", json!(-1.5), "must be positive"),
        // The indexer's selection width: at 0 every ratio-4 layer silently degrades to pure
        // sliding-window attention, and the `cat` that does it is perfectly legal.
        ("/index_topk", json!(0), "index_topk must be positive"),
        // The SwiGLU clamp, in the f32 domain the kernel works in. The OVERFLOW row is the
        // silent one: `as f32` saturates to +inf, which passes any bare `> 0.0` test, and
        // `fminf(gt, inf) == gt` makes the clamp a no-op. Both verified numerically 2026-08-05.
        ("/swiglu_limit", json!(0.0), "narrows to 0 in f32"),
        ("/swiglu_limit", json!(1e-46), "narrows to 0 in f32"),
        ("/swiglu_limit", json!(1e39), "narrows to inf in f32"),
        // The expert format. An fp8 export of the same model would be read as nibble pairs.
        ("/expert_dtype", json!("fp8"), "e2m1 nibble pairs"),
        // The resident fp8 scheme, both halves and the block size.
        ("/quantization_config/fmt", json!("e5m2"), "e4m3 weights"),
        (
            "/quantization_config/scale_fmt",
            json!("float"),
            "ue8m0 block scales",
        ),
        (
            "/quantization_config/weight_block_size",
            json!([64, 64]),
            "weight_block_size",
        ),
        // Both RoPE bases are live — ratio-0 layers use one, the other 41 use the other.
        ("/rope_theta", json!(0.0), "rope_theta 0 must be positive"),
        (
            "/compress_rope_theta",
            json!(0.0),
            "compress_rope_theta 0 must be positive",
        ),
        // YaRN. A zero original length disables the interpolation branch entirely.
        ("/rope_scaling/type", json!("linear"), "only YaRN"),
        ("/rope_scaling/factor", json!(0.0), "only YaRN"),
        (
            "/rope_scaling/original_max_position_embeddings",
            json!(0),
            "only YaRN",
        ),
        // The two mHC scalars. `hc_eps == 0` is refused by NOTHING downstream — it is
        // arithmetic, not a shape, and it perturbs every gate uniformly across 43 layers.
        ("/hc_sinkhorn_iters", json!(0), "hc_sinkhorn_iters is 0"),
        ("/hc_eps", json!(0.0), "hc_eps 0 must be positive"),
    ]);
}

/// The structural relations: a wrong value here reshapes a launch rather than refusing one.
#[test]
fn the_structurally_wrong_dimensions_are_refused() {
    each_refusal(&[
        // MQA. The whole attention frontend is written against one shared KV entry.
        ("/num_key_value_heads", json!(2), "shared-KV MQA"),
        // The rotated tail must sit inside the head, since it is the LAST 64 of one 512-wide
        // vector rather than a tensor of its own.
        ("/qk_rope_head_dim", json!(512), "must be inside head_dim"),
        ("/qk_rope_head_dim", json!(1024), "must be inside head_dim"),
        // `wo_a` is viewed as `(o_groups, o_lora_rank, n_heads·head_dim/o_groups)`, so a ragged
        // split reshapes into the wrong stride rather than failing.
        ("/o_groups", json!(0), "not divisible by o_groups"),
        ("/o_groups", json!(7), "not divisible by o_groups"),
        // The shared expert is a single always-on FFN; `MoE.__init__` asserts this outright.
        ("/n_shared_experts", json!(2), "n_shared_experts 2 != 1"),
        ("/num_experts_per_tok", json!(257), "top_k 257 > n_experts"),
        // The per-layer tables. One entry short is an index panic mid-load or, worse, a layer
        // silently treated as ratio 0 — and an UNSEEN ratio would land in the
        // compressor-no-indexer arm by default.
        ("/num_hidden_layers", json!(47), "compress_ratios has 46"),
        ("/compress_ratios/2", json!(8), "compress_ratios[2] = 8"),
        ("/compress_ratios/2", json!(64), "compress_ratios[2] = 64"),
        (
            "/num_hash_layers",
            json!(44),
            "num_hash_layers 44 > n_layers",
        ),
        // The FP4 group scale runs along the INPUT dim, so both expert widths must divide it:
        // `f4_groups` rounds up, and a ragged tail gives the last group a scale covering fewer
        // weights than the kernel assumes.
        ("/hidden_size", json!(4095), "not a multiple of F4_GROUP"),
        (
            "/moe_intermediate_size",
            json!(2047),
            "not a multiple of F4_GROUP",
        ),
    ]);
}

/// The layer roles the converter branches on, read off the shipped table.
///
/// `layer_has_compressor`/`layer_has_indexer`/`layer_routes_by_hash` are what
/// `convert_v4::write_layer_resident` uses to decide which tensors a layer carries, so a wrong
/// answer here is a converter that looks for `attn.indexer.wq_b` on a layer that has none — or,
/// worse, silently skips one that does.
#[test]
fn layer_roles_follow_compress_ratios() {
    let c: V4Config = parse_config(V4_SHIPPED).unwrap();
    // Layers 0 and 1 are pure sliding-window; 2 is ratio 4 (compressor + indexer); 3 is 128
    // (compressor only). The tail — 43, 44, 45 — is the mtp block's and must be unreachable.
    assert_eq!(
        (0..4)
            .map(|l| (
                c.layer_has_compressor(l).unwrap(),
                c.layer_has_indexer(l).unwrap()
            ))
            .collect::<Vec<_>>(),
        [(false, false), (false, false), (true, true), (true, false)]
    );
    // The whole model, counted: 41 compressor layers of which 21 carry an indexer. The counts
    // are what `v4_config.rs`'s doc comments claim, derived here rather than restated.
    let compressors = (0..c.n_layers)
        .filter(|&l| c.layer_has_compressor(l).unwrap())
        .count();
    let indexers = (0..c.n_layers)
        .filter(|&l| c.layer_has_indexer(l).unwrap())
        .count();
    assert_eq!((compressors, indexers), (41, 21));

    // Hash routing is the first `num_hash_layers` layers and nothing else. Those carry
    // `ffn.gate.tid2eid` and NO `ffn.gate.bias`; the rest are the reverse.
    assert_eq!(
        (0..5)
            .map(|l| c.layer_routes_by_hash(l))
            .collect::<Vec<_>>(),
        [true, true, true, false, false]
    );

    // Past `n_layers` is an ERROR, not the mtp block's ratio. `compress_ratios` is 46 long, so
    // `[43]` would answer 0 — "pure sliding-window" — for a layer that does not exist.
    for l in [c.n_layers, c.n_layers + 2, 45] {
        assert!(
            c.compress_ratio(l).is_err(),
            "layer {l} is past n_layers and must not read the mtp tail"
        );
    }
}

/// **The heart of the load boundary.** A V4 config must not become a zero-filled
/// `ModelConfig`, and the refusal must NAME the architecture rather than blaming whichever MLA
/// field V4 happens to omit first (`missing field kv_lora_rank` reads like a corrupt
/// checkpoint). Both directions, and against every other schema in the tree, because the
/// question is symmetric.
#[test]
fn v4_and_the_other_architectures_do_not_cross_parse() {
    let err = format!("{:#}", parse_config::<ModelConfig>(V4_SHIPPED).unwrap_err());
    assert!(
        err.contains("deepseek_v4") && err.contains("GlmMoeDsa"),
        "the refusal must name both the file's architecture and the schema's: {err}"
    );
    // The reverse, on a document that carries the DISCRIMINANT AND NOTHING ELSE. That is the
    // sharp form of the claim: `V4Config` needs 33 fields none of which are here, so a refusal
    // naming a missing dimension would be the weak reading. It must name the architecture.
    let glm = json!({"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"]});
    let err = format!(
        "{:#}",
        parse_config::<V4Config>(&glm.to_string()).unwrap_err()
    );
    assert!(
        err.contains("glm_moe_dsa") && err.contains("DeepseekV4"),
        "the refusal must name both the file's architecture and the schema's: {err}"
    );
    // Glimmer is the third, and both directions run against its REAL file.
    assert!(parse_config::<GlimmerConfig>(V4_SHIPPED).is_err());
    assert!(parse_config::<V4Config>(GLIMMER_SHIPPED).is_err());
    // K3 is the fourth (2026-08-16, with M9), both directions against its real file too —
    // the pair matters because both checkpoints ship MXFP4 routed experts, so the two
    // configs describe the same expert FORMAT around incompatible decode paths.
    assert!(parse_config::<K3Config>(V4_SHIPPED).is_err());
    assert!(parse_config::<V4Config>(K3_SHIPPED).is_err());
    // And a document that declares NEITHER field is refused rather than assumed to be this one.
    let mut doc = shipped();
    let obj = doc.as_object_mut().unwrap();
    obj.remove("model_type");
    obj.remove("architectures");
    let err = format!(
        "{:#}",
        parse_config::<V4Config>(&doc.to_string()).unwrap_err()
    );
    assert!(err.contains("declares neither"), "{err}");
}
