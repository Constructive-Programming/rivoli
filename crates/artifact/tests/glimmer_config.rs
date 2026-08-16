//! The Muse Glimmer load boundary: the shipped `config.json` parses, every field it declares
//! is required, and every setting that changes arithmetic without changing a shape is refused.
//!
//! Ported from `old:src/artifact/model.rs`'s `mod tests` Glimmer block (`wt/glimmer-s2` @
//! 6b7f496). It lives in `tests/` rather than in a `#[cfg(test)] mod tests` inside
//! `glimmer_config.rs` because that file is already ~430 lines of schema and argument, and the
//! soft line cap's contract is that the next edit shrinks a file rather than grows it. Nothing
//! here needs private access: `parse_config` is the binary's own entry.
//!
//! **These parse the SHIPPED config directly rather than assembling a base from transcribed
//! constants.** Mutating the real document is the stronger gate — every refusal test below
//! starts from a document that is known to parse — and a transcribed base is one more place for
//! the file's values to drift out of.
//!
//! The load boundary is the only place a foreign snapshot is inspected before its dimensions
//! reach a kernel, so every refusal here is one that would otherwise be an out-of-bounds panic
//! or, far more often in this model, fluent wrong text with no crash anywhere.
//!
//! No GPU, no network, no checkpoint — 5 KB of vendored JSON.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::glimmer_config::GlimmerConfig;
use rivoli_artifact::glm_config::ModelConfig;
use rivoli_artifact::schema::parse_config;
use serde_json::{Value, json};

/// `include_str!` rather than a path read: this port has no checkpoint on this machine, so a
/// path read would SKIP rather than run. 5 KB, pinned at the HF revision
/// `f84ecc3a0ea984a4c04542a84269e3d065350a6e`.
const GLIMMER_SHIPPED: &str =
    include_str!("../../../docs/measurement/glimmer-reference/config.json");

/// Kimi-K3's shipped config, vendored for its anchors and borrowed here for one assertion.
/// **The pair is what makes the cross-parse test worth writing**: both are
/// `*ForConditionalGeneration` wrappers around a `text_config`, so the shapes are similar
/// enough that only the discriminant separates them.
const K3_SHIPPED: &str = include_str!("../../../docs/measurement/k3-reference/config.json");

/// > **MOVED 2026-08-16.** `glimmer_err` and `each_refusal` lived here until `v4_config.rs`
/// > became the second gate built on the same two ideas and `build.rs`'s jscpd reported all four
/// > of their regions as clones. They are now `common::refusal::<T>` / `common::each_refusal::<T>`,
/// > generic over `ArchConfig`, with their arguments travelling verbatim. The two thin wrappers
/// > below keeps every call site in this file reading as it did. `glimmer_err` itself is gone:
/// > `each_refusal` was its only caller, and a wrapper with none is dead surface `-D warnings`
/// > reports on sight.
mod common;

/// [`common::each_refusal`], likewise.
fn each_refusal(rows: &[(&str, Value, &str)]) {
    common::each_refusal::<GlimmerConfig>(GLIMMER_SHIPPED, rows);
}

/// **Every field the schema declares is REQUIRED — enforced, not just claimed.**
///
/// Driven off the SHIPPED document rather than a list of field names, and asserted in **both**
/// directions: deleting a key the schema needs must refuse, and deleting one it does not carry
/// must still parse. The sets are compared whole, so a field that stops being required moves
/// between them and reddens.
///
/// **What it does and does not catch, measured 2026-08-11 rather than assumed.** The property
/// is "removing this key is refused" — by serde *or* by `validate`, which is the property that
/// actually matters, since either way the checkpoint is rejected. So:
///
/// - `#[serde(default)]` on `attention_bias` **reddens** (default `false` is the acceptable
///   value, so the config would parse and run).
/// - `#[serde(default)]` on `head_dim` **does not** — the default 0 is caught by the width loop
///   in `validate`, so the config is still refused and no defect exists. An earlier draft of
///   this doc named `head_dim` as the worked example and was wrong; the red-proof run is what
///   corrected it.
///
/// The gap that leaves: a defaulted field whose default is both *acceptable to `validate`* and
/// *wrong for this checkpoint*. Every such field here is a `bool` or `String` the shipped config
/// pins, and each is asserted by value in [`glimmer_shipped_config_parses_and_matches_the_doc`].
#[test]
fn every_glimmer_field_is_required() {
    // The `text_config` keys this schema does NOT bind. Everything else in the dict must be
    // load-bearing; a key added to the struct without being removed from here fails. Sorted,
    // because the comparison below is on the whole list.
    const NOT_IN_SCHEMA: [&str; 6] = [
        "attention_dropout",
        "bos_token_id",
        "eos_token_id", // EOS comes from `generation_config`, which lists TWO ids (trap 13)
        "initializer_range",
        "pad_token_id",
        "use_cache",
    ];
    let doc: Value = serde_json::from_str(GLIMMER_SHIPPED).unwrap();
    let text = doc["text_config"]
        .as_object()
        .expect("the shipped config's text_config is not an object");
    let (mut required, mut ignored) = (Vec::new(), Vec::new());
    for k in text.keys().cloned().collect::<Vec<String>>() {
        let mut d = doc.clone();
        d["text_config"].as_object_mut().unwrap().remove(&k);
        match parse_config::<GlimmerConfig>(&d.to_string()) {
            Err(_) => required.push(k),
            Ok(_) => ignored.push(k),
        }
    }
    // Sorted HERE, because `serde_json` runs with `preserve_order` and so `text.keys()` yields
    // FILE order. It happens to equal sorted order for the vendored config, which is why the
    // whole-list comparison held — but a re-vendor from a source that emits keys in declaration
    // order would redden this with a diff of two identical SETS, and point the reader at a serde
    // change that did not happen. Review, 2026-08-11.
    ignored.sort();
    assert_eq!(
        ignored, NOT_IN_SCHEMA,
        "the set of text_config keys this schema tolerates as ABSENT has changed.\n  \
         required: {required:?}\n  If a field gained #[serde(default)] it moved into the \
         tolerated set, which is the silent-wrong-text case this test exists for."
    );
    // The wrapper's own fields. `quantization_config` is deliberately absent from the shipped
    // file, so it is the one key whose ABSENCE is correct — asserted separately below rather
    // than folded in here.
    for k in ["text_config", "dtype"] {
        let mut d = doc.clone();
        d.as_object_mut().unwrap().remove(k);
        assert!(
            parse_config::<GlimmerConfig>(&d.to_string()).is_err(),
            "removing the wrapper's {k:?} must refuse"
        );
    }
    // Neither level, which is why `quantization_config` is not in NOT_IN_SCHEMA either: it is
    // bound by the schema at both and present at neither, so it never enters the loop above.
    for at in [&doc, &doc["text_config"]] {
        assert!(
            at.get("quantization_config").is_none(),
            "the shipped config must NOT carry a quantization_config at any level — the schema \
             refuses one, and K3's file is the counter-example that puts it in text_config"
        );
    }
}

/// The shipped config parses, and every value this port acts on is what
/// `old:docs/reference/glimmer-architecture.md` §1 records. Pinned against the FILE, so the doc
/// and the schema cannot drift apart silently.
#[test]
fn glimmer_shipped_config_parses_and_matches_the_doc() {
    let cfg: GlimmerConfig = parse_config(GLIMMER_SHIPPED).unwrap();
    assert_eq!(cfg.dtype, "bfloat16");
    let t = &cfg.text;
    assert_eq!(t.model_type, "muse_glimmer_text");
    assert_eq!(t.n_layers, 52);
    assert_eq!(t.hidden, 6656);
    assert_eq!(t.inter, 19968);
    assert_eq!(t.vocab, 202_048);
    assert_eq!(t.n_heads, 32);
    assert_eq!(t.num_key_value_heads, 2);
    assert_eq!(t.head_dim, 128);
    assert_eq!(t.sliding_window, 2048);
    assert_eq!(t.max_position_embeddings, 131_072);
    assert_eq!(t.rope_parameters.rope_theta, 500_000.0);
    assert_eq!(t.rope_parameters.rope_type, "default");
    assert_eq!(t.final_logit_softcapping, 20.0);
    assert_eq!(t.qk_scale_factor, 3.87);
    assert_eq!(t.rms_norm_eps, 1e-5);
    assert_eq!(t.post_norm_eps, 1e-8);
    assert!(!t.tie_word_embeddings);
    assert!(!t.attention_bias);
    assert_eq!(t.hidden_activation, "silu");

    // `head_dim` is NOT `hidden / n_heads` here (6656/32 = 208, against 128). The guard against
    // "drop the field and derive it" is that removing it fails to COMPILE, which is stronger
    // than any assertion — an `assert_ne!` over these three already-pinned constants was
    // deleted by review 2026-08-11 as unable to fail independently.

    // The layer map, counted from the arrays rather than from the [s,s,s,full] period — the
    // period is a fact about this checkpoint, the arrays are the contract. The exact array
    // equality below implies both the 39/13 split and that the LAST layer is full (a named gate
    // blind spot), so neither is asserted separately.
    let full: Vec<usize> = (0..t.n_layers)
        .filter(|&i| !t.layer_is_sliding(i).unwrap())
        .collect();
    assert_eq!(full, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]);
    // Out of range is an ERROR, not a silent "full" — see `layer_is_sliding`.
    assert!(t.layer_is_sliding(t.n_layers).is_err());
}

/// The shape table agrees with the tensor list, and disagrees loudly with anything else.
///
/// `GLIMMER_LAYER_TENSORS` and `layer_tensor_shape` are two statements that must stay in step:
/// the constant says which tensors exist and the table says how wide each is, and an entry
/// added to one without the other is the failure the `bail!` arm names.
///
/// The widths are the shipped ones, so this also pins the distinction the whole schema turns on
/// — `q_proj` is `[4096, 6656]` and NOT square, because `head_dim` is 128 rather than
/// `hidden / n_heads` = 208.
#[test]
fn the_layer_shape_table_covers_exactly_the_layer_tensors() {
    use rivoli_artifact::glimmer::GLIMMER_LAYER_TENSORS;
    let cfg: GlimmerConfig = parse_config(GLIMMER_SHIPPED).unwrap();
    let t = &cfg.text;
    for name in GLIMMER_LAYER_TENSORS {
        let shape = t
            .layer_tensor_shape(name)
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert!(
            !shape.is_empty() && shape.iter().all(|&d| d > 0),
            "{name} has a degenerate shape {shape:?}"
        );
    }
    assert_eq!(
        t.layer_tensor_shape("self_attn.q_proj").unwrap(),
        [4096, 6656]
    );
    assert_eq!(
        t.layer_tensor_shape("self_attn.k_proj").unwrap(),
        [256, 6656]
    );
    assert_eq!(
        t.layer_tensor_shape("self_attn.o_proj").unwrap(),
        [6656, 4096]
    );
    assert_eq!(
        t.layer_tensor_shape("mlp.down_proj").unwrap(),
        [6656, 19968]
    );
    assert_eq!(t.layer_tensor_shape("input_layernorm").unwrap(), [6656]);
    // A name the constant does not carry must BAIL rather than be given a plausible shape —
    // the arm that exists so an extended constant cannot silently acquire a wrong width.
    assert!(t.layer_tensor_shape("self_attn.qk_norm").is_err());
    assert!(t.layer_tensor_shape("mlp.experts.0.up_proj").is_err());
}

/// The rows whose refusal is a positive claim about arithmetic: the descent, the two per-layer
/// arrays, and the RoPE pairing invariant.
///
/// Split from [`glimmer_refuses_the_silently_wrong_settings`] by what the rows are ABOUT, which
/// is also what keeps either body from becoming an unnamed run of assertions the code-health
/// gate refuses.
#[test]
fn glimmer_refuses_a_wrong_descent_or_a_broken_layer_map() {
    each_refusal(&[
        // The descent. `vision_config` is a sibling carrying several of the same keys, so
        // landing in it is the realistic wrong-dict case rather than a hypothetical one.
        (
            "/text_config/model_type",
            json!("muse_glimmer_vision"),
            "muse_glimmer_text",
        ),
        // The two per-layer arrays are indexed by layer id everywhere downstream.
        (
            "/text_config/layer_types",
            json!(["sliding_attention"]),
            "1 entries",
        ),
        ("/text_config/num_hidden_layers", json!(51), "52 entries"),
        // An unknown layer kind, which would otherwise read as "not sliding" = full.
        (
            "/text_config/layer_types/0",
            json!("chunked_attention"),
            "expected",
        ),
        // **The pairing invariant, in BOTH directions** — the strongest claim the config alone
        // can make. Layer 0 is sliding and rotated; layer 3 is full and not.
        ("/text_config/layer_rope_theta/0", json!(0), "rotated IFF"),
        (
            "/text_config/layer_rope_theta/3",
            json!(500_000.0),
            "rotated IFF",
        ),
        // A rotated layer asking for a base the single shared table is not built from.
        (
            "/text_config/layer_rope_theta/0",
            json!(10_000.0),
            "read and then ignored",
        ),
        // GQA: 32 query heads do not divide into 3 KV heads.
        (
            "/text_config/num_key_value_heads",
            json!(3),
            "whole number of query heads",
        ),
    ]);
}

/// **The defect run.** Every load-bearing width and named setting, mutated one at a time, must
/// refuse — and the assertion is on the MESSAGE, so a refusal that happens to fire for an
/// unrelated reason does not count as this row passing.
///
/// Each row is a value that changes the arithmetic without changing a shape, which is the
/// failure class this model is full of: nothing downstream crashes on any of them.
#[test]
fn glimmer_refuses_the_silently_wrong_settings() {
    each_refusal(&[
        // **One row per entry of the width table**, so deleting any entry reddens exactly one
        // row. Three of the nine were sampled until review 2026-08-11 pointed out the rest were
        // free.
        ("/text_config/hidden_size", json!(0), "hidden_size is 0"),
        (
            "/text_config/intermediate_size",
            json!(0),
            "intermediate_size is 0",
        ),
        ("/text_config/vocab_size", json!(0), "vocab_size is 0"),
        (
            "/text_config/num_attention_heads",
            json!(0),
            "num_attention_heads is 0",
        ),
        (
            "/text_config/num_key_value_heads",
            json!(0),
            "num_key_value_heads is 0",
        ),
        ("/text_config/head_dim", json!(0), "head_dim is 0"),
        (
            "/text_config/sliding_window",
            json!(0),
            "sliding_window is 0",
        ),
        (
            "/text_config/max_position_embeddings",
            json!(0),
            "max_position_embeddings is 0",
        ),
        // Scaling schemes and tying: both silently unimplemented rather than refused.
        (
            "/text_config/rope_parameters/rope_type",
            json!("yarn"),
            "RopeNoYarn",
        ),
        (
            "/text_config/tie_word_embeddings",
            json!(true),
            "declares them untied",
        ),
        ("/text_config/attention_bias", json!(true), "none ships"),
        ("/text_config/hidden_activation", json!("gelu"), "SwiGLU"),
        // f32 narrowing. `1e-46` passes any f64 positivity test and reaches every RMSNorm as
        // `0.0f32`; the softcap at 0 is a divide-by-zero the greedy path cannot see.
        ("/text_config/rms_norm_eps", json!(1e-46), "narrows to 0"),
        (
            "/text_config/final_logit_softcapping",
            json!(0.0),
            "narrows to 0",
        ),
        // "narrows to", not the field name: the old tree records a review finding where two
        // rows shared a `want` substring and transposing an argument left both green.
        ("/text_config/output_multiplier", json!(-0.5), "narrows to"),
        ("/text_config/post_norm_eps", json!(0.0), "narrows to"),
        ("/text_config/qk_scale_factor", json!(0.0), "narrows to"),
        // NOT a row: `rope_parameters.rope_theta`. Its `ensure_f32_positive` entry is
        // unreachable by any SINGLE mutation — the pairing loop runs first and refuses ("read
        // and then ignored") as soon as the global base stops matching the 39 sliding layers'
        // own. It is reachable only by mutating the global base and all 39 together, which this
        // one-pointer helper cannot express. Recorded rather than asserted, so the gap is
        // visible instead of looking like coverage.
        // The wrapper-level dtype: a 4-bit or fp8 export read as BF16 is noise at every width,
        // and the model card advertises exactly such a release.
        ("/dtype", json!("float8_e4m3fn"), "BF16 throughout"),
    ]);
}

/// **A packed export must refuse even though its `dtype` is honest.**
///
/// Not a row in the table above because [`glimmer_err`] replaces an EXISTING value and this key
/// must be INSERTED — the shipped file has none, which is the correct state and is asserted by
/// [`every_glimmer_field_is_required`].
///
/// The block is K3's own, read out of its vendored config rather than hand-written, so this
/// stays a counter-example that exists rather than one that is imagined: `bfloat16` at the top
/// alongside 4-bit packed weights. Before this guard existed, serde ignored it and the document
/// parsed clean.
///
/// > **THE NESTED ARM IS NEW, 2026-08-16, and writing this test is what found the hole.** The
/// > reference inserts the block at the WRAPPER level and asserts one refusal. K3's real file —
/// > the checkpoint the whole argument cites — carries `quantization_config` **inside
/// > `text_config`**, with only `dtype` at the top. So the ported guard was checking the level
/// > its own counter-example does not use, and the first draft of this test failed on
/// > `k3["quantization_config"]` being `Null`. Both arms are asserted now, and the block is read
/// > from where K3 actually puts it, so the fixture cannot drift back into the imagined shape.
#[test]
fn a_packed_export_refuses_even_with_an_honest_dtype() {
    let k3: Value = serde_json::from_str(K3_SHIPPED).unwrap();
    let packed = k3["text_config"]["quantization_config"].clone();
    assert!(
        packed.is_object(),
        "K3's vendored config carries the block inside text_config — if that moved, the \
         counter-example this guard cites has changed shape and the guard needs re-arguing"
    );
    assert_eq!(
        k3["dtype"],
        json!("bfloat16"),
        "and an honest dtype with it"
    );
    // Both levels, each inserted into an otherwise-shipped document, so each arm is the only
    // thing wrong with the config it refuses.
    for nested in [false, true] {
        let mut doc: Value = serde_json::from_str(GLIMMER_SHIPPED).unwrap();
        let slot = if nested {
            doc["text_config"].as_object_mut()
        } else {
            doc.as_object_mut()
        };
        slot.unwrap()
            .insert("quantization_config".into(), packed.clone());
        let err = format!(
            "{:#}",
            parse_config::<GlimmerConfig>(&doc.to_string()).unwrap_err()
        );
        assert!(
            err.contains("unquantized release"),
            "a config carrying a quantization_config (nested={nested}) must refuse even with \
             dtype bfloat16, which is exactly how Kimi-K3's checkpoint ships: {err}"
        );
    }
}

/// The architecture check fires before serde reads a dimension, and it fires in both
/// directions — a Glimmer document must not parse as another schema either. Without the second
/// half this passes on a `parse_config` that refuses everything.
///
/// > **PORT NOTE 2026-08-16.** The reference reads the Glimmer document as all three of the
/// > other schemas. `V4Config` and `K3Config` do not exist in this tree yet (M8 and M9), so the
/// > first half is GLM's alone. The second half — **K3's document read as Glimmer's schema** —
/// > needs only K3's vendored *config*, not its Rust type, so the sharper of the two survives
/// > the port intact: both files are `*ForConditionalGeneration` wrappers with a `text_config`,
/// > which is what makes the pair worth asserting at all.
#[test]
fn glimmer_and_the_other_architectures_do_not_cross_parse() {
    let err = format!(
        "{:#}",
        parse_config::<ModelConfig>(GLIMMER_SHIPPED).unwrap_err()
    );
    assert!(
        err.contains("MuseGlimmer") && err.contains("GlmMoeDsa"),
        "GLM's schema must refuse a Glimmer document on the architecture, naming both: {err}"
    );
    let err = format!(
        "{:#}",
        parse_config::<GlimmerConfig>(K3_SHIPPED).unwrap_err()
    );
    assert!(
        err.contains("MuseGlimmer") && err.contains("kimi_k3"),
        "must refuse on the architecture, naming both sides: {err}"
    );
}
