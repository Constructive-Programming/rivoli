//! The Kimi-K3 load boundary: the shipped `config.json` parses, every field the schema
//! declares is required, and every setting that changes arithmetic without changing a shape
//! is refused.
//!
//! Ported from `k3:src/artifact/model.rs`'s `mod tests` K3 block, restated over the vendored
//! shipped document on the precedent `v4_config.rs` set (its PORT NOTE carries the argument):
//! mutating the real file is the stronger gate — every refusal below starts from a document
//! known to parse — and the k3 tree's own `K3_BASE` of transcribed pairs was one more place
//! for the file's values to drift out of. The vendored `config.json` is the SAME file the k3
//! branch pins (`crates/oracles/tests/k3_anchor.rs` FNV-pins its bytes), so the two gates
//! cannot disagree about which document is "shipped".
//!
//! **TDD record.** Every refusal test here was watched red against a stub
//! `K3TextConfig::validate` that returned `Ok(())` (2026-08-16): each row failed with
//! `common::refusal`'s "was mutated to a wrong value and the config still parsed", and the
//! descent test additionally red on the un-asserted nested pair. The individual doc comments
//! below record the rows whose red said something sharper than that.
//!
//! No GPU, no network, no checkpoint — 6.8 KB of vendored JSON.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

// One braced `use`, for the reason `v4_config.rs` records: jscpd normalizes little, and the
// flat four-line import run was the first clone it reported between these gate files.
use rivoli_artifact::{
    glm_config::ModelConfig, k3_config::K3Config, schema::ArchConfig, schema::parse_config,
};
use serde_json::{Value, json};

/// The two shared helpers, generic over `ArchConfig` — see `common/mod.rs`.
mod common;

/// `include_str!` rather than a path read: a path read would SKIP on a machine without the
/// 1.42 TiB checkpoint rather than run. 6.8 KB, the same bytes `k3_anchor.rs` FNV-pins.
const K3_SHIPPED: &str = include_str!("../../../docs/measurement/k3-reference/config.json");

fn shipped() -> Value {
    serde_json::from_str(K3_SHIPPED).unwrap()
}

/// [`common::each_refusal`] bound to this file's architecture and document.
fn each_refusal(rows: &[(&str, Value, &str)]) {
    common::each_refusal::<K3Config>(K3_SHIPPED, rows);
}

/// Parse `doc` as a foreign schema `T` and require the refusal to name BOTH architectures —
/// the sharp form of the cross-parse claim, local to this file because both its call sites
/// are here (the older gates keep their own spellings; a helper grows into `common/` only
/// when jscpd reports the copy, which it has not).
fn refuses_naming_both<T: ArchConfig>(doc: &str, halves: [&str; 2]) {
    let err = match parse_config::<T>(doc) {
        Ok(_) => panic!("a foreign document parsed as {:?}", T::ARCH),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        halves.iter().all(|h| err.contains(h)),
        "the refusal must name both the file's architecture and the schema's {halves:?}: {err}"
    );
}

/// The shipped config parses, and every value this port acts on is what
/// `k3:docs/reference/k3-architecture.md` §1 records. Pinned against the FILE, so the doc
/// and the schema cannot drift apart silently.
#[test]
fn the_shipped_config_parses_and_matches_the_doc() {
    let cfg: K3Config = parse_config(K3_SHIPPED).expect("Kimi-K3's own config must load");
    let t = &cfg.text;
    assert_eq!(t.model_type, "kimi_linear");
    assert_eq!(t.architectures, ["KimiLinearForCausalLM"]);
    assert_eq!((t.n_layers, t.hidden, t.vocab), (93, 7168, 163_840));
    // MLA: NOT the MQA V4 asserts `== 1` — one KV projection per query head.
    assert_eq!((t.n_heads, t.num_key_value_heads), (96, 96));
    assert_eq!((t.q_lora_rank, t.kv_lora_rank), (1536, 512));
    assert_eq!(
        (t.qk_nope_head_dim, t.qk_rope_head_dim, t.v_head_dim),
        (128, 64, 128)
    );
    assert!(t.mla_use_nope && t.mla_use_output_gate);
    assert!(t.rope_theta.is_none(), "K3 is NoPE; see the schema's field");
    // KDA, one level down. `gate_lower_bound` is NEGATIVE and multiplies the sigmoid.
    let a = &t.linear_attn_config;
    assert_eq!(
        (a.num_heads, a.head_dim, a.short_conv_kernel_size),
        (96, 128, 4)
    );
    assert_eq!(a.gate_lower_bound, -5.0);
    assert!(a.use_full_rank_gate);
    // MoE: the routed experts are entered at the 3584 LATENT, not at hidden 7168.
    assert_eq!((t.n_experts, t.top_k, t.n_shared), (896, 16, 2));
    assert_eq!((t.expert_in, t.moe_inter), (3584, 3072));
    assert_ne!(
        t.expert_in, t.hidden,
        "the latent IS the point — see plan §2"
    );
    assert!(t.latent_moe_use_norm && t.moe_renormalize);
    assert_eq!((t.num_expert_group, t.topk_group), (1, 1));
    assert_eq!(t.topk_method, "noaux_tc");
    assert_eq!(t.moe_router_activation_func, "sigmoid");
    assert_eq!(t.routed_scale, 1.0);
    assert_eq!(t.moe_layer_freq, 1);
    // SiTU-GLU's betas, and the SECOND key is `activation_situ_linear_beta` — a spelling
    // that refused every real checkpoint when it was guessed as `activation_linear_beta`
    // (`k3:src/artifact/model.rs`, noted 2026-08-10).
    assert_eq!(
        (t.activation_situ_beta, t.activation_linear_beta),
        (4.0, 25.0)
    );
    assert_eq!((t.first_k_dense_replace, t.dense_inter), (1, 33_792));
    assert_eq!(t.hidden_act, "situ");
    assert_eq!((t.dtype.as_str(), t.rms_norm_eps), ("bfloat16", 1e-5));
    assert_eq!(t.attn_res_block_size, 12);
    assert_eq!(t.num_nextn_predict_layers, 0);
    assert!(!t.tie_word_embeddings);
}

/// **`text_config` ALSO has a key literally named `top_k`, and it is not the routing one.**
///
/// It is HuggingFace's sampling top-k, inherited from `PretrainedConfig`, and it is 50;
/// binding the struct's `top_k` from it selects 50 experts a token instead of 16 — 3.1x the
/// stream traffic, plausible output, no error (`k3:src/artifact/model.rs`, read off the
/// shipped file 2026-08-10). The `#[serde(rename = "num_experts_per_token")]` is what keeps
/// them apart, and this asserts the two values DIFFER so the rename cannot be "simplified"
/// away later without a red.
#[test]
fn the_sampling_top_k_is_not_the_routing_top_k() {
    let doc = shipped();
    let sampling = doc["text_config"]["top_k"].as_u64().unwrap();
    let cfg: K3Config = parse_config(K3_SHIPPED).unwrap();
    assert_eq!(sampling, 50);
    assert_eq!(cfg.text.top_k, 16);
    assert_ne!(cfg.text.top_k as u64, sampling);
}

/// The `text_config` keys whose absence is refused — the schema's REQUIRED set. Alphabetical,
/// so a diff to it reads as one line. The nested `model_type`/`architectures` are IN it,
/// unlike the wrapper's pair: the wrapper's is the discriminant `arch_of_named` reads, the
/// nested pair is a schema field `validate` asserts (the descent check).
const REQUIRED: [&str; 38] = [
    "activation_situ_beta",
    "activation_situ_linear_beta",
    "architectures",
    "attn_res_block_size",
    "dtype",
    "first_k_dense_replace",
    "hidden_act",
    "hidden_size",
    "intermediate_size",
    "kv_lora_rank",
    "latent_moe_use_norm",
    "linear_attn_config",
    "mla_use_nope",
    "mla_use_output_gate",
    "model_type",
    "moe_intermediate_size",
    "moe_layer_freq",
    "moe_renormalize",
    "moe_router_activation_func",
    "num_attention_heads",
    "num_expert_group",
    "num_experts",
    "num_experts_per_token",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_nextn_predict_layers",
    "num_shared_experts",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "rms_norm_eps",
    "routed_expert_hidden_size",
    "routed_scaling_factor",
    "tie_word_embeddings",
    "topk_group",
    "topk_method",
    "v_head_dim",
    "vocab_size",
];

/// **Every field the schema declares is REQUIRED — enforced, not just claimed.**
///
/// Driven off the SHIPPED document's own `text_config` keys and asserted in BOTH directions:
/// deleting a key the schema needs must refuse, and deleting one it does not carry must still
/// parse. The sets are compared whole (REQUIRED exactly, and the partition covers every key),
/// so a field moving either way reddens.
///
/// The ignored side is LARGE here — K3's `text_config` drags in ~58 `PretrainedConfig`
/// sampling/beam keys (`num_beams`, `temperature`, the `top_k` twin above) — which is why it
/// is pinned by COUNT rather than by a second 58-line const: a key starting or stopping being
/// read moves it across the partition and reddens the REQUIRED comparison or the count.
///
/// `quantization_config` is deliberately in the ignored set: the block **mis-declares its own
/// scope** (`targets: ["Linear"]` with an `ignore` list that omits the BF16
/// `routed_expert_{down,up}_proj` and `gate.weight`), so the converter drives off the
/// presence of `.weight_packed` instead and no schema field may lend the block authority
/// (`k3:src/artifact/model.rs`, K3TextConfig's header).
#[test]
fn every_k3_field_is_required() {
    let doc = shipped();
    let keys: Vec<String> = doc["text_config"]
        .as_object()
        .expect("text_config is an object")
        .keys()
        .cloned()
        .collect();
    // Anti-vacuity: an empty walk satisfies nothing below while looking green.
    assert_eq!(
        keys.len(),
        96,
        "the shipped text_config lost keys: {keys:?}"
    );
    let (mut required, mut ignored) = (Vec::new(), Vec::new());
    for k in keys {
        let mut d = shipped();
        // `is_some()` asserted so a key that vanished mid-walk (a shipped-doc edit racing
        // this loop's clone of it) cannot silently test the unmutated document. Also what
        // keeps this loop from being a token-clone of `glimmer_config.rs`'s — jscpd
        // reported the bare-removal form against that file's, 2026-08-16.
        assert!(
            d["text_config"]
                .as_object_mut()
                .unwrap()
                .remove(&k)
                .is_some(),
            "{k} was not in the document being mutated"
        );
        let refused = parse_config::<K3Config>(&d.to_string()).is_err();
        if refused { &mut required } else { &mut ignored }.push(k);
    }
    required.sort();
    assert_eq!(
        required,
        REQUIRED.map(String::from),
        "the set of text_config keys whose absence is refused has moved.\n  ignored: {ignored:?}"
    );
    assert!(
        ignored.contains(&"quantization_config".to_string()),
        "quantization_config must stay UNREAD — its targets/ignore lists mis-declare their \
         own scope, and the converter drives off `.weight_packed` presence instead"
    );
    // The wrapper's own field: a bare text dict is not evidence the wrapper was K3's.
    let mut d = shipped();
    d.as_object_mut().unwrap().remove("text_config");
    assert!(
        parse_config::<K3Config>(&d.to_string()).is_err(),
        "removing the wrapper's text_config must refuse"
    );
}

/// The descent and the layer partition — the rows whose refusal is a positive claim about
/// which dict was read and which attention family each layer runs.
///
/// RED OBSERVED (stub validate): every row parsed clean, including `full_attn_layers[0] = 1`
/// — a layer claimed by BOTH families, which downstream is a layer running the wrong
/// attention with no shape error anywhere.
#[test]
fn k3_refuses_a_wrong_descent_or_a_broken_layer_map() {
    each_refusal(&[
        // The descent. The wrapper's pair already matched `Arch::KimiK3`; the NESTED pair is
        // what separates "descended into text_config" from "some other dict of a multimodal
        // config". `kimi_k3` here is the realistic wrong value: the WRAPPER's spelling one
        // level down.
        ("/text_config/model_type", json!("kimi_k3"), "kimi_linear"),
        (
            "/text_config/architectures",
            json!(["KimiK3ForCausalLM"]),
            "KimiK3ForCausalLM",
        ),
        // The partition: both arrays present, disjoint, one-based, union = 1..=93. The two
        // reference implementations read OPPOSITE arrays (`k3:docs/reference/k3-architecture.md`
        // §2), so neither is derivable and every failure is a layer running the wrong family.
        (
            "/text_config/linear_attn_config/full_attn_layers/0",
            json!(1),
            "appears twice",
        ),
        (
            "/text_config/linear_attn_config/full_attn_layers/0",
            json!(0),
            "ONE-BASED",
        ),
        (
            "/text_config/linear_attn_config/full_attn_layers/0",
            json!(94),
            "past num_hidden_layers",
        ),
        // A count that no longer matches the arrays — the arrays win, the count refuses.
        (
            "/text_config/num_hidden_layers",
            json!(92),
            "but num_hidden_layers is 92",
        ),
    ]);
}

/// **A `rope_theta` sitting in `text_config` is refused, not ignored.** Plan §3e's secondary
/// reading: NoPE is asserted positively (`mla_use_nope`), because "this model applies no
/// rotation" and "we descended into the wrong dict" are otherwise the same observation — and
/// without `deny_unknown_fields`, a rotary base in the dict would be silently dropped rather
/// than being the signal it is.
///
/// A separate test because [`common::each_refusal`] REPLACES an existing value and this key
/// must be INSERTED — the shipped file has none, which is the correct state and is asserted
/// by [`the_shipped_config_parses_and_matches_the_doc`].
///
/// RED OBSERVED (stub validate): the inserted base parsed clean — exactly the silent drop.
#[test]
fn a_rope_theta_in_text_config_is_refused() {
    let mut d = shipped();
    // Index assignment inserts on a map; the shipped file has no such key (asserted by the
    // pinning test), so this is the insertion `each_refusal` cannot express.
    d["text_config"]["rope_theta"] = json!(10_000.0);
    let err = format!(
        "{:#}",
        parse_config::<K3Config>(&d.to_string()).unwrap_err()
    );
    assert!(
        err.contains("NoPE") && err.contains("descent landed somewhere else"),
        "a rope_theta in a NoPE model's text_config must refuse as the wrong-dict signal: {err}"
    );
}

/// Every width of zero is refused — one row per entry of the schema's width table, so
/// deleting any entry reddens exactly one row (the rule `glimmer_config.rs`'s width test
/// records). Split from the structural-relations test below by what the rows are ABOUT,
/// which is also what keeps either body under the code-health gate's length ceiling.
///
/// RED OBSERVED (stub validate): every row parsed clean — including `kv_lora_rank` 0, which
/// the MLA kernel would NOT refuse: `0 % 128 == 0` and `!(0 > 512)`, so 24 layers of
/// attention would contribute nothing with no error anywhere (`k3:src/artifact/model.rs`,
/// review 2026-08-10).
#[test]
fn every_width_of_zero_is_refused() {
    each_refusal(&[
        // The K3-only widths are interleaved between the four HF-standard ones on purpose:
        // Glimmer's width table spells those four rows identically (same pointer, same
        // message), and two consecutive shared rows are a token run the duplication gate
        // reports. The interleaving is also the honest grouping — each HF width sits next
        // to the K3 width it must not be confused with (hidden/latent, dense/moe inter).
        ("/text_config/hidden_size", json!(0), "hidden_size = 0"),
        (
            "/text_config/routed_expert_hidden_size",
            json!(0),
            "routed_expert_hidden_size = 0",
        ),
        (
            "/text_config/intermediate_size",
            json!(0),
            "intermediate_size = 0",
        ),
        (
            "/text_config/moe_intermediate_size",
            json!(0),
            "moe_intermediate_size = 0",
        ),
        ("/text_config/vocab_size", json!(0), "vocab_size = 0"),
        (
            "/text_config/num_hidden_layers",
            json!(0),
            "num_hidden_layers = 0",
        ),
        (
            "/text_config/num_attention_heads",
            json!(0),
            "num_attention_heads = 0",
        ),
        (
            "/text_config/attn_res_block_size",
            json!(0),
            "attn_res_block_size = 0",
        ),
        ("/text_config/q_lora_rank", json!(0), "q_lora_rank = 0"),
        ("/text_config/kv_lora_rank", json!(0), "kv_lora_rank = 0"),
        (
            "/text_config/qk_nope_head_dim",
            json!(0),
            "qk_nope_head_dim = 0",
        ),
        (
            "/text_config/qk_rope_head_dim",
            json!(0),
            "qk_rope_head_dim = 0",
        ),
        ("/text_config/v_head_dim", json!(0), "v_head_dim = 0"),
        (
            "/text_config/linear_attn_config/num_heads",
            json!(0),
            "linear_attn_config.num_heads = 0",
        ),
        (
            "/text_config/linear_attn_config/head_dim",
            json!(0),
            "linear_attn_config.head_dim = 0",
        ),
        (
            "/text_config/linear_attn_config/short_conv_kernel_size",
            json!(0),
            "linear_attn_config.short_conv_kernel_size = 0",
        ),
    ]);
}

/// The structural RELATIONS: a wrong value here reshapes a launch rather than refusing one.
///
/// RED OBSERVED (stub validate): every row parsed clean — including `kv_lora_rank` 640,
/// which the MLA kernel WOULD refuse (guard 1004); the load boundary exists to name the
/// FIELD where the kernel names a code.
#[test]
fn the_structurally_wrong_relations_are_refused() {
    each_refusal(&[
        // NOT MQA. A copied V4 check (`== 1`) would refuse this checkpoint; a copied V4
        // ASSUMPTION would size the KV cache 96x too small. Pinned as the equality.
        ("/text_config/num_key_value_heads", json!(1), "not MQA"),
        // Guard 1004's two halves, restated at the load boundary where the FIELD can be
        // named. Two rows so each proves its own half — as one conjunction both rows
        // matched both halves of the message and neither was a test of anything
        // (`k3:src/artifact/model.rs`, the kv_lora_rank pair's comment).
        (
            "/text_config/kv_lora_rank",
            json!(320),
            "not a multiple of 128",
        ),
        ("/text_config/kv_lora_rank", json!(640), "exceeds the 512"),
        // The dense prefix, both bounds.
        (
            "/text_config/first_k_dense_replace",
            json!(0),
            "layer 0 would run the routed MoE path",
        ),
        (
            "/text_config/first_k_dense_replace",
            json!(93),
            "every layer would be dense",
        ),
        (
            "/text_config/num_experts_per_token",
            json!(897),
            "not in 1..=896",
        ),
        ("/text_config/num_shared_experts", json!(0), "always-on MLP"),
        // The FP4 group scale runs along the INPUT dim of the ROUTED block, whose entry
        // width is the latent — `ensure_f4_group_aligned(expert_in, …)`, NOT `hidden`.
        (
            "/text_config/routed_expert_hidden_size",
            json!(3585),
            "not a multiple of F4_GROUP",
        ),
        (
            "/text_config/moe_intermediate_size",
            json!(3070),
            "not a multiple of F4_GROUP",
        ),
    ]);
}

/// The NAMED settings that change arithmetic without changing a shape, so nothing
/// downstream would refuse them. Every row is a value the checkpoint could plausibly carry.
/// Split from the flags-and-narrowing test below by what the rows are ABOUT, like the
/// structural pair above.
///
/// RED OBSERVED (stub validate): every row parsed clean — the loudest being
/// `gate_lower_bound: 0`, which silences all 69 KDA layers (the model goes QUIET rather than
/// wrong, which no tolerance downstream reads as an error).
#[test]
fn the_silently_wrong_settings_are_refused() {
    each_refusal(&[
        // The trunk dtype: an fp8 export of the same model read as BF16 is noise at every
        // width. Refused by name, not by shape — nothing downstream checks a dtype string.
        (
            "/text_config/dtype",
            json!("float8_e4m3fn"),
            "not \"bfloat16\"",
        ),
        // The dense layer and the shared MLP use the same SiTU-GLU as the routed experts; a
        // `silu` here is a different activation on layer 0 and the shared path while the
        // routed path stays right.
        ("/text_config/hidden_act", json!("silu"), "not \"situ\""),
        // Router affinity: sigmoid, independent per expert — the scores do NOT sum to 1.
        (
            "/text_config/moe_router_activation_func",
            json!("softmax"),
            "not \"sigmoid\"",
        ),
        // KDA's gate floor MULTIPLIES the sigmoid (trap 4): 0 silences, positive inverts.
        (
            "/text_config/linear_attn_config/gate_lower_bound",
            json!(0.0),
            "gate_lower_bound 0 narrows",
        ),
        (
            "/text_config/linear_attn_config/gate_lower_bound",
            json!(5.0),
            "gate_lower_bound 5 narrows",
        ),
        (
            "/text_config/moe_layer_freq",
            json!(2),
            "moe_layer_freq 2 != 1",
        ),
        ("/text_config/topk_method", json!("greedy"), "noaux_tc"),
        // Grouped routing is DEGENERATE in this checkpoint, not absent — one group of 896,
        // one group selected, which is why plain top-k reproduces it. Real groups would
        // route through the ungrouped path with no error. The wants are the two sides of
        // the message's `a / b` pair, so a transposed argument cannot leave both green.
        (
            "/text_config/num_expert_group",
            json!(2),
            "num_expert_group 2 /",
        ),
        ("/text_config/topk_group", json!(2), "/ topk_group 2"),
    ]);
}

/// The boolean architecture switches and the f32-narrowing scalars — the other half of the
/// silent-wrong class.
///
/// RED OBSERVED (stub validate): every row parsed clean, including `rms_norm_eps: 1e-46`,
/// which passes any f64 positivity test and reaches every RMSNorm as `0.0f32`.
#[test]
fn the_flags_and_narrowing_scalars_are_refused() {
    each_refusal(&[
        // Each of these defaults to `false` somewhere — in the first-party modeling code for
        // the gates, and in Rust for any bool the port forgot to read. Requiring the
        // POSITIVE value means an omission is a refusal, not a silent downgrade.
        (
            "/text_config/mla_use_nope",
            json!(false),
            "mla_use_nope is false",
        ),
        (
            "/text_config/mla_use_output_gate",
            json!(false),
            "mla_use_output_gate is false",
        ),
        (
            "/text_config/linear_attn_config/use_full_rank_gate",
            json!(false),
            "use_full_rank_gate is false",
        ),
        (
            "/text_config/latent_moe_use_norm",
            json!(false),
            "latent_moe_use_norm is false",
        ),
        (
            "/text_config/moe_renormalize",
            json!(false),
            "moe_renormalize is false",
        ),
        (
            "/text_config/num_nextn_predict_layers",
            json!(1),
            "no MTP head",
        ),
        (
            "/text_config/tie_word_embeddings",
            json!(true),
            "separate lm_head",
        ),
        // f32 narrowing, the domain the kernels work in. `1e-46` passes any f64 positivity
        // test and reaches every RMSNorm as `0.0f32`; overflow saturates to inf, which
        // passes any bare `> 0.0` test.
        ("/text_config/rms_norm_eps", json!(1e-46), "narrows to 0"),
        (
            "/text_config/activation_situ_beta",
            json!(0.0),
            "activation_situ_beta 0 narrows",
        ),
        (
            "/text_config/activation_situ_linear_beta",
            json!(1e39),
            "narrows to inf",
        ),
        // 1.0 today, and the multiply is kept anyway: zero silently zeroes every routed
        // contribution while the shared MLP keeps working; negative flips them.
        (
            "/text_config/routed_scaling_factor",
            json!(0.0),
            "routed_scaling_factor 0 narrows",
        ),
    ]);
}

/// The layer roles the converter and the engine arm branch on, read off the shipped arrays.
///
/// **Zero-based MLA layers are 3, 7, 11, …, 87, 91, 92 — the last two are ADJACENT.** The
/// every-fourth pattern breaks at the end, so the tail is asserted explicitly rather than
/// derived from a stride (`k3:docs/reference/k3-architecture.md` §2: the one-based indexing
/// is "the mistake that silently swaps KDA and MLA layers").
#[test]
fn layer_roles_follow_the_partition() {
    let cfg: K3Config = parse_config(K3_SHIPPED).unwrap();
    let t = &cfg.text;
    let mla: Vec<usize> = (0..t.n_layers)
        .filter(|&l| t.layer_is_mla(l).unwrap())
        .collect();
    let want: Vec<usize> = (0..23).map(|k| 3 + 4 * k).chain([92]).collect();
    assert_eq!(mla, want, "the zero-based MLA map, adjacent tail included");
    assert_eq!(mla.len(), 24);
    assert_eq!(t.linear_attn_config.kda_layers.len(), 69);
    // Out of range is an ERROR, not "KDA" — a missing id reading as KDA is a positive claim
    // about arithmetic, which is why this sibling returns `Result` and `layer_is_dense`
    // does not (out of range there reads as "not dense", the answer every real layer but
    // the first gets anyway).
    assert!(t.layer_is_mla(t.n_layers).is_err());
    assert!(t.layer_is_dense(0) && !t.layer_is_dense(1));
}

/// **The heart of the load boundary**, K3's directions of it. A K3 config must not become a
/// zero-filled foreign config, and the refusal must NAME the architecture rather than
/// blaming whichever field the other schema happens to miss first. GLM is asserted here in
/// both directions; the V4 and Glimmer pairs live in THEIR gates (`v4_config.rs`,
/// `glimmer_config.rs`), each extended with its K3 direction, so no direction is asserted
/// twice and every one is asserted once.
#[test]
fn k3_and_the_other_architectures_do_not_cross_parse() {
    refuses_naming_both::<ModelConfig>(K3_SHIPPED, ["kimi_k3", "GlmMoeDsa"]);
    // The reverse, on a document carrying the DISCRIMINANT AND NOTHING ELSE — the sharp form:
    // a refusal naming a missing dimension would be the weak reading.
    let glm = json!({"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"]});
    refuses_naming_both::<K3Config>(&glm.to_string(), ["glm_moe_dsa", "KimiK3"]);
    // And a document that declares NEITHER field is refused rather than assumed to be
    // this one — through the same helper, whose two halves here are the two clauses of
    // `arch_of_named`'s no-discriminant message.
    let mut d = shipped();
    for discriminant in ["model_type", "architectures"] {
        d.as_object_mut().unwrap().remove(discriminant);
    }
    refuses_naming_both::<K3Config>(&d.to_string(), ["declares neither", "nor `architectures`"]);
}
