//! Muse Glimmer checkpoint naming. Forward-ported from `old:src/artifact/model.rs`
//! (`wt/glimmer-s2` @ 6b7f496) ahead of the rest of the Glimmer config, because the
//! anchor fixture-integrity gate (`crates/oracles/tests/glimmer_anchor.rs`) names every
//! expected tensor and this is the ONE statement of that list. The rest of the per-model
//! config arrives with M7; these constants arrive first so the vendored anchors are gated
//! from commit 2 onward.

/// The twelve tensors every Muse Glimmer decoder layer ships, as `<layer prefix>.{}.weight`.
///
/// **One statement of this fact, read by both the converter and its tests.** In the old
/// tree they had a copy each until jscpd reported it 2026-08-11, and a shared list of
/// *names* is exactly the thing that must not be duplicated: two copies can disagree, and
/// a name that exists but points at the wrong tensor copies silently — `k3_names.rs`
/// exists there because that failure mode already cost a round.
///
/// **Eight projections and four norms** — five projections in the attention block (`q`,
/// `k`, `v`, `o` and the output `gate`) and three in the MLP. The QK-norm is weightless
/// and ships nothing, and there is no bias anywhere (`attention_bias` is false and
/// asserted).
///
/// > **CORRECTED 2026-08-11** in the old tree, by review. It said "five projections and
/// > four norms", which is nine against a list of twelve — it counted the attention block
/// > and forgot the MLP, while the pin had it right, so the tree disagreed with itself
/// > about the length of the one constant that exists to stop exactly that.
pub const GLIMMER_LAYER_TENSORS: [&str; 12] = [
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.gate_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// The prefix Glimmer's text-side tensors carry. The `language_model.` segment is the
/// multimodal wrapper's, and it is on every text tensor — K3's port records the same
/// shape as a name nothing in its documentation mentioned.
pub const GLIMMER_LAYER_PREFIX: &str = "model.language_model.layers";
