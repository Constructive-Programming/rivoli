//! Inkling 2's per-layer weight names — the artifact's storage spelling (the converter's rename
//! TARGET census, `docs/measurement/inkling2-reference/mini/text-model-names.tsv`). The pin resolves
//! these against the artifact; the forward reads the resolved addresses. **Touching no device.**
//!
//! The names are the first-party `Inkling2LoopForCausalLM` state_dict keys, NOT the checkpoint's
//! `model.llm.*` spelling — the converter already renamed them (S2). Every spelling exists ONCE,
//! as the `pub const` below; the grouped suffix lists are built from those constants and
//! [`layer_tensor_names`] is the one expansion, so the pin, the forward and the census test cannot
//! drift on a name.

/// The layer prefix: `model.layers.{i}`.
pub fn layer_prefix(i: usize) -> String {
    format!("model.layers.{i}")
}

// ── attention ───────────────────────────────────────────────────────────────────────────────

/// `q_proj.linear` `[num_heads·head_dim, hidden]`.
pub const Q_LINEAR: &str = "self_attn.q_proj.linear.weight";
/// `q_proj.norm.scale` `[num_heads·head_dim]` — the learned L2Clip scale.
pub const Q_NORM: &str = "self_attn.q_proj.norm.scale";
/// `k_proj.linear` `[kv_heads·head_dim, hidden]`.
pub const K_LINEAR: &str = "self_attn.k_proj.linear.weight";
/// `k_proj.sconv.conv1d` `[kv_heads·head_dim, 1, k]` — SLIDING layers only.
pub const K_SCONV_STATIC: &str = "self_attn.k_proj.sconv.conv1d.weight";
/// `k_proj.sconv.static_weight` `[k, kv_heads·head_dim]` — FULL layers only.
pub const K_SCONV_TAPS: &str = "self_attn.k_proj.sconv.static_weight";
/// `k_proj.sconv.weight_proj` `[kv_heads·k, hidden]` — FULL layers only.
pub const K_SCONV_PROJ: &str = "self_attn.k_proj.sconv.weight_proj.weight";
/// `v_proj.linear`, same shape as [`K_LINEAR`], separable only by name.
pub const V_LINEAR: &str = "self_attn.v_proj.linear.weight";
/// `v_proj.sconv.conv1d` — SLIDING layers only.
pub const V_SCONV_STATIC: &str = "self_attn.v_proj.sconv.conv1d.weight";
/// `v_proj.sconv.static_weight` — FULL layers only.
pub const V_SCONV_TAPS: &str = "self_attn.v_proj.sconv.static_weight";
/// `v_proj.sconv.weight_proj` — FULL layers only.
pub const V_SCONV_PROJ: &str = "self_attn.v_proj.sconv.weight_proj.weight";
/// `o_proj.gate_proj` `[num_heads·head_dim, hidden]` — the sigmoid output gate.
pub const O_GATE: &str = "self_attn.o_proj.gate_proj.weight";
/// `o_proj.linear` `[hidden, num_heads·head_dim]`.
pub const O_LINEAR: &str = "self_attn.o_proj.linear.weight";
/// `sink_proj` `[num_heads, hidden]` — one sink logit per head.
pub const SINK: &str = "self_attn.sink_proj.weight";
/// `relative_position_bias.proj` `[rel_extent, d_rel]` — stored TRANSPOSED for `gemm_bf16`.
pub const REL_PROJ: &str = "self_attn.relative_position_bias.proj";
/// `relative_position_bias.linear` `[num_heads·d_rel, hidden]`.
pub const REL_LINEAR: &str = "self_attn.relative_position_bias.linear.weight";
/// `relative_position_bias.norm.scale` `[num_heads·d_rel]`.
pub const REL_NORM: &str = "self_attn.relative_position_bias.norm.scale";

// ── MoE, norms and convolutions ────────────────────────────────────────────────────────────

/// `mlp.experts.gate_up_proj` — the streamed expert weight (no `.weight` suffix).
pub const EXPERTS_GATE_UP: &str = "mlp.experts.gate_up_proj";
/// `mlp.experts.down_proj` — the streamed expert weight (no `.weight` suffix).
pub const EXPERTS_DOWN: &str = "mlp.experts.down_proj";
/// `mlp.gate.weight` `[n_experts, hidden]` — the router.
pub const ROUTER: &str = "mlp.gate.weight";
/// `mlp.gate.e_score_correction_bias` `[n_experts]` — selection-only bias (T5).
pub const ROUTER_BIAS: &str = "mlp.gate.e_score_correction_bias";
/// `post_attention_layernorm` — the pre-MLP norm.
pub const POST_ATTN_NORM: &str = "post_attention_layernorm.weight";
/// `post_attention_convolution.conv1d` `[hidden, 1, k]`.
pub const POST_ATTN_CONV: &str = "post_attention_convolution.conv1d.weight";
/// `post_mlp_convolution.conv1d` `[hidden, 1, k]`.
pub const POST_MLP_CONV: &str = "post_mlp_convolution.conv1d.weight";
/// `mlp_input.query` `[hidden]` — the block-residual MLP query bank.
pub const MLP_INPUT_QUERY: &str = "mlp_input.query";

/// The attention suffixes every layer carries: q/k/v/o projections and their norms, the sink, and
/// the relative-position-bias triple. The k/v sconv is NOT here — its spelling differs between the
/// static and dynamic forms ([`SLIDING_SCONV_SUFFIXES`], [`FULL_SCONV_SUFFIXES`]).
pub const ATTN_SHARED_SUFFIXES: &[&str] = &[
    Q_LINEAR, Q_NORM, K_LINEAR, V_LINEAR, O_GATE, O_LINEAR, SINK, REL_PROJ, REL_LINEAR, REL_NORM,
];

/// The SLIDING layers' static short convolution, one `conv1d` per k/v side
/// (`Inkling2ShortConvolution`, MOD:349).
pub const SLIDING_SCONV_SUFFIXES: &[&str] = &[K_SCONV_STATIC, V_SCONV_STATIC];

/// The FULL layers' dynamic short convolution (`Inkling2DynamicShortConvolution`, MOD:412): a
/// static k·kv_heads tap bank plus a `[kv_heads·k, hidden]` input projection, per k/v side.
pub const FULL_SCONV_SUFFIXES: &[&str] = &[K_SCONV_TAPS, K_SCONV_PROJ, V_SCONV_TAPS, V_SCONV_PROJ];

/// The per-layer MoE, convolution and query-bank suffixes. The input norm is NOT here: sliding
/// layers carry [`SLIDING_INPUT_NORM`], full layers the three [`FULL_INPUT_NORMS`].
pub const MLP_SHARED_SUFFIXES: &[&str] = &[
    EXPERTS_GATE_UP,
    EXPERTS_DOWN,
    ROUTER,
    ROUTER_BIAS,
    POST_ATTN_NORM,
    POST_ATTN_CONV,
    POST_MLP_CONV,
    MLP_INPUT_QUERY,
];

/// The block-residual query bank — present on **every layer `i > 0`**, absent only on layer 0
/// (the boundary reset opens the first bank without consuming one, T7). Its width follows the
/// attention form: `[hidden]` on a sliding layer, `[3, hidden]` on a full one (T11).
pub const ATTENTION_INPUT_QUERY: &str = "attention_input.query";

/// The sliding layers' single input norm.
pub const SLIDING_INPUT_NORM: &str = "input_layernorm.weight";

/// [`SLIDING_INPUT_NORM`] as the one-element list the layer expansion folds with
/// [`FULL_INPUT_NORMS`] — the same shape, so the two branches of the name walk are one expression.
pub const SLIDING_INPUT_NORMS: &[&str] = &[SLIDING_INPUT_NORM];

/// The full layers' three distinct input norms (T11) — `q/k/v_input_layernorm` replace the sliding
/// `input_layernorm`. The forward picks by `is_block_end`.
pub const FULL_INPUT_NORMS: &[&str] = &[
    "q_input_layernorm.weight",
    "k_input_layernorm.weight",
    "v_input_layernorm.weight",
];

/// The looped layers' four MessageGate prefixes (MOD:196). Each gate is a
/// [`MESSAGE_GATE_SUFFIXES`] triple: `down_proj` (2H → `gate_rank`), `forget_gate_proj` and
/// `message_gate_proj` (rank → H).
pub const MESSAGE_GATE_PREFIXES: &[&str] = &[
    "q_message_gate",
    "k_message_gate",
    "v_message_gate",
    "mlp_message_gate",
];

/// The three projections under each [`MESSAGE_GATE_PREFIXES`] entry.
pub const MESSAGE_GATE_SUFFIXES: &[&str] = &[
    "down_proj.weight",
    "forget_gate_proj.weight",
    "message_gate_proj.weight",
];

/// The global (non-layer) weights: the embedding + its norm, the final norm, the final/message
/// query banks, and the lm_head (the artifact spells it `lm_head.weight`, NOT `model.lm_head.*`).
pub const GLOBAL_SUFFIXES: &[&str] = &[
    "model.embed_tokens.weight",
    "model.embed_tokens.embed_norm.weight",
    "model.norm.weight",
    "model.final_input.query",
    "model.message_input.query",
    "lm_head.weight",
];

/// Every stored tensor name for layer `layer`, fully prefixed. `full` is `is_block_end` and
/// `looped` is `core_start <= layer < core_end`; the two decide the sconv form, the input norms,
/// the `attention_input.query` width and the MessageGate presence. Layer 0 is the one layer with
/// no `attention_input.query`, so `layer > 0` is a rule of its own, not a consequence of `full`.
pub fn layer_tensor_names(layer: usize, full: bool, looped: bool) -> Vec<String> {
    let p = layer_prefix(layer);
    // A full layer with its MessageGates is the largest shape: 26 + 12.
    let mut out = Vec::with_capacity(26 + usize::from(looped) * 12);
    let sconv = if full {
        FULL_SCONV_SUFFIXES
    } else {
        SLIDING_SCONV_SUFFIXES
    };
    let norms = if full {
        FULL_INPUT_NORMS
    } else {
        SLIDING_INPUT_NORMS
    };
    for suffixes in [ATTN_SHARED_SUFFIXES, sconv, MLP_SHARED_SUFFIXES, norms] {
        out.extend(suffixes.iter().map(|s| format!("{p}.{s}")));
    }
    if layer > 0 {
        out.push(format!("{p}.{ATTENTION_INPUT_QUERY}"));
    }
    if looped {
        for prefix in MESSAGE_GATE_PREFIXES {
            for suffix in MESSAGE_GATE_SUFFIXES {
                out.push(format!("{p}.{prefix}.{suffix}"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // tests: panic-on-failure is the idiom

    use super::*;

    /// The vendored Mini member's structure the expansion is driven with: 24 layers, block size 6
    /// (full attention at `i = 5, 11, 17, 23`), loop core `[6, 18)`.
    const MINI_LAYERS: usize = 24;
    const MINI_BLOCK: usize = 6;
    const MINI_CORE: (usize, usize) = (6, 18);

    #[test]
    fn layer_names_match_the_census() {
        assert_eq!(layer_prefix(0), "model.layers.0");
        assert_eq!(
            format!("{}.{}", layer_prefix(3), ATTN_SHARED_SUFFIXES[0]),
            "model.layers.3.self_attn.q_proj.linear.weight"
        );
        assert_eq!(
            format!("{}.{}", layer_prefix(0), MLP_SHARED_SUFFIXES[0]),
            "model.layers.0.mlp.experts.gate_up_proj"
        );
    }

    /// Exact set equality against all 693 census names — the expanded per-layer lists plus
    /// [`GLOBAL_SUFFIXES`]. A name the expansion invents and a name it forgets both redden, and
    /// the pre-dedup length catches a duplicate a set comparison would hide.
    #[test]
    fn layer_tensor_names_cover_the_mini_census() {
        let tsv = include_str!(
            "../../../../docs/measurement/inkling2-reference/mini/text-model-names.tsv"
        );
        let mut want: Vec<&str> = tsv
            .lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| l.split('\t').next().unwrap())
            .collect();
        assert_eq!(want.len(), 693, "census rows");
        want.sort_unstable();
        want.dedup();
        assert_eq!(want.len(), 693, "the census carries a duplicate name");

        let mut got: Vec<String> = GLOBAL_SUFFIXES.iter().map(|s| (*s).to_owned()).collect();
        for layer in 0..MINI_LAYERS {
            let full = (layer + 1).is_multiple_of(MINI_BLOCK);
            let looped = (MINI_CORE.0..MINI_CORE.1).contains(&layer);
            got.extend(layer_tensor_names(layer, full, looped));
        }
        assert_eq!(got.len(), 693, "expanded names, before dedup");
        got.sort_unstable();
        got.dedup();
        assert_eq!(
            got.iter().map(String::as_str).collect::<Vec<_>>(),
            want,
            "the expansion and the census disagree — diff the two sorted lists"
        );
    }
}
