//! The Inkling 2 weight set: [`place_layer`] resolves one decoder layer's DENSE weights — the
//! attention projections and their norm/sconv children, the sink and relative-bias triple, the
//! input/post-attention norms, the two post convolutions, the query banks, the router, and (on a
//! looped layer) the four MessageGates — to device addresses.
//!
//! The Mini artifact stores the text stack bf16 (the converter dequantized the NVFP4/MXFP8
//! experts), so every dense projection is a [`Bf16Weight`]. The routed experts (`gate_up_proj`,
//! `down_proj`) are NOT resolved here: they are the streamed unit, and the per-layer expert
//! partition against the budget (P6, [`rivoli_core::residency::partition`], INV-8) lands with the
//! decode loop that spends them. Nothing in this file derives placement.

use super::names;
use crate::device::DeviceTier;
use crate::resident::{Bf16Weight, place_bf16, place_f32};
use anyhow::Result;
use core::ptr;
use rivoli_artifact::format::Safetensors;

/// One decoder layer's short-convolution form. The SLIDING layers carry a self-contained static
/// `conv1d` per k/v side; the FULL layers carry the dynamic pair — a static tap bank plus an input
/// projection, both per side (`Inkling2DynamicShortConvolution`). Two arms rather than four
/// nullable pointers, so the forward cannot run the wrong conv on a layer it mis-classified.
#[derive(Clone, Copy)]
pub enum LayerSconv {
    /// `[kv_heads·head_dim, 1, k]` per side; divisor `√(k+1)`.
    Static { k: *const f32, v: *const f32 },
    /// `static_weight` `[k, kv_heads·head_dim]` and `weight_proj` `[kv_heads·k, hidden]` per side;
    /// divisor `√(2k+1)`.
    Dynamic {
        k_static: *const f32,
        k_proj: Bf16Weight,
        v_static: *const f32,
        v_proj: Bf16Weight,
    },
}

/// One MessageGate (`Inkling2MessageGate`, MOD:196): `down` is `[gate_rank, 2H]`, `forget` and
/// `gain` are `[H, gate_rank]`. The combine is the `message_gate` kernel's; these are its GEMVs.
pub struct MessageGatePin {
    pub down: Bf16Weight,
    pub forget: Bf16Weight,
    pub gain: Bf16Weight,
}

/// One Inkling decoder layer's DENSE weights, resolved to device addresses.
///
/// Not `Copy`: a streamed layer's addresses must not outlive a slot refill (the glimmer pin's
/// argument). The forward borrows the layer across its launches.
pub struct InklingLayerPin {
    /// The sliding layer's single `input_layernorm`, or the full layer's three q/k/v input norms
    /// (T11). `full` selects which; the unused slots are null.
    pub input_ln: [*const f32; 3],
    /// `post_attention_layernorm` — the pre-MLP norm.
    pub post_attn_ln: *const f32,
    /// `q_proj.linear` `[num_heads·head_dim, hidden]`.
    pub q: Bf16Weight,
    /// `q_proj.norm.scale` `[num_heads·head_dim]` — the learned L2Clip scale.
    pub q_norm_scale: *const f32,
    /// `k_proj.linear` `[kv_heads·head_dim, hidden]`.
    pub k: Bf16Weight,
    /// `v_proj.linear`, same shape as [`Self::k`], separable only by name.
    pub v: Bf16Weight,
    /// The k/v short convolution in this layer's form — static on sliding, dynamic on full.
    pub sconv: LayerSconv,
    /// `o_proj.gate_proj` `[num_heads·head_dim, hidden]` — the sigmoid output gate.
    pub attn_gate: Bf16Weight,
    /// `o_proj.linear` `[hidden, num_heads·head_dim]`.
    pub o: Bf16Weight,
    /// `sink_proj` `[num_heads, hidden]` — one sink logit per head, read on-device by
    /// `sink_attend`.
    pub sink: Bf16Weight,
    /// `relative_position_bias.linear` `[num_heads·d_rel, hidden]`.
    pub rel_linear: Bf16Weight,
    /// `relative_position_bias.proj` `[rel_extent, d_rel]` — stored TRANSPOSED (the converter emits
    /// it verbatim: 1024×16), because `gemm_bf16` computes `x·wᵀ` and the rel-logits GEMM is
    /// `[heads, d_rel] · [d_rel, rel_extent]`.
    pub rel_proj: Bf16Weight,
    /// `relative_position_bias.norm.scale` `[num_heads·d_rel]`.
    pub rel_norm_scale: *const f32,
    /// `post_attention_convolution.conv1d` `[hidden, 1, k]`.
    pub post_attn_conv: *const f32,
    /// `post_mlp_convolution.conv1d` `[hidden, 1, k]`.
    pub post_mlp_conv: *const f32,
    /// `mlp_input.query` `[hidden]` — the block-residual MLP query bank.
    pub mlp_query: *const f32,
    /// `attention_input.query` — `[hidden]` on a sliding layer, `[3, hidden]` on a full one;
    /// present on EVERY layer `i > 0`, null only on layer 0.
    pub attn_query: *const f32,
    /// `mlp.gate.weight` `[n_experts, hidden]` — the router.
    pub router: Bf16Weight,
    /// `mlp.gate.e_score_correction_bias` `[n_experts]` — selection-only bias (T5).
    pub router_bias: *const f32,
    /// The four MessageGates — `Some` exactly on a looped layer (`core_start <= i < core_end`).
    pub message_gates: Option<[MessageGatePin; 4]>,
}

/// One MessageGate's three projections, resolved from `<prefix>.{down,forget,message}_proj`.
fn place_message_gate(
    tier: &mut DeviceTier,
    st: &Safetensors,
    prefix: &str,
) -> Result<MessageGatePin> {
    let n = |suffix: &str| format!("{prefix}.{suffix}");
    Ok(MessageGatePin {
        down: place_bf16(tier, st, &n(names::MESSAGE_GATE_SUFFIXES[0]))?,
        forget: place_bf16(tier, st, &n(names::MESSAGE_GATE_SUFFIXES[1]))?,
        gain: place_bf16(tier, st, &n(names::MESSAGE_GATE_SUFFIXES[2]))?,
    })
}

/// Place one layer's input norms (three on full layers, one on sliding) and k/v sconv weights.
fn place_layer_norms_and_sconv(
    tier: &mut DeviceTier,
    st: &Safetensors,
    p: &str,
    full: bool,
) -> Result<([*const f32; 3], LayerSconv)> {
    let n = |suffix: &str| format!("{p}.{suffix}");
    if full {
        let input_ln = [
            place_f32(tier, st, &n(names::FULL_INPUT_NORMS[0]))?,
            place_f32(tier, st, &n(names::FULL_INPUT_NORMS[1]))?,
            place_f32(tier, st, &n(names::FULL_INPUT_NORMS[2]))?,
        ];
        let sconv = LayerSconv::Dynamic {
            k_static: place_f32(tier, st, &n(names::K_SCONV_TAPS))?,
            k_proj: place_bf16(tier, st, &n(names::K_SCONV_PROJ))?,
            v_static: place_f32(tier, st, &n(names::V_SCONV_TAPS))?,
            v_proj: place_bf16(tier, st, &n(names::V_SCONV_PROJ))?,
        };
        return Ok((input_ln, sconv));
    }
    let input_ln = [
        place_f32(tier, st, &n(names::SLIDING_INPUT_NORM))?,
        ptr::null(),
        ptr::null(),
    ];
    let sconv = LayerSconv::Static {
        k: place_f32(tier, st, &n(names::K_SCONV_STATIC))?,
        v: place_f32(tier, st, &n(names::V_SCONV_STATIC))?,
    };
    Ok((input_ln, sconv))
}

/// Resolve one layer's dense weights from the artifact. `flags` is `(full, looped)` — both derived
/// by the caller from `geometry`'s rules (`is_block_end` and the loop core), never guessed here;
/// `full` selects the three q/k/v input norms (T11) and the dynamic sconv, `looped` the four
/// MessageGates. The unused slots are null. Each projection's dims ride the [`Bf16Weight`] from the
/// tensor's own shape, so a mis-shaped artifact fails at load (`place_bf16`'s `dims2`), not at
/// launch.
pub fn place_layer(
    tier: &mut DeviceTier,
    st: &Safetensors,
    i: usize,
    flags: (bool, bool),
) -> Result<InklingLayerPin> {
    let (full, looped) = flags;
    let p = names::layer_prefix(i);
    let n = |suffix: &str| format!("{p}.{suffix}");
    let (input_ln, sconv) = place_layer_norms_and_sconv(tier, st, &p, full)?;
    let message_gates = if looped {
        Some([
            place_message_gate(tier, st, &n(names::MESSAGE_GATE_PREFIXES[0]))?,
            place_message_gate(tier, st, &n(names::MESSAGE_GATE_PREFIXES[1]))?,
            place_message_gate(tier, st, &n(names::MESSAGE_GATE_PREFIXES[2]))?,
            place_message_gate(tier, st, &n(names::MESSAGE_GATE_PREFIXES[3]))?,
        ])
    } else {
        None
    };
    Ok(InklingLayerPin {
        input_ln,
        post_attn_ln: place_f32(tier, st, &n(names::POST_ATTN_NORM))?,
        q: place_bf16(tier, st, &n(names::Q_LINEAR))?,
        q_norm_scale: place_f32(tier, st, &n(names::Q_NORM))?,
        k: place_bf16(tier, st, &n(names::K_LINEAR))?,
        v: place_bf16(tier, st, &n(names::V_LINEAR))?,
        sconv,
        attn_gate: place_bf16(tier, st, &n(names::O_GATE))?,
        o: place_bf16(tier, st, &n(names::O_LINEAR))?,
        sink: place_bf16(tier, st, &n(names::SINK))?,
        rel_linear: place_bf16(tier, st, &n(names::REL_LINEAR))?,
        rel_proj: place_bf16(tier, st, &n(names::REL_PROJ))?,
        rel_norm_scale: place_f32(tier, st, &n(names::REL_NORM))?,
        post_attn_conv: place_f32(tier, st, &n(names::POST_ATTN_CONV))?,
        post_mlp_conv: place_f32(tier, st, &n(names::POST_MLP_CONV))?,
        mlp_query: place_f32(tier, st, &n(names::MLP_INPUT_QUERY))?,
        attn_query: if i > 0 {
            place_f32(tier, st, &n(names::ATTENTION_INPUT_QUERY))?
        } else {
            ptr::null()
        },
        router: place_bf16(tier, st, &n(names::ROUTER))?,
        router_bias: place_f32(tier, st, &n(names::ROUTER_BIAS))?,
        message_gates,
    })
}
