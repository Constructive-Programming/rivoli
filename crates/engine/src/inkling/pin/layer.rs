//! One decoder layer's dense placement: the input norms and the k/v short convolution, the four
//! MessageGates, and the five large attention projections that may be bf16 or fp8.
//!
//! A child module of [`super`] for the 800-line soft cap, and the cohesion line is "resolve ONE
//! layer": the parent owns the whole pin (the globals, every layer, the routed experts'
//! residency), `dense` owns what the projections are stored IN, and this file is the walk that
//! reads the two together. It is `pub use`d from the parent, so `pin::place_layer` is still the
//! one path a reader follows.

use super::dense::place_dense_proj;
use super::*;
use core::ptr;

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

/// What one layer's placement is told about that layer: which one it is, its two `geometry` facts,
/// and the artifact's dense fp8 block (`None` on a bf16 trunk).
///
/// A struct rather than four arguments because [`place_layer`] would otherwise carry five and the
/// two `bool`s are a transposition nothing type-checks: `full` and `looped` differ by one letter
/// and select four different tensor sets between them.
#[derive(Clone, Copy)]
pub struct LayerSpec {
    pub layer: usize,
    pub full: bool,
    pub looped: bool,
    pub fp8_block: Option<usize>,
}

/// Resolve one layer's dense weights from the artifact. `full`/`looped` are derived by the caller
/// from `geometry`'s rules (`is_block_end` and the loop core), never guessed here; `full` selects
/// the three q/k/v input norms (T11) and the dynamic sconv, `looped` the four MessageGates. The
/// unused slots are null. Each projection's dims ride the weight from the tensor's own shape, so a
/// mis-shaped artifact fails at load (`place_bf16`'s `dims2`), not at launch — and the five large
/// attention projections come back as [`DenseWeight`]s, bf16 or fp8 by the artifact's own header.
pub fn place_layer(
    tier: &mut DeviceTier,
    st: &Safetensors,
    spec: LayerSpec,
) -> Result<InklingLayerPin> {
    let LayerSpec {
        layer: i,
        full,
        looped,
        fp8_block,
    } = spec;
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
        q: place_dense_proj(tier, st, &n(names::Q_LINEAR), fp8_block)?,
        q_norm_scale: place_f32(tier, st, &n(names::Q_NORM))?,
        k: place_dense_proj(tier, st, &n(names::K_LINEAR), fp8_block)?,
        v: place_dense_proj(tier, st, &n(names::V_LINEAR), fp8_block)?,
        sconv,
        attn_gate: place_dense_proj(tier, st, &n(names::O_GATE), fp8_block)?,
        o: place_dense_proj(tier, st, &n(names::O_LINEAR), fp8_block)?,
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
        message_gates,
    })
}
