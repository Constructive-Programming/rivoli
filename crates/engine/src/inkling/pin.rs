//! The Inkling 2 weight set: [`place_layer`] resolves one decoder layer's DENSE weights — the
//! attention projections and their norm/sconv children, the sink and relative-bias triple, the
//! input/post-attention norms, the two post convolutions, the query banks, the router, and (on a
//! looped layer) the four MessageGates — to device addresses.
//!
//! Every dense projection is a bf16 [`Bf16Weight`] on both members. The routed experts are the
//! STREAMED unit, and WHO streams them is the one thing the backends differ in
//! ([`ExpertBackend`]): Mini's converted artifact holds them dequantized bf16, so
//! [`InklingPin::build`] asks [`rivoli_core::residency::partition`] how many whole layers fit
//! (P6, INV-8) and reads ONE expert per routing decision into a single-expert staging slot;
//! Small's are native `.mxfp8` and stream through [`RoutedPool`] (the `mxfp8` child module).

use super::geometry::{
    ScratchSizes, expert_layer_bytes, global_weight_bytes, is_block_end, single_expert_bytes,
};
use super::names;
use crate::device::{DeviceBuf, DeviceTier};
use crate::resident::{Bf16Weight, place_bf16, place_f32};
use crate::routed::RoutedPool;
use crate::seam::Fetched;
use anyhow::{Context, Result, bail, ensure};
use core::ptr;
use rivoli_artifact::format::{Dtype, ExpertSet, Safetensors};
use rivoli_artifact::inkling_config::InklingTextConfig;
use rivoli_core::residency::{Bytes, Floor, Refusal, Unit, UnitId, partition};
use std::num::NonZeroU64;

/// The 256-byte alignment [`DeviceTier`]'s bump cursor rounds every placement up to, paid once as
/// slack on the floor and on the tier request so the last placement cannot push past the slab.
/// The artifact holds 822 tensors, so the worst case the alignment can lose is
/// `822 · 255 B ≈ 205 KB`; 4 MiB is ~20× that, and the 64 MiB this replaced charged the floor
/// for 64 MiB no placement could ever consume.
const PIN_SLACK: usize = 4 << 20;

/// Small's `.mxfp8` construction — the pool-backed half of this pin, split out when the file
/// crossed the workspace's 800-line soft cap. It is a child module so it can read this file's
/// private placers and its [`PIN_SLACK`] without widening one visibility.
mod mxfp8;

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
    /// `mlp.gate.weight` `[n_experts, hidden]` — the router. Its `e_score_correction_bias` is
    /// HOST-ONLY (`router_bias_host`, T5 selection): routing is host math, so a device copy had no
    /// reader and was placed (and paid for) on every layer for nothing.
    pub router: Bf16Weight,
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
        message_gates,
    })
}

// ── the whole pin: globals, dense layers, and the routed experts' placement ─────────────────

/// The six GLOBAL tensors: the embedding and its norm, the final norm, the final and message
/// query banks, and the lm_head (T14: the head keeps the live rows, the embed keeps them all).
pub struct InklingGlobalPin {
    pub embed: Bf16Weight,
    pub embed_norm: *const f32,
    pub final_norm: *const f32,
    pub final_query: *const f32,
    /// `model.message_input.query` `[core, 4, hidden]` — null on a member that does not loop.
    pub message_query: *const f32,
    pub lm_head: Bf16Weight,
}

/// Where one layer's routed experts live. `Resident` names the layer's whole `[E, 2I, H]` and
/// `[E, H, I]` slabs in the tier; `Streamed` names the artifact tensors a single expert is sliced
/// out of on every routing decision, into the pin's one-expert staging slot.
pub enum ExpertSource {
    Resident {
        gate_up: *const u16,
        down: *const u16,
    },
    Streamed {
        gate_up_name: String,
        down_name: String,
    },
}

/// Where a member's routed experts live — the ONE thing the two members do not share.
///
/// **Mini and Small run the same text stack** (attention, short convolutions, block-residual
/// quarries, router, decode and score loops); they differ in the container the experts are
/// stored in, and therefore in who owns residency. Mini's converted artifact holds them
/// DEQUANTIZED to bf16, so the pin slices one expert out of an mmap'd slab on every routing
/// decision. Small's artifact holds ~600 GB of native `.mxfp8` slots, far past this box's
/// memory, so residency is [`RoutedPool`]'s: O_DIRECT fills behind per-expert tickets,
/// overlapped with the resident experts' compute on two streams.
pub enum ExpertBackend {
    /// Mini: whole-layer slabs placed by [`partition`]'s prefix, streamed by host slice.
    Bf16 {
        experts: Vec<ExpertSource>,
        /// The one-expert staging slots, allocated only when some layer streams.
        w13_slot: Option<DeviceBuf>,
        w2_slot: Option<DeviceBuf>,
        pinned_layers: usize,
        /// Expert-slot hits and fills, at this backend's granularity (one expert).
        fetched: Fetched,
    },
    /// Small: the `.mxfp8` set behind the shared routed pool. The pool's own counters are the
    /// authority for [`InklingPin::fetched`] — a second tally here could disagree with it.
    MxFp8 {
        /// Keeps the read table's fds open for the pool's lifetime.
        _set: ExpertSet,
        routed: Box<RoutedPool>,
        pinned_experts: usize,
        total_experts: usize,
    },
}

/// What [`InklingPin::assemble`] needs to place the resident set — bundled because it is four
/// values travelling together into one call and two of them are byte counts a positional list
/// would let transpose.
pub(super) struct Parts<'a> {
    pub text: &'a InklingTextConfig,
    pub st: Safetensors,
    pub flags: &'a [(bool, bool)],
    pub tier_bytes: usize,
}

/// One Inkling 2 member's weight set: the placed globals and dense layers, and the routed
/// experts' residency — Mini's layer prefix or Small's routed pool.
///
/// **The placement is [`partition`]`/`PoolPlan`'s, never this file's.** The units are whole
/// layers for Mini (the dense weights are always resident; the expert slabs are the streamable
/// mass) and single experts for Small (the unit the `.mxfp8` container streams), the floor is
/// stated from `geometry`, and the returned prefix is executed verbatim (P6, INV-8).
pub struct InklingPin {
    /// RAII owner of the slab every pointer below points into. Small's tier holds the globals
    /// and dense weights only; its experts live in the pool's arena.
    _tier: DeviceTier,
    /// The mmap'd resident artifact. Mini's streamed layers read their expert slices from it;
    /// Small's dense placement does too at load, and never again.
    st: Safetensors,
    pub globals: InklingGlobalPin,
    pub layers: Vec<InklingLayerPin>,
    /// Each layer's `mlp.gate.e_score_correction_bias`, host-side — `route_into` takes `&[f32]`,
    /// and the bias shifts SELECTION only (T5).
    pub router_bias_host: Vec<Vec<f32>>,
    backend: ExpertBackend,
    pub hidden: usize,
    pub inter: usize,
    n_experts: usize,
}

/// `[E, 2I, H]` and `[E, H, I]` for layer `l`.
fn expert_names(l: usize) -> (String, String) {
    let p = names::layer_prefix(l);
    (
        format!("{p}.{}", names::EXPERTS_GATE_UP),
        format!("{p}.{}", names::EXPERTS_DOWN),
    )
}

/// The dense (non-expert) byte total over every layer, read off the artifact's own headers.
///
/// **The artifact's lengths, not `dense_layer_elements` × a dtype.** The per-layer census formula
/// counts ELEMENTS across two storage widths (bf16 projections, f32 norms), so multiplying it by
/// either width is wrong by the other one's share; the header is exact by construction and is the
/// same tensor list [`place_layer`] places.
fn dense_total(st: &Safetensors, flags: &[(bool, bool)]) -> Result<usize> {
    let mut n = 0;
    for (l, &(full, looped)) in flags.iter().enumerate() {
        for name in names::layer_tensor_names(l, full, looped) {
            if name.contains(".mlp.experts.") {
                continue; // the routed unit: `expert_layer_bytes`
            }
            n += st.raw(&name)?.0.len();
        }
    }
    Ok(n)
}

/// The global tensors' byte total, read off the artifact — equal to
/// [`global_weight_bytes`]'s derived figure, which is what the floor uses.
fn globals_bytes(st: &Safetensors, text: &InklingTextConfig) -> Result<usize> {
    let mut n = 0;
    for (i, name) in names::GLOBAL_SUFFIXES.iter().enumerate() {
        if i == 4 && text.loop_config.is_none() {
            continue; // message_input.query is the looped member's alone
        }
        n += st.raw(name)?.0.len();
    }
    Ok(n)
}

/// The residency units: one whole layer's expert slabs, in layer order. Layer order is the
/// priority order because decode's access is exactly cyclic over it.
fn expert_units(n_layers: usize, unit: usize) -> Result<Vec<Unit>> {
    let bytes = NonZeroU64::new(unit as u64).context("a zero-byte expert layer")?;
    Ok((0..n_layers)
        .map(|l| Unit {
            id: UnitId(l as u32),
            bytes,
        })
        .collect())
}

/// Place the six global tensors.
fn place_globals(
    tier: &mut DeviceTier,
    st: &Safetensors,
    text: &InklingTextConfig,
) -> Result<InklingGlobalPin> {
    let g = |i: usize| names::GLOBAL_SUFFIXES[i];
    let embed = place_bf16(tier, st, g(0))?;
    ensure!(
        embed.o_dim == text.vocab_size,
        "embed_tokens has {} rows against the config's {} — the embed is never unpadded (T14)",
        embed.o_dim,
        text.vocab_size
    );
    let lm_head = place_bf16(tier, st, g(5))?;
    ensure!(
        lm_head.o_dim == text.unpadded_vocab_size,
        "lm_head has {} rows against the config's {} live rows (T14)",
        lm_head.o_dim,
        text.unpadded_vocab_size
    );
    Ok(InklingGlobalPin {
        embed,
        embed_norm: place_f32(tier, st, g(1))?,
        final_norm: place_f32(tier, st, g(2))?,
        final_query: place_f32(tier, st, g(3))?,
        message_query: if text.loop_config.is_some() {
            place_f32(tier, st, g(4))?
        } else {
            ptr::null()
        },
        lm_head,
    })
}

/// Place one layer's whole expert slabs; the shape check is against the config's own
/// `(experts, inter, hidden)`, so a transposed slab fails at load rather than at a launch.
fn place_experts(
    tier: &mut DeviceTier,
    st: &Safetensors,
    l: usize,
    dims: (usize, usize, usize),
) -> Result<ExpertSource> {
    let (e, inter, hidden) = dims;
    let (gu_name, dn_name) = expert_names(l);
    let (gu, shape) = st.typed(&gu_name, Dtype::Bf16)?;
    ensure!(
        shape == &[e, 2 * inter, hidden][..],
        "{gu_name} is {shape:?}, not [{e}, {}, {hidden}]",
        2 * inter
    );
    let gate_up = tier.place(gu)? as *const u16;
    let (dn, shape) = st.typed(&dn_name, Dtype::Bf16)?;
    ensure!(
        shape == &[e, hidden, inter][..],
        "{dn_name} is {shape:?}, not [{e}, {hidden}, {inter}]"
    );
    Ok(ExpertSource::Resident {
        gate_up,
        down: tier.place(dn)? as *const u16,
    })
}

impl InklingPin {
    /// Whether this pin streamed its experts through [`RoutedPool`] — Small's backend.
    ///
    /// The engine asks this ONCE, at construction, to decide whether to allocate the two
    /// streams and the descriptor table the pool's dispatch needs: Mini's loop must pay no
    /// byte and no accounting for a path it never takes.
    pub fn is_mxfp8(&self) -> bool {
        matches!(self.backend, ExpertBackend::MxFp8 { .. })
    }

    /// The routed pool, on Small's backend only. `None` on Mini, whose experts are placed
    /// slabs rather than pool slots.
    pub fn routed_mut(&mut self) -> Option<&mut RoutedPool> {
        match &mut self.backend {
            ExpertBackend::MxFp8 { routed, .. } => Some(routed),
            ExpertBackend::Bf16 { .. } => None,
        }
    }

    /// How many units the budget pinned, and how many stream — in THIS backend's granularity:
    /// whole layers on Mini, single experts on Small.
    pub fn residency(&self) -> (usize, usize) {
        match &self.backend {
            ExpertBackend::Bf16 { pinned_layers, .. } => {
                (*pinned_layers, self.layers.len() - *pinned_layers)
            }
            ExpertBackend::MxFp8 {
                pinned_experts,
                total_experts,
                ..
            } => (*pinned_experts, *total_experts - *pinned_experts),
        }
    }

    /// Expert-slot hits and fills, at this backend's granularity. Small's are the pool's own
    /// counters, read at this instant — a copy kept here could disagree with the pool it
    /// describes, and the run's closing rebase ([`Fetched::since`]) reads the same numbers.
    pub fn fetched(&self) -> Fetched {
        match &self.backend {
            ExpertBackend::Bf16 { fetched, .. } => *fetched,
            ExpertBackend::MxFp8 { routed, .. } => Fetched {
                hits: routed.hits(),
                misses: routed.misses(),
            },
        }
    }

    /// Open the resident artifact, state the floor, ask [`partition`] for the placement, and place
    /// the globals, the dense layers, and the pinned layers' experts.
    ///
    /// **Every byte the run will allocate is charged before the decision.** `global_weight_bytes`
    /// and the members' dense headers are the always-resident charge, [`ScratchSizes`] splits into
    /// the KV bucket and the activation bucket exactly as `InklingScratch::new` allocates them, and
    /// the one-expert slot is reserved only as a floor charge (it is allocated only if a layer
    /// actually streams).
    pub fn build(
        dir: &str,
        text: &InklingTextConfig,
        capacity: usize,
        n_ctx: usize,
    ) -> Result<Self> {
        let path = format!("{dir}/resident.safetensors");
        let st = Safetensors::open_file(&path).with_context(|| format!("open {path}"))?;
        let (hidden, inter) = (text.hidden_size, text.intermediate_size);
        let n_layers = text.num_hidden_layers;
        let flags = layer_flags(text);
        let (globals, dense) = measure_resident(&st, text, &flags)?;
        let (n_pinned, tier_bytes) = plan_partition(text, (capacity, n_ctx), (globals, dense))?;
        let n_streamed = n_layers - n_pinned;
        let counts = (n_layers, n_pinned);
        let dims = (text.n_routed_experts, inter, hidden);
        let parts = Parts {
            text,
            st,
            flags: &flags,
            tier_bytes,
        };
        Self::assemble(parts, move |tier, st| {
            let experts = place_expert_sources(tier, st, counts, dims)?;
            let (w13_slot, w2_slot) = match n_streamed {
                0 => (None, None),
                _ => (
                    Some(DeviceBuf::new(2 * inter * hidden * 2)?),
                    Some(DeviceBuf::new(hidden * inter * 2)?),
                ),
            };
            Ok(ExpertBackend::Bf16 {
                experts,
                w13_slot,
                w2_slot,
                pinned_layers: n_pinned,
                fetched: Fetched { hits: 0, misses: 0 },
            })
        })
    }

    /// Allocate the resident tier, place the globals and every dense layer, then build the pin
    /// around the backend `make_backend` returns.
    ///
    /// **One author for the tail both members' constructors end with.** What differs is only the
    /// tier's SIZE (Mini adds its pinned slabs and staging slot, Small only the scratch charge
    /// the pool's budget excludes) and which [`ExpertBackend`] rides out — never the placement.
    /// The closure runs with the fresh tier so Mini can place its expert slabs into the same
    /// slab as the dense weights, and ignores it on Small, whose experts live in the pool.
    pub(super) fn assemble<F>(parts: Parts<'_>, make_backend: F) -> Result<Self>
    where
        F: FnOnce(&mut DeviceTier, &Safetensors) -> Result<ExpertBackend>,
    {
        let Parts {
            text,
            st,
            flags,
            tier_bytes,
        } = parts;
        let mut tier = DeviceTier::new(tier_bytes)?;
        let globals = place_globals(&mut tier, &st, text)?;
        let (layers, router_bias_host) = place_dense(&mut tier, &st, flags)?;
        let backend = make_backend(&mut tier, &st)?;
        Ok(Self {
            _tier: tier,
            st,
            globals,
            layers,
            router_bias_host,
            backend,
            hidden: text.hidden_size,
            inter: text.intermediate_size,
            n_experts: text.n_routed_experts,
        })
    }

    /// Layer `l`'s expert `e`, resolved either into the tier's resident slab or into the staging
    /// slot just refilled from the artifact.
    ///
    /// **A streamed fill joins the device first.** The slot is overwritten by a host copy with no
    /// synchronisation of its own, and launches are asynchronous — so without the fence the fill
    /// can land while a previous expert's GEMM is still reading the bytes. Every miss pays it; a
    /// resident hit pays nothing.
    pub fn expert_weights(
        &mut self,
        layer: usize,
        expert: usize,
    ) -> Result<(Bf16Weight, Bf16Weight)> {
        let (hidden, inter, n_experts) = (self.hidden, self.inter, self.n_experts);
        let ExpertBackend::Bf16 {
            experts,
            w13_slot,
            w2_slot,
            fetched,
            ..
        } = &mut self.backend
        else {
            bail!(
                "this pin streams .mxfp8 experts through the routed pool — there is no bf16 \
                 slab to slice"
            );
        };
        ensure!(
            layer < experts.len(),
            "layer {layer} is past the {} the pin placed",
            experts.len()
        );
        ensure!(
            expert < n_experts,
            "expert {expert} is past the config's {n_experts} routed experts"
        );
        let gu_plane = 2 * inter * hidden;
        match &experts[layer] {
            ExpertSource::Resident { gate_up, down } => {
                fetched.hits += 1;
                // SAFETY: the slabs are `[E, 2I, H]` / `[E, H, I]` bf16 (checked at placement)
                // and `expert < E` above; both pointers outlive every kernel the caller launches.
                let gu = unsafe { gate_up.add(expert * gu_plane) };
                let dn = unsafe { down.add(expert * hidden * inter) };
                Ok(expert_pair(hidden, inter, gu, dn))
            }
            ExpertSource::Streamed {
                gate_up_name,
                down_name,
            } => {
                fetched.misses += 1;
                // No `device_sync` here: the `copy_in_at` calls below are blocking null-stream
                // H2D `hipMemcpy`s ("waits for completion before returning control", documented
                // in `crate::device::DeviceBuf::copy_in_at`), so each one already joins every
                // launch enqueued on the null stream before it — including the previous step's
                // reads of this very slot. A sync here bought no ordering the copy does not.
                let (gu, _) = self.st.typed(gate_up_name, Dtype::Bf16)?;
                let (dn, _) = self.st.typed(down_name, Dtype::Bf16)?;
                let (gw, dw) = (gu_plane * 2, hidden * inter * 2);
                let span = (expert * gw)..((expert + 1) * gw);
                let slot = w13_slot
                    .as_mut()
                    .context("no w13 staging slot was allocated")?;
                slot.copy_in_at(0, &gu[span])?;
                let w13 = slot.ptr() as *const u16;
                let span = (expert * dw)..((expert + 1) * dw);
                let slot = w2_slot
                    .as_mut()
                    .context("no w2 staging slot was allocated")?;
                slot.copy_in_at(0, &dn[span])?;
                let w2 = slot.ptr() as *const u16;
                Ok(expert_pair(hidden, inter, w13, w2))
            }
        }
    }
}

/// One expert's `(w13, w2)` pair: `[2I, H]` and `[H, I]` bf16 at the given addresses. Free
/// rather than a method because the streaming arm holds `&mut` on the backend while it builds
/// the pair, and a `&self` receiver would borrow the whole pin against it.
fn expert_pair(
    hidden: usize,
    inter: usize,
    w13: *const u16,
    w2: *const u16,
) -> (Bf16Weight, Bf16Weight) {
    (
        Bf16Weight {
            packed: w13,
            o_dim: 2 * inter,
            i_dim: hidden,
        },
        Bf16Weight {
            packed: w2,
            o_dim: hidden,
            i_dim: inter,
        },
    )
}

/// The floor: the always-resident tensors, the KV cache, the activation scratch, and the
/// one-expert staging slot plus [`PIN_SLACK`].
fn pin_floor(text: &InklingTextConfig, n_ctx: usize, resident: usize) -> Floor {
    let sizes = ScratchSizes::of(text, n_ctx);
    let kv = sizes.kv_total();
    Floor {
        always_resident: Bytes(resident as u64),
        kv_at_max_ctx: Bytes(kv as u64),
        scratch: Bytes((sizes.total() - kv) as u64),
        slot_bytes: Bytes(
            (single_expert_bytes(text.hidden_size, text.intermediate_size, 2) + PIN_SLACK) as u64,
        ),
    }
}

/// Place every layer's DENSE weights and read its router bias.
fn place_dense(
    tier: &mut DeviceTier,
    st: &Safetensors,
    flags: &[(bool, bool)],
) -> Result<(Vec<InklingLayerPin>, Vec<Vec<f32>>)> {
    let mut layers = Vec::with_capacity(flags.len());
    let mut biases = Vec::with_capacity(flags.len());
    for (l, &f) in flags.iter().enumerate() {
        layers.push(place_layer(tier, st, l, f)?);
        let name = format!("{}.{}", names::layer_prefix(l), names::ROUTER_BIAS);
        let (bytes, _) = st.typed(&name, Dtype::F32)?;
        biases.push(rivoli_artifact::quant::read_f32(bytes));
    }
    Ok((layers, biases))
}

/// Each stored layer's `(full attention, looped)` flags — `geometry`'s rules, read once.
fn layer_flags(text: &InklingTextConfig) -> Vec<(bool, bool)> {
    let block = text.n_layers_per_attn_res_block;
    let core = text
        .loop_config
        .as_ref()
        .map(|lc| (lc.core_start, lc.core_end));
    (0..text.num_hidden_layers)
        .map(|l| {
            let looped = core.is_some_and(|(s, e)| (s..e).contains(&l));
            (is_block_end(l, block), looped)
        })
        .collect()
}

/// The per-layer expert residency: `counts.1` slabs in the tier, the rest named for streaming.
fn place_expert_sources(
    tier: &mut DeviceTier,
    st: &Safetensors,
    counts: (usize, usize),
    dims: (usize, usize, usize),
) -> Result<Vec<ExpertSource>> {
    let (n_layers, n_pinned) = counts;
    let mut out = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        out.push(if l < n_pinned {
            place_experts(tier, st, l, dims)?
        } else {
            let (gate_up_name, down_name) = expert_names(l);
            ExpertSource::Streamed {
                gate_up_name,
                down_name,
            }
        });
    }
    Ok(out)
}

/// A refusal names the arithmetic — "refused" without the numbers is the unhelpful half.
fn below_floor(r: Refusal) -> anyhow::Error {
    let Refusal::BelowFloor { need, have } = r;
    anyhow::anyhow!(
        "Inkling 2 Mini needs {:.2} GiB resident before a single expert layer can be pinned, and \
         {:.2} GiB is available — raise --max-mem or lower --ctx",
        need.0 as f64 / (1u64 << 30) as f64,
        have.0 as f64 / (1u64 << 30) as f64
    )
}

/// Measure the artifact's global + dense bytes and verify the globals agree with geometry.
fn measure_resident(
    st: &Safetensors,
    text: &InklingTextConfig,
    flags: &[(bool, bool)],
) -> Result<(usize, usize)> {
    let dense = dense_total(st, flags)?;
    let globals = global_weight_bytes(text);
    let carried = globals_bytes(st, text)?;
    ensure!(
        carried == globals,
        "the artifact's global tensors total {carried} bytes against geometry's {globals} — \
         the floor is stated from a derivation that no longer matches the artifact"
    );
    Ok((globals, dense))
}

/// Run `residency::partition` over the expert layers and return `(n_pinned, tier_bytes)`.
fn plan_partition(
    text: &InklingTextConfig,
    (capacity, n_ctx): (usize, usize),
    (globals, dense): (usize, usize),
) -> Result<(usize, usize)> {
    let (hidden, inter) = (text.hidden_size, text.intermediate_size);
    let n_layers = text.num_hidden_layers;
    let floor = pin_floor(text, n_ctx, globals + dense);
    let unit = expert_layer_bytes(text.n_routed_experts, inter, hidden, 2);
    let units = expert_units(n_layers, unit)?;
    let placement = partition(&units, Bytes(capacity as u64), floor).map_err(below_floor)?;
    for (i, id) in placement.pinned.iter().enumerate() {
        ensure!(
            id.0 as usize == i,
            "the pinned set is not the prefix layer order assumes"
        );
    }
    let (n_pinned, n_streamed) = (placement.pinned.len(), placement.streamed.len());
    let slot = if n_streamed > 0 {
        single_expert_bytes(hidden, inter, 2)
    } else {
        0
    };
    let tier_bytes = globals + dense + n_pinned * unit + PIN_SLACK + slot;
    tracing::info!(
        "partition: {n_pinned} of {n_layers} expert layers pinned, {n_streamed} streamed \
         through 1-expert slots ({:.1} GiB tier, {:.1} GiB floor)",
        tier_bytes as f64 / (1u64 << 30) as f64,
        floor.total().0 as f64 / (1u64 << 30) as f64,
    );
    Ok((n_pinned, tier_bytes))
}
