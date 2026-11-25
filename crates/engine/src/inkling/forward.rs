//! Inkling 2's forward pass: the looped decoder — sliding/full attention with a sink and a
//! relative-position bias, short convolutions on every k/v path and after both sub-blocks, the
//! block-residual query banks, the per-pass message gates, and the softcapped SwiGLU MoE.
//!
//! Everything consumes [`super::pin::InklingLayerPin`]'s resolved addresses and the launcher
//! family; residency is the pin's decision, not this file's, and the chain runs on the NULL stream
//! uniformly. The one host readback inside a layer is the router's (`route_into` is host math); the
//! token's one join is the sampler's.
//!
//! Traps this file is the last line of: **T7** layer 0 is a block boundary and a boundary's
//! residual is the attention output ALONE; **T8** an extra pass restarts from the pre-loop states
//! and skips the layer-`core_start` boundary; **T11** a sliding layer norms all three gated inputs
//! with ONE shared norm, a full layer with three distinct ones; **T16** each pass has its own cache
//! layers; **T17** `rel_extent`'s SOURCE differs by layer kind; **T13** message order is
//! `(q, k, v, mlp)`.

use super::decode::LayerStep;
use super::engine::{InklingEngine, InklingScratch};
use super::geometry::{ScratchSizes, qk_scale, sconv_dynamic_divisor, sconv_static_divisor};
use super::pin::{InklingLayerPin, LayerSconv};
use crate::device::DeviceBuf;
use crate::resident::Bf16Weight;
use anyhow::{Result, ensure};
use rivoli_artifact::inkling_config::InklingTextConfig;
use rivoli_backend::{
    NULL_STREAM, launch_gemm_bf16, launch_gemv_bf16, launch_kv_scatter, launch_l2_clip,
    launch_log_scaled_l2_clip, launch_logit_softcap, launch_message_gate, launch_rel_bias_gather,
    launch_rmsnorm_single, launch_rmsnorm_weightless_batch, launch_scaled_l2_clip, launch_sconv,
    launch_sconv_dynamic, launch_sigmoid_gate, launch_sink_attend, memcpy_dtod_async,
};

/// A buffer's device address as an f32 pointer. `DeviceBuf::ptr` takes `&self`, so the pointer is
/// a copy — no borrow of the scratch outlives the expression, which is what lets the loop launch
/// from `&mut self` without re-borrowing the struct.
pub(super) fn p(b: &DeviceBuf) -> *mut f32 {
    b.ptr() as *mut f32
}

pub(super) fn pc(b: &DeviceBuf) -> *const f32 {
    b.ptr() as *const f32
}

/// `dst = src` over `n` f32, enqueued on the null stream.
///
/// # Safety
/// `dst`/`src` are `n`-element live device regions, NON-OVERLAPPING, outliving `stream`'s
/// completion (`memcpy_dtod_async` returns as soon as the copy is enqueued).
pub(super) unsafe fn copy_dev(dst: *mut f32, src: *const f32, n: usize) -> Result<()> {
    // SAFETY: the caller's contract; `n` elements are `n·4` bytes.
    unsafe { memcpy_dtod_async(dst as *mut u8, src as *const u8, n * 4, NULL_STREAM) }
}

/// One projection on a single row: `out = w · x`, `w` being `[o_dim, i_dim]` bf16. `m = 1` is
/// the GEMV's shape, so this is [`launch_gemv_bf16`] — the wide-load kernel — and not the
/// scalar [`launch_gemm_bf16`] that `rel_logits` still needs at `m = num_heads`. The `i_dim`
/// agreement is checked here: a mismatch reads past a live buffer and produces fluent wrong
/// text.
///
/// # Safety
/// `x` is `i_dim` live f32, `out` is `w.o_dim` live f32, `w.packed` is `w.o_dim·w.i_dim` live
/// bf16; all outlive the null-stream launch's completion.
pub unsafe fn proj(x: *const f32, w: &Bf16Weight, out: *mut f32, i_dim: usize) -> Result<()> {
    ensure!(
        w.i_dim == i_dim,
        "projection expects an activation of {} but was handed {i_dim} — a shape mismatch here \
         reads past the end of a live buffer and produces fluent wrong text",
        w.i_dim
    );
    // SAFETY: the caller's contract, plus the `i_dim` agreement checked above.
    unsafe { launch_gemv_bf16(x, w.packed, w.o_dim, w.i_dim, out, NULL_STREAM) }
}

/// The logit tail's weights and dims: the final norm, the lm_head (live rows only, T14), the
/// softcap. Bundled so [`logit_tail`] stays at four arguments.
pub struct LogitTail<'a> {
    pub norm_w: *const f32,
    pub head: &'a Bf16Weight,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    pub cap: f32,
}

/// Final RMSNorm into `xn`, the lm_head into `logits`, the final-logit softcap in place.
///
/// # Safety
/// `x`/`xn` are `hidden` live f32, `logits` is `vocab` live f32, `norm_w` is `hidden` f32,
/// `head.packed` is `vocab·hidden` live bf16; all outlive the null-stream launches.
pub unsafe fn logit_tail(
    x: *const f32,
    xn: *mut f32,
    logits: *mut f32,
    t: &LogitTail,
) -> Result<()> {
    // SAFETY: the caller's contract; `xn` is `hidden` f32 for the norm's one row.
    unsafe { launch_rmsnorm_single(x, t.norm_w, t.hidden, t.eps, xn) }?;
    // SAFETY: `proj` checks the `i_dim` agreement.
    unsafe { proj(xn, t.head, logits, t.hidden) }?;
    // SAFETY: `logits` is `vocab` live f32; the softcap is in place.
    unsafe { launch_logit_softcap(logits, t.vocab, 1.0, t.cap, NULL_STREAM) }
}

/// The attention head geometry.
#[derive(Clone, Copy)]
pub struct AttnDims {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// `qk_l2_clip_threshold` (16√2).
    pub qk_clip: f32,
    pub eps: f32,
    /// `head_dim^−0.5`.
    pub scaling: f32,
}

/// The rel-bias geometry. `rel_extent` comes from a DIFFERENT config field per layer kind (T17).
#[derive(Clone, Copy)]
pub struct RelDims {
    pub heads: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub rel_clip: f32,
    /// `Some` on full layers (the LogScale τ); sliding layers clip without it.
    pub tau: Option<f32>,
}

/// One call's full geometry.
#[derive(Clone, Copy)]
pub struct LayerDims {
    pub attn: AttnDims,
    pub rel: RelDims,
    pub hidden: usize,
    pub k_size: usize,
    pub ctx: usize,
}

impl LayerDims {
    pub(super) fn of(cfg: &InklingTextConfig, full: bool, n_ctx: usize) -> Self {
        let (heads, kv_heads, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        Self {
            attn: AttnDims {
                heads,
                kv_heads,
                head_dim: hd,
                qk_clip: cfg.qk_l2_clip_threshold as f32,
                eps: cfg.rms_norm_eps as f32,
                scaling: qk_scale(hd),
            },
            rel: RelDims {
                heads,
                d_rel: cfg.d_rel,
                // T17: a sliding layer's extent is the WINDOW; a full layer's is `rel_extent`.
                rel_extent: match full {
                    true => cfg.rel_extent,
                    false => cfg.sliding_window_size,
                },
                rel_clip: cfg.rel_l2_clip_threshold as f32,
                tau: None, // a function of `pos`: the caller fills it in
            },
            hidden: cfg.hidden_size,
            k_size: cfg.sconv_kernel_size,
            ctx: n_ctx,
        }
    }
}

/// Batched scaled clip (sliding when `tau` is `None`, full-layer LogScale when `Some`), in place.
///
/// # Safety
/// `x`/`scale` are `rows·n` live f32; each row's clip is barrier-separated in place.
unsafe fn clip_scaled(
    x: *mut f32,
    scale: *const f32,
    (rows, n): (usize, usize),
    (threshold, tau): (f32, Option<f32>),
) -> Result<()> {
    // SAFETY: the caller's contract; the arms differ only in the LogScale factor.
    unsafe {
        match tau {
            Some(t) => launch_log_scaled_l2_clip(x, scale, rows, n, threshold, t, x, NULL_STREAM),
            None => launch_scaled_l2_clip(x, scale, rows, n, threshold, x, NULL_STREAM),
        }
    }
}

/// The q norm over ALL heads (T4/T18): learned per-head scale, then the L2 clip; full layers also
/// apply the LogScale τ.
///
/// # Safety
/// `q`/`scale` are `heads·head_dim` live f32; each row reads its vector before writing it.
unsafe fn q_norm(q: *mut f32, scale: *const f32, d: &AttnDims, tau: Option<f32>) -> Result<()> {
    // SAFETY: the caller's contract.
    unsafe { clip_scaled(q, scale, (d.heads, d.head_dim), (d.qk_clip, tau)) }
}

/// The k norm: the L2 clip over all KV heads, no learned scale (T18).
///
/// # Safety
/// `k` is `kv_heads·head_dim` live f32; barrier-separated in place.
unsafe fn k_norm(k: *mut f32, d: &AttnDims) -> Result<()> {
    // SAFETY: the caller's contract.
    unsafe { launch_l2_clip(k, d.kv_heads, d.head_dim, d.qk_clip, k, NULL_STREAM) }
}

/// The v norm: weightless RMSNorm over each KV head (T18).
///
/// # Safety
/// `v` is `kv_heads·head_dim` live f32; `kv_heads` rows of `head_dim` in place.
unsafe fn v_norm(v: *mut f32, d: &AttnDims) -> Result<()> {
    // SAFETY: the caller's contract; scale 1 (weightless).
    unsafe { launch_rmsnorm_weightless_batch(v, d.kv_heads, d.head_dim, d.eps, 1.0, NULL_STREAM) }
}

/// The gated output projection's buffers.
struct GatedOutIo {
    /// The LAYER INPUT (the normed Q input passed to `self_attn`) — the gate reads it.
    pub x: *const f32,
    pub attn: *mut f32,
    pub gate: *mut f32,
    /// The projection output `[hidden]`.
    pub out: *mut f32,
}

/// `sigmoid(gate_proj(x)) · attn_out`, then `o_proj` (MOD:488–500).
///
/// # Safety
/// `x`/`out` are `hidden` live f32, `attn`/`gate` are `heads·head_dim` live f32; the pin's
/// `attn_gate`/`o` weights are live bf16.
unsafe fn gated_output(
    pin: &InklingLayerPin,
    io: &GatedOutIo,
    d: &AttnDims,
    hidden: usize,
) -> Result<()> {
    let qd = d.heads * d.head_dim;
    // SAFETY: the caller's contract.
    unsafe { proj(io.x, &pin.attn_gate, io.gate, hidden) }?;
    // SAFETY: `attn`/`gate` are `qd` live f32.
    unsafe { launch_sigmoid_gate(io.attn, io.gate, qd, NULL_STREAM) }?;
    // SAFETY: `attn` is `qd` f32, `out` is `hidden` f32; `proj` checks the agreement.
    unsafe { proj(io.attn, &pin.o, io.out, qd) }
}

/// One k/v side's short-convolution form. The weights are COPIED (`Bf16Weight` is `Copy`) rather
/// than borrowed, so a value built inside a `match` arm on the pin can outlive that arm.
#[derive(Clone, Copy)]
enum SconvSide {
    /// Sliding: the self-contained `conv1d`, divisor `√(k+1)`.
    Static(*const f32),
    /// Full: `static_weight` plus the tap projection, divisor `√(2k+1)`.
    Dynamic { taps: *const f32, proj: Bf16Weight },
}

/// One k/v path's operands. `buf` is the projection's scratch (the sconv reads it as `current` and
/// writes it as `out`, safe in place); `cache` is this cache layer's deinterleaved K or V.
struct KvPath {
    pub proj_w: Bf16Weight,
    pub sconv: SconvSide,
    pub state: *mut f32,
    pub buf: *mut f32,
    pub cache: *mut f32,
    pub dyn_w: *mut f32,
    pub is_key: bool,
}

/// One k/v decode path: project, short conv (advancing `state`), norm (L2 clip for k, weightless
/// RMSNorm for v — T18), scatter into the cache at `pos`. `geo` is `(hidden, k_size, pos, ctx)`.
///
/// # Safety
/// `x` is `hidden` live f32; every pointer in `k` is a live scratch allocation sized by `geo`.
unsafe fn kv_path(
    x: *const f32,
    k: &KvPath,
    d: &AttnDims,
    geo: (usize, usize, usize, usize),
) -> Result<()> {
    let (hidden, k_size, pos, ctx) = geo;
    let channels = d.kv_heads * d.head_dim;
    // SAFETY: the caller's contract; the sconv `current`/`out` alias is read-then-write.
    unsafe {
        proj(x, &k.proj_w, k.buf, hidden)?;
        match k.sconv {
            SconvSide::Static(w) => launch_sconv(
                w,
                k.state,
                k.buf,
                channels,
                k_size,
                sconv_static_divisor(k_size),
                k.buf,
                NULL_STREAM,
            ),
            SconvSide::Dynamic { taps, proj: tap } => {
                proj(x, &tap, k.dyn_w, hidden)?;
                launch_sconv_dynamic(
                    taps,
                    k.dyn_w,
                    k.state,
                    k.buf,
                    channels,
                    d.head_dim,
                    k_size,
                    sconv_dynamic_divisor(k_size),
                    k.buf,
                    NULL_STREAM,
                )
            }
        }?;
        if k.is_key {
            k_norm(k.buf, d)?;
        } else {
            v_norm(k.buf, d)?;
        }
        launch_kv_scatter(k.buf, k.cache, pos, d.head_dim, ctx, channels, NULL_STREAM)
    }
}

/// The two k/v paths. A full layer's dynamic conv projects the SAME input its linear projection
/// read (`key_hidden_states`/`value_hidden_states`, MOD:524–526).
///
/// # Safety
/// The caller's `attention` contract; the scratch and pin are sized for the geometry.
unsafe fn kv_paths(io: &LayerCall<'_>) -> Result<()> {
    let (pin, sc, c, g) = (io.pin, io.sc, io.call, io.dims);
    let (k_side, v_side) = match pin.sconv {
        LayerSconv::Static { k, v } => (SconvSide::Static(k), SconvSide::Static(v)),
        LayerSconv::Dynamic {
            k_static,
            k_proj,
            v_static,
            v_proj,
        } => (
            SconvSide::Dynamic {
                taps: k_static,
                proj: k_proj,
            },
            SconvSide::Dynamic {
                taps: v_static,
                proj: v_proj,
            },
        ),
    };
    let geo = (g.hidden, g.k_size, c.step.pos, g.ctx);
    let cache = c.step.cache_layer;
    // SAFETY: the caller's contract; each pointer is a live scratch/pin allocation.
    unsafe {
        let kp = KvPath {
            proj_w: pin.k,
            sconv: k_side,
            state: p(&sc.sconv[cache][0]),
            buf: p(&sc.k),
            cache: p(&sc.kc[cache]),
            dyn_w: p(&sc.dyn_w),
            is_key: true,
        };
        kv_path(c.inputs[1], &kp, &g.attn, geo)?;
        let vp = KvPath {
            proj_w: pin.v,
            sconv: v_side,
            state: p(&sc.sconv[cache][1]),
            buf: p(&sc.v),
            cache: p(&sc.vc[cache]),
            dyn_w: p(&sc.dyn_w),
            is_key: false,
        };
        kv_path(c.inputs[2], &vp, &g.attn, geo)
    }
}

/// The rel-bias path's operands.
struct RelBiasPath<'a> {
    pub linear: &'a Bf16Weight,
    pub proj: &'a Bf16Weight,
    pub norm_scale: *const f32,
    pub states: *mut f32,
    pub logits: *mut f32,
    pub bias: *mut f32,
}

/// The rel-bias path (MOD:220–227): project, clip all heads, GEMM to `rel_logits` (`rel_proj` is
/// stored transposed so `x·wᵀ` reads it directly), then distance-gather into `[heads, n]`.
/// `geo` is `(pos, n)`.
///
/// # Safety
/// `xn` is `hidden` live f32; the scratch buffers are sized for the geometry; all outlive the
/// null-stream launches.
unsafe fn rel_bias_path(
    xn: *const f32,
    r: &RelBiasPath,
    rd: &RelDims,
    geo: (usize, usize),
) -> Result<()> {
    let (pos, n) = geo;
    ensure!(
        n > 0 && pos + 1 >= n,
        "rel-bias window {n} does not fit position {pos} — the caller decodes left-to-right"
    );
    // T17 both ends: the extent the caller derived from the layer KIND must be the row count the
    // placed projection actually has, or the gather reads past the weight.
    ensure!(
        rd.rel_extent == r.proj.o_dim,
        "rel extent {} against the placed projection's {} rows: the layer kind and the artifact \
         disagree about T17",
        rd.rel_extent,
        r.proj.o_dim
    );
    let start = pos + 1 - n;
    // SAFETY: the caller's contract; `heads` rows of `d_rel` are clipped in place.
    unsafe {
        proj(xn, r.linear, r.states, r.linear.i_dim)?;
        clip_scaled(
            r.states,
            r.norm_scale,
            (rd.heads, rd.d_rel),
            (rd.rel_clip, rd.tau),
        )?;
        launch_gemm_bf16(
            r.states,
            r.proj.packed,
            r.logits,
            rd.heads,
            rd.rel_extent,
            rd.d_rel,
            NULL_STREAM,
        )?;
        launch_rel_bias_gather(
            r.logits,
            r.bias,
            pos,
            start,
            n,
            rd.heads,
            rd.rel_extent,
            NULL_STREAM,
        )
    }
}

/// The GQA attend's buffers; `sink` is the `[heads]` logit vector, read ON THE DEVICE.
struct GqaIo {
    pub q: *const f32,
    pub kc: *const f32,
    pub vc: *const f32,
    pub bias: *const f32,
    pub sink: *const f32,
    pub attn: *mut f32,
    pub pos: usize,
    pub n: usize,
    pub ctx: usize,
}

/// The GQA decode attend once across all query heads, with the rel bias and the sink (T10).
///
/// # Safety
/// `q`/`attn` are `heads·head_dim` live f32; `kc`/`vc` hold `kv_heads·ctx·head_dim` at the
/// window's first slot; `bias` is `heads·n`; `sink` is `heads` device f32.
unsafe fn attend_gqa(io: &GqaIo, d: &AttnDims) -> Result<()> {
    ensure!(
        io.n > 0 && io.pos + 1 >= io.n,
        "attend window {} does not fit position {} — the caller decodes left-to-right",
        io.n,
        io.pos
    );
    ensure!(
        d.kv_heads > 0 && d.heads.is_multiple_of(d.kv_heads),
        "{} query heads do not group onto {} KV heads",
        d.heads,
        d.kv_heads
    );
    let start = io.pos + 1 - io.n;
    // SAFETY: the caller's contract; both cache pointers are the window's first slot.
    unsafe {
        launch_sink_attend(
            io.q,
            io.kc.add(start * d.head_dim),
            io.vc.add(start * d.head_dim),
            io.bias,
            io.sink,
            d.scaling,
            io.n,
            d.head_dim,
            d.heads,
            d.kv_heads,
            io.ctx,
            io.attn,
            NULL_STREAM,
        )
    }
}

/// One attention call: the three PRE-NORMED inputs in `(q, k, v)` order (T13 — the same triple
/// `attn_norms` hands over) and the branch output.
pub struct AttnCall {
    pub step: LayerStep,
    pub n: usize,
    pub tau: Option<f32>,
    pub inputs: [*const f32; 3],
    pub out: *mut f32,
}

/// What every per-layer launch chain reads: the layer's resolved weights, the scratch, this
/// position's call, and the geometry. Bundled because [`attention`] and [`kv_paths`] take exactly
/// these four and a separate four-argument signature each made the second a copy of the first.
pub(super) struct LayerCall<'a> {
    pub pin: &'a InklingLayerPin,
    pub sc: &'a InklingScratch,
    pub call: &'a AttnCall,
    pub dims: &'a LayerDims,
}

/// One MessageGate's operands: which gate of which layer, and the two vectors it mixes.
pub struct MsgIo {
    pub layer: usize,
    pub which: usize,
    pub hidden: *const f32,
    pub message: *const f32,
    pub out: *mut f32,
}

/// A block-residual attention's sources (MOD:833–858): the bank rows, the partial row, and the
/// query bank (`[nq, hidden]`).
pub struct BlockRes {
    pub bank: *const f32,
    pub n_bank: usize,
    pub partial: *const f32,
    pub query: *const f32,
    pub nq: usize,
}

impl InklingEngine<'_> {
    /// The block-residual attend: `softmax(q·kᵀ/√H)·v` over the normed `[bank ‖ partial]` stack,
    /// zero bias and a `−∞` sink (no mask, `is_causal = False`, one KV group, MOD:833–858).
    pub(super) fn block_res_attend(&mut self, br: &BlockRes, out: *mut f32) -> Result<()> {
        let hidden = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let n_src = br.n_bank + 1;
        ensure!(br.nq <= 4, "a block-residual bank has at most 4 queries");
        // `stacked` is `bank_slots·hidden` f32 and `res_bias` is `4·bank_slots` f32 — so a bank
        // past the slot count is an OOB write, not a short read. The count comes from its ONE
        // author (`ScratchSizes`) rather than a second `⌈layers/block⌉ + 1` here; `DeviceBuf`
        // exposes no length, so this is the only reading of what was allocated.
        let max_slots = ScratchSizes::of(self.cfg, self.n_ctx).bank_slots;
        ensure!(
            n_src <= max_slots,
            "block-residual bank wants {n_src} slots (n_bank {} plus the partial), past the \
             scratch's {max_slots}",
            br.n_bank
        );
        let stacked = p(&self.sc.stacked);
        // SAFETY: `stacked` and `res_bias` are sized for the whole bank (the slot count), and the
        // bank/partial rows are `hidden` f32; all outlive the null-stream launches.
        unsafe {
            if br.n_bank > 0 {
                copy_dev(stacked, br.bank, br.n_bank * hidden)?;
            }
            copy_dev(stacked.add(br.n_bank * hidden), br.partial, hidden)?;
            launch_rmsnorm_weightless_batch(stacked, n_src, hidden, eps, 1.0, NULL_STREAM)?;
            launch_sink_attend(
                br.query,
                stacked,
                stacked,
                pc(&self.sc.res_bias),
                pc(&self.sc.res_sink),
                (hidden as f32).sqrt().recip(),
                n_src,
                hidden,
                br.nq,
                1,
                n_src,
                out,
                NULL_STREAM,
            )
        }
    }

    /// One MessageGate (MOD:172–190): `(1−tanh(forget))·hidden + tanh(gain)·message`, fed by
    /// `cat[norm(hidden), norm(message)]` through the rank-width down projection.
    pub(super) fn message_gate(&mut self, io: &MsgIo) -> Result<()> {
        let hidden = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let gates = self.pin.layers[io.layer]
            .message_gates
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {} carries no MessageGates", io.layer))?;
        let g = &gates[io.which];
        let (cat, low) = (p(&self.sc.gate_cat), p(&self.sc.gate_low));
        let rank = g.forget.i_dim;
        // SAFETY: `gate_cat` is `2·hidden`, `gate_low` `rank`, `gate_f`/`gate_g` `hidden` f32,
        // and the gate's weights are live bf16 in the pin.
        unsafe {
            copy_dev(cat, io.hidden, hidden)?;
            copy_dev(cat.add(hidden), io.message, hidden)?;
            launch_rmsnorm_weightless_batch(cat, 2, hidden, eps, 1.0, NULL_STREAM)?;
            proj(cat, &g.down, low, 2 * hidden)?;
            proj(low, &g.forget, p(&self.sc.gate_f), rank)?;
            proj(low, &g.gain, p(&self.sc.gate_g), rank)?;
            launch_message_gate(
                pc(&self.sc.gate_f),
                pc(&self.sc.gate_g),
                io.hidden,
                io.message,
                hidden,
                io.out,
                NULL_STREAM,
            )
        }
    }
}

/// The attention sub-block (MOD:515–546): q projection + norm, rel bias, the per-head sink, the k/v
/// paths, the GQA attend, and the gated output projection.
///
/// # Safety
/// `io.call.inputs` are the `hidden` f32 the caller normed; `io.call.out` is `hidden` writable f32;
/// the scratch and pin are sized by `io.dims`; all outlive the null-stream launches.
pub(super) unsafe fn attention(io: &LayerCall<'_>) -> Result<()> {
    let (pin, sc, c, g) = (io.pin, io.sc, io.call, io.dims);
    let [q_in, _, _] = c.inputs;
    let q = p(&sc.q);
    // SAFETY: the caller's contract; every buffer is the scratch's live allocation.
    unsafe {
        proj(q_in, &pin.q, q, g.hidden)?;
        q_norm(q, pin.q_norm_scale, &g.attn, c.tau)?;
        let rbp = RelBiasPath {
            linear: &pin.rel_linear,
            proj: &pin.rel_proj,
            norm_scale: pin.rel_norm_scale,
            states: p(&sc.rel_states),
            logits: p(&sc.rel_logits),
            bias: p(&sc.bias),
        };
        rel_bias_path(q_in, &rbp, &g.rel, (c.step.pos, c.n))?;
        proj(q_in, &pin.sink, p(&sc.sink), g.hidden)?;
        kv_paths(io)?;
    }
    let gio = GqaIo {
        q,
        kc: pc(&sc.kc[c.step.cache_layer]),
        vc: pc(&sc.vc[c.step.cache_layer]),
        bias: pc(&sc.bias),
        sink: pc(&sc.sink),
        attn: p(&sc.attn),
        pos: c.step.pos,
        n: c.n,
        ctx: g.ctx,
    };
    // SAFETY: every pointer is the scratch's live allocation sized for the geometry.
    unsafe {
        attend_gqa(&gio, &g.attn)?;
        let goio = GatedOutIo {
            x: q_in,
            attn: p(&sc.attn),
            gate: p(&sc.gate),
            out: c.out,
        };
        gated_output(pin, &goio, &g.attn, g.hidden)
    }
}
