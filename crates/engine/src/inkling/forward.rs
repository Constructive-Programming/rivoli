//! Inkling 2's forward pass: the looped decoder layer loop — the sliding/full attention with a
//! sink and relative-position bias, the short convolutions on the k/v paths, the block-residual
//! query banks, the per-pass message gates, and the softcapped SwiGLU MoE. The loop body is unlike
//! the four arms on record, so it lives here rather than behind an arch flag.
//!
//! Everything consumes [`super::pin::InklingLayerPin`]'s resolved addresses and the launcher
//! family; residency is the pin's decision, not this file's. The chain runs on the NULL stream
//! uniformly (the glimmer argument: a compute stream at two call sites inside an otherwise
//! null-stream layer is the same bug inverted).
//!
//! **No per-head host loop and no per-layer sync.** The norm clips and the attend are batched over
//! their rows on-device (one launch over `heads` blocks), and the sink logits are read by
//! `sink_attend` from device memory — the `[heads]` vector `sink_proj` wrote stays a pointer.

use super::pin::LayerSconv;
use crate::resident::Bf16Weight;
use anyhow::{Result, ensure};
use rivoli_backend::{
    NULL_STREAM, launch_gemm_bf16, launch_kv_scatter, launch_l2_clip, launch_log_scaled_l2_clip,
    launch_logit_softcap, launch_rel_bias_gather, launch_rmsnorm_single,
    launch_rmsnorm_weightless_batch, launch_scaled_l2_clip, launch_sconv, launch_sigmoid_gate,
    launch_sink_attend,
};

/// One projection applied to a single row: `out = w · x`, `w` being `[o_dim, i_dim]` bf16. The
/// bf16 arm is a GEMM at `m = 1` rather than a GEMV because there is no bf16 GEMV in the tree
/// (the glimmer finding). The `i_dim` agreement is checked here — a shape mismatch reads past the
/// end of a live buffer and produces fluent wrong text.
///
/// # Safety
/// `x` is `i_dim` live f32, `out` is `w.o_dim` live f32, `w.packed` is `w.o_dim·w.i_dim` live bf16;
/// all outlive the null-stream launch's completion (the caller syncs before reading `out`).
pub unsafe fn proj(x: *const f32, w: &Bf16Weight, out: *mut f32, i_dim: usize) -> Result<()> {
    ensure!(
        w.i_dim == i_dim,
        "projection expects an activation of {} but was handed {i_dim} — a shape mismatch here \
         reads past the end of a live buffer and produces fluent wrong text",
        w.i_dim
    );
    // SAFETY: the caller's contract, plus the `i_dim` agreement checked above.
    unsafe { launch_gemm_bf16(x, w.packed, out, 1, w.o_dim, w.i_dim, NULL_STREAM) }
}

/// The logit tail's resolved weights and dims: the final norm, the lm_head (the live
/// `unpadded_vocab_size` rows, T14), and the softcap. Bundled so [`logit_tail`] stays at four
/// arguments (the arity gate fires at five).
pub struct LogitTail<'a> {
    /// `model.norm.weight` `[hidden]`.
    pub norm_w: *const f32,
    /// `lm_head.weight` `[vocab, hidden]` — the live rows, not the padded embed.
    pub head: &'a Bf16Weight,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    /// `final_logit_softcapping` (30).
    pub cap: f32,
}

/// The logit tail: final RMSNorm into `xn`, the lm_head projection into `logits`, then the
/// final-logit softcap in place (`mult 1`, `cap`). `xn` is `hidden` f32, `logits` is `vocab` f32.
///
/// # Safety
/// `x`/`xn` are `hidden` live f32, `logits` is `vocab` live f32, `norm_w` is `hidden` f32,
/// `head.packed` is `vocab·hidden` live bf16; all outlive the null-stream launches' completion
/// (the caller syncs before reading `logits`).
pub unsafe fn logit_tail(
    x: *const f32,
    xn: *mut f32,
    logits: *mut f32,
    t: &LogitTail,
) -> Result<()> {
    // SAFETY: the caller's contract; `xn` is `hidden` f32 for the norm's one row.
    unsafe { launch_rmsnorm_single(x, t.norm_w, t.hidden, t.eps, xn) }?;
    // SAFETY: `proj` checks the `i_dim` agreement; `xn` is `hidden` f32, `logits` is `vocab` f32.
    unsafe { proj(xn, t.head, logits, t.hidden) }?;
    // SAFETY: `logits` is `vocab` live f32; the softcap is in place.
    unsafe { launch_logit_softcap(logits, t.vocab, 1.0, t.cap, NULL_STREAM) }
}

/// The attention head geometry, bundled so the norm/attend helpers stay under the arity gate.
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

/// Batched scaled clip (sliding when `tau` is `None`, full-layer LogScale when `Some`), in place
/// over `rows` vectors of width `n`. Shared by `q_norm` and `rel_bias_path`.
///
/// # Safety
/// `x` and `scale` are `rows·n` live f32; each row's clip is barrier-separated in place.
unsafe fn clip_scaled(
    x: *mut f32,
    scale: *const f32,
    (rows, n): (usize, usize),
    (threshold, tau): (f32, Option<f32>),
) -> Result<()> {
    unsafe {
        match tau {
            Some(t) => launch_log_scaled_l2_clip(x, scale, rows, n, threshold, t, x, NULL_STREAM),
            None => launch_scaled_l2_clip(x, scale, rows, n, threshold, x, NULL_STREAM),
        }
    }
}

/// The q norm over ALL heads in one launch (T4): a learned per-head scale, then the L2 clip, in
/// place. `q` and `scale` are `[heads·head_dim]` f32, one row of `head_dim` per head — the rows are
/// the launch's blocks, so no host loop per head. Sliding layers clip (`tau = None`); full layers
/// also apply the LogScale τ (`tau = Some`, precomputed host-side from `pos`).
///
/// # Safety
/// `q`/`scale` are `heads·head_dim` live f32; each row's clip reads its vector before writing it
/// (the launcher's barrier-separated in-place contract).
pub unsafe fn q_norm(q: *mut f32, scale: *const f32, d: &AttnDims, tau: Option<f32>) -> Result<()> {
    unsafe { clip_scaled(q, scale, (d.heads, d.head_dim), (d.qk_clip, tau)) }
}

/// The k norm: the L2 clip over ALL KV heads in one launch, in place, no learned scale (T18). `k`
/// is one token's `[kv_heads·head_dim]` f32.
///
/// # Safety
/// `k` is `kv_heads·head_dim` live f32; each row's clip is barrier-separated in place.
pub unsafe fn k_norm(k: *mut f32, d: &AttnDims) -> Result<()> {
    // SAFETY: the caller's contract; `kv_heads` rows of `head_dim`, in place.
    unsafe { launch_l2_clip(k, d.kv_heads, d.head_dim, d.qk_clip, k, NULL_STREAM) }
}

/// The v norm: weightless RMSNorm over each KV head's `head_dim` vector, in place (T18,
/// `use_value_norm_no_gamma`). `v` is one token's `[kv_heads·head_dim]` f32.
///
/// # Safety
/// `v` is `kv_heads·head_dim` live f32; the batch norm norms `kv_heads` rows of `head_dim`.
pub unsafe fn v_norm(v: *mut f32, d: &AttnDims) -> Result<()> {
    // SAFETY: `kv_heads` rows of `head_dim` f32, in place; scale 1 (weightless).
    unsafe { launch_rmsnorm_weightless_batch(v, d.kv_heads, d.head_dim, d.eps, 1.0, NULL_STREAM) }
}

/// The gated output projection's buffers, bundled for the arity gate.
pub struct GatedOutIo {
    /// The layer input `[hidden]` — the gate reads the LAYER INPUT, not the attend output.
    pub x: *const f32,
    /// The attention output `[heads·head_dim]`, gated in place.
    pub attn: *mut f32,
    /// The `gate_proj` scratch `[heads·head_dim]`.
    pub gate: *mut f32,
    /// The projection output `[hidden]`.
    pub out: *mut f32,
}

/// The gated output projection (MOD:488–500): `sigmoid(gate_proj(x)) · attn_out`, then `o_proj`.
/// The gate reads the layer input `x`; the sigmoid gate multiplies the attention output in place.
///
/// # Safety
/// `x`/`out` are `hidden` live f32, `attn`/`gate` are `heads·head_dim` live f32; the pin's
/// `attn_gate`/`o` weights are live bf16; all outlive the null-stream launches' completion.
pub unsafe fn gated_output(
    pin: &super::pin::InklingLayerPin,
    io: &GatedOutIo,
    d: &AttnDims,
    hidden: usize,
) -> Result<()> {
    let qd = d.heads * d.head_dim;
    // SAFETY: the caller's contract; `proj` checks the `i_dim` agreement.
    unsafe { proj(io.x, &pin.attn_gate, io.gate, hidden) }?;
    // SAFETY: `attn`/`gate` are `qd` live f32; the gate multiplies in place.
    unsafe { launch_sigmoid_gate(io.attn, io.gate, qd, NULL_STREAM) }?;
    // SAFETY: `attn` is `qd` f32, `out` is `hidden` f32; `proj` checks the agreement.
    unsafe { proj(io.attn, &pin.o, io.out, qd) }
}

/// One k/v path's resolved operands, bundled for the arity gate. `buf` is the projection's scratch
/// (the sconv reads it as `current` and writes it as `out` — the alias is safe, the kernel reads the
/// current tap before writing); `cache` is this cache-layer's deinterleaved K or V.
pub struct KvPath<'a> {
    pub proj_w: &'a Bf16Weight,
    pub sconv_w: *const f32,
    pub state: *mut f32,
    pub buf: *mut f32,
    pub cache: *mut f32,
    pub is_key: bool,
}

/// The static k/v decode path: project `xn`, run the self-contained short conv (advancing `state`),
/// norm (L2 clip for k, weightless RMSNorm for v — T18), then scatter into the deinterleaved cache at
/// `pos`. `geo` is `(hidden, k_size, pos, n_ctx)`. The full layer's dynamic conv is a separate path.
///
/// # Safety
/// `xn` is `hidden` live f32; `buf`/`state`/`cache` are the scratch's live allocations sized for the
/// geometry; all outlive the null-stream launches' completion (the caller syncs before the attend).
pub unsafe fn kv_path(
    xn: *const f32,
    p: &KvPath,
    d: &AttnDims,
    geo: (usize, usize, usize, usize),
) -> Result<()> {
    let (hidden, k_size, pos, n_ctx) = geo;
    let kv_channels = d.kv_heads * d.head_dim;
    // SAFETY: the caller's contract; the sconv `current`/`out` alias is read-then-write per channel.
    unsafe {
        proj(xn, p.proj_w, p.buf, hidden)?;
        launch_sconv(
            p.sconv_w,
            p.state,
            p.buf,
            kv_channels,
            k_size,
            super::geometry::sconv_static_divisor(k_size),
            p.buf,
            NULL_STREAM,
        )?;
        if p.is_key {
            k_norm(p.buf, d)?;
        } else {
            v_norm(p.buf, d)?;
        }
        launch_kv_scatter(
            p.buf,
            p.cache,
            pos,
            d.head_dim,
            n_ctx,
            kv_channels,
            NULL_STREAM,
        )
    }
}

/// The rel-bias path's resolved operands, bundled for the arity gate. `states`/`logits`/`bias` are
/// the attention scratch's rel-bias buffers.
pub struct RelBiasPath<'a> {
    pub linear: &'a Bf16Weight,
    pub proj: &'a Bf16Weight,
    pub norm_scale: *const f32,
    pub states: *mut f32,
    pub logits: *mut f32,
    pub bias: *mut f32,
}

/// The rel-bias geometry. `tau` is `Some` on full layers (the LogScale τ, precomputed host-side from
/// `pos`); sliding layers clip without it.
#[derive(Clone, Copy)]
pub struct RelDims {
    pub heads: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub rel_clip: f32,
    pub tau: Option<f32>,
}

/// The rel-bias decode path (MOD:220–227): project `xn` to `rel_states`, clip ALL heads in one
/// launch (scaled, or log-scaled with τ on full layers), GEMM to `rel_logits`
/// (`[heads, rel_extent]` — `rel_proj` is stored transposed so `gemm_bf16`'s x·wᵀ reads it
/// directly), then distance-gather to `bias` (`[heads, n]`). `geo` is `(pos, n)`, where `n` is the
/// attended window and key slot `t` sits at absolute position `start + t`, `start = pos + 1 − n`.
///
/// # Safety
/// `xn` is `hidden` live f32; the scratch buffers are sized for the geometry; all outlive the
/// null-stream launches' completion (the caller syncs before the attend reads `bias`).
pub unsafe fn rel_bias_path(
    xn: *const f32,
    p: &RelBiasPath,
    rd: &RelDims,
    geo: (usize, usize),
) -> Result<()> {
    let (pos, n) = geo;
    ensure!(
        n > 0 && pos + 1 >= n,
        "rel-bias window {n} does not fit position {pos} — the caller decodes left-to-right"
    );
    let start = pos + 1 - n;
    // SAFETY: the caller's contract; `heads` rows of `d_rel` are clipped in place per row.
    unsafe {
        proj(xn, p.linear, p.states, p.linear.i_dim)?;
        clip_scaled(
            p.states,
            p.norm_scale,
            (rd.heads, rd.d_rel),
            (rd.rel_clip, rd.tau),
        )?;
        launch_gemm_bf16(
            p.states,
            p.proj.packed,
            p.logits,
            rd.heads,
            rd.rel_extent,
            rd.d_rel,
            NULL_STREAM,
        )?;
        launch_rel_bias_gather(
            p.logits,
            p.bias,
            pos,
            start,
            n,
            rd.heads,
            rd.rel_extent,
            NULL_STREAM,
        )
    }
}

/// The GQA attend's resolved buffers, bundled for the arity gate. `kc`/`vc` are one cache-layer's
/// deinterleaved K/V (`[kv_heads, ctx, head_dim]`); `bias` is `[heads, n]` (the `rel_bias_gather`
/// output); `sink` is the `[heads]` `sink_proj` output, read ON THE DEVICE by the kernel.
pub struct GqaIo {
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

/// The GQA decode attend, ONCE across all query heads: each head `h` attends its KV head
/// `h/(heads/kv_heads)`'s last `n` vectors (contiguous in the deinterleaved cache, at
/// `h·kv_stride·head_dim`), with the head's rel-bias and sink (T10). The kernel maps the heads and
/// runs the block softmax itself, so there is no host loop and no readback. The sliding window is
/// the caller's `n`; `pos + 1 >= n` (the caller decodes left-to-right).
///
/// # Safety
/// `q`/`attn` are `heads·head_dim` live f32; `kc`/`vc` hold `kv_heads·ctx·head_dim` starting at the
/// window's first slot; `bias` is `heads·n`; `sink` is `heads` device f32; all outlive the
/// null-stream launch's completion (the caller syncs before reading `attn`).
pub unsafe fn attend_gqa(io: &GqaIo, d: &AttnDims) -> Result<()> {
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
    // SAFETY: the caller's contract; both cache pointers are the window's first slot, and the
    // kernel's per-head stride is the caller's `ctx`.
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

/// The attention sub-block's input/output, bundled for the arity gate. `x` is the attention input
/// (the block-residual query's output, already the layer's residual stream); `out` is the gated
/// attention contribution the decoder layer adds back to the residual.
pub struct AttnIo {
    pub x: *const f32,
    pub out: *mut f32,
}

/// The attention sub-block's geometry and head dims, bundled for the arity gate. `cache_layer` is
/// the loop-expanded cache index (T16); `k_size` the sconv tap count; `rd.tau` the full-layer
/// LogScale (None on sliding layers).
pub struct AttnCfg {
    pub pos: usize,
    pub n: usize,
    pub ctx: usize,
    pub cache_layer: usize,
    pub hidden: usize,
    pub k_size: usize,
    pub d: AttnDims,
    pub rd: RelDims,
}

/// The attention sub-block (MOD:578–639): input norm(s) → q/k/v projections (k/v through the static
/// short conv into the deinterleaved cache) → rel-bias + sink → the GQA sink attend → the gated
/// output projection. Sliding layers use the single `input_layernorm`; full layers the three q/k/v
/// input norms (T11). The sink stays on the device — no readback, no per-layer sync. Writes the
/// gated contribution to `io.out`; the caller owns the residual.
///
/// # Safety
/// `io.x`/`io.out` are `hidden` live f32; the scratch and pin are sized for the geometry; all
/// outlive the null-stream launches' completion (the caller syncs before reading `io.out`).
pub unsafe fn attention(
    pin: &super::pin::InklingLayerPin,
    sc: &mut super::engine::InklingScratch,
    io: &AttnIo,
    c: &AttnCfg,
) -> Result<()> {
    let d = &c.d;
    let full = !pin.input_ln[1].is_null();
    // SAFETY: the caller's contract; every pointer below is a live scratch/pin allocation sized for
    // the geometry, and the whole chain is stream-ordered on the null stream.
    unsafe {
        let xn = sc.xn.ptr_mut() as *mut f32;
        launch_rmsnorm_single(io.x, pin.input_ln[0], c.hidden, d.eps, xn)?;
        let (xn_k, xn_v) = if full {
            let xk = sc.xn_k.ptr_mut() as *mut f32;
            let xv = sc.xn_v.ptr_mut() as *mut f32;
            launch_rmsnorm_single(io.x, pin.input_ln[1], c.hidden, d.eps, xk)?;
            launch_rmsnorm_single(io.x, pin.input_ln[2], c.hidden, d.eps, xv)?;
            (xk, xv)
        } else {
            (xn, xn)
        };
        let q = sc.q.ptr_mut() as *mut f32;
        proj(xn, &pin.q, q, c.hidden)?;
        q_norm(q, pin.q_norm_scale, d, c.rd.tau)?;
        let rbp = RelBiasPath {
            linear: &pin.rel_linear,
            proj: &pin.rel_proj,
            norm_scale: pin.rel_norm_scale,
            states: sc.rel_states.ptr_mut() as *mut f32,
            logits: sc.rel_logits.ptr_mut() as *mut f32,
            bias: sc.bias.ptr_mut() as *mut f32,
        };
        rel_bias_path(xn, &rbp, &c.rd, (c.pos, c.n))?;
        proj(xn, &pin.sink, sc.sink.ptr_mut() as *mut f32, c.hidden)?;
        kv_paths(pin, sc, (xn_k, xn_v), c)?;
        let gio = GqaIo {
            q: sc.q.ptr() as *const f32,
            kc: sc.kc[c.cache_layer].ptr() as *const f32,
            vc: sc.vc[c.cache_layer].ptr() as *const f32,
            bias: sc.bias.ptr() as *const f32,
            sink: sc.sink.ptr() as *const f32,
            attn: sc.attn.ptr_mut() as *mut f32,
            pos: c.pos,
            n: c.n,
            ctx: c.ctx,
        };
        attend_gqa(&gio, d)?;
        let goio = GatedOutIo {
            x: xn,
            attn: sc.attn.ptr_mut() as *mut f32,
            gate: sc.gate.ptr_mut() as *mut f32,
            out: io.out,
        };
        gated_output(pin, &goio, d, c.hidden)
    }
}

/// The k/v decode paths: project the normed k/v inputs and run the short conv into the
/// deinterleaved cache. `xns` is `(xn_k, xn_v)` — the same `xn` on sliding layers, the separate k/v
/// input norms on full layers (T11). Each cache layer has its OWN sconv state, indexed by
/// `c.cache_layer`, so a loop pass neither continues nor clobbers another's window.
///
/// # Safety
/// The caller's `attention` contract; the scratch and pin are sized for the geometry.
unsafe fn kv_paths(
    pin: &super::pin::InklingLayerPin,
    sc: &mut super::engine::InklingScratch,
    xns: (*const f32, *const f32),
    c: &AttnCfg,
) -> Result<()> {
    let (k_w, v_w) = match pin.sconv {
        LayerSconv::Static { k, v } => (k, v),
        LayerSconv::Dynamic { .. } => {
            anyhow::bail!("full-layer dynamic sconv step not yet implemented")
        }
    };
    // SAFETY: the caller's contract; each pointer is a live scratch/pin allocation for the geometry.
    unsafe {
        let kp = KvPath {
            proj_w: &pin.k,
            sconv_w: k_w,
            state: sc.sconv[c.cache_layer][0].ptr_mut() as *mut f32,
            buf: sc.k.ptr_mut() as *mut f32,
            cache: sc.kc[c.cache_layer].ptr_mut() as *mut f32,
            is_key: true,
        };
        kv_path(xns.0, &kp, &c.d, (c.hidden, c.k_size, c.pos, c.ctx))?;
        let vp = KvPath {
            proj_w: &pin.v,
            sconv_w: v_w,
            state: sc.sconv[c.cache_layer][1].ptr_mut() as *mut f32,
            buf: sc.v.ptr_mut() as *mut f32,
            cache: sc.vc[c.cache_layer].ptr_mut() as *mut f32,
            is_key: false,
        };
        kv_path(xns.1, &vp, &c.d, (c.hidden, c.k_size, c.pos, c.ctx))
    }
}
