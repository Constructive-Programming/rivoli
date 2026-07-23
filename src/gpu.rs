//! The GPU decode loop — the resident forward pass. Every per-token op runs
//! on-device against the [`Pin`]'s resident weights, using scratch [`DeviceBuf`]s
//! allocated once and reused each token (no per-token allocation). The only host
//! round-trips are the router-gate logits (MoE layers) and the final logits for
//! argmax — each a small D2H behind a join.
//!
//! Dense attention only (this checkpoint has no DSA indexer), fp8-e4m3 KV latent
//! cache, VQ-int3 routed + shared experts. `rocm`-only.
#![cfg(feature = "rocm")]

use crate::device::DeviceBuf;
use crate::hip::{
    ExpertDescVq, device_sync, launch_append_kv, launch_argmax, launch_attend, launch_embed_i8_row,
    launch_gather_rope, launch_gemv_f32, launch_gemv_fp8, launch_gemv_i8, launch_mla_absorb_fp8,
    launch_mla_value_fp8, launch_moe_vq, launch_rmsnorm, launch_rope, launch_swiglu, launch_vadd,
};
use crate::math::{E4M3_BLOCK, sigmoid, topk_into};
use crate::model::ModelConfig;
use crate::pin::{Fp8Mlp, LayerMlp, MlpVq, Pin};
use anyhow::{Result, bail, ensure};

fn desc_of_vq(m: &MlpVq) -> ExpertDescVq {
    ExpertDescVq {
        gate_indices: m.gate.indices,
        gate_scales: m.gate.scales,
        up_indices: m.up.indices,
        up_scales: m.up.scales,
        down_indices: m.down.indices,
        down_scales: m.down.scales,
    }
}

fn desc_bytes_vq(d: &[ExpertDescVq]) -> &[u8] {
    // SAFETY: ExpertDescVq is repr(C) POD (six pointers); this is its byte view.
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, std::mem::size_of_val(d)) }
}

/// Little-endian byte view of an f32 slice — zero-copy, since on this LE host
/// `[f32]`'s in-memory representation IS its little-endian serialization. Feeds the
/// per-token weight/scalar H2D with no staging buffer.
fn f32_le_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 is POD; the bytes are the LE serialization on this LE host.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Host routing: sigmoid the gate logits into `scores`, add the router `bias` into
/// `choice`, and select the top-`top_k` into `sel`. A free fn taking disjoint slices
/// so the caller can borrow `bias` out of `&self.pin` while it mutably borrows its
/// own routing scratch. Used for both the current layer's routing and the
/// cross-layer L+1 prediction (each with its own scratch triple).
fn route_into(
    gate_logits: &[u8],
    bias: &[f32],
    top_k: usize,
    scores: &mut [f32],
    choice: &mut [f32],
    sel: &mut Vec<usize>,
) {
    for (s, c) in gate_logits.chunks_exact(4).zip(scores.iter_mut()) {
        *c = sigmoid(f32::from_le_bytes([s[0], s[1], s[2], s[3]]));
    }
    for ((c, &s), &b) in choice.iter_mut().zip(scores.iter()).zip(bias) {
        *c = s + b;
    }
    topk_into(choice, top_k, sel);
}

/// Per-token time buckets, measured (not theorized). `trace`-only: the clock reads
/// buy nothing in a production build.
#[cfg(feature = "trace")]
#[derive(Default)]
struct Profile {
    fetch_ns: u128,
    fetch_n: u64,
    mlp_ns: u128,
    route_ns: u128,
    wall_ns: u128,
    tokens: u64,
}

#[cfg(feature = "trace")]
impl Profile {
    fn report(&self) {
        let tok = self.tokens.max(1) as f64;
        let per = |ns: u128| ns as f64 / 1e6 / tok; // ms/token
        tracing::info!(
            "PROFILE/tok: {:.0}ms wall | fetch {:.0}ms ({} miss, {:.2}ms/miss) | mlp {:.0}ms | route {:.0}ms",
            per(self.wall_ns),
            per(self.fetch_ns),
            self.fetch_n / self.tokens.max(1),
            self.fetch_ns as f64 / 1e6 / self.fetch_n.max(1) as f64,
            per(self.mlp_ns),
            per(self.route_ns),
        );
        let (read_ns, copy_ns) = crate::stream::ring_timings();
        tracing::info!(
            "  fetch split/tok: nvme-read {:.0}ms | bounce-copy {:.0}ms",
            per(read_ns as u128),
            per(copy_ns as u128),
        );
    }
}

pub struct GpuEngine<'a> {
    pin: Pin<'a>,
    cfg: &'a ModelConfig,
    /// KV-slab capacity in tokens; forward() refuses pos beyond it.
    max_ctx: usize,
    // Per-token device scratch (allocated once, reused).
    x: DeviceBuf,
    xn: DeviceBuf,
    sub: DeviceBuf,
    qr: DeviceBuf,
    q: DeviceBuf,
    comp: DeviceBuf,
    qabs: DeviceBuf,
    qrope: DeviceBuf,
    clat: DeviceBuf,
    /// Split-KV partial scratch, sized ONCE for the attend kernel's worst-case split
    /// count so every context length reuses it.
    attn_partial: DeviceBuf,
    ctx: DeviceBuf,
    gate_logits: DeviceBuf,
    // Cross-layer prefetch scratch: L+1's router-gate prediction.
    pred_xn: DeviceBuf,
    pred_gl: DeviceBuf,
    // Dense-MLP fp8 SwiGLU scratch (gate/up projections, dense_inter wide).
    mlp_g: DeviceBuf,
    mlp_u: DeviceBuf,
    moe_out: DeviceBuf,
    moe_partial: DeviceBuf, // [slots*hidden] per-expert outputs (deterministic reduce)
    moe_h: DeviceBuf,       // [slots*moe_inter] SwiGLU hidden scratch (VQ MoE)
    descs_buf: DeviceBuf,
    wexpert_buf: DeviceBuf,
    logits: DeviceBuf,
    /// Device argmax result: 8 bytes [i32 index | f32 max-value].
    argmax_dev: DeviceBuf,
    // Per-layer fp8 KV latent cache, grown in place to max_ctx: `lc` is e4m3
    // (max_ctx*kvl u8), `lc_scale` the per-128 block scales (max_ctx*n_blocks f32),
    // `rc` the roped key (max_ctx*rope u16, always bf16).
    lc: Vec<DeviceBuf>,
    lc_scale: Vec<DeviceBuf>,
    rc: Vec<DeviceBuf>,
    n_kv_blocks: usize, // kvl / E4M3_BLOCK
    heartbeat: Option<crate::watchdog::Heartbeat>,
    // Host routing/argmax scratch.
    scores: Vec<f32>,
    choice: Vec<f32>,
    sel: Vec<usize>,
    // Per-token host build scratch — reused every layer so the hot path allocates
    // nothing: resolved VQ descriptors + weights, the resolved batch, D2H staging.
    w: Vec<f32>,
    codebooks: [*const f32; 3],
    mlps_vq: Vec<MlpVq>,
    descs_vq: Vec<ExpertDescVq>,
    gl_host: Vec<u8>,
    pgl_host: Vec<u8>,
    argmax_host: Vec<u8>,
    // Cross-layer prefetch: separate host scratch for the L+1 prediction top-k.
    pred_scores: Vec<f32>,
    pred_choice: Vec<f32>,
    pred_sel: Vec<usize>,
    prefetch: bool,
    prefetch_depth: usize,
    /// DIAGNOSTIC (`--checksum-x`): hash the residual stream after every layer.
    #[cfg(feature = "trace")]
    checksum_x: bool,
    #[cfg(feature = "trace")]
    ck_buf: Vec<u8>,
    #[cfg(feature = "trace")]
    prof: Profile,
}

impl<'a> GpuEngine<'a> {
    pub fn new(pin: Pin<'a>, cfg: &'a ModelConfig, max_ctx: usize) -> Result<Self> {
        // The MoE block folds the shared expert into the routed batch at a single
        // kernel `inter = moe_inter`. Only valid when the shared expert has the routed
        // width, i.e. n_shared == 1 (GLM-5.2).
        ensure!(
            cfg.n_shared == 1,
            "GPU decode assumes n_shared==1 (shared folded into the routed batch); n_shared={}",
            cfg.n_shared
        );
        let f = |n: usize| DeviceBuf::new(n * 4); // f32 buffer of n elems
        let kvl = cfg.kv_lora_rank;
        let rope = cfg.qk_rope_head_dim;
        let h = cfg.n_heads;
        let slots = cfg.top_k + cfg.n_shared; // routed + shared per MoE launch
        ensure!(
            kvl.is_multiple_of(E4M3_BLOCK),
            "kv_lora_rank ({kvl}) must be a multiple of {E4M3_BLOCK} (fp8 KV block size)",
        );
        let n_kv_blocks = kvl / E4M3_BLOCK;
        let mut lc = Vec::with_capacity(cfg.n_layers);
        let mut lc_scale = Vec::with_capacity(cfg.n_layers);
        let mut rc = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            lc.push(DeviceBuf::new(max_ctx * kvl)?); // e4m3 latent (1 byte)
            lc_scale.push(DeviceBuf::new(max_ctx * n_kv_blocks * 4)?); // f32 block scales
            rc.push(DeviceBuf::new(max_ctx * rope * 2)?); // bf16 roped key
        }
        Ok(Self {
            cfg,
            max_ctx,
            x: f(cfg.hidden)?,
            xn: f(cfg.hidden)?,
            sub: f(cfg.hidden)?,
            qr: f(cfg.q_lora_rank)?,
            q: f(h * cfg.qk_head_dim())?,
            comp: f(kvl + rope)?,
            qabs: f(h * kvl)?,
            qrope: f(h * rope)?,
            clat: f(h * kvl)?,
            attn_partial: f(crate::hip::attend_scratch_floats(h, kvl))?,
            ctx: f(h * cfg.v_head_dim)?,
            gate_logits: f(cfg.n_experts)?,
            pred_xn: f(cfg.hidden)?,
            pred_gl: f(cfg.n_experts)?,
            mlp_g: f(cfg.dense_inter)?,
            mlp_u: f(cfg.dense_inter)?,
            moe_out: f(cfg.hidden)?,
            moe_partial: f(slots * cfg.hidden)?,
            moe_h: f(slots * cfg.moe_inter)?,
            descs_buf: DeviceBuf::new(slots * std::mem::size_of::<ExpertDescVq>())?,
            wexpert_buf: f(slots)?,
            logits: f(cfg.vocab)?,
            argmax_dev: DeviceBuf::new(8)?, // [i32 index | f32 value]
            lc,
            lc_scale,
            rc,
            n_kv_blocks,
            scores: vec![0.0; cfg.n_experts],
            choice: vec![0.0; cfg.n_experts],
            sel: Vec::with_capacity(cfg.top_k),
            w: Vec::with_capacity(slots),
            codebooks: pin.codebooks(),
            mlps_vq: Vec::with_capacity(cfg.top_k),
            descs_vq: Vec::with_capacity(slots),
            gl_host: Vec::with_capacity(cfg.n_experts * 4),
            pgl_host: Vec::with_capacity(cfg.n_experts * 4),
            argmax_host: Vec::with_capacity(8),
            pred_scores: vec![0.0; cfg.n_experts],
            pred_choice: vec![0.0; cfg.n_experts],
            pred_sel: Vec::with_capacity(cfg.top_k),
            prefetch: pin.prefetch_enabled(),
            prefetch_depth: pin.prefetch_depth(),
            #[cfg(feature = "trace")]
            checksum_x: false,
            #[cfg(feature = "trace")]
            ck_buf: Vec::new(),
            #[cfg(feature = "trace")]
            prof: Profile::default(),
            heartbeat: None,
            pin,
        })
    }

    /// Attach a wedge-watchdog heartbeat; the decode loop beats it each token.
    pub fn set_heartbeat(&mut self, hb: crate::watchdog::Heartbeat) {
        self.heartbeat = Some(hb);
    }

    pub fn hits(&self) -> u64 {
        self.pin.hits
    }
    pub fn misses(&self) -> u64 {
        self.pin.misses
    }
    /// Intra-batch slot-reuse collisions caught by the streaming guard (each one
    /// would have been a silently corrupted expert before the fix).
    pub fn slot_collisions(&self) -> u64 {
        self.pin.slot_collisions
    }

    /// DIAGNOSTIC: hash the residual stream after every layer (`--checksum-x`).
    #[cfg(feature = "trace")]
    pub fn set_checksum_x(&mut self, on: bool) {
        self.checksum_x = on;
    }

    /// Cross-layer prefetch recall: (predicted experts actually selected, predicted).
    #[cfg(feature = "trace")]
    pub fn prefetch_recall(&self) -> (u64, u64) {
        (self.pin.pred_correct, self.pin.pred_total)
    }
    /// Where routed experts' bytes came from (loaded/preloading/cold/pf_evict_*).
    #[cfg(feature = "trace")]
    pub fn source_split(&self) -> (u64, u64, u64, u64, u64) {
        self.pin.source_split()
    }
    #[cfg(feature = "trace")]
    pub fn prefetch_wait_ms(&self) -> f64 {
        self.pin.prefetch_wait_ns as f64 / 1e6
    }

    /// One forward pass for `token` at `pos`, leaving next-token logits device-side
    /// in `self.logits`.
    fn forward(&mut self, token: u32, pos: usize) -> Result<()> {
        let cfg = self.cfg;
        let eps = cfg.rms_norm_eps as f32;
        let (h, qh, nope, rope, kvl, vh, hidden) = (
            cfg.n_heads,
            cfg.qk_head_dim(),
            cfg.qk_nope_head_dim,
            cfg.qk_rope_head_dim,
            cfg.kv_lora_rank,
            cfg.v_head_dim,
            cfg.hidden,
        );
        let theta = cfg.rope_theta();
        let scale = 1.0 / (qh as f32).sqrt();
        let nb = self.n_kv_blocks;

        // Raw scratch pointers (Copy — don't hold borrows across the launches).
        let xp = self.x.ptr_mut() as *mut f32;
        let xnp = self.xn.ptr_mut() as *mut f32;
        let subp = self.sub.ptr_mut() as *mut f32;
        let qrp = self.qr.ptr_mut() as *mut f32;
        let qp = self.q.ptr_mut() as *mut f32;
        let compp = self.comp.ptr_mut() as *mut f32;
        let qabsp = self.qabs.ptr_mut() as *mut f32;
        let qropep = self.qrope.ptr_mut() as *mut f32;
        let clatp = self.clat.ptr_mut() as *mut f32;
        let apartp = self.attn_partial.ptr_mut() as *mut f32;
        let ctxp = self.ctx.ptr_mut() as *mut f32;
        let glp = self.gate_logits.ptr_mut() as *mut f32;

        // The KV slabs are sized to max_ctx; writing row pos beyond that is a device
        // out-of-bounds write, so refuse here rather than corrupt device memory.
        ensure!(
            pos < self.max_ctx,
            "pos {pos} exceeds engine capacity max_ctx={}",
            self.max_ctx
        );

        // Dense attention: attend over the whole causal prefix `0..pos+1`.
        let nr = pos + 1;

        // Embedding row → x.
        // SAFETY: all pointers are device-resident scratch/weights valid for their
        // dims; each launch's inputs are produced by a prior launch on the same
        // (default) stream, so ordering holds; a device_sync precedes every host read.
        unsafe {
            launch_embed_i8_row(self.pin.embed.packed, self.pin.embed.scale, token as usize, hidden, xp)?;
        }

        for l in 0..cfg.n_layers {
            // Copy the layer's weight pointers out (ends the &pin.layers borrow).
            let lw = &self.pin.layers[l];
            let (input_ln, post_ln) = (lw.input_ln, lw.post_ln);
            let (q_a, q_a_ln, q_b) = (lw.q_a, lw.q_a_ln, lw.q_b);
            let (kv_a, kv_a_ln, kv_b) = (lw.kv_a, lw.kv_a_ln, lw.kv_b);
            let o_proj = lw.o_proj;
            let dense_mlp: Option<Fp8Mlp> = match &lw.mlp {
                LayerMlp::Dense(m) => Some(*m),
                LayerMlp::Moe { .. } => None,
            };
            let (gate_w, shared) = match &lw.mlp {
                LayerMlp::Moe { gate_w, shared } => (*gate_w, Some(*shared)),
                LayerMlp::Dense(_) => (std::ptr::null(), None),
            };

            let lc8p = self.lc[l].ptr_mut();
            let lscalep = self.lc_scale[l].ptr_mut() as *mut f32;
            let rcp = self.rc[l].ptr_mut() as *mut u16;

            // --- Attention phase 1: projections, ropes, cache append, absorb. ---
            // SAFETY: see the forward-level note; every pointer is live scratch.
            unsafe {
                launch_rmsnorm(xp, input_ln, hidden, eps, xnp)?;
                launch_gemv_fp8(xnp, q_a.packed, q_a.scale, q_a.o_dim, q_a.i_dim, q_a.block, qrp)?;
                launch_rmsnorm(qrp, q_a_ln, cfg.q_lora_rank, eps, qrp)?; // in-place
                launch_gemv_fp8(qrp, q_b.packed, q_b.scale, q_b.o_dim, q_b.i_dim, q_b.block, qp)?;
                launch_gemv_fp8(xnp, kv_a.packed, kv_a.scale, kv_a.o_dim, kv_a.i_dim, kv_a.block, compp)?;
                launch_rmsnorm(compp, kv_a_ln, kvl, eps, compp)?; // normalize latent (first kvl)
                launch_rope(compp.add(kvl), 1, rope, rope, pos, theta)?; // rope the key
                launch_rope(qp.add(nope), h, qh, rope, pos, theta)?; // rope per-head query
                launch_append_kv(compp, compp.add(kvl), lc8p, lscalep, rcp, pos, kvl, rope, nb)?;
                launch_mla_absorb_fp8(qp, kv_b.packed, kv_b.scale, h, qh, nope, vh, kvl, kv_b.block, qabsp)?;
                launch_gather_rope(qp, qropep, h, qh, nope, rope)?;
            }

            // --- Attention phase 2: dense flash attend, value + output projection,
            //     residual, pre-MLP norm. ---
            // SAFETY: see the forward-level note; every pointer is live scratch.
            unsafe {
                launch_attend(qabsp, qropep, lc8p, lscalep, rcp, h, nr, kvl, rope, nb, scale, clatp, apartp)?;
                launch_mla_value_fp8(clatp, kv_b.packed, kv_b.scale, h, nope, vh, kvl, kv_b.block, ctxp)?;
                launch_gemv_fp8(ctxp, o_proj.packed, o_proj.scale, o_proj.o_dim, o_proj.i_dim, o_proj.block, subp)?;
                launch_vadd(xp, subp, hidden)?; // residual
                launch_rmsnorm(xp, post_ln, hidden, eps, xnp)?; // pre-MLP norm → xn
            }

            // --- MLP sublayer (out fully written; the outer vadd adds moe_out) ---
            if let Some(m) = dense_mlp {
                let inter = m.gate.o_dim;
                // fp8 SwiGLU: gate/up projections, silu-combine, down projection.
                // SAFETY: weights resident; mlp_g/mlp_u/moe_out device scratch.
                unsafe {
                    let gp = self.mlp_g.ptr_mut() as *mut f32;
                    let up = self.mlp_u.ptr_mut() as *mut f32;
                    let outp = self.moe_out.ptr_mut() as *mut f32;
                    launch_gemv_fp8(xnp, m.gate.packed, m.gate.scale, m.gate.o_dim, m.gate.i_dim, m.gate.block, gp)?;
                    launch_gemv_fp8(xnp, m.up.packed, m.up.scale, m.up.o_dim, m.up.i_dim, m.up.block, up)?;
                    launch_swiglu(gp, up, inter, gp)?; // in place: h = silu(gate)*up
                    launch_gemv_fp8(gp, m.down.packed, m.down.scale, m.down.o_dim, m.down.i_dim, m.down.block, outp)?;
                }
            } else {
                // Cross-layer prefetch: if the NEXT layer is also MoE, predict its
                // routed experts from L's post-attn residual `xp` (a cheap proxy for
                // L+1's input). The router gate + input_ln are resident, so this is a
                // small norm + gemv folded under the same routing sync below.
                let predict = self.prefetch && l + 1 < cfg.n_layers && l + 1 >= cfg.dense_layers;
                let next_pred = if predict {
                    let nl = &self.pin.layers[l + 1];
                    if let LayerMlp::Moe { gate_w, .. } = &nl.mlp {
                        Some((nl.input_ln, *gate_w))
                    } else {
                        None
                    }
                } else {
                    None
                };
                // Router gate on device, then read logits to route on host.
                // SAFETY: gate_w resident F32; glp device scratch.
                unsafe { launch_gemv_f32(xnp, gate_w, cfg.n_experts, hidden, glp)? };
                if let Some((next_ln, next_gate)) = next_pred {
                    // SAFETY: xp is the live post-attn residual; next_ln/next_gate are
                    // resident L+1 weights; pred_xn/pred_gl device scratch, same stream.
                    unsafe {
                        launch_rmsnorm(xp, next_ln, hidden, eps, self.pred_xn.ptr_mut() as *mut f32)?;
                        launch_gemv_f32(
                            self.pred_xn.ptr() as *const f32,
                            next_gate,
                            cfg.n_experts,
                            hidden,
                            self.pred_gl.ptr_mut() as *mut f32,
                        )?;
                    }
                }
                #[cfg(feature = "trace")]
                let t = std::time::Instant::now();
                // Read the gate logits, route with `bias` borrowed straight out of
                // `&self.pin` while the routing scratch is borrowed mutably.
                self.gate_logits.copy_out_into(&mut self.gl_host)?;
                route_into(
                    &self.gl_host,
                    self.pin.moe_bias(l),
                    cfg.top_k,
                    &mut self.scores,
                    &mut self.choice,
                    &mut self.sel,
                );
                if next_pred.is_some() {
                    self.pred_gl.copy_out_into(&mut self.pgl_host)?;
                    route_into(
                        &self.pgl_host,
                        self.pin.moe_bias(l + 1),
                        cfg.top_k,
                        &mut self.pred_scores,
                        &mut self.pred_choice,
                        &mut self.pred_sel,
                    );
                }
                #[cfg(feature = "trace")]
                {
                    self.prof.route_ns += t.elapsed().as_nanos();
                }
                // SUBMIT this layer's cold reads and take the descriptors back
                // immediately — the slot ADDRESSES are known at allocation time, only
                // the bytes are still arriving. Everything until `await_layer` runs
                // with the NVMe busy.
                #[cfg(feature = "trace")]
                let miss0 = self.pin.misses;
                #[cfg(feature = "trace")]
                let t = std::time::Instant::now();
                let batch = self.pin.submit_layer(l, &self.sel, &mut self.mlps_vq)?;
                #[cfg(feature = "trace")]
                {
                    self.prof.fetch_ns += t.elapsed().as_nanos();
                    self.prof.fetch_n += self.pin.misses - miss0;
                }
                // Submit L+1's predicted-expert reads NOW (non-blocking) while this
                // layer's demand reads are still in flight, so both batches sit in the
                // device queue together. Reaped by `submit_layer(l+1)`'s prefetch drain.
                if next_pred.is_some() {
                    let n = self.prefetch_depth.min(self.pred_sel.len());
                    self.pin.prefetch_layer(l + 1, &self.pred_sel[..n])?;
                }
                // Routed weights: sigmoid score, sum-normalized over the routed picks,
                // then scaled. The VQ shared expert (weight 1.0) folds into the batch.
                self.w.clear();
                for &e in &self.sel {
                    self.w.push(self.scores[e]);
                }
                let mut sm: f32 = self.w.iter().sum();
                if cfg.norm_topk_prob {
                    sm += 1e-20;
                    for wi in self.w.iter_mut() {
                        *wi /= sm;
                    }
                }
                for wi in self.w.iter_mut() {
                    *wi *= cfg.routed_scale as f32;
                }
                // VQ routed descriptors + the folded VQ shared expert (weight 1.0).
                self.descs_vq.clear();
                for m in &self.mlps_vq {
                    self.descs_vq.push(desc_of_vq(m));
                }
                if let Some(s) = shared {
                    self.descs_vq.push(desc_of_vq(&s));
                    self.w.push(1.0);
                }
                let ndesc = self.descs_vq.len();
                self.descs_buf.copy_in_at(0, desc_bytes_vq(&self.descs_vq))?;
                self.wexpert_buf.copy_in_at(0, f32_le_bytes(&self.w))?;
                // JOIN the cold reads — last moment before the first deref of a cold
                // slot. No kernel live, so the bounce drain's sync serialises nothing.
                #[cfg(feature = "trace")]
                let t = std::time::Instant::now();
                self.pin.await_layer(batch)?;
                #[cfg(feature = "trace")]
                {
                    self.prof.fetch_ns += t.elapsed().as_nanos();
                }
                // SAFETY: descs point at landed slot indices/scales + resident codebooks.
                unsafe {
                    launch_moe_vq(
                        xnp,
                        hidden,
                        cfg.moe_inter,
                        ndesc,
                        self.descs_buf.ptr() as *const ExpertDescVq,
                        self.codebooks[0],
                        self.codebooks[1],
                        self.codebooks[2],
                        self.wexpert_buf.ptr() as *const f32,
                        self.moe_h.ptr_mut() as *mut f32,
                        self.moe_partial.ptr_mut() as *mut f32,
                        self.moe_out.ptr_mut() as *mut f32,
                    )?;
                }
            }
            // SAFETY: residual add of the MLP contribution.
            unsafe { launch_vadd(xp, self.moe_out.ptr() as *const f32, hidden)? };
            // End-of-layer join: protects the reused descs/wexpert/moe_out buffers
            // before the next layer overwrites them, and surfaces faults.
            #[cfg(feature = "trace")]
            let t = std::time::Instant::now();
            device_sync()?;
            #[cfg(feature = "trace")]
            {
                self.prof.mlp_ns += t.elapsed().as_nanos();
            }
            // DIAGNOSTIC (`--checksum-x`): hash the residual stream after every layer.
            #[cfg(feature = "trace")]
            if self.checksum_x {
                let n = hidden * 4;
                // SAFETY: `x` is `hidden` f32; the sync above retired every writer.
                unsafe { DeviceBuf::copy_out_raw(self.x.ptr(), n, &mut self.ck_buf)? };
                let mut hh: u64 = 0xcbf2_9ce4_8422_2325;
                for &b in self.ck_buf.iter() {
                    hh ^= b as u64;
                    hh = hh.wrapping_mul(0x1000_0000_01b3);
                }
                tracing::info!("XSUM pos={pos} l={l} x={hh:016x}");
            }
        }

        // Final norm → lm_head → logits (device); caller reads via argmax.
        // SAFETY: final_norm/lm_head resident; xn/logits device scratch.
        unsafe {
            launch_rmsnorm(xp, self.pin.final_norm, hidden, eps, xnp)?;
            let head = self.pin.lm_head;
            launch_gemv_i8(xnp, head.packed, head.scale, head.o_dim, head.i_dim, self.logits.ptr_mut() as *mut f32)?;
        }
        Ok(())
    }

    /// Greedy argmax over the device logits — reduced ON DEVICE, so only 8 bytes come
    /// back per token. The kernel reproduces the host fold exactly (strict `>`: ties
    /// keep the lowest index, NaN never wins), returning `logits[best]` so the
    /// finiteness bail is the same `!value.is_finite()` check.
    fn argmax(&mut self) -> Result<u32> {
        // SAFETY: logits is `vocab` device f32 (written + joined); argmax_dev owns 8
        // device bytes for [i32 index|f32 value].
        unsafe {
            launch_argmax(
                self.logits.ptr() as *const f32,
                self.cfg.vocab,
                self.argmax_dev.ptr_mut() as *mut i32,
                self.argmax_dev.ptr_mut().add(4) as *mut f32,
            )?;
        }
        self.argmax_dev.copy_out_into(&mut self.argmax_host)?;
        debug_assert_eq!(self.argmax_host.len(), 8, "argmax result must be 8 bytes");
        let idx = i32::from_le_bytes([
            self.argmax_host[0],
            self.argmax_host[1],
            self.argmax_host[2],
            self.argmax_host[3],
        ]);
        let val = f32::from_le_bytes([
            self.argmax_host[4],
            self.argmax_host[5],
            self.argmax_host[6],
            self.argmax_host[7],
        ]);
        if !val.is_finite() {
            bail!("logits are non-finite (NaN/Inf in the GPU forward pass)");
        }
        debug_assert!(idx >= 0, "argmax returned negative index {idx}");
        Ok(idx as u32)
    }

    /// Greedy-decode up to `ngen` tokens continuing `prompt_ids`, stopping on any
    /// `eos`. Returns the generated ids.
    pub fn generate(&mut self, prompt_ids: &[u32], ngen: usize, eos: &[u32]) -> Result<Vec<u32>> {
        ensure!(!prompt_ids.is_empty(), "empty prompt");
        let mut pos = 0usize;
        for &tok in prompt_ids {
            self.forward(tok, pos)?;
            pos += 1;
        }
        // Profile the DECODE loop only (prefill is warm-up).
        #[cfg(feature = "trace")]
        {
            self.prof = Profile::default();
        }
        #[cfg(feature = "trace")]
        let decode_wall = std::time::Instant::now();
        let mut generated = Vec::with_capacity(ngen);
        #[cfg(feature = "trace")]
        const WIN: usize = 8;
        #[cfg(feature = "trace")]
        let mut win_t = std::time::Instant::now();
        #[cfg(feature = "trace")]
        let (mut win_hit, mut win_miss) = (self.pin.hits, self.pin.misses);
        for _i in 0..ngen {
            if let Some(hb) = &self.heartbeat {
                hb.beat();
            }
            let next = self.argmax()?;
            if eos.contains(&next) {
                break;
            }
            generated.push(next);
            self.forward(next, pos)?;
            pos += 1;
            #[cfg(feature = "trace")]
            if (_i + 1) % WIN == 0 {
                let dt = win_t.elapsed().as_secs_f64();
                let (dh, dm) = (self.pin.hits - win_hit, self.pin.misses - win_miss);
                let hit_pct = 100.0 * dh as f64 / (dh + dm).max(1) as f64;
                tracing::info!(
                    "  tok {}/{ngen}: {:.3} tok/s (window), hit {hit_pct:.1}%",
                    _i + 1,
                    WIN as f64 / dt.max(1e-9),
                );
                win_t = std::time::Instant::now();
                (win_hit, win_miss) = (self.pin.hits, self.pin.misses);
            }
        }
        #[cfg(feature = "trace")]
        {
            self.prof.wall_ns = decode_wall.elapsed().as_nanos();
            self.prof.tokens = generated.len() as u64;
            self.prof.report();
        }
        Ok(generated)
    }
}
