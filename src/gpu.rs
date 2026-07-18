//! The GPU decode loop — the M3 (4/4) resident forward pass. Every per-token
//! op runs on-device against the [`Pin`]'s resident weights, using scratch
//! [`DeviceBuf`]s allocated once and reused each token (no per-token allocation).
//! The only host round-trips are the router gate logits (MoE layers) and the
//! final logits for argmax — each a small D2H behind a join. This is the path
//! whose speed the ≥1 tok/s gate measures; correctness is checked by decoding the
//! same coherent continuation as the scalar reference.
//!
//! `rocm`-only.
#![cfg(feature = "rocm")]

use crate::device::DeviceBuf;
use crate::hip::{
    ExpertDesc, device_sync, launch_append_kv, launch_attend, launch_embed_i8_row,
    launch_gather_rope, launch_gemv_f32, launch_gemv_i4, launch_gemv_i8, launch_mla_absorb,
    launch_mla_value, launch_moe, launch_rmsnorm, launch_rope, launch_vadd,
};
use crate::math::{sigmoid, topk_into};
use crate::model::ModelConfig;
use crate::pin::{LayerMlp, Mlp, Pin};
use anyhow::{Result, bail, ensure};

fn desc_of(m: &Mlp) -> ExpertDesc {
    ExpertDesc {
        gate_packed: m.gate.packed,
        gate_scale: m.gate.scale,
        up_packed: m.up.packed,
        up_scale: m.up.scale,
        down_packed: m.down.packed,
        down_scale: m.down.scale,
    }
}

fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn desc_bytes(d: &[ExpertDesc]) -> &[u8] {
    // SAFETY: ExpertDesc is repr(C) POD (six pointers); this is its byte view.
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, std::mem::size_of_val(d)) }
}

pub struct GpuEngine<'a> {
    pin: Pin<'a>,
    cfg: &'a ModelConfig,
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
    ctx: DeviceBuf,
    gate_logits: DeviceBuf,
    moe_out: DeviceBuf,
    descs_buf: DeviceBuf,
    wexpert_buf: DeviceBuf,
    logits: DeviceBuf,
    // Per-layer bf16 KV slabs, grown in place to max_ctx.
    lc: Vec<DeviceBuf>,
    rc: Vec<DeviceBuf>,
    // Host routing/argmax scratch.
    scores: Vec<f32>,
    choice: Vec<f32>,
    sel: Vec<usize>,
    // Background page-cache warmer + the previous token's routed selection per
    // sparse layer (the predictor): at each token's start we warm the experts the
    // last token routed to, overlapping their fetch with this token's compute.
    prefetch: crate::prefetch::Prefetcher,
    last_sel: Vec<Vec<usize>>,
}

impl<'a> GpuEngine<'a> {
    pub fn new(pin: Pin<'a>, cfg: &'a ModelConfig, max_ctx: usize) -> Result<Self> {
        // The MoE block folds the shared expert into the routed batch (D6) at a
        // single kernel `inter = moe_inter`. That is only valid when the shared
        // expert has the routed width, i.e. n_shared == 1 (GLM-5.2). A wider
        // shared expert would need its own launch at moe_inter*n_shared — refuse
        // loudly rather than silently misread its rows.
        ensure!(
            cfg.n_shared == 1,
            "GPU decode assumes n_shared==1 (shared folded into the routed batch); \
             n_shared={} needs a separate shared launch",
            cfg.n_shared
        );
        let f = |n: usize| DeviceBuf::new(n * 4); // f32 buffer of n elems
        let kvl = cfg.kv_lora_rank;
        let rope = cfg.qk_rope_head_dim;
        let h = cfg.n_heads;
        let slots = cfg.top_k + cfg.n_shared; // routed + shared per MoE launch
        let mut lc = Vec::with_capacity(cfg.n_layers);
        let mut rc = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            lc.push(DeviceBuf::new(max_ctx * kvl * 2)?);
            rc.push(DeviceBuf::new(max_ctx * rope * 2)?);
        }
        Ok(Self {
            cfg,
            x: f(cfg.hidden)?,
            xn: f(cfg.hidden)?,
            sub: f(cfg.hidden)?,
            qr: f(cfg.q_lora_rank)?,
            q: f(h * cfg.qk_head_dim())?,
            comp: f(kvl + rope)?,
            qabs: f(h * kvl)?,
            qrope: f(h * rope)?,
            clat: f(h * kvl)?,
            ctx: f(h * cfg.v_head_dim)?,
            gate_logits: f(cfg.n_experts)?,
            moe_out: f(cfg.hidden)?,
            descs_buf: DeviceBuf::new(slots * std::mem::size_of::<ExpertDesc>())?,
            wexpert_buf: f(slots)?,
            logits: f(cfg.vocab)?,
            lc,
            rc,
            scores: vec![0.0; cfg.n_experts],
            choice: vec![0.0; cfg.n_experts],
            sel: Vec::with_capacity(cfg.top_k),
            prefetch: crate::prefetch::Prefetcher::new(),
            last_sel: vec![Vec::new(); cfg.n_layers - cfg.dense_layers],
            pin,
        })
    }

    pub fn hits(&self) -> u64 {
        self.pin.hits
    }
    pub fn misses(&self) -> u64 {
        self.pin.misses
    }

    /// One forward pass for `token` at `pos`, leaving next-token logits device-
    /// side in `self.logits`.
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
        let ctxp = self.ctx.ptr_mut() as *mut f32;
        let glp = self.gate_logits.ptr_mut() as *mut f32;

        // Embedding row → x.
        // SAFETY: all pointers below are device-resident scratch/weights valid
        // for their dims; each launch's inputs are produced by a prior launch on
        // the same (default) stream, so ordering is guaranteed; a device_sync
        // precedes every host read. Buffers are never freed mid-forward.
        unsafe {
            launch_embed_i8_row(
                self.pin.embed.packed,
                self.pin.embed.scale,
                token as usize,
                hidden,
                xp,
            )?;
        }

        // Streaming feed: warm the pages of the experts the PREVIOUS token routed
        // to (a stable predictor) so their NVMe read overlaps this token's compute
        // and the later copy_in hits the page cache, not cold disk.
        {
            let mut ranges = Vec::new();
            for (si, sel) in self.last_sel.iter().enumerate() {
                let layer = si + cfg.dense_layers;
                for &e in sel {
                    self.pin.cold_warm_ranges(layer, e, &mut ranges)?;
                }
            }
            self.prefetch.warm(ranges);
        }

        for l in 0..cfg.n_layers {
            // Copy the layer's weight pointers out (ends the &pin.layers borrow).
            let lw = &self.pin.layers[l];
            let (input_ln, post_ln) = (lw.input_ln, lw.post_ln);
            let (q_a, q_a_ln, q_b) = (lw.q_a, lw.q_a_ln, lw.q_b);
            let (kv_a, kv_a_ln, kv_b) = (lw.kv_a, lw.kv_a_ln, lw.kv_b);
            let o_proj = lw.o_proj;
            let is_dense = matches!(lw.mlp, LayerMlp::Dense(_));
            let dense_mlp = if let LayerMlp::Dense(m) = &lw.mlp {
                Some(*m)
            } else {
                None
            };
            let (gate_w, shared) = if let LayerMlp::Moe { gate_w, shared } = &lw.mlp {
                (*gate_w, Some(*shared))
            } else {
                (std::ptr::null(), None)
            };

            let lcp = self.lc[l].ptr_mut() as *mut u16;
            let rcp = self.rc[l].ptr_mut() as *mut u16;

            // --- Attention sublayer (all device) ---
            // SAFETY: see the forward-level note; every pointer is live scratch.
            unsafe {
                launch_rmsnorm(xp, input_ln, hidden, eps, xnp)?;
                launch_gemv_i4(xnp, q_a.packed, q_a.scale, q_a.o_dim, q_a.i_dim, qrp)?;
                launch_rmsnorm(qrp, q_a_ln, cfg.q_lora_rank, eps, qrp)?; // in-place
                launch_gemv_i4(qrp, q_b.packed, q_b.scale, q_b.o_dim, q_b.i_dim, qp)?;
                launch_gemv_i4(xnp, kv_a.packed, kv_a.scale, kv_a.o_dim, kv_a.i_dim, compp)?;
                launch_rmsnorm(compp, kv_a_ln, kvl, eps, compp)?; // normalize latent (first kvl)
                launch_rope(compp.add(kvl), 1, rope, rope, pos, theta)?; // rope the key
                launch_rope(qp.add(nope), h, qh, rope, pos, theta)?; // rope per-head query
                launch_append_kv(compp, compp.add(kvl), lcp, rcp, pos, kvl, rope)?;
                launch_mla_absorb(qp, kv_b.packed, kv_b.scale, h, qh, nope, vh, kvl, qabsp)?;
                launch_gather_rope(qp, qropep, h, qh, nope, rope)?;
                launch_attend(qabsp, qropep, lcp, rcp, h, pos + 1, kvl, rope, scale, clatp)?;
                launch_mla_value(clatp, kv_b.packed, kv_b.scale, h, nope, vh, kvl, ctxp)?;
                launch_gemv_i4(
                    ctxp,
                    o_proj.packed,
                    o_proj.scale,
                    o_proj.o_dim,
                    o_proj.i_dim,
                    subp,
                )?;
                launch_vadd(xp, subp, hidden)?; // residual
                launch_rmsnorm(xp, post_ln, hidden, eps, xnp)?; // pre-MLP norm → xn
            }

            // --- MLP sublayer ---
            self.moe_out.zero()?;
            if is_dense {
                let m = dense_mlp.ok_or_else(|| anyhow::anyhow!("dense layer {l} missing mlp"))?;
                let d = [desc_of(&m)];
                self.descs_buf.copy_in_at(0, desc_bytes(&d))?;
                self.wexpert_buf.copy_in_at(0, &f32_le(&[1.0]))?;
                // SAFETY: descs/wexpert/out are device scratch; weights resident.
                unsafe {
                    launch_moe(
                        xnp,
                        hidden,
                        cfg.dense_inter,
                        1,
                        self.descs_buf.ptr() as *const ExpertDesc,
                        self.wexpert_buf.ptr() as *const f32,
                        self.moe_out.ptr_mut() as *mut f32,
                    )?;
                }
            } else {
                // Router gate on device, then read logits to route on host.
                // SAFETY: gate_w resident F32; glp device scratch.
                unsafe { launch_gemv_f32(xnp, gate_w, cfg.n_experts, hidden, glp)? };
                device_sync()?; // read gate logits
                let gl = self.gate_logits.copy_out()?;
                let bias = self.pin.moe_bias(l).to_vec();
                self.route(&gl, &bias);
                // Remember this layer's selection to prefetch next token (stable).
                let ls = &mut self.last_sel[l - cfg.dense_layers];
                ls.clear();
                ls.extend_from_slice(&self.sel);
                self.pin.begin_layer();
                // Resolve selected experts (+ shared), build the descriptor batch.
                let ke = self.sel.len();
                let mut descs = Vec::with_capacity(ke + cfg.n_shared);
                let mut w = Vec::with_capacity(ke + cfg.n_shared);
                for i in 0..ke {
                    let e = self.sel[i];
                    let m = self.pin.expert(l, e)?;
                    descs.push(desc_of(&m));
                    w.push(self.scores[e]);
                }
                // Weight = original sigmoid score, sum-normalized then scaled.
                let mut sm: f32 = w.iter().sum();
                if cfg.norm_topk_prob {
                    sm += 1e-20;
                    for wi in w.iter_mut() {
                        *wi /= sm;
                    }
                }
                for wi in w.iter_mut() {
                    *wi *= cfg.routed_scale as f32;
                }
                // Shared expert(s), weight 1.0.
                if let Some(s) = shared {
                    descs.push(desc_of(&s));
                    w.push(1.0);
                }
                self.descs_buf.copy_in_at(0, desc_bytes(&descs))?;
                self.wexpert_buf.copy_in_at(0, &f32_le(&w))?;
                // SAFETY: descs point at resident/cold-slot weights valid until
                // the end-of-layer sync; out zeroed; all device-resident.
                unsafe {
                    launch_moe(
                        xnp,
                        hidden,
                        cfg.moe_inter,
                        descs.len(),
                        self.descs_buf.ptr() as *const ExpertDesc,
                        self.wexpert_buf.ptr() as *const f32,
                        self.moe_out.ptr_mut() as *mut f32,
                    )?;
                }
            }
            // SAFETY: residual add of the MLP contribution.
            unsafe { launch_vadd(xp, self.moe_out.ptr() as *const f32, hidden)? };
            // End-of-layer join: protects the reused descs/wexpert/moe_out
            // buffers before the next layer overwrites them, and surfaces faults.
            device_sync()?;
        }

        // Final norm → lm_head → logits (device); caller reads via argmax.
        // SAFETY: final_norm/lm_head resident; xn/logits device scratch.
        unsafe {
            launch_rmsnorm(xp, self.pin.final_norm, hidden, eps, xnp)?;
            let head = self.pin.lm_head;
            launch_gemv_i8(
                xnp,
                head.packed,
                head.scale,
                head.o_dim,
                head.i_dim,
                self.logits.ptr_mut() as *mut f32,
            )?;
        }
        device_sync()?;
        Ok(())
    }

    /// Host routing: sigmoid scores into `self.scores`, `choice = score + bias`,
    /// top-k into `self.sel` (mirrors moe.rs exactly).
    fn route(&mut self, gate_logits: &[u8], bias: &[f32]) {
        for (s, c) in gate_logits.chunks_exact(4).zip(self.scores.iter_mut()) {
            *c = sigmoid(f32::from_le_bytes([s[0], s[1], s[2], s[3]]));
        }
        for ((c, &s), &b) in self.choice.iter_mut().zip(&self.scores).zip(bias) {
            *c = s + b;
        }
        topk_into(&self.choice, self.cfg.top_k, &mut self.sel);
    }

    /// Greedy argmax over the device logits (one D2H after the final join).
    fn argmax(&self) -> Result<u32> {
        let logits = self.logits.copy_out()?;
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, c) in logits.chunks_exact(4).enumerate() {
            let l = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if l > bv {
                bv = l;
                best = i;
            }
        }
        if !bv.is_finite() {
            bail!("logits are non-finite (NaN/Inf in the GPU forward pass)");
        }
        Ok(best as u32)
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
        let mut generated = Vec::with_capacity(ngen);
        // Windowed timing so the cache-warming trend is visible (does per-token
        // time drop as the working set caches?).
        const WIN: usize = 8;
        let mut win_t = std::time::Instant::now();
        let (mut win_hit, mut win_miss) = (self.pin.hits, self.pin.misses);
        for i in 0..ngen {
            let next = self.argmax()?;
            if eos.contains(&next) {
                break;
            }
            generated.push(next);
            self.forward(next, pos)?;
            pos += 1;
            if (i + 1) % WIN == 0 {
                let dt = win_t.elapsed().as_secs_f64();
                let (dh, dm) = (self.pin.hits - win_hit, self.pin.misses - win_miss);
                let hit_pct = 100.0 * dh as f64 / (dh + dm).max(1) as f64;
                tracing::info!(
                    "  tok {}/{ngen}: {:.3} tok/s (window), hit {hit_pct:.1}%",
                    i + 1,
                    WIN as f64 / dt.max(1e-9),
                );
                win_t = std::time::Instant::now();
                (win_hit, win_miss) = (self.pin.hits, self.pin.misses);
            }
        }
        Ok(generated)
    }
}
