//! The GPU decode loop — the resident forward pass. Every per-token op runs
//! on-device against the [`Pin`]'s resident weights, using scratch [`DeviceBuf`]s
//! allocated once and reused each token (no per-token allocation). The only host
//! round-trips are the router-gate logits (MoE layers) and the final logits for
//! argmax — each a small D2H behind a join.
//!
//! Dense attention only (this checkpoint has no DSA indexer), fp8-e4m3 KV latent
//! cache, VQ-int3 routed + shared experts. `rocm`-only.
#![cfg(feature = "rocm")]

use crate::attn::{AttnMode, streaming_rows};
use crate::device::DeviceBuf;
use crate::gpustream::{HipEvent, HipStream, Signal, stream_signal};
use crate::hip::{
    ExpertDesc, device_sync, launch_append_kv, launch_argmax, launch_attend,
    launch_embed_i8_row, launch_gather_rope, launch_gemv_f32, launch_gemv_fp8, launch_gemv_i8,
    launch_index_append, launch_index_head_route, launch_index_pool_push, launch_index_score,
    launch_layernorm, launch_mla_absorb_fp8, launch_mla_value_fp8, launch_moe_expert_range,
    launch_moe_expert_range_i4, launch_moe_reduce, launch_rmsnorm, launch_rope, launch_swiglu,
    launch_vadd,
};
use crate::math::{E4M3_BLOCK, sigmoid, topk_into};
use crate::model::ModelConfig;
use crate::pin::{Fp8Mlp, IndexerPin, LayerMlp, MlpVq, Pin, TRACE_WINDOW};
use crate::telemetry::ProfileSummary;
use anyhow::{Result, bail, ensure};
use futures_util::stream::{StreamExt, TryStreamExt};

/// Little-endian byte view of a POD slice — zero-copy on this LE host (a `[T]`'s
/// in-memory bytes ARE its LE serialization). Feeds the per-token H2D uploads (attn
/// rows/heads u32, expert descriptors, weights f32) with no staging buffer.
fn as_le_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T: Copy` POD (u32/f32/repr(C) ExpertDesc); LE host == LE bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Build one expert's descriptor (six device pointers) from its resolved `MlpVq`.
/// One `ExpertDesc` for both formats — the int4 kernel reinterprets the same bytes.
fn desc_of_vq(m: &MlpVq) -> ExpertDesc {
    ExpertDesc {
        gate_indices: m.gate.indices,
        gate_scales: m.gate.scales,
        up_indices: m.up.indices,
        up_scales: m.up.scales,
        down_indices: m.down.indices,
        down_scales: m.down.scales,
    }
}


/// Host routing: sigmoid the gate logits into `scores`, add the router `bias` into
/// `choice`, and select the top-`top_k` into `sel`. A free fn taking disjoint slices
/// so the caller can borrow `bias` out of `&self.pin` while it mutably borrows its
/// own routing scratch.
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

/// Per-token time buckets. ALWAYS ON: every bucket wraps a join/D2H the forward
/// pass already pays (the end-of-layer sync, the gate-logits read, the stream
/// drain), so accumulating them costs only a clock read per layer — no extra GPU
/// sync, nothing that perturbs the run. The end-of-run [`Profile::report`] is the
/// engine's standing performance summary; the expensive fine-grained audits and
/// correctness probes live behind the `trace` feature instead.
#[derive(Default)]
struct Profile {
    fetch_n: u64,         // demand misses
    route_ns: u128,       // host routing (gate D2H + sigmoid/bias/top-k)
    moe_wall_ns: u128,    // the block_on wall of the overlapped MoE phase (CPU wall)
    compute_gpu_ns: u128, // HIP-event span of the compute stream (partials + reduce)
    wall_ns: u128,
    tokens: u64,
}

impl Profile {
    /// Fold the accumulated buckets into the per-token summary (also fed to OTLP).
    /// `fetch_wall_ns` is the reaper's off-thread load cost; `idle_ns`/`poll_ns` are
    /// the expert stream's tokio-metrics (load-wait / launch) — the accurate
    /// decomposition the async overlap hides from wall-clock.
    fn summary(
        &self,
        hits: u64,
        misses: u64,
        bytes_per_expert: usize,
        fetch_wall_ns: u64,
        idle_ns: u64,
        poll_ns: u64,
    ) -> ProfileSummary {
        let tok = self.tokens.max(1) as f64;
        let per = |ns: u128| ns as f64 / 1e6 / tok; // ms/token
        // Exposed fetch = the MoE wall in excess of the pure compute-stream span
        // (moe_wall − compute_gpu): the fetch that could NOT hide behind compute. The
        // rest of the reaper's fetch_wall overlapped. (tokio idle over-counts — it
        // sums per-expert waits across the ~9 concurrent tasks — so it's reported raw,
        // not used here.)
        let exposed_ns = self.moe_wall_ns.saturating_sub(self.compute_gpu_ns) as f64;
        let hidden = if fetch_wall_ns > 0 {
            (100.0 * (1.0 - exposed_ns / fetch_wall_ns as f64)).clamp(0.0, 100.0)
        } else {
            0.0
        };
        ProfileSummary {
            tok_per_s: self.tokens as f64 / (self.wall_ns as f64 / 1e9).max(1e-9),
            hit_pct: 100.0 * hits as f64 / (hits + misses).max(1) as f64,
            wall_ms: per(self.wall_ns),
            route_ms: per(self.route_ns),
            moe_wall_ms: per(self.moe_wall_ns),
            compute_gpu_ms: per(self.compute_gpu_ns),
            fetch_wall_ms: per(fetch_wall_ns as u128),
            load_wait_ms: per(idle_ns as u128),
            launch_ms: per(poll_ns as u128),
            fetch_hidden_pct: hidden,
            miss_per_tok: self.fetch_n as f64 / tok,
            ms_per_miss: fetch_wall_ns as f64 / 1e6 / self.fetch_n.max(1) as f64,
            gb_per_tok: self.fetch_n as f64 / tok * bytes_per_expert as f64 / 1e9,
        }
    }
}

/// Device-side DSA/MISA indexer state. Mirrors the trained lightning indexer but
/// everything is device-resident: per full layer a bf16 key slab grown in place,
/// plus per-token scratch and the host top-k buffers (the selection's only host
/// round-trip is the score D2H + top-k per full layer). MISA additionally maintains
/// a per-full-layer block-pooled key pool and routes the top-`active_heads` indexer
/// heads via a cheap device estimate before scoring.
struct DeviceIndexer {
    /// Per layer: `Some(slab_index)` for full layers, `None` for shared.
    slab_of: Vec<Option<usize>>,
    /// Per full layer, the bf16 key cache (max_ctx * index_head_dim u16).
    kc: Vec<DeviceBuf>,
    k: DeviceBuf,      // index_head_dim f32 (one key, pre-cache)
    q: DeviceBuf,      // index_n_heads * index_head_dim f32
    w: DeviceBuf,      // index_n_heads f32
    scores: DeviceBuf, // max_ctx f32
    scores_host: Vec<u8>,
    scores_f: Vec<f32>,
    sel: Vec<usize>,
    rows: Vec<u32>,
    /// The most recent full layer's selection this token (IndexShare reuse):
    /// `last_dense` = the whole causal prefix (null rows), else `last_nr` rows.
    last_nr: usize,
    last_dense: bool,
    // --- MISA head routing (empty/unused in dsa mode) ---
    pool: Vec<DeviceBuf>, // per full layer, ⌈max_ctx/MISA_BLOCK⌉ rows of index_head_dim f32
    e: DeviceBuf,         // index_n_heads f32 — router estimates
    e_host: Vec<u8>,
    e_f: Vec<f32>,
    head_sel: Vec<usize>,
    heads_u32: Vec<u32>,
    heads_buf: DeviceBuf, // index_n_heads u32 — active head set for index_score
}

pub struct GpuEngine<'a> {
    pin: Pin<'a>,
    cfg: &'a ModelConfig,
    /// Attention row-selection mode. Dense/Streaming pick rows by position; Dsa/Misa
    /// run the resident indexer per full layer.
    mode: AttnMode,
    /// Device copy of the selected rows — uploaded per token, shared by every layer's
    /// attend. Null-rows (dense) skips it.
    rows_buf: DeviceBuf,
    rows_host: Vec<u32>,
    /// Device-side DSA indexer (dsa/misa modes); `None` for dense/streaming.
    idx: Option<DeviceIndexer>,
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
    /// Trace-only: the ranked top-[`TRACE_WINDOW`] candidates. Stays empty unless
    /// `--trace` is on, and `--trace` is fixed for the run, so this is either filled
    /// every layer or never.
    window: Vec<usize>,
    // Per-token host build scratch — reused every layer so the hot path allocates
    // nothing: resolved VQ descriptors + weights, the resolved batch, D2H staging.
    w: Vec<f32>,
    /// The three per-projection VQ codebooks (gate/up/down), fp16, resident.
    codebooks: [*const u16; 3],
    mlps_vq: Vec<MlpVq>,
    descs_vq: Vec<ExpertDesc>,
    /// Per-expert format for the current layer's batch: `true` = int4 slab (launch the
    /// int4 kernel + reinterprets the descriptor bytes), `false` = int3-VQ.
    /// Filled by [`Pin::submit_layer`] for routed experts; the folded shared expert
    /// appends [`Pin::shared_i4`].
    fmt: Vec<bool>,
    gl_host: Vec<u8>,
    argmax_host: Vec<u8>,
    /// Always-on cheap per-token profiling (see [`Profile`]).
    prof: Profile,
    /// The MoE expert stream's compute stream — resident/loaded experts' partials
    /// run here concurrently with the fetch stream's loads (the overlap). Separate
    /// from the null stream the rest of the forward uses.
    compute_stream: HipStream,
    /// Compute-stream span events (bracket the MoE partials+reduce) + the expert
    /// stream's task monitor (idle = load-wait, poll = launch) — the accurate timing
    /// the async overlap hides from wall-clock.
    moe_ev_start: HipEvent,
    moe_ev_end: HipEvent,
    moe_monitor: tokio_metrics::TaskMonitor,
    /// DIAGNOSTIC (`--checksum-x`, `trace` only): hash the residual stream each layer.
    #[cfg(feature = "trace")]
    checksum_x: bool,
    #[cfg(feature = "trace")]
    ck_buf: Vec<u8>,
}

impl<'a> GpuEngine<'a> {
    pub fn new(pin: Pin<'a>, cfg: &'a ModelConfig, max_ctx: usize, mode: AttnMode) -> Result<Self> {
        // The MoE block folds the shared expert into the routed batch at a single
        // kernel `inter = moe_inter`. Only valid when the shared expert has the routed
        // width, i.e. n_shared == 1 (GLM-5.2).
        ensure!(
            cfg.n_shared == 1,
            "GPU decode assumes n_shared==1 (shared folded into the routed batch); n_shared={}",
            cfg.n_shared
        );
        let f = |n: usize| DeviceBuf::new(n * 4); // f32 buffer of n elems
        // dsa and misa both need the resident indexer; misa additionally routes heads.
        let idx = if matches!(mode, AttnMode::Dsa | AttnMode::Misa { .. }) {
            let misa = matches!(mode, AttnMode::Misa { .. });
            let full = cfg.indexer_layout()?;
            let hd = cfg.index_head_dim;
            let n_blocks = max_ctx.div_ceil(crate::indexer::MISA_BLOCK);
            let mut slab_of = vec![None; cfg.n_layers];
            let mut kc = Vec::new();
            let mut pool = Vec::new();
            for (l, &is_full) in full.iter().enumerate() {
                if is_full {
                    slab_of[l] = Some(kc.len());
                    kc.push(DeviceBuf::new(max_ctx * hd * 2)?); // bf16 key cache
                    if misa {
                        pool.push(DeviceBuf::new(n_blocks * hd * 4)?);
                    }
                }
            }
            Some(DeviceIndexer {
                slab_of,
                kc,
                k: DeviceBuf::new(hd * 4)?,
                q: DeviceBuf::new(cfg.index_n_heads * hd * 4)?,
                w: DeviceBuf::new(cfg.index_n_heads * 4)?,
                scores: DeviceBuf::new(max_ctx * 4)?,
                scores_host: Vec::new(),
                scores_f: Vec::new(),
                sel: Vec::new(),
                rows: Vec::new(),
                last_nr: 0,
                last_dense: true,
                pool,
                e: DeviceBuf::new(cfg.index_n_heads * 4)?,
                e_host: Vec::new(),
                e_f: Vec::new(),
                head_sel: Vec::new(),
                heads_u32: Vec::new(),
                heads_buf: DeviceBuf::new(cfg.index_n_heads * 4)?,
            })
        } else {
            None
        };
        let kvl = cfg.kv_lora_rank;
        let rope = cfg.qk_rope_head_dim;
        let h = cfg.n_heads;
        let slots = cfg.experts_per_layer(); // routed + shared per MoE launch
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
            mode,
            rows_buf: DeviceBuf::new(max_ctx * 4)?,
            rows_host: Vec::new(),
            idx,
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
            mlp_g: f(cfg.dense_inter)?,
            mlp_u: f(cfg.dense_inter)?,
            moe_out: f(cfg.hidden)?,
            moe_partial: f(slots * cfg.hidden)?,
            moe_h: f(slots * cfg.moe_inter)?,
            descs_buf: DeviceBuf::new(slots * std::mem::size_of::<ExpertDesc>())?,
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
            window: Vec::new(), // grown once by the first traced layer; empty otherwise
            w: Vec::with_capacity(slots),
            codebooks: pin.codebooks(),
            mlps_vq: Vec::with_capacity(cfg.top_k),
            descs_vq: Vec::with_capacity(slots),
            fmt: Vec::with_capacity(slots),
            gl_host: Vec::with_capacity(cfg.n_experts * 4),
            argmax_host: Vec::with_capacity(8),
            prof: Profile::default(),
            compute_stream: HipStream::new()?,
            moe_ev_start: HipEvent::new()?,
            moe_ev_end: HipEvent::new()?,
            moe_monitor: tokio_metrics::TaskMonitor::new(),
            #[cfg(feature = "trace")]
            checksum_x: false,
            #[cfg(feature = "trace")]
            ck_buf: Vec::new(),
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

    /// DIAGNOSTIC: hash the residual stream after every layer (`--checksum-x`).
    #[cfg(feature = "trace")]
    pub fn set_checksum_x(&mut self, on: bool) {
        self.checksum_x = on;
    }

    /// DSA/MISA row selection for one full/shared layer at `pos`, returning the attend
    /// row set `(rows_ptr, nr)` — a null pointer means dense over `0..nr`. `xnp` is the
    /// layer input (post input_layernorm), `qrp` the q-LoRA residual (both device
    /// pointers, valid until the next sync). Full layers append this token's indexer
    /// key, then score + host top-k once the cache exceeds index_topk (below that it is
    /// exactly dense); shared layers reuse the nearest preceding full layer's selection
    /// (IndexShare). MISA additionally routes the top-`active_heads` indexer heads via a
    /// block-pool estimate and scores only those.
    fn dsa_select_layer(
        &mut self,
        l: usize,
        pos: usize,
        xnp: *const f32,
        qrp: *const f32,
        ipin: Option<IndexerPin>,
    ) -> Result<(*const u32, usize)> {
        use crate::indexer::K_NORM_EPS;
        let cfg = self.cfg;
        let hd = cfg.index_head_dim;
        let nh = cfg.index_n_heads;
        let rope = cfg.qk_rope_head_dim;
        let theta = cfg.rope_theta();
        let topk = cfg.index_topk;
        let nt = pos + 1;
        // MISA routes a head subset; DSA scores all heads. Read the mode before
        // borrowing `self.idx` (Copy — no move of self.mode).
        let active_heads = match self.mode {
            AttnMode::Misa { active_heads } => Some(active_heads),
            _ => None,
        };
        // Disjoint field borrows: idx (mut) and rows_buf (mut) are distinct fields.
        let idx = self
            .idx
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("dsa_select_layer without a device indexer"))?;

        let slab = match idx.slab_of[l] {
            Some(s) => s,
            // Shared layer: reuse the last full layer's selection verbatim.
            None => {
                return Ok(if idx.last_dense {
                    (std::ptr::null(), idx.last_nr)
                } else {
                    (self.rows_buf.ptr() as *const u32, idx.last_nr)
                });
            }
        };
        let ip = ipin.ok_or_else(|| anyhow::anyhow!("full layer {l} missing resident indexer"))?;
        let kcp = idx.kc[slab].ptr_mut() as *mut u16;
        let kp = idx.k.ptr_mut() as *mut f32;
        let iqp = idx.q.ptr_mut() as *mut f32;
        let iwp = idx.w.ptr_mut() as *mut f32;
        let scp = idx.scores.ptr_mut() as *mut f32;
        let poolp = if active_heads.is_some() {
            idx.pool[slab].ptr_mut() as *mut f32
        } else {
            std::ptr::null_mut()
        };

        // Key: wk·xn → LayerNorm(k_norm) → RoPE(first `rope` dims) → append. Runs EVERY
        // token so the cache is ready when we cross the threshold. MISA folds the same
        // roped key into the block pool on every token, for the same reason.
        // SAFETY: indexer weights resident; scratch/kc/pool are live device bufs;
        // ordering is null-stream program order; a sync precedes any D2H.
        unsafe {
            launch_gemv_fp8(
                xnp,
                ip.wk.packed,
                ip.wk.scale,
                ip.wk.o_dim,
                ip.wk.i_dim,
                ip.wk.block,
                kp,
            )?;
            launch_layernorm(kp, ip.k_norm_w, ip.k_norm_b, hd, K_NORM_EPS, kp)?;
            launch_rope(kp, 1, rope, rope, pos, theta)?;
            launch_index_append(kp, kcp, pos, hd)?;
            if active_heads.is_some() {
                launch_index_pool_push(kp as *const f32, poolp, pos, hd)?;
            }
        }
        if nt <= topk {
            idx.last_dense = true;
            idx.last_nr = nt;
            return Ok((std::ptr::null(), nt));
        }

        // Query heads (wq_b·qr, roped per head) + gates (weights_proj·xn), then score
        // every cached token and pick the top-k host-side.
        let wscale = 1.0 / (nh as f32).sqrt();
        let dscale = 1.0 / (hd as f32).sqrt();
        // SAFETY: as above; iqp/iwp are live scratch sized nh·hd / nh.
        unsafe {
            launch_gemv_fp8(
                qrp,
                ip.wq_b.packed,
                ip.wq_b.scale,
                ip.wq_b.o_dim,
                ip.wq_b.i_dim,
                ip.wq_b.block,
                iqp,
            )?;
            launch_rope(iqp, nh, hd, rope, pos, theta)?; // per head: stride hd, seg rope
            // weights_proj is bf16→f32 [n_heads, hidden] — plain f32 GEMV.
            launch_gemv_f32(xnp, ip.weights_proj, nh, cfg.hidden, iwp)?;
        }

        // Active head set for the O(nt) scan: all `nh` heads (DSA), or the MISA-routed
        // top-h (a device estimate + tiny nh-float D2H). `h >= nh` degenerates to "all
        // heads", so guard on h < nh.
        let (heads_ptr, nact): (*const u32, usize) = match active_heads {
            Some(hh) if hh < nh => {
                let m_blocks = nt.div_ceil(crate::indexer::MISA_BLOCK);
                let ppool = idx.pool[slab].ptr() as *const f32;
                let ep = idx.e.ptr_mut() as *mut f32;
                // SAFETY: iqp/iwp/ppool/ep are live device scratch; a sync precedes the D2H.
                unsafe {
                    launch_index_head_route(iqp, iwp, ppool, m_blocks, nh, hd, ep)?;
                }
                device_sync()?;
                idx.e.copy_out_prefix(&mut idx.e_host, nh * 4)?;
                idx.e_f.clear();
                idx.e_f.extend(
                    idx.e_host
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                );
                topk_into(&idx.e_f, hh, &mut idx.head_sel);
                idx.heads_u32.clear();
                idx.heads_u32.extend(idx.head_sel.iter().map(|&i| i as u32));
                idx.heads_buf.copy_in_at(0, as_le_bytes(&idx.heads_u32))?;
                (idx.heads_buf.ptr() as *const u32, idx.heads_u32.len())
            }
            _ => (std::ptr::null(), nh),
        };

        // SAFETY: iqp/iwp/kcp/scp are live scratch; heads_ptr is null (DSA) or the
        // just-uploaded `nact`-entry head buffer (MISA).
        unsafe {
            launch_index_score(
                iqp,
                iwp,
                kcp as *const u16,
                heads_ptr,
                nt,
                nh,
                nact,
                hd,
                wscale,
                dscale,
                scp,
            )?;
        }
        device_sync()?;
        idx.scores.copy_out_prefix(&mut idx.scores_host, nt * 4)?;
        idx.scores_f.clear();
        idx.scores_f.extend(
            idx.scores_host
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        topk_into(&idx.scores_f, topk, &mut idx.sel);
        idx.sel.sort_unstable(); // ascending token order for the gather
        idx.rows.clear();
        idx.rows.extend(idx.sel.iter().map(|&i| i as u32));
        self.rows_buf.copy_in_at(0, as_le_bytes(&idx.rows))?;
        idx.last_dense = false;
        idx.last_nr = idx.rows.len();
        Ok((self.rows_buf.ptr() as *const u32, idx.rows.len()))
    }

    /// One forward pass for `token` at `pos`, leaving next-token logits device-side
    /// in `self.logits`.
    async fn forward(&mut self, token: u32, pos: usize) -> Result<()> {
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

        // Row selection: dense/streaming is layer-blind (computed once, reused by
        // every layer's attend — dense passes a null rows pointer, the kernel fast
        // path). Dsa/misa selects per full layer inside the loop (it needs the
        // mid-attention q-LoRA residual), signalled by `None` here.
        let hoisted_rows: Option<(*const u32, usize)> = match &self.mode {
            AttnMode::Dense => Some((std::ptr::null(), pos + 1)),
            AttnMode::Streaming { sinks, window } => {
                streaming_rows(pos + 1, *sinks, *window, &mut self.rows_host);
                if self.rows_host.len() == pos + 1 {
                    Some((std::ptr::null(), pos + 1)) // all selected → dense fast path
                } else {
                    self.rows_buf.copy_in_at(0, as_le_bytes(&self.rows_host))?;
                    Some((self.rows_buf.ptr() as *const u32, self.rows_host.len()))
                }
            }
            AttnMode::Dsa | AttnMode::Misa { .. } => None,
        };

        // Embedding row → x.
        // SAFETY: all pointers are device-resident scratch/weights valid for their
        // dims; each launch's inputs are produced by a prior launch on the same
        // (default) stream, so ordering holds; a device_sync precedes every host read.
        unsafe {
            launch_embed_i8_row(
                self.pin.embed.packed,
                self.pin.embed.scale,
                token as usize,
                hidden,
                xp,
            )?;
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
            let indexer_pin = lw.indexer; // ends the &pin.layers borrow (Copy)

            let lc8p = self.lc[l].ptr_mut();
            let lscalep = self.lc_scale[l].ptr_mut() as *mut f32;
            let rcp = self.rc[l].ptr_mut() as *mut u16;

            // --- Attention phase 1: projections, ropes, cache append, absorb. ---
            // SAFETY: see the forward-level note; every pointer is live scratch.
            unsafe {
                launch_rmsnorm(xp, input_ln, hidden, eps, xnp)?;
                launch_gemv_fp8(
                    xnp, q_a.packed, q_a.scale, q_a.o_dim, q_a.i_dim, q_a.block, qrp,
                )?;
                launch_rmsnorm(qrp, q_a_ln, cfg.q_lora_rank, eps, qrp)?; // in-place
                launch_gemv_fp8(
                    qrp, q_b.packed, q_b.scale, q_b.o_dim, q_b.i_dim, q_b.block, qp,
                )?;
                launch_gemv_fp8(
                    xnp,
                    kv_a.packed,
                    kv_a.scale,
                    kv_a.o_dim,
                    kv_a.i_dim,
                    kv_a.block,
                    compp,
                )?;
                launch_rmsnorm(compp, kv_a_ln, kvl, eps, compp)?; // normalize latent (first kvl)
                launch_rope(compp.add(kvl), 1, rope, rope, pos, theta)?; // rope the key
                launch_rope(qp.add(nope), h, qh, rope, pos, theta)?; // rope per-head query
                launch_append_kv(
                    compp,
                    compp.add(kvl),
                    lc8p,
                    lscalep,
                    rcp,
                    pos,
                    kvl,
                    rope,
                    nb,
                )?;
                launch_mla_absorb_fp8(
                    qp,
                    kv_b.packed,
                    kv_b.scale,
                    h,
                    qh,
                    nope,
                    vh,
                    kvl,
                    kv_b.block,
                    qabsp,
                )?;
                launch_gather_rope(qp, qropep, h, qh, nope, rope)?;
            }

            // Row selection: hoisted (dense/streaming) or per-layer DSA (needs `qrp`
            // the q-LoRA residual + `xnp` the layer input, both from phase 1). DSA
            // syncs mid-layer for the score D2H + host top-k.
            let (rows_ptr, nr) = match hoisted_rows {
                Some(rn) => rn,
                None => self.dsa_select_layer(l, pos, xnp, qrp, indexer_pin)?,
            };

            // --- Attention phase 2: dense flash attend, value + output projection,
            //     residual, pre-MLP norm. ---
            // SAFETY: see the forward-level note; every pointer is live scratch.
            unsafe {
                launch_attend(
                    qabsp, qropep, lc8p, lscalep, rcp, rows_ptr, h, nr, kvl, rope, nb, scale,
                    clatp, apartp,
                )?;
                launch_mla_value_fp8(
                    clatp,
                    kv_b.packed,
                    kv_b.scale,
                    h,
                    nope,
                    vh,
                    kvl,
                    kv_b.block,
                    ctxp,
                )?;
                launch_gemv_fp8(
                    ctxp,
                    o_proj.packed,
                    o_proj.scale,
                    o_proj.o_dim,
                    o_proj.i_dim,
                    o_proj.block,
                    subp,
                )?;
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
                    launch_gemv_fp8(
                        xnp,
                        m.gate.packed,
                        m.gate.scale,
                        m.gate.o_dim,
                        m.gate.i_dim,
                        m.gate.block,
                        gp,
                    )?;
                    launch_gemv_fp8(
                        xnp,
                        m.up.packed,
                        m.up.scale,
                        m.up.o_dim,
                        m.up.i_dim,
                        m.up.block,
                        up,
                    )?;
                    launch_swiglu(gp, up, inter, gp)?; // in place: h = silu(gate)*up
                    launch_gemv_fp8(
                        gp,
                        m.down.packed,
                        m.down.scale,
                        m.down.o_dim,
                        m.down.i_dim,
                        m.down.block,
                        outp,
                    )?;
                }
            } else {
                // Router gate on device, then read logits to route on host.
                // SAFETY: gate_w resident F32; glp device scratch.
                unsafe { launch_gemv_f32(xnp, gate_w, cfg.n_experts, hidden, glp)? };
                // The gate-logits D2H is a blocking join, so timing around it is free —
                // no sync we don't already pay. (All the always-on profile buckets wrap
                // existing join/D2H points; none add a sync.)
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
                self.prof.route_ns += t.elapsed().as_nanos();
                // Trace v2 only: re-rank the same `choice` array to the wider candidate
                // window the offline (J, M) grid needs. Deliberately outside the
                // `route_ns` clock and behind the trace gate, so the decode path is
                // byte-for-byte the work it was before and route_ns stays comparable
                // across the change.
                if self.pin.tracing() {
                    topk_into(&self.choice, TRACE_WINDOW, &mut self.window);
                }
                // SUBMIT this layer's cold reads — each selected expert gets a load
                // Signal (hit → ready; miss → resolves when its bytes land). The slot
                // ADDRESSES are known now, so the descriptors below are valid pointers.
                let miss0 = self.pin.misses;
                let mut signals = self.pin.submit_layer(
                    l,
                    &self.sel,
                    &self.window,
                    &self.choice,
                    &mut self.mlps_vq,
                    &mut self.fmt,
                )?;
                self.prof.fetch_n += self.pin.misses - miss0;
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
                // VQ routed descriptors + the folded VQ shared expert (resident, so its
                // load is `ready()`).
                self.descs_vq.clear();
                for m in &self.mlps_vq {
                    self.descs_vq.push(desc_of_vq(m));
                }
                if let Some(s) = shared {
                    self.descs_vq.push(desc_of_vq(&s));
                    self.w.push(1.0);
                    self.fmt.push(self.pin.shared_i4());
                    signals.push(Signal::ready());
                }
                let ndesc = self.descs_vq.len();
                self.descs_buf
                    .copy_in_at(0, as_le_bytes(&self.descs_vq))?;
                self.wexpert_buf.copy_in_at(0, as_le_bytes(&self.w))?;
                // THE EXPERT STREAM (stream 1): each expert, once its load Signal
                // resolves, launches its own partial on the compute stream. Concurrent
                // loads (buffer_unordered) overlap the misses' fetch with the resident/
                // loaded experts' compute; partials are independent rows so completion
                // order is irrelevant. Then a fixed-order reduce on the same stream,
                // awaited so moe_out is ready for the residual add. mlp bucket = the
                // whole overlapped wall (fetch now hidden inside it).
                let x_c = xnp as *const f32;
                let (h_c, part_c, out_c) = (
                    self.moe_h.ptr_mut() as *mut f32,
                    self.moe_partial.ptr_mut() as *mut f32,
                    self.moe_out.ptr_mut() as *mut f32,
                );
                // One descriptor buffer for both kernels — the int4 kernel reinterprets
                // the same six-pointer bytes (at its slot offsets).
                let descs_ptr = self.descs_buf.ptr() as *const ExpertDesc;
                // Per-expert format (routed experts from their slab; shared appended
                // above). Cloned so the expert stream owns it (the small bool vec moves
                // into the async closure). Hybrid mixes int4/vq3 within one batch.
                let fmt = self.fmt.clone();
                let w_ptr = self.wexpert_buf.ptr() as *const f32;
                let (cb0, cb1, cb2) = (self.codebooks[0], self.codebooks[1], self.codebooks[2]);
                let cs_raw = self.compute_stream.raw();
                let inter = cfg.moe_inter;
                let monitor = self.moe_monitor.clone();
                // Bracket the compute-stream span (partials+reduce) for the accurate
                // GPU-side timing; read at the end-of-layer join. Caveat: each partial
                // launches only after its per-expert `sig.await` resolves on the host,
                // so the compute stream sits idle between host-gated launches and those
                // bubbles fall inside this span — `compute_gpu` is thus an UPPER bound,
                // making the derived `fetch_hidden_pct` (1 − (moe_wall−compute_gpu)/…)
                // read slightly optimistic. Fine as a gauge; not a hard hidden-fetch %.
                self.moe_ev_start.record(cs_raw)?;
                let tm = std::time::Instant::now();
                // The expert stream runs inline in this (async) forward — awaited by
                // the single decode-loop runtime, so no per-layer block_on.
                // Width = the whole batch (ndesc = experts_per_layer), so every expert's
                // load is in flight at once — the misses fetch while the resident/loaded
                // experts compute. try_for_each_concurrent drives all to completion and
                // short-circuits on the first Err (no collected Vec).
                futures_util::stream::iter(0..ndesc)
                    .map(Ok::<usize, anyhow::Error>)
                    .try_for_each_concurrent(ndesc, move |e| {
                        let sig = signals[e].clone();
                        let i4 = fmt[e];
                        // Instrument: idle = time parked on the load Signal (the
                        // fetch-wait the stream sees), poll = the launch cost.
                        monitor.instrument(async move {
                            sig.await;
                            // SAFETY: descs/codebooks resident; slot loaded (sig
                            // resolved); h/part device scratch; cs_raw live.
                            unsafe {
                                if i4 {
                                    launch_moe_expert_range_i4(
                                        x_c, hidden, inter, e, 1, descs_ptr, w_ptr, h_c, part_c,
                                        cs_raw,
                                    )
                                } else {
                                    launch_moe_expert_range(
                                        x_c, hidden, inter, e, 1, descs_ptr, cb0, cb1, cb2, w_ptr,
                                        h_c, part_c, cs_raw,
                                    )
                                }
                            }
                        })
                    })
                    .await?;
                // SAFETY: partial holds ndesc·hidden f32; out is hidden f32; cs live.
                unsafe { launch_moe_reduce(part_c, ndesc, hidden, out_c, cs_raw)? };
                stream_signal(cs_raw)?.await;
                self.prof.moe_wall_ns += tm.elapsed().as_nanos();
                self.moe_ev_end.record(cs_raw)?;
            }
            // SAFETY: residual add of the MLP contribution.
            unsafe { launch_vadd(xp, self.moe_out.ptr() as *const f32, hidden)? };
            // End-of-layer join: protects the reused descs/wexpert/moe_out buffers
            // before the next layer overwrites them, and surfaces faults. Folds into
            // the MoE wall (near-0 for MoE layers — the compute stream already synced;
            // the dense MLP compute for the 3 dense layers).
            let t = std::time::Instant::now();
            device_sync()?;
            self.prof.moe_wall_ns += t.elapsed().as_nanos();
            // Both MoE span events retired by the sync — read the compute-stream span
            // (MoE layers only; dense layers never recorded them).
            if dense_mlp.is_none() {
                let ms = HipEvent::elapsed_ms(&self.moe_ev_start, &self.moe_ev_end)?;
                self.prof.compute_gpu_ns += (ms as f64 * 1e6) as u128;
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
            launch_gemv_i8(
                xnp,
                head.packed,
                head.scale,
                head.o_dim,
                head.i_dim,
                self.logits.ptr_mut() as *mut f32,
            )?;
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
    /// `eos`. Returns the generated ids + the always-on decode-loop [`ProfileSummary`]
    /// (also logged as the PROFILE line; `main` feeds it to the OTLP span).
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        ngen: usize,
        eos: &[u32],
    ) -> Result<(Vec<u32>, ProfileSummary)> {
        ensure!(!prompt_ids.is_empty(), "empty prompt");
        // The decode as ONE async flow: prefill (warm-up) then the token loop, driven
        // by a single current-thread runtime — `forward` awaits the expert stream
        // inline, so there's no per-layer block_on. The token loop is serial by data
        // dependency (T+1 needs T's argmax); this is the shape MTP/speculative decode
        // slots into. `rt` is local (not on `self`) so the future can borrow `&mut self`.
        #[cfg(feature = "trace")]
        const WIN: usize = 8;
        let rt = tokio::runtime::Builder::new_current_thread().build()?;
        let mut generated = Vec::with_capacity(ngen);
        let (hit0, miss0, decode_wall) = rt.block_on(async {
            let mut pos = 0usize;
            for &tok in prompt_ids {
                // Beat the watchdog per prefill token too — a long/cold prompt can
                // exceed the deadline mid-prefill while making normal progress, and only
                // the decode loop beat before, so it would kill a healthy process.
                if let Some(hb) = &self.heartbeat {
                    hb.beat();
                }
                self.forward(tok, pos).await?;
                pos += 1;
            }
            // Profile the DECODE loop only (prefill is warm-up); reset the pin counters
            // too so hit%/misses describe steady-state decode, not the cold prefill.
            self.prof = Profile::default();
            let hit0 = self.pin.hits;
            let miss0 = self.pin.misses;
            let decode_wall = std::time::Instant::now();
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
                self.forward(next, pos).await?;
                pos += 1;
                // Bound trace loss to one token: the watchdog exits without destructors,
                // so BufWriter's Drop is not a guarantee. No-op when not tracing.
                self.pin.flush_trace()?;
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
            Ok::<_, anyhow::Error>((hit0, miss0, decode_wall))
        })?;
        self.prof.wall_ns = decode_wall.elapsed().as_nanos();
        self.prof.tokens = generated.len() as u64;
        let bytes_per_expert = crate::quant::vq_expert_bytes(self.cfg.hidden, self.cfg.moe_inter);
        // The accurate async-side decomposition: reaper fetch wall + the expert
        // stream's tokio-metrics (idle = load-wait, poll = launch).
        let tm = self.moe_monitor.cumulative();
        let summary = self.prof.summary(
            self.pin.hits - hit0,
            self.pin.misses - miss0,
            bytes_per_expert,
            self.pin.fetch_ns(),
            tm.total_idle_duration.as_nanos() as u64,
            tm.total_poll_duration.as_nanos() as u64,
        );
        summary.report();
        Ok((generated, summary))
    }
}
