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

use crate::attn::{AttnMode, streaming_rows};
use crate::device::DeviceBuf;
use crate::hip::{
    ExpertDesc, device_sync, launch_append_kv, launch_append_kv_fp8, launch_argmax, launch_attend,
    launch_attend_fp8, launch_embed_i8_row, launch_gather_rope, launch_gemv_bf16, launch_gemv_f32,
    launch_gemv_i4, launch_gemv_i8, launch_index_append, launch_index_head_route,
    launch_index_pool_push, launch_index_score, launch_layernorm, launch_mla_absorb,
    launch_mla_value, launch_moe, launch_rmsnorm, launch_rope, launch_vadd,
};
use crate::math::{sigmoid, topk_into};
use crate::model::ModelConfig;
use crate::pin::{IndexerPin, LayerMlp, Mlp, Pin};
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

fn desc_bytes(d: &[ExpertDesc]) -> &[u8] {
    // SAFETY: ExpertDesc is repr(C) POD (six pointers); this is its byte view.
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, std::mem::size_of_val(d)) }
}

/// Little-endian byte view of an f32 slice — zero-copy, since on this LE host
/// `[f32]`'s in-memory representation IS its little-endian serialization (the
/// same transmute idiom `desc_bytes` relies on). Feeds the per-token weight H2D
/// with no staging buffer.
fn f32_le_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 is POD; the bytes are the LE serialization on this LE host.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Little-endian byte view of a u32 slice (same idiom as [`f32_le_bytes`]).
/// Feeds the per-token attention-rows H2D.
fn u32_le_bytes(v: &[u32]) -> &[u8] {
    // SAFETY: u32 is POD; the bytes are the LE serialization on this LE host.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Host routing: sigmoid the gate logits into `scores`, add the router `bias` into
/// `choice`, and select the top-`top_k` into `sel` (mirrors moe.rs exactly). A free
/// fn taking disjoint slices — so the caller can borrow `bias` out of `&self.pin`
/// while it mutably borrows its own routing scratch, no per-token bias clone. Used
/// for BOTH the current layer's routing and the cross-layer L+1 prediction (each
/// with its own scratch triple); only the selected indices matter, so no
/// normalization is done here (the caller weights the current layer's picks).
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

/// Per-token time buckets, measured (not theorized). The mid-layer sync drains
/// the attention kernels; the end-of-layer sync drains the MLP; the cold-expert
/// copy_in's between them are pure H2D (no kernel-wait). So these split cleanly
/// into I/O (fetch) vs GPU compute (attn/mlp/lmhead) vs host routing.
#[derive(Default)]
struct Profile {
    fetch_ns: u128,  // io_uring O_DIRECT cold stream (NVMe->VMM)
    fetch_n: u64,    // miss count
    attn_ns: u128,   // mid-layer sync — attention+gate GPU compute
    mlp_ns: u128,    // end-of-layer sync — MLP GPU compute (+ dense-layer attn)
    lmhead_ns: u128, // final sync — norm + lm_head
    route_ns: u128,  // gate-logits D2H + host sigmoid/bias/topk
    wall_ns: u128,   // total decode wall time
    tokens: u64,
}

impl Profile {
    fn report(&self) {
        let tok = self.tokens.max(1) as f64;
        let per = |ns: u128| ns as f64 / 1e6 / tok; // ms/token
        let pct = |ns: u128| 100.0 * ns as f64 / self.wall_ns.max(1) as f64;
        let accounted = self.fetch_ns + self.attn_ns + self.mlp_ns + self.lmhead_ns + self.route_ns;
        tracing::info!(
            "PROFILE/tok: {:.0}ms wall | fetch {:.0}ms {:.0}% ({} miss, {:.2}ms/miss) | attn {:.0}ms {:.0}% | mlp {:.0}ms {:.0}% | lmhead {:.0}ms {:.0}% | route {:.0}ms | other {:.0}ms",
            per(self.wall_ns),
            per(self.fetch_ns),
            pct(self.fetch_ns),
            self.fetch_n / self.tokens.max(1),
            self.fetch_ns as f64 / 1e6 / self.fetch_n.max(1) as f64,
            per(self.attn_ns),
            pct(self.attn_ns),
            per(self.mlp_ns),
            pct(self.mlp_ns),
            per(self.lmhead_ns),
            pct(self.lmhead_ns),
            per(self.route_ns),
            per(self.wall_ns.saturating_sub(accounted)),
        );
    }
}

/// Device-side DSA/MISA indexer state (dsa or misa mode). Mirrors the scalar
/// `Indexer` but everything is device-resident: per full layer a bf16 key slab
/// grown in place, plus per-token scratch and the host top-k buffers (the
/// selection's only host round-trip is the score D2H + top-k per full layer).
/// MISA additionally maintains a per-full-layer block-pooled key pool and routes
/// the top-`active_heads` indexer heads via a cheap device estimate (`e`), whose
/// nh-float D2H picks the head set host-side (`head_sel`/`heads_u32`/`heads_buf`).
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
    /// `last_dense` = the whole causal prefix (null rows), else `last_nr` rows
    /// live in `rows_buf`.
    last_nr: usize,
    last_dense: bool,
    // --- MISA head routing (empty/unused in dsa mode) ---
    /// Per full layer, the block-pooled running-mean keys (⌈max_ctx/MISA_BLOCK⌉
    /// rows of index_head_dim f32). Indexed by slab like `kc`. Empty for dsa.
    pool: Vec<DeviceBuf>,
    e: DeviceBuf,         // index_n_heads f32 — router estimates E_j
    e_host: Vec<u8>,      // E_j D2H staging (nh f32)
    e_f: Vec<f32>,        // E_j widened for host top-k
    head_sel: Vec<usize>, // routed head indices (topk_into output)
    heads_u32: Vec<u32>,  // head indices uploaded to `heads_buf`
    heads_buf: DeviceBuf, // index_n_heads u32 — active head set for index_score
}

pub struct GpuEngine<'a> {
    pin: Pin<'a>,
    cfg: &'a ModelConfig,
    /// Attention row-selection mode. Dense/Streaming/Dsa/Misa all run on device
    /// (Misa adds the block-pool head router over the resident DSA indexer).
    mode: AttnMode,
    /// Device copy of the selected rows — uploaded per token (streaming: once,
    /// layer-blind; dsa: per full layer). Shared by every layer's attend.
    rows_buf: DeviceBuf,
    rows_host: Vec<u32>,
    /// KV-slab + rows_buf capacity in tokens; forward() refuses pos beyond it
    /// (the append/copy would otherwise write past the device buffers).
    max_ctx: usize,
    /// Device-side DSA indexer (dsa mode). Per full layer: a bf16 key slab
    /// grown in place; plus reused per-token scratch and the host top-k
    /// buffers. Empty for dense/streaming/misa.
    idx: Option<DeviceIndexer>,
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
    // Cross-layer prefetch (`--prefetch`) scratch: L+1's router-gate prediction.
    // `pred_xn` = L's post-attn residual normed with L+1's input_ln; `pred_gl` = the
    // L+1 gate logits over it. Small, allocated unconditionally (cheap), used only
    // when `prefetch`.
    pred_xn: DeviceBuf,
    pred_gl: DeviceBuf,
    moe_out: DeviceBuf,
    moe_partial: DeviceBuf, // [slots*hidden] per-expert outputs (deterministic reduce)
    moe_h: DeviceBuf,       // [E*inter] SwiGLU hidden scratch (two-pass coalesced MoE)
    descs_buf: DeviceBuf,
    wexpert_buf: DeviceBuf,
    logits: DeviceBuf,
    // Device argmax result: 8 bytes [i32 index | f32 max-value]. The reduction
    // kernel writes it and only these 8 bytes come back per token (vs the full
    // vocab×f32 logits), preserving the host argmax's tie-break + finiteness bail.
    argmax_dev: DeviceBuf,
    // Per-layer latent KV slabs, grown in place to max_ctx. `lc` is bf16
    // (max_ctx*kvl u16) by default, or fp8-e4m3 (max_ctx*kvl u8) when `kv_fp8`,
    // in which case `lc_scale` holds the per-128 block scales (max_ctx*n_blocks
    // f32). `rc` (roped key) is always bf16.
    lc: Vec<DeviceBuf>,
    rc: Vec<DeviceBuf>,
    lc_scale: Vec<DeviceBuf>, // empty unless kv_fp8
    kv_fp8: bool,
    n_kv_blocks: usize, // kvl / E4M3_BLOCK (fp8 scales per token)
    // Host routing/argmax scratch.
    scores: Vec<f32>,
    choice: Vec<f32>,
    sel: Vec<usize>,
    // Per-token host build scratch — reused (cleared+refilled) every layer so the
    // forward hot path allocates nothing: resolved expert descriptors + weights, the
    // resolved `Mlp` batch, and the D2H staging buffers for the gate/prediction reads
    // (the weight H2D uploads a zero-copy LE view of `w`; the argmax D2H is 8 bytes).
    descs: Vec<ExpertDesc>,
    w: Vec<f32>,
    mlps: Vec<Mlp>,
    gl_host: Vec<u8>,
    pgl_host: Vec<u8>,
    argmax_host: Vec<u8>,
    // Cross-layer prefetch: separate host scratch for the L+1 prediction top-k, so
    // it never clobbers `scores`/`choice`/`sel` (still needed for L's own MoE
    // weights after the prediction runs).
    pred_scores: Vec<f32>,
    pred_choice: Vec<f32>,
    pred_sel: Vec<usize>,
    prefetch: bool,
    prefetch_depth: usize,
    prof: Profile,
}

impl<'a> GpuEngine<'a> {
    pub fn new(
        pin: Pin<'a>,
        cfg: &'a ModelConfig,
        max_ctx: usize,
        mode: AttnMode,
        kv_fp8: bool,
    ) -> Result<Self> {
        ensure!(
            matches!(
                mode,
                AttnMode::Dense
                    | AttnMode::Streaming { .. }
                    | AttnMode::Dsa
                    | AttnMode::Misa { .. }
            ),
            "GPU engine does not implement {mode:?} yet; dense, streaming, dsa, and misa only"
        );
        // dsa and misa both need the resident indexer; misa additionally routes
        // heads via the block pool (active_heads = Some(h)).
        let active_heads = match mode {
            AttnMode::Dsa => Some(None),
            AttnMode::Misa { active_heads } => Some(Some(active_heads)),
            _ => None,
        };
        let idx = if let Some(active_heads) = active_heads {
            let misa = active_heads.is_some();
            let full = cfg.indexer_layout()?;
            let hd = cfg.index_head_dim;
            let n_blocks = max_ctx.div_ceil(crate::indexer::MISA_BLOCK);
            let mut slab_of = vec![None; cfg.n_layers];
            let mut kc = Vec::new();
            let mut pool = Vec::new();
            for (l, &is_full) in full.iter().enumerate() {
                if is_full {
                    slab_of[l] = Some(kc.len());
                    kc.push(DeviceBuf::new(max_ctx * hd * 2)?);
                    // Pool is misa-only; dsa leaves it empty (never indexed).
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
        ensure!(
            !kv_fp8 || kvl.is_multiple_of(crate::math::E4M3_BLOCK),
            "--kv-fp8 needs kv_lora_rank ({kvl}) a multiple of {} (fp8 block size)",
            crate::math::E4M3_BLOCK
        );
        let n_kv_blocks = kvl / crate::math::E4M3_BLOCK;
        let mut lc = Vec::with_capacity(cfg.n_layers);
        let mut rc = Vec::with_capacity(cfg.n_layers);
        let mut lc_scale = Vec::with_capacity(if kv_fp8 { cfg.n_layers } else { 0 });
        for _ in 0..cfg.n_layers {
            // fp8: kvl u8 latent + n_kv_blocks f32 scales; bf16: kvl u16 latent.
            lc.push(DeviceBuf::new(max_ctx * kvl * if kv_fp8 { 1 } else { 2 })?);
            rc.push(DeviceBuf::new(max_ctx * rope * 2)?);
            if kv_fp8 {
                lc_scale.push(DeviceBuf::new(max_ctx * n_kv_blocks * 4)?);
            }
        }
        Ok(Self {
            cfg,
            mode,
            rows_buf: DeviceBuf::new(max_ctx * 4)?,
            rows_host: Vec::new(),
            max_ctx,
            idx,
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
            pred_xn: f(cfg.hidden)?,
            pred_gl: f(cfg.n_experts)?,
            moe_out: f(cfg.hidden)?,
            moe_partial: f(slots * cfg.hidden)?,
            moe_h: f((slots * cfg.moe_inter).max(cfg.dense_inter))?,
            descs_buf: DeviceBuf::new(slots * std::mem::size_of::<ExpertDesc>())?,
            wexpert_buf: f(slots)?,
            logits: f(cfg.vocab)?,
            argmax_dev: DeviceBuf::new(8)?, // [i32 index | f32 value]
            lc,
            rc,
            lc_scale,
            kv_fp8,
            n_kv_blocks,
            scores: vec![0.0; cfg.n_experts],
            choice: vec![0.0; cfg.n_experts],
            sel: Vec::with_capacity(cfg.top_k),
            descs: Vec::with_capacity(slots),
            w: Vec::with_capacity(slots),
            mlps: Vec::with_capacity(cfg.top_k),
            gl_host: Vec::with_capacity(cfg.n_experts * 4),
            pgl_host: Vec::with_capacity(cfg.n_experts * 4),
            argmax_host: Vec::with_capacity(8),
            pred_scores: vec![0.0; cfg.n_experts],
            pred_choice: vec![0.0; cfg.n_experts],
            pred_sel: Vec::with_capacity(cfg.top_k),
            prefetch: pin.prefetch_enabled(),
            prefetch_depth: pin.prefetch_depth(),
            prof: Profile::default(),
            pin,
        })
    }

    pub fn hits(&self) -> u64 {
        self.pin.hits
    }
    pub fn misses(&self) -> u64 {
        self.pin.misses
    }
    /// Cross-layer prefetch recall: (predicted experts actually selected, predicted).
    pub fn prefetch_recall(&self) -> (u64, u64) {
        (self.pin.pred_correct, self.pin.pred_total)
    }
    /// Total ms blocked in the prefetch drain (fetch NOT hidden behind compute).
    pub fn prefetch_wait_ms(&self) -> f64 {
        self.pin.prefetch_wait_ns as f64 / 1e6
    }

    /// DSA/MISA row selection for one full/shared layer at `pos`, returning the
    /// attend row set `(rows_ptr, nr)` — null pointer = dense over `0..nr`.
    /// `xnp` is the layer input (post input_layernorm), `qrp` the main path's
    /// q-LoRA residual (both device pointers, valid until the next sync). Full
    /// layers append this token's indexer key, then score + host top-k when the
    /// cache exceeds index_topk (below that it's exactly dense); shared layers
    /// reuse the nearest preceding full layer's selection (IndexShare).
    ///
    /// In MISA mode (`self.mode == AttnMode::Misa { active_heads }`) each token
    /// also folds its key into the block pool, and the scoring path first routes
    /// the top-`active_heads` indexer heads (a device estimate + nh-float D2H)
    /// and scores only those. DSA syncs once (the score D2H); MISA syncs twice on
    /// the scoring path (the router E_j D2H, then the score D2H).
    fn dsa_select_layer(
        &mut self,
        l: usize,
        pos: usize,
        xnp: *const f32,
        qrp: *const f32,
        ipin: Option<IndexerPin>,
    ) -> Result<(*const u32, usize)> {
        let cfg = self.cfg;
        let hd = cfg.index_head_dim;
        let nh = cfg.index_n_heads;
        let rope = cfg.qk_rope_head_dim;
        let theta = cfg.rope_theta();
        let topk = cfg.index_topk;
        let nt = pos + 1;
        // MISA routes a head subset; DSA scores all heads. Read the mode before
        // borrowing `self.idx` (usize is Copy — no move of self.mode).
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
        // MISA-only: this full layer's block pool (aligned with `kc` by slab).
        let poolp = if active_heads.is_some() {
            idx.pool[slab].ptr_mut() as *mut f32
        } else {
            std::ptr::null_mut()
        };

        // Key: wk·xn → LayerNorm(k_norm) → RoPE(first `rope` dims) → append. The
        // append runs EVERY token so the cache is ready when we cross the
        // threshold, even while the selection is still dense. MISA folds the same
        // roped key into the block pool on every token, for the same reason.
        // SAFETY: indexer weights are resident; scratch/kc/pool are live device
        // bufs; ordering is the null-stream program order; a sync precedes the D2H.
        unsafe {
            launch_gemv_bf16(xnp, ip.wk, hd, cfg.hidden, kp)?;
            launch_layernorm(
                kp,
                ip.k_norm_w,
                ip.k_norm_b,
                hd,
                crate::indexer::K_NORM_EPS,
                kp,
            )?;
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

        // Query heads (wq_b·qr, roped per head) + gates (weights_proj·xn), then
        // score every cached token and pick the top-k host-side.
        let wscale = 1.0 / (nh as f32).sqrt();
        let dscale = 1.0 / (hd as f32).sqrt();
        // SAFETY: as above; iqp/iwp are live scratch sized nh·hd / nh.
        unsafe {
            launch_gemv_bf16(qrp, ip.wq_b, nh * hd, cfg.q_lora_rank, iqp)?;
            launch_rope(iqp, nh, hd, rope, pos, theta)?; // per head: stride hd, seg rope
            launch_gemv_bf16(xnp, ip.weights_proj, nh, cfg.hidden, iwp)?;
        }

        // Active head set for the O(nt) token scan: all `nh` heads (DSA), or the
        // MISA-routed top-h. The router (paper Eq. 7-8) estimates each head's
        // contribution E_j = mean_b |w_j·ReLU(q_j·k̄_b)| from the block pool on
        // device, then a tiny nh-float D2H drives the host top-k pick. `h >= nh`
        // degenerates to "all heads" (the standard DSA path), so guard on h < nh.
        let (heads_ptr, nact): (*const u32, usize) = match active_heads {
            Some(h) if h < nh => {
                let m_blocks = nt.div_ceil(crate::indexer::MISA_BLOCK);
                let ppool = idx.pool[slab].ptr() as *const f32;
                let ep = idx.e.ptr_mut() as *mut f32;
                // SAFETY: iqp/iwp/ppool/ep are live device scratch sized nh·hd /
                // nh / m_blocks·hd / nh; a sync precedes the E_j D2H below.
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
                topk_into(&idx.e_f, h, &mut idx.head_sel);
                idx.heads_u32.clear();
                idx.heads_u32.extend(idx.head_sel.iter().map(|&i| i as u32));
                idx.heads_buf.copy_in_at(0, u32_le_bytes(&idx.heads_u32))?;
                (idx.heads_buf.ptr() as *const u32, idx.heads_u32.len())
            }
            _ => (std::ptr::null(), nh),
        };

        // SAFETY: iqp/iwp/kcp/scp are live scratch; heads_ptr is null (DSA) or
        // the just-uploaded `nact`-entry head buffer (MISA).
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
        self.rows_buf.copy_in_at(0, u32_le_bytes(&idx.rows))?;
        idx.last_dense = false;
        idx.last_nr = idx.rows.len();
        Ok((self.rows_buf.ptr() as *const u32, idx.rows.len()))
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

        // The KV slabs and rows_buf are sized to max_ctx; writing row pos
        // beyond that is a device-side out-of-bounds write, so refuse here
        // rather than corrupt device memory.
        ensure!(
            pos < self.max_ctx,
            "pos {pos} exceeds engine capacity max_ctx={}",
            self.max_ctx
        );

        // Position-based row selection (dense/streaming) is layer-blind, so it's
        // computed and uploaded ONCE per token here and reused by every layer's
        // attend; dense passes a null rows pointer (kernel fast path). DSA's
        // selection is per full layer and needs the mid-attention q-LoRA
        // residual, so it's computed inside the loop — `hoisted_rows` is None
        // then, signalling the per-layer path.
        let hoisted_rows: Option<(*const u32, usize)> = match &self.mode {
            AttnMode::Dense => Some((std::ptr::null(), pos + 1)),
            AttnMode::Streaming { sinks, window } => {
                streaming_rows(pos + 1, *sinks, *window, &mut self.rows_host);
                if self.rows_host.len() == pos + 1 {
                    Some((std::ptr::null(), pos + 1)) // all selected → dense
                } else {
                    self.rows_buf.copy_in_at(0, u32_le_bytes(&self.rows_host))?;
                    Some((self.rows_buf.ptr() as *const u32, self.rows_host.len()))
                }
            }
            // dsa/misa select per full layer inside the loop (they need the
            // mid-attention q-LoRA residual); `dsa_select_layer` reads the mode
            // to decide DSA vs MISA head routing.
            AttnMode::Dsa | AttnMode::Misa { .. } => None,
        };

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

        // Cold experts stream in per-MoE-layer via io_uring O_DIRECT (see
        // pin::resolve_layer) — no separate page-cache warm step.

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

            let rcp = self.rc[l].ptr_mut() as *mut u16;
            // fp8: `lc` is a u8 latent slab + `lc_scale` block scales; bf16:
            // `lc` is the u16 slab (lc8p/lscalep unused). One raw pointer each,
            // taken before the borrow-heavy launches.
            let lcp = self.lc[l].ptr_mut() as *mut u16;
            let lc8p = self.lc[l].ptr_mut();
            let lscalep = if self.kv_fp8 {
                self.lc_scale[l].ptr_mut() as *mut f32
            } else {
                std::ptr::null_mut()
            };
            let nb = self.n_kv_blocks;
            let kv_fp8 = self.kv_fp8;

            let indexer_pin = lw.indexer;

            // --- Attention phase 1: projections, ropes, cache append, absorb
            //     (all independent of the attended row set). ---
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
                if kv_fp8 {
                    launch_append_kv_fp8(
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
                } else {
                    launch_append_kv(compp, compp.add(kvl), lcp, rcp, pos, kvl, rope)?;
                }
                launch_mla_absorb(qp, kv_b.packed, kv_b.scale, h, qh, nope, vh, kvl, qabsp)?;
                launch_gather_rope(qp, qropep, h, qh, nope, rope)?;
            }

            // Row selection: hoisted (dense/streaming) or per-layer DSA (needs
            // `qrp`, the q-LoRA residual computed just above; `xnp` = the layer
            // input). DSA syncs mid-layer for the score D2H + host top-k.
            let (rows_ptr, nr) = match hoisted_rows {
                Some(rn) => rn,
                None => self.dsa_select_layer(l, pos, xnp, qrp, indexer_pin)?,
            };

            // --- Attention phase 2: sparse attend over the selected rows, then
            //     value projection, output projection, residual, pre-MLP norm. ---
            // SAFETY: see the forward-level note; every pointer is live scratch.
            unsafe {
                if kv_fp8 {
                    launch_attend_fp8(
                        qabsp, qropep, lc8p, lscalep, rcp, rows_ptr, h, nr, kvl, rope, nb, scale,
                        clatp,
                    )?;
                } else {
                    launch_attend(
                        qabsp, qropep, lcp, rcp, rows_ptr, h, nr, kvl, rope, scale, clatp,
                    )?;
                }
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

            // --- MLP sublayer (out fully written by the reduce; no pre-zero) ---
            if is_dense {
                let m = dense_mlp.ok_or_else(|| anyhow::anyhow!("dense layer {l} missing mlp"))?;
                self.descs.clear();
                self.descs.push(desc_of(&m));
                self.descs_buf.copy_in_at(0, desc_bytes(&self.descs))?;
                self.wexpert_buf.copy_in_at(0, f32_le_bytes(&[1.0f32]))?;
                // SAFETY: descs/wexpert/out are device scratch; weights resident.
                unsafe {
                    launch_moe(
                        xnp,
                        hidden,
                        cfg.dense_inter,
                        1,
                        self.descs_buf.ptr() as *const ExpertDesc,
                        self.wexpert_buf.ptr() as *const f32,
                        self.moe_h.ptr_mut() as *mut f32,
                        self.moe_partial.ptr_mut() as *mut f32,
                        self.moe_out.ptr_mut() as *mut f32,
                    )?;
                }
            } else {
                // Cross-layer prefetch: if the NEXT layer is also MoE, predict its
                // routed experts from L's post-attn residual `xp` (the cheap proxy
                // for L+1's input — the true input adds L's MLP delta we don't have
                // yet). The router gate + input_ln are always resident, so this is a
                // small norm + gemv folded under the same attention sync below.
                let predict = self.prefetch && l + 1 < cfg.n_layers && l + 1 >= cfg.dense_layers;
                let next_pred = if predict {
                    let nl = &self.pin.layers[l + 1];
                    if let LayerMlp::Moe { gate_w, .. } = &nl.mlp {
                        Some((nl.input_ln, *gate_w))
                    } else {
                        None // guarded MoE; stay safe rather than assume
                    }
                } else {
                    None
                };
                // Router gate on device, then read logits to route on host.
                // SAFETY: gate_w resident F32; glp device scratch.
                unsafe { launch_gemv_f32(xnp, gate_w, cfg.n_experts, hidden, glp)? };
                if let Some((next_ln, next_gate)) = next_pred {
                    // SAFETY: `xp` is the live post-attn residual; `next_ln`/
                    // `next_gate` are resident L+1 weights; pred_xn/pred_gl are device
                    // scratch. Same default stream → ordered after the gate above and
                    // drained by the sync that follows.
                    unsafe {
                        launch_rmsnorm(
                            xp,
                            next_ln,
                            hidden,
                            eps,
                            self.pred_xn.ptr_mut() as *mut f32,
                        )?;
                        launch_gemv_f32(
                            self.pred_xn.ptr() as *const f32,
                            next_gate,
                            cfg.n_experts,
                            hidden,
                            self.pred_gl.ptr_mut() as *mut f32,
                        )?;
                    }
                }
                let t = std::time::Instant::now();
                device_sync()?; // wait attention+gate (+ L+1 prediction) compute
                self.prof.attn_ns += t.elapsed().as_nanos();
                let t = std::time::Instant::now();
                // Split borrows: read the gate logits into a reused host buffer, then
                // route with `bias` borrowed straight out of `&self.pin` while the
                // routing scratch is borrowed mutably — no per-token bias clone.
                self.gate_logits.copy_out_into(&mut self.gl_host)?;
                route_into(
                    &self.gl_host,
                    self.pin.moe_bias(l),
                    cfg.top_k,
                    &mut self.scores,
                    &mut self.choice,
                    &mut self.sel,
                );
                // Predicted L+1 top-k (separate scratch; L's `scores`/`sel` still
                // feed L's own MoE weights below).
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
                self.prof.route_ns += t.elapsed().as_nanos();
                // Batch every cold miss through io_uring O_DIRECT (queue depth →
                // full NVMe bandwidth, straight into the VMM slots, one join) and
                // get the resolved descriptors back.
                let miss0 = self.pin.misses;
                let t = std::time::Instant::now();
                self.pin.resolve_layer(l, &self.sel, &mut self.mlps)?;
                self.prof.fetch_ns += t.elapsed().as_nanos();
                self.prof.fetch_n += self.pin.misses - miss0;
                // Submit L+1's predicted-expert reads NOW (non-blocking): the main
                // ring is quiescent (its drain just returned), and these reads run on
                // the NVMe/DMA side during this layer's MoE compute below. They are
                // reaped by `resolve_layer(l+1)`'s prefetch drain — hiding the fetch.
                //
                // Only the top `prefetch_depth` predictions (highest router score,
                // `pred_sel` is score-desc) are prefetched: the NVMe is bandwidth-
                // bound (~one 18 MB expert read saturates it), so the exploitable
                // budget is just the ~idle-during-compute window — a couple of experts
                // per layer. Higher-ranked predictions also have far higher per-expert
                // recall, so capping slashes the wasted-read volume that a full top_k
                // prefetch (36% mispredict) spends against the same saturated NVMe.
                if next_pred.is_some() {
                    let n = self.prefetch_depth.min(self.pred_sel.len());
                    self.pin.prefetch_layer(l + 1, &self.pred_sel[..n])?;
                }
                // Build the descriptor batch (+ record the hit-rate diagnostic) into
                // the reused `descs`/`w` fields — cleared, so no per-token alloc.
                self.descs.clear();
                self.w.clear();
                for (i, m) in self.mlps.iter().enumerate() {
                    self.descs.push(desc_of(m));
                    self.w.push(self.scores[self.sel[i]]);
                }
                // Weight = original sigmoid score, sum-normalized then scaled.
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
                // Shared expert(s), weight 1.0.
                if let Some(s) = shared {
                    self.descs.push(desc_of(&s));
                    self.w.push(1.0);
                }
                let ndesc = self.descs.len();
                self.descs_buf.copy_in_at(0, desc_bytes(&self.descs))?;
                self.wexpert_buf.copy_in_at(0, f32_le_bytes(&self.w))?;
                // SAFETY: descs point at resident/cold-slot weights valid until
                // the end-of-layer sync; all device-resident.
                unsafe {
                    launch_moe(
                        xnp,
                        hidden,
                        cfg.moe_inter,
                        ndesc,
                        self.descs_buf.ptr() as *const ExpertDesc,
                        self.wexpert_buf.ptr() as *const f32,
                        self.moe_h.ptr_mut() as *mut f32,
                        self.moe_partial.ptr_mut() as *mut f32,
                        self.moe_out.ptr_mut() as *mut f32,
                    )?;
                }
            }
            // SAFETY: residual add of the MLP contribution.
            unsafe { launch_vadd(xp, self.moe_out.ptr() as *const f32, hidden)? };
            // End-of-layer join: protects the reused descs/wexpert/moe_out
            // buffers before the next layer overwrites them, and surfaces faults.
            let t = std::time::Instant::now();
            device_sync()?;
            self.prof.mlp_ns += t.elapsed().as_nanos();
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
        let t = std::time::Instant::now();
        device_sync()?;
        self.prof.lmhead_ns += t.elapsed().as_nanos();
        Ok(())
    }

    /// Greedy argmax over the device logits — reduced ON DEVICE, so only 8 bytes
    /// (winning index + its value) come back per token instead of the full
    /// `vocab×f32` logits. The kernel reproduces the host fold EXACTLY: strict `>`
    /// (so ties keep the FIRST/lowest index and NaN never wins), returning
    /// `logits[best]` as the value; the finiteness bail is then the same
    /// `!value.is_finite()` check the host loop applied to `bv`.
    fn argmax(&mut self) -> Result<u32> {
        // SAFETY: logits is `vocab` device f32 (written + joined by the final
        // forward sync); argmax_dev owns 8 device bytes for [i32 index|f32 value].
        unsafe {
            launch_argmax(
                self.logits.ptr() as *const f32,
                self.cfg.vocab,
                self.argmax_dev.ptr_mut() as *mut i32,
                self.argmax_dev.ptr_mut().add(4) as *mut f32,
            )?;
        }
        // 8-byte D2H (blocking hipMemcpy, ordered after the kernel on the null stream).
        self.argmax_dev.copy_out_into(&mut self.argmax_host)?;
        ensure!(self.argmax_host.len() == 8, "argmax result must be 8 bytes");
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
        // Same bail as the host loop: `bv` == logits[best] == `val`.
        if !val.is_finite() {
            bail!("logits are non-finite (NaN/Inf in the GPU forward pass)");
        }
        ensure!(idx >= 0, "argmax returned negative index {idx}");
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
        self.prof = Profile::default();
        let decode_wall = std::time::Instant::now();
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
        self.prof.wall_ns = decode_wall.elapsed().as_nanos();
        self.prof.tokens = generated.len() as u64;
        self.prof.report();
        Ok(generated)
    }
}
