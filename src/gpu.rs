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
    launch_mla_value, launch_moe, launch_moe_batched, launch_rmsnorm, launch_rope, launch_vadd,
};
use crate::math::{sigmoid, topk_into};
use crate::model::ModelConfig;
use crate::pin::{IndexerPin, LayerMlp, Mlp, Pin};
use anyhow::{Result, bail, ensure};

/// Max positions verified in one speculative batch. The width-1 chain verifies
/// S=2 (the confirmed token + one draft). The width-2 SHARED-UNION TREE verifies
/// S=3 (the confirmed token + the MTP head's top-2 candidates for the next
/// position, both siblings sharing the per-layer expert union). Sized for the
/// larger; the fused-MoE kernel is S-generic (kernels/moe_fused.hip, `MAXS=8`).
const MAX_SPEC: usize = 3;

/// Topology of a [`GpuEngine::forward_batch`] batch: how the S positions map to
/// sequence positions and which KV rows each attends.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SpecTopo {
    /// Linear chain: position `s` sits at sequence `base_pos+s`, roped at
    /// `base_pos+s`, and attends the dense causal prefix `0..=base_pos+s`
    /// (each position sees the ones before it). The width-1 verify path.
    Chain,
    /// Width-2 tree, S=3: position 0 = the committed token at `base_pos`;
    /// positions 1 and 2 are SIBLINGS at sequence `base_pos+1` (both roped at
    /// `base_pos+1`). Position 1 lands at physical row `base_pos+1` and attends
    /// the dense prefix `0..=base_pos+1`; position 2 lands at physical row
    /// `base_pos+2` (a scratch slot) and attends the GATHERED prefix
    /// `[0..=base_pos, base_pos+2]` — the shared prefix plus its own row,
    /// SKIPPING sibling 1's row so the two candidates never see each other.
    Tree2,
}

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
    // MTP (layer-n_layers) device draft scratch — allocated only when the pin
    // carries the resident MTP layer. Its attention has its own bf16 KV slab.
    mtp_concat: DeviceBuf, // [2*hidden] f32: [enorm(emb) | hnorm(trunk)]
    mtp_x: DeviceBuf,      // [hidden] f32: the MTP residual stream
    mtp_lc: DeviceBuf,     // [max_ctx*kvl] bf16 latent
    mtp_rc: DeviceBuf,     // [max_ctx*rope] bf16 roped key
    // Speculative batched-verify scratch (S positions in one forward). Allocated
    // only with a resident MTP layer. `wtab` is the per-position dense expert
    // weight table (S*n_experts) for building the batched-MoE union weights.
    sx: DeviceBuf,       // [MAX_SPEC*hidden] f32: S residual streams
    sxn: DeviceBuf,      // [MAX_SPEC*hidden] f32: S post-attn-normed inputs
    smoe: DeviceBuf,     // [MAX_SPEC*hidden] f32: batched MoE output
    swexpert: DeviceBuf, // [MAX_SPEC*(MAX_SPEC*top_k+1)] f32: batched per-position weights
    swexpert_host: Vec<f32>,
    sh: DeviceBuf,          // batched MoE SwiGLU scratch [MAX_SPEC*Emax*moe_inter]
    spartial: DeviceBuf,    // batched MoE partials [MAX_SPEC*Emax*hidden]
    wtab: Vec<f32>,         // host [MAX_SPEC*n_experts] per-position weight table
    union: Vec<usize>,      // host union expert set this layer
    pred_union: Vec<usize>, // host union of the S positions' predicted L+1 experts (prefetch)
    // Speculative width: 1 = the S=2 linear chain (today's --spec); 2 = the
    // SHARED-UNION TREE (S=3, MTP top-2 candidates for the next position share
    // one expert union). Default 1; set via `set_spec_width`.
    spec_width: usize,
    tree_rows: DeviceBuf, // [max_ctx] u32: the gathered-KV-row list for sibling 2
    tree_rows_host: Vec<u32>, // host build buffer for `tree_rows`
    logits_host: Vec<u8>, // D2H staging for the MTP top-2 draft (width-2 only)
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
        // MTP scratch is sized only when the pin carries the resident MTP layer;
        // a non-MTP run must not pay for the (potentially large) MTP KV slabs.
        let has_mtp = pin.mtp().is_some();
        let mtp_kv = |elems: usize| DeviceBuf::new(if has_mtp { elems } else { 1 });
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
            // Shared by the single path (≤ `slots` descriptors) AND the batched
            // verify, whose per-layer union of S positions' experts can reach
            // MAX_SPEC*top_k + shared — size for the larger.
            descs_buf: DeviceBuf::new(
                (MAX_SPEC * cfg.top_k + cfg.n_shared).max(slots)
                    * std::mem::size_of::<ExpertDesc>(),
            )?,
            wexpert_buf: f(slots)?,
            logits: f(cfg.vocab)?,
            argmax_dev: DeviceBuf::new(8)?, // [i32 index | f32 value]
            lc,
            rc,
            lc_scale,
            kv_fp8,
            n_kv_blocks,
            mtp_concat: mtp_kv(2 * cfg.hidden * 4)?,
            mtp_x: mtp_kv(cfg.hidden * 4)?,
            mtp_lc: mtp_kv(max_ctx * kvl * 2)?,
            mtp_rc: mtp_kv(max_ctx * rope * 2)?,
            sx: mtp_kv(MAX_SPEC * cfg.hidden * 4)?,
            sxn: mtp_kv(MAX_SPEC * cfg.hidden * 4)?,
            smoe: mtp_kv(MAX_SPEC * cfg.hidden * 4)?,
            // batched-MoE union size ceiling: S disjoint top-k sets + 1 shared.
            swexpert: mtp_kv(MAX_SPEC * (MAX_SPEC * cfg.top_k + 1) * 4)?,
            swexpert_host: Vec::new(),
            // SwiGLU scratch: cover BOTH the MoE union path (Emax*moe_inter) and
            // the dense path (E=1, dense_inter) — same `.max` moe_h uses (line 387),
            // or the dense branch overruns `sh` on models where dense_inter is large.
            sh: mtp_kv(
                MAX_SPEC * ((MAX_SPEC * cfg.top_k + 1) * cfg.moe_inter).max(cfg.dense_inter) * 4,
            )?,
            spartial: mtp_kv(MAX_SPEC * (MAX_SPEC * cfg.top_k + 1) * cfg.hidden * 4)?,
            wtab: vec![0.0; MAX_SPEC * cfg.n_experts],
            union: Vec::with_capacity(MAX_SPEC * cfg.top_k),
            pred_union: Vec::with_capacity(MAX_SPEC * cfg.top_k),
            spec_width: 1,
            tree_rows: mtp_kv(max_ctx * 4)?, // [max_ctx] u32
            tree_rows_host: Vec::with_capacity(max_ctx),
            logits_host: Vec::new(),
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

    /// Select the speculative verify width: 1 = the S=2 linear chain (default),
    /// 2 = the shared-union tree (S=3, MTP top-2 candidates). Width 2 needs the
    /// resident MTP layer (same as any `--spec` run). Values other than 1 or 2
    /// are rejected — the tree scratch is sized for S ≤ [`MAX_SPEC`].
    pub fn set_spec_width(&mut self, width: usize) -> Result<()> {
        ensure!(
            width == 1 || width == 2,
            "spec width must be 1 or 2 (got {width})"
        );
        self.spec_width = width;
        Ok(())
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

    /// One main forward step: run the pass for `token` at `pos` and return the
    /// greedy prediction. The trunk hidden is left in `self.x` for a following
    /// [`mtp_draft`](Self::mtp_draft). (The MTP validation drives this.)
    pub fn step(&mut self, token: u32, pos: usize) -> Result<u32> {
        self.forward(token, pos)?;
        self.argmax()
    }

    /// Batched main forward over `tokens` through all 78 layers with **one**
    /// batched MoE per layer — the union of the S positions' routed experts is
    /// fetched once (the speculative-verify fetch amortization). Dense attention
    /// only (the spec loop runs below the sparsity threshold). Returns the S
    /// greedy predictions; each position's trunk is left in `sx[s*hidden]` for
    /// drafting. Appends KV at physical row `base_pos+s` for every position — the
    /// caller rolls back (by not advancing `pos`) any position whose draft was
    /// rejected.
    ///
    /// `topo` fixes the sequence geometry ([`SpecTopo`]): [`SpecTopo::Chain`]
    /// runs a linear causal chain (position `s` at sequence `base_pos+s`);
    /// [`SpecTopo::Tree2`] (requires S=3) makes positions 1 and 2 SIBLINGS at
    /// sequence `base_pos+1` — position 2 is roped there but lives at physical
    /// row `base_pos+2` and attends a GATHERED prefix that skips sibling 1, so
    /// the two candidates never see each other. In both topologies each of the S
    /// positions still routes independently, so the batched MoE output equals S
    /// separate forwards — only the KV geometry differs.
    fn forward_batch(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
        topo: SpecTopo,
    ) -> Result<Vec<u32>> {
        let cfg = self.cfg;
        let s_n = tokens.len();
        ensure!((1..=MAX_SPEC).contains(&s_n), "spec batch size {s_n}");
        ensure!(
            topo != SpecTopo::Tree2 || s_n == 3,
            "Tree2 topology needs exactly S=3 positions (got {s_n})"
        );
        // Tree2 physically uses rows base_pos..=base_pos+2; Chain uses ..base_pos+s_n.
        let rows_end = base_pos + s_n;
        ensure!(
            rows_end <= self.max_ctx,
            "spec batch [{base_pos},{rows_end}) exceeds max_ctx {}",
            self.max_ctx
        );
        // Build the sibling-2 gather list once (same for all layers): the shared
        // prefix rows 0..=base_pos plus its own physical row base_pos+2, ascending,
        // SKIPPING sibling 1's row base_pos+1. Empty pointer for the chain path.
        let tree_rows_ptr: *const u32 = if topo == SpecTopo::Tree2 {
            self.tree_rows_host.clear();
            self.tree_rows_host.extend(0..=base_pos as u32);
            self.tree_rows_host.push(base_pos as u32 + 2);
            self.tree_rows
                .copy_in_at(0, u32_le_bytes(&self.tree_rows_host))?;
            self.tree_rows.ptr() as *const u32
        } else {
            std::ptr::null()
        };
        let hidden = cfg.hidden;
        let eps = cfg.rms_norm_eps as f32;
        let (h, qh, nope, rope, kvl, vh) = (
            cfg.n_heads,
            cfg.qk_head_dim(),
            cfg.qk_nope_head_dim,
            cfg.qk_rope_head_dim,
            cfg.kv_lora_rank,
            cfg.v_head_dim,
        );
        let theta = cfg.rope_theta();
        let scale = 1.0 / (qh as f32).sqrt();
        let sxp = self.sx.ptr_mut() as *mut f32;
        let sxnp = self.sxn.ptr_mut() as *mut f32;
        let smoep = self.smoe.ptr_mut() as *mut f32;
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

        // SAFETY: all pointers are resident/scratch, valid until each device_sync.
        unsafe {
            for (s, &t) in tokens.iter().enumerate() {
                launch_embed_i8_row(
                    self.pin.embed.packed,
                    self.pin.embed.scale,
                    t as usize,
                    hidden,
                    sxp.add(s * hidden),
                )?;
            }
        }

        for l in 0..cfg.n_layers {
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

            // --- Attention, per position (Dense over its causal prefix). ---
            // SAFETY: null-stream ordered; each position appends its KV before
            // the next attends. In Chain, position s+1 sees position s; in Tree2,
            // sibling 2's gather list excludes sibling 1's row, so it does not.
            unsafe {
                for s in 0..s_n {
                    let phys = base_pos + s; // physical KV row this position writes
                    // Sequence geometry: Tree2's sibling 2 (s==2) is roped at
                    // base_pos+1 and attends the gathered prefix (skipping sibling
                    // 1's row base_pos+1); every other position is a chain node
                    // roped at its physical row over the dense causal prefix.
                    let (rope_pos, attend_len, rows_ptr): (usize, usize, *const u32) =
                        match (topo, s) {
                            (SpecTopo::Tree2, 2) => (base_pos + 1, base_pos + 2, tree_rows_ptr),
                            _ => (phys, phys + 1, std::ptr::null()),
                        };
                    let xsp = sxp.add(s * hidden);
                    launch_rmsnorm(xsp, input_ln, hidden, eps, xnp)?;
                    launch_gemv_i4(xnp, q_a.packed, q_a.scale, q_a.o_dim, q_a.i_dim, qrp)?;
                    launch_rmsnorm(qrp, q_a_ln, cfg.q_lora_rank, eps, qrp)?;
                    launch_gemv_i4(qrp, q_b.packed, q_b.scale, q_b.o_dim, q_b.i_dim, qp)?;
                    launch_gemv_i4(xnp, kv_a.packed, kv_a.scale, kv_a.o_dim, kv_a.i_dim, compp)?;
                    launch_rmsnorm(compp, kv_a_ln, kvl, eps, compp)?;
                    launch_rope(compp.add(kvl), 1, rope, rope, rope_pos, theta)?;
                    launch_rope(qp.add(nope), h, qh, rope, rope_pos, theta)?;
                    launch_append_kv(compp, compp.add(kvl), lcp, rcp, phys, kvl, rope)?;
                    launch_mla_absorb(qp, kv_b.packed, kv_b.scale, h, qh, nope, vh, kvl, qabsp)?;
                    launch_gather_rope(qp, qropep, h, qh, nope, rope)?;
                    launch_attend(
                        qabsp, qropep, lcp, rcp, rows_ptr, h, attend_len, kvl, rope, scale, clatp,
                    )?;
                    launch_mla_value(clatp, kv_b.packed, kv_b.scale, h, nope, vh, kvl, ctxp)?;
                    launch_gemv_i4(
                        ctxp,
                        o_proj.packed,
                        o_proj.scale,
                        o_proj.o_dim,
                        o_proj.i_dim,
                        subp,
                    )?;
                    launch_vadd(xsp, subp, hidden)?; // residual
                    launch_rmsnorm(xsp, post_ln, hidden, eps, sxnp.add(s * hidden))?; // → sxn[s]
                }
            }

            // --- MoE (batched over the S positions). ---
            if is_dense {
                let m = dense_mlp.ok_or_else(|| anyhow::anyhow!("dense layer {l} missing mlp"))?;
                self.descs.clear();
                self.descs.push(desc_of(&m));
                self.swexpert_host.clear();
                self.swexpert_host.resize(s_n, 1.0); // E=1, weight 1.0 each position
                self.descs_buf.copy_in_at(0, desc_bytes(&self.descs))?;
                self.swexpert
                    .copy_in_at(0, f32_le_bytes(&self.swexpert_host))?;
                // SAFETY: batched scratch sized for E=1; sxn/smoe device.
                unsafe {
                    launch_moe_batched(
                        sxnp,
                        hidden,
                        cfg.dense_inter,
                        1,
                        s_n,
                        self.descs_buf.ptr() as *const ExpertDesc,
                        self.swexpert.ptr() as *const f32,
                        self.sh.ptr_mut() as *mut f32,
                        self.spartial.ptr_mut() as *mut f32,
                        smoep,
                    )?;
                }
            } else {
                // Cross-layer prefetch (mirror forward()): predict L+1's routed
                // experts from each position's post-attn residual `sx[s]` (the cheap
                // proxy for L+1's input) and submit their cold reads after this
                // layer's demand fetch, so they overlap the batched MoE compute
                // below and are drained by resolve_layer(l+1) next iteration. This
                // is what makes the batched verify actually beat baseline — without
                // it the union fetch sits on the critical path (M3 was 0.53 vs 0.71).
                let predict = self.prefetch && l + 1 < cfg.n_layers && l + 1 >= cfg.dense_layers;
                let next_pred = if predict {
                    if let LayerMlp::Moe { gate_w, .. } = &self.pin.layers[l + 1].mlp {
                        Some((self.pin.layers[l + 1].input_ln, *gate_w))
                    } else {
                        None
                    }
                } else {
                    None
                };
                // Route each position → per-position weight table `wtab`, and
                // accumulate the predicted L+1 union across positions.
                for w in self.wtab.iter_mut() {
                    *w = 0.0;
                }
                self.pred_union.clear();
                for s in 0..s_n {
                    // SAFETY: gate_w resident; sxn[s]/glp device scratch. When
                    // predicting, also norm+gate sx[s] with L+1's weights into the
                    // pred scratch — same null stream, drained by the sync below.
                    unsafe {
                        launch_gemv_f32(sxnp.add(s * hidden), gate_w, cfg.n_experts, hidden, glp)?;
                        if let Some((next_ln, next_gate)) = next_pred {
                            launch_rmsnorm(
                                sxp.add(s * hidden),
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
                    };
                    device_sync()?;
                    self.gate_logits.copy_out_into(&mut self.gl_host)?;
                    route_into(
                        &self.gl_host,
                        self.pin.moe_bias(l),
                        cfg.top_k,
                        &mut self.scores,
                        &mut self.choice,
                        &mut self.sel,
                    );
                    let mut sm: f32 = self.sel.iter().map(|&e| self.scores[e]).sum();
                    if cfg.norm_topk_prob {
                        sm += 1e-20;
                    }
                    for &e in &self.sel {
                        let mut wv = self.scores[e];
                        if cfg.norm_topk_prob {
                            wv /= sm;
                        }
                        self.wtab[s * cfg.n_experts + e] = wv * cfg.routed_scale as f32;
                    }
                    // Predicted L+1 top-k for this position → merge the top
                    // `prefetch_depth` into the cross-position prefetch union.
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
                        let n = self.prefetch_depth.min(self.pred_sel.len());
                        for &e in &self.pred_sel[..n] {
                            if !self.pred_union.contains(&e) {
                                self.pred_union.push(e);
                            }
                        }
                    }
                }
                // Union of selected experts, fetched once.
                self.union.clear();
                for e in 0..cfg.n_experts {
                    if (0..s_n).any(|s| self.wtab[s * cfg.n_experts + e] != 0.0) {
                        self.union.push(e);
                    }
                }
                self.pin.resolve_layer(l, &self.union, &mut self.mlps)?;
                // Submit L+1's predicted reads now: the demand ring just drained, so
                // these run on the NVMe/DMA side during the batched MoE compute below.
                if next_pred.is_some() && !self.pred_union.is_empty() {
                    self.pin.prefetch_layer(l + 1, &self.pred_union)?;
                }
                self.descs.clear();
                for m in &self.mlps {
                    self.descs.push(desc_of(m));
                }
                let shared_expert =
                    shared.ok_or_else(|| anyhow::anyhow!("moe layer {l} missing shared"))?;
                self.descs.push(desc_of(&shared_expert));
                let e_total = self.descs.len();
                // Per-position weights over the union descs (+ shared, weight 1.0).
                self.swexpert_host.clear();
                self.swexpert_host.resize(s_n * e_total, 0.0);
                for s in 0..s_n {
                    for (i, &e) in self.union.iter().enumerate() {
                        self.swexpert_host[s * e_total + i] = self.wtab[s * cfg.n_experts + e];
                    }
                    self.swexpert_host[s * e_total + self.union.len()] = 1.0; // shared
                }
                self.descs_buf.copy_in_at(0, desc_bytes(&self.descs))?;
                self.swexpert
                    .copy_in_at(0, f32_le_bytes(&self.swexpert_host))?;
                // SAFETY: descs point at resident/streamed experts; batched scratch sized.
                unsafe {
                    launch_moe_batched(
                        sxnp,
                        hidden,
                        cfg.moe_inter,
                        e_total,
                        s_n,
                        self.descs_buf.ptr() as *const ExpertDesc,
                        self.swexpert.ptr() as *const f32,
                        self.sh.ptr_mut() as *mut f32,
                        self.spartial.ptr_mut() as *mut f32,
                        smoep,
                    )?;
                }
            }
            // Residual add per position.
            // SAFETY: sx[s]/smoe[s] device scratch.
            unsafe {
                for s in 0..s_n {
                    launch_vadd(sxp.add(s * hidden), smoep.add(s * hidden), hidden)?;
                }
            }
            device_sync()?;
        }

        // Final norm + tied head + argmax, per position.
        let mut preds = Vec::with_capacity(s_n);
        for s in 0..s_n {
            // SAFETY: final_norm/lm_head resident; sx[s]/xn/logits device.
            unsafe {
                launch_rmsnorm(sxp.add(s * hidden), self.pin.final_norm, hidden, eps, xnp)?;
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
            preds.push(self.argmax()?);
        }
        Ok(preds)
    }

    /// Device MTP draft (M2). Given the just-completed main forward's trunk
    /// hidden (`self.x`, valid after [`forward`]) and the token it emitted,
    /// drafts token t+2 through the resident MTP layer, mirroring the scalar
    /// oracle (src/mtp.rs): eh_proj([enorm(embed(next)) | hnorm(trunk)]) → the
    /// layer-`n_layers` transformer layer (its own bf16 KV, Dense attention) →
    /// tied lm_head. `pos` is the MTP KV's current token count. Every op is an
    /// already-validated launcher; no new kernels. Returns the greedy draft.
    pub fn mtp_draft(&mut self, next_token: u32, pos: usize) -> Result<u32> {
        let trunk = self.x.ptr() as *const f32;
        self.mtp_draft_trunk(trunk, next_token, pos)
    }

    /// [`mtp_draft`](Self::mtp_draft) from an explicit device trunk pointer. The
    /// speculative loop drafts from each batched position's trunk left in `sx`,
    /// not the single `self.x` — so it passes the right `sx[s*hidden]` here.
    fn mtp_draft_trunk(&mut self, trunk: *const f32, next_token: u32, pos: usize) -> Result<u32> {
        self.mtp_logits_trunk(trunk, next_token, pos)?;
        self.argmax()
    }

    /// Width-2 draft: the MTP head's TOP-2 candidate tokens for the next position
    /// (`(top1, top2)`, top1 == the [`mtp_draft_trunk`](Self::mtp_draft_trunk)
    /// argmax). Same forward as `mtp_draft_trunk`; only the tail differs (a host
    /// top-2 over the draft logits instead of the device argmax). Greedy-equiv is
    /// unaffected — these are only *candidates*; the emitted token is always the
    /// main model's argmax, confirmed in the batched verify.
    fn mtp_draft2_trunk(
        &mut self,
        trunk: *const f32,
        next_token: u32,
        pos: usize,
    ) -> Result<(u32, u32)> {
        self.mtp_logits_trunk(trunk, next_token, pos)?;
        // The draft logits are device-side in `self.logits`; a single D2H (drafts
        // happen once or twice per verify round, off the per-layer hot path) then
        // a host top-2 with the SAME lowest-index tie-break the device argmax uses,
        // so `top1` matches `mtp_draft_trunk` exactly.
        self.logits.copy_out_into(&mut self.logits_host)?;
        ensure!(
            self.logits_host.len() == self.cfg.vocab * 4,
            "draft logits D2H size mismatch"
        );
        let (mut i1, mut v1) = (0usize, f32::NEG_INFINITY);
        let (mut i2, mut v2) = (0usize, f32::NEG_INFINITY);
        for (i, c) in self.logits_host.chunks_exact(4).enumerate() {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if v > v1 {
                v2 = v1;
                i2 = i1;
                v1 = v;
                i1 = i;
            } else if v > v2 {
                v2 = v;
                i2 = i;
            }
        }
        if !v1.is_finite() {
            bail!("MTP draft logits non-finite (NaN/Inf in the MTP forward)");
        }
        Ok((i1 as u32, i2 as u32))
    }

    /// Shared core of the MTP draft: runs the full layer-`n_layers` MTP forward
    /// from `trunk` + `next_token`, leaving the draft logits in `self.logits`
    /// (device) after the join. The public drafts wrap this with an argmax /
    /// top-2 tail. See [`mtp_draft`](Self::mtp_draft) for the formulation.
    fn mtp_logits_trunk(&mut self, trunk: *const f32, next_token: u32, pos: usize) -> Result<()> {
        // The MTP KV slabs are sized to max_ctx; writing row `pos` beyond that is
        // an out-of-bounds device write (same guard forward() has). M3's decode
        // loop can advance past the main pos, so refuse rather than corrupt.
        ensure!(
            pos < self.max_ctx,
            "mtp_draft pos {pos} exceeds engine capacity max_ctx={}",
            self.max_ctx
        );
        let cfg = self.cfg;
        let hidden = cfg.hidden;
        let eps = cfg.rms_norm_eps as f32;
        let (h, qh, nope, rope, kvl, vh) = (
            cfg.n_heads,
            cfg.qk_head_dim(),
            cfg.qk_nope_head_dim,
            cfg.qk_rope_head_dim,
            cfg.kv_lora_rank,
            cfg.v_head_dim,
        );
        let theta = cfg.rope_theta();
        let scale = 1.0 / (qh as f32).sqrt();

        // Copy the resident MTP pointers/weights out (all Copy → ends the &pin
        // borrow so the scratch below can borrow &mut self freely).
        let mp = self
            .pin
            .mtp()
            .ok_or_else(|| anyhow::anyhow!("mtp_draft without a resident MTP layer"))?;
        let (input_ln, post_ln, enorm, hnorm, shnorm, eh_proj, gate_w) = (
            mp.input_ln,
            mp.post_ln,
            mp.enorm,
            mp.hnorm,
            mp.shnorm,
            mp.eh_proj,
            mp.gate_w,
        );
        let (q_a, q_a_ln, q_b, kv_a, kv_a_ln, kv_b, o_proj) = (
            mp.q_a, mp.q_a_ln, mp.q_b, mp.kv_a, mp.kv_a_ln, mp.kv_b, mp.o_proj,
        );
        let shared = mp.shared;

        let xp = trunk; // trunk hidden (self.x for mtp_draft, sx[s] for the spec loop)
        let concatp = self.mtp_concat.ptr_mut() as *mut f32;
        let xmp = self.mtp_x.ptr_mut() as *mut f32;
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
        let lcp = self.mtp_lc.ptr_mut() as *mut u16;
        let rcp = self.mtp_rc.ptr_mut() as *mut u16;

        // Glue + MTP-layer attention (Dense over the MTP's own KV).
        // SAFETY: every pointer is resident/scratch, valid until the sync below.
        unsafe {
            launch_embed_i8_row(
                self.pin.embed.packed,
                self.pin.embed.scale,
                next_token as usize,
                hidden,
                concatp,
            )?;
            launch_rmsnorm(concatp, enorm, hidden, eps, concatp)?; // enorm(emb)
            launch_rmsnorm(xp, hnorm, hidden, eps, concatp.add(hidden))?; // hnorm(trunk)
            launch_gemv_bf16(concatp, eh_proj, hidden, 2 * hidden, xmp)?; // x = eh_proj·concat
            launch_rmsnorm(xmp, input_ln, hidden, eps, xnp)?;
            launch_gemv_i4(xnp, q_a.packed, q_a.scale, q_a.o_dim, q_a.i_dim, qrp)?;
            launch_rmsnorm(qrp, q_a_ln, cfg.q_lora_rank, eps, qrp)?;
            launch_gemv_i4(qrp, q_b.packed, q_b.scale, q_b.o_dim, q_b.i_dim, qp)?;
            launch_gemv_i4(xnp, kv_a.packed, kv_a.scale, kv_a.o_dim, kv_a.i_dim, compp)?;
            launch_rmsnorm(compp, kv_a_ln, kvl, eps, compp)?;
            launch_rope(compp.add(kvl), 1, rope, rope, pos, theta)?;
            launch_rope(qp.add(nope), h, qh, rope, pos, theta)?;
            launch_append_kv(compp, compp.add(kvl), lcp, rcp, pos, kvl, rope)?;
            launch_mla_absorb(qp, kv_b.packed, kv_b.scale, h, qh, nope, vh, kvl, qabsp)?;
            launch_gather_rope(qp, qropep, h, qh, nope, rope)?;
            launch_attend(
                qabsp,
                qropep,
                lcp,
                rcp,
                std::ptr::null(),
                h,
                pos + 1,
                kvl,
                rope,
                scale,
                clatp,
            )?;
            launch_mla_value(clatp, kv_b.packed, kv_b.scale, h, nope, vh, kvl, ctxp)?;
            launch_gemv_i4(
                ctxp,
                o_proj.packed,
                o_proj.scale,
                o_proj.o_dim,
                o_proj.i_dim,
                subp,
            )?;
            launch_vadd(xmp, subp, hidden)?; // attn residual
            launch_rmsnorm(xmp, post_ln, hidden, eps, xnp)?; // pre-MoE norm
            launch_gemv_f32(xnp, gate_w, cfg.n_experts, hidden, glp)?; // router gate
        }
        device_sync()?;

        // Route on host over the 256 resident MTP experts (gate_bias borrowed
        // out of &self.pin while the routing scratch is &mut — disjoint fields).
        self.gate_logits.copy_out_into(&mut self.gl_host)?;
        route_into(
            &self.gl_host,
            &self
                .pin
                .mtp()
                .ok_or_else(|| anyhow::anyhow!("mtp gate_bias vanished"))?
                .gate_bias,
            cfg.top_k,
            &mut self.scores,
            &mut self.choice,
            &mut self.sel,
        );
        // Descriptors: routed experts[sel] (score-weighted) + shared (1.0).
        self.descs.clear();
        self.w.clear();
        for i in 0..self.sel.len() {
            let e = self.sel[i];
            let m = self
                .pin
                .mtp()
                .ok_or_else(|| anyhow::anyhow!("mtp experts vanished"))?
                .experts[e];
            self.descs.push(desc_of(&m));
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
        self.descs.push(desc_of(&shared));
        self.w.push(1.0);
        let ndesc = self.descs.len();
        self.descs_buf.copy_in_at(0, desc_bytes(&self.descs))?;
        self.wexpert_buf.copy_in_at(0, f32_le_bytes(&self.w))?;

        // MoE → residual → tied head → argmax.
        // SAFETY: descs/weights/moe scratch resident; lm_head resident.
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
            launch_vadd(xmp, self.moe_out.ptr() as *const f32, hidden)?; // MoE residual
            launch_rmsnorm(xmp, shnorm, hidden, eps, xnp)?; // shared_head.norm
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

    /// MTP speculative decode: draft the next token with the layer-78 module, then
    /// **verify** it by running the main model over `[cur, draft]` as one batched
    /// forward (the two positions share a single union expert-fetch — the whole
    /// point). On a correct draft both tokens land per main-model forward; on a
    /// wrong one only `cur`'s successor is committed and the draft's KV is rolled
    /// back (overwritten next iteration). The emitted tokens are always the main
    /// model's greedy argmaxes, so the output is **greedy-equivalent** to
    /// [`generate`]: the draft only changes batching, never the result. Not
    /// bit-identical by construction — the batched MoE reduces a position's experts
    /// in union (ascending-id) order while `generate` reduces in score-desc order,
    /// so logits can differ by a ULP and flip a genuine near-tie argmax. That is
    /// vanishingly rare (0 divergence over the M3 validation) and is the same
    /// FP-order freedom greedy decode already has. Returns the generated ids.
    pub fn generate_spec(
        &mut self,
        prompt_ids: &[u32],
        ngen: usize,
        eos: &[u32],
    ) -> Result<Vec<u32>> {
        ensure!(
            self.pin.mtp().is_some(),
            "generate_spec needs a resident MTP layer — build the snapshot with --mtp"
        );
        // `forward_batch` implements only the dense bf16 attention path. Refuse
        // rather than silently attend the wrong rows / reinterpret an fp8 KV slab
        // as bf16 — either would make the verify diverge from the main model.
        ensure!(
            matches!(self.mode, AttnMode::Dense),
            "generate_spec: the batched verify supports Dense attention only (mode is {:?})",
            self.mode
        );
        ensure!(
            !self.kv_fp8,
            "generate_spec: the batched verify uses the bf16 KV path; --kv-fp8 is unsupported"
        );
        ensure!(!prompt_ids.is_empty(), "empty prompt");
        // Width 2 dispatches to the shared-union tree; width 1 is the S=2 chain
        // below (byte-identical to the pre-tree --spec path).
        if self.spec_width >= 2 {
            return self.generate_spec_tree(prompt_ids, ngen, eos);
        }
        let hidden = self.cfg.hidden;

        // --- Prefill: main KV + MTP KV built in lockstep over the prompt. The
        // last prompt token's forward yields the first generated token (`cur`) and
        // its MTP draft (the guess for the token after it). ---
        let mut pos = 0usize;
        let mut cur = 0u32;
        let mut draft = 0u32;
        let n = prompt_ids.len();
        for (i, &tok) in prompt_ids.iter().enumerate() {
            self.forward(tok, pos)?; // main KV @pos; self.x = trunk_pos; logits ready
            let pred = self.argmax()?; // token at pos+1 (the real next for the last)
            let next = if i + 1 < n { prompt_ids[i + 1] } else { pred };
            let d = self.mtp_draft(next, pos)?; // MTP KV @pos (uses self.x); draft pos+2
            if i + 1 == n {
                cur = next; // == pred: first generated token, at position `pos`
                draft = d; // MTP guess for position pos+1
            }
            pos += 1;
        }
        // pos == n; main & MTP KV hold 0..pos-1; `cur` is the token at position `pos`.

        self.prof = Profile::default();
        let decode_wall = std::time::Instant::now();
        let mut out: Vec<u32> = Vec::with_capacity(ngen);
        let mut accepted = 0usize; // drafts that verified
        let mut spec_iters = 0usize; // batched-verify rounds

        loop {
            if eos.contains(&cur) {
                break;
            }
            out.push(cur);
            if out.len() >= ngen {
                break;
            }
            // Need room to append KV for both `cur`@pos and `draft`@pos+1.
            if pos + MAX_SPEC > self.max_ctx {
                // Out of context to batch a draft — finish `cur`'s successor with a
                // plain step and stop (benchmarks stay well under max_ctx).
                self.forward(cur, pos)?;
                let last = self.argmax()?;
                if !eos.contains(&last) && out.len() < ngen {
                    out.push(last);
                }
                break;
            }

            spec_iters += 1;
            let preds = self.forward_batch(&[cur, draft], pos, SpecTopo::Chain)?; // KV @pos, @pos+1
            let pa = preds[0]; // real token at pos+1
            let pb = preds[1]; // token at pos+2 (real iff draft == pa)
            // sx[0] = trunk_pos (predicts pa); sx[1] = trunk_{pos+1} (predicts pb).
            let sx = self.sx.ptr() as *const f32;

            if draft == pa {
                // ACCEPT — draft matched the main model; pa@pos+1 and pb@pos+2 real.
                accepted += 1;
                if eos.contains(&pa) {
                    cur = pa; // loop head breaks without emitting eos
                    continue;
                }
                out.push(pa);
                if out.len() >= ngen {
                    break;
                }
                // MTP KV lockstep: append @pos (trunk sx[0], next pa) and @pos+1
                // (trunk sx[1], next pb); keep the second draft for pos+3.
                self.mtp_draft_trunk(sx, pa, pos)?;
                let d2 = self.mtp_draft_trunk(unsafe { sx.add(hidden) }, pb, pos + 1)?;
                cur = pb;
                draft = d2;
                pos += 2;
            } else {
                // REJECT — draft wrong; only pa@pos+1 real. cur's KV @pos stays; the
                // draft's stale KV @pos+1 is overwritten next round (not advanced past).
                let d1 = self.mtp_draft_trunk(sx, pa, pos)?; // MTP KV @pos; draft pos+2
                cur = pa;
                draft = d1;
                pos += 1;
            }
        }

        self.prof.wall_ns = decode_wall.elapsed().as_nanos();
        self.prof.tokens = out.len() as u64;
        self.prof.report();
        let acc_pct = 100.0 * accepted as f64 / spec_iters.max(1) as f64;
        tracing::info!(
            "spec: {} tokens in {spec_iters} verify rounds, {accepted} accepted ({acc_pct:.1}%)",
            out.len(),
        );
        Ok(out)
    }

    /// Relocate one bf16 KV row (`from_row` → `to_row`) in every layer's latent
    /// and roped-key slab. The width-2 tree writes sibling 2's KV at physical row
    /// `base_pos+2`, but when that sibling wins its token is the confirmed one at
    /// sequence position `base_pos+1`, so its KV must occupy the canonical row
    /// `base_pos+1` before decode continues. The two rows are distinct, so each
    /// copy is a non-overlapping device-to-device move.
    fn spec_relocate_kv(&mut self, from_row: usize, to_row: usize) -> Result<()> {
        debug_assert_ne!(from_row, to_row, "relocate would alias a KV row");
        let n_layers = self.cfg.n_layers;
        let lat = self.cfg.kv_lora_rank * 2; // bf16 latent bytes per row
        let key = self.cfg.qk_rope_head_dim * 2; // bf16 roped-key bytes per row
        for l in 0..n_layers {
            self.lc[l].copy_within(to_row * lat, from_row * lat, lat)?;
            self.rc[l].copy_within(to_row * key, from_row * key, key)?;
        }
        Ok(())
    }

    /// Width-2 speculative decode — the SHARED-UNION TREE. Each round drafts the
    /// MTP head's TOP-2 candidates (`da`, `db`) for the next position and verifies
    /// BOTH in one S=3 batched forward `[cur, da, db]` ([`SpecTopo::Tree2`]) that
    /// shares the per-layer expert union across the two siblings — two near-tie
    /// candidates route through overlapping experts, so the union grows
    /// sub-linearly while covering more probability mass per fetched byte.
    /// Whichever candidate equals the main model's argmax `pa` is confirmed and
    /// its own next-token prediction is committed for free; if neither matches,
    /// only `pa` is emitted (like the chain reject). Greedy-equivalent by
    /// construction: every emitted token is the main model's argmax at its
    /// committed position — the tree only decides whether the NEXT token is
    /// confirmed early, never WHICH token is emitted.
    fn generate_spec_tree(
        &mut self,
        prompt_ids: &[u32],
        ngen: usize,
        eos: &[u32],
    ) -> Result<Vec<u32>> {
        let hidden = self.cfg.hidden;

        // --- Prefill: main + MTP KV in lockstep (as the chain path), but the
        // final step drafts the TOP-2 candidates for the first predicted position. ---
        let mut pos = 0usize;
        let mut cur = 0u32;
        let (mut da, mut db) = (0u32, 0u32); // top-2 candidates for position pos+1
        let n = prompt_ids.len();
        for (i, &tok) in prompt_ids.iter().enumerate() {
            self.forward(tok, pos)?; // main KV @pos; self.x = trunk_pos
            let pred = self.argmax()?; // token at pos+1 (real next for the last)
            let next = if i + 1 < n { prompt_ids[i + 1] } else { pred };
            let trunk = self.x.ptr() as *const f32;
            let (a, b) = self.mtp_draft2_trunk(trunk, next, pos)?; // MTP KV @pos
            if i + 1 == n {
                cur = next; // == pred: first generated token, at position `pos`
                da = a;
                db = b;
            }
            pos += 1;
        }

        self.prof = Profile::default();
        let decode_wall = std::time::Instant::now();
        let mut out: Vec<u32> = Vec::with_capacity(ngen);
        let mut accepted = 0usize; // rounds where a candidate confirmed
        let mut spec_iters = 0usize; // batched-verify rounds

        loop {
            if eos.contains(&cur) {
                break;
            }
            out.push(cur);
            if out.len() >= ngen {
                break;
            }
            // Need physical rows pos, pos+1, pos+2 for the S=3 tree.
            if pos + MAX_SPEC > self.max_ctx {
                self.forward(cur, pos)?;
                let last = self.argmax()?;
                if !eos.contains(&last) && out.len() < ngen {
                    out.push(last);
                }
                break;
            }

            spec_iters += 1;
            // S=3 tree verify: position 0 = cur@pos; siblings da,db @pos+1.
            let preds = self.forward_batch(&[cur, da, db], pos, SpecTopo::Tree2)?;
            let pa = preds[0]; // real token at pos+1 (the main model's argmax)
            // sx[s] = trunk of position s; sx[1]/sx[2] predict pos+2 given da/db.
            let sx = self.sx.ptr() as *const f32;

            // Confirm a candidate: prefer da (the top-1 draft). When da == db == pa
            // the choice is immaterial; da is checked first so sibling 1's already-
            // canonical KV row is used and no relocation is needed.
            let win = if da == pa {
                Some((1usize, preds[1])) // sibling 1: trunk sx[1], KV already @pos+1
            } else if db == pa {
                Some((2usize, preds[2])) // sibling 2: trunk sx[2], KV @pos+2
            } else {
                None
            };

            match win {
                Some((s_win, pb)) => {
                    // CONFIRM — pa@pos+1 real, and pb@pos+2 real (the winning
                    // sibling's input matched pa, so its forward is the true one).
                    accepted += 1;
                    if eos.contains(&pa) {
                        cur = pa; // loop head breaks without emitting eos
                        continue;
                    }
                    out.push(pa);
                    if out.len() >= ngen {
                        break;
                    }
                    // The confirmed token pa's canonical KV must live at row pos+1.
                    // Sibling 1 already wrote it there; sibling 2 wrote it at pos+2,
                    // so relocate that row into pos+1 across every layer.
                    if s_win == 2 {
                        self.spec_relocate_kv(pos + 2, pos + 1)?;
                    }
                    // SAFETY: sx[s_win] is within the S*hidden `sx` slab (s_win ≤ 2).
                    let win_trunk = unsafe { sx.add(s_win * hidden) };
                    // MTP KV lockstep: append @pos (sx[0], next pa) and @pos+1
                    // (winning trunk, next pb); redraft top-2 for pos+3 from pb.
                    self.mtp_draft_trunk(sx, pa, pos)?;
                    let (a2, b2) = self.mtp_draft2_trunk(win_trunk, pb, pos + 1)?;
                    cur = pb;
                    da = a2;
                    db = b2;
                    pos += 2;
                }
                None => {
                    // REJECT — neither candidate matched; only pa@pos+1 real. cur's
                    // KV @pos stays; the stale sibling KV @pos+1,@pos+2 is overwritten
                    // next round (pos advances by 1, never attended before rewrite).
                    let (a1, b1) = self.mtp_draft2_trunk(sx, pa, pos)?; // MTP KV @pos
                    cur = pa;
                    da = a1;
                    db = b1;
                    pos += 1;
                }
            }
        }

        self.prof.wall_ns = decode_wall.elapsed().as_nanos();
        self.prof.tokens = out.len() as u64;
        self.prof.report();
        let acc_pct = 100.0 * accepted as f64 / spec_iters.max(1) as f64;
        tracing::info!(
            "spec tree(width=2): {} tokens in {spec_iters} verify rounds, \
             {accepted} confirmed ({acc_pct:.1}%)",
            out.len(),
        );
        Ok(out)
    }
}
