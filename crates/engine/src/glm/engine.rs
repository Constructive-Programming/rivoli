//! The GLM engine: device scratch, KV slabs, streams — the state `forward.rs` and
//! `decode.rs` drive. Ported from `old:src/gpu.rs::GpuEngine` under the M4 narrowings
//! (dense attention, single routed format, no MTP, no instruments): what remains is the
//! decode loop's minimum state, and each deferred field returns with the feature that
//! reads it rather than as dead weight now.

use crate::device::DeviceBuf;
use crate::fetch::asyncfetch::Ticket;
use crate::glm::MAXROW;
use crate::glm::pin::GlmPin;
use crate::routed::ResolvedBatch;
use anyhow::{Result, ensure};
use rivoli_artifact::format::RoutedFmt;
use rivoli_artifact::glm_config::ModelConfig;
use rivoli_backend::gpustream::HipStream;
use rivoli_backend::{ExpertDesc, fill_u32};
use rivoli_core::num::E4M3_BLOCK;

/// Fixed-point MoE accumulator rows per token row — one per STREAM (compute + miss),
/// drained as a sum. Integer addition is associative, so the two streams need no join
/// to share a result; separate rows are a contention fix, not a correctness one.
pub(super) const MOE_ACC_ROWS: usize = 2;

/// Device argmax result bytes: `MAXROW` pairs of [i32 index | f32 max-value], then one
/// u32 non-finite tag shared by every row.
pub(super) const ARGMAX_BYTES: usize = MAXROW * 8 + 4;

/// The GLM decode engine: resident pin + per-token device scratch, allocated once.
pub struct GlmEngine<'a> {
    pub(super) pin: GlmPin<'a>,
    pub(super) cfg: &'a ModelConfig,
    /// KV-slab capacity in tokens; `forward` refuses `pos` beyond it.
    pub(super) max_ctx: usize,
    /// Prefill the prompt LAYER-MAJOR. Always on except while capturing a trace: a v2
    /// trace has no token delimiter and recovers one from the layer id DESCENDING,
    /// which a layer-major prefill never does — a capture under it is silently
    /// mis-segmented, the worst shape for a file that costs a sole-tenant GPU half an
    /// hour. Falling back (not refusing) because there is no flag left to comply with.
    pub(super) layer_major_prefill: bool,
    /// Residual stream. The ONE buffer that can be wider than [`MAXROW`]: layer-major
    /// prefill needs every prompt token's hidden state live across the whole model,
    /// since layer L reads what L−1 wrote for every row.
    pub(super) x: DeviceBuf,
    /// Rows `x` was allocated for — the bound `forward_inner` checks `x_off + nrow`
    /// against.
    pub(super) x_rows: usize,
    pub(super) xn: DeviceBuf,
    pub(super) attn_out: DeviceBuf,
    pub(super) q_lora: DeviceBuf,
    pub(super) q: DeviceBuf,
    pub(super) compressed_kv: DeviceBuf,
    pub(super) q_absorbed: DeviceBuf,
    pub(super) q_rope: DeviceBuf,
    pub(super) ctx_latent: DeviceBuf,
    /// Split-KV partial scratch, sized ONCE for the attend kernel's worst-case split
    /// count so every context length reuses it.
    pub(super) attn_partial: DeviceBuf,
    pub(super) attn_ctx: DeviceBuf,
    pub(super) gate_logits: DeviceBuf,
    // Dense-MLP fp8 SwiGLU scratch (gate/up projections, dense_inter wide).
    pub(super) mlp_g: DeviceBuf,
    pub(super) mlp_u: DeviceBuf,
    pub(super) moe_out: DeviceBuf,
    /// `[MOE_ACC_ROWS][MAXROW][hidden]` u64 fixed-point MoE accumulator, drained into
    /// the residual at end of layer. Token and hidden axes are contiguous, so the drain
    /// never has to know they are two axes.
    pub(super) moe_acc: DeviceBuf,
    /// `[slots][MAXROW][moe_inter]` SwiGLU hidden scratch (routed MoE).
    pub(super) moe_hidden: DeviceBuf,
    pub(super) descs_buf: DeviceBuf,
    pub(super) wexpert_buf: DeviceBuf,
    pub(super) logits: DeviceBuf,
    /// Device argmax result — see [`ARGMAX_BYTES`]. The non-finite tag rides this
    /// buffer deliberately: the tail's D2H is already paid, so localising a NaN costs
    /// no extra sync — and a sync is exactly what masks the fault it localises.
    pub(super) argmax_dev: DeviceBuf,
    // Per-layer fp8 KV latent cache, sized to max_ctx: `kv_latent` is e4m3 (max_ctx*kvl u8),
    // `kv_latent_scale` the per-128 block scales (max_ctx*n_blocks f32), `kv_rope` the roped key
    // (max_ctx*rope u16, always bf16).
    pub(super) kv_latent: Vec<DeviceBuf>,
    pub(super) kv_latent_scale: Vec<DeviceBuf>,
    pub(super) kv_rope: Vec<DeviceBuf>,
    pub(super) n_kv_blocks: usize,
    // Host routing/argmax scratch, reused every layer so the hot path allocates nothing.
    pub(super) scores: Vec<f32>,
    pub(super) choice: Vec<f32>,
    /// Per-expert routing weights for the current layer, laid out `[descriptor][row]` —
    /// the layout the down-projection kernel reads as `wexpert[e*R + t]`. A row that
    /// did not route to a union expert carries 0.0 there, and the kernel SKIPS a zero
    /// weight rather than multiplying by it, which is why the union cannot perturb a
    /// row's own result.
    pub(super) wexpert: Vec<f32>,
    /// Each token row's own top-`top_k` picks, before the union. Row 0's also feeds
    /// the trace, which stays defined on the real token.
    pub(super) sel_row: [Vec<usize>; MAXROW],
    /// Each row's normalized routed weights, parallel to `sel_row[r]`.
    pub(super) wrow: [Vec<f32>; MAXROW],
    /// The deduplicated union of every row's picks — what actually gets submitted and
    /// launched. Row 0's picks come first, so an `nrow == 1` pass submits exactly `sel`.
    pub(super) union: Vec<usize>,
    /// The three per-projection VQ codebooks, fp16, resident. Null in int4.
    pub(super) codebooks: [*const u16; 3],
    /// The run's ONE routed format — read off the pool once at build. Every expert in
    /// every batch decodes with this; there is no per-expert format anywhere (the old
    /// tree's format-follows-residency channel, deleted structurally at the pool).
    pub(super) fmt: RoutedFmt,
    /// The pool's answer for the current layer: slots + tickets, in union order.
    pub(super) resolved: ResolvedBatch,
    pub(super) descs: Vec<ExpertDesc>,
    /// Tickets for the launch loop, `descs`-parallel (the shared expert appends
    /// [`Ticket::RESIDENT`] after the pool's answer).
    pub(super) tickets: Vec<Ticket>,
    /// Gate-logits D2H staging (`nrow * n_experts` f32 as bytes).
    pub(super) gate_logits_host: Vec<u8>,
    pub(super) argmax_host: Vec<u8>,
    /// The MoE expert stream's compute stream — resident/loaded experts' partials run
    /// here concurrently with the fetch stream's loads (the overlap). Separate from the
    /// null stream the rest of the forward uses.
    pub(super) compute_stream: HipStream,
    /// Experts whose bytes are still arriving launch HERE, not on `compute_stream` —
    /// a stream is FIFO, so a wait enqueued on the compute stream is only reached after
    /// the residents finish, putting the GPU's wake latency on the critical path
    /// (measured +382 µs per layer-with-misses in the old tree).
    pub(super) miss_stream: HipStream,
}

impl<'a> GlmEngine<'a> {
    /// Build the engine over `pin`: allocate every per-token scratch buffer and the KV
    /// slabs ONCE, so the decode loop allocates nothing.
    pub fn new(pin: GlmPin<'a>, cfg: &'a ModelConfig, max_ctx: usize) -> Result<Self> {
        let layer_major_prefill = !pin.routed.tracing();
        // The MoE block folds the shared expert into the routed batch at a single
        // kernel `inter = moe_inter`. Only valid when the shared expert has the routed
        // width, i.e. n_shared == 1 (GLM-5.2).
        ensure!(
            cfg.n_shared == 1,
            "GPU decode assumes n_shared==1 (shared folded into the routed batch); \
             n_shared={}",
            cfg.n_shared
        );
        let (kvl, rope, h) = (cfg.kv_lora_rank, cfg.qk_rope_head_dim, cfg.n_heads);
        // Descriptor slots per MoE launch: the union of every row's picks plus the
        // shared expert. Rows overlap ~31% in practice (measured), so the union is
        // ~13.5 of the 16 the routed half reserves.
        let slots = cfg.top_k * MAXROW + cfg.n_shared;
        ensure!(
            kvl.is_multiple_of(E4M3_BLOCK),
            "kv_lora_rank ({kvl}) must be a multiple of {E4M3_BLOCK} (fp8 KV block size)",
        );
        // `mla_attend` holds its online accumulator in 512 registers per lane and
        // rejects a wider kvl. Check HERE: the kernel guard would not fire until the
        // first decoded token, by which point the whole resident pin is allocated.
        ensure!(
            kvl <= 512,
            "kv_lora_rank ({kvl}) exceeds 512, the attend kernel's register-resident \
             accumulator cap",
        );
        let n_kv_blocks = kvl / E4M3_BLOCK;
        let mut kv_latent = Vec::with_capacity(cfg.n_layers);
        let mut kv_latent_scale = Vec::with_capacity(cfg.n_layers);
        let mut kv_rope = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            kv_latent.push(DeviceBuf::new(max_ctx * kvl)?); // e4m3 latent (1 byte)
            kv_latent_scale.push(DeviceBuf::new(max_ctx * n_kv_blocks * 4)?); // f32 block scales
            kv_rope.push(DeviceBuf::new(max_ctx * rope * 2)?); // bf16 roped key
        }
        // Layer-major prefill keeps the WHOLE prompt's residual stream live across the
        // model, so `x` has to hold it. `max_ctx` is the honest bound — the server
        // sizes its context once and every prompt it accepts fits inside it.
        let x_rows = match layer_major_prefill {
            true => max_ctx.max(MAXROW),
            false => MAXROW,
        };
        tracing::info!(
            "residual stream: {x_rows} rows ({:.1} MB){}",
            (x_rows * cfg.hidden * 4) as f64 / 1e6,
            match layer_major_prefill {
                true => " — layer-major prefill",
                false =>
                    " — token-major prefill (--trace: layer-major mis-segments a \
                           v2 capture)",
            }
        );
        let f = |n: usize| DeviceBuf::new(n * 4); // f32 buffer of n elems
        Ok(Self {
            cfg,
            max_ctx,
            layer_major_prefill,
            x: f(x_rows * cfg.hidden)?,
            x_rows,
            xn: f(MAXROW * cfg.hidden)?,
            attn_out: f(MAXROW * cfg.hidden)?,
            q_lora: f(MAXROW * cfg.q_lora_rank)?,
            q: f(MAXROW * h * cfg.qk_head_dim())?,
            compressed_kv: f(MAXROW * (kvl + rope))?,
            q_absorbed: f(MAXROW * h * kvl)?,
            q_rope: f(MAXROW * h * rope)?,
            ctx_latent: f(MAXROW * h * kvl)?,
            attn_partial: f(MAXROW * rivoli_backend::attend_scratch_floats(h, kvl))?,
            attn_ctx: f(MAXROW * h * cfg.v_head_dim)?,
            gate_logits: f(MAXROW * cfg.n_experts)?,
            mlp_g: f(MAXROW * cfg.dense_inter)?,
            mlp_u: f(MAXROW * cfg.dense_inter)?,
            moe_out: f(MAXROW * cfg.hidden)?,
            // Zeroed HERE and nowhere else: the drain resets it as it converts, so
            // steady state needs no memset. hipMalloc does not zero, and layer 0 would
            // otherwise sum against whatever was resident.
            moe_acc: {
                let bytes = MOE_ACC_ROWS * MAXROW * cfg.hidden * 8;
                let mut b = DeviceBuf::new(bytes)?;
                // SAFETY: `b` owns `bytes`, just allocated.
                unsafe { fill_u32(b.ptr_mut(), 0, bytes)? };
                b
            },
            moe_hidden: f(slots * MAXROW * cfg.moe_inter)?,
            descs_buf: DeviceBuf::new(slots * std::mem::size_of::<ExpertDesc>())?,
            wexpert_buf: f(slots * MAXROW)?,
            logits: f(MAXROW * cfg.vocab)?,
            argmax_dev: {
                // hipMalloc does NOT zero. Tag 0 means "clean", so an unzeroed byte
                // would fabricate a layer coordinate on the first failure — the probe
                // would confidently point at the wrong place.
                let mut b = DeviceBuf::new(ARGMAX_BYTES)?;
                b.copy_in_at(0, &[0u8; ARGMAX_BYTES])?;
                b
            },
            kv_latent,
            kv_latent_scale,
            kv_rope,
            n_kv_blocks,
            scores: vec![0.0; cfg.n_experts],
            choice: vec![0.0; cfg.n_experts],
            wexpert: Vec::with_capacity(slots * MAXROW),
            sel_row: std::array::from_fn(|_| Vec::with_capacity(cfg.top_k)),
            wrow: std::array::from_fn(|_| Vec::with_capacity(cfg.top_k)),
            union: Vec::with_capacity(slots),
            codebooks: pin.codebooks(),
            fmt: pin.routed.fmt(),
            resolved: ResolvedBatch::default(),
            descs: Vec::with_capacity(slots),
            tickets: Vec::with_capacity(slots),
            gate_logits_host: Vec::with_capacity(MAXROW * cfg.n_experts * 4),
            argmax_host: Vec::with_capacity(ARGMAX_BYTES),
            compute_stream: HipStream::compute()?,
            miss_stream: HipStream::miss()?,
            pin,
        })
    }

    /// The KV ceiling this engine allocated for, in tokens — see `Engine::max_ctx`, whose
    /// only job is to hand this number to a caller that must refuse an over-long request
    /// before decoding it.
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn hits(&self) -> u64 {
        self.pin.routed.hits()
    }

    pub fn misses(&self) -> u64 {
        self.pin.routed.misses()
    }

    /// Flush the `--trace` sink (per token — `Drop` discards flush errors, and a trace
    /// is ~30 minutes of sole-tenant GPU time).
    pub fn flush_trace(&mut self) -> Result<()> {
        self.pin.routed.flush_trace()
    }
}
