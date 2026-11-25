//! The Inkling 2 engine: the device scratch the looped decoder reads — the residual stream, the
//! block-residual bank, the per-cache-layer short-convolution states, the per-pass KV cache, the
//! loop-pass snapshots, and the attention/MoE/logit scratch. `forward.rs` drives the state this
//! file allocates; `geometry.rs` owns every shape and byte count it reads.
//!
//! The loop's extra passes each carry their OWN KV cache copies (T16): the cache has
//! `num_layers + (num_passes−1)·(core_end−core_start)` layers, not `num_layers`. The sconv state is
//! per cache layer for the same reason — two passes must not shift one another's windows. The block
//! bank and the residual stream restart from the pre-loop snapshot on each extra pass (T8).
//!
//! **The scratch's size is [`ScratchSizes`]', not a second expression.** The pin's floor charges
//! for what this file allocates, and [`InklingScratch::new`] allocates exactly the fields
//! `ScratchSizes` counts — one author, so a buffer added here cannot be a floor the partition
//! never saw.

use super::geometry::ScratchSizes;
use super::pin::InklingPin;
use crate::device::DeviceBuf;
use crate::resident::PinCfg;
use crate::routed::ResolvedBatch;
use anyhow::{Result, ensure};
use rivoli_artifact::inkling_config::InklingTextConfig;
use rivoli_backend::{ExpertDescF4, fill_u32, gpustream::HipStream};

/// One accumulator row per STREAM, not per expert: every expert a stream runs adds into its
/// stream's row, and [`rivoli_backend::launch_moe_acc_drain_to`] sums the rows once.
pub(super) const MOE_ACC_ROWS: usize = 2;

/// The Inkling decode path's activation scratch and caches. Weights live in the pin.
pub struct InklingScratch {
    /// The residual stream for one position (`hidden` f32).
    pub x: DeviceBuf,
    /// The normed hidden feeding a branch (`hidden` f32).
    pub xn: DeviceBuf,
    /// The full layer's separate k/v input norms (`hidden` f32 each); sliding layers reuse `xn`.
    pub xn_k: DeviceBuf,
    pub xn_v: DeviceBuf,
    /// The block-residual bank: one `hidden` vector per block boundary, plus the partial.
    pub block_bank: DeviceBuf,
    /// The four short-convolution states (k, v, post-attn, post-mlp) of EACH cache layer, each
    /// `channels·(k−1)` f32. Indexed by cache layer: an extra loop pass must not continue the
    /// window a previous pass left behind, and the layers of one pass must not share a window.
    pub sconv: Vec<[DeviceBuf; 4]>,
    /// Per-cache-layer K and V, `kv_heads·head_dim·ctx` f32 EACH. The extra loop passes append
    /// their own copies (T16), so this is longer than `num_hidden_layers`.
    pub kc: Vec<DeviceBuf>,
    pub vc: Vec<DeviceBuf>,
    /// The residual stream as it stood when the pre-loop prefix ended, and the bank beside it —
    /// every extra pass restarts from these (T8).
    pub pre_loop_x: DeviceBuf,
    pub pre_loop_bank: DeviceBuf,
    /// The PREVIOUS pass's residual and bank, which the looped layers' MessageGates read
    /// (LOOP:537, T8). Pass 1 has no predecessor; the caller seeds these from the live state.
    pub prev_pass_x: DeviceBuf,
    pub prev_pass_bank: DeviceBuf,
    /// The query projection (`num_heads·head_dim` f32).
    pub q: DeviceBuf,
    /// The attention output (`num_heads·head_dim` f32).
    pub attn: DeviceBuf,
    /// The projected k/v for one token (`kv_heads·head_dim` f32 each), post-sconv, pre-cache.
    pub k: DeviceBuf,
    pub v: DeviceBuf,
    /// The rel-bias path: `rel_states` (`heads·d_rel`), `rel_logits` (`heads·max(rel_extent,
    /// sliding_window)` — the two layer kinds take the extent from different config fields, T17),
    /// and the distance-gathered `bias` (`heads·n_ctx`).
    pub rel_states: DeviceBuf,
    pub rel_logits: DeviceBuf,
    pub bias: DeviceBuf,
    /// The per-head sink logit (`heads`) — read ON-DEVICE by `sink_attend` — and the sigmoid
    /// output gate (`heads·head_dim`).
    pub sink: DeviceBuf,
    pub gate: DeviceBuf,
    /// The logits (`unpadded_vocab_size` f32 — the live rows, T14).
    pub logits: DeviceBuf,
    /// The block-residual attend's source stack: `bank_slots·hidden` f32, the bank rows followed
    /// by the partial row, normed in place by `rmsnorm_weightless_batch`.
    pub stacked: DeviceBuf,
    /// The block-attention output: 1, 3, or 4 query rows of `hidden` f32.
    pub res_out: DeviceBuf,
    /// The block-attention bias (`4·bank_slots` f32, ZEROED — the block attend has no mask and no
    /// relative bias, MOD:833–858) and its sink (`4` f32 at `−∞`, so the sink column is dropped
    /// exactly as the token attend's is).
    pub res_bias: DeviceBuf,
    pub res_sink: DeviceBuf,
    /// The four messages a looped layer's MessageGates read, `[q, k, v, mlp]` (T13).
    pub msg_buf: DeviceBuf,
    /// One dynamic short-convolution's per-KV-head taps, `kv_heads·k` f32.
    pub dyn_w: DeviceBuf,
    /// The branch: the post-attention/post-MLP convolution's output before the residual add.
    pub branch: DeviceBuf,
    /// The MessageGate's scratch: the two normed inputs (`2·hidden`), the rank activations
    /// (`gate_rank`, at least one element), and the forget/gain logits (`hidden` each).
    pub gate_cat: DeviceBuf,
    pub gate_low: DeviceBuf,
    pub gate_f: DeviceBuf,
    pub gate_g: DeviceBuf,
    /// The gate outputs before their input norm — the sliding layer norms all three with ONE
    /// shared norm (T11) and the full layer with three distinct norms, both reading these.
    pub xn_q_tmp: DeviceBuf,
    pub xn_k_tmp: DeviceBuf,
    pub xn_v_tmp: DeviceBuf,
    /// The router logits (`n_routed_experts` f32), read to the host for `route_into`.
    pub router_logits: DeviceBuf,
    /// The routed MLP's per-expert scratch: the `[gate ‖ up]` projection (`2·inter`), the SiTU-GLU
    /// output (`inter`), one expert's down projection and the weighted accumulator (`hidden`).
    pub moe_gu: DeviceBuf,
    pub moe_h: DeviceBuf,
    pub moe_out: DeviceBuf,
    pub moe_acc: DeviceBuf,
    /// `argmax`'s two outputs, one i32 then one f32.
    pub pick: DeviceBuf,
    /// Each sconv buffer's byte length, kept beside the buffers so [`Self::zero_sconv`] zeroes the
    /// whole state without re-deriving the geometry it was allocated from.
    sconv_len: [usize; 4],
}

/// `n_cache` KV layers: K and V each one tensor's worth — one buffer per tensor, not the K+V
/// layer total twice.
fn kv_caches(n_cache: usize, tensor_bytes: usize) -> Result<(Vec<DeviceBuf>, Vec<DeviceBuf>)> {
    let mut kc = Vec::with_capacity(n_cache);
    let mut vc = Vec::with_capacity(n_cache);
    for _ in 0..n_cache {
        kc.push(DeviceBuf::new(tensor_bytes)?);
        vc.push(DeviceBuf::new(tensor_bytes)?);
    }
    Ok((kc, vc))
}

/// One `[k, v, post-attn, post-mlp]` sconv state set per cache layer: the k/v sides are
/// `kv_heads·head_dim` channels wide, the post convolutions `hidden`.
fn sconv_states(
    n_cache: usize,
    (kv_bytes, h_bytes): (usize, usize),
) -> Result<Vec<[DeviceBuf; 4]>> {
    let mut states = Vec::with_capacity(n_cache);
    for _ in 0..n_cache {
        states.push([
            DeviceBuf::new(kv_bytes)?,
            DeviceBuf::new(kv_bytes)?,
            DeviceBuf::new(h_bytes)?,
            DeviceBuf::new(h_bytes)?,
        ]);
    }
    Ok(states)
}

/// The shapes the scratch and the sink-attend kernel need. `pub(super)` so
/// [`InklingEngine::open`] runs it BEFORE the pin places a byte: a `--ctx` past the kernel's LDS
/// ceiling is unservable, and the refusal must not cost a multi-gigabyte placement first.
pub(super) fn validate_scratch_cfg(text: &InklingTextConfig, n_ctx: usize) -> Result<()> {
    ensure!(n_ctx > 0, "n_ctx is 0 — no positions to decode");
    // The kernel's own cap, not a second number: one attention call's `n` is at most `n_ctx`
    // (a full layer attends every position), and `sink_attend` refuses past it (1001).
    ensure!(
        n_ctx <= rivoli_backend::abi::SINK_ATTEND_MAX_N as usize,
        "n_ctx {n_ctx} exceeds sink_attend LDS ceiling {} — lower --ctx",
        rivoli_backend::abi::SINK_ATTEND_MAX_N
    );
    ensure!(
        text.num_key_value_heads > 0 && text.head_dim > 0 && text.hidden_size > 0,
        "the config carries a zero attention dimension"
    );
    ensure!(
        text.n_layers_per_attn_res_block > 0,
        "n_layers_per_attn_res_block is 0"
    );
    ensure!(
        text.num_attention_heads
            .is_multiple_of(text.num_key_value_heads),
        "{} query heads do not group onto {} KV heads",
        text.num_attention_heads,
        text.num_key_value_heads
    );
    Ok(())
}

fn init_res_const(bank_slots: usize) -> Result<(DeviceBuf, DeviceBuf)> {
    let mut res_bias = DeviceBuf::new(4 * bank_slots * 4)?;
    // The launcher's own zero: one async memset on the null stream, no host staging vector.
    // SAFETY: `res_bias` owns exactly the length passed — the same geometry it was allocated
    // from — and every reader is enqueued on the null stream after this fill.
    unsafe { rivoli_backend::fill_u32(res_bias.ptr_mut(), 0, 4 * bank_slots * 4)? };
    let mut res_sink = DeviceBuf::new(4 * 4)?;
    res_sink.copy_in_at(0, crate::device::as_le_bytes(&[f32::NEG_INFINITY; 4]))?;
    Ok((res_bias, res_sink))
}

impl InklingScratch {
    /// Allocate the scratch for `n_ctx` positions. `n_ctx` is the caller's prompt plus intended
    /// generation, not `model_max_length` (1048576) — a full layer's cache is linear in it.
    pub fn new(text: &InklingTextConfig, n_ctx: usize) -> Result<Self> {
        validate_scratch_cfg(text, n_ctx)?;
        let s = ScratchSizes::of(text, n_ctx);
        let (kc, vc) = kv_caches(s.n_cache, s.kv_tensor)?;
        let sconv = sconv_states(s.n_cache, (s.sconv_kv, s.sconv_h))?;
        let (res_bias, res_sink) = init_res_const(s.bank_slots)?;
        Ok(Self {
            x: DeviceBuf::new(s.residual)?,
            xn: DeviceBuf::new(s.residual)?,
            xn_k: DeviceBuf::new(s.residual)?,
            xn_v: DeviceBuf::new(s.residual)?,
            block_bank: DeviceBuf::new(s.bank)?,
            sconv,
            kc,
            vc,
            pre_loop_x: DeviceBuf::new(s.residual)?,
            pre_loop_bank: DeviceBuf::new(s.bank)?,
            prev_pass_x: DeviceBuf::new(s.residual)?,
            prev_pass_bank: DeviceBuf::new(s.bank)?,
            q: DeviceBuf::new(s.qd)?,
            attn: DeviceBuf::new(s.qd)?,
            k: DeviceBuf::new(s.kv_vec)?,
            v: DeviceBuf::new(s.kv_vec)?,
            rel_states: DeviceBuf::new(s.rel_states)?,
            rel_logits: DeviceBuf::new(s.rel_logits)?,
            bias: DeviceBuf::new(s.bias)?,
            sink: DeviceBuf::new(s.sink)?,
            gate: DeviceBuf::new(s.qd)?,
            logits: DeviceBuf::new(s.logits)?,
            stacked: DeviceBuf::new(s.bank)?,
            res_out: DeviceBuf::new(4 * s.residual)?,
            res_bias,
            res_sink,
            msg_buf: DeviceBuf::new(4 * s.residual)?,
            dyn_w: DeviceBuf::new(s.dyn_w)?,
            branch: DeviceBuf::new(s.residual)?,
            gate_cat: DeviceBuf::new(2 * s.residual)?,
            gate_low: DeviceBuf::new(s.gate_low)?,
            gate_f: DeviceBuf::new(s.residual)?,
            gate_g: DeviceBuf::new(s.residual)?,
            xn_q_tmp: DeviceBuf::new(s.residual)?,
            xn_k_tmp: DeviceBuf::new(s.residual)?,
            xn_v_tmp: DeviceBuf::new(s.residual)?,
            router_logits: DeviceBuf::new(s.router)?,
            moe_gu: DeviceBuf::new(s.moe_gu)?,
            moe_h: DeviceBuf::new(s.moe_h)?,
            moe_out: DeviceBuf::new(s.residual)?,
            moe_acc: DeviceBuf::new(s.residual)?,
            pick: DeviceBuf::new(8)?,
            sconv_len: [s.sconv_kv, s.sconv_kv, s.sconv_h, s.sconv_h],
        })
    }

    /// Zero every short-convolution state on the device.
    ///
    /// **Called at the start of every `decode`/`score`, because the state is a REQUEST's, not the
    /// engine's.** A second prompt that continued the first one's windows would let tokens the
    /// caller never wrote influence the text — the sconv analogue of leaving a KV cache warm, and
    /// the loop's per-cache-layer split (T16) does not help: a new request reuses the same 48
    /// cache layers.
    ///
    /// `fill_u32` is the launcher's own zero (one async memset per state, no host staging); the
    /// buffers are `k−1` taps of f32, so every length is a multiple of 4.
    pub fn zero_sconv(&mut self) -> Result<()> {
        for state in &mut self.sconv {
            for (buf, &len) in state.iter_mut().zip(self.sconv_len.iter()) {
                // SAFETY: `buf` owns exactly `len` bytes (allocated from the same geometry), and
                // the fill is enqueued on the null stream — ordered before every later launch.
                unsafe { rivoli_backend::fill_u32(buf.ptr_mut(), 0, len)? };
            }
        }
        Ok(())
    }
}

/// Inkling 2 Small's routed-MoE dispatch state — allocated ONLY on the `.mxfp8` backend, so
/// Mini pays neither the bytes nor an entry in any accounting.
///
/// **Sized for ONE row's `top_k` picks, not for `n_experts`.** The launcher indexes `descs`,
/// `wexpert` and `h` by DESCRIPTOR index in `0..n_desc`, and this arm passes `n_desc = top_k`;
/// sizing them for 384 experts would be 48× the bytes for rows no launch can read. The pool's
/// `top_k` bound is checked at startup by [`Batch::union`](crate::resident::Batch::union).
pub(super) struct MxFp8MoeScratch {
    /// Resident experts' partials run here, overlapping the fetch stream's loads; misses
    /// launch on their own stream behind their tickets.
    pub compute_stream: HipStream,
    pub miss_stream: HipStream,
    /// The pool's answer for the current layer: slots + tickets, in selection order.
    pub resolved: ResolvedBatch,
    /// Selection indices in LAUNCH order — residents first, misses after — so the residents
    /// form the one contiguous range the launcher's `[e_start, e_count)` form exists for.
    pub launch_idx: Vec<usize>,
    /// The descriptor table and the routing weights, in launch order.
    pub descs_host: Vec<ExpertDescF4>,
    pub wexpert_launch: Vec<f32>,
    pub descs: DeviceBuf,
    pub wexpert: DeviceBuf,
    /// One expert's SiTU-GLU output, `[n_desc, inter]` f32 — indexed by descriptor.
    pub moe_h: DeviceBuf,
    /// [`MOE_ACC_ROWS`] rows of the hidden width, u64 fixed point.
    pub acc: DeviceBuf,
}

impl MxFp8MoeScratch {
    fn new(text: &InklingTextConfig) -> Result<Self> {
        let k = text.num_experts_per_tok;
        let (hidden, inter) = (text.hidden_size, text.intermediate_size);
        let acc_bytes = MOE_ACC_ROWS * hidden * size_of::<u64>();
        let mut acc = DeviceBuf::new(acc_bytes)?;
        // The drain RESETS the rows it drains, but the first layer reads rows nothing has ever
        // written — an `hipMalloc` slab is uninitialised, and a garbage accumulator is a
        // plausible wrong number rather than a fault.
        // SAFETY: `acc` owns exactly `acc_bytes` (a multiple of 4), and every reader is
        // enqueued after this async memset on the null stream.
        unsafe { fill_u32(acc.ptr_mut(), 0, acc_bytes)? };
        Ok(Self {
            compute_stream: HipStream::compute()?,
            miss_stream: HipStream::miss()?,
            resolved: ResolvedBatch::default(),
            launch_idx: Vec::with_capacity(k),
            descs_host: vec![ExpertDescF4::null(); k],
            wexpert_launch: vec![0.0; k],
            descs: DeviceBuf::new(k * size_of::<ExpertDescF4>())?,
            wexpert: DeviceBuf::new(k * size_of::<f32>())?,
            moe_h: DeviceBuf::new(k * inter * size_of::<f32>())?,
            acc,
        })
    }
}

/// The Inkling 2 text engine: the placed weights, the activation scratch, and the loop's host-side
/// routing workspace.
pub struct InklingEngine<'a> {
    pub(super) pin: InklingPin,
    pub(super) cfg: &'a InklingTextConfig,
    pub(super) sc: InklingScratch,
    pub(super) n_ctx: usize,
    /// Whether [`Self::sample_x`](super::forward) has ever run, so the logit accessor cannot hand
    /// back an uninitialised buffer.
    pub(super) sampled: bool,
    /// Decode-thread phase spans, stamped at this arm's own sync points.
    pub(super) prof: crate::telemetry::Phases,
    /// The router's per-token workspace (`route_into`'s three borrowed buffers), sized
    /// `n_routed_experts` once so routing allocates nothing per token.
    pub(super) route_scores: Vec<f32>,
    pub(super) route_choice: Vec<f32>,
    pub(super) route_sel: Vec<usize>,
    /// The router logits, read back as raw little-endian f32 — `route_into` takes bytes so the
    /// host copy needs no reinterpret step of its own.
    pub(super) router_host: Vec<u8>,
    /// The logits row, read back for teacher forcing.
    pub(super) row_host: Vec<u8>,
    /// `argmax`'s two outputs, read back together.
    pub(super) pick_host: Vec<u8>,
    /// Small's routed dispatch state and streams; `None` on Mini's bf16 backend, so the loop
    /// Mini runs is exactly what it ran before this field existed.
    pub(super) moe8: Option<MxFp8MoeScratch>,
}

impl<'a> InklingEngine<'a> {
    /// Open `dir` as an Inkling engine: let [`InklingPin::build`] place the weights against a floor
    /// derived from `(cfg, n_ctx)`, then allocate what that floor charged for.
    ///
    /// **`n_ctx` reaches the pin rather than two byte counts computed here.** Both sides derive the
    /// KV and scratch charges from `geometry`'s one [`ScratchSizes`], so the bytes the partition
    /// was refused or granted against are the bytes that get allocated.
    ///
    /// **The config is refused at the DOOR**, before [`InklingPin::build`] places a byte: an
    /// unservable `--ctx` (past `sink_attend`'s LDS ceiling) or a zero attention dimension must
    /// cost a message, not a placement that then has to be undone.
    pub fn open(
        dir: &str,
        cfg: &'a InklingTextConfig,
        capacity: usize,
        n_ctx: usize,
    ) -> Result<Self> {
        Self::open_with(cfg, n_ctx, || InklingPin::build(dir, cfg, capacity, n_ctx))
    }

    /// Open Inkling 2 Small: the same text stack and the same floor discipline as [`Self::open`],
    /// with the routed experts behind [`crate::routed::RoutedPool`].
    ///
    /// The door check runs first, for [`Self::open`]'s reason: a `--ctx` past the attend
    /// kernel's ceiling is unservable, and it must cost a message rather than a placement.
    pub fn open_mxfp8(
        dir: &str,
        cfg: &'a InklingTextConfig,
        pin: PinCfg<'_>,
        n_ctx: usize,
    ) -> Result<Self> {
        Self::open_with(cfg, n_ctx, || InklingPin::build_mxfp8(dir, cfg, pin, n_ctx))
    }

    /// Shared door-first open: validate the scratch ceiling BEFORE placing a byte, then build the
    /// pin and allocate scratch.
    fn open_with(
        cfg: &'a InklingTextConfig,
        n_ctx: usize,
        build: impl FnOnce() -> Result<InklingPin>,
    ) -> Result<Self> {
        validate_scratch_cfg(cfg, n_ctx)?;
        Self::new(build()?, cfg, n_ctx)
    }

    /// Build the engine over an already-placed `pin`.
    pub fn new(pin: InklingPin, cfg: &'a InklingTextConfig, n_ctx: usize) -> Result<Self> {
        let n_experts = cfg.n_routed_experts;
        // The pool's streams and descriptor table exist only where the pool does. `is_mxfp8`
        // is read off the pin the caller just built, so there is no second source of truth.
        let moe8 = match pin.is_mxfp8() {
            true => Some(MxFp8MoeScratch::new(cfg)?),
            false => None,
        };
        Ok(Self {
            pin,
            cfg,
            sc: InklingScratch::new(cfg, n_ctx)?,
            n_ctx,
            sampled: false,
            prof: crate::telemetry::Phases::default(),
            route_scores: vec![0.0; n_experts],
            route_choice: vec![0.0; n_experts],
            route_sel: Vec::with_capacity(cfg.num_experts_per_tok),
            router_host: vec![0u8; n_experts * 4],
            row_host: vec![0u8; cfg.unpadded_vocab_size * 4],
            pick_host: vec![0u8; 8],
            moe8,
        })
    }

    /// The KV ceiling this engine allocated for, in tokens.
    pub fn max_ctx(&self) -> usize {
        self.n_ctx
    }

    /// How many units the budget pinned, and how many stream — the pin's own answer, never a
    /// second count of the same placement. The granularity is the BACKEND's: whole layers on
    /// Mini, single experts on Small, so the two are not comparable across members.
    pub fn residency(&self) -> (usize, usize) {
        self.pin.residency()
    }

    /// Expert slot hits and fills — this arm's streaming granularity is ONE EXPERT (the routed
    /// unit), so these are not comparable with a whole-layer arm's counters.
    pub fn fetched(&self) -> crate::seam::Fetched {
        self.pin.fetched()
    }

    /// Flush the `--trace` sink when this pin owns a pool (Small). Mini's row refuses
    /// `--trace` and its pin carries no pool, so this is one branch and no work there.
    ///
    /// **Per token, never on `Drop`**: `BufWriter`'s `Drop` discards flush errors, so a wedged
    /// or ENOSPC sink would leave a silently short capture with a clean exit code — and a trace
    /// is tens of minutes of sole-tenant GPU time (`RoutedPool::flush_trace` carries the
    /// argument, and GLM/K3/V4 all call it the same way).
    pub(super) fn flush_trace(&mut self) -> Result<()> {
        match self.pin.routed_mut() {
            Some(routed) => routed.flush_trace(),
            None => Ok(()),
        }
    }

    /// Zero every per-request state (short-convolution windows and MXFP8 accumulator rows) before
    /// a request's first position so a failed earlier request cannot leave stale partials behind.
    pub(super) fn reset_state(&mut self) -> Result<()> {
        self.sampled = false;
        if let Some(m) = &mut self.moe8 {
            let acc_bytes = MOE_ACC_ROWS * self.cfg.hidden_size * size_of::<u64>();
            // SAFETY: `m.acc` was allocated with `acc_bytes` (multiple of 4), zeroed on the null
            // stream before any MoE dispatch runs.
            unsafe { fill_u32(m.acc.ptr_mut(), 0, acc_bytes)? };
        }
        self.sc.zero_sconv()
    }
}
