//! The GPU decode loop — the resident forward pass. Every per-token op runs
//! on-device against the [`Pin`]'s resident weights, using scratch [`DeviceBuf`]s
//! allocated once and reused each token (no per-token allocation). The only host
//! round-trips are the router-gate logits (MoE layers) and the final logits for
//! argmax — each a small D2H behind a join.
//!
//! Dense, streaming, DSA and MISA attention (the DSA/MISA row selection is
//! [`GpuEngine::dsa_select_layer`]), fp8-e4m3 KV latent cache, VQ-int3 routed + shared
//! experts.
//!
//! Every device call goes through [`crate::backend`], so this file is backend-independent:
//! it compiles under `rocm` and under `vulkan`. What is NOT equal across that seam —
//! single-queue serialisation, zero-valued GPU timing spans, and the DSA/int4 kernels that
//! refuse on Vulkan — is enumerated in `backend.rs`'s header. Needs a backend either way;
//! without a device there is nothing to decode on.
#![cfg(any(feature = "rocm", feature = "vulkan"))]

use crate::attn::{AttnMode, streaming_rows};
use crate::backend::{
    Event, ExpertDesc, Signal, Stream, device_sync, launch_append_kv, launch_argmax,
    launch_attend, launch_embed_i8_row, launch_flag_nonfinite, launch_gather_rope,
    launch_gemv_f32, launch_gemv_fp8, launch_gemv_i8,
    launch_index_append, launch_index_head_route, launch_index_pool_push, launch_index_score,
    launch_index_topk, launch_layernorm, launch_mla_absorb_fp8, launch_mla_value_fp8,
    launch_moe_expert_range, launch_moe_expert_range_i4, launch_moe_reduce, launch_rmsnorm,
    launch_rope, launch_swiglu, launch_vadd, launch_vaxpy, stream_signal,
};
use crate::device::DeviceBuf;
use crate::math::{E4M3_BLOCK, nll_of, route_into, topk_into};
use crate::model::ModelConfig;
use crate::pin::{Fp8Mlp, IndexerPin, LayerMlp, MlpVq, Pin, TRACE_WINDOW};

/// How many layers ahead the pilot predicts. `trace` builds evaluate both so LOOKA can
/// report the L+2 curve; `--pilot` alone only consumes L+1's rank 0, so it pays for one
/// rmsnorm+gemv instead of two.
const PILOT_H: [usize; 2] = [1, 2];
#[cfg(feature = "trace")]
const PILOT_HORIZONS: usize = 2;
#[cfg(not(feature = "trace"))]
const PILOT_HORIZONS: usize = 1;
// The stash in `looka` is indexed by position in ITS horizon array; if the two drift the
// recall counters silently attribute L+2 hits to L+1.
#[cfg(feature = "trace")]
const _: () = assert!(
    PILOT_H[0] == crate::looka::HORIZONS[0] && PILOT_H[1] == crate::looka::HORIZONS[1],
    "gpu::PILOT_H and looka::HORIZONS must agree"
);
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

// THE DSA ROW-SELECTION PATH is the DEVICE `index_topk` kernel, unconditionally.
//
// There used to be a `RIVOLI_TOPK=host|device|device-nosync|verify` switch here, four arms
// of one binary so the device top-k and a mid-layer-sync deletion could be costed
// separately. Both were costed (benchmarks.md, "Device top-k WIRED"): `host → device` is
// **−9.4 ms/token**, `device → device-nosync` is **−2.5 ms/token** — and the second was
// deliberately NOT taken, because 0.6% of wall is not worth making `route` incomparable
// with every historical row in benchmarks.md. The arms are deleted now that the answers are
// recorded: `host` was a baseline git already holds, `device-nosync` a rejected option, and
// `verify` a correctness gate that `tests/kernel.rs::index_topk_matches_host_selection`
// covers with the same over-selection sentinel trick, on data the test controls rather than
// on whatever a run happens to produce.

/// Per-token time buckets. ALWAYS ON: every bucket wraps a join/D2H the forward
/// pass already pays (the end-of-layer sync, the gate-logits read, the stream
/// drain), so accumulating them costs only a clock read per layer — no extra GPU
/// sync. Cost bound, since `wall_ns` is quoted as a measurement elsewhere: the
/// indexer buckets add 42 `hipEventRecord` enqueues + 21 clock reads per token,
/// O(0.2 ms) against a ~400 ms token, ~0.05%. Bounded by argument, not by an
/// un-instrumented control run — see docs/NPU.md "What was NOT measured". The end-of-run [`Profile::report`] is the
/// engine's standing performance summary; the expensive fine-grained audits and
/// correctness probes live behind the `trace` feature instead.
#[derive(Default)]
struct Profile {
    fetch_n: u64, // demand misses
    /// Speculative reads issued (`--pilot`). Kept SEPARATE from `fetch_n` because they are
    /// not misses — a hit-rate that counted them would flatter itself — but they are real
    /// bytes off the same device, so `gb_per_tok` and `ms_per_miss` are computed over the
    /// sum. Omitting them understated the pilot's first run by 0.43 GB/token.
    spec_n: u64,
    /// `top-m` only: chosen slots that were NOT in the true top-K. Stays 0 under every
    /// other policy, which is why the summary reports it as an Option rather than a 0%.
    swap_n: u64,
    route_ns: u128, // host routing (gate D2H + sigmoid/bias/top-k, + top-m substitution)
    /// The DSA indexer's HIP-event SPAN — including whatever falls between its kernels,
    /// so NOT comparable to a per-kernel microbench sum. Measured 27% above one, cause
    /// unestablished. Note the endpoints are themselves barrier packets whose dispatch
    /// cost lands inside the span. It covers `index_topk` too — deliberately, so the
    /// price of selecting on device is booked rather than hidden.
    idx_gpu_ns: u128,
    /// Full layers that scored, the denominator for both. Not `tokens * 21` — layers below
    /// `index_topk` return dense before scoring and record nothing.
    idx_layers: u64,
    moe_wall_ns: u128,    // the block_on wall of the overlapped MoE phase (CPU wall)
    compute_gpu_ns: u128, // HIP-event span of the compute stream (partials + reduce)
    /// Per-layer MoE bracket time, bucketed by how many of that layer's experts MISSED.
    ///
    /// `compute_gpu` is a bracket, not a sum: the compute stream idles inside it whenever
    /// the next partial is still waiting on bytes. That makes the aggregate unable to say
    /// WHY it is 2.2x the isolated-kernel floor. Bucketing by miss count answers it
    /// directly — if the gaps are fetch waits, bracket time rises with misses and the
    /// zero-miss bucket is pure kernel time. A regression over these buckets separates
    /// "the shaders are slow" from "the stream is starved" with no extra sync: the events
    /// are already read at the end-of-layer join, and this is two adds per layer.
    moe_ns_by_miss: [u128; 16],
    moe_n_by_miss: [u32; 16],
    wall_ns: u128,
    tokens: u64,

    // ---- CLASS spans: what the machine was DOING, all directly measured ----
    // The phase buckets above (route / moe / tail) are REGIONS, and each mixes host
    // compute with blocking waits — which is why `tail` spent so long with most of
    // itself attributable to no kernel. These cut the same work by activity instead.
    //
    // **They are measured spans, not a partition, and they may overlap and sum to MORE
    // than wall.** That is deliberate: `io_wait` runs on the reaper thread concurrently
    // with everything here, so forcing it into a share of wall is what made the earlier
    // version derive it as `moe_wall − compute_gpu` — a host clock minus a GPU clock,
    // which understated it. The cost of dropping the partition is that **no residual is
    // reported**: unattributed time is simply not shown rather than being swept into a
    // bucket that then looks like a measurement. An earlier `cpu` was exactly such a
    // residual, and it was worthless — it absorbed every error in the other two.
    /// Blocked in the gate-logits D2H (a subset of `route_ns`, which also holds the
    /// host `route_into` work). Splitting it answers whether `route` is attention the
    /// GPU is still running or routing the host is doing.
    route_wait_ns: u128,
    /// Blocked in a `device_sync()` — the mid-layer and end-of-layer joins.
    sync_wait_ns: u128,
    /// Blocked in the argmax D2H. This one call drains the final rmsnorm, lm_head AND
    /// argmax, so before this field the whole tail phase was a single opaque wait.
    tail_wait_ns: u128,

    // ---- CPU: three stamped regions, replacing what used to be a residual ----
    /// Host time issuing kernel launches — the per-layer attention/projection block and
    /// the tail's rmsnorm/lm_head/argmax. No blocking call is inside either, so this is
    /// host work by construction. Expected to dominate host cost: ~20 launches × 78
    /// layers is ~1.5k driver calls per token.
    cpu_launch_ns: u128,
    /// Host time in `route_into` — sigmoid, bias, top-k over 256 experts per MoE layer,
    /// plus any `top-m` substitution. Stamped directly rather than taken as
    /// `route_ns − route_wait_ns` so it survives someone adding a third thing to the
    /// route region.
    cpu_route_ns: u128,
    /// Host time in `Pin::submit_layer` — residency lookups, policy/eviction
    /// bookkeeping, slot assignment and read-spec construction for the layer's picks.
    cpu_submit_ns: u128,

    // ---- GPU-timeline span. OVERLAPS the classes above; NOT part of the partition ----
    /// HIP-event span across final rmsnorm → lm_head → argmax.
    ///
    /// **It is a bracket, not a sum, and the first version of this comment claimed
    /// otherwise.** The three kernels launch back-to-back so the gaps are small, but they
    /// are not zero: measured 5.50 ms against a microbench sum of 4.66 (lm_head 4.56 +
    /// argmax 0.089 + rmsnorm 0.008), i.e. **~0.84 ms — 15% — is inter-kernel gap**, the
    /// same caveat `idx_gpu_ns` carries. So this is an upper bound on the tail's GPU
    /// execution, and `tail_wait − tail_gpu` is a LOWER bound on sync overhead, not the
    /// whole of it.
    tail_gpu_ns: u128,
}

/// Time a blocking call into one of [`Profile`]'s class buckets. Five lines instead of
/// an RAII guard because the call sites are few and explicit beats clever here — but it
/// should be the ONLY way the decode path blocks: an unwrapped join silently lands in
/// the derived `cpu` bucket and looks like host compute.
fn blocked<T>(acc: &mut u128, name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let t = std::time::Instant::now();
    let r = f();
    let e = std::time::Instant::now();
    *acc += e.duration_since(t).as_nanos();
    // Also emit the interval, so the same wait that feeds the scalar counter can be
    // drawn on a timeline next to the reaper's io-wait. No-op unless RIVOLI_SPANS is set.
    crate::telemetry::spans::record(name, "decode", t, e);
    r
}

impl Profile {
    /// Fold the accumulated buckets into the per-token summary (also fed to OTLP).
    /// `fetch_wall_ns` is the reaper's off-thread load cost; `idle_ns`/`poll_ns` are
    /// the expert stream's tokio-metrics (load-wait / launch) — the accurate
    /// decomposition the async overlap hides from wall-clock.
    #[allow(clippy::too_many_arguments)] // one call site; the buckets are unrelated scalars
    fn summary(
        &self,
        hits: u64,
        misses: u64,
        bytes_per_expert: usize,
        fetch_wall_ns: u64,
        io_wait_ns: u64,
        idle_ns: u64,
        poll_ns: u64,
        advice: Option<(usize, usize)>,
    ) -> ProfileSummary {
        let tok = self.tokens.max(1) as f64;
        let per = |ns: u128| ns as f64 / 1e6 / tok; // ms/token
        // Exposed fetch = the MoE wall in excess of the pure compute-stream span
        // (moe_wall − compute_gpu): the fetch that could NOT hide behind compute. The
        // rest of the reaper's fetch_wall overlapped. (tokio idle over-counts — it
        // sums per-expert waits across the ~9 concurrent tasks — so it's reported raw,
        // not used here.)
        let exposed_ns = self.moe_wall_ns.saturating_sub(self.compute_gpu_ns) as f64;
        // GPU-wait: the decode thread parked in a device join. Every term is a stamped
        // `Instant` span except the MoE phase's, which is its own host wall net of the
        // exposed fetch and the tokio poll (host work inside the same block).
        //
        // Not a share of wall and not trying to be. An earlier version forced these into
        // a partition with `cpu` as the leftover; the leftover absorbed every error in
        // the other terms and measured nothing. `cpu` below is now three stamped regions
        // instead, and the cost of that honesty is that unattributed time is simply not
        // reported.
        let moe_gpu_wait_ns = (self.moe_wall_ns as f64 - exposed_ns - poll_ns as f64).max(0.0);
        let gpu_wait_ns = (self.route_wait_ns + self.sync_wait_ns + self.tail_wait_ns) as f64
            + moe_gpu_wait_ns;
        // CPU: measured host-compute regions. `poll_ns` is the expert stream's tokio
        // poll — host work inside the MoE block, which the three decode-thread stamps
        // cannot see — so it belongs here rather than being double-counted as a wait.
        let cpu_ns = (self.cpu_launch_ns + self.cpu_route_ns + self.cpu_submit_ns) as f64
            + poll_ns as f64;
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
            idx_gpu_ms: per(self.idx_gpu_ns),
            idx_layers_per_tok: self.idx_layers as f64 / tok,
            moe_wall_ms: per(self.moe_wall_ns),
            compute_gpu_ms: per(self.compute_gpu_ns),
            // Mean MoE bracket per layer, by miss count. The shape is the finding: a flat
            // profile means the gaps are not fetch waits, a rising one means they are.
            moe_us_by_miss: std::array::from_fn(|i| {
                let n = self.moe_n_by_miss[i];
                if n == 0 {
                    None
                } else {
                    Some((self.moe_ns_by_miss[i] as f64 / n as f64 / 1e3, n))
                }
            }),
            fetch_wall_ms: per(fetch_wall_ns as u128),
            load_wait_ms: per(idle_ns as u128),
            launch_ms: per(poll_ns as u128),
            fetch_hidden_pct: hidden,
            miss_per_tok: self.fetch_n as f64 / tok,
            // Over ALL reads the reaper serviced, demand and speculative alike —
            // `fetch_wall_ns` covers both, so `fetch_n` alone is the wrong denominator.
            ms_per_miss: fetch_wall_ns as f64 / 1e6 / (self.fetch_n + self.spec_n).max(1) as f64,
            gb_per_tok: (self.fetch_n + self.spec_n) as f64 / tok * bytes_per_expert as f64
                / 1e9,
            // CLASS partition of the same wall. `gpu_wait` collects the explicitly
            // wrapped joins PLUS `compute_gpu`: during the MoE compute-stream span the
            // host is parked in `stream_signal(...).await`, which is a GPU wait even
            // though no `device_sync` appears there. `io_wait` is the exposed fetch —
            // the part of the reaper's work that could NOT hide behind compute — and it
            // is reused rather than re-stamped, because `moe_wall − compute_gpu` already
            // measures exactly that.
            gpu_wait_ms: gpu_wait_ns / 1e6 / tok,
            // Measured at the io_uring ring on the reaper thread, NOT derived from
            // `moe_wall − compute_gpu` as it was before. Off-thread, so it overlaps the
            // decode wall and can exceed it.
            io_wait_ms: io_wait_ns as f64 / 1e6 / tok,
            exposed_fetch_ms: exposed_ns / 1e6 / tok,
            cpu_ms: cpu_ns / 1e6 / tok,
            cpu_launch_ms: per(self.cpu_launch_ns),
            cpu_route_ms: per(self.cpu_route_ns),
            cpu_submit_ms: per(self.cpu_submit_ns),
            route_wait_ms: per(self.route_wait_ns),
            tail_wait_ms: per(self.tail_wait_ns),
            tail_gpu_ms: per(self.tail_gpu_ns),
            // `hits + misses` IS the chosen-slot count (submit_layer looks each one up
            // exactly once), so it is the same denominator hit% already uses. Reported
            // only under `top-m` — a 0.0% next to lru would read as a measurement.
            swap_pct: advice
                .map(|_| 100.0 * self.swap_n as f64 / (hits + misses).max(1) as f64),
        }
    }
}

/// Device-side DSA/MISA indexer state. Mirrors the trained lightning indexer but
/// everything is device-resident: per full layer a bf16 key slab grown in place,
/// plus per-token scratch. The score-readback buffers below are filled only by the
/// `trace`-feature score dump — the selection itself never leaves the device. MISA
/// additionally maintains a per-full-layer
/// block-pooled key pool and routes the top-`active_heads` indexer heads via a cheap
/// device estimate before scoring.
struct DeviceIndexer {
    /// Per layer: `Some(slab_index)` for full layers, `None` for shared.
    slab_of: Vec<Option<usize>>,
    /// Per full layer, the bf16 key cache (max_ctx * index_head_dim u16).
    kc: Vec<DeviceBuf>,
    k: DeviceBuf,      // index_head_dim f32 (one key, pre-cache)
    q: DeviceBuf,      // index_n_heads * index_head_dim f32
    w: DeviceBuf,      // index_n_heads f32
    scores: DeviceBuf, // max_ctx f32
    /// Score readback, `trace` only: the `RIVOLI_DUMP_SCORES` dump is the sole reader
    /// now that the selection never leaves the device.
    #[cfg(feature = "trace")]
    scores_host: Vec<u8>,
    #[cfg(feature = "trace")]
    scores_f: Vec<f32>,
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
    /// Scales the MoE branch before the residual add. `1.0` takes the plain `vadd`
    /// path, so the default is BIT-IDENTICAL to an engine without this knob — which
    /// matters because the `g = 1.0` arm is the in-session anchor the sweep is read
    /// against, and an anchor that quietly ran different arithmetic anchors nothing.
    moe_gain: f32,
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
    /// LOOKA (CACHE_PILOT Step 1): scratch for running a FUTURE layer's router against
    /// this layer's post-attention residual. Separate from `xn`/`gate_logits` so the pilot
    /// cannot perturb the real forward pass — the whole point is a zero-effect measurement.
    /// `pilot_logits` holds both horizons back-to-back so one D2H serves both.
    pilot_xn: DeviceBuf,
    pilot_logits: DeviceBuf,
    /// `--pilot`: speculatively prefetch the rank-0 prediction for the NEXT layer.
    /// Measured at 99% precision (docs/CACHE_PILOT.md), which is what makes a one-expert
    /// gate worth issuing on a bandwidth-bound engine.
    pilot: bool,
    /// Routing scratch for the pilot, deliberately separate from the hot path's
    /// `scores`/`choice`/`sel`/`cand` so a prediction can never perturb a real selection.
    pilot_scores: Vec<f32>,
    pilot_choice: Vec<f32>,
    pilot_sel: Vec<usize>,
    pilot_cand: Vec<usize>,
    pilot_host: Vec<u8>,
    /// This layer's rank-0 guess for layer+1, handed to `submit_layer`.
    spec_req: Option<usize>,
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
    /// `top-m` only: the ranked top-M candidate window [`route_into`] substitutes over.
    /// Stays empty under every other policy (the advice-`None` early return never
    /// writes it).
    cand: Vec<usize>,
    /// The policy's routing advice, read ONCE — the policy is fixed for the run, and
    /// `None` must stay a plain early return on the per-layer routing path.
    route_advice: Option<(usize, usize)>,
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
    /// Per-selected-expert: was it already resident? Drives the batched launch below.
    hit: Vec<bool>,
    #[cfg(feature = "trace")]
    looka: crate::looka::Looka,
    gl_host: Vec<u8>,
    argmax_host: Vec<u8>,
    /// Always-on cheap per-token profiling (see [`Profile`]).
    prof: Profile,
    /// The MoE expert stream's compute stream — resident/loaded experts' partials
    /// run here concurrently with the fetch stream's loads (the overlap). Separate
    /// from the null stream the rest of the forward uses.
    compute_stream: Stream,
    /// Compute-stream span events (bracket the MoE partials+reduce) + the expert
    /// stream's task monitor (idle = load-wait, poll = launch) — the accurate timing
    /// the async overlap hides from wall-clock.
    moe_ev_start: Event,
    moe_ev_end: Event,
    /// Brackets the DSA indexer's kernels inside `dsa_select_layer`. Recorded on the
    /// null stream and read behind the END-OF-LAYER sync, which every layer already
    /// pays rather than behind the mid-layer one — so the span survives if the mid-layer
    /// join is ever deleted, which would otherwise retire the instrument that measures it.
    idx_ev_start: Event,
    idx_ev_end: Event,
    /// Both indexer events recorded this layer and not yet read.
    idx_ev_pending: bool,
    /// Brackets final rmsnorm → lm_head → argmax on the null stream, read behind the
    /// argmax D2H that the tail already pays. Unlike the indexer pair there is nothing
    /// between these kernels, so the span is their execution time rather than a bracket
    /// with gaps — which is what lets it separate GPU work from sync overhead inside
    /// `tail_wait_ns`.
    tail_ev_start: Event,
    tail_ev_end: Event,
    /// One-shot latch so the first non-finite residual is reported once, not 78x per
    /// position for the rest of the run (`trace` only).
    #[cfg(feature = "trace")]
    nan_seen: bool,
    /// Tail events recorded this token and not yet read. False on the very first
    /// `argmax` (the prompt's forward has run, but so has its `record`) — kept as a
    /// guard anyway so an early-exit path cannot read an unrecorded event.
    tail_ev_pending: bool,
    moe_monitor: tokio_metrics::TaskMonitor,
    /// DIAGNOSTIC (`--checksum-x`, `trace` only): hash the residual stream each layer.
    #[cfg(feature = "trace")]
    checksum_x: bool,
    /// `RIVOLI_DUMP_SCORES=<path>`: raw `index_score` output, for characterising the
    /// distribution the host top-k actually faces (docs/NPU.md). `(file, calls seen,
    /// records left)` — bounded so a long run cannot fill the disk.
    #[cfg(feature = "trace")]
    score_dump: Option<(std::fs::File, u64, usize)>,
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
            tracing::info!("DSA row selection: device index_topk");
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
                #[cfg(feature = "trace")]
                scores_host: Vec::new(),
                #[cfg(feature = "trace")]
                scores_f: Vec::new(),
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
        // `mla_latent_attend` holds its online accumulator in MLA_ACC_REGS*SUBW = 512
        // registers per lane and rejects a wider kvl (arg guard 1004). Check it HERE:
        // the kernel guard would not fire until the first decoded token, by which point
        // the KV cache and the whole resident pin are already allocated.
        ensure!(
            kvl <= 512,
            "kv_lora_rank ({kvl}) exceeds 512, the attend kernel's register-resident \
             accumulator cap (MLA_ACC_REGS*SUBW in kernels/attn.hip)",
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
            moe_gain: 1.0,
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
            attn_partial: f(crate::backend::attend_scratch_floats(h, kvl))?,
            ctx: f(h * cfg.v_head_dim)?,
            gate_logits: f(cfg.n_experts)?,
            pilot_xn: f(cfg.hidden)?,
            pilot_logits: f(cfg.n_experts * PILOT_HORIZONS)?,
            pilot: false,
            pilot_scores: vec![0.0; cfg.n_experts],
            pilot_choice: vec![0.0; cfg.n_experts],
            pilot_sel: Vec::with_capacity(16),
            pilot_cand: Vec::new(),
            pilot_host: Vec::new(),
            spec_req: None,
            mlp_g: f(cfg.dense_inter)?,
            mlp_u: f(cfg.dense_inter)?,
            moe_out: f(cfg.hidden)?,
            moe_partial: f(slots * cfg.hidden)?,
            moe_h: f(slots * cfg.moe_inter)?,
            descs_buf: DeviceBuf::new(slots * std::mem::size_of::<ExpertDesc>())?,
            wexpert_buf: f(slots)?,
            logits: f(cfg.vocab)?,
            // [i32 index | f32 value | u32 nonfinite-tag]. The tag rides this buffer
            // deliberately: the tail's D2H is already paid, so localising the NaN costs
            // no extra sync — and a sync is exactly what masks it (--checksum-x makes
            // the fault disappear entirely).
            argmax_dev: {
                // hipMalloc does NOT zero. Tag 0 means "clean", so an unzeroed byte
                // would fabricate a layer coordinate on the first failure — the probe
                // would confidently point at the wrong place.
                let mut b = DeviceBuf::new(12)?;
                b.copy_in_at(0, &[0u8; 12])?;
                b
            },
            lc,
            lc_scale,
            rc,
            n_kv_blocks,
            scores: vec![0.0; cfg.n_experts],
            choice: vec![0.0; cfg.n_experts],
            sel: Vec::with_capacity(cfg.top_k),
            window: Vec::new(), // grown once by the first traced layer; empty otherwise
            cand: Vec::new(),   // ditto, for the first substituted layer
            route_advice: pin.route_advice(),
            w: Vec::with_capacity(slots),
            codebooks: pin.codebooks(),
            mlps_vq: Vec::with_capacity(cfg.top_k),
            descs_vq: Vec::with_capacity(slots),
            fmt: Vec::with_capacity(slots),
            hit: Vec::with_capacity(slots),
            #[cfg(feature = "trace")]
            looka: crate::looka::Looka::new(cfg.n_layers),
            gl_host: Vec::with_capacity(cfg.n_experts * 4),
            argmax_host: Vec::with_capacity(12),
            prof: Profile::default(),
            compute_stream: Stream::compute()?,
            moe_ev_start: Event::new()?,
            moe_ev_end: Event::new()?,
            #[cfg(feature = "trace")]
            nan_seen: false,
            tail_ev_start: Event::new()?,
            tail_ev_end: Event::new()?,
            tail_ev_pending: false,
            idx_ev_start: Event::new()?,
            idx_ev_end: Event::new()?,
            idx_ev_pending: false,
            moe_monitor: tokio_metrics::TaskMonitor::new(),
            #[cfg(feature = "trace")]
            checksum_x: false,
            #[cfg(feature = "trace")]
            ck_buf: Vec::new(),
            #[cfg(feature = "trace")]
            score_dump: std::env::var("RIVOLI_DUMP_SCORES")
                .ok()
                .and_then(|p| std::fs::File::create(&p).map(|f| (f, 0, 64)).ok()),
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
    /// Speculative prefetches issued, and how many the next layer actually asked for.
    /// The ratio is LIVE precision and should track LOOKA's measured `p@0` (99% at L+1);
    /// a large gap means the loader is not prefetching what the predictor predicted.
    pub fn spec(&self) -> (u64, u64) {
        (self.pin.spec_issued, self.pin.spec_used)
    }

    pub fn misses(&self) -> u64 {
        self.pin.misses
    }

    /// `swap%` for a run that did not go through [`Profile::summary`] — `--ppl` scores a
    /// text rather than generating, so it never builds one. Same numerator and the same
    /// `hits + misses` denominator, and `None` under every policy but `top-m`.
    pub fn swap_pct(&self) -> Option<f64> {
        self.route_advice.map(|_| {
            100.0 * self.prof.swap_n as f64 / (self.pin.hits + self.pin.misses).max(1) as f64
        })
    }

    /// Set the MoE-branch gain (see the field and kernels/fwd.hip::vaxpy).
    pub fn set_moe_gain(&mut self, g: f32) {
        if g != 1.0 {
            tracing::warn!("MoE branch gain {g} — EXPERIMENT arithmetic, not a normal run");
        }
        self.moe_gain = g;
    }

    /// `--pilot`: speculatively prefetch the next layer's rank-0 predicted expert.
    pub fn set_pilot(&mut self, on: bool) {
        self.pilot = on;
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
    /// key, then score + top-k once the cache exceeds index_topk (below that it is
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

        // DSA only, and this is a correctness guard, not a scoping choice: MISA's
        // head-route runs its own `device_sync` + D2H *inside* this bracket, which would
        // fold host time into a GPU-timeline number. Under misa the buckets stay 0 and the
        // summary line stays silent. Read behind the end-of-layer join (`idx_ev_pending`).
        let bracket = active_heads.is_none();
        if bracket {
            self.idx_ev_start.record(std::ptr::null_mut())?;
        }
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
        // The attend's row count. Was an OBSERVED `idx.rows.len()`; with the selection
        // device-resident nothing reads it back, so it now holds by construction —
        // `min(topk, nt)`, matching `rivoli_index_topk`'s own clamp of `k` to `nt`. The
        // guard above already returned, so it is exactly `topk`; written as the min so it
        // survives a change to that guard.
        let nr = topk.min(nt);

        // Query heads (wq_b·qr, roped per head) + gates (weights_proj·xn), then score
        // every cached token and pick the top-k, both on device.
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
                blocked(&mut self.prof.sync_wait_ns, "gpu-wait/misa-sync", device_sync)?;
                // Same reason as the score D2H below — MISA-only (`--misa-heads`), so
                // dormant on every arm measured so far, but unwrapped it would land in
                // `cpu`.
                blocked(&mut self.prof.sync_wait_ns, "gpu-wait/misa-d2h", || {
                    idx.e.copy_out_prefix(&mut idx.e_host, nh * 4)
                })?;
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
        // Launched INSIDE the event bracket deliberately: the kernel's cost lands in
        // `idx_gpu_ns` rather than nowhere, so selecting on device is priced rather than
        // credited with the host round-trip it removed.
        //
        // SAFETY: scp holds nt f32 (just written by index_score, same stream); rows_buf
        // is max_ctx u32 and the kernel writes exactly nr = min(topk, nt) <= nt <=
        // max_ctx. Both buffers are engine-owned.
        unsafe {
            launch_index_topk(
                scp as *const f32,
                nt,
                topk,
                self.rows_buf.ptr_mut() as *mut u32,
            )?;
        }
        if bracket {
            self.idx_ev_end.record(std::ptr::null_mut())?;
            self.idx_ev_pending = true;
        }
        // The mid-layer join. TWO consumers — it makes the score D2H below safe AND
        // retires the event pair. Deleting it was measured as its own arm and is worth
        // −2.5 ms/token, 0.6% of wall, at the cost of making `route` incomparable with
        // every historical row in benchmarks.md; not taken, see the module note above.
        blocked(&mut self.prof.sync_wait_ns, "gpu-wait/idx-sync", device_sync)?;
        // `RIVOLI_DUMP_SCORES` is the only thing left that wants the scores host-side.
        // Gated on the REMAINING budget, not on the file, so a finished dump stops paying
        // for a D2H the selection itself no longer needs.
        #[cfg(feature = "trace")]
        if self
            .score_dump
            .as_ref()
            .is_some_and(|(_, _, left)| *left > 0)
        {
            // Classed as a GPU wait, not host compute: the thread is parked in a transfer.
            // Unwrapped, it inflated the `cpu` bucket on the `--attn dsa` arms.
            blocked(&mut self.prof.sync_wait_ns, "gpu-wait/idx-scores-d2h", || {
                idx.scores.copy_out_prefix(&mut idx.scores_host, nt * 4)
            })?;
            idx.scores_f.clear();
            idx.scores_f.extend(
                idx.scores_host
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        // The score dump writes AFTER the join above, deliberately: an ~8 KB write
        // inside a timed window would inflate the one bucket the offload analysis
        // attributes. Harmless while
        // the budget exhausts during prefill (the buckets reset before decode), but the
        // budget is meant to be raised — so the write must not be in the window at all.
        #[cfg(feature = "trace")]
        if let Some((w, seen, left)) = self.score_dump.as_mut() {
            // STRIDE is coprime with the 21 full layers per token (211 = 10*21 + 1), so
            // consecutive samples advance one layer AND ~10 tokens: 64 records cover all
            // 21 layers and a ~630-token span of context. A prefix dump of the same size
            // would cover 3 tokens at one context, which is the wrong shape for the
            // question this exists to answer — whether the distribution moves with nt.
            const STRIDE: u64 = 211;
            if *left > 0 && *seen % STRIDE == 0 {
                use std::io::Write;
                let _ = w.write_all(&(l as u32).to_le_bytes());
                let _ = w.write_all(&(nt as u32).to_le_bytes());
                // Record layout, so the file is readable without this source:
                // repeated `(u32 layer, u32 nt, f32[nt])`, native LE.
                let _ = w.write_all(as_le_bytes(&idx.scores_f));
                *left -= 1;
            }
            *seen += 1;
        }

        idx.last_dense = false;
        idx.last_nr = nr;
        Ok((self.rows_buf.ptr() as *const u32, nr))
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
            // Position for the span tree: two relaxed stores, free when RIVOLI_SPANS is
            // unset. The reaper reads these too, so its io-wait lands under the layer
            // whose batch it is servicing.
            crate::telemetry::spans::mark(pos as u32, token, l as i32);

            // This layer's miss count, hoisted so the end-of-layer bracket read can bucket
            // by it. 0 on dense layers, which never enter the MoE branch.
            let mut layer_misses = 0usize;
            // Open the launch-cost span. Everything from here to the gate D2H (MoE) or
            // the end of the dense MLP is host-side kernel issue — EXCEPT the indexer's
            // joins on the `--attn dsa` arms, so the close subtracts whatever
            // `sync_wait_ns` accrued inside rather than assuming the region never blocks.
            let t_launch = std::time::Instant::now();
            let sync_at_open = self.prof.sync_wait_ns;

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
            // the q-LoRA residual + `xnp` the layer input, both from phase 1). Whether DSA
            // syncs mid-layer is `dsa_select_layer`'s business, not this call site's.
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

            // --- LOOKA pilot (CACHE_PILOT Step 1, `--features trace`) ---
            // Run FUTURE layers' routers against THIS layer's post-attention residual and
            // stash what they name. Placed here deliberately: `xp` still holds the
            // post-attention residual, before the MoE result is added back into it — which
            // is exactly the state a real prefetcher would have to predict from, since
            // layer L+h's true input needs L's MoE output that has not been computed yet.
            // That staleness IS the thing being measured; correcting for it would measure a
            // predictor nobody can build.
            // Runs when the prefetcher needs it (`--pilot`) or when LOOKA is measuring.
            if self.pilot || cfg!(feature = "trace") {
                let mut any = false;
                for (hi, &dh) in PILOT_H[..PILOT_HORIZONS].iter().enumerate() {
                    let t = l + dh;
                    if t >= cfg.n_layers {
                        continue;
                    }
                    // Copy the pointers out before touching `self.looka` — same borrow
                    // dance as the layer header above.
                    let tw = &self.pin.layers[t];
                    let t_post_ln = tw.post_ln;
                    let t_gate_w = match &tw.mlp {
                        LayerMlp::Moe { gate_w, .. } => *gate_w,
                        // A dense target has no router to run. Leaving the stash empty
                        // keeps it out of the denominator rather than scoring a miss.
                        LayerMlp::Dense(_) => continue,
                    };
                    let pxn = self.pilot_xn.ptr_mut() as *mut f32;
                    let plp = (self.pilot_logits.ptr_mut() as *mut f32).wrapping_add(hi * cfg.n_experts);
                    // SAFETY: `xp` is the live residual; `t_post_ln`/`t_gate_w` are resident
                    // F32 dense weights (never streamed, so valid for any layer at any
                    // time); `pxn`/`plp` are engine-owned scratch, and `plp` is in bounds
                    // because `pilot_logits` is sized HORIZONS.len() * n_experts. Both
                    // launches are stream-ordered, so the gemv sees the norm's output.
                    unsafe {
                        launch_rmsnorm(xp, t_post_ln, hidden, eps, pxn)?;
                        launch_gemv_f32(pxn, t_gate_w, cfg.n_experts, hidden, plp)?;
                    }
                    any = true;
                }
                if any {
                    // ONE D2H for both horizons. This is the only sync the pilot adds that
                    // the forward pass does not already pay, so it is kept to a single
                    // join rather than one per horizon.
                    let (pl, host) = (&self.pilot_logits, &mut self.pilot_host);
                    blocked(&mut self.prof.sync_wait_ns, "gpu-wait/pilot-d2h", || {
                        pl.copy_out_into(host)
                    })?;
                    for (hi, &dh) in PILOT_H[..PILOT_HORIZONS].iter().enumerate() {
                        let t = l + dh;
                        if t >= cfg.n_layers
                            || !matches!(self.pin.layers[t].mlp, LayerMlp::Moe { .. })
                        {
                            continue;
                        }
                        let lo = hi * cfg.n_experts * 4;
                        // Route the pilot the SAME way the real path will, minus the
                        // cache-conditional advice: `top-m` reorders on residency, and a
                        // prediction that consulted residency would be scoring the cache
                        // rather than the predictor it is supposed to be scoring.
                        crate::math::route_into(
                            &self.pilot_host[lo..lo + cfg.n_experts * 4],
                            self.pin.moe_bias(t),
                            cfg.top_k,
                            None,
                            |_| true,
                            &mut self.pilot_scores,
                            &mut self.pilot_choice,
                            &mut self.pilot_sel,
                            &mut self.pilot_cand,
                        );
                        // The prefetch request: rank 0 of the L+1 prediction, the 99%
                        // p@0 arm. Taken before the trace-only stash so `--pilot` works in
                        // a build with no LOOKA counters at all.
                        if self.pilot && dh == 1 {
                            self.spec_req = self.pilot_sel.first().copied();
                        }
                        #[cfg(feature = "trace")]
                        self.looka.stash(t, hi, &self.pilot_sel);
                    }
                }
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
                // Dense layer: attention + MLP were all launches, nothing blocked.
                    let e_launch = std::time::Instant::now();
                    self.prof.cpu_launch_ns += e_launch
                        .duration_since(t_launch)
                        .as_nanos()
                        .saturating_sub(self.prof.sync_wait_ns - sync_at_open);
                    crate::telemetry::spans::record("cpu/launch", "decode", t_launch, e_launch);
            } else {
                // Router gate on device, then read logits to route on host.
                // SAFETY: gate_w resident F32; glp device scratch.
                unsafe { launch_gemv_f32(xnp, gate_w, cfg.n_experts, hidden, glp)? };
                // The gate-logits D2H is a blocking join, so timing around it is free —
                // no sync we don't already pay. (All the always-on profile buckets wrap
                // existing join/D2H points; none add a sync.)
                // MoE layer: close the launch span before the first blocking call.
                    let e_launch = std::time::Instant::now();
                    self.prof.cpu_launch_ns += e_launch
                        .duration_since(t_launch)
                        .as_nanos()
                        .saturating_sub(self.prof.sync_wait_ns - sync_at_open);
                    crate::telemetry::spans::record("cpu/launch", "decode", t_launch, e_launch);
                let t = std::time::Instant::now();
                // Read the gate logits, route with `bias` borrowed straight out of
                // `&self.pin` while the routing scratch is borrowed mutably.
                // The D2H is separately classed as a GPU wait: `route_ns` is a region
                // and this is the blocking half of it, so without the split a `route`
                // number cannot distinguish attention the GPU is still finishing from
                // routing the host is doing.
                blocked(&mut self.prof.route_wait_ns, "gpu-wait/gate-d2h", || {
                    self.gate_logits.copy_out_into(&mut self.gl_host)
                })?;
                // `--cache-policy top-m` (advice Some) makes this cache-CONDITIONAL:
                // residency reorders the selection below the sacred top-J. Every other
                // policy leaves the advice None and gets the pre-top-m routing back
                // bit-for-bit. Residency is read through `Pin::resident` →
                // `HybridPolicy::contains`, which does not touch the eviction clock;
                // the substitution is inside the `route_ns` clock because it IS routing.
                let t_route = std::time::Instant::now();
                self.prof.swap_n += route_into(
                    &self.gl_host,
                    self.pin.moe_bias(l),
                    cfg.top_k,
                    self.route_advice,
                    |e| self.pin.resident(l, e),
                    &mut self.scores,
                    &mut self.choice,
                    &mut self.sel,
                    &mut self.cand,
                );
                // The host half of the route region, stamped rather than derived.
                let e_route = std::time::Instant::now();
                self.prof.cpu_route_ns += e_route.duration_since(t_route).as_nanos();
                crate::telemetry::spans::record("cpu/route-into", "decode", t_route, e_route);
                self.prof.route_ns += t.elapsed().as_nanos();
                // LOOKA: score whatever layers `l-1` / `l-2` predicted for THIS layer
                // against what it actually chose, then roll `sel` into the previous-token
                // baseline. Deliberately outside both route clocks (they close above) —
                // this is measurement, not routing, and `route_ns` has to stay comparable
                // across this change.
                #[cfg(feature = "trace")]
                {
                    let (lk, sel) = (&mut self.looka, &self.sel);
                    lk.score(l, sel);
                }
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
                // Residency must be sampled BEFORE submit_layer: it allocates slots for
                // misses, so afterwards everything reads as resident. A bitmask, not a
                // Vec — this runs 78x per token and top_k is 8.
                let warm_mask: u64 = if crate::telemetry::spans::enabled() {
                    self.sel
                        .iter()
                        .take(64)
                        .enumerate()
                        .filter(|&(_, &e)| self.pin.resident(l, e))
                        .fold(0u64, |m, (i, _)| m | (1 << i))
                } else {
                    0
                };
                let t_sub = std::time::Instant::now();
                let mut signals = self.pin.submit_layer(
                    l,
                    &self.sel,
                    &self.window,
                    &self.choice,
                    &mut self.mlps_vq,
                    &mut self.fmt,
                    &mut self.hit,
                    self.spec_req.take(),
                )?;
                // Pure host work: residency lookups + policy bookkeeping + read specs.
                // It only ENQUEUES the reads; the reaper thread does the waiting.
                let e_sub = std::time::Instant::now();
                self.prof.cpu_submit_ns += e_sub.duration_since(t_sub).as_nanos();
                crate::telemetry::spans::record("cpu/submit-layer", "decode", t_sub, e_sub);
                // Residency x format, the pair that explains a layer's cost. `fmt` is
                // filled by submit_layer in `sel` order (the shared expert is pushed
                // after, so `take(sel.len())` keeps this to the routed picks).
                if crate::telemetry::spans::enabled() {
                    let mut st = crate::telemetry::spans::LayerState {
                        tok: pos as u32,
                        layer: l as i32,
                        ..Default::default()
                    };
                    for (i, &i4) in self.fmt.iter().take(self.sel.len()).enumerate() {
                        let warm = i < 64 && (warm_mask & (1 << i)) != 0;
                        match (warm, i4) {
                            (true, true) => st.warm_i4 += 1,
                            (false, true) => st.cold_i4 += 1,
                            (true, false) => st.warm_vq3 += 1,
                            (false, false) => st.cold_vq3 += 1,
                        }
                    }
                    crate::telemetry::spans::record_layer(st);
                }
                layer_misses = (self.pin.misses - miss0) as usize;
                self.prof.fetch_n += self.pin.misses - miss0;
                self.prof.spec_n = self.pin.spec_issued;
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
                    // The shared expert is in the RESIDENT tier, never streamed — its
                    // Signal is `ready()` right here. `hit` must grow with `fmt`/`descs`
                    // or it is shorter than `ndesc` and the batching loop below indexes
                    // past the end (it did: "len is 8 but the index is 8", because `hit`
                    // covers only the 8 routed picks).
                    self.hit.push(true);
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
                // Cloned alongside `fmt` for the same reason: the async block below moves
                // what it captures, and `self` is borrowed mutably across the await.
                let hit = self.hit.clone();
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
                // FIRST, launch every already-resident expert with NO host round-trip,
                // batching maximal runs into single dispatches.
                //
                // A hit's Signal is already resolved, so awaiting it before launching buys
                // nothing and costs the GPU an idle gap — and at the measured ~76% hit rate
                // that is ~7 of 9 experts per layer being gated for no reason. Measured
                // cost of the gating: the MoE phase runs at 38.8 GB/s in-engine against
                // 91.8 GB/s for the same kernels in `examples/moe_bench.rs`, i.e. 42% of
                // what the shaders do when the host is not in the dependency chain.
                //
                // BIT-IDENTICAL by construction, not by hope: `moe_expert_range` computes
                // `e = e_start + row/inter` with every row independent, so an `e_count > 1`
                // dispatch is exactly the same arithmetic as `e_count` separate ones, and
                // `moe_reduce` sums `for e in 0..e_count` in fixed order either way.
                //
                // Runs must be uniform in FORMAT as well as residency: hybrid mixes int4
                // and int3-vq within a layer and they are different kernels. `sel` order is
                // NOT permuted to make longer runs — `pin.rs`'s trace-v2 invariant requires
                // `window[..sel.len()] == sel` and `bin/replay` hard-fails otherwise.
                debug_assert_eq!(
                    hit.len(), ndesc,
                    "hit mask must cover every descriptor (routed picks + the shared expert)"
                );
                let mut i = 0usize;
                while i < ndesc.min(hit.len()).min(fmt.len()) {
                    if !hit[i] {
                        i += 1;
                        continue;
                    }
                    let f = fmt[i];
                    let mut j = i;
                    while j < ndesc && hit[j] && fmt[j] == f {
                        j += 1;
                    }
                    // SAFETY: descs/codebooks resident; these slots are HITS, so their
                    // bytes are already in place; h/part device scratch; cs_raw live.
                    unsafe {
                        if f {
                            launch_moe_expert_range_i4(
                                x_c, hidden, inter, i, j - i, descs_ptr, w_ptr, h_c, part_c,
                                cs_raw,
                            )?;
                        } else {
                            launch_moe_expert_range(
                                x_c, hidden, inter, i, j - i, descs_ptr, cb0, cb1, cb2, w_ptr,
                                h_c, part_c, cs_raw,
                            )?;
                        }
                    }
                    i = j;
                }
                // THEN the misses, still one await + launch each. Removing the host from
                // this path needs a device-side cross-stream wait (hipStreamWaitEvent /
                // a timeline wait) and is the next step, not this one.
                let miss: Vec<usize> = (0..ndesc).filter(|&e| !hit[e]).collect();
                futures_util::stream::iter(miss.clone())
                    .map(Ok::<usize, anyhow::Error>)
                    .try_for_each_concurrent(miss.len().max(1), move |e| {
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
            // SAFETY: residual add of the MLP contribution. `--moe-gain` applies ONLY
            // on MoE layers — the 3 dense layers share this add, and attenuating them
            // too would confound "the MoE branch is too strong" with "the MLP branch
            // is". At g == 1.0 this is the plain `vadd`, bit for bit.
            let mp = self.moe_out.ptr() as *const f32;
            unsafe {
                match (dense_mlp.is_none(), self.moe_gain) {
                    (true, g) if g != 1.0 => launch_vaxpy(xp, mp, g, hidden)?,
                    _ => launch_vadd(xp, mp, hidden)?,
                }
            }
            // End-of-layer join: protects the reused descs/wexpert/moe_out buffers
            // before the next layer overwrites them, and surfaces faults. Folds into
            // the MoE wall (near-0 for MoE layers — the compute stream already synced;
            // the dense MLP compute for the 3 dense layers).
            let t = std::time::Instant::now();
            // NOT stamped into `sync_wait_ns`: this join is already inside
            // `moe_wall_ns`, and the class axis takes the MoE phase's contribution from
            // that wall. Adding it to both would count it twice within one axis, which
            // is exactly what made the first version of the CLASS line overshoot.
            device_sync()?;
            let e_moe = std::time::Instant::now();
            self.prof.moe_wall_ns += e_moe.duration_since(t).as_nanos();
            crate::telemetry::spans::record("gpu-wait/end-of-layer-sync", "decode", t, e_moe);
            // Localise a non-finite residual to the earliest (pos, layer) that produced
            // one. `atomicCAS(flag, 0, tag)` keeps the FIRST, and tag 0 is reserved for
            // "clean" so the tag is offset by 1. One tiny kernel per layer, no sync.
            // SAFETY: `x` is `hidden` device f32; the flag is 4 bytes inside argmax_dev.
            unsafe {
                launch_flag_nonfinite(
                    xp,
                    hidden,
                    1 + (pos as u32) * 256 + l as u32,
                    self.argmax_dev.ptr_mut().add(8) as *mut u32,
                )?;
            }
            // Both MoE span events retired by the sync — read the compute-stream span
            // (MoE layers only; dense layers never recorded them).
            if dense_mlp.is_none() {
                let ms = Event::elapsed_ms(&self.moe_ev_start, &self.moe_ev_end)?;
                let ns = (ms as f64 * 1e6) as u128;
                self.prof.compute_gpu_ns += ns;
                // Bucket by this layer's miss count. Clamped rather than asserted: the
                // shared expert is never a miss and top_k is 8, so 16 is generous, but a
                // future top_k must not panic here.
                let b = layer_misses.min(self.prof.moe_ns_by_miss.len() - 1);
                self.prof.moe_ns_by_miss[b] += ns;
                self.prof.moe_n_by_miss[b] += 1;
            }
            // Same sync, same reason, for the indexer span this layer recorded — read
            // here because this join is unconditional and the mid-layer one is not.
            if self.idx_ev_pending {
                self.idx_ev_pending = false;
                let ms = Event::elapsed_ms(&self.idx_ev_start, &self.idx_ev_end)?;
                self.prof.idx_gpu_ns += (ms as f64 * 1e6) as u128;
                self.prof.idx_layers += 1;
            }
            // DIAGNOSTIC (`--checksum-x`): hash the residual stream after every layer,
            // and — the reason this also scans for non-finite values — localise the
            // intermittent NaN. The production guard only fires at `argmax`, i.e. after
            // all 78 layers of all ~70 prefill positions, so it says the run is broken
            // without saying where. This names the first (pos, layer) that goes bad, and
            // whether that layer had cold misses.
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
                let bad = self
                    .ck_buf
                    .chunks_exact(4)
                    .filter(|c| {
                        !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite()
                    })
                    .count();
                if bad > 0 && !self.nan_seen {
                    self.nan_seen = true;
                    tracing::error!(
                        "FIRST NON-FINITE RESIDUAL at pos={pos} layer={l}: {bad}/{hidden} \
                         elements. misses this layer={}, dense={}. Everything downstream \
                         is poisoned, so only this first report localises the fault.",
                        self.prof.fetch_n,
                        dense_mlp.is_some(),
                    );
                }
                tracing::info!("XSUM pos={pos} l={l} x={hh:016x} nonfinite={bad}");
            }
        }

        // Out of the layer loop — the tail hangs off the token, not off a layer.
        crate::telemetry::spans::mark(pos as u32, token, -1);
        // Open the tail GPU span. The end-of-layer `device_sync` just above drained
        // everything, so this timestamp sits on an idle stream and the span that
        // follows is the tail kernels and nothing else.
        self.tail_ev_start.record(std::ptr::null_mut())?;
        self.tail_ev_pending = true;
        // The tail's host launch cost, on the same `cpu_launch_ns` clock as the layers'.
        // Nothing blocks between here and the end of `forward`.
        let t_tail_launch = std::time::Instant::now();
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
        let e_tail_launch = std::time::Instant::now();
        self.prof.cpu_launch_ns += e_tail_launch.duration_since(t_tail_launch).as_nanos();
        crate::telemetry::spans::record("cpu/launch-tail", "decode", t_tail_launch, e_tail_launch);
        Ok(())
    }

    /// **Teacher-forced** negative log-likelihood of `ids`: one `-log softmax(logits)[t]`
    /// per predicted position, returned in order. `ids[0]` is context only, so the result
    /// has `ids.len() - 1` entries.
    ///
    /// Teacher-forced means at every position we feed the KNOWN next token, never our own
    /// argmax. That is the whole point: a free-running run's quality number is confounded
    /// by its own trajectory — a cache policy that degenerates into repetition routes to
    /// fewer experts, hits more, and looks *better* on every metric the run generates
    /// about itself. Forcing the text pins the trajectory so two policies are scored on
    /// literally the same positions, which is also what makes the per-token NLLs PAIRABLE
    /// across runs. See docs/CACHE_ROUTE.md "Quality".
    ///
    /// The full `vocab` logit vector comes back to the host each position. That is ~620 KB
    /// against ~0.96 GB/token of expert streaming — noise. A device-side log-softmax would
    /// be a kernel to write, test and debug in order to save 0.06% of the traffic.
    // ponytail: host log-softmax, no kernel.
    pub fn nll_forced(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        ensure!(ids.len() >= 2, "need at least 2 tokens to score a prediction");
        let vocab = self.cfg.vocab;
        // Same shape as `generate`: one current-thread runtime, `forward` awaited inline.
        let rt = tokio::runtime::Builder::new_current_thread().build()?;
        let out = rt.block_on(async {
            let mut out = Vec::with_capacity(ids.len() - 1);
            let mut host: Vec<u8> = Vec::with_capacity(vocab * 4);
            for (pos, &tok) in ids.iter().enumerate() {
                // Beat per position: scoring a long text is many forwards with no token
                // emitted, and the watchdog only knows about progress if we say so.
                if let Some(hb) = &self.heartbeat {
                    hb.beat();
                }
                self.forward(tok, pos).await?;
                let Some(&next) = ids.get(pos + 1) else { break };
                ensure!((next as usize) < vocab, "token {next} outside vocab {vocab}");
                self.logits.copy_out_into(&mut host)?;
                ensure!(host.len() == vocab * 4, "short logits D2H");
                out.push(nll_of(&host, next as usize)?);
            }
            Ok(out)
        })?;
        Ok(out)
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
        // Close the tail GPU span here — AFTER the argmax launch, BEFORE the D2H — so
        // it brackets exactly rmsnorm → lm_head → argmax. The start was recorded in
        // `forward`; the D2H below retires both, so reading it costs no extra sync.
        self.tail_ev_end.record(std::ptr::null_mut())?;
        // The one blocking call the whole tail phase hides behind: it drains the final
        // rmsnorm, lm_head AND argmax. Class it explicitly, or those milliseconds land
        // in the derived `cpu` bucket and look like host work.
        blocked(&mut self.prof.tail_wait_ns, "gpu-wait/argmax-d2h", || {
            self.argmax_dev.copy_out_into(&mut self.argmax_host)
        })?;
        // Both events retired by the D2H above.
        if self.tail_ev_pending {
            self.tail_ev_pending = false;
            let ms = Event::elapsed_ms(&self.tail_ev_start, &self.tail_ev_end)?;
            self.prof.tail_gpu_ns += (ms as f64 * 1e6) as u128;
        }
        debug_assert_eq!(
            self.argmax_host.len(),
            12,
            "argmax result must be 12 bytes: [idx | val | nonfinite tag]"
        );
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
            // The tag rode the same D2H, so this costs nothing and turns "somewhere in
            // 78 layers x every position" into a coordinate.
            let tag = u32::from_le_bytes([
                self.argmax_host[8],
                self.argmax_host[9],
                self.argmax_host[10],
                self.argmax_host[11],
            ]);
            let where_ = if tag == 0 {
                "no layer residual was non-finite — the fault is AFTER the last layer \
                 (final rmsnorm, lm_head or argmax itself), not in the MoE/attention stack"
                    .to_string()
            } else {
                format!(
                    "first non-finite residual at pos={} layer={}",
                    (tag - 1) / 256,
                    (tag - 1) % 256
                )
            };
            bail!("logits are non-finite (NaN/Inf in the GPU forward pass): {where_}");
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
        let (hit0, miss0, fetch0, io0, decode_wall) = rt.block_on(async {
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
            // Baseline the reaper's counters too. `hits`/`misses` were already rebased
            // here but `fetch_ns` never was, so `fetch_wall_ms` has always folded the
            // PREFILL's (cold, expensive) fetch into the decode average. Invisible at
            // -bench 512 where 5 prompt tokens amortize away; at -bench 8 it reported
            // io-wait at 136% of wall, which is how it was found.
            let fetch0 = self.pin.fetch_ns();
            let io0 = self.pin.io_wait_ns();
            // Tell the span recorder how long the run is, so it can spread its budget
            // across the whole decode instead of spending it all on the cold start.
            // Six leaf spans per MoE layer — cpu/launch, gate-d2h, route-into,
            // submit-layer, end-of-layer-sync, and the reaper's io-wait/uring-reap. The
            // first estimate said five (it forgot the reaper's, which is on the other
            // thread) and overshot the budget by 10%. Rounding UP is the safe direction:
            // a slightly long stride samples fewer tokens, a short one truncates the tail.
            crate::telemetry::spans::plan(ngen, self.cfg.n_layers * 6 + 4);
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
            Ok::<_, anyhow::Error>((hit0, miss0, fetch0, io0, decode_wall))
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
            self.pin.fetch_ns().saturating_sub(fetch0),
            self.pin.io_wait_ns().saturating_sub(io0),
            tm.total_idle_duration.as_nanos() as u64,
            tm.total_poll_duration.as_nanos() as u64,
            self.route_advice,
        );
        summary.report();
        // LOOKA sits beside the profile rather than inside it: it is a property of the
        // ROUTER (would a prefetcher guess right?), not of where this run spent its time,
        // and it only exists in `trace` builds.
        #[cfg(feature = "trace")]
        {
            tracing::info!("{}", self.looka.report());
            for line in self.looka.rank_report() {
                tracing::info!("{}", line);
            }
        }
        Ok((generated, summary))
    }
}
