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
    Event, ExpertDesc, Signal, Stream, device_sync, fill_u32, launch_append_kv, launch_argmax,
    launch_attend, launch_embed_i8_row, launch_flag_nonfinite, launch_gather_rope,
    launch_gemv_f32, launch_gemv_fp8, launch_gemv_i8,
    launch_index_append, launch_index_head_route, launch_index_pool_push, launch_index_score,
    launch_index_topk, launch_layernorm, launch_mla_absorb_fp8, launch_mla_value_fp8,
    launch_moe_acc_drain, launch_moe_expert_range, launch_moe_expert_range_i4, launch_rmsnorm,
    launch_rope, launch_swiglu, launch_vadd, stream_signal,
};
use crate::memory::device::DeviceBuf;
use crate::math::{E4M3_BLOCK, route_into, topk_into};
use crate::artifact::model::ModelConfig;
use crate::fetch::asyncfetch::Ticket;
use crate::memory::pin::{Fp8Mlp, IndexerPin, LayerMlp, MlpVq, Pin, TRACE_WINDOW};

/// Fixed-point MoE accumulator rows: ONE PER STREAM (compute = 0, miss = 1), summed by
/// `moe_acc_drain`. Not per expert — every expert on a stream shares its row, and integer
/// addition associating is what makes that safe without ordering them.
///
/// Sharing a SINGLE row across both streams was measured and is worse: −90 µs on a 0-miss
/// layer (the reduce is gone) but +106/+449/+825 µs at 1/3/6 misses. A 1-miss layer issues
/// the same 9·hidden atomics as a 0-miss one, so the cost is cache lines bouncing between
/// two queues, not the atomics. Splitting the rows keeps the reduce gone and the streams
/// independent; the drain pays 2 reads instead of the old reduce's 9.
const MOE_ACC_ROWS: usize = 2;

use crate::telemetry::ProfileSummary;
use anyhow::{Context, Result, bail, ensure};

/// Little-endian byte view of a POD slice — zero-copy on this LE host (a `[T]`'s
/// in-memory bytes ARE its LE serialization). Feeds the per-token H2D uploads (attn
/// rows/heads u32, expert descriptors, weights f32) with no staging buffer.
/// Confidence buckets for the MTP accept histogram: 5 even bins over [0,1].
const MTP_BINS: usize = 5;

/// Token rows one forward pass carries. 2 = the speculative verify pass: the real token
/// at `pos` plus the MTP head's draft for `pos+1`, through ONE read of every weight.
///
/// Every device scratch buffer is allocated at this width and the batched kernels take
/// the live `nrow` at launch, so an `nrow == 1` pass is bit-identical to the unbatched
/// engine — which is what makes "speculative decode emits the same bytes as greedy
/// sequential decode" a testable claim rather than an aspiration.
///
/// Fixed at 2 by measurement, not taste: chained depth-2 drafts land at 4.4% acceptance
/// (GLM-5.2 ships `num_nextn_predict_layers = 1`), so a 3-row pass verifies 1.559
/// tokens against 1.535 for two — more rows for no tokens.
pub const MAXROW: usize = 2;

/// The argmax result buffer: `MAXROW` × [i32 index | f32 value], then a u32 non-finite
/// tag. ONE tag for the whole pass — it names the earliest `(pos, layer)` that went bad,
/// and every row of a pass shares a base position, so per-row tags would say the same
/// thing twice. The tag rides this D2H, which is why localising a NaN costs no sync.
const ARGMAX_BYTES: usize = MAXROW * 8 + 4;

/// Which pass `forward_inner` is running.
#[derive(Clone, Copy, PartialEq)]
enum Draft {
    /// Not a draft: the main model's own forward, over every layer.
    No,
    /// The MTP head alone (pinned layer `n_layers`). Row `r` consumes
    /// `(x[r], emb(tokens[r]))` — `x` being the hidden state the last main-model forward
    /// left resident, which is the real thing the head was trained to read.
    Head,
}

fn mtp_bin(conf: f32) -> usize {
    ((conf * MTP_BINS as f32) as usize).min(MTP_BINS - 1)
}

/// `RIVOLI_MTP_PROBE=1` — bucket acceptance by the MAIN MODEL's confidence too. Off by
/// default because it adds a 1.2 MB D2H per verify pass, which would contaminate the very
/// `mtp_draft` timing it ships beside.
fn mtp_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RIVOLI_MTP_PROBE").is_ok_and(|v| v != "0"))
}

/// `softmax(logits)[argmax]` from a raw f32 logit vector — the draft head's confidence
/// in its own pick, and the only free signal a speculate-or-not gate could read.
fn top1_prob(bytes: &[u8]) -> f32 {
    let l = |c: &[u8]| f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    let max = bytes.chunks_exact(4).map(l).fold(f32::NEG_INFINITY, f32::max);
    // Shifted so the largest term is exp(0)=1; the sum is then ≥1 and 1/sum is the
    // top-1 probability without ever forming the full softmax.
    let sum: f64 = bytes
        .chunks_exact(4)
        .map(|c| f64::from(l(c) - max).exp())
        .sum();
    (1.0 / sum) as f32
}

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
    /// `fetch_wall_ns` is the reaper's off-thread load cost.
    ///
    /// The `idle_ns`/`poll_ns` tokio-metrics pair is GONE with the ticketed dataflow: it
    /// measured the per-expert async awaits, and there are none — the GPU gates itself now.
    /// Removed rather than left reporting zeros, because a metric whose subject no longer
    /// exists reads as "this cost is zero" instead of "this is not measured here", and this
    /// file has already shipped two metrics that quietly excluded their own subject.
    #[allow(clippy::too_many_arguments)] // one call site; the buckets are unrelated scalars
    fn summary(
        &self,
        hits: u64,
        misses: u64,
        bytes_per_expert: usize,
        fetch_wall_ns: u64,
        io_wait_ns: u64,
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
        let moe_gpu_wait_ns = (self.moe_wall_ns as f64 - exposed_ns).max(0.0);
        let gpu_wait_ns = (self.route_wait_ns + self.sync_wait_ns + self.tail_wait_ns) as f64
            + moe_gpu_wait_ns;
        // CPU: measured host-compute regions. The expert stream's tokio poll used to be
        // added here — host work inside the MoE block the three decode-thread stamps cannot
        // see. With the launches enqueued straight onto the compute stream there is no such
        // work left to attribute.
        let cpu_ns = (self.cpu_launch_ns + self.cpu_route_ns + self.cpu_submit_ns) as f64;
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
            fetch_hidden_pct: hidden,
            miss_per_tok: self.fetch_n as f64 / tok,
            // Over ALL reads the reaper serviced, demand and speculative alike —
            // `fetch_wall_ns` covers both, so `fetch_n` alone is the wrong denominator.
            ms_per_miss: fetch_wall_ns as f64 / 1e6 / self.fetch_n.max(1) as f64,
            gb_per_tok: self.fetch_n as f64 / tok * bytes_per_expert as f64
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
    /// MTP scratch. `mtp_cat` is the `[enorm(emb) ‖ hnorm(h)]` concatenation `eh_proj`
    /// consumes (2·hidden); `mtp_x` is the head's own residual stream, swapped with `x`
    /// for the duration of the draft so the main model's hidden state survives it.
    mtp_cat: DeviceBuf,
    mtp_x: DeviceBuf,
    /// Drafts produced / drafts that matched the main model's next argmax. `--mtp` only.
    mtp_seen: u64,
    mtp_hit: u64,
    /// The same pair, bucketed by the DRAFT'S OWN confidence (softmax probability of its
    /// argmax). This is the signal a "should we speculate on this token?" gate would read,
    /// and bucketing it is how we find out whether it separates before building the gate.
    mtp_bins: [(u64, u64); MTP_BINS],
    /// The same pair, bucketed by the MAIN MODEL'S confidence in the token the draft was
    /// tested against (`softmax(row 0 logits)[argmax]`). This is the discriminator between
    /// the two reasons a draft misses: a weak head (misses even where the target is
    /// decidable) and a noisy target (everyone misses where the model itself is unsure).
    /// Only the first is worth de-quantizing layer 78 for. `RIVOLI_MTP_PROBE=1` — it costs
    /// a second 620 KB D2H per verify pass, which the timing runs must not pay.
    mtp_tgt_bins: [(u64, u64); MTP_BINS],
    /// Wall spent in `mtp_draft`, and passes counted. This is `d` in the speculative cost
    /// model — the term inferred rather than measured when §13 was written, and the one
    /// that decides whether a pre-draft gate (skipping the draft, not just the verify)
    /// is worth building at all.
    mtp_draft_ns: u128,
    mtp_draft_n: u64,
    /// Drafts the confidence gate actually spent a verify pass on. `mtp_seen` counts every
    /// draft including the gated-out ones, so `mtp_seen` IS the pass count (each iteration
    /// runs exactly one) and `generated / mtp_seen` is tokens per pass — the speedup over
    /// sequential decode, measured rather than modelled.
    mtp_verify: u64,
    /// Host staging for the draft's logits — the confidence needs the whole vector, and
    /// a host pass costs one 620 KB D2H behind a sync `argmax` already pays.
    /// ponytail: host softmax, no kernel — it is a measurement, not a decode-path cost.
    mtp_host: Vec<u8>,
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
    /// [MOE_ACC_ROWS*hidden] u64 fixed-point MoE accumulator, drained into the residual at
    /// end of layer. Replaced a [slots*hidden] f32 partial slab plus a reduce behind a
    /// cross-stream join. One row PER STREAM — see `MOE_ACC_ROWS`.
    moe_acc: DeviceBuf,
    moe_h: DeviceBuf,       // [slots*MAXROW*moe_inter] SwiGLU hidden scratch (VQ MoE)
    descs_buf: DeviceBuf,
    wexpert_buf: DeviceBuf,
    logits: DeviceBuf,
    /// Device argmax result: `MAXROW` pairs of [i32 index | f32 max-value], then one
    /// u32 non-finite tag shared by every row (see [`ARGMAX_BYTES`]).
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
    /// Trace-only: the ranked top-[`TRACE_WINDOW`] candidates. Stays empty unless
    /// `--trace` is on, and `--trace` is fixed for the run, so this is either filled
    /// every layer or never.
    window: Vec<usize>,
    /// `top-m` only: the ranked top-M candidate window [`route_into`] substitutes over.
    /// Stays empty under every other policy (the advice-`None` early return never
    /// The policy's routing advice, read ONCE — the policy is fixed for the run, and
    // Per-token host build scratch — reused every layer so the hot path allocates
    // nothing: resolved VQ descriptors + weights, the resolved batch, D2H staging.
    /// Per-expert routing weights for the current layer, laid out `[descriptor][row]` —
    /// the layout `moe_down_vq` reads as `wexpert[e*R + t]`. A row that did not route to
    /// a union expert carries 0.0 there, and `moe_down_vq` SKIPS a zero weight rather
    /// than multiplying by it, which is why the union cannot perturb a row's own result.
    w: Vec<f32>,
    /// Each token row's own top-`top_k` picks, before the union. Row 0's also feeds
    /// trace, which stays defined on the real token rather than the draft.
    sel_row: [Vec<usize>; MAXROW],
    /// Each row's normalized routed weights, parallel to `sel_row[r]`.
    wrow: [Vec<f32>; MAXROW],
    /// The deduplicated union of every row's picks — what actually gets submitted and
    /// launched. Row 0's picks come first, so an `nrow == 1` pass submits exactly `sel`.
    union: Vec<usize>,
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
    /// Per-descriptor device-side dependency. Replaces the `hit: Vec<bool>` residency mask,
    /// which encoded "do not await" as host data and could silently disagree with the real
    /// dependency — see the launch loop for the failure that caused.
    tickets: Vec<Ticket>,
    gl_host: Vec<u8>,
    argmax_host: Vec<u8>,
    /// Always-on cheap per-token profiling (see [`Profile`]).
    prof: Profile,
    /// The MoE expert stream's compute stream — resident/loaded experts' partials
    /// run here concurrently with the fetch stream's loads (the overlap). Separate
    /// from the null stream the rest of the forward uses.
    compute_stream: Stream,
    /// Experts whose bytes are still arriving launch HERE, not on `compute_stream`. See
    /// `Stream::miss` for why. Both streams accumulate into the SAME `moe_acc` row and
    /// need no join to do it — see `launch_moe_expert_range`.
    miss_stream: Stream,
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
        // Descriptor slots per MoE launch. A batched pass submits the UNION of every
        // row's picks, so the routed half scales with MAXROW; the shared experts are
        // row-independent and appear once. Rows overlap ~31% in practice (measured), so
        // the union is ~13.5 of the 16 this reserves.
        let slots = cfg.top_k * MAXROW + cfg.n_shared;
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
        // One KV slab per PIN layer, not per model layer: the MTP head is pinned one
        // past the end and attends over its own cache (built at every position, prefill
        // included), so it needs a slab like any other layer.
        let n_pin_layers = pin.layers.len();
        let mut lc = Vec::with_capacity(n_pin_layers);
        let mut lc_scale = Vec::with_capacity(n_pin_layers);
        let mut rc = Vec::with_capacity(n_pin_layers);
        for _ in 0..n_pin_layers {
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
            x: f(MAXROW * cfg.hidden)?,
            xn: f(MAXROW * cfg.hidden)?,
            mtp_cat: f(MAXROW * 2 * cfg.hidden)?,
            mtp_x: f(MAXROW * cfg.hidden)?,
            mtp_seen: 0,
            mtp_hit: 0,
            mtp_bins: [(0, 0); MTP_BINS],
            mtp_tgt_bins: [(0, 0); MTP_BINS],
            mtp_draft_ns: 0,
            mtp_draft_n: 0,
            mtp_verify: 0,
            mtp_host: Vec::new(),
            sub: f(MAXROW * cfg.hidden)?,
            qr: f(MAXROW * cfg.q_lora_rank)?,
            q: f(MAXROW * h * cfg.qk_head_dim())?,
            comp: f(MAXROW * (kvl + rope))?,
            qabs: f(MAXROW * h * kvl)?,
            qrope: f(MAXROW * h * rope)?,
            clat: f(MAXROW * h * kvl)?,
            attn_partial: f(MAXROW * crate::backend::attend_scratch_floats(h, kvl))?,
            ctx: f(MAXROW * h * cfg.v_head_dim)?,
            gate_logits: f(MAXROW * cfg.n_experts)?,
            mlp_g: f(MAXROW * cfg.dense_inter)?,
            mlp_u: f(MAXROW * cfg.dense_inter)?,
            moe_out: f(MAXROW * cfg.hidden)?,
            // Zeroed HERE and nowhere else: `moe_acc_drain` resets it as it converts, so
            // steady state needs no memset. hipMalloc does not zero, and layer 0 would
            // otherwise sum against whatever was resident.
            // Laid out `[stream][token row][hidden]`, so `moe_acc_drain` over
            // `nrow·hidden` elements with `MOE_ACC_ROWS` stream rows drains every token
            // row in one launch — the token and hidden axes are contiguous and the kernel
            // never has to know they are two axes.
            moe_acc: {
                let bytes = MOE_ACC_ROWS * MAXROW * cfg.hidden * 8;
                let mut b = DeviceBuf::new(bytes)?;
                // SAFETY: `b` owns `bytes`, just allocated.
                unsafe { fill_u32(b.ptr_mut() as *mut u8, 0, bytes)? };
                b
            },
            moe_h: f(slots * MAXROW * cfg.moe_inter)?,
            descs_buf: DeviceBuf::new(slots * std::mem::size_of::<ExpertDesc>())?,
            wexpert_buf: f(slots * MAXROW)?,
            logits: f(MAXROW * cfg.vocab)?,
            // [i32 index | f32 value | u32 nonfinite-tag]. The tag rides this buffer
            // deliberately: the tail's D2H is already paid, so localising the NaN costs
            // no extra sync — and a sync is exactly what masks it (--checksum-x makes
            // the fault disappear entirely).
            argmax_dev: {
                // hipMalloc does NOT zero. Tag 0 means "clean", so an unzeroed byte
                // would fabricate a layer coordinate on the first failure — the probe
                // would confidently point at the wrong place.
                let mut b = DeviceBuf::new(ARGMAX_BYTES)?;
                b.copy_in_at(0, &[0u8; ARGMAX_BYTES])?;
                b
            },
            lc,
            lc_scale,
            rc,
            n_kv_blocks,
            scores: vec![0.0; cfg.n_experts],
            choice: vec![0.0; cfg.n_experts],
            window: Vec::new(), // grown once by the first traced layer; empty otherwise
            w: Vec::with_capacity(slots * MAXROW),
            sel_row: std::array::from_fn(|_| Vec::with_capacity(cfg.top_k)),
            wrow: std::array::from_fn(|_| Vec::with_capacity(cfg.top_k)),
            union: Vec::with_capacity(slots),
            codebooks: pin.codebooks(),
            mlps_vq: Vec::with_capacity(slots),
            descs_vq: Vec::with_capacity(slots),
            fmt: Vec::with_capacity(slots),
            tickets: Vec::with_capacity(slots),
            gl_host: Vec::with_capacity(MAXROW * cfg.n_experts * 4),
            argmax_host: Vec::with_capacity(ARGMAX_BYTES),
            prof: Profile::default(),
            compute_stream: Stream::compute()?,
            miss_stream: Stream::miss()?,
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

    pub fn misses(&self) -> u64 {
        self.pin.misses
    }


    /// Set the MoE-branch gain (see the field and kernels/fwd.hip::vaxpy).
    /// Does the loaded artifact carry the MTP head?
    pub fn has_mtp(&self) -> bool {
        self.pin.mtp.is_some()
    }

    /// Is the routed-expert trace sink active (`--trace`)? `main` reads this to decide
    /// whether speculative decode is available — a verify pass routes twice per layer and
    /// submits the union, which the v2 trace format cannot express.
    pub fn tracing(&self) -> bool {
        self.pin.tracing()
    }


    pub fn set_moe_gain(&mut self, g: f32) {
        if g != 1.0 {
            tracing::warn!("MoE branch gain {g} — EXPERIMENT arithmetic, not a normal run");
        }
        self.moe_gain = g;
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
                1,
                kp)?;
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
                1,
                iqp)?;
            launch_rope(iqp, nh, hd, rope, pos, theta)?; // per head: stride hd, seg rope
            // weights_proj is bf16→f32 [n_heads, hidden] — plain f32 GEMV.
            launch_gemv_f32(xnp, ip.weights_proj, nh, cfg.hidden, 1, iwp)?;
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
    /// The main model: embed `token`, run every layer, leave logits on the device.
    async fn forward(&mut self, token: u32, pos: usize) -> Result<()> {
        self.forward_inner(&[token], pos, Draft::No).await
    }

    /// The MTP head over `tokens.len()` rows. Row `r` is the head element at
    /// `pos + r`: it consumes `(x[r], emb(tokens[r]))` — `x[r]` being the hidden state
    /// row `r` of the last [`GpuEngine::forward`] — and predicts the token at
    /// `pos + r + 1`.
    ///
    /// `x` is restored on the way out, so a draft is invisible to the main residual
    /// stream. What it DOES mutate is the head's own KV slab (correctly — that is the
    /// head's context) and the routed-expert pool (its 8 picks are admitted like any
    /// other layer's, which is also why a rejected draft is not wasted here: the bytes
    /// it fetched stay cached).
    ///
    /// The head's KV must be filled at EVERY position, not just the ones we want a draft
    /// for. Accepting a draft advances `pos` by two, and the element it skipped would
    /// otherwise leave a row of uninitialised device memory inside the window the next
    /// draft attends over. That is what the multi-row form is for: on an accepted pass
    /// both the filler element and the real next draft ride one pass.
    ///
    /// Returns the LAST row's `(draft, confidence)` — the only one anyone asks for.
    async fn mtp_draft(&mut self, tokens: &[u32], pos: usize) -> Result<(u32, f32)> {
        let last = tokens.len() - 1;
        // `d` in the cost model. Spans the whole draft including its argmax D2H, because
        // that sync is on the critical path exactly as much as the kernels are — a
        // pre-draft gate would skip both or neither.
        let t0 = std::time::Instant::now();
        self.forward_inner(tokens, pos, Draft::Head).await?;
        let d = self.argmax_rows(tokens.len())?[last];
        // `argmax_rows` already synced on its D2H, so this read needs no further join.
        self.logits.copy_out_into(&mut self.mtp_host)?;
        let vocab = self.cfg.vocab * 4;
        let conf = top1_prob(&self.mtp_host[last * vocab..(last + 1) * vocab]);
        self.mtp_draft_ns += t0.elapsed().as_nanos();
        self.mtp_draft_n += 1;
        Ok((d, conf))
    }

    /// Score a draft against the token the model actually produced.
    ///
    /// Called on BOTH paths, which is the point: a gated-out step runs a plain pass that
    /// produces the very same `t1` a verify pass would have, so it still learns whether the
    /// draft WOULD have been accepted. The gate therefore never goes blind to the bins it
    /// stops speculating on, and no explore/exploit tempering is needed. A gate placed
    /// BEFORE the draft could not do this — it would never compute `d` to compare.
    ///
    /// Reads row 0's logits when `RIVOLI_MTP_PROBE=1`: bucketing the same accept/reject by
    /// the MAIN MODEL's confidence separates a weak head from a target too uncertain to
    /// predict. Must run before `mtp_draft`, which reuses `mtp_host`.
    fn score_draft(&mut self, ok: bool, conf: f32) -> Result<()> {
        self.mtp_seen += 1;
        self.mtp_hit += u64::from(ok);
        let b = &mut self.mtp_bins[mtp_bin(conf)];
        (b.0, b.1) = (b.0 + 1, b.1 + u64::from(ok));
        if mtp_probe() {
            self.logits.copy_out_into(&mut self.mtp_host)?;
            let tgt = top1_prob(&self.mtp_host[..self.cfg.vocab * 4]);
            let b = &mut self.mtp_tgt_bins[mtp_bin(tgt)];
            (b.0, b.1) = (b.0 + 1, b.1 + u64::from(ok));
        }
        Ok(())
    }

    /// `mtp`: run the MTP head alone instead of the model's layers — see
    /// [`GpuEngine::mtp_draft`] for what that means. Everything else is identical, which
    /// is the point: the head IS a layer, so it reuses the whole per-layer path
    /// (routing, the expert pool, tickets, the two streams, the profile).
    ///
    /// `tokens[r]` sits at position `pos + r`. Every device buffer is row-minor
    /// (`buf[r*dim + i]`) so the batched kernels take a row count and the rest launch `R`
    /// times at an offset; `tokens.len() == 1` reproduces the unbatched pass exactly,
    /// pointer arithmetic and all.
    ///
    /// ROW `r > 0` IS SPECULATIVE. Its KV lands at position `pos + r` like any other, and
    /// discarding it is just not advancing `pos` — the next pass overwrites that row.
    /// There is no compaction and no fixup, because `append_kv` writes by position index
    /// and `attend` reads `0..nr` with `nr` derived from position.
    async fn forward_inner(&mut self, tokens: &[u32], pos: usize, mtp: Draft) -> Result<()> {
        let nrow = tokens.len();
        ensure!(
            (1..=MAXROW).contains(&nrow),
            "forward: {nrow} token rows, but the engine's scratch is allocated for {MAXROW}"
        );
        let token = tokens[0];
        let cfg = self.cfg;
        let eps = cfg.rms_norm_eps as f32;
        // Which pinned layers this pass runs. The head sits one past the model's last.
        let layer_range = match mtp {
            Draft::No => 0..cfg.n_layers,
            _ => cfg.n_layers..cfg.n_layers + 1,
        };
        // The head's entry: `x[r]` ← eh_proj·[enorm(emb(tokens[r])) ‖ hnorm(x[r])]. Done
        // BEFORE the scratch pointers are taken, because it ends by swapping the head's
        // residual buffer into `x` — the last read of the main model's `h` is the hnorm
        // above it.
        let mtp_w = match mtp {
            Draft::No => None,
            Draft::Head => {
                let m = self.pin.mtp.context("--mtp: artifact carries no MTP head")?;
                let emb = self.pin.embed;
                let catp = self.mtp_cat.ptr_mut() as *mut f32;
                let hp = self.x.ptr() as *const f32;
                let dst = self.mtp_x.ptr_mut() as *mut f32;
                // SAFETY: cat is MAXROW·2·hidden f32 scratch; `hp` is the hidden state the
                // previous forward left resident (row r for element pos+r); eh_proj is
                // [hidden, 2·hidden] f32. All offsets are < nrow ≤ MAXROW.
                unsafe {
                    // Per row rather than batched: `cat`'s row stride (2·hidden) differs
                    // from the hnorm SOURCE's (hidden), which a single-stride rmsnorm
                    // cannot express. Four launches on the one layer the head runs — the
                    // eh_proj gemv below IS batched, and that is where the bytes are.
                    for (r, &t) in tokens.iter().enumerate() {
                        let cr = catp.add(r * 2 * cfg.hidden);
                        // Embedding half FIRST, hidden-state half second — the DeepSeek-V3
                        // convention this checkpoint inherits. Not documented anywhere in
                        // the artifact, so it was MEASURED: this order drafts at 53.5%, the
                        // swapped one at 0.0% over 63 drafts. A 0% arm is what makes 53.5%
                        // readable as "the head works", not "the metric is loose".
                        launch_embed_i8_row(emb.packed, emb.scale, t as usize, cfg.hidden, cr)?;
                        launch_rmsnorm(cr, m.enorm, cfg.hidden, eps, cr)?; // in-place
                        launch_rmsnorm(
                            hp.add(r * cfg.hidden),
                            m.hnorm,
                            cfg.hidden,
                            eps,
                            cr.add(cfg.hidden),
                        )?;
                    }
                    launch_gemv_f32(catp, m.eh_proj, cfg.hidden, 2 * cfg.hidden, nrow, dst)?;
                }
                std::mem::swap(&mut self.x, &mut self.mtp_x);
                Some(m)
            }
        };
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
        // out-of-bounds write, so refuse here rather than corrupt device memory. The
        // LAST row is the one that writes furthest.
        ensure!(
            pos + nrow <= self.max_ctx,
            "pos {pos} + {nrow} rows exceeds engine capacity max_ctx={}",
            self.max_ctx
        );

        // Row selection, one entry per token row: dense/streaming is layer-blind
        // (computed once, reused by every layer's attend — dense passes a null rows
        // pointer, the kernel fast path). Dsa/misa selects per full layer inside the loop
        // (it needs the mid-attention q-LoRA residual), signalled by `None` here.
        //
        // Row `r` attends over `pos + r + 1` KV rows: it must SEE the rows the earlier
        // rows of this same pass just appended, which they have because both launches sit
        // on the null stream in append-then-attend order. That differing `nr` IS the
        // causal mask — there is nothing else to add.
        let hoisted_rows: Option<[(*const u32, usize); MAXROW]> = match &self.mode {
            AttnMode::Dense => Some(std::array::from_fn(|r| (std::ptr::null(), pos + r + 1))),
            // Streaming and DSA select a DIFFERENT row set per position, and the engine
            // keeps one `rows_buf` — so a batched pass would need one upload per row and
            // a per-row buffer. Not built: dense is the measured configuration and the
            // work only pays once a batched pass is worth having under those modes too.
            AttnMode::Streaming { sinks, window } => {
                ensure!(
                    nrow == 1,
                    "--attn streaming: batched token rows are not implemented (each row \
                     selects a different KV window); use --attn dense with --mtp"
                );
                streaming_rows(pos + 1, *sinks, *window, &mut self.rows_host);
                let rn = if self.rows_host.len() == pos + 1 {
                    (std::ptr::null(), pos + 1) // all selected → dense fast path
                } else {
                    self.rows_buf.copy_in_at(0, as_le_bytes(&self.rows_host))?;
                    (self.rows_buf.ptr() as *const u32, self.rows_host.len())
                };
                Some([rn; MAXROW])
            }
            AttnMode::Dsa | AttnMode::Misa { .. } => {
                ensure!(
                    nrow == 1,
                    "--attn dsa/misa: batched token rows are not implemented (the indexer \
                     selects per position); use --attn dense with --mtp"
                );
                None
            }
        };

        // Embedding row → x, one per token row.
        // SAFETY: all pointers are device-resident scratch/weights valid for their
        // dims; each launch's inputs are produced by a prior launch on the same
        // (default) stream, so ordering holds; a device_sync precedes every host read.
        if mtp_w.is_none() {
            for (r, &t) in tokens.iter().enumerate() {
                unsafe {
                    launch_embed_i8_row(
                        self.pin.embed.packed,
                        self.pin.embed.scale,
                        t as usize,
                        hidden,
                        xp.add(r * hidden),
                    )?;
                }
            }
        }

        for l in layer_range {
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
            // The four fp8 GEMVs and the absorb carry every row through ONE read of their
            // weights (`nrow`); the norms/ropes/appends launch per row, because their
            // scalar arguments (`pos`) or their strides differ per row and each is a
            // microsecond kernel over ≤6144 floats — ~8 extra enqueues on a ~5 ms layer.
            // SAFETY: see the forward-level note; every pointer is live scratch, and
            // every `.add(r * …)` is within the MAXROW-wide allocation since r < nrow.
            unsafe {
                for r in 0..nrow {
                    launch_rmsnorm(xp.add(r * hidden), input_ln, hidden, eps, xnp.add(r * hidden))?;
                }
                launch_gemv_fp8(
                    xnp, q_a.packed, q_a.scale, q_a.o_dim, q_a.i_dim, q_a.block, nrow, qrp)?;
                for r in 0..nrow {
                    let p = qrp.add(r * cfg.q_lora_rank);
                    launch_rmsnorm(p, q_a_ln, cfg.q_lora_rank, eps, p)?; // in-place
                }
                launch_gemv_fp8(
                    qrp, q_b.packed, q_b.scale, q_b.o_dim, q_b.i_dim, q_b.block, nrow, qp)?;
                launch_gemv_fp8(
                    xnp,
                    kv_a.packed,
                    kv_a.scale,
                    kv_a.o_dim,
                    kv_a.i_dim,
                    kv_a.block,
                    nrow,
                    compp)?;
                for r in 0..nrow {
                    // `comp`'s row stride is kvl+rope but the norm covers only the first
                    // kvl, so this cannot ride a single-stride batched rmsnorm.
                    let c = compp.add(r * (kvl + rope));
                    launch_rmsnorm(c, kv_a_ln, kvl, eps, c)?; // normalize latent (first kvl)
                    launch_rope(c.add(kvl), 1, rope, rope, pos + r, theta)?; // rope the key
                    launch_rope(qp.add(r * h * qh + nope), h, qh, rope, pos + r, theta)?;
                    launch_append_kv(c, c.add(kvl), lc8p, lscalep, rcp, pos + r, kvl, rope, nb)?;
                }
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
                    nrow,
                    qabsp,
                )?;
                for r in 0..nrow {
                    launch_gather_rope(
                        qp.add(r * h * qh),
                        qropep.add(r * h * rope),
                        h,
                        qh,
                        nope,
                        rope,
                    )?;
                }
            }

            // Row selection: hoisted (dense/streaming) or per-layer DSA (needs `qrp`
            // the q-LoRA residual + `xnp` the layer input, both from phase 1). Whether DSA
            // syncs mid-layer is `dsa_select_layer`'s business, not this call site's.
            let rows: [(*const u32, usize); MAXROW] = match hoisted_rows {
                Some(rn) => rn,
                // `nrow == 1` here — the mode guard above refused anything wider.
                None => [self.dsa_select_layer(l, pos, xnp, qrp, indexer_pin)?; MAXROW],
            };

            // --- Attention phase 2: dense flash attend, value + output projection,
            //     residual, pre-MLP norm. ---
            // SAFETY: see the forward-level note; every pointer is live scratch.
            unsafe {
                for (r, &(rows_ptr, nr)) in rows.iter().take(nrow).enumerate() {
                    launch_attend(
                        qabsp.add(r * h * kvl),
                        qropep.add(r * h * rope),
                        lc8p,
                        lscalep,
                        rcp,
                        rows_ptr,
                        h,
                        nr,
                        kvl,
                        rope,
                        nb,
                        scale,
                        clatp.add(r * h * kvl),
                        apartp.add(r * crate::backend::attend_scratch_floats(h, kvl)),
                    )?;
                }
                launch_mla_value_fp8(
                    clatp,
                    kv_b.packed,
                    kv_b.scale,
                    h,
                    nope,
                    vh,
                    kvl,
                    kv_b.block,
                    nrow,
                    ctxp,
                )?;
                launch_gemv_fp8(
                    ctxp,
                    o_proj.packed,
                    o_proj.scale,
                    o_proj.o_dim,
                    o_proj.i_dim,
                    o_proj.block,
                    nrow,
                    subp)?;
                // Both rows in one launch: `x` and `sub` are contiguous row-minor, so the
                // residual add over nrow·hidden elements is the same elementwise op.
                launch_vadd(xp, subp, nrow * hidden)?; // residual
                for r in 0..nrow {
                    // pre-MLP norm → xn
                    launch_rmsnorm(xp.add(r * hidden), post_ln, hidden, eps, xnp.add(r * hidden))?;
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
                        nrow,
                        gp)?;
                    launch_gemv_fp8(
                        xnp,
                        m.up.packed,
                        m.up.scale,
                        m.up.o_dim,
                        m.up.i_dim,
                        m.up.block,
                        nrow,
                        up)?;
                    // One launch: elementwise over contiguous row-minor buffers.
                    launch_swiglu(gp, up, nrow * inter, gp)?; // in place: h = silu(gate)*up
                    launch_gemv_fp8(
                        gp,
                        m.down.packed,
                        m.down.scale,
                        m.down.o_dim,
                        m.down.i_dim,
                        m.down.block,
                        nrow,
                        outp)?;
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
                unsafe { launch_gemv_f32(xnp, gate_w, cfg.n_experts, hidden, nrow, glp)? };
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
                // Routing is a pure function of (logits, bias, top_k) — it does NOT
                // consult residency. `top-m` used to make it cache-conditional; removing it
                // is what makes every cache change output-neutral by construction
                // (math.rs `inv_1_routing_never_consults_the_cache`).
                let t_route = std::time::Instant::now();
                // One routing per token row, then the UNION. The rows are INDEPENDENT
                // routers over independent hidden states — batching them is a launch
                // decision, not a modelling one, and each row's weights are normalized
                // over its OWN top-k exactly as an unbatched pass would.
                //
                // Descending, so `scores`/`choice` are left holding ROW 0 for the
                // scorer and the trace window below: those measure the router against the
                // real token, and row 0 is the real token.
                let ne = cfg.n_experts * 4;
                for r in (0..nrow).rev() {
                    route_into(
                        &self.gl_host[r * ne..(r + 1) * ne],
                        self.pin.moe_bias(l),
                        cfg.top_k,
                        &mut self.scores,
                        &mut self.choice,
                        &mut self.sel_row[r],
                    );
                    // Routed weights: sigmoid score, sum-normalized over THIS row's picks,
                    // then scaled. Computed here rather than after the union because
                    // `scores` belongs to row `r` only until the next iteration.
                    let wr = &mut self.wrow[r];
                    wr.clear();
                    for &e in &self.sel_row[r] {
                        wr.push(self.scores[e]);
                    }
                    let mut sm: f32 = wr.iter().sum();
                    if cfg.norm_topk_prob {
                        sm += 1e-20;
                        for wi in wr.iter_mut() {
                            *wi /= sm;
                        }
                    }
                    for wi in wr.iter_mut() {
                        *wi *= cfg.routed_scale as f32;
                    }
                }
                // Row 0's picks first and in order, so an `nrow == 1` pass submits exactly
                // what the unbatched engine submitted — same experts, same order, same
                // descriptor indices.
                self.union.clear();
                for r in 0..nrow {
                    for &e in &self.sel_row[r] {
                        if !self.union.contains(&e) {
                            self.union.push(e);
                        }
                    }
                }
                // The host half of the route region, stamped rather than derived.
                let e_route = std::time::Instant::now();
                self.prof.cpu_route_ns += e_route.duration_since(t_route).as_nanos();
                crate::telemetry::spans::record("cpu/route-into", "decode", t_route, e_route);
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
                // Residency must be sampled BEFORE submit_layer: it allocates slots for
                // misses, so afterwards everything reads as resident. A bitmask, not a
                // Vec — this runs 78x per token and top_k is 8.
                let warm_mask: u64 = if crate::telemetry::spans::enabled() {
                    self.union
                        .iter()
                        .take(64)
                        .enumerate()
                        .filter(|&(_, &e)| self.pin.resident(l, e))
                        .fold(0u64, |m, (i, _)| m | (1 << i))
                } else {
                    0
                };
                let t_sub = std::time::Instant::now();
                // The UNION, not one row's picks: every expert any row routed to must be
                // resident before the batch launches. Rows overlap ~31% (measured over
                // 268 tokens x 75 layers), so 2 rows submit ~13.5 experts rather than 16.
                let mut signals = self.pin.submit_layer(
                    l,
                    &self.union,
                    &self.window,
                    &self.choice,
                    &mut self.mlps_vq,
                    &mut self.fmt,
                    &mut self.tickets,
                )?;
                // Pure host work: residency lookups + policy bookkeeping + read specs.
                // It only ENQUEUES the reads; the reaper thread does the waiting.
                let e_sub = std::time::Instant::now();
                self.prof.cpu_submit_ns += e_sub.duration_since(t_sub).as_nanos();
                crate::telemetry::spans::record("cpu/submit-layer", "decode", t_sub, e_sub);
                // Residency x format, the pair that explains a layer's cost. `fmt` is
                // filled by submit_layer in `union` order (the shared expert is pushed
                // after, so `take(union.len())` keeps this to the routed picks).
                if crate::telemetry::spans::enabled() {
                    let mut st = crate::telemetry::spans::LayerState {
                        tok: pos as u32,
                        layer: l as i32,
                        ..Default::default()
                    };
                    for (i, &i4) in self.fmt.iter().take(self.union.len()).enumerate() {
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
                // Scatter each row's weights into the `[descriptor][row]` matrix the
                // kernel reads as `wexpert[e*R + t]`. A row that did not route to a union
                // expert leaves 0.0 there, and `moe_down_vq` SKIPS a zero weight — so a
                // row's result is EXACTLY its own 8 + shared, whatever else the union
                // dragged in. That is what makes row 0 of a batched pass bit-identical to
                // an unbatched one, and the skip is correctness rather than thrift:
                // `0 * dv` with a non-finite `dv` is NaN, which the fixed-point clamp
                // would turn into a finite extreme.
                // Driven from the union rather than from each row's picks, so "this row
                // did not route here" is the natural `None` and leaves the 0.0 the resize
                // put there — no unwrap, and no way to silently drop a weight.
                self.w.clear();
                self.w.resize(self.union.len() * nrow, 0.0);
                for (u, &e) in self.union.iter().enumerate() {
                    for r in 0..nrow {
                        if let Some(i) = self.sel_row[r].iter().position(|&x| x == e) {
                            self.w[u * nrow + r] = self.wrow[r][i];
                        }
                    }
                }
                // VQ routed descriptors + the folded VQ shared expert (resident, so its
                // load is `ready()`).
                self.descs_vq.clear();
                for m in &self.mlps_vq {
                    self.descs_vq.push(desc_of_vq(m));
                }
                if let Some(s) = shared {
                    self.descs_vq.push(desc_of_vq(&s));
                    // Weight 1.0 for EVERY row: the shared expert is unconditional, so it
                    // contributes to each row of the batch.
                    self.w.extend(std::iter::repeat_n(1.0, nrow));
                    self.fmt.push(self.pin.shared_i4());
                    // The shared expert is in the RESIDENT tier, never streamed, so its
                    // dependency is already satisfied. It must still grow `tickets` with
                    // `fmt`/`descs` or the launch loop indexes past the end (it did once:
                    // "len is 8 but the index is 8", because the old mask covered only the
                    // 8 routed picks).
                    self.tickets.push(Ticket::RESIDENT);
                    signals.push(Signal::ready());
                }
                let ndesc = self.descs_vq.len();
                self.descs_buf
                    .copy_in_at(0, as_le_bytes(&self.descs_vq))?;
                self.wexpert_buf.copy_in_at(0, as_le_bytes(&self.w))?;
                // Two streams, no join: residents run on the compute stream while the
                // misses' bytes are still landing on the miss stream, and BOTH atomically
                // accumulate into the same fixed-point `moe_acc` row. Completion order is
                // irrelevant because integer addition is associative — where the old f32
                // partial slab needed disjoint rows plus a reduce plus a cross-stream wait
                // to say the same thing. mlp bucket = the whole overlapped wall.
                let x_c = xnp as *const f32;
                let h_c = self.moe_h.ptr_mut() as *mut f32;
                let acc_c = self.moe_acc.ptr_mut() as *mut u64;
                // SAFETY: `moe_acc` is MOE_ACC_ROWS·MAXROW·hidden u64, laid out
                // [stream][token row][hidden]; this is the miss stream's block, in bounds
                // for nrow ≤ MAXROW.
                let acc_miss = unsafe { acc_c.add(nrow * hidden) };
                // One descriptor buffer for both kernels — the int4 kernel reinterprets
                // the same six-pointer bytes (at its slot offsets).
                let descs_ptr = self.descs_buf.ptr() as *const ExpertDesc;
                // Per-expert format (routed experts from their slab; shared appended
                // above). Cloned so the expert stream owns it (the small bool vec moves
                // into the async closure). Hybrid mixes int4/vq3 within one batch.
                let fmt = self.fmt.clone();
                // Cloned alongside `fmt`: the launch loop indexes both while `self.pin` is
                // borrowed for `wait_on`.
                let tickets = self.tickets.clone();
                let w_ptr = self.wexpert_buf.ptr() as *const f32;
                let (cb0, cb1, cb2) = (self.codebooks[0], self.codebooks[1], self.codebooks[2]);
                let cs_raw = self.compute_stream.raw();
                let ms_raw = self.miss_stream.raw();
                let inter = cfg.moe_inter;
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
                    tickets.len(), ndesc,
                    "every descriptor needs a ticket (routed picks + the shared expert)"
                );
                // TICKETED DATAFLOW. Every expert is enqueued behind a DEVICE-SIDE wait on
                // its own data, in one loop, with no branch on residency and no host round
                // trip. Resident, missing and in-flight take the same path — a resident
                // expert simply carries `Ticket::RESIDENT` (value 0, satisfied on arrival).
                //
                // This replaces a `hit: Vec<bool>` that told this loop whether to await, and
                // deleting it is the point rather than a side effect. That mask was a second
                // host-side encoding of "is this data ready?", and when it disagreed with the
                // Signal it won SILENTLY: a `hit` expert launched with no wait at all, so a
                // slot still being written could be marked ready and read as garbage. A
                // ticket cannot disagree with anything — it IS the dependency, and
                // `wait_on` is the only way to consume one, so "launched without waiting" is
                // no longer expressible here.
                //
                // The wait must be enqueued BEFORE the producer has run (the reaper has not
                // seen these completions yet), which is exactly what `hipStreamWaitEvent`
                // cannot do and `hipStreamWaitValue64` can — tested as INV-4 on both
                // backends. Do not "simplify" this to events.
                //
                // Runs of consecutive experts sharing a format still batch into one dispatch
                // (`moe_expert_range` computes `e = e_start + row/inter`, so an `e_count > 1`
                // dispatch is bit-identical to that many single ones). What no longer gates
                // the batching is residency — only the format, and whether a wait had to be
                // enqueued between them.
                // ORDER MATTERS, and getting it wrong cost 20% before this comment existed.
                // The compute stream is FIFO: enqueueing in `sel` order puts every resident
                // expert BEHIND the first miss's wait, so nothing computes while that fetch
                // is in flight — which is the overlap the whole engine is built on. Measured
                // 3.05 -> 2.44 tok/s, `moe` 210 -> 289 ms.
                //
                // So residents go first, misses after. Reordering LAUNCHES is safe by
                // construction: each expert writes its own partial row and `moe_reduce` sums
                // `0..ndesc` in fixed order, so the result is bit-identical whatever order
                // the partials are produced in.
                //
                // NOTE this branches on `ticket.is_resident()`, and that is NOT the `hit`
                // mask coming back. The difference is what the branch controls. The mask
                // decided whether to WAIT — a wrong bit meant a kernel ran on unwritten
                // memory, silently. This decides only the ORDER of launches that each
                // enqueue their wait unconditionally, so a wrong bit costs throughput and
                // cannot cost correctness. Same data, and now it cannot gate the dependency.
                let mut i = 0usize;
                while i < ndesc {
                    if !tickets[i].is_resident() {
                        i += 1;
                        continue;
                    }
                    let f = fmt[i];
                    let mut j = i;
                    while j < ndesc && tickets[j].is_resident() && fmt[j] == f {
                        j += 1;
                    }
                    // Enqueued anyway rather than skipped: `wait_on` is the only way to
                    // consume a ticket, and a resident one costs nothing (it short-circuits
                    // on value 0). Keeping the call unconditional is what makes "every
                    // launch is behind its dependency" true by reading the code, not by
                    // trusting this loop's classification.
                    for k in i..j {
                        self.pin.wait_on(tickets[k], cs_raw)?;
                    }
                    // SAFETY: descs/codebooks resident; every expert in [i, j) has its
                    // dependency enqueued above; h/part device scratch; cs_raw live.
                    unsafe {
                        if f {
                            launch_moe_expert_range_i4(
                                x_c, hidden, inter, i, j - i, descs_ptr, w_ptr, h_c, acc_c,
                                nrow, cs_raw,
                            )?;
                        } else {
                            launch_moe_expert_range(
                                x_c, hidden, inter, i, j - i, descs_ptr, cb0, cb1, cb2, w_ptr,
                                h_c, acc_c, nrow, cs_raw,
                            )?;
                        }
                    }
                    i = j;
                }
                // THEN the misses — on the MISS STREAM, not this one. A stream is FIFO, so
                // a wait enqueued here is only REACHED after the residents above finish, and
                // the GPU's wake latency then lands on the critical path: measured +382 us
                // per layer-with-misses (a 1-miss layer cost +145 us over 0-miss when the
                // host gated it, +527 us when this stream did). On its own stream the same
                // wait starts at the top of the layer and that latency is absorbed by the
                // ~1557 us of resident compute running beside it.
                //
                // Misses accumulate into `moe_acc` ROW 1, residents into row 0, and the
                // drain sums the two. No ordering between the streams is required at all —
                // integers associate, so the split is a CONTENTION fix, not a correctness
                // one (`MOE_ACC_ROWS` has the measurement). Cross-queue visibility was
                // probed rather than assumed (docs/probes/waitvalue_visibility.hip): 0
                // mismatches over 8.4e8 checks.
                for e in 0..ndesc {
                    if tickets[e].is_resident() {
                        continue;
                    }
                    self.pin.wait_on(tickets[e], ms_raw)?;
                    // SAFETY: as above; this expert's bytes are gated by the wait just
                    // enqueued on the same stream.
                    unsafe {
                        if fmt[e] {
                            launch_moe_expert_range_i4(
                                x_c, hidden, inter, e, 1, descs_ptr, w_ptr, h_c, acc_miss, nrow,
                                ms_raw,
                            )?;
                        } else {
                            launch_moe_expert_range(
                                x_c, hidden, inter, e, 1, descs_ptr, cb0, cb1, cb2, w_ptr, h_c,
                                acc_miss, nrow, ms_raw,
                            )?;
                        }
                    }
                }
                // THERE IS NO JOIN HERE ANY MORE, and its absence is the point. The
                // compute stream used to wait on a timeline the miss stream signalled,
                // because `moe_reduce` could not start until every partial row existed. With
                // a fixed-point accumulator nothing between the two streams needs ordering:
                // the only consumer of all experts is the drain, and that already sits
                // behind the end-of-layer barrier.
                // The load Signals are no longer awaited per expert — the GPU gates itself
                // now. They remain the reaper's error/teardown channel (it resolves them on
                // poison so nothing hangs), and NOT awaiting them is what removes ~9 host
                // round trips per layer: the measured cost of that gating was the MoE phase
                // running at 38.8 GB/s in-engine against 91.8 GB/s for the same kernels in
                // `examples/moe_bench.rs`.
                drop(signals);
                // BOTH streams, because neither one waits for the other any more. The
                // drain below is the only thing that needs every expert, and it runs after
                // this. `moe_ev_start`/`_end` still bracket honestly: `_end` records on an
                // idle compute stream once both awaits have returned, so its timestamp is
                // after the miss stream's experts too — the bracket did NOT narrow when the
                // join left, which is the mis-measurement this file has shipped twice.
                stream_signal(cs_raw)?.await;
                stream_signal(ms_raw)?.await;
                self.prof.moe_wall_ns += tm.elapsed().as_nanos();
                self.moe_ev_end.record(cs_raw)?;
            }
            // Residual add of the MLP contribution. On a MoE layer the drain IS the add:
            // it converts the fixed-point accumulator straight into `x` and resets it, so
            // the conversion costs no extra pass. `--moe-gain` folds into the same multiply
            // and applies ONLY here — the 3 dense layers must not be attenuated too, or
            // "the MoE branch is too strong" and "the MLP branch is" stop being
            // distinguishable.
            //
            // No barrier is needed before this despite both MoE streams having written
            // `moe_acc`: they were awaited above.
            // SAFETY: `x` is `hidden` device f32; `moe_acc` is `hidden` u64 and every
            // stream that touched it has completed; `moe_out` is `hidden` f32.
            unsafe {
                match dense_mlp.is_none() {
                    // `nrow·hidden` in one launch: the accumulator's token and hidden
                    // axes are contiguous, so the drain never has to know they are two.
                    true => launch_moe_acc_drain(
                        xp,
                        self.moe_acc.ptr_mut() as *mut u64,
                        nrow * hidden,
                        MOE_ACC_ROWS,
                        self.moe_gain,
                        std::ptr::null_mut(),
                    )?,
                    false => launch_vadd(xp, self.moe_out.ptr() as *const f32, nrow * hidden)?,
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
                    nrow * hidden,
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
        // `shared_head.norm` for the MTP head; `lm_head` itself is SHARED with the main
        // model (the checkpoint ships no `shared_head.head.weight`).
        let tail_norm = mtp_w.map_or(self.pin.final_norm, |m| m.shared_norm);
        unsafe {
            for r in 0..nrow {
                launch_rmsnorm(xp.add(r * hidden), tail_norm, hidden, eps, xnp.add(r * hidden))?;
            }
            let head = self.pin.lm_head;
            // Batched: lm_head is 952 MB of int8, the single largest read in the pass, and
            // a second row through the same read costs one more f32 multiply per column.
            launch_gemv_i8(
                xnp,
                head.packed,
                head.scale,
                head.o_dim,
                head.i_dim,
                nrow,
                self.logits.ptr_mut() as *mut f32,
            )?;
        }
        // Give the main model its residual stream back.
        if mtp_w.is_some() {
            std::mem::swap(&mut self.x, &mut self.mtp_x);
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
    #[cfg(feature = "teacher-forcing")]
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
                // `logits` is MAXROW rows wide; a single-row forward writes only row 0.
                ensure!(host.len() >= vocab * 4, "short logits D2H");
                out.push(crate::eval::nll_of(&host[..vocab * 4], next as usize)?);
            }
            Ok(out)
        })?;
        Ok(out)
    }

    /// Greedy argmax over row 0 of the device logits — the only row a single-row forward
    /// wrote.
    fn argmax(&mut self) -> Result<u32> {
        Ok(self.argmax_rows(1)?[0])
    }

    /// Greedy argmax over each of the pass's `n` logit rows — reduced ON DEVICE, so only
    /// [`ARGMAX_BYTES`] come back per pass however many rows it carried. The kernel
    /// reproduces the host fold exactly (strict `>`: ties keep the lowest index, NaN never
    /// wins), returning `logits[best]` so the finiteness bail is the same
    /// `!value.is_finite()` check.
    ///
    /// ONE D2H for every row: the rows are independent reductions but their results are
    /// adjacent, and a per-row copy-out would add a blocking join per row to a path whose
    /// whole point is to amortise joins.
    fn argmax_rows(&mut self, n: usize) -> Result<[u32; MAXROW]> {
        debug_assert!((1..=MAXROW).contains(&n));
        for r in 0..n {
            // SAFETY: logits is MAXROW·vocab device f32 (written + joined); argmax_dev
            // owns ARGMAX_BYTES, and r < n ≤ MAXROW keeps both slots in bounds.
            unsafe {
                launch_argmax(
                    (self.logits.ptr() as *const f32).add(r * self.cfg.vocab),
                    self.cfg.vocab,
                    self.argmax_dev.ptr_mut().add(r * 8) as *mut i32,
                    self.argmax_dev.ptr_mut().add(r * 8 + 4) as *mut f32,
                )?;
            }
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
            ARGMAX_BYTES,
            "argmax result must be MAXROW*[idx | val] + a nonfinite tag"
        );
        let word = |o: usize| {
            let b = &self.argmax_host[o..o + 4];
            [b[0], b[1], b[2], b[3]]
        };
        let mut out = [0u32; MAXROW];
        for (r, o) in out.iter_mut().enumerate().take(n) {
            let idx = i32::from_le_bytes(word(r * 8));
            let val = f32::from_le_bytes(word(r * 8 + 4));
            if !val.is_finite() {
                // The tag rode the same D2H, so this costs nothing and turns "somewhere in
                // 78 layers x every position" into a coordinate.
                let tag = u32::from_le_bytes(word(MAXROW * 8));
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
                bail!("logits are non-finite (NaN/Inf in the GPU forward pass), row {r}: {where_}");
            }
            debug_assert!(idx >= 0, "argmax returned negative index {idx}");
            *o = idx as u32;
        }
        Ok(out)
    }

    /// Greedy-decode up to `ngen` tokens continuing `prompt_ids`, stopping on any
    /// `eos`. Returns the generated ids + the always-on decode-loop [`ProfileSummary`]
    /// (also logged as the PROFILE line; `main` feeds it to the OTLP span).
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        ngen: usize,
        eos: &[u32],
        mtp: bool,
        // Speculate only above this draft confidence; see `--mtp-min-conf`. 0 = never gate.
        min_conf: f32,
    ) -> Result<(Vec<u32>, ProfileSummary)> {
        ensure!(!prompt_ids.is_empty(), "empty prompt");
        // Both preconditions are resolved by `main` (which downgrades to sequential
        // decode and says why), so reaching either of these is a caller bug rather than a
        // user one — but they stay, because both fail SILENTLY otherwise: a missing head
        // would null-deref in the draft path, and tracing would emit per-layer selections
        // that read as a routing the model never made.
        ensure!(
            !mtp || self.has_mtp(),
            "speculative decode: this artifact carries no MTP head (reconvert with a \
             current bin/convert)"
        );
        ensure!(
            !mtp || !self.pin.tracing(),
            "speculative decode with --trace: a verify pass routes twice per layer and \
             submits the union, which the v2 trace format cannot express"
        );
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
                // Build the head's KV alongside the model's: MTP element i is
                // (h_i, emb(t_{i+1})) AT POSITION i+1, so the prompt supplies every
                // element but the last, whose successor has not been sampled yet.
                if mtp && let Some(&next) = prompt_ids.get(pos + 1) {
                    self.mtp_draft(&[next], pos + 1).await?;
                }
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
            #[cfg(feature = "trace")]
            let mut win_gen = 0usize;
            // `cur` is the token AT `pos`, decided but not yet fed through the model;
            // `x` row 0 holds h_{pos-1} and the head's KV is filled through row pos-1.
            // `draft` is the head's prediction for pos+1, made from that same h.
            let mut cur = self.argmax()?;
            let mut draft = match mtp {
                true => Some(self.mtp_draft(&[cur], pos).await?),
                false => None,
            };
            // Emit and report whether the loop is done: eos ends it, and so does the
            // token budget. A speculative iteration emits TWO tokens, so this cannot be
            // the loop counter it used to be.
            let emit = |g: &mut Vec<u32>, t: u32| {
                if eos.contains(&t) {
                    return true;
                }
                g.push(t);
                g.len() >= ngen
            };
            let mut _i = 0usize;
            loop {
                if let Some(hb) = &self.heartbeat {
                    hb.beat();
                }
                if emit(&mut generated, cur) {
                    break;
                }
                match draft {
                    None => {
                        self.forward(cur, pos).await?;
                        cur = self.argmax()?;
                        pos += 1;
                    }
                    // GATED OUT. The head is not confident enough for a verify pass to pay
                    // for itself, so run one row and take one token. The draft is already
                    // computed and is scored anyway — a plain pass yields the same `t1` the
                    // verify pass would have, so this costs nothing and keeps the histogram
                    // honest. Output is unaffected either way: both paths are exact, and
                    // the gate moves only which of them pays for the second row.
                    Some((d, conf)) if conf < min_conf => {
                        self.forward(cur, pos).await?;
                        let t1 = self.argmax()?;
                        self.score_draft(d == t1, conf)?;
                        // Same call the reject path makes: the head's KV must stay
                        // hole-free, and `x` row 0 holds h_pos for it to consume.
                        draft = Some(self.mtp_draft(&[t1], pos + 1).await?);
                        cur = t1;
                        pos += 1;
                    }
                    // THE VERIFY PASS. Two rows: the real token at `pos` and the draft at
                    // `pos + 1`, through one read of every weight. Row 0's logits give the
                    // TRUE token at pos+1; row 1's give the true token at pos+2 — but only
                    // if the draft it was computed from was right, which is exactly what
                    // comparing row 0's answer to the draft tests.
                    Some((d, conf)) => {
                        self.forward_inner(&[cur, d], pos, Draft::No).await?;
                        let rows = self.argmax_rows(2)?;
                        let (t1, t2) = (rows[0], rows[1]);
                        let ok = d == t1;
                        self.mtp_verify += 1;
                        self.score_draft(ok, conf)?;
                        // The head's element at pos+1 — needed whether or not the draft
                        // held: on a reject it IS the next draft, and on an accept it is
                        // the KV row that pos+2's draft will attend over. Batched with
                        // pos+2's element on an accept, since both inputs are now known
                        // (rows 0 and 1 of `x` are h_pos and h_{pos+1}).
                        draft = Some(match ok {
                            false => self.mtp_draft(&[t1], pos + 1).await?,
                            true => self.mtp_draft(&[t1, t2], pos + 1).await?,
                        });
                        if ok {
                            // pos+1's KV was computed from the right token, so keep it and
                            // take pos+2's token for free.
                            if emit(&mut generated, t1) {
                                break;
                            }
                            cur = t2;
                            pos += 2;
                        } else {
                            // Rejecting is not advancing `pos`. Row 1's KV at pos+1 stays
                            // where it is and the next pass overwrites it — `append_kv`
                            // writes by position and `attend` reads 0..nr, so there is
                            // nothing to compact. What is NOT rolled back is the expert
                            // pool: the rejected row's fetched bytes stay cached, which is
                            // the favourable direction.
                            cur = t1;
                            pos += 1;
                        }
                    }
                }
                _i += 1;
                // Bound trace loss to one token: the watchdog exits without destructors,
                // so BufWriter's Drop is not a guarantee. No-op when not tracing.
                self.pin.flush_trace()?;
                #[cfg(feature = "trace")]
                if _i % WIN == 0 {
                    let dt = win_t.elapsed().as_secs_f64();
                    let (dh, dm) = (self.pin.hits - win_hit, self.pin.misses - win_miss);
                    let hit_pct = 100.0 * dh as f64 / (dh + dm).max(1) as f64;
                    // Tokens per PASS is now ≥ 1, so the window's rate has to divide the
                    // tokens actually emitted by the wall, not WIN by it.
                    let dg = generated.len() - win_gen;
                    tracing::info!(
                        "  tok {}/{ngen} ({_i} passes): {:.3} tok/s (window), hit {hit_pct:.1}%",
                        generated.len(),
                        dg as f64 / dt.max(1e-9),
                    );
                    win_t = std::time::Instant::now();
                    win_gen = generated.len();
                    (win_hit, win_miss) = (self.pin.hits, self.pin.misses);
                }
            }
            Ok::<_, anyhow::Error>((hit0, miss0, fetch0, io0, decode_wall))
        })?;
        self.prof.wall_ns = decode_wall.elapsed().as_nanos();
        self.prof.tokens = generated.len() as u64;
        let bytes_per_expert = crate::artifact::quant::vq_expert_bytes(self.cfg.hidden, self.cfg.moe_inter);
        // The accurate async-side decomposition: reaper fetch wall + measured io-wait,
        // taken at the ring. The tokio-metrics `idle_ns`/`poll_ns` pair this comment used
        // to name went away with the ticketed dataflow (see `Prof`) — and the DEPENDENCY
        // outlived the last use of it until 2026-07-31, along with `tokio-stream`, which
        // had none at all.
        let summary = self.prof.summary(
            self.pin.hits - hit0,
            self.pin.misses - miss0,
            bytes_per_expert,
            self.pin.fetch_ns().saturating_sub(fetch0),
            self.pin.io_wait_ns().saturating_sub(io0),
        );
        summary.report();
        // Zero on every run measured so far, and that IS the claim: the ticket gate on
        // staging-slot hand-out is satisfied on arrival for every read the engine issues,
        // because each is awaited inside its issuing layer. Reported when it is not, since
        // the alternative is the gate silently becoming the bottleneck it exists to prevent.
        if self.pin.slot_stalls() > 0 {
            tracing::warn!(
                "staging-slot stalls: {} — a layer waited for a slot whose bounce copy had \
                 not retired. The ring is undersized for the lookahead.",
                self.pin.slot_stalls()
            );
        }
        if self.mtp_seen > 0 {
            // tokens/pass is MEASURED, not projected. Every iteration scores exactly one
            // draft — gated out or not — so `mtp_seen` IS the pass count and
            // `generated / mtp_seen` is the speedup over a sequential loop with everything
            // else held equal. It is no longer `1 + accept_rate`: a gated-out pass emits
            // one token however good its draft looked.
            tracing::info!(
                "MTP: {}/{} drafts accepted ({:.1}%) — {:.3} tokens/pass over {} passes",
                self.mtp_hit,
                self.mtp_seen,
                100.0 * self.mtp_hit as f64 / self.mtp_seen as f64,
                generated.len() as f64 / self.mtp_seen as f64,
                self.mtp_seen,
            );
            // What the gate actually did. `verify` counts the 2-row passes; the rest ran
            // one row and cost a sequential pass, so this is the `g` in the cost model.
            tracing::info!(
                "  MTP gate: {}/{} drafts speculated ({:.0}%) at min_conf {min_conf}",
                self.mtp_verify,
                self.mtp_seen,
                100.0 * self.mtp_verify as f64 / self.mtp_seen.max(1) as f64,
            );
            // Bucketed by the draft's own confidence: does it separate well enough to gate
            // on? A flat column of accept% means it does not, and no threshold helps.
            //
            // CORRECTED 2026-07-31: this comment used to read "still not worth gating: at
            // c ≈ 1.08 the break-even accept rate is ~8%, which the worst bin clears."
            // That `c` came from the pre-implementation estimate that was wrong (it applied
            // the union factor to fetch only — see §13). The measured verify pass costs
            // c = 1.53, so break-even is ~53% and only the top bins clear it. Gating is
            // worth building; the histogram below is what sizes it.
            let hist = |bins: &[(u64, u64); MTP_BINS]| {
                bins.iter()
                    .enumerate()
                    .map(|(i, &(n, ok))| {
                        let lo = i as f64 / MTP_BINS as f64;
                        match n {
                            0 => format!("p{lo:.1}+: -"),
                            _ => format!("p{lo:.1}+: {:.0}%(n={n})", 100.0 * ok as f64 / n as f64),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            tracing::info!("  MTP accept by draft confidence: {}", hist(&self.mtp_bins));
            // `d` in the cost model, as ms and as a fraction of a whole pass. A pre-draft
            // gate can only ever save this much, so if it is ~1% it is not worth building.
            if self.mtp_draft_n > 0 {
                let draft_ms = self.mtp_draft_ns as f64 / 1e6;
                let pass_ms = self.prof.wall_ns as f64 / 1e6 / self.mtp_seen.max(1) as f64;
                tracing::info!(
                    "  MTP draft cost: {:.1} ms/draft over {} drafts = {:.1}% of decode wall \
                     ({:.1} ms/pass) — this is `d`",
                    draft_ms / self.mtp_draft_n as f64,
                    self.mtp_draft_n,
                    100.0 * draft_ms / (self.prof.wall_ns as f64 / 1e6),
                    pass_ms,
                );
            }
            // The discriminator. If the top bin here is ~90%+, the head is fine wherever
            // the target is decidable and the ceiling is the model's own entropy — do not
            // de-quantize layer 78. If it sits near the overall rate, the head is weak.
            if self.mtp_tgt_bins.iter().any(|&(n, _)| n > 0) {
                tracing::info!(
                    "  MTP accept by TARGET confidence: {}",
                    hist(&self.mtp_tgt_bins)
                );
            }
        }
        Ok((generated, summary))
    }
}
