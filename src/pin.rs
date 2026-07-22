//! The resident weight set: places every weight the forward pass reads into the
//! [`DeviceTier`] once at startup and resolves each to a raw device pointer, so
//! per-token decode never touches the host for weights (PLAN.md D1/P3).
//!
//! Always resident (read every token): the per-layer norms, the full attention
//! stack (q_a/q_b/kv_a/kv_b/o_proj + their layernorms), the dense-layer MLPs, and
//! — for MoE layers — the router gate, its bias, and the shared expert. Its exact
//! device footprint is computed from `cfg` at build ([`resident_bytes`]; ~9-10 GiB
//! for GLM-5.2) and logged, so the tier is sized to what it holds and the rest of
//! the device budget grows the LRU. The 256 routed experts/layer are served by an
//! ADAPTIVE LRU pool
//! ([`Lru`] + one VMM slab): a hit reuses the resident slot; a miss evicts the
//! coldest and streams the expert in via io_uring O_DIRECT. The LRU adapts to the
//! actual workload (online priming) — measured ~74% hit vs ~43% for a static
//! frequency pin of the same size.
//!
//! `rocm`-only: without a device there is nothing to pin.
#![cfg(feature = "rocm")]

use crate::cache;
use crate::device::{DeviceTier, VmmBuf, mem_info};
use crate::model::ModelConfig;
use crate::snapshot::{Dtype, Snapshot};
use crate::stream::{ALIGN, Sqpoll, Streamer, slot_span};
use crate::usage::Usage;
use anyhow::{Context, Result, ensure};
use std::os::fd::RawFd;

/// The coalesced cold reads that stream one routed expert into a pool slot,
/// resolved ONCE at [`Pin::build`] so the per-miss path never re-derives them.
/// The colibri converter writes each expert's three projection weights CONTIGUOUS
/// on disk in down|gate|up order (and likewise the three scales), so all three
/// weights normally come in ONE ~18.9 MB O_DIRECT read — which hits the drive's
/// large-read bandwidth (~5 GB/s) instead of the ~3.6 GB/s three separate 6.3 MB
/// reads get — and all three scales in one tiny read (2 reads/expert). When an
/// expert straddles a shard-file boundary (~0.5% of them) the run splits there
/// into 2-3 reads; [`build_moe_table`] plans the runs so each is one contiguous
/// O_DIRECT read landing at its own block-aligned slot destination.
///
/// - `reads[k]` = `(fd, file_begin, len, dst)` — a fixed contiguous run and its
///   block-aligned slot-relative destination. `len == 0` marks an unused entry
///   (most experts use 2 of the 6). The fd is O_DIRECT or buffered per
///   `Pin::build`'s `direct_io`; offset/len are identical either way.
/// - `off` = the six slot-relative offsets where each projection's USEFUL bytes
///   land, in slot-layout order gate packed, gate scale, up packed, up scale, down
///   packed, down scale (the order [`Pin::submit_layer`] builds descriptors in).
///
/// All cfg-dim cross-checks (packed/scale byte lengths) are done at build when
/// this table is populated, so the miss path trusts it blindly.
#[derive(Clone, Copy)]
struct ExpertReads {
    reads: [(RawFd, usize, usize, usize); 6],
    off: [usize; 6],
}

/// A resolved quantized weight matrix in the tier: device pointers + dims. The
/// int4/int8 distinction lives in which launcher consumes it (`launch_gemv_i4`
/// vs `_i8`), not in the layout — both are packed bytes + per-row f32 scales.
#[derive(Clone, Copy)]
pub struct Weight {
    pub packed: *const u8,
    pub scale: *const f32,
    pub o_dim: usize,
    pub i_dim: usize,
}

/// A SwiGLU MLP's three int4 projections, resolved.
#[derive(Clone, Copy)]
pub struct Mlp {
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

/// One layer's MLP: dense for the first `dense_layers`, MoE after.
pub enum LayerMlp {
    Dense(Mlp),
    Moe {
        gate_w: *const f32, // router gate [n_experts, hidden] (F32), device
        shared: Mlp,        // shared expert (always resident)
    },
}

/// A full layer's resident DSA lightning-indexer weights (bf16 projections +
/// the widened f32 k_norm). Only "full" layers own one; shared layers reuse a
/// preceding full layer's selection and carry `None`.
#[derive(Clone, Copy)]
pub struct IndexerPin {
    pub wk: *const u16,           // [index_head_dim, hidden] bf16
    pub wq_b: *const u16,         // [n_heads·head_dim, q_lora_rank] bf16
    pub weights_proj: *const u16, // [n_heads, hidden] bf16
    pub k_norm_w: *const f32,     // [index_head_dim] (widened at placement)
    pub k_norm_b: *const f32,     // [index_head_dim]
}

/// One layer's resolved weights.
pub struct LayerPin {
    pub input_ln: *const f32,
    pub post_ln: *const f32,
    pub q_a: Weight,
    pub q_a_ln: *const f32,
    pub q_b: Weight,
    pub kv_a: Weight,
    pub kv_a_ln: *const f32,
    pub kv_b: Weight,
    pub o_proj: Weight,
    pub mlp: LayerMlp,
    /// The DSA indexer weights, `Some` for full layers (see [`IndexerPin`]).
    /// Populated only when the pin is built for a sparse attention mode.
    pub indexer: Option<IndexerPin>,
}

/// Proof that a layer's demand reads are in flight on the main ring, and the
/// obligation to join them. Returned by [`Pin::submit_layer`], consumed by
/// [`Pin::await_layer`]. Zero-sized: it exists to put the submit/await protocol in
/// the type system and at the call site, where the alternative is a comment nobody
/// re-reads while moving code around between the two.
///
/// Rust is affine, so this can still be dropped without awaiting — but only on a
/// `?` error path, where the forward pass is already unwinding and the run is over.
#[must_use = "the demand reads are still in flight; pass this to Pin::await_layer \
              before launching anything that reads the returned descriptors"]
pub struct DemandBatch(());

/// The resident weight set + cold-expert streaming pool. Borrows the snapshot for
/// its lifetime (cold fetches read the mmap in place).
pub struct Pin<'a> {
    /// The borrow that keeps the snapshot — and thus the O_DIRECT fds the
    /// `moe_table` reads hold as raw `RawFd`s — alive for the run. Not read
    /// through after `build` (the table captured every `(fd, begin, len)` it
    /// needs); it is purely the lifetime/fd anchor.
    #[allow(dead_code)]
    snap: &'a Snapshot,
    cfg: &'a ModelConfig,
    /// The resident weight slab. Never read through after `build` — held purely as
    /// the RAII owner of the VMM allocation that `embed`/`lm_head`/`final_norm`/
    /// `layers`/`moe_bias` point into; its `Drop` frees the slab at run end.
    #[allow(dead_code)]
    tier: DeviceTier,
    pub embed: Weight,
    pub lm_head: Weight,
    pub final_norm: *const f32,
    pub layers: Vec<LayerPin>,
    /// Router correction bias per MoE layer, kept HOST-side (the sigmoid/bias/
    /// top-k routing runs on the CPU). `moe_bias[sparse_idx]`, len n_experts.
    moe_bias: Vec<Vec<f32>>,
    /// The routed-expert LRU pool: ONE device-local VMM slab of `n_slots`
    /// contiguous expert slots (slot i at `i * expert_slot`), each laid out
    /// gate|up|down, every region an O_DIRECT aligned superset. One allocation
    /// (no per-slot granularity waste). Which (layer,expert) occupies each slot is
    /// managed by `lru`; `expert_slot` is the slot stride (the per-region gate/down
    /// stride is re-derived from `cfg` via [`slot_geom`]).
    lru_pool: VmmBuf,
    expert_slot: usize,
    /// Per-(MoE layer, expert) precomputed streaming plan (A1/A2): the coalesced
    /// `(fd, begin, len, dst)` reads (weights then scales, split at shard boundaries)
    /// plus the six slot-relative useful-byte offsets. Indexed by
    /// `(layer - dense_layers) * n_experts + expert`. Built ONCE; the miss and
    /// prefetch paths index it and queue reads with zero allocation or re-validation.
    moe_table: Vec<ExpertReads>,
    /// Adaptive routed tier: (layer,expert) -> slot, policy-evicted (`--cache-policy`).
    /// Online priming that keeps this run's hot experts resident.
    pool: Pool,
    /// io_uring O_DIRECT reader — a layer's cold misses submit as one queue-depth
    /// batch straight into their LRU slots (folds the old mmap-warm + memcpy).
    stream: Streamer,
    /// Cross-layer prefetch (`--prefetch`): a SECOND io_uring ring, dedicated to the
    /// predicted next-layer experts. `Some` iff prefetch is enabled. Its reads are
    /// SUBMITTED (non-blocking) during the current layer's `submit_layer`..`await_layer`
    /// run on the NVMe/DMA side during the current layer's MoE compute, and are
    /// window, and DRAINED at the next `submit_layer` — hiding the ~5ms/miss fetch behind
    /// compute. A separate ring keeps in-flight prefetch reads from entangling with a
    /// layer's synchronous miss batch.
    prefetch_stream: Option<Streamer>,
    /// Max predicted experts prefetched per layer (`--prefetch-depth`, the top-N by
    /// router score). Bounded by the idle-NVMe-during-compute window on this
    /// bandwidth-bound path; the caller slices to this before `prefetch_layer`.
    prefetch_depth: usize,
    /// An outstanding prefetch batch is in flight and must be drained by the next
    /// `submit_layer` before its slots are read.
    prefetch_pending: bool,
    /// Slots the demand ring is currently writing (between `submit_layer` and
    /// `await_layer`). Load-bearing in release: `submit_layer` phase 1b consults it
    /// to refuse reusing a slot whose read is still outstanding (the silent
    /// weight-corruption bug), and it also backs `prefetch_layer`'s slot-
    /// disjointness assertion in debug builds.
    inflight: Vec<usize>,
    /// How many times a batch reused a slot whose read was still outstanding —
    /// each one would have been a silent weight corruption before the guard.
    pub slot_collisions: u64,
    /// Per-`sel` index, whether the last `submit_layer` served that expert from
    /// residency (true) or streamed it (false). Diagnostic for the slot-corruption
    /// hunt: it says whether a wrong-bytes expert was freshly read or was already
    /// sitting in the pool with clobbered contents. Feeds only the
    /// `--checksum-layer` probe, so it is maintained only in a `trace` build.
    #[cfg(feature = "trace")]
    pub last_hit: Vec<bool>,
    /// The predicted expert set submitted for the layer named in `predicted_layer`
    /// (recall accounting at the matching `submit_layer`).
    predicted: Vec<usize>,
    predicted_layer: usize,
    /// Stats over the run — the two headline counters, always maintained (a plain
    /// u64 increment each). The `trace` build additionally splits `hits` into
    /// loaded-vs-preloaded via [`Pool`]'s HashSet accounting.
    pub hits: u64,
    pub misses: u64,
    /// Prefetch recall: `pred_correct` predicted experts were actually selected out
    /// of `pred_total` predicted (over all prefetched layers).
    #[cfg(feature = "trace")]
    pub pred_total: u64,
    #[cfg(feature = "trace")]
    pub pred_correct: u64,
    /// Nanoseconds spent blocked in the prefetch-ring drain (the part of the fetch
    /// that did NOT overlap the previous layer's compute). Near-zero means the reads
    /// were fully hidden; large means the overlap window was too small / NVMe-bound.
    #[cfg(feature = "trace")]
    pub prefetch_wait_ns: u128,
    /// `prefetch_layer` cost split: cache admission / SQE prep / io_uring_submit.
    /// The whole call is OUTSIDE every PROFILE bucket, so it lands in `other` —
    /// which grows ~60 ms per unit of prefetch depth. This says which third it is.
    /// Two clock reads PER PREFETCHED EXPERT, hence `trace`-only.
    #[cfg(feature = "trace")]
    pub pf_alloc_ns: u128,
    #[cfg(feature = "trace")]
    pub pf_queue_ns: u128,
    #[cfg(feature = "trace")]
    pub pf_submit_ns: u128,
    /// Optional access-trace sink (`--trace`): one line per resolved MoE layer —
    /// the space-separated `(layer,expert)` keys the pool looked up, in access
    /// order, then (when prefetch is on) ` | ` and the keys the PREVIOUS layer
    /// prefetched for this one. Feeds the offline cache-policy simulator
    /// (`src/cache.rs`, `bin/replay`). The prediction half is what lets `replay`
    /// model `insert_cold` — without it an offline sweep measures a no-prefetch
    /// engine, which is a materially different (and much worse) cache workload.
    trace: Option<std::io::BufWriter<std::fs::File>>,
}

/// Place a norm/f32 tensor (O-length) into the tier: reserve, then `pread` the
/// bytes straight from the shard file into the device-local slab (evict the
/// page-cache copy — resident weights are read once, never re-loaded).
fn place_f32(tier: &mut DeviceTier, snap: &Snapshot, name: &str) -> Result<*const f32> {
    let len = snap.require(name)?.len();
    let dst = tier.reserve(len)?;
    // SAFETY: dst owns `len` reserved bytes.
    unsafe { snap.read_into(name, dst, len, true)? };
    Ok(dst as *const f32)
}

/// Place an int4 weight (`<name>.weight` + `.qs`) into the tier via `pread`.
fn place_i4(tier: &mut DeviceTier, snap: &Snapshot, name: &str, i_dim: usize) -> Result<Weight> {
    let m = snap.int4(name, i_dim)?;
    let (plen, slen, o_dim) = (m.packed.len(), m.scale.len(), m.o_dim);
    let packed = tier.reserve(plen)?;
    let scale = tier.reserve(slen)?;
    // SAFETY: packed/scale each own their reserved byte counts.
    unsafe {
        snap.read_into(&format!("{name}.weight"), packed, plen, true)?;
        snap.read_into(&format!("{name}.weight.qs"), scale, slen, true)?;
    }
    Ok(Weight {
        packed,
        scale: scale as *const f32,
        o_dim,
        i_dim,
    })
}

/// Place an int8 weight into the tier via `pread`.
fn place_i8(tier: &mut DeviceTier, snap: &Snapshot, name: &str, i_dim: usize) -> Result<Weight> {
    let m = snap.int8(name, i_dim)?;
    let (plen, slen, o_dim) = (m.packed.len(), m.scale.len(), m.o_dim);
    let packed = tier.reserve(plen)?;
    let scale = tier.reserve(slen)?;
    // SAFETY: packed/scale each own their reserved byte counts.
    unsafe {
        snap.read_into(&format!("{name}.weight"), packed, plen, true)?;
        snap.read_into(&format!("{name}.weight.qs"), scale, slen, true)?;
    }
    Ok(Weight {
        packed,
        scale: scale as *const f32,
        o_dim,
        i_dim,
    })
}

/// Place a raw bf16 tensor (by full name) into the tier via `pread`, returning
/// the device `u16` pointer. For the indexer projections, which ship at bf16 in
/// the out-idx shard and are read as bf16 by `gemv_bf16` (no widening).
fn place_bf16(tier: &mut DeviceTier, snap: &Snapshot, name: &str) -> Result<*const u16> {
    let len = snap.typed(name, Dtype::Bf16)?.len();
    let dst = tier.reserve(len)?;
    // SAFETY: dst owns `len` reserved bytes.
    unsafe { snap.read_into(name, dst, len, true)? };
    Ok(dst as *const u16)
}

/// Place a bf16 tensor WIDENED to f32 into the tier (the indexer `k_norm`
/// weight/bias, read as f32 by the `layernorm` kernel). The tier slab is
/// host-writable, so widen on host then copy the f32 bytes in.
fn place_bf16_as_f32(tier: &mut DeviceTier, snap: &Snapshot, name: &str) -> Result<*const f32> {
    let vals = crate::quant::read_bf16(snap.typed(name, Dtype::Bf16)?);
    // SAFETY: f32 is POD; this is its LE byte serialization on this LE host.
    let bytes = unsafe {
        std::slice::from_raw_parts(vals.as_ptr() as *const u8, std::mem::size_of_val(&vals[..]))
    };
    let dst = tier.reserve(bytes.len())?;
    // SAFETY: dst owns `bytes.len()` reserved host-writable bytes.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
    Ok(dst as *const f32)
}

/// Place a full layer's DSA indexer weights (bf16 projections + f32 k_norm).
fn place_indexer(tier: &mut DeviceTier, snap: &Snapshot, layer: usize) -> Result<IndexerPin> {
    let base = format!("model.layers.{layer}.self_attn.indexer");
    Ok(IndexerPin {
        wk: place_bf16(tier, snap, &format!("{base}.wk.weight"))?,
        wq_b: place_bf16(tier, snap, &format!("{base}.wq_b.weight"))?,
        weights_proj: place_bf16(tier, snap, &format!("{base}.weights_proj.weight"))?,
        k_norm_w: place_bf16_as_f32(tier, snap, &format!("{base}.k_norm.weight"))?,
        k_norm_b: place_bf16_as_f32(tier, snap, &format!("{base}.k_norm.bias"))?,
    })
}

fn place_mlp(
    tier: &mut DeviceTier,
    snap: &Snapshot,
    base: &str,
    hidden: usize,
    inter: usize,
) -> Result<Mlp> {
    Ok(Mlp {
        gate: place_i4(tier, snap, &format!("{base}.gate_proj"), hidden)?,
        up: place_i4(tier, snap, &format!("{base}.up_proj"), hidden)?,
        down: place_i4(tier, snap, &format!("{base}.down_proj"), inter)?,
    })
}

/// Device bytes the always-resident set occupies — everything the forward pass
/// reads every token EXCEPT the routed experts: the int8 embed/lm_head tables +
/// final norm, and per layer the four norms, the five MLA projections, and either
/// the dense MLP (dense layers) or the router gate f32 + shared expert (MoE
/// layers). Summed from `cfg` so the resident tier is sized to what it actually
/// holds and the rest of the device budget grows the routed LRU (rather than a
/// fixed cap stranding several GiB). Byte counts mirror the placement path:
/// int4 `[o,i]` = `o*row_bytes(i)` packed + `o*4` scale; int8 `[o,i]` = `o*i` +
/// `o*4`; an f32 norm of `n` = `n*4`.
fn resident_bytes(cfg: &ModelConfig) -> usize {
    let rb = crate::quant::row_bytes;
    let i4 = |o: usize, i: usize| o * rb(i) + o * 4;
    let i8 = |o: usize, i: usize| o * i + o * 4;
    let f32n = |n: usize| n * 4;

    let qk = cfg.qk_head_dim();
    // Global int8 tables + final norm.
    let mut total = i8(cfg.vocab, cfg.hidden) // embed_tokens
        + i8(cfg.vocab, cfg.hidden)           // lm_head
        + f32n(cfg.hidden); // model.norm

    for l in 0..cfg.n_layers {
        // Norms: input, post-attn, q_a, kv_a.
        total += 2 * f32n(cfg.hidden) + f32n(cfg.q_lora_rank) + f32n(cfg.kv_lora_rank);
        // MLA projections (o_dim as placed by `Pin::build`).
        total += i4(cfg.q_lora_rank, cfg.hidden); // q_a
        total += i4(cfg.n_heads * qk, cfg.q_lora_rank); // q_b
        total += i4(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden); // kv_a
        total += i4(
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            cfg.kv_lora_rank,
        ); // kv_b
        total += i4(cfg.hidden, cfg.n_heads * cfg.v_head_dim); // o_proj
        // MLP: dense for the first `dense_layers`, else router gate + shared expert.
        if l < cfg.dense_layers {
            total += 2 * i4(cfg.dense_inter, cfg.hidden); // gate, up
            total += i4(cfg.hidden, cfg.dense_inter); // down
        } else {
            total += f32n(cfg.n_experts * cfg.hidden); // router gate (F32, device)
            let si = cfg.moe_inter * cfg.n_shared;
            total += 2 * i4(si, cfg.hidden); // shared gate, up
            total += i4(cfg.hidden, si); // shared down
        }
    }
    total
}

/// Resident bytes for ONE full layer's DSA indexer weights (bf16 projections +
/// the f32-widened k_norm), mirroring `place_indexer`. Multiply by the full-
/// layer count for the indexer footprint.
fn indexer_bytes(cfg: &ModelConfig) -> usize {
    let hd = cfg.index_head_dim;
    let nh = cfg.index_n_heads;
    let bf16 = |o: usize, i: usize| o * i * 2;
    bf16(hd, cfg.hidden)          // wk
        + bf16(nh * hd, cfg.q_lora_rank) // wq_b
        + bf16(nh, cfg.hidden)    // weights_proj
        + 2 * hd * 4 // k_norm weight + bias, widened to f32
}

/// Pack `(layer, expert)` into the LRU key. Both must fit in 16 bits — GLM is
/// ≤92 layers × 256 routed experts, comfortably under 2^16, but assert it so a
/// larger config can't silently collide keys.
fn expert_key(layer: usize, expert: usize) -> u32 {
    debug_assert!(
        layer < (1 << 16) && expert < (1 << 16),
        "layer {layer}/expert {expert} exceed the 16-bit LRU key packing"
    );
    ((layer as u32) << 16) | expert as u32
}

/// The routed-expert pool: maps `(layer,expert)` keys to slab slot indices, with
/// eviction delegated to a pluggable `cache::Cache` policy (`--cache-policy`
/// lru|2q|arc). The policy owns residency + eviction order; the pool owns the
/// key↔slot maps. On a miss the policy names the evicted key (if any) and the pool
/// reuses that key's slot — `slot_of` stays in lockstep with the policy's residency
/// (debug-asserted). The per-expert streaming offsets live in [`Pin::moe_table`]
/// (keyed by (layer,expert), run-invariant), NOT per slot, so the pool holds no
/// geometry.
struct Pool {
    policy: Box<dyn cache::Cache>,
    slot_of: std::collections::HashMap<u32, usize>,
    /// Reverse slot→key map, used ONLY to feed the stale-slot `debug_assert_eq!` in
    /// `get`. It carries no release-build behaviour, so the field, its writes, and
    /// the assertion all compile out in release.
    #[cfg(debug_assertions)]
    key_of: Vec<Option<u32>>,
    free: Vec<usize>,
    /// Keys bound by [`alloc_cold`] (a prefetch) whose data has been read off disk
    /// but which no `get` has claimed yet. A key lives here between its prefetch
    /// submit and its real selection. Two things read it:
    ///   - `get` — a hit on a key in here came off the disk THIS layer (preloading),
    ///     not out of residency (loaded). Without this the two are indistinguishable
    ///     and the reported hit rate silently counts disk reads as hits.
    ///   - `reuse` — a key evicted while still in here was read and thrown away
    ///     before anyone used it: a fully wasted expert read.
    #[cfg(feature = "trace")]
    pending_pf: std::collections::HashSet<u32>,
    /// Hits on genuinely-resident keys (no disk read behind them).
    #[cfg(feature = "trace")]
    pub hit_loaded: u64,
    /// Hits on keys a prefetch had just streamed in (disk read, but off the
    /// critical path — it overlapped the previous layer's compute).
    #[cfg(feature = "trace")]
    pub hit_preload: u64,
    /// Prefetched keys evicted before any `get` claimed them — wasted reads.
    #[cfg(feature = "trace")]
    pub pf_evict_unused: u64,
    /// Keys currently in the "was prefetched, evicted unused" state. A demand miss
    /// on one of these is the DIRECT signature of eviction-before-use: the expert
    /// really was wanted, we really did read it, and we threw it away in time to
    /// have to read it again. Distinguishes that from plain misprediction, which
    /// also lands in `pf_evict_unused` but is never demanded afterwards.
    #[cfg(feature = "trace")]
    evicted_pf: std::collections::HashSet<u32>,
    /// Demand misses on a key we had already prefetched and evicted unused.
    #[cfg(feature = "trace")]
    pub pf_evict_then_missed: u64,
}

impl Pool {
    fn new(n: usize, policy: &str, two_q: cache::TwoQSplit) -> Result<Self> {
        Ok(Self {
            policy: cache::make(policy, n, two_q)
                .with_context(|| format!("unknown --cache-policy {policy:?} (lru|2q|arc)"))?,
            slot_of: std::collections::HashMap::with_capacity(n),
            #[cfg(debug_assertions)]
            key_of: vec![None; n],
            free: (0..n).rev().collect(), // pop() hands out 0,1,2,...
            #[cfg(feature = "trace")]
            pending_pf: std::collections::HashSet::new(),
            #[cfg(feature = "trace")]
            hit_loaded: 0,
            #[cfg(feature = "trace")]
            hit_preload: 0,
            #[cfg(feature = "trace")]
            pf_evict_unused: 0,
            #[cfg(feature = "trace")]
            evicted_pf: std::collections::HashSet::new(),
            #[cfg(feature = "trace")]
            pf_evict_then_missed: 0,
        })
    }

    /// Resident slot for `key` (a hit, promoted by the policy), or `None` (a miss).
    fn get(&mut self, key: u32) -> Option<usize> {
        if self.policy.get(key) {
            let slot = self.slot_of[&key];
            // The slot must actually hold THIS key's streamed data (else a hit reads
            // a different expert's weights — stale-data corruption).
            #[cfg(debug_assertions)]
            debug_assert_eq!(self.key_of[slot], Some(key), "hit slot holds wrong key");
            // Claim the prediction: a hit on a pending prefetch is a PRELOADING hit
            // (its bytes came off disk during the previous layer), everything else is
            // a LOADED hit (already resident, no I/O at all). Accounting only — the
            // HashSet probe is why it lives behind `trace`.
            #[cfg(feature = "trace")]
            if self.pending_pf.remove(&key) {
                self.hit_preload += 1;
            } else {
                self.hit_loaded += 1;
            }
            Some(slot)
        } else {
            None
        }
    }

    /// Allocate a slot for a NEW `key` (miss): reuse the policy-evicted key's slot,
    /// else a free slot. The caller then fills the slot and sets `off[slot]`.
    fn alloc(&mut self, key: u32) -> Result<usize> {
        // Re-reading an expert we prefetched and then evicted unused: the read was
        // paid twice and the prediction was RIGHT, just too early / evicted too soon.
        #[cfg(feature = "trace")]
        if self.evicted_pf.remove(&key) {
            self.pf_evict_then_missed += 1;
        }
        let evicted = self.policy.insert(key);
        let slot = self.reuse(evicted)?;
        self.bind(key, slot);
        Ok(slot)
    }

    /// Like [`alloc`], but for a PREFETCHED (predicted) key: the policy parks it at
    /// the cold/probation end (`insert_cold`) so an unused prediction is evicted
    /// before any genuinely-accessed expert, and never pollutes the hot set. A later
    /// real selection of the key (`get`) promotes it normally.
    fn alloc_cold(&mut self, key: u32) -> Result<usize> {
        let evicted = self.policy.insert_cold(key);
        let slot = self.reuse(evicted)?;
        self.bind(key, slot);
        // Unclaimed until a real `get`; `reuse` charges it as wasted if it is
        // evicted first.
        #[cfg(feature = "trace")]
        self.pending_pf.insert(key);
        Ok(slot)
    }

    /// Shield a just-hit key from eviction for the rest of this batch (see
    /// [`cache::Cache::protect`]).
    fn protect(&mut self, key: u32) {
        self.policy.protect(key);
    }

    /// Is `key` currently resident in the policy? (Used to skip prefetching an
    /// expert the pool already holds.)
    fn contains(&self, key: u32) -> bool {
        self.policy.contains(key)
    }

    /// Map a policy eviction (or spare capacity) to a concrete slab slot.
    fn reuse(&mut self, evicted: Option<u32>) -> Result<usize> {
        match evicted {
            Some(ev) => {
                // The evicted key must no longer be resident in the policy, else two
                // keys would share a slot (stale-data corruption).
                debug_assert!(
                    !self.policy.contains(ev),
                    "evicted key {ev} still resident (identity drift)"
                );
                // Evicting a still-pending prediction throws away an expert read
                // that nobody ever used — the read cost was paid for nothing, and
                // a later selection of `ev` has to stream it again.
                #[cfg(feature = "trace")]
                if self.pending_pf.remove(&ev) {
                    self.pf_evict_unused += 1;
                    self.evicted_pf.insert(ev);
                }
                self.slot_of
                    .remove(&ev)
                    .context("evicted key had no slot (pool/policy drift)")
            }
            None => self
                .free
                .pop()
                .context("no free slot and policy evicted nothing"),
        }
    }

    fn bind(&mut self, key: u32, slot: usize) {
        #[cfg(debug_assertions)]
        {
            self.key_of[slot] = Some(key);
        }
        self.slot_of.insert(key, slot);
        // Rebinding clears the "prefetched then evicted unused" mark: `alloc` has
        // already charged it above, and a re-prefetch must not charge it again.
        #[cfg(feature = "trace")]
        self.evicted_pf.remove(&key);
        // The just-inserted key must be resident, and slot_of must mirror residency.
        debug_assert!(self.policy.contains(key), "inserted key {key} not resident");
        debug_assert_eq!(
            self.slot_of.len(),
            self.policy.resident_len(),
            "pool/policy residency drift"
        );
    }
}

impl<'a> Pin<'a> {
    /// Build the resident set. `capacity` is the tier size (auto-discovered).
    /// `direct_io` selects the cold-read fd set for the `moe_table`: `true` =
    /// O_DIRECT (page-cache bypass), `false` = buffered (through the OS page cache).
    // Each arg is a distinct, independent input (snapshot/model/usage + the
    // runtime knobs); bundling them into a struct used at one call site is churn.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        snap: &'a Snapshot,
        cfg: &'a ModelConfig,
        usage: &Usage,
        capacity: usize,
        pre_seed: bool,
        bounce: bool,
        trace_path: Option<&str>,
        cache_policy: &str,
        two_q: cache::TwoQSplit,
        prefetch: bool,
        prefetch_depth: usize,
        direct_io: bool,
        want_indexer: bool,
    ) -> Result<Self> {
        let (free, _total) = mem_info()?;
        // Full layers own a resident DSA indexer (dsa/misa modes). Empty when
        // dense/streaming — the mask drives both placement and the footprint.
        let full = if want_indexer {
            cfg.indexer_layout()?
        } else {
            vec![false; cfg.n_layers]
        };
        ensure!(
            capacity < free,
            "pin capacity {capacity} >= free device memory {free}"
        );
        // One-time bound for `submit_layer`'s fixed 32-slot scratch (checked per
        // layer only as a debug assertion).
        ensure!(
            cfg.top_k + cfg.n_shared <= 32,
            "top_k {} + n_shared {} exceeds the 32-slot batch scratch",
            cfg.top_k,
            cfg.n_shared
        );
        // The static tier holds only the always-resident set; the rest of the
        // budget goes to the routed LRU. Size the tier to the footprint computed
        // from `cfg` (not a fixed cap that strands 1-5 GiB the LRU could use, and
        // not `capacity`, which would hog everything the LRU needs). The slack
        // absorbs the per-reservation 256-byte alignment padding and a small
        // margin; anything left over widens the LRU below.
        const SLACK: usize = 256 << 20; // 256 MiB
        let n_full = full.iter().filter(|&&f| f).count();
        let resident = resident_bytes(cfg) + n_full * indexer_bytes(cfg);
        let tier_cap = resident + SLACK;
        tracing::info!(
            "computed resident footprint {:.2} GiB (tier {:.2} GiB incl. slack)",
            resident as f64 / (1u64 << 30) as f64,
            tier_cap as f64 / (1u64 << 30) as f64,
        );
        let mut tier = DeviceTier::new(tier_cap)?;

        // Global tensors.
        let embed = place_i8(&mut tier, snap, "model.embed_tokens", cfg.hidden)?;
        let lm_head = place_i8(&mut tier, snap, "lm_head", cfg.hidden)?;
        let final_norm = place_f32(&mut tier, snap, "model.norm.weight")?;

        // Per-layer always-resident weights.
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut moe_bias = Vec::new();
        // `l` indexes both the weight-name format!s and the `full` mask; iterating
        // `full` directly would lose the layer number the names need.
        #[allow(clippy::needless_range_loop)]
        for l in 0..cfg.n_layers {
            let lb = format!("model.layers.{l}");
            let a = format!("{lb}.self_attn");
            let mlp = if l < cfg.dense_layers {
                LayerMlp::Dense(place_mlp(
                    &mut tier,
                    snap,
                    &format!("{lb}.mlp"),
                    cfg.hidden,
                    cfg.dense_inter,
                )?)
            } else {
                let gate_w = place_f32(&mut tier, snap, &format!("{lb}.mlp.gate.weight"))?;
                let bias = snap.typed(
                    &format!("{lb}.mlp.gate.e_score_correction_bias"),
                    Dtype::F32,
                )?;
                let bias = crate::quant::read_f32(bias);
                ensure!(
                    bias.len() == cfg.n_experts,
                    "layer {l} gate bias has {} entries, expected {}",
                    bias.len(),
                    cfg.n_experts
                );
                moe_bias.push(bias);
                let shared = place_mlp(
                    &mut tier,
                    snap,
                    &format!("{lb}.mlp.shared_experts"),
                    cfg.hidden,
                    cfg.moe_inter * cfg.n_shared,
                )?;
                LayerMlp::Moe { gate_w, shared }
            };
            layers.push(LayerPin {
                input_ln: place_f32(&mut tier, snap, &format!("{lb}.input_layernorm.weight"))?,
                post_ln: place_f32(
                    &mut tier,
                    snap,
                    &format!("{lb}.post_attention_layernorm.weight"),
                )?,
                q_a: place_i4(&mut tier, snap, &format!("{a}.q_a_proj"), cfg.hidden)?,
                q_a_ln: place_f32(&mut tier, snap, &format!("{a}.q_a_layernorm.weight"))?,
                q_b: place_i4(&mut tier, snap, &format!("{a}.q_b_proj"), cfg.q_lora_rank)?,
                kv_a: place_i4(
                    &mut tier,
                    snap,
                    &format!("{a}.kv_a_proj_with_mqa"),
                    cfg.hidden,
                )?,
                kv_a_ln: place_f32(&mut tier, snap, &format!("{a}.kv_a_layernorm.weight"))?,
                kv_b: place_i4(&mut tier, snap, &format!("{a}.kv_b_proj"), cfg.kv_lora_rank)?,
                o_proj: place_i4(
                    &mut tier,
                    snap,
                    &format!("{a}.o_proj"),
                    cfg.n_heads * cfg.v_head_dim,
                )?,
                mlp,
                indexer: if full[l] {
                    Some(place_indexer(&mut tier, snap, l)?)
                } else {
                    None
                },
            });
        }

        // The routed tier is an adaptive LRU, not a static frequency pin: N reused
        // slots stream experts in on demand and keep this run's actually-hot ones
        // (online priming). Each slot holds one expert's two O_DIRECT aligned supersets
        // — the coalesced weight span then the coalesced scale span, each `slot_span`
        // (block-aligned + straddle pad) — so reads land in place and descriptors point
        // at the sub-block offsets. Size to the budget left after the always-resident set.
        let expert_slot = slot_geom(cfg);
        // Precompute the per-(layer,expert) streaming plan ONCE (A1/A2): coalesced
        // (fd, begin, len, dst) reads + six slot-relative useful-byte offsets, cfg-dim
        // cross-checked here. The miss/prefetch paths then just index + queue.
        let moe_table = build_moe_table(snap, cfg, direct_io)?;
        let budget = capacity.saturating_sub(tier_cap);
        // Floor the pool at `top_k + prefetch_depth` when prefetching: this makes
        // cross-layer prefetch's evictions provably DISJOINT from the current layer's
        // live experts. The current layer's `top_k` routed experts are the
        // most-recently-touched (MRU / protected) after `submit_layer`; `alloc_cold`
        // evicts the LRU. With this many slots, the `prefetch_depth` cold evictions
        // can never reach those MRU experts — so the prefetch DMA (whenever it lands:
        // async in direct mode, or at the drain memcpy in bounce mode) writes slots
        // the running MoE never reads. Disjoint memory ⇒ safe concurrency with NO
        // ordering barrier, in BOTH modes. (In practice n_slots ≫ this; the floor
        // just makes the invariant explicit and rejects a degenerate tiny tier.)
        let slot_floor = cfg.top_k + if prefetch { prefetch_depth } else { 0 };
        let n_slots = (budget / expert_slot).max(slot_floor);
        let mut lru_pool = VmmBuf::new(n_slots * expert_slot)?; // ONE slab
        tracing::info!(
            "routed pool [{cache_policy}]: {n_slots} slots ({:.1} GiB) + {:.1} GiB always-resident",
            (n_slots * expert_slot) as f64 / (1u64 << 30) as f64,
            tier.used() as f64 / (1u64 << 30) as f64,
        );
        let mut pool = Pool::new(n_slots, cache_policy, two_q)?;
        // Ring sized for one layer's worst case: top_k misses x 6 reads/expert. An
        // expert normally coalesces to 2 reads (weights, scales); the 6 bound covers
        // the pathological case where every projection straddles a shard boundary.
        // `sel` carries the ROUTED picks only — the shared expert is always resident
        // (placed in the tier above), so it never streams and never takes a ring
        // entry. Assert the bound the sizing actually rests on: a reader who assumes
        // the shared expert streams too concludes this ring is undersized, and an
        // overflow would otherwise surface as a mid-decode "SQ full" from
        // `Streamer::queue` rather than as a config error at build.
        let ring = (cfg.top_k * 6).next_power_of_two() * 2;
        ensure!(
            cfg.top_k * 6 <= ring,
            "io_uring ring {ring} too small for one layer: top_k {} x 6 reads/expert",
            cfg.top_k,
        );
        // Bounce span = largest single read superset. The coalesced weight read
        // (down|gate|up, ~18.9 MB) dwarfs the coalesced scale read.
        let (coal_w, coal_s) = coalesced_lens(cfg);
        let span = slot_span(coal_w).max(slot_span(coal_s));
        // The DEMAND ring gets a poller too. The earlier reasoning — "its submit is
        // immediately followed by a blocking drain, so it gains nothing" — conflated
        // two different things: the thread must indeed wait either way, but WITHOUT a
        // poller the waiting thread dispatches the batch's SQEs serially, so a layer's
        // reads (now 2 per expert, several experts) never sit in the device queue
        // together. See `Sqpoll`.
        let mut stream = Streamer::new(ring as u32, span, bounce, Sqpoll::Own)?;
        // Optional warm start (`--pre-seed`): seed the pool with the hottest experts
        // from .coli_usage so the first tokens hit at colibri's rate while the LRU
        // adapts. Worth ~+6.8pt on the first tokens but transient (the LRU warms up
        // in a few tokens regardless) and it dominates build time (~23s), so it is
        // OFF by default — enable it only for slow disks. Bounded by the pool size;
        // drained in queue-depth batches.
        if pre_seed {
            let slab = lru_pool.ptr_mut();
            let batch = (ring / 6).max(1); // experts per drain (<=6 reads each)
            let mut n = 0usize;
            for &((l, e), _) in usage.ranked().iter() {
                if n >= n_slots {
                    break;
                }
                let (l, e) = (l as usize, e as usize);
                if l < cfg.dense_layers || l >= cfg.n_layers || e >= cfg.n_experts {
                    continue; // stale ranking entry
                }
                let slot = pool.alloc(expert_key(l, e))?;
                let entry = &moe_table[(l - cfg.dense_layers) * cfg.n_experts + e];
                stream_expert(&mut stream, entry, slab, slot, expert_slot)?;
                n += 1;
                if n.is_multiple_of(batch) {
                    stream.drain()?;
                }
            }
            stream.drain()?;
            tracing::info!("LRU seeded {n} experts from .coli_usage (warm start)");
        }

        // Cross-layer prefetch (`--prefetch`): a second ring, same geometry as the
        // synchronous one, dedicated to the predicted next-layer experts.
        let prefetch_stream = if prefetch {
            tracing::info!(
                "cross-layer expert prefetch ENABLED (single-layer lookahead, depth {prefetch_depth})"
            );
            // Share the demand ring's poller rather than spinning a second kernel
            // thread: both rings are fed from the one decode thread, and a spinning
            // poller is not free on a box whose GPU already wants 214 of 230 GB/s.
            Some(Streamer::new(
                ring as u32,
                span,
                bounce,
                Sqpoll::SharedWith(&stream),
            )?)
        } else {
            None
        };

        Ok(Self {
            snap,
            cfg,
            tier,
            embed,
            lm_head,
            final_norm,
            layers,
            moe_bias,
            lru_pool,
            expert_slot,
            moe_table,
            pool,
            stream,
            prefetch_stream,
            prefetch_depth,
            prefetch_pending: false,
            inflight: Vec::new(),
            slot_collisions: 0,
            #[cfg(feature = "trace")]
            last_hit: Vec::new(),
            predicted: Vec::new(),
            predicted_layer: usize::MAX,
            hits: 0,
            misses: 0,
            #[cfg(feature = "trace")]
            pred_total: 0,
            #[cfg(feature = "trace")]
            pred_correct: 0,
            #[cfg(feature = "trace")]
            prefetch_wait_ns: 0,
            #[cfg(feature = "trace")]
            pf_alloc_ns: 0,
            #[cfg(feature = "trace")]
            pf_queue_ns: 0,
            #[cfg(feature = "trace")]
            pf_submit_ns: 0,
            trace: trace_path
                .map(|p| -> Result<_> {
                    Ok(std::io::BufWriter::new(
                        std::fs::File::create(p).with_context(|| format!("open trace {p}"))?,
                    ))
                })
                .transpose()?,
        })
    }

    /// Host router correction bias for a MoE `layer` (len n_experts).
    pub fn moe_bias(&self, layer: usize) -> &[f32] {
        &self.moe_bias[layer - self.cfg.dense_layers]
    }

    /// `(loaded, preloading, cold, pf_evict_unused, pf_evict_then_missed)` —
    /// where routed experts' bytes came from. `trace` builds only: the split is
    /// what the pool's HashSet accounting exists to produce.
    #[cfg(feature = "trace")]
    pub fn source_split(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.pool.hit_loaded,
            self.pool.hit_preload,
            self.misses,
            self.pool.pf_evict_unused,
            self.pool.pf_evict_then_missed,
        )
    }

    /// Resolve a MoE layer's `sel` routed experts to `Mlp` descriptors via the
    /// adaptive LRU and SUBMIT (non-blocking) the cold reads: a hit returns its
    /// resident slot's pointers (no I/O); a miss evicts the coldest slot and queues
    /// the expert's six O_DIRECT reads. All of the layer's misses go out as ONE
    /// batch. Fills the caller-owned `out` (cleared first, reused across tokens) so
    /// the decode hot path allocates nothing here on the all-hits steady state.
    ///
    /// **The returned descriptors are valid POINTERS but their bytes are still in
    /// flight.** Nothing may READ a cold slot until [`Pin::await_layer`] has consumed
    /// the returned [`DemandBatch`]. Building the descriptors here (rather than after
    /// the join) is sound because a slot's address is fixed at `pool.alloc` time — it
    /// is a function of the slot index and `cfg` geometry, not of the data arriving.
    /// That is what lets the caller do its descriptor build, its small H2D uploads,
    /// and its next-layer prefetch submit while the NVMe works.
    pub fn submit_layer(
        &mut self,
        layer: usize,
        sel: &[usize],
        out: &mut Vec<Mlp>,
    ) -> Result<DemandBatch> {
        let cfg = self.cfg;
        let sparse = (layer - cfg.dense_layers) * cfg.n_experts; // moe_table row base
        // `sel` is one layer's ROUTED pick (top_k; the shared expert is resident and
        // never streams); a fixed slot scratch avoids a per-layer alloc, mirroring
        // `cache::access_batch`'s 32-wide buffer.
        // The bound is cfg-derived and checked once at `build`; per-layer it is a
        // debug assertion only.
        debug_assert!(
            sel.len() <= 32,
            "submit_layer: {} experts exceeds the 32-slot scratch",
            sel.len()
        );
        // Cross-layer prefetch consume: an outstanding prefetch batch (submitted
        // during the PREVIOUS layer's compute) targets THIS layer. Wait for its reads
        // + bounce copy (+ device sync inside `drain`), so every predicted-correct
        // expert is resident in VMM BEFORE phase 1a's hit scan and this layer's MoE
        // read it. The predicted-correct keys were already `alloc_cold`-bound to their
        // slots with `off` set (at submit time), so they now surface as normal hits.
        //
        // THIS DRAIN CANNOT BE DEFERRED alongside the demand drain, tempting as that
        // is. It is the barrier that makes phase 1b's `pool.alloc` unconditionally
        // safe. A prefetched key sits at the policy's cold/probation end by
        // construction (`alloc_cold`), so it is precisely the kind of entry a demand
        // miss's `policy.insert` picks as its eviction victim — that is the measured
        // `pf_evict_unused` path. If the prefetch reads were still in flight, such an
        // eviction would hand phase 1b a slot an O_DIRECT read is actively filling,
        // and phase 1b would queue a SECOND read into the same bytes. Whichever
        // landed last would win while the descriptor claimed the other expert:
        // silent weight corruption, no error, wrong logits. Draining first costs
        // whatever `prefetch_wait_ns` reports (near zero when the overlap works) and
        // buys an invariant that needs no pinning machinery to hold.
        if self.prefetch_pending {
            if let Some(s) = self.prefetch_stream.as_mut() {
                #[cfg(feature = "trace")]
                let t = std::time::Instant::now();
                s.drain()?;
                #[cfg(feature = "trace")]
                {
                    self.prefetch_wait_ns += t.elapsed().as_nanos();
                }
            }
            self.prefetch_pending = false;
            // Recall: how many predicted experts are in this layer's actual `sel`.
            // An O(depth x top_k) scan feeding only the end-of-run recall line.
            #[cfg(feature = "trace")]
            if self.predicted_layer == layer {
                let hit = self.predicted.iter().filter(|e| sel.contains(e)).count();
                self.pred_total += self.predicted.len() as u64;
                self.pred_correct += hit as u64;
            }
        }
        // Trace sink (--trace): the exact keys this layer looks up, access order,
        // then the set the previous layer prefetched FOR this layer. Write each key
        // straight to the BufWriter (no per-layer Vec<String>+join); a leading space
        // before all but the first keeps the demand half byte-identical (space-
        // separated keys, no trailing space, one newline per layer). The ` |
        // <predicted>` tail is emitted only when a prediction targeted this layer,
        // so a `--no-prefetch` trace is byte-for-byte what it always was.
        if let Some(w) = &mut self.trace {
            use std::io::Write;
            for (j, &e) in sel.iter().enumerate() {
                if j > 0 {
                    write!(w, " ").context("write trace")?;
                }
                write!(w, "{}", expert_key(layer, e)).context("write trace")?;
            }
            // Separator written WITHOUT a leading space: `cut -d'|' -f1` must yield
            // byte-identical demand lists whether or not a prediction tail follows,
            // or a whitespace-only artifact reads as a workload divergence. That
            // false positive has already cost one bogus "second bug" report.
            if self.predicted_layer == layer && !self.predicted.is_empty() {
                write!(w, "|").context("write trace")?;
                for &e in &self.predicted {
                    write!(w, " {}", expert_key(layer, e)).context("write trace")?;
                }
            }
            writeln!(w).context("write trace")?;
        }
        // Phase 1a: touch EVERY hit first. Doing this before any miss's `alloc()`
        // bumps each hit's LRU tick above the eviction candidates, so a later miss
        // cannot evict a same-layer would-be hit out from under itself (which would
        // force a needless re-stream and understate the hit rate). `None` marks a
        // slot still to be filled in phase 1b.
        let mut slots: [Option<usize>; 32] = [None; 32];
        #[cfg(feature = "trace")]
        {
            self.last_hit.clear();
            self.last_hit.resize(sel.len(), false);
        }
        for (i, &e) in sel.iter().enumerate() {
            if let Some(slot) = self.pool.get(expert_key(layer, e)) {
                self.hits += 1;
                #[cfg(feature = "trace")]
                {
                    self.last_hit[i] = true;
                }
                // Hits must survive phase 1b's evictions: `slots[i]` (and the
                // descriptor built from it) stays live for the whole layer.
                self.pool.protect(expert_key(layer, e));
                slots[i] = Some(slot);
            }
        }
        // Phase 1b: allocate + QUEUE the misses, now that all hits are protected.
        // Reset the in-flight set here rather than only in `await_layer`, so an
        // error path that skips the join cannot leave a stale slot recorded.
        self.inflight.clear();
        for (i, &e) in sel.iter().enumerate() {
            if slots[i].is_some() {
                continue;
            }
            self.misses += 1;
            let slot = self.pool.alloc(expert_key(layer, e))?;
            // SLOT REUSE WITHIN ONE BATCH — the correctness hazard this guard exists
            // for. `alloc` may evict a key that an EARLIER miss in this same batch
            // just inserted, recycling its slot. Both experts' O_DIRECT reads would
            // then target the same destination, and io_uring completion order is NOT
            // submission order — so the evicted key's read can land LAST and leave
            // the slot holding the wrong expert's weights, while `slot_of`/`key_of`
            // record the new key. The bookkeeping stays self-consistent and the data
            // is silently wrong.
            //
            // Measured before this guard: `2q --no-prefetch --max-mem 50` vs the same
            // run at full pool diverged at pos=8, layer 31, expert 110 — all six
            // weight tensors different, correct at five other occurrences of the same
            // expert. One transient corruption is enough to perturb the residual
            // stream, reroute experts, and change the whole downstream workload.
            //
            // Fix: complete the outstanding reads before reusing their slot. The
            // earlier read then lands FIRST and this one overwrites it, which is the
            // correct final state (the evicted key is no longer resident, so nobody
            // reads its bytes). Costs one extra drain, and only on a real collision.
            if self.inflight.contains(&slot) {
                self.stream.drain()?;
                self.inflight.clear();
                self.slot_collisions += 1;
            }
            // ptr_mut() yields a raw ptr, so no borrow of lru_pool is held across
            // the &mut self.stream reads. Table entry + moe_table are disjoint fields
            // from stream — no alloc, no string hashing, no re-validation per miss.
            let slab = self.lru_pool.ptr_mut();
            let entry = &self.moe_table[sparse + e];
            stream_expert(&mut self.stream, entry, slab, slot, self.expert_slot)?;
            slots[i] = Some(slot);
            // Record that an O_DIRECT read is writing this slot. Load-bearing in
            // release now, not just a debug check: the collision guard above reads
            // it, and it still backs `prefetch_layer`'s disjointness assertion.
            self.inflight.push(slot);
        }
        // Phase 2: hand the whole batch to the kernel NOW, without waiting. With the
        // ring's poller thread this puts all of the layer's reads in the device queue
        // at once (the array wants P>=4), and returns to the caller so its descriptor
        // build, uploads and next-layer prefetch submit run while the NVMe works.
        // The matching join is `await_layer`.
        self.stream.submit()?;
        // Phase 3: build descriptors from each expert's precomputed sub-block offsets
        // (in `moe_table`, keyed by (layer,expert) — run-invariant, not per slot).
        // gate/up are [moe_inter, hidden]; down is [hidden, moe_inter].
        let (gate_o, gate_i) = (cfg.moe_inter, cfg.hidden);
        let (down_o, down_i) = (cfg.hidden, cfg.moe_inter);
        let slab = self.lru_pool.ptr();
        let es = self.expert_slot;
        out.clear();
        for (i, &e) in sel.iter().enumerate() {
            // Every entry was filled in phase 1a/1b; a `None` here is an internal
            // invariant break, not a data condition — fail loud rather than panic.
            let slot = slots[i].context("submit_layer: unresolved expert slot")?;
            let o = self.moe_table[sparse + e].off;
            // SAFETY: slot base within the slab. Only ADDRESS arithmetic — the bytes
            // of a cold slot are still arriving and are not read until `await_layer`.
            let b = unsafe { slab.add(slot * es) };
            let w = |poff: usize, soff: usize, o_dim: usize, i_dim: usize| Weight {
                packed: unsafe { b.add(poff) },
                scale: unsafe { b.add(soff) } as *const f32,
                o_dim,
                i_dim,
            };
            out.push(Mlp {
                gate: w(o[0], o[1], gate_o, gate_i),
                up: w(o[2], o[3], gate_o, gate_i),
                down: w(o[4], o[5], down_o, down_i),
            });
        }
        Ok(DemandBatch(()))
    }

    /// Join the demand batch [`Pin::submit_layer`] put in flight: every cold slot
    /// holds its expert's bytes when this returns, so the descriptors handed back by
    /// `submit_layer` are safe to dereference (i.e. to launch MoE against).
    ///
    /// Consuming the `DemandBatch` is what makes "await without submit" and "await
    /// twice" uncompilable; `#[must_use]` covers most of "submit without await".
    pub fn await_layer(&mut self, batch: DemandBatch) -> Result<()> {
        let DemandBatch(()) = batch; // consumed: the batch is no longer outstanding
        self.stream.drain()?;
        #[cfg(debug_assertions)]
        self.inflight.clear();
        Ok(())
    }

    /// Is cross-layer prefetch enabled (`--prefetch`)?
    pub fn prefetch_enabled(&self) -> bool {
        self.prefetch_stream.is_some()
    }

    /// Max predicted experts to prefetch per layer (`--prefetch-depth`).
    pub fn prefetch_depth(&self) -> usize {
        self.prefetch_depth
    }

    /// Submit io_uring reads for `pred` — the predicted routed experts of `layer`
    /// (the NEXT MoE layer) — on the prefetch ring, NON-blocking. Already-resident
    /// predictions are skipped; the rest are `alloc_cold`-bound to fresh slots (with
    /// `off` set now, from cfg geometry) and their reads submitted so they run during
    /// the current layer's GPU compute. The batch is drained by the matching
    /// `submit_layer(layer)`. Call this AFTER the current layer's `submit_layer` and
    /// BEFORE its `launch_moe` — i.e. WHILE the current layer's own demand reads are
    /// still in flight, which is the point: the two batches then sit in the device
    /// queue together instead of end to end.
    ///
    /// Correctness (BOTH bounce and direct-DMA modes): `alloc_cold` evicts only from
    /// the policy's cold end, and the current layer's `top_k` experts are the
    /// most-recently-touched (MRU / protected) — phase 1a `get`s every hit and phase
    /// 1b `insert`s every miss immediately before this call. With `n_slots >= top_k +
    /// prefetch_depth` (enforced in `build`), the `prefetch_depth` cold evictions here
    /// can NEVER reuse a slot in the current layer's live descriptor set.
    ///
    /// That argument is about SLOT IDENTITY — the allocator's state, not the data's
    /// arrival — so it is unchanged by moving this call ahead of the demand join. The
    /// sequence of `Pool` operations it depends on (get×hits, alloc×misses, then
    /// contains/alloc_cold×pred) is byte-identical to before; only `Streamer::drain`,
    /// which touches no `Pool` state, moved. The slots this writes are therefore
    /// disjoint from BOTH what the running MoE reads AND what the in-flight demand
    /// reads are writing. `debug_assert`ed against `inflight` below.
    ///
    /// The data lands before anyone reads it in both modes: direct-DMA writes VMM
    /// asynchronously into a disjoint slot, bounce writes the prefetch ring's OWN
    /// pinned arena (never VMM) until its drain's memcpy. Either way the matching
    /// `submit_layer(layer)` drains the ring before L+1's MoE reads it.
    pub fn prefetch_layer(&mut self, layer: usize, pred: &[usize]) -> Result<()> {
        if self.prefetch_stream.is_none() {
            return Ok(());
        }
        let cfg = self.cfg;
        let sparse = (layer - cfg.dense_layers) * cfg.n_experts; // moe_table row base
        // Record the full predicted set (recall is measured over all predictions,
        // including those skipped below as already resident).
        self.predicted.clear();
        self.predicted.extend_from_slice(pred);
        self.predicted_layer = layer;
        // Raw slab ptr (no borrow held across the &mut stream reads), mirroring
        // `submit_layer`'s phase 1b.
        let slab = self.lru_pool.ptr_mut();
        // Disjoint field borrows: `prefetch_stream`, `pool`, `moe_table` are distinct
        // fields.
        let table = &self.moe_table;
        let expert_slot = self.expert_slot;
        let stream = match self.prefetch_stream.as_mut() {
            Some(s) => s,
            None => return Ok(()),
        };
        let pool = &mut self.pool;
        #[cfg(debug_assertions)]
        let inflight = &self.inflight; // disjoint field borrow, alongside `pool`
        let mut queued = 0usize;
        #[cfg(feature = "trace")]
        let (mut alloc_ns, mut queue_ns) = (0u128, 0u128);
        for &e in pred {
            let key = expert_key(layer, e);
            if pool.contains(key) {
                continue; // already resident — no fetch needed
            }
            #[cfg(feature = "trace")]
            let t = std::time::Instant::now();
            let slot = pool.alloc_cold(key)?;
            #[cfg(feature = "trace")]
            {
                alloc_ns += t.elapsed().as_nanos();
            }
            // The disjointness argument above, checked: this prefetch read must never
            // target a slot the demand batch is still filling.
            #[cfg(debug_assertions)]
            debug_assert!(
                !inflight.contains(&slot),
                "prefetch slot {slot} collides with an in-flight demand read"
            );
            let entry = &table[sparse + e];
            #[cfg(feature = "trace")]
            let t = std::time::Instant::now();
            stream_expert(stream, entry, slab, slot, expert_slot)?;
            #[cfg(feature = "trace")]
            {
                queue_ns += t.elapsed().as_nanos();
            }
            queued += 1;
        }
        #[cfg(feature = "trace")]
        {
            self.pf_alloc_ns += alloc_ns;
            self.pf_queue_ns += queue_ns;
        }
        if queued > 0 {
            // Kick the reads off NOW so they overlap this layer's compute.
            #[cfg(feature = "trace")]
            let t = std::time::Instant::now();
            stream.submit()?;
            #[cfg(feature = "trace")]
            {
                self.pf_submit_ns += t.elapsed().as_nanos();
            }
            self.prefetch_pending = true;
        }
        Ok(())
    }
}

/// Per-projection packed/scale byte lengths, derived from `cfg`. gate/up are
/// `[moe_inter, hidden]` (equal), down is `[hidden, moe_inter]`.
fn proj_lens(cfg: &ModelConfig) -> (usize, usize, usize, usize) {
    let (gate_p, gate_s) = (cfg.moe_inter * cfg.hidden.div_ceil(2), cfg.moe_inter * 4);
    let (down_p, down_s) = (cfg.hidden * cfg.moe_inter.div_ceil(2), cfg.hidden * 4);
    (gate_p, gate_s, down_p, down_s)
}

/// The largest single coalesced read superset a slot may deliver: the full
/// down|gate|up weight run (contiguous case) and the full scale run. Sizes the
/// bounce span.
fn coalesced_lens(cfg: &ModelConfig) -> (usize, usize) {
    let (gate_p, gate_s, down_p, down_s) = proj_lens(cfg);
    (down_p + gate_p + gate_p, down_s + gate_s + gate_s)
}

/// One routed expert's pool-slot stride (`expert_slot`), derived from `cfg`. A slot
/// holds each of the six tensors' O_DIRECT aligned superset — the weight region
/// `[down|gate|up]` then the scale region — so a coalesced run (or a shard-split
/// pair) always fits, whatever the run boundaries. Single source of truth for the
/// pool sizing in [`Pin::build`].
fn slot_geom(cfg: &ModelConfig) -> usize {
    let (gate_p, gate_s, down_p, down_s) = proj_lens(cfg);
    slot_span(down_p) + slot_span(gate_p) + slot_span(gate_p)   // weight region
        + slot_span(down_s) + slot_span(gate_s) + slot_span(gate_s) // scale region
}

/// Plan the O_DIRECT reads for one three-tensor group (down, gate, up — in that
/// disk order) into a slot region starting at `base`. Byte-adjacent same-shard
/// tensors coalesce into one read (the common case: all three → one read); a
/// shard-file boundary splits the run there. Each read lands at its own
/// block-aligned destination — the previous read's O_DIRECT superset end — so the
/// runs never overlap even though io_uring completion order is not submission
/// order. Appends `(fd, begin, len, dst)` reads at `reads[*n..]`, bumps `n`, and
/// returns the three tensors' slot-relative USEFUL-byte offsets (down, gate, up).
fn plan_group(
    t: &[(RawFd, usize, usize); 3],
    base: usize,
    reads: &mut [(RawFd, usize, usize, usize); 6],
    n: &mut usize,
) -> [usize; 3] {
    let mut off = [0usize; 3];
    let mut dst = base;
    let mut i = 0;
    while i < 3 {
        let (fd, run_begin, _) = t[i];
        let sub = run_begin & (ALIGN - 1); // O_DIRECT sub-block offset of the run start
        // Extend the run over byte-adjacent same-shard tensors.
        let (mut acc, mut j) = (0usize, i);
        while j < 3 && t[j].0 == fd && t[j].1 == run_begin + acc {
            off[j] = dst + sub + acc;
            acc += t[j].2;
            j += 1;
        }
        reads[*n] = (fd, run_begin, acc, dst);
        *n += 1;
        // Next run starts at this read's aligned superset end (block-aligned), so the
        // regions are disjoint.
        dst += (sub + acc).div_ceil(ALIGN) * ALIGN;
        i = j;
    }
    off
}

/// Resolve every routed expert's coalesced `(fd, begin, len, dst)` reads + the six
/// useful-byte offsets ONCE (A1/A2), indexed `(layer - dense_layers) * n_experts +
/// expert`. The cfg-dim cross-checks (packed/scale byte lengths) run HERE, so the
/// miss path trusts the table blindly. A missing/mismatched expert tensor fails
/// loud at build. Coalescing (down|gate|up contiguous → one read) is opportunistic:
/// [`plan_group`] falls back to per-run reads wherever a shard boundary splits it.
fn build_moe_table(
    snap: &Snapshot,
    cfg: &ModelConfig,
    direct_io: bool,
) -> Result<Vec<ExpertReads>> {
    let (gate_p, gate_s, down_p, down_s) = proj_lens(cfg);
    // Region bases: weights at the slot start, scales after the weight region.
    let scale_base = slot_span(down_p) + slot_span(gate_p) + slot_span(gate_p);
    let n_moe = cfg.n_layers - cfg.dense_layers;
    let mut table = Vec::with_capacity(n_moe * cfg.n_experts);
    for l in cfg.dense_layers..cfg.n_layers {
        for e in 0..cfg.n_experts {
            let base = format!("model.layers.{l}.mlp.experts.{e}");
            // (fd, begin, len) for one tensor, cross-checked against its cfg length.
            let spec = |suffix: &str, want: usize| -> Result<(RawFd, usize, usize)> {
                let name = format!("{base}.{suffix}");
                let (fd, b, len) = snap
                    .read_spec(&name, direct_io)
                    .with_context(|| format!("read_spec {name}"))?;
                ensure!(
                    len == want,
                    "{name}: {len} bytes, cfg expects {want} (config/snapshot dim mismatch)"
                );
                Ok((fd, b, len))
            };
            // DISK order is down, gate, up — the order the converter lays them out
            // (contiguously within a shard), so the runs coalesce front-to-back.
            let w = [
                spec("down_proj.weight", down_p)?,
                spec("gate_proj.weight", gate_p)?,
                spec("up_proj.weight", gate_p)?,
            ];
            let s = [
                spec("down_proj.weight.qs", down_s)?,
                spec("gate_proj.weight.qs", gate_s)?,
                spec("up_proj.weight.qs", gate_s)?,
            ];
            let mut reads = [(0 as RawFd, 0usize, 0usize, 0usize); 6];
            let mut n = 0usize;
            let wo = plan_group(&w, 0, &mut reads, &mut n);
            let so = plan_group(&s, scale_base, &mut reads, &mut n);
            // off is in slot-layout order: gate.p, gate.s, up.p, up.s, down.p, down.s.
            // `plan_group` returns (down, gate, up), so index 1=gate, 2=up, 0=down.
            let off = [wo[1], so[1], wo[2], so[2], wo[0], so[0]];
            table.push(ExpertReads { reads, off });
        }
    }
    Ok(table)
}

/// Queue an expert's coalesced O_DIRECT reads (weights then scales; usually one
/// each, more if a shard boundary split the run) into pool slot `slot` at
/// `slot*expert_slot`, from its precomputed [`ExpertReads`]. `len == 0` entries are
/// unused padding in the fixed array. The miss path and the build-time warm-start
/// seed share this: pure indexing + queueing, NO allocation, string hashing, or
/// re-validation (all done at [`build_moe_table`]).
fn stream_expert(
    stream: &mut Streamer,
    entry: &ExpertReads,
    pool: *mut u8,
    slot: usize,
    expert_slot: usize,
) -> Result<()> {
    let slot_base = slot * expert_slot;
    for &(fd, begin, len, rdst) in entry.reads.iter() {
        if len == 0 {
            continue; // unused array entry
        }
        // SAFETY: slot_base + rdst is block-aligned (both multiples of ALIGN) and
        // within the slot's region (slot < n_slots; regions sized to hold each read's
        // superset), owning >= slot_span(len) writable bytes until the drain.
        let dst = unsafe { pool.add(slot_base + rdst) };
        // SAFETY: dst is block-aligned and owns the read's aligned superset.
        unsafe { stream.queue(fd, begin, len, dst)? };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // plan_group must (a) coalesce a fully-contiguous down|gate|up run into ONE read
    // with tight useful offsets, and (b) split at a shard boundary into disjoint,
    // non-overlapping reads — the property that keeps io_uring's out-of-order
    // completions from clobbering each other.
    fn superset_end(dst: usize, begin: usize, len: usize) -> usize {
        let sub = begin & (ALIGN - 1);
        dst + (sub + len).div_ceil(ALIGN) * ALIGN
    }

    #[test]
    fn contiguous_run_coalesces_to_one_read() {
        let p = 6_291_456; // ALIGN-multiple projection length
        let t = [(5, ALIGN, p), (5, ALIGN + p, p), (5, ALIGN + 2 * p, p)];
        let mut reads = [(0, 0, 0, 0); 6];
        let mut n = 0;
        let off = plan_group(&t, 0, &mut reads, &mut n);
        assert_eq!(n, 1, "byte-adjacent same-fd tensors must be ONE read");
        assert_eq!(reads[0], (5, ALIGN, 3 * p, 0));
        assert_eq!(off, [0, p, 2 * p]); // aligned begin ⇒ sub=0, tight tiling
    }

    #[test]
    fn sub_block_offset_shifts_useful_bytes() {
        let p = 6_291_456;
        let t = [
            (5, ALIGN + 2048, p),
            (5, ALIGN + 2048 + p, p),
            (5, ALIGN + 2048 + 2 * p, p),
        ];
        let mut reads = [(0, 0, 0, 0); 6];
        let mut n = 0;
        let off = plan_group(&t, 0, &mut reads, &mut n);
        assert_eq!(n, 1);
        assert_eq!(off, [2048, 2048 + p, 2048 + 2 * p]); // all shifted by the sub-block
    }

    #[test]
    fn shard_boundary_splits_into_disjoint_reads() {
        let p = 6_291_456;
        // down|gate in shard fd=5, up in shard fd=6 (its own offset/sub).
        let t = [(5, ALIGN, p), (5, ALIGN + p, p), (6, 8192 + 17, p)];
        let mut reads = [(0, 0, 0, 0); 6];
        let mut n = 0;
        let off = plan_group(&t, 0, &mut reads, &mut n);
        assert_eq!(n, 2, "a shard boundary forces a second read");
        // Second read starts exactly at the first read's aligned superset end.
        let end0 = superset_end(reads[0].3, reads[0].1, reads[0].2);
        assert_eq!(reads[1].3, end0, "reads must not overlap");
        // up's useful bytes land at its read dst + its own sub-block offset.
        assert_eq!(off[2], reads[1].3 + (17 & (ALIGN - 1)));
        // down/gate still tight within the first read.
        assert_eq!(off[0], 0);
        assert_eq!(off[1], p);
    }

    #[test]
    fn every_projection_in_its_own_shard_uses_three_reads() {
        let p = 6_291_456;
        let t = [(5, ALIGN, p), (6, ALIGN, p), (7, ALIGN, p)];
        let mut reads = [(0, 0, 0, 0); 6];
        let mut n = 0;
        let _ = plan_group(&t, 0, &mut reads, &mut n);
        assert_eq!(n, 3);
        // Each read's dst is the prior read's superset end — strictly increasing, disjoint.
        assert_eq!(reads[1].3, superset_end(reads[0].3, reads[0].1, reads[0].2));
        assert_eq!(reads[2].3, superset_end(reads[1].3, reads[1].1, reads[1].2));
    }
}
