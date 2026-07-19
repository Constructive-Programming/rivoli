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
use crate::stream::{ALIGN, Streamer, slot_span};
use crate::usage::Usage;
use anyhow::{Context, Result, ensure};
use std::os::fd::RawFd;

/// The six cold reads that stream one routed expert into a pool slot, resolved
/// ONCE at [`Pin::build`] so the per-miss path never re-derives them. Order is
/// slot-layout order: gate packed, gate scale, up packed, up scale, down packed,
/// down scale — the same order as [`Pin::slot_dst`], into which each read lands.
///
/// - `reads[k]` = `(fd, file_begin, len)` from [`Snapshot::read_spec`] — a fixed
///   `(RawFd, offset, length)` for the whole run (the mmap'd shard never moves).
///   The fd is O_DIRECT or buffered per `Pin::build`'s `direct_io`; the offset/len
///   are identical either way, so the aligned superset read is mode-agnostic.
/// - `off[k]` = the slot-relative offset where read `k`'s USEFUL bytes land
///   (`slot_dst[k] + (file_begin & (ALIGN-1))`, the block-aligned sub-block padding).
///   Precomputed so descriptor build is a pure index, not a re-computation.
///
/// All cfg-dim cross-checks (packed/scale byte lengths vs the slot geometry) are
/// done at build when this table is populated, so the miss path trusts it blindly.
#[derive(Clone, Copy)]
struct ExpertReads {
    reads: [(RawFd, usize, usize); 6],
    off: [usize; 6],
}

/// A resolved int4 weight matrix in the tier: device pointers + dims.
#[derive(Clone, Copy)]
pub struct Weight {
    pub packed: *const u8,
    pub scale: *const f32,
    pub o_dim: usize,
    pub i_dim: usize,
}

/// A resolved int8 weight matrix (embed table / lm_head).
#[derive(Clone, Copy)]
pub struct Weight8 {
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

/// The MTP (layer-`n_layers`) resident weights for the device draft path: a full
/// transformer layer (attention + `n_experts` routed experts + shared) plus the
/// MTP glue (enorm/hnorm/eh_proj) and the pre-head norm. Experts are placed
/// RESIDENT here for M2's oracle-match gate; M3 will stream them through the
/// shared LRU pool. `Some` only when the pin is built with `want_mtp`.
pub struct MtpPin {
    pub input_ln: *const f32,
    pub post_ln: *const f32,
    pub enorm: *const f32,   // RMSNorm on the next-token embedding
    pub hnorm: *const f32,   // RMSNorm on the trunk hidden
    pub shnorm: *const f32,  // shared_head.norm (pre-head)
    pub eh_proj: *const u16, // bf16 [hidden, 2*hidden]
    pub q_a: Weight,
    pub q_a_ln: *const f32,
    pub q_b: Weight,
    pub kv_a: Weight,
    pub kv_a_ln: *const f32,
    pub kv_b: Weight,
    pub o_proj: Weight,
    pub gate_w: *const f32,  // router gate [n_experts, hidden] F32
    pub gate_bias: Vec<f32>, // e_score_correction_bias, host (routing)
    pub experts: Vec<Mlp>,   // n_experts resident routed experts
    pub shared: Mlp,
}

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
    pub embed: Weight8,
    pub lm_head: Weight8,
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
    /// strides are re-derived from `cfg` via [`slot_geom`]).
    lru_pool: VmmBuf,
    expert_slot: usize,
    /// Per-(MoE layer, expert) precomputed streaming plan (A1/A2): the six fixed
    /// `(fd, begin, len)` reads + their slot-relative useful-byte offsets. Indexed
    /// by `(layer - dense_layers) * n_experts + expert`. Built ONCE; the miss and
    /// prefetch paths index it and queue reads with zero allocation, zero string
    /// hashing, and zero re-validation.
    moe_table: Vec<ExpertReads>,
    /// Slot-relative aligned DESTINATION offset of each of the six reads (gate
    /// packed | gate scale | up packed | up scale | down packed | down scale).
    /// Pure function of `cfg` (run-invariant), computed once at build.
    slot_dst: [usize; 6],
    /// Adaptive routed tier: (layer,expert) -> slot, policy-evicted (`--cache-policy`).
    /// Online priming that keeps this run's hot experts resident.
    pool: Pool,
    /// io_uring O_DIRECT reader — a layer's cold misses submit as one queue-depth
    /// batch straight into their LRU slots (folds the old mmap-warm + memcpy).
    stream: Streamer,
    /// Cross-layer prefetch (`--prefetch`): a SECOND io_uring ring, dedicated to the
    /// predicted next-layer experts. `Some` iff prefetch is enabled. Its reads are
    /// SUBMITTED (non-blocking) after the current layer's own `resolve_layer` drain,
    /// run on the NVMe/DMA side during the current layer's MoE compute, and are
    /// DRAINED at the next `resolve_layer` — hiding the ~5ms/miss fetch behind
    /// compute. A separate ring keeps in-flight prefetch reads from entangling with a
    /// layer's synchronous miss batch.
    prefetch_stream: Option<Streamer>,
    /// Max predicted experts prefetched per layer (`--prefetch-depth`, the top-N by
    /// router score). Bounded by the idle-NVMe-during-compute window on this
    /// bandwidth-bound path; the caller slices to this before `prefetch_layer`.
    prefetch_depth: usize,
    /// An outstanding prefetch batch is in flight and must be drained by the next
    /// `resolve_layer` before its slots are read.
    prefetch_pending: bool,
    /// The predicted expert set submitted for the layer named in `predicted_layer`
    /// (recall accounting at the matching `resolve_layer`).
    predicted: Vec<usize>,
    predicted_layer: usize,
    /// Stats over the run.
    pub hits: u64,
    pub misses: u64,
    /// Prefetch recall: `pred_correct` predicted experts were actually selected out
    /// of `pred_total` predicted (over all prefetched layers).
    pub pred_total: u64,
    pub pred_correct: u64,
    /// Nanoseconds spent blocked in the prefetch-ring drain (the part of the fetch
    /// that did NOT overlap the previous layer's compute). Near-zero means the reads
    /// were fully hidden; large means the overlap window was too small / NVMe-bound.
    pub prefetch_wait_ns: u128,
    /// Optional access-trace sink (`--trace`): one line per resolved MoE layer, the
    /// space-separated `(layer,expert)` keys the LRU looked up, in access order.
    /// Feeds the offline cache-policy simulator (`src/cache.rs`, `bin/replay`).
    trace: Option<std::io::BufWriter<std::fs::File>>,
    /// The resident MTP (layer-`n_layers`) weights for the device draft path;
    /// `Some` only when built with `want_mtp`. See [`MtpPin`].
    mtp: Option<MtpPin>,
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
fn place_i8(tier: &mut DeviceTier, snap: &Snapshot, name: &str, i_dim: usize) -> Result<Weight8> {
    let m = snap.int8(name, i_dim)?;
    let (plen, slen, o_dim) = (m.packed.len(), m.scale.len(), m.o_dim);
    let packed = tier.reserve(plen)?;
    let scale = tier.reserve(slen)?;
    // SAFETY: packed/scale each own their reserved byte counts.
    unsafe {
        snap.read_into(&format!("{name}.weight"), packed, plen, true)?;
        snap.read_into(&format!("{name}.weight.qs"), scale, slen, true)?;
    }
    Ok(Weight8 {
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

/// Place the MTP layer (index `cfg.n_layers`) resident: glue + attention stack +
/// gate + shared + all `n_experts` routed experts. Footprint = [`mtp_bytes`].
fn place_mtp(tier: &mut DeviceTier, snap: &Snapshot, cfg: &ModelConfig) -> Result<MtpPin> {
    let l = cfg.n_layers;
    let lb = format!("model.layers.{l}");
    let a = format!("{lb}.self_attn");
    let mut experts = Vec::with_capacity(cfg.n_experts);
    for e in 0..cfg.n_experts {
        experts.push(place_mlp(
            tier,
            snap,
            &format!("{lb}.mlp.experts.{e}"),
            cfg.hidden,
            cfg.moe_inter,
        )?);
    }
    let gate_bias = crate::quant::read_f32(snap.typed(
        &format!("{lb}.mlp.gate.e_score_correction_bias"),
        Dtype::F32,
    )?);
    ensure!(
        gate_bias.len() == cfg.n_experts,
        "MTP gate bias has {} entries, expected {}",
        gate_bias.len(),
        cfg.n_experts
    );
    Ok(MtpPin {
        input_ln: place_f32(tier, snap, &format!("{lb}.input_layernorm.weight"))?,
        post_ln: place_f32(tier, snap, &format!("{lb}.post_attention_layernorm.weight"))?,
        enorm: place_f32(tier, snap, &format!("{lb}.enorm.weight"))?,
        hnorm: place_f32(tier, snap, &format!("{lb}.hnorm.weight"))?,
        shnorm: place_f32(tier, snap, &format!("{lb}.shared_head.norm.weight"))?,
        eh_proj: place_bf16(tier, snap, &format!("{lb}.eh_proj.weight"))?,
        q_a: place_i4(tier, snap, &format!("{a}.q_a_proj"), cfg.hidden)?,
        q_a_ln: place_f32(tier, snap, &format!("{a}.q_a_layernorm.weight"))?,
        q_b: place_i4(tier, snap, &format!("{a}.q_b_proj"), cfg.q_lora_rank)?,
        kv_a: place_i4(tier, snap, &format!("{a}.kv_a_proj_with_mqa"), cfg.hidden)?,
        kv_a_ln: place_f32(tier, snap, &format!("{a}.kv_a_layernorm.weight"))?,
        kv_b: place_i4(tier, snap, &format!("{a}.kv_b_proj"), cfg.kv_lora_rank)?,
        o_proj: place_i4(
            tier,
            snap,
            &format!("{a}.o_proj"),
            cfg.n_heads * cfg.v_head_dim,
        )?,
        gate_w: place_f32(tier, snap, &format!("{lb}.mlp.gate.weight"))?,
        gate_bias,
        shared: place_mlp(
            tier,
            snap,
            &format!("{lb}.mlp.shared_experts"),
            cfg.hidden,
            cfg.moe_inter * cfg.n_shared,
        )?,
        experts,
    })
}

/// Resident bytes for the MTP layer (`place_mtp`) — one MoE layer's attn + gate +
/// shared + `n_experts` experts, plus the bf16 eh_proj and the extra f32 norms.
fn mtp_bytes(cfg: &ModelConfig) -> usize {
    let rb = crate::quant::row_bytes;
    let i4 = |o: usize, i: usize| o * rb(i) + o * 4;
    let f32n = |n: usize| n * 4;
    let qk = cfg.qk_head_dim();
    // norms: input, post, enorm, hnorm, shnorm (hidden) + q_a_ln + kv_a_ln
    let mut t = 5 * f32n(cfg.hidden) + f32n(cfg.q_lora_rank) + f32n(cfg.kv_lora_rank);
    t += i4(cfg.q_lora_rank, cfg.hidden); // q_a
    t += i4(cfg.n_heads * qk, cfg.q_lora_rank); // q_b
    t += i4(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden); // kv_a
    t += i4(
        cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
        cfg.kv_lora_rank,
    ); // kv_b
    t += i4(cfg.hidden, cfg.n_heads * cfg.v_head_dim); // o_proj
    t += f32n(cfg.n_experts * cfg.hidden); // gate
    t += cfg.hidden * (2 * cfg.hidden) * 2; // eh_proj bf16 [hidden, 2*hidden]
    let si = cfg.moe_inter * cfg.n_shared;
    t += 2 * i4(si, cfg.hidden) + i4(cfg.hidden, si); // shared
    t += cfg.n_experts * (2 * i4(cfg.moe_inter, cfg.hidden) + i4(cfg.hidden, cfg.moe_inter));
    t
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
}

impl Pool {
    fn new(n: usize, policy: &str) -> Result<Self> {
        Ok(Self {
            policy: cache::make(policy, n)
                .with_context(|| format!("unknown --cache-policy {policy:?} (lru|2q|arc)"))?,
            slot_of: std::collections::HashMap::with_capacity(n),
            #[cfg(debug_assertions)]
            key_of: vec![None; n],
            free: (0..n).rev().collect(), // pop() hands out 0,1,2,...
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
            Some(slot)
        } else {
            None
        }
    }

    /// Allocate a slot for a NEW `key` (miss): reuse the policy-evicted key's slot,
    /// else a free slot. The caller then fills the slot and sets `off[slot]`.
    fn alloc(&mut self, key: u32) -> Result<usize> {
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
        Ok(slot)
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
        prefetch: bool,
        prefetch_depth: usize,
        direct_io: bool,
        want_indexer: bool,
        want_mtp: bool,
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
        // The static tier holds only the always-resident set; the rest of the
        // budget goes to the routed LRU. Size the tier to the footprint computed
        // from `cfg` (not a fixed cap that strands 1-5 GiB the LRU could use, and
        // not `capacity`, which would hog everything the LRU needs). The slack
        // absorbs the per-reservation 256-byte alignment padding and a small
        // margin; anything left over widens the LRU below.
        const SLACK: usize = 256 << 20; // 256 MiB
        let n_full = full.iter().filter(|&&f| f).count();
        let resident = resident_bytes(cfg)
            + n_full * indexer_bytes(cfg)
            + if want_mtp { mtp_bytes(cfg) } else { 0 };
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

        // The MTP (layer-n_layers) draft layer, resident when requested.
        let mtp = if want_mtp {
            Some(place_mtp(&mut tier, snap, cfg)?)
        } else {
            None
        };

        // The routed tier is an adaptive LRU, not a static frequency pin: N reused
        // slots stream experts in on demand and keep this run's actually-hot ones
        // (online priming). Each slot holds a projection's O_DIRECT aligned superset
        // (packed then scale, each `slot_span` — block-aligned + straddle pad — so
        // reads land in place and descriptors point at the sub-block offsets). Size
        // to the budget left after the always-resident set.
        let (_, expert_slot) = slot_geom(cfg);
        // Precompute the per-(layer,expert) streaming plan ONCE (A1/A2): six fixed
        // (fd, begin, len) reads + slot-relative useful-byte offsets, cfg-dim cross-
        // checked here. The miss/prefetch paths then just index + queue.
        let slot_dst = slot_dst_of(cfg);
        let moe_table = build_moe_table(snap, cfg, &slot_dst, direct_io)?;
        let budget = capacity.saturating_sub(tier_cap);
        // Floor the pool at `top_k + prefetch_depth` when prefetching: this makes
        // cross-layer prefetch's evictions provably DISJOINT from the current layer's
        // live experts. The current layer's `top_k` routed experts are the
        // most-recently-touched (MRU / protected) after `resolve_layer`; `alloc_cold`
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
        let mut pool = Pool::new(n_slots, cache_policy)?;
        // Ring sized for one layer's worst case: top_k misses x 3 proj x 2 tensors.
        let ring = (cfg.top_k * 6).next_power_of_two() * 2;
        // Bounce span = largest single projection superset (gate/up packed =
        // moe_inter*rb(hidden); down packed = hidden*rb(moe_inter)); scales are tiny.
        let span = slot_span(cfg.moe_inter * cfg.hidden.div_ceil(2))
            .max(slot_span(cfg.hidden * cfg.moe_inter.div_ceil(2)));
        let mut stream = Streamer::new(ring as u32, span, bounce)?;
        // Optional warm start (`--pre-seed`): seed the pool with the hottest experts
        // from .coli_usage so the first tokens hit at colibri's rate while the LRU
        // adapts. Worth ~+6.8pt on the first tokens but transient (the LRU warms up
        // in a few tokens regardless) and it dominates build time (~23s), so it is
        // OFF by default — enable it only for slow disks. Bounded by the pool size;
        // drained in queue-depth batches.
        if pre_seed {
            let slab = lru_pool.ptr_mut();
            let batch = (ring / 6).max(1); // experts per drain (6 reads each)
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
                stream_expert(&mut stream, entry, &slot_dst, slab, slot, expert_slot)?;
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
            Some(Streamer::new(ring as u32, span, bounce)?)
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
            slot_dst,
            pool,
            stream,
            prefetch_stream,
            prefetch_depth,
            prefetch_pending: false,
            predicted: Vec::new(),
            predicted_layer: usize::MAX,
            hits: 0,
            misses: 0,
            pred_total: 0,
            pred_correct: 0,
            prefetch_wait_ns: 0,
            trace: trace_path
                .map(|p| -> Result<_> {
                    Ok(std::io::BufWriter::new(
                        std::fs::File::create(p).with_context(|| format!("open trace {p}"))?,
                    ))
                })
                .transpose()?,
            mtp,
        })
    }

    /// The resident MTP layer weights (`Some` iff built with `want_mtp`).
    pub fn mtp(&self) -> Option<&MtpPin> {
        self.mtp.as_ref()
    }

    /// Host router correction bias for a MoE `layer` (len n_experts).
    pub fn moe_bias(&self, layer: usize) -> &[f32] {
        &self.moe_bias[layer - self.cfg.dense_layers]
    }

    /// Resolve a MoE layer's `sel` routed experts to `Mlp` descriptors via the
    /// adaptive LRU: a hit returns its resident slot's pointers (no I/O); a miss
    /// evicts the coldest slot and streams the expert in. All of the layer's misses
    /// submit as ONE queue-depth io_uring O_DIRECT batch, joined once. Fills the
    /// caller-owned `out` (cleared first, reused across tokens) so the decode hot
    /// path allocates nothing here on the all-hits steady state.
    pub fn resolve_layer(&mut self, layer: usize, sel: &[usize], out: &mut Vec<Mlp>) -> Result<()> {
        let cfg = self.cfg;
        let sparse = (layer - cfg.dense_layers) * cfg.n_experts; // moe_table row base
        // `sel` is one layer's routed pick (top_k + shared); a fixed slot scratch
        // avoids a per-layer alloc, mirroring `cache::access_batch`'s 32-wide buffer.
        ensure!(
            sel.len() <= 32,
            "resolve_layer: {} experts exceeds the 32-slot scratch",
            sel.len()
        );
        // Cross-layer prefetch consume: an outstanding prefetch batch (submitted
        // during the PREVIOUS layer's compute) targets THIS layer. Wait for its reads
        // + bounce copy (+ device sync inside `drain`), so every predicted-correct
        // expert is resident in VMM BEFORE phase 1a's hit scan and this layer's MoE
        // read it. The predicted-correct keys were already `alloc_cold`-bound to their
        // slots with `off` set (at submit time), so they now surface as normal hits.
        if self.prefetch_pending {
            if let Some(s) = self.prefetch_stream.as_mut() {
                let t = std::time::Instant::now();
                s.drain()?;
                self.prefetch_wait_ns += t.elapsed().as_nanos();
            }
            self.prefetch_pending = false;
            // Recall: how many predicted experts are in this layer's actual `sel`.
            if self.predicted_layer == layer {
                let hit = self.predicted.iter().filter(|e| sel.contains(e)).count();
                self.pred_total += self.predicted.len() as u64;
                self.pred_correct += hit as u64;
            }
        }
        // Trace sink (--trace): the exact LRU keys this layer looks up, access order.
        // Write each key straight to the BufWriter (no per-layer Vec<String>+join);
        // a leading space before all but the first keeps the output byte-identical
        // (space-separated keys, no trailing space, one newline per layer).
        if let Some(w) = &mut self.trace {
            use std::io::Write;
            for (j, &e) in sel.iter().enumerate() {
                if j > 0 {
                    write!(w, " ").context("write trace")?;
                }
                write!(w, "{}", expert_key(layer, e)).context("write trace")?;
            }
            writeln!(w).context("write trace")?;
        }
        // Phase 1a: touch EVERY hit first. Doing this before any miss's `alloc()`
        // bumps each hit's LRU tick above the eviction candidates, so a later miss
        // cannot evict a same-layer would-be hit out from under itself (which would
        // force a needless re-stream and understate the hit rate). `None` marks a
        // slot still to be filled in phase 1b.
        let mut slots: [Option<usize>; 32] = [None; 32];
        for (i, &e) in sel.iter().enumerate() {
            if let Some(slot) = self.pool.get(expert_key(layer, e)) {
                self.hits += 1;
                slots[i] = Some(slot);
            }
        }
        // Phase 1b: allocate + QUEUE the misses, now that all hits are protected.
        for (i, &e) in sel.iter().enumerate() {
            if slots[i].is_some() {
                continue;
            }
            self.misses += 1;
            let slot = self.pool.alloc(expert_key(layer, e))?;
            // ptr_mut() yields a raw ptr, so no borrow of lru_pool is held across
            // the &mut self.stream reads. Table entry + slot_dst are disjoint fields
            // from stream — no alloc, no string hashing, no re-validation per miss.
            let slab = self.lru_pool.ptr_mut();
            let entry = &self.moe_table[sparse + e];
            stream_expert(
                &mut self.stream,
                entry,
                &self.slot_dst,
                slab,
                slot,
                self.expert_slot,
            )?;
            slots[i] = Some(slot);
        }
        // Phase 2: ONE join for the whole layer's cold reads.
        self.stream.drain()?;
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
            let slot = slots[i].context("resolve_layer: unresolved expert slot")?;
            let o = self.moe_table[sparse + e].off;
            // SAFETY: slot base within the slab; reads that filled it have joined.
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
    /// `resolve_layer(layer)`. Call this AFTER the current layer's own `resolve_layer`
    /// drain (main ring quiescent) and BEFORE its `launch_moe`.
    ///
    /// Correctness (BOTH bounce and direct-DMA modes): `alloc_cold` evicts only the
    /// LRU, and the current layer's `top_k` experts are the most-recently-touched
    /// (MRU / protected) after its `resolve_layer`. With `n_slots >= top_k +
    /// prefetch_depth` (enforced in `build`), the `prefetch_depth` cold evictions here
    /// can NEVER reuse a slot in the current layer's live descriptor set. So the
    /// prefetch reads target slots DISJOINT from what the running MoE reads — whether
    /// they land async (direct DMA into VMM) or at the drain-time memcpy (bounce),
    /// they never overwrite live data. The matching `resolve_layer(layer)` drains the
    /// ring before use, so the data is present before L+1's MoE reads it.
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
        // `resolve_layer`'s phase 1b.
        let slab = self.lru_pool.ptr_mut();
        // Disjoint field borrows: `prefetch_stream`, `pool`, `moe_table`, `slot_dst`
        // are distinct fields.
        let table = &self.moe_table;
        let slot_dst = &self.slot_dst;
        let expert_slot = self.expert_slot;
        let stream = match self.prefetch_stream.as_mut() {
            Some(s) => s,
            None => return Ok(()),
        };
        let pool = &mut self.pool;
        let mut queued = 0usize;
        for &e in pred {
            let key = expert_key(layer, e);
            if pool.contains(key) {
                continue; // already resident — no fetch needed
            }
            let slot = pool.alloc_cold(key)?;
            let entry = &table[sparse + e];
            stream_expert(stream, entry, slot_dst, slab, slot, expert_slot)?;
            queued += 1;
        }
        if queued > 0 {
            // Kick the reads off NOW so they overlap this layer's compute.
            stream.submit()?;
            self.prefetch_pending = true;
        }
        Ok(())
    }
}

/// One routed expert's pool-slot geometry, derived from `cfg`: `(gate_slot,
/// expert_slot)`. A slot is laid out gate|up|down; `gate` and `up` each span
/// `gate_slot` (packed superset + scale superset), `down` the rest, and the whole
/// expert occupies `expert_slot`. The single source of truth for both the pool
/// sizing in [`Pin::build`] and the per-region offsets in [`slot_dst_of`].
fn slot_geom(cfg: &ModelConfig) -> (usize, usize) {
    let rb_h = cfg.hidden.div_ceil(2);
    let rb_i = cfg.moe_inter.div_ceil(2);
    let gate_slot = slot_span(cfg.moe_inter * rb_h) + slot_span(cfg.moe_inter * 4);
    let down_slot = slot_span(cfg.hidden * rb_i) + slot_span(cfg.hidden * 4);
    (gate_slot, 2 * gate_slot + down_slot) // gate | up | down, one slot
}

/// The six slot-relative, block-aligned DESTINATION offsets a slot's reads land
/// at — gate packed | gate scale | up packed | up scale | down packed | down
/// scale — a pure function of `cfg` (run-invariant). Within each `gate_slot`/
/// `down` region the packed superset comes first, the scale superset next
/// (`slot_span(packed_len)` in). Matches the layout the old `stream_expert` built
/// per miss; now computed once.
fn slot_dst_of(cfg: &ModelConfig) -> [usize; 6] {
    let (gate_slot, _) = slot_geom(cfg);
    let gate_p = cfg.moe_inter * cfg.hidden.div_ceil(2);
    let down_p = cfg.hidden * cfg.moe_inter.div_ceil(2);
    let sp = slot_span; // packed superset length within a region
    [
        0,                          // gate packed
        sp(gate_p),                 // gate scale
        gate_slot,                  // up packed
        gate_slot + sp(gate_p),     // up scale
        2 * gate_slot,              // down packed
        2 * gate_slot + sp(down_p), // down scale
    ]
}

/// Resolve every routed expert's six `(fd, begin, len)` reads + useful-byte
/// offsets ONCE (A1/A2), indexed `(layer - dense_layers) * n_experts + expert`.
/// The cfg-dim cross-checks that used to run per miss (packed/scale byte lengths
/// vs the slot geometry that sized each region) run HERE, so the miss path trusts
/// the table blindly. A missing/mismatched expert tensor fails loud at build.
fn build_moe_table(
    snap: &Snapshot,
    cfg: &ModelConfig,
    slot_dst: &[usize; 6],
    direct_io: bool,
) -> Result<Vec<ExpertReads>> {
    // cfg-expected packed/scale byte lengths (the same budget the slot regions were
    // sized from). gate/up: [moe_inter, hidden]; down: [hidden, moe_inter].
    let rb_h = cfg.hidden.div_ceil(2);
    let rb_i = cfg.moe_inter.div_ceil(2);
    let (gate_p, gate_s) = (cfg.moe_inter * rb_h, cfg.moe_inter * 4);
    let (down_p, down_s) = (cfg.hidden * rb_i, cfg.hidden * 4);
    // (proj suffix, expected packed len, expected scale len), in slot-layout order.
    let projs = [
        ("gate_proj", gate_p, gate_s),
        ("up_proj", gate_p, gate_s),
        ("down_proj", down_p, down_s),
    ];
    let n_moe = cfg.n_layers - cfg.dense_layers;
    let mut table = Vec::with_capacity(n_moe * cfg.n_experts);
    for l in cfg.dense_layers..cfg.n_layers {
        for e in 0..cfg.n_experts {
            let base = format!("model.layers.{l}.mlp.experts.{e}");
            let mut reads = [(0 as RawFd, 0usize, 0usize); 6];
            let mut off = [0usize; 6];
            for (pi, &(proj, exp_p, exp_s)) in projs.iter().enumerate() {
                let (kp, ks) = (pi * 2, pi * 2 + 1); // packed/scale slot indices
                let (pfd, pb, plen) = snap
                    .read_spec(&format!("{base}.{proj}.weight"), direct_io)
                    .with_context(|| format!("read_spec {base}.{proj}.weight"))?;
                ensure!(
                    plen == exp_p,
                    "{base}.{proj}.weight: {plen} packed bytes, cfg expects {exp_p} \
                     (config/snapshot dim mismatch)"
                );
                let (sfd, sb, slen) = snap
                    .read_spec(&format!("{base}.{proj}.weight.qs"), direct_io)
                    .with_context(|| format!("read_spec {base}.{proj}.weight.qs"))?;
                ensure!(
                    slen == exp_s,
                    "{base}.{proj}.weight.qs: {slen} scale bytes, cfg expects {exp_s} \
                     (config/snapshot dim mismatch)"
                );
                reads[kp] = (pfd, pb, plen);
                reads[ks] = (sfd, sb, slen);
                // Useful bytes land `sub` (= begin's O_DIRECT sub-block offset) past
                // the aligned region destination — exactly `Streamer::queue`'s return.
                off[kp] = slot_dst[kp] + (pb & (ALIGN - 1));
                off[ks] = slot_dst[ks] + (sb & (ALIGN - 1));
            }
            table.push(ExpertReads { reads, off });
        }
    }
    Ok(table)
}

/// Queue an expert's 6 O_DIRECT reads into pool slot `slot` (regions gate|up|down
/// at `slot*expert_slot`), from its precomputed [`ExpertReads`]. The miss path and
/// the build-time warm-start seed share this: pure indexing + queueing, NO
/// allocation, string hashing, or re-validation (all done at [`build_moe_table`]).
fn stream_expert(
    stream: &mut Streamer,
    entry: &ExpertReads,
    slot_dst: &[usize; 6],
    pool: *mut u8,
    slot: usize,
    expert_slot: usize,
) -> Result<()> {
    let slot_base = slot * expert_slot;
    for (i, &(fd, begin, len)) in entry.reads.iter().enumerate() {
        // SAFETY: slot_base + slot_dst[i] is block-aligned (both multiples of ALIGN)
        // and within the slot's region (slot < n_slots; regions sized to hold each
        // read's superset), owning >= slot_span(len) writable bytes until the drain.
        let dst = unsafe { pool.add(slot_base + slot_dst[i]) };
        // SAFETY: dst is block-aligned and owns the read's aligned superset.
        unsafe { stream.queue(fd, begin, len, dst)? };
    }
    Ok(())
}
