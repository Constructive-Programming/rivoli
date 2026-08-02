//! The resident weight set: places every weight the forward pass reads into the
//! [`DeviceTier`] once at startup and resolves each to a raw device pointer, so
//! per-token decode never touches the host for weights.
//!
//! Always resident (read every token): the per-layer norms, the full attention
//! stack (q_a/q_b/kv_a/kv_b/o_proj + their layernorms, all fp8-e4m3 block-scaled),
//! the dense-layer MLPs (fp8), and — for MoE layers — the router gate (f32) and the
//! routed-format shared expert (VQ-int3, or int4 under `--i4`). The int8 embed and
//! lm_head plus the f32 final norm are global. The footprint is computed from `cfg`
//! at build ([`resident_bytes`]) so the tier is sized to what it holds and the rest
//! of the device budget grows the routed pool.
//!
//! The 256 routed experts/layer are served by an adaptive pool ([`cache`] policy +
//! one VMM slab): a hit reuses the resident slot; a miss evicts the coldest and
//! streams the expert in via io_uring O_DIRECT (`.vq3`/`.i4` block = one aligned read).
//!
//! Needs a backend (`rocm` or `vulkan`): without a device there is nothing to pin.
#![cfg(any(feature = "rocm", feature = "vulkan"))]

use crate::artifact::config::Mode;
use crate::artifact::format::{Dtype, ExpertSet, FormatMeta, Safetensors, SetDims, load_codebooks};
use crate::artifact::model::ModelConfig;
use crate::artifact::quant::{
    VQ_DIM, VQ_K, i4_expert_bytes, i4_slot_offsets, vq_expert_bytes, vq_slot_offsets,
};
use crate::backend::memcpy_dtod;
use crate::fetch::asyncfetch::{AsyncFetch, ReadSpec, Ticket};
use crate::fetch::stream::{Streamer, slot_span};
use crate::memory::arena::{Arena, Reloc, Step};
use crate::memory::cache;
use crate::memory::device::{DeviceTier, VmmBuf};
use crate::memory::hybrid::HybridPolicy;
use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;
use std::os::fd::RawFd;

/// A resolved fp8-e4m3 block-scaled weight matrix in the tier: device pointers +
/// dims + the `weight_scale_inv` block size. Consumed by `launch_gemv_fp8` (attn +
/// dense projections) and, for kv_b, by the MLA absorb/value kernels.
#[derive(Clone, Copy)]
pub struct Fp8Weight {
    pub packed: *const u8,
    pub scale: *const f32,
    /// `weight_scale_inv` tile size — the one field [`Int8Weight`] (per-ROW scales) has no
    /// analogue for, so it sits next to `scale` rather than trailing the dims.
    pub block: usize,
    pub o_dim: usize,
    pub i_dim: usize,
}

/// A dense SwiGLU MLP's three fp8 projections, resolved.
#[derive(Clone, Copy)]
pub struct Fp8Mlp {
    pub gate: Fp8Weight,
    pub up: Fp8Weight,
    pub down: Fp8Weight,
}

/// A resolved int8 per-row weight (embed / lm_head): packed bytes + per-row f32
/// scale. Consumed by `launch_embed_i8_row` (embed) / `launch_gemv_i8` (lm_head).
#[derive(Clone, Copy)]
pub struct Int8Weight {
    pub packed: *const u8,
    pub scale: *const f32,
    pub o_dim: usize,
    pub i_dim: usize,
}

/// One MoE projection resolved to two device pointers into its expert slot. For VQ
/// these are the packed 12-bit indices + bf16 group scales (decoded against a
/// codebook); reused as the int4 carrier — then `indices` is the packed 4-bit
/// weights and `scales` holds the f32 per-row-scale *address* (reinterpreted at the
/// launch site, see `place_shared`). The dims are the kernel's args, not carried
/// here. Consumed by `launch_moe_expert_range`(`_i4`) via `desc_of_vq`.
#[derive(Clone, Copy)]
pub struct VqWeight {
    pub indices: *const u8,
    pub scales: *const u16,
}

/// A SwiGLU MLP's three resolved projections (routed or shared expert), one format.
#[derive(Clone, Copy)]
pub struct MlpVq {
    pub gate: VqWeight,
    pub up: VqWeight,
    pub down: VqWeight,
}

/// Resolve one projection's two pointers at slot-relative offsets `(ioff, soff)`
/// from an expert-block base — the single builder shared by the resident shared
/// expert ([`place_shared`]) and the streamed routed experts (`submit_layer`).
#[inline]
fn vqweight_at(base: *const u8, ioff: usize, soff: usize) -> VqWeight {
    VqWeight {
        // SAFETY: both offsets lie within the expert block at `base` (fixed layout).
        indices: unsafe { base.add(ioff) },
        scales: unsafe { base.add(soff) } as *const u16,
    }
}

/// One layer's MLP: dense fp8 for the first `dense_layers`, MoE after. The MoE
/// shared expert is the routed format (VQ-int3, or int4 under `--i4`; from block
/// `n_experts`), folded into the single MoE batch alongside the routed picks.
pub enum LayerMlp {
    Dense(Fp8Mlp),
    Moe {
        gate_w: *const f32, // router gate [n_experts, hidden] (F32), device
        shared: MlpVq,      // routed-format shared expert (always resident)
    },
}

/// A full layer's resident DSA lightning-indexer weights: the fp8 projections
/// (`wk`/`wq_b`/`weights_proj` — fp8-e4m3 in this checkpoint) + the f32-widened
/// `k_norm` (weight + bias). Only "full" layers own one; "shared" layers reuse a
/// preceding full layer's selection and carry `None`.
#[derive(Clone, Copy)]
pub struct IndexerPin {
    pub wk: Fp8Weight,            // [index_head_dim, hidden] (fp8)
    pub wq_b: Fp8Weight,          // [index_n_heads·index_head_dim, q_lora_rank] (fp8)
    pub weights_proj: *const f32, // [index_n_heads, hidden] (bf16→f32; gemv_f32)
    pub k_norm_w: *const f32,     // [index_head_dim] (widened from bf16)
    pub k_norm_b: *const f32,     // [index_head_dim]
}

/// One layer's resolved weights.
pub struct LayerPin {
    pub input_ln: *const f32,
    pub post_ln: *const f32,
    pub q_a: Fp8Weight,
    pub q_a_ln: *const f32,
    pub q_b: Fp8Weight,
    pub kv_a: Fp8Weight,
    pub kv_a_ln: *const f32,
    pub kv_b: Fp8Weight,
    pub o_proj: Fp8Weight,
    pub mlp: LayerMlp,
    /// The DSA indexer weights, `Some` for full layers when the pin is built with
    /// `want_indexer` (dsa/misa modes). `None` for dense/streaming or shared layers.
    pub indexer: Option<IndexerPin>,
}

/// The MTP (multi-token prediction) head's own weights, on top of the ordinary
/// [`LayerPin`] it is stored as (`layers[cfg.n_layers]` — the checkpoint numbers it
/// one past the last real layer, and it is a full MoE layer in every other respect).
///
/// The head consumes the main model's last hidden state `h` and the embedding of the
/// token just sampled: `eh_proj·[enorm(emb) ‖ hnorm(h)]` is its residual-stream input.
/// `shared_norm` replaces `model.norm` before the (shared) `lm_head`.
#[derive(Clone, Copy)]
pub struct MtpPin {
    pub eh_proj: *const f32,     // [hidden, 2·hidden] (bf16→f32; gemv_f32)
    pub enorm: *const f32,       // [hidden]
    pub hnorm: *const f32,       // [hidden]
    pub shared_norm: *const f32, // [hidden]
}

/// The resident weight set + cold-expert streaming pool.
pub struct Pin<'a> {
    cfg: &'a ModelConfig,
    /// The resident weight slab. Never read through after `build` — held purely as
    /// the RAII owner of the VMM allocation that every resident pointer points into.
    #[allow(dead_code)]
    tier: DeviceTier,
    pub embed: Int8Weight,
    pub lm_head: Int8Weight,
    pub final_norm: *const f32,
    pub layers: Vec<LayerPin>,
    /// The MTP head's extra weights, `Some` when the artifact carries it. Its ordinary
    /// layer weights live at `layers[cfg.n_layers]`, so `layers.len()` is `n_layers + 1`.
    pub mtp: Option<MtpPin>,
    /// Router correction bias per MoE layer, kept HOST-side (the sigmoid/bias/top-k
    /// routing runs on the CPU). `moe_bias[layer - dense_layers]`, len n_experts.
    moe_bias: Vec<Vec<f32>>,
    /// The three per-projection codebooks (gate/up/down), resident, passed to
    /// `launch_moe_expert_range`. Each `VQ_K·VQ_DIM` fp16 (narrowed from the f32
    /// source at load — the random idx→cb gather is the MoE hot path, and fp16
    /// halves it into L1; see math::f32_to_f16 / dot_vq_wave). Null in `--i4` (int4
    /// decodes without a codebook).
    codebooks: [*const u16; 3],
    /// The `.vq3` / `.i4` streaming sources — fd owners backing the routed pool's read
    /// tables; held for the run, not read through after `build`.
    #[allow(dead_code)]
    vq_src: Option<ExpertSet>,
    #[allow(dead_code)]
    i4_src: Option<ExpertSet>,
    /// The always-resident shared expert's format: int4 in `--i4`/hybrid, else vq3.
    /// gpu.rs launches the folded shared expert with the matching kernel.
    shared_i4: bool,
    /// The routed-expert pool: a two-ended byte [`Arena`] + a byte-aware policy. The
    /// COLD/HOT tiers are one format (single-format) or int3-VQ/int4 (hybrid).
    routed: ArenaPool,
    /// Per-expert async cold-fetch: owns the io_uring demand ring on a reaper thread
    /// and signals each miss's [`Ticket`] when its bytes land. The expert
    /// stream awaits these; there is no batch join.
    fetch: AsyncFetch,
    pub hits: u64,
    pub misses: u64,
    /// Optional access-trace sink (`--trace`), format v2: a `#` header line, then one
    /// line per resolved MoE layer — the `(layer,expert)` keys looked up in access
    /// order, then `|`, then the top-[`TRACE_WINDOW`] router candidates as
    /// `key:choice` in rank order. Feeds the offline `replay` simulator.
    trace: Option<std::io::BufWriter<std::fs::File>>,
}

/// Place an F32 tensor (norms, router gate) into the tier: reserve, copy from the
/// resident mmap into the device-local slab.
fn place_f32(tier: &mut DeviceTier, st: &Safetensors, name: &str) -> Result<*const f32> {
    let (bytes, _) = st.typed(name, Dtype::F32)?;
    // f32 LE host == LE device.
    Ok(tier.place(bytes)? as *const f32)
}

/// Place an fp8-e4m3 block-scaled weight (`<name>.weight` F8E4M3 + `.weight_scale_inv`
/// F32) into the tier. Dims come from the weight's `[o_dim, i_dim]` shape.
fn place_fp8(
    tier: &mut DeviceTier,
    st: &Safetensors,
    name: &str,
    block: usize,
) -> Result<Fp8Weight> {
    let (w, shape) = st.typed(&format!("{name}.weight"), Dtype::F8E4M3)?;
    ensure!(
        shape.len() == 2,
        "{name}.weight: expected 2-D, got {shape:?}"
    );
    let (o_dim, i_dim) = (shape[0], shape[1]);
    let (sc, _) = st.typed(&format!("{name}.weight_scale_inv"), Dtype::F32)?;
    let packed = tier.place(w)?;
    let scale = tier.place(sc)?;
    Ok(Fp8Weight {
        packed,
        scale: scale as *const f32,
        block,
        o_dim,
        i_dim,
    })
}

/// Place an int8 per-row weight (`<name>` I8 + `<name>.scale` F32) into the tier
/// (embed / lm_head). Dims from the `[o_dim, i_dim]` shape.
fn place_i8(tier: &mut DeviceTier, st: &Safetensors, name: &str) -> Result<Int8Weight> {
    let (w, shape) = st.typed(name, Dtype::I8)?;
    ensure!(shape.len() == 2, "{name}: expected 2-D, got {shape:?}");
    let (o_dim, i_dim) = (shape[0], shape[1]);
    let (sc, _) = st.typed(&format!("{name}.scale"), Dtype::F32)?;
    let packed = tier.place(w)?;
    let scale = tier.place(sc)?;
    Ok(Int8Weight {
        packed,
        scale: scale as *const f32,
        o_dim,
        i_dim,
    })
}

fn place_dense_mlp(
    tier: &mut DeviceTier,
    st: &Safetensors,
    base: &str,
    block: usize,
) -> Result<Fp8Mlp> {
    Ok(Fp8Mlp {
        gate: place_fp8(tier, st, &format!("{base}.gate_proj"), block)?,
        up: place_fp8(tier, st, &format!("{base}.up_proj"), block)?,
        down: place_fp8(tier, st, &format!("{base}.down_proj"), block)?,
    })
}

/// Place a shared-expert `block` (gate‖up‖down) resident and resolve its six
/// pointers. `off` = the format's slot offsets (VQ or int4), so the same code
/// places either. The shared expert is the routed format (`--i4` ⇒ int4).
fn place_shared(tier: &mut DeviceTier, block: &[u8], off: &[usize; 6]) -> Result<MlpVq> {
    let dst = tier.place(block)?;
    Ok(MlpVq {
        gate: vqweight_at(dst, off[0], off[1]),
        up: vqweight_at(dst, off[2], off[3]),
        down: vqweight_at(dst, off[4], off[5]),
    })
}

/// Place a full layer's DSA indexer weights: fp8 `wk`/`wq_b`/`weights_proj` + the
/// f32-widened `k_norm` weight/bias (the converter stores k_norm as F32).
fn place_indexer(
    tier: &mut DeviceTier,
    st: &Safetensors,
    layer: usize,
    block: usize,
) -> Result<IndexerPin> {
    let base = format!("model.layers.{layer}.self_attn.indexer");
    Ok(IndexerPin {
        wk: place_fp8(tier, st, &format!("{base}.wk"), block)?,
        wq_b: place_fp8(tier, st, &format!("{base}.wq_b"), block)?,
        weights_proj: place_f32(tier, st, &format!("{base}.weights_proj.weight"))?,
        k_norm_w: place_f32(tier, st, &format!("{base}.k_norm.weight"))?,
        k_norm_b: place_f32(tier, st, &format!("{base}.k_norm.bias"))?,
    })
}

/// Device bytes the always-resident set occupies — everything read every token EXCEPT the
/// routed experts, so the tier is sized to what it holds and the rest of the device budget
/// grows the routed pool.
///
/// **It is the artifact's own `*.safetensors` byte length, not a second derivation of the
/// placement layout.** `bin/convert` writes `resident.safetensors` from exactly the tensor
/// list [`Pin::build`] places, so the file IS the footprint. This replaced 73 lines
/// (`fp8_bytes`/`i8_bytes`/`f32_bytes`/`indexer_bytes` + a per-layer `cfg` walk) that
/// re-derived every shard size and were kept in step with the placement path by a comment
/// saying so — two copies of one layout, and the copy nothing executed was free to drift.
///
/// The bias is deliberate: over-count only shrinks the routed pool, under-count is what
/// `DeviceTier::place` bails on. It over-counts by each file's header, by the ~77 KB of
/// router correction bias (read to the HOST, never placed), and by one layer's attention
/// stack when the artifact carries MTP tensors this run cannot use (a format's slab absent).
///
/// Skipping the whole `indexer.safetensors` when `!want_indexer` IS the per-tensor
/// `.indexer` filter — `bin/add_indexer` writes nothing else into it. `cb_resident` and
/// `shared_i4` add the two resident things no safetensors holds: the 3 fp16 VQ codebooks,
/// and one shared expert per MoE layer out of the `.vq3`/`.i4` slab.
fn resident_bytes(
    dir: &str,
    cfg: &ModelConfig,
    n_pin: usize,
    shared_i4: bool,
    cb_resident: bool,
    want_indexer: bool,
) -> Result<usize> {
    let mut total = 0usize;
    for entry in std::fs::read_dir(dir).with_context(|| format!("read dir {dir}"))? {
        let p = entry.with_context(|| format!("read dir {dir}"))?.path();
        let is_st = p.extension().is_some_and(|x| x == "safetensors");
        let unplaced_indexer =
            !want_indexer && p.file_name().is_some_and(|n| n == "indexer.safetensors");
        if !is_st || unplaced_indexer {
            continue;
        }
        total += p.metadata().with_context(|| format!("stat {p:?}"))?.len() as usize;
    }
    let shared = if shared_i4 {
        i4_expert_bytes(cfg.hidden, cfg.moe_inter)
    } else {
        vq_expert_bytes(cfg.hidden, cfg.moe_inter)
    };
    Ok(total
        + (n_pin - cfg.dense_layers) * shared
        + if cb_resident {
            3 * VQ_K * VQ_DIM * 2
        } else {
            0
        })
}

/// Width of the trace-v2 candidate window: the top-W router candidates recorded per
/// routing decision, on top of the `top_k` that actually ran. W bounds the largest M
/// the offline (J, M) substitution grid in docs/investigations/cache-conditional-routing.md can explore — an M
/// wider than this cannot be evaluated from a captured trace without recapturing.
/// 32 is 4× `top_k` (8) and an eighth of `n_experts` (256): far past any M where
/// promoting a resident-but-lower-ranked expert is still defensible, and only ~380
/// bytes a line.
pub const TRACE_WINDOW: usize = 32;

/// Pack `(layer, expert)` into the pool key. Both must fit in 16 bits — GLM is
/// ≤92 layers × 256 routed experts, comfortably under 2^16.
pub fn expert_key(layer: usize, expert: usize) -> u32 {
    debug_assert!(
        layer < (1 << 16) && expert < (1 << 16),
        "layer {layer}/expert {expert} exceed the 16-bit pool key packing"
    );
    ((layer as u32) << 16) | expert as u32
}

/// One arena tier's format: the projection offsets, the format flag, the slot stride,
/// and the per-`(layer,expert)` O_DIRECT read-spec table. COLD/HOT are the SAME format
/// in single-format modes (uniform stride; any compaction is a cheap same-size move)
/// or int3-VQ vs int4 (hybrid).
#[derive(Clone)]
struct TierFmt {
    off: [usize; 6],
    int4: bool,
    stride: usize,
    table: Vec<(RawFd, usize, usize)>, // (fd, begin, len) per (layer-dense)*n_experts+expert
}

/// The routed-expert pool over the two-ended byte [`Arena`]. A byte-aware
/// [`HybridPolicy`] owns residency and the (floating) COLD/HOT split; on a cross-tier
/// rebalance the arena emits a relocation, which we execute as a synchronous device
/// memcpy of the expert's bytes and remap its key. `slot_of`/`key_at` are inverse maps.
struct ArenaPool {
    #[allow(dead_code)] // RAII owner of the pool VMM; addressed via `base`/`host_base`
    buf: VmmBuf,
    /// The DEVICE base: what every expert descriptor's six projection pointers are built
    /// from, and never dereferenced on the CPU.
    base: *mut u8,
    /// The HOST base: the io_uring O_DIRECT DMA target (`ReadSpec.dst`), and the only one
    /// of the two the CPU may touch.
    ///
    /// Under HIP these are the SAME NUMBER — unified addressing — so this field costs
    /// nothing there and changes no behaviour. Under Vulkan they are unrelated, and
    /// resolving both once here is what keeps [`ArenaPool::ptr`] and
    /// [`ArenaPool::host_ptr`] a single `add` each on the fetch path. See
    /// docs/investigations/vulkan-port.md, "Host pointer != device address".
    host_base: *mut u8,
    arena: Arena,
    policy: Box<dyn HybridPolicy>,
    slot_of: HashMap<u32, (bool, usize)>, // key -> (hot, idx)
    key_at: HashMap<(bool, usize), u32>,  // (hot, idx) -> key, for relocation remap
    /// Keys whose bytes are known to have LANDED in their current slot.
    ///
    /// The engine has no other way to distinguish "the policy says resident" from "the
    /// bytes are actually there", and that distinction is the leading hypothesis for the
    /// intermittent non-finite-logits bug: a HIT carries `Ticket::RESIDENT` and the kernel
    /// reads the slot immediately, so if a key is ever counted resident before its load
    /// completed, the read is of uninitialised (-> NaN, the visible case) or stale (->
    /// finite and WRONG, the silent case) memory.
    ///
    /// A key is removed on eviction and on relocation-into, and inserted only when its
    /// read signal has resolved.
    ///
    /// **Was `trace`-only until 2026-08-03; it is now always compiled, and that change is
    /// the point.** The fault it detects is silent by construction — a hit whose bytes never
    /// landed reads the previous expert's weights, which are finite and plausible — and the
    /// only build that could see it was the one whose poison fill adds a `device_sync` that
    /// masks it. So the detector existed and could not fire. Long runs were measured
    /// non-deterministic (~40% of 5k-token scores wrong, benchmarks.md 2026-08-02) with this
    /// check compiled out.
    ///
    /// It costs a hash op per expert per layer: ~600 per token against a ~400 ms token, i.e.
    /// ~0.003%, and adds NO device work and NO ordering — which is what lets it hunt a race
    /// rather than hide one. The poison fill and its `device_sync` stay behind `trace`.
    loaded: std::collections::HashSet<u32>,
    /// Misses submitted by the PREVIOUS layer, marked loaded at the top of the next
    /// `submit_layer`. Correct because layer L's per-expert awaits and its unconditional
    /// end-of-layer `device_sync` both complete before layer L+1 submits — so by then
    /// every byte of L's batch has landed. Deferring this way avoids plumbing a
    /// completion callback through `gpu.rs`'s async expert loop.
    pending_loaded: Vec<u32>,
    cold: TierFmt,
    hot: TierFmt,
}

impl ArenaPool {
    fn tier(&self, hot: bool) -> &TierFmt {
        if hot { &self.hot } else { &self.cold }
    }
    /// The slot's DEVICE address — the base every expert descriptor's six projection
    /// pointers are built from, and what `memcpy_dtod`/`fill_u32` take.
    ///
    /// NOT host-dereferenceable under Vulkan. It happens to be under HIP, where unified
    /// addressing makes this and [`ArenaPool::host_ptr`] the same number; relying on that
    /// is what the split exists to prevent.
    fn ptr(&self, hot: bool, idx: usize) -> *mut u8 {
        // SAFETY: arena.offset < budget, within the pool VMM.
        unsafe { self.base.add(self.arena.offset(hot, idx)) }
    }

    /// The slot's HOST address — the io_uring O_DIRECT destination (`ReadSpec.dst`).
    ///
    /// Same offset arithmetic as [`ArenaPool::ptr`], different base. The arena's slot
    /// strides and the pool base are both `crate::fetch::stream::ALIGN`-aligned (checked in
    /// `VmmBuf::new` and by the budget rounding in `Pin::build`), so every result satisfies
    /// the O_DIRECT alignment the streamer asserts.
    fn host_ptr(&self, hot: bool, idx: usize) -> *mut u8 {
        // SAFETY: arena.offset < budget, within the pool VMM's host mapping.
        unsafe { self.host_base.add(self.arena.offset(hot, idx)) }
    }
    /// Record that `key`'s bytes have landed in its current slot. Called once per MISS,
    /// after that read's signal resolves.
    fn mark_loaded(&mut self, key: u32) {
        self.loaded.insert(key);
    }

    /// Has `key`'s data actually landed since it was last admitted? A HIT on a key for
    /// which this is false is a read of uninitialised or stale bytes — the fault this
    /// check exists to catch, and the leading explanation for the measured
    /// non-determinism of long runs.
    fn is_loaded(&self, key: u32) -> bool {
        self.loaded.contains(&key)
    }

    fn slot(&self, key: u32) -> Option<(bool, usize)> {
        self.slot_of.get(&key).copied()
    }

    /// Admit a MISS: the policy evicts (by its own rule) until the incoming slot's bytes
    /// fit; free each victim's slot, then place the new key — compacting the arena (one
    /// device memcpy per relocation) as needed. Records the key's final slot.
    fn alloc(&mut self, key: u32) -> Result<()> {
        let adm = self.policy.admit(key);
        for ev in adm.evicted {
            let s = self
                .slot_of
                .remove(&ev)
                .context("evicted key had no slot")?;
            self.key_at.remove(&s);
            // Evicted: its bytes are no longer this key's, and the slot is about to be
            // handed to someone else.
            self.loaded.remove(&ev);
            self.arena.free(s.0, s.1);
        }
        let hot = adm.tier == cache::Tier::Hot;
        let idx = loop {
            match self.arena.alloc_step(hot) {
                Step::Placed(idx) => break idx,
                Step::Relocated(r) => self.relocate(r)?,
                Step::NeedFree => {
                    bail!("arena NeedFree after policy eviction — byte-accounting bug")
                }
            }
        };
        self.slot_of.insert(key, (hot, idx));
        self.key_at.insert((hot, idx), key);
        // Freshly admitted: a slot with no bytes in it yet. `mark_loaded` clears this
        // once the read lands. A HIT observed while a key is in this state is the bug.
        self.loaded.remove(&key);
        // POISON the slot before its bytes land, so a read-before-write is deterministic.
        //
        // Without this an unloaded slot holds whatever was there: uninitialised memory
        // (-> NaN, seen in ~6% of long runs) or the evicted expert's weights (-> finite,
        // plausible, SILENTLY wrong). 0x7FC0_7FC0 is a quiet NaN in f32 and in both bf16
        // halves, so every format's scales read back non-finite and both cases collapse
        // into the loud one — which the per-layer localiser then pins to a (pos, layer).
        //
        // Costs a ~20 MB device fill per miss (~3% of wall at 148 misses/token), which is
        // why it is `trace`-only. It is a diagnostic, not a safety net.
        #[cfg(feature = "trace")]
        {
            let stride = self.tier(hot).stride;
            let dst = self.ptr(hot, idx);
            // SAFETY: `dst` owns `stride` bytes in the pool VMM; the slot is not yet
            // handed to any kernel (that happens in phase 1c, after this returns).
            unsafe { crate::backend::fill_u32(dst, 0x7FC0_7FC0, stride)? };
        }
        Ok(())
    }

    /// Execute one compaction relocation: memcpy the slot's bytes `from`→`to` (distinct,
    /// non-overlapping slots) and remap the key that lived there. Synchronous, so it
    /// lands before the layer's compute or any later cold read touches the new slot.
    fn relocate(&mut self, r: Reloc) -> Result<()> {
        let moved = self
            .key_at
            .remove(&(r.hot, r.from))
            .context("relocated slot had no key")?;
        let stride = self.tier(r.hot).stride;
        let src = self.ptr(r.hot, r.from) as *const u8;
        let dst = self.ptr(r.hot, r.to);
        // SAFETY: distinct slots (non-overlapping), each `stride` bytes within the VMM.
        unsafe { memcpy_dtod(dst, src, stride)? };
        self.slot_of.insert(moved, (r.hot, r.to));
        self.key_at.insert((r.hot, r.to), moved);
        // The relocation copies the bytes with the key, so `moved` stays loaded. Nothing
        // else changes state: the source slot is now free and holds no key.
        Ok(())
    }
}

impl<'a> Pin<'a> {
    /// Enqueue the device-side wait for `t` on `stream_raw`. The ONLY way to consume a
    /// ticket — so a launch cannot happen without its dependency.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn wait_on(&self, t: Ticket, stream_raw: *mut std::ffi::c_void) -> Result<()> {
        self.fetch.wait(t, stream_raw)
    }

    /// Build the resident set from the artifact directory `dir`. `capacity` is the
    /// total device budget (auto-discovered); the always-resident set takes its
    /// computed footprint and the rest grows the routed pool.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        dir: &str,
        cfg: &'a ModelConfig,
        capacity: usize,
        trace_path: Option<&str>,
        cache_policy: &str,
        two_q: cache::TwoQSplit,
        want_indexer: bool,
        mode: Mode,
    ) -> Result<Self> {
        // `i4` = the int4 placement path is needed (int4 mode, or the hybrid HOT tier +
        // shared expert). int3-VQ uses neither. See docs/reference/modes.md.
        let i4 = mode.uses_int4();
        // ponytail: no free-memory pre-check — the budget is the user's literal
        // request (--max-mem), so let the device allocation itself OOM/fail.
        // One-time bound for `submit_layer`'s fixed 32-slot scratch. A batched forward
        // submits the UNION of every token row's picks, so the worst case is
        // `top_k · MAXROW + n_shared`, not one row's `experts_per_layer()`.
        let max_batch = cfg.top_k * crate::gpu::MAXROW + cfg.n_shared;
        ensure!(
            max_batch <= 32,
            "top_k {} x {} rows + n_shared {} = {max_batch} exceeds the 32-slot batch scratch",
            cfg.top_k,
            crate::gpu::MAXROW,
            cfg.n_shared
        );

        // Open the artifact: format meta (fp8 block), resident mmap, codebooks, and
        // the `.vq3` streaming source.
        let fmt = FormatMeta::load(dir)?;
        let block = fmt.fp8_block;
        // open_dir merges every *.safetensors in the artifact: resident.safetensors
        // plus, when present, indexer.safetensors (the DSA weights, added post-hoc from
        // the fp8 stash — see bin/add_indexer). The .vq3/codebooks files are ignored.
        let st = Safetensors::open_dir(dir)?;
        let cbs = load_codebooks(dir)?;
        // Routed-expert sources. Single-format opens one; the hybrid opens BOTH (cold
        // vq3 + hot int4) and the fixed-partition policy routes each key to its slab.
        // Both file sets sit in the artifact side by side. `shared_i4`: the
        // always-resident shared expert rides the primary/hot format (int4 in hybrid).
        let vq_present = mode != Mode::Int4; // int3-vq or hybrid needs a vq slab
        // The MTP head is checkpoint layer `n_layers` and is a full MoE layer, so it
        // rides the routed pool like any other and needs its expert slab in EVERY format
        // this run opens. An artifact converted before 2026-07-31 has no `L78.i4` — that
        // is a missing slab, not a missing feature, so an int4/hybrid run on one decodes
        // without a draft head rather than failing to load. Re-run `bin/fp8_to_i4` to
        // emit it; it covers layer `n_layers` since the range bound was widened.
        let mtp = st.has(&format!("model.layers.{}.eh_proj.weight", cfg.n_layers))
            && [(vq_present, "vq3"), (i4, "i4")]
                .iter()
                .all(|&(want, ext)| {
                    !want || std::fs::metadata(format!("{dir}/L{:02}.{ext}", cfg.n_layers)).is_ok()
                });
        // Layers the PIN holds: the model's, plus the MTP head one past the end.
        let n_pin = cfg.n_layers + usize::from(mtp);
        tracing::info!("mtp head: {}", if mtp { "present" } else { "absent" });
        // Both formats open against the SAME dims, named once — the two cannot end up
        // opened against different ones (which every length check would happily pass).
        let dims = SetDims {
            dense_layers: cfg.dense_layers,
            n_layers: n_pin,
            n_experts: cfg.n_experts,
            hidden: cfg.hidden,
            moe_inter: cfg.moe_inter,
        };
        let vq_src = vq_present
            .then(|| ExpertSet::open_routed(dir, false, dims))
            .transpose()?;
        let i4_src = i4
            .then(|| ExpertSet::open_routed(dir, true, dims))
            .transpose()?;
        let shared_i4 = i4;
        // Slot byte layouts, one per format (which projection's indices/scales sit
        // where in an expert block). Shared expert + routed slabs reuse these.
        let vq_off = vq_slot_offsets(cfg.hidden, cfg.moe_inter);
        let i4_off = i4_slot_offsets(cfg.hidden, cfg.moe_inter);
        let shared_off = if shared_i4 { i4_off } else { vq_off };

        // Full layers own a resident DSA indexer (dsa/misa modes). Empty when
        // dense/streaming — the mask drives both placement and the footprint.
        // `index_share_for_mtp_iteration` means the head reuses the main model's
        // selection, so it never owns an indexer — push a `false` for it either way.
        let full = if want_indexer {
            let mut f = cfg.indexer_layout()?;
            f.resize(n_pin, false);
            f
        } else {
            vec![false; n_pin]
        };

        // Size the tier to the always-resident footprint plus slack (absorbs the
        // per-reservation 256-byte alignment padding); anything left widens the pool.
        const SLACK: usize = 256 << 20; // 256 MiB
        let resident = resident_bytes(dir, cfg, n_pin, shared_i4, vq_present, want_indexer)?;
        let tier_cap = resident + SLACK;
        tracing::info!(
            "computed resident footprint {:.2} GiB (tier {:.2} GiB incl. slack)",
            resident as f64 / (1u64 << 30) as f64,
            tier_cap as f64 / (1u64 << 30) as f64,
        );
        let mut tier = DeviceTier::new(tier_cap)?;

        // Codebooks resident (gate/up/down), narrowed f32 → fp16 at load and passed
        // to launch_moe_expert_range. fp16 halves the hot idx→cb gather into L1.
        // Uploaded whenever a VQ slab is present (vq-only or the hybrid cold slab);
        // int4-only decodes without a codebook.
        let mut codebooks = [std::ptr::null(); 3];
        if vq_present {
            for (i, cb) in cbs.iter().enumerate() {
                let half: Vec<u8> = cb
                    .iter()
                    .flat_map(|&v| crate::math::f32_to_f16(v).to_le_bytes())
                    .collect();
                codebooks[i] = tier.place(&half)? as *const u16;
            }
        }

        // Global tensors.
        let embed = place_i8(&mut tier, &st, "model.embed_tokens.weight")?;
        let lm_head = place_i8(&mut tier, &st, "lm_head.weight")?;
        let final_norm = place_f32(&mut tier, &st, "model.norm.weight")?;

        // Per-layer always-resident weights. `l` indexes both the weight-name
        // format!s and the `full` indexer mask; iterating `full` would lose it.
        let mut layers = Vec::with_capacity(n_pin);
        let mut moe_bias = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for l in 0..n_pin {
            let lb = format!("model.layers.{l}");
            let a = format!("{lb}.self_attn");
            let mlp = if l < cfg.dense_layers {
                LayerMlp::Dense(place_dense_mlp(
                    &mut tier,
                    &st,
                    &format!("{lb}.mlp"),
                    block,
                )?)
            } else {
                let gate_w = place_f32(&mut tier, &st, &format!("{lb}.mlp.gate.weight"))?;
                let (bias, _) = st.typed(
                    &format!("{lb}.mlp.gate.e_score_correction_bias"),
                    Dtype::F32,
                )?;
                let bias = crate::artifact::quant::read_f32(bias);
                ensure!(
                    bias.len() == cfg.n_experts,
                    "layer {l} gate bias has {} entries, expected {}",
                    bias.len(),
                    cfg.n_experts
                );
                moe_bias.push(bias);
                let shared_block = if shared_i4 {
                    i4_src
                        .as_ref()
                        .context("i4 shared: source missing")?
                        .shared_block(l)?
                } else {
                    vq_src
                        .as_ref()
                        .context("vq shared: source missing")?
                        .shared_block(l)?
                };
                let shared = place_shared(&mut tier, &shared_block, &shared_off)?;
                LayerMlp::Moe { gate_w, shared }
            };
            layers.push(LayerPin {
                input_ln: place_f32(&mut tier, &st, &format!("{lb}.input_layernorm.weight"))?,
                post_ln: place_f32(
                    &mut tier,
                    &st,
                    &format!("{lb}.post_attention_layernorm.weight"),
                )?,
                q_a: place_fp8(&mut tier, &st, &format!("{a}.q_a_proj"), block)?,
                q_a_ln: place_f32(&mut tier, &st, &format!("{a}.q_a_layernorm.weight"))?,
                q_b: place_fp8(&mut tier, &st, &format!("{a}.q_b_proj"), block)?,
                kv_a: place_fp8(&mut tier, &st, &format!("{a}.kv_a_proj_with_mqa"), block)?,
                kv_a_ln: place_f32(&mut tier, &st, &format!("{a}.kv_a_layernorm.weight"))?,
                kv_b: place_fp8(&mut tier, &st, &format!("{a}.kv_b_proj"), block)?,
                o_proj: place_fp8(&mut tier, &st, &format!("{a}.o_proj"), block)?,
                mlp,
                indexer: if full[l] {
                    Some(place_indexer(&mut tier, &st, l, block)?)
                } else {
                    None
                },
            });
        }
        let mtp = mtp
            .then(|| -> Result<MtpPin> {
                let lb = format!("model.layers.{}", cfg.n_layers);
                Ok(MtpPin {
                    eh_proj: place_f32(&mut tier, &st, &format!("{lb}.eh_proj.weight"))?,
                    enorm: place_f32(&mut tier, &st, &format!("{lb}.enorm.weight"))?,
                    hnorm: place_f32(&mut tier, &st, &format!("{lb}.hnorm.weight"))?,
                    shared_norm: place_f32(
                        &mut tier,
                        &st,
                        &format!("{lb}.shared_head.norm.weight"),
                    )?,
                })
            })
            .transpose()?;

        // Routed pool: a two-ended byte Arena over the budget left after the resident
        // set. Each expert is ONE aligned block read into a slot. COLD/HOT tiers are one
        // format (single-format: uniform stride, so a compaction relocation is always a
        // single cheap same-size move) or int3-VQ/int4 (hybrid). The byte-aware policy
        // floats the split; a cross-tier rebalance relocates a slot.
        // Round the pool budget DOWN to the O_DIRECT block so the arena's HIGH-end
        // anchor is aligned: HOT slots sit at `budget − (idx+1)*hot_stride`, so an
        // unaligned `budget` makes every hot-slot dst violate the O_DIRECT alignment
        // the streamer asserts (base + strides are already 4096-aligned). Costs <4 KiB.
        let budget = capacity.saturating_sub(tier_cap) & !(crate::fetch::stream::ALIGN - 1);
        // A tier descriptor per format. The two differ only in which source, which slot
        // layout and which kernel flag — the read table is built the same way from either,
        // so it is built in one place.
        let tier_fmt = |int4: bool| -> Result<TierFmt> {
            let (src, off, what) = if int4 {
                (&i4_src, i4_off, "i4")
            } else {
                (&vq_src, vq_off, "vq")
            };
            let s = src
                .as_ref()
                .with_context(|| format!("{what} source missing"))?;
            Ok(TierFmt {
                off,
                int4,
                stride: s.expert_slot(),
                table: build_moe_table(cfg, n_pin, |l, e| s.read_spec(l, e))?,
            })
        };
        // COLD/HOT tiers by mode. Single-format shares one format across both tiers.
        let (cold, hot) = match mode {
            Mode::Int3Vq => {
                let t = tier_fmt(false)?;
                (t.clone(), t)
            }
            Mode::Int4 => {
                let t = tier_fmt(true)?;
                (t.clone(), t)
            }
            Mode::Hybrid => (tier_fmt(false)?, tier_fmt(true)?),
        };
        let (cold_stride, hot_stride) = (cold.stride, hot.stride);
        let policy =
            crate::memory::hybrid::make(cache_policy, budget, cold_stride, hot_stride, two_q)
                .with_context(|| format!("unknown --cache-policy {cache_policy} (lru|2q|arc)"))?;
        tracing::info!(
            "routed pool [{cache_policy} {mode}]: {:.1} GiB budget (~{} slots, cold {cold_stride}B / hot {hot_stride}B)",
            budget as f64 / (1u64 << 30) as f64,
            budget / cold_stride.min(hot_stride),
        );
        let mut buf = VmmBuf::new(budget)?;
        let base = buf.ptr_mut();
        // Both bases resolved ONCE, here. Under HIP `host_mut` and `ptr_mut` return the
        // same number and this is a no-op; under Vulkan they are the device address and the
        // permanent host mapping, and the two consumers below must not be able to confuse
        // them. Taking them in this order matters only in that `ptr_mut`/`host_mut` both
        // need `&mut buf` and `buf` is moved into the struct after.
        let host_base = buf.host_mut();
        let routed = ArenaPool {
            buf,
            base,
            host_base,
            arena: Arena::new(budget, cold_stride, hot_stride),
            policy,
            slot_of: HashMap::new(),
            key_at: HashMap::new(),
            loaded: std::collections::HashSet::new(),
            pending_loaded: Vec::new(),
            cold,
            hot,
        };
        // Ring sized for one layer's worst case: top_k demand reads (1/expert). One
        // read per expert — one aligned `.vq3`/`.i4` block, either format.
        let ring = (cfg.top_k + 4).next_power_of_two();
        ensure!(
            cfg.top_k <= ring,
            "io_uring ring {ring} too small for top_k {}",
            cfg.top_k,
        );
        // Bounce span = the largest expert block across the tiers (one read).
        let span = slot_span(cold_stride.max(hot_stride));
        let fetch = AsyncFetch::new(Streamer::new(ring as u32, span)?)?;

        Ok(Self {
            cfg,
            tier,
            embed,
            lm_head,
            final_norm,
            layers,
            mtp,
            moe_bias,
            codebooks,
            vq_src,
            i4_src,
            shared_i4,
            routed,
            fetch,
            hits: 0,
            misses: 0,
            trace: trace_path
                .map(|p| -> Result<_> {
                    use std::io::Write;
                    let mut w = std::io::BufWriter::new(
                        std::fs::File::create(p).with_context(|| format!("open trace {p}"))?,
                    );
                    // Version header. It is deliberately unparseable as data: `replay`
                    // reads each line for whitespace-separated u32s and drops the empty
                    // ones, so this line contributes nothing and a v2 trace replays
                    // through a v1 reader byte-identically.
                    let top_k = cfg.top_k;
                    writeln!(w, "# rivoli-trace v2 top_k={top_k} window={TRACE_WINDOW}")
                        .context("write trace")?;
                    Ok(w)
                })
                .transpose()?,
        })
    }

    /// Host router correction bias for a MoE `layer` (len n_experts).
    pub fn moe_bias(&self, layer: usize) -> &[f32] {
        &self.moe_bias[layer - self.cfg.dense_layers]
    }

    /// The three per-projection codebooks (gate/up/down), fp16, for `launch_moe_expert_range`.
    /// Null pointers in int4 mode (int4 decodes without a codebook).
    pub fn codebooks(&self) -> [*const u16; 3] {
        self.codebooks
    }

    /// The always-resident shared expert's format: int4 (`--i4`/hybrid) vs int3-VQ.
    /// gpu.rs appends the folded shared expert with this format flag; routed experts
    /// carry their own per-expert flag from [`submit_layer`].
    pub fn shared_i4(&self) -> bool {
        self.shared_i4
    }

    /// Is the `--trace` sink on? gpu.rs gates the candidate-window `topk_into` on this
    /// so a non-tracing decode pays literally nothing for trace v2.
    pub fn tracing(&self) -> bool {
        self.trace.is_some()
    }

    /// Is `(layer, expert)` resident? Deliberately routed through
    /// [`HybridPolicy::contains`], which takes `&self` and does NOT refresh recency —
    /// `get` would count the whole candidate window as an access and corrupt the
    /// eviction clock, which is the failure mode that would make `top-m` look like it
    /// works while destroying the cache underneath it.
    pub fn resident(&self, layer: usize, expert: usize) -> bool {
        self.routed.policy.contains(expert_key(layer, expert))
    }

    /// Flush the trace sink. Called per token, because the trace CANNOT rely on
    /// `BufWriter`'s `Drop`: the wedge watchdog kills a hung decode with
    /// `std::process::exit`, which runs no destructors, and `Drop` discards flush errors
    /// anyway — so a wedged or ENOSPC run would leave a silently short capture with a
    /// clean exit code. A trace is ~30 minutes of sole-tenant GPU time; losing it quietly
    /// is far worse than one `write` per token. Errors propagate here, unlike in `Drop`.
    pub fn flush_trace(&mut self) -> Result<()> {
        if let Some(w) = &mut self.trace {
            use std::io::Write;
            w.flush().context("flush trace")?;
        }
        Ok(())
    }

    /// Accumulated reaper fetch wall (ns) — the off-main-thread load cost the expert
    /// stream's compute overlaps. The profile reads it against the MoE wall.
    pub fn fetch_ns(&self) -> u64 {
        self.fetch.fetch_ns()
    }

    /// Accumulated ns the reaper spent blocked in `io_uring` completions — the measured
    /// io-wait, taken at the ring rather than inferred from phase subtraction.
    pub fn io_wait_ns(&self) -> u64 {
        self.fetch.io_wait_ns()
    }

    /// Times a layer had to WAIT for a staging slot whose bounce copy had not retired.
    /// Should stay 0: a layer uses ~2 of 16 slots and a copy retires in ~1.2 ms against a
    /// ~3.5 ms layer. Non-zero means the ring is undersized for the lookahead — surfaced
    /// rather than merely counted, because a counter nobody reads is how the last two dead
    /// fields in this engine got there.
    pub fn slot_stalls(&self) -> u64 {
        self.fetch.slot_stalls()
    }

    /// Submit one layer's cold reads and resolve each selected expert to its [`MlpVq`]
    /// (device pointers into the pool), its format flag, and its [`Ticket`] — the
    /// DEVICE-SIDE dependency its data is behind.
    ///
    /// Trace sink, then three phases over the arena pool. 1a: touch every HIT (protect it
    /// so a same-batch miss can't evict it). 1b: allocate every MISS — this is where the
    /// byte-aware policy evicts and the arena may RELOCATE resident slots. 1c: only NOW,
    /// after all relocations have settled, resolve each key's final slot into `out`/`fmt`
    /// and build the misses' cold reads — so a read never targets a slot that later moves.
    ///
    /// **There is no residency mask, and its absence is the point.** This used to also
    /// return `hit: Vec<bool>`, a second host-side encoding of "is this expert's data
    /// ready?" that `gpu.rs` consumed to decide whether to await. When the two disagreed the
    /// bool won silently — `gpu.rs` launches a `hit` expert with no wait at all — so a slot
    /// still being written could be marked ready and the kernel would read it. A ticket
    /// cannot disagree with anything: it IS the dependency, and the only way to launch is to
    /// enqueue its wait (`AsyncFetch::wait`). Resident experts carry [`Ticket::RESIDENT`],
    /// so resident / missing / in-flight are one code path.
    ///
    /// `window`/`choice` feed the trace sink only: the ranked top-[`TRACE_WINDOW`]
    /// candidate expert ids and the full per-expert `choice` array they index into.
    /// Pass an empty `window` when not tracing — nothing else reads them.
    // ONE function, not the `submit_spine` + unwrap pair this was: the split's only artefact
    // was a `[Option<ResolvedSlot>; 32]` filled unconditionally over `sel` and then unwrapped
    // with a second `.context("unresolved expert slot")` that could never fire.
    // Eight arguments, all distinct runtime values on the per-layer hot path; bundling
    // them into a struct built once per layer would allocate to satisfy a lint.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_layer(
        &mut self,
        layer: usize,
        sel: &[usize],
        window: &[usize],
        choice: &[f32],
        out: &mut Vec<MlpVq>,
        fmt: &mut Vec<bool>,
        tickets: &mut Vec<Ticket>,
    ) -> Result<()> {
        out.clear();
        fmt.clear();
        tickets.clear();
        let cfg = self.cfg;
        let sparse = (layer - cfg.dense_layers) * cfg.n_experts; // read-table row base
        debug_assert!(
            sel.len() <= 32,
            "submit_layer: {} experts exceeds the 32-slot scratch",
            sel.len()
        );
        // New batch: clear the policy's per-batch pin set. Phase 1a's protect() and 1b's
        // admit() then pin every touched key so a later miss's eviction can't reclaim it.
        // The previous layer's reads have all landed (its awaits + end-of-layer sync).
        {
            let done = std::mem::take(&mut self.routed.pending_loaded);
            for k in done {
                self.routed.mark_loaded(k);
            }
        }
        self.routed.policy.begin_batch();
        // Trace sink (--trace), v2: the demand keys this layer looks up, then `|`, then
        // the top-`TRACE_WINDOW` candidates as `key:choice`.
        //
        // BOTH lists are in router RANK order, and that is LOAD-BEARING, not incidental.
        // `sel` and `window` both come out of `topk_into` over the same `choice` buffer
        // with the same comparator (value-desc, index-asc), and `topk_into` finishes with
        // a full sort — so `window[..sel.len()] == sel` element for element, and
        // `bin/replay` hard-fails a trace where that prefix does not hold. Reordering
        // `sel` for any local reason (coalescing reads by expert id, say) would silently
        // change the meaning of every captured trace. The debug_assert is the tripwire.
        debug_assert!(
            window.is_empty() || window.starts_with(sel),
            "trace v2: the candidate window must be the ranking that produced `sel`"
        );
        if let Some(w) = &mut self.trace {
            use std::io::Write;
            for (j, &e) in sel.iter().enumerate() {
                if j > 0 {
                    write!(w, " ").context("write trace")?;
                }
                write!(w, "{}", expert_key(layer, e)).context("write trace")?;
            }
            // ponytail: the `choice` values have no consumer yet — the (J, M) grid needs
            // only the RANK order, which the list already carries. Written anyway because
            // a capture is GPU-gated, sole-tenant and ~30 minutes, so these few bytes are
            // cheap now and unrecoverable later without another capture; and `route_kl`
            // (docs/investigations/cache-conditional-routing.md "Counters") is deferred, not cancelled, and needs the
            // mass distribution.
            write!(w, " |").context("write trace")?;
            for &e in window {
                write!(w, " {}:{:.6}", expert_key(layer, e), choice[e]).context("write trace")?;
            }
            writeln!(w).context("write trace")?;
        }
        // Phase 1a: touch every hit first, so a later miss's admit can't evict it.
        let mut is_hit = [false; 32];
        for (i, &e) in sel.iter().enumerate() {
            let key = expert_key(layer, e);
            // `get` refreshes recency; the physical slot is deliberately NOT read here —
            // phase 1c takes it from `slot_of` after any same-batch relocation settles.
            if self.routed.policy.get(key) {
                self.hits += 1;
                // THE CHECK. A hit hands the kernel a slot pointer and a resolved
                // RESIDENT ticket, so nothing downstream waits. If the bytes never landed, the
                // kernel reads uninitialised memory (NaN) or another expert's weights
                // (finite, wrong, silent). Reported once per occurrence rather than
                // fataling, so a run keeps going and the pattern is visible.
                if !self.routed.is_loaded(key) {
                    tracing::error!(
                        "READ-BEFORE-WRITE: layer={layer} expert={e} counted as a cache \
                         HIT but its bytes never landed since admission. The kernel is \
                         about to read an unloaded slot — uninitialised memory (-> NaN) \
                         or a previous expert's weights (-> silently wrong)."
                    );
                }
                self.routed.policy.protect(key);
                is_hit[i] = true;
            }
        }
        // Phase 1b: allocate the misses (evict + place + compact). Slots may relocate.
        #[cfg(feature = "trace")]
        let mut poisoned_any = false;
        for (i, &e) in sel.iter().enumerate() {
            if !is_hit[i] {
                self.misses += 1;
                self.routed.alloc(expert_key(layer, e))?;
                #[cfg(feature = "trace")]
                {
                    poisoned_any = true;
                }
            }
        }
        // The poison fills above run on the DEFAULT stream; the reaper's bounce->slot
        // copies run on a `hipStreamNonBlocking` fetch stream, which does not synchronise
        // with it. Unordered, a fill could land AFTER the read and destroy good data —
        // the diagnostic would then cause the corruption it exists to detect. One join
        // per layer-with-misses orders them.
        //
        // Under Vulkan the hazard is the same one wearing different clothes, and the fix
        // covers it for a DIFFERENT reason: the fill is a `vkCmdFillBuffer` recorded into
        // the open command buffer, while the reaper's staging copy is a synchronous host
        // memcpy on another thread (see stream.rs's `stage`). Nothing orders a recorded-
        // but-unsubmitted fill against a host write, so the join is what submits and
        // retires it before any read is queued. It happens to be load-bearing on both
        // backends; do not delete it as HIP-specific.
        //
        // CAVEAT, and it is the same trap `--checksum-x` fell into: this sync may itself
        // perturb the race being hunted. It sits at a different point (after allocation,
        // before reads are submitted) than the per-layer D2H that masked the fault, so it
        // is not the same barrier — but a clean run under poisoning is NOT proof the bug
        // is absent. Only a poison HIT is positive evidence.
        #[cfg(feature = "trace")]
        if poisoned_any {
            crate::backend::device_sync()?;
        }
        // Phase 1c: relocations have settled — resolve final slots and build the reads.
        let mut reads: Vec<ReadSpec> = Vec::new();
        let mut miss_sel: Vec<usize> = Vec::new();
        for (i, &e) in sel.iter().enumerate() {
            let (hot, idx) = self.routed.slot(expert_key(layer, e)).context(
                "expert not resident after alloc (batch exceeds pool — raise --max-mem)",
            )?;
            let (b, t) = (self.routed.ptr(hot, idx), self.routed.tier(hot));
            // SAFETY: address arithmetic into the resolved slot; the bytes land when
            // `tickets[i]` is satisfied. The six pointers are identical for both formats
            // (gpu.rs reinterprets as ExpertDescI4 when `fmt[i]`).
            out.push(MlpVq {
                gate: vqweight_at(b, t.off[0], t.off[1]),
                up: vqweight_at(b, t.off[2], t.off[3]),
                down: vqweight_at(b, t.off[4], t.off[5]),
            });
            fmt.push(t.int4);
            if !is_hit[i] {
                let (fd, begin, len) = t.table[sparse + e];
                reads.push(ReadSpec {
                    fd,
                    begin,
                    len,
                    dst: self.routed.host_ptr(hot, idx),
                });
                miss_sel.push(i);
            }
        }
        // Phase 2: hand the whole batch to the reaper — it queues+submits (all reads
        // start at once) and signals each miss's ticket when its copy lands.
        // Queue this batch's misses to be marked loaded at the next layer.
        for &i in &miss_sel {
            self.routed.pending_loaded.push(expert_key(layer, sel[i]));
        }
        let miss_tickets = self.fetch.submit(reads)?;
        // A resident expert's data is already there, so it carries the RESIDENT ticket
        // (value 0, satisfied on arrival). Every expert therefore has a ticket and the
        // caller has one code path — there is no residency bool for anyone to branch on.
        tickets.resize(sel.len(), Ticket::RESIDENT);
        for (k, &i) in miss_sel.iter().enumerate() {
            tickets[i] = miss_tickets[k];
        }
        Ok(())
    }
}

/// Resolve every routed expert's cold-read spec `(fd, begin, len)` ONCE, indexed
/// `(layer - dense_layers) * n_experts + expert`. Each `.vq3`/`.i4` expert is a
/// single O_DIRECT-aligned block, so this just tabulates `read` (the source's
/// `read_spec`; range/dim checks ran at open).
fn build_moe_table(
    cfg: &ModelConfig,
    n_layers: usize,
    read: impl Fn(usize, usize) -> Result<(RawFd, usize, usize)>,
) -> Result<Vec<(RawFd, usize, usize)>> {
    let n_moe = n_layers - cfg.dense_layers;
    let mut table = Vec::with_capacity(n_moe * cfg.n_experts);
    for l in cfg.dense_layers..n_layers {
        for e in 0..cfg.n_experts {
            table.push(read(l, e)?);
        }
    }
    Ok(table)
}

#[cfg(test)]
mod ticket_tests {
    use crate::fetch::asyncfetch::Ticket;

    /// **INV-5: an expert cannot be launched without enqueueing its data dependency.**
    ///
    /// The structural half of this is enforced by types and cannot be tested at runtime:
    /// `submit_layer` returns `Vec<Ticket>` and no longer returns a residency mask, so
    /// `gpu.rs` has nothing to branch on and no way to spell "launch without waiting". What
    /// IS testable, and what actually broke before, is the encoding — a resident ticket must
    /// be a real satisfied dependency rather than a sentinel the consumer has to recognise
    /// and skip.
    ///
    /// Timelines start at 0, so `RESIDENT.value == 0` means "wait on 0", which every
    /// timeline satisfies on arrival. If it were, say, `u64::MAX` as an "N/A" marker, the
    /// consumer would need a branch to avoid deadlocking on it — and that branch is exactly
    /// the `hit` mask growing back.
    #[test]
    fn inv_5_every_descriptor_carries_a_ticket() {
        assert!(
            Ticket::RESIDENT.is_resident(),
            "the resident ticket must read as resident"
        );
        assert_eq!(
            Ticket::RESIDENT.value,
            0,
            "RESIDENT must be value 0 — a timeline starts there, so waiting on it is \
             satisfied immediately. A sentinel that had to be SKIPPED would put a residency \
             branch back in the consumer, which is the bug class this removed."
        );
        // A real fetch ticket is never confusable with a resident one: values are assigned
        // from 1 upward per slot.
        let fetched = Ticket { slot: 3, value: 1 };
        assert!(
            !fetched.is_resident(),
            "the first value a slot hands out must NOT read as already-satisfied"
        );
    }
}
