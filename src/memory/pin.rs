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
use crate::artifact::format::{
    Dtype, ExpertSet, FormatMeta, RoutedFmt, Safetensors, SetDims, f4_layer_range,
    load_codebooks,
};
use crate::artifact::model::{ModelConfig, V4Config};
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
    let (o_dim, i_dim) = dims2(&format!("{name}.weight"), shape)?;
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

/// A weight matrix's `(o_dim, i_dim)`, refusing anything that is not 2-D. Every placer
/// that carries dims takes them from the tensor's own shape rather than from `cfg`, so this
/// is the one place the rank is confronted.
fn dims2(name: &str, shape: &[usize]) -> Result<(usize, usize)> {
    ensure!(shape.len() == 2, "{name}: expected 2-D, got {shape:?}");
    Ok((shape[0], shape[1]))
}

/// Place an int8 per-row weight (`<name>` I8 + `<name>.scale` F32) into the tier
/// (embed / lm_head). Dims from the `[o_dim, i_dim]` shape.
fn place_i8(tier: &mut DeviceTier, st: &Safetensors, name: &str) -> Result<Int8Weight> {
    let (w, shape) = st.typed(name, Dtype::I8)?;
    let (o_dim, i_dim) = dims2(name, shape)?;
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
/// Total bytes of the `*.safetensors` in an artifact directory, less one file by name.
///
/// Shared by [`resident_bytes`] (GLM) and [`V4Pin::build`], and the reason is the one
/// [`resident_bytes`] gives at length: the converter writes exactly the tensor list the pin
/// places, so the FILE is the footprint and a second derivation of the placement layout is
/// the copy that drifts. `skip` is GLM's unplaced `indexer.safetensors`; V4 has none.
fn safetensors_bytes(dir: &str, skip: Option<&str>) -> Result<usize> {
    let mut total = 0usize;
    for entry in std::fs::read_dir(dir).with_context(|| format!("read dir {dir}"))? {
        let p = entry.with_context(|| format!("read dir {dir}"))?.path();
        let is_st = p.extension().is_some_and(|x| x == "safetensors");
        let skipped = skip.is_some_and(|s| p.file_name().is_some_and(|n| n == s));
        if !is_st || skipped {
            continue;
        }
        total += p.metadata().with_context(|| format!("stat {p:?}"))?.len() as usize;
    }
    Ok(total)
}

fn resident_bytes(
    dir: &str,
    cfg: &ModelConfig,
    n_pin: usize,
    shared_i4: bool,
    cb_resident: bool,
    want_indexer: bool,
) -> Result<usize> {
    let total = safetensors_bytes(dir, (!want_indexer).then_some("indexer.safetensors"))?;
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
    /// read signal has resolved. `trace` only — it costs a hash op per expert per layer.
    #[cfg(feature = "trace")]
    loaded: std::collections::HashSet<u32>,
    /// Misses submitted by the PREVIOUS layer, marked loaded at the top of the next
    /// `submit_layer`. Correct because layer L's per-expert awaits and its unconditional
    /// end-of-layer `device_sync` both complete before layer L+1 submits — so by then
    /// every byte of L's batch has landed. Deferring this way avoids plumbing a
    /// completion callback through `gpu.rs`'s async expert loop.
    #[cfg(feature = "trace")]
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
    #[cfg(feature = "trace")]
    fn mark_loaded(&mut self, key: u32) {
        self.loaded.insert(key);
    }

    /// Has `key`'s data actually landed since it was last admitted? A HIT on a key for
    /// which this is false is a read of uninitialised or stale bytes — the fault this
    /// instrumentation exists to catch.
    #[cfg(feature = "trace")]
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
            #[cfg(feature = "trace")]
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
        #[cfg(feature = "trace")]
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
        let dims = SetDims::new(
            cfg.dense_layers..n_pin,
            cfg.n_experts,
            cfg.hidden,
            cfg.moe_inter,
        );
        let vq_src = vq_present
            .then(|| ExpertSet::open_routed(dir, RoutedFmt::Vq3, dims))
            .transpose()?;
        let i4_src = i4
            .then(|| ExpertSet::open_routed(dir, RoutedFmt::I4, dims))
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
            #[cfg(feature = "trace")]
            loaded: std::collections::HashSet::new(),
            #[cfg(feature = "trace")]
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
        #[cfg(feature = "trace")]
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
                #[cfg(feature = "trace")]
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
        #[cfg(feature = "trace")]
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

// ── DeepSeek-V4-Flash resident set ──────────────────────────────────────────────
//
// A SEPARATE type from [`Pin`], for the reason `artifact::model` keeps `V4Config` separate
// from `ModelConfig`: the two architectures share no tensor name, no attention shape and
// no expert format, and a single pin parameterised by an arch flag would be a GLM-shaped
// placement path one `if` away from running on a V4 artifact. What IS shared is factored —
// `place_f32`/`place_fp8`/`place_shared`'s `DeviceTier` plumbing, `safetensors_bytes`, and
// `ExpertSet` itself.
//
// What this does NOT own, and why:
//   * The routed FP4 streaming pool. `ArenaPool` needs an `f4_slot_offsets` (the six
//     projection offsets inside one block) in `artifact::quant` and a format flag on the
//     MoE launcher in `gpu.rs` — neither file is this agent's. The `.f4` set is opened and
//     validated here and left public as `f4`, which is the input the pool is built from.
//   * The embed/head kernels. V4 carries `embed` and `head` as bf16 (`convert_v4`'s
//     `MODEL_LEVEL`), while `launch_embed_i8_row`/`launch_gemv_i8` are int8-only.

/// A resolved bf16 matrix in the tier: raw bf16 halves + dims.
///
/// Distinct from [`Int8Weight`] rather than widened into an f32 buffer at load: widening
/// would double `embed`+`head` from 2.1 GB to 4.2 GB of resident set to paper over the
/// missing kernels, and `convert_v4` kept them bf16 on purpose — whether to requantize is a
/// quality question with a paired-dNLL measurement attached, not a loader's decision.
#[derive(Clone, Copy)]
pub struct Bf16Weight {
    pub packed: *const u16,
    pub o_dim: usize,
    pub i_dim: usize,
}

/// One hyper-connection block's three f32 tables: `<base>_base`, `<base>_fn`, `<base>_scale`.
/// The same triple serves `layers.{l}.hc_attn`, `layers.{l}.hc_ffn` and the model-level
/// `hc_head`, which is why it is one type and one placer rather than three.
#[derive(Clone, Copy)]
pub struct HyperConn {
    pub base: *const f32,
    /// `_fn` — spelled out because `fn` is a keyword.
    pub func: *const f32,
    pub scale: *const f32,
}

/// One `Compressor`'s four tensors, all f32 in the artifact.
///
/// They are f32 because `Compressor.__init__` declares `wkv`/`wgate` fp32 and its forward
/// runs `x.float()` ("compression need fp32", the reference's own comment), so
/// `convert_v4::write_compressor` widens rather than choosing. Extents are NOT stored: they
/// vary with `compress_ratio` (`ape` is `[ratio, coff·head_dim]`, `coff = 1 + (ratio == 4)`)
/// and belong to the attention path that reads them, not to the pin that places them.
#[derive(Clone, Copy)]
pub struct V4Compressor {
    pub ape: *const f32,
    pub norm: *const f32,
    pub wgate: *const f32,
    pub wkv: *const f32,
}

/// The lightning indexer on a `compress_ratio == 4` layer.
///
/// **Nothing like GLM's [`IndexerPin`].** V4's `Indexer.__init__` gives it a second
/// `Compressor` of its own to build the keys it scores against — it has no `wk`/`k_norm`.
/// Its `compressor` is not `Option`: an indexer without one cannot exist.
#[derive(Clone, Copy)]
pub struct V4IndexerPin {
    pub wq_b: Fp8Weight,
    pub weights_proj: *const f32,
    pub compressor: V4Compressor,
}

/// How a layer's gate SELECTS experts.
///
/// A sum type rather than two `Option`s because the two are exclusive in the checkpoint —
/// a hash layer carries `ffn.gate.tid2eid` and no `.bias`, a scored layer the reverse — so
/// `Some`/`Some` and `None`/`None` are states no artifact can be in.
/// `V4Config::layer_routes_by_hash` decides which, and `convert_v4` wrote whichever it
/// decided, so the config and the artifact cannot disagree without one of them failing.
///
/// Both variants are HOST-side, like [`Pin::moe_bias`]: V4's routing (sqrtsoftplus scoring,
/// bias, top-k) runs on the CPU, and the hash path is a lookup by TOKEN ID, which the host
/// already holds. Placing 6.2 MB of `tid2eid` per hash layer on the device to index it
/// there would buy nothing.
pub enum V4Route {
    /// `tid2eid[token * top_k + j]` — already range-checked, see [`parse_tid2eid`].
    Hash { tid2eid: Vec<u32> },
    /// The router correction bias added before top-k, `n_experts` long.
    Scored { bias: Vec<f32> },
}

/// One V4 layer's resident weights.
pub struct V4LayerPin {
    pub attn_norm: *const f32,
    pub ffn_norm: *const f32,
    /// `q_norm` (over `q_lora_rank`) and `kv_norm` (over `head_dim`). The weightless QK-norm
    /// that follows `wq_b` has no tensor at all — it is `rsqrt(mean(q²) + eps)`.
    pub q_norm: *const f32,
    pub kv_norm: *const f32,
    /// `[n_heads]` f32, added to the softmax DENOMINATOR only.
    pub attn_sink: *const f32,
    pub wq_a: Fp8Weight,
    pub wq_b: Fp8Weight,
    /// ONE kv entry, `head_dim` wide, serving as both K and V for every head.
    pub wkv: Fp8Weight,
    pub wo_a: Fp8Weight,
    pub wo_b: Fp8Weight,
    /// Router gate `[n_experts, hidden]` f32, device-side (the scores are a GEMV).
    pub gate_w: *const f32,
    pub route: V4Route,
    pub hc_attn: HyperConn,
    pub hc_ffn: HyperConn,
    /// The always-on shared expert — fp8 e4m3, resident, NOT in the `.f4`.
    pub shared: Fp8Mlp,
    /// Present iff `compress_ratio != 0`.
    pub compressor: Option<V4Compressor>,
    /// Present iff `compress_ratio == 4`.
    pub indexer: Option<V4IndexerPin>,
}

/// The V4 resident weight set, plus the validated `.f4` routed-expert source.
pub struct V4Pin {
    #[allow(dead_code)] // RAII owner of the slab every pointer below points into.
    tier: DeviceTier,
    /// `[vocab, hidden]` bf16. See [`Bf16Weight`] — exposed, not consumed.
    pub embed: Bf16Weight,
    /// `head.weight`, `[vocab, hidden]` bf16. Untied from `embed` in this checkpoint.
    pub head: Bf16Weight,
    pub final_norm: *const f32,
    pub hc_head: HyperConn,
    /// Artifact order, so `layers[0]` is layer [`Self::range`]`.start` — which is NOT
    /// always 0. Private, and reachable only through [`Self::layer`], because that offset is
    /// exactly the kind of thing a caller gets right once and then forgets: an absolute
    /// layer id used as a direct index into a pin over layers 3..6 reads layer 6's weights
    /// for layer 3 and never fails.
    layers: Vec<V4LayerPin>,
    range: std::ops::Range<usize>,
    /// The `.f4` set: fd owner for the routed pool, and the thing whose headers and lengths
    /// — one per layer in the artifact's range — were validated at startup rather than at
    /// the first miss. `read_spec` is the streaming pool's input.
    pub f4: ExpertSet,
}

/// Place a bf16 matrix verbatim (`embed` / `head`). Dims from its `[o_dim, i_dim]` shape.
fn place_bf16(tier: &mut DeviceTier, st: &Safetensors, name: &str) -> Result<Bf16Weight> {
    let (w, shape) = st.typed(name, Dtype::Bf16)?;
    let (o_dim, i_dim) = dims2(name, shape)?;
    Ok(Bf16Weight {
        packed: tier.place(w)? as *const u16,
        o_dim,
        i_dim,
    })
}

/// Place one hyper-connection triple. `base` is the tensor-name PREFIX (`hc_head`,
/// `layers.3.hc_attn`), not a directory.
fn place_hc(tier: &mut DeviceTier, st: &Safetensors, base: &str) -> Result<HyperConn> {
    Ok(HyperConn {
        base: place_f32(tier, st, &format!("{base}_base"))?,
        func: place_f32(tier, st, &format!("{base}_fn"))?,
        scale: place_f32(tier, st, &format!("{base}_scale"))?,
    })
}

/// Place one `Compressor` — the attention's own or the indexer's, which differ only in
/// width and are therefore the same four names under different prefixes.
fn place_compressor(tier: &mut DeviceTier, st: &Safetensors, base: &str) -> Result<V4Compressor> {
    Ok(V4Compressor {
        ape: place_f32(tier, st, &format!("{base}.ape"))?,
        norm: place_f32(tier, st, &format!("{base}.norm.weight"))?,
        wgate: place_f32(tier, st, &format!("{base}.wgate.weight"))?,
        wkv: place_f32(tier, st, &format!("{base}.wkv.weight"))?,
    })
}

/// `ffn.gate.tid2eid` (I64) parsed into expert ids that are valid by construction.
///
/// **Nothing has ever looked at these.** `convert_v4` shape-checks the tensor against
/// `[vocab, top_k]` and then `copy_verbatim`s it, so no VALUE is ever read, and the MoE
/// launch path indexes its descriptor array with whatever arrives. An entry
/// outside `0..n_experts` therefore selects another expert's slot, or reads past the array;
/// `docs/investigations/v4-flash-port.md` §S3 item 10 records it as a load-time obligation
/// precisely because the kernels cannot make it.
///
/// Parsed to `u32` rather than checked and left `i64`, so the check and the storage are one
/// act: a negative or oversized id cannot exist in the returned vector. `u32::try_from`
/// rather than `as u32` because the cast truncates — 2^32 would become 0, an id that is
/// perfectly in range. 775,680 entries per hash layer x 3 layers, read once at startup.
///
/// Takes the three scalars rather than `&V4Config` so its test needs no config, and so no
/// machine can end up running a vacuous version of it.
fn parse_tid2eid(
    raw: &[u8],
    shape: &[usize],
    vocab: usize,
    top_k: usize,
    n_experts: usize,
) -> Result<Vec<u32>> {
    ensure!(
        shape == [vocab, top_k],
        "ffn.gate.tid2eid: shape {shape:?} != [{vocab}, {top_k}]"
    );
    // The EXTENT, separately from the shape. `Safetensors::typed` matches the dtype and
    // nothing else — `Loc.len` comes from the header's `data_offsets` and is never
    // confronted with `product(shape) x 8` — so a tensor whose byte span disagrees with its
    // declared shape passes the check above, `chunks_exact` drops the partial tail, and the
    // returned table is SHORT. `V4Route::Hash`'s consumer indexes it at
    // `token * top_k + j`, so a short table is an out-of-bounds read for the last tokens.
    let want = vocab * top_k * 8;
    ensure!(
        raw.len() == want,
        "ffn.gate.tid2eid: {} bytes for shape [{vocab}, {top_k}] — expected {want}",
        raw.len()
    );
    raw.chunks_exact(8)
        .enumerate()
        .map(|(k, c)| {
            let v = i64::from_le_bytes(c.try_into()?);
            let e = u32::try_from(v).ok().filter(|&e| (e as usize) < n_experts);
            e.with_context(|| {
                format!(
                    "ffn.gate.tid2eid[{}][{}] = {v}, outside 0..n_experts={n_experts}",
                    k / top_k,
                    k % top_k,
                )
            })
        })
        .collect()
}

impl V4Pin {
    /// Build the V4 resident set from artifact directory `dir`.
    ///
    /// No `capacity` argument, unlike [`Pin::build`]: the resident footprint is the
    /// artifact's own size, not a budget, and the rest of the device budget has nothing to
    /// grow yet — the routed pool is not wired here (see the module note above). A
    /// `capacity` this ignored would be a parameter documenting a policy that does not run.
    pub fn build(dir: &str, cfg: &V4Config) -> Result<Self> {
        // Which layers the artifact HOLDS. `num_hidden_layers` is the model's; the two
        // differ on every partial convert. See `f4_layer_range`.
        let range = f4_layer_range(dir, cfg.n_layers)?;
        // NOT required to start at 0. That is a property of a DECODE — a forward pass has
        // no residual stream to enter at layer 3 — and belongs where the decode is set up,
        // not in a loader. Refusing it here made every partial artifact but the first
        // unloadable, and `/var/db/rivoli/v4-f4-l3-5` is the one that carries the scored
        // router and the ratio-128-without-indexer shape that layers 0-2 have neither of.
        // fp8 block size, and the same VQ-param/version gate every artifact passes.
        let block = FormatMeta::load(dir)?.fp8_block;
        let st = Safetensors::open_dir(dir)?;
        // Spans exactly the artifact's range — NOT `0..cfg.n_layers` — and validates every
        // header and length here rather than at the first miss.
        let f4 = ExpertSet::open_routed(
            dir,
            RoutedFmt::F4,
            SetDims::new(range.clone(), cfg.n_experts, cfg.hidden, cfg.moe_inter),
        )?;

        // The total of every `*.safetensors` in `dir` — `convert_v4` writes only
        // `resident.safetensors`, so that is what this is. Everything placed comes out of
        // it, so the length is an upper bound and can never UNDER-count, which is the
        // failure `DeviceTier::place` bails on. It over-counts by the two host-side tensors
        // (`tid2eid`, `ffn.gate.bias`), costing only unused tier. Nothing to add: V4 has no
        // codebooks, and its shared expert is already inside the file rather than in the
        // routed slab.
        // Alignment slack. `DeviceTier::place` starts every reservation at
        // `used.next_multiple_of(256)` (`device::bump`), so total padding is under 256 B per
        // placement; V4 places ~40 tensors a layer over <=43 layers plus 4 model-level, so
        // under 0.5 MB. 16 MiB is ~30x that bound — deliberately loose, because an
        // under-count is what `DeviceTier::place` bails on while an over-count only costs
        // unused tier (there is no routed pool competing for the remainder yet).
        const SLACK: usize = 16 << 20;
        let resident = safetensors_bytes(dir, None)?;
        tracing::info!(
            "v4 resident footprint {:.2} GiB over layers [{}, {})",
            resident as f64 / (1u64 << 30) as f64,
            range.start,
            range.end,
        );
        let mut tier = DeviceTier::new(resident + SLACK)?;

        let embed = place_bf16(&mut tier, &st, "embed.weight")?;
        let head = place_bf16(&mut tier, &st, "head.weight")?;
        let final_norm = place_f32(&mut tier, &st, "norm.weight")?;
        let hc_head = place_hc(&mut tier, &st, "hc_head")?;

        let mut layers = Vec::with_capacity(range.len());
        for l in range.clone() {
            let lb = format!("layers.{l}");
            let a = format!("{lb}.attn");
            // Driven off the CONFIG, never off `st.has(…)` — the same choice `convert_v4`
            // made and for the same reason: a layer whose tensors disagree with
            // `compress_ratios`/`num_hash_layers` must fail here, not silently take
            // whichever branch the artifact happens to satisfy.
            let route = if cfg.layer_routes_by_hash(l) {
                let (raw, shape) = st.typed(&format!("{lb}.ffn.gate.tid2eid"), Dtype::I64)?;
                V4Route::Hash {
                    tid2eid: parse_tid2eid(raw, shape, cfg.vocab, cfg.top_k, cfg.n_experts)
                        .with_context(|| format!("layer {l}"))?,
                }
            } else {
                let (raw, _) = st.typed(&format!("{lb}.ffn.gate.bias"), Dtype::F32)?;
                let bias = crate::artifact::quant::read_f32(raw);
                ensure!(
                    bias.len() == cfg.n_experts,
                    "layer {l} ffn.gate.bias has {} entries, expected {}",
                    bias.len(),
                    cfg.n_experts
                );
                V4Route::Scored { bias }
            };
            let mut fp8 = |name: &str| place_fp8(&mut tier, &st, &format!("{a}.{name}"), block);
            let (wq_a, wq_b) = (fp8("wq_a")?, fp8("wq_b")?);
            let (wkv, wo_a, wo_b) = (fp8("wkv")?, fp8("wo_a")?, fp8("wo_b")?);
            // `e == n_experts` selects `ffn.shared_experts` — see `v4_expert_base`.
            let shared_base =
                crate::artifact::quant::v4_expert_base(l, cfg.n_experts, cfg.n_experts);
            // gate/up/down == w1/w3/w2. `V4_PROJ` carries the argument for that order; the
            // slot meaning has to match `Fp8Mlp`'s fields, not the tensors' names.
            let [w1, w3, w2] = crate::artifact::quant::V4_PROJ;
            let mut sh = |p: &str| place_fp8(&mut tier, &st, &format!("{shared_base}.{p}"), block);
            let shared = Fp8Mlp {
                gate: sh(w1)?,
                up: sh(w3)?,
                down: sh(w2)?,
            };
            layers.push(V4LayerPin {
                attn_norm: place_f32(&mut tier, &st, &format!("{lb}.attn_norm.weight"))?,
                ffn_norm: place_f32(&mut tier, &st, &format!("{lb}.ffn_norm.weight"))?,
                q_norm: place_f32(&mut tier, &st, &format!("{a}.q_norm.weight"))?,
                kv_norm: place_f32(&mut tier, &st, &format!("{a}.kv_norm.weight"))?,
                attn_sink: place_f32(&mut tier, &st, &format!("{a}.attn_sink"))?,
                wq_a,
                wq_b,
                wkv,
                wo_a,
                wo_b,
                gate_w: place_f32(&mut tier, &st, &format!("{lb}.ffn.gate.weight"))?,
                route,
                hc_attn: place_hc(&mut tier, &st, &format!("{lb}.hc_attn"))?,
                hc_ffn: place_hc(&mut tier, &st, &format!("{lb}.hc_ffn"))?,
                shared,
                compressor: cfg
                    .layer_has_compressor(l)?
                    .then(|| place_compressor(&mut tier, &st, &format!("{a}.compressor")))
                    .transpose()?,
                indexer: cfg
                    .layer_has_indexer(l)?
                    .then(|| -> Result<V4IndexerPin> {
                        Ok(V4IndexerPin {
                            wq_b: place_fp8(&mut tier, &st, &format!("{a}.indexer.wq_b"), block)?,
                            weights_proj: place_f32(
                                &mut tier,
                                &st,
                                &format!("{a}.indexer.weights_proj.weight"),
                            )?,
                            compressor: place_compressor(
                                &mut tier,
                                &st,
                                &format!("{a}.indexer.compressor"),
                            )?,
                        })
                    })
                    .transpose()?,
            });
        }

        Ok(Self {
            tier,
            embed,
            head,
            final_norm,
            hc_head,
            layers,
            range,
            f4,
        })
    }

    /// The layer range this pin holds, in the model's own numbering.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.range.clone()
    }

    /// One layer's resident weights by ABSOLUTE layer id.
    ///
    /// The only way in, so the artifact-order offset cannot be applied twice or not at all.
    pub fn layer(&self, l: usize) -> Result<&V4LayerPin> {
        self.layers
            .get(l.checked_sub(self.range.start).unwrap_or(usize::MAX))
            .with_context(|| {
                format!(
                    "layer {l} is outside this artifact's range [{}, {})",
                    self.range.start, self.range.end
                )
            })
    }
}

#[cfg(test)]
mod v4_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
    use super::*;

    const VOCAB: usize = 4;
    const TOP_K: usize = 3;
    const N_EXPERTS: usize = 8;

    fn le(v: &[i64]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// **`tid2eid` values are checked at load because nothing else can check them.**
    /// `convert_v4` shape-checks the tensor and copies it verbatim, and the MoE launch path
    /// indexes descriptors with whatever arrives, so an out-of-range id reads another
    /// expert's slot.
    ///
    /// Both directions: a table of valid ids must survive UNCHANGED — a check that mangled
    /// good data would be worse than none — and each way of being invalid must be caught.
    /// Unconditional: `parse_tid2eid` takes three scalars rather than a `&V4Config`
    /// precisely so this cannot degrade into a skip on a machine without the checkpoint.
    #[test]
    fn tid2eid_entries_are_range_checked_and_valid_ones_pass_through() {
        let good: Vec<i64> = (0..(VOCAB * TOP_K) as i64).map(|k| k % 8).collect();
        let got = parse_tid2eid(&le(&good), &[VOCAB, TOP_K], VOCAB, TOP_K, N_EXPERTS).unwrap();
        assert_eq!(
            got,
            good.iter().map(|&v| v as u32).collect::<Vec<_>>(),
            "valid ids must pass through bit-identically"
        );

        // One entry at a time, so each failure is attributable to that value rather than to
        // a table broken in several ways at once.
        for (label, bad) in [
            ("== n_experts", N_EXPERTS as i64),
            ("> n_experts", N_EXPERTS as i64 + 1),
            ("negative", -1i64),
            // 2^32 truncates to 0 under `as u32` — an id that is perfectly IN range. This
            // is the case only `u32::try_from` catches.
            ("wraps u32", 1i64 << 32),
        ] {
            let mut v = good.clone();
            v[5] = bad;
            let e = format!(
                "{:#}",
                parse_tid2eid(&le(&v), &[VOCAB, TOP_K], VOCAB, TOP_K, N_EXPERTS)
                    .err()
                    .unwrap_or_else(|| panic!("{label} ({bad}) must be refused"))
            );
            assert!(
                e.contains(&format!("[{}][{}] = {bad}", 5 / TOP_K, 5 % TOP_K)),
                "{label}: the refusal must name the offending entry, got: {e}"
            );
        }

        // A shape that is not [vocab, top_k] is refused before any value is read, or the
        // row/column arithmetic in the message above would be nonsense.
        assert!(parse_tid2eid(&le(&good), &[TOP_K, VOCAB], VOCAB, TOP_K, N_EXPERTS).is_err());

        // The EXTENT, which the shape does not imply. `Safetensors::typed` matches the
        // dtype only, so a tensor whose byte span is shorter than its declared shape gets
        // here — and `chunks_exact` would silently drop the tail and return a SHORT table
        // that the consumer indexes at `token * top_k + j`. Both a truncated buffer and a
        // ragged one (not a multiple of 8) must be refused, not rounded off.
        for (label, raw) in [
            ("truncated", le(&good[..good.len() - 1])),
            ("ragged", le(&good)[..VOCAB * TOP_K * 8 - 3].to_vec()),
            ("too long", le(&[good.clone(), vec![0]].concat())),
        ] {
            let e = format!(
                "{:#}",
                parse_tid2eid(&raw, &[VOCAB, TOP_K], VOCAB, TOP_K, N_EXPERTS)
                    .err()
                    .unwrap_or_else(|| panic!("{label} must be refused"))
            );
            assert!(e.contains("expected 96"), "{label}: got {e}");
        }
    }
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
