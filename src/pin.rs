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
//! `rocm`-only: without a device there is nothing to pin.
#![cfg(feature = "rocm")]

use crate::asyncfetch::{AsyncFetch, ReadSpec};
use crate::arena::{Arena, Reloc, Step};
use crate::cache;
use crate::config::Mode;
use crate::device::{DeviceTier, VmmBuf};
use crate::hip::memcpy_dtod;
use crate::hybrid::HybridPolicy;
use std::collections::HashMap;
use crate::format::{Dtype, ExpertSet, FormatMeta, Safetensors, load_codebooks};
use crate::gpustream::Signal;
use crate::model::ModelConfig;
use crate::quant::{
    VQ_DIM, VQ_K, i4_expert_bytes, i4_slot_offsets, vq_expert_bytes, vq_slot_offsets,
};
use crate::stream::{Streamer, slot_span};
use anyhow::{Context, Result, bail, ensure};
use std::os::fd::RawFd;


/// A resolved fp8-e4m3 block-scaled weight matrix in the tier: device pointers +
/// dims + the `weight_scale_inv` block size. Consumed by `launch_gemv_fp8` (attn +
/// dense projections) and, for kv_b, by the MLA absorb/value kernels.
#[derive(Clone, Copy)]
pub struct Fp8Weight {
    pub packed: *const u8,
    pub scale: *const f32,
    pub o_dim: usize,
    pub i_dim: usize,
    pub block: usize,
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
    /// and resolves each miss's load [`Signal`] when its bytes land. The expert
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
        o_dim,
        i_dim,
        block,
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
fn place_shared(
    tier: &mut DeviceTier,
    block: &[u8],
    off: &[usize; 6],
) -> Result<MlpVq> {
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

/// Device bytes an fp8-e4m3 block-scaled `[o,i]` weight occupies: `o·i` packed +
/// `⌈o/block⌉·⌈i/block⌉·4` for the F32 block-scale. Single source for the sizing
/// formulas below (`place_fp8` reserves the shard's own byte length, which this
/// mirrors) — write the layout once so it can't drift between the two sizers.
fn fp8_bytes(o: usize, i: usize, block: usize) -> usize {
    o * i + o.div_ceil(block) * i.div_ceil(block) * 4
}
/// Device bytes an int8 per-row `[o,i]` weight occupies: `o·i` packed + `o·4` F32 scale.
fn i8_bytes(o: usize, i: usize) -> usize {
    o * i + o * 4
}
/// Device bytes an f32 vector of `n` elements occupies.
fn f32_bytes(n: usize) -> usize {
    n * 4
}

/// Device bytes ONE full layer's DSA indexer weights occupy, mirroring
/// [`place_indexer`]: fp8 `wk`/`wq_b`, f32 `weights_proj` + f32 `k_norm`.
fn indexer_bytes(cfg: &ModelConfig, block: usize) -> usize {
    let fp8 = |o: usize, i: usize| fp8_bytes(o, i, block);
    let hd = cfg.index_head_dim;
    let nh = cfg.index_n_heads;
    fp8(hd, cfg.hidden)                  // wk (fp8)
        + fp8(nh * hd, cfg.q_lora_rank)  // wq_b (fp8)
        + f32_bytes(nh * cfg.hidden)     // weights_proj (bf16→f32)
        + 2 * f32_bytes(hd) // k_norm weight + bias (f32)
}

/// Device bytes the always-resident set occupies — everything read every token
/// EXCEPT the routed experts. Summed from `cfg` so the resident tier is sized to
/// what it holds and the rest of the device budget grows the routed pool. Mirrors
/// the placement path: fp8 `[o,i]` = `o·i` packed + `⌈o/block⌉·⌈i/block⌉·4` scale;
/// int8 `[o,i]` = `o·i` + `o·4`; an f32 norm of `n` = `n·4`; plus the 3 codebooks
/// and one VQ shared expert per MoE layer.
/// `shared_i4`: the always-resident shared expert is int4 (else int3-VQ).
/// `cb_resident`: the 3 fp16 VQ codebooks are resident (any VQ slab present).
fn resident_bytes(cfg: &ModelConfig, block: usize, shared_i4: bool, cb_resident: bool) -> usize {
    let fp8 = |o: usize, i: usize| fp8_bytes(o, i, block);
    let i8 = i8_bytes;
    let f32n = f32_bytes;

    let qk = cfg.qk_head_dim();
    let mut total = i8(cfg.vocab, cfg.hidden) // embed_tokens
        + i8(cfg.vocab, cfg.hidden)           // lm_head
        + f32n(cfg.hidden); // model.norm

    for l in 0..cfg.n_layers {
        // Norms: input, post-attn, q_a, kv_a.
        total += 2 * f32n(cfg.hidden) + f32n(cfg.q_lora_rank) + f32n(cfg.kv_lora_rank);
        // MLA projections (fp8).
        total += fp8(cfg.q_lora_rank, cfg.hidden); // q_a
        total += fp8(cfg.n_heads * qk, cfg.q_lora_rank); // q_b
        total += fp8(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden); // kv_a
        total += fp8(
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            cfg.kv_lora_rank,
        ); // kv_b
        total += fp8(cfg.hidden, cfg.n_heads * cfg.v_head_dim); // o_proj
        if l < cfg.dense_layers {
            total += 2 * fp8(cfg.dense_inter, cfg.hidden); // gate, up
            total += fp8(cfg.hidden, cfg.dense_inter); // down
        } else {
            total += f32n(cfg.n_experts * cfg.hidden); // router gate (F32, device)
            // Shared expert, in the routed format (int4 blocks are larger than VQ).
            total += if shared_i4 {
                i4_expert_bytes(cfg.hidden, cfg.moe_inter)
            } else {
                vq_expert_bytes(cfg.hidden, cfg.moe_inter)
            };
        }
    }
    // 3 per-projection fp16 codebooks — resident whenever a VQ slab is present.
    total + if cb_resident { 3 * VQ_K * VQ_DIM * 2 } else { 0 }
}

/// Width of the trace-v2 candidate window: the top-W router candidates recorded per
/// routing decision, on top of the `top_k` that actually ran. W bounds the largest M
/// the offline (J, M) substitution grid in docs/CACHE_ROUTE.md can explore — an M
/// wider than this cannot be evaluated from a captured trace without recapturing.
/// 32 is 4× `top_k` (8) and an eighth of `n_experts` (256): far past any M where
/// promoting a resident-but-lower-ranked expert is still defensible, and only ~380
/// bytes a line.
pub const TRACE_WINDOW: usize = 32;

/// Pack `(layer, expert)` into the pool key. Both must fit in 16 bits — GLM is
/// ≤92 layers × 256 routed experts, comfortably under 2^16.
fn expert_key(layer: usize, expert: usize) -> u32 {
    debug_assert!(
        layer < (1 << 16) && expert < (1 << 16),
        "layer {layer}/expert {expert} exceed the 16-bit pool key packing"
    );
    ((layer as u32) << 16) | expert as u32
}

/// Everything `submit_layer` needs to turn a resolved slot into an expert descriptor:
/// its device base pointer, the six projection offsets, and its weight format (which
/// kernel gpu.rs launches). Captured AFTER a batch's relocations settle.
#[derive(Clone, Copy)]
struct ResolvedSlot {
    ptr: *mut u8,
    off: [usize; 6],
    int4: bool,
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
    #[allow(dead_code)] // RAII owner of the pool VMM; addressed via `base`
    buf: VmmBuf,
    base: *mut u8,
    arena: Arena,
    policy: Box<dyn HybridPolicy>,
    slot_of: HashMap<u32, (bool, usize)>, // key -> (hot, idx)
    key_at: HashMap<(bool, usize), u32>,  // (hot, idx) -> key, for relocation remap
    /// Keys whose bytes are known to have LANDED in their current slot.
    ///
    /// The engine has no other way to distinguish "the policy says resident" from "the
    /// bytes are actually there", and that distinction is the leading hypothesis for the
    /// intermittent non-finite-logits bug: a HIT returns `Signal::ready()` and the kernel
    /// reads the slot immediately, so if a key is ever counted resident before its load
    /// completed, the read is of uninitialised (-> NaN, the visible case) or stale (->
    /// finite and WRONG, the silent case) memory.
    ///
    /// A key is removed on eviction and on relocation-into, and inserted only when its
    /// read signal has resolved. `trace` only — it costs a hash op per expert per layer.
    #[cfg(feature = "trace")]
    loaded: std::collections::HashSet<u32>,
    /// Misses submitted by the PREVIOUS layer, marked loaded at the top of the next
    /// `submit_spine`. Correct because layer L's per-expert awaits and its unconditional
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
    /// The slot's pointer, valid TODAY as both a host DMA target (`ReadSpec.dst`, the
    /// io_uring O_DIRECT destination) and a device address (the base every expert
    /// descriptor's six projection pointers are built from). Those are one number
    /// only because HIP unified addressing makes them one.
    ///
    /// A second backend must split this per consumer — descriptors take the device
    /// base, `ReadSpec.dst` takes the host base — with both bases still resolved once
    /// at setup, so this stays a single `add`. See docs/VULKAN.md, "Host pointer !=
    /// device address".
    fn ptr(&self, hot: bool, idx: usize) -> *mut u8 {
        // SAFETY: arena.offset < budget, within the pool VMM.
        unsafe { self.base.add(self.arena.offset(hot, idx)) }
    }
    fn resolved(&self, hot: bool, idx: usize) -> ResolvedSlot {
        let t = self.tier(hot);
        ResolvedSlot { ptr: self.ptr(hot, idx), off: t.off, int4: t.int4 }
    }
    /// Start a batch: reset the policy's per-batch pin set so evictions this batch never
    /// reclaim a key this batch touches (would surface as "expert not resident").
    fn begin_batch(&mut self) {
        self.policy.begin_batch();
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

    /// A hit: the policy refreshes recency; the physical slot is unchanged and read from
    /// `slot_of` LATER (after any same-batch relocations settle).
    fn get(&mut self, key: u32) -> bool {
        self.policy.get(key)
    }
    fn protect(&mut self, key: u32) {
        self.policy.protect(key);
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
            let s = self.slot_of.remove(&ev).context("evicted key had no slot")?;
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
        Ok(())
    }

    /// Execute one compaction relocation: memcpy the slot's bytes `from`→`to` (distinct,
    /// non-overlapping slots) and remap the key that lived there. Synchronous, so it
    /// lands before the layer's compute or any later cold read touches the new slot.
    fn relocate(&mut self, r: Reloc) -> Result<()> {
        let moved = self.key_at.remove(&(r.hot, r.from)).context("relocated slot had no key")?;
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
    /// Build the resident set from the artifact directory `dir`. `capacity` is the
    /// total device budget (auto-discovered); the always-resident set takes its
    /// computed footprint and the rest grows the routed pool. `bounce` selects the
    /// streamer's destination path (see [`Streamer`]).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        dir: &str,
        cfg: &'a ModelConfig,
        capacity: usize,
        bounce: bool,
        trace_path: Option<&str>,
        cache_policy: &str,
        two_q: cache::TwoQSplit,
        route: crate::hybrid::RouteAdvice,
        want_indexer: bool,
        mode: Mode,
    ) -> Result<Self> {
        // `i4` = the int4 placement path is needed (int4 mode, or the hybrid HOT tier +
        // shared expert). int3-VQ uses neither. See MODES.md.
        let i4 = mode.uses_int4();
        // ponytail: no free-memory pre-check — the budget is the user's literal
        // request (--max-mem), so let the device allocation itself OOM/fail.
        // One-time bound for `submit_layer`'s fixed 32-slot scratch.
        ensure!(
            cfg.experts_per_layer() <= 32,
            "top_k {} + n_shared {} exceeds the 32-slot batch scratch",
            cfg.top_k,
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
        let vq_src = vq_present
            .then(|| {
                ExpertSet::open_vq3(
                    dir,
                    cfg.dense_layers,
                    cfg.n_layers,
                    cfg.n_experts,
                    cfg.hidden,
                    cfg.moe_inter,
                )
            })
            .transpose()?;
        let i4_src = i4
            .then(|| {
                ExpertSet::open_i4(
                    dir,
                    cfg.dense_layers,
                    cfg.n_layers,
                    cfg.n_experts,
                    cfg.hidden,
                    cfg.moe_inter,
                )
            })
            .transpose()?;
        let shared_i4 = i4;
        // Slot byte layouts, one per format (which projection's indices/scales sit
        // where in an expert block). Shared expert + routed slabs reuse these.
        let vq_off = vq_slot_offsets(cfg.hidden, cfg.moe_inter);
        let i4_off = i4_slot_offsets(cfg.hidden, cfg.moe_inter);
        let shared_off = if shared_i4 { i4_off } else { vq_off };

        // Full layers own a resident DSA indexer (dsa/misa modes). Empty when
        // dense/streaming — the mask drives both placement and the footprint.
        let full = if want_indexer {
            cfg.indexer_layout()?
        } else {
            vec![false; cfg.n_layers]
        };
        let n_full = full.iter().filter(|&&f| f).count();

        // Size the tier to the always-resident footprint plus slack (absorbs the
        // per-reservation 256-byte alignment padding); anything left widens the pool.
        const SLACK: usize = 256 << 20; // 256 MiB
        let resident =
            resident_bytes(cfg, block, shared_i4, vq_present) + n_full * indexer_bytes(cfg, block);
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
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut moe_bias = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for l in 0..cfg.n_layers {
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
                let bias = crate::quant::read_f32(bias);
                ensure!(
                    bias.len() == cfg.n_experts,
                    "layer {l} gate bias has {} entries, expected {}",
                    bias.len(),
                    cfg.n_experts
                );
                moe_bias.push(bias);
                let shared_block = if shared_i4 {
                    i4_src.as_ref().context("i4 shared: source missing")?.shared_block(l)?
                } else {
                    vq_src.as_ref().context("vq shared: source missing")?.shared_block(l)?
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

        // Routed pool: a two-ended byte Arena over the budget left after the resident
        // set. Each expert is ONE aligned block read into a slot. COLD/HOT tiers are one
        // format (single-format: uniform stride, so a compaction relocation is always a
        // single cheap same-size move) or int3-VQ/int4 (hybrid). The byte-aware policy
        // floats the split; a cross-tier rebalance relocates a slot.
        // Round the pool budget DOWN to the O_DIRECT block so the arena's HIGH-end
        // anchor is aligned: HOT slots sit at `budget − (idx+1)*hot_stride`, so an
        // unaligned `budget` makes every hot-slot dst violate the O_DIRECT alignment
        // the streamer asserts (base + strides are already 4096-aligned). Costs <4 KiB.
        let budget = capacity.saturating_sub(tier_cap) & !(crate::stream::ALIGN - 1);
        let vq_tier = || -> Result<TierFmt> {
            let s = vq_src.as_ref().context("vq source missing")?;
            Ok(TierFmt {
                off: vq_off,
                int4: false,
                stride: s.expert_slot(),
                table: build_moe_table(cfg, |l, e| s.read_spec(l, e))?,
            })
        };
        let i4_tier = || -> Result<TierFmt> {
            let s = i4_src.as_ref().context("i4 source missing")?;
            Ok(TierFmt {
                off: i4_off,
                int4: true,
                stride: s.expert_slot(),
                table: build_moe_table(cfg, |l, e| s.read_spec(l, e))?,
            })
        };
        // COLD/HOT tiers by mode. Single-format shares one format across both tiers.
        let (cold, hot) = match mode {
            Mode::Int3Vq => {
                let t = vq_tier()?;
                (t.clone(), t)
            }
            Mode::Int4 => {
                let t = i4_tier()?;
                (t.clone(), t)
            }
            Mode::Hybrid => (vq_tier()?, i4_tier()?),
        };
        let (cold_stride, hot_stride) = (cold.stride, hot.stride);
        let policy =
            crate::hybrid::make(cache_policy, budget, cold_stride, hot_stride, two_q, route)
                .with_context(|| {
                    format!("unknown --cache-policy {cache_policy} (lru|2q|arc|top-m)")
                })?;
        tracing::info!(
            "routed pool [{cache_policy} {mode}]: {:.1} GiB budget (~{} slots, cold {cold_stride}B / hot {hot_stride}B)",
            budget as f64 / (1u64 << 30) as f64,
            budget / cold_stride.min(hot_stride),
        );
        let mut buf = VmmBuf::new(budget)?;
        let base = buf.ptr_mut();
        let routed = ArenaPool {
            buf,
            base,
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
        let fetch = AsyncFetch::new(Streamer::new(ring as u32, span, bounce)?)?;

        Ok(Self {
            cfg,
            tier,
            embed,
            lm_head,
            final_norm,
            layers,
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

    /// The active policy's routing advice — `Some((j, m))` only under `--cache-policy
    /// top-m`. gpu.rs reads this ONCE at engine construction: the policy is fixed for
    /// the run, and `None` has to stay a compile-time-cheap early return on the hot
    /// routing path.
    pub fn route_advice(&self) -> Option<(usize, usize)> {
        self.routed.policy.route_advice()
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

    /// The streaming half of `submit_layer`: trace sink, then three phases over the
    /// arena pool. 1a: touch every HIT (protect it so a same-batch miss can't evict it).
    /// 1b: allocate every MISS — this is where the byte-aware policy evicts and the arena
    /// may RELOCATE resident slots. 1c: only NOW, after all relocations have settled,
    /// resolve each key's final slot and build the misses' cold reads — so a read never
    /// targets a slot that later moves. Returns the per-`sel` resolved slots + signals.
    fn submit_spine(
        &mut self,
        layer: usize,
        sel: &[usize],
        window: &[usize],
        choice: &[f32],
    ) -> Result<([Option<ResolvedSlot>; 32], Vec<Signal>)> {
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
        self.routed.begin_batch();
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
            // (docs/CACHE_ROUTE.md "Counters") is deferred, not cancelled, and needs the
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
            if self.routed.get(expert_key(layer, e)) {
                self.hits += 1;
                // THE CHECK. A hit hands the kernel a slot pointer and a resolved
                // Signal, so nothing downstream waits. If the bytes never landed, the
                // kernel reads uninitialised memory (NaN) or another expert's weights
                // (finite, wrong, silent). Reported once per occurrence rather than
                // fataling, so a run keeps going and the pattern is visible.
                #[cfg(feature = "trace")]
                if !self.routed.is_loaded(expert_key(layer, e)) {
                    tracing::error!(
                        "READ-BEFORE-WRITE: layer={layer} expert={e} counted as a cache \
                         HIT but its bytes never landed since admission. The kernel is \
                         about to read an unloaded slot — uninitialised memory (-> NaN) \
                         or a previous expert's weights (-> silently wrong)."
                    );
                }
                self.routed.protect(expert_key(layer, e));
                is_hit[i] = true;
            }
        }
        // Phase 1b: allocate the misses (evict + place + compact). Slots may relocate.
        for (i, &e) in sel.iter().enumerate() {
            if !is_hit[i] {
                self.misses += 1;
                self.routed.alloc(expert_key(layer, e))?;
            }
        }
        // Phase 1c: relocations have settled — resolve final slots and build the reads.
        let mut slots: [Option<ResolvedSlot>; 32] = [None; 32];
        let mut reads: Vec<ReadSpec> = Vec::new();
        let mut miss_sel: Vec<usize> = Vec::new();
        for (i, &e) in sel.iter().enumerate() {
            let (hot, idx) = self.routed.slot(expert_key(layer, e)).context(
                "expert not resident after alloc (batch exceeds pool — raise --max-mem)",
            )?;
            slots[i] = Some(self.routed.resolved(hot, idx));
            if !is_hit[i] {
                let (fd, begin, len) = self.routed.tier(hot).table[sparse + e];
                reads.push(ReadSpec { fd, begin, len, dst: self.routed.ptr(hot, idx) });
                miss_sel.push(i);
            }
        }
        // Phase 2: hand the whole batch to the reaper — it queues+submits (all reads
        // start at once) and resolves each miss's Signal when its copy lands. Hits
        // default to `ready()`; overwrite each miss with its load Signal.
        // Queue this batch's misses to be marked loaded at the next layer.
        #[cfg(feature = "trace")]
        for &i in &miss_sel {
            self.routed.pending_loaded.push(expert_key(layer, sel[i]));
        }
        let miss_signals = self.fetch.submit(reads)?;
        let mut signals: Vec<Signal> = (0..sel.len()).map(|_| Signal::ready()).collect();
        for (k, &i) in miss_sel.iter().enumerate() {
            signals[i] = miss_signals[k].clone();
        }
        Ok((slots, signals))
    }

    /// Submit one layer's cold reads and resolve each selected expert to its [`MlpVq`]
    /// (device pointers into the pool) + per-expert format flag + load [`Signal`]. The
    /// descriptor pointers are final (post-relocation); the bytes land when `signals[i]`
    /// resolves. The expert stream awaits each signal before computing that expert.
    ///
    /// `window`/`choice` feed the trace sink only: the ranked top-[`TRACE_WINDOW`]
    /// candidate expert ids and the full per-expert `choice` array they index into.
    /// Pass an empty `window` when not tracing — nothing else reads them.
    pub fn submit_layer(
        &mut self,
        layer: usize,
        sel: &[usize],
        window: &[usize],
        choice: &[f32],
        out: &mut Vec<MlpVq>,
        fmt: &mut Vec<bool>,
    ) -> Result<Vec<Signal>> {
        let (slots, signals) = self.submit_spine(layer, sel, window, choice)?;
        out.clear();
        fmt.clear();
        for (i, _e) in sel.iter().enumerate() {
            let s = slots[i].context("submit_layer: unresolved expert slot")?;
            let (b, o) = (s.ptr, s.off);
            // SAFETY: address arithmetic into the resolved slot; the bytes land when
            // `signals[i]` resolves. The six pointers are identical for both formats
            // (gpu.rs reinterprets as ExpertDescI4 when `fmt[i]`).
            out.push(MlpVq {
                gate: vqweight_at(b, o[0], o[1]),
                up: vqweight_at(b, o[2], o[3]),
                down: vqweight_at(b, o[4], o[5]),
            });
            fmt.push(s.int4);
        }
        Ok(signals)
    }
}

/// Resolve every routed expert's cold-read spec `(fd, begin, len)` ONCE, indexed
/// `(layer - dense_layers) * n_experts + expert`. Each `.vq3`/`.i4` expert is a
/// single O_DIRECT-aligned block, so this just tabulates `read` (the source's
/// `read_spec`; range/dim checks ran at open).
fn build_moe_table(
    cfg: &ModelConfig,
    read: impl Fn(usize, usize) -> Result<(RawFd, usize, usize)>,
) -> Result<Vec<(RawFd, usize, usize)>> {
    let n_moe = cfg.n_layers - cfg.dense_layers;
    let mut table = Vec::with_capacity(n_moe * cfg.n_experts);
    for l in cfg.dense_layers..cfg.n_layers {
        for e in 0..cfg.n_experts {
            table.push(read(l, e)?);
        }
    }
    Ok(table)
}
