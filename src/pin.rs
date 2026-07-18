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

use crate::device::{DeviceTier, VmmBuf, mem_info};
use crate::model::ModelConfig;
use crate::snapshot::Snapshot;
use crate::stream::{Streamer, slot_span};
use crate::usage::Usage;
use anyhow::{Context, Result, ensure};

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
}

/// The resident weight set + cold-expert streaming pool. Borrows the snapshot for
/// its lifetime (cold fetches read the mmap in place).
pub struct Pin<'a> {
    snap: &'a Snapshot,
    cfg: &'a ModelConfig,
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
    /// Adaptive routed tier: (layer,expert) -> slot, LRU-evicted. Replaces the
    /// static frequency pin — online priming that keeps this run's hot experts.
    lru: Lru,
    /// io_uring O_DIRECT reader — a layer's cold misses submit as one queue-depth
    /// batch straight into their LRU slots (folds the old mmap-warm + memcpy).
    stream: Streamer,
    /// Stats over the run.
    pub hits: u64,
    pub misses: u64,
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

/// The routed-expert LRU: maps `(layer,expert)` keys to pool slot indices, evicting
/// the least-recently-used when full. Exact LRU via a tick clock + a BTreeMap
/// ordering (O(log N) touch/evict — trivial next to the ~19 MB NVMe read a miss
/// triggers). `off[slot]` caches each resident expert's 6 sub-block offsets.
struct Lru {
    slot_of: std::collections::HashMap<u32, usize>,
    key_of: Vec<Option<u32>>,
    off: Vec<[usize; 6]>, // [gate_p, gate_s, up_p, up_s, down_p, down_s]
    order: std::collections::BTreeMap<u64, usize>, // tick -> slot (front = LRU)
    at: Vec<u64>,         // slot -> its current tick
    clock: u64,
    free: Vec<usize>,
}

impl Lru {
    fn new(n: usize) -> Self {
        Self {
            slot_of: std::collections::HashMap::with_capacity(n),
            key_of: vec![None; n],
            off: vec![[0; 6]; n],
            order: std::collections::BTreeMap::new(),
            at: vec![0; n],
            clock: 0,
            free: (0..n).rev().collect(), // pop() hands out 0,1,2,...
        }
    }

    /// Mark `slot` most-recently-used.
    fn touch(&mut self, slot: usize) {
        self.order.remove(&self.at[slot]);
        self.clock += 1;
        self.at[slot] = self.clock;
        self.order.insert(self.clock, slot);
    }

    /// Resident slot for `key`, touched — or `None` (miss).
    fn get(&mut self, key: u32) -> Option<usize> {
        let slot = *self.slot_of.get(&key)?;
        self.touch(slot);
        Some(slot)
    }

    /// Allocate a slot for a NEW `key` (miss): a free slot, else evict the LRU. The
    /// caller then fills the slot's buffers and sets `off[slot]`.
    fn alloc(&mut self, key: u32) -> Result<usize> {
        let slot = if let Some(s) = self.free.pop() {
            s
        } else {
            // Full pool → evict the front of `order` (least-recently-used). Non-empty
            // by construction (n_slots ≥ top_k ≥ 1), but fail loud rather than panic.
            let (&t, &victim) = self
                .order
                .iter()
                .next()
                .context("LRU eviction on an empty ordering (n_slots == 0?)")?;
            self.order.remove(&t);
            if let Some(old) = self.key_of[victim] {
                self.slot_of.remove(&old);
            }
            victim
        };
        self.key_of[slot] = Some(key);
        self.slot_of.insert(key, slot);
        self.touch(slot);
        Ok(slot)
    }
}

impl<'a> Pin<'a> {
    /// Build the resident set. `capacity` is the tier size (auto-discovered).
    pub fn build(
        snap: &'a Snapshot,
        cfg: &'a ModelConfig,
        usage: &Usage,
        capacity: usize,
        pre_seed: bool,
    ) -> Result<Self> {
        let (free, _total) = mem_info()?;
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
        let resident = resident_bytes(cfg);
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
                let bias = snap.require(&format!("{lb}.mlp.gate.e_score_correction_bias"))?;
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
            });
        }

        // The routed tier is an adaptive LRU, not a static frequency pin: N reused
        // slots stream experts in on demand and keep this run's actually-hot ones
        // (online priming). Each slot holds a projection's O_DIRECT aligned superset
        // (packed then scale, each `slot_span` — block-aligned + straddle pad — so
        // reads land in place and descriptors point at the sub-block offsets). Size
        // to the budget left after the always-resident set.
        let (_, expert_slot) = slot_geom(cfg);
        let budget = capacity.saturating_sub(tier_cap);
        let n_slots = (budget / expert_slot).max(cfg.top_k);
        let mut lru_pool = VmmBuf::new(n_slots * expert_slot)?; // ONE slab
        tracing::info!(
            "routed LRU: {n_slots} slots ({:.1} GiB) + {:.1} GiB always-resident",
            (n_slots * expert_slot) as f64 / (1u64 << 30) as f64,
            tier.used() as f64 / (1u64 << 30) as f64,
        );
        let mut lru = Lru::new(n_slots);
        // Ring sized for one layer's worst case: top_k misses x 3 proj x 2 tensors.
        let ring = (cfg.top_k * 6).next_power_of_two() * 2;
        let mut stream = Streamer::new(ring as u32)?;
        // Optional warm start (`--pre-seed`): seed the pool with the hottest experts
        // from .coli_usage so the first tokens hit at colibri's rate while the LRU
        // adapts. Worth ~+6.8pt on the first tokens but transient (the LRU warms up
        // in a few tokens regardless) and it dominates build time (~23s), so it is
        // OFF by default — enable it only for slow disks. Bounded by the pool size;
        // drained in queue-depth batches.
        if pre_seed {
            let pool = lru_pool.ptr_mut();
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
                let slot = lru.alloc(expert_key(l, e))?;
                let base = format!("model.layers.{l}.mlp.experts.{e}");
                lru.off[slot] = stream_expert(&mut stream, snap, cfg, pool, slot, &base)?;
                n += 1;
                if n.is_multiple_of(batch) {
                    stream.drain()?;
                }
            }
            stream.drain()?;
            tracing::info!("LRU seeded {n} experts from .coli_usage (warm start)");
        }

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
            lru,
            stream,
            hits: 0,
            misses: 0,
        })
    }

    /// Device bytes used by the resident set.
    pub fn used(&self) -> usize {
        self.tier.used()
    }

    /// Host router correction bias for a MoE `layer` (len n_experts).
    pub fn moe_bias(&self, layer: usize) -> &[f32] {
        &self.moe_bias[layer - self.cfg.dense_layers]
    }

    /// Resolve a MoE layer's `sel` routed experts to `Mlp` descriptors via the
    /// adaptive LRU: a hit returns its resident slot's pointers (no I/O); a miss
    /// evicts the coldest slot and streams the expert in. All of the layer's misses
    /// submit as ONE queue-depth io_uring O_DIRECT batch, joined once.
    pub fn resolve_layer(&mut self, layer: usize, sel: &[usize]) -> Result<Vec<Mlp>> {
        let (cfg, snap) = (self.cfg, self.snap);
        // Phase 1a: touch EVERY hit first. Doing this before any miss's `alloc()`
        // bumps each hit's LRU tick above the eviction candidates, so a later miss
        // cannot evict a same-layer would-be hit out from under itself (which would
        // force a needless re-stream and understate the hit rate). `None` marks a
        // slot still to be filled in phase 1b.
        let mut slots: Vec<Option<usize>> = Vec::with_capacity(sel.len());
        for &e in sel {
            match self.lru.get(expert_key(layer, e)) {
                Some(slot) => {
                    self.hits += 1;
                    slots.push(Some(slot));
                }
                None => slots.push(None),
            }
        }
        // Phase 1b: allocate + QUEUE the misses, now that all hits are protected.
        for (i, &e) in sel.iter().enumerate() {
            if slots[i].is_some() {
                continue;
            }
            self.misses += 1;
            let slot = self.lru.alloc(expert_key(layer, e))?;
            let base = format!("model.layers.{layer}.mlp.experts.{e}");
            // ptr_mut() yields a raw ptr, so no borrow of lru_pool is held across
            // the &mut self.stream reads.
            let pool = self.lru_pool.ptr_mut();
            self.lru.off[slot] = stream_expert(&mut self.stream, snap, cfg, pool, slot, &base)?;
            slots[i] = Some(slot);
        }
        // Phase 2: ONE join for the whole layer's cold reads.
        self.stream.drain()?;
        // Phase 3: build descriptors from each slot's stored sub-block offsets.
        // gate/up are [moe_inter, hidden]; down is [hidden, moe_inter].
        let (gate_o, gate_i) = (cfg.moe_inter, cfg.hidden);
        let (down_o, down_i) = (cfg.hidden, cfg.moe_inter);
        let pool = self.lru_pool.ptr();
        let es = self.expert_slot;
        let mut out = Vec::with_capacity(sel.len());
        for &slot in &slots {
            // Every entry was filled in phase 1a/1b; a `None` here is an internal
            // invariant break, not a data condition — fail loud rather than panic.
            let slot = slot.context("resolve_layer: unresolved expert slot")?;
            let o = self.lru.off[slot];
            // SAFETY: slot base within the slab; reads that filled it have joined.
            let b = unsafe { pool.add(slot * es) };
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
        Ok(out)
    }
}

/// Queue an expert's 6 O_DIRECT reads into pool slot `slot` (regions gate|up|down
/// at `slot*expert_slot`), returning the 6 slot-relative sub-block offsets
/// `[gate_p, gate_s, up_p, up_s, down_p, down_s]`. Shared by the decode-miss path
/// and the build-time warm-start seed.
/// One routed expert's pool-slot geometry, derived from `cfg`: `(gate_slot,
/// expert_slot)`. A slot is laid out gate|up|down; `gate` and `up` each span
/// `gate_slot` (packed superset + scale superset), `down` the rest, and the whole
/// expert occupies `expert_slot`. The single source of truth for both the pool
/// sizing in [`Pin::build`] and the per-region offsets in [`stream_expert`].
fn slot_geom(cfg: &ModelConfig) -> (usize, usize) {
    let rb_h = cfg.hidden.div_ceil(2);
    let rb_i = cfg.moe_inter.div_ceil(2);
    let gate_slot = slot_span(cfg.moe_inter * rb_h) + slot_span(cfg.moe_inter * 4);
    let down_slot = slot_span(cfg.hidden * rb_i) + slot_span(cfg.hidden * 4);
    (gate_slot, 2 * gate_slot + down_slot) // gate | up | down, one slot
}

fn stream_expert(
    stream: &mut Streamer,
    snap: &Snapshot,
    cfg: &ModelConfig,
    pool: *mut u8,
    slot: usize,
    base: &str,
) -> Result<[usize; 6]> {
    let (gate_slot, expert_slot) = slot_geom(cfg);
    // cfg-expected packed/scale byte lengths per projection — the SAME budget the
    // slot regions (`gate_slot`/`down_slot` in `build`) were sized from. The cold
    // io_uring path reads a RAW file length (`read_spec`) with no dim, so cross-check
    // it here exactly as the resident path does via `Snapshot::int4`: a config/
    // snapshot dim mismatch would otherwise overrun one projection's region into the
    // next (or the last slot's `down` past the slab) and stream wrong-expert bytes.
    let rb_h = cfg.hidden.div_ceil(2);
    let rb_i = cfg.moe_inter.div_ceil(2);
    // gate/up: [moe_inter, hidden]; down: [hidden, moe_inter]; one f32 scale per row.
    let (gate_p, gate_s) = (cfg.moe_inter * rb_h, cfg.moe_inter * 4);
    let (down_p, down_s) = (cfg.hidden * rb_i, cfg.hidden * 4);
    let o = slot * expert_slot;
    // SAFETY: o + 2*gate_slot + down region <= pool len (slot < n_slots).
    let (gp, up, dp) = unsafe {
        (
            pool.add(o),
            pool.add(o + gate_slot),
            pool.add(o + 2 * gate_slot),
        )
    };
    // SAFETY: each region is block-aligned and sized for its two supersets.
    let (pg, sg) = unsafe {
        queue_proj(
            stream,
            snap,
            gp,
            &format!("{base}.gate_proj"),
            gate_p,
            gate_s,
        )?
    };
    let (pu, su) =
        unsafe { queue_proj(stream, snap, up, &format!("{base}.up_proj"), gate_p, gate_s)? };
    let (pd, sd) = unsafe {
        queue_proj(
            stream,
            snap,
            dp,
            &format!("{base}.down_proj"),
            down_p,
            down_s,
        )?
    };
    Ok([
        pg,
        sg,
        gate_slot + pu,
        gate_slot + su,
        2 * gate_slot + pd,
        2 * gate_slot + sd,
    ])
}

/// Queue one projection's packed + scale O_DIRECT reads into `region` (a
/// block-aligned pointer owning `>= slot_span(plen)+slot_span(slen)` bytes):
/// packed in the first `slot_span`, scale in the next. Returns the sub-block
/// offsets RELATIVE to `region` where the useful bytes land.
///
/// `exp_packed`/`exp_scale` are the cfg-derived byte lengths the region was sized
/// for; the RAW `read_spec` lengths are `ensure!`d to match them (fail loud on a
/// config/snapshot dim mismatch, like the resident `Snapshot::int4` path).
///
/// # Safety
/// `region` must be block-aligned and own the projection's two supersets
/// (`slot_span(exp_packed) + slot_span(exp_scale)` bytes).
unsafe fn queue_proj(
    stream: &mut Streamer,
    snap: &Snapshot,
    region: *mut u8,
    proj: &str,
    exp_packed: usize,
    exp_scale: usize,
) -> Result<(usize, usize)> {
    let base = region;
    let (pfd, pb, plen) = snap
        .read_spec(&format!("{proj}.weight"))
        .with_context(|| format!("cold read_spec {proj}.weight"))?;
    ensure!(
        plen == exp_packed,
        "cold {proj}.weight: {plen} packed bytes, cfg expects {exp_packed} \
         (config/snapshot dim mismatch)"
    );
    // SAFETY: base is VMM (block-aligned); slot sized >= slot_span(plen)+slot_span(slen).
    let poff = unsafe { stream.queue(pfd, pb, plen, base)? };
    let sreg = slot_span(plen); // block-aligned scale region within the slot
    let (sfd, sb, slen) = snap
        .read_spec(&format!("{proj}.weight.qs"))
        .with_context(|| format!("cold read_spec {proj}.weight.qs"))?;
    ensure!(
        slen == exp_scale,
        "cold {proj}.weight.qs: {slen} scale bytes, cfg expects {exp_scale} \
         (config/snapshot dim mismatch)"
    );
    // SAFETY: base+sreg is block-aligned and within the slot's second region.
    let soff = unsafe { stream.queue(sfd, sb, slen, base.add(sreg))? };
    Ok((poff, sreg + soff))
}
