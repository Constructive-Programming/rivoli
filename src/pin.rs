//! The resident weight set: places every weight the forward pass reads into the
//! [`DeviceTier`] once at startup and resolves each to a raw device pointer, so
//! per-token decode never touches the host for weights (PLAN.md D1/P3).
//!
//! Always resident (read every token, ~11 GB): the per-layer norms, the full
//! attention stack (q_a/q_b/kv_a/kv_b/o_proj + their layernorms), the dense-layer
//! MLPs, and — for MoE layers — the router gate, its bias, and the shared expert.
//! Routed experts are pinned hottest-first from the `.coli_usage` ranking until
//! the tier's expert budget is exhausted (~90–95% hit at colibri's numbers); the
//! cold tail is fetched on demand into a small pool of reused device slots (never
//! a per-token allocation — that is the amdgpu-GTT-wedge trigger, PLAN.md #6).
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
    /// `experts[sparse_idx][expert]` — resident routed experts (None = cold).
    /// `sparse_idx = layer - dense_layers`.
    experts: Vec<Vec<Option<Mlp>>>,
    /// (layer,expert) → rank in the `.coli_usage` ranking (0 = hottest); absent
    /// means colibri never selected it. The hit-rate-gap diagnostic.
    rank: std::collections::HashMap<(u16, u16), u32>,
    ranked_len: usize,
    /// Reused device slots for cold routed experts: one projection each, sized to
    /// hold O_DIRECT aligned supersets of its packed + scale tensors. Filled by the
    /// io_uring streamer on a miss (no per-token allocation). top_k slots = the
    /// worst case of one layer's misses.
    cold_gate: Vec<VmmBuf>,
    cold_up: Vec<VmmBuf>,
    cold_down: Vec<VmmBuf>,
    cold_next: usize,
    /// io_uring O_DIRECT reader — a layer's cold reads submit as one queue-depth
    /// batch straight into the slots above (folds the old mmap-warm + memcpy).
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

/// Bytes one routed expert (gate+up+down int4 + scales) occupies — the cold-slot
/// and pin-budget accounting unit.
fn expert_bytes(cfg: &ModelConfig) -> usize {
    let rb_h = cfg.hidden.div_ceil(2);
    let rb_i = cfg.moe_inter.div_ceil(2);
    // gate+up: [moe_inter, hidden]; down: [hidden, moe_inter]; + f32 scales.
    2 * cfg.moe_inter * rb_h + cfg.hidden * rb_i + (2 * cfg.moe_inter + cfg.hidden) * 4
}

impl<'a> Pin<'a> {
    /// Build the resident set. `capacity` is the tier size (auto-discovered).
    pub fn build(
        snap: &'a Snapshot,
        cfg: &'a ModelConfig,
        usage: &Usage,
        capacity: usize,
    ) -> Result<Self> {
        let (free, _total) = mem_info()?;
        ensure!(
            capacity < free,
            "pin capacity {capacity} >= free device memory {free}"
        );
        let mut tier = DeviceTier::new(capacity)?;

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

        // Pin routed experts hottest-first until the remaining budget runs out.
        let sparse = cfg.n_layers - cfg.dense_layers;
        let mut experts: Vec<Vec<Option<Mlp>>> =
            (0..sparse).map(|_| vec![None; cfg.n_experts]).collect();
        let ebytes = expert_bytes(cfg);
        // Reserve the cold pool (top_k slots) before spending the rest on the pin.
        let cold_reserve = cfg.top_k * ebytes;
        let budget = capacity.saturating_sub(tier.used() + cold_reserve);
        // Rank each (layer,expert) by its .coli_usage position (0 = hottest), so
        // decode can report the rank of missed experts (routing-vs-breadth diag).
        let ranked = usage.ranked();
        let mut rank = std::collections::HashMap::with_capacity(ranked.len());
        for (i, &((l, e), _)) in ranked.iter().enumerate() {
            rank.insert((l, e), i as u32);
        }
        let mut pinned = 0usize;
        for &((layer, expert), _count) in &ranked {
            let (layer, expert) = (layer as usize, expert as usize);
            if layer < cfg.dense_layers || layer >= cfg.n_layers || expert >= cfg.n_experts {
                continue; // stale ranking entry
            }
            if (pinned + 1) * ebytes > budget {
                break; // budget exhausted
            }
            let base = format!("model.layers.{layer}.mlp.experts.{expert}");
            let m = place_mlp(&mut tier, snap, &base, cfg.hidden, cfg.moe_inter)?;
            experts[layer - cfg.dense_layers][expert] = Some(m);
            pinned += 1;
        }

        // Cold pool: top_k reusable slots per projection. Each slot holds an
        // O_DIRECT aligned superset of the packed tensor, then of the scale tensor
        // (each `slot_span` — block-aligned start + room for the straddle pad), so
        // the io_uring reads land directly in place and the descriptor points at
        // the sub-block offsets.
        let rb_h = cfg.hidden.div_ceil(2);
        let rb_i = cfg.moe_inter.div_ceil(2);
        let gate_slot = slot_span(cfg.moe_inter * rb_h) + slot_span(cfg.moe_inter * 4);
        let down_slot = slot_span(cfg.hidden * rb_i) + slot_span(cfg.hidden * 4);
        let mut cold_gate = Vec::with_capacity(cfg.top_k);
        let mut cold_up = Vec::with_capacity(cfg.top_k);
        let mut cold_down = Vec::with_capacity(cfg.top_k);
        for _ in 0..cfg.top_k {
            cold_gate.push(VmmBuf::new(gate_slot)?);
            cold_up.push(VmmBuf::new(gate_slot)?);
            cold_down.push(VmmBuf::new(down_slot)?);
        }
        // Ring sized for a layer's worst case: top_k misses × 3 projections × 2
        // tensors, with margin.
        let stream = Streamer::new((cfg.top_k * 6).next_power_of_two() as u32 * 2)?;

        Ok(Self {
            snap,
            cfg,
            tier,
            embed,
            lm_head,
            final_norm,
            layers,
            moe_bias,
            experts,
            ranked_len: ranked.len(),
            rank,
            cold_gate,
            cold_up,
            cold_down,
            cold_next: 0,
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

    /// Rank of a routed expert in the `.coli_usage` ranking (0 = hottest); None
    /// means colibri never selected it — a routing-divergence signal.
    pub fn expert_rank(&self, layer: usize, expert: usize) -> Option<u32> {
        self.rank.get(&(layer as u16, expert as u16)).copied()
    }

    /// Total number of ranked (layer,expert) pairs in the usage file.
    pub fn ranked_len(&self) -> usize {
        self.ranked_len
    }

    pub fn pinned_experts(&self) -> usize {
        self.experts
            .iter()
            .flat_map(|l| l.iter())
            .filter(|e| e.is_some())
            .count()
    }

    /// Whether a routed expert is resident (a hit) — for the hit-rate diagnostic.
    pub fn is_resident(&self, layer: usize, expert: usize) -> bool {
        self.experts[layer - self.cfg.dense_layers][expert].is_some()
    }

    /// Resolve a MoE layer's `sel` routed experts to their `Mlp` descriptors,
    /// batching every cold miss through the io_uring O_DIRECT streamer: submit ALL
    /// of the layer's cold reads at once (queue depth → full NVMe bandwidth),
    /// straight into the VMM slots, join ONCE, then build the descriptors. Resident
    /// hits contribute their pinned pointers; misses come from reused cold slots,
    /// valid until the next call. Replaces the old mmap-warm + per-expert copy.
    pub fn resolve_layer(&mut self, layer: usize, sel: &[usize]) -> Result<Vec<Mlp>> {
        self.cold_next = 0;
        let (cfg, snap) = (self.cfg, self.snap);
        let sparse = layer - cfg.dense_layers;
        enum R {
            Hit(Mlp),
            Cold {
                slot: usize,
                g: (usize, usize),
                u: (usize, usize),
                d: (usize, usize),
            },
        }
        // Phase 1: resident hit, or assign a slot and QUEUE the miss's 6 reads.
        let mut plan = Vec::with_capacity(sel.len());
        for &e in sel {
            if let Some(m) = self.experts[sparse][e] {
                self.hits += 1;
                plan.push(R::Hit(m));
                continue;
            }
            self.misses += 1;
            ensure!(
                self.cold_next < cfg.top_k,
                "cold-slot pool exhausted ({} slots) — >top_k misses in a layer",
                cfg.top_k
            );
            let slot = self.cold_next;
            self.cold_next += 1;
            let base = format!("model.layers.{layer}.mlp.experts.{e}");
            // self.stream and self.cold_* are distinct fields → disjoint &mut.
            let g = queue_proj(
                &mut self.stream,
                snap,
                &mut self.cold_gate[slot],
                &format!("{base}.gate_proj"),
            )?;
            let u = queue_proj(
                &mut self.stream,
                snap,
                &mut self.cold_up[slot],
                &format!("{base}.up_proj"),
            )?;
            let d = queue_proj(
                &mut self.stream,
                snap,
                &mut self.cold_down[slot],
                &format!("{base}.down_proj"),
            )?;
            plan.push(R::Cold { slot, g, u, d });
        }
        // Phase 2: ONE join for the whole layer's cold reads.
        self.stream.drain()?;
        // Phase 3: build descriptors (slots now filled). gate/up are [moe_inter,
        // hidden]; down is [hidden, moe_inter].
        let (gate_o, gate_i) = (cfg.moe_inter, cfg.hidden);
        let (down_o, down_i) = (cfg.hidden, cfg.moe_inter);
        let mk = |buf: &VmmBuf, off: (usize, usize), o_dim: usize, i_dim: usize| Weight {
            // SAFETY: off within the slot; the reads that filled them have joined.
            packed: unsafe { buf.ptr().add(off.0) },
            scale: unsafe { buf.ptr().add(off.1) } as *const f32,
            o_dim,
            i_dim,
        };
        let mut out = Vec::with_capacity(sel.len());
        for r in &plan {
            match r {
                R::Hit(m) => out.push(*m),
                R::Cold { slot, g, u, d } => out.push(Mlp {
                    gate: mk(&self.cold_gate[*slot], *g, gate_o, gate_i),
                    up: mk(&self.cold_up[*slot], *u, gate_o, gate_i),
                    down: mk(&self.cold_down[*slot], *d, down_o, down_i),
                }),
            }
        }
        Ok(out)
    }
}

/// Queue one projection's packed + scale O_DIRECT reads into `slot`: packed in the
/// first `slot_span`, scale in the next. Returns the sub-block offsets where the
/// useful bytes land within the slot: `(packed_off, scale_off)`.
fn queue_proj(
    stream: &mut Streamer,
    snap: &Snapshot,
    slot: &mut VmmBuf,
    proj: &str,
) -> Result<(usize, usize)> {
    let base = slot.ptr_mut();
    let (pfd, pb, plen) = snap
        .read_spec(&format!("{proj}.weight"))
        .with_context(|| format!("cold read_spec {proj}.weight"))?;
    // SAFETY: base is VMM (block-aligned); slot sized >= slot_span(plen)+slot_span(slen).
    let poff = unsafe { stream.queue(pfd, pb, plen, base)? };
    let sreg = slot_span(plen); // block-aligned scale region within the slot
    let (sfd, sb, slen) = snap
        .read_spec(&format!("{proj}.weight.qs"))
        .with_context(|| format!("cold read_spec {proj}.weight.qs"))?;
    // SAFETY: base+sreg is block-aligned and within the slot's second region.
    let soff = unsafe { stream.queue(sfd, sb, slen, base.add(sreg))? };
    Ok((poff, sreg + soff))
}
