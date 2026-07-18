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
use crate::usage::Usage;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;

/// Drop the page-cache copy of a just-placed mmap range (`MADV_DONTNEED`) so the
/// resident pin doesn't pollute the cache the cold-expert set needs. Only whole
/// pages inside the range are dropped; best-effort — a failure just leaves the
/// duplicate, never wrong (the pages are clean file-backed, re-faulted on access).
fn drop_pages(bytes: &[u8]) {
    const PAGE: usize = 4096;
    let start = bytes.as_ptr() as usize;
    let end = start + bytes.len();
    let a = start.div_ceil(PAGE) * PAGE; // round up to a page
    let b = (end / PAGE) * PAGE; // round down to a page
    if b > a {
        // SAFETY: [a,b) is a page-aligned subrange of the live snapshot mmap;
        // MADV_DONTNEED discards clean file pages, never corrupts.
        unsafe { libc::madvise(a as *mut libc::c_void, b - a, libc::MADV_DONTNEED) };
    }
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
    /// Reused device slots for cold routed experts: one `Mlp` worth of bytes
    /// each, filled via `copy_in` on a miss (no per-token allocation). Sized to
    /// the worst case of one layer's misses (top_k slots).
    cold_gate: Vec<VmmBuf>,
    cold_up: Vec<VmmBuf>,
    cold_down: Vec<VmmBuf>,
    cold_next: usize,
    /// Stats over the run.
    pub hits: u64,
    pub misses: u64,
}

/// Place a norm/f32 tensor (O-length) into the tier, returning its device ptr.
/// Drops the source page-cache copy after the H2D (see [`drop_pages`]).
fn place_f32(tier: &mut DeviceTier, snap: &Snapshot, name: &str) -> Result<*const f32> {
    let bytes = snap.require(name)?;
    let p = tier.place(bytes)? as *const f32;
    drop_pages(bytes);
    Ok(p)
}

/// Place an int4 weight (`<name>.weight` + `.qs`) into the tier.
fn place_i4(tier: &mut DeviceTier, snap: &Snapshot, name: &str, i_dim: usize) -> Result<Weight> {
    let m = snap.int4(name, i_dim)?;
    let packed = tier.place(m.packed)?;
    drop_pages(m.packed);
    let scale = tier.place(m.scale)? as *const f32;
    drop_pages(m.scale);
    Ok(Weight {
        packed,
        scale,
        o_dim: m.o_dim,
        i_dim: m.i_dim,
    })
}

/// Place an int8 weight into the tier.
fn place_i8(tier: &mut DeviceTier, snap: &Snapshot, name: &str, i_dim: usize) -> Result<Weight8> {
    let m = snap.int8(name, i_dim)?;
    let packed = tier.place(m.packed)?;
    drop_pages(m.packed);
    let scale = tier.place(m.scale)? as *const f32;
    drop_pages(m.scale);
    Ok(Weight8 {
        packed,
        scale,
        o_dim: m.o_dim,
        i_dim: m.i_dim,
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

        // Cold pool: top_k reusable slots, one Mlp's three tensors each.
        let rb_h = cfg.hidden.div_ceil(2);
        let rb_i = cfg.moe_inter.div_ceil(2);
        let (gate_bytes, down_bytes) = (cfg.moe_inter * rb_h, cfg.hidden * rb_i);
        let mut cold_gate = Vec::with_capacity(cfg.top_k);
        let mut cold_up = Vec::with_capacity(cfg.top_k);
        let mut cold_down = Vec::with_capacity(cfg.top_k);
        for _ in 0..cfg.top_k {
            // gate/up carry their scale inline after the packed nibbles; keep
            // packed and scale in one buffer per projection to halve slot count.
            cold_gate.push(VmmBuf::new(gate_bytes + cfg.moe_inter * 4)?);
            cold_up.push(VmmBuf::new(gate_bytes + cfg.moe_inter * 4)?);
            cold_down.push(VmmBuf::new(down_bytes + cfg.hidden * 4)?);
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
            experts,
            ranked_len: ranked.len(),
            rank,
            cold_gate,
            cold_up,
            cold_down,
            cold_next: 0,
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

    /// Fault a layer's COLD experts' pages into the cache CONCURRENTLY (multiple
    /// threads → NVMe queue depth) right before the serial `copy_in`s, so the
    /// per-layer NVMe reads overlap instead of stalling one-at-a-time (QD1). The
    /// subsequent `expert()` copies then read a warm cache. Best-effort.
    pub fn warm_cold(&self, layer: usize, experts: &[usize]) {
        let mut ranges = Vec::new();
        for &e in experts {
            let _ = self.cold_warm_ranges(layer, e, &mut ranges);
        }
        if ranges.len() < 2 {
            for &(addr, len) in &ranges {
                touch(addr, len);
            }
            return;
        }
        // Dispatch the page-faulting to rayon's persistent global pool instead of
        // spawning OS threads per call (~600/token). Chunk so at most ~8 chunks are
        // produced (matching the old `.min(8)` fan-out); `par_chunks(..).for_each`
        // uses the lazily-initialized global workers and blocks until every page is
        // faulted — the synchronous barrier the serial `copy_in` depends on.
        let nchunks = ranges.len().min(8);
        let per = ranges.len().div_ceil(nchunks);
        ranges.par_chunks(per).for_each(|chunk| {
            for &(addr, len) in chunk {
                touch(addr, len);
            }
        });
    }

    /// Append the mmap byte ranges of a COLD routed expert's packed weights to
    /// `out`, for background page-cache warming. A resident (pinned) expert
    /// appends nothing — it is already on device, no NVMe read to overlap.
    pub fn cold_warm_ranges(
        &self,
        layer: usize,
        expert: usize,
        out: &mut Vec<(usize, usize)>,
    ) -> Result<()> {
        if self.experts[layer - self.cfg.dense_layers][expert].is_some() {
            return Ok(());
        }
        let base = format!("model.layers.{layer}.mlp.experts.{expert}");
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let w = self.snap.require(&format!("{base}.{proj}.weight"))?;
            out.push((w.as_ptr() as usize, w.len()));
        }
        Ok(())
    }

    pub fn pinned_experts(&self) -> usize {
        self.experts
            .iter()
            .flat_map(|l| l.iter())
            .filter(|e| e.is_some())
            .count()
    }

    /// Reset the cold-slot ring at the start of a layer's MoE block (the previous
    /// layer's misses have been consumed and joined).
    pub fn begin_layer(&mut self) {
        self.cold_next = 0;
    }

    /// Resolve a routed expert's `Mlp` for `layer`: a resident hit returns its
    /// pinned pointers; a miss fetches the three int4 tensors from the snapshot
    /// mmap into the next cold slot (reused, no allocation) and returns those.
    /// The returned pointers stay valid until the next `begin_layer`.
    pub fn expert(&mut self, layer: usize, expert: usize) -> Result<Mlp> {
        if let Some(m) = self.experts[layer - self.cfg.dense_layers][expert] {
            self.hits += 1;
            return Ok(m);
        }
        self.misses += 1;
        ensure!(
            self.cold_next < self.cfg.top_k,
            "cold-slot pool exhausted ({} slots) — more than top_k misses in a layer",
            self.cfg.top_k
        );
        let slot = self.cold_next;
        self.cold_next += 1;
        let base = format!("model.layers.{layer}.mlp.experts.{expert}");
        let cfg = self.cfg;
        let g = self.snap.int4(&format!("{base}.gate_proj"), cfg.hidden)?;
        let u = self.snap.int4(&format!("{base}.up_proj"), cfg.hidden)?;
        let d = self
            .snap
            .int4(&format!("{base}.down_proj"), cfg.moe_inter)?;
        // Fill the slot buffers: packed nibbles then f32 scale bytes, contiguous.
        fill_slot(&mut self.cold_gate[slot], g.packed, g.scale).context("cold gate")?;
        fill_slot(&mut self.cold_up[slot], u.packed, u.scale).context("cold up")?;
        fill_slot(&mut self.cold_down[slot], d.packed, d.scale).context("cold down")?;
        let gp = self.cold_gate[slot].ptr();
        let up = self.cold_up[slot].ptr();
        let dp = self.cold_down[slot].ptr();
        Ok(Mlp {
            gate: Weight {
                packed: gp,
                scale: unsafe { gp.add(g.packed.len()) } as *const f32,
                o_dim: g.o_dim,
                i_dim: g.i_dim,
            },
            up: Weight {
                packed: up,
                scale: unsafe { up.add(u.packed.len()) } as *const f32,
                o_dim: u.o_dim,
                i_dim: u.i_dim,
            },
            down: Weight {
                packed: dp,
                scale: unsafe { dp.add(d.packed.len()) } as *const f32,
                o_dim: d.o_dim,
                i_dim: d.i_dim,
            },
        })
    }
}

/// Fault a read-only mmap range into the page cache: one byte per 4 KiB page.
/// The `black_box` keeps the compiler from eliding the loads.
fn touch(addr: usize, len: usize) {
    let p = addr as *const u8;
    let mut off = 0usize;
    let mut sink = 0u8;
    while off < len {
        // SAFETY: [addr,addr+len) is a live read-only snapshot mmap range.
        sink = sink.wrapping_add(unsafe { *p.add(off) });
        off += 4096;
    }
    std::hint::black_box(sink);
}

/// Copy packed nibbles then scale bytes into one contiguous device slot.
fn fill_slot(buf: &mut VmmBuf, packed: &[u8], scale: &[u8]) -> Result<()> {
    ensure!(
        packed.len() + scale.len() == buf.len(),
        "cold slot size {} != packed {} + scale {}",
        buf.len(),
        packed.len(),
        scale.len()
    );
    // Host memcpy straight into device-local memory (no H2D copy, no sync).
    buf.write_at(0, packed)?;
    buf.write_at(packed.len(), scale)?;
    Ok(())
}
