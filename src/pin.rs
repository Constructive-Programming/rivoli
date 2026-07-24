//! The resident weight set: places every weight the forward pass reads into the
//! [`DeviceTier`] once at startup and resolves each to a raw device pointer, so
//! per-token decode never touches the host for weights.
//!
//! Always resident (read every token): the per-layer norms, the full attention
//! stack (q_a/q_b/kv_a/kv_b/o_proj + their layernorms, all fp8-e4m3 block-scaled),
//! the dense-layer MLPs (fp8), and — for MoE layers — the router gate (f32) and the
//! VQ-int3 shared expert. The int8 embed/lm_head + f32 final norm are global. The
//! footprint is computed from `cfg` at build ([`resident_bytes`]) so the tier is
//! sized to what it holds and the rest of the device budget grows the routed pool.
//!
//! The 256 routed experts/layer are served by an adaptive pool ([`cache`] policy +
//! one VMM slab): a hit reuses the resident slot; a miss evicts the coldest and
//! streams the expert in via io_uring O_DIRECT (`.vq3` block = one aligned read).
//!
//! `rocm`-only: without a device there is nothing to pin.
#![cfg(feature = "rocm")]

use crate::asyncfetch::{AsyncFetch, ReadSpec};
use crate::cache;
use crate::device::{DeviceTier, VmmBuf};
use crate::format::{Dtype, FormatMeta, Safetensors, Vq3Set, load_codebooks};
use crate::gpustream::Signal;
use crate::model::ModelConfig;
use crate::quant::{VQ_DIM, VQ_K, vq_expert_bytes, vq_expert_layout, vq_proj_bytes, vq_row_bytes};
use crate::stream::{Sqpoll, Streamer, slot_span};
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

/// A resolved VQ-int3 projection: packed 12-bit codebook indices + bf16 group
/// scales (device pointers), decoded against the per-projection codebook. Consumed
/// by `launch_moe_expert_range` via `desc_of_vq`.
#[derive(Clone, Copy)]
pub struct VqWeight {
    pub indices: *const u8,
    pub scales: *const u16,
    pub o_dim: usize,
    pub i_dim: usize,
}

/// A SwiGLU MLP's three VQ-int3 projections, resolved (routed or shared expert).
#[derive(Clone, Copy)]
pub struct MlpVq {
    pub gate: VqWeight,
    pub up: VqWeight,
    pub down: VqWeight,
}

/// One layer's MLP: dense fp8 for the first `dense_layers`, MoE after. The MoE
/// shared expert is VQ-int3 (from the `.vq3` block `n_experts`), folded into the
/// single `launch_moe_expert_range` batch alongside the routed picks.
pub enum LayerMlp {
    Dense(Fp8Mlp),
    Moe {
        gate_w: *const f32, // router gate [n_experts, hidden] (F32), device
        shared: MlpVq,      // VQ-int3 shared expert (always resident)
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
    /// `launch_moe_expert_range`. Each `VQ_K·VQ_DIM` f32.
    codebooks: [*const f32; 3],
    /// The routed-expert pool: ONE device-local VMM slab of `n_slots` expert slots
    /// (slot i at `i * expert_slot`), each one O_DIRECT-aligned `.vq3` block.
    #[allow(dead_code)] // RAII owner of the pool slab; addressed via `lru_pool.ptr()`
    lru_pool: VmmBuf,
    expert_slot: usize,
    /// The `.vq3` streaming source — its O_DIRECT fds back every `moe_table` read.
    /// Held for the run; not read through after `build` populated the table.
    #[allow(dead_code)]
    vq: Vq3Set,
    /// Per-(MoE layer, expert) cold-read spec `(fd, begin, len)`, indexed by
    /// `(layer - dense_layers) * n_experts + expert`. One aligned read per expert.
    moe_table: Vec<(RawFd, usize, usize)>,
    /// The six slot-relative byte offsets `[gate.idx, gate.sc, up.idx, up.sc,
    /// down.idx, down.sc]` of a VQ expert block — identical for every expert (fixed
    /// layout), so stored once here rather than per `moe_table` row.
    vq_off: [usize; 6],
    /// Adaptive routed tier: (layer,expert) -> slot, policy-evicted (`--cache-policy`).
    pool: Pool,
    /// Per-expert async cold-fetch: owns the io_uring demand ring on a reaper thread
    /// and resolves each miss's load [`Signal`] when its bytes land. The expert
    /// stream awaits these; there is no batch join.
    fetch: AsyncFetch,
    /// How many times a batch reused a slot whose read was still outstanding — each
    /// one would have been a silent weight corruption; the async path refuses it.
    pub slot_collisions: u64,
    pub hits: u64,
    pub misses: u64,
    /// Optional access-trace sink (`--trace`): one line per resolved MoE layer — the
    /// `(layer,expert)` keys looked up, in access order. Feeds the offline `replay`
    /// simulator.
    trace: Option<std::io::BufWriter<std::fs::File>>,
}

/// Place an F32 tensor (norms, router gate) into the tier: reserve, copy from the
/// resident mmap into the device-local slab.
fn place_f32(tier: &mut DeviceTier, st: &Safetensors, name: &str) -> Result<*const f32> {
    let (bytes, _) = st.typed(name, Dtype::F32)?;
    let dst = tier.reserve(bytes.len())?;
    // SAFETY: dst owns bytes.len() reserved bytes; f32 LE host == LE device.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
    Ok(dst as *const f32)
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
    let packed = tier.reserve(w.len())?;
    let scale = tier.reserve(sc.len())?;
    // SAFETY: each dst owns its reserved byte count.
    unsafe {
        std::ptr::copy_nonoverlapping(w.as_ptr(), packed, w.len());
        std::ptr::copy_nonoverlapping(sc.as_ptr(), scale, sc.len());
    }
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
    let packed = tier.reserve(w.len())?;
    let scale = tier.reserve(sc.len())?;
    // SAFETY: each dst owns its reserved byte count.
    unsafe {
        std::ptr::copy_nonoverlapping(w.as_ptr(), packed, w.len());
        std::ptr::copy_nonoverlapping(sc.as_ptr(), scale, sc.len());
    }
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

/// Place a VQ shared-expert `block` (gate‖up‖down) resident and resolve it to an
/// `MlpVq`. Its sub-offsets are the routed-expert layout ([`vq_slot_offsets`]).
fn place_vq_shared(
    tier: &mut DeviceTier,
    block: &[u8],
    hidden: usize,
    moe_inter: usize,
) -> Result<MlpVq> {
    let dst = tier.reserve(block.len())?;
    // SAFETY: dst owns block.len() reserved bytes.
    unsafe { std::ptr::copy_nonoverlapping(block.as_ptr(), dst, block.len()) };
    let off = vq_slot_offsets(hidden, moe_inter);
    let vw = |ioff: usize, soff: usize, o_dim: usize, i_dim: usize| VqWeight {
        // SAFETY: offsets lie within the block just copied into the tier.
        indices: unsafe { dst.add(ioff) },
        scales: unsafe { dst.add(soff) } as *const u16,
        o_dim,
        i_dim,
    };
    Ok(MlpVq {
        gate: vw(off[0], off[1], moe_inter, hidden),
        up: vw(off[2], off[3], moe_inter, hidden),
        down: vw(off[4], off[5], hidden, moe_inter),
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

/// Device bytes ONE full layer's DSA indexer weights occupy, mirroring
/// [`place_indexer`]: fp8 `wk`/`wq_b`, f32 `weights_proj` + f32 `k_norm`.
fn indexer_bytes(cfg: &ModelConfig, block: usize) -> usize {
    let fp8 = |o: usize, i: usize| o * i + o.div_ceil(block) * i.div_ceil(block) * 4;
    let hd = cfg.index_head_dim;
    let nh = cfg.index_n_heads;
    fp8(hd, cfg.hidden)                 // wk (fp8)
        + fp8(nh * hd, cfg.q_lora_rank) // wq_b (fp8)
        + nh * cfg.hidden * 4           // weights_proj (bf16→f32)
        + 2 * hd * 4 // k_norm weight + bias (f32)
}

/// Device bytes the always-resident set occupies — everything read every token
/// EXCEPT the routed experts. Summed from `cfg` so the resident tier is sized to
/// what it holds and the rest of the device budget grows the routed pool. Mirrors
/// the placement path: fp8 `[o,i]` = `o·i` packed + `⌈o/block⌉·⌈i/block⌉·4` scale;
/// int8 `[o,i]` = `o·i` + `o·4`; an f32 norm of `n` = `n·4`; plus the 3 codebooks
/// and one VQ shared expert per MoE layer.
fn resident_bytes(cfg: &ModelConfig, block: usize) -> usize {
    let fp8 = |o: usize, i: usize| o * i + o.div_ceil(block) * i.div_ceil(block) * 4;
    let i8 = |o: usize, i: usize| o * i + o * 4;
    let f32n = |n: usize| n * 4;

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
            total += vq_expert_bytes(cfg.hidden, cfg.moe_inter); // VQ shared expert
        }
    }
    total + 3 * VQ_K * VQ_DIM * 4 // 3 per-projection codebooks
}

/// Pack `(layer, expert)` into the pool key. Both must fit in 16 bits — GLM is
/// ≤92 layers × 256 routed experts, comfortably under 2^16.
fn expert_key(layer: usize, expert: usize) -> u32 {
    debug_assert!(
        layer < (1 << 16) && expert < (1 << 16),
        "layer {layer}/expert {expert} exceed the 16-bit pool key packing"
    );
    ((layer as u32) << 16) | expert as u32
}

/// The six slot-relative byte offsets `[gate.idx, gate.sc, up.idx, up.sc, down.idx,
/// down.sc]` of a VQ expert block: per projection, indices then bf16 group scales,
/// gate‖up‖down concatenated. MUST match [`crate::quant::vq_expert`]'s slicing (the
/// `vq_offsets_match_loader` test locks this).
fn vq_slot_offsets(hidden: usize, moe_inter: usize) -> [usize; 6] {
    let dims = vq_expert_layout(hidden, moe_inter);
    let mut off = [0usize; 6];
    let mut base = 0usize;
    for (p, &(o, i)) in dims.iter().enumerate() {
        off[p * 2] = base; // projection indices
        off[p * 2 + 1] = base + o * vq_row_bytes(i); // bf16 group scales
        base += vq_proj_bytes(o, i);
    }
    off
}

/// The routed-expert pool: maps `(layer,expert)` keys to slab slot indices, with
/// eviction delegated to a pluggable `cache::Cache` policy. The policy owns
/// residency + eviction order; the pool owns the key↔slot maps. Format-agnostic —
/// the per-expert geometry lives in [`Pin::moe_table`], not per slot.
struct Pool {
    policy: Box<dyn cache::Cache>,
    slot_of: std::collections::HashMap<u32, usize>,
    #[cfg(debug_assertions)]
    key_of: Vec<Option<u32>>,
    free: Vec<usize>,
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
        })
    }

    /// Resident slot for `key` (a hit, promoted by the policy), or `None` (a miss).
    fn get(&mut self, key: u32) -> Option<usize> {
        if self.policy.get(key) {
            let slot = self.slot_of[&key];
            #[cfg(debug_assertions)]
            debug_assert_eq!(self.key_of[slot], Some(key), "hit slot holds wrong key");
            Some(slot)
        } else {
            None
        }
    }

    /// Allocate a slot for a NEW `key` (miss): reuse the policy-evicted key's slot,
    /// else a free slot. The caller then fills the slot.
    fn alloc(&mut self, key: u32) -> Result<usize> {
        let evicted = self.policy.insert(key);
        let slot = self.reuse(evicted)?;
        self.bind(key, slot);
        Ok(slot)
    }

    fn protect(&mut self, key: u32) {
        self.policy.protect(key);
    }

    fn reuse(&mut self, evicted: Option<u32>) -> Result<usize> {
        match evicted {
            Some(ev) => {
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
        debug_assert!(self.policy.contains(key), "inserted key {key} not resident");
        debug_assert_eq!(
            self.slot_of.len(),
            self.policy.resident_len(),
            "pool/policy residency drift"
        );
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
        want_indexer: bool,
    ) -> Result<Self> {
        // ponytail: no free-memory pre-check — the budget is the user's literal
        // request (--max-mem), so let the device allocation itself OOM/fail.
        // One-time bound for `submit_layer`'s fixed 32-slot scratch.
        ensure!(
            cfg.top_k + cfg.n_shared <= 32,
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
        let vq = Vq3Set::open(
            dir,
            cfg.dense_layers,
            cfg.n_layers,
            cfg.n_experts,
            cfg.hidden,
            cfg.moe_inter,
        )?;

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
        let resident = resident_bytes(cfg, block) + n_full * indexer_bytes(cfg, block);
        let tier_cap = resident + SLACK;
        tracing::info!(
            "computed resident footprint {:.2} GiB (tier {:.2} GiB incl. slack)",
            resident as f64 / (1u64 << 30) as f64,
            tier_cap as f64 / (1u64 << 30) as f64,
        );
        let mut tier = DeviceTier::new(tier_cap)?;

        // Codebooks resident (gate/up/down), passed to launch_moe_expert_range.
        let mut codebooks = [std::ptr::null(); 3];
        for (i, cb) in cbs.iter().enumerate() {
            let bytes = cb.len() * 4;
            let dst = tier.reserve(bytes)?;
            // SAFETY: dst owns bytes just reserved; f32 LE host == LE device.
            unsafe { std::ptr::copy_nonoverlapping(cb.as_ptr() as *const u8, dst, bytes) };
            codebooks[i] = dst as *const f32;
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
                let shared =
                    place_vq_shared(&mut tier, &vq.shared_block(l)?, cfg.hidden, cfg.moe_inter)?;
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

        // Routed pool: each `.vq3` expert is ONE aligned block read into a slot of
        // `expert_slot` bytes. Size the pool to the budget left after the resident set.
        let expert_slot = vq.expert_slot();
        let vq_off = vq_slot_offsets(cfg.hidden, cfg.moe_inter);
        let moe_table = build_moe_table(&vq, cfg)?;
        let budget = capacity.saturating_sub(tier_cap);
        let n_slots = (budget / expert_slot).max(cfg.top_k);
        let lru_pool = VmmBuf::new(n_slots * expert_slot)?; // ONE slab
        tracing::info!(
            "routed pool [{cache_policy}]: {n_slots} slots ({:.1} GiB) + {:.1} GiB always-resident",
            (n_slots * expert_slot) as f64 / (1u64 << 30) as f64,
            tier.used() as f64 / (1u64 << 30) as f64,
        );
        let pool = Pool::new(n_slots, cache_policy, two_q)?;
        // Ring sized for one layer's worst case: top_k demand reads (1/expert). One
        // read per expert (VQ block), so far smaller than int4's six-read coalescing.
        let ring = (cfg.top_k + 4).next_power_of_two();
        ensure!(
            cfg.top_k <= ring,
            "io_uring ring {ring} too small for top_k {}",
            cfg.top_k,
        );
        // Bounce span = the whole expert block (one read).
        let span = slot_span(vq_expert_bytes(cfg.hidden, cfg.moe_inter));
        let fetch = AsyncFetch::new(Streamer::new(ring as u32, span, bounce, Sqpoll::Own)?)?;

        Ok(Self {
            cfg,
            tier,
            embed,
            lm_head,
            final_norm,
            layers,
            moe_bias,
            codebooks,
            lru_pool,
            expert_slot,
            vq,
            moe_table,
            vq_off,
            pool,
            fetch,
            slot_collisions: 0,
            hits: 0,
            misses: 0,
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

    /// The three per-projection codebooks (gate/up/down) for `launch_moe_expert_range`.
    pub fn codebooks(&self) -> [*const f32; 3] {
        self.codebooks
    }

    /// Accumulated reaper fetch wall (ns) — the off-main-thread load cost the expert
    /// stream's compute overlaps. The profile reads it against the MoE wall.
    pub fn fetch_ns(&self) -> u64 {
        self.fetch.fetch_ns()
    }

    /// The format-agnostic streaming half of `submit_layer`: trace sink, phase 1a
    /// (hit+protect), phase 1b (alloc + slot-collision guard + `stream_expert`), phase
    /// 2 (`submit`). Returns the per-`sel` resolved slots + the `moe_table` row base.
    fn submit_spine(
        &mut self,
        layer: usize,
        sel: &[usize],
    ) -> Result<([Option<usize>; 32], usize, Vec<Signal>)> {
        let cfg = self.cfg;
        let sparse = (layer - cfg.dense_layers) * cfg.n_experts; // moe_table row base
        debug_assert!(
            sel.len() <= 32,
            "submit_layer: {} experts exceeds the 32-slot scratch",
            sel.len()
        );
        // Trace sink (--trace): the keys this layer looks up, in access order.
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
        // Phase 1a: touch EVERY hit first so a later miss's `alloc()` cannot evict a
        // same-layer would-be hit out from under itself.
        let mut slots: [Option<usize>; 32] = [None; 32];
        for (i, &e) in sel.iter().enumerate() {
            if let Some(slot) = self.pool.get(expert_key(layer, e)) {
                self.hits += 1;
                self.pool.protect(expert_key(layer, e));
                slots[i] = Some(slot);
            }
        }
        // Phase 1b: allocate the misses and build their cold-read specs, now that all
        // hits are protected. `batch_slots` is the async slot-collision guard: within
        // one batch two reads must never target the same slot (io_uring completion
        // order is NOT submission order, so overlapping writes corrupt silently). It
        // can only happen if a layer's misses exceed the free pool — impossible in a
        // real config (misses ≤ top_k ≪ the 2Q probation segment), so refusing is the
        // lazy-correct guard. ponytail: bail on collision; upgrade to a sub-batch await
        // (submit + await the colliding slot, then continue) if a tiny pool ever needs it.
        let base = self.lru_pool.ptr_mut();
        let mut reads: Vec<ReadSpec> = Vec::new();
        let mut miss_sel: Vec<usize> = Vec::new(); // sel-index of each read, for signal mapping
        let mut batch_slots: Vec<usize> = Vec::new();
        for (i, &e) in sel.iter().enumerate() {
            if slots[i].is_some() {
                continue;
            }
            self.misses += 1;
            let slot = self.pool.alloc(expert_key(layer, e))?;
            if batch_slots.contains(&slot) {
                self.slot_collisions += 1;
                bail!(
                    "slot-collision: layer {layer} batch reuses an in-flight slot \
                     (misses exceed the free pool) — raise the pool (--max-mem)"
                );
            }
            batch_slots.push(slot);
            let (fd, begin, len) = self.moe_table[sparse + e];
            // SAFETY: slot*expert_slot is block-aligned (expert_slot is a VQ_ALIGN
            // multiple) and within the slab; the slot stays live until this expert's
            // Signal resolves (the pipeline holds it).
            let dst = unsafe { base.add(slot * self.expert_slot) };
            reads.push(ReadSpec {
                fd,
                begin,
                len,
                dst,
            });
            miss_sel.push(i);
            slots[i] = Some(slot);
        }
        // Phase 2: hand the whole batch to the reaper — it queues+submits (all reads
        // start on the NVMe at once) and resolves each miss's Signal when its copy
        // lands. Hits default to `ready()`; overwrite each miss with its load Signal.
        let miss_signals = self.fetch.submit(reads)?;
        let mut signals: Vec<Signal> = (0..sel.len()).map(|_| Signal::ready()).collect();
        for (k, &i) in miss_sel.iter().enumerate() {
            signals[i] = miss_signals[k].clone();
        }
        Ok((slots, sparse, signals))
    }

    /// Submit one layer's cold reads and resolve each selected expert to its
    /// [`MlpVq`] (device pointers into the pool slots) plus its load [`Signal`]. The
    /// descriptors are valid POINTERS immediately (a slot's address is fixed at
    /// `pool.alloc` time); the bytes land when the matching `signals[i]` resolves.
    /// The expert stream awaits each signal before computing that expert.
    pub fn submit_layer(
        &mut self,
        layer: usize,
        sel: &[usize],
        out: &mut Vec<MlpVq>,
    ) -> Result<Vec<Signal>> {
        let (slots, _sparse, signals) = self.submit_spine(layer, sel)?;
        let cfg = self.cfg;
        let (gate_o, gate_i) = (cfg.moe_inter, cfg.hidden);
        let (down_o, down_i) = (cfg.hidden, cfg.moe_inter);
        let slab = self.lru_pool.ptr();
        let es = self.expert_slot;
        let o = self.vq_off;
        out.clear();
        for (i, _e) in sel.iter().enumerate() {
            let slot = slots[i].context("submit_layer: unresolved expert slot")?;
            // SAFETY: slot base within the slab; address arithmetic only (the bytes
            // land when `signals[i]` resolves).
            let b = unsafe { slab.add(slot * es) };
            let vw = |ioff: usize, soff: usize, o_dim: usize, i_dim: usize| VqWeight {
                indices: unsafe { b.add(ioff) },
                scales: unsafe { b.add(soff) } as *const u16,
                o_dim,
                i_dim,
            };
            out.push(MlpVq {
                gate: vw(o[0], o[1], gate_o, gate_i),
                up: vw(o[2], o[3], gate_o, gate_i),
                down: vw(o[4], o[5], down_o, down_i),
            });
        }
        Ok(signals)
    }
}

/// Resolve every routed expert's cold-read spec `(fd, begin, len)` ONCE, indexed
/// `(layer - dense_layers) * n_experts + expert`. Each `.vq3` expert is a single
/// O_DIRECT-aligned block, so this is just [`Vq3Set::read_spec`] tabulated (the
/// range/dim checks run at `Vq3Set::open`).
fn build_moe_table(vq: &Vq3Set, cfg: &ModelConfig) -> Result<Vec<(RawFd, usize, usize)>> {
    let n_moe = cfg.n_layers - cfg.dense_layers;
    let mut table = Vec::with_capacity(n_moe * cfg.n_experts);
    for l in cfg.dense_layers..cfg.n_layers {
        for e in 0..cfg.n_experts {
            table.push(vq.read_spec(l, e)?);
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The VQ slot offsets the descriptor build hands the kernel MUST equal where the
    // loader (`quant::vq_expert`) slices each projection's indices/scales out of an
    // expert block — else the streamed bytes and the descriptor pointers disagree
    // (silent wrong weights). Locks the single source of truth for the .vq3 layout.
    #[test]
    fn vq_offsets_match_loader() {
        let (hidden, moe_inter) = (crate::quant::VQ_GROUP, crate::quant::VQ_GROUP);
        let off = vq_slot_offsets(hidden, moe_inter);
        let block = vec![0u8; crate::quant::vq_expert_bytes(hidden, moe_inter)];
        let projs = crate::quant::vq_expert(&block, 0, hidden, moe_inter);
        let base = block.as_ptr() as usize;
        for (k, proj) in projs.iter().enumerate() {
            assert_eq!(
                proj.indices.as_ptr() as usize - base,
                off[k * 2],
                "indices proj {k}"
            );
            assert_eq!(
                proj.scales.as_ptr() as usize - base,
                off[k * 2 + 1],
                "scales proj {k}"
            );
        }
    }
}
