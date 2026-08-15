//! The GLM resident set + routed pool, placed by `core::residency::partition()`.
//!
//! Ported from `old:src/memory/pin.rs` with the M4 narrowings (single routed format, no
//! DSA indexer, no MTP head) and ONE re-architecture: **the old `Pin::build` derived
//! placement itself** — resident tier sized to a footprint it computed, pool handed
//! whatever was left. Here the split has one author: the pin enumerates its streamable
//! units (the routed experts, layer-major), states its [`Floor`], and EXECUTES the
//! [`Partition`] that `partition()` returns. A budget below the floor is a [`Refusal`]
//! with the arithmetic in it, at startup, before any device allocation — the run never
//! degrades (P6; INV-8 is the monotonicity gate on the function this defers to).
//!
//! The partition's `pinned` prefix sizes the pool beyond its minimum batch slots; the
//! pool's cache then decides WHICH experts occupy those bytes as the run's access
//! pattern emerges. That division of labour is deliberate: `partition()` owns how many
//! bytes of experts are resident (a pure function of free memory), the pool owns which
//! (residency moves bytes — never arithmetic, and in this tree never a format either).

use crate::device::DeviceTier;
use crate::routed::{ExpertSlot, MAX_BATCH, PoolCfg, RoutedGeom, RoutedPool, pool_budget, slot_at};
use anyhow::{Context, Result, ensure};
use rivoli_artifact::format::{
    Dtype, ExpertSet, FormatMeta, RoutedFmt, Safetensors, SetDims, load_codebooks,
};
use rivoli_artifact::glm_config::ModelConfig;
use rivoli_artifact::quant::vq::{VQ_DIM, VQ_K};
use rivoli_core::residency::{Bytes, Floor, Partition, Refusal, Unit, UnitId, partition};

/// An fp8-e4m3 block-scaled weight resolved to device addresses, with the dims the
/// launch site needs. Dims ride the weight (taken from the tensor's own shape at place
/// time) rather than `cfg`, so a mis-shaped artifact fails at load, not at launch.
#[derive(Clone, Copy)]
pub struct Fp8Weight {
    pub packed: *const u8,
    pub scale: *const f32,
    pub block: usize,
    pub o_dim: usize,
    pub i_dim: usize,
}

/// A dense layer's three-projection MLP, all fp8.
#[derive(Clone, Copy)]
pub struct Fp8Mlp {
    pub gate: Fp8Weight,
    pub up: Fp8Weight,
    pub down: Fp8Weight,
}

/// An int8 per-row-scaled weight (embed / lm_head).
#[derive(Clone, Copy)]
pub struct Int8Weight {
    pub packed: *const u8,
    pub scale: *const f32,
    pub o_dim: usize,
    pub i_dim: usize,
}

/// A layer's MLP: the first `dense_layers` are ordinary fp8 MLPs; the rest are MoE —
/// a host-side router gate plus the always-resident shared expert (same format as the
/// routed pool; there is exactly one format per run at M4).
pub enum LayerMlp {
    Dense(Fp8Mlp),
    Moe {
        gate_w: *const f32,
        shared: ExpertSlot,
    },
}

/// One layer's resolved always-resident weights. No indexer at M4 (dense attention
/// first — the DSA weights join when `--attn dsa` does).
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
}

/// What [`GlmPin::build`] needs beyond the artifact: the run's budget and the pool
/// knobs. Bundled for the same reason as [`PoolCfg`]: every field is a startup-time
/// decision.
#[derive(Clone, Copy)]
pub struct GlmPinCfg<'a> {
    /// Total device budget (`--max-mem`, auto-discovered when absent).
    pub capacity: usize,
    /// The run's ONE routed format (M4: `Vq3` or `I4`; hybrid returns as a
    /// `FormatPlan`, not as a pool property).
    pub fmt: RoutedFmt,
    pub cache_policy: &'a str,
    pub two_q: rivoli_core::cache::TwoQSplit,
    pub trace_path: Option<&'a str>,
}

/// The GLM resident weight set + cold-expert streaming pool.
pub struct GlmPin<'a> {
    cfg: &'a ModelConfig,
    /// The resident weight slab. Never read through after `build` — held purely as the
    /// RAII owner of the VMM allocation every resident pointer points into.
    #[allow(dead_code)]
    tier: DeviceTier,
    pub embed: Int8Weight,
    pub lm_head: Int8Weight,
    pub final_norm: *const f32,
    pub layers: Vec<LayerPin>,
    /// Router correction bias per MoE layer, kept HOST-side (the sigmoid/bias/top-k
    /// routing runs on the CPU). `moe_bias[layer - dense_layers]`, len n_experts.
    moe_bias: Vec<Vec<f32>>,
    /// The three per-projection codebooks (gate/up/down), resident, fp16 (narrowed
    /// from the f32 source at load — the random idx→cb gather is the MoE hot path,
    /// and fp16 halves it into L1). Null in int4 (decodes without a codebook).
    codebooks: [*const u16; 3],
    /// The streaming source — fd owner backing the pool's read table; held for the
    /// run, not read through after `build`.
    #[allow(dead_code)]
    src: ExpertSet,
    /// The partition that placed this run: how many routed experts fit resident
    /// beyond the batch slots. Kept so the startup log and tests can cite the
    /// decision rather than re-deriving it.
    pub placement: Partition,
    pub routed: RoutedPool,
}

/// Place an F32 tensor (norms, router gate) into the tier.
fn place_f32(tier: &mut DeviceTier, st: &Safetensors, name: &str) -> Result<*const f32> {
    let (bytes, _) = st.typed(name, Dtype::F32)?;
    // f32 LE host == LE device.
    Ok(tier.place(bytes)? as *const f32)
}

/// Place an fp8-e4m3 block-scaled weight (`<name>.weight` F8E4M3 + `.weight_scale_inv`
/// F32) into the tier. Dims come from the weight's `[o_dim, i_dim]` shape.
///
/// **The SCALE GRID's shape is checked.** The kernel indexes
/// `scale[(o/block)·sc_cols + i/block]` with `sc_cols` derived from `i_dim` — so a grid
/// stored `[sc_cols, sc_rows]` has every tile taking a neighbour's scale (fluent wrong
/// text, no error), and a SHORTER grid has the kernel reading past the placement into
/// the next tensor's e4m3 bytes reinterpreted as f32. Found by review in the old tree
/// on the Glimmer fp8 path, where the check ran only for streamed layers and the
/// shipping partition pinned all 52 — it iterated zero times.
fn place_fp8(
    tier: &mut DeviceTier,
    st: &Safetensors,
    name: &str,
    block: usize,
) -> Result<Fp8Weight> {
    let (w, shape) = st.typed(&format!("{name}.weight"), Dtype::F8E4M3)?;
    let (o_dim, i_dim) = dims2(&format!("{name}.weight"), shape)?;
    let (sc, sshape) = st.typed(&format!("{name}.weight_scale_inv"), Dtype::F32)?;
    let want = [o_dim.div_ceil(block), i_dim.div_ceil(block)];
    ensure!(
        sshape == want,
        "{name}.weight_scale_inv is {sshape:?}, but a [{o_dim}, {i_dim}] weight at \
         block {block} implies {want:?} — a grid of the wrong extent mis-tiles silently"
    );
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
/// that carries dims takes them from the tensor's own shape rather than from `cfg`, so
/// this is the one place the rank is confronted.
fn dims2(name: &str, shape: &[usize]) -> Result<(usize, usize)> {
    ensure!(shape.len() == 2, "{name}: expected 2-D, got {shape:?}");
    Ok((shape[0], shape[1]))
}

/// Place an int8 per-row weight (`<name>` I8 + `<name>.scale` F32) into the tier
/// (embed / lm_head).
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

/// Device bytes the always-resident set occupies — everything read every token EXCEPT
/// the routed experts.
///
/// **It is the artifact's own `*.safetensors` byte length, not a second derivation of
/// the placement layout.** The converter writes `resident.safetensors` from exactly the
/// tensor list [`GlmPin::build`] places, so the file IS the footprint — the old tree
/// replaced 73 lines of per-shard re-derivation with this, because the copy nothing
/// executed was free to drift. The bias is deliberate: over-count only shrinks the
/// routed pool; under-count is what `DeviceTier::place` bails on. It over-counts by each
/// file's header and by the ~77 KB of router bias (read to the HOST, never placed).
///
/// `indexer.safetensors` is skipped whole — that IS the per-tensor `.indexer` filter
/// (the old converter writes nothing else into it), and M4 places no indexer. On top of
/// the files: one shared expert block per MoE layer (carved from the routed slab, which
/// no safetensors holds) and, under VQ, the three fp16 codebooks.
fn resident_bytes(dir: &str, cfg: &ModelConfig, fmt: RoutedFmt) -> Result<usize> {
    let mut total = 0usize;
    for entry in std::fs::read_dir(dir).with_context(|| format!("read dir {dir}"))? {
        let p = entry.with_context(|| format!("read dir {dir}"))?.path();
        let is_st = p.extension().is_some_and(|x| x == "safetensors");
        let skipped = p.file_name().is_some_and(|n| n == "indexer.safetensors");
        if !is_st || skipped {
            continue;
        }
        total += p.metadata().with_context(|| format!("stat {p:?}"))?.len() as usize;
    }
    let shared = expert_bytes(cfg, fmt)?;
    let cbs = if fmt == RoutedFmt::Vq3 {
        3 * VQ_K * VQ_DIM * 2
    } else {
        0
    };
    Ok(total + (cfg.n_layers - cfg.dense_layers) * shared + cbs)
}

/// One expert block's bytes in `fmt` — the routed unit size and the shared expert's.
/// The one place GLM's format set is confronted: `.f4` is a V4 container, and refusing
/// it here (rather than an `unreachable!` after a check somewhere upstream) keeps the
/// function total when a new caller arrives without the check.
fn expert_bytes(cfg: &ModelConfig, fmt: RoutedFmt) -> Result<usize> {
    match fmt {
        RoutedFmt::Vq3 => Ok(rivoli_artifact::quant::vq::vq_expert_bytes(
            cfg.hidden,
            cfg.moe_inter,
        )),
        RoutedFmt::I4 => Ok(rivoli_artifact::quant::int4::i4_expert_bytes(
            cfg.hidden,
            cfg.moe_inter,
        )),
        RoutedFmt::F4 => anyhow::bail!("GLM streams .vq3 or .i4; .f4 is a V4 format"),
    }
}

/// The routed experts as residency units, layer-major — the priority order handed to
/// `partition()`. Layer-major because decode's access is cyclic over layers: pinning a
/// prefix of it is the static partition that cyclic access makes optimal (the Belady
/// degenerate), and any per-expert evidence that beats uniform arrives later as a
/// reordering of this list, never as a second placement author.
fn expert_units(cfg: &ModelConfig, unit: usize) -> Vec<Unit> {
    let unit = u64::try_from(unit).unwrap_or(u64::MAX);
    let bytes = std::num::NonZeroU64::new(unit).unwrap_or(std::num::NonZeroU64::MIN);
    let moe_layers = cfg.n_layers - cfg.dense_layers;
    (0..moe_layers * cfg.n_experts)
        .map(|i| Unit {
            id: UnitId(u32::try_from(i).unwrap_or(u32::MAX)),
            bytes,
        })
        .collect()
}

impl<'a> GlmPin<'a> {
    /// Build the resident set from the artifact directory `dir`, placing by
    /// `partition()`. See the module doc for the split of authorship.
    pub fn build(dir: &str, cfg: &'a ModelConfig, pin: GlmPinCfg<'_>) -> Result<Self> {
        // One-time bound for `RoutedPool::submit`'s fixed scratch. A batched forward
        // submits the UNION of every token row's picks, so the worst case is
        // `top_k · MAXROW + n_shared`, not one row's picks.
        let max_batch = cfg.top_k * super::MAXROW + cfg.n_shared;
        ensure!(
            max_batch <= MAX_BATCH,
            "top_k {} x {} rows + n_shared {} = {max_batch} exceeds the \
             {MAX_BATCH}-slot batch scratch",
            cfg.top_k,
            super::MAXROW,
            cfg.n_shared
        );
        // Open the artifact: format meta (fp8 block), the resident safetensors,
        // codebooks, and the one streaming source. `open_dir` merges every
        // *.safetensors; the routed slab files are not safetensors and are ignored.
        let fmt_meta = FormatMeta::load(dir)?;
        let block = fmt_meta.fp8_block;
        let st = Safetensors::open_dir(dir)?;
        let dims = SetDims::new(
            cfg.dense_layers..cfg.n_layers,
            cfg.n_experts,
            cfg.hidden,
            cfg.moe_inter,
        );
        let src = ExpertSet::open_routed(dir, pin.fmt, dims)?;
        let geom = RoutedGeom::new(&src)?;

        // THE PLACEMENT DECISION — one call, one author. The floor is what the run
        // pays before any expert is resident: the always-resident set (plus slack for
        // per-reservation alignment padding) and the pool's minimum batch slots. KV
        // and scratch are 0 here NOT because they are free but because GLM's
        // `--max-mem` has always budgeted weights only (every recorded benchmark
        // reads it that way); folding them in is a semantic change to the flag,
        // owed its own measured change when the loop owns KV allocation.
        const SLACK: usize = 256 << 20; // 256 MiB
        let resident = resident_bytes(dir, cfg, pin.fmt)?;
        let tier_cap = resident + SLACK;
        let unit = src.expert_slot();
        let floor = Floor {
            always_resident: Bytes(tier_cap as u64),
            kv_at_max_ctx: Bytes(0),
            scratch: Bytes(0),
            slot_bytes: Bytes((max_batch * unit) as u64),
        };
        let units = expert_units(cfg, unit);
        let placement = partition(&units, Bytes(pin.capacity as u64), floor)
            .map_err(|r: Refusal| anyhow::anyhow!("{r} (GLM pin, --max-mem)"))?;
        // Execute: the pool's byte budget IS the partition — batch slots plus the
        // pinned prefix. `pool_budget`'s O_DIRECT rounding still applies (the arena
        // anchors HOT slots at the high end, so an unaligned budget would misalign
        // every hot-slot DMA destination).
        let budget = pool_budget(
            tier_cap + (max_batch + placement.pinned.len()) * unit,
            tier_cap,
        );
        tracing::info!(
            "partition: {} of {} routed experts fit resident beyond the {max_batch} \
             batch slots ({:.1} GiB pool)",
            placement.pinned.len(),
            units.len(),
            budget as f64 / (1u64 << 30) as f64,
        );
        let mut tier = DeviceTier::new(tier_cap)?;

        // Codebooks resident (gate/up/down), narrowed f32 → fp16 at load. VQ only —
        // int4 decodes without a codebook.
        let mut codebooks = [std::ptr::null(); 3];
        if pin.fmt == RoutedFmt::Vq3 {
            let cbs = load_codebooks(dir)?;
            for (i, cb) in cbs.iter().enumerate() {
                let half: Vec<u8> = cb
                    .iter()
                    .flat_map(|&v| rivoli_core::num::f32_to_f16(v).to_le_bytes())
                    .collect();
                codebooks[i] = tier.place(&half)? as *const u16;
            }
        }

        // Global tensors.
        let embed = place_i8(&mut tier, &st, "model.embed_tokens.weight")?;
        let lm_head = place_i8(&mut tier, &st, "lm_head.weight")?;
        let final_norm = place_f32(&mut tier, &st, "model.norm.weight")?;
        let (layers, moe_bias) = place_layers(&mut tier, &st, cfg, &src, block)?;

        let routed = RoutedPool::new(
            PoolCfg {
                budget,
                top_k: cfg.top_k,
                max_batch,
                policy: pin.cache_policy,
                two_q: pin.two_q,
                trace: pin.trace_path,
            },
            geom,
        )?;

        Ok(Self {
            cfg,
            tier,
            embed,
            lm_head,
            final_norm,
            layers,
            moe_bias,
            codebooks,
            src,
            placement,
            routed,
        })
    }

    /// Host router correction bias for a MoE `layer` (len n_experts).
    pub fn moe_bias(&self, layer: usize) -> &[f32] {
        &self.moe_bias[layer - self.cfg.dense_layers]
    }

    /// The three per-projection codebooks (gate/up/down), fp16. Null pointers in int4.
    pub fn codebooks(&self) -> [*const u16; 3] {
        self.codebooks
    }
}

/// Place every layer's always-resident weights. `l` indexes both the weight-name
/// `format!`s and the dense/MoE split. Returns the layers and the host-side router
/// bias, which accrete together because both are per-MoE-layer.
fn place_layers(
    tier: &mut DeviceTier,
    st: &Safetensors,
    cfg: &ModelConfig,
    src: &ExpertSet,
    block: usize,
) -> Result<(Vec<LayerPin>, Vec<Vec<f32>>)> {
    let off = src.slot_offsets();
    let mut layers = Vec::with_capacity(cfg.n_layers);
    let mut moe_bias = Vec::new();
    for l in 0..cfg.n_layers {
        let lb = format!("model.layers.{l}");
        let a = format!("{lb}.self_attn");
        let mlp = if l < cfg.dense_layers {
            LayerMlp::Dense(place_dense_mlp(tier, st, &format!("{lb}.mlp"), block)?)
        } else {
            let gate_w = place_f32(tier, st, &format!("{lb}.mlp.gate.weight"))?;
            let (bias, _) = st.typed(
                &format!("{lb}.mlp.gate.e_score_correction_bias"),
                Dtype::F32,
            )?;
            let bias = rivoli_artifact::quant::read_f32(bias);
            ensure!(
                bias.len() == cfg.n_experts,
                "layer {l} gate bias has {} entries, expected {}",
                bias.len(),
                cfg.n_experts
            );
            moe_bias.push(bias);
            // The always-resident shared expert rides the run's one routed format,
            // carved from the same slab the pool streams from.
            let shared_block = src.shared_block(l)?;
            let dst = tier.place(&shared_block)?;
            // SAFETY: `off` are this format's slot offsets and `shared_block` is one
            // whole expert block, so every offset lies inside the reservation.
            let shared = unsafe { slot_at(dst, &off) };
            LayerMlp::Moe { gate_w, shared }
        };
        layers.push(LayerPin {
            input_ln: place_f32(tier, st, &format!("{lb}.input_layernorm.weight"))?,
            post_ln: place_f32(tier, st, &format!("{lb}.post_attention_layernorm.weight"))?,
            q_a: place_fp8(tier, st, &format!("{a}.q_a_proj"), block)?,
            q_a_ln: place_f32(tier, st, &format!("{a}.q_a_layernorm.weight"))?,
            q_b: place_fp8(tier, st, &format!("{a}.q_b_proj"), block)?,
            kv_a: place_fp8(tier, st, &format!("{a}.kv_a_proj_with_mqa"), block)?,
            kv_a_ln: place_f32(tier, st, &format!("{a}.kv_a_layernorm.weight"))?,
            kv_b: place_fp8(tier, st, &format!("{a}.kv_b_proj"), block)?,
            o_proj: place_fp8(tier, st, &format!("{a}.o_proj"), block)?,
            mlp,
        });
    }
    Ok((layers, moe_bias))
}
