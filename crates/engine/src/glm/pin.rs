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
// `Fp8Weight` and its placers moved to `crate::resident` when the V4 arm needed the same
// three (M8) — same types, same bodies, one home. Re-exported below so every
// `crate::glm::pin::Fp8Weight` path in this arm is unchanged.
use crate::resident::{
    Batch, PinCfg, PoolPlan, dims2, place_f32, place_fp8, safetensors_bytes, stream_units,
};
use crate::routed::{ExpertSlot, RoutedGeom, RoutedPool, slot_at};
use anyhow::{Result, ensure};
use rivoli_artifact::format::{
    Dtype, ExpertSet, FormatMeta, RoutedFmt, Safetensors, SetDims, load_codebooks,
};
use rivoli_artifact::glm_config::ModelConfig;
use rivoli_artifact::quant::vq::{VQ_DIM, VQ_K};
use rivoli_core::residency::{Partition, Unit};

/// `Fp8Weight` and `Fp8Mlp` moved to [`crate::resident`] when the V4 arm needed the same
/// two (M8) — same types, same bodies, one home. Re-exported so every
/// `crate::glm::pin::Fp8Weight` path in this arm is unchanged. A dense GLM layer's MLP and
/// V4's resident shared expert ARE the same object; see [`Fp8Mlp`]'s own note.
pub use crate::resident::{Fp8Mlp, Fp8Weight};

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
/// The file total and its bias are [`safetensors_bytes`]'s; `indexer.safetensors` is the
/// `skip` it takes, which IS the per-tensor `.indexer` filter (the old converter writes
/// nothing else into it) and M4 places no indexer. What is GLM's own, and therefore here:
/// one shared expert block per MoE layer (carved from the routed slab, which no safetensors
/// holds) and, under VQ, the three fp16 codebooks.
fn resident_bytes(dir: &str, cfg: &ModelConfig, fmt: RoutedFmt) -> Result<usize> {
    let total = safetensors_bytes(dir, Some("indexer.safetensors"))?;
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

/// The routed experts as residency units, LAYER-MAJOR — the priority order handed to
/// `partition()`. Layer-major because decode's access is cyclic over layers: pinning a
/// prefix of it is the static partition that cyclic access makes optimal (the Belady
/// degenerate), and any per-expert evidence that beats uniform arrives later as a
/// reordering of this list, never as a second placement author.
///
/// The construction is [`stream_units`]'s and the ORDER is this function's, which is the
/// whole split: GLM's list skips the dense layers and V4's spans the artifact's own range,
/// and neither fact belongs in a shared helper.
fn expert_units(cfg: &ModelConfig, unit: usize) -> Vec<Unit> {
    stream_units((cfg.n_layers - cfg.dense_layers) * cfg.n_experts, unit)
}

impl<'a> GlmPin<'a> {
    /// Build the resident set from the artifact directory `dir`, placing by
    /// `partition()`. See the module doc for the split of authorship.
    pub fn build(dir: &str, cfg: &'a ModelConfig, fmt: RoutedFmt, pin: PinCfg<'_>) -> Result<Self> {
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
        let src = ExpertSet::open_routed(dir, fmt, dims)?;
        let geom = RoutedGeom::new(&src)?;

        // THE PLACEMENT DECISION — one call, one author, and it runs before any device
        // allocation. The tier's SLACK is per-reservation alignment padding, which
        // `DeviceTier::place` charges at 256 B a placement.
        const SLACK: usize = 256 << 20; // 256 MiB
        let resident = resident_bytes(dir, cfg, fmt)?;
        let tier_cap = resident + SLACK;
        let unit = src.expert_slot();
        let units = expert_units(cfg, unit);
        // A batched forward submits the UNION of every token row's picks, and folds the ONE
        // routed-format shared expert in beside them — so the scratch bound is
        // `top_k * MAXROW + n_shared` and not one row's picks. `Batch::union` is where that
        // bound is checked, at startup, with the friendly message.
        let batch = Batch::union(cfg.top_k, super::MAXROW, cfg.n_shared, unit)?;
        let (placement, pool) = PoolPlan::new("GLM", &units, tier_cap, batch).decide(pin)?;
        let mut tier = DeviceTier::new(tier_cap)?;

        // Codebooks resident (gate/up/down), narrowed f32 → fp16 at load. VQ only —
        // int4 decodes without a codebook.
        let mut codebooks = [std::ptr::null(); 3];
        if fmt == RoutedFmt::Vq3 {
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

        let routed = RoutedPool::new(pool, geom)?;

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
