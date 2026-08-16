//! The K3 resident set (bf16 trunk, widened norms and folds) plus the `.f4` routed pool,
//! placed by `core::residency::partition()` through [`PoolPlan`] — the same one-author
//! placement contract as `crate::v4::pin`, over a different tensor census.
//!
//! # The two load-time transforms, and why the loader owns both
//!
//! * **Every norm the kernels read as f32 is WIDENED here.** `convert_k3` copies the
//!   checkpoint verbatim, so the norms arrive BF16 — its own header hands the widening
//!   question to "whoever writes the K3 loader", because a BF16 tensor read as f32 is not a
//!   length error, it is half the rows at wrong magnitudes. Widening at load costs ~11 MB
//!   across 93 layers and buys every norm launch the `*const f32` the launchers declare.
//! * **The AttnRes fold vectors are COLLAPSED here**: `fold[i] = norm.weight[i] *
//!   proj.weight[i]`, per layer twice plus once model-level
//!   (`k3:docs/reference/k3-architecture.md` §3 — "foldable at load time", and the C does it
//!   in its loader). The kernel takes the product; placing the factors would put a per-token
//!   multiply on the hot path to preserve a distinction nothing downstream reads.
//!
//! # What this pin does NOT do, each a decision
//!
//! * **`A_log` is placed at its shipped `[128]` and READ at `[96]`** — trap 1's disk half.
//!   The checkpoint disagrees with its own modeling code (`k3:docs/reference/k3-architecture.md`
//!   §4: modeling declares `[num_heads]`, disk ships F32 `[128]`); the first `num_heads`
//!   entries are the real ones and they are contiguous, so the slice is the same pointer.
//!   The shape is ASSERTED so a future checkpoint that ships `[96]` fails loudly here rather
//!   than silently reading 32 entries of the next tensor.
//! * **The router bias stays on the HOST.** Routing is host math (the pool's `submit` is
//!   host code — `crate::v4::moe` carries the full argument), so
//!   `gate.e_score_correction_bias` is read into a `Vec<f32>` and never placed.
//! * **No requantization anywhere.** The trunk decodes bf16 through `launch_gemm_bf16`;
//!   whether an int8 embed/head is worth it is a quality question with a paired-dNLL
//!   measurement attached, not a loader's decision. [`Bf16Weight`] is the type that says so.

use super::geometry::Dims;
use crate::device::{DeviceTier, as_le_bytes};
use crate::resident::{
    Batch, Bf16Weight, PinCfg, PoolPlan, place_bf16, safetensors_bytes, stream_units,
};
use crate::routed::{RoutedGeom, RoutedPool};
// `Context as _`: the trait for `with_context`, imported anonymously — the named form
// reproduces `v4/pin.rs`'s preamble token for token and the jscpd gate reports the pair
// (an import list is the one duplication Rust gives no way to factor; `golden_read.rs`'s
// rule is to have FEWER imports, and this is its anonymous-import corollary).
use anyhow::{Context as _, Result, ensure};
use rivoli_artifact::format::{Dtype, Safetensors};
use rivoli_artifact::format::{ExpertSet, RoutedFmt, SetDims, f4_layer_range};
use rivoli_artifact::k3_config::K3TextConfig;
use rivoli_core::num::bf16_to_f32;

/// Alignment and widening slack on top of the artifact's own byte length.
///
/// [`safetensors_bytes`] measures the FILE, and this pin then places ~11 MB the file does
/// not hold: the widened f32 norms and the collapsed folds (4 vectors x 93 layers x 7168
/// f32, plus the LoRA and latent norms). The bf16 originals it skips give most of that
/// back, but the direction must be safe BY CONSTRUCTION, not by netting — so the slack
/// covers the widened set outright, plus `DeviceTier`'s per-placement alignment. 64 MiB
/// against a ~113 GiB trunk is 0.06%.
const PIN_SLACK: usize = 64 << 20;

/// One KDA layer's weights (`k3:docs/reference/k3-architecture.md` §4's table, resolved to
/// device addresses). The three conv weights are `[ch][1][taps]` F32 on disk and the middle
/// axis is size 1, so the placed bytes ARE the `[ch][taps]` layout the conv launcher indexes.
pub struct KdaWeights {
    pub q: Bf16Weight,
    pub k: Bf16Weight,
    pub v: Bf16Weight,
    pub q_conv: *const f32,
    pub k_conv: *const f32,
    pub v_conv: *const f32,
    /// The shared rank-`head_dim` pair feeding all heads' decay: `z = f_b(f_a(x))`.
    pub f_a: Bf16Weight,
    pub f_b: Bf16Weight,
    /// Per-head beta projection, `[heads][hidden]`.
    pub b: Bf16Weight,
    /// Full-rank output gate, `[ch][hidden]` — KDA norms THEN gates (trap 10's KDA half).
    pub g: Bf16Weight,
    /// Shipped F32 `[128]`; the first `heads` entries are the per-HEAD decay (trap 1).
    pub a_log: *const f32,
    /// F32 `[ch]` — per (head, channel), unlike `a_log`.
    pub dt_bias: *const f32,
    /// F32 `[head_dim]`, shared across heads — the fused gated head norm's weight.
    pub o_norm: *const f32,
    pub o: Bf16Weight,
}

/// One gated-MLA layer's weights (`k3:docs/reference/k3-architecture.md` §5). The two LoRA
/// norms are widened; their EPS is [`super::geometry::MLA_LORA_EPS`], not the model's.
pub struct MlaWeights {
    pub q_a: Bf16Weight,
    pub q_a_norm: *const f32,
    pub q_b: Bf16Weight,
    /// ONE projection emits the compressed latent AND the shared rope slot.
    pub kv_a: Bf16Weight,
    /// Covers the LATENT half only, `[kv_lora]` — never the rope slot.
    pub kv_a_norm: *const f32,
    pub kv_b: Bf16Weight,
    /// Output gate, `[heads * v_head][hidden]` — MLA gates WITHOUT a norm (trap 10).
    pub g: Bf16Weight,
    pub o: Bf16Weight,
}

/// Which attention family this layer runs — the pin's copy of the layer map, decided once
/// at build from `layer_is_mla` and never re-derived from a modulus (layer_types-not-modulo
/// is a scar in this house; K3's map breaks its own period at 91/92).
pub enum Attn {
    Kda(KdaWeights),
    Mla(MlaWeights),
}

/// The MoE block's TRUNK side: everything §6 runs at bf16 outside the streamed experts.
pub struct MoeTrunk {
    /// `[n_experts][hidden]` bf16 — the router scores on the FULL width, before any
    /// projection (§6 step 1).
    pub gate: Bf16Weight,
    /// Host-side, f32 on disk. Steers SELECTION only; `state::combine_weights` cannot
    /// receive it.
    pub bias: Vec<f32>,
    /// `[latent][hidden]` — 7168 -> 3584, once per token, before the experts.
    pub down: Bf16Weight,
    /// Widened `[latent]` — RMSNorms the AGGREGATE between the weighted sum and the up
    /// projection (trap 12).
    pub latent_norm: *const f32,
    /// `[hidden][latent]` — 3584 -> 7168, after the norm.
    pub up: Bf16Weight,
    /// The ONE fused shared MLP, `[shared_inter][hidden]` x2 and `[hidden][shared_inter]` —
    /// trunk-side bf16, not MXFP4, not in the routed cache.
    pub shared: SituMlp,
}

/// One SiTU-GLU MLP's three projections. Two widths wear this shape: layer 0's dense FFN
/// (`dense_inter` 33792) and every MoE layer's fused shared MLP (`shared_inter` 6144) —
/// same chain, same betas, one type, which is also what keeps `forward.rs::situ_mlp` to
/// one author.
#[derive(Clone, Copy)]
pub struct SituMlp {
    pub gate: Bf16Weight,
    pub up: Bf16Weight,
    pub down: Bf16Weight,
}

pub enum Ffn {
    Dense(SituMlp),
    Moe(MoeTrunk),
}

/// One layer's resident weights: the two sandwich norms, the two collapsed folds, and the
/// two family enums. There is no illegal (attn, ffn) pairing to preclude — three of the
/// four combinations occur in the real map (layer 0 is KDA + dense, layer 1 KDA + MoE,
/// layer 3 MLA + MoE), and MLA + dense is merely absent, not wrong.
pub struct K3LayerPin {
    pub input_norm: *const f32,
    pub post_norm: *const f32,
    /// `self_attention_res_{norm,proj}` collapsed — the layer-entry fold's vector.
    pub attn_fold: *const f32,
    /// `mlp_res_{norm,proj}` collapsed — the pre-FFN fold's vector.
    pub mlp_fold: *const f32,
    pub attn: Attn,
    pub ffn: Ffn,
}

pub struct K3Pin {
    /// RAII owner of the resident slab; never read through.
    #[allow(dead_code)]
    tier: DeviceTier,
    pub embed: Bf16Weight,
    pub head: Bf16Weight,
    pub final_norm: *const f32,
    /// The model-level fold (`output_attn_res_{norm,proj}` collapsed) — §7's third
    /// aggregation, whose omission is silent.
    pub output_fold: *const f32,
    layers: Vec<K3LayerPin>,
    /// Which layers carry `.f4` expert files — `first_k_dense_replace..end`.
    moe_layers: std::ops::Range<usize>,
    /// fd owner for the pool's read table.
    #[allow(dead_code)]
    f4: ExpertSet,
    pub routed: RoutedPool,
}

impl K3Pin {
    /// Open `dir`, decide placement, and load the resident set. The DECISION happens before
    /// any device allocation, exactly as `PoolPlan::decide` requires.
    pub fn build(dir: &str, cfg: &K3TextConfig, pin: PinCfg<'_>) -> Result<Self> {
        let d = Dims::from_config(cfg)?;
        // The artifact's own `.f4` range. K3's starts at `first_k_dense_replace`, not 0 —
        // layer 0 is dense and has no expert file — and a decode needs it to START there
        // (the engine refuses otherwise; a shorter END is a golden-comparison prefix).
        let moe_layers = f4_layer_range(dir, cfg.n_layers)?;
        let st = Safetensors::open_dir(dir)?;
        // `expert_in` is the LATENT 3584, and the argument spelling matches the parameter —
        // the substitution `SetDims` warns about (`cfg.hidden` here is a self-consistent
        // artifact with every expert stride 2x wrong).
        let f4 = ExpertSet::open_routed(
            dir,
            RoutedFmt::F4,
            SetDims::new(
                moe_layers.clone(),
                cfg.n_experts,
                cfg.expert_in,
                cfg.moe_inter,
            ),
        )?;
        let geom = RoutedGeom::new(&f4)?;

        // The placement decision, before any allocation (P6: a function of free memory,
        // never architecture).
        let tier_cap = safetensors_bytes(dir, None)? + PIN_SLACK;
        let unit = f4.expert_slot();
        let units = stream_units(moe_layers.len() * cfg.n_experts, unit);
        let batch = Batch::union(cfg.top_k, super::ROWS, 0, unit)?;
        let (placement, pool) = PoolPlan::new("K3", &units, tier_cap, batch).decide(pin)?;
        tracing::info!(
            "K3 pin: {:.1} GiB trunk resident, layers 0..{} ({} MoE), {:.0} GiB routed set, \
             {:.1}% resident",
            tier_cap as f64 / (1u64 << 30) as f64,
            cfg.n_layers.min(moe_layers.end),
            moe_layers.len(),
            (units.len() * unit) as f64 / (1u64 << 30) as f64,
            100.0 * placement.pinned.len() as f64 / units.len().max(1) as f64,
        );

        let mut tier = DeviceTier::new(tier_cap)?;
        let mut p = Placer {
            tier: &mut tier,
            st: &st,
        };
        let embed = p.bf16("language_model.model.embed_tokens.weight")?;
        let head = p.bf16("language_model.lm_head.weight")?;
        ensure!(
            embed.o_dim == cfg.vocab && head.o_dim == cfg.vocab,
            "embed/head vocab {} / {} against the config's {} — no tied embeddings to fall \
             back to",
            embed.o_dim,
            head.o_dim,
            cfg.vocab
        );
        let final_norm = p.widen("language_model.model.norm.weight".into(), cfg.hidden)?;
        let output_fold = p.fold("language_model.model.output_attn_res")?;
        let layers = place_layers(&mut p, cfg, &d, moe_layers.end.min(cfg.n_layers))?;
        Ok(Self {
            tier,
            embed,
            head,
            final_norm,
            output_fold,
            layers,
            moe_layers,
            routed: RoutedPool::new(pool, geom)?,
            f4,
        })
    }

    /// Which layers this pin loaded — the decode range.
    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// The `.f4` layer range, for the engine's start-at-dense check.
    pub fn moe_layers(&self) -> std::ops::Range<usize> {
        self.moe_layers.clone()
    }

    pub fn layer(&self, l: usize) -> Result<&K3LayerPin> {
        self.layers
            .get(l)
            .ok_or_else(|| anyhow::anyhow!("layer {l} is outside the {} loaded", self.layers.len()))
    }
}

/// The `(tier, st)` pair every placement reads, held once so each of the ~30 tensor sites
/// per layer is ONE line naming the tensor and its expected extents — the shape a reader
/// can diff against the census in `docs/measurement/k3-reference/tensor-families.tsv`.
struct Placer<'a, 'b> {
    tier: &'a mut DeviceTier,
    st: &'b Safetensors,
}

impl Placer<'_, '_> {
    /// A bf16 matrix, dims from its own shape.
    fn bf16(&mut self, name: &str) -> Result<Bf16Weight> {
        place_bf16(self.tier, self.st, name)
    }

    /// A bf16 projection confronted against the dims the config implies. The shape
    /// authority stays the TENSOR's; this adds the cross-check, because three different
    /// 12288-wide tensors exist on this model and a transposed load is in-bounds.
    fn proj(&mut self, name: String, o: usize, i: usize) -> Result<Bf16Weight> {
        let w = self.bf16(&name)?;
        ensure!(
            (w.o_dim, w.i_dim) == (o, i),
            "{name}: [{}, {}], expected [{o}, {i}]",
            w.o_dim,
            w.i_dim
        );
        Ok(w)
    }

    /// A BF16 vector widened to the f32 the launchers declare, its extent asserted — a norm
    /// of the wrong length reads in-bounds garbage, and no launch has a length of its own.
    fn widen(&mut self, name: String, want: usize) -> Result<*const f32> {
        let (v, len) = read_bf16_1d(self.st, &name)?;
        ensure!(len == want, "{name}: [{len}], expected [{want}]");
        Ok(self.tier.place(as_le_bytes(&v))? as *const f32)
    }

    /// One AttnRes pair collapsed: `place(widen(<base>_norm) ⊙ widen(<base>_proj))`.
    ///
    /// The product rather than the factors, because the collapse is a load-time transform
    /// the C performs in its loader and the fold kernel takes ONE vector
    /// (`k3:docs/reference/k3-architecture.md` §3). The proj really is `[1][hidden]` — a
    /// single scoring vector, not a matrix — so both flatten and must agree elementwise.
    fn fold(&mut self, base: &str) -> Result<*const f32> {
        let (norm, n_len) = read_bf16_1d(self.st, &format!("{base}_norm.weight"))?;
        let (proj, p_len) = read_bf16_1d(self.st, &format!("{base}_proj.weight"))?;
        ensure!(
            n_len == p_len,
            "{base}: norm [{n_len}] against proj [{p_len}] — the fold is elementwise"
        );
        let prod: Vec<f32> = norm.iter().zip(&proj).map(|(a, b)| a * b).collect();
        Ok(self.tier.place(as_le_bytes(&prod))? as *const f32)
    }

    /// An F32 tensor whose shape the caller pins exactly — the KDA family's four
    /// disk-f32 tensors, each with its own shape story.
    fn f32_shaped(&mut self, name: String, want: &[usize], why: &str) -> Result<*const f32> {
        let (bytes, shape) = self.st.typed(&name, Dtype::F32)?;
        ensure!(
            shape == want,
            "{name}: {shape:?}, expected {want:?} — {why}"
        );
        Ok(self.tier.place(bytes)? as *const f32)
    }
}

/// A BF16 tensor flattened to host f32, with its element count. Rank is NOT constrained
/// here: the fold proj is `[1, hidden]` and the norms are `[hidden]`, and both flatten to
/// the vector the caller length-checks.
fn read_bf16_1d(st: &Safetensors, name: &str) -> Result<(Vec<f32>, usize)> {
    let (bytes, shape) = st.typed(name, Dtype::Bf16)?;
    let v: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    ensure!(
        v.len() == shape.iter().product::<usize>(),
        "{name}: {} elements against shape {shape:?}",
        v.len()
    );
    let len = v.len();
    Ok((v, len))
}

/// One KDA layer's family weights, each site one line against the census.
fn place_kda(p: &mut Placer<'_, '_>, cfg: &K3TextConfig, d: &Dims, at: &str) -> Result<KdaWeights> {
    let la = &cfg.linear_attn_config;
    let (ch, hid, taps, hd) = (d.kda_ch, cfg.hidden, la.short_conv_kernel_size, la.head_dim);
    let conv_why = "squeezing any other middle axis would reorder the taps";
    Ok(KdaWeights {
        q: p.proj(format!("{at}.q_proj.weight"), ch, hid)?,
        k: p.proj(format!("{at}.k_proj.weight"), ch, hid)?,
        v: p.proj(format!("{at}.v_proj.weight"), ch, hid)?,
        q_conv: p.f32_shaped(format!("{at}.q_conv1d.weight"), &[ch, 1, taps], conv_why)?,
        k_conv: p.f32_shaped(format!("{at}.k_conv1d.weight"), &[ch, 1, taps], conv_why)?,
        v_conv: p.f32_shaped(format!("{at}.v_conv1d.weight"), &[ch, 1, taps], conv_why)?,
        f_a: p.proj(format!("{at}.f_a_proj.weight"), hd, hid)?,
        f_b: p.proj(format!("{at}.f_b_proj.weight"), ch, hd)?,
        b: p.proj(format!("{at}.b_proj.weight"), la.num_heads, hid)?,
        g: p.proj(format!("{at}.g_proj.weight"), ch, hid)?,
        // Trap 1's disk half, checked as a LOWER bound rather than pinned at the shipped
        // [128]: the real checkpoint disagrees with its own modeling code and ships 128
        // entries for 96 heads, the anchor's tiny model ships exactly `num_heads` — both
        // are valid because only the contiguous first `num_heads` are ever read.
        a_log: {
            let (bytes, shape) = p.st.typed(&format!("{at}.A_log"), Dtype::F32)?;
            ensure!(
                shape.len() == 1 && shape[0] >= la.num_heads,
                "{at}.A_log: {shape:?} cannot cover {} heads",
                la.num_heads
            );
            p.tier.place(bytes)? as *const f32
        },
        dt_bias: p.f32_shaped(
            format!("{at}.dt_bias"),
            &[ch],
            "per (head, channel), NOT per head",
        )?,
        o_norm: p.f32_shaped(
            format!("{at}.o_norm.weight"),
            &[hd],
            "one weight shared across heads",
        )?,
        o: p.proj(format!("{at}.o_proj.weight"), hid, ch)?,
    })
}

/// One gated-MLA layer's family weights, same one-line-per-tensor shape.
fn place_mla(p: &mut Placer<'_, '_>, cfg: &K3TextConfig, d: &Dims, at: &str) -> Result<MlaWeights> {
    let (hid, nh) = (cfg.hidden, cfg.n_heads);
    Ok(MlaWeights {
        q_a: p.proj(format!("{at}.q_a_proj.weight"), cfg.q_lora_rank, hid)?,
        q_a_norm: p.widen(format!("{at}.q_a_layernorm.weight"), cfg.q_lora_rank)?,
        q_b: p.proj(
            format!("{at}.q_b_proj.weight"),
            nh * d.q_head,
            cfg.q_lora_rank,
        )?,
        kv_a: p.proj(format!("{at}.kv_a_proj_with_mqa.weight"), d.kv_a_out, hid)?,
        kv_a_norm: p.widen(format!("{at}.kv_a_layernorm.weight"), cfg.kv_lora_rank)?,
        kv_b: p.proj(
            format!("{at}.kv_b_proj.weight"),
            nh * d.kv_b_head,
            cfg.kv_lora_rank,
        )?,
        g: p.proj(format!("{at}.g_proj.weight"), nh * cfg.v_head_dim, hid)?,
        o: p.proj(format!("{at}.o_proj.weight"), hid, nh * cfg.v_head_dim)?,
    })
}

/// One MoE layer's trunk side (`k3:docs/reference/k3-architecture.md` §6).
fn place_moe(p: &mut Placer<'_, '_>, cfg: &K3TextConfig, d: &Dims, base: &str) -> Result<MoeTrunk> {
    let (hid, lat, sh) = (cfg.hidden, cfg.expert_in, d.shared_inter);
    let bias_name = format!("{base}.gate.e_score_correction_bias");
    let (b_bytes, b_shape) = p.st.typed(&bias_name, Dtype::F32)?;
    ensure!(
        b_shape == [cfg.n_experts],
        "{bias_name}: {b_shape:?}, expected [{}]",
        cfg.n_experts
    );
    let bias: Vec<f32> = b_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(MoeTrunk {
        gate: p.proj(format!("{base}.gate.weight"), cfg.n_experts, hid)?,
        bias,
        down: p.proj(format!("{base}.routed_expert_down_proj.weight"), lat, hid)?,
        latent_norm: p.widen(format!("{base}.routed_expert_norm.weight"), lat)?,
        up: p.proj(format!("{base}.routed_expert_up_proj.weight"), hid, lat)?,
        shared: SituMlp {
            gate: p.proj(format!("{base}.shared_experts.gate_proj.weight"), sh, hid)?,
            up: p.proj(format!("{base}.shared_experts.up_proj.weight"), sh, hid)?,
            down: p.proj(format!("{base}.shared_experts.down_proj.weight"), hid, sh)?,
        },
    })
}

/// Every layer in `0..end`: the sandwich norms, the folds, and the two family dispatches —
/// `layer_is_mla` (the explicit map, never a modulus) and `layer_is_dense`.
fn place_layers(
    p: &mut Placer<'_, '_>,
    cfg: &K3TextConfig,
    d: &Dims,
    end: usize,
) -> Result<Vec<K3LayerPin>> {
    let mut layers = Vec::with_capacity(end);
    for l in 0..end {
        let at = format!("language_model.model.layers.{l}");
        let attn = match cfg.layer_is_mla(l)? {
            true => Attn::Mla(place_mla(p, cfg, d, &format!("{at}.self_attn"))?),
            false => Attn::Kda(place_kda(p, cfg, d, &format!("{at}.self_attn"))?),
        };
        let ffn = match cfg.layer_is_dense(l) {
            true => Ffn::Dense(SituMlp {
                gate: p.proj(
                    format!("{at}.mlp.gate_proj.weight"),
                    cfg.dense_inter,
                    cfg.hidden,
                )?,
                up: p.proj(
                    format!("{at}.mlp.up_proj.weight"),
                    cfg.dense_inter,
                    cfg.hidden,
                )?,
                down: p.proj(
                    format!("{at}.mlp.down_proj.weight"),
                    cfg.hidden,
                    cfg.dense_inter,
                )?,
            }),
            false => Ffn::Moe(place_moe(p, cfg, d, &format!("{at}.block_sparse_moe"))?),
        };
        let sandwich = K3LayerPin {
            input_norm: p
                .widen(format!("{at}.input_layernorm.weight"), cfg.hidden)
                .with_context(|| format!("placing layer {l}"))?,
            post_norm: p.widen(format!("{at}.post_attention_layernorm.weight"), cfg.hidden)?,
            attn_fold: p.fold(&format!("{at}.self_attention_res"))?,
            mlp_fold: p.fold(&format!("{at}.mlp_res"))?,
            attn,
            ffn,
        };
        layers.push(sandwich);
    }
    Ok(layers)
}
