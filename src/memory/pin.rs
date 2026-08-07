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
#![cfg(feature = "rocm")]

use crate::artifact::config::Mode;
use crate::artifact::format::{
    Dtype, ExpertSet, FormatMeta, RoutedFmt, Safetensors, SetDims, f4_layer_range, load_codebooks,
};
use crate::artifact::model::{ModelConfig, V4Config};
use crate::artifact::quant::{
    VQ_DIM, VQ_K, i4_expert_bytes, i4_slot_offsets, vq_expert_bytes, vq_slot_offsets,
};
use crate::memory::cache;
use crate::memory::device::DeviceTier;
use crate::memory::routed::{ExpertSlot, MAX_BATCH, RoutedPool, TierFmt, pool_budget, slot_at};
use anyhow::{Context, Result, ensure};

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

/// One layer's MLP: dense fp8 for the first `dense_layers`, MoE after. The MoE
/// shared expert is the routed format (VQ-int3, or int4 under `--i4`; from block
/// `n_experts`), folded into the single MoE batch alongside the routed picks.
pub enum LayerMlp {
    Dense(Fp8Mlp),
    Moe {
        gate_w: *const f32, // router gate [n_experts, hidden] (F32), device
        shared: ExpertSlot, // routed-format shared expert (always resident)
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
    /// The routed-expert pool: residency, eviction, relocation, the io_uring cold reads
    /// and the `--trace` sink. Public because `gpu.rs` drives it directly — a set of
    /// forwarding methods on `Pin` was the alternative and it added a second name for
    /// every one of them. See [`crate::memory::routed`].
    pub routed: RoutedPool,
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
fn place_shared(tier: &mut DeviceTier, block: &[u8], off: &[usize; 6]) -> Result<ExpertSlot> {
    let dst = tier.place(block)?;
    // SAFETY: `off` are this format's slot offsets and `block` is one whole expert block,
    // so every offset lies inside the reservation `place` just made.
    Ok(unsafe { slot_at(dst, off) })
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

impl<'a> Pin<'a> {
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
        // One-time bound for `RoutedPool::submit`'s fixed `MAX_BATCH` hit scratch. A batched forward
        // submits the UNION of every token row's picks, so the worst case is
        // `top_k · MAXROW + n_shared`, not one row's `experts_per_layer()`.
        let max_batch = cfg.top_k * crate::gpu::MAXROW + cfg.n_shared;
        ensure!(
            max_batch <= MAX_BATCH,
            "top_k {} x {} rows + n_shared {} = {max_batch} exceeds the {MAX_BATCH}-slot batch scratch",
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
        // A tier descriptor per format. Everything but WHICH SET comes off the set itself
        // (format, slot offsets, stride, layer range), so the two arms differ in one word.
        let tier_fmt = |src: &Option<ExpertSet>, what: &str| -> Result<TierFmt> {
            TierFmt::new(
                src.as_ref()
                    .with_context(|| format!("{what} source missing"))?,
            )
        };
        // COLD/HOT tiers by mode. Single-format shares one format across both tiers.
        let (cold, hot) = match mode {
            Mode::Int3Vq => {
                let t = tier_fmt(&vq_src, "vq3")?;
                (t.clone(), t)
            }
            Mode::Int4 => {
                let t = tier_fmt(&i4_src, "i4")?;
                (t.clone(), t)
            }
            Mode::Hybrid => (tier_fmt(&vq_src, "vq3")?, tier_fmt(&i4_src, "i4")?),
        };
        let budget = pool_budget(capacity, tier_cap);
        let routed = RoutedPool::new(
            budget,
            cfg.top_k,
            // GLM's batch is the UNION of every row's picks, not one row's — checked against
            // `MAX_BATCH` above and passed on so the pool's budget floor sizes to the same
            // number rather than to `top_k`.
            max_batch,
            cache_policy,
            two_q,
            trace_path,
            cold,
            hot,
        )?;

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
    /// gpu.rs appends the folded shared expert with this format; routed experts carry
    /// their own per-expert one from [`RoutedPool::submit`].
    pub fn shared_fmt(&self) -> RoutedFmt {
        if self.shared_i4 {
            RoutedFmt::I4
        } else {
            RoutedFmt::Vq3
        }
    }
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
// The routed FP4 streaming pool IS owned here now (`V4Pin::routed`), over the same
// [`crate::memory::routed::RoutedPool`] GLM's `Pin` uses — see that module for why it is
// shared and what was verified byte-parameterised before sharing it.
//
// What this does NOT own, and why:
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
    ///
    /// Held for the run: [`Self::routed`]'s read table is `(fd, begin, len)` triples whose
    /// fds these `File`s own, so dropping this closes them under a live pool.
    pub f4: ExpertSet,
    /// The routed FP4 streaming pool. **Not optional, and that is the size argument**: the
    /// shipped 43-layer `.f4` set is 137 GiB against a ~115 GiB budget, so unlike GLM's
    /// (~41% residency) V4's routed experts cannot all be resident on any configuration
    /// this machine has. A `V4Pin` without a pool cannot run the model at all.
    pub routed: RoutedPool,
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
    // nothing else — `TensorDesc.len` comes from the header's `data_offsets` and is never
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
    /// Build the V4 resident set and its `.f4` streaming pool from artifact directory
    /// `dir`. `capacity` is the total device budget: the resident set takes the artifact's
    /// own `*.safetensors` size and everything left grows the pool.
    ///
    /// `capacity` was DECLINED here while the pool did not exist, on the argument that a
    /// parameter documenting a policy that does not run is worse than none. It runs now.
    pub fn build(
        dir: &str,
        cfg: &V4Config,
        capacity: usize,
        cache_policy: &str,
        two_q: cache::TwoQSplit,
        trace_path: Option<&str>,
    ) -> Result<Self> {
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
        // under 0.5 MB. 16 MiB is ~30x that bound.
        //
        // **The reason it was loose has changed and the looseness has not.** It used to be
        // free ("an over-count only costs unused tier; there is no routed pool competing for
        // the remainder yet") — the pool exists now, and every over-counted byte comes
        // straight out of `pool_budget(capacity, tier_cap)` below. Still 16 MiB, because
        // that is 0.015% of a ~106 GiB pool and an UNDER-count is what `DeviceTier::place`
        // bails on; but it is a trade now, not a freebie.
        const SLACK: usize = 16 << 20;
        let resident = safetensors_bytes(dir, None)?;
        let tier_cap = resident + SLACK;
        let pool = pool_budget(capacity, tier_cap);
        // The routed set the pool has to carry: 43 layers x 256 experts x 13.37 MB =
        // 137 GiB, against ~115 GiB of budget on this machine. **A log line, not a check** —
        // it cannot fit and the streaming is the point, so there is nothing here to refuse.
        // It is printed because "77% residency" is the number that explains every later
        // measurement, and a run whose parameters are not in its log never happened.
        // `RoutedPool::new` makes the one refusal that is meaningful: a budget too small for
        // a single layer's demand.
        let routed_total = (range.len() * cfg.n_experts) as u64
            * crate::artifact::quant::f4_expert_stride(cfg.hidden, cfg.moe_inter) as u64;
        tracing::info!(
            "v4 resident footprint {:.2} GiB over layers [{}, {}); routed set {:.2} GiB, \
             pool budget {:.2} GiB ({:.1}% residency)",
            resident as f64 / (1u64 << 30) as f64,
            range.start,
            range.end,
            routed_total as f64 / (1u64 << 30) as f64,
            pool as f64 / (1u64 << 30) as f64,
            100.0 * pool as f64 / routed_total as f64,
        );
        let mut tier = DeviceTier::new(tier_cap)?;

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

        // The pool, over the ONE `.f4` set. Single-format: `cold` and `hot` are the same
        // tier, as they are in GLM's `int3-vq`/`int4` modes — there is no second FP4
        // container to pair `.f4` with, so the arena's floating split never has to relocate
        // between different strides here.
        let tier_fmt = TierFmt::new(&f4)?;
        // `top_k` and not a `MAXROW` union: V4's FP4 MoE kernel refuses `nrow != 1`
        // (`kernels/moe.hip:409`), so a V4 decode is structurally single-row — and there is
        // no folded shared expert either, because V4's is fp8 and RESIDENT rather than a
        // routed-format block.
        let routed = RoutedPool::new(
            pool,
            cfg.top_k,
            // Batch == `top_k`: single-row, and V4's shared expert is fp8 and resident rather
            // than a routed-format block folded into the batch.
            cfg.top_k,
            cache_policy,
            two_q,
            trace_path,
            tier_fmt.clone(),
            tier_fmt,
        )?;

        Ok(Self {
            tier,
            embed,
            head,
            final_norm,
            hc_head,
            layers,
            range,
            f4,
            routed,
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
