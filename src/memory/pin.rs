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
use crate::artifact::model::{
    GLIMMER_LAYER_PREFIX, GLIMMER_LAYER_TENSORS, GlimmerFormat, GlimmerTextConfig, ModelConfig,
    V4Config,
};
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
///
/// **The SCALE GRID's shape is checked, and until 2026-08-15 it was the only fp8 reader in this
/// tree that did not check it.** `Safetensors::dequant_fp8` and [`place_fp8_qkv`] both do, and
/// both say why in nearly the same words: a grid of the wrong extent mis-tiles SILENTLY, which is
/// a wrong-but-plausible dequant rather than a failure. The kernel indexes
/// `scale[(o/block)·sc_cols + i/block]` with `sc_cols` derived from `i_dim` — so a grid stored
/// `[sc_cols, sc_rows]` has every tile taking a neighbour's scale (fluent wrong text, no error),
/// and a SHORTER grid has the kernel reading past the placement into the next tensor's e4m3 bytes
/// reinterpreted as f32.
///
/// Found by review on the Glimmer fp8 path, where the shape check the same commit added ran only
/// for STREAMED layers — and the shipping partition on this host pins all 52, so it iterated zero
/// times. Fixed here rather than there because it is every model's hole, not Glimmer's.
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
        "{name}.weight_scale_inv is {sshape:?}, but a [{o_dim}, {i_dim}] weight at block {block} \
         implies {want:?} — a grid of the wrong extent mis-tiles silently"
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

/// Place a V4 layer's `wkv` and `wq_a` as ONE fp8 placement — `wkv`'s rows FIRST, then
/// `wq_a`'s, weight bytes and scale grids each concatenated row-wise — and resolve the
/// two original weights as VIEWS into it (offsets, zero extra bytes), plus the whole
/// `[head_dim + q_lora_rank, dim]` concat the M10 fused decode GEMV reads
/// (`attn::v4::Weights::wqkv` carries the fusion's argument; M8's serial table carries
/// the reason — both split launches sit grid-starved at 66–67 GB/s).
///
/// kv first because the fused output row must land the KV entry at `s.kv`'s base,
/// where every consumer already looks. The seam is scale-exact iff `wkv`'s row count
/// is a multiple of `block` — the kernel's scale row is `j >> log2(block)`, so row
/// `head_dim + r` reads concatenated scale row `head_dim/block + r/block` exactly when
/// nothing straddles. A checkpoint where it does not divide gets two ordinary
/// placements and NO fused weight, and decode runs the two-launch path unchanged.
fn place_fp8_qkv(
    tier: &mut DeviceTier,
    st: &Safetensors,
    attn: &str,
    block: usize,
) -> Result<(Fp8Weight, Fp8Weight, Option<Fp8Weight>)> {
    let mut parts = Vec::with_capacity(2);
    for name in ["wkv", "wq_a"] {
        let wname = format!("{attn}.{name}.weight");
        let (w, shape) = st.typed(&wname, Dtype::F8E4M3)?;
        let (sc, _) = st.typed(&format!("{attn}.{name}.weight_scale_inv"), Dtype::F32)?;
        let (o, i) = dims2(&wname, shape)?;
        parts.push((w, sc, o, i));
    }
    let [(kw, ks, ko, ki), (qw, qs, qo, qi)] = parts[..] else {
        unreachable!("two names pushed above");
    };
    if ki != qi || !ko.is_multiple_of(block) {
        return Ok((
            place_fp8(tier, st, &format!("{attn}.wkv"), block)?,
            place_fp8(tier, st, &format!("{attn}.wq_a"), block)?,
            None,
        ));
    }
    // The grids must BE the `ceil(o/block) × ceil(i/block)` f32 the kernel indexes, or
    // the row-wise concat below would misattribute scale rows silently.
    let cols = ki.div_ceil(block);
    for (sc, o, what) in [(ks, ko, "wkv"), (qs, qo, "wq_a")] {
        ensure!(
            sc.len() == o.div_ceil(block) * cols * size_of::<f32>(),
            "{attn}.{what}.weight_scale_inv is {} bytes, expected [{}, {cols}] f32 = {}",
            sc.len(),
            o.div_ceil(block),
            o.div_ceil(block) * cols * size_of::<f32>()
        );
    }
    let packed = tier.place(&[kw, qw].concat())?;
    let scale = tier.place(&[ks, qs].concat())? as *const f32;
    // SAFETY: both offsets are inside the two placements above — `wq_a`'s rows start
    // `ko` weight rows (`ko/block` scale rows, exact by the divisibility gate) in.
    let (a_packed, a_scale) = unsafe { (packed.add(ko * ki), scale.add((ko / block) * cols)) };
    let view = |packed, scale, o_dim| Fp8Weight {
        packed,
        scale,
        block,
        o_dim,
        i_dim: ki,
    };
    Ok((
        view(packed, scale, ko),
        view(a_packed, a_scale, qo),
        Some(view(packed, scale, ko + qo)),
    ))
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
/// Shared by [`resident_bytes`] (GLM) and [`F4Pin::build`], and the reason is the one
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
// The routed FP4 streaming pool IS owned here now (`F4Pin::routed`), over the same
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
pub struct CompressorWeights {
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
pub struct F4IndexerPin {
    pub wq_b: Fp8Weight,
    pub weights_proj: *const f32,
    pub compressor: CompressorWeights,
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
pub enum GateRoute {
    /// `tid2eid[token * top_k + j]` — already range-checked, see [`parse_tid2eid`].
    Hash { tid2eid: Vec<u32> },
    /// The router correction bias added before top-k, `n_experts` long.
    Scored { bias: Vec<f32> },
}

/// One V4 layer's resident weights.
pub struct F4LayerPin {
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
    /// `[wkv ‖ wq_a]` as one `[head_dim + q_lora_rank, hidden]` weight — `wkv` and
    /// `wq_a` above are VIEWS into this placement, so it costs no bytes. `Some`
    /// whenever the seam divides the scale block (this checkpoint: 512 % 128 == 0);
    /// the engine passes it through as `attn::v4::Weights::wqkv`, the M10 decode
    /// width fusion. See [`place_fp8_qkv`].
    pub wqkv: Option<Fp8Weight>,
    pub wo_a: Fp8Weight,
    pub wo_b: Fp8Weight,
    /// Router gate `[n_experts, hidden]` f32, device-side (the scores are a GEMV).
    pub gate_w: *const f32,
    pub route: GateRoute,
    pub hc_attn: HyperConn,
    pub hc_ffn: HyperConn,
    /// The always-on shared expert — fp8 e4m3, resident, NOT in the `.f4`.
    pub shared: Fp8Mlp,
    /// Present iff `compress_ratio != 0`.
    pub compressor: Option<CompressorWeights>,
    /// Present iff `compress_ratio == 4`.
    pub indexer: Option<F4IndexerPin>,
}

/// The V4 resident weight set, plus the validated `.f4` routed-expert source.
pub struct F4Pin {
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
    layers: Vec<F4LayerPin>,
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
    /// this machine has. A `F4Pin` without a pool cannot run the model at all.
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
fn place_compressor(
    tier: &mut DeviceTier,
    st: &Safetensors,
    base: &str,
) -> Result<CompressorWeights> {
    Ok(CompressorWeights {
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
    // returned table is SHORT. `GateRoute::Hash`'s consumer indexes it at
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

impl F4Pin {
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
                GateRoute::Hash {
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
                GateRoute::Scored { bias }
            };
            let mut fp8 = |name: &str| place_fp8(&mut tier, &st, &format!("{a}.{name}"), block);
            let (wq_b, wo_a, wo_b) = (fp8("wq_b")?, fp8("wo_a")?, fp8("wo_b")?);
            // wkv + wq_a go through the concat placer: one placement, three views (M10).
            let (wkv, wq_a, wqkv) = place_fp8_qkv(&mut tier, &st, &a, block)?;
            // The concat seams at the TENSOR's row count while `attention`'s fused
            // landing offset is the CONFIG's `head_dim` — a checkpoint where they
            // disagree would put the q intermediate at the wrong offset silently (the
            // split path is equally wrong under such a mismatch, but wrongly
            // DIMENSIONED, which fails louder). One load-time check closes the fused
            // expression of it; review-surfaced, and load-time because the in-path
            // `debug_assert!` is compiled out exactly when benchmarks run.
            if wqkv.is_some() {
                ensure!(
                    wkv.o_dim == cfg.head_dim && wq_a.o_dim == cfg.q_lora_rank,
                    "layer {l}: wkv/wq_a rows ({}, {}) disagree with the config's \
                     head_dim/q_lora_rank ({}, {}) — the fused qkv landing offset would be wrong",
                    wkv.o_dim,
                    wq_a.o_dim,
                    cfg.head_dim,
                    cfg.q_lora_rank
                );
            }
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
            layers.push(F4LayerPin {
                attn_norm: place_f32(&mut tier, &st, &format!("{lb}.attn_norm.weight"))?,
                ffn_norm: place_f32(&mut tier, &st, &format!("{lb}.ffn_norm.weight"))?,
                q_norm: place_f32(&mut tier, &st, &format!("{a}.q_norm.weight"))?,
                kv_norm: place_f32(&mut tier, &st, &format!("{a}.kv_norm.weight"))?,
                attn_sink: place_f32(&mut tier, &st, &format!("{a}.attn_sink"))?,
                wq_a,
                wq_b,
                wkv,
                wqkv,
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
                    .then(|| -> Result<F4IndexerPin> {
                        Ok(F4IndexerPin {
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
        // (`kernels/moe.hip`), so a V4 decode is structurally single-row — and there is
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
    pub fn layer(&self, l: usize) -> Result<&F4LayerPin> {
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

// ---------------------------------------------------------------------------------------
// Muse Glimmer. Dense, so this is the whole model — there is no second half.
// ---------------------------------------------------------------------------------------

/// One Muse Glimmer decoder layer's resident weights.
///
/// **One layer, resolved — whether the budget pinned it or a slot holds it.**
///
/// > This said "Glimmer is dense, so there is nothing to stream, no pool, no cache policy and
/// > no residency decision. The entire model is this struct, 52 times." R1 replaced that
/// > premise (see [`GlimmerPin`]) and review found this copy of it surviving eight lines above
/// > the replacement, 2026-08-12. [`Pin`] and [`F4Pin`] split a layer into a resident part and
/// > a routed part because 256 experts do not fit; Glimmer splits at WHOLE LAYERS because 52
/// > of them do not fit either.
///
/// Twelve tensors, matching [`crate::artifact::model::GLIMMER_LAYER_TENSORS`]: four norms
/// (widened to f32 by the converter) and eight bf16 projections. **Five projections in the
/// attention block, not four** — `self_attn.gate_proj` is a per-head output gate applied
/// before `o_proj`, it is the same shape as `q_proj`, and it is the one an HF-shaped port
/// drops on the floor.
/// **Deliberately NOT `Copy`, unlike every other pin struct here** — but that buys less than
/// the first version of this note claimed, and the gap matters.
///
/// A copy of a streamed layer's addresses stays valid-looking after its slot has been refilled
/// with another layer — the read-outlives-its-slot shape still open on the GLM arena path — and
/// borrowing from [`GlimmerPin::layer`] (`&mut self`) makes holding the whole struct across a
/// refill a compile error. **What it does NOT stop is extracting a field:** [`Bf16Weight`] is
/// `Copy` and the four norms are bare pointers, so `let q = pin.layer(5)?.q;` yields a handle
/// that outlives the borrow, and `tests/glimmer_residency.rs::tensors_of` does exactly that.
/// Review found this 2026-08-12. The type narrows the mistake; it does not forbid it, and the
/// invariant on [`GlimmerPin::layer`] is what a caller actually has to honour.
/// One Glimmer projection, in whichever format the artifact stores — see [`GlimmerFormat`].
///
/// **A sum type rather than a second `GlimmerLayerPin`.** Every other property of the layer is
/// format-independent: the same twelve tensors, the same shapes, the same order, the same
/// streaming slot. Only the arithmetic differs, and [`Self::dims`] is what lets the shape checks
/// stay written once for both.
#[derive(Clone, Copy)]
pub enum GlimmerProj {
    Bf16(Bf16Weight),
    Fp8(Fp8Weight),
}

impl GlimmerProj {
    /// `[o_dim, i_dim]`, whichever variant this is. Both carry them, and every caller that
    /// checks a shape or launches a GEMV wants them rather than the format.
    pub fn dims(&self) -> [usize; 2] {
        match self {
            Self::Bf16(w) => [w.o_dim, w.i_dim],
            Self::Fp8(w) => [w.o_dim, w.i_dim],
        }
    }

    /// Append the device addresses this placement occupies, in the order
    /// [`glimmer_tensor_tails`] emits their names — one for bf16, weight-then-scale for fp8.
    ///
    /// Appends rather than returning, so a whole layer's twenty addresses cost ONE allocation
    /// instead of nine: `GlimmerLayerPin::addrs` runs on every streamed-layer refill, which is
    /// per layer per token.
    fn push_addrs(&self, v: &mut Vec<*const u8>) {
        match self {
            Self::Bf16(w) => v.push(w.packed as *const u8),
            Self::Fp8(w) => v.extend([w.packed, w.scale as *const u8]),
        }
    }
}

pub struct GlimmerLayerPin {
    /// Pre-attention, `rms_norm_eps`.
    pub input_ln: *const f32,
    /// **Post-attention, `post_norm_eps` — a different eps, by three orders of magnitude.**
    /// The two post-norms sit on the BRANCH, before the residual add (sandwich norms), which
    /// is why they are separate fields rather than a `[*const f32; 4]` a loop indexes.
    pub post_attn_ln: *const f32,
    /// Pre-MLP, `rms_norm_eps`.
    pub pre_ffn_ln: *const f32,
    /// Post-MLP, `post_norm_eps`.
    pub post_ffn_ln: *const f32,
    /// `[n_heads·head_dim, hidden]` = `[4096, 6656]`. **Not square**, and not derivable from
    /// `hidden / n_heads`.
    pub q: GlimmerProj,
    /// `[kv_heads·head_dim, hidden]` = `[256, 6656]`. GQA at 16 query heads per KV head.
    pub k: GlimmerProj,
    /// Same shape as [`Self::k`], and separable from it only by NAME.
    pub v: GlimmerProj,
    /// `[hidden, n_heads·head_dim]` — the transposed one.
    pub o: GlimmerProj,
    /// `self_attn.gate_proj`, same shape as [`Self::q`] and likewise separable only by name.
    pub attn_gate: GlimmerProj,
    pub mlp_gate: GlimmerProj,
    pub mlp_up: GlimmerProj,
    /// `[hidden, inter]` — the other transposed one.
    pub mlp_down: GlimmerProj,
}

impl GlimmerLayerPin {
    /// The eight projections in [`GLIMMER_LAYER_TENSORS`] order — the four norms come first
    /// there, so this is the tail of that list.
    ///
    /// Exists so [`Self::addrs`] and the format checks iterate one list instead of restating
    /// eight field names each; the four norms stay spelled out because they are four different
    /// eps positions and a loop over them would hide that.
    fn projs(&self) -> [GlimmerProj; 8] {
        [
            self.q,
            self.k,
            self.v,
            self.o,
            self.attn_gate,
            self.mlp_gate,
            self.mlp_up,
            self.mlp_down,
        ]
    }

    /// This layer's device addresses, in [`GLIMMER_LAYER_TENSORS`] order — **twelve at bf16,
    /// twenty at fp8**, since each fp8 projection contributes its weight and then its scale grid.
    ///
    /// Exists so [`Slot::fill`] can write to each tensor's own address instead of computing an
    /// offset. A permutation here would make a streamed layer a permutation of a pinned one —
    /// every tensor the right shape, the model silently wrong.
    ///
    /// **This and [`glimmer_layer_names`] must agree element-for-element, and they do NOT derive
    /// it from the same place — this doc claimed they did until 2026-08-15.** That function walks
    /// [`GLIMMER_LAYER_TENSORS`] and asks [`GlimmerTextConfig::layer_tensor_shape`] which entries
    /// are 2-D; this one walks a hand-written list of four norms and then [`Self::projs`]. Two
    /// procedures, one required answer. [`Slot::new`]'s `ensure!` compares only their LENGTHS,
    /// which is arithmetically incapable of failing today (both are `12 + 8·[fp8]`) and would
    /// catch only a future edit that changed the 2-D count in one place and not the other. **What
    /// actually gates the permutation is
    /// `glimmer_residency.rs::every_budget_resolves_every_layer_to_the_same_bytes_at_fp8`**, which
    /// is red-proved by swapping [`GlimmerProj::addrs`]'s weight-then-scale order — and reddens
    /// there while the bf16 sweep beside it stays green, because at bf16 there is nothing
    /// interleaved to permute.
    ///
    /// **Nothing asserts the order directly, and this doc claimed otherwise until 2026-08-12**
    /// ("asserted against that constant by `glimmer_residency.rs`" — that file never names
    /// either). What DOES cover it is transitive: the fixture gives every tensor distinct bytes,
    /// so any swap of two same-length entries reddens the byte-identity gate, and a swap of
    /// different-length entries fails `Slot::fill`'s length check. Weaker than the assertion the
    /// doc promised, and worth knowing which one you have.
    fn addrs(&self) -> Vec<*const u8> {
        let mut a = vec![
            self.input_ln as *const u8,
            self.post_attn_ln as *const u8,
            self.pre_ffn_ln as *const u8,
            self.post_ffn_ln as *const u8,
        ];
        for p in self.projs() {
            p.push_addrs(&mut a);
        }
        a
    }
}

use crate::artifact::model::GLIMMER_STREAM_SLOTS;

/// The Muse Glimmer weight set: what the budget pinned, plus slots for the rest.
///
/// > **REPLACED 2026-08-12.** This was "the resident weight set. **All of it**", on the
/// > argument that "Glimmer is dense, so there is nothing to stream, no pool, no cache policy
/// > and no residency decision." That inverts `reference/principles.md` **P6** — the pin is a
/// > function of free memory at run time, never of model architecture — and it is wrong about
/// > dense models specifically: every weight is read every token (53.02 GB bf16, 26.51 fp8,
/// > 13.65 int4), so there is no routed union to hide behind and the resident FRACTION is the
/// > whole tok/s story. 55.7 GB placed unconditionally is not a model that runs beside a 1.7
/// > GB KV cache, a 2.6 GB drafter, or another tenant. `investigations/glimmer-integration.md`
/// > §R1.
///
/// Layers `0..pinned.len()` live in the tier for the run; the rest cycle through
/// [`GLIMMER_STREAM_SLOTS`] slots. **Which one a caller got is not observable** — [`Self::layer`] returns
/// the same `GlimmerLayerPin` shape either way, and `glimmer_residency.rs` gates that the
/// BYTES behind it are identical at every budget. That indistinguishability is P4 (the budget
/// trades speed, never text) expressed as a type rather than as a convention.
pub struct GlimmerPin {
    #[allow(dead_code)] // RAII owner of the slab every pointer below points into.
    tier: DeviceTier,
    /// The mmap'd artifact, kept alive because a streamed layer is read from it on every
    /// visit. An all-resident pin holds it too and never reads it again — one field rather
    /// than an `Option` nothing checks.
    src: Safetensors,
    /// `[vocab, hidden]` bf16, 2.690 GB.
    pub embed: Bf16Weight,
    /// `lm_head.weight`, `[vocab, hidden]` bf16 — a second 2.690 GB, because
    /// `tie_word_embeddings` is false and both tensors ship.
    pub head: Bf16Weight,
    pub final_norm: *const f32,
    /// Layers the budget pinned, indexed by ABSOLUTE layer id — a PREFIX, so the index is
    /// the layer id with no offset to get wrong. `convert_glimmer` refuses a checkpoint
    /// missing any layer's tensors, so a gap here can only come from the budget.
    pinned: Vec<GlimmerLayerPin>,
    /// The streaming slots: fixed device addresses, refilled per visit. Empty when the budget
    /// pinned everything, which is why an all-resident run allocates none.
    slots: Vec<Slot>,
    /// Which layer each slot currently holds, so a re-visit inside one slot's lifetime is a
    /// hit rather than a second copy of the same bytes.
    slot_layer: Vec<Option<usize>>,
    /// The model's layer count, carried because `pinned.len()` is now a partition rather than
    /// the model — without it, "is `l` a real layer?" would be unanswerable here.
    n_layers: usize,
    /// Cheap counters. Not a policy input — see [`Self::layer`] on why there is no policy —
    /// but the only way to tell a partition that is working from one that thrashes.
    hits: u64,
    fills: u64,
}

/// One streaming slot: a layer-sized region of the tier, with the twelve device addresses
/// inside it precomputed.
///
/// **The addresses are computed ONCE and never move.** A fill overwrites bytes at fixed
/// offsets; it does not re-place anything. That is what lets `layer()` hand out a
/// `GlimmerLayerPin` whose pointers are stable for as long as the slot holds that layer, and
/// it is the difference between this and the GLM arena, whose compaction can move a slot out
/// from under an in-flight read (`docs` — arena relocation, still open there).
struct Slot {
    pin: GlimmerLayerPin,
    /// Each tensor's NAME TAIL and byte length, in [`GLIMMER_LAYER_TENSORS`] order and as long as
    /// `pin.addrs()` — the extent of the placement the matching address points at. A refill checks
    /// the incoming tensor against the length, which is what bounds the write to its own
    /// placement. `Vec` rather than a fixed array because an fp8 layer carries twenty entries to a
    /// bf16 layer's twelve.
    ///
    /// **The tails are stored, not re-derived.** They are layer-independent, so `Slot::fill` only
    /// prepends `{GLIMMER_LAYER_PREFIX}.{l}.`; the alternative had every refill rebuild the whole
    /// list through `layer_tensor_shape`, which is why `GlimmerPin` carried a cloned
    /// `GlimmerTextConfig` and a `fmt` field solely to feed it. Both reviews found that
    /// independently, 2026-08-15, one as a hot-path allocation and one as a field whose doc was
    /// three lines apologising for its own existence.
    lens: Vec<(String, usize)>,
}

/// Place one Glimmer f32 norm, with its LENGTH checked against `hidden`.
///
/// **The check is the whole reason this exists rather than a bare [`place_f32`] call.**
/// `place_f32` discards the shape, and `GlimmerLayerPin`'s norm fields are bare `*const f32`
/// carrying no extent — so a norm shorter than `hidden` is accepted, sized into a tier that
/// has room to spare (`resident_bytes` budgeted the full width), and handed to S2's RMSNorm as
/// a `hidden`-long array. It then reads inter-placement padding and the next tensor's bytes:
/// a scaled-wrong residual stream, in bounds of the slab, with no error anywhere. That is the
/// same class `place_glimmer_proj` guards against, and the five norms were the only placements
/// on this path with nothing at all. Found by two independent reviews, 2026-08-11.
fn place_glimmer_norm(
    tier: &mut DeviceTier,
    st: &Safetensors,
    hidden: usize,
    name: &str,
) -> Result<*const f32> {
    let (bytes, shape) = st.typed(name, Dtype::F32)?;
    ensure!(
        shape == [hidden],
        "{name} is {shape:?}, but this config implies [{hidden}] — the artifact and the config \
         describe different models"
    );
    Ok(tier.place(bytes)? as *const f32)
}

/// Place one Glimmer layer projection in `fmt`, with the shape the config implies.
///
/// The shape comes from [`GlimmerTextConfig::layer_tensor_shape`] rather than from the caller,
/// so the twelve call sites below cannot each get it slightly wrong, and so the one table
/// `tests/glimmer_names.rs` validates against the shipped checkpoint is the one used here.
///
/// **The dtype is not a parameter to get wrong either** — `place_bf16` and `place_fp8` each go
/// through `Safetensors::typed`, which refuses any other dtype. So an fp8 artifact loaded as bf16
/// fails on the first projection with a dtype error rather than reading e4m3 bytes as bf16 halves,
/// which would be fluent wrong text at exactly half the expected magnitude.
fn place_glimmer_proj(
    tier: &mut DeviceTier,
    src: &LayerSrc,
    prefix: &str,
    tensor: &str,
) -> Result<GlimmerProj> {
    let want = src.cfg.layer_tensor_shape(tensor)?;
    let name = format!("{prefix}.{tensor}");
    let w = match src.fmt {
        GlimmerFormat::Bf16 => {
            GlimmerProj::Bf16(place_bf16(tier, src.st, &format!("{name}.weight"))?)
        }
        GlimmerFormat::Fp8 => GlimmerProj::Fp8(place_fp8(
            tier,
            src.st,
            &name,
            crate::artifact::quant::FP8_BLOCK,
        )?),
    };
    // Checked AFTER placing, which is safe and not merely tolerable: the tier is sized from
    // `resident_bytes`, so a tensor larger than its declared shape bails inside `place` on
    // capacity instead, and either way the pin refuses before a decode ever runs. Checking
    // first would mean reading the header twice for the sake of which error message fires.
    let got = w.dims();
    ensure!(
        got == want[..],
        "{name}.weight is {got:?}, but this config implies {want:?} — the \
         artifact and the config describe different models"
    );
    Ok(w)
}

/// One layer's tensor names, in the order [`GLIMMER_LAYER_TENSORS`] declares them, with an fp8
/// artifact's scale grids interleaved after the weights they scale.
///
/// The prefix is spelled once here rather than at each of the two call sites (pinned
/// placement and slot layout), because the two must agree tensor-for-tensor or a streamed
/// layer would be a permutation of a pinned one — and a permutation of correctly-shaped
/// tensors is exactly the silent wrongness this port keeps finding.
///
/// **Which entries take a scale is decided by `layer_tensor_shape`**, the same table
/// `layer_bytes` consults to decide whether an entry is a norm or a projection — not by a second
/// reading of the name. A `t.ends_with("layernorm")` here would be a third spelling of that split,
/// and the one that silently disagrees is the one that costs a debugging session.
fn glimmer_tensor_tails(src: &LayerSrc) -> Result<Vec<String>> {
    let mut v = Vec::with_capacity(GLIMMER_LAYER_TENSORS.len());
    for t in GLIMMER_LAYER_TENSORS {
        v.push(format!("{t}.weight"));
        if src.fmt == GlimmerFormat::Fp8 && src.cfg.layer_tensor_shape(t)?.len() == 2 {
            v.push(format!("{t}.weight_scale_inv"));
        }
    }
    Ok(v)
}

/// Where one Glimmer layer's tensors come from: the open artifact, the config that says what shape
/// each of them should be, and the format they are stored in.
///
/// **One struct because the three are never apart on the PLACEMENT path.** `place_glimmer_layer`,
/// `place_glimmer_proj` and `Slot::new` each need all three, and once `fmt` joined the list on
/// 2026-08-14 their parameter lists became a 39-token jscpd clone of each other. This is the
/// factoring that removes the text rather than an exemption that hides it — `build.rs`'s own note
/// is that the gate did not author what it found.
///
/// **`Slot::fill` is deliberately NOT on that list**, though it was until 2026-08-15. A refill
/// needs only the artifact and a layer index, because `Slot` stores the name tails it computed at
/// layout; making it take a `LayerSrc` too is what forced `GlimmerPin` to carry a cloned
/// `GlimmerTextConfig` for the sake of one `layer_tensor_shape` call per miss.
struct LayerSrc<'a> {
    st: &'a Safetensors,
    cfg: &'a GlimmerTextConfig,
    fmt: GlimmerFormat,
}

/// Place one whole layer into the tier — the pinned path.
///
/// Extracted from `GlimmerPin::build`'s loop so [`Slot::new`] can build the same twelve-field
/// pin over slot-relative addresses. Both go through [`place_glimmer_norm`] and
/// [`place_glimmer_proj`], so both keep the extent and shape checks that two reviews added on
/// 2026-08-11.
fn place_glimmer_layer(tier: &mut DeviceTier, src: &LayerSrc, l: usize) -> Result<GlimmerLayerPin> {
    let p = format!("{GLIMMER_LAYER_PREFIX}.{l}");
    let norm = |tier: &mut DeviceTier, t: &str| {
        place_glimmer_norm(tier, src.st, src.cfg.hidden, &format!("{p}.{t}"))
    };
    let proj = |tier: &mut DeviceTier, t: &str| place_glimmer_proj(tier, src, &p, t);
    Ok(GlimmerLayerPin {
        input_ln: norm(tier, "input_layernorm.weight")?,
        post_attn_ln: norm(tier, "post_attention_layernorm.weight")?,
        pre_ffn_ln: norm(tier, "pre_feedforward_layernorm.weight")?,
        post_ffn_ln: norm(tier, "post_feedforward_layernorm.weight")?,
        q: proj(tier, "self_attn.q_proj")?,
        k: proj(tier, "self_attn.k_proj")?,
        v: proj(tier, "self_attn.v_proj")?,
        o: proj(tier, "self_attn.o_proj")?,
        attn_gate: proj(tier, "self_attn.gate_proj")?,
        mlp_gate: proj(tier, "mlp.gate_proj")?,
        mlp_up: proj(tier, "mlp.up_proj")?,
        mlp_down: proj(tier, "mlp.down_proj")?,
    })
}

impl Slot {
    /// Reserve a layer-sized region and precompute the twelve addresses inside it.
    ///
    /// **Built by placing layer 0 and then recording where each tensor landed.** That reuses
    /// the pinned path's shape and extent checks instead of restating a layout — the
    /// alternative was a second offset table, which is a second thing to get wrong and which
    /// jscpd would have matched against the first. The bytes placed here are layer 0's and
    /// are overwritten by the first [`Self::fill`]; what survives is the geometry.
    fn new(tier: &mut DeviceTier, src: &LayerSrc) -> Result<Self> {
        let pin = place_glimmer_layer(tier, src, 0)?;
        // **A refill's destination is the PIN'S OWN ADDRESS — there is no offset arithmetic at
        // all.** Two earlier shapes were worse. The first recomputed `next_multiple_of(256)` per
        // tensor to predict where `place` had put things: a second copy of `bump`, correct only
        // while it agreed. The second stored `base = addrs[0]` plus per-tensor offsets obtained
        // by subtracting pointers — which worked, but rested on the first placement being the
        // LOWEST address (true for a bump allocator, promised by nothing), and would have
        // underflowed into `base.add(huge)` if that ever changed, silently under `--release`
        // where overflow checks are off. Reviews found both, 2026-08-12.
        //
        // Writing to each address directly needs no ordering assumption and no subtraction, and
        // each write is bounded by the extent of the placement it targets — which is what `lens`
        // records and `fill` enforces.
        let tails = glimmer_tensor_tails(src)?;
        // The two lists are built from the same `fmt` and must come out the same length; a
        // mismatch would silently truncate the shorter side in `fill`'s `zip` and leave whatever
        // tensor fell off the end holding layer 0's bytes for the whole run.
        ensure!(
            tails.len() == pin.addrs().len(),
            "slot layout: {} names against {} placements",
            tails.len(),
            pin.addrs().len()
        );
        let mut lens = Vec::with_capacity(tails.len());
        for tail in tails {
            let n = src
                .st
                .raw(&format!("{GLIMMER_LAYER_PREFIX}.0.{tail}"))?
                .0
                .len();
            lens.push((tail, n));
        }
        Ok(Self { pin, lens })
    }

    /// Overwrite this slot with layer `l`'s bytes.
    ///
    /// A host `memcpy` per tensor, from the mmap'd artifact into the tier — the same operation
    /// `DeviceTier::place` performs, and valid for the same reason: under HIP's unified
    /// addressing the tier's device pointer IS a host address. **That coincidence is not
    /// portable and `DeviceTier::place`'s own doc says callers must not depend on it** — which
    /// is why this goes through [`DeviceTier::write_at`] rather than dereferencing `base`
    /// here, so the one place that knows about unified addressing stays the one place.
    ///
    /// `ponytail:` buffered I/O through the mmap, not the O_DIRECT io_uring path the routed
    /// experts use. `ExpertSet` streams per-layer sidecar files whose blocks are aligned for
    /// O_DIRECT; a safetensors tensor starts wherever the header left it, so the same path
    /// needs the converter to emit aligned per-layer files. **Upgrade path: `convert_glimmer`
    /// grows a layer-blocked output and this becomes an `AsyncFetch` submit.** Until then the
    /// page cache is doing the caching, which S5 must account for before quoting any
    /// bytes-from-disk number.
    fn fill(&mut self, st: &Safetensors, l: usize) -> Result<()> {
        for (&dst, (tail, len)) in self.pin.addrs().iter().zip(&self.lens) {
            let (name, len) = (format!("{GLIMMER_LAYER_PREFIX}.{l}.{tail}"), *len);
            let bytes = st.raw(&name)?.0;
            ensure!(
                bytes.len() == len,
                "{name} is {} bytes but layer 0's was {len} — every Glimmer layer is the same \
                 shape, so this artifact is not the one this slot was laid out for",
                bytes.len()
            );
            // SAFETY: `dst` is the address `DeviceTier::place` returned for this tensor of layer
            // 0, and `bytes.len() == len` is that placement's own extent (checked immediately
            // above), so the write stays inside the placement it targets.
            unsafe { DeviceTier::write_to(dst as *mut u8, bytes) };
        }
        Ok(())
    }
}

impl GlimmerPin {
    /// Build from artifact directory `dir`, pinning as much as `budget` allows.
    ///
    /// `budget` is `None` for "pin everything" — the all-resident case, which is a BUDGET
    /// VALUE and not a separate code path (see the struct's note on why that distinction is
    /// load-bearing). `Some(bytes)` partitions: model-level tensors first, then whole layers
    /// in ascending order while they fit, then [`GLIMMER_STREAM_SLOTS`] slots for the remainder.
    ///
    /// **The partition is a fixed PREFIX, and that is the optimal policy rather than a
    /// simplification.** A dense model reads its layers in fixed cyclic order, which is
    /// LRU's pathological case: at any deficit LRU evicts exactly the layer needed next and
    /// the hit rate is **0**, not `pinned/n_layers`. Belady on a cyclic scan — evict the
    /// block whose next use is farthest, i.e. the one just used — degenerates to holding a
    /// fixed subset, and every fixed subset of size `k` has the same hit rate `k/n`. So the
    /// whole `--cache-policy` axis collapses to one answer here, `run_glimmer` still refuses
    /// the flag, and nothing in this file consults residency to make a decision.
    pub fn build(dir: &str, cfg: &GlimmerTextConfig, budget: Option<usize>) -> Result<Self> {
        // **The cheap refusals first, then the 30-55 GB allocation.** `FormatMeta::load` is the
        // version and VQ-parameter gate every artifact passes — skipping it would let one
        // written by an older converter load silently — and `open_dir` is where a missing or
        // malformed shard is found. Both were BELOW `DeviceTier::new` until review pointed out
        // that this made a stale artifact pay a full-model allocation before its cheap refusal
        // fired, which inverts the ordering the rest of this port is careful about.
        let meta = FormatMeta::load(dir)?;
        let st = Safetensors::open_dir(dir)?;
        // The partition moved BELOW the open because it now needs the format, and the format is
        // a property of the tensors rather than of the config — an artifact is what says whether
        // a layer costs 967.942 MB or 484.142 MB, so a tier sized before reading one would be
        // sized for whichever format the code happened to assume.
        let fmt = GlimmerFormat::of(&st)?;
        // **The artifact's DECLARED block has to be the one this build computes with, and until
        // 2026-08-15 this path loaded it and threw it away.** `FormatMeta::load` only asserts the
        // field is a positive power of two; every other fp8 consumer then READS it
        // (`Pin::build`, `bin/fp8_to_i4`), while this one passed the compiled-in constant into
        // `place_fp8` and into `layer_bytes`. An artifact written at block 256 would have loaded
        // clean, placed a 4x-too-small grid, and had `gemv_fp8` index far past it into the next
        // tensor's e4m3 bytes read as f32 scales — garbage magnitudes, no error.
        //
        // A refusal rather than threading the value through: `layer_bytes`/`partition` are
        // deviceless config arithmetic with no manifest in reach, so a variable block would need
        // it in two places that cannot see each other. If that is ever wanted, this is the line
        // that says so.
        let block = crate::artifact::quant::FP8_BLOCK;
        ensure!(
            fmt != GlimmerFormat::Fp8 || meta.fp8_block == block,
            "this artifact declares fp8_block {} and this build computes at {block}; the scale \
             grids and the tier sizing would describe different tilings",
            meta.fp8_block
        );
        let (n_pinned, capacity) = cfg.partition(budget, fmt)?;
        let mut tier = DeviceTier::new(capacity)?;

        let embed = place_bf16(&mut tier, &st, "model.language_model.embed_tokens.weight")?;
        let head = place_bf16(&mut tier, &st, "lm_head.weight")?;
        ensure!(
            [embed.o_dim, embed.i_dim] == [cfg.vocab, cfg.hidden]
                && [head.o_dim, head.i_dim] == [cfg.vocab, cfg.hidden],
            "embed_tokens is [{}, {}] and lm_head is [{}, {}]; this config implies [{}, {}] \
             for both",
            embed.o_dim,
            embed.i_dim,
            head.o_dim,
            head.i_dim,
            cfg.vocab,
            cfg.hidden
        );
        let final_norm = place_glimmer_norm(
            &mut tier,
            &st,
            cfg.hidden,
            "model.language_model.norm.weight",
        )?;

        // **Every layer's headers are checked, whether or not the budget pins it.**
        //
        // > **Found by review 2026-08-12, and it was a correctness regression this diff
        // > introduced.** Before the budget existed, `build` placed all 52 layers, so all 52
        // > went through `place_glimmer_norm` (dtype F32 + extent `[hidden]`) and
        // > `place_glimmer_proj` (shape from `layer_tensor_shape`). Afterwards only the PINNED
        // > prefix did, and a streamed layer's only check was `Slot::fill`'s byte length against
        // > layer 0's — so **which checks a layer received became a function of `--max-mem`**.
        // > That is not a theoretical gap: `tests/glimmer_pin.rs` builds the artifact
        // > `convert_glimmer` provably emits at exit 0 with a SHORT norm and asserts the pin
        // > refuses it; at a low budget the same artifact loaded clean and died mid-token with
        // > "not the one this slot was laid out for", diagnosis gone. Worse, a `q_proj` stored
        // > `[6656, 4096]` instead of `[4096, 6656]` is byte-identical in length and was
        // > accepted outright — a transposed matrix, fluent wrong text, no error.
        //
        // Headers only: this reads the index, not the bytes, so it costs 12 lookups per layer
        // and no device memory. It also restores the ordering this function argues for above —
        // the cheap refusals before the big allocation.
        //
        // The projection dtype comes from `fmt`, which was read off layer 0 — so this loop is
        // also what refuses a MIXED artifact, where some layers are bf16 and some fp8. The scale
        // grid is checked here too, because at fp8 a projection with no `weight_scale_inv` would
        // otherwise reach `Slot::fill` as a missing-tensor error mid-token.
        for l in n_pinned..cfg.n_layers {
            let p = format!("{GLIMMER_LAYER_PREFIX}.{l}");
            for t in GLIMMER_LAYER_TENSORS {
                let name = format!("{p}.{t}.weight");
                let want = cfg.layer_tensor_shape(t)?;
                let (_, dtype, shape) = st.raw(&name)?;
                let want_dtype = match (want.len(), fmt) {
                    (1, _) => Dtype::F32,
                    (_, GlimmerFormat::Bf16) => Dtype::Bf16,
                    (_, GlimmerFormat::Fp8) => Dtype::F8E4M3,
                };
                ensure!(
                    *shape == want[..] && dtype == want_dtype,
                    "{name} is {shape:?} {dtype:?}, but this config implies {want:?} \
                     {want_dtype:?} — the artifact and the config describe different models"
                );
                if want.len() == 2 && fmt == GlimmerFormat::Fp8 {
                    let b = crate::artifact::quant::FP8_BLOCK;
                    let sname = format!("{p}.{t}.weight_scale_inv");
                    let grid: Vec<usize> = want.iter().map(|d| d.div_ceil(b)).collect();
                    let (_, sdtype, sshape) = st.raw(&sname)?;
                    ensure!(
                        *sshape == grid[..] && sdtype == Dtype::F32,
                        "{sname} is {sshape:?} {sdtype:?}, but a {want:?} weight at block {b} \
                         implies {grid:?} F32"
                    );
                }
            }
        }

        let src = LayerSrc { st: &st, cfg, fmt };
        let mut pinned = Vec::with_capacity(n_pinned);
        for l in 0..n_pinned {
            pinned.push(place_glimmer_layer(&mut tier, &src, l)?);
        }
        // Slots only when something streams. `n_pinned == n_layers` allocates none, so the
        // all-resident partition costs exactly what it did before this budget existed.
        let n_slots = if n_pinned < cfg.n_layers {
            GLIMMER_STREAM_SLOTS
        } else {
            0
        };
        let mut slots = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            slots.push(Slot::new(&mut tier, &src)?);
        }
        Ok(Self {
            tier,
            src: st,
            embed,
            head,
            final_norm,
            pinned,
            slot_layer: vec![None; slots.len()],
            slots,
            n_layers: cfg.n_layers,
            hits: 0,
            fills: 0,
        })
    }

    /// Layer `l`'s twelve device addresses — **pinned or streamed, indistinguishably.**
    ///
    /// `&mut self` because a miss fills a slot. That is the honest signature: a caller cannot
    /// hold two layers' pins at once, which is exactly the aliasing a slot reuse would break.
    ///
    /// # THE WRITE-AFTER-READ HAZARD, and who closes it
    ///
    /// **Before this refills a slot, every kernel still reading that slot's previous occupant must
    /// have retired.** A fill is a host `memcpy` with no synchronization of any kind, and kernel
    /// launches are asynchronous — so a host that runs even one layer ahead of the device can
    /// overwrite weights a live GEMV is streaming. The symptom is position-dependent,
    /// nondeterministic wrong text: this repo's arena-relocation signature.
    ///
    /// > **CORRECTED 2026-08-12: this is no longer an invariant a CALLER owes.** The section below
    /// > read "the caller must have retired every kernel" and "S5 owes … the write-after-read fence
    /// > in the same change as any increase to `GLIMMER_STREAM_SLOTS`". S3 item 0 paid it instead,
    /// > because S3's loop is the first code that could violate it: **this function now performs the
    /// > `device_sync` itself**, on every miss, and
    /// > `glimmer_residency.rs::a_slot_refill_cannot_land_under_a_live_kernel` gates both directions
    /// > — with the fence removed, all 4096 rows of a live gemm read the overwritten bytes. A caller
    /// > owes nothing here now, which is the only version of this contract a loop cannot get wrong.
    ///
    /// **SCOPE, and it is narrower than "the hazard is closed".** The sync orders the refill after
    /// every kernel ALREADY ENQUEUED when this is called. It cannot help the other order:
    /// [`Bf16Weight`] is `Copy` and every field of [`GlimmerLayerPin`] is a raw pointer, so the
    /// `&mut self` borrow that serialises these calls ends the moment a caller copies one out — and
    /// a caller that captures layer `l`'s pointers, calls `layer(l+1)` (which fences and refills),
    /// and only THEN launches the `l`-th kernel reads `l+1`'s weights as `l`'s. Nothing here can see
    /// that. **Do not launch from pointers captured across a `layer()` call**; the failure is the
    /// same silent position-dependent wrong text, and review named it 2026-08-12.
    ///
    /// > **Two reviews rejected the first version of this section, and they were right.** It
    /// > argued: "a fill is a host memcpy ... it has completed by the time this returns — there
    /// > is no dependency for a caller to await." That covers fill-then-read (the caller does
    /// > not need to wait for the fill) and says nothing about read-then-refill, which is the
    /// > direction that corrupts. The claim that this is "valid for the same reason as
    /// > `DeviceTier::place`" was the tell: `place` runs before any kernel exists.
    /// > `VmmBuf::ptr_mut`'s own contract covers only the same one direction and names
    /// > `device_sync` as the mechanism for slot reuse on the io_uring path.
    ///
    /// **There is still deliberately no ticket**, for a different and narrower reason than the
    /// first version gave: a ticket expresses fill-then-read, the dependency that genuinely does
    /// not exist while the fill is synchronous, and an always-satisfied one is the
    /// `hit: Vec<bool>` mistake `fetch/asyncfetch.rs`'s `Ticket` doc records. The dependency that
    /// DOES exist is the opposite one, and it is the `device_sync` in the body.
    /// **S5 still owes the ticket** — when the fill goes async, fill-then-read becomes a real
    /// dependency and the sync below becomes the wrong instrument for it (an event on the fetch
    /// stream, not a whole-device join).
    ///
    /// `ponytail:` a synchronous fill blocks the decode thread for a layer's worth of memcpy
    /// and buys no overlap. Deliberate at R1, whose gates are about WHAT the bytes are, not
    /// when they arrive; S5 measures the overlap and owns the upgrade.
    pub fn layer(&mut self, l: usize) -> Result<&GlimmerLayerPin> {
        ensure!(
            l < self.n_layers,
            "layer {l} is past this model's {} layers",
            self.n_layers
        );
        if l < self.pinned.len() {
            return Ok(&self.pinned[l]);
        }
        // Round-robin over the streamed suffix. **At `GLIMMER_STREAM_SLOTS` = 1 this maps every
        // streamed layer to slot 0, so it separates nothing** — and the first version of this
        // comment claimed injectivity across consecutive
        // layers "which is what stops a fill landing on the slot a kernel is still reading".
        // Nothing here stops that; only a fence does (S3 item 0), and review found this comment
        // sending a reader to the wrong place. 2026-08-12.
        let s = (l - self.pinned.len()) % self.slots.len();
        if self.slot_layer[s] == Some(l) {
            self.hits += 1;
            return Ok(&self.slots[s].pin);
        }
        // **THE WRITE-AFTER-READ FENCE** — the hazard is on this function's doc; this is what closes
        // it. `glimmer_residency.rs::a_slot_refill_cannot_land_under_a_live_kernel` gates it.
        //
        // **Unconditional on a miss, not conditional on `slot_layer[s].is_some()`.** The narrower
        // version reads as tighter and is wrong: a fill that failed halfway leaves `slot_layer[s]`
        // at `None` over a slot whose previous occupant's pointers WERE handed out, so it would skip
        // the fence exactly when the slot is least trustworthy.
        //
        // Cost is one sync per streamed layer, which GLM's loop already pays per layer — EXPECTED to
        // be free on the decode path, not measured, because no Glimmer loop exists to measure it on.
        crate::backend::hip::device_sync()?;
        // **Invalidated BEFORE the fill, not after it.** `fill` writes tensor-by-tensor and can
        // bail in the middle, at which point the slot holds a prefix of layer `l` and a suffix of
        // its previous occupant. Assigning only on success left the flag claiming the OLD layer,
        // so a caller that handled the error and re-requested that old layer took the hit path
        // and got the mixture, silently. Review finding, 2026-08-12: a flag set in the success
        // path and not cleared in the failure path.
        self.slot_layer[s] = None;
        self.slots[s].fill(&self.src, l)?;
        self.slot_layer[s] = Some(l);
        self.fills += 1;
        Ok(&self.slots[s].pin)
    }

    /// How many layers this budget pinned. `n_layers` means all-resident.
    pub fn pinned_layers(&self) -> usize {
        self.pinned.len()
    }

    /// How many layers stream. Zero when the budget pinned everything.
    pub fn streamed_layers(&self) -> usize {
        self.n_layers - self.pinned.len()
    }

    /// Slot hits and fills. A partition working as designed fills once per streamed layer per
    /// token; a fill count above that is thrash and means the slot map is wrong.
    pub fn slot_stats(&self) -> (u64, u64) {
        (self.hits, self.fills)
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
