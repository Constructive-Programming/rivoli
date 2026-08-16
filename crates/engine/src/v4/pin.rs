//! The V4 resident set (attention core, hyper-connections, the fp8 shared expert) plus the
//! `.f4` routed pool, placed by `core::residency::partition()`.
//!
//! Ported from `old:src/memory/pin.rs`'s `F4Pin` half, with the same ONE re-architecture
//! [`crate::glm::pin`] took and for the same reason: **the old build derived placement
//! itself** — the resident tier took the artifact's own byte length and the pool took
//! `pool_budget(capacity, tier_cap)`, i.e. "whatever is left". Here the split has one author:
//! this pin enumerates its streamable units (the routed experts, layer-major), states its
//! [`Floor`](rivoli_core::residency::Floor) through
//! [`weights_only_floor`](crate::resident::weights_only_floor), and EXECUTES the
//! [`Partition`] that `partition()` returns. A budget below the floor is a refusal with the
//! arithmetic in it, at startup, before any device allocation (P6; INV-8 is the monotonicity
//! gate on the function this defers to).
//!
//! **A separate type from [`GlmPin`](crate::glm::pin::GlmPin), for the reason
//! `rivoli_artifact` keeps `V4Config` separate from `ModelConfig`:** the two architectures
//! share no tensor name, no attention shape and no expert format, and one pin parameterised
//! by an arch flag is a GLM-shaped placement path one `if` away from running on a V4
//! artifact. What IS shared is factored — [`crate::resident`]'s placers and floor,
//! [`crate::routed`]'s pool, `ExpertSet` itself — and this arm is where `RoutedFmt::F4` is
//! OWNED rather than refused (`glm::pin::expert_bytes` bails on it by name).
//!
//! # What this pin does NOT place, and why each one is a decision
//!
//! * **The lightning indexer** (`attn.indexer.*` on the 21 ratio-4 layers). This arm selects
//!   compressed blocks POSITIONALLY — `crate::v4`'s second declared deviation — so nothing
//!   reads an indexer weight, and placing ~1 GB of them would take those bytes straight out
//!   of the routed pool. They are still COUNTED, because [`safetensors_bytes`] is the file's
//!   length and the file holds them; that over-counts the tier, which is the safe direction
//!   ([`safetensors_bytes`] carries the bias argument). The reference placed them and had no
//!   reader either. They join with the scored selection, beside the caller that needs them.
//! * **An int8 embed or head.** V4 carries both as bf16 and the int8 launchers are GLM's;
//!   widening them at load would double 2.1 GB of resident set to paper over a missing
//!   kernel, and whether to requantize is a quality question with a paired-dNLL measurement
//!   attached. [`Bf16Weight`] is the type that says so.

use super::geometry::FP8_BLOCK;
use crate::device::DeviceTier;
use crate::resident::{
    Batch, Bf16Weight, Fp8Mlp, Fp8Weight, PinCfg, PoolPlan, dims2, place_bf16, place_f32,
    place_fp8, safetensors_bytes, stream_units,
};
use crate::routed::{RoutedGeom, RoutedPool};
use anyhow::{Context, Result, ensure};
use rivoli_artifact::format::{
    Dtype, ExpertSet, FormatMeta, RoutedFmt, Safetensors, SetDims, f4_layer_range,
};
use rivoli_artifact::quant::V4_PROJ;
use rivoli_artifact::quant::f4::f4_expert_stride;
use rivoli_artifact::v4_config::V4Config;
use rivoli_core::residency::Unit;

/// One hyper-connection block's three f32 tables: `<base>_base`, `<base>_fn`, `<base>_scale`.
///
/// The same triple serves `layers.{l}.hc_attn`, `layers.{l}.hc_ffn` and the model-level
/// `hc_head`, which is why it is one type and one placer rather than three — and why the
/// three fields are NAMED: `launch_hc_pre` takes `(scale, base)` and `launch_hc_head_collapse`
/// takes `(base, scale)`, both `*const f32`, so a swap compiles, runs, and is finite.
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
/// runs `x.float()` ("compression need fp32", the reference's own comment), so the converter
/// widens rather than choosing. Extents are NOT stored: they vary with `compress_ratio`
/// (`ape` is `[ratio, coff·head_dim]`) and belong to the [`super::geometry::Geom`] that
/// derives them, not to the pin that places them.
#[derive(Clone, Copy)]
pub struct CompressorWeights {
    pub ape: *const f32,
    pub norm: *const f32,
    pub wgate: *const f32,
    pub wkv: *const f32,
}

/// How a layer's gate SELECTS experts.
///
/// A sum type rather than two `Option`s because the two are exclusive in the checkpoint — a
/// hash layer carries `ffn.gate.tid2eid` and no `.bias`, a scored layer the reverse — so
/// `Some`/`Some` and `None`/`None` are states no artifact can be in.
/// `V4Config::layer_routes_by_hash` decides which and the converter wrote whichever it
/// decided, so the config and the artifact cannot disagree without one of them failing.
///
/// Both variants are HOST-side, like GLM's router bias: V4's routing (sqrt-softplus scoring,
/// bias, top-k) runs on the CPU, and the hash path is a lookup by TOKEN ID, which the host
/// already holds. Placing 6.2 MB of `tid2eid` per hash layer on the device to index it there
/// would buy nothing — and the range check below is only expressible host-side.
pub enum GateRoute {
    /// `tid2eid[token * top_k + j]` — already range-checked, see [`parse_tid2eid`].
    Hash { tid2eid: Vec<u32> },
    /// The router correction bias added before top-k, `n_experts` long.
    Scored { bias: Vec<f32> },
}

/// One V4 layer's resident weights.
pub struct V4LayerPin {
    pub attn_norm: *const f32,
    pub ffn_norm: *const f32,
    /// `q_norm` (over `q_lora_rank`). The weightless QK-norm that follows `wq_b` has no
    /// tensor at all — it is `rsqrt(mean(q²) + eps)`.
    pub q_norm: *const f32,
    /// `kv_norm` (over `head_dim`). Also `[d]` and also f32, which is exactly why the
    /// compressor's own `norm` is a different field on a different struct.
    pub kv_norm: *const f32,
    /// `[n_heads]` f32, added to the softmax DENOMINATOR only.
    pub attn_sink: *const f32,
    pub wq_a: Fp8Weight,
    pub wq_b: Fp8Weight,
    /// ONE kv entry, `head_dim` wide, serving as both K and V for every head.
    pub wkv: Fp8Weight,
    /// `[wkv ‖ wq_a]` as one `[head_dim + q_lora_rank, hidden]` weight — [`Self::wkv`] and
    /// [`Self::wq_a`] are VIEWS into this placement, so it costs no bytes. `Some` whenever
    /// the seam divides the scale block (this checkpoint: `512 % 128 == 0`). See
    /// [`place_fp8_qkv`], and `super::attn::Weights::wqkv` for what a decode does with it.
    pub wqkv: Option<Fp8Weight>,
    pub wo_a: Fp8Weight,
    pub wo_b: Fp8Weight,
    /// Router gate `[n_experts, hidden]` f32, device-side (the scores are a GEMV).
    pub gate_w: *const f32,
    pub route: GateRoute,
    pub hc_attn: HyperConn,
    pub hc_ffn: HyperConn,
    /// The always-on shared expert — fp8 e4m3, resident, NOT in the `.f4`. This is what makes
    /// `.f4` a name for the ROUTED experts only. Same type as a GLM dense layer's MLP, which
    /// is the honest reading: one weight set, three fp8 projections, read by every row.
    pub shared: Fp8Mlp,
    /// Present iff `compress_ratio != 0` — the same condition
    /// [`super::geometry::Geom::attention`] returns `Some` for, decided from the same config
    /// in a different file. `super::kvcompress` asserts the two agree rather than assuming.
    pub compressor: Option<CompressorWeights>,
}

/// The V4 resident weight set, plus the validated `.f4` routed-expert source and its pool.
pub struct V4Pin {
    /// The resident weight slab. Never read through after `build` — held purely as the RAII
    /// owner of the VMM allocation every resident pointer points into.
    #[allow(dead_code)]
    tier: DeviceTier,
    /// `[vocab, hidden]` bf16. See [`Bf16Weight`] — exposed, not consumed.
    pub embed: Bf16Weight,
    /// `head.weight`, `[vocab, hidden]` bf16. Untied from [`Self::embed`] in this checkpoint.
    pub head: Bf16Weight,
    pub final_norm: *const f32,
    pub hc_head: HyperConn,
    /// Artifact order, so `layers[0]` is [`Self::range`]`.start` — which is NOT always 0.
    /// Private, and reachable only through [`Self::layer`], because that offset is exactly
    /// the kind of thing a caller gets right once and then forgets: an absolute layer id used
    /// as a direct index into a pin over layers 3..6 reads layer 6's weights for layer 3 and
    /// never fails.
    layers: Vec<V4LayerPin>,
    range: std::ops::Range<usize>,
    /// The `.f4` set: fd owner for the routed pool, and the thing whose headers and lengths —
    /// one per layer in the artifact's range — were validated at startup rather than at the
    /// first miss. Held for the run: the pool's read table is `(fd, begin, len)` triples whose
    /// fds these `File`s own, so dropping this closes them under a live pool.
    #[allow(dead_code)]
    f4: ExpertSet,
    /// The routed FP4 streaming pool. **Not optional, and that is the size argument**: the
    /// shipped 43-layer `.f4` set is 137 GiB against a ~115 GiB budget, so unlike GLM's
    /// (~41% residency) V4's routed experts cannot all be resident on any configuration this
    /// machine has. A `V4Pin` without a pool cannot run the model at all.
    pub routed: RoutedPool,
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

/// Place one `Compressor` — the attention's own, and (when this arm ever runs one) the
/// indexer's nested one, which differ only in width and are therefore the same four names
/// under different prefixes.
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

/// Place a V4 layer's `wkv` and `wq_a` as ONE fp8 placement — `wkv`'s rows FIRST, then
/// `wq_a`'s, weight bytes and scale grids each concatenated row-wise — and resolve the two
/// original weights as VIEWS into it (offsets, zero extra bytes), plus the whole
/// `[head_dim + q_lora_rank, hidden]` concat a fused decode GEMV reads.
///
/// kv FIRST because the fused output row must land the KV entry at the base of the attention
/// scratch's `kv` buffer, where every consumer already looks. The seam is scale-exact iff
/// `wkv`'s row count is a multiple of `block` — the kernel's scale row is
/// `j >> log2(block)`, so row `head_dim + r` reads concatenated scale row
/// `head_dim/block + r/block` exactly when nothing straddles. **A checkpoint where it does
/// not divide gets two ordinary placements and NO fused weight**, and decode runs the
/// two-launch path unchanged; that is the whole reason the return is an `Option` rather than
/// an `ensure!`.
///
/// The two GEMVs it replaces are grid-starved rather than bandwidth-bound (512 and 1024 waves
/// against ~2560 wave slots), so this is a launch-shape win and not a byte one — and every
/// output value is bit-identical, because it is the same kernel over the same per-row `k`
/// with a taller grid. What MOVES is where the q intermediate lands; `super::attn` owns that.
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
        // The unfused fallback goes back through `place_fp8`, so both tensors still get the
        // scale-grid extent check — which is the reason this arm reaches for that placer
        // rather than the reference's, and the reason this branch is not a second placer.
        return Ok((
            place_fp8(tier, st, &format!("{attn}.wkv"), block)?,
            place_fp8(tier, st, &format!("{attn}.wq_a"), block)?,
            None,
        ));
    }
    // The grids must BE the `ceil(o/block) x ceil(i/block)` f32 the kernel indexes, or the
    // row-wise concat below would misattribute scale rows silently. `place_fp8` makes the
    // same check from a `shape`; this one is on the BYTE LENGTH, because the concat is what
    // the offset arithmetic below indexes into.
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
    // SAFETY: both offsets are inside the two placements above — `wq_a`'s rows start `ko`
    // weight rows (`ko/block` scale rows, exact by the divisibility gate) in.
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

/// The three extents a hash table is checked against.
///
/// A struct because they are three bare `usize` and every one is plausible in another's
/// position: this checkpoint's `vocab` and `n_experts` are both counts of things and `top_k`
/// is a small one, so a transposed pair produces a shape check that passes on the wrong
/// rectangle. Named fields move that mistake to the one construction site.
#[derive(Clone, Copy)]
struct HashDims {
    vocab: usize,
    top_k: usize,
    n_experts: usize,
}

impl HashDims {
    fn of(cfg: &V4Config) -> Self {
        Self {
            vocab: cfg.vocab,
            top_k: cfg.top_k,
            n_experts: cfg.n_experts,
        }
    }
}

/// `ffn.gate.tid2eid` (I64) parsed into expert ids that are valid by construction.
///
/// **Nothing downstream has ever looked at these.** The converter shape-checks the tensor
/// against `[vocab, top_k]` and then copies it verbatim, so no VALUE is ever read, and the
/// MoE launch path indexes its descriptor array with whatever arrives. An entry outside
/// `0..n_experts` therefore selects another expert's slot, or reads past the array, and
/// `moe.hip`'s own note records that the kernel does not check.
///
/// Parsed to `u32` rather than checked and left `i64`, so the check and the storage are one
/// act: a negative or oversized id cannot exist in the returned vector. `u32::try_from`
/// rather than `as u32` because the cast truncates — 2^32 would become 0, an id that is
/// perfectly in range. 775,680 entries per hash layer x 3 layers, read once at startup.
///
/// Takes [`HashDims`] rather than `&V4Config` so its test needs no config, and so no machine
/// can end up running a vacuous version of it.
fn parse_tid2eid(raw: &[u8], shape: &[usize], d: HashDims) -> Result<Vec<u32>> {
    let (vocab, top_k, n_experts) = (d.vocab, d.top_k, d.n_experts);
    ensure!(
        shape == [vocab, top_k],
        "ffn.gate.tid2eid: shape {shape:?} != [{vocab}, {top_k}]"
    );
    // The EXTENT, separately from the shape. `Safetensors::typed` matches the dtype and
    // nothing else — a tensor's byte span comes from the header's `data_offsets` and is never
    // confronted with `product(shape) * 8` — so a tensor whose span disagrees with its
    // declared shape passes the check above, `chunks_exact` drops the partial tail, and the
    // returned table is SHORT. `GateRoute::Hash`'s consumer indexes it at `token * top_k + j`,
    // so a short table is an out-of-bounds read for the last tokens.
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

/// The routed experts as residency units, layer-major — the priority order handed to
/// `partition()`.
///
/// Layer-major because decode's access is cyclic over layers: pinning a prefix of it is the
/// static partition that cyclic access makes optimal (the Belady degenerate), and any
/// per-expert evidence that beats uniform arrives later as a REORDERING of this list, never
/// as a second placement author.
///
/// **Spans the ARTIFACT's range, not the model's** — a partial convert enumerates what it
/// holds, or the partition would size the pool for experts the `.f4` set does not contain.
/// That is the fact this function exists to carry; the construction is [`stream_units`]'s.
fn expert_units(layers: usize, n_experts: usize, unit: usize) -> Vec<Unit> {
    stream_units(layers * n_experts, unit)
}

impl V4Pin {
    /// Build the V4 resident set and its `.f4` streaming pool from artifact directory `dir`.
    pub fn build(dir: &str, cfg: &V4Config, pin: PinCfg<'_>) -> Result<Self> {
        // Which layers the artifact HOLDS. `num_hidden_layers` is the model's; the two differ
        // on every partial convert. NOT required to start at 0 — that is a property of a
        // DECODE (a forward pass has no residual stream to enter at layer 3) and
        // `super::engine::V4Engine::new` is where it is refused. Refusing it here made every
        // partial artifact but the first unloadable in the old tree.
        let range = f4_layer_range(dir, cfg.n_layers)?;
        // The artifact's declared fp8 block, checked equal to the one the layer loop spells.
        // Every quantized `Linear` on this arm quantizes its ACTIVATION at the same block its
        // WEIGHT was quantized on — that is `kernel.py`'s rule, not a coincidence — so the
        // loop reads the constant while the weights carry the file's number. If the two ever
        // disagreed, every fp8 GEMV would tile its activation against the wrong scale grid:
        // right addresses, wrong arithmetic, and nothing downstream that could see it.
        let block = FormatMeta::load(dir)?.fp8_block;
        ensure!(
            block == FP8_BLOCK,
            "this artifact declares an fp8 block of {block}; the V4 layer loop quantizes \
             activations at {FP8_BLOCK} and the two must be the same number"
        );
        let st = Safetensors::open_dir(dir)?;
        // Spans exactly the artifact's range — NOT `0..cfg.n_layers` — and validates every
        // header and length here rather than at the first miss.
        let f4 = ExpertSet::open_routed(
            dir,
            RoutedFmt::F4,
            SetDims::new(range.clone(), cfg.n_experts, cfg.hidden, cfg.moe_inter),
        )?;
        let geom = RoutedGeom::new(&f4)?;

        // THE PLACEMENT DECISION — one call, one author, and it runs before any device
        // allocation. Alignment slack: `DeviceTier::place` starts every reservation at
        // `used.next_multiple_of(256)`, so total padding is under 256 B per placement; V4
        // places ~30 tensors a layer over <= 43 layers plus 4 model-level, i.e. under 0.5 MB.
        // 16 MiB is ~30x that bound — and it is a TRADE rather than a freebie, because every
        // over-counted byte comes out of the pool.
        const SLACK: usize = 16 << 20;
        let resident = safetensors_bytes(dir, None)?;
        let tier_cap = resident + SLACK;
        let unit = f4.expert_slot();
        let units = expert_units(range.len(), cfg.n_experts, unit);
        // `super::ROWS` rows and ZERO shared blocks — both facts about this arm's kernels
        // rather than preferences. `moe.hip` instantiates the FP4 expert range at `R = 1`
        // only, so a V4 decode is structurally single-row and there is no union to take; and
        // V4's shared expert is fp8 e4m3 and RESIDENT, so the `.f4` set holds `n_experts`
        // blocks with none folded in (which is also why `launch_moe_expert_range_f4` takes an
        // `n_desc` its two siblings do not).
        let batch = Batch::union(cfg.top_k, super::ROWS, 0, unit)?;
        let (placement, pool) = PoolPlan::new("V4", &units, tier_cap, batch).decide(pin)?;
        // The residency FRACTION, which `plan_pool`'s line does not carry and which is the
        // number that explains every later V4 measurement: the 43-layer `.f4` set is 137 GiB
        // against ~115 GiB of budget, so a run whose fraction is not in its log never
        // happened. A log line and not a check — the set CANNOT fit and the streaming is the
        // point, so there is nothing here to refuse.
        let routed_total = (range.len() * cfg.n_experts) as u64
            * f4_expert_stride(cfg.hidden, cfg.moe_inter) as u64;
        tracing::info!(
            "v4 resident set {:.2} GiB over layers [{}, {}); routed set {:.2} GiB, \
             {:.1}% resident",
            resident as f64 / (1u64 << 30) as f64,
            range.start,
            range.end,
            routed_total as f64 / (1u64 << 30) as f64,
            100.0 * (placement.pinned.len() * unit) as f64 / routed_total.max(1) as f64,
        );
        let mut tier = DeviceTier::new(tier_cap)?;
        let g = Globals::place(&mut tier, &st)?;
        let layers = place_layers(&mut tier, &st, cfg, &range, block)?;
        Ok(Self {
            tier,
            embed: g.embed,
            head: g.head,
            final_norm: g.final_norm,
            hc_head: g.hc_head,
            layers,
            range,
            f4,
            // Opened LAST, after every resident byte is placed: the tier's allocation is the
            // one that must not fail, and the pool's reservation is ~100x its size.
            routed: RoutedPool::new(pool, geom)?,
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

/// The four model-level tensors, placed.
///
/// A struct and a constructor rather than four `let`s in [`V4Pin::build`], for the reason
/// [`place_layers`] is also a function: `build` already owns the artifact open, the placement
/// decision and the pool, and every tensor name it also spells is one more thing in a
/// function whose subject is supposed to be the DECISION.
struct Globals {
    embed: Bf16Weight,
    head: Bf16Weight,
    final_norm: *const f32,
    hc_head: HyperConn,
}

impl Globals {
    fn place(tier: &mut DeviceTier, st: &Safetensors) -> Result<Self> {
        Ok(Self {
            embed: place_bf16(tier, st, "embed.weight")?,
            head: place_bf16(tier, st, "head.weight")?,
            final_norm: place_f32(tier, st, "norm.weight")?,
            hc_head: place_hc(tier, st, "hc_head")?,
        })
    }
}

/// Place every layer the artifact holds, in artifact order.
///
/// Split out of [`V4Pin::build`] for cohesion and for the borrow: `build` holds the tier
/// mutably across the pool construction, and a 90-line layer body inlined there put the
/// partition arithmetic and the tensor names in one function that scored badly on both.
fn place_layers(
    tier: &mut DeviceTier,
    st: &Safetensors,
    cfg: &V4Config,
    range: &std::ops::Range<usize>,
    block: usize,
) -> Result<Vec<V4LayerPin>> {
    let mut layers = Vec::with_capacity(range.len());
    for l in range.clone() {
        let lb = format!("layers.{l}");
        let a = format!("{lb}.attn");
        let route = place_route(st, cfg, l, &lb)?;
        // wkv + wq_a go through the concat placer: one placement, three views.
        let (wkv, wq_a, wqkv) = place_fp8_qkv(tier, st, &a, block)?;
        // The concat seams at the TENSOR's row count while the fused landing offset is the
        // CONFIG's `head_dim` — a checkpoint where they disagree would put the q intermediate
        // at the wrong offset silently. (The split path is equally wrong under such a
        // mismatch, but wrongly DIMENSIONED, which fails louder.) Load-time and not a
        // `debug_assert!` in the launch path, because that one is compiled out exactly when
        // benchmarks run.
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
        let mut fp8 = |name: &str| place_fp8(tier, st, &format!("{a}.{name}"), block);
        let (wq_b, wo_a, wo_b) = (fp8("wq_b")?, fp8("wo_a")?, fp8("wo_b")?);
        layers.push(V4LayerPin {
            attn_norm: place_f32(tier, st, &format!("{lb}.attn_norm.weight"))?,
            ffn_norm: place_f32(tier, st, &format!("{lb}.ffn_norm.weight"))?,
            q_norm: place_f32(tier, st, &format!("{a}.q_norm.weight"))?,
            kv_norm: place_f32(tier, st, &format!("{a}.kv_norm.weight"))?,
            attn_sink: place_f32(tier, st, &format!("{a}.attn_sink"))?,
            wq_a,
            wq_b,
            wkv,
            wqkv,
            wo_a,
            wo_b,
            gate_w: place_f32(tier, st, &format!("{lb}.ffn.gate.weight"))?,
            route,
            hc_attn: place_hc(tier, st, &format!("{lb}.hc_attn"))?,
            hc_ffn: place_hc(tier, st, &format!("{lb}.hc_ffn"))?,
            shared: place_shared(tier, st, cfg, l, block)?,
            // Driven off the CONFIG, never off `st.has(..)` — the same choice the converter
            // made and for the same reason: a layer whose tensors disagree with
            // `compress_ratios` must fail here, not silently take whichever branch the
            // artifact happens to satisfy.
            compressor: cfg
                .layer_has_compressor(l)?
                .then(|| place_compressor(tier, st, &format!("{a}.compressor")))
                .transpose()?,
        });
    }
    Ok(layers)
}

/// One layer's [`GateRoute`], read HOST-side. Driven off the config for [`place_layers`]'s
/// reason.
fn place_route(st: &Safetensors, cfg: &V4Config, l: usize, lb: &str) -> Result<GateRoute> {
    if cfg.layer_routes_by_hash(l) {
        let (raw, shape) = st.typed(&format!("{lb}.ffn.gate.tid2eid"), Dtype::I64)?;
        let tid2eid =
            parse_tid2eid(raw, shape, HashDims::of(cfg)).with_context(|| format!("layer {l}"))?;
        return Ok(GateRoute::Hash { tid2eid });
    }
    let (raw, _) = st.typed(&format!("{lb}.ffn.gate.bias"), Dtype::F32)?;
    let bias = rivoli_artifact::quant::read_f32(raw);
    ensure!(
        bias.len() == cfg.n_experts,
        "layer {l} ffn.gate.bias has {} entries, expected {}",
        bias.len(),
        cfg.n_experts
    );
    Ok(GateRoute::Scored { bias })
}

/// Place one layer's always-resident fp8 shared expert.
///
/// `e == n_experts` selects `ffn.shared_experts` — the convention `v4_expert_base` owns. The
/// slot meanings must match [`Fp8Mlp`]'s FIELDS rather than the tensors' names, which is what
/// `V4_PROJ`'s `[w1, w3, w2]` order carries: gate/up/down against w1/w3/w2.
fn place_shared(
    tier: &mut DeviceTier,
    st: &Safetensors,
    cfg: &V4Config,
    l: usize,
    block: usize,
) -> Result<Fp8Mlp> {
    let base = rivoli_artifact::quant::v4_expert_base(l, cfg.n_experts, cfg.n_experts);
    let [w1, w3, w2] = V4_PROJ;
    let mut sh = |p: &str| place_fp8(tier, st, &format!("{base}.{p}"), block);
    Ok(Fp8Mlp {
        gate: sh(w1)?,
        up: sh(w3)?,
        down: sh(w2)?,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)] // tests: panic-on-failure is the idiom

    use super::*;

    /// The table's own bytes, `[vocab, top_k]` row-major i64.
    fn table(ids: &[i64]) -> Vec<u8> {
        ids.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// **Every way a hash table can be unusable, refused by name** — and each case starts
    /// from one that PASSES, so no case can be passing for another's reason.
    ///
    /// The two length checks are separate for a reason a shape check alone cannot cover: a
    /// declared `[vocab, top_k]` whose byte span is short passes the shape test, and
    /// `chunks_exact` then drops the tail rather than failing, leaving a table the consumer
    /// indexes past for the last tokens.
    #[test]
    fn a_hash_table_is_refused_for_shape_extent_and_every_out_of_range_id() {
        let d = HashDims {
            vocab: 3,
            top_k: 2,
            n_experts: 8,
        };
        let (vocab, top_k) = (d.vocab, d.top_k);
        let good = table(&[0, 7, 3, 3, 1, 6]);
        assert_eq!(
            parse_tid2eid(&good, &[vocab, top_k], d).expect("the shipped shape must parse"),
            vec![0u32, 7, 3, 3, 1, 6]
        );
        let cases: [(&[usize], Vec<u8>, &str); 4] = [
            (&[top_k, vocab], good.clone(), "shape"),
            (&[vocab, top_k], good[..40].to_vec(), "expected 48"),
            (&[vocab, top_k], table(&[0, 7, 3, -1, 1, 6]), "= -1"),
            (
                &[vocab, top_k],
                table(&[0, 7, 3, 8, 1, 6]),
                "outside 0..n_experts=8",
            ),
        ];
        for (shape, raw, want) in cases {
            let msg = format!(
                "{}",
                parse_tid2eid(&raw, shape, d).expect_err("must refuse")
            );
            assert!(msg.contains(want), "wrong refusal for {want:?}: {msg}");
        }
    }

    /// `u32::try_from`, never `as u32`. The cast truncates, so `2^32` would arrive as `0` —
    /// an id that is perfectly in range and selects a real expert's slot. This is the one
    /// case a range check written the obvious way lets through.
    #[test]
    fn an_id_that_truncates_into_range_is_still_refused() {
        let raw = table(&[1 << 32, 1]);
        let d = HashDims {
            vocab: 1,
            top_k: 2,
            n_experts: 8,
        };
        let msg = format!(
            "{}",
            parse_tid2eid(&raw, &[1, 2], d).expect_err("must refuse")
        );
        assert!(msg.contains("4294967296"), "wrong refusal: {msg}");
    }

    /// The unit list is the priority ORDER `partition()` pins a prefix of, and it must span
    /// the ARTIFACT's layer count rather than the model's — a partial convert that enumerated
    /// 43 layers' worth of units would size the pool for experts the set does not hold.
    #[test]
    fn the_unit_list_is_layer_major_over_the_artifacts_own_range() {
        let units = expert_units(3, 4, 1024);
        assert_eq!(units.len(), 12, "3 layers x 4 experts");
        assert!(
            units.iter().enumerate().all(|(i, u)| u.id.0 as usize == i),
            "ids must be dense and ascending — the prefix `partition` pins is positional"
        );
        assert!(units.iter().all(|u| u.bytes.get() == 1024));
    }
}
