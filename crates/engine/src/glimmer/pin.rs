//! The Muse Glimmer weight set: what the budget pinned, plus slots for the rest.
//!
//! Ported from `old:src/memory/pin.rs` (the Glimmer half) and `old:src/artifact/model.rs`
//! (the residency arithmetic that sat on `GlimmerTextConfig`), with ONE re-architecture,
//! the same one `GlmPin` took: **the old `GlimmerTextConfig::partition` derived placement
//! itself**, returning `(layers pinned, tier bytes)` from arithmetic it owned. Here the
//! split has one author — this pin enumerates its streamable units (whole layers, in
//! order), states its [`Floor`], and EXECUTES the [`Partition`] that
//! [`rivoli_core::residency::partition`] returns (INV-8).
//!
//! # Dense is not "nothing to stream" — it is the opposite, and the old tree learned it twice
//!
//! > **The premise this file replaces**, quoted from `old:` so it cannot come back: *"Glimmer
//! > is dense, so there is nothing to stream, no pool, no cache policy and no residency
//! > decision. The entire model is this struct, 52 times."* That inverts P6 — the pin is a
//! > function of free memory at run time, never of model architecture — and it is wrong about
//! > dense models specifically: every weight is read every token (53.02 GB bf16), so there is
//! > no routed union to hide behind and the resident FRACTION is the whole tok/s story. 55.712
//! > GB placed unconditionally is not a model that runs beside a 1.7 GB KV cache, a 2.6 GB
//! > drafter, or another tenant.
//!
//! [`GlmPin`](crate::glm::pin::GlmPin) splits a layer into a resident part and a routed part
//! because 256 experts do not fit. Glimmer splits at WHOLE LAYERS, because 52 of them do not
//! fit either. Same function, same argument, different granularity — which is exactly what
//! `partition()` having no architecture parameter is supposed to buy.
//!
//! # The partition is a fixed PREFIX, and that is optimal rather than a simplification
//!
//! A dense model reads its layers in fixed cyclic order, which is LRU's pathological case: at
//! any deficit LRU evicts exactly the layer needed next and the hit rate is **0**, not
//! `pinned/n_layers`. Belady on a cyclic scan — evict the block whose next use is farthest,
//! i.e. the one just used — degenerates to holding a fixed subset, and every fixed subset of
//! size `k` has the same hit rate `k/n`. So the whole `--cache-policy` axis collapses to one
//! answer here, which is why `rivoli_core::legality` refuses that flag on this architecture
//! while accepting `--max-mem`, and why nothing in this file consults residency to decide
//! anything. `partition()`'s prefix shape is that policy already.

use super::geometry::{PIN_SLACK, ProjFmt, STREAM_SLOTS, floor_of, global_bytes, layer_bytes};
use crate::device::{DeviceBuf, DeviceTier};
use anyhow::{Result, ensure};
use rivoli_artifact::format::{Dtype, Safetensors};
use rivoli_artifact::glimmer::{GLIMMER_LAYER_PREFIX, GLIMMER_LAYER_TENSORS};
use rivoli_artifact::glimmer_config::GlimmerTextConfig;
use rivoli_core::residency::{Bytes, Partition, Refusal, Unit, UnitId, partition};

/// A bf16 weight matrix resolved to a device address, with the dims the launch site needs.
///
/// Moved to [`crate::resident`] at M8, when V4's `embed`/`head` needed the identical three
/// fields; re-exported here so every `crate::glimmer::pin::Bf16Weight` path is unchanged.
/// What did NOT move is this arm's [`place_proj`] — it checks the tensor against the shape
/// its config implies, which is Glimmer's own artifact contract rather than the intersection
/// the shared module holds. The refill path casts to `*const u8` at
/// [`GlimmerLayerPin::addrs`], where the unit genuinely IS the byte.
pub use crate::resident::Bf16Weight;
// The fp8 twin, likewise shared: GLM's attention and V4's shared expert read the same
// `packed + scale + block` shape, and `place_fp8` carries the scale-grid extent check that
// was found missing on exactly this arm's fp8 path in `old:`. A plain `use`, unlike
// `Bf16Weight` above: no pre-M8 `glimmer::pin::Fp8Weight` path ever existed to keep alive.
use crate::resident::Fp8Weight;

/// One projection, in whichever format the artifact stores — the pin-side spelling of
/// [`ProjFmt`].
///
/// An enum per weight rather than a generic pin, because the two variants genuinely differ in
/// what a launch needs (the fp8 one carries a scale grid and its block) while EVERYTHING else
/// about the layer — names, order, the slot lifecycle, the partition — is format-blind. The
/// launch-site dispatch in `super::forward::proj` is the only consumer that looks inside.
#[derive(Clone, Copy)]
pub enum ProjPin {
    Bf16(Bf16Weight),
    Fp8(Fp8Weight),
}

/// The two facts every layer placement reads together: the config's shape table and the
/// artifact's sniffed [`ProjFmt`].
///
/// One value because they are ONE claim about the artifact — "these names, these shapes, this
/// storage" — and because it keeps every placer below at four parameters where threading `fmt`
/// beside `cfg` would widen five of them to the same `(tier, st, cfg, fmt, …)` run that
/// `v4/pin.rs`'s placers already open with — and would make [`place_proj`] a **six**-parameter
/// function, which `crates/cli/tests/codescene.rs` fails on its own (the arity rule fires at
/// 5) whatever jscpd thinks. Two gates, one of them checkable by counting; a reader who
/// deletes this struct after satisfying themselves about the other still breaks the build.
///
/// > **TRIMMED 2026-08-16, by review.** This carried two more sentences and neither survives.
/// > It said jscpd "promptly reported" that widened run as a clone: **that measurement was not
/// > reproduced in this session** — jscpd is green over `crates/` as written, which says
/// > nothing either way about the shape that is not written — so the claim is removed rather
/// > than repeated, per the rule against inheriting a number nobody re-derived. It also said
/// > the pair "cannot be transposed the way two loose parameters of one kind can", which is
/// > vacuous: `&GlimmerTextConfig` and `ProjFmt` are different types, so the compiler already
/// > refuses the transposition the sentence claimed credit for preventing. What is left above
/// > is the arity argument, which is checkable by reading the five signatures.
#[derive(Clone, Copy)]
struct Schema<'a> {
    cfg: &'a GlimmerTextConfig,
    fmt: ProjFmt,
}

/// One Muse Glimmer decoder layer's weights, resolved — **whether the budget pinned it or a
/// slot holds it.**
///
/// Four norms (widened to f32 by the converter) and eight projections in the artifact's own
/// [`ProjFmt`] — twelve tensors bf16, twenty fp8 (each projection gains its scale grid),
/// matching [`GLIMMER_LAYER_TENSORS`] with [`layer_tails`]' expansion. **Five projections in
/// the attention block, not four** — `self_attn.gate_proj` is a per-head output gate applied
/// before `o_proj`, it is the same shape as `q_proj`, and it is the one an HF-shaped port
/// drops on the floor.
///
/// **Deliberately NOT `Copy`, unlike [`Bf16Weight`]** — but that buys less than it looks like
/// and the gap matters. A copy of a streamed layer's addresses stays valid-LOOKING after its
/// slot has been refilled with another layer, and borrowing from [`GlimmerPin::layer`]
/// (`&mut self`) makes holding the whole struct across a refill a compile error. **What it
/// does NOT stop is extracting a field:** `Bf16Weight` is `Copy` and the four norms are bare
/// pointers, so `let q = pin.layer(5)?.q;` yields a handle that outlives the borrow. The type
/// narrows the mistake; it does not forbid it, and the invariant on [`GlimmerPin::layer`] is
/// what a caller actually has to honour — see `super::forward::Handles`, which is this tree's
/// structural answer to it.
pub struct GlimmerLayerPin {
    /// Pre-attention, `rms_norm_eps`.
    pub input_ln: *const f32,
    /// **Post-attention, `post_norm_eps` — a different eps, by three orders of magnitude.**
    /// The two post-norms sit on the BRANCH, before the residual add (sandwich norms), which
    /// is why these are separate fields rather than a `[*const f32; 4]` a loop indexes.
    pub post_attn_ln: *const f32,
    /// Pre-MLP, `rms_norm_eps`.
    pub pre_ffn_ln: *const f32,
    /// Post-MLP, `post_norm_eps`.
    pub post_ffn_ln: *const f32,
    /// `[n_heads·head_dim, hidden]`. **Not square**, and not derivable from `hidden / n_heads`.
    pub q: ProjPin,
    /// `[kv_heads·head_dim, hidden]`. GQA at 16 query heads per KV head.
    pub k: ProjPin,
    /// Same shape as [`Self::k`], and separable from it only by NAME.
    pub v: ProjPin,
    /// `[hidden, n_heads·head_dim]` — the transposed one.
    pub o: ProjPin,
    /// `self_attn.gate_proj`, same shape as [`Self::q`] and likewise separable only by name.
    pub attn_gate: ProjPin,
    pub mlp_gate: ProjPin,
    pub mlp_up: ProjPin,
    /// `[hidden, inter]` — the other transposed one.
    pub mlp_down: ProjPin,
}

impl GlimmerLayerPin {
    /// The layer's device addresses, in [`GLIMMER_LAYER_TENSORS`] order — **with each fp8
    /// projection's scale grid immediately after its packed bytes**, which is the same
    /// expansion [`layer_tails`] applies to the names. Twelve entries bf16, twenty fp8.
    ///
    /// Exists so [`Slot::fill`] can write to each tensor's own address instead of computing an
    /// offset. A permutation here would make a streamed layer a permutation of a pinned one —
    /// every tensor the right shape, the model silently wrong.
    ///
    /// **Nothing asserts the order directly**, and it is worth knowing which guarantee you
    /// have. What covers it is transitive: a fixture that gives every tensor distinct bytes
    /// makes any swap of two same-length entries redden a byte-identity gate, and a swap of
    /// different-length entries fails [`Slot::fill`]'s length check.
    fn addrs(&self) -> Vec<*const u8> {
        let mut a = vec![
            self.input_ln as *const u8,
            self.post_attn_ln as *const u8,
            self.pre_ffn_ln as *const u8,
            self.post_ffn_ln as *const u8,
        ];
        for p in [
            &self.q,
            &self.k,
            &self.v,
            &self.o,
            &self.attn_gate,
            &self.mlp_gate,
            &self.mlp_up,
            &self.mlp_down,
        ] {
            match p {
                ProjPin::Bf16(w) => a.push(w.packed as *const u8),
                ProjPin::Fp8(w) => {
                    a.push(w.packed);
                    a.push(w.scale as *const u8);
                }
            }
        }
        a
    }
}

/// One streaming slot: a layer-sized region of the tier, with the device addresses inside it
/// precomputed — twelve on a bf16 artifact, twenty on an fp8 one.
///
/// **The addresses are computed ONCE and never move.** A fill overwrites bytes at fixed
/// offsets; it does not re-place anything. That is what lets [`GlimmerPin::layer`] hand out a
/// [`GlimmerLayerPin`] whose pointers are stable for as long as the slot holds that layer, and
/// it is the difference between this and an arena whose compaction can move a slot out from
/// under an in-flight read.
struct Slot {
    pin: GlimmerLayerPin,
    /// Each tensor's name TAIL and the byte length of the placement the matching address in
    /// `pin.addrs()` points at, in [`layer_tails`] order.
    ///
    /// **One list rather than a `lens` vector beside a `tails` parameter**, because a refill
    /// zips it against `pin.addrs()` and a zip cannot report that a leg ran out. The parameter
    /// shape existed until review 2026-08-16 and had exactly this hole: a short `tails` would
    /// have truncated the copy silently, leaving a trailing tensor holding layer 0's bytes —
    /// fluent wrong text at every budget that streams. The first fix was a runtime `ensure!`;
    /// this one is that the argument has nowhere to arrive from. The name is carried rather
    /// than re-derived per visit because `fill` runs on the streaming hot path.
    tensors: Vec<(String, usize)>,
}

/// Place one Glimmer f32 norm, with its LENGTH checked against `hidden`.
///
/// **The check is the whole reason this exists rather than a bare `place`.**
/// [`GlimmerLayerPin`]'s norm fields are bare `*const f32` carrying no extent, so a norm
/// shorter than `hidden` is accepted, sized into a tier that has room to spare, and handed to
/// the RMSNorm launcher as a `hidden`-long array. It then reads inter-placement padding and
/// the next tensor's bytes: a scaled-wrong residual stream, in bounds of the slab, with no
/// error anywhere. Found by two independent reviews in `old:`, 2026-08-11.
fn place_norm(
    tier: &mut DeviceTier,
    st: &Safetensors,
    hidden: usize,
    name: &str,
) -> Result<*const f32> {
    let (bytes, shape) = st.typed(name, Dtype::F32)?;
    ensure!(
        shape == [hidden],
        "{name} is {shape:?}, but this config implies [{hidden}] — the artifact and the \
         config describe different models"
    );
    Ok(tier.place(bytes)? as *const f32)
}

/// Place one Glimmer layer projection, with the shape the config implies, in the format the
/// artifact was sniffed to carry.
///
/// The shape comes from [`GlimmerTextConfig::layer_tensor_shape`] rather than from the caller,
/// so the eight call sites below cannot each get it slightly wrong — and it is checked from
/// the HEADER, once, before either arm reads a byte, so both formats refuse a transposed
/// tensor with the same message.
///
/// The pairs are why a shape check is not enough on its own: `q_proj` and `self_attn.gate_proj`
/// are both `[n_heads·head_dim, hidden]` and `k_proj`/`v_proj` are both
/// `[kv_heads·head_dim, hidden]`, so within each pair only the NAME separates them. What this
/// DOES catch is the transposition: a `q_proj` stored `[hidden, n_heads·head_dim]` is
/// byte-identical in length and would otherwise be accepted outright — a transposed matrix,
/// fluent wrong text, no error.
///
/// The fp8 arm delegates to [`crate::resident::place_fp8`], which adds the check this shape
/// one cannot make: the SCALE GRID's extent against the weight's dims — the check `old:`'s
/// review found running zero times on exactly this path.
fn place_proj(
    tier: &mut DeviceTier,
    st: &Safetensors,
    s: Schema<'_>,
    prefix: &str,
    tensor: &str,
) -> Result<ProjPin> {
    let want = s.cfg.layer_tensor_shape(tensor)?;
    let name = format!("{prefix}.{tensor}.weight");
    let (_, _, shape) = st.raw(&name)?;
    ensure!(
        *shape == want[..],
        "{name} is {shape:?}, but this config implies {want:?} — the artifact and the config \
         describe different models"
    );
    match s.fmt {
        ProjFmt::Bf16 => {
            let (bytes, _) = st.typed(&name, Dtype::Bf16)?;
            Ok(ProjPin::Bf16(Bf16Weight {
                packed: tier.place(bytes)? as *const u16,
                o_dim: want[0],
                i_dim: want[1],
            }))
        }
        ProjFmt::Fp8 { block } => Ok(ProjPin::Fp8(crate::resident::place_fp8(
            tier,
            st,
            &format!("{prefix}.{tensor}"),
            block,
        )?)),
    }
}

/// One layer's tensor-name TAILS (the part after `{GLIMMER_LAYER_PREFIX}.{l}.`), in
/// [`GLIMMER_LAYER_TENSORS`] order — with each fp8 projection's `weight_scale_inv`
/// immediately after its `weight`, the same expansion [`GlimmerLayerPin::addrs`] applies to
/// the addresses.
///
/// Spelled once here rather than at each of the two call sites (pinned placement and slot
/// refill), because the two must agree tensor-for-tensor or a streamed layer would be a
/// permutation of a pinned one — and a permutation of correctly-shaped tensors is exactly the
/// silent wrongness this port keeps finding. Whether a tensor has a grid is decided by its
/// RANK from the config's own shape table — the same discriminator [`check_layer_headers`]
/// and `geometry::layer_bytes` use, so the three cannot disagree.
fn layer_tails(s: Schema<'_>) -> Result<Vec<String>> {
    let mut tails = Vec::new();
    for t in GLIMMER_LAYER_TENSORS {
        tails.push(format!("{t}.weight"));
        if matches!(s.fmt, ProjFmt::Fp8 { .. }) && s.cfg.layer_tensor_shape(t)?.len() == 2 {
            tails.push(format!("{t}.weight_scale_inv"));
        }
    }
    Ok(tails)
}

/// Tail → the FULL tensor name for layer `l`. The one place the prefix joins the two.
///
/// Named for what it returns, not for what it takes: in a file whose central hazard is one
/// tensor's identity being mistaken for another's, `tail_name` read as "the name of the tail"
/// (review, 2026-08-16).
fn layer_tensor_name(l: usize, tail: &str) -> String {
    format!("{GLIMMER_LAYER_PREFIX}.{l}.{tail}")
}

/// Place one whole layer into the tier — the pinned path.
///
/// Extracted so [`Slot::new`] can build the same twelve-FIELD pin over slot-relative
/// addresses. Both go through [`place_norm`] and [`place_proj`], so both keep the extent and
/// shape checks.
fn place_layer(
    tier: &mut DeviceTier,
    st: &Safetensors,
    s: Schema<'_>,
    l: usize,
) -> Result<GlimmerLayerPin> {
    let p = format!("{GLIMMER_LAYER_PREFIX}.{l}");
    let norm = |tier: &mut DeviceTier, t: &str| {
        place_norm(tier, st, s.cfg.hidden, &format!("{p}.{t}.weight"))
    };
    let proj = |tier: &mut DeviceTier, t: &str| place_proj(tier, st, s, &p, t);
    Ok(GlimmerLayerPin {
        input_ln: norm(tier, "input_layernorm")?,
        post_attn_ln: norm(tier, "post_attention_layernorm")?,
        pre_ffn_ln: norm(tier, "pre_feedforward_layernorm")?,
        post_ffn_ln: norm(tier, "post_feedforward_layernorm")?,
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
    /// Reserve a layer-sized region and precompute the layer's addresses inside it — twelve
    /// on a bf16 artifact, twenty on an fp8 one.
    ///
    /// **Built by placing layer 0 and then recording where each tensor landed.** That reuses
    /// the pinned path's shape and extent checks instead of restating a layout — the
    /// alternative was a second offset table, which is a second thing to get wrong and which
    /// jscpd would have matched against the first. The bytes placed here are layer 0's and are
    /// overwritten by the first [`Self::fill`]; what survives is the geometry.
    ///
    /// **A refill's destination is the PIN'S OWN ADDRESS — there is no offset arithmetic at
    /// all.** Two earlier shapes in `old:` were worse and both were caught by review
    /// 2026-08-12: one recomputed `next_multiple_of(256)` per tensor to predict where `place`
    /// had put things (a second copy of the bump allocator, correct only while it agreed); the
    /// other stored `base = addrs[0]` plus offsets obtained by subtracting pointers, which
    /// rested on the first placement being the LOWEST address — true for a bump allocator,
    /// promised by nothing, and an underflow into `base.add(huge)` if it ever changed,
    /// silently under `--release` where overflow checks are off.
    fn new(tier: &mut DeviceTier, st: &Safetensors, s: Schema<'_>) -> Result<Self> {
        let pin = place_layer(tier, st, s, 0)?;
        let mut tensors = Vec::new();
        for tail in layer_tails(s)? {
            let len = st.raw(&layer_tensor_name(0, &tail))?.0.len();
            tensors.push((tail, len));
        }
        // The two sides of every future refill zip: computed by DIFFERENT walks (the pin's
        // field-by-field placement against the config-driven tail list), so their agreement is
        // asserted ONCE here rather than trusted. This is the check that survives — unlike the
        // one on `fill`'s old third leg, which the parameter's removal made unwritable, this
        // pair genuinely could diverge without anyone noticing.
        ensure!(
            pin.addrs().len() == tensors.len(),
            "slot geometry: {} addresses for {} tensors — the address walk and the tail list \
             have diverged",
            pin.addrs().len(),
            tensors.len()
        );
        Ok(Self { pin, tensors })
    }

    /// Overwrite this slot with layer `l`'s bytes.
    ///
    /// A host `memcpy` per tensor, from the mmap'd artifact into the tier — the same operation
    /// [`DeviceTier::place`] performs, and valid for the same reason: under HIP's unified
    /// addressing the tier's device pointer IS a host address. **That coincidence is not
    /// portable and `place`'s own doc says callers must not depend on it** — which is why this
    /// goes through [`DeviceTier::write_to`], so the one place that knows about unified
    /// addressing stays the one place.
    ///
    /// `ponytail:` buffered I/O through the mmap, not the O_DIRECT io_uring path
    /// [`crate::fetch`] gives the routed experts. That path reads per-layer sidecar files whose
    /// blocks are aligned for O_DIRECT; a safetensors tensor starts wherever the header left
    /// it. **Upgrade path: `convert_glimmer` grows a layer-blocked output and this becomes an
    /// `AsyncFetch` submit.** Until then the page cache is doing the caching, which anyone
    /// quoting a bytes-from-disk number for this path must account for first.
    fn fill(&mut self, st: &Safetensors, l: usize) -> Result<()> {
        // TWO legs, both this slot's own and equal in length by `Self::new`'s assertion. A
        // third leg arriving as a parameter is what review found unguarded 2026-08-16.
        for (&dst, (tail, len)) in self.pin.addrs().iter().zip(&self.tensors) {
            let name = layer_tensor_name(l, tail);
            let bytes = st.raw(&name)?.0;
            ensure!(
                bytes.len() == *len,
                "{name} is {} bytes but layer 0's was {len} — every Glimmer layer is the same \
                 shape, so this artifact is not the one this slot was laid out for",
                bytes.len()
            );
            // SAFETY: `dst` is the address `DeviceTier::place` returned for this tensor of
            // layer 0, and `bytes.len() == len` is that placement's own extent (checked
            // immediately above), so the write stays inside the placement it targets.
            unsafe { DeviceTier::write_to(dst as *mut u8, bytes) };
        }
        Ok(())
    }
}

/// What [`GlimmerPin::build`] needs beyond the artifact: the run's device budget and the
/// context it will allocate KV for. Bundled for the same reason as
/// [`GlmPinCfg`](crate::glm::pin::GlmPinCfg): every field is a startup-time decision.
///
/// **`n_ctx` and not two pre-computed byte counts.** The KV cache and the activation scratch
/// are both charges this pin's floor must cover, and both are functions of `(cfg, n_ctx)` —
/// handing them in as numbers would let the bytes the floor charged for and the bytes the
/// engine then allocates be two different calls, which is exactly the drift
/// [`super::geometry::floor_of`] exists to make impossible.
#[derive(Clone, Copy)]
pub struct GlimmerPinCfg {
    /// Total device budget (`--max-mem`, auto-discovered when absent).
    pub capacity: usize,
    /// The run's context in positions — prompt plus generated, not
    /// `max_position_embeddings`.
    pub n_ctx: usize,
}

/// The Muse Glimmer weight set: the pinned prefix, plus [`STREAM_SLOTS`] slots for the rest.
///
/// Layers `0..pinned.len()` live in the tier for the run; the rest cycle through the slots.
/// **Which one a caller got is not observable** — [`Self::layer`] returns the same
/// [`GlimmerLayerPin`] shape either way, and the BYTES behind it are identical at every
/// budget. That indistinguishability is P4 (the budget trades speed, never text) expressed as
/// a type rather than as a convention.
pub struct GlimmerPin {
    #[allow(dead_code)] // RAII owner of the slab every pointer below points into.
    tier: DeviceTier,
    /// The mmap'd artifact, kept alive because a streamed layer is read from it on every
    /// visit. An all-resident pin holds it too and never reads it again — one field rather
    /// than an `Option` nothing checks.
    src: Safetensors,
    /// `[vocab, hidden]` bf16, 2.690 GB.
    pub embed: Bf16Weight,
    /// `lm_head.weight`, a SECOND 2.690 GB — `tie_word_embeddings` is false and both ship.
    pub head: Bf16Weight,
    pub final_norm: *const f32,
    /// Layers the budget pinned, indexed by ABSOLUTE layer id — a PREFIX, so the index is the
    /// layer id with no offset to get wrong.
    pinned: Vec<GlimmerLayerPin>,
    /// The streaming slots: fixed device addresses, refilled per visit. Empty when the budget
    /// pinned everything, which is why an all-resident run allocates none.
    slots: Vec<Slot>,
    /// Which layer each slot currently holds, so a re-visit inside one slot's lifetime is a
    /// hit rather than a second copy of the same bytes.
    slot_layer: Vec<Option<usize>>,
    /// The model's layer count, carried because `pinned.len()` is a partition rather than the
    /// model — without it, "is `l` a real layer?" would be unanswerable here.
    n_layers: usize,
    /// The partition that placed this run. Kept so the startup log and tests can cite the
    /// decision rather than re-deriving it — the disagreement between a reported split and a
    /// built one is one this repo has already been bitten by.
    pub placement: Partition,
    /// Cheap counters. Not a policy input — see [`Self::layer`] on why there is no policy —
    /// but the only way to tell a partition that is working from one that thrashes.
    hits: u64,
    fills: u64,
}

impl GlimmerPin {
    /// Build from artifact directory `dir`, pinning as many whole layers as the budget allows.
    pub fn build(dir: &str, cfg: &GlimmerTextConfig, pin: GlimmerPinCfg) -> Result<Self> {
        // **The cheap refusals first, then the 55.712 GB allocation.** `open_dir` is where a
        // missing or malformed shard is found; it sat BELOW `DeviceTier::new` in `old:` until
        // review pointed out that this made a stale artifact pay a full-model allocation
        // before its cheap refusal fired.
        let st = Safetensors::open_dir(dir)?;
        // THE FORMAT IS THE ARTIFACT'S, sniffed by dtype — no flag exists (M11's contract).
        // Sniffed before any byte arithmetic because everything below — the unit size, the
        // floor's slot charge, the tier request — is a function of it.
        let fmt = ProjFmt::sniff(dir, &st)?;
        let schema = Schema { cfg, fmt };
        let layer = layer_bytes(cfg, fmt)?;

        // THE PLACEMENT DECISION — **one author, asked twice.** `partition()` is still the
        // only thing that decides what is resident; what the second call changes is the
        // FLOOR, because a streaming slot is a charge the run pays only if it streams and
        // `Floor` cannot express that conditionality (see `floor_of`).
        //
        // First: does the whole model fit with no slot at all? That is P1's degenerate happy
        // case — everything resident, the streaming path idles and allocates nothing.
        let units = layer_units(cfg.n_layers, layer)?;
        let budget = Bytes(pin.capacity as u64);
        let resident_only = floor_of(cfg, pin.n_ctx, 0, fmt)?;
        let placement = partition(&units, budget, resident_only)
            .map_err(|r| below_floor(r, cfg, resident_only))?;
        // Only when something streams is the slot real, and only then is the answer that
        // gets executed the one taken against a floor that reserves it. Charging it
        // unconditionally costs a whole PINNED layer at the budgets where the model just
        // fits — 967.942 MB of host memcpy per token, bought for nothing.
        let (placement, n_slots) = if placement.streamed.is_empty() {
            (placement, 0)
        } else {
            let with_slot = floor_of(cfg, pin.n_ctx, STREAM_SLOTS, fmt)?;
            let p =
                partition(&units, budget, with_slot).map_err(|r| below_floor(r, cfg, with_slot))?;
            (p, STREAM_SLOTS)
        };

        // **Ask the tier for what the partition USES, never for the whole budget.**
        // `DeviceTier::new` allocates its capacity rather than treating it as a ceiling AND
        // feeds `guard_capacity`, so an over-request both wastes GTT and can turn a workable
        // budget into a refusal.
        let n_pinned = placement.pinned.len();
        let tier_cap = global_bytes(cfg) + (n_pinned + n_slots) * layer + PIN_SLACK;
        // The FORMAT and the tier bytes in one line: the fp8 tier is ~half the bf16 one, so
        // this figure is the cheap independent witness against the named silent-fallback
        // failure — a run that claims fp8 while reporting a bf16-sized tier is lying to
        // itself, and this is where it shows.
        tracing::info!(
            "partition: {n_pinned} of {} layers pinned, {} streamed through {n_slots} slot(s) \
             ({:?} projections, {:.1} GiB tier)",
            cfg.n_layers,
            placement.streamed.len(),
            fmt,
            tier_cap as f64 / (1u64 << 30) as f64,
        );

        // **Every layer's headers are checked, whether or not the budget pins it.**
        //
        // > **Found by review in `old:` 2026-08-12, and it was a correctness regression the
        // > budget introduced.** Before a partition existed, `build` placed all 52 layers, so
        // > all 52 went through the dtype and shape checks. Afterwards only the PINNED prefix
        // > did, and a streamed layer's only check was `Slot::fill`'s byte length against
        // > layer 0's — so **which checks a layer received became a function of `--max-mem`.**
        // > A short norm loaded clean at a low budget and died mid-token with "not the one
        // > this slot was laid out for", diagnosis gone; a `q_proj` stored transposed is
        // > byte-identical in length and was accepted outright.
        //
        // Headers only: this reads the index, not the bytes, so it costs 12 lookups per
        // streamed layer on a bf16 artifact and 20 on an fp8 one, and no device memory. It also keeps the ordering this function argues
        // for above — the cheap refusals before the big allocation.
        for l in n_pinned..cfg.n_layers {
            check_layer_headers(&st, schema, l)?;
        }

        let mut tier = DeviceTier::new(tier_cap)?;
        let embed = place_global(
            &mut tier,
            &st,
            cfg,
            "model.language_model.embed_tokens.weight",
        )?;
        let head = place_global(&mut tier, &st, cfg, "lm_head.weight")?;
        let final_norm = place_norm(
            &mut tier,
            &st,
            cfg.hidden,
            "model.language_model.norm.weight",
        )?;
        let mut pinned = Vec::with_capacity(n_pinned);
        for l in 0..n_pinned {
            pinned.push(place_layer(&mut tier, &st, schema, l)?);
        }
        let mut slots = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            slots.push(Slot::new(&mut tier, &st, schema)?);
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
            placement,
            hits: 0,
            fills: 0,
        })
    }

    /// Layer `l`'s resolved weights — **pinned or streamed, indistinguishably.**
    ///
    /// `&mut self` because a miss fills a slot. That is the honest signature: a caller cannot
    /// hold two layers' pins at once, which is exactly the aliasing a slot reuse would break.
    ///
    /// # THE WRITE-AFTER-READ HAZARD, and who closes it
    ///
    /// **Before this refills a slot, every kernel still reading that slot's previous occupant
    /// must have retired.** A fill is a host `memcpy` with no synchronization of any kind, and
    /// kernel launches are asynchronous — so a host that runs even one layer ahead of the
    /// device can overwrite weights a live GEMM is streaming. The symptom is
    /// position-dependent, nondeterministic wrong text: this repo's arena-relocation signature.
    ///
    /// **It is not an invariant a caller owes: this function performs the `device_sync`
    /// itself**, on every miss. With the fence removed, `old:`'s gate measured all 4096 rows of
    /// a live gemm reading the overwritten bytes.
    ///
    /// **SCOPE, and it is narrower than "the hazard is closed".** The sync orders the refill
    /// after every kernel ALREADY ENQUEUED when this is called. It cannot help the other
    /// order: [`Bf16Weight`] is `Copy` and every norm field is a raw pointer, so the `&mut
    /// self` borrow that serialises these calls ends the moment a caller copies one out — and
    /// a caller that captures layer `l`'s pointers, calls `layer(l+1)` (which fences and
    /// refills), and only THEN launches the `l`-th kernel reads `l+1`'s weights as `l`'s.
    /// Nothing here can see that. **Do not launch from pointers captured across a `layer()`
    /// call.** `super::forward::Handles` is what makes that structural for this tree's loop.
    ///
    /// **There is deliberately no ticket.** A ticket expresses fill-then-read, the dependency
    /// that genuinely does not exist while the fill is synchronous, and an always-satisfied
    /// one is the `hit: Vec<bool>` mistake [`crate::fetch::asyncfetch`]'s `Ticket` doc
    /// records. The dependency that DOES exist is the opposite one, and it is the
    /// `device_sync` below. When the fill goes async, fill-then-read becomes real and this
    /// whole-device join becomes the wrong instrument for it.
    pub fn layer(&mut self, l: usize) -> Result<&GlimmerLayerPin> {
        ensure!(
            l < self.n_layers,
            "layer {l} is past this model's {} layers",
            self.n_layers
        );
        if l < self.pinned.len() {
            return Ok(&self.pinned[l]);
        }
        // Round-robin over the streamed suffix. **At `STREAM_SLOTS` = 1 this maps every
        // streamed layer to slot 0, so it separates nothing** — and it is not what stops a
        // fill landing on a slot a kernel is still reading. Only the fence below does.
        let s = (l - self.pinned.len()) % self.slots.len();
        if self.slot_layer[s] == Some(l) {
            self.hits += 1;
            return Ok(&self.slots[s].pin);
        }
        // **THE WRITE-AFTER-READ FENCE** — the hazard is on this function's doc; this closes
        // it. **Unconditional on a miss, not conditional on `slot_layer[s].is_some()`:** the
        // narrower version reads as tighter and is wrong, because a fill that failed halfway
        // leaves `slot_layer[s]` at `None` over a slot whose previous occupant's pointers WERE
        // handed out — so it would skip the fence exactly when the slot is least trustworthy.
        rivoli_backend::device_sync()?;
        // **Invalidated BEFORE the fill, not after it.** `fill` writes tensor-by-tensor and
        // can bail in the middle, at which point the slot holds a prefix of layer `l` and a
        // suffix of its previous occupant. Assigning only on success left the flag claiming
        // the OLD layer, so a caller that handled the error and re-requested that old layer
        // took the hit path and got the mixture, silently.
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

    /// `(hits, fills)` — how many layer visits found their slot already loaded, and how many
    /// paid a 967.942 MB host memcpy.
    ///
    /// **The only way to see whether a prefill is actually layer-major**, which is a residency
    /// property and therefore invisible to every numeric gate: the reorder is bit-for-bit
    /// identical arithmetic, so nothing that compares outputs can tell it from a token-major
    /// loop.
    pub fn slot_stats(&self) -> (u64, u64) {
        (self.hits, self.fills)
    }
}

/// The layers as residency units, in layer order — the priority order handed to
/// `partition()`.
///
/// Layer order because decode's access is exactly cyclic over it: pinning a prefix is the
/// static partition that cyclic access makes optimal (the Belady degenerate, argued in this
/// module's header). Any future per-layer evidence that beats uniform arrives as a REORDERING
/// of this list, never as a second placement author.
fn layer_units(n_layers: usize, layer: usize) -> Result<Vec<Unit>> {
    let bytes = std::num::NonZeroU64::new(layer as u64)
        .ok_or_else(|| anyhow::anyhow!("a Glimmer layer computes to 0 bytes"))?;
    Ok((0..n_layers)
        .map(|l| Unit {
            id: UnitId(u32::try_from(l).unwrap_or(u32::MAX)),
            bytes,
        })
        .collect())
}

/// A [`Refusal`] with the charges that make it up named.
///
/// `Refusal`'s own `Display` gives the two totals; what an operator needs on top is WHICH of
/// them they can move. `--max-mem` raises the budget, `--ctx` lowers the KV cache and the
/// scratch, and the model-level tensors and the slot are not negotiable at all — so the
/// message splits the floor along exactly that line.
fn below_floor(
    r: Refusal,
    cfg: &GlimmerTextConfig,
    floor: rivoli_core::residency::Floor,
) -> anyhow::Error {
    anyhow::anyhow!(
        "{r} (Muse Glimmer pin, --max-mem). The model-level tensors are {:.3} GB (embed + \
         lm_head + final norm, each read once per TOKEN, so streaming them would cost more \
         than it frees), the streaming slots are {:.3} GB, and the KV cache plus activation \
         scratch at this context are {:.3} GB — lower --ctx or raise --max-mem",
        global_bytes(cfg) as f64 / 1e9,
        floor.slot_bytes.0 as f64 / 1e9,
        (floor.kv_at_max_ctx.0 + floor.scratch.0) as f64 / 1e9,
    )
}

/// One streamed layer's tensors, checked against the config from the index alone.
/// See [`GlimmerPin::build`]'s note for why this runs on the layers the budget did NOT pin —
/// and `resident::place_fp8`'s for why the fp8 arm checks the SCALE GRID here too: in `old:`
/// the grid check existed only on a path the shipping partition never took, and a streamed
/// grid of the wrong extent has the kernel reading a neighbour tensor's bytes as f32 scales.
fn check_layer_headers(st: &Safetensors, s: Schema<'_>, l: usize) -> Result<()> {
    let check = |name: &str, want: &[usize], want_dtype: Dtype| -> Result<()> {
        let (_, dtype, shape) = st.raw(name)?;
        ensure!(
            shape == want && dtype == want_dtype,
            "{name} is {shape:?} {dtype:?}, but this config implies {want:?} {want_dtype:?} — \
             the artifact and the config describe different models"
        );
        Ok(())
    };
    for t in GLIMMER_LAYER_TENSORS {
        let want = s.cfg.layer_tensor_shape(t)?;
        let name = format!("{GLIMMER_LAYER_PREFIX}.{l}.{t}");
        let want_dtype = match (want.len(), s.fmt) {
            (1, _) => Dtype::F32,
            (_, ProjFmt::Bf16) => Dtype::Bf16,
            (_, ProjFmt::Fp8 { .. }) => Dtype::F8E4M3,
        };
        check(&format!("{name}.weight"), &want, want_dtype)?;
        if let (2, ProjFmt::Fp8 { block }) = (want.len(), s.fmt) {
            let grid = [want[0].div_ceil(block), want[1].div_ceil(block)];
            check(&format!("{name}.weight_scale_inv"), &grid, Dtype::F32)?;
        }
    }
    Ok(())
}

/// Place a `[vocab, hidden]` bf16 model-level tensor (embed / lm_head), shape checked.
fn place_global(
    tier: &mut DeviceTier,
    st: &Safetensors,
    cfg: &GlimmerTextConfig,
    name: &str,
) -> Result<Bf16Weight> {
    let (bytes, shape) = st.typed(name, Dtype::Bf16)?;
    ensure!(
        shape == [cfg.vocab, cfg.hidden],
        "{name} is {shape:?}, but this config implies [{}, {}]",
        cfg.vocab,
        cfg.hidden
    );
    Ok(Bf16Weight {
        packed: tier.place(bytes)? as *const u16,
        o_dim: cfg.vocab,
        i_dim: cfg.hidden,
    })
}

/// f32 device scratch of `n` elements. Shared by every activation allocation in
/// [`super::engine`], which is why it lives beside the pin rather than being spelled at each
/// of the fourteen call sites.
pub fn scratch(n: usize) -> Result<DeviceBuf> {
    DeviceBuf::new(n * 4)
}
