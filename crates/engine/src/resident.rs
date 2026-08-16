//! What a resident weight set is made of: the resolved-weight types every arm's pin hands
//! to a launcher, the placers that build them out of a `Safetensors`, the artifact's own
//! byte total, and the [`Floor`] a weights-only budget states.
//!
//! **This is a factoring, not a new authority.** Every item below was written once per arm
//! in `old:src/memory/pin.rs` — where all three pins shared one file and therefore one copy
//! — and the rewrite's one-file-per-arm split gave each a chance to acquire its own. Two of
//! them already had: [`Fp8Weight`] and [`place_fp8`] were `crate::glm::pin`'s, [`Bf16Weight`]
//! was `crate::glimmer::pin`'s, and the V4 arm needs both. `build.rs`'s duplication gate
//! reports a third copy on the first compile, which is the gate arriving at this file.
//!
//! **What did NOT move, and the rule that decided it.** A placer belongs here when the
//! semantics coincide, not when the shape does. `glimmer::pin::place_proj` checks the tensor
//! against the shape its config implies and `place_norm` checks a norm's extent; those are
//! Glimmer's own artifact contract and they stay with it. `glm::pin::place_i8` stays for the
//! opposite reason — int8 per-row scaling is GLM's embed/head format and no other arm has
//! one. What is here is the intersection: "reserve this tensor's bytes in the tier and
//! resolve the dims off its own shape".
//!
//! Not gated on the backend at the module level, because `lib.rs` gates it: every item takes
//! a [`DeviceTier`], which does not exist without one.

use crate::device::DeviceTier;
use crate::routed::{PoolCfg, pool_budget};
use anyhow::{Context, Result, ensure};
use rivoli_artifact::format::{Dtype, Safetensors};
use rivoli_core::residency::{Bytes, Floor, Partition, Refusal, Unit, UnitId, partition};

/// An fp8-e4m3 block-scaled weight resolved to device addresses, with the dims the launch
/// site needs. Dims ride the weight (taken from the tensor's own shape at place time) rather
/// than `cfg`, so a mis-shaped artifact fails at load, not at launch.
#[derive(Clone, Copy)]
pub struct Fp8Weight {
    pub packed: *const u8,
    pub scale: *const f32,
    pub block: usize,
    pub o_dim: usize,
    pub i_dim: usize,
}

/// A three-projection fp8 SwiGLU MLP: gate (`w1`), up (`w3`), down (`w2`).
///
/// GLM's three DENSE layers and V4's always-resident SHARED expert are the same object under
/// two names — one weight set, three fp8 block-scaled projections, read by every row — so
/// this is one type. What differs is which kernel combines them (GLM's plain `swiglu`, V4's
/// `swiglu_clamped_bf16`), and that is a property of the layer loop, not of the weights.
///
/// Named fields rather than an array, on `V4_PROJ`'s argument: `w1` and `w3` have IDENTICAL
/// shapes, so a swap of gate and up is invisible to every structural check there is, and the
/// only defence is that the two names are written once, at each arm's placer.
#[derive(Clone, Copy)]
pub struct Fp8Mlp {
    pub gate: Fp8Weight,
    pub up: Fp8Weight,
    pub down: Fp8Weight,
}

/// A bf16 weight matrix resolved to a device address, with the dims the launch site needs.
///
/// Distinct from an f32 buffer rather than widened at load: V4 carries `embed` and `head` as
/// bf16 and widening them would double 2.1 GB of resident set to 4.2 GB to paper over a
/// missing kernel — and whether to requantize is a quality question with a paired-dNLL
/// measurement attached, not a loader's decision.
///
/// `*const u16` and not `*const u8`, which is what every launcher that consumes one already
/// declares: bf16 is the element type, so the pointer that names it should be the one the
/// ABI wall takes. A refill path that genuinely deals in bytes casts at its own site.
#[derive(Clone, Copy)]
pub struct Bf16Weight {
    pub packed: *const u16,
    pub o_dim: usize,
    pub i_dim: usize,
}

/// A weight matrix's `(o_dim, i_dim)`, refusing anything that is not 2-D. Every placer that
/// carries dims takes them from the tensor's own shape rather than from `cfg`, so this is the
/// one place the rank is confronted.
pub fn dims2(name: &str, shape: &[usize]) -> Result<(usize, usize)> {
    ensure!(shape.len() == 2, "{name}: expected 2-D, got {shape:?}");
    Ok((shape[0], shape[1]))
}

/// Place an F32 tensor (norms, router gate, the hyper-connection tables) into the tier.
pub fn place_f32(tier: &mut DeviceTier, st: &Safetensors, name: &str) -> Result<*const f32> {
    let (bytes, _) = st.typed(name, Dtype::F32)?;
    // f32 LE host == LE device.
    Ok(tier.place(bytes)? as *const f32)
}

/// Place a bf16 matrix verbatim (V4's `embed` / `head`). Dims from its `[o_dim, i_dim]` shape.
pub fn place_bf16(tier: &mut DeviceTier, st: &Safetensors, name: &str) -> Result<Bf16Weight> {
    let (w, shape) = st.typed(name, Dtype::Bf16)?;
    let (o_dim, i_dim) = dims2(name, shape)?;
    Ok(Bf16Weight {
        packed: tier.place(w)? as *const u16,
        o_dim,
        i_dim,
    })
}

/// Place an fp8-e4m3 block-scaled weight (`<name>.weight` F8E4M3 + `.weight_scale_inv` F32)
/// into the tier. Dims come from the weight's `[o_dim, i_dim]` shape.
///
/// **The SCALE GRID's shape is checked.** The kernel indexes
/// `scale[(o/block)·sc_cols + i/block]` with `sc_cols` derived from `i_dim` — so a grid
/// stored `[sc_cols, sc_rows]` has every tile taking a neighbour's scale (fluent wrong text,
/// no error), and a SHORTER grid has the kernel reading past the placement into the next
/// tensor's e4m3 bytes reinterpreted as f32. Found by review in the old tree on the Glimmer
/// fp8 path, where the check ran only for streamed layers and the shipping partition pinned
/// all 52 — it iterated zero times.
///
/// The V4 arm inherited the check by using this placer rather than the reference's, which had
/// none: `old:src/memory/pin.rs::place_fp8` placed both tensors and read the grid's shape not
/// at all. That is the whole return on the factoring above.
pub fn place_fp8(
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

/// Every `*.safetensors` byte in `dir`, optionally skipping one file by name.
///
/// **It is the artifact's own file length, not a second derivation of the placement layout.**
/// A converter writes the resident file from exactly the tensor list its pin places, so the
/// file IS the footprint — the old tree replaced 73 lines of per-shard re-derivation with
/// this, because the copy nothing executed was free to drift. The bias is deliberate:
/// over-count only shrinks the routed pool; under-count is what `DeviceTier::place` bails on.
/// It over-counts by each file's header and by whatever the pin reads to the HOST rather than
/// placing (GLM's router bias, V4's `tid2eid` and gate bias).
///
/// `skip` is GLM's unplaced `indexer.safetensors`; V4 and Glimmer pass `None`. It is a
/// parameter rather than a filter each caller applies because the answer is one arithmetic
/// with one exception, and two copies of a directory walk is what this replaced.
pub fn safetensors_bytes(dir: &str, skip: Option<&str>) -> Result<usize> {
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

/// The [`Floor`] a **weights-only** budget states: the always-resident tier, plus the pool's
/// minimum batch slots, and nothing for KV or scratch.
///
/// **The two zeros are a semantic claim about `--max-mem`, not an assertion that KV is
/// free.** Both routed-expert arms' `--max-mem` has always budgeted weights only — every
/// recorded benchmark in `docs/measurement/benchmarks.md` reads it that way — so folding KV
/// in is a change to what the flag MEANS, owed its own measured change rather than a quiet
/// one inside a port. `crate::glimmer::pin` deliberately does NOT use this: its floor charges
/// for the KV cache and the activation scratch, because that arm's flag was defined that way
/// from the start and its per-layer state is the same order as its weights.
///
/// What the omission costs, per arm, so a reader can size the lie. GLM allocates its KV slabs
/// outside this (~51 KB/token) and its activation scratch is `MAXROW`-shaped, i.e. constant.
/// **V4's is the larger one and it is NOT constant**: its prefill is one whole-prompt pass, so
/// every `[m, ..]` buffer is sized for `--ctx` — `crate::v4::engine::V4Engine::max_m` states
/// the shape — and the per-layer `[ring ‖ compressed]` cache adds
/// `(window + max_ctx/ratio) · head_dim · 4` bytes per layer on top. Both grow with `--ctx`
/// while this floor does not, which is the direction that matters: raising `--ctx` shrinks the
/// pool by an amount the partition never sees.
///
/// No total is stated here on purpose. It is a product of six config values and would be a
/// prose number nothing checks; what a reader needs is that the term EXISTS, is O(`--ctx`),
/// and is unbudgeted.
///
/// One function and not a literal per pin, because the literal is four fields of one type
/// and a transposed pair type-checks: `always_resident` and `slot_bytes` are both a
/// [`Bytes`] and swapping them sizes the tier from the batch and the batch from the tier.
///
/// Private: [`plan_pool`] is the only caller and the only way a routed arm should reach a
/// [`Floor`], because stating the floor and executing the partition over it are one act.
fn weights_only_floor(tier_cap: usize, slot_bytes: usize) -> Floor {
    Floor {
        always_resident: Bytes(tier_cap as u64),
        kv_at_max_ctx: Bytes(0),
        scratch: Bytes(0),
        slot_bytes: Bytes(slot_bytes as u64),
    }
}

/// `count` equal-sized streamable units with dense ascending ids — the priority list
/// `partition()` pins a PREFIX of.
///
/// The ORDER is the caller's claim and stays documented at the caller: both routed-expert
/// arms enumerate layer-major, because decode's access is cyclic over layers and a static
/// prefix is what cyclic access makes optimal. What is here is only the construction, which
/// is where the saturating conversions live and is identical wherever the units are equal.
///
/// `saturating` rather than `?` on both casts: a `usize` that does not fit a `u32` id means
/// four billion experts, and a partition over a truncated list would be a wrong answer where
/// a clamped one is merely a refusal downstream.
pub fn stream_units(count: usize, unit: usize) -> Vec<Unit> {
    let unit = u64::try_from(unit).unwrap_or(u64::MAX);
    let bytes = std::num::NonZeroU64::new(unit).unwrap_or(std::num::NonZeroU64::MIN);
    (0..count)
        .map(|i| Unit {
            id: UnitId(u32::try_from(i).unwrap_or(u32::MAX)),
            bytes,
        })
        .collect()
}

/// The run's device budget and the routed pool's startup knobs — the four things a pin needs
/// that come from the command line rather than from the artifact.
///
/// One type for both routed arms. **GLM's `--mode` is deliberately NOT in it**: V4's routed
/// format is `.f4` because that is what the checkpoint stores, so it has no `RoutedFmt` to
/// carry and a field it always filled in the same way would be exactly the "knob nothing
/// spends" `rivoli_core::legality` exists to stop. GLM takes its format as its own argument.
#[derive(Clone, Copy)]
pub struct PinCfg<'a> {
    /// Total device budget (`--max-mem`, auto-discovered when absent).
    pub capacity: usize,
    pub cache_policy: &'a str,
    pub two_q: rivoli_core::cache::TwoQSplit,
    pub trace_path: Option<&'a str>,
}

/// How many experts one `submit` carries, and what one costs — the shape an arm's MoE
/// kernels impose on the pool.
///
/// Constructed rather than written, because the bound is the pool's and every arm was
/// checking it with its own `ensure!` and its own wording. `top_k` and `max_batch` are
/// different numbers whenever an arm batches rows (GLM's is the UNION of `rows` rows' picks
/// plus its folded shared expert), and sizing the pool's scratch from `top_k` alone left GLM
/// budgets between 8 and 16 slots passing startup and failing mid-run — which is the case
/// this type exists to make unwriteable.
#[derive(Clone, Copy)]
pub struct Batch {
    /// One layer's demand count; sizes the io_uring ring, not the batch scratch.
    pub top_k: usize,
    /// The largest `submit` batch this arm will send.
    pub max_batch: usize,
    /// One streamed expert's slot stride.
    pub unit_bytes: usize,
}

impl Batch {
    /// The batch a pass of `rows` token rows submits: the UNION of their picks, plus
    /// `n_shared` routed-format shared blocks folded in.
    ///
    /// `rows` and `n_shared` are the two arm-shaped facts. GLM batches up to `MAXROW` rows
    /// and folds ONE shared expert into the routed dispatch; V4 passes `rows = 1` because its
    /// FP4 kernel has no other instantiation and `n_shared = 0` because its shared expert is
    /// fp8 and RESIDENT rather than a `.f4` block. Both facts are stated at the call sites,
    /// where the kernel that imposes them is named.
    ///
    /// The union is an over-estimate whenever two rows pick the same expert (measured ~31%
    /// overlap on GLM), and that is the right direction: it is what the fixed scratch must
    /// hold, not what a typical layer uses.
    pub fn union(top_k: usize, rows: usize, n_shared: usize, unit_bytes: usize) -> Result<Self> {
        let max_batch = top_k * rows + n_shared;
        ensure!(
            max_batch <= crate::routed::MAX_BATCH,
            "top_k {top_k} x {rows} row(s) + {n_shared} shared = {max_batch} exceeds the \
             {}-slot batch scratch",
            crate::routed::MAX_BATCH
        );
        Ok(Self {
            top_k,
            max_batch,
            unit_bytes,
        })
    }
}

/// What a routed arm brings to the placement decision: the units it can stream, what the
/// resident tier takes, and its [`Batch`] shape.
///
/// A named struct rather than four more arguments on [`PoolPlan::decide`], so the pin's three
/// inputs to the decision read as what they are at the one call site each arm has.
pub struct PoolPlan<'a> {
    /// Names the arm in the refusal, so a `--max-mem` message says which pin could not be
    /// placed on a machine that holds more than one artifact.
    arm: &'static str,
    /// The priority list, in the order a prefix of it should be pinned. See [`stream_units`].
    units: &'a [Unit],
    /// The always-resident tier, INCLUDING its alignment slack — what the floor charges.
    tier_cap: usize,
    batch: Batch,
}

impl<'a> PoolPlan<'a> {
    /// The inputs to one arm's placement decision. Four positional arguments and no
    /// transposition hazard: every type is distinct, which is why this is a constructor
    /// rather than a literal a caller fills in.
    pub fn new(arm: &'static str, units: &'a [Unit], tier_cap: usize, batch: Batch) -> Self {
        Self {
            arm,
            units,
            tier_cap,
            batch,
        }
    }

    /// **THE PLACEMENT DECISION**, for both routed arms: state the floor, execute the
    /// partition, and turn its answer into the pool's byte budget and config.
    ///
    /// Returns rather than allocates, and that is load-bearing: every caller runs this BEFORE
    /// any device allocation, so a budget below the floor is a refusal with the arithmetic in
    /// it at startup and the run never degrades (P6). Opening the pool here would put a
    /// ~100 GiB VMM reservation ahead of the resident tier's, which is the reverse of the
    /// order both pins want — the tier is the allocation that must not fail.
    ///
    /// The pool's byte budget IS the partition: batch slots plus the pinned prefix.
    /// [`pool_budget`]'s O_DIRECT rounding still applies, because the arena anchors HOT slots
    /// at the high end and an unaligned budget would misalign every hot-slot DMA destination.
    ///
    /// One method and not a sequence per pin, because the sequence is the same five steps over
    /// the same numbers and `build.rs`'s duplication gate said so the moment the second routed
    /// arm landed. What stays at each caller is what differs: which units, which batch shape,
    /// and any arm-specific line the log owes — V4 prints its routed set's size and residency
    /// fraction, because that percentage is what explains every later V4 measurement.
    ///
    /// **A method on the plan, so each arm's call is ONE statement.** Two callers of one
    /// function is not duplication, but the residue is still a token run, and a four-field
    /// struct literal is five lines under rustfmt's `struct_lit_width` — long enough for the
    /// gate to match it. Written as a statement it is what it should have been anyway.
    pub fn decide(self, pin: PinCfg<'a>) -> Result<(Partition, PoolCfg<'a>)> {
        let b = self.batch;
        let floor = weights_only_floor(self.tier_cap, b.max_batch * b.unit_bytes);
        let placement = partition(self.units, Bytes(pin.capacity as u64), floor)
            .map_err(|r: Refusal| anyhow::anyhow!("{r} ({} pin, --max-mem)", self.arm))?;
        let budget = pool_budget(
            self.tier_cap + (b.max_batch + placement.pinned.len()) * b.unit_bytes,
            self.tier_cap,
        );
        tracing::info!(
            "{} partition: {} of {} routed experts fit resident beyond the {} batch slots \
             ({:.1} GiB pool)",
            self.arm,
            placement.pinned.len(),
            self.units.len(),
            b.max_batch,
            budget as f64 / (1u64 << 30) as f64,
        );
        let cfg = PoolCfg {
            budget,
            top_k: b.top_k,
            max_batch: b.max_batch,
            policy: pin.cache_policy,
            two_q: pin.two_q,
            trace: pin.trace_path,
        };
        Ok((placement, cfg))
    }
}
