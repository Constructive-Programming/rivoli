//! The pool-backed pin: the same placement machinery as its parent, with the experts behind
//! [`RoutedPool`] instead of in resident slabs. **Two containers, one pin** — `.mxfp8` and
//! `.f4` differ in the expert bytes and their scale grid, and in nothing this file decides.
//!
//! **One pin, both members.** Small ships `.mxfp8` natively and Mini's native 24-layer
//! checkpoint does too (layers 20–23) — the converter turns the whole member into one such
//! artifact, so which member is being opened is a property of the CONFIG above this pin (the
//! loop schedule), not of the container.
//!
//! A child module of [`super`] so the private placers (`place_globals`, `place_dense`,
//! `measure_resident`) and [`PIN_SLACK`](super::PIN_SLACK) are readable here without widening
//! one visibility; the split is the 800-line cap's, and the cohesion line is "the parent
//! places resident slabs, this one places a tier beside a pool".

use super::*;
use crate::resident::{Batch, PinCfg, PoolPlan, stream_units};
use crate::routed::RoutedGeom;
use rivoli_artifact::format::{RoutedFmt, SetDims, f4_layer_range, mxfp8_layer_range};

/// The two containers' pins, emitted from one body.
///
/// **The constructors differ in ONE identifier.** `.mxfp8` and `.f4` are the same tier, the same
/// pool and the same floor discipline over different expert bytes, so a hand-written pair is a
/// five-line parameter list followed by a shared `Self::build_pooled(…, RoutedFmt::` prefix —
/// six lines and 66 tokens, measured 2026-11-25 against this tree's `.jscpd.json`, which the
/// build's duplication gate refuses (it reports a clone, not a near miss, because the common
/// prefix ends exactly where the identifier differs). One body, two names, nothing to drift.
macro_rules! build_routed {
    ($( $(#[$doc:meta])* $build:ident => $fmt:ident ),+ $(,)?) => {$(
        $(#[$doc])*
        pub fn $build(
            dir: &str,
            text: &InklingTextConfig,
            pin: PinCfg<'_>,
            n_ctx: usize,
        ) -> Result<Self> {
            Self::build_pooled(dir, text, pin, (n_ctx, RoutedFmt::$fmt))
        }
    )+};
}

impl InklingPin {
    build_routed! {
        /// Open an Inkling 2 member whose experts are native `.mxfp8`: **both** members, because
        /// the container is the ARTIFACT's fact rather than the member's. Small's 48 stored layers
        /// and Mini's looped 24 are the same pool-backed pin; what differs is the loop schedule
        /// above it, which never reaches this file.
        ///
        /// **A looped pass indexes the STORED layer, so it is not a special case here.** Decode
        /// hands the pool `s.layer` (`decode.rs`'s `step((l, l + offset), …)`), so pass 2 over
        /// `core_start..core_end` looks up the same `(layer, expert)` keys pass 1 did and may HIT
        /// them in the pool — an improvement, not a hazard: neither this pin nor
        /// [`RoutedPool::submit`] assumes the layer sequence ascends (the key is
        /// `expert_key(stored_layer, expert)` and the trace's layer field is a label).
        build_mxfp8 => MxFp8,
        /// Open an Inkling 2 member whose experts are MXFP4 (`.f4`): [`Self::build_mxfp8`]'s pin
        /// over the group-32 fp4 blocks and their ue8m0 scale bytes.
        ///
        /// **Nothing above the container changes.** The slot size, the pool's units, the batch
        /// bound and the preload condition are all `ExpertSet`'s answers, so an `.f4` artifact
        /// pins MORE experts in the same budget (half the bytes each) and the placement executes
        /// verbatim, exactly as it does for `.mxfp8` (P6, INV-8).
        build_f4 => F4,
    }

    /// Open an Inkling 2 member whose experts are `fmt`'s container — the one body both public
    /// constructors above spend, with the container as its only variable.
    ///
    /// **Everything the run will allocate is charged before the pool is sized.** The floor the
    /// partition sees is the resident tier — globals, dense layers, [`PIN_SLACK`] — PLUS the
    /// whole activation and KV scratch [`ScratchSizes`] names, because the tier ([`DeviceTier`])
    /// allocates the weights while [`crate::inkling::engine::InklingScratch`] allocates the scratch
    /// separately: charging both to `PoolPlan`'s `tier_cap` is what leaves the pool a budget
    /// that cannot OOM the buffers the loop needs (P6, P4's memory knob).
    ///
    /// **The set must cover every layer.** Decode runs layers `0..num_hidden_layers` and a
    /// missing expert file would otherwise surface as a read past the pool's table; the
    /// artifact's own range is checked against the config here, at load — and the range is read
    /// from the format's OWN provenance block (`mxfp8_source` / `f4_source`), never inferred
    /// from the layer count, because a partial artifact is normal.
    pub fn build_pooled(
        dir: &str,
        text: &InklingTextConfig,
        pin: PinCfg<'_>,
        (n_ctx, fmt): (usize, RoutedFmt),
    ) -> Result<Self> {
        // The member's NAME, from the one field that distinguishes them: both run this pin, and
        // a log line that said "Small" while a looped Mini was decoding would send the next
        // reader looking for a bug in the wrong member.
        let label = match text.loop_config.is_some() {
            true => "Inkling-2-Mini",
            false => "Inkling-2-Small",
        };
        let path = format!("{dir}/resident.safetensors");
        let st = Safetensors::open_file(&path).with_context(|| format!("open {path}"))?;
        let flags = layer_flags(text);
        let fp8_block = sniff(dir, &st)?;
        let (globals, dense) = measure_resident(&st, text, &flags)?;
        let set = open_routed_set(dir, text, fmt)?;
        let geom = RoutedGeom::new(&set)?;
        let scratch = ScratchSizes::of(text, n_ctx).total();
        let resident = globals + dense + PIN_SLACK;
        let unit = set.expert_slot();
        let units = stream_units(text.num_hidden_layers * text.n_routed_experts, unit);
        let (pinned_experts, total_experts, pool) =
            plan_routed_pool((label, fmt), (resident, scratch), (text, &units), pin)?;
        let parts = Parts {
            fp8_block,
            tier_bytes: resident,
            flags: &flags,
            st,
            text,
        };
        Self::assemble(parts, move |_, _| {
            let mut routed = RoutedPool::new(pool, geom)?;
            if pinned_experts == total_experts {
                routed.preload_all()?;
            }
            Ok(ExpertBackend::MxFp8 {
                set,
                routed: Box::new(routed),
                pinned_experts,
                total_experts,
            })
        })
    }
}

fn open_routed_set(dir: &str, text: &InklingTextConfig, fmt: RoutedFmt) -> Result<ExpertSet> {
    let layers = match fmt {
        RoutedFmt::MxFp8 => mxfp8_layer_range(dir, text.num_hidden_layers)?,
        RoutedFmt::F4 => f4_layer_range(dir, text.num_hidden_layers)?,
        other => bail!(
            "Inkling's routed experts are .mxfp8 or .f4, not .{}",
            other.ext()
        ),
    };
    ensure!(
        layers == (0..text.num_hidden_layers),
        "the .{} set spans layers {layers:?}, not every one of the {} this config decodes",
        fmt.ext(),
        text.num_hidden_layers
    );
    let dims = SetDims::new(
        layers,
        text.n_routed_experts,
        text.hidden_size,
        text.intermediate_size,
    );
    ExpertSet::open_routed(dir, fmt, dims)
}

fn plan_routed_pool<'a>(
    (label, fmt): (&'static str, RoutedFmt),
    (resident, scratch): (usize, usize),
    (text, units): (&InklingTextConfig, &'a [Unit]),
    pin: PinCfg<'a>,
) -> Result<(usize, usize, crate::routed::PoolCfg<'a>)> {
    let unit = units.first().map_or(0, |u| u.bytes.get() as usize);
    let batch = Batch::union(text.num_experts_per_tok, 1, 0, unit)?;
    let (placement, pool) = PoolPlan::new(label, units, resident + scratch, batch).decide(pin)?;
    for (i, id) in placement.pinned.iter().enumerate() {
        ensure!(
            id.0 as usize == i,
            "the pinned set is not the prefix the layer-major unit order assumes"
        );
    }
    let (pinned, total) = (placement.pinned.len(), units.len());
    tracing::info!(
        "{label} pin (.{}): {:.1} GiB resident (globals + dense), {:.1} GiB activation \
         scratch, {pinned} of {total} experts pinned ({:.1}% of the {:.0} GiB routed set)",
        fmt.ext(),
        resident as f64 / (1u64 << 30) as f64,
        scratch as f64 / (1u64 << 30) as f64,
        100.0 * pinned as f64 / total.max(1) as f64,
        (total * unit) as f64 / (1u64 << 30) as f64,
    );
    Ok((pinned, total, pool))
}
