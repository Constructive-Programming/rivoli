//! The `.mxfp8` pin: the same placement machinery as its parent, with the experts behind
//! [`RoutedPool`] instead of in resident slabs.
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
use rivoli_artifact::format::{RoutedFmt, SetDims, mxfp8_layer_range};

impl InklingPin {
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
    /// artifact's own range is checked against the config here, at load.
    pub fn build_mxfp8(
        dir: &str,
        text: &InklingTextConfig,
        pin: PinCfg<'_>,
        n_ctx: usize,
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
        let (hidden, inter) = (text.hidden_size, text.intermediate_size);
        let flags = layer_flags(text);
        let (globals, dense) = measure_resident(&st, text, &flags)?;
        let layers = mxfp8_layer_range(dir, text.num_hidden_layers)?;
        ensure!(
            layers == (0..text.num_hidden_layers),
            "the .mxfp8 set spans layers {layers:?}, not every one of the {} this config decodes",
            text.num_hidden_layers
        );
        let dims = SetDims::new(layers.clone(), text.n_routed_experts, hidden, inter);
        let set = ExpertSet::open_routed(dir, RoutedFmt::MxFp8, dims)?;
        let geom = RoutedGeom::new(&set)?;
        let scratch = ScratchSizes::of(text, n_ctx).total();
        let resident = globals + dense + PIN_SLACK;
        let tier_cap = resident + scratch;
        let unit = set.expert_slot();
        let units = stream_units(layers.len() * text.n_routed_experts, unit);
        let batch = Batch::union(text.num_experts_per_tok, 1, 0, unit)?;
        let plan = PoolPlan::new(label, &units, tier_cap, batch);
        let (placement, pool) = plan.decide(pin)?;
        for (i, id) in placement.pinned.iter().enumerate() {
            ensure!(
                id.0 as usize == i,
                "the pinned set is not the prefix the layer-major unit order assumes"
            );
        }
        let pinned_experts = placement.pinned.len();
        let total_experts = units.len();
        tracing::info!(
            "{label} pin: {:.1} GiB resident (globals + dense), {:.1} GiB activation \
             scratch, {pinned_experts} of {total_experts} experts pinned ({:.1}% of the {:.0} GiB \
             routed set)",
            resident as f64 / (1u64 << 30) as f64,
            scratch as f64 / (1u64 << 30) as f64,
            100.0 * pinned_experts as f64 / total_experts.max(1) as f64,
            (total_experts * unit) as f64 / (1u64 << 30) as f64,
        );
        // The placement tail is the SHARED one; this arm adds only the pool the dense layers
        // sit beside, and ignores the tier the closure is handed — its experts are in the pool's
        // arena, not the resident slab.
        let parts = Parts {
            text,
            st,
            flags: &flags,
            tier_bytes: resident,
        };
        Self::assemble(parts, move |_, _| {
            let mut routed = RoutedPool::new(pool, geom)?;
            // A pool the placement already filled has nothing to stream: load the set now
            // and decode starts at 0 misses / 0 fetch-wait, instead of pulling every
            // expert no prefill layer touched through the demand path at NVMe latency.
            // `preload_all` restates this condition at the byte level and owns the `--trace`
            // opt-out, so the two cannot disagree about what "fully resident" means.
            if pinned_experts == total_experts {
                routed.preload_all()?;
            }
            Ok(ExpertBackend::MxFp8 {
                _set: set,
                routed: Box::new(routed),
                pinned_experts,
                total_experts,
            })
        })
    }
}
