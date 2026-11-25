//! Inkling 2 Small's pin: the same placement machinery as its parent, over the routed pool.
//!
//! A child module of [`super`] so the private placers (`place_globals`, `place_dense`,
//! `measure_resident`) and [`PIN_SLACK`](super::PIN_SLACK) are readable here without widening
//! one visibility; the split is the 800-line cap's, and the cohesion line is "Mini places
//! slabs, Small places a tier beside a pool".

use super::*;
use crate::resident::{Batch, PinCfg, PoolPlan, stream_units};
use crate::routed::RoutedGeom;
use rivoli_artifact::format::{RoutedFmt, SetDims, mxfp8_layer_range};

impl InklingPin {
    /// Open Inkling 2 Small: the same text stack, with the experts behind [`RoutedPool`].
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
        ensure!(
            text.loop_config.is_none(),
            "the .mxfp8 streaming backend is the NON-looped member's; this config loops"
        );
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
        let plan = PoolPlan::new("Inkling-2-Small", &units, tier_cap, batch);
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
            "Inkling-2-Small pin: {:.1} GiB resident (globals + dense), {:.1} GiB activation \
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
            Ok(ExpertBackend::MxFp8 {
                _set: set,
                routed: Box::new(RoutedPool::new(pool, geom)?),
                pinned_experts,
                total_experts,
            })
        })
    }
}
