//! Muse Glimmer's greedy decode loop. Ported from `old:src/glimmer_gpu.rs::decode`, reshaped
//! to the seam's [`GenSpec`]/[`DecodeStats`] so `main` and `serve` see one signature across
//! architectures.
//!
//! **Synchronous, where GLM's is an async flow.** GLM's `forward` awaits the expert stream
//! inline, so its loop runs under `block_on`. A Glimmer layer's fill is a host memcpy that has
//! completed by the time the pin returns — there is nothing to await — so an async wrapper
//! here would be a runtime with no suspension point in it. When the fill goes async (see
//! `pin::Slot::fill`'s upgrade path) this grows the same shape GLM already has.

use super::engine::GlimmerEngine;
use crate::seam::{Decoded, Emit, GenSpec, TokenSink};
use anyhow::{Result, ensure};
use rivoli_backend::device_sync;

impl GlimmerEngine<'_> {
    /// Greedy-decode up to `spec.ngen` tokens continuing `spec.prompt`, stopping on any
    /// `spec.eos`. `on_tok` is called with each token the moment it lands, BEFORE the next
    /// forward; return false to stop early — server mode streams from it and returns false
    /// when the client hangs up, otherwise a closed connection would keep the sole-tenant GPU
    /// busy for the rest of the budget.
    ///
    /// **Every error path joins the device before returning.** [`GlimmerEngine`] has no
    /// `Drop`, so its fields drop in declaration order and `pin` is FIRST — `DeviceTier`'s
    /// allocation is released with no synchronisation, while the activation buffers whose free
    /// would implicitly join drop AFTER it. So an error returned mid-layer would unmap the
    /// weight slab with that layer's kernels still in flight. The success path is already
    /// joined by `sample`; this covers the rest.
    ///
    /// **Returns [`Decoded`], where `GlmEngine::generate` still returns the bare tuple.** That
    /// named pair is what `Decoded`'s own doc argues for — `.0`/`.1` read the same whichever
    /// way round they are, which is how a swapped destructuring survives review — and this
    /// loop was written after the type existed. The seam carries the asymmetry and the note
    /// about closing it.
    pub fn decode(&mut self, spec: GenSpec<'_>, sink: TokenSink<'_>) -> Result<Decoded> {
        let r = self.decode_inner(&spec, sink);
        if r.is_err() {
            let _ = device_sync();
        }
        r
    }

    /// [`Self::decode`]'s body. Split out so the join above covers every `?` in it rather
    /// than each one carrying its own. `spec` is borrowed rather than moved — the inner half
    /// only reads it.
    fn decode_inner(
        &mut self,
        spec: &GenSpec<'_>,
        sink: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Decoded> {
        ensure!(!spec.prompt.is_empty(), "empty prompt");
        // `checked_add`, because the sum is the ONLY thing standing between a caller's numbers
        // and a device write past the end of a KV cache: a wrapped sum satisfies this bound
        // and then `slot = pos` on a full-attention layer indexes wherever it likes. Release
        // builds compile the overflow check out, so the guard has to be the explicit one.
        let need = spec
            .prompt
            .len()
            .checked_add(spec.ngen)
            .filter(|n| *n <= self.max_ctx());
        ensure!(
            need.is_some(),
            "{} prompt tokens plus {} new is past the {} positions this engine's KV cache was \
             built for",
            spec.prompt.len(),
            spec.ngen,
            self.max_ctx()
        );

        // The prefill's OWN cost, reported before the decode counters are rebased — those
        // exist precisely to EXCLUDE the prefill, so without this line the phase that pays
        // every streamed layer's first fill would never report one of them.
        let wall = std::time::Instant::now();
        let mut cur = self.prefill_and_sample(spec.prompt)?;
        let (pinned, streamed) = self.residency();
        let (h0, f0) = self.slot_stats();
        tracing::info!(
            "PREFILL: {} tokens in {:.1} s (layer-major) | {pinned} layers pinned, {streamed} \
             streamed | {f0} slot fills, {h0} hits",
            spec.prompt.len(),
            wall.elapsed().as_secs_f64(),
        );

        let decode_wall = std::time::Instant::now();
        // Whether the stop token is emitted, whether the sink may end the run, and when the
        // budget binds are all [`Emit`]'s — see its doc for the `old:` defect that made
        // sharing them mandatory rather than tidy. `cur` is the token AT `pos`, decided but
        // not yet fed through the model.
        let mut emit = Emit::new(spec);
        let mut pos = spec.prompt.len();
        while emit.offer(cur, sink) {
            self.hidden_state(cur, pos)?;
            pos += 1;
            cur = self.sample_x()?;
        }
        // Layer-slot hits and fills, rebased past the prefill. They ride the seam's
        // `hits`/`misses` because they answer that field's question at this arm's own
        // granularity — see `DecodeStats`, which is where the counts-are-not-comparable
        // warning belongs rather than in this one log line.
        let (h, f) = self.slot_stats();
        let (ids, stats) = emit.finish(decode_wall.elapsed(), h - h0, f - f0);
        Ok(Decoded { ids, stats })
    }
}
