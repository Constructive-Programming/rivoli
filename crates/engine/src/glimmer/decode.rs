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
    /// **Returns [`Decoded`], which every arm now does** — this loop was written after the
    /// type existed and GLM's `generate` joined it with M10, closing the asymmetry the seam
    /// used to carry a note about. The named struct is what `Decoded`'s own doc argues for:
    /// `.0`/`.1` read the same whichever way round they are, which is how a swapped
    /// destructuring survives review.
    pub fn decode(&mut self, spec: GenSpec<'_>, sink: TokenSink<'_>) -> Result<Decoded> {
        // `inspect_err`, not a `match`: join the device on ANY error before the caller
        // can drop buffers whose kernels may still be in flight. `GlimmerEngine` has no
        // `Drop` and its field order frees the weight slab first — see this method's doc.
        self.decode_inner(&spec, sink)
            .inspect_err(|_| drop(device_sync()))
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
        // The phase counters rebase with the slot counters: stats describe steady-state
        // decode, and the prefill's fills are warm-up.
        let prof0 = self.prof;
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
        let ph = self.prof.since(&prof0);
        Ok(emit.finish(decode_wall.elapsed(), h - h0, f - f0, &ph))
    }

    /// Teacher-forced scoring on the Glimmer arm — `prefill_and_sample` then
    /// `hidden_state`/`sample_x`, `decode_inner`'s own calls. The scored row comes from
    /// [`Self::logits`]: post-softcap, the model's real distribution, whose doc names
    /// TF scoring as the one gate that can SEE the softcap at all. It is read after the
    /// `device_sync` that `sample` already pays, so no join is added to measure.
    #[cfg(feature = "teacher-forcing")]
    pub fn score(&mut self, ids: &[u32]) -> Result<crate::seam::Scored> {
        crate::score::admit(ids, self.max_ctx())?;
        // Same error discipline as `decode`, and the same one-liner.
        self.score_inner(ids).inspect_err(|_| drop(device_sync()))
    }

    /// [`Self::score`]'s body, split out so the error join covers every `?` in it.
    #[cfg(feature = "teacher-forcing")]
    fn score_inner(&mut self, ids: &[u32]) -> Result<crate::seam::Scored> {
        // Position 0 through `prefill_and_sample`, unlike GLM's `score`, which was moved
        // onto `forward` on 2026-08-16 so its first row came off the same schedule as
        // every other. The asymmetry is deliberate and narrow: GLM is the arm the
        // TF-alignment gate pairs against a pinned reference, so a first row from a
        // different schedule would be charged to the engine; this arm has no such
        // comparison, and `prefill_and_sample` is also where its slot placement happens,
        // so bypassing it for one token would be the larger change. If a Glimmer
        // reference pairing ever lands, move this to `hidden_state`/`sample_x` first.
        let mut own = self.prefill_and_sample(&ids[..1])?;
        // Counters rebase AFTER position 0, matching `decode_inner` and the other three
        // arms — and it matters more here than anywhere: on this arm the first forward is
        // where every streamed layer pays its 967.942 MB slot fill, so reading them before
        // it would put the whole warm-up into the `.nll` header's `hit_pct` while
        // `--bench`'s own hits/misses excluded it, and the two would disagree about the
        // same run (review, 2026-08-16).
        let (h0, f0) = self.slot_stats();
        let tally = crate::score::walk(ids, |next| {
            let row = self.logits()?;
            let scored_own = own;
            if let Some((t, pos)) = next {
                self.hidden_state(t, pos)?;
                own = self.sample_x()?;
            }
            Ok((row, scored_own))
        })?;
        let (h, f) = self.slot_stats();
        tally.into_scored(h - h0, f - f0)
    }
}
