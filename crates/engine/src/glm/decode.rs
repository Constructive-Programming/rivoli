//! The decode loop: device argmax and greedy generation. Ported from
//! `old:src/gpu.rs::{argmax_rows, generate}` — greedy only at M4 (MTP deferred past
//! parity; the `Rows` dimension the verify pass rides is designed in everywhere else,
//! so nothing is foreclosed).

use super::MAXROW;
use super::engine::{ARGMAX_BYTES, GlmEngine};
use anyhow::{Result, bail, ensure};
use rivoli_backend::launch_argmax;
// The request and result shapes live at the seam, not here: neither names anything
// GLM-shaped, and `Engine::generate` must be able to name both in a build where this
// module does not exist at all.
use crate::seam::{Decoded, Emit, GenSpec, TokenSink};

impl GlmEngine<'_> {
    /// Greedy argmax over each of the pass's `n` logit rows — reduced ON DEVICE, so
    /// only [`ARGMAX_BYTES`] come back per pass however many rows it carried. The
    /// kernel reproduces the host fold exactly (strict `>`: ties keep the lowest
    /// index, NaN never wins), returning `logits[best]` so the finiteness bail is the
    /// same `!value.is_finite()` check.
    fn argmax_rows(&mut self, n: usize) -> Result<[u32; MAXROW]> {
        debug_assert!((1..=MAXROW).contains(&n));
        let t = std::time::Instant::now();
        for r in 0..n {
            // SAFETY: logits is MAXROW·vocab device f32 (written + joined); argmax_dev
            // owns ARGMAX_BYTES, and r < n ≤ MAXROW keeps both slots in bounds.
            unsafe {
                launch_argmax(
                    (self.logits.ptr() as *const f32).add(r * self.cfg.vocab),
                    self.cfg.vocab,
                    self.argmax_dev.ptr_mut().add(r * 8) as *mut i32,
                    self.argmax_dev.ptr_mut().add(r * 8 + 4) as *mut f32,
                )?;
            }
        }
        // The one blocking call the whole tail phase hides behind: this D2H drains the
        // final rmsnorm, lm_head AND argmax — which is why its span (stamped below,
        // into Head with the tail launches from `forward.rs`) is the tail's real cost,
        // not the launches'.
        self.argmax_dev.copy_out_into(&mut self.argmax_host)?;
        self.prof.lap(crate::telemetry::Phase::Head, t);
        debug_assert_eq!(
            self.argmax_host.len(),
            ARGMAX_BYTES,
            "argmax result must be MAXROW*[idx | val] + a nonfinite tag"
        );
        let word = |o: usize| {
            let b = &self.argmax_host[o..o + 4];
            [b[0], b[1], b[2], b[3]]
        };
        let mut out = [0u32; MAXROW];
        for (r, o) in out.iter_mut().enumerate().take(n) {
            let idx = i32::from_le_bytes(word(r * 8));
            let val = f32::from_le_bytes(word(r * 8 + 4));
            if !val.is_finite() {
                // The tag rode the same D2H, so this costs nothing and turns
                // "somewhere in 78 layers x every position" into a coordinate.
                let tag = u32::from_le_bytes(word(MAXROW * 8));
                let where_ = if tag == 0 {
                    "no layer residual was non-finite — the fault is AFTER the last \
                     layer (final rmsnorm, lm_head or argmax itself), not in the \
                     MoE/attention stack"
                        .to_string()
                } else {
                    format!(
                        "first non-finite residual at pos={} layer={}",
                        (tag - 1) / 256,
                        (tag - 1) % 256
                    )
                };
                bail!("logits are non-finite (NaN/Inf in the GPU forward pass), row {r}: {where_}");
            }
            debug_assert!(idx >= 0, "argmax returned negative index {idx}");
            *o = idx as u32;
        }
        Ok(out)
    }

    /// Row 0's argmax — the next token on the sequential path.
    fn argmax(&mut self) -> Result<u32> {
        Ok(self.argmax_rows(1)?[0])
    }

    /// The prompt through the model (layer-major when possible), returning the decode
    /// start position. The prefill's OWN cost is reported here, before `generate`
    /// rebases the counters — those exist precisely to EXCLUDE the prefill from the
    /// decode stats, so without this line the phase doing 6.4x the reads would never
    /// report one of them; `reads/token` is the comparable figure across prompt
    /// lengths.
    async fn prefill(&mut self, prompt: &[u32]) -> Result<usize> {
        let wall = std::time::Instant::now();
        let mut pos = 0;
        if self.layer_major_prefill {
            self.prefill_layer_major(prompt).await?;
            pos = prompt.len();
        } else {
            for &tok in prompt {
                self.forward(tok, pos).await?;
                pos += 1;
            }
        }
        let (h, m) = (self.hits(), self.misses());
        tracing::info!(
            "PREFILL: {} tokens in {:.1} s ({}) | {m} expert reads, {:.2}/token | {h} \
             hits, {:.1}%",
            prompt.len(),
            wall.elapsed().as_secs_f64(),
            match self.layer_major_prefill {
                true => "layer-major",
                false => "token-major",
            },
            m as f64 / prompt.len() as f64,
            100.0 * h as f64 / (h + m).max(1) as f64,
        );
        Ok(pos)
    }

    /// Greedy-decode up to `spec.ngen` tokens continuing `spec.prompt`, stopping on any
    /// `spec.eos`. `on_tok` is called with each token the moment it lands, BEFORE the
    /// next forward; return false to stop early — server mode streams from it and
    /// returns false when the client hangs up, otherwise a closed connection would keep
    /// the sole-tenant GPU busy for the rest of the budget.
    ///
    /// The decode is ONE async flow: prefill (warm-up) then the token loop, driven by a
    /// single current-thread runtime — `forward` awaits the expert stream inline, so
    /// there is no per-layer block_on. The token loop is serial by data dependency
    /// (T+1 needs T's argmax); this is the shape speculative decode later slots into.
    pub fn generate(&mut self, spec: GenSpec<'_>, on_tok: TokenSink<'_>) -> Result<Decoded> {
        ensure!(!spec.prompt.is_empty(), "empty prompt");
        // The emission protocol — which tokens are output, and when the run may continue —
        // is [`Emit`]'s, shared with every other arm. What stays here is this loop's own
        // shape: one `block_on` around the whole flow, because `forward` awaits the expert
        // stream inline and a per-layer `block_on` is what that avoids.
        let mut emit = Emit::new(&spec);
        let (hit0, miss0, prof0, decode_wall) = rivoli_backend::block_on(async {
            let mut pos = self.prefill(spec.prompt).await?;
            // Stats describe steady-state DECODE, not the cold prefill — the phase
            // counters rebase here for exactly the reason the hit counters do.
            let (hit0, miss0, prof0) = (self.hits(), self.misses(), self.prof);
            let decode_wall = std::time::Instant::now();
            // `cur` is the token AT `pos`, decided but not yet fed through the model.
            let mut cur = self.argmax()?;
            while emit.offer(cur, on_tok) {
                self.forward(cur, pos).await?;
                self.flush_trace()?;
                pos += 1;
                cur = self.argmax()?;
            }
            Ok::<_, anyhow::Error>((hit0, miss0, prof0, decode_wall.elapsed()))
        })?;
        let ph = self.prof.since(&prof0);
        Ok(emit.finish(decode_wall, self.hits() - hit0, self.misses() - miss0, &ph))
    }

    /// Teacher-forced scoring: walk `ids`, score each position's logits against the true
    /// next token, then FORCE that token as the consumed input. **The loop is
    /// `generate`'s loop** — the same `prefill`/`forward`/`argmax` calls on the same
    /// scratch, so the number it produces is about the engine that decodes; the one
    /// addition is a read-only D2H of the logits row. `crate::score::score_row` checks
    /// on every position that the row read back argmaxes (on the host, with the device
    /// kernel's own tie rule) to the token the device picked — the guard against
    /// scoring a stale or mis-addressed row.
    #[cfg(feature = "teacher-forcing")]
    pub fn score(&mut self, ids: &[u32]) -> Result<crate::seam::Scored> {
        use anyhow::Context as _;
        crate::score::admit(ids, self.max_ctx())?;
        // Bespoke async loop rather than `score::walk`, deliberately: this arm's whole
        // decode runs inside ONE `block_on` (`generate`'s shape), and a per-token
        // `block_on` under the shared sync walk would be a different runtime pattern
        // than the engine being measured. The scoring protocol itself still has one
        // author — `Tally` — so the arithmetic cannot drift.
        let vocab = self.cfg.vocab;
        let mut raw = Vec::with_capacity(vocab * 4);
        let mut tally = crate::score::Tally::new(ids.len());
        rivoli_backend::block_on(async {
            // Position 0 goes through `forward`, NOT `prefill` — deliberately, and it is
            // the difference between a comparable number and a confounded one. `prefill`
            // is layer-major by default (`prefill_layer_major`), a different schedule
            // plus a host round-trip of the residual row; every OTHER scored position
            // comes from `forward`, and so does every position in the pinned reference's
            // own `nll_forced` (`old:src/gpu.rs`, which walks `forward(tok, pos)` from
            // pos 0 with no prefill at all). Prefilling position 0 here would make the
            // FIRST scored row the only one produced by a different schedule — which the
            // TF-alignment gate would then have to charge to the engine.
            self.forward(ids[0], 0).await?;
            self.flush_trace()?;
            let mut pos = 1;
            let (hit0, miss0) = (self.hits(), self.misses());
            for (i, &target) in ids.iter().enumerate().skip(1) {
                let own = self.argmax()?;
                // Row 0 of the logits buffer — the row `argmax` just reduced.
                self.logits.copy_out_prefix(&mut raw, vocab * 4)?;
                let row = rivoli_core::num::f32s_le(&raw).context("ragged logit row")?;
                tally.push(&row, own, target)?;
                if i + 1 < ids.len() {
                    self.forward(target, pos).await?;
                    self.flush_trace()?;
                    pos += 1;
                }
            }
            tally.into_scored(self.hits() - hit0, self.misses() - miss0)
        })
    }
}
