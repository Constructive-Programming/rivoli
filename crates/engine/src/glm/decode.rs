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
use crate::seam::{DecodeStats, GenSpec};

impl GlmEngine<'_> {
    /// Greedy argmax over each of the pass's `n` logit rows — reduced ON DEVICE, so
    /// only [`ARGMAX_BYTES`] come back per pass however many rows it carried. The
    /// kernel reproduces the host fold exactly (strict `>`: ties keep the lowest
    /// index, NaN never wins), returning `logits[best]` so the finiteness bail is the
    /// same `!value.is_finite()` check.
    fn argmax_rows(&mut self, n: usize) -> Result<[u32; MAXROW]> {
        debug_assert!((1..=MAXROW).contains(&n));
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
        // final rmsnorm, lm_head AND argmax.
        self.argmax_dev.copy_out_into(&mut self.argmax_host)?;
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
    pub fn generate(
        &mut self,
        spec: GenSpec<'_>,
        on_tok: &mut dyn FnMut(u32) -> bool,
    ) -> Result<(Vec<u32>, DecodeStats)> {
        ensure!(!spec.prompt.is_empty(), "empty prompt");
        let mut generated = Vec::with_capacity(spec.ngen);
        let (hit0, miss0, decode_wall) = rivoli_backend::block_on(async {
            let mut pos = self.prefill(spec.prompt).await?;
            // Stats describe steady-state DECODE, not the cold prefill.
            let (hit0, miss0) = (self.hits(), self.misses());
            let decode_wall = std::time::Instant::now();
            // `cur` is the token AT `pos`, decided but not yet fed through the model.
            let mut cur = self.argmax()?;
            loop {
                if spec.eos.contains(&cur) {
                    break;
                }
                generated.push(cur);
                if !on_tok(cur) || generated.len() >= spec.ngen {
                    break;
                }
                self.forward(cur, pos).await?;
                self.flush_trace()?;
                pos += 1;
                cur = self.argmax()?;
            }
            Ok::<_, anyhow::Error>((hit0, miss0, decode_wall.elapsed()))
        })?;
        let decode_s = decode_wall.as_secs_f64();
        let stats = DecodeStats {
            decode_s,
            tok_s: generated.len() as f64 / decode_s.max(1e-9),
            hits: self.hits() - hit0,
            misses: self.misses() - miss0,
        };
        tracing::info!(
            "DECODE: {} tokens in {:.1} s = {:.2} tok/s | {} hits / {} misses",
            generated.len(),
            stats.decode_s,
            stats.tok_s,
            stats.hits,
            stats.misses,
        );
        Ok((generated, stats))
    }
}
