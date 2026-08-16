//! The V4 decode loop: device argmax and greedy generation.
//!
//! Ported from `old:src/f4gpu.rs`'s `argmax` / `generate`, minus the four instrument blocks
//! that made that function 180 lines: the per-phase PROFILE line, the route split, the attn
//! split and the moe split. Each printed a decomposition of a wall this tree has not measured
//! once, and a bucket that reads zero because nothing filled it is this repo's named telemetry
//! trap. What remains is the loop, and the one stat line every arm prints.
//!
//! **Speculative decode is not available on this path and cannot be.** `moe.hip` instantiates
//! the FP4 expert range at `R = 1` only, so there is no `--mtp` argument here to switch off;
//! `rivoli_core::legality` says so once, at startup, and says it as a missing KERNEL rather
//! than a missing head — because that is which of the two would have to arrive first.

use super::engine::{ARGMAX_BYTES, V4Engine};
use crate::seam::{Decoded, Emit, GenSpec, TokenSink};
use anyhow::{Context, Result, ensure};
use rivoli_backend::{device_sync, launch_argmax};

impl V4Engine<'_> {
    /// Greedy argmax over the logit buffer, with the finiteness check that turns a NaN blow-up
    /// into a message instead of a plausible token.
    ///
    /// The `device_sync` before the D2H is nominally redundant today — the logits were
    /// produced on the null stream and the argmax runs there too — and it is exactly the
    /// redundancy that stops being redundant the moment the head tail moves onto a stream.
    /// This port's record is that the join someone forgets is the failure mode, so it is
    /// written as owned rather than inferred from where the launchers happen to run.
    fn argmax(&mut self) -> Result<u32> {
        // SAFETY: `logits` holds `vocab` f32; `argmax_dev` holds an i32 then an f32, and the
        // second pointer is offset by exactly the first's width.
        unsafe {
            launch_argmax(
                self.logits.ptr().cast(),
                self.cfg.vocab,
                self.argmax_dev.ptr_mut().cast(),
                self.argmax_dev.ptr_mut().add(size_of::<i32>()).cast(),
            )?;
        }
        device_sync()?;
        self.argmax_dev.copy_out_into(&mut self.argmax_host)?;
        // `argmax_dev` was allocated at exactly `ARGMAX_BYTES` and `copy_out_into` yields that
        // or errors, so the two slices below are total — but the length is asserted rather
        // than assumed, because a short read would index a stale suffix of the reused host
        // buffer and produce a plausible token.
        ensure!(
            self.argmax_host.len() == ARGMAX_BYTES,
            "argmax read {} bytes, expected {ARGMAX_BYTES}",
            self.argmax_host.len()
        );
        let word = |o: usize| [0, 1, 2, 3].map(|i| self.argmax_host[o + i]);
        let idx = i32::from_le_bytes(word(0));
        let val = f32::from_le_bytes(word(4));
        // A non-finite top logit is a defect and not a sampling outcome. It is also the VISIBLE
        // half of the read-before-write hazard the ticket protocol exists to prevent — an
        // unloaded `.f4` slot decodes to large-but-FINITE garbage (`0x7FC0_7FC0` is e8m0 `0x7f`
        // and `0xc0`, scales of 2^0 and 2^65), so this catches the loud case only and the
        // silent one is a scoring run's job.
        ensure!(
            val.is_finite(),
            "argmax read a non-finite logit ({val}): the forward pass produced NaN or inf"
        );
        u32::try_from(idx).context("argmax returned a negative index")
    }

    /// Prefill, then greedy-decode up to `spec.ngen` tokens, stopping on any `spec.eos`.
    ///
    /// `sink` is called with each token the moment it lands, BEFORE the next forward; return
    /// false from it to stop early — server mode streams from it and returns false when the
    /// client hangs up, otherwise a closed connection would keep the sole-tenant GPU busy for
    /// the rest of the budget. [`Emit`] owns that protocol, shared with every other arm.
    ///
    /// **The prefill is ONE call over the whole prompt**, and not a choice: the attention
    /// block's prefill arm seeds the ring from the prompt's last `window` positions and the
    /// compressor's prefill arm pools every complete block in one go. Both are whole-prompt by
    /// construction.
    ///
    /// **Prefill and decode index different SPACES**, and nothing in the types says so:
    /// `start_pos == 0` means the selection's window columns are absolute positions over the
    /// prompt's own KV, and `start_pos > 0` means they are ring slots. What this loop
    /// contributes is that `pos` is never 0 after the prefill consumed position 0 — so the
    /// decode arm is unreachable with a prefill's index space and vice versa.
    pub fn decode(&mut self, spec: GenSpec<'_>, sink: TokenSink<'_>) -> Result<Decoded> {
        ensure!(!spec.prompt.is_empty(), "empty prompt");
        self.reset()?;
        let mut emit = Emit::new(&spec);
        let wall = std::time::Instant::now();
        self.forward(spec.prompt, 0)?;
        let (hit0, miss0) = (self.hits(), self.misses());
        tracing::info!(
            "PREFILL: {} tokens in {:.1} s (one whole-prompt pass) | {} expert reads, {:.2}/token",
            spec.prompt.len(),
            wall.elapsed().as_secs_f64(),
            miss0 - self.misses0,
            (miss0 - self.misses0) as f64 / spec.prompt.len() as f64,
        );
        // Stats describe steady-state DECODE, not the cold prefill — which is why the counters
        // are re-read here rather than at `reset`.
        let decode_wall = std::time::Instant::now();
        // `cur` is the token AT `pos`, decided but not yet fed through the model.
        let mut pos = spec.prompt.len();
        let mut cur = self.argmax()?;
        while emit.offer(cur, sink) {
            self.forward(&[cur], pos)?;
            self.pin.routed.flush_trace()?;
            pos += 1;
            cur = self.argmax()?;
        }
        let (ids, stats) = emit.finish(
            decode_wall.elapsed(),
            self.hits() - hit0,
            self.misses() - miss0,
        );
        Ok(Decoded { ids, stats })
    }
}
