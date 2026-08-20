//! The K3 decode loop: token-sequential prefill, device argmax, greedy generation.
//!
//! **The prefill and the decode are the SAME path here** — one token per `forward`, because
//! the KDA recurrence makes every token depend on the previous one and this tree has no
//! chunked KDA kernel (`crate::k3`'s declared deviation). That identity is load-bearing
//! twice: `--trace`'s token-major recovery is exact on this arm, and the anchor gate's
//! state-carry test can compare a carried decode against a replayed prefix knowing the two
//! run identical launch sequences.

use super::engine::{ARGMAX_BYTES, K3Engine};
// One nested use where V4's twin has three lines, so the two preambles cannot re-become the
// token-identical pair the jscpd gate reported at the first compile.
use crate::seam::{Decoded, Emit, GenSpec, TokenSink};
use anyhow::{Context as _, Result, ensure};
use rivoli_backend::{device_sync, launch_argmax};

impl K3Engine<'_> {
    /// Greedy argmax over the logit buffer, refusing a non-finite top logit — a NaN
    /// blow-up must become a message, not a plausible token.
    fn argmax(&mut self) -> Result<u32> {
        // Head bucket: the sync below drains the head chain (`head_tail`'s launches).
        let t = std::time::Instant::now();
        // SAFETY: `logits` holds `vocab` f32; `argmax_out` is an i32 then an f32, the
        // second pointer offset by exactly the first's width.
        unsafe {
            let out = self.argmax_out.ptr_mut();
            launch_argmax(
                self.logits.ptr().cast(),
                self.cfg.vocab,
                out.cast(),
                out.add(size_of::<i32>()).cast(),
            )?;
        }
        // Nominally redundant while everything rides the null stream, and exactly the
        // redundancy that stops being redundant when the tail moves onto one — the join
        // someone forgets is this port's recorded failure mode, so it is owned here.
        device_sync()?;
        // The sync above is the drain; the 8-byte D2H below is µs and stays outside the
        // span (it would also extend a borrow across the lap).
        self.prof.lap(crate::telemetry::Phase::Head, t);
        let host = &mut self.argmax_bytes;
        self.argmax_out.copy_out_into(host)?;
        // Asserted rather than assumed: a short read would leave a stale suffix in the
        // reused host buffer and decode a plausible token from it.
        ensure!(
            host.len() == ARGMAX_BYTES,
            "argmax read {} bytes",
            host.len()
        );
        let idx = i32::from_le_bytes([host[0], host[1], host[2], host[3]]);
        let val = f32::from_le_bytes([host[4], host[5], host[6], host[7]]);
        ensure!(
            val.is_finite(),
            "the top logit is {val} — NaN/inf left the forward pass"
        );
        u32::try_from(idx).context("device argmax handed back a negative row")
    }

    /// Prefill token by token, then greedy-decode up to `spec.ngen` tokens, stopping on any
    /// `spec.eos`. [`Emit`] owns the emission protocol, shared with every arm; `sink` sees
    /// each token before the next forward and may stop the run by returning false.
    pub fn decode(&mut self, spec: GenSpec<'_>, sink: TokenSink<'_>) -> Result<Decoded> {
        ensure!(!spec.prompt.is_empty(), "empty prompt");
        ensure!(
            spec.prompt.len() < self.max_ctx,
            "a {}-token prompt leaves no room to generate under --ctx {}",
            spec.prompt.len(),
            self.max_ctx
        );
        self.reset()?;
        let mut emit = Emit::new(&spec);
        let wall = std::time::Instant::now();
        for (p, &t) in spec.prompt.iter().enumerate() {
            self.forward(t, p)?;
        }
        let (hit0, miss0) = self.pool_counters();
        tracing::info!(
            "PREFILL: {} tokens in {:.1} s (token-sequential — the recurrence's order) | \
             {} expert reads",
            spec.prompt.len(),
            wall.elapsed().as_secs_f64(),
            miss0 - self.misses0,
        );
        // Stats describe steady-state DECODE, so the counters (phases included) re-read
        // here, after the cold prefill, not at `reset`.
        let decode_wall = std::time::Instant::now();
        let prof0 = self.prof;
        let mut pos = spec.prompt.len();
        let mut cur = self.argmax()?;
        while emit.offer(cur, sink) {
            // Flush the PREVIOUS step's trace records before this forward buffers more —
            // first iteration flushes the prefill's, which token-sequential prefill makes
            // well-formed (the legality row's `--trace` cell rests on that).
            self.pin.routed.flush_trace()?;
            self.forward(cur, pos)?;
            pos += 1;
            cur = self.argmax()?;
        }
        let (hit1, miss1) = self.pool_counters();
        Ok(emit.finish(
            decode_wall.elapsed(),
            hit1 - hit0,
            miss1 - miss0,
            &self.prof.since(&prof0),
        ))
    }

    /// Teacher-forced scoring on the K3 arm, keeping this loop's flush-BEFORE-forward
    /// ordering (the trace-well-formedness argument above). The scored row comes from
    /// [`K3Engine::logits`], the arm's one public readback, guard included.
    /// Unexercised on a device until a K3 checkpoint lands, like the rest of this arm.
    #[cfg(feature = "teacher-forcing")]
    pub fn score(&mut self, ids: &[u32]) -> Result<crate::seam::Scored> {
        crate::score::admit(ids, self.max_ctx)?;
        self.reset()?;
        self.forward(ids[0], 0)?;
        let (hit0, miss0) = self.pool_counters();
        let mut own = self.argmax()?;
        let tally = crate::score::walk(ids, |next| {
            let (row, scored_own) = (self.logits()?, own);
            if let Some((t, pos)) = next {
                self.pin.routed.flush_trace()?;
                self.forward(t, pos)?;
                own = self.argmax()?;
            }
            Ok((row, scored_own))
        })?;
        let (hit1, miss1) = self.pool_counters();
        tally.into_scored(hit1 - hit0, miss1 - miss0)
    }
}
