//! The V4 forward pass: the hyper-connection sandwich around each sublayer, the layer loop,
//! and the head tail.
//!
//! Ported from `old:src/f4gpu.rs`'s `pre_norm` / `layer` / `begin_step` / `forward` /
//! `head_tail`. The sublayers live in their own modules (`attn.rs`, `moe.rs`); this file owns
//! the ORDER — embed → layers → tail — and the residual discipline the hyper-connections
//! impose on it.
//!
//! # The residual is `[m, hc_mult, dim]`, and the SECOND `residual = h` is the trap
//!
//! `Block.forward` is `residual = h; hc_pre; attn_norm; attention; hc_post;` then
//! `residual = h; hc_pre; ffn_norm; moe; hc_post`. Note the second assignment: **the FFN's
//! residual is the POST-ATTENTION `h`, not the block's input.** Taking the block input for
//! both is a silent-wrong no shape check sees, which is why the loop below re-reads the
//! current buffer after the flip rather than hoisting a pointer.
//!
//! # There is ONE pass over the prompt, and it is not a schedule
//!
//! GLM prefills layer-major because its attention is per row and its experts are re-fetched
//! per token; V4 prefills in a single whole-prompt call because both the ring seeding and the
//! compressor's block pooling are whole-prompt BY CONSTRUCTION — attention is the only
//! operation here with a cross-token dependency, and it takes every row at once. That is why
//! every `[m, ..]` buffer is sized for the prompt rather than for a row budget, and why this
//! file has no schedule to test.

use super::engine::V4Engine;
use super::select::Extent;
use anyhow::{Result, ensure};
use rivoli_backend::{
    NULL_STREAM, device_sync, launch_embed_bf16_row_bcast, launch_gemm_bf16,
    launch_hc_head_collapse, launch_hc_post, launch_hc_pre, launch_rmsnorm_batch,
};

impl V4Engine<'_> {
    /// `hc_pre` then the sublayer's `RMSNorm`, into the working buffer and the two mixing
    /// tables `hc_post` will consume.
    ///
    /// Idempotent: it reads the residual and writes only scratch, touching no KV ring and no
    /// pooling state.
    fn pre_norm(&mut self, layer: usize, ffn: bool, m: usize) -> Result<()> {
        let (dim, hc) = (self.cfg.hidden, self.cfg.hc_mult);
        let lp = self.pin.layer(layer)?;
        let (hcw, norm) = match ffn {
            true => (lp.hc_ffn, lp.ffn_norm),
            false => (lp.hc_attn, lp.attn_norm),
        };
        // SAFETY: `h[cur]` is `m * hc * dim`; the working buffer is `m * dim`, `post` is
        // `m * hc` and `comb` is `m * hc * hc`, and none of the three aliases `h` — which is
        // exactly what `launch_hc_pre` requires, since `h` is `__restrict__` in the kernel.
        unsafe {
            launch_hc_pre(
                self.h[self.cur].ptr().cast(),
                hcw.func,
                // `scale` THEN `base`. `launch_hc_head_collapse` takes them the other way
                // round and both are `*const f32`, so a swap compiles, runs, and is finite.
                hcw.scale,
                hcw.base,
                m,
                hc,
                dim,
                self.cfg.hc_sinkhorn_iters,
                self.dims.norm_eps,
                self.cfg.hc_eps as f32,
                self.xw.ptr_mut().cast(),
                self.post.ptr_mut().cast(),
                self.comb.ptr_mut().cast(),
                NULL_STREAM,
            )?;
            // `launch_rmsnorm_batch`, in place on `hc_pre`'s output — NOT the single-row
            // `launch_rmsnorm_single` GLM uses, which is out-of-place, does not bf16-round, and
            // takes ONE mean over its whole `n`: handing it `m * dim` would compute a joint
            // statistic over every token and read the norm weight past its allocation. V4's
            // `RMSNorm` returns bf16 and this kernel rounds, so the requirement is satisfied by
            // SELECTION rather than by editing a shared kernel.
            launch_rmsnorm_batch(
                self.xw.ptr_mut().cast(),
                norm,
                m,
                dim,
                self.dims.norm_eps,
                NULL_STREAM,
            )?;
        }
        Ok(())
    }

    /// One `Block.forward` over `m` rows at `start_pos`.
    fn layer(&mut self, layer: usize, at: Extent) -> Result<()> {
        let (dim, hc, m) = (self.cfg.hidden, self.cfg.hc_mult, at.query_rows());
        for ffn in [false, true] {
            // Phase stamp per sublayer half — COARSE on this arm, deliberately: the
            // gate D2H (this arm's one host join besides the layer sync) sits inside
            // the ffn half and its span lands there, so `attend` is launch time only.
            // `telemetry::ProfileSummary` carries the table; sharpening it the way GLM
            // does is deferred until a V4 phase number is actually needed — V4 decodes
            // at the old tree's speed (9.17 vs 9.10–9.17), so nothing is being
            // attributed here yet.
            let t = std::time::Instant::now();
            self.pre_norm(layer, ffn, m)?;
            match ffn {
                true => self.moe_sublayer(layer, m)?,
                false => self.attention_block(layer, at)?,
            }
            let dst = 1 - self.cur;
            // SAFETY: the sublayer buffer is `m * dim`, `post`/`comb` are as above, and
            // `h[cur]`/`h[dst]` are DISTINCT allocations — that is `launch_hc_post`'s "`y` must
            // not alias `residual`", which holds twice over: both are `__restrict__`, and
            // thread `i` writes `y[i]` while other threads still read every source copy.
            unsafe {
                launch_hc_post(
                    self.sub.ptr().cast(),
                    self.h[self.cur].ptr().cast(),
                    self.post.ptr().cast(),
                    self.comb.ptr().cast(),
                    m,
                    hc,
                    dim,
                    self.h[dst].ptr_mut().cast(),
                    NULL_STREAM,
                )?;
            }
            self.cur = dst;
            self.prof.lap(
                match ffn {
                    true => crate::telemetry::Phase::Ffn,
                    false => crate::telemetry::Phase::Attend,
                },
                t,
            );
        }
        // ONE join per layer, so layer `L+1`'s first atomic into the accumulator cannot race
        // `L`'s drain. GLM pays the same one for the same reason.
        let t = std::time::Instant::now();
        device_sync()?;
        self.prof.lap(crate::telemetry::Phase::Ffn, t);
        Ok(())
    }

    /// Bind this step's token ids and bound its position; returns the row count.
    ///
    /// **`ids` and not just `m`**, because a hash layer's gate indexes `tid2eid` by TOKEN ID: a
    /// caller that passed row counts alone would route every hash layer by the previous step's
    /// tokens, and the difference looks like ordinary routing variation.
    ///
    /// One function because the reference had two entry points into its layer loop and they
    /// fell out of step: one had the position bound and the other did not, and the rotary
    /// tables are built at exactly `max_ctx` positions while `launch_rope_adjacent` reads
    /// `tbl + pos0 * rd` with no bound of its own — so a driver walking further than the engine
    /// was sized for reads past the table, into plausible garbage frequencies.
    fn begin_step(&mut self, ids: &[u32], start_pos: usize) -> Result<usize> {
        let m = ids.len();
        ensure!(
            m > 0 && m <= self.max_m,
            "{m} rows into buffers sized for {}",
            self.max_m
        );
        ensure!(
            start_pos + m <= self.max_ctx,
            "position {} exceeds the {} this engine was sized for",
            start_pos + m,
            self.max_ctx
        );
        self.step_ids.clear();
        self.step_ids.extend_from_slice(ids);
        Ok(m)
    }

    /// One forward pass over `ids` starting at `start_pos`, leaving the LAST row's logits in
    /// the logit buffer.
    pub(super) fn forward(&mut self, ids: &[u32], start_pos: usize) -> Result<()> {
        let m = self.begin_step(ids, start_pos)?;
        let (dim, hc) = (self.cfg.hidden, self.cfg.hc_mult);
        // `h[0]` is where the embedding lands, so every pass starts from the same buffer
        // regardless of how many times the previous one flipped. `layer` flips TWICE, so the
        // count is always even and this returns to 0 on its own — belt, not the mechanism.
        self.cur = 0;
        for (t, &tok) in ids.iter().enumerate() {
            ensure!(
                (tok as usize) < self.cfg.vocab,
                "token id {tok} is outside the {} the artifact holds",
                self.cfg.vocab
            );
            // SAFETY: `embed.packed` is `[vocab, hidden]` bf16 and `tok < vocab` was just
            // checked; the destination is row `t` of an `m * hc * dim` allocation with
            // `t < m <= max_m`.
            unsafe {
                launch_embed_bf16_row_bcast(
                    self.pin.embed.packed,
                    tok as usize,
                    dim,
                    hc,
                    self.h[0].ptr_mut().cast::<f32>().add(t * hc * dim),
                    NULL_STREAM,
                )?;
            }
        }
        let at = Extent {
            seqlen: m,
            start_pos,
        };
        for l in self.range.clone() {
            self.layer(l, at)?;
        }
        self.head_tail(m)
    }

    /// `hc_head`, the final `RMSNorm`, and the head projection — the last three operations of
    /// `Transformer.forward`, over the LAST of `m` rows.
    ///
    /// The head slices `x[:, -1]` AFTER the norm, but RMSNorm is per row, so norming one row
    /// and norming all `m` then keeping the last are the same arithmetic. A norm whose
    /// statistic was taken over ALL tokens is a real defect and it is a defect precisely
    /// because the statistic is per row.
    fn head_tail(&mut self, m: usize) -> Result<()> {
        // Launch time only; the head's execution drains inside `argmax`'s
        // `device_sync`, which `decode.rs` stamps into the same Head bucket.
        let t = std::time::Instant::now();
        let (dim, hc) = (self.cfg.hidden, self.cfg.hc_mult);
        let last = m - 1;
        let h_last = self.h[self.cur].ptr();
        // SAFETY: `h[cur]` row `last` is `hc * dim` f32 within an `m * hc * dim` allocation;
        // `head_pre` is `hc` writable, `head_x` is `dim`, `logits` is `vocab`. None aliases
        // another, which every parameter of these three launchers requires.
        unsafe {
            launch_hc_head_collapse(
                h_last.cast::<f32>().add(last * hc * dim),
                self.pin.hc_head.func,
                // `base` THEN `scale` — the OPPOSITE order from `launch_hc_pre` above. Both are
                // `*const f32`, and the head's scale is `[1]` where a block's is `[3]`, so a
                // swap reads one of three floats as the scalar and three floats of `base` as
                // the `[3]`: finite, plausible, wrong.
                self.pin.hc_head.base,
                self.pin.hc_head.scale,
                self.head_pre.ptr_mut().cast(),
                self.head_x.ptr_mut().cast(),
                1,
                hc,
                dim,
                self.dims.norm_eps,
                self.cfg.hc_eps as f32,
                NULL_STREAM,
            )?;
            launch_rmsnorm_batch(
                self.head_x.ptr_mut().cast(),
                self.pin.final_norm,
                1,
                dim,
                self.dims.norm_eps,
                NULL_STREAM,
            )?;
            // `head.weight` is bf16 in the artifact and there is no int8 head to reach for.
            // `launch_gemm_bf16` computes exactly this at `m = 1`: a runtime `(m, n, k)` over a
            // bf16 `[n, k]` weight IS a head GEMV at `m = 1, n = vocab, k = hidden`.
            //
            // The objection to reusing it is SHAPE, not capability: one wave per output element
            // over a one-row activation is a `vocab`-wide wave launch. That is a performance
            // argument with no measurement attached in this tree, so the honest instruction is
            // the reference's — call it first, then price it.
            launch_gemm_bf16(
                self.head_x.ptr().cast(),
                self.pin.head.packed,
                self.logits.ptr_mut().cast(),
                1,
                self.cfg.vocab,
                dim,
                NULL_STREAM,
            )?;
        }
        self.prof.lap(crate::telemetry::Phase::Head, t);
        Ok(())
    }
}
