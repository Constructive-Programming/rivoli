//! The GLM forward pass: the pass/span vocabulary, the layer loop, and the tail.
//! Ported from `old:src/gpu.rs`'s layer loop under the M4 narrowings — dense attention
//! only, one routed format, no MTP. The sublayers live in their own modules
//! (`attn.rs`, `mlp.rs`); this file owns the ORDER: embed → layers → tail, and the
//! two joins every layer pays.

use super::MAXROW;
use super::engine::GlmEngine;
use crate::glm::pin::LayerMlp;
use anyhow::{Result, ensure};
use rivoli_backend::{
    device_sync, launch_embed_i8_row, launch_flag_nonfinite, launch_gemv_i8, launch_rmsnorm_single,
};

/// One pass's slice of the model. `x_off` applies to `x` ONLY — every other scratch
/// buffer is `MAXROW` rows starting at 0. `tail` runs the final norm → lm_head on the
/// LAST `tail` rows; 0 skips it, which is what makes layer-major prefill affordable
/// (logits are 620 KB per row at this vocab and only the prompt's final row is read).
pub(super) struct Span {
    pub layers: std::ops::Range<usize>,
    pub x_off: usize,
    pub tail: usize,
}

impl Span {
    /// The decode shape: the whole model at row 0, logits for every row of the pass.
    pub(super) fn whole(n_layers: usize, tail: usize) -> Self {
        Self {
            layers: 0..n_layers,
            x_off: 0,
            tail,
        }
    }
}

/// The passes a layer-major prefill runs, in order: `(layer, first row, rows, tail)`.
///
/// Split out so the ordering contract is testable without a GPU, because every part of
/// it is load-bearing: layers must ascend (layer L reads what L−1 wrote), rows must
/// ascend within a layer (row `r` attends over the KV rows the passes before it
/// appended), every `(layer, row)` appears exactly once, and the tail fires on the LAST
/// row of the LAST layer and nowhere else.
pub(super) fn layer_major_schedule(
    n: usize,
    n_layers: usize,
    width: usize,
) -> impl Iterator<Item = (usize, usize, usize, usize)> {
    (0..n_layers).flat_map(move |l| {
        (0..n).step_by(width).map(move |lo| {
            let rows = (lo + width).min(n) - lo;
            let last = l + 1 == n_layers && lo + rows == n;
            (l, lo, rows, usize::from(last))
        })
    })
}

/// One pass's row geometry, threaded through every phase.
#[derive(Clone, Copy)]
pub(super) struct Pass {
    pub pos: usize,
    pub nrow: usize,
}

/// This pass's residual-row pointer — `x` offset to the pass's first row. Copy, so the
/// phases can hold it across `&mut self` calls without borrowing `self.x`.
#[derive(Clone, Copy)]
pub(super) struct XRow(pub *mut f32);

/// A row-wise rmsnorm's operands: `dst[r] = rmsnorm(src[r], w)`.
#[derive(Clone, Copy)]
pub(super) struct RowNorm {
    pub src: *const f32,
    pub w: *const f32,
    pub dst: *mut f32,
}

impl GlmEngine<'_> {
    /// One decode step: `token` at `pos`, whole model, logits for the row.
    pub(super) async fn forward(&mut self, token: u32, pos: usize) -> Result<()> {
        let span = Span::whole(self.cfg.n_layers, 1);
        self.forward_inner(&[token], pos, span).await
    }

    /// Prefill LAYER-MAJOR: every prompt token through layer L before any token reaches
    /// layer L+1. Same arithmetic in a different order — token-major re-fetches layer
    /// L's experts per token (154.75 reads/token measured over a 769-token prompt);
    /// layer-major reads each `(layer, expert)` once (24.02/token, the compulsory
    /// floor). Legality rests on one property: layer L for all tokens needs only layer
    /// L−1 for all tokens — row `r` attends over `pos + r + 1` KV rows, and every row
    /// below it appended its KV in an earlier (or the same) pass. The differing row
    /// count IS the causal mask.
    pub(super) async fn prefill_layer_major(&mut self, ids: &[u32]) -> Result<()> {
        let n_layers = self.cfg.n_layers;
        for (l, lo, rows, tail) in layer_major_schedule(ids.len(), n_layers, MAXROW) {
            let span = Span {
                layers: l..l + 1,
                x_off: lo,
                tail,
            };
            self.forward_inner(&ids[lo..lo + rows], lo, span).await?;
        }
        // Normalise the residual stream back to the shape every other caller assumes:
        // row 0 holds the LIVE hidden state (a layer-major prefill left it in row n-1).
        // ponytail: 24 KB round-tripped through the host, once per prefill, rather than
        // a device-to-device copy primitive for this one call. The end-of-layer
        // `device_sync` inside the last pass already retired every writer.
        let row = self.cfg.hidden * 4;
        let mut last = Vec::with_capacity(row);
        // SAFETY: `x` holds `x_rows * hidden` f32 and every pass above bounded
        // `ids.len()` by `x_rows`; the sync closing the last pass retired its writer.
        unsafe {
            let src = self.x.ptr().add((ids.len() - 1) * row);
            crate::device::DeviceBuf::copy_out_raw(src, row, &mut last)?;
        }
        self.x.copy_in_at(0, &last)?;
        Ok(())
    }

    /// The pass over `span`. `tokens[r]` sits at position `pos + r`; every device
    /// buffer is row-minor so batched kernels take a row count and the rest launch `R`
    /// times at an offset — `tokens.len() == 1` reproduces the unbatched pass exactly.
    pub(super) async fn forward_inner(
        &mut self,
        tokens: &[u32],
        pos: usize,
        span: Span,
    ) -> Result<()> {
        let nrow = tokens.len();
        let pass = Pass { pos, nrow };
        self.check_bounds(pass, &span)?;
        // SAFETY: bounded by `check_bounds`' x_off ensure.
        let xp = XRow(unsafe { (self.x.ptr_mut() as *mut f32).add(span.x_off * self.cfg.hidden) });
        // Embedding row → x, ONLY when this pass starts at layer 0: under layer-major
        // prefill a row visits `forward_inner` once per layer, and re-embedding would
        // overwrite the residual stream with the token it started from.
        if span.layers.start == 0 {
            self.embed_rows(tokens, xp)?;
        }
        for l in span.layers.clone() {
            self.run_layer(l, pass, xp).await?;
        }
        if span.tail > 0 {
            self.tail(pass, span.tail, xp)?;
        }
        Ok(())
    }

    /// The pass's refusals, each before the fault it prevents: the row bound before the
    /// scratch is indexed, the max_ctx bound before a KV row is written out of the
    /// slabs, the `x` bound BEFORE the offset pointer is formed (`.add()` past the end
    /// of an allocation is UB even when nothing dereferences it).
    fn check_bounds(&self, pass: Pass, span: &Span) -> Result<()> {
        ensure!(
            (1..=MAXROW).contains(&pass.nrow),
            "forward: {} token rows, but the engine's scratch is allocated for {MAXROW}",
            pass.nrow
        );
        ensure!(
            pass.pos + pass.nrow <= self.max_ctx,
            "pos {} + {} rows exceeds engine capacity max_ctx={}",
            pass.pos,
            pass.nrow,
            self.max_ctx
        );
        ensure!(
            span.x_off + pass.nrow <= self.x_rows,
            "forward: residual rows {}..{} but `x` holds {}",
            span.x_off,
            span.x_off + pass.nrow,
            self.x_rows
        );
        ensure!(
            span.tail <= pass.nrow,
            "forward: tail over {} rows of a {}-row pass",
            span.tail,
            pass.nrow
        );
        Ok(())
    }

    /// One embedding row per token into the pass's residual rows.
    fn embed_rows(&mut self, tokens: &[u32], xp: XRow) -> Result<()> {
        let (emb, hidden) = (self.pin.embed, self.cfg.hidden);
        for (r, &t) in tokens.iter().enumerate() {
            // SAFETY: embed resident; xp is this pass's rows, r < nrow.
            unsafe {
                launch_embed_i8_row(
                    emb.packed,
                    emb.scale,
                    t as usize,
                    hidden,
                    xp.0.add(r * hidden),
                )?;
            }
        }
        Ok(())
    }

    /// One layer: attention, the MLP sublayer, the end-of-layer join, and the
    /// non-finite probe. The join protects the reused descs/wexpert/moe_out buffers
    /// before the next layer overwrites them, and surfaces faults.
    async fn run_layer(&mut self, l: usize, pass: Pass, xp: XRow) -> Result<()> {
        self.attention(l, pass, xp)?;
        match &self.pin.layers[l].mlp {
            LayerMlp::Dense(m) => {
                let m = *m; // Copy — ends the &pin borrow
                self.dense_sublayer(m, pass, xp)?;
            }
            LayerMlp::Moe { .. } => self.moe_sublayer(l, pass, xp).await?,
        }
        device_sync()?;
        // Localise a non-finite residual to the earliest (pos, layer) that produced
        // one. `atomicCAS(flag, 0, tag)` keeps the FIRST; tag 0 is reserved for
        // "clean" so the tag is offset by 1. One tiny kernel per layer, no sync.
        // SAFETY: xp nrow·hidden live f32; the flag is 4 bytes inside argmax_dev.
        unsafe {
            launch_flag_nonfinite(
                xp.0,
                pass.nrow * self.cfg.hidden,
                1 + (pass.pos as u32) * 256 + l as u32,
                self.argmax_dev.ptr_mut().add(MAXROW * 8) as *mut u32,
            )?;
        }
        Ok(())
    }

    /// Row-wise rmsnorm: `dst[r] = rmsnorm(src[r], w)` for `r < nrow`. One launch per
    /// row rather than one batched launch, deliberately: rmsnorm is a microsecond
    /// kernel over ≤6144 floats, so the ~2 extra enqueues on a ~5 ms layer are not
    /// worth a second stride argument in the kernel.
    ///
    /// # Safety
    /// `src`/`dst` valid for `nrow * hidden` device f32, `w` a resident norm weight of
    /// `hidden` f32, all live until the next `device_sync`.
    pub(super) unsafe fn norm_rows(&self, n: RowNorm, nrow: usize) -> Result<()> {
        let (d, eps) = (self.cfg.hidden, self.cfg.rms_norm_eps as f32);
        for r in 0..nrow {
            // SAFETY: forwarded from this function's contract; r < nrow keeps both
            // `.add(r * d)` inside the caller's allocations.
            unsafe { launch_rmsnorm_single(n.src.add(r * d), n.w, d, eps, n.dst.add(r * d))? };
        }
        Ok(())
    }

    /// The tail: final norm → lm_head over the LAST `tail` rows, logits landing in rows
    /// `0..tail` — a prefill's closing pass takes 1 and puts the prompt's final row in
    /// row 0, which is where `argmax` already looks. lm_head is the single largest read
    /// in the pass, so the GEMV is batched.
    fn tail(&mut self, pass: Pass, tail: usize, xp: XRow) -> Result<()> {
        let (hidden, head) = (self.cfg.hidden, self.pin.lm_head);
        let xnp = self.xn.ptr_mut() as *mut f32;
        // SAFETY: final_norm/lm_head resident; xn/logits device scratch; `tail <= nrow`
        // (checked at entry) keeps the offset inside the rows this pass owns.
        unsafe {
            let last = xp.0.add((pass.nrow - tail) * hidden);
            self.norm_rows(
                RowNorm {
                    src: last,
                    w: self.pin.final_norm,
                    dst: xnp,
                },
                tail,
            )?;
            launch_gemv_i8(
                xnp,
                head.packed,
                head.scale,
                head.o_dim,
                head.i_dim,
                tail,
                self.logits.ptr_mut() as *mut f32,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod schedule_tests {
    #![allow(clippy::unwrap_used)] // tests: panic-on-failure is the idiom

    use super::layer_major_schedule;

    /// The whole ordering contract in one walk: layers ascend, rows ascend within a
    /// layer, every (layer, row) appears once, and the tail fires exactly on the last
    /// row of the last layer.
    #[test]
    fn the_layer_major_schedule_covers_every_row_of_every_layer_once_in_order() {
        let (n, n_layers, width) = (7, 3, 2);
        let mut seen = std::collections::HashSet::new();
        let mut tails = 0;
        let mut last: Option<(usize, usize)> = None;
        for (l, lo, rows, tail) in layer_major_schedule(n, n_layers, width) {
            assert!(rows >= 1 && rows <= width);
            if let Some((pl, plo)) = last {
                assert!(l > pl || (l == pl && lo > plo), "passes must ascend");
            }
            last = Some((l, lo));
            for r in lo..lo + rows {
                assert!(seen.insert((l, r)), "(layer {l}, row {r}) visited twice");
            }
            if tail > 0 {
                tails += 1;
                assert_eq!(
                    (l, lo + rows),
                    (n_layers - 1, n),
                    "tail must be the last pass"
                );
            }
        }
        assert_eq!(seen.len(), n * n_layers, "every (layer, row) exactly once");
        assert_eq!(tails, 1, "exactly one tail");
    }

    /// A prompt shorter than a pass is one pass per layer.
    #[test]
    fn a_prompt_shorter_than_a_pass_is_one_pass_per_layer() {
        let passes: Vec<_> = layer_major_schedule(1, 3, 2).collect();
        assert_eq!(passes, vec![(0, 0, 1, 0), (1, 0, 1, 0), (2, 0, 1, 1)]);
    }
}
