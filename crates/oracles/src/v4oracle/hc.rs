//! `Block.hc_pre`, `Block.hc_post` and the `kernel.py::hc_split_sinkhorn` they share — the
//! hyper-connection mixture that opens and closes every sublayer.
//!
//! **Split out of `forward.rs` on 2026-08-15, verbatim**, under the 800-line file gate
//! (`crates/cli/tests/line_limit.rs`) and the whole-tree CodeScene 10/10 gate
//! (`crates/cli/tests/codescene.rs`). The cut is by COHESION: `run_layer` calls `hc_pre` and
//! `hc_post` around each of its two sublayers and threads [`HcW`] and [`HcMix`] between
//! them; that pair is the ENTIRE seam. `Oracle::hc_blend` did NOT come along — the head tail
//! spells it a second time (model.py:687 and :715 are the same expression), so it is a
//! primitive with callers in two modules and stayed in the shared root with `linear` and the
//! norms.
//!
//! Every body moved unchanged. This is a frozen transliteration — see `forward.rs`'s module
//! doc for what is reproduced exactly, what is reproduced only up to summation order, and
//! what is out of scope; all of it still governs this file. Nothing public moved: every
//! item below is `pub(super)` or private.

use crate::v4oracle::breakages::Defect;
use crate::v4oracle::forward::{Oracle, softmax_strided};
use crate::v4oracle::numerics::sigmoid;

/// The Sinkhorn mixture one `hc_pre` produced, for the `hc_post` that closes the SAME
/// sublayer — `post` is `[s, hc]` and `comb` is `[s, hc, hc]`.
///
/// A block runs the pair twice, around attention and around the FFN, at identical types and
/// shapes: crossing the two halves was a well-typed call before this type existed.
pub(super) struct HcMix {
    /// The query rows this mixture was built for. Carried rather than passed again, because
    /// `hc_post` used to take its own `s` and a mixture used at the wrong length is exactly
    /// the well-typed mistake the struct above exists to preclude.
    s: usize,
    post: Vec<f32>,
    comb: Vec<f32>,
}

/// One sublayer's mHC parameters — the `hc_{attn,ffn}_{fn,scale,base}` triple.
///
/// Three views of ONE learned block, always read together. Passed separately they let an
/// attention `scale` reach an FFN `base`: same types, same shapes, silently wrong values,
/// which is the same hazard [`HcMix`] closes on the other side of the sublayer.
#[derive(Clone, Copy)]
pub(super) struct HcW<'a> {
    pub(super) fnw: &'a [f32],
    pub(super) scale: &'a [f32],
    pub(super) base: &'a [f32],
}

impl Oracle {
    /// `kernel.py::hc_split_sinkhorn` for one token.
    ///
    /// `mixes` is `[(2 + hc) * hc]`: `hc` pre-weights, `hc` post-weights, then `hc * hc`
    /// combination logits. Note the FIRST normalisation pair is a row *softmax* followed by
    /// a column divide, and only the remaining `iters - 1` passes are plain row/column
    /// divides — that asymmetry is easy to lose in a port.
    fn hc_split_sinkhorn(&self, mixes: &[f32], w: HcW) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (pre, post) = self.hc_gates(mixes, w);
        let mut comb = self.hc_comb_logits(mixes, w);
        self.hc_norm(&mut comb, false);
        let iters = if self.defect == Defect::SinkhornIterCountProbe {
            self.cfg.hc_sinkhorn_iters - 1
        } else {
            self.cfg.hc_sinkhorn_iters
        };
        for _ in 0..iters.saturating_sub(1) {
            self.hc_norm(&mut comb, true);
            self.hc_norm(&mut comb, false);
        }
        (pre, post, comb)
    }

    /// The two gate vectors `hc_split_sinkhorn` returns beside the combination matrix: the
    /// `pre` weights that collapse the residual copies and the `post` weights that reopen
    /// them. `post` carries the reference's factor of 2 and no eps; `pre` the reverse.
    fn hc_gates(&self, mixes: &[f32], w: HcW) -> (Vec<f32>, Vec<f32>) {
        let hc = self.cfg.hc_mult;
        let eps = self.cfg.hc_eps;
        let mut pre = vec![0.0f32; hc];
        let mut post = vec![0.0f32; hc];
        for j in 0..hc {
            pre[j] = sigmoid(mixes[j] * w.scale[0] + w.base[j]) + eps;
            post[j] = 2.0 * sigmoid(mixes[j + hc] * w.scale[1] + w.base[j + hc]);
        }
        (pre, post)
    }

    /// The combination matrix before any Sinkhorn pass: the `hc * hc` logits tail of
    /// `mixes`, row-softmaxed and shifted by eps. That leading row *softmax* is the
    /// asymmetry [`Oracle::hc_split_sinkhorn`]'s doc warns about — the remaining passes are
    /// plain divides, so a port that starts the loop one pass early loses it silently.
    fn hc_comb_logits(&self, mixes: &[f32], w: HcW) -> Vec<f32> {
        let (hc, eps) = (self.cfg.hc_mult, self.cfg.hc_eps);
        let mut comb = vec![0.0f32; hc * hc];
        // `j * hc + k` walks `0..hc*hc` in order, so the flat index IS `[j][k]`; the reads
        // and the order they happen in are the nested loops', with one level less nesting.
        for (i, c) in comb.iter_mut().enumerate() {
            *c = mixes[i + 2 * hc] * w.scale[2] + w.base[i + 2 * hc];
        }
        // comb = comb.softmax(-1) + eps. The row softmax reads and writes only its own row,
        // so the eps shift is one pass afterwards rather than interleaved per row.
        for j in 0..hc {
            softmax_strided(&mut comb, hc, 1, j * hc);
        }
        for c in comb.iter_mut() {
            *c += eps;
        }
        comb
    }

    /// One Sinkhorn pass. `comb / (comb.sum(-1) + eps)` and `comb / (comb.sum(-2) + eps)`
    /// differ only in which index they hold fixed, so one normaliser takes that as an
    /// index function. Two copies would be two places to get the eps or the axis wrong.
    fn hc_norm(&self, c: &mut [f32], by_row: bool) {
        let hc = self.cfg.hc_mult;
        let eps = self.cfg.hc_eps;
        let at = |fixed: usize, run: usize| {
            if by_row {
                fixed * hc + run
            } else {
                run * hc + fixed
            }
        };
        for fixed in 0..hc {
            let s: f32 = (0..hc).map(|r| c[at(fixed, r)]).sum();
            for r in 0..hc {
                c[at(fixed, r)] /= s + eps;
            }
        }
    }

    /// `Block.hc_pre`. `h` is `[s, hc, dim]`; the rsqrt is over the FULL `hc * dim`
    /// flattened row, not per copy.
    pub(super) fn hc_pre(&self, h: &[f32], s: usize, w: HcW) -> (Vec<f32>, HcMix) {
        let (hc, dim) = (self.cfg.hc_mult, self.cfg.dim);
        let hcd = hc * dim;
        let mut y = vec![0.0f32; s * dim];
        let mut post = vec![0.0f32; s * hc];
        let mut comb = vec![0.0f32; s * hc * hc];
        for t in 0..s {
            let flat = &h[t * hcd..(t + 1) * hcd];
            let mixes = self.hc_mixes(flat, w.fnw);
            let (pre, row_post, row_comb) = self.hc_split_sinkhorn(&mixes, w);
            self.hc_blend(&pre, flat, &mut y[t * dim..(t + 1) * dim]);
            post[t * hc..(t + 1) * hc].copy_from_slice(&row_post);
            comb[t * hc * hc..(t + 1) * hc * hc].copy_from_slice(&row_comb);
        }
        // `y.to(dtype)` — back to bf16.
        self.round_bf16(&mut y);
        (y, HcMix { s, post, comb })
    }

    /// One token's `mix_hc()` mixture logits: `hc_*_fn @ flat * rsqrt(mean(flat^2) + eps)`.
    ///
    /// The rsqrt is over the FULL `hc * dim` flattened row, not per copy — the same
    /// statistic scales every mix, which is what `Defect::HcPreNoRsqrt` drops. `hc_head`
    /// carries its own copy of both the statistic and the mistake because the reference
    /// spells it a second time there (model.py:712-713), and a port can get one right and
    /// the other wrong.
    fn hc_mixes(&self, flat: &[f32], fnw: &[f32]) -> Vec<f32> {
        let hcd = flat.len();
        let var = flat.iter().map(|v| v * v).sum::<f32>() / hcd as f32;
        let rs = if self.defect == Defect::HcPreNoRsqrt {
            1.0
        } else {
            (var + self.cfg.norm_eps).sqrt().recip()
        };
        let mut mixes = vec![0.0f32; self.cfg.mix_hc()];
        for (j, m) in mixes.iter_mut().enumerate() {
            let w = &fnw[j * hcd..(j + 1) * hcd];
            *m = flat.iter().zip(w).map(|(a, b)| a * b).sum::<f32>() * rs;
        }
        mixes
    }

    /// `Block.hc_post`: `y[k] = post[k] * x + sum_j comb[j, k] * residual[j]`.
    ///
    /// `mix.comb` is indexed `[source, dest]` — the Sinkhorn row-softmax runs over the DEST
    /// index and the column normalisation over the SOURCE index. Transposing it keeps every
    /// row of the result a convex-ish combination of the same vectors and is therefore
    /// invisible to any magnitude check.
    pub(super) fn hc_post(&self, x: &[f32], residual: &[f32], mix: &HcMix) -> Vec<f32> {
        let HcMix { s, post, comb } = mix;
        let (hc, dim) = (self.cfg.hc_mult, self.cfg.dim);
        let mut y = vec![0.0f32; s * hc * dim];
        for t in 0..*s {
            for k in 0..hc {
                let cell = HcPostCell {
                    x: &x[t * dim..(t + 1) * dim],
                    residual: &residual[t * hc * dim..(t + 1) * hc * dim],
                    col: self.hc_comb_col(&comb[t * hc * hc..(t + 1) * hc * hc], k),
                    post: post[t * hc + k],
                };
                let dst = (t * hc + k) * dim;
                self.hc_post_cell(&cell, k, &mut y[dst..dst + dim]);
            }
        }
        // `.type_as(x)` — the residual stream is bf16.
        self.round_bf16(&mut y);
        y
    }

    /// The `comb` column one destination copy reads: `comb[j, k]` for every source `j`.
    ///
    /// `SinkhornCombTransposed` swaps the two indices. Decided ONCE per destination copy
    /// rather than inside the `dim` loop, which is what keeps [`Oracle::hc_post`] off a
    /// fifth level of nesting; the values, and the order they are consumed in, are the ones
    /// the inline index expression produced.
    fn hc_comb_col(&self, comb: &[f32], k: usize) -> Vec<f32> {
        let hc = self.cfg.hc_mult;
        (0..hc)
            .map(|j| {
                if self.defect == Defect::SinkhornCombTransposed {
                    comb[k * hc + j]
                } else {
                    comb[j * hc + k]
                }
            })
            .collect()
    }

    /// One `(token, destination copy)` cell of `hc_post`, over the whole `dim` width.
    ///
    /// `HcPostNoComb` replaces the mix with a plain residual add, so it is a branch here
    /// rather than a zero column: a one-hot column would add `0.0 * residual[j]` terms that
    /// are not free in IEEE (`0.0 * inf` is NaN, and `-0.0 + 0.0` is `+0.0`). The `d`-outer
    /// / `j`-inner order, and therefore the summation order, is the one this always had.
    fn hc_post_cell(&self, cell: &HcPostCell, k: usize, out: &mut [f32]) {
        let dim = self.cfg.dim;
        let no_comb = self.defect == Defect::HcPostNoComb;
        for (d, o) in out.iter_mut().enumerate() {
            let mut acc = cell.post * cell.x[d];
            if no_comb {
                acc += cell.residual[k * dim + d];
            } else {
                for (j, &c) in cell.col.iter().enumerate() {
                    acc += c * cell.residual[j * dim + d];
                }
            }
            *o = acc;
        }
    }
}

/// Everything one `(token, destination copy)` cell of `hc_post` reads. A struct because the
/// four are only ever read together for one cell, and separating them is how a `k` reaches
/// the wrong `post` — a well-typed mistake that changes values and nothing else.
struct HcPostCell<'a> {
    /// `x[t]`, the sublayer output row, `[dim]`.
    x: &'a [f32],
    /// `residual[t]`, all `hc` copies, `[hc * dim]`.
    residual: &'a [f32],
    /// `comb[·, k]` for this destination copy, `[hc]`.
    col: Vec<f32>,
    /// `post[t, k]`.
    post: f32,
}
