//! The head tail: `Transformer.forward`'s embedding, `Block.hc_head`, the final `RMSNorm`
//! and `ParallelHead` — everything outside the 43 blocks.
//!
//! **Split out of `forward.rs` on 2026-08-15, verbatim**, under the 800-line file gate
//! (`crates/cli/tests/line_limit.rs`) and the whole-tree CodeScene 10/10 gate
//! (`crates/cli/tests/codescene.rs`). The cut is by COHESION, and it is the same cohesion
//! [`HeadTailW`]'s own doc already argued: `bin/v4-oracle` never composes
//! [`Oracle::head_tail`] with `run_layer`, so the head tail is reached from a declared
//! synthetic probe and from nowhere in the block chain. Separating the files makes that
//! restriction structural rather than a convention.
//!
//! Every body moved unchanged. This is a frozen transliteration — see `forward.rs`'s module
//! doc for what is reproduced exactly, what is reproduced only up to summation order, and
//! what is out of scope; all of it still governs this file. `forward.rs` re-exports
//! [`HeadTailW`], so `v4oracle::forward::HeadTailW` still resolves; [`Oracle::embed`] and
//! [`Oracle::head_tail`] are methods on `Oracle` and every call site resolves unchanged.

use crate::v4oracle::breakages::Defect;
use crate::v4oracle::capture::Capture;
use crate::v4oracle::forward::Oracle;
use crate::v4oracle::numerics::sigmoid;
use crate::v4oracle::weights::WMat;

/// `Block.hc_head`, the final `RMSNorm`, and `ParallelHead` — the head tail's weights.
///
/// **Transliterated 2026-08-05, but never driven from the layer chain, and that restriction
/// is the point.** The note this replaces refused to transliterate the head tail at all, on
/// the grounds that the goldens stop at layer 4 of 43 and so a logits vector taken there is
/// not any quantity the model computes. That argument is still correct and is preserved by
/// construction: `bin/v4-oracle` never composes [`Oracle::head_tail`] with `run_layer`. It
/// drives it from a declared synthetic probe instead, so the emitted golden cannot be
/// mistaken for the model's logits — its input is visibly not a residual stream — while still
/// exercising the real weights at the real `dim` and `vocab_size`, which is all the device
/// side needs to be scored against.
///
/// What that buys is the whole point of the exercise: before it, the first decode's logits
/// were **ungated by construction**. Every per-layer golden could be perfect and the sampled
/// token still wrong, with nothing in the tree able to say so.
///
/// What it still does NOT cover, stated so nobody has to infer it:
/// - **The composition.** No golden anywhere asserts that 43 layers followed by this head
///   tail produce a particular logits vector. Only S4, with a full-depth run, can.
/// - **Weight SELECTION.** These are arithmetic goldens over whatever this struct is handed.
///   A port that fed `layers.42.hc_ffn_fn` where `hc_head_fn` was due would reproduce every
///   golden here exactly. The loader is what has to get that right.
/// - **Sampling.** `sample(logits, temperature)` (model.py:924) is out of scope, as is
///   `forward_spec` and everything MTP.
///
/// A struct of its own rather than fields hung off the embedding, because the two are
/// separately loadable: `bin/v4-oracle defects` drives the head tail and never embeds
/// anything, and `embed.weight` is 2.1 GB once widened to f32 on a machine whose memory is
/// shared with a live decode.
pub struct HeadTailW {
    /// `hc_head_fn`, `[hc_mult, hc_mult * dim]`. F32 on disk — `Transformer.__init__` builds
    /// it under `with set_dtype(torch.float32)`, so unlike the block weights there is no
    /// quantization and no bf16 store anywhere in its use.
    pub hc_head_fn: Vec<f32>,
    /// `hc_head_base`, `[hc_mult]` — one bias per hyper-connection copy.
    pub hc_head_base: Vec<f32>,
    /// `hc_head_scale`, `[1]`. A single scalar broadcast over every mix, where a `Block`'s
    /// `hc_*_scale` is `[3]` (pre, post, comb). Reusing the block's layout here would index
    /// past the tensor rather than merely computing the wrong thing, which is the one mistake
    /// on this path that fails loudly.
    pub hc_head_scale: Vec<f32>,
    /// `norm.weight`, `[dim]` — the final `RMSNorm`'s learned gain.
    pub norm: Vec<f32>,
    /// `head.weight`, `[vocab_size, dim]`. bf16 in the checkpoint and held as f32 by
    /// `ParallelHead`, so it takes `linear()`'s dense branch: **no activation quantization**,
    /// and the logits come out f32 and are never rounded.
    pub lm_head: WMat,
}

/// The residual rows the head consumes and the step tag their captures carry. Grouped
/// 2026-08-15 from three loose args.
pub struct HeadRows<'a> {
    pub h: &'a [f32],
    pub s: usize,
    pub step_tag: &'a str,
}

impl Oracle {
    /// `Transformer.forward` lines 914-916: embed, then expand to `hc_mult` copies.
    /// Returns `[s, hc_mult, dim]`.
    pub fn embed(&self, embed: &WMat, ids: &[u32]) -> Vec<f32> {
        let (hc, dim) = (self.cfg.hc_mult, self.cfg.dim);
        let mut row = Vec::with_capacity(dim);
        let mut out = Vec::with_capacity(ids.len() * hc * dim);
        for &t in ids {
            embed.row(t as usize, &mut row);
            for _ in 0..hc {
                out.extend_from_slice(&row);
            }
        }
        out
    }

    /// `Block.hc_head` (model.py:709-716). `h` is `[s, hc_mult, dim]`; the result is
    /// `[s, dim]`, bf16 as `y.to(dtype)` leaves it.
    ///
    /// This is `hc_pre`'s *pre* branch and nothing else: no Sinkhorn, no `post`, no
    /// combination matrix. It cannot be reached by reusing `hc_split_sinkhorn` even by
    /// accident — that wants `(2 + hc) * hc = 24` mixes and `hc_head_fn` yields `hc = 4` —
    /// which is why there is no defect for "ran the Sinkhorn here".
    ///
    /// Called as `layer.hc_head(...)` on the LAST block, so `norm_eps`/`hc_eps` are that
    /// block's. They are `args.norm_eps`/`args.hc_eps`, identical to the Transformer's, so
    /// reading them from `cfg` is exact rather than approximate.
    fn hc_head(&self, hw: &HeadTailW, h: &[f32], s: usize) -> Vec<f32> {
        let (hc, dim) = (self.cfg.hc_mult, self.cfg.dim);
        let hcd = hc * dim;
        assert_eq!(h.len(), s * hcd);
        assert_eq!(hw.hc_head_fn.len(), hc * hcd);
        assert_eq!(hw.hc_head_base.len(), hc);
        // `[1]`, not `[3]`: read the shape rather than trusting the caller to have loaded the
        // right tensor. `hc_attn_scale` has the same name shape and three entries, and
        // indexing [0] of it would be silently plausible.
        assert_eq!(
            hw.hc_head_scale.len(),
            1,
            "hc_head_scale is a scalar, not a Block's [3]"
        );
        let mut y = vec![0.0f32; s * dim];
        for t in 0..s {
            let flat = &h[t * hcd..(t + 1) * hcd];
            // `torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)` over the FULL
            // flattened row. One statistic for every mix; the per-copy variant below is the
            // wrong version, kept so the gate can be shown to reject it.
            let rs: Vec<f32> = match self.defect {
                Defect::HeadHcNoRsqrt => vec![1.0; hc],
                Defect::HeadHcRsqrtPerCopy => (0..hc)
                    .map(|c| {
                        let seg = &flat[c * dim..(c + 1) * dim];
                        let var = seg.iter().map(|v| v * v).sum::<f32>() / dim as f32;
                        (var + self.cfg.norm_eps).sqrt().recip()
                    })
                    .collect(),
                _ => {
                    let var = flat.iter().map(|v| v * v).sum::<f32>() / hcd as f32;
                    vec![(var + self.cfg.norm_eps).sqrt().recip(); hc]
                }
            };
            let mut pre = vec![0.0f32; hc];
            for (j, p) in pre.iter_mut().enumerate() {
                let w = &hw.hc_head_fn[j * hcd..(j + 1) * hcd];
                let m = flat.iter().zip(w).map(|(a, b)| a * b).sum::<f32>() * rs[j];
                *p = sigmoid(m * hw.hc_head_scale[0] + hw.hc_head_base[j]) + self.cfg.hc_eps;
            }
            self.hc_blend(&pre, flat, &mut y[t * dim..(t + 1) * dim]);
        }
        // `y.to(dtype)` — the residual stream this came from is bf16.
        self.round_bf16(&mut y);
        y
    }

    /// The whole head tail: `hc_head`, the final `RMSNorm`, then `ParallelHead`
    /// (model.py:922-923). Returns the `[vocab_size]` logits and records three goldens under
    /// `head.{step_tag}.`.
    ///
    /// It does NOT record its input. The caller owns that: in `bin/v4-oracle` the input is a
    /// declared probe and is pushed there as `head.probe.in`, and in the defect matrix it is
    /// already recorded as the layer's `.out`. Recording it here as well would put a second
    /// copy under a name ending in `.in`, which the matrix treats as fixed-by-construction —
    /// a silent claim no implementation could violate would then be attached to a tensor that
    /// every upstream defect moves.
    pub fn head_tail(&self, hw: &HeadTailW, rows: HeadRows<'_>, cap: &mut Capture) {
        let HeadRows { h, s, step_tag } = rows;
        let (dim, vocab) = (self.cfg.dim, self.cfg.vocab_size);
        // Both `[s, dim]` goldens below go through here. Two spelled-out `cap.push` calls is
        // what they were until the `step_tag` rename pushed each past `max_width` and rustfmt
        // reflowed them into a literal 27-token clone — the manufactured-duplication case
        // CLAUDE.md warns about. One closure states the prefix and the shape once.
        let mut record =
            |name: &str, v: Vec<f32>| cap.push(&format!("head.{step_tag}.{name}"), &[s, dim], v);
        let mut x = self.hc_head(hw, h, s);
        record("hc_head_out", x.clone());

        if self.defect != Defect::HeadNormSkipped {
            match self.defect {
                // A single-row norm kernel handed `s x dim`: one statistic over everything,
                // the learned gain still landing per dim because it repeats every `dim`.
                // Tiling the weight is how that is expressed with one norm routine rather
                // than a second copy of the loop.
                Defect::HeadNormOverAllTokens => {
                    let tiled: Vec<f32> = (0..s).flat_map(|_| hw.norm.iter().copied()).collect();
                    self.rmsnorm_raw(&mut x, s * dim, &tiled);
                }
                _ => self.rmsnorm_raw(&mut x, dim, &hw.norm),
            }
            if self.defect != Defect::HeadNormNotBf16 {
                self.round_bf16(&mut x);
            }
        }
        // `final_norm_out`, not `norm_out`: the matrix selects goldens by NAME SUFFIX, and
        // `.norm_out` would sit one character away from matching `.attn_norm_out` and
        // `.ffn_norm_out` for anyone who later writes the suffix without the leading dot.
        record("final_norm_out", x.clone());

        // `ParallelHead.forward` with `full_logits=False`: `x[:, -1]` — the LAST row only.
        let row = if self.defect == Defect::HeadLogitsFromFirstRow {
            0
        } else {
            s - 1
        };
        // `F.linear(x.float(), self.weight)` on an f32 parameter: the dense branch, so no
        // activation quantization, and the result is f32 and stays f32. Rounding it to bf16
        // here would be a defect all of its own; there is no store in the reference to model.
        let logits = self.linear(&x[row * dim..(row + 1) * dim], 1, &hw.lm_head);
        // Recorded, not returned. Both callers read their logits back out of `cap`, and a
        // return value would offer a second path to the number the goldens are the record of.
        cap.push(&format!("head.{step_tag}.logits"), &[1, vocab], logits);
    }
}
