//! `Compressor.forward` and `Indexer.forward` — the pooling that turns `ratio` positions
//! into one compressed entry, and the scoring that then picks which of those entries a
//! query may attend to.
//!
//! **Split out of `forward.rs` on 2026-08-15, verbatim**, under the 800-line file gate
//! (`crates/cli/tests/line_limit.rs`) and the whole-tree CodeScene 10/10 gate
//! (`crates/cli/tests/codescene.rs`). The cut is by COHESION, and the two stages are ONE
//! file rather than two because the indexer owns a second `Compressor` of its own
//! (`rotate=True`: Hadamard + fp4 where the attention one does partial fp8) and calls it
//! directly — [`Oracle::indexer_spread`] is the pair they share, and separating them would
//! put that helper across a seam the reference does not have.
//!
//! Every body moved unchanged. This is a frozen transliteration — see `forward.rs`'s module
//! doc for what is reproduced exactly, what is reproduced only up to summation order, and
//! what is out of scope; all of it still governs this file. [`Oracle::compressor`] and
//! [`Oracle::indexer`] keep their `pub` visibility for S2c's isolated drives, and
//! `Oracle::kv_act_quant` stays `pub(super)` for `attention.rs`.

use crate::v4oracle::breakages::Defect;
use crate::v4oracle::capture::Counters;
use crate::v4oracle::forward::{Oracle, softmax_strided, topk_idx};
use crate::v4oracle::layer::{CompState, CompressorW, IndexerW, LayerCtx};
use crate::v4oracle::numerics::{
    act_quant_inplace, bf16_decode, bf16_encode, fp4_act_quant_inplace, hadamard_rotate,
};

impl Oracle {
    /// `Compressor.forward`. `x` is `[s, dim]` in the block's activation dtype; the
    /// reference immediately casts it to f32 and stays there until the final `.to(dtype)`.
    ///
    /// Returns the compressed rows `[n_blocks, d]` — post-norm, post-RoPE, post-quantization,
    /// exactly what lands in the cache — or `None` when `should_compress` is false. In the
    /// `None` case the state updates have still happened, exactly as the reference does
    /// (model.py:331-367).
    #[allow(clippy::too_many_arguments)]
    /// `pub` for S2c: the compressor and indexer goldens are otherwise reachable only
    /// through `run_layer`, which drags in the full MoE and 3.4 GB of experts per layer.
    /// Driving these in isolation is what closes three measured coverage holes -- ratio-128
    /// pooling (no golden at all at a 13-token prompt), its empty `[13,0]` selection
    /// tensor, and the ranking, which `index_topk` never truncates below 2052 tokens.
    /// Visibility only; no behaviour change.
    pub fn compressor(
        &self,
        cw: &CompressorW,
        cs: &mut CompState,
        x: &[f32],
        s: usize,
        start_pos: usize,
        freqs: &[(f32, f32)],
        counters: &mut Counters,
    ) -> Option<Vec<f32>> {
        let (ratio, d, coff) = (cw.ratio, cw.d, cw.coff());
        let overlap = cw.overlap && self.defect != Defect::CompressorNoOverlap;
        let rd = self.cfg.rope_head_dim;
        let cd = coff * d;
        let use_ape = self.defect != Defect::CompressorNoApe;

        let mut kv = self.linear(x, s, &cw.wkv);
        let mut score = self.linear(x, s, &cw.wgate);

        let (mut pooled, first_block) = if start_pos == 0 {
            let should = s >= ratio;
            let remainder = s % ratio;
            let cutoff = s - remainder;
            let state_off = if overlap { ratio } else { 0 };
            if overlap && cutoff >= ratio {
                for j in 0..ratio {
                    let src = (cutoff - ratio + j) * cd;
                    cs.kv_state[j * cd..(j + 1) * cd].copy_from_slice(&kv[src..src + cd]);
                    for e in 0..cd {
                        cs.score_state[j * cd + e] =
                            score[src + e] + if use_ape { cw.ape[j * cd + e] } else { 0.0 };
                    }
                }
            }
            if remainder > 0 {
                for j in 0..remainder {
                    let src = (cutoff + j) * cd;
                    let dst = (state_off + j) * cd;
                    cs.kv_state[dst..dst + cd].copy_from_slice(&kv[src..src + cd]);
                    for e in 0..cd {
                        cs.score_state[dst + e] =
                            score[src + e] + if use_ape { cw.ape[j * cd + e] } else { 0.0 };
                    }
                }
                kv.truncate(cutoff * cd);
                score.truncate(cutoff * cd);
            }
            if !should {
                return None;
            }
            let nblk = cutoff / ratio;
            // score += ape, per position within the block
            if use_ape {
                for b in 0..nblk {
                    for j in 0..ratio {
                        for e in 0..cd {
                            score[(b * ratio + j) * cd + e] += cw.ape[j * cd + e];
                        }
                    }
                }
            }
            // `overlap_transform`: 2*ratio entries of width d per block — the current
            // block's "normal" half in slots [ratio, 2*ratio) and the PREVIOUS block's
            // "overlap" half in slots [0, ratio), zero / -inf for block 0.
            let ents = if overlap { 2 * ratio } else { ratio };
            let mut kb = vec![0.0f32; nblk * ents * d];
            let mut sb = vec![f32::NEG_INFINITY; nblk * ents * d];
            for b in 0..nblk {
                for j in 0..ents {
                    let (src_blk, src_j, half) = if !overlap {
                        (Some(b), j, 0)
                    } else if j >= ratio {
                        (Some(b), j - ratio, d)
                    } else if b > 0 {
                        (Some(b - 1), j, 0)
                    } else {
                        (None, 0, 0)
                    };
                    let Some(sblk) = src_blk else { continue };
                    let src = (sblk * ratio + src_j) * cd + half;
                    let dst = (b * ents + j) * d;
                    kb[dst..dst + d].copy_from_slice(&kv[src..src + d]);
                    sb[dst..dst + d].copy_from_slice(&score[src..src + d]);
                }
            }
            // softmax over the ENTRY axis, independently per feature.
            let mut out = vec![0.0f32; nblk * d];
            for b in 0..nblk {
                for e in 0..d {
                    softmax_strided(&mut sb, ents, d, b * ents * d + e);
                    for j in 0..ents {
                        out[b * d + e] += kb[(b * ents + j) * d + e] * sb[(b * ents + j) * d + e];
                    }
                }
            }
            (out, 0)
        } else {
            let should = (start_pos + 1).is_multiple_of(ratio);
            let slot_in_block = start_pos % ratio;
            if use_ape {
                let ape = &cw.ape[slot_in_block * cd..(slot_in_block + 1) * cd];
                for (v, a) in score.iter_mut().zip(ape) {
                    *v += a;
                }
            }
            let state_off = if overlap { ratio } else { 0 };
            let dst = (state_off + slot_in_block) * cd;
            cs.kv_state[dst..dst + cd].copy_from_slice(&kv[..cd]);
            cs.score_state[dst..dst + cd].copy_from_slice(&score[..cd]);
            if !should {
                return None;
            }
            let ents = if overlap { 2 * ratio } else { ratio };
            // Gather: slots [0, ratio) contribute their first d dims (the overlap half),
            // slots [ratio, 2*ratio) their last d dims. Without overlap, cd == d and every
            // slot contributes all of itself.
            let mut kb = vec![0.0f32; ents * d];
            let mut sb = vec![0.0f32; ents * d];
            for j in 0..ents {
                let half = if overlap && j >= ratio { d } else { 0 };
                kb[j * d..(j + 1) * d]
                    .copy_from_slice(&cs.kv_state[j * cd + half..j * cd + half + d]);
                sb[j * d..(j + 1) * d]
                    .copy_from_slice(&cs.score_state[j * cd + half..j * cd + half + d]);
            }
            let mut out = vec![0.0f32; d];
            for e in 0..d {
                softmax_strided(&mut sb, ents, d, e);
                for j in 0..ents {
                    out[e] += kb[j * d + e] * sb[j * d + e];
                }
            }
            if overlap {
                let (lo, hi) = cs.kv_state.split_at_mut(ratio * cd);
                lo.copy_from_slice(&hi[..ratio * cd]);
                let (lo, hi) = cs.score_state.split_at_mut(ratio * cd);
                lo.copy_from_slice(&hi[..ratio * cd]);
            }
            (out, start_pos / ratio)
        };

        let nblk = pooled.len() / d;
        // `self.norm(kv.to(dtype))` — bf16 store, then RMSNorm back to bf16.
        self.round_bf16(&mut pooled);
        self.rmsnorm(&mut pooled, d, &cw.norm);
        for b in 0..nblk {
            // model.py:370/372 — the block is rotated at its FIRST position.
            let block = first_block + b;
            let pos = if self.defect == Defect::CompressorRopeAtBlockEnd {
                block * ratio + ratio - 1
            } else {
                block * ratio
            };
            let row = &mut pooled[b * d..(b + 1) * d];
            self.rope_row(row, rd, (pos, freqs), false);
            if cw.rotate {
                self.indexer_spread(row);
            } else {
                self.kv_act_quant(row, d, rd);
            }
            let dst = block * d;
            // Fail CLOSED. Silently dropping the row would leave the indexer scoring
            // queries against a zero slot -- fluent wrong text, no crash, which is the
            // exact failure mode this oracle exists to make impossible.
            assert!(
                dst + d <= cs.cache.len(),
                "compressed block {block} exceeds max_seq_len/{ratio}; raise cfg.max_seq_len"
            );
            cs.cache[dst..dst + d].copy_from_slice(row);
        }
        counters.compressed_blocks += nblk;
        Some(pooled)
    }

    /// `rotate_activation` then `fp4_act_quant(·, 32, inplace=True)` — what the indexer does
    /// to BOTH its query rows and its compressed kv rows (`Indexer.forward` lines 420-422,
    /// `Compressor.forward` lines 374-376). One helper because the pair must stay together:
    /// the Hadamard spread exists to make the fp4 grouping well-conditioned, so applying
    /// one without the other is a different algorithm, not a partial one.
    fn indexer_spread(&self, row: &mut [f32]) {
        if self.defect != Defect::IndexerNoHadamard {
            hadamard_rotate(row);
            self.round_bf16(row);
        }
        if self.defect != Defect::IndexerNoFp4Quant {
            fp4_act_quant_inplace(row, 32);
            self.round_bf16(row);
        }
    }

    /// The PARTIAL fp8 simulation of a KV entry: `act_quant(kv[..., :-rope_head_dim], 64,
    /// scale_fmt, scale_dtype, inplace=True)` — dims `[0, d - rd)` at block 64, dims
    /// `[d - rd, d)` left alone so the positional information keeps bf16 precision.
    pub(super) fn kv_act_quant(&self, row: &mut [f32], d: usize, rd: usize) {
        let (n, block, round) = match self.defect {
            Defect::SkipKvActQuant => return,
            Defect::KvActQuantWholeTensor => (d, 64, true),
            Defect::KvActQuantBlock128 => (d - rd, 128, true),
            Defect::KvActQuantNoRoundScale => (d - rd, 64, false),
            _ => (d - rd, 64, true),
        };
        act_quant_inplace(&mut row[..n], block, round);
        self.round_bf16(row);
    }
}

// ---------------------------------------------------------------------------------------
// Indexer
// ---------------------------------------------------------------------------------------

impl Oracle {
    /// `Indexer.forward` — selects which compressed positions each query may attend to.
    ///
    /// Returns one index list per query row, already offset into the attention's kv space,
    /// with `-1` for masked slots. `scores_out` receives the FULL `[s, n_compressed]` score
    /// matrix — not just the selected entries — so that its length does not depend on the
    /// selection under test, and so a consumer can tell a `topk` tie-break disagreement from
    /// a real scoring disagreement.
    #[allow(clippy::too_many_arguments)]
    /// `pub` for S2c: the compressor and indexer goldens are otherwise reachable only
    /// through `run_layer`, which drags in the full MoE and 3.4 GB of experts per layer.
    /// Driving these in isolation is what closes three measured coverage holes -- ratio-128
    /// pooling (no golden at all at a 13-token prompt), its empty `[13,0]` selection
    /// tensor, and the ranking, which `index_topk` never truncates below 2052 tokens.
    /// Visibility only; no behaviour change.
    pub fn indexer(
        &self,
        step: &LayerCtx,
        iw: &IndexerW,
        cs: &mut CompState,
        x: &[f32],
        qr: &[f32],
        offset: usize,
        freqs: &[(f32, f32)],
        counters: &mut Counters,
        scores_out: &mut Vec<f32>,
    ) -> Vec<Vec<i64>> {
        let LayerCtx { s, start_pos, .. } = *step;
        let c = &self.cfg;
        let (h, hd, ratio, rd) = (
            c.index_n_heads,
            c.index_head_dim,
            iw.compressor.ratio,
            c.rope_head_dim,
        );
        let end_pos = start_pos + s;

        let mut q = self.linear(qr, s, &iw.wq_b);
        self.round_bf16(&mut q);
        for t in 0..s {
            for hh in 0..h {
                let row = &mut q[(t * h + hh) * hd..(t * h + hh + 1) * hd];
                self.rope_row(row, rd, (start_pos + t, freqs), false);
                self.indexer_spread(row);
            }
        }

        // Its own scratch counter: the indexer's compressor is not the attention's.
        let mut own = Counters::default();
        self.compressor(&iw.compressor, cs, x, s, start_pos, freqs, &mut own);
        counters.indexer_compressed_blocks += own.compressed_blocks;

        // `weights_proj(x) * (softmax_scale * n_heads ** -0.5)`.
        let mut w = self.linear(x, s, &iw.weights_proj);
        self.round_bf16(&mut w);
        // bf16 all the way: `weights_proj` is a bf16 `Linear`, so the scale multiply lands
        // in bf16 too (model.py:424).
        let wscale = (hd as f32).powf(-0.5) * (h as f32).powf(-0.5);
        for v in w.iter_mut() {
            *v = bf16_decode(bf16_encode(*v * wscale));
        }

        let n_comp = end_pos / ratio;
        let mut out = Vec::with_capacity(s);
        counters.indexer_ran = true;
        for t in 0..s {
            let mut score = vec![0.0f32; n_comp];
            for (ci, sc) in score.iter_mut().enumerate() {
                // `einsum` -> bf16, `relu_()` in place -> bf16, `* weights` -> bf16 are all
                // ELEMENTWISE and genuinely land in bf16 (model.py:426-427); the final
                // `.sum(dim=2)` is a REDUCTION and accumulates in f32, rounding once. Those
                // two halves were conflated here until 2026-08-05 -- see `bf16_sum`, which
                // carries the measurement. This chain decides WHICH blocks are attended, so
                // a faithful kernel's bf16 and an f32 oracle can disagree on the SET near a
                // tie, which no numeric tolerance would show.
                *sc = self.bf16_sum((0..h).map(|hh| {
                    let qh = &q[(t * h + hh) * hd..(t * h + hh + 1) * hd];
                    let kvc = &cs.cache[ci * hd..(ci + 1) * hd];
                    // The einsum itself: f32 accumulation, one bf16 store, which is torch's
                    // bf16 matmul and was already right.
                    let mut dot =
                        bf16_decode(bf16_encode((0..hd).map(|i| qh[i] * kvc[i]).sum::<f32>()));
                    if self.defect != Defect::IndexerNoRelu {
                        dot = dot.max(0.0);
                    }
                    let wt = if self.defect == Defect::IndexerNoWeights {
                        1.0
                    } else {
                        w[t * h + hh]
                    };
                    bf16_decode(bf16_encode(dot * wt))
                }));
            }
            // Causal mask over compressed blocks, applied before topk (model.py:430-432)
            // and again to the SELECTED indices afterwards (:434-436) — the second pass is
            // what turns a fully-masked row's arbitrary topk into -1s.
            let limit = if start_pos == 0 {
                (t + 1) / ratio
            } else {
                n_comp
            };
            if start_pos == 0 {
                for (ci, sc) in score.iter_mut().enumerate() {
                    if ci >= limit {
                        *sc = f32::NEG_INFINITY;
                    }
                }
            }
            scores_out.extend_from_slice(&score);
            let k = c.index_topk.min(n_comp);
            counters.indexer_truncated += usize::from(k < n_comp);
            let sel = topk_idx(&score, k);
            out.push(
                sel.iter()
                    .map(|&i| {
                        if start_pos == 0 && i >= limit {
                            -1
                        } else {
                            (i + offset) as i64
                        }
                    })
                    .collect(),
            );
        }
        out
    }
}
