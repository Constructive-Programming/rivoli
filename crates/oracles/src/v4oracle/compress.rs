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
//!
//! **Restructured 2026-11-20** under the CodeScene gate (Brain Method, Bumpy Road, nesting,
//! Complex Method and Excess Arguments on `compressor` and `indexer`), so "every body moved
//! unchanged" above now holds for the arithmetic, not the layout: the two bodies are cut into
//! the stages the reference's own comments name -- prefill stash, block pooling, decode slot
//! write, the post-pool store; the indexer's queries, weights, score row and causal selection --
//! and the public pair takes its operands as [`CompressIn`] / [`CompSinks`] / [`IndexerIn`].
//! Every arithmetic expression and its evaluation order is kept: the toy oracle's capture
//! fingerprints and counters agreed bit-for-bit before and after, across every defect, every
//! toy layer, seven prompt lengths from 1 to 17 and 9 decode steps (a scratch test hashing
//! every `Capture` of `run_layer`, run at eb801b7 and on this change). `freqs` is no longer an
//! indexer argument (every caller passed `self.freqs(step.layer)`), and the score matrix is
//! returned rather than appended to an out-parameter (both callers passed a fresh `Vec`).

use crate::v4oracle::breakages::Defect;
use crate::v4oracle::capture::Counters;
use crate::v4oracle::forward::{Oracle, softmax_strided, topk_idx};
use crate::v4oracle::layer::{CompState, CompressorW, IndexerW, LayerCtx};
use crate::v4oracle::numerics::{
    act_quant_inplace, bf16_decode, bf16_encode, fp4_act_quant_inplace, hadamard_rotate,
};
use crate::v4oracle::weights::WMat;

/// What one `Compressor.forward(x, start_pos)` call reads: `x` as `[s, dim]` in the block's
/// activation dtype, where it starts, and the RoPE table its layer uses.
#[derive(Clone, Copy)]
pub struct CompressIn<'a> {
    pub x: &'a [f32],
    pub s: usize,
    pub start_pos: usize,
    pub freqs: &'a [(f32, f32)],
}

/// What one compressor call mutates: the state it carries across calls (whose `cache` is
/// where the compressed rows land) and the run's counters.
pub struct CompSinks<'a> {
    pub st: &'a mut CompState,
    pub counters: &'a mut Counters,
}

/// What `Indexer.forward` reads besides the layer step: its weights, the block's
/// normalised input `x` (its compressor's and `weights_proj`'s operand), the LoRA-rank query
/// `qr` (`wq_b`'s), and `offset`, where its selections land in the attention's kv space.
pub struct IndexerIn<'a> {
    pub w: &'a IndexerW,
    pub x: &'a [f32],
    pub qr: &'a [f32],
    pub offset: usize,
}

/// One compressor call with its weights, the geometry `Compressor.__init__` fixes, and the
/// defect arms that alter it resolved once, so every stage below reads the same answer.
struct Pass<'a> {
    w: &'a CompressorW,
    ratio: usize,
    d: usize,
    /// `coff * d` — a projection/state row's width, from the weights' own `overlap`.
    cd: usize,
    /// `overlap` after `CompressorNoOverlap`.
    overlap: bool,
    /// `ape`, or `None` under `CompressorNoApe`.
    ape: Option<&'a [f32]>,
    inp: CompressIn<'a>,
}

impl Pass<'_> {
    /// Entries pooled per block: `ratio`, or `2 * ratio` with overlap.
    fn ents(&self) -> usize {
        if self.overlap {
            2 * self.ratio
        } else {
            self.ratio
        }
    }

    /// The state slot this block's own positions start at.
    fn state_off(&self) -> usize {
        if self.overlap { self.ratio } else { 0 }
    }

    /// `ape[j, e]`, or `0.0` when the bias is off.
    fn ape_at(&self, j: usize, e: usize) -> f32 {
        self.ape.map_or(0.0, |a| a[j * self.cd + e])
    }

    /// `overlap_transform`'s source for entry `j` of block `b`: `(block, position, half)`.
    /// With overlap the current block's "normal" half fills slots `[ratio, 2*ratio)` and the
    /// PREVIOUS block's "overlap" half slots `[0, ratio)`; block 0 has no previous block.
    fn overlap_src(&self, b: usize, j: usize) -> Option<(usize, usize, usize)> {
        if !self.overlap {
            Some((b, j, 0))
        } else if j >= self.ratio {
            Some((b, j - self.ratio, self.d))
        } else if b > 0 {
            Some((b - 1, j, 0))
        } else {
            None
        }
    }
}

/// The compressor's two projections of `x`: `wkv(x)` and `wgate(x)`, `[s, coff * d]` each.
struct Proj {
    kv: Vec<f32>,
    score: Vec<f32>,
}

/// Prefill (`start_pos == 0`): stash what the next call needs, then pool every whole block.
/// `None` when the prompt holds no whole block — after the stash, as the reference does.
fn prefill_pool(p: &Pass, st: &mut CompState, proj: Proj) -> Option<Vec<f32>> {
    stash_tail(p, st, &proj);
    let Proj { mut kv, mut score } = proj;
    let remainder = p.inp.s % p.ratio;
    let cutoff = p.inp.s - remainder;
    if remainder > 0 {
        kv.truncate(cutoff * p.cd);
        score.truncate(cutoff * p.cd);
    }
    if p.inp.s < p.ratio {
        return None;
    }
    add_ape(p, &mut score);
    let (kb, mut sb) = overlap_gather(p, &kv, &score);
    Some(softmax_pool(&kb, &mut sb, p.ents(), p.d))
}

/// The prefill state stash: with overlap the last whole block (the next block's "overlap"
/// half), and the positions past it in the slots after `state_off`, each score row biased by
/// `ape` at its position within the block.
fn stash_tail(p: &Pass, st: &mut CompState, proj: &Proj) {
    let (ratio, cd, kv, score) = (p.ratio, p.cd, &proj.kv, &proj.score);
    let remainder = p.inp.s % ratio;
    let cutoff = p.inp.s - remainder;
    let mut stash = |pos: usize, j: usize, slot: usize| {
        let (src, dst) = (pos * cd, slot * cd);
        st.kv_state[dst..dst + cd].copy_from_slice(&kv[src..src + cd]);
        for e in 0..cd {
            st.score_state[dst + e] = score[src + e] + p.ape_at(j, e);
        }
    };
    if p.overlap && cutoff >= ratio {
        for j in 0..ratio {
            stash(cutoff - ratio + j, j, j);
        }
    }
    for j in 0..remainder {
        stash(cutoff + j, j, p.state_off() + j);
    }
}

/// `score += ape`, per position within the block, over every whole block.
fn add_ape(p: &Pass, score: &mut [f32]) {
    let Some(ape) = p.ape else { return };
    let (ratio, cd) = (p.ratio, p.cd);
    for b in 0..p.inp.s / ratio {
        for j in 0..ratio {
            for e in 0..cd {
                score[(b * ratio + j) * cd + e] += ape[j * cd + e];
            }
        }
    }
}

/// `overlap_transform`: `ents` entries of width `d` per block — zero kv and `-inf` score
/// where block 0 has no previous block to borrow from.
fn overlap_gather(p: &Pass, kv: &[f32], score: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (nblk, ents, d) = (p.inp.s / p.ratio, p.ents(), p.d);
    let mut kb = vec![0.0f32; nblk * ents * d];
    let mut sb = vec![f32::NEG_INFINITY; nblk * ents * d];
    for b in 0..nblk {
        for j in 0..ents {
            let Some((sblk, src_j, half)) = p.overlap_src(b, j) else {
                continue;
            };
            let src = (sblk * p.ratio + src_j) * p.cd + half;
            let dst = (b * ents + j) * d;
            kb[dst..dst + d].copy_from_slice(&kv[src..src + d]);
            sb[dst..dst + d].copy_from_slice(&score[src..src + d]);
        }
    }
    (kb, sb)
}

/// Softmax over the ENTRY axis, independently per feature, then the weighted sum —
/// `[nblk, ents, d]` in, `[nblk, d]` out. Prefill and decode pool through this one body.
fn softmax_pool(kb: &[f32], sb: &mut [f32], ents: usize, d: usize) -> Vec<f32> {
    let nblk = kb.len() / (ents * d);
    let mut out = vec![0.0f32; nblk * d];
    for b in 0..nblk {
        for e in 0..d {
            softmax_strided(sb, ents, d, b * ents * d + e);
            for j in 0..ents {
                out[b * d + e] += kb[(b * ents + j) * d + e] * sb[(b * ents + j) * d + e];
            }
        }
    }
    out
}

/// Decode: write this position into its state slot, and on a block's last position pool
/// the block from the state and shift the overlap half down. `None` mid-block.
fn decode_pool(p: &Pass, st: &mut CompState, proj: Proj) -> Option<Vec<f32>> {
    let (ratio, d, cd) = (p.ratio, p.d, p.cd);
    let Proj { kv, mut score } = proj;
    let slot_in_block = p.inp.start_pos % ratio;
    if let Some(ape) = p.ape {
        let ape = &ape[slot_in_block * cd..(slot_in_block + 1) * cd];
        for (v, a) in score.iter_mut().zip(ape) {
            *v += a;
        }
    }
    let dst = (p.state_off() + slot_in_block) * cd;
    st.kv_state[dst..dst + cd].copy_from_slice(&kv[..cd]);
    st.score_state[dst..dst + cd].copy_from_slice(&score[..cd]);
    if !(p.inp.start_pos + 1).is_multiple_of(ratio) {
        return None;
    }
    let (kb, mut sb) = state_gather(p, st);
    let out = softmax_pool(&kb, &mut sb, p.ents(), d);
    if p.overlap {
        let (lo, hi) = st.kv_state.split_at_mut(ratio * cd);
        lo.copy_from_slice(&hi[..ratio * cd]);
        let (lo, hi) = st.score_state.split_at_mut(ratio * cd);
        lo.copy_from_slice(&hi[..ratio * cd]);
    }
    Some(out)
}

/// Gather the state into `ents` entries: slots `[0, ratio)` contribute their first `d` dims
/// (the overlap half), slots `[ratio, 2*ratio)` their last `d` dims. Without overlap,
/// `cd == d` and every slot contributes all of itself.
fn state_gather(p: &Pass, st: &CompState) -> (Vec<f32>, Vec<f32>) {
    let (ents, d, cd) = (p.ents(), p.d, p.cd);
    let mut kb = vec![0.0f32; ents * d];
    let mut sb = vec![0.0f32; ents * d];
    for j in 0..ents {
        let half = if p.overlap && j >= p.ratio { d } else { 0 };
        kb[j * d..(j + 1) * d].copy_from_slice(&st.kv_state[j * cd + half..j * cd + half + d]);
        sb[j * d..(j + 1) * d].copy_from_slice(&st.score_state[j * cd + half..j * cd + half + d]);
    }
    (kb, sb)
}

impl Oracle {
    /// `Compressor.forward`. `x` is `[s, dim]` in the block's activation dtype; the
    /// reference immediately casts it to f32 and stays there until the final `.to(dtype)`.
    ///
    /// Returns the compressed rows `[n_blocks, d]` — post-norm, post-RoPE, post-quantization,
    /// exactly what lands in the cache — or `None` when `should_compress` is false. In the
    /// `None` case the state updates have still happened, exactly as the reference does
    /// (model.py:331-367).
    ///
    /// `pub` for S2c: the compressor and indexer goldens are otherwise reachable only
    /// through `run_layer`, which drags in the full MoE and 3.4 GB of experts per layer.
    /// Driving these in isolation is what closes three measured coverage holes -- ratio-128
    /// pooling (no golden at all at a 13-token prompt), its empty `[13,0]` selection
    /// tensor, and the ranking, which `index_topk` never truncates below 2052 tokens.
    /// Visibility only; no behaviour change.
    pub fn compressor(
        &self,
        cw: &CompressorW,
        inp: CompressIn<'_>,
        sinks: CompSinks<'_>,
    ) -> Option<Vec<f32>> {
        let p = self.pass(cw, inp);
        let proj = Proj {
            kv: self.linear(inp.x, inp.s, &cw.wkv),
            score: self.linear(inp.x, inp.s, &cw.wgate),
        };
        let mut pooled = if inp.start_pos == 0 {
            prefill_pool(&p, sinks.st, proj)
        } else {
            decode_pool(&p, sinks.st, proj)
        }?;
        self.store_blocks(&p, &mut pooled, &mut sinks.st.cache);
        sinks.counters.compressed_blocks += pooled.len() / p.d;
        Some(pooled)
    }

    fn pass<'a>(&self, cw: &'a CompressorW, inp: CompressIn<'a>) -> Pass<'a> {
        Pass {
            w: cw,
            ratio: cw.ratio,
            d: cw.d,
            cd: cw.coff() * cw.d,
            overlap: cw.overlap && self.defect != Defect::CompressorNoOverlap,
            ape: (self.defect != Defect::CompressorNoApe).then_some(&cw.ape[..]),
            inp,
        }
    }

    /// `self.norm(kv.to(dtype))` — bf16 store, then RMSNorm back to bf16 — then per block
    /// RoPE at the block's FIRST position, the quantization its compressor uses, and the
    /// write into the compressed cache.
    fn store_blocks(&self, p: &Pass, pooled: &mut [f32], cache: &mut [f32]) {
        let (ratio, d) = (p.ratio, p.d);
        let nblk = pooled.len() / d;
        let first_block = p.inp.start_pos / ratio;
        self.round_bf16(pooled);
        self.rmsnorm(pooled, d, &p.w.norm);
        for b in 0..nblk {
            let block = first_block + b;
            let row = &mut pooled[b * d..(b + 1) * d];
            // model.py:370/372 — the block is rotated at its FIRST position.
            let pos = if self.defect == Defect::CompressorRopeAtBlockEnd {
                block * ratio + ratio - 1
            } else {
                block * ratio
            };
            self.rope_row(row, (pos, p.inp.freqs), false);
            if p.w.rotate {
                self.indexer_spread(row);
            } else {
                self.kv_act_quant(row, d, self.cfg.rope_head_dim);
            }
            let dst = block * d;
            // Fail CLOSED. Silently dropping the row would leave the indexer scoring
            // queries against a zero slot -- fluent wrong text, no crash, which is the
            // exact failure mode this oracle exists to make impossible.
            assert!(
                dst + d <= cache.len(),
                "compressed block {block} exceeds max_seq_len/{ratio}; raise cfg.max_seq_len"
            );
            cache[dst..dst + d].copy_from_slice(row);
        }
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
    /// with `-1` for masked slots, and the FULL `[s, n_compressed]` score matrix — not just
    /// the selected entries — so that its length does not depend on the selection under
    /// test, and so a consumer can tell a `topk` tie-break disagreement from a real scoring
    /// disagreement.
    ///
    /// `pub` for S2c: the compressor and indexer goldens are otherwise reachable only
    /// through `run_layer`, which drags in the full MoE and 3.4 GB of experts per layer.
    /// Driving these in isolation is what closes three measured coverage holes -- ratio-128
    /// pooling (no golden at all at a 13-token prompt), its empty `[13,0]` selection
    /// tensor, and the ranking, which `index_topk` never truncates below 2052 tokens.
    /// Visibility only; no behaviour change.
    pub fn indexer(
        &self,
        step: &LayerCtx,
        inp: IndexerIn<'_>,
        sinks: CompSinks<'_>,
    ) -> (Vec<Vec<i64>>, Vec<f32>) {
        let LayerCtx { s, start_pos, .. } = *step;
        let c = &self.cfg;
        let (h, hd, ratio) = (c.index_n_heads, c.index_head_dim, inp.w.compressor.ratio);
        let freqs = self.freqs(step.layer);
        let q = self.indexer_q(step, &inp.w.wq_b, inp.qr);

        // Its own scratch counter: the indexer's compressor is not the attention's.
        let mut own = Counters::default();
        let at = CompressIn {
            x: inp.x,
            s,
            start_pos,
            freqs,
        };
        let comp = CompSinks {
            st: &mut *sinks.st,
            counters: &mut own,
        };
        self.compressor(&inp.w.compressor, at, comp);
        sinks.counters.indexer_compressed_blocks += own.compressed_blocks;

        let w = self.indexer_weights(inp.x, s, &inp.w.weights_proj);
        let n_comp = (start_pos + s) / ratio;
        let (mut out, mut scores) = (Vec::with_capacity(s), Vec::new());
        sinks.counters.indexer_ran = true;
        for t in 0..s {
            let mut score = self.index_scores(
                &q[t * h * hd..(t + 1) * h * hd],
                &w[t * h..(t + 1) * h],
                &sinks.st.cache[..n_comp * hd],
            );
            // Causal mask over compressed blocks, applied before topk (model.py:430-432)
            // and again to the SELECTED indices afterwards (:434-436) — the second pass is
            // what turns a fully-masked row's arbitrary topk into -1s.
            let limit = (start_pos == 0).then_some((t + 1) / ratio);
            if let Some(limit) = limit {
                score
                    .iter_mut()
                    .skip(limit)
                    .for_each(|sc| *sc = f32::NEG_INFINITY);
            }
            scores.extend_from_slice(&score);
            let k = c.index_topk.min(n_comp);
            sinks.counters.indexer_truncated += usize::from(k < n_comp);
            let sel = topk_idx(&score, k).into_iter();
            out.push(
                sel.map(|i| match limit {
                    Some(limit) if i >= limit => -1,
                    _ => (i + inp.offset) as i64,
                })
                .collect(),
            );
        }
        (out, scores)
    }

    /// The indexer's queries: `wq_b(qr)` to bf16, then per head RoPE and the
    /// Hadamard + fp4 spread its compressed kv rows also get.
    fn indexer_q(&self, step: &LayerCtx, wq_b: &WMat, qr: &[f32]) -> Vec<f32> {
        let freqs = self.freqs(step.layer);
        let (h, hd) = (self.cfg.index_n_heads, self.cfg.index_head_dim);
        let mut q = self.linear(qr, step.s, wq_b);
        self.round_bf16(&mut q);
        for t in 0..step.s {
            for hh in 0..h {
                let row = &mut q[(t * h + hh) * hd..(t * h + hh + 1) * hd];
                self.rope_row(row, (step.start_pos + t, freqs), false);
                self.indexer_spread(row);
            }
        }
        q
    }

    /// `weights_proj(x) * (softmax_scale * n_heads ** -0.5)`, `[s, h]`. bf16 all the way:
    /// `weights_proj` is a bf16 `Linear`, so the scale multiply lands in bf16 too
    /// (model.py:424).
    fn indexer_weights(&self, x: &[f32], s: usize, weights_proj: &WMat) -> Vec<f32> {
        let (h, hd) = (self.cfg.index_n_heads, self.cfg.index_head_dim);
        let mut w = self.linear(x, s, weights_proj);
        self.round_bf16(&mut w);
        let wscale = (hd as f32).powf(-0.5) * (h as f32).powf(-0.5);
        for v in w.iter_mut() {
            *v = bf16_decode(bf16_encode(*v * wscale));
        }
        w
    }

    /// One query row's score against every compressed entry in `cache` (`[n_comp, hd]`).
    /// `qt` is the row's `[h, hd]` queries and `wt` its `[h]` head weights.
    fn index_scores(&self, qt: &[f32], wt: &[f32], cache: &[f32]) -> Vec<f32> {
        let hd = self.cfg.index_head_dim;
        let mut score = vec![0.0f32; cache.len() / hd];
        for (ci, sc) in score.iter_mut().enumerate() {
            let kvc = &cache[ci * hd..(ci + 1) * hd];
            // `einsum` -> bf16, `relu_()` in place -> bf16, `* weights` -> bf16 are all
            // ELEMENTWISE and genuinely land in bf16 (model.py:426-427); the final
            // `.sum(dim=2)` is a REDUCTION and accumulates in f32, rounding once. Those
            // two halves were conflated here until 2026-08-05 -- see `bf16_sum`, which
            // carries the measurement. This chain decides WHICH blocks are attended, so
            // a faithful kernel's bf16 and an f32 oracle can disagree on the SET near a
            // tie, which no numeric tolerance would show.
            *sc = self.bf16_sum(
                wt.iter()
                    .enumerate()
                    .map(|(hh, &w)| self.index_term(&qt[hh * hd..(hh + 1) * hd], kvc, w)),
            );
        }
        score
    }

    /// One head's term: the einsum itself — f32 accumulation, one bf16 store, which is
    /// torch's bf16 matmul — then `relu` and the head weight, each landing in bf16.
    fn index_term(&self, qh: &[f32], kvc: &[f32], w: f32) -> f32 {
        let hd = qh.len();
        let mut dot = bf16_decode(bf16_encode((0..hd).map(|i| qh[i] * kvc[i]).sum::<f32>()));
        if self.defect != Defect::IndexerNoRelu {
            dot = dot.max(0.0);
        }
        let wt = if self.defect == Defect::IndexerNoWeights {
            1.0
        } else {
            w
        };
        bf16_decode(bf16_encode(dot * wt))
    }
}
