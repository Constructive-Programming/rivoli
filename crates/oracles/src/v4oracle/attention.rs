//! `Attention.forward` (model.py:490-548) — the sparse-attention frontend of the V4
//! transliteration: the two index-list constructors, the online softmax that reads them, and
//! the six stages `Oracle::attention` runs in the reference's order.
//!
//! **Split out of `forward.rs` on 2026-08-15, verbatim.** `forward.rs` had crossed
//! CodeScene's file-size cliff — 1068 code lines against a cliff measured at ~880, scoring
//! 9.09 — and the whole-tree 10/10 gate in `crates/cli/tests/codescene.rs` is a merge gate.
//! The cut is by COHESION, not by size: everything below is reached from `Oracle::attention`
//! and nothing else in the block body reaches into it. `run_layer` calls `attention` and
//! threads [`Sinks`]; that pair is the ENTIRE seam.
//!
//! Every body moved unchanged. This is a frozen transliteration — the arithmetic is
//! `model.py`'s, and the goldens are what fp8/bf16 arithmetic produces rather than "f32
//! reference values" — so tidying anything on the way across would be a value change wearing
//! a refactor's clothes. Read `forward.rs`'s module doc for what is reproduced exactly, what
//! is reproduced only up to summation order, and what is out of scope; all of it still
//! governs this file.
//!
//! The seam's cost, stated once so it is not rediscovered: `forward.rs` widens five
//! module-internal primitives to `pub(super)` — `Oracle::{wrow, round_bf16, rmsnorm,
//! rope_row, kv_act_quant}` — and [`Sinks`] plus `Oracle::attention` travel the other way.
//! None of that is public API. [`window_topk`] and [`compress_topk`] are the only `pub` items
//! that moved, and `forward.rs` re-exports both, so every existing
//! `v4oracle::forward::{window_topk, compress_topk}` path still resolves.

use crate::v4oracle::forward::{Capture, Defect, LayerCtx, LayerRings, Oracle};

/// `[[f(t, j) for j in range(cols)] for t in range(seqlen)]` — the shape BOTH prefill
/// comprehensions in `model.py` have, and nothing else.
///
/// It owns the RECTANGULARITY that `attn::gather_slot_idxs` refuses a ragged buffer for, and no
/// selection logic: every index and every `-1` below stays with its own function, so a wrong
/// one cannot travel between [`window_topk`] and [`compress_topk`].
fn prefill_rows(seqlen: usize, cols: usize, f: impl Fn(usize, usize) -> i64) -> Vec<Vec<i64>> {
    (0..seqlen)
        .map(|t| (0..cols).map(|j| f(t, j)).collect())
        .collect()
}

/// `model.py::get_window_topk_idxs` — the sliding-window index list for each query row.
pub fn window_topk(win: usize, seqlen: usize, start_pos: usize) -> Vec<Vec<i64>> {
    if start_pos >= win.saturating_sub(1) && start_pos > 0 {
        let sp = start_pos % win;
        let mut v: Vec<i64> = ((sp + 1)..win).map(|i| i as i64).collect();
        v.extend((0..=sp).map(|i| i as i64));
        vec![v]
    } else if start_pos > 0 {
        let mut v: Vec<i64> = (0..=start_pos).map(|i| i as i64).collect();
        v.resize(win, -1);
        vec![v]
    } else {
        // `win == 0` gives `cols == 0`, so `win - 1` is never reached and this is `seqlen`
        // empty rows rather than the overflow panic a dev-profile build used to take. Not
        // reachable: `attn::gather_slot_idxs` returns early on `win == 0` and `sliding_window` is
        // 128 in the shipped config. Recorded because it is the one input on which this
        // spelling and the nested one it replaced differ.
        prefill_rows(seqlen, win.min(seqlen), |t, j| {
            let v = t.saturating_sub(win - 1) + j;
            if v > t { -1 } else { v as i64 }
        })
    }
}

/// `model.py::get_compress_topk_idxs` — the arithmetic (indexer-free) compressed selection
/// used where `compress_ratio != 4`.
/// `pub` for S2c: the compressor and indexer goldens are otherwise reachable only
/// through `run_layer`, which drags in the full MoE and 3.4 GB of experts per layer.
/// Driving these in isolation is what closes three measured coverage holes -- ratio-128
/// pooling (no golden at all at a 13-token prompt), its empty `[13,0]` selection
/// tensor, and the ranking, which `index_topk` never truncates below 2052 tokens.
/// Visibility only; no behaviour change.
pub fn compress_topk(
    ratio: usize,
    seqlen: usize,
    start_pos: usize,
    offset: usize,
) -> Vec<Vec<i64>> {
    if start_pos > 0 {
        vec![
            (0..(start_pos + 1) / ratio)
                .map(|i| (i + offset) as i64)
                .collect(),
        ]
    } else {
        // `seqlen / ratio` is evaluated here rather than inside the row, so `ratio == 0`
        // divides by zero even at `seqlen == 0`, where the nested spelling returned `vec![]`.
        // `ratio == 0` already panicked for every other `seqlen` — on `(t + 1) / ratio` — so
        // it was never a supported input; only the empty case moved.
        prefill_rows(seqlen, seqlen / ratio, |t, c| {
            if c >= (t + 1) / ratio {
                -1
            } else {
                (c + offset) as i64
            }
        })
    }
}

/// The tensors one `sparse_attn` call reads, in the shapes `Attention.forward` hands them.
///
/// One parameter because they are meaningless apart: `topk` indexes `kv`, `q` and `sink`
/// are both per-head, and `m` is `topk`'s own row count. Split into six positional
/// arguments, a swapped `q`/`kv` or a stale `scale` still type-checks.
struct SparseAttnIn<'a> {
    q: &'a [f32],
    kv: &'a [f32],
    sink: &'a [f32],
    topk: &'a [Vec<i64>],
    /// Query rows — the prompt length at prefill, 1 at decode.
    m: usize,
    scale: f32,
}

impl SparseAttnIn<'_> {
    /// One head's logits over its selected slots, and the running max the online softmax
    /// subtracts. A masked slot (`-1`) keeps its place as `-inf` so the caller can zip the
    /// list against `idxs` — and contributes nothing to the max, which is what makes a
    /// fully-masked query detectable as a non-finite one.
    fn logits(&self, qh: &[f32], idxs: &[i64]) -> (Vec<f32>, f32) {
        let d = qh.len();
        let mut logits = Vec::with_capacity(idxs.len());
        let mut mx = f32::NEG_INFINITY;
        for &ix in idxs {
            if ix < 0 {
                logits.push(f32::NEG_INFINITY);
                continue;
            }
            let k = &self.kv[ix as usize * d..(ix as usize + 1) * d];
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += qh[i] * k[i];
            }
            let l = acc * self.scale;
            mx = mx.max(l);
            logits.push(l);
        }
        (logits, mx)
    }
}

impl Oracle {
    /// `kernel.py::sparse_attn`, as mathematics rather than as a tiling.
    ///
    /// `attn_sink` enters the softmax DENOMINATOR only: the kernel adds
    /// `exp(attn_sink[h] - running_max)` to `sum_exp` after the last block and never adds a
    /// matching term to `acc_o`. It is therefore a learned per-head leak of probability
    /// mass, not an extra key. Note that "sink as a real key with a zero value vector" is
    /// exactly this and NOT a defect, which is why no variant models it.
    fn sparse_attn(&self, a: &SparseAttnIn) -> Vec<f32> {
        let (h, d) = (self.cfg.n_heads, self.cfg.head_dim);
        let mut o = vec![0.0f32; a.m * h * d];
        for t in 0..a.m {
            for hh in 0..h {
                let dst = (t * h + hh) * d;
                o[dst..dst + d].copy_from_slice(&self.attn_head(a, t, hh));
            }
        }
        // `sparse_attn` writes a bf16 tensor.
        self.round_bf16(&mut o);
        o
    }

    /// One `(query row, head)` cell of `sparse_attn`: the online softmax over that head's
    /// selected slots, returning the `head_dim`-wide output row.
    fn attn_head(&self, a: &SparseAttnIn, t: usize, hh: usize) -> Vec<f32> {
        let (h, d) = (self.cfg.n_heads, self.cfg.head_dim);
        let idxs = &a.topk[t];
        let qh = &a.q[(t * h + hh) * d..(t * h + hh + 1) * d];
        let (logits, mx) = a.logits(qh, idxs);
        // A fully-masked query would divide by a sum of `exp(-inf)` and write NaN
        // goldens. Loud, and `assert` because release builds skip debug assertions.
        assert!(mx.is_finite(), "query {t} head {hh} attends to nothing");
        let mut sum = 0.0f32;
        let mut acc = vec![0.0f32; d];
        for (&ix, &l) in idxs.iter().zip(&logits) {
            if ix < 0 {
                continue;
            }
            let e = (l - mx).exp();
            sum += e;
            let k = &a.kv[ix as usize * d..(ix as usize + 1) * d];
            for i in 0..d {
                acc[i] += e * k[i];
            }
        }
        sum += self.sink_mass(a.sink[hh], mx);
        acc.iter().map(|v| v / sum).collect()
    }

    /// The probability mass `attn_sink` leaks out of one head's softmax denominator.
    ///
    /// `SkipAttnSink` returns 0.0 rather than skipping the add, which is bit-identical
    /// here: the slot achieving `mx` contributes `exp(0) = 1.0`, so `sum >= 1.0` whenever
    /// the assert above passed and `sum + 0.0 == sum` exactly.
    fn sink_mass(&self, sink_h: f32, mx: f32) -> f32 {
        match self.defect {
            Defect::SkipAttnSink => 0.0,
            Defect::AttnSinkNotMaxShifted => sink_h.exp(),
            _ => (sink_h - mx).exp(),
        }
    }

    /// `Attention.forward` (model.py:490-548), as its six stages in the reference's order.
    ///
    /// The ORDER is the load-bearing part and is the reason the ring write and the
    /// compressor are two calls rather than one: the reference writes the window entry
    /// first in both phases (model.py:523-537), and a port that compresses first is wrong
    /// only at the block boundary.
    pub(super) fn attention(&self, step: &LayerCtx, sk: &mut Sinks, x: &[f32]) -> Vec<f32> {
        let LayerCtx { lw, s, .. } = *step;
        let tag = step.tag();
        let (d, nh) = (self.cfg.head_dim, self.cfg.n_heads);

        let (qr, q) = self.attn_q(step, x);
        sk.cap.push(&format!("{tag}.q"), &[s, nh, d], q.clone());
        let kv = self.attn_kv(step, x);
        sk.cap.push(&format!("{tag}.kv_entry"), &[s, d], kv.clone());
        let topk = self.attn_select(step, sk, SelectIn { x, qr: &qr });

        self.attn_ring_write(step, sk, &kv);
        let full = self.attn_kv_view(step, sk, KvIn { x, kv: &kv });
        let mut o = self.sparse_attn(&SparseAttnIn {
            q: &q,
            kv: &full,
            sink: &lw.attn_sink,
            topk: &topk,
            m: s,
            scale: (d as f32).powf(-0.5),
        });

        // Captured on BOTH sides of the de-rotation on purpose: without the pre-image, a
        // de-rotation defect has no golden that must stay identical and the check loses its
        // silent half.
        sk.cap
            .push(&format!("{tag}.attn_core_out"), &[s, nh, d], o.clone());
        self.attn_derotate(step, &mut o);
        sk.cap
            .push(&format!("{tag}.attn_derot"), &[s, nh, d], o.clone());
        self.attn_out_proj(step, &o)
    }

    /// The query path (model.py:497-505): `wq_a` → `q_norm` → `wq_b`, then the QK-norm and
    /// RoPE. Returns BOTH the LoRA-rank `qr`, which the indexer consumes, and the per-head
    /// `q` — they are one computation and the indexer taking a re-derived `qr` would be a
    /// second place for the projection to drift.
    fn attn_q(&self, step: &LayerCtx, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (lw, layer, s, start_pos) = (step.lw, step.layer, step.s, step.start_pos);
        let c = &self.cfg;
        let (rd, d, nh) = (c.rope_head_dim, c.head_dim, c.n_heads);
        let freqs = self.freqs(layer);
        let mut qr = self.linear(x, s, &lw.wq_a);
        self.round_bf16(&mut qr);
        self.rmsnorm(&mut qr, c.q_lora_rank, &lw.q_norm);
        let mut q = self.linear(&qr, s, &lw.wq_b);
        self.round_bf16(&mut q);
        // `Defect::QkNormAfterRope` is a SWAP and nothing else, so the rope loop is written
        // once and only the `qk_norm` call moves across it. Two transcriptions of the loop
        // are two places for the defect arm to stop being the clean arm's mirror image,
        // which would make the A/B measure something other than the swap.
        let after = self.defect == Defect::QkNormAfterRope;
        if !after {
            self.qk_norm(&mut q, d, &lw.q_norm);
        }
        for i in 0..s * nh {
            let row = &mut q[i * d..(i + 1) * d];
            self.rope_row(row, rd, (start_pos + i / nh, freqs), false);
        }
        if after {
            self.qk_norm(&mut q, d, &lw.q_norm);
        }
        (qr, q)
    }

    /// The kv path (model.py:507-513): `wkv` → `kv_norm` → RoPE → the PARTIAL fp8
    /// `act_quant` that leaves the positional dims at bf16.
    fn attn_kv(&self, step: &LayerCtx, x: &[f32]) -> Vec<f32> {
        let (lw, layer, s, start_pos) = (step.lw, step.layer, step.s, step.start_pos);
        let (rd, d) = (self.cfg.rope_head_dim, self.cfg.head_dim);
        let freqs = self.freqs(layer);
        let mut kv = self.linear(x, s, &lw.wkv);
        self.round_bf16(&mut kv);
        self.rmsnorm(&mut kv, d, &lw.kv_norm);
        for t in 0..s {
            let row = &mut kv[t * d..(t + 1) * d];
            self.rope_row(row, rd, (start_pos + t, freqs), false);
            self.kv_act_quant(row, d, rd);
        }
        kv
    }

    /// The slots each query may attend to: the sliding window always, plus the compressed
    /// selection on a compressed layer. Records `.compress_idxs`.
    fn attn_select(&self, step: &LayerCtx, sk: &mut Sinks, act: SelectIn) -> Vec<Vec<i64>> {
        let (layer, s) = (step.layer, step.s);
        let mut topk = window_topk(self.cfg.window_size, s, step.start_pos);
        // `window_topk` yields one row per query: `s` at prefill, and exactly 1 at decode,
        // where `s` is also 1. So the query count is `s` and there is no second extent --
        // and this is what ENFORCES the bsz=1 scope cut rather than merely stating it.
        assert_eq!(topk.len(), s, "query count and row count disagree");
        if self.cfg.compress_ratio(layer) != 0 {
            let extra = self.attn_compress_idxs(step, sk, act);
            let flat: Vec<i64> = extra.iter().flatten().copied().collect();
            let cols = extra.first().map_or(0, Vec::len);
            let tag = format!("{}.compress_idxs", step.tag());
            sk.cap.push_i(&tag, &[extra.len(), cols], flat);
            for (row, e) in topk.iter_mut().zip(extra) {
                row.extend(e);
            }
        }
        topk
    }

    /// The compressed half of the selection: the indexer where the layer has one, and the
    /// arithmetic `get_compress_topk_idxs` where it does not. `.indexer_scores` is recorded
    /// here rather than by the caller because the score matrix exists on this branch only.
    fn attn_compress_idxs(&self, step: &LayerCtx, sk: &mut Sinks, act: SelectIn) -> Vec<Vec<i64>> {
        let (lw, layer, s, start_pos) = (step.lw, step.layer, step.s, step.start_pos);
        // At prefill the compressed rows are appended to THIS call's kv at index `s`; at
        // decode they follow the whole ring, so the offset is the window size.
        let offset = if start_pos == 0 {
            s
        } else {
            self.cfg.window_size
        };
        let Some((iw, ics)) = lw.indexer.as_ref().zip(sk.st.idx_comp.as_mut()) else {
            return compress_topk(self.cfg.compress_ratio(layer), s, start_pos, offset);
        };
        let (freqs, cnt) = (self.freqs(layer), &mut sk.cap.counters);
        let mut scores = Vec::new();
        let e = self.indexer(
            step,
            iw,
            ics,
            act.x,
            act.qr,
            offset,
            freqs,
            cnt,
            &mut scores,
        );
        let rows = if scores.is_empty() { 0 } else { s };
        let cols = scores.len().checked_div(rows).unwrap_or(0);
        let tag = format!("{}.indexer_scores", step.tag());
        sk.cap.push(&tag, &[rows, cols], scores);
        e
    }

    /// The sliding-window ring write, which differs by phase: at decode one slot, at
    /// prefill the LAST `win` positions rotated so slot `t % win` holds position `t`
    /// (model.py:526-528).
    fn attn_ring_write(&self, step: &LayerCtx, sk: &mut Sinks, kv: &[f32]) {
        let LayerCtx { s, start_pos, .. } = *step;
        let (win, d) = (self.cfg.window_size, self.cfg.head_dim);
        if start_pos != 0 {
            let slot = start_pos % win;
            sk.st.win_cache[slot * d..(slot + 1) * d].copy_from_slice(&kv[..d]);
            return;
        }
        if s <= win {
            sk.st.win_cache[..s * d].copy_from_slice(&kv[..s * d]);
        } else if self.defect == Defect::PrefillRingWritesFirstWindow {
            sk.st.win_cache[..win * d].copy_from_slice(&kv[..win * d]);
        } else {
            // slot (t % win) holds position t, for the last `win` positions.
            for t in (s - win)..s {
                let slot = t % win;
                sk.st.win_cache[slot * d..(slot + 1) * d].copy_from_slice(&kv[t * d..(t + 1) * d]);
            }
        }
        sk.cap.counters.prefill_evicted = s.saturating_sub(win);
    }

    /// Runs the attention's compressor — recording `.compressed` — and returns what the
    /// attention then reads. At prefill that is the whole prompt's KV with THIS call's
    /// compressed rows appended at index `s` (which is why the selection offset was `s`);
    /// at decode it is the ring followed by the whole compressed region, which is the
    /// reference's single `kv_cache` buffer split in two here.
    fn attn_kv_view(&self, step: &LayerCtx, sk: &mut Sinks, act: KvIn) -> Vec<f32> {
        let (lw, layer, s, start_pos) = (step.lw, step.layer, step.s, step.start_pos);
        let d = self.cfg.head_dim;
        let ratio = self.cfg.compress_ratio(layer);
        let freqs = self.freqs(layer);
        let compressed = match (ratio, &lw.compressor, sk.st.comp.as_mut()) {
            (0, _, _) => None,
            (_, Some(cw), Some(cs)) => {
                let cnt = &mut sk.cap.counters;
                self.compressor(cw, cs, act.x, s, start_pos, freqs, cnt)
            }
            _ => None,
        };
        if let Some(z) = &compressed {
            let tag = format!("{}.compressed", step.tag());
            sk.cap.push(&tag, &[z.len() / d, d], z.clone());
        }
        if start_pos == 0 {
            let mut f = act.kv.to_vec();
            if let Some(z) = compressed {
                f.extend(z);
            }
            return f;
        }
        let mut f = sk.st.win_cache.clone();
        if let Some(cs) = sk.st.comp.as_ref() {
            f.extend_from_slice(&cs.cache);
        }
        f
    }

    /// `apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)` (model.py:539), per head
    /// and in place.
    fn attn_derotate(&self, step: &LayerCtx, o: &mut [f32]) {
        if self.defect == Defect::SkipOutputDerotation {
            return;
        }
        let (layer, s, start_pos) = (step.layer, step.s, step.start_pos);
        let c = &self.cfg;
        let (rd, d, nh) = (c.rope_head_dim, c.head_dim, c.n_heads);
        let freqs = self.freqs(layer);
        let inverse = self.defect != Defect::OutputDerotationForward;
        for i in 0..s * nh {
            let row = &mut o[i * d..(i + 1) * d];
            self.rope_row(row, rd, (start_pos + i / nh, freqs), inverse);
        }
    }

    /// The grouped output projection (model.py:541-547): regroup into `o_groups`, the raw
    /// `wo_a` einsum, then `wo_b`. `wo_a` goes through `wrow` rather than `linear` because
    /// `Attention.forward` consumes it in an einsum, so there is no activation
    /// quantization on this one.
    fn attn_out_proj(&self, step: &LayerCtx, o: &[f32]) -> Vec<f32> {
        let LayerCtx { lw, s, .. } = *step;
        let c = &self.cfg;
        let (g, r) = (c.o_groups, c.o_lora_rank);
        let gd = c.n_heads * c.head_dim / g;
        let og = self.o_groups_gather(o, s);
        let mut y = vec![0.0f32; s * g * r];
        let mut wr = Vec::with_capacity(gd);
        for gi in 0..g {
            for ri in 0..r {
                self.wrow(&lw.wo_a, gi * r + ri, &mut wr);
                for t in 0..s {
                    let ov = &og[(t * g + gi) * gd..(t * g + gi + 1) * gd];
                    y[(t * g + gi) * r + ri] = ov.iter().zip(&wr).map(|(a, b)| a * b).sum::<f32>();
                }
            }
        }
        self.round_bf16(&mut y);
        let mut out = self.linear(&y, s, &lw.wo_b);
        self.round_bf16(&mut out);
        out
    }

    /// `o.view(b, s, n_groups, -1)` — the `[s, g, gd]` regrouping the `wo_a` einsum reads.
    fn o_groups_gather(&self, o: &[f32], s: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (g, d, nh) = (c.o_groups, c.head_dim, c.n_heads);
        let gd = nh * d / g;
        let mut og = vec![0.0f32; s * g * gd];
        for t in 0..s {
            for gi in 0..g {
                for e in 0..gd {
                    let (head, dim_i) = self.o_group_src(gi, e);
                    og[(t * g + gi) * gd + e] = o[(t * nh + head) * d + dim_i];
                }
            }
        }
        og
    }

    /// Which `(head, head_dim)` element of `o` group `gi`'s slot `e` reads.
    ///
    /// The reference's flattening is head-major, so group `gi` is the contiguous run of
    /// heads `[gi*hpg, (gi+1)*hpg)`, each contributing its whole `head_dim`. Both
    /// `WoGroups*` defects live entirely in this mapping: they are permutations of the same
    /// 4096 numbers into the same number of dot products, so magnitudes, norms and every
    /// summary statistic survive them and only the golden does not.
    fn o_group_src(&self, gi: usize, e: usize) -> (usize, usize) {
        let c = &self.cfg;
        let (g, d) = (c.o_groups, c.head_dim);
        let hpg = c.n_heads / g;
        match self.defect {
            Defect::WoGroupsSplitHeadDim => (e / (d / g), gi * (d / g) + e % (d / g)),
            Defect::WoGroupsInterleaved => (gi + (e / d) * g, e % d),
            _ => (gi * hpg + e / d, e % d),
        }
    }
}

/// The two mutable sinks a block body threads through every stage: the layer's caches and
/// the golden recorder.
///
/// One parameter because they always travel together — no stage below is handed one without
/// the other — and because a stage that took both separately plus its own inputs is exactly
/// the argument pile-up that made `attention` a Brain Method.
/// The two mutable sinks a layer writes: its ring state and the capture. Public since
/// 2026-08-15 — `Oracle::run_layer` takes one, so the pair callers used to thread as two
/// loose `&mut`s travels as the value this path already used internally.
pub struct Sinks<'a> {
    pub st: &'a mut LayerRings,
    pub cap: &'a mut Capture,
}

/// The two activations the compressed selection reads: the block's normalised input `x`,
/// which the indexer's own compressor and `weights_proj` consume, and the LoRA-rank query
/// `qr`, which its `wq_b` consumes. Produced together by the q path, so passed together.
#[derive(Clone, Copy)]
struct SelectIn<'a> {
    x: &'a [f32],
    qr: &'a [f32],
}

/// The two activations the KV view is built from: the block's normalised input `x`, which
/// the attention compressor pools, and this call's own KV entries.
#[derive(Clone, Copy)]
struct KvIn<'a> {
    x: &'a [f32],
    kv: &'a [f32],
}
