//! **The QSA lightning indexer's block-key pipeline, scored against the S2 prefill window** —
//! one new kernel (`index_pool_norm_f32`) and two REUSES proven rather than assumed.
//!
//! The reference's pipeline is four steps (`qwen-architecture.md` §3, MOD:672-701): mean-pool
//! `compress_ratio` cached raw keys per block **in fp32**, RMSNorm each pooled row, partial-RoPE
//! it at the block's FIRST token position, and score every query against every block as
//! `I_ib = sum_h ReLU(<q_i^h, kbar_b>) / sqrt(d)`. This suite drives all four and scores the two
//! the anchor captures: `pooled_keys` (the final `kbar`, post-RoPE — see below) and `scores`.
//!
//! # What is new and what is reused, and why
//!
//! * **NEW `index_pool_norm_f32`.** `index_pool_push` folds one token into a block pool whose
//!   width is the compile-time `MISA_BLOCK` (1024); `compress_ratio` here is **4**, so it cannot
//!   express the geometry at all. The kernel fuses the pool with the RMSNorm for the reason the
//!   measurement below forces: the norm CANCELS the pool's scale, so there is no scoreable
//!   boundary between them and a second pass over the keys would buy nothing.
//! * **REUSE `launch_rope_split_half`.** Its pairing is `(x[j], x[j + seg/2])` over the FIRST
//!   `seg` elements of each `stride`-strided row, with `inv_freq = theta^(-2j/seg)` — which is
//!   `apply_rotary_pos_emb` over `q[..., :rotary_dim]` with `rotate_half` exactly (MOD:566-600).
//!   Measured, not read: [`the_split_half_rope_launcher_is_qwens_partial_rope`].
//! * **REUSE `launch_index_score_blocks`.** Its `score[s][b] = sum_h w[s][h] · ReLU(<q, kv_b>)`
//!   becomes qwen's formula with `w` filled with `1/sqrt(d)` — a positive uniform scale
//!   distributes over the sum, and `w` is not an invented tensor but the scale MOD:693 applies
//!   after it. Trap T6 says there are no learned per-head weights and there is no
//!   `indexer_head_weights` tensor to plant; what this suite shows instead is that the existing
//!   launcher, given that `w`, reproduces the reference's own `scores`.
//!
//! # `pooled_keys` is the POST-RoPE key, and that had to be measured
//!
//! The capture order (`q_layernorm`, thirteen `k_layernorm`s, `pooled_keys`, `scores`) reads as
//! though `pooled_keys` were the pool's output. It is not: the mean of the raw key rows misses it
//! by **8.842e-1**, while `rope(k_layernorm#13, block_starts)` reproduces it to **6.177e-8**. So
//! `pooled_keys` is the finished `kbar` the score consumes, and the pool's own output is captured
//! nowhere — which is why the ladder below is scored through a RECOVERED norm weight.
//!
//! # The recovered weight, and exactly what it can and cannot say
//!
//! No weight tensor is in the golden. The norm's `[d]` weight is recovered from block 0 of the
//! thirteenth call and used to predict every other row of the ladder — 1 row in, 3 distinct block
//! rows out across 13 calls. That is a real asymmetry and it is a NARROW one, so the two things
//! it cannot see are stated here rather than left to be assumed:
//!
//! * **the norm's offset convention.** Recovering `w` under the bare `w . xhat` form and
//!   predicting with `(1 + w)` cancels exactly, so this fixture is blind to it. It is settled
//!   elsewhere, on real bytes:
//!   `crates/artifact/tests/qwen_names.rs::the_gdn_output_norm_weight_is_ones_centred`.
//! * **mean versus sum in the pool.** `RMSNorm(c·x) = RMSNorm(x)` for any `c > 0`, so a pool that
//!   SUMS instead of averaging is cancelled by the very next operation. Measured on the vendored
//!   bytes: summing moves the ladder by **4.843e-5**, and that residue is the epsilon term
//!   becoming 16x smaller relative to a 4x larger row — not a scale signal. A port must therefore
//!   read MOD:681's `.mean(dim=1)`; this fixture cannot tell, and [`POOL_SUM_IS_CANCELLED`]
//!   records the number so nobody re-derives it as a defect.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod kernel_qwen_harness;

use kernel_qwen_harness as h;

// See `kernel_qwen_gdn_recurrent.rs` for the `duplicate_mod` argument: `common/mod.rs` does not
// compile deviceless, the harness reaches `common/scoring.rs` by `#[path]`, and this declaration
// loads that same ONE file a second time through the umbrella under `rocm`.
#[cfg(feature = "rocm")]
#[allow(clippy::duplicate_mod)]
mod common;

/// The QSA layer the prefill window captures. Index 3 is the first `full_attention` entry in the
/// checkpoint's own `layer_types`, which the reference ALIASES to `qwen_sparse_attention` before
/// validating (CFG:180-184) — so it is an indexer layer, which is trap T19.
const QSA_LAYER: usize = 3;

/// The worst the f64 host pipeline shows against the anchor: the pool-and-norm ladder, the RoPE
/// composition and the block scores. Measured 2026-09-01 over all thirteen ladder calls and the
/// last query's four block scores; the three readings are 8.7523e-8, 6.1766e-8 and 7.6405e-8, so
/// this is the ladder's.
const HOST_WORST: f32 = 8.7523e-8;

/// Defect separations, **minima over the thirteen ladder rungs**, measured 2026-09-01, with the
/// maxima beside them so the spread is visible: stride 6.0676e-1..6.2726e-1, first-row
/// 1.1043e0..1.3693e0, norm-dropped 8.6482e-1..8.9855e-1, norm-over-the-sum 7.9587e-1 flat.
///
/// Minima and not maxima: the bar a variant must clear is the WEAKEST rung's, and quoting the
/// strongest would let a form that is nearly invisible at one block count pass on another's
/// number. `first_row_only` spans 1.24x across the rungs and was the one that caught this — the
/// first draft of these constants recorded maxima and the test reddened at 1.10428e0 against a
/// 1.3693e0 floor.
const POOL_STRIDE_OFF_BY_ONE: f32 = 6.0676e-1;
const POOL_FIRST_ROW_ONLY: f32 = 1.1043e0;
const NORM_DROPPED: f32 = 8.6482e-1;
const NORM_OVER_THE_SUM: f32 = 7.9587e-1;

/// **Not a defect this fixture can price** — recorded so it is not re-derived as one.
///
/// A pool that sums instead of averaging moves the ladder by this much, which is 1.9x the bar the
/// fixture enforces and 0.4x the `indexer` tolerance. It is the epsilon residue of a 4x larger
/// row, not the 4x itself: the RMSNorm divides the scale straight back out. The three epsilon
/// readings sit in the same band and are equally unpriceable here — eps 0 at 3.6167e-5..5.1665e-5,
/// eps outside the root at 2.7662e-5..4.1663e-5, eps 1e-5 at 3.2533e-4..4.6491e-4 (the only one
/// above the tolerance, and only at its strongest rung). So the
/// pool's reduction and the norm's epsilon are both pinned by READING the reference (MOD:681's
/// `.mean(dim=1)`, `rms_norm_eps` 1e-6), which is the same conclusion K3's anchor reached for the
/// MLA eps and for the same arithmetic reason.
const POOL_SUM_IS_CANCELLED: f32 = 4.843e-5;

/// The score-side separations, measured on the last query's four block scores.
const SCORE_RELU_OFF: f32 = 6.867e-1;
const SCORE_UNSCALED: f32 = 3.899e0;
const ROPE_AT_BLOCK_END: f32 = 1.774e-1;
const ROPE_AT_POSITION_ZERO: f32 = 2.134e-1;

/// The indexer's prefill window, at the widths the launchers take.
struct Idx {
    /// `indexer_n_heads`.
    heads: usize,
    /// `indexer_head_dim`.
    hd: usize,
    /// `indexer_compress_ratio` — TOKENS per block, not blocks per anything (trap T8).
    ratio: usize,
    /// `rotary_dim`, the partial-RoPE width inside `hd`.
    rd: usize,
    theta: f64,
    seq: usize,
    /// `[seq][hd]` — the ONE shared indexer key head, raw and un-normed, as the cache holds it.
    keys: Vec<f32>,
    /// `[seq][heads][hd]` — post-RMSNorm, pre-RoPE query rows.
    q_normed: Vec<f32>,
    /// `[n_blocks]` — the FIRST token position of each complete block.
    starts: Vec<i64>,
    /// The recovered `[hd]` norm weight. See the header for what it can and cannot say.
    weight: Vec<f32>,
    /// `[n_blocks][hd]` — the finished, RoPE'd block keys.
    want_kbar: Vec<f32>,
    /// `[n_blocks]` — the last query's block scores.
    want_scores: Vec<f32>,
}

impl Idx {
    /// The complete-block count visible to query `q`: `(q + 1) / ratio`, floored (MOD:672).
    fn blocks_at(&self, q: usize) -> usize {
        (q + 1) / self.ratio
    }
    fn n_blocks(&self) -> usize {
        self.starts.len()
    }
}

fn rms_over_mean(row: &[f64], eps: f64) -> f64 {
    let m: f64 = row.iter().map(|x| x * x).sum::<f64>() / row.len() as f64;
    1.0 / (m + eps).sqrt()
}

/// Which arithmetic the pool-and-norm oracle performs. The variants ARE the defects.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pool {
    Reference,
    /// Blocks grouped one token late — `[b·r+1, (b+1)·r+1)`. Every block still reads `r` real
    /// cached rows, so nothing structural sees it.
    StrideOffByOne,
    /// The block's FIRST key taken as the block key. Not a scale, so unlike the sum it survives
    /// the norm.
    FirstRowOnly,
    /// The norm skipped entirely, the weight applied to the pooled row.
    NormDropped,
    /// The norm taken over the SUM of squares (an L2 norm) rather than their MEAN — a real
    /// `sqrt(d)` factor, which is a scale and would cancel if anything followed it; nothing does.
    NormOverTheSum,
}

/// Mean-pool and RMSNorm the first `n_blocks` complete blocks.
fn pool_norm(c: &Idx, n_blocks: usize, form: Pool) -> Vec<f32> {
    let mut out = vec![0.0f32; n_blocks * c.hd];
    for b in 0..n_blocks {
        let base = match form {
            Pool::StrideOffByOne => (b * c.ratio + 1).min(c.seq - c.ratio),
            _ => b * c.ratio,
        };
        let rows = if form == Pool::FirstRowOnly {
            1
        } else {
            c.ratio
        };
        let pooled: Vec<f64> = (0..c.hd)
            .map(|j| {
                let s: f64 = (0..rows)
                    .map(|r| f64::from(c.keys[(base + r) * c.hd + j]))
                    .sum();
                s / rows as f64
            })
            .collect();
        let inv = match form {
            Pool::NormDropped => 1.0,
            Pool::NormOverTheSum => {
                let s: f64 = pooled.iter().map(|x| x * x).sum();
                1.0 / (s + 1e-6).sqrt()
            }
            _ => rms_over_mean(&pooled, 1e-6),
        };
        for j in 0..c.hd {
            out[b * c.hd + j] = (pooled[j] * inv * f64::from(c.weight[j])) as f32;
        }
    }
    out
}

/// Split-half partial RoPE over the first `rd` dims of one row, at `pos`. The pair is
/// `(j, j + rd/2)` and the frequency is `theta^(-2j/rd)` — the convention
/// [`rivoli_backend::hip::launch_rope_split_half`] implements and MOD:566-600 specifies.
fn rope_row(row: &mut [f32], rd: usize, pos: usize, theta: f64) {
    let half = rd / 2;
    for j in 0..half {
        let ang = pos as f64 * theta.powf(-(2.0 * j as f64) / rd as f64);
        let (cs, sn) = (ang.cos(), ang.sin());
        let (a, b) = (f64::from(row[j]), f64::from(row[half + j]));
        row[j] = (a * cs - b * sn) as f32;
        row[half + j] = (b * cs + a * sn) as f32;
    }
}

/// Where a block's RoPE position comes from. `Start` is the reference (MOD:683-688, TR Eq. 13-14:
/// pooling before roping "avoids averaging token representations with different rotary phases").
#[derive(Clone, Copy, PartialEq, Eq)]
enum At {
    Start,
    End,
    Zero,
}

fn rope_blocks(c: &Idx, kbar: &[f32], at: At) -> Vec<f32> {
    let mut out = kbar.to_vec();
    for b in 0..c.n_blocks() {
        let pos = match at {
            At::Start => c.starts[b] as usize,
            At::End => c.starts[b] as usize + c.ratio - 1,
            At::Zero => 0,
        };
        rope_row(&mut out[b * c.hd..(b + 1) * c.hd], c.rd, pos, c.theta);
    }
    out
}

/// `I_ib = sum_h ReLU(<q_i^h, kbar_b>) / sqrt(hd)` for one query row.
fn block_scores(c: &Idx, q_roped: &[f32], kbar: &[f32], relu: bool, scaled: bool) -> Vec<f32> {
    let scale = if scaled {
        1.0 / (c.hd as f64).sqrt()
    } else {
        1.0
    };
    (0..c.n_blocks())
        .map(|b| {
            let s: f64 = (0..c.heads)
                .map(|hh| {
                    let dot: f64 = (0..c.hd)
                        .map(|j| f64::from(q_roped[hh * c.hd + j]) * f64::from(kbar[b * c.hd + j]))
                        .sum();
                    if relu { dot.max(0.0) } else { dot }
                })
                .sum();
            (s * scale) as f32
        })
        .collect()
}

/// The last query's RoPE'd indexer query rows, `[heads][hd]`.
fn last_query_roped(c: &Idx) -> Vec<f32> {
    let q = c.seq - 1;
    let mut rows = c.q_normed[q * c.heads * c.hd..(q + 1) * c.heads * c.hd].to_vec();
    for hh in 0..c.heads {
        rope_row(&mut rows[hh * c.hd..(hh + 1) * c.hd], c.rd, q, c.theta);
    }
    rows
}

/// The window, assembled. Widths from the golden's own recorded maps and config, never literals.
fn window() -> Idx {
    let d = h::prefill_draw();
    let heads = d.c("indexer_n_heads");
    let hd = d.w("indexer_head_dim");
    let ratio = d.c("indexer_compress_ratio");
    let rd = d.w("rotary_dim");
    let seq = d.decode_pos();
    let a = format!("model.layers.{QSA_LAYER}.self_attn.indexer");
    // `index_qk_proj` is ONE matrix, split `[heads·hd | kv_heads·hd]` with the QUERY heads FIRST
    // (MOD:645-650). The k half is the single shared key head; reading the split the other way
    // round addresses real floats and is trap T3's shape one operator over.
    let qk = d.flat(&format!("{a}.index_qk_proj"), seq * (heads + 1) * hd);
    let stride = (heads + 1) * hd;
    let keys: Vec<f32> = (0..seq)
        .flat_map(|t| qk[t * stride + heads * hd..(t + 1) * stride].to_vec())
        .collect();
    let starts = d.i(&format!("{a}.block_starts"));
    let nb = starts.len();
    let want_kbar = d.flat(&format!("{a}.pooled_keys"), nb * hd);
    // The norm weight, recovered from block 0 of the LAST ladder call. Recovered from ONE row and
    // then used to predict every other, which is where the power is — see the header for the two
    // things it cannot see.
    // The LAST rung of the ladder — the only call whose block count is `n_blocks`. Its index is
    // derived, not written: the norm runs once per query that has at least one complete block, so
    // there are `seq - (ratio - 1)` calls and the last is the one query 15 made.
    let last_call = seq - (ratio - 1);
    let ladder_last = d.flat(&format!("{a}.k_layernorm#{last_call}"), nb * hd);
    let block0: Vec<f64> = (0..hd)
        .map(|j| (0..ratio).map(|r| f64::from(keys[r * hd + j])).sum::<f64>() / ratio as f64)
        .collect();
    let inv = rms_over_mean(&block0, 1e-6);
    let weight: Vec<f32> = (0..hd)
        .map(|j| (f64::from(ladder_last[j]) / (block0[j] * inv)) as f32)
        .collect();
    Idx {
        heads,
        hd,
        ratio,
        rd,
        theta: d.rope_theta(),
        seq,
        keys,
        q_normed: d.flat(&format!("{a}.q_layernorm"), seq * heads * hd),
        starts,
        weight,
        want_kbar,
        want_scores: d.flat(&format!("{a}.scores"), nb),
    }
}

/// One ladder rung: the k-th `k_layernorm` capture, the query it belongs to, and its block count.
///
/// The thirteen calls are queries 3..15 — queries 0, 1 and 2 have no complete block and never
/// reach the norm, which is why 8 rungs are dense while the dense TOTAL is 11.
fn ladder(c: &Idx, d: &h::Draw) -> Vec<(usize, usize, Vec<f32>)> {
    let a = format!("model.layers.{QSA_LAYER}.self_attn.indexer");
    let calls = c.seq - (c.ratio - 1);
    (1..=calls)
        .map(|call| {
            let q = call + c.ratio - 2;
            let nb = c.blocks_at(q);
            let name = if call == 1 {
                format!("{a}.k_layernorm")
            } else {
                format!("{a}.k_layernorm#{call}")
            };
            (q, nb, d.flat(&name, nb * c.hd))
        })
        .collect()
}

// ── deviceless ──────────────────────────────────────────────────────────────────────────────

/// The thirteen `k_layernorm` captures have exactly the block counts `(query + 1) / ratio`
/// predicts, and the window holds both selection regimes.
///
/// **This is the fixture's own claim about what it covers**, and it is the reason the prefill
/// window exists at all: with `indexer_budget` 8 and `ratio` 4 the top-k is TOTAL whenever the
/// complete-block count is at most `block_topk` = 2, so queries 3..10 are DENSE by construction
/// and 11..15 are selective (MOD:695). At the real budget that boundary sits at 2051 cached
/// tokens, which is why trap T19 is invisible to every gate this port plans.
#[test]
fn the_ladder_carries_both_selection_regimes() {
    let c = window();
    let d = h::prefill_draw();
    let rungs = ladder(&c, &d);
    assert_eq!(rungs.len(), 13, "queries 3..15 reach the block norm");
    let topk = d.w("block_topk");
    assert_eq!(
        topk,
        d.c("indexer_budget") / c.ratio,
        "block_topk = budget / ratio"
    );
    let counts: Vec<usize> = rungs.iter().map(|(_, nb, _)| *nb).collect();
    assert_eq!(counts, vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4]);
    let dense = counts.iter().filter(|nb| **nb <= topk).count();
    let selective = counts.len() - dense;
    assert_eq!(
        (dense, selective),
        (8, 5),
        "8 dense rungs and 5 selective ones; the three queries with no complete block never \
         reach the norm, which is what makes 8 + 5 + 3 = 16"
    );
    assert!(
        selective > 0,
        "a window entirely below the budget would score the selection path vacuously — the \
         whole reason `indexer_budget` is 8 in the tiny config"
    );
    // The `|visible| mod ratio` tail is ALWAYS included and NEVER scored (MOD:700-701), so the
    // last rung's blocks account for 4·4 = 16 of 16 tokens here and no tail row exists to leak
    // into the pool. Asserted because a kernel that pooled ceil() blocks instead of floor() would
    // read past the cache on every other query.
    assert_eq!(
        counts[12] * c.ratio,
        c.seq,
        "query 15 sees 16 tokens in 4 complete blocks and no tail; a ceil() pool would invent a \
         fifth block out of rows the cache does not have"
    );
}

/// The f64 pool-and-norm reproduces every rung of the ladder from ONE recovered weight.
#[test]
fn the_host_pool_and_norm_reproduces_the_whole_ladder() {
    let c = window();
    let d = h::prefill_draw();
    let b = h::Bar::at("indexer", HOST_WORST);
    for (q, nb, want) in ladder(&c, &d) {
        let got = pool_norm(&c, nb, Pool::Reference);
        b.hold(&format!("q{q}"), "k_layernorm", &got, &want);
    }
}

/// `pooled_keys` is the pool, the norm and the RoPE at the block's FIRST token — and neither of
/// the two other plausible positions reproduces it.
#[test]
fn the_block_keys_are_roped_at_the_blocks_first_token() {
    let c = window();
    let b = h::Bar::at("indexer", HOST_WORST);
    let kbar = pool_norm(&c, c.n_blocks(), Pool::Reference);
    b.hold(
        "prefill",
        "pooled_keys",
        &rope_blocks(&c, &kbar, At::Start),
        &c.want_kbar,
    );
    for (at, name, floor) in [
        (At::End, "block end", ROPE_AT_BLOCK_END),
        (At::Zero, "position zero", ROPE_AT_POSITION_ZERO),
    ] {
        let moved = h::rel(&rope_blocks(&c, &kbar, at), &c.want_kbar);
        b.separates("prefill", name, moved);
        assert!(
            moved >= floor * 0.9,
            "{name}: {moved:e} against the {floor:e} recorded — the variant is not moving what \
             its name says"
        );
    }
}

/// The block scores reproduce, and dropping either the ReLU or the `1/sqrt(hd)` moves them clear.
#[test]
fn the_host_block_scores_reproduce_the_reference() {
    let c = window();
    let b = h::Bar::at("indexer", HOST_WORST);
    let q = last_query_roped(&c);
    b.hold(
        "prefill",
        "scores",
        &block_scores(&c, &q, &c.want_kbar, true, true),
        &c.want_scores,
    );
    for (relu, scaled, name, floor) in [
        (false, true, "ReLU dropped", SCORE_RELU_OFF),
        (true, false, "1/sqrt(hd) dropped", SCORE_UNSCALED),
    ] {
        let moved = h::rel(
            &block_scores(&c, &q, &c.want_kbar, relu, scaled),
            &c.want_scores,
        );
        b.separates("prefill", name, moved);
        assert!(
            moved >= floor * 0.9,
            "{name}: {moved:e} against the {floor:e} recorded"
        );
    }
    // Two blocks can tie at EXACTLY 0.0 — a block whose every raw head score is negative scores
    // zero after the ReLU, and `qwen-anchor-2`'s do. So the claim that matters is not that the
    // four scores are distinct, it is that the cut is clean: rank `block_topk - 1` must sit
    // strictly above rank `block_topk`, or which blocks the budget keeps is decided by the sort
    // and not by the model. `torch.topk`'s tie-break is OPEN and unpriced (anchor.md), and this
    // is the assertion that says this fixture is not standing on it.
    let mut sorted = c.want_scores.clone();
    sorted.sort_by(|x, y| y.total_cmp(x));
    let topk = 2usize;
    assert!(
        sorted[topk - 1] > sorted[topk],
        "the budget cut falls inside a tie ({sorted:?}); which blocks are selected is then the \
         sort's decision, and no comparison against these bytes can see a different tie-break"
    );
}

/// Each pool-and-norm defect moves the ladder clear of the bar, by at least its recorded minimum.
#[test]
fn every_pool_defect_prices_the_difference_it_names() {
    let c = window();
    let d = h::prefill_draw();
    let b = h::Bar::at("indexer", HOST_WORST);
    let rungs = ladder(&c, &d);
    for (form, name, floor) in [
        (
            Pool::StrideOffByOne,
            "blocks grouped one token late",
            POOL_STRIDE_OFF_BY_ONE,
        ),
        (
            Pool::FirstRowOnly,
            "first key taken as the block key",
            POOL_FIRST_ROW_ONLY,
        ),
        (Pool::NormDropped, "block norm dropped", NORM_DROPPED),
        (
            Pool::NormOverTheSum,
            "norm over the sum, not the mean",
            NORM_OVER_THE_SUM,
        ),
    ] {
        let mut weakest = f32::INFINITY;
        for (q, nb, want) in &rungs {
            let moved = h::rel(&pool_norm(&c, *nb, form), want);
            b.separates(&format!("q{q}"), name, moved);
            weakest = weakest.min(moved);
        }
        assert!(
            weakest >= floor * 0.9,
            "{name}: the weakest rung now separates by {weakest:e}, well under the {floor:e} \
             recorded — the variant is landing somewhere other than its name says"
        );
    }
}

/// **The pool's reduction is NOT priceable at this boundary, and the number is the evidence.**
///
/// A gate that asserted otherwise would be asserting a rounding residue. This test asserts the
/// residue is still a residue: summing must move the ladder by something in the epsilon band and
/// NOT by the 4x a scale would give — so if a future re-vendor made it separable, this test says
/// so and the constant's argument has to be rewritten.
#[test]
fn a_summing_pool_is_cancelled_by_the_norm_that_follows_it() {
    let c = window();
    let d = h::prefill_draw();
    let mut worst = 0.0f32;
    for (_, nb, want) in ladder(&c, &d) {
        let summed: Vec<f32> = pool_norm(&c, nb, Pool::Reference);
        // The sum, spelled here rather than as a `Pool` variant, because it is not a defect the
        // variant set should offer: nothing may score against it.
        let scaled: Vec<f32> = {
            let mut v = vec![0.0f32; nb * c.hd];
            for bb in 0..nb {
                let pooled: Vec<f64> = (0..c.hd)
                    .map(|j| {
                        (0..c.ratio)
                            .map(|r| f64::from(c.keys[(bb * c.ratio + r) * c.hd + j]))
                            .sum()
                    })
                    .collect();
                let inv = rms_over_mean(&pooled, 1e-6);
                for j in 0..c.hd {
                    v[bb * c.hd + j] = (pooled[j] * inv * f64::from(c.weight[j])) as f32;
                }
            }
            v
        };
        assert!(
            h::rel(&summed, &want) < 1e-6,
            "the reference arm must still agree"
        );
        worst = worst.max(h::rel(&scaled, &want));
    }
    let tol = h::qwen_tol("indexer");
    assert!(
        worst < tol,
        "summing now moves the ladder by {worst:e}, ABOVE the {tol:e} indexer tolerance — the \
         norm no longer cancels the pool's scale, so this claim and the kernel's comment about \
         reading MOD:681 both need rewriting"
    );
    assert!(
        (worst - POOL_SUM_IS_CANCELLED).abs() < POOL_SUM_IS_CANCELLED,
        "the residue is {worst:e} against the {POOL_SUM_IS_CANCELLED:e} recorded; it is the \
         epsilon term, so a factor-of-two move means the epsilon or the widths changed"
    );
}

// ── on the device ───────────────────────────────────────────────────────────────────────────

/// **The whole block-key pipeline on the device**: the new pool-and-norm, the reused split-half
/// RoPE at each block's first token, the reused block scorer — scored against `pooled_keys` and
/// `scores`, the two tensors the reference actually captured.
///
/// Four launches on one stream, which is what the decode path does: a block key is finished ONCE,
/// when its fourth token arrives, and then cached. That is why the RoPE reuse needs no per-row
/// position array — `launch_rope_split_half` takes one `pos`, and one block per call is exactly
/// the granularity production has. The prefill batch case is a loop here and stays a loop until
/// something measures it as the bottleneck.
///
/// # RED-PROOF PLAN — for the integrator's first device run
///
/// Mutations go in `kernels/indexer.hip`'s `index_pool_norm_f32`; each must redden this test while
/// [`the_host_pool_and_norm_reproduces_the_whole_ladder`] stays green. Magnitudes are the
/// constants above, already shown reachable deviceless:
///
/// * pool from `b·ratio + 1` — ~6.1e-1;
/// * take `keys[b·ratio]` instead of the mean — ~1.1e0;
/// * drop the norm — ~8.6e-1;
/// * normalise over the SUM of squares instead of the mean — 8.0e-1;
/// * **do NOT plant a summing pool** — it moves the result by 3.4e-5, inside the 1.14e-4
///   tolerance, and a plant that cannot redden is not a proof of anything about this kernel.
///
/// On the RoPE stage, roping each block at its LAST token instead of its first moves
/// `pooled_keys` by 1.8e-1 and roping at position 0 by 2.1e-1 — both reachable by changing the
/// `pos` this test passes, so they are proofs of the CALL rather than of the kernel.
#[cfg(feature = "rocm")]
#[test]
fn the_block_key_pipeline_matches_the_anchor() {
    use common::{DeviceBuf, back, dev, f32b, f32v, ok, stream, zeros};
    use rivoli_backend::abi::ScoreDims;
    use rivoli_backend::hip::{
        ScoreBufs, launch_index_pool_norm_f32, launch_index_score_blocks, launch_rope_split_half,
    };

    let c = window();
    let b = h::Bar::at("indexer", HOST_WORST);
    let s = stream();
    let nb = c.n_blocks();
    let cp = |d: &DeviceBuf| d.ptr() as *const f32;

    let keys = dev(&f32b(&c.keys));
    let weight = dev(&f32b(&c.weight));
    let mut kbar = zeros(nb * c.hd * 4);
    let kbar_out = kbar.ptr_mut() as *mut f32;
    // SAFETY: `keys` holds `n_blocks · ratio · hd` f32 (the whole 16-token window), `weight` holds
    // `hd`, `kbar` is a distinct `n_blocks · hd` destination, and all three outlive the stream.
    ok(
        unsafe {
            launch_index_pool_norm_f32(
                cp(&keys),
                cp(&weight),
                nb,
                c.ratio,
                c.hd,
                1e-6,
                kbar_out,
                s.raw(),
            )
        },
        "index_pool_norm_f32",
    );
    // One RoPE launch per block, each at that block's FIRST token position. `count = 1` and
    // `stride = hd` address exactly one row; `seg = rotary_dim` is the partial width, so dims
    // `rotary_dim..hd` pass through untouched — which is the half of trap T16 a fixture with
    // `hd == rotary_dim` could not see.
    for blk in 0..nb {
        let row = unsafe { kbar_out.add(blk * c.hd) };
        // SAFETY: `row` is inside the `n_blocks · hd` buffer above, and the kernel rewrites the
        // first `seg` of the `hd` it is handed, in place.
        ok(
            unsafe { launch_rope_split_half(row, 1, c.hd, c.rd, c.starts[blk] as usize, c.theta) },
            "rope_split_half on a block key",
        );
    }
    b.hold("device", "pooled_keys", &f32v(&back(&kbar)), &c.want_kbar);

    // The query side: the reference's own normed rows, RoPE'd at the query's position in ONE
    // launch — `count = heads`, `stride = hd`, so the four head rows are contiguous and share a
    // position, which is what a decode step has.
    let q = c.seq - 1;
    let mut qrows = dev(&f32b(
        &c.q_normed[q * c.heads * c.hd..(q + 1) * c.heads * c.hd],
    ));
    let q_out = qrows.ptr_mut() as *mut f32;
    // SAFETY: `qrows` holds `heads · hd` f32 and the kernel rotates in place.
    ok(
        unsafe { launch_rope_split_half(q_out, c.heads, c.hd, c.rd, q, c.theta) },
        "rope_split_half on the indexer query",
    );

    // The score REUSE: `w` filled with `1/sqrt(hd)` turns
    // `score[s][b] = sum_h w[s][h]·ReLU(<q,kv_b>)` into MOD:693's
    // `sum_h ReLU(<q,kbar_b>) / sqrt(hd)`. A positive uniform factor distributes over the sum, so
    // the only difference from applying it after is the fp32 rounding order.
    let w = dev(&f32b(&vec![1.0f32 / (c.hd as f32).sqrt(); c.heads]));
    let mut scores = zeros(nb * 4);
    let bufs = ScoreBufs {
        q: q_out as *const f32,
        kv: kbar_out as *const f32,
        w: cp(&w),
        score: scores.ptr_mut() as *mut f32,
    };
    let dims = ScoreDims {
        s: 1,
        n_comp: nb,
        heads: c.heads,
        hd: c.hd,
    };
    // SAFETY: `q` is `1 · heads · hd`, `kv` is `n_comp · hd`, `w` is `1 · heads`, `score` is
    // `1 · n_comp`; all four are distinct live allocations outliving the stream.
    ok(
        unsafe { launch_index_score_blocks(bufs, dims, s.raw()) },
        "index_score_blocks",
    );
    b.hold("device", "scores", &f32v(&back(&scores)), &c.want_scores);
}

/// The new launcher refuses the four argument shapes that mean the caller misread a config field.
#[cfg(feature = "rocm")]
#[test]
fn the_pool_launcher_refuses_geometry_it_cannot_mean() {
    use common::{assert_guard, dev, f32b, zeros};
    use rivoli_backend::hip::launch_index_pool_norm_f32;

    let src = dev(&f32b(&[1.0f32; 256]));
    let mut out = zeros(256 * 4);
    let dst = out.ptr_mut() as *mut f32;
    let p = src.ptr() as *const f32;
    let call = |n_blocks: usize, ratio: usize, hd: usize, eps: f32| {
        // SAFETY: the buffers are large enough for every legal case; the illegal ones are refused
        // before a pointer is read, which is what is under test.
        unsafe {
            launch_index_pool_norm_f32(p, p, n_blocks, ratio, hd, eps, dst, std::ptr::null_mut())
        }
    };
    assert_guard(call(0, 4, 8, 1e-6), Some(1001), "zero blocks");
    assert_guard(
        call(2, 1, 8, 1e-6),
        Some(1004),
        "ratio 1 — the budget read as blocks",
    );
    assert_guard(
        call(1, 4, 2048, 1e-6),
        Some(1002),
        "a block past the launch limit",
    );
    assert_guard(call(1, 4, 8, f32::NAN), Some(1006), "a NaN epsilon");
    assert_guard(call(2, 4, 8, 0.0), None, "eps zero is legal and exact");
}

/// **`launch_rope_split_half` IS Qwen3.8-Flash-Next's partial RoPE**, measured against the
/// reference's own `rope.in`/`rope.out` triple rather than read off the two implementations.
///
/// The QSA attention captures both sides of the rotation at every layer and both draws, which is
/// the only complete weightless triple in the decode goldens. Three things are pinned at once, and
/// each is a trap:
///
/// * the PAIRING is split-half, `(j, j + rotary_dim/2)`, not interleaved `(2j, 2j+1)` — T16, and
///   `mrope_interleaved: true` in the config invites exactly that confusion (it is about the mRoPE
///   SECTIONS, not the pairing);
/// * the rotation covers the FIRST `rotary_dim` of `head_dim` and dims above it pass through —
///   which needs `head_dim > rotary_dim` to be visible at all, and 32 > 8 here;
/// * the POSITION is `seq`, not `seq - 1`: the warm prefill occupies `0..seq`, so the decoded
///   token is at `seq`. Both directions are asserted, because off by one here is a different
///   rotation and not a rounding error.
#[cfg(feature = "rocm")]
#[test]
fn the_split_half_rope_launcher_is_qwens_partial_rope() {
    use common::{back, dev, f32b, f32v, ok};
    use rivoli_backend::hip::launch_rope_split_half;

    // No tolerance row targets the RoPE, so the bar is the `indexer` row it feeds, at the worst
    // this composition measures on the vendored bytes (1.4596e-7 on q, 2.0805e-7 on k at the
    // prefill widths; the decode sites are tighter).
    let b = h::Bar::at("indexer", 2.0805e-7);
    for d in h::decode_draws() {
        let (hd, rd) = (d.w("head_dim"), d.w("rotary_dim"));
        assert!(
            hd > rd,
            "head_dim {hd} must exceed rotary_dim {rd}, or the partial-RoPE claim is vacuous"
        );
        let pos = d.decode_pos();
        for layer in [3usize, 27, 47] {
            for (which, n) in [
                ("q", d.w("num_attention_heads")),
                ("k", d.w("num_key_value_heads")),
            ] {
                let a = format!("model.layers.{layer}.self_attn.rope");
                let want = d.flat(&format!("{a}.out.{which}"), n * hd);
                let mut x = dev(&f32b(&d.flat(&format!("{a}.in.{which}"), n * hd)));
                let p = x.ptr_mut() as *mut f32;
                // SAFETY: `x` holds `n · hd` f32 and the kernel rotates the first `rd` of each
                // `hd`-strided row in place.
                ok(
                    unsafe { launch_rope_split_half(p, n, hd, rd, pos, d.rope_theta()) },
                    "rope_split_half",
                );
                let got = f32v(&back(&x));
                let at = format!("{} L{layer}", d.salt);
                b.hold(&at, which, &got, &want);
                // The dims above `rotary_dim` must be untouched — the half of the partial-RoPE
                // claim that agreement on the whole row cannot separate from a full rotation
                // whose tail happened to land close.
                for hh in 0..n {
                    let row = &got[hh * hd..(hh + 1) * hd];
                    let src = &want[hh * hd..(hh + 1) * hd];
                    assert_eq!(
                        &row[rd..],
                        &src[rd..],
                        "{at} {which} head {hh}: the tail dims moved, so this is not a PARTIAL \
                         rotation"
                    );
                }
                // And the position is load-bearing: `pos - 1` must NOT reproduce it.
                let mut y = dev(&f32b(&d.flat(&format!("{a}.in.{which}"), n * hd)));
                let q = y.ptr_mut() as *mut f32;
                // SAFETY: as above.
                ok(
                    unsafe { launch_rope_split_half(q, n, hd, rd, pos - 1, d.rope_theta()) },
                    "rope_split_half at the previous position",
                );
                b.separates(&at, "one position early", h::rel(&f32v(&back(&y)), &want));
            }
        }
    }
}
