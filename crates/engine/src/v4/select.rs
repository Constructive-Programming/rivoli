//! Which rows a query may read, and where a pooled block goes — the two coordinate systems
//! this architecture keeps, and the arithmetic that keeps them apart.
//!
//! Ported from `old:src/kvcompress.rs` and `old:src/attn.rs`'s selection half. Those were two
//! files, and a review found the second had RE-DERIVED four of the first's functions under
//! different spellings, so the duplication gate could not see it: the values agreed, they were
//! not equivalent by construction, and the two copies had already diverged on whether the
//! indexer's cap applied. They are one module here for that reason.
//!
//! # The two coordinate systems, because confusing them is silent
//!
//! **Selection space** is what the attend kernel indexes. At prefill it is absolute positions
//! `0..seqlen` followed by the pooled blocks; at decode it is ring SLOTS `0..window` followed
//! by the pooled blocks. [`compress_offset`] owns that split.
//!
//! **Cache space** is the persistent `[ring ‖ compressed]` buffer, whose compressed region
//! always begins at `window`. [`compress_dst`] owns that.
//!
//! The two coincide at decode and disagree at prefill, and passing one where the other is due
//! aims every selected index at the wrong buffer — in bounds, finite, and wrong.

use super::geometry::{LayerKind, tightest_ratio};
use anyhow::{Context, Result, bail, ensure};

/// The rows one call covers: how many query rows, and where the first one sits.
///
/// A struct because every function below takes exactly this pair, and `(1, 47)` and `(47, 1)`
/// both type-check while meaning "one row at position 47" and "a 47-row prefill at position
/// 1". `start_pos == 0` is prefill throughout, which is the reference's own discriminant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    /// Query rows: the prompt length at prefill, exactly one at decode.
    pub seqlen: usize,
    /// 0 means prefill.
    pub start_pos: usize,
}

impl Extent {
    /// True at prefill. Spelled once so no call site re-derives the discriminant.
    pub fn is_prefill(self) -> bool {
        self.start_pos == 0
    }

    /// Query rows this call actually drives: every prompt row at prefill, exactly one at
    /// decode.
    ///
    /// `pub` because the attention block's every launch is sized by it and re-deriving
    /// `if is_prefill { seqlen } else { 1 }` there would be a second copy of the phase
    /// discriminant — the thing this type exists to hold once.
    pub fn query_rows(self) -> usize {
        if self.is_prefill() { self.seqlen } else { 1 }
    }

    /// One past the last position this call touches.
    ///
    /// `pub` since the scored selection: the engine's trigger for RUNNING the indexer is
    /// `end_pos / ratio > index_topk`, and a caller re-deriving `start_pos + query_rows()`
    /// would be a second copy of the phase discriminant this type exists to hold once.
    pub fn end_pos(self) -> usize {
        self.start_pos + self.query_rows()
    }

    /// **The bsz=1 scope cut, as a live check rather than a comment.**
    ///
    /// Every function here that branches on the phase ignores `seqlen` in its decode arm, so
    /// a caller handing over several rows at `start_pos > 0` gets ONE row back for N queries
    /// and no error — every row after the first would then be scored against row 0's index
    /// set. The reference stated this as two `debug_assert!`s and its own docs called them
    /// "what ENFORCES the scope cut"; they were then measured compiled out of every binary
    /// anyone ran. This is an `ensure!` for that reason, and it is stated ONCE, here, rather
    /// than in each arm.
    ///
    /// **`pub` because the compressor is a third entry point**, not because anything outside
    /// wants it: `super::kvcompress::LayerCompressor::run` branches on the same phase and
    /// ignores `seqlen` in its decode arm exactly as [`should_compress`] and [`compress_dst`]
    /// do, and the reference stated the cut a second time there. One `ensure!` with three
    /// callers is the shape; three `ensure!`s is how two of them drift.
    pub fn check_single_row_decode(self) -> Result<()> {
        ensure!(
            self.is_prefill() || self.seqlen == 1,
            "v4 selection: decode consumes one query row (the bsz=1 scope cut), got {}",
            self.seqlen
        );
        Ok(())
    }
}

/// The total context past which a positional block selection stops agreeing with the
/// trained-in indexer: `4 * (index_topk + 1)`.
///
/// Below it, the causally-legal block count never exceeds `index_topk`, so the indexer keeps
/// every block it is offered and the two selections agree on the SET. Above it the indexer
/// keeps the top `index_topk` BY SCORE while a positional rule keeps the OLDEST — so every
/// position between the cap and the window goes unattended, on every indexed layer, for the
/// rest of the sequence. Fluent, wrong, and permanent.
///
/// Takes the number rather than a config so it is checkable without one.
///
/// **This stopped being an engine-level refusal when the scored selection landed (M15).**
/// It is now the boundary the engine PLANS around: an engine whose `max_ctx` is below it can
/// never need a score (every legal set fits under the cap, and [`scored_rows`] over any
/// scores reproduces the positional buffer byte for byte there), so it builds no indexer
/// state at all and runs exactly the pre-M15 arm. At or above it the indexer runs, and
/// [`Sel::gather`]'s positional path — which cannot rank — still refuses in [`Sel::shape`],
/// because a positional selection past this point is as wrong as it ever was.
pub fn positional_context_limit(index_topk: usize) -> usize {
    tightest_ratio() * (index_topk + 1)
}

/// The rule [`scored_rows`] ranks under — everything about the selection question except the
/// score matrix itself.
///
/// A struct for [`Sel`]'s own reason: `index_topk` and `offset` are both `usize`, and their
/// transposition type-checks while aiming every selected index at the wrong buffer. It is
/// also what keeps `scored_rows` under the arity gate (`crates/cli/tests/codescene.rs`),
/// which refused the loose six-parameter spelling on 2026-08-21.
#[derive(Clone, Copy, Debug)]
pub struct SelRule {
    /// The layer's class. See [`compressed_rows`] for why this is not a raw ratio.
    pub kind: LayerKind,
    /// The length past which the top-k truncates by score.
    pub index_topk: usize,
    pub at: Extent,
    /// Where the compressed region starts in SELECTION space — [`compress_offset`]'s value.
    pub offset: usize,
}

/// The indexer's top-k block selection, one row per query, from a `[rows, n_comp]` score
/// matrix — the HOST half of the scored selection, and deliberately deviceless: the gate
/// that compares it against the frozen oracle's `Indexer.forward` runs on every
/// `cargo test`, with no GPU and no checkpoint.
///
/// # What is reproduced from the reference, and what is deliberately not
///
/// * **The set.** Per query row: the causal mask (a prefill row may see only blocks that
///   closed at or before it — `c < (t+1)/ratio`, exactly [`compressed_rows`]'s rule), then
///   the top `min(index_topk, n_comp)` by score. Ties break toward the LOWER block index —
///   the frozen oracle's own rule (`v4oracle::forward::topk_idx`: descending `total_cmp`,
///   `.then(a.cmp(&b))`), restated here because the two top-ks must agree given equal
///   scores or the set gate has a hole exactly at ties.
/// * **Not the order.** The oracle emits score order; this emits the selected blocks
///   ASCENDING, `-1`-padded at the tail. The attend kernel's online softmax makes list
///   order a summation-order detail, and ascending buys the property the below-cap gates
///   stand on: where `limit <= k` the row is `[offset+0 .. offset+limit-1, -1 ..]` — BYTE
///   IDENTICAL to the positional fill, whatever the scores say. A run that never exceeds
///   the cap is therefore unchanged bit for bit, which is what makes the M8 parity record
///   still binding on the scored arm.
///
/// `scores` is row-major `[query_rows, n_comp]`, exactly what `launch_index_score_blocks`
/// wrote and D2H handed back; masked entries need NOT be pre-masked to `-inf` — the mask
/// here is by INDEX, not by value, so a stale score in a causally-illegal slot cannot leak
/// into the selection.
pub fn scored_rows(scores: &[f32], n_comp: usize, rule: SelRule) -> Result<Vec<Vec<i32>>> {
    rule.at.check_single_row_decode()?;
    let ratio = rule
        .kind
        .compressor_ratio()
        .context("scored selection on a layer with no compressor")?;
    let rows = rule.at.query_rows();
    ensure!(
        scores.len() == rows * n_comp,
        "v4 scored selection: {} scores for [{rows}, {n_comp}]",
        scores.len()
    );
    let k = rule.index_topk.min(n_comp);
    Ok((0..rows)
        .map(|t| {
            let row = &scores[t * n_comp..(t + 1) * n_comp];
            // The causal mask, by index: at prefill query `t` may read a block only once it
            // is COMPLETE. At decode every block in the matrix has closed.
            let limit = if rule.at.is_prefill() {
                ((t + 1) / ratio).min(n_comp)
            } else {
                n_comp
            };
            let mut sel: Vec<usize> = (0..limit).collect();
            sel.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            sel.truncate(k);
            // Ascending, then the `-1` tail out to the rectangle's width — see the header
            // for why ascending is load-bearing and not a tidiness choice.
            sel.sort_unstable();
            let mut out: Vec<i32> = sel.into_iter().map(|c| (c + rule.offset) as i32).collect();
            out.resize(k, -1);
            out
        })
        .collect())
}

/// Which sliding-window ring slots each query row may read, `-1` for "nothing there yet".
///
/// One row per query: `seqlen` rows at prefill, exactly one at decode. The values are ring
/// SLOTS at decode and absolute positions at prefill — see this module's header — and the
/// decode branch is a ROTATION rather than a range precisely because slot `t % win` holds
/// position `t`.
fn window_rows(win: usize, at: Extent) -> Vec<Vec<i32>> {
    let sp = at.start_pos;
    if at.is_prefill() {
        // ABSOLUTE POSITIONS here, not slots: the last `win` of them, causally masked.
        return rows_from(at.seqlen, win.min(at.seqlen), |t, j| {
            let v = t.saturating_sub(win - 1) + j;
            (v <= t).then_some(v)
        });
    }
    if sp >= win - 1 {
        // The ring is full: read oldest-first, starting just past the slot about to be
        // overwritten. ONE chained range rather than a build-then-extend — the shorter
        // spelling, and no cell of it is maskable, which is why it does not go through
        // [`rows_from`].
        let cut = sp % win;
        return vec![((cut + 1)..win).chain(0..=cut).map(|i| i as i32).collect()];
    }
    // Not yet full: slots `[0, sp]` are live and the rest pad out to the ring's WIDTH, not to
    // the live count, because `Sel::shape` fixes the row length before the fill runs.
    rows_from(1, win, |_, j| (j <= sp).then_some(j))
}

/// Which pooled blocks each query row may read, in SELECTION space — the indexer-free
/// selection, used on every layer whose class has a compressor.
///
/// Pure arithmetic: with no ranking there is nothing to rank, so every causally-legal block is
/// selected. `offset` shifts the indices past the window region, which occupies the low slots.
///
/// Takes a [`LayerKind`] and not a raw ratio, because every path here divides by it.
fn compressed_rows(kind: LayerKind, at: Extent, offset: usize) -> Vec<Vec<i32>> {
    let Some(ratio) = kind.compressor_ratio() else {
        return Vec::new();
    };
    if !at.is_prefill() {
        // One row, and nothing in it is masked: every block a decode step can see has closed.
        return rows_from(1, at.end_pos() / ratio, |_, c| Some(c + offset));
    }
    // Block `c` covers positions `[c*ratio, (c+1)*ratio)`, so query `t` may read it only once
    // that block is COMPLETE — `c < (t+1)/ratio`. Off by one here silently grants every query
    // one block of the future.
    rows_from(at.seqlen, at.seqlen / ratio, |t, c| {
        (c < (t + 1) / ratio).then_some(c + offset)
    })
}

/// `rows` selection rows of `cols` entries, from a per-cell rule that returns `None` for a
/// masked slot.
///
/// The two prefill fills above differ ONLY in that rule — one over ring positions, one over
/// pooled blocks — and writing the nested loop twice is what the duplication gate reported the
/// moment both existed. Masking is `-1` in exactly one place as a result, which is the half
/// that matters: the kernel reads a row's worth of whatever it is given, so a fill that
/// forgot to mask is in-bounds and wrong.
fn rows_from(
    rows: usize,
    cols: usize,
    cell: impl Fn(usize, usize) -> Option<usize>,
) -> Vec<Vec<i32>> {
    (0..rows)
        .map(|t| {
            (0..cols)
                .map(|j| cell(t, j).map_or(-1, |v| v as i32))
                .collect()
        })
        .collect()
}

/// Where the compressed region starts in SELECTION space.
///
/// At prefill the window region is the prompt itself, so the pooled blocks are appended at
/// `seqlen`; at decode the window region is the full ring, so they start at `window`. Two
/// different values for one variable, and passing the prefill one at decode aims every
/// selected index at the wrong buffer.
pub fn compress_offset(window: usize, at: Extent) -> usize {
    if at.is_prefill() { at.seqlen } else { window }
}

/// Does this call's compressor emit a block at all?
///
/// Prefill pools every complete block in one go and needs at least a full window of positions
/// to emit anything; decode emits one block only on the position that completes one. A
/// [`LayerKind::Plain`] layer has no compressor object at all, so the answer is false in BOTH
/// phases — which is the arm a raw `usize` gets wrong in opposite directions.
pub fn should_compress(kind: LayerKind, at: Extent) -> bool {
    let Some(ratio) = kind.compressor_ratio() else {
        return false;
    };
    if at.is_prefill() {
        at.seqlen >= ratio
    } else {
        (at.start_pos + 1).is_multiple_of(ratio)
    }
}

/// Where this call's pooled block(s) belong in CACHE space, as `(first_row, count)`. `None`
/// exactly when [`should_compress`] is false.
///
/// # `region_base` is the parameter, and it is 0 for the indexer's compressor
///
/// There are two compressors on an indexed layer and they write into differently-shaped
/// buffers. The attention compressor's blocks land in the `[ring ‖ compressed]` cache, so its
/// base is the window. The indexer's own buffer has NO ring and no window prefix, so its base
/// is 0 — and the layers that need the zero are exactly the layers that also need the
/// non-zero. The parameter is taken rather than derived from the [`LayerKind`] precisely so
/// both callers can exist; it cannot be defaulted, and a name like `window` on it would read
/// correct at one of the two call sites and wrong at the other.
///
/// # `start_pos / ratio`, never "the next free slot"
///
/// The decode row is a pure function of the POSITION. A caller that APPENDS — keeps a running
/// count and writes the next row each time a block is emitted — agrees with this on every
/// sequence that visits every position in order, and diverges the first time a step is
/// skipped. An appended block is a real pooled row of real numbers sitting one slot early, so
/// every later query selects it by position, weights it, and attends a window that is off by
/// `ratio` tokens for the rest of the sequence.
///
/// **That divergence is not currently reachable on this arm**, and saying otherwise would be
/// the load-bearing lie: [`super::ROWS`] is 1 because the expert kernel has no other
/// instantiation, so no verify pass can advance the position by two. The rule is still right
/// and still worth having — a skipped step is not exclusive to speculation — and the day an
/// `R = 2` kernel exists this becomes live with nothing to warn anyone.
pub fn compress_dst(kind: LayerKind, region_base: usize, at: Extent) -> Option<(usize, usize)> {
    let ratio = kind.compressor_ratio()?;
    if !should_compress(kind, at) {
        return None;
    }
    Some(if at.is_prefill() {
        // Prefill emits every COMPLETE block at once, starting at the region's base.
        (region_base, at.seqlen / ratio)
    } else {
        (region_base + at.start_pos / ratio, 1)
    })
}

/// What one selection is over — everything [`Sel::shape`] and [`Sel::gather`] share.
///
/// A struct because four of its fields are `usize` and any permutation of them type-checks,
/// while the failure is not a panic: it attends real vectors at the wrong positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sel {
    /// `sliding_window` — the ring's size, and the window part's column count at decode.
    pub win: usize,
    /// The layer's class. See [`compressed_rows`] for why this is not a raw ratio.
    pub kind: LayerKind,
    /// `index_topk` from the config — read only on a layer that has an indexer, where it is
    /// the length past which this positional selection REFUSES.
    pub index_topk: usize,
    pub at: Extent,
}

impl Sel {
    /// The `(rows, cols)` a buffer for this selection must hold.
    ///
    /// Fallible, and derived from the same functions the fill uses, so the attention block can
    /// check a caller's uploaded shape without building the selection twice.
    pub fn shape(&self) -> Result<(usize, usize)> {
        self.at.check_single_row_decode()?;
        ensure!(self.win > 0, "v4 selection: sliding_window is zero");
        let win_cols = if self.at.is_prefill() {
            self.at.seqlen.min(self.win)
        } else {
            self.win
        };
        Ok((self.at.query_rows(), win_cols + self.n_comp()?))
    }

    /// Compressed columns — the causally-complete block count, refused past the point where
    /// a positional rule stops agreeing with the indexer.
    fn n_comp(&self) -> Result<usize> {
        let Some(ratio) = self.kind.compressor_ratio() else {
            return Ok(0);
        };
        let live = self.at.end_pos() / ratio;
        // **REFUSE, do not cap.** `min(live, index_topk)` was written here first and is
        // exactly the bug [`positional_context_limit`] describes: it keeps the OLDEST blocks
        // and silently stops attending everything newer.
        if self.kind.has_indexer() && live > self.index_topk {
            bail!(
                "v4 selection: {live} compressed blocks at position {} exceeds index_topk {} \
                 on an indexed layer. Past {} positions the block set is decided by the \
                 indexer's SCORES — the scored path (gather_scored) is how it is answered, \
                 and a positional selection here keeps the oldest blocks and silently stops \
                 attending everything newer",
                self.at.end_pos(),
                self.index_topk,
                positional_context_limit(self.index_topk)
            );
        }
        Ok(live)
    }

    /// The window rows and the compressed rows concatenated into one row-major `i32` buffer,
    /// which is what the attend kernel indexes. Appends `rows * cols` entries to `out` and
    /// returns `(rows, cols)`; `-1` masks a slot.
    ///
    /// **This is the POSITIONAL compressed selection.** On a layer with no indexer that is the
    /// whole story — and not a deviation: the reference's own `get_compress_topk_idxs` is this
    /// same arithmetic there. On an indexed one it agrees with the trained-in ranking on the
    /// SET exactly while `limit <= index_topk` — where [`scored_rows`] produces the identical
    /// buffer — and refuses past that, where only [`Sel::gather_scored`] can answer.
    pub fn gather(&self, out: &mut Vec<i32>) -> Result<(usize, usize)> {
        let (rows, cols) = self.shape()?;
        let win = window_rows(self.win, self.at);
        // The compressed columns are the rest of `cols`, derived from the fill rather than by
        // re-running `n_comp` — which `shape` has already called, so a second call could only
        // return the same value or diverge.
        let n_comp = cols - win.first().map_or(0, Vec::len);
        let comp = match n_comp {
            0 => Vec::new(),
            _ => compressed_rows(self.kind, self.at, compress_offset(self.win, self.at)),
        };
        assemble(&win, &comp, (rows, cols), out)
    }

    /// The window rows concatenated with the INDEXER's rows — [`scored_rows`]'s output, or
    /// any per-row block selection the caller ranked — into the same row-major buffer
    /// [`Sel::gather`] fills, under the same rectangle check.
    ///
    /// This is the path with no cap and no refusal: the ranking is the caller's, so the
    /// positional path's "keeps the oldest blocks" hazard does not exist here. What it keeps
    /// is everything else — one window fill, one masking convention, one rectangle check —
    /// because a second copy of that assembly was exactly how the reference's two selection
    /// files diverged (this module's header).
    pub fn gather_scored(&self, comp: &[Vec<i32>], out: &mut Vec<i32>) -> Result<(usize, usize)> {
        self.at.check_single_row_decode()?;
        let rows = self.at.query_rows();
        ensure!(
            comp.len() == rows,
            "v4 scored selection: {} ranked rows for {rows} query rows",
            comp.len()
        );
        // BEFORE `window_rows`, which is where a zero window becomes a `win - 1` underflow
        // rather than this message — [`Sel::shape`] orders the same two the same way.
        ensure!(self.win > 0, "v4 selection: sliding_window is zero");
        let win = window_rows(self.win, self.at);
        let cols = win.first().map_or(0, Vec::len) + comp.first().map_or(0, Vec::len);
        assemble(&win, comp, (rows, cols), out)
    }
}

/// Interleave one window fill with one compressed fill into the row-major rectangle the
/// attend kernel indexes. Appends to `out` and returns the `(rows, cols)` it wrote.
///
/// The one home of the rectangle check, and it must be live in every build: a ragged
/// selection buffer is a silent-wrong, because the kernel reads a row's worth of whatever
/// follows as attention indices.
///
/// **Two checks and not one, because a total is blind to compensation.** The PER-ROW width
/// is what catches raggedness: rows of 2 and 4 against `cols == 3` sum to exactly `2 * 3`,
/// so a total-only test passes a buffer every row of which is the wrong length — and
/// `gather_scored` takes its compressed rows from a CALLER, which is the one path that can
/// produce them. The total then covers what a per-row check cannot see: too few or too many
/// rows. Against a start mark because this function appends — a `% cols == 0` test on the
/// whole buffer fires on a legitimate non-empty append and passes on a short final row.
fn assemble(
    win: &[Vec<i32>],
    comp: &[Vec<i32>],
    (rows, cols): (usize, usize),
    out: &mut Vec<i32>,
) -> Result<(usize, usize)> {
    let start = out.len();
    for (t, w) in win.iter().enumerate() {
        let c = comp.get(t).map_or(&[][..], Vec::as_slice);
        ensure!(
            w.len() + c.len() == cols,
            "v4 selection: row {t} is {} wide ({} window + {} compressed), the rectangle is \
             {cols}",
            w.len() + c.len(),
            w.len(),
            c.len()
        );
        out.extend_from_slice(w);
        out.extend_from_slice(c);
    }
    ensure!(
        out.len() - start == rows * cols,
        "v4 selection: wrote {} entries, {rows}x{cols} needs {}",
        out.len() - start,
        rows * cols
    );
    Ok((rows, cols))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)] // tests: panic-on-failure is the idiom

    // Named rather than a glob, because the private fills are exactly what these cases drive
    // and a glob would let one of them be deleted without a single case going red.
    use super::{
        Extent, LayerKind, Sel, SelRule, compress_dst, compress_offset, compressed_rows,
        positional_context_limit, scored_rows, window_rows,
    };

    const PLAIN: LayerKind = LayerKind::Plain;

    fn at(seqlen: usize, start_pos: usize) -> Extent {
        Extent { seqlen, start_pos }
    }

    /// What every case below starts from, so each one names only the axis it varies. The
    /// window is small on purpose: at the shipped 128 a fixture would need a 128-token prompt
    /// before the ring ever wrapped, and the wrap is where the two index spaces separate.
    const BASE: Sel = Sel {
        win: 8,
        kind: LayerKind::Plain,
        index_topk: 512,
        at: Extent {
            seqlen: 1,
            start_pos: 0,
        },
    };

    fn sel(kind: LayerKind, at: Extent) -> Sel {
        Sel { kind, at, ..BASE }
    }

    /// **The scope cut, live.** A multi-row decode must be refused and not silently answered
    /// with row 0's set, which is what every phase branch here would otherwise do.
    #[test]
    fn a_multi_row_decode_is_refused_rather_than_answered_for_row_zero() {
        let s = sel(PLAIN, at(3, 5));
        let msg = format!("{}", s.shape().expect_err("must refuse"));
        assert!(msg.contains("one query row"), "wrong refusal: {msg}");
        // The two legal shapes still pass, so the check is not refusing everything.
        assert!(
            sel(PLAIN, at(3, 0)).shape().is_ok(),
            "a 3-row prefill is legal"
        );
        assert!(
            sel(PLAIN, at(1, 5)).shape().is_ok(),
            "a 1-row decode is legal"
        );
    }

    /// The ring is a ROTATION, not a range: once it is full the oldest slot comes first, and
    /// the slot about to be overwritten comes last. A range would read the same slots in the
    /// wrong causal order.
    #[test]
    fn a_full_ring_is_read_oldest_first_from_just_past_the_slot_being_overwritten() {
        // win 4, position 6 -> slot 2 holds position 6, so the oldest live slot is 3.
        assert_eq!(window_rows(4, at(1, 6)), vec![vec![3, 0, 1, 2]]);
        // Not yet full: live slots then padding out to the ring's width.
        assert_eq!(window_rows(4, at(1, 2)), vec![vec![0, 1, 2, -1]]);
    }

    /// A query may read a pooled block only once that block is COMPLETE. Off by one here
    /// grants every query one block of the future, which no shape check can see.
    #[test]
    fn a_prefill_query_reads_only_blocks_that_closed_at_or_before_its_own_position() {
        let k = LayerKind::from_ratio(2);
        // 6 positions, ratio 2 -> 3 blocks; query t may read blocks below (t+1)/2.
        let rows = compressed_rows(k, at(6, 0), 100);
        assert_eq!(rows[0], vec![-1, -1, -1], "query 0 has closed no block");
        assert_eq!(rows[1], vec![100, -1, -1], "query 1 closes block 0");
        assert_eq!(rows[3], vec![100, 101, -1], "query 3 closes block 1");
        assert_eq!(rows[5], vec![100, 101, 102], "query 5 closes block 2");
    }

    /// **INV-7's selection half.** At prefill one emitted block has TWO destinations — a cache
    /// row and a selection column — and they differ; at decode they coincide. That difference
    /// is why the loop's placement branches on the two BASES rather than on the phase.
    #[test]
    fn the_cache_row_and_the_selection_column_differ_at_prefill_and_coincide_at_decode() {
        let k = LayerKind::from_ratio(4);
        let (win, prompt) = (128, 16);
        let pre = at(prompt, 0);
        let cache = compress_dst(k, win, pre).expect("16 >= 4 emits");
        let selection = compress_dst(k, compress_offset(win, pre), pre).expect("same call");
        assert_eq!(cache, (win, 4), "cache space always starts at the window");
        assert_eq!(
            selection,
            (prompt, 4),
            "selection space starts past the prompt"
        );
        assert_ne!(cache.0, selection.0, "the prefill bases must differ");
        let dec = at(1, 47);
        let (c, s) = (
            compress_dst(k, win, dec).expect("48 completes a block"),
            compress_dst(k, compress_offset(win, dec), dec).expect("same call"),
        );
        assert_eq!(
            (c, s),
            ((win + 11, 1), (win + 11, 1)),
            "decode bases coincide"
        );
    }

    /// **A positional placer must be POSITIONAL.** An appending placer agrees on every
    /// sequence that visits every position, so only a skipped step separates them — and this
    /// is the assertion that keeps the rule from being quietly replaced by a counter.
    #[test]
    fn an_appending_placer_disagrees_with_the_positional_rule_the_first_time_a_step_is_skipped() {
        let k = LayerKind::from_ratio(4);
        // Positions 15 and 31 both complete a block; 16..30 are skipped. `enumerate` IS the
        // appending placer — a running count of blocks emitted, which is the rule this
        // function must not be.
        let rows: Vec<(usize, usize)> = [15usize, 31]
            .into_iter()
            .enumerate()
            .map(|(appended, pos)| {
                let positional = compress_dst(k, 0, at(1, pos)).expect("completes a block").0;
                (positional, appended)
            })
            .collect();
        assert_eq!(
            rows,
            vec![(3, 0), (7, 1)],
            "positional 3,7 against appended 0,1"
        );
        assert!(
            rows.iter().all(|(p, a)| p != a),
            "the two placers agreed on a gapped script — the positional rule has been \
             replaced by a counter"
        );
    }

    /// The compressed selection is capped by the indexer's ranking, and the answer is a
    /// refusal rather than a truncation. A cap would keep the oldest blocks forever.
    #[test]
    fn an_indexed_layer_refuses_past_its_cap_and_an_unindexed_one_never_does() {
        let topk = 2;
        let limit = positional_context_limit(topk);
        assert_eq!(limit, 12, "4 * (topk + 1)");
        let indexed = |pos| Sel {
            index_topk: topk,
            ..sel(LayerKind::from_ratio(4), at(1, pos))
        };
        assert!(indexed(limit - 2).shape().is_ok(), "inside the cap");
        let msg = format!("{}", indexed(limit).shape().expect_err("must refuse"));
        assert!(msg.contains("index_topk"), "wrong refusal: {msg}");
        // The same position on a layer with no indexer is fine: there is no ranking to
        // disagree with, so nothing is being stood in for.
        assert!(
            Sel {
                index_topk: topk,
                ..sel(LayerKind::from_ratio(2), at(1, limit * 4))
            }
            .shape()
            .is_ok(),
            "an unindexed layer has no cap"
        );
    }

    /// **The below-cap identity the whole M15 landing stands on:** wherever the causal
    /// limit never exceeds `index_topk`, the scored fill over ARBITRARY scores is byte
    /// identical to the positional fill — the top-k keeps everything it is offered, and
    /// ascending order plus the `-1` tail is exactly the positional layout. This is what
    /// makes a `--ctx` below [`positional_context_limit`] the pre-M15 arm bit for bit.
    #[test]
    fn below_the_cap_the_scored_fill_reproduces_the_positional_buffer_byte_for_byte() {
        let k = LayerKind::from_ratio(4);
        // Decode at position 19 (5 live blocks) and a 12-row prefill (3 blocks) — both
        // under a topk of 512, and both phases, since the mask rule differs between them.
        for at in [at(1, 19), at(12, 0)] {
            let s = sel(k, at);
            let (rows, cols) = s.shape().expect("legal below the cap");
            let n_comp = cols - window_rows(s.win, at).first().map_or(0, Vec::len);
            // Scores chosen ADVERSARIALLY: strictly increasing, so a top-k that leaked into
            // the selection ORDER would put the newest block first and the comparison would
            // catch it. Below the cap the scores must not matter at all.
            let scores: Vec<f32> = (0..rows * n_comp).map(|i| i as f32).collect();
            let rule = SelRule {
                kind: k,
                index_topk: s.index_topk,
                at,
                offset: compress_offset(s.win, at),
            };
            let comp = scored_rows(&scores, n_comp, rule).expect("legal scored rows");
            let (mut pos, mut scr) = (Vec::new(), Vec::new());
            assert_eq!(s.gather(&mut pos).expect("positional"), (rows, cols));
            assert_eq!(
                s.gather_scored(&comp, &mut scr).expect("scored"),
                (rows, cols)
            );
            assert_eq!(
                pos, scr,
                "{at:?}: the two fills must be identical below the cap"
            );
        }
    }

    /// Above the cap the scores DECIDE the set: the top `index_topk` by score survive, ties
    /// break toward the lower block index (the frozen oracle's `topk_idx` rule), and the
    /// output is ascending. Every clause here is one the below-cap identity cannot see.
    #[test]
    fn above_the_cap_the_scores_decide_the_set_with_ties_toward_the_lower_index() {
        let k = LayerKind::from_ratio(4);
        // Decode with 5 live blocks and topk 3: block 2 scores highest, blocks 0 and 4 tie
        // for the next two slots ALONG WITH block 1 — the three-way tie at 1.0 makes the
        // tie rule observable: {0, 1} must win over 4.
        let scores = [1.0f32, 1.0, 9.0, 0.5, 1.0];
        let rule = SelRule {
            kind: k,
            index_topk: 3,
            at: at(1, 19),
            offset: 100,
        };
        let rows = scored_rows(&scores, 5, rule).expect("legal");
        assert_eq!(
            rows,
            vec![vec![100, 101, 102]],
            "ascending; ties kept 0 and 1"
        );
        // A score bump on block 4 must MOVE the set — the resolution the gate stands on.
        let mut moved = scores;
        moved[4] = 2.0;
        let rows = scored_rows(&moved, 5, rule).expect("legal");
        assert_eq!(
            rows,
            vec![vec![100, 102, 104]],
            "the perturbed score changed the set"
        );
    }

    /// At prefill the mask is BY INDEX, so a hostile score in a causally-illegal slot cannot
    /// buy that block a place, and a row with fewer legal blocks than the rectangle pads
    /// with `-1` at the tail.
    #[test]
    fn a_prefill_row_cannot_select_a_block_that_closed_after_it() {
        let k = LayerKind::from_ratio(4);
        // 8 rows, 2 blocks live over the whole prompt, topk 1 -> every row keeps its single
        // best LEGAL block. Block 1 outscores block 0 everywhere, but only rows 7.. may
        // read it (block 1 closes at position 7).
        let scores: Vec<f32> = (0..8).flat_map(|_| [0.1f32, 9.0]).collect();
        let rule = SelRule {
            kind: k,
            index_topk: 1,
            at: at(8, 0),
            offset: 10,
        };
        let rows = scored_rows(&scores, 2, rule).expect("legal");
        let want: Vec<Vec<i32>> = vec![
            vec![-1],
            vec![-1],
            vec![-1],
            vec![10], // row 3 closes block 0; block 1 is still open
            vec![10],
            vec![10],
            vec![10],
            vec![11], // row 7 closes block 1, which then wins on score
        ];
        assert_eq!(rows, want);
        // And a score matrix of the wrong extent is refused, not truncated.
        let msg = format!(
            "{}",
            scored_rows(&scores[..15], 2, rule).expect_err("must refuse")
        );
        assert!(msg.contains("15 scores"), "wrong refusal: {msg}");
    }

    /// `gather_scored` enforces the same rectangle as `gather`: a ragged ranked row is a
    /// refusal, not a short row the kernel reads past.
    #[test]
    fn a_ragged_scored_row_is_refused_by_the_rectangle_check() {
        let s = sel(LayerKind::from_ratio(4), at(1, 19));
        let mut out = Vec::new();
        assert!(
            s.gather_scored(&[vec![100, 101, -1]], &mut out).is_ok(),
            "uniform passes"
        );
        let mut out = Vec::new();
        // Two rows for a one-row decode: the row-count ensure fires first and by name.
        let msg = format!(
            "{}",
            s.gather_scored(&[vec![100], vec![101]], &mut out)
                .expect_err("must refuse")
        );
        assert!(msg.contains("2 ranked rows"), "wrong refusal: {msg}");
        // **The case a TOTAL cannot see, which is why `assemble` checks per row.** Widths
        // 2, 1, 3 over a 3-row prefill sum to 6 — exactly `3 * comp[0].len()` — so every
        // entry count agrees while rows 1 and 2 are the wrong length, and the kernel would
        // read row 1's fifth index out of row 2's window. One row of this is what a
        // caller-ranked selection can hand over; the positional fill cannot produce it.
        let p = sel(LayerKind::from_ratio(4), at(3, 0));
        let ragged = [vec![10, 11], vec![10], vec![10, 11, 12]];
        let mut out = Vec::new();
        let msg = format!(
            "{}",
            p.gather_scored(&ragged, &mut out).expect_err("must refuse")
        );
        assert!(msg.contains("row 1 is 4 wide"), "wrong refusal: {msg}");
        // And the same three rows made rectangular pass, so the check is not refusing the
        // shape rather than the raggedness.
        let square = [vec![10, 11], vec![10, -1], vec![10, 11]];
        let mut out = Vec::new();
        assert_eq!(
            p.gather_scored(&square, &mut out).expect("uniform passes"),
            (3, 5)
        );
    }

    /// `gather` must fill exactly the rectangle `shape` promised. The two are derived
    /// separately — one from the phase, one from the fill — and a disagreement hands the
    /// kernel a ragged buffer it reads past.
    #[test]
    fn the_gathered_buffer_is_exactly_the_rectangle_shape_promised() {
        let cases = [
            (LayerKind::from_ratio(0), at(5, 0)),
            (LayerKind::from_ratio(0), at(1, 9)),
            (LayerKind::from_ratio(4), at(12, 0)),
            (LayerKind::from_ratio(4), at(1, 19)),
            (LayerKind::from_ratio(128), at(1, 300)),
        ];
        for (kind, step) in cases {
            let s = sel(kind, step);
            let want = s.shape().expect("legal shape");
            // Pre-fill `out` so the append-relative accounting is actually exercised: a
            // check against `out.len()` outright would pass here and fail on a real caller.
            let mut out = vec![-7i32; 3];
            assert_eq!(s.gather(&mut out).expect("legal fill"), want);
            assert_eq!(out.len(), 3 + want.0 * want.1, "{kind:?} {step:?}");
            assert_eq!(&out[..3], &[-7, -7, -7], "the caller's prefix is untouched");
        }
    }
}
