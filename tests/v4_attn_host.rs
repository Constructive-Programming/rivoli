//! The V4 selection arithmetic, scored against S1b's oracle on the HOST.
//!
//! `src/attn.rs::v4_topk_idxs` decides which cache slots every query attends. It is a
//! *selection*, so no numeric tolerance can stand in for it: a wrong rotation attends to
//! real vectors at the wrong positions and produces fluent wrong text. It is also pure,
//! so it can be gated without a GPU — which is why it lives here rather than in
//! `tests/v4_attn.rs`, and why this file carries no `#![cfg(feature = "rocm")]`.
//!
//! **What agreement here does and does not prove.** Both sides transliterate the same
//! ten lines of `get_window_topk_idxs`, so a shared misreading passes. What makes the
//! comparison worth running is that the two are structurally different — the oracle
//! builds `Vec<Vec<i64>>` per row, this builds one flat `i32` buffer and returns its
//! shape — so a transcription slip in either shows up. The properties asserted below do
//! not depend on either implementation and are the part that survives a shared
//! misreading.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::attn::{CompressSel, Sel, v4_topk_idxs};
use rivoli::v4oracle::forward::window_topk;

/// The engine's flat buffer, re-shaped into the oracle's row form so the two can be
/// compared as the same object.
fn engine_rows(win: usize, seqlen: usize, start_pos: usize) -> Vec<Vec<i64>> {
    engine_sel(win, None, seqlen, start_pos)
}

/// The general form: `comp: None` is a ratio-0 layer, `Some` a compressed one.
fn engine_sel(
    win: usize,
    comp: Option<CompressSel>,
    seqlen: usize,
    start_pos: usize,
) -> Vec<Vec<i64>> {
    let mut flat = Vec::new();
    let (rows, cols) = v4_topk_idxs(Sel { win, comp, seqlen, start_pos }, &mut flat);
    assert_eq!(flat.len(), rows * cols, "returned shape does not describe the buffer");
    flat.chunks_exact(cols).map(|r| r.iter().map(|&v| i64::from(v)).collect()).collect()
}

/// Every shape the ratio-0 layers reach, plus the boundaries around the ring wrap.
///
/// `start_pos` values are chosen around `window - 1`, which is where the reference
/// switches from "the ring is still filling, list it ascending" to "the ring is full,
/// rotate it". Getting that boundary off by one leaves the newest token in the wrong
/// place in the list, which no magnitude check can see.
fn cases() -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for &win in &[2usize, 4, 8, 128] {
        // Prefill, both sides of `seqlen == win`.
        for seqlen in [1usize, win - 1, win, win + 1, 2 * win, 2 * win + 3] {
            v.push((win, seqlen, 0));
        }
        // Decode, straddling the wrap. `start_pos > 0` throughout: `start_pos == 0` is
        // prefill by definition in the reference.
        for sp in [1usize, win - 2, win - 1, win, win + 1, 3 * win - 1, 5 * win + 2] {
            if sp > 0 {
                v.push((win, 1, sp));
            }
        }
    }
    v
}

#[test]
fn window_selection_matches_the_oracle_at_every_reachable_shape() {
    let mut checked = 0usize;
    for (win, seqlen, start_pos) in cases() {
        let want = window_topk(win, seqlen, start_pos);
        let got = engine_rows(win, seqlen, start_pos);
        assert_eq!(got, want, "win={win} seqlen={seqlen} start_pos={start_pos}");
        checked += 1;
    }
    // Anti-vacuity: an empty `cases()` would make the loop above pass by doing nothing.
    assert!(checked >= 40, "only {checked} shapes compared");
}

#[test]
fn the_comparison_above_can_fail() {
    // The gate is worth nothing if it agrees with anything. Three transcription slips a
    // port actually makes, each fed through the SAME comparison, each of which must be
    // rejected at a shape the suite above covers.
    // `sp` must be a shape `cases()` covers, or this demonstrates a separation the
    // comparison suite never exercises. 9 is in the `win + 1` slot of the decode sweep.
    let (win, sp) = (8usize, 9usize);
    let right = window_topk(win, 1, sp);
    // Off-by-one on the wrap boundary: rotate from `start_pos % win` instead of
    // `start_pos % win + 1`, so the newest slot is listed first instead of last.
    let rotated_wrong: Vec<Vec<i64>> =
        vec![(sp % win..win).chain(0..sp % win).map(|i| i as i64).collect()];
    assert_ne!(rotated_wrong, right, "the wrap boundary is not observable at win={win}");
    // Ascending slots — right while the ring is filling, wrong once it has wrapped.
    let unrotated: Vec<Vec<i64>> = vec![(0..win).map(|i| i as i64).collect()];
    assert_ne!(unrotated, right, "rotation is not observable at start_pos={sp}");
    // A prefill row list handed to a decode step: right length, wrong space entirely.
    assert_ne!(window_topk(win, win, 0)[win - 1], right[0], "prefill and decode coincide");
}

#[test]
fn a_full_ring_lists_every_slot_once_with_the_newest_last() {
    // Independent of both implementations: once the ring has wrapped, the list must be a
    // PERMUTATION of the slots with the just-written one at the end. An implementation
    // that dropped a slot, or listed one twice, would still look ordered and would still
    // have the right length.
    for win in [2usize, 4, 8, 128] {
        for sp in [win - 1, win, win + 1, 4 * win + 3] {
            let rows = engine_rows(win, 1, sp);
            assert_eq!(rows.len(), 1);
            let mut sorted = rows[0].clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                (0..win as i64).collect::<Vec<_>>(),
                "win={win} start_pos={sp}: not a permutation of the ring"
            );
            assert_eq!(
                *rows[0].last().unwrap(),
                (sp % win) as i64,
                "win={win} start_pos={sp}: the newest slot is not last"
            );
        }
    }
}

#[test]
fn prefill_is_causal_and_never_names_a_position_it_has_not_reached() {
    // Also independent of both: row `t` may name only `0..=t`, must name `t` itself
    // (otherwise `sparse_attn` divides by a denominator with no numerator), and must
    // name at most `window` distinct positions.
    for win in [2usize, 8, 128] {
        for seqlen in [1usize, win, 2 * win + 3] {
            let rows = engine_rows(win, seqlen, 0);
            assert_eq!(rows.len(), seqlen);
            for (t, row) in rows.iter().enumerate() {
                let live: Vec<i64> = row.iter().copied().filter(|&v| v >= 0).collect();
                assert!(
                    live.iter().all(|&v| v >= 0 && v as usize <= t),
                    "win={win} seqlen={seqlen} row {t} attends the future: {row:?}"
                );
                assert!(live.contains(&(t as i64)), "row {t} does not attend itself");
                assert_eq!(live.len(), (t + 1).min(win), "row {t} has the wrong span");
            }
        }
    }
}

#[test]
fn win_one_diverges_from_the_reference_and_is_unreachable() {
    // A KNOWN, DELIBERATE divergence, recorded rather than hidden. `model.py`'s first
    // branch is `if start_pos >= window_size - 1`, which at `window_size == 1` is true
    // even for `start_pos == 0` — so the reference takes the DECODE branch on a prefill
    // call. Both this engine and the oracle guard that with `start_pos > 0` and take the
    // prefill branch instead.
    //
    // It is unreachable: `sliding_window` is 128 in the shipped config and a window of 1
    // would make the model attend only its own token. The assertion is here so that the
    // divergence is a decision on the record and not a latent surprise for S3.
    let mut flat = Vec::new();
    let (rows, cols) = v4_topk_idxs(Sel { win: 1, comp: None, seqlen: 3, start_pos: 0 }, &mut flat);
    assert_eq!((rows, cols), (3, 1), "win=1 prefill no longer takes the prefill branch");
    assert_eq!(flat, vec![0, 1, 2]);
    assert_eq!(engine_rows(1, 3, 0), window_topk(1, 3, 0), "engine and oracle still agree");
}

/// **`wo_a` may be read as fp8 and dequantized rather than held as bf16 — and here is
/// the range over which that is lossless, plus where it stops being.**
///
/// The plan's DECIDED note of 2026-08-05 fixes `wo_a` at bf16 because `convert.py`
/// dequantizes it and `Attention.forward` does a bf16 einsum. `src/attn.rs::v4` keeps the
/// checkpoint's fp8 bytes and dequantizes in the GEMV instead, which is only equivalent
/// because an e4m3 value carries at most 4 significant bits and the block scale is a bare
/// power of two, so `e4m3 * scale` fits bf16's 8 exactly.
///
/// That is a claim, so it is checked rather than asserted — over EVERY e4m3 code against
/// every e8m0 scale, and the boundary where it fails is exhibited rather than left as a
/// surprise for whoever first meets a tiny scale.
#[test]
fn fp8_times_a_power_of_two_is_exact_in_bf16_over_the_range_the_checkpoint_uses() {
    use rivoli::math::{bf16_to_f32, e4m3_to_f32, f32_to_bf16};

    let exact = |v: f32| bf16_to_f32(f32_to_bf16(v)).to_bits() == v.to_bits();
    // `layers.0.attn.wo_a`'s scale codes span 115..=117 (measured on the checkpoint).
    // The band below is far wider than any tensor in the model uses.
    let mut checked = 0usize;
    for code in 40u8..=200 {
        let s = f32::exp2(f32::from(code) - 127.0);
        for b in 0u8..=255 {
            let v = e4m3_to_f32(b);
            if v.is_nan() {
                continue;
            }
            assert!(exact(v * s), "e4m3 {b:#04x} ({v}) * 2^{} is not exact in bf16", i32::from(code) - 127);
            checked += 1;
        }
    }
    assert!(checked > 40_000, "only {checked} pairs checked");

    // THE BOUNDARY. bf16's smallest subnormal is 2^-133, so once `e4m3_subnormal * scale`
    // falls under it the product is NOT representable and the equivalence above stops
    // holding. e4m3's smallest subnormal is 2^-9, so that is scale codes below 127-124.
    // Unreachable for a weight tensor -- `act_quant`'s own scales bottom out at 2^-22 --
    // but stated here so the claim above carries its range and not just its conclusion.
    let tiny = e4m3_to_f32(0x01) * f32::exp2(-126.0); // 2^-9 * 2^-126 = 2^-135
    assert!(tiny > 0.0, "the f32 product itself underflowed, which is a different failure");
    assert!(!exact(tiny), "the bf16 subnormal floor moved; re-derive the range above");
}

// ═══════════════════════════════════════════════════════════════════════════════════
// The COMPRESSED half of the selection — S3 prerequisite 5.
// ═══════════════════════════════════════════════════════════════════════════════════
//
// `v4_topk_idxs(win, Some(comp), ..)` is `torch.cat([get_window_topk_idxs(...),
// get_compress_topk_idxs(...)], dim=-1)`. The oracle's twin is the pair
// `window_topk` / `compress_topk`, which it also concatenates — so the two sides are
// again structurally different (one flat i32 buffer with a derived offset, versus two
// `Vec<Vec<i64>>` joined by the caller) and a transcription slip in either shows up.
//
// **WHAT THIS DOES NOT COVER, stated because the shape of it has burned this repo.**
// `compress_topk` is the reference's `else` branch — the path a **ratio-128** layer takes,
// where there is no `Indexer`. A **ratio-4** layer runs `Indexer.forward` instead, which
// SCORES. Everything below is therefore a full gate for ratio-128 and, for ratio-4, a gate
// on the positional stand-in only: it agrees with the indexer on the SET while
// `index_topk` cannot truncate, and never on the score ORDER that `sparse_attn`'s online
// softmax folds in. No test in this file exercises one line of sparse selection, and none
// is named as though it did.

use rivoli::v4oracle::forward::compress_topk;

/// The reference's `torch.cat`, built from the oracle's two halves at an EXPLICIT offset.
///
/// Explicit so the defect cases below can pass a wrong one through the identical
/// concatenation — otherwise each would need its own copy of these four lines, and each
/// copy would be a place for the defect to be built differently from the thing it claims
/// to perturb.
fn oracle_cat(
    win: usize,
    ratio: usize,
    seqlen: usize,
    start_pos: usize,
    offset: usize,
) -> Vec<Vec<i64>> {
    let w = window_topk(win, seqlen, start_pos);
    let c = compress_topk(ratio, seqlen, start_pos, offset);
    assert_eq!(w.len(), c.len(), "the two halves disagree on the row count");
    w.into_iter().zip(c).map(|(a, b)| a.into_iter().chain(b).collect()).collect()
}

/// `offset` is `kv.size(1)` at prefill and `window_size` at decode — model.py:521.
fn oracle_rows(win: usize, ratio: usize, seqlen: usize, start_pos: usize) -> Vec<Vec<i64>> {
    oracle_cat(win, ratio, seqlen, start_pos, if start_pos == 0 { seqlen } else { win })
}

fn engine_comp_rows(win: usize, ratio: usize, seqlen: usize, start_pos: usize) -> Vec<Vec<i64>> {
    let sel = CompressSel { ratio, index_topk: (ratio == 4).then_some(512) };
    engine_sel(win, Some(sel), seqlen, start_pos)
}

/// Shapes a compressed layer actually reaches. `seqlen` straddles multiples of `ratio`
/// because the remainder is what decides how many blocks exist, and `start_pos` straddles
/// multiples of `ratio` because that is where a decode step emits one.
fn comp_cases() -> Vec<(usize, usize, usize, usize)> {
    let mut v = Vec::new();
    for &(win, ratio) in &[(8usize, 4usize), (128, 4), (128, 128), (4, 4)] {
        for seqlen in [1usize, ratio - 1, ratio, ratio + 1, 2 * ratio, 3 * ratio + 2, 130] {
            v.push((win, ratio, seqlen, 0));
        }
        for sp in [1usize, ratio - 1, ratio, ratio + 1, 2 * ratio - 1, win, win + 1, 300] {
            v.push((win, ratio, 1, sp));
        }
    }
    v
}

#[test]
fn compressed_selection_matches_the_oracle_below_the_truncation_point() {
    let mut checked = 0usize;
    let mut saw_blocks = 0usize;
    for (win, ratio, seqlen, start_pos) in comp_cases() {
        let want = oracle_rows(win, ratio, seqlen, start_pos);
        let got = engine_comp_rows(win, ratio, seqlen, start_pos);
        assert_eq!(got, want, "win={win} ratio={ratio} seqlen={seqlen} start_pos={start_pos}");
        // How many compressed columns this shape actually produced.
        let n = got[0].len() - if start_pos == 0 { seqlen.min(win) } else { win };
        saw_blocks += usize::from(n > 0);
        checked += 1;
    }
    assert!(checked >= 50, "only {checked} shapes compared");
    // ANTI-VACUITY, and the specific one this file needs: every case could agree because
    // every case produced ZERO compressed columns, which is `window_topk` re-tested under
    // a longer name. `saw_blocks` counts the cases that actually appended blocks.
    assert!(saw_blocks >= 30, "only {saw_blocks} of {checked} shapes produced any block");
}

#[test]
fn the_compressed_comparison_can_fail() {
    // Same discipline as `the_comparison_above_can_fail`: three slips a port makes here,
    // each rejected by the SAME comparison, at shapes `comp_cases()` covers.
    let (win, ratio, seqlen) = (8usize, 4usize, 12usize);
    let right = oracle_rows(win, ratio, seqlen, 0);

    // 1. The offset omitted — the classic. Blocks then name rows inside the WINDOW
    //    region, which are real KV vectors at completely unrelated positions.
    let no_offset = oracle_cat(win, ratio, seqlen, 0, 0);
    assert_ne!(no_offset, right, "the compressed offset is not observable at seqlen={seqlen}");

    // 2. Decode's offset used at prefill (`win` instead of `kv.size(1)`). Same shape,
    //    every block shifted by `seqlen - win`.
    let decode_offset = oracle_cat(win, ratio, seqlen, 0, win);
    assert_ne!(decode_offset, right, "the two offsets coincide at seqlen={seqlen} win={win}");

    // 3. The causal mask dropped, so row `t` attends blocks built from tokens after it.
    //    Row 0 is the discriminating one: it may see NO block at all.
    let unmasked: Vec<i64> = (0..seqlen / ratio).map(|c| (c + seqlen) as i64).collect();
    assert_ne!(unmasked, right[0][win.min(seqlen)..], "row 0's mask is not observable");
}

#[test]
fn index_topk_caps_the_indexer_layers_and_only_those() {
    // Behaviour, not a constructor: `index_topk: None` must mean "no cap", which is what a
    // ratio-128 layer runs (`get_compress_topk_idxs` has no cap at all). Carrying a cap
    // there would stop attending blocks past `128 * 513` positions, fluently.
    //
    // A SMALL cap, because the property needs a length where a cap would actually bite and
    // the shipped 512 needs a 2052-token fixture to reach — the coverage cliff recorded in
    // v4-flash-port.md. The arithmetic is the same at 3 as at 512.
    let mut flat = Vec::new();
    let capped = CompressSel { ratio: 4, index_topk: Some(3) };
    let uncapped = CompressSel { ratio: 4, index_topk: None };

    // Below the cap the two agree, which is what makes the pair above it meaningful:
    // 3 blocks exist at start_pos 11 (`(11+1)/4`), and the cap is 3.
    assert_eq!(Sel { win: 8, comp: Some(capped), seqlen: 1, start_pos: 11 }.shape().1, 8 + 3);
    assert_eq!(Sel { win: 8, comp: Some(uncapped), seqlen: 1, start_pos: 11 }.shape().1, 8 + 3);

    // Above it they must diverge: 4 blocks exist at start_pos 15.
    assert_eq!(
        Sel { win: 8, comp: Some(capped), seqlen: 1, start_pos: 15 }.shape().1,
        8 + 3,
        "index_topk did not truncate where it must"
    );
    assert_eq!(
        Sel { win: 8, comp: Some(uncapped), seqlen: 1, start_pos: 15 }.shape().1,
        8 + 4,
        "`None` applied a cap — a ratio-128 layer would lose blocks at long context"
    );

    // ...and the buffer the shape describes is the buffer that gets filled. A `shape()`
    // that agreed with nothing would satisfy every assertion above.
    flat.clear();
    let sel = Sel { win: 8, comp: Some(capped), seqlen: 1, start_pos: 15 };
    let (rows, cols) = v4_topk_idxs(sel, &mut flat);
    assert_eq!((rows, cols), sel.shape());
    assert_eq!(flat.len(), rows * cols);
}
