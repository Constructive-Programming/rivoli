//! The V4 selection arithmetic, scored against S1b's oracle on the HOST.
//!
//! `src/attn.rs::v4_window_topk` decides which cache slots every query attends. It is a
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
#![allow(clippy::unwrap_used)]

use rivoli::attn::v4_window_topk;
use rivoli::v4oracle::forward::window_topk;

/// The engine's flat buffer, re-shaped into the oracle's row form so the two can be
/// compared as the same object.
fn engine_rows(win: usize, seqlen: usize, start_pos: usize) -> Vec<Vec<i64>> {
    let mut flat = Vec::new();
    let (rows, cols) = v4_window_topk(win, seqlen, start_pos, &mut flat);
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
    let (rows, cols) = v4_window_topk(1, 3, 0, &mut flat);
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
