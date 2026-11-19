//! The CPU halves of Phase 1 L3's red proofs (`docs/measurement/gate-red-proofs.md` §18, its
//! "Phase 1 L3" paragraph): each launcher-row plant under `tests/bit-identity/plants/launch/` is
//! shown ADVERSARIAL on the gate's own inputs here, deviceless, before its device arm runs — the
//! spike's first plant passed at 8 seeds (§1 of `rust-kernels.md`), so a plant that moves no cell
//! on the drawn inputs proves nothing.

use super::inputs::{SCORE_BLOCKS_DIMS, SEEDS, TOPK_SHAPES, score_blocks_ins, topk_scores};
use rivoli_core::num::{bf16_to_f32, f32_to_bf16};

/// `index_topk`'s order-preserving key; `canonical == false` is the `indexer` plant (the `-0.0`
/// canonicalisation dropped, so `-0.0` orders strictly below `+0.0` instead of tying with it).
fn topk_key(v: f32, canonical: bool) -> u32 {
    let b = if canonical && v == 0.0 {
        0
    } else {
        v.to_bits()
    };
    b ^ (((b as i32) >> 31) as u32 | 0x8000_0000)
}

/// The kernel's selection, as its contract states it: score descending, index ascending, the first
/// `min(k, n)`, emitted in index order.
fn select(s: &[f32], k: usize, canonical: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..s.len()).collect();
    idx.sort_by_key(|&i| (std::cmp::Reverse(topk_key(s[i], canonical)), i));
    idx.truncate(k.min(s.len()));
    idx.sort_unstable();
    idx
}

/// The `indexer` plant moves the selected row set on most tie-heavy cells. Only cells with
/// `k < nt` can move at all — the clamp shapes select every entry, which no tie rule changes —
/// seed class 0 is untied by design, and at `(65536, 2048)` the boundary sits inside the POSITIVE
/// ties (it scores the index tiebreak among equal positives, which this plant leaves alone). So
/// the demand is a third of the partial-selection cells (measured 2026-11-19: 126/256, 0 at that
/// shape and 30–48 of 64 at the other three) — far from sub-threshold, where the gate reddens on
/// one differing word.
#[test]
fn topk_signed_zero_plant_is_adversarial_on_cpu() {
    let (mut cells, mut moved) = (0usize, 0usize);
    for &(nt, k) in TOPK_SHAPES.iter().filter(|(nt, k)| k < nt) {
        let m = (1..=SEEDS)
            .filter(|&seed| {
                let s = topk_scores(nt, seed);
                select(&s, k, true) != select(&s, k, false)
            })
            .count();
        println!("index_topk signed-zero plant at ({nt}, {k}): {m}/{SEEDS} seeds move");
        cells += SEEDS as usize;
        moved += m;
    }
    println!("index_topk signed-zero plant: {moved}/{cells} partial-selection cells move");
    assert!(
        moved * 3 >= cells,
        "plant moved only {moved}/{cells} cells — sub-threshold"
    );
}

fn rbf16(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

/// `index_score_blocks`' output cell `(t, c)`; `per_term == false` is the `blockindex` plant (the
/// per-head term no longer rounded to bf16 before it joins the f32 accumulator — the
/// `rbf16(acc + …)`-class mistake the C's note warns against, moved one rounding).
fn score_cell(
    ins: &[Vec<f32>],
    [_, _, heads, hd]: [usize; 4],
    (t, c): (usize, usize),
    per_term: bool,
) -> f32 {
    let (q, kv, w) = (&ins[0], &ins[1], &ins[2]);
    let acc = (0..heads).fold(0.0f32, |acc, h| {
        let dot = (0..hd).fold(0.0f32, |d, i| {
            d + q[(t * heads + h) * hd + i] * kv[c * hd + i]
        });
        let term = rbf16(dot).max(0.0) * w[t * heads + h];
        acc + if per_term { rbf16(term) } else { term }
    });
    rbf16(acc)
}

/// The `blockindex` plant moves most scorer cells. One differing output word reddens a cell; the
/// single-block shape `[1, 1, 64, 128]` has ONE output per cell and moves least (measured
/// 2026-11-19: 203/256 cells overall), so the demand is three quarters — far from the spike's
/// 3/256 sub-threshold plant.
#[test]
fn score_blocks_per_term_plant_is_adversarial_on_cpu() {
    let (mut cells, mut moved) = (0usize, 0usize);
    for dims in SCORE_BLOCKS_DIMS {
        let [s, n_comp, ..] = dims;
        let m = (1..=SEEDS)
            .filter(|&seed| {
                let ins = score_blocks_ins(dims, seed);
                (0..s * n_comp).any(|id| {
                    let tc = (id / n_comp, id % n_comp);
                    let (a, b) = (
                        score_cell(&ins, dims, tc, true),
                        score_cell(&ins, dims, tc, false),
                    );
                    a.to_bits() != b.to_bits()
                })
            })
            .count();
        println!("index_score_blocks per-term plant at {dims:?}: {m}/{SEEDS} seeds move");
        cells += SEEDS as usize;
        moved += m;
    }
    println!("index_score_blocks per-term plant: {moved}/{cells} cells differ");
    assert!(
        moved * 4 >= cells * 3,
        "plant moved only {moved}/{cells} cells — sub-threshold"
    );
}
