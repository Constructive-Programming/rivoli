//! The CPU half of the linalg, fwd and headtail red proofs — one plant per family, each a
//! patch under `tests/bit-identity/plants/` — shown ADVERSARIAL on the gate's own inputs
//! before any device arm runs (the rope model's rule in `main.rs`, and §1's lesson: a plant
//! must move outputs on the draws the gate actually makes, or its green says nothing).
//!
//! Host `exp` and an f64 stand-in for the gate's mixes are not the device's bits, so each
//! model measures the plant's RATE on these operands; the device arm is the measurement. The
//! shapes are a subset of each row's sweep, drawn on the row's own streams.

use super::inputs::{SEEDS, bytes, draw, uniform};

/// `wave_sum`'s shuffle-down tree on lane 0: each step every lane adds the lane `o` above it
/// (or itself, past the wave's end), `o = 16, 8, 4, 2, 1`.
fn wave_sum(mut v: [f32; 32]) -> f32 {
    for o in [16, 8, 4, 2, 1] {
        v = std::array::from_fn(|l| v[l] + v[if l + o < 32 { l + o } else { l }]);
    }
    v[0]
}

/// One wave-per-row dot as `lane_cols` computes it: lane `l` folds columns `l, l + 32, …`
/// fused (the shipped kernel) or as `x·w + acc` (the plant), then `wave_sum`.
fn lane_dot(x: &[f32], w: impl Fn(usize) -> f32, fused: bool) -> f32 {
    wave_sum(std::array::from_fn(|lane| {
        (lane..x.len()).step_by(32).fold(0.0f32, |acc, i| {
            if fused {
                x[i].mul_add(w(i), acc)
            } else {
                x[i] * w(i) + acc
            }
        })
    }))
}

/// `(cells moved, cells)` over the seeds, a cell moving when `differs(seed)`.
fn rate(differs: impl Fn(u64) -> bool) -> (u64, u64) {
    ((1..=SEEDS).filter(|&s| differs(s)).count() as u64, SEEDS)
}

/// §1's bar: the gate reddens on one word, so demand most cells move.
fn assert_adversarial(what: &str, (moved, cells): (u64, u64)) {
    println!("{what}: {moved}/{cells} cells moved");
    assert!(
        moved * 10 >= cells * 9,
        "{what}: plant moved only {moved}/{cells} cells — sub-threshold"
    );
}

/// `linalg.patch`: `lane_cols` without its fma, on `gemv_f32`'s (160, 6144) and `gemv_i8`'s
/// misaligned rows of (9, 1001) — the two rows the plant must redden.
#[test]
fn linalg_lane_column_plant_is_adversarial_on_cpu() {
    let (o, i) = (160, 6144);
    assert_adversarial(
        "gemv_f32 (160, 6144) rows 0..8",
        rate(|seed| {
            let x = draw(i, (seed, 1), 1.0, true);
            let w = draw(o * i, (seed, 2), 0.1, true);
            (0..8).any(|r| {
                let row = &w[r * i..(r + 1) * i];
                let dot = |fused| lane_dot(&x, |c| row[c], fused).to_bits();
                dot(true) != dot(false)
            })
        }),
    );
    // `gemv_i8`'s scalar-tail work in its sweep: rows that start off a dword boundary, and
    // every row at R = 2 with `i_dim % 4 != 0` (the quads need each `x` row float4-aligned).
    assert_adversarial(
        "gemv_i8 scalar-tail rows",
        rate(|seed| {
            [(9, 1001, 1), (9, 1001, 2), (5, 130, 2), (3, 127, 1)]
                .iter()
                .any(|&(o, i, n)| i8_tail_moves(seed, (o, i, n)))
        }),
    );
}

/// Whether the lane-column plant moves any output of `gemv_i8` at `(o, i, nrow)` that takes
/// the scalar path from column 0.
fn i8_tail_moves(seed: u64, (o, i, n): (usize, usize, usize)) -> bool {
    let x = draw(n * i, (seed, 1), 1.0, true);
    let w = bytes(o * i, (seed, 2));
    let scalar = |r: usize| n == 2 && !i.is_multiple_of(4) || !(r * i).is_multiple_of(4);
    (0..o).filter(|&r| scalar(r)).any(|r| {
        let code = |c: usize| f32::from(w[r * i + c] as i8);
        x.chunks(i)
            .any(|xr| lane_dot(xr, code, true).to_bits() != lane_dot(xr, code, false).to_bits())
    })
}

/// `fwd.patch`: `sigmoid_gate` as `x / (1 + e^-g)` instead of `x · (1 / (1 + e^-g))`.
#[test]
fn fwd_sigmoid_division_plant_is_adversarial_on_cpu() {
    for n in [257, 2048, 6144] {
        assert_adversarial(
            &format!("sigmoid_gate n={n}"),
            rate(|seed| {
                let x = draw(n, (seed, 1), 1.0, true);
                let g = draw(n, (seed, 2), 4.0, true);
                x.iter().zip(&g).any(|(&x, &g)| {
                    let d = 1.0 + (-g).exp();
                    (x * (1.0 / d)).to_bits() != (x / d).to_bits()
                })
            }),
        );
    }
}

/// `headtail.patch`: the gate's strided dot (`lane_dot`, both the sum of squares and the
/// mixes) without its fma — the gate modelled in f32 as the kernel computes it, so `rs`, `m`
/// and `pre` all carry the plant. The sigmoid swallows a few-ulp move in `m` often (one token
/// at a time it moves ~3 cells in 4), so a cell is a seed over the row's two V4 shapes.
#[test]
fn headtail_gate_dot_plant_is_adversarial_on_cpu() {
    assert_adversarial(
        "hc_head_collapse (1|3, 4, 4096)",
        rate(|seed| [1, 3].iter().any(|&s| gate_moves(seed, (s, 4, 4096)))),
    );
}

/// Whether the gate-dot plant moves any `pre` of `hc_head_collapse` at `(s, hc, dim)`.
fn gate_moves(seed: u64, (s, hc, dim): (usize, usize, usize)) -> bool {
    let h = draw(s * hc * dim, (seed, 1), 1.0, true);
    let fnw = draw(hc * hc * dim, (seed, 2), 0.05, false);
    let base = draw(hc, (seed, 3), 1.0, false);
    let sc = uniform(1, (seed, 4), 0.5..2.0)[0];
    let gate = |flat: &[f32], fused| -> Vec<u32> {
        let ss = lane_dot(flat, |i| flat[i], fused);
        let rs = 1.0 / (ss / flat.len() as f32 + 1e-6).sqrt();
        (0..hc)
            .map(|j| {
                let wj = &fnw[j * flat.len()..];
                let m = lane_dot(flat, |i| wj[i], fused) * rs;
                (1.0 / (1.0 + (-m.mul_add(sc, base[j])).exp()) + 1e-6).to_bits()
            })
            .collect()
    };
    h.chunks(hc * dim)
        .any(|flat| gate(flat, true) != gate(flat, false))
}
