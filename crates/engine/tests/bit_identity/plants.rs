//! The CPU halves of the launcher rows' recorded red proofs (Phase 0a's rope plant, and each
//! Phase 1 family's `tests/bit-identity/plants/<family>.patch`), modelled on the host over the
//! gate's own draws,
//! so "the plant is adversarial" is measured before a device arm runs (§1's lesson — the
//! spike's first plant was sub-threshold). Host `exp`/`sqrt` differ from the device's in the
//! last bits, so each model measures the plant's RATE on these operands, not the device's
//! exact cells; the device arm is the measurement.

use super::inputs::{SEEDS, draw, rope_at, uniform};

/// The gate demands a plant move at least 90% of seeds, so it cannot hide in a lucky draw.
fn assert_adversarial(what: &str, moved: u64, cells: u64) {
    println!("{what}: {moved}/{cells} seeds affected");
    assert!(
        moved * 10 >= cells * 9,
        "{what} moved only {moved}/{cells} seeds — sub-threshold"
    );
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `x / sqrt(Σx² + 1e-6)`, the delta rules' L2 norm, times `scale`.
fn l2(x: &[f32], scale: f32) -> Vec<f32> {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
    x.iter().map(|v| v * inv * scale).collect()
}

/// One head's normalized `q`/`k`, per-channel decay, update gate, and state row stride.
struct Head {
    stride: usize,
    qn: Vec<f32>,
    kn: Vec<f32>,
    alpha: Vec<f32>,
    beta: f32,
}

/// One state column through both passes (the kernel's `state_passes`), the rank-one update
/// fused or not; returns `o`.
fn column(col: &mut [f32], hd: &Head, v: f32, fused: bool) -> f32 {
    let stride = hd.stride;
    let mut u = 0.0f32;
    for (i, a) in hd.alpha.iter().enumerate() {
        col[i * stride] *= a;
        u = hd.kn[i].mul_add(col[i * stride], u);
    }
    let dv = hd.beta * (v - u);
    let mut o = 0.0f32;
    for (i, (k, q)) in hd.kn.iter().zip(&hd.qn).enumerate() {
        let s = col[i * stride];
        col[i * stride] = if fused { k.mul_add(dv, s) } else { s + k * dv };
        o = q.mul_add(col[i * stride], o);
    }
    o
}

/// `gated_delta_recurrent_f32` at the gate's (4, 32) cell on the row's streams 1-8, with
/// `lower_bound` fixed at -5 (the row's even-seed value) and without the row's one-in-eight
/// NaN gate, so every head of every seed can move: the rank-one update fused
/// (`fma(k, dv, s)`, hipcc) or not (`s + k·dv`, the `recurrent` plant). Returns `(out, state)`.
fn gated_delta_model(seed: u64, fused: bool) -> (Vec<f32>, Vec<f32>) {
    let (h, d) = (4, 32);
    let n = h * d;
    let [q, k, v] = [1, 2, 3].map(|s| draw(n, (seed, s), 1.0, true));
    let g = draw(n, (seed, 4), 8.0, true);
    let (bp, al) = (
        draw(h, (seed, 5), 4.0, false),
        uniform(h, (seed, 6), -3.0..2.0),
    );
    let dt = draw(n, (seed, 7), 1.0, false);
    let mut state = draw(h * d * d, (seed, 8), 1.0, true);
    let mut out = vec![0.0f32; n];
    for hh in 0..h {
        let r = hh * d..(hh + 1) * d;
        let z = |t: usize| al[hh].exp() * (g[hh * d + t] + dt[hh * d + t]);
        let head = Head {
            stride: d,
            qn: l2(&q[r.clone()], 1.0 / (d as f32).sqrt()),
            kn: l2(&k[r], 1.0),
            alpha: (0..d).map(|t| (-5.0 * sigmoid(z(t))).exp()).collect(),
            beta: sigmoid(bp[hh]),
        };
        for t in 0..d {
            let col = &mut state[hh * d * d + t..];
            out[hh * d + t] = column(col, &head, v[hh * d + t], fused);
        }
    }
    (out, state)
}

/// `recurrent.patch` unfuses the rank-one update both gated delta rules share, so its device
/// arm is `--expect-red=gated_delta_recurrent_f32,gdn_recurrent_f32` on a binary built with the
/// patch applied to `crates/kernels`, with `short_conv_silu_f32` and `rmsnorm_gate_heads_f32`
/// holding.
#[test]
fn recurrent_unfused_update_plant_is_adversarial_on_cpu() {
    let moved = (1..=SEEDS)
        .filter(|&seed| gated_delta_model(seed, true) != gated_delta_model(seed, false))
        .count() as u64;
    assert_adversarial("recurrent unfused-update plant", moved, SEEDS);
}

/// The CPU half of the recorded red proof (`docs/measurement/gate-red-proofs.md` §18): the
/// planted defect is `rope`'s `lo` without its FMA — `x·cs − y·sn` rounded twice instead of
/// once — and this shows it is ADVERSARIAL on the gate's own rope inputs before any device
/// arm runs. §1's lesson is why the order matters: the spike's first plant was
/// sub-threshold and passed at 8 seeds, so a plant must be shown to move outputs on the
/// inputs the gate actually draws, or a green run says nothing.
///
/// `cos`/`sin` here are the host's, not OCML's — the device values differ in the last bits,
/// so this measures the plant's RATE on these operands, not the device's exact cells. A rate
/// far from zero is the claim; the device arm is the measurement.
#[test]
fn rope_unfused_plant_is_adversarial_on_cpu() {
    let (mut cells, mut moved, mut elems, mut diff) = (0u64, 0u64, 0usize, 0usize);
    for seed in 1..=SEEDS {
        let (pos, theta) = rope_at(seed);
        let v = draw(32 * 128, (seed, 1), 1.0, true);
        let before = diff;
        for head in v.chunks_exact(128) {
            for j in 0..64 {
                let (x, y) = (head[2 * j], head[2 * j + 1]);
                let ang = pos as f64 * theta.powf(-2.0 * j as f64 / 128.0);
                let (cs, sn) = (ang.cos() as f32, ang.sin() as f32);
                elems += 1;
                diff +=
                    usize::from(x.mul_add(cs, -(y * sn)).to_bits() != (x * cs - y * sn).to_bits());
            }
        }
        cells += 1;
        moved += u64::from(diff > before);
    }
    println!("rope unfused plant: {diff}/{elems} lo outputs differ");
    assert_adversarial("rope unfused plant", moved, cells);
}
