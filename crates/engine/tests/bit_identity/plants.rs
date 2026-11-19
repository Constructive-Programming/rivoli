//! The CPU halves of the launcher rows' recorded red proofs (Phase 0a's rope plant, and each
//! Phase 1 family's `tests/bit-identity/plants/<family>.patch`), modelled on the host over the
//! gate's own draws,
//! so "the plant is adversarial" is measured before a device arm runs (§1's lesson — the
//! spike's first plant was sub-threshold). Host `exp`/`sqrt` differ from the device's in the
//! last bits, so each model measures the plant's RATE on these operands, not the device's
//! exact cells; the device arm is the measurement.

use super::inputs::{SEEDS, bytes, draw, rope_at, uniform};

/// One modelled output row: its seed, its reduction dim, and its index.
#[derive(Clone, Copy)]
struct RowAt {
    seed: u64,
    dim: usize,
    row: usize,
}

/// Whether the rank-one update is hipcc's fused `fma(k, dv, s)` or the plant's `s + k·dv`.
#[derive(Clone, Copy, PartialEq)]
enum Update {
    Fused,
    Split,
}

/// A per-row plant model over 64 rows of width `dim` per seed: `moves(row)` says whether the plant
/// changes that row's output; prints the row rate and demands the seed rate.
fn row_plant(what: &str, dim: usize, moves: impl Fn(RowAt) -> bool) {
    let (mut moved, mut rows) = (0u64, 0usize);
    for seed in 1..=SEEDS {
        let d = (0..64)
            .filter(|&row| moves(RowAt { seed, dim, row }))
            .count();
        rows += d;
        moved += u64::from(d > 0);
    }
    println!("{what}: {rows}/{} rows differ", 64 * SEEDS);
    assert_adversarial(what, moved, SEEDS);
}

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
fn column(col: &mut [f32], hd: &Head, v: f32, update: Update) -> f32 {
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
        col[i * stride] = match update {
            Update::Fused => k.mul_add(dv, s),
            Update::Split => s + k * dv,
        };
        o = q.mul_add(col[i * stride], o);
    }
    o
}

/// `gated_delta_recurrent_f32` at the gate's (4, 32) cell on the row's streams 1-8, with
/// `lower_bound` fixed at -5 (the row's even-seed value) and without the row's one-in-eight
/// NaN gate, so every head of every seed can move: the rank-one update fused
/// (`fma(k, dv, s)`, hipcc) or not (`s + k·dv`, the `recurrent` plant). Returns `(out, state)`.
fn gated_delta_model(seed: u64, update: Update) -> (Vec<f32>, Vec<f32>) {
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
            out[hh * d + t] = column(col, &head, v[hh * d + t], update);
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
        .filter(|&seed| {
            gated_delta_model(seed, Update::Fused) != gated_delta_model(seed, Update::Split)
        })
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

/// fp16 bits → f32 (the codebook's storage), every class.
fn f16f(h: u16) -> f32 {
    let (e, m) = (i32::from((h >> 10) & 0x1f), f32::from(h & 0x3ff));
    let mag = match e {
        0 => m * 2f32.powi(-24),
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    };
    if h & 0x8000 != 0 { -mag } else { mag }
}

/// A scale draw as the `moe` rows store it: `v · 2^-6`, truncated to bf16.
fn bf16_scale(v: f32) -> f32 {
    f32::from_bits((v * 2f32.powi(-6)).to_bits() & 0xffff_0000)
}

/// One VQ-int3 gate dot on the `moe` rows' operand streams for token row 1 of an R=2 cell —
/// row 0 carries the nasty seeds' NaN (`nan_some`), row 1 never does: `x` stream 1, index arena
/// stream 3, bf16 scales stream 4, f16 gate codebook stream 10. `Σ_t s_t · ⟨x[4t..], cb[idx_t]⟩`
/// with 12-bit `idx_t` read at row `j`, summed sequentially rather than in the wave's fold: the
/// model measures the plant's rate on these operands, not the device's cells.
fn vq_gate_dot(
    RowAt {
        seed,
        dim: hidden,
        row: j,
    }: RowAt,
) -> f32 {
    let x = draw(2 * hidden, (seed, 1), 1.0, true).split_off(hidden);
    let idx = bytes(hidden / 4 * 12 / 8 * (j + 1) + 2, (seed, 3));
    let cb: Vec<f32> = draw(4096 * 4, (seed, 10), 1.0, false)
        .iter()
        .map(|&v| f16f(rivoli_core::num::f32_to_f16(v)))
        .collect();
    let scale = uniform(hidden / 64 * (j + 1), (seed, 4), 0.25..2.0);
    let row = j * (hidden / 4 * 12 / 8);
    (0..hidden / 4)
        .map(|t| {
            let at = row + t * 12 / 8;
            let raw = u32::from(idx[at]) | (u32::from(idx[at + 1]) << 8);
            let c = ((raw >> ((t * 12) & 7)) & 0xfff) as usize * 4;
            let dot: f32 = (0..4).map(|k| x[4 * t + k] * cb[c + k]).sum();
            bf16_scale(scale[j * (hidden / 64) + t / 16]) * dot
        })
        .sum()
}

/// `moe.patch` writes the range kernels' SiLU in the multiply form, `g·(1/(1+e^-g))`, against
/// hipcc's division form — the form hazard `activation.hip` documents and the spike measured
/// (~20% of outputs). It reaches both GLM range rows (`moe_expert_range`,
/// `moe_expert_range_i4`) through every `h` they write, and neither drain: the device arm is
/// `--expect-red=moe_expert_range,moe_expert_range_i4`. Modelled on 64 gate rows of the
/// (192, 64, ·, ·, 2) cells' clean token row per seed.
#[test]
fn moe_silu_multiply_form_plant_is_adversarial_on_cpu() {
    let silu_div = |g: f32| g / (1.0 + (-g).exp());
    let silu_mul = |g: f32| g * (1.0 / (1.0 + (-g).exp()));
    row_plant("moe silu multiply-form plant", 192, |at| {
        let g = vq_gate_dot(at);
        silu_div(g).to_bits() != silu_mul(g).to_bits()
    });
}

/// bf16 round trip, round-to-nearest-even — `rivoli_core`'s, which matches the kernels' on the
/// finite values this model feeds it.
fn rbf16(x: f32) -> f32 {
    rivoli_core::num::bf16_to_f32(rivoli_core::num::f32_to_bf16(x))
}

/// Expert `e`'s projection `p` (gate, up, down) at output `row`, reducing over `dim`.
#[derive(Clone, Copy)]
struct ProjRow {
    e: usize,
    p: usize,
    row: usize,
    dim: usize,
}

/// Which K3 cell: its range and seed. (`CellAt` and `Activ` are what took this file from
/// CodeScene 9.68 to 10 — measured with each removed, primitive obsession — not decoration.)
#[derive(Clone, Copy)]
struct CellAt {
    range: super::ranges::Range,
    seed: u64,
}

/// The activation a projection row reduces: `x` for gate/up, `h` for down.
#[derive(Clone, Copy)]
struct Activ<'a>(&'a [f32]);

/// `moe_fixed`'s saturation bound, `MOE_ACC_MAX` = 2^(58 − 44) in the kernels' prelude.
const MOE_ACC_MAX: f32 = 16384.0;

/// One K3 range cell as the `moe_f4` row builds it (`ranges::cell_inputs`: the same arenas,
/// scale codes, routing weights, per-expert descriptor shifts and betas) through both passes,
/// summed sequentially rather than in the wave's fold — so `dv` is a proxy for the device's
/// bits, close enough to measure whether a plant can move a cell. One token row only.
struct K3Cell {
    range: super::ranges::Range,
    lay: super::ranges::Layout,
    betas: (f32, f32),
    x: Vec<f32>,
    w: Vec<f32>,
    wb: Vec<u8>,
    sb: Vec<u8>,
    /// e2m1 nibble (sign in bit 3) and e8m0 byte (`2^(b−127)`, 0xff NaN) → f32.
    e2m1: [f32; 16],
    e8m0: [f32; 256],
}

impl K3Cell {
    fn new(CellAt { range, seed }: CellAt) -> Self {
        use super::ranges::{K3, cell_inputs, k3_betas};
        assert_eq!(range.4, 1, "K3Cell models one token row");
        let (lay, ins) = cell_inputs(&K3, range, seed);
        let (x, w) = (super::common::f32v(&ins[0]), super::common::f32v(&ins[1]));
        let [_, _, wb, sb]: [Vec<u8>; 4] = ins.try_into().expect("four range inputs");
        const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let e2m1 = std::array::from_fn(|n| MAG[n & 7] * if n & 8 != 0 { -1.0 } else { 1.0 });
        let e8m0 = std::array::from_fn(|b| match b {
            0xff => f32::NAN,
            _ => 2f32.powi(b as i32 - 127),
        });
        let betas = k3_betas(seed);
        Self {
            range,
            lay,
            betas,
            x,
            w,
            wb,
            sb,
            e2m1,
            e8m0,
        }
    }

    /// One row of one expert's projection against `v`: `Σ v[i]·e2m1·e8m0` over its `dim`.
    fn dot(&self, at: ProjRow, Activ(v): Activ) -> f32 {
        use super::ranges::{S_SHIFT, W_SHIFT};
        let ProjRow { e, p, row, dim } = at;
        let (wo, so) = (self.lay.w[p] + e * W_SHIFT, self.lay.s[p] + e * S_SHIFT);
        let (rb, ng) = (dim / 2, dim.div_ceil(32));
        (0..dim)
            .map(|i| {
                let nib = (self.wb[wo + row * rb + i / 2] >> (4 * (i & 1))) & 0xf;
                let scale = self.e8m0[usize::from(self.sb[so + row * ng + i / 32])];
                v[i] * self.e2m1[usize::from(nib)] * scale
            })
            .sum()
    }

    /// Whether the `moe_f4` plant (`bf16(dv·w)` for hipcc's `bf16(dv)·w`) changes any of the
    /// first 16 down rows' `moe_fixed` contributions of any live expert: NaN goes to the bound
    /// (`max`/`min` drop it, as `fmaxf`/`fminf` do), then `·2^44`, rounded to even.
    fn plant_moves(&self) -> bool {
        let (hidden, inter, es, ec, _) = self.range;
        let (b1, b2) = self.betas;
        // Not `clamp`: it passes a NaN through, where `moe_fixed`'s `fmaxf`/`fminf` send it to a
        // bound — which is the behaviour being modelled.
        #[allow(clippy::manual_clamp)]
        let fixed = |v: f32| {
            (v.max(-MOE_ACC_MAX).min(MOE_ACC_MAX) * 2f32.powi(44)).round_ties_even() as i64
        };
        (es..es + ec).filter(|&e| self.w[e] != 0.0).any(|e| {
            let base = ProjRow {
                e,
                p: 0,
                row: 0,
                dim: hidden,
            };
            let h: Vec<f32> = (0..inter)
                .map(|j| {
                    let gate = ProjRow { row: j, ..base };
                    let g = self.dot(gate, Activ(&self.x));
                    let u = self.dot(ProjRow { p: 1, ..gate }, Activ(&self.x));
                    rbf16(b1 * (g / b1).tanh() * sigmoid(g) * (b2 * (u / b2).tanh()))
                })
                .collect();
            (0..16).any(|o| {
                let dv = self.dot(
                    ProjRow {
                        p: 2,
                        row: o,
                        dim: inter,
                        ..base
                    },
                    Activ(&h),
                );
                let w = self.w[e];
                fixed(rbf16(dv) * w) != fixed(rbf16(dv * w))
            })
        })
    }
}

/// `moe_f4.patch` moves K3's pass-2 bf16 store after the routing weight — `bf16(dv·w)`, V4's
/// fold, against hipcc's `bf16(dv)·w` (the trap `moe_f4.hip`'s `WEIGHTED` note argues). It
/// reaches only `moe_down_f4_weighted`, so its device arm is
/// `--expect-red=moe_expert_range_f4_situ` with V4's `moe_expert_range_f4` holding.
///
/// Modelled on BOTH of K3's production shapes, on the cells the row builds (`K3Cell`), and
/// required to move ≥ 90% of the seeds where pass 2 is live, at EACH production shape. Pass 2
/// is dead on the nasty seeds (a NaN in `x`) and the edge seeds (non-finite gate/up scale
/// codes): one non-finite `h` makes every `dv` NaN on both sides there. Its first version modelled a stand-in `h` and claimed 64/64 while the
/// device reddened 0 cells at these shapes: the arenas then let every expert past the first
/// read a planted NaN scale into `h` (`ranges::Layout::new`'s note). The model now builds the
/// real cell, so it would have seen that.
#[test]
fn moe_f4_late_bf16_plant_reaches_k3_production_shapes() {
    for range in [(3584, 3072, 2, 1, 1), (3584, 3072, 0, 4, 1)] {
        let live = |s: u64| !super::inputs::nasty(s) && !super::ranges::edge_seed(s);
        let finite: Vec<u64> = (1..=SEEDS).filter(|&s| live(s)).collect();
        let moved = finite
            .iter()
            .filter(|&&s| K3Cell::new(CellAt { range, seed: s }).plant_moves())
            .count() as u64;
        assert_adversarial(
            &format!("moe_f4 late-bf16 plant at {range:?}"),
            moved,
            finite.len() as u64,
        );
    }
}
