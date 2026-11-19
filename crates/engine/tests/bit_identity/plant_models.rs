//! The CPU halves of Phase 1 L3's red proofs (`docs/measurement/gate-red-proofs.md` §18, its
//! "Phase 1 L3" paragraph): each launcher-row plant under `tests/bit-identity/plants/launch/` is
//! shown ADVERSARIAL on the gate's own inputs here, deviceless, before its device arm runs — the
//! spike's first plant passed at 8 seeds (§1 of `rust-kernels.md`), so a plant that moves no cell
//! on the drawn inputs proves nothing.

use super::cells::{
    GEMM_SHAPES, GEOMS, MLA_DIMS, MlaShape, PREFILL_SHAPES, SCORE_BLOCKS_DIMS, gemm_ins, geom,
    mla_ins, prefill_ins, score_blocks_ins,
};
use super::inputs::{SEEDS, TOPK_SHAPES, topk_scores};
use rivoli_backend::abi::CompGeom;
use rivoli_core::num::{bf16_to_f32, e4m3_to_f32, f32_to_bf16};

/// A plant's verdict: print its rate, and refuse it below `num/den` of the cells — the demand each
/// test argues for its own plant, never a round number that merely passes.
fn demand(plant: &str, (moved, cells): (usize, usize), (num, den): (usize, usize)) {
    println!("{plant} plant: {moved}/{cells} cells move");
    assert!(
        moved * den >= cells * num,
        "{plant} plant moved only {moved}/{cells} cells — sub-threshold"
    );
}

/// Over `shapes` × every seed, how many cells `moves(shape, seed)` says the plant moves, printing
/// each shape's count: `(moved, cells)`.
fn rate<S: Copy + std::fmt::Debug>(
    plant: &str,
    shapes: impl IntoIterator<Item = S>,
    moves: impl Fn(S, u64) -> bool,
) -> (usize, usize) {
    shapes.into_iter().fold((0, 0), |(moved, cells), shape| {
        let m = (1..=SEEDS).filter(|&seed| moves(shape, seed)).count();
        println!("{plant} at {shape:?}: {m}/{SEEDS} seeds move");
        (moved + m, cells + SEEDS as usize)
    })
}

/// Whether the clean arm (`arm(true)`) and the planted one (`arm(false)`) part, bit for bit.
fn splits(arm: impl Fn(bool) -> f32) -> bool {
    arm(true).to_bits() != arm(false).to_bits()
}

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
    let plant = "index_topk signed-zero (partial selections)";
    let partial = TOPK_SHAPES.into_iter().filter(|(nt, k)| k < nt);
    let r = rate(plant, partial, |(nt, k), seed| {
        let s = topk_scores(nt, seed);
        select(&s, k, true) != select(&s, k, false)
    });
    demand(plant, r, (1, 3));
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
    let plant = "index_score_blocks per-term";
    let r = rate(plant, SCORE_BLOCKS_DIMS, |dims, seed| {
        let ins = score_blocks_ins(dims, seed);
        let n_comp = dims[1];
        (0..dims[0] * n_comp).any(|id| {
            let tc = (id / n_comp, id % n_comp);
            splits(|clean| score_cell(&ins, dims, tc, clean))
        })
    });
    demand(plant, r, (3, 4));
}

/// One `kv_compress_prefill` output row on the CPU up to the RoPE — the pool, the bf16 store and
/// the norm — as its rope pairs `[x, y, c, s]`. The host's `exp` and a sequential norm sum stand
/// in for the device's, so this measures a rate on these operands, not the device's cells.
fn prefill_pairs(ins: &[Vec<f32>], p: &CompGeom, b: usize) -> Vec<[f32; 4]> {
    let (ratio, cd, d, rd) = (p.ratio as usize, p.cd as usize, p.d as usize, p.rd as usize);
    let (kv, sc, ape, norm, freqs) = (&ins[0], &ins[1], &ins[2], &ins[3], &ins[4]);
    // `overlap_transform`'s index map: entry j's (source offset, ape offset), or masked.
    let entry = |j: usize| match (p.coff == 2, j >= ratio, b > 0) {
        (false, ..) => Some(((b * ratio + j) * cd, j * cd)),
        (true, true, _) => Some(((b * ratio + j - ratio) * cd + d, (j - ratio) * cd + d)),
        (true, false, true) => Some((((b - 1) * ratio + j) * cd, j * cd)),
        (true, false, false) => None,
    };
    let live: Vec<(usize, usize)> = (0..p.ents as usize).filter_map(entry).collect();
    let row: Vec<f32> = (0..d)
        .map(|e| {
            let score = |&(src, a): &(usize, usize)| sc[src + e] + ape[a + e];
            let mx = live.iter().map(score).fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = live.iter().map(|x| (score(x) - mx).exp()).sum();
            let pool = live.iter().fold(0.0f32, |acc, x| {
                kv[x.0 + e].mul_add((score(x) - mx).exp() / sum, acc)
            });
            rbf16(pool)
        })
        .collect();
    let ss = row.iter().fold(0.0f32, |a, &v| v.mul_add(v, a));
    let rs = 1.0 / (ss / d as f32 + p.eps).sqrt();
    let at = |k: usize| rbf16(norm[k] * (row[k] * rs));
    let t = &freqs[b * ratio * rd..];
    (0..rd / 2)
        .map(|i| {
            [
                at(d - rd + 2 * i),
                at(d - rd + 2 * i + 1),
                t[2 * i],
                t[2 * i + 1],
            ]
        })
        .collect()
}

/// NOT a plant's proof — a SENSITIVITY measurement: a compressor RoPE that fuses the wrong product
/// of `a·s + b·c` (`fma(b, c, a·s)` against hipcc's `fma(a, s, b·c)`) mostly vanishes in the final
/// bf16 store. This port nearly shipped exactly that: an ISA misread "fixed" a correct draft
/// backwards, and the correctness review caught it on the IR before any device run. Measured
/// 2026-11-19: 10/384 prefill cells move, all at the 512-block hot shape (10/64), none at the small
/// ones. The rows CAN see it — `Tally::verdict` reddens a row on one differing word — and this
/// asserts at least one cell moves, but at the spike's sub-threshold rate. That is why the
/// `kvcompress` plant is `gemm_bf16`'s, and why the RoPE sites rest on the ISA reading.
#[test]
fn compress_rope_fusion_is_visible_to_the_prefill_rows() {
    let (moved, cells) = rate(
        "kv_compress_prefill rope mis-fusion",
        PREFILL_SHAPES,
        |(gi, nblk), seed| {
            let p = geom(GEOMS[gi].0, GEOMS[gi].1);
            let ins = prefill_ins(&p, nblk, seed);
            (0..nblk).any(|b| {
                prefill_pairs(&ins, &p, b).iter().any(|&[x, y, c, s]| {
                    rbf16(x.mul_add(s, y * c)).to_bits() != rbf16(y.mul_add(c, x * s)).to_bits()
                })
            })
        },
    );
    println!("kv_compress_prefill rope mis-fusion: {moved}/{cells} cells differ");
    assert!(moved > 0, "no prefill cell can see a RoPE mis-fusion");
}

/// `gemm_bf16`'s output `o` on the CPU: 32 lanes each fold their strided share (`fused` = hipcc's
/// `v_fmac`; `false` = the `kvcompress` plant, product then sum), then `wave_sum`'s shuffle-down
/// ladder into lane 0.
fn gemm_out(ins: &[Vec<f32>], [_, n, k]: [usize; 3], o: usize, fused: bool) -> f32 {
    let (r, c) = (o / n, o % n);
    let w = |i: usize| {
        let j = c * k + i;
        bf16_to_f32((ins[1][j / 2].to_bits() >> (16 * (j % 2))) as u16)
    };
    let mut lanes: Vec<f32> = (0..32)
        .map(|l| {
            (l..k).step_by(32).fold(0.0f32, |acc, i| {
                let x = ins[0][r * k + i];
                if fused {
                    x.mul_add(w(i), acc)
                } else {
                    acc + x * w(i)
                }
            })
        })
        .collect();
    for off in [16, 8, 4, 2, 1] {
        lanes = (0..32)
            .map(|l| {
                lanes[l]
                    + if l + off < 32 {
                        lanes[l + off]
                    } else {
                        lanes[l]
                    }
            })
            .collect();
    }
    lanes[0]
}

/// The `kvcompress` plant (the GEMM's dot accumulator unfused) moves nearly every cell: the output
/// is f32 with no rounding store after it. Measured 2026-11-19: 64/64 seeds at the four real
/// shapes, 10/64 at the ragged `[3, 7, 33]` (one or two products per lane), 266/320 overall.
#[test]
fn gemm_unfused_plant_is_adversarial_on_cpu() {
    let unfused = |dims: [usize; 3], seed| {
        let ins = gemm_ins(dims, seed);
        (0..dims[0] * dims[1]).any(|o| splits(|clean| gemm_out(&ins, dims, o, clean)))
    };
    demand(
        "gemm_bf16 unfused",
        rate("gemm_bf16 unfused", GEMM_SHAPES, unfused),
        (3, 4),
    );
}

/// `mla_absorb_fp8`'s output `(r, head, i)` on the quad path: `Σ_d fma(q, lut·s, acc)`, the
/// product `lut·s` rounded first (hipcc's hoist, the C's parenthesisation); `hoisted == false` is
/// the `mla` plant, `fma(q·lut, s, acc)`.
fn absorb_out(
    ins: &[Vec<f32>],
    (dims, _): MlaShape,
    (r, head, i): (usize, usize, usize),
    hoisted: bool,
) -> f32 {
    let [h, qh, nope, vh, kvl, block] = dims;
    let byte = |j: usize| (ins[1][j / 4].to_bits() >> (8 * (j % 4))) as u8;
    let rbase = head * (nope + vh);
    (0..nope).fold(0.0f32, |acc, d| {
        let l = e4m3_to_f32(byte((rbase + d) * kvl + i));
        let s = ins[2][((rbase + d) / block) * kvl.div_ceil(block) + i / block];
        let q = ins[0][r * h * qh + head * qh + d];
        if hoisted {
            q.mul_add(l * s, acc)
        } else {
            (q * l).mul_add(s, acc)
        }
    })
}

/// The `mla` plant (the absorb's hoisted `lut · s` re-associated) moves every quad-path cell: the
/// output is f32, unrounded, and each sums `nope` (128-192) products. The two scalar-fallback
/// shapes cannot move — the patch touches only the quad path — and are excluded by construction.
#[test]
fn absorb_reassociation_plant_is_adversarial_on_cpu() {
    let plant = "mla_absorb_fp8 re-association (quad path)";
    let quads = MLA_DIMS
        .into_iter()
        .filter(|(d, _)| d[4] % 4 == 0 && d[5] >= 4);
    let r = rate(plant, quads, |shape, seed| {
        let ([h, qh, _, _, kvl, _], nrow) = shape;
        let ins = mla_ins(shape, nrow * h * qh, seed);
        (0..nrow * h * kvl).any(|o| {
            let at = (o / (h * kvl), o / kvl % h, o % kvl);
            splits(|clean| absorb_out(&ins, shape, at, clean))
        })
    });
    demand(plant, r, (9, 10));
}
