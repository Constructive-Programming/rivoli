//! The gate's inputs: seeded, adversarial, deviceless — the plant model reads the same
//! draws the device arms do, so "the plant is adversarial on these inputs" is a statement
//! about THESE inputs and not a lookalike set.

use super::common::Lcg;

/// Seeds per (row, shape). §1's lesson: a no-fmuladd plant passed at 8 seeds and reddened
/// 3/256 draws at 64, so 64 is the floor, not a target.
pub const SEEDS: u64 = 64;

/// Values a kernel's arithmetic treats specially, planted at the front of every operand so
/// every seed carries all of them (and the rest is seeded, so they move between lanes and
/// waves as the shape changes):
///
/// - `±100`, `88.7`, `-87.4`: past and at the edges of `expf`'s finite range;
/// - `±0`: signed zero through divides, max/min and bf16 rounding;
/// - `±1e-40`: f32 subnormals, through every FMA and the e4m3 subnormal branch;
/// - bf16 round-to-nearest-even TIES (low half exactly `0x8000`, both parities) — the one
///   place `rbf16`'s `+ 0x7fff + lsb` decides;
/// - `448`, `464`, `-465`: e4m3's saturation boundary (`act_quant` scales first, so these
///   land on boundaries only for some groups — the seeded tail covers the rest).
pub const EDGES: [f32; 14] = [
    100.0,
    -100.0,
    88.7,
    -87.4,
    0.0,
    -0.0,
    1e-40,
    -1e-40,
    f32::from_bits(0x3f80_8000),
    f32::from_bits(0x3f81_8000),
    f32::from_bits(0xbf80_8000),
    448.0,
    464.0,
    -465.0,
];

/// `n` draws for `(seed, stream)`: uniform in ±`scale` times a power of two in
/// `2^-12..2^4`, so magnitudes span sixteen binades rather than one, with [`EDGES`] first.
/// `edges` is false where a kernel's contract does not admit them (a norm over a row of
/// ±100s is a different test from a norm, not a harder one).
pub fn draw(n: usize, (seed, stream): (u64, u64), scale: f32, edges: bool) -> Vec<f32> {
    let mut r = Lcg(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ stream);
    let mut v: Vec<f32> = (0..n)
        .map(|_| {
            let binade = ((r.f() + 1.0) * 8.0).floor() as i32 - 12;
            r.f() * scale * 2f32.powi(binade)
        })
        .collect();
    if edges {
        for (slot, e) in v.iter_mut().zip(EDGES) {
            *slot = e;
        }
    }
    v
}

/// `n` uniform draws in `range` for `(seed, stream)`, for operands that are a positive config value or a
/// probability-like weight rather than an activation.
pub fn uniform(n: usize, (seed, stream): (u64, u64), range: std::ops::Range<f32>) -> Vec<f32> {
    let (lo, hi) = (range.start, range.end);
    let mut r = Lcg(seed.wrapping_mul(0xd1b5_4a32_d192_ed03) ^ stream);
    (0..n)
        .map(|_| lo + (r.f() + 1.0) * 0.5 * (hi - lo))
        .collect()
}

/// A rope cell's `(pos, theta)` for `seed`: positions from 0 (identity) to past 128K, three
/// bases. Shared by the rope rows and the CPU plant model, so the model's rate is measured on
/// the operands the device arms draw.
pub fn rope_at(seed: u64) -> (usize, f64) {
    let pos = [0, 1, 7, 4095, 65_535, 131_071][(seed % 6) as usize] + seed as usize;
    (pos, [1e4, 1e6, 5e7][(seed % 3) as usize])
}

/// Integer operands ride the gate's f32 cells as raw words: `Cell` uploads bytes, so a `u32` (or a
/// pair of bf16 halves, [`bf16_words`]) is carried bit for bit.
#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
pub fn words32(v: &[u32]) -> Vec<f32> {
    v.iter().map(|&u| f32::from_bits(u)).collect()
}

/// A bf16 operand by truncation of a [`draw`] — so the edges (±0, subnormals, the ±100s) reach a
/// bf16 cache as themselves — packed two halves per word (an odd count padded with a zero half).
#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
pub fn bf16_words(v: &[f32]) -> Vec<f32> {
    let half = |x: Option<&f32>| x.map_or(0, |x| x.to_bits() >> 16);
    v.chunks(2)
        .map(|p| f32::from_bits(half(p.first()) | half(p.get(1)) << 16))
        .collect()
}

/// `(nt, k)`. `k` near 0.6·nt puts the selection boundary inside the `±0` run of the tie classes
/// (above it the classes hold 1/2 and 1/4 of the entries); `k > nt` is the launcher's clamp.
pub const TOPK_SHAPES: [(usize, usize); 7] = [
    (4096, 2048),
    (4096, 2500),
    (300, 180),
    (65_536, 2048),
    (7, 512),
    (1, 1),
    (1000, 1000),
];

/// A score array for the top-k rows, TIE-HEAVY by construction: random scores almost never tie,
/// so a row over them passes whatever the tie rule. Four classes by `seed % 4`:
///
/// - 0: [`draw`]s with the edges (one `+0`/`-0` pair, never tied at the boundary by luck);
/// - 1: six values only — `±0`, `0.5`, `1`, `2`, `-1` — so every boundary is a tie, and `±0`
///   (which the kernel canonicalises and a raw bit-pattern key would order) dominate;
/// - 2: ReLU-shaped — 3/4 of entries `+0` or `-0` (the reason `index_score`'s output ties at all),
///   the rest a dozen distinct positives;
/// - 3: class 1 with `±inf` and both NaN signs planted — out of the host comparator's contract
///   but fully defined on the key bits, so both twins must still agree to the byte.
pub fn topk_scores(n: usize, seed: u64) -> Vec<f32> {
    let pick = |r: &mut Lcg, set: &[f32]| {
        set[((r.f() + 1.0) * 0.5 * set.len() as f32) as usize % set.len()]
    };
    let mut r = Lcg(seed.wrapping_mul(0xa076_1d64_78bd_642f) ^ 7);
    let ties = [0.0, -0.0, 0.5, 1.0, 2.0, -1.0];
    let relu: Vec<f32> = [
        0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
    ]
    .into_iter()
    .chain((1..=4).map(|k| k as f32 * 0.25))
    .collect();
    let mut v: Vec<f32> = match seed % 4 {
        0 => return draw(n, (seed, 9), 1.0, true),
        2 => (0..n).map(|_| pick(&mut r, &relu)).collect(),
        _ => (0..n).map(|_| pick(&mut r, &ties)).collect(),
    };
    if seed % 4 == 3 {
        let odd = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -f32::NAN];
        v.iter_mut()
            .step_by(5)
            .zip(odd.iter().cycle())
            .for_each(|(s, &o)| *s = o);
    }
    v
}

/// The block scorers' `(s, n_comp, heads, hd)`: V4's decode (one query against 512 blocks, 64
/// heads of 128) first — the hot shape — then a prefill tile at Qwen's anchor width 24, a single
/// block, and a 64-row prefill.
pub const SCORE_BLOCKS_DIMS: [[usize; 4]; 4] = [
    [1, 512, 64, 128],
    [7, 37, 4, 24],
    [1, 1, 64, 128],
    [64, 300, 16, 128],
];

/// One block-scorer cell's `[q, kv, w]`: the edges in `q` and `kv` (so dots cross zero, which is
/// where the ReLU and the bf16 roundings disagree), small positive per-head weights.
pub fn score_blocks_ins([s, n_comp, heads, hd]: [usize; 4], seed: u64) -> Vec<Vec<f32>> {
    vec![
        draw(s * heads * hd, (seed, 1), 1.0, true),
        draw(n_comp * hd, (seed, 2), 1.0, true),
        uniform(s * heads, (seed, 3), 0.0..0.2),
    ]
}
