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
// The deviceless build compiles only the CPU plant model, which draws activations only.
#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
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

/// `n` random bytes for `(seed, stream)` — every bit pattern a packed weight code can take.
pub fn bytes(n: usize, (seed, stream): (u64, u64)) -> Vec<u8> {
    let mut r = Lcg(seed.wrapping_mul(0xa076_1d64_78bd_642f) ^ stream);
    (0..n)
        .map(|_| {
            r.f();
            (r.0 >> 56) as u8
        })
        .collect()
}

/// Bytes as the f32 WORDS a cell uploads, bit patterns untouched (NaN payloads included —
/// nothing here does arithmetic on them), zero-padded to a whole word.
// Only the device rows build cells; the deviceless build compiles the plant models alone.
#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
pub fn words(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            f32::from_bits(u32::from_le_bytes(w))
        })
        .collect()
}

/// Halfwords (bf16 or fp16 bits) as cell words.
#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
pub fn u16_words(v: impl IntoIterator<Item = u16>) -> Vec<f32> {
    words(&v.into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>())
}
