//! The range rows' INPUTS, deviceless: the arenas, descriptors, scale codes and routing weights
//! a range cell is built from, and the fp4 formats. Split from `moe.rs`/`moe_f4.rs` so the CPU
//! plant models (`plants.rs`) build the very cells the device rows launch rather than a
//! lookalike (the rule a plant model is held to).
// Device rows use all of it; the deviceless build's plant models only part.
#![cfg_attr(not(feature = "rocm"), allow(dead_code))]

use super::common::f32b;
use super::inputs::{bytes, draw, nan_some, nasty, uniform};

/// `(hidden, inter, e_start, e_count, nrow)` of one range cell.
pub type Range = (usize, usize, usize, usize, usize);

/// Bytes between consecutive experts' weight and scale spans in the arenas.
pub const W_SHIFT: usize = 2;
pub const S_SHIFT: usize = 4;

/// Per-row byte extents of one format: `(weight bytes, scale bytes)` of a row whose reduction
/// dim is the argument.
pub type RowBytes = fn(usize) -> (usize, usize);

/// Byte offsets of `[gate, up, down]` in the weight and scale arenas, and their totals.
pub struct Layout {
    pub w: [usize; 3],
    pub s: [usize; 3],
    pub w_len: usize,
    pub s_len: usize,
}

impl Layout {
    /// Gate and up are `inter` rows over `hidden`, down is `hidden` rows over `inter`, each
    /// projection padded by every expert's shift so no expert's span reaches the next
    /// projection's.
    ///
    /// > **The padding is a fix, 2026-11-19.** Without it, the last up rows of every expert past
    /// > the first read the first bytes of the DOWN scales, where the edge codes are planted
    /// > (NaN, 2^127): one non-finite `h`, so every down dot NaN and every contribution the same
    /// > clamped bound on both sides. Pass 2 compared nothing real at any shape whose live
    /// > experts sat past expert 0, which hid the `moe_f4` plant at K3's production shapes on
    /// > the device (rust-kernels.md §5, Phase 1 L2) and is shown by the CPU model in
    /// > `plants.rs`.
    pub fn new((hidden, inter, e_start, e_count, _): Range, row: RowBytes) -> Self {
        // Rounded to 16 so each projection's base keeps its natural alignment and only the
        // deliberate per-expert shift misaligns.
        let pad = |shift: usize| ((e_start + e_count) * shift).next_multiple_of(16);
        let (gate, down) = (row(hidden), row(inter));
        let wg = inter * gate.0 + pad(W_SHIFT);
        let sg = inter * gate.1 + pad(S_SHIFT);
        Self {
            w: [0, wg, 2 * wg],
            s: [0, sg, 2 * sg],
            w_len: 2 * wg + hidden * down.0,
            s_len: 2 * sg + hidden * down.1,
        }
    }

    /// The scale arena's `[gate/up, down]` split, for the scale fills.
    fn down_scales(&self) -> usize {
        self.s[2]
    }

    /// `n` experts' descriptors as raw bytes: six pointers, `(weights, scales)` per projection —
    /// the layout `ExpertDesc` and `ExpertDescF4` share.
    pub fn descs(&self, (w, s): (usize, usize), n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|k| {
                (0..3).flat_map(move |p| [w + self.w[p] + k * W_SHIFT, s + self.s[p] + k * S_SHIFT])
            })
            .flat_map(|a| (a as u64).to_le_bytes())
            .collect()
    }
}

/// One range cell's inputs before the descriptors: `x`, `wexpert`, both arenas (with room for
/// every expert's shift), then the format's extras.
pub fn range_inputs(
    (hidden, _, e_start, e_count, nrow): Range,
    seed: u64,
    lay: &Layout,
    scales: Vec<u8>,
) -> Vec<Vec<u8>> {
    let n_desc = e_start + e_count;
    let mut w = uniform(n_desc * nrow, (seed, 2), 0.0..1.0);
    // Inside the range, and always leaving one slot a live weight: exact zeros of both signs
    // are skipped rows, and the third is a non-power-of-two weight (so a bf16 rounding before
    // or after it differs) or, on the nasty seeds, a NaN that takes the clamp.
    // The specials' slots ROTATE with the seed, so every expert of the range — the first and
    // the last included — carries a live weight on some seeds: the two zeros take two slots
    // per seed, so an expert is live on (live − 2)/live of seeds (1/2 when live is 2 or 4,
    // 1/3 when 3, 2/3 when 6).
    let specials = [0.0, -0.0, if nasty(seed) { f32::NAN } else { 0.3 }];
    let live = e_count * nrow;
    let range = &mut w[e_start * nrow..];
    for (k, v) in specials.into_iter().enumerate().take(live - 1) {
        range[(seed as usize + k) % live] = v;
    }
    vec![
        f32b(&nan_some(draw(nrow * hidden, (seed, 1), 1.0, true), seed)),
        f32b(&w),
        bytes(lay.w_len + n_desc * W_SHIFT, (seed, 3)),
        scales,
    ]
}

/// A format's scale arena (with room for `n_desc` experts' shifts): its `scale` code of each
/// finite draw (`scale_width` bytes) — at 2^-6, near the real checkpoints' group scales, so a
/// dot over thousands of weights stays in range — with its `edges`, the non-finite/−0/subnormal
/// codes, planted in the DOWN scales every seed and in all of them on the [`edge_seed`]s.
pub fn scale_fill(lay: &Layout, (seed, n_desc): (u64, usize), fmt: &Format) -> Vec<u8> {
    let (width, code, edges) = (fmt.scale_width, fmt.scale, (fmt.edges)());
    let n = (lay.s_len + n_desc * S_SHIFT) / width;
    let vals = uniform(n, (seed, 4), 0.25..2.0);
    let mut s: Vec<u8> = vals.iter().flat_map(|&v| code(v * 2f32.powi(-6))).collect();
    let starts = [lay.down_scales() / width, 0];
    let spots = if edge_seed(seed) {
        &starts[..]
    } else {
        &starts[..1]
    };
    for &at in spots {
        for (k, e) in edges.iter().enumerate() {
            let i = (at + 3 + 7 * k) * width;
            s[i..i + width].copy_from_slice(e);
        }
    }
    s
}

/// The seed class whose gate/up scales and codebooks carry the edge codes (one in eight). NOT
/// the nasty class: those seeds put a NaN in `x`, which would turn every R=1 gate/up dot NaN
/// and hide the edges from the activations.
pub fn edge_seed(seed: u64) -> bool {
    seed % 8 == 3
}

/// `n` fixed-point accumulator words for `(seed, stream)`: draws (|v| ≤ 16) at the kernels'
/// scale `MOE_ACC_SCALE` = 2^44, so up to the 2^48 `moe.hip`'s drain note sizes the f64 convert
/// for.
pub fn fixed_words(n: usize, at: (u64, u64)) -> Vec<u8> {
    let v = draw(n, at, 1.0, false);
    v.iter()
        .flat_map(|x| ((x * 2f32.powi(44)) as i64).to_le_bytes())
        .collect()
}

/// What separates one format's range row from another's: its launcher's name, the arena
/// extents, the scale codes, the extras, and the shapes.
pub struct Format {
    pub name: &'static str,
    pub row: RowBytes,
    pub scale_width: usize,
    pub scale: fn(f32) -> Vec<u8>,
    pub edges: fn() -> Vec<Vec<u8>>,
    pub extras: fn(u64) -> Vec<Vec<u8>>,
    pub shapes: &'static [Range],
    /// How many of `shapes`, from the first, are also timed (the engine's real shapes).
    pub timed: usize,
}

/// Byte codes as the owned rows `scale_fill` plants.
pub fn vecs<const W: usize>(codes: impl IntoIterator<Item = [u8; W]>) -> Vec<Vec<u8>> {
    codes.into_iter().map(|c| c.to_vec()).collect()
}

/// No format extras.
pub fn no_extras(_: u64) -> Vec<Vec<u8>> {
    vec![]
}

/// fp4 rows: `i_dim / 2` packed bytes and `ceil(i_dim / 32)` e8m0 scales.
fn f4_row(i_dim: usize) -> (usize, usize) {
    (i_dim / 2, i_dim.div_ceil(32))
}

/// A positive finite draw as its e8m0 code (the exponent byte: a power-of-two scale).
fn e8m0(v: f32) -> Vec<u8> {
    vec![(v.to_bits() >> 23) as u8]
}

/// e8m0's NaN, its 2^-127 floor, its 2^127 ceiling, and 2^-126.
fn e8m0_edges() -> Vec<Vec<u8>> {
    vecs([0xffu8, 0x00, 0xfe, 0x01].map(|b| [b]))
}

/// Both fp4 formats' arenas; they differ in launcher and shapes only.
const fn f4_format(name: &'static str, shapes: &'static [Range]) -> Format {
    Format {
        name,
        row: f4_row,
        scale_width: 1,
        scale: e8m0,
        edges: e8m0_edges,
        extras: no_extras,
        shapes,
        timed: 2,
    }
}

pub const V4: Format = f4_format(
    "moe_expert_range_f4",
    &[
        (4096, 2048, 2, 1, 1),
        (4096, 2048, 0, 6, 1),
        (256, 128, 0, 3, 1),
        (384, 256, 1, 2, 1),
        (128, 128, 0, 1, 1),
    ],
);

pub const K3: Format = f4_format(
    "moe_expert_range_f4_situ",
    &[
        (3584, 3072, 2, 1, 1),
        (3584, 3072, 0, 4, 1),
        (96, 64, 0, 3, 1),
        (160, 96, 1, 2, 1),
        (32, 32, 0, 1, 1),
    ],
);

/// K3's SiTU betas for a seed: the shipped (4, 25) on odd seeds, seeded positives on even ones
/// so tanh's range moves. One rule for the row and its CPU model.
pub fn k3_betas(seed: u64) -> (f32, f32) {
    match uniform(2, (seed, 7), 0.25..40.0)[..] {
        [b1, b2] if seed.is_multiple_of(2) => (b1, b2),
        _ => (4.0, 25.0),
    }
}

/// One range cell's layout and inputs before the descriptors — what `moe.rs`'s rows launch and
/// the CPU plant models read.
pub fn cell_inputs(fmt: &Format, range: Range, seed: u64) -> (Layout, Vec<Vec<u8>>) {
    let lay = Layout::new(range, fmt.row);
    let scales = scale_fill(&lay, (seed, range.2 + range.3), fmt);
    let ins = range_inputs(range, seed, &lay, scales);
    (lay, ins)
}
