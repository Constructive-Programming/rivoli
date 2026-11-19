//! The per-family CELL builders — the operands one shape × seed cell of a row draws — kept
//! deviceless beside [`super::inputs`] so a CPU plant model and the device row read the SAME
//! operands (`plant_models.rs`), and apart from it because they are layouts over the draws, not
//! draws: the block scorers, `gemm_bf16`, and the compressor's geometries and prefill.

use super::inputs::{bf16_words, draw, uniform};
use rivoli_backend::abi::CompGeom;

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

/// A self-consistent compressor geometry — `cd` and `ents` derived from `coff` as
/// `v4/geometry.rs::Geom` derives them — with `rd` 64 and V4's `norm_eps`.
pub fn derived(ratio: i32, coff: i32, d: i32) -> CompGeom {
    CompGeom {
        ratio,
        coff,
        d,
        cd: coff * d,
        ents: coff * ratio,
        rd: 64,
        eps: 1e-6,
    }
}

/// A shipped geometry: `coff` from the overlap rule (`ratio == 4`).
pub fn geom(ratio: i32, d: i32) -> CompGeom {
    derived(ratio, 1 + i32::from(ratio == 4), d)
}

/// The four shipped geometries: attention and indexer compressors at both ratios.
pub const GEOMS: [(i32, i32); 4] = [(4, 512), (128, 512), (4, 128), (128, 128)];

/// The finish operands every pooling cell draws: the compressor's own norm weight (positive, as
/// RMSNorm weights are) and a RoPE table of `(cos, sin)` pairs covering `positions` rows.
pub fn finish_ins(p: &CompGeom, positions: usize, seed: u64) -> [Vec<f32>; 2] {
    [
        uniform(p.d as usize, (seed, 6), 0.5..1.5),
        uniform(positions * p.rd as usize, (seed, 7), -1.0..1.0),
    ]
}

/// `kv_compress_prefill`'s `(geometry index into GEOMS, nblk)`: block 0 alone (its masked overlap
/// half), a few, and a 2048-token prefill at ratio 4.
pub const PREFILL_SHAPES: [(usize, usize); 6] = [(0, 1), (0, 3), (1, 2), (2, 5), (3, 1), (0, 512)];

/// One prefill cell's `[kv, score, ape, norm, freqs]`: scores spread wide (the softmax's `exp`
/// edges), the table covering every block's first position.
pub fn prefill_ins(p: &CompGeom, nblk: usize, seed: u64) -> Vec<Vec<f32>> {
    let (ratio, cd) = (p.ratio as usize, p.cd as usize);
    let [norm, freqs] = finish_ins(p, nblk * ratio, seed);
    vec![
        draw(nblk * ratio * cd, (seed, 1), 1.0, true),
        draw(nblk * ratio * cd, (seed, 2), 4.0, true),
        draw(ratio * cd, (seed, 3), 1.0, false),
        norm,
        freqs,
    ]
}

/// `gemm_bf16`'s `[m, n, k]`: V4's compressor projection (decode, then a 13-token prefill), the
/// indexer's `wproj` (64 heads), K3/Glimmer's one-row `bproj` shape, and a ragged tail.
pub const GEMM_SHAPES: [[usize; 3]; 5] = [
    [1, 1024, 4096],
    [13, 1024, 4096],
    [1, 64, 4096],
    [1, 512, 7168],
    [3, 7, 33],
];

/// One `gemm_bf16` cell's `[x, w]`, `w` packed bf16 (two halves per word).
pub fn gemm_ins([m, n, k]: [usize; 3], seed: u64) -> Vec<Vec<f32>> {
    vec![
        draw(m * k, (seed, 1), 1.0, true),
        bf16_words(&draw(n * k, (seed, 2), 0.05, true)),
    ]
}
