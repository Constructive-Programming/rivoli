//! The **fp8-e4m3 block-scaled** format the attention and dense projections are stored in:
//! its tile grid, the dequant every converter reads a checkpoint through, the quantizer
//! rivoli applies when a publisher shipped none, and the GEMV oracle.
//!
//! **Split out of `quant.rs` on 2026-08-15, by FORMAT** — see [`super::vq`] for the measured
//! reason. Bodies and comments travelled verbatim, including [`quantize_fp8_block`]'s
//! argument for the amax scale and the three failure modes its test asserts; those are the
//! record of the first quantizer this engine applies to a checkpoint rather than reads out
//! of one.
//!
//! The [`super::Fp8W`] view itself stays in the parent beside its two siblings, because the
//! argument for having three per-format view structs rather than one enum is made once, for
//! all three. Every public name is re-exported by `quant.rs`.

use super::{Fp8W, matvec_bytes};
use rivoli_core::num::e4m3_to_f32;

/// The `weight_scale_inv` grid extent covering `dim` weights: `ceil` in BOTH axes, so one
/// rule states the tile grid's rows and its columns. Sibling of [`vq_groups`]/[`i4_groups`]
/// /[`f4_groups`] — every format states its scale-grid arithmetic exactly once, because the
/// three readers of an fp8 grid had spelled this `div_ceil` separately and a disagreement
/// between them moves every scale lookup without changing a single length.
#[inline]
fn fp8_blocks(dim: usize, block: usize) -> usize {
    dim.div_ceil(block)
}

/// GEMV against an **fp8-e4m3** block-scaled matrix `W[o_dim, i_dim]` — the CPU
/// oracle the HIP `gemv_fp8` kernel (attention/dense projections) is validated
/// against. `scale` is the F32 `weight_scale_inv`, one value per `block × block`
/// tile: `w[o,i] = e4m3(packed[o·i_dim+i]) · scale[(o/block)·sc_cols + i/block]`.
pub fn matvec_fp8(y: &mut [f32], x: &[f32], w: Fp8W<'_>, i_dim: usize) {
    debug_assert_eq!(x.len(), i_dim);
    let sc_cols = fp8_blocks(i_dim, w.block);
    matvec_bytes(y, w.packed, i_dim, |o, row| {
        let mut acc = 0.0f32;
        for (i, (&b, &xi)) in row.iter().zip(x).enumerate() {
            acc += e4m3_to_f32(b) * w.tile_scale(sc_cols, o, i) * xi;
        }
        acc
    });
}

/// Dequantize an fp8 (e4m3) block-scaled weight matrix `W[o_dim, i_dim]` to f32
/// (row-major). `scale` is the F32 `weight_scale_inv` tensor, one value per
/// `block × block` tile: `w[o,i] = e4m3(packed[o,i]) · scale[(o/block)·sc_cols +
/// i/block]`. The DeepSeek/GLM fp8 convention (mirrors colibri's converter); the
/// `_inv` name is historical — it is the dequant multiplier. Offline (converter)
/// use only: this materializes `o_dim·i_dim` f32.
pub fn dequant_fp8_block(w: Fp8W<'_>, shape: [usize; 2]) -> Vec<f32> {
    let [o_dim, i_dim] = shape;
    debug_assert_eq!(w.packed.len(), o_dim * i_dim);
    let sc_cols = fp8_blocks(i_dim, w.block);
    debug_assert_eq!(w.scale.len(), fp8_blocks(o_dim, w.block) * sc_cols);
    let mut out = vec![0.0f32; o_dim * i_dim];
    for o in 0..o_dim {
        for i in 0..i_dim {
            out[o * i_dim + i] = e4m3_to_f32(w.packed[o * i_dim + i]) * w.tile_scale(sc_cols, o, i);
        }
    }
    out
}

/// Quantize an f32 weight matrix `W[o_dim, i_dim]` to **fp8-e4m3 with one f32 scale per
/// `block × block` tile** — the exact inverse of [`dequant_fp8_block`], and the format the
/// resident fp8 GEMV already reads.
///
/// **This is the first quantizer rivoli applies to a checkpoint rather than reads out of
/// one.** GLM ships fp8, DeepSeek-V4 ships fp8 + e8m0, Kimi-K3 ships mxfp4 — every existing
/// converter *copies* a decision the publisher made. Muse Glimmer ships BF16 and nothing
/// else, so the choice of scale per tile is **ours**, and it is a quality decision with no
/// upstream to defer to. `docs/investigations/glimmer-port.md` S5 carries the dNLL gate that
/// obligation creates; do not treat a round-trip test as evidence about model quality.
///
/// The scale is `amax / E4M3_MAX`, i.e. the tile's largest magnitude is mapped exactly onto
/// e4m3's largest finite value. That choice is what makes saturation unreachable rather than
/// merely unlikely: `f32_to_e4m3` clamps at ±448, and clamping is silent, so a scale derived
/// from anything but the amax would round the tile's extremes to a value the dequant cannot
/// distinguish from a genuine 448.
///
/// An all-zero tile takes scale `1.0`, not `0.0`. Both dequantize to zeros, but a zero scale
/// makes the encode `0/0` — and `f32_to_e4m3(NaN)` is `0x7f`, so the tile would come back as
/// 128×128 NaNs that no length or shape check can see.
///
/// Refuses a non-finite input rather than encoding it. `f32_to_e4m3` maps NaN to `0x7f` and
/// saturates ±inf to ±448, so both would survive into the artifact looking like ordinary
/// weights.
pub fn quantize_fp8_block(
    w: &[f32],
    shape: [usize; 2],
    block: usize,
) -> anyhow::Result<(Vec<u8>, Vec<f32>)> {
    use anyhow::ensure;
    // `shape` as an array rather than two `usize` parameters: it is what `Safetensors` hands
    // a caller, so it passes straight through instead of being unpacked and possibly
    // transposed at the call site — and a bare `(w, o_dim, i_dim)` list is a jscpd clone of
    // `convert.rs`'s quantizers, which would have cost an exemption for a coincidence rather
    // than for a copy that is the point.
    let [o_dim, i_dim] = shape;
    ensure!(block > 0, "fp8 block size is 0");
    ensure!(
        w.len() == o_dim * i_dim,
        "fp8 quantize: {} values for a {shape:?} matrix",
        w.len()
    );
    let sc_cols = fp8_blocks(i_dim, block);
    let mut packed = vec![0u8; o_dim * i_dim];
    let mut scale = vec![0f32; fp8_blocks(o_dim, block) * sc_cols];
    // Row-major over the scale grid, so the tile a scale belongs to is `n` itself: the
    // grid IS the loop, rather than a pair of counters the body has to re-multiply.
    for (n, s) in scale.iter_mut().enumerate() {
        *s = quantize_fp8_tile(
            w,
            &mut packed,
            i_dim,
            &Tile::new(n / sc_cols, n % sc_cols, block, shape),
        )?;
    }
    Ok((packed, scale))
}

/// One `block × block` tile of a row-major matrix, as the two index ranges that name it —
/// one value because the amax scan and the encode must cover exactly the same elements, and
/// two loose ranges are two chances to clip one of them differently at a ragged edge.
struct Tile {
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
}

impl Tile {
    /// Tile `(ob, ib)` of a `shape` matrix, clipped to the matrix at the ragged edges.
    fn new(ob: usize, ib: usize, block: usize, shape: [usize; 2]) -> Self {
        let [o_dim, i_dim] = shape;
        Tile {
            rows: ob * block..((ob + 1) * block).min(o_dim),
            cols: ib * block..((ib + 1) * block).min(i_dim),
        }
    }

    /// The tile's elements as flat row-major indices, so each pass over a tile is one loop
    /// instead of a nested pair.
    fn flat(&self, i_dim: usize) -> impl Iterator<Item = usize> + '_ {
        self.rows
            .clone()
            .flat_map(move |o| self.cols.clone().map(move |i| o * i_dim + i))
    }
}

/// Encode one tile at its own amax scale, returning that scale for the grid.
fn quantize_fp8_tile(w: &[f32], packed: &mut [u8], i_dim: usize, t: &Tile) -> anyhow::Result<f32> {
    let s = block_amax_scale(w, i_dim, t)?;
    for n in t.flat(i_dim) {
        packed[n] = rivoli_core::num::f32_to_e4m3(w[n] / s);
    }
    Ok(s)
}

/// One block's e4m3 scale: amax over the block divided by the format max, with the
/// finiteness refusal (NaN would encode as a plausible 0x7f, ±inf saturates to ±448 —
/// silent both ways, so the refusal happens here, before any byte is written).
fn block_amax_scale(w: &[f32], i_dim: usize, t: &Tile) -> anyhow::Result<f32> {
    use anyhow::ensure;
    let mut amax = 0f32;
    for n in t.flat(i_dim) {
        let v = w[n];
        ensure!(
            v.is_finite(),
            "weight [{}, {}] is {v}, which e4m3 would encode as a plausible \
             finite byte (NaN -> 0x7f, +-inf -> saturated +-448)",
            n / i_dim,
            n % i_dim
        );
        amax = amax.max(v.abs());
    }
    Ok(if amax == 0.0 {
        1.0
    } else {
        amax / rivoli_core::num::E4M3_MAX
    })
}

/// fp8 `weight_scale_inv` tile size: 128×128 for both the GLM-5.2 checkpoint and
/// DeepSeek-V4-Flash's `quantization_config.weight_block_size`. Distinct from
/// [`super::f4::F4_GROUP`] (32, along the input dim only), which is the FP4 experts' scheme.
pub const FP8_BLOCK: usize = 128;

#[cfg(test)]
mod tests {
    // `quantize_fp8_block` returns `Result` and three of its four failure modes are the
    // subject here, so these unwrap deliberately — the Ok path is what is being asserted.
    // Crate-wide `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn fp8_block_dequant_applies_tile_scale() {
        // 2×2 fp8 matrix, block=1 (per-element scale for the test) — check the tile
        // indexing and that e4m3 decode is wired. 0x38=1.0, 0x40=2.0, 0xB8=-1.0.
        let packed = [0x38u8, 0x40, 0xB8, 0x00]; // [[1, 2],[-1, 0]]
        let scale = [10.0f32, 100.0, 1000.0, 1.0]; // per element (block=1)
        let out = dequant_fp8_block(Fp8W::new(&packed, &scale, 1), [2, 2]);
        assert_eq!(out, [10.0, 200.0, -1000.0, 0.0]);
    }

    /// `quantize_fp8_block` is the inverse `dequant_fp8_block` reads, and the error it leaves
    /// is bounded by e4m3's own half-ULP — not merely "small".
    ///
    /// The bound is the point. e4m3 carries 3 mantissa bits, so a correctly scaled value
    /// round-trips to within `2^-4` **relative to itself**; asserting instead against the
    /// tile's amax (the loose form) would pass on a quantizer that collapsed every small
    /// value in the tile to zero, which is exactly the defect a per-tile scale can hide.
    #[test]
    fn fp8_block_quantize_round_trips_within_half_ulp() {
        // Deterministic pseudo-random weights spanning three orders of magnitude, so the
        // tiles differ in amax and a scale taken from the wrong tile shows up.
        let (o_dim, i_dim, block) = (5, 7, 2);
        let w: Vec<f32> = (0..o_dim * i_dim)
            .map(|k| {
                let x = ((k * 2654435761usize) % 1013) as f32 / 1013.0 - 0.5;
                x * 10f32.powi((k % 3) as i32 - 1)
            })
            .collect();
        let (packed, scale) = quantize_fp8_block(&w, [o_dim, i_dim], block).unwrap();
        assert_eq!(packed.len(), o_dim * i_dim);
        assert_eq!(scale.len(), o_dim.div_ceil(block) * i_dim.div_ceil(block));
        let back = dequant_fp8_block(Fp8W::new(&packed, &scale, block), [o_dim, i_dim]);
        for (k, (&want, &got)) in w.iter().zip(back.iter()).enumerate() {
            // Relative to the VALUE, with an absolute floor for the ones that land in e4m3's
            // subnormal range after scaling.
            let tol = want.abs() / 16.0 + f32::EPSILON;
            assert!(
                (want - got).abs() <= tol,
                "element {k}: {want} round-tripped to {got}, off by more than a half-ULP"
            );
        }
    }

    /// The three failure modes `quantize_fp8_block`'s doc claims it prevents, asserted — each
    /// produces bytes that pass every length and shape check downstream.
    #[test]
    fn fp8_block_quantize_refuses_what_would_look_like_weights() {
        // An all-zero tile must take scale 1.0, not 0.0. At 0.0 the encode is 0/0 = NaN and
        // `f32_to_e4m3(NaN)` is 0x7f, so the tile would come back as NaNs.
        let (packed, scale) = quantize_fp8_block(&[0.0f32; 4], [2, 2], 2).unwrap();
        assert_eq!(scale, [1.0], "an all-zero tile must not take a zero scale");
        assert!(packed.iter().all(|&b| b == 0));
        assert!(
            dequant_fp8_block(Fp8W::new(&packed, &scale, 2), [2, 2])
                .iter()
                .all(|v| *v == 0.0)
        );

        // Non-finite input refuses rather than encoding: NaN -> 0x7f and +-inf -> +-448 both
        // survive as ordinary-looking bytes.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                quantize_fp8_block(&[1.0, bad, 1.0, 1.0], [2, 2], 2).is_err(),
                "{bad} must refuse"
            );
        }

        // The tile's extreme maps ONTO e4m3's largest finite value, so saturation is
        // unreachable rather than merely unlikely. 0x7e is +448; anything larger would clamp
        // silently.
        let (packed, scale) = quantize_fp8_block(&[1.0, -3.0, 0.5, 2.0], [2, 2], 2).unwrap();
        assert_eq!(scale, [3.0 / rivoli_core::num::E4M3_MAX]);
        assert_eq!(packed[1], 0xfe, "the tile amax must encode as exactly -448");
        assert!(packed.iter().all(|&b| b & 0x7f != 0x7f), "no NaN bytes");

        quantize_fp8_block(&[1.0, 2.0], [2, 2], 2).unwrap_err(); // wrong element count
        quantize_fp8_block(&[1.0; 4], [2, 2], 0).unwrap_err(); // zero block
    }

    #[test]
    fn matvec_fp8_matches_dequant_dot() {
        // Same 2×2 fp8 as above; GEMV must equal the dequant-then-dot reference.
        let packed = [0x38u8, 0x40, 0xB8, 0x00]; // [[1,2],[-1,0]] × per-elem scale
        let scale = [10.0f32, 100.0, 1000.0, 1.0];
        let x = [3.0f32, 5.0];
        let mut y = [0.0f32; 2];
        matvec_fp8(&mut y, &x, Fp8W::new(&packed, &scale, 1), 2);
        // row0: 10·3 + 200·5 = 1030 ; row1: -1000·3 + 0·5 = -3000
        assert_eq!(y, [1030.0, -3000.0]);
    }
}
