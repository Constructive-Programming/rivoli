//! Inkling 2's expert dequant and layout ops — the FIRST-PARTY oracle transcribed from
//! `core_model_loading.py` @ `d8f10a82` (`Inkling2LoopDequantizeExperts` :772–816, `Interleave`
//! :232–251, `Inkling2LoopMergeMessageQueries` :823–849). The converter's repack inverts these
//! exactly; S4's kernels are scored against the same arithmetic.
//!
//! **Arch-scoped port boundary** (the naming exception `inkling_config.rs` argues): this module
//! names the Inkling 2 port's dequant oracle, not a reusable engine ABI. NVFP4 is a NEW scale ABI
//! (e2m1 codes + e4m3 block-16 scale + per-expert amax/denominator), NOT rivoli's `.f4` MX-FP4
//! (e2m1 + e8m0 block-32, `quant/f4.rs`); MXFP8 here is row-wise `[1,32]` ue8m0, not the
//! `[128,128]` block of `quant/fp8.rs`. Reusing either existing path would be trap T3/T1.
//!
//! Every function returns `Result` and asserts its input lengths: a wrong-but-plausible dequant
//! (a scale grid read at the wrong stride) is far worse than a hard failure, the same argument
//! `format::tensors::dequant_fp8` makes.
//!
//! **One walk, two outputs.** [`Nvfp4`] and [`Mxfp8`] each expose `dequant` (f32, the oracle the
//! transformers golden is bit-compared against) and `dequant_to_bf16_le` (the artifact's stored
//! bytes, with the `w13` row de-interleave fused in). Both outputs run the SAME per-block decode
//! through the private `Plane` walk + `decode_row`, so the association
//! `(code · block_scale) · expert_factor` exists in one place and the fused emitter cannot drift
//! from the oracle.

use anyhow::{Result, ensure};
use rivoli_core::num::{e4m3_to_f32, f32_to_bf16};

/// The E2M1 codebook (LOAD:782). Index is the 4-bit code; even element is the LOW nibble.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// One `[E, rows, cols]` expert plane's row walk: the source weight/scale row strides and the
/// optional `w13` de-interleave permutation. Both schemes and both output dtypes walk this
/// identically — only the per-block arithmetic differs — so the geometry is written once and the
/// four emitters cannot disagree about a stride or a permutation.
struct Plane {
    e: usize,
    rows: usize,
    cols: usize,
    /// Source weight row stride: `cols_packed` for NVFP4, `cols` for MXFP8.
    w_stride: usize,
    /// Source scale row stride: `cols / 16` for NVFP4, `cols / 32` for MXFP8.
    scale_cols: usize,
    /// `Some(rows / 2)` when the output rows are the de-interleaved `[gate_all; up_all]` order.
    half: Option<usize>,
}

impl Plane {
    /// `geom` is `(experts, rows, cols)` and `strides` is `(weight row stride, scale row stride)`;
    /// `scheme` names the format in the refusal. The de-interleave precondition is checked HERE,
    /// so every caller of the permuted walk is held to an even row count.
    fn new(
        geom: (usize, usize, usize),
        strides: (usize, usize),
        deinterleave: bool,
        scheme: &str,
    ) -> Result<Self> {
        let (e, rows, cols) = geom;
        ensure!(
            !deinterleave || rows.is_multiple_of(2),
            "{scheme}: de-interleave needs an even row count, got {rows}"
        );
        Ok(Self {
            e,
            rows,
            cols,
            w_stride: strides.0,
            scale_cols: strides.1,
            half: deinterleave.then_some(rows / 2),
        })
    }

    /// Every `(expert, source weight row base, source scale row base, destination row index)`.
    /// The destination index already carries the de-interleave permutation, so an emitter writes
    /// `orow * cols` (f32) or `orow * cols * 2` (bf16 bytes) and nothing else.
    fn rows(&self) -> impl Iterator<Item = (usize, usize, usize, usize)> + '_ {
        (0..self.e).flat_map(move |ei| {
            (0..self.rows).map(move |r| {
                let base = ei * self.rows + r;
                (
                    ei,
                    base * self.w_stride,
                    base * self.scale_cols,
                    ei * self.rows + dst_row(r, self.half),
                )
            })
        })
    }

    /// Run `decode` over every row into an owned f32 plane.
    fn emit_f32(
        &self,
        mut decode: impl FnMut(usize, usize, usize, &mut dyn FnMut(usize, f32)),
    ) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.e * self.rows * self.cols];
        for (ei, wrow, srow, orow) in self.rows() {
            let base = orow * self.cols;
            decode(ei, wrow, srow, &mut |i, v| out[base + i] = v);
        }
        Ok(out)
    }

    /// Run `decode` over every row into an owned little-endian bf16 byte plane.
    fn emit_bf16_le(
        &self,
        mut decode: impl FnMut(usize, usize, usize, &mut dyn FnMut(usize, f32)),
    ) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.e * self.rows * self.cols * 2];
        for (ei, wrow, srow, orow) in self.rows() {
            let base = orow * self.cols * 2;
            decode(ei, wrow, srow, &mut |i, v| {
                put_bf16(&mut out, base + i * 2, v)
            });
        }
        Ok(out)
    }
}

/// The destination row for source row `r`: identity, or the `w13` de-interleave permutation
/// (`[gate_0, up_0, gate_1, up_1, …]` → `[gate_all; up_all]`, `half = rows / 2`) —
/// `deinterleave_rows`'s permutation, stated once for the fused emitter.
fn dst_row(r: usize, half: Option<usize>) -> usize {
    match half {
        None => r,
        Some(_) if r.is_multiple_of(2) => r / 2,
        Some(h) => h + r / 2,
    }
}

/// Write one f32 as little-endian bf16 at byte offset `at`.
fn put_bf16(out: &mut [u8], at: usize, v: f32) {
    out[at..at + 2].copy_from_slice(&f32_to_bf16(v).to_le_bytes());
}

/// `2^(s − 127)` for a ue8m0 scale byte, **bit-identical to `2f32.powi(i32::from(s) - 127)` for
/// every `s` in `0..=255`** — including the subnormal `s = 0` (2⁻¹²⁷) and the `+inf` `s = 255`
/// (2¹²⁸), the two ends where `from_bits` spells the answer and `powi` reproduces it exactly.
/// Gated by `exp2_u8_is_bit_identical_to_powi` below; hoisting the scale decode out of the inner
/// loop is only sound because this is exact.
#[inline]
fn exp2_u8(s: u8) -> f32 {
    if s == 0 {
        f32::from_bits(0x0040_0000)
    } else {
        f32::from_bits(u32::from(s) << 23)
    }
}

/// NVFP4 expert weights (Mini layers 0–19): two E2M1 codes per byte, one e4m3 scale per 16 values
/// along the INPUT (last) dim, and a per-expert factor `amax[e] / denominator` (LOAD:800–814).
pub struct Nvfp4<'a> {
    /// U8 `[E, rows, cols_packed]`; the logical width is `cols_packed * 2`.
    pub weight: &'a [u8],
    /// F8_E4M3 `[E, rows, cols_packed / 8]` (= logical cols / 16).
    pub scale: &'a [u8],
    /// F32 `[E]`, the per-expert amax.
    pub amax: &'a [f32],
    /// F32 scalar (`outer_scale_denominator`, shape `[1]`).
    pub denom: f32,
    pub rows: usize,
    pub cols_packed: usize,
}

impl Nvfp4<'_> {
    /// The scale row stride: one e4m3 scale per 16 logical values = 8 packed bytes.
    fn scale_cols(&self) -> usize {
        self.cols_packed / 8
    }

    /// Geometry, lengths and the expert denominator, shared by both outputs so the f32 oracle and
    /// the fused emitter cannot disagree about what a valid input is. `denom` is the per-expert
    /// divisor `amax[e] / denom` (LOAD:804): zero, negative or non-finite makes every expert's
    /// factor wrong in a way no length check would catch.
    fn validate(&self) -> Result<(usize, usize, usize)> {
        let e = self.amax.len();
        let cols = self.cols_packed * 2;
        ensure!(
            e > 0 && self.rows > 0 && self.cols_packed > 0,
            "NVFP4: empty geometry"
        );
        ensure!(
            self.denom > 0.0 && self.denom.is_finite(),
            "NVFP4: denom {} must be positive and finite",
            self.denom
        );
        ensure!(
            cols.is_multiple_of(16),
            "NVFP4: logical width {cols} not a multiple of 16"
        );
        ensure!(
            self.weight.len() == e * self.rows * self.cols_packed,
            "NVFP4 weight: {} bytes for [{e}, {}, {}] packed",
            self.weight.len(),
            self.rows,
            self.cols_packed
        );
        ensure!(
            self.scale.len() == e * self.rows * self.scale_cols(),
            "NVFP4 scale: {} bytes for [{e}, {}, {}] (block 16)",
            self.scale.len(),
            self.rows,
            self.scale_cols()
        );
        Ok((e, self.rows, cols))
    }

    /// The shared row walk, at NVFP4's strides.
    fn plane(&self, deinterleave: bool) -> Result<Plane> {
        let (e, rows, cols) = self.validate()?;
        Plane::new(
            (e, rows, cols),
            (self.cols_packed, self.scale_cols()),
            deinterleave,
            "NVFP4",
        )
    }

    /// Decode one row, block by block, handing each value to `emit(index, value)`. One body for
    /// both outputs: the association matches LOAD:813–814 exactly — `(code_value · block_scale)
    /// · expert_factor`, not `code_value · (block_scale · factor)`, because f32 multiply is not
    /// associative and the bf16 golden is bit-compared.
    fn decode_row(
        &self,
        (wrow, srow): (usize, usize),
        factor: f32,
        mut emit: impl FnMut(usize, f32),
    ) {
        for sb in 0..self.scale_cols() {
            let bs = e4m3_to_f32(self.scale[srow + sb]);
            for k in 0..8 {
                let byte = self.weight[wrow + sb * 8 + k];
                emit(
                    (sb * 8 + k) * 2,
                    (E2M1[(byte & 0xF) as usize] * bs) * factor,
                );
                emit(
                    (sb * 8 + k) * 2 + 1,
                    (E2M1[(byte >> 4) as usize] * bs) * factor,
                );
            }
        }
    }

    /// Dequantize to row-major f32 `[E, rows, cols_packed * 2]`, BEFORE the bf16 narrowing the
    /// artifact stores. `value[e,r,c] = E2M1[code] · e4m3(scale[e,r,c/16]) · (amax[e]/denom)`,
    /// the code being the low nibble of `weight[e,r,c/2]` for even `c`, the high for odd.
    pub fn dequant(&self) -> Result<Vec<f32>> {
        let plane = self.plane(false)?;
        plane.emit_f32(|ei, wrow, srow, emit| self.decode_with_factor(ei, (wrow, srow), emit))
    }

    /// The artifact's stored bytes: dequant, the optional `w13` row de-interleave and the bf16
    /// narrowing in ONE pass, so the converter never materialises the 2.7 GB f32 plane. Bitwise
    /// identical to `dequant` → `deinterleave_rows` → `f32_to_bf16`, which the deviceless gate
    /// asserts both ways.
    pub fn dequant_to_bf16_le(&self, deinterleave: bool) -> Result<Vec<u8>> {
        self.plane(deinterleave)?
            .emit_bf16_le(|ei, wrow, srow, emit| self.decode_with_factor(ei, (wrow, srow), emit))
    }

    /// One row's block walk WITH this projection's per-expert factor: `(code · block_scale) ·
    /// (amax[e] / denom)`, the association LOAD:813–814 uses. ONE body for the f32 oracle, the
    /// bf16 plane and [`Self::decode_expert_row`], because a second spelling could re-associate
    /// it into `code · (block_scale · factor)` — a different f32 with no error anywhere, which
    /// the bf16 golden would then be compared against.
    fn decode_with_factor(
        &self,
        ei: usize,
        (wrow, srow): (usize, usize),
        emit: &mut dyn FnMut(usize, f32),
    ) {
        self.decode_row((wrow, srow), self.amax[ei] / self.denom, emit);
    }

    /// Decode ONE row of ONE expert into `out`, which must be exactly the logical row width
    /// (`cols_packed * 2`).
    ///
    /// For a caller that re-encodes a row at a time and must not allocate the whole
    /// `[E, rows, cols]` f32 plane: `convert_inkling --mxfp8` walks Mini's 20 NVFP4 layers this
    /// way, one 2304-value row per call, straight into its MXFP8 quantizer. The values are
    /// `decode_row`'s, at the same `(code · block_scale) · factor` association, so a row read
    /// here is the f32 oracle's row — not a second decode that could drift from it.
    ///
    /// Geometry is re-validated per call — a handful of comparisons against the 32-value block
    /// walk this hands back — and that is deliberately the cheap side of the trade: it keeps a
    /// plane built for one projection from being decoded at another's width, which is otherwise
    /// a wrong-by-a-stride read with no length check in its way.
    pub fn decode_expert_row(&self, expert: usize, row: usize, out: &mut [f32]) -> Result<()> {
        let (e, rows, cols) = self.validate()?;
        ensure!(
            expert < e && row < rows,
            "NVFP4: expert {expert} row {row} outside [{e}, {rows})"
        );
        ensure!(
            out.len() == cols,
            "NVFP4: {}-value buffer for a {cols}-wide row",
            out.len()
        );
        let base = expert * rows + row;
        let spans = (base * self.cols_packed, base * self.scale_cols());
        self.decode_with_factor(expert, spans, &mut |i, v| out[i] = v);
        Ok(())
    }
}

/// MXFP8 expert weights (Mini layers 20–23; all Small layers): one e4m3 value per byte, one
/// power-of-two scale `2^(scale−127)` per 32 values along the INPUT dim — ROW-WISE `[1,32]`
/// (LOAD:808–810). The scale is a raw U8 exponent, NOT e4m3 (trap T1).
pub struct Mxfp8<'a> {
    /// F8_E4M3 `[E, rows, cols]`.
    pub weight: &'a [u8],
    /// U8 `[E, rows, cols / 32]`.
    pub scale: &'a [u8],
    pub rows: usize,
    pub cols: usize,
}

impl Mxfp8<'_> {
    /// Geometry and lengths, shared by both outputs (see [`Nvfp4::validate`]).
    fn validate(&self) -> Result<(usize, usize, usize)> {
        ensure!(self.rows > 0 && self.cols > 0, "MXFP8: empty geometry");
        ensure!(
            self.cols.is_multiple_of(32),
            "MXFP8: width {} not a multiple of 32",
            self.cols
        );
        let plane = self.rows * self.cols;
        ensure!(
            !self.weight.is_empty() && self.weight.len().is_multiple_of(plane),
            "MXFP8 weight: {} bytes not a whole number of [{}, {plane}] planes",
            self.weight.len(),
            self.rows
        );
        let e = self.weight.len() / plane;
        ensure!(
            self.scale.len() == e * self.rows * (self.cols / 32),
            "MXFP8 scale: {} bytes for [{e}, {}, {}] (row-wise block 32)",
            self.scale.len(),
            self.rows,
            self.cols / 32
        );
        Ok((e, self.rows, self.cols))
    }

    /// The shared row walk, at MXFP8's strides.
    fn plane(&self, deinterleave: bool) -> Result<Plane> {
        let (e, rows, cols) = self.validate()?;
        Plane::new((e, rows, cols), (cols, cols / 32), deinterleave, "MXFP8")
    }

    /// Decode one row, block by block; the block scale is hoisted out of the 32-value inner loop
    /// (exact, `exp2_u8`), so it is decoded once per block rather than once per value.
    fn decode_row(&self, (wrow, srow): (usize, usize), mut emit: impl FnMut(usize, f32)) {
        let scale_cols = self.cols / 32;
        for sb in 0..scale_cols {
            let sc = exp2_u8(self.scale[srow + sb]);
            for k in 0..32 {
                emit(
                    sb * 32 + k,
                    e4m3_to_f32(self.weight[wrow + sb * 32 + k]) * sc,
                );
            }
        }
    }

    /// Dequantize to row-major f32 `[E, rows, cols]`.
    /// `value[e,r,c] = e4m3(weight[e,r,c]) · 2^(scale[e,r,c/32] − 127)`.
    pub fn dequant(&self) -> Result<Vec<f32>> {
        self.plane(false)?.emit_f32(|_, wrow, srow, emit| {
            self.decode_row((wrow, srow), emit);
        })
    }

    /// The fused bf16 emitter — see [`Nvfp4::dequant_to_bf16_le`]; same bytes as `dequant` →
    /// `deinterleave_rows` → `f32_to_bf16` with no f32 plane allocated.
    pub fn dequant_to_bf16_le(&self, deinterleave: bool) -> Result<Vec<u8>> {
        self.plane(deinterleave)?
            .emit_bf16_le(|_, wrow, srow, emit| {
                self.decode_row((wrow, srow), emit);
            })
    }
}

/// De-interleave `w13` rows (LOAD `Interleave(dim=1)`, :243–250): the checkpoint stores
/// `[gate_0, up_0, gate_1, up_1, …]` along dim 1; the artifact wants `[gate_all; up_all]`.
/// `rows` is `2·I`; the result has the same shape, rows permuted. Applied to the DEQUANTED f32
/// weight (the row permutation commutes with the per-row dequant, and the scale rows carry the
/// same interleave, CONV:365–370).
pub fn deinterleave_rows(w: &[f32], e: usize, rows: usize, cols: usize) -> Result<Vec<f32>> {
    ensure!(rows.is_multiple_of(2), "deinterleave: rows {rows} is odd");
    ensure!(
        w.len() == e * rows * cols,
        "deinterleave: {} values for [{e},{rows},{cols}]",
        w.len()
    );
    let half = rows / 2;
    let mut out = vec![0f32; w.len()];
    for ei in 0..e {
        let base = ei * rows * cols;
        for i in 0..half {
            let gate_src = base + (2 * i) * cols;
            let up_src = base + (2 * i + 1) * cols;
            let gate_dst = base + i * cols;
            let up_dst = base + (half + i) * cols;
            out[gate_dst..gate_dst + cols].copy_from_slice(&w[gate_src..gate_src + cols]);
            out[up_dst..up_dst + cols].copy_from_slice(&w[up_src..up_src + cols]);
        }
    }
    Ok(out)
}

/// Merge the two message-query banks (LOAD:843–849): `q_mlp` `[2L, H]` holds `[q, mlp]` per loop
/// layer, `kv` `[2L, H]` holds `[k, v]`; the result is `[L, 4, H]` in order **(q, k, v, mlp)**
/// (trap T13). `L` is the number of loop layers (Mini: 12).
pub fn merge_message_queries(
    q_mlp: &[f32],
    kv: &[f32],
    loop_layers: usize,
    h: usize,
) -> Result<Vec<f32>> {
    ensure!(
        q_mlp.len() == 2 * loop_layers * h,
        "merge: q_mlp {} values for [2·{loop_layers},{h}]",
        q_mlp.len()
    );
    ensure!(
        kv.len() == 2 * loop_layers * h,
        "merge: kv {} values for [2·{loop_layers},{h}]",
        kv.len()
    );
    let mut out = vec![0f32; loop_layers * 4 * h];
    for l in 0..loop_layers {
        let q = &q_mlp[2 * l * h..(2 * l + 1) * h];
        let mlp = &q_mlp[(2 * l + 1) * h..(2 * l + 2) * h];
        let k = &kv[2 * l * h..(2 * l + 1) * h];
        let v = &kv[(2 * l + 1) * h..(2 * l + 2) * h];
        let ob = l * 4 * h;
        out[ob..ob + h].copy_from_slice(q);
        out[ob + h..ob + 2 * h].copy_from_slice(k);
        out[ob + 2 * h..ob + 3 * h].copy_from_slice(v);
        out[ob + 3 * h..ob + 4 * h].copy_from_slice(mlp);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    // The subject is arithmetic over in-memory planes, so nothing here touches a file. Crate-wide
    // `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used)]
    use super::{Nvfp4, exp2_u8};

    /// One expert, two 16-value rows of distinct weights, one e4m3 scale each, and a factor that
    /// is not 1 so the per-expert multiply is exercised.
    fn two_row_plane() -> Nvfp4<'static> {
        // Two packed bytes per 16-value block: 8 bytes a row, so `scale_cols` is 1.
        static WEIGHT: [u8; 16] = [
            0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, // row 0
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, // row 1
        ];
        static SCALE: [u8; 2] = [0x3c, 0x40]; // 1.5 and 2.0 in e4m3
        static AMAX: [f32; 1] = [3.5];
        Nvfp4 {
            weight: &WEIGHT,
            scale: &SCALE,
            amax: &AMAX,
            denom: 7.0,
            rows: 2,
            cols_packed: 8,
        }
    }

    /// **The per-row decode is the plane decode's own row**, bit for bit — the claim
    /// `convert_inkling --mxfp8` rests on when it re-encodes Mini's NVFP4 experts one row at a
    /// time instead of materialising `[E, rows, cols]`. A second decode written for the row path
    /// could re-associate `(code · block_scale) · factor` into `code · (block_scale · factor)`
    /// and produce a plausible, different f32; an equality here is what rules that out.
    #[test]
    fn decode_expert_row_matches_the_plane_oracle() {
        let plane = two_row_plane();
        let all = plane.dequant().unwrap();
        assert_eq!(all.len(), 32);
        let mut row = vec![0f32; 16];
        plane.decode_expert_row(0, 1, &mut row).unwrap();
        assert_eq!(row, all[16..32].to_vec());
        // And it is not a constant: row 0 must differ from row 1, both having decoded.
        plane.decode_expert_row(0, 0, &mut row).unwrap();
        assert_ne!(row, all[16..32].to_vec());
        assert_eq!(row, all[..16].to_vec());
        // A buffer of the wrong width, or an expert/row outside the plane, is refused.
        assert!(plane.decode_expert_row(0, 0, &mut [0f32; 8]).is_err());
        assert!(plane.decode_expert_row(0, 2, &mut [0f32; 16]).is_err());
        assert!(plane.decode_expert_row(1, 0, &mut [0f32; 16]).is_err());
    }

    /// The bit-exactness claim `exp2_u8` is written against, over the WHOLE scale-byte domain —
    /// every ue8m0 value a checkpoint can carry, not a sample. Hoisting the block scale out of
    /// the inner loop (and decoding a u8 exponent with `from_bits`) is only sound if this holds
    /// at every byte, including the subnormal 0 and the infinite 255.
    #[test]
    fn exp2_u8_is_bit_identical_to_powi() {
        for s in 0u8..=255 {
            assert_eq!(
                exp2_u8(s).to_bits(),
                2f32.powi(i32::from(s) - 127).to_bits(),
                "scale byte {s}: exp2_u8 and powi disagree"
            );
        }
    }
}
