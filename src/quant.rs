//! int4 weight decode, matching colibri's snapshot layout exactly (glm.c
//! `matmul_i4` / `pack_int4`), so numerics are bit-comparable to the C engine.
//!
//! Per-row int4: a weight matrix `W[O, I]` is stored as
//!   - `packed`: `O` rows of `rb = (I+1)/2` bytes; within a byte the LOW nibble
//!     is column `2j`, the HIGH nibble column `2j+1`; each nibble is a value in
//!     `0..=15` encoding the signed weight as `nibble - 8`.
//!   - `scale`: one `f32` per output row.
//!
//! Dequantized weight `W[o,i] = (nibble - 8) * scale[o]`.

use crate::snapshot::{Int4Matrix, Int8Matrix};

/// Row stride in bytes for an int4 matrix with `i_dim` input columns.
pub fn row_bytes(i_dim: usize) -> usize {
    i_dim.div_ceil(2)
}

/// Read an F32 tensor's raw little-endian bytes into a `Vec<f32>`. For
/// **O-length** tensors only — norm weights and embeddings — loaded once at
/// startup. NEVER call this on a packed `.weight` tensor: that would dequant an
/// int4 expert into host f32 and blow up per-token RAM traffic ~4×. Expert
/// weights are reached only as packed bytes via [`Int4Matrix`].
pub fn read_f32(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 4, 0, "F32 tensor length not a multiple of 4");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Decode one nibble at column `i` of a packed row.
#[inline]
fn nibble(row: &[u8], i: usize) -> i32 {
    let byte = row[i >> 1];
    let n = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
    n as i32 - 8
}

/// One little-endian f32 scale at row `o` of the raw scale bytes.
#[inline]
fn scale_at(scale: &[u8], o: usize) -> f32 {
    let b = &scale[o * 4..o * 4 + 4];
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Accumulate `scalar × row(o)` of an int4 matrix into `acc` (length `i_dim`):
/// `acc[i] += scalar * (nibble(o,i) - 8) * scale[o]`. The MLA absorb primitive
/// (colibri `qt_addrow`): folds a query component through one `kv_b` row.
pub fn addrow(w: &Int4Matrix, o: usize, scalar: f32, acc: &mut [f32]) {
    debug_assert_eq!(acc.len(), w.i_dim);
    let rb = row_bytes(w.i_dim);
    let row = &w.packed[o * rb..(o + 1) * rb];
    let s = scalar * scale_at(w.scale, o);
    for (i, a) in acc.iter_mut().enumerate() {
        *a += nibble(row, i) as f32 * s;
    }
}

/// GEMV over a contiguous row range `[row0, row0+y.len())` of an int4 matrix:
/// `y[j] = scale[row0+j] * Σ_i x[i] * nibble(row0+j, i)`. The MLA value
/// projection (colibri `qt_matvec_rows`): projects the weighted latent through
/// `kv_b`'s value rows for one head.
pub fn matvec_i4_rows(y: &mut [f32], x: &[f32], w: &Int4Matrix, row0: usize) {
    debug_assert_eq!(x.len(), w.i_dim);
    debug_assert!(row0 + y.len() <= w.o_dim);
    let rb = row_bytes(w.i_dim);
    for (j, yj) in y.iter_mut().enumerate() {
        let o = row0 + j;
        let row = &w.packed[o * rb..(o + 1) * rb];
        let mut acc = 0.0f32;
        for (i, &xi) in x.iter().enumerate() {
            acc += xi * nibble(row, i) as f32;
        }
        *yj = acc * scale_at(w.scale, o);
    }
}

/// Reference GEMV against a validated int4 matrix: `y[o] = scale[o] * Σ_i
/// x[i] * w(o,i)`, where `w = nibble(o,i)` (the accessor applies the −8
/// offset). The scale is decoded inline from bytes — no per-expert `Vec<f32>`
/// allocation on the hot path. `w`'s length invariants are guaranteed by
/// `Int4Matrix`'s constructor. This is the scalar oracle the HIP kernel is
/// validated against (M2).
pub fn matvec_i4(y: &mut [f32], x: &[f32], w: &Int4Matrix) {
    debug_assert_eq!(y.len(), w.o_dim);
    // The full GEMV is the row-range GEMV starting at row 0.
    matvec_i4_rows(y, x, w, 0);
}

/// Dequantize row `row` of a per-row **int8** matrix into `out` (length =
/// row width): `out[i] = (int8)packed[row*dim + i] * scale[row]`. Used for the
/// embedding table and lm_head, which the snapshot keeps as int8 (one byte per
/// weight) rather than int4 — a distinct, small tensor class from the int4
/// expert weights that dominate per-token traffic.
pub fn dequant_int8_row(w: &Int8Matrix, row: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), w.i_dim);
    debug_assert!(row < w.o_dim, "int8 row {row} >= o_dim {}", w.o_dim);
    let s = scale_at(w.scale, row);
    let base = row * w.i_dim;
    for (i, o) in out.iter_mut().enumerate() {
        *o = (w.packed[base + i] as i8) as f32 * s;
    }
}

/// GEMV against a per-row **int8** matrix `W[o_dim, i_dim]` (row major, one
/// byte per weight + per-row f32 scale) — the lm_head projection to logits.
/// `y[o] = scale[o] * Σ_i (int8)packed[o·i_dim+i] * x[i]`.
pub fn matvec_i8(y: &mut [f32], x: &[f32], w: &Int8Matrix) {
    debug_assert_eq!(y.len(), w.o_dim);
    debug_assert_eq!(x.len(), w.i_dim);
    for (o, (yo, row)) in y.iter_mut().zip(w.packed.chunks_exact(w.i_dim)).enumerate() {
        let mut acc = 0.0f32;
        for (&b, &xi) in row.iter().zip(x) {
            acc += (b as i8) as f32 * xi;
        }
        *yo = acc * scale_at(w.scale, o);
    }
}

/// GEMV against a plain f32 weight matrix `W[o_dim, i_dim]` (row major),
/// borrowing the raw little-endian bytes from the mmap and decoding inline —
/// no `Vec<f32>` materialization. Used for the F32 router gate on the per-token
/// path (the gate is the only unquantized weight; everything else is int4).
pub fn matvec_f32_bytes(y: &mut [f32], x: &[f32], w_bytes: &[u8], i_dim: usize) {
    debug_assert_eq!(x.len(), i_dim);
    debug_assert_eq!(w_bytes.len(), y.len() * i_dim * 4);
    for (yo, row) in y.iter_mut().zip(w_bytes.chunks_exact(i_dim * 4)) {
        let mut acc = 0.0f32;
        for (c, &xi) in row.chunks_exact(4).zip(x) {
            acc += f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * xi;
        }
        *yo = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack a float matrix per-row exactly as colibri does, so the round-trip
    /// through `matvec_i4` reproduces the quantized dot product.
    fn pack_row(vals: &[i32]) -> Vec<u8> {
        let rb = row_bytes(vals.len());
        let mut out = vec![0u8; rb];
        for (i, &v) in vals.iter().enumerate() {
            let nib = ((v + 8).clamp(0, 15)) as u8;
            if i & 1 == 0 {
                out[i >> 1] |= nib;
            } else {
                out[i >> 1] |= nib << 4;
            }
        }
        out
    }

    #[test]
    fn nibble_roundtrip_both_lanes() {
        // I=4 → one row of 2 bytes; check low/high nibble columns decode right.
        let row = pack_row(&[-8, 7, 0, 3]); // maps to nibbles 0,15,8,11
        assert_eq!(nibble(&row, 0), -8);
        assert_eq!(nibble(&row, 1), 7);
        assert_eq!(nibble(&row, 2), 0);
        assert_eq!(nibble(&row, 3), 3);
    }

    /// Little-endian scale bytes for a set of f32 scales.
    fn scale_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn matvec_matches_hand_dot() {
        // W = [[1,-2, 3, 0], [-1, 4,-8, 7]], scale = [0.5, 2.0], x = [1,1,1,1].
        let mut packed = Vec::new();
        packed.extend(pack_row(&[1, -2, 3, 0]));
        packed.extend(pack_row(&[-1, 4, -8, 7]));
        let scale = scale_bytes(&[0.5, 2.0]);
        let w = Int4Matrix {
            packed: &packed,
            scale: &scale,
            o_dim: 2,
            i_dim: 4,
        };
        let x = [1.0f32, 1.0, 1.0, 1.0];
        let mut y = [0.0f32; 2];
        matvec_i4(&mut y, &x, &w);
        // row0: (1-2+3+0)=2 *0.5 = 1.0 ; row1: (-1+4-8+7)=2 *2.0 = 4.0
        assert!((y[0] - 1.0).abs() < 1e-6, "y0={}", y[0]);
        assert!((y[1] - 4.0).abs() < 1e-6, "y1={}", y[1]);
    }

    #[test]
    fn read_f32_roundtrips() {
        let vals = [1.5f32, -2.25, 0.0, 3.75];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(read_f32(&bytes), vals);
    }

    #[test]
    fn odd_input_dim_row_bytes() {
        // I=5 → rb=3 bytes; last byte holds one used nibble + one pad.
        assert_eq!(row_bytes(5), 3);
        let packed = pack_row(&[2, 2, 2, 2, 2]); // 5 nibbles
        let scale = scale_bytes(&[1.0]);
        let w = Int4Matrix {
            packed: &packed,
            scale: &scale,
            o_dim: 1,
            i_dim: 5,
        };
        let x = [1.0f32; 5];
        let mut y = [0.0f32; 1];
        matvec_i4(&mut y, &x, &w);
        assert!((y[0] - 10.0).abs() < 1e-6, "y0={}", y[0]);
    }
}
