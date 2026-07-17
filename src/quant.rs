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

/// Row stride in bytes for an int4 matrix with `i_dim` input columns.
pub fn row_bytes(i_dim: usize) -> usize {
    i_dim.div_ceil(2)
}

/// Read an F32 tensor's raw little-endian bytes into a `Vec<f32>`. Used for
/// norm weights and per-row int4 scales (the `.qs` tensors). Copies — these
/// are small (O values), and the mmap has no 4-byte alignment guarantee, so a
/// zero-copy cast would be unsound.
pub fn read_f32(bytes: &[u8]) -> Vec<f32> {
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

/// Reference GEMV: `y[o] = scale[o] * Σ_i x[i] * w(o,i)`, where the signed
/// weight `w = nibble(o,i)` (the `nibble` accessor already applies the −8
/// offset). For a single activation row `x` of length `i_dim` against
/// `W[o_dim, i_dim]`. This is the scalar oracle the HIP kernel is validated
/// against (M2).
pub fn matvec_i4(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scale: &[f32],
    i_dim: usize,
    o_dim: usize,
) {
    debug_assert_eq!(y.len(), o_dim);
    debug_assert_eq!(x.len(), i_dim);
    debug_assert_eq!(scale.len(), o_dim);
    let rb = row_bytes(i_dim);
    debug_assert_eq!(packed.len(), rb * o_dim);
    for o in 0..o_dim {
        let row = &packed[o * rb..(o + 1) * rb];
        let mut acc = 0.0f32;
        for (i, &xi) in x.iter().enumerate() {
            acc += xi * nibble(row, i) as f32;
        }
        y[o] = acc * scale[o];
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

    #[test]
    fn matvec_matches_hand_dot() {
        // W = [[1,-2, 3, 0], [-1, 4,-8, 7]], scale = [0.5, 2.0], x = [1,1,1,1].
        let i_dim = 4;
        let o_dim = 2;
        let mut packed = Vec::new();
        packed.extend(pack_row(&[1, -2, 3, 0]));
        packed.extend(pack_row(&[-1, 4, -8, 7]));
        let scale = [0.5f32, 2.0];
        let x = [1.0f32, 1.0, 1.0, 1.0];
        let mut y = [0.0f32; 2];
        matvec_i4(&mut y, &x, &packed, &scale, i_dim, o_dim);
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
        let x = [1.0f32; 5];
        let scale = [1.0f32];
        let mut y = [0.0f32; 1];
        matvec_i4(&mut y, &x, &packed, &scale, 5, 1);
        assert!((y[0] - 10.0).abs() < 1e-6, "y0={}", y[0]);
    }
}
