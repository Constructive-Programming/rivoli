//! The VQ-int3 CODEBOOK dot: 12-bit indices into a shared fp16 codebook of 4-vectors, one
//! bf16 scale per 64 weights. Port of `dot_vq.hpp` at tag `archive/hip-kernels`.
//!
//! The codebook is fp16 so all of it (4096·4·2 = 32 KB) fits the L1 the random gather hits.
//! Fused sites: the subvector dot is `rowview::quad`, then `acc = fma(s, dot, acc)`.

use core::intrinsics::fmuladd;

use super::formats::bf16f;
use super::reduce::{WAVE, wave_sum};
use super::rowview::{F32x4, RowsView, for_strided, quad};

/// Must match quant.rs.
pub const VQ_DIM: usize = 4;
pub const VQ_K: usize = 4096;
pub const VQ_INDEX_BITS: usize = 12;
pub const VQ_GROUP: usize = 64;
pub const VQ_SUBS_PER_GROUP: usize = VQ_GROUP / VQ_DIM;

/// One VQ weight row and the codebook it indexes: subvector `t` reads the 12-bit index at bit
/// `12t` of `idx`, dots `x[4t..]` with `cb[4·index..]`, and scales by bf16
/// `scale[t / 16]`. `i_dim` is a multiple of 8, which keeps the two-byte index read in
/// bounds (quant.rs `get_idx`/`vq_row_bytes`).
#[derive(Clone, Copy)]
pub struct VqRow {
    pub idx: *const u8,
    pub scale: *const u16,
    pub cb: *const f16,
    pub i_dim: usize,
}

/// Two codebook halves, the 4-byte `__half2` load.
#[derive(Clone, Copy)]
#[repr(C, align(4))]
struct F16x2([f16; 2]);

/// R activation rows against ONE read of the weight row, result on lane 0 — one decode, R
/// dots, which is what makes a speculative verify pass cheap. R=1 is the single-row form.
///
/// # Safety
/// `w`'s spans must be readable for its `i_dim`, `cb` for every index the row holds, and each
/// of the `R` rows of `x` 16-byte aligned and readable over `i_dim`. Every lane of the wave
/// must call it.
#[inline(always)]
pub unsafe fn dot_vq_wave_r<const R: usize>(x: RowsView, w: VqRow, lane: usize) -> [f32; R] {
    let mut acc = [0.0f32; R];
    // SAFETY: the caller's obligation; `t < nsub` keeps `byte + 1` inside the row's index
    // bytes and `4t + 3 < i_dim`.
    unsafe {
        for_strided::<1>(lane, w.i_dim / VQ_DIM, WAVE, |t| {
            let bitpos = t * VQ_INDEX_BITS;
            let raw = u32::from(*w.idx.add(bitpos >> 3))
                | (u32::from(*w.idx.add((bitpos >> 3) + 1)) << 8);
            let c =
                w.cb.add((((raw >> (bitpos & 7)) & 0xFFF) as usize) * VQ_DIM);
            let (c01, c23) = (*c.cast::<F16x2>(), *c.add(2).cast::<F16x2>());
            let cv = [
                c01.0[0] as f32,
                c01.0[1] as f32,
                c23.0[0] as f32,
                c23.0[1] as f32,
            ];
            let s = bf16f(*w.scale.add(t / VQ_SUBS_PER_GROUP));
            for (r, a) in acc.iter_mut().enumerate() {
                let xv = F32x4::load(x.at(r, t * VQ_DIM)).0;
                *a = fmuladd(s, quad(xv, cv), *a);
            }
        });
    }
    acc.map(wave_sum)
}
