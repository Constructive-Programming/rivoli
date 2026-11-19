//! `kernels/linalg.hip`'s kernels: the batch-1 GEMVs and the offline VQ encoder.

/// Every GEMV: `y[r·o_dim + o] = Σ_i x[r·i_dim + i]·W[o, i]`, with `W` decoded from `w` —
/// the one field the formats differ in, so each format is an instantiation, not a restated
/// field list.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Gemv<W> {
    pub x: *const f32,
    pub w: W,
    pub y: *mut f32,
    pub o_dim: i32,
    pub i_dim: i32,
}

/// `gemv_i8` (one f32 scale per row), `gemv_i4` (one per 128-column group) and, through
/// [`GemvFp8`], fp8's tile grid: packed codes and their f32 scales.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Scaled {
    pub packed: *const u8,
    pub scale: *const f32,
}
const _: () = assert!(size_of::<Gemv<Scaled>>() == 40);
// `gemv_f32`: plain f32 weights `w[o·i_dim + i]`.
const _: () = assert!(size_of::<Gemv<*const f32>>() == 32);

/// `gemv_vq`: 12-bit `indices` into the shared fp16 `codebook` (passed as its bits), one
/// bf16 `scales` entry per 64 weights.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vq {
    pub indices: *const u8,
    pub scales: *const u16,
    pub codebook: *const u16,
}
const _: () = assert!(size_of::<Gemv<Vq>>() == 48);

/// `gemv_fp8` and its split-K form — a [`Scaled`] GEMV whose scale is a `block x block` tile
/// grid (`block` a power of two; the launcher refuses otherwise).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GemvFp8 {
    pub gemv: Gemv<Scaled>,
    pub block: i32,
}
const _: () = assert!(size_of::<GemvFp8>() == 48);

/// `vq_encode` — `idx[i] = argmin_k (cbnorm[k] − 2·sub_i·codebook_k)` over `n` subvectors.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VqEncode {
    pub sub: *const f32,
    pub codebook: *const f32,
    pub cbnorm: *const f32,
    pub idx: *mut u16,
    pub n: i32,
}
const _: () = assert!(size_of::<VqEncode>() == 40);
