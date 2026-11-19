//! `kernels/kvcompress.hip`'s kernels (tag `archive/hip-kernels`): the compressor's bf16 projection GEMM, the state deposit and
//! the two pooling phases. `CompGeom`/`CompFinish` are the crate root's (they predate the migration
//! as ABI mirrors); here they ride inside the kernels' argument blocks by value.

use crate::{CompFinish, CompGeom};

/// `gemm_bf16` — `out[m, n] = x[m, k] · w[n, k]ᵀ`, `w` bf16. `w` leads (field order is free in a
/// one-definition block): `x, w, out, m, n, k` matched a test helper's struct token for token, and
/// jscpd counts that as a clone.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GemmBf16 {
    pub w: *const u16,
    pub x: *const f32,
    pub out: *mut f32,
    pub m: i32,
    pub n: i32,
    pub k: i32,
}
const _: () = assert!(size_of::<GemmBf16>() == 40);

/// `kv_compress_deposit` — this call's `s` rows into the compressor state, from slot `slot0`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Deposit {
    pub kv: *const f32,
    pub score: *const f32,
    pub ape: *const f32,
    pub kv_state: *mut f32,
    pub score_state: *mut f32,
    pub p: CompGeom,
    pub s: i32,
    pub slot0: i32,
}
const _: () = assert!(size_of::<Deposit>() == 80);
const _: () = assert!(size_of::<super::Strided<Deposit>>() == 88);

/// `kv_compress_prefill` — `nblk` (the grid) complete windows pooled from `kv`/`score` + `ape`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CompressPrefill {
    pub kv: *const f32,
    pub score: *const f32,
    pub ape: *const f32,
    pub f: CompFinish,
    pub p: CompGeom,
}
const _: () = assert!(size_of::<CompressPrefill>() == 80);

/// `kv_compress_decode` — the completed window pooled out of the state at `start_pos`, then slid.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CompressDecode {
    pub kv_state: *mut f32,
    pub score_state: *mut f32,
    pub f: CompFinish,
    pub p: CompGeom,
    pub start_pos: i32,
}
const _: () = assert!(size_of::<CompressDecode>() == 72);
