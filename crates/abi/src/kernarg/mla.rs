//! `kernels/mla.hip`'s kernels (tag `archive/hip-kernels`): MLA's absorb and value projections
//! through the fp8 `kv_b`, and V4's attention-side helpers (the fp8 prefix quantizer, the batched
//! and QK norms, the adjacent-pair RoPE, the fp8 GEMV with a bf16 store).

/// The fp8-e4m3 block-scaled `kv_b` both MLA projections read: `[h·(nope+vh) rows, kvl cols]`,
/// row stride `kvl` bytes, `block × block` scale tiles. Head `h` owns rows `[h·(nope+vh), +nope)`
/// for the absorb and the next `vh` for the value. One definition because the two halves of one
/// absorb-attend-value chain must read ONE layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Fp8KvB {
    pub w: *const u8,
    pub scale: *const f32,
    pub h: i32,
    pub nope: i32,
    pub vh: i32,
    pub kvl: i32,
    pub block: i32,
}
const _: () = assert!(size_of::<Fp8KvB>() == 40);

/// `mla_absorb_fp8{,_r2}` — `qabs[head][i] = Σ_d q[head·qh + d] · kv_b[rbase + d][i]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlaAbsorb {
    pub q: *const f32,
    pub qabs: *mut f32,
    pub kvb: Fp8KvB,
    pub qh: i32,
}
const _: () = assert!(size_of::<MlaAbsorb>() == 64);

/// `mla_value_fp8{,_r2}` — `ctx[head][j] = Σ_i clat[head][i] · kv_b[rbase + nope + j][i]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlaValue {
    pub clat: *const f32,
    pub ctx: *mut f32,
    pub kvb: Fp8KvB,
}
const _: () = assert!(size_of::<MlaValue>() == 56);

/// `act_quant_f8_prefix` — the first `n` of each `row_stride`-wide row, block-quantized through
/// fp8 into `dst` (which may BE `src`; the launcher refuses a strided copy).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ActQuantPrefix {
    pub src: *const f32,
    pub dst: *mut f32,
    pub rows: i32,
    pub row_stride: i32,
    pub n: i32,
    pub block: i32,
}
const _: () = assert!(size_of::<ActQuantPrefix>() == 32);

/// `rmsnorm_batch` — rows of `d` normed IN PLACE by a learned `w`, stored bf16-rounded.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RmsNormRows {
    pub x: *mut f32,
    pub w: *const f32,
    pub d: i32,
    pub eps: f32,
}
const _: () = assert!(size_of::<RmsNormRows>() == 24);

/// `qk_norm` — rows of `d` normed IN PLACE, weightless, with V4's bf16 statistic.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QkNorm {
    pub q: *mut f32,
    pub d: i32,
    pub eps: f32,
}
const _: () = assert!(size_of::<QkNorm>() == 16);

/// `rope_adjacent` — the last `rd` of each `row_len` row rotated IN PLACE in adjacent pairs at
/// `pos0 + row / rows_per_pos`, the inverse rotation when `inverse != 0`; the row then bf16.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RopeAdjacent {
    pub x: *mut f32,
    pub tbl: *const f32,
    pub row_len: i32,
    pub rd: i32,
    pub pos0: i32,
    pub rows_per_pos: i32,
    pub inverse: i32,
}
const _: () = assert!(size_of::<RopeAdjacent>() == 40);

/// `gemv_fp8_bf16` — `out[r][j] = bf16(x[r, group(j)] · w[j])`, `w` fp8 tile-scaled.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GemvFp8Bf16 {
    pub x: *const f32,
    pub w: *const u8,
    pub wscale: *const f32,
    pub out: *mut f32,
    pub m: i32,
    pub n_out: i32,
    pub k: i32,
    pub block: i32,
    pub groups: i32,
}
const _: () = assert!(size_of::<GemvFp8Bf16>() == 56);
