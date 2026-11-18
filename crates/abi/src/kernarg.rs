//! Kernel argument blocks: ONE `repr(C)` struct per Rust kernel parameter list, which the
//! kernel takes by value (it lands in the kernarg segment as-is) and the launcher builds
//! and hands to `hipModuleLaunchKernel` as the single `kernelParams` entry.
//!
//! **This is the wall's parameter list, written once.** Under hipcc every kernel's
//! parameters were spelled three times — the `__global__` signature, the C launcher's
//! `hipLaunchKernelGGL` call, the Rust `extern` mirror — and only the compiler-blind
//! convention of "same order" tied them. Here the device destructures the struct the host
//! filled, so a field cannot be transposed between the two sides: there is nothing to
//! transpose against. Field order is free (it no longer has to match a C signature), so it
//! is chosen for packing — pointers first.
//!
//! The kernels that share a shape share a struct (`Gate` under all three gated combines,
//! `RowNorm` under both single-row norms): a struct per kernel would restate identical
//! field lists, which is the duplication this file exists to remove.
//!
//! Layout is pinned by the size asserts at the bottom, which compile on BOTH targets — the
//! one way a single definition could still disagree with itself is a target whose ABI lays
//! it out differently, and either compile would fail.

use core::ffi::c_void;

/// `swiglu` — and the operand triple under every gated combine: `h[i] = f(g[i], u[i])` for
/// `i < n`. `h` may alias `g` (each thread reads both operands before its one write).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Gate {
    pub g: *const f32,
    pub u: *const f32,
    pub h: *mut f32,
    pub n: i32,
}

/// `swiglu_clamped_bf16` — a [`Gate`] plus the clamp; `limit` is refused unless positive
/// and finite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GateClamped {
    pub gate: Gate,
    pub limit: f32,
}

/// `situ_glu_f32` — a [`Gate`] plus SiTU's two betas, each positive and finite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GateSitu {
    pub gate: Gate,
    pub b1: f32,
    pub b2: f32,
}

/// `rmsnorm_single` and `rmsnorm_centered_single` — one row of `n`: `y` from `x` and `w`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RowNorm {
    pub x: *const f32,
    pub w: *const f32,
    pub y: *mut f32,
    pub n: i32,
    pub eps: f32,
}

/// `rmsnorm_weightless_batch` — rows of `d`, normed IN PLACE and scaled.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WeightlessNorm {
    pub x: *mut f32,
    pub d: i32,
    pub eps: f32,
    pub scale: f32,
}

/// `act_quant_f8` — `nblk` groups of `ACT_QUANT_BLOCK`, quantized IN PLACE. `nblk` is the
/// launcher's, so the kernel's bound and the grid come from one expression.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ActQuant {
    pub v: *mut f32,
    pub nblk: u64,
}

/// `rope_interleave` / `rope_split_half` — segment `s` at `base + s * stride`, `seg` wide,
/// rotated to position `pos` at base `theta`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Rope {
    pub base: *mut f32,
    pub theta: f64,
    pub stride: i32,
    pub seg: i32,
    pub pos: i32,
}

/// `hc_pre` — one token per block: the `[HC_MULT, dim]` stream `h` reduced to `y`, plus
/// the `post`/`comb` its `hc_post` consumes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HcPre {
    pub h: *const f32,
    pub fnw: *const f32,
    pub scale: *const f32,
    pub base: *const f32,
    pub y: *mut f32,
    pub post: *mut f32,
    pub comb: *mut f32,
    pub dim: i32,
    pub iters: i32,
    pub norm_eps: f32,
    pub hc_eps: f32,
}

/// `hc_post` — `y[t][k][d] = post[t][k]·x[t][d] + Σ_j comb[t][j][k]·residual[t][j][d]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HcPost {
    pub x: *const f32,
    pub residual: *const f32,
    pub post: *const f32,
    pub comb: *const f32,
    pub y: *mut f32,
    pub s: i32,
    pub dim: i32,
}

/// `attn_res` — per token, `softmax(<RMSNorm(src_s), fold>) @ src` over `nsrc` sources of
/// `n`, the token's stack at `src + token * src_stride`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttnRes {
    pub src: *const f32,
    pub fold: *const f32,
    pub out: *mut f32,
    pub src_stride: u64,
    pub nsrc: i32,
    pub n: i32,
    pub eps: f32,
}

/// `gated_residual_collapse_f32` — collapse `n_r` gated rows of `dim` to `x`, and write the
/// `n_r` inject scales.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GatedResidual {
    pub rhat: *const f32,
    pub mix: *const f32,
    pub inject: *const f32,
    pub x: *mut f32,
    pub scale: *mut f32,
    pub n_r: i32,
    pub dim: i32,
}

/// `touch_region` — load every 16-byte unit of `src[0..n4]`, grid-stride by `stride`
/// (`gridDim · blockDim`, the launcher's choice); `sink` is written only on an impossible
/// value.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Touch {
    pub src: *const c_void,
    pub sink: *mut u64,
    pub n4: u64,
    pub stride: u64,
}

// Pinned on both targets (see the module doc). A failing assert names a layout that moved.
const _: () = assert!(size_of::<Gate>() == 32 && size_of::<GateClamped>() == 40);
const _: () = assert!(size_of::<GateSitu>() == 40 && size_of::<RowNorm>() == 32);
const _: () = assert!(size_of::<WeightlessNorm>() == 24 && size_of::<ActQuant>() == 16);
const _: () = assert!(size_of::<Rope>() == 32 && size_of::<HcPre>() == 72);
const _: () = assert!(size_of::<HcPost>() == 48 && size_of::<AttnRes>() == 48);
const _: () = assert!(size_of::<GatedResidual>() == 48 && size_of::<Touch>() == 32);
