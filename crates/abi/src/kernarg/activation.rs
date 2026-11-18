//! `kernels/activation.hip`'s kernels: the gated combines, the row norms, the fp8 activation
//! quantizer and RoPE.

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
const _: () = assert!(size_of::<Gate>() == 32);

/// `swiglu_clamped_bf16` — a [`Gate`] plus the clamp; `limit` is refused unless positive
/// and finite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GateClamped {
    pub gate: Gate,
    pub limit: f32,
}
const _: () = assert!(size_of::<GateClamped>() == 40);

/// `situ_glu_f32` — a [`Gate`] plus SiTU's two betas, each positive and finite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GateSitu {
    pub gate: Gate,
    pub b1: f32,
    pub b2: f32,
}
const _: () = assert!(size_of::<GateSitu>() == 40);

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
const _: () = assert!(size_of::<RowNorm>() == 32);

/// `rmsnorm_weightless_batch` — rows of `d`, normed IN PLACE and scaled.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WeightlessNorm {
    pub x: *mut f32,
    pub d: i32,
    pub eps: f32,
    pub scale: f32,
}
const _: () = assert!(size_of::<WeightlessNorm>() == 24);

/// `act_quant_f8` — `nblk` groups of `ACT_QUANT_BLOCK`, quantized IN PLACE. `nblk` is the
/// launcher's, so the kernel's bound and the grid come from one expression.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ActQuant {
    pub v: *mut f32,
    pub nblk: u64,
}
const _: () = assert!(size_of::<ActQuant>() == 16);

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
const _: () = assert!(size_of::<Rope>() == 32);
