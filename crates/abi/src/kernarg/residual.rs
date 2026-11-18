//! `kernels/residual.hip`'s kernels: the mHC pair, the attention residual and the gated
//! residual collapse.

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
const _: () = assert!(size_of::<HcPre>() == 72);

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
const _: () = assert!(size_of::<HcPost>() == 48);

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
const _: () = assert!(size_of::<AttnRes>() == 48);

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
const _: () = assert!(size_of::<GatedResidual>() == 48);
