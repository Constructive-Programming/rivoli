//! `kernels/headtail.hip`'s kernels: V4's bf16 embedding broadcast and `Block.hc_head`.

/// `embed_bf16_row_bcast` — token `token`'s bf16 row of `w` into `hc` f32 copies of `x`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbedBf16 {
    pub w: *const u16,
    pub x: *mut f32,
    pub token: i32,
    pub hidden: i32,
    pub hc: i32,
}
const _: () = assert!(size_of::<EmbedBf16>() == 32);

/// `hc_head_collapse_gate` — one wave per token: `pre[t][j]` from the `[hc, dim]` stream `h`,
/// the `[hc, hc·dim]` `fnw`, `base[hc]` and the ONE-element `scale`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HcHeadGate {
    pub h: *const f32,
    pub fnw: *const f32,
    pub base: *const f32,
    pub scale: *const f32,
    pub pre: *mut f32,
    pub s: i32,
    pub hc: i32,
    pub dim: i32,
    pub eps: f32,
    pub hc_eps: f32,
}
const _: () = assert!(size_of::<HcHeadGate>() == 64);

/// `hc_head_collapse_blend` — `y[t][d] = bf16(Σ_c pre[t][c]·h[t][c][d])`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HcHeadBlend {
    pub h: *const f32,
    pub pre: *const f32,
    pub y: *mut f32,
    pub hc: i32,
    pub dim: i32,
    pub s: i32,
}
const _: () = assert!(size_of::<HcHeadBlend>() == 40);
