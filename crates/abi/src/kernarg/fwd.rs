//! `kernels/fwd.hip`'s kernels: the forward pass's glue — embedding, KV append, gathers,
//! elementwise gates, argmax and the two divergence instruments.

/// `embed_i8_row` — `x[i] = (int8)packed[token·hidden + i]·scale[token]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbedI8 {
    pub packed: *const u8,
    pub scale: *const f32,
    pub x: *mut f32,
    pub token: i32,
    pub hidden: i32,
}
const _: () = assert!(size_of::<EmbedI8>() == 32);

/// `append_kv` — one token's latent (fp8 + per-128 scale) and roped key (bf16) at row `pos`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AppendKv {
    pub latent: *const f32,
    pub rope: *const f32,
    pub lc8: *mut u8,
    pub lscale: *mut f32,
    pub rc: *mut u16,
    pub pos: i32,
    pub kvl: i32,
    pub ropn: i32,
    pub n_blocks: i32,
}
const _: () = assert!(size_of::<AppendKv>() == 56);

/// `gather_rope` — `qrope[head·ropn + d] = q[head·qh + nope + d]` for `h` heads.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GatherRope {
    pub q: *const f32,
    pub qrope: *mut f32,
    pub h: i32,
    pub qh: i32,
    pub nope: i32,
    pub ropn: i32,
}
const _: () = assert!(size_of::<GatherRope>() == 32);

/// `vadd` (`x += y`) and `sigmoid_gate` (`x *= sigmoid(y)`) — `x[i] op= f(y[i])`, `i < n`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InPlaceBinary {
    pub x: *mut f32,
    pub y: *const f32,
    pub n: i32,
}
const _: () = assert!(size_of::<InPlaceBinary>() == 24);

/// `logit_softcap` — `x[i] = cap·tanh(x[i]·mult/cap)` for finite `x[i]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Softcap {
    pub x: *mut f32,
    pub n: i32,
    pub mult: f32,
    pub cap: f32,
}
const _: () = assert!(size_of::<Softcap>() == 24);

/// `argmax` — lowest index of the maximum of `logits[0..n]` (NaN never wins), and its value.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Argmax {
    pub logits: *const f32,
    pub out_idx: *mut i32,
    pub out_val: *mut f32,
    pub n: i32,
}
const _: () = assert!(size_of::<Argmax>() == 32);

/// `flag_nonfinite` — `tag` into `*flag` (first writer wins) if any of `x[0..n]` is
/// non-finite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FlagNonfinite {
    pub x: *const f32,
    pub flag: *mut u32,
    pub n: i32,
    pub tag: u32,
}
const _: () = assert!(size_of::<FlagNonfinite>() == 24);

/// `hash_rows` — XOR-fold the bits of `x[0], x[stride], … < n` into `*out`, indices offset by
/// `i_base`; `grid_stride` is `gridDim · blockDim`, the launcher's choice (the Phase 0c rule:
/// no dispatch-packet reads).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HashRows {
    pub x: *const f32,
    pub out: *mut u64,
    pub i_base: u64,
    pub grid_stride: i64,
    pub n: i32,
    pub stride: i32,
}
const _: () = assert!(size_of::<HashRows>() == 40);
