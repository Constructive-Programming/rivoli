//! `kernels/recurrent.hip`'s kernels: the two gated delta rules, K3's short convolution and
//! the gated head norm. Every reduction width the C read off `blockDim.x` is a field here or
//! is derived on both sides by [`crate::geom::reduce_block`] (Phase 0c's rule).

/// The seven buffers both gated delta rules READ, every one RAW (the kernel applies the norms,
/// the sigmoid and the decay). The two models name two of them differently and mean the same
/// role: `decay` is K3's `g` and Qwen's `a`, `beta` is K3's `beta_pre` and Qwen's `b`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DeltaInputs {
    pub q: *const f32,
    pub k: *const f32,
    pub v: *const f32,
    pub decay: *const f32,
    pub beta: *const f32,
    pub a_log: *const f32,
    pub dt_bias: *const f32,
}
const _: () = assert!(size_of::<DeltaInputs>() == 56);

/// `gated_delta_recurrent_f32` — one block of `head_dim` threads per head; `head_dim` IS the
/// block (the launcher refuses a non-power-of-two, 1003), and both state loops run over it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GatedDelta {
    pub inp: DeltaInputs,
    pub state: *mut f32,
    pub out: *mut f32,
    pub head_dim: i32,
    pub lower_bound: f32,
}
const _: () = assert!(size_of::<GatedDelta>() == 80);

/// `short_conv_silu_f32` — one thread per channel, `win` shifted and appended in place.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ShortConv {
    pub cur: *const f32,
    pub w: *const f32,
    pub win: *mut f32,
    pub out: *mut f32,
    pub channels: i32,
    pub taps: i32,
}
const _: () = assert!(size_of::<ShortConv>() == 40);

/// `rmsnorm_gate_heads_f32` — one block per head of `reduce_block(head_dim)` threads, the
/// lanes past `head_dim` masked.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GateHeads {
    pub o: *const f32,
    pub gate: *const f32,
    pub weight: *const f32,
    pub out: *mut f32,
    pub head_dim: i32,
    pub eps: f32,
}
const _: () = assert!(size_of::<GateHeads>() == 40);

/// `gdn_recurrent_f32` — one block per V head of `reduce_block(max(dk, dv))` threads; `rep`
/// is `v_heads / qk_heads`, which the launcher derives after refusing a ragged ratio (1003).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Gdn {
    pub inp: DeltaInputs,
    pub state: *mut f32,
    pub out: *mut f32,
    pub rep: i32,
    pub dk: i32,
    pub dv: i32,
}
const _: () = assert!(size_of::<Gdn>() == 88);
