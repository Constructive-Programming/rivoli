//! `kernels/moe.hip`'s and `moe_f4.hip`'s kernels: the routed-expert ranges in three weight
//! formats and the fixed-point drains they land in.
//!
//! **One block per FORMAT, not per kernel.** A launcher issues gate/up then down for one
//! expert range, and both passes read the same range, descriptors, `h` and routing weights,
//! so both kernels take the same struct and each reads the fields its pass needs. Splitting it
//! per pass would restate the shared prefix twice and let the two passes of one launch disagree
//! about the range — which no check downstream would notice.

use crate::{ExpertDesc, ExpertDescF4};

/// The absolute expert range `[e_start, e_start + e_count)` of a `[hidden, inter]` expert.
/// `hidden` is the activation width — K3's LATENT 3584 for the situ pair (`expert_in`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MoeSpan {
    pub hidden: i32,
    pub inter: i32,
    pub e_start: i32,
    pub e_count: i32,
}
const _: () = assert!(size_of::<MoeSpan>() == 16);

/// The buffers every format's range shares: `R` token rows of `x` (row-fastest everywhere —
/// `h[(e·R + t)·inter + j]`, `wexpert[e·R + t]`, `acc[t·hidden + o]`), the descriptor array,
/// the per-expert `h` staging, the routing weights and the fixed-point accumulator.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MoeRange<D> {
    pub x: *const f32,
    pub descs: *const D,
    pub h: *mut f32,
    pub wexpert: *const f32,
    pub acc: *mut u64,
    pub span: MoeSpan,
}
const _: () = assert!(size_of::<MoeRange<ExpertDesc>>() == 56);

/// The int4 range: exactly a [`MoeRange`] over the shared six-pointer descriptor.
pub type MoeI4 = MoeRange<ExpertDesc>;

/// The VQ-int3 range: a [`MoeRange`] plus the three per-projection fp16 codebooks
/// `[gate, up, down]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MoeVq {
    pub range: MoeRange<ExpertDesc>,
    pub cb: [*const u16; 3],
}
const _: () = assert!(size_of::<MoeVq>() == 80);

/// DeepSeek-V4's fp4 range: `wexpert` folds into pass 1's clamped SwiGLU at `limit`; pass 2
/// reads neither.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MoeF4 {
    pub range: MoeRange<ExpertDescF4>,
    pub limit: f32,
}
const _: () = assert!(size_of::<MoeF4>() == 64);

/// Kimi-K3's fp4 range: pass 1 runs SiTU-GLU at `(b1, b2)` and reads no `wexpert`; pass 2
/// applies it to the bf16 `w2` output.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MoeF4Situ {
    pub range: MoeRange<ExpertDescF4>,
    pub b1: f32,
    pub b2: f32,
}
const _: () = assert!(size_of::<MoeF4Situ>() == 64);

/// `moe_acc_drain_to` — `rows` accumulator rows of `n` summed into `out` (`=`), `acc` reset.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AccDrainTo {
    pub out: *mut f32,
    pub acc: *mut u64,
    pub n: i32,
    pub rows: i32,
}
const _: () = assert!(size_of::<AccDrainTo>() == 24);

/// `moe_acc_drain` — the same, folded into the residual: `out += gain · Σ`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AccDrain {
    pub to: AccDrainTo,
    pub gain: f32,
}
const _: () = assert!(size_of::<AccDrain>() == 32);
