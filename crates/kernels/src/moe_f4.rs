//! The OCP MXFP4 routed-expert pairs — the twins of `kernels/moe_f4.hip` at tag
//! `archive/hip-kernels`: DeepSeek-V4's (`moe_gateup_f4`/`moe_down_f4`, clamped SwiGLU, the
//! routing weight folded into pass 1) and Kimi-K3's (`moe_gateup_f4_situ`/
//! `moe_down_f4_weighted`, SiTU-GLU, the weight applied to the bf16 `w2` output). That file
//! carries the five V4 differences from the GLM pair and the four K3 substitutions; this one
//! carries what the port decided, on `moe.rs`'s skeleton.
//!
//! - **Fused sites**: none outside the prelude (`dot_nib`'s, and SiTU's ocml `tanh`).
//! - **`gu8p` is not expressible, and not needed.** The C typed the descriptor spans
//!   `address_space(1)` so their loads lower to `global_load`. Rust cannot name an address
//!   space, but copying the descriptor by value (`moe.rs`'s passes) lets rustc prove the
//!   spans' loads unclobbered and infer the same `global_load`s — census in rust-kernels.md §5
//!   (L2).

use crate::moe::{down, gate_up, nib_row, routed};
// `U` as in `moe.rs`; only R=1 is instantiated for fp4.
use crate::prelude::dot_nib::{F4, F4_GROUP, NIBBLE_BATCHED_UNROLL as U, dot_nib_wave_r};
use crate::prelude::formats::rbf16;
use crate::prelude::{situ_glu, swiglu_clamped};
use rivoli_abi::ExpertDescF4;
use rivoli_abi::kernarg::{MoeF4, MoeF4Situ, MoeRange};

// The launcher refuses by the ABI crate's group, the dot strides by the prelude's.
const _: () = assert!(F4_GROUP == rivoli_abi::geom::F4_GROUP as usize);

/// Both pass-1 kernels' row mapping and dots (`moe_f4.hip::f4_gate_up_dots`); `store(e·R + t,
/// g, u)` is the value lane 0 writes — the two kernels' activations.
#[inline(always)]
pub(crate) unsafe fn gateup_f4(r: &MoeRange<ExpertDescF4>, store: impl Fn(usize, f32, f32) -> f32) {
    let hidden = r.span.hidden;
    let dots = |d: &ExpertDescF4, j, x, lane| {
        let g = nib_row::<F4>(d.gate_packed, d.gate_scale, (j, hidden));
        let u = nib_row::<F4>(d.up_packed, d.up_scale, (j, hidden));
        (
            dot_nib_wave_r::<1, U, F4>(x, g, lane),
            dot_nib_wave_r::<1, U, F4>(x, u, lane),
        )
    };
    gate_up(r, dots, store);
}

/// Both pass-2 kernels: the fp4 down dot, `contribution(e·R + t, dv)` deciding what is added.
#[inline(always)]
pub(crate) unsafe fn down_f4(r: &MoeRange<ExpertDescF4>, contribution: impl Fn(usize, f32) -> Option<f32>) {
    let inter = r.span.inter;
    let dot = |d: &ExpertDescF4, o, h, lane| {
        dot_nib_wave_r::<1, U, F4>(
            h,
            nib_row::<F4>(d.down_packed, d.down_scale, (o, inter)),
            lane,
        )
    };
    down(r, dot, contribution);
}

/// V4's pass 1: `bf16(swiglu_clamped(g, u) · w)`, and exactly `0` where `w == 0` — enforced,
/// because the gate has no lower clamp and `NaN · 0` is not `0`.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_gateup_f4(a: MoeF4) {
    gateup_f4(&a.range, |k, g, u| {
        let sw = swiglu_clamped(g, u, a.limit);
        let w = *a.range.wexpert.add(k);
        if w == 0.0 { 0.0 } else { rbf16(sw * w) }
    });
}

/// K3's pass 1: `bf16(situ_glu(g, u))`, every expert in the range, no routing weight.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_gateup_f4_situ(a: MoeF4Situ) {
    gateup_f4(&a.range, |_, g, u| rbf16(situ_glu(g, u, a.b1, a.b2)));
}

/// V4's pass 2: `fixed(bf16(dv))` for every row — the routing weight is already in `h`.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_down_f4(a: MoeF4) {
    down_f4(&a.range, |_, dv| Some(rbf16(dv)));
}

/// K3's pass 2: `fixed(bf16(dv) · w)`, skipped where `w == 0` (`moe_f4.hip` argues why the
/// skip is still needed although SiTU-GLU is bounded).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_down_f4_weighted(a: MoeF4Situ) {
    // `routed` multiplies `w · y` where the C wrote `y · w`: IEEE multiplication commutes, so
    // only a NaN's payload could differ, and `moe_fixed` clamps every NaN to the same bound.
    let routed = routed(a.range.wexpert);
    down_f4(&a.range, |k, dv| routed(k, rbf16(dv)));
}
