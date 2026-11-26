//! Inkling-2 Small's MXFP8 routed-expert pair (`moe_gateup_mxfp8_situ`,
//! `moe_down_mxfp8_weighted`) — the K3 fp4-SiTU pair's arithmetic at MXFP8's weight format.
//! No C twin (owner direction 2026-11-24): Inkling-2's kernels are scored by oracle suites
//! against the first-party CPU references, not by the C→Rust bit-identity gate.
//!
//! What it shares with `moe_f4.rs` is `moe.rs`'s skeleton — the row mapping, the two passes'
//! shapes, the fixed-point drain — and the SiTU-GLU activation, so nothing here is a second
//! composition of the expert; what it differs in is the WEIGHT format, and that is the whole
//! of [`crate::prelude::dot_mxfp8`]:
//!
//! - **The scale is row-wise `[1, 32]` and a raw ue8m0 byte** (trap T1), not fp4's nibble
//!   pair and not fp8's 128×128 tile grid.
//! - **`h` stays f32 between the passes.** The fp4 pair rounds it through `rbf16` because
//!   K3's `w2` takes a bf16 latent; Inkling-2's MoE runs `situ_glu` and the down projection
//!   in fp32 (`Inkling2Experts` fp32 anchor / bf16 model: bf16 is 1 ulp = 2^-8 of the model,
//!   and the routed experts' own reference accumulates fp32), so truncating here would be an
//!   error the reference does not have, at no saving.
//! - **The routing weight lands on pass 2's output**, as in K3 (`moe_infer`'s
//!   `.mul_(topk_weight)`) — but in f32, not through a bf16 store, for the same reason.
//! - **Pass 1 reads the activation row ONCE for the pair.** Gate and up are the same
//!   `[inter, hidden]` shape over the same `x`, so [`dot_mxfp8_pair_wave`] spends one read
//!   of `x` on both rows instead of two sequential dots reading it twice; pass 2 is a
//!   single row and keeps [`dot_mxfp8_wave`].
//!
//! - **Nothing here allocates shared memory or barriers.** The e4m3 weights decode
//!   arithmetically in registers ([`crate::prelude::dot_mxfp8`]'s note), so neither kernel
//!   builds a table, neither requests dynamic LDS, and there is no block-wide setup step for
//!   an early return to strand: `gate_up`'s row guard may return a thread at any point.

use crate::moe::{down, gate_up, routed};
use crate::prelude::dot_mxfp8::{MXFP8_GROUP, MxFp8Row, dot_mxfp8_pair_wave, dot_mxfp8_wave};
use crate::prelude::situ_glu;
use rivoli_abi::ExpertDescF4;
use rivoli_abi::kernarg::{MoeF4Situ, MoeRange};

// The launcher refuses by the ABI crate's group, the dot strides by the prelude's.
const _: () = assert!(MXFP8_GROUP == rivoli_abi::geom::F4_GROUP as usize);

/// Unroll depth of the 16-byte pass on the DOWN dot: ONE trip, keeping the live set under
/// gfx1151's 96-VGPR/wave occupancy cliff (measured: U=2 hit 100 VGPR/wave, dropping wave
/// occupancy from 12 to 10 waves/SIMD).
const DOWN_UNROLL: usize = 1;
/// Unroll depth of the same pass on the fused PAIR: ONE trip, because a pair step already
/// carries two weight streams (1 KB) beside the shared activation, and a second copy would
/// hoist 2 KB through a body whose live set is twice the single dot's.
const PAIR_UNROLL: usize = 1;

/// One MXFP8 row of `w`/`scale`: row `row` of a `[rows, i_dim]` e4m3 plane and of its
/// `[rows, ceil(i_dim / 32)]` ue8m0 scale plane.
#[inline(always)]
unsafe fn mxfp8_row(w: *const u8, scale: *const u8, (row, i_dim): (usize, i32)) -> MxFp8Row {
    let dim = i_dim as usize;
    let groups = dim.div_ceil(MXFP8_GROUP);
    MxFp8Row {
        w: w.add(row * dim),
        scale: scale.add(row * groups),
        dim,
    }
}

/// Pass 1's dots and activation: `situ_glu(g, u)` in f32, every expert in the range.
#[inline(always)]
unsafe fn gateup_mxfp8(r: &MoeRange<ExpertDescF4>, (b1, b2): (f32, f32)) {
    let hidden = r.span.hidden;
    let dots = |d: &ExpertDescF4, j, x, lane| {
        let g = mxfp8_row(d.gate_packed, d.gate_scale, (j, hidden));
        let u = mxfp8_row(d.up_packed, d.up_scale, (j, hidden));
        // ONE read of `x` for both rows — the two are the same width over the same
        // activation, which is the whole reason the fused dot exists.
        let (gate, up) = dot_mxfp8_pair_wave::<PAIR_UNROLL>(x, g, u, lane);
        ([gate], [up])
    };
    gate_up(r, dots, |_, g, u| situ_glu(g, u, b1, b2));
}

/// Pass 2: the MXFP8 down dot, the routing weight applied to its f32 output, `w == 0`
/// skipped (`moe.rs::routed` argues why the skip is correctness).
#[inline(always)]
unsafe fn down_mxfp8(r: &MoeRange<ExpertDescF4>) {
    let inter = r.span.inter;
    let dot = |d: &ExpertDescF4, o, h, lane| {
        let row = mxfp8_row(d.down_packed, d.down_scale, (o, inter));
        [dot_mxfp8_wave::<DOWN_UNROLL>(h, row, lane)]
    };
    down(r, dot, routed(r.wexpert));
}

/// Inkling-2 Small's pass 1: `situ_glu(g, u, b1, b2)` in f32 for every row of the range.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_gateup_mxfp8_situ(a: MoeF4Situ) {
    gateup_mxfp8(&a.range, (a.b1, a.b2));
}

/// Inkling-2 Small's pass 2: `fixed(f32 dv · w)`, skipped where `w == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_down_mxfp8_weighted(a: MoeF4Situ) {
    down_mxfp8(&a.range);
}

/// Inkling-2's fp32-latent MXFP4 pass 1: `situ_glu(g, u)` kept in f32 (no K3 `rbf16` store).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_gateup_f4_situ_f32(a: MoeF4Situ) {
    crate::moe_f4::gateup_f4(&a.range, |_, g, u| situ_glu(g, u, a.b1, a.b2));
}

/// Inkling-2's fp32-latent MXFP4 pass 2: `fixed(dv · w)` in f32, skipped where `w == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_down_f4_weighted_f32(a: MoeF4Situ) {
    let routed = routed(a.range.wexpert);
    crate::moe_f4::down_f4(&a.range, |k, dv| routed(k, dv));
}
