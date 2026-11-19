//! GLM's routed-expert ranges and the fixed-point drains every format shares — the twins of
//! `kernels/moe.hip` at tag `archive/hip-kernels` (the VQ-int3 and int4 pairs, `R` = 1 and
//! 2 token rows, and `moe_acc_drain{,_to}`). That file carries the batching layout argument,
//! the `w != 0` skip's correctness argument and the drains' contracts; this one carries what
//! the port decided. The skeleton (row mapping, the two passes' shapes, the atomic add) is
//! shared with `moe_f4.rs`.
//!
//! - **Fused sites**: none outside the prelude dots, and the drain's `x += gain·f`
//!   (`fma(gain, f, x)`, a `v_fmac` in hipcc's ISA).
//! - **The atomic** is `atomicrmw add … syncscope("agent") monotonic` in hipcc's IR; see
//!   [`acc_add`].
//! - **No `blockDim`**: the passes index by `ROWS_PER_BLOCK` waves, the drains by
//!   `ELEMENTWISE_BLOCK` threads — the constants their launchers launch with.

use crate::gpu::{bid, fma, tid};
// `U`: the batched nibble dots' unroll (`dot_i4_wave_r<R>`/`dot_f4_wave_r<R>`'s `#pragma unroll 4`).
use crate::prelude::dot_nib::{I4, NIBBLE_BATCHED_UNROLL as U, NibFormat, NibRow, dot_nib_wave_r};
use crate::prelude::dot_vq::{VQ_DIM, VQ_GROUP, VQ_INDEX_BITS, VqRow, dot_vq_wave_r};
use crate::prelude::reduce::{ROWS_PER_BLOCK, WAVE};
use crate::prelude::rowview::RowsView;
use crate::prelude::{MOE_ACC_SCALE, moe_fixed};
use core::intrinsics::exp;
use rivoli_abi::ExpertDesc;
use rivoli_abi::geom::ELEMENTWISE_BLOCK;
use rivoli_abi::kernarg::{AccDrain, AccDrainTo, MoeI4, MoeRange, MoeSpan, MoeVq};

/// `x / (1 + e^-x)` — `moe.hip::siluf`, the division form.
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

/// The wave's `(lane, e, j)`: wave `r` of the grid is row `j = r % len` of absolute expert
/// `e = e_start + r / len`, in the C's `size_t` arithmetic (an unsigned 64-bit divide, as
/// hipcc emits). `None` for the waves past `e_count · len` (uniform per wave).
#[inline(always)]
fn wave_row(span: MoeSpan, len: i32) -> Option<(usize, usize, usize)> {
    let lane = tid() as usize & (WAVE - 1);
    let r = bid() as usize * ROWS_PER_BLOCK + tid() as usize / WAVE;
    let len = len as usize;
    if r >= span.e_count as usize * len {
        return None;
    }
    let e = span.e_start + (r / len) as i32;
    Some((lane, e as usize, r % len))
}

/// Pass 1 over any format: `dots(desc, j, x, lane)` returns the gate and up dots of row `j`
/// for the `R` rows of `x`, and `store(e·R + t, g, u)` the value lane 0 writes to
/// `h[(e·R + t)·inter + j]`.
#[inline(always)]
pub(crate) unsafe fn gate_up<D: Copy, const R: usize>(
    r: &MoeRange<D>,
    dots: impl FnOnce(&D, usize, RowsView, usize) -> ([f32; R], [f32; R]),
    store: impl Fn(usize, f32, f32) -> f32,
) {
    let s = r.span;
    let Some((lane, e, j)) = wave_row(s, s.inter) else {
        return;
    };
    let x = RowsView {
        x: r.x,
        stride: s.hidden as usize,
    };
    // By value, for the reason `down` gives.
    let d = *r.descs.add(e);
    let (g, u) = dots(&d, j, x, lane);
    if lane == 0 {
        for t in 0..R {
            *r.h.add((e * R + t) * s.inter as usize + j) = store(e * R + t, g[t], u[t]);
        }
    }
}

/// `atomicAdd` on the fixed-point accumulator. HIP's is `atomicrmw add … syncscope("agent")
/// monotonic`; `core`'s `fetch_add` is the SYSTEM scope, and the two must lower to the same
/// `global_atomic_add_u64` — checked in the ISA, not assumed (rust-kernels.md §5, L2). The
/// value is an integer, so the sum is independent of the order the waves land in.
#[inline(always)]
unsafe fn acc_add(p: *mut u64, v: u64) {
    core::sync::atomic::AtomicU64::from_ptr(p).fetch_add(v, core::sync::atomic::Ordering::Relaxed);
}

/// Pass 2 over any format: `dot(desc, o, h, lane)` is output row `o`'s dot over this expert's
/// `R` adjacent `h` rows, and `contribution(e·R + t, dv)` the value lane 0 adds in fixed point
/// to `acc[t·hidden + o]`, or `None` to skip it.
#[inline(always)]
pub(crate) unsafe fn down<D: Copy, const R: usize>(
    r: &MoeRange<D>,
    dot: impl FnOnce(&D, usize, RowsView, usize) -> [f32; R],
    contribution: impl Fn(usize, f32) -> Option<f32>,
) {
    let s = r.span;
    let Some((lane, e, o)) = wave_row(s, s.hidden) else {
        return;
    };
    let inter = s.inter as usize;
    let he = RowsView {
        x: r.h.add(e * R * inter),
        stride: inter,
    };
    // By value, as the C's `const ExpertDesc d = descs[e]`: handed out as `&*descs.add(e)`
    // instead, rustc could not prove the pointer loads unclobbered and lowered every weight and
    // scale load through it to `flat_load` (measured on all six down kernels); copied, it
    // infers `global_load` — what the C's `gu8p` typing bought fp4 (rust-kernels.md §5, L2).
    let d = *r.descs.add(e);
    let dv = dot(&d, o, he, lane);
    if lane == 0 {
        for (t, &v) in dv.iter().enumerate() {
            if let Some(c) = contribution(e * R + t, v) {
                acc_add(r.acc.add(t * s.hidden as usize + o), moe_fixed(c));
            }
        }
    }
}

/// The GLM pair's pass-2 contribution: `w · dv`, skipped where `w == 0` (a row that did not
/// route here — the skip is correctness, not thrift; `moe.hip` argues it).
#[inline(always)]
pub(crate) unsafe fn routed(wexpert: *const f32) -> impl Fn(usize, f32) -> Option<f32> {
    move |k, dv| {
        let w = *wexpert.add(k);
        (w != 0.0).then(|| w * dv)
    }
}

/// One VQ row: `i_dim / VQ_DIM` 12-bit indices and `i_dim / VQ_GROUP` bf16 scales per row.
#[inline(always)]
unsafe fn vq_row(
    idx: *const u8,
    scale: *const u16,
    cb: *const u16,
    (row, i_dim): (usize, i32),
) -> VqRow {
    let rb = (i_dim / VQ_DIM as i32) as usize * VQ_INDEX_BITS / 8;
    let ng = i_dim as usize / VQ_GROUP;
    VqRow {
        idx: idx.add(row * rb),
        scale: scale.add(row * ng),
        cb: cb.cast(),
        i_dim: i_dim as usize,
    }
}

#[inline(always)]
unsafe fn gateup_vq<const R: usize>(a: &MoeVq) {
    let hidden = a.range.span.hidden;
    let dots = |d: &ExpertDesc, j, x, lane| {
        let g = vq_row(d.gate_indices, d.gate_scales, a.cb[0], (j, hidden));
        let u = vq_row(d.up_indices, d.up_scales, a.cb[1], (j, hidden));
        (
            dot_vq_wave_r::<R>(x, g, lane),
            dot_vq_wave_r::<R>(x, u, lane),
        )
    };
    gate_up(&a.range, dots, |_, g, u| silu(g) * u);
}

#[inline(always)]
unsafe fn down_vq<const R: usize>(a: &MoeVq) {
    let inter = a.range.span.inter;
    let dot = |d: &ExpertDesc, o, h, lane| {
        let w = vq_row(d.down_indices, d.down_scales, a.cb[2], (o, inter));
        dot_vq_wave_r::<R>(h, w, lane)
    };
    down(&a.range, dot, routed(a.range.wexpert));
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_gateup_vq(a: MoeVq) {
    gateup_vq::<1>(&a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_gateup_vq_r2(a: MoeVq) {
    gateup_vq::<2>(&a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_down_vq(a: MoeVq) {
    down_vq::<1>(&a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_down_vq_r2(a: MoeVq) {
    down_vq::<2>(&a);
}

/// One nibble-format row: `(i_dim + 1) / 2` packed bytes (fp4's dims are even, where this is
/// the C's `i_dim / 2`) and `ceil(i_dim / group)` scales of the format's type.
#[inline(always)]
pub(crate) unsafe fn nib_row<F: NibFormat>(
    w: *const u8,
    scale: *const F::Scale,
    (row, i_dim): (usize, i32),
) -> NibRow<F> {
    let group = 1i32 << F::GROUP_SHIFT;
    let rb = ((i_dim + 1) / 2) as usize;
    let ng = ((i_dim + group - 1) / group) as usize;
    NibRow {
        w: w.add(row * rb),
        scale: scale.add(row * ng),
        dim: i_dim as usize,
    }
}

/// An int4 projection of `d`: the descriptor's scale pointer is typed for VQ's bf16 and READ
/// as f32 here (one layout, the kernel picks the interpretation; `ExpertDesc`'s note).
#[inline(always)]
unsafe fn i4_row(w: *const u8, scale: *const u16, at: (usize, i32)) -> NibRow<I4> {
    nib_row::<I4>(w, scale.cast(), at)
}

#[inline(always)]
unsafe fn gateup_i4<const R: usize>(a: &MoeI4) {
    let hidden = a.span.hidden;
    let dots = |d: &ExpertDesc, j, x, lane| {
        let g = i4_row(d.gate_indices, d.gate_scales, (j, hidden));
        let u = i4_row(d.up_indices, d.up_scales, (j, hidden));
        (
            dot_nib_wave_r::<R, U, I4>(x, g, lane),
            dot_nib_wave_r::<R, U, I4>(x, u, lane),
        )
    };
    gate_up(a, dots, |_, g, u| silu(g) * u);
}

#[inline(always)]
unsafe fn down_i4<const R: usize>(a: &MoeI4) {
    let inter = a.span.inter;
    let dot = |d: &ExpertDesc, o, h, lane| {
        dot_nib_wave_r::<R, U, I4>(h, i4_row(d.down_indices, d.down_scales, (o, inter)), lane)
    };
    down(a, dot, routed(a.wexpert));
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_gateup_i4(a: MoeI4) {
    gateup_i4::<1>(&a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_gateup_i4_r2(a: MoeI4) {
    gateup_i4::<2>(&a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_down_i4(a: MoeI4) {
    down_i4::<1>(&a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_down_i4_r2(a: MoeI4) {
    down_i4::<2>(&a);
}

/// The drains' one body: element `o`'s `rows` accumulator words summed (as i64, wrapping —
/// the C's signed sum) and zeroed, converted through f64 (acc reaches 2^48; f32 would round
/// it away — `moe.hip`'s note), then `write(&mut out[o], f)`.
#[inline(always)]
unsafe fn drain(a: &AccDrainTo, write: impl FnOnce(*mut f32, f32)) {
    let o = bid().wrapping_mul(ELEMENTWISE_BLOCK).wrapping_add(tid()) as i32;
    if o >= a.n {
        return;
    }
    let (o, n) = (o as usize, a.n as usize);
    let mut t = 0i64;
    for r in 0..a.rows as usize {
        let p = a.acc.add(r * n + o);
        t = t.wrapping_add(*p as i64);
        *p = 0;
    }
    write(a.out.add(o), (t as f64 / f64::from(MOE_ACC_SCALE)) as f32);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_acc_drain(a: AccDrain) {
    drain(&a.to, |x, f| *x = fma(a.gain, f, *x));
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn moe_acc_drain_to(a: AccDrainTo) {
    drain(&a, |out, f| *out = f);
}
