//! The gated-delta-rule family — the twins of `kernels/recurrent.hip` at tag
//! `archive/hip-kernels`: Kimi-K3's recurrence, short convolution and gated head norm, and
//! Qwen3.8-Flash-Next's gated delta rule. That file carries each operator's reference,
//! traps and measurements; this one carries what the port decided.
//!
//! - **No `blockDim`.** `gated_delta_recurrent_f32` takes its block as `head_dim`; the two
//!   masked kernels re-derive theirs with `rivoli_abi::geom::reduce_block`, the function their
//!   launchers launch with; `short_conv_silu_f32` strides by `ELEMENTWISE_BLOCK`.
//! - **Fused sites** (hipcc's `contract` pairs, every one a `v_fmac` in its ISA): each state
//!   loop's running sum `acc += x·sv` is `fma(x, sv, acc)`, the rank-one update
//!   `s + k·dv` is `fma(k, dv, s)`, and the conv's `acc += w·win`. Nothing else pairs a
//!   multiply with an add.
//! - **`#pragma unroll 4`** on the four state loops is `each_row`'s groups of four, loads before
//!   stores — schedule only, the fold stays ascending.

use crate::gpu::{bid, fma, lds, ocml, tid};
use crate::prelude::reduce::{block_sum_lds, sync_threads};
use crate::residual::sigmoid;
use core::intrinsics::{exp, sqrt};
use rivoli_abi::geom::{ELEMENTWISE_BLOCK, reduce_block};
use rivoli_abi::kernarg::{GateHeads, GatedDelta, Gdn, ShortConv};

/// The C's `#pragma unroll 4` on every state loop (`each_row` spells its four rows).
const STATE_UNROLL: usize = 4;
const _: () = assert!(STATE_UNROLL == 4, "each_row names exactly four rows");

/// `x · (1/sqrt(sum + 1e-6))` — both kernels' L2 norm, eps on the SUM.
#[inline(always)]
fn l2_scale(x: f32, sum: f32) -> f32 {
    x * (1.0 / sqrt(sum + 1e-6))
}

/// One thread's column of a head's `[rows][stride]` state: `s` already points at the column.
#[derive(Clone, Copy)]
struct Column {
    s: *mut f32,
    rows: usize,
    stride: usize,
}

/// `S[i] = step(i, S[i])` for `i` in `0..rows`, in groups of [`STATE_UNROLL`] whose state loads
/// are all issued before any of the group's stores, then a scalar remainder — the schedule
/// hipcc's `#pragma unroll 4` produces where it knows the stride non-zero (its `d` is
/// `blockDim.x`). Rust reads `d` from the kernel argument, could not prove the rows distinct,
/// and kept each load behind the previous row's store: one request in flight, measured 1.647x
/// hipcc on `gated_delta_recurrent_f32` (rust-kernels.md §5, Phase 1 L2). The values and their
/// order are unchanged: the rows ARE distinct (the stride — `head_dim`, or `dv` — is ≥ 1 by the
/// launchers' guard 1001), so a
/// load issued early reads what it would have read late.
#[inline(always)]
unsafe fn each_row(col: Column, mut step: impl FnMut(usize, f32) -> f32) {
    let Column { s, rows, stride } = col;
    let at = |i: usize| s.add(i * stride);
    let mut i = 0;
    // Four named loads, then four stores: an array of pointers here was measured to land in
    // scratch (`scratch_store_b128`), and the stores to go `flat`.
    while i + STATE_UNROLL <= rows {
        let (p0, p1, p2, p3) = (at(i), at(i + 1), at(i + 2), at(i + 3));
        let (v0, v1, v2, v3) = (*p0, *p1, *p2, *p3);
        *p0 = step(i, v0);
        *p1 = step(i + 1, v1);
        *p2 = step(i + 2, v2);
        *p3 = step(i + 3, v3);
        i += STATE_UNROLL;
    }
    while i < rows {
        *at(i) = step(i, *at(i));
        i += 1;
    }
}

/// Both passes over a [`Column`]: decay by `decay(i)` and read `u = Σ k[i]·S[i]`, then
/// `dv = delta(u)`, the rank-one update and `o = Σ q[i]·S[i]` of the UPDATED state. Returns `o`.
#[inline(always)]
unsafe fn state_passes(
    col: Column,
    (qn, kn): (*const f32, *const f32),
    decay: impl Fn(usize) -> f32,
    delta: impl FnOnce(f32) -> f32,
) -> f32 {
    let mut u = 0.0f32;
    each_row(col, |i, s| {
        let sv = decay(i) * s;
        u = fma(*kn.add(i), sv, u);
        sv
    });
    let dv = delta(u);
    let mut o = 0.0f32;
    each_row(col, |i, s| {
        let sv = fma(*kn.add(i), dv, s);
        o = fma(*qn.add(i), sv, o);
        sv
    });
    o
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gated_delta_recurrent_f32(a: GatedDelta) {
    let d = a.head_dim as usize;
    let (h, t) = (bid() as usize, tid() as usize);
    let sh = lds::<f32>();
    let (qn, kn, alpha, red) = (sh, sh.add(d), sh.add(2 * d), sh.add(3 * d));
    let off = h * d + t;
    let (qv, kv) = (*a.inp.q.add(off), *a.inp.k.add(off));
    let qsum = block_sum_lds(qv * qv, red, t, d);
    let ksum = block_sum_lds(kv * kv, red, t, d);
    *qn.add(t) = l2_scale(qv, qsum) * (1.0 / sqrt(d as f32));
    *kn.add(t) = l2_scale(kv, ksum);
    let z = exp(*a.inp.a_log.add(h)) * (*a.inp.decay.add(off) + *a.inp.dt_bias.add(off));
    *alpha.add(t) = exp(a.lower_bound * sigmoid(z));
    sync_threads();
    let beta = sigmoid(*a.inp.beta.add(h));
    let col = Column {
        s: a.state.add(h * d * d + t),
        rows: d,
        stride: d,
    };
    let o = state_passes(
        col,
        (qn, kn),
        |i| *alpha.add(i),
        |u| beta * (*a.inp.v.add(off) - u),
    );
    *a.out.add(off) = o;
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn short_conv_silu_f32(a: ShortConv) {
    // HIP's `blockIdx.x * blockDim.x + threadIdx.x`, unsigned and wrapping, read as `int`.
    let c = bid().wrapping_mul(ELEMENTWISE_BLOCK).wrapping_add(tid()) as i32;
    if c >= a.channels {
        return;
    }
    let (c, taps) = (c as usize, a.taps as usize);
    let (wc, wn) = (a.w.add(c * taps), a.win.add(c * taps));
    for j in 0..taps - 1 {
        *wn.add(j) = *wn.add(j + 1);
    }
    *wn.add(taps - 1) = *a.cur.add(c);
    let mut acc = 0.0f32;
    for j in 0..taps {
        acc = fma(*wc.add(j), *wn.add(j), acc);
    }
    *a.out.add(c) = acc * sigmoid(acc);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rmsnorm_gate_heads_f32(a: GateHeads) {
    let t = tid() as usize;
    let live = (t as i32) < a.head_dim;
    let off = bid() as usize * a.head_dim as usize + t;
    // Masked on the VALUE and the load, never on the reduction (every thread barriers).
    let x = if live { *a.o.add(off) } else { 0.0 };
    let block = reduce_block(a.head_dim as u32) as usize;
    let mean = block_sum_lds(x * x, lds(), t, block) / a.head_dim as f32;
    let inv = 1.0 / sqrt(mean + a.eps);
    if !live {
        return;
    }
    *a.out.add(off) = x * inv * *a.weight.add(t) * sigmoid(*a.gate.add(off));
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_recurrent_f32(a: Gdn) {
    let (h, t) = (bid() as usize, tid() as usize);
    let (dk, dv) = (a.dk as usize, a.dv as usize);
    let sh = lds::<f32>();
    let (qn, kn, red) = (sh, sh.add(dk), sh.add(2 * dk));
    let block = reduce_block(a.dk.max(a.dv) as u32) as usize;
    let koff = (h as i32 / a.rep) as usize * dk + t;
    let live_k = t < dk;
    let qv = if live_k { *a.inp.q.add(koff) } else { 0.0 };
    let kv = if live_k { *a.inp.k.add(koff) } else { 0.0 };
    let qsum = block_sum_lds(qv * qv, red, t, block);
    let ksum = block_sum_lds(kv * kv, red, t, block);
    if live_k {
        *qn.add(t) = l2_scale(qv, qsum) * (1.0 / sqrt(dk as f32));
        *kn.add(t) = l2_scale(kv, ksum);
    }
    // torch's softplus branch: `x` above 20, else `log1p(exp(x))` (ocml's, as hipcc inlines).
    let z = *a.inp.decay.add(h) + *a.inp.dt_bias.add(h);
    let sp = if z > 20.0 { z } else { ocml::log1pf(exp(z)) };
    let alpha = exp(-exp(*a.inp.a_log.add(h)) * sp);
    let beta = sigmoid(*a.inp.beta.add(h));
    // The LAST barrier: past it the block diverges on `dv`.
    sync_threads();
    if t >= dv {
        return;
    }
    let col = Column {
        s: a.state.add(h * dk * dv + t),
        rows: dk,
        stride: dv,
    };
    let vv = *a.inp.v.add(h * dv + t);
    let o = state_passes(col, (qn, kn), |_| alpha, |u| beta * (vv - u));
    *a.out.add(h * dv + t) = o;
}
