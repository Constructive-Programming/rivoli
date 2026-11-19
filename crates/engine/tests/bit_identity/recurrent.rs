//! Rows for `kernels/recurrent.hip`'s four launchers: K3's KDA trio at its real 96 heads ×
//! 128 and 3 × 12288 conv channels, Qwen's gated delta rule at 16 QK / 48 V heads × 128, the
//! anchors' widths (32, 20), and degenerate and padded widths (1, 33, 96, 1024) that leave
//! lanes masked or a block of one. The recurrent state is IN PLACE on both sides, as the
//! engine launches it.

use super::device::{Cell, I, O, Out, Ref, S, Shapes, Side, Tally, both, code, guard_ptrs, sweep};
use super::inputs::{draw, nan_some, uniform};
use rivoli_backend::hip::{self, GdnBufs, HeadCount, HeadDim};
use std::ptr::null_mut;

type DeltaFn = unsafe extern "C" fn(I, I, I, I, I, I, I, i32, i32, f32, O, O, S) -> i32;
type ConvFn = unsafe extern "C" fn(I, I, i32, i32, O, O, S) -> i32;
type GateHeadsFn = unsafe extern "C" fn(I, I, I, i32, i32, f32, O, S) -> i32;
type GdnFn = unsafe extern "C" fn(I, I, I, I, I, I, I, i32, i32, i32, i32, O, O, S) -> i32;

/// `(heads, head_dim)`; K3's real (96, 128) is the hot shape.
const DELTA_SWEEP: Shapes<(usize, usize)> = Shapes {
    all: &[(96, 128), (4, 32), (3, 64), (1, 1), (2, 2), (1, 1024)],
    hot: (96, 128),
};
/// `(channels, taps)`; K3's 12288 channels × 4 taps is hot.
const CONV_SWEEP: Shapes<(usize, usize)> = Shapes {
    all: &[(12288, 4), (1, 2), (257, 3), (1000, 16), (33, 4)],
    hot: (12288, 4),
};
/// `(heads, head_dim)`: K3 (96, 128), Qwen's 48 × 128 and anchor 20, and padded widths.
const GATE_HEADS_SWEEP: Shapes<(usize, usize)> = Shapes {
    all: &[
        (96, 128),
        (48, 128),
        (9, 20),
        (2, 1),
        (1, 1024),
        (5, 96),
        (7, 33),
    ],
    hot: (96, 128),
};
/// `(qk_heads, v_heads, dk, dv)`: Qwen's real widths, its anchor, and asymmetric widths that
/// mask each axis on its own.
type GdnCell = (usize, usize, usize, usize);
const GDN_SWEEP: Shapes<GdnCell> = Shapes {
    all: &[
        (16, 48, 128, 128),
        (3, 9, 20, 20),
        (2, 4, 32, 20),
        (1, 3, 20, 64),
        (1, 1, 1, 1),
        (4, 4, 128, 96),
    ],
    hot: (16, 48, 128, 128),
};

/// `f(ins[0], …, ins[6], rest…)`: the seven read buffers both delta rules take first, spelled
/// once rather than in every C and Rust call.
macro_rules! with7 {
    ($f:expr, $i:expr; $($rest:expr),*) => {{
        let i = $i;
        $f(i[0], i[1], i[2], i[3], i[4], i[5], i[6], $($rest),*)
    }};
}

/// The seven read buffers in (`q, k, v`, then K3's `g, beta_pre` or Qwen's `a, b`, then
/// `a_log, dt_bias`), `state, out` out.
type DeltaBufs = ([I; 7], [O; 2]);

fn delta_on(
    c: DeltaFn,
    side: Side,
    (ins, [st, out]): DeltaBufs,
    (h, d, lb): (usize, usize, f32),
) -> i32 {
    // SAFETY: live buffers sized for `(h, d)`, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { with7!(c, ins; h as i32, d as i32, lb, st, out, null_mut()) },
        Side::Rust => code(unsafe {
            let (h, d) = (HeadCount(h), HeadDim(d));
            with7!(hip::launch_gated_delta_recurrent_f32, ins; h, d, lb, st, out, null_mut())
        }),
    }
}

pub fn gated_delta_recurrent_f32(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gated_delta_recurrent_f32`'s C signature.
    let c: DeltaFn = unsafe { r.sym(c"rivoli_gated_delta_recurrent_f32") };
    // `lower_bound` sits exactly on fla's inclusive end (-5, K3's shipped value) on even
    // seeds; seeded inside the range otherwise.
    let make = |(h, d): (usize, usize), seed: u64| {
        let n = h * d;
        let ins = vec![
            draw(n, (seed, 1), 1.0, true),
            draw(n, (seed, 2), 1.0, true),
            draw(n, (seed, 3), 1.0, true),
            nan_some(draw(n, (seed, 4), 8.0, true), seed),
            draw(h, (seed, 5), 4.0, false),
            uniform(h, (seed, 6), -3.0..2.0),
            draw(n, (seed, 7), 1.0, false),
        ];
        let outs = vec![
            Out::InPlace(draw(h * d * d, (seed, 8), 1.0, true)),
            Out::Poison(n),
        ];
        let lb = if seed.is_multiple_of(2) {
            -5.0
        } else {
            uniform(1, (seed, 9), -5.0..-0.01)[0]
        };
        (ins, outs, lb)
    };
    sweep(t, DELTA_SWEEP, make, |side, x: &mut Cell, (h, d), &lb| {
        let ins = std::array::from_fn(|i| x.i(i));
        delta_on(c, side, (ins, [x.o(side, 0), x.o(side, 1)]), (h, d, lb))
    });
    // Zero heads, zero width, a block past 1024, a non-power-of-two width, the bound's three
    // refusals (NaN, 0, below -5), then pairs that pin the ORDER of every adjacent two
    // (1002 before 1003, 1003 before 1006) and every guard at once.
    for args in [
        (0, 32, -5.0),
        (1, 0, -5.0),
        (1, 2048, -5.0),
        (1, 96, -5.0),
        (1, 32, f32::NAN),
        (1, 32, 0.0),
        (1, 32, -5.5),
        (1, 1536, f32::NAN),
        (1, 96, f32::NAN),
        (0, 96, f32::NAN),
    ] {
        let (gi, go) = guard_ptrs();
        let refusal = both(|side| delta_on(c, side, ([gi; 7], [go; 2]), args));
        t.guard(&format!("gated_delta {args:?}"), refusal);
    }
}

fn conv_on(
    c: ConvFn,
    side: Side,
    ([cur, w], [win, out]): ([I; 2], [O; 2]),
    (ch, taps): (usize, usize),
) -> i32 {
    // SAFETY: live buffers sized for `(ch, taps)`, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c(cur, w, ch as i32, taps as i32, win, out, null_mut()) },
        Side::Rust => {
            code(unsafe { hip::launch_short_conv_silu_f32(cur, w, ch, taps, win, out, null_mut()) })
        }
    }
}

pub fn short_conv_silu_f32(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_short_conv_silu_f32`'s C signature.
    let c: ConvFn = unsafe { r.sym(c"rivoli_short_conv_silu_f32") };
    let make = |(ch, taps): (usize, usize), seed: u64| {
        let ins = vec![
            nan_some(draw(ch, (seed, 1), 4.0, true), seed),
            draw(ch * taps, (seed, 2), 1.0, true),
        ];
        let outs = vec![
            Out::InPlace(draw(ch * taps, (seed, 3), 4.0, true)),
            Out::Poison(ch),
        ];
        (ins, outs, ())
    };
    sweep(t, CONV_SWEEP, make, |side, x: &mut Cell, shape, _| {
        conv_on(
            c,
            side,
            ([x.i(0), x.i(1)], [x.o(side, 0), x.o(side, 1)]),
            shape,
        )
    });
    for shape in [(0, 4), (8, 1), (8, 17), (0, 1)] {
        let (gi, go) = guard_ptrs();
        let refusal = both(|side| conv_on(c, side, ([gi; 2], [go; 2]), shape));
        t.guard(&format!("short_conv {shape:?}"), refusal);
    }
}

fn gate_heads_on(
    c: GateHeadsFn,
    side: Side,
    ([o, g, w], out): ([I; 3], O),
    (h, d, eps): (usize, usize, f32),
) -> i32 {
    // SAFETY: live buffers sized for `(h, d)`, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c(o, g, w, h as i32, d as i32, eps, out, null_mut()) },
        Side::Rust => code(unsafe {
            hip::launch_rmsnorm_gate_heads_f32(
                o,
                g,
                w,
                HeadCount(h),
                HeadDim(d),
                eps,
                out,
                null_mut(),
            )
        }),
    }
}

pub fn rmsnorm_gate_heads_f32(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_rmsnorm_gate_heads_f32`'s C signature.
    let c: GateHeadsFn = unsafe { r.sym(c"rivoli_rmsnorm_gate_heads_f32") };
    // The tree's two eps (1e-5, 1e-6) and the legal exact 0.
    let make = |(h, d): (usize, usize), seed: u64| {
        let ins = vec![
            nan_some(draw(h * d, (seed, 1), 1.0, true), seed),
            draw(h * d, (seed, 2), 8.0, true),
            draw(d, (seed, 3), 1.0, false),
        ];
        (
            ins,
            vec![Out::Poison(h * d)],
            [1e-5, 1e-6, 0.0][(seed % 3) as usize],
        )
    };
    sweep(
        t,
        GATE_HEADS_SWEEP,
        make,
        |side, x: &mut Cell, (h, d), &eps| {
            gate_heads_on(
                c,
                side,
                ([x.i(0), x.i(1), x.i(2)], x.o(side, 0)),
                (h, d, eps),
            )
        },
    );
    for args in [
        (0, 8, 1e-5),
        (1, 0, 1e-5),
        (1, 1025, 1e-5),
        (1, 8, f32::NAN),
        (1, 8, -1.0),
        (1, 8, f32::INFINITY),
        (1, 1025, f32::NAN),
        (0, 1025, f32::NAN),
    ] {
        let (gi, go) = guard_ptrs();
        let refusal = both(|side| gate_heads_on(c, side, ([gi; 3], go), args));
        t.guard(&format!("gate_heads {args:?}"), refusal);
    }
}

fn gdn_on(
    c: GdnFn,
    side: Side,
    (ins, [st, out]): ([I; 7], [O; 2]),
    (qk, vh, dk, dv): GdnCell,
) -> i32 {
    // SAFETY: live buffers sized for the shape, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe {
            let dims = [qk, vh, dk, dv].map(|x| x as i32);
            with7!(c, ins; dims[0], dims[1], dims[2], dims[3], st, out, null_mut())
        },
        Side::Rust => {
            let [q, k, v, a, b, a_log, dt_bias] = ins;
            let bufs = GdnBufs {
                q,
                k,
                v,
                a,
                b,
                a_log,
                dt_bias,
            };
            let (qk, vh, dk, dv) = (HeadCount(qk), HeadCount(vh), HeadDim(dk), HeadDim(dv));
            code(unsafe {
                hip::launch_gdn_recurrent_f32(bufs, qk, vh, dk, dv, st, out, null_mut())
            })
        }
    }
}

pub fn gdn_recurrent_f32(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gdn_recurrent_f32`'s C signature.
    let c: GdnFn = unsafe { r.sym(c"rivoli_gdn_recurrent_f32") };
    // `a` at scale 32 so `a + dt_bias` crosses softplus's threshold of 20 on some heads.
    let make = |(qk, vh, dk, dv): GdnCell, seed: u64| {
        let ins = vec![
            draw(qk * dk, (seed, 1), 1.0, true),
            draw(qk * dk, (seed, 2), 1.0, true),
            nan_some(draw(vh * dv, (seed, 3), 1.0, true), seed),
            uniform(vh, (seed, 4), -32.0..32.0),
            draw(vh, (seed, 5), 4.0, false),
            uniform(vh, (seed, 6), -3.0..2.0),
            uniform(vh, (seed, 7), -2.0..2.0),
        ];
        let outs = vec![
            Out::InPlace(draw(vh * dk * dv, (seed, 8), 1.0, true)),
            Out::Poison(vh * dv),
        ];
        (ins, outs, ())
    };
    sweep(t, GDN_SWEEP, make, |side, x: &mut Cell, shape, _| {
        let ins = std::array::from_fn(|i| x.i(i));
        gdn_on(c, side, (ins, [x.o(side, 0), x.o(side, 1)]), shape)
    });
    for shape in [
        (0, 3, 20, 20),
        (1, 3, 0, 20),
        (1, 3, 1025, 20),
        (1, 3, 20, 2000),
        (3, 4, 20, 20),
        (3, 4, 1025, 20),
        (0, 4, 1025, 0),
    ] {
        let (gi, go) = guard_ptrs();
        let refusal = both(|side| gdn_on(c, side, ([gi; 7], [go; 2]), shape));
        t.guard(&format!("gdn {shape:?}"), refusal);
    }
}
