//! Rows for `kernels/residual.hip`'s four launchers: V4's mHC pair at its real 4096 hidden,
//! K3's `attn_res` at 7168 with its nine-slot arena stride, Qwen's gated collapse at 2560 —
//! plus the degenerate widths (7, 1) that leave most of a block idle.
//!
//! Each launcher has ONE adapter that calls both twins with one argument set — value cells
//! pass live buffers, guard cases pass nulls (refused before any launch) — so each argument
//! list is spelled once per twin, side by side, and nowhere else.

use super::device::{Cell, I, O, Out, Ref, S, Shapes, Side, Tally, both, code, guard_ptrs, sweep};
use super::inputs::{draw, uniform};
use rivoli_backend::hip;
use std::ptr::null_mut;

type HcPreFn = unsafe extern "C" fn(I, I, I, I, i32, i32, i32, i32, f32, f32, O, O, O, S) -> i32;
type HcPostFn = unsafe extern "C" fn(I, I, I, I, i32, i32, i32, O, S) -> i32;
type AttnResFn = unsafe extern "C" fn(I, I, i32, i32, i32, usize, f32, O) -> i32;
type GatedFn = unsafe extern "C" fn(I, I, I, i32, i32, O, O, S) -> i32;

/// V4's `HC_MULT`; the launchers refuse anything else.
const HC: usize = 4;
const HC_MIX: usize = (2 + HC) * HC;
/// Both mHC eps values (V4 ships 1e-6 for each).
const EPS: f32 = 1e-6;

/// Each row's shapes and the hot one it is timed at.
const HC_PRE_SHAPES: Shapes<(usize, usize)> = Shapes {
    all: &[(1, 4096), (2, 256), (1, 7)],
    hot: (1, 4096),
};
const HC_POST_SHAPES: Shapes<(usize, usize)> = Shapes {
    all: &[(1, 4096), (3, 4096), (2, 7)],
    hot: (1, 4096),
};
const ATTN_RES_SWEEP: Shapes<ArShape> = Shapes {
    all: &ATTN_RES_SHAPES,
    hot: (1, 9, 7168, 9),
};
const GATED_SHAPES: Shapes<(usize, usize)> = Shapes {
    all: &[(4, 2560), (1, 300), (64, 257), (4, 1)],
    hot: (4, 2560),
};

/// The C launchers take `int` extents; the twins get the same values.
fn c_ints<const N: usize>(d: [usize; N]) -> [i32; N] {
    d.map(|v| v as i32)
}

/// `[h, fnw, scale, base]` in, `[y, post, comb]` out.
type HcBufs = ([I; 4], [O; 3]);

/// `rivoli_hc_pre` on `side`: `dims` is `(s, hc, dim, iters)`.
fn hc_pre_on(c: HcPreFn, side: Side, ([h, f, sc, b], [y, p, cb]): HcBufs, dims: [usize; 4]) -> i32 {
    let ([s, hc, dim, it], [si, hi, di, ii]) = (dims, c_ints(dims));
    // SAFETY: live buffers sized for `dims`, or nulls under a refused guard.
    match side {
        Side::Hip => unsafe { c(h, f, sc, b, si, hi, di, ii, EPS, EPS, y, p, cb, null_mut()) },
        Side::Rust => code(unsafe {
            hip::launch_hc_pre(h, f, sc, b, s, hc, dim, it, EPS, EPS, y, p, cb, null_mut())
        }),
    }
}

pub fn hc_pre(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_hc_pre`'s C signature.
    let c: HcPreFn = unsafe { r.sym(c"rivoli_hc_pre") };
    // Sinkhorn at one pass (the softmax only), two, and V4's twenty.
    let make = |(s, dim): (usize, usize), seed: u64| {
        let ins = vec![
            draw(s * HC * dim, (seed, 1), 1.0, true),
            draw(HC_MIX * HC * dim, (seed, 2), 0.05, false),
            uniform(3, (seed, 3), 0.5..2.0),
            draw(HC_MIX, (seed, 4), 1.0, false),
        ];
        let outs = vec![
            Out::Poison(s * dim),
            Out::Poison(s * HC),
            Out::Poison(s * HC * HC),
        ];
        (ins, outs, [1, 2, 20][(seed % 3) as usize])
    };
    sweep(
        t,
        HC_PRE_SHAPES,
        make,
        |side, x: &mut Cell, (s, dim), &it| {
            let bufs = (
                [x.i(0), x.i(1), x.i(2), x.i(3)],
                [x.o(side, 0), x.o(side, 1), x.o(side, 2)],
            );
            hc_pre_on(c, side, bufs, [s, HC, dim, it])
        },
    );
    let (gi, go) = guard_ptrs();
    let none: HcBufs = ([gi; 4], [go; 3]);
    for dims in [
        [0, 4, 8, 1],
        [1, 4, 0, 1],
        [1, 3, 8, 1],
        [1, 4, 8, 0],
        [1, 3, 8, 0],
        [0, 3, 8, 0],
    ] {
        t.guard(
            &format!("hc_pre {dims:?}"),
            both(|side| hc_pre_on(c, side, none, dims)),
        );
    }
}

/// `rivoli_hc_post` on `side`: `[x, residual, post, comb]` into `y`, `dims` `(s, hc, dim)`.
fn hc_post_on(c: HcPostFn, side: Side, ([x, res, p, cb], y): ([I; 4], O), dims: [usize; 3]) -> i32 {
    let ([s, hc, dim], [si, hi, di]) = (dims, c_ints(dims));
    // SAFETY: live buffers sized for `dims`, or nulls under a refused guard.
    match side {
        Side::Hip => unsafe { c(x, res, p, cb, si, hi, di, y, null_mut()) },
        Side::Rust => {
            code(unsafe { hip::launch_hc_post(x, res, p, cb, s, hc, dim, y, null_mut()) })
        }
    }
}

pub fn hc_post(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_hc_post`'s C signature.
    let c: HcPostFn = unsafe { r.sym(c"rivoli_hc_post") };
    let make = |(s, dim): (usize, usize), seed: u64| {
        let ins = vec![
            draw(s * dim, (seed, 1), 1.0, true),
            draw(s * HC * dim, (seed, 2), 1.0, true),
            uniform(s * HC, (seed, 3), 0.0..2.0),
            uniform(s * HC * HC, (seed, 4), 0.0..1.0),
        ];
        (ins, vec![Out::Poison(s * HC * dim)], ())
    };
    sweep(
        t,
        HC_POST_SHAPES,
        make,
        |side, x: &mut Cell, (s, dim), _| {
            let ins = [x.i(0), x.i(1), x.i(2), x.i(3)];
            hc_post_on(c, side, (ins, x.o(side, 0)), [s, HC, dim])
        },
    );
    for dims in [[0, 4, 8], [1, 4, 0], [1, 5, 8], [0, 5, 0]] {
        let refusal = both(|side| hc_post_on(c, side, ([guard_ptrs().0; 4], guard_ptrs().1), dims));
        t.guard(&format!("hc_post {dims:?}"), refusal);
    }
}

/// `(tokens, nsrc, n, slots)`: `slots · n` is the token stride — K3's arena keeps NINE slots
/// per token whatever the depth, so a stride longer than the live stack is the real case.
type ArShape = (usize, usize, usize, usize);
const ATTN_RES_SHAPES: [ArShape; 4] = [
    (1, 1, 7168, 1),
    (1, 9, 7168, 9),
    (2, 5, 100, 9),
    (1, 16, 256, 16),
];

/// `rivoli_attn_res` on `side`: `[src, fold]` into `out`.
fn attn_res_on(c: AttnResFn, side: Side, ([src, fold], out): ([I; 2], O), shape: ArShape) -> i32 {
    let (tok, nsrc, n, slots) = shape;
    let stride = slots * n;
    // SAFETY: live buffers sized for the shape, or nulls under a refused guard.
    match side {
        Side::Hip => unsafe {
            c(
                src,
                fold,
                tok as i32,
                nsrc as i32,
                n as i32,
                stride,
                EPS,
                out,
            )
        },
        Side::Rust => {
            code(unsafe { hip::launch_attn_res(src, fold, tok, nsrc, n, stride, EPS, out) })
        }
    }
}

pub fn attn_res(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_attn_res`'s C signature.
    let c: AttnResFn = unsafe { r.sym(c"rivoli_attn_res") };
    let make = |(tok, _, n, slots): ArShape, seed: u64| {
        let ins = vec![
            draw(tok * slots * n, (seed, 1), 1.0, true),
            draw(n, (seed, 2), 0.1, false),
        ];
        (ins, vec![Out::Poison(tok * n)], ())
    };
    sweep(t, ATTN_RES_SWEEP, make, |side, x: &mut Cell, shape, _| {
        attn_res_on(c, side, ([x.i(0), x.i(1)], x.o(side, 0)), shape)
    });
    // Zero tokens, zero width, an empty and an over-deep stack, a stride one short of the
    // live stack (slots < nsrc), then two that fail several guards at once (order).
    for shape in [
        (0, 1, 8, 1),
        (1, 1, 0, 1),
        (1, 0, 8, 1),
        (1, 17, 8, 17),
        (1, 2, 8, 1),
        (1, 17, 8, 1),
        (0, 17, 0, 1),
    ] {
        let refusal =
            both(|side| attn_res_on(c, side, ([guard_ptrs().0; 2], guard_ptrs().1), shape));
        t.guard(&format!("attn_res {shape:?}"), refusal);
    }
}

/// `rivoli_gated_residual_collapse_f32` on `side`: `[rhat, mix, inject]` into `[x, scale]`.
fn gated_on(c: GatedFn, side: Side, bufs: ([I; 3], [O; 2]), (n_r, dim): (usize, usize)) -> i32 {
    let ([rh, mx, inj], [x, sc]) = bufs;
    // SAFETY: live buffers sized for `(n_r, dim)`, or nulls under a refused guard.
    match side {
        Side::Hip => unsafe { c(rh, mx, inj, n_r as i32, dim as i32, x, sc, null_mut()) },
        Side::Rust => code(unsafe {
            hip::launch_gated_residual_collapse_f32(rh, mx, inj, n_r, dim, x, sc, null_mut())
        }),
    }
}

pub fn gated_residual_collapse_f32(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gated_residual_collapse_f32`'s C signature.
    let c: GatedFn = unsafe { r.sym(c"rivoli_gated_residual_collapse_f32") };
    let make = |(n_r, dim): (usize, usize), seed: u64| {
        let ins = vec![
            draw(n_r * dim, (seed, 1), 1.0, true),
            draw(n_r * dim, (seed, 2), 4.0, true),
            draw(n_r, (seed, 3), 4.0, false),
        ];
        (ins, vec![Out::Poison(dim), Out::Poison(n_r)], ())
    };
    sweep(t, GATED_SHAPES, make, |side, x: &mut Cell, shape, _| {
        let outs = [x.o(side, 0), x.o(side, 1)];
        gated_on(c, side, ([x.i(0), x.i(1), x.i(2)], outs), shape)
    });
    for shape in [(0, 8), (4, 0), (65, 8), (65, 0)] {
        let refusal =
            both(|side| gated_on(c, side, ([guard_ptrs().0; 3], [guard_ptrs().1; 2]), shape));
        t.guard(&format!("gated {shape:?}"), refusal);
    }
}
