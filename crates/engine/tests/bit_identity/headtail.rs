//! Rows for `kernels/headtail.hip`'s two launchers at V4's shapes: the 4096 hidden, `hc_mult`
//! 4, decode (`s = 1`) and a short prefill, plus widths that leave most of a block idle.

use super::device::{Cell, I, O, Out, Ref, S, Shapes, Side, Tally, code, guards, sweep};
use super::inputs::{draw, u16_words, uniform};
use rivoli_backend::hip;
use rivoli_core::num::f32_to_bf16;
use std::ptr::null_mut;

type EmbedFn = unsafe extern "C" fn(*const u16, i32, i32, i32, O, S) -> i32;
type CollapseFn = unsafe extern "C" fn(I, I, I, I, O, O, i32, i32, i32, f32, f32, S) -> i32;

/// `(vocab rows in the table, hidden, hc)`; the token is drawn per seed.
const BCAST_SWEEP: Shapes<(usize, usize, usize)> = Shapes {
    all: &[(5, 4096, 4), (3, 2048, 1), (2, 1, 1), (4, 257, 4)],
    hot: (5, 4096, 4),
};
/// `(s, hc, dim)`.
const COLLAPSE_SWEEP: Shapes<(usize, usize, usize)> = Shapes {
    all: &[
        (1, 4, 4096),
        (3, 4, 4096),
        (2, 4, 7),
        (1, 1, 33),
        (5, 3, 100),
    ],
    hot: (1, 4, 4096),
};
/// V4's `norm_eps` and `hc_eps`.
const EPS: (f32, f32) = (1e-6, 1e-6);

/// `rivoli_embed_bf16_row_bcast` on `side`: `dims` `(token, hidden, hc)`.
fn embed_on(c: EmbedFn, side: Side, (w, x): (*const u16, O), dims: [usize; 3]) -> i32 {
    let [tok, hid, hc] = dims;
    let [ti, di, ci] = dims.map(|v| v as i32);
    // SAFETY: live buffers sized for the table, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c(w, ti, di, ci, x, null_mut()) },
        Side::Rust => {
            code(unsafe { hip::launch_embed_bf16_row_bcast(w, tok, hid, hc, x, null_mut()) })
        }
    }
}

pub fn embed_bf16_row_bcast(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_embed_bf16_row_bcast`'s C signature.
    let c: EmbedFn = unsafe { r.sym(c"rivoli_embed_bf16_row_bcast") };
    // The table's bf16 rows from draws with the edges (bf16 ties, subnormals, ±0), plus a
    // bf16 NaN at slot 3. `bf16f` is a pure shift, so the output is compared as BITS — the
    // NaN's payload included, which the f32 NaN class would not see.
    let make = |(vocab, hid, hc): (usize, usize, usize), seed: u64| {
        let mut w: Vec<u16> = draw(vocab * hid, (seed, 1), 1.0, true)
            .into_iter()
            .map(f32_to_bf16)
            .collect();
        if let Some(s) = w.get_mut(3) {
            *s = 0x7fc1;
        }
        (
            vec![u16_words(w)],
            vec![Out::PoisonBits(hc * hid)],
            seed as usize % vocab,
        )
    };
    sweep(
        t,
        BCAST_SWEEP,
        make,
        |side, x: &mut Cell, (_, hid, hc), &tok| {
            embed_on(c, side, (x.i(0).cast(), x.o(side, 0)), [tok, hid, hc])
        },
    );
    guards(
        t,
        "embed_bf16_row_bcast",
        &[[0, 0, 4], [usize::MAX, 8, 4], [0, 8, 0], [usize::MAX, 0, 0]],
        |side, dims, (gi, go)| embed_on(c, side, (gi.cast(), go), dims),
    );
}

/// `[h, fnw, base, scale]` in, `[pre, y]` out.
type CollapseBufs = ([I; 4], [O; 2]);

/// `rivoli_hc_head_collapse` on `side`: `dims` `(s, hc, dim)`.
fn collapse_on(
    c: CollapseFn,
    side: Side,
    ([h, f, b, sc], [p, y]): CollapseBufs,
    dims: [usize; 3],
) -> i32 {
    let ([s, hc, dim], [si, hi, di]) = (dims, dims.map(|v| v as i32));
    let (eps, hc_eps) = EPS;
    // SAFETY: as `embed_on`.
    match side {
        Side::Hip => unsafe { c(h, f, b, sc, p, y, si, hi, di, eps, hc_eps, null_mut()) },
        Side::Rust => code(unsafe {
            hip::launch_hc_head_collapse(h, f, b, sc, p, y, s, hc, dim, eps, hc_eps, null_mut())
        }),
    }
}

pub fn hc_head_collapse(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_hc_head_collapse`'s C signature.
    let c: CollapseFn = unsafe { r.sym(c"rivoli_hc_head_collapse") };
    let make = |(s, hc, dim): (usize, usize, usize), seed: u64| {
        let ins = vec![
            draw(s * hc * dim, (seed, 1), 1.0, true),
            draw(hc * hc * dim, (seed, 2), 0.05, false),
            draw(hc, (seed, 3), 1.0, false),
            uniform(1, (seed, 4), 0.5..2.0),
        ];
        (ins, vec![Out::Poison(s * hc), Out::Poison(s * dim)], ())
    };
    sweep(t, COLLAPSE_SWEEP, make, |side, x: &mut Cell, shape, _| {
        let bufs = (
            [x.i(0), x.i(1), x.i(2), x.i(3)],
            [x.o(side, 0), x.o(side, 1)],
        );
        collapse_on(c, side, bufs, [shape.0, shape.1, shape.2])
    });
    guards(
        t,
        "hc_head_collapse",
        &[[0, 4, 8], [1, 0, 8], [1, 4, 0], [0, 0, 0]],
        |side, dims, (gi, go)| collapse_on(c, side, ([gi; 4], [go; 2]), dims),
    );
}
