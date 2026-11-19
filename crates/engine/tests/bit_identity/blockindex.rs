//! Rows for `kernels/blockindex.hip`'s two launchers at the shapes `v4/blocksel.rs` and
//! `v4/kvcompress.rs` launch: the rotated fp4 simulator over `m · 64` query-head rows of
//! `index_head_dim` 128 (decode m = 1, a prefill tile of 7) and over the compressor's emitted rows,
//! plus every power-of-two width the launcher admits; the bf16 block scorer at the fp32 twin's
//! shapes (`inputs::SCORE_BLOCKS_DIMS`). `act_quant_f4_rotated` is in place in production — the
//! only way it is launched — so its row is in place.
//!
//! The `.hip` file named here is deleted from the tree; cite it at tag `archive/hip-kernels`.

use super::device::{Cell, O, Out, Ref, S, Shapes, Side, Tally, both, code, guard_ptrs, sweep};
use super::indexer::score_blocks_row;
use super::inputs::draw;
use rivoli_backend::hip;
use std::ptr::null_mut;

type RotateFn = unsafe extern "C" fn(O, i32, i32, S) -> i32;

/// `(rows, d)`.
const ROTATE_SHAPES: Shapes<(usize, usize)> = Shapes {
    all: &[(64, 128), (1, 128), (448, 128), (3, 256), (2, 32), (5, 64)],
    hot: (64, 128),
};

fn rotate_on(c: RotateFn, side: Side, x: O, [rows, d]: [i64; 2]) -> i32 {
    // SAFETY: `x` of `rows · d`, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c(x, rows as i32, d as i32, null_mut()) },
        Side::Rust => code(unsafe {
            hip::launch_act_quant_f4_rotated(x, rows as usize, d as usize, null_mut())
        }),
    }
}

pub fn act_quant_f4_rotated(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_act_quant_f4_rotated`'s C signature.
    let c: RotateFn = unsafe { r.sym(c"rivoli_act_quant_f4_rotated") };
    // The edges reach the Hadamard (±100s, ±0, subnormals) and the scales span 2^-12..2^4, so
    // fp4 blocks land on both the amax floor and saturation.
    let make = |(rows, d): (usize, usize), seed: u64| {
        let x = draw(
            rows * d,
            (seed, 1),
            [1.0, 1e-3, 30.0][(seed % 3) as usize],
            true,
        );
        (vec![], vec![Out::InPlace(x)], ())
    };
    sweep(
        t,
        ROTATE_SHAPES,
        make,
        |side, x: &mut Cell, (rows, d), _| {
            rotate_on(c, side, x.o(side, 0), [rows as i64, d as i64])
        },
    );
    // Each extent, the tile cap, a non-power-of-two, a power of two below the fp4 block, then
    // one case failing every family (order).
    for (rows, d) in [
        (0, 128),
        (4, 0),
        (4, 512),
        (4, 96),
        (4, 16),
        (0, 96),
        (-1, 600),
    ] {
        let rc = both(|side| rotate_on(c, side, guard_ptrs().1, [rows, d]));
        t.guard(&format!("act_quant_f4_rotated rows={rows} d={d}"), rc);
    }
}

pub fn index_score_blocks(r: &Ref, t: &mut Tally) {
    score_blocks_row(
        r,
        t,
        c"rivoli_index_score_blocks",
        hip::launch_index_score_blocks,
    );
}
