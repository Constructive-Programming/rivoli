//! Rows for `kernels/linalg.hip`'s six GEMV launchers and the VQ encoder. A GEMV's row
//! arithmetic depends on `i_dim` (and fp8's `block`), never on `o_dim`, so the swept shapes
//! keep the engine's real `i_dim`s — GLM's 6144 hidden and 16384 o_proj, its 1536 LoRA rank,
//! V4's 7168 — at a few dozen rows, with odd `o_dim`s that leave a block's last waves idle;
//! the timed hot shape fills the machine. Ragged `i_dim`s reach every tail: fp8's per-column
//! path (`block < 4`, or `i_dim % 4` on the one aligned row), int8's scalar tail and its
//! misaligned-row fallback, int4's unaligned-row path.
//!
//! Weights are adversarial but not saturating: a random fp8 code is NaN 2 times in 256,
//! which would turn every 6144-long dot NaN and leave the NaN-class compare nothing to see,
//! so NaN codes and NaN scales are PLANTED on one seed in eight each instead.

use super::device::{Cell, I, O, Out, Ref, S, Shapes, Side, Tally, code, guards, sweep};
use super::inputs::{bytes, draw, u16_words, words};
use rivoli_backend::hip;
use rivoli_core::num::{f32_to_bf16, f32_to_f16};
use std::ptr::null_mut;

type B = *const u8;
type Fp8Fn = unsafe extern "C" fn(I, B, I, i32, i32, i32, i32, O) -> i32;
type I8Fn = unsafe extern "C" fn(I, B, I, i32, i32, i32, O) -> i32;
type I4Fn = unsafe extern "C" fn(I, B, I, i32, i32, O) -> i32;
type VqFn = unsafe extern "C" fn(I, B, *const u16, *const u16, i32, i32, O) -> i32;
type F32Fn = unsafe extern "C" fn(I, I, i32, i32, i32, O, S) -> i32;
type EncodeFn = unsafe extern "C" fn(I, I, I, i32, *mut u16) -> i32;

/// `(o_dim, i_dim, block, nrow)`.
type Fp8Shape = (usize, usize, usize, usize);
/// `(o_dim, i_dim, nrow)`.
type RowShape = (usize, usize, usize);

const FP8_ROWWISE: Shapes<Fp8Shape> = Shapes {
    all: &[
        (20, 6144, 128, 1),
        (20, 6144, 128, 2),
        (13, 1536, 128, 2),
        (24, 2048, 64, 1),
        (1, 1001, 128, 1),
        (17, 1000, 2, 2),
        (8, 2048, 1, 1),
        (2048, 6144, 128, 1),
    ],
    hot: (2048, 6144, 128, 1),
};
/// `i_dim >= 4096` routes to split-K (the launcher's threshold, hit exactly by 4096).
const FP8_SPLITK: Shapes<Fp8Shape> = Shapes {
    all: &[
        (9, 16384, 128, 2),
        (5, 4096, 128, 1),
        (3, 4098, 2, 2),
        (1024, 16384, 128, 1),
    ],
    hot: (1024, 16384, 128, 1),
};
const I8_SWEEP: Shapes<RowShape> = Shapes {
    all: &[
        (16, 6144, 2),
        (9, 1001, 1),
        (9, 1001, 2),
        (5, 130, 2),
        (3, 127, 1),
        (1, 1, 1),
        (2048, 6144, 1),
    ],
    hot: (2048, 6144, 1),
};
const I4_SWEEP: Shapes<RowShape> = Shapes {
    all: &[
        (2048, 6144, 1),
        (13, 6144, 1),
        (11, 1001, 1),
        (16, 2048, 1),
        (8, 130, 1),
        (3, 7, 1),
    ],
    hot: (2048, 6144, 1),
};
const VQ_SWEEP: Shapes<(usize, usize)> = Shapes {
    all: &[(9, 2048), (5, 64), (1, 128), (2048, 6144)],
    hot: (2048, 6144),
};
const F32_SWEEP: Shapes<RowShape> = Shapes {
    all: &[
        (160, 6144, 1),
        (256, 7168, 1),
        (256, 7168, 2),
        (9, 33, 2),
        (3, 1, 1),
    ],
    hot: (256, 7168, 1),
};
/// Subvector counts; even, because `idx` is u16 and the compare is by word.
const ENCODE_SWEEP: Shapes<usize> = Shapes {
    all: &[2, 256, 1000, 8192],
    hot: 8192,
};

/// The codebook the VQ kernels index (`quant.rs`'s `VQ_K x VQ_DIM`).
const VQ_K: usize = 4096;
const VQ_DIM: usize = 4;

/// `x`'s `nrow·i_dim` activations, edges first.
fn x_rows(nrow: usize, i_dim: usize, seed: u64) -> Vec<f32> {
    draw(nrow * i_dim, (seed, 1), 1.0, true)
}

/// `n` f32 scales, one NaN planted at slot 0 on one seed in eight.
fn scales(n: usize, seed: u64) -> Vec<f32> {
    let mut s = draw(n, (seed, 3), 1.0, false);
    if seed % 8 == 3 {
        s[0] = f32::NAN;
    }
    s
}

/// `n` fp8 codes with the two NaN codes (`0x7f`, `0xff`) moved to their finite neighbours,
/// then ONE planted at a seed-chosen column of row 0 on one seed in eight.
fn fp8_codes(n: usize, seed: u64) -> Vec<u8> {
    let mut b = bytes(n, (seed, 2));
    b.iter_mut()
        .filter(|c| **c & 0x7f == 0x7f)
        .for_each(|c| *c -= 1);
    if seed % 8 == 7 {
        b[seed as usize % n] = 0x7f;
    }
    b
}

/// The C launchers take `int` extents.
fn c_ints<const N: usize>(d: [usize; N]) -> [i32; N] {
    d.map(|v| v as i32)
}

/// `rivoli_gemv_fp8` on `side`: `(x, packed, scale)` into `y`.
fn fp8_on(c: Fp8Fn, side: Side, (x, p, s, y): (I, B, I, O), dims: [usize; 4]) -> i32 {
    let ([o, i, b, r], [oi, ii, bi, ri]) = (dims, c_ints(dims));
    // SAFETY: live buffers sized for `dims`, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c(x, p, s, oi, ii, bi, ri, y) },
        Side::Rust => code(unsafe { hip::launch_gemv_fp8(x, p, s, o, i, b, r, y) }),
    }
}

pub fn gemv_fp8(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gemv_fp8`'s C signature.
    let c: Fp8Fn = unsafe { r.sym(c"rivoli_gemv_fp8") };
    let make = |(o, i, b, n): Fp8Shape, seed: u64| {
        let tiles = o.div_ceil(b) * i.div_ceil(b);
        let ins = vec![
            x_rows(n, i, seed),
            words(&fp8_codes(o * i, seed)),
            scales(tiles, seed),
        ];
        (ins, vec![Out::Poison(n * o)], ())
    };
    for shapes in [FP8_ROWWISE, FP8_SPLITK] {
        sweep(t, shapes, make, |side, x: &mut Cell, (o, i, b, n), _| {
            let io = (x.i(0), x.i(1).cast(), x.i(2), x.o(side, 0));
            fp8_on(c, side, io, [o, i, b, n])
        });
    }
    // Zero extents and tile, a non-power-of-two tile (1003), nrow 3 (1004) on both routes,
    // then several at once (order), and a NEGATIVE tile through the usize wrapper.
    guards(
        t,
        "gemv_fp8",
        &[
            [0, 8, 128, 1],
            [8, 0, 128, 1],
            [8, 8, 0, 1],
            [8, 8, 3, 1],
            [8, 8, 128, 3],
            [8, 4096, 128, 3],
            [8, 8, 3, 3],
            [0, 8, 3, 3],
            [8, 8, usize::MAX, 1],
        ],
        |side, dims, (gi, go)| fp8_on(c, side, (gi, gi.cast(), gi, go), dims),
    );
}

/// The two scaled-code GEMVs' sweep: `x` rows, `layout(o, i)` = (code bytes, f32 scales)
/// of packed weights, into `y`; `launch(side, (x, packed, scale, y), [o, i, nrow])`.
fn scaled_rows(
    t: &mut Tally,
    shapes: Shapes<RowShape>,
    layout: fn(usize, usize) -> (usize, usize),
    launch: impl Fn(Side, (I, B, I, O), [usize; 3]) -> i32,
) {
    let make = |(o, i, n): RowShape, seed: u64| {
        let (codes, n_scales) = layout(o, i);
        let ins = vec![
            x_rows(n, i, seed),
            words(&bytes(codes, (seed, 2))),
            scales(n_scales, seed),
        ];
        (ins, vec![Out::Poison(n * o)], ())
    };
    sweep(t, shapes, make, |side, x: &mut Cell, (o, i, n), _| {
        launch(
            side,
            (x.i(0), x.i(1).cast(), x.i(2), x.o(side, 0)),
            [o, i, n],
        )
    });
}

/// `rivoli_gemv_i8` on `side`.
fn i8_on(c: I8Fn, side: Side, (x, p, s, y): (I, B, I, O), dims: [usize; 3]) -> i32 {
    let ([o, i, r], [oi, ii, ri]) = (dims, c_ints(dims));
    // SAFETY: as `fp8_on`.
    match side {
        Side::Hip => unsafe { c(x, p, s, oi, ii, ri, y) },
        Side::Rust => code(unsafe { hip::launch_gemv_i8(x, p, s, o, i, r, y) }),
    }
}

pub fn gemv_i8(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gemv_i8`'s C signature.
    let c: I8Fn = unsafe { r.sym(c"rivoli_gemv_i8") };
    // One int8 per weight, one scale per row.
    scaled_rows(
        t,
        I8_SWEEP,
        |o, i| (o * i, o),
        |side, io, dims| i8_on(c, side, io, dims),
    );
    let cases = [[0, 8, 1], [8, 0, 1], [8, 8, 3], [0, 8, 3]];
    guards(t, "gemv_i8", &cases, |side, dims, (gi, go)| {
        i8_on(c, side, (gi, gi.cast(), gi, go), dims)
    });
}

/// `rivoli_gemv_i4` on `side` (single-row: `dims[2]` is always 1).
fn i4_on(c: I4Fn, side: Side, (x, p, s, y): (I, B, I, O), dims: [usize; 3]) -> i32 {
    let ([o, i, _], [oi, ii, _]) = (dims, c_ints(dims));
    // SAFETY: as `fp8_on`.
    match side {
        Side::Hip => unsafe { c(x, p, s, oi, ii, y) },
        Side::Rust => code(unsafe { hip::launch_gemv_i4(x, p, s, o, i, y) }),
    }
}

pub fn gemv_i4(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gemv_i4`'s C signature.
    let c: I4Fn = unsafe { r.sym(c"rivoli_gemv_i4") };
    // Two nibbles a byte (rows `ceil(i/2)` bytes, so odd `i` misaligns odd rows), one scale
    // per 128-column group.
    let layout = |o: usize, i: usize| (o * i.div_ceil(2), o * i.div_ceil(128));
    scaled_rows(t, I4_SWEEP, layout, |side, io, dims| {
        i4_on(c, side, io, dims)
    });
    guards(
        t,
        "gemv_i4",
        &[[0, 8, 1], [8, 0, 1]],
        |side, dims, (gi, go)| i4_on(c, side, (gi, gi.cast(), gi, go), dims),
    );
}

/// `(x, indices, scales, codebook, y)`.
type VqIo = (I, B, *const u16, *const u16, O);

/// `rivoli_gemv_vq` on `side`.
fn vq_on(c: VqFn, side: Side, (x, idx, s, cb, y): VqIo, dims: [usize; 2]) -> i32 {
    let ([o, i], [oi, ii]) = (dims, c_ints(dims));
    // SAFETY: as `fp8_on`.
    match side {
        Side::Hip => unsafe { c(x, idx, s, cb, oi, ii, y) },
        Side::Rust => code(unsafe { hip::launch_gemv_vq(x, idx, s, cb, o, i, y) }),
    }
}

pub fn gemv_vq(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gemv_vq`'s C signature.
    let c: VqFn = unsafe { r.sym(c"rivoli_gemv_vq") };
    let make = |(o, i): (usize, usize), seed: u64| {
        let cb = draw(VQ_K * VQ_DIM, (seed, 4), 1.0, false);
        let ins = vec![
            x_rows(1, i, seed),
            words(&bytes(o * i * 3 / 8, (seed, 2))),
            u16_words(scales(o * i / 64, seed).into_iter().map(f32_to_bf16)),
            u16_words(cb.into_iter().map(f32_to_f16)),
        ];
        (ins, vec![Out::Poison(o)], ())
    };
    sweep(t, VQ_SWEEP, make, |side, x: &mut Cell, (o, i), _| {
        let io = (
            x.i(0),
            x.i(1).cast(),
            x.i(2).cast(),
            x.i(3).cast(),
            x.o(side, 0),
        );
        vq_on(c, side, io, [o, i])
    });
    guards(t, "gemv_vq", &[[0, 64], [8, 0]], |side, dims, (gi, go)| {
        vq_on(c, side, (gi, gi.cast(), gi.cast(), gi.cast(), go), dims)
    });
}

/// `rivoli_gemv_f32` on `side`.
fn f32_on(c: F32Fn, side: Side, (x, w, y): (I, I, O), dims: [usize; 3]) -> i32 {
    let ([o, i, r], [oi, ii, ri]) = (dims, c_ints(dims));
    // SAFETY: as `fp8_on`.
    match side {
        Side::Hip => unsafe { c(x, w, oi, ii, ri, y, null_mut()) },
        Side::Rust => code(unsafe { hip::launch_gemv_f32(x, w, o, i, r, y, null_mut()) }),
    }
}

pub fn gemv_f32(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gemv_f32`'s C signature.
    let c: F32Fn = unsafe { r.sym(c"rivoli_gemv_f32") };
    let make = |(o, i, n): RowShape, seed: u64| {
        let ins = vec![x_rows(n, i, seed), draw(o * i, (seed, 2), 0.1, true)];
        (ins, vec![Out::Poison(n * o)], ())
    };
    sweep(t, F32_SWEEP, make, |side, x: &mut Cell, (o, i, n), _| {
        f32_on(c, side, (x.i(0), x.i(1), x.o(side, 0)), [o, i, n])
    });
    guards(
        t,
        "gemv_f32",
        &[[0, 8, 1], [8, 0, 1], [8, 8, 0], [0, 8, 3]],
        |side, dims, (gi, go)| f32_on(c, side, (gi, gi, go), dims),
    );
}

/// `rivoli_vq_encode` on `side`: `[sub, codebook, cbnorm]` into `idx`.
fn encode_on(c: EncodeFn, side: Side, ([s, cb, nrm], idx): ([I; 3], *mut u16), n: usize) -> i32 {
    // SAFETY: as `fp8_on`.
    match side {
        Side::Hip => unsafe { c(s, cb, nrm, n as i32, idx) },
        Side::Rust => code(unsafe { hip::launch_vq_encode(s, cb, nrm, n, idx) }),
    }
}

pub fn vq_encode(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_vq_encode`'s C signature.
    let c: EncodeFn = unsafe { r.sym(c"rivoli_vq_encode") };
    let make = |n: usize, seed: u64| {
        let mut cb = draw(VQ_K * VQ_DIM, (seed, 2), 1.0, false);
        // A duplicate entry (100 copies 5) and a subvector sitting exactly on it: an exact
        // tie the argmin must break to the LOWER index.
        cb.copy_within(5 * VQ_DIM..6 * VQ_DIM, 100 * VQ_DIM);
        let norm: Vec<f32> = cb
            .chunks(VQ_DIM)
            .map(|c| c.iter().map(|v| v * v).sum())
            .collect();
        let mut sub = draw(n * VQ_DIM, (seed, 1), 1.0, false);
        sub[..VQ_DIM].copy_from_slice(&cb[5 * VQ_DIM..6 * VQ_DIM]);
        (vec![sub, cb, norm], vec![Out::PoisonBits(n / 2)], ())
    };
    sweep(t, ENCODE_SWEEP, make, |side, x: &mut Cell, n, _| {
        encode_on(c, side, ([x.i(0), x.i(1), x.i(2)], x.o(side, 0).cast()), n)
    });
    guards(t, "vq_encode n", &[0], |side, n, (gi, go)| {
        encode_on(c, side, ([gi; 3], go.cast()), n)
    });
}
