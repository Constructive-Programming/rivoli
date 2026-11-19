//! Rows for `kernels/fwd.hip`'s nine launchers — the forward pass's glue — at the engine's
//! shapes: GLM's 6144 hidden, 512-wide latent with a 64-wide rope key, 64 heads of 192, its
//! 151552 vocabulary and Glimmer's 202048, plus ragged and degenerate extents.
//!
//! Outputs that are integers (fp8 codes, bf16 keys, the argmax index, the flag, the hash) are
//! `PoisonBits`/`InPlaceBits` (the latter from zero): compared bit for bit, never as a NaN class.

use super::device::{Cell, I, O, Out, Ref, S, Shapes, Side, Tally, code, guards, sweep};
use super::inputs::{bytes, draw, words};
use rivoli_backend::hip::{self, HashSpan};
use std::ptr::null_mut;

type EmbedFn = unsafe extern "C" fn(*const u8, I, i32, i32, O) -> i32;
type AppendFn = unsafe extern "C" fn(I, I, *mut u8, O, *mut u16, i32, i32, i32, i32) -> i32;
type GatherFn = unsafe extern "C" fn(I, O, i32, i32, i32, i32) -> i32;
type VaddFn = unsafe extern "C" fn(O, I, i32) -> i32;
type GateFn = unsafe extern "C" fn(O, I, i32, S) -> i32;
type SoftcapFn = unsafe extern "C" fn(O, i32, f32, f32, S) -> i32;
type ArgmaxFn = unsafe extern "C" fn(I, i32, *mut i32, O) -> i32;
type FlagFn = unsafe extern "C" fn(I, i32, u32, *mut u32) -> i32;
type HashFn = unsafe extern "C" fn(I, i32, i32, u64, *mut u64, S) -> i32;

/// The elementwise rows' extents: one lane, a partial wave, a partial block, GLM's hidden
/// and a ragged FFN-sized width.
const ELEMENTWISE_SWEEP: Shapes<usize> = Shapes {
    all: &[1, 33, 257, 6144, 18439],
    hot: 6144,
};
/// `(vocab rows in the table, hidden)`; the token is drawn per seed.
const EMBED_SWEEP: Shapes<(usize, usize)> = Shapes {
    all: &[(5, 6144), (4, 2048), (2, 257), (2, 1)],
    hot: (5, 6144),
};
/// `(kvl, ropn, pos)` — `pos` far past zero too, the slab offset the index math carries.
const APPEND_SWEEP: Shapes<(usize, usize, usize)> = Shapes {
    all: &[
        (512, 64, 0),
        (512, 64, 777),
        (128, 128, 5),
        (1024, 2, 3),
        (384, 64, 131_071),
    ],
    hot: (512, 64, 777),
};
/// `(h, qh, nope, ropn)`.
const GATHER_SWEEP: Shapes<(usize, usize, usize, usize)> = Shapes {
    all: &[
        (64, 192, 128, 64),
        (128, 192, 128, 64),
        (1, 64, 0, 64),
        (7, 33, 1, 31),
    ],
    hot: (64, 192, 128, 64),
};
/// Logit rows: up to Glimmer's 202048 vocabulary, the width the softcap runs at.
const LOGIT_SWEEP: Shapes<usize> = Shapes {
    all: &[1, 257, 6144, 202_048],
    hot: 202_048,
};
const ARGMAX_SWEEP: Shapes<usize> = Shapes {
    all: &[1, 255, 256, 257, 6144, 151_552, 202_048],
    hot: 151_552,
};
/// `(n, stride)`; `i_base` is drawn per seed. 300000 samples exceed the 1024-block cap.
const HASH_SWEEP: Shapes<(usize, usize)> = Shapes {
    all: &[
        (1, 1),
        (255, 1),
        (6144, 1),
        (6144, 32),
        (1000, 7),
        (300_000, 1),
    ],
    hot: (6144, 1),
};

/// `v` with the non-finite values planted at seed-dependent slots: NaN, +inf, -inf on one
/// seed in two, so the other half are all-finite cells.
fn with_nonfinite(mut v: Vec<f32>, seed: u64) -> Vec<f32> {
    if seed % 2 == 1 {
        let n = v.len();
        for (k, bad) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            v[(seed as usize * 7 + k * 13) % n] = bad;
        }
    }
    v
}

/// `rivoli_embed_i8_row` on `side`.
fn embed_on(
    c: EmbedFn,
    side: Side,
    (p, s, x): (*const u8, I, O),
    (tok, hid): (usize, usize),
) -> i32 {
    // SAFETY: live buffers sized for the table, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c(p, s, tok as i32, hid as i32, x) },
        Side::Rust => code(unsafe { hip::launch_embed_i8_row(p, s, tok, hid, x) }),
    }
}

pub fn embed_i8_row(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_embed_i8_row`'s C signature.
    let c: EmbedFn = unsafe { r.sym(c"rivoli_embed_i8_row") };
    let make = |(vocab, hid): (usize, usize), seed: u64| {
        let ins = vec![
            words(&bytes(vocab * hid, (seed, 1))),
            draw(vocab, (seed, 2), 1.0, true),
        ];
        (ins, vec![Out::Poison(hid)], seed as usize % vocab)
    };
    sweep(
        t,
        EMBED_SWEEP,
        make,
        |side, x: &mut Cell, (_, hid), &tok| {
            embed_on(c, side, (x.i(0).cast(), x.i(1), x.o(side, 0)), (tok, hid))
        },
    );
    // A NEGATIVE token reaches the C through the usize wrapper as `usize::MAX`.
    guards(
        t,
        "embed_i8_row",
        &[(0, 0), (usize::MAX, 8), (usize::MAX, 0)],
        |side, dims, (gi, go)| embed_on(c, side, (gi.cast(), gi, go), dims),
    );
}

/// `append_kv`'s three slabs: `lc8` (u8), `lscale` (f32), `rc` (u16).
type Slabs = (*mut u8, O, *mut u16);

/// `rivoli_append_kv` on `side`: `[latent, rope]` into the slabs, `dims` `(pos, kvl, ropn,
/// n_blocks)`.
fn append_on(
    c: AppendFn,
    side: Side,
    ([l, rp], (lc, ls, rc)): ([I; 2], Slabs),
    dims: [usize; 4],
) -> i32 {
    let [pos, kvl, ropn, nb] = dims;
    let [pi, ki, ri, ni] = dims.map(|v| v as i32);
    // SAFETY: as `embed_on`.
    match side {
        Side::Hip => unsafe { c(l, rp, lc, ls, rc, pi, ki, ri, ni) },
        Side::Rust => code(unsafe { hip::launch_append_kv(l, rp, lc, ls, rc, pos, kvl, ropn, nb) }),
    }
}

pub fn append_kv(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_append_kv`'s C signature.
    let c: AppendFn = unsafe { r.sym(c"rivoli_append_kv") };
    let make = |(kvl, ropn, _): (usize, usize, usize), seed: u64| {
        let mut latent = draw(kvl, (seed, 1), 1.0, true);
        // An all-zero group (amax 0, scale 1) on one seed in four, a NaN on another.
        match seed % 4 {
            1 => latent[kvl - 128..].fill(0.0),
            2 => latent[kvl / 2] = f32::NAN,
            _ => {}
        }
        let outs = vec![
            Out::PoisonBits(kvl / 4),
            Out::Poison(kvl / 128),
            Out::PoisonBits(ropn / 2),
        ];
        (vec![latent, draw(ropn, (seed, 2), 1.0, true)], outs, ())
    };
    // Each output buffer holds exactly row `pos`, so every element is written and a per-side
    // poison catches one that is not; the kernel is handed the slab base `pos` rows before
    // it (an address only — rows before `pos` are never touched).
    sweep(
        t,
        APPEND_SWEEP,
        make,
        |side, x: &mut Cell, (kvl, ropn, pos), _| {
            let slabs = (
                x.o(side, 0).cast::<u8>().wrapping_sub(pos * kvl),
                x.o(side, 1).wrapping_sub(pos * (kvl / 128)),
                x.o(side, 2).cast::<u16>().wrapping_sub(pos * ropn),
            );
            append_on(
                c,
                side,
                ([x.i(0), x.i(1)], slabs),
                [pos, kvl, ropn, kvl / 128],
            )
        },
    );
    guards(
        t,
        "append_kv",
        &[
            [usize::MAX, 512, 64, 4],
            [0, 0, 64, 4],
            [0, 512, 0, 4],
            [0, 512, 64, 0],
            [0, 100, 64, 1],
            [0, 2048, 64, 16],
            [0, 200, 64, 1],
            [0, 128, 129, 1],
            [usize::MAX, 100, 0, 0],
        ],
        |side, dims, (gi, go)| append_on(c, side, ([gi; 2], (go.cast(), go, go.cast())), dims),
    );
}

/// `rivoli_gather_rope` on `side`.
fn gather_on(c: GatherFn, side: Side, (q, out): (I, O), dims: [usize; 4]) -> i32 {
    let [h, qh, nope, ropn] = dims;
    let [hi, qi, ni, ri] = dims.map(|v| v as i32);
    // SAFETY: as `embed_on`.
    match side {
        Side::Hip => unsafe { c(q, out, hi, qi, ni, ri) },
        Side::Rust => code(unsafe { hip::launch_gather_rope(q, out, h, qh, nope, ropn) }),
    }
}

pub fn gather_rope(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gather_rope`'s C signature.
    let c: GatherFn = unsafe { r.sym(c"rivoli_gather_rope") };
    sweep(
        t,
        GATHER_SWEEP,
        |(h, qh, _, ropn): (usize, usize, usize, usize), seed: u64| {
            let q = draw(h * qh, (seed, 1), 1.0, true);
            (vec![q], vec![Out::Poison(h * ropn)], ())
        },
        |side, x: &mut Cell, (h, qh, nope, ropn), _| {
            gather_on(c, side, (x.i(0), x.o(side, 0)), [h, qh, nope, ropn])
        },
    );
    guards(
        t,
        "gather_rope",
        &[
            [0, 8, 0, 4],
            [2, 0, 0, 4],
            [2, 8, usize::MAX, 4],
            [2, 8, 0, 0],
            [0, 0, 0, 0],
        ],
        |side, dims, (gi, go)| gather_on(c, side, (gi, go), dims),
    );
}

/// An in-place elementwise row (`vadd`, `sigmoid_gate`): `x` starts as a draw on both sides,
/// `y` is `operand(n, seed)`; `launch(side, x, y, n)`.
fn in_place_sweep(
    t: &mut Tally,
    operand: impl Fn(usize, u64) -> Vec<f32>,
    launch: impl Fn(Side, O, I, usize) -> i32,
) {
    let make = |n: usize, seed: u64| {
        let x = draw(n, (seed, 1), 1.0, true);
        (vec![operand(n, seed)], vec![Out::InPlace(x)], ())
    };
    sweep(t, ELEMENTWISE_SWEEP, make, |side, x: &mut Cell, n, _| {
        launch(side, x.o(side, 0), x.i(0), n)
    });
}

pub fn vadd(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_vadd`'s C signature.
    let c: VaddFn = unsafe { r.sym(c"rivoli_vadd") };
    // SAFETY (both closures): live buffers of `n`, or the guard buffer under a refused guard.
    let on = |side, x, y, n: usize| match side {
        Side::Hip => unsafe { c(x, y, n as i32) },
        Side::Rust => code(unsafe { hip::launch_vadd(x, y, n) }),
    };
    in_place_sweep(
        t,
        |n, seed| with_nonfinite(draw(n, (seed, 2), 1.0, true), seed),
        on,
    );
    guards(t, "vadd n", &[0], |side, n, (gi, go)| on(side, go, gi, n));
}

pub fn sigmoid_gate(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_sigmoid_gate`'s C signature.
    let c: GateFn = unsafe { r.sym(c"rivoli_sigmoid_gate") };
    // SAFETY: as `vadd`.
    let on = |side, x, g, n: usize| match side {
        Side::Hip => unsafe { c(x, g, n as i32, null_mut()) },
        Side::Rust => code(unsafe { hip::launch_sigmoid_gate(x, g, n, null_mut()) }),
    };
    // Scale 4 with the edges: `exp` overflows and underflows inside the sweep.
    in_place_sweep(
        t,
        |n, seed| with_nonfinite(draw(n, (seed, 2), 4.0, true), seed),
        on,
    );
    guards(t, "sigmoid_gate n", &[0], |side, n, (gi, go)| {
        on(side, go, gi, n)
    });
}

/// `rivoli_logit_softcap` on `side`: `x` in place, `(n, mult, cap)`.
fn softcap_on(c: SoftcapFn, side: Side, x: O, (n, mult, cap): (usize, f32, f32)) -> i32 {
    // SAFETY: as `embed_on`.
    match side {
        Side::Hip => unsafe { c(x, n as i32, mult, cap, null_mut()) },
        Side::Rust => code(unsafe { hip::launch_logit_softcap(x, n, mult, cap, null_mut()) }),
    }
}

pub fn logit_softcap(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_logit_softcap`'s C signature.
    let c: SoftcapFn = unsafe { r.sym(c"rivoli_logit_softcap") };
    // Logits at scale 64 so `tanh` sees both its polynomial (|z| < 0.625) and its exp branch;
    // `(mult, cap)` per seed, Glimmer's `1/sqrt(128)`-scale multiplier among them.
    let make = |n: usize, seed: u64| {
        let x = with_nonfinite(draw(n, (seed, 1), 64.0, true), seed);
        let mult = [0.088_388_35, 1.0, 0.5][(seed % 3) as usize];
        let cap = [30.0, 15.0, 50.0, 2.0][(seed % 4) as usize];
        (vec![], vec![Out::InPlace(x)], (mult, cap))
    };
    sweep(
        t,
        LOGIT_SWEEP,
        make,
        |side, cell: &mut Cell, n, &(m, cap)| softcap_on(c, side, cell.o(side, 0), (n, m, cap)),
    );
    let mut cases = vec![
        (0, 0.5, 30.0),
        (8, 30.0, 30.0),
        (8, 31.0, 30.0),
        (0, 31.0, f32::NAN),
    ];
    for bad in [f32::NAN, 0.0, -1.0, f32::INFINITY] {
        cases.extend([(8, 0.5, bad), (8, bad, 30.0)]);
    }
    guards(t, "logit_softcap", &cases, |side, case, (_, go)| {
        softcap_on(c, side, go, case)
    });
}

/// `rivoli_argmax` on `side`: `logits` into `(index, value)`.
fn argmax_on(c: ArgmaxFn, side: Side, (l, idx, val): (I, *mut i32, O), n: usize) -> i32 {
    // SAFETY: as `embed_on`.
    match side {
        Side::Hip => unsafe { c(l, n as i32, idx, val) },
        Side::Rust => code(unsafe { hip::launch_argmax(l, n, idx, val) }),
    }
}

/// Logits for `seed`: draws with an exact tie for the maximum (the lower index must win) on
/// one seed class, NaN/inf on another, and all-NaN and all-`-inf` rows (the identity must
/// survive) on two more.
fn logits(n: usize, seed: u64) -> Vec<f32> {
    let mut v = draw(n, (seed, 1), 8.0, true);
    let at = |k: u64| (seed as usize * 31 + k as usize * 977) % n;
    match seed % 8 {
        0 => {
            let (a, b) = (at(1), at(2));
            v[a] = 1e30;
            v[b] = 1e30;
        }
        1 => v = with_nonfinite(v, seed),
        2 => v.fill(f32::NAN),
        3 => v.fill(f32::NEG_INFINITY),
        4 => v[at(3)] = f32::INFINITY,
        _ => {}
    }
    v
}

pub fn argmax(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_argmax`'s C signature.
    let c: ArgmaxFn = unsafe { r.sym(c"rivoli_argmax") };
    let make = |n: usize, seed: u64| {
        (
            vec![logits(n, seed)],
            vec![Out::PoisonBits(1), Out::Poison(1)],
            (),
        )
    };
    sweep(t, ARGMAX_SWEEP, make, |side, x: &mut Cell, n, _| {
        argmax_on(c, side, (x.i(0), x.o(side, 0).cast(), x.o(side, 1)), n)
    });
    guards(t, "argmax n", &[0], |side, n, (gi, go)| {
        argmax_on(c, side, (gi, go.cast(), go), n)
    });
}

/// `rivoli_flag_nonfinite` on `side`: `x` into `flag` with `tag`.
fn flag_on(c: FlagFn, side: Side, (x, flag): (I, *mut u32), (n, tag): (usize, u32)) -> i32 {
    // SAFETY: as `embed_on`.
    match side {
        Side::Hip => unsafe { c(x, n as i32, tag, flag) },
        Side::Rust => code(unsafe { hip::launch_flag_nonfinite(x, n, tag, flag) }),
    }
}

pub fn flag_nonfinite(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_flag_nonfinite`'s C signature.
    let c: FlagFn = unsafe { r.sym(c"rivoli_flag_nonfinite") };
    // GLM's tag shape, `1 + pos·256 + layer`; half the cells all-finite (flag must stay 0).
    let make = |n: usize, seed: u64| {
        let x = with_nonfinite(draw(n, (seed, 1), 1.0, true), seed);
        (
            vec![x],
            vec![Out::InPlaceBits(vec![0.0])],
            1 + seed as u32 * 256 + 17,
        )
    };
    sweep(t, ELEMENTWISE_SWEEP, make, |side, x: &mut Cell, n, &tag| {
        flag_on(c, side, (x.i(0), x.o(side, 0).cast()), (n, tag))
    });
    guards(t, "flag_nonfinite n", &[0], |side, n, (gi, go)| {
        flag_on(c, side, (gi, go.cast()), (n, 7))
    });
}

/// `rivoli_hash_rows` on `side`: `x` into `out`.
fn hash_on(c: HashFn, side: Side, (x, out): (I, *mut u64), span: HashSpan) -> i32 {
    let HashSpan { n, stride, i_base } = span;
    // SAFETY: as `embed_on`.
    match side {
        Side::Hip => unsafe { c(x, n as i32, stride as i32, i_base, out, null_mut()) },
        Side::Rust => code(unsafe { hip::launch_hash_rows(x, span, out, null_mut()) }),
    }
}

pub fn hash_rows(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_hash_rows`'s C signature.
    let c: HashFn = unsafe { r.sym(c"rivoli_hash_rows") };
    // Raw bits, NaN payloads included — the fold reads bits, never values; `i_base` 0 (one
    // buffer) or a slot offset (several folded into one accumulator).
    let make = |(n, _): (usize, usize), seed: u64| {
        let i_base = [0, n as u64 * (seed % 5), 1 << 40][(seed % 3) as usize];
        (
            vec![words(&bytes(n * 4, (seed, 1)))],
            vec![Out::InPlaceBits(vec![0.0; 2])],
            i_base,
        )
    };
    sweep(
        t,
        HASH_SWEEP,
        make,
        |side, x: &mut Cell, (n, stride), &i_base| {
            let span = HashSpan { n, stride, i_base };
            hash_on(c, side, (x.i(0), x.o(side, 0).cast()), span)
        },
    );
    guards(
        t,
        "hash_rows",
        &[(0, 1), (8, 0), (0, 0)],
        |side, (n, stride), (gi, go)| {
            hash_on(
                c,
                side,
                (gi, go.cast()),
                HashSpan {
                    n,
                    stride,
                    i_base: 0,
                },
            )
        },
    );
}
