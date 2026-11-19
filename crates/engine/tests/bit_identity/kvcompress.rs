//! Rows for `kernels/kvcompress.hip`'s four launchers at the geometries `v4/kvcompress.rs` builds:
//! the attention compressor (`head_dim` 512) and the indexer's (`index_head_dim` 128), each at ratio
//! 4 (overlapping, `coff` 2) and ratio 128, with `rope_head_dim` 64 — plus `gemm_bf16` at the
//! compressor's projection, the indexer's `wproj`, K3/Glimmer's one-row projection shape, and a
//! ragged tail.
//!
//! The C launchers' null-pointer refusal (1000) has no Rust spelling (the wrappers take
//! references), so it has no guard case here; every other code does.
//!
//! The `.hip` file named here is deleted from the tree; cite it at tag `archive/hip-kernels`.

use super::cells::{
    GEMM_SHAPES, GEOMS, PREFILL_SHAPES, derived, finish_ins, gemm_ins, geom, prefill_ins,
};
use super::device::{Cell, I, O, Out, Ref, S, Shapes, Side, Tally, both, code, guard_ptrs, sweep};
use super::inputs::draw;
use rivoli_backend::abi::{CompFinish, CompGeom};
use rivoli_backend::hip;
use std::ptr::null_mut;

type GemmFn = unsafe extern "C" fn(I, *const u16, O, i32, i32, i32, S) -> i32;
type DepositFn = unsafe extern "C" fn(I, I, I, O, O, *const CompGeom, i32, i32, S) -> i32;
type PrefillFn = unsafe extern "C" fn(I, I, I, *const CompFinish, *const CompGeom, i32, S) -> i32;
type DecodeFn = unsafe extern "C" fn(O, O, *const CompFinish, *const CompGeom, i32, S) -> i32;

/// Every compressor guard (`compress_guard`): each code reached on its own by a geometry that
/// clears the ones before it, and one failing several (order).
fn bad_geoms() -> Vec<(CompGeom, i32)> {
    let with = |f: fn(&mut CompGeom)| {
        let mut b = geom(4, 128);
        f(&mut b);
        b
    };
    vec![
        (with(|b| b.ratio = 1), 1001),
        (with(|b| b.coff = 3), 1001),
        (with(|b| b.rd = 130), 1002), // past d
        (with(|b| b.rd = 63), 1002),  // odd
        (with(|b| b.cd += 1), 1003),
        (with(|b| b.ents += 1), 1003),
        (derived(128, 2, 128), 1004), // 256 entries
        (derived(8, 2, 128), 1010),   // overlap off ratio 4
        (derived(4, 1, 128), 1010),   // ratio 4 without overlap
        (
            with(|b| {
                b.d = 0;
                b.rd = 63;
            }),
            1001, // before 1002
        ),
    ]
}

/// The guard buffer as inputs and output, and a `CompFinish` over it — what every pooling guard
/// case passes (a refused guard launches nothing; a regressed one writes the poisoned buffer).
fn guard_finish() -> (I, O, CompFinish) {
    let (gi, go) = guard_ptrs();
    (
        gi,
        go,
        CompFinish {
            norm: gi,
            freqs: gi,
            out: go,
        },
    )
}

/// A pooling launcher's guard cases: every [`bad_geoms`] case at the launcher's valid scalar `ok`,
/// then its own `phase` cases — each `(geometry, scalar, the code it must reach)`.
fn compress_guards(
    t: &mut Tally,
    name: &str,
    (ok, phase): (i64, &[(CompGeom, i64, i32)]),
    call: impl Fn(Side, &CompGeom, i64) -> i32,
) {
    let cases = bad_geoms().into_iter().map(|(p, want)| (p, ok, want));
    for (p, x, want) in cases.chain(phase.iter().copied()) {
        t.guard_code(
            &format!("{name}={x} {p:?}"),
            want,
            both(|side| call(side, &p, x)),
        );
    }
}

// ── gemm_bf16 ───────────────────────────────────────────────────────────────────────────────

fn gemm_on(c: GemmFn, side: Side, (x, w, out): (I, *const u16, O), [m, n, k]: [usize; 3]) -> i32 {
    let ci = |v: usize| v as i32;
    // SAFETY: live buffers sized for `[m, n, k]`, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c(x, w, out, ci(m), ci(n), ci(k), null_mut()) },
        Side::Rust => code(unsafe { hip::launch_gemm_bf16(x, w, out, m, n, k, null_mut()) }),
    }
}

pub fn gemm_bf16(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gemm_bf16`'s C signature.
    let c: GemmFn = unsafe { r.sym(c"rivoli_gemm_bf16") };
    let projections = Shapes {
        all: &GEMM_SHAPES,
        hot: GEMM_SHAPES[0],
    };
    let make = |dims: [usize; 3], seed: u64| {
        (
            gemm_ins(dims, seed),
            vec![Out::Poison(dims[0] * dims[1])],
            (),
        )
    };
    sweep(t, projections, make, |side, x: &mut Cell, dims, _| {
        gemm_on(c, side, (x.i(0), x.i(1).cast(), x.o(side, 0)), dims)
    });
    let (gi, go) = guard_ptrs();
    for dims in [[0, 4, 8], [4, 0, 8], [4, 4, 0], [0, 0, 0]] {
        let rc = both(|side| gemm_on(c, side, (gi, gi.cast(), go), dims));
        t.guard(&format!("gemm_bf16 {dims:?}"), rc);
    }
}

// ── kv_compress_deposit ─────────────────────────────────────────────────────────────────────

/// `[kv, score, ape]` in, `[kv_state, score_state]` out.
type DepositBufs = ([I; 3], [O; 2]);

fn deposit_on(
    c: DepositFn,
    side: Side,
    ([kv, sc, ape], [ks, ss]): DepositBufs,
    (p, s, slot0): (&CompGeom, i64, i64),
) -> i32 {
    let stream = null_mut();
    // SAFETY: live buffers sized for the geometry and run, or the guard buffer.
    match side {
        Side::Hip => unsafe {
            c(
                kv,
                sc,
                ape,
                ks,
                ss,
                std::ptr::from_ref(p),
                s as i32,
                slot0 as i32,
                stream,
            )
        },
        Side::Rust => code(unsafe {
            hip::launch_kv_compress_deposit(
                kv,
                sc,
                ape,
                ks,
                ss,
                p,
                s as usize,
                slot0 as usize,
                stream,
            )
        }),
    }
}

pub fn kv_compress_deposit(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_kv_compress_deposit`'s C signature.
    let c: DepositFn = unsafe { r.sym(c"rivoli_kv_compress_deposit") };
    // `(geometry index, s, slot0)`: decode steps at the first, a middle and the last slot; a short
    // prefill (only a remainder); prefills long enough to carry the overlap copy; whole windows.
    let runs = Shapes {
        all: &[
            (0usize, 1usize, 2usize),
            (0, 1, 0),
            (0, 1, 3),
            (1, 1, 127),
            (0, 13, 0),
            (2, 7, 0),
            (1, 300, 0),
            (3, 256, 0),
            (0, 8, 0),
        ],
        hot: (0, 1, 2),
    };
    let make = |(gi, s, _): (usize, usize, usize), seed: u64| {
        let p = geom(GEOMS[gi].0, GEOMS[gi].1);
        let (cd, state) = (p.cd as usize, (p.coff * p.ratio * p.cd) as usize);
        let ins = vec![
            draw(s * cd, (seed, 1), 1.0, true),
            draw(s * cd, (seed, 2), 4.0, true),
            draw(p.ratio as usize * cd, (seed, 3), 1.0, false),
        ];
        let outs = vec![
            Out::InPlace(draw(state, (seed, 4), 1.0, false)),
            Out::InPlace(draw(state, (seed, 5), 1.0, false)),
        ];
        (ins, outs, p)
    };
    sweep(t, runs, make, |side, x: &mut Cell, (_, s, slot0), p| {
        let bufs = ([x.i(0), x.i(1), x.i(2)], [x.o(side, 0), x.o(side, 1)]);
        deposit_on(c, side, bufs, (p, s as i64, slot0 as i64))
    });
    let (gi, go) = guard_ptrs();
    let none: DepositBufs = ([gi; 3], [go; 2]);
    let good = geom(4, 128);
    for (p, want) in bad_geoms() {
        let rc = both(|side| deposit_on(c, side, none, (&p, 1, 0)));
        t.guard_code(&format!("kv_compress_deposit {p:?}"), want, rc);
    }
    // `s <= 0`, a negative slot, a run past the table, and both phase guards at once.
    for (s, slot0, want) in [(0, 0, 1005), (1, -1, 1008), (3, 2, 1008), (0, -1, 1005)] {
        let rc = both(|side| deposit_on(c, side, none, (&good, s, slot0)));
        t.guard_code(
            &format!("kv_compress_deposit s={s} slot0={slot0}"),
            want,
            rc,
        );
    }
}

// ── the two pooling phases ──────────────────────────────────────────────────────────────────

fn prefill_on(
    c: PrefillFn,
    side: Side,
    ([kv, sc, ape], f): ([I; 3], &CompFinish),
    (p, nblk): (&CompGeom, i64),
) -> i32 {
    // SAFETY: live buffers sized for the geometry and `nblk`, or the guard buffer.
    match side {
        Side::Hip => unsafe { c(kv, sc, ape, f, p, nblk as i32, null_mut()) },
        Side::Rust => code(unsafe {
            hip::launch_kv_compress_prefill(kv, sc, ape, f, p, nblk as usize, null_mut())
        }),
    }
}

pub fn kv_compress_prefill(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_kv_compress_prefill`'s C signature.
    let c: PrefillFn = unsafe { r.sym(c"rivoli_kv_compress_prefill") };
    // `(geometry index, nblk)`: block 0 alone (its masked overlap half), a few, and a 2048-token
    // prefill at ratio 4.
    let blocks = Shapes {
        all: &PREFILL_SHAPES,
        hot: (0, 512),
    };
    let make = |(gi, nblk): (usize, usize), seed: u64| {
        let p = geom(GEOMS[gi].0, GEOMS[gi].1);
        (
            prefill_ins(&p, nblk, seed),
            vec![Out::Poison(nblk * p.d as usize)],
            p,
        )
    };
    sweep(t, blocks, make, |side, x: &mut Cell, (_, nblk), p| {
        let f = CompFinish {
            norm: x.i(3),
            freqs: x.i(4),
            out: x.o(side, 0),
        };
        prefill_on(c, side, ([x.i(0), x.i(1), x.i(2)], &f), (p, nblk as i64))
    });
    let (gi, _, f) = guard_finish();
    let phase = [(geom(4, 128), 0, 1006), (geom(1, 0), 0, 1001)];
    compress_guards(
        t,
        "kv_compress_prefill nblk",
        (1, &phase),
        |side, p, nblk| prefill_on(c, side, ([gi; 3], &f), (p, nblk)),
    );
}

/// `rivoli_kv_compress_decode` on `side`: the state pair, the finish, and `(p, start_pos)`.
fn decode_on(
    c: DecodeFn,
    side: Side,
    (state, f): ([O; 2], &CompFinish),
    (p, pos): (&CompGeom, i64),
) -> i32 {
    let [ks, ss] = state;
    let stream = null_mut();
    // SAFETY: live buffers sized for the geometry and position, or the guard buffer.
    unsafe {
        if side == Side::Hip {
            return c(ks, ss, f, p, pos as i32, stream);
        }
        code(hip::launch_kv_compress_decode(
            ks,
            ss,
            f,
            p,
            pos as usize,
            stream,
        ))
    }
}

pub fn kv_compress_decode(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_kv_compress_decode`'s C signature.
    let c: DecodeFn = unsafe { r.sym(c"rivoli_kv_compress_decode") };
    // `(geometry index, start_pos)`: each geometry's first completed window and a later one —
    // ratio 128's SECOND block (255) is where the block index and position part (the C's note).
    let steps = Shapes {
        all: &[
            (0usize, 3usize),
            (0, 4095),
            (1, 127),
            (1, 255),
            (2, 7),
            (3, 255),
        ],
        hot: (0, 4095),
    };
    let make = |(gi, pos): (usize, usize), seed: u64| {
        let p = geom(GEOMS[gi].0, GEOMS[gi].1);
        let state = (p.coff * p.ratio * p.cd) as usize;
        let [norm, freqs] = finish_ins(&p, pos + 1, seed);
        let outs = vec![
            Out::InPlace(draw(state, (seed, 1), 1.0, true)),
            Out::InPlace(draw(state, (seed, 2), 4.0, true)),
            Out::Poison(p.d as usize),
        ];
        (vec![norm, freqs], outs, p)
    };
    sweep(t, steps, make, |side, x: &mut Cell, (_, pos), p| {
        let f = CompFinish {
            norm: x.i(0),
            freqs: x.i(1),
            out: x.o(side, 2),
        };
        decode_on(c, side, ([x.o(side, 0), x.o(side, 1)], &f), (p, pos as i64))
    });
    let (_, go, f) = guard_finish();
    let good = geom(4, 128);
    let phase = [
        (good, 0, 1007),
        (good, -4, 1007),
        (good, 4, 1009),
        (geom(1, 0), 0, 1001),
    ];
    compress_guards(t, "kv_compress_decode pos", (3, &phase), |side, p, pos| {
        decode_on(c, side, ([go; 2], &f), (p, pos))
    });
}
