//! Rows for `kernels/attn.hip`'s GQA pair (Muse Glimmer's 32Q/2KV causal attend with its
//! 2048-row ring, and the DFlash drafter's 32Q/8KV bidirectional block) and K3's dense MHA
//! (96 heads, the 192-wide nope‖rope key, a 128-wide value, an additive mask). The inputs
//! follow `attn.rs`'s rules (binade spread, edge seeds, nasty seeds with ±inf/NaN).

use rivoli_backend::hip;
use std::ptr::{null, null_mut};

use super::attn_inputs::{nasty, plant_nonfinite};
use super::common::{AttendIo, AttendLauncher, attend_launch};
use super::device::{Cell, I, O, Out, Ref, S, Shapes, Side, Tally, both, code, guard_ptrs, sweep};
use super::inputs::draw;

/// Both GQA launchers' C signature: `(q, k, v, hq, hkv, d, tq, pos, win, slots, scale, out,
/// stream)` — `pos`/`slots` are `start_pos`/`ring_cap` for one, `q_offset`/`kv_len` for the
/// other. The Rust pair's is `common::AttendLauncher`.
type GqaFn = unsafe extern "C" fn(I, I, I, i32, i32, i32, i32, i32, i32, i32, f32, O, S) -> i32;
type MhaFn = unsafe extern "C" fn(I, I, I, I, i32, i32, i32, i32, f32, O) -> i32;

/// `(hq, hkv, d, tq, pos, win, slots)` — the launcher's integer arguments, in its order.
type GqaShape = [usize; 7];

/// Glimmer decode at a sliding layer (ring 2048) before the ring fills, as it fills, past one
/// wrap and deep into a later one; at a global layer (win 0, no ring) at row 0 and at 8191;
/// a 4-row batch over a `win + tq − 1` ring; the S1b golden width (d 8), a width that does
/// not tile the wave (100) and the cap (256).
const GQA_SWEEP: Shapes<GqaShape> = Shapes {
    all: &[
        [32, 2, 128, 1, 5, 2048, 2048],
        [32, 2, 128, 1, 2047, 2048, 2048],
        [32, 2, 128, 1, 2048, 2048, 2048],
        [32, 2, 128, 1, 4095, 2048, 2048],
        [32, 2, 128, 1, 0, 0, 0],
        [32, 2, 128, 1, 8191, 0, 0],
        [32, 2, 128, 4, 300, 16, 19],
        [4, 2, 8, 3, 9, 4, 0],
        [6, 3, 100, 2, 40, 0, 0],
        [2, 1, 256, 1, 33, 7, 7],
    ],
    hot: [32, 2, 128, 1, 4095, 2048, 2048],
};

/// The drafter: ctx 4096 (the shipped geometry — each row's window reaches back 2048), the
/// golden's `q_offset` 0 with ctx 12 / block 4 / win 13, a block whose window clamps at both
/// ends, and odd widths.
const BLOCK_SWEEP: Shapes<GqaShape> = Shapes {
    all: &[
        [32, 8, 128, 16, 4096, 2048, 4112],
        [32, 8, 128, 16, 100, 2048, 116],
        [32, 8, 128, 4, 0, 13, 16],
        [8, 8, 100, 16, 50, 3, 66],
        [4, 2, 256, 1, 0, 1, 1],
        [2, 1, 8, 5, 9, 2, 14],
    ],
    hot: [32, 8, 128, 16, 4096, 2048, 4112],
};

/// A GQA launcher on `side`: `[q, k, v]` into `out`.
fn gqa_on(
    c: (GqaFn, AttendLauncher),
    side: Side,
    ([q, k, v], out): ([I; 3], O),
    (shape, scale): (GqaShape, f32),
) -> i32 {
    let [hq, hkv, hd, tq, pos, win, slots] = shape.map(|x| x as i32);
    let (io, n) = (AttendIo { q, k, v, out }, null_mut());
    // SAFETY: live buffers sized for the shape, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c.0(q, k, v, hq, hkv, hd, tq, pos, win, slots, scale, out, n) },
        Side::Rust => code(unsafe { attend_launch(c.1, io, shape, scale) }),
    }
}

/// The cache rows a shape needs: the ring, the linear prefix, or `kv_len`.
type CacheRows = fn(GqaShape) -> usize;

/// One GQA row: sweep, then the guards (each must refuse on both sides with one code).
fn gqa_row(
    t: &mut Tally,
    c: (GqaFn, AttendLauncher),
    (sweep_at, rows): (Shapes<GqaShape>, CacheRows),
    guards: &[GqaShape],
) {
    let make = |s: GqaShape, seed: u64| {
        let [hq, hkv, d, tq, ..] = s;
        let mut q = draw(tq * hq * d, (seed, 1), 0.5, seed % 4 == 1);
        plant_nonfinite(&mut q, (seed, 1));
        let kv = rows(s) * hkv * d;
        let mut v = draw(kv, (seed, 3), 1.0, seed % 4 == 3);
        plant_nonfinite(&mut v, (seed, 3));
        let ins = vec![q, draw(kv, (seed, 2), 1.0, seed % 4 == 2), v];
        let scale = [(d as f32).powf(-0.5), 1.0, 3.87][(seed % 3) as usize];
        (ins, vec![Out::Poison(tq * hq * d)], scale)
    };
    sweep(t, sweep_at, make, |side, x: &mut Cell, s, &scale| {
        gqa_on(
            c,
            side,
            ([x.i(0), x.i(1), x.i(2)], x.o(side, 0)),
            (s, scale),
        )
    });
    let (gi, go) = guard_ptrs();
    for &s in guards {
        let refusal = both(|side| gqa_on(c, side, ([gi; 3], go), (s, 1.0)));
        t.guard(&format!("{s:?}"), refusal);
    }
}

/// `usize` for the C's `-1` (the Rust wrappers take `usize` and narrow with `as i32`).
const NEG: usize = usize::MAX;

pub fn gqa_attend(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gqa_attend`'s C signature.
    let c: GqaFn = unsafe { r.sym(c"rivoli_gqa_attend") };
    let rows: CacheRows = |[.., tq, pos, _, ring]| if ring == 0 { pos + tq } else { ring };
    // Each extent at 0 and each signed one at −1 (1001); an inexact group (1003); d past the
    // registers (1002); a ring on a global layer and a ring shorter than `win + tq − 1`
    // (1005); then order (1001 before 1003, 1003 before 1002, 1002 before 1005).
    let guards = [
        [0, 2, 8, 1, 0, 0, 0],
        [4, 0, 8, 1, 0, 0, 0],
        [4, 2, 0, 1, 0, 0, 0],
        [4, 2, 8, 0, 0, 0, 0],
        [4, 2, 8, 1, NEG, 0, 0],
        [4, 2, 8, 1, 0, NEG, 0],
        [4, 2, 8, 1, 0, 4, NEG],
        [32, 3, 8, 1, 0, 0, 0],
        [4, 2, 257, 1, 0, 0, 0],
        [4, 2, 8, 1, 0, 0, 5],
        [4, 2, 8, 2, 0, 4, 4],
        [32, 3, 8, 0, 0, 0, 0],
        [32, 3, 300, 1, 0, 0, 0],
        [4, 2, 300, 1, 0, 0, 5],
    ];
    gqa_row(t, (c, hip::launch_gqa_attend), (GQA_SWEEP, rows), &guards);
}

pub fn gqa_block_attend(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gqa_block_attend`'s C signature.
    let c: GqaFn = unsafe { r.sym(c"rivoli_gqa_block_attend") };
    let rows: CacheRows = |s| s[6];
    // Extents, `q_offset` −1, `kv_len` 0 and `win` 0 (1001); group (1003); d (1002); a block
    // longer than its buffer, and one placed past it (1005); then order.
    let guards = [
        [0, 2, 8, 1, 0, 1, 1],
        [4, 2, 8, 0, 0, 1, 1],
        [4, 2, 8, 1, NEG, 1, 1],
        [4, 2, 8, 1, 0, 1, 0],
        [4, 2, 8, 1, 0, 0, 1],
        [32, 3, 8, 1, 0, 1, 1],
        [4, 2, 257, 1, 0, 1, 1],
        [4, 2, 8, 4, 0, 1, 3],
        [4, 2, 8, 2, 100, 10, 50],
        [32, 3, 8, 1, 0, 0, 1],
        [4, 2, 257, 2, 100, 10, 50],
    ];
    gqa_row(
        t,
        (c, hip::launch_gqa_block_attend),
        (BLOCK_SWEEP, rows),
        &guards,
    );
}

// ── MHA ─────────────────────────────────────────────────────────────────────────────────

/// `(heads, kv, d, dv, masked)`: `masked` hands an additive row (0 up to a seeded position,
/// −inf past it, as K3's engine does); otherwise the mask is null.
type MhaShape = (usize, usize, usize, usize, bool);

/// K3's attend (96 heads, 192/128) at 1024 cached rows behind its mask; a lone row; the
/// staging ceiling (8192); unmasked odd widths that leave threads idle (kv 300 < 2·256).
const MHA_SWEEP: Shapes<MhaShape> = Shapes {
    all: &[
        (96, 1024, 192, 128, true),
        (4, 1, 192, 128, true),
        (2, 8192, 192, 128, true),
        (3, 300, 64, 70, false),
        (1, 257, 7, 300, false),
    ],
    hot: (96, 1024, 192, 128, true),
};

/// `rivoli_mha_attend` on `side`: `[q, k, v, mask]` into `out` (`mask` may be null).
fn mha_on(c: MhaFn, side: Side, (ins, out): ([I; 4], O), (shape, scale): (MhaShape, f32)) -> i32 {
    let [q, k, v, mask] = ins;
    let (heads, kv, d, dv, _) = shape;
    let [hi, kvi, di, dvi] = [heads, kv, d, dv].map(|x| x as i32);
    // SAFETY: live buffers sized for the shape, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c(q, k, v, mask, hi, kvi, di, dvi, scale, out) },
        Side::Rust => {
            code(unsafe { hip::launch_mha_attend(q, k, v, mask, heads, kv, d, dv, scale, out) })
        }
    }
}

/// The mask row: 0 through a seeded position, −inf after; on nasty seeds every slot −inf
/// (a fully-masked row — NaN, deliberately) with one NaN slot.
fn mha_mask(kv: usize, seed: u64) -> Vec<f32> {
    let reach = (seed as usize * 7919) % kv;
    let mut m: Vec<f32> = (0..kv)
        .map(|s| {
            if s <= reach && !nasty(seed) {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    if nasty(seed) {
        m[kv / 2] = f32::NAN;
    }
    m
}

pub fn mha_attend(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_mha_attend`'s C signature.
    let c: MhaFn = unsafe { r.sym(c"rivoli_mha_attend") };
    let make = |(heads, kv, d, dv, masked): MhaShape, seed: u64| {
        let mut q = draw(heads * d, (seed, 1), 0.3, seed % 4 == 1);
        plant_nonfinite(&mut q, (seed, 1));
        let mask = if masked {
            mha_mask(kv, seed)
        } else {
            Vec::new()
        };
        let ins = vec![
            q,
            draw(heads * kv * d, (seed, 2), 1.0, seed % 4 == 2),
            draw(heads * kv * dv, (seed, 3), 1.0, seed % 4 == 3),
            mask,
        ];
        let scale = [(192f32).powf(-0.5), 1.0][(seed % 2) as usize];
        (ins, vec![Out::Poison(heads * dv)], scale)
    };
    sweep(t, MHA_SWEEP, make, |side, x: &mut Cell, s, &scale| {
        let ins = [x.i(0), x.i(1), x.i(2), if s.4 { x.i(3) } else { null() }];
        mha_on(c, side, (ins, x.o(side, 0)), (s, scale))
    });
    let (gi, go) = guard_ptrs();
    // Each extent at 0 (1001); kv 0 and one past the ceiling (1004); then order.
    for s in [
        (0, 8, 8, 8, false),
        (1, 8, 0, 8, false),
        (1, 8, 8, 0, false),
        (1, 0, 8, 8, false),
        (1, 8193, 8, 8, false),
        (0, 0, 8, 8, false),
    ] {
        let refusal = both(|side| mha_on(c, side, ([gi; 4], go), (s, 1.0)));
        t.guard(&format!("mha {s:?}"), refusal);
    }
}
