//! Rows for `kernels/attn.hip`'s split-softmax pair: MLA's latent attend (two kernels behind
//! one launcher) and V4's shared-KV gather attend. `attn_gqa.rs` holds the GQA pair and K3's
//! dense MHA. Shapes are the engine's (GLM's 64 heads × kvl 512 + rope 64; V4's 64 heads ×
//! 512 over a 128-slot window or 128 + 512 compressed) plus the shapes that reach every
//! branch: one row, a ragged tile, one split, two, a ragged last split, the 16-split ceiling,
//! no scratch at long context, heads that do not fill a block, sparse rows, a fully-masked
//! query, and a `topk` past the block width.
//!
//! Inputs: `attn_inputs.rs`.

use super::attn_inputs::{
    Mla, ROPE, as_words, index_words, mla, mla_inputs, nasty, plant_nonfinite,
};
use super::common::Lcg;
use super::device::{Cell, I, O, Out, Ref, S, Shapes, Side, Tally, both, code, guard_ptrs, sweep};
use super::inputs::draw;
use rivoli_backend::hip;
use std::ptr::{null, null_mut};

// ── MLA ─────────────────────────────────────────────────────────────────────────────────

// The non-f32 operand pointers: e4m3 bytes, bf16 halves, u32 row indices, i32 slots.
type B = *const u8;
type H = *const u16;
type W = *const u32;
type Ix = *const i32;
type MlaFn = unsafe extern "C" fn(I, I, B, I, H, W, i32, i32, i32, i32, i32, f32, O, O) -> i32;
type ScratchFn = unsafe extern "C" fn(i32, i32) -> usize;

/// With GLM's 64 heads (4 head-blocks, `by_grid` 20 above the 16 cap), the split plan gives:
/// nr 1, 17, 64 → one split (a lone ragged tile, two tiles, four); 130 → 2 (ragged last);
/// 1000 → 13 (tps 5, ragged); 2048 → 16 (the ceiling); 4100 → 16 (tps 17, ragged last tile).
/// 2048 without scratch is the single-split path at long context. 128 heads (8 head-blocks)
/// bind `by_grid` instead: 10 splits at 2048 (tps 13, ragged last). 20 heads leave 12 of the
/// second block's 16 waves idle and split 300 sparse rows 4 ways; 8 heads × 40 rows at kvl
/// 256 is one split over 8 accumulator registers of 16.
const MLA_SWEEP: Shapes<Mla> = Shapes {
    all: &[
        mla([64, 1, 512], false, true),
        mla([64, 17, 512], false, true),
        mla([64, 64, 512], false, true),
        mla([64, 130, 512], false, true),
        mla([64, 1000, 512], false, true),
        mla([64, 2048, 512], false, true),
        mla([64, 4100, 512], false, true),
        mla([64, 2048, 512], false, false),
        mla([128, 2048, 512], false, true),
        mla([20, 300, 512], true, true),
        mla([8, 40, 256], false, true),
    ],
    hot: mla([64, 2048, 512], false, true),
};

/// `rivoli_mla_attend` on `side`: `[qabs, qrope, lc8, lscale, rc, rows]` in (`rows` null =
/// dense), `[clat, partial]` out (`partial` null = no scratch). `dims` is `(h, nr, kvl, rope,
/// n_blocks)`.
fn mla_on(
    c: MlaFn,
    side: Side,
    (ins, outs): ([I; 6], [O; 2]),
    (dims, scale): ([usize; 5], f32),
) -> i32 {
    let [qa, qr, lc, ls, rc, rows] = ins;
    let (lc, rc, rows): (B, H, W) = (lc.cast(), rc.cast(), rows.cast());
    let [clat, part] = outs;
    let [h, nr, kvl, rope, nb] = dims;
    let [hi, nri, kvli, ropei, nbi] = dims.map(|v| v as i32);
    // SAFETY: live buffers sized for `dims`, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe {
            c(
                qa, qr, lc, ls, rc, rows, hi, nri, kvli, ropei, nbi, scale, clat, part,
            )
        },
        Side::Rust => code(unsafe {
            hip::launch_attend(
                qa, qr, lc, ls, rc, rows, h, nr, kvl, rope, nb, scale, clat, part,
            )
        }),
    }
}

pub fn mla_attend(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_mla_attend`'s and `rivoli_mla_attend_scratch_floats`'s C signatures.
    let c: MlaFn = unsafe { r.sym(c"rivoli_mla_attend") };
    let floats: ScratchFn = unsafe { r.sym(c"rivoli_mla_attend_scratch_floats") };
    // The scratch size is the C's arithmetic too: compared, not assumed (a guard-shaped
    // check: a mismatch is a bad guard with a note).
    for (h, kvl) in [(64, 512), (20, 512), (8, 256), (0, 512), (64, -1)] {
        // SAFETY: pure arithmetic.
        let want = unsafe { floats(h, kvl) };
        let got = hip::attend_scratch_floats(h.max(0) as usize, kvl.max(0) as usize);
        t.guards += 1;
        if got != want {
            t.bad_guards += 1;
            t.notes.push(format!(
                "scratch floats ({h}, {kvl}): hipcc {want}, Rust {got}"
            ));
        }
    }
    let make = |s: Mla, seed: u64| {
        let partial = hip::attend_scratch_floats(s.h, s.kvl);
        // The partials are compared too: zero-filled on both sides, so a split slot neither
        // twin wrote agrees and every written `acc`/`(m, l)` must match.
        let mut outs = vec![Out::Poison(s.h * s.kvl)];
        if s.scratch {
            outs.push(Out::InPlace(vec![0.0; partial]));
        }
        let scale = [0.072_168_78, 1.0, 0.01][(seed % 3) as usize];
        (mla_inputs(s, seed), outs, scale)
    };
    sweep(t, MLA_SWEEP, make, |side, x: &mut Cell, s, &scale| {
        let rows = if s.sparse { x.i(5) } else { null() };
        let part = if s.scratch { x.o(side, 1) } else { null_mut() };
        let ins = [x.i(0), x.i(1), x.i(2), x.i(3), x.i(4), rows];
        let dims = [s.h, s.nr, s.kvl, ROPE, s.kvl / 128];
        mla_on(c, side, (ins, [x.o(side, 0), part]), (dims, scale))
    });
    let (gi, go) = guard_ptrs();
    // Each extent at 0 (1001); a rope whose tile overflows LDS (1002); kvl past the registers
    // and kvl not a multiple of 128 (1004); then three that fail several at once (order).
    // `dims` is `(h, nr, kvl, rope, n_blocks)`.
    for dims in [
        [0, 8, 512, 64, 4],
        [4, 0, 512, 64, 4],
        [4, 8, 0, 64, 4],
        [4, 8, 512, 64, 0],
        [4, 8, 512, 0, 4],
        [4, 8, 512, 4096, 4],
        [4, 8, 640, 64, 5],
        [4, 8, 160, 64, 1],
        [0, 8, 160, 4096, 1],
        [4, 8, 160, 4096, 1],
        [4, 0, 640, 64, 5],
    ] {
        let none = ([gi, gi, gi, gi, gi, null()], [go, go]);
        let refusal = both(|side| mla_on(c, side, none, (dims, 1.0)));
        t.guard(&format!("mla {dims:?}"), refusal);
    }
}

// ── gather ──────────────────────────────────────────────────────────────────────────────

type GatherFn = unsafe extern "C" fn(I, I, I, Ix, i32, i32, i32, i32, f32, O, S) -> i32;

/// `(m query rows, h heads, d, topk slots, kv rows)`.
type GatherShape = (usize, usize, usize, usize, usize);

/// V4's decode at a ratio-0 layer (topk 128 over the window) and a compressed one (640 over
/// window + compressed tail); a prefill batch; odd widths (d 100 leaves most of a block
/// idle, 1000 runs four columns per thread); `topk` 2000 strides every thread past the block;
/// d 8 × topk 16118 sits exactly at the C's LDS ceiling (65536 B with its static `red`/`bc`).
const GATHER_SWEEP: Shapes<GatherShape> = Shapes {
    all: &[
        (1, 64, 512, 128, 128),
        (1, 64, 512, 640, 640),
        (7, 64, 512, 128, 135),
        (2, 3, 100, 37, 50),
        (1, 2, 1000, 300, 400),
        (1, 4, 64, 2000, 3000),
        (1, 1, 8, 16118, 16118),
    ],
    hot: (1, 64, 512, 640, 640),
};

/// `rivoli_gather_attn_shared_kv` on `side`: `[q, kv, sink, idxs]` into `o`.
fn gather_on(
    c: GatherFn,
    side: Side,
    (ins, o): ([I; 4], O),
    (shape, scale): (GatherShape, f32),
) -> i32 {
    let [q, kv, sink, ix] = ins;
    let ix: Ix = ix.cast();
    let (m, h, d, topk, _) = shape;
    let [mi, hi, di, ki] = [m, h, d, topk].map(|v| v as i32);
    // SAFETY: live buffers sized for the shape, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c(q, kv, sink, ix, mi, hi, di, ki, scale, o, null_mut()) },
        Side::Rust => code(unsafe {
            hip::launch_gather_attn_shared_kv(q, kv, sink, ix, m, h, d, topk, scale, o, null_mut())
        }),
    }
}

/// The index list: ~1 in 10 slots masked (`-1`); on nasty seeds query row 0 is masked
/// entirely — the NaN a query that attended nothing owes.
fn gather_idxs((m, _, _, topk, rows): GatherShape, seed: u64) -> Vec<f32> {
    let mut r = Lcg(seed.wrapping_mul(0x9e37_79b9) ^ 0x1d);
    let mut ix = index_words(m * topk, rows, &mut r);
    for (n, w) in ix.iter_mut().enumerate() {
        let dropped = r.f() > 0.8;
        let row0 = nasty(seed) && n < topk;
        if dropped || row0 {
            *w = u32::MAX; // -1
        }
    }
    as_words(&ix)
}

pub fn gather_attn_shared_kv(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gather_attn_shared_kv`'s C signature.
    let c: GatherFn = unsafe { r.sym(c"rivoli_gather_attn_shared_kv") };
    let make = |shape: GatherShape, seed: u64| {
        let (m, h, d, _, rows) = shape;
        let mut q = draw(m * h * d, (seed, 1), 0.1, seed % 4 == 1);
        plant_nonfinite(&mut q, (seed, 1));
        let ins = vec![
            q,
            draw(rows * d, (seed, 2), 1.0, seed % 4 == 2),
            draw(h, (seed, 3), 2.0, false),
            gather_idxs(shape, seed),
        ];
        let scale = [(d as f32).powf(-0.5), 1.0][(seed % 2) as usize];
        (ins, vec![Out::Poison(m * h * d)], scale)
    };
    sweep(
        t,
        GATHER_SWEEP,
        make,
        |side, x: &mut Cell, shape, &scale| {
            let ins = [x.i(0), x.i(1), x.i(2), x.i(3)];
            gather_on(c, side, (ins, x.o(side, 0)), (shape, scale))
        },
    );
    let (gi, go) = guard_ptrs();
    // Each extent at 0 (1001); d past the accumulator (1002); LDS one float over the ceiling
    // the sweep sits on (1006 — a guard that forgot `bc[2]` would accept it); then order.
    for shape in [
        (0, 1, 8, 8, 1),
        (1, 0, 8, 8, 1),
        (1, 1, 0, 8, 1),
        (1, 1, 8, 0, 1),
        (1, 1, 1025, 8, 1),
        (1, 1, 8, 16119, 1),
        (1, 1, 1025, 16119, 1),
        (0, 1, 1025, 16119, 1),
    ] {
        let refusal = both(|side| gather_on(c, side, ([gi; 4], go), (shape, 1.0)));
        t.guard(&format!("gather {shape:?}"), refusal);
    }
}
