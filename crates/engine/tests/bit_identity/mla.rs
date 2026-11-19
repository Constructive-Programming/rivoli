//! Rows for `kernels/mla.hip`'s seven launchers.
//!
//! - **MLA pair** (`glm/attn.rs`): a GLM-class shape (64 heads, nope 192, v 256, `kv_lora` 512,
//!   128-tiles: the DeepSeek-family layout the kernels were written for) and Kimi-K3's pinned
//!   dims (96 heads, 128/128, 512; `artifact/tests/k3_config.rs`), each at one and two token rows,
//!   which are the two kernels each launcher picks between. Plus the oracle suite's two fallback
//!   shapes (`kvl % 4 != 0`, `block < 4`; `tests/kernel_attend.rs`).
//! - **V4 helpers** (`v4/{attn,forward,kvcompress,moe,engine}.rs`): V4-Flash's `head_dim` 512 with `rope_head_dim` 64 over 64 heads, `q_lora` 1024, `dim` 4096,
//!   the KV quantizer's partial prefix at block 64, and the grouped output projection.
//!
//! Every production call to `rmsnorm_batch`, `qk_norm` and `rope_adjacent` is in place (the
//! kernels take one buffer), and so is `act_quant_f8_prefix` at most of its call sites (`src ==
//! dst`). Those rows are in place; the quantizer also sweeps the copy form.
//!
//! The `.hip` file named here is deleted from the tree; cite it at tag `archive/hip-kernels`.

use super::cells::{MLA_DIMS, MlaShape, mla_ins};
use super::device::{Cell, I, O, Out, Ref, S, Shapes, Side, Tally, both, guard_ptrs, sweep, twin};
use super::inputs::{draw, fp8_words, uniform};
use rivoli_backend::hip;
use std::ptr::null_mut;

type AbsorbFn = unsafe extern "C" fn(I, *const u8, I, i32, i32, i32, i32, i32, i32, i32, O) -> i32;
type ValueFn = unsafe extern "C" fn(I, *const u8, I, i32, i32, i32, i32, i32, i32, O) -> i32;
type PrefixFn = unsafe extern "C" fn(I, O, i32, i32, i32, i32, S) -> i32;
type NormFn = unsafe extern "C" fn(O, I, i32, i32, f32, S) -> i32;
type QkFn = unsafe extern "C" fn(O, i32, i32, f32, S) -> i32;
type RopeFn = unsafe extern "C" fn(O, I, i32, i32, i32, i32, i32, i32, S) -> i32;
type GemvFn = unsafe extern "C" fn(I, *const u8, I, i32, i32, i32, i32, i32, O, S) -> i32;

/// The MLA pair's shared guard cases, `((dims, nrow), the code it must reach)`: each extent, a
/// non-power-of-two tile, a row count with no kernel, then the order — the tile before the row
/// count, and the extents before both. `qh` (index 1) is the absorb's alone.
const MLA_GUARDS: [(MlaShape, i32); 10] = [
    (([0, 8, 8, 8, 8, 4], 1), 1001),
    (([2, 0, 8, 8, 8, 4], 1), 1001),
    (([2, 8, 0, 8, 8, 4], 1), 1001),
    (([2, 8, 8, 0, 8, 4], 1), 1001),
    (([2, 8, 8, 8, 0, 4], 1), 1001),
    (([2, 8, 8, 8, 8, 0], 1), 1001),
    (([2, 8, 8, 8, 8, 6], 1), 1003),
    (([2, 8, 8, 8, 8, 4], 3), 1004),
    (([2, 8, 8, 8, 8, 6], 3), 1003),
    (([0, 8, 8, 8, 8, 6], 3), 1001),
];

// ── the MLA pair ────────────────────────────────────────────────────────────────────────────

/// Which MLA projection a row drives, with its reference launcher. One adapter and one row body
/// serve both: they read the same `kv_b` layout and differ in `qh` (the absorb's alone), the input
/// they project, and the output they write.
enum Mla {
    Absorb(AbsorbFn),
    Value(ValueFn),
}

fn mla_on(m: &Mla, side: Side, ([x, w, sc], out): ([I; 3], O), (dims, nrow): MlaShape) -> i32 {
    let [h, qh, nope, vh, kvl, block] = dims;
    let ci = |v: usize| v as i32;
    let w = w.cast::<u8>();
    let kv = [h, nope, vh, kvl, block].map(ci);
    // SAFETY: live buffers sized for the shape, or the guard buffer under a refused guard.
    unsafe {
        match m {
            Mla::Absorb(c) => twin(
                side,
                || {
                    c(
                        x,
                        w,
                        sc,
                        kv[0],
                        ci(qh),
                        kv[1],
                        kv[2],
                        kv[3],
                        kv[4],
                        ci(nrow),
                        out,
                    )
                },
                || hip::launch_mla_absorb_fp8(x, w, sc, h, qh, nope, vh, kvl, block, nrow, out),
            ),
            Mla::Value(c) => twin(
                side,
                || c(x, w, sc, kv[0], kv[1], kv[2], kv[3], kv[4], ci(nrow), out),
                || hip::launch_mla_value_fp8(x, w, sc, h, nope, vh, kvl, block, nrow, out),
            ),
        }
    }
}

/// Which MLA shapes a row sweeps, and each shape's `[input, output]` float counts.
/// Which MLA shapes a row sweeps, and each shape's `[input, output]` float counts.
type RowPick = (fn(&MlaShape) -> bool, fn(MlaShape) -> [usize; 2]);

/// An MLA row: every `MLA_DIMS` shape `keep` admits × seed, then the guard cases, each asserted
/// to reach its code.
fn mla_row(t: &mut Tally, m: &Mla, (keep, lens): RowPick, guards: &[(MlaShape, i32)]) {
    let all: Vec<MlaShape> = MLA_DIMS.into_iter().filter(keep).collect();
    let shapes = Shapes {
        all: &all,
        hot: MLA_DIMS[0],
    };
    let make = |shape: MlaShape, seed: u64| {
        let [x_len, out_len] = lens(shape);
        (mla_ins(shape, x_len, seed), vec![Out::Poison(out_len)], ())
    };
    sweep(t, shapes, make, |side, x: &mut Cell, shape, _| {
        mla_on(m, side, ([x.i(0), x.i(1), x.i(2)], x.o(side, 0)), shape)
    });
    let (gi, go) = guard_ptrs();
    for &(shape, want) in guards {
        let rc = both(|side| mla_on(m, side, ([gi; 3], go), shape));
        t.guard_code(&format!("{shape:?}"), want, rc);
    }
}

pub fn mla_absorb_fp8(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_mla_absorb_fp8`'s C signature.
    let m = Mla::Absorb(unsafe { r.sym(c"rivoli_mla_absorb_fp8") });
    let lens = |([h, qh, _, _, kvl, _], nrow): MlaShape| [nrow * h * qh, nrow * h * kvl];
    mla_row(t, &m, (|_| true, lens), &MLA_GUARDS);
}

pub fn mla_value_fp8(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_mla_value_fp8`'s C signature.
    let m = Mla::Value(unsafe { r.sym(c"rivoli_mla_value_fp8") });
    let lens = |([h, _, _, vh, kvl, _], nrow): MlaShape| [nrow * h * kvl, nrow * h * vh];
    // No `kvl % 4` fallback here (that is 1002), so its scalar shape is `block < 4` only. It never
    // reads `qh`, so the absorb's `qh = 0` case is no refusal; `kvl % 4 != 0` is.
    // `kvl % 4 != 0` is its last refusal: after the tile and the row count.
    let ragged = [
        (([2, 8, 8, 8, 10, 4], 1), 1002),
        (([2, 8, 8, 8, 10, 6], 3), 1003),
        (([2, 8, 8, 8, 10, 4], 3), 1004),
    ];
    let guards: Vec<(MlaShape, i32)> = MLA_GUARDS
        .into_iter()
        .filter(|((d, _), _)| d[1] != 0)
        .chain(ragged)
        .collect();
    mla_row(t, &m, (|(d, _)| d[4] % 4 == 0, lens), &guards);
}

// ── act_quant_f8_prefix ─────────────────────────────────────────────────────────────────────

/// `[rows, row_stride, n, block]` and whether `dst` is `src` (the in-place form).
type PrefixShape = ([usize; 4], bool);

fn prefix_on(c: PrefixFn, side: Side, (src, dst): (I, O), dims: [usize; 4]) -> i32 {
    let [rows, stride, n, block] = dims;
    let ci = dims.map(|v| v as i32);
    // SAFETY: live buffers sized for the shape, or the guard buffer under a refused guard.
    unsafe {
        twin(
            side,
            || c(src, dst, ci[0], ci[1], ci[2], ci[3], null_mut()),
            || hip::launch_act_quant_f8_prefix(src, dst, rows, stride, n, block, null_mut()),
        )
    }
}

pub fn act_quant_f8_prefix(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_act_quant_f8_prefix`'s C signature.
    let c: PrefixFn = unsafe { r.sym(c"rivoli_act_quant_f8_prefix") };
    // The MoE/attention input in place, `q_lora` over a prefill tile, the KV quantizer's partial
    // prefix (`d - rd` of `d`, block 64), the grouped output in place, and the copy form.
    let groups = Shapes {
        all: &[
            ([1, 4096, 4096, 128], true),
            ([7, 1024, 1024, 128], true),
            ([3, 512, 448, 64], true),
            ([1, 8192, 8192, 128], true),
            ([5, 4096, 4096, 128], false),
        ],
        hot: ([1, 4096, 4096, 128], true),
    };
    let make = |([rows, stride, _, _], in_place): PrefixShape, seed: u64| {
        let x = draw(
            rows * stride,
            (seed, 1),
            [1.0, 1e-3, 300.0][(seed % 3) as usize],
            true,
        );
        if in_place {
            (vec![], vec![Out::InPlace(x)], in_place)
        } else {
            (vec![x], vec![Out::Poison(rows * stride)], in_place)
        }
    };
    sweep(
        t,
        groups,
        make,
        |side, x: &mut Cell, (dims, _), &in_place| {
            let o = x.o(side, 0);
            let src = if in_place { o.cast_const() } else { x.i(0) };
            prefix_on(c, side, (src, o), dims)
        },
    );
    let (gi, go) = guard_ptrs();
    // Each extent, `row_stride < n`, a strided copy, ragged groups, and the order across them.
    for (dims, want) in [
        ([0, 8, 8, 4], 1001),
        ([1, 8, 0, 4], 1001),
        ([1, 8, 8, 0], 1001),
        ([1, 4, 8, 4], 1001),
        ([1, 16, 8, 4], 1002),
        ([1, 8, 6, 4], 1004),
        ([1, 16, 6, 4], 1002),
        ([0, 16, 6, 4], 1001),
    ] {
        // The guard buffer is one address for input and output; the strided-copy case needs a
        // `src` that is NOT `dst`, and a refused guard reads neither.
        let src = if dims[1] == 16 {
            gi.wrapping_add(1)
        } else {
            go.cast_const()
        };
        let dst = go;
        let rc = both(|side| prefix_on(c, side, (src, dst), dims));
        t.guard_code(&format!("act_quant_f8_prefix {dims:?}"), want, rc);
    }
}

// ── the row kernels: rmsnorm_batch, qk_norm, rope_adjacent ──────────────────────────────────

fn norm_on(c: NormFn, side: Side, (x, w): (O, I), (rows, d, eps): (i64, i64, f32)) -> i32 {
    // SAFETY: live buffers sized for `rows · d`, or the guard buffer under a refused guard.
    unsafe {
        twin(
            side,
            || c(x, w, rows as i32, d as i32, eps, null_mut()),
            || hip::launch_rmsnorm_batch(x, w, rows as usize, d as usize, eps, null_mut()),
        )
    }
}

pub fn rmsnorm_batch(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_rmsnorm_batch`'s C signature.
    let c: NormFn = unsafe { r.sym(c"rivoli_rmsnorm_batch") };
    // `q_norm` over `q_lora`, `kv_norm` over `head_dim`, the hidden norm over a prefill tile.
    let rows = Shapes {
        all: &[(1usize, 1024usize), (1, 512), (7, 4096), (2, 7)],
        hot: (1, 1024),
    };
    let make = |(n, d): (usize, usize), seed: u64| {
        let x = draw(n * d, (seed, 1), 1.0, true);
        (
            vec![uniform(d, (seed, 2), 0.5..1.5)],
            vec![Out::InPlace(x)],
            (),
        )
    };
    sweep(t, rows, make, |side, x: &mut Cell, (n, d), _| {
        norm_on(c, side, (x.o(side, 0), x.i(0)), (n as i64, d as i64, 1e-6))
    });
    let (gi, go) = guard_ptrs();
    for (n, d) in [(0, 8), (2, 0), (0, 0)] {
        let rc = both(|side| norm_on(c, side, (go, gi), (n, d, 1e-6)));
        t.guard_code(&format!("rmsnorm_batch rows={n} d={d}"), 1001, rc);
    }
}

fn qk_on(c: QkFn, side: Side, q: O, dims: [i64; 2]) -> i32 {
    let ([rows, d], eps) = (dims.map(|v| v as usize), 1e-6);
    let [r32, d32] = dims.map(|v| v as i32);
    // SAFETY: a live buffer of `rows · d`, or the guard buffer under a refused guard.
    unsafe {
        twin(
            side,
            || c(q, r32, d32, eps, null_mut()),
            || hip::launch_qk_norm(q, rows, d, eps, null_mut()),
        )
    }
}

pub fn qk_norm(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_qk_norm`'s C signature.
    let c: QkFn = unsafe { r.sym(c"rivoli_qk_norm") };
    // `s · n_heads` rows of `head_dim`: decode, a 7-token prefill, and a ragged width.
    let heads = Shapes {
        all: &[(64usize, 512usize), (448, 512), (3, 7)],
        hot: (64, 512),
    };
    let make = |(n, d): (usize, usize), seed: u64| {
        (
            vec![],
            vec![Out::InPlace(draw(n * d, (seed, 1), 1.0, true))],
            (),
        )
    };
    sweep(t, heads, make, |side, x: &mut Cell, (n, d), _| {
        qk_on(c, side, x.o(side, 0), [n as i64, d as i64])
    });
    for (n, d) in [(0, 8), (2, 0), (0, 0)] {
        let rc = both(|side| qk_on(c, side, guard_ptrs().1, [n, d]));
        t.guard_code(&format!("qk_norm rows={n} d={d}"), 1001, rc);
    }
}

/// `rope_adjacent`'s `(rows, row_len, rd, rows_per_pos, inverse)`; `pos0` is drawn per seed.
type RopeShape = (usize, usize, usize, usize, bool);

fn rope_on(c: RopeFn, side: Side, (x, tbl): (O, I), (shape, pos0): (RopeShape, i64)) -> i32 {
    let (rows, row_len, rd, per, inverse) = shape;
    let ci = |v: usize| v as i32;
    let dims = [rows, row_len, rd].map(ci);
    // SAFETY: live buffers sized for the shape and position, or the guard buffer.
    unsafe {
        twin(
            side,
            || {
                c(
                    x,
                    tbl,
                    dims[0],
                    dims[1],
                    dims[2],
                    pos0 as i32,
                    ci(per),
                    i32::from(inverse),
                    null_mut(),
                )
            },
            || {
                hip::launch_rope_adjacent(
                    x,
                    tbl,
                    rows,
                    row_len,
                    rd,
                    pos0 as usize,
                    per,
                    inverse,
                    null_mut(),
                )
            },
        )
    }
}

pub fn rope_adjacent(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_rope_adjacent`'s C signature.
    let c: RopeFn = unsafe { r.sym(c"rivoli_rope_adjacent") };
    // q (64 heads per position), kv (one row per position), the output de-rotation (inverse), a
    // 7-token prefill of q, the block selector's 128-wide indexer rows, and an odd row length.
    let rows = Shapes {
        all: &[
            (64, 512, 64, 64, false),
            (1, 512, 64, 1, false),
            (64, 512, 64, 64, true),
            (448, 512, 64, 64, false),
            (64, 128, 64, 64, false),
            (6, 7, 4, 2, true),
        ],
        hot: (64, 512, 64, 64, false),
    };
    let make = |(n, len, rd, per, _): RopeShape, seed: u64| {
        let pos0 = [0, 1, 7, 4095, 65_535][(seed % 5) as usize];
        let tbl = uniform((pos0 + n / per) * rd, (seed, 2), -1.0..1.0);
        (
            vec![tbl],
            vec![Out::InPlace(draw(n * len, (seed, 1), 1.0, true))],
            pos0,
        )
    };
    sweep(t, rows, make, |side, x: &mut Cell, shape, &pos0| {
        rope_on(c, side, (x.o(side, 0), x.i(0)), (shape, pos0 as i64))
    });
    let (gi, go) = guard_ptrs();
    // Each extent, a negative position, `rd > row_len`, an odd `rd`, ragged position groups, and
    // the order across them.
    for (shape, pos0, want) in [
        ((0, 8, 4, 1, false), 0, 1001),
        ((4, 0, 4, 1, false), 0, 1001),
        ((4, 8, 0, 1, false), 0, 1001),
        ((4, 8, 4, 0, false), 0, 1001),
        ((4, 8, 4, 1, false), -1, 1001),
        ((4, 8, 10, 1, false), 0, 1002),
        ((4, 8, 3, 1, false), 0, 1005),
        ((4, 8, 4, 3, false), 0, 1004),
        ((4, 8, 11, 3, false), 0, 1002),
        ((4, 8, 3, 3, false), 0, 1005),
        ((4, 8, 11, 3, false), -1, 1001),
    ] {
        let rc = both(|side| rope_on(c, side, (go, gi), (shape, pos0)));
        t.guard_code(&format!("rope_adjacent {shape:?} pos0={pos0}"), want, rc);
    }
}

// ── gemv_fp8_bf16 ───────────────────────────────────────────────────────────────────────────

fn gemv_on(c: GemvFn, side: Side, ([x, w, sc], out): ([I; 3], O), dims: [usize; 5]) -> i32 {
    let [m, n, k, block, groups] = dims;
    let ci = dims.map(|v| v as i32);
    let w = w.cast::<u8>();
    // SAFETY: live buffers sized for the shape, or the guard buffer under a refused guard.
    unsafe {
        twin(
            side,
            || c(x, w, sc, ci[0], ci[1], ci[2], ci[3], ci[4], out, null_mut()),
            || hip::launch_gemv_fp8_bf16(x, w, sc, m, n, k, block, groups, out, null_mut()),
        )
    }
}

pub fn gemv_fp8_bf16(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_gemv_fp8_bf16`'s C signature.
    let c: GemvFn = unsafe { r.sym(c"rivoli_gemv_fp8_bf16") };
    // `[m, n_out, k, block, groups]`: V4's grouped `wo_a` (8 groups of 4096; a 1024-row slice of
    // its 8192 outputs — every row runs the same code), the engine's ungrouped form over a prefill
    // tile, and a ragged one: 2-tiles, so the per-column tail runs alone — a dword load wants
    // 4-aligned rows, which `k = 101` would break, in the C and the Rust alike — over three groups.
    let outputs = Shapes {
        all: &[
            [1usize, 1024, 4096, 128, 8],
            [3, 64, 512, 128, 1],
            [2, 24, 101, 2, 3],
        ],
        hot: [1, 1024, 4096, 128, 8],
    };
    let make = |[m, n, k, block, groups]: [usize; 5], seed: u64| {
        let ins = vec![
            draw(m * groups * k, (seed, 1), 1.0, true),
            fp8_words(n * k, (seed, 2)),
            uniform(n.div_ceil(block) * k.div_ceil(block), (seed, 3), 1e-3..1e-1),
        ];
        (ins, vec![Out::Poison(m * n)], ())
    };
    sweep(t, outputs, make, |side, x: &mut Cell, dims, _| {
        gemv_on(c, side, ([x.i(0), x.i(1), x.i(2)], x.o(side, 0)), dims)
    });
    let (gi, go) = guard_ptrs();
    // Each extent, a non-power-of-two tile, ragged groups, and the order across them.
    for (dims, want) in [
        ([0, 8, 8, 4, 1], 1001),
        ([1, 0, 8, 4, 1], 1001),
        ([1, 8, 0, 4, 1], 1001),
        ([1, 8, 8, 0, 1], 1001),
        ([1, 8, 8, 4, 0], 1001),
        ([1, 8, 8, 6, 1], 1003),
        ([1, 8, 8, 4, 3], 1004),
        ([1, 8, 8, 6, 3], 1003),
        ([0, 8, 8, 6, 3], 1001),
    ] {
        let rc = both(|side| gemv_on(c, side, ([gi; 3], go), dims));
        t.guard_code(&format!("gemv_fp8_bf16 {dims:?}"), want, rc);
    }
}
