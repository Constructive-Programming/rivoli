//! Rows for `kernels/activation.hip`'s nine launchers. Shapes are the engine's (GLM's 6144
//! hidden and 12288/18432 FFN widths, V4's 7168, K3/Glimmer's 2048-4096, Qwen's 20-wide
//! heads) plus ragged and degenerate ones (1, 33, 257, odd tails) that land a block edge
//! mid-wave. One adapter per launcher calls both twins with one argument set, as in
//! `residual.rs`.

use super::device::{Cell, I, O, Out, Ref, S, Shapes, Side, Tally, both, code, guard_ptrs, sweep};
use super::inputs::{draw, rope_at, uniform};
use rivoli_backend::hip;
use std::ffi::CStr;
use std::ptr::null_mut;

type GateFn = unsafe extern "C" fn(I, I, i32, O, S) -> i32;
type ClampFn = unsafe extern "C" fn(I, I, i32, f32, O, S) -> i32;
type SituFn = unsafe extern "C" fn(I, I, i32, f32, f32, O, S) -> i32;
type NormFn = unsafe extern "C" fn(I, I, i32, f32, O) -> i32;
type CenteredFn = unsafe extern "C" fn(I, I, i32, f32, O, S) -> i32;
type WeightlessFn = unsafe extern "C" fn(O, i32, i32, f32, f32, S) -> i32;
type QuantFn = unsafe extern "C" fn(O, i32, i32, S) -> i32;
type RopeFn = unsafe extern "C" fn(O, i32, i32, i32, i32, f64) -> i32;
/// The two Rust rope launchers' shared signature.
type RopeLauncher = unsafe fn(O, usize, usize, usize, usize, f64) -> anyhow::Result<()>;

const GATE_SHAPES: [usize; 8] = [1, 33, 257, 1536, 2048, 6144, 12288, 18439];
const ROW_SHAPES: [usize; 6] = [1, 7, 1000, 4096, 6144, 7168];
/// The four values every positive-finite guard refuses: NaN, 0, negative, +inf.
const NOT_POSITIVE_FINITE: [f32; 4] = [f32::NAN, 0.0, -1.0, f32::INFINITY];

/// Each row's shapes and the hot one it is timed at.
const RMSNORM_SWEEP: Shapes<usize> = Shapes {
    all: &ROW_SHAPES,
    hot: 6144,
};
const CENTERED_SWEEP: Shapes<usize> = Shapes {
    all: &ROW_SHAPES,
    hot: 4096,
};
const WEIGHTLESS_SWEEP: Shapes<(usize, usize)> = Shapes {
    all: &[(1, 64), (8, 128), (32, 128), (16, 20), (4, 256), (3, 1000)],
    hot: (32, 128),
};
const ACT_QUANT_SWEEP: Shapes<(usize, usize)> = Shapes {
    all: &[(1, 128), (1, 2048), (1, 7168), (4, 4096), (3, 384)],
    hot: (1, 7168),
};
const ROPE_SWEEP: Shapes<RopeShape> = Shapes {
    all: &ROPE_SHAPES,
    hot: (32, 128, 128),
};

/// A gated combine's operands: `(g, u, h, n)`. Guard cases pass nulls with `n` chosen to
/// pass or fail the extent check.
type GateIo = (I, I, O, usize);
fn no_gate(n: usize) -> GateIo {
    let (i, o) = guard_ptrs();
    (i, i, o, n)
}

/// The gated combines' cell: `g` (scale 4, so `silu` sees its saturating range), `u`, a NaN
/// at slot 14 — elementwise kernels admit one and must propagate it identically — and one
/// output.
fn gate_cell(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Out>) {
    let mut g = draw(n, (seed, 1), 4.0, true);
    if let Some(s) = g.get_mut(14) {
        *s = f32::NAN;
    }
    (vec![g, draw(n, (seed, 2), 1.0, true)], vec![Out::Poison(n)])
}

fn gate_io(x: &mut Cell, side: Side, n: usize) -> GateIo {
    (x.i(0), x.i(1), x.o(side, 0), n)
}

/// The gated combines' sweep: [`GATE_SHAPES`] × seeds over [`gate_cell`], the per-seed
/// scalar operand(s) from `extra`, timed at `time_at`.
fn gate_sweep<P>(
    t: &mut Tally,
    time_at: usize,
    extra: impl Fn(u64) -> P,
    launch: impl Fn(Side, GateIo, &P) -> i32,
) {
    let make = |n, seed| {
        let (i, o) = gate_cell(n, seed);
        (i, o, extra(seed))
    };
    let shapes = Shapes {
        all: &GATE_SHAPES,
        hot: time_at,
    };
    sweep(t, shapes, make, |side, x: &mut Cell, n, p| {
        launch(side, gate_io(x, side, n), p)
    });
}

fn swiglu_on(c: GateFn, side: Side, (g, u, h, n): GateIo) -> i32 {
    // SAFETY: n-element live buffers, or nulls under a refused guard.
    match side {
        Side::Hip => unsafe { c(g, u, n as i32, h, null_mut()) },
        Side::Rust => code(unsafe { hip::launch_swiglu(g, u, n, h, null_mut()) }),
    }
}

pub fn swiglu(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_swiglu`'s C signature.
    let c: GateFn = unsafe { r.sym(c"rivoli_swiglu") };
    gate_sweep(t, 6144, |_| (), |side, io, _| swiglu_on(c, side, io));
    t.guard("swiglu n=0", both(|side| swiglu_on(c, side, no_gate(0))));
}

fn clamped_on(c: ClampFn, side: Side, (g, u, h, n): GateIo, limit: f32) -> i32 {
    // SAFETY: n-element live buffers, or nulls under a refused guard.
    match side {
        Side::Hip => unsafe { c(g, u, n as i32, limit, h, null_mut()) },
        Side::Rust => {
            code(unsafe { hip::launch_swiglu_clamped_bf16(g, u, n, limit, h, null_mut()) })
        }
    }
}

pub fn swiglu_clamped_bf16(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_swiglu_clamped_bf16`'s C signature.
    let c: ClampFn = unsafe { r.sym(c"rivoli_swiglu_clamped_bf16") };
    let limit = |seed| uniform(1, (seed, 3), 0.5..12.0)[0];
    gate_sweep(t, 2048, limit, |side, io, &l| clamped_on(c, side, io, l));
    // Every guard failing at once comes LAST in each table: it pins guard ORDER (the
    // extent check before the operand check), which single-failure cases cannot see.
    let cases = NOT_POSITIVE_FINITE
        .map(|l| (8, l))
        .into_iter()
        .chain([(0, 7.0), (0, f32::NAN)]);
    for (n, limit) in cases {
        let refusal = both(|side| clamped_on(c, side, no_gate(n), limit));
        t.guard(&format!("clamped n={n} limit={limit}"), refusal);
    }
}

fn situ_on(c: SituFn, side: Side, (g, u, h, n): GateIo, (b1, b2): (f32, f32)) -> i32 {
    // SAFETY: n-element live buffers, or nulls under a refused guard.
    match side {
        Side::Hip => unsafe { c(g, u, n as i32, b1, b2, h, null_mut()) },
        Side::Rust => code(unsafe { hip::launch_situ_glu_f32(g, u, n, b1, b2, h, null_mut()) }),
    }
}

pub fn situ_glu_f32(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_situ_glu_f32`'s C signature.
    let c: SituFn = unsafe { r.sym(c"rivoli_situ_glu_f32") };
    // K3's betas (4, 25) on odd seeds; seeded positives elsewhere, so the tanh argument
    // range moves.
    let betas = |seed: u64| match uniform(2, (seed, 3), 0.25..40.0)[..] {
        [b1, b2] if seed.is_multiple_of(2) => (b1, b2),
        _ => (4.0, 25.0),
    };
    gate_sweep(t, 2048, betas, |side, io, &b| situ_on(c, side, io, b));
    let cases = [
        (0, (4.0, 25.0)),
        (8, (f32::NAN, 25.0)),
        (8, (4.0, f32::INFINITY)),
        (8, (-4.0, 0.0)),
        (0, (f32::NAN, f32::INFINITY)),
    ];
    for (n, betas) in cases {
        let refusal = both(|side| situ_on(c, side, no_gate(n), betas));
        t.guard(&format!("situ n={n} betas={betas:?}"), refusal);
    }
}

/// A norm row: `x`, `w`, one output, and an eps from the tree's two (1e-5 GLM, 1e-6 V4/K3).
fn norm_cell(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Out>, f32) {
    let ins = vec![draw(n, (seed, 1), 1.0, true), draw(n, (seed, 2), 0.1, true)];
    (
        ins,
        vec![Out::Poison(n)],
        if seed.is_multiple_of(2) { 1e-5 } else { 1e-6 },
    )
}

/// A single-row norm's operands: `(x, w, y, n, eps)`.
type NormIo = (I, I, O, usize, f32);

fn norm_io(x: &mut Cell, side: Side, n: usize, eps: f32) -> NormIo {
    (x.i(0), x.i(1), x.o(side, 0), n, eps)
}

fn rmsnorm_on(c: NormFn, side: Side, (x, w, y, n, eps): NormIo) -> i32 {
    // SAFETY: n-element live buffers, or nulls under a refused guard.
    match side {
        Side::Hip => unsafe { c(x, w, n as i32, eps, y) },
        Side::Rust => code(unsafe { hip::launch_rmsnorm_single(x, w, n, eps, y) }),
    }
}

pub fn rmsnorm_single(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_rmsnorm_single`'s C signature.
    let c: NormFn = unsafe { r.sym(c"rivoli_rmsnorm_single") };
    sweep(
        t,
        RMSNORM_SWEEP,
        norm_cell,
        |side, x: &mut Cell, n, &eps| rmsnorm_on(c, side, norm_io(x, side, n, eps)),
    );
    let (gi, go) = guard_ptrs();
    let none = (gi, gi, go, 0, 1e-5);
    t.guard("rmsnorm_single n=0", both(|side| rmsnorm_on(c, side, none)));
}

fn centered_on(c: CenteredFn, side: Side, (x, w, y, n, eps): NormIo) -> i32 {
    // SAFETY: n-element live buffers, or nulls under a refused guard.
    match side {
        Side::Hip => unsafe { c(x, w, n as i32, eps, y, null_mut()) },
        Side::Rust => {
            code(unsafe { hip::launch_rmsnorm_centered_single(x, w, n, eps, y, null_mut()) })
        }
    }
}

pub fn rmsnorm_centered_single(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_rmsnorm_centered_single`'s C signature.
    let c: CenteredFn = unsafe { r.sym(c"rivoli_rmsnorm_centered_single") };
    sweep(
        t,
        CENTERED_SWEEP,
        norm_cell,
        |side, x: &mut Cell, n, &eps| centered_on(c, side, norm_io(x, side, n, eps)),
    );
    let cases = NOT_POSITIVE_FINITE
        .map(|e| (8, e))
        .into_iter()
        .chain([(0, 1e-5), (0, f32::NAN)]);
    for (n, eps) in cases {
        let (gi, go) = guard_ptrs();
        let refusal = both(|side| centered_on(c, side, (gi, gi, go, n, eps)));
        t.guard(&format!("centered n={n} eps={eps}"), refusal);
    }
}

/// `(rows, d, eps, scale)` for the weightless norm.
type WlArgs = (usize, usize, f32, f32);

fn weightless_on(c: WeightlessFn, side: Side, v: O, (rows, d, eps, scale): WlArgs) -> i32 {
    // SAFETY: rows·d live floats, or null under a refused guard.
    match side {
        Side::Hip => unsafe { c(v, rows as i32, d as i32, eps, scale, null_mut()) },
        Side::Rust => code(unsafe {
            hip::launch_rmsnorm_weightless_batch(v, rows, d, eps, scale, null_mut())
        }),
    }
}

pub fn rmsnorm_weightless_batch(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_rmsnorm_weightless_batch`'s C signature.
    let c: WeightlessFn = unsafe { r.sym(c"rivoli_rmsnorm_weightless_batch") };
    // Glimmer's Q scale (3.87) and K's 1.0, alternating.
    let make = |(rows, d): (usize, usize), seed: u64| {
        let x = draw(rows * d, (seed, 1), 1.0, true);
        (
            vec![],
            vec![Out::InPlace(x)],
            if seed.is_multiple_of(2) { 3.87f32 } else { 1.0 },
        )
    };
    sweep(
        t,
        WEIGHTLESS_SWEEP,
        make,
        |side, x: &mut Cell, (rows, d), &scale| {
            weightless_on(c, side, x.o(side, 0), (rows, d, 1e-6, scale))
        },
    );
    for args in [
        (0, 8, 1e-6, 1.0),
        (1, 0, 1e-6, 1.0),
        (1, 8, 0.0, 1.0),
        (1, 8, 1e-6, f32::NAN),
        (1, 8, 0.0, f32::NAN),
        (0, 0, 0.0, f32::NAN),
    ] {
        let refusal = both(|side| weightless_on(c, side, guard_ptrs().1, args));
        t.guard(&format!("weightless {args:?}"), refusal);
    }
}

fn act_quant_on(c: QuantFn, side: Side, v: O, (rows, len): (usize, usize)) -> i32 {
    // SAFETY: rows·len live floats, or null under a refused guard.
    match side {
        Side::Hip => unsafe { c(v, rows as i32, len as i32, null_mut()) },
        Side::Rust => code(unsafe { hip::launch_act_quant_f8(v, rows, len, null_mut()) }),
    }
}

pub fn act_quant_f8(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_act_quant_f8`'s C signature.
    let c: QuantFn = unsafe { r.sym(c"rivoli_act_quant_f8") };
    // A NaN at slot 20: `act_quant_roundtrip` is written with comparisons precisely so a
    // NaN propagates, and claims bit identity there.
    let make = |(rows, len): (usize, usize), seed| {
        let mut v = draw(rows * len, (seed, 1), 4.0, true);
        if let Some(s) = v.get_mut(20) {
            *s = f32::NAN;
        }
        (vec![], vec![Out::InPlace(v)], ())
    };
    sweep(t, ACT_QUANT_SWEEP, make, |side, x: &mut Cell, shape, _| {
        act_quant_on(c, side, x.o(side, 0), shape)
    });
    for shape in [(0, 128), (1, 0), (1, 100), (0, 100)] {
        let refusal = both(|side| act_quant_on(c, side, guard_ptrs().1, shape));
        t.guard(&format!("act_quant {shape:?}"), refusal);
    }
}

/// `(count, stride, seg)`: one pair, GLM/V4's 64-wide rope dim inside a 192 stride, full
/// 128-wide heads at 8 and 32 heads, and a 20-wide one.
type RopeShape = (usize, usize, usize);
const ROPE_SHAPES: [RopeShape; 5] = [
    (1, 2, 2),
    (64, 192, 64),
    (8, 128, 128),
    (32, 128, 128),
    (6, 20, 20),
];

/// A rope cell: the segments, rotated in place to [`rope_at`]'s position.
fn rope_cell((count, stride, _): RopeShape, seed: u64) -> (Vec<Vec<f32>>, Vec<Out>, (usize, f64)) {
    let v = draw(count * stride, (seed, 1), 1.0, true);
    (vec![], vec![Out::InPlace(v)], rope_at(seed))
}

/// One rope launch's buffer, shape, and `(pos, theta)`.
type RopeCall = (O, RopeShape, (usize, f64));

/// One rope launch on `side`; `pair` is the C and Rust launcher of one convention.
fn rope_on(pair: (RopeFn, RopeLauncher), side: Side, call: RopeCall) -> i32 {
    let ((c, rust), (v, (count, stride, seg), (pos, theta))) = (pair, call);
    // SAFETY: count·stride live floats, or null under a refused guard.
    match side {
        Side::Hip => unsafe {
            c(
                v,
                count as i32,
                stride as i32,
                seg as i32,
                pos as i32,
                theta,
            )
        },
        Side::Rust => code(unsafe { rust(v, count, stride, seg, pos, theta) }),
    }
}

/// Both rope rows: same cells, same guards, a different symbol and launcher.
fn rope_row(r: &Ref, t: &mut Tally, sym: &CStr, rust: RopeLauncher) {
    // SAFETY: the rope launchers' shared C signature.
    let pair = (unsafe { r.sym::<RopeFn>(sym) }, rust);
    sweep(
        t,
        ROPE_SWEEP,
        rope_cell,
        |side, x: &mut Cell, shape, &at| rope_on(pair, side, (x.o(side, 0), shape, at)),
    );
    // Zero segments, an odd seg, a block past 1024 threads, and overlapping segments.
    for shape in [
        (0, 2, 2),
        (1, 3, 3),
        (1, 2050, 2050),
        (2, 63, 64),
        (0, 63, 2051),
    ] {
        let refusal = both(|side| rope_on(pair, side, (guard_ptrs().1, shape, (0, 1e4))));
        t.guard(&format!("rope {shape:?}"), refusal);
    }
}

pub fn rope_interleave(r: &Ref, t: &mut Tally) {
    rope_row(r, t, c"rivoli_rope_interleave", hip::launch_rope_interleave);
}

pub fn rope_split_half(r: &Ref, t: &mut Tally) {
    rope_row(r, t, c"rivoli_rope_split_half", hip::launch_rope_split_half);
}
