//! Rows for `kernels/moe.hip`'s four launchers (and the expert-range machinery `moe_f4.rs`
//! shares): GLM's VQ-int3 and int4 ranges at its real 6144 × 2048 experts, one and two token
//! rows, plus small and ragged widths; and both fixed-point drains.
//!
//! **Each range cell is a real descriptor array.** The weights and scales live in two arena
//! buffers and every expert's six pointers address them at a per-expert shift — the int4/fp4
//! weight shift alternates 4-byte alignment, so odd experts take the dots' scalar tail on
//! every row, as a misplaced slot would. `h` and `acc` start as the same data on both sides
//! (`acc` compared strictly: it is integers), so a write outside the range or a missed
//! accumulation is a mismatch. Scales carry NaN/inf/−0/subnormal codes: in the down
//! projection on every seed (each poisons only its own output rows), everywhere on the nasty
//! seeds; routing weights carry exact `0.0` and `-0.0` (both skipped), a 0.3 or nasty-seed NaN,
//! at slots that rotate with the seed so every expert of a range is live on some seeds. The
//! inputs themselves are `ranges.rs`'s, shared with the CPU plant models.

use super::device::{Cell, I, O, Out, Ref, S, Side, Tally, accepted, both, code, guard_ptrs};
use super::inputs::{SEEDS, draw, uniform};
use super::ranges::{Format, Range, cell_inputs, edge_seed, fixed_words, no_extras, vecs};
use rivoli_backend::hip::{self, ExpertDesc};
use std::ffi::c_void;
use std::ptr::null_mut;

pub type U64 = *mut u64;
// One line per C signature: the C launchers' parameter lists, read side by side.
#[rustfmt::skip]
type RangeVqFn = unsafe extern "C" fn(I, i32, i32, i32, i32, *const c_void, *const u16, *const u16, *const u16, I, O, U64, i32, S) -> i32;
#[rustfmt::skip]
type RangeI4Fn = unsafe extern "C" fn(I, i32, i32, i32, i32, *const c_void, I, O, U64, i32, S) -> i32;
type DrainFn = unsafe extern "C" fn(O, U64, i32, i32, f32, S) -> i32;
type DrainToFn = unsafe extern "C" fn(O, U64, i32, i32, S) -> i32;

/// The cell's fixed inputs, in [`Cell::i`] order; the descriptor array is pushed after them.
const X: usize = 0;
const WEXPERT: usize = 1;
const WEIGHTS: usize = 2;
const SCALES: usize = 3;
/// `h` (in place, as f32) and `acc` (in place, strict) for a range cell.
fn range_outputs((hidden, inter, e_start, e_count, nrow): Range, seed: u64) -> Vec<Out> {
    let acc = fixed_words(nrow * hidden, (seed, 6));
    vec![
        Out::InPlace(draw(
            (e_start + e_count) * nrow * inter,
            (seed, 5),
            1.0,
            false,
        )),
        Out::Words(acc),
    ]
}

/// One range launch's buffers on one side: `x`, `wexpert`, `h`, `acc`, the descriptor array
/// and up to three format extras (VQ's codebooks).
#[derive(Clone, Copy)]
pub struct RangeBufs {
    pub x: I,
    pub w: I,
    pub h: O,
    pub acc: U64,
    pub descs: *const c_void,
    pub extra: [*const u16; 3],
}

/// A range launcher's `(hidden, inter, e_start, e_count, nrow)` with the start as the C's
/// `int`, so a guard case can pass `-1` (the Rust wrapper gets `usize::MAX`, which its
/// `as i32` narrows back).
pub type Args = (usize, usize, i32, usize, usize);

/// A whole range row: every shape × seed cell (descriptors pushed after the arenas they
/// address; `value(range, seed)` the format's scalar operands), then each guard case. The
/// twins are `(hip, rust)`: `hip(bufs, ints, p)` calls the C launcher with its `int`s
/// `[hidden, inter, e_start, e_count, nrow]`, `rust(bufs, sizes, p)` the wrapper with its
/// `usize`s (the start through [`rust_start`]).
pub fn range_row<P: Copy + std::fmt::Debug>(
    t: &mut Tally,
    fmt: &Format,
    (value, guards): (impl Fn(Range, u64) -> P, &[(Args, P)]),
    (hip, rust): (
        impl Fn(RangeBufs, [i32; 5], P) -> i32,
        impl Fn(RangeBufs, [usize; 5], P) -> anyhow::Result<()>,
    ),
) {
    let call = |side, b, (hi, it, es, ec, nr): Args, p| match side {
        Side::Hip => hip(b, [hi as i32, it as i32, es, ec as i32, nr as i32], p),
        Side::Rust => code(rust(b, [hi, it, rust_start(es), ec, nr], p)),
    };
    let hot = &fmt.shapes[..fmt.timed];
    for &range in fmt.shapes {
        for seed in 1..=SEEDS {
            let what = format!("{range:?} seed {seed}");
            let (lay, mut ins) = cell_inputs(fmt, range, seed);
            let first_extra = ins.len();
            ins.extend((fmt.extras)(seed));
            let (hi, it, es, ec, nr) = range;
            let (args, p) = ((hi, it, es as i32, ec, nr), value(range, seed));
            let go = |side, c: &mut Cell, d: usize| {
                let extra = std::array::from_fn(|k| match ins.get(first_extra + k) {
                    Some(_) => c.i(first_extra + k).cast(),
                    None => std::ptr::null(),
                });
                let bufs = RangeBufs {
                    x: c.i(X),
                    w: c.i(WEXPERT),
                    h: c.o(side, 0),
                    acc: c.o(side, 1).cast(),
                    descs: c.i(d).cast(),
                    extra,
                };
                accepted(call(side, bufs, args, p), side, &what)
            };
            let cell = || {
                let mut c = Cell::from_bytes(&ins, range_outputs(range, seed));
                let bases = (c.i(WEIGHTS) as usize, c.i(SCALES) as usize);
                let d = c.push(&lay.descs(bases, es + ec));
                (c, d)
            };
            if seed == 1 && hot.contains(&range) {
                let (mut c, d) = cell();
                t.time(&format!("{range:?}"), &mut c, |side, c| go(side, c, d));
            }
            let (c, d) = cell();
            t.run(&what, c, |side, c| go(side, c, d));
        }
    }
    let (gi, go) = guard_io();
    let bufs = RangeBufs {
        x: gi,
        w: gi,
        h: go,
        acc: go.cast(),
        descs: guard_descs().cast(),
        extra: [gi.cast(); 3],
    };
    for &(args, p) in guards {
        let refusal = both(|side| call(side, bufs, args, p));
        t.guard(&format!("{} {args:?} {p:?}", fmt.name), refusal);
    }
}

/// The GLM pair's refusals, `(args, ())`: each extent, a negative start, the two
/// out-of-set row counts, then two at once (order).
const GLM_GUARDS: [(Args, ()); 7] = [
    ((0, 64, 0, 1, 1), ()),
    ((64, 0, 0, 1, 1), ()),
    ((64, 64, 0, 0, 1), ()),
    ((64, 64, -1, 1, 1), ()),
    ((64, 64, 0, 1, 0), ()),
    ((64, 64, 0, 1, 3), ()),
    ((0, 64, 0, 1, 3), ()),
];

/// The descriptor array every guard case passes: nine descriptors whose 54 pointers all
/// address the guard buffer, handed out from the SECOND so `descs[-1]` is in bounds too. A
/// regressed Rust guard that accepted a case then launches on live memory and reports rc 0
/// against the reference's code — a clean RED, not a fault on the shared GPU (the guard
/// buffer's own argument, `device::guard_ptrs`). Allocated once and leaked.
fn guard_descs() -> *const ExpertDesc {
    static ADDR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let a = *ADDR.get_or_init(|| {
        let g = guard_io().0 as u64;
        let bytes: Vec<u8> = (0..9 * 6).flat_map(|_| g.to_le_bytes()).collect();
        let b = Box::leak(Box::new(super::common::dev(&bytes)));
        b.ptr() as usize + size_of::<ExpertDesc>()
    });
    a as *const ExpertDesc
}

/// The guard buffer, 4096 floats in: a regressed guard that accepted `e_start = -1` would index
/// `h`, `wexpert` and `x` up to `2 · 64` floats BEFORE the pointer, still inside the live buffer.
fn guard_io() -> (I, O) {
    let (i, o) = guard_ptrs();
    (i.wrapping_add(4096), o.wrapping_add(4096))
}

/// A guard row's `usize` start: `-1` as `usize::MAX`, which the wrapper's `as i32` narrows
/// back to `-1`.
fn rust_start(e_start: i32) -> usize {
    e_start as isize as usize
}

/// `f(x, hidden, inter, e_start, e_count, mid…, h, acc, nrow, null)` — every range
/// launcher's argument order, C and Rust alike, with the format's own operands (descriptors,
/// extras, routing weights, `n_desc`, parameters) in the middle; spelled once rather than in
/// every twin's call.
macro_rules! range_call {
    ($f:expr, $b:expr, $i:expr; $($mid:expr),*) => {{
        let ([hi, it, es, ec, nr], b) = ($i, $b);
        $f(b.x, hi, it, es, ec, $($mid,)* b.h, b.acc, nr, std::ptr::null_mut())
    }};
}
pub(super) use range_call;

/// VQ rows: `(i_dim / 4) · 12 / 8` index bytes and `i_dim / 64` bf16 scales.
fn vq_row(i_dim: usize) -> (usize, usize) {
    (i_dim / 4 * 12 / 8, i_dim / 64 * 2)
}

fn bf16(v: f32) -> Vec<u8> {
    ((v.to_bits() >> 16) as u16).to_le_bytes().to_vec()
}

/// bf16 NaN, ±inf, −0 and the smallest subnormal.
fn bf16_edges() -> Vec<Vec<u8>> {
    vecs([0x7fc1u16, 0x7f80, 0xff80, 0x8000, 0x0001].map(u16::to_le_bytes))
}

/// The three fp16 codebooks `[gate, up, down]`, 4096 × 4 each; fp16 inf and NaN entries on the
/// [`edge_seed`]s.
fn codebooks(seed: u64) -> Vec<Vec<u8>> {
    (0..3)
        .map(|p| {
            let mut v: Vec<u16> = draw(4096 * 4, (seed, 10 + p), 1.0, false)
                .iter()
                .map(|&x| rivoli_core::num::f32_to_f16(x))
                .collect();
            if edge_seed(seed) {
                v[5] = 0x7c00;
                v[9] = 0x7e01;
            }
            v.iter().flat_map(|c| c.to_le_bytes()).collect()
        })
        .collect()
}

const VQ: Format = Format {
    name: "moe_expert_range",
    row: vq_row,
    scale_width: 2,
    scale: bf16,
    edges: bf16_edges,
    extras: codebooks,
    shapes: &[
        (6144, 2048, 3, 1, 1),
        (6144, 2048, 3, 1, 2),
        (192, 64, 0, 3, 1),
        (192, 64, 1, 2, 2),
        (1024, 320, 5, 2, 1),
        (64, 64, 0, 1, 2),
    ],
    timed: 2,
};

/// A GLM range's C launcher: VQ-int3 (with its three codebooks) or int4. One row function over
/// both rather than two: the two would differ only in these calls (jscpd and CodeScene both
/// measured them as clones).
#[derive(Clone, Copy)]
enum Glm {
    Vq(RangeVqFn),
    I4(RangeI4Fn),
}

/// Both GLM range rows: the format's cells and the shared refusals, each twin's call taking
/// the format's middle operands.
fn glm_row(t: &mut Tally, c: Glm) {
    let fmt = match c {
        Glm::Vq(_) => &VQ,
        Glm::I4(_) => &I4,
    };
    let hip = |b: RangeBufs, i, _| {
        let [g, u, dn] = b.extra;
        // SAFETY: live buffers sized for the range, or the guard buffers under a refused guard.
        unsafe {
            match c {
                Glm::Vq(c) => range_call!(c, b, i; b.descs, g, u, dn, b.w),
                Glm::I4(c) => range_call!(c, b, i; b.descs, b.w),
            }
        }
    };
    let rust = |b: RangeBufs, i, _| {
        let ([g, u, dn], d) = (b.extra, b.descs.cast());
        // SAFETY: as above.
        unsafe {
            match c {
                Glm::Vq(_) => range_call!(hip::launch_moe_expert_range, b, i; d, g, u, dn, b.w),
                Glm::I4(_) => range_call!(hip::launch_moe_expert_range_i4, b, i; d, b.w),
            }
        }
    };
    range_row(t, fmt, (|_, _| (), &GLM_GUARDS), (hip, rust));
}

pub fn moe_expert_range(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_moe_expert_range`'s C signature.
    glm_row(t, Glm::Vq(unsafe { r.sym(c"rivoli_moe_expert_range") }));
}

/// int4 rows: `(i_dim + 1) / 2` packed bytes and `ceil(i_dim / 128)` f32 scales.
fn i4_row(i_dim: usize) -> (usize, usize) {
    (i_dim.div_ceil(2), i_dim.div_ceil(128) * 4)
}

fn f32_code(v: f32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// f32 NaN, ±inf, −0 and the smallest subnormal.
fn f32_edges() -> Vec<Vec<u8>> {
    let codes = [
        0x7fc0_0001u32,
        0x7f80_0000,
        0xff80_0000,
        0x8000_0000,
        0x0000_0001,
    ];
    vecs(codes.map(u32::to_le_bytes))
}

const I4: Format = Format {
    name: "moe_expert_range_i4",
    row: i4_row,
    scale_width: 4,
    scale: f32_code,
    edges: f32_edges,
    extras: no_extras,
    shapes: &[
        (6144, 2048, 3, 1, 1),
        (6144, 2048, 3, 1, 2),
        (1028, 132, 0, 3, 1),
        (1028, 132, 1, 2, 2),
        (1000, 136, 2, 1, 1),
        (256, 128, 0, 2, 2),
    ],
    timed: 2,
};

pub fn moe_expert_range_i4(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_moe_expert_range_i4`'s C signature.
    glm_row(t, Glm::I4(unsafe { r.sym(c"rivoli_moe_expert_range_i4") }));
}

/// `(n, rows)`: GLM's two-row 2 × 6144 accumulator, V4's 4096 and K3's 3584 latent, and
/// degenerate widths.
const DRAIN_SHAPES: [(usize, usize); 5] = [(12288, 2), (4096, 2), (3584, 2), (1, 1), (257, 3)];

/// `rows · n` accumulator words: [`fixed_words`] with the extremes a 16-term sum can reach
/// (±2^58, `MOE_ACC_MAX`'s bound) and zero planted first.
fn acc_words(len: usize, seed: u64) -> Vec<u8> {
    let mut w = fixed_words(len, (seed, 2));
    for (slot, e) in w
        .chunks_exact_mut(8)
        .zip([1i64 << 58, -(1i64 << 58), 0, -1, 1])
    {
        slot.copy_from_slice(&e.to_le_bytes());
    }
    w
}

/// A drain's C launcher: the residual add (`gain`) or the separate-buffer assign.
#[derive(Clone, Copy)]
enum Drain {
    Add(DrainFn),
    To(DrainToFn),
}

/// One drain launch on `side`: `out` and `acc`, `(n, rows, gain)` (the assign takes no gain).
fn drain_on(
    c: Drain,
    side: Side,
    (out, acc): (O, U64),
    (n, rows, gain): (usize, usize, f32),
) -> i32 {
    let (ni, ri) = (n as i32, rows as i32);
    // SAFETY: live buffers sized for `(n, rows)`, or the guard buffer under a refused guard.
    unsafe {
        match (c, side) {
            (Drain::Add(c), Side::Hip) => c(out, acc, ni, ri, gain, null_mut()),
            (Drain::To(c), Side::Hip) => c(out, acc, ni, ri, null_mut()),
            (Drain::Add(_), Side::Rust) => code(hip::launch_moe_acc_drain(
                out,
                acc,
                n,
                rows,
                gain,
                null_mut(),
            )),
            (Drain::To(_), Side::Rust) => {
                code(hip::launch_moe_acc_drain_to(out, acc, n, rows, null_mut()))
            }
        }
    }
}

/// Both drains' row: `out` starts as an f32 residual for the add and POISON for the assign
/// (which must write every element), the accumulator as strict words; then the three
/// refusals.
fn drain_row(t: &mut Tally, (c, name): (Drain, &str)) {
    let shapes = super::device::Shapes {
        all: &DRAIN_SHAPES,
        hot: DRAIN_SHAPES[0],
    };
    let make = |(n, rows): (usize, usize), seed: u64| {
        let gain = [1.0, 0.5, uniform(1, (seed, 3), 0.0..4.0)[0]][(seed % 3) as usize];
        let out = match c {
            Drain::Add(_) => Out::InPlace(draw(n, (seed, 1), 1.0, true)),
            Drain::To(_) => Out::Poison(n),
        };
        (
            vec![],
            vec![out, Out::Words(acc_words(rows * n, seed))],
            gain,
        )
    };
    super::device::sweep(t, shapes, make, |side, x: &mut Cell, (n, rows), &gain| {
        drain_on(
            c,
            side,
            (x.o(side, 0), x.o(side, 1).cast()),
            (n, rows, gain),
        )
    });
    let (_, go) = guard_ptrs();
    for (n, rows) in [(0, 1), (8, 0), (0, 0)] {
        let refusal = both(|side| drain_on(c, side, (go, go.cast()), (n, rows, 1.0)));
        t.guard(&format!("{name} n={n} rows={rows}"), refusal);
    }
}

pub fn moe_acc_drain_s(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_moe_acc_drain_s`'s C signature.
    let c = Drain::Add(unsafe { r.sym(c"rivoli_moe_acc_drain_s") });
    drain_row(t, (c, "moe_acc_drain_s"));
}

pub fn moe_acc_drain_to_s(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_moe_acc_drain_to_s`'s C signature.
    let c = Drain::To(unsafe { r.sym(c"rivoli_moe_acc_drain_to_s") });
    drain_row(t, (c, "moe_acc_drain_to_s"));
}
