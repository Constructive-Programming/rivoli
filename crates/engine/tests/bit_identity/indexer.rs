//! Rows for `kernels/indexer.hip`'s eight launchers. Only `index_score_blocks` (in
//! `blockindex.rs`) has a production caller; these shapes come from the checkpoints the kernels
//! were written for and their oracle suites: DeepSeek-V4's indexer (`index_n_heads` 64,
//! `index_head_dim` 128, `index_topk` 2048), MISA's head subsets, Qwen's QSA stage (ratio 4,
//! `indexer_head_dim` 128, and the anchor's 24) — plus the widths that leave blocks ragged and the
//! run-time block sizes at their clamps (`index_score`'s floor of 32, `index_head_route`'s cap of
//! 256 at `hd` 384).
//!
//! `index_topk`'s rows are TIE-HEAVY by construction ([`topk_scores`]): random scores almost never
//! tie, and its output is integers, where a tie-break difference is a byte difference.

use super::device::{Cell, I, O, Out, Ref, S, Shapes, Side, Tally, both, code, guard_ptrs, sweep};
use super::inputs::{
    SCORE_BLOCKS_DIMS, TOPK_SHAPES, bf16_words, draw, score_blocks_ins, topk_scores, uniform,
    words32,
};
use rivoli_backend::abi::ScoreDims;
use rivoli_backend::hip::{self, ScoreBufs};
use std::ptr::{null, null_mut};

type LayerNormFn = unsafe extern "C" fn(I, I, I, i32, f32, O) -> i32;
type AppendFn = unsafe extern "C" fn(I, *mut u16, i32, i32) -> i32;
type ScoreFn =
    unsafe extern "C" fn(I, I, *const u16, *const u32, i32, i32, i32, i32, f32, f32, O) -> i32;
type PushFn = unsafe extern "C" fn(I, O, i32, i32) -> i32;
type RouteFn = unsafe extern "C" fn(I, I, I, i32, i32, i32, O) -> i32;
type TopkFn = unsafe extern "C" fn(I, i32, i32, *mut u32) -> i32;
type PoolNormFn = unsafe extern "C" fn(I, I, i32, i32, i32, f32, O, S) -> i32;
type BlocksFn = unsafe extern "C" fn(I, I, I, O, i32, i32, i32, i32, S) -> i32;

/// A guard case's extents are `i64` so a NEGATIVE one can be spelled: the C side receives it as
/// `int`, the Rust wrapper as the `usize` whose `as i32` narrowing gives back the same `int`.
fn us(v: i64) -> usize {
    v as usize
}

// ── layernorm ───────────────────────────────────────────────────────────────────────────────

fn layernorm_on(
    c: LayerNormFn,
    side: Side,
    ([x, w, b], y): ([I; 3], O),
    (n, eps): (i64, f32),
) -> i32 {
    // SAFETY: live buffers of `n`, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c(x, w, b, n as i32, eps, y) },
        Side::Rust => code(unsafe { hip::launch_layernorm(x, w, b, us(n), eps, y) }),
    }
}

pub fn layernorm(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_layernorm`'s C signature.
    let c: LayerNormFn = unsafe { r.sym(c"rivoli_layernorm") };
    let widths = Shapes {
        all: &[128usize, 7, 1000, 4096],
        hot: 128,
    };
    let make = |n: usize, seed: u64| {
        let ins = vec![
            draw(n, (seed, 1), 1.0, true),
            draw(n, (seed, 2), 1.0, false),
            draw(n, (seed, 3), 0.5, false),
        ];
        (ins, vec![Out::Poison(n)], [1e-6, 1e-5][(seed % 2) as usize])
    };
    sweep(t, widths, make, |side, x: &mut Cell, n, &eps| {
        layernorm_on(
            c,
            side,
            ([x.i(0), x.i(1), x.i(2)], x.o(side, 0)),
            (n as i64, eps),
        )
    });
    let (gi, go) = guard_ptrs();
    for n in [0, -3] {
        let rc = both(|side| layernorm_on(c, side, ([gi; 3], go), (n, 1e-6)));
        t.guard(&format!("layernorm n={n}"), rc);
    }
}

// ── index_append ────────────────────────────────────────────────────────────────────────────

fn append_on(c: AppendFn, side: Side, (k, kc): (I, *mut u16), [pos, hd]: [i64; 2]) -> i32 {
    // SAFETY: `k` of `hd`, `kc` holding row `pos`, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe { c(k, kc, pos as i32, hd as i32) },
        Side::Rust => code(unsafe { hip::launch_index_append(k, kc, us(pos), us(hd)) }),
    }
}

pub fn index_append(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_index_append`'s C signature.
    let c: AppendFn = unsafe { r.sym(c"rivoli_index_append") };
    let rows_at = Shapes {
        all: &[(0usize, 128usize), (5, 128), (77, 7), (3, 300)],
        hot: (5, 128),
    };
    // The cache starts as the same seeded bf16 on both sides: only row `pos` may move, and every
    // other half must come back untouched — compared byte for byte.
    let make = |(pos, hd): (usize, usize), seed: u64| {
        let cache = bf16_words(&draw((pos + 1) * hd, (seed, 2), 1.0, true));
        let out = Out::Exact(Box::new(Out::InPlace(cache)));
        (vec![draw(hd, (seed, 1), 1.0, true)], vec![out], ())
    };
    sweep(t, rows_at, make, |side, x: &mut Cell, (pos, hd), _| {
        append_on(
            c,
            side,
            (x.i(0), x.o(side, 0).cast()),
            [pos as i64, hd as i64],
        )
    });
    let (gi, go) = guard_ptrs();
    for (pos, hd) in [(0, 0), (-1, 8), (3, -2), (-1, 0)] {
        let rc = both(|side| append_on(c, side, (gi, go.cast()), [pos, hd]));
        t.guard(&format!("index_append pos={pos} hd={hd}"), rc);
    }
}

// ── index_score ─────────────────────────────────────────────────────────────────────────────

/// `(nt, nh, nact, hd, listed)`: `listed` passes a head list (MISA), else null (DSA, `nact` = the
/// first heads).
type ScoreShape = (usize, usize, usize, usize, bool);
/// `[nt, nh, nact, hd]` as the guard cases spell them.
type ScoreDimsI = [i64; 4];

fn score_on(
    c: ScoreFn,
    side: Side,
    ([q, w], kc, heads, out): ([I; 2], *const u16, *const u32, O),
    ([nt, nh, nact, hd], (ws, ds)): (ScoreDimsI, (f32, f32)),
) -> i32 {
    let ci = |v: i64| v as i32;
    // SAFETY: live buffers sized for the dims (or null `heads`), or the guard buffer.
    match side {
        Side::Hip => unsafe {
            c(
                q,
                w,
                kc,
                heads,
                ci(nt),
                ci(nh),
                ci(nact),
                ci(hd),
                ws,
                ds,
                out,
            )
        },
        Side::Rust => code(unsafe {
            hip::launch_index_score(
                q,
                w,
                kc,
                heads,
                us(nt),
                us(nh),
                us(nact),
                us(hd),
                ws,
                ds,
                out,
            )
        }),
    }
}

pub fn index_score(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_index_score`'s C signature.
    let c: ScoreFn = unsafe { r.sym(c"rivoli_index_score") };
    let caches = Shapes {
        all: &[
            (4096, 64, 64, 128, false),
            (300, 64, 16, 128, true),
            (1, 64, 8, 128, true),
            (37, 4, 3, 7, false),
            (5, 40, 33, 384, true),
            // The 1002 caps' accepted edges: a 1024-thread block, and `hd + bd` floats at
            // exactly 64 KiB — a cap mistyped as `>=` would refuse these.
            (4, 1024, 1024, 8, true),
            (3, 32, 32, 16_352, false),
        ],
        hot: (4096, 64, 64, 128, false),
    };
    let make = |(nt, nh, nact, hd, listed): ScoreShape, seed: u64| {
        // A reversed stride over the heads: listed heads are neither the first `nact` nor sorted.
        let heads: Vec<u32> = (0..nact)
            .map(|i| ((nh - 1 - i * 3 % nh) % nh) as u32)
            .collect();
        let ins = vec![
            draw(nh * hd, (seed, 1), 1.0, true),
            uniform(nh, (seed, 2), -1.0..1.0),
            bf16_words(&draw(nt * hd, (seed, 3), 1.0, true)),
            words32(if listed { &heads } else { &[0] }),
        ];
        (
            ins,
            vec![Out::Poison(nt)],
            (listed, uniform(2, (seed, 4), 0.01..2.0)),
        )
    };
    sweep(
        t,
        caches,
        make,
        |side, x: &mut Cell, (nt, nh, nact, hd, _), (listed, sc)| {
            let heads = if *listed { x.i(3).cast() } else { null() };
            let bufs = ([x.i(0), x.i(1)], x.i(2).cast(), heads, x.o(side, 0));
            let dims = [nt, nh, nact, hd].map(|v| v as i64);
            score_on(c, side, bufs, (dims, (sc[0], sc[1])))
        },
    );
    let (gi, go) = guard_ptrs();
    // Each extent, nact > nh, then the 1002 pair: a 2048-thread block, and hd + bd floats past
    // 64 KiB — and one case failing both families (order).
    for dims in [
        [0, 4, 4, 8],
        [4, 0, 1, 8],
        [4, 4, 0, 8],
        [4, 4, 5, 8],
        [4, 4, 4, 0],
        [4, 2048, 1025, 8],
        [4, 64, 64, 16_384],
        [0, 2048, 1025, 16_384],
    ] {
        let rc = both(|side| {
            score_on(
                c,
                side,
                ([gi; 2], gi.cast(), null(), go),
                (dims, (1.0, 1.0)),
            )
        });
        t.guard(&format!("index_score {dims:?}"), rc);
    }
}

// ── index_pool_push ─────────────────────────────────────────────────────────────────────────

fn push_on(c: PushFn, side: Side, (k, pool): (I, O), [tok, hd]: [i64; 2]) -> i32 {
    // SAFETY: `k` of `hd`, `pool` holding block `tok / 1024`, or the guard buffer.
    match side {
        Side::Hip => unsafe { c(k, pool, tok as i32, hd as i32) },
        Side::Rust => code(unsafe { hip::launch_index_pool_push(k, pool, us(tok), us(hd)) }),
    }
}

pub fn index_pool_push(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_index_pool_push`'s C signature.
    let c: PushFn = unsafe { r.sym(c"rivoli_index_pool_push") };
    // A block's first token (overwrite), its second and last (fold), the next block's first, deep.
    let tokens = Shapes {
        all: &[
            (0usize, 128usize),
            (1, 128),
            (1023, 128),
            (1024, 128),
            (5000, 128),
            (77, 7),
        ],
        hot: (1, 128),
    };
    let make = |(tok, hd): (usize, usize), seed: u64| {
        let pool = draw((tok / 1024 + 1) * hd, (seed, 2), 1.0, true);
        (
            vec![draw(hd, (seed, 1), 1.0, true)],
            vec![Out::InPlace(pool)],
            (),
        )
    };
    sweep(t, tokens, make, |side, x: &mut Cell, (tok, hd), _| {
        push_on(c, side, (x.i(0), x.o(side, 0)), [tok as i64, hd as i64])
    });
    let (gi, go) = guard_ptrs();
    for (tok, hd) in [(0, 0), (-1, 8), (-1, -1)] {
        let rc = both(|side| push_on(c, side, (gi, go), [tok, hd]));
        t.guard(&format!("index_pool_push t={tok} hd={hd}"), rc);
    }
}

// ── index_head_route ────────────────────────────────────────────────────────────────────────

fn route_on(c: RouteFn, side: Side, ([q, w, p], e): ([I; 3], O), [m, nh, hd]: [i64; 3]) -> i32 {
    // SAFETY: live buffers sized for the dims, or the guard buffer.
    match side {
        Side::Hip => unsafe { c(q, w, p, m as i32, nh as i32, hd as i32, e) },
        Side::Rust => {
            code(unsafe { hip::launch_index_head_route(q, w, p, us(m), us(nh), us(hd), e) })
        }
    }
}

pub fn index_head_route(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_index_head_route`'s C signature.
    let c: RouteFn = unsafe { r.sym(c"rivoli_index_head_route") };
    let pools = Shapes {
        all: &[
            (1usize, 64usize, 128usize),
            (4, 64, 128),
            (3, 8, 384),
            (2, 5, 7),
            (130, 64, 128),
        ],
        hot: (4, 64, 128),
    };
    let make = |(m, nh, hd): (usize, usize, usize), seed: u64| {
        let ins = vec![
            draw(nh * hd, (seed, 1), 1.0, true),
            draw(nh, (seed, 2), 1.0, true),
            draw(m * hd, (seed, 3), 1.0, true),
        ];
        (ins, vec![Out::Poison(nh)], ())
    };
    sweep(t, pools, make, |side, x: &mut Cell, (m, nh, hd), _| {
        let bufs = ([x.i(0), x.i(1), x.i(2)], x.o(side, 0));
        route_on(c, side, bufs, [m, nh, hd].map(|v| v as i64))
    });
    let (gi, go) = guard_ptrs();
    for dims in [[0, 4, 8], [1, 0, 8], [1, 4, 0], [0, 0, 0]] {
        let rc = both(|side| route_on(c, side, ([gi; 3], go), dims));
        t.guard(&format!("index_head_route {dims:?}"), rc);
    }
}

// ── index_topk ──────────────────────────────────────────────────────────────────────────────

fn topk_on(c: TopkFn, side: Side, (s, rows): (I, *mut u32), [nt, k]: [i64; 2]) -> i32 {
    // SAFETY: `s` of `nt`, `rows` of `min(k, nt)`, or the guard buffer.
    match side {
        Side::Hip => unsafe { c(s, nt as i32, k as i32, rows) },
        Side::Rust => code(unsafe { hip::launch_index_topk(s, us(nt), us(k), rows) }),
    }
}

pub fn index_topk(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_index_topk`'s C signature.
    let c: TopkFn = unsafe { r.sym(c"rivoli_index_topk") };
    let arrays = Shapes {
        all: &TOPK_SHAPES,
        hot: (4096, 2048),
    };
    let make = |(nt, k): (usize, usize), seed: u64| {
        let rows = Out::Exact(Box::new(Out::Poison(k.min(nt))));
        (vec![topk_scores(nt, seed)], vec![rows], ())
    };
    sweep(t, arrays, make, |side, x: &mut Cell, (nt, k), _| {
        topk_on(
            c,
            side,
            (x.i(0), x.o(side, 0).cast()),
            [nt as i64, k as i64],
        )
    });
    let (gi, go) = guard_ptrs();
    for (nt, k) in [(0, 4), (4, 0), (-1, 4), (0, 0)] {
        let rc = both(|side| topk_on(c, side, (gi, go.cast()), [nt, k]));
        t.guard(&format!("index_topk nt={nt} k={k}"), rc);
    }
}

// ── index_pool_norm_f32 ─────────────────────────────────────────────────────────────────────

fn pool_norm_on(
    c: PoolNormFn,
    side: Side,
    ([k, w], o): ([I; 2], O),
    ([nb, ra, hd], eps): ([i64; 3], f32),
) -> i32 {
    // SAFETY: live buffers sized for the dims, or the guard buffer.
    match side {
        Side::Hip => unsafe { c(k, w, nb as i32, ra as i32, hd as i32, eps, o, null_mut()) },
        Side::Rust => code(unsafe {
            hip::launch_index_pool_norm_f32(k, w, us(nb), us(ra), us(hd), eps, o, null_mut())
        }),
    }
}

pub fn index_pool_norm_f32(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_index_pool_norm_f32`'s C signature.
    let c: PoolNormFn = unsafe { r.sym(c"rivoli_index_pool_norm_f32") };
    let blocks = Shapes {
        all: &[
            (512usize, 4usize, 128usize),
            (1, 4, 128),
            (37, 4, 24),
            (3, 2, 7),
            (2, 4, 1024),
        ],
        hot: (512, 4, 128),
    };
    let make = |(nb, ra, hd): (usize, usize, usize), seed: u64| {
        let ins = vec![
            draw(nb * ra * hd, (seed, 1), 1.0, true),
            draw(hd, (seed, 2), 1.0, false),
        ];
        // Zero is a legal eps (an exact RMSNorm), and the edges include all-zero rows.
        (
            ins,
            vec![Out::Poison(nb * hd)],
            [1e-6, 0.0, 1e-5][(seed % 3) as usize],
        )
    };
    sweep(t, blocks, make, |side, x: &mut Cell, (nb, ra, hd), &eps| {
        let dims = [nb, ra, hd].map(|v| v as i64);
        pool_norm_on(c, side, ([x.i(0), x.i(1)], x.o(side, 0)), (dims, eps))
    });
    let (gi, go) = guard_ptrs();
    // Each extent, the block cap, ratio 1, eps negative/NaN/+inf, then the order across families.
    for (dims, eps) in [
        ([0, 4, 8], 1e-6),
        ([1, 0, 8], 1e-6),
        ([1, 4, 0], 1e-6),
        ([1, 4, 1025], 1e-6),
        ([1, 1, 8], 1e-6),
        ([1, 4, 8], -1e-6),
        ([1, 4, 8], f32::NAN),
        ([1, 4, 8], f32::INFINITY),
        ([1, 1, 1025], f32::NAN),
        ([0, 1, 1025], f32::NAN),
    ] {
        let rc = both(|side| pool_norm_on(c, side, ([gi; 2], go), (dims, eps)));
        t.guard(&format!("index_pool_norm_f32 {dims:?} eps={eps}"), rc);
    }
}

// ── the block scorers (shared with `blockindex.rs`) ─────────────────────────────────────────

const SCORE_BLOCKS_SHAPES: Shapes<[usize; 4]> = Shapes {
    all: &SCORE_BLOCKS_DIMS,
    hot: SCORE_BLOCKS_DIMS[0],
};

/// The scorers' guard cases: each extent zero, then all four at once.
const SCORE_BLOCKS_GUARDS: [[usize; 4]; 5] = [
    [0, 4, 4, 8],
    [4, 0, 4, 8],
    [4, 4, 0, 8],
    [4, 4, 4, 0],
    [0, 0, 0, 0],
];

/// A block scorer's shipped wrapper.
type RustScorer = unsafe fn(ScoreBufs, ScoreDims, S) -> anyhow::Result<()>;
/// A block scorer's two twins: the reference's C launcher and the shipped wrapper.
type Twins = (BlocksFn, RustScorer);

/// Both twins of a block scorer over one argument set.
fn score_blocks_on(
    (c, rust): Twins,
    side: Side,
    bufs: ScoreBufs,
    [s, n_comp, heads, hd]: [usize; 4],
) -> i32 {
    let ScoreBufs { q, kv, w, score } = bufs;
    let ci = |v: usize| v as i32;
    // SAFETY: live buffers sized for the dims, or the guard buffer under a refused guard.
    match side {
        Side::Hip => unsafe {
            c(
                q,
                kv,
                w,
                score,
                ci(s),
                ci(n_comp),
                ci(heads),
                ci(hd),
                null_mut(),
            )
        },
        Side::Rust => code(unsafe {
            rust(
                bufs,
                ScoreDims {
                    s,
                    n_comp,
                    heads,
                    hd,
                },
                null_mut(),
            )
        }),
    }
}

/// A block scorer's whole row: the sweep and the guards, `name` its C launcher.
pub fn score_blocks_row(r: &Ref, t: &mut Tally, name: &std::ffi::CStr, rust: RustScorer) {
    // SAFETY: both block scorers share this C signature.
    let c: BlocksFn = unsafe { r.sym(name) };
    sweep(
        t,
        SCORE_BLOCKS_SHAPES,
        |dims: [usize; 4], seed| {
            let out = Out::Poison(dims[0] * dims[1]);
            (score_blocks_ins(dims, seed), vec![out], ())
        },
        |side, x: &mut Cell, dims, _| {
            let bufs = ScoreBufs {
                q: x.i(0),
                kv: x.i(1),
                w: x.i(2),
                score: x.o(side, 0),
            };
            score_blocks_on((c, rust), side, bufs, dims)
        },
    );
    let (gi, go) = guard_ptrs();
    for dims in SCORE_BLOCKS_GUARDS {
        let bufs = ScoreBufs {
            q: gi,
            kv: gi,
            w: gi,
            score: go,
        };
        t.guard(
            &format!("{name:?} {dims:?}"),
            both(|side| score_blocks_on((c, rust), side, bufs, dims)),
        );
    }
}

pub fn index_score_blocks_f32(r: &Ref, t: &mut Tally) {
    score_blocks_row(
        r,
        t,
        c"rivoli_index_score_blocks_f32",
        hip::launch_index_score_blocks_f32,
    );
}
