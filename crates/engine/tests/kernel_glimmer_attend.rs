//! Muse Glimmer's two position-indexed kernels — `gqa_attend` and `rope_split_half` — against
//! host oracles written beside them.
//!
//! **One file because they share one argument: a POSITION or a PAIRING convention that the type
//! checker cannot hold, and whose wrong choice stays fluent.** The rotary feeds the queries and
//! keys the attend consumes, and each of the two has a near-twin already in this tree that takes
//! the identical arguments and produces plausible text:
//!
//! * `rope_split_half` pairs `(x[j], x[j+seg/2])`; `rope_interleave` pairs `(x[2j], x[2j+1])`.
//!   Same frequencies, same output positions, only the READ moves
//!   (`old:docs/reference/glimmer-architecture.md` §9 trap 9). They are two entry points rather
//!   than one launcher with a flag precisely so a GLM or V4 call site cannot reach Glimmer's
//!   convention by changing an argument.
//! * `gqa_attend` broadcasts Q head `i` onto KV head `i / (hq/hkv)` — a per-head BLOCK. The
//!   interleave `i % hkv` is trap 10: both are attention over the same tensors, both are fluent,
//!   and one of them is the model.
//! * And `gqa_attend` derives its causal bound from `(start_pos, win)` instead of taking a mask,
//!   because at Glimmer's 131072 context a `[tq][s]` mask is larger than the model. `win > 0`
//!   bounds a query to `[pos - win + 1, pos]` INCLUSIVE of its own position — trap 14 is the
//!   `+ 1` — and the cache is indexed two different ways depending on `ring_cap`.
//!
//! # What this covers, and what carries the power
//!
//! Every kernel comparison here is against a HOST oracle at widths the engine actually runs,
//! which is the half of `old:tests/glimmer_attend.rs` and `old:tests/glimmer_rope.rs` that does
//! not need the anchor's captured bytes. The goldens live in `crates/oracles/` in this tree and
//! the census that owns this port scans `crates/engine/tests`, so the golden-scored halves — the
//! 112-case sweep and the captured-mask comparison — stay beside the bytes they read.
//!
//! **Four of the gates below are BIT-EXACT and need no tolerance at all**: the rotary at position
//! zero, the three cache indexings against each other, the declined permutation route, and the
//! half of the causal-bound test that says an excluded row is unreachable. That is deliberate,
//! because the loosest thing here is the attend's bar — `tolerance::GLIMMER`'s `attend` row,
//! `Rel(1.64e-4)` over a floor measured at double precision before this kernel existed, which
//! `crates/oracles/tests/common/tolerance.rs` carries and which nothing in this crate can name.
//! It is transcribed as [`ATTEND_TOL`] and the number is checkable against that file.
//! `old:tests/glimmer_attend.rs` recorded the kernel at **8.93e-7** and its host oracle at
//! **4.24e-7** against it, and declined to bar the width sweep any tighter for a reason worth
//! keeping: the sweep sums up to 128 terms per dot against the goldens' 8, so the ladder diverges
//! further from a sequential sum here, and if that ever put a correct kernel over the row the
//! honest response is to record the width dependence in the row rather than to widen a bar
//! locally for widths nobody measured. So the observed worst is PRINTED as well as asserted — a
//! green oracle that passed on 180x of headroom looks exactly like one that passed on 2x.
//!
//! Bodies and their comments travelled with their arguments; where the scaffolding differs
//! (`common`'s `Lcg` and `window_lo` for the old fixture's own copies) the argument is restated
//! in place and the change is named.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli_backend::hip::{
    device_sync, launch_gqa_attend, launch_rope_interleave, launch_rope_split_half,
};

mod common;
use common::{
    AttendCase, AttendIo, AttendSpan, Got, Lcg, Want, assert_bits, assert_guard, attend_head,
    attend_launch, attn_scale, back, dev, draws, f32b, f32v, ok, window_lo, worst_rel,
};

/// `tolerance::GLIMMER`'s `attend` row — `Rel(1.64e-4)` over a floor of 1.639e-5, measured at
/// double precision on the reference BEFORE this kernel existed, which is the only order in which
/// the number means anything.
///
/// Transcribed rather than read: the table is `crates/oracles/tests/common/tolerance.rs`, a
/// test-local module of another crate that this binary cannot name. The value is checkable there.
const ATTEND_TOL: f32 = 1.64e-4;

/// The rotary's bar, and it is 48x TIGHTER than the `rope` row (`Rel(4.77e-5)`, same table).
///
/// The row prices a whole bucket of the reference's own fp32 rounding; this comparison is one
/// `pow`/`cos`/`sin` ladder against another, both libm, on identical inputs. A relative error `e`
/// in the angle's cosine is `e` in the output, so the bar is a few ULP of the result — and the
/// host ladder below runs in f64 exactly as the kernel does, so the arg reduction is not part of
/// the disagreement being bounded.
const ROPE_TOL: f32 = 1.0e-6;

/// Glimmer's rope theta. The real value, not a shrunken one: a wrong theta is fluent, and a small
/// one hides the long-context arg reduction the kernel performs in double.
/// `crates/artifact/tests/glimmer_config.rs` pins it against the shipped `config.json`.
const THETA: f64 = 500_000.0;

/// 32 Q heads over 2 KV heads — a group of 16, and `hq / hkv != hkv`, which is what lets a
/// fixture separate the block broadcast from the interleaved one at all.
const HQ: usize = 32;
const HKV: usize = 2;

/// One attention call's inputs, in engine layout.
///
/// A struct because the device path and the host oracle take exactly the same five values, and
/// five of them are `usize` or `&[f32]` — so any permutation type-checks and the failure is not a
/// panic but attention over the wrong rows. `build.rs`'s jscpd gate reached the same conclusion
/// mechanically on the two signatures in the ported file.
#[derive(Clone, Copy)]
struct Case<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    /// `(hq, hkv, head_dim)`.
    dims: (usize, usize, usize),
    /// `(query rows, absolute position of row 0, window — 0 for a global layer)`.
    geom: (usize, usize, usize),
}

impl<'a> Case<'a> {
    /// A case over one [`operands`] draw.
    ///
    /// A constructor and not a literal per test: rustfmt's `struct_lit_width` turns every
    /// `Case { .. }` into one line per field, and five of those differing only in `d` and `geom`
    /// are five identical four-line runs — which `build.rs`'s duplication gate reports, correctly.
    /// Tests that swap one operand say `Case { k: &kk, ..case }`, which keeps the substitution
    /// visible as the one thing that changed.
    fn new(o: &'a (Vec<f32>, Vec<f32>, Vec<f32>), d: usize, geom: (usize, usize, usize)) -> Self {
        Self {
            q: &o.0,
            k: &o.1,
            v: &o.2,
            dims: (HQ, HKV, d),
            geom,
        }
    }
}

// `GqaIo` MOVED to `common::AttendIo` on 2026-08-17, when M17c's block attend became its second
// consumer and jscpd reported the pair as a cross-file clone. Its argument travelled with it.

// `gqa_launch` MOVED to `common::attend_launch` on 2026-08-17, when M17c's block attend became
// its second consumer: the two launchers share an ABI exactly, and jscpd refused the second
// wrapper. Its dims-array argument travelled with it.

/// Launch and read back. `ring_cap` 0 is a cache indexed by position; anything else is a ring
/// whose slot is `position % ring_cap`.
fn run(c: &Case<'_>, ring_cap: usize) -> Vec<f32> {
    let (hq, hkv, d) = c.dims;
    let (tq, start_pos, win) = c.geom;
    let (qb, kb, vb) = (dev(&f32b(c.q)), dev(&f32b(c.k)), dev(&f32b(c.v)));
    let mut ob = dev(&vec![0u8; tq * hq * d * 4]);

    // SAFETY: the three inputs are live device buffers of exactly the sizes `attend_launch`
    // requires, `ob` is writable and distinct from all three (held by `AttendIo::new`'s `&mut`),
    // and all four outlive the join inside `back`.
    let r = unsafe {
        let io = AttendIo::new(&qb, &kb, &vb, &mut ob);
        attend_launch(
            launch_gqa_attend,
            io,
            [hq, hkv, d, tq, start_pos, win, ring_cap],
            attn_scale(d),
        )
    };
    ok(r, "gqa_attend");
    f32v(&back(&ob))
}

/// The reference attention on the host, with the KV broadcast selectable.
///
/// `block` is the reference's own `repeat_kv` (`expand(b, hkv, group, s, d).reshape(...)`, so Q
/// head `i` reads KV head `i / group`); `false` is the interleave `i % hkv`, which is trap 10 and
/// the only reason this function takes an argument at all.
fn attend_host(c: &Case<'_>, block: bool) -> Vec<f32> {
    let (hq, hkv, d) = c.dims;
    let (tq, start_pos, win) = c.geom;
    let group = hq / hkv;
    let mut out = vec![0.0; tq * hq * d];
    // ONE loop over `(row, head)` pairs rather than two nested ones. Q and the destination are
    // both laid out `[row][head][d]`, so the flat index IS `row * hq + h` and splitting it back
    // out is one line — where nesting the loops put the whole softmax three levels deep, which is
    // what CodeScene's Bumpy Road rule reports on this function (measured 2026-08-16, bumps = 2).
    for n in 0..tq * hq {
        let (row, h) = (n / hq, n % hq);
        let pos = start_pos + row;
        let kvh = if block { h / group } else { h % hkv };
        let span = AttendSpan {
            n,
            kvh,
            // The CAUSAL span: `window_lo`'s strict lower edge, and `pos` as the upper bound.
            // The drafter's block attend passes a different pair at both ends.
            span: (window_lo(pos, win), pos),
        };
        let ac = AttendCase {
            q: c.q,
            k: c.k,
            v: c.v,
            hkv,
            d,
        };
        out[n * d..][..d].copy_from_slice(&attend_head(&ac, span));
    }
    out
}

// `attn_scale`, `Slot`/`AttendSpan` and `attend_head` MOVED to `common/reference.rs` on
// 2026-08-17, when M17c's block attend became their second consumer. They were always general
// enough to share — the span here was already documented as inclusive at both ends — and
// `build.rs`'s duplication gate is what would have refused the copy. One owner now, and the
// bidirectional-versus-causal edge distinction is stated where the span type is defined.

/// `max|Δ| / max|reference|` — the metric every Glimmer tolerance is stated in.
///
/// `worst_rel` and not `assert_rel`, and the difference is what keeps this file non-vacuous:
/// `assert_rel` folds `f32::max` over the differences, and **`f32::max` returns the other operand
/// on a NaN**, so an all-NaN kernel output scores 0.0 — a perfect match. A broken kernel in this
/// repo passed 9 of 9 comparisons that way. `worst_rel` returns INFINITY for a non-finite `got`
/// and PANICS on a non-finite `want`, which is the other half of the same trap: a corrupt
/// reference is a diagnosis of the wrong side.
fn worst(got: &[f32], want: &[f32]) -> f32 {
    worst_rel(Got(got), Want(want))
}

// The stream-form fixture draw this file spelled as `draw(r, n)` is now `common::draws`,
// HOISTED 2026-08-16 with M8's V4 oracles — the "fourth consumer" the debt note here made the
// condition for hoisting. Its argument travelled with the body and is stated at `common::draws`:
// the three operands of one attend case come from ONE cursor in a fixed order, because a seed
// means the same data at two call sites only while the draw ORDER is shared.

/// q, k and v for `rows` cached positions at `tq` query rows — everything one attend case needs,
/// in draw order, so two cases of the same shape are two cases over the same data.
fn operands(tq: usize, rows: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = Lcg(0x91a5_0a55);
    (
        draws(&mut r, tq * HQ * d),
        draws(&mut r, rows * HKV * d),
        draws(&mut r, rows * HKV * d),
    )
}

// ---- gqa_attend ----------------------------------------------------------------------------

/// The kernel against the host oracle at the widths the engine runs, which no golden reaches.
///
/// Every anchor capture is at `head_dim` 8, so the accumulator's `nacc` is 1 and register `c = 0`
/// is the only one ever live. Glimmer's real `head_dim` is **128** — four registers — and the
/// kernel spends nine lines justifying a lane mapping nothing had exercised past its first step.
/// 40 is deliberately NOT a multiple of 32: it puts 8 lanes on `c = 1` and idles 24, which is the
/// divergent tail the kernel comment argues for and `mla_latent_attend` refuses.
///
/// The four rows cover a sliding decode step (`tq == 1` with a live window), the same at the real
/// width, a GLOBAL layer at prefill (`win == 0`, four query rows, so the bound moves per row), and
/// a sliding prefill where the window and the row count interact.
#[test]
fn gqa_attend_matches_a_host_oracle_at_the_widths_the_engine_runs() {
    let mut seen = 0.0f32;
    for (d, tq, start_pos, win) in [
        (40usize, 1usize, 63usize, 16usize),
        (128, 1, 63, 16),
        (128, 4, 0, 0),
        (96, 3, 0, 8),
    ] {
        let o = operands(tq, start_pos + tq, d);
        let case = Case::new(&o, d, (tq, start_pos, win));
        let r = worst(&run(&case, 0), &attend_host(&case, true));
        assert!(
            r <= ATTEND_TOL,
            "d {d}, tq {tq}, win {win}: worst rel {r:e} > {ATTEND_TOL:e}"
        );
        seen = seen.max(r);
    }
    // Printed as well as asserted: the row is ~180x the 8.93e-7 the ported file measured, so a
    // regression of two decades would still pass it and only this line would show.
    println!("gqa_attend over four widths: worst rel {seen:e} against tol {ATTEND_TOL:e}");
}

/// **The red proof for trap 10.** `i / group` and `i % hkv` are both attention over the same
/// tensors, and if the fixture could not tell them apart the oracle above would be green on
/// either.
///
/// Asserted on the HOST oracle rather than by breaking the kernel: the kernel has one mapping
/// compiled into it, and the claim being proved is about the FIXTURE's power. What earns the
/// oracle the right to make that claim is the device agreeing with it, which is asserted first
/// and in the same case — a common-mode failure that broke both would move numerator and
/// denominator together, and the ratio below is what survives it.
#[test]
fn the_block_kv_broadcast_is_not_the_interleaved_one() {
    // The two mappings are only distinguishable while `hq / hkv != hkv`; at 32 over 2 the group
    // is 16 and they coincide nowhere. Asserted, because a future shape where they agreed would
    // make this test pass while proving nothing.
    assert_ne!(HQ / HKV, HKV, "these widths cannot separate the mappings");
    let (mut weakest, mut ran) = (f32::INFINITY, 0usize);
    for (d, tq, start_pos, win) in [(64usize, 1usize, 40usize, 8usize), (128, 2, 30, 0)] {
        let o = operands(tq, start_pos + tq, d);
        let case = Case::new(&o, d, (tq, start_pos, win));
        let right = attend_host(&case, true);
        let device = worst(&run(&case, 0), &right);
        assert!(
            device <= ATTEND_TOL,
            "d {d} win {win}: the host oracle and the device disagree at {device:e}, so nothing \
             it says about the wrong mapping is evidence"
        );
        weakest = weakest.min(worst(&attend_host(&case, false), &right));
        ran += 1;
    }
    // Census BEFORE the judgment: `weakest` is INFINITY-seeded, so an empty case set would satisfy
    // the bar below vacuously.
    assert_eq!(ran, 2, "the separation loop ran over {ran} cases, not 2");
    println!("trap 10: weakest interleaved-broadcast signal over {ran} cases: {weakest:e}");
    assert!(
        weakest > 100.0 * ATTEND_TOL,
        "the interleaved broadcast produced only {weakest:e}, {:.0}x the tolerance — this fixture \
         is blind to trap 10",
        weakest / ATTEND_TOL
    );
}

/// **The derived bound IS the sliding window, proved by perturbation rather than against a mask.**
///
/// The ported file compares the derivation to the reference's captured mask; that comparison
/// belongs with the goldens. What survives without them is stronger in one way and weaker in
/// another, and the trade is worth naming: the mask test restates the bound in Rust, so breaking
/// `lo` in the kernel leaves it green (measured — `pos - win` in place of `pos - win + 1` reddens
/// the kernel comparison at 1.07 absolute and the mask test not at all). This one reads no bound
/// at all. It corrupts the cache rows OUTSIDE the union the launch may attend and requires the
/// output to be **bit-identical**, then corrupts the OLDEST row inside it and requires the output
/// to move.
///
/// So `pos - win` instead of `pos - win + 1` reddens the first half directly: row `lo - 1` would
/// be read, and it is one of the rows this test poisons. The second half is what stops the first
/// from passing vacuously on a kernel that reads nothing.
#[test]
fn the_derived_causal_bound_is_the_sliding_window() {
    let (d, tq, start_pos, win) = (64usize, 1usize, 63usize, 16usize);
    let o = operands(tq, start_pos + tq, d);
    let base = Case::new(&o, d, (tq, start_pos, win));
    let stride = HKV * d;
    let lo = window_lo(start_pos, win);
    assert_eq!(lo, 48, "the fixture's own window bound moved");
    let clean = run(&base, 0);

    // Poison every row the bound excludes. A row is `stride` floats; `f32::MAX` rather than a NaN
    // so that a kernel which DID read one produces a finite, large, comparable wrong answer
    // instead of a NaN that `assert_bits` would report as a bit difference for the wrong reason.
    let poison = |from: usize, to: usize| -> Vec<f32> {
        let mut kk = base.k.to_vec();
        kk[from * stride..to * stride].fill(f32::MAX);
        kk
    };
    let outside = poison(0, lo);
    assert_bits(
        &clean,
        &run(
            &Case {
                k: &outside,
                ..base
            },
            0,
        ),
        "rows before the window's lower bound must be unreachable",
    );

    // And the converse, or the assertion above is satisfied by a kernel that reads nothing.
    let inside = poison(lo, lo + 1);
    let moved = worst(&run(&Case { k: &inside, ..base }, 0), &clean);
    println!("poisoning cache row {lo} (the oldest attended) moves it {moved:e}");
    assert!(
        moved > 100.0 * ATTEND_TOL,
        "poisoning the OLDEST attended row moved the output by only {moved:e} — the bit-identity \
         above is then satisfied by a kernel that attends nothing at all"
    );
}

/// **The three ways this kernel is told where a cached position lives, all attending the same
/// rows, bit for bit.**
///
/// `start_pos` is the absolute position of query row 0, and the cache must be indexed to match —
/// the two modes index it DIFFERENTLY, which is the launcher's own emphasis:
///
/// * `ring_cap != 0`: slot is `position % ring_cap`, so `start_pos` stays absolute and the ring
///   holds any window of history. At real scale a local layer holds 2048 slots and position `p`
///   lives at `p % 2048` — that is what makes a 131072-token context cost 2048 rows, and an
///   off-by-one in the mapping reads a row 2048 positions stale while every shape stays right.
/// * `ring_cap == 0`: the slot IS the position, so the cache must run from position 0. A caller
///   that trims a linear cache to its last `win` rows and leaves `start_pos` absolute reads past
///   the end; one that trims without ALSO shifting `start_pos` attends the wrong rows fluently.
///   The trimmed arm below does exactly that shift, which is the only correct spelling of it.
///
/// Bit-identical and not a tolerance: the three runs read the same values in the same order, so
/// the reduction is the same reduction and any difference at all is the indexing.
#[test]
fn the_three_cache_indexings_attend_the_same_rows() {
    let (d, tq, start_pos, win) = (64usize, 1usize, 63usize, 16usize);
    let rows = start_pos + tq;
    let o = operands(tq, rows, d);
    let base = Case::new(&o, d, (tq, start_pos, win));
    let stride = HKV * d;
    let lo = window_lo(start_pos, win);
    let linear = run(&base, 0);

    // Trimmed linear: the cache holds positions `[lo, start_pos]` from its own row 0, so the
    // kernel's numbering starts there and `start_pos` shifts down by `lo`.
    let cut = |src: &[f32]| src[lo * stride..rows * stride].to_vec();
    let (kt, vt) = (cut(base.k), cut(base.v));
    let trimmed = Case {
        k: &kt,
        v: &vt,
        geom: (tq, start_pos - lo, win),
        ..base
    };
    assert_bits(
        &linear,
        &run(&trimmed, 0),
        "a trimmed linear cache with start_pos shifted to match",
    );

    // Ring: absolute positions this time, which is what a ring is indexed by. Scattered by
    // `position % win`, which is where the engine's cache writer puts them.
    let (mut kr, mut vr) = (vec![0.0f32; win * stride], vec![0.0f32; win * stride]);
    for j in lo..rows {
        let slot = j % win;
        kr[slot * stride..][..stride].copy_from_slice(&base.k[j * stride..][..stride]);
        vr[slot * stride..][..stride].copy_from_slice(&base.v[j * stride..][..stride]);
    }
    let ringed = Case {
        k: &kr,
        v: &vr,
        ..base
    };
    assert_bits(
        &linear,
        &run(&ringed, win),
        "a ring cache at ring_cap == win",
    );
}

/// Each argument guard rejects before any launch, so a bad call is an error code and not a fault
/// in someone else's kernel three launches later.
///
/// The CODE is asserted rather than merely `is_err`: one that accepted any error would still pass
/// if an unrelated dimension guard started swallowing the case first.
///
/// **`win == 0` with a ring is the one worth spelling out** — a global layer holding a ring would
/// attend the last `ring_cap` positions and silently drop everything older, which is fluent,
/// wrong and permanent, so the launcher refuses rather than choosing one of the two meanings.
/// **And a `win`-row ring cannot serve two query rows**: the union a launch attends is
/// `win + tq - 1` positions, so at `ring_cap == win, tq == 2` the slot holding
/// `start_pos - win + 1` is the slot `start_pos + 1` was written to — a row overwritten inside
/// its own batch, every shape right, no error. The goldens cannot reach it (the reference hands
/// one query row per sliding step), which is exactly why it has to be a guard.
///
/// Rows are `[hq, hkv, d, tq, start_pos, win, ring_cap]`, [`gqa_launch`]'s own order.
#[test]
fn the_gqa_launcher_refuses_what_it_cannot_compute() {
    let mut b = dev(&vec![0u8; 4096]);
    let out = b.ptr_mut() as *mut f32;
    let operand = b.ptr() as *const f32;
    // SAFETY, for every row below: the six rejected calls return before `hipLaunchKernelGGL`, so
    // no pointer is dereferenced and the aliasing here is unobservable. The two accepted ones do
    // launch, at 6 heads x at most 2 rows x head_dim 8 — 96 f32 in and 96 out, well inside the
    // 4096-byte buffer every argument points into. It is nonsense arithmetic over aliased inputs,
    // which is fine: this test reads the return code and nothing else.
    // The LITERAL form, not `AttendIo::new`: this row set points all four addresses at ONE buffer
    // on purpose, and `new` takes `&mut` for the destination precisely so the ordinary call sites
    // cannot do that by accident. A guard table that never launches is the one place aliasing is
    // the point, so it constructs the fields directly and says so.
    let io = AttendIo {
        q: operand,
        k: operand,
        v: operand,
        out,
    };
    // One-argument closure over the fixed launcher and `io`, so the assert below fits a line.
    // Not cosmetic: at the inline spelling the call exceeded rustfmt's `fn_call_width`, rustfmt
    // broke it across five lines, and jscpd then reported the resulting loop tail as a 27-token
    // clone of `kernel_glimmer_pointwise.rs`'s guard table. rustfmt did not create that
    // duplication — it made visible that two guard tables end in the same four lines — and the
    // closure is the factoring rather than a reformat reverted.
    let fire = |dims: [usize; 7]| unsafe { attend_launch(launch_gqa_attend, io, dims, 1.0) };
    for (dims, want, what) in [
        ([0usize, 2, 8, 1, 0, 1, 0], Some(1001), "zero Q heads"),
        (
            [7, 2, 8, 1, 0, 1, 0],
            Some(1003),
            "hq is not a multiple of hkv",
        ),
        (
            [6, 2, 512, 1, 0, 1, 0],
            Some(1002),
            "head_dim past the accumulator",
        ),
        (
            [6, 2, 8, 1, 0, 0, 4],
            Some(1005),
            "a ring on a global layer",
        ),
        (
            [6, 2, 8, 1, 0, 8, 4],
            Some(1005),
            "a ring shorter than the window",
        ),
        (
            [6, 2, 8, 1, 0, 4, 4],
            None,
            "a ring == the window at ONE query row, legal",
        ),
        (
            [6, 2, 8, 2, 0, 4, 4],
            Some(1005),
            "a ring == the window at TWO query rows",
        ),
        (
            [6, 2, 8, 2, 0, 4, 5],
            None,
            "a ring of win + tq - 1, the smallest legal",
        ),
    ] {
        assert_guard(fire(dims), want, what);
    }
    device_sync().expect("sync the two accepted gqa dispatches");
}

// ---- rope_split_half -----------------------------------------------------------------------

/// One rotation launch's geometry: `count` segments of `seg` at `stride`, at absolute position
/// `pos`.
///
/// Four bare `usize` where every entry is plausible in any other's position, spelled at a launch
/// wrapper, a host reference and a permutation — so a transposed pair would move all three
/// together and the comparison would still agree.
#[derive(Clone, Copy)]
struct Rope {
    count: usize,
    stride: usize,
    seg: usize,
    pos: usize,
}

impl Rope {
    /// The four in launcher order: `[count, stride, seg, pos]`.
    ///
    /// **One array argument, and the order is spelled HERE and nowhere else** —
    /// `common::Mla::new`'s argument, made small. Three tests below want the same segment layout,
    /// and three `Rope { .. }` literals are three identical four-line runs once rustfmt's
    /// `struct_lit_width` has spread them; `build.rs`'s duplication gate reported exactly that. A
    /// caller moving one field says `Rope { pos: 0, ..g }`, which keeps the change visible.
    fn new(dims: [usize; 4]) -> Self {
        let [count, stride, seg, pos] = dims;
        Self {
            count,
            stride,
            seg,
            pos,
        }
    }
}

/// Which of the two rotation kernels to run. They are separate entry points on purpose, so a
/// fixture has to NAME one — which is as much the property under test as the arithmetic is.
#[derive(Clone, Copy)]
enum Conv {
    /// Glimmer's own: pair `(x[j], x[j + seg/2])`. The shipped path.
    SplitHalf,
    /// GLM's and V4's: pair `(x[2j], x[2j+1])`. Correct on Glimmer's activations ONLY after the
    /// permutation `the_declined_permutation_route_computes_the_same_thing` measures, and trap 9
    /// when it is not.
    Interleaved,
}

/// Rotate in place and read back.
fn rope_on_device(data: &[f32], g: Rope, conv: Conv) -> Vec<f32> {
    let mut buf = dev(&f32b(data));
    let base = buf.ptr_mut() as *mut f32;
    // SAFETY: `buf` holds `g.count * g.stride` live f32 and outlives the join inside `back`.
    // In place is the kernel's contract: every pair is read before any write, behind a barrier.
    let r = unsafe {
        match conv {
            Conv::SplitHalf => launch_rope_split_half(base, g.count, g.stride, g.seg, g.pos, THETA),
            Conv::Interleaved => {
                launch_rope_interleave(base, g.count, g.stride, g.seg, g.pos, THETA)
            }
        }
    };
    ok(r, "rope");
    f32v(&back(&buf))
}

/// One rotary fixture: `count * stride` floats under `salt`. Its own function because three tests
/// below want the same segment layout and two of them want the same bytes as each other.
fn rope_base(g: Rope, salt: u64) -> Vec<f32> {
    draws(&mut Lcg(salt), g.count * g.stride)
}

/// Split-half RoPE on the host: pair `(x[j], x[j+half])` rotates by `pos·theta^(-2j/seg)` and
/// writes back to the same two slots.
///
/// **f64 throughout the angle, matching the kernel**, which computes `pow` and the product in
/// double and narrows only the final `cos`/`sin`. Doing the ladder in f32 would disagree with a
/// CORRECT kernel by more than [`ROPE_TOL`] at large `pos`, which is the arg reduction the double
/// is there for.
fn host_split_half(data: &[f32], g: Rope) -> Vec<f32> {
    let half = g.seg / 2;
    let mut out = data.to_vec();
    for s in 0..g.count {
        let row = &data[s * g.stride..s * g.stride + g.seg];
        for j in 0..half {
            let (a, b) = (row[j], row[half + j]);
            let ang = g.pos as f64 * THETA.powf(-2.0 * j as f64 / g.seg as f64);
            let (cs, sn) = (ang.cos() as f32, ang.sin() as f32);
            out[s * g.stride + j] = a * cs - b * sn;
            out[s * g.stride + half + j] = b * cs + a * sn;
        }
    }
    out
}

/// The permutation `old:docs/reference/glimmer-architecture.md` §6 proposed applying at conversion
/// time, within each head: `y[2i] = x[i]`, `y[2i+1] = x[i + half]`.
fn permute(rows: &[f32], g: Rope) -> Vec<f32> {
    let half = g.seg / 2;
    let mut out = rows.to_vec();
    for (s, src) in rows.chunks_exact(g.stride).enumerate() {
        for i in 0..half {
            out[s * g.stride + 2 * i] = src[i];
            out[s * g.stride + 2 * i + 1] = src[i + half];
        }
    }
    out
}

/// The rotary's two arms, and the first is the sharp one.
///
/// * **Position 0 is the IDENTITY, bit for bit.** `cos 0 = 1` and `sin 0 = 0`, and split-half
///   writes its pair back to the two slots it read, so nothing moves — no transcendental, no
///   tolerance. That is not a weak claim, it is the layout half of the kernel pinned against its
///   twin: `rope_interleave` at position 0 is a de-interleave PERMUTATION (it reads `(2j, 2j+1)`
///   and writes `(j, half+j)`), so a suite that confused the two conventions reddens here with no
///   bar to widen.
/// * **A real position** for the angles, at [`ROPE_TOL`], since `pow`/`cos`/`sin` are libm on both
///   sides and are not required to agree bit for bit.
///
/// `stride > seg` in both arms. Glimmer ropes each of `hq` heads at `stride == seg == head_dim`,
/// so a kernel that walked `seg` where `stride` belongs is invisible on the shipped call, and the
/// ragged stride is what makes it visible here.
#[test]
fn rope_split_half_is_the_identity_at_position_zero_and_rotates_at_a_real_one() {
    let g = Rope::new([5, 24, 16, 0]);
    let base = rope_base(g, 0x2909);
    assert_bits(
        &base,
        &rope_on_device(&base, g, Conv::SplitHalf),
        "split-half at pos 0 must be the identity",
    );

    // 137 is deliberately not a multiple of anything in the shape: at theta 500000 the angle
    // ladder spans four decades across the 8 pairs, so both ends of the arg reduction are live.
    let g = Rope { pos: 137, ..g };
    let r = worst(
        &rope_on_device(&base, g, Conv::SplitHalf),
        &host_split_half(&base, g),
    );
    println!(
        "split-half at pos {}: worst rel {r:e} vs {ROPE_TOL:e}",
        g.pos
    );
    assert!(r <= ROPE_TOL, "split-half at pos {}: {r:e}", g.pos);
}

/// **Trap 9, run.** The interleaved kernel on the same input — the mistake a single `bool` on one
/// launcher would have put one argument away from every GLM and V4 call site.
///
/// Two arms, and they prove different things. At a real position the two conventions produce
/// different numbers and the signal is a magnitude. At position 0 the separation is EXACT and
/// needs no bar at all: split-half is the identity there and interleave is a permutation, so the
/// two agree only on the elements the permutation happens to fix. That arm is what a widening of
/// the first arm's bar could never silence.
#[test]
fn the_interleaved_convention_is_not_the_split_half_one() {
    let g = Rope::new([4, 32, 32, 91]);
    let base = rope_base(g, 0x7a09);
    let d = worst(
        &rope_on_device(&base, g, Conv::Interleaved),
        &rope_on_device(&base, g, Conv::SplitHalf),
    );
    println!("trap 9 at pos {}: interleaved differs by {d:e}", g.pos);
    assert!(
        d > 1.0e-1,
        "the interleaved kernel produced only {d:e} of difference — this fixture cannot tell the \
         two conventions apart"
    );

    // The exact half. The permutation fixes only index 0 and index `seg - 1` per segment
    // (`2i == i` and `2i+1 == i+half` have one solution each), so a COUNT is asserted rather than
    // mere inequality: "they differ" alone is satisfied by a single stray write.
    let zero = Rope { pos: 0, ..g };
    let moved = rope_on_device(&base, zero, Conv::Interleaved)
        .iter()
        .zip(&base)
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "at pos 0 the de-interleave moves {moved}/{} elements",
        base.len()
    );
    assert!(
        moved > base.len() / 2,
        "at pos 0 the interleaved kernel moved only {moved} of {} elements, so it is not behaving \
         as the permutation that separates it from split-half",
        base.len()
    );
}

/// **§6's declined route, kept measured — and it is BIT-EXACT, not a tolerance.**
///
/// Permuting the input and running the INTERLEAVED kernel computes the same rotation: interleave
/// reads `(P[2j], P[2j+1])`, which is `(x[j], x[j+half])`, hands them to the same shared
/// `rope_pair`, and writes the same two slots. So the two paths are the same arithmetic on the
/// same values and any difference at all would be a defect — which is a stronger statement than
/// the 1.41e-7 the ported file measured against goldens, because that figure carried the
/// reference's own rounding on both sides.
///
/// Retained rather than deleted so the decision keeps its alternative checked. §6 was declined on
/// 2026-08-12 for a cost the proposal did not price: the permutation forces `q_proj`/`k_proj` out
/// of `copy_verbatim`, so ~3 GB of the artifact stops being a borrowed mapping of the checkpoint.
/// If giving that up ever becomes cheap, this is the evidence the converter route works, already
/// run.
#[test]
fn the_declined_permutation_route_computes_the_same_thing() {
    let g = Rope::new([4, 32, 32, 91]);
    let base = rope_base(g, 0x7a09);
    assert_bits(
        &rope_on_device(&base, g, Conv::SplitHalf),
        &rope_on_device(&permute(&base, g), g, Conv::Interleaved),
        "the permutation route against the split-half kernel",
    );
}
