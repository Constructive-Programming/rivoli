//! What a comparison MEASURES, what it is held to, and what it prints — the numbers, not the
//! verdict. `assert.rs` next door turns these into a panic; the retired `vk.rs`'s
//! `Shapes::close` recorded them and kept going, which is the seam and the reason `report`
//! returns a pair instead of asserting.
//!
//! ONE error metric, ONE tolerance formula, ONE comparison line: a second copy of the metric
//! compares two numbers that are not the same quantity, a second copy of the formula is a
//! second tolerance, and a second copy of the print is a second format. Each body carries the
//! incident that proved its own case.
//!
//! **Split out of `common/mod.rs` 2026-08-15** under the file-size gate, then split again from
//! `asserts.rs` the same day for a MEASURED reason: the two together score 9.68 on CodeScene's
//! Primitive Obsession rule (the `assert_*` surface is `&[f32], &[f32], &str` by design, and
//! the ratio is a whole-file property), and 10.0 apiece on this seam. Bodies and their comments
//! travelled verbatim, and `mod.rs` re-exports both with a glob, so every
//! `use common::{err_tol, rel, report, …}` is untouched.

///
/// A newtype, with [`Got`] as its opposite, because this module spells the pair in **both**
/// orders: `rel(got, want)` and [`worst_rel`]`(got, want)` against [`report`]`(want, got)`,
/// [`err_tol`]`(want, got)` and every `assert_*`. Both sides are `&[f32]`, so a swap compiles —
/// and it is not cosmetic: every bound here is `…max_abs(want)`, so a swapped pair scales the
/// tolerance by the KERNEL's own output and the gate ends up grading itself. Wrapped, the swap is
/// an `E0308` instead of a green run.
#[derive(Clone, Copy)]
pub struct Want<'a>(pub &'a [f32]);

/// The MEASURED side of a comparison — the thing under test. See [`Want`] for why it is a newtype.
#[derive(Clone, Copy)]
pub struct Got<'a>(pub &'a [f32]);

// **MOVED HERE from `glimmer_fixture.rs` 2026-08-13**, for the reason `window_lo` moved the same
// day and one commit earlier: a test binary that includes only this module cannot reach that one,
// so `glimmer_chain.rs` wrote its own scorer — and reintroduced the NaN trap the history below
// records, for the THIRD time. A guard that lives where half the callers cannot see it is a guard
// with a hole in it.
/// `max|got - want| / max|want|` — **the metric every Glimmer tolerance is stated in**, and the one
/// `glimmer_anchor_driver.py::by_operator` computes to produce the floors. Stated once, here,
/// because a fixture that scores against a row in a different metric is comparing two numbers that
/// are not the same quantity.
///
/// Scaled by the reference side's own magnitude, once per tensor, not per element: a per-element
/// ratio divides one rounding error by another wherever the reference is near zero.
pub fn worst_rel(got: Got, want: Want) -> f32 {
    let (got, want) = (got.0, want.0);
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().copied().fold(0.0f32, |m, w| m.max(w.abs()));
    // An all-zero reference has no scale to divide by; any difference is then infinitely relative,
    // and reporting infinity is more honest than dividing by an epsilon.
    if scale == 0.0 {
        return if got.iter().all(|g| *g == 0.0) {
            0.0
        } else {
            f32::INFINITY
        };
    }
    // A non-finite `got` is INFINITY, checked BEFORE the max — `f32::max` returns the other
    // argument when one side is NaN, so the fold below silently discards every NaN difference
    // and an all-NaN kernel output would otherwise score 0.0, a perfect match. That is not
    // hypothetical: a broken kernel in this repo once passed 9 of 9 comparisons that way
    // (2026-08-05), and a review found this helper reintroducing the trap on 2026-08-12.
    if got.iter().any(|g| !g.is_finite()) {
        return f32::INFINITY;
    }
    // The SAME trap on the reference side, and it needs a different answer. `scale` above is
    // another `f32::max` fold, so a NaN in `want` is silently skipped there too — but returning
    // INFINITY would report it as the kernel being wrong, which is a diagnosis of the wrong side.
    // Added 2026-08-12 when the chain gates put golden bytes on this side of a score for the first
    // time; `glimmer_anchor.rs` asserts the captures are finite, so this fires only if that gate
    // and this one disagree, and then the message has to say so.
    assert!(
        want.iter().all(|w| w.is_finite()),
        "the REFERENCE side holds a non-finite value — this is a corrupt or mis-read capture, not \
         a kernel result"
    );
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
        / scale
}

/// `golden.rs::Diff.rel` — max absolute disagreement over the largest expected magnitude.
///
/// The metric the oracle's own gate uses, so an anti-vacuity arm here is scored the same way
/// the goldens are. Moved out of `tests/headtail.rs` on 2026-08-06 when
/// `tests/indexer_kernel.rs` reimplemented it under another name.
pub fn rel(got: &[f32], want: &[f32]) -> f32 {
    // Length first: `zip` truncates, so a short `got` would score 0.0 and read as perfect
    // agreement.
    assert_eq!(
        got.len(),
        want.len(),
        "comparing tensors of different length"
    );
    max_err(Want(want), Got(got)) / max_abs(Want(want)).max(1e-30)
}

/// The largest magnitude in a slice — the scale every tolerance in this suite is stated
/// against.
///
/// Extracted because a second tolerance FORMULA now exists: `tests/f4_kernel.rs` bounds
/// relative to the bf16 quantum instead of [`err_tol`]'s `1e-3·max + 1e-3`, whose absolute
/// floor is 5% of the signal at that fixture's scale. The formulas differ on purpose; the
/// SCALE they are stated against must not, and three copies of this fold were the duplicate
/// the gate found.
///
/// Takes [`Want`] rather than a bare slice because every caller here folds the REFERENCE: a
/// tolerance scaled by the measured side is a gate that grades itself.
pub fn max_abs(v: Want) -> f32 {
    v.0.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

/// [`err_tol`] plus the comparison line, returning the pair so the caller decides what a
/// failure means. It had two callers with two answers — [`super::assert_close`] panics, and the
/// retired `vk.rs`'s `Shapes::close` recorded and kept going. The PRINT is what they shared,
/// and a second copy of the format string is a second format.
pub fn report(want: Want, got: Got, label: &str) -> (f32, f32) {
    let (err, tol) = err_tol(want.0, got.0);
    report_line(
        label,
        Scored {
            err,
            tol,
            mx: max_abs(want),
        },
    )
}

/// [`report`] against a tolerance RELATIVE to the largest expected element, for callers
/// whose signal is too small for [`err_tol`]'s `1e-3` absolute floor to mean anything —
/// `tests/f4_kernel.rs`, where one routed MoE layer's output is ~2e-2 and that floor would
/// be 5% of it.
///
/// Takes the ratio and computes the metric itself. The `(err, tol, mx)` an earlier version took
/// bare is now [`Scored`], which carries that argument.
pub fn report_rel(want: Want, got: Got, label: &str, rel: f32) -> (f32, f32) {
    let mx = max_abs(want);
    report_line(
        label,
        Scored {
            err: max_err(want, got),
            tol: rel * mx,
            mx,
        },
    )
}

/// The three numbers a comparison line prints: the error, the bound it was held to, and the
/// scale both are stated against.
///
/// One value rather than three positional `f32`, and the argument is [`report_rel`]'s own, moved
/// here with the fields it is about: an earlier version of that function took `(err, tol, mx)`
/// bare — three interchangeable `f32`s, where swapping the first two turns the caller's
/// `err <= tol` into `tol <= err`, a gate that goes green on every failure. That is this module's
/// argument about six bare `usize` in a row, made about `f32`; naming the fields answers it.
#[derive(Clone, Copy)]
struct Scored {
    err: f32,
    tol: f32,
    mx: f32,
}

/// The comparison LINE, given an error and whatever bound the caller holds it to. Named for
/// what it emits: it was `report_margin` until 2026-08-05 and the margin is gone. Private:
/// [`report`] and [`report_rel`] are the two ways in, and a third caller would be a third
/// tolerance with no argument attached to it.
///
/// **Prints `err` and `tol` side by side, not a ratio.** It printed `margin = tol/err`
/// until 2026-08-05, and that number is pathological at both ends of its range: a bit-exact
/// result rendered as `margin=532543503195029799199619132512272384.0x`, which reads as
/// corruption rather than as the best possible outcome, and a deliberate-break test — where
/// passing means err EXCEEDS tol — rendered as `margin=0.0x`, which reads as failure beside
/// a green test. Two numbers the reader compares themselves have neither pathology, and the
/// distance is still on the page.
fn report_line(label: &str, s: Scored) -> (f32, f32) {
    let Scored { err, tol, mx } = s;
    println!("{label}: err={err:.3e} tol={tol:.3e} max={mx:.3e}");
    (err, tol)
}

/// `(max abs error, tolerance)` for a want/got pair — the shared arithmetic behind
/// [`super::assert_close`] and [`report`]. Two copies of a tolerance formula is two tolerances.
pub fn err_tol(want: &[f32], got: &[f32]) -> (f32, f32) {
    let (want, got) = (Want(want), Got(got));
    (max_err(want, got), 1e-3 * max_abs(want) + 1e-3)
}

/// The largest absolute disagreement between two slices — the error metric every
/// comparison in this suite uses, whatever tolerance it is held to.
fn max_err(want: Want, got: Got) -> f32 {
    want.0
        .iter()
        .zip(got.0)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
}
