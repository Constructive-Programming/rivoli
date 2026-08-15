//! The assertions: a measured comparison promoted to a PANIC, with the message that says what
//! it was held to.
//!
//! `scoring.rs` next door owns the metric, the tolerance and the printed line — every function
//! here is one of those plus a decision about what a failure means. Which assertion an oracle
//! reaches for is the load-bearing choice, and each body argues its own case in place: bitwise
//! for one-thread-per-element kernels, a relative bound for anything that reduces, a guard CODE
//! rather than `is_err` for a launcher rejection.
//!
//! **Split out of `common/mod.rs` 2026-08-15** under the file-size gate, then split from
//! `scoring.rs` the same day — see that file's header for the measurement (9.68 together on
//! Primitive Obsession, 10.0 apiece on this seam). Bodies and their comments travelled
//! verbatim, and `mod.rs` re-exports both with a glob, so every
//! `use common::{assert_bits, assert_close, assert_rel, …}` is untouched.

use super::scoring::{Got, Want, max_abs, report, report_rel};

/// Assert two slices are BIT-IDENTICAL, reporting the first disagreement and how many
/// there are.
///
/// For the kernels with **one thread per output element and no reduction**, where exact
/// agreement with a host transliteration is a property of the arithmetic rather than luck.
/// Anything that reduces gets [`assert_rel`] instead — or [`assert_close`] where its shared
/// `1e-3·max + 1e-3` floor is honest for the fixture's scale. Measured on this tree, a
/// correct wave-reduced kernel differs from its oracle on ~0.08% of bf16 elements at dim
/// 4096, so a bitwise gate there rejects correct code.
///
/// Prints the element count on success: "identical" over 4 elements and over 4096 are not
/// the same evidence.
pub fn assert_bitwise<T: PartialEq + std::fmt::Debug>(want: &[T], got: &[T], label: &str) {
    assert_eq!(want.len(), got.len(), "{label}: length");
    let bad: Vec<usize> = (0..want.len()).filter(|&i| want[i] != got[i]).collect();
    match bad.first() {
        None => println!("{label}: {} elements, bit-identical", want.len()),
        Some(&i) => panic!(
            "{label}: {} of {} elements differ; first at {i}: want {:?}, got {:?}",
            bad.len(),
            want.len(),
            want[i],
            got[i]
        ),
    }
}

/// [`assert_bitwise`] over f32 BIT PATTERNS.
///
/// Not `assert_bitwise(want, got)` directly on the floats: `PartialEq` for f32 says
/// `-0.0 == 0.0`, so a sign-dropping defect passes an assertion that claims exactness, and
/// says `NaN != NaN`, so a NaN-poisoned buffer fails one for the wrong reason. Five call
/// sites spelled the `.to_bits()` fold on both operands; `tests/f4_kernel.rs` and
/// `tests/hadamard_basis.rs` each keep a private `bits()` for the same reason.
pub fn assert_bits(want: &[f32], got: &[f32], label: &str) {
    let b = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert_bitwise(&b(want), &b(got), label);
}

// **MOVED HERE from `kernel.rs` 2026-08-15**, when the MoE expert-range oracles left it for
// `kernel_moe.rs`: the batched-row claim and the guard-code assertion are each made on BOTH
// sides of that split, and a second copy of either is what `build.rs`'s duplication gate is
// for. `assert_out` did NOT come — it is `DeviceBuf`-typed and only the file that kept the
// GEMV/MLA destinations still calls it.
/// Both rows of a two-row batch against their own single-row runs, named per row.
///
/// Row 0 alone would pass a kernel that batches correctly but leaks row 0's input into row 1
/// (a missing `r * stride`), so both rows are asserted and the message says which failed.
pub fn assert_rows<T: PartialEq + std::fmt::Debug>(got: &[T], want: &[Vec<T>], w: usize, k: &str) {
    assert_eq!(got[..w], want[0][..], "{k} row 0 must be bit-identical");
    assert_eq!(got[w..], want[1][..], "{k} row 1 must be bit-identical");
}

/// A launcher result against an expected guard code: `None` must be ACCEPTED, `Some(n)`
/// rejected with `n` somewhere in the message.
///
/// The CODE is asserted rather than merely `is_err`, and that is the whole value of these
/// tests: one that accepted any error would still pass if someone replaced a power-of-two
/// check with `block != 128`, or if an unrelated dimension guard started swallowing the
/// case first.
///
/// (That paragraph sat on [`assert_rows`] in `kernel.rs` — two doc blocks stacked on one
/// function, describing two. It is re-anchored, not rewritten.)
pub fn assert_guard<T: std::fmt::Debug>(r: anyhow::Result<T>, want: Option<u32>, what: &str) {
    match want {
        None => assert!(r.is_ok(), "{what}: {r:?}"),
        Some(code) => {
            let msg = format!("{:#}", r.expect_err("expected a guard rejection"));
            assert!(
                msg.contains(&code.to_string()),
                "{what}: want guard {code}, got {msg:?}"
            );
        }
    }
}

/// [`report_rel`] promoted to an assertion, for oracles whose agreement is far tighter
/// than [`super::err_tol`]'s `1e-3·max + 1e-3` floor and would pass on two orders of headroom
/// under it.
///
/// Takes its ratio per call rather than sharing one: a single `TOL` shared across a file is
/// how a widening made for one comparison silently degrades every other, and these
/// oracles' honest tolerances differ by four orders of magnitude — `swiglu` is one `expf`
/// apart from the host, `index_head_route` is an LDS tree reduction.
pub fn assert_rel(want: &[f32], got: &[f32], label: &str, ratio: f32) {
    let (err, tol) = report_rel(Want(want), Got(got), label, ratio);
    assert!(
        err <= tol,
        "{label}: err={err:.3e} > tol={tol:.3e} (rel={ratio:.1e} of max={:.3e})",
        max_abs(Want(want))
    );
}

/// Report the max error AND the threshold it was compared against. Printing BOTH is the
/// point: a green oracle that passed on 100x of headroom looks exactly like one that passed
/// on 2x, and only one of them is evidence of anything.
pub fn assert_close(want: &[f32], got: &[f32], label: &str) {
    let (err, tol) = report(Want(want), Got(got), label);
    assert!(
        err <= tol,
        "{label}: err={err:.3e} > tol={tol:.3e} max={:.3e}",
        max_abs(Want(want))
    );
}
