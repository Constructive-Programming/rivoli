//! **What a Kimi-K3 HIP kernel is allowed to differ from the anchor golden by, per operator.**
//!
//! G1b listed the per-operator tolerances as owed after the anchor itself. This is them, and the
//! numbers are **measured, not chosen** — `docs/measurement/k3-reference/anchor.md` §tolerances
//! records the commands. Two measurements per operator:
//!
//! * **`floor`** — the fp32 run's own rounding error, from running the identical reference at
//!   double precision and diffing (`--dtype float64`, then `--by-operator`). An independent correct
//!   implementation in fp32, associating its sums differently, cannot beat this. `kda_op`'s floor
//!   comes from `--mode kda-equiv` instead: fla's KDA kernel does not compile for fp64 at all, so
//!   its floor is the disagreement between the two paths fla itself ships for the same recurrence
//!   (`chunk_kda` vs `fused_recurrent_kda`, worst of 69 layers).
//! * **`weakest_defect`** — the smallest signal among the defect runs that TARGET this operator.
//!   Another operator's defect leaking downstream is not what this operator's tolerance is for.
//!
//! A tolerance is only meaningful in the gap between them. Stated precisely, because the looser
//! wording here has now overclaimed twice: the CHOICE between [`Policy::Rel`] and
//! [`Policy::ExactOnly`] follows from the ratio and nothing else, and each `Rel` value is a rule
//! rather than a fourth measurement — **10x the floor, admitted within two-significant-figure
//! rounding** ([`FLOOR_MULT`]), since the tolerances are written to 2 s.f. against floors recorded
//! to 4 and land between 9.998x and 10.185x. An earlier version of this paragraph said "10x, which
//! is where it sits for all five" — false for `kda_op` at 9.998x, and the undocumented 9.9 in the
//! gate was the only thing admitting it. It also claimed a value drifting off the rule "in either
//! direction" went red when only the lower side was bounded; `Rel(5.0e-2)` on `attn_res` — 3183x
//! its floor — passed. Both halves are bounded now.
//!
//! [`tolerances_leave_room`] keeps the floor and the tolerance as separate statements that must
//! agree, rather than computing the value from the floor: a typo that makes a `floor` too LARGE is
//! then caught by its own tolerance, and marking `mla` as a `Rel` stays expressible so the gate can
//! be proven able to reject it.
//! **A tolerance nobody can justify from a measurement is a number that will be widened the first
//! time a kernel disagrees.**
//!
//! Included by `#[path]` (this repo's pattern, cf. `common/f4_artifact_dir.rs`) so S2's kernel tests
//! and `k3_anchor.rs` share one table.

// Included by `#[path]` into more than one test binary, each using a subset. Without this, every
// binary warns about what it does not happen to reference — noise that says nothing about whether
// the module is dead, since deadness here is a per-binary accident.
#![allow(dead_code)]

/// How a kernel's output may be compared against the golden for one operator.
pub enum Policy {
    /// Relative difference, `max|a-b| / max|b|`, must not exceed this.
    Rel(f32),
    /// **No tolerance can separate a correct implementation from a known defect here.** Compare
    /// bit-exactly, and settle the defect structurally instead — see `mla` below.
    ExactOnly,
}

pub struct Tol {
    pub operator: &'static str,
    pub floor: f32,
    pub weakest_defect: f32,
    pub policy: Policy,
}

/// Measured 2026-08-11 on gfx1151, decode, `--salt k3-anchor-1`. Re-derive with the two commands in
/// `anchor.md`; these are not transcribed from anywhere else.
pub const TOLERANCES: &[Tol] = &[
    // AttnRes, the S2 item 1 fold. Floor is dominated by the softmax over the block axis.
    Tol {
        operator: "attn_res",
        floor: 1.571e-5,
        weakest_defect: 1.80e0,
        policy: Policy::Rel(1.6e-4),
    },
    // **MLA is EXACT-ONLY, and this is the load-bearing finding of the whole tolerance exercise.**
    // The C reference's LoRA-norm eps (1e-5 against first-party 1e-6) shifts this operator by
    // 2.22e-5 while the operator's own fp32 rounding floor is 1.70e-5 — a margin of **1.3x**. There
    // is no threshold that admits a correct kernel and rejects that eps, so the eps cannot be
    // settled numerically AT ALL: S2/S3 must pin the constant by reading it, and MLA's fixture is
    // scored bit-exactly. Had S1b shipped tolerance fixtures instead of exact bytes, the divergence
    // G0 item 11 found would have been invisible to its own gate.
    Tol {
        operator: "mla",
        floor: 1.697e-5,
        weakest_defect: 2.22e-5,
        policy: Policy::ExactOnly,
    },
    Tol {
        operator: "moe_latent",
        floor: 2.851e-5,
        weakest_defect: 2.05e2,
        policy: Policy::Rel(2.9e-4),
    },
    Tol {
        operator: "moe_route",
        floor: 2.472e-5,
        weakest_defect: 2.23e0,
        policy: Policy::Rel(2.5e-4),
    },
    // `kda_op`'s floor is 6.301e-5 from chunk-vs-recurrent, an order of magnitude above the 5.99e-6
    // the fp64 island reports — and the larger number is the honest one. The island measures the
    // kernel's sensitivity to slightly different fp32 inputs; chunk-vs-recurrent measures two real
    // implementations of the recurrence disagreeing, which is what a HIP port will be.
    Tol {
        operator: "kda_op",
        floor: 6.301e-5,
        weakest_defect: 1.75e0,
        policy: Policy::Rel(6.3e-4),
    },
    Tol {
        operator: "dense_mlp",
        floor: 9.374e-7,
        weakest_defect: 1.28e0,
        policy: Policy::Rel(9.4e-6),
    },
];

/// The tolerance for one operator, or `None` if the table does not cover it.
pub fn tolerance(operator: &str) -> Option<&'static Policy> {
    TOLERANCES
        .iter()
        .find(|t| t.operator == operator)
        .map(|t| &t.policy)
}

/// A `Rel` tolerance is placed at **10x the floor**, and admitted within two-significant-figure
/// rounding of that. The band is not slack: every tolerance in the table is written to 2 s.f. while
/// its floor is recorded to 4, so the realised ratios run 9.998x (`kda_op`, 6.3e-4 over 6.301e-5)
/// to 10.185x (`attn_res`). A literal `== 10.0` would reject `kda_op` for a rounding digit, and a
/// bare lower bound would let a tolerance sit three orders of magnitude high — which it did, and
/// which the module doc above wrongly claimed was caught.
const FLOOR_MULT: (f32, f32) = (9.9, 10.2);

/// And at least this far UNDER the weakest defect it has to catch.
const DEFECT_MARGIN: f32 = 30.0;

/// **Every row's policy has to follow from its two measurements.**
///
/// `Rel(t)` is defensible only when `t` sits clear of the floor and far below the weakest defect:
/// [`FLOOR_MULT`] above the one, [`DEFECT_MARGIN`] under the other. `ExactOnly` is defensible only
/// when no `t` can satisfy both, because otherwise it is pessimism dressed as rigour and an exact
/// comparison that did not need to be exact will be relaxed by whoever hits it next.
///
/// **That boundary is DERIVED, and was wrong when it was written by hand.** A review found the
/// original constants — `>= floor*9.9`, `<= defect/30`, and `ExactOnly` iff `margin < 3.0` — left
/// every margin in `[3.0, 297)` inexpressible: no `Rel` value exists below 297x, because
/// `floor*9.9 <= defect/30` requires `margin >= 9.9*30`. An operator measured at margin 100 (floor
/// 1e-5, defect 1e-3) needed `t >= 9.9e-5` and `t <= 3.33e-5` at once, and its `ExactOnly` was
/// rejected as pessimism. Worse, the two messages each told the author to do what the other
/// refused. Nothing was in that band — `mla` sits at 1.31x and the rest above 27,000x — so the
/// gate was green while being unusable for the next operator measured, which `k3_anchor.rs` now
/// explicitly instructs S2 to do for `kda_trunk`, `norm`, `residual` and `head`.
///
/// This is the gate on the table. Widening `mla` to a `Rel` fails here, which is the point.
pub fn tolerances_leave_room() {
    // The one boundary, so the two branches partition the ratio line with no gap and no overlap.
    let exact_below = FLOOR_MULT.0 * DEFECT_MARGIN;
    for t in TOLERANCES {
        let margin = t.weakest_defect / t.floor;
        match t.policy {
            Policy::Rel(tol) => {
                assert!(
                    margin >= exact_below,
                    "{}: the weakest defect is only {margin:.1}x its floor, and a Rel tolerance \
                     needs {exact_below:.0}x to clear the floor by {}x and the defect by {}x at \
                     once — mark it ExactOnly and settle the difference structurally, as `mla` is",
                    t.operator,
                    FLOOR_MULT.0,
                    DEFECT_MARGIN
                );
                assert!(
                    tol >= t.floor * FLOOR_MULT.0 && tol <= t.floor * FLOOR_MULT.1,
                    "{}: tolerance {tol:e} is {:.3}x its {:e} floor, outside the {}..{}x the rule \
                     places it at (10x, admitted within 2 s.f. rounding)",
                    t.operator,
                    tol / t.floor,
                    t.floor,
                    FLOOR_MULT.0,
                    FLOOR_MULT.1
                );
                assert!(
                    tol <= t.weakest_defect / DEFECT_MARGIN,
                    "{}: tolerance {tol:e} is within {}x of the {:e} defect it must catch",
                    t.operator,
                    DEFECT_MARGIN,
                    t.weakest_defect
                );
            }
            Policy::ExactOnly => assert!(
                margin < exact_below,
                "{}: the weakest defect is {margin:.1}x its floor, which leaves room for a Rel \
                 tolerance at {:e} — ExactOnly here is pessimism, not rigour",
                t.operator,
                t.floor * FLOOR_MULT.0
            ),
        }
    }
}
