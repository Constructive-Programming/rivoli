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
//! A tolerance is only meaningful in the gap between them, so [`Policy`] is derived from the ratio
//! rather than written down, and [`tolerances_leave_room`] fails if a row's numbers stop supporting
//! its policy. **A tolerance nobody can justify from a measurement is a number that will be widened
//! the first time a kernel disagrees.**
//!
//! Included by `#[path]` (this repo's pattern, cf. `common/f4_artifact_dir.rs`) so S2's kernel tests
//! and `k3_anchor.rs` share one table.

/// How a kernel's output may be compared against the golden for one operator.
#[derive(Debug, PartialEq)]
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

/// **Every row's policy has to follow from its two measurements.**
///
/// `Rel(t)` is only defensible when `t` sits clear of the floor and far below the weakest defect it
/// must catch: 10x above the floor, and at least 30x under the defect. `ExactOnly` is only
/// defensible when the gap is too small for any threshold — under 3x — because otherwise it is
/// pessimism dressed as rigour, and an exact comparison that did not need to be exact will be
/// relaxed by whoever hits it next.
///
/// This is the gate on the table. Widening `mla` to a `Rel` fails here, which is the point.
pub fn tolerances_leave_room() {
    for t in TOLERANCES {
        let margin = t.weakest_defect / t.floor;
        match t.policy {
            Policy::Rel(tol) => {
                assert!(
                    margin >= 30.0,
                    "{}: the weakest defect is only {margin:.1}x its floor — no Rel tolerance is \
                     defensible, mark it ExactOnly",
                    t.operator
                );
                assert!(
                    tol >= t.floor * 9.9,
                    "{}: tolerance {tol:e} is not clear of the {:e} rounding floor",
                    t.operator,
                    t.floor
                );
                assert!(
                    tol <= t.weakest_defect / 30.0,
                    "{}: tolerance {tol:e} is within 30x of the {:e} defect it must catch",
                    t.operator,
                    t.weakest_defect
                );
            }
            Policy::ExactOnly => assert!(
                margin < 3.0,
                "{}: the weakest defect is {margin:.1}x its floor, so a Rel tolerance IS \
                 defensible — ExactOnly here is pessimism, not rigour",
                t.operator
            ),
        }
    }
}
