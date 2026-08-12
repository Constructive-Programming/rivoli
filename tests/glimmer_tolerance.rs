//! **The per-operator tolerances S2 scores Muse Glimmer's kernels against, and the gate on them.**
//!
//! `docs/measurement/glimmer-reference/anchor.md` shipped saying "No tolerances", and S2 item 1 is
//! where that stopped being acceptable: a kernel cannot be scored against a golden without a number
//! saying how far apart they are allowed to be, and **a number chosen after seeing the kernel's
//! output is not a measurement**. The table lives in `common/tolerance.rs` beside K3's, sharing one
//! `Policy`, one floor rule and one gate; only the rows are per-model, because a floor is a
//! measurement on one model's arithmetic and means nothing on another's.
//!
//! Its own binary rather than a test inside `glimmer_anchor.rs`. That file is a fixture-INTEGRITY
//! gate — it asks whether the vendored bytes are the ones the doc describes — and this asks whether
//! a set of thresholds follows from two sets of measurements. Different question, different
//! failure, and putting them together made `build.rs`'s jscpd gate fire on the shared include
//! block, which was the structure telling the truth before the author did.
//!
//! No GPU, no python, no goldens read: this is arithmetic over a table of constants.

#[path = "common/tolerance.rs"]
mod tolerance;

/// **Every row's policy still follows from the two numbers behind it.**
///
/// The rows that do NOT exist carry as much of the meaning. Thirteen operator buckets have measured
/// fp32 floors recorded in `anchor.md`; one has a row here. A floor is only half of a tolerance —
/// the other half is deciding which defects the operator is answerable for, and that is per-kernel
/// reasoning rather than a number a sweep produces. `attend`'s row exists because S2 item 1 did
/// that reasoning, including the exclusion that decided its policy (`qk_scale_on_k` is invisible to
/// this kernel by algebra, not by resolution — see the row's own comment).
///
/// **Until an operator has a row, S2 must compare it exactly rather than pick a threshold.** That
/// is the rule this gate enforces from the other side: a row for an unanalysed operator fails here.
#[test]
fn every_glimmer_row_follows_from_its_measurements() {
    tolerance::tolerances_leave_room(tolerance::GLIMMER);
    // The spelling S2's fixtures look a row up by. A rename goes red here rather than silently
    // returning `None`, which a caller would experience as "no tolerance applies" — i.e. as
    // nothing happening at all.
    tolerance::table_covers_exactly(tolerance::GLIMMER, &["attend", "rope", "o_proj", "logits"]);
}
