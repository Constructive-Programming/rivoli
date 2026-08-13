//! **What a HIP kernel is allowed to differ from its anchor golden by, per operator, per model.**
//!
//! One table per port ([`K3`], [`GLIMMER`]) behind one shape and one gate. It was
//! `common/k3_tolerance.rs` until Muse Glimmer's S2 measured its own floors and needed the same
//! `Policy`/`Tol`/[`tolerances_leave_room`] apparatus: a second copy would have been a jscpd build
//! error, and — the reason that gate exists — the two would have drifted on which multiple of the
//! floor a tolerance sits at. The tables stay separate because a floor is a measurement on one
//! model's arithmetic and means nothing on another's; everything around them is shared.
//!
//! G1b listed the per-operator tolerances as owed after the anchor itself. These are them, and the
//! numbers are **measured, not chosen** — `docs/measurement/{k3,glimmer}-reference/anchor.md`
//! §tolerances record the commands. Two measurements per operator:
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
//! and each anchor test share one table.

// Included by `#[path]` into several test binaries, each of which uses one model's table. Without
// this, every binary warns about the tables and policies it does not happen to reference — noise
// that says nothing about whether the module is dead, since deadness is a per-binary accident here.
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

/// Kimi-K3, measured 2026-08-11 on gfx1151, decode, `--salt k3-anchor-1`. Re-derive with the two
/// commands in `k3-reference/anchor.md`; these are not transcribed from anywhere else.
pub const K3: &[Tol] = &[
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

/// Muse Glimmer, measured 2026-08-11 on CPU (this reference needs no device), `--mode text`, **both
/// weight draws**. Re-derive with the two commands in `glimmer-reference/anchor.md`.
///
/// One row per item that earned one — four from S2, `norm` for S3 item 1 and `qk_norm` for item 2.
/// (The count was spelled out here and went stale twice; `glimmer_tolerance.rs`'s gate is what
/// actually pins the set, so the number is gone rather than corrected a third time.)
/// before its kernel existed (this doc said "One row, and that is deliberate" until 2026-08-12,
/// having been left behind by three items that landed rows). Floors are measured for
/// all thirteen buckets and recorded in `anchor.md`, but a floor is half a row — the other half
/// is deciding which defects TARGET the operator, and that decision is per-kernel work. **Do not
/// score an operator with no row against a threshold**; compare it exactly.
pub const GLIMMER: &[Tol] = &[
    // GQA attend, S2 item 1. Floor is the max over the two draws — 7.819e-6 at draw 1 and this at
    // draw 2, 2.1x apart because the softmax's rounding follows where the scores landed.
    //
    // **`qk_scale_on_k` is EXCLUDED from this row's defect set, and the exclusion is the finding.**
    // It scores 6.232e-4 here, only 38x the floor — under the 297x a `Rel` policy needs, so
    // counting it would force `attend` to `ExactOnly`. It must not: `(s*q)·k` and `q·(s*k)` are the
    // same product, so moving the scale across the dot is invisible to this kernel by ALGEBRA, not
    // by insufficient resolution, and 6.232e-4 is the rounding difference between two spellings of
    // one number. The defect is real and is caught where it is not equivalent — the norm runs
    // between the scale and the product, so `qk_norm` and `proj` see it. An operator's tolerance is
    // for the defects it can distinguish; pricing it against one it provably cannot would have
    // made `attend` exact-only on a false premise.
    //
    // What remains — `kv_broadcast_blocked` 2.086e0, `window_off_by_one` 2.187e0,
    // `full_layers_slide` 2.282e0, each the weaker of the two draws — are genuine attend
    // wrongnesses, and the weakest of them is the number below.
    //
    // Written to 3 s.f., not 2. `1.6e-4` is 9.76x this floor and the gate's own rule rejects it;
    // more digits is strictly more precise, and the rule is 10x, not "two significant figures".
    Tol {
        operator: "attend",
        floor: 1.639e-5,
        weakest_defect: 2.086e0,
        policy: Policy::Rel(1.64e-4),
    },
    // Per-layer RoPE, S2 item 2. Floor from `anchor.md`'s table, max over the two draws
    // (4.490e-6 and this) like every row here.
    //
    // **Two defects target this operator and both are counted**, unlike `attend`'s excluded
    // `qk_scale_on_k`: there is no algebraic identity hiding either of them from a rope kernel.
    // `rope_interleaved` swaps the pairing convention (2.505e0 / 2.214e0 by draw) and
    // `rope_on_nope_layers` rotates the 13 layers whose `layer_rope_theta` is 0 (2.011e0 /
    // 1.811e0). Each is shown as the WEAKER of its two draws, and the weakest of those is below —
    // a margin of 379,000x, so `Rel` is founded with room to spare.
    //
    // Measured 2026-08-12 by regenerating both defects at both salts and running `--by-operator`
    // against the clean run; the fresh `None` goldens came back byte-identical to the vendored
    // ones, which re-checks the anchor's reproducibility claim in passing.
    Tol {
        operator: "rope",
        floor: 4.773e-6,
        weakest_defect: 1.811e0,
        policy: Policy::Rel(4.77e-5),
    },
    // The attention output gate, S2 item 3. The bucket is `o_proj` because that is where
    // `--by-operator` files `attn.o_proj.in_gated` — the gated value BEFORE the projection, which
    // is exactly what the gate kernel produces and what its fixture compares.
    //
    // **One defect targets it: `gate_disabled`** (the gate saturated to sigmoid(20) ~ 1), at
    // 3.759e0 / 3.688e0 by draw, so the weaker is below. Everything upstream also moves `o_proj` —
    // it is late in the layer — but a defect that reaches it by leaking downstream is not what
    // this operator's tolerance is for, which is the same rule `attend`'s row states.
    //
    // Floor 8.293e-6 from `anchor.md`, max over both draws. Margin 444,700x.
    Tol {
        operator: "o_proj",
        floor: 8.293e-6,
        weakest_defect: 3.688e0,
        policy: Policy::Rel(8.29e-5),
    },
    // Muse Glimmer's FOUR SANDWICH NORMS, S3 item 1 — and the bucket is exactly them: 224
    // tensors = 4 norms x 8 layers x 7 steps. `final_norm` (7) and `qk_norm` (112) are their own
    // buckets and their own call sites, which matters because **this model carries two norm
    // formulas**: the four sandwich norms are CENTERED, `x*(1+w)`, while the final norm and the
    // two weightless norms are plain `x*w` (`glimmer-architecture.md` sections 3 and 5).
    //
    // **Measured 2026-08-12, BEFORE the kernel exists** — the S2 discipline, and the only order
    // in which the number means anything. Both defects that target this operator are counted and
    // neither is excluded:
    //
    // | defect | draw 1 | draw 2 | weaker |
    // |---|---|---|---|
    // | `norm_not_centered` — `x*w` for `x*(1+w)`, the form itself | 1.139e0 | 1.131e0 | 1.131e0 |
    // | `post_norm_eps_shared` — `rms_norm_eps` where `post_norm_eps` belongs | 2.571e-2 | 2.024e-2 | **2.024e-2** |
    //
    // The eps defect is the weaker by 56x and it is the one that sets the row. **Counted rather
    // than excluded, unlike `attend`'s `qk_scale_on_k`**: there the defect was invisible to the
    // kernel by ALGEBRA — `(s*q).k` and `q.(s*k)` are the same product — whereas the two eps
    // differ by three orders of magnitude (1e-5 against 1e-8) and a kernel handed the wrong one
    // computes a genuinely different value. So it is a defect this operator is answerable for.
    //
    // Margin 2.024e-2 / 7.701e-6 = **2,628x**, against the 297x a `Rel` policy needs.
    Tol {
        operator: "norm",
        floor: 7.701e-6,
        weakest_defect: 2.024e-2,
        policy: Policy::Rel(7.70e-5),
    },
    // Muse Glimmer's WEIGHTLESS QK-norm, S3 item 2 — 112 tensors, `qk_norm.q` and `qk_norm.k` at
    // every (step, layer), captured BEFORE the 3.87 scale so the golden separates the norm from it.
    //
    // **Measured 2026-08-12, BEFORE the kernel exists.** Two defects touch this bucket and only one
    // of them targets it:
    //
    // | defect | draw 1 | draw 2 | weaker |
    // |---|---|---|---|
    // | `qk_norm_off` — the norm skipped, which is trap 2 (it ships no tensor) | 1.575e0 | 1.483e0 | **1.483e0** |
    // | `qk_scale_on_k` — `qk_scale_factor` on K too | 4.324e-4 | 2.825e-4 | 2.825e-4 |
    //
    // **`qk_scale_on_k` is EXCLUDED on MARGIN, and the "it is invisible" argument this block used to
    // make is false.** There is a real difference from `attend`'s exclusion of the same defect: there
    // it was an algebraic identity, `(s*q).k` and `q.(s*k)` being one product, so the cancellation is
    // exact. Here the driver applies the scale to `k_proj`'s OUTPUT, i.e. to this operator's input,
    // and an RMS norm cancels a scalar on its input **only up to the eps term** — which is not
    // nothing. The residue is `sqrt(1 + (s²−1)·eps/(s²·m + eps)) − 1`, and it reproduces this table's
    // own measured figures: 2.916e-4 at the back-solved `mean(k²)` 0.016 against a measured 2.825e-4,
    // and 6.219e-4 at m = 7.5e-3 against `attend`'s measured 6.232e-4.
    //
    // > **CORRECTED 2026-08-13, and this is the third argument in this operator's round that
    // > measurement overturned.** This paragraph said "a correct kernel handed the scaled input
    // > reproduces the perturbed reference exactly, and no kernel-vs-golden comparison could ever
    // > produce this signal." Both halves are wrong. Against this row's own `Rel(7.85e-5)` the
    // > residue is **3.71x to 7.92x the tolerance** over the m range this anchor can reach — so the
    // > comparison the sentence says could never produce the signal produces it, and a wiring defect
    // > that scaled K would redden a `qk_norm.k` score rather than sail through it.
    // >
    // > **What that leaves OPEN, stated rather than resolved.** The framework's rule is that a
    // > defect this operator provably CANCELS is not one it can be priced against, and the measured
    // > residue says this one is not fully cancelled. Counting it makes the weakest targeting defect
    // > 2.825e-4, a margin of 36x the floor against the 297x a `Rel` policy needs — so by the rule as
    // > written the row would have to be `ExactOnly`, which a kernel scored against a HOST oracle
    // > cannot satisfy at all (its own best is 1.41e-7, not 0). The row stays `Rel` and the exclusion
    // > stays, on margin: every m in this analysis comes from the anchor's toy widths with a
    // > deterministic draw and NO real weights, the residue falls as 1/m, and at m ≈ 1 it is 4.67e-6
    // > = 0.06x tol. **The real 6656-wide checkpoint has never been measured, and this exclusion is a
    // > function of that unmeasured number.** Re-derive it at S4 rather than inheriting this.
    //
    // > **Two corrections review made to this paragraph, 2026-08-12.** (1) It said "an operator
    // > cannot be priced against a defect in its INPUT", which is too broad and contradicts this
    // > anchor's own bucketing — `operator_of` files `attend.q` and both caches, all inputs, into
    // > `attend`, and that row is priced on them. The narrow true form is the one above: a defect
    // > this operator provably CANCELS is not one it can be priced against. (2) It said the
    // > 2.825e-4 residue "is the eps term alone", with `mean(k²) ≈ 0.016` back-solved from the
    // > answer — a fitted identity, not a check. The axis measurement in the same round bounds it:
    // > `1 − mean(y²) = eps/(m+eps)` and the worst run is 8.106e-4, so the smallest `m` anywhere is
    // > 1.233e-2, which caps the eps residue at **3.771e-4** — under draw 1's 4.324e-4. So part of
    // > that figure is cross-layer contamination reaching the `qk_norm.q` captures, the very
    // > category this table's other rows refuse to price on. The exclusion is right twice over; the
    // > sentence explaining it was right once.
    //
    // Margin 1.483e0 / 7.845e-6 = **189,000x**, against the 297x a `Rel` policy needs.
    Tol {
        operator: "qk_norm",
        floor: 7.845e-6,
        weakest_defect: 1.483e0,
        policy: Policy::Rel(7.85e-5),
    },
    // The logit path, S2 item 5 — and the one Glimmer operator this anchor CANNOT price.
    //
    // `softcap_off` moves `logits` by 4.993e-5 / 4.879e-5 by draw, so the weaker is below: only
    // **13.9x** the floor, far under the 297x a `Rel` policy needs. Hence `ExactOnly`, and the
    // reason is not insufficient resolution — it is the TINY MODEL. The softcap is
    // `20*tanh(x*0.196/20)`, and at untrained weights the logits are small enough that `tanh` is
    // in its linear region, where the operation is very nearly the identity. At vocab 202048 with
    // trained weights it bites much harder. So this row says "the anchor cannot separate a correct
    // softcap from an omitted one", NOT "the softcap barely matters".
    //
    // `ids` moves by **exactly 0.000e0** at both draws, which is §5's argmax-invariance stated as
    // a measurement: every greedy gate in this repo is provably blind here.
    //
    // What follows for S2 item 5: its kernel is scored against a host `tanh` at magnitudes
    // where the function has shape, not against these goldens. **Nothing catches the omission
    // in a decode** — an S3 head path that never calls the kernel leaves every gate in the tree
    // green, which is why `glimmer-port.md` §G3 owes a probability-space check. (This sentence
    // said "the omission is caught structurally" until 2026-08-12; the claim had no referent —
    // `kernel_coverage`'s empty-owners row can only notice a caller APPEARING, never demand one
    // exist.) S4 is where trained logits can price it.
    Tol {
        operator: "logits",
        floor: 3.520e-6,
        weakest_defect: 4.879e-5,
        policy: Policy::ExactOnly,
    },
];

/// The tolerance for one operator in one model's table, or `None` if that table does not cover it.
pub fn tolerance(table: &'static [Tol], operator: &str) -> Option<&'static Policy> {
    table
        .iter()
        .find(|t| t.operator == operator)
        .map(|t| &t.policy)
}

/// **The table holds a row for exactly the operators named, no more and no fewer.**
///
/// Both halves earn their place. A missing row means a fixture looks its tolerance up, gets `None`,
/// and falls back to whatever the caller does without one — which is silence, not a failure. An
/// extra row is a threshold for an operator nobody measured, i.e. a number that arrived from
/// somewhere other than a measurement, which is the whole failure this table exists to prevent.
///
/// Factored out of the two call sites once Glimmer's table gained a gate: jscpd caught the second
/// copy at 52 tokens the moment it was written, which is the gate working exactly as intended.
pub fn table_covers_exactly(table: &'static [Tol], measured: &[&str]) {
    for op in measured {
        assert!(
            tolerance(table, op).is_some(),
            "no tolerance row for {op}, so a fixture asking for one gets None and scores nothing"
        );
    }
    let extra: Vec<&str> = table
        .iter()
        .map(|t| t.operator)
        .filter(|o| !measured.contains(o))
        .collect();
    assert!(
        extra.is_empty(),
        "the table has a row for an operator outside the measured set: {extra:?}"
    );
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
///
/// **A floor measured at ONE weight draw is not a floor.** Glimmer's S2 measured its `attend`
/// bucket at both draws and got 7.819e-6 and 1.639e-5 — the same arithmetic, 2.1x apart, because
/// the softmax's rounding depends on where the scores landed. The larger is the floor; the smaller
/// would have placed the threshold at half what a correct kernel can need, and the failure mode is
/// a kernel that is right and cannot pass. [`GLIMMER`] therefore records the max over draws.
/// [`K3`]'s rows were measured at `k3-anchor-1` alone and are **not** known to be draw-robust —
/// that is an open item against K3's table, not a defect this gate can see, since nothing here
/// knows how many draws a number came from.
pub fn tolerances_leave_room(table: &[Tol]) {
    // The one boundary, so the two branches partition the ratio line with no gap and no overlap.
    let exact_below = FLOOR_MULT.0 * DEFECT_MARGIN;
    for t in table {
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
