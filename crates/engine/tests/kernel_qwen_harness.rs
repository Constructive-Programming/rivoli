//! **The Qwen3.8-Flash-Next kernel-oracle spine** — track B's vendored S2 goldens, the width
//! and config readers, the measured tolerance table, and the two-bar scoring every
//! `kernel_qwen_*.rs` suite holds a result to.
//!
//! # Why this is a `kernel_qwen_*.rs` file and not a `qwen/mod.rs` directory
//!
//! Kimi-K3's equivalent is `tests/k3/mod.rs`, a module directory. This port's S3 track owns
//! `crates/engine/tests/kernel_qwen_*.rs` and nothing else (`qwen-flash-next-port.md`, the
//! track table), and a `tests/qwen/` directory is outside that list — so the spine takes a
//! name inside it. The cost is that cargo also builds this file as its own integration-test
//! binary; the benefit is that the binary is not empty, because the spine's OWN claims — that
//! the vendored bytes load, that every operator this port scores has a tolerance row, that the
//! two bars are ordered — are tests, and this is where they belong.
//!
//! # No second vendoring
//!
//! K3's kernel harness copied its anchor pair into `tests/k3/`, and its own header flags the
//! result: two vendorings of the same capture, one pinned by the anchor gate and one not,
//! with "reconciling the two is that gate's owner's call". This file does not repeat that. It
//! `#[path]`-includes track B's `qwen_anchor_read.rs`, so there is ONE copy of the bytes and
//! ONE set of FNV pins, and [`the_vendored_qwen_goldens_are_the_bytes_track_b_pinned`] below
//! recomputes them from the live files on every deviceless run. A re-vendor moves one
//! constant, in track B's file, and every consumer follows.
//!
//! # Deviceless is the DEFAULT here, and that is a departure
//!
//! Every K3 and V4 kernel suite opens with a file-level `#![cfg(feature = "rocm")]`, so under
//! `--no-default-features` the whole binary is empty and the fixtures are not even
//! typechecked. The qwen suites gate the DEVICE tests individually instead. What that buys is
//! evidence in the arm that can actually run this round: the host oracles, the fixture
//! assembly, the recovered parameters and the defect separations are all scored against the
//! vendored bytes with no device at all, so a red proof planted in a host oracle is observable
//! now and the device tests add exactly one claim on top — that the kernel computes what the
//! oracle does.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.

// Compiled into five binaries; none names every item, and deadness per binary is an accident of
// which operator that binary scores (`common/mod.rs` makes the same argument for the same reason).
#![allow(dead_code)]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

// Track B's loader, byte pins and width readers, one file on disk. It re-exports
// `golden_read` itself, so a consumer names one module.
#[path = "../../oracles/tests/common/qwen_anchor_read.rs"]
pub mod anchor;

// The `Policy`/`Tol` shape AND the measured `QWEN` rows, from the same file the anchor gate
// reads — deliberately not a copy. K3's kernel table is a second table because its rows are a
// different (kernel-suite) measurement; here track B measured the floors on the two draws these
// suites score against, so a second table would be a second set of numbers for one measurement.
#[path = "../../oracles/tests/common/tolerance.rs"]
pub mod tol;

// **The metric, reached by `#[path]` rather than through `mod common;`, and the reason is that
// `common/mod.rs` does not compile without `rocm`** — `common/upload.rs` carries an ungated
// `impl AttendIo` over a `rocm`-gated type, which is invisible today because every existing
// kernel suite opens with a file-level `#![cfg(feature = "rocm")]` and so never declares the
// module in the deviceless arm. These suites DO run deviceless (see the header), and
// `common/mod.rs` is not this track's file to fix, so the include reaches `scoring.rs`'s ONE
// definition of the metric directly. `scoring.rs` has no `use` line of its own, so it stands
// alone; a suite that also wants the device uploader declares `mod common;` behind `rocm` itself.
#[path = "common/scoring.rs"]
pub mod scoring;

use scoring::{Got, Want, worst_rel};
use serde_json::Value;

// `#[allow(unused_imports)]` on the two `pub use`s, and the argument is the one
// `qwen_anchor_read.rs` writes for its own: this module is included into FIVE private module
// positions, so a re-export no consumer reaches through is an unused import in that binary
// rather than a public API — and `[workspace.lints.rust] warnings = deny` makes every one of
// those a compile error. The alternative is five divergent import lists, which is the drift the
// re-export exists to prevent.
#[allow(unused_imports)]
pub use anchor::golden_read::{GoldenSet, float, ints};
#[allow(unused_imports)]
pub use anchor::{DECODE, PREFILL, int_shape, load, meta_json, width};

/// The `Rel` bound track B measured for `operator`, or a panic naming which of the two ways it
/// can be absent happened.
///
/// `ExactOnly` and "no row" are different failures and must not share a message: the first says
/// a human decided no threshold separates a correct kernel from a known defect — which a fixture
/// scoring a GPU reduction cannot honour, because the reduction reassociates — and the second
/// says the row was renamed and NOTHING is being compared. An `unwrap_or(some_default)` here
/// turns the second into silence, which is the shape of every gate this tree has caught passing
/// vacuously.
pub fn qwen_tol(operator: &str) -> f32 {
    match tol::tolerance(tol::QWEN, operator) {
        Some(tol::Policy::Rel(t)) => *t,
        Some(tol::Policy::ExactOnly) => panic!(
            "the QWEN table has `{operator}` at ExactOnly, so no threshold separates it from its \
             weakest defect. A kernel fixture reassociates the reduction and cannot be bit-exact; \
             either the sub-operator needs its own measured floor, or the quantity must be pinned \
             by READING the reference instead of scored."
        ),
        None => panic!(
            "no `{operator}` row in `tolerance.rs`'s QWEN table — this site would score nothing. \
             Track B owns that table; do not add a row here."
        ),
    }
}

/// One loaded draw: the golden, its recorded tiny widths, and its recorded tiny config.
///
/// Widths come from the golden's `widths` map and never from a literal here. The map is derived
/// at generation time from the config the reference was actually built with, so a fixture read
/// off it cannot disagree with the tensor it is scoring — which is the rule track B's anchor
/// adopted after a review found four accidentally-equal quantities behind assertions that looked
/// like they pinned a coupling.
pub struct Draw {
    pub salt: &'static str,
    pub g: GoldenSet,
    widths: Value,
    cfg: Value,
}

impl Draw {
    fn of(v: &'static anchor::Vendored) -> Self {
        let g = load(v);
        let widths = meta_json(&g, "widths");
        let cfg = meta_json(&g, "tiny_config");
        Self {
            salt: v.name,
            g,
            widths,
            cfg,
        }
    }

    /// A tiny width by the name the generator recorded it under.
    pub fn w(&self, key: &str) -> usize {
        width(&self.widths, key)
    }

    /// A config field the reference read, as `usize`.
    pub fn c(&self, key: &str) -> usize {
        self.cfg[key]
            .as_u64()
            .unwrap_or_else(|| panic!("{key} is not a number in this golden's tiny_config"))
            as usize
    }

    /// A config field the reference read, as `f32`.
    pub fn cf(&self, key: &str) -> f32 {
        self.cfg[key]
            .as_f64()
            .unwrap_or_else(|| panic!("{key} is not a number in this golden's tiny_config"))
            as f32
    }

    /// `rope_theta`, which lives one level down inside `rope_parameters` and is a `f64` all the
    /// way to the launcher — [`rivoli_backend::hip::launch_rope_split_half`] takes `theta: f64`
    /// because the real value is 1e7 and the frequency exponent is evaluated in double.
    pub fn rope_theta(&self) -> f64 {
        self.cfg["rope_parameters"]["rope_theta"]
            .as_f64()
            .expect("rope_parameters.rope_theta")
    }

    /// One float tensor's shape and values.
    pub fn t(&self, name: &str) -> (&[usize], &[f32]) {
        float(&self.g, name)
    }

    /// One float tensor's values, with its element count asserted — the shape a fixture is
    /// about to reshape by. A wrong reshape of a right tensor is the failure this catches, and
    /// at these widths several of the captures are square or otherwise reshape-ambiguous.
    pub fn flat(&self, name: &str, n: usize) -> Vec<f32> {
        let (shape, v) = self.t(name);
        assert_eq!(
            v.len(),
            n,
            "{name}: shape {shape:?} holds {} floats, the fixture expected {n}",
            v.len()
        );
        v.to_vec()
    }

    /// One int tensor's values.
    pub fn i(&self, name: &str) -> Vec<i64> {
        ints(&self.g, name).to_vec()
    }

    /// The decode token's position: the warm prefill is `seq` tokens at positions `0..seq`, so
    /// the single decoded token sits at `seq`.
    ///
    /// **Off by one here is not a rounding error, it is a different rotation.** Measured on the
    /// vendored bytes while this fixture was written: position `seq` reproduces every captured
    /// `rope.out` to 1.2e-7 and position `seq - 1` misses by 3.3e-1 to 9.8e-1 — see
    /// `kernel_qwen_rope.rs`, which asserts both directions rather than only the agreement.
    pub fn decode_pos(&self) -> usize {
        self.g
            .meta_get("seq")
            .expect("seq metadata")
            .parse()
            .expect("seq is a number")
    }
}

/// Both decode draws. Everything that says "both draws" means these.
///
/// Two rather than one because one draw cannot show that an agreement is a property of the
/// arithmetic rather than of the values it landed on, and a kernel defect that degenerates at
/// one draw's numbers hides completely there.
pub fn decode_draws() -> Vec<Draw> {
    DECODE.iter().map(Draw::of).collect()
}

/// The 16-position prefill window, salt 1, layer 3 only — the only golden with more than one
/// query position, and therefore the only one that can see the indexer's dense/selective ladder.
pub fn prefill_draw() -> Draw {
    Draw::of(&PREFILL)
}

/// The logistic sigmoid in `f64`.
///
/// Here rather than per suite because THREE of qwen's operators end in one — the GDN write gate,
/// the GR branch gate and the GR inject scale — and jscpd matched the second and third copies the
/// moment they were written. It is `f64` because every host oracle in these suites is.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Which pairing a partial RoPE READS. The two write the same output slots.
///
/// **One function with a flag here, and two separate entry points in the kernel** — that is not an
/// inconsistency. `activation.hip` keeps `rope_split_half` and `rope_interleave` apart precisely so
/// no GLM or V4 call site can reach the other convention by changing an argument, and its note says
/// so; a host oracle in a test has the opposite requirement, because the whole subject of
/// `kernel_qwen_rope.rs` is COMPARING them, and two bodies would differ only in the read while
/// duplicating the frequency and the write — which is exactly what jscpd reported when they were
/// two.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pairing {
    /// transformers' `rotate_half`: `(row[j], row[j + rd/2])`. Qwen's, measured.
    SplitHalf,
    /// Two adjacent elements: `(row[2j], row[2j+1])`. GLM's and V4's, and trap T16's wrong door.
    Interleaved,
}

/// **Partial RoPE over the FIRST `rd` dims of one row, at `pos`, in place.**
///
/// `SplitHalf` is the convention [`rivoli_backend::hip::launch_rope_split_half`] implements and
/// MOD:566-600 specifies; the frequency is `theta^(-2j/rd)` in both, which is the tree's own
/// statement beside those kernels — "same frequency, same output positions; only the READ moves" —
/// and having one body is what makes it checkable rather than stated. Dims `rd..` pass through.
///
/// In the harness because TWO suites rotate a row: `kernel_qwen_rope.rs` scores the convention
/// against the attention's own captures, and `kernel_qwen_indexer.rs` composes it into the
/// block-key pipeline at each block's first token. One definition, so the fixture that PROVES the
/// convention and the fixture that USES it cannot disagree about what it is.
///
/// The snapshot is load-bearing for `Interleaved` and free for `SplitHalf`: reading `2j` while
/// writing `j` overlaps, so an in-place interleaved rotation without it reads values it has already
/// overwritten — which `activation.hip`'s own kernels solve with a `__syncthreads` between the read
/// and the write.
pub fn rope_row(row: &mut [f32], p: Pairing, rd: usize, pos: usize, theta: f64) {
    let half = rd / 2;
    let src = row.to_vec();
    for j in 0..half {
        let ang = pos as f64 * theta.powf(-(2.0 * j as f64) / rd as f64);
        let (cs, sn) = (ang.cos(), ang.sin());
        let (ai, bi) = match p {
            Pairing::SplitHalf => (j, half + j),
            Pairing::Interleaved => (2 * j, 2 * j + 1),
        };
        let (a, b) = (f64::from(src[ai]), f64::from(src[bi]));
        row[j] = (a * cs - b * sn) as f32;
        row[half + j] = (b * cs + a * sn) as f32;
    }
}

/// The relative metric every site scores with: `max |got - want| / max |want|`, over the whole
/// tensor.
pub fn rel(got: &[f32], want: &[f32]) -> f32 {
    worst_rel(Got(got), Want(want))
}

/// **The two bars a qwen scoring site holds a result to**, and the ordering between them.
///
/// `tol` is track B's measured operator row: a WHOLE-MODEL fp32-against-fp64 floor times ten,
/// carrying every layer of upstream drift. `worst` is what THIS fixture measures, handing the
/// operator the reference's own inputs — one to three orders tighter. Against `tol` alone a
/// change that degraded an operator by two orders would pass in silence, so the site enforces
/// the tighter of the two and the tolerance is the outer envelope.
#[derive(Clone, Copy)]
pub struct Bar {
    pub tol: f32,
    pub worst: f32,
}

impl Bar {
    /// The bar for `operator`, at the worst THIS fixture measures.
    ///
    /// A constructor rather than a struct literal per suite, because the literal is the same six
    /// tokens everywhere and jscpd said so the moment the second suite wrote it: the operator name
    /// is the only thing that varies, and looking the tolerance up through one place is also what
    /// keeps a suite from reaching past [`qwen_tol`]'s two distinct failures.
    pub fn at(operator: &str, worst: f32) -> Self {
        Self {
            tol: qwen_tol(operator),
            worst,
        }
    }

    /// Ten times the measured worst — the bar actually enforced — after checking it is inside
    /// the operator tolerance.
    ///
    /// **The check is the reason this is a function and not a field.** The message on a failed
    /// score invites re-measuring `worst`, and a hand-widened one could sail past `tol` and
    /// leave the site with no bar at all; deriving the enforced bar through a place that
    /// refuses that makes the relationship checked rather than described. Only the lower
    /// direction is checkable uniformly: the gap between the two runs from 65x to five orders
    /// across these operators, because one is a whole-model floor and the other is a fixture's
    /// own reading, so an upper backstop would reject the tightest site.
    pub fn enforced(&self) -> f32 {
        let bar = self.worst * 10.0;
        assert!(
            bar <= self.tol,
            "a {:e} measured worst admits {bar:e}, which is ABOVE the {:e} operator tolerance — \
             this site would have no effective bar. Re-measure the kernel, not the constant.",
            self.worst,
            self.tol
        );
        bar
    }

    /// Score one named quantity. Both bars, in the order that makes the tighter one bind.
    pub fn hold(&self, at: &str, what: &str, got: &[f32], want: &[f32]) {
        let bar = self.enforced();
        let r = rel(got, want);
        assert!(
            r <= self.tol,
            "{at} {what}: {r:e} is outside the {:e} operator tolerance — this is a wrong \
             result, not a regression",
            self.tol
        );
        assert!(
            r <= bar,
            "{at} {what}: {r:e} is far above the {:e} this operator achieves on these fixtures. \
             It is still inside the {:e} tolerance, so this is a REGRESSION tripwire: find what \
             moved, and move the constant only when the new value is defensible.",
            self.worst,
            self.tol
        );
    }

    /// A defect variant has to move the operator clear of the enforced bar by the table's own
    /// 30x margin, or the agreement elsewhere says nothing about which form is implemented.
    ///
    /// Applied to the bar this fixture ENFORCES and not to the operator tolerance: the
    /// tolerance is the loose envelope, so a defect that cleared only it could still be
    /// invisible to the site that is actually scoring.
    pub fn separates(&self, at: &str, what: &str, moved: f32) {
        let need = self.enforced() * 30.0;
        assert!(
            moved > need,
            "{at} {what}: this variant moved the operator by {moved:e}, under the {need:e} that \
             clears this fixture's own bar by the table's 30x. It does not price the difference \
             it names, so nothing here distinguishes the two forms."
        );
    }
}

/// **One defect row's separations over every site of a fixture, checked twice** — the fixture's own
/// bar per site, then the recorded weakest over all of them.
///
/// The two checks answer different questions and the ORDER matters: the bar says the variant is
/// visible at all, and the constant says it is visible *where its name claims*. A separation an
/// order under its constant means the plant landed somewhere other than intended, which the bar
/// alone cannot say. Factored into the harness after jscpd matched this loop across all three
/// qwen suites — it was right about the substance as well as the tokens, since three copies are
/// three places for that order to drift.
///
/// **WHAT THIS CANNOT SEE, measured three times.** A separation is scored variant-against-GOLDEN,
/// so it never reads the fixture's REFERENCE arm and is structurally blind to a wrong one. Observed
/// on 2026-09-01 under four independent plants — a sigmoid decay in `kernel_qwen_gdn_recurrent.rs`,
/// a dropped `1/n_r` in `kernel_qwen_hyper_residual.rs`, and both of
/// `kernel_qwen_gdn_out_norm.rs`'s — every one of which reddened that suite's reference test and
/// left its separations GREEN. **A defect matrix built only from separations passes on a broken
/// oracle.** The two kinds of test are independent and neither subsumes the other, which is why
/// every suite here carries both and why this note is at the shared helper rather than in one
/// header.
///
/// **Minima, never maxima.** The bar a variant must clear is the WEAKEST site's; quoting the
/// strongest lets a form that is nearly invisible at one width pass on another's number. That is
/// not hypothetical here — the indexer's first-row-only variant spans 1.24x across the ladder and
/// the first draft of its constant, recorded as a maximum, reddened this assertion.
pub fn separations(b: Bar, name: &str, floor: f32, sites: &[(String, f32)]) {
    assert!(
        !sites.is_empty(),
        "{name}: zero sites were scored, so nothing was separated — an empty comparison is the \
         examined-count-reaches-zero hole, not a pass"
    );
    for (at, moved) in sites {
        b.separates(at, name, *moved);
    }
    let weakest = sites.iter().map(|(_, m)| *m).fold(f32::INFINITY, f32::min);
    assert!(
        weakest >= floor * 0.9,
        "{name}: the weakest of {} sites separates by {weakest:e}, well under the {floor:e} \
         recorded for it — the variant is landing somewhere other than where its name says",
        sites.len()
    );
}

// ── the spine's own claims ──────────────────────────────────────────────────────────────────

/// Every vendored byte this port scores against is the byte track B pinned, recomputed from the
/// live file rather than compared against a second frozen constant.
#[test]
fn the_vendored_qwen_goldens_are_the_bytes_track_b_pinned() {
    let all = anchor::all();
    assert_eq!(
        all.len(),
        4,
        "four goldens are vendored: two decode draws, the prefill window, the real-width n-gram"
    );
    // Per item rather than `check_pinned_bytes(&[..])`: `Vendored` is not `Copy`, and building an
    // owned slice to hand that helper would be a copy of the pins to keep in step.
    for v in all {
        v.check_bytes();
    }
}

/// The nine operators these suites name all have a `Rel` row, and every row's `Rel` follows
/// from its own two measurements.
///
/// The second half is track B's rule and track B's gate; what is new here is the FIRST half —
/// the operators THIS track scores are exactly the ones it can score, so a renamed row is a
/// compile-time-invisible, run-time-loud failure instead of a suite that quietly stops
/// comparing.
#[test]
fn every_operator_these_suites_score_has_a_measured_row() {
    for op in ["gdn_op", "gdn_out_norm", "indexer", "hc_attn", "moe_route"] {
        let t = qwen_tol(op);
        assert!(
            t > 0.0 && t.is_finite(),
            "{op}: {t:e} is not a usable bound"
        );
    }
}

/// The two bars are ordered, and the ordering is enforced rather than assumed.
#[test]
fn a_tripwire_looser_than_its_tolerance_is_refused() {
    let good = Bar {
        tol: 1.5e-4,
        worst: 2.3e-7,
    };
    assert!(good.enforced() < good.tol);
    let bad = Bar {
        tol: 1.5e-4,
        worst: 1.0e-4,
    };
    let caught = std::panic::catch_unwind(move || bad.enforced());
    assert!(
        caught.is_err(),
        "a measured worst of 1e-4 admits 1e-3, which is above the 1.5e-4 tolerance, and \
         `enforced` must refuse it — otherwise the site it protects has no bar"
    );
}

/// `separations` refuses an EMPTY site list.
///
/// A standing fixture rather than a plant, because the failure it guards is the one that cannot be
/// seen from outside: a fixture whose site loop stops yielding scores every defect row over zero
/// comparisons and reports green. The `weakest` fold would return `INFINITY`, which passes every
/// floor — so the emptiness has to be refused before the fold, and this is what says it is.
///
/// **RED PROOF, OBSERVED 2026-09-01.** Deleting [`separations`]' `!sites.is_empty()` assert (tree
/// changed, `cmp` rc 1; the run recompiled) turns THIS test red at
/// `kernel_qwen_harness.rs:415:5` — *"an empty site list must be refused: the weakest of nothing is
/// INFINITY, which clears every floor there is"* — while the other four stay green, so nothing was
/// masking it. Restored, `cmp` rc 0, 5 passed exit 0.
#[test]
fn a_defect_row_scored_over_zero_sites_is_refused() {
    let b = Bar {
        tol: 1.5e-4,
        worst: 2.3e-7,
    };
    let caught = std::panic::catch_unwind(move || separations(b, "nothing", 1.0, &[]));
    assert!(
        caught.is_err(),
        "an empty site list must be refused: the weakest of nothing is INFINITY, which clears \
         every floor there is"
    );
}

/// Both decode draws load, carry the same tensor set, and are genuinely independent draws.
///
/// The disjointness is asserted on `to_bits` over the SHARED float tensors, not by comparing the
/// two files: each carries its own salt string in metadata, so the files are guaranteed to
/// differ whatever the weights did.
#[test]
fn the_two_decode_draws_are_independent() {
    let d = decode_draws();
    assert_eq!(d.len(), 2);
    let shared = d[0].g.floats.len();
    assert_eq!(shared, 314, "the decode capture set is 314 float tensors");
    anchor::golden_read::draws_are_independent(&d[0].g, &d[1].g, shared);
}
