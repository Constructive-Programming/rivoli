//! The divergence probe's two PURE surfaces, tested without a device.
//!
//! **Here rather than in `src/probe.rs`, and that placement is the finding.** These tests were
//! written inside the module they test, where nothing ran them: `mod probe` is gated on
//! `corruption-probe`, the prescribed `cargo test --workspace` does not set it,
//! `tests/feature-matrix.sh` only `cargo check`s the feature cells, and CI has no rocm arm. So a
//! red proof recorded in `docs/measurement/gate-red-proofs.md` was, in practice, a one-shot manual
//! run. The same file records `NQ`'s check becoming a `const` assertion for exactly this reason —
//! the trap was documented forty lines from where it was walked into again.
//!
//! `tests/feature-matrix.sh` runs this target, so it executes in every cell that arms the feature.
#![cfg(all(feature = "rocm", feature = "corruption-probe"))]
#![allow(clippy::expect_used)] // a panic in a test is loud and correct; the workspace opts back in

use rivoli_engine::fetch::asyncfetch::ScProbe;
use rivoli_engine::probe::{Cols, Folds, NQ, format_row};

// A panic in a test is loud and correct, which is the workspace's stated reason for opting back
// in per file; a panic on the decode path would be a crash in a server, which is why it is
// deny-level everywhere else.

/// `--divergence-folds` parsing, with the two refusals that matter.
///
/// A silently-misparsed fold set is the worst outcome this flag has: every Phase 1 cell is
/// "enable exactly one fold and see whether the pair still diverges", so a typo that quietly
/// enabled NOTHING would make the cell green and be read as "this fold is the mask" — the
/// precise inversion of the truth. Hence unknown names are refused rather than ignored, and the
/// three `sc` forms refuse each other rather than the last one winning.
#[test]
fn folds_parse_refuses_what_it_cannot_honour() {
    // The default is the LIGHT probe — the configuration that produced the coordinate.
    assert_eq!(Folds::default().label(), "light");

    let f = Folds::parse("bh").expect("bh");
    assert!(f.bh && !f.se && f.sc == ScProbe::Off, "bh must not arm sc");
    assert_eq!(f.label(), "bh");

    let f = Folds::parse("sc-decoy").expect("sc-decoy");
    assert_eq!(f.sc, ScProbe::Decoy);
    assert!(!f.bh && !f.se, "an sc variant must not arm the others");
    assert_eq!(f.label(), "sc-decoy");

    // Order-independent, and the label is canonical so two logs of the same config compare
    // equal however the operator typed it.
    assert_eq!(
        Folds::parse("se, bh ,sc-line").expect("spaced").label(),
        Folds::parse("bh,sc-line,se").expect("plain").label()
    );

    assert!(
        Folds::parse("bh,nope").is_err(),
        "unknown names are refused"
    );
    assert!(
        Folds::parse("sc,sc-line").is_err(),
        "two sc variants occupy one pipeline position and must refuse"
    );
    assert!(Folds::parse("sc-nop,sc").is_err(), "in either order");
    assert!(
        Folds::parse("bh,bh").is_err(),
        "a repeat is a spec the operator did not mean"
    );
    assert!(
        Folds::parse("").is_err() && Folds::parse("  ").is_err(),
        "an explicitly EMPTY list must refuse — omit the flag to ask for the light probe"
    );
}

/// **A DISABLED OR ABSENT FOLD PRINTS `-`, NEVER `0`.** The red proof
/// `docs/measurement/gate-red-proofs.md` §5g recorded as OWED, now paid.
///
/// This is the instrument's one unacceptable failure mode and it has already happened once:
/// `xn` was folded on MoE layers only, so GLM's three dense layers carried `0` in both runs of
/// a pair, and a diff reads two equal zeros as "attention agreed" when nothing was measured.
/// A false EXCLUSION is worse than a missing column, because it is indistinguishable from
/// evidence. Every combination that can produce an unmeasured column is enumerated here.
///
/// Note the fold words are deliberately all-nonzero, so a `-` in the output can only come from
/// the rule and never from the data happening to be zero.
#[test]
fn an_unmeasured_column_prints_a_dash_and_never_a_zero() {
    let w: Vec<u64> = (1..=NQ as u64).collect();
    let cols = |miss: u64| {
        Some(Cols {
            gl: 0xA,
            pk: 0xB,
            sl: 0xC,
            miss,
            reloc: 0,
        })
    };
    let all = Folds {
        xa: true,
        ac: true,
        bh: true,
        sc: ScProbe::Full,
        se: true,
    };
    let field = |row: &str, i: usize| row.split_whitespace().nth(i).expect("field").to_string();
    // One helper for "every fetch-path column is unmeasured", because the two sites that assert
    // it differ only in WHY — and rustfmt made them identical text, which the duplication gate
    // then reported.
    let fetch_all_dash = |row: &str, why: &str| {
        for c in [13, 14, 15] {
            assert_eq!(field(row, c), "-", "{why}");
        }
    };
    // Columns: 0 pos 1 nrow 2 layer | 3 xa 4 xn 5 h 6 ac 7 x | 8..12 host | 13 bh 14 sc 15 se.
    let (bh, sc, se, h, ac, xa, xn, x) = (13, 14, 15, 5, 6, 3, 4, 7);

    // A DENSE layer: no router, no pool, no moe_hidden. `xn`/`x` are still measured.
    let r = format_row(7, 1, 3, &w, None, all);
    assert_eq!(field(&r, h), "-", "a dense layer has no h");
    fetch_all_dash(&r, "a dense layer folds nothing on the fetch path");
    assert_ne!(
        field(&r, xn),
        "-",
        "xn is folded on EVERY layer — that was the bug"
    );
    assert_ne!(field(&r, x), "-", "and so is x");
    assert_ne!(field(&r, xa), "-", "xa too: a residual, not a MoE quantity");
    assert_eq!(field(&r, ac), "-", "a dense layer has no MoE accumulator");

    // A MoE layer with NO MISS: nothing was copied, so bh/sc cannot exist; se still can.
    let r = format_row(7, 1, 4, &w, cols(0), all);
    assert_eq!(field(&r, bh), "-");
    assert_eq!(field(&r, sc), "-");
    assert_ne!(
        field(&r, se),
        "-",
        "se is folded on every MoE layer, misses or not"
    );

    // Folds DISABLED by the flag, with a miss present: still `-`, for a different reason.
    let r = format_row(7, 1, 4, &w, cols(1), Folds::default());
    fetch_all_dash(&r, "a fold the run did not enable is not measured");

    // The NO-TOUCH arms hash a DECOY buffer, so their column must not look like a payload hash.
    for arm in [ScProbe::Nop, ScProbe::Decoy] {
        let f = Folds {
            xa: false,
            ac: false,
            bh: false,
            sc: arm,
            se: false,
        };
        assert_eq!(
            field(&format_row(7, 1, 4, &w, cols(1), f), sc),
            "-",
            "{arm:?} hashes a decoy — rendering that as a payload hash is a false exclusion"
        );
    }
    // `sc-line` IS a payload hash, but over ~1/32 of the bytes, so it is marked partial: an
    // agreement there exonerates far less than an agreement under `sc`.
    let line = Folds {
        xa: false,
        ac: false,
        bh: false,
        sc: ScProbe::Line,
        se: false,
    };
    let lr = field(&format_row(7, 1, 4, &w, cols(1), line), sc);
    assert!(
        lr.starts_with('~') && lr.len() > 1,
        "a partial fold must be marked as one, got {lr:?}"
    );

    // ...and when everything IS measured, every column is a hash. Without this the assertions
    // above are satisfied by a formatter that prints `-` unconditionally.
    let r = format_row(7, 1, 4, &w, cols(1), all);
    for c in [h, ac, bh, sc, se] {
        assert_ne!(
            field(&r, c),
            "-",
            "an enabled, applicable fold must print its hash"
        );
    }
}
