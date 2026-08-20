//! Every `launch_*` under `crates/backend/src/` is exercised by a test — or sits in the
//! DEFERRED table naming the milestone whose oracle files cover it, checked at both ends.
//!
//! Ported from the old tree, where IT EXISTS BECAUSE OF A SPECIFIC MISS: a port tranche
//! reported complete off a subagent's completion rather than the tranche's definition,
//! the suite grew 16 tests to 23 all green, and the two hardest kernels in the batch had
//! never executed once. **Coverage grew while a gap grew faster.** A green suite is not a
//! claim about what is in it.
//!
//! Not feature-gated — the rule is about what the repo contains. This passing featureless
//! means the oracles EXIST, not that they ran; every one needs a device.
//!
//! **The DEFERRED table is this tree's port-transient, not an exemption list.** The old
//! census blessed the mirror case ("kernels land before their launchers — the legitimate
//! transient during a port"); here the launchers landed wholesale with the waist (M1)
//! while each model's oracle files arrive with its milestone. The difference from an
//! exemption list is that both ends are checked: a row dies loudly the moment its
//! launcher stops existing, and dies loudly the moment ANY test covers it — a deferral
//! that outlives its reason is refused, so the table can only shrink.

#![allow(clippy::expect_used, clippy::unwrap_used)] // meta-gate: panic loudly

mod common;

/// Launchers whose oracle files arrive with a later milestone. `arrives` names the
/// milestone and the old-tree oracle file that covers it there, so the port task is
/// written down where the gap is enforced.
const DEFERRED: &[(&str, &str)] = &[
    // (EMPTY again — the cycle's second full turn. First empty 2026-08-16: Muse Glimmer's six
    // rows retired with M7's oracles (`kernel_glimmer_{norm,attend,pointwise}.rs`);
    // DeepSeek-V4-Flash's fifteen retired the same day with M8's
    // (`kernel_v4_{moe,quant,shared_expert,hc,compress,compress_defects,indexer,attend}.rs`).
    // Kimi-K3's seven rows — attn_res, gated_delta_recurrent_f32, mha_attend,
    // moe_expert_range_f4_situ, rmsnorm_gate_heads_f32, short_conv_silu_f32, situ_glu_f32 —
    // were added that day and retired with M9's `kernel_k3_{attn_res,attend,latent,situ,
    // expert_f4,recurrent,conv_norm}.rs`, ported from `k3:tests/k3_kernels.rs`. K3's MLA
    // output gate never had a row: it maps onto the EXISTING `sigmoid_gate` launcher (in
    // place, covered since M7; `kernels/fwd.hip` records the decision), and the k3 tests'
    // out-of-place gate fixture took the one-buffer adaptation in `kernel_k3_attend.rs`. The
    // table can only shrink — within one architecture's port; the next port adds rows and
    // then removes them, and that cycle is the design.)
    //
    // **THIRD TURN, opened AND CLOSED 2026-08-17 by M17c.** The cycle above predicted it — "the
    // next port adds rows and then removes them" — and this turn lasted one commit: the row was
    // added with `gqa_block_attend`'s launcher, and retired the moment
    // `kernel_glimmer_block_attend.rs` landed. **The census refused the stale row rather than
    // letting it stand**, which is the both-ends check doing the one thing an exemption list
    // cannot. EMPTY again, for the third time.)
];

/// Launcher names in `text`, under both declaration forms (hand-written `pub unsafe fn
/// launch_*` and the `launchers!` DSL's `launch_* -> rivoli_*` rows). `decl`/`stem` are
/// built by the caller at runtime so this file's own text cannot trip its scanner.
fn launcher_names(text: &str, decl: &str, stem: &str) -> Vec<String> {
    text.lines()
        .map(str::trim_start)
        .filter_map(|l| {
            l.strip_prefix(decl)
                .and_then(|rest| rest.split('(').next())
                .or_else(|| {
                    l.strip_prefix(stem)
                        .filter(|_| l.contains(" -> rivoli_"))
                        .and_then(|rest| rest.split(" ->").next())
                })
        })
        .map(String::from)
        .collect()
}

/// Every `.rs` under `dir` (workspace-relative), concatenated, read LOUDLY — a missing
/// file is a shrunken corpus and a shrunken corpus is a silent pass. `skip` excludes this
/// file itself from the corpus it searches (its own table text would satisfy any probe),
/// and the exclusion must actually exclude something.
fn corpus(dir: &str, skip: &str) -> String {
    let root = common::repo_root().join(dir);
    let all = common::walk(&root, "rs");
    let kept: Vec<_> = all
        .iter()
        .filter(|p| p.file_name().is_none_or(|n| n != skip))
        .collect();
    assert_eq!(
        all.len() - kept.len(),
        usize::from(!skip.is_empty()),
        "corpus({dir:?}) was told to skip {skip:?} and skipped {} file(s)",
        all.len() - kept.len()
    );
    assert!(!kept.is_empty(), "corpus({dir:?}) is empty — wrong root?");
    kept.iter()
        .map(|p| std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {p:?}: {e}")))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_launcher_has_an_oracle_or_a_live_deferral() {
    let decl = format!("pub unsafe fn {}", "launch_");
    let stem = format!("{}{}", "launch", "_");

    let backend = corpus("crates/backend/src", "");
    let launchers = launcher_names(&backend, &decl, &stem);

    // Anti-vacuity floor, re-derived for this tree: 53 measured 2026-08-15. A parse that
    // silently matches nothing passes forever; this has bitten the old census twice.
    assert!(
        launchers.len() >= 45,
        "found only {} launchers under crates/backend/src (53 on 2026-08-15) — the \
         declaration pattern has changed underneath this scanner",
        launchers.len()
    );

    // The oracle corpus: engine's device tests. No size floor — a truncated corpus only
    // makes launchers look UNCOVERED, which fails loudly by itself.
    let tests = corpus("crates/engine/tests", "");
    // **Two forms count as exercising a launcher, and the second was added 2026-08-17 because a
    // legitimate refactor made real coverage invisible.**
    //
    //  * `launch_x(` — called directly. The original and still the common case.
    //  * `(launch_x,` — handed to a shared launch helper as its FIRST argument. M17c's block
    //    attend and the causal `gqa_attend` take the same thirteen arguments in the same order, so
    //    `build.rs`'s duplication gate refused the second per-file wrapper and both now go through
    //    `common::attend_launch(launch_x, io, dims, scale)`. Under the old pattern alone
    //    `gqa_attend` — covered since M7 — read as UNCOVERED, and this test said so. It was right
    //    to fail: its claim is about the pattern it can see, and the pattern had moved.
    //
    // **The `(` in the second form is load-bearing anti-vacuity, not punctuation.** A bare
    // `launch_x,` would be matched by the `use` list at the top of every oracle file, so importing
    // a launcher would read as testing it — the exact "silently matches everything" failure this
    // file's header warns about, arriving from the opposite direction. An import is preceded by
    // `, ` or `{`, never by `(`.
    //
    // The residual limit, stated rather than discovered later: a helper that took the launcher in
    // any position but FIRST would not be seen. That is a constraint on how a launch helper is
    // written, and it is cheap to honour — `attend_launch` does.
    // Whitespace-stripped, because the value form is otherwise defeated by line breaking: rustfmt
    // puts `attend_launch(` and its first argument on separate lines whenever the call exceeds
    // `fn_call_width`, so `(launch_x,` never appears literally. Measured 2026-08-17 — the census
    // passed with a stale DEFERRED row because of exactly this, which is a FALSE GREEN and the
    // reason the stripping is here rather than a comment about being careful.
    let dense: String = tests.chars().filter(|c| !c.is_whitespace()).collect();
    let covered = |name: &str| {
        let base = format!("{stem}{name}");
        tests.contains(&format!("{base}(")) || dense.contains(&format!("({base},"))
    };

    // Both ends of every deferral, before the census consults it.
    for (name, arrives) in DEFERRED {
        assert!(
            launchers.iter().any(|l| l == name),
            "DEFERRED names `{stem}{name}`, which is not a launcher under \
             crates/backend/src. A row for something that does not exist covers nothing — \
             delete it. (was: {arrives})"
        );
        assert!(
            !covered(name),
            "DEFERRED row for `{stem}{name}` ({arrives}) — but a test now covers it. The \
             deferral has outlived its reason; delete the row so the census owns the claim."
        );
    }

    // `launcher_names` yields BARE names (both declaration forms strip the stem) — the
    // first run of this file compared stem-prefixed strings and the both-ends check
    // refused its own table, which is the check working on its author.
    let missing: Vec<&String> = launchers
        .iter()
        .filter(|l| !covered(l) && !DEFERRED.iter().any(|(n, _)| n == l))
        .collect();

    assert!(
        missing.is_empty(),
        "\n\n{} kernel(s) have a launcher under crates/backend/src and NO oracle under \
         crates/engine/tests, and no DEFERRED row naming their arrival:\n  {}\n\nThey \
         compile and may be dispatched, and nothing has ever checked what they compute. \
         Either port the oracle file or add a DEFERRED row with its milestone.",
        missing.len(),
        missing
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );

    println!(
        "{} launchers: {} with oracles, {} deferred to their milestones",
        launchers.len(),
        launchers.len() - DEFERRED.len(),
        DEFERRED.len()
    );
}
