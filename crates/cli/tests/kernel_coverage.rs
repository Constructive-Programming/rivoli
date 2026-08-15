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
    // Muse Glimmer (M7) — old:tests/glimmer_{attend,chain,gate,head,norm,...}.rs
    ("gqa_attend", "M7: old:tests/glimmer_attend.rs"),
    ("logit_softcap", "M7: old:tests/glimmer_head.rs"),
    ("rmsnorm_centered_single", "M7: old:tests/glimmer_norm.rs"),
    ("rmsnorm_weightless_batch", "M7: old:tests/glimmer_norm.rs"),
    ("rope_split_half", "M7: old:tests/glimmer_chain.rs"),
    ("sigmoid_gate", "M7: old:tests/glimmer_gate.rs"),
    // DeepSeek-V4-Flash (M8) — old:tests/{f4_kernel,f4_attn,kvcompress_kernel,blockindex_kernel}.rs
    ("act_quant_f4_rotated", "M8: old:tests/kvcompress_kernel.rs"),
    ("act_quant_f8", "M8: old:tests/f4_kernel.rs"),
    ("act_quant_f8_prefix", "M8: old:tests/kvcompress_kernel.rs"),
    ("gather_attn_shared_kv", "M8: old:tests/f4_attn.rs"),
    ("gemv_fp8_bf16", "M8: old:tests/f4_kernel.rs"),
    ("hc_post", "M8: old:tests/f4_kernel.rs"),
    ("hc_pre", "M8: old:tests/f4_kernel.rs"),
    ("index_score_blocks", "M8: old:tests/blockindex_kernel.rs"),
    ("kv_compress_decode", "M8: old:tests/kvcompress_kernel.rs"),
    ("kv_compress_deposit", "M8: old:tests/kvcompress_kernel.rs"),
    ("kv_compress_prefill", "M8: old:tests/kvcompress_kernel.rs"),
    ("moe_expert_range_f4", "M8: old:tests/f4_kernel.rs"),
    ("rmsnorm_single", "M8: old:tests/f4_kernel.rs"),
    ("rope_adjacent", "M8: old:tests/f4_attn.rs"),
    ("swiglu_clamped_bf16", "M8: old:tests/f4_kernel.rs"),
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
    let covered = |name: &str| tests.contains(&format!("{stem}{}", format_args!("{name}(")));

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
