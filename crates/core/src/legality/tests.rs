//! The legality table's own gates: axis completeness, the full product, anti-vacuity, and
//! one pinned row per architecture. Split from `legality.rs` when the table plus its tests
//! crossed the 800-line cap (`v4_encoding.rs` is the precedent).

#![allow(clippy::expect_used)] // tests: panic-on-failure is the idiom

use super::*;

/// **Every `--mode` value degrades and SAYS so**, on an architecture with no routed
/// format to select — whether because it is dense (Glimmer) or because the checkpoint
/// already chose (V4, K3).
///
/// Not a refusal: [`muse_glimmer`]'s doc carries the argument, and
/// [`every_architecture_with_an_arm_decodes_with_no_flags_typed`] asserts the property.
/// Asserted for the arms through one helper because the property is one property; the
/// rows' own tests still name it, so a reader of any sees the claim.
fn every_mode_falls_back_loudly(arch: Arch) {
    for (name, m) in MODES {
        assert!(
            matches!(decide(arch, Flag::Mode(m)), Outcome::FallbackLoudly(_)),
            "--mode {name} must fall back loudly on {}, not refuse or silently pass: a \
             refusal breaks the no-flags invocation",
            arch.name()
        );
    }
}

/// The three sparse selections refuse on every row that carries them — the loop all four
/// row tests need, hoisted beside [`every_mode_falls_back_loudly`] on its precedent.
fn sparse_selections_refuse(arch: Arch) {
    for kind in [AttnKind::Streaming, AttnKind::Dsa, AttnKind::Misa] {
        assert!(
            matches!(decide(arch, Flag::Attn(kind)), Outcome::Refuse(_)),
            "--attn {kind:?} must refuse on {}",
            arch.name()
        );
    }
}

/// `ordinals` must be exactly `0..n`, each once. This is the guard that makes an
/// `ALL` list's completeness checkable: a forgotten variant either collides with an
/// existing ordinal or leaves a gap, and both read as false here.
fn is_dense_permutation(n: usize, ordinals: impl Iterator<Item = usize>) -> bool {
    let mut seen = vec![false; n];
    let mut count = 0;
    for o in ordinals {
        match seen.get_mut(o) {
            Some(slot) if !*slot => *slot = true,
            _ => return false,
        }
        count += 1;
    }
    count == n
}

/// The product test's domain is only as honest as the two lists it iterates, so the
/// lists are checked first. Dropping an entry is already a compile error (the arrays
/// are fixed-size); what this catches is the shape that still compiles — an entry
/// duplicated or mistyped, which leaves the product covering less than it claims.
#[test]
fn the_axis_lists_are_complete() {
    assert!(
        is_dense_permutation(ARCH_COUNT, Arch::ALL.iter().map(|a| a.ordinal())),
        "Arch::ALL is not every architecture exactly once"
    );
    assert!(
        is_dense_permutation(FLAG_COUNT, Flag::ALL.iter().map(|f| f.ordinal())),
        "Flag::ALL is not every flag exactly once"
    );
}

/// The FULL (arch × flag) product, every cell decided and every non-Support cell
/// carrying a message that names its deferral. An empty or stub message is the
/// failure mode this table exists to prevent — "refused" without a reason is the
/// unhelpful half of a refusal.
#[test]
fn every_arch_flag_cell_is_decided_with_its_reason() {
    let mut cells = 0;
    for arch in Arch::ALL {
        for flag in Flag::ALL {
            let msg = match decide(arch, flag) {
                Outcome::Support => "",
                Outcome::FallbackLoudly(m) | Outcome::Refuse(m) => m,
            };
            assert!(
                msg.is_empty() || msg.len() > 40,
                "{} on {}: a one-word reason is not a reason ({msg:?})",
                flag.spelling(),
                arch.name()
            );
            cells += 1;
        }
    }
    assert_eq!(cells, ARCH_COUNT * FLAG_COUNT, "product iterated short");
}

/// Anti-vacuity. A table that answered `Support` everywhere would pass every check
/// above while deciding nothing, and one that answered `Refuse` everywhere would
/// pass them while refusing the working engine. All three verdicts must have a live
/// cell — including `FallbackLoudly`, which is a real row (`--trace`) rather than a
/// variant kept warm for later; if that row ever goes, so should the variant.
#[test]
fn all_three_verdicts_have_a_live_cell() {
    let all: Vec<Outcome> = Arch::ALL
        .iter()
        .flat_map(|&a| Flag::ALL.iter().map(move |&f| decide(a, f)))
        .collect();
    assert!(all.contains(&Outcome::Support), "nothing is supported");
    assert!(
        all.iter().any(|o| matches!(o, Outcome::Refuse(_))),
        "nothing is refused"
    );
    assert!(
        all.iter().any(|o| matches!(o, Outcome::FallbackLoudly(_))),
        "nothing falls back — delete the variant or restore the row"
    );
}

/// GLM's row, cell by cell, as of M6.
///
/// A deliberate change-detector: the checks above prove the table is TOTAL and not
/// VACUOUS, and neither of them would notice a cell being flipped — which is the one
/// edit that changes what the engine accepts. Pinning the row makes flipping one a
/// decision someone has to record here, next to the argument for it, rather than a
/// diff nothing reads. Update it in the same commit that lands the arm.
#[test]
fn glm_row_is_the_m6_truth() {
    let glm = |f| decide(Arch::GlmMoeDsa, f);
    let refused = |f| matches!(glm(f), Outcome::Refuse(_));
    // The two single-format modes decode; hybrid returns as a FormatPlan.
    assert_eq!(glm(Flag::Mode(Mode::Int3Vq)), Outcome::Support);
    assert_eq!(glm(Flag::Mode(Mode::Int4)), Outcome::Support);
    assert!(refused(Flag::Mode(Mode::Hybrid)), "hybrid must refuse");
    // Dense attention only; the three sparse selections are the post-dense increment.
    assert_eq!(glm(Flag::Attn(AttnKind::Dense)), Outcome::Support);
    sparse_selections_refuse(Arch::GlmMoeDsa);
    assert!(refused(Flag::Mtp), "speculative decode must refuse");
    // --trace is the one degrade-and-say-so cell.
    assert!(matches!(glm(Flag::Trace), Outcome::FallbackLoudly(_)));
    for f in [Flag::CachePolicy, Flag::MaxMem, Flag::Ctx] {
        assert_eq!(glm(f), Outcome::Support, "{} must decode", f.spelling());
    }
}

/// Muse Glimmer's row, cell by cell, as of M7 — the same change-detector as
/// [`glm_row_is_the_m6_truth`] and for the same reason: the total and anti-vacuity
/// checks above would not notice a cell being flipped, which is the one edit that
/// changes what the engine accepts.
///
/// **The `--mode` cells are the load-bearing ones** — `FallbackLoudly`, not `Refuse`:
/// [`muse_glimmer`]'s doc carries the argument, and
/// [`every_architecture_with_an_arm_decodes_with_no_flags_typed`] asserts the property.
#[test]
fn muse_glimmer_row_is_the_m7_truth() {
    let g = |f| decide(Arch::MuseGlimmer, f);
    let refused = |f| matches!(g(f), Outcome::Refuse(_));
    // No routed format exists, so every --mode value degrades and SAYS so.
    every_mode_falls_back_loudly(Arch::MuseGlimmer);
    // Dense attention is what this model DOES, not a placeholder.
    assert_eq!(g(Flag::Attn(AttnKind::Dense)), Outcome::Support);
    sparse_selections_refuse(Arch::MuseGlimmer);
    // The three routed-pool knobs. Each is presence-judged by `main`, so refusing them
    // costs an untyped run nothing.
    assert!(refused(Flag::CachePolicy), "a cyclic scan has one policy");
    assert!(
        refused(Flag::Trace),
        "there is no routed-expert stream to trace"
    );
    assert!(refused(Flag::Mtp), "speculative decode must refuse");
    // The two numbers the partition is a function of.
    for f in [Flag::MaxMem, Flag::Ctx] {
        assert_eq!(g(f), Outcome::Support, "{} must decode", f.spelling());
    }
}

/// The claim the row above rests on, stated as the property rather than as cell
/// values: **an architecture that has an engine arm must decode with NO flags typed.**
///
/// `main` submits `Flag::Mode` and `Flag::Attn` on their resolved defaults and nothing
/// else, so a `Refuse` on either default is a model that cannot be run without knowing
/// which flag to work around. GLM has satisfied this since M4 by accident of every
/// default being `Support`; Glimmer satisfies it by a deliberate `FallbackLoudly`, and
/// the next arm will meet the same bar or fail here.
///
/// The defaults are spelled from [`MODES`]/[`ATTNS`] rather than as variants, because
/// what is being pinned is the pair `main`'s clap attributes actually resolve to.
#[test]
fn every_architecture_with_an_arm_decodes_with_no_flags_typed() {
    let default_mode = parse_in(&MODES, "--mode", "int3-vq").expect("int3-vq parses");
    let default_attn = parse_in(&ATTNS, "--attn", "dense").expect("dense parses");
    // Every architecture, because as of M9 every architecture HAS an arm — the day a
    // fifth lands armless, restore a named list here excluding it (and `main`'s ARMS
    // test is the cross-check that the two lists agree).
    for arch in Arch::ALL {
        for flag in [Flag::Mode(default_mode), Flag::Attn(default_attn)] {
            assert!(
                !matches!(decide(arch, flag), Outcome::Refuse(_)),
                "{} refuses {} — the default invocation `rivoli DIR --bench N` cannot \
                 start on an architecture that HAS an engine arm",
                arch.name(),
                flag.spelling()
            );
        }
    }
}

/// DeepSeek-V4-Flash's row, cell by cell, as of M8 — the same change-detector as the two
/// above and for the same reason: the total and anti-vacuity checks would not notice a
/// cell being flipped, which is the one edit that changes what the engine accepts.
///
/// **The five `FallbackLoudly` cells are the load-bearing ones**, and they are two facts
/// rather than one: `--mode` degrades because the checkpoint owns the routed format, and
/// `--attn dense` because this architecture attends a window plus pooled blocks. The
/// no-flags argument is [`muse_glimmer`]'s doc and the property test above.
#[test]
fn deepseek_v4_row_is_the_m8_truth() {
    let v4 = |f| decide(Arch::DeepseekV4, f);
    every_mode_falls_back_loudly(Arch::DeepseekV4);
    assert!(
        matches!(v4(Flag::Attn(AttnKind::Dense)), Outcome::FallbackLoudly(_)),
        "--attn dense must fall back loudly: this arm attends a window plus pooled \
         blocks, so `dense` names something the run does not do — and refusing the \
         DEFAULT would break the no-flags invocation"
    );
    sparse_selections_refuse(Arch::DeepseekV4);
    // The MTP refusal is V4's OWN, not the shared deferral: a user told to wait for a
    // draft head would be waiting for the wrong thing. Pinned to the named const, so
    // quoting GLM's wording is a red diff here rather than a drift.
    assert_eq!(v4(Flag::Mtp), Outcome::Refuse(V4_MTP_NEEDS_A_KERNEL));
    assert!(
        V4_MTP_NEEDS_A_KERNEL.contains("KERNEL"),
        "V4's --mtp reason must name the kernel"
    );
    // --trace degrades rather than refusing: the capture is still written and its decode
    // half is faithful.
    assert!(matches!(v4(Flag::Trace), Outcome::FallbackLoudly(_)));
    // The three knobs that are MORE real here than on GLM — the routed set cannot fit.
    for f in [Flag::CachePolicy, Flag::MaxMem, Flag::Ctx] {
        assert_eq!(v4(f), Outcome::Support, "{} must decode", f.spelling());
    }
}

/// Kimi-K3's row, cell by cell, as of M9 — the same change-detector as the three above
/// and for the same reason: the total and anti-vacuity checks would not notice a cell
/// being flipped, which is the one edit that changes what the engine accepts.
///
/// **The `--mode` cells are the load-bearing ones**, exactly as on V4 — the no-flags
/// argument is [`muse_glimmer`]'s doc and the property test above. `--attn dense` is
/// `Support` on Glimmer's precedent (the row's own doc carries the argument), and
/// `--trace` is plain `Support` — the one cell where this arm is strictly simpler than
/// GLM, because a token-sequential prefill has nothing to fall back FROM.
#[test]
fn kimi_k3_row_is_the_m9_truth() {
    let k3 = |f| decide(Arch::KimiK3, f);
    every_mode_falls_back_loudly(Arch::KimiK3);
    assert_eq!(k3(Flag::Attn(AttnKind::Dense)), Outcome::Support);
    sparse_selections_refuse(Arch::KimiK3);
    // The MTP refusal is K3's OWN and must not quote either sibling's: GLM waits on a
    // draft head, V4 on one kernel — a K3 user waits on TWO kernels, and the recurrence
    // is the one no other arm has. Pinned to the named const.
    assert_eq!(k3(Flag::Mtp), Outcome::Refuse(K3_MTP_NEEDS_TWO_KERNELS));
    assert!(
        K3_MTP_NEEDS_TWO_KERNELS.contains("KDA"),
        "K3's --mtp reason must name the recurrence"
    );
    // --trace: Support, not GLM's FallbackLoudly — flipping this cell to a fallback
    // would warn about a token-major degrade this arm cannot even perform.
    assert_eq!(k3(Flag::Trace), Outcome::Support);
    // The three knobs, of which --ctx is the header's own example of per-arch variance:
    // it sizes the 24 MLA caches and the KDA state not at all.
    for f in [Flag::CachePolicy, Flag::MaxMem, Flag::Ctx] {
        assert_eq!(k3(f), Outcome::Support, "{} must decode", f.spelling());
    }
}

/// Every spelling parses back to the variant it names, which is what makes
/// [`name_in`]'s `"?"` fallback unreachable and lets `--dump-ids` headers be compared
/// against command lines.
#[test]
fn vocabularies_round_trip() {
    for (name, m) in MODES {
        assert_eq!(parse_in(&MODES, "--mode", name), Ok(m));
        assert_eq!(name_in(&MODES, m), name);
    }
    for (name, a) in ATTNS {
        assert_eq!(parse_in(&ATTNS, "--attn", name), Ok(a));
        assert_eq!(name_in(&ATTNS, a), name);
    }
    assert!(parse_in(&MODES, "--mode", "int3").is_err());
}
