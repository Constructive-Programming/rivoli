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

/// The three sparse selections refuse on every row that still DEFERS them — GLM, Glimmer
/// and K3, hoisted beside [`every_mode_falls_back_loudly`] on its precedent. V4 left this
/// list at M15: its sparse cells fall back loudly onto the native scored selection, and
/// its row test pins that flip cell by cell.
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
///
/// > **NARROWED 2026-08-31 (S1).** This iterated all of [`Arch::ALL`], with a note saying "as
/// > of M9 every architecture HAS an arm — the day a fifth lands armless, restore a named list
/// > here excluding it". That day is today: [`Arch::QwenFlashNext`] refuses every cell, so the
/// > domain is `Arch::ALL` MINUS a named armless list, and the excluded one is asserted from
/// > the other side by [`qwen_flash_next_row_refuses_every_flag_as_of_s1`]. Naming the
/// > EXCLUSION and not the inclusion is deliberate twice over: a hand-written armed list was a
/// > 35-token jscpd clone of `Arch::ALL` (caught on this change's first full build), and it
/// > would have failed OPEN — a sixth architecture would have gone unchecked here instead of
/// > reddening until someone said which side of the line it is on.
#[test]
fn every_architecture_with_an_arm_decodes_with_no_flags_typed() {
    let default_mode = parse_in(&MODES, "--mode", "int3-vq").expect("int3-vq parses");
    let default_attn = parse_in(&ATTNS, "--attn", "dense").expect("dense parses");
    // The architectures with NO engine arm, which is the shorter and the fail-closed half of
    // the statement. `main`'s `the_arms_and_the_legality_table_agree_about_who_can_start` is
    // the cross-check that this exclusion and the arms `main` dispatches to are the same
    // set — including, since S1, its previously unexercised no-arm branch.
    const ARMLESS: [Arch; 1] = [Arch::QwenFlashNext];
    for arch in Arch::ALL.into_iter().filter(|a| !ARMLESS.contains(a)) {
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

/// DeepSeek-V4-Flash's row, cell by cell, as of M15 — the same change-detector as the two
/// above and for the same reason: the total and anti-vacuity checks would not notice a
/// cell being flipped, which is the one edit that changes what the engine accepts.
///
/// **The eight `FallbackLoudly` cells are the load-bearing ones**, and they are two facts
/// rather than one: `--mode` degrades because the checkpoint owns the routed format, and
/// ALL FOUR `--attn` values because the checkpoint owns the attention — a window plus
/// pooled blocks, with the trained-in block indexer ranking the blocks natively since M15.
/// The sparse three were `Refuse` from M8 until then (the indexer was counted-not-placed
/// and the positional stand-in imposed the 2052 ceiling); the flip is recorded here, cell
/// by cell, with [`deepseek_v4`]'s row doc carrying the argument. The no-flags half is
/// [`muse_glimmer`]'s doc and the property test above.
#[test]
fn deepseek_v4_row_is_the_m15_truth() {
    let v4 = |f| decide(Arch::DeepseekV4, f);
    every_mode_falls_back_loudly(Arch::DeepseekV4);
    assert!(
        matches!(v4(Flag::Attn(AttnKind::Dense)), Outcome::FallbackLoudly(_)),
        "--attn dense must fall back loudly: this arm attends a window plus pooled \
         blocks, so `dense` names something the run does not do — and refusing the \
         DEFAULT would break the no-flags invocation"
    );
    // The three sparse values fall back together onto the ONE const, exactly as the three
    // `--mode` values do — the checkpoint's attention is the same whichever was typed.
    for kind in [AttnKind::Streaming, AttnKind::Dsa, AttnKind::Misa] {
        assert_eq!(
            v4(Flag::Attn(kind)),
            Outcome::FallbackLoudly(V4_SPARSE_ATTN_IS_NOT_A_CHOICE),
            "--attn {kind:?} must proceed loudly: the scored selection is native, so a \
             refusal would kill exactly the runs the flag's intent describes"
        );
    }
    assert!(
        V4_SPARSE_ATTN_IS_NOT_A_CHOICE.contains("no context ceiling"),
        "the const must say the 2052 ceiling is gone — its old text cited the refusal"
    );
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

/// Qwen3.8-Flash-Next's row, cell by cell, as of S1 — **the reddening commit's own gate.**
///
/// The other four row tests are change-detectors on a working arm. This one is the opposite
/// claim: that the fifth architecture refuses EVERYWHERE, with the two named consts and not
/// with a sibling's wording, for as long as there is no arm. It is what makes
/// [`every_architecture_with_an_arm_decodes_with_no_flags_typed`]'s narrowed domain honest —
/// the excluded architecture is asserted here rather than merely omitted there.
///
/// Pinned to the CONSTS, so an arm landing without this test moving is a red diff rather than
/// a silently accepted flag; and the fragments are pinned too, because
/// `tests/smoke-qwen.sh` (S6) asserts the CLI's refusal text against the table's own message
/// fragments, and a fragment that can drift here would drift there unnoticed.
///
/// > **CORRECTED 2026-08-31 (S1 review).** The sentence above implied the smoke script can
/// > quote BOTH fragments today. It cannot: `main`'s `requested_flags` puts `Flag::Mode`
/// > first and `check_legality` bails on the first `Refuse`, so a `--mtp` invocation prints
/// > the ARM_NOT_BUILT text and never reaches the `Mtp` cell. This test is therefore the
/// > ONLY reader of [`QWEN_MTP_NOT_LOADED`] until the row flips at S6 — the ordering is
/// > written down beside the const, and the CLI half is gated by `main.rs`'s
/// > `a_qwen_mtp_invocation_is_refused_by_the_mode_cell_first`.
#[test]
fn qwen_flash_next_row_refuses_every_flag_as_of_s1() {
    for flag in Flag::ALL {
        let want = if flag == Flag::Mtp {
            QWEN_MTP_NOT_LOADED
        } else {
            QWEN_ARM_NOT_BUILT
        };
        assert_eq!(
            decide(Arch::QwenFlashNext, flag),
            Outcome::Refuse(want),
            "{} must REFUSE on {} while the arm does not exist — a Support cell would let a \
             run start into a dispatch that bails, and a FallbackLoudly cell would promise a \
             run that never happens",
            flag.spelling(),
            Arch::QwenFlashNext.name()
        );
    }
    // The two fragments the smoke script quotes. `--mtp` is the cell that stays refused after
    // the arm lands, so its reason must name the DRAFT LAYER's absence from the artifact and
    // the recurrence, not a decode-loop increment on a batch shape that already exists.
    assert!(
        QWEN_ARM_NOT_BUILT.contains("no decode path for this architecture yet"),
        "the arm-not-built refusal must say there is no decode path"
    );
    assert!(
        QWEN_MTP_NOT_LOADED.contains("DRAFT LAYER IS NOT IN THE ARTIFACT")
            && QWEN_MTP_NOT_LOADED.contains("gated-delta"),
        "qwen's --mtp reason must name the unconverted draft layer AND the recurrence a \
         verify pass would have to step: quoting another arm's wording tells the user to \
         wait for the wrong thing"
    );
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
