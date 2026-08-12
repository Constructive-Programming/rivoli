//! Every `launch_*` under `src/backend/` must be exercised by a test.
//!
//! IT EXISTS BECAUSE OF A SPECIFIC MISS: tranche 2a of the Vulkan port ported six kernels,
//! delegated three oracles, and reported the tranche complete off the subagent's completion
//! rather than off the tranche's definition. The suite went 16 tests to 23, every one
//! passed, and `gemv_i8` and `gemv_fp8` — the two hardest kernels in the batch, carrying the
//! e4m3 LUT, a `blk_shift` port, 64-bit addressing and a split-K path — had never executed
//! once.
//!
//! **Coverage grew while a gap grew faster.** The count moved in the reassuring
//! direction while the fraction covered fell, which is why "we added tests" is exactly
//! the evidence someone would cite to argue the opposite. A green suite is not a claim
//! about what is in it.
//!
//! A remembered checklist does not survive the next tranche at eleven at night. This
//! does: port a kernel, forget its oracle, and the build fails with the kernel named.
//!
//! Not feature-gated — the rule is about what the repo contains, not about what someone
//! compiled. Note the consequence, because it is easy to over-read a green run: this test
//! passing under a featureless build means the oracles EXIST, not that they ran. Every one
//! of them is `#![cfg(feature = "rocm")]` and needs a device.
//!
//! > **RE-KEYED 2026-08-06, from `src/backend/vk.rs` onto `src/backend/`.** The paragraphs
//! > above are the Vulkan port's history, kept verbatim: they are the argument for the rule,
//! > not a description of the file it happened to police first. That backend was retired the
//! > same day and this census went with it, so for one commit no launcher in the tree had a
//! > census at all.
//! >
//! > On arrival here it found **18 of 48 launchers with no oracle** — `src/backend/hip.rs`
//! > had simply never been scanned. Three places had already RECORDED that gap in prose
//! > (`src/backend/hip.rs`'s V4 launcher note, `tests/f4_loop.rs`, `tests/common/mod.rs`'s
//! > `absent`) and not one of them was a gate, which is the same lesson one turn further on:
//! > a written-down gap is not an enforced one.
//! >
//! > It was "the TENTH MECHANISED RULE" here until the re-key. That ordinal indexed the
//! > Vulkan port's numbered rule table (`docs/investigations/vulkan-kernels.md`), which is
//! > archived and no longer maintained, so it is dropped rather than renumbered against a
//! > list that no longer counts.
//!
//! KNOWN LIMIT: A KERNEL WITH NO LAUNCHER IS INVISIBLE HERE. This check is keyed on
//! `pub unsafe fn launch_*` under `src/backend/`, so an `extern "C"` kernel in `kernels/`
//! that exists, compiles, and links while having no launcher at all is not counted as
//! uncovered — it is not counted at all.
//!
//! Deliberately not fixed, because that state is the legitimate transient during a port:
//! kernels land before their launchers, and a check that failed on it would fire on every
//! honest checkpoint until the tranche closed. Keying on `kernels/*.hip` instead would
//! trade a silent gap for a noisy one, and the V4-Flash port is live.
//!
//! SECOND KNOWN LIMIT, and it is the one to distrust: **this counts a NAME, not an
//! assertion.** A test that launches a kernel and checks nothing satisfies it. The census is
//! the floor — it says no launcher is unexamined — and the oracles in the files it scans are
//! what make that mean something.

mod common;

/// A launcher a test drives through an ENGINE ENTRY POINT rather than by name.
///
/// **Not an exemption list, and the difference is that both ends are checked.** An
/// exemption asserts nothing and rots silently; the original census refused to carry one
/// for exactly that reason, and the reason still holds. Each entry fails three ways: if
/// `launched_by` stops launching one of them, if no test calls `entry`, or if a name is not
/// a launcher at all.
///
/// The three compressor launchers are here because `tests/kvcompress_kernel.rs` scores
/// them as ONE unit against S1b's oracle over four cells with exact defect impersonation,
/// which is a stronger claim than three by-name launches would be — `kvcompress::compress`
/// picks which of the three runs from the geometry and the position, and that dispatch is
/// itself part of what the oracle checks. Splitting them apart to satisfy a string match
/// would test less.
///
/// A struct rather than three `&str` in a row, following `common`'s `Mla`/`Att`/`MoeRange`:
/// three interchangeable strings is the same shape as the "six bare `usize`, every one of
/// them plausible in any other's position" those carry their argument for.
struct Indirect {
    launchers: &'static [&'static str],
    launched_by: &'static str,
    entry: &'static str,
}

const INDIRECT: &[Indirect] = &[Indirect {
    launchers: &[
        "kv_compress_deposit",
        "kv_compress_prefill",
        "kv_compress_decode",
    ],
    launched_by: "src/kvcompress.rs",
    entry: "kvcompress::compress(",
}];

/// Every launcher name declared in `backend`, in both declaration forms.
///
/// Factored out when the OWNERS census below became a second reader of it — `build.rs`'s
/// jscpd gate rejected the copy, which is the gate doing its job: two parsers of the same
/// convention drift, and the one that drifts silently is the one whose failure mode is a
/// pass. `decl` and `stem` stay parameters rather than literals here for the reason the
/// caller builds them at runtime: a literal `pub unsafe fn launch_` in this file would make
/// the file look like a launcher declaration to its own scanner.
fn launcher_names(backend: &str, decl: &str, stem: &str) -> Vec<String> {
    backend
        .lines()
        .map(str::trim_start)
        .filter_map(|l| {
            l.strip_prefix(decl)
                .and_then(|rest| rest.split('(').next())
                .or_else(|| {
                    // `launch_<name> -> rivoli_<sym>, "<tag>" (`
                    l.strip_prefix(stem)
                        .filter(|_| l.contains(" -> rivoli_"))
                        .and_then(|rest| rest.split(" ->").next())
                })
        })
        // `String::from`, not `str::to_string` — the latter made this function's last five
        // lines a token-for-token match with `tests/matrix.rs`'s array parser, which ends the
        // same way and is followed by the same `read_to_string` helper shape. jscpd counts a
        // shared SUFFIX as a clone (`src/backend/hip.rs` records the same effect on parameter
        // lists), and there is nothing real to factor between two unrelated parsers.
        .map(String::from)
        .collect()
}

/// Read a source file relative to the crate root.
///
/// PANICS on a missing file rather than returning empty. That is deliberate: this scanner
/// and `tests/invariants.rs` both derive their coverage from paths, and a silent empty read
/// turns "nothing to check" into a PASS. Moving `vk.rs` into `src/backend/` on 2026-07-31
/// broke both; this one failed loudly because of this panic, the other reported every
/// invariant as untested. Prefer loud.
fn source(p: &std::path::Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Every `.rs` under `dir`, concatenated, read LOUDLY — a dropped file is a shrunken corpus,
/// which is [`source`]'s argument one directory up.
///
/// WALKED, not listed. The Vulkan census read ONE file (`tests/vk.rs`) because that
/// backend's oracles all lived in it; `hip.rs`'s are spread over a dozen, and a
/// hand-maintained list of them is the failure `common::walk` already carries the
/// regression for.
///
/// `skip` drops one file by name, and it has exactly one caller and one reason:
/// **this file is inside `tests/`, so it is inside the corpus it searches.** Every probe
/// below is a string that also appears here — `INDIRECT.entry` verbatim, and
/// `launch_<name>(` if anyone ever writes one — so without the exclusion the entry hop is
/// satisfied by the table that declares it and can never fire. That is not hypothetical:
/// it shipped in the first draft of this restoration and a review caught it. Excluding the
/// file is the fix that covers every future probe, where obfuscating each literal covers
/// only the ones someone remembered.
fn corpus(dir: &str, skip: &str) -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(dir);
    let all = common::walk(&root, "rs");
    let kept: Vec<_> = all
        .iter()
        .filter(|p| p.file_name().is_none_or(|n| n != skip))
        .collect();
    // The exclusion must actually EXCLUDE something. `skip` is a hard-coded copy of a file
    // name; rename or move that file and the filter matches nothing, silently, and the
    // self-satisfying `entry` hop comes back with no test failure — a guard whose own
    // failure mode is a pass, which is the thing this file exists to argue against.
    assert_eq!(
        all.len() - kept.len(),
        usize::from(!skip.is_empty()),
        "corpus({dir:?}) was told to skip {skip:?} and skipped {} file(s). A skip that \
         matches nothing puts this file back inside the corpus it searches.",
        all.len() - kept.len()
    );
    kept.iter()
        .map(|p| source(p))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_launcher_has_an_oracle() {
    // Built at runtime, so this file's own text cannot trip the scan: a literal
    // `pub unsafe fn launch_` here would make this file look like a launcher declaration
    // to its own reader.
    let decl = format!("pub unsafe fn {}", "launch_");

    // TWO declaration forms since 2026-08-06, and the census must see both.
    //
    // Track 2 replaced 41 of the 47 hand-written launcher/extern pairs with `launchers! { … }`
    // declarations, where a launcher reads `launch_gemv_vq -> rivoli_gemv_vq, "gemv_vq" (`.
    // Six remain hand-written (a `bool` argument, three `&T`-to-`*const T` coercions, one
    // symbol with no matching extern, one destructuring body) — so the old pattern is still
    // live and dropping it would silently shrink the denominator by six.
    //
    // **This census is why the macro's DSL spells `launch_*` out instead of pasting it from a
    // stem.** A `paste!`-built ident would put the name nowhere in the source, and the only
    // thing that noticed would be this floor — which is exactly what happened when the batch
    // conversion first ran: 6 found against a floor of 40, failing loudly. That failure was
    // the check working. It is recorded here because the tempting repair is to lower the
    // floor, and the correct one was to make the names greppable again.
    let stem = format!("{}{}", "launch", "_");

    // The whole backend directory, not one file. Keying on the single path
    // `src/backend/vk.rs` is what let the last move break this census, and a launcher that
    // relocated into a sibling module would leave the census silently — the denominator
    // shrinking is precisely what the rule exists to notice.
    let backend = corpus("src/backend", "");
    let launchers = launcher_names(&backend, &decl, &stem);

    // Anti-vacuity: a parse that silently matches nothing passes forever. This has bitten
    // twice already — once when a filter skipped the only file it existed to police, and
    // once when a naming convention changed underneath a scanner.
    //
    // The floor is 40 against a MEASURED 46 on 2026-08-09 (47 until `launch_moe_gate_v4` went
    // with the device router; 48 until `launch_vaxpy`, which this census found uncovered and
    // which turned out to be dead, was deleted 2026-08-06). Re-derived rather than left at the
    // old number: this file's own argument is that not re-deriving is how the guard goes
    // vacuous, and a count that drifts silently under a floor it never approaches is the
    // slowest possible version of that. It was 5 when this scanned
    // `vk.rs`, which had ~16; carried over unexamined it would have tolerated a parse that
    // found a tenth of them, and re-keying a check onto a 3x larger subject without
    // re-deriving its floor is how an anti-vacuity guard quietly becomes vacuous.
    assert!(
        launchers.len() >= 40,
        "found only {} launchers under src/backend/ (46 on 2026-08-09) — the declaration \
         pattern has changed and this check is no longer examining what it claims to",
        launchers.len()
    );

    // There is deliberately NO floor on the corpus size to match. A truncated corpus can
    // only make launchers look UNCOVERED, so it fails loudly by itself; there is no
    // silent-green path for a bound to guard, and the one drafted here was set at 100 KB
    // against a real 817 KB — it would have passed on losing 88% of `tests/`. `source`
    // panicking on a bad read is what actually covers the case.
    let tests = corpus("tests", "kernel_coverage.rs");
    let has_direct_oracle = |name: &str| tests.contains(&format!("launch_{name}("));

    // Both ends of every indirection, before the census consults it. Checked first so a
    // rotted row is reported as a rotted row rather than as a missing oracle.
    for ind in INDIRECT {
        assert!(
            tests.contains(ind.entry),
            "the indirect-coverage entry `{}` is called by no test. Whatever it stood for \
             is not exercised.",
            ind.entry
        );
        for name in ind.launchers {
            assert!(
                launchers.iter().any(|l| l == name),
                "the indirect-coverage row names `{}{name}`, which is not a launcher under \
                 src/backend/. A row for something that does not exist is inert — it \
                 cannot fail, and it cannot cover anything either.",
                "launch_"
            );
            assert!(
                source(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(ind.launched_by))
                    .contains(&format!("launch_{name}(")),
                "the indirect-coverage row for {}{name} names {}, which no longer launches \
                 it. The row is stale: find where it moved, or delete the row and let the \
                 census demand a direct oracle.",
                "launch_",
                ind.launched_by
            );
        }
    }
    // Deliberately NOT asserted: that an indirectly-covered launcher has no direct oracle
    // too. A draft did, on the grounds that two spellings of one claim is one that can rot
    // — but it turns the build RED when coverage IMPROVES, and a census is a floor.
    let indirect: Vec<&str> = INDIRECT.iter().flat_map(|i| i.launchers).copied().collect();

    let missing = common::absent(&launchers, |name| {
        has_direct_oracle(name) || indirect.contains(&name)
    });

    assert!(
        missing.is_empty(),
        "\n\n{} kernel(s) have a launcher under src/backend/ and NO oracle under \
         tests/:\n  {}\n\n\
         They compile, they may even be dispatched by other code, and nothing has ever \
         checked what they compute. A passing suite says nothing about them.\n\n\
         Oracles are filed by the kernel source the launcher wraps: kernels/fwd.hip -> \
         tests/fwd_kernel.rs, indexer.hip -> tests/indexer_kernel.rs, linalg.hip and \
         mla.hip and moe.hip -> tests/kernel.rs, kvcompress.hip -> \
         tests/kvcompress_kernel.rs, blockindex.hip -> tests/blockindex_kernel.rs, \
         headtail.hip -> tests/headtail.rs, and the .f4 engine's mla.hip/linalg.hip/\
         attn.hip launchers -> tests/f4_kernel.rs and tests/f4_attn.rs. Two live \
         exceptions, so you are not misled into moving them: \
         index_topk's oracle predates tests/indexer_kernel.rs and stays in \
         tests/kernel.rs, and mla.hip's qk_norm is scored in tests/headtail.rs \
         beside its sibling rmsnorm_batch. ({} has no exemption list — the empty one it used \
         to carry only invited parking work here. `INDIRECT` is not one: every hop is \
         asserted.)\n",
        missing.len(),
        missing
            .iter()
            .map(|n| format!("launch_{n}"))
            .collect::<Vec<_>>()
            .join("\n  "),
        file!()
    );

    println!(
        "{} launchers, all exercised ({} of them through an engine entry point)",
        launchers.len(),
        indirect.len()
    );
}

/// Which engine source files launch each kernel — the model affiliation the `v4_` prefixes
/// carried until they were renamed for behaviour on 2026-08-09.
///
/// **This exists because the same information, written as a comment in `src/backend/hip.rs`,
/// was wrong on SIX of its entries the day it was written.** Review caught `swiglu` filed as
/// GLM-only (`f4gpu.rs` calls it — and that call is the tree's most-documented open defect),
/// `swiglu_clamped_bf16` credited with a V4 caller it does not have, and `act_quant_f8`,
/// `vadd` and `flag_nonfinite` each under the wrong engine. The naming principle this
/// refactor serves says to move the model list into the comments; a list nothing checks is
/// how that trade turns into a net loss, so it moved here instead.
///
/// `gpu.rs` is the GLM-5.2 decode path (`.vq3`/`.i4`); `f4gpu.rs`, `attn.rs` and
/// `kvcompress.rs` are DeepSeek-V4-Flash-0731's (`.f4`). An empty slice asserts the launcher
/// has NO engine caller, which is a real and interesting state — those are staged work, not
/// dead code, and saying so is the point.
///
/// > **CORRECTED 2026-08-11.** This said "three of them are staged work". There were **five**
/// > before `gqa_attend` was added and six after, so the count had already drifted twice with
/// > nothing to notice. It is dropped rather than re-counted: the rows themselves are the
/// > record, and a hand-written tally beside a list that grows is a defect waiting for its
/// > next edit.
const OWNERS: &[(&str, &[&str])] = &[
    ("act_quant_f8", &["f4gpu.rs"]),
    ("act_quant_f8_prefix", &["attn.rs", "kvcompress.rs"]),
    ("act_quant_f4_rotated", &["kvcompress.rs"]),
    ("append_kv", &["gpu.rs"]),
    ("argmax", &["gpu.rs", "f4gpu.rs"]),
    ("attend", &["gpu.rs"]),
    ("embed_bf16_row_bcast", &["f4gpu.rs"]),
    ("embed_i8_row", &["gpu.rs"]),
    ("flag_nonfinite", &["gpu.rs"]),
    ("gather_attn_shared_kv", &["attn.rs"]),
    ("gather_rope", &["gpu.rs"]),
    ("gemm_bf16", &["kvcompress.rs", "f4gpu.rs"]),
    ("gemv_f32", &["gpu.rs", "f4gpu.rs"]),
    ("gemv_fp8", &["gpu.rs"]),
    ("gemv_fp8_bf16", &["attn.rs", "f4gpu.rs"]),
    ("gemv_i4", &[]),
    ("gemv_i8", &["gpu.rs"]),
    ("gemv_vq", &[]),
    // Muse Glimmer's GQA attend. No engine caller yet BY CONSTRUCTION: S2 gates each kernel
    // against the S1b goldens before S3 writes the layer loop that calls it, so this row is
    // empty for exactly as long as that stage lasts.
    ("gqa_attend", &[]),
    ("hc_head_collapse", &["f4gpu.rs"]),
    ("hc_post", &["f4gpu.rs"]),
    ("hc_pre", &["f4gpu.rs"]),
    ("index_append", &["gpu.rs"]),
    ("index_head_route", &["gpu.rs"]),
    ("index_pool_push", &["gpu.rs"]),
    ("index_score", &["gpu.rs"]),
    ("index_score_blocks", &[]),
    ("index_topk", &["gpu.rs"]),
    ("kv_compress_decode", &["kvcompress.rs"]),
    ("kv_compress_deposit", &["kvcompress.rs"]),
    ("kv_compress_prefill", &["kvcompress.rs"]),
    ("layernorm", &["gpu.rs"]),
    ("mla_absorb_fp8", &["gpu.rs"]),
    ("mla_value_fp8", &["gpu.rs"]),
    ("moe_acc_drain", &["gpu.rs", "f4gpu.rs"]),
    // **No engine caller yet, and that is the interesting part.** Kimi-K3's MoE block is the only
    // one whose routed sum is intercepted in a latent rather than folded into the residual, so this
    // kernel exists ahead of the layer loop that will call it (S1a item 4, S3 wires it). The empty
    // slice is the assertion — see this table's header: it says "no engine caller" out loud rather
    // than letting the absence read as an oversight. Its oracle is
    // `tests/kernel.rs::moe_acc_drain_to_writes_the_latent_aggregate_and_resets`, which is what
    // keeps a staged kernel from being an unexecuted one.
    ("moe_acc_drain_to", &[]),
    ("moe_expert_range", &["gpu.rs"]),
    ("moe_expert_range_f4", &["f4gpu.rs"]),
    ("moe_expert_range_i4", &["gpu.rs"]),
    ("qk_norm", &["attn.rs"]),
    ("rmsnorm_batch", &["attn.rs", "f4gpu.rs"]),
    ("rmsnorm_single", &["gpu.rs"]),
    ("rope_adjacent", &["attn.rs"]),
    ("rope_interleave", &["gpu.rs"]),
    // Muse Glimmer's rotation convention. Empty for the same reason `gqa_attend` is: S2 gates
    // each kernel against the S1b goldens before S3 writes the layer loop that calls it.
    ("rope_split_half", &[]),
    // Muse Glimmer's attention output gate. Empty for the same reason as its two siblings.
    ("logit_softcap", &[]),
    ("sigmoid_gate", &[]),
    ("swiglu", &["gpu.rs", "f4gpu.rs"]),
    ("swiglu_clamped_bf16", &[]),
    ("vadd", &["gpu.rs"]),
    ("vq_encode", &["bin/convert.rs"]),
];

/// Every launcher appears in [`OWNERS`] exactly once, and every claim in it is true of the
/// tree — both directions, because a list that only checks the files it names cannot notice
/// a caller appearing somewhere new, which is precisely how the prose version went stale.
#[test]
fn every_launcher_is_attributed_to_the_paths_that_call_it() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let backend = corpus("src/backend", "");
    let decl = format!("pub unsafe fn {}", "launch_");
    let stem = format!("{}{}", "launch", "_");
    let mut launchers = launcher_names(&backend, &decl, &stem);
    launchers.sort();
    launchers.dedup();

    let listed: Vec<String> = OWNERS.iter().map(|(n, _)| n.to_string()).collect();
    let mut sorted = listed.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), OWNERS.len(), "OWNERS has a duplicate name");
    assert_eq!(
        sorted, launchers,
        "OWNERS and the launcher set disagree. Every launcher needs a row and every row needs \
         a launcher — a name that is neither is the stale-exemption failure this file exists \
         to refuse."
    );

    // Which files ACTUALLY call each launcher, comments excluded. A doc comment naming
    // `launch_swiglu_clamped_bf16(...)` is what made the prose version credit it with a
    // caller it does not have, so a line starting `//` is not a call site.
    for (name, owners) in OWNERS {
        let mut actual: Vec<String> = Vec::new();
        for p in common::walk(&root, "rs") {
            if p.ends_with("backend/hip.rs") {
                continue;
            }
            let rel = p
                .strip_prefix(&root)
                .unwrap_or_else(|e| panic!("{} is not under {}: {e}", p.display(), root.display()))
                .to_string_lossy()
                .to_string();
            let calls = source(&p).lines().any(|l| {
                let t = l.trim_start();
                !t.starts_with("//") && t.contains(&format!("{stem}{name}("))
            });
            if calls {
                actual.push(rel);
            }
        }
        actual.sort();
        let mut want: Vec<String> = owners.iter().map(|s| s.to_string()).collect();
        want.sort();
        assert_eq!(
            want, actual,
            "OWNERS says launch_{name} is called from {want:?}, the tree says {actual:?}"
        );
    }
    println!(
        "{} launchers attributed, all verified against the tree",
        OWNERS.len()
    );
}
