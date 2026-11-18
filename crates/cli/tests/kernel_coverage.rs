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
//!
//! **Two censuses, two questions.** The oracle census above asks *"does a test exercise
//! this launcher?"*; the liveness census (2026-09-10) asks *"does anything that ships
//! call it?"* — a reference under `crates/engine/src` or `crates/cli/src`, where test
//! files never count. A launcher with no production caller is refused unless it is a
//! **classified row** in `docs/reference/kernel-inventory.md`'s uncalled table, and that
//! table is checked at both ends exactly like DEFERRED: a row dies loudly when its
//! launcher stops existing, and dies loudly the moment production code calls it.

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
    // **FOURTH TURN: not opened at all, 2026-09-01 by the qwen S3 track — and the three-edge
    // proof was run anyway.** Three launchers landed (`gdn_recurrent_f32`,
    // `index_pool_norm_f32`, `gated_residual_collapse_f32`), each with its oracle suite in the
    // same commit, so the table stays EMPTY and the count went 62/62/0 -> 65/65/0. Because no row
    // was owed, this file's own three arms were proven by plant instead, each observed to change
    // the tree (`cmp` rc 1 against a saved copy) and to redden with ITS OWN message, then reverted
    // and observed green on a run that recompiled:
    //
    //  * a row naming `launch_gdn_recurrent_prefill_chunked_f32`, which is not a launcher ->
    //    `:148` *"DEFERRED names `launch_gdn_recurrent_prefill_chunked_f32`, which is not a
    //    launcher under crates/backend/src"*;
    //  * a row naming `gdn_recurrent_f32`, which IS covered -> `:154` *"DEFERRED row for
    //    `launch_gdn_recurrent_f32` … but a test now covers it"*;
    //  * renaming the `launch_gated_residual_collapse_f32(` call in its suite so the launcher reads
    //    as uncovered -> `:168` *"1 kernel(s) have a launcher under crates/backend/src and NO
    //    oracle under crates/engine/tests"*, naming `gated_residual_collapse_f32`.
    //
    // The third is the arm that matters most and the one an empty table never exercises: it is the
    // only one that fires when a launcher lands with no oracle AND no row, which is the miss this
    // file exists for.
    //
    // **THIRD TURN, opened AND CLOSED 2026-08-17 by M17c.** The cycle above predicted it — "the
    // next port adds rows and then removes them" — and this turn lasted one commit: the row was
    // added with `gqa_block_attend`'s launcher, and retired the moment
    // `kernel_glimmer_block_attend.rs` landed. **The census refused the stale row rather than
    // letting it stand**, which is the both-ends check doing the one thing an exemption list
    // cannot. EMPTY again, for the third time.)
];

/// The two arrows a `launchers!` row may carry: `->` (a hipcc launcher, declared `extern`)
/// and `=>` (a Rust launcher, `src/rk_launchers.rs`, since 2026-11-18).
///
/// **The `=>` form was added because the scanner went blind without it, and only half the
/// census noticed.** With the 13 activation/residual rows moved to `=>` and this list still
/// `["->"]`, the oracle census PASSED at 53 launchers — thirteen fewer, under a floor of 45 —
/// and the liveness census reddened only because one of the thirteen
/// (`gated_residual_collapse_f32`) happens to carry an uncalled-table row (measured
/// 2026-11-18, `docs/measurement/gate-red-proofs.md` §18). Hence also the floor below.
const ARROWS: [&str; 2] = [" -> rivoli_", " => rivoli_"];

/// Launcher names in `text`, under both declaration forms (hand-written `pub unsafe fn
/// launch_*` and the `launchers!` DSL's `launch_* -> rivoli_*` / `launch_* => rivoli_*`
/// rows). `decl`/`stem` are built by the caller at runtime so this file's own text cannot
/// trip its scanner.
fn launcher_names(text: &str, decl: &str, stem: &str) -> Vec<String> {
    text.lines()
        .map(str::trim_start)
        .filter_map(|l| {
            l.strip_prefix(decl)
                .and_then(|rest| rest.split('(').next())
                .or_else(|| {
                    let rest = l.strip_prefix(stem)?;
                    let arrow = ARROWS.iter().find(|a| rest.contains(*a))?;
                    rest.split(*arrow).next()
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

/// The launcher population plus the probe stem, shared by both censuses so they cannot
/// disagree about what the backend contains.
fn backend_launchers() -> (String, Vec<String>) {
    let decl = format!("pub unsafe fn {}", "launch_");
    let stem = format!("{}{}", "launch", "_");
    let launchers = launcher_names(&corpus("crates/backend/src", ""), &decl, &stem);

    // Anti-vacuity floor: 66 measured 2026-11-18 (53 on 2026-08-15, under a floor of 45 that
    // then sat 21 BELOW the population — low enough that losing every Rust launcher at once
    // passed, see `ARROWS`). A floor is only a check while it sits AT the population; raise
    // it with each lane that adds launchers, never let the slack regrow.
    assert!(
        launchers.len() >= 66,
        "found only {} launchers under crates/backend/src (66 on 2026-11-18) — the \
         declaration pattern has changed underneath this scanner",
        launchers.len()
    );
    (stem, launchers)
}

/// `plain` cites `base` in either form: a direct call `base(`, or `(base,` — handed to a
/// shared launch helper as its FIRST argument, probed against `dense` (the caller's
/// whitespace-stripped view of the same corpus) because rustfmt line-breaks long calls.
/// The block comment at the oracle census's call site argues both forms and why the `(`
/// is load-bearing.
fn references(plain: &str, dense: &str, base: &str) -> bool {
    plain.contains(&format!("{base}(")) || dense.contains(&format!("({base},"))
}

/// One name per line, indented — the shape both refusals print their populations in.
fn bullet(names: &[&String]) -> String {
    names
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join("\n  ")
}

#[test]
fn every_launcher_has_an_oracle_or_a_live_deferral() {
    let (stem, launchers) = backend_launchers();

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
    let covered = |name: &str| references(&tests, &dense, &format!("{stem}{name}"));

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
        bullet(&missing)
    );

    println!(
        "{} launchers: {} with oracles, {} deferred to their milestones",
        launchers.len(),
        launchers.len() - DEFERRED.len(),
        DEFERRED.len()
    );
}

/// The uncalled table in `docs/reference/kernel-inventory.md`: each row's launcher (first
/// cell, backticks stripped) and its category cell. Loud at every step — a heading that
/// went missing or a row without a category is a shrunken registry, and a shrunken
/// registry lets a dead launcher pass as classified.
fn inventory_uncalled_rows() -> Vec<(String, String)> {
    let path = common::repo_root().join("docs/reference/kernel-inventory.md");
    let doc = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let heading = doc
        .lines()
        .find(|l| l.starts_with("## ") && l.ends_with("that no production code calls"))
        .expect("kernel-inventory.md lost its uncalled-launchers heading");
    let rows: Vec<(String, String)> = doc
        .split_once(heading)
        .expect("heading came from doc")
        .1
        .split("\n## ")
        .next()
        .expect("split yields at least one piece")
        .lines()
        .filter(|l| l.starts_with("| `"))
        .map(|l| {
            let cells: Vec<&str> = l.split('|').map(str::trim).collect();
            assert!(
                cells.len() >= 5 && !cells[3].is_empty(),
                "inventory row {l:?} has no category cell — an unclassified row is not a \
                 classification"
            );
            (cells[1].trim_matches('`').to_string(), cells[3].to_string())
        })
        .collect();
    // The heading restates its own table's row count; derive it rather than trust it —
    // prose counts drift (the census figure stood at 61 for four milestones, and CLAUDE.md
    // §13 records a hand-count that stood four days beside a table one row shorter).
    assert_eq!(
        heading,
        format!("## The {} that no production code calls", rows.len()),
        "kernel-inventory.md's heading disagrees with its own table"
    );
    rows
}

#[test]
fn every_launcher_is_shipped_or_classified_uncalled() {
    let (stem, launchers) = backend_launchers();

    // The production corpus: what ships. Test files never count — that distinction is the
    // point (`kernel-inventory.md`, "Model columns"): the oracle census answers yes for
    // every launcher while thirteen have no caller in any binary anyone ships, and only
    // this test asks the second question. `crates/cli/src` is in because the converter is
    // production (`vq_encode`'s only caller), and shared streaming infrastructure lives
    // outside the arch directories (`hash_rows`) — both were misclassified once by a scan
    // that looked only at architecture directories.
    //
    // Stated residual, same shape as the oracle probe's: a production comment or string
    // literal containing `launch_x(` would read as live (none of the classified rows
    // appears anywhere under either root, checked 2026-09-10), and a launcher reached
    // only through a fn-pointer table would read as uncalled — which fails LOUDLY here,
    // the safe direction.
    let prod = format!(
        "{}\n{}",
        corpus("crates/engine/src", ""),
        corpus("crates/cli/src", "")
    );
    let dense: String = prod.chars().filter(|c| !c.is_whitespace()).collect();
    let live = |name: &str| references(&prod, &dense, &format!("{stem}{name}"));

    // Both ends of every classification, before the census consults the table.
    let rows = inventory_uncalled_rows();
    for (name, category) in &rows {
        assert!(
            launchers.iter().any(|l| l == name),
            "kernel-inventory.md classifies `{name}` ({category}) as uncalled, but it is \
             not a launcher under crates/backend/src. A row for something that does not \
             exist explains nothing — delete it."
        );
        assert!(
            !live(name),
            "kernel-inventory.md carries `{name}` as uncalled ({category}), but production \
             code now calls `{stem}{name}`. The classification has outlived its reason — \
             delete the row and light the model column instead."
        );
    }

    let classified: Vec<&str> = rows.iter().map(|(n, _)| n.as_str()).collect();
    let unclassified: Vec<&String> = launchers
        .iter()
        .filter(|l| !live(l) && !classified.contains(&l.as_str()))
        .collect();
    assert!(
        unclassified.is_empty(),
        "\n\n{} launcher(s) have no caller under crates/engine/src or crates/cli/src and \
         no classified row in docs/reference/kernel-inventory.md's uncalled table:\n  \
         {}\n\nAn oracle proves what a kernel computes, not that anything ships it. Wire \
         the launcher, or classify it in the table with primary evidence.",
        unclassified.len(),
        bullet(&unclassified)
    );

    println!(
        "{} launchers: {} with production callers, {} classified uncalled",
        launchers.len(),
        launchers.len() - rows.len(),
        rows.len()
    );
}

/// The Rust kernels' two ends, checked against each other (added 2026-11-18 with
/// `crates/kernels`): every `#[no_mangle]` gpu-kernel in the device crate has exactly one
/// host `Kernel::new(c"…")` handle, and every handle names a kernel that exists.
///
/// A handle without a kernel compiles and fails only at its first launch, on the GPU
/// (`hipModuleGetFunction`), which no deviceless run ever reaches; a kernel without a handle
/// is dead code in the shipped code object. This test is where both become a deviceless red.
#[test]
fn every_rust_kernel_has_exactly_one_host_handle() {
    let decl = format!("pub unsafe extern {:?} fn ", "gpu-kernel");
    // The prelude PROBES (`crates/kernels/src/probes`) are not shipped kernels: the gate
    // launches them by name from a separate code object, so they owe no host handle — their
    // own census is `every_prelude_probe_has_its_hipcc_twin` below. Excluded by PATH, so a
    // shipped kernel can never drop out of this census by sharing a probe's name.
    let root = common::repo_root().join("crates/kernels/src");
    let probes = root.join("probes");
    let shipped: Vec<String> = common::walk(&root, "rs")
        .iter()
        .filter(|p| !p.starts_with(&probes))
        .map(|p| std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {p:?}: {e}")))
        .collect();
    let mut device: Vec<String> = shipped
        .join("\n")
        .lines()
        .filter_map(|l| {
            l.trim_start()
                .strip_prefix(decl.as_str())?
                .split('(')
                .next()
        })
        .map(String::from)
        .collect();
    // `Kernel::new(c"name")`, keyed by the C-string name the handle resolves — the name the
    // code object exports. (A `=>` row naming a missing `rivoli_*` host half needs no census:
    // the wrapper calls it, so that is a compile error, and an unused one is a denied
    // dead-code warning — the compiler checks both ends of that pair.)
    let mut host: Vec<String> = corpus("crates/backend/src", "")
        .split(&format!("{}::new(c\"", "Kernel"))
        .skip(1)
        .filter_map(|rest| rest.split('"').next())
        .map(String::from)
        .collect();
    device.sort();
    host.sort();
    // Anti-vacuity: 14 measured 2026-11-18 (activation 9, residual 4, touch 1).
    assert!(
        device.len() >= 14,
        "found only {} gpu-kernels in crates/kernels/src",
        device.len()
    );
    let dup: Vec<&String> = host
        .windows(2)
        .filter(|w| w[0] == w[1])
        .map(|w| &w[0])
        .collect();
    assert!(
        dup.is_empty(),
        "kernel(s) with more than one host handle:\n  {}",
        bullet(&dup)
    );
    let orphans: Vec<&String> = device.iter().filter(|k| !host.contains(k)).collect();
    let dangling: Vec<&String> = host.iter().filter(|k| !device.contains(k)).collect();
    assert!(
        orphans.is_empty() && dangling.is_empty(),
        "\n\ngpu-kernels with no host handle (dead in the code object):\n  {}\n\nhost handles \
         naming no gpu-kernel (would fail at first launch, on the device):\n  {}",
        bullet(&orphans),
        bullet(&dangling)
    );
    println!("{} Rust kernels, each with one host handle", device.len());
}

/// `__global__` kernel names in the bit-identity gate's hipcc probe source.
fn hip_globals() -> Vec<String> {
    let path = common::repo_root().join("crates/engine/tests/bit_identity/probes/probe.hip");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {path:?}: {e}"))
        .split("__global__ void ")
        .skip(1)
        .filter_map(|rest| rest.split('(').next())
        .map(|n| n.trim().to_string())
        .collect()
}

/// Every prelude probe kernel in `crates/kernels/src/probes` has a same-named hipcc twin in
/// `probe.hip`, and every hipcc probe has a Rust twin — both ends, so a family added on one
/// side only, or renamed on one side, is a deviceless red instead of a `hipModuleGetFunction`
/// failure on the device (added 2026-11-18 with the probe rows). The documented one-sided
/// names: `cmp_bytes`/`cmp_nan_words` (the gate's device compare, hipcc-only on purpose) and
/// `k_vq_wave_r1` (LTO merges the identical Rust function; the gate scores it against
/// `k_vq_wave`) — each checked to exist. And every probe is LAUNCHED: its name is a string
/// literal in the harness, so an orphan pair present on both sides cannot pass silently.
#[test]
fn every_prelude_probe_has_its_hipcc_twin() {
    const HIP_ONLY: [&str; 3] = ["cmp_bytes", "cmp_nan_words", "k_vq_wave_r1"];
    let src = corpus("crates/kernels/src/probes", "");
    let decl = format!("pub unsafe extern {:?} fn ", "gpu-kernel");
    // Three spellings: a plain kernel, `sweep!(name, …)` and `probe!(name(…) = …)` — matched
    // anywhere, not per line, because rustfmt breaks a long macro call after its `(`.
    let mut rust: Vec<String> = [decl.as_str(), "sweep!(", "probe!("]
        .iter()
        .flat_map(|opener| src.split(opener).skip(1))
        .filter_map(|rest| rest.trim_start().split(['(', ',']).next())
        .filter(|n| !n.starts_with('$'))
        .map(String::from)
        .collect();
    let all_hip = hip_globals();
    let missing: Vec<&&str> = HIP_ONLY
        .iter()
        .filter(|n| !all_hip.iter().any(|h| h == *n))
        .collect();
    assert!(
        missing.is_empty(),
        "HIP_ONLY names absent from probe.hip: {missing:?}"
    );
    let mut hip: Vec<String> = all_hip
        .into_iter()
        .filter(|n| !HIP_ONLY.contains(&n.as_str()))
        .collect();
    rust.sort();
    hip.sort();
    // Anti-vacuity: 35 measured 2026-11-18 (0b's probe set less the merged `k_vq_wave_r1`).
    assert!(
        rust.len() >= 35,
        "found only {} probe kernels in crates/kernels/src/probes",
        rust.len()
    );
    let rust_only: Vec<&String> = rust.iter().filter(|n| !hip.contains(n)).collect();
    let hip_only: Vec<&String> = hip.iter().filter(|n| !rust.contains(n)).collect();
    assert!(
        rust_only.is_empty() && hip_only.is_empty(),
        "\n\nRust probes with no hipcc twin in probe.hip:\n  {}\n\nhipcc probes with no Rust twin:\n  {}",
        bullet(&rust_only),
        bullet(&hip_only)
    );
    let harness = corpus("crates/engine/tests/bit_identity/probes", "");
    let unlaunched: Vec<&str> = rust
        .iter()
        .map(String::as_str)
        .chain(HIP_ONLY)
        .filter(|n| !harness.contains(&format!("\"{n}\"")))
        .collect();
    assert!(
        unlaunched.is_empty(),
        "probes no harness row launches: {unlaunched:?}"
    );
    println!(
        "{} prelude probes, each with its hipcc twin and a launching row",
        rust.len()
    );
}
