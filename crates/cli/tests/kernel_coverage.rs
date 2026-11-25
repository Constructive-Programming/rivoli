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
#[path = "kernel_coverage/launched_by.rs"]
mod launched_by;

use launched_by::{rowed_names, stale_launched_by};
use std::path::Path;

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
    // place, covered since M7; `kernels/fwd.hip` (tag `archive/hip-kernels`) records the decision), and the k3 tests'
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

/// The arrow a `launchers!` row carries: `=>` (a Rust launcher, `src/launch/`, since
/// 2026-11-18). A row could carry `->` (a hipcc launcher, declared `extern`) until the Phase 1
/// integration (2026-11-19), when the macro lost that arm; the scanner kept matching it until
/// Phase 2 (2026-11-20).
///
/// **The `=>` form was added because the scanner went blind without it, and only half the
/// census noticed.** With the 13 activation/residual rows moved to `=>` and this list still
/// `["->"]`, the oracle census PASSED at 53 launchers — thirteen fewer, under a floor of 45 —
/// and the liveness census reddened only because one of the thirteen
/// (`gated_residual_collapse_f32`) happens to carry an uncalled-table row (measured
/// 2026-11-18, `docs/measurement/gate-red-proofs.md` §18).
const ARROW: &str = " => rivoli_";

/// The launcher stem every census strips (a `const` here is no scanner hazard: no scanner
/// matches the bare stem).
const LAUNCH_STEM: &str = "launch_";

/// Launcher names in the backend's sources, under both declaration forms (hand-written `pub
/// unsafe fn launch_*` and the `launchers!` DSL's `launch_* => rivoli_*` rows). The patterns
/// are built at runtime so this file's own text cannot trip its scanner.
fn launcher_names() -> Vec<String> {
    let decl = format!("pub unsafe fn {}", LAUNCH_STEM);
    corpus(Path::new("crates/backend/src"))
        .lines()
        .map(str::trim_start)
        .filter_map(|l| {
            l.strip_prefix(decl.as_str())
                .and_then(|rest| rest.split('(').next())
                .or_else(|| {
                    let rest = l.strip_prefix(LAUNCH_STEM)?;
                    rest.split_once(ARROW).map(|(name, _)| name)
                })
        })
        .map(String::from)
        .collect()
}

/// Every `.rs` under `dir` (workspace-relative), concatenated, read LOUDLY — a missing
/// file is a shrunken corpus and a shrunken corpus is a silent pass. (It once took a file
/// name to skip, for when this file sat inside a corpus it searched; no caller has passed
/// one since the censuses moved to `crates/cli/tests`, so the parameter is gone — 2026-11-18.)
fn corpus(dir: &Path) -> String {
    let root = common::repo_root().join(dir);
    let all = common::walk(&root, "rs");
    assert!(!all.is_empty(), "corpus({dir:?}) is empty — wrong root?");
    all.iter()
        .map(|p| std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {p:?}: {e}")))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The launcher population plus the probe stem, shared by both censuses so they cannot
/// disagree about what the backend contains.
fn backend_launchers() -> Vec<String> {
    let launchers = launcher_names();

    // Anti-vacuity, DERIVED (Phase 0 merge review, 2026-11-18): the scanned population must
    // equal `kernel-inventory.md`'s table, both ends. It was a hand-bumped floor — 45 under a
    // population of 53 on 2026-08-15, low enough that losing every Rust launcher at once
    // passed (see `ARROW`), then 66 — and a floor is only a check while someone remembers to
    // move it. A launcher added without its inventory row, or a row outliving its launcher,
    // is now a deviceless red, and no number lives in this file.
    let table = inventory_table();
    let mut sorted = launchers.clone();
    sorted.sort();
    let (unlisted, stale) = one_sided(&sorted, &table);
    assert!(
        unlisted.is_empty() && stale.is_empty(),
        "\n\nlaunchers with no kernel-inventory.md table row:\n  {}\n\ninventory rows naming no \
         launcher (or the declaration pattern changed underneath this scanner):\n  {}",
        bullet(&unlisted),
        bullet(&stale)
    );
    launchers
}

/// A corpus in both views the citation probe reads: as written, and whitespace-stripped
/// (rustfmt line-breaks long calls, so `(launch_x,` may never appear literally).
struct Cited {
    plain: String,
    dense: String,
}

impl Cited {
    /// The `.rs` corpora under `dirs`, joined.
    fn of(dirs: &[&Path]) -> Self {
        let plain = dirs
            .iter()
            .map(|d| corpus(d))
            .collect::<Vec<_>>()
            .join("\n");
        let dense = plain.chars().filter(|c| !c.is_whitespace()).collect();
        Self { plain, dense }
    }

    /// The corpus cites launcher `name` (bare, as `launcher_names` yields it) in either
    /// form: a direct call `launch_name(`, or `(launch_name,` — handed to a shared launch
    /// helper as its FIRST argument, probed against the dense view. The block comment at
    /// the oracle census's call site argues both forms and why the `(` is load-bearing.
    fn cites(&self, name: &str) -> bool {
        let base = format!("{LAUNCH_STEM}{name}");
        self.plain.contains(&format!("{base}(")) || self.dense.contains(&format!("({base},"))
    }
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
    let (stem, launchers) = (LAUNCH_STEM, backend_launchers());

    // The oracle corpus: engine's device tests. No size floor — a truncated corpus only
    // makes launchers look UNCOVERED, which fails loudly by itself.
    let tests = Cited::of(&[Path::new("crates/engine/tests")]);
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

    // Both ends of every deferral, before the census consults it.
    for (name, arrives) in DEFERRED {
        assert!(
            launchers.iter().any(|l| l == name),
            "DEFERRED names `{stem}{name}`, which is not a launcher under \
             crates/backend/src. A row for something that does not exist covers nothing — \
             delete it. (was: {arrives})"
        );
        assert!(
            !tests.cites(name),
            "DEFERRED row for `{stem}{name}` ({arrives}) — but a test now covers it. The \
             deferral has outlived its reason; delete the row so the census owns the claim."
        );
    }

    // `launcher_names` yields BARE names (both declaration forms strip the stem) — the
    // first run of this file compared stem-prefixed strings and the both-ends check
    // refused its own table, which is the check working on its author.
    let missing: Vec<&String> = launchers
        .iter()
        .filter(|l| !tests.cites(l) && !DEFERRED.iter().any(|(n, _)| n == l))
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

/// A file or corpus the censuses scan, and every way they read one — the scanning lives
/// here so each census states WHAT it reads, not how.
struct Text(String);

impl Text {
    /// A workspace file, read loudly.
    fn read(rel: &Path) -> Self {
        let path = common::repo_root().join(rel);
        Self(std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}")))
    }

    /// The `.rs` corpus under a workspace directory.
    fn corpus(dir: &Path) -> Self {
        Self(corpus(dir))
    }

    fn lines(&self) -> std::str::Lines<'_> {
        self.0.lines()
    }

    /// The text from the first `start` to the next `end` (or to the end of the text),
    /// whitespace-stripped — rustfmt breaks long entries anywhere, so a table is read in this
    /// form. Loud if `start` is gone.
    fn dense_between(&self, start: &str, end: &str) -> Self {
        let rest = self
            .0
            .split_once(start)
            .unwrap_or_else(|| panic!("lost {start:?}"))
            .1;
        let body = rest.split_once(end).map_or(rest, |(b, _)| b);
        Self(body.chars().filter(|c| !c.is_whitespace()).collect())
    }

    /// The table rows (`| \`…`) of the section headed by the line `heading`, up to the
    /// next `## `.
    fn table(&self, heading: &str) -> Vec<&str> {
        self.0
            .split_once(&format!("\n{heading}\n"))
            .unwrap_or_else(|| panic!("lost the {heading:?} section"))
            .1
            .split("\n## ")
            .next()
            .expect("split yields at least one piece")
            .lines()
            .filter(|l| l.starts_with("| `"))
            .collect()
    }

    /// Every name that follows `opener`, up to `close` (trimmed), sorted.
    fn names_after(&self, opener: &str, close: char) -> Vec<String> {
        let mut names: Vec<String> = self
            .0
            .split(opener)
            .skip(1)
            .filter_map(|rest| rest.split(close).next())
            .map(|n| n.trim().to_string())
            .collect();
        names.sort();
        names
    }

    fn contains(&self, needle: &str) -> bool {
        self.0.contains(needle)
    }
}

/// The first cell (backticks stripped) of every row of `kernel-inventory.md`'s main table
/// — the launcher population, one row each. Loud when the section is gone or empty. Sorted.
fn inventory_table() -> Vec<String> {
    let heading = "## The table";
    let doc = Text::read(Path::new("docs/reference/kernel-inventory.md"));
    let mut rows: Vec<String> = doc
        .table(heading)
        .iter()
        .filter_map(|l| l.strip_prefix("| `")?.split('`').next())
        .map(String::from)
        .collect();
    assert!(
        !rows.is_empty(),
        "kernel-inventory.md's {heading:?} section has no rows"
    );
    rows.sort();
    rows
}

/// The uncalled table in `docs/reference/kernel-inventory.md`: each row's launcher (first
/// cell, backticks stripped) and its category cell. Loud at every step — a heading that
/// went missing or a row without a category is a shrunken registry, and a shrunken
/// registry lets a dead launcher pass as classified.
fn inventory_uncalled_rows() -> Vec<(String, String)> {
    let doc = Text::read(Path::new("docs/reference/kernel-inventory.md"));
    let heading = doc
        .lines()
        .find(|l| l.starts_with("## ") && l.ends_with("that no production code calls"))
        .expect("kernel-inventory.md lost its uncalled-launchers heading");
    let rows: Vec<(String, String)> = doc
        .table(heading)
        .into_iter()
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
    let (stem, launchers) = (LAUNCH_STEM, backend_launchers());

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
    let prod = Cited::of(&[Path::new("crates/engine/src"), Path::new("crates/cli/src")]);

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
            !prod.cites(name),
            "kernel-inventory.md carries `{name}` as uncalled ({category}), but production \
             code now calls `{stem}{name}`. The classification has outlived its reason — \
             delete the row and light the model column instead."
        );
    }

    let classified: Vec<&str> = rows.iter().map(|(n, _)| n.as_str()).collect();
    let unclassified: Vec<&String> = launchers
        .iter()
        .filter(|l| !prod.cites(l) && !classified.contains(&l.as_str()))
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

/// The shipped gpu-kernels: every `pub unsafe extern "gpu-kernel" fn` under
/// `crates/kernels/src` except the prelude PROBES (`src/probes`), which are not shipped — the
/// gate launches them by name from a separate code object, so they owe no host handle and no
/// gate row; their own census is `every_prelude_probe_has_its_hipcc_twin` below. Excluded by
/// PATH, so a shipped kernel can never drop out by sharing a probe's name. Sorted.
fn shipped_kernels() -> Vec<String> {
    let decl = format!("pub unsafe extern {:?} fn ", "gpu-kernel");
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
    device.sort();
    assert!(
        !device.is_empty(),
        "no gpu-kernels found in crates/kernels/src"
    );
    device
}

/// `Kernel::new(c"name")` handles in the backend, keyed by the C-string name the handle
/// resolves — the name the code object exports. Sorted, duplicates kept (one census refuses
/// them).
fn host_handles() -> Vec<String> {
    Text::corpus(Path::new("crates/backend/src"))
        .names_after(&format!("{}::new(c\"", "Kernel"), '"')
}

/// Both ends of a pairing: what `a` has that `b` lacks, and the reverse.
fn one_sided<'a, T: PartialEq>(a: &'a [T], b: &'a [T]) -> (Vec<&'a T>, Vec<&'a T>) {
    (
        a.iter().filter(|k| !b.contains(k)).collect(),
        b.iter().filter(|k| !a.contains(k)).collect(),
    )
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
    // (A `=>` row naming a missing `rivoli_*` host half needs no census: the wrapper calls
    // it, so that is a compile error, and an unused one is a denied dead-code warning — the
    // compiler checks both ends of that pair.)
    let (device, host) = (shipped_kernels(), host_handles());
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
    let (orphans, dangling) = one_sided(&device, &host);
    assert!(
        orphans.is_empty() && dangling.is_empty(),
        "\n\ngpu-kernels with no host handle (dead in the code object):\n  {}\n\nhost handles \
         naming no gpu-kernel (would fail at first launch, on the device):\n  {}",
        bullet(&orphans),
        bullet(&dangling)
    );
    println!("{} Rust kernels, each with one host handle", device.len());
}

/// The one shipped Rust kernel with no `launchers!` row and no bit-identity row:
/// `touch_region` writes nothing comparable (its `sink` is stored only on an impossible
/// value), and its launcher is the fetch path's hand-written `rivoli_touch_region_async`,
/// reached from `engine/src/fetch/stage.rs` rather than through a `launch_*` wrapper. What
/// stands in for a value comparison is its ISA (one `global_load_b128` per 16 bytes, the C's
/// `uint4` read, checked deviceless at the Phase 0 merge review) and
/// `tests/determinism-glm.sh` on the `--arena-refresh` arm, which is what the read exists for.
const UNROWED: [&str; 1] = ["touch_region"];

/// New-model kernels with no legacy C twin — owner direction 2026-11-24: inkling needs no C
/// versions. Each is validated against the first-party CPU oracle (`rivoli_oracles`) and the S3
/// anchor through its oracle suite, NOT the C→Rust bit-identity harness, so it is waived the
/// C-twin gate row. It still owes a `=>` launchers row (the arrows census) and an oracle suite
/// (the oracle census). `(kernel, suite)` is checked at both ends in
/// [`every_rust_kernel_has_a_gate_row_and_a_launchers_row`]: the name is in `rowed` — a shipped
/// kernel's own row name, or the `LAUNCHED_BY` launcher of a multi-kernel pair — and the suite
/// exists and cites its launcher. A row dies loudly when its kernel stops being shipped or its
/// suite stops covering it — the same both-ends discipline as [`DEFERRED`], not an exemption list.
const ORACLE_VALIDATED: &[(&str, &str)] = &[
    ("l2_clip", "kernel_inkling_l2clip.rs"),
    ("scaled_l2_clip", "kernel_inkling_l2clip.rs"),
    ("log_scaled_l2_clip", "kernel_inkling_l2clip.rs"),
    ("mxfp8_dequant", "kernel_inkling_dequant.rs"),
    ("nvfp4_dequant", "kernel_inkling_dequant.rs"),
    ("sconv", "kernel_inkling_sconv.rs"),
    ("sconv_dynamic", "kernel_inkling_sconv.rs"),
    ("vaxpy", "kernel_inkling_vaxpy.rs"),
    ("sink_attend", "kernel_inkling_sink_attend.rs"),
    ("message_gate", "kernel_inkling_message_gate.rs"),
    ("kv_scatter", "kernel_inkling_kv_scatter.rs"),
    ("rel_bias_gather", "kernel_inkling_rel_bias.rs"),
    // Inkling-2 Small's MXFP8 routed-expert pair rides its RANGE LAUNCHER's name, not either
    // kernel's: the two phases are one `launch_moe_expert_range_mxfp8_situ` call, which is
    // what the suite cites and what `LAUNCHED_BY` rows them under. The check below therefore
    // asks whether the name is in `rowed` — the launcher names plus the ungrouped kernels —
    // rather than in the shipped-kernel list, which holds only the two phases.
    ("moe_expert_range_mxfp8_situ", "kernel_inkling_moe_mxfp8.rs"),
    // The decode projections' bf16 GEMV: NEW in the rewrite, so there is no C twin to score
    // bit-for-bit — its suite's f64 reference is the oracle (the kernel's own note says why
    // `gemm_bf16` could not simply be widened instead).
    ("gemv_bf16", "kernel_gemv_bf16.rs"),
];

/// The bit-identity gate's `ROWS` table: each entry's kernel name, and whether its row
/// function binds that kernel's tagged C launcher (`c"rivoli_<name>"`, the symbol it scores
/// against) — so an entry pointing at another kernel's row cannot stand in for this one.
fn gate_rows() -> Vec<(String, bool)> {
    let dir = Path::new("crates/engine/tests/bit_identity");
    let table = Text::read(&dir.join("main.rs")).dense_between("const ROWS: &[Row] = &[", "];");
    table
        .names_after("(\"", ')')
        .iter()
        .map(|entry| {
            // `name",module::row`
            let (name, path) = entry
                .split_once("\",")
                .expect("a ROWS entry is (name, row)");
            // rustfmt leaves a trailing comma on an entry it broke across lines.
            let path = path.trim_end_matches(',');
            let (module, row) = path.split_once("::").expect("a row is module::fn");
            let src = Text::read(&dir.join(format!("{module}.rs")));
            let body = src.dense_between(&format!("pub fn {row}("), "\npub fn ");
            (
                name.to_string(),
                body.contains(&format!("c\"rivoli_{name}\"")),
            )
        })
        .collect()
}

/// Every shipped Rust kernel is scored by the bit-identity gate and reached through a
/// `launchers!` `=>` row, and every gate row and `=>` row names a shipped Rust kernel — three
/// populations, both directions each, with [`UNROWED`] the one argued exemption (checked to
/// exist) and [`launched_by::LAUNCHED_BY`] rowing a multi-kernel launcher's phases under that
/// launcher (checked at both ends). Without it a lane could port a kernel with no row and
/// every other census plus the gate would stay green; CLAUDE.md's "every Rust kernel with an
/// output" was prose until this (Phase 0 merge review, 2026-11-18).
#[test]
fn every_rust_kernel_has_a_gate_row_and_a_launchers_row() {
    let device = shipped_kernels();
    for u in UNROWED {
        assert!(
            device.iter().any(|k| k == u),
            "UNROWED names `{u}`, which is not a shipped gpu-kernel — delete the exemption"
        );
    }
    // `rowed` first: the oracle waiver's shipped-ness check asks it, not `device` — see
    // `assert_oracle_validated`.
    let (rowed, stale) = (rowed_names(), stale_launched_by());
    assert_oracle_validated(&rowed);
    // A `=>` row names its Rust host launcher `rivoli_<kernel>`; that launcher's own
    // `Kernel::new(c"<kernel>")` is held by the host-handle census above.
    // Only declaration lines (they start with the stem): the DSL's own docs spell the form.
    let decls: String = Text::corpus(Path::new("crates/backend/src"))
        .lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with(LAUNCH_STEM))
        .collect::<Vec<_>>()
        .join("\n");
    let arrows = Text(decls).names_after(ARROW, '(');
    let entries = gate_rows();
    let unbound: Vec<&String> = entries.iter().filter(|(_, b)| !b).map(|(n, _)| n).collect();
    let rows: Vec<String> = entries.iter().map(|(n, _)| n.clone()).collect();
    // Every finding from both populations, gathered before one assert, so a defect that
    // breaks both shows both (a first assert would mask the second).
    let rowed_gate: Vec<String> = rowed
        .iter()
        .filter(|n| !ORACLE_VALIDATED.iter().any(|(k, _)| *k == n.as_str()))
        .cloned()
        .collect();
    let report: Vec<String> = [
        ("the bit-identity ROWS table", &rows, &rowed_gate),
        ("the `=>` launchers! rows", &arrows, &rowed),
    ]
    .iter()
    .flat_map(|(what, population, rowed_pop)| {
        let (missing, extra) = one_sided(rowed_pop, population);
        [
            (
                format!("row names (a kernel, or its LAUNCHED_BY launcher) missing from {what}"),
                missing,
            ),
            (
                format!(
                    "entries in {what} that are no row name (a kernel, or its LAUNCHED_BY launcher)"
                ),
                extra,
            ),
        ]
    })
    .chain([
        (
            "ROWS entries whose row function never binds `c\"rivoli_<name>\"`".to_string(),
            unbound,
        ),
        (
            "stale LAUNCHED_BY entries".to_string(),
            stale.iter().collect(),
        ),
    ])
    .filter(|(_, names)| !names.is_empty())
    .map(|(head, names)| format!("{head}:\n  {}", bullet(&names)))
    .collect();
    assert!(report.is_empty(), "\n\n{}", report.join("\n\n"));
    println!(
        "{} Rust kernels: {} C-twin rows ({} kernels rowed via LAUNCHED_BY), {} oracle-validated \
         (no C twin), {} argued exemption",
        device.len(),
        rows.len(),
        launched_by::LAUNCHED_BY.len(),
        ORACLE_VALIDATED.len(),
        UNROWED.len()
    );
}

/// The [`ORACLE_VALIDATED`] both-ends check: each waived name is a row name (`rowed`), and its
/// oracle suite exists and cites its launcher. A waiver covering a kernel nothing checks is
/// refused — the same discipline as [`DEFERRED`], split out to keep the census under the
/// code-health length cap.
fn assert_oracle_validated(rowed: &[String]) {
    let cited = Cited::of(&[Path::new("crates/engine/tests")]);
    for (kernel, suite) in ORACLE_VALIDATED {
        // `rowed`, not the shipped-kernel list: an ungrouped kernel's rowed name IS its own
        // name, and a `LAUNCHED_BY` phase's is its launcher's — which `stale_launched_by`
        // separately holds to a shipped kernel, a named static and a launch. So a row here
        // still dies loudly when its kernel or its launcher leaves the tree.
        assert!(
            rowed.iter().any(|k| k == kernel),
            "ORACLE_VALIDATED names `{kernel}`, which is neither a shipped gpu-kernel nor the \
             launcher of one — delete the row"
        );
        assert!(
            common::repo_root()
                .join("crates/engine/tests")
                .join(suite)
                .exists(),
            "ORACLE_VALIDATED `{kernel}`: oracle suite {suite} does not exist"
        );
        assert!(
            cited.cites(kernel),
            "ORACLE_VALIDATED `{kernel}`: no oracle cites launch_{kernel} — the waiver covers a \
             kernel nothing checks"
        );
    }
}

/// `__global__` kernel names in the bit-identity gate's hipcc probe source.
fn hip_globals() -> Vec<String> {
    Text::read(Path::new(
        "crates/engine/tests/bit_identity/probes/probe.hip",
    ))
    .names_after("__global__ void ", '(')
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
    let src = corpus(Path::new("crates/kernels/src/probes"));
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
    let harness = corpus(Path::new("crates/engine/tests/bit_identity/probes"));
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
