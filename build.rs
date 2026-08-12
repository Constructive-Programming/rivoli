//! Build script: panics are the correct failure mode for a broken toolchain.
#![allow(clippy::expect_used)]

//! Compile the HIP kernels with hipcc and link libamdhip64 — only under the `rocm`
//! feature, so `cargo check` / CPU-side dev needs no GPU toolchain. (The cold-expert
//! io_uring streamer is the `io-uring` crate now, talking to the syscalls directly —
//! no liburing system lib.)

use std::path::Path;
use std::process::Command;

fn main() {
    // Backend-independent; runs in BOTH arms, including featureless. There were three arms
    // until 2026-08-06, when the `vulkan` one — glslc instead of hipcc, plus a SPIR-V rule
    // engine — was retired with the backend (tag `archive/vulkan-backend-hb16`).
    no_duplicated_rust();
    if std::env::var("CARGO_FEATURE_ROCM").is_err() {
        return; // CPU-only build: no HIP toolchain required.
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    // True when `out` is missing or older than any of `deps` — the mtime comparison cargo
    // does for its own targets, which build scripts do not get for free. Missing metadata
    // means REBUILD: an unreadable timestamp is not evidence the output is current.
    fn stale(out: &str, deps: &[&str]) -> bool {
        let Ok(t) = std::fs::metadata(out).and_then(|m| m.modified()) else {
            return true;
        };
        deps.iter().any(|d| {
            std::fs::metadata(d)
                .and_then(|m| m.modified())
                .map(|s| s > t)
                .unwrap_or(true)
        })
    }
    let kernels = [
        "linalg",
        "moe",
        "mla",
        "attn",
        "fwd",
        "vmm",
        "indexer",
        "async",
        "kvcompress",
        "blockindex",
        "headtail",
        "recurrent",
    ];
    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "hipcc".into());
    // ponytail: default gfx1151 (Strix Halo); override for other ROCm nodes (e.g.
    // gfx1010 on rh-desktop's RX 5700) with RIVOLI_OFFLOAD_ARCH.
    println!("cargo:rerun-if-env-changed=RIVOLI_OFFLOAD_ARCH");
    let arch = std::env::var("RIVOLI_OFFLOAD_ARCH").unwrap_or_else(|_| "gfx1151".into());
    let offload = format!("--offload-arch={arch}");

    // Headers are #included, not compiled units — track them so an edit forces a rebuild.
    println!("cargo:rerun-if-changed=kernels/common.hpp");
    let mut objs = Vec::new();
    for k in kernels {
        let src = format!("kernels/{k}.hip");
        let obj = format!("{out_dir}/{k}.o");
        println!("cargo:rerun-if-changed={src}");
        // SKIP an object that is newer than both its source and the shared header.
        //
        // This script re-runs whenever anything it watches changes, and since the
        // duplication gate watches `src/` that now includes every Rust edit. Without this
        // check each one recompiled all eight kernels through hipcc — seconds of rebuild
        // for a change that cannot affect a single one of them. `common.hpp` is in the
        // comparison because it is #included rather than compiled, so touching it must
        // invalidate everything.
        if !stale(&obj, &[&src, "kernels/common.hpp"]) {
            objs.push(obj);
            continue;
        }
        let mut cmd = Command::new(&hipcc);
        cmd.args([&offload, "-O3", "-fPIC"]);
        let status = cmd
            .args(["-c", &src, "-o", &obj])
            .status()
            .expect("run hipcc");
        assert!(status.success(), "hipcc failed on {src}");
        objs.push(obj);
    }

    let lib = format!("{out_dir}/librivolikernels.a");
    // `ar crs` only adds/replaces members — it never prunes ones dropped from the
    // list, so a since-deleted kernel's stale .o lingers and duplicate-symbol-clashes.
    // Start from a clean archive every build.
    let _ = std::fs::remove_file(&lib);
    let mut ar = Command::new("ar");
    ar.arg("crs").arg(&lib);
    for o in &objs {
        ar.arg(o);
    }
    assert!(ar.status().expect("run ar").success(), "ar failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=rivolikernels");
    for dir in ["/usr/lib64", "/opt/rocm/lib", "/usr/lib"] {
        if Path::new(dir).join("libamdhip64.so").exists() {
            println!("cargo:rustc-link-search=native={dir}");
            break;
        }
    }
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}

/// Fail the build if jscpd finds ANY duplicated block in this repo's own Rust — `src/`,
/// `tests/`, and this file — at `.jscpd.json`'s `minTokens: 15`.
///
/// ## Opt-in today, via `RIVOLI_JSCPD=1`
///
/// Measured 2026-08-01 at min-tokens 15 across all 46 Rust files: **181 clones, 1290
/// duplicated lines, 4.51% of 28594**. An unconditional gate would therefore make the
/// crate unbuildable, which is not a gate, it is a brick. Clearing that backlog is
/// separate work; when it reaches zero, delete the `RIVOLI_JSCPD` read below and the gate
/// is armed for everyone.
///
/// ## `.jscpd.json` raises maxLines/maxSize, and that is coverage, not tuning
///
/// jscpd's defaults (`max-lines 1000`, `max-size 100kb`) SILENTLY SKIP 9 of the 46 files,
/// including the four largest: `src/backend/vk.rs` (4125 lines), `tests/vk.rs` (3841),
/// `src/gpu.rs` (2591), `tests/kernel.rs` (1586). Those are precisely where copy-paste
/// accumulates, and the skip is printed nowhere except under `--debug`. At the defaults
/// this tree reports 31 clones / 1.92%; the gap to 4.51% is the size of the hole. Cost of
/// closing it: detection 0.6s -> 11.9s, paid only by whoever arms the gate. `maxLines`
/// must stay above the largest file in the tree or the hole silently reopens.
///
/// > **CORRECTED 2026-08-06.** Two of those four named files — `src/backend/vk.rs` and
/// > `tests/vk.rs` — were deleted with the Vulkan backend. The tree is now 72 Rust files
/// > and the largest is `src/gpu.rs` at 3251 lines. **The argument is unchanged and
/// > `maxLines: 10000` must NOT be lowered to suit the smaller tree**: it is a ceiling
/// > chosen to sit far above the largest file precisely so that growth cannot silently
/// > reopen the hole. The measured 4.51%/31-clone figures above describe the 2026-08-01
/// > tree and are kept as the record of why the setting exists, not as current numbers.
///
/// ## An env var, which CLAUDE.md forbids for engine instruments
///
/// The objection there is that an env var is invisible to `--help`, absent from the
/// recorded command line in `docs/measurement/benchmarks.md`, and silently active in a
/// build that looks stock. None of it applies here: a build script has no `--help` to be
/// absent from, cargo already configures this one entirely through the environment
/// (`CARGO_FEATURE_*`, `OUT_DIR`, `HIPCC`, `RIVOLI_OFFLOAD_ARCH`), and
/// `rerun-if-env-changed` puts the toggle inside cargo's own fingerprint.
///
/// ## A missing jscpd SKIPS; only a DETECTED CLONE fails
///
/// This is a Node tool in a Rust crate, and a box without Node must still be able to build
/// it. (Until 2026-08-06 this read "same treatment as `spirv-val` and `spirv-dis` above" —
/// those lived in the `vulkan` arm and went with it. The policy they shared is the one
/// spelled out here.) The two outcomes are told apart by exit code, never by parsing the
/// report:
///
/// - `--exitCode 7` is jscpd's own "clones were found" signal and fires on
///   `clones.length > 0`. `--threshold 0` is the obvious alternative and is WRONG for
///   "strictly forbidden": it compares a percentage rounded to 2dp, so a small enough
///   clone in a large enough tree reads as 0.00% and passes. It also throws instead of
///   returning, so it exits 1 — indistinguishable from the tool being absent.
/// - Missing package, unreadable config, anything else: exit 1, which warns and carries on.
///
/// `npx --no` is load-bearing: without it npx DOWNLOADS jscpd from the network mid-build.
/// So is the bare `--` — npm otherwise claims `--exitCode` as its own flag and refuses to
/// run ("Unknown cli config", exit 1), which this function would have read as "absent" and
/// skipped silently forever.
fn no_duplicated_rust() {
    // ALWAYS ON as of 2026-08-02. It was opt-in behind `RIVOLI_JSCPD` for exactly one day,
    // while the backlog it was measuring — 181 clones, 4.49% — was cleared to zero. A gate
    // nobody arms is a gate that reports nothing, which is the same failure as the empty
    // exemption lists this build script used to carry.
    //
    // The `rerun-if-changed=src` below means every Rust edit re-runs this script. That used
    // to imply recompiling all eight HIP kernels; `stale()` in `hip()` now skips objects
    // newer than their source, so the cost of a re-run is the jscpd scan alone (~8 s over
    // 46 files) and only when something actually changed.
    //
    // ONE list, serving as both the scan set and cargo's rerun set, so the two cannot
    // drift apart.
    const SCAN: &[&str] = &["src", "tests", "build.rs"];
    for p in SCAN {
        println!("cargo:rerun-if-changed={p}");
    }
    println!("cargo:rerun-if-changed=.jscpd.json");

    // `-c` is explicit on purpose. jscpd's default is ".jscpd.json in <path>", and <path>
    // here is `src` — a silent fall-back to the built-in minTokens 50 would leave the gate
    // more than three times looser than the file that is supposed to govern it.
    let out = match Command::new("npx")
        .args([
            "--no",
            "--",
            "jscpd",
            "-c",
            ".jscpd.json",
            "--exitCode",
            "7",
        ])
        .args(SCAN)
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            println!("cargo:warning=jscpd not run ({e}); Rust duplication unchecked");
            return;
        }
    };
    // A CLEAN RESULT IS ONLY MEANINGFUL ON A RUSTFMT-CLEAN TREE, and that is not a style
    // preference — it is the gate's correctness precondition.
    //
    // jscpd tokenizes. Two blocks that differ only in line breaking tokenize differently
    // enough to fall under `minTokens`, so unformatted code can carry real duplication past
    // this gate. Measured 2026-08-06: the tree reported **0 clones** with 680 rustfmt hunks
    // outstanding, and **52** the moment `cargo fmt` ran. Nothing was added — the formatter
    // let the tokenizer see what was already there.
    //
    // That also puts an asterisk on this file's own history: the "181 clones -> 0" recorded
    // above was measured across a window in which the tree drifted out of rustfmt, so an
    // unknown share of that zero was drift rather than dedup.
    //
    // A WARNING, not a hard failure: `cargo build` on a tree someone is mid-edit in must not
    // refuse, and CI gates `cargo fmt --check` in its own step anyway. What this must never
    // do is let a green jscpd run be read as "no duplication" when it can only mean "no
    // duplication the tokenizer could see".
    let fmt_clean = std::process::Command::new("cargo")
        .args(["fmt", "--check", "--quiet"])
        .output()
        .map_or(true, |o| o.status.success());
    if !fmt_clean {
        println!(
            "cargo:warning=tree is not rustfmt-clean, so the jscpd result below is a LOWER \
             BOUND -- formatting differences hide clones from the tokenizer (measured \
             2026-08-06: 0 reported at 680 outstanding hunks, 52 after `cargo fmt`)"
        );
    }
    match out.status.code() {
        Some(0) => {}
        // BOTH streams: the clone list is on stdout, but an invocation-level complaint can
        // land on either. (The reason used to be cross-referenced to `spirv_val`, deleted
        // with the Vulkan arm on 2026-08-06.)
        Some(7) => panic!(
            "\n\njscpd found duplicated Rust. Duplicates are FORBIDDEN here, not \
             budgeted — .jscpd.json carries no `threshold`:\n\n{}\n{}\n",
            String::from_utf8_lossy(&out.stdout).trim(),
            String::from_utf8_lossy(&out.stderr).trim()
        ),
        _ => println!(
            "cargo:warning=jscpd did not run ({}); Rust duplication unchecked. Run \
             `npx --no -- jscpd -c .jscpd.json src tests build.rs` to see why.",
            out.status
        ),
    }
}
