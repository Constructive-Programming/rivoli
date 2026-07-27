//! Every `launch_*` in `src/vk.rs` must be exercised by a test.
//!
//! THE TENTH MECHANISED RULE, and it exists because of a specific miss: tranche 2a
//! ported six kernels, delegated three oracles, and reported the tranche complete off
//! the subagent's completion rather than off the tranche's definition. The suite went
//! 16 tests to 23, every one passed, and `gemv_i8` and `gemv_fp8` — the two hardest
//! kernels in the batch, carrying the e4m3 LUT, a `blk_shift` port, 64-bit addressing
//! and a split-K path — had never executed once.
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
//! compiled.
//!
//! KNOWN LIMIT: A SHADER WITH NO LAUNCHER IS INVISIBLE HERE. This check is keyed on
//! `pub unsafe fn launch_*` in `src/vk.rs`, so a `.comp` that exists, compiles, and passes
//! every SPIR-V guard while having no launcher at all is not counted as uncovered — it is
//! not counted at all. `kernel_coverage` going green says nothing about it.
//!
//! Deliberately not fixed, because that state is the legitimate transient during a port:
//! shaders land before their launchers, and a check that failed on it would fire on every
//! honest checkpoint until the tranche closed. Keying on the shader directory instead
//! would trade a silent gap for a noisy one.
//!
//! What covers it is not mechanical: whoever commits shaders ahead of their launchers must
//! SAY SO, and not let a green suite imply otherwise. Recorded here so the next reader
//! knows the boundary of the claim rather than inferring a wider one — which is exactly
//! the inference the `bf16f` note in `common.glsl` had to be rewritten to stop.
#![allow(clippy::expect_used)]

/// Kernels with no oracle, and why. An entry here costs an argument in a reviewable
/// diff, which is the point; it is not a place to park work.
const ALLOWED: &[(&str, &str)] = &[];

/// Read a source file relative to the crate root.
fn source(rel: &str) -> String {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

#[test]
fn every_launcher_has_an_oracle() {
    // Built at runtime, so this file's own text cannot trip the scan — the same reason
    // `oracle_independence` assembles its needle rather than writing it as a literal.
    let decl = format!("pub unsafe fn {}", "launch_");

    let backend = source("src/vk.rs");
    let launchers: Vec<String> = backend
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix(&decl))
        .filter_map(|rest| rest.split('(').next())
        .map(|n| n.to_string())
        .collect();

    // Anti-vacuity: a parse that silently matches nothing passes forever. This has bitten
    // twice already — once when a filter skipped the only file it existed to police, and
    // once when a naming convention changed underneath a scanner.
    assert!(
        launchers.len() >= 5,
        "found only {} launchers in src/vk.rs — the declaration pattern has changed and \
         this check has been passing without examining anything",
        launchers.len()
    );

    let tests = source("tests/vk.rs");
    let missing: Vec<&String> = launchers
        .iter()
        .filter(|name| !ALLOWED.iter().any(|(a, _)| *a == name.as_str()))
        .filter(|name| !tests.contains(&format!("launch_{name}(")))
        .collect();

    assert!(
        missing.is_empty(),
        "\n\n{} kernel(s) have a launcher in src/vk.rs and NO oracle in tests/vk.rs:\n  \
         {}\n\n\
         They compile, they may even be dispatched by other code, and nothing has ever \
         checked what they compute. A passing suite says nothing about them.\n\
         Write the oracle, or add the kernel to ALLOWED in {} with the reason.\n",
        missing.len(),
        missing
            .iter()
            .map(|n| format!("launch_{n}"))
            .collect::<Vec<_>>()
            .join("\n  "),
        file!()
    );

    println!("{} launchers, all exercised", launchers.len());
}
