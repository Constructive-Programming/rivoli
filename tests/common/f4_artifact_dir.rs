//! Locating the shipped `.f4` artifact, for the two V4 loading-path test binaries.
//!
//! The `RIVOLI_V4_*` env vars and the `v4_artifact*` helpers KEEP the model name (2026-08-09
//! rename pass): they point at one model's checkpoint and artifacts
//! on this machine's disk, and an env var name is user-visible configuration recorded in
//! command lines. The FILE is `f4_artifact_dir.rs` because what it locates is a `.f4`
//! artifact directory — the mechanism half of the split.
//!
//! A free-standing file included by `#[path]` rather than an item in `common/mod.rs`:
//! `mod common;` compiles that whole module, and `tests/f4_loading.rs` is deliberately
//! host-only — it must not pull in the GPU-shaped helpers to borrow one path lookup.
//!
//! **`#![allow(dead_code)]` because `#[path]` inclusion compiles this file separately into EACH of
//! its six including binaries**, and none of them uses every helper: `f4_loading.rs` wants the two
//! three-layer fixtures, `artifact_compat.rs` the two full ones. Without it every binary warns
//! about the others' helpers, in a tree whose union clippy run is expected to be silent.
//!
//! Here rather than at each `mod f4_artifact_dir;` — which is what `v4_encoding.rs` did, and was
//! the right call while there were four helpers and one site that skipped some. Adding the two
//! full-artifact helpers (2026-08-11) made every site partial, so six per-site attributes would say
//! the same thing six times and the redundant one in `v4_encoding.rs` is now deleted. The cost is
//! real and worth naming: a helper here that becomes genuinely dead will not be reported.
#![allow(dead_code)]

/// A V4 artifact directory, or `None` when this machine has none.
///
/// `probe` is a file that must exist inside it — the caller names what it actually needs
/// (`L00.f4` for the container tests, `resident.safetensors` for the pin), so a directory
/// holding half an artifact fails on the half that is missing rather than later and
/// elsewhere. `var`/`default` select WHICH fixture: `l0-2` is hash-routed and starts at
/// layer 0, `l3-5` is scored and does not.
///
/// **An explicitly-set `RIVOLI_V4_ARTIFACT` that does not resolve is a failure, not a
/// skip.** libtest captures stderr on passing tests, so an `eprintln!` skip is invisible in
/// a green run: someone who pointed this at an artifact and got all-pass would have no way
/// to tell the ground-truth cases never ran. The default path still skips, because a
/// machine without the 10 GB artifact is the ordinary case.
pub fn v4_artifact_at(var: &str, default: &str, probe: &str) -> Option<String> {
    let named = std::env::var(var).ok();
    let dir = named.clone().unwrap_or_else(|| default.into());
    if std::fs::metadata(format!("{dir}/{probe}")).is_ok() {
        return Some(dir);
    }
    assert!(
        named.is_none(),
        "{var}={dir} has no {probe} — refusing to pass by skipping"
    );
    eprintln!("SKIP: no V4 artifact at {dir} (set RIVOLI_V4_ARTIFACT)");
    None
}

/// Layers 0-2: hash-routed, ratios 0/0/4, range starts at 0.
pub fn v4_artifact(probe: &str) -> Option<String> {
    v4_artifact_at("RIVOLI_V4_ARTIFACT", "/var/db/rivoli/v4-f4-l0-2", probe)
}

/// Layers 3-5: scored routing, ratios 128/4/128, range does NOT start at 0 — the only
/// fixture that can catch an `ExpertSet` or a `F4Pin` ignoring `first_layer`.
pub fn v4_artifact_l3_5(probe: &str) -> Option<String> {
    v4_artifact_at(
        "RIVOLI_V4_ARTIFACT_L3_5",
        "/var/db/rivoli/v4-f4-l3-5",
        probe,
    )
}

/// The FULL V4 artifact — all 43 layers, 145.97 GiB.
///
/// Distinct from [`v4_artifact`]'s three-layer fixture because the question it answers is
/// different: the small ones test the loader's arithmetic, this one tests that **the artifact
/// this machine already has still opens** after a change to the container code. K3's stage S1a
/// renamed the expert-geometry parameter and re-typed `F4Expert`, and neither is allowed to
/// move a byte or an offset in a 43-layer file nobody wants to rebuild.
pub fn v4_artifact_full(probe: &str) -> Option<String> {
    v4_artifact_at(
        "RIVOLI_V4_ARTIFACT_FULL",
        "/var/db/rivoli/v4-f4-full",
        probe,
    )
}

/// The full GLM-5.2 artifact — 76 layers of `.vq3` AND `.i4`, 659.25 GiB together.
///
/// The other half of the same question, and the more informative half: GLM's two formats are
/// the ones with a SHARED block, and `.i4` is the one with no header at all, so between them
/// they exercise every branch of `RoutedFmt::{hbytes, has_shared}` that `.f4` does not.
pub fn glm_artifact_full(probe: &str) -> Option<String> {
    v4_artifact_at(
        "RIVOLI_GLM_ARTIFACT_FULL",
        "/var/db/rivoli/glm52-vq3-full",
        probe,
    )
}
