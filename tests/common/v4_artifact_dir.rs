//! Locating the shipped `.f4` artifact, for the two V4 loading-path test binaries.
//!
//! A free-standing file included by `#[path]` rather than an item in `common/mod.rs`:
//! `mod common;` compiles that whole module, and `tests/v4_loading.rs` is deliberately
//! host-only — it must not pull in the GPU-shaped helpers to borrow one path lookup.

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
