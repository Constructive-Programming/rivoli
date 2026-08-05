//! Locating the shipped `.f4` artifact, for the two V4 loading-path test binaries.
//!
//! A free-standing file included by `#[path]` rather than an item in `common/mod.rs`:
//! `mod common;` compiles that whole module, and `tests/v4_loading.rs` is deliberately
//! host-only — it must not pull in the GPU-shaped helpers to borrow one path lookup.

/// The V4 artifact directory, or `None` when this machine has none.
///
/// `probe` is a file that must exist inside it — the caller names what it actually needs
/// (`L00.f4` for the container tests, `resident.safetensors` for the pin), so a directory
/// holding half an artifact fails on the half that is missing rather than later and
/// elsewhere.
///
/// **An explicitly-set `RIVOLI_V4_ARTIFACT` that does not resolve is a failure, not a
/// skip.** libtest captures stderr on passing tests, so an `eprintln!` skip is invisible in
/// a green run: someone who pointed this at an artifact and got all-pass would have no way
/// to tell the ground-truth cases never ran. The default path still skips, because a
/// machine without the 10 GB artifact is the ordinary case.
pub fn v4_artifact(probe: &str) -> Option<String> {
    let named = std::env::var("RIVOLI_V4_ARTIFACT").ok();
    let dir = named
        .clone()
        .unwrap_or_else(|| "/var/db/rivoli/v4-f4-l0-2".into());
    if std::fs::metadata(format!("{dir}/{probe}")).is_ok() {
        return Some(dir);
    }
    assert!(
        named.is_none(),
        "RIVOLI_V4_ARTIFACT={dir} has no {probe} — refusing to pass by skipping"
    );
    eprintln!("SKIP: no V4 artifact at {dir} (set RIVOLI_V4_ARTIFACT)");
    None
}
