//! Shared helpers for the workspace meta-gates. Grows only when a second test needs the
//! same helper — the old tree's `tests/common/mod.rs` reached 1050 lines one forced
//! factoring at a time, and each move was jscpd telling it a copy existed.

// Compiled into EACH meta-gate binary; none uses every helper (matrix.rs needs only
// repo_root) — the engine tests' common carries the same argument.
#![allow(dead_code)]

/// Every file under `root` with extension `ext`, recursively. Unsorted.
///
/// WALK, do not list files. The old tree's registry checks each grew their own copy of
/// this, and the hand-maintained path list one of them replaced named five files — moving
/// three of them into subsystem folders silently emptied it, after which the registry
/// reported every invariant as untested. A coverage check keyed on a remembered list fails
/// in the direction that looks like a real regression, which costs more than the walk.
pub fn walk(root: &std::path::Path, ext: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        // Split each directory's entries in one pass instead of branching per entry inside the
        // loop: `partition` keeps the descend/collect decision to a single flat expression, and
        // the `filter` drops the third case (a file with the wrong extension) before it.
        // The double `flatten` swallows an unreadable directory — `walk` feeds coverage checks
        // that must not go red because a path vanished under them, and a directory that cannot
        // be read contributes nothing either way.
        let (subdirs, matches): (Vec<_>, Vec<_>) = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir() || p.extension().is_some_and(|x| x == ext))
            .partition(|p| p.is_dir());
        stack.extend(subdirs);
        out.extend(matches);
    }
    out
}

/// The workspace root: two levels above this crate's manifest.
pub fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default()
}

// ── converter-gate fixtures ─────────────────────────────────────────────────────────────
//
// The three below arrived here on 2026-08-16, when `glimmer_convert.rs` became the second
// converter gate and `build.rs`'s jscpd reported all three as clones of `glm_convert.rs`'s
// copies. That is this file's stated growth rule working exactly as written — it grows when a
// second test needs the same helper, and each move is the duplication gate saying a copy
// exists. They are fixture plumbing, shared by construction rather than by coincidence: a
// converter gate needs deterministic weights, their bf16 encoding, and a scratch directory.

/// Deterministic pseudo-weights: a cheap hash of (name, index) — no RNG dependency, and values
/// in a plausible ±0.1 range so fp8 block scales stay finite.
///
/// **Keyed on the NAME**, which is what makes a byte comparison mean something: a converter
/// that wrote the right tensor's bytes under the wrong name, or the wrong tensor's under the
/// right one, fails rather than passing on identical content.
pub fn weights(name: &str, n: usize) -> Vec<f32> {
    let seed = rivoli_core::hash::fnv1a(name.as_bytes());
    (0..n)
        .map(|i| {
            let h = rivoli_core::hash::fnv1a(&(seed ^ i as u64).to_le_bytes());
            ((h % 2001) as f32 / 1000.0 - 1.0) * 0.1
        })
        .collect()
}

pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| rivoli_core::num::f32_to_bf16(x).to_le_bytes())
        .collect()
}

/// A scratch root under `$TMPDIR`, **removed first**.
///
/// The `remove_dir_all` is load-bearing rather than tidiness: a stale `out1`/`out2` from a
/// killed run would satisfy a determinism compare vacuously, and a stale artifact directory
/// would satisfy a refusal test's "the output must not exist" in reverse.
///
/// `tag` carries the caller's own model and arm (`"glm-convert-rt"`), so one helper serves every
/// gate without the names colliding. Shaped unlike `ppl.rs`'s `tmp()` on purpose — jscpd matched
/// those two temp-dir helpers at 27 tokens once already.
///
/// `#[expect(clippy::expect_used)]` rather than a file-level allow: this module is compiled into
/// the meta-gates too, and a scratch directory that cannot be created is a broken harness that
/// should die loudly rather than a test failure to report.
#[expect(
    clippy::expect_used,
    reason = "a harness that cannot make a temp dir must die loudly"
)]
pub fn scratch(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rivoli-{tag}-{}", std::process::id()));
    assert!(!d.exists() || std::fs::remove_dir_all(&d).is_ok());
    std::fs::create_dir_all(&d).expect("create scratch dir");
    d
}
