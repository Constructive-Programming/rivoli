//! Shared helpers for the workspace meta-gates. Grows only when a second test needs the
//! same helper — the old tree's `tests/common/mod.rs` reached 1050 lines one forced
//! factoring at a time, and each move was jscpd telling it a copy existed.

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
