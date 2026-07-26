//! A tripwire for the independent-oracle rule in `tests/vk.rs`'s header.
//!
//! THE RULE: an oracle is derived from the HIP original and `src/math.rs`, which are
//! the specification — never from the Vulkan shader it is checking. An oracle written
//! by someone who has seen the implementation is a consistency check wearing a
//! correctness check's clothes: it agrees because it was copied, and it will ratify a
//! shared misreading of the HIP rather than catching it.
//!
//! WHAT THIS CAN AND CANNOT DO. "Did not read a file" is not mechanically checkable,
//! and a determined violation — reading a shader, internalising it, writing the oracle
//! from memory — leaves no trace this or anything else could find. What it catches is
//! the violation that actually happens: someone COPIES shader logic into an oracle
//! because deriving it from the HIP is more work, and reaches for the shader path to
//! do it. That leaves an artifact, and artifacts are checkable.
//!
//! So this is documentation with a tripwire rather than documentation alone. In this
//! codebase the mechanised rules have a perfect record and the documented ones have
//! failed repeatedly, so catching the lazy half is worth ten lines.
//!
//! The second thing it buys is that the ONE legitimate exception is explicit. Adding a
//! second one means editing an allowlist in a diff someone reviews, rather than
//! quietly opening a file.
//!
//! Not feature-gated: the rule applies to whoever is writing tests, not to whoever
//! compiled the Vulkan backend.
#![allow(clippy::expect_used)]

/// The single sanctioned exception. `glsl_numerics.rs` reads `common.glsl` ON PURPOSE:
/// it hashes the two numeric helpers so the literal Rust transcriptions beside them
/// cannot silently drift. That is the inverse of the rule — its whole job is to notice
/// when the shader changes — so it is exempt by design, not by oversight.
const ALLOWED: &[&str] = &["glsl_numerics.rs"];

/// Assembled at runtime rather than written as one literal, so this file does not trip
/// its own check and need a second allowlist entry that would blunt the "one exception"
/// framing.
///
/// A literal `/`, NOT `MAIN_SEPARATOR`. Paths in Rust source are written with forward
/// slashes on every platform, so keying off the host separator would make this silently
/// match nothing on Windows — a tripwire that disarms itself rather than failing is the
/// exact shape this file exists to prevent.
fn shader_dir_needle() -> String {
    format!("kernels{}vk", '/')
}

/// Every `.rs` file under `dir`, recursively. Non-recursive scanning would make a
/// `tests/oracles/` submodule invisible, which is a normal way for a test suite to grow.
fn rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn oracles_do_not_reference_the_shaders() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests");
    let needle = shader_dir_needle();
    let me = std::path::Path::new(file!())
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    // tests/ ONLY, recursively. Scanning src/ was tried and reverted: `src/vk.rs` is
    // the backend and legitimately names `kernels/vk/` in its module doc, so including
    // src/ turns the tripwire into a false-positive generator — and a check that cries
    // wolf gets deleted. KNOWN LIMIT, stated rather than papered over: an oracle placed
    // in a `#[cfg(test)] mod tests` inside src/ is outside this check's reach. The rule
    // still applies there; only the enforcement stops at the directory boundary.
    let mut files = Vec::new();
    rs_files(std::path::Path::new(dir), &mut files);

    let mut scanned = Vec::new();
    for path in files {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        // The scanner excludes ITSELF structurally, not by allowlist — it is not an
        // oracle, and listing it would imply the rule has two exceptions when it has
        // one.
        if name == me || ALLOWED.contains(&name.as_str()) {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read test file");
        assert!(
            !src.contains(&needle),
            "\n\n{name} references the Vulkan shader directory.\n\
             Oracles must be derived from the HIP original and src/math.rs — the \
             SPECIFICATION — not from the shader they check. An oracle copied from the \
             implementation agrees with it by construction and will ratify a shared \
             misreading of the HIP instead of catching it.\n\
             If this file genuinely needs to read a shader (as glsl_numerics.rs does, \
             to hash it), add it to ALLOWED in {me} and say why.\n"
        );
        scanned.push(name);
    }
    // A scanner that silently matched nothing would pass forever. Prove it looked.
    assert!(
        scanned.iter().any(|n| n == "vk.rs"),
        "scanned {scanned:?} but not vk.rs — the tests directory moved, or the filter \
         is wrong, and this check has been passing without examining the file it \
         exists to police"
    );
}
