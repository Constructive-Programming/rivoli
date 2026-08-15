//! Compile the HIP kernels with hipcc and link libamdhip64 — only under the `rocm`
//! feature, so `cargo check` / CPU-side dev needs no GPU toolchain. Ported from the old
//! tree's build.rs HIP half (`wt/glimmer-s2` @ 6b7f496); the duplication gate that shared
//! that file lives in `crates/cli/build.rs` now — one gate per concern, one crate each.
//!
//! Build scripts panic on a broken toolchain; that is the correct failure mode.
#![allow(clippy::expect_used)]

use std::path::Path;
use std::process::Command;

/// Every `.hip` translation unit that goes into the archive. One list, so a kernel added
/// here is compiled, archived and rerun-tracked by the same pass.
const KERNELS: [&str; 12] = [
    "linalg",
    // Split out of `linalg.hip` 2026-08-15 (the per-token activation transforms), and
    // listed next to it because the pair is one cut, not two subsystems.
    "activation",
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
];

/// EVERY header under `kernels/`, DISCOVERED rather than listed — the result is BOTH a
/// rerun trigger and a staleness input, and the two must never drift apart or an edited
/// header rebuilds nothing.
///
/// It was one hard-coded entry (`common.hpp`) until 2026-08-15, and that was already a hole
/// rather than a simplification: `formats.hpp` and `reduce.hpp` split out of it that
/// morning and neither was added, so an edit to either recompiled NOTHING and linked the
/// previous objects. The dot-family split the same day would have widened the hole to
/// eight. Unlike `KERNELS` — where a missing name means a translation unit is never
/// compiled, so the list has to be explicit and is checked by the linker — a missing HEADER
/// name is silent, which is exactly the case for reading the directory instead. Sorted so
/// the emitted `rerun-if-changed` lines do not churn with directory order.
///
/// A read failure yields an EMPTY list, which makes every object look stale and forces a
/// full recompile: the safe direction. Returning "nothing changed" from an unreadable
/// directory is the failure this function exists to prevent.
fn headers() -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir("kernels")
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "hpp"))
        .map(|e| format!("kernels/{}", e.file_name().to_string_lossy()))
        .collect();
    out.sort();
    out
}

/// True when `out` is missing or older than any of `deps` — the mtime comparison cargo
/// does for its own targets, which build scripts do not get for free. Missing metadata
/// means REBUILD: an unreadable timestamp is not evidence the output is current.
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

/// The four settings every kernel compile shares, resolved once — they co-vary (the arch
/// reaches both the flag and the object name), so they travel as one value.
struct Toolchain {
    out_dir: String,
    hipcc: String,
    arch: String,
    /// Derived from `arch`, hoisted so the loop does not re-format it per kernel.
    offload: String,
}

impl Toolchain {
    fn from_env() -> Self {
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
        let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "hipcc".into());
        // ponytail: default gfx1151 (Strix Halo); override for other ROCm nodes with
        // RIVOLI_OFFLOAD_ARCH. (An env var — the build-script carve-out argument from the old
        // tree applies verbatim: cargo already configures this script through the
        // environment, and rerun-if-env-changed puts the toggle inside cargo's fingerprint.)
        println!("cargo:rerun-if-env-changed=RIVOLI_OFFLOAD_ARCH");
        println!("cargo:rerun-if-env-changed=HIPCC");
        let arch = std::env::var("RIVOLI_OFFLOAD_ARCH").unwrap_or_else(|_| "gfx1151".into());
        let offload = format!("--offload-arch={arch}");
        Self {
            out_dir,
            hipcc,
            arch,
            offload,
        }
    }

    /// Compile one kernel unless its object is already current, and return that object.
    /// `hdrs` is `headers()`, resolved once by the caller rather than re-read per kernel.
    fn object(&self, kernel: &str, hdrs: &[&str]) -> String {
        let src = format!("kernels/{kernel}.hip");
        // The ARCH is part of the object's identity, not just its mtime: review 2026-08-15
        // found that switching RIVOLI_OFFLOAD_ARCH re-ran this script (rerun-if-env-changed)
        // but `stale()` then found every .o "fresh" and silently linked the OLD arch's
        // objects. Baking the arch into the filename makes a switched arch a cache MISS.
        let obj = format!("{}/{kernel}.{}.o", self.out_dir, self.arch);
        println!("cargo:rerun-if-changed={src}");
        // SKIP an object that is newer than its source AND every shared header — without
        // this every re-run recompiled all the kernels through hipcc, seconds of rebuild
        // for a change that cannot affect a single one of them.
        let mut deps: Vec<&str> = vec![&src];
        deps.extend_from_slice(hdrs);
        if stale(&obj, &deps) {
            self.run_hipcc(&src, &obj);
        }
        obj
    }

    fn run_hipcc(&self, src: &str, obj: &str) {
        let mut cmd = Command::new(&self.hipcc);
        cmd.args([&self.offload, "-O3", "-fPIC"]);
        let status = cmd
            .args(["-c", src, "-o", obj])
            .status()
            .expect("run hipcc");
        assert!(status.success(), "hipcc failed on {src}");
    }
}

/// Bundle the objects into the one static library rustc is told to link.
fn archive(out_dir: &str, objs: &[String]) {
    let lib = format!("{out_dir}/librivolikernels.a");
    // `ar crs` only adds/replaces members — it never prunes ones dropped from the list,
    // so a since-deleted kernel's stale .o lingers and duplicate-symbol-clashes. Start
    // from a clean archive every build.
    let _ = std::fs::remove_file(&lib);
    let mut ar = Command::new("ar");
    ar.arg("crs").arg(&lib);
    for o in objs {
        ar.arg(o);
    }
    assert!(ar.status().expect("run ar").success(), "ar failed");
}

/// The HIP runtime ships in a different directory on every distro, so probe rather than
/// hard-code one; the first hit wins and the rest are not searched.
fn hip_runtime_dir() -> Option<&'static str> {
    ["/usr/lib64", "/opt/rocm/lib", "/usr/lib"]
        .into_iter()
        .find(|dir| Path::new(dir).join("libamdhip64.so").exists())
}

fn emit_link_flags(out_dir: &str) {
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=rivolikernels");
    if let Some(dir) = hip_runtime_dir() {
        println!("cargo:rustc-link-search=native={dir}");
    }
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}

fn main() {
    if std::env::var("CARGO_FEATURE_ROCM").is_err() {
        return; // CPU-only build: no HIP toolchain required.
    }
    let tc = Toolchain::from_env();
    // Headers are #included, not compiled units — track them so an edit forces a rebuild.
    let hdrs = headers();
    let hdrs: Vec<&str> = hdrs.iter().map(String::as_str).collect();
    for h in &hdrs {
        println!("cargo:rerun-if-changed={h}");
    }
    let objs: Vec<String> = KERNELS.iter().map(|k| tc.object(k, &hdrs)).collect();
    archive(&tc.out_dir, &objs);
    emit_link_flags(&tc.out_dir);
}
