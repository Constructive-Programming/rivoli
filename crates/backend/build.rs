//! Compile the HIP kernels with hipcc and link libamdhip64 — only under the `rocm`
//! feature, so `cargo check` / CPU-side dev needs no GPU toolchain. Ported from the old
//! tree's build.rs HIP half (`wt/glimmer-s2` @ 6b7f496); the duplication gate that shared
//! that file lives in `crates/cli/build.rs` now — one gate per concern, one crate each.
//!
//! Build scripts panic on a broken toolchain; that is the correct failure mode.
#![allow(clippy::expect_used)]

use std::path::Path;
use std::process::Command;

fn main() {
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
    ];
    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "hipcc".into());
    // ponytail: default gfx1151 (Strix Halo); override for other ROCm nodes with
    // RIVOLI_OFFLOAD_ARCH. (An env var — the build-script carve-out argument from the old
    // tree applies verbatim: cargo already configures this script through the
    // environment, and rerun-if-env-changed puts the toggle inside cargo's fingerprint.)
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
        // SKIP an object that is newer than both its source and the shared header —
        // without this every re-run recompiled all the kernels through hipcc, seconds of
        // rebuild for a change that cannot affect a single one of them.
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
    // `ar crs` only adds/replaces members — it never prunes ones dropped from the list,
    // so a since-deleted kernel's stale .o lingers and duplicate-symbol-clashes. Start
    // from a clean archive every build.
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
