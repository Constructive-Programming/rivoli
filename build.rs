//! Build script: panics are the correct failure mode for a broken toolchain.
#![allow(clippy::expect_used)]

//! Compile the HIP kernels with hipcc and link libamdhip64 — only under the `rocm`
//! feature, so `cargo check` / CPU-side dev needs no GPU toolchain. (The cold-expert
//! io_uring streamer is the `io-uring` crate now, talking to the syscalls directly —
//! no liburing system lib.)

use std::path::Path;
use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_ROCM").is_err() {
        return; // CPU-only build: no HIP toolchain required.
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let kernels = [
        "linalg", "moe", "mla", "attn", "fwd", "vmm", "indexer", "async",
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
