//! Build script: panics are the correct failure mode for a broken toolchain.
#![allow(clippy::expect_used)]

//! Compile the HIP kernels with hipcc and link libamdhip64 — only under the `rocm`
//! feature, so `cargo check` / CPU-side dev needs no GPU toolchain. (The cold-expert
//! io_uring streamer is the `io-uring` crate now, talking to the syscalls directly —
//! no liburing system lib.)

use std::path::Path;
use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_VULKAN").is_ok() {
        return vulkan(); // second backend: glslc instead of hipcc, no HIP at all.
    }
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

/// The `vulkan` arm: compile every `kernels/vk/*.comp` to SPIR-V in OUT_DIR, which
/// `src/vk.rs` picks up with `include_bytes!`. Nothing is linked — the loader is
/// opened at runtime by `ash`, so a Vulkan build needs no GPU library at link time.
fn vulkan() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| "glslc".into());
    println!("cargo:rerun-if-env-changed=GLSLC");
    // The DIRECTORY entry catches an added/removed shader; the per-file entries below
    // catch edits. Both are needed — and `common.glsl` is #included, not compiled, so
    // without its own entry an edit to it silently ships stale SPIR-V (exactly the
    // common.hpp staleness bug this repo already hit once).
    println!("cargo:rerun-if-changed=kernels/vk");

    let mut shaders: Vec<_> = std::fs::read_dir("kernels/vk")
        .expect("read kernels/vk")
        .map(|e| e.expect("dir entry").path())
        .collect();
    shaders.sort(); // deterministic build order
    for path in &shaders {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    for src in shaders.iter().filter(|p| p.extension().is_some_and(|e| e == "comp")) {
        let stem = src.file_stem().expect("shader stem").to_string_lossy();
        let spv = format!("{out_dir}/{stem}.spv");
        let status = Command::new(&glslc)
            .args(["--target-env=vulkan1.3", "-O", "-Ikernels/vk"])
            .arg(src)
            .args(["-o", &spv])
            .status()
            .expect("run glslc");
        assert!(status.success(), "glslc failed on {}", src.display());
        spirv_val(&spv);
    }
}

/// Validate a compiled module with `spirv-val`.
///
/// This is STATIC module validation — it catches malformed SPIR-V, bad decorations,
/// and capability/extension mismatches. It does NOT see synchronisation, descriptor,
/// or buffer-device-address misuse; only the VK_LAYER_KHRONOS_validation runtime layer
/// does, and that layer is a separate install (see docs/VULKAN.md "Risks").
///
/// A missing `spirv-val` warns rather than fails: it ships in a different package from
/// glslc, and a box that can build shaders should not be blocked on the checker.
fn spirv_val(spv: &str) {
    match Command::new("spirv-val").arg(spv).status() {
        Ok(status) => assert!(status.success(), "spirv-val rejected {spv}"),
        Err(e) => println!("cargo:warning=spirv-val not run on {spv} ({e}); SPIR-V unchecked"),
    }
}
