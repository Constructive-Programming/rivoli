//! Build the Rust device crate (`crates/kernels`) and link libamdhip64 — only under the
//! `rocm` feature, so `cargo check` / CPU-side dev needs neither nightly nor ROCm.
//!
//! > **2026-11-20 (Rust-kernel Phase 2):** hipcc left the build. Until then this file also
//! > compiled `crates/backend/kernels/*.hip` into `librivolikernels.a`; the C++ now lives
//! > only at tag `archive/hip-kernels`, the bit-identity reference (`rust-kernels.md` §5).
//!
//! Build scripts panic on a broken toolchain; that is the correct failure mode.
#![allow(clippy::expect_used)]

use std::path::Path;

/// The HIP runtime ships in a different directory on every distro, so probe rather than
/// hard-code one; the first hit wins and the rest are not searched.
fn hip_runtime_dir() -> Option<&'static str> {
    ["/usr/lib64", "/opt/rocm/lib", "/usr/lib"]
        .into_iter()
        .find(|dir| Path::new(dir).join("libamdhip64.so").exists())
}

// The Rust device crate's build — its own file, because it is a different toolchain (the
// pinned nightly) doing a different job from this host-side link.
#[path = "build/device.rs"]
mod device;

fn main() {
    if std::env::var("CARGO_FEATURE_ROCM").is_err() {
        return; // CPU-only build: no HIP toolchain required.
    }
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    // `src/module.rs` embeds it with `include_bytes!(env!("RIVOLI_KERNELS_HSACO"))`.
    println!(
        "cargo:rustc-env=RIVOLI_KERNELS_HSACO={}",
        device::rivoli_kernels(Path::new(&out_dir)).display()
    );
    if let Some(dir) = hip_runtime_dir() {
        println!("cargo:rustc-link-search=native={dir}");
    }
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
