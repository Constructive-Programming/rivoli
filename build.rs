//! Build script: panics are the correct failure mode for a broken toolchain.
#![allow(clippy::expect_used)]

//! Compile the HIP kernels with hipcc and link libamdhip64 + liburing — only under
//! the `rocm` feature, so `cargo check` / CPU-side dev needs no GPU toolchain.

use std::path::Path;
use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_ROCM").is_err() {
        return; // CPU-only build: no HIP toolchain required.
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let kernels = ["linalg", "moe", "mla", "attn", "fwd", "vmm", "stream"];
    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "hipcc".into());
    let trace = std::env::var("CARGO_FEATURE_TRACE").is_ok();

    let mut objs = Vec::new();
    for k in kernels {
        let src = format!("kernels/{k}.hip");
        let obj = format!("{out_dir}/{k}.o");
        println!("cargo:rerun-if-changed={src}");
        let mut cmd = Command::new(&hipcc);
        cmd.args(["--offload-arch=gfx1151", "-O3", "-fPIC"]);
        if trace {
            cmd.arg("-DRIVOLI_TRACE");
        }
        let status = cmd
            .args(["-c", &src, "-o", &obj])
            .status()
            .expect("run hipcc");
        assert!(status.success(), "hipcc failed on {src}");
        objs.push(obj);
    }

    let lib = format!("{out_dir}/librivolikernels.a");
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
    println!("cargo:rustc-link-lib=dylib=uring"); // kernels/stream.hip io_uring streamer
}
