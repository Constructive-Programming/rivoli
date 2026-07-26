//! Panics are the right failure mode for a broken toolchain in a hand-run probe.
#![allow(clippy::expect_used)]

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=oob.comp");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let status = Command::new("glslc")
        .args(["--target-env=vulkan1.3", "oob.comp", "-o"])
        .arg(format!("{out}/oob.spv"))
        .status()
        .expect("run glslc");
    assert!(status.success(), "glslc failed on oob.comp");
}
