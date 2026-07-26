//! Panics are the right failure mode for a broken toolchain in a hand-run probe.
#![allow(clippy::expect_used)]

use std::process::Command;

fn main() {
    let out = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    for shader in ["oob", "descwrite"] {
        println!("cargo:rerun-if-changed={shader}.comp");
        let status = Command::new("glslc")
            .args(["--target-env=vulkan1.3", &format!("{shader}.comp"), "-o"])
            .arg(format!("{out}/{shader}.spv"))
            .status()
            .expect("run glslc");
        assert!(status.success(), "glslc failed on {shader}.comp");
    }
}
