//! The device half of `build.rs`: build `crates/kernels` with its pinned nightly and hand
//! the code object to `src/module.rs`. Only reached under the `rocm` feature, so the
//! deviceless arm never needs nightly, rustup or ROCm's device libraries.

use super::Toolchain;
use std::path::{Path, PathBuf};
use std::process::Command;

/// ROCm's device-library directory (`ocml.bc` and the `oclc_*` control libraries hipcc
/// links). Probed like the runtime below; `RIVOLI_DEVICE_LIBS` overrides, for the same
/// build-script-environment reason `RIVOLI_OFFLOAD_ARCH` gives.
fn device_libs() -> PathBuf {
    println!("cargo:rerun-if-env-changed=RIVOLI_DEVICE_LIBS");
    if let Some(d) = std::env::var_os("RIVOLI_DEVICE_LIBS") {
        return d.into();
    }
    [
        "/usr/lib/amdgcn/bitcode",
        "/opt/rocm/amdgcn/bitcode",
        "/opt/rocm/lib/llvm/lib/clang/amdgcn/bitcode",
    ]
    .into_iter()
    .find(|d| Path::new(d).join("ocml.bc").exists())
    .expect("ROCm device libraries (ocml.bc) not found; set RIVOLI_DEVICE_LIBS")
    .into()
}

/// The pinned channel, read from the file rustup itself would read — one spelling of the pin.
fn device_toolchain(kdir: &Path) -> String {
    let pin = std::fs::read_to_string(kdir.join("rust-toolchain.toml")).expect("read the pin");
    pin.lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("channel = ")
                .map(|c| c.trim_matches('"').to_string())
        })
        .expect("rust-toolchain.toml names a channel")
}

/// The device rustflags. Every item is something hipcc does for its own kernels, restated
/// for rustc:
///
/// - `-Clinker-plugin-lto` + the `.bc` link args: the device libraries join the LTO, so a
///   `__ocml_*` call inlines the same body hipcc inlined (`-mlink-builtin-bitcode`); the
///   `oclc_*` set is exactly hipcc's (`hipcc -###`, 2026-11-18).
/// - `--undefined-version`: rustc's version script names each kernel's `.kd` descriptor,
///   which under linker-plugin LTO does not exist until LTO has run — lld refuses the
///   script without this (measured: `version script assignment of 'global' to symbol
///   't.kd' failed`).
fn device_rustflags(tc: &Toolchain, libs: &Path) -> Vec<String> {
    let isa = tc.arch.trim_start_matches("gfx");
    let mut f = vec![
        format!("-Ctarget-cpu={}", tc.arch),
        "-Clinker-plugin-lto".into(),
        "-Clink-arg=--undefined-version".into(),
    ];
    let bcs = [
        "ocml",
        "oclc_finite_only_off",
        "oclc_unsafe_math_off",
        "oclc_wavefrontsize64_off",
        "oclc_abi_version_600",
    ];
    let isa_bc = format!("oclc_isa_version_{isa}");
    for bc in bcs.iter().copied().chain([isa_bc.as_str()]) {
        let path = libs.join(format!("{bc}.bc"));
        // A ROCm upgrade replaces these; the hipcc reference rebuilds against the new OCML,
        // so the embedded code object must too.
        println!("cargo:rerun-if-changed={}", path.display());
        f.push(format!("-Clink-arg={}", path.display()));
    }
    f
}

/// Build `crates/kernels` with its pinned nightly into its own target dir under OUT_DIR,
/// and return the code object. `rustup run` names the toolchain explicitly, and the outer
/// cargo's environment is scrubbed of everything that would otherwise reach the nested one
/// — above all `CARGO_TARGET_DIR` (the per-checkout dir this outer build is using; sharing
/// it is the `gate-red-proofs.md` §7e-bis defect) and the outer `CARGO_ENCODED_RUSTFLAGS`
/// (host flags), which is REPLACED rather than removed because it outranks `RUSTFLAGS`.
pub fn rivoli_kernels(tc: &Toolchain) -> PathBuf {
    let kdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels");
    for p in [
        "src",
        "Cargo.toml",
        "rust-toolchain.toml",
        "Cargo.lock",
        "../abi/src",
        "../abi/Cargo.toml",
        // rivoli-abi inherits its edition and lints from the workspace root.
        "../../Cargo.toml",
    ] {
        println!("cargo:rerun-if-changed={}", kdir.join(p).display());
    }
    let target_dir = Path::new(&tc.out_dir).join("kernels-target");
    let flags = device_rustflags(tc, &device_libs()).join("\x1f");
    // The same flags, for `tests/bit-identity.sh --build-probes`: the gate builds the probe
    // object (`--features probes`, from a copy of this crate that may carry a plant patch)
    // outside cargo, and reads the one spelling of the device flags from here rather than
    // restating ROCm's bitcode paths.
    std::fs::write(Path::new(&tc.out_dir).join("device-rustflags"), &flags)
        .expect("write device-rustflags");
    let mut cmd = Command::new("rustup");
    cmd.args([
        "run",
        &device_toolchain(&kdir),
        "cargo",
        "build",
        "--release",
        "--locked",
    ])
    .args([
        "--target",
        "amdgcn-amd-amdhsa",
        "-Zbuild-std=core",
        "--target-dir",
    ])
    .arg(&target_dir)
    .arg("--manifest-path")
    .arg(kdir.join("Cargo.toml"));
    for v in [
        "CARGO_TARGET_DIR",
        "RUSTFLAGS",
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTUP_TOOLCHAIN",
        "CARGO_BUILD_TARGET",
        "CARGO",
    ] {
        cmd.env_remove(v);
    }
    let status = cmd
        .env("CARGO_ENCODED_RUSTFLAGS", flags)
        .status()
        .expect("run rustup (the device build needs rustup and the pinned nightly)");
    assert!(
        status.success(),
        "device crate build failed (crates/kernels)"
    );
    // rustc names the code object `.elf`; it is the image `hipModuleLoadData` takes as-is.
    target_dir.join("amdgcn-amd-amdhsa/release/rivoli_kernels.elf")
}
