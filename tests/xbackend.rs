//! Cross-backend byte comparison — the first in this repo, and the acceptance gate's
//! method in miniature.
//!
//! WHY THIS FILE HAS TO EXIST. Every other test here compares ONE backend against a CPU
//! reference. That is the right instrument for almost everything, and it is structurally
//! incapable of settling the one question docs/investigations/vulkan-port.md's token-ID gate actually asks:
//! *do the two backends agree with each other?* Where a kernel's result depends on a
//! HARDWARE function — `v_exp_f32` on gfx1151 — the CPU has no way to evaluate it, so a
//! Rust oracle is a third implementation and its disagreement with either GPU proves
//! nothing about the pair. `hip_expf` is exactly that case: it reproduces HIP's argument
//! reduction instruction for instruction, and a CPU oracle still failed 122 of 255
//! elements because Rust's `f32::exp2` is not the hardware's.
//!
//! HOW IT WORKS, and why it is two runs rather than one. `rocm` and `vulkan` are mutually
//! exclusive features (`backend.rs` makes that a compile error), so no single binary can
//! hold both. Each arm therefore runs under its own feature, writes its raw output bytes
//! to the path in `RIVOLI_XBACKEND_OUT`, and a third step compares the files:
//!
//! ```sh
//! RIVOLI_XBACKEND_OUT=/tmp/x/hip.bin cargo test --features rocm   --test xbackend -- --ignored
//! RIVOLI_XBACKEND_OUT=/tmp/x/vk.bin  cargo test --features vulkan --test xbackend -- --ignored
//! cmp /tmp/x/hip.bin /tmp/x/vk.bin
//! ```
//!
//! `#[ignore]` by default: it needs the env var and a device, and a normal `cargo test`
//! run should not silently produce half a comparison. The INPUTS are generated from a
//! fixed formula in this file, not read from disk, so the two arms cannot drift apart in
//! their data — the same reason the per-kernel oracles seed from a fixed `Lcg`.
//!
//! # FIRST RESULT: `swiglu` DIFFERS, and reproducing HIP's `expf` did not fix it
//!
//! The measurement and the argument built on it are in
//! `docs/investigations/vulkan-port.md` §"1463 of 4096": reproducing hipcc's `expf`
//! argument reduction instruction-for-instruction left that count UNCHANGED, moving only
//! 12 values, so the residue is in `v_exp_f32` itself and `exp` stays PRE-REGISTERED.
//!
//! Deliberately a pointer and not a copy. This repo corrects a claim in one place, and a
//! measurement restated in a doc comment is the copy that goes stale — the same failure
//! `tests/docs.rs` exists to catch one level up.
//!
//! The harness is kept because the question it answers cannot be answered any other way,
//! and phase 4's token-ID gate is this comparison scaled up.
#![allow(clippy::expect_used)]

/// The shared input set. A pure function of the index, so both arms compute identical
/// bytes without exchanging anything but the output.
///
/// Spans the whole interesting range of `expf`: both saturation clamps HIP applies
/// (~-103.28 underflows to zero, ~88.72 overflows to +inf), the region where the argument
/// reduction's integer part `n` is large, and the near-zero region where it is zero and
/// the two-word correction is all that matters.
fn inputs(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32; // 0..1
            -120.0 + t * 240.0 // -120 .. +120, straddling both clamps
        })
        .collect()
}

const N: usize = 4096;

fn out_path() -> Option<String> {
    std::env::var("RIVOLI_XBACKEND_OUT").ok()
}

fn write_out(bytes: &[u8]) {
    let Some(p) = out_path() else {
        panic!("set RIVOLI_XBACKEND_OUT to the file this arm should write");
    };
    std::fs::write(&p, bytes).unwrap_or_else(|e| panic!("write {p}: {e}"));
    println!("wrote {} bytes to {p}", bytes.len());
}

/// `swiglu` under whichever backend this binary was built with.
///
/// swiglu is the smallest kernel whose result depends on `exp`, which makes it the
/// cheapest possible probe for the question. `u` is all ones so the output is exactly
/// `silu(g)` and nothing else can absorb a difference.
#[test]
#[ignore = "needs RIVOLI_XBACKEND_OUT and a GPU; run one arm per backend"]
fn swiglu_bytes() {
    let g = inputs(N);
    let u = vec![1.0f32; N];
    let bytes = run_swiglu(&g, &u);
    write_out(&bytes);
}

#[cfg(all(feature = "rocm", not(feature = "vulkan")))]
use rivoli::backend::hip::{device_sync, launch_swiglu};
#[cfg(all(feature = "vulkan", not(feature = "rocm")))]
use rivoli::backend::vk::{Buf as Dev, device_sync, launch_swiglu};
#[cfg(all(feature = "rocm", not(feature = "vulkan")))]
use rivoli::memory::device::DeviceBuf as Dev;

/// A device buffer holding `v`. Upload is the FIRST of the two spellings that differ
/// between the backends — `copy_in_at` under HIP, `write_at` under Vulkan.
#[cfg(all(feature = "rocm", not(feature = "vulkan")))]
fn up(v: &[f32]) -> Dev {
    let mut d = Dev::new(std::mem::size_of_val(v)).expect("alloc");
    d.copy_in_at(0, bytemuck_f32(v)).expect("fill");
    d
}

#[cfg(all(feature = "vulkan", not(feature = "rocm")))]
fn up(v: &[f32]) -> Dev {
    let mut d = Dev::new(std::mem::size_of_val(v)).expect("alloc");
    d.write_at(0, bytemuck_f32(v)).expect("fill");
    d
}

/// `bytes` back from the device. The SECOND spelling that differs: HIP hands back the
/// whole allocation, Vulkan fills a caller's buffer to a length.
#[cfg(all(feature = "rocm", not(feature = "vulkan")))]
fn down(d: &Dev, _bytes: usize) -> Vec<u8> {
    d.copy_out().expect("copy out")
}

#[cfg(all(feature = "vulkan", not(feature = "rocm")))]
fn down(d: &Dev, bytes: usize) -> Vec<u8> {
    let mut out = Vec::new();
    d.read_into(&mut out, bytes).expect("read");
    out
}

/// The dispatch itself is backend-neutral — `launch_swiglu` takes raw device addresses
/// under both — so the arms above are the only cfg in this file. Two copies of this body
/// is two chances for the arms to stop dispatching the same work, which would make the
/// byte comparison this file exists for compare the wrong pair.
#[cfg(any(feature = "rocm", feature = "vulkan"))]
fn run_swiglu(g: &[f32], u: &[f32]) -> Vec<u8> {
    let n = g.len();
    let (gb, ub) = (up(g), up(u));
    let mut hb = Dev::new(n * 4).expect("h");
    // SAFETY: three live device addresses of n f32 each, joined before they drop.
    unsafe {
        launch_swiglu(
            gb.ptr() as *const f32,
            ub.ptr() as *const f32,
            n,
            hb.ptr_mut() as *mut f32,
        )
        .expect("launch");
    }
    device_sync().expect("sync");
    down(&hb, n * 4)
}

#[cfg(not(any(feature = "rocm", feature = "vulkan")))]
fn run_swiglu(_g: &[f32], _u: &[f32]) -> Vec<u8> {
    panic!("build with --features rocm or --features vulkan");
}

/// f32 slice as bytes, without pulling in a dependency for four lines.
#[allow(dead_code)]
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and no invalid bit patterns; the lifetime is borrowed
    // from `v` and the length is scaled to match.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
