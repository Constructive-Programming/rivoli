//! **Two `#[test]`s that both build engines, in one binary — and that is the whole point.**
//!
//! libtest gives every `#[test]` its own thread, and on **ROCm 7.2.53210** two threads reaching
//! HIP's VMM API in one process crashed inside `libamdhip64` — 2 runs in 8 of this shape, ~3 in 52
//! of `glimmer_reference`, with nothing of rivoli's below the fault. **It was an upstream bug and
//! ROCm 7.14.0 fixed it**: 0 in 20 with no workaround in the tree at all. There is no rivoli-side
//! fix to name here, because the one that was written — a dedicated VMM thread — was measured
//! unnecessary on 7.14 and removed. `docs/investigations/hip-vmm-segv.md` has the bisect.
//!
//! **The SHAPE is the test.** Nothing here asserts a number — what it exercises is two threads
//! reaching the VMM API in one process, and the failure mode is a SIGSEGV rather than a failed
//! assertion. Merging these two into one `#[test]` would silently retire the gate, so they have to
//! stay two; a comment cannot enforce that, which is why this paragraph explains a split that
//! otherwise looks arbitrary.
//!
//! **What six cycles per test does and does not buy, since the number looks like a measurement.**
//! Every rate above was measured per RUN of a whole suite, never per allocate/free cycle, so
//! nothing here can claim a detection probability — only that the two-thread shape is exercised on
//! every dev-profile run for well under a second. If a regression ever slips past, raise it.
//!
//! A GPU arm — it needs the device, the flock and `--test-threads=1` like every other device suite.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;
use common::{GLIMMER_FIXTURE_DIM as DIM, TempRoot, decode_one, glimmer_convert_fixture};
use rivoli::artifact::model as gm;

/// Build, decode and drop `n` engines, varying the context so each sizes its KV cache and
/// activations differently — the shape `glimmer_reference.rs` has, and the one that surfaced the
/// crash. The weight tier is the whole model at every iteration (`budget: None`), so the variation
/// is in the per-run allocations rather than in the pin.
fn cycle(tag: &str, n: usize) {
    let root = TempRoot::new(tag);
    let _ = glimmer_convert_fixture(root.path(), DIM);
    let cfg: gm::GlimmerConfig = gm::load_config(root.join("out").to_str().unwrap()).unwrap();
    for i in 0..n {
        // Dropped immediately: one allocate and one free per iteration is the churn under test,
        // and an engine held across the loop would be neither.
        let prompt: Vec<u32> = (1..=3 + i as u32 % 5).collect();
        drop(decode_one(&root.join("out"), &cfg.text, &prompt));
    }
}

#[test]
fn engines_build_and_drop_on_this_thread() {
    cycle("vmm-a", 6);
}

/// The second thread. See the module header: this is not a copy of the test above, it is the
/// other half of the condition under test.
#[test]
fn engines_build_and_drop_on_a_second_thread_too() {
    cycle("vmm-b", 6);
}
