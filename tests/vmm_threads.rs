//! **Two `#[test]`s that both build engines, in one binary — and that is the whole point.**
//!
//! libtest gives every `#[test]` its own thread, and HIP's VMM handles are thread-affine on this
//! runtime: creating a mapping on one thread and another on a second crashes inside
//! `libamdhip64` a few times in a hundred. `docs/investigations/glimmer-open-items.md` §4b has
//! the bisect; `memory::device`'s `vmm_thread` carries the fix, which is to marshal every
//! `rivoli_vmm_alloc`/`rivoli_vmm_free` onto one long-lived thread.
//!
//! **The SHAPE is the test.** Nothing here asserts a number — what it exercises is two threads
//! reaching the VMM API in one process, and the failure mode is a SIGSEGV rather than a failed
//! assertion. Merging these two into one `#[test]` would silently retire the gate, so they have
//! to stay two; a comment cannot enforce that, which is why this paragraph explains a split that
//! otherwise looks arbitrary.
//!
//! Sized small on purpose: the defect showed at ~25% of runs with 20 cycles per test, so a
//! handful catches a regression without making this a slow suite. A GPU arm — it needs the
//! device, the flock and `--test-threads=1` like every other device suite.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;
use common::{GLIMMER_FIXTURE_DIM as DIM, TempRoot, glimmer_convert_fixture};
use rivoli::artifact::model as gm;
use rivoli::glimmer_gpu::Glimmer;

/// Build, decode and drop `n` engines, varying the context so each asks for a differently-sized
/// tier — the shape `glimmer_reference.rs` has, and the one that surfaced the crash.
fn cycle(tag: &str, n: usize) {
    let root = TempRoot::new(tag);
    let _ = glimmer_convert_fixture(root.path(), DIM);
    let cfg: gm::GlimmerConfig = gm::load_config(root.join("out").to_str().unwrap()).unwrap();
    for i in 0..n {
        let mut e = Glimmer::new(
            root.join("out").to_str().unwrap(),
            &cfg.text,
            None,
            13 + i % 5,
        )
        .unwrap();
        let _ = e.decode(&[1, 2, 3], 1, &[]).unwrap();
        let _ = e.logits().unwrap();
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
