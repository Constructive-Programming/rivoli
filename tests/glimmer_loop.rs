//! **S3: the two decisions the layer loop makes per layer, gated without a device.**
//!
//! `glimmer_gpu.rs` needs 55.7 GB of weights to run one token, so every rule it enforces that a
//! test can only reach through a decode is a rule nothing here runs. Two of them do not need one —
//! which cache a layer reads and which scale each of Q and K takes — and both are traps this port
//! has already recorded as unreachable-until-a-call-site-exists. They are reachable now.
//!
//! # What this covers, and what it does NOT
//!
//! It covers the SELECTION: given a layer's kind, what `win`/`ring_cap`/`slot` go to the launcher,
//! and which scale goes to which of Q and K. It says nothing about the arithmetic — `gqa_attend`
//! and `rmsnorm_weightless_batch` are scored against the anchor goldens by `glimmer_attend.rs` and
//! `glimmer_qk_norm.rs`, and this file would pass with either kernel replaced by a stub.
//!
//! **It also cannot see a call site that ignores these functions.** `kernel_coverage.rs`'s OWNERS
//! census is the half that catches a loop calling neither, and it now lists `glimmer_gpu.rs`
//! against both launchers. Between the two: the census proves the kernel is called, and this file
//! proves the values it is called with are derived rather than assumed. Neither proves the loop
//! decodes; that is G3's, and it needs the tiny checkpoint's weights, which are not vendored — the
//! goldens hold activations only.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

use rivoli::artifact::model::LayerKind;
use rivoli::glimmer_gpu::{qk_scales, window_of};

/// Muse Glimmer's shipped window, and a context comfortably past it so the ring wraps.
const WIN: usize = 2048;
const CTX: usize = 5000;

/// **The pair that silently truncates a full layer cannot be constructed.**
///
/// `launch_gqa_attend` derives its causal bound from `(start_pos, win)` and its slot map from
/// `ring_cap`, and its guard rejects only `ring_cap != 0` with `ring_cap < win + tq - 1`. The
/// INVERSE — `win != 0` with `ring_cap == 0` on a full-attention layer — is accepted, and it
/// bounds a global layer to its last `win` rows: fluent, wrong, and invisible to any shape check.
/// The plan named it as the first thing this loop could get wrong.
///
/// Swept over both kinds and a range of positions rather than asserted at one, because the failure
/// this guards against is a `match` arm, and a single position cannot tell an arm from a constant.
#[test]
fn a_full_layer_can_never_be_handed_a_window_without_a_ring() {
    let mut full = 0;
    let mut sliding = 0;
    for pos in [0, 1, WIN - 1, WIN, WIN + 1, 2 * WIN, CTX - 1] {
        for kind in [LayerKind::SlidingAttention, LayerKind::FullAttention] {
            let w = window_of(kind, WIN, CTX, pos);
            assert!(
                !(w.win != 0 && w.ring_cap == 0),
                "{kind:?} at {pos} asked for a {}-row window against a LINEAR cache — the \
                 launcher accepts that pair and truncates the causal prefix to the last {} rows",
                w.win,
                w.win
            );
            match kind {
                // A full layer attends the whole prefix and indexes the cache BY POSITION, so its
                // slot is the position and its cache must run from 0.
                LayerKind::FullAttention => {
                    assert_eq!((w.win, w.ring_cap, w.slot), (0, 0, pos));
                    full += 1;
                }
                // A sliding layer's ring is exactly `sliding_window` — the launcher's floor of
                // `win + tq - 1` at the `tq == 1` this loop decodes at.
                LayerKind::SlidingAttention => {
                    assert_eq!((w.win, w.ring_cap, w.slot), (WIN, WIN, pos % WIN));
                    sliding += 1;
                }
            }
        }
    }
    // A census, because both arms above are inside a loop whose iteration count a future edit
    // could take to zero while every assertion still "passes".
    assert_eq!((full, sliding), (7, 7), "the sweep did not run both arms");
}

/// **A run shorter than the window still gets a ring it can index.**
///
/// `slot` is `pos % ring_cap`, so a `ring_cap` of 0 would divide by zero and a `ring_cap` above
/// the allocation would write past it. `Glimmer::new` sizes each layer's cache from this same
/// function, which is what keeps the two from disagreeing — the reason the clamp is here and not
/// at the allocation.
#[test]
fn a_context_shorter_than_the_window_clamps_the_ring_to_the_context() {
    let w = window_of(LayerKind::SlidingAttention, WIN, 12, 5);
    assert_eq!((w.win, w.ring_cap, w.slot), (WIN, 12, 5));
    // The window handed to the kernel is still the model's, not the clamp: a shorter run does not
    // make a sliding layer attend further back than it was trained to.
    assert_eq!(w.win, WIN);
    // And the linear layers are unaffected — the clamp is the ring's, not the context's.
    assert_eq!(window_of(LayerKind::FullAttention, WIN, 12, 5).ring_cap, 0);
}

/// **Q takes `qk_scale_factor` and K takes 1.0 — trap 3, and it had no gate at all until a call
/// site existed.**
///
/// Every scoring path in this tree hands the same scale to the kernel and to the oracle, so no
/// numeric test can observe which one a caller chose; `glimmer_qk_norm.rs` would pass with the two
/// swapped. What makes the swap catchable is that the values arrive at the launcher by NAME, and
/// this asserts the names carry the right numbers.
///
/// It does NOT catch a call site that reads `s.q` twice. That one is a rename away from correct
/// rather than an argument order away, which is the whole reason for the named pair — but it is a
/// narrowing, not a closure, and saying so is the point of this paragraph.
#[test]
fn the_qk_scale_is_qs_alone_and_k_takes_unity() {
    let s = qk_scales(3.87);
    assert_eq!(
        s.q, 3.87,
        "Q takes qk_scale_factor, after the weightless norm"
    );
    assert_eq!(
        s.k, 1.0,
        "K takes 1.0 — a K scaled by qk_scale_factor is fluent and wrong, and the goldens gate \
         the reference rather than the caller"
    );
    // The identity case is the one that would make a swapped call site pass, so it is asserted to
    // be the only value at which the two agree.
    let one = qk_scales(1.0);
    assert_eq!((one.q, one.k), (1.0, 1.0));
    // And the scale is passed through, not derived: a version that returned a constant would pass
    // the first assertion alone.
    assert_eq!(qk_scales(0.5).q, 0.5);
}
