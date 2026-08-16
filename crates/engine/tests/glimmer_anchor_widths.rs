//! **The half of M7's exit gate that needs no GPU — and therefore the half CI runs.**
//!
//! `glimmer_anchor_decode.rs` is the gate: it runs [`rivoli_engine::glimmer`]'s own loop on
//! Muse Glimmer's own parameters and scores the logits against the reference's. It is
//! `#![cfg(feature = "rocm")]` end to end, like every `kernel_glimmer_*.rs` sibling, and this
//! workspace has no rocm CI job — so every claim in it is checked exactly as often as someone
//! runs a GPU build by hand.
//!
//! **Three things about that gate are checkable without a device, and all three are ways it
//! silently stops being a gate rather than ways it goes red:**
//!
//! 1. **The capture names.** A renamed or re-vendored golden is the commonest way a
//!    golden-scoring test quietly scores nothing; `float`'s absent-name panic is the check,
//!    and running it here means a rename reddens in CI instead of on the next GPU run.
//! 2. **The config the engine will parse.** The tiny config is spliced into the shipped
//!    wrapper and handed to the same `parse_config` the artifact loader uses, so a
//!    `tiny_config` that stopped describing a model this port can run fails as a schema
//!    refusal here rather than as a construction error there.
//! 3. **Whether these widths DRIVE anything.** The window clamps to the run when the run is
//!    shorter than the window, both layer kinds have to occur, and the prompt has to fit one
//!    prefill chunk. Each of those, wrong, leaves the device gate green over a path it never
//!    entered — the `--attn dsa` vacuity that cost the old tree an A/B, asked as a shape.
//!
//! Plus the one thing about the tolerances a machine can check at all: that each bound still
//! stands the house's ~3x over the envelope it was measured from, in BOTH directions.
//!
//! Its own binary rather than an ungated module inside the device file, on
//! `old:tests/glimmer_tolerance.rs`'s argument: that file asks whether a set of thresholds
//! follows from a set of measurements and this asks whether a fixture is intact and capable —
//! different question, different failure. Mixing them also means the deviceless assertions
//! live in a binary the featureless build does not compile, which is the whole problem.
//!
//! # Red proofs, RUN — no GPU, 2026-08-16
//!
//! Both planted, observed red, reverted, observed green again in one session (P7).
//!
//! Append a `2` to `glimmer_anchor::BRANCH_CAP`:
//!
//! ```text
//! t0.L0.post_feedforward_layernorm2.out is not in the golden; it holds 1099 float
//! tensors, e.g. ["t0.embed_norm.out", "t0.rope.cos", "t0.rope.sin"]
//! test result: FAILED. 1 passed; 1 failed
//! ```
//!
//! Halve `glimmer_anchor::TOL` to 1e-1:
//!
//! ```text
//! TOL is 1.47x its measured envelope 6.8e-2. Under ~2x it reddens on a correct engine …
//! test result: FAILED. 1 passed; 1 failed
//! ```
//!
//! The first is the load-bearing one: it is the whole reason this binary exists, and it fires
//! in the featureless build CI actually runs. The 1,099 in that message is also the census
//! this gate's 15-per-salt coverage is a fraction of.

// The panic-on-failure idiom; the fixture module carries the same allow and the same reason.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod glimmer_anchor;

use glimmer_anchor::{Anchor, BRANCH_CAP, ENVELOPES, anchors, float, ints, is_alias};
use rivoli_artifact::glimmer::GLIMMER_LAYER_TENSORS;
use rivoli_artifact::glimmer_config::{GlimmerTextConfig, LayerKind};
use rivoli_engine::glimmer::geometry;

/// **Everything this gate will read is in the goldens, the config the engine will parse is
/// one the schema accepts, and these widths really drive the paths the device tests claim.**
///
/// The three device tests below are `#[cfg(feature = "rocm")]` and CI has no rocm job, so
/// without this they would be checked exactly as often as someone runs a GPU build by hand.
/// What is checkable without a device is: the capture names (a rename is the commonest way a
/// golden gate silently stops scoring), the config (the schema's own refusals), and the
/// geometry (whether the tiny widths reach the ring, the streaming split and the footprint
/// guards, rather than a clamped degenerate case that passes vacuously).
#[test]
fn the_anchor_carries_what_this_gate_scores() {
    let mut scored = 0usize;
    for a in &anchors() {
        let cfg = a.config();
        let (t, layers) = (&cfg.text, cfg.text.n_layers);
        // Cross-checked against an INDEPENDENT source and both refused at zero. `steps` comes
        // from the golden's metadata and `n_layers` from `tiny_config`, which the driver
        // writes separately, so the two CAN disagree — and a census multiplied from one of
        // them can never notice the other shrinking.
        let steps = a.steps();
        assert!(
            steps > 0 && layers > 0,
            "{}: {steps} steps, {layers} layers",
            a.name
        );
        assert_eq!(
            ints(&a.caps, "emitted.ids").len(),
            steps,
            "{}: one emitted token per captured step",
            a.name
        );
        assert_eq!(
            ints(&a.caps, "layer_is_sliding").len(),
            layers,
            "{}: layer_is_sliding disagrees with num_hidden_layers",
            a.name
        );
        assert_eq!(
            ints(&a.caps, "prompt.ids").len(),
            a.meta_usize("prompt_len"),
            "{}: prompt.ids is not prompt_len long",
            a.name
        );

        // Every name the device tests read, asked for through `float`'s absent-name panic.
        for step in 0..steps {
            assert_eq!(
                float(&a.caps, &format!("t{step}.logits")).0.to_vec(),
                vec![1, t.vocab],
                "{}: t{step}.logits is not one vocabulary row",
                a.name
            );
            scored += 1;
        }
        for l in 0..layers {
            assert_eq!(
                float(&a.caps, &format!("t0.L{l}.{BRANCH_CAP}")).0.to_vec(),
                vec![1, a.meta_usize("prompt_len"), t.hidden],
                "{}: L{l}'s branch capture is not the whole prompt at hidden width",
                a.name
            );
            scored += 1;
        }
        // `effective_prompt`'s input, and the whole parameter set it recovers against.
        float(&a.caps, "t0.embed_norm.out");
        assert_eq!(
            a.weights
                .floats
                .iter()
                .filter(|(n, _, _)| !is_alias(n))
                .count(),
            3 + layers * GLIMMER_LAYER_TENSORS.len(),
            "{}: the weight set is not the tiny model's whole parameter set",
            a.name
        );

        check_the_widths_drive_the_engine(a, t);
    }
    // **An absolute, not `scored > 0`.** Both loops above are derived from the goldens' own
    // metadata, so a re-vendor whose `decode_steps` dropped to 1 would leave every assertion
    // green over a third of the cells. 30 = 2 salts x (7 logit steps + 8 layer branches), and
    // it is the number the two device censuses below (14 and 16) are the halves of.
    assert_eq!(scored, 30, "the census covered {scored} captures, not 30");
}

/// **Every bound is above the envelope it was measured from, and not far above it.**
///
/// Both directions have cost this port a round and both are recorded. Below its envelope, a
/// bound reddens on a CORRECT engine — `old:` shipped `TOL` as "2.5e-3 and 2.6e-3", numbers
/// nobody had measured, and `TOL_BRANCH` as a guess of 2e-2; each went red on a working loop.
/// Far above it, a bound is a tolerance picked to make something pass, which this tree calls a
/// confession. The house convention is ~3x, and the four sit at 2.94x, 3.33x, 3.42x and 3.17x.
///
/// **This is the one thing about the tolerances a machine can check without a device**, which
/// is why it is here and not in a comment: the constants are otherwise read only by three
/// tests CI never compiles.
#[test]
fn every_bound_stands_over_the_envelope_it_was_measured_from() {
    for (what, bound, envelope) in ENVELOPES {
        let margin = bound / envelope;
        println!("{what} = {bound:e} is {margin:.2}x the measured {envelope:e}");
        assert!(
            (2.0..=5.0).contains(&margin),
            "{what} is {margin:.2}x its measured envelope {envelope:e}. Under ~2x it reddens \
             on a correct engine — the envelope is one realisation of a rounding process and a \
             re-drawn salt lands elsewhere in the same range. Over ~5x it has stopped being \
             derived from a measurement"
        );
    }
}

/// The geometry half of the census: these widths reach the engine's real paths.
///
/// Split out of the loop above because it answers a different question — that one asks
/// whether the FIXTURE is intact, this one whether the fixture is CAPABLE. Every claim here
/// is made by calling `glimmer::geometry`'s own function rather than restating its rule.
fn check_the_widths_drive_the_engine(a: &Anchor, t: &GlimmerTextConfig) {
    let n_ctx = a.positions();
    // The engine's own footprint guard, which refuses `n_ctx` past the trained positions and
    // a sliding layer with a zero window. If it refuses here, every device test below would
    // fail at construction with the same message and none of them would be scoring anything.
    geometry::check_footprint_inputs(t, n_ctx).unwrap_or_else(|e| {
        panic!(
            "{}: {n_ctx} positions is not a runnable context: {e:#}",
            a.name
        )
    });

    // **The ring must actually wrap.** `window_of` CLAMPS the window to `n_ctx` when the run
    // is shorter than the window, and a clamped sliding layer is a dense layer wearing the
    // name — it has nothing to slide past, so a gate over it tests the causal path twice.
    // This is the `--attn dsa` vacuity that cost the old tree an A/B, asked as a shape.
    let win = geometry::window_of(LayerKind::SlidingAttention, t.sliding_window, n_ctx, 0)
        .expect("a sliding window at the anchor's widths");
    assert_eq!(
        win.ring_cap, t.sliding_window,
        "{}: the {}-row window was clamped to {} by an {n_ctx}-position run, so nothing ever \
         evicts and the ring is never exercised",
        a.name, t.sliding_window, win.ring_cap
    );
    // The call itself is the check that the last position is representable; its slot is
    // `pos % cap` by construction, so nothing further about it can redden alone.
    let _ = geometry::window_of(
        LayerKind::SlidingAttention,
        t.sliding_window,
        n_ctx,
        n_ctx - 1,
    )
    .expect("the last position");
    // The wrap condition alone is the claim — a slot-bound conjunct would be true by
    // construction and make the message ambiguous (glimmer_config.rs's own rule;
    // review 2026-08-16).
    assert!(
        n_ctx > t.sliding_window,
        "{}: position {} does not wrap a {}-slot ring",
        a.name,
        n_ctx - 1,
        win.ring_cap
    );
    // Both layer kinds occur, or half the KV path is unvisited. A model of one kind would
    // still decode, and every score above would still be green.
    let sliding = t
        .layer_types
        .iter()
        .filter(|k| **k == LayerKind::SlidingAttention)
        .count();
    assert!(
        sliding > 0 && sliding < t.n_layers,
        "{}: {sliding} of {} layers slide — this anchor drives only one cache shape",
        a.name,
        t.n_layers
    );
    // The prefill runs in ONE chunk at these widths, which is what makes the branch test's
    // "the buffer holds the last prompt row" true. Asserted rather than assumed: at a longer
    // prompt the last chunk's last row is a different position.
    assert!(
        a.meta_usize("prompt_len") <= geometry::PREFILL_CHUNK,
        "{}: the prompt spans more than one prefill chunk",
        a.name
    );
}
