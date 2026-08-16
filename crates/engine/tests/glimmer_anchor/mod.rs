//! **The vendored Muse Glimmer anchor, and the tolerances two gates score it under.**
//!
//! Shared by `glimmer_anchor_widths.rs` (deviceless: is the fixture intact and capable?) and
//! `glimmer_anchor_decode.rs` (the device gate: does the engine reproduce the reference?).
//! One module rather than two copies, because the two halves must agree on which captures are
//! scored and under which bound — and a copy that drifted would leave the deviceless half
//! certifying names the device half no longer reads.
//!
//! Read `glimmer_anchor_decode.rs`'s header first: it carries the argument for the whole gate,
//! the bf16 term the tolerances below come from, and the red-proof recipe.
//!
//! No device, no python, no network: everything here reads bytes and parses a config.

// Each of the two binaries uses a SUBSET — the deviceless one never encodes a tensor, the
// device one never reads the envelope table. The same argument `common/mod.rs` makes for the
// kernel scaffolding, and the alternative is a `#[cfg]` per item keyed on which consumer
// happens to exist today.
#![allow(dead_code)]
// The panic-on-failure idiom: a fixture that cannot read its own bytes must die loudly and
// name the file, not thread a `Result` out to an assertion.
#![allow(clippy::unwrap_used, clippy::expect_used)]

// **A cross-crate `#[path]`, and it is the alternative to a jscpd violation.** `float`, `ints`
// and their absent-name panics live in `crates/oracles/tests/common/golden_read.rs`, a
// test-local module no crate can `use`. Re-spelling `float` here would be a ~35-token clone of
// that file at `.jscpd.json`'s `minTokens: 15`, and the copy that drifted would still pass.
// That file's own header argues it is `#[path]`-included rather than routed through a
// `common/mod.rs`; this extends the same reasoning by one directory level. It compiles because
// its only outside names are `rivoli_oracles::golden` and `rivoli_core::hash`, and
// `rivoli-engine` depends on both.
#[path = "../../../oracles/tests/common/golden_read.rs"]
pub mod golden_read;

pub use golden_read::{GoldenSet, float, ints};
use rivoli_artifact::glimmer_config::GlimmerConfig;
use rivoli_artifact::schema::parse_config;
use serde_json::Value;

/// The two vendored text goldens and the two weight sets, by the same bytes
/// `crates/oracles/tests/glimmer_anchor.rs` pins.
///
/// **Provenance is NOT re-checked here**, and that is `old:tests/common/glimmer_fixture.rs`'s
/// rule rather than an omission: that file gates the length and the FNV of all six, and a
/// second frozen copy of those numbers agreeing with the first is not a check. What this file
/// asserts about the bytes is what it needs FROM them — see
/// [`the_anchor_carries_what_this_gate_scores`].
///
/// **Both salts, everywhere.** One draw cannot show that an agreement is a fact about the
/// arithmetic rather than about the numbers it landed on.
pub const ANCHORS: [(&str, &[u8], &[u8]); 2] = [
    (
        "text-1",
        include_bytes!("../../../oracles/tests/glimmer-anchor-text-1.bin"),
        include_bytes!("../../../oracles/tests/glimmer-anchor-weights-1.bin"),
    ),
    (
        "text-2",
        include_bytes!("../../../oracles/tests/glimmer-anchor-text-2.bin"),
        include_bytes!("../../../oracles/tests/glimmer-anchor-weights-2.bin"),
    ),
];

/// The shipped `config.json`, at HF revision `f84ecc3` — the same file
/// `old:tests/common/mod.rs::GLIMMER_SHIPPED_CONFIG` splices, byte-identical in this tree.
///
/// The WRAPPER is the real one and only `text_config` is replaced, because the schema's
/// descent asserts it landed in the text dict rather than in `vision_config`, a sibling that
/// parses several of the same field names. A hand-built wrapper would be testing a different
/// descent than the one that ships.
const SHIPPED_CONFIG: &str =
    include_str!("../../../../docs/measurement/glimmer-reference/config.json");

/// The per-layer capture this gate scores the engine's branch buffer against.
///
/// Named once because the deviceless census asserts its presence and the device test reads
/// it; two spellings is one rename away from a census that certifies a name nothing reads.
pub const BRANCH_CAP: &str = "post_feedforward_layernorm.out";

// ---------------------------------------------------------------------------------------
// Tolerances. Each is `old:`-MEASURED, in `worst_rel`'s metric (`max|got-want| / max|want|`),
// and carries where the measurement is recorded. None was chosen after seeing this engine's
// output.
// ---------------------------------------------------------------------------------------

/// Relative tolerance on the logits — the **bf16 weight-rounding envelope**, measured, not an
/// arithmetic floor.
///
/// > The experiment (`old:tests/glimmer_reference.rs:74-97`, recorded in
/// > `old:docs/investigations/glimmer-open-items.md` §4.1): run the whole chain in f64 twice
/// > on the reference's own parameters, once exact and once with every weight rounded to
/// > bf16, and score each against `tN.logits`.
/// >
/// > | | worst over 7 steps x 2 salts |
/// > |---|---|
/// > | f64, exact weights | **3.8e-6** — the transcription reproduces the reference |
/// > | f64, bf16 weights | **6.8e-2** (range 9.3e-3 – 6.8e-2) — the format's own error |
/// >
/// > The first row is the load-bearing one: it says an independent f64 reading of the
/// > architecture reproduces Muse Glimmer to 3.8e-6, so the reference this gate scores
/// > against is being read correctly.
///
/// `2e-1` is ~3x the worst observed envelope. That margin is not generosity — the envelope is
/// one realisation of a rounding process and a re-drawn salt lands elsewhere in the same
/// range. `old:`'s green worst on a correct engine was **4.8e-2**.
///
/// **A round number would be a confession**; this one is a bound on a measured range, and the
/// range is what the assertion message quotes.
pub const TOL: f32 = 2e-1;

/// The same envelope one layer up, on the post-FFN branch — measured per layer exactly as
/// [`TOL`] was, and for the same reason: `old:`'s first value here was a guess (2e-2,
/// "tighter, because it is one layer's error") and layer 3 went red on a correct engine.
///
/// The bf16 floor on this tensor GROWS with depth, because the branch is computed from a
/// residual stream that has already accumulated it: **4.7e-3 at L0 to 3.0e-2 at L6**, over
/// both salts (`old:tests/glimmer_reference.rs:100-107`). `1e-1` is ~3.3x the worst.
pub const TOL_BRANCH: f32 = 1e-1;

/// Total variation between the engine's output distribution and the reference's — measured
/// at **1.249e-3 (text-1) and 1.464e-3 (text-2)**, set at ~3.4x the worse.
///
/// **A second NORM on the same logits, and narrower than it looks.** [`TOL`] is a
/// relative-max over the vector; this is an L1 over the distribution, so a defect
/// concentrated on one high-probability token moves it far more. It is NOT independent
/// evidence and it is NOT a softcap gate — the softcap moves it from 1.249e-3 to 1.249e-3.
/// Red-proved in `old:` by a x10 scale on the metric itself: 1.464e-3 → 6.468e-3, past this
/// bound. That proof is recorded because a bound whose only documented experiment cannot
/// redden it is a bound with no evidence it is a gate at all.
pub const TOL_TV: f32 = 5e-3;

/// `|NLL_got − NLL_want|` for the reference's own emitted token — measured at **5.316e-3 and
/// 6.305e-3**, set at ~3.2x.
///
/// **Its resolution is a ~10% temperature error, measured**: scaling only the engine's logits,
/// x1.05 scores 1.2e-2 and passes, x1.10 scores 2.231e-2 and reddens. The anchor's logits span
/// |0.24| over a vocabulary of 61, so the distribution is nearly uniform and one token's NLL
/// barely moves — a coarse instrument HERE, on this fixture. Tightening it to catch 5% would
/// leave 1.6x over the clean value, which is the exact thinness that made `old:`'s first
/// whole-chain gates go red on a correct engine.
///
/// A common-mode mutation cannot redden it: scaling BOTH sides cancels in the difference.
/// That is a property of a differential metric, not a hole, and it is why the red proof above
/// perturbs one side only.
pub const TOL_NLL: f32 = 2e-2;

// ---------------------------------------------------------------------------------------
// The anchor, and the model it describes.
// ---------------------------------------------------------------------------------------

/// One salt: the captures, the reference's own parameters, and the tiny config both were
/// produced under.
pub struct Anchor {
    pub name: &'static str,
    pub caps: GoldenSet,
    pub weights: GoldenSet,
    /// The golden's own `tiny_config`, parsed. **Every width below is read from here**, never
    /// written as a literal — a literal agrees with drift and a derived value fails on it.
    pub tiny: Value,
}

impl Anchor {
    pub fn meta_usize(&self, key: &str) -> usize {
        self.caps
            .meta_get(key)
            .unwrap_or_else(|| panic!("{}: no {key} in the golden's metadata", self.name))
            .parse()
            .expect("a numeric metadata value")
    }

    /// How many logit vectors the golden carries: one prefill pick plus every decode step.
    pub fn steps(&self) -> usize {
        self.meta_usize("decode_steps") + 1
    }

    /// The positions a run over this golden occupies — prompt plus every emitted token. It is
    /// what `n_ctx` must cover and what the sliding window has to be crossable within.
    pub fn positions(&self) -> usize {
        self.meta_usize("prompt_len") + self.steps()
    }

    /// The shipped `config.json` with this golden's `tiny_config` spliced into `text_config`.
    ///
    /// **The WRAPPER is the real one and only `text_config` is replaced**, because the schema's
    /// descent asserts it landed in the text dict rather than in `vision_config`, a sibling
    /// that parses several of the same field names. A hand-built wrapper would exercise a
    /// different descent than the one that ships.
    ///
    /// **One owner for the splice.** [`Self::config`] parses this and the device gate's fixture
    /// WRITES it, so the config the schema validated and the config that lands on disk cannot
    /// become two documents — which is the class of drift where a gate certifies one model and
    /// the engine runs another.
    pub fn config_json(&self) -> Value {
        let mut doc: Value = serde_json::from_str(SHIPPED_CONFIG).expect("the shipped config");
        doc["text_config"] = self.tiny.clone();
        doc
    }

    /// [`Self::config_json`], parsed and VALIDATED by the schema the engine reads.
    ///
    /// Parsing here rather than trusting the splice is the point: `validate` enforces the
    /// pairing invariant (a layer is rotated IFF it slides), the GQA divisibility and the four
    /// argmax-invariant scalars, so a `tiny_config` that stopped describing a model this port
    /// can run fails as a config error and not as a wrong number seven layers downstream.
    pub fn config(&self) -> GlimmerConfig {
        parse_config(&self.config_json().to_string()).unwrap_or_else(|e| {
            panic!(
                "{}: the spliced config is not a Glimmer one: {e:#}",
                self.name
            )
        })
    }
}

pub fn anchors() -> Vec<Anchor> {
    ANCHORS
        .iter()
        .map(|(name, caps, weights)| {
            let read = |b: &[u8], what: &str| {
                GoldenSet::read_glimmer(&mut &b[..])
                    .unwrap_or_else(|e| panic!("{name} {what}: {e:#}"))
            };
            let caps = read(caps, "captures");
            let raw = caps.meta_get("tiny_config").expect("tiny_config");
            Anchor {
                name,
                tiny: serde_json::from_str(raw).expect("tiny_config is JSON"),
                weights: read(weights, "weights"),
                caps,
            }
        })
        .collect()
}

/// Each bound above, beside the measured worst it was set from, so the margin convention is a
/// check rather than a habit — see [`every_bound_stands_over_the_envelope_it_was_measured_from`].
///
/// The envelopes are `old:`'s, restated here as DATA rather than only in prose, because that is
/// the only form a test can read. Their derivations are in the constants' own doc blocks.
pub const ENVELOPES: [(&str, f32, f32); 4] = [
    ("TOL", TOL, 6.8e-2),
    ("TOL_BRANCH", TOL_BRANCH, 3.0e-2),
    ("TOL_TV", TOL_TV, 1.464e-3),
    ("TOL_NLL", TOL_NLL, 6.305e-3),
];

/// One tensor as the artifact stores it: the dtype and the bytes, decided together **from
/// its RANK and nothing else**.
///
/// A 1-D layer tensor is a norm and the artifact holds norms at f32; everything else is a
/// projection and stays bf16. That is the same discriminator `glimmer::geometry::layer_bytes`
/// charges the budget by (`if shape.len() == 1 { 4 } else { 2 }`) and the same one
/// `pin::place_norm`/`place_proj` demand at load, so this fixture cannot disagree with the
/// engine about which is which. It is deliberately NOT a copy of `convert_glimmer::is_norm`'s
/// name-suffix test: two spellings of one rule is how the two drift, and the rank is the one
/// the engine actually enforces.
///
/// The dtype and the encoding are returned together because choosing one without the other is
/// a header that describes bytes it does not have — in bounds, correctly sized, and read as
/// the wrong numbers.
///
/// Its only caller is the fixture writer in `glimmer_anchor_decode.rs`, so it is dead in the
/// deviceless binary — covered by this module's file-level `dead_code` allow rather than by a
/// `#[cfg(feature = "rocm")]`, because gating it would make the rule this function states
/// invisible to the featureless build that is the only one CI compiles.
pub fn stored(shape: &[usize], vals: &[f32]) -> (rivoli_artifact::format::Dtype, Vec<u8>) {
    use rivoli_artifact::format::Dtype;
    if shape.len() == 1 {
        (
            Dtype::F32,
            vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    } else {
        let bf16 = |v: &f32| rivoli_core::num::f32_to_bf16(*v).to_le_bytes();
        (Dtype::Bf16, vals.iter().flat_map(bf16).collect())
    }
}

/// The reference's parameters as the bytes an artifact holds.
///
/// Skips the `L{l}.attn.gate_proj.weight` alias the driver still emits beside
/// `self_attn.gate_proj` — a checkpoint carrying both would fail the engine's own layer walk
/// with a name it has never heard of, which is that walk doing its job.
pub fn is_alias(name: &str) -> bool {
    name.starts_with('L')
}
