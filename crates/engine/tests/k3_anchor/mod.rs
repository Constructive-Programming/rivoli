//! **The vendored Kimi-K3 anchor, and what the engine's exit gate may honestly score
//! against it.**
//!
//! Shared by `k3_anchor_widths.rs` (deviceless: is the fixture intact, does the REAL config
//! drive the engine's arithmetic, do the schedules agree?) and `k3_anchor_decode.rs` (the
//! device gate). One module so the two halves cannot drift on which captures are scored
//! under which bound — `glimmer_anchor/mod.rs`'s argument, inherited whole.
//!
//! # What this anchor CANNOT support, stated once so neither half overclaims
//!
//! Unlike Muse Glimmer's anchor, K3's S1b vendored **no weight tensors** — the tiny model's
//! parameters are `torch.Generator` draws keyed by `sha256(salt/name)`
//! (`crates/oracles/tests/k3_anchor_driver.py::init_weights`), unreproducible outside torch
//! — and its tiny widths are **not engine-runnable**: `moe_intermediate_size` 24 breaks the
//! `.f4` container's 32-wide group rule, which `K3TextConfig::validate` refuses
//! ([`the_anchor_widths_are_not_engine_runnable`] in the widths gate pins the refusal). So
//! an end-to-end engine-vs-reference decode comparison is impossible on two independent
//! counts, and the division of labour follows:
//!
//! * the **KDA recurrence boundary** is the one operator whose ENTIRE decode-step interface
//!   the anchor captures (q, k, v, g, beta, `A_log`, `dt_bias`, state in → o, state out).
//!   The device gate scores the engine's own launch composition against it, under the
//!   tolerance table's `kda_op` row — the one anchor-derived number with provenance.
//! * everything else end-to-end is scored **structurally** on a synthetic, F4-legal tiny
//!   artifact: bit-identity across residency budgets (P4), and carried-state-vs-replayed-
//!   prefix bit-identity, which is the property that makes every per-operator kernel gate
//!   COMPOSE into a correct decode.
//! * the per-operator numerics stay with `kernel_k3_*.rs` against their own fixtures.

// Each of the two binaries uses a subset — the same argument `glimmer_anchor/mod.rs` makes.
#![allow(dead_code)]
// Panic-on-failure is the fixture idiom: a golden that cannot read its own bytes dies
// naming the file.
#![allow(clippy::unwrap_used, clippy::expect_used)]

// The cross-crate `#[path]` pattern `glimmer_anchor/mod.rs` argues for: `float`/`ints` and
// their absent-name panics have ONE owner in `crates/oracles/tests/common/`.
#[path = "../../../oracles/tests/common/golden_read.rs"]
pub mod golden_read;
#[path = "../../../oracles/tests/common/tolerance.rs"]
pub mod tolerance;

pub use golden_read::{GoldenSet, float};
use serde_json::Value;
use tolerance::{Policy, Tol};

/// The two vendored decode goldens, by the same bytes `crates/oracles/tests/k3_anchor.rs`
/// pins (provenance is not re-checked here — a second frozen copy agreeing with the first
/// is not a check; what this module asserts about the bytes is what the gates need FROM
/// them).
pub const ANCHORS: [(&str, &[u8]); 2] = [
    (
        "k3-anchor-1",
        include_bytes!("../../../oracles/tests/k3-anchor-decode-k3-anchor-1.bin"),
    ),
    (
        "k3-anchor-2",
        include_bytes!("../../../oracles/tests/k3-anchor-decode-k3-anchor-2.bin"),
    ),
];

/// The KDA layers the anchor captured. 0, 1 and 12 — NOT 3, 91 or 92, which are MLA; the
/// widths gate re-derives this split from the REAL config's own layer lists rather than
/// trusting this constant alone.
pub const KDA_LAYERS: [usize; 3] = [0, 1, 12];

/// One salt: the captures and the tiny config they were produced under.
pub struct Anchor {
    pub name: &'static str,
    pub caps: GoldenSet,
    /// The golden's own `tiny_config`, parsed as raw JSON — NOT through `K3Config`, whose
    /// `validate` rightly refuses these widths (see the module header). Every width below
    /// is read from here, never written as a literal.
    pub tiny: Value,
}

impl Anchor {
    /// A `linear_attn_config` integer.
    pub fn attn_field(&self, key: &str) -> usize {
        self.tiny["linear_attn_config"][key]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: linear_attn_config.{key}", self.name)) as usize
    }

    /// A top-level `tiny_config` integer.
    pub fn field(&self, key: &str) -> usize {
        self.tiny[key]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: tiny_config.{key}", self.name)) as usize
    }

    /// The KDA gate's lower bound, through the reader's own accessor — see
    /// `GoldenSet::k3_gate_lower_bound` for the inclusive-bound argument (hoisted there when
    /// two blind-authored harnesses each spelled the JSON path, 2026-08-16).
    pub fn lower_bound(&self) -> f32 {
        GoldenSet::k3_gate_lower_bound(&self.tiny)
    }
}

pub fn anchors() -> Vec<Anchor> {
    let mut out = Vec::with_capacity(ANCHORS.len());
    for (name, bytes) in ANCHORS {
        let caps = GoldenSet::read_k3(&mut &bytes[..])
            .unwrap_or_else(|e| panic!("{name}: the vendored golden must load: {e:#}"));
        let tiny = serde_json::from_str(caps.meta_get("tiny_config").expect("tiny_config"))
            .expect("tiny_config is JSON");
        out.push(Anchor { name, caps, tiny });
    }
    out
}

/// The fla-side KDA fixture names under one layer, spelled once: the deviceless census
/// asserts their presence and the device gate reads them, so a rename cannot leave the
/// census certifying names nothing reads.
pub fn kda_tag(layer: usize) -> String {
    format!("model.layers.{layer}.kda.fused_recurrent_kda")
}

/// The ten tensors of one KDA fixture — the operator's whole decode-step boundary.
pub const KDA_FIXTURE: [&str; 10] = [
    "in.q",
    "in.k",
    "in.v",
    "in.g",
    "in.beta",
    "in.A_log",
    "in.dt_bias",
    "in.initial_state",
    "out.o",
    "out.state",
];

/// fla's recurrent state is `[value][key]`; rivoli's is `[key][value]`
/// (`launch_gated_delta_recurrent_f32`'s documented axis order, chosen for coalescing).
/// The transpose lives in the FIXTURE, not the engine — `square tensors hide their axis
/// order`, and the anchor's own `KdaStateLayout` defect run is what prices getting this
/// backwards. `[heads][d][d]`, per head.
pub fn to_key_major(v: &[f32], heads: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; v.len()];
    for h in 0..heads {
        let base = h * d * d;
        for a in 0..d {
            for b in 0..d {
                out[base + b * d + a] = v[base + a * d + b];
            }
        }
    }
    out
}

/// The device gate's ONE anchor-derived tolerance: the table's `kda_op` row, looked up
/// rather than transcribed, so the constant cannot drift from the measurement that owns it
/// (`tolerances_leave_room` in `k3_anchor.rs` is the gate on the row itself).
pub fn kda_tol() -> f32 {
    let row: &Tol = tolerance::K3
        .iter()
        .find(|t| t.operator == "kda_op")
        .expect("the tolerance table carries kda_op");
    match row.policy {
        Policy::Rel(t) => t,
        Policy::ExactOnly => panic!("kda_op became ExactOnly; the device gate must follow"),
    }
}
