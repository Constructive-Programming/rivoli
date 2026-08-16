//! **The half of M9's exit gate that needs no GPU — and therefore the half CI runs.**
//!
//! `k3_anchor_decode.rs` is the device gate: it drives the engine's KDA launch composition
//! against the anchor's captured recurrence boundary and the whole decode loop against a
//! synthetic artifact. It is `#![cfg(feature = "rocm")]` end to end and CI has no rocm job,
//! so every claim in it is checked exactly as often as someone runs a GPU build by hand.
//!
//! What is checkable without a device is `glimmer_anchor_widths.rs`'s trio, adapted to what
//! K3's anchor can honestly support (`k3_anchor/mod.rs`'s header carries that argument):
//!
//! 1. **The capture names and shapes** the device gate reads — a renamed or re-vendored
//!    golden is the commonest way a golden gate silently stops scoring, and `float`'s
//!    absent-name panic here reddens in CI instead of on the next GPU run.
//! 2. **The schedules against the anchor.** `state::fold_at` is the engine's AttnRes
//!    machinery; the anchor's captured `block_residual` widths are the reference's. The two
//!    must agree at every captured layer, including the 8-deep stacks at 91/92 that the
//!    broken 12-periodicity produces.
//! 3. **The REAL config through the engine's real arithmetic** — the layer map's period
//!    break (91 and 92 adjacent MLA), the composed widths, the context door — plus the
//!    PINNED FACT that the anchor's own tiny widths are NOT engine-runnable, so nobody
//!    silently assumes the missing end-to-end comparison is one `write_artifact` away.
//!
//! # Red proofs, RUN — no GPU, 2026-08-16
//!
//! Both planted, observed red, reverted, observed green again in one session (P7).
//!
//! Rename `"in.q"` to `"in.qq"` in `k3_anchor::KDA_FIXTURE`:
//!
//! ```text
//! model.layers.0.kda.fused_recurrent_kda.in.qq is not in the golden; it holds 223 float
//! tensors, e.g. ["model.layers.0.input_layernorm", "model.layers.0.self_attn.q_proj", ...]
//! test result: FAILED. 3 passed; 1 failed
//! ```
//!
//! Change `state::fold_at`'s `stack` to `layer / res_block` (the off-by-one that counts a
//! boundary layer's own push into its entry stack — floor instead of ceil):
//!
//! ```text
//! k3-anchor-1: L1 entry fold on an empty stack
//! test result: FAILED. 3 passed; 1 failed
//! ```
//!
//! The first is the load-bearing one: it is the whole reason this binary exists, and it
//! fires in the featureless build CI actually runs.

// The panic-on-failure idiom; the fixture module carries the same allow and the reason.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod k3_anchor;

use k3_anchor::{Anchor, KDA_FIXTURE, KDA_LAYERS, anchors, float, kda_tag, kda_tol};
use rivoli_artifact::k3_config::K3Config;
use rivoli_artifact::schema::parse_config;
use rivoli_engine::k3::geometry::{ATTEND_MAX_KV, Dims, check_context};
use rivoli_engine::k3::state::{final_sources, fold_at};

/// The vendored real config — the same bytes `k3_anchor.rs` FNV-pins and `k3_config.rs`
/// tests against, read here so every claim about "the real model" is the checkpoint's.
const REAL_CONFIG: &str = include_str!("../../../docs/measurement/k3-reference/config.json");

/// The shape each KDA boundary tensor must carry, from the tiny config's own widths —
/// derived, never literal, because a literal agrees with drift. The state is SQUARE, which
/// is why the axis order is invisible to any shape check and the fixture-side transpose
/// (`to_key_major`) exists.
fn kda_shape(name: &str, nh: usize, hd: usize) -> Vec<usize> {
    match name {
        "in.beta" => vec![1, 1, nh],
        "in.A_log" => vec![nh],
        "in.dt_bias" => vec![nh * hd],
        "in.initial_state" | "out.state" => vec![1, nh, hd, hd],
        _ => vec![1, 1, nh, hd],
    }
}

/// A capture's shape if present — the non-panicking lookup the SCHEDULE tests need, because
/// a fold's ABSENCE at layer 0 is itself the claim under test.
fn shape_if(a: &Anchor, name: &str) -> Option<Vec<usize>> {
    a.caps
        .floats
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, s, _)| s.clone())
}

/// **Everything the device gate reads is in the goldens, at the shapes it assumes.**
///
/// The device test's KDA scoring reads exactly [`KDA_FIXTURE`] under [`kda_tag`]; each name
/// is asked for through `float`'s absent-name panic and its shape checked against the tiny
/// config's own widths — derived, never literal, because a literal agrees with drift.
#[test]
fn the_anchor_carries_what_the_device_gate_scores() {
    let mut scored = 0usize;
    for a in &anchors() {
        assert_eq!(a.caps.meta_get("mode"), Some("decode"), "{}", a.name);
        let (nh, hd) = (a.attn_field("num_heads"), a.attn_field("head_dim"));
        for l in KDA_LAYERS {
            let tag = kda_tag(l);
            for name in KDA_FIXTURE {
                let (shape, vals) = float(&a.caps, &format!("{tag}.{name}"));
                assert_eq!(shape, kda_shape(name, nh, hd), "{}: {tag}.{name}", a.name);
                assert!(
                    vals.iter().all(|v| v.is_finite()),
                    "{}: {tag}.{name} holds a non-finite value — every comparison the \
                     device gate makes against it would be meaningless",
                    a.name
                );
                scored += 1;
            }
        }
    }
    // An absolute, not `scored > 0`: both loops derive from constants, so a shrunk
    // KDA_LAYERS or FIXTURE list would quietly shrink the census with every assertion
    // green. 60 = 2 salts x 3 KDA layers x 10 boundary tensors.
    assert_eq!(scored, 60, "the census covered {scored} captures, not 60");
    // The tolerance lookup itself — a renamed `kda_op` row panics here, in CI, rather
    // than in the device gate nobody compiles.
    assert!(kda_tol() > 0.0);
}

/// **The engine's fold schedule against the anchor's captured stacks, at every captured
/// layer.** `state::fold_at` IS the machinery `forward.rs` drives; the `block_residual`
/// widths are the reference's own — so this is the composition question "does the arena
/// hold what the reference's stack held?" asked deviceless.
#[test]
fn the_fold_schedule_matches_the_anchors_stacks() {
    let mut cells = 0usize;
    // Pinned as the STRING the metadata declares, then iterated as the list — the string
    // equality is what binds the constant to the goldens' own declaration, without
    // re-spelling the parse `crates/oracles/tests/k3_anchor.rs` already owns.
    const CAPTURED: [usize; 6] = [0, 1, 3, 12, 91, 92];
    for a in &anchors() {
        let (layers, block) = (a.field("num_hidden_layers"), a.field("attn_res_block_size"));
        assert_eq!(
            a.caps.meta_get("capture_layers"),
            Some("0,1,3,12,91,92"),
            "{}: the captured-layer set moved; update CAPTURED beside this assertion",
            a.name
        );
        for l in CAPTURED {
            let f = fold_at(l, block);
            let entry = shape_if(
                a,
                &format!("model.layers.{l}.self_attention_res.in.block_residual"),
            );
            match f.entry_sources {
                // The reference GUARDS the entry fold on an empty stack, so layer 0 must
                // have NO capture — an entry capture there means the guard is gone.
                None => assert!(
                    entry.is_none(),
                    "{}: L{l} entry fold on an empty stack",
                    a.name
                ),
                Some(nsrc) => {
                    let shape = entry.unwrap_or_else(|| {
                        panic!("{}: L{l} entry fold missing from the capture", a.name)
                    });
                    assert_eq!(
                        shape[1],
                        nsrc - 1,
                        "{}: L{l} entry stack: the anchor captured {} block(s), the \
                         engine's schedule says {}",
                        a.name,
                        shape[1],
                        nsrc - 1
                    );
                }
            }
            let mlp = shape_if(a, &format!("model.layers.{l}.mlp_res.in.block_residual"))
                .unwrap_or_else(|| panic!("{}: L{l} mlp fold missing", a.name));
            assert_eq!(mlp[1], f.mlp_sources - 1, "{}: L{l} mlp stack", a.name);
            cells += 1;
        }
        // The model-level fold: every snapshot plus the final prefix — §7's silent-to-skip
        // aggregation, whose width ties schedule, anchor and config together.
        let out = shape_if(a, "model.output_attn_res.in.block_residual")
            .expect("the model-level fold's stack");
        assert_eq!(out[1], final_sources(layers, block) - 1, "{}", a.name);
        cells += 1;
    }
    assert_eq!(
        cells, 14,
        "2 salts x (6 captured layers + the model-level fold)"
    );
}

/// **The real checkpoint's config through the engine's real deviceless paths** — parsed by
/// the REAL parser (`validate` included), widths composed by `Dims::from_config`, the layer
/// map read through `layer_is_mla`. The map's period BREAK is the load-bearing cell: 91 and
/// 92 are adjacent MLA, so any modulo reconstruction reddens here.
#[test]
fn the_real_config_drives_the_engines_arithmetic() {
    let cfg = parse_config::<K3Config>(REAL_CONFIG).expect("the real config parses");
    let t = &cfg.text;
    let d = Dims::from_config(t).expect("the real widths compose");
    assert_eq!(d.res_blocks, 8);
    assert_eq!(final_sources(t.n_layers, t.attn_res_block_size), 9);
    // The anchor's KDA capture layers really are KDA, its MLA layers really MLA — and the
    // pattern's tail breaks: 87 MLA, 88..=90 KDA, then 91 AND 92 MLA.
    for l in KDA_LAYERS {
        assert!(!t.layer_is_mla(l).unwrap(), "layer {l} must be KDA");
    }
    for l in [3, 87, 91, 92] {
        assert!(t.layer_is_mla(l).unwrap(), "layer {l} must be MLA");
    }
    for l in [88, 89, 90] {
        assert!(
            !t.layer_is_mla(l).unwrap(),
            "layer {l} breaks the every-4th tail"
        );
    }
    let mla = (0..t.n_layers)
        .filter(|&l| t.layer_is_mla(l).unwrap())
        .count();
    assert_eq!((mla, t.n_layers - mla), (24, 69), "the 24/69 partition");
    // The context door at the real ceiling — the same function the seam calls.
    check_context(ATTEND_MAX_KV).expect("the ceiling itself is admitted");
    assert!(check_context(ATTEND_MAX_KV + 1).is_err());
}

/// **The anchor's tiny widths are NOT engine-runnable, pinned as a refusal.** The tiny
/// `moe_intermediate_size` 24 breaks the `.f4` 32-wide group rule and the real parser says
/// so — which is half of why the device gate scores the KDA boundary and a SYNTHETIC
/// artifact instead of decoding the anchor's model (the other half: no weights are
/// vendored; `k3_anchor/mod.rs`'s header carries both). If this ever turns green, an
/// end-to-end anchor decode became possible and the gate should grow one.
#[test]
fn the_anchor_widths_are_not_engine_runnable() {
    let real: serde_json::Value = serde_json::from_str(REAL_CONFIG).unwrap();
    for a in &anchors() {
        // The refusal ladder is DEEP — null `architectures`/`dtype`, a junk HF
        // `rope_theta` on a NoPE model, and then the width itself — and pinning any one
        // rung makes the test brittle about refusal ORDER, which is not the claim. The
        // claim is two facts: the real parser refuses this config at all, and the one
        // rung no dressing can fix is arithmetic — `moe_intermediate_size` is not a
        // whole number of `.f4`'s 32-wide scale groups, so no expert file for this model
        // can exist.
        let mut doc = real.clone();
        doc["text_config"] = a.tiny.clone();
        assert!(
            parse_config::<K3Config>(&doc.to_string()).is_err(),
            "{}: the tiny config PARSED — see this test's doc",
            a.name
        );
        let inter = a.field("moe_intermediate_size");
        assert!(
            !inter.is_multiple_of(32),
            "{}: moe_inter {inter} became F4-group-legal — an end-to-end anchor decode \
             may now be one weight dump away; revisit the gate's division of labour",
            a.name
        );
    }
}
