//! The S3 Inkling 2 anchor gate: the vendored goldens are the first-party
//! `transformers.models.inkling2_loop` text model's own forward (transformers @ d8f10a82, CPU
//! fp32), not a transliteration. `inkling-anchor.sh` regenerates them; this reads what it vendors.
//!
//! The defect matrix and the tolerance floors are scored by the driver itself (the matrix
//! reddens against the None golden, the floors are carved fp32-vs-fp64); this gate holds the
//! deviceless claims over the vendored bytes: provenance, two independent draws, finite spread
//! logits, the capture census, every loop pass being captured and differing, and the shapes — a
//! decode golden is ONE stepped token (`[1, 1, …]`) against the prefill's cache, not a re-run
//! forward of the same length. No GPU, no torch.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

#[path = "common/golden_read.rs"]
mod golden_read;

use golden_read::{
    GoldenSet, Vendored, capture_census, draws_are_independent, float,
    logits_are_finite_and_spread, provenance,
};
use rivoli_core::legality::Arch;

const DECODE: &[Vendored] = &[
    Vendored {
        name: "inkling-1",
        bytes: include_bytes!("inkling-anchor-decode-inkling-1.bin"),
        len: 18_002,
        fnv: 0xc06e_4a7e_f0b2_f164,
    },
    Vendored {
        name: "inkling-2",
        bytes: include_bytes!("inkling-anchor-decode-inkling-2.bin"),
        len: 18_002,
        fnv: 0xb05a_31b0_f65a_c4be,
    },
];

const PREFILL: Vendored = Vendored {
    name: "inkling-prefill",
    bytes: include_bytes!("inkling-anchor-prefill-inkling-1.bin"),
    len: 263_406,
    fnv: 0xabcb_ffe6_0fc8_80b3,
};

const CAPTURE_LAYERS: &[usize] = &[0, 5, 6, 11, 17, 23];

/// The capture layers inside the loop range `[6, 18)`: each is applied once per loop pass, so
/// the golden carries `pass1..pass3` for every one of them.
const LOOP_PASS_LAYERS: &[usize] = &[6, 11, 17];

/// Float tensors in one golden: six `model.layers.{L}.hidden`, nine loop-pass tensors (3 layers
/// × 3 passes) and `logits`.
const TENSORS: usize = 6 + 3 * 3 + 1;

fn load(v: &Vendored) -> GoldenSet {
    GoldenSet::read_anchor_for(Arch::Inkling2Loop, &mut &v.bytes[..])
        .unwrap_or_else(|e| panic!("the vendored {} golden must load: {e:#}", v.name))
}

fn all() -> Vec<&'static Vendored> {
    DECODE.iter().chain([&PREFILL]).collect()
}

#[test]
fn the_vendored_goldens_match_their_byte_pins() {
    for v in all() {
        v.check_bytes();
    }
}

#[test]
fn the_provenance_is_the_first_party_text_model() {
    for v in all() {
        let g = load(v);
        let salt = if v.name == "inkling-2" {
            "inkling-2"
        } else {
            "inkling-1"
        };
        provenance(
            &g,
            salt,
            &[
                ("model", "inkling2_loop"),
                ("entry_point", "Inkling2LoopForCausalLM"),
                ("defect", "None"),
                ("dtype", "float32"),
            ],
        );
    }
}

#[test]
fn the_two_decode_draws_are_independent() {
    // `TENSORS`, not the old 7: the loop-pass captures are float tensors the two draws share,
    // and a shrinking intersection is what this assertion exists to catch.
    draws_are_independent(&load(&DECODE[0]), &load(&DECODE[1]), TENSORS);
}

#[test]
fn the_logits_are_finite_and_spread() {
    for v in all() {
        logits_are_finite_and_spread(&load(v), "logits");
    }
}

/// The census reads a tensor's layer index before the first field dot, so a loop layer's
/// `pass{p}.hidden` counts as that layer — the pass captures widen the buckets without claiming
/// a layer the driver did not hook.
#[test]
fn the_capture_census_is_the_six_declared_layers() {
    for v in all() {
        capture_census(&load(v), CAPTURE_LAYERS, v.name);
    }
}

#[test]
fn the_shapes_are_the_tiny_configs() {
    // The decode golden is ONE stepped token against the prefill's cache, so every bucket is
    // `[1, 1, …]` — the shape a cached-KV readback has and a re-run prefill does not. The
    // hidden stream is the tiny hidden_size 256 and the head is unpadded to 250 rows.
    for v in DECODE {
        let d = load(v);
        assert_eq!(
            float(&d, "logits").0,
            &[1, 1, 250],
            "{}: stepped logits",
            v.name
        );
        assert_eq!(
            float(&d, "model.layers.0.hidden").0,
            &[1, 1, 256],
            "{}: stepped hidden",
            v.name
        );
    }
    let p = load(&PREFILL);
    assert_eq!(float(&p, "logits").0, &[1, 16, 250]);
    assert_eq!(float(&p, "model.layers.23.hidden").0, &[1, 16, 256]);
}

/// One loop capture's tensor, by layer and pass. A named fn rather than a closure: the slice is
/// borrowed from the golden, and that lifetime is the one thing this file cannot afford to have
/// inferred differently.
fn pass_of(g: &GoldenSet, idx: usize, p: usize) -> &[f32] {
    float(g, &format!("model.layers.{idx}.pass{p}.hidden")).1
}

#[test]
fn the_loop_layers_capture_every_pass_and_the_passes_differ() {
    for v in all() {
        let g = load(v);
        for &idx in LOOP_PASS_LAYERS {
            // `hidden` is the LAST pass's output — the value the next layer consumed — not the
            // first one wearing a different name.
            assert_eq!(
                float(&g, &format!("model.layers.{idx}.hidden")).1,
                pass_of(&g, idx, 3),
                "{}: layer {idx}'s `hidden` must be the third pass",
                v.name
            );
            for p in 1..3 {
                let (a, b) = (pass_of(&g, idx, p), pass_of(&g, idx, p + 1));
                assert_eq!(
                    a.len(),
                    b.len(),
                    "{}: layer {idx} pass {p}/{} widths",
                    v.name,
                    p + 1
                );
                // Pass 2+ restart from the pre-loop states with a message mixed in (MOD:542), so
                // two passes that agreed bit-for-bit would mean the recurrence stopped happening
                // while every shape assertion still passed.
                assert!(
                    !a.iter()
                        .map(|x| x.to_bits())
                        .eq(b.iter().map(|x| x.to_bits())),
                    "{}: layer {idx} passes {p} and {} are bit-identical — the loop is not \
                     re-entering the layer with a different state",
                    v.name,
                    p + 1
                );
            }
        }
    }
}
