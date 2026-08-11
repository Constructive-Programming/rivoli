//! `GlimmerPin` over a converted artifact — the resident set, on the device.
//!
//! **A GPU arm.** `DeviceTier::new` allocates, so this binary needs the device, the flock and
//! `--test-threads=1` like every other device suite. It is a separate file from
//! `glimmer_convert` for exactly that reason: those two tests are deviceless and must stay
//! runnable on a machine with no GPU and in CI, which has no rocm job at all.
//!
//! The fixture is `glimmer_convert`'s, shared through `tests/common`, and that sharing is the
//! design rather than a convenience — this test asserts about the artifact the converter
//! produces, so a pin test on its own differently-built checkpoint would establish nothing
//! about the pipeline.
//!
//! **What this can check that no unit test can: the bytes arrived.** The tier is a
//! host-fillable VMM allocation (`DeviceTier::place` fills it with an ordinary host memcpy),
//! so every pointer the pin hands out is readable from here. Dims alone would pass a pin that
//! recorded the right shape for the wrong tensor.

#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod common;
use common::{FixtureTensor, GLIMMER_FIXTURE_LAYERS as L, glimmer_convert_fixture};
use rivoli::artifact::model as gm;
use rivoli::memory::pin::GlimmerPin;

const DIM: usize = 8;

/// Convert the synthetic checkpoint and return `(artifact dir, config, source tensors)`.
/// The caller owns the temp root and removes it.
fn convert(tag: &str) -> (std::path::PathBuf, gm::GlimmerConfig, Vec<FixtureTensor>) {
    let root = std::env::temp_dir().join(format!("glimmer-pin-{tag}-{}", std::process::id()));
    let (tensors, _) = glimmer_convert_fixture(&root, DIM);
    let cfg = gm::load_config(root.join("out").to_str().unwrap()).unwrap();
    (root, cfg, tensors)
}

/// Read a placement back out of the tier. Safe because the slab is host-fillable — this is
/// the same mapping `DeviceTier::place` memcpy'd into.
fn slab(ptr: *const u8, bytes: usize) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(ptr, bytes) }.to_vec()
}

#[test]
fn glimmer_pin_places_every_tensor_with_the_shape_the_config_implies() {
    // The pin names all twelve per-layer tensors individually, as struct fields — the one
    // thing it cannot derive from the constant. A thirteenth entry therefore needs a field,
    // and this is what says so before it is silently never placed.
    assert_eq!(
        gm::GLIMMER_LAYER_TENSORS.len(),
        12,
        "GlimmerLayerPin has a field per entry; a new entry needs one too"
    );

    let (root, cfg, tensors) = convert("ok");
    let src: std::collections::HashMap<&str, &Vec<u8>> =
        tensors.iter().map(|(n, _, b)| (n.as_str(), b)).collect();
    let pin = GlimmerPin::build(root.join("out").to_str().unwrap(), &cfg.text).unwrap();

    assert_eq!(pin.layers.len(), L);
    // Globals. Both `[vocab, hidden]` and both present: `tie_word_embeddings` is false, so a
    // pin that aliased one to the other would be wrong about 2.690 GB on the real model.
    for (w, name) in [
        (pin.embed, "model.language_model.embed_tokens.weight"),
        (pin.head, "lm_head.weight"),
    ] {
        assert_eq!(
            [w.o_dim, w.i_dim],
            [cfg.text.vocab, cfg.text.hidden],
            "{name}"
        );
        assert_eq!(
            slab(w.packed as *const u8, w.o_dim * w.i_dim * 2),
            **src.get(name).unwrap(),
            "{name} did not arrive in the tier"
        );
    }
    assert_ne!(
        pin.embed.packed, pin.head.packed,
        "embed and head are untied"
    );

    for (l, layer) in pin.layers.iter().enumerate() {
        // Paired with the tensor name so the assertion below reads the shape table rather
        // than restating eight shapes — and so a field wired to the wrong name is visible as
        // a shape mismatch on the pairs the shapes DO separate.
        let mats = [
            (layer.q, "self_attn.q_proj"),
            (layer.k, "self_attn.k_proj"),
            (layer.v, "self_attn.v_proj"),
            (layer.o, "self_attn.o_proj"),
            (layer.attn_gate, "self_attn.gate_proj"),
            (layer.mlp_gate, "mlp.gate_proj"),
            (layer.mlp_up, "mlp.up_proj"),
            (layer.mlp_down, "mlp.down_proj"),
        ];
        for (w, t) in mats {
            let full = format!("{}.{l}.{t}.weight", gm::GLIMMER_LAYER_PREFIX);
            assert_eq!(
                [w.o_dim, w.i_dim],
                cfg.text.layer_tensor_shape(t).unwrap()[..],
                "{full} shape"
            );
            // The bytes, per tensor — which is what separates `q` from `attn_gate` and `k`
            // from `v`, since within each of those pairs the shapes are identical. The
            // fixture gives every tensor a distinct blob precisely so this can tell them
            // apart.
            assert_eq!(
                slab(w.packed as *const u8, w.o_dim * w.i_dim * 2),
                **src.get(full.as_str()).unwrap(),
                "{full} is wired to the wrong tensor"
            );
        }

        // The four norms, f32 in the artifact and bf16 in the source, so the comparison is
        // against the widened values. Order matters here in a way it does not for the
        // projections: the two POST norms take `post_norm_eps` and the two pre-norms
        // `rms_norm_eps`, three orders of magnitude apart, and all four are the same shape.
        for (p, t) in [
            (layer.input_ln, "input_layernorm"),
            (layer.post_attn_ln, "post_attention_layernorm"),
            (layer.pre_ffn_ln, "pre_feedforward_layernorm"),
            (layer.post_ffn_ln, "post_feedforward_layernorm"),
        ] {
            let full = format!("{}.{l}.{t}.weight", gm::GLIMMER_LAYER_PREFIX);
            let want: Vec<u8> = src
                .get(full.as_str())
                .unwrap()
                .chunks_exact(2)
                .flat_map(|c| {
                    rivoli::math::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])).to_le_bytes()
                })
                .collect();
            assert_eq!(
                slab(p as *const u8, cfg.text.hidden * 4),
                want,
                "{full} is wired to the wrong norm"
            );
        }
    }
    drop(pin); // release the tier before the next test's `DeviceTier::new`
    let _ = std::fs::remove_dir_all(&root);
}

/// **A config that describes a different model refuses, and names the tensor.**
///
/// The pin is the last place the two can be confronted: the converter checked the source
/// against the source's own config, and every kernel after this reads dims out of the pin. A
/// mismatch that reaches S2 is a GEMV over the wrong extent, which produces numbers.
///
/// `num_key_value_heads` doubled is the defect chosen because it makes the tier LARGER, so
/// what fires is the shape check rather than a capacity bail — a smaller config would refuse
/// for the wrong reason and this test would pass without exercising anything.
#[test]
fn glimmer_pin_refuses_a_config_the_artifact_does_not_match() {
    let (root, mut cfg, _) = convert("mismatch");
    cfg.text.num_key_value_heads *= 2;
    let e = format!(
        "{:#}",
        GlimmerPin::build(root.join("out").to_str().unwrap(), &cfg.text)
            .err()
            .expect("a config implying different dims must be refused")
    );
    assert!(
        e.contains("self_attn.k_proj") && e.contains("different models"),
        "the refusal must name the tensor, got: {e}"
    );
    let _ = std::fs::remove_dir_all(&root);
}
