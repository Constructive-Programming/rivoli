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
use common::{
    FixtureTensor, GLIMMER_FIXTURE_DIM as DIM, GLIMMER_FIXTURE_LAYERS as L, TempRoot,
    glimmer_convert_fixture,
};
use rivoli::artifact::model as gm;
use rivoli::memory::pin::GlimmerPin;

/// Convert the synthetic checkpoint and return `(artifact dir, config, source tensors)`.
/// The caller owns the temp root and removes it.
fn convert(tag: &str) -> (TempRoot, gm::GlimmerConfig, Vec<FixtureTensor>) {
    let root = TempRoot::new(&format!("glimmer-pin-{tag}"));
    let (tensors, _) = glimmer_convert_fixture(root.path(), DIM);
    let cfg = gm::load_config(root.join("out").to_str().unwrap()).unwrap();
    (root, cfg, tensors)
}

/// The source's bf16 bytes for `name`, widened to f32 — what the converter wrote and
/// therefore what the tier must hold.
fn widened(src: &std::collections::HashMap<&str, &Vec<u8>>, name: &str) -> Vec<u8> {
    src.get(name)
        .unwrap_or_else(|| panic!("{name} is not in the fixture"))
        .chunks_exact(2)
        .flat_map(|c| rivoli::math::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])).to_le_bytes())
        .collect()
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
    // `None` = pin everything, the all-resident partition this test has always exercised.
    // `tests/glimmer_residency.rs` covers the budgeted ones and the equivalence between them.
    let mut pin = GlimmerPin::build(root.join("out").to_str().unwrap(), &cfg.text, None).unwrap();

    assert_eq!(pin.pinned_layers(), L);
    assert_eq!(pin.streamed_layers(), 0, "None must pin every layer");
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

    // **`final_norm`, which had no assertion at all until review found it 2026-08-11.** It is
    // the RMSNorm before `lm_head`, it is a bare `*const f32` carrying no extent, and
    // `place_f32` checks only the dtype — so before `place_glimmer_norm` existed it could have
    // been wired to ANY other f32 tensor in the artifact, of any length, and every test in
    // this branch still passed. The fixture gives each tensor a distinct blob, so comparing
    // the bytes is what tells `model.norm` apart from four layer norms of the same shape.
    assert_eq!(
        slab(pin.final_norm as *const u8, cfg.text.hidden * 4),
        widened(&src, "model.language_model.norm.weight"),
        "final_norm is wired to the wrong tensor"
    );

    for l in 0..L {
        let layer = pin.layer(l).unwrap();
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
            assert_eq!(
                slab(p as *const u8, cfg.text.hidden * 4),
                widened(&src, &full),
                "{full} is wired to the wrong norm"
            );
        }
    }
    drop(pin); // release the tier before the next test's `DeviceTier::new`
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
///
/// The norm class has its own test below: its defect has to be in the ARTIFACT rather than in
/// the config, because every norm is checked against `hidden` and changing `hidden` trips the
/// embed/lm_head check first.
#[test]
fn glimmer_pin_refuses_a_config_the_artifact_does_not_match() {
    let (root, cfg, _) = convert("mismatch");
    let mut broken = cfg.text.clone();
    broken.num_key_value_heads *= 2;
    let e = format!(
        "{:#}",
        GlimmerPin::build(root.join("out").to_str().unwrap(), &broken, None)
            .err()
            .expect("a config implying different dims must be refused")
    );
    // Both halves separately, so which one failed is the useful half of the message: a refusal
    // that fires without naming the tensor is nearly as unhelpful as none.
    for (needle, must) in [
        ("self_attn.k_proj", "NAME the offending tensor"),
        (
            "different models",
            "say the config and the artifact disagree",
        ),
    ] {
        assert!(e.contains(needle), "the refusal must {must}. Got: {e}");
    }
}

/// **A norm of the wrong LENGTH is refused too** — the placement class that had no check at
/// all until 2026-08-11, found by two independent reviews.
///
/// The defect is in the ARTIFACT, and that is the truer form: an adversarial review
/// demonstrated that `convert_glimmer` accepts a short norm and **exits 0**, because
/// `SafeWriter::add_widened` copies the source shape and the converter's completeness loop
/// checks names only. So this is an artifact the shipped converter will really produce.
///
/// What the pin prevents: `place_f32` discards the shape and `GlimmerLayerPin`'s norm fields
/// are bare `*const f32` carrying no extent, so a short norm would be placed into a tier sized
/// for the full width and handed to S2's RMSNorm as a `hidden`-long array — reading
/// inter-placement padding and the next tensor's bytes. In bounds of the slab, no error, a
/// scaled-wrong residual stream on one layer's tail channels.
#[test]
fn glimmer_pin_refuses_a_norm_that_is_not_hidden_long() {
    let root = TempRoot::new("glimmer-shortnorm");
    let src = root.join("src");
    let mut tensors = common::glimmer_fixture(&src, DIM);

    // One layer norm, one element short. Nothing else is touched, so the refusal cannot be
    // attributed to any other tensor.
    let short = format!("{}.1.input_layernorm.weight", gm::GLIMMER_LAYER_PREFIX);
    let t = tensors
        .iter_mut()
        .find(|(n, _, _)| *n == short)
        .expect("the fixture ships this norm");
    t.1 = vec![DIM - 1];
    t.2.truncate((DIM - 1) * 2);
    common::write_safetensors(&src.join("model-00001-of-00001.safetensors"), &tensors);
    common::write_index(&src, &tensors);

    let out = root.join("out");
    let o = common::run_convert_glimmer(&src, &out);
    assert!(
        o.status.success(),
        "the converter is expected to ACCEPT this — it is the gap the pin closes: {}",
        String::from_utf8_lossy(&o.stderr)
    );

    let cfg: gm::GlimmerConfig = gm::load_config(out.to_str().unwrap()).unwrap();
    let e = format!(
        "{:#}",
        GlimmerPin::build(out.to_str().unwrap(), &cfg.text, None)
            .err()
            .expect("a norm shorter than hidden must be refused")
    );
    assert!(
        e.contains(&short),
        "the refusal must name the norm. Got: {e}"
    );
}
