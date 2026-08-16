//! **The TEXT goldens' structure: the tiny config, the widths it declares, the layer pattern, and
//! the capture census both ways.**
//!
//! One of the binaries the Muse Glimmer S1b anchor gate is split across — `glimmer_anchor.rs`
//! carries the framing, the byte pins and the argument for why a fixture-integrity gate is worth
//! having; `glimmer_anchor_common/mod.rs` carries the tables and the accessors. Nothing here reads a
//! value: every assertion below is about a SHAPE, a name, or a count, all of them derived from each
//! golden's own `tiny_config` so that a config drift fails rather than being agreed with.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#[path = "glimmer_anchor_common/mod.rs"]
mod anchor; // keep this preamble blank-line-free: spread out, these are a jscpd clone
use anchor::{
    GoldenSet, Vendored, Widths, cfg, each_text, ints, load, meta_usize, num, real, shape_is,
    shape_of, text_goldens,
};
use serde_json::Value;

// ------------------------------------------------------------------------------------------

/// Every one of these is a number a kernel will hard-code or a formula it will evaluate, and
/// getting any of them from the wrong place is a silent quality loss rather than a crash.
const REAL_FIELDS: &[&str] = &[
    "rms_norm_eps",
    "post_norm_eps",
    "qk_scale_factor",
    "output_multiplier",
    "final_logit_softcapping",
    "num_key_value_heads",
    "hidden_activation",
    "attention_bias",
    "attention_dropout",
    "tie_word_embeddings",
];

/// **Every field the driver calls REAL must still equal the real checkpoint's**, read out of the
/// vendored `config.json` rather than restated here.
///
/// This is the check that catches an upstream revision moving a value out from under the port: the
/// tiny config is a hand-written derivative of the real one, and a hand-written derivative is
/// exactly the thing that goes quietly stale.
#[test]
fn the_tiny_config_kept_the_real_values() {
    let want = real();
    each_text(|v, _g, got, _| {
        for key in REAL_FIELDS {
            assert_eq!(
                got[key], want[key],
                "{}: tiny config lost the real {key}",
                v.name
            );
        }
        assert_eq!(
            got["rope_parameters"]["rope_theta"], want["rope_parameters"]["rope_theta"],
            "{}: rope_theta",
            v.name
        );
        // The two-eps sandwich is the trap the whole anchor exists for. Asserting they are the REAL
        // values is not enough — assert they are DIFFERENT, because a future config where they
        // coincide would make every eps-related golden vacuous without failing the loop above.
        assert_ne!(
            got["rms_norm_eps"], got["post_norm_eps"],
            "{}: the two eps collapsed",
            v.name
        );
    });
}

// ------------------------------------------------------------------------------------------

/// The width collisions that would let a wrong reading pass.
///
/// K3's anchor review found an assertion satisfied by the wrong reading too, because four widths
/// had collided. These are the collisions that would matter here.
fn check_no_width_collision(v: &Vendored, c: &Value, w: Widths) {
    let who = v.name;
    // The real model is 6656 vs 32*128 = 4096. A port that assumes they are equal — the usual
    // assumption — passes on any tiny config where they are.
    assert_ne!(
        w.hidden,
        w.concat(),
        "{who}: hidden_size collapsed onto num_heads*head_dim"
    );
    assert_eq!(
        w.hidden % w.heads,
        0,
        "{who}: the reference's own validate_architecture requires this"
    );
    assert!(
        w.heads > w.kv && w.heads.is_multiple_of(w.kv),
        "{who}: GQA groups are not a clean ratio"
    );
    let group = w.group();
    assert!(
        group > 1,
        "{who}: group 1 is MHA and exercises no broadcast at all"
    );
    assert_ne!(
        group, w.kv,
        "{who}: group and kv-head count are equal, so the two cannot be told apart"
    );
    assert_ne!(
        num(c, "intermediate_size"),
        w.hidden,
        "{who}: SwiGLU width collapsed onto hidden"
    );
}

/// The window must be crossable: a sequence shorter than the window tests the dense path and
/// passes vacuously, which is exactly how a `--attn dsa` A/B covered nothing on GLM.
fn check_the_window_is_crossable(v: &Vendored, g: &GoldenSet, c: &Value) {
    let total = meta_usize(g, "prompt_len") + meta_usize(g, "decode_steps");
    let win = num(c, "sliding_window");
    assert!(
        win < total,
        "{}: window {win} >= the {total} positions generated, so nothing ever crosses it",
        v.name
    );
}

/// The widths were shrunk in a way that keeps every structural distinction the real model has.
#[test]
fn the_tiny_widths_did_not_collapse_a_distinction() {
    each_text(|v, g, c, w| {
        check_no_width_collision(v, c, w);
        check_the_window_is_crossable(v, g, c);
    });
}

// ------------------------------------------------------------------------------------------

/// One golden's own captured layer flags: the census, and the IFF that binds the two arrays.
fn check_the_captured_layer_flags(v: &Vendored, g: &GoldenSet) {
    let sliding = ints(g, "layer_is_sliding");
    let roped = ints(g, "layer_is_roped");
    assert_eq!(
        sliding.len(),
        num(&cfg(g), "num_hidden_layers"),
        "{}: layer census",
        v.name
    );
    assert_eq!(
        sliding, roped,
        "{}: a layer slides IFF it is rotated",
        v.name
    );
    check_the_backward_fourth_rule(sliding, v.name);
}

/// The same rule at the REAL 52 layers, from the vendored config, because a rule that holds only
/// at the tiny depth is a coincidence.
///
/// There `layer_types` and `layer_rope_theta` are two independent arrays and the fact that binds
/// them is nowhere in the file — it is in `__post_init__`, which computes both from the same
/// "every 4th counted backward from the last" rule.
fn check_the_real_configs_layer_pattern() {
    let want = real();
    let types: Vec<i64> = want["layer_types"]
        .as_array()
        .expect("layer_types")
        .iter()
        .map(|t| i64::from(t == "sliding_attention"))
        .collect();
    let thetas: Vec<i64> = want["layer_rope_theta"]
        .as_array()
        .expect("layer_rope_theta")
        .iter()
        .map(|t| i64::from(t.as_f64().expect("a theta") != 0.0))
        .collect();
    assert_eq!(
        types, thetas,
        "the REAL config: a layer slides IFF it is rotated"
    );
    check_the_backward_fourth_rule(&types, "the real 52-layer config");
}

fn check_the_backward_fourth_rule(sliding: &[i64], who: &str) {
    let n = sliding.len();
    for (i, s) in sliding.iter().enumerate() {
        let want = i64::from(!(n - 1 - i).is_multiple_of(4));
        assert_eq!(
            *s, want,
            "{who}: layer {i} of {n} is on the wrong side of the [w,w,w,full] rule"
        );
    }
    assert_eq!(
        sliding[0], 1,
        "{who}: layer 0 must be sliding, as it is in the real model"
    );
    assert_eq!(
        sliding[n - 1],
        0,
        "{who}: the last layer must be full attention"
    );
}

/// **The layer-type pattern and its NoPE coupling are the real rule, checked at BOTH depths.**
///
/// `layer_types` and `layer_rope_theta` are two independent arrays in the config, and the fact that
/// binds them — a layer is full attention IF AND ONLY IF it is NoPE — is nowhere in the file. It is
/// in `__post_init__`, which computes both from the same "every 4th counted backward from the last"
/// rule. Trap #1 in `glimmer-architecture.md` §9 is a port that reads the top-level `rope_theta`
/// and rotates all 52 layers; this is the assertion that would have caught it.
///
/// Checked at 8 layers from the goldens' own captured flags AND at 52 from the vendored real
/// config, because a rule that holds only at the tiny depth is a coincidence.
#[test]
fn full_attention_layers_are_exactly_the_nope_layers() {
    for v in text_goldens() {
        check_the_captured_layer_flags(v, &load(v));
    }
    check_the_real_configs_layer_pattern();
}

// ------------------------------------------------------------------------------------------

/// One decode step, with the two lengths the KV ring is derived from.
///
/// Bundled rather than passed apart because every shape check below needs most of it, and eight
/// loose arguments on each of them is what the code-health gate refuses.
#[derive(Clone, Copy)]
struct Step {
    w: Widths,
    /// Which step: 0 is the prefill.
    t: usize,
    /// Query rows this step — the whole prompt at `t == 0`, one token per decode step after.
    q: usize,
    prompt: usize,
    win: usize,
}

impl Step {
    /// The rows a layer's KV cache holds now. **Eviction, as a shape.** On a sliding layer the
    /// prefill still sees the whole prompt and is windowed by the MASK; from the first decode step
    /// the cache itself holds only `sliding_window` rows. A port may truncate during prefill
    /// instead and get the same numbers — what it may not do is keep more than the window after it.
    fn k_len(self, sliding: bool) -> usize {
        if sliding && self.t > 0 {
            self.win
        } else {
            self.prompt + self.t
        }
    }
}

/// The captures taken once per step, at the model's own two widths.
fn check_step_captures(g: &GoldenSet, c: &Value, s: Step) {
    let p = format!("t{}", s.t);
    shape_is(g, &format!("{p}.rope.cos"), &[1, s.q, s.w.head_dim]);
    shape_is(g, &format!("{p}.rope.sin"), &[1, s.q, s.w.head_dim]);
    shape_is(g, &format!("{p}.embed_norm.out"), &[1, s.q, s.w.hidden]);
    shape_is(g, &format!("{p}.final_norm.out"), &[1, s.q, s.w.hidden]);
    shape_is(g, &format!("{p}.logits"), &[1, num(c, "vocab_size")]);
}

/// The captures that come back at one of the layer's two widths: the four sandwich norms and the
/// two projections at hidden width, in the order the layer applies them — then the output gate and
/// the gated value it multiplies, both at Q width and both BEFORE `o_proj`, which is the point of
/// capturing `in_gated` separately.
fn check_layer_width_captures(g: &GoldenSet, p: &str, s: Step) {
    for what in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "mlp.down_proj",
        "attn.o_proj",
    ] {
        shape_is(g, &format!("{p}.{what}.out"), &[1, s.q, s.w.hidden]);
    }
    for what in ["attn.gate_proj.out", "attn.o_proj.in_gated"] {
        shape_is(g, &format!("{p}.{what}"), &[1, s.q, s.w.concat()]);
    }
}

/// Q, K, the attention itself, and the ring the layer type implies.
fn check_layer_attention_captures(g: &GoldenSet, p: &str, s: Step, klen: usize) {
    let (heads, kv, hd) = (s.w.heads, s.w.kv, s.w.head_dim);
    shape_is(g, &format!("{p}.qk_norm.q"), &[1, heads, s.q, hd]);
    shape_is(g, &format!("{p}.qk_norm.k"), &[1, kv, s.q, hd]);
    shape_is(g, &format!("{p}.attend.q"), &[1, heads, s.q, hd]);
    shape_is(g, &format!("{p}.attend.out"), &[1, s.q, heads, hd]);
    for what in ["attend.k_cache", "attend.v_cache"] {
        let name = format!("{p}.{what}");
        assert_eq!(
            shape_of(g, &name),
            vec![1, kv, klen, hd],
            "{name}: the ring did not hold what the layer type implies"
        );
    }
    shape_is(g, &format!("{p}.attend.mask"), &[1, 1, s.q, klen]);
    let weights = format!("{p}.attend.weights");
    assert_eq!(
        shape_of(g, &weights),
        vec![1, heads, s.q, klen],
        "{weights}: GQA broadcast did not reach the head count"
    );
}

/// **The rope captures exist on exactly the rotated layers**, which is the same coupling as
/// `full_attention_layers_are_exactly_the_nope_layers` seen from the capture side: a NoPE layer
/// that produced one would mean the reference rotated it.
fn check_layer_rope_captures(g: &GoldenSet, p: &str, s: Step, roped: bool) {
    let has_rope = g
        .floats
        .iter()
        .any(|(n, _, _)| n == &format!("{p}.q.roped"));
    assert_eq!(
        has_rope, roped,
        "{p}: rope captures present={has_rope} but layer_is_roped={roped}"
    );
    if !has_rope {
        return;
    }
    for what in ["q.pre_rope", "q.roped"] {
        shape_is(
            g,
            &format!("{p}.{what}"),
            &[1, s.w.heads, s.q, s.w.head_dim],
        );
    }
    for what in ["k.pre_rope", "k.roped"] {
        shape_is(g, &format!("{p}.{what}"), &[1, s.w.kv, s.q, s.w.head_dim]);
    }
}

/// One layer at one step, in the three groups its captures fall into.
fn check_layer_captures(g: &GoldenSet, l: usize, s: Step, flags: (bool, bool)) {
    let (sliding, roped) = flags;
    let p = format!("t{}.L{l}", s.t);
    check_layer_width_captures(g, &p, s);
    check_layer_attention_captures(g, &p, s, s.k_len(sliding));
    check_layer_rope_captures(g, &p, s, roped);
}

/// One text golden's whole capture set, step by step and layer by layer.
fn check_text_captures(g: &GoldenSet, c: &Value, w: Widths) {
    let prompt = meta_usize(g, "prompt_len");
    let win = num(c, "sliding_window");
    let layers = num(c, "num_hidden_layers");
    let sliding = ints(g, "layer_is_sliding").to_vec();
    let roped = ints(g, "layer_is_roped").to_vec();
    for t in 0..=meta_usize(g, "decode_steps") {
        let q = if t == 0 { prompt } else { 1 };
        let s = Step {
            w,
            t,
            q,
            prompt,
            win,
        };
        check_step_captures(g, c, s);
        for l in 0..layers {
            check_layer_captures(g, l, s, (sliding[l] == 1, roped[l] == 1));
        }
    }
}

/// Every per-operator fixture S2 will reach for is present, at the width its config implies.
///
/// The shapes are computed from `tiny_config`, so this fails when the config drifts instead of
/// agreeing with it. **`attend.k_cache`'s length is the ring-KV assertion**: on a sliding layer the
/// cache holds exactly `sliding_window` rows once decoding starts, and on a full layer it grows —
/// which is eviction, observed rather than described.
#[test]
fn the_operator_fixtures_s2_needs_are_present() {
    each_text(|_v, g, c, w| check_text_captures(g, c, w));
}

// ------------------------------------------------------------------------------------------

/// The int captures: two token lists at the lengths the metadata declares, and no others.
fn check_int_captures(v: &Vendored, g: &GoldenSet, steps: usize) {
    assert_eq!(
        g.ints.len(),
        4,
        "{}: prompt.ids, emitted.ids and the two layer flags",
        v.name
    );
    assert_eq!(
        ints(g, "prompt.ids").len(),
        meta_usize(g, "prompt_len"),
        "{}",
        v.name
    );
    assert_eq!(
        ints(g, "emitted.ids").len(),
        steps,
        "{}: one token per step",
        v.name
    );
}

/// Nothing was captured beyond what the census implies.
///
/// The shape test above asserts that every expected capture is PRESENT; on its own that would pass
/// a file carrying an extra hundred tensors from a stale run. This is the other direction, and it
/// is derived rather than written: the count follows from the config and the step count.
#[test]
fn exactly_the_declared_captures_are_present() {
    each_text(|v, g, c, _| {
        let steps = meta_usize(g, "decode_steps") + 1;
        let layers = num(c, "num_hidden_layers");
        let roped = ints(g, "layer_is_roped")
            .iter()
            .filter(|r| **r == 1)
            .count();
        // Per step: cos, sin, embed_norm, final_norm, logits.
        // Per layer: 4 norms + mlp + o_proj.out + o_proj.in_gated + gate_proj + qk_norm x2
        //            + attend q/k/v/mask/weights/out = 16.
        // Per ROTATED layer: 4 more.
        let want = steps * (5 + layers * 16 + roped * 4);
        assert_eq!(g.floats.len(), want, "{}: float capture census", v.name);
        check_int_captures(v, g, steps);
    });
}
