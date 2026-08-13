//! **The vendored Muse Glimmer S1b anchor goldens must stay loadable, complete, self-describing,
//! and the exact bytes that were measured.**
//!
//! `docs/measurement/glimmer-reference/anchor.md` is the record; `tests/glimmer_anchor_driver.py`
//! produced the files and `tests/glimmer-anchor.sh` reproduces them.
//!
//! **This is a fixture-integrity gate, not a correctness gate for the port**, and that is worth
//! saying plainly: nothing here compares any rivoli output to a golden, because at S1b there is no
//! Glimmer kernel to score — so the literal answer to "what wrong implementation passes this" is
//! every one. What it does is hold the files to the shape S2's kernels will reach for, refuse a
//! file that is not the one the doc describes, and refuse a tiny config that has stopped matching
//! the real checkpoint's structure.
//!
//! **No GPU, no python, no network.** Generating a golden needs a pinned venv; reading one needs
//! nothing, which is why the bytes are vendored. Unlike K3's anchor it does not need a device even
//! to generate — this reference is plain PyTorch — but the vendoring argument is unchanged.
//!
//! **Two salts and two modes are vendored and every test runs over all of them.** One draw cannot
//! show that a property is a fact about the arithmetic rather than about the numbers it landed on.
//!
//! Widths are **derived from each golden's own `tiny_config`**, never written as literals, and the
//! fields that are supposed to be REAL are compared against the vendored `config.json` rather than
//! against constants. A literal agrees with drift; a derived value fails on it.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use serde_json::Value;

#[path = "common/golden_read.rs"]
mod golden_read;

use golden_read::{GoldenSet, Vendored, ints, shape_of};

/// Which mode a vendored file holds, read off the entry's own name.
///
/// **The mode is not a field here**, deliberately: the file already says what it is, the byte pin
/// binds this entry to that file, and a second copy of a fact the file carries is the shape this
/// port has already been bitten by. `the_anchor_goldens_record_what_produced_them` checks the two
/// agree.
fn is_mode(v: &Vendored, mode: &str) -> bool {
    v.name.starts_with(mode)
}

const GOLDENS: &[Vendored] = &[
    Vendored {
        name: "text-1",
        bytes: include_bytes!("glimmer-anchor-text-1.bin"),
        len: 643_957,
        fnv: 0xc765_6dea_dd50_3c51,
    },
    Vendored {
        name: "text-2",
        bytes: include_bytes!("glimmer-anchor-text-2.bin"),
        len: 643_957,
        fnv: 0xe778_0679_924e_cd5f,
    },
    Vendored {
        name: "draft-1",
        bytes: include_bytes!("glimmer-anchor-draft-1.bin"),
        len: 72_145,
        fnv: 0x3dcf_a1ed_6536_a6f0,
    },
    Vendored {
        name: "draft-2",
        bytes: include_bytes!("glimmer-anchor-draft-2.bin"),
        len: 72_145,
        fnv: 0xd15d_109a_9a72_f7ab,
    },
];

/// The vendored real config, the same file `glimmer-architecture.md` was extracted from.
const REAL_CONFIG: &str = include_str!("../docs/measurement/glimmer-reference/config.json");

fn load(v: &Vendored) -> GoldenSet {
    GoldenSet::read_glimmer(&mut &v.bytes[..]).unwrap_or_else(|e| panic!("{}: {e:#}", v.name))
}

fn text_goldens() -> impl Iterator<Item = &'static Vendored> {
    GOLDENS.iter().filter(|v| is_mode(v, "text"))
}

fn draft_goldens() -> impl Iterator<Item = &'static Vendored> {
    GOLDENS.iter().filter(|v| is_mode(v, "draft"))
}

/// Run `f` over every text golden, already loaded, with its config and the four widths.
///
/// Three checks below started with the same four lines — load, parse the config, read the widths,
/// name the file — and `build.rs`'s jscpd gate rejected them at 73 tokens. The duplication was
/// real; this is the factoring, not an exemption.
fn each_text(mut f: impl FnMut(&Vendored, &GoldenSet, &Value, (usize, usize, usize, usize))) {
    for v in text_goldens() {
        let g = load(v);
        let c = cfg(&g);
        let w = widths(&c);
        f(v, &g, &c, w);
    }
}

/// The four widths every shape below is built from, so that a config drift fails the gate instead
/// of being agreed with. Read together because they are only meaningful together.
fn widths(c: &Value) -> (usize, usize, usize, usize) {
    (
        num(c, "hidden_size"),
        num(c, "num_attention_heads"),
        num(c, "num_key_value_heads"),
        num(c, "head_dim"),
    )
}

/// The tiny config a golden was produced under, parsed out of its own metadata.
fn cfg(g: &GoldenSet) -> Value {
    serde_json::from_str(meta(g, "tiny_config")).expect("tiny_config is JSON")
}

fn meta<'g>(g: &'g GoldenSet, key: &str) -> &'g str {
    g.meta_get(key)
        .unwrap_or_else(|| panic!("the golden carries no {key:?} in its metadata"))
}

fn num(c: &Value, key: &str) -> usize {
    c[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} is not an integer in the config: {}", c[key])) as usize
}

fn meta_usize(g: &GoldenSet, key: &str) -> usize {
    meta(g, key).parse().expect("a numeric metadata value")
}

fn real() -> Value {
    let top: Value = serde_json::from_str(REAL_CONFIG).expect("the vendored config.json parses");
    top["text_config"].clone()
}

// ------------------------------------------------------------------------------------------

/// The provenance every consumer has to be able to read off the file, **by value**.
///
/// A golden separated from the versions that produced it cannot be re-derived, and these were
/// produced by a stack that is not in the repo — transformers at a git revision, in a venv that
/// exists on one machine. `transformers_commit` is checked to be a full hex sha rather than merely
/// present, because the driver's own fallback for "not installed from git" is the string
/// `"unknown"`, which is non-empty and would satisfy a presence check.
#[test]
fn the_anchor_goldens_record_what_produced_them() {
    let mut env = None;
    for v in GOLDENS {
        let g = load(v);
        let who = v.name;

        assert_eq!(meta(&g, "model"), "muse-glimmer", "{who}: model");
        assert!(
            is_mode(v, meta(&g, "mode")),
            "{who}: the file says mode={}",
            meta(&g, "mode")
        );
        // The salt is read off the file, and the entry name carries only its index.
        assert!(
            meta(&g, "salt").ends_with(v.name.rsplit('-').next().unwrap()),
            "{who}: the file says salt={}",
            meta(&g, "salt")
        );
        assert_eq!(
            g.defect(),
            "None",
            "{who}: a vendored golden must be unperturbed"
        );
        assert_eq!(
            meta(&g, "driver"),
            "glimmer_anchor_driver.py",
            "{who}: driver"
        );
        assert_eq!(meta(&g, "attn_implementation"), "eager", "{who}: attn impl");
        // The declared deviation that S2's tolerance decision starts from: fp32 against a bf16
        // checkpoint. If this ever changes, the tolerances derived from these goldens are stale.
        assert_eq!(meta(&g, "dtype"), "torch.float32", "{who}: dtype");

        let commit = meta(&g, "transformers_commit");
        assert_eq!(
            commit.len(),
            40,
            "{who}: transformers_commit is not a full sha: {commit:?}"
        );
        assert!(
            commit.bytes().all(|b| b.is_ascii_hexdigit()),
            "{who}: transformers_commit is not hex: {commit:?}"
        );

        // Every golden must come from ONE environment. Two files pinned to different transformers
        // revisions cannot be compared to each other, and a port that scored against both would be
        // scoring against two references.
        let here: Vec<String> = [
            "torch",
            "transformers",
            "transformers_commit",
            "numpy",
            "python",
        ]
        .iter()
        .map(|k| format!("{k}={}", meta(&g, k)))
        .collect();
        match &env {
            None => env = Some((who, here)),
            Some((first, want)) => assert_eq!(
                &here, want,
                "{who} was produced under a different env than {first}"
            ),
        }
    }
    assert_eq!(GOLDENS.len(), 4, "two modes x two salts");
}

/// The bytes are the ones the doc describes, to the byte.
///
/// **When this fails after a deliberate regeneration, update the constants and say so in
/// `anchor.md`.** That is the intended workflow: re-vendoring is a reviewed change, not a side
/// effect of running the driver.
#[test]
fn the_vendored_bytes_are_the_measured_ones() {
    for v in GOLDENS {
        v.check_bytes();
    }
    // Two salts are coverage, not redundancy. Identical bytes would mean the salt never reached the
    // weights, and every "both salts" claim above would be one claim wearing two names.
    for mode in ["text", "draft"] {
        let mut it = GOLDENS.iter().filter(|v| is_mode(v, mode));
        let (a, b) = (it.next().unwrap(), it.next().unwrap());
        assert_ne!(
            a.fnv, b.fnv,
            "{mode}: the two salts produced identical files"
        );
    }
}

/// **Every field the driver calls REAL must still equal the real checkpoint's**, read out of the
/// vendored `config.json` rather than restated here.
///
/// This is the check that catches an upstream revision moving a value out from under the port: the
/// tiny config is a hand-written derivative of the real one, and a hand-written derivative is
/// exactly the thing that goes quietly stale.
#[test]
fn the_tiny_config_kept_the_real_values() {
    let want = real();
    // Every one of these is a number a kernel will hard-code or a formula it will evaluate, and
    // getting any of them from the wrong place is a silent quality loss rather than a crash.
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
    for v in text_goldens() {
        let g = load(v);
        let got = cfg(&g);
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
    }
}

/// The widths were shrunk in a way that keeps every structural distinction the real model has.
///
/// K3's anchor review found an assertion satisfied by the wrong reading too, because four widths
/// had collided. These are the collisions that would matter here.
#[test]
fn the_tiny_widths_did_not_collapse_a_distinction() {
    each_text(|v, g, c, (h, heads, kv, hd)| {
        let who = v.name;
        // The real model is 6656 vs 32*128 = 4096. A port that assumes they are equal — the usual
        // assumption — passes on any tiny config where they are.
        assert_ne!(
            h,
            heads * hd,
            "{who}: hidden_size collapsed onto num_heads*head_dim"
        );
        assert_eq!(
            h % heads,
            0,
            "{who}: the reference's own validate_architecture requires this"
        );
        assert!(
            heads > kv && heads % kv == 0,
            "{who}: GQA groups are not a clean ratio"
        );
        let group = heads / kv;
        assert!(
            group > 1,
            "{who}: group 1 is MHA and exercises no broadcast at all"
        );
        assert_ne!(
            group, kv,
            "{who}: group and kv-head count are equal, so the two cannot be told apart"
        );
        assert_ne!(
            num(c, "intermediate_size"),
            h,
            "{who}: SwiGLU width collapsed onto hidden"
        );
        // The window must be crossable: a sequence shorter than the window tests the dense path and
        // passes vacuously, which is exactly how a `--attn dsa` A/B covered nothing on GLM.
        let total = meta_usize(g, "prompt_len") + meta_usize(g, "decode_steps");
        assert!(
            num(c, "sliding_window") < total,
            "{who}: window {} >= the {total} positions generated, so nothing ever crosses it",
            num(c, "sliding_window")
        );
    });
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
        let g = load(v);
        let sliding = ints(&g, "layer_is_sliding");
        let roped = ints(&g, "layer_is_roped");
        assert_eq!(
            sliding.len(),
            num(&cfg(&g), "num_hidden_layers"),
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
        let want = i64::from((n - 1 - i) % 4 != 0);
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

/// Every per-operator fixture S2 will reach for is present, at the width its config implies.
///
/// The shapes are computed from `tiny_config`, so this fails when the config drifts instead of
/// agreeing with it. **`attend.k_cache`'s length is the ring-KV assertion**: on a sliding layer the
/// cache holds exactly `sliding_window` rows once decoding starts, and on a full layer it grows —
/// which is eviction, observed rather than described.
#[test]
fn the_operator_fixtures_s2_needs_are_present() {
    each_text(|_v, g, c, (h, heads, kv, hd)| {
        let (layers, win) = (num(c, "num_hidden_layers"), num(c, "sliding_window"));
        let (prompt, steps) = (meta_usize(g, "prompt_len"), meta_usize(g, "decode_steps"));
        let sliding = ints(g, "layer_is_sliding").to_vec();
        let roped = ints(g, "layer_is_roped").to_vec();

        for t in 0..=steps {
            let q = if t == 0 { prompt } else { 1 };
            let p = format!("t{t}");
            assert_eq!(
                shape_of(g, &format!("{p}.rope.cos")),
                vec![1, q, hd],
                "{p} rope.cos"
            );
            assert_eq!(
                shape_of(g, &format!("{p}.rope.sin")),
                vec![1, q, hd],
                "{p} rope.sin"
            );
            assert_eq!(
                shape_of(g, &format!("{p}.embed_norm.out")),
                vec![1, q, h],
                "{p} embed_norm"
            );
            assert_eq!(
                shape_of(g, &format!("{p}.final_norm.out")),
                vec![1, q, h],
                "{p} final_norm"
            );
            assert_eq!(
                shape_of(g, &format!("{p}.logits")),
                vec![1, num(c, "vocab_size")],
                "{p} logits"
            );

            for l in 0..layers {
                let p = format!("t{t}.L{l}");
                // The four sandwich norms, in the order the layer applies them.
                for what in [
                    "input_layernorm",
                    "post_attention_layernorm",
                    "pre_feedforward_layernorm",
                    "post_feedforward_layernorm",
                    "mlp.down_proj",
                    "attn.o_proj",
                ] {
                    assert_eq!(
                        shape_of(g, &format!("{p}.{what}.out")),
                        vec![1, q, h],
                        "{p}.{what}"
                    );
                }
                // The output gate and the gated value it multiplies, both at Q width, both BEFORE
                // `o_proj` — which is the point of capturing `in_gated` separately.
                for what in ["attn.gate_proj.out", "attn.o_proj.in_gated"] {
                    assert_eq!(
                        shape_of(g, &format!("{p}.{what}")),
                        vec![1, q, heads * hd],
                        "{p}.{what}"
                    );
                }
                assert_eq!(
                    shape_of(g, &format!("{p}.qk_norm.q")),
                    vec![1, heads, q, hd],
                    "{p} qk_norm.q"
                );
                assert_eq!(
                    shape_of(g, &format!("{p}.qk_norm.k")),
                    vec![1, kv, q, hd],
                    "{p} qk_norm.k"
                );
                assert_eq!(
                    shape_of(g, &format!("{p}.attend.q")),
                    vec![1, heads, q, hd],
                    "{p} attend.q"
                );
                assert_eq!(
                    shape_of(g, &format!("{p}.attend.out")),
                    vec![1, q, heads, hd],
                    "{p} attend.out"
                );

                // Eviction, as a shape. On a sliding layer the prefill still sees the whole prompt
                // and is windowed by the MASK; from the first decode step the cache itself holds
                // only `sliding_window` rows. A port may truncate during prefill instead and get
                // the same numbers — what it may not do is keep more than the window after it.
                let klen = if sliding[l] == 1 && t > 0 {
                    win
                } else {
                    prompt + t
                };
                for what in ["attend.k_cache", "attend.v_cache"] {
                    assert_eq!(
                        shape_of(g, &format!("{p}.{what}")),
                        vec![1, kv, klen, hd],
                        "{p}.{what}: the ring did not hold what the layer type implies"
                    );
                }
                assert_eq!(
                    shape_of(g, &format!("{p}.attend.mask")),
                    vec![1, 1, q, klen],
                    "{p} mask"
                );
                assert_eq!(
                    shape_of(g, &format!("{p}.attend.weights")),
                    vec![1, heads, q, klen],
                    "{p} attend.weights: GQA broadcast did not reach the head count"
                );

                // **The rope captures exist on exactly the rotated layers**, which is the same
                // coupling as `full_attention_layers_are_exactly_the_nope_layers` seen from the
                // capture side: a NoPE layer that produced one would mean the reference rotated it.
                let has_rope = g
                    .floats
                    .iter()
                    .any(|(n, _, _)| n == &format!("{p}.q.roped"));
                assert_eq!(
                    has_rope,
                    roped[l] == 1,
                    "{p}: rope captures present={has_rope} but layer_is_roped={}",
                    roped[l]
                );
                if has_rope {
                    for what in ["q.pre_rope", "q.roped"] {
                        assert_eq!(
                            shape_of(g, &format!("{p}.{what}")),
                            vec![1, heads, q, hd],
                            "{p}.{what}"
                        );
                    }
                    for what in ["k.pre_rope", "k.roped"] {
                        assert_eq!(
                            shape_of(g, &format!("{p}.{what}")),
                            vec![1, kv, q, hd],
                            "{p}.{what}"
                        );
                    }
                }
            }
        }
    });
}

/// Nothing was captured beyond what the census implies.
///
/// The shape test above asserts that every expected capture is PRESENT; on its own that would pass
/// a file carrying an extra hundred tensors from a stale run. This is the other direction, and it
/// is derived rather than written: the count follows from the config and the step count.
#[test]
fn exactly_the_declared_captures_are_present() {
    for v in text_goldens() {
        let g = load(v);
        let c = cfg(&g);
        let layers = num(&c, "num_hidden_layers");
        let steps = meta_usize(&g, "decode_steps") + 1;
        let roped: usize = ints(&g, "layer_is_roped")
            .iter()
            .filter(|r| **r == 1)
            .count();

        // Per step: cos, sin, embed_norm, final_norm, logits.
        // Per layer: 4 norms + mlp + o_proj.out + o_proj.in_gated + gate_proj + qk_norm x2
        //            + attend q/k/v/mask/weights/out = 16.
        // Per ROTATED layer: 4 more.
        let want = steps * (5 + layers * 16 + roped * 4);
        assert_eq!(g.floats.len(), want, "{}: float capture census", v.name);
        assert_eq!(
            g.ints.len(),
            4,
            "{}: prompt.ids, emitted.ids and the two layer flags",
            v.name
        );
        assert_eq!(
            ints(&g, "prompt.ids").len(),
            meta_usize(&g, "prompt_len"),
            "{}",
            v.name
        );
        assert_eq!(
            ints(&g, "emitted.ids").len(),
            steps,
            "{}: one token per step",
            v.name
        );
    }
}

/// The captured values are finite, and not the degenerate all-equal tensors a broken draw produces.
///
/// A golden of zeros, NaNs or one repeated constant agrees with every implementation, and the
/// centered norms make that a live risk rather than a hypothetical: they apply `(1 + w)`, so a `w`
/// drawn near 1.0 instead of near 0.0 doubles every activation per layer and the tail of an
/// eight-layer model is numerical noise.
#[test]
fn the_captured_values_are_finite_and_not_degenerate() {
    for v in GOLDENS {
        let g = load(v);
        for (name, _shape, vals) in &g.floats {
            let who = format!("{} {name}", v.name);
            assert!(!vals.is_empty(), "{who}: empty tensor");
            assert!(
                vals.iter().all(|x| x.is_finite()),
                "{who}: non-finite value"
            );
            // Masks are legitimately two-valued and legitimately all-ones on a full layer's first
            // row, so they are exempt from the spread check but not from finiteness.
            if name.ends_with("attend.mask") {
                continue;
            }
            let (lo, hi) = vals
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(*x), hi.max(*x)));
            assert!(
                hi > lo,
                "{who}: every element is {lo}, which agrees with any implementation"
            );
            assert!(
                hi.abs() < 1e6,
                "{who}: magnitude {hi} — the draw has blown up"
            );
        }
    }
}

/// **The DFlash golden has the shapes that make the drafter a drafter, not a small target.**
///
/// `glimmer-architecture.md` §11: Q comes from the block alone while K/V span `context + block`
/// inside one call, the context enters through `encoder.fc` at `len(target_layer_ids) * hidden`,
/// and the KV group count is not the target's. Each of those is a way a port goes wrong by reusing
/// the target's attention path, and each is a shape.
#[test]
fn the_draft_golden_has_the_shapes_that_make_it_a_drafter() {
    for v in draft_goldens() {
        let g = load(v);
        let c = cfg(&g);
        let (h, heads, kv, hd) = widths(&c);
        let block = meta_usize(&g, "block_size");
        let ctx = meta_usize(&g, "context_len");
        let layers = num(&c, "num_hidden_layers");
        let who = v.name;

        let targets = ints(&g, "target_layer_ids");
        assert_eq!(
            shape_of(&g, "draft.context_concat"),
            vec![1, ctx, targets.len() * h],
            "{who}: the context is one column block per target_layer_id"
        );
        assert_eq!(
            shape_of(&g, "draft.encoder.out"),
            vec![1, ctx, h],
            "{who}: encoder output"
        );
        assert_eq!(
            shape_of(&g, "draft.noise_embeds"),
            vec![1, block, h],
            "{who}: the draft block"
        );
        assert_eq!(ints(&g, "draft.block_ids").len(), block, "{who}: block ids");
        // One anchor token plus masks in, `block - 1` candidates out: index 0 is sliced off.
        assert_eq!(
            ints(&g, "draft.candidates").len(),
            block - 1,
            "{who}: candidates"
        );
        // **The drafter's own config has no `vocab_size`**, because it owns neither the embedding
        // nor the lm_head — it borrows the target's (section 11). So the logit width has to come
        // from the TARGET's config, and asserting the absence is asserting the borrow.
        assert!(
            c["vocab_size"].is_null(),
            "{who}: the drafter has acquired a vocab of its own"
        );
        let vocab = num(&cfg(&load(text_goldens().next().unwrap())), "vocab_size");
        assert_eq!(
            shape_of(&g, "draft.logits"),
            vec![1, block, vocab],
            "{who}: logits"
        );

        for l in 0..layers {
            let p = format!("draft.L{l}");
            assert_eq!(
                shape_of(&g, &format!("{p}.attend.q")),
                vec![1, heads, block, hd],
                "{p} Q"
            );
            for what in ["attend.k", "attend.v"] {
                assert_eq!(
                    shape_of(&g, &format!("{p}.{what}")),
                    vec![1, kv, ctx + block, hd],
                    "{p}.{what}: K/V must span context+block while Q spans block alone"
                );
            }
            assert_eq!(
                shape_of(&g, &format!("{p}.attend.mask")),
                vec![1, 1, block, ctx + block],
                "{p} mask"
            );
            // Two norms per layer, not four: the drafter is plain pre-norm and has no post-FFN norm.
            for what in ["input_layernorm", "post_attention_layernorm", "mlp"] {
                assert_eq!(
                    shape_of(&g, &format!("{p}.{what}.out")),
                    vec![1, block, h],
                    "{p}.{what}"
                );
            }
            assert!(
                g.floats
                    .iter()
                    .all(|(n, _, _)| n != &format!("{p}.post_feedforward_layernorm.out")),
                "{p}: the drafter has no post-FFN norm; a capture for one means it was built as a target layer"
            );
        }
    }
}

/// **The drafter's attention shape differs from the target's**, which is the property that makes a
/// port reusing the target's path fail rather than silently pass.
#[test]
fn the_drafter_does_not_share_the_targets_attention_shape() {
    let target = cfg(&load(text_goldens().next().unwrap()));
    let drafter = cfg(&load(draft_goldens().next().unwrap()));
    let group = |c: &Value| num(c, "num_attention_heads") / num(c, "num_key_value_heads");
    assert_ne!(
        group(&target),
        group(&drafter),
        "the two GQA group counts are equal, so a port that reuses the target's shape passes here \
         and fails on the real 16:1 against 4:1"
    );
    assert_eq!(
        num(&target, "hidden_size"),
        num(&drafter, "hidden_size"),
        "the drafter borrows the target's embedding and lm_head, so the widths must match"
    );
    // The real pairing, from the vendored config, so the tiny ratio is not the only evidence.
    let real = real();
    assert_eq!(
        num(&real, "num_attention_heads") / num(&real, "num_key_value_heads"),
        16
    );
}

/// **The weightless QK-norm's AXIS, from the goldens' own bytes.**
///
/// A weightless RMS over `d` leaves `mean(y²) = mean(x²)/(mean(x²)+eps)`, i.e. 1 to within
/// `eps/mean(x²)`. So every contiguous `head_dim` run of every `qk_norm.*` capture must have unit
/// mean square — and that is a property only of a norm taken over THAT axis. A reference (or a port)
/// normalising over the whole hidden state, over rows, or over the head COUNT leaves runs whose mean
/// square is anything else, and every shape check still passes.
///
/// **Here rather than beside the kernel, because it needs no device.** `tests/glimmer_qk_norm.rs` is
/// `#![cfg(feature = "rocm")]` end to end, and the only automated job in this repo is the FEATURELESS
/// `host` one — so a golden-property check parked in that file is checked exactly as often as
/// someone runs a GPU build by hand. Review, 2026-08-12.
///
/// **MEASURED: 2,304 head rows, worst |mean(y²) − 1| = 8.106e-4**, which is the eps term and nothing
/// else at the reference's own activation scale.
#[test]
fn the_qk_norm_captures_are_normalised_over_head_dim_per_head() {
    let (mut worst, mut runs) = (0.0f64, 0usize);
    each_text(|v, g, c, (_, heads, kv, hd)| {
        let layers = num(c, "num_hidden_layers");
        let steps = meta_usize(g, "decode_steps");
        for t in 0..=steps {
            for l in 0..layers {
                for (side, h) in [("q", heads), ("k", kv)] {
                    let name = format!("t{t}.L{l}.qk_norm.{side}");
                    let (shape, vals) = golden_read::float(g, &name);
                    assert_eq!(shape[1], h, "{}: {name} head count", v.name);
                    assert_eq!(shape[3], hd, "{}: {name} head_dim", v.name);
                    for row in vals.chunks(hd) {
                        let m =
                            row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / hd as f64;
                        worst = worst.max((m - 1.0).abs());
                        runs += 1;
                    }
                }
            }
        }
    });
    println!("{runs} head rows, worst |mean(y^2) - 1| = {worst:e}");
    // The bound is the eps term itself, not a tolerance: a run's mean square falls below 1 by at
    // most eps/mean(x²), which is ~1e-3 at these activations. 1e-2 leaves an order of margin while
    // rejecting any other axis by orders.
    assert!(
        worst < 1e-2,
        "a head_dim run has mean(y^2) off 1 by {worst:e} — the reference did not normalise over \
         this axis"
    );
    // **An absolute, not `runs > 0`.** `layers` comes from `tiny_config` and `steps` from the
    // golden's own metadata, so a census derived from them cannot notice either one shrinking — a
    // re-vendor whose `decode_steps` drops to 1, or a `num_hidden_layers` drifted from
    // `layer_is_sliding`, would leave `worst` a max over fewer rows with every assert still green.
    // That is the rule this tree wrote down for the sandwich norms' 612/34 one item ago, applied
    // here after review pointed out this test had the weak form. 2,304 is what the line above
    // PRINTS on the vendored goldens; no factorisation of it is asserted, because a factorisation
    // would be a second derived count standing in for the absolute.
    assert_eq!(
        runs, 2304,
        "the axis census covered {runs} head rows, not 2,304"
    );
}

/// **Trap 3, refuted by the reference's own bytes: Q is scaled by 3.87 and K is not.**
///
/// `q.pre_rope` is captured on entry to `apply_rotary_pos_emb`, i.e. after the norm AND after the
/// scale, while `qk_norm.q` is the norm's output before it — so their ratio is `qk_scale_factor`
/// elementwise. `k.pre_rope` must be BIT-IDENTICAL to `qk_norm.k`.
///
/// **What each half is worth is NOT the same, and review corrected the framing.** The Q ratio is
/// genuinely informative: it falls out of two DIFFERENT tensors and would break if the scale moved or
/// changed value. The K assert is a tautology over these bytes — `modeling_muse_glimmer.py:342`
/// normalises K and line 347 is the next statement touching it, so the forward hook's `out` and the
/// rope tap's `k` are the SAME tensor object serialised twice. It cannot distinguish "the reference
/// does not scale K" from "the harness captured one tensor under two names". Kept, because what it
/// CAN catch is real: **a re-vendor where a future transformers release inserts any op between those
/// two lines.** Call it a tripwire, not a gate.
///
/// **And it constrains the reference, not rivoli.** Trap 3 is a port-side defect; nothing here stops
/// a caller passing 3.87 for K, the anchor has no defect run for that form, and until the layer loop
/// lands there is no call site to gate. `kernels/linalg.hip` says so at the kernel. The
/// `qk_scale_on_k` defect that exists scales `k_proj`'s output, upstream of a norm that cancels a
/// scalar only up to the eps term. Its residue peaks at 6.2e-4 — which this line used to call
/// "nothing", and which is **7.9x the `qk_norm` row's own tolerance**. See `tolerance.rs` for the
/// correction, the closed form, and why the row's exclusion now rests on margin instead.
///
/// **MEASURED: the ratio over 10,368 elements is [3.8699996, 3.8700001]** — f32 rounding on one
/// multiply, not a tolerance — **and 3,456 K elements are unchanged, across 28 roped cases.**
#[test]
fn the_reference_scales_q_by_qk_scale_factor_and_leaves_k_alone() {
    let want = real()["qk_scale_factor"].as_f64().expect("qk_scale_factor") as f32;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut pairs, mut k_elems, mut skipped) = (0usize, 0usize, 0usize);
    each_text(|v, g, c, _| {
        let layers = num(c, "num_hidden_layers");
        let steps = meta_usize(g, "decode_steps");
        let roped = ints(g, "layer_is_roped").to_vec();
        for t in 0..=steps {
            for l in 0..layers {
                // **NoPE layers have no `pre_rope` capture at all** — they skip the rotation (§8)
                // and both captures are taken inside its wrapper. Reading the name unconditionally
                // is how this first ran, and `float()` panicked, which is the gate working.
                if roped[l] == 0 {
                    skipped += 1;
                    continue;
                }
                let (_, qn) = golden_read::float(g, &format!("t{t}.L{l}.qk_norm.q"));
                let (_, qs) = golden_read::float(g, &format!("t{t}.L{l}.q.pre_rope"));
                assert_eq!(qn.len(), qs.len(), "{}: t{t}.L{l} q lengths", v.name);
                for (n, s) in qn.iter().zip(qs) {
                    // Exact zeros carry no ratio, and a zero norm output means the whole head was
                    // zero — which says nothing about the scale.
                    if *n != 0.0 {
                        let r = s / n;
                        lo = lo.min(r);
                        hi = hi.max(r);
                        pairs += 1;
                    }
                }
                let (_, kn) = golden_read::float(g, &format!("t{t}.L{l}.qk_norm.k"));
                let (_, kp) = golden_read::float(g, &format!("t{t}.L{l}.k.pre_rope"));
                assert_eq!(
                    kn, kp,
                    "{}: t{t}.L{l} K changed between the norm and the rotation — nothing may \
                     scale K (trap 3)",
                    v.name
                );
                k_elems += kn.len();
            }
        }
    });
    println!(
        "q ratio over {pairs} elements: [{lo:.7}, {hi:.7}]; {k_elems} K elements unchanged; {skipped} NoPE cases skipped"
    );
    assert!(
        (lo - want).abs() < 1e-5 && (hi - want).abs() < 1e-5,
        "Q's post-norm scale is [{lo}, {hi}], not the config's {want}"
    );
    // Both kinds of layer must occur, or the skip above is quietly covering everything.
    assert!(
        pairs > 0 && k_elems > 0 && skipped > 0,
        "{pairs} q pairs, {k_elems} k elements, {skipped} skipped — the goldens must carry both \
         roped and NoPE layers"
    );
}
