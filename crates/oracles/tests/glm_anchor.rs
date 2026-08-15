//! Fixture-integrity gate for the GLM-5.2 anchor goldens — the deviceless half of the
//! anchor: it proves the vendored bytes are the ones `glm_anchor_driver.py` wrote, that
//! their structure is the one the driver promises, and that the fixture is non-degenerate
//! enough to carry evidence. It runs with no python, no venv, no network, no device.
//!
//! `tests/glm-anchor.sh` is the regeneration path; `docs/measurement/glm-reference/anchor.md`
//! is the record.

// tests: panic-on-failure is the idiom
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/golden_read.rs"]
mod golden_read;

// `GoldenSet` through the facade, like the glimmer gate — which also keeps this preamble
// from tokenizing identically to `k3_anchor.rs`'s (jscpd found exactly that pair when
// this file was first written; module preambles are where the duplication gate bites
// every new anchor, and the facade is the honest fix rather than an exemption).
use golden_read::{GoldenSet, Vendored, fnv1a, ints};
use serde_json::Value;

/// The two salts, byte-pinned. A vendored fixture that changed by one byte cannot pass as
/// the one the record describes.
const GOLDENS: &[Vendored] = &[
    Vendored {
        name: "glm-anchor-1",
        bytes: include_bytes!("glm-anchor-1.bin"),
        len: 265_019,
        fnv: 0x78f5_be85_0546_296e,
    },
    Vendored {
        name: "glm-anchor-2",
        bytes: include_bytes!("glm-anchor-2.bin"),
        len: 265_019,
        fnv: 0xd43e_3d2d_8b0d_6601,
    },
];

fn read(v: &Vendored) -> GoldenSet {
    GoldenSet::read_glm(&mut &v.bytes[..]).expect(v.name)
}

fn cfg(g: &GoldenSet) -> Value {
    let raw = g.meta_get("config").expect("config meta");
    serde_json::from_str(raw).expect("config json")
}

fn num(c: &Value, k: &str) -> usize {
    usize::try_from(c[k].as_u64().unwrap_or_else(|| panic!("{k} missing"))).unwrap()
}

#[test]
fn the_bytes_are_the_pinned_bytes() {
    golden_read::check_pinned_bytes(GOLDENS);
}

#[test]
fn a_wrong_magic_is_refused_by_name() {
    // A Glimmer golden in a GLM slot must refuse at the magic, not at a shape three
    // gates downstream.
    let glimmer = include_bytes!("glimmer-anchor-text-1.bin");
    let Err(err) = GoldenSet::read_glm(&mut &glimmer[..]) else {
        panic!("a Glimmer golden parsed as a GLM one");
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("RIVGMGLD") && msg.contains("RIVGLGLD"),
        "refusal must name both the wanted and the found magic: {msg}"
    );
}

#[test]
fn the_tiny_config_is_the_declared_one_and_non_degenerate() {
    for v in GOLDENS {
        let g = read(v);
        let c = cfg(&g);
        // The structural claims the driver makes, read back rather than trusted.
        assert_eq!(num(&c, "num_hidden_layers"), 6);
        assert_eq!(num(&c, "first_k_dense_replace"), 2);
        assert_eq!(num(&c, "n_routed_experts"), 10);
        assert_eq!(num(&c, "num_experts_per_tok"), 3);
        // Non-degeneracy (the fixture-geometry lesson): every width distinct, so no two
        // candidate interpretations of a transposed pair are the same number.
        let widths = [
            num(&c, "hidden_size"),
            num(&c, "intermediate_size"),
            num(&c, "moe_intermediate_size"),
            num(&c, "kv_lora_rank"),
            num(&c, "q_lora_rank"),
            num(&c, "qk_rope_head_dim"),
            num(&c, "qk_nope_head_dim"),
            num(&c, "v_head_dim"),
            num(&c, "index_head_dim"),
            num(&c, "index_n_heads"),
            num(&c, "num_attention_heads"),
        ];
        let mut sorted = widths.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            widths.len(),
            "{}: widths collide: {widths:?}",
            v.name
        );
        // qk_head (nope+rope) must not collide with kv_lora_rank either — the pair a
        // latent/key transposition would conflate.
        let qk_head = num(&c, "qk_nope_head_dim") + num(&c, "qk_rope_head_dim");
        assert_ne!(qk_head, num(&c, "kv_lora_rank"), "{}", v.name);
        // The DSA selection must be REAL: prompt strictly longer than index_topk, or the
        // whole run rides the dense fast path and the sparse goldens are vacuous.
        let prompt = ints(&g, "prompt.ids").len();
        assert!(
            prompt > num(&c, "index_topk"),
            "{}: prompt {prompt} <= index_topk — selection never happened",
            v.name
        );
    }
}

#[test]
fn the_capture_census_is_the_derived_one() {
    // Counts derived from the structure ints the golden itself carries — never constants.
    for v in GOLDENS {
        let g = read(v);
        let sparse: Vec<i64> = ints(&g, "structure.mlp_is_sparse").to_vec();
        let full: Vec<i64> = ints(&g, "structure.indexer_is_full").to_vec();
        let layers = sparse.len();
        assert_eq!(layers, full.len());
        let steps: usize = 1 + g
            .meta_get("decode_steps")
            .expect("decode_steps metadata")
            .parse::<usize>()
            .expect("numeric decode_steps"); // prefill + the golden's own DECODE_STEPS
        let n_sparse: i64 = sparse.iter().sum();
        let n_full: i64 = full.iter().sum();
        // Alternation is a driver promise (pattern FSFSFS), and the sharing mechanism is
        // only exercised if some layers actually share.
        assert!(
            n_full >= 2 && n_full < layers as i64,
            "{}: {full:?}",
            v.name
        );

        let floats = g.floats.len();
        let ints_n = g.ints.len();
        // Per step: per layer {attn.out, q_resid, kv_latent, norm.in, norm.post_attn,
        // attend.q, attend.out, attend.mask_last_row} = 8 plus the attention rope PAIR
        // (q and k are two captures) = 10 per layer; + the index rope pair on full
        // layers, + router {logits, weights} + {moe, shared, experts}.out on sparse
        // layers, + mlp.out on dense layers, + logits. (First derivation counted the
        // rope pair as one and the gate refused 623 != 7*83 — the census doing its job
        // against its own author.)
        let per_step = layers * 10
            + 2 * usize::try_from(n_full).unwrap()
            + 5 * usize::try_from(n_sparse).unwrap()
            + (layers - usize::try_from(n_sparse).unwrap())
            + 1;
        assert_eq!(
            floats,
            steps * per_step,
            "{}: float census: {floats} != {steps}*{per_step}",
            v.name
        );
        // Ints: prompt + 2 structure + emitted + per-step (topk_indices per layer +
        // router.topk_last per sparse layer).
        let per_step_ints = layers + usize::try_from(n_sparse).unwrap();
        assert_eq!(ints_n, 4 + steps * per_step_ints, "{}: int census", v.name);
    }
}

#[test]
fn the_captures_are_finite_and_alive() {
    for v in GOLDENS {
        let g = read(v);
        let mut checked = 0usize;
        for (name, _shape, vals) in &g.floats {
            assert!(
                vals.iter().all(|x| x.is_finite()),
                "{}: {name} carries a non-finite value",
                v.name
            );
            checked += 1;
        }
        assert!(
            checked > 600,
            "{}: only {checked} float tensors seen",
            v.name
        );
        // Non-degeneracy of the values themselves: the last-step logits must not be
        // constant (a broken tap or a zeroed model both produce flat rows that every
        // tolerance passes).
        let (_, logits) = golden_read::float(&g, "t6.logits");
        let (lo, hi) = logits
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        assert!(hi > lo, "{}: t6.logits is constant", v.name);
    }
}

#[test]
fn the_selection_shapes_say_dsa_happened() {
    for v in GOLDENS {
        let g = read(v);
        let c = cfg(&g);
        let topk = num(&c, "index_topk");
        let prompt = ints(&g, "prompt.ids").len();
        // At the last prefill position the selection is exactly index_topk wide; on the
        // first decode step the context is prompt+1 and still clamps to index_topk.
        for l in 0..num(&c, "num_hidden_layers") {
            let sel = ints(&g, &format!("t0.L{l}.topk_indices"));
            assert_eq!(sel.len(), topk, "{}: t0.L{l} selection width", v.name);
            // Every selected index addresses a real position.
            assert!(
                sel.iter().all(|&i| (i as usize) < prompt),
                "{}: t0.L{l} selects out of range",
                v.name
            );
        }
        // Shared layers carry the SAME selection as their full predecessor at the last
        // row — the cross-layer mechanism, visible in the ints.
        let full: Vec<i64> = ints(&g, "structure.indexer_is_full").to_vec();
        for l in 1..full.len() {
            if full[l] == 0 {
                let prev = ints(&g, &format!("t0.L{}.topk_indices", l - 1));
                let here = ints(&g, &format!("t0.L{l}.topk_indices"));
                assert_eq!(
                    prev,
                    here,
                    "{}: L{l} does not share L{}'s selection",
                    v.name,
                    l - 1
                );
            }
        }
    }
}

#[test]
fn the_environment_is_recorded() {
    for v in GOLDENS {
        let g = read(v);
        for k in [
            "python",
            "torch",
            "transformers",
            "prompt_len",
            "decode_steps",
        ] {
            assert!(
                g.meta_get(k).is_some(),
                "{}: metadata lacks {k} — the regeneration pin is incomplete",
                v.name
            );
        }
        assert_eq!(g.meta_get("defect"), Some("None"));
    }
}
