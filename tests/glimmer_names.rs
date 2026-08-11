//! Muse Glimmer's tensor names, pinned against the checkpoint's own index.
//!
//! **Why this file exists.** `tests/glimmer_convert.rs` builds its fixture from the same
//! constant the converter checks, so a name wrong in both is wrong in both and that test
//! stays green — it says so in its own header. This closes that gap from the only side that
//! can: the shipped `model.safetensors.index.json`, reduced to families and vendored at
//! `docs/measurement/glimmer-reference/tensor-families.tsv`.
//!
//! The failure mode is worse than a wrong config key. A key that does not exist refuses
//! loudly on `missing field`; a *tensor* name that does not exist looks like a corrupt shard,
//! and a name that exists but points at the wrong tensor is copied silently into an artifact
//! that then decodes fluent wrong text. `tests/k3_names.rs` is the same instrument for K3 and
//! records the round that cost.
//!
//! The TSV is data: `#`-comments, then `count \t dtype \t shape \t family`, where a family is
//! a name with `.layers.<n>.` collapsed to `.layers.{L}.`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod common;
use common::{GLIMMER_SHIPPED_CONFIG, TsvFamily, tsv_families};
use rivoli::artifact::model as gm;

const FAMILIES: &str = include_str!("../docs/measurement/glimmer-reference/tensor-families.tsv");

fn families() -> Vec<TsvFamily> {
    tsv_families(FAMILIES)
}

/// Every family the port places: the twelve per-layer tensors plus the three model-level ones.
fn implemented_families() -> Vec<String> {
    gm::GLIMMER_LAYER_TENSORS
        .iter()
        .map(|t| format!("{}.{{L}}.{}.weight", gm::GLIMMER_LAYER_PREFIX, t))
        .chain([
            "lm_head.weight".to_string(),
            "model.language_model.embed_tokens.weight".to_string(),
            "model.language_model.norm.weight".to_string(),
        ])
        .collect()
}

/// The shipped config, at full dims — so the shapes checked below are the model's, not a
/// fixture's.
fn shipped() -> gm::GlimmerTextConfig {
    serde_json::from_str::<gm::GlimmerConfig>(GLIMMER_SHIPPED_CONFIG)
        .expect("the vendored config parses")
        .text
}

/// The vendored reduction is of the real checkpoint, and says so in numbers the header can be
/// checked against. If this fails, the TSV was regenerated from something else.
#[test]
fn the_reduction_is_of_the_shipped_checkpoint() {
    let fams = families();
    assert_eq!(fams.len(), 40, "family count");
    assert_eq!(
        fams.iter().map(|f| f.count).sum::<usize>(),
        1436,
        "total tensors"
    );
    // Every tensor in this checkpoint is BF16 — the fact `GlimmerConfig::validate` asserts
    // from `dtype`, here confirmed tensor by tensor rather than from the config's word.
    assert!(fams.iter().all(|f| f.dtype == "BF16"), "not all BF16");
    // 59.553 GB, reconciled against the index's own metadata.total_size in the header.
    let bytes: usize = fams
        .iter()
        .map(|f| f.count * f.shape.iter().product::<usize>() * 2)
        .sum();
    assert_eq!(bytes, 59_553_253_376, "summed bytes vs metadata.total_size");
}

/// **Every name the converter and the pin will ask for exists, with the shape the config
/// implies.** This is the half that catches a transliteration error.
///
/// It is also what validates [`gm::GlimmerTextConfig::layer_tensor_shape`], the one table the
/// pin checks its placements against and `resident_bytes` sizes the tier from. That table is
/// arithmetic over config keys; here it meets the shipped `model.safetensors.index.json`, so
/// it is confronted with the checkpoint rather than believed.
#[test]
fn the_layer_tensor_names_match_the_checkpoint() {
    let fams = families();
    let cfg = shipped();
    let by_name = |n: &str| fams.iter().find(|f| f.name == n);
    let (hidden, vocab) = (cfg.hidden, cfg.vocab);

    for t in gm::GLIMMER_LAYER_TENSORS {
        let n = format!("{}.{{L}}.{}.weight", gm::GLIMMER_LAYER_PREFIX, t);
        let f = by_name(&n).unwrap_or_else(|| {
            panic!("{n} is not a family in the checkpoint — the name is wrong, not the model")
        });
        assert_eq!(f.count, cfg.n_layers, "{n}: one per layer");
        // Asserted per family rather than as a total, because a q/gate mix-up (both
        // [4096, 6656]) and a k/v mix-up (both [256, 6656]) are the two that a byte total
        // cannot see — those pairs are distinguishable only by NAME, which is the loop key.
        assert_eq!(f.shape, cfg.layer_tensor_shape(t).unwrap(), "{n} shape");
    }

    for (n, want) in [
        ("lm_head.weight", vec![vocab, hidden]),
        (
            "model.language_model.embed_tokens.weight",
            vec![vocab, hidden],
        ),
        ("model.language_model.norm.weight", vec![hidden]),
    ] {
        let f = by_name(n).unwrap_or_else(|| panic!("{n} is not in the checkpoint"));
        assert_eq!((f.count, &f.shape), (1, &want), "{n}");
    }

    // **Untied, from the weights rather than from the config's word.** `tie_word_embeddings`
    // is false and `validate` asserts it, but the class the model comes from declares a tied
    // mapping — so the evidence that matters is that both tensors ship, 2.690 GB each.
    assert_eq!(
        by_name("lm_head.weight").unwrap().shape,
        by_name("model.language_model.embed_tokens.weight")
            .unwrap()
            .shape
    );
}

/// **Nothing in the checkpoint is unexplained** — the property K3's G0 item 10 established
/// for its own index, and the one that makes "we skip the vision half" a measurement.
///
/// Every family is either a text tensor this port implements or one `convert_glimmer`'s
/// `is_vision` predicate skips. A family matching neither is a tensor nobody has decided
/// about, which is exactly how a checkpoint feature gets silently dropped.
#[test]
fn every_family_is_either_implemented_or_deliberately_skipped() {
    // `is_vision`'s three prefixes, restated. Not imported: it lives in a binary.
    //
    // > **CORRECTED 2026-08-11**, by review. This said "if these drift from
    // > `convert_glimmer.rs`, the counts below stop adding up". They do not: `text` and
    // > `vision` are computed entirely from THIS closure, the vendored TSV and
    // > `GLIMMER_LAYER_TENSORS` — the converter contributes nothing to either, so deleting its
    // > `model.vision_projection` clause leaves 627/809 bit-identical. What actually catches
    // > that drift is `tests/glimmer_convert.rs`, whose fixture ships one tensor per prefix.
    // > The numbers below are real and are the completeness claim; the coupling this comment
    // > asserted was not.
    let is_vision = |n: &str| {
        n.starts_with("model.vision_tower.")
            || n.starts_with("model.vision_adapter.")
            || n.starts_with("model.vision_projection")
    };
    let implemented = implemented_families();

    let (mut text, mut vision) = (0usize, 0usize);
    for f in families() {
        if is_vision(&f.name) {
            vision += f.count;
        } else if implemented.contains(&f.name) {
            text += f.count;
        } else {
            panic!(
                "{} ({} tensors) is neither implemented nor skipped — decide about it before \
                 it is silently dropped",
                f.name, f.count
            );
        }
    }
    assert_eq!(text, 627, "text-side tensors");
    assert_eq!(vision, 809, "vision-side tensors");
    assert_eq!(text + vision, 1436);
}

/// **The resident byte accounting, reproduced from the checkpoint's own shapes** — G1a's
/// fourth clause, and it needs no device.
///
/// `GlimmerTextConfig::resident_bytes` sizes the tier, so an under-count is a mid-placement
/// bail on a 55.7 GB load and an over-count is device memory nothing can use. It is derived
/// from `layer_tensor_shape`, which the test above pins to this same index — so what is left
/// to establish is the *dtype* arithmetic, which the index alone decides: bf16 everywhere
/// except the norms, which `convert_glimmer` widens to f32.
///
/// The expectation is computed from the vendored families rather than written down, so the
/// 55.712 GB below is a reconciliation and not a second assertion of the same belief. It is
/// 2.782 MB above the checkpoint's own text half — 209 norms x 6656 x 2 extra bytes — and
/// that difference is exactly what the `shape.len() == 1` arm accounts for.
#[test]
fn the_resident_byte_accounting_reproduces_the_checkpoint() {
    let implemented = implemented_families();
    let want: usize = families()
        .iter()
        .filter(|f| implemented.contains(&f.name))
        .map(|f| {
            let n: usize = f.shape.iter().product::<usize>() * f.count;
            n * if f.shape.len() == 1 { 4 } else { 2 }
        })
        .sum();
    assert_eq!(shipped().resident_bytes().unwrap(), want);
    assert_eq!(want, 55_712_344_064, "55.712 GB resident");
}
