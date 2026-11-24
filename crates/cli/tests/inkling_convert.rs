//! `convert_inkling`'s output gate (S2): the artifact's tensor names and shapes are the FIRST-PARTY
//! text model's state dict (`text-model-names.tsv`, vendored from a meta-device
//! `Inkling2LoopForCausalLM` at transformers @ d8f10a82), not a name list this port believes in.
//! The rename + fusion (176 expert tensors → 48 fused projections, 50 query tensors → 49, unembed
//! → unpadded lm_head) is correct exactly when the emitted set equals that census, both ends.
//!
//! **One deliberate shape divergence from the transformers state dict**:
//! `relative_position_bias.proj` is stored TRANSPOSED here as `[1024, 16]` (transformers ships
//! `[16, 1024]`). The engine's rel_logits GEMM is `gemm_bf16`, which computes `x·wᵀ`; storing the
//! projection already transposed is what lets that GEMM read it without a runtime transpose. The
//! census cell below therefore expects the transposed shape — it is the artifact's contract, not
//! the checkpoint's, and the converter's `emit_verbatim` performs the transpose (with a shape
//! guard) at convert time.
//!
//! The live cell reads the artifact produced by the documented run
//! (`/var/cache/rivoli/scratch/inkling-mini-artifact`); it skips unless `RIVOLI_INKLING_REQUIRED`
//! is set, on `inkling_pins.rs`'s policy. The converter's own runtime guards (the census confront
//! and `consumed == 822`) are the other half. No GPU, no network.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::census::inkling::collapse;
use rivoli_artifact::format::Safetensors;
use std::collections::{HashMap, HashSet};

const NAMES: &str =
    include_str!("../../../docs/measurement/inkling2-reference/mini/text-model-names.tsv");
const ARTIFACT: &str = "/var/cache/rivoli/scratch/inkling-mini-artifact/resident.safetensors";

/// The first-party name → shape census. The dtype column is the meta-device default (f32) and is
/// NOT authoritative — the artifact stores the checkpoint's bf16/f32 — so only names and shapes
/// are compared.
fn expected() -> HashMap<String, Vec<usize>> {
    let mut m = HashMap::new();
    for line in NAMES
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split('\t').collect();
        let shape: Vec<usize> = f[1].split('x').map(|d| d.parse().unwrap()).collect();
        assert!(
            m.insert(f[0].to_string(), shape).is_none(),
            "duplicate name {}",
            f[0]
        );
    }
    m
}

/// The census itself: 693 params, 47 distinct layer-collapsed patterns, and the fusion shapes the
/// converter must hit (unpadded lm_head, fused gate_up/down, merged message queries).
#[test]
fn the_text_model_census_has_its_declared_shape() {
    let m = expected();
    assert_eq!(m.len(), 693);
    assert_eq!(m["lm_head.weight"], vec![200058, 2304]);
    assert_eq!(m["model.embed_tokens.weight"], vec![201024, 2304]);
    assert_eq!(m["model.message_input.query"], vec![12, 4, 2304]);
    assert_eq!(
        m["model.layers.0.mlp.experts.gate_up_proj"],
        vec![128, 2304, 2304]
    );
    assert_eq!(
        m["model.layers.0.mlp.experts.down_proj"],
        vec![128, 2304, 1152]
    );
    assert_eq!(
        m["model.layers.23.mlp.experts.gate_up_proj"],
        vec![128, 2304, 2304]
    );
    let pats: HashSet<String> = m.keys().map(|k| collapse(k)).collect();
    assert_eq!(pats.len(), 47);
}

/// The converted artifact's names and shapes equal the first-party census, both ends: every
/// expected name is present with the right shape, and no unexpected name rides along.
#[test]
fn the_converted_artifact_matches_the_first_party_names() {
    if std::fs::metadata(ARTIFACT).is_err() {
        assert!(
            std::env::var_os("RIVOLI_INKLING_REQUIRED").is_none(),
            "Inkling convert gate REQUIRED but did not run: {ARTIFACT} absent"
        );
        eprintln!(
            "skip live artifact name gate: {ARTIFACT} absent (set RIVOLI_INKLING_REQUIRED=1)"
        );
        return;
    }
    let exp = expected();
    let src = Safetensors::open_file(ARTIFACT).unwrap();
    let got: HashMap<String, Vec<usize>> = src
        .names()
        .iter()
        .map(|n| (n.to_string(), src.shape(n).unwrap().to_vec()))
        .collect();
    for (name, shape) in &exp {
        let g = got
            .get(name)
            .unwrap_or_else(|| panic!("artifact missing {name}"));
        assert_eq!(
            g, shape,
            "{name}: artifact shape {g:?} != first-party {shape:?}"
        );
    }
    for name in got.keys() {
        assert!(
            exp.contains_key(name),
            "artifact carries unexpected name {name}"
        );
    }
    assert_eq!(got.len(), exp.len(), "artifact tensor count");
}
