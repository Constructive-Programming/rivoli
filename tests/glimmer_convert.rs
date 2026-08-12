//! `convert_glimmer` end to end, on a synthetic checkpoint.
//!
//! **Why synthetic rather than a slice of the real one.** The Muse Glimmer checkpoint is
//! 59.553 GB and is not on this machine; `convert_k3`'s equivalent gate fetches one expert by
//! HTTP Range because K3's unit of work *is* one expert. This converter's unit of work is the
//! whole tensor set — which tensors are copied, which are widened, which are skipped, and
//! whether the artifact re-opens as the same model — and none of that is testable on a slice.
//! A four-layer model exercises every branch.
//!
//! The fixture itself lives in `tests/common` and is shared with `tests/glimmer_pin.rs`, which
//! consumes the artifact this test asserts about. Deliberately one fixture: a pin test on a
//! differently-built checkpoint would prove nothing about what the converter produces.
//!
//! What this does NOT establish: anything about the real checkpoint's tensor *names*. The
//! completeness check in the converter is written from the shard headers recorded in
//! `docs/reference/glimmer-architecture.md` §1, and the fixture is built from the same list —
//! so a name wrong in both is wrong in both. `tests/glimmer_names.rs` closes that gap against
//! the shipped index, and its red-proof (2026-08-11) showed this test staying green while a
//! mis-transliterated name failed there twice.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod common;
use common::{
    GLIMMER_FIXTURE_DIM as DIM, GLIMMER_FIXTURE_LAYERS as L, TempRoot, glimmer_convert_fixture,
    glimmer_fixture, run_convert_glimmer, write_index,
};
// Module alias rather than a second flat `use` list: the converter imports the same names
// from the same two modules, and jscpd (which normalizes identifiers) reports the matching
// import blocks as a clone. Aliasing here is the smaller change and reads fine in a test.
use rivoli::artifact::format::{Dtype, FormatMeta, Safetensors};
use rivoli::artifact::model as gm;

#[test]
fn convert_glimmer_writes_a_bf16_artifact_that_reopens_as_the_same_model() {
    let root = TempRoot::new("glimmer-conv");
    let out = root.join("out");
    let (tensors, log) = glimmer_convert_fixture(root.path(), DIM);

    // The counts are the observation that the vision half was excluded, rather than the
    // assumption — 3 skipped, and the 4 norms per layer plus the model-level one widened.
    assert!(log.contains("3 vision tensors skipped"), "{log}");
    assert!(
        log.contains(&format!("{} norms widened", L * 4 + 1)),
        "{log}"
    );

    // It re-opens as the same model: the manifest still carries the wrapper and its
    // text_config, so the architecture resolves and every validate check runs again.
    let cfg: gm::GlimmerConfig = gm::load_config(out.to_str().unwrap()).unwrap();
    assert_eq!(cfg.text.n_layers, L);
    assert_eq!(cfg.text.layer_types.len(), L);
    FormatMeta::load(out.to_str().unwrap()).unwrap();

    let art = Safetensors::open_file(out.join("resident.safetensors").to_str().unwrap()).unwrap();
    for (name, shape, bytes) in &tensors {
        if name.starts_with("model.vision") {
            assert!(
                art.raw(name).is_err(),
                "{name} is vision and must not be in the artifact"
            );
            continue;
        }
        if name.ends_with("norm.weight") {
            // Widened, and widened CORRECTLY — not merely present at the right length. A
            // byte-length check alone passes on a zeroed tensor.
            let (got, got_shape) = art.typed(name, Dtype::F32).unwrap();
            assert_eq!(got_shape, &shape[..]);
            let want: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| rivoli::math::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let got: Vec<f32> = got
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(got, want, "{name} widened to the wrong values");
        } else {
            let (got, got_shape) = art.typed(name, Dtype::Bf16).unwrap();
            assert_eq!(got_shape, &shape[..], "{name} shape");
            assert_eq!(got, &bytes[..], "{name} is not byte-identical");
        }
    }
}

/// The guards that fire before 55 GB is written: an incomplete checkpoint, and an output
/// directory that is the input.
#[test]
fn convert_glimmer_refuses_before_it_writes() {
    let root = TempRoot::new("glimmer-refuse");
    let (src, out) = (root.join("src"), root.join("out"));
    let mut tensors = glimmer_fixture(&src, DIM);

    // Writing into the source directory is a SIGBUS risk, not an error — the writer maps the
    // shards while it writes. Refused by path identity, so `src/.` must refuse too.
    let o = run_convert_glimmer(&src, &src.join("."));
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(!o.status.success() && err.contains("SIGBUS"), "{err}");

    // A source missing a REQUIRED_AUX file refuses too, and refuses EARLY — before the config
    // is even parsed. `finish_artifact` would only have warned, three hours in, and the
    // artifact would ship with trap 13's scalar EOS as its only one.
    std::fs::remove_file(src.join("generation_config.json")).unwrap();
    let o = run_convert_glimmer(&src, &out);
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(
        !o.status.success() && err.contains("generation_config.json is missing"),
        "{err}"
    );
    std::fs::write(src.join("generation_config.json"), b"{}").unwrap();

    // A checkpoint missing one per-layer tensor refuses by NAME, before the write.
    let dropped = format!("{}.2.mlp.up_proj.weight", gm::GLIMMER_LAYER_PREFIX);
    tensors.retain(|(n, _, _)| *n != dropped);
    common::write_safetensors(&src.join("model-00001-of-00001.safetensors"), &tensors);
    write_index(&src, &tensors);
    let o = run_convert_glimmer(&src, &out);
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(!o.status.success() && err.contains(&dropped), "{err}");
    assert!(
        !out.join("resident.safetensors").exists(),
        "the artifact must not exist after a refusal"
    );
}

/// **The fixture's own value generator, which nothing else checks and which had three defects.**
///
/// `bf16_blob` feeds every Glimmer fixture in the tree, and until 2026-08-12 its contract ("distinct
/// per seed", finite, non-zero) was enforced by nothing — so it overflowed on the dev profile past
/// 9,363 indices, emitted NaN for one value in sixteen, and then, in the fix for that, collapsed to
/// a period of 1,024. Each was invisible at `GLIMMER_FIXTURE_DIM` = 8 and each broke a gate one
/// width up. This runs at a width no fixture uses, deliberately, and it needs no device.
#[test]
fn the_fixture_generator_produces_finite_signed_values_with_no_short_period() {
    // 131,072 = the element count of a [512, 256] weight, the first size that reached all three
    // defects. Well past the 9,363 that used to panic.
    const N: usize = 131_072;
    const K: usize = 256;
    let bytes = common::bf16_blob(1, N);
    assert_eq!(bytes.len(), N * 2, "two bytes per bf16 value");
    let words: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let vals: Vec<f32> = words
        .iter()
        .map(|w| f32::from_bits((*w as u32) << 16))
        .collect();

    assert!(
        vals.iter().all(|v| v.is_finite() && *v != 0.0),
        "a non-finite or zero fixture value makes every comparison against it meaningless"
    );
    assert!(
        vals.iter().any(|v| *v < 0.0) && vals.iter().any(|v| *v > 0.0),
        "both signs must occur, or the fixture cannot catch a sign or abs defect"
    );
    // The period, at the stride that matters: a weight is read row-major at `K`, and two equal rows
    // let a kernel read the wrong one and pass bit-identically. Checked as ROWS rather than as a
    // distinct-value count, because it is row equality the readers can be fooled by — the masked
    // version had 1,024 distinct values AND only four distinct rows.
    let rows: std::collections::HashSet<&[u16]> = words.chunks_exact(K).collect();
    assert_eq!(
        rows.len(),
        N / K,
        "the generator repeats a whole {K}-wide row, so a strided misread would pass"
    );
    // Distinctness across seeds, its other stated contract.
    assert_ne!(
        common::bf16_blob(1, K),
        common::bf16_blob(2, K),
        "two seeds produced identical bytes"
    );
}
