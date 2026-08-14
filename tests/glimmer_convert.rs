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
    glimmer_fixture, glimmer_fixture_eos, run_convert_glimmer, write_glimmer_eos, write_index,
};
// Module alias rather than a second flat `use` list: the converter imports the same names
// from the same two modules, and jscpd (which normalizes identifiers) reports the matching
// import blocks as a clone. Aliasing here is the smaller change and reads fine in a test.
use rivoli::artifact::format::{Dtype, FormatMeta, Safetensors};
use rivoli::artifact::model as gm;

/// The one file that carries Muse Glimmer's stop tokens. Spelled once because two tests name it.
const GEN: &str = "generation_config.json";

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
    let o = run_convert_glimmer(&src, &src.join("."), gm::GlimmerFormat::Bf16);
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(!o.status.success() && err.contains("SIGBUS"), "{err}");

    // A source missing a REQUIRED_AUX file refuses too, and refuses EARLY — before the config
    // is even parsed. `finish_artifact` would only have warned, three hours in, and the
    // artifact would ship with trap 13's scalar EOS as its only one.
    std::fs::remove_file(src.join(GEN)).unwrap();
    let o = run_convert_glimmer(&src, &out, gm::GlimmerFormat::Bf16);
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(
        !o.status.success() && err.contains("generation_config.json is missing"),
        "{err}"
    );
    // Restored with USABLE ids, not `{}`. `{}` is a file that exists and says nothing, which
    // `eos_ids` now refuses on its own — restoring it that way would mask every assertion below
    // behind the EOS refusal and this test would still be green.
    write_glimmer_eos(&src, &glimmer_fixture_eos(DIM));

    // A checkpoint missing one per-layer tensor refuses by NAME, before the write.
    let dropped = format!("{}.2.mlp.up_proj.weight", gm::GLIMMER_LAYER_PREFIX);
    tensors.retain(|(n, _, _)| *n != dropped);
    common::write_safetensors(&src.join("model-00001-of-00001.safetensors"), &tensors);
    write_index(&src, &tensors);
    let o = run_convert_glimmer(&src, &out, gm::GlimmerFormat::Bf16);
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(!o.status.success() && err.contains(&dropped), "{err}");
    assert!(
        !out.join("resident.safetensors").exists(),
        "the artifact must not exist after a refusal"
    );
}

/// **S3 item 5, the EOS clause: both ids reach the artifact, and a file that exists but says
/// nothing is refused.**
///
/// The plan states this as "two EOS ids (`[200001, 200008]` — a scalar-EOS port stops on one)".
/// The engine half of that was already safe before this test: `Tokenizer` holds `eos: Vec<u32>`,
/// `load_eos` reads both the array and the bare-int spellings, and `gpu.rs` stops on
/// `eos.contains(&t)` — plural throughout, shared with every model here.
///
/// **What was NOT safe is one step worse than the trap the plan names.** `REQUIRED_AUX` checks that
/// `generation_config.json` EXISTS. This tree's own fixture wrote it as `{}`, which passes that
/// check, copies into the artifact, and yields **zero** stop tokens — so the port does not stop on
/// one of the two, it stops on none, announced by a single `warn!` at load. The signature is the
/// one behind `docs/measurement/benchmarks.md`'s retraction: 56 runs, not one terminating
/// naturally, every one to its token limit.
///
/// Three arms, and the two refusals are the red proof for the first: without them "the ids reached
/// the artifact" is satisfied by any converter that copies a file.
#[test]
fn both_eos_ids_reach_the_artifact_and_an_empty_generation_config_is_refused() {
    let root = TempRoot::new("glimmer-eos");
    let (src, out) = (root.join("src"), root.join("out"));
    glimmer_fixture(&src, DIM);

    // **A DISTINCT pair, written here rather than taken from the fixture's default**, so the
    // assertion proves the artifact TRACKED the source. Against the fixture's own ids it would be
    // satisfied by a converter that emitted that constant from anywhere.
    //
    // Compared as BYTES, not as a parsed id list: the copy is `std::fs::copy`, so byte equality is
    // the property, and a parse-and-compare would pass a reordering or an added key. It is also
    // the third parser of this field in the tree if written out (review, 2026-08-13).
    let ids = [(DIM + 4 - 3) as u32, (DIM + 4 - 1) as u32];
    write_glimmer_eos(&src, &ids);
    let want = std::fs::read(src.join(GEN)).unwrap();
    let o = run_convert_glimmer(&src, &out, gm::GlimmerFormat::Bf16);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        std::fs::read(out.join(GEN)).expect("generation_config in the artifact"),
        want,
        "the artifact's stop tokens are not the checkpoint's — a decode built on this stops on \
         the wrong set, or on nothing"
    );

    // An id past the vocabulary is refused: it is a stop token no argmax can return, which is the
    // same unstoppable decode as having none. This is why the fixture's ids scale with its width.
    write_glimmer_eos(&src, &[(DIM + 4) as u32]);
    let o = run_convert_glimmer(&src, &root.join("out-vocab"), gm::GlimmerFormat::Bf16);
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(
        !o.status.success() && err.contains("past this model's vocabulary"),
        "{err}"
    );

    // Red proof, and the case that was live: a file that satisfies the presence check and carries
    // no ids. `{}` first, then the shape a hand-edit produces — the key present and empty.
    for (bytes, what) in [
        (&b"{}"[..], "an empty object"),
        (br#"{"eos_token_id": []}"#, "an empty array"),
        (br#"{"eos_token_id": null}"#, "a null"),
    ] {
        let dst = root.join("out-red");
        std::fs::write(src.join(GEN), bytes).unwrap();
        let o = run_convert_glimmer(&src, &dst, gm::GlimmerFormat::Bf16);
        let err = String::from_utf8_lossy(&o.stderr).to_string();
        assert!(
            !o.status.success() && err.contains("no usable `eos_token_id`"),
            "{what} was accepted, so the artifact ships with no stop token: {err}"
        );
        // Refused BEFORE any tensor is read, and before `create_dir_all` — the whole argument for
        // checking here rather than at load is that a three-hour convert must not end in this.
        // The DIRECTORY, not the artifact inside it: the converter creates it at the point the
        // check has already passed, so its absence is the stronger statement and it catches the
        // check being moved one line later. Inside the loop so all three arms are covered rather
        // than only the last (review, 2026-08-13).
        assert!(
            !dst.exists(),
            "{what}: the converter got past the EOS check"
        );
    }
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
