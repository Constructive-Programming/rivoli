//! `convert_glimmer` end to end, on a synthetic checkpoint.
//!
//! **Why synthetic rather than a slice of the real one.** The Muse Glimmer checkpoint is
//! 59.553 GB and is not on this machine; `convert_k3`'s equivalent gate fetches one expert by
//! HTTP Range because K3's unit of work *is* one expert. This converter's unit of work is the
//! whole tensor set — which tensors are copied, which are widened, which are skipped, and
//! whether the artifact re-opens as the same model — and none of that is testable on a slice.
//! A four-layer model at width 8 exercises every branch.
//!
//! What this does NOT establish: anything about the real checkpoint's tensor *names*. The
//! completeness check in the converter is written from the shard headers recorded in
//! `docs/reference/glimmer-architecture.md` §1, and this test builds its fixture from the
//! same list — so a name wrong in both is wrong in both. `tests/k3_names.rs` is the shape that
//! closes that gap, against a vendored index reduction, and Glimmer has no such reduction yet.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

// Module alias rather than a second flat `use` list: the converter imports the same names
// from the same two modules, and jscpd (which normalizes identifiers) reports the matching
// import blocks as a clone. Aliasing here is the smaller change and reads fine in a test.
use rivoli::artifact::format::{Dtype, FormatMeta, Safetensors};
use rivoli::artifact::model as gm;
use std::path::Path;

const SHIPPED: &str = include_str!("../docs/measurement/glimmer-reference/config.json");
const L: usize = 4; // one full [sliding, sliding, sliding, full] period
/// bf16 bytes for `n` values, distinct per tensor so a mixed-up copy is visible.
fn bf16_blob(seed: u16, n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (seed.wrapping_mul(37).wrapping_add(i as u16 * 7) | 0x3c00).to_le_bytes())
        .collect()
}

/// Write a minimal safetensors file: `u64` header length, header JSON, then the data block.
fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape, bytes) in tensors {
        let end = offset + bytes.len();
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let hjson = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = (hjson.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&hjson);
    for (_, _, b) in tensors {
        out.extend_from_slice(b);
    }
    std::fs::write(path, out).unwrap();
}

/// The single-shard index mapping every tensor to the one file `write_safetensors` wrote.
fn write_index(dir: &Path, tensors: &[(String, Vec<usize>, Vec<u8>)]) {
    let map: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|(n, _, _)| (n.clone(), "model-00001-of-00001.safetensors".into()))
        .collect();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap(),
    )
    .unwrap();
}

/// A four-layer Glimmer checkpoint at width 8, plus one vision tensor to be skipped.
fn build_fixture(dir: &Path) -> Vec<(String, Vec<usize>, Vec<u8>)> {
    std::fs::create_dir_all(dir).unwrap();
    // The shipped config, shrunk — the per-layer arrays truncated to L so the pairing
    // invariant still holds, since it is the thing `GlimmerConfig::validate` checks hardest.
    let mut cfg: serde_json::Value = serde_json::from_str(SHIPPED).unwrap();
    let t = cfg["text_config"].as_object_mut().unwrap();
    t["num_hidden_layers"] = serde_json::json!(L);
    t["hidden_size"] = serde_json::json!(8);
    t["intermediate_size"] = serde_json::json!(16);
    t["vocab_size"] = serde_json::json!(12);
    t["num_attention_heads"] = serde_json::json!(2);
    t["num_key_value_heads"] = serde_json::json!(1);
    t["head_dim"] = serde_json::json!(4);
    t["sliding_window"] = serde_json::json!(2);
    t["layer_types"] = serde_json::json!(t["layer_types"].as_array().unwrap()[..L]);
    t["layer_rope_theta"] = serde_json::json!(t["layer_rope_theta"].as_array().unwrap()[..L]);
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&cfg).unwrap(),
    )
    .unwrap();

    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut push = |name: String, n: usize| {
        let seed = tensors.len() as u16 + 1;
        tensors.push((name, vec![n], bf16_blob(seed, n)));
    };
    push("lm_head.weight".into(), 8);
    push("model.language_model.embed_tokens.weight".into(), 8);
    push("model.language_model.norm.weight".into(), 8);
    let prefix = gm::GLIMMER_LAYER_PREFIX;
    for l in 0..L {
        for name in gm::GLIMMER_LAYER_TENSORS {
            push(format!("{prefix}.{l}.{name}.weight"), 8);
        }
    }
    // The three vision families the converter must skip, one each.
    push("model.vision_tower.layers.0.attn.q_proj.weight".into(), 8);
    push("model.vision_adapter.fc1.weight".into(), 8);
    push("model.vision_projection.weight".into(), 8);

    write_safetensors(&dir.join("model-00001-of-00001.safetensors"), &tensors);
    write_index(dir, &tensors);
    for aux in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::write(dir.join(aux), b"{}").unwrap();
    }
    tensors
}

fn run_converter(src: &Path, out: &Path) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_convert_glimmer"))
        .arg(src)
        .arg(out)
        .output()
        .expect("run convert_glimmer")
}

#[test]
fn convert_glimmer_writes_a_bf16_artifact_that_reopens_as_the_same_model() {
    let root = std::env::temp_dir().join(format!("glimmer-conv-{}", std::process::id()));
    let (src, out) = (root.join("src"), root.join("out"));
    let _ = std::fs::remove_dir_all(&root);
    let tensors = build_fixture(&src);

    let o = run_converter(&src, &out);
    assert!(
        o.status.success(),
        "converter failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    // The counts are the observation that the vision half was excluded, rather than the
    // assumption — 3 skipped, and the 4 norms per layer plus the model-level one widened.
    let log = String::from_utf8_lossy(&o.stderr);
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
    for (name, _, bytes) in &tensors {
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
            let (got, shape) = art.typed(name, Dtype::F32).unwrap();
            assert_eq!(shape, [8]);
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
            let (got, _) = art.typed(name, Dtype::Bf16).unwrap();
            assert_eq!(got, &bytes[..], "{name} is not byte-identical");
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// The guards that fire before 55 GB is written: an incomplete checkpoint, and an output
/// directory that is the input.
#[test]
fn convert_glimmer_refuses_before_it_writes() {
    let root = std::env::temp_dir().join(format!("glimmer-refuse-{}", std::process::id()));
    let (src, out) = (root.join("src"), root.join("out"));
    let _ = std::fs::remove_dir_all(&root);
    let mut tensors = build_fixture(&src);

    // Writing into the source directory is a SIGBUS risk, not an error — the writer maps the
    // shards while it writes. Refused by path identity, so `src/.` must refuse too.
    let o = run_converter(&src, &src.join("."));
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(!o.status.success() && err.contains("SIGBUS"), "{err}");

    // A checkpoint missing one per-layer tensor refuses by NAME, before the write.
    let dropped = format!("model.language_model.layers.2.mlp.up_proj.weight");
    tensors.retain(|(n, _, _)| *n != dropped);
    write_safetensors(&src.join("model-00001-of-00001.safetensors"), &tensors);
    write_index(&src, &tensors);
    let o = run_converter(&src, &out);
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(!o.status.success() && err.contains(&dropped), "{err}");
    assert!(
        !out.join("resident.safetensors").exists(),
        "the artifact must not exist after a refusal"
    );
    let _ = std::fs::remove_dir_all(&root);
}
