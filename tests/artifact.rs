//! Integration check: an artifact produced by `convert` reads back through the
//! loader (format.rs) — the converter↔loader byte contract, end to end. Runs only
//! when RIVOLI_ARTIFACT points to an artifact dir (a `--layers N` conversion), else
//! it skips, so CI without a checkpoint stays green.
#![allow(clippy::unwrap_used)]

use rivoli::format::{Dtype, FormatMeta, ExpertSet, Safetensors, load_codebooks};
use rivoli::model::ModelConfig;
use rivoli::quant::{VQ_ALIGN, VQ_DIM, VQ_K, vq_expert_bytes};

#[test]
fn artifact_reads_back() {
    let Ok(dir) = std::env::var("RIVOLI_ARTIFACT") else {
        eprintln!("skip: set RIVOLI_ARTIFACT to an artifact dir");
        return;
    };
    // manifest: format section + config fields
    let meta = FormatMeta::load(&dir).unwrap();
    assert_eq!((meta.vq_dim, meta.vq_k, meta.vq_group), (VQ_DIM, VQ_K, 64));
    let cfg = ModelConfig::load(&dir).unwrap();

    // 3 per-projection codebooks
    let cbs = load_codebooks(&dir).unwrap();
    for cb in &cbs {
        assert_eq!(cb.len(), VQ_K * VQ_DIM);
    }

    // resident.safetensors: embed int8 with the right shape, attention fp8 present
    let res = Safetensors::open_file(&format!("{dir}/resident.safetensors")).unwrap();
    let (_, sh) = res.typed("model.embed_tokens.weight", Dtype::I8).unwrap();
    assert_eq!(sh, &[cfg.vocab, cfg.hidden]);
    assert!(res.has("model.embed_tokens.weight.scale"));
    res.typed("model.layers.3.self_attn.q_a_proj.weight", Dtype::F8E4M3)
        .unwrap();
    res.typed("model.layers.3.mlp.gate.weight", Dtype::F32)
        .unwrap(); // widened

    // .vq3: headers validate against config; routed reads aligned; shared block sized
    let n_present = std::fs::read_dir(&dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|x| x == "vq3")
        })
        .count();
    let last = cfg.dense_layers + n_present;
    let vq = ExpertSet::open_vq3(
        &dir,
        cfg.dense_layers,
        last,
        cfg.n_experts,
        cfg.hidden,
        cfg.moe_inter,
    )
    .unwrap();
    let (_, begin, len) = vq.read_spec(cfg.dense_layers, 0).unwrap();
    assert_eq!(begin % VQ_ALIGN, 0, "routed read must be O_DIRECT aligned");
    assert_eq!(len, vq_expert_bytes(cfg.hidden, cfg.moe_inter));
    let shared = vq.shared_block(cfg.dense_layers).unwrap();
    assert_eq!(shared.len(), vq_expert_bytes(cfg.hidden, cfg.moe_inter));
    eprintln!(
        "artifact OK: {n_present} layers, {} experts + shared",
        cfg.n_experts
    );
}
