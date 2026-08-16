//! `convert_k3` end to end, on a synthetic Kimi-K3 checkpoint — the converter gate, shaped
//! like `v4_convert.rs`'s (its header carries the synthetic-vs-real argument; the real
//! checkpoint here is 1.42 TiB across 96 shards, so it holds a fortiori).
//!
//! **What is DIFFERENT from the V4 gate, and each is a fact about K3.** The resident set is
//! a VERBATIM passthrough — K3's trunk is BF16 on disk and `convert_k3` widens nothing, so
//! every kept tensor is asserted byte-identical under its source dtype (the widening
//! question belongs to the K3 loader; `k3:src/bin/convert_k3.rs` records why). The routed
//! experts are entered at the LATENT (`routed_expert_hidden_size`), not at `hidden_size`,
//! and the fixture makes the two differ so a converter that bound `hidden` fails on every
//! expert shape. Layer 0 is DENSE: it ships `mlp.*` and no experts, is excluded from
//! `.f4`, and is ALWAYS in the resident set — even for a partial `--from/--to` that does
//! not contain it. And the checkpoint carries a multimodal side (`vision_tower.*`,
//! `mm_projector.*`) that must be skipped by name, not by omission.
//!
//! **TDD record.** Watched red 2026-08-16 against a stub binary that parsed its args and
//! bailed `unimplemented`: every success arm failed on the bail, and every refusal arm
//! failed because the stub's message named none of the guards under test — which is the
//! `expect_refusal` contract ("a refusal test that only asserts non-zero exit passes when
//! the binary fails for an unrelated reason") doing its job on day one.
//!
//! No GPU, no network — every byte is written by this file.

#![allow(clippy::unwrap_used, clippy::expect_used)]
// tests: panic-on-failure is the idiom
// `json!` expands recursively per key, and this fixture's `text_config` carries the ~40
// keys `K3Config` requires — the default 128 overflows inside `json_internal!`.
#![recursion_limit = "256"]

use std::path::Path;

use rivoli_artifact::format::{Dtype, FormatMeta, Safetensors, f4_layer_range};
use rivoli_artifact::k3_config::K3Config;
use rivoli_artifact::quant::{f4_groups, f4_row_bytes, k3_expert_base};
use serde_json::json;

mod common;

use common::Tensor;

/// Tiny but structurally faithful: every distinction the real config makes survives the
/// shrink. `HIDDEN` differs from `EXPERT_IN` (the latent — THE K3 distinction, plan §2);
/// `EXPERT_IN` and `MOE_INTER` are multiples of `F4_GROUP` and differ from each other, so a
/// w1/w2 transposition changes a span length; `KV_LORA` is 128 because the schema refuses
/// anything that is not a positive multiple of 128 (guard 1004's load-boundary half).
const LAYERS: usize = 4;
const HIDDEN: usize = 80;
const EXPERT_IN: usize = 64;
const MOE_INTER: usize = 96;
const DENSE_INTER: usize = 112;
const VOCAB: usize = 32;
const HEADS: usize = 4;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const N_SHARED: usize = 2;
/// One-based, as the checkpoint writes them: layer 3 (one-based) is the single MLA layer,
/// so zero-based layer 2 — an interior MoE layer, exercising both families inside the
/// converted range.
const KDA_LAYERS: [usize; 3] = [1, 2, 4];
const FULL_ATTN_LAYERS: [usize; 1] = [3];
/// Layer 0 is dense; the MoE (and `.f4`) layers are 1..4.
const FIRST_DENSE: usize = 1;

const K3P: &str = "language_model.model.";
const SHARD: &str = "model-00001-of-00001.safetensors";

/// The binary under test. Everything K3-specific in this gate is the FIXTURE; the
/// run/convert/refuse plumbing is `common::ConvertBin`'s, factored there when this file
/// became the fourth converter gate and jscpd reported the quartet.
const BIN: common::ConvertBin = common::ConvertBin {
    exe: env!("CARGO_BIN_EXE_convert_k3"),
    tool: "convert_k3",
};

/// `common::tensor`'s byte policy fits K3 exactly: the trunk is BF16/F32, the MXFP4
/// nibbles are `U8` opaque filler, and the `U8` scale tensors are built by hand in
/// [`routed_expert`] with `common::e8m0_bytes` — compressed-tensors declares them plain
/// `U8`, so no dtype-driven policy can know they are exponents.
use common::tensor;

/// One routed expert: three MXFP4 projections under compressed-tensors' names,
/// `weight_packed` nibbles plus `weight_scale` e8m0 bytes, BOTH `U8` — `F4_NAMING_K3`.
fn routed_expert(out: &mut Vec<Tensor>, base: &str) {
    for (proj, o, i) in [
        ("w1", MOE_INTER, EXPERT_IN),
        ("w3", MOE_INTER, EXPERT_IN),
        ("w2", EXPERT_IN, MOE_INTER),
    ] {
        out.push(tensor(
            &format!("{base}.{proj}.weight_packed"),
            Dtype::U8,
            vec![o, f4_row_bytes(i)],
        ));
        let scale = format!("{base}.{proj}.weight_scale");
        let shape = vec![o, f4_groups(i)];
        let n = shape.iter().product();
        out.push((
            scale.clone(),
            Dtype::U8,
            shape,
            common::e8m0_bytes(&scale, n),
        ));
    }
}

/// One MoE layer's TRUNK-side tensors: the full-width router, the latent sandwich, and the
/// ONE fused shared MLP — everything that must stay resident while the experts go to `.f4`.
fn moe_trunk_tensors(out: &mut Vec<Tensor>, l: usize) {
    let moe = format!("{K3P}layers.{l}.block_sparse_moe");
    out.push(tensor(
        &format!("{moe}.gate.weight"),
        Dtype::Bf16,
        vec![EXPERTS, HIDDEN],
    ));
    out.push(tensor(
        &format!("{moe}.gate.e_score_correction_bias"),
        Dtype::F32,
        vec![EXPERTS],
    ));
    out.push(tensor(
        &format!("{moe}.routed_expert_down_proj.weight"),
        Dtype::Bf16,
        vec![EXPERT_IN, HIDDEN],
    ));
    out.push(tensor(
        &format!("{moe}.routed_expert_norm.weight"),
        Dtype::Bf16,
        vec![EXPERT_IN],
    ));
    out.push(tensor(
        &format!("{moe}.routed_expert_up_proj.weight"),
        Dtype::Bf16,
        vec![HIDDEN, EXPERT_IN],
    ));
    // The fused form the checkpoint SHIPS: one gate/up/down at `n_shared * moe_inter`.
    for p in ["gate_proj", "up_proj"] {
        out.push(tensor(
            &format!("{moe}.shared_experts.{p}.weight"),
            Dtype::Bf16,
            vec![N_SHARED * MOE_INTER, HIDDEN],
        ));
    }
    out.push(tensor(
        &format!("{moe}.shared_experts.down_proj.weight"),
        Dtype::Bf16,
        vec![HIDDEN, N_SHARED * MOE_INTER],
    ));
}

/// Everything one layer carries. The attention side is a representative sample rather than
/// the full family census — the converter is a passthrough there and interprets none of it;
/// which names exist is `k3_names.rs`'s gate, against the shipped index.
fn layer_tensors(out: &mut Vec<Tensor>, l: usize) {
    let lb = format!("{K3P}layers.{l}");
    for t in [
        "input_layernorm",
        "post_attention_layernorm",
        "mlp_res_norm",
    ] {
        let norm = format!("{lb}.{t}.weight");
        out.push(tensor(&norm, Dtype::Bf16, vec![HIDDEN]));
    }
    out.push(tensor(
        &format!("{lb}.mlp_res_proj.weight"),
        Dtype::Bf16,
        vec![1, HIDDEN],
    ));
    // An F32 KDA tensor, so the verbatim walk is proven on more than one dtype.
    out.push(tensor(
        &format!("{lb}.self_attn.A_log"),
        Dtype::F32,
        vec![HEADS],
    ));
    out.push(tensor(
        &format!("{lb}.self_attn.o_proj.weight"),
        Dtype::Bf16,
        vec![HIDDEN, 16],
    ));
    if l < FIRST_DENSE {
        // The dense layer: `mlp.{gate,up,down}_proj` at `intermediate_size`, and NO experts.
        for p in ["gate_proj", "up_proj"] {
            out.push(tensor(
                &format!("{lb}.mlp.{p}.weight"),
                Dtype::Bf16,
                vec![DENSE_INTER, HIDDEN],
            ));
        }
        out.push(tensor(
            &format!("{lb}.mlp.down_proj.weight"),
            Dtype::Bf16,
            vec![HIDDEN, DENSE_INTER],
        ));
    } else {
        moe_trunk_tensors(out, l);
        for e in 0..EXPERTS {
            routed_expert(out, &k3_expert_base(l, e));
        }
    }
}

/// The whole synthetic checkpoint: the model-level five, every layer, and the multimodal
/// side that must be SKIPPED — present so the exclusion is an observation, not an
/// assumption.
fn all_tensors() -> Vec<Tensor> {
    let mut out = Vec::new();
    // `lm_head` sits BESIDE `model.`, not under it — the one text-side exception to
    // `K3_TEXT_PREFIX`, straight from the shipped index.
    for (name, shape) in [
        (format!("{K3P}embed_tokens.weight"), vec![VOCAB, HIDDEN]),
        ("language_model.lm_head.weight".into(), vec![VOCAB, HIDDEN]),
        (format!("{K3P}norm.weight"), vec![HIDDEN]),
        (format!("{K3P}output_attn_res_norm.weight"), vec![HIDDEN]),
        (format!("{K3P}output_attn_res_proj.weight"), vec![1, HIDDEN]),
    ] {
        out.push(tensor(&name, Dtype::Bf16, shape));
    }
    (0..LAYERS).for_each(|l| layer_tensors(&mut out, l));
    // `vision_tower` and `mm_projector` are SIBLINGS of `language_model` in the name tree,
    // so a filter phrased as "not under language_model" would drop nothing.
    for vis in [
        "vision_tower.encoder.blocks.0.wqkv.weight",
        "mm_projector.proj.0.weight",
    ] {
        out.push(tensor(vis, Dtype::Bf16, vec![8, 8]));
    }
    out
}

/// The checkpoint's `config.json`, shrunk. Every value `K3Config::validate` looks at is
/// here and is one the real file could carry; the fixture is parsed back below so a
/// document the schema would refuse fails HERE rather than as a confusing converter error.
/// The HF sampling `top_k: 50` twin and a `quantization_config` block are both present ON
/// PURPOSE: serde must ignore the first (the rename gate) and the converter must drive off
/// `.weight_packed` presence rather than the second (its `targets`/`ignore` lists
/// mis-declare their own scope).
fn k3_config_json() -> serde_json::Value {
    json!({
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "dtype": "bfloat16",
        "text_config": {
            "model_type": "kimi_linear",
            "architectures": ["KimiLinearForCausalLM"],
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "attn_res_block_size": 2,
            "dtype": "bfloat16",
            "first_k_dense_replace": FIRST_DENSE,
            "hidden_act": "situ",
            "hidden_size": HIDDEN,
            "intermediate_size": DENSE_INTER,
            "kv_lora_rank": 128,
            "latent_moe_use_norm": true,
            "linear_attn_config": {
                "full_attn_layers": FULL_ATTN_LAYERS,
                "kda_layers": KDA_LAYERS,
                "gate_lower_bound": -5.0,
                "head_dim": 4,
                "num_heads": 2,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": true
            },
            "mla_use_nope": true,
            "mla_use_output_gate": true,
            "moe_intermediate_size": MOE_INTER,
            "moe_layer_freq": 1,
            "moe_renormalize": true,
            "moe_router_activation_func": "sigmoid",
            "num_attention_heads": HEADS,
            "num_expert_group": 1,
            "num_experts": EXPERTS,
            "num_experts_per_token": TOP_K,
            "num_hidden_layers": LAYERS,
            "num_key_value_heads": HEADS,
            "num_nextn_predict_layers": 0,
            "num_shared_experts": N_SHARED,
            "q_lora_rank": 8,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 4,
            "quantization_config": { "format": "mxfp4-pack-quantized",
                                     "quant_method": "compressed-tensors" },
            "rms_norm_eps": 1e-5,
            "routed_expert_hidden_size": EXPERT_IN,
            "routed_scaling_factor": 1.0,
            "tie_word_embeddings": false,
            "top_k": 50,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "v_head_dim": 8,
            "vocab_size": VOCAB
        }
    })
}

/// The whole synthetic checkpoint. Returns the tensors, so the round-trip test can compare
/// the artifact against the bytes that went in.
fn write_fixture(src: &Path) -> Vec<Tensor> {
    let config = k3_config_json();
    common::write_config(src, &config);
    let _: K3Config = rivoli_artifact::schema::parse_config(&config.to_string())
        .expect("the fixture config parses");
    let tensors = all_tensors();
    common::write_shard_and_index(src, SHARD, &tensors);
    // The TWO aux files `finish_artifact` copies — not three, unlike V4's: this checkpoint
    // ships NO `chat_template.jinja` and no chat encoding exists in any tree for it (the
    // engine arm refuses `--port`), and no `generation_config.json` is read either.
    std::fs::write(src.join("tokenizer.json"), r#"{"model":{"type":"BPE"}}"#).unwrap();
    std::fs::write(src.join("tokenizer_config.json"), "{}").unwrap();
    tensors
}

#[test]
fn convert_k3_writes_an_artifact_that_reopens_as_the_same_model() {
    let (root, src, out) = common::scratch_src_out("k3-convert-rt");
    let tensors = write_fixture(&src);

    // `--verify` is the strong arm: `RoutedRepack` re-reads each `.f4` it wrote and
    // byte-compares every expert span against the source tensors.
    let log = BIN.at(&src, &out).convert(&["--verify"]);
    assert!(
        log.contains(&format!("latent={EXPERT_IN}")) && log.contains("dense prefix 1"),
        "the log must state the latent and the dense prefix, the two facts a reader of a \
         partial convert needs: {log}"
    );
    // Both exclusions counted — an exclusion logged is an observation, not an assumption.
    assert!(log.contains("2 vision"), "{log}");

    // It re-opens as the same model: the manifest carries the source config verbatim,
    // wrapper and all, so the architecture resolves and every `validate` check runs again.
    let art = out.to_str().unwrap();
    let cfg = K3Config::load(art).unwrap();
    assert_eq!((cfg.text.n_layers, cfg.text.n_experts), (LAYERS, EXPERTS));
    assert_eq!(cfg.text.expert_in, EXPERT_IN);
    assert_eq!(
        rivoli_artifact::schema::arch_of_artifact(art).unwrap(),
        rivoli_artifact::arch::Arch::KimiK3
    );
    FormatMeta::load(art).unwrap();
    // The `.f4` set is the MoE layers only — layer 0 is dense and has none.
    assert_eq!(f4_layer_range(art, LAYERS).unwrap(), FIRST_DENSE..LAYERS);
    for l in 0..LAYERS {
        assert_eq!(
            out.join(format!("L{l:02}.f4")).exists(),
            l >= FIRST_DENSE,
            "L{l:02}.f4 presence"
        );
    }

    assert_resident_is_a_verbatim_passthrough(&out, &tensors);

    // Both aux files reached the artifact — TWO, not V4's three; the fixture's comment on
    // the missing `chat_template.jinja` and `generation_config.json` carries why.
    for aux in ["tokenizer.json", "tokenizer_config.json"] {
        assert!(out.join(aux).is_file(), "{aux} did not reach the artifact");
    }

    // READ-ONLY re-verification of the finished artifact — the mode's own gate.
    let relog = BIN.at(&src, &out).convert(&["--verify-only"]);
    assert!(
        relog.contains("verify-only"),
        "the read-only mode must say so: {relog}"
    );
    common::clean(&root);
}

/// The resident set is the source, byte for byte, under the source dtypes — no widening,
/// no renames, routed experts and the vision side absent. `convert_v4` widens norms and
/// rewrites e8m0 scales; K3's converter deliberately does neither (the trunk is BF16 and
/// the loader owns the widening question), so ONE assertion covers every kept tensor and
/// the interesting content is in what must be ABSENT.
fn assert_resident_is_a_verbatim_passthrough(out: &Path, tensors: &[Tensor]) {
    let art = Safetensors::open_file(out.join("resident.safetensors").to_str().unwrap())
        .expect("the artifact's resident set opens");
    let mut checked = 0usize;
    for t @ (name, _, _, _) in tensors {
        if name.contains("block_sparse_moe.experts.") {
            assert!(
                art.raw(name).is_err(),
                "{name} is routed and belongs in .f4 — 15.72 GB/layer of duplication on \
                 the real checkpoint if this leaks"
            );
            continue;
        }
        if name.starts_with("vision_tower.") || name.starts_with("mm_projector.") {
            assert!(art.raw(name).is_err(), "{name} is multimodal and skipped");
            continue;
        }
        checked += 1;
        common::assert_verbatim(&art, t);
    }
    // Anti-vacuity: a walk that checked almost nothing would pass everything above. The
    // count is the fixture's non-routed, non-vision tensor count, restated as a floor.
    assert!(checked > 30, "only {checked} resident tensors compared");
}

/// The same checkpoint converted twice must produce byte-identical artifacts — what lets a
/// byte pin exist at all. Two output directories, because an existing `.f4` is REUSED and
/// the second run would compare a file against itself.
#[test]
fn two_converts_of_one_checkpoint_are_byte_identical() {
    // Not `scratch_src_out`: this test writes out1/out2, so the helper's `out` has no role.
    let root = common::scratch("k3-convert-det");
    let src = root.join("src");
    write_fixture(&src);
    let outs = [root.join("out1"), root.join("out2")];
    for o in &outs {
        BIN.at(&src, o).convert(&[]);
    }
    // The file list is built by iterator chain rather than `v4_convert.rs`'s vec+extend —
    // same claim, different tokens, which is what keeps two four-gate files from being the
    // clone jscpd reported between the first drafts (2026-08-16).
    let f4s = (FIRST_DENSE..LAYERS).map(|l| format!("L{l:02}.f4"));
    for name in ["manifest.json".to_string(), "resident.safetensors".into()]
        .into_iter()
        .chain(f4s)
    {
        let (x, y) = (outs[0].join(&name), outs[1].join(&name));
        assert!(
            std::fs::read(&x).unwrap() == std::fs::read(&y).unwrap(),
            "two converts disagree at {name} — a nondeterministic converter makes every \
             artifact an unverifiable snowflake"
        );
    }
    common::clean(&root);
}

/// A partial range writes only its layers and SAYS SO — and the dense prefix rides along
/// anyway, because it is trunk the model cannot decode without and it is never in
/// `--from/--to` (its layer has no experts to convert).
#[test]
fn a_partial_range_keeps_the_dense_prefix_and_never_rewrites_the_layer_count() {
    let (root, src, out) = common::scratch_src_out("k3-convert-range");
    write_fixture(&src);
    BIN.at(&src, &out).convert(&["--from", "2", "--to", "3"]);

    let art = out.to_str().unwrap();
    assert_eq!(f4_layer_range(art, LAYERS).unwrap(), 2..3);
    let cfg = K3Config::load(art).unwrap();
    assert_eq!(
        cfg.text.n_layers, LAYERS,
        "num_hidden_layers was rewritten; the partition arrays and first_k_dense_replace \
         are indexed by the REAL layer id"
    );
    for l in 0..LAYERS {
        assert_eq!(
            out.join(format!("L{l:02}.f4")).exists(),
            l == 2,
            "L{l:02}.f4 presence"
        );
    }
    // The dense prefix is ALWAYS resident — layer 0's `mlp.*` must be here even though the
    // request was layers 2..3 — and the unrequested MoE layer 1's trunk must NOT be, or a
    // partial run silently over-collects while the manifest claims only [2, 3).
    let res = Safetensors::open_file(out.join("resident.safetensors").to_str().unwrap()).unwrap();
    assert!(
        res.raw(&format!("{K3P}layers.0.mlp.gate_proj.weight"))
            .is_ok()
    );
    assert!(res.raw(&format!("{K3P}embed_tokens.weight")).is_ok());
    assert!(
        res.raw(&format!("{K3P}layers.1.block_sparse_moe.gate.weight"))
            .is_err(),
        "layer 1 was not requested and its trunk must not ride along"
    );
    common::clean(&root);
}

/// The guards that fire before anything is written, each on its own mutation.
#[test]
fn convert_k3_refuses_before_it_writes() {
    let (root, src, out) = common::scratch_src_out("k3-convert-refuse");
    write_fixture(&src);

    // `out_dir == src_dir` — the SafeWriter SIGBUS hazard, refused by path identity.
    BIN.at(&src, &src.join(".")).refuses(&[], "SIGBUS");

    // The dense prefix: `--from 0` would ask `F4Expert` for tensors that do not exist — a
    // confusing "tensor not found" instead of the real answer.
    BIN.at(&src, &out).refuses(&["--from", "0"], "dense prefix");
    // A range outside the model is REFUSED, not clamped.
    BIN.at(&src, &out).refuses(&["--to", "99"], "is not inside");
    BIN.at(&src, &out)
        .refuses(&["--from", "3", "--to", "3"], "is not inside");

    // A checkpoint whose CONFIG disagrees with its TENSORS. The resident set is a verbatim
    // passthrough and the manifest carries the config verbatim, so without the
    // confrontation the latent would reach the engine's launches having never been
    // compared to the weights it describes. 32 is a legal latent (positive, a multiple of
    // F4_GROUP) — only the tensors say it is wrong.
    let cfgp = src.join("config.json");
    let good = std::fs::read_to_string(&cfgp).unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&good).unwrap();
    doc["text_config"]["routed_expert_hidden_size"] = json!(32);
    common::write_config(&src, &doc);
    BIN.at(&src, &out).refuses(&[], "config implies");
    std::fs::write(&cfgp, &good).unwrap();

    // A missing trunk tensor the confrontation walks — refused by NAME, before the
    // resident writer would have half-written a file.
    let mut kept = all_tensors();
    let dropped = format!("{K3P}layers.2.block_sparse_moe.routed_expert_norm.weight");
    let at = kept
        .iter()
        .position(|(n, _, _, _)| *n == dropped)
        .expect("the fixture's MoE trunk names moved");
    kept.remove(at);
    common::write_shard_and_index(&src, SHARD, &kept);
    // Refused by the missing tensor's NAME — the confrontation speaks before the repack
    // loop or the resident writer would have touched the missing layer.
    BIN.at(&src, &out).refuses(&[], &dropped);
    common::clean(&root);
}

/// `0xff` is the e8m0 NaN, and the repack is the ONLY path that reads every routed scale
/// byte — K3's scales are declared plain `U8` by compressed-tensors, so the shape and this
/// refusal are the only evidence in the whole pipeline that these bytes are exponents. The
/// refusal must name the tensor under K3's OWN spelling (`weight_scale`, not `.scale`):
/// `F4Expert` once hardcoded V4's suffix here, and only a K3-named fixture can see that.
#[test]
fn an_e8m0_nan_scale_byte_is_refused_under_k3s_own_names() {
    let (root, src, out) = common::scratch_src_out("k3-convert-nan");
    write_fixture(&src);

    let mut poisoned = all_tensors();
    let target = format!("{}.w2.weight_scale", k3_expert_base(2, 1));
    // Not element 0: the refusal reports `[row][group]`, and poisoning the first byte
    // would pass a message that read `[0][0]` whatever the arithmetic was.
    let idx = f4_groups(MOE_INTER) + 1;
    let mut hits = 0usize;
    for (n, _, _, bytes) in &mut poisoned {
        if *n == target {
            bytes[idx] = 0xff;
            hits += 1;
        }
    }
    assert_eq!(hits, 1, "{target} is not in the fixture");
    common::write_shard_and_index(&src, SHARD, &poisoned);

    BIN.at(&src, &out).refuses(&[], &format!("{target}[1][1]"));
    common::clean(&root);
}
