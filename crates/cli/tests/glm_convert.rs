//! `convert` end to end, on a synthetic GLM-5.2 fp8 checkpoint — M2's converter gate.
//!
//! **Why synthetic rather than the real checkpoint.** The fp8 source is ~700 GB and a
//! full conversion is hours of IO; the converter's unit of work is the whole tensor walk
//! (which tensors are fp8-copied, which widened, which int8'd, which VQ-encoded), and a
//! two-layer model with one dense and one MoE layer exercises every branch of it.
//!
//! **The round-trip claim is DETERMINISM**: the same checkpoint converted twice must
//! produce byte-identical artifacts (manifest, resident set, codebooks, expert files).
//! That is what lets `artifact_compat`-style byte pins exist at all — a nondeterministic
//! converter would make every artifact its own unverifiable snowflake.
//!
//! What this does NOT establish: anything about the real checkpoint's tensor names
//! (written here from the converter's own walk, so a name wrong in both is wrong in
//! both — the old tree's `k3_names.rs` closes that class against the shipped index and
//! the same gate arrives here with the real-checkpoint work), and nothing about decode
//! quality — the artifact is structurally valid, not measured.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::format::{Dtype, FormatMeta, SafeWriter, Safetensors};
use rivoli_artifact::glm_config::ModelConfig;
use rivoli_artifact::quant::{FP8_BLOCK, quantize_fp8_block};
use rivoli_artifact::schema::load_config;
use std::path::Path;
use std::process::Command;

mod common;
use common::{bf16_bytes, scratch, weights};

// Tiny but non-degenerate, and sized for the converter's own floors: hidden and
// moe_inter must be multiples of VQ_GROUP (64), and the codebook learner needs at least
// VQ_K (4096) subvectors from its sample walk — moe_inter*hidden/VQ_DIM = 128*128/4 =
// 4096, exactly the floor.
const HIDDEN: usize = 128;
const INTER: usize = 192; // dense-layer MLP
const MOE_INTER: usize = 128;
const LAYERS: usize = 2; // layer 0 dense, layer 1 MoE — both branches of the walk
const EXPERTS: usize = 4;
const VOCAB: usize = 61;
const Q_LORA: usize = 48;
const KV_LORA: usize = 32;
const QK_ROPE: usize = 8;
const QK_NOPE: usize = 16;
const V_HEAD: usize = 16;
const HEADS: usize = 2;

/// `f32_bytes` stays here: it encodes the fp8 scale blocks and this is its only caller. Its
/// two neighbours (`weights`, `bf16_bytes`) moved to `common` on 2026-08-16, when
/// `glimmer_convert.rs` became the second converter gate and jscpd reported them as clones.
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Add an fp8 projection under `<base>.weight` + its `weight_scale_inv`, the pair
/// `copy_fp8`/`deq` read.
fn add_fp8(w: &mut SafeWriter<'_>, base: &str, o: usize, i: usize) {
    let vals = weights(base, o * i);
    let (packed, scales) = quantize_fp8_block(&vals, [o, i], FP8_BLOCK).unwrap();
    w.add(format!("{base}.weight"), Dtype::F8E4M3, vec![o, i], packed);
    let sb = [o.div_ceil(FP8_BLOCK), i.div_ceil(FP8_BLOCK)];
    w.add(
        format!("{base}.weight_scale_inv"),
        Dtype::F32,
        sb.to_vec(),
        f32_bytes(&scales),
    );
}

fn add_bf16(w: &mut SafeWriter<'_>, name: &str, shape: &[usize]) {
    let n: usize = shape.iter().product();
    w.add(
        name.to_string(),
        Dtype::Bf16,
        shape.to_vec(),
        bf16_bytes(&weights(name, n)),
    );
}

/// The gate/up/down triple every MLP is made of — dense, per-expert and shared alike, so
/// the walk's only real branch is which bases exist, not what hangs off each one.
fn add_mlp(w: &mut SafeWriter<'_>, base: &str, inter: usize) {
    add_fp8(w, &format!("{base}.gate_proj"), inter, HIDDEN);
    add_fp8(w, &format!("{base}.up_proj"), inter, HIDDEN);
    add_fp8(w, &format!("{base}.down_proj"), HIDDEN, inter);
}

/// The two norms and the five MLA projections — identical in the dense and the MoE layer,
/// which is why the walk branches only below `.mlp`.
fn add_attn(w: &mut SafeWriter<'_>, lb: &str) {
    add_bf16(w, &format!("{lb}.input_layernorm.weight"), &[HIDDEN]);
    add_bf16(
        w,
        &format!("{lb}.post_attention_layernorm.weight"),
        &[HIDDEN],
    );
    add_bf16(
        w,
        &format!("{lb}.self_attn.q_a_layernorm.weight"),
        &[Q_LORA],
    );
    add_bf16(
        w,
        &format!("{lb}.self_attn.kv_a_layernorm.weight"),
        &[KV_LORA],
    );
    let qk_head = QK_NOPE + QK_ROPE;
    add_fp8(w, &format!("{lb}.self_attn.q_a_proj"), Q_LORA, HIDDEN);
    add_fp8(
        w,
        &format!("{lb}.self_attn.q_b_proj"),
        HEADS * qk_head,
        Q_LORA,
    );
    let kv_a = format!("{lb}.self_attn.kv_a_proj_with_mqa");
    add_fp8(w, &kv_a, KV_LORA + QK_ROPE, HIDDEN);
    let kv_b_out = HEADS * (QK_NOPE + V_HEAD);
    add_fp8(w, &format!("{lb}.self_attn.kv_b_proj"), kv_b_out, KV_LORA);
    add_fp8(w, &format!("{lb}.self_attn.o_proj"), HIDDEN, HEADS * V_HEAD);
}

/// The MoE layer's own tensors: the router gate and its bias, `EXPERTS` streamed triples,
/// and the resident shared expert — the VQ-encode and resident-set branches of the walk.
fn add_moe(w: &mut SafeWriter<'_>, lb: &str) {
    add_bf16(w, &format!("{lb}.mlp.gate.weight"), &[EXPERTS, HIDDEN]);
    w.add(
        format!("{lb}.mlp.gate.e_score_correction_bias"),
        Dtype::F32,
        vec![EXPERTS],
        f32_bytes(&weights("bias", EXPERTS)),
    );
    for e in 0..EXPERTS {
        add_mlp(w, &format!("{lb}.mlp.experts.{e}"), MOE_INTER);
    }
    add_mlp(w, &format!("{lb}.mlp.shared_experts"), MOE_INTER);
}

/// The safetensors half of the fixture: the converter's tensor walk, written out.
fn write_tensors(dir: &Path) {
    let mut w = SafeWriter::new();
    add_bf16(&mut w, "model.embed_tokens.weight", &[VOCAB, HIDDEN]);
    add_bf16(&mut w, "lm_head.weight", &[VOCAB, HIDDEN]);
    add_bf16(&mut w, "model.norm.weight", &[HIDDEN]);
    for l in 0..LAYERS {
        let lb = format!("model.layers.{l}");
        add_attn(&mut w, &lb);
        if l == 0 {
            // Dense layer: the plain MLP rides copy_fp8.
            add_mlp(&mut w, &format!("{lb}.mlp"), INTER);
        } else {
            add_moe(&mut w, &lb);
        }
    }
    w.write(
        dir.join("model-00001-of-00001.safetensors")
            .to_str()
            .unwrap(),
    )
    .unwrap();
}

/// The HF-side `config.json` the converter reads — every field its `validate` checks, so
/// dropping one here is how that check is kept honest.
fn write_config(dir: &Path) {
    let config = serde_json::json!({
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": LAYERS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "moe_intermediate_size": MOE_INTER,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "indexer_types": ["full", "shared"],
        "first_k_dense_replace": 1,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "rope_parameters": { "rope_theta": 8_000_000.0, "rope_type": "default" },
        "q_lora_rank": Q_LORA,
        "kv_lora_rank": KV_LORA,
        "qk_rope_head_dim": QK_ROPE,
        "qk_nope_head_dim": QK_NOPE,
        "v_head_dim": V_HEAD,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "scoring_func": "sigmoid",
        "hidden_act": "silu",
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true,
        "index_topk": 64,
        "index_head_dim": 16,
        "index_n_heads": 2,
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

/// The two files the converter copies through rather than reads. Stub contents on purpose:
/// this gate asserts they REACH the artifact, and `generation_config.json` is the one the
/// refusal test deletes.
fn write_sidecars(dir: &Path) {
    let tok = dir.join("tokenizer.json");
    std::fs::write(tok, b"{\"model\":{\"type\":\"BPE\"}}").unwrap();
    let gen_cfg = dir.join("generation_config.json");
    std::fs::write(gen_cfg, b"{\"eos_token_id\":[1]}").unwrap();
}

/// The synthetic checkpoint, written from the converter's own tensor walk.
fn write_fixture(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    write_tensors(dir);
    write_config(dir);
    write_sidecars(dir);
}

fn run_convert(src: &Path, out: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_convert"))
        .args([src.to_str().unwrap(), out.to_str().unwrap()])
        // The learner floor: 4096 subvectors exist in exactly one expert of the one MoE
        // layer, so sample from it directly and keep iterations small — this is a
        // structure gate, not a quality measurement.
        .args(["--sample-experts", "1", "--kmeans-iters", "2"])
        .output()
        .expect("run convert")
}

#[test]
fn convert_writes_an_artifact_that_reopens_and_is_deterministic() {
    let root = scratch("glm-convert-rt");
    let src = root.join("src");
    write_fixture(&src);

    let out1 = root.join("out1");
    let o = run_convert(&src, &out1);
    assert!(
        o.status.success(),
        "convert failed:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );

    // Re-opens as the same model: the manifest still parses as a GLM config with every
    // validate check live, and the format section loads.
    let cfg: ModelConfig = load_config(out1.to_str().unwrap()).unwrap();
    assert_eq!(cfg.n_layers, LAYERS);
    FormatMeta::load(out1.to_str().unwrap()).unwrap();
    Safetensors::open_file(out1.join("resident.safetensors").to_str().unwrap()).unwrap();
    assert!(out1.join("L01.vq3").exists() || out1.join("L1.vq3").exists());
    for aux in ["tokenizer.json", "generation_config.json"] {
        assert!(out1.join(aux).exists(), "{aux} missing from the artifact");
    }

    // THE ROUND-TRIP CLAIM: converting the same checkpoint again is byte-identical.
    let out2 = root.join("out2");
    let o = run_convert(&src, &out2);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let mut compared = 0usize;
    for entry in std::fs::read_dir(&out1).unwrap() {
        let name = entry.unwrap().file_name();
        let (a, b) = (
            std::fs::read(out1.join(&name)).unwrap(),
            std::fs::read(out2.join(&name)).unwrap_or_default(),
        );
        assert!(
            a == b,
            "{}: differs between two conversions of the same checkpoint",
            name.to_string_lossy()
        );
        compared += 1;
    }
    assert!(compared >= 5, "only {compared} files compared — wrong dir?");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_checkpoint_without_generation_config_is_refused() {
    // The lesson-34 gate: the artifact carries the stop tokens or the converter fails —
    // never a warning (the old tree shipped 56 unterminated runs on exactly this).
    let root = scratch("glm-convert-noeos");
    let src = root.join("src");
    write_fixture(&src);
    std::fs::remove_file(src.join("generation_config.json")).unwrap();
    let out = root.join("out");
    let o = run_convert(&src, &out);
    assert!(
        !o.status.success(),
        "convert succeeded without generation_config.json — the artifact has no stop tokens"
    );
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("generation_config.json"),
        "refusal must name the missing file: {err}"
    );
    let _ = std::fs::remove_dir_all(&root);
}
