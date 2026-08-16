//! `convert_v4` end to end, on a synthetic DeepSeek-V4-Flash checkpoint — M8's converter gate.
//!
//! **Why synthetic rather than the real checkpoint.** The V4-Flash source is 146 GB and a full
//! conversion is hours of IO; this converter's unit of work is the whole tensor walk — which
//! tensors are copied verbatim, which have their e8m0 scale widened, which are repacked into
//! `.f4`, and which of the two router tensors a layer carries — and a four-layer model with one
//! of every layer role exercises every branch of it.
//!
//! **The strongest arm is the converter's own `--verify`, not an assertion in this file.**
//! `RoutedRepack` re-reads each `.f4` it wrote and byte-compares every expert span against the
//! source tensors; the repack is a copy, so the only correct answer is zero differing bytes. A
//! test that re-derived the block layout here would be checking that a copy of the arithmetic
//! agrees with itself. So the round-trip arm runs `--verify`, and then runs `--verify-only`
//! against the finished artifact — which is also the read-only mode's own gate.
//!
//! **What this does NOT establish**, stated because a converter gate is easy to over-read: the
//! real checkpoint's tensor NAMES. The fixture is written from the converter's own walk, so a
//! name wrong in both is wrong in both. What closes that class is
//! `crates/artifact/src/quant/naming.rs`'s `v4_proj_order_matches_the_reference_expert_forward`
//! (which reads the shipped `model.py`) plus the shipped `model.safetensors.index.json`; the
//! index-side gate arrives with the real-checkpoint work, as it did for Glimmer.
//!
//! No GPU, no network — every byte is written by this file.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

// One braced `use`, and `common::` qualified at every call site rather than a `use common::{}`
// list. Both are jscpd's doing, and `glimmer_convert.rs` records reaching for the same two fixes
// for the same reason: with the flat form, this preamble was a 30-token clone of
// `glm_convert.rs`'s. Naming the module at each call also says where a fixture helper comes
// from, which a bare `write_shard(...)` in a converter gate does not.
use std::path::Path;

use rivoli_artifact::format::{Dtype, FormatMeta, Safetensors, f4_layer_range};
use rivoli_artifact::quant::{F4_GROUP, FP8_BLOCK, e8m0, f4_groups, v4_expert_base};
use rivoli_artifact::v4_config::V4Config;
use serde_json::json;

mod common;

/// One tensor of the fixture. The type is `common`'s, which carries a dtype because this model
/// needs five of them — see its doc for why the Glimmer gate's bf16-only local type moved.
use common::Tensor;

/// Tiny but structurally faithful: every distinction the real config makes survives the shrink.
///
/// `HIDDEN` and `MOE_INTER` are both multiples of [`F4_GROUP`] and **differ from each other**,
/// so a w1/w2 transposition changes a span length; `HEADS * HEAD_DIM` divides `O_GROUPS` but is
/// not equal to `HIDDEN`; `Q_LORA` differs from both. `LAYERS` is 4 against `RATIOS`' 6 entries,
/// which is the mtp tail the real config has and the reason `compress_ratio` bounds on
/// `n_layers` rather than on the vector.
const LAYERS: usize = 4;
const HIDDEN: usize = 64;
const MOE_INTER: usize = 96;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const Q_LORA: usize = 8;
const O_GROUPS: usize = 2;
const O_LORA: usize = 8;
const VOCAB: usize = 32;
const HC_MULT: usize = 2;
const INDEX_HEADS: usize = 2;
const INDEX_HEAD_DIM: usize = 16;

/// One per layer: 0 = sliding-window only, 4 = compressor + indexer, 128 = compressor only.
/// Two extra entries stand in for the real config's mtp tail.
const RATIOS: [usize; 6] = [0, 4, 128, 4, 0, 0];
/// Layers 0 and 1 route by `tid2eid`; 2 and 3 carry a `bias` instead. Both branches, and the
/// boundary is inside the layer range rather than at its edge.
const HASH_LAYERS: usize = 2;

const SHARD: &str = "model-00001-of-00001.safetensors";

// > **MOVED 2026-08-16.** `f32_bytes`, `opaque_bytes`, `e8m0_bytes` and the dtype-driven
// > byte policy lived here until `k3_convert.rs` became the fourth converter gate and
// > `build.rs`'s jscpd reported each as a clone. They are `common::{f32_bytes,
// > opaque_bytes, e8m0_bytes, tensor}` now, arguments travelling verbatim — including the
// > measured e8m0 range (`0x76..=0x7e`, zero `0x00`/`0xff` over the shipped 43-layer set)
// > that [`an_e8m0_nan_scale_byte_is_refused`] relies on.

/// [`common::tensor`], except `I64`: V4's only I64 tensor is `tid2eid`, whose values must
/// be LEGAL EXPERT IDS — bounded by this fixture's `EXPERTS` — so its bytes cannot come
/// from the model-agnostic helper.
fn tensor(name: &str, dtype: Dtype, shape: Vec<usize>) -> Tensor {
    if dtype == Dtype::I64 {
        let n: usize = shape.iter().product();
        let bytes = (0..n)
            .flat_map(|i| ((i % EXPERTS) as i64).to_le_bytes())
            .collect();
        return (name.to_string(), dtype, shape, bytes);
    }
    common::tensor(name, dtype, shape)
}

/// An fp8 pair in V4's spelling: `<base>.weight` (F8_E4M3) + `<base>.scale` (**F8_E8M0**), the
/// pair `SafeWriter::copy_fp8_e8m0` reads. The scale grid is `ceil(dim / FP8_BLOCK)` per axis,
/// which that function checks rather than assumes.
fn fp8_pair(out: &mut Vec<Tensor>, base: &str, o: usize, i: usize) {
    out.push(tensor(&format!("{base}.weight"), Dtype::F8E4M3, vec![o, i]));
    out.push(tensor(
        &format!("{base}.scale"),
        Dtype::F8E8M0,
        vec![o.div_ceil(FP8_BLOCK), i.div_ceil(FP8_BLOCK)],
    ));
}

/// One `Compressor`'s four tensors. Widths are read from the source by the converter rather
/// than derived, because on the real checkpoint they vary with `compress_ratio`; the fixture
/// therefore makes the two compressors on a layer DIFFERENT widths, so a converter that
/// assumed one shape for both would fail.
fn compressor(out: &mut Vec<Tensor>, base: &str, rows: usize, cols: usize) {
    out.push(tensor(&format!("{base}.ape"), Dtype::F32, vec![rows, cols]));
    out.push(tensor(
        &format!("{base}.norm.weight"),
        Dtype::Bf16,
        vec![cols],
    ));
    for t in ["wgate", "wkv"] {
        out.push(tensor(
            &format!("{base}.{t}.weight"),
            Dtype::Bf16,
            vec![cols, HIDDEN],
        ));
    }
}

/// An expert's three projections and their `(o_dim, i_dim)`, in `V4_PROJ`'s gate/up/down slot
/// order. ONE list because both kinds of expert are made of it — the routed one as FP4 and the
/// shared one as fp8 — and a second copy of an order-bearing list is the kind that goes wrong
/// silently: a reordered copy still compiles and still runs.
///
/// `w1`/`w3` are `[moe_inter, hidden]` and `w2` is `[hidden, moe_inter]` — the transposition
/// `F4Expert::spans` checks for, and the reason the two widths differ above.
const EXPERT_PROJS: [(&str, usize, usize); 3] = [
    ("w1", MOE_INTER, HIDDEN),
    ("w3", MOE_INTER, HIDDEN),
    ("w2", HIDDEN, MOE_INTER),
];

/// One routed expert: three FP4 projections, each nibble pairs plus an e8m0 group scale along
/// the INPUT dim.
fn routed_expert(out: &mut Vec<Tensor>, base: &str) {
    for (proj, o, i) in EXPERT_PROJS {
        out.push(tensor(
            &format!("{base}.{proj}.weight"),
            Dtype::I8,
            vec![o, i / 2],
        ));
        out.push(tensor(
            &format!("{base}.{proj}.scale"),
            Dtype::F8E8M0,
            vec![o, f4_groups(i)],
        ));
    }
}

/// Everything one layer carries, branching exactly as the converter does.
///
/// Three parts, because the converter has three: the attention frontend, the MoE half (whose
/// two kinds of expert are in two different FORMATS), and the per-layer tables plus the optional
/// compressor/indexer. One body was 84 lines and cc 9, which is the shape the code-health gate
/// refuses — and the split is the converter's own structure rather than an arbitrary cut.
fn layer_tensors(out: &mut Vec<Tensor>, l: usize) {
    attn_tensors(out, l);
    moe_tensors(out, l);
    per_layer_tables(out, l);
}

/// The attention frontend: four norms, the sink, and the five fp8 projections.
fn attn_tensors(out: &mut Vec<Tensor>, l: usize) {
    let lb = format!("layers.{l}");
    for (t, dim) in [
        ("attn_norm", HIDDEN),
        ("ffn_norm", HIDDEN),
        ("attn.q_norm", Q_LORA),
        ("attn.kv_norm", HEAD_DIM),
    ] {
        out.push(tensor(&format!("{lb}.{t}.weight"), Dtype::Bf16, vec![dim]));
    }
    out.push(tensor(
        &format!("{lb}.attn.attn_sink"),
        Dtype::F32,
        vec![HEADS],
    ));
    let qhd = HEADS * HEAD_DIM;
    fp8_pair(out, &format!("{lb}.attn.wq_a"), Q_LORA, HIDDEN);
    fp8_pair(out, &format!("{lb}.attn.wq_b"), qhd, Q_LORA);
    fp8_pair(out, &format!("{lb}.attn.wkv"), HEAD_DIM, HIDDEN);
    fp8_pair(
        out,
        &format!("{lb}.attn.wo_a"),
        O_GROUPS * O_LORA,
        qhd / O_GROUPS,
    );
    fp8_pair(out, &format!("{lb}.attn.wo_b"), HIDDEN, O_GROUPS * O_LORA);
}

/// The MoE half: the SHARED expert (fp8 e4m3, resident) and the routed ones (FP4, streamed).
///
/// The two are in different FORMATS, which is why the boundary matters more here than a name:
/// a block written past it is not merely the wrong weights, it is the wrong arithmetic.
fn moe_tensors(out: &mut Vec<Tensor>, l: usize) {
    // Named through `v4_expert_base(l, EXPERTS, EXPERTS)`, which is the one definition of the
    // routed/shared boundary — index `n_experts` is the shared slot.
    let shared = v4_expert_base(l, EXPERTS, EXPERTS);
    for (proj, o, i) in EXPERT_PROJS {
        fp8_pair(out, &format!("{shared}.{proj}"), o, i);
    }
    for e in 0..EXPERTS {
        routed_expert(out, &v4_expert_base(l, e, EXPERTS));
    }
}

/// The router (one of two tensors, by layer role), the hyper-connection tables, and the
/// compressor/indexer this layer's `compress_ratio` calls for.
fn per_layer_tables(out: &mut Vec<Tensor>, l: usize) {
    let lb = format!("layers.{l}");
    out.push(tensor(
        &format!("{lb}.ffn.gate.weight"),
        Dtype::Bf16,
        vec![EXPERTS, HIDDEN],
    ));
    if l < HASH_LAYERS {
        out.push(tensor(
            &format!("{lb}.ffn.gate.tid2eid"),
            Dtype::I64,
            vec![VOCAB, TOP_K],
        ));
    } else {
        out.push(tensor(
            &format!("{lb}.ffn.gate.bias"),
            Dtype::F32,
            vec![EXPERTS],
        ));
    }
    for t in ["hc_attn", "hc_ffn"] {
        for (s, shape) in [
            ("base", vec![HC_MULT]),
            ("fn", vec![HC_MULT, HC_MULT * HIDDEN]),
            ("scale", vec![HC_MULT]),
        ] {
            out.push(tensor(&format!("{lb}.{t}_{s}"), Dtype::F32, shape));
        }
    }
    let ratio = RATIOS[l];
    if ratio != 0 {
        compressor(out, &format!("{lb}.attn.compressor"), ratio, HEAD_DIM);
    }
    if ratio == 4 {
        fp8_pair(
            out,
            &format!("{lb}.attn.indexer.wq_b"),
            INDEX_HEADS * INDEX_HEAD_DIM,
            Q_LORA,
        );
        out.push(tensor(
            &format!("{lb}.attn.indexer.weights_proj.weight"),
            Dtype::Bf16,
            vec![INDEX_HEADS, Q_LORA],
        ));
        // Deliberately a DIFFERENT width from the attention compressor above: the converter
        // takes both shapes from the source, and a version that assumed one would fail here.
        compressor(
            out,
            &format!("{lb}.attn.indexer.compressor"),
            ratio,
            INDEX_HEAD_DIM,
        );
    }
}

/// The whole synthetic checkpoint's tensor set: the six model-level ones, then every layer.
fn all_tensors() -> Vec<Tensor> {
    let mut out = Vec::new();
    for n in ["embed.weight", "head.weight"] {
        out.push(tensor(n, Dtype::Bf16, vec![VOCAB, HIDDEN]));
    }
    out.push(tensor("norm.weight", Dtype::Bf16, vec![HIDDEN]));
    for n in ["hc_head_base", "hc_head_fn", "hc_head_scale"] {
        out.push(tensor(n, Dtype::F32, vec![HC_MULT]));
    }
    for l in 0..LAYERS {
        layer_tensors(&mut out, l);
    }
    out
}

/// The checkpoint's `config.json`, shrunk. Every value `V4Config::validate` looks at is here
/// and is one the real file could carry — the fixture is parsed back below, so a document this
/// writes that the schema would refuse fails HERE rather than as a confusing converter error.
fn v4_config_json() -> serde_json::Value {
    json!({
        // KEY ORDER is the shipped file's, which is roughly alphabetical — not the struct's.
        // That is worth a line because it is also what keeps the `num_hidden_layers` /
        // `hidden_size` / `vocab_size` / `num_attention_heads` run from being a jscpd clone of
        // `glimmer_convert.rs`'s fixture: two checkpoints declare those four under the same
        // HuGGingFace names, so only their neighbours differ.
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "expert_dtype": "fp4",
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": 1,
        "vocab_size": VOCAB,
        "qk_rope_head_dim": 8,
        "q_lora_rank": Q_LORA,
        "o_groups": O_GROUPS,
        "o_lora_rank": O_LORA,
        "sliding_window": 16,
        "rms_norm_eps": 1e-6,
        "compress_ratios": RATIOS,
        "compress_rope_theta": 160000.0,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "beta_fast": 32, "beta_slow": 1, "factor": 16.0,
            "original_max_position_embeddings": 65536, "type": "yarn"
        },
        "max_position_embeddings": 4096,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": MOE_INTER,
        "n_shared_experts": 1,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "num_hash_layers": HASH_LAYERS,
        "swiglu_limit": 10.0,
        "quantization_config": {
            "fmt": "e4m3", "scale_fmt": "ue8m0",
            "weight_block_size": [FP8_BLOCK, FP8_BLOCK]
        },
        "index_n_heads": INDEX_HEADS,
        "index_head_dim": INDEX_HEAD_DIM,
        "index_topk": 8,
        "hc_mult": HC_MULT,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
    })
}

/// The whole synthetic checkpoint. Returns the tensors, so the round-trip test can compare the
/// artifact against the bytes that went in. (`write_shard_and_index` moved to `common/` with
/// the byte helpers above, same trigger.)
fn write_fixture(src: &Path) -> Vec<Tensor> {
    let config = v4_config_json();
    common::write_config(src, &config);
    // Parsed back rather than trusted: the fixture's config is the one the converter will read,
    // and a document this file writes that `validate` would refuse must fail here.
    let _: V4Config = rivoli_artifact::schema::parse_config(&config.to_string())
        .expect("the fixture config parses");
    let tensors = all_tensors();
    common::write_shard_and_index(src, SHARD, &tensors);
    // The three AUX files `finish_artifact` copies. Stubs: this converter reads none of them —
    // V4's stop tokens reach the engine through the copied `generation_config.json`, and there
    // is deliberately no `chat_template.jinja` (see `rivoli_artifact::v4_encoding`).
    common::write_aux(
        src,
        &[
            ("tokenizer.json", r#"{"model":{"type":"BPE"}}"#),
            ("tokenizer_config.json", "{}"),
            ("generation_config.json", r#"{"eos_token_id": 1}"#),
        ],
    );
    tensors
}

/// The binary under test — the run/convert/refuse plumbing is `common::ConvertBin`'s,
/// factored there when `k3_convert.rs` became the fourth converter gate and jscpd reported
/// the free-function quartet the first two gates each carried.
const BIN: common::ConvertBin = common::ConvertBin {
    exe: env!("CARGO_BIN_EXE_convert_v4"),
    tool: "convert_v4",
};

#[test]
fn convert_v4_writes_an_artifact_that_reopens_as_the_same_model() {
    let (root, src, out) = common::scratch_src_out("v4-convert-rt");
    let tensors = write_fixture(&src);

    // `--verify` is the strong arm: the converter re-reads every `.f4` it wrote and
    // byte-compares each expert span against the source tensors.
    let log = BIN.at(&src, &out).convert(&["--verify"]);
    assert!(
        log.contains(&format!("experts={EXPERTS} layers 0..{LAYERS}")),
        "{log}"
    );

    // It re-opens as the same model: the manifest carries the source config verbatim, so the
    // architecture resolves and every `validate` check runs again.
    let art = out.to_str().unwrap();
    let cfg = V4Config::load(art).unwrap();
    assert_eq!((cfg.n_layers, cfg.n_experts), (LAYERS, EXPERTS));
    assert_eq!(cfg.compress_ratios.len(), RATIOS.len());
    FormatMeta::load(art).unwrap();
    // The provenance range is what the artifact HOLDS, not `num_hidden_layers` — the config's
    // per-layer tables stay indexed by the REAL layer id, which is why the two must differ in
    // kind. Here they coincide because this run covered the whole model; the partial-range arm
    // below is where they separate.
    assert_eq!(f4_layer_range(art, LAYERS).unwrap(), 0..LAYERS);
    for l in 0..LAYERS {
        assert!(out.join(format!("L{l:02}.f4")).exists(), "L{l:02}.f4");
    }

    assert_resident_matches_source(&out, &tensors);

    // Every aux file reached the artifact. `finish_artifact` refuses a failed copy, so this
    // asserts the LIST — a file dropped from it would leave the engine with no stop tokens.
    for aux in [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ] {
        assert!(out.join(aux).exists(), "{aux} missing from the artifact");
    }

    // READ-ONLY re-verification of the finished artifact — the mode's own gate, and a second,
    // independent statement that the `.f4` bytes are the source's.
    let log = BIN.at(&src, &out).convert(&["--verify-only"]);
    assert!(log.contains("verify-only"), "{log}");
    common::clean(&root);
}

/// The resident set, tensor by tensor: fp8 weights VERBATIM, e8m0 scales widened EXACTLY, norms
/// widened, `embed`/`head` still bf16.
///
/// Not a length check. A byte-length comparison passes on a zeroed tensor, and the widening is
/// where a converter can be plausibly wrong: `copy_fp8_e8m0` is lossless only because every
/// e8m0 code is exactly representable in f32, and asserting `e8m0(b)` per element is what makes
/// that claim testable rather than asserted.
fn assert_resident_matches_source(out: &Path, tensors: &[Tensor]) {
    let art = Safetensors::open_file(out.join("resident.safetensors").to_str().unwrap())
        .expect("the artifact's resident set opens");
    let mut checked = 0usize;
    for t @ (name, dtype, shape, bytes) in tensors {
        // Routed experts live in the `.f4` files and must NOT be in the resident set. This is
        // the boundary `v4_expert_base` defines, and the shared expert is on the other side of
        // it — so `experts.` and not `ffn.` is the test.
        if name.contains(".experts.") {
            assert!(
                art.raw(name).is_err(),
                "{name} is routed and belongs in .f4"
            );
            continue;
        }
        checked += 1;
        match dtype {
            // An e8m0 scale is REWRITTEN under this engine's own name for it, `weight_scale_inv`,
            // as f32 — the name `dequant_fp8` reads. Its source name is gone.
            Dtype::F8E8M0 => {
                let base = name.strip_suffix(".scale").unwrap();
                let inv = format!("{base}.weight_scale_inv");
                assert!(art.raw(name).is_err(), "{name} kept its source name");
                let (got, got_shape) = art.typed(&inv, Dtype::F32).unwrap();
                assert_eq!(got_shape, &shape[..], "{inv} shape");
                let want: Vec<f32> = bytes.iter().map(|&b| e8m0(b).unwrap()).collect();
                assert_eq!(common::as_f32(got), want, "{inv} is not the exact e8m0");
            }
            // bf16 is widened to f32 for norms/gate/weights_proj/compressor, and kept for
            // `embed`/`head`. Which is which is the converter's decision under test.
            Dtype::Bf16 if name == "embed.weight" || name == "head.weight" => {
                common::assert_verbatim(&art, t);
            }
            Dtype::Bf16 => common::assert_widened(&art, t),
            // F32, I64 and the fp8 WEIGHT half: verbatim, same dtype, same bytes.
            _ => common::assert_verbatim(&art, t),
        }
    }
    // Anti-vacuity: a walk that checked almost nothing would pass every assertion above.
    assert!(checked > 100, "only {checked} resident tensors compared");
}

/// The same checkpoint converted twice must produce byte-identical artifacts.
///
/// That is what lets a byte pin exist at all — a nondeterministic converter makes every
/// artifact its own unverifiable snowflake. Two separate output directories, because a `.f4`
/// that already exists is REUSED rather than rewritten and the second run would compare a file
/// against itself.
#[test]
fn two_converts_of_one_checkpoint_are_byte_identical() {
    let root = common::scratch("v4-convert-det");
    let src = root.join("src");
    write_fixture(&src);
    let (a, b) = (root.join("out1"), root.join("out2"));
    BIN.at(&src, &a).convert(&[]);
    BIN.at(&src, &b).convert(&[]);
    let mut names = vec![
        "manifest.json".to_string(),
        "resident.safetensors".to_string(),
    ];
    names.extend((0..LAYERS).map(|l| format!("L{l:02}.f4")));
    for name in &names {
        assert_eq!(
            std::fs::read(a.join(name)).unwrap(),
            std::fs::read(b.join(name)).unwrap(),
            "{name} differs between two converts of the same checkpoint"
        );
    }
    common::clean(&root);
}

/// A partial range writes only its layers, and SAYS SO in the manifest.
///
/// The `f4_source` range is the artifact's only statement of what it holds: two `.f4` sets built
/// from different checkpoints — or different ranges — are byte-indistinguishable on disk. And
/// `num_hidden_layers` must NOT be rewritten, or a 2-layer artifact would claim to be a 2-layer
/// MODEL and every per-layer table in the config would be re-indexed under it.
#[test]
fn a_partial_range_is_recorded_and_never_rewrites_the_layer_count() {
    let (root, src, out) = common::scratch_src_out("v4-convert-range");
    write_fixture(&src);
    BIN.at(&src, &out).convert(&["--from", "1", "--to", "3"]);

    let art = out.to_str().unwrap();
    assert_eq!(f4_layer_range(art, LAYERS).unwrap(), 1..3);
    let cfg = V4Config::load(art).unwrap();
    assert_eq!(
        cfg.n_layers, LAYERS,
        "num_hidden_layers was rewritten; compress_ratios is indexed by the REAL layer id"
    );
    // Layer 1 is ratio 4 and layer 2 is ratio 128 — the roles survive the partial convert
    // because the table did.
    assert!(cfg.layer_has_indexer(1).unwrap() && !cfg.layer_has_indexer(2).unwrap());
    for l in 0..LAYERS {
        assert_eq!(
            out.join(format!("L{l:02}.f4")).exists(),
            (1..3).contains(&l),
            "L{l:02}.f4 presence"
        );
    }
    common::clean(&root);
}

/// The guards that fire before anything is written, each on its own mutation of the fixture.
#[test]
fn convert_v4_refuses_before_it_writes() {
    let (root, src, out) = common::scratch_src_out("v4-convert-refuse");
    write_fixture(&src);

    // `out_dir == src_dir` is refused by path identity — `src/.` canonicalizes to the same
    // inode, so the trailing component must not fool it. The hazard is `SafeWriter`'s and the
    // guard is now its own; see `SafeWriter::refuse_writing_into_source`.
    BIN.at(&src, &src.join(".")).refuses(&[], "SIGBUS");

    // A range outside the model is REFUSED, not clamped: `--to 999` silently converting 4
    // layers would look like it did what was asked.
    BIN.at(&src, &out).refuses(&["--to", "99"], "is not inside");
    BIN.at(&src, &out)
        .refuses(&["--from", "3", "--to", "3"], "is not inside");

    // A checkpoint whose CONFIG disagrees with its TENSORS. Nothing downstream would catch it:
    // every tensor here is copied verbatim and the manifest carries the config verbatim, so a
    // wrong `n_heads` would size every attention launch at decode time with no error at convert
    // time. Mutated in the config rather than in the shard, which is the direction the
    // reference's own `convert.py --model-parallel` produces.
    let good = std::fs::read(src.join("config.json")).unwrap();
    let mut doc: serde_json::Value = serde_json::from_slice(&good).unwrap();
    doc["num_attention_heads"] = json!(HEADS * 2);
    std::fs::write(src.join("config.json"), doc.to_string()).unwrap();
    BIN.at(&src, &out).refuses(&[], "config implies");
    std::fs::write(src.join("config.json"), &good).unwrap();

    // A layer whose ROUTER tensor disagrees with `num_hash_layers`. Driven off the config, so
    // a checkpoint that carries the other branch's tensor fails instead of silently taking
    // whichever branch it happens to satisfy.
    let mut kept = all_tensors();
    let dropped = format!("layers.{}.ffn.gate.tid2eid", HASH_LAYERS - 1);
    kept.retain(|(n, _, _, _)| *n != dropped);
    common::write_shard_and_index(&src, SHARD, &kept);
    BIN.at(&src, &out).refuses(&[], &dropped);
    common::clean(&root);
}

/// `0xff` is the e8m0 NaN, and the repack is the ONLY path that reads every routed scale byte.
///
/// At decode those bytes DMA from NVMe straight into the pool slot and the host never sees
/// them; the kernel decodes the NaN correctly and cannot refuse it, and `moe_fixed`'s
/// saturating clamp then launders it into a finite ±2^14 — one bad byte becomes 32 weights of
/// plausible garbage with no error anywhere. Measured on the shipped 43-layer set (9.26e9 scale
/// bytes): zero `0x00` and zero `0xff`, so this guard is green on every artifact that exists
/// and an injection is the only thing that has ever made it speak.
#[test]
fn an_e8m0_nan_scale_byte_is_refused() {
    let (root, src, out) = common::scratch_src_out("v4-convert-nan");
    write_fixture(&src);

    let mut poisoned = all_tensors();
    let target = format!("{}.w3.scale", v4_expert_base(1, 2, EXPERTS));
    let hit = poisoned
        .iter_mut()
        .find(|(n, _, _, _)| *n == target)
        .unwrap_or_else(|| panic!("{target} is not in the fixture"));
    // Not element 0: the refusal reports `[row][group]`, and poisoning the first byte would
    // pass a message that read `[0][0]` whatever the arithmetic was.
    let idx = f4_groups(HIDDEN) + 1;
    hit.3[idx] = 0xff;
    common::write_shard_and_index(&src, SHARD, &poisoned);

    // Named down to the row and group, because that is the whole value of refusing here rather
    // than letting the kernel launder it.
    BIN.at(&src, &out).refuses(&[], &format!("{target}[1][1]"));
    BIN.at(&src, &out)
        .refuses(&[], &format!("{F4_GROUP}-weight group"));
    common::clean(&root);
}
