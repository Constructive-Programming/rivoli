//! **The vendored S1b anchor golden must stay loadable, complete, self-describing, and the exact
//! bytes that were measured.**
//!
//! `docs/measurement/k3-reference/anchor.md` is the record; `tests/k3_anchor_driver.py` produced
//! the file and `tests/k3-anchor.sh` reproduces it. What is asserted here is deliberately narrow,
//! and worth saying plainly: **this is a fixture-integrity gate, not a correctness gate for the
//! port.** Nothing here compares any rivoli output to the golden, because at S1b there is no K3
//! kernel to score — so the literal answer to "what wrong implementation passes this" is every
//! one. What it does is hold the file to the shape S2's kernels will reach for, and refuse a file
//! that is not the one the doc describes.
//!
//! **No GPU, no python, no network.** Generating the golden needs all three (fla's KDA ops are
//! triton kernels with no CPU path); reading it needs none, which is the entire reason the bytes
//! are vendored instead of regenerated. This runs on the featureless dev profile like any other
//! host test — and it is the only automated thing that touches the anchor, so if it does not name
//! a tensor, nothing notices that tensor disappearing.
//!
//! Widths are **derived from the golden's own `tiny_config`**, not written as literals. That is
//! not tidiness: review found that the first version's widths made four structurally distinct
//! quantities accidentally equal, so an assertion that looked like it pinned a coupling
//! (`shared_experts` at `num_shared_experts * moe_intermediate_size`) was satisfied by the wrong
//! reading too. A derived width fails when the config drifts; a literal agrees with it.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::v4oracle::golden::GoldenSet;
use serde_json::Value;

const GOLDEN: &[u8] = include_bytes!("k3-anchor-decode.bin");
/// The vendored real config, the same one `model.rs` pins its fixture values to.
const REAL_CONFIG: &str = include_str!("../docs/measurement/k3-reference/config.json");

fn load() -> GoldenSet {
    GoldenSet::read_k3(&mut &GOLDEN[..]).expect("the vendored K3 anchor golden must load")
}

/// The tiny config the golden was produced from.
fn cfg(g: &GoldenSet) -> Value {
    serde_json::from_str(g.meta_get("tiny_config").expect("tiny_config")).expect("valid json")
}

fn field(c: &Value, key: &str) -> usize {
    c[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} is not an integer in tiny_config")) as usize
}

fn attn_field(c: &Value, key: &str) -> usize {
    c["linear_attn_config"][key]
        .as_u64()
        .unwrap_or_else(|| panic!("linear_attn_config.{key} is not an integer")) as usize
}

/// One float tensor's shape and values, by name. Panics with the file's own contents, because
/// "not found" is almost always a renamed capture and the next question is always "then what IS
/// in there".
fn float<'g>(g: &'g GoldenSet, name: &str) -> (&'g [usize], &'g [f32]) {
    g.floats
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, s, v)| (s.as_slice(), v.as_slice()))
        .unwrap_or_else(|| {
            let some: Vec<&String> = g.floats.iter().take(3).map(|(n, _, _)| n).collect();
            panic!(
                "{name} is not in the golden; it holds {} float tensors, e.g. {some:?}",
                g.floats.len()
            )
        })
}

fn shape_of(g: &GoldenSet, name: &str) -> Vec<usize> {
    float(g, name).0.to_vec()
}

/// One structural field: the driver must have asserted it, and it must equal the real config's.
///
/// Factored rather than written twice — the top-level and `linear_attn_config` loops had identical
/// bodies, and `build.rs`'s jscpd gate rejected them at 35 tokens. The duplication was there from
/// the moment they were written; it only crossed the 15-token floor once `cargo fmt` broke each
/// `assert!` across four lines, so the formatter is what made it VISIBLE, not what caused it. The
/// fix is this function, never a reverted format or an exemption.
fn check(declared: &[&str], scope: &str, key: &str, got: &Value, want: &Value) {
    assert!(
        declared.contains(&key),
        "{key} is not in structural_asserted"
    );
    assert_eq!(got[key], want[key], "tiny config lost {scope}{key}");
}

/// The provenance every consumer has to be able to read off the file, **by value**.
///
/// Not decoration. A golden separated from the versions that produced it cannot be re-derived,
/// and this one was produced by a stack that is not in the repo — a pinned reference downloaded
/// at a revision, plus fla, plus triton on one specific GPU.
///
/// Every one of these was a presence-only check until review 2026-08-11, and two could be
/// satisfied by the driver's own failure sentinels: `gpu` fell back to the string `"none"` when no
/// device was visible and `triton` to `"none"` on `ImportError`, both non-empty, so a golden
/// produced on a machine with neither passed — the exact configuration `anchor.md` spends a
/// section arguing is impossible. The sentinels are gone from the driver and the values are pinned
/// here.
#[test]
fn the_anchor_golden_records_what_produced_it() {
    let g = load();
    g.expect_defect("None")
        .expect("the vendored golden is the unperturbed run");
    for (key, want) in [
        ("mode", "decode"),
        ("seq", "8"),
        ("salt", "k3-anchor-1"),
        ("dtype", "torch.float32"),
        ("entry_point", "KimiLinearForCausalLM"),
        ("quantized", "no"),
        ("torch", "2.13.0+rocm7.2"),
        ("transformers", "4.56.2"),
        ("fla", "0.5.2"),
        ("triton", "3.5.1"),
        ("gpu", "AMD Radeon 8060S Graphics"),
        // `eager`, not `flash_attention_2`: the reference's `__init__` forces the latter and the
        // driver overrides it afterwards. Pinned because it is a DEVIATION — if a future run
        // silently keeps flash, the goldens change meaning and nothing else would say so.
        ("attn_implementation", "eager"),
        ("capture_layers", "0,1,3,12,91,92"),
        // The reference the whole independence argument rests on, at the pinned revision. Its
        // sibling `real_config_sha256_16` was pinned from the start and this one was not, which is
        // the asymmetry review noticed: a golden regenerated against a later revision of
        // `modeling_kimi_linear.py` — one where a kernel kwarg or the LoRA eps changed — passed
        // every test, with the metadata truthfully recording a hash nothing compared.
        ("ref_modeling_sha256_16", "9e3564c70ac21854"),
        ("ref_config_sha256_16", "735eb9ebe593e17d"),
        ("real_config_sha256_16", "9710e121a58d03ac"),
    ] {
        assert_eq!(g.meta_get(key), Some(want), "metadata {key}");
    }
}

/// The bytes are the ones that were measured.
///
/// `anchor.md` records this golden as bit-identical across separate runs on this GPU; without a
/// gate that claim rests on prose, and a regenerated file with the same tensor names, shapes and
/// metadata but different NUMBERS passes every other test here. FNV-1a rather than a real hash
/// because it needs no dependency and this is a tripwire, not a signature.
///
/// **When this fails after a deliberate regeneration, update both constants and say so in
/// `anchor.md`.** That is the intended workflow: re-vendoring is a reviewed change, not a side
/// effect of running the driver.
#[test]
fn the_vendored_bytes_are_the_measured_ones() {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in GOLDEN {
        h = (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3);
    }
    assert_eq!(GOLDEN.len(), 292_781, "vendored golden size");
    assert_eq!(h, 0xab97_8ba4_78e1_78ca, "vendored golden FNV-1a");
    let g = load();
    assert_eq!(g.floats.len(), 223, "float tensors");
    assert_eq!(g.ints.len(), 5, "int tensors");
}

/// The tiny config keeps the real model's STRUCTURE, which is what the traps live in.
///
/// The driver asserts this at generation time; it is re-asserted from the file because the file is
/// what survives, and a golden whose config drifted to something structurally unlike K3 would
/// still load and still look like a few hundred plausible tensors.
///
/// The two lists are kept in step by `structural_asserted`: the driver writes every field it
/// checked, and this test refuses to claim a field that list does not name. Two lists with nothing
/// tying them together is the drift review found here — `gate_lower_bound` and
/// `short_conv_kernel_size` were documented as asserted on both sides and were asserted on
/// neither.
#[test]
fn the_tiny_config_kept_the_real_structure() {
    let g = load();
    let c = cfg(&g);
    let real: Value = serde_json::from_str(REAL_CONFIG).unwrap();
    let real = &real["text_config"];
    let declared: Vec<&str> = g
        .meta_get("structural_asserted")
        .expect("structural_asserted")
        .split(',')
        .collect();

    // Every top-level structural field, compared against the REAL config rather than a literal —
    // so this cannot drift from the checkpoint the way a transcribed number can.
    for key in [
        "num_hidden_layers",
        "first_k_dense_replace",
        "moe_layer_freq",
        "attn_res_block_size",
        "num_shared_experts",
        "routed_scaling_factor",
        "latent_moe_use_norm",
        "moe_renormalize",
        "moe_router_activation_func",
        "activation_situ_beta",
        "activation_situ_linear_beta",
        "hidden_act",
        "mla_use_nope",
        "mla_use_output_gate",
        "rms_norm_eps",
        "use_grouped_topk",
        "num_expert_group",
        "topk_group",
        "topk_method",
    ] {
        check(&declared, "", key, &c, real);
    }
    // And the three inside `linear_attn_config`, which the driver REBUILDS by a dict merge rather
    // than inheriting whole — so their survival is the thing least guaranteed by construction.
    // `gate_lower_bound` is also a KDA kernel kwarg, with its own defect run.
    for key in [
        "gate_lower_bound",
        "short_conv_kernel_size",
        "use_full_rank_gate",
    ] {
        check(
            &declared,
            "linear_attn_config.",
            key,
            &c["linear_attn_config"],
            &real["linear_attn_config"],
        );
    }
    // The layer partition itself, every entry of both lists, 1-based exactly as the checkpoint
    // writes it. This is the field whose convention was documented INVERTED until G0 item 11 read
    // the code (`is_kda_layer` tests `layer_idx + 1`), so a golden that silently used the other
    // reading would be a golden of a different model. It was pinned by a four-element substring
    // prefix until review pointed out that a partition drifting in the MIDDLE passed.
    for key in ["kda_layers", "full_attn_layers"] {
        let got = c["linear_attn_config"][key].as_array().unwrap();
        let want = real["linear_attn_config"][key].as_array().unwrap();
        assert_eq!(got, want, "linear_attn_config.{key} must be the real list");
    }
    let kda = c["linear_attn_config"]["kda_layers"]
        .as_array()
        .unwrap()
        .len();
    let mla = c["linear_attn_config"]["full_attn_layers"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(
        kda + mla,
        field(&c, "num_hidden_layers"),
        "the two lists must partition every layer ({kda} KDA + {mla} MLA)"
    );
}

/// The operator fixtures S2's kernels will be scored against, named and shaped.
///
/// One test rather than one per operator: they share the file and the failure is always the same
/// sentence. Shapes are asserted, not just names — a `[1, 4, 32, 32]` recurrent state reshaped to
/// `[1, 4, 1024]` carries the same numbers and a different meaning, which is the fail-open
/// `golden::diff` already refuses for the same reason.
#[test]
fn the_operator_fixtures_s2_needs_are_present() {
    let g = load();
    let c = cfg(&g);
    let (hidden, latent) = (
        field(&c, "hidden_size"),
        field(&c, "routed_expert_hidden_size"),
    );
    let (nh, hd) = (attn_field(&c, "num_heads"), attn_field(&c, "head_dim"));

    // KDA, on a layer the real map makes KDA (1-based 1). Decode goes through
    // `fused_recurrent_kda`, and these inputs plus the state are the whole boundary of the one
    // operator no document attests to: `A_log`, `dt_bias`, the qk l2-norm, the beta sigmoid and
    // the gate lower bound all live INSIDE fla's kernel.
    let kda = "model.layers.0.kda.fused_recurrent_kda";
    assert_eq!(shape_of(&g, &format!("{kda}.in.q")), vec![1, 1, nh, hd]);
    assert_eq!(shape_of(&g, &format!("{kda}.in.g")), vec![1, 1, nh, hd]);
    assert_eq!(shape_of(&g, &format!("{kda}.in.beta")), vec![1, 1, nh]);
    assert_eq!(shape_of(&g, &format!("{kda}.in.A_log")), vec![nh]);
    assert_eq!(shape_of(&g, &format!("{kda}.in.dt_bias")), vec![nh * hd]);
    // The state, in and out: the recurrence IS the decode path, and a kernel that produces the
    // right `o` from the wrong state agrees for exactly one token. Square because
    // `head_k_dim == head_dim` — in the tiny model and in the real one (128 == 128) — so the
    // (K,V)-vs-(V,K) axis order is invisible to any shape assertion, which is why
    // `--defect KdaStateLayout` exists.
    let state = vec![1, nh, hd, hd];
    assert_eq!(shape_of(&g, &format!("{kda}.in.initial_state")), state);
    assert_eq!(shape_of(&g, &format!("{kda}.out.state")), state);

    // MLA, on the first MLA layer (1-based 4). The KV latent path is `kv_lora_rank + rope`, which
    // is NOT `q_head_dim` — the two were accidentally equal until the widths were fixed, and a
    // port reading the latent width off `qk_nope_head_dim` produced a bit-identical fixture.
    let mla = "model.layers.3.self_attn";
    assert_eq!(
        shape_of(&g, &format!("{mla}.kv_a_proj_with_mqa")),
        vec![
            1,
            1,
            field(&c, "kv_lora_rank") + field(&c, "qk_rope_head_dim")
        ]
    );
    assert_eq!(
        shape_of(&g, &format!("{mla}.g_proj")),
        vec![
            1,
            1,
            field(&c, "num_attention_heads") * field(&c, "v_head_dim")
        ]
    );
    assert_eq!(shape_of(&g, mla), vec![1, 1, hidden]);

    // The latent sandwich, and the norm that sits BETWEEN the weighted sum and the up projection —
    // not after it, which is the ordering trap `--defect LatentNormAfterUp` prices.
    let moe = "model.layers.1.block_sparse_moe";
    assert_eq!(
        shape_of(&g, &format!("{moe}.routed_expert_down_proj")),
        vec![1, latent]
    );
    assert_eq!(
        shape_of(&g, &format!("{moe}.routed_expert_norm")),
        vec![1, latent]
    );
    assert_eq!(
        shape_of(&g, &format!("{moe}.routed_expert_up_proj")),
        vec![1, hidden]
    );
    // The shared expert's width, DERIVED — `num_shared_experts * moe_intermediate_size`, the
    // `[hidden, 2*moe_inter]` coupling `validate` cannot see. This assertion was a literal and
    // therefore worthless until the widths were fixed: `2 * moe_inter` and the latent width were
    // both 64, so a port reading the shared expert's width off the latent passed it.
    assert_eq!(
        shape_of(&g, &format!("{moe}.shared_experts.gate_proj")),
        vec![
            1,
            1,
            field(&c, "num_shared_experts") * field(&c, "moe_intermediate_size")
        ]
    );

    // AttnRes: twice per layer plus once model-level. Captured by wrapping the reference's free
    // function, because `_apply_attn_res` reads `proj.weight` and `norm.weight` DIRECTLY and never
    // calls either module — so forward hooks on them fired zero times and this operator, S2's
    // FIRST, had no fixture at all until review 2026-08-11.
    for tag in [
        "model.layers.1.self_attention_res",
        "model.layers.1.mlp_res",
    ] {
        assert_eq!(
            shape_of(&g, &format!("{tag}.in.prefix_sum")),
            vec![1, hidden]
        );
        assert_eq!(shape_of(&g, &format!("{tag}.out")), vec![1, hidden]);
        let br = shape_of(&g, &format!("{tag}.in.block_residual"));
        assert_eq!(
            (br[0], br[2]),
            (1, hidden),
            "block residual is [tokens, blocks, hidden]"
        );
        assert!(br[1] >= 1, "layer 1 mixes at least the layer-0 snapshot");
    }
    // The model-level fold, whose accumulated stack is one block per `attn_res_block_size` layers.
    let br = shape_of(&g, "model.output_attn_res.in.block_residual");
    assert_eq!(
        br[1],
        field(&c, "num_hidden_layers").div_ceil(field(&c, "attn_res_block_size")),
        "one block residual per attn-res block"
    );
    assert_eq!(shape_of(&g, "model.output_attn_res.out"), vec![1, hidden]);

    assert_eq!(shape_of(&g, "model.norm"), vec![1, 1, hidden]);
    assert_eq!(shape_of(&g, "logits"), vec![1, 1, field(&c, "vocab_size")]);
}

/// The numbers are numbers, not noise.
///
/// A few hundred float tensors and, until review 2026-08-11, exactly one value-level assertion —
/// so a golden of zeros, of NaN, or drawn on wrong scales passed every test. These are the
/// invariants the driver's `init_weights` docstring says are load-bearing, checked from the bytes:
/// `A_log` is the decay rate and a wrong scale would freeze or erase the recurrent state; norm
/// weights near zero would make every downstream activation a denormal. Both depend on string
/// matching over parameter names, which is precisely what a reference revision bump breaks
/// silently.
#[test]
fn the_captured_values_are_on_their_declared_scales() {
    let g = load();
    let c = cfg(&g);
    let (_, a_log) = float(&g, "model.layers.0.kda.fused_recurrent_kda.in.A_log");
    assert!(
        a_log.iter().all(|v| (0.0..=16f32.ln()).contains(v)),
        "A_log must be log(uniform(1,16)): {a_log:?}"
    );
    let (_, dt) = float(&g, "model.layers.0.kda.fused_recurrent_kda.in.dt_bias");
    assert!(
        dt.iter().all(|v| (-4.0..=1.0).contains(v)),
        "dt_bias out of its draw range"
    );
    let (_, logits) = float(&g, "logits");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must be finite"
    );
    assert!(
        logits.iter().any(|v| *v != logits[0]),
        "all {} logits are identical — the forward pass collapsed",
        logits.len()
    );

    // Routing, as ints: selection is exact-or-not, and `golden::diff` scores the int section that
    // way. Distinctness is the real check — `[0, 0]` is a pair top-k cannot produce, and a bound
    // of `0..num_experts` alone would accept it.
    let (idx_shape, idx) = g
        .ints
        .iter()
        .find(|(n, _, _)| n == "model.layers.1.block_sparse_moe.gate.0")
        .map(|(_, s, v)| (s, v))
        .expect("the router's top-k indices");
    let (top_k, n_experts) = (field(&c, "num_experts_per_token"), field(&c, "num_experts"));
    assert_eq!(idx_shape, &vec![1, top_k]);
    assert!(
        idx.iter().all(|&e| (e as usize) < n_experts),
        "expert id out of range: {idx:?}"
    );
    assert_eq!(
        idx.iter().collect::<std::collections::BTreeSet<_>>().len(),
        top_k,
        "top-{top_k} selected the same expert twice: {idx:?}"
    );
    // The gate WEIGHTS, which are what `--defect RouterBiasInWeight` moves and which nothing named
    // before. Their sum pins the renormalisation AND `routed_scaling_factor`, read from the config
    // rather than assumed to be 1.
    let (_, w) = float(&g, "model.layers.1.block_sparse_moe.gate.1");
    let scale = c["routed_scaling_factor"].as_f64().unwrap() as f32;
    let sum: f32 = w.iter().sum();
    assert!(
        (sum - scale).abs() < 1e-5,
        "top-k weights sum to {sum}, not the renormalised {scale}"
    );
}

/// Every captured layer is present, and no others.
///
/// The exact set matters twice: `capture_layers` is the metadata a reader trusts, and the driver
/// once matched `model.layers.1` as a prefix and captured layers 1 and 10-19 (and 3 with 30-39) —
/// 25 layers instead of 6. That bug was invisible in every per-tensor assertion and obvious here.
#[test]
fn exactly_the_declared_layers_were_captured() {
    let g = load();
    let declared: Vec<usize> = g
        .meta_get("capture_layers")
        .expect("capture_layers")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(declared, vec![0, 1, 3, 12, 91, 92]);
    let mut seen: Vec<usize> = g
        .floats
        .iter()
        .map(|(n, _, _)| n.as_str())
        .chain(g.ints.iter().map(|(n, _, _)| n.as_str()))
        .filter_map(|n| n.strip_prefix("model.layers."))
        .filter_map(|rest| rest.split('.').next())
        .map(|d| d.parse().unwrap())
        .collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen, declared,
        "captured layers must be exactly the declared ones"
    );
}
