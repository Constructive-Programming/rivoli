//! **The vendored S2 Qwen3.8-Flash-Next anchor goldens are the exact bytes that were measured, and
//! they describe themselves.**
//!
//! `docs/measurement/qwen-reference/anchor.md` is the record; `tests/qwen_anchor_driver.py`
//! produced the files and `tests/qwen-anchor.sh` reproduces them. What is asserted across this
//! file, `qwen_anchor_fixtures.rs` and `qwen_anchor_windows.rs` is deliberately narrow, and worth
//! saying plainly: **these are fixture-integrity gates, not correctness gates for the port.**
//! Nothing here compares a rivoli output to a golden, because at S2 there is no Qwen kernel to
//! score — so the literal answer to "what wrong implementation passes this" is every one. What
//! they do is hold the files to the shape S3's kernels will reach for, refuse a file that is not
//! the one the doc describes, and refuse a tolerance table the measurements do not support.
//!
//! **No GPU, no python, no network — and for this model no device was needed to GENERATE them
//! either**, which is the one structural difference from `k3_anchor.rs`. fla's KDA ops are triton
//! kernels with no CPU path; Qwen's GatedDeltaNet ships pure-torch bodies inside transformers, so
//! the whole matrix was measured on CPU in about ten minutes. That is recorded in each golden's
//! `gdn_entry` and `forbidden_absent` metadata and pinned below, because "it ran without fla" is
//! exactly the claim a venv with fla installed would quietly falsify.
//!
//! This file owns provenance, the byte pins, the structural survival of the tiny config, the width
//! audit, and the tolerance table.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use serde_json::Value;
use sha2::{Digest, Sha256};

#[path = "common/qwen_anchor_read.rs"]
mod read;
#[path = "common/tolerance.rs"]
mod tolerance;

use read::{
    DECODE, REAL_CONFIG, declared_fields, draws_are_independent, field, load, meta_json,
    provenance, width,
};

// ---------------------------------------------------------------------------------------------
// Provenance.

/// The provenance every consumer has to be able to read off the file, **by value**.
///
/// Not decoration. A golden separated from the versions that produced it cannot be re-derived, and
/// these were produced by a pinned venv that is not in the repo.
///
/// **The four selectors are the point of this test.** `attn_implementation`,
/// `experts_implementation` and the GDN and conv entry points each default to something OTHER than
/// what this run pins, and none of them is visible in the numbers: the reference force-overwrites
/// `_attn_implementation` in `__init__` and the driver overrides it afterwards; `_grouped_mm`
/// refuses Double, so the fp64 floor run needs `eager` experts and the fp32 run must match it; and
/// the kernel-from-hub decorator selects fla's fused ops the moment fla is importable. A golden
/// regenerated in a venv where any of those flipped is a golden of a different computation, with
/// metadata truthfully recording a configuration nothing compared.
/// The stack and the four selectors, as a named table rather than an inline literal — so the
/// list READS as the documentation it is, and so the call site below is one line. Extracted
/// 2026-08-31 for the second reason too: inline, the loop plus the first three pairs matched
/// `k3_anchor.rs`'s identical loop at 38 tokens, and the fix for a duplicate is to stop
/// writing it twice rather than to reword one copy.
const STACK_AND_SELECTORS: &[(&str, &str)] = &[
    ("mode", "decode"),
    ("seq", "16"),
    ("dtype", "torch.float32"),
    ("entry_point", "Qwen4ExpForCausalLM"),
    ("quantized", "no"),
    ("torch", "2.13.0+cpu"),
    ("transformers", "5.16.1"),
    ("python", "3.14.6"),
    ("attn_implementation", "eager"),
    ("experts_implementation", "eager"),
    // BOTH torch bodies, because a decode golden is a warm 16-token prefill (chunked)
    // followed by one token (recurrent). The prefill golden records only the chunked one,
    // and that asymmetry is what `common/tolerance.rs`'s two-floor discussion for
    // `gdn_op` rests on.
    (
        "gdn_entry",
        "torch_chunk_gated_delta_rule,torch_recurrent_gated_delta_rule",
    ),
    ("conv_entry", "causal_conv1d_fn,causal_conv1d_update"),
    ("forbidden_absent", "fla,kernels,causal_conv1d"),
    ("capture_layers", "0,1,2,3,27,46,47"),
    // Empty, and that is a statement rather than a formality: no module was called twice
    // in a decode forward, so every per-name lookup in these three files is unambiguous.
    // The prefill golden's map is NOT empty, and `qwen_anchor_windows.rs` is where that
    // matters.
    ("repeated_captures", "{}"),
    // Read off the CONSTRUCTED `nn.Conv1d`, not restated from `ngram_size`. It replaced a
    // width-audit row that compared `ngram_size` to itself and could not fail.
    ("ple_conv_dilation_observed", "3"),
    ("ref_modeling_sha256_16", "77fec77d87f2a0eb"),
    ("ref_config_sha256_16", "26b47995740e3bc5"),
];

#[test]
fn the_anchor_goldens_record_what_produced_them() {
    for v in DECODE {
        provenance(&load(v), v.name, STACK_AND_SELECTORS);
    }
}

/// **The vendored `config.json` is the one the goldens were generated against — RECOMPUTED, with no
/// constant in this file to drift.**
///
/// K3's and Glimmer's equivalents pin an FNV-1a of the config here and compare it to a number
/// written beside it, which leaves two frozen copies agreeing with each other while neither is
/// recomputed from the live file. This does better and it costs nothing: each golden's metadata
/// already carries `real_config_sha256_16`, the first 16 hex digits of the sha256 of the file the
/// run consumed, so the check is to hash the LIVE file and compare against what the BYTES say.
/// There is no third number to update, a config edit fails immediately, and a deliberate re-vendor
/// makes it pass again by construction.
///
/// `sha2` rather than FNV-1a only because the driver's record is a sha256 — this repo's "FNV, not a
/// sha256 dep" rule is about hashes it is free to CHOOSE, and `rivoli-oracles` already depends on
/// `sha2` because the reference-matching parameter draw does.
///
/// This does not duplicate `crates/artifact/tests/qwen_names.rs`'s
/// `the_vendored_files_hash_as_the_header_records`: that gate ties the file to
/// `tensor-families.tsv`'s recorded header, this one ties it to the goldens. Both can be true and
/// either can fail alone.
#[test]
fn the_vendored_real_config_is_the_one_the_goldens_saw() {
    let live = format!("{:x}", Sha256::digest(REAL_CONFIG.as_bytes()));
    for v in read::all() {
        let g = load(v);
        let recorded = g.meta_get("real_config_sha256_16").expect("config sha");
        assert_eq!(
            &live[..16],
            recorded,
            "{}: docs/measurement/qwen-reference/config.json hashes to {} but this golden was \
             generated against {recorded}. Regenerate and re-vendor, or restore the file.",
            v.name,
            &live[..16]
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The bytes.

/// The bytes are the ones that were measured, and the two draws are two draws.
#[test]
fn the_vendored_bytes_are_the_measured_ones() {
    read::check_pinned_bytes(DECODE);
    read::PREFILL.check_bytes();
    read::NGRAM_REAL.check_bytes();
    for v in DECODE {
        let g = load(v);
        assert_eq!(g.floats.len(), 314, "{}: float tensors", v.name);
        assert_eq!(g.ints.len(), 11, "{}: int tensors", v.name);
    }
    // 314 exactly, not a lower bound: the two goldens must hold the SAME tensors and differ in
    // their VALUES, so a shrinking intersection is its own finding. The argument for comparing
    // per tensor rather than per file is in the helper.
    draws_are_independent(&load(&DECODE[0]), &load(&DECODE[1]), 314);
}

// ---------------------------------------------------------------------------------------------
// Structure.

/// Each tiny config keeps the real model's STRUCTURE, which is what the traps live in.
///
/// The driver asserts this at generation time; it is re-asserted from the file because the file is
/// what survives, and a golden whose config drifted to something structurally unlike this model
/// would still load and still look like a few hundred plausible tensors.
///
/// The two lists are kept in step by `structural_asserted`: the driver writes every field it
/// checked, and this test refuses to claim a field that list does not name. Two lists with nothing
/// tying them together is the drift K3's review found.
#[test]
fn the_tiny_configs_kept_the_real_structure() {
    let real: Value = serde_json::from_str(REAL_CONFIG).unwrap();
    let real = &real["text_config"];
    for v in DECODE {
        let g = load(v);
        let c = meta_json(&g, "tiny_config");
        let declared = declared_fields(&g);
        for key in [
            "num_hidden_layers",
            "full_attention_interval",
            "hc_count",
            "linear_conv_kernel_dim",
            "ple_conv_kernel_size",
            "ngram_size",
            "heads_per_ngram",
            "make_ngram_vocab_size_divisible_by",
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_compress_ratio",
            "output_gate_type",
            "hidden_act",
            "rms_norm_eps",
        ] {
            structural_field_survived(&declared, &c, real, key);
        }
        layer_partition_survived(&c, real, &declared);
        the_defaulted_fields_are_asserted_against_their_default(&c, real, &declared);
    }
}

/// The 48-entry layer partition and the one-indexed PLE host, both entry by entry.
///
/// **`layer_types` is compared in its ALIASED form.** The checkpoint spells the QSA layers
/// `full_attention` and the reference rewrites that to `qwen_sparse_attention` before validating
/// (CFG:180-184, comment: "layers that are actually using an indexer"), so the aliased list is what
/// the model runs. Entry by entry rather than by count or by prefix: a partition that drifted in
/// the MIDDLE is exactly what either of those would pass.
///
/// `ple_layer_ids` is ONE-INDEXED on disk — MOD:1202 tests `layer_idx + 1` — so `[2]` puts the
/// injection on `layer_idx` 1. That was the S0 correction, and it is the reason a capture list
/// written from the config's own digits would watch the wrong layer.
fn layer_partition_survived(c: &Value, real: &Value, declared: &[&str]) {
    field_was_declared(declared, "layer_types");
    let got = c["layer_types"].as_array().expect("layer_types");
    let want: Vec<String> = real["layer_types"]
        .as_array()
        .expect("real layer_types")
        .iter()
        .map(|t| match t.as_str() {
            Some("full_attention") => "qwen_sparse_attention".to_string(),
            Some(other) => other.to_string(),
            None => panic!("layer_types holds a non-string"),
        })
        .collect();
    assert_eq!(
        got.len(),
        field(c, "num_hidden_layers"),
        "one type per layer"
    );
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert_eq!(a.as_str(), Some(b.as_str()), "layer_types[{i}]");
    }
    // One-indexed on disk, compared whole. Routed through the shared helper so its declaration
    // half is asserted in the one place that asserts every other field's.
    structural_field_survived(declared, c, real, "ple_layer_ids");
}

/// **`structural_asserted` names this field, and the tiny config still carries the real config's
/// value for it.**
///
/// Factored rather than written at each of its three call sites: jscpd matched the second and third
/// at 26 tokens the moment they existed, which is the duplication gate doing exactly its job on
/// this file's own author. The fix is this function — never a reverted format and never an
/// exemption, because the matched tokens here are not the point of anything.
fn structural_field_survived(declared: &[&str], c: &Value, real: &Value, key: &str) {
    field_was_declared(declared, key);
    assert_eq!(c[key], real[key], "tiny config lost {key}");
}

/// The declaration half alone, for a field whose VALUE is compared by a rule rather than by
/// equality — `layer_types`, which is compared against its aliased form, and the two defaulted
/// fields, which are compared against a class default the real file does not hold.
fn field_was_declared(declared: &[&str], key: &str) {
    assert!(
        declared.contains(&key),
        "{key} is not in structural_asserted, so nothing on the generating side checked it"
    );
}

/// **The two structural fields that are ABSENT from the checkpoint's `config.json`**, and the
/// assertion that they are absent.
///
/// `norm_topk_prob` (True) and `seed` (1234) come from `Qwen4ExpTextConfig`'s own defaults, CFG:163
/// and CFG:156. The driver asserted them against the config's EFFECTIVE value rather than against
/// the file, which is what makes trap T25 real: a port that reads only the file finds nothing to
/// honour, and the default is ON. Asserting the ABSENCE is the half that keeps this honest — if a
/// later checkpoint revision starts shipping them, the claim "these are defaults" stops being true
/// and this is what says so.
fn the_defaulted_fields_are_asserted_against_their_default(
    c: &Value,
    real: &Value,
    declared: &[&str],
) {
    for (key, default) in [("norm_topk_prob", Value::Bool(true)), ("seed", 1234.into())] {
        field_was_declared(declared, key);
        assert!(
            real.get(key).is_none(),
            "{key} is now IN the checkpoint's config.json — it is no longer a class default, and \
             the T25 argument built on that has to be re-derived"
        );
        assert_eq!(c[key], default, "the tiny config lost the {key} default");
    }
}

// ---------------------------------------------------------------------------------------------
// The width audit.

/// **The width-collision audit, re-run from the bytes at BOTH ends — and the inherited version of
/// this claim was false.**
///
/// The golden carries three lists (`width_audit`) and two width maps (`widths` for the tiny config,
/// `real_widths` for the checkpoint's own). The lists are the ones the driver actually checked, so
/// there is no second copy here to drift; what this test adds is that a shortened list cannot pass
/// silently, which is why the row counts are asserted.
///
/// * **`distinct`** — pairs the real config keeps distinct, so a tiny width that collapsed one
///   would let a port read the wrong field and still produce a bit-identical fixture. Checked
///   distinct in BOTH maps.
/// * **`equal`** — equalities the real config HAS, kept so the fixture stays like the model. The
///   square `[d_k][d_v]` GDN state is here (128 == 128 in the real model, 20 == 20 in the tiny): a
///   hazard the fixture must CARRY rather than hide, and `gdn_state_axis_swapped` is what prices
///   it. `640 == 640`, routed against shared expert width, is here too. Checked equal in BOTH maps.
/// * **`collide_in_real`** — collisions the real config HAS and the tiny config deliberately
///   BREAKS. Checked at both ends: equal in real, distinct in tiny.
///
/// > **The third list did not exist until 2026-08-31 and it is the finding.** The inherited audit
/// > put `half_head_dim` vs `indexer_head_dim` in `distinct` under a comment reading "pairs the
/// > real config keeps DISTINCT" — and they are BOTH 128 in the real config, which also makes 128
/// > equal to `2 * rotary_dim` (64) there. So a port reading the index head width off either
/// > `head_dim / 2` or twice the rotary width produces a bit-identical fixture at real widths, and
/// > the list asserted the opposite. Evaluating all nineteen rows against the real config found two
/// > further collisions the tiny config already broke and nobody had written down — `hc_width` vs
/// > `conv_dim` (both 10240) and `index_qk_width` vs `moe_intermediate_size` (both 640). Coverage
/// > nobody recorded is coverage the next width edit deletes for free.
///
/// A fifth `equal` row died in the same pass: it compared `ple_conv_dilation` to `ngram_size`, and
/// `derived_widths` DEFINES the former as the latter, so the row could not fail. The claim now
/// lives in `ple_conv_dilation_observed`, read off the constructed conv and pinned above.
#[test]
fn the_width_audit_holds_at_both_ends() {
    for v in DECODE {
        let g = load(v);
        let (tiny, real) = (meta_json(&g, "widths"), meta_json(&g, "real_widths"));
        let audit = meta_json(&g, "width_audit");
        // A shortened list is a check whose examined-count reached zero quietly.
        for (list, n) in [("distinct", 18), ("equal", 4), ("collide_in_real", 3)] {
            assert_eq!(rows(&audit, list).len(), n, "{}: {list} rows", v.name);
        }
        for (label, a, b) in rows(&audit, "distinct") {
            assert_ne!(
                width(&tiny, &a),
                width(&tiny, &b),
                "tiny collision: {label}"
            );
            assert_ne!(width(&real, &a), width(&real, &b), "EQUAL in real: {label}");
        }
        for (label, a, b) in rows(&audit, "equal") {
            assert_eq!(width(&tiny, &a), width(&tiny, &b), "lost in tiny: {label}");
            assert_eq!(width(&real, &a), width(&real, &b), "not real: {label}");
        }
        for (label, a, b) in rows(&audit, "collide_in_real") {
            assert_eq!(width(&real, &a), width(&real, &b), "stale claim: {label}");
            assert_ne!(width(&tiny, &a), width(&tiny, &b), "tiny collides: {label}");
        }
        the_three_named_asymmetries_are_in_the_recorded_widths(&tiny, &real);
    }
}

/// One audit list as `(label, lhs_key, rhs_key)` triples.
fn rows(audit: &Value, list: &str) -> Vec<(String, String, String)> {
    audit[list]
        .as_array()
        .unwrap_or_else(|| panic!("width_audit.{list}"))
        .iter()
        .map(|r| {
            let s = |i: usize| r[i].as_str().expect("audit row entry").to_string();
            (s(0), s(1), s(2))
        })
        .collect()
}

/// The three width facts S3 will be tempted to read the wrong way, stated as numbers from the bytes
/// rather than as prose.
///
/// Named individually on top of the list-driven loops above because each is a specific trap and a
/// generic loop cannot say which row mattered. Every number comes from the recorded maps.
fn the_three_named_asymmetries_are_in_the_recorded_widths(tiny: &Value, real: &Value) {
    // 1. The GDN head asymmetry: 16 key heads feeding 48 value heads, a 1:3 `repeat_interleave`.
    //    The tiny config keeps the RATIO (3 -> 9) and keeps 3 distinct from `num_key_value_heads`,
    //    which the real config also separates (16 against 2). A port that broadcast keys to values
    //    with the ATTENTION's n_rep would be wrong by 12/3 in real and 2/3 here — so the two
    //    repeats are asserted different, which is the audit row that makes the confusion visible.
    assert_eq!(width(real, "gdn_repeat"), 3, "real GDN key->value repeat");
    assert_eq!(width(tiny, "gdn_repeat"), 3, "tiny keeps the 1:3 repeat");
    assert_eq!(width(real, "linear_num_key_heads"), 16);
    assert_eq!(width(real, "linear_num_value_heads"), 48);
    assert_ne!(
        width(tiny, "attn_n_rep"),
        width(tiny, "gdn_repeat"),
        "the attention's n_rep and the GDN's key->value repeat must be different numbers here"
    );
    // 2. The rope widths. Real: rotary 64, head_dim/2 128, indexer head 128 — the last two EQUAL,
    //    which is the collision the tiny config breaks. Tiny: 8, 16, 24, all three distinct, and
    //    `2 * rotary` (16) is not the indexer head width either, so both wrong readings fail.
    assert_eq!(width(real, "rotary_dim"), 64);
    assert_eq!(width(real, "half_head_dim"), 128);
    assert_eq!(width(real, "indexer_head_dim"), 128);
    let (rot, half, idx) = (
        width(tiny, "rotary_dim"),
        width(tiny, "half_head_dim"),
        width(tiny, "indexer_head_dim"),
    );
    assert!(
        rot != half && half != idx && rot != idx,
        "{rot}/{half}/{idx} must be three different widths"
    );
    assert_ne!(
        2 * rot,
        idx,
        "2*rotary_dim must not be the indexer head width either"
    );
    // 3. The KEPT equality: routed and shared expert widths, 640 == 640 in real. Kept BECAUSE the
    //    model has it — a fixture that separated them would stop scoring a port that sized the
    //    shared expert off the routed one, which at real widths IS what the model does.
    assert_eq!(width(real, "moe_intermediate_size"), 640);
    assert_eq!(width(real, "shared_expert_intermediate_size"), 640);
    assert_eq!(
        width(tiny, "moe_intermediate_size"),
        width(tiny, "shared_expert_intermediate_size"),
        "the 640 == 640 equality is KEPT, not broken"
    );
}

// ---------------------------------------------------------------------------------------------
// The tolerance table.

/// **The per-operator tolerances, and the gate on them.**
///
/// `common/tolerance.rs` carries the table — shared with K3 and Muse Glimmer — and this asserts
/// that every row's policy still follows from its two measured numbers, and that the table holds a
/// row for exactly the operators that were measured WITH a targeting defect.
///
/// The nine are those operators. Four buckets the anchor measures a floor for get NO row on purpose
/// (`gdn_out`, `hc_mlp`, `qsa` and `ple_inject` are downstream of every defect that reaches them),
/// and neither do the four localisation buckets; `ngram_hash`'s floor is exactly 0.000e0 because its
/// outputs are int64, so no `Rel` value is expressible at all. Each of those is argued at the table.
/// **S3 must not score an operator with no row against a threshold** — compare it exactly, or
/// measure the floor the same way (`--dtype float64`, then `--by-operator`) and add a row.
#[test]
fn the_tolerance_table_is_supported_by_its_measurements() {
    tolerance::tolerances_leave_room(tolerance::QWEN);
    const MEASURED: [&str; 9] = [
        "gdn_conv",
        "gdn_op",
        "gdn_out_norm",
        "gdn_proj",
        "hc_attn",
        "indexer",
        "moe",
        "moe_route",
        "qsa_proj",
    ];
    tolerance::table_covers_exactly(tolerance::QWEN, &MEASURED);
}
