//! # The bit-identity gate — every Rust kernel against its hipcc twin at `archive/hip-kernels`
//!
//! `docs/investigations/rust-kernels.md` §3. The migration's safety net: each kernel ported
//! to `crates/kernels` must produce BYTE-IDENTICAL outputs to the hipcc kernel it replaces,
//! or the gate is RED. The reference is not a rebuilt copy and not the Rust kernel compared
//! with itself: `tests/bit-identity.sh` extracts the tagged `.hip` sources with
//! `git archive`, builds them with hipcc (the flags `build.rs` used) into a shared object,
//! and this binary calls that object's `rivoli_*` C launchers beside the shipped Rust
//! `launch_*` wrappers — so guards, grid geometry and kernel are compared together, guard
//! codes included.
//!
//! **On-demand and `#[ignore]`d**: it needs the GPU (run it only through the script, which
//! takes the flock and the contention witness) and the reference object. The deviceless
//! tests in `plants` are not ignored: they are the CPU halves of the recorded red proofs.
//!
//! ## Adding a row (Phase 1 lanes)
//!
//! One `fn(&Ref, &mut Tally)` per launcher, in a module per `.hip` file, listed in
//! [`ROWS`]; [`device::sweep`] runs shapes × [`inputs::SEEDS`] seeds and times the hot shape,
//! [`device::Tally::guard`] compares one refusal. The script's `REF_SOURCES` gains the file.
//! A row whose counts reach zero is RED, never skipped. A ported kernel with no row here (or
//! a row naming no shipped kernel) is a deviceless red in `kernel_coverage.rs`'s
//! `every_rust_kernel_has_a_gate_row_and_a_launchers_row`; `touch_region` is its one argued
//! exemption. Where production launches a kernel in place (`h == g`, `y == x`), the row
//! sweeps that aliasing too.

// Test code: a failed expectation IS the report, so `expect` is the right spelling.
#![allow(clippy::expect_used)]

mod attn_inputs;
mod attn_plants;
mod cells;
#[path = "../common/mod.rs"]
mod common;
mod inputs;
mod linalg_plants;
mod plant_bar;
mod plant_models;
mod plants;
mod ranges;

#[cfg(feature = "rocm")]
mod activation;
#[cfg(feature = "rocm")]
mod attn;
#[cfg(feature = "rocm")]
mod attn_gqa;
#[cfg(feature = "rocm")]
mod blockindex;
#[cfg(feature = "rocm")]
mod device;
#[cfg(feature = "rocm")]
mod fwd;
#[cfg(feature = "rocm")]
mod headtail;
#[cfg(feature = "rocm")]
mod indexer;
#[cfg(feature = "rocm")]
mod kvcompress;
#[cfg(feature = "rocm")]
mod linalg;
#[cfg(feature = "rocm")]
mod mla;
#[cfg(feature = "rocm")]
mod moe;
#[cfg(feature = "rocm")]
mod moe_f4;
#[cfg(feature = "rocm")]
mod probes;
#[cfg(feature = "rocm")]
mod recurrent;
#[cfg(feature = "rocm")]
mod residual;

/// The table Phase 1 extends: `(launcher, row)`.
#[cfg(feature = "rocm")]
type Row = (&'static str, fn(&device::Ref, &mut device::Tally));

#[cfg(feature = "rocm")]
const ROWS: &[Row] = &[
    ("swiglu", activation::swiglu),
    ("swiglu_clamped_bf16", activation::swiglu_clamped_bf16),
    ("situ_glu_f32", activation::situ_glu_f32),
    ("rmsnorm_single", activation::rmsnorm_single),
    (
        "rmsnorm_centered_single",
        activation::rmsnorm_centered_single,
    ),
    (
        "rmsnorm_weightless_batch",
        activation::rmsnorm_weightless_batch,
    ),
    ("act_quant_f8", activation::act_quant_f8),
    ("rope_interleave", activation::rope_interleave),
    ("rope_split_half", activation::rope_split_half),
    ("hc_pre", residual::hc_pre),
    ("hc_post", residual::hc_post),
    ("attn_res", residual::attn_res),
    (
        "gated_residual_collapse_f32",
        residual::gated_residual_collapse_f32,
    ),
    ("mla_attend", attn::mla_attend),
    ("gather_attn_shared_kv", attn::gather_attn_shared_kv),
    ("gqa_attend", attn_gqa::gqa_attend),
    ("gqa_block_attend", attn_gqa::gqa_block_attend),
    ("mha_attend", attn_gqa::mha_attend),
    ("layernorm", indexer::layernorm),
    ("index_append", indexer::index_append),
    ("index_score", indexer::index_score),
    ("index_pool_push", indexer::index_pool_push),
    ("index_head_route", indexer::index_head_route),
    ("index_topk", indexer::index_topk),
    ("index_pool_norm_f32", indexer::index_pool_norm_f32),
    ("index_score_blocks_f32", indexer::index_score_blocks_f32),
    ("act_quant_f4_rotated", blockindex::act_quant_f4_rotated),
    ("index_score_blocks", blockindex::index_score_blocks),
    ("gemm_bf16", kvcompress::gemm_bf16),
    ("kv_compress_deposit", kvcompress::kv_compress_deposit),
    ("kv_compress_prefill", kvcompress::kv_compress_prefill),
    ("kv_compress_decode", kvcompress::kv_compress_decode),
    ("mla_absorb_fp8", mla::mla_absorb_fp8),
    ("mla_value_fp8", mla::mla_value_fp8),
    ("act_quant_f8_prefix", mla::act_quant_f8_prefix),
    ("rmsnorm_batch", mla::rmsnorm_batch),
    ("qk_norm", mla::qk_norm),
    ("rope_adjacent", mla::rope_adjacent),
    ("gemv_fp8_bf16", mla::gemv_fp8_bf16),
    ("gemv_fp8", linalg::gemv_fp8),
    ("gemv_i8", linalg::gemv_i8),
    ("gemv_i4", linalg::gemv_i4),
    ("gemv_vq", linalg::gemv_vq),
    ("gemv_f32", linalg::gemv_f32),
    ("vq_encode", linalg::vq_encode),
    ("embed_bf16_row_bcast", headtail::embed_bf16_row_bcast),
    ("hc_head_collapse", headtail::hc_head_collapse),
    ("embed_i8_row", fwd::embed_i8_row),
    ("append_kv", fwd::append_kv),
    ("gather_rope", fwd::gather_rope),
    ("vadd", fwd::vadd),
    ("sigmoid_gate", fwd::sigmoid_gate),
    ("logit_softcap", fwd::logit_softcap),
    ("argmax", fwd::argmax),
    ("flag_nonfinite", fwd::flag_nonfinite),
    ("hash_rows", fwd::hash_rows),
    (
        "gated_delta_recurrent_f32",
        recurrent::gated_delta_recurrent_f32,
    ),
    ("short_conv_silu_f32", recurrent::short_conv_silu_f32),
    ("rmsnorm_gate_heads_f32", recurrent::rmsnorm_gate_heads_f32),
    ("gdn_recurrent_f32", recurrent::gdn_recurrent_f32),
    ("moe_expert_range", moe::moe_expert_range),
    ("moe_expert_range_i4", moe::moe_expert_range_i4),
    ("moe_acc_drain_s", moe::moe_acc_drain_s),
    ("moe_acc_drain_to_s", moe::moe_acc_drain_to_s),
    ("moe_expert_range_f4", moe_f4::moe_expert_range_f4),
    ("moe_expert_range_f4_situ", moe_f4::moe_expert_range_f4_situ),
];

#[cfg(feature = "rocm")]
#[test]
#[ignore = "on-demand GPU gate — run through tests/bit-identity.sh"]
fn every_rust_kernel_is_byte_identical_to_its_tagged_hipcc_twin() {
    let r = device::Ref::open();
    // libtest prints `test <name> ... ` with no newline before `--nocapture` output; without
    // this the first verdict line is mid-line and the classifier miscounts (rust-kernels.md §5,
    // Phase 0a lease 1).
    println!();
    let mut red = Vec::new();
    for (name, row) in ROWS {
        let mut t = device::Tally::default();
        row(&r, &mut t);
        let (line, green) = t.verdict(name);
        println!("{line}");
        t.timing
            .iter()
            .for_each(|l| println!("    TIME {name} {l}"));
        println!("    {}", t.shapes_line());
        t.notes.iter().for_each(|l| println!("    {l}"));
        if !green {
            red.push(*name);
        }
    }
    finish(&red, ROWS.len(), "");
}

/// The RESULT line the script checks its row count against, then the test's own verdict.
#[cfg(feature = "rocm")]
fn finish(red: &[&str], total: usize, note: &str) {
    println!(
        "RESULT {} of {total} rows BIT-IDENTICAL{note}",
        total - red.len()
    );
    assert!(red.is_empty(), "RED rows: {red:?}");
}

/// The prelude probe rows (`probes`): Phase 0b's families against the hipcc probe object,
/// one verdict line per group, the same tokens and RESULT line the launcher rows print.
#[cfg(feature = "rocm")]
#[test]
#[ignore = "on-demand GPU gate — run through tests/bit-identity.sh --probes"]
fn every_prelude_probe_is_identical_to_its_hipcc_twin() {
    let p = probes::Probes::open();
    println!();
    let t = probes::run(&p);
    let red: Vec<&str> = t
        .rows
        .iter()
        .filter(|(name, g)| !g.verdict(name).1)
        .map(|(name, _)| name.as_str())
        .collect();
    let nan: u64 = t.rows.iter().map(|(_, g)| g.nan_only).sum();
    let note = format!(" (NaN as a class; {nan} NaN-payload-only cells)");
    finish(&red, t.rows.len(), &note);
}
