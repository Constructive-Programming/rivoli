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

#[path = "../common/mod.rs"]
mod common;
mod inputs;
mod plants;
mod ranges;

#[cfg(feature = "rocm")]
mod activation;
#[cfg(feature = "rocm")]
mod device;
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
