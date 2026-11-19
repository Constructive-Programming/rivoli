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
//! test below is not ignored: it is the CPU half of the recorded red proof.
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

mod cells;
#[path = "../common/mod.rs"]
mod common;
mod inputs;

#[cfg(feature = "rocm")]
mod activation;
#[cfg(feature = "rocm")]
mod blockindex;
#[cfg(feature = "rocm")]
mod device;
#[cfg(feature = "rocm")]
mod indexer;
#[cfg(feature = "rocm")]
mod kvcompress;
mod plant_models;
#[cfg(feature = "rocm")]
mod probes;
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

/// The CPU half of the recorded red proof (`docs/measurement/gate-red-proofs.md` §18): the
/// planted defect is `rope`'s `lo` without its FMA — `x·cs − y·sn` rounded twice instead of
/// once — and this shows it is ADVERSARIAL on the gate's own rope inputs before any device
/// arm runs. §1's lesson is why the order matters: the spike's first plant was
/// sub-threshold and passed at 8 seeds, so a plant must be shown to move outputs on the
/// inputs the gate actually draws, or a green run says nothing.
///
/// `cos`/`sin` here are the host's, not OCML's — the device values differ in the last bits,
/// so this measures the plant's RATE on these operands, not the device's exact cells. A rate
/// far from zero is the claim; the device arm is the measurement.
#[test]
fn rope_unfused_plant_is_adversarial_on_cpu() {
    let (mut cells, mut moved, mut elems, mut diff) = (0, 0, 0, 0);
    for seed in 1..=inputs::SEEDS {
        let (pos, theta) = inputs::rope_at(seed);
        let v = inputs::draw(32 * 128, (seed, 1), 1.0, true);
        let before = diff;
        for head in v.chunks_exact(128) {
            for j in 0..64 {
                let (x, y) = (head[2 * j], head[2 * j + 1]);
                let ang = pos as f64 * theta.powf(-2.0 * j as f64 / 128.0);
                let (cs, sn) = (ang.cos() as f32, ang.sin() as f32);
                elems += 1;
                diff +=
                    usize::from(x.mul_add(cs, -(y * sn)).to_bits() != (x * cs - y * sn).to_bits());
            }
        }
        cells += 1;
        moved += usize::from(diff > before);
    }
    println!(
        "rope unfused plant: {diff}/{elems} lo outputs differ; {moved}/{cells} seeds affected"
    );
    // The gate reddens on ONE differing word. Demand the plant move most seeds, so it cannot
    // be the §1 kind that hides in a lucky draw.
    assert!(
        moved * 10 >= cells * 9,
        "plant moved only {moved}/{cells} seeds — sub-threshold"
    );
}
