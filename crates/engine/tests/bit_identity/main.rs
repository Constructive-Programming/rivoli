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
//! A row whose counts reach zero is RED, never skipped.

// Test code: a failed expectation IS the report, so `expect` is the right spelling.
#![allow(clippy::expect_used)]

#[path = "../common/mod.rs"]
mod common;
mod inputs;

#[cfg(feature = "rocm")]
mod activation;
#[cfg(feature = "rocm")]
mod device;
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
    println!(
        "RESULT {} of {} rows BIT-IDENTICAL",
        ROWS.len() - red.len(),
        ROWS.len()
    );
    assert!(red.is_empty(), "RED rows: {red:?}");
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
