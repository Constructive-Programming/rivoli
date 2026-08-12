//! **S2 item 4: the MLP, which needs no new kernel — checked rather than assumed.**
//!
//! `glimmer-architecture.md` §3: `down(silu(gate(h)) * up(h))`, at 6656 → 19968 → 6656 with bf16
//! weights. Two kernels already in the tree cover it — `gemm_bf16` for the three projections (the
//! un-quantized `F.linear` path) and `swiglu` for the activation, which is `silu(g)*u` exactly and
//! unclamped, as Glimmer's config has no `swiglu_limit`.
//!
//! The plan's item 4 reads "no new kernel; delete the guard, bind the dims". Two of those three
//! were already true when this was written: `MAX_FUSED_INTER` is gone (`artifact/model.rs` has a
//! test asserting there is no intermediate ceiling any more), and the kernels exist. **What was
//! missing is that nothing ran them at Glimmer's widths** — the same hole S2 item 1 had, where
//! every golden is at head_dim 8 and the production path is 128.
//!
//! # This is a WIDTH test, not a second oracle
//!
//! `gemm_bf16` and `swiglu` are already scored against their own oracles elsewhere
//! (`tests/kernel.rs`, `tests/kvcompress.rs`). Re-proving their arithmetic here would be a second
//! copy of a check that exists — and the first draft of this file did exactly that for `swiglu`,
//! until jscpd matched its host silu against `tests/kernel.rs`'s and refused the build. The
//! composition test below runs `swiglu` at 19968 as part of the pipeline, which is the width
//! evidence; the arithmetic evidence already existed. What is new is 6656/19968: a launcher guard that refuses these
//! dims, an index that overflows i32 at `19968*6656 = 132,913,152` elements, or an LDS/occupancy
//! assumption that only holds at small `k`, would all be invisible to every existing fixture and
//! to the tiny-width goldens both.
//!
//! # What this canNOT do, and why item 4 has no golden
//!
//! **The anchor cannot score an MLP end to end.** It captures `pre_feedforward_layernorm.out` (the
//! input) and `mlp.down_proj.out` (the output) but neither the intermediate activations nor the
//! WEIGHTS, and the weights exist only as a deterministic draw inside the driver's python. Scoring
//! the composition against the reference would mean either re-implementing that draw in Rust — a
//! second copy of a generator, which is the thing this repo's gates exist to refuse — or teaching
//! the driver to export weights and re-vendoring the goldens, whose bytes and FNV are pinned.
//! So this stage checks the pieces at the real widths, and the composition is S3's business, where
//! the layer loop can be scored against `mlp.down_proj.out` with the real checkpoint behind it.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

#[path = "common/glimmer_fixture.rs"]
mod fixture;
use fixture::{fill, from_bf16, gemv_bf16, swiglu, to_bf16, worst_rel};

/// Glimmer's text-side MLP shape, from `glimmer-architecture.md` §1 and the config this port
/// vendors. Not read from the tiny config: the tiny config deliberately shrinks these, and the
/// widths are the entire subject of this file.
const HIDDEN: usize = 6656;
const INTER: usize = 19968;

/// How many output elements are re-derived on the host. The full product is 133 M MACs per
/// projection and a scalar host loop over all of it would dominate the suite for no extra
/// evidence — every output is the same dot at the same width, so a random-ish spread of them
/// tests the same arithmetic. 64 covers the first row, the last, and the tail past every power of
/// two where an indexing bug would live.
const SPOT: usize = 64;

/// Spot indices spread across `n`: the two ends, then a stride that is coprime with every power of
/// two so the sample does not sit on one alignment.
fn spots(n: usize) -> Vec<usize> {
    let mut v = vec![0, n - 1];
    let stride = n / SPOT + 1;
    let mut i = 1usize;
    while v.len() < SPOT {
        v.push((i * stride + i) % n);
        i += 1;
    }
    v
}

// ------------------------------------------------------------------------------------------

/// The three projections at Glimmer's real widths, spot-checked against a host dot product.
///
/// **`132,913,152` elements per weight matrix is the number that matters here.** It is past 2^27,
/// so any index arithmetic that stays in `i32` through a multiply is fine but a `int` product of
/// two dims is not; the launcher takes `n` and `k` separately for that reason and this is what
/// notices if one of them stops being enough.
#[test]
fn the_projections_run_at_glimmers_real_widths() {
    // MEASURED 2026-08-12: 3.27e-6 for gate/up and 5.17e-6 for down. The bar is ~20x the worse
    // of them. It is not derived from an anchor floor — the anchor has no captures at these
    // widths, which is the whole reason this file exists — so the measurement is the only
    // evidence behind it, and saying so is better than a round number that looks principled.
    let bar = 1e-4_f32;
    for (label, n, k) in [("gate/up", INTER, HIDDEN), ("down", HIDDEN, INTER)] {
        let x = fill(k, 1, 1.0);
        let wf = fill(n * k, 2, 1.0);
        let w = to_bf16(&wf);
        let got = gemv_bf16(&x, &w, n, k);
        assert_eq!(got.len(), n, "{label}: output width");
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{label}: non-finite output"
        );

        // Re-derive only `SPOT` of the outputs. Each is the same dot at the same width, so this
        // tests the arithmetic; what it cannot see is an output the kernel never wrote, which is
        // why the two ends are always in the sample and the length is asserted above.
        let (mut worst, mut checked) = (0.0f32, 0);
        for j in spots(n) {
            let want: f32 = (0..k).map(|i| x[i] * from_bf16(w[j * k + i])).sum();
            worst = worst.max((got[j] - want).abs() / want.abs().max(1.0));
            checked += 1;
        }
        assert_eq!(checked, SPOT, "{label}: the spot sample was short");
        assert!(
            worst <= bar,
            "{label}: worst spot error {worst:e} > {bar:e}"
        );
        println!("{label} [{n} x {k}]: worst spot error over {checked} outputs: {worst:e}");
    }
}

/// The composition, end to end at the real widths, against a host MLP.
///
/// Not a golden comparison — see the module header for why the anchor cannot supply one — but it
/// is the shape S3 will wire, and it catches the thing neither piece alone can: feeding `up` where
/// `gate` belongs. `silu(g)*u` and `silu(u)*g` are different functions, both fluent, and the two
/// projections have identical dimensions so nothing about the types objects.
#[test]
fn the_composition_matches_a_host_mlp_and_the_operand_order_matters() {
    let x = fill(HIDDEN, 5, 1.0);
    let wg = to_bf16(&fill(INTER * HIDDEN, 6, 1.0));
    let wu = to_bf16(&fill(INTER * HIDDEN, 7, 1.0));
    let wd = to_bf16(&fill(HIDDEN * INTER, 8, 1.0));

    let host = |gate_first: bool| -> Vec<f32> {
        let dot = |w: &[u16], j: usize, k: usize, v: &[f32]| -> f32 {
            (0..k).map(|i| v[i] * from_bf16(w[j * k + i])).sum()
        };
        let g: Vec<f32> = (0..INTER).map(|j| dot(&wg, j, HIDDEN, &x)).collect();
        let u: Vec<f32> = (0..INTER).map(|j| dot(&wu, j, HIDDEN, &x)).collect();
        let (a, b) = if gate_first { (&g, &u) } else { (&u, &g) };
        let h: Vec<f32> = a
            .iter()
            .zip(b.iter())
            .map(|(av, bv)| (av / (1.0 + (-av).exp())) * bv)
            .collect();
        spots(HIDDEN)
            .iter()
            .map(|&j| dot(&wd, j, INTER, &h))
            .collect()
    };

    let g = gemv_bf16(&x, &wg, INTER, HIDDEN);
    let u = gemv_bf16(&x, &wu, INTER, HIDDEN);
    let out = gemv_bf16(&swiglu(&g, &u), &wd, HIDDEN, INTER);
    let sampled: Vec<f32> = spots(HIDDEN).iter().map(|&j| out[j]).collect();

    let right = host(true);
    let wrong = host(false);
    let r = worst_rel(&sampled, &right);
    println!("mlp composition: worst rel over {SPOT} outputs: {r:e}");
    // MEASURED 2026-08-12 at 4.94e-6 — three chained reductions over 6656 and 19968 terms, each
    // summed in a different order on the two sides. The first draft guessed 5e-2 from "bf16 is
    // three decimal digits", which is four decades of slack nobody would have noticed spending.
    assert!(r <= 1e-4, "worst rel {r:e}");
    let swapped = worst_rel(&wrong, &right);
    assert!(
        swapped > 100.0 * r,
        "silu(gate)*up and silu(up)*gate differ by only {swapped:e}, so this fixture cannot tell \
         the operand order — the two projections have identical shapes and nothing else would"
    );
    println!("operand order: swapping gate and up moves the output by {swapped:e}");
}
