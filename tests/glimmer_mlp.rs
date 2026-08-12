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
//! evidence; the arithmetic evidence already existed. What is new is 6656/19968: a launcher
//! guard that refuses these dims, a 19968-term reduction (no golden reduces past 72), a 12.9
//! M-thread grid, or an LDS/occupancy assumption that only holds at small `k` would all be
//! invisible to every existing fixture and to the tiny-width goldens both.
//!
//! > **CORRECTED 2026-08-12, by review.** This paragraph claimed the widths catch "an index
//! > that overflows i32 at `19968*6656 = 132,913,152` elements". Both halves were false: the
//! > product is **132,907,008** (the stated figure was off by exactly 6,144 — GLM's hidden
//! > size, a copy-paste tell), and it is BELOW 2^27 and 16.2x under `i32::MAX`, so no i32
//! > index overflows at these dims — `kvcompress.hip` casts its offsets through `size_t`
//! > besides. The overflow that DOES exist at Glimmer's widths is the one tensor this file
//! > deliberately does not build: `lm_head` at 202048 x 6656 bf16 is 2.69 GB, 1.25x
//! > `i32::MAX` BYTES, untested at every stage until S3 allocates it.
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

/// Spot indices spread across `n`: the two ends, then an ODD stride so the sample hits both
/// parities. The first draft's stride came out even at BOTH widths (106 at 6656, 314 at 19968),
/// so 63 of 64 samples were even-indexed — a defect confined to odd offsets, the shape a bf16
/// packing bug takes, was invisible to a comment claiming the opposite. The distinctness assert
/// is here because `checked == SPOT` upstream could not fail (the loop returns SPOT elements by
/// construction); what CAN vary is collisions, and a collision silently shrinks coverage.
fn spots(n: usize) -> Vec<usize> {
    let mut v = vec![0, n - 1];
    let stride = (n / SPOT) | 1;
    let mut i = 1usize;
    while v.len() < SPOT {
        v.push((i * stride) % n);
        i += 1;
    }
    let distinct: std::collections::HashSet<usize> = v.iter().copied().collect();
    assert_eq!(distinct.len(), v.len(), "spot indices collided at n = {n}");
    v
}

// ------------------------------------------------------------------------------------------

/// The three projections at Glimmer's real widths, spot-checked against a host dot product.
///
/// The number that matters here is the REDUCTION: no golden reduces past 72 terms, and these
/// dots run 6656 and 19968 — plus a 19968-row launch whose grid no fixture had sized.
#[test]
fn the_projections_run_at_glimmers_real_widths() {
    // MEASURED 2026-08-12: 2.03e-6 for gate/up and 2.06e-6 for down (re-measured the same day
    // after the fill's row-aliasing fix — the first figures, 3.27e-6/5.17e-6, were taken on a
    // fill whose weight matrices held only 128 distinct rows). The bar is ~24x the worse of
    // them. It is not derived from an anchor floor — the anchor has no captures at these
    // widths, which is the whole reason this file exists — so the measurement is the only
    // evidence behind it, and saying so is better than a round number that looks principled.
    let bar = 5e-5_f32;
    for (label, n, k) in [("gate/up", INTER, HIDDEN), ("down", HIDDEN, INTER)] {
        let x = fill(k, 1, 1.0);
        let wf = fill(n * k, 2, 1.0);
        let w = to_bf16(&wf);
        let got = gemv_bf16(&x, &w, n, k);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{label}: non-finite output"
        );

        // Re-derive only `SPOT` of the outputs, and score them in `worst_rel` — the port's ONE
        // metric. The first draft used a per-element ratio with a 1.0 floor, which was STRICTER
        // (denominator |want| ~ 27 rather than max|want| ~ 130) but incomparable with every
        // other number in the port; a bar defended in a metric nobody else reports is the
        // absolute-metric mistake this port already corrected once, wearing new clothes.
        let idx = spots(n);
        let sampled: Vec<f32> = idx.iter().map(|&j| got[j]).collect();
        let want: Vec<f32> = idx
            .iter()
            .map(|&j| (0..k).map(|i| x[i] * from_bf16(w[j * k + i])).sum())
            .collect();
        let worst = worst_rel(&sampled, &want);
        assert!(
            worst <= bar,
            "{label}: worst spot error {worst:e} > {bar:e}"
        );
        println!(
            "{label} [{n} x {k}]: worst rel over {} outputs: {worst:e}",
            idx.len()
        );
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

    let dot = |w: &[u16], j: usize, k: usize, v: &[f32]| -> f32 {
        (0..k).map(|i| v[i] * from_bf16(w[j * k + i])).sum()
    };
    // The two projections once, OUTSIDE the closure: they do not depend on the operand order,
    // and the first draft recomputed both per call — 266 M scalar MACs paid twice for no new
    // information, on the dev profile where this suite lives.
    let hg: Vec<f32> = (0..INTER).map(|j| dot(&wg, j, HIDDEN, &x)).collect();
    let hu: Vec<f32> = (0..INTER).map(|j| dot(&wu, j, HIDDEN, &x)).collect();
    let host = |gate_first: bool| -> Vec<f32> {
        let (a, b) = if gate_first { (&hg, &hu) } else { (&hu, &hg) };
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
    let r = worst_rel(&sampled, &right);
    println!("mlp composition: worst rel over {SPOT} outputs: {r:e}");
    // MEASURED 2026-08-12 at 4.04e-6 (post fill fix) — three chained reductions over 6656 and
    // 19968 terms, each summed in a different order on the two sides. The first draft guessed
    // 5e-2 from "bf16 is three decimal digits", four decades of slack nobody would have noticed.
    assert!(r <= 1e-4, "worst rel {r:e}");

    // **The red proof runs on the DEVICE.** The first draft compared `host(false)` to
    // `host(true)` — two host computations, which proves the METRIC discriminates the operand
    // order and says nothing about the subject: a device `swiglu` that ignored its operand
    // order would have passed it. So the swapped arm is the device pipeline with `u` and `g`
    // exchanged, scored against the straight host reference.
    let swapped_out = gemv_bf16(&swiglu(&u, &g), &wd, HIDDEN, INTER);
    let swapped_sampled: Vec<f32> = spots(HIDDEN).iter().map(|&j| swapped_out[j]).collect();
    let swapped = worst_rel(&swapped_sampled, &right);
    assert!(
        swapped > 100.0 * r,
        "the DEVICE silu(up)*gate differs from silu(gate)*up by only {swapped:e}, so this \
         fixture cannot tell the operand order — the two projections have identical shapes and \
         nothing else would"
    );
    println!("operand order: swapping gate and up on the device moves the output by {swapped:e}");
}
