//! **SiTU-GLU on the device, scored against the S2 anchor** — `situ_glu_f32` and nothing
//! else. Ported from `k3:tests/k3_kernels.rs` item 4a (banner at :1481); shared spine —
//! including `situ1`/`host_situ`, which the fp4 expert oracle next door composes — in
//! `tests/k3/mod.rs`.
//!
//! The first k3 item whose fixture needed no anchor regeneration: `SituAndMul` is an
//! `nn.Module`, so its output was already captured as `<mlp>.act_fn`, and its input is
//! `torch.cat([gate_proj(x), up_proj(x)])` — both halves separately captured
//! (`k3:tests/k3_kernels.rs:1482`).
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: no PR-triggered rocm CI arm, no GPU for this port. One mutation in
//! `kernels/common.hpp::situ_glu`: feed the sigmoid the CAPPED gate — `sigmoid(b1·tanh(g/b1))`
//! instead of `sigmoid(g)`. That is the one trap this three-line operator has, and at the
//! shipped `b1 = 4` it agrees to three figures near zero, which is where a spot check looks.
//! [`situ_glu_matches_the_anchor_at_every_mlp`] must go RED — worst sites in the 1e-2 region
//! at the dense MLP, 4.1e-3 at the shared experts — and
//! [`the_situ_sigmoid_takes_the_uncapped_gate`] must stay GREEN, because it perturbs the
//! HOST oracle the same way and the kernel now agrees with its defect arm. Both directions
//! are the proof; a mutation that reddens both changed something other than the sigmoid's
//! argument. `situ_glu_saturates_at_the_product_of_its_betas` must go red independently at
//! `gain = 40`, where the two forms separate hardest.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_backend::hip::launch_situ_glu_f32;

mod common;
mod k3;

use k3::*;

/// Every MLP the anchor captures that runs SiTU-GLU, as `(layer, module, tolerance bucket)`.
///
/// Layer 0's is the DENSE one — `first_k_dense_replace` is 1, so it is the only `mlp` in the
/// file — and the other five are the shared experts, which every MoE layer has. **The routed
/// experts are deliberately absent and cannot be added**: `moe_infer` calls only the experts
/// that won tokens, so which modules fire is routing-dependent and any defect that moved the
/// routing would change the golden's tensor SET rather than its numbers — that gap is closed
/// a different way by `kernel_k3_expert_f4.rs`. **The third field is NOT the same for all
/// six**: the anchor driver's `operator_of` buckets `model.layers.N.mlp.*` as `dense_mlp`
/// and everything under `block_sparse_moe` that is not `routed_expert*` as `moe_route` — so
/// the shared experts are scored against the ROUTER's tolerance. A classification artifact
/// rather than a judgement, flagged rather than worked around: scoring all six against
/// `dense_mlp`'s tighter 9.4e-6 would pass — this fixture lands two orders under either —
/// but it would be asserting a bar the anchor does not set for five of them
/// (`k3:tests/k3_kernels.rs:1489`).
const SITU_MLPS: [(usize, &str, &str); 6] = [
    (0, "mlp", "dense_mlp"),
    (1, "block_sparse_moe.shared_experts", "moe_route"),
    (3, "block_sparse_moe.shared_experts", "moe_route"),
    (12, "block_sparse_moe.shared_experts", "moe_route"),
    (91, "block_sparse_moe.shared_experts", "moe_route"),
    (92, "block_sparse_moe.shared_experts", "moe_route"),
];

/// The worst relative difference `situ_glu_f32` shows against the anchor, over both draws
/// and all six MLPs: salt 2, layer 12. Named because two tests need it — the fixture as its
/// tripwire, and the defect run as the bar a defect has to clear
/// (`k3:tests/k3_kernels.rs:1515`).
const SITU_OBSERVED_WORST: f32 = 1.454e-7;

/// One MLP's activation boundary: the two projections in, what `SituAndMul` made of them.
struct Situ {
    gate: Vec<f32>,
    up: Vec<f32>,
    want: Vec<f32>,
}

fn situ(g: &GoldenSet, layer: usize, module: &str) -> Situ {
    let m = format!("model.layers.{layer}.{module}");
    let (gs, gate) = float(g, &format!("{m}.gate_proj"));
    let (us, up) = float(g, &format!("{m}.up_proj"));
    let (os, want) = float(g, &format!("{m}.act_fn"));
    assert_eq!(gs, us, "{m}: the two projections are the same width");
    assert_eq!(
        os, gs,
        "{m}: SiTU-GLU is width-preserving over one half of its input"
    );
    let [gate, up, want] = [gate, up, want].map(<[f32]>::to_vec);
    Situ { gate, up, want }
}

fn for_each_situ(mut f: impl FnMut(&str, usize, f32, (f32, f32), Situ)) {
    for (salt, bytes) in GOLDENS {
        let g = load(bytes);
        let b = betas(&g);
        for (layer, module, operator) in SITU_MLPS {
            f(
                salt,
                layer,
                tolerance::rel_tolerance(operator),
                b,
                situ(&g, layer, module),
            );
        }
    }
}

/// One launch, returning the launcher's own `Result`. `alias_h` points `h` at `g` instead of
/// at its own buffer — the contract the launcher documents and the layer loop is invited to
/// rely on. Every path goes through here — the scoring path, the guard test and the aliasing
/// test — which is the shape that makes the guard and aliasing tests mean anything: they
/// exercise the entry point callers use, not a second spelling of it
/// (`k3:tests/k3_kernels.rs:1611`).
fn situ_launch(
    gate: &[f32],
    up: &[f32],
    (b1, b2): (f32, f32),
    alias_h: bool,
) -> anyhow::Result<Vec<f32>> {
    let s = stream();
    let (mut gb, ub) = (dev(&f32b(gate)), dev(&f32b(up)));
    let mut hb = zeros(gate.len().max(1) * 4);
    let h = if alias_h {
        gb.ptr_mut() as *mut f32
    } else {
        hb.ptr_mut() as *mut f32
    };
    // SAFETY: `g`, `u` and `h` are each `n` live f32; `h` either aliases `g` — which the
    // launcher permits, every thread reading both operands at `i` before writing `i` — or is
    // its own buffer. All outlive the stream, which `back` synchronises on.
    unsafe {
        launch_situ_glu_f32(
            gb.ptr() as *const f32,
            ub.ptr() as *const f32,
            gate.len(),
            b1,
            b2,
            h,
            s.raw(),
        )
    }?;
    Ok(f32v(&back(if alias_h { &gb } else { &hb })))
}

/// **The kernel reproduces every SiTU-GLU the anchor captured, at both draws.**
#[test]
fn situ_glu_matches_the_anchor_at_every_mlp() {
    for_each_situ(|salt, layer, tol, betas, s| {
        let r = rel(
            &ok(situ_launch(&s.gate, &s.up, betas, false), "situ_glu_f32"),
            &s.want,
        );
        assert!(r <= tol, "{salt} layer {layer}: {r:e} exceeds {tol:e}");
        // Measured worst over both draws and all six MLPs: 1.454e-7, at salt 2 layer 12 —
        // 65x under `dense_mlp`'s 9.4e-6, the tightest tolerance in the table and the one
        // that binds the only cell here the anchor actually buckets that way.
        tripwire(
            r,
            Bars {
                tol,
                observed: SITU_OBSERVED_WORST,
            },
            &format!("{salt} layer {layer}"),
        );
    });
}

/// **The sigmoid takes the UNCAPPED gate, and the capped version is a different function.**
///
/// `a = b1·tanh(g/b1)·sigmoid(g)`: the first factor saturates at ±b1, the second at 0/1, and
/// they saturate on different scales. Feeding `b1·tanh(g/b1)` to the sigmoid instead is
/// smooth, monotone, bounded and wrong. Scored on the anchor's own values rather than
/// synthetically, so the separation is the one the real activations produce; per SITE,
/// because a single worst-over-all would be carried by whichever MLP separates best. Scored
/// against the bar this fixture ENFORCES — the tripwire — rather than the operator
/// tolerance, and the distinction is the finding of this test: at the shared experts the
/// separation is 4.10e-3 against `moe_route`'s 6.0e-4, only 6.8x — under the 30x the table
/// requires — so **the bucket tolerance could not be relied on to catch a capped-sigmoid
/// SiTU there**; the tripwire can, by 2,800x (`k3:tests/k3_kernels.rs:1673`).
#[test]
fn the_situ_sigmoid_takes_the_uncapped_gate() {
    for_each_situ(|salt, layer, tol, betas, s| {
        let moved = rel(&host_situ(&s.gate, &s.up, betas, true), &s.want);
        let bar = SITU_OBSERVED_WORST * 10.0;
        assert!(
            moved > bar * 30.0,
            "{salt} layer {layer}: capping the sigmoid's argument moved the activation by \
             only {moved:e}, under the {:e} this fixture would need to clear its own {bar:e} \
             tripwire by the table's 30x — so it does not price the one trap SiTU-GLU has, \
             and the agreement above says nothing about which form the kernel implements. \
             (The operator tolerance here is {tol:e}.)",
            bar * 30.0
        );
    });
}

/// **SiTU-GLU at K3's real widths, and at magnitudes the goldens cannot reach.**
///
/// Three gaps the anchor leaves. Width: `moe_intermediate_size` is 24 there against a real
/// 3072, and `intermediate_size` 256 against 33792. Magnitude: the captured activations are
/// ~1, so neither `tanh` saturates and `expf(-g)` never overflows — the whole point of the
/// two betas is what happens when they do. And the BOUND: `|y| <= b1·b2 = 100` is the
/// property §8 states, and nothing in the goldens comes near it
/// (`k3:tests/k3_kernels.rs:1716`).
#[test]
fn situ_glu_saturates_at_the_product_of_its_betas() {
    let mut r = Lcg(0x517A);
    for &(n, gain) in &[
        (3072usize, 1.0f32),
        (33792, 1.0),
        (3072, 40.0),
        (3072, 400.0),
        (1, 1.0),
    ] {
        let gate: Vec<f32> = (0..n).map(|_| r.f() * gain).collect();
        let up: Vec<f32> = (0..n).map(|_| r.f() * gain).collect();
        let got = ok(
            situ_launch(&gate, &up, SHIPPED_BETAS, false),
            "situ_glu_f32",
        );
        let d = rel(&got, &host_situ(&gate, &up, SHIPPED_BETAS, false));
        // 10x the 1.721e-7 measured over these cases. `tanhf` and `expf` are the device's
        // own against Rust's f64 ones, so this is the one fixture here whose bound is a libm
        // difference rather than a reassociated sum. The `gain = 400` case measures SMALLER
        // (7.6e-8), not larger: both `tanh`s are hard against ±1 there and the sigmoid
        // against 0 or 1, so the saturated regime is the easy one and the interesting
        // magnitudes are the ones in between — stated because "we tested the extreme" reads
        // as coverage and here it is the opposite (`k3:tests/k3_kernels.rs:1738`).
        assert!(d <= 1.72e-6, "n={n} gain={gain}: {d:e}");
        let cap = SHIPPED_BETAS.0 * SHIPPED_BETAS.1;
        assert!(
            got.iter().all(|y| y.abs() <= cap),
            "n={n} gain={gain}: SiTU-GLU exceeded |b1·b2| = {cap}, which §8 states as a \
             property of the function rather than of its inputs"
        );
    }
}

/// **`h` may alias `g`, and the layer loop is invited to rely on it.**
///
/// Both the kernel comment and the launcher doc promise the aliasing — it saves a scratch
/// buffer per layer — and `situ_launch` says outright that the scoring fixtures do not use
/// it. So the one property the caller is being offered would otherwise be the one property
/// untested. Every thread reads both operands at `i` and then writes `i`, so the write
/// depends only on reads that thread has already made; that is the argument, and this is the
/// check (`k3:tests/k3_kernels.rs:1788`).
#[test]
fn the_situ_output_may_alias_its_gate() {
    let mut r = Lcg(0x51_7A_A1);
    let n = 3072;
    let gate: Vec<f32> = (0..n).map(|_| r.f() * 4.0).collect();
    let up: Vec<f32> = (0..n).map(|_| r.f() * 4.0).collect();
    let separate = ok(
        situ_launch(&gate, &up, SHIPPED_BETAS, false),
        "situ_glu_f32",
    );
    let aliased = ok(
        situ_launch(&gate, &up, SHIPPED_BETAS, true),
        "situ_glu_f32 aliased",
    );
    assert_eq!(
        aliased, separate,
        "aliasing `h` onto `g` changed the result, so the launcher's documented contract is \
         false and the layer loop must not take it"
    );
}

/// **A zero-length launch is refused (1001), not silently successful** — a launch of nothing
/// that returns success is indistinguishable from work done, to its caller.
#[test]
fn the_situ_launcher_refuses_an_empty_range() {
    common::assert_guard(
        situ_launch(&[], &[], SHIPPED_BETAS, false),
        Some(1001),
        "situ_glu_f32 over n = 0",
    );
}

/// **Both betas must be finite and positive — every other value is refused, not clamped.**
///
/// Seven failure modes, each quiet in its own way, argued at the launcher; the shared
/// [`assert_betas_guarded`] also holds `moe_expert_range_f4_situ` to the same table, which
/// is what makes the two launchers' "same code, same expression" claims checkable rather
/// than aspirational (`k3:tests/k3_kernels.rs:1825`).
#[test]
fn the_situ_betas_are_guarded() {
    assert_betas_guarded("situ_glu_f32", |b1, b2| {
        situ_launch(&[1.0, 2.0], &[3.0, 4.0], (b1, b2), false).is_err()
    });
}
