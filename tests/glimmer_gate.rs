//! **S2 item 3: the attention output gate, and the operand that decides whether it is the model.**
//!
//! Glimmer multiplies the attend output by `sigmoid(gate_proj(h))` before `o_proj`
//! (`glimmer-architecture.md` §4 item 3). The arithmetic is one elementwise product. What this
//! suite exists for is §9 **trap 4**: `gate_proj` consumes the **layer input** `h` — the
//! post-`input_layernorm` activation — and NOT the attention output. A port that gates on `attn`
//! has the right shapes, the right tensor, and the wrong model.
//!
//! The anchor keeps `attn.gate_proj.out` and `attn.o_proj.in_gated` as separate captures for
//! exactly this reason, which is what makes the two spellings distinguishable here at all.
//!
//! # What it measured (2026-08-12, both goldens, every layer and step — 112 cases)
//!
//! | | `max|Δ| / max|reference|` |
//! |---|---:|
//! | the kernel | **1.58e-7** |
//! | **the tolerance** (`o_proj` row) | **8.29e-5** |
//! | gating on the attention output — trap 4, weakest case | **1.67e-1** |
//! | how far apart the two operands are, closest case | 8.92e-1 |
//!
//! 53x under the fp32 floor, against a trap 2,010x the other side of the bar.
//!
//! # What this covers, and what it does not
//!
//! It scores `sigmoid_gate` against `in_gated` given the reference's own `attend.out` and
//! `gate_proj.out`. So it covers the product, the sigmoid and the operand ORDER — and it covers
//! **nothing about where `gate_proj.out` came from**. Whether the engine feeds this kernel a
//! projection of the layer input or a projection of something else is a call-site question that
//! S3 answers; `the_gate_operand_is_not_recoverable_from_the_attend_output` is the closest this
//! stage can come, and it is a statement about the fixture's power rather than about the engine.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

use rivoli::backend::hip::launch_sigmoid_gate;

#[path = "common/glimmer_fixture.rs"]
mod fixture;
use fixture::{cap, dev, each_case, f32b, sync_read, worst_rel};

/// `x *= sigmoid(g)` on the device, returning the product.
fn gate_on_device(x: &[f32], g: &[f32]) -> Vec<f32> {
    assert_eq!(
        x.len(),
        g.len(),
        "the gate and the value must be the same width"
    );
    let xb = dev(&f32b(x));
    let gb = dev(&f32b(g));
    // SAFETY: both buffers hold exactly `x.len()` live f32, they are distinct allocations (the
    // kernel's parameters are `__restrict__`), and both outlive the `device_sync` below. Null
    // stream: this fixture launches one kernel and joins, so there is nothing to order against.
    unsafe {
        launch_sigmoid_gate(
            xb.ptr() as *mut f32,
            gb.ptr() as *const f32,
            x.len(),
            std::ptr::null_mut(),
        )
    }
    .expect("sigmoid_gate launch");
    sync_read(&xb)
}

/// The quantity trap-4 detectability actually rides on. The kernel multiplies by `sigmoid(g)`,
/// so two operands separate the gated outputs by how far apart their SIGMOIDS are — `|g - a|`
/// diverges from that without bound as magnitudes grow (g=20 vs a=14 is 0.43 apart raw and
/// 8.3e-7 apart through the sigmoid, four decades the wrong side of the bar).
fn sigmoids(v: &[f32]) -> Vec<f32> {
    v.iter().map(|g| 1.0 / (1.0 + (-g).exp())).collect()
}

/// The three tensors this stage needs, at one (step, layer): the attend output flattened to the
/// gate's width, the gate projection, and the gated result the reference produced.
///
/// `attend.out` is captured `[1, tq, hq, d]` — already rows-first — and `gate_proj.out` /
/// `in_gated` are `[1, tq, hq*d]`. Flattening the first is a reshape and not a transpose, which
/// is the reason this stage can compare them at all.
fn tensors(gold: &fixture::Golden, t: usize, l: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (hq, _, d) = gold.dims();
    let (tq, _) = gold.geometry(t);
    let p = format!("t{t}.L{l}");
    (
        cap(gold, &format!("{p}.attend.out"), &[1, tq, hq, d], false),
        cap(
            gold,
            &format!("{p}.attn.gate_proj.out"),
            &[1, tq, hq * d],
            false,
        ),
        cap(
            gold,
            &format!("{p}.attn.o_proj.in_gated"),
            &[1, tq, hq * d],
            false,
        ),
    )
}

// ------------------------------------------------------------------------------------------

/// The kernel against the reference, at every layer and step of both goldens.
#[test]
fn the_gate_reproduces_the_reference() {
    let tol = fixture::rel_tolerance("o_proj");
    let (mut worst, mut cases) = (0.0f32, 0);
    each_case(|gold, t, l, _win| {
        let (attn, gate, want) = tensors(gold, t, l);
        let r = worst_rel(&gate_on_device(&attn, &gate), &want);
        assert!(
            r <= tol,
            "{}: t{t}.L{l} worst rel {r:e} > {tol:e}",
            gold.name
        );
        worst = worst.max(r);
        cases += 1;
    });
    println!("gate: worst rel over {cases} cases: {worst:e} against tol {tol:e}");
    assert_eq!(
        cases,
        fixture::expected().0,
        "the loop did not cover every step and layer"
    );
    // A second, tighter bar against this kernel's OWN measurement (1.58e-7, 2026-08-12), at the
    // usual ~20x. The anchor row is 8.29e-5 because the o_proj BUCKET's floor is dominated by
    // the projection's accumulation — 525x above what one multiply produces — so the row alone
    // cannot see this kernel regress by two decades, and the `expf`-not-`__expf` decision the
    // kernel comment insists on would otherwise be ungated.
    assert!(worst <= 3.2e-6, "worst rel {worst:e} > 20x the 2026-08-12 measurement");
}

/// **Trap 4, run.** Gating on the attention output instead of the layer input's projection.
///
/// This is the mistake with the right shapes: `attn` and `gate_proj(h)` are both
/// `[rows][hq*d]`, so the wrong one substitutes cleanly and the model stays fluent. Judged on the
/// WEAKEST case — a trap that is loud at one layer and quiet at another is a trap this fixture
/// does not catch.
#[test]
fn gating_on_the_attention_output_is_caught() {
    let tol = fixture::rel_tolerance("o_proj");
    let (mut weakest, mut cases) = (f32::INFINITY, 0);
    each_case(|gold, t, l, _win| {
        let (attn, _, want) = tensors(gold, t, l);
        // `sigmoid(attn)` in place of `sigmoid(gate_proj(h))` — same width, same kernel.
        let r = worst_rel(&gate_on_device(&attn, &attn), &want);
        weakest = weakest.min(r);
        cases += 1;
    });
    println!("trap 4: weakest signal over {cases} cases: {weakest:e} against tol {tol:e}");
    // Census BEFORE the judgment: `weakest` is INFINITY-seeded, so an empty case set would
    // otherwise pass this red proof vacuously (the fixture header records the 0 == 0 incident).
    assert_eq!(cases, fixture::expected().0, "the trap ran over nothing");
    assert!(
        weakest > 100.0 * tol,
        "gating on the attention output produced only {weakest:e}, {:.0}x the tolerance",
        weakest / tol
    );
}

/// **What the fixture can and cannot say about the operand.**
///
/// The gate projection is not a function of the attend output — it is a different matrix applied
/// to a different tensor — so no amount of scoring `in_gated` can tell the engine WHERE its `g`
/// came from. What it can say is that the two are far apart everywhere IN THE QUANTITY THE
/// KERNEL CONSUMES, which is why the test above has a signal at all. Stated as its own check so
/// the limit is a measured claim rather than a caveat in prose: if `sigmoid(gate_proj.out)` ever
/// tracked `sigmoid(attend.out)` closely, trap 4 would stop being detectable here and this would
/// go red first.
///
/// **Sigmoid space, corrected 2026-08-12 — the first draft compared the RAW tensors, which is a
/// different claim in exactly the regime production moves toward.** `|g - a|` and
/// `|sigmoid(g) - sigmoid(a)|` diverge without bound as magnitudes grow: at g=20 vs a=14 the raw
/// separation is 0.43 (green by 4x) while the trap-4 signal is 8.3e-7 — four decades under the
/// red proof's bar. The goldens sit at |g| <= 1.6 where the two agree to ~2.7x (raw closest
/// 8.92e-1, sigmoid closest measured this date at ~3e-1), so the raw form passed while
/// guaranteeing nothing. The tolerance row's own `gate_disabled` defect models sigmoid(20) —
/// saturation is the reference's expected regime, not a corner.
#[test]
fn the_gate_operand_is_not_recoverable_from_the_attend_output() {
    let (mut closest, mut cases) = (f32::INFINITY, 0);
    each_case(|gold, t, l, _win| {
        let (attn, gate, _) = tensors(gold, t, l);
        closest = closest.min(worst_rel(&sigmoids(&gate), &sigmoids(&attn)));
        cases += 1;
    });
    println!("operand separation (sigmoid space): closest over {cases} cases: {closest:e}");
    // Same census, same reason: `closest` is INFINITY-seeded.
    assert_eq!(cases, fixture::expected().0, "the separation ran over nothing");
    assert!(
        closest > 0.1,
        "sigmoid(gate_proj(h)) and sigmoid(attend.out) agree to {closest:e} somewhere, so trap 4 \
         is not reliably visible from `in_gated` alone"
    );
}

/// The launcher's one guard, driven — `n == 0` must be a code, not a zero-block launch.
#[test]
fn the_launcher_refuses_an_empty_row() {
    let b = dev(&f32b(&[1.0f32; 4]));
    let (x, g) = (b.ptr() as *mut f32, b.ptr() as *const f32);
    // SAFETY: the call must return before any launch — that is what is being asserted — so the
    // aliased pointers are never dereferenced.
    let e = unsafe { launch_sigmoid_gate(x, g, 0, std::ptr::null_mut()) }
        .expect_err("an empty row must be refused")
        .to_string();
    assert!(
        e.contains("argument guard rejected (1001)"),
        "expected guard 1001, got {e:?}"
    );
}
