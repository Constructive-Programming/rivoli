//! **The two properties read off the goldens' VALUES rather than their shapes: the QK-norm's axis,
//! and Q's scale against K's.**
//!
//! One of the four binaries the Muse Glimmer S1b anchor gate is split across — `glimmer_anchor.rs`
//! carries the framing and the byte pins, `glimmer_anchor_common/mod.rs` the tables and accessors,
//! and `glimmer_anchor_text.rs` everything about the text goldens that is a shape or a count. What
//! is here is different in kind: each of these two folds the captured floats and asserts a fact
//! about the ARITHMETIC the reference performed, which every shape check in the tree passes over
//! in silence.
//!
//! Both run over both text goldens, because one draw cannot show that a property is a fact about
//! the arithmetic rather than about the numbers it landed on.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#[path = "glimmer_anchor_common/mod.rs"]
mod anchor; // keep this preamble blank-line-free: spread out, the four are a jscpd clone
use anchor::{GoldenSet, Vendored, Widths, each_text, golden_read, ints, meta_usize, num, real};
use serde_json::Value;

// ------------------------------------------------------------------------------------------

/// The worst `|mean(y²) − 1|` over every contiguous `d`-element run, and how many runs that was.
///
/// A weightless RMS over `d` leaves `mean(y²) = mean(x²)/(mean(x²)+eps)`, i.e. 1 to within
/// `eps/mean(x²)`. That is a property only of a norm taken over THAT axis: a reference (or a port)
/// normalising over the whole hidden state, over rows, or over the head COUNT leaves runs whose
/// mean square is anything else, and every shape check still passes.
fn worst_unit_mean_square(vals: &[f32], d: usize) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut runs = 0usize;
    for row in vals.chunks(d) {
        let m = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / d as f64;
        worst = worst.max((m - 1.0).abs());
        runs += 1;
    }
    (worst, runs)
}

/// Every `qk_norm` capture of one text golden, folded to the worst deviation and the row count.
///
/// The head count and `head_dim` are read off each capture's own shape first, so a golden whose
/// axes moved fails here rather than being folded under the wrong reading.
fn fold_qk_norm_rows(v: &Vendored, g: &GoldenSet, c: &Value, w: Widths) -> (f64, usize) {
    let (mut worst, mut runs) = (0.0f64, 0usize);
    for t in 0..=meta_usize(g, "decode_steps") {
        for l in 0..num(c, "num_hidden_layers") {
            for (side, heads) in [("q", w.heads), ("k", w.kv)] {
                let name = format!("t{t}.L{l}.qk_norm.{side}");
                let (shape, vals) = golden_read::float(g, &name);
                assert_eq!(shape[1], heads, "{}: {name} head count", v.name);
                assert_eq!(shape[3], w.head_dim, "{}: {name} head_dim", v.name);
                let (dev, n) = worst_unit_mean_square(vals, w.head_dim);
                worst = worst.max(dev);
                runs += n;
            }
        }
    }
    (worst, runs)
}

/// **The weightless QK-norm's AXIS, from the goldens' own bytes.**
///
/// Every contiguous `head_dim` run of every `qk_norm.*` capture must have unit mean square — see
/// `worst_unit_mean_square` for why that is a fact about the axis and nothing else.
///
/// **Here rather than beside the kernel, because it needs no device.** `tests/glimmer_qk_norm.rs` is
/// `#![cfg(feature = "rocm")]` end to end, and the only automated job in this repo is the FEATURELESS
/// `host` one — so a golden-property check parked in that file is checked exactly as often as
/// someone runs a GPU build by hand. Review, 2026-08-12.
///
/// **MEASURED: 2,304 head rows, worst |mean(y²) − 1| = 8.106e-4**, which is the eps term and nothing
/// else at the reference's own activation scale.
#[test]
fn the_qk_norm_captures_are_normalised_over_head_dim_per_head() {
    let (mut worst, mut runs) = (0.0f64, 0usize);
    each_text(|v, g, c, w| {
        let (dev, n) = fold_qk_norm_rows(v, g, c, w);
        worst = worst.max(dev);
        runs += n;
    });
    println!("{runs} head rows, worst |mean(y^2) - 1| = {worst:e}");
    // The bound is the eps term itself, not a tolerance: a run's mean square falls below 1 by at
    // most eps/mean(x²), which is ~1e-3 at these activations. 1e-2 leaves an order of margin while
    // rejecting any other axis by orders.
    assert!(
        worst < 1e-2,
        "a head_dim run has mean(y^2) off 1 by {worst:e} — the reference did not normalise over \
         this axis"
    );
    // **An absolute, not `runs > 0`.** `layers` comes from `tiny_config` and `steps` from the
    // golden's own metadata, so a census derived from them cannot notice either one shrinking — a
    // re-vendor whose `decode_steps` drops to 1, or a `num_hidden_layers` drifted from
    // `layer_is_sliding`, would leave `worst` a max over fewer rows with every assert still green.
    // That is the rule this tree wrote down for the sandwich norms' 612/34 one item ago, applied
    // here after review pointed out this test had the weak form. 2,304 is what the line above
    // PRINTS on the vendored goldens; no factorisation of it is asserted, because a factorisation
    // would be a second derived count standing in for the absolute.
    assert_eq!(
        runs, 2304,
        "the axis census covered {runs} head rows, not 2,304"
    );
}

// ------------------------------------------------------------------------------------------

/// What the Q/K scale sweep accumulates across every roped layer of every text golden.
///
/// `skipped` is carried rather than dropped because the NoPE skip below is what makes the sweep
/// silent on those layers, and a skip that quietly covered EVERY layer would leave the ratio
/// bounds vacuous — so the test asserts all three counters are non-zero.
struct ScaleSweep {
    lo: f32,
    hi: f32,
    pairs: usize,
    k_elems: usize,
    skipped: usize,
}

impl ScaleSweep {
    /// Every roped layer of one golden, at every step.
    ///
    /// **NoPE layers have no `pre_rope` capture at all** — they skip the rotation (§8) and both
    /// captures are taken inside its wrapper. Reading the name unconditionally is how this first
    /// ran, and `float()` panicked, which is the gate working.
    fn add_golden(&mut self, v: &Vendored, g: &GoldenSet, c: &Value) {
        let layers = num(c, "num_hidden_layers");
        let roped = ints(g, "layer_is_roped").to_vec();
        for t in 0..=meta_usize(g, "decode_steps") {
            for (l, &roped_l) in roped.iter().enumerate().take(layers) {
                if roped_l == 0 {
                    self.skipped += 1;
                } else {
                    self.add_layer(v, g, &format!("t{t}.L{l}"));
                }
            }
        }
    }

    /// One roped layer at one step: Q's ratio against the norm output it came from, and K
    /// bit-identical across the same boundary.
    fn add_layer(&mut self, v: &Vendored, g: &GoldenSet, p: &str) {
        let (_, qn) = golden_read::float(g, &format!("{p}.qk_norm.q"));
        let (_, qs) = golden_read::float(g, &format!("{p}.q.pre_rope"));
        assert_eq!(qn.len(), qs.len(), "{}: {p} q lengths", v.name);
        // Exact zeros carry no ratio, and a zero norm output means the whole head was zero — which
        // says nothing about the scale.
        for (n, s) in qn.iter().zip(qs).filter(|(n, _)| **n != 0.0) {
            let r = s / n;
            self.lo = self.lo.min(r);
            self.hi = self.hi.max(r);
            self.pairs += 1;
        }
        let (_, kn) = golden_read::float(g, &format!("{p}.qk_norm.k"));
        let (_, kp) = golden_read::float(g, &format!("{p}.k.pre_rope"));
        assert_eq!(
            kn, kp,
            "{}: {p} K changed between the norm and the rotation — nothing may scale K (trap 3)",
            v.name
        );
        self.k_elems += kn.len();
    }
}

/// **Trap 3, refuted by the reference's own bytes: Q is scaled by 3.87 and K is not.**
///
/// `q.pre_rope` is captured on entry to `apply_rotary_pos_emb`, i.e. after the norm AND after the
/// scale, while `qk_norm.q` is the norm's output before it — so their ratio is `qk_scale_factor`
/// elementwise. `k.pre_rope` must be BIT-IDENTICAL to `qk_norm.k`.
///
/// **What each half is worth is NOT the same, and review corrected the framing.** The Q ratio is
/// genuinely informative: it falls out of two DIFFERENT tensors and would break if the scale moved or
/// changed value. The K assert is a tautology over these bytes — `modeling_muse_glimmer.py:342`
/// normalises K and line 347 is the next statement touching it, so the forward hook's `out` and the
/// rope tap's `k` are the SAME tensor object serialised twice. It cannot distinguish "the reference
/// does not scale K" from "the harness captured one tensor under two names". Kept, because what it
/// CAN catch is real: **a re-vendor where a future transformers release inserts any op between those
/// two lines.** Call it a tripwire, not a gate.
///
/// **And it constrains the reference, not rivoli.** Trap 3 is a port-side defect; nothing here stops
/// a caller passing 3.87 for K, the anchor has no defect run for that form, and until the layer loop
/// lands there is no call site to gate. `kernels/linalg.hip` says so at the kernel. The
/// `qk_scale_on_k` defect that exists scales `k_proj`'s output, upstream of a norm that cancels a
/// scalar only up to the eps term. Its residue peaks at 6.2e-4 — which this line used to call
/// "nothing", and which is **7.9x the `qk_norm` row's own tolerance**. See `tolerance.rs` for the
/// correction, the closed form, and why the row's exclusion now rests on margin instead.
///
/// **MEASURED: the ratio over 10,368 elements is [3.8699996, 3.8700001]** — f32 rounding on one
/// multiply, not a tolerance — **and 3,456 K elements are unchanged, across 28 roped cases.**
#[test]
fn the_reference_scales_q_by_qk_scale_factor_and_leaves_k_alone() {
    let want = real()["qk_scale_factor"].as_f64().expect("qk_scale_factor") as f32;
    let mut sweep = ScaleSweep {
        lo: f32::INFINITY,
        hi: f32::NEG_INFINITY,
        pairs: 0,
        k_elems: 0,
        skipped: 0,
    };
    each_text(|v, g, c, _| sweep.add_golden(v, g, c));
    let ScaleSweep {
        lo,
        hi,
        pairs,
        k_elems,
        skipped,
    } = sweep;
    println!(
        "q ratio over {pairs} elements: [{lo:.7}, {hi:.7}]; {k_elems} K elements unchanged; \
         {skipped} NoPE cases skipped"
    );
    assert!(
        (lo - want).abs() < 1e-5 && (hi - want).abs() < 1e-5,
        "Q's post-norm scale is [{lo}, {hi}], not the config's {want}"
    );
    // Both kinds of layer must occur, or the skip above is quietly covering everything.
    assert!(
        pairs > 0 && k_elems > 0 && skipped > 0,
        "{pairs} q pairs, {k_elems} k elements, {skipped} skipped — the goldens must carry both \
         roped and NoPE layers"
    );
}
