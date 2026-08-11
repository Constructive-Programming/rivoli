//! **S2 item 2: the RoPE permutation, measured instead of argued.**
//!
//! `glimmer-architecture.md` §6 proposes that Glimmer needs no new rotation kernel. Glimmer uses
//! `rotate_half` — within a head, pairs are `(x[i], x[i+64])` sharing `inv_freq[i]` — while
//! rivoli's `rope_interleave` pairs `(x[2j], x[2j+1])`. §6 argues that permuting `q_proj`/`k_proj`
//! rows at conversion time (`y[2i] = x[i]`, `y[2i+1] = x[i+half]`) makes the existing kernel
//! compute the right thing, and says outright: **"It is an argument, not a measurement — G1b owes
//! it a numeric fixture that reddens when `P` is replaced by identity."** This is that fixture.
//!
//! # What reading the kernel adds to the argument
//!
//! §6 asks only that interleaved RoPE on `P(x)` equal split-half RoPE on `x`, and expects to have
//! to undo `P` afterwards. It does not: `rope_interleave` **reads `(2j, 2j+1)` and writes
//! `(j, half+j)`** (`kernels/linalg.hip`). So with `P` on the input it produces
//!
//! * `v[j]      = x[j]*cos - x[j+half]*sin`
//! * `v[half+j] = x[j+half]*cos + x[j]*sin`
//!
//! which is split-half RoPE **already in split-half layout**. Half of `P` is built into the
//! kernel's write pattern, and the conversion-time row permutation supplies the other half. The
//! frequency matches too: the kernel uses `theta^(-2j/seg)` for pair `j`, and split-half's pair
//! `(j, j+half)` wants `inv_freq[j] = theta^(-2j/d)`.
//!
//! # What it measured (2026-08-12, both goldens, every roped layer, q and k)
//!
//! | | `max|Δ| / max|reference|` |
//! |---|---:|
//! | the permutation, 168 cases | **1.41e-7** |
//! | **the tolerance** (`rope` row) | **4.77e-5** |
//! | identity in its place, weakest of 84 cases | **9.98e-1** |
//!
//! So the permuted kernel sits **34x under the FLOOR** the tolerance is built on, while doing
//! nothing instead sits 20,900x above the tolerance. §6 is settled.
//!
//! Reading is not proof, hence the tests: `the_permutation_makes_the_existing_kernel_compute_glimmers_rope`
//! runs it against the anchor's own `q.roped`/`k.roped`, and
//! `identity_in_place_of_the_permutation_is_caught` is the red proof §6 asked for, run rather than
//! asserted.
//!
//! # What this does NOT establish
//!
//! **Nothing about the conversion.** This applies `P` to a captured activation; the shipped plan is
//! to permute the WEIGHT ROWS so the activation arrives already permuted. §6's argument that the two
//! are equivalent (`v_proj` is never rotated so `o_proj`'s basis is untouched; `gate_proj` acts on
//! the attention output; `qk_norm` is an RMS over all 128 dims and so commutes with a permutation
//! within the head) is still an argument. It becomes a measurement when `convert_glimmer` emits a
//! permuted checkpoint and a golden scores it — S2 item 2's second half.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

use rivoli::backend::hip::{device_sync, launch_rope_interleave};

mod common;
use common::{back, dev, f32b, f32v};

#[path = "common/tolerance.rs"]
mod tolerance;

#[path = "common/glimmer_fixture.rs"]
mod fixture;
use fixture::{Golden, cap, each_case, goldens, present, worst_rel};

/// `tolerance::GLIMMER`'s `rope` row, measured 2026-08-12 before this fixture ran — floor
/// 4.773e-6, weakest targeting defect 1.811e0, `Rel(4.77e-5)`. Same metric as everywhere else in
/// the port: `max|Δ| / max|reference|`.
fn rope_tol() -> f32 {
    match tolerance::tolerance(tolerance::GLIMMER, "rope") {
        Some(tolerance::Policy::Rel(t)) => *t,
        // The kernel computes `cos`/`sin` from `pow` in double and rounds to f32, while the
        // reference builds a table in torch. Those cannot be bit-equal, so an exact-only policy
        // here would not be a stricter version of this test — it would be an unpassable one.
        Some(tolerance::Policy::ExactOnly) => panic!(
            "rope is ExactOnly, which this kernel cannot honour: it derives cos/sin itself rather \
             than reading the reference's table"
        ),
        None => panic!("tolerance::GLIMMER has no `rope` row, so nothing here is scored"),
    }
}

/// Glimmer's rope theta. The REAL value at the tiny widths — the anchor keeps it rather than
/// shrinking it, because a wrong theta is fluent and a small one hides the long-context arg
/// reduction the kernel does in double.
const THETA: f64 = 500000.0;

/// §6's permutation, within each head: `y[2i] = x[i]`, `y[2i+1] = x[i + half]`.
///
/// `flip` swaps it for the identity — the red proof §6 asked for, and the only reason this takes
/// an argument.
fn permute(rows: &[f32], heads: usize, d: usize, flip: bool) -> Vec<f32> {
    let mut out = vec![0.0; rows.len()];
    let half = d / 2;
    for (seg, src) in rows.chunks_exact(d).enumerate() {
        let dst = &mut out[seg * d..][..d];
        if flip {
            dst.copy_from_slice(src);
            continue;
        }
        for i in 0..half {
            dst[2 * i] = src[i];
            dst[2 * i + 1] = src[i + half];
        }
    }
    let _ = heads;
    out
}

/// Run `rope_interleave` over `rows` segments of `d`, one launch per query position.
///
/// One launch per position because the launcher takes a single `pos` — every head of one row
/// rotates by the same angle, and rows differ. At decode that is one launch; at prefill it is
/// `tq` of them, which is what the engine will do too.
fn rope_on_device(data: &[f32], heads: usize, d: usize, start_pos: usize) -> Vec<f32> {
    let buf = dev(&f32b(data));
    let rows = data.len() / (heads * d);
    for r in 0..rows {
        // SAFETY: `buf` holds `rows * heads * d` live f32 and the segment starting at
        // `r * heads * d` covers exactly `heads` segments of `d`, all inside it. In-place is the
        // kernel's contract (all pairs are read before any write, behind a barrier).
        unsafe {
            let base = (buf.ptr() as *mut f32).add(r * heads * d);
            launch_rope_interleave(base, heads, d, d, start_pos + r, THETA)
        }
        .expect("rope_interleave launch");
    }
    device_sync().unwrap();
    f32v(&back(&buf))
}

/// Every (golden, step, layer) that carries a rotation, with q's and k's head counts.
///
/// A layer is roped iff `layer_rope_theta[l] != 0`, which the golden records as `layer_is_roped`.
/// **Asserted against the capture set rather than trusted**: the reference only calls
/// `apply_rotary_pos_emb` on roped layers, so `q.roped` exists exactly where the flag is set, and
/// a disagreement between the two means one of them is lying about which layers rotate.
fn each_roped(mut f: impl FnMut(&Golden, usize, usize, usize)) {
    each_case(|gold, t, l, _win| {
        let roped = fixture::ints_of(gold, "layer_is_roped")[l] != 0;
        let has = present(gold, &format!("t{t}.L{l}.q.roped"));
        assert_eq!(
            roped, has,
            "{}: layer {l} has layer_is_roped={roped} but q.roped present={has}",
            gold.name
        );
        if roped {
            f(gold, t, l, 0);
        }
    });
}

// ------------------------------------------------------------------------------------------

/// **§6's claim, run.** Permute, rotate with the kernel rivoli already ships, compare against the
/// reference's own rotated q and k.
#[test]
fn the_permutation_makes_the_existing_kernel_compute_glimmers_rope() {
    let tol = rope_tol();
    let (mut worst, mut cases) = (0.0f32, 0);
    each_roped(|gold, t, l, _| {
        let (hq, hkv, d) = gold.dims();
        let (tq, start_pos) = gold.geometry(t);
        let p = format!("t{t}.L{l}");
        for (what, heads) in [("q", hq), ("k", hkv)] {
            let pre = cap(
                gold,
                &format!("{p}.{what}.pre_rope"),
                [1, heads, tq, d],
                true,
            );
            let want = cap(gold, &format!("{p}.{what}.roped"), [1, heads, tq, d], true);
            let got = rope_on_device(&permute(&pre, heads, d, false), heads, d, start_pos);
            let r = worst_rel(&got, &want);
            assert!(
                r <= tol,
                "{}: {p}.{what} worst rel {r:e} > {tol:e}",
                gold.name
            );
            worst = worst.max(r);
            cases += 1;
        }
    });
    println!("rope: worst rel over {cases} cases: {worst:e}");
    assert!(
        cases > 0,
        "no roped layer was scored, so this test proved nothing"
    );
}

/// **The red proof §6 asked for.** Identity in place of `P` must not pass.
///
/// Stated as a signal rather than as "it fails": the point is that the disagreement is enormous
/// compared to the tolerance, so no plausible widening of the bar could ever admit it. The two
/// spellings pair the same 128 numbers differently, so this is a wrong ROTATION, not a rounding.
#[test]
fn identity_in_place_of_the_permutation_is_caught() {
    let tol = rope_tol();
    let (mut weakest, mut cases) = (f32::INFINITY, 0);
    each_roped(|gold, t, l, _| {
        let (hq, _, d) = gold.dims();
        let (tq, start_pos) = gold.geometry(t);
        let p = format!("t{t}.L{l}");
        let pre = cap(gold, &format!("{p}.q.pre_rope"), [1, hq, tq, d], true);
        let want = cap(gold, &format!("{p}.q.roped"), [1, hq, tq, d], true);
        let got = rope_on_device(&permute(&pre, hq, d, true), hq, d, start_pos);
        let r = worst_rel(&got, &want);
        assert!(
            r > 1000.0 * tol,
            "{}: {p} identity produced {r:e}, only {:.0}x the tolerance — this fixture cannot \
             tell §6's permutation from doing nothing",
            gold.name,
            r / tol
        );
        weakest = weakest.min(r);
        cases += 1;
    });
    println!("identity: weakest signal over {cases} cases: {weakest:e} against tol {tol:e}");
    assert!(
        cases > 0,
        "no roped layer was scored, so this proof proved nothing"
    );
}

/// A NoPE layer must not be rotated at all, and the flag is asserted rather than defaulted.
///
/// K3's `mla_use_nope` lesson, which `glimmer-port.md` §S2 item 2 names: **assert the flag, never
/// default it.** Trap 1 is reading the top-level `rope_theta` for every layer instead of
/// `layer_rope_theta[i]`, which rotates the 13 NoPE layers and stays fluent. The reference's
/// capture set is the evidence: no `.roped` capture exists where the flag is 0, and `each_roped`
/// cross-checks that in both directions on every case it visits.
#[test]
fn the_nope_layers_are_exactly_the_unroped_ones() {
    let mut seen = (0, 0);
    for gold in &goldens() {
        let roped = fixture::ints_of(gold, "layer_is_roped").to_vec();
        let sliding = fixture::ints_of(gold, "layer_is_sliding").to_vec();
        assert_eq!(
            roped.len(),
            sliding.len(),
            "{}: the two layer tables disagree",
            gold.name
        );
        // §2: the NoPE layers ARE the full-attention layers. Two independently captured tables
        // agreeing on that is worth more than either one alone, and a port that reads the wrong
        // one gets the right answer today and the wrong one on the next model.
        for (l, (r, s)) in roped.iter().zip(&sliding).enumerate() {
            assert_eq!(
                *r, *s,
                "{}: layer {l} is roped={r} but sliding={s}",
                gold.name
            );
        }
        seen.0 += roped.iter().filter(|r| **r == 0).count();
        seen.1 += roped.len();
    }
    assert!(
        seen.0 > 0 && seen.0 < seen.1,
        "the goldens carry {} unroped of {}",
        seen.0,
        seen.1
    );
}
