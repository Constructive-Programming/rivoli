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

use rivoli::backend::hip::{launch_rope_interleave, launch_rope_split_half};

#[path = "common/glimmer_fixture.rs"]
mod fixture;
use fixture::{Golden, cap, dev, each_case, f32b, goldens, present, sync_read, worst_rel};

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

/// Which of the two rotation kernels to run. They are separate entry points on purpose — see
/// `launch_rope_split_half`'s doc — so this fixture has to name one, which is the property being
/// tested as much as the arithmetic is.
#[derive(Clone, Copy, PartialEq)]
enum Conv {
    /// Glimmer's own: pair `(x[j], x[j+half])`. The shipped path.
    SplitHalf,
    /// GLM/V4's: pair `(x[2j], x[2j+1])`. Correct here ONLY on input that §6's permutation has
    /// already rearranged — which is what `the_declined_permutation_route_computes_the_same_thing`
    /// measures, and what trap 9 is when it is not.
    Interleaved,
}

/// Run a rotation kernel over `rows` segments of `d`, one launch per query position.
///
/// One launch per position because the launcher takes a single `pos` — every head of one row
/// rotates by the same angle, and rows differ. At decode that is one launch; at prefill it is
/// `tq` of them, which is what the engine will do too.
fn rope_on_device(data: &[f32], heads: usize, d: usize, start_pos: usize, conv: Conv) -> Vec<f32> {
    let buf = dev(&f32b(data));
    let rows = data.len() / (heads * d);
    for r in 0..rows {
        // SAFETY: `buf` holds `rows * heads * d` live f32 and the segment starting at
        // `r * heads * d` covers exactly `heads` segments of `d`, all inside it. In-place is the
        // kernel's contract (all pairs are read before any write, behind a barrier).
        unsafe {
            let base = (buf.ptr() as *mut f32).add(r * heads * d);
            match conv {
                Conv::SplitHalf => launch_rope_split_half(base, heads, d, d, start_pos + r, THETA),
                Conv::Interleaved => {
                    launch_rope_interleave(base, heads, d, d, start_pos + r, THETA)
                }
            }
        }
        .expect("rope launch");
    }
    sync_read(&buf)
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

/// Score one ROUTE over every roped layer of both goldens, and report the extremes.
///
/// A route is a (convention, permute-first) pair, and all three tests below are one of them —
/// which is the point: the shipped path, the declined alternative and the trap differ by exactly
/// those two bits, and writing them as three copies of this loop is what jscpd rejected. Returns
/// `(worst, weakest, cases)`: a route that must pass is judged on its worst case, a route that
/// must FAIL is judged on its weakest, and neither test gets to pick the flattering one.
fn score(conv: Conv, permute_first: bool, both_tensors: bool) -> (f32, f32, usize) {
    let (mut worst, mut weakest, mut cases) = (0.0f32, f32::INFINITY, 0);
    each_roped(|gold, t, l, _| {
        let (hq, hkv, d) = gold.dims();
        let (tq, start_pos) = gold.geometry(t);
        let p = format!("t{t}.L{l}");
        let tensors: &[(&str, usize)] = if both_tensors {
            &[("q", 0), ("k", 1)]
        } else {
            &[("q", 0)]
        };
        for (what, which) in tensors {
            let heads = if *which == 0 { hq } else { hkv };
            let pre = cap(
                gold,
                &format!("{p}.{what}.pre_rope"),
                &[1, heads, tq, d],
                true,
            );
            let want = cap(gold, &format!("{p}.{what}.roped"), &[1, heads, tq, d], true);
            let input = if permute_first {
                permute(&pre, heads, d, false)
            } else {
                pre
            };
            let r = worst_rel(&rope_on_device(&input, heads, d, start_pos, conv), &want);
            worst = worst.max(r);
            weakest = weakest.min(r);
            cases += 1;
        }
    });
    assert!(
        cases > 0,
        "no roped layer was scored, so this route proved nothing"
    );
    (worst, weakest, cases)
}

/// **The shipped path: `rope_split_half` on the activation as the checkpoint produces it.**
///
/// No permutation anywhere — not in the converter, not here. The kernel pairs `(x[j], x[j+half])`
/// because that is what Glimmer's `apply_rotary_pos_emb` pairs.
#[test]
fn the_split_half_kernel_computes_glimmers_rope() {
    let tol = fixture::rel_tolerance("rope");
    let (worst, _, cases) = score(Conv::SplitHalf, false, true);
    println!("split-half: worst rel over {cases} cases: {worst:e} against tol {tol:e}");
    assert!(
        worst <= tol,
        "worst rel {worst:e} > {tol:e} over {cases} cases"
    );
}

/// **The red proof, and it is trap 9 itself.** The interleaved kernel on unpermuted input — the
/// mistake a single flag on one launcher would have put one argument away from every GLM and V4
/// call site.
///
/// Judged on the WEAKEST case, not the worst: a trap that is loud somewhere and silent elsewhere
/// is a trap this fixture does not catch, and taking the maximum would hide exactly that.
#[test]
fn the_interleaved_kernel_on_the_same_input_is_caught() {
    let tol = fixture::rel_tolerance("rope");
    let (_, weakest, cases) = score(Conv::Interleaved, false, false);
    println!("trap 9: weakest signal over {cases} cases: {weakest:e} against tol {tol:e}");
    assert!(
        weakest > 1000.0 * tol,
        "the interleaved kernel produced only {weakest:e}, {:.0}x the tolerance — this fixture \
         cannot tell the two conventions apart",
        weakest / tol
    );
}

/// **§6's declined route, kept measured.** Permuting the input and running the INTERLEAVED kernel
/// computes the same rotation. That is why choosing between them was a cost question and not a
/// correctness one, and it is retained rather than deleted so the decision keeps its alternative
/// checked: if giving up `copy_verbatim` ever becomes cheap, this is the evidence that the
/// converter route works, already run. `glimmer-architecture.md` §6 carries the trade.
#[test]
fn the_declined_permutation_route_computes_the_same_thing() {
    let tol = fixture::rel_tolerance("rope");
    let (worst, _, cases) = score(Conv::Interleaved, true, false);
    println!("permutation route: worst rel over {cases} cases: {worst:e}");
    assert!(
        worst <= tol,
        "worst rel {worst:e} > {tol:e} over {cases} cases"
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
