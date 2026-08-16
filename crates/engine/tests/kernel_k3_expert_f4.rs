//! **Kimi-K3's fused fp4 routed expert on the device, against a composed host oracle** —
//! `moe_expert_range_f4_situ` and nothing else. Ported from `k3:tests/k3_kernels.rs` item 4b
//! (banner at :1839); the upload machinery is `tests/v4_moe/mod.rs`'s `F4Experts`, shared
//! with the V4 twin because the descriptor layout IS the same six device pointers.
//!
//! The routed experts are **the one K3 operator with no anchor fixture and no way to get
//! one**: `.experts` is unhooked in the anchor driver on purpose — `moe_infer` calls only
//! the experts that won tokens, so which modules fire is routing-dependent, and any defect
//! that moved the routing would change the golden's tensor SET rather than its numbers. So
//! this is scored against a host oracle composed of parts that are each pinned somewhere
//! else (`k3:tests/k3_kernels.rs:1839`):
//!
//! * the fp4 **layout** — the k3 tree's `repack-one-expert.md` converted one real K3 expert
//!   with 0 bytes differing, independently of rivoli's code;
//! * the fp4 **codes** — `WMat::Fp4::row`, whose decode is the transliteration the V4 path
//!   scores against DeepSeek's own reference (low nibble = even k, stated at the decoder);
//! * the **activation** — `k3::host_situ`, which `kernel_k3_situ.rs` scores against the
//!   first-party reference at both draws and six MLPs.
//!
//! What is left for THIS fixture to check is what the K3 kernel actually adds: that the two
//! passes compose in the reference's order, with the routing weight where the reference puts
//! it — after `w2`'s bf16 store, not folded into `h` (V4's placement).
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: no PR-triggered rocm CI arm, no GPU for this port. Two mutations in
//! `kernels/moe_f4.hip`:
//!
//! * In the K3 pass-2 body, move the routing weight INSIDE the bf16 store —
//!   `rbf16(dv * w)` for `rbf16(dv) * w`, which is V4's arrangement.
//!   [`the_fp4_expert_pair_matches_the_host_oracle`] must go RED on the differing-slot
//!   COUNT (a folded weight moves essentially every slot; a rounding-boundary crossing
//!   moves ~0.08%), and [`folding_the_routing_weight_before_w2_is_not_the_same_function`]
//!   must stay GREEN — it is host-vs-host and does not touch the kernel.
//! * In `dot_f4_wave_r`, force the eighth nibble to zero when `base != 0`. Only the
//!   `(2, 64, 3072)` case may go red — pass 2 at the real reduction depth is the only cell
//!   here that enters the dword fast path more than once; the V4 twin recorded 65x over
//!   tolerance for exactly this injection (`kernel_v4_moe.rs`'s red-proof plan).
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_backend::hip::{ExpertDescF4, launch_moe_expert_range_f4_situ};
use rivoli_core::num::{bf16_to_f32, f32_to_bf16};
use rivoli_oracles::v4oracle::layer::ExpertW;
use rivoli_oracles::v4oracle::weights::WMat;

mod common;
mod k3;
mod v4_moe;

use k3::*;
use v4_moe::{F4Experts, Wiring};

/// Packed nibbles and group-32 e8m0 scales for one `[rows][cols]` fp4 matrix, as a
/// [`WMat::Fp4`] so the host side decodes through the SAME transliteration the V4 oracle is
/// scored by, and the device side uploads through `F4Experts` unchanged.
///
/// Codes are drawn rather than quantized from floats: the fixture's question is what the
/// kernel does with a given set of codes, and going through an encoder would make a wrong
/// encoder look like a right kernel. Scales stay in `0x70..=0x7a` — the band the real K3
/// checkpoint's scale bytes measured in `repack-one-expert.md` actually hold (11 distinct
/// codes), so this is the shipped range and not a convenient one
/// (`k3:tests/k3_kernels.rs:1872`).
fn f4_matrix(r: &mut Lcg, rows: usize, cols: usize) -> WMat {
    let groups = cols / 32;
    let w = (0..rows * cols / 2)
        .map(|_| ((r.f() * 0.5 + 0.5) * 255.0) as u8)
        .collect();
    let s = (0..rows * groups)
        .map(|_| 0x70u8 + (((r.f() * 0.5 + 0.5) * 10.99) as u8).min(10))
        .collect();
    WMat::Fp4 { rows, cols, w, s }
}

/// `w[row][*]` decoded to f64 — [`WMat::Fp4::row`] widened, so the nibble order, the group
/// stride and the e8m0 decode are the oracle crate's own and not a fourth spelling. The
/// decode is exact in f32 (e2m1 magnitudes times power-of-two scales), so the widening loses
/// nothing.
fn f4_row(m: &WMat, row: usize) -> Vec<f64> {
    let mut buf = Vec::new();
    m.row(row, &mut buf);
    buf.into_iter().map(f64::from).collect()
}

/// Where the routing weight is applied. **A named enum, not a bool**, because the two are
/// references rather than settings and a call site has to say which model it is talking
/// about (`k3:tests/k3_kernels.rs:1914`).
#[derive(Clone, Copy, PartialEq)]
enum WeightAt {
    /// Kimi-K3: `moe_infer` ends `.type(topk_weight.dtype).mul_(topk_weight).sum(dim=1)` on
    /// the expert's full output — the weight multiplies the **bf16 `w2` output**.
    AfterW2,
    /// DeepSeek-V4: `Expert.forward` does `weights * x` and THEN `x.to(dtype)` in front of
    /// `w2`, so the weight is inside the bf16 store that feeds the down projection — what
    /// `moe_gateup_f4_impl` computes, correctly, for V4.
    FoldedIntoH,
}

/// `common.hpp::rbf16` on the host: round-to-nearest-even into bf16, back to f32.
fn bf16(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

/// `common.hpp::moe_fixed` on the host — saturate, then round at scale `2^44`.
///
/// The multiply is f64 where the kernel's is f32; that difference is the systematic low-bit
/// disagreement the f32 projection at the device-vs-host scoring sites absorbs
/// (`k3:tests/k3_kernels.rs:2014`).
fn fixed44(v: f32) -> i64 {
    const MAX: f32 = (1u64 << 14) as f32; // 2^(58-44), the clamp that keeps 16 terms in an i64
    (f64::from(v.clamp(-MAX, MAX)) * ((1u64 << 44) as f64)).round() as i64
}

/// One synthetic expert set, its input, and its routing weights — quantities that always
/// travel together are one type (`k3:tests/k3_kernels.rs:2141`), and the type is also what
/// keeps [`F4Case::host_expert`]'s signature at four arguments instead of the six bare ones
/// the k3 tree's free function carried.
struct F4Case {
    experts: Vec<ExpertW>,
    x: Vec<f32>,
    inter: usize,
    weights: Vec<f32>,
}

impl F4Case {
    /// The reference's expert `e`, in f64 where it can be and at the kernel's rounding
    /// points where it must be. §6, and every step is a placement the port could get wrong:
    /// `g = w1·x`, `u = w3·x` (f64 dots over decoded fp4) → `h = bf16(situ(g, u))` →
    /// `dv = w2·h` (f64) → `bf16(dv)` → **times the routing weight** → fixed point at
    /// `2^-44`.
    ///
    /// The three bf16 points are the reference's dtype boundaries, not rivoli's choices; the
    /// fixed point is rivoli's declared deviation (`MOE_ACC_SHIFT 44`, associative so the
    /// sum stops depending on stream order). **`at` is the defect flag** — a defect run is
    /// the correct oracle with ONE thing changed; a second function was tried in the k3 tree
    /// and its jscpd rejected it at 98 tokens (`k3:tests/k3_kernels.rs:1938`).
    fn host_expert(&self, e: usize, at: WeightAt, acc: &mut [i64]) {
        let (ew, weight) = (&self.experts[e], self.weights[e]);
        let dot = |w: &[f64]| -> f32 {
            w.iter()
                .zip(&self.x)
                .map(|(&wi, &xi)| wi * f64::from(xi))
                .sum::<f64>() as f32
        };
        let folded = at == WeightAt::FoldedIntoH;
        let h: Vec<f32> = (0..self.inter)
            .map(|j| {
                let g = dot(&f4_row(&ew.w1, j));
                let u = dot(&f4_row(&ew.w3, j));
                let y = situ1(g, u, SHIPPED_BETAS, false);
                // The fold happens BEFORE this bf16 store when it happens at all — V4's
                // `weights * x` then `x.to(dtype)`. That is the whole difference, and it is
                // one multiply's worth of position.
                bf16(if folded { y * weight } else { y })
            })
            .collect();
        for (o, slot) in acc.iter_mut().enumerate() {
            let row = f4_row(&ew.w2, o);
            let dv: f64 = row
                .iter()
                .zip(&h)
                .map(|(&wi, &hi)| wi * f64::from(hi))
                .sum();
            // `bf16(dv)` THEN the weight, for K3 — `w2`'s output is bf16 and the multiply
            // comes after.
            let y = bf16(dv as f32);
            *slot += fixed44(if folded { y } else { y * weight });
        }
    }
}

/// `weights` deliberately includes a **zero**: a row that did not route to an expert is the
/// case pass 2's `w != 0.0f` skip exists for, and it is the one an "every expert
/// contributes" fixture never reaches (`k3:tests/k3_kernels.rs:2153`).
fn f4_case(n_experts: usize, expert_in: usize, inter: usize) -> F4Case {
    let mut r = Lcg(0xF4_51_70);
    let experts: Vec<ExpertW> = (0..n_experts)
        .map(|_| ExpertW {
            w1: f4_matrix(&mut r, inter, expert_in),
            w2: f4_matrix(&mut r, expert_in, inter),
            w3: f4_matrix(&mut r, inter, expert_in),
        })
        .collect();
    let x: Vec<f32> = (0..expert_in).map(|_| r.f()).collect();
    let weights: Vec<f32> = (0..n_experts)
        .map(|i| if i == 1 { 0.0 } else { 0.5 + 0.5 * r.f() })
        .collect();
    F4Case {
        experts,
        x,
        inter,
        weights,
    }
}

/// One descriptor RANGE summed into a fixed-point accumulator, the way the launcher does.
///
/// The range is a parameter rather than "all of them" because the launcher's whole reason
/// for existing is that the layer loop calls it with an OFFSET — and in the k3 tree every
/// call site pinned `e_start` to 0 until review 2026-08-12 found `e = e_start + r / inter`,
/// `descs[e]` and `h_out[e * inter + j]` exercised only in their degenerate form
/// (`k3:tests/k3_kernels.rs:1993`).
fn host_acc(c: &F4Case, at: WeightAt, r: std::ops::Range<usize>) -> Vec<i64> {
    let mut acc = vec![0i64; c.x.len()];
    for e in r {
        c.host_expert(e, at, &mut acc);
    }
    acc
}

/// One launch, returning the launcher's own `Result` and the raw fixed-point accumulator.
///
/// The accumulator is read as raw bytes and reinterpreted, NOT drained through
/// `moe_acc_drain_to`: draining is a second kernel with its own `2^-44` multiply, and
/// folding it in would make a pass-2 placement error and a drain error indistinguishable —
/// that kernel has its own oracle (`k3:tests/k3_kernels.rs:2026`). The upload is
/// `v4_moe::F4Experts`, so the nibble-swap red-proof rides the SAME `Wiring` the V4 twin
/// proved, and the descriptor addresses are taken before the buffers move — the dangling-
/// descriptor hazard is argued once, there.
fn expert_launch(
    c: &F4Case,
    r: std::ops::Range<usize>,
    wiring: Wiring,
    (b1, b2): (f32, f32),
) -> anyhow::Result<Vec<i64>> {
    let refs: Vec<&ExpertW> = c.experts.iter().collect();
    let e = F4Experts::upload(&refs, wiring);
    let s = stream();
    let expert_in = c.x.len();
    let (xb, wb) = (dev(&f32b(&c.x)), dev(&f32b(&c.weights)));
    let mut hb = zeros(e.n * c.inter * 4);
    let mut ab = zeros(expert_in * 8);
    // SAFETY: every span is sized as the launcher's `# Safety` requires — `x` is `expert_in`
    // f32 (16-byte aligned: `dev` allocates whole device buffers), `descs` holds `e.n`
    // entries whose spans cover `inter x expert_in` fp4 and their group-32 scales, `wexpert`
    // is one f32 per descriptor, `h` is `e.n * inter` f32 (whole-buffer allocation, so
    // 16-byte aligned too), `acc` is `expert_in` u64, and `x`/`h` do not alias. All outlive
    // the stream, which `back` synchronises on; rejected cases return before any launch.
    unsafe {
        launch_moe_expert_range_f4_situ(
            xb.ptr() as *const f32,
            expert_in,
            c.inter,
            r.start,
            r.len(),
            e.n,
            e.descs.ptr() as *const ExpertDescF4,
            wb.ptr() as *const f32,
            b1,
            b2,
            hb.ptr_mut() as *mut f32,
            ab.ptr_mut() as *mut u64,
            1,
            s.raw(),
        )
    }?;
    Ok(back(&ab)
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn device_expert_f4(c: &F4Case, r: std::ops::Range<usize>, wiring: Wiring) -> Vec<i64> {
    ok(
        expert_launch(c, r, wiring, SHIPPED_BETAS),
        "moe_expert_range_f4_situ",
    )
}

/// Two fixed-point accumulators as f32, with the number of slots that differ.
///
/// **The f32 projection is load-bearing at the device-vs-host sites and inert at the
/// host-vs-host one**, which is why the count is taken HERE, after it: the kernel multiplies
/// and rounds in f32 where [`fixed44`] does it in f64, so a systematic low-bit disagreement
/// would otherwise put every slot in the count. Near 2^44 the projection merges ~2^20
/// distinct i64 values, so it can only ever LOWER a count — the host-vs-host test compares
/// the i64s directly for that reason (`k3:tests/k3_kernels.rs:2122`).
fn slot_diff(got: &[i64], want: &[i64], scale: f32) -> (Vec<f32>, Vec<f32>, usize) {
    let (g, w): (Vec<f32>, Vec<f32>) = (
        got.iter().map(|&v| v as f32 * scale).collect(),
        want.iter().map(|&v| v as f32 * scale).collect(),
    );
    let d = g.iter().zip(&w).filter(|(a, b)| a != b).count();
    (g, w, d)
}

/// One bf16 ulp relative — bf16 carries 8 mantissa bits, so `2^-8` is the quantum every
/// element of the reference's output is rounded to (the same derivation
/// `kernel_v4_attend.rs`'s `ATTEND_TOL` and `v4_moe::TOL` carry).
const BF16_ULP: f32 = 1.0 / 256.0;

/// **The fused fp4 expert pair reproduces the reference's composition, at K3's real
/// reduction depths.**
#[test]
fn the_fp4_expert_pair_matches_the_host_oracle() {
    // `(n_experts, expert_in, inter)`, chosen to put each pass's REDUCTION at its real depth
    // without paying for the full matrix. The real expert is `expert_in = 3584` (the latent
    // width) by `inter = 3072` (`moe_intermediate_size`), top-16 of 896; the whole shape was
    // tried in the k3 tree and ABANDONED — the host oracle is ~220M f64 operations, over ten
    // minutes on the dev profile. The depths are what the arithmetic depends on — pass 1
    // reduces over `expert_in`, pass 2 over `inter`, each output element an independent wave
    // — so one case per real depth covers the reassociation and the row counts only exercise
    // the grid mapping. The third case is the smallest legal pair, one F4_GROUP each, where
    // an index error presents as an index error rather than as a small number
    // (`k3:tests/k3_kernels.rs:2186`).
    for &(ne, expert_in, inter) in &[(2usize, 3584usize, 32usize), (2, 64, 3072), (3, 64, 32)] {
        let c = f4_case(ne, expert_in, inter);
        let got = device_expert_f4(&c, 0..ne, Wiring::Correct);
        let want = host_acc(&c, WeightAt::AfterW2, 0..ne);
        // Scaled back out of fixed point, so the ulp bound is in the units bf16 rounds in.
        let (gf, wf, differing) = slot_diff(&got, &want, 1.0 / (1u64 << 44) as f32);
        // **Scored as "how MANY elements disagree" AND a one-ulp ceiling, and the choice is
        // forced by a measurement this repo already has**: a correct wave-reduced kernel
        // differs from its f64 oracle on ~0.08% of bf16 elements (`common/asserts.rs`
        // records it), because the kernel's f32 dot and the oracle's f64 one occasionally
        // land on opposite sides of a bf16 rounding boundary. A tight bound rejects correct
        // code the first time a draw puts an element on a boundary; a loose bound admits
        // 3.9e-3 of anything, including a folded routing weight on a handful of elements.
        // Together they discriminate: no element may differ by more than ONE bf16 ulp, and
        // the number that differ at all stays inside `2 + len/100`. **The `2 +` is not
        // padding** — a pure percentage was tried first in the k3 tree and failed at
        // `expert_in = 64` on 1 differing element out of 64 (1.6% over a 1% rule, where
        // 0.08% of 64 is 0.05 elements); a rate bound is unusable at small n, and small n is
        // exactly where the index-error case lives (`k3:tests/k3_kernels.rs:2211`).
        let d = rel(&gf, &wf);
        assert!(
            d <= BF16_ULP,
            "ne={ne} expert_in={expert_in} inter={inter}: {d:e} exceeds one bf16 ulp \
             ({BF16_ULP:e}), which is larger than a rounding-boundary crossing can be — so \
             this is a composition error, not reassociation"
        );
        assert!(
            differing <= 2 + gf.len() / 100,
            "ne={ne} expert_in={expert_in} inter={inter}: {differing} of {} elements differ. \
             Each is inside a bf16 ulp, so no single one is wrong — but a boundary crossing \
             is ~0.08% of elements and this is {:.1}%, above the {} this case allows: the \
             shape of a SYSTEMATIC difference, i.e. a folded routing weight, a swapped \
             nibble order, or a rounding at the wrong point.",
            gf.len(),
            100.0 * differing as f64 / gf.len() as f64,
            2 + gf.len() / 100
        );
    }
}

/// **The routing weight belongs AFTER `w2`, and folding it in before is a different
/// function.**
///
/// The whole reason the K3 kernel needed a down-pass variant. `w2` is linear, so `w2(w·h)`
/// and `w·w2(h)` agree in exact arithmetic and the fold reads as a free reassociation of
/// V4's arrangement. It is not free, and this measures by how much: a **bf16 store sits
/// between the two passes** and pass 2 sums in **fixed point**, so `bf16(sw·w)` is not
/// `bf16(sw)` scaled afterwards. Computed on the host both ways, because the device has only
/// the correct one — which is the point: a defect run that needed a second kernel would be
/// pricing a kernel nobody will ship (`k3:tests/k3_kernels.rs:2257`).
#[test]
fn folding_the_routing_weight_before_w2_is_not_the_same_function() {
    let (ne, expert_in, inter) = (3usize, 64usize, 32usize);
    let c = f4_case(ne, expert_in, inter);
    let after = host_acc(&c, WeightAt::AfterW2, 0..ne);
    let before = host_acc(&c, WeightAt::FoldedIntoH, 0..ne);
    // **The i64s, not their f32 projection.** Both sides are host results, so nothing here
    // needs the projection the device-vs-host sites do — and near 2^44 an f32 collapses
    // ~2^20 distinct accumulator values onto one, which can only LOWER this count.
    let differing = after.iter().zip(&before).filter(|(p, q)| p != q).count();
    let (a, b): (Vec<f32>, Vec<f32>) = (
        after.iter().map(|&v| v as f32).collect(),
        before.iter().map(|&v| v as f32).collect(),
    );
    // Two claims, and the second is the one that matters: if the placements differed on one
    // element out of 64 this would be a rounding accident and the fold defensible. A
    // MAJORITY differing is what makes the placement a property of the arithmetic.
    assert!(
        differing * 2 > a.len(),
        "folding the routing weight before `w2` changed only {differing} of {} accumulator \
         slots, so at this geometry the two placements are within rounding of each other and \
         this test is not pricing the difference the down-pass variant was written for",
        a.len()
    );
    assert!(
        rel(&b, &a) > 1.0e-4,
        "the two weight placements agree to {:e} — see above",
        rel(&b, &a)
    );
    assert!(
        after.iter().any(|&v| v != 0) && before.iter().any(|&v| v != 0),
        "one of the two placements produced an all-zero accumulator, which would satisfy the \
         majority-differ count above without either oracle having computed anything"
    );
}

/// **The low nibble is the EVEN element, and no statistic can tell.**
///
/// §9 states it and `WMat::Fp4`'s decode carries the same line. Swapping it is a permutation
/// INSIDE each 32-element scale group, so group boundaries, the amax/scale relation and the
/// code histogram are all invariant — which is why this needs an end-to-end fixture rather
/// than a check on the bytes. Run against the DEVICE with swapped bytes
/// (`v4_moe::Wiring::SwapNibbles`, the same experiment the V4 twin runs), so what is proved
/// is that this fixture would catch a kernel that decoded the other way, not merely that two
/// host functions differ (`k3:tests/k3_kernels.rs:2303`).
#[test]
fn the_fp4_nibble_order_is_the_even_low_one() {
    let (ne, expert_in, inter) = (3usize, 64usize, 32usize);
    let c = f4_case(ne, expert_in, inter);
    let want = host_acc(&c, WeightAt::AfterW2, 0..ne);
    let got = device_expert_f4(&c, 0..ne, Wiring::SwapNibbles);
    let (g, _w, differing) = slot_diff(&got, &want, 1.0);
    assert!(
        differing * 2 > g.len(),
        "exchanging every weight byte's nibbles changed only {differing} of {} slots — so \
         this fixture cannot see the one property of the fp4 layout that no statistic can \
         check",
        g.len()
    );
    // A kernel that wrote NOTHING would satisfy the count above, since `want` is non-zero
    // and every zero slot then counts as differing — carry the anti-vacuity clause rather
    // than lean on another test passing in the same binary.
    assert!(
        got.iter().any(|&v| v != 0),
        "the swapped-nibble launch produced an all-zero accumulator, so the count above is \
         measuring an absent kernel rather than a decoded one"
    );
}

/// **The descriptor range is an OFFSET window, and every other test here starts it at
/// zero.**
///
/// The K3 pass-1 kernel is a hand-written body rather than an instantiation of the shared
/// `moe_gateup_f4_impl`, and it re-derives all three of `e = e_start + r / inter`,
/// `descs[e]` and `h_out[e * inter + j]`. With `e_start` pinned to 0 those agree with the
/// wrong arithmetic — an offset dropped, or applied to the descriptors but not to `h` — and
/// the layer loop is exactly the caller that will pass an offset, since the whole point of a
/// `_range` entry point is a window into 896 experts (`k3:tests/k3_kernels.rs:2338`).
#[test]
fn the_expert_range_is_a_window_and_not_the_whole_set() {
    let (ne, expert_in, inter) = (4usize, 64usize, 32usize);
    let c = f4_case(ne, expert_in, inter);
    // Skip the first descriptor. `weights[1]` is the deliberate zero, so this window also
    // keeps the `w != 0.0f` skip in play rather than testing only weighted experts.
    let got = device_expert_f4(&c, 1..ne, Wiring::Correct);
    let want = host_acc(&c, WeightAt::AfterW2, 1..ne);
    let (g, w, differing) = slot_diff(&got, &want, 1.0);
    assert!(
        rel(&g, &w) <= BF16_ULP && differing <= 2 + g.len() / 100,
        "the offset range disagrees with its oracle: {:e} over {differing} of {} slots",
        rel(&g, &w),
        g.len()
    );
    // And the window is not the whole set: scoring it against ALL FOUR experts must FAIL, or
    // the test above would pass for a kernel that ignored `e_start` entirely.
    let all = host_acc(&c, WeightAt::AfterW2, 0..ne);
    let a: Vec<f32> = all.iter().map(|&v| v as f32).collect();
    assert!(
        rel(&g, &a) > BF16_ULP,
        "experts 1..{ne} sum to the same accumulator as 0..{ne}, so this fixture cannot tell \
         a window from the whole set and says nothing about `e_start`"
    );
}

/// **The shape guards, by CODE — the F4_GROUP alignment above all.**
///
/// 48 is not a multiple of 32 and is a plausible wrong value rather than a silly one: it is
/// a multiple of 16, which is what the packed-nibble stride alone would need, and `f4_ng`'s
/// ceil-vs-floor agreement with `WMat::Fp4::row` depends on both dims being 0 mod F4_GROUP.
/// The looser check is itself the finding — the V4 twin requires `ACT_QUANT_BLOCK` (128),
/// and K3's 3584 and 3072 are both 0 mod 128, so keeping the inherited guard would have been
/// a constraint nothing measured and nothing needed (`k3:tests/k3_kernels.rs:2374`).
#[test]
fn the_expert_shape_guards_refuse_by_code() {
    let go = |(ne, expert_in, inter, r): (usize, usize, usize, std::ops::Range<usize>)| {
        expert_launch(
            &f4_case(ne, expert_in, inter),
            r,
            Wiring::Correct,
            SHIPPED_BETAS,
        )
    };
    assert_guards([
        (
            1002,
            "expert_in not a whole F4 group",
            go((2, 48, 32, 0..2)),
        ),
        (1002, "inter not a whole F4 group", go((2, 64, 48, 0..2))),
        // One past the descriptor array. `wexpert` and `h` are sized by the DESCRIPTOR
        // index, so a range past `n_desc` is a read off the end of both, refused rather
        // than defined.
        (
            1004,
            "range past the descriptor array",
            go((2, 64, 32, 1..3)),
        ),
        (
            1004,
            "e_start past the descriptor array",
            go((2, 64, 32, 2..3)),
        ),
    ]);
    // Not refusing everything: the shipped alignment is accepted.
    assert!(
        go((2, 64, 32, 0..2)).is_ok(),
        "a 32-aligned geometry was refused, so the refusals above carry no information"
    );
}

/// **The K3 expert launcher refuses the same beta table as `situ_glu_f32`** — its guards are
/// its own (betas at 1006 where the V4 sibling takes a `swiglu_limit`), and the shared
/// assertion is what makes the two kernels' "same code, same expression" claims checkable
/// (`k3:tests/k3_kernels.rs:2399`).
#[test]
fn the_k3_expert_betas_are_guarded() {
    let c = f4_case(2, 64, 32);
    assert_betas_guarded("moe_expert_range_f4_situ", |b1, b2| {
        expert_launch(&c, 0..2, Wiring::Correct, (b1, b2)).is_err()
    });
}
