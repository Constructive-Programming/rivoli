//! `kernels/indexer.hip`'s DSA/MISA launchers against host references. Compiles to nothing
//! without rocm.
//!
//! Five of the six kernels in that file were uncovered until 2026-08-06; `index_topk` was
//! not, and its oracle stays in `tests/kernel.rs` where it was written. Named by
//! `tests/kernel_coverage.rs` when the census was re-keyed onto `src/backend/hip.rs`.
//!
//! Not to be confused with `tests/blockindex_kernel.rs`, which scores `kernels/blockindex.hip`
//! — DeepSeek-V4's indexer, a different arithmetic on a different checkpoint. The naming
//! follows the kernel files.
//!
//! `index_append` and `index_pool_push` are held BIT-EXACT and the other three under a
//! relative tolerance; `common::assert_bitwise` carries the argument for which kernels can
//! be held to which. Each tolerance here is stated per assertion, with its reason, rather
//! than shared.
//!
//! **Known hole:** `index_score`'s `bd > 1024` and `lds > 65536` rejects need `nact > 1024`
//! and `hd + bd > 16384`, which no shape here reaches, so neither is exercised.
//! `index_head_route` has no guard arm at all.
//!
//! **Every anti-vacuity arm compares two HOST references**, never the kernel against a
//! perturbed one. A defect-injected oracle scored against the clean oracle involves no code
//! under test, so a small separation can only mean the metric lacks resolution; made
//! kernel-facing, the identical number reads as a tolerance needing widening — and widening
//! it is how a suite loses the defect it was built to catch.
#![cfg(feature = "rocm")]

use rivoli::backend::hip::{
    launch_index_append, launch_index_head_route, launch_index_pool_push, launch_index_score,
    launch_layernorm,
};
use rivoli::indexer::{K_NORM_EPS, MISA_BLOCK};
use rivoli::math::{bf16_to_f32, f32_to_bf16};

mod common;
use common::{
    Lcg, assert_bits, assert_bitwise, assert_rel, back, dev, f32b, f32v, ok, rel, u16b, u16v, zeros,
};

/// `layernorm` — the indexer's `k_norm`, `y = (x-mean)/sqrt(var+eps)·w + b`.
///
/// **The fixture has a non-zero MEAN, and that is deliberate.** `Lcg` is symmetric about
/// zero, so on its raw output `mean ≈ 0`, `(x - mean) ≈ x`, and a kernel that never
/// subtracted the mean at all would agree with this oracle to five digits. The `+1.5`
/// offset is what makes the centring load-bearing; the anti-vacuity arm below measures
/// exactly how much.
///
/// The host uses the kernel's OWN variance formula, `E[x²] − mean²`, not the numerically
/// stable two-pass form. They are different functions in floating point — at this offset
/// the cancellation costs about one digit — and a reference that quietly used the better
/// one would be scoring the kernel against arithmetic it does not perform.
#[test]
fn layernorm_centres_scales_and_biases() {
    let n = 1024;
    let mut r = Lcg(0x1AE6);
    let x: Vec<f32> = (0..n).map(|_| r.f() + 1.5).collect();
    let w: Vec<f32> = (0..n).map(|_| r.f() + 1.0).collect();
    let b: Vec<f32> = (0..n).map(|_| r.f() * 0.25).collect();

    // f64 sums, f32 formula. The double removes the HOST's summation-order error, which is
    // not the thing under test; the shape of the expression is the kernel's, which is.
    let host = |centre: bool, bias: bool| -> Vec<f32> {
        let s: f64 = x.iter().map(|&v| f64::from(v)).sum();
        let sq: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let mean = (s / n as f64) as f32;
        let var = (sq / n as f64) as f32 - mean * mean;
        let inv = 1.0 / (var + K_NORM_EPS).sqrt();
        let m = if centre { mean } else { 0.0 };
        (0..n)
            .map(|i| (x[i] - m) * inv * w[i] + if bias { b[i] } else { 0.0 })
            .collect()
    };
    let want = host(true, true);

    // Anti-vacuity FIRST, and entirely between host references: if dropping the centring or
    // the bias moved the oracle by less than the tolerance below, the kernel-facing
    // assertion could not see either defect and would be reporting nothing about them.
    let (no_centre, no_bias) = (
        rel(&host(false, true), &want),
        rel(&host(true, false), &want),
    );
    println!("layernorm: no-centre separation {no_centre:.3e}, no-bias {no_bias:.3e}");
    assert!(
        no_centre > 1e-2 && no_bias > 1e-2,
        "the fixture cannot resolve a missing mean ({no_centre:.3e}) or bias ({no_bias:.3e})"
    );

    let (xb, wb, bb) = (dev(&f32b(&x)), dev(&f32b(&w)), dev(&f32b(&b)));
    let mut yb = zeros(n * 4);
    // SAFETY: `x`, `w`, `b` and `y` are each `n` live device f32 for the call.
    ok(
        unsafe {
            launch_layernorm(
                xb.ptr() as *const f32,
                wb.ptr() as *const f32,
                bb.ptr() as *const f32,
                n,
                K_NORM_EPS,
                yb.ptr_mut() as *mut f32,
            )
        },
        "layernorm",
    );

    // 1e-4 relative. The floor is set by `E[x²] − mean²`, not by the reduction: at this
    // offset the two terms agree to about one part in four, which costs the variance
    // roughly a digit of the f32 mantissa, and the tolerance has to cover that on top of
    // the LDS tree's order. Both defects above clear it by two orders.
    assert_rel(&want, &f32v(&back(&yb)), "layernorm", 1e-4);
}

/// `index_append` — `kcache[pos·hd + i] = bf16(k[i])`, one row of the key cache.
///
/// Bit-exact against `math::f32_to_bf16`: `common.hpp::f2bf16` claims to mirror it
/// round-to-nearest-even including the non-finite carry, and this is the only thing in the
/// tree that checks the claim on the DSA path. The slab holds seven rows and one is
/// written, so a wrong `pos` stride lands in a row this asserts is still zero.
#[test]
fn index_append_stores_one_bf16_key_row() {
    let (rows, hd, pos) = (7usize, 128usize, 5usize);
    let mut r = Lcg(0xC0DE);
    // Spread over four binades so the RNE rounding is exercised at more than one exponent;
    // bf16 keeps 8 mantissa bits, so a tie-break defect shows up as a one-code error.
    let k: Vec<f32> = (0..hd).map(|_| r.f() * 8.0).collect();
    let mut want = vec![0u16; rows * hd];
    for (i, &v) in k.iter().enumerate() {
        want[pos * hd + i] = f32_to_bf16(v);
    }

    let kb = dev(&f32b(&k));
    let mut cb = zeros(rows * hd * 2);
    // SAFETY: `k` is `hd` live device f32 and the cache holds `rows` rows of `hd` u16 with
    // `pos < rows`.
    ok(
        unsafe { launch_index_append(kb.ptr() as *const f32, cb.ptr_mut() as *mut u16, pos, hd) },
        "index_append",
    );
    assert_bitwise(&want, &u16v(&back(&cb)), "index_append");
}

/// `index_pool_push` — fold token `t`'s key into its MISA block's running mean.
///
/// Bit-exact: `m + (k − m)/(j+1)` is a subtract, a divide and an add per element, and a
/// divide gives an FMA nothing to absorb, so the host runs the identical expression. It is
/// asserted against the incremental form rather than against `Σk/n` — those are different
/// floating-point numbers, and the kernel computes the first.
///
/// The sequence crosses a block boundary on purpose. `in_block == 0` OPENS a block with a
/// plain store and every other index folds, so a kernel that folded on the open would carry
/// the previous block's mean into the new one — which is a slow drift, not a crash, and the
/// third block is asserted untouched to catch the mirror-image error.
#[test]
fn index_pool_push_keeps_the_exact_block_running_mean() {
    let (hd, m_blocks) = (128usize, 3usize);
    // One token before the boundary and two after it — 1023 folds into block 0, 1024 opens
    // block 1. Anything short of that tests only the open.
    let ts = [0usize, 1, 2, MISA_BLOCK - 1, MISA_BLOCK, MISA_BLOCK + 1];
    let mut r = Lcg(0x9111);
    let keys: Vec<Vec<f32>> = ts
        .iter()
        .map(|_| (0..hd).map(|_| r.f()).collect())
        .collect();

    let mut want = vec![0.0f32; m_blocks * hd];
    let mut pool = zeros(m_blocks * hd * 4);
    // Uploaded BEFORE the loop. These launchers are launch-only and nothing syncs between
    // iterations, so a `DeviceBuf` declared inside would `hipFree` while its launch was
    // still reading it — against the launcher's stated contract that device pointers live
    // until the next `device_sync`.
    let kbs: Vec<_> = keys.iter().map(|k| dev(&f32b(k))).collect();
    for ((&t, k), kb) in ts.iter().zip(&keys).zip(&kbs) {
        let (blk, in_blk) = (t / MISA_BLOCK, t % MISA_BLOCK);
        for i in 0..hd {
            let m = &mut want[blk * hd + i];
            *m = match in_blk {
                0 => k[i],
                j => *m + (k[i] - *m) / (j + 1) as f32,
            };
        }
        // SAFETY: `k` is `hd` live device f32 held by `kbs` past the readback; the pool
        // holds `m_blocks` rows of `hd` f32 and `t / MISA_BLOCK < m_blocks` for every `t`.
        ok(
            unsafe {
                launch_index_pool_push(kb.ptr() as *const f32, pool.ptr_mut() as *mut f32, t, hd)
            },
            "index_pool_push",
        );
    }

    // The fold must have MOVED the block-0 mean off the last key it saw, or "the running
    // mean is exact" would be indistinguishable from "the kernel overwrites every time".
    assert!(
        (0..hd).any(|i| want[i] != keys[3][i]),
        "block 0's mean equals its last key — the fold is not being exercised"
    );
    assert_bits(&want, &f32v(&back(&pool)), "index_pool_push");
}

/// `index_score` — `scores[t] = Σ_{h∈active} w[h]·wscale·ReLU((q_h·k_t)·dscale)`.
///
/// Both head-selection modes, because they are not the same code path: `heads == null` is
/// DSA and the kernel uses `h = tid`, while a non-null `heads` is MISA and it indirects
/// through the list. A kernel that ignored `heads` entirely would pass the DSA arm and
/// score the wrong heads in production.
///
/// The MISA arm's head list is deliberately NOT a prefix — `[5, 0, 6, 2]` — so
/// `h = heads[tid]` and `h = tid` disagree on every entry. A `[0, 1, 2, 3]` list would make
/// the indirection invisible, which is the shape this kind of test usually ships in; the
/// break corpus substitutes exactly that, and it moves the scores by 0.64 relative.
#[test]
fn index_score_reduces_over_the_active_heads_only() {
    // `nh = 64`, not 8, and the reason is the LAUNCHER rather than the kernel: it picks
    // `bd = next_pow2(nact)` floored at 32, so `nact` of 8 and 4 BOTH clamp to 32 and run
    // the identical reduction ladder. The DSA arm passes `nact = nh`, which in production is
    // `index_n_heads` (`gpu.rs`), so at 64 this exercises `bd = 64` — a different halving
    // ladder, a longer `part`, and a 768 B dynamic LDS request against 640 B — while the
    // MISA arm still covers the clamped 32. (`part`'s OFFSET is `kf + hd` and does not move
    // with `bd`; only its length and the request do.)
    let (nt, nh, hd) = (64usize, 64usize, 128usize);
    let (wscale, dscale) = (0.75f32, 0.125f32);
    let mut r = Lcg(0x5C04);
    let q: Vec<f32> = (0..nh * hd).map(|_| r.f()).collect();
    let w: Vec<f32> = (0..nh).map(|_| r.f()).collect();
    // The cache is bf16 on device, so the reference reads the DECODED values, not the f32
    // it drew. Scoring against the pre-rounding numbers would charge the kernel for a
    // conversion the engine also performs.
    let kf: Vec<f32> = (0..nt * hd)
        .map(|_| bf16_to_f32(f32_to_bf16(r.f())))
        .collect();

    let host = |active: &[usize]| -> Vec<f32> {
        (0..nt)
            .map(|t| {
                active.iter().fold(0.0f32, |acc, &h| {
                    let dot: f32 = (0..hd).map(|i| q[h * hd + i] * kf[t * hd + i]).sum();
                    acc + w[h] * wscale * (dot * dscale).max(0.0)
                })
            })
            .collect()
    };

    let (qb, wb) = (dev(&f32b(&q)), dev(&f32b(&w)));
    let cache = dev(&u16b(
        &kf.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>(),
    ));
    let run = |heads: *const u32, nact: usize| -> Vec<f32> {
        let mut sb = zeros(nt * 4);
        // SAFETY: `q` is nh·hd f32, `w` nh f32, the cache nt·hd u16, `heads` either null or
        // `nact` live u32, and `scores` nt f32. All live for the call.
        ok(
            unsafe {
                launch_index_score(
                    qb.ptr() as *const f32,
                    wb.ptr() as *const f32,
                    cache.ptr() as *const u16,
                    heads,
                    nt,
                    nh,
                    nact,
                    hd,
                    wscale,
                    dscale,
                    sb.ptr_mut() as *mut f32,
                )
            },
            "index_score",
        );
        f32v(&back(&sb))
    };

    let all: Vec<usize> = (0..nh).collect();
    let subset = [5usize, 0, 6, 2];
    let want_all = host(&all);
    let want_sub = host(&subset);
    // Anti-vacuity, host against host: if the two head sets scored the same, the MISA arm
    // below would pass on a kernel that ignored `heads`.
    let sep = rel(&want_sub, &want_all);
    println!("index_score: DSA-vs-MISA head-set separation {sep:.3e}");
    assert!(
        sep > 1e-2,
        "the two head sets are indistinguishable at this seed"
    );

    // 1e-5 relative. The ReLU'd FACTOR is non-negative but `w[h]` is signed, so the outer
    // sum over heads does cancel — an earlier version of this comment claimed it could not,
    // which was wrong. What makes 1e-5 safe anyway is that the tolerance is stated against
    // the global `max_abs`, not per element: a score that cancels to near zero contributes
    // a near-zero ABSOLUTE error, and the bound never divides by it. What is actually being
    // absorbed is the LDS tree's order against the host's fold plus FMA contraction inside
    // the per-head dot, which the host does not spell — a few times 1e-7 of the scale.
    //
    // The ReLU boundary is not a flakiness source for the same reason: where `dot * dscale`
    // is near zero the two sides may disagree about whether to include the head at all, and
    // the term they disagree about is itself near zero.
    assert_rel(
        &want_all,
        &run(std::ptr::null(), nh),
        "index_score (DSA, all heads)",
        1e-5,
    );
    let hb = dev(&subset
        .iter()
        .flat_map(|&h| (h as u32).to_le_bytes())
        .collect::<Vec<u8>>());
    assert_rel(
        &want_sub,
        &run(hb.ptr() as *const u32, subset.len()),
        "index_score (MISA, 4 of 64 heads)",
        1e-5,
    );
}

/// `index_head_route` — the MISA router estimate `e[j] = mean_b |w[j]·ReLU(q_j·k̄_b)|`.
///
/// The `mean` over blocks is the part worth pinning: `m_blocks` is 5 here, so a kernel that
/// summed without dividing, or divided by the wrong count, is off by a factor the tolerance
/// cannot absorb. The anti-vacuity arm measures that the per-block spread is real — with
/// identical pool rows every block contributes the same value and dividing by anything
/// consistent would agree.
///
/// The absolute value is outside the ReLU in both, which makes it redundant on paper. It is
/// reproduced rather than simplified away: `w[j]` is signed, so `|w·ReLU(·)|` and
/// `w·ReLU(·)` differ on every head with a negative weight, and half of them do.
#[test]
fn index_head_route_averages_over_the_block_pool() {
    // `hd = 384`, above the launcher's 256 block cap, so `for (i = tid; i < hd; i += 256)`
    // runs twice for threads 0..127 and once for 128..255 — a ragged tail as well as a real
    // accumulation. At `hd <= 256` it runs once for EVERY thread and `p += qj[i]*kb[i]` is
    // indistinguishable from `p = qj[i]*kb[i]`; the per-thread accumulation is the only
    // place in this kernel with an FMA chain and a partial-sum order, and the first version
    // of this fixture never entered it.
    let (nh, hd, m_blocks) = (8usize, 384usize, 5usize);
    let mut r = Lcg(0x4074);
    let q: Vec<f32> = (0..nh * hd).map(|_| r.f()).collect();
    let w: Vec<f32> = (0..nh).map(|_| r.f()).collect();
    let pool: Vec<f32> = (0..m_blocks * hd).map(|_| r.f()).collect();
    assert!(
        w.iter().any(|&v| v < 0.0),
        "the |·| arm needs a negative weight"
    );

    let host = |blocks: usize| -> Vec<f32> {
        (0..nh)
            .map(|j| {
                let s = (0..blocks).fold(0.0f32, |acc, b| {
                    let dot: f32 = (0..hd).map(|i| q[j * hd + i] * pool[b * hd + i]).sum();
                    acc + (w[j] * dot.max(0.0)).abs()
                });
                s / blocks as f32
            })
            .collect()
    };
    let want = host(m_blocks);

    // Host against host: the estimate must actually depend on how many blocks are pooled,
    // or "it averaged over 5" and "it averaged over 1" are the same assertion.
    let sep = rel(&host(1), &want);
    println!("index_head_route: 5-block vs 1-block separation {sep:.3e}");
    assert!(
        sep > 1e-2,
        "the pool rows are too alike to resolve the average"
    );

    let (qb, wb, pb) = (dev(&f32b(&q)), dev(&f32b(&w)), dev(&f32b(&pool)));
    let mut eb = zeros(nh * 4);
    // SAFETY: `q` is nh·hd f32, `w` nh f32, `pool` m_blocks·hd f32 and `e` nh f32, all live.
    ok(
        unsafe {
            launch_index_head_route(
                qb.ptr() as *const f32,
                wb.ptr() as *const f32,
                pb.ptr() as *const f32,
                m_blocks,
                nh,
                hd,
                eb.ptr_mut() as *mut f32,
            )
        },
        "index_head_route",
    );

    // 1e-5 relative. Here the outer fold really IS over non-negative terms — the `|·|` is
    // outside the sign of `w[j]`, unlike `index_score`, where only the ReLU'd factor is
    // non-negative — so it cannot cancel at all. The only disagreement is the inner dot's
    // LDS tree order against the host's serial fold, plus contraction the host does not
    // spell.
    assert_rel(&want, &f32v(&back(&eb)), "index_head_route", 1e-5);
}
