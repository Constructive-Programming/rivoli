//! **The Hadamard basis order, confronted against the reference rather than inferred.**
//!
//! `src/v4oracle/numerics.rs::hadamard_rotate` shipped carrying S1b's own warning that it is
//! "INFERRED, and the highest-risk inference in this file": `model.py:256` imports
//! `hadamard_transform` from `fast_hadamard_transform`, an external package that is **not**
//! vendored with the checkpoint, so the basis ORDER could not be read off the reference. An
//! oracle is only as good as its transliteration, and agreeing with an unverified oracle
//! proves nothing — so this file does not compare the oracle against itself. It pins
//! `hadamard_rotate` to the **definition the package documents**, and then measures whether
//! the question was ever load-bearing.
//!
//! # The evidence chain, so a later reader need not re-derive it
//!
//! 1. `inference/model.py:253-257` — `rotate_activation` is
//!    `hadamard_transform(x, scale=x.size(-1) ** -0.5)` from `fast_hadamard_transform`.
//!    `inference/requirements.txt` names the package with **no version pin**.
//! 2. `fast_hadamard_transform`'s own `hadamard_transform` docstring, verbatim: *"Multiply
//!    each row of x by the Hadamard transform matrix. **Equivalent to
//!    `F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale`**."* The package ships
//!    that equivalence as executable code — `hadamard_transform_ref` — and its own test
//!    suite (`tests/test_fast_hadamard_transform.py`) asserts the CUDA kernel agrees with it
//!    **elementwise** at `dim = 128` among others. Elementwise agreement is what rules a
//!    permutation out: a reordered output disagrees maximally on random input, nowhere near
//!    the `atol` that test allows. Checked in **both** sdists that could satisfy the
//!    unpinned requirement — 1.0.4.post1 and 1.1.0 — and the docstring and the reference
//!    body are character-identical between them, so the absence of a version pin does not
//!    reopen the question.
//! 3. `scipy.linalg.hadamard`'s docstring and body: *"Constructs an n-by-n Hadamard matrix,
//!    using **Sylvester's construction**"*, implemented as
//!    `H = vstack((hstack((H, H)), hstack((H, -H))))` from `[[1]]` — the natural / Kronecker
//!    order, **not** sequency order. Its docstring pins the n=4 case concretely, and
//!    [`SCIPY_DOCSTRING_H4`] is that literal matrix.
//!
//! So the order is **natural (Sylvester)**, established from the package's documented
//! contract and scipy's source rather than from anybody's recollection of what a "fast
//! Walsh-Hadamard transform" computes. [`hadamard_rotate`] implements that, and
//! [`hadamard_rotate_is_the_sylvester_matrix_and_not_the_sequency_one`] is the gate — one
//! test asserting both halves, because a gate whose negative control lives in a different
//! test can be left green by a rename.
//!
//! # Nothing shipped could see a wrong answer — measured, not supposed
//!
//! `hadamard_rotate` was patched to emit the **sequency** order (gray-code then
//! bit-reversal, a pure permutation that keeps the matrix orthogonal *and* symmetric) and
//! the whole CPU suite re-run on 2026-08-05:
//!
//! | suite | result under the wrong basis |
//! |---|---|
//! | `tests/v4_oracle.rs` (27) | **all pass** |
//! | `tests/kvcompress.rs` (7) | **all pass** |
//! | `tests/kvcompress_probe.rs` (4) — incl. the ranking probe | **all pass** |
//!
//! `hadamard_is_its_own_inverse` passes because a symmetric matrix is involutive whichever
//! order its rows are in — [`both_candidate_matrices_are_symmetric_so_involution_cannot_separate_them`]
//! is that fact made into an assertion. Everything else passes for the more general reason:
//! **every oracle test is self-relative.** A `Defect` arm and its baseline both run through
//! the same `hadamard_rotate`, so a shared error cancels. That is not a flaw in those tests;
//! it is the limit of what comparing an implementation to an oracle can establish, and it is
//! why this file compares the oracle to the *reference's dependency* instead.
//!
//! # And the question was load-bearing
//!
//! [`basis_order_survives_only_through_the_fp4_grouping`] measures the alternative rather
//! than arguing it. Both candidate orders are orthogonal and share the same rows, so
//! `(Hq)·(Hk) = q·k` — the order cannot matter until something groups coordinates, and the
//! only thing that does is `fp4_act_quant`'s block of 32. Measured over 64 row pairs at
//! `index_head_dim`, in the **bf16 score** the reference actually computes: without the fp4
//! step the two orders are **bit-identical on all 64**; with it they differ on **56 of 64**,
//! by a median of **7%** and a maximum of **104%** of the larger score.
//!
//! Those numbers live HERE and in the test that produces them, and nowhere else. An earlier
//! version quoted a figure in "bf16 ulps" from a formula that was one binade wrong, and had
//! copied it into three other documents before review caught it; `numerics.rs` now carries a
//! verdict and a pointer rather than a second copy of a measurement it does not run.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::v4oracle::numerics::{
    bf16_decode, bf16_encode, fp4_act_quant_inplace, hadamard_rotate,
};
use rivoli::v4oracle::weights::{NamedRng, V4Config};

// ---------------------------------------------------------------------------------------
// the two candidate orders, built from their definitions
// ---------------------------------------------------------------------------------------

/// `scipy.linalg.hadamard(4)`, copied from its docstring's own worked example.
///
/// The concrete discriminator, and the reason this is spelled out instead of generated: the
/// **sequency** order of the same four rows is `[1,1,1,1; 1,1,-1,-1; 1,-1,-1,1; 1,-1,1,-1]`,
/// which differs from this in rows 1 and 3. Anyone who reads "fast Walsh-Hadamard transform"
/// and reaches for the sequency ordering lands there. Four rows of four separates them.
const SCIPY_DOCSTRING_H4: [i32; 16] = [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1];

/// `scipy.linalg.hadamard(n)` — Sylvester's construction, transcribed from the source:
/// `H = np.array([[1]]); for _ in range(lg2): H = vstack((hstack((H, H)), hstack((H, -H))))`.
///
/// Row-major `[n * n]` of ±1. Built by the recursion rather than by `(-1)^popcount(i & j)`,
/// which is the same matrix and is exactly the kind of "equivalent" restatement that would
/// let a transcription slip agree with itself: the recursion is what scipy runs.
fn sylvester(n: usize) -> Vec<i32> {
    assert!(n.is_power_of_two());
    let mut h = vec![1i32];
    let mut m = 1usize;
    while m < n {
        let mut next = vec![0i32; 4 * m * m];
        for r in 0..m {
            for c in 0..m {
                let v = h[r * m + c];
                next[r * 2 * m + c] = v;
                next[r * 2 * m + m + c] = v;
                next[(m + r) * 2 * m + c] = v;
                next[(m + r) * 2 * m + m + c] = -v;
            }
        }
        h = next;
        m *= 2;
    }
    h
}

/// The **sequency**-ordered (Walsh) matrix: the same rows, permuted so row `k` has exactly
/// `k` sign changes. This is the rival candidate the oracle's warning names.
///
/// Built by SORTING the Sylvester rows on their sign-change count rather than by a
/// gray-code/bit-reversal index formula. The definition of sequency order *is* "sorted by
/// sign changes", so the sort cannot encode a permutation bug; a formula could, and this
/// matrix is the thing a failing test would send the reader to check.
///
/// The key is asserted to be a PERMUTATION of `0..n` before it is used. `sort_by_key` is
/// stable, so a tie would silently produce a deterministic matrix that is not the Walsh one
/// — an ordering bug that looks exactly like a working function.
fn sequency(n: usize) -> Vec<i32> {
    let h = sylvester(n);
    let changes = |r: usize| {
        (0..n - 1)
            .filter(|&c| h[r * n + c] != h[r * n + c + 1])
            .count()
    };
    let mut keys: Vec<usize> = (0..n).map(changes).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        (0..n).collect::<Vec<_>>(),
        "n={n}: sign-change counts must be 0..n"
    );
    let mut rows: Vec<usize> = (0..n).collect();
    rows.sort_by_key(|&r| changes(r));
    rows.iter()
        .flat_map(|&r| h[r * n..(r + 1) * n].iter().copied())
        .collect()
}

/// `F.linear(x, H) * scale` = `x @ Hᵀ * scale`, the operation the package documents.
///
/// The reference accumulates in **f64** and rounds once at the end. That is deliberate: the
/// fast butterfly and a naive row-dot are different associations of the same sum, so an f32
/// reference would make every comparison a tolerance question about summation order instead
/// of about the basis. With exactly-representable inputs (see [`exact_row`]) the f64 sum is
/// the exact integer and the single rounding is the same one the butterfly performs, which
/// is what lets the gate below assert **bit** identity.
///
/// `n` is derived from `x` rather than passed: a caller pairing a matrix with a row of a
/// different width would otherwise read the wrong rows and still return a plausible vector.
fn linear_by(h: &[i32], x: &[f32], scale: f32) -> Vec<f32> {
    let n = x.len();
    debug_assert_eq!(h.len(), n * n, "matrix must be [{n}, {n}]");
    (0..n)
        .map(|r| {
            let acc: f64 = (0..n)
                .map(|c| f64::from(h[r * n + c]) * f64::from(x[c]))
                .sum();
            (acc as f32) * scale
        })
        .collect()
}

/// A row whose Hadamard transform is exact in f32: small integers, so every partial sum of
/// `n <= 128` of them is an integer below 2^24 and no intermediate rounds.
///
/// This is what makes the gate a **bit** comparison rather than a tolerance one. A tolerance
/// would have admitted a basis that is *nearly* right, and there is no such thing.
fn exact_row(r: &mut NamedRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| (r.unit() * 8.0).trunc()).collect()
}

/// `bf16_decode(bf16_encode(·))` — one bf16 store, the rounding the reference performs
/// between every step of this chain.
fn rbf(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x = bf16_decode(bf16_encode(*x)));
}

/// The width `rotate_activation` is called at, read from the shipped config.
///
/// Read rather than declared as a constant: `assert!(128usize.is_power_of_two())` is
/// constant-folded and cannot fail, which is precisely the tautological-guard shape this
/// repo has shipped twice. Coming from the config, the power-of-two assertion below is a
/// real check on a real value — and it matters, because `hadamard_transform` zero-pads a
/// non-power-of-two `dim` up to the next one, a fourth behaviour to reproduce, while
/// `hadamard_rotate`'s own `debug_assert!` does not run in the `--release` builds this repo
/// tests with.
fn index_head_dim() -> usize {
    let n = V4Config::v4_flash().index_head_dim;
    assert!(
        n.is_power_of_two(),
        "index_head_dim={n} would need the package's zero-padding"
    );
    n
}

// ---------------------------------------------------------------------------------------
// the gates
// ---------------------------------------------------------------------------------------

/// [`sylvester`] reproduces scipy's own documented `hadamard(4)`, and the sequency order
/// does not.
///
/// The second half is what makes the first half mean something: if both orders produced the
/// docstring matrix there would be nothing to be right about.
#[test]
fn sylvester_is_the_scipy_docstring_matrix_and_sequency_is_not() {
    assert_eq!(
        sylvester(4),
        SCIPY_DOCSTRING_H4.to_vec(),
        "scipy.linalg.hadamard(4)"
    );
    assert_ne!(
        sequency(4),
        SCIPY_DOCSTRING_H4.to_vec(),
        "the two candidate orders must be distinguishable at n=4, else this file's whole \
         question is vacuous"
    );
}

/// Both candidate matrices are **symmetric**, and that is what makes
/// `tests/v4_oracle.rs::hadamard_is_its_own_inverse` structurally unable to settle the
/// order.
///
/// Two consequences, and the second is the one worth having in a test:
///
/// 1. `F.linear(x, H)` is `x @ Hᵀ`, so left- versus right-multiplication is not a third
///    candidate to confront — the oracle's warning says so and this checks it.
/// 2. `H·Hᵀ = n·I` holds for **any** Hadamard matrix, so symmetry is exactly the condition
///    under which `H·H = n·I` and the transform is an involution. Since *both* orders are
///    symmetric, the involution property is satisfied by both, and a test built on it agrees
///    with a wrong basis. That is not a criticism of that test — it pins the shape, which is
///    what it claims — but it is why the order needed its own confrontation rather than one
///    more property.
///
/// This asserts a property of two locally-built matrices, which on its own would prove
/// nothing about shipped code. It is kept because the *conclusion* is about shipped code:
/// combined with the gate below, which pins `hadamard_rotate` to `sylvester` bit-for-bit,
/// it says the butterfly's matrix is symmetric too — and that sequency's is as well, which
/// is the half that could not be predicted from the desk and had to be run.
#[test]
fn both_candidate_matrices_are_symmetric_so_involution_cannot_separate_them() {
    let symmetric =
        |h: &[i32], n: usize| (0..n).all(|r| (0..n).all(|c| h[r * n + c] == h[c * n + r]));
    for lg in 0..=7 {
        let n = 1usize << lg;
        assert!(
            symmetric(&sylvester(n), n),
            "n={n}: Sylvester order is not symmetric"
        );
        assert!(
            symmetric(&sequency(n), n),
            "n={n}: sequency order is not symmetric"
        );
    }
}

/// **The gate, with its own negative control.** `hadamard_rotate` is
/// `x @ scipy.linalg.hadamard(n) * n^-0.5` bit for bit at every width from 1 to
/// `index_head_dim`, and is bit-for-bit **not** the sequency order at every width where the
/// two differ.
///
/// Both halves in one test on purpose. They were two tests, and the negative control did not
/// call `hadamard_rotate` at all — it compared two locally-built matrices to each other and
/// so would have stayed green under the very defect it claimed to catch. All three reviews
/// found it. Written this way the control cannot drift away from the thing it controls.
///
/// What would have to be true for this to fail: `hadamard_rotate`'s butterfly computes a
/// different matrix from Sylvester's construction — a permuted basis, a missing or extra
/// stage, a wrong scale. Proved to fire on 2026-08-05 by patching `hadamard_rotate` to emit
/// the sequency order (gray code then bit reversal): this test failed and the five other
/// tests in the file, none of which consume `hadamard_rotate`, stayed green.
///
/// `n <= 2` is asserted SILENT rather than skipped. There the two orderings are the same
/// matrix — there is nothing to permute — so a control that fired there would be reacting to
/// something other than the basis, and a model that disagrees everywhere proves nothing.
#[test]
fn hadamard_rotate_is_the_sylvester_matrix_and_not_the_sequency_one() {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut rng = NamedRng::new("hadamard-exact");
    let mut separated = Vec::new();
    for lg in 0..=7 {
        let n = 1usize << lg;
        let x = exact_row(&mut rng, n);
        let scale = (n as f32).sqrt().recip();
        let mut got = x.clone();
        hadamard_rotate(&mut got);

        assert_eq!(
            bits(&linear_by(&sylvester(n), &x, scale)),
            bits(&got),
            "n={n}: hadamard_rotate must be x @ scipy.linalg.hadamard({n}) * {n}^-0.5"
        );

        let seq = linear_by(&sequency(n), &x, scale);
        if sylvester(n) == sequency(n) {
            assert!(n <= 2, "n={n}: the orders may coincide only at n <= 2");
            assert_eq!(
                bits(&seq),
                bits(&got),
                "n={n}: identical matrices cannot disagree"
            );
        } else {
            assert_ne!(
                bits(&seq),
                bits(&got),
                "n={n}: hadamard_rotate must NOT be the sequency order -- this is the \
                 negative control, and if it cannot fail the gate above proves nothing"
            );
            separated.push(n);
        }
    }
    assert_eq!(
        separated,
        vec![4, 8, 16, 32, 64, 128],
        "every width above 2 separates"
    );
    assert!(
        separated.contains(&index_head_dim()),
        "the model's own width must be covered"
    );
}

// ---------------------------------------------------------------------------------------
// was the question load-bearing?
// ---------------------------------------------------------------------------------------

/// **The measurement the confrontation is for.** The basis order reaches the indexer's score
/// through exactly one channel — which coordinates share an fp4 block — and through that
/// channel it changes the score the selection is computed from.
///
/// Measured in the **bf16 score** `Indexer.forward` actually computes (`einsum` → bf16,
/// `relu_` → bf16, `* weights` → bf16, model.py:426-427), because that is the quantity a
/// wrong basis would have to move in order to move anything. Counting distinct bf16 scores
/// needs no ulp arithmetic: an earlier version of this test scored in "bf16 ulps" using
/// `|v| * 2^-8`, which is one binade wrong (8 significand bits give a spacing of `2^-7` of
/// the binade base, not `2^-8` of the value), and the inflated figure had been copied into
/// three other documents before review caught it.
///
/// Both halves are asserted, because only the pair is informative:
///
/// * **Without** `fp4_act_quant` the two orders are indistinguishable — bit-identical on all
///   64 pairs. Every candidate is orthogonal and `(Hq)·(Hk) = q·k`, so the dot does not
///   depend on the basis at all. That is the null result, and it bounds where a wrong order
///   could possibly have hurt.
/// * **With** it they differ on **56 of 64** pairs, by a median of **7%** and a maximum of
///   **104%** of the larger score (measured 2026-08-05). One bf16 step is ~0.8% at these
///   magnitudes, so that is not a rounding difference; it is a different ranking input.
///
/// The mechanism, stated so the numbers are not a mystery: a permutation of the output
/// coordinates leaves the dot product alone — it is a sum — but `fp4_act_quant` derives one
/// power-of-two scale per **contiguous** block of 32, so permuting first changes which
/// coordinates share an amax, and a 2-bit-mantissa codec is coarse enough for that to move
/// individual values by a large fraction of themselves.
///
/// The eight pairs that do not move are why this asserts a **count** and a **median** rather
/// than a minimum: a coarse codec on a finite sample coincides sometimes, and a bound built
/// on the luckiest — or the loudest — pair would be a claim about the sample rather than
/// about the effect.
#[test]
fn basis_order_survives_only_through_the_fp4_grouping() {
    let n = index_head_dim();
    let (nat, seq) = (sylvester(n), sequency(n));
    let scale = (n as f32).sqrt().recip();

    // `rotate_activation` asserts a bf16 input, and every producer of these rows — `wq_b`,
    // and the compressor's norm-then-rope — hands it one.
    let row = |name: &str| -> Vec<f32> {
        let mut r = NamedRng::new(name);
        let mut v: Vec<f32> = (0..n).map(|_| r.unit()).collect();
        rbf(&mut v);
        v
    };

    // `Oracle::indexer_spread` exactly (forward.rs:1130-1138): rotate, bf16, fp4 block 32,
    // bf16. The bf16 store BETWEEN the two is not cosmetic — it is what the per-block amax
    // is computed from, and that amax is the entire mechanism under measurement here.
    let spread = |h: &[i32], x: &[f32], fp4: bool| {
        let mut v = linear_by(h, x, scale);
        rbf(&mut v);
        if fp4 {
            fp4_act_quant_inplace(&mut v, 32);
            rbf(&mut v);
        }
        v
    };
    // `einsum(...)` → bf16. One store, as model.py:426 performs.
    let score = |a: &[f32], b: &[f32]| -> f32 {
        bf16_decode(bf16_encode(
            a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>(),
        ))
    };

    // `|dn - ds| / max(|dn|, |ds|)`, which lies in [0, 2] and needs no floor. Dividing by
    // `|dn|` alone reported 2500% on one pair, because that pair's natural-order score
    // happened to land near zero — a statement about the denominator, not about the effect.
    let (mut null_moved, mut rels) = (0usize, Vec::new());
    const PAIRS: usize = 64;
    for i in 0..PAIRS {
        let (q, k) = (row(&format!("q{i}")), row(&format!("k{i}")));
        for fp4 in [false, true] {
            let dn = score(&spread(&nat, &q, fp4), &spread(&nat, &k, fp4));
            let ds = score(&spread(&seq, &q, fp4), &spread(&seq, &k, fp4));
            if dn.to_bits() == ds.to_bits() {
                continue;
            }
            match fp4 {
                true => rels.push((dn - ds).abs() / dn.abs().max(ds.abs())),
                false => null_moved += 1,
            }
        }
    }
    rels.sort_by(f32::total_cmp);
    let fp4_moved = rels.len();
    println!(
        "basis order over {PAIRS} row pairs: un-quantized {null_moved} scores differ; \
         fp4 block-32 {fp4_moved} differ, by a median {:.0}% and a max {:.0}% of the larger \
         score",
        rels[fp4_moved / 2] * 100.0,
        rels[fp4_moved - 1] * 100.0
    );

    // The null half. BIT equality, not a tolerance: both orders are orthogonal, so the
    // pre-quantization dot is the same sum in a different summation order, and at these
    // magnitudes one bf16 step is ~1% — far coarser than any f32 re-association. A
    // tolerance here would have been an exact-equality test wearing a tolerance's clothes.
    assert_eq!(
        null_moved, 0,
        "un-quantized, the basis order must be invisible in the bf16 score on every pair"
    );

    // The live half.
    assert!(
        fp4_moved > PAIRS / 2,
        "with fp4 block-32 grouping the basis order must move the bf16 score on most row \
         pairs, else the confrontation above answered a question that did not matter: only \
         {fp4_moved} of {PAIRS}"
    );
    // The MEDIAN over the pairs that moved, not the maximum: a single large outlier would be
    // a claim about one sample. One bf16 step at these magnitudes is ~0.8% of the score, so
    // 5% is comfortably above "the last bit wobbled".
    assert!(
        rels[fp4_moved / 2] > 0.05,
        "and by a real fraction of the score, not a last-bit wobble: median was {:.2}%",
        rels[fp4_moved / 2] * 100.0
    );
}
