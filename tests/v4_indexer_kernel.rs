//! **The V4-Flash indexer's device-side scoring.** S2c-indexer of
//! `docs/investigations/v4-flash-port.md`.
//!
//! The two halves are gated against DIFFERENT references, and the difference matters:
//! `v4_indexer_spread` against S1b's oracle, `v4_indexer_score` against a host
//! transliteration of `model.py` that **the oracle currently contradicts** (see below). A
//! green run of the score half is not a correctness verdict until the oracle's fix lands.
//!
//! Two kernels:
//!
//! * `v4_indexer_spread` — `rotate_activation` then `fp4_act_quant(·, 32)`, the finish
//!   `Compressor.forward` performs when `rotate = true` and the same operation
//!   `Indexer.forward` applies to its `q` rows. It is `Geom::indexer`'s finish, and
//!   `geom_indexer_and_geom_attention_do_not_finish_the_same_way` is the test that the two
//!   cannot be interchanged. Scored against the ORACLE's own `hadamard_rotate` and
//!   `fp4_act_quant_inplace`, on **synthetic** rows built to reach binades real activations
//!   may not.
//! * `v4_indexer_score` — the `einsum` / `relu_` / `weights` / sum chain, on the
//!   **checkpoint's own** compressed KV, scored against a host transliteration of
//!   `model.py:425-427`.
//!
//! **The score comparison is NOT against `Oracle::indexer`, and an earlier version of this
//! header said it was.** The oracle computes that chain internally but exposes neither the
//! roped-and-spread `q` nor the scaled `weights`, so the kernel cannot be handed its
//! intermediates. [`host_score`] states what the substitute costs; making `Oracle::linear`
//! public would close it and is recorded as an open item rather than done here, since that
//! is S1b's file and the gate S3 is scored against.
//!
//! # Why the SCORE matrix and not the selected sets
//!
//! The plan doc records that the shipped goldens are **set-invariant** at
//! `index_topk = 512`: `IndexerNoWeights` and `IndexerNoRelu` both move `.indexer_scores`
//! and leave `.compress_idxs` bit-identical, because the causal mask alone determines the
//! selection until the top-k truncates, which needs >= 2052 tokens. A gate resting on the
//! sets therefore accepts an arbitrarily wrong ranking. `Oracle::indexer` exports the full
//! pre-top-k matrix precisely so a consumer can be scored on it instead, and that comparison
//! is **strictly stronger**: every defect that moves a set moves the scores, and four that
//! move the scores move no set at the shipped configuration. So this file does not
//! reimplement the mask or the top-k, and does not claim to cover them.
//!
//! # What this file provably cannot detect — read before trusting it
//!
//! * **Anything the oracle is also wrong about** — though the one instance this file found
//!   is now CONFIRMED and fixed on this side. `Oracle::indexer` sums the per-head products
//!   as a bf16 RUNNING fold, while `torch.sum` over bf16 accumulates through `acc_type` —
//!   f32 — and rounds ONCE. That is a property of the reduction, measured off-repo against
//!   CPU torch on 2026-08-05 by the coordinator, with no reproducer in this tree. The kernel
//!   and [`host_score`] now both do what torch does and
//!   [`host_score_accumulates_in_f32_not_bf16`] is the in-tree guard; **the oracle's fix is
//!   owned elsewhere and has not landed**, so do not read a comparison against the current
//!   indexer goldens as evidence in either direction.
//! * **The summation ORDER.** The bf16 fold pinned it as a side effect and an f32
//!   accumulator does not; torch's own reduction is vectorized and tree-shaped. The kernel
//!   and [`host_score`] agree with each other exactly, and neither is pinned to torch's
//!   order.
//! * **The basis order** is *not* in this list any more. It was S1b's highest-risk
//!   inference; `tests/v4_hadamard_basis.rs` settled it against `fast_hadamard_transform`'s
//!   own documented contract on 2026-08-05, so the Hadamard this file exercises is pinned to
//!   something other than the oracle's opinion of it.
//! * **`wq_b` and `weights_proj`.** Their GEMVs are S2b's kernels and are scored there; this
//!   file drives the two indexer-specific kernels from host-computed inputs so that a
//!   projection's re-association cannot be mistaken for a scoring error.
//! * **A misreading of `model.py` shared between [`host_score`] and the oracle**, since the
//!   score chain is now transliterated twice. Same class of gap as the one
//!   `v4compress.rs`'s `jscpd:ignore` region names for `freqs_cis`.
//!
//! Skips with a printed reason when the checkpoint is absent; there is no CI here.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::backend::hip::{device_sync, launch_v4_indexer_score, launch_v4_indexer_spread};
use rivoli::memory::device::DeviceBuf;
use rivoli::v4compress::{Geom, LayerKind, Quantize, ScoreDims};
use rivoli::v4oracle::forward::{Counters, Defect, Oracle};
use rivoli::v4oracle::numerics::{bf16_decode, bf16_encode, fp4_act_quant_inplace, hadamard_rotate};
use rivoli::v4oracle::weights::V4Config;

mod common;
use common::{checkpoint, indexer_w, probe};

/// A probe long enough that `end_pos / 4` gives many compressed blocks, so the score matrix
/// has columns whose ranking could be wrong: 16 of them, against the 3 the 13-token emit
/// prompt the shipped goldens use would give.
const SCORE_PROBE_LEN: usize = 64;

// =======================================================================================
// device plumbing
// =======================================================================================

/// Upload, launch, read back. Kept to three lines of intent per call site because the
/// interesting content of this file is the comparison, not the transfer.
fn up(v: &[f32]) -> DeviceBuf {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut b = DeviceBuf::new(bytes.len().max(1)).expect("v4i: device alloc");
    b.copy_in_at(0, &bytes).expect("v4i: upload");
    b
}

/// Read `n` f32 back off the device.
///
/// Decodes through `common::f32v`, i.e. the engine's own `artifact::quant::read_f32`, rather
/// than a local `from_le_bytes` loop -- which is what `build.rs`'s duplication gate found
/// here, correctly: a test that read bytes back differently from the code under test could
/// agree with itself while both were wrong about the layout.
fn down(b: &DeviceBuf, n: usize) -> Vec<f32> {
    let mut bytes = Vec::new();
    b.copy_out_prefix(&mut bytes, n * 4).expect("v4i: readback");
    common::f32v(&bytes)
}

/// Bit-for-bit, and REPORT the first disagreement rather than a count.
///
/// Exact, not a tolerance. Both kernels here are bit-exact against the oracle by
/// construction — the Hadamard visits the same operand pairs in the same order, and the
/// score reduction is deliberately not parallelised and is compiled with
/// `#pragma clang fp contract(off)` — so a tolerance would be admitting something nothing in
/// the design produces. That pragma is load-bearing and was missing until review: without it
/// `dot += q*kv` fuses to `v_fmac_f32`, one rounding per term where the host does two, and
/// every comparison here would have failed for a reason that looks like a numerics bug. If
/// this ever needs loosening, that is a finding about the kernel and belongs in the plan doc,
/// not in the epsilon.
fn assert_bits(want: &[f32], got: &[f32], label: &str) {
    assert_eq!(want.len(), got.len(), "{label}: length");
    let differs = |a: &f32, b: &f32| a.to_bits() != b.to_bits();
    let Some(i) = want.iter().zip(got).position(|(a, b)| differs(a, b)) else {
        println!("{label}: {}/{} bit-identical", want.len(), want.len());
        return;
    };
    // Counted only on the failure path: the first index is what a reader needs, and the
    // total is what says whether it is one boundary flip or a wrong algorithm.
    let n = want.iter().zip(got).filter(|&(a, b)| differs(a, b)).count();
    panic!(
        "{label}: {n}/{} elements differ; first at {i} want={:e} got={:e}",
        want.len(),
        want[i],
        got[i]
    );
}

/// `Oracle::indexer_spread` on the host, over `rows` rows of `d` — the value
/// `v4_indexer_spread` must reproduce.
///
/// Written from the oracle's public primitives rather than by calling `indexer_spread`,
/// which is private. That is not a re-derivation: `hadamard_rotate` and
/// `fp4_act_quant_inplace` ARE the oracle's, so the only thing restated is the order of the
/// four steps — and the bf16 store between the rotation and the quantization is the step a
/// port drops, so having it visible at the comparison site is worth more than hiding it
/// behind one call.
fn host_spread(rows: &mut [f32], d: usize) {
    for row in rows.chunks_exact_mut(d) {
        hadamard_rotate(row);
        row.iter_mut().for_each(|x| *x = bf16_decode(bf16_encode(*x)));
        fp4_act_quant_inplace(row, 32);
        row.iter_mut().for_each(|x| *x = bf16_decode(bf16_encode(*x)));
    }
}


/// Upload, spread in place, read back — the whole device side of one spread comparison.
///
/// One helper rather than two call sites: `build.rs`'s duplication gate found the copy, and
/// it was right about the shape of the risk. The two callers differ only in their fixture,
/// and a divergence in the row count passed to the launcher versus the count read back is
/// exactly the kind of slip that would make a comparison vacuous rather than wrong.
fn device_spread(host_in: &[f32], rows: usize, d: usize) -> Vec<f32> {
    let buf = up(host_in);
    // SAFETY: `buf` is `rows * d` writable f32 by the caller's fixture and outlives the sync.
    unsafe { launch_v4_indexer_spread(buf.ptr() as *mut f32, rows, d, std::ptr::null_mut()) }
        .expect("spread launch");
    device_sync().expect("sync");
    down(&buf, rows * d)
}

// =======================================================================================
// the spread — Geom::indexer's finish
// =======================================================================================

/// `v4_indexer_spread` reproduces `Oracle::indexer_spread` bit for bit.
///
/// Synthetic rows rather than checkpoint ones, and deliberately: what this kernel must get
/// right is the *arithmetic*, and the fixture is built to reach the corners real
/// activations may not — see the amax spread below. The checkpoint's own tensors drive the
/// scoring test.
#[test]
fn indexer_spread_is_bit_identical_to_the_oracle() {
    let d = V4Config::v4_flash().index_head_dim;
    let rows = 9usize;
    // `probe` draws bf16-rounded values in [-1, 1), which is what `rotate_activation`'s
    // `assert x.dtype == torch.bfloat16` guarantees it receives.
    let host_in = probe("spread", rows, d);

    let mut want = host_in.clone();
    host_spread(&mut want, d);

    assert_bits(&want, &device_spread(&host_in, rows, d), "indexer_spread");
}

/// **Widening the e8m0 exponent coverage the whole S2 suite lacks.**
///
/// The plan doc banks it as requirement 15: e8m0 exercises **2 distinct codes of 254**
/// (`119..=120`) across everything shipped, so a decode bug on any other is invisible. Two
/// corrections to the premise, both worth stating because they change what "widening" means
/// here:
///
/// 1. The indexer consumes **no e8m0 scale BYTES at all**. Its weights are fp8 (`wq_b`,
///    which ships a `.scale`) and bf16 (`weights_proj`, `wkv`, `wgate`); there is no packed
///    fp4 weight tensor on this path, so `dot_f4_wave_r`'s `e8m0f` is never called. Checked
///    against the checkpoint index: layer 2 carries exactly seven `attn.indexer.*` tensors
///    and only `wq_b` has a `.scale`.
/// 2. What this path DOES exercise is the same exponent domain through
///    `fast_round_scale` — `fp4_act_quant`'s block scale is a bare power of two, derived by
///    the identical `fast_log2_ceil`/`fast_pow2` bit surgery that produces an e8m0 code.
///
/// So this test sweeps the **scale exponent**, which is the reachable half. Each row's
/// magnitude is set to a different binade across 60 of them, from `2^-30` to `2^29`, so the
/// block scale lands on 60 distinct powers of two rather than the two the shipped fixtures
/// reach. A `fast_round_scale` that is wrong on any of them shows up as a bit difference.
///
/// **What remains uncovered, precisely:** `e8m0f`'s decode of a scale BYTE, including its
/// two named endpoint cases (`0x00` → `2^-127`, `0xff` → NaN). Nothing on the indexer path
/// can reach them, and the plan's requirement 10 — rejecting those two bytes at load — is
/// still the only thing that will.
#[test]
fn fp4_block_scale_covers_sixty_binades_not_two() {
    let d = V4Config::v4_flash().index_head_dim;
    let base = probe("binade", 60, d);
    // Row `r` scaled into binade `r - 30`. `exp2` of an integer is exact, so the fixture
    // adds no rounding of its own and every difference the comparison finds is the kernel's.
    let host_in: Vec<f32> = base
        .iter()
        .enumerate()
        .map(|(i, v)| bf16_decode(bf16_encode(v * (2.0f32).powi(i as i32 / d as i32 - 30))))
        .collect();
    // The fixture must actually SPREAD, or this is the shipped coverage under a new name.
    // A PROXY, and a loose one — stated precisely because the first version of this comment
    // claimed it was "counted from the oracle's own `fast_round_scale` path", which it is
    // not. It uses libm `log2().ceil()` on the PRE-Hadamard, per-ROW amax and without the
    // `x FP4_MAX_INV` factor, where the kernel uses `fast_log2_ceil` on the post-Hadamard,
    // per-BLOCK-of-32 amax with it. `common.hpp` records that those two log2s disagree at
    // binade edges. What the proxy does establish is the only thing needed here: the fixture
    // ROWS span a wide range of magnitudes rather than the one the shipped fixtures sit in.
    // The bit comparison below is what actually gates the arithmetic.
    let binades: std::collections::BTreeSet<i32> = host_in
        .chunks_exact(d)
        .map(|r| r.iter().fold(0.0f32, |m, v| m.max(v.abs())).log2().ceil() as i32)
        .collect();
    assert!(binades.len() >= 55, "the fixture must span many binades, got {}", binades.len());

    let mut want = host_in.clone();
    host_spread(&mut want, d);
    assert_bits(&want, &device_spread(&host_in, 60, d), "spread over 60 binades");
}

/// The launcher refuses the shapes whose failure would be silent, and each refusal is shown
/// to be reachable.
///
/// Every one of these can fire: they are not restatements of a type. A `d` that is not a
/// power of two is what a config change to `index_head_dim` produces, and the reference
/// would zero-pad it rather than transform it — the Hadamard would then run over a length
/// the model never intended and return finite numbers.
#[test]
fn the_spread_launcher_refuses_widths_the_hadamard_cannot_transform() {
    let buf = up(&vec![0.0f32; 1024]);
    let p = buf.ptr() as *mut f32;
    let bad = |rows: usize, d: usize| {
        // SAFETY: refused before any launch; `buf` is 1024 f32, larger than any accepted here.
        unsafe { launch_v4_indexer_spread(p, rows, d, std::ptr::null_mut()) }.is_err()
    };
    assert!(bad(0, 128), "zero rows");
    assert!(bad(1, 0), "zero width");
    assert!(bad(1, 512), "wider than the LDS tile");
    assert!(bad(1, 96), "not a power of two -- the reference would zero-pad this");
    // Guard 1004, and it is NOT unreachable behind the power-of-two check as the kernel's
    // comment claimed until review: 16 is a power of two and is still a ragged fp4 block.
    assert!(bad(1, 16), "a power of two SMALLER than the fp4 block of 32");
    // And the shape the model actually uses is ACCEPTED, so the guards above are not simply
    // refusing everything.
    // SAFETY: `buf` is 1024 f32 >= 1 * 128.
    unsafe { launch_v4_indexer_spread(p, 1, 128, std::ptr::null_mut()) }.expect("128 is legal");
    device_sync().expect("sync");
}

// =======================================================================================
// Geom::indexer — requirement 5
// =======================================================================================

/// **The trap requirement 5 names, made unrepresentable.** A geometry built for the indexer
/// carries the fp4 finish; one built for the attention compressor carries the partial fp8
/// one; and the two are otherwise identical in every dimension a guard could check.
///
/// The point is the last clause. At `head_dim` this is trivially distinguishable, but the
/// indexer's compressor is built at `index_head_dim` with the same `ratio`, `coff` and
/// `rope_head_dim`, so `Geom::attention(Overlap, 128, 64, eps)` and
/// `Geom::indexer(Overlap, 128, 64, eps)` agree on all six integers the kernel sees. Only
/// the [`Quantize`] separates them, which is why it is a field and not an argument to
/// `compress`.
#[test]
fn geom_indexer_and_geom_attention_do_not_finish_the_same_way() {
    let c = V4Config::v4_flash();
    let (hd, rd, eps) = (c.index_head_dim, c.rope_head_dim, c.norm_eps);
    let a = Geom::attention(LayerKind::Overlap, hd, rd, eps).unwrap();
    let i = Geom::indexer(LayerKind::Overlap, hd, rd, eps).unwrap();

    assert_eq!(a.quantize(), Quantize::PartialFp8);
    assert_eq!(i.quantize(), Quantize::HadamardFp4);
    // Identical to the kernel, different to `compress`. This IS entailed by `Geom::build`,
    // whose `GeomAbi` literal does not read `quant` — review flagged it as tautological and
    // it is kept anyway, as the executable statement of the HAZARD rather than as a test of
    // the port: it is the reason no dimension guard can catch the confusion, and if it ever
    // fails the argument for the `Quantize` field has weakened, which is a thing to know.
    // The companion `assert_ne!(a, i)` was deleted; it restated the two lines above it.
    assert_eq!(a.abi(), i.abi(), "the ABI halves must be indistinguishable");

    // An `Indexer` exists ONLY at ratio 4 (model.py:474), so the other two classes cannot
    // produce one -- stricter than `Geom::attention`, which accepts every compressor.
    assert!(Geom::indexer(LayerKind::NonOverlap(128), hd, rd, eps).is_none(), "ratio 128");
    assert!(Geom::indexer(LayerKind::Plain, hd, rd, eps).is_none(), "ratio 0");
    assert!(
        Geom::attention(LayerKind::NonOverlap(128), c.head_dim, rd, eps).is_some(),
        "...while the attention compressor does exist at ratio 128, so the line above is a \
         real restriction and not a broken constructor"
    );
}

// =======================================================================================
// the scoring chain
// =======================================================================================

/// Upload, score, read back. The second helper the duplication gate forced, and the second
/// one worth having: six pointers and a `ScoreDims` at two call sites is six chances for a
/// transposed argument, and `q`/`kv`/`w` are all `*const f32`.
fn device_score(q: &[f32], kv: &[f32], w: &[f32], d: ScoreDims) -> Vec<f32> {
    let (dq, dkv, dw) = (up(q), up(kv), up(w));
    let out = up(&vec![0.0f32; d.s * d.n_comp]);
    // SAFETY: every buffer is sized by `d`, device-resident, non-aliasing, and outlives the
    // sync below.
    unsafe {
        launch_v4_indexer_score(
            dq.ptr() as *const f32,
            dkv.ptr() as *const f32,
            dw.ptr() as *const f32,
            out.ptr() as *mut f32,
            d,
            std::ptr::null_mut(),
        )
    }
    .expect("score launch");
    device_sync().expect("sync");
    down(&out, d.s * d.n_comp)
}

/// `model.py:425-427` on the host — the value `v4_indexer_score` must reproduce.
///
/// **Transcribed from `model.py`, not from `Oracle::indexer`** — and on 2026-08-05 that
/// stopped being only a limitation and became the reason this file is right where the
/// oracle is wrong: `Oracle::indexer` folds the head sum in bf16 per term, where
/// `torch.sum` accumulates in f32 and rounds once. The oracle's fix is owned elsewhere;
/// this reference already matches the measured behaviour, so **the two will disagree until
/// it lands**, and a comparison against the current indexer goldens is not evidence either
/// way.
///
/// The rest of the distinction is still the honest limit of this comparison. The oracle computes this chain inside `indexer`, but
/// exposes neither the roped-and-spread `q` nor the scaled `weights`, so the kernel cannot be
/// handed the oracle's own intermediates and scored on the oracle's own exported matrix.
/// Making `Oracle::linear` public would close that, and it is recorded as an open item rather
/// than done here: it is S1b's file and the gate S3 is scored against.
///
/// What that costs, precisely: this is a second transliteration of the same nine lines, so a
/// shared misreading of `model.py` between it and the oracle is invisible — the same class of
/// gap `v4compress.rs`'s `jscpd:ignore` region names for `freqs_cis`. What it still buys is
/// everything the KERNEL could get wrong on its own: the bf16 store placement, the relu
/// order, the summation order, and the head/block indexing.
fn host_score(q: &[f32], kv: &[f32], w: &[f32], d: ScoreDims) -> Vec<f32> {
    let rbf = |x: f32| bf16_decode(bf16_encode(x));
    let mut out = vec![0.0f32; d.s * d.n_comp];
    for t in 0..d.s {
        for c in 0..d.n_comp {
            let mut acc = 0.0f32;
            for hh in 0..d.heads {
                let qh = &q[(t * d.heads + hh) * d.hd..(t * d.heads + hh + 1) * d.hd];
                let kvc = &kv[c * d.hd..(c + 1) * d.hd];
                // `einsum` -> bf16, `relu_()` in place -> bf16, `* weights` -> bf16.
                // Those three are elementwise tensors the reference materializes, so each
                // store is real.
                let mut dot = rbf(qh.iter().zip(kvc).map(|(a, b)| a * b).sum::<f32>());
                dot = dot.max(0.0);
                // `.sum(dim=2)` accumulates in f32 (`acc_type`) and rounds ONCE. This read
                // a bf16 running fold until 2026-08-05, copying the oracle's error; see the
                // kernel's note for what settled it.
                acc += rbf(dot * w[t * d.heads + hh]);
            }
            out[t * d.n_comp + c] = rbf(acc);
        }
    }
    out
}

/// `v4_indexer_score` is bit-identical to the reference chain, on the checkpoint's own
/// compressed KV.
///
/// The `kv` side is REAL: it is `CompState::cache` after `Oracle::indexer` has run layer 2's
/// indexer on a `SCORE_PROBE_LEN` prompt, so the rows are the genuine output of the pooling
/// plus the Hadamard-and-fp4 finish — the fp4 codebook's eight magnitudes and its zeros as
/// the model actually produces them, not as a fixture imagines them. `q` and `w` are
/// synthetic because the oracle does not expose its own.
#[test]
fn indexer_score_is_bit_identical_on_the_checkpoints_compressed_kv() {
    let Some(ck) = checkpoint() else { return };
    let c = V4Config::v4_flash();
    let n = SCORE_PROBE_LEN;
    let iw = indexer_w(&ck, 2, &c);
    let o = Oracle::new(c.clone(), Defect::None);
    let mut cs = o.fresh_state(2).idx_comp.expect("layer 2 has an indexer");
    let mut ctr = Counters::default();
    o.compressor(&iw.compressor, &mut cs, &probe("l2-x", n, c.dim), n, 0, o.freqs(2), &mut ctr);
    assert_eq!(ctr.compressed_blocks, n / 4, "the probe must fill the compressed cache");

    let d = ScoreDims { s: 3, n_comp: n / 4, heads: c.index_n_heads, hd: c.index_head_dim };
    let kv = cs.cache[..d.n_comp * d.hd].to_vec();
    assert!(kv.iter().any(|v| *v != 0.0), "the cache must not be all zeros");

    // `q` through the same spread the real path applies, so the two sides of the dot are
    // both fp4-quantized values — the regime the score chain actually runs in.
    let mut q = probe("l2-q", d.s * d.heads, d.hd);
    host_spread(&mut q, d.hd);
    let w = probe("l2-w", d.s, d.heads);

    let want = host_score(&q, &kv, &w, d);
    let got = device_score(&q, &kv, &w, d);
    assert_bits(&want, &got, "indexer_score");

    // Anti-vacuity. A kernel that wrote zeros everywhere would match a host reference that
    // also produced zeros, and `relu_()` makes an all-zero result entirely plausible here —
    // it is what a sign error upstream would give. So the matrix must have real content, and
    // this is checked on the DEVICE's output rather than the host's.
    let nz = got.iter().filter(|v| **v != 0.0).count();
    assert!(nz * 4 > got.len(), "the score matrix must not be mostly zero: {nz}/{}", got.len());
}

/// **The deliberate breaks.** Each perturbs one input the way a real defect would and is
/// required to move the score — and one is required NOT to.
///
/// Two of the oracle's four indexer breakages are expressible as a change to a kernel INPUT
/// rather than to the kernel, which is the strongest technique this suite has (it is
/// `v4_compress_kernel.rs`'s "exact defect impersonation"): `IndexerNoWeights` is `w` set to
/// all ones, and `IndexerNoFp4Quant` is a `q` that skipped the fp4 step. For each, the
/// kernel must track the correspondingly-perturbed host reference and must be far from the
/// clean one.
///
/// The silent case is what makes it a matrix rather than a list. Scaling a `q` row belonging
/// to a query OTHER than the ones under comparison must leave every score bit-identical:
/// `v4_indexer_score`'s `t` indexing is the thing that would break, and a defect model that
/// moved everything would prove nothing about it.
#[test]
fn the_score_comparison_rejects_perturbed_inputs_and_only_the_right_ones() {
    let c = V4Config::v4_flash();
    let d = ScoreDims { s: 4, n_comp: 5, heads: c.index_n_heads, hd: c.index_head_dim };
    let mut q = probe("brk-q", d.s * d.heads, d.hd);
    host_spread(&mut q, d.hd);
    let mut kv = probe("brk-kv", d.n_comp, d.hd);
    host_spread(&mut kv, d.hd);
    let w = probe("brk-w", d.s, d.heads);

    let device = |q: &[f32], kv: &[f32], w: &[f32]| device_score(q, kv, w, d);

    let base = device(&q, &kv, &w);
    assert_bits(&host_score(&q, &kv, &w, d), &base, "score/base");

    // `IndexerNoWeights`: the per-head weights dropped. The kernel must FOLLOW the host
    // reference under the same perturbation (so the comparison has resolution) and must be
    // far from the clean result (so the defect is real).
    let ones = vec![1.0f32; d.s * d.heads];
    let broken = device(&q, &kv, &ones);
    assert_bits(&host_score(&q, &kv, &ones, d), &broken, "score/no-weights");
    assert!(
        broken.iter().zip(&base).any(|(a, b)| a.to_bits() != b.to_bits()),
        "IndexerNoWeights must move the score matrix, else the fixture's weights are all 1"
    );

    // `IndexerNoFp4Quant`: `q` rotated but not quantized.
    let mut q_nofp4 = probe("brk-q", d.s * d.heads, d.hd);
    for row in q_nofp4.chunks_exact_mut(d.hd) {
        hadamard_rotate(row);
        row.iter_mut().for_each(|x| *x = bf16_decode(bf16_encode(*x)));
    }
    let broken = device(&q_nofp4, &kv, &w);
    assert_bits(&host_score(&q_nofp4, &kv, &w, d), &broken, "score/no-fp4");
    assert!(
        broken.iter().zip(&base).any(|(a, b)| a.to_bits() != b.to_bits()),
        "IndexerNoFp4Quant must move the score matrix"
    );

    // THE SILENT HALF. Query row `s-1` is scaled; rows 0..s-1 must not move one bit. This is
    // what pins the `t` indexing: a kernel that read `q` without the `t * heads` stride, or
    // that reduced across queries, would move them.
    let mut q_last = q.clone();
    for v in q_last[(d.s - 1) * d.heads * d.hd..].iter_mut() {
        *v *= 4.0;
    }
    let moved = device(&q_last, &kv, &w);
    let untouched = (d.s - 1) * d.n_comp;
    assert_eq!(
        moved[..untouched].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        base[..untouched].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "perturbing query {} must leave every earlier query's row bit-identical",
        d.s - 1
    );
    assert!(
        moved[untouched..].iter().zip(&base[untouched..]).any(|(a, b)| a.to_bits() != b.to_bits()),
        "...and must move its own row, else the perturbation did nothing anywhere"
    );
}

/// The score launcher refuses an empty compressed region rather than handing back an
/// unwritten buffer.
///
/// `n_comp == 0` is `end_pos / ratio` before the first block completes — a state the
/// reference genuinely reaches — but there it produces an EMPTY score tensor and never calls
/// `topk` on it. A launcher that silently launched nothing would leave `score` unwritten,
/// which reads as a row of zeros: a legal-looking score for "no compressed block is worth
/// attending", and the caller would proceed.
#[test]
fn the_score_launcher_refuses_an_empty_compressed_region() {
    let c = V4Config::v4_flash();
    let buf = up(&vec![0.0f32; 64]);
    let (p, m) = (buf.ptr() as *const f32, buf.ptr() as *mut f32);
    let go = |d: ScoreDims| {
        // SAFETY: refused before any launch in every case asserted here.
        unsafe { launch_v4_indexer_score(p, p, p, m, d, std::ptr::null_mut()) }.is_err()
    };
    let ok = ScoreDims { s: 1, n_comp: 1, heads: c.index_n_heads, hd: c.index_head_dim };
    assert!(go(ScoreDims { n_comp: 0, ..ok }), "empty compressed region");
    assert!(go(ScoreDims { s: 0, ..ok }), "no query rows");
    assert!(go(ScoreDims { heads: 0, ..ok }), "no heads");
    assert!(go(ScoreDims { hd: 0, ..ok }), "zero head width");
}

/// **The guard that did not exist.** [`host_score`] accumulates the head sum in f32 and
/// rounds once; a bf16 running fold gives a demonstrably different answer on this input.
///
/// Added on review. Nothing else in this file catches a revert: `host_score` was *copied
/// from* the oracle's bf16 fold, which is how the defect arrived, and the only test that
/// compares it to the kernel needs a GPU and the checkpoint and has never executed. So the
/// one thing standing between this file and the bug it just fixed was a comment. This test
/// runs on CPU, needs no checkpoint, and pins the arithmetic itself.
///
/// The construction is exact in both accumulators, so the expected values are literals
/// rather than a tolerance:
///
/// * `kv` is the unit vector `e_0`, so each head's dot is just that head's `q[0]`.
/// * head 0 contributes `1.0`; the other 63 contribute `2^-9` each.
/// * bf16's ulp at 1.0 is `2^-7`, so `2^-9` is a quarter of it and **every one of the 63
///   rounds away**: a running fold returns exactly `1.0`.
/// * f32 accumulates `1 + 63·2^-9 = 1.123046875`, and the single closing bf16 round takes
///   it to exactly `1.125`.
///
/// A one-in-eight relative gap out of a per-term difference of a quarter ulp is the point:
/// the error compounds rather than averaging out, whatever the sign of the weights — which
/// matters, because they ARE signed and the first version of this fix's rationale wrongly
/// said otherwise.
///
/// Proved to fire on 2026-08-05 by reverting `host_score` to the fold: `got 1.0`, want
/// 1.125.
#[test]
fn host_score_accumulates_in_f32_not_bf16() {
    let d = ScoreDims { s: 1, n_comp: 1, heads: 64, hd: 128 };
    let mut kv = vec![0.0f32; d.hd];
    kv[0] = 1.0;
    let mut q = vec![0.0f32; d.s * d.heads * d.hd];
    for hh in 0..d.heads {
        q[hh * d.hd] = if hh == 0 { 1.0 } else { (2.0f32).powi(-9) };
    }
    let w = vec![1.0f32; d.s * d.heads];

    let got = host_score(&q, &kv, &w, d);
    assert_eq!(got.len(), 1);
    assert_eq!(
        got[0].to_bits(),
        1.125f32.to_bits(),
        "f32 accumulation then one bf16 round is 1.125; got {:?}. A bf16 RUNNING fold gives \
         exactly 1.0 here, which is what this test exists to reject.",
        got[0]
    );
    // The negative control, computed inline so the claim above is executable rather than
    // asserted: the fold this file used to implement really does return 1.0 on this input.
    let rbf = |x: f32| bf16_decode(bf16_encode(x));
    let mut fold = 0.0f32;
    for hh in 0..d.heads {
        fold = rbf(fold + rbf(q[hh * d.hd]));
    }
    assert_eq!(fold.to_bits(), 1.0f32.to_bits(), "the rejected fold must reach 1.0");
    assert_ne!(fold.to_bits(), got[0].to_bits(), "...and must differ from the correct value");
}
