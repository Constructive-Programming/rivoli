//! **The V4-Flash lightning indexer's two device kernels** — `act_quant_f4_rotated` (the
//! Hadamard-and-fp4 spread) and `index_score_blocks` (the `einsum`/`relu_`/`weights`/sum chain).
//!
//! Ported from `old:tests/blockindex_kernel.rs`. The two halves are gated against DIFFERENT
//! references and the difference matters: the spread against the frozen oracle's own primitives,
//! the score against a host transliteration of `model.py:425-427` that **the oracle currently
//! contradicts** (see [`host_score`]). A green run of the score half is not a correctness verdict
//! until the oracle's fix lands.
//!
//! # Why the SCORE matrix and not the selected sets
//!
//! The shipped goldens are **set-invariant** at `index_topk = 512`: `IndexerNoWeights` and
//! `IndexerNoRelu` both move the scores and leave the selection bit-identical, because the causal
//! mask alone determines it until the top-k truncates, which needs >= 2052 tokens. A gate resting
//! on the sets therefore accepts an arbitrarily wrong RANKING. `launch_index_score_blocks` writes
//! the full pre-top-k matrix precisely so a consumer can be scored on it instead, and that
//! comparison is **strictly stronger**: every defect that moves a set moves the scores, and four
//! that move the scores move no set at the shipped configuration. So this file does not
//! reimplement the mask or the top-k, and does not claim to cover them.
//!
//! # What this file provably cannot detect — read before trusting it
//!
//! * **Anything the oracle is also wrong about**, though the one instance found is CONFIRMED and
//!   fixed on this side. `Oracle::indexer` sums the per-head products as a bf16 RUNNING fold,
//!   while `torch.sum` over bf16 accumulates through `acc_type` — f32 — and rounds ONCE. That is
//!   a property of the reduction, measured off-repo against CPU torch, with no reproducer in this
//!   tree. The kernel and [`host_score`] both do what torch does and
//!   [`host_score_accumulates_in_f32_not_bf16`] is the in-tree guard; **the oracle's fix is owned
//!   elsewhere**, so do not read a comparison against the current indexer goldens as evidence in
//!   either direction.
//! * **The summation ORDER.** The bf16 fold pinned it as a side effect and an f32 accumulator does
//!   not; torch's own reduction is vectorized and tree-shaped. The kernel and [`host_score`] agree
//!   with each other exactly, and neither is pinned to torch's order.
//! * **`wq_b` and `weights_proj`.** Their GEMVs are `kernel_v4_quant.rs`'s; this file drives the
//!   two indexer-specific kernels from host-computed inputs so that a projection's re-association
//!   cannot be mistaken for a scoring error.
//! * **A misreading of `model.py` SHARED between [`host_score`] and the oracle**, since the score
//!   chain is now transliterated twice.
//! * **The basis order is NOT in this list.** It was the highest-risk inference of the oracle
//!   stage and was settled against `fast_hadamard_transform`'s own documented contract, so the
//!   Hadamard exercised here is pinned to something other than the oracle's opinion of it.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: no `rocm` CI arm, and no GPU for this port. Two mutations in
//! `kernels/linalg.hip`:
//!
//! * In `rivoli_act_quant_f4_rotated`, drop the bf16 store BETWEEN the rotation and the fp4
//!   quantization. [`indexer_spread_is_bit_identical_to_the_oracle`] must go RED naming a first
//!   differing element — the store is the step a port drops, which is why [`host_spread`] keeps
//!   its four steps visible rather than hiding them behind one call.
//!   [`fp4_block_scale_covers_sixty_binades_not_two`] must go red too, and if it is the ONLY one
//!   that reddens, the 9-row fixture has stopped reaching a value the store moves.
//! * In `rivoli_index_score_blocks`, remove `#pragma clang fp contract(off)` (or fold the head sum
//!   in bf16). [`indexer_score_is_bit_identical_on_the_checkpoints_compressed_kv`] and
//!   [`the_score_comparison_rejects_perturbed_inputs_and_only_the_right_ones`] must go RED, and
//!   [`host_score_accumulates_in_f32_not_bf16`] must stay GREEN — it is a CPU test of the
//!   reference, and a red there means the host side drifted, which is a different repair. That
//!   pragma is load-bearing and was missing until review: without it `dot += q*kv` fuses to
//!   `v_fmac_f32`, one rounding per term where the host does two, and every comparison here fails
//!   for a reason that looks like a numerics bug.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_backend::abi::ScoreDims;
use rivoli_backend::hip::{
    ScoreBufs, device_sync, launch_act_quant_f4_rotated, launch_index_score_blocks,
};
use rivoli_engine::device::DeviceBuf;
use rivoli_engine::v4::geometry::{Geom, LayerKind, Quantize};
use rivoli_oracles::v4oracle::forward::{Counters, Defect, IndexerW, Oracle};
use rivoli_oracles::v4oracle::numerics::{
    bf16_decode, bf16_encode, fp4_act_quant_inplace, hadamard_rotate,
};
use rivoli_oracles::v4oracle::weights::{Checkpoint, fixed_bf16};

mod common;
use common::{
    CompSpec, Configs, assert_bits, assert_guard, back, checkpoint, compressor_w, dev, f32b, f32v,
};

/// The indexer's head width — a POWER OF TWO, because the indexer Hadamard-rotates rows of it,
/// and a multiple of the fp4 block of 32.
///
/// **The value is the shipped `index_head_dim`, and it is pinned to
/// `rivoli_artifact::v4_config::V4Config` by [`the_indexer_widths_are_the_shipped_ones`]** — not
/// to `v4oracle::weights::V4Config`, whose hard-coded copy of the same number is the
/// transliteration's independence and must not become this file's source. It is a local constant
/// rather than a config read at each site so that the SHAPE tests below — the spread, the binade
/// sweep, the launcher guards — run on a machine with a GPU and no checkpoint; the claim that
/// this is the model's number is made once, where a checkpoint is available to make it against.
const INDEX_HEAD_DIM: usize = 128;

/// The indexer's own head count — NOT the attention's, which coincides at 64 on this checkpoint,
/// which is exactly why `ScoreDims` names them separately. Pinned with [`INDEX_HEAD_DIM`].
const INDEX_N_HEADS: usize = 64;

/// A probe long enough that `end_pos / 4` gives many compressed blocks, so the score matrix has
/// columns whose ranking could be wrong: 16 of them, against the 3 that the 13-token emit prompt
/// the shipped goldens use would give.
const SCORE_PROBE_LEN: usize = 64;

// =======================================================================================
// device plumbing
// =======================================================================================

/// Read `n` f32 back off the device, through the engine's own decoder rather than a local
/// `from_le_bytes` loop — a test that read bytes back differently from the code under test could
/// agree with itself while both were wrong about the layout.
fn down(b: &DeviceBuf, n: usize) -> Vec<f32> {
    let mut bytes = back(b);
    bytes.truncate(n * 4);
    f32v(&bytes)
}

/// Whether ANY element disagrees bit-wise — the resolution check every deliberate break below
/// makes, and the exact negation of what `assert_bits` requires.
///
/// One spelling, because the three call sites each state it about a different perturbation and
/// each keeps its own `assert!` and message: naming which break went inert is the whole value of
/// that failure, and only the comparison is common. Bits and not `!=`: a break that moved a value
/// to a different NaN payload is a move, and `f32::ne` would call it a match.
fn any_bits_differ(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits())
}

/// `x = bf16(x)` over a row — the reference's store after each transform.
///
/// Named because it appears three times IN THIS FILE: twice in [`host_spread`] and once in the
/// `IndexerNoFp4Quant` break, which is that function with the quantization removed. Three
/// spellings of a round-trip through `bf16_encode`/`bf16_decode` is three places to write
/// `bf16_decode(bf16_encode(x))` the other way round, which is the identity and would make every
/// store silently absent.
fn store_bf16(row: &mut [f32]) {
    row.iter_mut()
        .for_each(|x| *x = bf16_decode(bf16_encode(*x)));
}

/// `Oracle::indexer_spread` on the host, over `rows` rows of `d` — the value
/// `act_quant_f4_rotated` must reproduce.
///
/// Written from the oracle's public primitives rather than by calling `indexer_spread`, which is
/// private. That is not a re-derivation: `hadamard_rotate` and `fp4_act_quant_inplace` ARE the
/// oracle's, so the only thing restated is the ORDER of the four steps — and the bf16 store
/// between the rotation and the quantization is the step a port drops, so having it visible at
/// the comparison site is worth more than hiding it behind one call. Only the STORE is named; the
/// four steps still read as four here, which is the whole of that argument.
fn host_spread(rows: &mut [f32], d: usize) {
    for row in rows.chunks_exact_mut(d) {
        hadamard_rotate(row);
        store_bf16(row);
        fp4_act_quant_inplace(row, 32);
        store_bf16(row);
    }
}

/// One spread fixture's shape: how many rows, and how wide each is.
///
/// A pair rather than two adjacent `usize`, and the failure a transposition produces is the one
/// this whole file is about: `launch_act_quant_f4_rotated(p, d, rows, ..)` at `rows = 9, d = 128`
/// transforms 128 rows of 9 — a width the Hadamard REFUSES, so that particular swap is loud, while
/// `rows = 60, d = 128` against `rows = 128, d = 60` is silent on the second guard and wrong on
/// both. Naming the two moves the choice to the fixture, where the argument for each is written.
#[derive(Clone, Copy)]
struct Spread {
    rows: usize,
    d: usize,
}

/// Upload, spread in place, read back — the whole device side of one spread comparison.
///
/// One helper rather than two call sites: the duplication gate found the copy in the reference
/// tree and was right about the shape of the risk. The two callers differ only in their fixture,
/// and a divergence in the row count passed to the launcher versus the count read back is exactly
/// the kind of slip that would make a comparison vacuous rather than wrong.
fn device_spread(host_in: &[f32], g: Spread) -> Vec<f32> {
    let (rows, d) = (g.rows, g.d);
    let buf = dev(&f32b(host_in));
    // SAFETY: `buf` is `rows * d` writable f32 by the caller's fixture, 4-byte aligned as every
    // `DeviceBuf` allocation is, and outlives the join inside `down`.
    unsafe { launch_act_quant_f4_rotated(buf.ptr() as *mut f32, rows, d, std::ptr::null_mut()) }
        .expect("spread launch");
    down(&buf, rows * d)
}

/// Spread `host_in` on the host and on the device and require the two to agree BIT FOR BIT.
///
/// Exact, not a tolerance. This kernel is bit-exact against the oracle by construction — the
/// Hadamard visits the same operand pairs in the same order — so a tolerance would be admitting
/// something nothing in the design produces. If this ever needs loosening, that is a finding
/// about the kernel and belongs in a doc, not in an epsilon.
///
/// The `label` stays the caller's, and that is the whole reason this takes one: the two callers
/// mean different things by a difference — one says the spread arithmetic is wrong, the other
/// says it is wrong on a binade the shipped fixtures never reach. A shared message would send the
/// reader to the wrong fixture.
fn assert_spread_matches(host_in: &[f32], g: Spread, label: &str) {
    let mut want = host_in.to_vec();
    host_spread(&mut want, g.d);
    assert_bits(&want, &device_spread(host_in, g), label);
}

// =======================================================================================
// the spread — Geom::indexer's finish
// =======================================================================================

/// `act_quant_f4_rotated` reproduces `Oracle::indexer_spread` bit for bit.
///
/// Synthetic rows rather than checkpoint ones, and deliberately: what this kernel must get right
/// is the *arithmetic*, and the fixture is built to reach the corners real activations may not.
/// The checkpoint's own tensors drive the scoring test.
#[test]
fn indexer_spread_is_bit_identical_to_the_oracle() {
    let g = Spread {
        rows: 9,
        d: INDEX_HEAD_DIM,
    };
    // `fixed_bf16` draws bf16-rounded values in [-1, 1), which is what `rotate_activation`'s
    // `assert x.dtype == torch.bfloat16` guarantees it receives.
    let host_in = fixed_bf16("spread", g.rows * g.d, 1.0);
    assert_spread_matches(&host_in, g, "indexer_spread");
}

/// **Widening the e8m0 exponent coverage the whole V4 suite lacks.**
///
/// e8m0 exercises **2 distinct codes of 254** (`119..=120`) across everything shipped, so a decode
/// bug on any other is invisible. Two corrections to that premise, both worth stating because they
/// change what "widening" means here:
///
/// 1. The indexer consumes **no e8m0 scale BYTES at all**. Its weights are fp8 (`wq_b`, which
///    ships a `.scale`) and bf16 (`weights_proj`, `wkv`, `wgate`); there is no packed fp4 weight
///    tensor on this path, so the fp4 dot's `e8m0f` is never called.
/// 2. What this path DOES exercise is the same exponent domain through `fast_round_scale` —
///    `fp4_act_quant`'s block scale is a bare power of two, derived by the identical
///    `fast_log2_ceil`/`fast_pow2` bit surgery that produces an e8m0 code.
///
/// So this test sweeps the **scale exponent**, which is the reachable half: each row's magnitude
/// is set to a different binade across 60 of them, from `2^-30` to `2^29`, so the block scale
/// lands on 60 distinct powers of two rather than the two the shipped fixtures reach.
///
/// **What remains uncovered, precisely:** `e8m0f`'s decode of a scale BYTE, including its two
/// named endpoint cases (`0x00` → `2^-127`, `0xff` → NaN). Nothing on the indexer path can reach
/// them; `kernel_v4_moe.rs` covers the formula on the host and nothing covers them on a device.
#[test]
fn fp4_block_scale_covers_sixty_binades_not_two() {
    let g = Spread {
        rows: 60,
        d: INDEX_HEAD_DIM,
    };
    let (rows, d) = (g.rows, g.d);
    let base = fixed_bf16("binade", rows * d, 1.0);
    // Row `r` scaled into binade `r - 30`. `exp2` of an integer is exact, so the fixture adds no
    // rounding of its own and every difference the comparison finds is the kernel's.
    let host_in: Vec<f32> = base
        .iter()
        .enumerate()
        .map(|(i, v)| bf16_decode(bf16_encode(v * (2.0f32).powi(i as i32 / d as i32 - 30))))
        .collect();
    // The fixture must actually SPREAD, or this is the shipped coverage under a new name. A
    // PROXY, and a loose one — stated precisely because it is not "counted from the oracle's own
    // `fast_round_scale` path". It uses libm `log2().ceil()` on the PRE-Hadamard, per-ROW amax and
    // without the `x FP4_MAX_INV` factor, where the kernel uses `fast_log2_ceil` on the
    // post-Hadamard, per-BLOCK-of-32 amax with it, and those two log2s disagree at binade edges.
    // What the proxy does establish is the only thing needed: the fixture ROWS span a wide range
    // of magnitudes rather than the one the shipped fixtures sit in.
    let binades: std::collections::BTreeSet<i32> = host_in
        .chunks_exact(d)
        .map(|r| r.iter().fold(0.0f32, |m, v| m.max(v.abs())).log2().ceil() as i32)
        .collect();
    assert!(
        binades.len() >= 55,
        "the fixture must span many binades, got {}",
        binades.len()
    );
    assert_spread_matches(&host_in, g, "spread over 60 binades");
}

/// The spread launcher refuses the shapes whose failure would be silent, and each refusal is shown
/// to be REACHABLE.
///
/// Every one of these can fire: they are not restatements of a type. A `d` that is not a power of
/// two is what a config change to `index_head_dim` produces, and the reference would zero-pad it
/// rather than transform it — the Hadamard would then run over a length the model never intended
/// and return finite numbers.
#[test]
fn the_spread_launcher_refuses_widths_the_hadamard_cannot_transform() {
    let buf = dev(&f32b(&[0.0f32; 1024]));
    let p = buf.ptr() as *mut f32;
    // SAFETY: every case below is refused before any launch; `buf` is 1024 f32, larger than the
    // one accepted shape, which is the only case that reaches the device.
    let go = |rows: usize, d: usize| unsafe {
        launch_act_quant_f4_rotated(p, rows, d, std::ptr::null_mut())
    };
    // The shape the model actually uses is ACCEPTED, so the guards below are not simply refusing
    // everything.
    assert_guard(go(1, INDEX_HEAD_DIM), None, "the shipped width");
    device_sync().expect("sync"); // the accepted case LAUNCHED — join before `buf` drops
    for (what, r) in [
        ("zero rows", go(0, 128)),
        ("zero width", go(1, 0)),
        ("wider than the LDS tile", go(1, 512)),
        (
            "not a power of two -- the reference would zero-pad this",
            go(1, 96),
        ),
        // And it is NOT unreachable behind the power-of-two check as the kernel's comment claimed
        // until review: 16 is a power of two and is still a ragged fp4 block.
        ("a power of two SMALLER than the fp4 block of 32", go(1, 16)),
    ] {
        assert!(r.is_err(), "{what} must be refused");
    }
}

/// **The trap made unrepresentable.** A geometry built for the indexer carries the fp4 finish; one
/// built for the attention compressor carries the partial fp8 one; and the two are otherwise
/// identical in every dimension a guard could check.
///
/// The point is the last clause. At `head_dim` this is trivially distinguishable, but the
/// indexer's compressor is built at `index_head_dim` with the same `ratio`, `coff` and
/// `rope_head_dim`, so `Geom::attention` and `Geom::indexer` at those arguments agree on all six
/// integers the kernel sees. Only the [`Quantize`] separates them, which is why it is a field and
/// not an argument to the launch.
#[test]
fn geom_indexer_and_geom_attention_do_not_finish_the_same_way() {
    let (hd, rd, eps) = (INDEX_HEAD_DIM, 64usize, 1e-6f32);
    let a = Geom::attention(LayerKind::Overlap, hd, rd, eps).unwrap();
    let i = Geom::indexer(LayerKind::Overlap, hd, rd, eps).unwrap();
    assert_eq!(a.quantize(), Quantize::PartialFp8);
    assert_eq!(i.quantize(), Quantize::HadamardFp4);
    // Identical to the kernel, different to the finish. This IS entailed — `Geom::indexer` builds
    // its abi as `..Self::attention(..)?`, a functional update overriding only `quant`, so the two
    // halves cannot differ. Kept anyway as the executable statement of the HAZARD rather than as a
    // test of that code: it is the reason no dimension guard can catch the confusion, and if it
    // ever fails the argument for the `Quantize` field has weakened, which is a thing to know.
    assert_eq!(a.abi(), i.abi(), "the ABI halves must be indistinguishable");

    // An `Indexer` exists ONLY at ratio 4 (model.py:474), so the other two classes cannot produce
    // one — stricter than `Geom::attention`, which accepts every compressor.
    assert!(
        Geom::indexer(LayerKind::NonOverlap(128), hd, rd, eps).is_none(),
        "ratio 128"
    );
    assert!(
        Geom::indexer(LayerKind::Plain, hd, rd, eps).is_none(),
        "ratio 0"
    );
    assert!(
        Geom::attention(LayerKind::NonOverlap(128), 512, rd, eps).is_some(),
        "...while the attention compressor does exist at ratio 128, so the line above is a real \
         restriction and not a broken constructor"
    );
}

// =======================================================================================
// the scoring chain
// =======================================================================================

/// One layer's `attn.indexer.*`.
///
/// The comment that matters: `wq_b` is fp8 on disk (it ships a `.scale`), unlike `weights_proj`,
/// which is bare bf16 — and V4's `Indexer` has **no `wk` and no `k_norm`**. Guessing GLM's names
/// here is what broke the first convert.
///
/// `rotate = true` on the nested compressor: the indexer's own Hadamard-and-fp4 finish where the
/// attention compressor partially fp8-quantizes. Same class, different arithmetic.
fn indexer_w(ck: &Checkpoint, layer: usize, c: &Configs) -> IndexerW {
    IndexerW {
        wq_b: ck
            .fp8(&format!("layers.{layer}.attn.indexer.wq_b.weight"))
            .unwrap(),
        weights_proj: ck
            .dense(&format!("layers.{layer}.attn.indexer.weights_proj.weight"))
            .unwrap(),
        compressor: compressor_w(
            ck,
            &format!("layers.{layer}.attn.indexer.compressor"),
            CompSpec {
                ratio: 4,
                d: c.engine.index_head_dim,
                rotate: true,
            },
        ),
    }
}

/// One score comparison's three operands and the extents they are contracted over.
///
/// A struct for the reason `ScoreBufs` is one on the device side, made about the HOST: `q`, `kv`
/// and `w` are three `&[f32]` in a row and no type check can tell any of them from another, so a
/// transposed pair still indexes real f32 — finite, plausible, wrong — and it would move the host
/// reference and the kernel TOGETHER, which is the shape of disagreement neither side can report.
/// `d` travels with them because it is what says how each is to be read.
#[derive(Clone, Copy)]
struct ScoreCase<'a> {
    q: &'a [f32],
    kv: &'a [f32],
    w: &'a [f32],
    d: ScoreDims,
}

/// Upload, score, read back. Six pointers and a `ScoreDims` at two call sites is six chances for a
/// transposed argument, and `q`/`kv`/`w` are all `*const f32` — which is `ScoreBufs`'s own
/// argument, made once more at the call.
fn device_score(c: ScoreCase<'_>) -> Vec<f32> {
    let d = c.d;
    let (dq, dkv, dw) = (dev(&f32b(c.q)), dev(&f32b(c.kv)), dev(&f32b(c.w)));
    let out = dev(&f32b(&vec![0.0f32; d.s * d.n_comp]));
    let bufs = ScoreBufs {
        q: dq.ptr().cast(),
        kv: dkv.ptr().cast(),
        w: dw.ptr().cast(),
        score: out.ptr() as *mut f32,
    };
    // SAFETY: every buffer is sized by `d` (`q` is `s·heads·hd`, `kv` is `n_comp·hd`, `w` is
    // `s·heads`, `score` is `s·n_comp`), each is a distinct `DeviceBuf` allocation so none aliases
    // another, and all outlive the join inside `down`.
    unsafe { launch_index_score_blocks(bufs, d, std::ptr::null_mut()) }.expect("score launch");
    down(&out, d.s * d.n_comp)
}

/// `model.py:425-427` on the host — the value `index_score_blocks` must reproduce.
///
/// **Transcribed from `model.py`, not from `Oracle::indexer`** — and that stopped being only a
/// limitation and became the reason this file is right where the oracle is wrong: `Oracle::indexer`
/// folds the head sum in bf16 per term, where `torch.sum` accumulates in f32 and rounds once. The
/// oracle's fix is owned elsewhere; this reference already matches the measured behaviour, so
/// **the two will disagree until it lands.**
///
/// The rest of the distinction is the honest limit of this comparison. The oracle computes this
/// chain inside `indexer` but exposes neither the roped-and-spread `q` nor the scaled `weights`,
/// so the kernel cannot be handed the oracle's own intermediates and scored on the oracle's own
/// exported matrix. What that costs, precisely: this is a second transliteration of the same nine
/// lines, so a shared misreading of `model.py` between it and the oracle is invisible. What it
/// still buys is everything the KERNEL could get wrong on its own: the bf16 store placement, the
/// relu order, the summation order, and the head/block indexing.
fn host_score(c: ScoreCase<'_>) -> Vec<f32> {
    let ScoreCase { q, kv, w, d } = c;
    let rbf = |x: f32| bf16_decode(bf16_encode(x));
    let mut out = vec![0.0f32; d.s * d.n_comp];
    for t in 0..d.s {
        for c in 0..d.n_comp {
            let mut acc = 0.0f32;
            for hh in 0..d.heads {
                let qh = &q[(t * d.heads + hh) * d.hd..(t * d.heads + hh + 1) * d.hd];
                let kvc = &kv[c * d.hd..(c + 1) * d.hd];
                // `einsum` -> bf16, `relu_()` in place -> bf16, `* weights` -> bf16. Those three
                // are elementwise tensors the reference materializes, so each store is real.
                let mut dot = rbf(qh.iter().zip(kvc).map(|(a, b)| a * b).sum::<f32>());
                dot = dot.max(0.0);
                // `.sum(dim=2)` accumulates in f32 (`acc_type`) and rounds ONCE. This read as a
                // bf16 running fold until the oracle's own error was found, which is how the
                // defect arrived here in the first place.
                acc += rbf(dot * w[t * d.heads + hh]);
            }
            out[t * d.n_comp + c] = rbf(acc);
        }
    }
    out
}

/// The four extents one score comparison runs at, from the INDEXER's own head geometry.
fn score_dims(s: usize, n_comp: usize) -> ScoreDims {
    ScoreDims {
        s,
        n_comp,
        heads: INDEX_N_HEADS,
        hd: INDEX_HEAD_DIM,
    }
}

/// `index_score_blocks` is bit-identical to the reference chain, on the CHECKPOINT's own
/// compressed KV.
///
/// The `kv` side is REAL: it is the indexer's compressor cache after layer 2's indexer has run on
/// a `SCORE_PROBE_LEN` prompt, so the rows are the genuine output of the pooling plus the
/// Hadamard-and-fp4 finish — the fp4 codebook's eight magnitudes and its zeros as the model
/// actually produces them, not as a fixture imagines them. `q` and `w` are synthetic because the
/// oracle does not expose its own.
#[test]
fn indexer_score_is_bit_identical_on_the_checkpoints_compressed_kv() {
    let Some(c) = Configs::new() else { return };
    let Some(ck) = checkpoint() else { return };
    let n = SCORE_PROBE_LEN;
    let iw = indexer_w(&ck, 2, &c);
    let o = Oracle::new(c.oracle.clone(), Defect::None);
    let mut cs = o.fresh_state(2).idx_comp.expect("layer 2 has an indexer");
    let mut ctr = Counters::default();
    o.compressor(
        &iw.compressor,
        &mut cs,
        &fixed_bf16("l2-x", n * c.engine.hidden, 1.0),
        n,
        0,
        o.freqs(2),
        &mut ctr,
    );
    assert_eq!(
        ctr.compressed_blocks,
        n / 4,
        "the probe must fill the compressed cache"
    );

    let d = score_dims(3, n / 4);
    let kv = cs.cache[..d.n_comp * d.hd].to_vec();
    assert!(
        kv.iter().any(|v| *v != 0.0),
        "the cache must not be all zeros"
    );

    // `q` through the same spread the real path applies, so both sides of the dot are fp4-quantized
    // values — the regime the score chain actually runs in.
    let mut q = fixed_bf16("l2-q", d.s * d.heads * d.hd, 1.0);
    host_spread(&mut q, d.hd);
    let w = fixed_bf16("l2-w", d.s * d.heads, 1.0);

    // A closure and not a literal, for the reason the break matrix next door gives: the four-field
    // spelling exploded across five lines is what `build.rs`'s duplication gate reports when two
    // tests build one, and a one-line binding of the shared operands says the same thing.
    let case = |q, w| ScoreCase { q, kv: &kv, w, d };
    let got = device_score(case(&q, &w));
    assert_bits(&host_score(case(&q, &w)), &got, "indexer_score");

    // Anti-vacuity. A kernel that wrote zeros everywhere would match a host reference that also
    // produced zeros, and `relu_()` makes an all-zero result entirely plausible here — it is what
    // a sign error upstream would give. So the matrix must have real content, and this is checked
    // on the DEVICE's output rather than the host's.
    let nz = got.iter().filter(|v| **v != 0.0).count();
    assert!(
        nz * 4 > got.len(),
        "the score matrix must not be mostly zero: {nz}/{}",
        got.len()
    );
}

/// The two indexer widths this file states as constants ARE the shipped ones, read from the
/// checkpoint's own `config.json` through the engine's schema.
///
/// Separated from the comparisons deliberately: it is the ONE place the numbers are claimed to be
/// the model's, so the shape tests above can run without a checkpoint while the claim still has
/// somewhere to fail. Read from `rivoli_artifact::v4_config` and never from the oracle's
/// hard-coded transliteration, whose independence is what makes it a reference at all —
/// `common::Configs` holds the two against each other for the same reason.
#[test]
fn the_indexer_widths_are_the_shipped_ones() {
    let Some(c) = Configs::new() else { return };
    assert_eq!(
        (INDEX_HEAD_DIM, INDEX_N_HEADS),
        (c.engine.index_head_dim, c.engine.index_n_heads),
        "this file's indexer geometry has drifted from the checkpoint's config.json"
    );
    assert!(
        INDEX_HEAD_DIM.is_power_of_two() && INDEX_HEAD_DIM.is_multiple_of(32),
        "the Hadamard needs a power of two and the fp4 block needs a multiple of 32"
    );
}

/// **The deliberate breaks.** Each perturbs one input the way a real defect would and is required
/// to MOVE the score — and one is required NOT to.
///
/// Two of the oracle's four indexer breakages are expressible as a change to a kernel INPUT rather
/// than to the kernel, which is the strongest technique available here: `IndexerNoWeights` is `w`
/// set to all ones, and `IndexerNoFp4Quant` is a `q` that skipped the fp4 step. For each, the
/// kernel must track the correspondingly-perturbed host reference and must be far from the clean
/// one.
///
/// The silent case is what makes it a matrix rather than a list. Scaling a `q` row belonging to a
/// query OTHER than the ones under comparison must leave every earlier score bit-identical: the
/// kernel's `t` indexing is the thing that would break, and a defect model that moved everything
/// would prove nothing about it.
#[test]
fn the_score_comparison_rejects_perturbed_inputs_and_only_the_right_ones() {
    let d = score_dims(4, 5);
    let mut q = fixed_bf16("brk-q", d.s * d.heads * d.hd, 1.0);
    host_spread(&mut q, d.hd);
    let mut kv = fixed_bf16("brk-kv", d.n_comp * d.hd, 1.0);
    host_spread(&mut kv, d.hd);
    let w = fixed_bf16("brk-w", d.s * d.heads, 1.0);

    // ONE closure, so the shared compressed KV and the extents are named once: the comparisons
    // below differ only in which query rows and which per-head weights they hand over, and
    // spelling the four-field literal six times is six chances to pair one perturbation's operand
    // with another's — which is the exact mistake `ScoreCase` exists to make unspellable.
    let case = |q, w| ScoreCase { q, kv: &kv, w, d };
    let base = device_score(case(&q, &w));
    assert_bits(&host_score(case(&q, &w)), &base, "score/base");

    // `IndexerNoWeights`: the per-head weights dropped. The kernel must FOLLOW the host reference
    // under the same perturbation (so the comparison has resolution) and must be far from the
    // clean result (so the defect is real).
    let ones = vec![1.0f32; d.s * d.heads];
    let broken = device_score(case(&q, &ones));
    assert_bits(&host_score(case(&q, &ones)), &broken, "score/no-weights");
    assert!(
        any_bits_differ(&broken, &base),
        "IndexerNoWeights must move the score matrix, else the fixture's weights are all 1"
    );

    // `IndexerNoFp4Quant`: `q` rotated but not quantized. This is `host_spread` with the
    // `fp4_act_quant_inplace` line and its trailing store deleted, and it is spelled out rather
    // than shared so the two steps that DO run stay visible beside the one that does not — a
    // defect model whose omission you cannot see is not evidence of anything.
    let mut q_nofp4 = fixed_bf16("brk-q", d.s * d.heads * d.hd, 1.0);
    for row in q_nofp4.chunks_exact_mut(d.hd) {
        hadamard_rotate(row);
        store_bf16(row);
    }
    let broken = device_score(case(&q_nofp4, &w));
    assert_bits(&host_score(case(&q_nofp4, &w)), &broken, "score/no-fp4");
    assert!(
        any_bits_differ(&broken, &base),
        "IndexerNoFp4Quant must move the score matrix"
    );

    // THE SILENT HALF. Query row `s-1` is scaled; rows `0..s-1` must not move one bit. This is what
    // pins the `t` indexing: a kernel that read `q` without the `t * heads` stride, or that reduced
    // across queries, would move them.
    let mut q_last = q.clone();
    for v in q_last[(d.s - 1) * d.heads * d.hd..].iter_mut() {
        *v *= 4.0;
    }
    let moved = device_score(case(&q_last, &w));
    let untouched = (d.s - 1) * d.n_comp;
    assert_bits(
        &base[..untouched],
        &moved[..untouched],
        "perturbing the last query must leave every earlier query's row bit-identical",
    );
    assert!(
        any_bits_differ(&moved[untouched..], &base[untouched..]),
        "...and must move its own row, else the perturbation did nothing anywhere"
    );
}

/// The score launcher refuses an empty compressed region rather than handing back an unwritten
/// buffer.
///
/// `n_comp == 0` is `end_pos / ratio` before the first block completes — a state the reference
/// genuinely reaches — but there it produces an EMPTY score tensor and never calls `topk` on it. A
/// launcher that silently launched nothing would leave `score` unwritten, which reads as a row of
/// zeros: a legal-looking score for "no compressed block is worth attending", and the caller would
/// proceed.
#[test]
fn the_score_launcher_refuses_an_empty_compressed_region() {
    let buf = dev(&f32b(&[0.0f32; 64]));
    let bufs = ScoreBufs {
        q: buf.ptr().cast(),
        kv: buf.ptr().cast(),
        w: buf.ptr().cast(),
        score: buf.ptr() as *mut f32,
    };
    // SAFETY: refused before any launch in every case asserted here, so no pointer is
    // dereferenced and the deliberate aliasing of one buffer across all four never reaches a
    // kernel.
    let go = |d: ScoreDims| unsafe { launch_index_score_blocks(bufs, d, std::ptr::null_mut()) };
    let ok = score_dims(1, 1);
    for (what, d) in [
        ("empty compressed region", ScoreDims { n_comp: 0, ..ok }),
        ("no query rows", ScoreDims { s: 0, ..ok }),
        ("no heads", ScoreDims { heads: 0, ..ok }),
        ("zero head width", ScoreDims { hd: 0, ..ok }),
    ] {
        assert!(go(d).is_err(), "{what} must be refused");
    }
}

/// **The guard that did not exist.** [`host_score`] accumulates the head sum in f32 and rounds
/// once; a bf16 running fold gives a demonstrably different answer on this input.
///
/// Nothing else in this file catches a revert: `host_score` was *copied from* the oracle's bf16
/// fold, which is how the defect arrived, and the only tests that compare it to the kernel need a
/// GPU. So the one thing standing between this file and the bug it just fixed was a comment. This
/// runs on CPU, needs no checkpoint, and pins the arithmetic itself.
///
/// The construction is exact in both accumulators, so the expected values are literals rather than
/// a tolerance:
///
/// * `kv` is the unit vector `e_0`, so each head's dot is just that head's `q[0]`.
/// * head 0 contributes `1.0`; the other 63 contribute `2^-9` each.
/// * bf16's ulp at 1.0 is `2^-7`, so `2^-9` is a quarter of it and **every one of the 63 rounds
///   away**: a running fold returns exactly `1.0`.
/// * f32 accumulates `1 + 63·2^-9 = 1.123046875`, and the single closing bf16 round takes it to
///   exactly `1.125`.
///
/// A one-in-eight relative gap out of a per-term difference of a quarter ulp is the point: the
/// error compounds rather than averaging out, whatever the sign of the weights — which matters,
/// because they ARE signed.
#[test]
fn host_score_accumulates_in_f32_not_bf16() {
    let d = score_dims(1, 1);
    let mut kv = vec![0.0f32; d.hd];
    kv[0] = 1.0;
    let mut q = vec![0.0f32; d.s * d.heads * d.hd];
    for hh in 0..d.heads {
        q[hh * d.hd] = if hh == 0 { 1.0 } else { (2.0f32).powi(-9) };
    }
    let w = vec![1.0f32; d.s * d.heads];

    let case = |q, w| ScoreCase { q, kv: &kv, w, d };
    let got = host_score(case(&q, &w));
    assert_eq!(got.len(), 1);
    assert_eq!(
        got[0].to_bits(),
        1.125f32.to_bits(),
        "f32 accumulation then one bf16 round is 1.125; got {:?}. A bf16 RUNNING fold gives \
         exactly 1.0 here, which is what this test exists to reject.",
        got[0]
    );
    // The negative control, computed inline so the claim above is executable rather than asserted:
    // the fold this reference used to implement really does return 1.0 on this input.
    let rbf = |x: f32| bf16_decode(bf16_encode(x));
    let mut fold = 0.0f32;
    for hh in 0..d.heads {
        fold = rbf(fold + rbf(q[hh * d.hd]));
    }
    assert_eq!(
        fold.to_bits(),
        1.0f32.to_bits(),
        "the rejected fold must reach 1.0"
    );
    assert_ne!(
        fold.to_bits(),
        got[0].to_bits(),
        "...and must differ from the correct value"
    );
}
