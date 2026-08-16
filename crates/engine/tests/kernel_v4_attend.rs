//! **DeepSeek-V4's attention core and its rotary** — `gather_attn_shared_kv` and
//! `rope_adjacent`, each against a reference written beside it.
//!
//! Ported from `old:tests/f4_attn.rs`. **What came and what could not**: that file drove the whole
//! `attention` block through one public entry point and scored five goldens off one call. This
//! tree has no such entry point — `V4Engine::attention_block` is `pub(super)` to the V4 arm — so
//! what is portable is the half of it that drives these two launchers DIRECTLY, which is also
//! the half that attributes a failure to one kernel. The five-golden pipeline comparison belongs
//! to the layer loop's own gate and is named here as absent rather than quietly dropped.
//!
//! # One file, because the two share a convention the type checker cannot hold
//!
//! * `rope_adjacent` pairs `(x[2j], x[2j+1])` — `view_as_complex` — over the LAST `rd` dims of a
//!   row. Three plausible wrong versions have the same signature and produce fluent output:
//!   rotating every dim, rotating the FIRST `rd`, and the half-split pairing `(x[j], x[j+rd/2])`
//!   that `rope_split_half` implements for another model in this same tree. Each is one of the
//!   oracle's named defects and each is a host reference here.
//! * `gather_attn_shared_kv` is MQA over ONE `d`-wide entry that is both key and value for all
//!   `h` heads, gathered by an index list where `-1` masks a slot, with `attn_sink` entering the
//!   softmax DENOMINATOR only. It consumes exactly what the rotary produces.
//!
//! # What carries the power, and what does not
//!
//! The attend is scored against the frozen oracle's own `.attn_core_out` golden, driven from the
//! oracle's own `.q` and `.kv_entry` — so a disagreement cannot be blamed on an upstream
//! projection, and the selection comes from the ENGINE's `Sel::gather`, which means the
//! comparison also says the two implementations agree about WHICH rows a query may read.
//!
//! **Prefill only, and that is a real limit.** At prefill `sparse_attn` reads the prompt's own KV,
//! so `.kv_entry` IS the whole of what it attends. At decode it reads the ring, which is state
//! this file does not own; that path needs the layer loop's gate.
//!
//! **A LAYOUT permutation of the compressed region is invisible to this comparison and to every
//! other one built on these goldens** — measured in the reference tree by injecting it: moving a
//! compressed block from its slot to another slot the selection also names leaves all seven
//! goldens bit-identical, because the attend folds a softmax over a SET. That is a property of the
//! kernel, not a hole in the fixture, and no tightening here can close it.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: no `rocm` CI arm, and no GPU for this port. Two mutations in `kernels/`:
//!
//! * In `rivoli_rope_adjacent`, pair `(x[j], x[j + rd/2])` instead of `(x[2j], x[2j+1])` — i.e.
//!   make it `rope_split_half` over the same span. [`rope_adjacent_matches_the_reference_rotation`]
//!   must go RED, and its `half-split` arm of
//!   [`the_three_plausible_wrong_rotations_are_all_visible`] must go red the OTHER way (the kernel
//!   now agrees with the defect reference). The position-zero bitwise arm must stay GREEN, because
//!   at `pos = 0` every pairing rotates by the identity — that arm is there to say a green
//!   elsewhere is not coming from a fixture that never rotates.
//! * In `rivoli_gather_attn_shared_kv`, drop `sink` from the denominator.
//!   [`sparse_attn_alone_matches_the_oracle_including_the_sink`] must go RED, and the
//!   `SkipAttnSink` row of [`the_sink_defects_are_further_from_the_kernel_than_the_clean_oracle`]
//!   must FLIP — the kernel becomes the defect, so its distance to the defect oracle collapses
//!   and its distance to the clean one grows. Both directions matter: a mutation that reddens the
//!   positive gate while leaving the separation row unchanged has broken something other than the
//!   sink.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_backend::hip::{device_sync, launch_gather_attn_shared_kv, launch_rope_adjacent};
use rivoli_engine::device::DeviceBuf;
use rivoli_engine::v4::geometry::LayerKind;
use rivoli_engine::v4::select::{Extent, Sel};
use rivoli_oracles::v4oracle::forward::{Capture, Defect, Oracle};
use rivoli_oracles::v4oracle::weights::fixed_bf16;

mod common;
use common::{
    Got, Prefill, Want, assert_bits, assert_guard, assert_rel, assert_separates, back, dev, f32b,
    f32v, flat_freqs, prefill, toy_fixture, worst_rel,
};

/// The ratio-0 layer the attend is scored on. Toy layer 0 has `compress_ratio == 0`: no
/// compressor, no indexer, no YaRN, base `rope_theta` — so `Sel::gather` yields window columns
/// alone and every index refers to a row of `.kv_entry`. That is exactly what makes the isolated
/// drive possible, and it is also the least representative layer class in the model, which is why
/// the compressed classes are `kernel_v4_compress.rs`'s.
const LAYER: usize = 0;

/// Prompt long enough to OUTRUN the toy's 8-slot window, so the selection carries masked slots
/// and a wrapped ring rather than a dense prefix. A prompt that fits the window makes the mask
/// structurally unable to matter.
const PROMPT: usize = 12;

/// The attend's bar, relative to the largest expected element.
///
/// The kernel reduces — an online softmax over `cols` gathered rows — where the oracle folds
/// sequentially, so this cannot be bitwise and a bitwise gate here would reject correct code. Two
/// bf16 ulps: the reference stores bf16 at every step, so a re-associated f32 sum that lands on
/// the other side of a rounding boundary moves an element by one whole ulp rather than by its own
/// magnitude, and one ulp is the floor. The observed error is PRINTED beside the bar on every run
/// — a green comparison that passed on 100x of headroom looks exactly like one that passed on 2x.
const ATTEND_TOL: f32 = 1.0 / 128.0;

/// The rotary's bar, and it is far tighter than [`ATTEND_TOL`] because there is no reduction in
/// it: each output element is `a·cos − b·sin`, two products and one sum, over values the host and
/// the device read from the SAME uploaded table — and both sides then store the row bf16-rounded
/// (the kernel's trailing sweep; the oracle's `round_bf16`). On the bf16 grid the two agree
/// BITWISE unless an FMA-contraction difference of a few f32 ulp lands astride a bf16 rounding
/// boundary, which at this fixture it does not; `1e-6` admits contraction and rejects a
/// one-bf16-ulp store difference (3.9e-3), which is the failure it exists to catch.
///
/// Position zero is asserted BITWISE instead — see
/// [`rope_adjacent_matches_the_reference_rotation`] — because there the rotation is the identity
/// under any contraction, so an inexact result there is not a rounding difference at all.
const ROPE_TOL: f32 = 1.0e-6;

/// The round trip's bar, and it CANNOT be [`ROPE_TOL`]: the forward store quantizes the
/// intermediate to the bf16 grid, and the inverse rotation transports that quantization
/// faithfully — each pair's rotation is an isometry — so the honest round-trip error is the
/// intermediate's rounding, half a bf16 ulp of the row's scale, and no f32 care inside the
/// kernel can shrink it. `2^-8` is one bf16 ulp at unit scale: derived from the format like
/// `kernel_v4_quant.rs`'s e4m3 ulp, not measured, and a kernel that rotated the wrong span or
/// forgot to conjugate misses it by orders of magnitude.
const ROPE_ROUNDTRIP_TOL: f32 = 3.906_25e-3;

/// Drive one PREFILL `run_layer` of [`LAYER`] under `defect` and return what it captured.
///
/// The activations are drawn through the ORACLE's own `fixed_bf16`, seeded by name, so a rerun
/// compares the same numbers and a defect run and a clean run see the identical input — which is
/// what makes the difference between two captures attributable to the defect.
fn capture(defect: Defect) -> Capture {
    let fx = toy_fixture();
    let o = Oracle::new(fx.0.clone(), defect);
    prefill(
        fx,
        Prefill {
            o: &o,
            layer: LAYER,
            tag: "attn-h",
            s: PROMPT,
            scale: 0.5,
        },
    )
    .0
}

/// One golden, by suffix. A missing one is the oracle no longer emitting it, which is a fixture
/// failure and not a comparison failure — so it panics with the name rather than returning empty.
fn golden(cap: &Capture, suffix: &str) -> Vec<f32> {
    cap.float(&format!("L{LAYER}.pre.{suffix}"))
        .unwrap_or_else(|| panic!("golden L{LAYER}.pre.{suffix} is missing"))
        .to_vec()
}

/// The prefill selection for [`LAYER`]: the `(rows, cols)` shape and the uploaded index buffer,
/// from ONE fill.
///
/// The pairing is why this is one function: the kernel reads the buffer at the stride the SHAPE
/// names, so a caller that uploaded one fill's indices beside another fill's shape would hand it a
/// selection indexing off the end of its own buffer — silently, since every value in range is a
/// legal row.
///
/// `index_topk` is 0 because `Sel::n_comp` reads it only under `kind.has_indexer()`, which `Plain`
/// is not — so the value is UNREACHABLE rather than chosen, and a nonzero one here would be a
/// number with no argument behind it.
fn prefill_selection(win: usize) -> ((usize, usize), DeviceBuf) {
    let mut idx = Vec::new();
    let shape = Sel {
        win,
        kind: LayerKind::Plain,
        index_topk: 0,
        at: Extent {
            seqlen: PROMPT,
            start_pos: 0,
        },
    }
    .gather(&mut idx)
    .expect("a ratio-0 prefill selection");
    let bytes: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert!(
        idx.iter().any(|&v| v < 0),
        "the prompt must outrun the window, or the masked-slot path is unexercised"
    );
    (shape, dev(&bytes))
}

/// `gather_attn_shared_kv` over one prefill, from the oracle's own `q` and `kv`.
fn device_attend(q: &[f32], kv: &[f32], sink: &[f32], win: usize) -> Vec<f32> {
    let (cfg, _, _) = toy_fixture();
    let (dq, dkv, dsink) = (dev(&f32b(q)), dev(&f32b(kv)), dev(&f32b(sink)));
    let ((rows, cols), idxb) = prefill_selection(win);
    assert_eq!(rows, PROMPT, "one selection row per prompt token");
    let mut out = dev(&vec![0u8; PROMPT * cfg.n_heads * cfg.head_dim * 4]);
    // SAFETY: `q` is `m·h·d` f32 and `kv` is `d` f32 per row for at least `max(idxs)+1` rows —
    // both are the oracle's own goldens at exactly those shapes; `sink` is `h`; `idxs` is
    // `m·topk` i32 from one `Sel::gather` fill, whose `(rows, cols)` this passes verbatim; `o` is
    // `m·h·d` writable. Every buffer is a distinct allocation, and all outlive the join in `back`.
    unsafe {
        launch_gather_attn_shared_kv(
            dq.ptr().cast(),
            dkv.ptr().cast(),
            dsink.ptr().cast(),
            idxb.ptr().cast(),
            rows,
            cfg.n_heads,
            cfg.head_dim,
            cols,
            (cfg.head_dim as f32).powf(-0.5),
            out.ptr_mut().cast(),
            std::ptr::null_mut(),
        )
    }
    .expect("gather_attn_shared_kv");
    f32v(&back(&out))
}

/// `sparse_attn` alone, driven from the oracle's own `.q` and `.kv_entry`.
///
/// Isolated because it is the only stage whose output the layer loop overwrites in place, and
/// because it is where `attn_sink` lives: feeding the oracle's exact inputs means a disagreement
/// here cannot be blamed on an upstream projection.
#[test]
fn sparse_attn_alone_matches_the_oracle_including_the_sink() {
    let (cfg, m, _) = toy_fixture();
    let cap = capture(Defect::None);
    let (q, kv) = (golden(&cap, "q"), golden(&cap, "kv_entry"));
    let got = device_attend(&q, &kv, &m.layers[LAYER].attn_sink, cfg.window_size);
    assert_rel(
        &golden(&cap, "attn_core_out"),
        &got,
        "attn_core_out, prefill",
        ATTEND_TOL,
    );
}

/// The two sink breakages are further from the kernel than the clean oracle is.
///
/// Neither is expressible as a change to a kernel INPUT — `SkipAttnSink` drops the sink term from
/// the denominator and `AttnSinkNotMaxShifted` exponentiates it unshifted, and both live inside
/// the softmax — so this is the weaker instrument: it proves the COMPARISON has resolution, not
/// that this kernel would fail if broken in that specific way. It is what is available without
/// shipping a break switch in a kernel.
///
/// The `q` and `kv` handed to the device stay the CLEAN ones in both rows, which is what makes the
/// distance attributable: both defects are inside the attend, so a defect capture's own `.q` and
/// `.kv_entry` are bit-identical to the clean ones, and the comparison would be unchanged if they
/// were used. Asserting that is cheaper than arguing it, so the first two lines do.
#[test]
fn the_sink_defects_are_further_from_the_kernel_than_the_clean_oracle() {
    let (cfg, m, _) = toy_fixture();
    let clean = capture(Defect::None);
    let (q, kv) = (golden(&clean, "q"), golden(&clean, "kv_entry"));
    let got = device_attend(&q, &kv, &m.layers[LAYER].attn_sink, cfg.window_size);
    let base = worst_rel(Got(&got), Want(&golden(&clean, "attn_core_out")));
    for defect in [Defect::SkipAttnSink, Defect::AttnSinkNotMaxShifted] {
        let broken = capture(defect);
        assert_bits(
            &q,
            &golden(&broken, "q"),
            "a sink defect moved the QUERY path — the distance below would not be about the sink",
        );
        assert_bits(
            &kv,
            &golden(&broken, "kv_entry"),
            "a sink defect moved the KV",
        );
        let far = worst_rel(Got(&got), Want(&golden(&broken, "attn_core_out")));
        println!("{defect:?}: clean={base:.3e} defect={far:.3e}");
        assert!(
            far > 100.0 * base.max(1e-30),
            "{defect:?} is only {far:.3e} away where the clean oracle is {base:.3e} — the \
             comparison cannot resolve it, so a kernel carrying it might well pass"
        );
    }
}

/// The C-ABI argument guards on the attend, by CODE.
///
/// The code, not `is_err`: a check that accepted any error would still pass if an unrelated
/// dimension guard started swallowing the case first. The two cases differ ONLY in `head_dim` and
/// `topk`; the other nine arguments are stand-ins the guard returns before reading, which puts
/// the boundary on one line each instead of burying it in the eleventh and twelfth positions of a
/// repeated list.
#[test]
fn the_attend_guards_reject_out_of_domain_shapes() {
    let mut b = dev(&vec![0u8; 64 * 4]);
    let (p, pm) = (b.ptr().cast::<f32>(), b.ptr_mut().cast::<f32>());
    // SAFETY: every call is rejected by an argument guard before `hipLaunchKernelGGL`, so no
    // pointer is dereferenced and the shapes never have to be real. Null stream for the same
    // reason: there is no launch for a stream to order.
    let at = |head_dim, topk| unsafe {
        launch_gather_attn_shared_kv(
            p,
            p,
            p,
            b.ptr().cast(),
            1,
            1,
            head_dim,
            topk,
            1.0,
            pm,
            std::ptr::null_mut(),
        )
    };
    // head_dim past the accumulator cap — silently dropped dims otherwise.
    assert_guard(at(1025, 8), Some(1002), "head_dim over the accumulator cap");
    // ...and 1024 exactly is accepted by THAT guard, so the cap is a boundary and not a blanket
    // no: this case is refused by the topk guard instead, which is the point of pairing them.
    assert_guard(at(1024, 1 << 20), Some(1006), "a topk that overflows LDS");
}

// =======================================================================================
// the rotary
// =======================================================================================

/// One `rope_adjacent` launch's geometry.
///
/// A struct because five bare `usize` in a row are each plausible in another's position, and the
/// failure is not a panic: `row_len` and `rd` transposed rotates the whole row, `pos0` and
/// `rows_per_pos` transposed rotates every row at the same position. Both produce finite output.
#[derive(Clone, Copy)]
struct Rope {
    rows: usize,
    row_len: usize,
    /// Only the LAST this-many dims are rotated. `rd < row_len` is what makes the "all dims"
    /// defect distinguishable at all.
    rd: usize,
    pos0: usize,
    rows_per_pos: usize,
}

/// Which of the four rotations a host reference implements — the reference's own, and the three
/// plausible wrong ones the oracle names.
///
/// An enum matched exhaustively rather than three booleans: the three defects are mutually
/// exclusive readings of `apply_rotary_emb`, and a flag per defect would let a caller ask for two
/// at once, which is a rotation no implementation could have.
/// A whole rotation: which pairing, and which DIRECTION.
///
/// The two travel together because `inverse` is only meaningful about a chosen pairing, and
/// because a `bool` at the end of an argument list is unreadable at the call — `host_rope(.., how,
/// true)` says nothing, `Rotation { how, inverse: true }` says which `true`.
#[derive(Clone, Copy, Debug)]
struct Rotation {
    how: Pairing,
    inverse: bool,
}

impl Rotation {
    /// The reference's own rotation, forward — the only one three of the four tests want.
    fn reference() -> Self {
        Self {
            how: Pairing::Adjacent,
            inverse: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pairing {
    /// `(x[2j], x[2j+1])` over the LAST `rd` dims — `view_as_complex`, the reference.
    Adjacent,
    /// The same pairing applied to EVERY dim of the row (`Defect::RopeAllDims`).
    AllDims,
    /// The same pairing applied to the FIRST `rd` dims (`Defect::RopeFirstDims`).
    FirstDims,
    /// `(x[j], x[j + rd/2])` over the last `rd` dims — transformers' `rotate_half`, which this
    /// tree implements for another model under `rope_split_half` (`Defect::RopeHalfSplit`).
    HalfSplit,
}

/// `apply_rotary_emb` on the host, in whichever [`Pairing`] is asked for.
///
/// The table is read the way the kernel reads it — `tbl[pos * rd + 2j]` is the cosine and
/// `+ 1` the sine — so table CONSTRUCTION is out of scope on both sides and every difference
/// found is the rotation. `inverse` conjugates, which is the output de-rotation.
///
/// The row is stored **bf16-rounded**, because both the kernel (its trailing `rbf16` sweep)
/// and the frozen oracle (`round_bf16` after `apply_rotary_emb`, `attention.rs`) store it
/// that way. Measured before this line existed, 2026-08-16 on device: omitting it reads as
/// `err=3.886e-3` on the forward gate — one bf16 ulp at the fixture's scale, the store
/// masquerading as a rotation defect.
fn host_rope(x: &mut [f32], tbl: &[f32], g: Rope, r: Rotation) {
    let Rotation { how, inverse } = r;
    // The span and the stride between the two elements of a pair — the whole content of the four
    // variants, chosen once so the loop below cannot disagree with the choice.
    let (off, span, stride) = match how {
        Pairing::Adjacent => (g.row_len - g.rd, g.rd, 1),
        Pairing::AllDims => (0, g.row_len, 1),
        Pairing::FirstDims => (0, g.rd, 1),
        Pairing::HalfSplit => (g.row_len - g.rd, g.rd, g.rd / 2),
    };
    for r in 0..g.rows {
        let pos = g.pos0 + r / g.rows_per_pos;
        let row = &mut x[r * g.row_len..(r + 1) * g.row_len];
        for j in 0..span / 2 {
            // The table is indexed by the FREQUENCY index, which is `j` in every variant — the
            // pairing changes which two elements share a frequency, not which frequency they get.
            let (c, mut s) = (tbl[pos * g.rd + 2 * j], tbl[pos * g.rd + 2 * j + 1]);
            if inverse {
                s = -s;
            }
            let (lo, hi) = match stride {
                1 => (off + 2 * j, off + 2 * j + 1),
                _ => (off + j, off + j + stride),
            };
            let (a, b) = (row[lo], row[hi]);
            row[lo] = a * c - b * s;
            row[hi] = a * s + b * c;
        }
        for v in row.iter_mut() {
            *v = rivoli_core::num::bf16_to_f32(rivoli_core::num::f32_to_bf16(*v));
        }
    }
}

/// `rope_adjacent` on the device, in place, over a copy of `x`.
fn device_rope(x: &[f32], tbl: &[f32], g: Rope, inverse: bool) -> Vec<f32> {
    let (mut xb, tb) = (dev(&f32b(x)), dev(&f32b(tbl)));
    // SAFETY: `xb` is `rows * row_len` live writable f32 (the caller's fixture, checked by the
    // assertion in every test below) and `tb` covers positions `pos0 ..= pos0 + rows/rows_per_pos`
    // at stride `rd`; the two are distinct allocations and both outlive the join inside `back`.
    unsafe {
        launch_rope_adjacent(
            xb.ptr_mut().cast(),
            tb.ptr().cast(),
            g.rows,
            g.row_len,
            g.rd,
            g.pos0,
            g.rows_per_pos,
            inverse,
            std::ptr::null_mut(),
        )
    }
    .expect("rope_adjacent");
    f32v(&back(&xb))
}

/// The fixture every rotary test drives: the geometry, the activation, and the ORACLE's own
/// rotary table for [`LAYER`].
///
/// The table is the oracle's rather than the engine's `rope::table`, deliberately: both sides here
/// read the SAME uploaded bytes, so construction is out of scope, and taking the reference's
/// removes the one way this file could disagree with the goldens next door for a reason that is
/// not the rotation.
fn rope_case() -> (Rope, Vec<f32>, Vec<f32>) {
    let (cfg, _, o) = toy_fixture();
    // `rows_per_pos > 1` because the real caller rotates every HEAD of one token at one position,
    // and `rows / rows_per_pos > 1` so more than one position is reached — at a single position
    // the `pos0 + r / rows_per_pos` arithmetic is unexercised.
    let g = Rope {
        rows: cfg.n_heads * 3,
        row_len: cfg.head_dim,
        rd: cfg.rope_head_dim,
        pos0: 1,
        rows_per_pos: cfg.n_heads,
    };
    assert!(
        g.rd < g.row_len && g.rows / g.rows_per_pos > 1,
        "the fixture must leave an un-rotated prefix and span more than one position"
    );
    let x = fixed_bf16("rope-x", g.rows * g.row_len, 1.0);
    (g, x, flat_freqs(o.freqs(LAYER)))
}

/// `rope_adjacent` reproduces `apply_rotary_emb` — and at position zero it does so BITWISE.
///
/// Two claims with two units, and the second is why the first is not vacuous. At `pos = 0` the
/// table is `(1, 0)`, so the rotation is `a·1 − b·0` — exact under any FMA contraction, in every
/// pairing — and a bitwise comparison there says the kernel wrote the row it was given rather than
/// something merely close to it. Away from zero both sides land on the bf16 grid, where the only
/// disagreement [`ROPE_TOL`] admits is contraction below a rounding boundary.
#[test]
fn rope_adjacent_matches_the_reference_rotation() {
    let (g, x, tbl) = rope_case();
    let mut want = x.clone();
    host_rope(&mut want, &tbl, g, Rotation::reference());
    assert_rel(
        &want,
        &device_rope(&x, &tbl, g, false),
        "rope_adjacent, forward",
        ROPE_TOL,
    );
    // The fixture must actually ROTATE, or every comparison here is between two copies of `x`.
    assert!(
        want.iter().zip(&x).any(|(a, b)| a != b),
        "the rotation moved nothing — `pos0` or the table is degenerate"
    );

    // Position zero, bitwise, and the un-rotated prefix with it: `rope_adjacent` must leave dims
    // `[0, row_len - rd)` untouched at EVERY position, which is the half that separates it from
    // `Pairing::AllDims` without needing a tolerance at all.
    let at0 = Rope { pos0: 0, ..g };
    let got0 = device_rope(&x, &tbl, at0, false);
    assert_bits(
        &x[..g.rows_per_pos * g.row_len],
        &got0[..g.rows_per_pos * g.row_len],
        "position 0 is the identity rotation",
    );
    for r in 0..g.rows {
        let head = r * g.row_len;
        assert_bits(
            &x[head..head + g.row_len - g.rd],
            &got0[head..head + g.row_len - g.rd],
            "the un-rotated prefix of a row must be copied verbatim",
        );
    }
}

/// The de-rotation is the exact inverse of the rotation.
///
/// `inverse` is one `bool` on a launcher whose forward and backward calls are otherwise identical,
/// and getting it wrong is `Defect::OutputDerotationForward` — the output rotated a second time
/// instead of back, which is fluent and wrong. A round trip is the strongest statement available
/// about it without a second reference, and it needs [`ROPE_ROUNDTRIP_TOL`] rather than bits or
/// [`ROPE_TOL`] because the forward store bf16-quantized the intermediate.
#[test]
fn the_inverse_rotation_undoes_the_forward_one() {
    let (g, x, tbl) = rope_case();
    let forward = device_rope(&x, &tbl, g, false);
    assert_rel(
        &x,
        &device_rope(&forward, &tbl, g, true),
        "rope forward then inverse",
        ROPE_ROUNDTRIP_TOL,
    );
    // ...and rotating TWICE forward must NOT come back, or the fixture's angles are multiples of
    // pi and the round trip above proves nothing about the sign of the sine.
    assert_separates(
        &x,
        &device_rope(&forward, &tbl, g, false),
        "rope forward twice",
        ROPE_ROUNDTRIP_TOL,
    );
}

/// The three plausible wrong rotations are each visible to the comparison above.
///
/// Each is one of the oracle's named defects and each has the SAME signature as the reference, so
/// nothing structural separates them. The kernel must be far from all three at the same tolerance
/// [`rope_adjacent_matches_the_reference_rotation`] passes at — a break that moved the result by
/// less than that is a break the positive gate would not have caught.
#[test]
fn the_three_plausible_wrong_rotations_are_all_visible() {
    let (g, x, tbl) = rope_case();
    let got = device_rope(&x, &tbl, g, false);
    for how in [Pairing::AllDims, Pairing::FirstDims, Pairing::HalfSplit] {
        let mut broken = x.clone();
        host_rope(
            &mut broken,
            &tbl,
            g,
            Rotation {
                how,
                ..Rotation::reference()
            },
        );
        assert_separates(&broken, &got, &format!("{how:?} rotation"), ROPE_TOL);
    }
}

/// The rotary's guards, by CODE.
///
/// `view_as_complex` cannot pair an odd count, and a rotary span wider than the row it sits in
/// would read another row's dims — in bounds, finite, and wrong.
#[test]
fn the_rotary_guards_reject_shapes_view_as_complex_cannot_pair() {
    let mut b = dev(&vec![0u8; 64 * 4]);
    let (p, pm) = (b.ptr().cast::<f32>(), b.ptr_mut().cast::<f32>());
    // SAFETY: each call is rejected by an argument guard before any launch.
    let at = |row_len, rd| unsafe {
        launch_rope_adjacent(pm, p, 1, row_len, rd, 0, 1, false, std::ptr::null_mut())
    };
    assert_guard(at(8, 3), Some(1005), "odd rope_head_dim");
    assert_guard(at(8, 16), Some(1002), "rope span over the row");
    // The shape the model runs is accepted, so the two guards above are not refusing everything.
    let (cfg, _, _) = toy_fixture();
    assert_guard(
        at(cfg.head_dim, cfg.rope_head_dim),
        None,
        "the shipped rotary span",
    );
    device_sync().expect("device sync"); // the accepted case LAUNCHED — join before `b` drops
}
