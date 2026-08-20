//! **`gqa_block_attend` — the DFlash drafter's bidirectional attend, scored against a host
//! oracle.** M17c's on-device gate, and the row that retires the kernel census's third DEFERRED
//! turn.
//!
//! # Why this is a separate file from `kernel_glimmer_attend.rs`
//!
//! Not because the arithmetic differs — it does not; the softmax is literally the same
//! `common::attend_head`, hoisted here on 2026-08-17 when this became its second consumer. It is
//! separate because **every arm below is about a SPAN**, and the spans are the whole difference
//! between a causal decode attend and a block drafter:
//!
//! | | lower edge | upper edge |
//! |---|---|---|
//! | `gqa_attend` (causal) | `window_lo(pos, win)` = `pos - win + 1`, **strict** | `pos` |
//! | `gqa_block_attend` | `pos - win`, **inclusive** | `min(kv_len - 1, pos + win)` |
//!
//! Both edges move, and neither move is visible in a shape, a byte count, or a dtype.
//!
//! # What carries the power, and what provably cannot
//!
//! **THE VENDORED S1b GOLDENS CANNOT GATE THIS KERNEL.** That is measured, not suspected, and it
//! is why this file exists as a host-oracle comparison rather than a golden comparison:
//!
//! * **The `q_offset` branch.** The reference's overlay is `abs(q_idx - kv_idx) <= win` with
//!   `q_idx = row + q_offset`, and `masking_utils.py` takes `q_offset` from the cache when one is
//!   present and **0** when it is not. The anchor ran `use_cache=False`, so every golden pins the
//!   `q_offset = 0` branch — which at any `ctx > win` attends **no block row at all**. Scoring
//!   against a golden therefore validates the branch the drafter must NOT use in decode.
//! * **The lower edge.** At the fixture's ctx 12 / win 13 the lower bound is negative for every
//!   row and clamps to 0, so `pos - win` and `pos - win + 1` agree on every cell. **0 of its 4
//!   query rows change.** No golden can ever catch that off-by-one.
//! * **The scale.** The drafter has no `qk_scale_factor`; the target's 3.87 applied here is a
//!   plain multiplier on every logit, and at toy widths with drawn weights it stays finite and
//!   plausible.
//!
//! `crates/cli/tests/drafter_convert.rs` gates the two SPAN facts deviceless, from the shipped
//! config, as geometry. This file gates the same facts as VALUES, on the device, against a host
//! that computes each span explicitly. `glimmer-reference/drafter-checkpoint.md` carries the
//! measurements and the arithmetic.
//!
//! # The widths, and why they are not the anchor's
//!
//! `ctx 32 / block 16 / win 8`, so `ctx > win` and every arm above is LIVE — which the anchor
//! geometry is precisely what fails to be. `hq 32 / hkv 8 / head_dim 128` are the shipped
//! drafter's own, and `32 * 128 = 4096 != 6656` keeps §9 trap 15 (head_dim is not
//! `hidden / heads`) visible.

#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli_backend::hip::launch_gqa_block_attend;

mod common;
use common::{
    AttendCase, AttendIo, AttendSpan, Lcg, assert_guard, assert_separates, attend_head,
    attend_launch, attn_scale, back, dev, draws, f32b, f32v, ok, window_lo,
};

/// `tolerance::GLIMMER`'s `attend` row — `Rel(1.64e-4)` over a measured floor of 1.639e-5.
///
/// The SAME row `kernel_glimmer_attend.rs` uses, and deliberately: this is the same bucket, the
/// same operator and the same reduction shape (lane `l` owns columns `i ≡ l (mod 32)`, partials
/// meeting in a `__shfl_down` ladder against the host's sequential sum), so the row that prices
/// that re-association prices this one. **A new tolerance here would be a number with no
/// measurement behind it**, and the draft-mode floors in
/// `crates/oracles/tests/glimmer_draft_oracle.rs` are not it — those price a CPU f64 oracle
/// against fp32 goldens, a different comparison entirely.
const ATTEND_TOL: f32 = 1.64e-4;

/// The shipped drafter's own attention widths (`assistant-config.json`).
const HQ: usize = 32;
const HKV: usize = 8;
const D: usize = 128;
/// `block_size` is 16 in the shipped config; this is that.
const BLOCK: usize = 16;
/// **Not the shipped 2048.** The arms need `ctx > win` for the block to be out of reach of a
/// row-indexed query, and a 2048 window at a context past it would make every launch a full
/// sweep for no extra coverage. 8 with `ctx 32` puts every edge in play.
const WIN: usize = 8;
const CTX: usize = 32;
const KV_LEN: usize = CTX + BLOCK;

/// The target's `qk_scale_factor`. The drafter does not have one; §11's drafter is a plain GQA
/// block. Applied here it is the "leaked from the target" defect.
const TARGET_QK_SCALE: f32 = 3.87;

/// q, k and v for one case, in one draw order — a seed means the same data at two call sites only
/// while the order is shared.
fn operands() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = Lcg(0x5f3d_c071);
    (
        draws(&mut r, BLOCK * HQ * D),
        draws(&mut r, KV_LEN * HKV * D),
        draws(&mut r, KV_LEN * HKV * D),
    )
}

// `BlockIo` MOVED to `common::AttendIo` on 2026-08-17: `kernel_glimmer_attend.rs`'s `GqaIo` is
// the same four fields, and jscpd reported the pair as a cross-file clone the moment this file
// existed. Its constructor takes the four BUFFERS, which removed a second clone — both call sites
// here were spelling the same `ptr() as *const f32` casts.

// `block_launch` MOVED to `common::attend_launch` on 2026-08-17. `gqa_attend` and
// `gqa_block_attend` take the SAME thirteen arguments in the same order and differ only in what
// two of them mean, so the two wrappers were a clone jscpd correctly refused — and the shared type
// says the thing the two wrappers only implied.

/// Launch the kernel and read the result back.
///
/// `q_offset` and `scale` are the caller's because they are exactly what the arms vary. Nothing
/// here derives either: a helper that computed `q_offset` from `CTX` would make the trap
/// unspellable in a test, which is the opposite of what this file is for.
fn run(qkv: &(Vec<f32>, Vec<f32>, Vec<f32>), q_offset: usize, scale: f32) -> Vec<f32> {
    let (q, k, v) = qkv;
    let (qb, kb, vb) = (dev(&f32b(q)), dev(&f32b(k)), dev(&f32b(v)));
    let mut ob = dev(&vec![0u8; BLOCK * HQ * D * 4]);
    // SAFETY: buffers of exactly the sizes `block_launch` requires, `ob` distinct from all three
    // (the borrow checker holds that at `AttendIo::new`), and all four outlive the `device_sync`
    // inside `back`.
    let r = unsafe {
        let io = AttendIo::new(&qb, &kb, &vb, &mut ob);
        attend_launch(
            launch_gqa_block_attend,
            io,
            [HQ, HKV, D, BLOCK, q_offset, WIN, KV_LEN],
            scale,
        )
    };
    ok(r, "gqa_block_attend");
    f32v(&back(&ob))
}

/// Which span rule a host run should use — the axis every arm in this file moves along.
#[derive(Clone, Copy)]
enum Span {
    /// The reference's bidirectional overlay: `abs(q - kv) <= win`, inclusive both ends.
    Bidirectional,
    /// The bidirectional upper edge with the CAUSAL lower edge — the one-row defect a kernel
    /// author reproduces by carrying `gqa_attend`'s `lo` across. Invisible to every golden.
    StrictLowerEdge,
    /// `gqa_attend`'s span outright: strict lower edge AND `pos` as the upper bound. What the
    /// kernel would compute if it had simply not been written.
    Causal,
}

/// Which KV head a Q head reads. The kernel does `head / group`; the interleave is trap 10.
#[derive(Clone, Copy)]
enum Broadcast {
    Block,
    Interleaved,
}

/// The host oracle: `BLOCK * HQ` rows, each a softmax over the span `rule` prescribes.
fn host(
    qkv: &(Vec<f32>, Vec<f32>, Vec<f32>),
    q_offset: usize,
    rule: Span,
    bc: Broadcast,
) -> Vec<f32> {
    let (q, k, v) = qkv;
    let group = HQ / HKV;
    let ac = AttendCase {
        q,
        k,
        v,
        hkv: HKV,
        d: D,
    };
    let mut out = vec![0.0; BLOCK * HQ * D];
    for n in 0..BLOCK * HQ {
        let (row, h) = (n / HQ, n % HQ);
        let pos = q_offset + row;
        let span = match rule {
            // `saturating_sub`, not `pos - win`: the inclusive lower edge goes negative for a
            // query near the front and clamps to 0, which is exactly why the anchor fixture
            // cannot distinguish this rule from the next one.
            Span::Bidirectional => (pos.saturating_sub(WIN), (pos + WIN).min(KV_LEN - 1)),
            Span::StrictLowerEdge => (window_lo(pos, WIN), (pos + WIN).min(KV_LEN - 1)),
            Span::Causal => (window_lo(pos, WIN), pos.min(KV_LEN - 1)),
        };
        let kvh = match bc {
            Broadcast::Block => h / group,
            Broadcast::Interleaved => h % HKV,
        };
        out[n * D..][..D].copy_from_slice(&attend_head(&ac, AttendSpan { n, kvh, span }));
    }
    out
}

/// The reference span at the offset a real decode uses — the case every other arm is measured
/// against.
fn reference(qkv: &(Vec<f32>, Vec<f32>, Vec<f32>)) -> Vec<f32> {
    host(qkv, CTX, Span::Bidirectional, Broadcast::Block)
}

#[test]
fn block_attend_matches_a_host_oracle_at_the_drafter_widths() {
    let qkv = operands();
    let got = run(&qkv, CTX, attn_scale(D));
    common::assert_rel(&reference(&qkv), &got, "gqa_block_attend", ATTEND_TOL);
}

/// **The block attends LATER rows of its own block, and a causal span does not.** §11 step 5 as a
/// value comparison rather than a mask pattern — the property the whole kernel exists for.
#[test]
fn the_span_is_bidirectional_and_a_causal_one_is_a_different_operator() {
    let qkv = operands();
    let got = run(&qkv, CTX, attn_scale(D));
    // The positive direction is the test above; this arm's content is that the CAUSAL host — the
    // span `gqa_attend` would have derived from the same `(pos, win)` — does not describe what
    // the device computed. Through `assert_separates`, so the break has to be visible at the
    // SAME tolerance the positive gate passes at.
    assert_separates(
        &host(&qkv, CTX, Span::Causal, Broadcast::Block),
        &got,
        "a causal span against the bidirectional kernel",
        ATTEND_TOL,
    );
}

/// **The lower edge is INCLUSIVE, and this is the arm no golden can carry.** One KV row per query
/// — the defect a kernel author reproduces by copying `gqa_attend`'s `lo = pos - win + 1`, whose
/// own comment calls that `+ 1` trap 14.
#[test]
fn the_inclusive_lower_edge_is_not_the_strict_one() {
    let qkv = operands();
    let got = run(&qkv, CTX, attn_scale(D));
    assert_separates(
        &host(&qkv, CTX, Span::StrictLowerEdge, Broadcast::Block),
        &got,
        "the strict causal lower edge against the inclusive bidirectional one",
        ATTEND_TOL,
    );
}

/// **`q_offset` selects between two different operators, and the goldens pin the wrong one.**
/// Both launches are legal, both produce finite output, and at `ctx > win` the `q_offset = 0` one
/// attends **no block row at all**. Asserted on device because the fixture answers this question
/// in the direction serving does not want.
#[test]
fn q_offset_zero_is_the_no_cache_branch_and_not_the_decode_one() {
    let qkv = operands();
    let at_ctx = run(&qkv, CTX, attn_scale(D));
    let at_zero = run(&qkv, 0, attn_scale(D));

    // Each matches its OWN host span, so neither is a broken launch — they are two operators.
    common::assert_rel(&reference(&qkv), &at_ctx, "q_offset = ctx", ATTEND_TOL);
    common::assert_rel(
        &host(&qkv, 0, Span::Bidirectional, Broadcast::Block),
        &at_zero,
        "q_offset = 0",
        ATTEND_TOL,
    );

    // And they are not each other.
    assert_separates(
        &at_zero,
        &at_ctx,
        "q_offset 0 against q_offset ctx",
        ATTEND_TOL,
    );

    // The reason, stated as a property rather than a magnitude: at `q_offset = 0` no query can
    // reach a key at `kv >= CTX`, so the whole block is out of every span. Checked here on the
    // SPANS the host used, so the claim is about the geometry rather than about float noise.
    for row in 0..BLOCK {
        let hi = (row + WIN).min(KV_LEN - 1);
        assert!(
            hi < CTX,
            "row {row} at q_offset 0 reaches kv {hi}, which is inside the block at {CTX} — the \
             widths no longer exercise the trap and this test is testing nothing"
        );
    }
}

/// The KV broadcast is a per-head BLOCK, not an interleave — trap 10, for this kernel's own
/// `head / group`.
#[test]
fn the_block_kv_broadcast_is_not_the_interleaved_one() {
    let qkv = operands();
    let got = run(&qkv, CTX, attn_scale(D));
    assert_separates(
        &host(&qkv, CTX, Span::Bidirectional, Broadcast::Interleaved),
        &got,
        "the interleaved KV broadcast",
        ATTEND_TOL,
    );
}

/// **The target's `qk_scale_factor` is not this kernel's scale.** The drafter has none; 3.87 on
/// every logit stays finite and plausible, which is what makes it worth a gate.
#[test]
fn the_targets_qk_scale_factor_is_not_the_drafters_scale() {
    let qkv = operands();
    let leaked = run(&qkv, CTX, attn_scale(D) * TARGET_QK_SCALE);
    assert_separates(
        &reference(&qkv),
        &leaked,
        "the target's qk_scale_factor leaked into the drafter",
        ATTEND_TOL,
    );
}

/// Every argument guard, by CODE. A refusal that fires for the wrong reason is not a refusal.
#[test]
fn the_block_attend_launcher_refuses_what_it_cannot_compute() {
    let qkv = operands();
    let (q, k, v) = &qkv;
    let (qb, kb, vb) = (dev(&f32b(q)), dev(&f32b(k)), dev(&f32b(v)));
    let mut ob = dev(&vec![0u8; BLOCK * HQ * D * 4]);
    let mut fire = |dims: [usize; 7]| {
        // SAFETY: every row below is REJECTED before `hipLaunchKernelGGL`, so no pointer is
        // dereferenced — which is what lets one buffer set serve the whole table. The last row
        // is the accepted one and its dims match the real buffers exactly.
        unsafe {
            attend_launch(
                launch_gqa_block_attend,
                AttendIo::new(&qb, &kb, &vb, &mut ob),
                dims,
                attn_scale(D),
            )
        }
    };
    // `win == 0`: 1001, NOT `gqa_attend`'s "global layer" reading. All five drafter layers are
    // `sliding_attention` and a bidirectional bound with win 0 is the diagonal — not an operator
    // this model has, so it is refused rather than silently computed.
    assert_guard(
        fire([HQ, HKV, D, BLOCK, CTX, 0, KV_LEN]),
        Some(1001),
        "win 0",
    );
    assert_guard(
        fire([HQ, HKV, D, BLOCK, CTX, WIN, 0]),
        Some(1001),
        "kv_len 0",
    );
    // The group must be exact, or `head / group` sends the tail of the Q heads to a KV head that
    // does not exist.
    assert_guard(
        fire([HQ, 5, D, BLOCK, CTX, WIN, KV_LEN]),
        Some(1003),
        "hq % hkv",
    );
    // Past the lane accumulator the tail of every head is silently never written.
    assert_guard(
        fire([HQ, HKV, 257, BLOCK, CTX, WIN, KV_LEN]),
        Some(1002),
        "head_dim past the accumulator",
    );
    // The block cannot be larger than the buffer it lives in.
    assert_guard(
        fire([HQ, HKV, D, BLOCK, CTX, WIN, BLOCK - 1]),
        Some(1005),
        "kv_len below tq",
    );
    // And it cannot be placed past the buffer entirely: with `q_offset` far beyond `kv_len` the
    // last row's `lo` exceeds `hi`, every span is empty, and the kernel would divide by a zero
    // softmax denominator. Refused here rather than poisoned there.
    assert_guard(
        fire([HQ, HKV, D, BLOCK, KV_LEN + WIN, WIN, KV_LEN]),
        Some(1005),
        "the block placed past its own K/V",
    );
    // The shipped shape is accepted, so the table above is rejecting arguments rather than
    // everything.
    assert_guard(
        fire([HQ, HKV, D, BLOCK, CTX, WIN, KV_LEN]),
        None,
        "the real shape",
    );
}
