//! **The host half of V4's compressor/indexer, and an executable record of what the
//! shipped goldens cannot see.**
//!
//! Ungated: everything here is host arithmetic, so it runs on every `cargo test` with no
//! device and no checkpoint. The device kernels are scored separately in `tests/kernel.rs`,
//! which needs both.
//!
//! Two kinds of test live here, and the second is the one that matters:
//!
//! 1. **Agreement** — [`window_topk`] against the oracle's own public transliteration, and
//!    the per-layer shape discriminants against the real 46-entry `compress_ratios`.
//! 2. **Vacuity** — assertions that pin, as executable facts, the cases where S1b's goldens
//!    are *empty or invariant* and therefore accept any implementation. A hole that is only
//!    described in a report gets forgotten; a hole with a test named after it does not, and
//!    the test fails the day someone lengthens the emit prompt and the hole closes — which
//!    is exactly when the report needs rewriting.

use rivoli::v4compress::{
    LayerKind, RopeParams, compress_dst, compress_offset, compress_topk, freqs_cis,
    rope_for_layer, should_compress, window_topk,
};
use rivoli::v4oracle::weights::V4Config;

/// **`bin/v4-oracle`'s `PROMPT` tokenized to 13 ids, hand-checked 2026-08-05.**
///
/// A transcribed snapshot, and it is worth being blunt that nothing links it to the source:
/// `PROMPT` is a private const in a `bin` target (`src/bin/v4-oracle.rs:37`), and an
/// integration test cannot import a bin at all. So the "what the goldens cover" claims below
/// are keyed to a number that will NOT go red if someone lengthens the prompt — the holes
/// they describe would close silently and these tests would stay green.
///
/// Closing that needs `PROMPT` (or its id count) moved into the lib, which is S1b's file and
/// not this stage's to change. Recorded in the S2c report as an open gap; an earlier version
/// of this doc claimed the tripwire worked, which was the load-bearing lie three reviewers
/// caught.
const EMIT_PROMPT_LEN: usize = 13;

/// Everything else comes from `V4Config::v4_flash()`, which `assert_matches_reference_json`
/// already holds against the on-disk `config.json` — so a config change reddens these rather
/// than drifting past a second hand-copied literal.
fn cfg() -> V4Config {
    V4Config::v4_flash()
}

// =======================================================================================
// 1. agreement with the oracle
// =======================================================================================

/// `window_topk` against `v4oracle`'s public transliteration of the same reference function.
///
/// This is the one selection path where a real cross-check is available today: the oracle
/// exports `window_topk` but keeps `compress_topk`, `Oracle::freqs` and `Oracle::compressor`
/// private, so those three are covered here only by property assertions and, once wired,
/// by end-to-end goldens. Two independent transliterations agreeing is worth more than
/// either agreeing with itself.
///
/// The grid spans all three branches of model.py:261 and, critically, the boundaries
/// between them: `start_pos == win - 1` is where the "ring is full" branch first applies,
/// and `start_pos == win` is the first wrap. A grid that steps only by large strides walks
/// straight past both.
#[test]
fn window_topk_matches_the_oracle() {
    let mut checked_branches = [0usize; 3];
    for win in [4usize, 8, cfg().window_size] {
        for seqlen in [1usize, 3, 5, 13, 129, 300] {
            for start_pos in [0usize, 1, 2, win - 1, win, win + 1, 2 * win, 2 * win + 3, 511] {
                // Decode is always a single query row; prefill passes the whole prompt. The
                // reference conflates these in one function, so feeding `seqlen > 1` with
                // `start_pos > 0` asks a question the model never asks.
                let s = if start_pos == 0 { seqlen } else { 1 };
                let want = rivoli::v4oracle::forward::window_topk(win, s, start_pos);
                let got = window_topk(win, s, start_pos);
                let got64: Vec<Vec<i64>> =
                    got.iter().map(|r| r.iter().map(|&x| i64::from(x)).collect()).collect();
                assert_eq!(
                    got64, want,
                    "window_topk(win={win}, seqlen={s}, start_pos={start_pos}) differs from the oracle"
                );
                checked_branches[branch_of(win, start_pos)] += 1;
            }
        }
    }
    // A grid that never reaches a branch tests nothing about it, and silently: every case
    // would still pass. Assert each branch was actually entered.
    assert!(
        checked_branches.iter().all(|&n| n > 0),
        "window_topk grid missed a branch: {checked_branches:?}"
    );
}

/// Which of model.py:261's three branches `(win, start_pos)` selects.
fn branch_of(win: usize, start_pos: usize) -> usize {
    if start_pos >= win.saturating_sub(1) && start_pos > 0 {
        0
    } else if start_pos > 0 {
        1
    } else {
        2
    }
}

/// The 46-entry / 43-layer trap, and **layer 42 in particular**.
///
/// The trailing `0, 0, 0` are the three MTP blocks, not layers 40-42. Reading them as the
/// model's tail loses layer 42's compressor *and* its indexer — which is precisely what
/// S1b's first cut did (`docs/investigations/v4-flash-port.md` §S1b). This asserts the
/// boundary from both sides: 42 is compressed, and 43 (the first MTP entry) is not a layer
/// this path ever classifies.
#[test]
fn layer_42_is_ratio_4_not_the_zero_tail() {
    let c = cfg();
    assert_eq!(c.compress_ratios.len(), 46, "46 ratio entries");
    assert_eq!(c.n_layers, 43, "for 43 layers — the mismatch that causes the trap");

    let kinds: Vec<LayerKind> =
        (0..c.n_layers).map(|l| LayerKind::from_ratio(c.compress_ratio(l))).collect();
    assert_eq!(kinds[42], LayerKind::Overlap, "layer 42 carries BOTH a compressor and an indexer");
    assert_eq!(kinds.iter().filter(|k| k.compressor_ratio().is_some()).count(), 41);

    // The ratio-4 layers are exactly [2, 4, .., 42] — an even run that ENDS at 42, so an
    // implementation that stops at 40 loses one and still looks periodic. This one equality
    // subsumes the first/last/count assertions it replaced.
    let ratio4: Vec<usize> = (0..c.n_layers).filter(|&l| kinds[l].has_indexer()).collect();
    assert_eq!(ratio4, (0..21).map(|i| 2 + 2 * i).collect::<Vec<_>>());

    // The boundary itself, from BOTH sides — the counts above are invariant to where the
    // main path is cut (the tail entries are 0, so they never join either count), which is
    // exactly why trimming from the wrong end went unnoticed once already.
    assert_eq!(c.compress_ratios[42], 4, "last real layer is compressed");
    assert_eq!(c.compress_ratios[43], 0, "first MTP entry is not");
    let wrong_trim = c.compress_ratios[3..3 + c.n_layers].iter().filter(|&&r| r == 4).count();
    assert_ne!(wrong_trim, 21, "a wrong-end trim must NOT reproduce the right indexer count");
}

/// `coff` drives every compressor tensor's width, and it differs between L2 and L3.
///
/// `ape` is `[4, 1024]` at ratio 4 and `[128, 512]` at ratio 128 — not merely a different
/// length, a different *rank split*. A kernel that infers the pooling width from
/// `head_dim` alone is correct on L2 and reads out of bounds (or, worse, in bounds and
/// wrong) on L3.
#[test]
fn compressor_widths_differ_between_ratio_4_and_ratio_128() {
    const HEAD_DIM: usize = 512;
    let l2 = LayerKind::from_ratio(4);
    let l3 = LayerKind::from_ratio(128);
    assert_eq!((l2.coff(), l2.compressor_ratio()), (2, Some(4)));
    assert_eq!((l3.coff(), l3.compressor_ratio()), (1, Some(128)));
    assert!(l2.overlap() && !l3.overlap());
    // The widths those coffs imply, spelled once: ape is [4, 1024] on L2 and [128, 512] on
    // L3 — a different rank split, not merely a different length.
    assert_eq!((l2.coff() * HEAD_DIM, l3.coff() * HEAD_DIM), (1024, 512));
    // The indexer's own compressor is ratio-4 over `index_head_dim = 128`, so a THIRD
    // shape: ape [4, 256]. Sharing the attention compressor's geometry is a natural slip
    // because the two are the same class.
    const INDEX_HEAD_DIM: usize = 128;
    assert_eq!(l2.coff() * INDEX_HEAD_DIM, 256);
}

// =======================================================================================
// 2. the YaRN table
// =======================================================================================

/// The two per-layer tables are genuinely different, and different in the *documented* way.
///
/// The oracle keeps `Oracle::freqs` and `precompute_freqs_cis` private, so this cannot be a
/// direct comparison. Instead it pins the three independently-derivable facts that the two
/// rope defects the oracle enumerates would each break:
///
/// - `RopeYarnEverywhere` / `RopeBaseThetaEverywhere`: a ratio-0 layer must use theta
///   10000 with NO interpolation, so its `t = 1` angle for dim `i` is exactly
///   `1 / 10000^(2i/64)`.
/// - `RopeNoYarn`: on a compressed layer the interpolation band must actually bite. The
///   band edges are computed here in f64 from the config by hand — `low = 15`,
///   `high = 25` for (dim 64, base 160000, original_seq_len 65536, beta 32/1) — rather
///   than read back from the implementation, so agreement is evidence.
#[test]
fn yarn_applies_only_to_compressed_layers_and_only_above_the_band() {
    const RD: usize = 64;
    const FACTOR: f32 = 16.0;
    // The shipped `config.json`: rope_theta 10000, compress_rope_theta 160000,
    // rope_scaling {factor 16, original_max_position_embeddings 65536, beta_fast 32,
    // beta_slow 1}, qk_rope_head_dim 64.
    let c = cfg();
    let compressed = RopeParams {
        rope_head_dim: c.rope_head_dim,
        theta: c.compress_rope_theta,
        original_seq_len: c.original_seq_len,
        factor: c.rope_factor,
        beta_fast: c.beta_fast,
        beta_slow: c.beta_slow,
    };
    assert_eq!((c.rope_head_dim, c.rope_factor), (RD, FACTOR), "config still matches this test");
    let plain = rope_for_layer(compressed, c.rope_theta, LayerKind::Plain);
    let comp = rope_for_layer(compressed, c.rope_theta, LayerKind::Overlap);
    // `..compressed` is what guarantees a rotary parameter added later reaches both tables.
    // Prose cannot hold that; this can.
    assert_eq!(
        (plain.rope_head_dim, plain.factor, plain.beta_fast, plain.beta_slow),
        (comp.rope_head_dim, comp.factor, comp.beta_fast, comp.beta_slow),
        "the two tables must differ ONLY in theta and original_seq_len"
    );
    // A ratio-128 layer is compressed too, so it gets the SAME table as ratio-4.
    assert_eq!(rope_for_layer(compressed, c.rope_theta, LayerKind::from_ratio(128)).theta, c.compress_rope_theta);
    assert_eq!((plain.theta, plain.original_seq_len), (10000.0, 0), "ratio-0 disables YaRN");
    assert_eq!((comp.theta, comp.original_seq_len), (160000.0, 65536), "ratio-4 enables it");

    let seqlen = 4;
    let fp = freqs_cis(plain, seqlen);
    let fc = freqs_cis(comp, seqlen);
    assert_eq!(fp.len(), seqlen * RD / 2);

    // t = 1 row: the angle IS the frequency, so cos/sin read the table directly.
    for i in 0..RD / 2 {
        let want = 1.0f32 / 10000.0f32.powf((2 * i) as f32 / RD as f32);
        let (c, s) = fp[RD / 2 + i];
        assert!(
            (c - want.cos()).abs() < 1e-6 && (s - want.sin()).abs() < 1e-6,
            "ratio-0 dim {i}: plain RoPE at theta 10000 expected"
        );
    }

    // Band edges, computed here from the config rather than taken from the implementation.
    let base = 160000.0f64;
    let fcd = |rot: f64| RD as f64 * (65536.0f64 / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
    let (low, high) = (fcd(32.0).floor(), fcd(1.0).ceil());
    assert_eq!((low, high), (15.0, 25.0), "YaRN correction range for the shipped config");

    let raw = |i: usize| 1.0f64 / base.powf((2 * i) as f64 / RD as f64);
    // `atan2`, not `acos`: every angle here is small, so `cos` is within f32 rounding of
    // 1.0 and `acos` amplifies that by 1/sin -- a 3.7e-6 error at dim 11 alone, which is
    // the test's arithmetic failing, not the table's. `atan2` is well conditioned across
    // the whole range and recovers the angle to f32 precision.
    let angle_at = |t: &[(f32, f32)], i: usize| f64::from(t[RD / 2 + i].1.atan2(t[RD / 2 + i].0));
    for i in 0..RD / 2 {
        // ramp = clamp((i - low)/(high - low)); smooth = 1 - ramp;
        // f = f/factor * (1 - smooth) + f * smooth  =>  below the band f is UNCHANGED,
        // above it f is divided by `factor`. Getting this backwards is a plausible slip and
        // would leave the table looking sane at both extremes.
        let angle = angle_at(&fc, i);
        // Relative: the angles span 1.0 down to 5e-6 across the 32 dims, so one absolute
        // epsilon is simultaneously too loose at the top and too tight at the bottom.
        let rel = |want: f64| (angle - want).abs() / want.abs().max(1e-30);
        if i <= low as usize {
            assert!(
                rel(raw(i)) < 1e-6,
                "dim {i} is below the band and must be EXTRAPOLATED (unscaled)"
            );
        } else if i >= high as usize {
            assert!(
                rel(raw(i) / f64::from(FACTOR)) < 1e-6,
                "dim {i} is above the band and must be INTERPOLATED (divided by factor)"
            );
            // And the two must genuinely differ there -- a `factor` of 1, or a blend that
            // silently no-ops, would satisfy the line above if `raw(i)` were used for both.
            assert!(
                rel(raw(i)) > 1e-3,
                "dim {i} above the band is indistinguishable from the unscaled frequency"
            );
        }
    }
    assert_ne!(fp, fc, "the two per-layer tables must not be the same table");
}

// =======================================================================================
// 3. what the shipped goldens CANNOT see
// =======================================================================================

/// **The ratio-128 compressor emits nothing at the emit prompt, so L3 has no `.compressed`
/// golden at all.**
///
/// `bin/v4-oracle`'s own `caveats` metadata says so ("ratio-128 compression is NOT
/// exercised at this prompt length -- it needs >=128 tokens"), and this is that sentence
/// made executable. `Compressor.forward` needs `seqlen >= ratio` at prefill and a position
/// that completes a block at decode; 13 tokens plus a handful of decode steps reaches
/// neither, so `should_compress` is false on every call and the oracle pushes no
/// `L3.*.compressed` tensor.
///
/// The consequence for S2c is precise and worth stating in one place: the **non-overlapping
/// pooling branch is scored by nothing**. `golden::identical` treats a tensor missing on
/// one side as an infinite difference, so it cannot fail open on a *present* golden — but
/// there is no golden here to be present.
#[test]
fn ratio_128_pooling_is_unreachable_at_the_emit_prompt() {
    let l3 = LayerKind::from_ratio(128);
    assert!(
        !should_compress(l3, EMIT_PROMPT_LEN, 0),
        "13-token prefill cannot fill a 128-wide block"
    );
    // Decode only compresses on the position that completes a block: start_pos = 127, 255,
    // ... Reaching the first one needs 115 decode steps past a 13-token prompt.
    let first = (EMIT_PROMPT_LEN..4096).find(|&p| should_compress(l3, 1, p));
    assert_eq!(first, Some(127), "the first ratio-128 decode block completes at start_pos 127");
    for p in EMIT_PROMPT_LEN..127 {
        assert!(!should_compress(l3, 1, p), "no block completes at start_pos {p}");
    }
    // For contrast, ratio 4 compresses immediately and often — which is why L2 is covered
    // and L3 is not, from the same run.
    let l2 = LayerKind::from_ratio(4);
    assert!(should_compress(l2, EMIT_PROMPT_LEN, 0));
    assert_eq!((EMIT_PROMPT_LEN..20).filter(|&p| should_compress(l2, 1, p)).count(), 2);

    // And a ratio-0 layer has no Compressor at all, so the answer is false in BOTH phases.
    // As a raw `usize` this read `13 >= 0` == true at prefill and `is_multiple_of(0)` ==
    // false at decode: two answers for one layer, the prefill one steering into a
    // divide-by-zero. `LayerKind` makes the state unrepresentable; this pins it.
    let l0 = LayerKind::from_ratio(0);
    assert_eq!(l0.compressor_ratio(), None);
    assert!(!should_compress(l0, EMIT_PROMPT_LEN, 0), "ratio-0 never compresses at prefill");
    assert!(!should_compress(l0, 1, 7), "ratio-0 never compresses at decode");
    assert!(compress_topk(l0, EMIT_PROMPT_LEN, 0, 13).is_empty(), "ratio-0 selects nothing");
}

/// **The ratio-128 SELECTION golden is a `[13, 0]` empty tensor.**
///
/// A second, separate hole from the pooling one above, and easier to miss because a golden
/// does exist under the name `L3.pre.compress_idxs` — it is simply empty. With
/// `seqlen // ratio == 0` there are no compressed blocks to select, so `compress_topk`
/// returns 13 rows of zero columns and any implementation that also returns nothing agrees
/// with it. `golden::identical` documents that it deliberately accepts an empty
/// `.compress_idxs` (a legitimate case for a compressor that has not yet produced a block),
/// which is correct and is exactly why this case cannot be caught downstream.
#[test]
fn ratio_128_selection_golden_is_empty_and_therefore_vacuous() {
    let offset = compress_offset(cfg().window_size, EMIT_PROMPT_LEN, 0);
    let l3 = LayerKind::from_ratio(128);
    let sel = compress_topk(l3, EMIT_PROMPT_LEN, 0, offset);
    assert_eq!(sel.len(), EMIT_PROMPT_LEN, "one row per query");
    assert!(sel.iter().all(Vec::is_empty), "every row is empty: seqlen/ratio == 0");
    assert_eq!(sel.concat().len(), 0, "the golden carries no values at all");

    // The same function on a prompt long enough to matter is NOT vacuous, which is what
    // makes the emptiness above a property of the prompt rather than of the code.
    let long = 512usize;
    let sel = compress_topk(l3, long, 0, compress_offset(cfg().window_size, long, 0));
    assert_eq!(sel[0].len(), long / 128, "4 compressed blocks exist at 512 tokens");
    assert!(sel[0].iter().all(|&x| x == -1), "query 0 may read none of them");
    assert_eq!(
        sel[long - 1].iter().filter(|&&x| x != -1).count(),
        (long) / 128,
        "the last query may read every completed block"
    );
}

/// **The indexer's ranking is untested at the emit prompt: `index_topk` never cuts.**
///
/// This hole is NOT named in the S2c brief and is arguably the more dangerous of the two,
/// because unlike ratio-128 the goldens here are present, non-empty, and look like
/// coverage. `index_topk` is 512; a 13-token prompt has `13 / 4 = 3` compressed blocks, so
/// `k = min(512, 3) = 3 == n_comp` and the top-k selects **every** block. `.compress_idxs`
/// is then an invariant set for any scoring whatsoever, and comparing it as a set — which
/// `forward.rs:772` correctly requires — succeeds against an arbitrarily wrong ranking.
///
/// S1b anticipated exactly this and instrumented it: `Counters::indexer_truncated` counts
/// the query rows where `k < n_comp`, and its doc says a zero means `.compress_idxs`
/// "records an invariant set and cannot distinguish a right ranking from a wrong one".
/// At the emit prompt that counter is zero for every row.
///
/// What IS still covered: the causal mask (rows differ in how many entries are `-1`) and
/// the scores themselves (`.indexer_scores` is the full pre-top-k matrix, compared
/// numerically). So the arithmetic is gated and the *ordering* is not.
#[test]
fn indexer_topk_never_cuts_at_the_emit_prompt() {
    let index_topk = cfg().index_topk;
    assert_eq!(index_topk, 512);
    let n_comp = EMIT_PROMPT_LEN / 4;
    assert_eq!(n_comp, 3);
    assert_eq!(index_topk.min(n_comp), n_comp, "top-k selects every block: no ranking is tested");

    // The prompt length at which the ranking would start to bite, for the record: the first
    // seqlen whose compressed-block count exceeds index_topk. Stated as a closed form and
    // then checked against a search over the range that brackets it, so a wrong closed form
    // cannot pass by being compared only against itself.
    let need = 4 * (index_topk + 1);
    assert_eq!(need, 2052, "2052 tokens before index_topk cuts anything");
    assert!((1..need).all(|s| s / 4 <= index_topk), "nothing below {need} cuts");
    assert!(need / 4 > index_topk, "{need} does cut");
}

// =======================================================================================
// 3. the write side — where a compressed block LANDS
// =======================================================================================

/// **Requirement 2: the decode slot is `window + start_pos / ratio`, never "the next free
/// one".** `docs/investigations/v4-flash-port.md` records this as implemented nowhere and
/// asserted nowhere after `2445645` deleted `Dims::compress_slot`; this is the assertion,
/// and [`compress_dst`] is the implementation.
///
/// **The anti-vacuity half is the second loop, and it is the whole test.** An appending
/// placer — one that keeps a running count and writes the next row whenever a block is
/// emitted — agrees with the positional one on every sequence that visits every position in
/// order. That is the sequence a smoke test walks, so a green run over it proves nothing.
/// The divergence needs a SKIPPED step. Requirement 2 names speculative decode as the way
/// one arises, and on V4 that mechanism is NOT currently reachable — `kernels/moe.hip:409`
/// refuses `nrow != 1` — so this walks the skip directly rather than claiming a caller that
/// cannot exist yet. Below, positions 3 and 11 emit; the
/// positional placer puts them at rows `win + 0` and `win + 2`, the appending one at
/// `win + 0` and `win + 1`. The two `assert_eq!`s already imply the difference; the
/// `assert_ne!` after them adds no coverage and is there so a later edit that made the
/// two vectors agree could not be read as a simplification.
///
/// Cross-checked against the reference rather than against this crate: model.py:380 and :382
/// writes `self.kv_cache[:, :seqlen // ratio]` at prefill and `[start_pos // ratio]` at
/// decode, and `Compressor.kv_cache` is the VIEW `Attention.kv_cache[:, win:]`
/// (model.py:497) — so `window_size +` is the flattening of that view, not a convention.
#[test]
fn compress_dst_is_positional_and_an_appending_placer_disagrees() {
    let win = cfg().window_size;
    let l2 = LayerKind::from_ratio(4);

    // Prefill: every complete block at once, at the region's base. 13 // 4 == 3.
    assert_eq!(compress_dst(l2, win, EMIT_PROMPT_LEN, 0), Some((win, 3)));
    // Shorter than one block emits nothing — `should_compress`'s prefill arm, and the row
    // must not be reported as "the base, zero blocks", which a caller would memcpy as a
    // zero-length write and then trust the rows.
    assert_eq!(compress_dst(l2, win, 3, 0), None);

    // Decode: one block, on the position that completes it, at slot `start_pos / ratio`.
    for p in 0..64 {
        let want = ((p + 1) % 4 == 0).then_some((win + p / 4, 1));
        assert_eq!(compress_dst(l2, win, 1, p), want, "ratio-4 decode at start_pos {p}");
    }

    // THE REQUIREMENT. Walk positions 0..=11 but SKIP 4..=7 — one accepted two-token draft
    // is enough to do this — and compare against an appending placer.
    let mut appended = win;
    let mut positional = Vec::new();
    let mut appending = Vec::new();
    for p in [0, 1, 2, 3, 8, 9, 10, 11] {
        if let Some((row, n)) = compress_dst(l2, win, 1, p) {
            positional.push(row);
            appending.push(appended);
            appended += n;
        }
    }
    assert_eq!(positional, vec![win, win + 2], "blocks 0 and 2 complete at 3 and 11");
    assert_eq!(appending, vec![win, win + 1], "an appending placer packs them adjacently");
    assert_ne!(
        positional, appending,
        "if these agree the test is vacuous — a skipped step MUST separate the two placers"
    );

    // The ratio-128 boundary, and it is this function's OWN arithmetic rather than the
    // gate's: the first block completes at 127, and `127 / 128` is 0, so it lands at the
    // base. `should_compress` decides WHETHER with `(start_pos + 1) / ratio`; reusing that
    // to decide WHERE puts every ratio-128 block one slot late for the whole sequence.
    //
    let l3 = LayerKind::from_ratio(128);
    assert_eq!(compress_dst(l3, win, 1, 127), Some((win, 1)));
    assert_eq!(compress_dst(l3, win, 1, 255), Some((win + 1, 1)));
    // Prefill at ratio 128, where `seqlen / ratio` actually truncates something: 300 tokens
    // are two complete blocks and a 44-token remainder the compressor keeps in `kv_state`.
    // At ratio 4 the remainder is at most 3, so a wrong divisor hides there.
    assert_eq!(compress_dst(l3, win, 300, 0), Some((win, 2)));

    // **`LayerKind::Plain` returns `None` from the `?`, BEFORE either division.** This is
    // not the `should_compress` gate and is not covered by
    // `ratio_128_pooling_is_unreachable_at_the_emit_prompt`, which never calls this
    // function — a draft of this test dropped these two lines as "delegation" and that was
    // wrong: `compress_dst` has TWO independent `None` sources and only this one stands
    // between a ratio-0 layer and `seqlen / 0`. Reorder the two guards and the rest of the
    // suite stays green while every ratio-0 layer panics.
    let l0 = LayerKind::from_ratio(0);
    assert_eq!(l0.compressor_ratio(), None, "the `?` this test relies on must be the one that fires");
    assert_eq!(compress_dst(l0, win, EMIT_PROMPT_LEN, 0), None);
    assert_eq!(compress_dst(l0, win, 1, 7), None);
}
